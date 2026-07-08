"""
Parallel stitching pipeline: threaded I/O loading, process-pool SIFT
extraction, and thread-pool tiled blending.

Strategy:
Three independent forms of parallelism, applied where the data
dependencies allow it:
    - Loading   : I/O-bound, images are independent  -> ThreadPoolExecutor
    - Extraction: CPU-bound, images are independent   -> ProcessPoolExecutor
    - Blending  : one warp/blend split into horizontal
                  tiles, independently blendable       -> ThreadPoolExecutor

The match -> homography -> warp -> reextract chain itself stays
sequential (each stitch depends on the previous panorama).
"""

import cProfile
import sys
import pstats
import cv2
import numpy as np
import time
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

NUM_CORES = 8


def load_single_image(path):
    """
    Loads a single image from disk and downscales it by 50%.
    
    Designed to be executed within a ThreadPoolExecutor.

    Args:
        path (Path or str): Path to the image file.

    Returns:
        numpy.ndarray or None: The loaded and resized image, or None on failure.
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    
    # Downscale to reduce memory footprint and processing time
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    return img


def extract_single_image_features(img):
    """
    Extracts SIFT keypoints and descriptors for a single image.
    
    Designed to run inside a ProcessPoolExecutor. Since feature extraction 
    is a heavily CPU-bound task, it uses separate OS processes to bypass 
    GIL and achieve parallelism.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        tuple: (List of serialized keypoints, Descriptors array).
    """
    
    cv2.setNumThreads(1)  # avoid oversubscription across worker processes

    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # Multiprocessing relies on pickle to pass data back to the main process.
    # cv2.KeyPoint objects are C++ bindings and cannot be natively pickled by Python.
    # So, it must unpack and serialize their essential attributes into primitive tuples.
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]

    return kp_serialized, des


def blend_tile_worker(args):
    """
    Applies alpha-blending to a specific horizontal tile of the canvas.
    
    By splitting the large canvas into smaller chunks, it can distribute 
    the heavy matrix arithmetic across multiple threads.

    Args:
        args (tuple): A tuple containing (canvas_chunk, warped_chunk) matrices.

    Returns:
        numpy.ndarray: The blended image chunk ready to be stitched back together.
    """
    canvas_chunk, warped_chunk = args

    # Generate logical masks (1 where a pixel exists, 0 where it's black/empty)
    mask1 = (canvas_chunk > 0).any(axis=2).astype(np.float32)
    mask2 = (warped_chunk > 0).any(axis=2).astype(np.float32)

    # Calculate exclusivity and overlap regions
    overlap = (mask1 * mask2)[..., np.newaxis]
    only1   = (mask1 * (1 - mask2))[..., np.newaxis]
    only2   = ((1 - mask1) * mask2)[..., np.newaxis]

    # Perform alpha blending (average the overlap, keep exclusive regions intact)
    res_chunk = (
        canvas_chunk.astype(np.float32) * (only1 + 0.5 * overlap) +
        warped_chunk.astype(np.float32) * (only2 + 0.5 * overlap)
    ).clip(0, 255).astype(np.uint8)

    return res_chunk


def load_images_parallel(input_dir, start_idx=-4, end_idx=None):
    """
    Orchestrates the parallel loading and downscaling of a batch of images.
    
    Uses a ThreadPoolExecutor because reading files is an I/O-bound operation. 
    The Python GIL is naturally released during I/O waits, making threads 
    suited for this task without the overhead of spinning up processes.

    Args:
        input_dir (str): Path to the directory containing source images.
        start_idx (int): Starting index for the image slice.
        end_idx (int): Ending index for the image slice.

    Returns:
        list: A list of successfully loaded OpenCV images.
    """

    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    image_paths = image_paths[start_idx:end_idx]

    print(f"   [Parallel] Loading {len(image_paths)} images via ThreadPool...", file=sys.stderr)

    # Spawn one thread per logical CPU core
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        # executor.map preserves the original order of the paths
        results = list(executor.map(load_single_image, image_paths))

    # Filter out any images that failed to load (None values)
    images = [img for img in results if img is not None]
    return images


def extract_features_parallel(images, process_executor=None):
    """
    Extracts SIFT features for a batch of images using multiprocessing.

    This function distributes the CPU-heavy SIFT extraction across multiple 
    logical cores. It also handles the reconstruction of OpenCV C++ objects 
    that were broken down into primitive Python types for IPC.

    Args:
        images (list): List of input images (numpy arrays).
        process_executor (ProcessPoolExecutor, optional): An existing active 
            pool. If provided, reuses the processes to avoid spin-up overhead. 
            Otherwise, a temporary pool is created and shut down internally.

    Returns:
        tuple: (list of cv2.KeyPoint lists, list of descriptor arrays, total time)
    """
    start_time = time.perf_counter()

    print(f"   [Parallel] SIFT Feature Extraction via ProcessPool ({NUM_CORES} cores)...", file=sys.stderr)

    # Use the injected executor to save overhead, or spin up a new one if None
    if process_executor is None:
        with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            results = list(executor.map(extract_single_image_features, images))
    else:
        results = list(process_executor.map(extract_single_image_features, images))

    keypoints_list = []
    descriptors_list = []

    # Reconstruct the C++ cv2.KeyPoint objects in the main process
    for i, (kp_serialized, des) in enumerate(results):

        # It mapped the primitive tuples back to the cv2.KeyPoint constructor.
        # This completes the serialization/deserialization cycle required by Pickle.
        kp = [
            cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
            for pt, size, angle, response, octave, class_id in kp_serialized
        ]
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"      - Image {i+1}: Found {len(kp)} keypoints", file=sys.stderr)

    extraction_time = time.perf_counter() - start_time
    return keypoints_list, descriptors_list, extraction_time


def match_features(des1, des2):
    """
    Finds robust feature matches between two sets of descriptors.

    Uses the Fast Library for Approximate Nearest Neighbors (FLANN) optimized 
    for KD-Tree search. Filters out ambiguous matches using Lowe's ratio test.

    Args:
        des1 (numpy.ndarray): Descriptors from the query image (panorama).
        des2 (numpy.ndarray): Descriptors from the train image (new tile).

    Returns:
        list: A list of robust cv2.DMatch objects.
    """

    # algorithm=1 is FLANN_INDEX_KDTREE, optimal for SIFT float32 descriptors.
    # trees=5 and checks=50 balance search precision with execution speed.
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # knnMatch returns the top k=2 closest descriptors for every point
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []

    # Lowe's ratio test: A match is considered robust only if the closest neighbor 
    # is significantly closer (distance < 70%) than the second closest neighbor.
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


def estimate_homography(kp1, kp2, matches):
    """
    Estimates the perspective transformation (Homography) between two images.

    Uses the RANSAC algorithm to ignore mismatched features (outliers) and 
    computes a 3x3 matrix that aligns the new image (kp2) onto the panorama (kp1).

    Args:
        kp1 (list): Keypoints of the base panorama.
        kp2 (list): Keypoints of the new image.
        matches (list): Validated matches connecting kp1 and kp2.

    Returns:
        numpy.ndarray or None: The 3x3 Homography matrix, or None if the 
                               geometric link is too weak.
    """

    # A minimum of 4 points is mathematically required to solve a homography
    if len(matches) < 4:
        print("   ERROR: Not enough matches to compute homography (need at least 4).", file=sys.stderr)
        return None

    # Extract the (x, y) coordinates from the DMatch objects and format them 
    # exactly as OpenCV expects: a 3D float32 array of shape (N, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate Homography with RANSAC (5.0 pixel reprojection error threshold)
    # The mask array tells which points actually fit the final geometric model
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("   ERROR: RANSAC failed to find a valid homography.", file=sys.stderr)
        return None

    # Count how many matches actually contributed to the transformation
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"     Homography inliers: {inliers}/{len(matches)}", file=sys.stderr)

    # Reject weakly-supported homographies to avoid degenerate merges.
    # Even if RANSAC finds a mathematical solution, fewer than 15 points 
    # usually results in extreme, degenerate warping ("canvas explosion")
    min_inliers_threshold = 15
    if inliers < min_inliers_threshold:
        print(f"   WARNING: Too few RANSAC inliers ({inliers}/{len(matches)}). Rejecting Homography to prevent canvas explosion.", file=sys.stderr)
        return None

    return H


def warp_and_blend_tiling(img1, img2, thread_executor, H, num_workers=4):
    """
    Warps img2 onto img1's coordinate plane and blends them using Data Parallelism.

    Instead of performing alpha-blending on the entire massive canvas sequentially,
    this function splits the overlapping matrices into horizontal strips. 
    Each tile is then blended concurrently on a ThreadPoolExecutor.

    Args:
        img1 (numpy.ndarray): The base panorama canvas.
        img2 (numpy.ndarray): The new image to stitch.
        thread_executor (ThreadPoolExecutor): Active thread pool for parallel blending.
        H (numpy.ndarray): 3x3 Homography matrix mapping img2 to img1.
        num_workers (int): Number of horizontal slices to divide the canvas into.

    Returns:
        numpy.ndarray: The combined, blended panorama image.

    Raises:
        ValueError: If the computed canvas area is empty or exceeds 4x the 
            combined input area (indicates a mathematically degenerate Homography).
    """

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # First Phase: calculate the bounding box of the new panorama
    # Find the 4 corners of the new image and project them through the Homography
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    # Find the 4 corners of the base image (which stays at origin 0,0)
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Combine all corners to find the absolute min and max (x, y) coordinates
    all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Second Phase: canvas explosion guardrail
    # A bad Homography can calculate theoretical coordinates in the millions, 
    # causing instant RAM exhaustion (OOM).
    # It strictly calculate and verify the required dimensions before allocating memory.
    max_canvas_multiplier = 4.0
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    canvas_area = canvas_w * canvas_h
    input_area  = (h1 * w1) + (h2 * w2)

    if canvas_w <= 0 or canvas_h <= 0 or canvas_area > max_canvas_multiplier * input_area:
        raise ValueError(
            f"Canvas explosion detected: computed canvas {canvas_w}x{canvas_h} "
            f"({canvas_area} px) vs combined input area {input_area} px "
            f"(limit: {max_canvas_multiplier}x). Homography is likely degenerate."
        )


    # Third Phase: global translation and warping
    # Warping might result in negative coordinates and matrices cannot have negative indices. 
    # It creates a translation matrix to shift everything into the positive quadrant (origin at 0,0).
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (x_max - x_min, y_max - y_min)

    # Warp the new image. Note: OpenCV's warpPerspective is implemented in C++ 
    # and natively releases the Python GIL, running multi-threaded internally.
    warped_img2 = cv2.warpPerspective(img2, translation.dot(H), canvas_size)

    # Create a blank black canvas and place the base image onto it at the translated position
    canvas_img1 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    y_off, x_off = -y_min, -x_min
    canvas_img1[y_off:y_off + h1, x_off:x_off + w1] = img1

    # Fourth Phase: tiling
    # Split the massive canvas arrays into 'num_workers' horizontal chunks.    
    total_rows = canvas_size[1]
    chunk_size = int(np.ceil(total_rows / num_workers))

    tiles_args = []
    for i in range(num_workers):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, total_rows)

        # Ensure it doesn't create empty slices
        if start_row < end_row:
            tile_canvas = canvas_img1[start_row:end_row, :]
            tile_warped = warped_img2[start_row:end_row, :]
            tiles_args.append((tile_canvas, tile_warped))

    # Fifth Phase: parallel blending and reassembly 
    # Dispatch the pairs of chunks to the blend_tile_worker function via threads.
    # The map function preserves the exact vertical order of the chunks.
    blended_chunks = list(thread_executor.map(blend_tile_worker, tiles_args))

    # Vertically stack the successfully blended horizontal strips back together
    result = np.vstack(blended_chunks)
    return result


def stitch_images_parallel(input_dir, output_dir, start_idx=0, end_idx=4):
    """
    Runs the parallel stitching pipeline on a custom image range and saves the panorama.

    This function sets up a dual-pool execution environment (processes + threads)
    to maximize CPU utilization. It extracts features in parallel across files, 
    then sequentially merges them while using internal thread-based data 
    parallelism (tiling) for the heavy warping and blending matrix operations.

    Args:
        input_dir (str): Directory containing the source images.
        output_dir (str): Directory where the final panorama will be saved.
        start_idx (int): Starting index for the image range.
        end_idx (int): Ending index for the image range.
    """

    print(f"\nSTARTING PARALLEL PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    # Multi-threaded concurrent disk I/O loading
    images = load_images_parallel(input_dir, start_idx=start_idx, end_idx=end_idx)

    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    print("\nStarting Parallel SIFT Feature Extraction...", file=sys.stderr)

    # Nesting both executors using a single context manager block.
    # This keeps the resource management clean and ensures all workers are safely
    # reaped/terminated when exiting the block.
    with ProcessPoolExecutor(max_workers=NUM_CORES) as process_executor, \
         ThreadPoolExecutor(max_workers=NUM_CORES) as thread_executor:
        
        # Heavy CPU-bound step: distributed keypoint extraction
        kp_list, des_list, t_extract = extract_features_parallel(images, process_executor=process_executor)

        print("\nStarting Iterative Stitching with Internal Parallelism...", file=sys.stderr)
        stitch_start = time.perf_counter()

        base_image = images[0]
        base_kp    = kp_list[0]
        base_des   = des_list[0]
        sift = cv2.SIFT_create(nfeatures=8000)

        t_match_sub = 0.0
        t_homo_sub = 0.0
        t_warp_sub = 0.0
        t_reext_sub = 0.0

        for i in range(1, len(images)):
            print(f"\n   - Stitching image {i+1} onto current panorama...", file=sys.stderr)

            # First Phase: Fast FLANN Matching
            t_start_match = time.perf_counter()
            matches = match_features(base_des, des_list[i])
            t_match_sub += time.perf_counter() - t_start_match
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {i+1}, skipping.", file=sys.stderr)
                continue
            
            # Second Phase: RANSACC Homography
            t_start_homo = time.perf_counter()
            H = estimate_homography(base_kp, kp_list[i], matches)
            t_homo_sub += time.perf_counter() - t_start_homo
            if H is None:
                print(f"     WARNING: Homography failed for image {i+1}, skipping.", file=sys.stderr)
                continue

            # Third Phase: Warp and Blend (Tiling)
            t_start_warp = time.perf_counter()
            base_image = warp_and_blend_tiling(base_image, images[i], thread_executor, H, num_workers=NUM_CORES)
            t_warp_sub += time.perf_counter() - t_start_warp

            # Fourth Phase: Single image Re-extraction
            t_start_reext = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            t_reext_sub += time.perf_counter() - t_start_reext
            print(f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

    t_stitch   = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    # Save the final stitched panorama to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Final Benchmark Report
    print("\n" + "=" * 50, file=sys.stderr)
    print(f"PARALLEL REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:   {t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total Time:{t_stitch:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:   {t_match_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:    {t_homo_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend (Tile):{t_warp_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract: {t_reext_sub:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    """
    Executes the parallel stitching pipeline over a massive dataset using sliding windows.

    Instead of trying to stitch 50 images together (which causes extreme 
    perspective distortion and memory exhaustion), this function breaks the dataset 
    into smaller, manageable "windows" (e.g., 4 images per window) and generates 
    a separate panorama for each window.

    Args:
        input_dir (str): Directory containing the large image sequence.
        output_dir (str): Destination directory for the independent panoramas.
        window_size (int): Max number of adjacent images to merge into a single panorama.
    """
    print(f"STARTING PARALLEL SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    # Metrics accumulators
    total_t_extract = 0.0
    total_t_match   = 0.0
    total_t_homo    = 0.0
    total_t_warp    = 0.0
    total_t_reext   = 0.0

    total_start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=8000)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # The processes and threads remain "warm" in the background, ready to process 
    # each sliding window chunk sequentially without re-allocation overhead.
    with ProcessPoolExecutor(max_workers=NUM_CORES) as process_executor, \
         ThreadPoolExecutor(max_workers=NUM_CORES) as thread_executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx-1} ---", file=sys.stderr)

            # Concurrent multi-threaded I/O chunk loading
            current_images = load_images_parallel(input_dir, start_idx, end_idx)

            if len(current_images) < 2:
                print("   WARNING: Insufficient images in this window to stitch. Skipping.", file=sys.stderr)
                continue

            print("Starting SIFT Feature Extraction for current window...", file=sys.stderr)

            # Injecting the warm process_executor to bypass spin-up cost
            kp_list, des_list, t_extract = extract_features_parallel(current_images, process_executor=process_executor)
            total_t_extract += t_extract

            # Establish the base canvas coordinate system for the current window
            base_image = current_images[0]
            base_kp    = kp_list[0]
            base_des   = des_list[0]

            # Iteratively stitch the remainder of the window sequence
            for i in range(1, len(current_images)):
                global_img_idx = start_idx + i
                print(f"\n   - Stitching image {global_img_idx}/{total_images-1} onto window panorama...", file=sys.stderr)

                # Match features between the current panorama and the next image in the window
                t_start_match = time.perf_counter()
                matches = match_features(base_des, des_list[i])
                total_t_match += time.perf_counter() - t_start_match
                print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_img_idx}, skipping.", file=sys.stderr)
                    continue
                
                # Homography estimation using RANSAC to align the new image onto the current panorama
                t_start_homo = time.perf_counter()
                H = estimate_homography(base_kp, kp_list[i], matches)
                total_t_homo += time.perf_counter() - t_start_homo
                if H is None:
                    print(f"     WARNING: Homography estimation failed for image {global_img_idx}, skipping.", file=sys.stderr)
                    continue
                
                # Warp and Parallel Blend
                t_start_warp = time.perf_counter()
                base_image = warp_and_blend_tiling(base_image, current_images[i], thread_executor, H, num_workers=NUM_CORES)
                total_t_warp += time.perf_counter() - t_start_warp

                # Re-extract features from the updated panorama
                t_start_reext = time.perf_counter()
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                total_t_reext += time.perf_counter() - t_start_reext

                print(f"     Updated window panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                    f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

            # Save the stitched panorama for the current window
            final_file_path = output_path / f"panorama_window_{start_idx}_to_{end_idx-1}.jpg"
            cv2.imwrite(str(final_file_path), base_image)
            print(f"\nWindow Panorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Compute aggregate execution times
    total_time = time.perf_counter() - total_start
    total_t_stitch = total_t_match + total_t_homo + total_t_warp + total_t_reext

    # Global Performance Summary
    print("\n" + "=" * 50, file=sys.stderr)
    print("PARALLEL PIPELINE PERFORMANCE REPORT", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Total Images Processed:    {total_images}", file=sys.stderr)
    print(f"Window/Batch Size:         {window_size}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:      {total_t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total Time:   {total_t_stitch:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:      {total_t_match:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:       {total_t_homo:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend (Tiling): {total_t_warp:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract:    {total_t_reext:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:      {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def main():
    input_dir  = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input not found.", file=sys.stderr)
        return

    # Let OpenCV's own OpenMP threads run free for internal ops.
    cv2.setNumThreads(NUM_CORES)

    profiler = cProfile.Profile()
    profiler.enable()

    sliding_window_pipeline(input_dir, output_dir, window_size=4)

    profiler.disable()

    output_file = Path("profiling_results/parallel_profiling")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("tottime")
        stats.print_stats()

    print(f"\nProfiling results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()