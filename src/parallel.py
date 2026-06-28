import cv2
import numpy as np
import time
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# SUPPORT FUNCTIONS FOR PROCESS WORKERS

def load_single_image(path):
    """Worker for parallel image loading"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    # Downscale by 50% to optimize memory usage during warping operations
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    return img

def extract_single_image_features(img):
    """Worker for SIFT feature extraction"""

    cv2.setNumThreads(1)

    # Initialize SIFT inside the worker to avoid state concurrency issues
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    
    # cv2.KeyPoint objects are not natively picklable (serializable), 
    # so we convert them into tuples of primitive data types
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
    return kp_serialized, des

def blend_tile_worker(args):
    """Parallel blending of a single image strip/tile"""
    canvas_chunk, warped_chunk = args
    
    # Local calculation of blend masks on the assigned tile
    mask1 = (canvas_chunk > 0).any(axis=2).astype(np.float32)
    mask2 = (warped_chunk > 0).any(axis=2).astype(np.float32)
    overlap = (mask1 * mask2)[..., np.newaxis]
    only1   = (mask1 * (1 - mask2))[..., np.newaxis]
    only2   = ((1 - mask1) * mask2)[..., np.newaxis]

    # Mathematical blending of the strip
    res_chunk = (
        canvas_chunk.astype(np.float32) * (only1 + 0.5 * overlap) +
        warped_chunk.astype(np.float32) * (only2 + 0.5 * overlap)
    ).clip(0, 255).astype(np.uint8)
    
    return res_chunk


# MAIN PIPELINE 

def load_images_parallel(input_dir, start_idx=-4, end_idx=None):
    """
    Phase 1: Image Loading and Preprocessing with Custom Range.
    (I/O-Bound Parallelization via ThreadPoolExecutor)
    """
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    image_paths = image_paths[start_idx:end_idx]

    print(f"   [Parallel] Loading {len(image_paths)} images via ThreadPool...")
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(load_single_image, image_paths))
    
    images = [img for img in results if img is not None]
    return images


def extract_features_parallel(images, process_executor=None):
    """
    Phase 2: SIFT Feature Extraction.
    (CPU-Bound Parallelization via ProcessPoolExecutor)
    """
    start_time = time.perf_counter()
    
    print(f"   [Parallel] SIFT Feature Extraction via ProcessPool ({os.cpu_count()} cores)...")
    if process_executor is None:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(extract_single_image_features, images))
    else:
        results = list(process_executor.map(extract_single_image_features, images))
    
    keypoints_list = []
    descriptors_list = []
    
    # Reconstruct cv2.KeyPoint objects from serialized data received from workers
    for i, (kp_serialized, des) in enumerate(results):
        kp = [
            cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
            for pt, size, angle, response, octave, class_id in kp_serialized
        ]
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"      - Image {i+1}: Found {len(kp)} keypoints")

    extraction_time = time.perf_counter() - start_time
    return keypoints_list, descriptors_list, extraction_time


def match_features(des1, des2):
    """Phase 3: Feature Matching for a pair of images using FLANN + Lowe's ratio test"""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


def estimate_homography(kp1, kp2, matches):
    if len(matches) < 4:
        print("   ERROR: Not enough matches to compute homography (need at least 4).")
        return None

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("   ERROR: RANSAC failed to find a valid homography.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"     Homography inliers: {inliers}/{len(matches)}")
    
    # Set a minimum inliers threshold to avoid canvas explosion
    min_inliers_threshold = 15
    if inliers < min_inliers_threshold:
        print(f"   WARNING: Too few RANSAC inliers ({inliers}/{len(matches)}). Rejecting Homography to prevent canvas explosion.")
        return None

    return H


def warp_and_blend_tiling(img1, img2, H, num_workers=4):
    """
    Phase 5: Image Warping and Alpha Blending.
    (Domain Decomposition / Horizontal Tiling via ThreadPoolExecutor)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (x_max - x_min, y_max - y_min)

    # OpenCV's native geometric warping internally releases the GIL and runs multi-threaded
    warped_img2 = cv2.warpPerspective(img2, translation.dot(H), canvas_size)

    canvas_img1 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    y_off, x_off = -y_min, -x_min
    canvas_img1[y_off:y_off + h1, x_off:x_off + w1] = img1

    # Start Tiling
    total_rows = canvas_size[1]
    chunk_size = int(np.ceil(total_rows / num_workers))
    
    tiles_args = []
    for i in range(num_workers):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, total_rows)
        if start_row < end_row:
            # Create independent matrix views/slices without copying memory yet
            tile_canvas = canvas_img1[start_row:end_row, :]
            tile_warped = warped_img2[start_row:end_row, :]
            tiles_args.append((tile_canvas, tile_warped))
            
    # Distribute the logical Alpha Blending computations across CPU threads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        blended_chunks = list(executor.map(blend_tile_worker, tiles_args))
        
    # Final recomposition by vertically stacking the processed strips
    result = np.vstack(blended_chunks)
    return result


def stitch_images_parallel(input_dir, output_dir, start_idx=0, end_idx=4):
    """
    Executes the parallel stitching pipeline ONLY on a custom range of images.
    Leverages ThreadPool for loading, ProcessPool for SIFT, and Domain Decomposition (Tiling) for Blending.
    """
    print(f"\nSTARTING PARALLEL PIPELINE (Range index {start_idx}:{end_idx})")
    total_start = time.perf_counter()

    # Phase 1: Parallel I/O-Bound image loading via ThreadPoolExecutor
    images = load_images_parallel(input_dir, start_idx=start_idx, end_idx=end_idx)
    
    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    # Phase 2: Parallel CPU-Bound SIFT feature extraction via ProcessPoolExecutor
    print("\nStarting Parallel SIFT Feature Extraction...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:
        kp_list, des_list, t_extract = extract_features_parallel(images, process_executor=process_executor)

    print("\nStarting Iterative Stitching with Internal Parallelism...")
    stitch_start = time.perf_counter()

    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]
    sift = cv2.SIFT_create()

    t_match_sub = 0.0
    t_homo_sub = 0.0
    t_warp_sub = 0.0
    t_reext_sub = 0.0

    for i in range(1, len(images)):
        print(f"\n   - Stitching image {i+1} onto current panorama...")

        # Phase 3: Feature Matching
        t_start_match = time.perf_counter()
        matches = match_features(base_des, des_list[i])
        t_match_sub += time.perf_counter() - t_start_match
        print(f"     Found {len(matches)} robust matches after Lowe's ratio test.")

        if len(matches) < 4:
            print(f"     WARNING: Too few matches for image {i+1}, skipping.")
            continue

        # Phase 4: Homography Estimation
        t_start_homo = time.perf_counter()
        H = estimate_homography(base_kp, kp_list[i], matches)
        t_homo_sub += time.perf_counter() - t_start_homo
        if H is None:
            print(f"     WARNING: Homography failed for image {i+1}, skipping.")
            continue

        # Phase 5: Warp and Blend using Horizontal Tiling
        t_start_warp = time.perf_counter()
        base_image = warp_and_blend_tiling(base_image, images[i], H, num_workers=os.cpu_count())
        t_warp_sub += time.perf_counter() - t_start_warp

        # Sequential Feature Re-extraction on the newly updated canvas
        t_start_reext = time.perf_counter()
        gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray_base, None)
        t_reext_sub += time.perf_counter() - t_start_reext
        print(f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
              f"{len(base_kp)} keypoints re-extracted.")

    t_stitch   = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}")

    print("\n" + "=" * 50)
    print(f"PARALLEL REPORT (RANGE {start_idx}:{end_idx})")
    print("=" * 50)
    print(f"SIFT Extraction Time:   {t_extract:.3f} seconds")
    print(f"Match & Warp Total Time:{t_stitch:.3f} seconds")
    print(f"  - Feature Matching:   {t_match_sub:.3f} seconds")
    print(f"  - Homography Est.:    {t_homo_sub:.3f} seconds")
    print(f"  - Warp & Blend (Tile):{t_warp_sub:.3f} seconds")
    print(f"  - Feature Re-extract: {t_reext_sub:.3f} seconds")
    print(f"Total Execution Time:   {total_time:.3f} seconds")
    print("=" * 50)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    print(f"STARTING PARALLEL SLIDING WINDOW PIPELINE (Window Size: {window_size})")
    
    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)
    
    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    total_t_extract = 0.0
    total_t_match   = 0.0
    total_t_homo    = 0.0
    total_t_warp    = 0.0
    total_t_reext   = 0.0
    
    total_start = time.perf_counter()
    sift = cv2.SIFT_create()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize a single ProcessPoolExecutor to avoid spawning overhead in the loop

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:    
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx-1} ---")
            
            # Internal window parallelism for loading images
            current_images = load_images_parallel(input_dir, start_idx, end_idx)
            
            if len(current_images) < 2:
                print("   WARNING: Insufficient images in this window to stitch. Skipping.")
                continue

            print("Starting SIFT Feature Extraction for current window...")
            # Internal window parallelism for SIFT extraction
            kp_list, des_list, t_extract = extract_features_parallel(current_images, process_executor=process_executor)
            total_t_extract += t_extract

            # Initialize base canvas specifically for THIS window
            base_image = current_images[0]
            base_kp    = kp_list[0]
            base_des   = des_list[0]

            for i in range(1, len(current_images)):
                global_img_idx = start_idx + i
                print(f"\n   - Stitching image {global_img_idx}/{total_images-1} onto window panorama...")

                t_start_match = time.perf_counter()
                matches = match_features(base_des, des_list[i])
                total_t_match += time.perf_counter() - t_start_match
                print(f"     Found {len(matches)} robust matches after Lowe's ratio test.")

                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_img_idx}, skipping.")
                    continue

                t_start_homo = time.perf_counter()
                H = estimate_homography(base_kp, kp_list[i], matches)
                total_t_homo += time.perf_counter() - t_start_homo
                if H is None:
                    print(f"     WARNING: Homography estimation failed for image {global_img_idx}, skipping.")
                    continue

                # Internal parallelism via Tiling for heavy blend operations
                t_start_warp = time.perf_counter()
                base_image = warp_and_blend_tiling(base_image, current_images[i], H, num_workers=os.cpu_count())
                total_t_warp += time.perf_counter() - t_start_warp

                t_start_reext = time.perf_counter()
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                total_t_reext += time.perf_counter() - t_start_reext
                
                print(f"     Updated window panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                    f"{len(base_kp)} keypoints re-extracted.")

            # Save the result for the current window
            final_file_path = output_path / f"panorama_window_{start_idx}_to_{end_idx-1}.jpg"
            cv2.imwrite(str(final_file_path), base_image)
            print(f"\nWindow Panorama saved successfully to: {final_file_path}")

    total_time = time.perf_counter() - total_start
    total_t_stitch = total_t_match + total_t_homo + total_t_warp + total_t_reext

    print("\n" + "=" * 50)
    print("PARALLEL PIPELINE PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Total Images Processed:    {total_images}")
    print(f"Window/Batch Size:         {window_size}")
    print("-" * 50)
    print(f"SIFT Extraction Time:      {total_t_extract:.3f} seconds")
    print(f"Match & Warp Total Time:   {total_t_stitch:.3f} seconds")
    print(f"  - Feature Matching:      {total_t_match:.3f} seconds")
    print(f"  - Homography Est.:       {total_t_homo:.3f} seconds")
    print(f"  - Warp & Blend (Tiling): {total_t_warp:.3f} seconds")
    print(f"  - Feature Re-extract:    {total_t_reext:.3f} seconds")
    print(f"Total Execution Time:      {total_time:.3f} seconds")
    print("=" * 50)


def main():
    input_dir  = "data/input"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input not found.")
        return
    
    # Configure OpenCV to use all available native OpenMP threads in its sub-functions
    cv2.setNumThreads(os.cpu_count())

    sliding_window_pipeline(input_dir, output_dir, window_size=4)

    


if __name__ == "__main__":
    main()