"""
Single-threaded, single-process baseline stitching pipeline.

This is the control condition for the whole project: no ProcessPool, no
ThreadPool, one image at a time. Every other pipeline in this project is
compared against it to measure potential speedups and parallelization overhead.
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path


def load_images(input_dir, start_idx=-4, end_idx=None):
    """
    Loads and downscales images from a specified directory.
    
    Images are resized to 50% of their original dimensions to reduce memory 
    footprint and speed up SIFT extraction, while maintaining enough detail 
    for accurate feature matching.

    Args:
        input_dir (str): Path to the directory containing source images.
        start_idx (int): Starting index for the image slice.
        end_idx (int): Ending index for the image slice.

    Returns:
        list: A list of loaded and downscaled OpenCV images.
    """
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    image_paths = image_paths[start_idx:end_idx]

    images = []
    print(f"Loading {len(image_paths)} images (range index {start_idx}:{end_idx})...", file=sys.stderr)
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"   WARNING: Could not load {path}, skipping.", file=sys.stderr)
            continue
        # Downscale by 50% for performance
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        images.append(img)
    return images


def extract_features(images):
    """
    Extracts SIFT keypoints and descriptors for a batch of images sequentially.

    Args:
        images (list): List of input images.

    Returns:
        tuple: (list of keypoints, list of descriptors, total extraction time)
    """
    sift = cv2.SIFT_create(nfeatures=8000)
    keypoints_list = []
    descriptors_list = []

    start_time = time.perf_counter()
    for i, img in enumerate(images):
        # SIFT requires grayscale images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"   - Image {i+1}: Found {len(kp)} keypoints", file=sys.stderr)

    extraction_time = time.perf_counter() - start_time
    return keypoints_list, descriptors_list, extraction_time


def match_features(des1, des2):
    """
    Matches features between two descriptors using FLANN and Lowe's ratio test.

    Args:
        des1 (numpy.ndarray): Descriptors from the query image (base panorama).
        des2 (numpy.ndarray): Descriptors from the train image (new image).

    Returns:
        list: A list of robust DMatch objects.
    """

    # FLANN configuration for fast approximate nearest neighbor search
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the 2 nearest neighbors for each descriptor
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    # Lowe's ratio test: discard matches where the primary distance is 
    # not significantly better than the secondary distance
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


def estimate_homography(kp1, kp2, matches):
    """
    Estimates the 3x3 perspective transformation matrix (Homography) using RANSAC.

    Args:
        kp1 (tuple): Keypoints from the base panorama.
        kp2 (tuple): Keypoints from the new image to stitch.
        matches (list): Good matches connecting kp1 and kp2.

    Returns:
        numpy.ndarray or None: The 3x3 homography matrix, or None if rejected.
    """

    if len(matches) < 4:
        print("   ERROR: Not enough matches to compute homography (need at least 4).", file=sys.stderr)
        return None

    # Extract coordinate pairs from the match objects
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC to filter out geometric outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("   ERROR: RANSAC failed to find a valid homography.", file=sys.stderr)
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"     Homography inliers: {inliers}/{len(matches)}", file=sys.stderr)

    # Reject weakly-supported homographies to avoid degenerate merges
    min_inliers_threshold = 15
    if inliers < min_inliers_threshold:
        print(f"   WARNING: Too few RANSAC inliers ({inliers}/{len(matches)}). Rejecting Homography to prevent canvas explosion.", file=sys.stderr)
        return None

    return H


def warp_and_blend(img1, img2, H):
    """
    Warps img2 onto img1's coordinate plane and applies alpha-blending.

    Args:
        img1 (numpy.ndarray): The base panorama image.
        img2 (numpy.ndarray): The new image to be warped.
        H (numpy.ndarray): 3x3 Homography matrix aligning img2 to img1.

    Raises:
        ValueError: If the computed canvas area is abnormally large, indicating 
                    a degenerate homography matrix.
                    
    Returns:
        numpy.ndarray: The blended composite image.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the transformed bounding box of img2
    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    # Combine with img1's bounding box to find the global canvas dimensions
    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_img1, warped_corners), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Canvas-explosion guard: check size before allocating anything
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

    # Create a translation matrix to shift negative coordinates into the visible canvas
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (x_max - x_min, y_max - y_min)

    # Warp the new image into the calculated canvas
    warped_img2 = cv2.warpPerspective(img2, translation.dot(H), canvas_size)

    # Place the base panorama into the canvas
    canvas_img1 = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    y_off, x_off = -y_min, -x_min
    canvas_img1[y_off:y_off + h1, x_off:x_off + w1] = img1

    # Generate logical masks for alpha-blending
    mask1 = (canvas_img1 > 0).any(axis=2).astype(np.float32)
    mask2 = (warped_img2 > 0).any(axis=2).astype(np.float32)
    overlap = (mask1 * mask2)[..., np.newaxis]
    only1   = (mask1 * (1 - mask2))[..., np.newaxis]
    only2   = ((1 - mask1) * mask2)[..., np.newaxis]

    # Blend: average pixels in the overlap region, keep original pixels elsewhere
    result = (
        canvas_img1.astype(np.float32) * (only1 + 0.5 * overlap) +
        warped_img2.astype(np.float32) * (only2 + 0.5 * overlap)
    ).clip(0, 255).astype(np.uint8)

    return result


def stitch_images(input_dir, output_dir, start_idx=0, end_idx=4):
    """
    Run the sequential stitching pipeline on a specific subset of images.
    
    This function processes a single continuous range of images, stitching them 
    iteratively one by one onto a growing base canvas. It tracks and prints 
    detailed execution times for each micro-phase (matching, homography, etc.).

    Args:
        input_dir (str): Directory containing the source images.
        output_dir (str): Directory where the final panorama will be saved.
        start_idx (int): Starting index for the image sequence.
        end_idx (int): Ending index for the image sequence.
    """
    print(f"\nSTARTING SEQUENTIAL PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    # Load and downscale images
    images = load_images(input_dir, start_idx=start_idx, end_idx=end_idx)

    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    # Global Feature Extraction
    print("\nStarting SIFT Feature Extraction...", file=sys.stderr)
    kp_list, des_list, t_extract = extract_features(images)

    # Iterative Stitching Process
    print("\nStarting Iterative Stitching...", file=sys.stderr)
    stitch_start = time.perf_counter()

    # Initialize the base canvas with the first image
    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]

    # Pre-instantiate SIFT object for fast re-extraction during the loop
    sift = cv2.SIFT_create(nfeatures=8000)

    # Timers for sub-phases
    t_match_sub = 0.0   
    t_homo_sub = 0.0
    t_warp_sub = 0.0
    t_reext_sub = 0.0

    # Sequentially stitch image i onto the growing base_image
    for i in range(1, len(images)):
        print(f"\n   - Stitching image {i+1} onto current panorama...", file=sys.stderr)

        # First Phase: Feature Matching
        t_start_match = time.perf_counter()
        matches = match_features(base_des, des_list[i])
        t_match_sub += time.perf_counter() - t_start_match
        print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

        if len(matches) < 4:
            print(f"     WARNING: Too few matches for image {i+1}, skipping.", file=sys.stderr)
            continue
        
        # Second Phase: Homography Estimation
        t_start_homo = time.perf_counter()
        H = estimate_homography(base_kp, kp_list[i], matches)
        t_homo_sub += time.perf_counter() - t_start_homo
        if H is None:
            print(f"     WARNING: Homography failed for image {i+1}, skipping.", file=sys.stderr)
            continue
        
        # Third Phase: Warping and Blending
        t_start_warp = time.perf_counter()
        try:
            base_image = warp_and_blend(base_image, images[i], H)
        except ValueError as e:
            # Catches canvas explosion exceptions from degenerate homographies
            print(f"     WARNING: {e} -- skipping this image.", file=sys.stderr)
            continue
        t_warp_sub += time.perf_counter() - t_start_warp

        # Fourth Phase: Feature Re-extraction
        # Note: We do not use extract_features() here because we only need to 
        # process a single image (the updated canvas) rather than a list of images.
        # Calling SIFT directly avoids list packing/unpacking overhead.
        t_start_reext = time.perf_counter()
        gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray_base, None)
        t_reext_sub += time.perf_counter() - t_start_reext
        print(f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
              f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

    # Calculate final elapsed times
    t_stitch   = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    # Save output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Print report
    print("\n" + "=" * 50, file=sys.stderr)
    print(f"SEQUENTIAL REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:   {t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total Time:{t_stitch:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:   {t_match_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:    {t_homo_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend:       {t_warp_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract: {t_reext_sub:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    """
    Run the sequential pipeline over the entire dataset in batches (windows).

    Instead of trying to stitch 50 images together (which causes extreme 
    perspective distortion and memory exhaustion), this function breaks the dataset 
    into smaller, manageable "windows" (e.g., 4 images per window) and generates 
    a separate panorama for each window.

    Args:
        input_dir (str): Directory containing the source dataset.
        output_dir (str): Directory where the windowed panoramas will be saved.
        window_size (int): Number of images to stitch per window.
    """
    print(f"STARTING SEQUENTIAL SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    # Accumulators for global performance metrics across all windows
    total_t_extract = 0.0
    total_t_match   = 0.0
    total_t_homo    = 0.0
    total_t_warp    = 0.0
    total_t_reext   = 0.0

    total_start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=8000)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images in chunks [start_idx : start_idx + window_size]
    for start_idx in range(0, total_images, window_size):
        end_idx = min(start_idx + window_size, total_images)
        print(f"\n--- Processing window: Images {start_idx} to {end_idx-1} ---", file=sys.stderr)

        current_images = load_images(input_dir, start_idx, end_idx)

        if len(current_images) < 2:
            print("   WARNING: Not enough images in this window to stitch. Skipping.", file=sys.stderr)
            continue
        
        # Extract features for the current batch
        print("Starting SIFT Feature Extraction for current window...", file=sys.stderr)
        kp_list, des_list, t_extract = extract_features(current_images)
        total_t_extract += t_extract

        # Initialize the new panorama base for this specific window
        base_image = current_images[0]
        base_kp    = kp_list[0]
        base_des   = des_list[0]

        # Iteratively stitch the rest of the window
        for i in range(1, len(current_images)):
            global_img_idx = start_idx + i
            print(f"\n   - Stitching image {global_img_idx}/{total_images-1} onto window panorama...", file=sys.stderr)

            # Match features between the current base panorama and the next image
            t_start_match = time.perf_counter()
            matches = match_features(base_des, des_list[i])
            total_t_match += time.perf_counter() - t_start_match
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {global_img_idx}, skipping.", file=sys.stderr)
                continue
            
            # Homography estimation to align the new image with the current panorama
            t_start_homo = time.perf_counter()
            H = estimate_homography(base_kp, kp_list[i], matches)
            total_t_homo += time.perf_counter() - t_start_homo
            if H is None:
                print(f"     WARNING: Homography failed for image {global_img_idx}, skipping.", file=sys.stderr)
                continue
            
            # Warp and blend
            t_start_warp = time.perf_counter()
            try:
                base_image = warp_and_blend(base_image, current_images[i], H)
            except ValueError as e:
                print(f"     WARNING: {e} -- skipping this image.", file=sys.stderr)
                continue
            total_t_warp += time.perf_counter() - t_start_warp

            # Re-extract features from the updated panorama
            t_start_reext = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            total_t_reext += time.perf_counter() - t_start_reext

            print(f"     Updated window panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                  f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

        # Save the completed window panorama
        final_file_path = output_path / f"panorama_window_{start_idx}_to_{end_idx-1}.jpg"
        cv2.imwrite(str(final_file_path), base_image)
        print(f"\nWindow Panorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Compute overall global times
    total_time = time.perf_counter() - total_start
    total_t_stitch = total_t_match + total_t_homo + total_t_warp + total_t_reext

    # Print final benchmark report
    print("\n" + "=" * 50, file=sys.stderr)
    print("SLIDING WINDOW PERFORMANCE REPORT", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Total Images Processed: {total_images}", file=sys.stderr)
    print(f"Window/Batch Size:      {window_size}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:   {total_t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total Time:{total_t_stitch:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:   {total_t_match:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:    {total_t_homo:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend:       {total_t_warp:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract: {total_t_reext:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def main():
    input_dir  = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input not found. Run the download script first.", file=sys.stderr)
        return

    sliding_window_pipeline(input_dir, output_dir, window_size=4)


if __name__ == "__main__":
    main()