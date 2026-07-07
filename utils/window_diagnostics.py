"""
Standalone diagnostic: for each non-overlapping window of images, tests
EVERY pair (i, j) with i < j ( not just pairs anchored to the window's
first image, which is all the stitching pipelines ever try).

Why this exists:
In the stitching pipelines (sequential.py, parallel.py, ...), a failed
homography leaves the anchor unchanged:

    if H is None:
        continue   # base_kp/base_des are NOT updated

So a window that fails on image[0] vs image[1] never even attempts
image[1] vs image[2] or image[2] vs image[3]. A window reported as
"0 successful stitches" by the benchmark may still contain perfectly
good pairs that the anchoring logic never got to try.

This script removes that blind spot: it extracts features once per
image, then brute-force tests all C(n,2) pairs within each window and
reports, per window:
    - an inlier-count matrix for every pair
    - which pairs pass the RANSAC inlier threshold
    - a verdict: EMPTY (no usable pair at all, then safe to consider for
      removal), PARTIAL (some pairs work, the anchoring algorithm is the
      bottleneck, not the data), or CONNECTED (the naive anchor-first
      pair already works, matches the pipelines' own behavior).

This script is intentionally self-contained: it does not import
sequential.py / parallel.py, so it can be run and reasoned about in
isolation. The SIFT/FLANN/RANSAC parameters below are kept identical to
sequential.py's so the "usable pair" verdict is a fair, apples-to-apples
signal for what the actual pipelines could achieve.

Usage:
    python window_diagnostics.py

Output:
    - printed per-window matrices and verdicts
    - results/window_diagnostics_pairs.csv   (one row per tested pair)
    - results/window_diagnostics_summary.csv (one row per window)
"""

import csv
import time
from pathlib import Path

import cv2
import numpy as np

# Configuration
INPUT_DIR              = "data/input"
RESULTS_DIR            = "results"
WINDOW_SIZE            = 4        # Number of images per diagnostic batch
N_FEATURES             = 8000     # Maximum SIFT keypoints to extract
RATIO_TEST_THRESHOLD   = 0.7      # Lowe's ratio test strictness (lower = stricter)
RANSAC_REPROJ_THRESH   = 5.0      # Maximum pixel distance to consider a point an inlier
MIN_INLIERS_THRESHOLD  = 15       # Minimum inliers required to declare a pair "usable"
MIN_MATCHES_REQUIRED   = 4        # Mathematical minimum to compute a 3x3 Homography matrix
RNG_SEED               = 42         


def load_images(input_dir: str, start_idx: int, end_idx: int) -> list:
    """
    Loads and downscales a specific range of images from the input directory.

    Mirrors the loading convention used in sequential.py to ensure 
    diagnostics are run on the exact same pixel data as the main pipeline.

    Args:
        input_dir (str): Path to the folder containing source images.
        start_idx (int): Starting index for the current window.
        end_idx (int): Ending index (exclusive) for the current window.

    Returns:
        list: A list of BGR image arrays, downscaled by 50%.
    """
    paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])[start_idx:end_idx]

    images = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"   WARNING: could not load {path}, skipping.")
            continue

        # Downscale by a factor of 2 to match the main pipeline's memory footprint
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        images.append(img)
    return images


def extract_features(img):
    """
    Detects keypoints and computes SIFT descriptors for a single image.

    Args:
        img (np.ndarray): The input BGR image.

    Returns:
        tuple: (list of cv2.KeyPoint, np.ndarray of descriptors)
    """

    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_features(des1, des2):
    """
    Finds robust matches between two sets of descriptors using FLANN 
    and Lowe's ratio test.

    Args:
        des1 (np.ndarray): Descriptors from the first image.
        des2 (np.ndarray): Descriptors from the second image.

    Returns:
        list: A list of cv2.DMatch objects that passed the ratio test.
    """
    # Guard clause against empty or insufficient descriptors
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    # Initialize Fast Library for Approximate Nearest Neighbors (FLANN)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the 2 nearest neighbors for each descriptor
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair

        # Lowe's ratio test: keep match only if the closest neighbor is 
        # significantly closer than the second closest.
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good.append(m)
    return good


def count_inliers(kp1, kp2, matches):
    """
    Estimates a homography matrix using RANSAC to count geometric inliers.

    Unlike the main pipeline which acts as a boolean gate (returning H or None), 
    this function returns the raw inlier count. This provides high-resolution 
    signal about pair quality, even if it falls below MIN_INLIERS_THRESHOLD.

    Args:
        kp1 (list): Keypoints from the query image.
        kp2 (list): Keypoints from the train image.
        matches (list): Valid cv2.DMatch objects between kp1 and kp2.

    Returns:
        tuple: (number of inliers (int), whether homography computation succeeded (bool))
    """

    if len(matches) < MIN_MATCHES_REQUIRED:
        return 0, False

    # Extract strictly the (x, y) coordinates for the matched keypoints
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # findHomography returns the transformation matrix and a mask (1 for inlier, 0 for outlier)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)

    if H is None or mask is None:
        return 0, False

    # Summing the mask gives the exact count of geometrically consistent points
    inliers = int(mask.sum())
    return inliers, True


def diagnose_window(images: list) -> dict:
    """
    Performs a brute-force comparison of all image pairs in the window.

    Extracts features once per image to save compute, then matches every possible 
    (i, j) combination where i < j. Classifies the entire window based on the 
    topology of successful matches.

    Args:
        images (list): Batch of images for the current window.

    Returns:
        dict: A comprehensive report containing the inlier matrix, list of pairs, 
              and the final window verdict (EMPTY, PARTIAL, or CONNECTED).
    """
    n = len(images)
    keypoints, descriptors = [], []

    # Extract features once per image (O(N))
    for img in images:
        kp, des = extract_features(img)
        keypoints.append(kp)
        descriptors.append(des)

    inlier_matrix = np.zeros((n, n), dtype=int)
    pairs = []

    # Brute-force pairwise matching (O(N^2))
    for i in range(n):
        for j in range(i + 1, n):
            matches = match_features(descriptors[i], descriptors[j])
            inliers, h_valid = count_inliers(keypoints[i], keypoints[j], matches)
            passes = h_valid and inliers >= MIN_INLIERS_THRESHOLD

            # Populate the symmetric matrix for reporting
            inlier_matrix[i, j] = inliers
            inlier_matrix[j, i] = inliers
            pairs.append({
                "i": i, "j": j,
                "num_matches": len(matches),
                "num_inliers": inliers,
                "passes": passes,
            })

    # Topology Analysis & Verdict Generation
    passing_pairs = [p for p in pairs if p["passes"]]
    anchor_pairs = [p for p in pairs if p["i"] == 0]
    anchor_passes = any(p["passes"] for p in anchor_pairs)

    if len(passing_pairs) == 0:

        # Absolutely no pairs can be stitched. Safe to drop
        verdict = "EMPTY (no usable pair at all: this window is a genuine candidate for removal)"

    elif anchor_passes:
        # Image 0 connects to something. The standard pipeline handles this fine
        verdict = "CONNECTED (the pair anchored on image[0] works: the current pipeline already exploits it)"

    else:
        # Image 0 is a dead-end, but e.g., Image 1 and Image 2 stitch perfectly.
        # The standard pipeline would fail this window entirely.
        verdict = "PARTIAL (valid pairs exist, but not anchored on image[0]: an algorithm limitation, not a data problem)"

    return {
        "n": n,
        "inlier_matrix": inlier_matrix,
        "pairs": pairs,
        "passing_pairs": passing_pairs,
        "verdict": verdict,
    }


def print_report(win_idx: int, start: int, end: int, diag: dict):
    """Prints a visually formatted terminal report for a single window."""
    n = diag["n"]
    print(f"\n{'-' * 60}")
    print(f"WINDOW {win_idx}  [images {start}:{end}]  ({n} images)")
    print(f"{'-' * 60}")

    print("Inlier matrix (rows/cols = local image index within window):")
    header = "      " + "".join(f"{j:>7d}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"  {i:>2d} |"
        for j in range(n):
            if i == j:
                row += f"{'--':>7}"
            else:
                mark = "*" if diag["inlier_matrix"][i, j] >= MIN_INLIERS_THRESHOLD else " "
                row += f"{diag['inlier_matrix'][i, j]:>6d}{mark}"
        print(row)
    print(f"  (* = >= {MIN_INLIERS_THRESHOLD} inliers, passes the threshold)")

    print(f"\n  Total pairs tested: {len(diag['pairs'])}")
    print(f"  Passing pairs:      {len(diag['passing_pairs'])}")
    if diag["passing_pairs"]:
        pair_list = ", ".join(f"({p['i']},{p['j']})" for p in diag["passing_pairs"])
        print(f"  Valid pairs:        {pair_list}")
    print(f"  Verdict:            {diag['verdict']}")


def write_pairs_csv(results_dir: str, win_idx: int, start: int, end: int, diag: dict):
    """Appends granular, pair-level diagnostic data to a CSV file."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "window_diagnostics_pairs.csv"
    write_header = not csv_path.exists()

    fieldnames = [
        "window_idx", "img_start", "img_end",
        "local_i", "local_j", "global_i", "global_j",
        "num_matches", "num_inliers", "passes_threshold",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for p in diag["pairs"]:
            writer.writerow({
                "window_idx": win_idx,
                "img_start": start,
                "img_end": end,
                "local_i": p["i"],
                "local_j": p["j"],
                "global_i": start + p["i"],
                "global_j": start + p["j"],
                "num_matches": p["num_matches"],
                "num_inliers": p["num_inliers"],
                "passes_threshold": p["passes"],
            })


def write_summary_csv(results_dir: str, win_idx: int, start: int, end: int, diag: dict):
    """Appends high-level window verdicts to a summary CSV file."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "window_diagnostics_summary.csv"
    write_header = not csv_path.exists()

    fieldnames = [
        "window_idx", "img_start", "img_end", "num_images",
        "total_pairs", "passing_pairs", "verdict",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "window_idx": win_idx,
            "img_start": start,
            "img_end": end,
            "num_images": diag["n"],
            "total_pairs": len(diag["pairs"]),
            "passing_pairs": len(diag["passing_pairs"]),
            "verdict": diag["verdict"],
        })


def main():
    """
    Main entry point. Iterates over the dataset in chunks (windows),
    runs the pairwise diagnostic on each chunk, and prints a final aggregate report.
    """
    if not Path(INPUT_DIR).exists():
        print(f"ERROR: directory '{INPUT_DIR}' not found.")
        return

    # Lock RNG seeds and thread limits for strict reproducibility
    cv2.setRNGSeed(RNG_SEED)
    cv2.setNumThreads(1)

    all_paths = sorted([
        p for p in Path(INPUT_DIR).iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])
    total_images = len(all_paths)
    if total_images < 2:
        print("ERROR: at least 2 images are required.")
        return

    # Generate non-overlapping sliding windows
    windows = [
        (start, min(start + WINDOW_SIZE, total_images))
        for start in range(0, total_images, WINDOW_SIZE)
        if min(start + WINDOW_SIZE, total_images) - start >= 2
    ]

    print("=" * 60)
    print(f"WINDOW DIAGNOSTICS: {len(windows)} windows, window_size={WINDOW_SIZE}")
    print(f"Inlier threshold: {MIN_INLIERS_THRESHOLD}  |  SIFT nfeatures: {N_FEATURES}")
    print("=" * 60)

    t_start = time.perf_counter()

    # Tracking arrays for final summary
    empty_windows, partial_windows, connected_windows = [], [], []

    for win_idx, (start, end) in enumerate(windows):
        images = load_images(INPUT_DIR, start, end)
        if len(images) < 2:
            print(f"\nWindow {win_idx} [{start}:{end}]: fewer than 2 images loaded, skipping.")
            continue
        
        # Execute core logic
        diag = diagnose_window(images)

        # Persist and print results
        print_report(win_idx, start, end, diag)
        write_pairs_csv(RESULTS_DIR, win_idx, start, end, diag)
        write_summary_csv(RESULTS_DIR, win_idx, start, end, diag)

        # Categorize the result for the summary
        if diag["verdict"].startswith("EMPTY"):
            empty_windows.append(win_idx)
        elif diag["verdict"].startswith("PARTIAL"):
            partial_windows.append(win_idx)
        else:
            connected_windows.append(win_idx)

    total_time = time.perf_counter() - t_start

    # Final Execution Summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total windows analyzed: {len(windows)}")
    print(f"  CONNECTED (fine with the current algorithm): {len(connected_windows)}  {connected_windows}")
    print(f"  PARTIAL   (valid data, algorithm loses it):  {len(partial_windows)}  {partial_windows}")
    print(f"  EMPTY     (no usable pair, removal candidates): {len(empty_windows)}  {empty_windows}")
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Pair-level detail: {RESULTS_DIR}/window_diagnostics_pairs.csv")
    print(f"Window summary:    {RESULTS_DIR}/window_diagnostics_summary.csv")
    print("=" * 60)

    # Actionable warnings based on the findings
    if partial_windows:
        print(f"\nWARNING: {len(partial_windows)} window(s) have valid pairs that the current "
              f"anchoring algorithm never finds (it stays anchored on image[0] after a failure). "
              f"Deleting these windows would throw away recoverable panorama.")
    if empty_windows and not partial_windows:
        print(f"\n{len(empty_windows)} window(s) have NO usable pair at all: "
              f"removal is an informed decision, not a shortcut.")


if __name__ == "__main__":
    main()