"""
window_diagnostics.py
======================
Standalone diagnostic: for each non-overlapping window of images, tests
EVERY pair (i, j) with i < j -- not just pairs anchored to the window's
first image, which is all the stitching pipelines ever try.

Why this exists
----------------
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
    - a verdict: EMPTY (no usable pair at all -- safe to consider for
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
WINDOW_SIZE            = 4        # number of images per window
N_FEATURES             = 8000     # SIFT threshold
RATIO_TEST_THRESHOLD   = 0.7      # Lowe's ratio test
RANSAC_REPROJ_THRESH   = 5.0
MIN_INLIERS_THRESHOLD  = 15       # minimum inliers to consider a pair "usable" (pass/fail)
MIN_MATCHES_REQUIRED   = 4        # minimum matches to even attempt RANSAC (otherwise cv2.findHomography fails)
RNG_SEED               = 42


def load_images(input_dir: str, start_idx: int, end_idx: int) -> list:
    """Same loading/downscale convention as sequential.py's load_images."""
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
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        images.append(img)
    return images


def extract_features(img):
    """SIFT with the same nfeatures cap used everywhere else in the project."""
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def match_features(des1, des2):
    """FLANN + Lowe's ratio test, same parameters as sequential.py."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_TEST_THRESHOLD * n.distance:
            good.append(m)
    return good


def count_inliers(kp1, kp2, matches):
    """
    Runs RANSAC homography estimation and returns (num_inliers, H_is_valid).
    Mirrors estimate_homography's logic in sequential.py, but always
    returns the inlier count even below MIN_INLIERS_THRESHOLD, since here
    we want the raw signal, not a pass/fail gate baked into the return value.
    """
    if len(matches) < MIN_MATCHES_REQUIRED:
        return 0, False

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or mask is None:
        return 0, False

    inliers = int(mask.sum())
    return inliers, True


def diagnose_window(images: list) -> dict:
    """
    Tests every pair (i, j), i < j, within one window's images.
    Returns a dict with the full inlier matrix and per-pair details.
    """
    n = len(images)
    keypoints, descriptors = [], []
    for img in images:
        kp, des = extract_features(img)
        keypoints.append(kp)
        descriptors.append(des)

    inlier_matrix = np.zeros((n, n), dtype=int)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            matches = match_features(descriptors[i], descriptors[j])
            inliers, h_valid = count_inliers(keypoints[i], keypoints[j], matches)
            passes = h_valid and inliers >= MIN_INLIERS_THRESHOLD
            inlier_matrix[i, j] = inliers
            inlier_matrix[j, i] = inliers
            pairs.append({
                "i": i, "j": j,
                "num_matches": len(matches),
                "num_inliers": inliers,
                "passes": passes,
            })

    passing_pairs = [p for p in pairs if p["passes"]]
    anchor_pairs = [p for p in pairs if p["i"] == 0]
    anchor_passes = any(p["passes"] for p in anchor_pairs)

    if len(passing_pairs) == 0:
        verdict = "EMPTY (no usable pair at all: this window is a genuine candidate for removal)"
    elif anchor_passes:
        verdict = "CONNECTED (the pair anchored on image[0] works: the current pipeline already exploits it)"
    else:
        verdict = "PARTIAL (valid pairs exist, but not anchored on image[0]: an algorithm limitation, not a data problem)"

    return {
        "n": n,
        "inlier_matrix": inlier_matrix,
        "pairs": pairs,
        "passing_pairs": passing_pairs,
        "verdict": verdict,
    }


def print_report(win_idx: int, start: int, end: int, diag: dict):
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
    if not Path(INPUT_DIR).exists():
        print(f"ERROR: directory '{INPUT_DIR}' not found.")
        return

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
    empty_windows, partial_windows, connected_windows = [], [], []

    for win_idx, (start, end) in enumerate(windows):
        images = load_images(INPUT_DIR, start, end)
        if len(images) < 2:
            print(f"\nWindow {win_idx} [{start}:{end}]: fewer than 2 images loaded, skipping.")
            continue

        diag = diagnose_window(images)
        print_report(win_idx, start, end, diag)
        write_pairs_csv(RESULTS_DIR, win_idx, start, end, diag)
        write_summary_csv(RESULTS_DIR, win_idx, start, end, diag)

        if diag["verdict"].startswith("EMPTY"):
            empty_windows.append(win_idx)
        elif diag["verdict"].startswith("PARTIAL"):
            partial_windows.append(win_idx)
        else:
            connected_windows.append(win_idx)

    total_time = time.perf_counter() - t_start

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

    if partial_windows:
        print(f"\nWARNING: {len(partial_windows)} window(s) have valid pairs that the current "
              f"anchoring algorithm never finds (it stays anchored on image[0] after a failure). "
              f"Deleting these windows would throw away recoverable panorama.")
    if empty_windows and not partial_windows:
        print(f"\n{len(empty_windows)} window(s) have NO usable pair at all: "
              f"removal is an informed decision, not a shortcut.")


if __name__ == "__main__":
    main()