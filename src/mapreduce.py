"""
MapReduce-style hierarchical (tree) stitching for image stitching.

Strategy
--------
sequential.py and parallel.py both fold images into the panorama one at a
time, left to right: img0+img1 -> P1, P1+img2 -> P2, P2+img3 -> P3, ...
This is an O(n) chain of *sequential* dependencies: the step that stitches
image i can only start once the step that stitched image i-1 has finished.
No matter how many cores are available, this linear fold itself never
parallelizes -- exactly the kind of bottleneck Amdahl's law penalizes.

MapReduce breaks this into O(log2 n) levels via a merge tree:

    MAP    (level 0): extract SIFT features for all N images in parallel
    REDUCE (level 1): stitch pairs (0,1), (2,3), (4,5), ... in parallel
                       -> N/2 partial panoramas
    REDUCE (level 2): stitch pairs of partial panoramas in parallel
                       -> N/4 partial panoramas
    ...                                     until 1 panorama remains

Each REDUCE level only depends on the level below it, but within a
level all pairs are independent of each other and can run concurrently
in separate processes. An odd node out at any level is carried over
unchanged to the next level.

Note on phase granularity
---------------------------
_merge_pair_worker performs match + homography + warp + re-extraction as
a single atomic unit inside one worker process. This module's own reports
(stitch_mapreduce, sliding_window_pipeline) only ever measure the MAP
phase ("t_map") and the overall REDUCE phase ("t_reduce_total", all tree
levels combined) as wall-clock time (they never claim a separate match/
homography/warp/re-extraction breakdown, so there's no ambiguous 0.0 to
worry about in this file).


Note on correctness
---------------------------
This is NOT expected to be bit-identical to sequential.py / parallel.py.
Composing homographies in a different order (tree merge vs. linear fold)
changes which images get warped relative to which, and warp interpolation
is order-dependent. The panorama should still be geometrically correct
and visually equivalent, but pixel-level correctness comparisons like the
ones in benchmark.py's compare_outputs() do not apply here -- a dedicated
visual/geometric check is needed instead of a diff against the linear
pipelines.
"""

import cProfile
import pstats
import sys

import sys

import cv2
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from sequential import load_images, match_features, estimate_homography, warp_and_blend


def _extract_worker(img):
    """MAP step: runs in a worker process."""
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
    return img, kp_serialized, des


def _deserialize_kp(kp_serialized):
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


def _merge_pair_worker(args):
    """
    REDUCE step: runs in a worker process.
    Stitches two (image, kp_serialized, des) nodes into one, then
    re-extracts SIFT features on the merged canvas so the result can be
    fed into the next reduce level as an ordinary node.
    """
    cv2.setNumThreads(1)
    (img_a, kp_a_ser, des_a), (img_b, kp_b_ser, des_b) = args

    kp_a = _deserialize_kp(kp_a_ser)
    kp_b = _deserialize_kp(kp_b_ser)

    matches = match_features(des_a, des_b)
    if len(matches) < 4:
        # Not enough matches: keep the left node unchanged, drop the right one
        return (img_a, kp_a_ser, des_a)

    H = estimate_homography(kp_a, kp_b, matches)
    if H is None:
        return (img_a, kp_a_ser, des_a)

    try:
        merged = warp_and_blend(img_a, img_b, H)
    except ValueError as e:
        print(f"   WARNING: {e} keeping left node unchanged for this pair.", file=sys.stderr)
        return (img_a, kp_a_ser, des_a)
    
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    kp_m, des_m = sift.detectAndCompute(gray, None)
    kp_m_ser = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp_m]

    return (merged, kp_m_ser, des_m)


def stitch_mapreduce(input_dir, output_dir, start_idx=0, end_idx=4, num_workers=None):
    """
    Executes the MapReduce (tree-merge) stitching pipeline on a custom
    range of images.
    """
    print(f"\nSTARTING MAPREDUCE PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    images = load_images(input_dir, start_idx=start_idx, end_idx=end_idx)
    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    num_workers = num_workers or os.cpu_count()
    level = 0
    t_map = 0.0
    t_reduce_total = 0.0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # MAP: extract features for every image in parallel
        print("\nMAP phase: parallel SIFT extraction...", file=sys.stderr)
        t_map_start = time.perf_counter()
        nodes = list(executor.map(_extract_worker, images))
        t_map = time.perf_counter() - t_map_start
        print(f"   Extracted features for {len(nodes)} images in {t_map:.3f}s", file=sys.stderr)

        # REDUCE: pairwise tree merge 
        while len(nodes) > 1:
            level += 1
            n_pairs = len(nodes) // 2
            odd_one_out = nodes[-1] if len(nodes) % 2 == 1 else None

            print(f"\nREDUCE level {level}: merging {len(nodes)} nodes into "
                  f"{n_pairs + (1 if odd_one_out is not None else 0)} "
                  f"({n_pairs} parallel merges)...", file=sys.stderr)

            pairs = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(n_pairs)]

            t0 = time.perf_counter()
            merged = list(executor.map(_merge_pair_worker, pairs))
            t_reduce_total += time.perf_counter() - t0

            if odd_one_out is not None:
                merged.append(odd_one_out)

            nodes = merged

    total_time = time.perf_counter() - total_start
    final_image = nodes[0][0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_mr_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), final_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    print("\n" + "=" * 50, file=sys.stderr)
    print(f"MAPREDUCE REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Tree depth (levels):     {level}", file=sys.stderr)
    print(f"MAP (extraction) time:   {t_map:.3f} seconds", file=sys.stderr)
    print(f"REDUCE (merge) time:     {t_reduce_total:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:    {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    print(f"STARTING MAPREDUCE SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---", file=sys.stderr)

            images = load_images(input_dir, start_idx, end_idx)
            if len(images) < 2:
                print("   WARNING: Not enough images in this window to stitch. Skipping.", file=sys.stderr)
                continue

            nodes = list(executor.map(_extract_worker, images))

            while len(nodes) > 1:
                n_pairs = len(nodes) // 2
                odd_one_out = nodes[-1] if len(nodes) % 2 == 1 else None
                pairs = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(n_pairs)]
                merged = list(executor.map(_merge_pair_worker, pairs))
                if odd_one_out is not None:
                    merged.append(odd_one_out)
                nodes = merged

            final_image = nodes[0][0]
            final_file_path = output_path / f"panorama_window_mr_{start_idx}_to_{end_idx - 1}.jpg"
            cv2.imwrite(str(final_file_path), final_image)
            print(f"Window Panorama saved successfully to: {final_file_path}", file=sys.stderr)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 50, file=sys.stderr)
    print("MAPREDUCE SLIDING WINDOW REPORT", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Total Images Processed: {total_images}", file=sys.stderr)
    print(f"Window/Batch Size:      {window_size}", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def main():
    input_dir = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input_reordered not found.", file=sys.stderr)
        return

    profiler = cProfile.Profile()
    profiler.enable()
    
    sliding_window_pipeline(input_dir, output_dir, window_size=4)

    profiler.disable()

    output_file = Path("profiling_results/mapreduce_profiling")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("tottime")
        stats.print_stats()

    print(f"\nProfiling results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()