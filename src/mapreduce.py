"""
MapReduce-style hierarchical (tree) stitching for image stitching.

Strategy:
sequential.py and parallel.py both fold images into the panorama one at a
time, left to right: img0+img1 -> P1, P1+img2 -> P2, P2+img3 -> P3, ...
This is an O(n) chain of *sequential* dependencies: the step that stitches
image i can only start once the step that stitched image i-1 has finished.
No matter how many cores are available, this linear fold itself never
parallelizes.

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

Note on phase granularity:
_merge_pair_worker performs match + homography + warp + re-extraction as
a single atomic unit inside one worker process. This module's own reports
(stitch_mapreduce, sliding_window_pipeline) only ever measure the MAP
phase ("t_map") and the overall REDUCE phase ("t_reduce_total", all tree
levels combined) as wall-clock time (they never claim a separate match/
homography/warp/re-extraction breakdown, so there's no ambiguous 0.0 to
worry about in this file).


Note on correctness:
This is NOT expected to be bit-identical to sequential.py / parallel.py.
Composing homographies in a different order (tree merge vs. linear fold)
changes which images get warped relative to which, and warp interpolation
is order-dependent.
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

# Reusing proven sequential matching, homography, and warping blocks
from sequential import load_images, match_features, estimate_homography, warp_and_blend

NUM_CORES = 8

def _extract_worker(img):
    """
    MAP Phase Worker: Performs concurrent SIFT extraction inside a worker process.

    Each image is processed completely independently. Since this is an isolated process,
    OpenCV's internal multi-threading is limited to prevent massive CPU oversubscription 
    when running across all logical cores.

    Args:
        img (numpy.ndarray): Input source image array.

    Returns:
        tuple: (Original image matrix, List of serialized keypoint primitive tuples, Descriptors array)
    """

    cv2.setNumThreads(1)    # Prevent CPU oversubscription across parallel worker processes
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # Pack C++ cv2.KeyPoint fields into picklable python tuples for IPC
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
    return img, kp_serialized, des


def _deserialize_kp(kp_serialized):
    """
    Utility helper to deserialize primitive python tuples back into concrete cv2.KeyPoint objects.

    Args:
        kp_serialized (list): List of keypoint attribute tuples passed from a worker process.

    Returns:
        list: Reconstructed cv2.KeyPoint instances ready for OpenCV structural calculations.
    """
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


def _merge_pair_worker(args):
    """
    REDUCE Phase Worker: Stitches a pair of nodes and computes the next tree-level node.

    This function acts as an atomic MapReduce transaction block:
    1. Deserializes keypoints for two incoming adjacent nodes.
    2. Matches features, estimates RANSAC homography, and blends them into a partial panorama.
    3. Immediately re-extracts features on the new canvas, preparing it to serve as a single
       homogeneous node for the next hierarchical reduction level up the tree.

    Args:
        args (tuple): Contiguous pair of nodes -> ((img_a, kp_a_ser, des_a), (img_b, kp_b_ser, des_b))

    Returns:
        tuple: A single merged node (merged_img, kp_m_ser, des_m). If blending fails or matches 
               are insufficient, it degrades gracefully by returning the left node intact.
    """
    cv2.setNumThreads(1)
    (img_a, kp_a_ser, des_a), (img_b, kp_b_ser, des_b) = args

    # Restore serialized keypoints to concrete cv2.KeyPoint objects for OpenCV operations
    kp_a = _deserialize_kp(kp_a_ser)
    kp_b = _deserialize_kp(kp_b_ser)

    # Core stitching workflow executed atomically inside the worker
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
    
    # Re-extraction: Turn the merged canvas into a valid node for the next tree level
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    kp_m, des_m = sift.detectAndCompute(gray, None)
    kp_m_ser = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp_m]

    return (merged, kp_m_ser, des_m)


def stitch_mapreduce(input_dir, output_dir, start_idx=0, end_idx=4, num_workers=None):
    """
    Executes a single standalone MapReduce (tree-merge) stitching pipeline over a selected range.

    Args:
        input_dir (str): Input directory containing source image files.
        output_dir (str): Destination directory for the generated panorama slice.
        start_idx (int): Inclusive starting image index.
        end_idx (int): Exclusive ending image index.
        num_workers (int, optional): Execution width. Defaults to total logical system CPU cores.
    """

    print(f"\nSTARTING MAPREDUCE PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    images = load_images(input_dir, start_idx=start_idx, end_idx=end_idx)
    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    num_workers = num_workers or NUM_CORES
    level = 0
    t_map = 0.0
    t_reduce_total = 0.0

    # Instantiate the process pool context manager
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # MAP: extract features for every image in parallel
        print("\nMAP phase: parallel SIFT extraction...", file=sys.stderr)

        # Parallel feature extraction maps across all loaded raw input images
        t_map_start = time.perf_counter()
        nodes = list(executor.map(_extract_worker, images))
        t_map = time.perf_counter() - t_map_start
        print(f"   Extracted features for {len(nodes)} images in {t_map:.3f}s", file=sys.stderr)

        # REDUCE: pairwise tree merge 
        while len(nodes) > 1:
            level += 1
            n_pairs = len(nodes) // 2

            # Handle odd datasets: isolate the trailing node to pass it up to the next level unchanged
            odd_one_out = nodes[-1] if len(nodes) % 2 == 1 else None

            print(f"\nREDUCE level {level}: merging {len(nodes)} nodes into "
                  f"{n_pairs + (1 if odd_one_out is not None else 0)} "
                  f"({n_pairs} parallel merges)...", file=sys.stderr)

            # Zip pairs of adjacent nodes (0 with 1, 2 with 3, etc.)
            pairs = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(n_pairs)]

            t0 = time.perf_counter()

            # Parallel execution of independent pairwise combinations at the current tree depth
            merged = list(executor.map(_merge_pair_worker, pairs))
            t_reduce_total += time.perf_counter() - t0

            # Re-attach the odd unmerged element back into the tree sequence
            if odd_one_out is not None:
                merged.append(odd_one_out)

            # Update nodes list: the newly reduced partial panoramas become the next layer inputs
            nodes = merged

    total_time = time.perf_counter() - total_start
    final_image = nodes[0][0]   # The root node represents the completed panorama matrix

    # Save the final panorama to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_mr_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), final_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Log execution timings
    print("\n" + "=" * 50, file=sys.stderr)
    print(f"MAPREDUCE REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Tree depth (levels):     {level}", file=sys.stderr)
    print(f"MAP (extraction) time:   {t_map:.3f} seconds", file=sys.stderr)
    print(f"REDUCE (merge) time:     {t_reduce_total:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:    {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    """
    Slices a large dataset into segments and executes MapReduce tree-merging inside each window.

    Maintains a single long-lived ProcessPoolExecutor block encapsulating the main loop.
    This structure prevents the system from re-allocating or killing child worker processes 
    between windows, significantly optimizing memory throughput and system time.

    Args:
        input_dir (str): Location directory containing source imagery frames.
        output_dir (str): Target output destination path for independent window results.
        window_size (int): Max images allowed inside an isolated tree-merge transaction context.
    """

    print(f"STARTING MAPREDUCE SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    # Workers remain warm and allocated across the entire sequential window tracking path.
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---", file=sys.stderr)

            images = load_images(input_dir, start_idx, end_idx)
            if len(images) < 2:
                print("   WARNING: Not enough images in this window to stitch. Skipping.", file=sys.stderr)
                continue
            
            # Parallel MAP phase for the current window chunk
            nodes = list(executor.map(_extract_worker, images))

            # Internal Tree Reduction loop running inside the window bounds
            while len(nodes) > 1:
                n_pairs = len(nodes) // 2
                odd_one_out = nodes[-1] if len(nodes) % 2 == 1 else None
                pairs = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(n_pairs)]

                # Parallel REDUCE step via warm background workers
                merged = list(executor.map(_merge_pair_worker, pairs))
                if odd_one_out is not None:
                    merged.append(odd_one_out)
                nodes = merged

            # Save the localized root panorama node
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