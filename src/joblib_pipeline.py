"""
joblib_pipeline.py
==================
Joblib-based variant of the parallel stitching pipeline.

Why Joblib?
-----------
extract_features_parallel() in parallel.py and the MAP step in mapreduce.py
are "embarrassingly parallel" loops written manually with
ProcessPoolExecutor.map().  Joblib's Parallel() achieves the same
parallelism with:

  1. Less boilerplate — no manual serialisation of cv2.KeyPoint objects is
     needed; the loky backend does it transparently.

  2. Automatic memmapping of large NumPy arrays (max_nbytes parameter).
     When an input array exceeds max_nbytes Joblib writes it to a temporary
     memory-mapped file and sends only the filename+metadata to each worker,
     which maps it read-only with zero extra copy.  For half-resolution
     frames (≈ 1.5 MB each, well above the 1 MB threshold used here) this
     resolves the same IPC-copy bottleneck addressed by shared_memory_pipeline
     without requiring manual SharedMemory management.

  3. Warm worker pool — when the same Parallel() object (or the same loky
     process pool) is reused across calls, no new processes are spawned for
     subsequent calls, amortising startup cost exactly as a pre-created
     ProcessPoolExecutor would.

Architecture
------------
The pipeline is a drop-in replacement for parallel.py:

    MAP  : Parallel(n_jobs=-1, backend="loky", max_nbytes="1M")(
               delayed(_extract_worker)(img) for img in images
           )

    REDUCE (sequential linear fold, same as parallel.py): match, homography,
           warp_and_blend_tiling.

Key parameters
--------------
  n_jobs=-1        : use all available logical CPUs.
  backend="loky"   : loky spawns fresh processes and avoids GIL / OpenCV
                     state-sharing issues (same backend Joblib uses by default
                     on most platforms).
  max_nbytes="1M"  : arrays larger than 1 MiB are memmapped automatically
                     instead of pickled -- the threshold is intentionally low
                     so that even a single channel of a half-resolution frame
                     (≈ 520 KB) is just below it while a full BGR frame
                     (≈ 1.5 MB) triggers memmap.
  prefer="processes": explicit hint that keeps Joblib from downgrading to
                     threads even in unusual environments.

Benchmark compatibility
-----------------------
The module exposes:
  * extract_features_joblib(images) -> (kp_list, des_list, t)
    Drop-in for extract_features_parallel() from parallel.py.
  * stitch_joblib(input_dir, output_dir, start_idx, end_idx)
    Full stand-alone pipeline entry-point.
  * sliding_window_pipeline(input_dir, output_dir, window_size)
    Sliding-window entry-point.
  * _time_joblib_window(images, thread_executor, seed)
    Benchmark harness callable for benchmark.py PipelineSpec.

Note: Joblib manages its own internal worker pool (via loky); the benchmark's
shared ProcessPoolExecutor is intentionally NOT passed to Joblib — it would
conflict with loky's own pool management.  Therefore PipelineSpec for this
module should set needs_process_pool=False and needs_thread_pool=True (the
ThreadPoolExecutor is still needed for warp_and_blend_tiling).
"""
import cProfile
import pstats
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from joblib import Parallel, delayed

from sequential import (
    match_features,
    estimate_homography,
)
from parallel import (
    load_images_parallel,
    warp_and_blend_tiling,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Arrays larger than this threshold are memmapped instead of pickled.
# At 1 MB, all half-resolution BGR frames (≈ 1.5 MB) are memmapped;
# small arrays (keypoint data, descriptors slices) are still pickled normally.
_MEMMAP_THRESHOLD = "1M"

# Number of parallel jobs: -1 = all CPUs.
_N_JOBS = -1

# Joblib backend: loky spawns fresh processes, avoids GIL, compatible with
# OpenCV's internal state.
_BACKEND = "loky"


# ---------------------------------------------------------------------------
# Worker function (runs inside each loky worker process)
# ---------------------------------------------------------------------------

def _extract_worker(img):
    """
    MAP step: compute SIFT keypoints + descriptors for one image.

    This is intentionally identical in signature to the worker used by
    parallel.py so that the two pipelines can be compared directly.
    cv2.setNumThreads(1) prevents OpenCV from spawning its own sub-threads
    inside each already-parallel worker process (oversubscription avoidance).

    When called via Parallel(max_nbytes="1M"), Joblib will have written `img`
    to a memory-mapped tempfile if it exceeded the threshold; the worker
    receives a read-only np.memmap instead of a copy, so no IPC-copy occurs.
    """
    cv2.setNumThreads(1)

    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # cv2.KeyPoint objects are not natively picklable; serialise to tuples.
    kp_ser = [
        (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
        for p in kp
    ]
    return kp_ser, des


def _deserialize_kp(kp_serialized):
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


# ---------------------------------------------------------------------------
# Public API – extraction phase
# ---------------------------------------------------------------------------

def extract_features_joblib(images):
    """
    Phase 2 (MAP): parallel SIFT extraction via Joblib.

    Replaces the manual ProcessPoolExecutor.map() in parallel.py with
    Parallel(n_jobs=-1, backend="loky", max_nbytes="1M"), which handles:
      - process spawning / pool reuse (loky keeps a warm pool between calls);
      - automatic memmapping of large arrays (max_nbytes threshold);
      - serialisation of return values (kp_ser, des) as usual.

    Args:
        images : list of np.ndarray (BGR, uint8)

    Returns:
        (keypoints_list, descriptors_list, extraction_time)
        Same signature as extract_features_parallel() in parallel.py.
    """
    start = time.perf_counter()

    n_workers = os.cpu_count()
    print(
        f"   [Joblib] SIFT extraction — loky backend, {n_workers} workers, "
        f"memmap threshold={_MEMMAP_THRESHOLD}...", file=sys.stderr
    )

    # Joblib dispatches one _extract_worker call per image; arrays > 1 MB
    # are memmapped, smaller ones are pickled as normal.
    results = Parallel(
        n_jobs=_N_JOBS,
        backend=_BACKEND,
        max_nbytes=_MEMMAP_THRESHOLD,
        prefer="processes",
    )(delayed(_extract_worker)(img) for img in images)

    keypoints_list = []
    descriptors_list = []

    for i, (kp_ser, des) in enumerate(results):
        kp = _deserialize_kp(kp_ser)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"      - Image {i + 1}: Found {len(kp)} keypoints", file=sys.stderr)

    extraction_time = time.perf_counter() - start
    return keypoints_list, descriptors_list, extraction_time


# ---------------------------------------------------------------------------
# Public API – full pipeline (linear fold, mirrors parallel.py)
# ---------------------------------------------------------------------------

def stitch_joblib(input_dir, output_dir, start_idx=0, end_idx=4):
    """
    Full Joblib stitching pipeline on a custom image range.
    Uses the same left-to-right linear fold as parallel.py.
    """
    print(f"\nSTARTING JOBLIB PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    images = load_images_parallel(input_dir, start_idx=start_idx, end_idx=end_idx)

    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    print("\nStarting Joblib SIFT Feature Extraction...", file=sys.stderr)

    # Joblib manages its own loky pool; we only need a ThreadPoolExecutor for
    # the tile-blending phase.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:

        kp_list, des_list, t_extract = extract_features_joblib(images)

        print("\nStarting Iterative Stitching...", file=sys.stderr)
        stitch_start = time.perf_counter()

        base_image = images[0]
        base_kp    = kp_list[0]
        base_des   = des_list[0]
        sift = cv2.SIFT_create(nfeatures=8000)

        t_match_sub = t_homo_sub = t_warp_sub = t_reext_sub = 0.0

        for i in range(1, len(images)):
            print(f"\n   - Stitching image {i + 1} onto current panorama...", file=sys.stderr)

            t0 = time.perf_counter()
            matches = match_features(base_des, des_list[i])
            t_match_sub += time.perf_counter() - t0
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {i + 1}, skipping.", file=sys.stderr)
                continue

            t0 = time.perf_counter()
            H = estimate_homography(base_kp, kp_list[i], matches)
            t_homo_sub += time.perf_counter() - t0
            if H is None:
                print(f"     WARNING: Homography failed for image {i + 1}, skipping.", file=sys.stderr)
                continue

            t0 = time.perf_counter()
            try:
                base_image = warp_and_blend_tiling(base_image, images[i], thread_executor, H)
            except ValueError as e:
                print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
                continue
            t_warp_sub += time.perf_counter() - t0

            t0 = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            t_reext_sub += time.perf_counter() - t0
            print(
                f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

    t_stitch   = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_joblib_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    print("\n" + "=" * 50, file=sys.stderr)
    print(f"JOBLIB REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:    {t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total:      {t_stitch:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:    {t_match_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:     {t_homo_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend (Tile): {t_warp_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract:  {t_reext_sub:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:    {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    """Sliding-window entry-point — mirrors parallel.py's sliding_window_pipeline."""
    print(f"STARTING JOBLIB SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted(
        [p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')]
    )
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    total_t_extract = total_t_match = total_t_homo = total_t_warp = total_t_reext = 0.0
    total_start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=8000)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # A single ThreadPoolExecutor is reused across windows (same pattern as
    # parallel.py). Joblib manages its own loky pool internally.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---", file=sys.stderr)

            current_images = load_images_parallel(input_dir, start_idx, end_idx)
            if len(current_images) < 2:
                print("   WARNING: Insufficient images in this window. Skipping.", file=sys.stderr)
                continue

            print("Starting Joblib SIFT Feature Extraction for current window...", file=sys.stderr)
            kp_list, des_list, t_extract = extract_features_joblib(current_images)
            total_t_extract += t_extract

            base_image = current_images[0]
            base_kp    = kp_list[0]
            base_des   = des_list[0]

            for i in range(1, len(current_images)):
                global_img_idx = start_idx + i
                print(
                    f"\n   - Stitching image {global_img_idx}/{total_images - 1} "
                    f"onto window panorama...", file=sys.stderr
                )

                t0 = time.perf_counter()
                matches = match_features(base_des, des_list[i])
                total_t_match += time.perf_counter() - t0
                print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_img_idx}, skipping.", file=sys.stderr)
                    continue

                t0 = time.perf_counter()
                H = estimate_homography(base_kp, kp_list[i], matches)
                total_t_homo += time.perf_counter() - t0
                if H is None:
                    print(f"     WARNING: Homography failed for image {global_img_idx}, skipping.", file=sys.stderr)
                    continue

                t0 = time.perf_counter()
                try:
                    base_image = warp_and_blend_tiling(base_image, current_images[i], thread_executor, H)
                except ValueError as e:
                    print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
                    continue
                total_t_warp += time.perf_counter() - t0

                t0 = time.perf_counter()
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                total_t_reext += time.perf_counter() - t0
                print(
                    f"     Updated window panorama: "
                    f"{base_image.shape[1]}x{base_image.shape[0]} px, "
                    f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr
                )

            final_file_path = (
                output_path / f"panorama_window_joblib_{start_idx}_to_{end_idx - 1}.jpg"
            )
            cv2.imwrite(str(final_file_path), base_image)
            print(f"\nWindow Panorama saved successfully to: {final_file_path}", file=sys.stderr)

    total_time = time.perf_counter() - total_start
    total_t_stitch = total_t_match + total_t_homo + total_t_warp + total_t_reext

    print("\n" + "=" * 50, file=sys.stderr)
    print("JOBLIB PIPELINE PERFORMANCE REPORT", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Total Images Processed:    {total_images}", file=sys.stderr)
    print(f"Window/Batch Size:         {window_size}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)
    print(f"SIFT Extraction Time:      {total_t_extract:.3f} seconds", file=sys.stderr)
    print(f"Match & Warp Total:        {total_t_stitch:.3f} seconds", file=sys.stderr)
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
        print("ERROR: Directory data/input_reordered not found.", file=sys.stderr)
        return

    # Let OpenCV use all native threads for its own internal parallelism
    # (fine here since Joblib workers each call cv2.setNumThreads(1)).
    cv2.setNumThreads(os.cpu_count())

    profiler = cProfile.Profile()
    profiler.enable()

    sliding_window_pipeline(input_dir, output_dir, window_size=4)

    profiler.disable()

    output_file = Path("profiling_results/joblib_profiling")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("tottime")
        stats.print_stats()

    print(f"\nProfiling results saved to: {output_file}", file=sys.stderr)

if __name__ == "__main__":
    main()
