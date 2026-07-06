"""
shared_memory_pipeline.py
==========================
Shared-Memory variant of the parallel stitching pipeline.

Problem solved
--------------
In parallel.py, mapreduce.py and producer_consumer.py every image array that
crosses a process boundary is serialised (pickle) and copied in full by the
multiprocessing IPC mechanism.  For half-resolution frames (≈ 960×540 × 3 B ≈
1.5 MB each) this overhead is paid:

  * once per image in the MAP / extraction phase (executor.map sends the full
    array to each worker);
  * once per merge in the REDUCE levels of the tree (mapreduce.py sends two
    partial panoramas to _merge_pair_worker, which can be many MB each).

In a benchmark with 2 warm-up + N_RUNS timed passes the same set of images is
resent every single pass.

Strategy
--------
1. After load_images() allocates the window's image list, each array is copied
   *once* into a multiprocessing.shared_memory.SharedMemory block.
2. Worker functions (_extract_worker_shm) receive only a lightweight descriptor
   tuple (shm_name, shape, dtype) instead of the array itself.  They re-attach
   the block read-only, reconstruct a NumPy view with
   np.ndarray(..., buffer=shm.buf), do their work, and detach.
3. The result written back from the worker is still serialised (serialised
   keypoints + descriptor matrix), but those are orders of magnitude smaller
   than a raw image frame.

Correctness
-----------
The stitching order is identical to parallel.py (left-to-right linear fold),
so the final panorama is pixel-equivalent to parallel.py for the same window
(same caveats about RANSAC seeding apply).

Benchmark compatibility
-----------------------
The module exposes:
  * extract_features_shm(images, process_executor) -> (kp_list, des_list, t)
    Drop-in replacement for extract_features_parallel() from parallel.py.
  * stitch_shm(input_dir, output_dir, start_idx, end_idx)
    Full stand-alone pipeline entry-point.
  * sliding_window_pipeline(input_dir, output_dir, window_size)
    Sliding-window entry-point mirroring the one in parallel.py.
  * _time_shm_window(images, process_executor, thread_executor, seed)
    Benchmark harness callable, compatible with benchmark.py PipelineSpec.
"""

import contextlib
import os
import time
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sequential import (
    match_features,
    estimate_homography,
)
from parallel import (
    load_images_parallel,
    warp_and_blend_tiling,
)


def _alloc_shm_for_images(images):
    """
    Copy each image into a fresh SharedMemory block.

    Returns:
        blocks      : list[SharedMemory]  -- caller owns these; must
                      call shm.close() + shm.unlink() when done.
        descriptors : list[tuple]         -- (shm_name, shape, dtype_str)
    """
    blocks = []
    descriptors = []

    for img in images:
        shm = SharedMemory(create=True, size=img.nbytes)
        buf_arr = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
        np.copyto(buf_arr, img)
        blocks.append(shm)
        descriptors.append((shm.name, img.shape, img.dtype.str))

    return blocks, descriptors


def _attach_shm_image(descriptor):
    """
    Attach to an existing SharedMemory block and return a NumPy view.
    Caller must call shm.close() when done (do NOT unlink from workers).
    """
    shm_name, shape, dtype_str = descriptor
    shm = SharedMemory(name=shm_name, create=False)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    return arr, shm


def _extract_worker_shm(descriptor):
    """
    MAP step: receives only (shm_name, shape, dtype_str).
    Attaches to the shared block, runs SIFT, then detaches immediately.

    Returns (shm_name, kp_serialized, des).
    The shm_name is echoed back so the main process can pair results with
    their origin block (executor.map preserves order, but explicit tagging
    makes the pairing unambiguous and facilitates future refactoring).
    """
    cv2.setNumThreads(1)
    shm_name, shape, dtype_str = descriptor

    # Attach and copy locally before detaching -- workers must not keep the
    # shared segment open, otherwise cleanup in the main process may block.
    img, shm = _attach_shm_image(descriptor)
    img_local = img.copy()
    shm.close()

    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img_local, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    kp_ser = [
        (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
        for p in kp
    ]
    return shm_name, kp_ser, des


def _deserialize_kp(kp_serialized):
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


def extract_features_shm(images, process_executor=None):
    """
    Phase 2 (MAP): parallel SIFT extraction via shared memory.

    Images are loaded into SharedMemory once; workers receive only lightweight
    descriptors (shm_name, shape, dtype) instead of the full array bytes.

    Args:
        images           : list of np.ndarray (BGR, uint8)
        process_executor : optional running ProcessPoolExecutor to reuse
                           (avoids spawn overhead when called in a loop).

    Returns:
        (keypoints_list, descriptors_list, extraction_time)
        -- same signature as extract_features_parallel() in parallel.py.
    """
    start = time.perf_counter()
    print(f"   [SHM] Allocating {len(images)} images in SharedMemory...")

    blocks, descriptors = _alloc_shm_for_images(images)

    try:
        print(
            f"   [SHM] Dispatching SIFT extraction via ProcessPool "
            f"({os.cpu_count()} cores, zero-copy IPC)..."
        )

        if process_executor is None:
            ctx = ProcessPoolExecutor(max_workers=os.cpu_count())
        else:
            ctx = contextlib.nullcontext(process_executor)

        with ctx as executor:
            results = list(executor.map(_extract_worker_shm, descriptors))

    finally:
        # Always release shared memory blocks, even on exception.
        for shm in blocks:
            shm.close()
            shm.unlink()

    keypoints_list = []
    descriptors_list = []

    for i, (shm_name, kp_ser, des) in enumerate(results):
        kp = _deserialize_kp(kp_ser)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"      - Image {i + 1}: Found {len(kp)} keypoints")

    extraction_time = time.perf_counter() - start
    return keypoints_list, descriptors_list, extraction_time


def stitch_shm(input_dir, output_dir, start_idx=0, end_idx=4):
    """
    Full shared-memory stitching pipeline on a custom image range.
    Uses the same left-to-right linear fold as parallel.py.
    """
    print(f"\nSTARTING SHARED-MEMORY PIPELINE (Range index {start_idx}:{end_idx})")
    total_start = time.perf_counter()

    images = load_images_parallel(input_dir, start_idx=start_idx, end_idx=end_idx)

    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    print("\nStarting Shared-Memory SIFT Feature Extraction...")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor, \
         ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:

        kp_list, des_list, t_extract = extract_features_shm(images, process_executor)

        print("\nStarting Iterative Stitching...")
        stitch_start = time.perf_counter()

        base_image = images[0]
        base_kp    = kp_list[0]
        base_des   = des_list[0]
        sift = cv2.SIFT_create(nfeatures=8000)

        t_match_sub = t_homo_sub = t_warp_sub = t_reext_sub = 0.0

        for i in range(1, len(images)):
            print(f"\n   - Stitching image {i + 1} onto current panorama...")

            t0 = time.perf_counter()
            matches = match_features(base_des, des_list[i])
            t_match_sub += time.perf_counter() - t0
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.")

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {i + 1}, skipping.")
                continue

            t0 = time.perf_counter()
            H = estimate_homography(base_kp, kp_list[i], matches)
            t_homo_sub += time.perf_counter() - t0
            if H is None:
                print(f"     WARNING: Homography failed for image {i + 1}, skipping.")
                continue

            t0 = time.perf_counter()
            try:
                base_image = warp_and_blend_tiling(base_image, images[i], thread_executor, H)
            except ValueError as e:
                print(f"     WARNING: {e} -- skipping this image.")
                continue
            t_warp_sub += time.perf_counter() - t0

            t0 = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            t_reext_sub += time.perf_counter() - t0
            print(
                f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                f"{len(base_kp)} keypoints re-extracted."
            )

    t_stitch   = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_shm_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}")

    print("\n" + "=" * 50)
    print(f"SHARED-MEMORY REPORT (RANGE {start_idx}:{end_idx})")
    print("=" * 50)
    print(f"SIFT Extraction Time:    {t_extract:.3f} seconds")
    print(f"Match & Warp Total:      {t_stitch:.3f} seconds")
    print(f"  - Feature Matching:    {t_match_sub:.3f} seconds")
    print(f"  - Homography Est.:     {t_homo_sub:.3f} seconds")
    print(f"  - Warp & Blend (Tile): {t_warp_sub:.3f} seconds")
    print(f"  - Feature Re-extract:  {t_reext_sub:.3f} seconds")
    print(f"Total Execution Time:    {total_time:.3f} seconds")
    print("=" * 50)


def sliding_window_pipeline(input_dir, output_dir, window_size=4):
    """Sliding-window entry-point — mirrors parallel.py's sliding_window_pipeline."""
    print(f"STARTING SHARED-MEMORY SLIDING WINDOW PIPELINE (Window Size: {window_size})")

    all_paths = sorted(
        [p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')]
    )
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    total_t_extract = total_t_match = total_t_homo = total_t_warp = total_t_reext = 0.0
    total_start = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=8000)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor, \
         ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:

        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---")

            current_images = load_images_parallel(input_dir, start_idx, end_idx)
            if len(current_images) < 2:
                print("   WARNING: Insufficient images in this window. Skipping.")
                continue

            print("Starting Shared-Memory SIFT Feature Extraction for current window...")
            kp_list, des_list, t_extract = extract_features_shm(
                current_images, process_executor
            )
            total_t_extract += t_extract

            base_image = current_images[0]
            base_kp    = kp_list[0]
            base_des   = des_list[0]

            for i in range(1, len(current_images)):
                global_img_idx = start_idx + i
                print(
                    f"\n   - Stitching image {global_img_idx}/{total_images - 1} "
                    f"onto window panorama..."
                )

                t0 = time.perf_counter()
                matches = match_features(base_des, des_list[i])
                total_t_match += time.perf_counter() - t0
                print(f"     Found {len(matches)} robust matches after Lowe's ratio test.")

                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_img_idx}, skipping.")
                    continue

                t0 = time.perf_counter()
                H = estimate_homography(base_kp, kp_list[i], matches)
                total_t_homo += time.perf_counter() - t0
                if H is None:
                    print(f"     WARNING: Homography failed for image {global_img_idx}, skipping.")
                    continue

                t0 = time.perf_counter()
                try:
                    base_image = warp_and_blend_tiling(base_image, current_images[i], thread_executor, H)
                except ValueError as e:
                    print(f"     WARNING: {e} -- skipping this image.")
                    continue
                total_t_warp += time.perf_counter() - t0

                t0 = time.perf_counter()
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                total_t_reext += time.perf_counter() - t0
                print(
                    f"     Updated window panorama: "
                    f"{base_image.shape[1]}x{base_image.shape[0]} px, "
                    f"{len(base_kp)} keypoints re-extracted."
                )

            final_file_path = (
                output_path / f"panorama_window_shm_{start_idx}_to_{end_idx - 1}.jpg"
            )
            cv2.imwrite(str(final_file_path), base_image)
            print(f"\nWindow Panorama saved successfully to: {final_file_path}")

    total_time = time.perf_counter() - total_start
    total_t_stitch = total_t_match + total_t_homo + total_t_warp + total_t_reext

    print("\n" + "=" * 50)
    print("SHARED-MEMORY PIPELINE PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Total Images Processed:    {total_images}")
    print(f"Window/Batch Size:         {window_size}")
    print("-" * 50)
    print(f"SIFT Extraction Time:      {total_t_extract:.3f} seconds")
    print(f"Match & Warp Total:        {total_t_stitch:.3f} seconds")
    print(f"  - Feature Matching:      {total_t_match:.3f} seconds")
    print(f"  - Homography Est.:       {total_t_homo:.3f} seconds")
    print(f"  - Warp & Blend (Tiling): {total_t_warp:.3f} seconds")
    print(f"  - Feature Re-extract:    {total_t_reext:.3f} seconds")
    print(f"Total Execution Time:      {total_time:.3f} seconds")
    print("=" * 50)



def main():
    input_dir  = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input_reordered not found.")
        return

    # Allow OpenCV to use all native threads for its own internal parallelism.
    cv2.setNumThreads(os.cpu_count())
    sliding_window_pipeline(input_dir, output_dir, window_size=4)


if __name__ == "__main__":
    main()
