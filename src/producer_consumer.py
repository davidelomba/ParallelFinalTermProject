"""
producer_consumer.py
=====================
Producer-Consumer pipeline parallelism for image stitching.

Strategy
--------
Feature extraction (SIFT) is the ONE phase with no dependency on the
running panorama: image i's keypoints do not depend on the stitch result
of image i-1. Matching -> Homography -> Warp -> Re-extraction, on the
other hand, form a strictly sequential chain (each stitch depends on the
previous panorama), exactly like in sequential.py and parallel.py.

This module overlaps the two: a PRODUCER thread continuously submits SIFT
extraction jobs to a ProcessPoolExecutor and pushes completed
(index, image, keypoints, descriptors) tuples onto a bounded queue, in
image order. A CONSUMER (main thread) pulls from the queue and performs
the sequential match -> homography -> warp -> reextract chain.

While the consumer is busy warping/blending image i onto the panorama,
the producer has already dispatched extraction for image i+1 (and possibly
further ahead, depending on queue_depth) to the process pool, so that work
overlaps with the consumer's CPU-bound stitching instead of waiting for it
to finish first.

Task + data parallelism combined
----------------------------------
The consumer's warp/blend step now uses warp_and_blend_tiling (from
parallel.py) instead of the plain single-threaded warp_and_blend: the
warped canvas is split into horizontal strips and blended concurrently on
a ThreadPoolExecutor, exactly as in parallel.py. This is orthogonal to the
producer-consumer overlap above -- one is TASK parallelism (overlapping
different phases in time), the other is DATA parallelism (splitting one
operation's data across workers). Combining them means the ProcessPool
(SIFT extraction) and the ThreadPool (tile blending) can both have workers
active at the same time, competing for the same physical cores; whether
the net effect is faster than either technique alone is an empirical
question for the benchmark, not something to assume.

Note on correctness
--------------------
The stitching ORDER is identical to sequential.py (left-to-right linear
fold), so the final panorama should match sequential.py's output for the
same window (subject to the same RANSAC seeding caveats discussed for the
other pipelines). Only the *scheduling* of feature extraction changes.
"""

import cv2
import os
import queue
import threading
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from sequential import (
    load_images,
    match_features,
    estimate_homography,
)
from parallel import warp_and_blend_tiling

_SENTINEL = None


def _extract_worker(img):
    """Runs in a worker PROCESS. Returns serialized keypoints + descriptors."""
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
    return kp_serialized, des


def _deserialize_kp(kp_serialized):
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


def _producer(images, process_executor, out_queue, sift_seed=None):
    """
    Producer thread: submits extraction jobs to the process pool and
    pushes completed results onto out_queue strictly in image order, as
    soon as each one is ready. The out_queue's maxsize provides backpressure
    (the producer blocks on .put() if the consumer falls behind).
    """
    futures = [process_executor.submit(_extract_worker, img) for img in images]
    for i, fut in enumerate(futures):
        kp_serialized, des = fut.result()
        kp = _deserialize_kp(kp_serialized)
        out_queue.put((i, images[i], kp, des))
    out_queue.put(_SENTINEL)


def stitch_producer_consumer(input_dir, output_dir, start_idx=0, end_idx=4,
                              queue_depth=2, num_extract_workers=None, num_blend_workers=None):
    """
    Executes the producer-consumer stitching pipeline on a custom range
    of images. Blending uses warp_and_blend_tiling (data parallelism)
    while extraction overlaps with stitching (task parallelism) -- see
    module docstring.
    """
    print(f"\nSTARTING PRODUCER-CONSUMER PIPELINE (Range index {start_idx}:{end_idx})")
    total_start = time.perf_counter()

    images = load_images(input_dir, start_idx=start_idx, end_idx=end_idx)
    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    num_extract_workers = num_extract_workers or os.cpu_count()
    num_blend_workers = num_blend_workers or os.cpu_count()
    result_queue = queue.Queue(maxsize=queue_depth)

    t_match_sub = t_homo_sub = t_warp_sub = t_reext_sub = 0.0

    with ProcessPoolExecutor(max_workers=num_extract_workers) as process_executor, \
         ThreadPoolExecutor(max_workers=num_blend_workers) as thread_executor:
        t_extract_start = time.perf_counter()

        producer_thread = threading.Thread(
            target=_producer,
            args=(images, process_executor, result_queue),
            daemon=True,
        )
        producer_thread.start()

        # --- Consumer: sequential stitching chain, fed by the queue ---
        base_image = base_kp = base_des = None
        sift = cv2.SIFT_create(nfeatures=8000)
        stitch_start = time.perf_counter()

        while True:
            item = result_queue.get()
            if item is _SENTINEL:
                break
            idx, img, kp, des = item

            if base_image is None:
                # First image just seeds the panorama
                base_image, base_kp, base_des = img, kp, des
                print(f"   - Image {idx + 1}: seeded panorama with {len(kp)} keypoints")
                continue

            print(f"\n   - Stitching image {idx + 1} onto current panorama "
                  f"(overlapped with producer extraction)...")

            t0 = time.perf_counter()
            matches = match_features(base_des, des)
            t_match_sub += time.perf_counter() - t0
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.")

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {idx + 1}, skipping.")
                continue

            t0 = time.perf_counter()
            H = estimate_homography(base_kp, kp, matches)
            t_homo_sub += time.perf_counter() - t0
            if H is None:
                print(f"     WARNING: Homography failed for image {idx + 1}, skipping.")
                continue

            t0 = time.perf_counter()
            base_image = warp_and_blend_tiling(base_image, img, thread_executor, H, num_workers=num_blend_workers)
            t_warp_sub += time.perf_counter() - t0

            t0 = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            t_reext_sub += time.perf_counter() - t0
            print(f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                  f"{len(base_kp)} keypoints re-extracted.")

        producer_thread.join()
        t_extract_total = time.perf_counter() - t_extract_start

    t_stitch = time.perf_counter() - stitch_start
    total_time = time.perf_counter() - total_start

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_pc_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}")

    print("\n" + "=" * 50)
    print(f"PRODUCER-CONSUMER REPORT (RANGE {start_idx}:{end_idx})")
    print("=" * 50)
    print(f"Overlapped Extraction+Stitch wall time: {t_extract_total:.3f} seconds")
    print(f"  - Feature Matching:   {t_match_sub:.3f} seconds")
    print(f"  - Homography Est.:    {t_homo_sub:.3f} seconds")
    print(f"  - Warp & Blend:       {t_warp_sub:.3f} seconds")
    print(f"  - Feature Re-extract: {t_reext_sub:.3f} seconds")
    print(f"Total Execution Time:   {total_time:.3f} seconds")
    print("=" * 50)


def sliding_window_pipeline(input_dir, output_dir, window_size=4, queue_depth=2):
    print(f"STARTING PRODUCER-CONSUMER SLIDING WINDOW PIPELINE (Window Size: {window_size})")

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    # Single ProcessPoolExecutor + ThreadPoolExecutor, both reused across
    # all windows to avoid per-window spawn overhead (same pattern used
    # in parallel.py).
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor, \
         ThreadPoolExecutor(max_workers=os.cpu_count()) as thread_executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---")

            images = load_images(input_dir, start_idx, end_idx)
            if len(images) < 2:
                print("   WARNING: Not enough images in this window to stitch. Skipping.")
                continue

            result_queue = queue.Queue(maxsize=queue_depth)
            producer_thread = threading.Thread(
                target=_producer,
                args=(images, process_executor, result_queue),
                daemon=True,
            )
            producer_thread.start()

            base_image = base_kp = base_des = None
            sift = cv2.SIFT_create(nfeatures=8000)

            while True:
                item = result_queue.get()
                if item is _SENTINEL:
                    break
                idx, img, kp, des = item

                if base_image is None:
                    base_image, base_kp, base_des = img, kp, des
                    continue

                global_idx = start_idx + idx
                print(f"   - Stitching image {global_idx}/{total_images - 1} onto window panorama...")

                matches = match_features(base_des, des)
                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_idx}, skipping.")
                    continue

                H = estimate_homography(base_kp, kp, matches)
                if H is None:
                    print(f"     WARNING: Homography failed for image {global_idx}, skipping.")
                    continue

                base_image = warp_and_blend_tiling(base_image, img, thread_executor, H, num_workers=os.cpu_count())
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                print(f"     Updated window panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                      f"{len(base_kp)} keypoints re-extracted.")

            producer_thread.join()

            final_file_path = output_path / f"panorama_window_pc_{start_idx}_to_{end_idx - 1}.jpg"
            cv2.imwrite(str(final_file_path), base_image)
            print(f"Window Panorama saved successfully to: {final_file_path}")

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 50)
    print("PRODUCER-CONSUMER SLIDING WINDOW REPORT")
    print("=" * 50)
    print(f"Total Images Processed: {total_images}")
    print(f"Window/Batch Size:      {window_size}")
    print(f"Total Execution Time:   {total_time:.3f} seconds")
    print("=" * 50)


def main():
    input_dir = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input not found.")
        return

    sliding_window_pipeline(input_dir, output_dir, window_size=4)


if __name__ == "__main__":
    main()