"""
Producer-Consumer pipeline parallelism for image stitching.

Strategy:
Feature extraction (SIFT) is the ONE phase with no dependency on the
running panorama: image i's keypoints do not depend on the stitch result
of image i-1. Matching -> Homography -> Warp -> Re-extraction, on the
other hand, form a strictly sequential chain (each stitch depends on the
previous panorama).

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

Task + data parallelism combined:
The consumer's warp/blend step now uses warp_and_blend_tiling (from
parallel.py) instead of the plain single-threaded warp_and_blend: the
warped canvas is split into horizontal strips and blended concurrently on
a ThreadPoolExecutor, exactly as in parallel.py. This is orthogonal to the
producer-consumer overlap above: one is TASK parallelism (overlapping
different phases in time), the other is DATA parallelism (splitting one
operation's data across workers). Combining them means the ProcessPool
(SIFT extraction) and the ThreadPool (tile blending) can both have workers
active at the same time, competing for the same physical cores.
"""

import cProfile
import pstats
import sys

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
NUM_CORES = 8

def _extract_worker(img):
    """
    Worker function executed inside a separate Process.
    
    Isolates CPU-bound SIFT feature extraction. Disables internal OpenCV 
    multithreading to prevent CPU thrashing when multiple processes run concurrently.
    
    Args:
        img (numpy.ndarray): Source image array.
        
    Returns:
        tuple: (List of serialized keypoint tuples, numpy array of descriptors)
    """
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    # Serialize KeyPoints into primitives so they can be safely pickled via IPC
    kp_serialized = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
    return kp_serialized, des


def _deserialize_kp(kp_serialized):
    """
    Reconstructs OpenCV KeyPoint objects from serialized primitive tuples.
    """
    return [
        cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
        for pt, size, angle, response, octave, class_id in kp_serialized
    ]


def _producer(images, process_executor, out_queue, sift_seed=None):
    """
    Producer Thread: Asynchronously orchestrates SIFT feature extraction ahead of the Consumer.
    
    This thread maps extraction jobs to the background ProcessPool. As soon as a process 
    finishes an image, the result is yielded in strictly sequential order and pushed 
    into the thread-safe `out_queue`. 
    
    The `out_queue` is deliberately bounded (e.g., maxsize=2). If the fast Producer gets 
    too far ahead of the slow stitching Consumer, the `.put()` operation will naturally block, 
    creating safe backpressure to prevent RAM explosion.
    
    Args:
        images (list): Ordered list of loaded numpy images.
        process_executor (ProcessPoolExecutor): Warm process pool to offload CPU work.
        out_queue (queue.Queue): Thread-safe queue connecting the Producer to the Consumer.
    """
    # Dispatch all extraction tasks to the pool immediately
    futures = [process_executor.submit(_extract_worker, img) for img in images]

    for i, fut in enumerate(futures):
        # Wait for the next image in the sequence to finish
        kp_serialized, des = fut.result()
        kp = _deserialize_kp(kp_serialized)

        # Push the payload to the consumer. Blocks if the queue is full (backpressure mechanism).
        out_queue.put((i, images[i], kp, des))

    # Inject a termination signal into the queue to gracefully kill the consumer loop
    out_queue.put(_SENTINEL)


def stitch_producer_consumer(input_dir, output_dir, start_idx=0, end_idx=4,
                              queue_depth=2, num_extract_workers=None, num_blend_workers=None):
    """
    Executes a hybrid Task-and-Data Parallel stitching pipeline on a custom image range.
    
    Architecture:
    1. TASK PARALLELISM: A Producer thread offloads SIFT extraction to a ProcessPool 
       while the Consumer (main thread) simultaneously handles blending.
    2. DATA PARALLELISM: During the Consumer's blending phase, `warp_and_blend_tiling` 
       distributes the canvas rendering across a ThreadPool.
       
    Args:
        input_dir (str): Directory containing source images.
        output_dir (str): Output destination.
        start_idx (int): Inclusive starting frame.
        end_idx (int): Exclusive ending frame.
        queue_depth (int): Maximum items allowed in the queue. Controls memory footprint 
                           and regulates the backpressure on the Producer.
    """
    print(f"\nSTARTING PRODUCER-CONSUMER PIPELINE (Range index {start_idx}:{end_idx})", file=sys.stderr)
    total_start = time.perf_counter()

    images = load_images(input_dir, start_idx=start_idx, end_idx=end_idx)
    if len(images) < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    num_extract_workers = num_extract_workers or NUM_CORES
    num_blend_workers = num_blend_workers or NUM_CORES

    # Thread-safe buffer enforcing backpressure between extraction and stitching
    result_queue = queue.Queue(maxsize=queue_depth)

    t_match_sub = t_homo_sub = t_warp_sub = t_reext_sub = 0.0

    # Ensure worker pools are warm and cleanly terminated upon exit
    with ProcessPoolExecutor(max_workers=num_extract_workers) as process_executor, \
         ThreadPoolExecutor(max_workers=num_blend_workers) as thread_executor:
        t_extract_start = time.perf_counter()

        # Launch the asynchronous Producer thread
        producer_thread = threading.Thread(
            target=_producer,
            args=(images, process_executor, result_queue),
            daemon=True,
        )
        producer_thread.start()

        # Consumer: sequential stitching chain, fed by the queue
        base_image = base_kp = base_des = None
        sift = cv2.SIFT_create(nfeatures=8000)
        stitch_start = time.perf_counter()

        while True:

            # Block until the Producer drops the next extracted frame into the queue
            item = result_queue.get()

            # Check for termination signal
            if item is _SENTINEL:
                break
            idx, img, kp, des = item

            if base_image is None:
                # Seed the panorama with the first image and its features
                base_image, base_kp, base_des = img, kp, des
                print(f"   - Image {idx + 1}: seeded panorama with {len(kp)} keypoints", file=sys.stderr)
                continue

            print(f"\n   - Stitching image {idx + 1} onto current panorama "
                  f"(overlapped with producer extraction)...", file=sys.stderr)

            # Match features between the current panorama and the new image
            t0 = time.perf_counter()
            matches = match_features(base_des, des)
            t_match_sub += time.perf_counter() - t0
            print(f"     Found {len(matches)} robust matches after Lowe's ratio test.", file=sys.stderr)

            if len(matches) < 4:
                print(f"     WARNING: Too few matches for image {idx + 1}, skipping.", file=sys.stderr)
                continue
            
            # Estimate the homography matrix to align the new image with the current panorama
            t0 = time.perf_counter()
            H = estimate_homography(base_kp, kp, matches)
            t_homo_sub += time.perf_counter() - t0
            if H is None:
                print(f"     WARNING: Homography failed for image {idx + 1}, skipping.", file=sys.stderr)
                continue
            
            # Apply the homography to warp the new image and blend it with the current panorama
            t0 = time.perf_counter()
            try:
                # Offload horizontal slice blending to the ThreadPool
                base_image = warp_and_blend_tiling(base_image, img, thread_executor, H)
            except ValueError as e:
                print(f"     WARNING: {e} -- skipping this image.", file=sys.stderr)
                continue
            t_warp_sub += time.perf_counter() - t0

            # Re-extract features from the updated panorama
            t0 = time.perf_counter()
            gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
            base_kp, base_des = sift.detectAndCompute(gray_base, None)
            t_reext_sub += time.perf_counter() - t0
            print(f"     Updated panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                  f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

        # Synchronize and clean up the producer thread
        producer_thread.join()
        t_extract_total = time.perf_counter() - t_extract_start

    total_time = time.perf_counter() - total_start

    # Save the final stitched panorama to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file_path = output_path / f"final_panorama_pc_{start_idx}_to_{end_idx}.jpg"
    cv2.imwrite(str(final_file_path), base_image)
    print(f"\nPanorama saved successfully to: {final_file_path}", file=sys.stderr)

    # Log execution timings
    print("\n" + "=" * 50, file=sys.stderr)
    print(f"PRODUCER-CONSUMER REPORT (RANGE {start_idx}:{end_idx})", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Overlapped Extraction+Stitch wall time: {t_extract_total:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Matching:   {t_match_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Homography Est.:    {t_homo_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Warp & Blend:       {t_warp_sub:.3f} seconds", file=sys.stderr)
    print(f"  - Feature Re-extract: {t_reext_sub:.3f} seconds", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def sliding_window_pipeline(input_dir, output_dir, window_size=4, queue_depth=2):
    """
    Executes the Producer-Consumer pattern across a large dataset using localized sliding windows.
    
    Optimized to reuse the heavy process and thread pools continuously across the entire 
    job, eliminating spin-up/spin-down latency between window transitions.
    
    Args:
        input_dir (str): Folder path containing input sequence.
        output_dir (str): Folder path for output windows.
        window_size (int): Image capacity per stitched panorama segment.
        queue_depth (int): Size of the inter-thread communication queue.
    """

    print(f"STARTING PRODUCER-CONSUMER SLIDING WINDOW PIPELINE (Window Size: {window_size})", file=sys.stderr)

    all_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required for stitching.", file=sys.stderr)
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()

    # Single ProcessPoolExecutor + ThreadPoolExecutor, both reused across
    # all windows to avoid per-window spawn overhead
    with ProcessPoolExecutor(max_workers=NUM_CORES) as process_executor, \
         ThreadPoolExecutor(max_workers=NUM_CORES) as thread_executor:
        for start_idx in range(0, total_images, window_size):
            end_idx = min(start_idx + window_size, total_images)
            print(f"\n--- Processing window: Images {start_idx} to {end_idx - 1} ---", file=sys.stderr)

            images = load_images(input_dir, start_idx, end_idx)
            if len(images) < 2:
                print("   WARNING: Not enough images in this window to stitch. Skipping.", file=sys.stderr)
                continue
            
            # Reset the synchronized queue and launch a fresh Producer thread for this specific window
            result_queue = queue.Queue(maxsize=queue_depth)
            producer_thread = threading.Thread(
                target=_producer,
                args=(images, process_executor, result_queue),
                daemon=True,
            )
            producer_thread.start()

            base_image = base_kp = base_des = None
            sift = cv2.SIFT_create(nfeatures=8000)
            
            # Consume the queue synchronously
            while True:
                item = result_queue.get()
                if item is _SENTINEL:
                    break
                idx, img, kp, des = item

                if base_image is None:
                    base_image, base_kp, base_des = img, kp, des
                    continue

                global_idx = start_idx + idx
                print(f"   - Stitching image {global_idx}/{total_images - 1} onto window panorama...", file=sys.stderr)

                matches = match_features(base_des, des)
                if len(matches) < 4:
                    print(f"     WARNING: Too few matches for image {global_idx}, skipping.", file=sys.stderr)
                    continue

                H = estimate_homography(base_kp, kp, matches)
                if H is None:
                    print(f"     WARNING: Homography failed for image {global_idx}, skipping.", file=sys.stderr)
                    continue

                try:
                    # Tile-based thread blending
                    base_image = warp_and_blend_tiling(base_image, img, thread_executor, H)
                except ValueError as e:
                    print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
                    continue
                gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
                base_kp, base_des = sift.detectAndCompute(gray_base, None)
                print(f"     Updated window panorama: {base_image.shape[1]}x{base_image.shape[0]} px, "
                      f"{len(base_kp)} keypoints re-extracted.", file=sys.stderr)

            producer_thread.join()

            final_file_path = output_path / f"panorama_window_pc_{start_idx}_to_{end_idx - 1}.jpg"
            cv2.imwrite(str(final_file_path), base_image)
            print(f"Window Panorama saved successfully to: {final_file_path}", file=sys.stderr)

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 50, file=sys.stderr)
    print("PRODUCER-CONSUMER SLIDING WINDOW REPORT", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"Total Images Processed: {total_images}", file=sys.stderr)
    print(f"Window/Batch Size:      {window_size}", file=sys.stderr)
    print(f"Total Execution Time:   {total_time:.3f} seconds", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


def main():
    input_dir = "data/input_reordered"
    output_dir = "data/output"

    if not Path(input_dir).exists():
        print("ERROR: Directory data/input not found.", file=sys.stderr)
        return

    profiler = cProfile.Profile()
    profiler.enable()

    sliding_window_pipeline(input_dir, output_dir, window_size=4)

    profiler.disable()

    output_file = Path("profiling_results/producer_consumer_profiling")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("tottime")
        stats.print_stats()

    print(f"\nProfiling results saved to: {output_file}", file=sys.stderr)

if __name__ == "__main__":
    main()