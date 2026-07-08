"""
Performance and correctness comparison across image stitching pipelines
(sequential, parallel, producer_consumer, mapreduce, joblib, shared_memory).

Usage:
    python benchmark.py

Requirements:
    - sequential.py, parallel.py, producer_consumer.py, mapreduce.py,
      joblib_pipeline.py, shared_memory_pipeline.py in the same directory
    - Directory data/input_reordered with at least 2 .jpg/.png images

Design:
Each pipeline is described by a PipelineSpec:
    - name               : display name / CSV tag
    - run                : callable(images, resources, seed=42) -> dict
                            with per-phase timings + the final "panorama"
    - needs_process_pool / needs_thread_pool :
                            executors the pipeline needs; run_benchmark()
                            opens the union required by all selected
                            pipelines, once per window, reused across
                            warm-up + timed runs.

run_benchmark() takes pipelines: list[PipelineSpec]. pipelines[0] is the
BASELINE; every other entry is a candidate compared against it (speedup,
95% CI, pixel-level correctness check).


Some pipelines can't report a clean, standalone duration for every phase
(extract/match/homo/warp/reext/total). Those phases return None.
Reports and CSV export both detect None and show "n/a" instead of doing arithmetic on it.
Known cases in this file:
    - producer_consumer: "extract" is None (overlapped with the rest of
      the pipeline, no standalone duration). Compare "total" instead.
    - mapreduce: "match"/"homo"/"warp"/"reext" are None (executed
      atomically per pair inside the worker process, not separately
      timeable from the main thread). Compare "extract" + "total".

Canvas-explosion guard:
warp_and_blend() / warp_and_blend_tiling() raise ValueError when a
degenerate homography would allocate an oversized canvas. Every call
site here catches it and skips that single image instead of crashing
the whole benchmark run.

Partial-failure resilience:
Both warm-up passes and timed runs are wrapped in try/except: a failure
excludes that pipeline from the current window (or, if it's the
baseline, aborts the rest of that window) without losing results already
collected for other windows/pipelines.
"""

import contextlib
import csv
import gc
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from sequential import (
    load_images,
    extract_features,
    match_features as match_features_seq,
    estimate_homography as estimate_homography_seq,
    warp_and_blend,
)
from parallel import (
    extract_features_parallel,
    match_features as match_features_par,
    estimate_homography as estimate_homography_par,
    warp_and_blend_tiling,
)
from producer_consumer import _producer as _pc_producer, _SENTINEL as _PC_SENTINEL
from mapreduce import _extract_worker, _merge_pair_worker
from joblib_pipeline import extract_features_joblib
from shared_memory_pipeline import extract_features_shm


INPUT_DIR   = "data/input_reordered"
OUTPUT_DIR  = "data/output/benchmark"
RESULTS_DIR = "results"
WINDOW_SIZE = 4
N_RUNS      = 3     # timed repetitions per pipeline
CONFIDENCE  = 0.95  # confidence level for intervals (t-distribution)

NUM_CORES = 8


NA_LABEL = "n/a"    # shown whenever a phase isn't separately measurable for a pipeline

# Two-tailed Student t critical values for alpha=0.05, indexed by n-1
_T_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
    5: 2.571,  6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228,
}


def t_critical(n: int) -> float:
    """Two-tailed 95% t critical value for n-1 degrees of freedom"""
    return _T_CRITICAL.get(n - 1, 2.0)  # conservative fallback


def _is_measurable(times: list) -> bool:
    """
    True unless every value is None (non-measurable phase for this pipeline).
    """
    return all(t is not None for t in times)


def confidence_interval(times: list[float]) -> tuple[float, float, float]:
    """
    It calculates the mean, std, and 95% CI.


    Returns:
        (mean, std, margin)
    """
    n    = len(times)
    mean = sum(times) / n
    std  = math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1)) if n > 1 else 0.0
    margin = t_critical(n) * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, margin


def speedup_ci(
    baseline_times: list[float],
    candidate_times: list[float],
) -> tuple[float, float, float]:
    """
    Speedup S = T_baseline / T_candidate with a 95% CI via error
    propagation (delta method):

        sigma_S / S ~= sqrt( (sigma_base/mu_base)^2 + (sigma_cand/mu_cand)^2 )

    The two samples may have different sizes (e.g. some runs failed and
    were skipped); each side uses its own n for its variance term, and
    the smaller n is used for the t-critical lookup.

    Returns:
        (speedup, ci_lower, ci_upper)
    """
    mu_base, sigma_base, _ = confidence_interval(baseline_times)
    mu_cand, sigma_cand, _ = confidence_interval(candidate_times)

    if mu_cand == 0:
        return 0.0, 0.0, 0.0

    S = mu_base / mu_cand
    n_base = len(baseline_times)
    n_cand = len(candidate_times)
    tc = t_critical(min(n_base, n_cand))
    cv_base = sigma_base / mu_base if mu_base > 0 else 0
    cv_cand = sigma_cand / mu_cand if mu_cand > 0 else 0

    rel_margin = tc * math.sqrt(cv_base**2 / n_base + cv_cand**2 / n_cand)
    return S, S * (1 - rel_margin), S * (1 + rel_margin)



def _time_sequential_window(images: list, seed: int = 42) -> dict:
    """Time the sequential pipeline on preloaded images."""
    cv2.setRNGSeed(seed)
    sift = cv2.SIFT_create(nfeatures=8000)

    t0 = time.perf_counter()
    kp_list, des_list, t_extract = extract_features(images)

    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]
    t_match = t_homo = t_warp = t_reext = 0.0

    for i in range(1, len(images)):
        t1 = time.perf_counter()
        matches = match_features_seq(base_des, des_list[i])
        t_match += time.perf_counter() - t1

        if len(matches) < 4:
            continue

        t1 = time.perf_counter()
        H = estimate_homography_seq(base_kp, kp_list[i], matches)
        t_homo += time.perf_counter() - t1

        if H is None:
            continue

        t1 = time.perf_counter()
        try:
            base_image = warp_and_blend(base_image, images[i], H)
        except ValueError as e:
            print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
            continue
        t_warp += time.perf_counter() - t1

        t1 = time.perf_counter()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray, None)
        t_reext += time.perf_counter() - t1

    return {
        "extract" : t_extract,
        "match"   : t_match,
        "homo"    : t_homo,
        "warp"    : t_warp,
        "reext"   : t_reext,
        "total"   : time.perf_counter() - t0,
        "panorama": base_image,
    }


def _time_parallel_window(
    images: list,
    process_executor: ProcessPoolExecutor,
    thread_executor: ThreadPoolExecutor,
    seed: int = 42,
) -> dict:
    """
    Time the parallel pipeline (ProcessPool extraction + tiled blend) on
    preloaded images. Executors are supplied by the caller so spawn
    overhead is excluded from the timed section and shared across runs.
    """
    cv2.setRNGSeed(seed)
    sift = cv2.SIFT_create(nfeatures=8000)

    t0 = time.perf_counter()
    kp_list, des_list, t_extract = extract_features_parallel(images, process_executor=process_executor)

    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]
    t_match = t_homo = t_warp = t_reext = 0.0

    for i in range(1, len(images)):
        t1 = time.perf_counter()
        matches = match_features_par(base_des, des_list[i])
        t_match += time.perf_counter() - t1

        if len(matches) < 4:
            continue

        t1 = time.perf_counter()
        H = estimate_homography_par(base_kp, kp_list[i], matches)
        t_homo += time.perf_counter() - t1

        if H is None:
            continue

        t1 = time.perf_counter()
        try:
            base_image = warp_and_blend_tiling(base_image, images[i], thread_executor, H, num_workers=NUM_CORES)
        except ValueError as e:
            print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
            continue
        t_warp += time.perf_counter() - t1

        t1 = time.perf_counter()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray, None)
        t_reext += time.perf_counter() - t1

    return {
        "extract" : t_extract,
        "match"   : t_match,
        "homo"    : t_homo,
        "warp"    : t_warp,
        "reext"   : t_reext,
        "total"   : time.perf_counter() - t0,
        "panorama": base_image,
    }


def _time_producer_consumer_window(
    images: list,
    process_executor: ProcessPoolExecutor,
    thread_executor: ThreadPoolExecutor,
    seed: int = 42,
    queue_depth: int = 2,
) -> dict:
    """
    Time the producer-consumer pipeline: extraction runs on a background
    thread overlapped with the match/homography/warp/reext chain (task
    parallelism), blending is tiled across thread_executor (data
    parallelism).

    "extract" is always None: it has no standalone duration by design
    (overlapped with the rest). Compare "total" instead.
    """
    cv2.setRNGSeed(seed)
    sift = cv2.SIFT_create(nfeatures=8000)

    result_queue: "queue.Queue" = queue.Queue(maxsize=queue_depth)
    t0 = time.perf_counter()

    producer_thread = threading.Thread(
        target=_pc_producer,
        args=(images, process_executor, result_queue),
        daemon=True,
    )
    producer_thread.start()

    base_image = base_kp = base_des = None
    t_match = t_homo = t_warp = t_reext = 0.0

    while True:
        item = result_queue.get()
        if item is _PC_SENTINEL:
            break
        idx, img, kp, des = item

        if base_image is None:
            base_image, base_kp, base_des = img, kp, des
            continue

        t1 = time.perf_counter()
        matches = match_features_seq(base_des, des)
        t_match += time.perf_counter() - t1

        if len(matches) < 4:
            continue

        t1 = time.perf_counter()
        H = estimate_homography_seq(base_kp, kp, matches)
        t_homo += time.perf_counter() - t1
        if H is None:
            continue

        t1 = time.perf_counter()
        try:
            base_image = warp_and_blend_tiling(base_image, img, thread_executor, H, num_workers=NUM_CORES)
        except ValueError as e:
            print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
            continue
        t_warp += time.perf_counter() - t1

        t1 = time.perf_counter()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray, None)
        t_reext += time.perf_counter() - t1

    producer_thread.join()

    return {
        "extract" : None,
        "match"   : t_match,
        "homo"    : t_homo,
        "warp"    : t_warp,
        "reext"   : t_reext,
        "total"   : time.perf_counter() - t0,
        "panorama": base_image,
    }


def _time_mapreduce_window(
    images: list,
    process_executor: ProcessPoolExecutor,
    seed: int = 42,
) -> dict:
    """
    Time the mapreduce pipeline: parallel MAP (extraction), then a
    pairwise-tree REDUCE (merge) on process_executor.

    "match"/"homo"/"warp"/"reext" are always None: each pair's full
    match->homography->warp->reext chain runs atomically inside one
    worker call and isn't separately timeable from here. Compare
    "extract" + "total".
    """
    cv2.setRNGSeed(seed)
    t0 = time.perf_counter()

    t_map_start = time.perf_counter()
    nodes = list(process_executor.map(_extract_worker, images))
    t_extract = time.perf_counter() - t_map_start

    while len(nodes) > 1:
        n_pairs = len(nodes) // 2
        odd_one_out = nodes[-1] if len(nodes) % 2 == 1 else None

        pairs = [(nodes[2 * i], nodes[2 * i + 1]) for i in range(n_pairs)]
        merged = list(process_executor.map(_merge_pair_worker, pairs))

        if odd_one_out is not None:
            merged.append(odd_one_out)

        nodes = merged

    final_image = nodes[0][0]

    return {
        "extract" : t_extract,
        "match"   : None,
        "homo"    : None,
        "warp"    : None,
        "reext"   : None,
        "total"   : time.perf_counter() - t0,
        "panorama": final_image,
    }


def _time_joblib_window(images, thread_executor, seed=42):
    """
    Time the Joblib pipeline: features extraction runs via joblib (loky backend)
    multiprocessing, followed by a data-parallel tiled blend across
    thread_executor on preloaded images.
    """

    cv2.setRNGSeed(seed)
    sift = cv2.SIFT_create(nfeatures=8000)

    t0 = time.perf_counter()
    kp_list, des_list, t_extract = extract_features_joblib(images)

    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]
    t_match = t_homo = t_warp = t_reext = 0.0

    for i in range(1, len(images)):
        t1 = time.perf_counter()
        matches = match_features_seq(base_des, des_list[i])
        t_match += time.perf_counter() - t1

        if len(matches) < 4:
            continue

        t1 = time.perf_counter()
        H = estimate_homography_seq(base_kp, kp_list[i], matches)
        t_homo += time.perf_counter() - t1

        if H is None:
            continue

        t1 = time.perf_counter()
        try:
            base_image = warp_and_blend_tiling(
                base_image, images[i], thread_executor, H, num_workers=NUM_CORES
            )
        except ValueError as e:
            print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
            continue
        t_warp += time.perf_counter() - t1

        t1 = time.perf_counter()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray, None)
        t_reext += time.perf_counter() - t1

    return {
        "extract" : t_extract,
        "match"   : t_match,
        "homo"    : t_homo,
        "warp"    : t_warp,
        "reext"   : t_reext,
        "total"   : time.perf_counter() - t0,
        "panorama": base_image,
    }


def _time_shm_window(images, process_executor, thread_executor, seed=42):
    """
    Time the Shared Memory pipeline: features extraction runs via multiprocessing
    with zero-copy shared memory, followed by a data-parallel tiled blend across
    thread_executor on preloaded images.
    """
    cv2.setRNGSeed(seed)
    sift = cv2.SIFT_create(nfeatures=8000)

    t0 = time.perf_counter()
    kp_list, des_list, t_extract = extract_features_shm(images, process_executor)

    base_image = images[0]
    base_kp    = kp_list[0]
    base_des   = des_list[0]
    t_match = t_homo = t_warp = t_reext = 0.0

    for i in range(1, len(images)):
        t1 = time.perf_counter()
        matches = match_features_seq(base_des, des_list[i])
        t_match += time.perf_counter() - t1

        if len(matches) < 4:
            continue

        t1 = time.perf_counter()
        H = estimate_homography_seq(base_kp, kp_list[i], matches)
        t_homo += time.perf_counter() - t1

        if H is None:
            continue

        t1 = time.perf_counter()
        try:
            base_image = warp_and_blend_tiling(
                base_image, images[i], thread_executor, H, num_workers=NUM_CORES
            )
        except ValueError as e:
            print(f"     WARNING: {e} skipping this image.", file=sys.stderr)
            continue
        t_warp += time.perf_counter() - t1

        t1 = time.perf_counter()
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_kp, base_des = sift.detectAndCompute(gray, None)
        t_reext += time.perf_counter() - t1

    return {
        "extract" : t_extract,
        "match"   : t_match,
        "homo"    : t_homo,
        "warp"    : t_warp,
        "reext"   : t_reext,
        "total"   : time.perf_counter() - t0,
        "panorama": base_image,
    }


@dataclass
class PipelineSpec:
    """Describes one pipeline for run_benchmark(): name, runner, executor needs."""
    name: str
    run: Callable[..., dict]
    needs_process_pool: bool = True
    needs_thread_pool: bool = False

# Adapter for the sequential pipeline (baseline, requires no pools)
def _run_sequential(images, resources, seed: int = 42) -> dict:
    return _time_sequential_window(images, seed=seed)

# Adapter for the standard parallel pipeline (uses processes for SIFT and threads for blending)
def _run_parallel(images, resources, seed: int = 42) -> dict:
    return _time_parallel_window(images, resources["process_executor"], resources["thread_executor"], seed=seed)

# Closure to pre-configure queue depth without altering the standard benchmark signature
def _make_producer_consumer_runner(queue_depth: int = 2) -> Callable[..., dict]:
    def _run(images, resources, seed: int = 42) -> dict:
        return _time_producer_consumer_window(
            images, resources["process_executor"], resources["thread_executor"],
            seed=seed, queue_depth=queue_depth,
        )
    return _run

# Adapter for the mapreduce pipeline (requires only the ProcessPool for tree reduction)
def _run_mapreduce(images, resources, seed: int = 42) -> dict:
    return _time_mapreduce_window(images, resources["process_executor"], seed=seed)

# Adapter for the joblib pipeline (automatic loky extraction and thread-based blending)
def _run_joblib(images, resources, seed: int = 42) -> dict:
    return _time_joblib_window(images, resources["thread_executor"], seed=seed)

# Adapter for the shared memory pipeline (optimized zero-copy multiprocessing + ThreadPool)
def _run_shm(images, resources, seed: int = 42) -> dict:
    return _time_shm_window(images, resources["process_executor"], resources["thread_executor"], seed=seed)


SEQUENTIAL_SPEC = PipelineSpec(
    name="sequential",
    run=_run_sequential,
    needs_process_pool=False,
    needs_thread_pool=False,
)
PARALLEL_SPEC = PipelineSpec(
    name="parallel",
    run=_run_parallel,
    needs_process_pool=True,
    needs_thread_pool=True,
)
PRODUCER_CONSUMER_SPEC = PipelineSpec(
    name="producer_consumer",
    run=_make_producer_consumer_runner(queue_depth=2),
    needs_process_pool=True,
    needs_thread_pool=True,
)
MAPREDUCE_SPEC = PipelineSpec(
    name="mapreduce",
    run=_run_mapreduce,
    needs_process_pool=True,
    needs_thread_pool=False,
)
JOBLIB_SPEC = PipelineSpec(
    name="joblib",
    run=_run_joblib,
    needs_process_pool=False,
    needs_thread_pool=True,
)
SHM_SPEC = PipelineSpec(
    name="shared_memory",
    run=_run_shm,
    needs_process_pool=True,
    needs_thread_pool=True,
)


def _compare_panoramas(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Pixel-by-pixel comparison of two panoramas. Returns a metrics dict."""

    # Shape validation: pixel-wise subtraction is impossible if dimensions differ
    shape_ok = img_a.shape == img_b.shape
    print(f"    Shape  a: {img_a.shape}  b: {img_b.shape}  ->  "
          f"{'match' if shape_ok else 'MISMATCH'}", file=sys.stderr)

    if not shape_ok:
        print("      Shape mismatch: pixel comparison not possible.", file=sys.stderr)
        return {"shape_ok": False}

    # Absolute pixel difference (cast to int32 prevents uint8 underflow)
    diff = np.abs(img_a.astype(np.int32) - img_b.astype(np.int32))
    max_diff      = int(diff.max())
    mean_diff     = float(diff.mean())
    identical_pct = float((diff == 0).mean()) * 100

    print(f"    Max pixel diff       : {max_diff}", file=sys.stderr)
    print(f"    Mean pixel diff      : {mean_diff:.6f}", file=sys.stderr)
    print(f"    Identical pixels     : {identical_pct:.2f}%", file=sys.stderr)


    # PSNR (Peak Signal-to-Noise Ratio): standard image quality metric
    # A PSNR > 40 dB means the differences are imperceptible to the human eye
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    if mse == 0:
        psnr_value = None
        print("    PSNR                 : inf  (bit-identical images)", file=sys.stderr)
    else:
        psnr_value = 10 * math.log10(255.0 ** 2 / mse)
        tag = ">40 dB (visually identical)" if psnr_value > 40 else "perceptible differences"
        print(f"    PSNR                 : {psnr_value:.2f} dB  {tag}", file=sys.stderr)


    # OpenCV uses BGR format
    per_channel = []
    for ch, name in enumerate(["Blue", "Green", "Red"]):
        ch_diff = diff[:, :, ch]
        ch_max  = int(ch_diff.max())
        ch_mean = float(ch_diff.mean())
        per_channel.append((name, ch_max, ch_mean))
        print(f"    Channel {name:<5}        : max={ch_max:3d}  mean={ch_mean:.4f}", file=sys.stderr)


    # Verdict: accounts for tiny floating-point math variations caused by different execution orders in multi-threading/processing
    if max_diff == 0:
        verdict = "Bit-identical"
    elif max_diff <= 1:
        verdict = "Equivalent (float32 rounding, max 1 LSB)"
    elif psnr_value is not None and psnr_value > 40:
        verdict = "Visually identical (PSNR > 40 dB)"
    else:
        verdict = "Different merge order or perceptible differences — inspect the pipeline"
    print(f"    Verdict              : {verdict}", file=sys.stderr)

    return {
        "shape_ok": True,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "identical_pct": identical_pct,
        "psnr_value": psnr_value,
        "per_channel": per_channel,
        "verdict": verdict,
    }



def _write_benchmark_csv(
    results_dir: str,
    window_idx: int,
    start: int,
    end: int,
    phases: list[str],
    baseline_name: str,
    candidate_name: str,
    baseline_runs: list[dict],
    candidate_runs: list[dict],
) -> None:
    """
    Append per-phase benchmark stats for one (baseline, candidate) pair to
    results/benchmark_results.csv. Non-measurable phases are written as
    NA_LABEL rather than a computed number.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "benchmark_results.csv"

    fieldnames = [
        "window_idx", "img_start", "img_end", "baseline", "candidate", "phase",
        "baseline_mean_s", "baseline_std_s", "baseline_margin_s",
        "candidate_mean_s", "candidate_std_s", "candidate_margin_s",
        "speedup", "ci_lower", "ci_upper",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for ph in phases:
            base_times = [r[ph] for r in baseline_runs]
            cand_times = [r[ph] for r in candidate_runs]

            base_ok = _is_measurable(base_times)
            cand_ok = _is_measurable(cand_times)

            row = {
                "window_idx" : window_idx,
                "img_start"  : start,
                "img_end"    : end,
                "baseline"   : baseline_name,
                "candidate"  : candidate_name,
                "phase"      : ph,
            }

            if base_ok and cand_ok:
                mu_b, std_b, m_b = confidence_interval(base_times)
                mu_c, std_c, m_c = confidence_interval(cand_times)
                S, lo, hi        = speedup_ci(base_times, cand_times)

                row.update({
                    "baseline_mean_s"   : round(mu_b, 6),
                    "baseline_std_s"    : round(std_b, 6),
                    "baseline_margin_s" : round(m_b, 6),
                    "candidate_mean_s"  : round(mu_c, 6),
                    "candidate_std_s"   : round(std_c, 6),
                    "candidate_margin_s": round(m_c, 6),
                    "speedup"           : round(S, 4),
                    "ci_lower"          : round(lo, 4),
                    "ci_upper"          : round(hi, 4),
                })
            else:
                # Report whichever side was measured; only the comparison itself (speedup/CI) is meaningless here
                if base_ok:
                    mu_b, std_b, m_b = confidence_interval(base_times)
                    base_stats = (round(mu_b, 6), round(std_b, 6), round(m_b, 6))
                else:
                    base_stats = (NA_LABEL, NA_LABEL, NA_LABEL)

                if cand_ok:
                    mu_c, std_c, m_c = confidence_interval(cand_times)
                    cand_stats = (round(mu_c, 6), round(std_c, 6), round(m_c, 6))
                else:
                    cand_stats = (NA_LABEL, NA_LABEL, NA_LABEL)

                row.update({
                    "baseline_mean_s"   : base_stats[0],
                    "baseline_std_s"    : base_stats[1],
                    "baseline_margin_s" : base_stats[2],
                    "candidate_mean_s"  : cand_stats[0],
                    "candidate_std_s"   : cand_stats[1],
                    "candidate_margin_s": cand_stats[2],
                    "speedup"           : NA_LABEL,
                    "ci_lower"          : NA_LABEL,
                    "ci_upper"          : NA_LABEL,
                })

            writer.writerow(row)


def _write_correctness_csv(
    results_dir: str,
    window_idx: int,
    start: int,
    end: int,
    baseline_name: str,
    candidate_name: str,
    cmp_result: dict,
) -> None:
    """Append one row of pixel-comparison results for a (baseline, candidate) pair"""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "correctness_results.csv"

    fieldnames = [
        "window_idx", "img_start", "img_end", "baseline", "candidate", "shape_match",
        "max_pixel_diff", "mean_pixel_diff", "identical_pixels_pct",
        "psnr_db",
        "blue_max_diff", "blue_mean_diff",
        "green_max_diff", "green_mean_diff",
        "red_max_diff", "red_mean_diff",
        "verdict",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        if not cmp_result.get("shape_ok"):
            writer.writerow({
                "window_idx": window_idx, "img_start": start, "img_end": end,
                "baseline": baseline_name, "candidate": candidate_name,
                "shape_match": False, "verdict": "Shape mismatch",
            })
            return

        ch_data = {name.lower(): (mx, mn) for name, mx, mn in cmp_result["per_channel"]}
        writer.writerow({
            "window_idx"          : window_idx,
            "img_start"           : start,
            "img_end"             : end,
            "baseline"            : baseline_name,
            "candidate"           : candidate_name,
            "shape_match"         : True,
            "max_pixel_diff"      : cmp_result["max_diff"],
            "mean_pixel_diff"     : round(cmp_result["mean_diff"], 6),
            "identical_pixels_pct": round(cmp_result["identical_pct"], 4),
            "psnr_db"             : round(cmp_result["psnr_value"], 4) if cmp_result["psnr_value"] is not None else "inf",
            "blue_max_diff"       : ch_data["blue"][0],
            "blue_mean_diff"      : round(ch_data["blue"][1], 6),
            "green_max_diff"      : ch_data["green"][0],
            "green_mean_diff"     : round(ch_data["green"][1], 6),
            "red_max_diff"        : ch_data["red"][0],
            "red_mean_diff"       : round(ch_data["red"][1], 6),
            "verdict"             : cmp_result["verdict"],
        })


def run_benchmark(
    input_dir: str,
    pipelines: list[PipelineSpec],
    n_runs: int = N_RUNS,
    window_size: int = WINDOW_SIZE,
):
    """
    Orchestrates the execution of multiple panorama stitching pipelines, 
    measuring performance and ensuring functional correctness.

    pipelines[0] acts as the baseline (e.g., sequential), and subsequent 
    entries are candidates evaluated against it. The benchmark operates 
    on sliding windows of images to simulate incremental stitching.

    Fault tolerance:
    - If a pipeline fails warm-up, it is excluded from the current window.
    - If a timed run fails, it is skipped without aborting the suite.
    - If the baseline completely fails, the current window is aborted early 
      (as speedup/comparisons cannot be computed).
    """
    if len(pipelines) < 2:
        raise ValueError("Provide at least a baseline and one candidate pipeline.")

    baseline   = pipelines[0]
    candidates = pipelines[1:]
    
    # Fetch input images and calculate sliding windows
    all_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required.", file=sys.stderr)
        return

    windows = [
        (start, min(start + window_size, total_images))
        for start in range(0, total_images, window_size)
        if min(start + window_size, total_images) - start >= 2
    ]

    pipeline_names = [p.name for p in pipelines]
    print("=" * 70, file=sys.stderr)
    print(f"BENCHMARK: {n_runs} runs x {len(windows)} window(s)", file=sys.stderr)
    print(f"Pipelines: {pipeline_names}  (baseline: '{baseline.name}')", file=sys.stderr)
    print(f"Total images: {total_images}  |  Window size: {window_size}", file=sys.stderr)
    print(f"Logical CPUs: {NUM_CORES}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    phases = ["extract", "match", "homo", "warp", "reext", "total"]

    # Identify if system-heavy resource pools are needed by any pipeline
    needs_process = any(p.needs_process_pool for p in pipelines)
    needs_thread  = any(p.needs_thread_pool for p in pipelines)

    for win_idx, (start, end) in enumerate(windows):
        print(f"\n{'-' * 70}", file=sys.stderr)
        print(f"WINDOW {win_idx + 1}/{len(windows)}  [images {start}:{end}]", file=sys.stderr)
        print(f"{'-' * 70}", file=sys.stderr)

        images = load_images(input_dir, start_idx=start, end_idx=end)
        if len(images) < 2:
            print("  Not enough images in this window, skipping.", file=sys.stderr)
            continue

        results_by_pipeline: dict[str, list[dict]] = {}

        # ExitStack ensures thread/process pools are safely terminated at the end of the window to prevent memory leaks
        with contextlib.ExitStack() as stack:
            resources = {}
            if needs_process:
                resources["process_executor"] = stack.enter_context(
                    ProcessPoolExecutor(max_workers=NUM_CORES)
                )
            if needs_thread:
                resources["thread_executor"] = stack.enter_context(
                    ThreadPoolExecutor(max_workers=NUM_CORES)
                )

            for spec in pipelines:
                # Warm-up Phase: Pre-allocates OS memory, loads libraries, 
                # and warms up CPU caches to guarantee stable timings later
                print(f"\n  [warm-up] Warming up '{spec.name}' pipeline (2 passes)...", file=sys.stderr)
                warmup_ok = True
                for _ in range(2):
                    try:
                        spec.run(images, resources)
                    except Exception as e:
                        print(f"  WARNING: warm-up failed for '{spec.name}' "
                              f"({type(e).__name__}: {e})", file=sys.stderr)
                        warmup_ok = False
                        break

                if not warmup_ok:
                    print(f"  Skipping '{spec.name}' entirely for this window "
                          f"(warm-up failure).", file=sys.stderr)
                    gc.collect()
                    time.sleep(1.0)
                    if spec is baseline:
                        print(f"  Baseline '{baseline.name}' failed warm-up — "
                              f"aborting the rest of this window early.", file=sys.stderr)
                        break
                    continue

                gc.collect()
                time.sleep(1.0)

                # Collects n_runs of performance metrics               
                runs: list[dict] = []
                for run in range(n_runs):
                    print(f"  {spec.name} Run {run + 1}/{n_runs}...", end=" ", flush=True, file=sys.stderr)
                    try:
                        result = spec.run(images, resources)
                        runs.append(result)
                        print("done", file=sys.stderr)
                    except Exception as e:
                        print(f"FAILED ({type(e).__name__}: {e})", file=sys.stderr)
                        print(f"  Skipping this run for '{spec.name}'; continuing with remaining runs/pipelines.", file=sys.stderr)

                    gc.collect()
                    time.sleep(0.3)

                print(f"  [Note] '{spec.name}': {len(runs)}/{n_runs} runs completed successfully.", file=sys.stderr)

                if len(runs) == 0:
                    print(f"  ERROR: all runs for '{spec.name}' failed in this window; excluding it from results.", file=sys.stderr)
                    if spec is baseline:
                        print(f"  Baseline '{baseline.name}' failed all runs — "
                              f"aborting the rest of this window early.", file=sys.stderr)
                        break
                    continue

                results_by_pipeline[spec.name] = runs

                # Force GC and let the CPU cool down to prevent thermal throttling
                print(f"  [cooldown] Letting CPU rest between pipelines...", file=sys.stderr)
                gc.collect()
                time.sleep(2.0)

        # Filter out pipelines that failed all runs
        if baseline.name not in results_by_pipeline:
            print(f"\n  ERROR: baseline '{baseline.name}' failed entirely in "
                  f"this window -- nothing to compare against, skipping "
                  f"window {win_idx + 1} entirely.", file=sys.stderr)
            continue

        succeeded_candidates = [s for s in candidates if s.name in results_by_pipeline]
        succeeded_specs = [baseline] + succeeded_candidates

        if len(succeeded_candidates) < len(candidates):
            missing = [s.name for s in candidates if s.name not in results_by_pipeline]
            print(f"\n  WARNING: these candidates failed entirely in this "
                  f"window and are excluded from correctness/report/CSV: {missing}", file=sys.stderr)

        baseline_runs = results_by_pipeline[baseline.name]

        # Ensures optimized pipelines output identical matrices to the baseline sequentially generated one
        print(f"\n  {'-' * 66}", file=sys.stderr)
        print(f"  CORRECTNESS CHECKS (window {win_idx + 1})", file=sys.stderr)
        print(f"  {'-' * 66}", file=sys.stderr)
        for spec in succeeded_candidates:
            cand_runs = results_by_pipeline[spec.name]
            print(f"\n  {baseline.name}  vs  {spec.name}:", file=sys.stderr)
            cmp_result = _compare_panoramas(baseline_runs[-1]["panorama"], cand_runs[-1]["panorama"])
            _write_correctness_csv(RESULTS_DIR, win_idx, start, end, baseline.name, spec.name, cmp_result)

        # Output Serialization and Sanity Checks
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        for spec in succeeded_specs:
            panorama = results_by_pipeline[spec.name][-1]["panorama"]

            # Sanity check: flag an oversized panorama before trusting its timing
            h, w = panorama.shape[:2]
            avg_img_area = sum(img.shape[0] * img.shape[1] for img in images) / len(images)
            if h * w > 10 * avg_img_area:
                print(f"SANITY CHECK: '{spec.name}' panorama in window {win_idx} is "
                    f"{w}x{h} ({h*w} px), {h*w/avg_img_area:.1f}x larger than the average "
                    f"input image area — inspect this output before trusting the timing.", file=sys.stderr)

            cv2.imwrite(
                f"{OUTPUT_DIR}/{spec.name}_window_{start}_{end}.jpg",
                panorama,
            )

        # Dump statistics to CSV
        for spec in succeeded_candidates:
            _write_benchmark_csv(
                RESULTS_DIR, win_idx, start, end, phases,
                baseline.name, spec.name,
                baseline_runs, results_by_pipeline[spec.name],
            )

    print(f"\n{'=' * 70}", file=sys.stderr)
    print("Benchmark complete.", file=sys.stderr)
    print(f"Output images saved to: {OUTPUT_DIR}/", file=sys.stderr)
    print(f"Results saved to: {RESULTS_DIR}/benchmark_results.csv", file=sys.stderr)
    print(f"Correctness saved to: {RESULTS_DIR}/correctness_results.csv", file=sys.stderr)
    print("=" * 70, file=sys.stderr)


if __name__ == "__main__":
    if not Path(INPUT_DIR).exists():
        print(f"ERROR: Directory '{INPUT_DIR}' not found.", file=sys.stderr)
    else:
        # Disable OpenCV internal threading to avoid oversubscription when
        # ProcessPoolExecutor/ThreadPoolExecutor are used explicitly.
        cv2.setNumThreads(1)

        # pipelines[0] is the baseline; the others are candidates.
        pipelines_to_run = [
            SEQUENTIAL_SPEC,
            PARALLEL_SPEC,
            PRODUCER_CONSUMER_SPEC,
            MAPREDUCE_SPEC,
            JOBLIB_SPEC,
            SHM_SPEC,
        ]

        run_benchmark(INPUT_DIR, pipelines_to_run, n_runs=N_RUNS, window_size=WINDOW_SIZE)