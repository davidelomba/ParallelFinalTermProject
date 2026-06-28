"""
benchmark.py
============
Performance and correctness comparison between the sequential and parallel
image stitching pipelines.

Usage:
    python benchmark.py

Requirements:
    - sequential.py and parallel.py in the same directory
    - Directory data/input with at least 2 .jpg/.png images
"""

import csv
import gc
import cv2
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sequential import (
    load_images,
    extract_features,
    match_features          as match_features_seq,
    estimate_homography     as estimate_homography_seq,
    warp_and_blend,
)
from parallel import (
    extract_features_parallel,
    match_features          as match_features_par,
    estimate_homography     as estimate_homography_par,
    warp_and_blend_tiling,
)


INPUT_DIR   = "data/input"
OUTPUT_DIR  = "data/output/benchmark"
RESULTS_DIR = "results"
WINDOW_SIZE = 4
N_RUNS      = 5     # number of timed repetitions per pipeline
CONFIDENCE  = 0.95  # confidence level for intervals (t-distribution)



# Two-tailed Student t critical values for alpha=0.05
# Index = degrees of freedom (n-1); covers up to N_RUNS=10
_T_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
    5: 2.571,  6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228,
}

def t_critical(n: int) -> float:
    """Return the t critical value for n-1 degrees of freedom (95% CI)."""
    return _T_CRITICAL.get(n - 1, 2.0)  # conservative fallback


def confidence_interval(times: list[float]) -> tuple[float, float, float]:
    """
    Compute mean, standard deviation, and 95% CI half-width
    using Student's t-distribution (correct for small samples).

    Returns (mean, std, margin).
    """
    n    = len(times)
    mean = sum(times) / n
    std  = math.sqrt(sum((t - mean) ** 2 for t in times) / (n - 1))
    margin = t_critical(n) * std / math.sqrt(n)
    return mean, std, margin


def speedup_ci(
    seq_times: list[float],
    par_times: list[float],
) -> tuple[float, float, float]:
    """
    Estimate speedup S = T_seq / T_par and its 95% confidence interval
    via error propagation (delta method):

        σ_S / S ≈ sqrt( (σ_seq/μ_seq)² + (σ_par/μ_par)² )

    Returns (speedup, lower_bound, upper_bound).
    """
    mu_seq, sigma_seq, _ = confidence_interval(seq_times)
    mu_par, sigma_par, _ = confidence_interval(par_times)

    if mu_par == 0:
        return 0.0, 0.0, 0.0

    S      = mu_seq / mu_par
    n      = len(seq_times)
    tc     = t_critical(n)
    cv_seq = sigma_seq / mu_seq if mu_seq > 0 else 0
    cv_par = sigma_par / mu_par if mu_par > 0 else 0

    rel_margin = tc * math.sqrt(cv_seq**2 / n + cv_par**2 / n)
    return S, S * (1 - rel_margin), S * (1 + rel_margin)




def _time_sequential_window(images: list) -> dict[str, float]:
    """
    Run the sequential pipeline on a pre-loaded list of images.
    Returns per-phase timings (seconds) and the final panorama.
    """
    sift = cv2.SIFT_create()

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
        base_image = warp_and_blend(base_image, images[i], H)
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


def _time_parallel_window(images: list, process_executor: ProcessPoolExecutor) -> dict[str, float]:
    """
    Run the parallel pipeline on a pre-loaded list of images.
    Returns per-phase timings (seconds) and the final panorama.

    The ProcessPoolExecutor is received from the caller so that process
    spawn overhead (significant on Windows due to 'spawn' start method)
    is excluded from the timed section and amortised across all runs.
    """

    sift = cv2.SIFT_create()

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
        base_image = warp_and_blend_tiling(base_image, images[i], H, num_workers=os.cpu_count())
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



def _write_benchmark_csv(
    results_dir: str,
    window_idx: int,
    start: int,
    end: int,
    phases: list[str],
    seq_runs: list[dict],
    par_runs: list[dict],
) -> None:
    """
    Write per-phase benchmark statistics to results/benchmark_results.csv.
    Each row represents one phase of one window, with mean, std, margin,
    speedup, and 95% CI bounds for both pipelines.
    Appends to the file if it already exists, so results from multiple
    windows accumulate in a single file across calls.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "benchmark_results.csv"

    fieldnames = [
        "window_idx", "img_start", "img_end", "phase",
        "seq_mean_s", "seq_std_s", "seq_margin_s",
        "par_mean_s", "par_std_s", "par_margin_s",
        "speedup", "ci_lower", "ci_upper",
    ]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for ph in phases:
            seq_times = [r[ph] for r in seq_runs]
            par_times = [r[ph] for r in par_runs]

            mu_s, std_s, m_s = confidence_interval(seq_times)
            mu_p, std_p, m_p = confidence_interval(par_times)
            S, lo, hi        = speedup_ci(seq_times, par_times)

            writer.writerow({
                "window_idx" : window_idx,
                "img_start"  : start,
                "img_end"    : end,
                "phase"      : ph,
                "seq_mean_s" : round(mu_s,  6),
                "seq_std_s"  : round(std_s, 6),
                "seq_margin_s": round(m_s,  6),
                "par_mean_s" : round(mu_p,  6),
                "par_std_s"  : round(std_p, 6),
                "par_margin_s": round(m_p,  6),
                "speedup"    : round(S,  4),
                "ci_lower"   : round(lo, 4),
                "ci_upper"   : round(hi, 4),
            })


def _write_correctness_csv(
    results_dir: str,
    window_idx: int,
    start: int,
    end: int,
    shape_ok: bool,
    max_diff: int,
    mean_diff: float,
    identical_pct: float,
    psnr: float | None,
    per_channel: list[tuple[str, int, float]],
    verdict: str,
) -> None:
    """
    Write correctness comparison results to results/correctness_results.csv.
    One row per window. psnr is None when images are bit-identical (inf PSNR).
    Appends to the file if it already exists.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(results_dir) / "correctness_results.csv"

    fieldnames = [
        "window_idx", "img_start", "img_end", "shape_match",
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

        ch_data = {name.lower(): (mx, mn) for name, mx, mn in per_channel}
        writer.writerow({
            "window_idx"          : window_idx,
            "img_start"           : start,
            "img_end"             : end,
            "shape_match"         : shape_ok,
            "max_pixel_diff"      : max_diff,
            "mean_pixel_diff"     : round(mean_diff, 6),
            "identical_pixels_pct": round(identical_pct, 4),
            "psnr_db"             : round(psnr, 4) if psnr is not None else "inf",
            "blue_max_diff"       : ch_data["blue"][0],
            "blue_mean_diff"      : round(ch_data["blue"][1], 6),
            "green_max_diff"      : ch_data["green"][0],
            "green_mean_diff"     : round(ch_data["green"][1], 6),
            "red_max_diff"        : ch_data["red"][0],
            "red_mean_diff"       : round(ch_data["red"][1], 6),
            "verdict"             : verdict,
        })



def run_benchmark(input_dir: str, n_runs: int = N_RUNS, window_size: int = WINDOW_SIZE):
    """
    Run n_runs timed repetitions of each pipeline on every image window
    and print a report with per-phase speedup and 95% confidence intervals.
    """
    all_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])
    total_images = len(all_paths)

    if total_images < 2:
        print("ERROR: At least 2 images are required.")
        return

    windows = [
        (start, min(start + window_size, total_images))
        for start in range(0, total_images, window_size)
        if min(start + window_size, total_images) - start >= 2
    ]

    print("=" * 60)
    print(f"BENCHMARK: {n_runs} runs × {len(windows)} window(s)")
    print(f"Total images: {total_images}  |  Window size: {window_size}")
    print(f"Logical CPUs: {os.cpu_count()}")
    print("=" * 60)

    phases = ["extract", "match", "homo", "warp", "reext", "total"]
    labels = {
        "extract": "SIFT Extraction ",
        "match"  : "Feature Matching",
        "homo"   : "Homography Est. ",
        "warp"   : "Warp & Blend    ",
        "reext"  : "Feature Re-ext. ",
        "total"  : "TOTAL           ",
    }

    for win_idx, (start, end) in enumerate(windows):
        print(f"\n{'─'*60}")
        print(f"WINDOW {win_idx+1}/{len(windows)}  [images {start}:{end}]")
        print(f"{'─'*60}")

        # Images are loaded once outside the timed loop
        images = load_images(input_dir, start_idx=start, end_idx=end)
        if len(images) < 2:
            print("  Not enough images in this window, skipping.")
            continue


        with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:


            print("  [warm-up] Warming up sequential pipeline (2 passes)...")
            _time_sequential_window(images)
            _time_sequential_window(images)
            
            # Cooldown post-warmup
            gc.collect()
            time.sleep(1.0)

            seq_runs: list[dict] = []
            for run in range(n_runs):
                print(f"  Seq Run {run+1}/{n_runs}...", end=" ", flush=True)
                seq_runs.append(_time_sequential_window(images))
                print("done")
                
                gc.collect()       
                time.sleep(0.3)    

            print("  [cooldown] Letting CPU rest and clear caches between pipelines...")
            gc.collect()
            time.sleep(3.0)    
            print("  [warm-up] Warming up parallel pipeline (2 passes)...")

            _time_parallel_window(images, process_executor)
            _time_parallel_window(images, process_executor)
            
            # Cooldown post-warmup
            gc.collect()
            time.sleep(1.0)

            par_runs: list[dict] = []
            for run in range(n_runs):
                print(f"  Par Run {run+1}/{n_runs}...", end=" ", flush=True)
                par_runs.append(_time_parallel_window(images, process_executor))
                print("done")
                
                gc.collect()
                time.sleep(0.5)

        # Print per-phase report
        print(f"\n  {'Phase':<20} {'Seq (s)':>14}  {'Par (s)':>14}  {'Speedup':>10}  {'95% CI':>16}")
        print(f"  {'─'*20} {'─'*14}  {'─'*14}  {'─'*10}  {'─'*16}")

        for ph in phases:
            seq_times = [r[ph] for r in seq_runs]
            par_times = [r[ph] for r in par_runs]

            mu_s, _, m_s = confidence_interval(seq_times)
            mu_p, _, m_p = confidence_interval(par_times)
            S, lo, hi    = speedup_ci(seq_times, par_times)

            marker = " ◀" if ph == "total" else ""
            print(
                f"  {labels[ph]:<20} "
                f"{mu_s:.3f} ±{m_s:.3f}  "
                f"{mu_p:.3f} ±{m_p:.3f}  "
                f"{S:>10.2f}x  "
                f"[{lo:.2f}, {hi:.2f}]{marker}"
            )

        # Save last-run panoramas for visual inspection
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{OUTPUT_DIR}/seq_window_{start}_{end}.jpg", seq_runs[-1]["panorama"])
        cv2.imwrite(f"{OUTPUT_DIR}/par_window_{start}_{end}.jpg", par_runs[-1]["panorama"])

        # Export statistics to CSV
        _write_benchmark_csv(
            RESULTS_DIR, win_idx, start, end, phases, seq_runs, par_runs
        )

    print(f"\n{'='*60}")
    print("Benchmark complete.")
    print(f"Output images saved to: {OUTPUT_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/benchmark_results.csv")
    print("=" * 60)



def compare_outputs(input_dir: str, window_size: int = WINDOW_SIZE):
    """
    Pixel-by-pixel comparison of panoramas produced by the two pipelines.

    For each window prints:
      - whether shapes match
      - max and mean absolute pixel difference (uint8, range 0-255)
      - percentage of identical pixels
      - PSNR in dB (inf = bit-identical; >40 dB = visually indistinguishable)
      - per-channel breakdown (BGR)
    """
    all_paths = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])
    total_images = len(all_paths)

    windows = [
        (start, min(start + window_size, total_images))
        for start in range(0, total_images, window_size)
        if min(start + window_size, total_images) - start >= 2
    ]

    print("\n" + "=" * 60)
    print("CORRECTNESS COMPARISON (sequential vs parallel)")
    print("=" * 60)

    for win_idx, (start, end) in enumerate(windows):
        print(f"\nWindow {win_idx+1}: images [{start}:{end}]")
        images = load_images(input_dir, start_idx=start, end_idx=end)
        if len(images) < 2:
            print("  Not enough images, skipping.")
            continue

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as process_executor:
            img_s = _time_sequential_window(images)["panorama"]
            img_p = _time_parallel_window(images, process_executor)["panorama"]

        # Shape check
        shape_ok = img_s.shape == img_p.shape
        print(f"  Shape  seq: {img_s.shape}  par: {img_p.shape}  →  "
              f"{' match' if shape_ok else '❌ MISMATCH'}")

        if not shape_ok:
            print("    Shape mismatch: pixel comparison not possible.")
            continue

        # Pixel difference
        diff = np.abs(img_s.astype(np.int32) - img_p.astype(np.int32))
        max_diff      = int(diff.max())
        mean_diff     = float(diff.mean())
        identical_pct = float((diff == 0).mean()) * 100

        print(f"  Max pixel diff       : {max_diff}")
        print(f"  Mean pixel diff      : {mean_diff:.6f}")
        print(f"  Identical pixels     : {identical_pct:.2f}%")

        # PSNR
        mse = float(np.mean(diff.astype(np.float64) ** 2))
        if mse == 0:
            psnr_str = "∞  (bit-identical images)"
        else:
            psnr = 10 * math.log10(255.0 ** 2 / mse)
            tag  = ">40 dB (visually identical)" if psnr > 40 else " perceptible differences"
            psnr_str = f"{psnr:.2f} dB  {tag}"
        print(f"  PSNR                 : {psnr_str}")

        # Per-channel breakdown
        per_channel = []
        for ch, name in enumerate(["Blue", "Green", "Red"]):
            ch_diff = diff[:, :, ch]
            ch_max  = int(ch_diff.max())
            ch_mean = float(ch_diff.mean())
            per_channel.append((name, ch_max, ch_mean))
            print(f"  Channel {name:<5}        : max={ch_max:3d}  mean={ch_mean:.4f}")

        # Verdict
        if max_diff == 0:
            verdict = "Bit-identical"
        elif max_diff <= 1:
            verdict = "Equivalent (float32 rounding, max 1 LSB)"
        elif mse > 0 and 10 * math.log10(255.0 ** 2 / mse) > 40:
            verdict = "Visually identical (PSNR > 40 dB)"
        else:
            verdict = "Perceptible differences — please inspect the pipeline"
        print(f"  Verdict              : {verdict}")

        # Export to CSV
        psnr_value = None if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
        _write_correctness_csv(
            RESULTS_DIR, win_idx, start, end,
            shape_ok, max_diff, mean_diff, identical_pct,
            psnr_value, per_channel, verdict,
        )

    print("\n" + "=" * 60)
    print("Correctness comparison complete.")
    print(f"Results saved to: {RESULTS_DIR}/correctness_results.csv")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not Path(INPUT_DIR).exists():
        print(f"ERROR: Directory '{INPUT_DIR}' not found.")
    else:  

        cv2.setRNGSeed(42)

        # Disable OpenCV internal threading to avoid oversubscription in parallel pipelines
        cv2.setNumThreads(1)



        # 1. Performance benchmark
        run_benchmark(INPUT_DIR, n_runs=N_RUNS, window_size=WINDOW_SIZE)

        # 2. Correctness comparison
        compare_outputs(INPUT_DIR, window_size=WINDOW_SIZE)
