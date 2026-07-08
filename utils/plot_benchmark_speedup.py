"""
Reads results/benchmark_results.csv (produced by benchmark.py) and
generates two bar charts:

    1. speedup_per_phase.png
       Average speedup per phase (extract/match/homo/warp/reext/total),
       one group of bars per phase, one bar per candidate pipeline.
       Averaged across all windows where that phase was measurable
       (rows with speedup == "n/a" are excluded from the average).

    2. speedup_total_per_window.png
       Speedup on the "total" phase only, one group of bars per window,
       one bar per candidate pipeline. Shows how speedup evolves as the
       image content changes across windows.

Usage:
    python plot_benchmark_speedup.py [path/to/benchmark_results.csv]

    If no path is given, defaults to results/benchmark_results.csv
    (relative to the current working directory).

Requirements:
    pip install pandas matplotlib
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Phase order used consistently across both charts.
PHASE_ORDER = ["extract", "match", "homo", "warp", "reext", "total"]
PHASE_LABELS = {
    "extract": "SIFT\nExtraction",
    "match": "Feature\nMatching",
    "homo": "Homography\nEst.",
    "warp": "Warp &\nBlend",
    "reext": "Feature\nRe-ext.",
    "total": "TOTAL",
}


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load benchmark_results.csv and coerce the "speedup" column to float.

    Rows written as "n/a" (non-measurable phase for that pipeline, see
    benchmark.py's _is_measurable()) become NaN and are automatically
    excluded from any mean()/plotting downstream.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"'{csv_path}' not found. Run benchmark.py first, or pass the "
            f"correct path as a command-line argument."
        )

    df = pd.read_csv(csv_path)
    df["speedup"] = pd.to_numeric(df["speedup"], errors="coerce")
    return df


def plot_speedup_per_phase(df: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart: aggregate speedup per phase (x-axis), one bar per
    candidate pipeline (color), calculated correctly as:
    sum(baseline_times) / sum(candidate_times) across all windows.
    """
    candidates = sorted(df["candidate"].unique())
    phases = [p for p in PHASE_ORDER if p in df["phase"].unique()]

    # Calculate the aggregate speedup for each candidate and phase
    means = {}
    for cand in candidates:
        cand_values = []
        for ph in phases:
            # Filter dataframe for the current candidate and phase
            subset = df[(df["candidate"] == cand) & (df["phase"] == ph)]
            
            # If the phase is not measurable (e.g., mapreduce -> warp is NaN)
            if subset["speedup"].isna().all():
                cand_values.append(np.nan)
            else:
                # Exclude any rows with missing or corrupted time data
                valid_subset = subset.dropna(subset=["baseline_mean_s", "candidate_mean_s"])
                
                # Correct aggregate speedup calculation: Sum(T_seq) / Sum(T_par)
                sum_baseline = valid_subset["baseline_mean_s"].sum()
                sum_candidate = valid_subset["candidate_mean_s"].sum()
                
                if sum_candidate > 0:
                    aggregate_speedup = sum_baseline / sum_candidate
                    cand_values.append(aggregate_speedup)
                else:
                    cand_values.append(np.nan)
                    
        means[cand] = cand_values

    x = np.arange(len(phases))
    n_candidates = len(candidates)
    bar_width = 0.8 / n_candidates

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cand in enumerate(candidates):
        offsets = x + (i - (n_candidates - 1) / 2) * bar_width
        values = means[cand]
        bars = ax.bar(offsets, values, width=bar_width, label=cand)

        # Annotate each bar with its value, skipping NaN (non-measurable) bars.
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1.0x (no speedup)")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(p, p) for p in phases])
    ax.set_ylabel("Aggregate Speedup (x)")
    ax.set_title("Aggregate Speedup per phase, by pipeline\n(Sum of Baseline Times / Sum of Candidate Times)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_speedup_total_per_window(df: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart: speedup on the "total" phase (x-axis = window),
    one bar per candidate pipeline (color).
    """
    total_df = df[df["phase"] == "total"].copy()
    total_df = total_df.sort_values("window_idx")

    windows = sorted(total_df["window_idx"].unique())
    candidates = sorted(total_df["candidate"].unique())

    # window_labels[i] = "img_start-img_end" for readability on the x-axis
    window_labels = []
    for w in windows:
        row = total_df[total_df["window_idx"] == w].iloc[0]
        window_labels.append(f"{int(row['img_start'])}-{int(row['img_end'])}")

    x = np.arange(len(windows))
    n_candidates = len(candidates)
    bar_width = 0.8 / n_candidates

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, cand in enumerate(candidates):
        offsets = x + (i - (n_candidates - 1) / 2) * bar_width
        values = [
            total_df[(total_df["window_idx"] == w) & (total_df["candidate"] == cand)]["speedup"].mean()
            for w in windows
        ]
        ax.bar(offsets, values, width=bar_width, label=cand)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="1.0x (no speedup)")
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels, rotation=45, ha="right")
    ax.set_xlabel("Window (image range)")
    ax.set_ylabel("Speedup on TOTAL time (x)")
    ax.set_title("Total speedup per window, by pipeline")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark_results.csv")
    output_dir = Path("plots")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_benchmark_csv(csv_path)

    plot_speedup_per_phase(df, output_dir / "speedup_per_phase.png")
    plot_speedup_total_per_window(df, output_dir / "speedup_total_per_window.png")


if __name__ == "__main__":
    main()
