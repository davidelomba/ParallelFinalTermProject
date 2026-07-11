"""
Reads results/benchmark_results.csv (produced by benchmark.py) and
generates bar charts, plus an optional scaling chart across multiple
core-count runs.

    1. speedup_per_phase.png
       Average (aggregate) speedup per phase (extract/match/homo/warp/
       reext/total), one group of bars per phase, one bar per candidate
       pipeline. Computed as sum(baseline_mean_s) / sum(candidate_mean_s)
       across all windows where that phase was measurable.

    2. speedup_total_per_window.png
       Speedup on the "total" phase only, one group of bars per window,
       one bar per candidate pipeline. Shows how speedup evolves as the
       image content changes across windows.

    3. speedup_vs_cores.png (only produced in --scaling mode)
       Line chart with one subplot per phase in SCALING_PHASES (SIFT
       Extraction, Warp & Blend, TOTAL), core count on the x-axis,
       aggregate speedup on the y-axis, one line per candidate pipeline.
       Reads multiple CSVs, one per core count, matching the glob pattern
       given via --scaling (default: results/benchmark_*c_4s.csv), and
       extracts the core count from each filename via the "<N>c" pattern
       (e.g. benchmark_8c_4s.csv -> 8 cores).

Usage:
    python plot_benchmark_speedup.py [path/to/benchmark_results.csv]
    python plot_benchmark_speedup.py --scaling ["glob/pattern/*.csv"]

    If no path is given for the single-file mode, defaults to
    results/benchmark_results.csv (relative to the current working
    directory). If no pattern is given for --scaling, defaults to
    results/benchmark_*c_4s.csv.

Requirements:
    pip install pandas matplotlib
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Phase order used consistently across all charts.
PHASE_ORDER = ["extract", "match", "homo", "warp", "reext", "total"]
PHASE_LABELS = {
    "extract": "SIFT\nExtraction",
    "match": "Feature\nMatching",
    "homo": "Homography\nEst.",
    "warp": "Warp &\nBlend",
    "reext": "Feature\nRe-ext.",
    "total": "TOTAL",
}

# Phases shown in the core-scaling chart (kept separate from PHASE_ORDER
# since the scaling chart intentionally focuses on the phases that are
# actually expected to scale with core count, plus the overall total).
SCALING_PHASES = ["extract", "warp", "total"]

# Pattern used to extract the core count from a scaling CSV's filename,
# e.g. "benchmark_8c_4s.csv" -> 8.
_CORE_COUNT_PATTERN = re.compile(r"(\d+)c")


def load_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a benchmark CSV and coerce the numeric columns to float.

    Rows written as "n/a" (non-measurable phase for that pipeline, see
    benchmark.py's _is_measurable()) become NaN and are automatically
    excluded from any sum()/mean()/plotting downstream.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"'{csv_path}' not found. Run benchmark.py first, or pass the "
            f"correct path as a command-line argument."
        )

    df = pd.read_csv(csv_path)
    for col in ("baseline_mean_s", "candidate_mean_s", "speedup"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate_speedup(df: pd.DataFrame, candidate: str, phase: str) -> float:
    """
    Aggregate speedup for one (candidate, phase) pair across every row
    (i.e. every window) in df: sum(baseline_mean_s) / sum(candidate_mean_s).

    Returns NaN if the phase isn't measurable for this candidate (all
    candidate_mean_s values NaN) or if the candidate sum is non-positive.
    """
    subset = df[(df["candidate"] == candidate) & (df["phase"] == phase)]

    if subset["candidate_mean_s"].isna().all():
        return np.nan

    valid = subset.dropna(subset=["baseline_mean_s", "candidate_mean_s"])
    sum_baseline = valid["baseline_mean_s"].sum()
    sum_candidate = valid["candidate_mean_s"].sum()

    if sum_candidate <= 0:
        return np.nan

    return sum_baseline / sum_candidate


def plot_speedup_per_phase(df: pd.DataFrame, output_path: Path) -> None:
    """
    Grouped bar chart: aggregate speedup per phase (x-axis), one bar per
    candidate pipeline (color). Aggregate speedup = sum(baseline_mean_s) /
    sum(candidate_mean_s) across all windows.
    """
    candidates = sorted(df["candidate"].unique())
    phases = [p for p in PHASE_ORDER if p in df["phase"].unique()]

    means = {
        cand: [_aggregate_speedup(df, cand, ph) for ph in phases]
        for cand in candidates
    }

    x = np.arange(len(phases))
    n_candidates = len(candidates)
    bar_width = 0.8 / n_candidates

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cand in enumerate(candidates):
        offsets = x + (i - (n_candidates - 1) / 2) * bar_width
        values = means[cand]
        bars = ax.bar(offsets, values, width=bar_width, label=cand)

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
    ax.set_ylabel("Aggregate Speedup")
    ax.set_ylim(0, 3)
    ax.set_title("Aggregate Speedup per phase")
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
    ax.set_ylabel("Speedup on total time")
    ax.set_title("Total speedup per window")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _discover_scaling_csvs(pattern: str) -> dict[int, Path]:
    """
    Resolve a glob pattern to {core_count: csv_path}, extracting the core
    count from each filename via _CORE_COUNT_PATTERN (e.g.
    "benchmark_8c_4s.csv" -> 8). Raises if a match can't be parsed, or if
    no files are found at all.
    """
    paths = sorted(Path(".").glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}'. Run benchmark.py once "
            f"per core count and name/copy the resulting CSVs so the "
            f"pattern matches (e.g. results/benchmark_1c_4s.csv, "
            f"results/benchmark_2c_4s.csv, ...)."
        )

    csv_by_cores: dict[int, Path] = {}
    for path in paths:
        match = _CORE_COUNT_PATTERN.search(path.stem)
        if not match:
            raise ValueError(
                f"Could not extract a core count from filename '{path.name}' "
                f"(expected something like '..._8c_...'). Rename the file or "
                f"adjust _CORE_COUNT_PATTERN."
            )
        csv_by_cores[int(match.group(1))] = path

    return dict(sorted(csv_by_cores.items()))


def plot_speedup_vs_cores(csv_by_cores: dict[int, Path], output_path: Path) -> None:
    """
    Line chart: one subplot per phase in SCALING_PHASES, core count on the
    x-axis, aggregate speedup (sum(baseline)/sum(candidate), across all
    windows in that core-count's CSV) on the y-axis, one line per
    candidate pipeline. A dashed diagonal marks ideal linear speedup
    (speedup == core count) as a strong-scaling reference.
    """
    core_counts = sorted(csv_by_cores.keys())

    # data[phase][candidate] = [speedup_at_core_1, speedup_at_core_2, ...]
    # aligned positionally with core_counts.
    data: dict[str, dict[str, list[float]]] = {ph: {} for ph in SCALING_PHASES}
    all_candidates: set[str] = set()

    for cores in core_counts:
        df = load_benchmark_csv(csv_by_cores[cores])
        candidates = df["candidate"].unique()
        all_candidates.update(candidates)
        for ph in SCALING_PHASES:
            for cand in candidates:
                data[ph].setdefault(cand, []).append(_aggregate_speedup(df, cand, ph))

    candidates_sorted = sorted(all_candidates)
    fig, axes = plt.subplots(1, len(SCALING_PHASES), figsize=(6 * len(SCALING_PHASES), 5.5), sharey=False)

    if len(SCALING_PHASES) == 1:
        axes = [axes]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cand_colors = {cand: color_cycle[i % len(color_cycle)] for i, cand in enumerate(candidates_sorted)}

    for ax, ph in zip(axes, SCALING_PHASES):
        for cand in candidates_sorted:
            values = data[ph].get(cand, [np.nan] * len(core_counts))
            ax.plot(
                core_counts, values,
                marker="o", label=cand, color=cand_colors[cand],
            )

        ax.axhline(1.0, color="lightgray", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(core_counts)
        ax.set_xticklabels([str(c) for c in core_counts])
        ax.set_xlabel("Cores")
        ax.set_ylabel("Aggregate Speedup")
        ax.set_title(PHASE_LABELS.get(ph, ph).replace("\n", " "))
        ax.grid(True, linestyle=":", alpha=0.5)

    # Single shared legend (avoid repeating it per subplot).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=len(labels))

    fig.suptitle("Aggregate speedup vs. core count", y=1.15, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--scaling":
        pattern = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark_*c_4s.csv"
        csv_by_cores = _discover_scaling_csvs(pattern)
        print(f"Found {len(csv_by_cores)} core-count runs: {sorted(csv_by_cores.keys())}")
        plot_speedup_vs_cores(csv_by_cores, output_dir / "speedup_vs_cores.png")
        return

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/benchmark_results_8c_8s.csv")
    df = load_benchmark_csv(csv_path)

    plot_speedup_per_phase(df, output_dir / "speedup_per_phase_8c_8s.png")
    plot_speedup_total_per_window(df, output_dir / "speedup_total_per_window_8c_8s.png")


if __name__ == "__main__":
    main()