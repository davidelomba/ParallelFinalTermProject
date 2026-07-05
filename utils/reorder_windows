"""
reorder_windows.py
====================
Standalone script: physically reorders the images WITHIN each window by
copying them into a new folder with new sortable filenames, so that
sequential.py's load_images() (which sorts by filename) processes each
window starting from its best "hub" image instead of always image[0].

Why this can help without touching the stitching algorithm
-------------------------------------------------------------
sequential.py's fold anchors on whatever comes first alphabetically and
never re-anchors on failure (see stitch_reanchored.py for the algorithmic
fix). If the anchor happens to be a poor hub for that window, real
recoverable pairs are silently lost -- window_diagnostics.py already
proved this happens (e.g. one window had a 97%-inlier pair that the
current file order never even tries).

This script sidesteps the algorithm entirely: it uses the SAME pairwise
data from window_diagnostics.py (results/window_diagnostics_pairs.csv)
to pick a better first image per window, and copies files into a new
directory so the existing, unmodified pipelines pick them up in the new
order.

What the reordering heuristic does (and does NOT guarantee)
----------------------------------------------------------------
For each window:
  1. Pick the ANCHOR = the image with the most passing pairs (ties broken
     by total inlier count). This is picked directly from real,
     already-measured data, so hop 1 (anchor vs. its best partner) is
     guaranteed to be the best possible first move for that window.
  2. Order the remaining images by descending raw inlier count against
     the anchor. This is a heuristic, not a guarantee: hop 2 onward
     compares against the FUSED panorama (anchor+partner), not the raw
     anchor image alone, so its actual success is not verified here --
     it is simply the best-informed guess available from static pairwise
     data. Re-run window_diagnostics.py (or just inspect the pipeline's
     own output) after reordering to confirm how many hops actually go
     through.

This script only COPIES files into a new folder -- it never renames or
deletes anything in the original input directory.

Usage:
    python reorder_windows.py
"""

import csv
from collections import defaultdict
from pathlib import Path

INPUT_DIR   = "data/input"
OUTPUT_DIR  = "data/input_reordered"
PAIRS_CSV   = "results/window_diagnostics_pairs.csv"
WINDOW_SIZE = 4


def load_pairs(csv_path: str) -> dict:
    """Groups window_diagnostics_pairs.csv rows by window_idx."""
    windows = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = int(row["window_idx"])
            windows[w].append({
                "i": int(row["local_i"]),
                "j": int(row["local_j"]),
                "inliers": int(row["num_inliers"]),
                "passes": row["passes_threshold"].strip().lower() in ("true", "1"),
            })
    return windows


def compute_order(pairs: list, n: int) -> list:
    """
    Returns a list of local indices in the new desired order.
    See module docstring for exactly what is (and isn't) guaranteed.
    """
    inlier = [[0] * n for _ in range(n)]
    degree = [0] * n
    for p in pairs:
        i, j = p["i"], p["j"]
        inlier[i][j] = inlier[j][i] = p["inliers"]
        if p["passes"]:
            degree[i] += 1
            degree[j] += 1

    def anchor_score(node):
        return (degree[node], sum(inlier[node]))

    anchor = max(range(n), key=anchor_score)
    remaining = [k for k in range(n) if k != anchor]
    remaining.sort(key=lambda k: inlier[anchor][k], reverse=True)

    return [anchor] + remaining


def main():
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERROR: '{INPUT_DIR}' not found.")
        return

    pairs_path = Path(PAIRS_CSV)
    if not pairs_path.exists():
        print(f"ERROR: '{PAIRS_CSV}' not found. Run window_diagnostics.py first.")
        return

    all_paths = sorted([
        p for p in input_path.iterdir()
        if p.suffix.lower() in ('.jpg', '.png')
    ])
    total = len(all_paths)
    windows_pairs = load_pairs(PAIRS_CSV)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Input images: {total}  |  window_size: {WINDOW_SIZE}")
    print(f"Output: {OUTPUT_DIR}/  (copies only, the original is left untouched)\n")

    global_pos = 0
    for win_idx, start in enumerate(range(0, total, WINDOW_SIZE)):
        end = min(start + WINDOW_SIZE, total)
        n = end - start
        window_files = all_paths[start:end]

        if n < 2:
            order = list(range(n))
        else:
            pairs = windows_pairs.get(win_idx)
            if not pairs:
                print(f"Window {win_idx}: no diagnostic data, copying without reordering.")
                order = list(range(n))
            else:
                order = compute_order(pairs, n)

        original_names = [window_files[k].name for k in range(n)]
        new_names = [window_files[k].name for k in order]
        if order != list(range(n)):
            print(f"Window {win_idx}: {original_names} -> {new_names}  (anchor: {new_names[0]})")
        else:
            print(f"Window {win_idx}: {original_names}  (no change needed)")

        for k in order:
            src = window_files[k]
            dest = output_path / f"{global_pos:04d}{src.suffix.lower()}"
            dest.write_bytes(src.read_bytes())
            global_pos += 1

    print(f"\nDone: {global_pos} images copied to {OUTPUT_DIR}/")
    print("Point the pipelines' INPUT_DIR to this folder to test the new order, e.g.:")
    print(f'  INPUT_DIR = "{OUTPUT_DIR}"')
    print("\nRemember: only the first hop per window is guaranteed by real data.")
    print("Re-run window_diagnostics.py (or inspect the pipeline's own output) on the")
    print("new folder to confirm how many of the following hops actually succeed.")


if __name__ == "__main__":
    main()