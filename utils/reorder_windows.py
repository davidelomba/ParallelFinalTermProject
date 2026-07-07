"""
Standalone script: physically reorders the images within each window by
copying them into a new folder with new sortable filenames, so that
every script's load_images() (which sorts by filename) processes each
window starting from its best "hub" image instead of always image[0].

Why this can help without touching the stitching algorithm:
Script's fold anchors on whatever comes first alphabetically and
never re-anchors on failure (see stitch_reanchored.py for the algorithmic
fix). If the anchor happens to be a poor hub for that window, real
recoverable pairs are silently lost.

This script sidesteps the algorithm entirely: it uses the same pairwise
data from window_diagnostics.py (results/window_diagnostics_pairs.csv)
to pick a better first image per window, and copies files into a new
directory so the existing, unmodified pipelines pick them up in the new
order.

What the reordering heuristic does (and does not guarantee) for each window:
  1. Pick the anchor = the image with the most passing pairs (ties broken
     by total inlier count). This is picked directly from real,
     already-measured data, so hop 1 (anchor vs. its best partner) is
     guaranteed to be the best possible first move for that window.
  2. Order the remaining images by descending raw inlier count against
     the anchor. This is a heuristic, not a guarantee: hop 2 onward
     compares against the fused panorama (anchor+partner), not the raw
     anchor image alone, so its actual success is not verified here.

This script only copies files into a new folder.

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
    """
    Parses the diagnostic CSV file and groups pair records by their window index.

    Args:
        csv_path (str): Path to the window_diagnostics_pairs.csv file.

    Returns:
        dict: A dictionary where keys are window indices (int) and values are 
              lists of dictionaries containing granular pair metrics.
    """
    windows = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = int(row["window_idx"])
            windows[w].append({
                "i": int(row["local_i"]),
                "j": int(row["local_j"]),
                "inliers": int(row["num_inliers"]),

                # Convert string representation of boolean to actual Python bool
                "passes": row["passes_threshold"].strip().lower() in ("true", "1"),
            })
    return windows


def compute_order(pairs: list, n: int) -> list:
    """
    Determines the optimal processing order of images within a single window.

    This function implements a "Hub Selection" heuristic. It models the window 
    as a graph where images are nodes and passing pairs are edges. The node 
    acting as the best central hub (most connections) is chosen as the anchor.

    

    Args:
        pairs (list): List of pairwise match records for the current window.
        n (int): The total number of images present in this window.

    Returns:
        list: A list of local indices (e.g., [2, 0, 1, 3]) representing 
              the optimized execution sequence.
    """

    # Initialize structures: a symmetric matrix for fast lookup, and a degree counter
    inlier = [[0] * n for _ in range(n)]
    degree = [0] * n    # Degree represents the number of valid overlapping neighbors

    for p in pairs:
        i, j = p["i"], p["j"]
        inlier[i][j] = inlier[j][i] = p["inliers"]
        if p["passes"]:
            degree[i] += 1
            degree[j] += 1

    def anchor_score(node):
        """
        Generates a sorting key tuple: (valid_connections, total_structural_inliers).
        Python compares tuples element-by-element, guaranteeing that ties in 
        the number of passing pairs are broken by total absolute image quality.
        """
        return (degree[node], sum(inlier[node]))

    # Choose the optimal Anchor (the ultimate 'hub' node)
    anchor = max(range(n), key=anchor_score)

    # Segregate remaining images and sort them based on proximity to the hub
    remaining = [k for k in range(n) if k != anchor]

    # Sort remaining elements so that images with higher geometric overlap 
    # with our anchor are fed into the pipeline first.
    remaining.sort(key=lambda k: inlier[anchor][k], reverse=True)

    # Return the combined, flattened ideal sequence
    return [anchor] + remaining


def main():
    """
    Main orchestrator. Reads the dataset, processes images window-by-window,
    calculates their optimized local sequence, and copies them to a new 
    directory with structured, alphabetically-sortable names.
    """
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERROR: '{INPUT_DIR}' not found.")
        return

    pairs_path = Path(PAIRS_CSV)
    if not pairs_path.exists():
        print(f"ERROR: '{PAIRS_CSV}' not found. Run window_diagnostics.py first.")
        return

    # Gather and sort original source images to guarantee matching baseline order
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

    # global_pos generates a strict, contiguous zero-padded naming convention (e.g., 0000, 0001)
    global_pos = 0
    for win_idx, start in enumerate(range(0, total, WINDOW_SIZE)):
        end = min(start + WINDOW_SIZE, total)
        n = end - start
        window_files = all_paths[start:end]

        # Guard clause: standalone single files cannot be paired or reordered
        if n < 2:
            order = list(range(n))
        else:
            pairs = windows_pairs.get(win_idx)
            if not pairs:
                print(f"Window {win_idx}: no diagnostic data, copying without reordering.")
                order = list(range(n))
            else:
                # Calculate the optimized permutation list
                order = compute_order(pairs, n)

        # Log visual transformations for debugging purposes
        original_names = [window_files[k].name for k in range(n)]
        new_names = [window_files[k].name for k in order]
        if order != list(range(n)):
            print(f"Window {win_idx}: {original_names} -> {new_names}  (anchor: {new_names[0]})")
        else:
            print(f"Window {win_idx}: {original_names}  (no change needed)")

        # Physically copy and map original files to the newly ordered padded names
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