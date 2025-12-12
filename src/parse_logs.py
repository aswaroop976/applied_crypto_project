#!/usr/bin/env python3
import re
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Regex to capture per-image block for pHash logs:
# - image name
# - pHash distance (small)
# - pHash distance (original)
# - L2 per pixel
PHASH_BLOCK_RE = re.compile(
    r"""
    Processed\s+(?P<name>\S+)\s+.*?          # image name
    Phash\ final\s+\(small\):\s+Phashdist=(?P<small>\d+).*?
    Phash\ final\s+\(original\):\s+Phashdist=(?P<orig>\d+).*?
    Final\ L2\ per\ pixel\ for\ original:\s+(?P<l2>[0-9.eE+-]+)
    """,
    re.S | re.VERBOSE,
)

# Regex to capture per-image block for PDQ logs:
# - image name
# - PDQ distance (small)
# - PDQ distance (original)
# - L2 per pixel
PDQ_BLOCK_RE = re.compile(
    r"""
    Processed\s+(?P<name>\S+)\s+.*?          # image name
    PDQ\ final\s+\(small\):\s+PDQdist=(?P<small>\d+).*?
    PDQ\ final\s+\(original\):\s+PDQdist=(?P<orig>\d+).*?
    Final\ L2\ per\ pixel\ for\ original:\s+(?P<l2>[0-9.eE+-]+)
    """,
    re.S | re.VERBOSE,
)


def parse_phash_log(text: str) -> Tuple[int, List[float]]:
    total_images = 0
    success_l2s: List[float] = []

    for m in PHASH_BLOCK_RE.finditer(text):
        total_images += 1
        small_dist = int(m.group("small"))
        orig_dist = int(m.group("orig"))
        l2 = float(m.group("l2"))

        # Attack considered successful if *either* pHash dist == 2
        if small_dist == 2 or orig_dist == 2:
            success_l2s.append(l2)

    return total_images, success_l2s


def parse_pdq_log(
    text: str, threshold: int = 30
) -> Tuple[int, List[float], List[Tuple[str, int, int, float]]]:
    total_images = 0
    success_l2s: List[float] = []
    success_meta: List[Tuple[str, int, int, float]] = []

    for m in PDQ_BLOCK_RE.finditer(text):
        total_images += 1
        name = m.group("name")
        small_dist = int(m.group("small"))
        orig_dist = int(m.group("orig"))
        l2 = float(m.group("l2"))

        # Successful if either PDQ distance hits or exceeds the threshold
        if small_dist >= threshold or orig_dist >= threshold:
            success_l2s.append(l2)
            success_meta.append((name, small_dist, orig_dist, l2))

    return total_images, success_l2s, success_meta


def plot_cdf(l2_values: List[float], title: str, outfile: str) -> None:
    if not l2_values:
        print("No successful attacks to plot.")
        return

    data = np.array(l2_values)
    data.sort()

    # Empirical CDF: for sorted x, y = rank / n
    n = len(data)
    y = np.arange(1, n + 1) / n

    plt.figure()
    plt.plot(data, y, linestyle="-", marker="", linewidth=2)

    plt.xlabel(r"$L_2$ perturbation per pixel")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(outfile, dpi=300)
    print(f"SAVED: {outfile}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} LOGFILE.txt")
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Heuristic: decide whether this is a pHash log or a PDQ log
    if "PDQ final (small)" in text:
        # --- PDQ MODE ---
        threshold = 30
        total_images, success_l2s, success_meta = parse_pdq_log(text, threshold=threshold)

        print(f"[PDQ] Total images parsed: {total_images}")
        print(f"[PDQ] Successful attacks (PDQdist >= {threshold} in small/original): {len(success_l2s)}")
        #print()
        #print("Successful PDQ images:")
        #for name, small_dist, orig_dist, l2 in success_meta:
        #    print(
        #        f"  {name}: PDQdist_small={small_dist}, "
        #        f"PDQdist_orig={orig_dist}, L2_per_pixel={l2:.10f}"
        #    )

        plot_cdf(success_l2s, title=f"PDQ, T = {threshold}", outfile="pdq_T30_cdf.png")

    else:
        # --- PHASH MODE (original behavior) ---
        total_images, success_l2s = parse_phash_log(text)

        print(f"[pHash] Total images parsed: {total_images}")
        print(f"[pHash] Successful attacks (pHashdist == 2): {len(success_l2s)}")

        plot_cdf(success_l2s, title="pHash, T = 2", outfile="phash_T2_cdf.png")


if __name__ == "__main__":
    main()
