#!/usr/bin/env python3
import re
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Regex to capture per-image block:
# - image name
# - pHash distance (small)
# - pHash distance (original)
# - L2 per pixel
BLOCK_RE = re.compile(
    r"""
    Processed\s+(?P<name>\S+)\s+.*?          # image name
    Phash\ final\s+\(small\):\s+Phashdist=(?P<small>\d+).*?
    Phash\ final\s+\(original\):\s+Phashdist=(?P<orig>\d+).*?
    Final\ L2\ per\ pixel\ for\ original:\s+(?P<l2>[0-9.]+)
    """,
    re.S | re.VERBOSE,
)


def parse_log(text: str) -> Tuple[int, List[float]]:
    """
    Parse the full log text.

    Returns:
        total_images: number of images parsed
        success_l2s: list of L2 per pixel values where
                     pHash distance == 2 for either small or original.
    """
    total_images = 0
    success_l2s: List[float] = []

    for m in BLOCK_RE.finditer(text):
        total_images += 1
        small_dist = int(m.group("small"))
        orig_dist = int(m.group("orig"))
        l2 = float(m.group("l2"))

        # Attack considered successful if *either* pHash dist == 2
        if small_dist == 2 or orig_dist == 2:
            success_l2s.append(l2)

    return total_images, success_l2s


def plot_cdf(l2_values: List[float], title: str = "pHash, T = 2") -> None:
    """
    Plot the empirical CDF of the given L2 per pixel values.
    """
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

    # Show on screen; replace with savefig(...) if you prefer saving
    plt.savefig("phash_T2_cdf.png", dpi=300)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} LOGFILE.txt")
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    total_images, success_l2s = parse_log(text)

    print(f"Total images parsed: {total_images}")
    print(f"Successful attacks (pHashdist == 2): {len(success_l2s)}")
    print("L2 per pixel for successful attacks:")
    for v in success_l2s:
        print(f"  {v:.10f}")

    plot_cdf(success_l2s, title="pHash continuous, T = 2")


if __name__ == "__main__":
    main()
