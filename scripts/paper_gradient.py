"""The two views of first-wave lag used in main-text Figure 3B."""

from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "States",
    "CSA",
    "CBSA>=2",
    "Leiden coarse",
    "Leiden intermediate",
    "Leiden fine",
]
BLUE, GREEN = "#2878B5", "#38A66A"


def draw_gradient(axes):
    units = pd.read_csv(ROOT / "results/paper/M3_units.csv")
    summary = pd.read_csv(ROOT / "results/paper/M3_gradient.csv").set_index("partition")
    left, right = axes
    for family, color, offset in [(0, BLUE, -0.15), (1, GREEN, 0.15)]:
        medians = []
        for level in range(3):
            values = units[units.partition == NAMES[family * 3 + level]].lag.dropna()
            medians.append(values.median())
            box = left.boxplot(
                values,
                positions=[level + offset],
                widths=0.25,
                whis=0,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.32)
            box["medians"][0].set(color=color, lw=1.5)
            for artist in box["whiskers"] + box["caps"]:
                artist.set_visible(False)
        left.plot(np.arange(3) + offset, medians, "--", color=color, lw=1.3)
        values = summary.loc[NAMES[family * 3 : family * 3 + 3]]
        right.bar(
            np.arange(3) + offset,
            values.county_gt7,
            width=0.28,
            color=color,
            alpha=0.28,
        )
        right.bar(np.arange(3) + offset, values.county_gt14, width=0.28, color=color)
    left.axhline(
        summary.loc["States", "unit_median_lag"], color="#C66B78", ls=":", lw=1
    )
    left.set_ylabel("Mean absolute optimal lag (days)")
    left.set_ylim(bottom=-0.45)
    left.legend(
        handles=[
            Patch(color=BLUE, alpha=0.32, label="Canonical"),
            Patch(color=GREEN, alpha=0.32, label="Leiden"),
            Line2D([], [], color="#C66B78", ls=":", label="States median"),
        ],
        frameon=False,
        fontsize=8,
    )
    right.set_ylabel("Counties lagging their unit (%)")
    right.set_ylim(0, np.ceil(summary.county_gt7.max() / 10) * 10)
    right.legend(
        handles=[
            Patch(color=".75", label="> 7 days"),
            Patch(color=".35", label="> 14 days"),
        ],
        frameon=False,
        fontsize=8,
    )
    for axis in axes:
        axis.set_xticks([0, 1, 2], ["Coarse", "Intermediate", "Fine"])
        axis.set_xlim(-0.55, 2.55)
        axis.grid(axis="y", alpha=0.13)
