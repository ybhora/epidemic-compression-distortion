"""Main empirical figure layouts. Display smoothing never enters distortion metrics."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from plot_empirical import master, parts, polys, save, mapplot, C, M
from validate_pipeline import ROOT, DATA, OUT, metrics, select_coarse_partition
from summarize_results import interp, PH

BLUE = "#2878B5"
GREEN = "#38A66A"
ORANGE = "#D55E00"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def letter(fig, ax, text, dx=-0.055, dy=0.035):
    b = ax.get_position()
    fig.text(b.x0 + dx, b.y1 + dy, text, fontsize=16, fontweight="bold", va="bottom")


def compact(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}K"


def series_panel():
    p = pd.read_parquet(
        DATA / "epidemic_panel_daily_with_nyc.parquet",
        columns=["GEOID", "date", "smoothed_cases"],
    )
    return (
        p[p.GEOID.isin(master.index)]
        .pivot(index="GEOID", columns="date", values="smoothed_cases")
        .fillna(0)
    )


def figure3(raw):
    ids = raw.index
    mm = master.reindex(ids)
    cb = mm.CBSA.dropna()
    cb = cb[cb.map(cb.value_counts()) >= 2]
    us = [
        mm.state,
        mm.CSA.dropna(),
        cb,
        select_coarse_partition(parts, master)[1],
        parts[("core85", 33.471236)],
        parts[("core85", 111.186983)],
    ]
    names = [
        "States",
        "CSA",
        "CBSA (≥2 counties)",
        "Leiden coarse",
        "Leiden intermediate",
        "Leiden fine",
    ]
    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.9, 0.9, 1.35],
        left=0.075,
        right=0.985,
        bottom=0.085,
        top=0.925,
        hspace=0.20,
        wspace=0.48,
    )
    maps = []
    for i, u in enumerate(us):
        u = u.loc[u.index.intersection(ids)]
        pop = master.population.reindex(u.index).sum() / u.nunique()
        a = fig.add_subplot(gs[i // 3, (i % 3) * 2 : (i % 3) * 2 + 2])
        maps.append(a)
        mapplot(a, u, f"{names[i]}\n{u.nunique():,} units | {compact(pop)} avg. pop.")
    from paper_gradient import draw_gradient

    bax = [fig.add_subplot(gs[2, :3]), fig.add_subplot(gs[2, 3:])]
    draw_gradient(bax)
    letter(fig, maps[0], "A", dx=-0.055, dy=0.055)
    letter(fig, bax[0], "B", dx=-0.055, dy=0.015)
    save(fig, "figure3")


def figure4(raw):
    cross = pd.read_csv(ROOT / "data/reference/statistical_areas_2023.csv", dtype=str)
    specs = [
        (
            "36",
            "New York",
            "Spring 2020",
            "2020-03-15",
            "2020-06-15",
            ["New York-Newark", "Albany-Schenectady"],
        ),
        (
            "48",
            "Texas",
            "Delta 2021",
            "2021-07-01",
            "2021-11-30",
            ["Houston-Pasadena", "Dallas-Fort Worth"],
        ),
        (
            "06",
            "California",
            "Omicron 2022",
            "2021-12-01",
            "2022-03-31",
            ["Los Angeles-Long Beach", "San Jose-San Francisco-Oakland"],
        ),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(11, 7.2))
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.08, top=0.94, wspace=0.12, hspace=0.30
    )
    rows = []
    for k, (st, name, period, lo, hi, prefixes) in enumerate(specs):
        ids = raw.index[raw.index.str[:2] == st]
        x = raw.loc[ids, (raw.columns >= lo) & (raw.columns <= hi)]
        codes = []
        for prefix in prefixes:
            found = (
                cross.loc[cross.csa_title.fillna("").str.startswith(prefix), "csa_code"]
                .dropna()
                .unique()
            )
            assert len(found) == 1, (prefix, found)
            codes.append(found[0])
        groups = [
            [set(ids)],
            [
                set(master.reindex(ids).index[master.reindex(ids).CSA == v])
                for v in codes
            ],
        ]
        lu = parts[("core85", 33.471236)]
        choices = [set(g.index) & set(ids) for _, g in lu.groupby(lu)]
        groups.append(
            [
                max(choices, key=lambda z: len(z & target) / max(1, len(z | target)))
                for target in groups[1]
            ]
        )
        short = [
            p.replace("San Jose-San Francisco-Oakland", "San Jose–SF–Oakland").replace(
                "Los Angeles-Long Beach", "LA–Long Beach"
            )
            for p in prefixes
        ]
        for j, sets in enumerate(groups):
            a = axs[k, j]
            z = lambda q: (q - q.mean()) / q.std(ddof=0)
            smooth = lambda q: q.rolling(9, center=True, min_periods=1).mean()
            for h, g in enumerate(sets):
                g = sorted(g & set(x.index))
                assert g
                xx = x.loc[g]
                agg = xx.sum(axis=0)
                col = "black" if j == 0 else [BLUE, ORANGE][h]
                a.plot(
                    xx.columns,
                    smooth(z(xx.T)) - smooth(z(xx.T)).min(),
                    c=BLUE if j == 0 else col,
                    lw=0.55,
                    alpha=0.17,
                )
                a.plot(xx.columns, smooth(z(agg)) - smooth(z(agg)).min(), c=col, lw=1.8)
                _, u = metrics(xx, pd.Series("u", index=g))
                v = u.lag.mean()
                inc = u.inc.mean()
                rows.append(
                    dict(
                        state=name,
                        level=["State", "CSA", "Leiden"][j],
                        group=h,
                        region=name if j == 0 else prefixes[h],
                        n_counties=len(g),
                        lag=v,
                        inc=inc,
                        display_extra_smoothing_days=9,
                    )
                )
                label = (
                    f"n = {len(g)} counties | mean lag = {v:.1f} d"
                    if j == 0
                    else f"{short[h]}: n = {len(g)}, lag = {v:.1f} d"
                )
                a.text(
                    0.025,
                    0.97 - 0.11 * h,
                    label,
                    transform=a.transAxes,
                    color=col,
                    fontsize=7,
                    va="top",
                )
            if k == 0:
                a.set_title(
                    [
                        "State-wide aggregate",
                        "CSA aggregates",
                        "Mobility-community aggregates",
                    ][j],
                    fontsize=10,
                    pad=8,
                )
            locator = mdates.MonthLocator()
            a.xaxis.set_major_locator(locator)
            a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            a.set_xlim(pd.Timestamp(lo), pd.Timestamp(hi))
            a.grid(alpha=0.14)
            a.tick_params(labelsize=8, length=2)
            peak = max(float(np.nanmax(line.get_ydata())) for line in a.lines)
            # Add a small lower margin while keeping the tallest curve at
            # 68% of the axes height, safely below the annotation block.
            lower = -0.06 * peak
            a.set_ylim(lower, lower + (peak - lower) / 0.68)
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color("#888888")
                sp.set_linewidth(0.65)
            if j == 0:
                a.set_ylabel(f"{name} · {period}\nCases (shifted z-score)", fontsize=10)
            if k == 2:
                a.set_xlabel("Date")
    pd.DataFrame(rows).to_csv(OUT / "case_study_metrics.csv", index=False)
    save(fig, "figure4")


def figure6(raw, d, metric="lag", figure_name="figure6"):
    """Display mean lag (main text) or incoherence (S13c) at CSA compression."""
    dy = pd.read_csv(OUT / "dynamic_summary.csv")
    dy = dy[dy.tolerance == 3].copy()
    dy["date"] = pd.to_datetime(dy.center)
    csa = d.loc[d.family == "CSA", "compression"].iloc[0]
    state = d.loc[d.family == "States", "compression"].iloc[0]
    national = raw.sum(axis=0)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.09, top=0.94, hspace=0.18)
    for i, a in enumerate(axs):
        for j, (name, lo, hi) in enumerate(PH):
            if j % 2 == 0:
                a.axvspan(
                    pd.Timestamp(lo),
                    pd.Timestamp(hi),
                    color="#888888",
                    alpha=0.065,
                    lw=0,
                )
            if i == 0:
                a.text(
                    pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2,
                    1.02,
                    name.replace("Recovery/Delta", "Recovery / Delta"),
                    transform=a.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#777777",
                    style="italic",
                )
        twin = a.twinx()
        v = national[
            (national.index >= dy.date.min()) & (national.index <= dy.date.max())
        ]
        twin.plot(v.index, v.where(v > 0), c="#888888", ls="--", lw=1.2, alpha=0.8)
        twin.set_yscale("log")
        twin.set_ylim(2000, 1.3e6)
        twin.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 5)))
        twin.yaxis.set_major_formatter(FuncFormatter(lambda x, p: compact(x)))
        twin.yaxis.set_minor_formatter(NullFormatter())
        twin.set_ylabel(
            "Daily cases, continental support", color="#888888", fontsize=10
        )
        twin.tick_params(axis="y", labelcolor="#888888", labelsize=8)
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_color("#aaaaaa")
        a.grid(alpha=0.12)
    axs[0].plot(
        dy.date,
        dy[metric + "_at_csa"],
        "-o",
        c=BLUE,
        lw=1.8,
        ms=2.6,
        label="Distortion at CSA scale",
    )
    axs[0].fill_between(dy.date, 0, dy[metric + "_at_csa"], color=BLUE, alpha=0.08)
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel(
        (
            "Mean absolute lag at CSA scale (days)"
            if metric == "lag"
            else r"Incoherence at CSA scale ($1-\rho_0$)"
        ),
        color=BLUE,
    )
    baseline = dy.loc[dy.phase == "Recovery/Delta", metric + "_at_csa"].mean()
    axs[0].axhline(baseline, ls="--", c="gray", lw=0.8)
    axs[0].text(
        0.99,
        baseline + 0.005,
        "Recovery / Delta mean",
        transform=axs[0].get_yaxis_transform(),
        ha="right",
        color="gray",
        fontsize=8,
    )
    axs[0].text(
        0.5,
        0.94,
        f"Fixed compression = {csa/1e6:.2f}M people per unit",
        transform=axs[0].transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    axs[1].plot(dy.date, dy.admissible, "-o", c=GREEN, lw=1.7, ms=2.5)
    axs[1].fill_between(
        dy.date, dy.admissible.min() * 0.8, dy.admissible, color=GREEN, alpha=0.08
    )
    axs[1].set_yscale("log")
    axs[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3, 6)))
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: compact(x)))
    axs[1].yaxis.set_minor_formatter(NullFormatter())
    axs[1].set_ylabel("Admissible scale\n(mean population per unit)", color=GREEN)
    axs[1].text(
        0.5,
        0.96,
        "Mean absolute lag tolerance = 3 days",
        ha="center",
        va="top",
        transform=axs[1].transAxes,
        fontsize=10,
    )
    for v, col, label in [(csa, BLUE, "CSA"), (state, "#C65C64", "States")]:
        axs[1].axhline(v, c=col, ls="--", lw=0.8)
        axs[1].text(
            0.99,
            v * 1.03,
            label,
            transform=axs[1].get_yaxis_transform(),
            ha="right",
            va="bottom",
            c=col,
            fontsize=9,
        )
    for a in axs:
        a.set_xlim(dy.date.min(), dy.date.max())
    locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    axs[1].xaxis.set_major_locator(locator)
    axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axs[1].set_xlabel("Window center / date")
    fig.legend(
        handles=[
            Line2D([], [], c=BLUE, marker="o", ms=3, label="Distortion at CSA scale"),
            Line2D([], [], c=GREEN, marker="o", ms=3, label="Admissible scale"),
            Line2D([], [], c="#888888", ls="--", label="Cases (right axis)"),
        ],
        loc="center",
        bbox_to_anchor=(0.5, 0.50),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    letter(fig, axs[0], "A", dx=-0.078, dy=0.01)
    letter(fig, axs[1], "B", dx=-0.078, dy=0.01)
    save(fig, figure_name)


def main():
    figure4(series_panel())
    print("Figure 4 generated.")


if __name__ == "__main__":
    main()
