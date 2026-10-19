from validate_pipeline import select_coarse_partition
from pathlib import Path
import sys, pickle
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
import shapefile
from validate_pipeline import load_master, ROOT, DATA, OUT, metrics, EX
from summarize_results import phase, interp

F = ROOT / "results/figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 400,
    }
)
C = {
    "core85": "#0072B2",
    "full": "#777777",
    "Louvain": "#009E73",
    "States": "#202020",
    "CSA": "#D55E00",
    "CBSA": "#CC79A7",
    "CBSA>=2": "#E69F00",
    "CBSA>=3": "#882255",
}
M = {"States": "X", "CSA": "D", "CBSA": "s", "CBSA>=2": "^", "CBSA>=3": "v"}
master = load_master()
master = master[~master.index.str[:2].isin(EX)]
parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
polys = {}
r = shapefile.Reader(str(ROOT / "data/reference/maps/UScounties_fixed_46102.shp"))
for sr in r.iterShapeRecords():
    f = sr.record["FIPS"]
    sh = sr.shape
    if f not in master.index:
        continue
    cuts = list(sh.parts) + [len(sh.points)]
    polys[f] = [
        np.array(sh.points[cuts[i] : cuts[i + 1]]) for i in range(len(cuts) - 1)
    ]


def save(fig, name):
    fig.savefig(F / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(F / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def mapplot(ax, u, title, values=False):
    ps = []
    cs = []
    rng = np.random.default_rng(42)
    labels = {v: rng.random() for v in u.dropna().unique()}
    for f, pol in polys.items():
        val = u.get(f, np.nan)
        color = (
            plt.cm.viridis(np.clip(val / 0.6, 0, 1))
            if values and pd.notna(val)
            else (plt.cm.tab20(labels[val]) if pd.notna(val) else "#eeeeee")
        )
        for p in pol:
            ps.append(p)
            cs.append(color)
    ax.add_collection(
        PolyCollection(ps, facecolors=cs, edgecolors="white", linewidths=0.12)
    )
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.3)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def axesformat(a, key):
    a.set_xscale("log")
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/1e6:g}"))
    a.set_xlabel("Mean population per unit (millions)")
    a.set_ylabel("Zero-lag incoherence" if key == "inc" else "Mean absolute lag (days)")
    a.grid(alpha=0.15)
    a.set_ylim(bottom=0)


def main():
    d = pd.read_csv(OUT / "corrected_delta_frontiers.csv")
    csa_ref = d.loc[d.family == "CSA", "compression"].iloc[0]
    # Dynamic figures and tolerance sensitivity.
    dy = pd.read_csv(OUT / "dynamic_summary.csv")
    dy["date"] = pd.to_datetime(dy.center)
    dyn = pd.read_csv(OUT / "corrected_dynamic.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for t, c in zip([2, 3, 4, 5], ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]):
        g = dy[dy.tolerance == t]
        ax[0].plot(g.date, g.admissible / 1e6, c=c, label=f"{t} days")
        a = g.groupby("phase").admissible.median()
        ax[1].plot(
            range(5),
            [
                a.get(n, np.nan) / 1e6
                for n in [
                    "Lockdown",
                    "Reopening",
                    "Fall/Winter",
                    "Recovery/Delta",
                    "Omicron",
                ]
            ],
            "-o",
            ms=3,
            c=c,
        )
    locator = mdates.MonthLocator(interval=6)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[0].legend(fontsize=10)
    ax[0].set_ylabel("Admissible population/unit (M)")
    ax[1].set_xticks(
        range(5),
        ["Lockdown", "Reopening", "Fall/Winter", "Recovery/Delta", "Omicron"],
        rotation=25,
        ha="right",
    )
    ax[1].set_ylabel("Phase median (M)")
    save(fig, "figureS7_dynamic_robustness")
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    dyn["phase"] = dyn.center.map(phase)
    for a, key in zip(axs, ["inc", "lag"]):
        for name, c in zip(
            ["Lockdown", "Reopening", "Fall/Winter", "Recovery/Delta", "Omicron"],
            ["#332288", "#882255", "#CC79A7", "#D55E00", "#009E73"],
        ):
            g = dyn[dyn.phase == name]
            for _, w in g.groupby("center"):
                a.plot(w.compression, w[key], c=c, alpha=0.07, lw=0.5)
            m = g.groupby("compression")[key].median().sort_index()
            a.plot(m.index, m.values, c=c, label=name, lw=1.6)
        axesformat(a, key)
        a.axvline(csa_ref, c="gray", ls="--", lw=0.7)
    axs[0].legend(fontsize=10)
    save(fig, "figureS8")
    dg = pd.read_csv(OUT / "deaths_gradient.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for a, key in zip(axs, ["lag", "inc"]):
        for idx, c, label in [
            (slice(0, 3), "#0072B2", "Canonical"),
            (slice(3, 6), "#009E73", "Leiden"),
        ]:
            g = dg.iloc[idx]
            a.plot(range(3), g[key], "-o", c=c, label=label)
        a.set_xticks(range(3), ["Coarse", "Intermediate", "Fine"])
        a.set_ylabel(
            "Median absolute lag (days)" if key == "lag" else "Median incoherence"
        )
        a.set_ylim(bottom=0)
    axs[1].legend()
    fig.suptitle(
        "Deaths: March 15–June 15, 2020 (same window as Figure 3)", fontsize=11
    )
    save(fig, "figureS10b_deaths_gradient")
    pr = pd.read_csv(OUT / "prediction_holdout.csv")
    sc = pd.read_csv(OUT / "prediction_scores.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for col, c, label in [
        ("observed", "black", "Observed"),
        ("mobility_prediction", "#0072B2", "Mobility"),
        ("persistence_prediction", "#D55E00", "Persistence"),
    ]:
        ax[0].plot(pd.to_datetime(pr.center), np.exp(pr[col]) / 1e6, c=c, label=label)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Population/unit (M)")
    ax[0].legend(fontsize=10)
    locator = mdates.MonthLocator(interval=6)
    ax[0].xaxis.set_major_locator(locator)
    ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax[1].bar(np.arange(3) - 0.18, sc.MAE_log_scale, 0.36, label="MAE")
    ax[1].bar(np.arange(3) + 0.18, sc.RMSE_log_scale, 0.36, label="RMSE")
    ax[1].set_xticks(range(3), sc.model)
    ax[1].set_ylabel("Error in log population/unit")
    ax[1].legend(fontsize=10)
    save(fig, "figureS9_mixing_vs_distortion")
    print("Empirical figures rendered.", flush=True)


if __name__ == "__main__":
    main()
