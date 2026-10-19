from pathlib import Path
import sys, os

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import plot_publication as pub
from summarize_results import interp, PH

O = ROOT / "results/paper"
F = ROOT / "results/figures"
OLD = ROOT / "results/analysis"
BLUE = pub.BLUE
GREEN = pub.GREEN
original_rc = plt.rcParams.copy()
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def save(fig, name):
    fig.savefig(F / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(F / (name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(name, flush=True)


def letter(a, t):
    a.text(-0.13, 1.04, t, transform=a.transAxes, weight="bold", fontsize=14)


def fmt(a, key):
    a.set_xscale("log")
    a.set_ylim(bottom=0)
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, p: pub.compact(x)))
    a.set_xlabel("Mean population per unit")
    a.grid(alpha=0.13)
    if key:
        a.set_ylabel(
            "Mean absolute optimal lag (days)"
            if key == "lag"
            else r"Incoherence ($1-\rho_0$)"
        )


colors = {
    **pub.C,
    "CSA": "#C0392B",
    "CBSA": "#82368C",
    "CBSA>=2": "#9B531A",
    "CBSA>=3": "#E60058",
}
d = pd.read_csv(O / "static_frontiers.csv")


def frontier3(frame, key, name, indep=False, families=None, connect=True):
    phases = [x for x in ["Lockdown", "Delta", "Omicron"] if x in set(frame.phase)]
    fig, axs = plt.subplots(
        1,
        len(phases),
        figsize=(5.5, 4.6) if len(phases) == 1 else (4.05 * len(phases), 3.85),
        sharey=not indep,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        bottom=0.30 if len(phases) == 1 else 0.24,
        top=0.89,
        wspace=0.16 if not indep else 0.3,
    )
    for j, (a, ph) in enumerate(zip(axs[0], phases)):
        sub = frame[frame.phase == ph]
        can = sub[sub.resolution.isna()]
        core = sub[sub.family == "core85"]
        band = (can.compression.min(), can.compression.max())
        a.axvspan(*band, color=BLUE, alpha=0.055, lw=0)
        fams = families or [
            ("full", "#aaaaaa", "Full support"),
            ("core85", BLUE, "85% core"),
        ]
        for fam, col, label in fams:
            g = sub[sub.family == fam].sort_values("compression")
            a.plot(g.compression, g[key], "-o", c=col, ms=2.3, lw=1.35, label=label)
            if key + "_lo" in g:
                a.fill_between(
                    g.compression,
                    g[key + "_lo"],
                    g[key + "_hi"],
                    color=col,
                    alpha=0.1,
                    lw=0,
                )
        for n, mark in pub.M.items():
            q = can[can.family == n]
            if q.empty:
                continue
            q = q.iloc[0]
            v = interp(core, key, q.compression)
            if connect:
                a.plot([q.compression] * 2, [v, q[key]], ":", c=colors[n], lw=0.8)
                a.scatter(
                    q.compression,
                    v,
                    facecolor="white",
                    edgecolor=colors[n],
                    s=24,
                    zorder=5,
                )
            a.scatter(
                q.compression,
                q[key],
                c=colors[n],
                marker="x" if n == "States" else mark,
                s=34,
                zorder=6,
                label=n.replace(">=", " ≥ "),
            )
        fmt(a, key if j == 0 or indep else None)
        a.set_title(ph, weight="bold")
        letter(a, chr(65 + j))
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3 if len(phases) == 1 else 7, len(labels)),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.52, 0.01),
    )
    save(fig, name)


# First-wave incoherence companion (S13a).
u = pd.read_csv(O / "M3_units.csv")
g = pd.read_csv(O / "M3_gradient.csv").set_index("partition")
names = [
    "States",
    "CSA",
    "CBSA>=2",
    "Leiden coarse",
    "Leiden intermediate",
    "Leiden fine",
]


def boxes(a, key):
    for f, col, offset in [(0, BLUE, -0.15), (1, GREEN, 0.15)]:
        meds = []
        for j in range(3):
            vals = u[u.partition == names[f * 3 + j]][key].dropna()
            meds.append(vals.median())
            bp = a.boxplot(
                vals,
                positions=[j + offset],
                widths=0.25,
                whis=0,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            bp["boxes"][0].set(facecolor=col, edgecolor=col, alpha=0.32)
            bp["medians"][0].set(color=col, lw=1.5)
            for q in bp["whiskers"] + bp["caps"]:
                q.set_visible(False)
        a.plot(np.arange(3) + offset, meds, "--", c=col, lw=1.3)
    a.axhline(u[u.partition == "States"][key].median(), c="#C66B78", ls=":", lw=1)
    a.set_xticks([0, 1, 2], ["Coarse", "Intermediate", "Fine"])
    a.set_xlim(-0.55, 2.55)
    a.set_ylim(bottom=-0.45 if key == "lag" else -0.025)
    a.grid(axis="y", alpha=0.13)
    a.set_ylabel(
        "Mean absolute optimal lag (days)"
        if key == "lag"
        else r"Incoherence ($1-\rho_0$)"
    )
    a.legend(
        handles=[
            Patch(color=BLUE, alpha=0.32, label="Canonical"),
            Patch(color=GREEN, alpha=0.32, label="Leiden"),
            Line2D([], [], c="#C66B78", ls=":", label="States median"),
        ],
        fontsize=8,
        frameon=False,
    )


fig, a = plt.subplots(figsize=(4.8, 3.5), layout="constrained")
boxes(a, "inc")
save(fig, "figureS13a_gradient_incoherence")
# Figure 3 maps and unit-level lag boxes, using the final paper's native supports.
base_save = pub.save
pub.save = save
pub.figure3(pub.series_panel())
pub.save = base_save
# Main and supplementary phase frontiers.
cases = d[d.signal == "cases"]
death = d[d.signal == "deaths"]
frontier3(cases, "lag", "figure5")
frontier3(cases, "inc", "figureS13b_frontier_incoherence")
frontier3(death, "lag", "figureS10_deaths_lag")
frontier3(death, "inc", "figureS10_deaths_incoherence")
frontier3(
    cases,
    "lag",
    "figureS6_core_thresholds_lag",
    families=[
        ("full", "#999999", "Full support"),
        ("core95", "#CC79A7", "95% core"),
        ("core90", "#E69F00", "90% core"),
        ("core85", BLUE, "85% core"),
    ],
)
frontier3(
    cases,
    "lag",
    "figureS11_algorithm_lag",
    families=[
        ("core85", BLUE, "Leiden, 85% core"),
        ("Louvain", GREEN, "Louvain, 85% core"),
    ],
    connect=False,
)
if (
    cases[
        (cases.phase == "Lockdown") & (cases.family.isin(list(pub.M) + ["core85"]))
    ].bound_pct.max()
    > 5
):
    b = pd.read_csv(O / "S-I_lag_bound_frontiers.csv")
    frontier3(
        b[(b.phase == "Lockdown") & (b.max_lag == 42)],
        "lag",
        "figureS12b_lockdown_lag42",
    )
# M6: reuse original layout and unchanged bottom panel; replace only top metric.
dy = pd.read_csv(OLD / "dynamic_summary.csv")
new = pd.read_csv(O / "M6_dynamic.csv")
dy = dy.merge(new[["center", "lag_at_csa"]], on="center", validate="many_to_one")
dy.to_csv(O / "dynamic_summary.csv", index=False)
raw = pub.series_panel()
pub.OUT = O
pub.save = save
with plt.rc_context(original_rc):
    pub.figure6(raw, cases[cases.phase == "Delta"])


# Regenerate original incoherence top, crop vector output at the subplot boundary.
def save_top(fig, name):
    a, b, twin, bottom_twin = fig.axes
    fig.delaxes(b)
    fig.delaxes(bottom_twin)
    for t in list(fig.texts):
        t.remove()
    for l in list(fig.legends):
        l.remove()
    fig.set_size_inches(10, 3.8)
    a.set_position([0.10, 0.2, 0.80, 0.68])
    twin.set_position(a.get_position())
    a.tick_params(axis="x", labelbottom=True)
    a.set_xlabel("Window center / date")
    a.xaxis.set_major_locator(pub.mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a.xaxis.set_major_formatter(
        pub.mdates.ConciseDateFormatter(a.xaxis.get_major_locator())
    )
    save(fig, "figureS13c_dynamic_incoherence")


pub.save = save_top
with plt.rc_context(original_rc):
    pub.figure6(raw, cases[cases.phase == "Delta"], metric="inc")
pub.save = save
# S-D.
fig, axs = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
for a, key in zip(axs, ["lag", "inc"]):
    for ph, col in zip(["Lockdown", "Delta", "Omicron"], [BLUE, "#D55E00", GREEN]):
        sub = cases[(cases.family == "core85") & (cases.phase == ph)].sort_values(
            "n_units"
        )
        a.plot(sub.n_units, sub[key], "-o", ms=3, c=col, label=ph)
    a.set_xscale("log")
    a.set_xlabel("Number of Leiden units")
    a.set_ylabel(
        "Mean absolute optimal lag (days)"
        if key == "lag"
        else r"Incoherence ($1-\rho_0$)"
    )
    a.grid(alpha=0.13)
    a.set_ylim(bottom=0)
axs[1].legend(fontsize=9, frameon=False)
letter(axs[0], "A")
letter(axs[1], "B")
save(fig, "figureS4_refinement")
# S-C4: decomposition uses partition means, not gradient medians.
cs = pd.read_parquet(O / "county_metrics.parquet")
c4 = []
for n, f, r in [
    ("States", "States", None),
    ("CSA", "CSA", None),
    ("CBSA>=2", "CBSA>=2", None),
    ("Leiden coarse", "full", 4.650257),
    ("Leiden intermediate", "core85", 33.471236),
    ("Leiden fine", "core85", 111.186983),
]:
    q = cases[(cases.phase == "Lockdown") & (cases.family == f)]
    q = q[q.resolution == r] if r else q
    q = q.iloc[0]
    c4.append(
        dict(
            phase="Lockdown",
            partition=n,
            support=q.support,
            timing=q.timing,
            shape=q["shape"],
            inc=q.inc,
        )
    )
for ph in ["Lockdown", "Delta", "Omicron"]:
    sub = cases[cases.phase == ph]
    core = sub[sub.family == "core85"]
    for n in ["States", "CSA"]:
        comp = sub[sub.family == n].compression.iloc[0]
        c4.append(
            dict(
                phase=ph,
                partition="Core at " + n,
                support="core85",
                timing=interp(core, "timing", comp),
                shape=interp(core, "shape", comp),
                inc=interp(core, "inc", comp),
            )
        )
c4 = pd.DataFrame(c4)
c4.to_csv(O / "S-C4_decomposition.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
for f, col, off in [(0, BLUE, -0.16), (1, GREEN, 0.16)]:
    q = c4.iloc[f * 3 : f * 3 + 3]
    axs[0].bar(np.arange(3) + off, q.timing, 0.28, color=col)
    axs[0].bar(
        np.arange(3) + off, q["shape"], 0.28, bottom=q.timing, color=col, alpha=0.3
    )
axs[0].set_xticks([0, 1, 2], ["Coarse", "Intermediate", "Fine"])
axs[0].set_title("Lockdown: six partitions")
for j, n in enumerate(["States", "CSA"]):
    q = c4[c4.partition == "Core at " + n]
    x = np.arange(3) + (j - 0.5) * 0.32
    axs[1].bar(x, q.timing, 0.28, color=[BLUE, GREEN][j])
    axs[1].bar(x, q["shape"], 0.28, bottom=q.timing, color=[BLUE, GREEN][j], alpha=0.3)
axs[1].set_xticks([0, 1, 2], ["Lockdown", "Delta", "Omicron"])
axs[1].set_title("85% core at matched compression")
for a in axs:
    a.set_ylabel("Mean incoherence components")
    a.grid(axis="y", alpha=0.12)
axs[0].legend(
    handles=[
        Patch(color=BLUE, label="Canonical"),
        Patch(color=GREEN, label="Leiden"),
        Patch(color=".35", label="Timing"),
        Patch(color=".75", label="Shape"),
    ],
    fontsize=7,
    ncol=2,
    frameon=False,
)
axs[1].legend(
    handles=[
        Patch(color=BLUE, label="States scale"),
        Patch(color=GREEN, label="CSA scale"),
    ],
    fontsize=8,
    frameon=False,
)
letter(axs[0], "A")
letter(axs[1], "B")
save(fig, "figureS13d_incoherence_decomposition")
# S-F maps. Values recomputed above, shared 0..14-day color scale.
ret = []
for ph in ["Delta", "Lockdown"]:
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
    fig.subplots_adjust(
        left=0.015, right=0.9, bottom=0.035, top=0.94, wspace=0.03, hspace=0.3
    )
    for a, fam, label in zip(
        axs.flat,
        ["full", "core95", "core90", "core85"],
        ["Full support", "95% core", "90% core", "85% core"],
    ):
        cm = cs[
            (cs.signal == "cases")
            & (cs.phase == ph)
            & (cs.family == fam)
            & (cs.resolution == 33.471236)
        ].set_index("geoid")
        v = cm.lag
        ps = []
        cols = []
        for f, shapes in pub.polys.items():
            z = v.get(f, np.nan)
            color = plt.cm.YlOrRd(min(z / 14, 1)) if pd.notna(z) else "#dedede"
            for shape in shapes:
                ps.append(shape)
                cols.append(color)
        a.add_collection(
            PolyCollection(ps, facecolors=cols, edgecolors="#cccccc", linewidths=0.15)
        )
        a.set(xlim=(-125, -66), ylim=(24, 50), aspect=1.3)
        a.axis("off")
        a.set_title(
            f"{label} | {cm.unit.nunique()} units\n{ph}; Leiden r = 33.47", fontsize=10
        )
        if fam == "full":
            keep = cm.index.isin(pub.parts["core85", 33.471236].index)
            for mask, name in [(keep, "retained"), (~keep, "excluded")]:
                ret.append(
                    dict(
                        phase=ph,
                        partition="full Leiden r=33.47",
                        support=name,
                        mean_county_lag=cm.loc[mask, "lag"].mean(),
                        n_valid=cm.loc[mask, "lag"].notna().sum(),
                    )
                )
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 14), cmap="YlOrRd"),
        cax=fig.add_axes([0.93, 0.17, 0.016, 0.64]),
        extend="max",
    )
    cb.solids.set_rasterized(False)
    cb.set_label("County-to-unit absolute optimal lag (days)")
    save(fig, "figureS6b_core_geography_lag_" + ph.lower())
pd.DataFrame(ret).to_csv(O / "S-F_retained_excluded.csv", index=False)
# S6c: two metrics per phase, lag first.
mt = pd.read_csv(O / "S-F_common_frontiers.csv")
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
fig.subplots_adjust(
    left=0.065, right=0.99, bottom=0.14, top=0.94, wspace=0.25, hspace=0.28
)
for j, ph in enumerate(["Lockdown", "Delta", "Omicron"]):
    for i, key in enumerate(["lag", "inc"]):
        a = axs[i, j]
        for n, gp in mt[mt.phase == ph].groupby("reference"):
            line = gp[gp.family == "Leiden"].sort_values("compression")
            q = gp[gp.family != "Leiden"].iloc[0]
            a.plot(
                line.compression, line[key], c=colors[n], label=n.replace(">=", " ≥ ")
            )
            a.scatter(
                q.compression, q[key], c=colors[n], marker=pub.M[n], s=35, zorder=3
            )
        fmt(a, key if j == 0 else None)
        if i == 0:
            a.set_title(ph, weight="bold")
        letter(a, chr(65 + i * 3 + j))
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(
    h,
    l,
    loc="lower center",
    ncol=3,
    frameon=False,
    title="Line: Leiden on common counties; marker: canonical",
)
save(fig, "figureS6c_matched_support")
# S-I smoothing + three-phase matched-bound summaries.
s = pd.read_csv(O / "S-I_existing_smoothing.csv")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
for n, gp in s[s.max_lag == 28].groupby("partition"):
    col = colors.get(n, GREEN)
    axs[0].plot(gp.smoothing, gp.lag, "-o", c=col, label=n.replace(">=", " ≥ "))
for n, gp in s[s.smoothing == 7].groupby("partition"):
    axs[1].plot(gp.max_lag, gp.lag, "-o", c=colors.get(n, GREEN))
for a in axs:
    a.set_ylabel("Mean absolute optimal lag (days)")
    a.grid(alpha=0.13)
    a.set_ylim(bottom=0)
axs[0].set(xlabel="Smoothing window (days)", title="Delta: smoothing")
axs[1].set(
    xlabel="Maximum absolute lag (days)", title="Delta: lag bound (7-day smoothing)"
)
axs[0].legend(fontsize=8, frameon=False)
letter(axs[0], "A")
letter(axs[1], "B")
save(fig, "figureS12_sensitivity_lag")
print("FIGURES COMPLETE", flush=True)
