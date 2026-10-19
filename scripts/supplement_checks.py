"""Supplementary G1/G2 statistics and 200-permutation summaries (S3, S5)."""

import pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from validate_pipeline import load_master, ROOT, DATA, OUT, EX, metrics
from plot_empirical import plt, save, axesformat, C

master = load_master()
master = master[~master.index.str[:2].isin(EX)]
panel = pd.read_parquet(
    DATA / "epidemic_panel_daily_with_nyc.parquet",
    columns=["GEOID", "date", "smoothed_cases"],
)
panel = panel[panel.GEOID.isin(master.index)]
raw = panel.pivot(index="GEOID", columns="date", values="smoothed_cases").fillna(0)
delta = raw.loc[:, (raw.columns >= "2021-07-01") & (raw.columns <= "2021-11-30")]
parts = pickle.loads((OUT / "partitions.pkl").read_bytes())
fig, ax = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
stats = []
for i, gname in enumerate(["graph_1", "graph_2"]):
    e = pd.read_parquet(
        DATA
        / (
            f"graphs/edge_tables/{gname}_census2019.parquet"
            if gname == "graph_1"
            else "graphs/edge_tables/graph_2.parquet"
        )
    )
    e = e[e.geoid_i.isin(master.index) & e.geoid_j.isin(master.index)]
    strength = pd.concat(
        [e.groupby("geoid_i").weight.sum(), e.groupby("geoid_j").weight.sum()], axis=1
    ).sum(axis=1)
    p = master.population.reindex(strength.index)
    valid = (p > 0) & (strength > 0)
    c = ["#D55E00", "#0072B2"][i]
    rho = spearmanr(p[valid], strength[valid]).statistic
    v = np.sort(strength.to_numpy())
    lor = np.r_[0, np.cumsum(v) / v.sum()]
    gini = 1 - 2 * np.trapezoid(lor, np.linspace(0, 1, len(lor)))
    stats.append(
        dict(graph=gname, n_counties=len(v), n_edges=len(e), spearman=rho, gini=gini)
    )
    ax[0].scatter(
        p[valid],
        strength[valid] / strength.median(),
        s=2,
        alpha=0.3,
        c=c,
        label=f"G{i+1}: ρ = {rho:.3f}",
    )
    ax[1].plot(np.linspace(0, 1, len(lor)), lor, c=c, label=f"G{i+1}")
ax[0].set(
    xscale="log",
    yscale="log",
    xlabel="County population",
    ylabel="Strength / median strength",
    title="A  Population and mobility",
)
ax[0].legend(markerscale=3, fontsize=8)
ax[1].plot([0, 1], [0, 1], ":", c="gray")
ax[1].set(
    xlabel="Cumulative county fraction",
    ylabel="Cumulative strength fraction",
    title="B  Strength concentration",
)
ax[1].legend()
save(fig, "figureS3_graph_choice")
pd.DataFrame(stats).to_csv(OUT / "graph_statistics.csv", index=False)
# Permutations are descriptive; 16 comparisons share a Holm correction.
n = pd.read_csv(OUT / "corrected_nulls.csv")
fig, ax = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
nr = []
for j, (phase, g) in enumerate(n.groupby("phase")):
    names = list(g.partition.unique())
    for k, key in enumerate(["lag", "inc"]):
        a = ax[j, k]
        for i, name in enumerate(names):
            q = g[g.partition == name]
            obs = q.loc[q.rep == -1, key].iloc[0]
            v = q.loc[q.rep >= 0, key]
            lo, hi = np.quantile(v, [0.025, 0.975])
            a.plot([i, i], [lo, hi], c="gray", lw=5)
            a.scatter(i, obs, c="#D55E00", marker="D", zorder=4)
            nr.append(
                dict(
                    phase=phase,
                    partition=name,
                    metric=key,
                    observed=obs,
                    null_mean=v.mean(),
                    null_lo=lo,
                    null_hi=hi,
                    p=(1 + (v <= obs).sum()) / (len(v) + 1),
                )
            )
        a.set_xticks(range(len(names)), names, rotation=15, fontsize=8)
        a.set_ylabel(
            "Mean absolute lag (days)" if key == "lag" else "Zero-lag incoherence"
        )
        a.set_title(phase)
        a.set_ylim(bottom=0)
q = pd.DataFrame(nr)
order = np.argsort(q.p.to_numpy())
adj = np.maximum.accumulate(q.p.to_numpy()[order] * (len(q) - np.arange(len(q))))
q["p_holm"] = 0.0
q.loc[order, "p_holm"] = np.minimum(adj, 1)
q.to_csv(OUT / "null_summary.csv", index=False)
save(fig, "figureS5_null_permutation")
print("Graph and permutation summaries complete")
