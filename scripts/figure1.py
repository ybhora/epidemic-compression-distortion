"""Figure 1: canonical metapopulation SIR examples; requires no external data."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.integrate import solve_ivp
from sir_model import network_sir_deriv
from topologies import (
    star_shaped,
    fully_connected,
    unidirectional_cycle,
    two_cluster_bridge,
)
from validate_pipeline import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis"
F = ROOT / "results/figures"
BLUE, GREEN, ORANGE = "#2878B5", "#38A66A", "#D55E00"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(F / (name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def simulation(fun, m, phi, seed=0):
    P = fun(m, phi)
    N = np.full(m, 1000.0)
    I = np.zeros(m)
    I[seed] = 1
    t = np.linspace(0, 450, 8001)
    sol = solve_ivp(
        network_sir_deriv,
        (0, 450),
        np.r_[N - I, I, np.zeros(m)],
        args=(P, np.full(m, 0.3), np.full(m, 0.1), P.T @ N),
        t_eval=t,
        rtol=1e-7,
        atol=1e-9,
    )
    assert (
        sol.success
        and np.max(abs(sol.y[:m] + sol.y[m : 2 * m] + sol.y[2 * m :] - 1000)) < 1e-5
    )
    return t, sol.y[m : 2 * m]


def schematic(ax, P, bridge=False):
    m = len(P)
    if bridge:
        xy = np.array(
            [
                [-1.2, 0],
                [-0.9, 0.8],
                [-0.9, -0.8],
                [-0.3, 0.45],
                [-0.3, -0.45],
                [0.3, 0.45],
                [0.3, -0.45],
                [0.9, 0.8],
                [0.9, -0.8],
                [1.2, 0],
            ]
        )
        seed = 1
    elif np.count_nonzero(P[0]) == m and np.count_nonzero(P[1]) == 2:
        angles = np.linspace(0, 2 * np.pi, m - 1, endpoint=False) + np.pi / 4
        xy = np.vstack([[0, 0], np.c_[np.cos(angles), np.sin(angles)]])
        seed = 0
    else:
        angles = np.linspace(0, 2 * np.pi, m, endpoint=False) - np.pi / 2
        xy = np.c_[np.cos(angles), np.sin(angles)]
        seed = 0
    for i in range(m):
        for j in range(m):
            if i == j or P[i, j] <= 0:
                continue
            bidir = P[j, i] > 0
            if bidir and j < i:
                continue
            ax.add_patch(
                FancyArrowPatch(
                    xy[i],
                    xy[j],
                    arrowstyle="<->" if bidir else "-|>",
                    mutation_scale=7,
                    lw=0.65,
                    color="#8997A6",
                    shrinkA=10,
                    shrinkB=10,
                )
            )
    for i, (x, y) in enumerate(xy):
        col = (
            "#D96583" if i == seed else ("#F5CF97" if bridge and i >= 5 else "#BCD5EA")
        )
        ax.add_patch(
            Circle(
                (x, y), 0.18 if bridge else 0.23, fc=col, ec="#657789", lw=0.8, zorder=3
            )
        )
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=8, zorder=4)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.3, 1.3), aspect="equal")
    ax.axis("off")


def figure1():
    fig = plt.figure(figsize=(11.2, 7.0))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.0, 1, 1, 1],
        left=0.105,
        right=0.94,
        bottom=0.09,
        top=0.96,
        wspace=0.27,
        hspace=0.24,
    )
    bridge_rows = []
    funs = [star_shaped, fully_connected, unidirectional_cycle, two_cluster_bridge]
    names = [
        "Star shaped",
        "Fully connected",
        "Unidirectional cycle",
        "Two-cluster bridge",
    ]
    for j, fun in enumerate(funs):
        a = fig.add_subplot(gs[0, j])
        schematic(a, fun(10 if j == 3 else 5, 0.02), j == 3)
        a.text(
            0.5,
            -0.03,
            names[j],
            ha="center",
            va="top",
            transform=a.transAxes,
            fontweight="bold",
            fontsize=11,
        )
    for row, m in enumerate([5, 25, 50]):
        for j, fun in enumerate(funs):
            a = fig.add_subplot(gs[row + 1, j])
            phi = [0.002, 0.02, 0.2][row] if j == 3 else 0.02
            t, I = simulation(fun, 10 if j == 3 else m, phi, 1 if j == 3 else 0)
            scale = I.max()
            if j < 3:
                a.plot(t, (I / scale).T, color=BLUE, alpha=0.24, lw=0.65)
                agg = I.mean(axis=0)
                a.plot(t, agg / agg.max(), color="#202020", lw=1.6)
                a.set_xlim(0, 350)
            else:
                for ids, col in [(slice(0, 5), BLUE), (slice(5, 10), ORANGE)]:
                    a.plot(t, (I[ids] / scale).T, color=col, alpha=0.22, lw=0.65)
                    a.plot(
                        t,
                        I[ids].mean(axis=0) / I[ids].mean(axis=0).max(),
                        c=col,
                        lw=1.6,
                    )
                a.plot(t, I.mean(axis=0) / I.mean(axis=0).max(), color="black", lw=1.7)
                a.set_xlim(0, 200)
                # Exactly the same daily-interpolated, ±90-day, unit-first lag calculation.
                daily = pd.DataFrame([np.interp(np.arange(451), t, q) for q in I])
                _, u = metrics(daily, pd.Series("unit", index=daily.index), max_lag=90)
                _, clusters = metrics(
                    daily,
                    pd.Series(["Cluster 1"] * 5 + ["Cluster 2"] * 5, index=daily.index),
                    max_lag=90,
                )
                values = [
                    ("Cluster 1", clusters.lag.iloc[0], BLUE),
                    ("Cluster 2", clusters.lag.iloc[1], ORANGE),
                    ("Total", u.lag.iloc[0], "black"),
                ]
                for h, (label, value, col) in enumerate(values):
                    a.text(
                        0.97,
                        0.96 - 0.09 * h,
                        f"{label}: {value:.1f} d",
                        transform=a.transAxes,
                        ha="right",
                        va="top",
                        fontsize=7.8,
                        color=col,
                    )
                    bridge_rows.append(dict(phi=phi, unit=label, lag=value))
                a.set_ylabel(r"$\varphi$ = " + f"{phi:g}", fontsize=10)
            a.set_ylim(-0.02, 1.5 if j == 3 else 1.05)
            a.set_yticks([])
            a.grid(alpha=0.08)
            for sp in a.spines.values():
                sp.set_visible(True)
                sp.set_color("#999999")
                sp.set_linewidth(0.6)
            if j == 0:
                a.set_ylabel(f"m = {m}", fontsize=11)
            if row < 2:
                a.set_xticklabels([])
            else:
                a.set_xlabel("Time (days)")
            a.tick_params(labelsize=8, length=2)
            if row == 0 and j == 1:
                a.legend(
                    [Line2D([], [], c=BLUE, alpha=0.5), Line2D([], [], c="black")],
                    ["Node", "Aggregate"],
                    fontsize=8,
                    loc="upper right",
                    frameon=False,
                )
            if row == 0 and j == 3:
                a.legend(
                    [
                        Line2D([], [], c=BLUE),
                        Line2D([], [], c=ORANGE),
                        Line2D([], [], c="black"),
                    ],
                    ["Cluster 1", "Cluster 2", "Total"],
                    fontsize=7.0,
                    loc="upper left",
                    frameon=False,
                    handlelength=1.0,
                    labelspacing=0.25,
                )
    fig.text(0.03, 0.955, "A", fontweight="bold", fontsize=16)
    fig.text(0.75, 0.955, "B", fontweight="bold", fontsize=16)
    fig.text(
        0.048,
        0.40,
        "Normalized infections",
        rotation=90,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.016,
        0.40,
        "Increasing network size (m)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.032, 0.69),
            (0.032, 0.15),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=10,
            color="gray",
        )
    )
    fig.add_artist(
        Line2D(
            [0.735, 0.735],
            [0.07, 0.96],
            transform=fig.transFigure,
            color="#dddddd",
            lw=0.8,
        )
    )
    fig.text(
        0.985,
        0.40,
        "Increasing mobility coupling",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.967, 0.69),
            (0.967, 0.15),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=10,
            color="gray",
        )
    )
    pd.DataFrame(bridge_rows).to_csv(OUT / "bridge_cluster_lags.csv", index=False)
    save(fig, "figure1")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
