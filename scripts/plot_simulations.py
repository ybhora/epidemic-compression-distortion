from pathlib import Path
import sys
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R = Path(__file__).resolve().parents[1]

from sir_model import network_sir_deriv
from topologies import (
    star_shaped,
    fully_connected,
    unidirectional_cycle,
)

F = R / "results/figures"
O = R / "results/analysis"
F.mkdir(parents=True, exist_ok=True)
O.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 400,
    }
)


def sim(fun, m, phi, seed=0):
    P = fun(m, phi)
    I = np.zeros(m)
    I[seed] = 1
    t = np.linspace(0, 450, 8001)
    N = np.full(m, 1000.0)
    sol = solve_ivp(
        network_sir_deriv,
        (0, 450),
        np.r_[N - I, I, np.zeros(m)],
        args=(P, np.full(m, 0.3), np.full(m, 0.1), P.T @ N),
        t_eval=t,
        rtol=1e-7,
        atol=1e-9,
    )
    assert sol.success
    assert np.max(abs(sol.y[:m] + sol.y[m : 2 * m] + sol.y[2 * m :] - 1000)) < 1e-5
    return t, sol.y[m : 2 * m]


def lag(I, t):
    d = np.vstack([np.interp(np.arange(451), t, row) for row in I])
    agg = d.mean(axis=0)
    ls = []
    for x in d:
        best = -np.inf
        lbest = 0
        for l in sorted(range(-90, 91), key=lambda k: (abs(k), k)):
            a, b = (
                (x[l:], agg[:-l])
                if l > 0
                else ((x[:l], agg[-l:]) if l < 0 else (x, agg))
            )
            c = np.corrcoef(a, b)[0, 1]
            if c > best + 1e-12:
                best = c
                lbest = l
        ls.append(abs(lbest))
    return np.mean(ls)


def curve(ax, t, I, label="", color="#0072B2"):
    ax.plot(t, I.T, color=color, lw=0.5, alpha=0.4)
    a = I.mean(axis=0)
    ax.plot(t, a / a.max() * I.max(), color="#202020", lw=1.7)
    ax.set_xlim(0, 450)
    ax.set_ylim(bottom=0)
    ax.text(
        0.97, 0.9, f"{lag(I,t):.2f} d", transform=ax.transAxes, ha="right", fontsize=10
    )
    ax.set_title(label, fontsize=11)
    ax.tick_params(labelsize=10)


def save(fig, name):
    fig.savefig(F / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(F / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


funs = [star_shaped, fully_connected, unidirectional_cycle]
names = ["Hub-seeded star", "Fully connected", "Directed cycle"]
rows = []
cache = {}
for m in [2, 5, 10, 20, 25, 50, 100]:
    for j, fun in enumerate(funs):
        if j == 0 and m == 100:
            continue
        t, I = sim(fun, m, 0.02)
        cache[(m, j)] = (t, I)
        peaks = t[I.argmax(axis=1)]
        rows.append(
            dict(
                m=m,
                topology=names[j],
                lag=lag(I, t),
                peak_spread=np.ptp(peaks),
                seed_rest_gap=peaks[1:].mean() - peaks[0],
            )
        )
pd.DataFrame(rows).to_csv(O / "simulation_validation.csv", index=False)
# Split the long grid for legible supplementary pages; retain its combined vector master.
for part, ms in enumerate([[2, 5, 10, 20], [25, 50, 100]], 1):
    fig, axs = plt.subplots(
        len(ms), 3, figsize=(8, 2.1 * len(ms)), layout="constrained"
    )
    for k, m in enumerate(ms):
        for j in range(3):
            if (m, j) not in cache:
                axs[k, j].axis("off")
                axs[k, j].text(
                    0.5,
                    0.5,
                    "Infeasible at φ = 0.02\nHub outflow would exceed 1",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                continue
            curve(axs[k, j], *cache[(m, j)], names[j] if k == 0 else "")
            axs[k, j].set_ylabel(f"m = {m}") if j == 0 else None
    for a in axs[-1]:
        a.set_xlabel("Time (days)")
    save(fig, f"figureS2a_part{part}")

fig, axs = plt.subplots(3, 2, figsize=(8, 6), layout="constrained")
for k, m in enumerate([5, 25, 50]):
    for j, seed in enumerate([0, 1]):
        curve(
            axs[k, j],
            *sim(star_shaped, m, 0.02, seed),
            f"m = {m}; " + ["hub seed", "leaf seed"][j],
        )
for a in axs[-1]:
    a.set_xlabel("Time (days)")
save(fig, "figureS2b_star_seeding")
fig, axs = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
t, I = cache[(50, 2)]
curve(axs[0], t, I, "Rescaled aggregate")
axs[1].plot(t, I.T, color="#0072B2", lw=0.4, alpha=0.35)
axs[1].plot(t, I.mean(axis=0), color="black", lw=1.7)
axs[1].set_title("Unscaled mean across nodes")
for a in axs:
    a.set_xlabel("Time (days)")
    a.set_ylabel("Infected")
save(fig, "figureS2c_unrescaled_plateau")
print("Simulation figures regenerated; m=100 star infeasible documented.")
