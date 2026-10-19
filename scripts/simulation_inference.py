"""Recompute the historical pooling/inference demonstration with validated dynamics.

Fits deterministic total infectious prevalence, not reported case incidence.
The susceptible-fraction companion is not an inferred network reproduction number.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from validate_pipeline import ROOT, OUT
from figure1 import save

OUT.mkdir(parents=True, exist_ok=True)


from sir_model import network_sir_deriv
from topologies import star_shaped, fully_connected, unidirectional_cycle


def fitted_curve(t, beta, n):
    def rhs(_, y):
        force = beta * y[0] * y[1] / n
        return [-force, force - 0.1 * y[1], 0.1 * y[1]]

    sol = solve_ivp(rhs, (0, 450), [n - 1, 1, 0], t_eval=t, rtol=1e-7, atol=1e-9)
    assert sol.success
    return sol.y[1]


def main():
    t = np.arange(451, dtype=float)
    rows = []
    curves = []
    fig, axs = plt.subplots(6, 3, figsize=(10, 10), sharex=True)
    fig2, bxs = plt.subplots(6, 3, figsize=(10, 10), sharex=True, sharey=True)
    for f in [fig, fig2]:
        f.subplots_adjust(
            left=0.135, right=0.99, bottom=0.055, top=0.92, hspace=0.28, wspace=0.23
        )
    for h, phi in enumerate([0.001, 0.01, 0.1]):
        for k, m in enumerate([5, 50]):
            row = 2 * h + k
            for j, (name, fun) in enumerate(
                [
                    ("Hub-seeded star", star_shaped),
                    ("Fully connected", fully_connected),
                    ("Directed cycle", unidirectional_cycle),
                ]
            ):
                a, b = axs[row, j], bxs[row, j]
                for ax in [a, b]:
                    if row == 0:
                        ax.set_title(name, fontsize=10, fontweight="bold")
                    if j == 0:
                        ax.set_ylabel(
                            f"m = {m}\n" + r"$\varphi$ = " + f"{phi:g}", fontsize=9
                        )
                    ax.set_xlim(0, 450)
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.10)
                if name == "Hub-seeded star" and phi > 1 / (m - 1):
                    for ax in [a, b]:
                        ax.text(
                            0.5,
                            0.5,
                            "Infeasible hub outflow\n(no simulation)",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                        ax.set_facecolor("#f5f5f5")
                    continue
                P = fun(m, phi)
                N = np.full(m, 1000.0)
                I = np.zeros(m)
                I[0] = 1
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
                    and np.max(
                        abs(sol.y[:m] + sol.y[m : 2 * m] + sol.y[2 * m :] - 1000)
                    )
                    < 1e-5
                )
                total = sol.y[m : 2 * m].sum(axis=0)
                n = 1000 * m
                # Multiple starts check sensitivity to nonlinear-fit initialization.
                fits = [
                    least_squares(
                        lambda v: (fitted_curve(t, v[0], n) - total) / n,
                        [v],
                        bounds=([0], [10]),
                        xtol=1e-10,
                        ftol=1e-10,
                        gtol=1e-10,
                        max_nfev=300,
                    )
                    for v in [0.08, 0.15, 0.3, 0.6]
                ]
                assert all(q.success for q in fits)
                best = min(fits, key=lambda q: np.sum(q.fun**2))
                beta = best.x[0]
                pred = fitted_curve(t, beta, n)
                rows.append(
                    dict(
                        topology=name,
                        m=m,
                        phi=phi,
                        true_beta=0.3,
                        fitted_beta=beta,
                        relative_error=(beta - 0.3) / 0.3,
                        nrmse=np.sqrt(np.mean((pred - total) ** 2)) / total.max(),
                        n_timepoints=len(t),
                        best_start_objective=np.sum(best.fun**2),
                    )
                )
                a.plot(t, total, c="#D55E00", lw=1.3, label="Network aggregate")
                a.plot(t, pred, "k--", lw=1.15, label="Homogeneous SIR fit")
                a.set_ylim(bottom=0)
                a.text(
                    0.98,
                    0.93,
                    rf"$\hat\beta={beta:.3f}$",
                    ha="right",
                    transform=a.transAxes,
                    fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1),
                )
                q = 3 * sol.y[:m] / 1000
                b.plot(t, q.T, c="#2878B5", alpha=0.23, lw=0.6)
                b.plot(t, q.mean(axis=0), c="#D55E00", lw=1.4)
                b.axhline(1, c="black", ls="--", lw=0.7)
                b.set_ylim(0, 3.15)
                curves.extend(
                    dict(
                        topology=name,
                        m=m,
                        phi=phi,
                        day=day,
                        aggregate_I=obs,
                        fitted_I=fit,
                        aggregate_q=qq,
                    )
                    for day, obs, fit, qq in zip(t, total, pred, q.mean(axis=0))
                )
    fig.text(
        0.02, 0.5, "Total infectious individuals", rotation=90, va="center", fontsize=11
    )
    fig2.text(
        0.02,
        0.5,
        r"Susceptible-fraction proxy $q(t)$",
        rotation=90,
        va="center",
        fontsize=11,
    )
    for f in [fig, fig2]:
        f.text(0.5, 0.01, "Time (days)", ha="center", fontsize=11)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([], [], c="#D55E00", label="Network aggregate"),
            Line2D([], [], c="black", ls="--", label="Homogeneous SIR fit"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig2.legend(
        handles=[
            Line2D([], [], c="#2878B5", label="Node proxy"),
            Line2D([], [], c="#D55E00", label="Population-average proxy"),
            Line2D([], [], c="black", ls="--", label="q = 1"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    pd.DataFrame(rows).to_csv(OUT / "simulation_inference.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "simulation_inference_curves.csv", index=False)
    save(fig, "figureS2d_inference")
    save(fig2, "figureS2e_proxy")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
