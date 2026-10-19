"""Resident-based SIR dynamics, Eq. (network) in the paper."""

import numpy as np


def network_sir_deriv(
    _t: float,
    y: np.ndarray,
    phi: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    present_population: np.ndarray,
) -> np.ndarray:
    """Derivative for the network SIR ODE system."""
    m = len(beta)
    s = y[:m]
    i = y[m : 2 * m]

    infection_sum = phi.T @ i
    force_j = beta * (infection_sum / present_population)
    effective_force = phi @ force_j

    ds = -s * effective_force
    di = s * effective_force - gamma * i
    dr = gamma * i
    return np.concatenate([ds, di, dr])
