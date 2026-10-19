from __future__ import annotations

import numpy as np

TWO_CLUSTER_BRIDGE_EDGES_1_INDEXED: tuple[tuple[int, int], ...] = (
    # Cluster A (1-5)
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 5),
    (3, 4),
    (4, 5),
    # Cluster B (6-10)
    (6, 7),
    (6, 8),
    (6, 9),
    (7, 8),
    (7, 10),
    (8, 9),
    (9, 10),
    # Bridges
    (4, 6),
    (5, 7),
)


def two_cluster_bridge_edges(one_indexed: bool = False) -> list[tuple[int, int]]:
    """Return edge list for the fixed 10-node two-cluster bridge toy network."""
    if one_indexed:
        return list(TWO_CLUSTER_BRIDGE_EDGES_1_INDEXED)
    return [(u - 1, v - 1) for (u, v) in TWO_CLUSTER_BRIDGE_EDGES_1_INDEXED]


def fully_connected(m: int, mobility: float) -> np.ndarray:
    """Row-stochastic matrix for a fully-connected commuter network."""
    if m < 1:
        raise ValueError("m must be at least 1.")
    if not (0.0 <= mobility <= 1.0):
        raise ValueError("mobility must be between 0 and 1.")
    if m == 1:
        return np.array([[1.0]], dtype=float)

    outflow_each = mobility / (m - 1)
    phi = np.full((m, m), outflow_each, dtype=float)
    np.fill_diagonal(phi, 1.0 - mobility)
    return phi


def star_shaped(m: int, mobility: float) -> np.ndarray:
    """
    Row-stochastic matrix for a star topology with node 0 as hub.

    Leaves keep (1 - mobility) at home and send mobility to hub.
    Hub sends mobility to each leaf and keeps the residual.
    """
    if m < 2:
        raise ValueError("m must be at least 2 for star topology.")
    max_mobility = 1.0 / (m - 1)
    if not (0.0 < mobility <= max_mobility):
        raise ValueError(f"mobility must be in (0, {max_mobility}] for m={m}.")

    phi = np.zeros((m, m), dtype=float)

    # Leaves: nodes 1, ..., m-1
    for i in range(1, m):
        phi[i, i] = 1.0 - mobility
        phi[i, 0] = mobility

    # Hub: node 0
    phi[0, 1:m] = mobility
    phi[0, 0] = 1.0 - (m - 1) * mobility

    return phi


def unidirectional_cycle(m: int, mobility: float) -> np.ndarray:
    """Row-stochastic matrix for a unidirectional cycle."""
    if m < 1:
        raise ValueError("m must be at least 1.")
    if not (0.0 <= mobility <= 1.0):
        raise ValueError("mobility must be between 0 and 1.")

    phi = np.zeros((m, m), dtype=float)
    for i in range(m):
        phi[i, i] = 1.0 - mobility
        phi[i, (i + 1) % m] = mobility
    return phi


def two_cluster_bridge(m: int, mobility: float) -> np.ndarray:
    """
    Row-stochastic matrix for a fixed 10-node undirected two-cluster bridge network.

    Nodes are split into two local clusters (1-5 and 6-10) with exactly two
    cross-cluster links. Each node keeps (1 - mobility) at home and splits
    `mobility` equally among its graph neighbors.
    """
    if m != 10:
        raise ValueError("two-cluster bridge topology is fixed and requires m=10.")
    if not (0.0 <= mobility <= 1.0):
        raise ValueError("mobility must be between 0 and 1.")

    adjacency = np.zeros((m, m), dtype=float)
    for u, v in two_cluster_bridge_edges(one_indexed=False):
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0

    phi = np.zeros((m, m), dtype=float)
    for i in range(m):
        neighbors = np.flatnonzero(adjacency[i, :])
        if neighbors.size == 0:
            raise ValueError(f"Node {i} has no neighbors; topology must be connected.")
        phi[i, i] = 1.0 - mobility
        phi[i, neighbors] = mobility / float(neighbors.size)

    return phi
