"""Netlist hypergraph clustering for ANCHOR_SOFT init (MTK DreamPlace++ recipe).

Builds an undirected weighted graph from the netlist (clique expansion of
hypernets with weight 1/(k-1) per pair) and finds communities via networkx
Louvain. Big nets (> max_net_size pins) are skipped — they connect everything
to everything and just smear the partition.

Returned `cluster_id` is a numpy int array of shape [n_total] indexed exactly
like benchmark.macro_positions: [0, n_hard) hard, [n_hard, n_total) soft.

API:
    cluster_id, num_clusters, stats = cluster_macros(benchmark, ...)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _build_pair_weights(
    benchmark, max_net_size: int = 20,
) -> Dict[Tuple[int, int], float]:
    n_total = benchmark.num_macros
    pair_weight: Dict[Tuple[int, int], float] = {}
    for net in benchmark.net_nodes:
        nodes = [int(x) for x in net.tolist() if 0 <= int(x) < n_total]
        k = len(nodes)
        if k < 2 or k > max_net_size:
            continue
        w = 1.0 / (k - 1)
        for i in range(k):
            ni = nodes[i]
            for j in range(i + 1, k):
                nj = nodes[j]
                if ni == nj:
                    continue
                a, b = (ni, nj) if ni < nj else (nj, ni)
                pair_weight[(a, b)] = pair_weight.get((a, b), 0.0) + w
    return pair_weight


def _louvain_partition(
    pair_weight: Dict[Tuple[int, int], float],
    n_total: int,
    seed: int,
    resolution: float,
):
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    g = nx.Graph()
    g.add_nodes_from(range(n_total))
    for (a, b), w in pair_weight.items():
        g.add_edge(a, b, weight=w)

    communities = louvain_communities(g, seed=seed, resolution=resolution)
    cluster_id = np.full(n_total, -1, dtype=np.int32)
    for cid, comm in enumerate(communities):
        for node in comm:
            cluster_id[node] = cid
    leftover = int(np.sum(cluster_id < 0))
    if leftover > 0:
        next_cid = int(cluster_id.max()) + 1 if cluster_id.max() >= 0 else 0
        for i in np.where(cluster_id < 0)[0]:
            cluster_id[i] = next_cid
            next_cid += 1
    return cluster_id


def _label_propagation_partition(
    pair_weight: Dict[Tuple[int, int], float],
    n_total: int,
    seed: int,
):
    import networkx as nx
    from networkx.algorithms.community import label_propagation_communities

    g = nx.Graph()
    g.add_nodes_from(range(n_total))
    for (a, b), w in pair_weight.items():
        g.add_edge(a, b, weight=w)

    communities = list(label_propagation_communities(g))
    cluster_id = np.full(n_total, -1, dtype=np.int32)
    for cid, comm in enumerate(communities):
        for node in comm:
            cluster_id[node] = cid
    leftover = int(np.sum(cluster_id < 0))
    if leftover > 0:
        next_cid = int(cluster_id.max()) + 1 if cluster_id.max() >= 0 else 0
        for i in np.where(cluster_id < 0)[0]:
            cluster_id[i] = next_cid
            next_cid += 1
    return cluster_id


def cluster_macros(
    benchmark,
    method: str = "auto",
    seed: int = 42,
    max_net_size: int = 20,
    resolution: float = 1.0,
    target_num_clusters: int = 0,
) -> Tuple[np.ndarray, int, dict]:
    """Cluster macros (hard + soft) by netlist topology.

    Args:
        benchmark: macro_place.benchmark.Benchmark
        method: "auto" (louvain unless n_total > 5000), "louvain",
                "label_propagation"
        seed: RNG seed for Louvain
        max_net_size: nets with more pins are skipped (they project to dense
                      cliques and erode the partition)
        resolution: Louvain resolution (>1 = more, smaller clusters)
        target_num_clusters: if > 0, tune resolution by binary search to land
                             near this count (slow — 5-8 louvain runs)

    Returns:
        cluster_id: int32 array [n_total]
        num_clusters: int
        stats: {pair_count, edge_count, mean_cluster_size, max_cluster_size,
                singleton_count}
    """
    n_total = benchmark.num_macros
    pair_weight = _build_pair_weights(benchmark, max_net_size=max_net_size)

    if method == "auto":
        method = "louvain" if n_total <= 5000 else "label_propagation"

    if target_num_clusters > 0 and method == "louvain":
        lo, hi = 0.1, 8.0
        for _ in range(7):
            mid = (lo + hi) * 0.5
            cid = _louvain_partition(pair_weight, n_total, seed, mid)
            k = int(cid.max()) + 1
            if k < target_num_clusters:
                lo = mid
            elif k > target_num_clusters:
                hi = mid
            else:
                break
        cluster_id = cid
    elif method == "louvain":
        cluster_id = _louvain_partition(pair_weight, n_total, seed, resolution)
    elif method == "label_propagation":
        cluster_id = _label_propagation_partition(pair_weight, n_total, seed)
    else:
        raise ValueError(f"unknown clustering method {method!r}")

    num_clusters = int(cluster_id.max()) + 1
    cluster_sizes = np.bincount(cluster_id, minlength=num_clusters)
    stats = {
        "pair_count": len(pair_weight),
        "edge_weight_sum": float(sum(pair_weight.values())),
        "num_clusters": num_clusters,
        "mean_cluster_size": float(cluster_sizes.mean()),
        "max_cluster_size": int(cluster_sizes.max()),
        "min_cluster_size": int(cluster_sizes.min()),
        "singleton_count": int((cluster_sizes == 1).sum()),
        "method": method,
    }
    return cluster_id, num_clusters, stats


def distribute_anchors_grid(
    num_clusters: int, canvas_w: float, canvas_h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place K anchors on canvas as ~regular grid + small jitter.

    Returns: float64 array [num_clusters, 2] of (x, y) positions in microns.
    """
    if num_clusters <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    cols = int(np.ceil(np.sqrt(num_clusters * canvas_w / max(canvas_h, 1e-9))))
    cols = max(1, cols)
    rows = int(np.ceil(num_clusters / cols))
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    jitter_w = cell_w * 0.15
    jitter_h = cell_h * 0.15
    anchors = np.zeros((num_clusters, 2), dtype=np.float64)
    for c in range(num_clusters):
        r = c // cols
        col = c % cols
        cx = (col + 0.5) * cell_w + rng.uniform(-jitter_w, jitter_w)
        cy = (r + 0.5) * cell_h + rng.uniform(-jitter_h, jitter_h)
        anchors[c, 0] = cx
        anchors[c, 1] = cy
    return anchors


def distribute_anchors_initial_centroid(
    cluster_id: np.ndarray,
    initial_pos: np.ndarray,
    movable: np.ndarray,
) -> np.ndarray:
    """Anchor per cluster = centroid of MOVABLE members at their initial pos.

    For clusters with no movable members, falls back to the centroid of all
    members. Useful when initial_pos is non-trivial (e.g. real chip floorplan).
    """
    num_clusters = int(cluster_id.max()) + 1
    anchors = np.zeros((num_clusters, 2), dtype=np.float64)
    for c in range(num_clusters):
        mask = cluster_id == c
        mask_mov = mask & movable
        use = mask_mov if mask_mov.any() else mask
        anchors[c, 0] = float(initial_pos[use, 0].mean())
        anchors[c, 1] = float(initial_pos[use, 1].mean())
    return anchors


def anchor_soft_init(
    benchmark,
    cluster_id: np.ndarray,
    seed: int = 42,
    spawn_radius_frac: float = 0.02,
    anchor_strategy: str = "grid",
) -> np.ndarray:
    """Generate ANCHOR_SOFT initial positions a-la MTK DreamPlace++.

    All movable macros sharing a cluster start in a tight Gaussian cloud
    around their cluster anchor. Fixed macros keep their initial position.

    Args:
        anchor_strategy: "grid" — anchors on uniform grid covering canvas
                         "centroid" — anchors at cluster centroid of initial pos

    Returns: float32 numpy array [n_total, 2]
    """
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_min = min(canvas_w, canvas_h)
    n_total = benchmark.num_macros

    rng = np.random.default_rng(seed)
    num_clusters = int(cluster_id.max()) + 1
    if anchor_strategy == "centroid":
        initial_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
        anchors = distribute_anchors_initial_centroid(cluster_id, initial_pos, movable)
    else:
        anchors = distribute_anchors_grid(num_clusters, canvas_w, canvas_h, rng)

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    movable = benchmark.get_movable_mask().cpu().numpy().astype(bool)
    fixed_pos = benchmark.macro_positions.cpu().numpy().astype(np.float64)

    sigma = canvas_min * spawn_radius_frac
    pos = np.zeros((n_total, 2), dtype=np.float64)
    for i in range(n_total):
        if not movable[i]:
            pos[i] = fixed_pos[i]
            continue
        c = cluster_id[i]
        ax, ay = anchors[c]
        pos[i, 0] = ax + rng.normal(0.0, sigma)
        pos[i, 1] = ay + rng.normal(0.0, sigma)

    pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)
    return pos.astype(np.float32)
