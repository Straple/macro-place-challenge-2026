"""Alternative initial-position generators for the K-batch gradient placer.

Three strategies, all returning ``pos_init`` of shape ``[K, n_active, 2]``
as float32 numpy arrays with fixed macros at their original positions:

    spectral_init     — solve a graph Laplacian system ``L x = b`` on movable
                        macros (clique-expanded netlist, fixed macros and
                        ports as boundary). Per-K diversity comes from
                        gaussian noise added on top of the WL-optimal base.

    constructive_init — greedy macro-by-macro placement in order of
                        descending net degree. Each macro samples its grid
                        cell from a Boltzmann distribution over scores that
                        penalise weighted L1 distance to already-placed
                        neighbours. Per-K diversity = different temperatures.

    hybrid_init       — quarter spectral + three quarters constructive with
                        temperatures spread across [0.5, 2.0].

No overlap check is performed; the gradient placer's overlap penalty is
expected to spread macros apart. Fixed macros and ports are NEVER moved.

Inputs taken from the ``Benchmark`` object only — no PlacementCost
required, so these helpers can be unit-tested without the C++ env.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _build_pair_weights(
    net_nodes,
    n_total: int,
    max_net_size: int = 20,
) -> Dict[Tuple[int, int], float]:
    """Clique-expand each hyperedge with weight ``w = 1 / (k - 1)``.

    Big nets (``> max_net_size`` pins) are skipped — they connect everything
    to everything and dominate the topology with noise. Same convention as
    ``clustering._build_pair_weights``.
    """
    pair_weight: Dict[Tuple[int, int], float] = {}
    for net in net_nodes:
        nodes_raw = net.tolist() if hasattr(net, "tolist") else list(net)
        nodes = sorted({int(x) for x in nodes_raw if 0 <= int(x) < n_total})
        k = len(nodes)
        if k < 2 or k > max_net_size:
            continue
        w = 1.0 / (k - 1)
        for i in range(k):
            ni = nodes[i]
            for j in range(i + 1, k):
                nj = nodes[j]
                key = (ni, nj)
                pair_weight[key] = pair_weight.get(key, 0.0) + w
    return pair_weight


def _build_adjacency_lists(
    pair_weight: Dict[Tuple[int, int], float],
    n_total: int,
) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_total)]
    for (a, b), w in pair_weight.items():
        adj[a].append((b, w))
        adj[b].append((a, w))
    return adj


def _benchmark_arrays(benchmark):
    """Pull plain numpy arrays from a Benchmark (works on CPU and CUDA tensors)."""
    n_total = int(benchmark.num_macros)
    initial_pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float32)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float32)
    movable = benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    return n_total, initial_pos, sizes, movable, canvas_w, canvas_h


def _clamp_and_restore(
    pos_K: np.ndarray,
    sizes: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    movable: np.ndarray,
    initial_pos: np.ndarray,
) -> np.ndarray:
    """Clip macro centers into the canvas and force fixed macros to original positions."""
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    pos_K[..., 0] = np.clip(pos_K[..., 0], half_w[None, :], canvas_w - half_w[None, :])
    pos_K[..., 1] = np.clip(pos_K[..., 1], half_h[None, :], canvas_h - half_h[None, :])
    fixed_idx = ~movable
    pos_K[:, fixed_idx, :] = initial_pos[None, fixed_idx, :]
    return pos_K


def _build_csr_from_pairs(
    pair_weight: Dict[Tuple[int, int], float],
    movable: np.ndarray,
    movable_idx: np.ndarray,
    pos_to_local: np.ndarray,
    initial_pos: np.ndarray,
):
    """Build movable-movable adjacency in CSR-like form + boundary RHS.

    Returns:
        indptr, indices, data: CSR pieces of A_mm (n_mov x n_mov), entries
                               are negative weights (so that L_mm = D + A_mm
                               with D = total weighted degree on movable nodes).
        D_mov: float64 [n_mov] — weighted degree of movable nodes (sum over
               all neighbours, fixed ones included).
        b_x, b_y: float64 [n_mov] — boundary contributions from fixed nodes.
    """
    n_mov = int(movable_idx.size)

    nbr_lists: List[List[Tuple[int, float]]] = [[] for _ in range(n_mov)]
    deg = np.zeros(n_mov, dtype=np.float64)
    b_x = np.zeros(n_mov, dtype=np.float64)
    b_y = np.zeros(n_mov, dtype=np.float64)
    fx = initial_pos[:, 0].astype(np.float64)
    fy = initial_pos[:, 1].astype(np.float64)

    for (a, b), w in pair_weight.items():
        a_mov = movable[a]
        b_mov = movable[b]
        if a_mov and b_mov:
            la = int(pos_to_local[a])
            lb = int(pos_to_local[b])
            nbr_lists[la].append((lb, -float(w)))
            nbr_lists[lb].append((la, -float(w)))
            deg[la] += w
            deg[lb] += w
        elif a_mov and not b_mov:
            la = int(pos_to_local[a])
            deg[la] += w
            b_x[la] += w * fx[b]
            b_y[la] += w * fy[b]
        elif b_mov and not a_mov:
            lb = int(pos_to_local[b])
            deg[lb] += w
            b_x[lb] += w * fx[a]
            b_y[lb] += w * fy[a]

    indptr = np.zeros(n_mov + 1, dtype=np.int64)
    for i in range(n_mov):
        indptr[i + 1] = indptr[i] + len(nbr_lists[i])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    p = 0
    for i in range(n_mov):
        for j, w in nbr_lists[i]:
            indices[p] = j
            data[p] = w
            p += 1

    return indptr, indices, data, deg, b_x, b_y


def _spmv_csr(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
              diag: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute (diag(D) + A) @ x where A is the off-diagonal CSR matrix."""
    n = diag.shape[0]
    y = diag * x
    for i in range(n):
        s = indptr[i]
        e = indptr[i + 1]
        if e > s:
            y[i] += np.dot(data[s:e], x[indices[s:e]])
    return y


def _cg_solve(indptr, indices, data, diag, b, tol=1e-5, maxiter=300):
    """Conjugate gradient on (diag(D) + A) x = b with Jacobi preconditioner."""
    n = b.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = b - _spmv_csr(indptr, indices, data, diag, x)
    M_inv = np.where(diag > 1e-12, 1.0 / np.maximum(diag, 1e-12), 1.0)
    z = M_inv * r
    p = z.copy()
    rz_old = float(np.dot(r, z))
    b_norm = float(np.linalg.norm(b)) + 1e-30
    for _ in range(maxiter):
        Ap = _spmv_csr(indptr, indices, data, diag, p)
        alpha = rz_old / (float(np.dot(p, Ap)) + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / b_norm < tol:
            break
        z = M_inv * r
        rz_new = float(np.dot(r, z))
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new
    return x


def spectral_init(
    benchmark,
    plc=None,
    K: int = 64,
    seed: int = 42,
    max_net_size: int = 20,
    cg_tol: float = 1e-4,
    cg_maxiter: int = 300,
    noise_frac: float = 0.05,
    base_jitter_frac: float = 0.005,
) -> np.ndarray:
    """Quadratic-WL spectral placement with per-K gaussian perturbation.

    Solves on the movable subset:
        min_x  sum_(i,j) w_ij * (x_i - x_j)^2  + boundary terms from fixed
        equiv: (D - A_mm) x = A_mf x_fixed
    where the graph is the clique-expanded netlist (skip big nets), ``D``
    is the weighted degree of each movable node (counting both movable and
    fixed neighbours), ``A_mm`` is the movable-movable adjacency.

    Per-K diversity:
      * each seed adds ``noise_frac * canvas_min`` gaussian to (x, y).
      * tiny ``base_jitter_frac`` jitter prevents seeds collapsing on top
        of each other when ``noise_frac`` is small.

    Implementation note: a small in-house Jacobi-preconditioned CG (pure
    numpy) is used so we don't depend on scipy.

    Returns:
        pos_init: float32 numpy array [K, n_active, 2].
    """
    n_total, initial_pos, sizes, movable, canvas_w, canvas_h = _benchmark_arrays(benchmark)
    canvas_min = float(min(canvas_w, canvas_h))

    pair_weight = _build_pair_weights(benchmark.net_nodes, n_total, max_net_size=max_net_size)

    movable_idx = np.where(movable)[0]
    n_mov = int(movable_idx.size)

    pos_K = np.broadcast_to(initial_pos[None, :, :], (K, n_total, 2)).copy().astype(np.float32)

    if n_mov == 0:
        return pos_K

    if not pair_weight:
        rng_fb = np.random.default_rng(seed)
        for k in range(K):
            r = rng_fb.uniform(0.0, 1.0, size=(n_mov, 2))
            pos_K[k, movable_idx, 0] = r[:, 0] * canvas_w
            pos_K[k, movable_idx, 1] = r[:, 1] * canvas_h
        return _clamp_and_restore(pos_K, sizes, canvas_w, canvas_h, movable, initial_pos)

    pos_to_local = -np.ones(n_total, dtype=np.int64)
    pos_to_local[movable_idx] = np.arange(n_mov)

    indptr, indices, data, deg, b_x, b_y = _build_csr_from_pairs(
        pair_weight, movable, movable_idx, pos_to_local, initial_pos)

    iso_mask = deg < 1e-12
    if iso_mask.any():
        deg = deg + iso_mask.astype(np.float64)
        b_x[iso_mask] = canvas_w / 2.0
        b_y[iso_mask] = canvas_h / 2.0

    x_base = _cg_solve(indptr, indices, data, deg, b_x,
                       tol=cg_tol, maxiter=cg_maxiter)
    y_base = _cg_solve(indptr, indices, data, deg, b_y,
                       tol=cg_tol, maxiter=cg_maxiter)

    if not (np.isfinite(x_base).all() and np.isfinite(y_base).all()):
        x_base = np.full(n_mov, canvas_w / 2.0)
        y_base = np.full(n_mov, canvas_h / 2.0)

    base = np.stack([x_base, y_base], axis=1).astype(np.float32)

    sigma = canvas_min * noise_frac
    base_sigma = canvas_min * base_jitter_frac
    rng = np.random.default_rng(seed)
    base_jitter = rng.normal(0.0, base_sigma, size=(n_mov, 2)).astype(np.float32)
    base = base + base_jitter

    for k in range(K):
        rng_k = np.random.default_rng(seed + k * 1009 + 1)
        noise = rng_k.normal(0.0, sigma, size=(n_mov, 2)).astype(np.float32)
        pos_K[k, movable_idx, :] = base + noise

    return _clamp_and_restore(pos_K, sizes, canvas_w, canvas_h, movable, initial_pos)


def constructive_init(
    benchmark,
    plc=None,
    K: int = 64,
    seed: int = 42,
    temperature_K: np.ndarray = None,
    grid_resolution: int = 16,
    top_n: int = 10,
    max_net_size: int = 20,
    jitter_frac: float = 0.4,
    spread_weight: float = 0.0,
    spread_weight_K: np.ndarray = None,
) -> np.ndarray:
    """Greedy-Boltzmann constructive placer with optional density-aware spread.

    Macros are placed in descending net-degree order. For each macro and
    each grid cell c we compute two terms:

        wl_term(c)     = sum_n w_n * (|c.x - n.x| + |c.y - n.y|)
                         over already-placed neighbours.
        spread_term(c) = cumulative macro area placed in c so far,
                         normalised by cell capacity (so 0 = empty,
                         1 = full at target_density).

    The score is the z-normalised sum (per K, per macro):

        score(c) = - wl_z(c) - spread_weight * spread_z(c)

    z-normalisation makes ``spread_weight`` a relative knob: 0 = pure WL,
    1 = WL and spread weighted equally, >1 = prioritise spread.

    All K seeds share the same ordering but draw independent samples —
    diversity comes from Boltzmann (varying ``T_k``) and per-cell jitter.

    Args:
        benchmark: macro_place.benchmark.Benchmark
        plc: unused, kept for signature symmetry.
        K: number of seeds.
        seed: base RNG seed.
        temperature_K: optional [K] float array. Default = log-spaced
                       between 0.5 and 2.0 (low T = greedy, high T = random).
        grid_resolution: cells per side. ``G*G`` candidate positions.
        top_n: keep this many top cells per macro before softmax sampling.
        max_net_size: skip nets bigger than this.
        jitter_frac: gaussian jitter inside the chosen cell, as a fraction
                     of cell side. 0.4 ~= 0.4 * cell_w/h sigma.
        spread_weight: scalar — relative weight of the cell-load penalty
                       vs the WL term. 0 = pure WL-greedy (legacy).
        spread_weight_K: optional per-K spread weight, shape [K]. Overrides
                         ``spread_weight`` when given.

    Returns:
        pos_init: float32 numpy array [K, n_active, 2].
    """
    n_total, initial_pos, sizes, movable, canvas_w, canvas_h = _benchmark_arrays(benchmark)
    canvas_min = float(min(canvas_w, canvas_h))

    pair_weight = _build_pair_weights(benchmark.net_nodes, n_total, max_net_size=max_net_size)
    adj = _build_adjacency_lists(pair_weight, n_total)

    degree = np.zeros(n_total, dtype=np.float64)
    for (a, b), w in pair_weight.items():
        degree[a] += w
        degree[b] += w

    movable_idx = np.where(movable)[0]
    if movable_idx.size == 0:
        return np.broadcast_to(initial_pos[None, :, :], (K, n_total, 2)).copy().astype(np.float32)

    order = movable_idx[np.argsort(-degree[movable_idx], kind="stable")]

    G = int(grid_resolution)
    cell_w = canvas_w / G
    cell_h = canvas_h / G
    cell_capacity = cell_w * cell_h
    gx = (np.arange(G) + 0.5) * cell_w
    gy = (np.arange(G) + 0.5) * cell_h
    grid_centers = np.stack(
        np.meshgrid(gx, gy, indexing="xy"), axis=-1
    ).reshape(-1, 2).astype(np.float32)
    GG = grid_centers.shape[0]

    if temperature_K is None:
        temperature_K = np.linspace(0.5, 2.0, K, dtype=np.float32)
    else:
        temperature_K = np.asarray(temperature_K, dtype=np.float32)
        assert temperature_K.shape == (K,)

    if spread_weight_K is None:
        spread_weight_K = np.full(K, float(spread_weight), dtype=np.float32)
    else:
        spread_weight_K = np.asarray(spread_weight_K, dtype=np.float32)
        assert spread_weight_K.shape == (K,)
    use_spread = bool((spread_weight_K > 0).any())

    pos_K = np.broadcast_to(initial_pos[None, :, :], (K, n_total, 2)).copy().astype(np.float32)
    placed = ~movable.copy()

    macro_areas = (sizes[:, 0] * sizes[:, 1]).astype(np.float32)

    cell_load_K = np.zeros((K, GG), dtype=np.float32)
    if use_spread:
        for j in np.where(placed)[0]:
            x0 = float(initial_pos[j, 0])
            y0 = float(initial_pos[j, 1])
            cx = int(min(max(int(x0 / cell_w), 0), G - 1))
            cy = int(min(max(int(y0 / cell_h), 0), G - 1))
            cell_load_K[:, cy * G + cx] += macro_areas[j] / cell_capacity

    rng = np.random.default_rng(seed)
    jitter_sigma_x = jitter_frac * cell_w
    jitter_sigma_y = jitter_frac * cell_h
    n_top = min(top_n, GG)

    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0

    for m in order:
        placed_neigh = [(nb, w) for (nb, w) in adj[m] if placed[nb]]
        if not placed_neigh:
            if use_spread:
                load_z = cell_load_K - cell_load_K.mean(axis=1, keepdims=True)
                load_z_std = load_z.std(axis=1, keepdims=True)
                load_z = load_z / np.maximum(load_z_std, 1e-6)
                scores = -spread_weight_K[:, None] * load_z
            else:
                scores = None
            if scores is None:
                cell_choice = rng.integers(0, GG, size=K)
            else:
                top_idx = np.argpartition(scores, GG - n_top, axis=1)[:, GG - n_top:]
                top_scores = np.take_along_axis(scores, top_idx, axis=1)
                top_scores = top_scores - top_scores.max(axis=1, keepdims=True)
                T_safe = np.maximum(temperature_K[:, None], 1e-3)
                probs = np.exp(top_scores / T_safe)
                probs = probs / probs.sum(axis=1, keepdims=True)
                u = rng.random((K, 1))
                cum = np.cumsum(probs, axis=1)
                local_choice = (cum >= u).argmax(axis=1)
                cell_choice = np.take_along_axis(top_idx, local_choice[:, None], axis=1).ravel()
        else:
            neigh_idx = np.array([nb for nb, _ in placed_neigh], dtype=np.int64)
            neigh_w = np.array([w for _, w in placed_neigh], dtype=np.float32)
            neigh_pos_K = pos_K[:, neigh_idx, :]
            dx = grid_centers[None, :, None, 0] - neigh_pos_K[:, None, :, 0]
            dy = grid_centers[None, :, None, 1] - neigh_pos_K[:, None, :, 1]
            dist_l1 = np.abs(dx) + np.abs(dy)
            wdist = (dist_l1 * neigh_w[None, None, :]).sum(axis=2)

            wl_z = wdist - wdist.mean(axis=1, keepdims=True)
            wl_z_std = wl_z.std(axis=1, keepdims=True)
            wl_z = wl_z / np.maximum(wl_z_std, 1e-6)
            score_total = -wl_z

            if use_spread:
                load_z = cell_load_K - cell_load_K.mean(axis=1, keepdims=True)
                load_z_std = load_z.std(axis=1, keepdims=True)
                load_z = load_z / np.maximum(load_z_std, 1e-6)
                score_total = score_total - spread_weight_K[:, None] * load_z

            scores = score_total
            top_idx = np.argpartition(scores, GG - n_top, axis=1)[:, GG - n_top:]
            top_scores = np.take_along_axis(scores, top_idx, axis=1)
            top_scores = top_scores - top_scores.max(axis=1, keepdims=True)
            T_safe = np.maximum(temperature_K[:, None], 1e-3)
            probs = np.exp(top_scores / T_safe)
            probs = probs / probs.sum(axis=1, keepdims=True)
            u = rng.random((K, 1))
            cum = np.cumsum(probs, axis=1)
            local_choice = (cum >= u).argmax(axis=1)
            cell_choice = np.take_along_axis(top_idx, local_choice[:, None], axis=1).ravel()

        chosen = grid_centers[cell_choice]
        jitter = rng.normal(0.0, 1.0, size=(K, 2))
        jitter[:, 0] *= jitter_sigma_x
        jitter[:, 1] *= jitter_sigma_y
        new_pos = (chosen + jitter).astype(np.float32)
        new_pos[:, 0] = np.clip(new_pos[:, 0], half_w[m], canvas_w - half_w[m])
        new_pos[:, 1] = np.clip(new_pos[:, 1], half_h[m], canvas_h - half_h[m])
        pos_K[:, m, :] = new_pos
        placed[m] = True

        if use_spread:
            k_idx = np.arange(K)
            cell_load_K[k_idx, cell_choice] += macro_areas[m] / cell_capacity

    return _clamp_and_restore(pos_K, sizes, canvas_w, canvas_h, movable, initial_pos)


def louvain_refined_init(
    benchmark,
    plc=None,
    K: int = 64,
    seed: int = 42,
    cluster_target: int = 0,
    spawn_radius_frac: float = 0.05,
    spawn_adaptive: bool = True,
    anchor_jitter_frac: float = 0.05,
    grid_radius_frac: float = 0.08,
    grid_radius_steps: int = 4,
    top_n: int = 8,
    temperature_K: np.ndarray = None,
    max_net_size: int = 20,
) -> np.ndarray:
    """Louvain-anchor init + WL-aware refinement inside the anchor radius.

    Mirrors the existing gradient_batch.py init (Louvain clustering + 3
    anchor flavours per K + adaptive spawn sigma) so the cohesion loss
    can latch onto the same cluster geometry. Then, instead of pure
    gaussian jitter, places each macro by sampling from a small grid
    AROUND its cluster anchor, scored by HPWL to already-placed
    neighbours. This keeps the cluster spread (good congestion) while
    biasing within-cluster positions toward low WL.

    Args:
        cluster_target: Louvain target cluster count (auto if 0).
        spawn_radius_frac, spawn_adaptive, anchor_jitter_frac: same
            knobs as gradient_batch — anchor build path is duplicated
            here.
        grid_radius_frac: candidate-grid radius around the anchor, as
            a fraction of canvas_min. ``grid_radius_steps`` steps in
            each direction → ``(2*steps+1)^2`` candidate cells.
        top_n: keep this many best cells before Boltzmann sample.
        temperature_K: optional [K] temperature.

    Returns:
        pos_init: float32 numpy array [K, n_active, 2].
    """
    sd = str(__import__("pathlib").Path(__file__).resolve().parent)
    import sys as _sys
    if sd not in _sys.path:
        _sys.path.insert(0, sd)
    from clustering import (cluster_macros, distribute_anchors_grid,
                             distribute_anchors_initial_centroid)

    n_total, initial_pos, sizes, movable, canvas_w, canvas_h = _benchmark_arrays(benchmark)
    canvas_min = float(min(canvas_w, canvas_h))

    if cluster_target <= 0:
        cluster_target = max(15, n_total // 30)

    cluster_id, num_clusters, _ = cluster_macros(
        benchmark, method="louvain", seed=seed,
        max_net_size=max_net_size, target_num_clusters=cluster_target,
    )
    cluster_id_np = cluster_id.astype(np.int64)

    anchors_centroid = distribute_anchors_initial_centroid(
        cluster_id_np, initial_pos.astype(np.float64), movable)
    anchors_grid = distribute_anchors_grid(
        num_clusters, canvas_w, canvas_h, np.random.default_rng(seed + 1))

    rng = np.random.default_rng(seed)
    anchor_jitter = canvas_min * anchor_jitter_frac
    anchors_K = np.zeros((K, num_clusters, 2), dtype=np.float64)
    for k in range(K):
        kind = k % 3
        rng_k = np.random.default_rng(seed + k * 1009)
        if kind == 0:
            anchors_K[k] = anchors_centroid
        elif kind == 1:
            anchors_K[k] = anchors_grid
        else:
            perm = rng_k.permutation(num_clusters)
            anchors_K[k] = anchors_grid[perm]
        anchors_K[k] += rng_k.normal(0.0, anchor_jitter,
                                     size=(num_clusters, 2))

    if spawn_adaptive:
        cluster_sizes = np.bincount(cluster_id_np, minlength=num_clusters)
        mean_size = max(1.0, float(cluster_sizes.mean()))
        sigma_per_cluster = (canvas_min * spawn_radius_frac
                             * np.sqrt(cluster_sizes / mean_size))
    else:
        sigma_per_cluster = np.full(num_clusters, canvas_min * spawn_radius_frac)

    pair_weight = _build_pair_weights(benchmark.net_nodes, n_total,
                                       max_net_size=max_net_size)
    adj = _build_adjacency_lists(pair_weight, n_total)
    degree = np.zeros(n_total, dtype=np.float64)
    for (a, b), w in pair_weight.items():
        degree[a] += w
        degree[b] += w

    movable_idx = np.where(movable)[0]
    if movable_idx.size == 0:
        return np.broadcast_to(initial_pos[None, :, :], (K, n_total, 2)).copy().astype(np.float32)
    order = movable_idx[np.argsort(-degree[movable_idx], kind="stable")]

    R = int(grid_radius_steps)
    M = (2 * R + 1) ** 2
    offs = np.arange(-R, R + 1, dtype=np.float32)
    ox, oy = np.meshgrid(offs, offs, indexing="xy")
    grid_offsets = np.stack([ox.ravel(), oy.ravel()], axis=1)
    grid_radius = canvas_min * grid_radius_frac

    if temperature_K is None:
        temperature_K = np.linspace(0.5, 1.5, K, dtype=np.float32)
    else:
        temperature_K = np.asarray(temperature_K, dtype=np.float32)
        assert temperature_K.shape == (K,)

    pos_K = np.broadcast_to(initial_pos[None, :, :], (K, n_total, 2)).copy().astype(np.float32)
    pos_K = np.ascontiguousarray(pos_K)
    placed = ~movable.copy()

    for k in range(K):
        rng_k_init = np.random.default_rng(seed + k * 1009 + 7)
        anchor_pos_k = anchors_K[k][cluster_id_np]
        sigma_k = sigma_per_cluster[cluster_id_np]
        noise = rng_k_init.normal(0.0, 1.0, size=(n_total, 2)) * sigma_k[:, None]
        pos_K[k, movable_idx, :] = (anchor_pos_k[movable_idx]
                                     + noise[movable_idx]).astype(np.float32)

    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    rng_top = np.random.default_rng(seed + 33333)
    n_top = min(top_n, M)
    step = grid_radius / max(R, 1)

    for m in order:
        anchor_m_K = anchors_K[:, cluster_id_np[m], :].astype(np.float32)
        candidates_K = (anchor_m_K[:, None, :]
                        + grid_offsets[None, :, :] * step)

        placed_neigh = [(nb, w) for (nb, w) in adj[m] if placed[nb]]
        if not placed_neigh:
            cand_idx = rng_top.integers(0, M, size=K)
            chosen = candidates_K[np.arange(K), cand_idx]
        else:
            neigh_idx = np.array([nb for nb, _ in placed_neigh], dtype=np.int64)
            neigh_w = np.array([w for _, w in placed_neigh], dtype=np.float32)
            neigh_pos_K = pos_K[:, neigh_idx, :]
            dx = candidates_K[:, :, None, 0] - neigh_pos_K[:, None, :, 0]
            dy = candidates_K[:, :, None, 1] - neigh_pos_K[:, None, :, 1]
            dist_l1 = np.abs(dx) + np.abs(dy)
            wdist = (dist_l1 * neigh_w[None, None, :]).sum(axis=2)
            scores = -wdist
            top_idx = np.argpartition(scores, M - n_top, axis=1)[:, M - n_top:]
            top_scores = np.take_along_axis(scores, top_idx, axis=1)
            top_scores = top_scores - top_scores.max(axis=1, keepdims=True)
            T_safe = np.maximum(temperature_K[:, None], 1e-3)
            score_scale = np.maximum(np.abs(top_scores).max(axis=1, keepdims=True), 1e-6)
            probs = np.exp(top_scores / (T_safe * score_scale))
            probs = probs / probs.sum(axis=1, keepdims=True)
            u = rng_top.random((K, 1))
            cum = np.cumsum(probs, axis=1)
            local_choice = (cum >= u).argmax(axis=1)
            cand_idx = np.take_along_axis(top_idx, local_choice[:, None], axis=1).ravel()
            k_arange = np.arange(K)
            chosen = candidates_K[k_arange, cand_idx]

        chosen[:, 0] = np.clip(chosen[:, 0], half_w[m], canvas_w - half_w[m])
        chosen[:, 1] = np.clip(chosen[:, 1], half_h[m], canvas_h - half_h[m])
        pos_K[:, m, :] = chosen
        placed[m] = True

    return _clamp_and_restore(pos_K, sizes, canvas_w, canvas_h, movable, initial_pos)


def hybrid_init(
    benchmark,
    plc=None,
    K: int = 64,
    seed: int = 42,
    spectral_frac: float = 0.0,
    temperature_range: Tuple[float, float] = (0.5, 2.0),
    spread_range: Tuple[float, float] = (0.5, 2.0),
    grid_resolution: int = 16,
    top_n: int = 10,
    max_net_size: int = 20,
) -> np.ndarray:
    """Constructive-with-spread seeds + optional spectral fraction.

    Default (``spectral_frac=0``): all K seeds are constructive with per-K
    temperature and spread weight evenly distributed over the given
    ranges. Low T = greedy, high T = random; low spread = WL-only, high
    spread = avoid crowded cells. The cross product gives a diverse pool
    of local minima at small init cost.

    The first ``ceil(spectral_frac * K)`` seeds are spectral when
    ``spectral_frac > 0``. Spectral collapses macros into a single basin
    (good for WL but hurts congestion), so it is OFF by default.

    Returns:
        pos_init: float32 numpy array [K, n_active, 2].
    """
    n_total, initial_pos, _, _, _, _ = _benchmark_arrays(benchmark)

    K_spec = int(np.ceil(spectral_frac * K)) if spectral_frac > 0 else 0
    K_spec = min(K_spec, K)
    K_cons = K - K_spec

    out = np.empty((K, n_total, 2), dtype=np.float32)

    if K_spec > 0:
        out[:K_spec] = spectral_init(
            benchmark, plc=plc, K=K_spec, seed=seed,
            max_net_size=max_net_size,
        )

    if K_cons > 0:
        if K_cons == 1:
            T_K = np.array([0.5 * (temperature_range[0] + temperature_range[1])],
                           dtype=np.float32)
            S_K = np.array([0.5 * (spread_range[0] + spread_range[1])],
                           dtype=np.float32)
        else:
            T_K = np.linspace(temperature_range[0], temperature_range[1],
                              K_cons, dtype=np.float32)
            S_K = np.linspace(spread_range[1], spread_range[0],
                              K_cons, dtype=np.float32)
        out[K_spec:] = constructive_init(
            benchmark, plc=plc, K=K_cons, seed=seed + 31337,
            temperature_K=T_K, spread_weight_K=S_K,
            grid_resolution=grid_resolution,
            top_n=top_n, max_net_size=max_net_size,
        )

    return out
