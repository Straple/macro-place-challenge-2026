"""Coordinate Descent post-process polish for hard macros.

После legalize: для каждого hard макроса пробуем окрестные позиции,
фильтруем по incremental WL (cheap), верифицируем proxy_cost (full eval).
Round-robin до сходимости. vmallela-style local refinement.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch


def _build_macro_net_membership(benchmark):
    """For each hard macro i, return list of (net_idx, pin_slot) where slot
    is the index in net_pin_nodes[net_idx] for this macro's pin."""
    n_hard = int(benchmark.num_hard_macros)
    macro_nets = [[] for _ in range(n_hard)]
    if not benchmark.net_pin_nodes:
        return macro_nets
    for n_idx, t in enumerate(benchmark.net_pin_nodes):
        owners = t[:, 0].cpu().numpy()
        for j, own in enumerate(owners):
            if 0 <= own < n_hard:
                macro_nets[int(own)].append((n_idx, j))
    return macro_nets


def _build_initial_pin_positions(benchmark, pos_full):
    """For each net, current pin (x, y) positions."""
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    if not benchmark.net_pin_nodes:
        return [], []
    macro_pin_offsets = []
    for i in range(n_hard):
        t = benchmark.macro_pin_offsets[i]
        if t.numel() > 0:
            macro_pin_offsets.append(t.cpu().numpy().astype(np.float64))
        else:
            macro_pin_offsets.append(np.zeros((0, 2), dtype=np.float64))
    if benchmark.port_positions.numel() > 0:
        port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
    else:
        port_pos = np.zeros((0, 2), dtype=np.float64)

    net_pin_x = []
    net_pin_y = []
    for t in benchmark.net_pin_nodes:
        owners = t[:, 0].cpu().numpy().astype(np.int64)
        slots = t[:, 1].cpu().numpy().astype(np.int64)
        xs = np.zeros(len(owners), dtype=np.float64)
        ys = np.zeros(len(owners), dtype=np.float64)
        for j in range(len(owners)):
            own = int(owners[j])
            slot = int(slots[j])
            if own < n_hard:
                if slot < len(macro_pin_offsets[own]):
                    ox, oy = macro_pin_offsets[own][slot]
                else:
                    ox, oy = 0.0, 0.0
                xs[j] = pos_full[own, 0] + ox
                ys[j] = pos_full[own, 1] + oy
            elif own < n_total:
                xs[j] = pos_full[own, 0]
                ys[j] = pos_full[own, 1]
            else:
                pi = own - n_total
                if pi < port_pos.shape[0]:
                    xs[j] = port_pos[pi, 0]
                    ys[j] = port_pos[pi, 1]
        net_pin_x.append(xs)
        net_pin_y.append(ys)
    return net_pin_x, net_pin_y, macro_pin_offsets


def _net_hpwl(xs, ys):
    if len(xs) == 0:
        return 0.0
    return (float(xs.max() - xs.min()) + float(ys.max() - ys.min()))


def cd_polish(benchmark, plc, pos_full: np.ndarray,
              rounds: int = 3,
              step_factors: tuple = (1.0, 0.5, 0.25),
              n_directions: int = 8,
              verbose: bool = False) -> tuple[np.ndarray, float]:
    """Round-robin CD polish with incremental WL filter + proxy verify."""
    from macro_place.objective import compute_proxy_cost

    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos = pos_full.astype(np.float64).copy()

    macro_nets = _build_macro_net_membership(benchmark)
    if benchmark.net_pin_nodes:
        net_pin_x, net_pin_y, macro_pin_offsets = _build_initial_pin_positions(
            benchmark, pos)
    else:
        net_pin_x, net_pin_y, macro_pin_offsets = [], [], []

    def _local_hpwl(macro_id: int) -> float:
        total = 0.0
        for n_idx, _slot in macro_nets[macro_id]:
            total += _net_hpwl(net_pin_x[n_idx], net_pin_y[n_idx])
        return total

    def _local_hpwl_with_move(macro_id: int, dx: float, dy: float) -> float:
        total = 0.0
        for n_idx, slot in macro_nets[macro_id]:
            xs = net_pin_x[n_idx].copy()
            ys = net_pin_y[n_idx].copy()
            xs[slot] += dx
            ys[slot] += dy
            total += _net_hpwl(xs, ys)
        return total

    def _commit_move(macro_id: int, dx: float, dy: float):
        for n_idx, slot in macro_nets[macro_id]:
            net_pin_x[n_idx][slot] += dx
            net_pin_y[n_idx][slot] += dy

    def _proxy_at(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def _has_new_overlap_with(i: int, x: float, y: float) -> bool:
        hw_i = half_w[i]
        hh_i = half_h[i]
        for j in range(n_hard):
            if j == i:
                continue
            dx = abs(x - pos[j, 0])
            dy = abs(y - pos[j, 1])
            if dx < hw_i + half_w[j] - 1e-9 and dy < hh_i + half_h[j] - 1e-9:
                return True
        return False

    base_proxy, base_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[cd_polish] starting proxy={base_proxy:.4f} "
              f"ovrlp={base_ovrlp}", flush=True)
    accept_threshold_overlap = base_ovrlp

    for r in range(rounds):
        sf = step_factors[min(r, len(step_factors) - 1)]
        sw = cell_w * sf
        sh = cell_h * sf
        if n_directions == 4:
            offsets = [(-sw, 0.0), (+sw, 0.0), (0.0, -sh), (0.0, +sh)]
        else:
            offsets = [(dx, dy) for dx in (-sw, 0.0, +sw) for dy in (-sh, 0.0, +sh)
                       if not (dx == 0.0 and dy == 0.0)]
        improvements = 0
        proxy_calls = 0
        skipped_by_local_hpwl = 0
        round_start = time.time()
        for i in range(n_hard):
            if fixed[i]:
                continue
            best_dx, best_dy = 0.0, 0.0
            best_proxy_local = base_proxy
            for dx, dy in offsets:
                nx = pos[i, 0] + dx
                ny = pos[i, 1] + dy
                if nx - half_w[i] < -1e-6 or nx + half_w[i] > canvas_w + 1e-6:
                    continue
                if ny - half_h[i] < -1e-6 or ny + half_h[i] > canvas_h + 1e-6:
                    continue
                if _has_new_overlap_with(i, nx, ny):
                    continue
                pos[i, 0] = nx
                pos[i, 1] = ny
                p_try, o_try = _proxy_at(pos)
                proxy_calls += 1
                pos[i, 0] -= dx
                pos[i, 1] -= dy
                if o_try > accept_threshold_overlap:
                    continue
                if p_try < best_proxy_local - 1e-6:
                    best_proxy_local = p_try
                    best_dx, best_dy = dx, dy
            if best_dx == 0.0 and best_dy == 0.0:
                continue
            pos[i, 0] += best_dx
            pos[i, 1] += best_dy
            base_proxy = best_proxy_local
            _commit_move(i, best_dx, best_dy)
            improvements += 1
        elapsed = time.time() - round_start
        if verbose:
            print(f"[cd_polish] round {r+1}/{rounds} sf={sf:.2f}: "
                  f"{improvements}/{n_hard} improvements "
                  f"({proxy_calls} proxy calls, "
                  f"{skipped_by_local_hpwl} skipped by local HPWL filter) "
                  f"in {elapsed:.1f}s, proxy={base_proxy:.4f}", flush=True)
        if improvements == 0:
            break

    final_proxy, final_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[cd_polish] final proxy={final_proxy:.4f} ovrlp={final_ovrlp}",
              flush=True)
    return pos, final_proxy


def _kmeans_clusters_simple(positions: np.ndarray, n_clusters: int,
                              max_iter: int = 30,
                              seed: int = 42) -> np.ndarray:
    """Simple k-means returning cluster_id per macro. positions: [N, 2]."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    if n_clusters >= n:
        return np.arange(n, dtype=np.int64)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = positions[init_idx].copy()
    for _ in range(max_iter):
        dists = np.linalg.norm(
            positions[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for c in range(n_clusters):
            mask_c = labels == c
            if mask_c.any():
                new_centers[c] = positions[mask_c].mean(axis=0)
        if np.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return labels.astype(np.int64)


def cluster_polish_gpu(benchmark, plc, pos_full: np.ndarray,
                        proxy_pkgs: dict,
                        n_clusters: int = 30,
                        n_rounds: int = 4,
                        sf_list: tuple = (0.5, 0.25, 0.125, 0.0625),
                        n_grid: int = 5,
                        verbose: bool = False,
                        time_budget: float = 0.0,
                        proxy_chunk_n: int = 32) -> tuple[np.ndarray, float]:
    """Cluster-aware CD polish. Move clusters of hard macros as rigid units.

    Pipeline:
      1. k-means cluster hard macros by spatial position.
      2. For each cluster, try grid of (dx, dy) shifts at sf ∈ sf_list.
      3. Check inter-cluster overlap (intra-cluster preserved by rigid shift).
      4. GPU-batched proxy eval for each shift candidate.
      5. Accept best valid shift, commit, iterate.
      6. Final TILOS verify gates whole-run acceptance.
    """
    sys_path_added = False
    try:
        from gpu_proxy import gpu_proxy_batched
    except ImportError:
        from pathlib import Path as _P
        import sys as _s
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        sys_path_added = True
        from gpu_proxy import gpu_proxy_batched
    from macro_place.objective import compute_proxy_cost

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos = pos_full.astype(np.float64).copy()
    pos_orig = pos.copy()

    sizes_t = benchmark.macro_sizes[:n_total].float().to(device)
    edges_pkg = proxy_pkgs["edges_pkg"]
    smooth_matrices = proxy_pkgs["smooth_matrices"]
    routing_consts = proxy_pkgs["routing_consts"]
    wl_pkg = proxy_pkgs["wl_pkg"]

    from analytical_seed import (_build_net_pin_tensors_full,
                                  _build_padded_net_tensors)
    net_macro_idx, net_pin_offsets_lst = _build_net_pin_tensors_full(
        benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets_lst)
    if padded is None:
        macro_idx_p = torch.zeros((1, 1), dtype=torch.long, device=device)
        offsets_p = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        mask_p = torch.zeros((1, 1), dtype=torch.bool, device=device)
        num_nets_used = 0
    else:
        macro_idx_p, offsets_p, mask_p = padded
        macro_idx_p = macro_idx_p.to(device)
        offsets_p = offsets_p.to(device)
        mask_p = mask_p.to(device)
        num_nets_used = int(macro_idx_p.shape[0])

    def _proxy_at(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def _gpu_proxy(pos_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            single = pos_tensor.unsqueeze(0)
            p, _ = gpu_proxy_batched(
                single, sizes_t, macro_idx_p, offsets_p, mask_p,
                canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                n_hard=n_hard,
                edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                routing_consts=routing_consts, wl_pkg=wl_pkg,
                chunk_n=proxy_chunk_n)
        return float(p[0].item())

    base_proxy_orig, base_ovrlp = _proxy_at(pos)
    accept_threshold_overlap = base_ovrlp
    if verbose:
        print(f"[cluster_polish] start proxy={base_proxy_orig:.4f} "
              f"ovrlp={base_ovrlp} (n_clusters={n_clusters} n_rounds={n_rounds})",
              flush=True)

    movable = np.array([i for i in range(n_hard) if not fixed[i]],
                        dtype=np.int64)
    if len(movable) < 2:
        return pos_orig, base_proxy_orig
    cluster_ids = _kmeans_clusters_simple(pos[movable], n_clusters)
    cluster_to_macros: dict = {}
    for local_i, c in enumerate(cluster_ids):
        cluster_to_macros.setdefault(int(c), []).append(int(movable[local_i]))

    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    t_start = time.time()
    base_gpu = _gpu_proxy(pos_t)

    for r in range(n_rounds):
        if time_budget > 0 and (time.time() - t_start) >= time_budget:
            if verbose:
                print(f"[cluster_polish] time budget {time_budget:.0f}s "
                      f"reached at round {r}", flush=True)
            break
        sf = sf_list[min(r, len(sf_list) - 1)]
        sw = cell_w * sf
        sh = cell_h * sf
        offsets = [(dx, dy)
                    for dx in np.linspace(-sw, sw, n_grid)
                    for dy in np.linspace(-sh, sh, n_grid)
                    if not (dx == 0.0 and dy == 0.0)]
        n_dirs = len(offsets)
        offsets_np = np.array(offsets, dtype=np.float64)
        offsets_t = torch.tensor(offsets, dtype=torch.float32,
                                  device=device)

        round_improvements = 0
        for c, macros_in_c in cluster_to_macros.items():
            if not macros_in_c:
                continue
            ids = np.array(macros_in_c, dtype=np.int64)
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)

            cand = pos_t.unsqueeze(0).expand(n_dirs, -1, -1).contiguous()
            base_xy = pos_t[ids_t]
            for d_idx in range(n_dirs):
                cand[d_idx, ids_t] = base_xy + offsets_t[d_idx]

            valid_mask = np.ones(n_dirs, dtype=bool)
            others_mask = np.ones(n_hard, dtype=bool)
            others_mask[ids] = False
            others_mask[fixed[:n_hard]] = others_mask[fixed[:n_hard]]
            others_idx = np.where(others_mask[:n_hard])[0]
            other_pos = pos[others_idx]
            other_hw = half_w[others_idx]
            other_hh = half_h[others_idx]
            for d_idx in range(n_dirs):
                shifted = pos[ids] + offsets_np[d_idx]
                if (shifted[:, 0] - half_w[ids] < -1e-6).any() or \
                   (shifted[:, 0] + half_w[ids] > canvas_w + 1e-6).any() or \
                   (shifted[:, 1] - half_h[ids] < -1e-6).any() or \
                   (shifted[:, 1] + half_h[ids] > canvas_h + 1e-6).any():
                    valid_mask[d_idx] = False
                    continue
                bad = False
                for k in range(len(ids)):
                    dxs = np.abs(shifted[k, 0] - other_pos[:, 0])
                    dys = np.abs(shifted[k, 1] - other_pos[:, 1])
                    overlap_x = dxs < (half_w[ids[k]] + other_hw - 1e-9)
                    overlap_y = dys < (half_h[ids[k]] + other_hh - 1e-9)
                    if (overlap_x & overlap_y).any():
                        bad = True
                        break
                if bad:
                    valid_mask[d_idx] = False
            if not valid_mask.any():
                continue

            with torch.no_grad():
                proxy_K, _comp = gpu_proxy_batched(
                    cand, sizes_t, macro_idx_p, offsets_p, mask_p,
                    canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                    n_hard=n_hard,
                    edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                    routing_consts=routing_consts, wl_pkg=wl_pkg,
                    chunk_n=proxy_chunk_n)
            proxy_arr = proxy_K.cpu().numpy()
            valid_proxies = np.where(valid_mask, proxy_arr, np.inf)
            best_d = int(np.argmin(valid_proxies))
            if valid_proxies[best_d] >= base_gpu - 1e-6:
                continue
            dx, dy = float(offsets_np[best_d, 0]), float(offsets_np[best_d, 1])
            pos[ids, 0] += dx
            pos[ids, 1] += dy
            for i in ids:
                pos_t[int(i), 0] = float(pos[int(i), 0])
                pos_t[int(i), 1] = float(pos[int(i), 1])
            base_gpu = float(valid_proxies[best_d])
            round_improvements += 1

        if verbose:
            print(f"[cluster_polish] round {r+1}/{n_rounds} sf={sf:.4f}: "
                  f"{round_improvements}/{len(cluster_to_macros)} clusters "
                  f"shifted; gpu_base={base_gpu:.4f} "
                  f"elapsed={time.time()-t_start:.1f}s", flush=True)
        if round_improvements == 0:
            break

    final_proxy, final_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[cluster_polish] final TILOS proxy={final_proxy:.4f} "
              f"ovrlp={final_ovrlp}", flush=True)
    if final_proxy >= base_proxy_orig - 1e-6 or final_ovrlp > base_ovrlp:
        if verbose:
            print(f"[cluster_polish] REVERT: not better than orig "
                  f"{base_proxy_orig:.4f}", flush=True)
        return pos_orig, base_proxy_orig
    return pos, final_proxy


def pair_swap_polish_gpu(benchmark, plc, pos_full: np.ndarray,
                          proxy_pkgs: dict,
                          n_neighbors: int = 10,
                          n_rounds: int = 5,
                          verbose: bool = False,
                          time_budget: float = 0.0,
                          proxy_chunk_n: int = 32,
                          chunk_pairs: int = 256,
                          rank_mode: str = "proxy",
                          rank_cong_weight: float = 1.0,
                          rank_wl_weight: float = 1.0,
                          rank_dens_weight: float = 0.5,
                          lahc_length: int = 0,
                          neighbor_mode: str = "spatial",
                          ) -> tuple[np.ndarray, float]:
    """Pair-swap polish: for each macro, try swap with K-nearest neighbors.

    Pipeline per round:
      1. Build candidate pairs (i, j) — top n_neighbors closest hard macros.
      2. For each pair: swap pos[i] <-> pos[j], check overlap with
         non-{i,j} macros (vectorized geometric AABB intersection).
      3. GPU rank valid swaps via gpu_proxy_batched (batched chunk_pairs).
      4. Accept best valid swap, commit, iterate.
      5. Final TILOS verify gates whole-run acceptance.
    """
    sys_path_added = False
    try:
        from gpu_proxy import gpu_proxy_batched
    except ImportError:
        from pathlib import Path as _P
        import sys as _s
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        sys_path_added = True
        from gpu_proxy import gpu_proxy_batched
    from macro_place.objective import compute_proxy_cost

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos = pos_full.astype(np.float64).copy()
    pos_orig = pos.copy()

    sizes_t = benchmark.macro_sizes[:n_total].float().to(device)
    edges_pkg = proxy_pkgs["edges_pkg"]
    smooth_matrices = proxy_pkgs["smooth_matrices"]
    routing_consts = proxy_pkgs["routing_consts"]
    wl_pkg = proxy_pkgs["wl_pkg"]

    from analytical_seed import (_build_net_pin_tensors_full,
                                  _build_padded_net_tensors)
    net_macro_idx, net_pin_offsets_lst = _build_net_pin_tensors_full(
        benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets_lst)
    if padded is None:
        macro_idx_p = torch.zeros((1, 1), dtype=torch.long, device=device)
        offsets_p = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        mask_p = torch.zeros((1, 1), dtype=torch.bool, device=device)
        num_nets_used = 0
    else:
        macro_idx_p, offsets_p, mask_p = padded
        macro_idx_p = macro_idx_p.to(device)
        offsets_p = offsets_p.to(device)
        mask_p = mask_p.to(device)
        num_nets_used = int(macro_idx_p.shape[0])

    def _proxy_at(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def _gpu_proxy(pos_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            single = pos_tensor.unsqueeze(0)
            p, _ = gpu_proxy_batched(
                single, sizes_t, macro_idx_p, offsets_p, mask_p,
                canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                n_hard=n_hard,
                edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                routing_consts=routing_consts, wl_pkg=wl_pkg,
                chunk_n=proxy_chunk_n)
        return float(p[0].item())

    base_proxy_orig, base_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[pair_swap] start proxy={base_proxy_orig:.4f} "
              f"ovrlp={base_ovrlp} (n_hard={n_hard} "
              f"k_neighbors={n_neighbors} n_rounds={n_rounds})", flush=True)

    include_soft_swap = os.environ.get(
        "STRAPLE_BATCH_PAIR_SWAP_SOFT", "0") == "1"
    swap_search_n = n_total if include_soft_swap else n_hard
    movable_mask = ~fixed[:swap_search_n]
    movable_idx = np.where(movable_mask)[0]
    if verbose and include_soft_swap:
        print(f"[pair_swap] extended to soft macros (search_n={swap_search_n})",
              flush=True)
    if len(movable_idx) < 2:
        return pos_orig, base_proxy_orig

    connectivity_knn = None
    if neighbor_mode == "graph":
        n_mov = len(movable_idx)
        connectivity = np.zeros((n_mov, n_mov), dtype=np.int32)
        movable_set = set(int(i) for i in movable_idx)
        idx_to_local = {int(g): li for li, g in enumerate(movable_idx)}
        for net_t in benchmark.net_pin_nodes:
            owners = net_t[:, 0].cpu().numpy()
            macros_in_net = set()
            for own in owners:
                own_int = int(own)
                if own_int in movable_set:
                    macros_in_net.add(own_int)
            macros_list = list(macros_in_net)
            for i_idx, gi in enumerate(macros_list):
                li = idx_to_local[gi]
                for j_idx in range(i_idx + 1, len(macros_list)):
                    gj = macros_list[j_idx]
                    lj = idx_to_local[gj]
                    connectivity[li, lj] += 1
                    connectivity[lj, li] += 1
        np.fill_diagonal(connectivity, -1)
        connectivity_knn = np.argsort(-connectivity, axis=1)[:, :n_neighbors]
        if verbose:
            mean_share = connectivity[connectivity > 0].mean() if (
                connectivity > 0).any() else 0.0
            n_isolated = int((connectivity.max(axis=1) <= 0).sum())
            print(f"[pair_swap] graph-KNN built: mean_shared_nets="
                  f"{mean_share:.2f} isolated_macros={n_isolated}/{n_mov}",
                  flush=True)

    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    base_gpu = _gpu_proxy(pos_t)

    def _gpu_rank(pos_tensor: torch.Tensor) -> float:
        if rank_mode == "proxy":
            return _gpu_proxy(pos_tensor)
        with torch.no_grad():
            single = pos_tensor.unsqueeze(0)
            _, c = gpu_proxy_batched(
                single, sizes_t, macro_idx_p, offsets_p, mask_p,
                canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                n_hard=n_hard,
                edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                routing_consts=routing_consts, wl_pkg=wl_pkg,
                chunk_n=proxy_chunk_n)
        if rank_mode == "cong":
            return float(c["congestion"][0].item())
        return float(rank_wl_weight * c["wl"][0].item()
                     + rank_dens_weight * c["density"][0].item()
                     + rank_cong_weight * c["congestion"][0].item())

    base_rank = _gpu_rank(pos_t)
    base_proxy_running = base_proxy_orig
    best_proxy_seen = base_proxy_orig
    best_pos_seen = pos_orig.copy()
    lahc_buf: "list[float] | None" = None
    if lahc_length > 0:
        lahc_buf = [base_rank] * int(lahc_length)
    lahc_idx = 0
    t_start = time.time()

    for r in range(n_rounds):
        if time_budget > 0 and (time.time() - t_start) >= time_budget:
            if verbose:
                print(f"[pair_swap] time budget {time_budget:.0f}s reached "
                      f"at round {r}", flush=True)
            break

        if connectivity_knn is not None:
            knn_local = connectivity_knn
        else:
            movable_pos = pos[movable_idx]
            dists = np.linalg.norm(
                movable_pos[:, None, :] - movable_pos[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            knn_local = np.argsort(dists, axis=1)[:, :n_neighbors]
        pairs_seen = set()
        pairs = []
        for li, i in enumerate(movable_idx):
            for j_local in knn_local[li]:
                j = int(movable_idx[j_local])
                a, b = int(min(i, j)), int(max(i, j))
                if (a, b) in pairs_seen:
                    continue
                pairs_seen.add((a, b))
                pairs.append((a, b))
        pairs = np.array(pairs, dtype=np.int64)
        n_pairs = len(pairs)
        if n_pairs == 0:
            break

        round_accepts = 0
        for chunk_start in range(0, n_pairs, chunk_pairs):
            if time_budget > 0 and (time.time() - t_start) >= time_budget:
                break
            chunk_end = min(chunk_start + chunk_pairs, n_pairs)
            chunk = pairs[chunk_start:chunk_end]
            chunk_size = len(chunk)

            cand = pos_t.unsqueeze(0).expand(chunk_size, -1, -1).contiguous()
            i_idx = chunk[:, 0]
            j_idx = chunk[:, 1]
            i_t = torch.tensor(i_idx, dtype=torch.long, device=device)
            j_t = torch.tensor(j_idx, dtype=torch.long, device=device)
            arange = torch.arange(chunk_size, device=device)
            pos_i_t = pos_t[i_t]
            pos_j_t = pos_t[j_t]
            cand[arange, i_t] = pos_j_t
            cand[arange, j_t] = pos_i_t

            valid = np.ones(chunk_size, dtype=bool)
            for ci in range(chunk_size):
                i, j = int(chunk[ci, 0]), int(chunk[ci, 1])
                pos_i_new = pos[j].copy()
                pos_j_new = pos[i].copy()
                if (pos_i_new[0] - half_w[i] < -1e-6 or
                        pos_i_new[0] + half_w[i] > canvas_w + 1e-6 or
                        pos_i_new[1] - half_h[i] < -1e-6 or
                        pos_i_new[1] + half_h[i] > canvas_h + 1e-6 or
                        pos_j_new[0] - half_w[j] < -1e-6 or
                        pos_j_new[0] + half_w[j] > canvas_w + 1e-6 or
                        pos_j_new[1] - half_h[j] < -1e-6 or
                        pos_j_new[1] + half_h[j] > canvas_h + 1e-6):
                    valid[ci] = False
                    continue
                bad = False
                for k in range(n_hard):
                    if k == i or k == j:
                        continue
                    if (abs(pos_i_new[0] - pos[k, 0]) < half_w[i] + half_w[k] - 1e-9
                            and abs(pos_i_new[1] - pos[k, 1]) < half_h[i] + half_h[k] - 1e-9):
                        bad = True
                        break
                    if (abs(pos_j_new[0] - pos[k, 0]) < half_w[j] + half_w[k] - 1e-9
                            and abs(pos_j_new[1] - pos[k, 1]) < half_h[j] + half_h[k] - 1e-9):
                        bad = True
                        break
                if bad:
                    valid[ci] = False
            if not valid.any():
                continue

            with torch.no_grad():
                proxy_K, comp = gpu_proxy_batched(
                    cand, sizes_t, macro_idx_p, offsets_p, mask_p,
                    canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                    n_hard=n_hard,
                    edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                    routing_consts=routing_consts, wl_pkg=wl_pkg,
                    chunk_n=proxy_chunk_n)
            if rank_mode == "cong":
                rank_K = comp["congestion"]
            elif rank_mode == "blend":
                rank_K = (rank_wl_weight * comp["wl"]
                          + rank_dens_weight * comp["density"]
                          + rank_cong_weight * comp["congestion"])
            else:
                rank_K = proxy_K
            arr = rank_K.cpu().numpy()
            arr_v = np.where(valid, arr, np.inf)
            best_ci = int(np.argmin(arr_v))
            best_val = float(arr_v[best_ci])
            if lahc_buf is not None:
                threshold = lahc_buf[lahc_idx % len(lahc_buf)]
                if best_val >= threshold - 1e-6:
                    continue
            else:
                if best_val >= base_rank - 1e-6:
                    continue
            i_best, j_best = int(chunk[best_ci, 0]), int(chunk[best_ci, 1])
            tmp = pos[i_best].copy()
            pos[i_best] = pos[j_best].copy()
            pos[j_best] = tmp
            pos_t[i_best, 0] = float(pos[i_best, 0])
            pos_t[i_best, 1] = float(pos[i_best, 1])
            pos_t[j_best, 0] = float(pos[j_best, 0])
            pos_t[j_best, 1] = float(pos[j_best, 1])
            base_rank = best_val
            base_gpu = _gpu_proxy(pos_t) if rank_mode != "proxy" else best_val
            if lahc_buf is not None:
                lahc_buf[lahc_idx % len(lahc_buf)] = base_rank
                lahc_idx += 1
                if base_gpu < best_proxy_seen - 1e-9:
                    best_proxy_seen = base_gpu
                    best_pos_seen = pos.copy()
            round_accepts += 1

        if verbose:
            print(f"[pair_swap] round {r+1}/{n_rounds}: {round_accepts} swaps "
                  f"accepted; n_pairs={n_pairs} gpu_base={base_gpu:.4f} "
                  f"elapsed={time.time()-t_start:.1f}s", flush=True)
        if round_accepts == 0:
            break

    if lahc_buf is not None:
        full_seen = torch.tensor(best_pos_seen, dtype=torch.float32)
        seen_cost = compute_proxy_cost(full_seen, benchmark, plc)
        seen_proxy = float(seen_cost["proxy_cost"])
        seen_ovrlp = int(seen_cost["overlap_count"])
        full_t = torch.tensor(pos, dtype=torch.float32)
        cur_cost = compute_proxy_cost(full_t, benchmark, plc)
        cur_proxy = float(cur_cost["proxy_cost"])
        cur_ovrlp = int(cur_cost["overlap_count"])
        if seen_ovrlp <= base_ovrlp and seen_proxy <= cur_proxy:
            pos = best_pos_seen
            cost_dict = seen_cost
        else:
            cost_dict = cur_cost
        if verbose:
            print(f"[pair_swap-lahc] best_seen={seen_proxy:.4f} "
                  f"final_state={cur_proxy:.4f}", flush=True)
    else:
        full_t = torch.tensor(pos, dtype=torch.float32)
        cost_dict = compute_proxy_cost(full_t, benchmark, plc)
    final_proxy = float(cost_dict["proxy_cost"])
    final_ovrlp = int(cost_dict["overlap_count"])
    if verbose:
        wl_c = float(cost_dict.get("wirelength_cost", 0))
        d_c = float(cost_dict.get("density_cost", 0))
        cong_c = float(cost_dict.get("congestion_cost", 0))
        print(f"[pair_swap] final TILOS proxy={final_proxy:.4f} "
              f"ovrlp={final_ovrlp} (WL={wl_c:.4f} dens={d_c:.4f} "
              f"cong={cong_c:.4f})", flush=True)
    if final_proxy >= base_proxy_orig - 1e-6 or final_ovrlp > base_ovrlp:
        if verbose:
            print(f"[pair_swap] REVERT: not better than orig "
                  f"{base_proxy_orig:.4f}", flush=True)
        return pos_orig, base_proxy_orig
    return pos, final_proxy


def triple_cycle_polish_gpu(benchmark, plc, pos_full: np.ndarray,
                              proxy_pkgs: dict,
                              n_neighbors: int = 6,
                              n_rounds: int = 4,
                              verbose: bool = False,
                              time_budget: float = 0.0,
                              proxy_chunk_n: int = 32,
                              chunk_triples: int = 256) -> tuple[np.ndarray, float]:
    """3-cycle swap polish: pos[i] -> pos[j] -> pos[k] -> pos[i].

    Extension of pair_swap_polish_gpu with cyclic permutation of 3 macros.
    More degrees of freedom -> may find moves that pair-swap misses.

    Pipeline per round:
      1. Spatial KNN for each macro
      2. Generate triples (i, j, k) where j, k are near i (and j, k near each other)
      3. For each triple: candidate = cyclic swap
      4. Geometric overlap check (vectorized)
      5. GPU rank batched
      6. Accept best valid cycle
      7. Final TILOS verify gates run.
    """
    sys_path_added = False
    try:
        from gpu_proxy import gpu_proxy_batched
    except ImportError:
        from pathlib import Path as _P
        import sys as _s
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        sys_path_added = True
        from gpu_proxy import gpu_proxy_batched
    from macro_place.objective import compute_proxy_cost

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos = pos_full.astype(np.float64).copy()
    pos_orig = pos.copy()

    sizes_t = benchmark.macro_sizes[:n_total].float().to(device)
    edges_pkg = proxy_pkgs["edges_pkg"]
    smooth_matrices = proxy_pkgs["smooth_matrices"]
    routing_consts = proxy_pkgs["routing_consts"]
    wl_pkg = proxy_pkgs["wl_pkg"]

    from analytical_seed import (_build_net_pin_tensors_full,
                                  _build_padded_net_tensors)
    net_macro_idx, net_pin_offsets_lst = _build_net_pin_tensors_full(
        benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets_lst)
    if padded is None:
        macro_idx_p = torch.zeros((1, 1), dtype=torch.long, device=device)
        offsets_p = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        mask_p = torch.zeros((1, 1), dtype=torch.bool, device=device)
        num_nets_used = 0
    else:
        macro_idx_p, offsets_p, mask_p = padded
        macro_idx_p = macro_idx_p.to(device)
        offsets_p = offsets_p.to(device)
        mask_p = mask_p.to(device)
        num_nets_used = int(macro_idx_p.shape[0])

    def _proxy_at(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def _gpu_proxy(pos_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            single = pos_tensor.unsqueeze(0)
            p, _ = gpu_proxy_batched(
                single, sizes_t, macro_idx_p, offsets_p, mask_p,
                canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                n_hard=n_hard,
                edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                routing_consts=routing_consts, wl_pkg=wl_pkg,
                chunk_n=proxy_chunk_n)
        return float(p[0].item())

    base_proxy_orig, base_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[triple_cycle] start proxy={base_proxy_orig:.4f} "
              f"ovrlp={base_ovrlp} (n_hard={n_hard} "
              f"k_neighbors={n_neighbors} n_rounds={n_rounds})", flush=True)

    movable_mask = ~fixed[:n_hard]
    movable_idx = np.where(movable_mask)[0]
    if len(movable_idx) < 3:
        return pos_orig, base_proxy_orig

    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    base_gpu = _gpu_proxy(pos_t)
    t_start = time.time()

    for r in range(n_rounds):
        if time_budget > 0 and (time.time() - t_start) >= time_budget:
            if verbose:
                print(f"[triple_cycle] time budget {time_budget:.0f}s "
                      f"reached at round {r}", flush=True)
            break

        movable_pos = pos[movable_idx]
        dists = np.linalg.norm(
            movable_pos[:, None, :] - movable_pos[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        knn_local = np.argsort(dists, axis=1)[:, :n_neighbors]
        triples_seen = set()
        triples = []
        for li, i in enumerate(movable_idx):
            i_int = int(i)
            for j_local in knn_local[li]:
                j = int(movable_idx[j_local])
                if j == i_int:
                    continue
                lj = j_local
                for k_local in knn_local[lj]:
                    k = int(movable_idx[k_local])
                    if k == i_int or k == j:
                        continue
                    key = (i_int, j, k)
                    if key in triples_seen:
                        continue
                    triples_seen.add(key)
                    triples.append((i_int, j, k))
        if not triples:
            break
        triples = np.array(triples, dtype=np.int64)
        n_triples = len(triples)

        round_accepts = 0
        for chunk_start in range(0, n_triples, chunk_triples):
            if time_budget > 0 and (time.time() - t_start) >= time_budget:
                break
            chunk_end = min(chunk_start + chunk_triples, n_triples)
            chunk = triples[chunk_start:chunk_end]
            cs = len(chunk)

            cand = pos_t.unsqueeze(0).expand(cs, -1, -1).contiguous()
            i_idx = chunk[:, 0]
            j_idx = chunk[:, 1]
            k_idx = chunk[:, 2]
            i_t = torch.tensor(i_idx, dtype=torch.long, device=device)
            j_t = torch.tensor(j_idx, dtype=torch.long, device=device)
            k_t = torch.tensor(k_idx, dtype=torch.long, device=device)
            arange = torch.arange(cs, device=device)
            pos_i_t = pos_t[i_t]
            pos_j_t = pos_t[j_t]
            pos_k_t = pos_t[k_t]
            cand[arange, i_t] = pos_j_t
            cand[arange, j_t] = pos_k_t
            cand[arange, k_t] = pos_i_t

            valid = np.ones(cs, dtype=bool)
            for ci in range(cs):
                i, j, k = int(chunk[ci, 0]), int(chunk[ci, 1]), int(chunk[ci, 2])
                pos_i_new = pos[j].copy()
                pos_j_new = pos[k].copy()
                pos_k_new = pos[i].copy()
                triple_set = {i, j, k}
                bad = False
                for nm, hw, hh, npos in [
                    (i, half_w[i], half_h[i], pos_i_new),
                    (j, half_w[j], half_h[j], pos_j_new),
                    (k, half_w[k], half_h[k], pos_k_new),
                ]:
                    if (npos[0] - hw < -1e-6 or
                            npos[0] + hw > canvas_w + 1e-6 or
                            npos[1] - hh < -1e-6 or
                            npos[1] + hh > canvas_h + 1e-6):
                        bad = True
                        break
                if bad:
                    valid[ci] = False
                    continue
                triple_data = [
                    (i, half_w[i], half_h[i], pos_i_new),
                    (j, half_w[j], half_h[j], pos_j_new),
                    (k, half_w[k], half_h[k], pos_k_new),
                ]
                for ai in range(3):
                    nm_a, hw_a, hh_a, npos_a = triple_data[ai]
                    for bi in range(ai + 1, 3):
                        nm_b, hw_b, hh_b, npos_b = triple_data[bi]
                        if (abs(npos_a[0] - npos_b[0]) < hw_a + hw_b - 1e-9
                                and abs(npos_a[1] - npos_b[1])
                                < hh_a + hh_b - 1e-9):
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    valid[ci] = False
                    continue
                for nm, hw, hh, npos in triple_data:
                    for kk in range(n_hard):
                        if kk in triple_set:
                            continue
                        if (abs(npos[0] - pos[kk, 0]) < hw + half_w[kk] - 1e-9
                                and abs(npos[1] - pos[kk, 1]) < hh + half_h[kk] - 1e-9):
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    valid[ci] = False
            if not valid.any():
                continue

            with torch.no_grad():
                proxy_K, _comp = gpu_proxy_batched(
                    cand, sizes_t, macro_idx_p, offsets_p, mask_p,
                    canvas_w, canvas_h, grid_rows, grid_cols, num_nets_used,
                    n_hard=n_hard,
                    edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
                    routing_consts=routing_consts, wl_pkg=wl_pkg,
                    chunk_n=proxy_chunk_n)
            arr = proxy_K.cpu().numpy()
            arr_v = np.where(valid, arr, np.inf)
            best_ci = int(np.argmin(arr_v))
            best_val = float(arr_v[best_ci])
            if best_val >= base_gpu - 1e-6:
                continue
            i_b, j_b, k_b = (int(chunk[best_ci, 0]),
                              int(chunk[best_ci, 1]),
                              int(chunk[best_ci, 2]))
            tmp_i = pos[i_b].copy()
            pos[i_b] = pos[j_b].copy()
            pos[j_b] = pos[k_b].copy()
            pos[k_b] = tmp_i
            for m in (i_b, j_b, k_b):
                pos_t[m, 0] = float(pos[m, 0])
                pos_t[m, 1] = float(pos[m, 1])
            base_gpu = best_val
            round_accepts += 1

        if verbose:
            print(f"[triple_cycle] round {r+1}/{n_rounds}: {round_accepts} "
                  f"cycles accepted; n_triples={n_triples} "
                  f"gpu_base={base_gpu:.4f} "
                  f"elapsed={time.time()-t_start:.1f}s", flush=True)
        if round_accepts == 0:
            break

    full_t = torch.tensor(pos, dtype=torch.float32)
    cost_dict = compute_proxy_cost(full_t, benchmark, plc)
    final_proxy = float(cost_dict["proxy_cost"])
    final_ovrlp = int(cost_dict["overlap_count"])
    if verbose:
        wl_c = float(cost_dict.get("wirelength_cost", 0))
        d_c = float(cost_dict.get("density_cost", 0))
        cong_c = float(cost_dict.get("congestion_cost", 0))
        print(f"[triple_cycle] final TILOS proxy={final_proxy:.4f} "
              f"ovrlp={final_ovrlp} (WL={wl_c:.4f} dens={d_c:.4f} "
              f"cong={cong_c:.4f})", flush=True)
    if final_proxy >= base_proxy_orig - 1e-6 or final_ovrlp > base_ovrlp:
        if verbose:
            print(f"[triple_cycle] REVERT: not better than orig "
                  f"{base_proxy_orig:.4f}", flush=True)
        return pos_orig, base_proxy_orig
    return pos, final_proxy


def cd_polish_gpu_with_restart(benchmark, plc, pos_full: np.ndarray,
                                proxy_pkgs: dict,
                                restart_cycles: int = 0,
                                jitter_frac: float = 0.2,
                                jitter_step: float = 0.5,
                                jitter_seed: int = 42,
                                verbose: bool = False,
                                **cd_kwargs) -> tuple[np.ndarray, float]:
    """CD polish with random-restart cycles to escape basin floor.

    Pipeline: initial CD → if restart_cycles>0, jitter + re-CD repeatedly,
    keep best. Each restart jitter_frac of hard macros uniformly within
    ±jitter_step×cell_size.
    """
    pos_best, proxy_best = cd_polish_gpu(benchmark, plc, pos_full,
                                          proxy_pkgs=proxy_pkgs,
                                          verbose=verbose, **cd_kwargs)
    if restart_cycles <= 0:
        return pos_best, proxy_best

    rng = np.random.default_rng(jitter_seed)
    n_hard = int(benchmark.num_hard_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    cell_w = canvas_w / int(benchmark.grid_cols)
    cell_h = canvas_h / int(benchmark.grid_rows)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    n_jitter = max(1, int(jitter_frac * n_hard))

    for cycle in range(restart_cycles):
        pos_try = pos_best.copy()
        movable = [i for i in range(n_hard) if not fixed[i]]
        jitter_idx = rng.choice(movable, size=min(n_jitter, len(movable)),
                                replace=False)
        for j in jitter_idx:
            dx = rng.uniform(-cell_w * jitter_step, cell_w * jitter_step)
            dy = rng.uniform(-cell_h * jitter_step, cell_h * jitter_step)
            pos_try[j, 0] = float(np.clip(pos_try[j, 0] + dx,
                                          half_w[j], canvas_w - half_w[j]))
            pos_try[j, 1] = float(np.clip(pos_try[j, 1] + dy,
                                          half_h[j], canvas_h - half_h[j]))
        if verbose:
            print(f"[cd_restart] cycle {cycle+1}/{restart_cycles} "
                  f"jittered {len(jitter_idx)} macros (±{jitter_step:.2f} cell) "
                  f"from base proxy={proxy_best:.4f}",
                  flush=True)
        pos_cd, proxy_cd = cd_polish_gpu(benchmark, plc, pos_try,
                                          proxy_pkgs=proxy_pkgs,
                                          verbose=verbose, **cd_kwargs)
        if proxy_cd < proxy_best - 1e-6:
            if verbose:
                print(f"[cd_restart] cycle {cycle+1} IMPROVED: "
                      f"{proxy_cd:.4f} < {proxy_best:.4f}", flush=True)
            pos_best = pos_cd
            proxy_best = proxy_cd
        else:
            if verbose:
                print(f"[cd_restart] cycle {cycle+1} no improvement: "
                      f"{proxy_cd:.4f} >= {proxy_best:.4f}", flush=True)
    return pos_best, proxy_best


def cd_polish_gpu(benchmark, plc, pos_full: np.ndarray,
                  proxy_pkgs: dict,
                  rounds: int = 6,
                  step_factors: tuple = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625),
                  n_directions: int = 8,
                  topk_verify: int = 3,
                  macro_chunk: int = 64,
                  verbose: bool = False,
                  time_budget: float = 0.0,
                  proxy_chunk_n: int = 32,
                  approx_verify: bool = False,
                  approx_threshold: float = 1e-5,
                  approx_refresh_per_accept: bool = False) -> tuple[np.ndarray, float]:
    """GPU-batched CD polish.

    Two-stage filter for each hard macro i:
      Stage 1 (GPU): rank all candidate placements (one per direction) for all
                     hard macros simultaneously via gpu_proxy_batched. ~50ms
                     per chunk on T4.
      Stage 2 (TILOS): exact compute_proxy_cost only on top-K candidates per
                       macro; pick the best valid one, commit if better.

    Args:
        proxy_pkgs: dict with edges_pkg, smooth_matrices, routing_consts, wl_pkg.
        rounds: max rounds (early stop when 0 improvements).
        step_factors: per-round step multiplier of cell size.
        n_directions: 4 or 8 candidate offsets.
        topk_verify: how many top GPU-ranked candidates to verify with TILOS.
        macro_chunk: batch size for GPU evaluation (memory budget).
        time_budget: seconds; 0 = no limit (only round count limits).
        approx_verify: if True, skip TILOS verify per move and trust GPU
            ranking. Final TILOS verify gates whole-run acceptance.
        approx_threshold: in approx mode, accept move only if GPU proxy
            improves by at least this absolute amount over chunk baseline.
    """
    import torch
    sys_path_added = False
    try:
        from gpu_proxy import gpu_proxy_batched
    except ImportError:
        from pathlib import Path as _P
        import sys as _s
        _s.path.insert(0, str(_P(__file__).resolve().parent))
        sys_path_added = True
        from gpu_proxy import gpu_proxy_batched
    from macro_place.objective import compute_proxy_cost

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    n_hard = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    n_total = int(benchmark.num_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    pos = pos_full.astype(np.float64).copy()

    macro_nets = _build_macro_net_membership(benchmark)
    if benchmark.net_pin_nodes:
        net_pin_x, net_pin_y, macro_pin_offsets = _build_initial_pin_positions(
            benchmark, pos)
    else:
        net_pin_x, net_pin_y, macro_pin_offsets = [], [], []

    sizes_t = benchmark.macro_sizes[:n_total].float().to(device)

    edges_pkg = proxy_pkgs["edges_pkg"]
    smooth_matrices = proxy_pkgs["smooth_matrices"]
    routing_consts = proxy_pkgs["routing_consts"]
    wl_pkg = proxy_pkgs["wl_pkg"]

    from analytical_seed import (_build_net_pin_tensors_full,
                                  _build_padded_net_tensors)
    net_macro_idx, net_pin_offsets_lst = _build_net_pin_tensors_full(
        benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets_lst)
    if padded is None:
        macro_idx_p = torch.zeros((1, 1), dtype=torch.long, device=device)
        offsets_p = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        mask_p = torch.zeros((1, 1), dtype=torch.bool, device=device)
        num_nets_used = 0
    else:
        macro_idx_p, offsets_p, mask_p = padded
        macro_idx_p = macro_idx_p.to(device)
        offsets_p = offsets_p.to(device)
        mask_p = mask_p.to(device)
        num_nets_used = int(macro_idx_p.shape[0])

    def _proxy_at(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    def _has_new_overlap_with(i: int, x: float, y: float) -> bool:
        hw_i = half_w[i]
        hh_i = half_h[i]
        for j in range(n_hard):
            if j == i:
                continue
            dx = abs(x - pos[j, 0])
            dy = abs(y - pos[j, 1])
            if dx < hw_i + half_w[j] - 1e-9 and dy < hh_i + half_h[j] - 1e-9:
                return True
        return False

    base_proxy, base_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[cd_polish_gpu] starting proxy={base_proxy:.4f} "
              f"ovrlp={base_ovrlp} (device={device}) "
              f"approx_verify={approx_verify}", flush=True)
    accept_threshold_overlap = base_ovrlp
    pos_orig = pos.copy()
    base_proxy_orig = base_proxy

    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)

    def _gpu_proxy_at(pos_tensor: torch.Tensor) -> float:
        with torch.no_grad():
            single_pos = pos_tensor.unsqueeze(0)
            proxy_one, _ = gpu_proxy_batched(
                single_pos, sizes_t,
                macro_idx_p, offsets_p, mask_p,
                canvas_w, canvas_h, grid_rows, grid_cols,
                num_nets_used,
                n_hard=n_hard,
                edges_pkg=edges_pkg,
                smooth_matrices=smooth_matrices,
                routing_consts=routing_consts,
                wl_pkg=wl_pkg,
                chunk_n=proxy_chunk_n,
            )
        return float(proxy_one[0].item())

    t_start = time.time()
    time_exceeded = False
    for r in range(rounds):
        if time_budget > 0 and (time.time() - t_start) >= time_budget:
            if verbose:
                print(f"[cd_polish_gpu] time budget {time_budget:.0f}s reached "
                      f"at round {r}", flush=True)
            break
        sf = step_factors[min(r, len(step_factors) - 1)]
        sw = cell_w * sf
        sh = cell_h * sf
        if n_directions == 4:
            offsets = [(-sw, 0.0), (+sw, 0.0), (0.0, -sh), (0.0, +sh)]
        elif n_directions == 24:
            offsets = [(dx, dy)
                       for dx in (-2.0*sw, -sw, 0.0, +sw, +2.0*sw)
                       for dy in (-2.0*sh, -sh, 0.0, +sh, +2.0*sh)
                       if not (dx == 0.0 and dy == 0.0)]
        elif n_directions == 48:
            offsets = [(dx, dy)
                       for dx in (-3.0*sw, -2.0*sw, -sw, 0.0,
                                  +sw, +2.0*sw, +3.0*sw)
                       for dy in (-3.0*sh, -2.0*sh, -sh, 0.0,
                                  +sh, +2.0*sh, +3.0*sh)
                       if not (dx == 0.0 and dy == 0.0)]
        else:
            offsets = [(dx, dy) for dx in (-sw, 0.0, +sw)
                       for dy in (-sh, 0.0, +sh)
                       if not (dx == 0.0 and dy == 0.0)]
        n_dirs_total = len(offsets)
        offsets_np = np.array(offsets, dtype=np.float64)
        offsets_t = torch.tensor(offsets, dtype=torch.float32, device=device)

        movable_macros = [i for i in range(n_hard) if not fixed[i]]
        if not movable_macros:
            break

        round_start = time.time()
        improvements = 0
        proxy_calls = 0
        gpu_filter_time = 0.0
        verify_time = 0.0
        skip_border = 0
        skip_overlap = 0
        skip_no_improve = 0
        accept_count = 0
        proxy_call_time_sum = 0.0
        proxy_call_time_max = 0.0

        for chunk_start in range(0, len(movable_macros), macro_chunk):
            if time_budget > 0 and (time.time() - t_start) >= time_budget:
                if verbose:
                    print(f"[cd_polish_gpu] time budget {time_budget:.0f}s "
                          f"reached mid-round {r+1} at chunk "
                          f"{chunk_start}/{len(movable_macros)}", flush=True)
                time_exceeded = True
                break
            chunk_end = min(chunk_start + macro_chunk, len(movable_macros))
            chunk_ids = movable_macros[chunk_start:chunk_end]
            chunk_size = len(chunk_ids)
            n_cands = chunk_size * n_dirs_total

            t_g = time.time()
            cand_pos_full = pos_t.unsqueeze(0).expand(n_cands, -1, -1).contiguous()
            chunk_ids_t = torch.tensor(chunk_ids, dtype=torch.long, device=device)
            macro_idx_per_cand = chunk_ids_t.repeat_interleave(n_dirs_total)
            dir_idx_per_cand = torch.arange(n_dirs_total, device=device).repeat(chunk_size)

            base_xy = pos_t[macro_idx_per_cand]
            new_xy = base_xy + offsets_t[dir_idx_per_cand]
            cand_pos_full[torch.arange(n_cands, device=device), macro_idx_per_cand] = new_xy

            with torch.no_grad():
                proxy_K, _comp = gpu_proxy_batched(
                    cand_pos_full, sizes_t,
                    macro_idx_p, offsets_p, mask_p,
                    canvas_w, canvas_h, grid_rows, grid_cols,
                    num_nets_used,
                    n_hard=n_hard,
                    edges_pkg=edges_pkg,
                    smooth_matrices=smooth_matrices,
                    routing_consts=routing_consts,
                    wl_pkg=wl_pkg,
                    chunk_n=proxy_chunk_n,
                )
            proxy_K_2d = proxy_K.view(chunk_size, n_dirs_total).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            del cand_pos_full, base_xy, new_xy
            gpu_filter_time += time.time() - t_g

            t_v = time.time()
            chunk_baseline_gpu = None
            if approx_verify:
                chunk_baseline_gpu = _gpu_proxy_at(pos_t)
            for local_i, i in enumerate(chunk_ids):
                if time_budget > 0 and (time.time() - t_start) >= time_budget:
                    time_exceeded = True
                    break
                proxy_dirs = proxy_K_2d[local_i]
                order = np.argsort(proxy_dirs)
                best_dx, best_dy = 0.0, 0.0
                if approx_verify:
                    for d_idx in order:
                        dx = float(offsets_np[d_idx, 0])
                        dy = float(offsets_np[d_idx, 1])
                        nx = pos[i, 0] + dx
                        ny = pos[i, 1] + dy
                        if nx - half_w[i] < -1e-6 or nx + half_w[i] > canvas_w + 1e-6:
                            skip_border += 1
                            continue
                        if ny - half_h[i] < -1e-6 or ny + half_h[i] > canvas_h + 1e-6:
                            skip_border += 1
                            continue
                        if _has_new_overlap_with(i, nx, ny):
                            skip_overlap += 1
                            continue
                        gpu_cand = float(proxy_dirs[d_idx])
                        if gpu_cand < chunk_baseline_gpu - approx_threshold:
                            best_dx, best_dy = dx, dy
                        else:
                            skip_no_improve += 1
                        break
                else:
                    tried = 0
                    best_proxy_local = base_proxy
                    for d_idx in order:
                        if tried >= topk_verify:
                            break
                        dx = float(offsets_np[d_idx, 0])
                        dy = float(offsets_np[d_idx, 1])
                        nx = pos[i, 0] + dx
                        ny = pos[i, 1] + dy
                        if nx - half_w[i] < -1e-6 or nx + half_w[i] > canvas_w + 1e-6:
                            skip_border += 1
                            continue
                        if ny - half_h[i] < -1e-6 or ny + half_h[i] > canvas_h + 1e-6:
                            skip_border += 1
                            continue
                        if _has_new_overlap_with(i, nx, ny):
                            skip_overlap += 1
                            continue
                        tried += 1
                        pos[i, 0] = nx
                        pos[i, 1] = ny
                        t_call = time.time()
                        p_try, o_try = _proxy_at(pos)
                        dt_call = time.time() - t_call
                        proxy_call_time_sum += dt_call
                        if dt_call > proxy_call_time_max:
                            proxy_call_time_max = dt_call
                        proxy_calls += 1
                        pos[i, 0] -= dx
                        pos[i, 1] -= dy
                        if o_try > accept_threshold_overlap:
                            skip_overlap += 1
                            continue
                        if p_try < best_proxy_local - 1e-6:
                            best_proxy_local = p_try
                            best_dx, best_dy = dx, dy
                        else:
                            skip_no_improve += 1
                    if best_dx != 0.0 or best_dy != 0.0:
                        base_proxy = best_proxy_local
                if best_dx == 0.0 and best_dy == 0.0:
                    continue
                pos[i, 0] += best_dx
                pos[i, 1] += best_dy
                pos_t[i, 0] = float(pos[i, 0])
                pos_t[i, 1] = float(pos[i, 1])
                improvements += 1
                accept_count += 1
                if approx_verify and approx_refresh_per_accept:
                    chunk_baseline_gpu = _gpu_proxy_at(pos_t)
            verify_time += time.time() - t_v
            if time_exceeded:
                break

        elapsed = time.time() - round_start
        avg_call_ms = (proxy_call_time_sum / proxy_calls * 1000.0
                       if proxy_calls > 0 else 0.0)
        max_call_ms = proxy_call_time_max * 1000.0
        if verbose:
            print(f"[cd_polish_gpu] round {r+1}/{rounds} sf={sf:.4f}: "
                  f"{improvements}/{len(movable_macros)} improvements "
                  f"(proxy_calls={proxy_calls} "
                  f"avg={avg_call_ms:.0f}ms max={max_call_ms:.0f}ms; "
                  f"skip_border={skip_border} skip_overlap={skip_overlap} "
                  f"skip_no_improve={skip_no_improve} accept={accept_count}) "
                  f"gpu={gpu_filter_time:.1f}s verify={verify_time:.1f}s "
                  f"total={elapsed:.1f}s proxy={base_proxy:.4f}",
                  flush=True)
        if time_exceeded:
            break
        if improvements == 0:
            break

    final_proxy, final_ovrlp = _proxy_at(pos)
    if verbose:
        print(f"[cd_polish_gpu] final proxy={final_proxy:.4f} "
              f"ovrlp={final_ovrlp}", flush=True)
    if approx_verify and (final_proxy >= base_proxy_orig - 1e-6
                          or final_ovrlp > base_ovrlp):
        if verbose:
            print(f"[cd_polish_gpu] approx_verify REVERT: TILOS final "
                  f"{final_proxy:.4f} ovrlp={final_ovrlp} not better than "
                  f"orig {base_proxy_orig:.4f} ovrlp={base_ovrlp} "
                  f"-- returning original pos", flush=True)
        return pos_orig, base_proxy_orig
    return pos, final_proxy
