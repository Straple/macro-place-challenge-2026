"""Coordinate Descent post-process polish for hard macros.

После legalize: для каждого hard макроса пробуем окрестные позиции,
фильтруем по incremental WL (cheap), верифицируем proxy_cost (full eval).
Round-robin до сходимости. vmallela-style local refinement.
"""

from __future__ import annotations

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
