"""GPU batch multi-start gradient placer.

K параллельных runs одним vectorized tensor [K, n, 2] на GPU. Использует T4
(16GB) по максимуму: каждый шаг Adam — один backward pass, обновляющий все K
seeds одновременно. Это даёт реальный GPU speedup vs sequential.

Loss = WL_smooth + λ_d × density + λ_o × overlap, всё vectorized по K.

API:
    best_pos, all_costs = gradient_batch(benchmark, plc, K=64, ...)
    best_pos: numpy [n_total, 2]
    all_costs: numpy [K]  (proxy_cost для каждого из K seeds)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def _import_helpers():
    sd = str(Path(__file__).resolve().parent)
    if sd not in sys.path:
        sys.path.insert(0, sd)
    from analytical_seed import (
        _build_net_pin_tensors_full,
        _build_padded_net_tensors,
    )
    from clustering import cluster_macros, distribute_anchors_initial_centroid, distribute_anchors_grid
    return (_build_net_pin_tensors_full, _build_padded_net_tensors,
            cluster_macros, distribute_anchors_initial_centroid,
            distribute_anchors_grid)


def gradient_batch(benchmark, plc, K: int = 64, num_steps: int = 400,
                   seed: int = 42, device: str = "cuda",
                   anchor_strategy: str = "centroid",
                   spawn_radius_frac: float = 0.05,
                   spawn_adaptive: bool = True,
                   anchor_loss_beta_start: float = 0.0,
                   anchor_loss_beta_end: float = 0.0,
                   cluster_target: int = 0,
                   target_util: float = 0.0,
                   lambda_max: float = 0.0,
                   lambda_growth: float = 1.05,
                   lambda_start: float = 0.01,
                   gamma_frac: float = 0.05,
                   gamma_start_factor: float = 1.5,
                   gamma_end_factor: float = 0.3,
                   overlap_weight: float = 50.0,
                   overlap_w_max: float = 500000.0,
                   overlap_w_growth: float = 1.015,
                   stop_overflow: float = 0.07,
                   lr: float = 0.3,
                   lr_end_factor: float = 0.05,
                   verbose: bool = True,
                   anchor_jitter_frac: float = 0.05):
    import torch
    (_build_net_pin_tensors_full, _build_padded_net_tensors,
     cluster_macros, distribute_anchors_initial_centroid,
     distribute_anchors_grid) = _import_helpers()

    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("[gradient_batch] CUDA not available, fallback CPU", flush=True)
        device = "cpu"
    dev = torch.device(device)
    if verbose:
        if device == "cuda":
            print(f"[gradient_batch] GPU: {torch.cuda.get_device_name(0)} "
                  f"(K={K} parallel seeds)", flush=True)
        else:
            print(f"[gradient_batch] CPU mode (K={K})", flush=True)

    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    n_soft = benchmark.num_soft_macros
    n_active = n_total

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_min = min(canvas_w, canvas_h)

    sizes_t = benchmark.macro_sizes[:n_active].float().to(dev)
    half_w = sizes_t[:, 0] / 2.0
    half_h = sizes_t[:, 1] / 2.0
    movable = benchmark.get_movable_mask()[:n_active].to(dev)
    fixed_pos = benchmark.macro_positions[:n_active].float().to(dev)

    # Auto defaults
    if cluster_target <= 0:
        cluster_target = max(15, n_total // 30)
    if lambda_max <= 0:
        lambda_max = 100.0 if n_total < 1500 else (1000.0 if n_total < 2500 else 2000.0)
    if target_util <= 0:
        macro_areas = (benchmark.macro_sizes[:, 0]
                       * benchmark.macro_sizes[:, 1]).sum().item()
        actual_util = macro_areas / max(canvas_w * canvas_h, 1e-9)
        if n_total < 1500:
            target_util = max(0.1, min(0.95, actual_util * 0.95))
        else:
            target_util = max(0.1, min(0.95, actual_util * 1.05))

    if verbose:
        print(f"[gradient_batch] auto cluster_target={cluster_target} "
              f"lambda_max={lambda_max:.0f} target_util={target_util:.3f}",
              flush=True)

    # Cluster once (deterministic for this base seed)
    if verbose:
        print(f"[gradient_batch] clustering (Louvain target={cluster_target})...",
              flush=True)
    t0 = time.time()
    cluster_id, num_clusters, _ = cluster_macros(
        benchmark, method="louvain", seed=seed,
        max_net_size=20, target_num_clusters=cluster_target,
    )
    if verbose:
        print(f"[gradient_batch] clusters K={num_clusters} ({time.time()-t0:.2f}s)",
              flush=True)

    initial_pos_np = benchmark.macro_positions.cpu().numpy().astype(np.float64)
    movable_np = benchmark.get_movable_mask().cpu().numpy().astype(bool)

    if anchor_strategy == "centroid":
        anchors_base = distribute_anchors_initial_centroid(
            cluster_id, initial_pos_np, movable_np)
    else:
        rng_a = np.random.default_rng(seed)
        anchors_base = distribute_anchors_grid(
            num_clusters, canvas_w, canvas_h, rng_a)

    cluster_id_np = cluster_id.astype(np.int64)
    cluster_id_t = torch.tensor(cluster_id_np, dtype=torch.long, device=dev)

    # Generate K different initializations: K different anchor jitters and spawn noises
    # anchors_K [K, num_clusters, 2]: each K has its own jitter
    rng = np.random.default_rng(seed)
    anchor_jitter = canvas_min * anchor_jitter_frac
    anchors_K = np.broadcast_to(anchors_base[None, :, :],
                                (K, num_clusters, 2)).copy()
    anchors_K += rng.normal(0.0, anchor_jitter, size=(K, num_clusters, 2))

    # spawn pos: [K, n, 2]
    if spawn_adaptive:
        cluster_sizes = np.bincount(cluster_id_np, minlength=num_clusters)
        mean_size = max(1.0, float(cluster_sizes.mean()))
        sigma_per_cluster = (canvas_min * spawn_radius_frac
                             * np.sqrt(cluster_sizes / mean_size))
        sigma_per_macro = sigma_per_cluster[cluster_id_np]  # [n]
    else:
        sigma_per_macro = np.full(n_active, canvas_min * spawn_radius_frac)

    pos_init = np.zeros((K, n_active, 2), dtype=np.float32)
    for k in range(K):
        anchor_pos_k = anchors_K[k][cluster_id_np]  # [n, 2]
        noise_k = np.random.default_rng(seed + k * 1009).normal(
            0.0, 1.0, size=(n_active, 2)) * sigma_per_macro[:, None]
        pos_init[k] = anchor_pos_k + noise_k
    half_w_np = (sizes_t[:, 0] / 2.0).cpu().numpy()
    half_h_np = (sizes_t[:, 1] / 2.0).cpu().numpy()
    pos_init[..., 0] = np.clip(pos_init[..., 0],
                               half_w_np[None, :], canvas_w - half_w_np[None, :])
    pos_init[..., 1] = np.clip(pos_init[..., 1],
                               half_h_np[None, :], canvas_h - half_h_np[None, :])
    fixed_idx_np = ~movable_np
    fixed_pos_np = initial_pos_np
    pos_init[:, fixed_idx_np] = fixed_pos_np[None, fixed_idx_np]

    pos = torch.tensor(pos_init, dtype=torch.float32,
                       requires_grad=True, device=dev)

    optimizer = torch.optim.Adam([pos], lr=lr)

    # Build padded net tensors
    if verbose:
        print(f"[gradient_batch] building net tensors...", flush=True)
    t0 = time.time()
    net_macro_idx, net_pin_offsets = _build_net_pin_tensors_full(benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets)
    if padded is None:
        raise RuntimeError("no nets")
    macro_idx_p, offsets_p, mask_p = padded
    macro_idx_p = macro_idx_p.to(dev)         # [num_nets, max_pins]
    offsets_p = offsets_p.to(dev)             # [num_nets, max_pins, 2]
    mask_p = mask_p.to(dev)                   # [num_nets, max_pins]
    num_nets = macro_idx_p.shape[0]
    if verbose:
        print(f"[gradient_batch] {num_nets} nets, max_pins={macro_idx_p.shape[1]} "
              f"({time.time()-t0:.2f}s)", flush=True)

    # Sizes pairs for hard overlap
    sizes_hard = sizes_t[:n_hard]
    sizes_x_pair = (sizes_hard[:, 0:1] + sizes_hard[:, 0].unsqueeze(0)) * 0.5  # [nh, nh]
    sizes_y_pair = (sizes_hard[:, 1:2] + sizes_hard[:, 1].unsqueeze(0)) * 0.5
    eye_mask = (1.0 - torch.eye(n_hard, dtype=torch.float32, device=dev))      # [nh, nh]

    # Density grid
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    sigma_dx = cell_w
    sigma_dy = cell_h
    grid_x = torch.arange(grid_cols, dtype=torch.float32, device=dev) * cell_w + cell_w / 2  # [ncols]
    grid_y = torch.arange(grid_rows, dtype=torch.float32, device=dev) * cell_h + cell_h / 2  # [nrows]
    macro_areas = sizes_t[:, 0] * sizes_t[:, 1]  # [n]
    cell_capacity = cell_w * cell_h

    gamma_base = canvas_min * gamma_frac
    half_w_t = half_w
    half_h_t = half_h
    fixed_mask = ~movable
    density_weight = lambda_start
    cur_overlap_w = overlap_weight

    if verbose:
        print(f"[gradient_batch] starting {num_steps} iters...", flush=True)
    t_loop = time.time()

    for step in range(num_steps):
        progress = step / max(1, num_steps - 1)
        gamma_factor = gamma_start_factor + (gamma_end_factor - gamma_start_factor) * progress
        gamma = gamma_base * gamma_factor

        cur_lr = lr * (1.0 + (lr_end_factor - 1.0) * progress)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()

        # WL smooth (batched over K)
        # pin_xy [K, num_nets, max_pins, 2] = pos[K, macro_idx_p, :] + offsets_p[None, ...]
        pin_xy = pos[:, macro_idx_p, :] + offsets_p[None, ...]
        x = pin_xy[..., 0]   # [K, num_nets, max_pins]
        y = pin_xy[..., 1]
        neg_inf = torch.finfo(pos.dtype).min
        x_for_max = torch.where(mask_p[None, :, :], x, x.new_full((), neg_inf))
        x_for_min = torch.where(mask_p[None, :, :], -x, x.new_full((), neg_inf))
        y_for_max = torch.where(mask_p[None, :, :], y, y.new_full((), neg_inf))
        y_for_min = torch.where(mask_p[None, :, :], -y, y.new_full((), neg_inf))
        max_x = gamma * torch.logsumexp(x_for_max / gamma, dim=2)   # [K, num_nets]
        min_x = -gamma * torch.logsumexp(x_for_min / gamma, dim=2)
        max_y = gamma * torch.logsumexp(y_for_max / gamma, dim=2)
        min_y = -gamma * torch.logsumexp(y_for_min / gamma, dim=2)
        wl_K = ((max_x - min_x) + (max_y - min_y)).sum(dim=1)        # [K]
        wl_total = wl_K.sum()

        # Density (batched)
        # bell_x [K, n, ncols], bell_y [K, n, nrows]
        dx = pos[..., 0:1] - grid_x[None, None, :]   # [K, n, ncols]
        dy = pos[..., 1:2] - grid_y[None, None, :]   # [K, n, nrows]
        bell_x = torch.exp(-(dx * dx) / (2 * sigma_dx * sigma_dx))
        bell_y = torch.exp(-(dy * dy) / (2 * sigma_dy * sigma_dy))
        norm_x = bell_x / bell_x.sum(dim=2, keepdim=True).clamp_min(1e-12)
        norm_y = bell_y / bell_y.sum(dim=2, keepdim=True).clamp_min(1e-12)
        # cell_density [K, nrows, ncols] = sum macro_areas * norm_y * norm_x over n
        cell_density = (macro_areas[None, :, None, None]
                        * norm_y[:, :, :, None]
                        * norm_x[:, :, None, :]).sum(dim=1)
        excess = (cell_density / cell_capacity - target_util).clamp_min(0.0)
        dpen_K = (excess * excess).sum(dim=(1, 2))   # [K]
        dpen_total = dpen_K.sum()

        # Overlap (only between hard pairs, batched over K)
        pos_hard = pos[:, :n_hard, :]   # [K, nh, 2]
        diff_x = pos_hard[:, :, 0:1] - pos_hard[:, :, 0].unsqueeze(1)   # [K, nh, nh]
        diff_y = pos_hard[:, :, 1:2] - pos_hard[:, :, 1].unsqueeze(1)
        ovlap_x = torch.relu(sizes_x_pair[None, :, :] - torch.abs(diff_x))
        ovlap_y = torch.relu(sizes_y_pair[None, :, :] - torch.abs(diff_y))
        ovlap_area = ovlap_x * ovlap_y * eye_mask[None, :, :]
        # gauss_overlap form (best from prior experiments)
        sigma_x_sq = sizes_x_pair * sizes_x_pair + 1e-6
        sigma_y_sq = sizes_y_pair * sizes_y_pair + 1e-6
        gauss = torch.exp(-(diff_x * diff_x / sigma_x_sq[None, :, :]
                            + diff_y * diff_y / sigma_y_sq[None, :, :]))
        overlap_K = ((gauss + 5.0 * ovlap_area) * eye_mask[None, :, :]).sum(dim=(1, 2)) * 0.5
        overlap_total = overlap_K.sum()

        loss = wl_total + density_weight * dpen_total + cur_overlap_w * overlap_total
        loss.backward()
        optimizer.step()

        # Clamp pos and zero gradients on fixed
        with torch.no_grad():
            pos[..., 0].clamp_(min=half_w_t.min().item(),
                               max=canvas_w - half_w_t.min().item())
            pos[..., 1].clamp_(min=half_h_t.min().item(),
                               max=canvas_h - half_h_t.min().item())
            # Per-macro clamp (more accurate)
            pos[..., 0] = torch.clamp(pos[..., 0],
                                      half_w_t[None, :].expand(K, -1),
                                      (canvas_w - half_w_t[None, :].expand(K, -1)))
            pos[..., 1] = torch.clamp(pos[..., 1],
                                      half_h_t[None, :].expand(K, -1),
                                      (canvas_h - half_h_t[None, :].expand(K, -1)))
            # restore fixed
            if fixed_mask.any():
                pos.data[:, fixed_mask, :] = fixed_pos[fixed_mask][None, :, :].expand(K, -1, -1)

            mean_dpen_per_K = float(dpen_K.mean().item())
            mean_ovlap_per_K = float(overlap_K.mean().item())
        if mean_dpen_per_K > stop_overflow:
            density_weight = min(density_weight * lambda_growth, lambda_max)
        # Overlap weight grows aggressively until last 30% iters
        if progress < 0.7:
            cur_overlap_w = min(cur_overlap_w * overlap_w_growth, overlap_w_max)
        else:
            # Last 30% — bigger boost to push overlap to 0
            cur_overlap_w = min(cur_overlap_w * 1.05, overlap_w_max)

        if verbose and (step + 1) % 100 == 0:
            print(f"[gradient_batch] step={step+1}/{num_steps} "
                  f"wl={float(wl_K.mean()):.3f} dpen={mean_dpen_per_K:.3f} "
                  f"ovrlp={mean_ovlap_per_K:.3f} λ_d={density_weight:.2f} "
                  f"λ_o={cur_overlap_w:.1f}", flush=True)

    if verbose:
        print(f"[gradient_batch] {num_steps} steps in {time.time()-t_loop:.1f}s",
              flush=True)
    return pos.detach().cpu().numpy(), {
        "wl_K": wl_K.detach().cpu().numpy(),
        "dpen_K": dpen_K.detach().cpu().numpy(),
        "overlap_K": overlap_K.detach().cpu().numpy(),
    }
