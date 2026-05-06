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
                   time_budget: float = 0.0,
                   seed: int = 42, device: str = "cuda",
                   anchor_strategy: str = "centroid",
                   spawn_radius_frac: float = 0.05,
                   spawn_adaptive: bool = True,
                   anchor_loss_beta_start: float = 0.0,
                   anchor_loss_beta_end: float = 0.0,
                   cohesion_beta_start: float = 0.0,
                   cohesion_beta_end: float = 0.0,
                   cong_weight: float = 0.0,
                   cong_top_pct: float = 0.10,
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
                   overlap_w_growth: float = 1.008,
                   stop_overflow: float = 0.07,
                   lr: float = 0.3,
                   lr_end_factor: float = 0.05,
                   verbose: bool = True,
                   anchor_jitter_frac: float = 0.05,
                   per_k_diversity: bool = False,
                   use_eplace_density: bool = False,
                   eplace_grid_size: int = 256,
                   multi_phase: bool = True,
                   phase_breaks: tuple = (0.30, 0.70),
                   phase1_gamma_mul: float = 3.0,
                   phase2_gamma_mul: float = 1.0,
                   phase3_gamma_mul: float = 0.3,
                   phase1_lambda: float = 0.001,
                   phase2_lambda_target: float = 100.0,
                   phase3_lambda_min: float = 1000.0,
                   phase1_overlap_mul: float = 0.1,
                   phase2_overlap_mul: float = 1.0,
                   phase3_overlap_mul: float = 10.0,
                   proxy_pkgs: dict = None):
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
        if multi_phase:
            # Must accommodate phase3_lambda_min (default 1000)
            lambda_max = max(phase3_lambda_min * 2.0,
                             100.0 if n_total < 1500 else
                             (1000.0 if n_total < 2500 else 2000.0))
        else:
            lambda_max = 100.0 if n_total < 1500 else (1000.0 if n_total < 2500 else 2000.0)
    if target_util <= 0:
        macro_areas = (benchmark.macro_sizes[:, 0]
                       * benchmark.macro_sizes[:, 1]).sum().item()
        actual_util = macro_areas / max(canvas_w * canvas_h, 1e-9)
        if n_total < 1500:
            target_util = max(0.1, min(0.95, actual_util * 0.95))
        else:
            target_util = max(0.1, min(0.95, actual_util * 1.05))

    # Per-K diversity hyperparams: each seed has its own (target_util, gamma,
    # density λ multiplier, congestion weight multiplier, anchor β multiplier).
    if per_k_diversity:
        rng_div = np.random.default_rng(seed + 7777)
        target_util_K_np = np.clip(
            target_util * (1.0 + rng_div.uniform(-0.15, 0.15, K)),
            0.1, 0.95).astype(np.float32)
        gamma_K_np = np.clip(rng_div.uniform(0.7, 1.4, K), 0.5, 2.0).astype(np.float32)
        # density_weight_K: multiplier applied to phase λ_d (range 0.5..2.0)
        lambda_mul_K_np = rng_div.uniform(0.5, 2.0, K).astype(np.float32)
        # congestion weight per K: 0.3..3.0 of base
        cong_mul_K_np = rng_div.uniform(0.3, 3.0, K).astype(np.float32)
        # anchor beta multiplier: 0.3..3.0
        anchor_mul_K_np = rng_div.uniform(0.3, 3.0, K).astype(np.float32)
    else:
        target_util_K_np = np.full(K, target_util, dtype=np.float32)
        gamma_K_np = np.ones(K, dtype=np.float32)
        lambda_mul_K_np = np.ones(K, dtype=np.float32)
        cong_mul_K_np = np.ones(K, dtype=np.float32)
        anchor_mul_K_np = np.ones(K, dtype=np.float32)

    if verbose:
        print(f"[gradient_batch] auto cluster_target={cluster_target} "
              f"lambda_max={lambda_max:.0f} target_util={target_util:.3f}",
              flush=True)
        if per_k_diversity:
            print(f"[gradient_batch] per-K diversity: "
                  f"target_util range [{target_util_K_np.min():.3f}, "
                  f"{target_util_K_np.max():.3f}], "
                  f"gamma_mul range [{gamma_K_np.min():.2f}, "
                  f"{gamma_K_np.max():.2f}]", flush=True)

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
    elif anchor_strategy == "center":
        # MTK-style: ВСЕ anchors в центре canvas (one point start).
        anchors_base = np.full((num_clusters, 2),
                               [canvas_w / 2, canvas_h / 2],
                               dtype=np.float64)
    else:
        rng_a = np.random.default_rng(seed)
        anchors_base = distribute_anchors_grid(
            num_clusters, canvas_w, canvas_h, rng_a)

    cluster_id_np = cluster_id.astype(np.int64)
    cluster_id_t = torch.tensor(cluster_id_np, dtype=torch.long, device=dev)

    # Generate K different initializations.
    # Diversity sources:
    #   1) anchor jitter per K (each K has its own anchors)
    #   2) Mix of strategies: 1/3 use centroid anchors, 1/3 use grid anchors,
    #      1/3 use shuffled grid (anchors permuted across clusters)
    #   3) Different spawn radii per K (varies cluster spread)
    rng = np.random.default_rng(seed)
    anchor_jitter = canvas_min * anchor_jitter_frac
    anchors_K = np.zeros((K, num_clusters, 2), dtype=np.float64)
    anchors_grid = distribute_anchors_grid(
        num_clusters, canvas_w, canvas_h, np.random.default_rng(seed + 1))
    for k in range(K):
        kind = k % 3
        rng_k = np.random.default_rng(seed + k * 1009)
        if kind == 0:
            anchors_K[k] = anchors_base
        elif kind == 1:
            anchors_K[k] = anchors_grid
        else:
            perm = rng_k.permutation(num_clusters)
            anchors_K[k] = anchors_grid[perm]
        anchors_K[k] += rng_k.normal(0.0, anchor_jitter,
                                     size=(num_clusters, 2))

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

    # Optimizer: Adam (default), Nesterov SGD (DREAMPlace standard), or AdamW
    opt_kind = os.environ.get("STRAPLE_BATCH_OPT", "adam")
    if opt_kind == "nesterov":
        optimizer = torch.optim.SGD([pos], lr=lr, momentum=0.95, nesterov=True)
        if verbose:
            print(f"[gradient_batch] Nesterov SGD lr={lr} momentum=0.95",
                  flush=True)
    elif opt_kind == "adamw":
        wd = float(os.environ.get("STRAPLE_BATCH_WEIGHT_DECAY", "1e-4"))
        optimizer = torch.optim.AdamW([pos], lr=lr, weight_decay=wd)
        if verbose:
            print(f"[gradient_batch] AdamW lr={lr} wd={wd}", flush=True)
    else:
        optimizer = torch.optim.Adam([pos], lr=lr)

    target_util_K_t = torch.tensor(target_util_K_np, device=dev).view(K, 1, 1)
    gamma_K_t = torch.tensor(gamma_K_np, device=dev)
    lambda_mul_K_t = torch.tensor(lambda_mul_K_np, device=dev)
    cong_mul_K_t = torch.tensor(cong_mul_K_np, device=dev)
    anchor_mul_K_t = torch.tensor(anchor_mul_K_np, device=dev)

    # Anchor displacement: anchors_K [K, num_clusters, 2] -> per-macro anchor [K, n, 2]
    anchor_loss_active = anchor_loss_beta_start > 0 or anchor_loss_beta_end > 0
    if anchor_loss_active:
        anchors_K_t = torch.tensor(anchors_K, dtype=torch.float32, device=dev)
        cluster_id_t_int = torch.tensor(cluster_id_np, dtype=torch.long, device=dev)
        # anchor_pos_per_macro [K, n, 2] = anchors_K[K, cluster_id_t_int, :]
        anchor_pos_per_macro = anchors_K_t[:, cluster_id_t_int, :]
        anchor_norm_factor = canvas_min * canvas_min
        movable_t = movable.float()
        if verbose:
            print(f"[gradient_batch] anchor_loss enabled: "
                  f"beta {anchor_loss_beta_start}->{anchor_loss_beta_end}",
                  flush=True)

    # Cluster cohesion: dynamic per-cluster centroid -> attract members.
    # Adaptive — anchor пересчитывается каждый step (vs fixed initial anchor).
    cohesion_active = cohesion_beta_start > 0 or cohesion_beta_end > 0
    if cohesion_active:
        cluster_id_for_scatter = torch.tensor(cluster_id_np, dtype=torch.long,
                                              device=dev)
        # Counts per cluster (same across K)
        cluster_counts = torch.bincount(cluster_id_for_scatter,
                                        minlength=num_clusters).float()
        cluster_counts_safe = cluster_counts.clamp_min(1.0).view(1, num_clusters, 1)
        cohesion_norm_factor = canvas_min * canvas_min
        movable_t_coh = movable.float()
        if verbose:
            print(f"[gradient_batch] cluster_cohesion enabled: "
                  f"beta {cohesion_beta_start}->{cohesion_beta_end}",
                  flush=True)

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

    # Congestion-aware loss: smooth bbox per net via LSE -> grid demand
    if cong_weight > 0:
        cong_smooth_sigma = (cell_w + cell_h) * 0.25
        if verbose:
            print(f"[gradient_batch] cong_weight={cong_weight}, "
                  f"top_pct={cong_top_pct}", flush=True)

    # ePlace electrostatic Poisson kernel on a HIGH-RES grid (independent of
    # benchmark's proxy_cost grid). Use eplace_grid_size for FFT resolution.
    if use_eplace_density:
        ep_n = int(eplace_grid_size)
        ep_cell_w = canvas_w / ep_n
        ep_cell_h = canvas_h / ep_n
        ep_grid_x = (torch.arange(ep_n, dtype=torch.float32, device=dev)
                     * ep_cell_w + ep_cell_w / 2)
        ep_grid_y = (torch.arange(ep_n, dtype=torch.float32, device=dev)
                     * ep_cell_h + ep_cell_h / 2)
        ep_sigma_x = ep_cell_w
        ep_sigma_y = ep_cell_h
        kx = torch.fft.fftfreq(ep_n, d=ep_cell_w, device=dev) * 2.0 * float(np.pi)
        ky = torch.fft.fftfreq(ep_n, d=ep_cell_h, device=dev) * 2.0 * float(np.pi)
        kx_g, ky_g = torch.meshgrid(kx, ky, indexing="xy")
        k_sq = kx_g * kx_g + ky_g * ky_g                     # [ep_n, ep_n]
        k_sq[0, 0] = 1.0
        inv_k_sq = 1.0 / k_sq
        inv_k_sq[0, 0] = 0.0
        total_macro_area = float(macro_areas.sum().item())
        canvas_area = canvas_w * canvas_h
        target_density_eplace = total_macro_area / canvas_area
        if verbose:
            ep_mem_mb = K * ep_n * ep_n * 8 / 1e6  # complex64 storage estimate
            print(f"[gradient_batch] ePlace density ON: grid={ep_n}x{ep_n}, "
                  f"target={target_density_eplace:.3f}, est_density_mem~{ep_mem_mb:.0f} MB",
                  flush=True)

    gamma_base = canvas_min * gamma_frac
    half_w_t = half_w
    half_h_t = half_h
    fixed_mask = ~movable
    density_weight = lambda_start
    cur_overlap_w = overlap_weight

    if verbose:
        print(f"[gradient_batch] starting {num_steps} iters...", flush=True)
    t_loop = time.time()
    # Snapshot pos every snapshot_every steps. We only keep best_idx slice
    # later, but we don't know it now -> save best-overlap-area heuristic:
    # actually save ALL K (memory ~OK), trim later in caller.
    # For K=384, n=1140, 250 snapshots: 875 MB. Fits in 64GB RAM.
    snapshot_every = int(os.environ.get("STRAPLE_BATCH_SNAPSHOT_EVERY", "1"))
    snapshots_pos = []
    snapshots_step = []

    # Per-seed plateau detection: window of `patience` steps, fire if relative
    # spread (max-min) over window for that seed is below `eps · |median|`.
    plateau_ops_enable = os.environ.get("STRAPLE_BATCH_PLATEAU_OPS", "0") == "1"
    plateau_patience = int(os.environ.get("STRAPLE_BATCH_PLATEAU_PATIENCE", "30"))
    plateau_interval = int(os.environ.get("STRAPLE_BATCH_PLATEAU_INTERVAL", "20"))
    plateau_eps = float(os.environ.get("STRAPLE_BATCH_PLATEAU_EPS", "0.005"))

    # Per-seed crossover: when seed k hits plateau, mate it with one random
    # seed from the top `elite_pct` (by current fitness) and replace pos[k]
    # with the result of cluster-aware crossover + light mutation.
    elite_pct = float(os.environ.get("STRAPLE_BATCH_GA_ELITE_PCT", "0.25"))
    mutation_rate = float(os.environ.get("STRAPLE_BATCH_GA_MUTATION_RATE", "0.01"))
    mutation_sigma = float(os.environ.get("STRAPLE_BATCH_GA_MUTATION_SIGMA", "0.005"))

    loss_history = None
    if plateau_ops_enable:
        loss_history = torch.full((plateau_patience, K), float("nan"),
                                   dtype=torch.float32, device=dev)
        if verbose:
            print(f"[gradient_batch] per-seed plateau crossover ON: "
                  f"patience={plateau_patience} interval={plateau_interval} "
                  f"eps={plateau_eps} elite_pct={elite_pct} "
                  f"mut_rate={mutation_rate} mut_sigma={mutation_sigma}",
                  flush=True)
    plateau_evt_count = 0
    seeds_recombined_total = 0
    cluster_id_for_ga = cluster_id_t
    n_hard_t_int = int(n_hard)

    step_log = os.environ.get("STRAPLE_BATCH_STEP_LOG", "0") == "1"
    fitness_history = []          # list of np arrays [K] per step
    overlap_area_history = []     # list of np arrays [K] per step

    # GPU proxy_cost as fitness ----------------------------------
    # GPU proxy as fitness is OPT-IN: between refreshes the cached proxy
    # is constant, which collapses plateau detection.  Default off — fitness
    # falls back to the per-step weighted gradient loss.  We still compute
    # proxy_history (sparse) for monitoring/plot purposes, see below.
    use_gpu_proxy_fitness = (
        proxy_pkgs is not None
        and os.environ.get("STRAPLE_BATCH_USE_GPU_PROXY", "0") == "1")
    # Recording proxy_history is OFF by default — overhead even with sparse
    # refresh (one full GPU proxy compute every proxy_interval steps), and
    # the submission only needs the final selection.  Enable explicitly
    # for plotting / diagnostics.
    record_proxy_history = (
        proxy_pkgs is not None
        and os.environ.get("STRAPLE_BATCH_RECORD_PROXY", "0") == "1")
    proxy_interval = int(os.environ.get(
        "STRAPLE_BATCH_PROXY_INTERVAL", "20"))
    proxy_K_cached = None         # last cached gpu proxy_K [K]
    proxy_history = []            # list of (step_idx, proxy_K_np) tuples
    _gpu_proxy_batched = None
    if use_gpu_proxy_fitness or record_proxy_history:
        from gpu_proxy import gpu_proxy_batched as _gpu_proxy_batched
        _gp_edges = proxy_pkgs["edges_pkg"]
        _gp_smooth = proxy_pkgs["smooth_matrices"]
        _gp_routing = proxy_pkgs["routing_consts"]
        _gp_wl = proxy_pkgs["wl_pkg"]
        if verbose:
            mode = "FITNESS" if use_gpu_proxy_fitness else "RECORD only"
            print(f"[gradient_batch] GPU proxy {mode}: "
                  f"interval={proxy_interval} step", flush=True)

    for step in range(num_steps):
        if time_budget > 0 and (time.time() - t_loop) >= time_budget:
            if verbose:
                print(f"[gradient_batch] time budget {time_budget:.0f}s reached "
                      f"at step {step}", flush=True)
            break
        if time_budget > 0:
            progress = min(1.0, (time.time() - t_loop) / time_budget)
        else:
            progress = step / max(1, num_steps - 1)

        if multi_phase:
            # 3 phases: spreading [0, b1), refining [b1, b2), settling [b2, 1)
            b1, b2 = phase_breaks
            if progress < b1:
                # Phase 1: spreading - high gamma, low lambda, low overlap
                phase_progress = progress / b1
                gamma_mul = phase1_gamma_mul + (phase2_gamma_mul - phase1_gamma_mul) * phase_progress
                phase_lambda_target = phase1_lambda + (phase2_lambda_target - phase1_lambda) * phase_progress
                overlap_mul = phase1_overlap_mul + (phase2_overlap_mul - phase1_overlap_mul) * phase_progress
                cur_phase = 1
            elif progress < b2:
                # Phase 2: refining - medium gamma, growing lambda, normal overlap
                phase_progress = (progress - b1) / max(b2 - b1, 1e-6)
                gamma_mul = phase2_gamma_mul + (phase3_gamma_mul - phase2_gamma_mul) * phase_progress
                phase_lambda_target = phase2_lambda_target + (phase3_lambda_min - phase2_lambda_target) * phase_progress
                overlap_mul = phase2_overlap_mul + (phase3_overlap_mul - phase2_overlap_mul) * phase_progress
                cur_phase = 2
            else:
                # Phase 3: settling - low gamma, high lambda, strong overlap
                gamma_mul = phase3_gamma_mul
                phase_lambda_target = phase3_lambda_min
                overlap_mul = phase3_overlap_mul
                cur_phase = 3
            gamma = gamma_base * gamma_mul
            # Snap density_weight toward phase target via exp approach
            density_weight = phase_lambda_target + (density_weight - phase_lambda_target) * 0.95
            density_weight = max(0.001, min(density_weight, lambda_max))
            cur_overlap_w_phase = cur_overlap_w * overlap_mul
        else:
            gamma_factor = gamma_start_factor + (gamma_end_factor - gamma_start_factor) * progress
            gamma = gamma_base * gamma_factor
            cur_overlap_w_phase = cur_overlap_w
            cur_phase = 0

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
        # Per-K gamma: each K seed has own gamma (gamma_K_t multiplier)
        # shape [K, 1, 1] for division on [K, num_nets, max_pins]
        # shape [K, 1] for multiplication on logsumexp result [K, num_nets]
        gamma_per_K_div = (gamma * gamma_K_t).view(K, 1, 1)
        gamma_per_K_mul = (gamma * gamma_K_t).view(K, 1)
        max_x = gamma_per_K_mul * torch.logsumexp(x_for_max / gamma_per_K_div, dim=2)
        min_x = -gamma_per_K_mul * torch.logsumexp(x_for_min / gamma_per_K_div, dim=2)
        max_y = gamma_per_K_mul * torch.logsumexp(y_for_max / gamma_per_K_div, dim=2)
        min_y = -gamma_per_K_mul * torch.logsumexp(y_for_min / gamma_per_K_div, dim=2)
        wl_K = ((max_x - min_x) + (max_y - min_y)).sum(dim=1)        # [K]
        wl_total = wl_K.sum()

        # Congestion-aware loss: net bbox dimensions (max_x - min_x, max_y - min_y)
        # already computed above. Smooth indicator if cell is inside bbox -> sum
        # over all nets -> top-k cells. Larger bbox = more cells covered = more demand.
        if cong_weight > 0:
            # bbox per net per K: x_lo, x_hi [K, num_nets]
            x_hi = max_x   # [K, num_nets]
            x_lo = min_x
            y_hi = max_y
            y_lo = min_y
            # Sigmoid indicator: cell c contributes to net n if x_lo <= cx <= x_hi (similar for y)
            # in_x [K, num_nets, ncols] = sigmoid((x_hi - cx)/σ) * sigmoid((cx - x_lo)/σ)
            cx_e = grid_x[None, None, :]   # [1, 1, ncols]
            cy_e = grid_y[None, None, :]   # [1, 1, nrows]
            in_x = (torch.sigmoid((x_hi[..., None] - cx_e) / cong_smooth_sigma)
                    * torch.sigmoid((cx_e - x_lo[..., None]) / cong_smooth_sigma))  # [K, num_nets, ncols]
            in_y = (torch.sigmoid((y_hi[..., None] - cy_e) / cong_smooth_sigma)
                    * torch.sigmoid((cy_e - y_lo[..., None]) / cong_smooth_sigma))  # [K, num_nets, nrows]
            # cell_demand [K, nrows, ncols] = sum over nets in_y[K,n,r] * in_x[K,n,c]
            cell_demand = torch.einsum("knr,knc->krc", in_y, in_x)
            flat = cell_demand.reshape(K, -1)
            top_k = max(1, int(flat.shape[-1] * cong_top_pct))
            top_vals, _ = torch.topk(flat, top_k, dim=-1)
            cong_K = top_vals.mean(dim=-1)   # [K]
            cong_total = cong_K.sum()
        else:
            cong_total = pos.new_zeros(())
            cong_K = pos.new_zeros(K)

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
        if use_eplace_density:
            # Hi-res ePlace density via chunked accumulation over n.
            # cell_density_ep [K, ep_n, ep_n] = Σ_i area_i * gauss_y(i) * gauss_x(i)
            # Memory peak per chunk: K * chunk_n * ep_n * 4 bytes (gauss_x slab)
            # For ep_n=256, K=384, chunk_n=64 -> 16 MB. Manageable.
            ep_cell_capacity = ep_cell_w * ep_cell_h
            cell_density_ep = torch.zeros(K, ep_n, ep_n, device=dev, dtype=pos.dtype)
            chunk_n = 64
            for i0 in range(0, n_active, chunk_n):
                i1 = min(i0 + chunk_n, n_active)
                pos_c = pos[:, i0:i1, :]
                area_c = macro_areas[i0:i1]
                ep_dx_c = pos_c[..., 0:1] - ep_grid_x[None, None, :]   # [K, c, ep_n]
                ep_dy_c = pos_c[..., 1:2] - ep_grid_y[None, None, :]
                ep_bx_c = torch.exp(-(ep_dx_c * ep_dx_c) / (2 * ep_sigma_x * ep_sigma_x))
                ep_by_c = torch.exp(-(ep_dy_c * ep_dy_c) / (2 * ep_sigma_y * ep_sigma_y))
                ep_nx_c = ep_bx_c / ep_bx_c.sum(dim=2, keepdim=True).clamp_min(1e-12)
                ep_ny_c = ep_by_c / ep_by_c.sum(dim=2, keepdim=True).clamp_min(1e-12)
                # contribution [K, ep_n, ep_n] = Σ_i area * ep_ny[K,c,r] * ep_nx[K,c,c2]
                cell_density_ep = cell_density_ep + (
                    area_c[None, :, None, None]
                    * ep_ny_c[:, :, :, None]
                    * ep_nx_c[:, :, None, :]).sum(dim=1)
            rho = cell_density_ep / ep_cell_capacity - target_density_eplace
            rho_fft = torch.fft.fft2(rho, dim=(-2, -1))
            phi_fft = rho_fft * inv_k_sq[None, ...]
            phi = torch.fft.ifft2(phi_fft, dim=(-2, -1)).real
            dpen_K = 0.5 * (rho * phi).sum(dim=(-2, -1))
            dpen_total = dpen_K.sum()
        else:
            excess = (cell_density / cell_capacity - target_util_K_t).clamp_min(0.0)
            dpen_K = (excess * excess).sum(dim=(1, 2))   # [K]
            dpen_total = dpen_K.sum()

        # Overlap (only between hard pairs, batched over K)
        # Form is configurable via STRAPLE_BATCH_OVERLAP_FORM env (default 'rect_quad'):
        #   'rect_quad': pure ovlap_area² — penalty ТОЛЬКО при actual overlap (no
        #                halo / dead zones around hard rects). Match MTK style.
        #   'rect_lin':  pure ovlap_area
        #   'gauss_overlap': gauss + 5×area (legacy, creates circular dead zones)
        pos_hard = pos[:, :n_hard, :]   # [K, nh, 2]
        diff_x = pos_hard[:, :, 0:1] - pos_hard[:, :, 0].unsqueeze(1)   # [K, nh, nh]
        diff_y = pos_hard[:, :, 1:2] - pos_hard[:, :, 1].unsqueeze(1)
        ovlap_x = torch.relu(sizes_x_pair[None, :, :] - torch.abs(diff_x))
        ovlap_y = torch.relu(sizes_y_pair[None, :, :] - torch.abs(diff_y))
        ovlap_area = ovlap_x * ovlap_y * eye_mask[None, :, :]
        ov_form = os.environ.get("STRAPLE_BATCH_OVERLAP_FORM", "rect_quad")
        if ov_form == "rect_quad":
            overlap_K = (ovlap_area * ovlap_area).sum(dim=(1, 2)) * 0.5
        elif ov_form == "rect_lin":
            overlap_K = ovlap_area.sum(dim=(1, 2)) * 0.5
        elif ov_form == "rect_cubic":
            overlap_K = (ovlap_area * ovlap_area * ovlap_area).sum(dim=(1, 2)) * 0.5
        elif ov_form == "rect_hinge":
            # Ignore overlaps below δ (small overlaps OK), grow quadratic above
            delta = 0.05
            hinged = torch.relu(ovlap_area - delta)
            overlap_K = (hinged * hinged).sum(dim=(1, 2)) * 0.5
        elif ov_form == "rect_log":
            # log(1 + a) — softer push for large overlaps
            overlap_K = torch.log(1.0 + ovlap_area).sum(dim=(1, 2)) * 0.5
        else:  # gauss_overlap legacy
            sigma_x_sq = sizes_x_pair * sizes_x_pair + 1e-6
            sigma_y_sq = sizes_y_pair * sizes_y_pair + 1e-6
            gauss = torch.exp(-(diff_x * diff_x / sigma_x_sq[None, :, :]
                                + diff_y * diff_y / sigma_y_sq[None, :, :]))
            overlap_K = ((gauss + 5.0 * ovlap_area) * eye_mask[None, :, :]).sum(dim=(1, 2)) * 0.5
        overlap_total = overlap_K.sum()

        # Anchor displacement loss: keep clusters tight in early phases, release in late
        if anchor_loss_active:
            beta_t = anchor_loss_beta_start * (
                (anchor_loss_beta_end / max(anchor_loss_beta_start, 1e-9)) ** progress)
            sq = (pos - anchor_pos_per_macro).pow(2).sum(dim=2)  # [K, n]
            sq = sq * movable_t[None, :]
            anchor_loss_total = (beta_t * sq.sum() / anchor_norm_factor)
        else:
            anchor_loss_total = pos.new_zeros(())

        # Cluster cohesion loss: attract members to dynamic cluster centroid.
        # MTK-style "sticky cluster" — макросы одного кластера движутся группой.
        if cohesion_active:
            # Sum positions per cluster: [K, num_clusters, 2]
            sum_pos = pos.new_zeros(K, num_clusters, 2)
            idx = cluster_id_for_scatter.view(1, n_active, 1).expand(K, n_active, 2)
            sum_pos.scatter_add_(1, idx, pos)
            centroid_dyn = sum_pos / cluster_counts_safe                # [K, num_clusters, 2]
            anchor_per_macro_dyn = centroid_dyn[:, cluster_id_for_scatter, :]
            sq_coh = (pos - anchor_per_macro_dyn).pow(2).sum(dim=2)
            sq_coh = sq_coh * movable_t_coh[None, :]
            beta_coh_t = cohesion_beta_start * (
                (cohesion_beta_end / max(cohesion_beta_start, 1e-9)) ** progress)
            cohesion_loss_total = (beta_coh_t * sq_coh.sum() / cohesion_norm_factor)
        else:
            cohesion_loss_total = pos.new_zeros(())

        # Per-K weighting: instead of summed total, sum (per-K loss * per-K mul)
        # for components that vary per seed: density, anchor, cong.
        # WL and overlap stay shared (already vectorized over K).
        if per_k_diversity:
            density_per_K = density_weight * lambda_mul_K_t * dpen_K
            cong_per_K = cong_weight * cong_mul_K_t * cong_K
            # anchor_loss_total already weighted globally; if per_k, we can't
            # easily decompose without redoing — skip for now (uses global β)
            loss = (wl_total + density_per_K.sum()
                    + cur_overlap_w_phase * overlap_total
                    + anchor_loss_total
                    + cohesion_loss_total
                    + cong_per_K.sum())
        else:
            loss = (wl_total + density_weight * dpen_total
                    + cur_overlap_w_phase * overlap_total
                    + anchor_loss_total
                    + cohesion_loss_total
                    + cong_weight * cong_total)
        loss.backward()
        # Optional gradient clipping for stability
        grad_clip = float(os.environ.get("STRAPLE_BATCH_GRAD_CLIP", "0"))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([pos], max_norm=grad_clip)
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
        if not multi_phase:
            # Legacy path: smooth growth of lambda + overlap
            if mean_dpen_per_K > stop_overflow:
                density_weight = min(density_weight * lambda_growth, lambda_max)
            cur_overlap_w = min(cur_overlap_w * overlap_w_growth, overlap_w_max)
        # In multi_phase mode, density_weight and cur_overlap_w_phase are
        # set explicitly above by phase scheduler — no exponential growth needed.

        if verbose and (step + 1) % 100 == 0:
            phase_str = f"P{cur_phase}" if multi_phase else "linear"
            print(f"[gradient_batch] step={step+1}/{num_steps} {phase_str} "
                  f"wl={float(wl_K.mean()):.3f} dpen={mean_dpen_per_K:.3f} "
                  f"ovrlp={mean_ovlap_per_K:.3f} γ={gamma:.3f} λ_d={density_weight:.2f} "
                  f"λ_o={cur_overlap_w_phase:.1f}", flush=True)

        # ----- Evolution operations: plateau escape + GA -----
        # Run after optimizer step + clamp. Track per-K loss for fitness/plateau.
        # Active in P2 + early P3 (up to 85% of time) — give children time to
        # settle after the last GA event before the final eval.
        evo_active_phase = ((not multi_phase)
                            or cur_phase == 2
                            or (cur_phase == 3 and progress < 0.85))
        with torch.no_grad():
            wl_K_d = wl_K.detach()
            dpen_K_d = dpen_K.detach()
            overlap_K_d = overlap_K.detach()
            # Default fitness — weighted gradient loss (always cheap).
            if per_k_diversity:
                fitness_grad = (wl_K_d
                                + density_weight * lambda_mul_K_t * dpen_K_d
                                + cur_overlap_w_phase * overlap_K_d
                                + cong_weight * cong_mul_K_t * cong_K.detach())
            else:
                fitness_grad = (wl_K_d
                                + density_weight * dpen_K_d
                                + cur_overlap_w_phase * overlap_K_d
                                + cong_weight * cong_K.detach())

            # If GPU proxy fitness enabled, refresh every proxy_interval
            # steps and use as primary selection signal.  Between refreshes
            # we keep the cached proxy values (placement changes slowly per
            # step so this is fine for plateau / GA use).
            do_proxy_refresh = (
                _gpu_proxy_batched is not None
                and (step % proxy_interval == 0 or proxy_K_cached is None))
            if do_proxy_refresh:
                proxy_K_now, _comp = _gpu_proxy_batched(
                    pos.detach(), sizes_t,
                    macro_idx_p, offsets_p, mask_p,
                    canvas_w, canvas_h,
                    int(benchmark.grid_rows), int(benchmark.grid_cols),
                    macro_idx_p.shape[0],
                    n_hard=n_hard,
                    edges_pkg=_gp_edges,
                    smooth_matrices=_gp_smooth,
                    routing_consts=_gp_routing,
                    wl_pkg=_gp_wl,
                )
                proxy_K_cached = proxy_K_now
                proxy_history.append(
                    (step + 1, proxy_K_now.cpu().numpy().astype(np.float32)))
            if use_gpu_proxy_fitness and proxy_K_cached is not None:
                fitness_K = proxy_K_cached
            else:
                fitness_K = fitness_grad
            fitness_cpu = fitness_K.cpu().numpy().astype(np.float32)
            fitness_history.append(fitness_cpu)
            # Pair-wise overlap area per K (raw geometry — independent of weight).
            pos_hard_now = pos[:, :n_hard, :].detach()
            dx_now = pos_hard_now[:, :, 0:1] - pos_hard_now[:, :, 0].unsqueeze(1)
            dy_now = pos_hard_now[:, :, 1:2] - pos_hard_now[:, :, 1].unsqueeze(1)
            ox = torch.relu(sizes_x_pair[None, :, :] - torch.abs(dx_now))
            oy = torch.relu(sizes_y_pair[None, :, :] - torch.abs(dy_now))
            ov_area_now = ((ox * oy * eye_mask[None, :, :])
                           .sum(dim=(1, 2)) * 0.5)
            overlap_area_history.append(ov_area_now.cpu().numpy().astype(np.float32))

            if step_log:
                f_np = fitness_cpu
                f_min = float(f_np.min())
                f_p25 = float(np.percentile(f_np, 25))
                f_med = float(np.median(f_np))
                f_p75 = float(np.percentile(f_np, 75))
                f_max = float(f_np.max())
                ov_np = ov_area_now.cpu().numpy()
                ov_min = float(ov_np.min())
                ov_med = float(np.median(ov_np))
                phase_str = f"P{cur_phase}" if multi_phase else "lin"
                print(f"[gradient_batch] step={step+1} {phase_str} "
                      f"fit[min/p25/p50/p75/max]="
                      f"{f_min:.0f}/{f_p25:.0f}/{f_med:.0f}/{f_p75:.0f}/{f_max:.0f} "
                      f"ovlp_area[min/med]={ov_min:.2f}/{ov_med:.2f}", flush=True)

        # ----- Per-seed plateau crossover -----
        # For each seed k: detect plateau on its loss trajectory over
        # `plateau_patience` steps. If stuck → mate it with one random elite
        # (top `elite_pct` by current fitness), cluster-aware crossover +
        # light mutation, write back into pos.data[k].
        if plateau_ops_enable:
            with torch.no_grad():
                slot = step % plateau_patience
                loss_history[slot] = fitness_K
                if (evo_active_phase and step >= plateau_patience
                        and (step + 1) % plateau_interval == 0):
                    spread = (loss_history.max(0).values
                              - loss_history.min(0).values)        # [K]
                    median_K = loss_history.median(0).values.abs().clamp_min(1e-6)
                    rel = spread / median_K     # [K] relative spread
                    plateau_mask = rel < plateau_eps                  # [K]
                    plateau_idx = torch.nonzero(plateau_mask, as_tuple=False).flatten()
                    n_plat = int(plateau_idx.numel())
                    if verbose:
                        print(f"[gradient_batch] plateau check @step={step+1}: "
                              f"plateau={n_plat}/{K} "
                              f"rel_spread[min/med/max]="
                              f"{float(rel.min()):.4f}/{float(rel.median()):.4f}/"
                              f"{float(rel.max()):.4f}", flush=True)
                    if n_plat > 0:
                        n_elite = max(1, int(round(K * elite_pct)))
                        sorted_idx = torch.argsort(fitness_K)
                        elite_idx = sorted_idx[:n_elite]   # [n_elite]
                        # Random elite partner per plateau seed (with replacement).
                        partner_pick = torch.randint(0, n_elite, (n_plat,),
                                                     device=dev)
                        partner_idx = elite_idx[partner_pick]   # [n_plat]

                        pos_self = pos.data[plateau_idx]          # [n_plat, n, 2]
                        pos_partner = pos.data[partner_idx]       # [n_plat, n, 2]
                        # Cluster-aware crossover: per cluster, Bernoulli(0.5)
                        # picks self vs partner. All macros of that cluster
                        # follow the choice → keeps cluster geometry intact.
                        cluster_pick = (torch.rand(n_plat, num_clusters,
                                                    device=dev) < 0.5)
                        per_macro_pick = cluster_pick[:, cluster_id_for_ga]
                        pos_new = torch.where(per_macro_pick.unsqueeze(-1),
                                               pos_self, pos_partner)

                        # Light mutation: gaussian noise + rare teleport.
                        if mutation_sigma > 0:
                            noise = (torch.randn_like(pos_new)
                                     * (mutation_sigma * canvas_min))
                            noise = noise * movable.view(1, -1, 1).float()
                            pos_new = pos_new + noise
                        if mutation_rate > 0:
                            tp_mask = ((torch.rand(n_plat, n_active, device=dev)
                                        < mutation_rate)
                                       & movable.unsqueeze(0))
                            rand_pos = torch.stack([
                                torch.rand(n_plat, n_active, device=dev) * canvas_w,
                                torch.rand(n_plat, n_active, device=dev) * canvas_h,
                            ], dim=-1)
                            pos_new = torch.where(tp_mask.unsqueeze(-1),
                                                   rand_pos, pos_new)

                        # Clamp + restore fixed
                        pos_new[..., 0] = torch.clamp(
                            pos_new[..., 0],
                            half_w_t.unsqueeze(0).expand(n_plat, -1),
                            (canvas_w - half_w_t).unsqueeze(0).expand(n_plat, -1))
                        pos_new[..., 1] = torch.clamp(
                            pos_new[..., 1],
                            half_h_t.unsqueeze(0).expand(n_plat, -1),
                            (canvas_h - half_h_t).unsqueeze(0).expand(n_plat, -1))
                        if fixed_mask.any():
                            pos_new[:, fixed_mask, :] = (
                                fixed_pos[fixed_mask][None, :, :]
                                .expand(n_plat, -1, -1))

                        pos.data[plateau_idx] = pos_new
                        # Reset Adam state for plateau seeds (old momentum is
                        # what got them stuck).
                        opt_state = optimizer.state.get(pos, {})
                        if "exp_avg" in opt_state:
                            opt_state["exp_avg"][plateau_idx] = 0
                        if "exp_avg_sq" in opt_state:
                            opt_state["exp_avg_sq"][plateau_idx] = 0
                        # Invalidate history rows for these seeds.
                        loss_history[:, plateau_idx] = float("nan")

                        plateau_evt_count += 1
                        seeds_recombined_total += n_plat
                        if verbose:
                            elite_fit = fitness_K[elite_idx]
                            print(f"[gradient_batch] plateau crossover "
                                  f"@step={step+1}: {n_plat}/{K} seeds "
                                  f"recombined with random elite "
                                  f"(top-{n_elite}, fit_min={float(elite_fit.min()):.0f})",
                                  flush=True)

        # Snapshot every snapshot_every steps (CPU copy, no autograd link)
        if (step + 1) % snapshot_every == 0 or step == 0:
            snapshots_pos.append(pos.detach().cpu().numpy().astype(np.float32))
            snapshots_step.append(step + 1)

    # Final overlap_K computation (clean, after step loop)
    with torch.no_grad():
        pos_hard = pos[:, :n_hard, :]
        diff_x = pos_hard[:, :, 0:1] - pos_hard[:, :, 0].unsqueeze(1)
        diff_y = pos_hard[:, :, 1:2] - pos_hard[:, :, 1].unsqueeze(1)
        ovlap_x = torch.relu(sizes_x_pair[None, :, :] - torch.abs(diff_x))
        ovlap_y = torch.relu(sizes_y_pair[None, :, :] - torch.abs(diff_y))
        ovlap_area_K = (ovlap_x * ovlap_y * eye_mask[None, :, :]).sum(dim=(1, 2)) * 0.5
        # ovlap_area_K[k] = total overlap area for k-th seed (smaller = better)

    if verbose:
        print(f"[gradient_batch] {num_steps} steps in {time.time()-t_loop:.1f}s",
              flush=True)
        if plateau_ops_enable:
            print(f"[gradient_batch] evolution: plateau_events="
                  f"{plateau_evt_count} seeds_recombined_total="
                  f"{seeds_recombined_total}", flush=True)
    # Add final snapshot if not already added
    if not snapshots_step or snapshots_step[-1] != step:
        snapshots_pos.append(pos.detach().cpu().numpy().astype(np.float32))
        snapshots_step.append(step)
    fitness_history_np = (np.stack(fitness_history, axis=0)
                          if fitness_history else None)
    overlap_area_history_np = (np.stack(overlap_area_history, axis=0)
                               if overlap_area_history else None)
    return pos.detach().cpu().numpy(), {
        "wl_K": wl_K.detach().cpu().numpy(),
        "dpen_K": dpen_K.detach().cpu().numpy(),
        "overlap_K": overlap_K.detach().cpu().numpy(),
        "overlap_area_K": ovlap_area_K.detach().cpu().numpy(),
        "snapshots_pos": np.stack(snapshots_pos, axis=0) if snapshots_pos else None,
        "snapshots_step": snapshots_step,
        "plateau_events": plateau_evt_count,
        "seeds_recombined_total": seeds_recombined_total,
        "fitness_history": fitness_history_np,
        "overlap_area_history": overlap_area_history_np,
        "proxy_history": (
            np.stack([p for _, p in proxy_history], axis=0)
            if proxy_history else None),
        "proxy_history_steps": [s for s, _ in proxy_history],
        "fitness_is_proxy": bool(use_gpu_proxy_fitness),
    }
