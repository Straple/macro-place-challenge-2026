"""Gradient-based placement demo (a-la DREAMPlace simplified).

Adam optimizer on smooth HPWL + density penalty. Differentiable surrogate of
proxy. Adaptive density_weight: starts low, grows when overflow stays high
(DREAMPlace's RePlAce-style update). Gamma cooling: large gamma early
(broad WL), small gamma late (sharp).

Activated via env var `STRAPLE_DEMO=gradient` in placer.py.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def _import_analytical():
    _STRAPLE_DIR = str(Path(__file__).resolve().parent)
    if _STRAPLE_DIR not in sys.path:
        sys.path.insert(0, _STRAPLE_DIR)
    from analytical_seed import (
        _build_net_pin_tensors,
        _build_net_pin_tensors_full,
        _build_padded_net_tensors,
        _smooth_hpwl_padded,
        _density_penalty,
    )
    return (_build_net_pin_tensors, _build_net_pin_tensors_full,
            _build_padded_net_tensors,
            _smooth_hpwl_padded, _density_penalty)


def gradient_demo(benchmark, plc, recorder=None, num_steps: int = 300,
                  seed: int = 42, time_budget: float = 0.0,
                  score_png: str = "", score_sample_every_s: float = 1.0):
    import torch
    from macro_place.objective import compute_proxy_cost

    (_build_net_pin_tensors, _build_net_pin_tensors_full,
     _build_padded_net_tensors,
     _smooth_hpwl_padded, _density_penalty) = _import_analytical()

    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros
    place_all = os.environ.get("STRAPLE_DEMO_PLACE_ALL", "1") != "0"
    n_active = n_total if place_all else n_hard

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_min = min(canvas_w, canvas_h)

    sizes_t = benchmark.macro_sizes[:n_active].float()
    half_w = sizes_t[:, 0] / 2.0
    half_h = sizes_t[:, 1] / 2.0
    movable = benchmark.get_movable_mask()[:n_active]
    fixed_pos = benchmark.macro_positions[:n_active].float().clone()

    rng_np = np.random.default_rng(seed)
    init_mode = os.environ.get("STRAPLE_DEMO_INIT", "center")
    pos_np = np.zeros((n_active, 2), dtype=np.float32)
    cluster_id_arr = None
    if init_mode == "random":
        pos_np[:, 0] = rng_np.uniform(half_w.numpy(), canvas_w - half_w.numpy())
        pos_np[:, 1] = rng_np.uniform(half_h.numpy(), canvas_h - half_h.numpy())
    elif init_mode == "anchor_soft":
        from clustering import cluster_macros, anchor_soft_init
        cluster_method = os.environ.get("STRAPLE_DEMO_CLUSTER_METHOD", "louvain")
        cluster_resolution = float(
            os.environ.get("STRAPLE_DEMO_CLUSTER_RESOLUTION", "1.0"))
        cluster_target_env = os.environ.get("STRAPLE_DEMO_CLUSTER_TARGET", "auto")
        cluster_max_net = int(os.environ.get("STRAPLE_DEMO_CLUSTER_MAX_NET", "20"))
        anchor_strategy = os.environ.get("STRAPLE_DEMO_ANCHOR_STRATEGY", "grid")
        spawn_radius_frac = float(
            os.environ.get("STRAPLE_DEMO_SPAWN_RADIUS_FRAC", "0.05"))
        if cluster_target_env == "auto":
            cluster_target = max(15, n_total // 30)
            print(f"[gradient_demo] auto cluster_target={cluster_target} "
                  f"(n_total={n_total})", flush=True)
        else:
            cluster_target = int(cluster_target_env)
        print(f"[gradient_demo] clustering ({cluster_method}, "
              f"target={cluster_target}, max_net={cluster_max_net})...", flush=True)
        cluster_id_arr, num_clusters, cluster_stats = cluster_macros(
            benchmark, method=cluster_method, seed=seed,
            max_net_size=cluster_max_net, resolution=cluster_resolution,
            target_num_clusters=cluster_target,
        )
        print(f"[gradient_demo] clusters K={num_clusters} "
              f"sizes(min/mean/max)="
              f"{cluster_stats['min_cluster_size']}/"
              f"{cluster_stats['mean_cluster_size']:.1f}/"
              f"{cluster_stats['max_cluster_size']}", flush=True)
        if place_all:
            cluster_id_active = cluster_id_arr
        else:
            cluster_id_active = cluster_id_arr[:n_active]
        full_init = anchor_soft_init(
            benchmark, cluster_id_arr, seed=seed,
            spawn_radius_frac=spawn_radius_frac,
            anchor_strategy=anchor_strategy,
        )
        pos_np[:] = full_init[:n_active]
    else:
        spawn_jitter = canvas_min * float(
            os.environ.get("STRAPLE_DEMO_SPAWN_JITTER", "0.005"))
        pos_np[:, 0] = canvas_w / 2.0 + rng_np.normal(0, spawn_jitter, n_active)
        pos_np[:, 1] = canvas_h / 2.0 + rng_np.normal(0, spawn_jitter, n_active)
        pos_np[:, 0] = np.clip(pos_np[:, 0],
                               half_w.numpy(), canvas_w - half_w.numpy())
        pos_np[:, 1] = np.clip(pos_np[:, 1],
                               half_h.numpy(), canvas_h - half_h.numpy())
    fixed_idx = (~movable).numpy()
    pos_np[fixed_idx] = fixed_pos.numpy()[fixed_idx]

    if cluster_id_arr is not None and recorder is not None:
        if hasattr(recorder, "set_cluster_ids"):
            recorder.set_cluster_ids(cluster_id_arr)

    pos = torch.tensor(pos_np, dtype=torch.float32, requires_grad=True)

    lr = float(os.environ.get("STRAPLE_DEMO_LR", "0.3"))
    optimizer = torch.optim.Adam([pos], lr=lr)

    use_plateau = os.environ.get("STRAPLE_DEMO_LR_PLATEAU", "0") == "1"
    plateau_factor = float(os.environ.get("STRAPLE_DEMO_PLATEAU_FACTOR", "0.5"))
    plateau_patience = int(os.environ.get("STRAPLE_DEMO_PLATEAU_PATIENCE", "30"))
    plateau_min_lr = float(os.environ.get("STRAPLE_DEMO_PLATEAU_MIN_LR", "0.001"))
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    if use_plateau:
        plateau_sched = ReduceLROnPlateau(
            optimizer, mode="min", factor=plateau_factor,
            patience=plateau_patience, threshold=1e-4,
            min_lr=plateau_min_lr,
        )

    op_on_plateau = os.environ.get("STRAPLE_DEMO_OP_ON_PLATEAU", "0") == "1"
    op_patience = int(os.environ.get("STRAPLE_DEMO_OP_PATIENCE", "50"))
    op_every = int(os.environ.get("STRAPLE_DEMO_OP_EVERY", "0"))
    op_k = int(os.environ.get("STRAPLE_DEMO_OP_K", "8"))
    op_warmup_progress = float(os.environ.get("STRAPLE_DEMO_OP_WARMUP", "0.15"))
    op_max_progress = float(os.environ.get("STRAPLE_DEMO_OP_MAX_PROGRESS", "0.7"))
    restart_on_plateau = os.environ.get("STRAPLE_DEMO_RESTART_ON_PLATEAU", "0") == "1"
    op_kinds_env = os.environ.get(
        "STRAPLE_DEMO_OPS", "teleport,swap,pull,shake,scatter_cluster")
    op_kinds = [s.strip() for s in op_kinds_env.split(",") if s.strip()]
    op_rng = np.random.default_rng(seed + 12345)
    op_events_grad = []
    best_loss = float("inf")
    no_improve_count = 0

    global_best_pos = None
    global_best_proxy = float("inf")
    restart_count = 0

    print(f"[gradient_demo] building net tensors...", flush=True)
    if place_all:
        net_macro_idx, net_pin_offsets = _build_net_pin_tensors_full(benchmark, plc)
    else:
        net_macro_idx, net_pin_offsets = _build_net_pin_tensors(benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets)
    n_nets = len(net_macro_idx)
    print(f"[gradient_demo] n_hard={n_hard} n_nets={n_nets} "
          f"canvas={canvas_w:.1f}x{canvas_h:.1f}", flush=True)

    gamma_base = canvas_min * float(os.environ.get("STRAPLE_DEMO_GAMMA_FRAC", "0.05"))
    gamma_factor_start = float(os.environ.get("STRAPLE_DEMO_GAMMA_START", "1.5"))
    gamma_factor_end = float(os.environ.get("STRAPLE_DEMO_GAMMA_END", "0.3"))

    density_weight = float(os.environ.get("STRAPLE_DEMO_LAMBDA_START", "0.01"))
    lambda_max_env = os.environ.get("STRAPLE_DEMO_LAMBDA_MAX", "auto")
    if lambda_max_env == "auto":
        if n_total < 1500:
            lambda_max = 100.0
        elif n_total < 2500:
            lambda_max = 1000.0
        else:
            lambda_max = 2000.0
        print(f"[gradient_demo] auto lambda_max={lambda_max:.0f} "
              f"(n_total={n_total})", flush=True)
    else:
        lambda_max = float(lambda_max_env)
    weight_growth = float(os.environ.get("STRAPLE_DEMO_LAMBDA_GROWTH", "1.05"))
    target_util_env = os.environ.get("STRAPLE_DEMO_TARGET_UTIL", "auto")
    if target_util_env == "auto":
        macro_areas = (benchmark.macro_sizes[:, 0]
                       * benchmark.macro_sizes[:, 1]).sum().item()
        canvas_area = canvas_w * canvas_h
        actual_util = macro_areas / max(canvas_area, 1e-9)
        if n_total < 1500:
            target_util = max(0.1, min(0.95, actual_util * 0.95))
        else:
            target_util = max(0.1, min(0.95, actual_util * 1.05))
        print(f"[gradient_demo] auto target_util={target_util:.3f} "
              f"(actual macro/canvas={actual_util:.3f})", flush=True)
    else:
        target_util = float(target_util_env)
    stop_overflow = float(os.environ.get("STRAPLE_DEMO_STOP_OVERFLOW", "0.07"))

    cong_weight = float(os.environ.get("STRAPLE_DEMO_CONG_W", "0.0"))
    cong_top_pct = float(os.environ.get("STRAPLE_DEMO_CONG_TOP_PCT", "0.1"))

    overlap_weight = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W", "1.0"))
    overlap_w_max = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W_MAX", "100000.0"))
    overlap_w_growth = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W_GROWTH", "1.01"))
    overlap_form = os.environ.get("STRAPLE_DEMO_OVERLAP_FORM", "linear")
    overlap_grow_threshold = float(
        os.environ.get("STRAPLE_DEMO_OVERLAP_GROW_THRESHOLD", "0.5"))
    use_lagrangian = os.environ.get("STRAPLE_DEMO_LAGRANGIAN", "1") != "0"
    lagrangian_rho = float(os.environ.get("STRAPLE_DEMO_RHO", "5.0"))

    lr_end_factor = float(os.environ.get("STRAPLE_DEMO_LR_END_FACTOR", "0.05"))

    sizes_hard = sizes_t[:n_hard]
    sizes_x_pair = (sizes_hard[:, 0:1] + sizes_hard[:, 0].unsqueeze(0)) * 0.5
    sizes_y_pair = (sizes_hard[:, 1:2] + sizes_hard[:, 1].unsqueeze(0)) * 0.5
    eye_mask = (1.0 - torch.eye(n_hard, dtype=torch.float32))

    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)

    cell_w_t = canvas_w / grid_cols
    cell_h_t = canvas_h / grid_rows
    cell_cx_t = (torch.arange(grid_cols, dtype=torch.float32) + 0.5) * cell_w_t
    cell_cy_t = (torch.arange(grid_rows, dtype=torch.float32) + 0.5) * cell_h_t
    cong_smooth_sigma = (cell_w_t + cell_h_t) * 0.25

    if recorder is not None:
        recorder.add(pos.detach().numpy().astype(np.float64),
                     f"gradient: init lambda={density_weight:.4f}")

    score_history = [] if (time_budget > 0 or score_png) else None
    op_events = op_events_grad
    full_template = benchmark.macro_positions.clone()

    if score_history is not None:
        full = full_template.clone()
        full[:n_active] = pos.detach()
        c0 = compute_proxy_cost(full, benchmark, plc)
        score_history.append({
            "step": 0, "elapsed": 0.0,
            "wl": float(c0["wirelength_cost"]),
            "den": float(c0["density_cost"]),
            "cong": float(c0["congestion_cost"]),
            "proxy": float(c0["proxy_cost"]),
            "ovrlp": int(c0["overlap_count"]),
            "loss": 0.0,
            "wl_smooth": 0.0,
            "dpen": 0.0,
            "overlap_term": 0.0,
            "lambda_d": density_weight,
            "lambda_o": overlap_weight,
        })

    t_start = time.time()
    last_score_t = t_start
    step = 0

    while True:
        if time_budget > 0:
            if time.time() - t_start >= time_budget:
                break
            progress = (time.time() - t_start) / time_budget
        else:
            if step >= num_steps:
                break
            progress = step / max(1, num_steps - 1)

        gamma_factor = (gamma_factor_start
                        + (gamma_factor_end - gamma_factor_start) * progress)
        gamma = gamma_base * gamma_factor

        optimizer.zero_grad()

        wl = _smooth_hpwl_padded(pos, padded, gamma) if padded is not None else pos.new_zeros(())
        dpen = _density_penalty(
            pos, sizes_t, canvas_w, canvas_h,
            grid_rows, grid_cols, target_util,
        )

        pos_hard = pos[:n_hard]
        diff_x = pos_hard[:, 0:1] - pos_hard[:, 0].unsqueeze(0)
        diff_y = pos_hard[:, 1:2] - pos_hard[:, 1].unsqueeze(0)

        if overlap_form == "gauss":
            sigma_x_sq = sizes_x_pair * sizes_x_pair + 1e-6
            sigma_y_sq = sizes_y_pair * sizes_y_pair + 1e-6
            gauss = torch.exp(-(diff_x * diff_x / sigma_x_sq
                                + diff_y * diff_y / sigma_y_sq))
            overlap_loss = (gauss * eye_mask).sum() * 0.5
            ovlap_area = torch.relu(sizes_x_pair - torch.abs(diff_x)) * \
                         torch.relu(sizes_y_pair - torch.abs(diff_y)) * eye_mask
        elif overlap_form == "coulomb":
            soft_sq = (sizes_x_pair * sizes_x_pair + sizes_y_pair * sizes_y_pair) * 0.04
            dist_sq = diff_x * diff_x + diff_y * diff_y + soft_sq
            coulomb = 1.0 / dist_sq
            overlap_loss = (coulomb * eye_mask).sum() * 0.5
            ovlap_area = torch.relu(sizes_x_pair - torch.abs(diff_x)) * \
                         torch.relu(sizes_y_pair - torch.abs(diff_y)) * eye_mask
        elif overlap_form == "gauss_overlap":
            sigma_x_sq = sizes_x_pair * sizes_x_pair + 1e-6
            sigma_y_sq = sizes_y_pair * sizes_y_pair + 1e-6
            gauss = torch.exp(-(diff_x * diff_x / sigma_x_sq
                                + diff_y * diff_y / sigma_y_sq))
            ovlap_x = torch.relu(sizes_x_pair - torch.abs(diff_x))
            ovlap_y = torch.relu(sizes_y_pair - torch.abs(diff_y))
            ovlap_area = ovlap_x * ovlap_y * eye_mask
            overlap_loss = ((gauss + 5.0 * ovlap_area) * eye_mask).sum() * 0.5
        else:
            ovlap_x = torch.relu(sizes_x_pair - torch.abs(diff_x))
            ovlap_y = torch.relu(sizes_y_pair - torch.abs(diff_y))
            ovlap_area = (ovlap_x * ovlap_y) * eye_mask
            if overlap_form == "quadratic":
                overlap_loss = (ovlap_area * ovlap_area).sum() * 0.5
            elif overlap_form == "cubic":
                overlap_loss = (ovlap_area ** 3).sum() * 0.5
            elif overlap_form == "huber":
                delta = 0.5
                overlap_loss = torch.where(
                    ovlap_area < delta,
                    0.5 * ovlap_area * ovlap_area,
                    delta * (ovlap_area - 0.5 * delta)
                ).sum() * 0.5
            else:
                overlap_loss = ovlap_area.sum() * 0.5

        cur_density_weight = density_weight

        if use_plateau:
            cur_lr = optimizer.param_groups[0]["lr"]
        else:
            cur_lr = lr * (1.0 + (lr_end_factor - 1.0) * progress)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

        if cong_weight > 0 and padded is not None:
            macro_idx, offsets, mask = padded
            pin_xy = pos[macro_idx] + offsets
            x = pin_xy[..., 0]
            y = pin_xy[..., 1]
            neg_inf = torch.finfo(pos.dtype).min
            x_for_max = torch.where(mask, x, x.new_full((), neg_inf))
            x_for_min = torch.where(mask, -x, x.new_full((), neg_inf))
            y_for_max = torch.where(mask, y, y.new_full((), neg_inf))
            y_for_min = torch.where(mask, -y, y.new_full((), neg_inf))
            x_max = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
            x_min = -gamma * torch.logsumexp(x_for_min / gamma, dim=1)
            y_max = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
            y_min = -gamma * torch.logsumexp(y_for_min / gamma, dim=1)
            in_x = (torch.sigmoid((x_max[:, None] - cell_cx_t[None, :]) / cong_smooth_sigma)
                    * torch.sigmoid((cell_cx_t[None, :] - x_min[:, None]) / cong_smooth_sigma))
            in_y = (torch.sigmoid((y_max[:, None] - cell_cy_t[None, :]) / cong_smooth_sigma)
                    * torch.sigmoid((cell_cy_t[None, :] - y_min[:, None]) / cong_smooth_sigma))
            cell_demand = (in_y[:, :, None] * in_x[:, None, :]).sum(dim=0)
            flat_demand = cell_demand.flatten()
            top_k = max(1, int(flat_demand.numel() * cong_top_pct))
            top_vals, _ = torch.topk(flat_demand, top_k)
            cong_loss = top_vals.mean()
        else:
            cong_loss = pos.new_zeros(())

        loss = (wl + cur_density_weight * dpen + overlap_weight * overlap_loss
                + cong_weight * cong_loss)
        loss.backward()
        optimizer.step()
        loss_val_pre = float(loss.item())

        if use_plateau:
            plateau_sched.step(loss_val_pre)

        op_label_grad = None
        trigger_op = False
        if op_on_plateau:
            rel_threshold = max(1e-3, abs(best_loss) * 0.001)
            if loss_val_pre < best_loss - rel_threshold:
                best_loss = loss_val_pre
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= op_patience:
                trigger_op = True
                no_improve_count = 0
                best_loss = loss_val_pre
        if op_every > 0 and step > 0 and step % op_every == 0:
            trigger_op = True

        if (trigger_op
                and op_warmup_progress <= progress <= op_max_progress
                and restart_on_plateau):
            cur_pos_np = pos.detach().numpy().astype(np.float64).copy()
            cpp_dir = str(Path(__file__).resolve().parent / "cpp")
            if cpp_dir not in sys.path:
                sys.path.insert(0, cpp_dir)
            import _placer_core
            sizes_np_hard = sizes_t[:n_hard].numpy().astype(np.float64)
            movable_np_hard = movable[:n_hard].numpy().astype(np.bool_)
            try_state = _placer_core.PlacerState()
            try_state.initialize(
                cur_pos_np[:n_hard].copy(), sizes_np_hard, movable_np_hard,
                np.zeros((0, 2), dtype=np.int32),
                np.zeros(0, dtype=np.float64),
                canvas_w, canvas_h, int(seed),
            )
            try_state.legalize_min_displacement(500)
            try_state.legalize()
            legalized_hard = try_state.current_positions()
            legalized_full = cur_pos_np.copy()
            legalized_full[:n_hard] = legalized_hard
            full = full_template.clone()
            full[:n_active] = torch.tensor(legalized_full, dtype=torch.float32)
            cur_costs = compute_proxy_cost(full, benchmark, plc)
            cur_proxy = float(cur_costs["proxy_cost"])
            cur_overlaps = int(cur_costs["overlap_count"])
            if cur_overlaps == 0 and cur_proxy < global_best_proxy:
                global_best_proxy = cur_proxy
                global_best_pos = legalized_full.astype(np.float64).copy()
            restart_count += 1
            restart_rng = np.random.default_rng(seed + 7000 + restart_count)
            new_pos = np.zeros((n_active, 2), dtype=np.float32)
            half_w_np = (sizes_t[:, 0].numpy() / 2.0)
            half_h_np = (sizes_t[:, 1].numpy() / 2.0)
            new_pos[:, 0] = restart_rng.uniform(half_w_np, canvas_w - half_w_np)
            new_pos[:, 1] = restart_rng.uniform(half_h_np, canvas_h - half_h_np)
            mov_np = movable.numpy().astype(np.bool_)
            fixed_np = fixed_pos.numpy().astype(np.float32)
            new_pos[~mov_np] = fixed_np[~mov_np]
            with torch.no_grad():
                pos.data.copy_(torch.tensor(new_pos, dtype=torch.float32))
            optimizer = torch.optim.Adam([pos], lr=lr)
            if use_plateau:
                plateau_sched = ReduceLROnPlateau(
                    optimizer, mode="min", factor=plateau_factor,
                    patience=plateau_patience, threshold=1e-4,
                    min_lr=plateau_min_lr,
                )
            density_weight = float(os.environ.get("STRAPLE_DEMO_LAMBDA_START", "0.05"))
            overlap_weight = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W", "15"))
            no_improve_count = 0
            best_loss = float("inf")
            op_label_grad = (f"RESTART#{restart_count} "
                             f"prev_proxy={cur_proxy:.4f} ovrlp={cur_overlaps}")
            op_events_grad.append({
                "step": step + 1, "label": op_label_grad,
                "elapsed": time.time() - t_start,
            })
            print(f"[gradient_demo] 🔄 step={step+1} {op_label_grad} "
                  f"global_best={global_best_proxy:.4f}", flush=True)
        elif (trigger_op
                and op_warmup_progress <= progress <= op_max_progress
                and len(op_kinds) > 0):
                from force_demo import _apply_random_op
                pos_np = pos.detach().numpy().astype(np.float64)
                velocity_np = np.zeros_like(pos_np)
                sizes_np = sizes_t.numpy().astype(np.float64)
                half_w_np = (sizes_np[:, 0] / 2.0)
                half_h_np = (sizes_np[:, 1] / 2.0)
                movable_np = movable.numpy().astype(np.bool_)
                edges_a_np = np.zeros(0, dtype=np.int32)
                edges_b_np = np.zeros(0, dtype=np.int32)
                op_label_grad = _apply_random_op(
                    pos_np, velocity_np, sizes_np, half_w_np, half_h_np,
                    canvas_w, canvas_h, movable_np, edges_a_np, edges_b_np,
                    op_k, op_rng, op_kinds,
                )
                if op_label_grad:
                    with torch.no_grad():
                        pos.data.copy_(torch.tensor(pos_np, dtype=torch.float32))
                        for state in optimizer.state.values():
                            if "exp_avg" in state:
                                state["exp_avg"].zero_()
                    no_improve_count = 0
                    best_loss = float("inf")
                    op_events_grad.append({
                        "step": step + 1, "label": op_label_grad,
                        "elapsed": time.time() - t_start,
                    })
                    print(f"[gradient_demo] ⚡ step={step+1} OP: {op_label_grad}",
                          flush=True)

        with torch.no_grad():
            pos[:, 0].clamp_(min=half_w, max=canvas_w - half_w)
            pos[:, 1].clamp_(min=half_h, max=canvas_h - half_h)
            pos[~movable] = fixed_pos[~movable]

        with torch.no_grad():
            dpen_val = float(dpen.item())
            ovlap_val = float(overlap_loss.item())
        if dpen_val > stop_overflow:
            density_weight = min(density_weight * weight_growth, lambda_max)
        if use_lagrangian:
            overlap_weight = min(overlap_weight + lagrangian_rho * ovlap_val,
                                 overlap_w_max)
        else:
            if ovlap_val > overlap_grow_threshold:
                overlap_weight = min(overlap_weight * overlap_w_growth,
                                     overlap_w_max)

        if recorder is not None:
            label = (f"grad step={step+1} "
                     f"λ_d={cur_density_weight:.2f} "
                     f"λ_o={overlap_weight:.1f} λ_c={cong_weight:.1f} "
                     f"ov={ovlap_val:.2f} cong={float(cong_loss):.2f} "
                     f"lr={cur_lr:.3f}")
            if op_label_grad:
                label = f"⚡ {op_label_grad} | " + label
            recorder.add(pos.detach().numpy().astype(np.float64), label)

        if score_history is not None:
            now = time.time()
            if now - last_score_t >= score_sample_every_s:
                last_score_t = now
                full = full_template.clone()
                full[:n_active] = pos.detach()
                cs = compute_proxy_cost(full, benchmark, plc)
                score_history.append({
                    "step": step + 1, "elapsed": now - t_start,
                    "wl": float(cs["wirelength_cost"]),
                    "den": float(cs["density_cost"]),
                    "cong": float(cs["congestion_cost"]),
                    "proxy": float(cs["proxy_cost"]),
                    "ovrlp": int(cs["overlap_count"]),
                    "loss": float(loss.item()),
                    "wl_smooth": float(wl.item()),
                    "dpen": float(dpen.item()),
                    "overlap_term": float(overlap_loss.item()),
                    "cong_term": float(cong_loss.item()) if cong_weight > 0 else 0.0,
                    "lambda_d": density_weight,
                    "lambda_o": overlap_weight,
                    "lambda_c": cong_weight,
                })
        step += 1

    final_pos = pos.detach().numpy().astype(np.float64)

    if restart_on_plateau:
        cpp_dir = str(Path(__file__).resolve().parent / "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        import _placer_core
        sizes_np_hard = sizes_t[:n_hard].numpy().astype(np.float64)
        movable_np_hard = movable[:n_hard].numpy().astype(np.bool_)
        end_state = _placer_core.PlacerState()
        end_state.initialize(
            final_pos[:n_hard].copy(), sizes_np_hard, movable_np_hard,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            canvas_w, canvas_h, int(seed),
        )
        end_state.legalize_min_displacement(500)
        end_state.legalize()
        legalized_hard = end_state.current_positions()
        legalized_end = final_pos.copy()
        legalized_end[:n_hard] = legalized_hard
        full = full_template.clone()
        full[:n_active] = torch.tensor(legalized_end, dtype=torch.float32)
        end_costs = compute_proxy_cost(full, benchmark, plc)
        end_proxy = float(end_costs["proxy_cost"])
        end_overlaps = int(end_costs["overlap_count"])
        if end_overlaps == 0 and end_proxy < global_best_proxy:
            global_best_proxy = end_proxy
            global_best_pos = legalized_end.astype(np.float64).copy()
        if global_best_pos is not None:
            print(f"[gradient_demo] using global_best_proxy={global_best_proxy:.4f} "
                  f"(restart_count={restart_count})", flush=True)
            final_pos = global_best_pos.copy()

    do_legalize = os.environ.get("STRAPLE_DEMO_FINISH_LEGALIZE", "1") != "0"
    if do_legalize:
        cpp_dir = str(Path(__file__).resolve().parent / "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        import _placer_core
        sizes_np_hard = sizes_t[:n_hard].numpy().astype(np.float64)
        movable_np_hard = movable[:n_hard].numpy().astype(np.bool_)
        state = _placer_core.PlacerState()
        state.initialize(
            final_pos[:n_hard].copy(), sizes_np_hard, movable_np_hard,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            canvas_w, canvas_h, int(seed),
        )
        moves = state.legalize_min_displacement(500)
        state.legalize()
        legalized_hard = state.current_positions()
        final_pos[:n_hard] = legalized_hard
        print(f"[gradient_demo] post-legalize: min_disp moves={moves}", flush=True)
        if recorder is not None:
            recorder.add(final_pos.astype(np.float64), "gradient: FINISH legalize")

    if score_history is not None:
        full = full_template.clone()
        full[:n_active] = torch.tensor(final_pos, dtype=torch.float32)
        cf = compute_proxy_cost(full, benchmark, plc)
        last_record = score_history[-1] if score_history else {}
        score_history.append({
            "step": step + 1, "elapsed": time.time() - t_start,
            "wl": float(cf["wirelength_cost"]),
            "den": float(cf["density_cost"]),
            "cong": float(cf["congestion_cost"]),
            "proxy": float(cf["proxy_cost"]),
            "ovrlp": int(cf["overlap_count"]),
            "loss": last_record.get("loss", 0.0),
            "wl_smooth": last_record.get("wl_smooth", 0.0),
            "dpen": last_record.get("dpen", 0.0),
            "overlap_term": last_record.get("overlap_term", 0.0),
            "lambda_d": density_weight,
            "lambda_o": overlap_weight,
        })
        if score_png:
            from force_demo import _save_score_png
            _save_score_png(score_history, op_events, score_png, benchmark.name)

    print(f"[gradient_demo] {step} steps in {time.time()-t_start:.1f}s "
          f"final λ={density_weight:.3f}", flush=True)

    if score_history is not None:
        return final_pos, score_history, op_events
    return final_pos
