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
        _build_padded_net_tensors,
        _smooth_hpwl_padded,
        _density_penalty,
    )
    return (_build_net_pin_tensors, _build_padded_net_tensors,
            _smooth_hpwl_padded, _density_penalty)


def gradient_demo(benchmark, plc, recorder=None, num_steps: int = 300,
                  seed: int = 42, time_budget: float = 0.0,
                  score_png: str = "", score_sample_every_s: float = 1.0):
    import torch
    from macro_place.objective import compute_proxy_cost

    (_build_net_pin_tensors, _build_padded_net_tensors,
     _smooth_hpwl_padded, _density_penalty) = _import_analytical()

    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    canvas_min = min(canvas_w, canvas_h)

    sizes_t = benchmark.macro_sizes[:n_hard].float()
    half_w = sizes_t[:, 0] / 2.0
    half_h = sizes_t[:, 1] / 2.0
    movable = benchmark.get_movable_mask()[:n_hard]
    fixed_pos = benchmark.macro_positions[:n_hard].float().clone()

    rng_np = np.random.default_rng(seed)
    init_mode = os.environ.get("STRAPLE_DEMO_INIT", "center")
    pos_np = np.zeros((n_hard, 2), dtype=np.float32)
    if init_mode == "random":
        pos_np[:, 0] = rng_np.uniform(half_w.numpy(), canvas_w - half_w.numpy())
        pos_np[:, 1] = rng_np.uniform(half_h.numpy(), canvas_h - half_h.numpy())
    else:
        spawn_jitter = canvas_min * float(
            os.environ.get("STRAPLE_DEMO_SPAWN_JITTER", "0.005"))
        pos_np[:, 0] = canvas_w / 2.0 + rng_np.normal(0, spawn_jitter, n_hard)
        pos_np[:, 1] = canvas_h / 2.0 + rng_np.normal(0, spawn_jitter, n_hard)
        pos_np[:, 0] = np.clip(pos_np[:, 0],
                               half_w.numpy(), canvas_w - half_w.numpy())
        pos_np[:, 1] = np.clip(pos_np[:, 1],
                               half_h.numpy(), canvas_h - half_h.numpy())
    fixed_idx = (~movable).numpy()
    pos_np[fixed_idx] = fixed_pos.numpy()[fixed_idx]

    pos = torch.tensor(pos_np, dtype=torch.float32, requires_grad=True)

    lr = float(os.environ.get("STRAPLE_DEMO_LR", "0.3"))
    optimizer = torch.optim.Adam([pos], lr=lr)

    print(f"[gradient_demo] building net tensors...", flush=True)
    net_macro_idx, net_pin_offsets = _build_net_pin_tensors(benchmark, plc)
    padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets)
    n_nets = len(net_macro_idx)
    print(f"[gradient_demo] n_hard={n_hard} n_nets={n_nets} "
          f"canvas={canvas_w:.1f}x{canvas_h:.1f}", flush=True)

    gamma_base = canvas_min * float(os.environ.get("STRAPLE_DEMO_GAMMA_FRAC", "0.05"))
    gamma_factor_start = float(os.environ.get("STRAPLE_DEMO_GAMMA_START", "1.5"))
    gamma_factor_end = float(os.environ.get("STRAPLE_DEMO_GAMMA_END", "0.3"))

    density_weight = float(os.environ.get("STRAPLE_DEMO_LAMBDA_START", "0.01"))
    lambda_max = float(os.environ.get("STRAPLE_DEMO_LAMBDA_MAX", "100.0"))
    weight_growth = float(os.environ.get("STRAPLE_DEMO_LAMBDA_GROWTH", "1.05"))
    target_util = float(os.environ.get("STRAPLE_DEMO_TARGET_UTIL", "0.7"))
    stop_overflow = float(os.environ.get("STRAPLE_DEMO_STOP_OVERFLOW", "0.07"))

    overlap_weight = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W", "1.0"))
    overlap_w_max = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W_MAX", "100000.0"))
    overlap_w_growth = float(os.environ.get("STRAPLE_DEMO_OVERLAP_W_GROWTH", "1.01"))
    overlap_form = os.environ.get("STRAPLE_DEMO_OVERLAP_FORM", "linear")
    overlap_grow_threshold = float(
        os.environ.get("STRAPLE_DEMO_OVERLAP_GROW_THRESHOLD", "0.5"))
    use_lagrangian = os.environ.get("STRAPLE_DEMO_LAGRANGIAN", "1") != "0"
    lagrangian_rho = float(os.environ.get("STRAPLE_DEMO_RHO", "5.0"))

    lr_end_factor = float(os.environ.get("STRAPLE_DEMO_LR_END_FACTOR", "0.05"))

    sizes_x_pair = (sizes_t[:, 0:1] + sizes_t[:, 0].unsqueeze(0)) * 0.5
    sizes_y_pair = (sizes_t[:, 1:2] + sizes_t[:, 1].unsqueeze(0)) * 0.5
    eye_mask = (1.0 - torch.eye(n_hard, dtype=torch.float32))

    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)

    if recorder is not None:
        recorder.add(pos.detach().numpy().astype(np.float64),
                     f"gradient: init lambda={density_weight:.4f}")

    score_history = [] if (time_budget > 0 or score_png) else None
    op_events = None
    full_template = benchmark.macro_positions.clone()

    if score_history is not None:
        full = full_template.clone()
        full[:n_hard] = pos.detach()
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

        diff_x = pos[:, 0:1] - pos[:, 0].unsqueeze(0)
        diff_y = pos[:, 1:2] - pos[:, 1].unsqueeze(0)
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

        cur_lr = lr * (1.0 + (lr_end_factor - 1.0) * progress)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        loss = wl + cur_density_weight * dpen + overlap_weight * overlap_loss
        loss.backward()
        optimizer.step()

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
            recorder.add(pos.detach().numpy().astype(np.float64),
                         f"grad step={step+1} "
                         f"λ_d={cur_density_weight:.2f} "
                         f"λ_o={overlap_weight:.1f} ov={ovlap_val:.2f} "
                         f"lr={cur_lr:.3f}")

        if score_history is not None:
            now = time.time()
            if now - last_score_t >= score_sample_every_s:
                last_score_t = now
                full = full_template.clone()
                full[:n_hard] = pos.detach()
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
                    "lambda_d": density_weight,
                    "lambda_o": overlap_weight,
                })
        step += 1

    final_pos = pos.detach().numpy().astype(np.float64)

    do_legalize = os.environ.get("STRAPLE_DEMO_FINISH_LEGALIZE", "1") != "0"
    if do_legalize:
        cpp_dir = str(Path(__file__).resolve().parent / "cpp")
        if cpp_dir not in sys.path:
            sys.path.insert(0, cpp_dir)
        import _placer_core
        sizes_np = sizes_t.numpy().astype(np.float64)
        movable_np = movable.numpy().astype(np.bool_)
        state = _placer_core.PlacerState()
        state.initialize(
            final_pos.copy(), sizes_np, movable_np,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            canvas_w, canvas_h, int(seed),
        )
        moves = state.legalize_min_displacement(500)
        state.legalize()
        final_pos = state.current_positions()
        print(f"[gradient_demo] post-legalize: min_disp moves={moves}", flush=True)
        if recorder is not None:
            recorder.add(final_pos.astype(np.float64), "gradient: FINISH legalize")

    if score_history is not None:
        full = full_template.clone()
        full[:n_hard] = torch.tensor(final_pos, dtype=torch.float32)
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
