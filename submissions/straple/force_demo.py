"""Force-directed placement demo (visual mode, not for submission).

Simulates physics on hard macros:
  - Initial position: all at canvas center (or random) with tiny symmetry-breaking jitter
  - Repulsion when two macros overlap (push along axis of minimum overlap)
  - Optional spring attraction along netlist edges (HPWL surrogate)
  - Boundary bounce with damping
  - Damped integration with damping schedule (low → high), so motion fades and final state stabilizes

Records every step into recorder if provided. Activated by env var
`STRAPLE_DEMO=force` in placer.py — bypasses the normal LNS pipeline.
"""

from __future__ import annotations

import os

import numpy as np


def force_directed_demo(benchmark, plc, edges, edge_weights,
                        recorder=None, num_iters: int = 300, seed: int = 42,
                        time_budget: float = 0.0, score_png: str = "",
                        score_sample_every_s: float = 1.0):
    import time
    import torch
    from macro_place.objective import _set_placement, compute_proxy_cost

    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    movable_full = benchmark.get_movable_mask()[:n_hard].numpy().astype(np.bool_)
    fixed_full = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    canvas_min = min(canvas_w, canvas_h)
    grid_rows = int(benchmark.grid_rows)
    grid_cols = int(benchmark.grid_cols)
    grid_width = canvas_w / grid_cols
    grid_height = canvas_h / grid_rows
    cell_cx = (np.arange(grid_cols) + 0.5) * grid_width
    cell_cy = (np.arange(grid_rows) + 0.5) * grid_height
    full_template = benchmark.macro_positions.clone()

    rng = np.random.default_rng(seed)

    init_mode = os.environ.get("STRAPLE_DEMO_INIT", "center")
    pos = np.zeros((n_hard, 2), dtype=np.float64)
    if init_mode == "random":
        pos[:, 0] = rng.uniform(half_w, canvas_w - half_w)
        pos[:, 1] = rng.uniform(half_h, canvas_h - half_h)
        init_label = f"random seed={seed}"
    else:
        center_x = canvas_w / 2.0
        center_y = canvas_h / 2.0
        spawn_jitter = canvas_min * float(
            os.environ.get("STRAPLE_DEMO_SPAWN_JITTER", "0.005"))
        pos[:, 0] = center_x + rng.normal(0.0, spawn_jitter, n_hard)
        pos[:, 1] = center_y + rng.normal(0.0, spawn_jitter, n_hard)
        pos[:, 0] = np.clip(pos[:, 0], half_w, canvas_w - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, canvas_h - half_h)
        init_label = f"center spawn seed={seed}"
    pos[~movable_full] = fixed_full[~movable_full]

    velocity = np.zeros_like(pos)

    repulsion = float(os.environ.get("STRAPLE_DEMO_REPULSION", "0.8"))
    spring = float(os.environ.get("STRAPLE_DEMO_SPRING", "0.0008"))
    spread = float(os.environ.get("STRAPLE_DEMO_SPREAD", "0.008"))
    cong_push = float(os.environ.get("STRAPLE_DEMO_CONG_PUSH", "0.0"))
    cong_recompute_every = int(os.environ.get("STRAPLE_DEMO_CONG_EVERY", "1"))
    cong_grid = None
    bounce_factor = float(os.environ.get("STRAPLE_DEMO_BOUNCE", "-0.7"))
    op_every = int(os.environ.get("STRAPLE_DEMO_OP_EVERY", "60"))
    op_k = int(os.environ.get("STRAPLE_DEMO_OP_K", "8"))
    op_progress_max = float(os.environ.get("STRAPLE_DEMO_OP_PROGRESS_MAX", "0.65"))
    op_kinds_env = os.environ.get(
        "STRAPLE_DEMO_OPS", "teleport,swap,pull,shake,scatter_cluster")
    op_kinds = [s.strip() for s in op_kinds_env.split(",") if s.strip()]
    damping_start = float(os.environ.get("STRAPLE_DEMO_DAMPING", "0.85"))
    damping_end = float(os.environ.get("STRAPLE_DEMO_DAMPING_END", "0.99"))
    max_velocity_start = canvas_min * float(
        os.environ.get("STRAPLE_DEMO_MAX_VEL", "0.06"))
    max_velocity_end = max_velocity_start * float(
        os.environ.get("STRAPLE_DEMO_MAX_VEL_END_FRAC", "0.05"))
    jitter_start_frac = float(os.environ.get("STRAPLE_DEMO_JITTER", "0.002"))
    soft = canvas_min * 0.05

    if recorder is not None:
        recorder.add(pos.copy(), f"demo: init {init_label}")

    edges_a = edges[:, 0] if len(edges) > 0 else np.zeros(0, dtype=np.int32)
    edges_b = edges[:, 1] if len(edges) > 0 else np.zeros(0, dtype=np.int32)

    score_history = [] if (time_budget > 0 or score_png) else None
    op_events = [] if score_history is not None else None
    t_sim_start = time.time()
    last_score_t = t_sim_start

    if score_history is not None:
        full = full_template.clone()
        full[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        c0 = compute_proxy_cost(full, benchmark, plc)
        score_history.append({
            "step": 0, "elapsed": 0.0,
            "wl": float(c0["wirelength_cost"]),
            "den": float(c0["density_cost"]),
            "cong": float(c0["congestion_cost"]),
            "proxy": float(c0["proxy_cost"]),
            "ovrlp": int(c0["overlap_count"]),
        })

    step = 0
    while True:
        if time_budget > 0:
            if time.time() - t_sim_start >= time_budget:
                break
            progress = (time.time() - t_sim_start) / time_budget
        else:
            if step >= num_iters:
                break
            progress = step / max(1, num_iters - 1)
        damping = damping_start + (damping_end - damping_start) * progress
        cur_max_velocity = (max_velocity_start
                            + (max_velocity_end - max_velocity_start) * progress)
        cur_jitter = canvas_min * jitter_start_frac * (1.0 - progress)

        if cong_push > 0 and step % cong_recompute_every == 0 and progress < 0.7:
            full = full_template.clone()
            full[:n_hard] = torch.tensor(pos, dtype=torch.float32)
            try:
                _set_placement(plc, full, benchmark)
                plc.get_congestion_cost()
                h_grid = np.asarray(plc.H_routing_cong, dtype=np.float64).reshape(
                    grid_rows, grid_cols)
                v_grid = np.asarray(plc.V_routing_cong, dtype=np.float64).reshape(
                    grid_rows, grid_cols)
                cong_grid = np.maximum(h_grid, v_grid)
                threshold = np.percentile(cong_grid, 80)
                cong_active = cong_grid > threshold
                cong_grad_y, cong_grad_x = np.gradient(cong_grid)
                cong_grad_x = np.where(cong_active, cong_grad_x, 0.0)
                cong_grad_y = np.where(cong_active, cong_grad_y, 0.0)
            except Exception:
                cong_grid = None
                cong_grad_x = cong_grad_y = None

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dx = diff[..., 0]
        dy = diff[..., 1]
        abs_dx = np.abs(dx)
        abs_dy = np.abs(dy)

        sep_x = (sizes[:, 0:1] + sizes[np.newaxis, :, 0]) * 0.5 + 0.05
        sep_y = (sizes[:, 1:2] + sizes[np.newaxis, :, 1]) * 0.5 + 0.05

        overlap_x = np.maximum(sep_x - abs_dx, 0.0)
        overlap_y = np.maximum(sep_y - abs_dy, 0.0)
        overlap_mask = (overlap_x > 0) & (overlap_y > 0)
        np.fill_diagonal(overlap_mask, False)

        push_x_axis = (overlap_x < overlap_y) & overlap_mask
        push_y_axis = (~push_x_axis) & overlap_mask

        sign_x = np.where(dx >= 0.0, 1.0, -1.0)
        sign_y = np.where(dy >= 0.0, 1.0, -1.0)

        fx_pairs = np.where(push_x_axis, sign_x * overlap_x * repulsion, 0.0)
        fy_pairs = np.where(push_y_axis, sign_y * overlap_y * repulsion, 0.0)

        jitter_x = rng.normal(0.0, cur_jitter, n_hard) * movable_full if cur_jitter > 0 else 0.0
        jitter_y = rng.normal(0.0, cur_jitter, n_hard) * movable_full if cur_jitter > 0 else 0.0

        forces = np.zeros_like(pos)
        forces[:, 0] = fx_pairs.sum(axis=1) + jitter_x
        forces[:, 1] = fy_pairs.sum(axis=1) + jitter_y

        metric_decay = float(os.environ.get("STRAPLE_DEMO_METRIC_DECAY", "2.0"))
        metric_scale = max(0.0, 1.0 - metric_decay * progress)

        if spread > 0:
            dist_sq = dx * dx + dy * dy + soft * soft
            inv_dist = spread * metric_scale / dist_sq
            spread_fx = sign_x * np.abs(dx) * inv_dist
            spread_fy = sign_y * np.abs(dy) * inv_dist
            np.fill_diagonal(spread_fx, 0.0)
            np.fill_diagonal(spread_fy, 0.0)
            forces[:, 0] += spread_fx.sum(axis=1)
            forces[:, 1] += spread_fy.sum(axis=1)

        if len(edges_a) > 0 and spring > 0:
            dxe = pos[edges_a, 0] - pos[edges_b, 0]
            dye = pos[edges_a, 1] - pos[edges_b, 1]
            we = edge_weights * spring * metric_scale
            np.add.at(forces[:, 0], edges_a, -dxe * we)
            np.add.at(forces[:, 0], edges_b,  dxe * we)
            np.add.at(forces[:, 1], edges_a, -dye * we)
            np.add.at(forces[:, 1], edges_b,  dye * we)

        if cong_push > 0 and cong_grid is not None and metric_scale > 0:
            macro_col = np.clip((pos[:, 0] / grid_width).astype(np.int64),
                                0, grid_cols - 1)
            macro_row = np.clip((pos[:, 1] / grid_height).astype(np.int64),
                                0, grid_rows - 1)
            gx = cong_grad_x[macro_row, macro_col]
            gy = cong_grad_y[macro_row, macro_col]
            forces[:, 0] -= gx * cong_push * metric_scale * grid_width
            forces[:, 1] -= gy * cong_push * metric_scale * grid_height

        velocity = velocity * damping + forces
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        too_fast = speed > cur_max_velocity
        velocity = np.where(too_fast,
                            velocity * (cur_max_velocity / np.maximum(speed, 1e-9)),
                            velocity)

        pos = pos + velocity
        pos[~movable_full] = fixed_full[~movable_full]
        velocity[~movable_full] = 0.0

        below_x = pos[:, 0] < half_w
        above_x = pos[:, 0] > canvas_w - half_w
        below_y = pos[:, 1] < half_h
        above_y = pos[:, 1] > canvas_h - half_h
        pos[below_x, 0] = half_w[below_x]
        pos[above_x, 0] = canvas_w - half_w[above_x]
        pos[below_y, 1] = half_h[below_y]
        pos[above_y, 1] = canvas_h - half_h[above_y]
        velocity[below_x | above_x, 0] *= bounce_factor
        velocity[below_y | above_y, 1] *= bounce_factor

        op_label = None
        if (op_every > 0 and step > 0 and step % op_every == 0
                and progress < op_progress_max):
            op_label = _apply_random_op(
                pos, velocity, sizes, half_w, half_h, canvas_w, canvas_h,
                movable_full, edges_a, edges_b, op_k, rng, op_kinds)
            if op_label and op_events is not None:
                op_events.append({"step": step + 1, "label": op_label,
                                  "elapsed": time.time() - t_sim_start})

        if recorder is not None:
            label = f"demo: step={step+1}/{num_iters}"
            if op_label:
                label = f"demo: step={step+1}  ⚡ {op_label}"
            recorder.add(pos.copy(), label)

        if score_history is not None:
            now = time.time()
            if now - last_score_t >= score_sample_every_s:
                last_score_t = now
                full = full_template.clone()
                full[:n_hard] = torch.tensor(pos, dtype=torch.float32)
                cs = compute_proxy_cost(full, benchmark, plc)
                score_history.append({
                    "step": step + 1, "elapsed": now - t_sim_start,
                    "wl": float(cs["wirelength_cost"]),
                    "den": float(cs["density_cost"]),
                    "cong": float(cs["congestion_cost"]),
                    "proxy": float(cs["proxy_cost"]),
                    "ovrlp": int(cs["overlap_count"]),
                })
        step += 1

    if score_history is not None:
        full = full_template.clone()
        full[:n_hard] = torch.tensor(pos, dtype=torch.float32)
        cf = compute_proxy_cost(full, benchmark, plc)
        score_history.append({
            "step": step, "elapsed": time.time() - t_sim_start,
            "wl": float(cf["wirelength_cost"]),
            "den": float(cf["density_cost"]),
            "cong": float(cf["congestion_cost"]),
            "proxy": float(cf["proxy_cost"]),
            "ovrlp": int(cf["overlap_count"]),
        })
        if score_png:
            _save_score_png(score_history, op_events, score_png, benchmark.name)

    if score_history is not None:
        return pos, score_history, op_events
    return pos


def _save_score_png(history, op_events, out_path, bench_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    steps = [h["step"] for h in history]
    has_loss = any(h.get("loss", 0) > 0 for h in history)

    if has_loss:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(steps, [h["proxy"] for h in history], color="black", lw=2.4,
            label="proxy_cost (real)")
    ax.plot(steps, [h["wl"] for h in history], color="green", lw=1.4,
            label="wirelength")
    ax.plot(steps, [h["den"] for h in history], color="red", lw=1.4,
            label="density")
    ax.plot(steps, [h["cong"] for h in history], color="blue", lw=1.4,
            label="congestion")
    if op_events:
        for ev in op_events:
            ax.axvline(ev["step"], color="orange", lw=0.5, alpha=0.4)
        ax.plot([], [], color="orange", lw=1, alpha=0.6,
                label=f"random ops ({len(op_events)})")
    ax.set_ylabel("real cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    last = history[-1]
    ax.set_title(
        f"{bench_name} · simulation score history ({last['elapsed']:.1f}s, "
        f"{last['step']} steps) · final proxy={last['proxy']:.4f} "
        f"WL={last['wl']:.3f} D={last['den']:.3f} C={last['cong']:.3f} "
        f"ovrlp={last['ovrlp']}",
        fontsize=11,
    )

    if has_loss:
        ax_loss = axes[1]
        loss_vals = np.array([h.get("loss", 0) for h in history])
        wl_smooth = np.array([h.get("wl_smooth", 0) for h in history])
        dpen_vals = np.array([h.get("dpen", 0) for h in history])
        ovlap_terms = np.array([h.get("overlap_term", 0) for h in history])
        lambda_d = np.array([h.get("lambda_d", 0) for h in history])
        lambda_o = np.array([h.get("lambda_o", 0) for h in history])

        ax_loss.plot(steps, loss_vals, color="black", lw=2.2,
                     label="total loss (gradient target)")
        ax_loss.plot(steps, wl_smooth, color="green", lw=1.2, ls="--",
                     label="WL_smooth")
        ax_loss.plot(steps, dpen_vals * lambda_d, color="red", lw=1.2, ls="--",
                     label="λ_d × density_bell")
        ax_loss.plot(steps, ovlap_terms * lambda_o, color="purple", lw=1.2,
                     ls="--", label="λ_o × overlap_term")
        ax_loss.set_yscale("symlog", linthresh=1.0)
        ax_loss.set_ylabel("loss (symlog)")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc="upper right", fontsize=8)
        ax_loss.set_title(
            f"gradient loss components · final loss={loss_vals[-1]:.1f} "
            f"λ_d={lambda_d[-1]:.2f} λ_o={lambda_o[-1]:.1f}",
            fontsize=10,
        )

    ax2 = axes[-1]
    ax2.plot(steps, [h["ovrlp"] for h in history], color="purple", lw=1.4)
    ax2.fill_between(steps, [h["ovrlp"] for h in history], alpha=0.2,
                     color="purple")
    ax2.set_xlabel("simulation step")
    ax2.set_ylabel("overlaps")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[force_demo] saved score history → {out_path}", flush=True)


def _apply_random_op(pos, velocity, sizes, half_w, half_h, canvas_w, canvas_h,
                     movable_full, edges_a, edges_b, op_k, rng, op_kinds):
    movable_idx = np.where(movable_full)[0]
    if len(movable_idx) < 2:
        return None
    op = rng.choice(op_kinds)
    if op == "teleport":
        k = min(op_k, len(movable_idx))
        targets = rng.choice(movable_idx, size=k, replace=False)
        for i in targets:
            pos[i, 0] = rng.uniform(half_w[i], canvas_w - half_w[i])
            pos[i, 1] = rng.uniform(half_h[i], canvas_h - half_h[i])
            velocity[i] = 0.0
        return f"teleport k={k}"
    elif op == "swap":
        k_pairs = max(1, op_k // 2)
        k_pairs = min(k_pairs, len(movable_idx) // 2)
        chosen = rng.choice(movable_idx, size=2 * k_pairs, replace=False)
        for p in range(k_pairs):
            a = int(chosen[2 * p])
            b = int(chosen[2 * p + 1])
            tx, ty = pos[a, 0], pos[a, 1]
            pos[a, 0], pos[a, 1] = pos[b, 0], pos[b, 1]
            pos[b, 0], pos[b, 1] = tx, ty
            velocity[a] = 0.0
            velocity[b] = 0.0
        return f"swap pairs={k_pairs}"
    elif op == "pull":
        if len(edges_a) == 0:
            return None
        n_edges = min(op_k, len(edges_a))
        chosen_e = rng.choice(len(edges_a), size=n_edges, replace=False)
        applied = 0
        for e in chosen_e:
            a = int(edges_a[e])
            b = int(edges_b[e])
            if not (movable_full[a] and movable_full[b]):
                continue
            mid_x = (pos[a, 0] + pos[b, 0]) * 0.5
            mid_y = (pos[a, 1] + pos[b, 1]) * 0.5
            velocity[a, 0] += (mid_x - pos[a, 0]) * 0.5
            velocity[a, 1] += (mid_y - pos[a, 1]) * 0.5
            velocity[b, 0] += (mid_x - pos[b, 0]) * 0.5
            velocity[b, 1] += (mid_y - pos[b, 1]) * 0.5
            applied += 1
        return f"pull edges={applied}"
    elif op == "shake":
        canvas_min = min(canvas_w, canvas_h)
        kick = canvas_min * 0.05
        velocity[movable_full, 0] += rng.normal(0.0, kick, movable_full.sum())
        velocity[movable_full, 1] += rng.normal(0.0, kick, movable_full.sum())
        return f"shake all"
    elif op == "scatter_cluster":
        seed = int(rng.choice(movable_idx))
        cx, cy = pos[seed, 0], pos[seed, 1]
        d2 = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
        radius_sq = (min(canvas_w, canvas_h) * 0.15) ** 2
        nearby = (d2 < radius_sq) & movable_full
        if nearby.sum() < 2:
            return None
        idxs = np.where(nearby)[0]
        for i in idxs[:op_k]:
            pos[i, 0] = rng.uniform(half_w[i], canvas_w - half_w[i])
            pos[i, 1] = rng.uniform(half_h[i], canvas_h - half_h[i])
            velocity[i] = 0.0
        return f"scatter cluster of {min(len(idxs), op_k)}"
    return None
