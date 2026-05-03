"""
Analytical-style differentiable seed (Phase 2 prototype).

Gradient descent on a smooth objective that approximates HPWL + density penalty.
Uses PyTorch autograd, runs on CPU, no external deps.

  WL_smooth(net) = LSE(x) - LSE(-x) + LSE(y) - LSE(-y)     (log-sum-exp surrogate)
  D_pen          = sum over grid cells of (cell_density - target)^2
                   computed via Gaussian-bell smoothing of macro footprints.

This is meant as a SEED only — output is not legalized. Pass to existing
legalize() + SA + LNS pipeline downstream.
"""

import math

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _build_net_pin_tensors(benchmark, plc):
    """Group net pin endpoints by macro index for vectorized HPWL.

    Returns:
        net_macro_idx: list of int64 tensors [P_i] — for each net, the macro index of each pin
        net_pin_offsets: list of float32 tensors [P_i, 2] — pin offset relative to macro center
    """
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx

    net_macro_idx = []
    net_pin_offsets = []
    for driver, sinks in plc.nets.items():
        macros = []
        offsets = []
        for pin_name in [driver] + sinks:
            parent = pin_name.split("/")[0]
            if parent not in name_to_bidx:
                continue
            bidx = name_to_bidx[parent]
            pin_idx = plc.mod_name_to_indices.get(pin_name, -1)
            if pin_idx < 0:
                continue
            pin_node = plc.modules_w_pins[pin_idx]
            ox, oy = pin_node.get_offset()
            macros.append(bidx)
            offsets.append((ox, oy))
        if len(macros) < 2:
            continue
        net_macro_idx.append(torch.tensor(macros, dtype=torch.long))
        net_pin_offsets.append(torch.tensor(offsets, dtype=torch.float32))
    return net_macro_idx, net_pin_offsets


def _build_padded_net_tensors(net_macro_idx, net_pin_offsets):
    num_nets = len(net_macro_idx)
    if num_nets == 0:
        return None
    max_pins = max(int(t.numel()) for t in net_macro_idx)

    macro_idx_padded = torch.zeros((num_nets, max_pins), dtype=torch.long)
    offsets_padded = torch.zeros((num_nets, max_pins, 2), dtype=torch.float32)
    mask = torch.zeros((num_nets, max_pins), dtype=torch.bool)

    for n, (macros, offsets) in enumerate(zip(net_macro_idx, net_pin_offsets)):
        p = int(macros.numel())
        macro_idx_padded[n, :p] = macros
        offsets_padded[n, :p] = offsets
        mask[n, :p] = True

    return macro_idx_padded, offsets_padded, mask


def _smooth_hpwl_padded(pos, padded_data, gamma):
    macro_idx, offsets, mask = padded_data
    pin_xy = pos[macro_idx] + offsets
    x = pin_xy[..., 0]
    y = pin_xy[..., 1]

    neg_inf = torch.finfo(pos.dtype).min
    x_for_max = torch.where(mask, x, x.new_full((), neg_inf))
    x_for_min = torch.where(mask, -x, x.new_full((), neg_inf))
    y_for_max = torch.where(mask, y, y.new_full((), neg_inf))
    y_for_min = torch.where(mask, -y, y.new_full((), neg_inf))

    max_x = gamma * torch.logsumexp(x_for_max / gamma, dim=1)
    min_x = -gamma * torch.logsumexp(x_for_min / gamma, dim=1)
    max_y = gamma * torch.logsumexp(y_for_max / gamma, dim=1)
    min_y = -gamma * torch.logsumexp(y_for_min / gamma, dim=1)

    return ((max_x - min_x) + (max_y - min_y)).sum()


def _smooth_hpwl(pos, net_macro_idx, net_pin_offsets, gamma):
    total = pos.new_zeros(())
    for macros, offsets in zip(net_macro_idx, net_pin_offsets):
        pin_xy = pos[macros] + offsets
        x = pin_xy[:, 0]
        y = pin_xy[:, 1]
        max_x = gamma * torch.logsumexp(x / gamma, dim=0)
        min_x = -gamma * torch.logsumexp(-x / gamma, dim=0)
        max_y = gamma * torch.logsumexp(y / gamma, dim=0)
        min_y = -gamma * torch.logsumexp(-y / gamma, dim=0)
        total = total + (max_x - min_x) + (max_y - min_y)
    return total


def _density_penalty(pos, sizes, canvas_w, canvas_h, grid_rows, grid_cols, target_util):
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    sigma_x = cell_w
    sigma_y = cell_h

    grid_x = torch.arange(grid_cols, dtype=pos.dtype, device=pos.device) * cell_w + cell_w / 2
    grid_y = torch.arange(grid_rows, dtype=pos.dtype, device=pos.device) * cell_h + cell_h / 2

    macro_areas = sizes[:, 0] * sizes[:, 1]

    dx = pos[:, 0:1] - grid_x.unsqueeze(0)
    dy = pos[:, 1:2] - grid_y.unsqueeze(0)
    bell_x = torch.exp(-(dx ** 2) / (2 * sigma_x ** 2))
    bell_y = torch.exp(-(dy ** 2) / (2 * sigma_y ** 2))

    norm_x = bell_x / bell_x.sum(dim=1, keepdim=True).clamp_min(1e-12)
    norm_y = bell_y / bell_y.sum(dim=1, keepdim=True).clamp_min(1e-12)

    cell_density = (macro_areas.unsqueeze(1).unsqueeze(2)
                    * norm_x.unsqueeze(1)
                    * norm_y.unsqueeze(2)).sum(dim=0)

    cell_capacity = cell_w * cell_h
    excess = (cell_density / cell_capacity - target_util).clamp(min=0.0)
    return (excess ** 2).sum()


def _count_pairwise_overlaps(pos, sizes, n_hard):
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    overlaps = 0
    overlap_area = 0.0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            sep_x = (sizes[i, 0] + sizes[j, 0]) / 2
            sep_y = (sizes[i, 1] + sizes[j, 1]) / 2
            if dx < sep_x and dy < sep_y:
                overlaps += 1
                overlap_area += (sep_x - dx) * (sep_y - dy)
    return overlaps, overlap_area


def analytical_seed(benchmark: Benchmark, plc, num_steps=200, lr=0.5,
                    lambda_density=0.0, gamma_frac=0.05, target_util=0.6,
                    log_every=0, log_proxy=False, label="",
                    lambda_schedule=None, gamma_schedule=None):
    """DREAMPlace-style analytical placer.

    lambda_schedule: optional list of (step_fraction, lambda_value). At each
        Adam step, lambda is interpolated linearly through the schedule.
        Example: [(0.0, 0.01), (0.3, 1.0), (0.7, 50.0), (1.0, 200.0)]
        — cold start (WL only), then ramp up density penalty.
        If None, constant `lambda_density` is used.

    gamma_schedule: optional list of (step_fraction, gamma_factor). gamma is
        multiplied by gamma_factor at each step (linear interp).
        Example: [(0.0, 1.0), (1.0, 0.2)] — cool down gamma over time for
        sharper convergence near the end.
    """
    import time
    t_start = time.time()

    n_hard = benchmark.num_hard_macros
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].float()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    movable = benchmark.get_movable_mask()[:n_hard]
    fixed_pos = benchmark.macro_positions[:n_hard].float().clone()

    pos = fixed_pos.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pos], lr=lr)

    t_setup0 = time.time()
    net_macro_idx, net_pin_offsets = _build_net_pin_tensors(benchmark, plc)
    padded_data = _build_padded_net_tensors(net_macro_idx, net_pin_offsets)
    gamma_base = max(canvas_w, canvas_h) * gamma_frac
    setup_time = time.time() - t_setup0

    def _interp_schedule(schedule, frac, default):
        if not schedule:
            return default
        for i in range(len(schedule) - 1):
            f0, v0 = schedule[i]
            f1, v1 = schedule[i + 1]
            if f0 <= frac <= f1:
                if f1 == f0:
                    return v1
                t = (frac - f0) / (f1 - f0)
                return v0 + t * (v1 - v0)
        return schedule[-1][1]

    if log_every > 0:
        n_nets = len(net_macro_idx)
        max_pins = padded_data[0].shape[1] if padded_data is not None else 0
        ls = lambda_schedule or [("const", lambda_density)]
        gs = gamma_schedule or [("const", 1.0)]
        print(f"  [analytical {label}] n_hard={n_hard} n_nets={n_nets} "
              f"max_pins/net={max_pins} canvas={canvas_w:.0f}x{canvas_h:.0f} "
              f"gamma_base={gamma_base:.1f} lr={lr} steps={num_steps} "
              f"lambda_sched={ls} gamma_sched={gs} setup={setup_time:.2f}s",
              flush=True)

    for step in range(num_steps):
        t_step = time.time()
        frac = step / max(1, num_steps - 1)

        cur_lambda = _interp_schedule(lambda_schedule, frac, lambda_density)
        cur_gamma_factor = _interp_schedule(gamma_schedule, frac, 1.0)
        gamma = gamma_base * cur_gamma_factor

        optimizer.zero_grad()
        if padded_data is not None:
            wl = _smooth_hpwl_padded(pos, padded_data, gamma)
        else:
            wl = pos.new_zeros(())
        if cur_lambda > 0:
            dpen = _density_penalty(
                pos, sizes, canvas_w, canvas_h,
                benchmark.grid_rows, benchmark.grid_cols, target_util,
            )
            loss = wl + cur_lambda * dpen
        else:
            dpen = pos.new_zeros(())
            loss = wl
        loss.backward()
        grad_norm = pos.grad.norm().item() if pos.grad is not None else 0.0
        optimizer.step()

        with torch.no_grad():
            pos[:, 0].clamp_(min=half_w, max=canvas_w - half_w)
            pos[:, 1].clamp_(min=half_h, max=canvas_h - half_h)
            pos[~movable] = fixed_pos[~movable]

        if log_every > 0 and (step % log_every == 0 or step == num_steps - 1):
            with torch.no_grad():
                wl_v = wl.item()
                dpen_v = dpen.item() if cur_lambda > 0 else 0.0
            elapsed = time.time() - t_start
            step_time = time.time() - t_step
            extra = ""
            if log_proxy and (step == num_steps - 1 or step % (log_every * 5) == 0):
                from macro_place.objective import compute_proxy_cost
                with torch.no_grad():
                    full = benchmark.macro_positions.clone()
                    full[:n_hard] = pos.detach()
                    cost = compute_proxy_cost(full, benchmark, plc)
                    extra = (f" proxy={cost['proxy_cost']:.4f}"
                             f" wl_real={cost['wirelength_cost']:.3f}"
                             f" den={cost['density_cost']:.3f}"
                             f" cong={cost['congestion_cost']:.3f}"
                             f" ovrlp={cost['overlap_count']}")
            print(f"  [analytical {label}] step={step:>4} ld={cur_lambda:>7.2f} "
                  f"gamma={gamma:>5.2f} wl_smooth={wl_v:>10.1f} "
                  f"dpen={dpen_v:>10.4f} loss={loss.item():>10.1f} "
                  f"|grad|={grad_norm:>8.2f} step_t={step_time*1000:.0f}ms "
                  f"total_t={elapsed:.1f}s{extra}", flush=True)

    if log_every > 0:
        total = time.time() - t_start
        print(f"  [analytical {label}] DONE in {total:.2f}s ({total/num_steps*1000:.1f}ms/step)",
              flush=True)

    return pos.detach().cpu().numpy().astype(np.float64)


if __name__ == "__main__":
    import sys
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    bench_name = sys.argv[1] if len(sys.argv) > 1 else "ibm03"
    bench, plc = load_benchmark_from_dir(f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}")

    pos = analytical_seed(bench, plc, num_steps=200, lr=0.3, lambda_density=0.0)
    full = bench.macro_positions.clone()
    full[:bench.num_hard_macros] = torch.tensor(pos, dtype=torch.float32)
    cost = compute_proxy_cost(full, bench, plc)
    print(f"{bench_name} analytical: proxy={cost['proxy_cost']:.4f}  wl={cost['wirelength_cost']:.4f}  "
          f"den={cost['density_cost']:.4f}  cong={cost['congestion_cost']:.4f}  overlaps={cost['overlap_count']}")
