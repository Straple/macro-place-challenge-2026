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


def analytical_seed(benchmark: Benchmark, plc, num_steps=200, lr=0.5,
                    lambda_density=0.0, gamma_frac=0.05, target_util=0.6):
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

    net_macro_idx, net_pin_offsets = _build_net_pin_tensors(benchmark, plc)
    gamma = max(canvas_w, canvas_h) * gamma_frac

    for step in range(num_steps):
        optimizer.zero_grad()
        wl = _smooth_hpwl(pos, net_macro_idx, net_pin_offsets, gamma)
        if lambda_density > 0:
            dpen = _density_penalty(
                pos, sizes, canvas_w, canvas_h,
                benchmark.grid_rows, benchmark.grid_cols, target_util,
            )
            loss = wl + lambda_density * dpen
        else:
            loss = wl
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pos[:, 0].clamp_(min=half_w, max=canvas_w - half_w)
            pos[:, 1].clamp_(min=half_h, max=canvas_h - half_h)
            pos[~movable] = fixed_pos[~movable]

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
