"""Greedy Klein-4 orientation flip optimizer (post-legalize).

Для каждого hard макроса выбираем одну из 4 plain Klein-4 ориентаций
(N, FN, S, FS) — это flips без rotation, размер макроса не меняется.

Греди раунд-робин: пройти по всем hard макросам, для каждого попробовать
4 ориентации, выбрать ту что минимизирует sum HPWL по нетам касающимся
этого макроса. Несколько проходов до сходимости (no improvement -> stop).

Soft макросы и порты не двигаем (у soft offsets всегда 0).

API:
    orientations, adjusted_offsets = orient_flip_optimize(benchmark, hard_pos, ...)

Дальше caller сам применяет adjusted_offsets к plc:
    for i, off in enumerate(adjusted_offsets):
        macro_idx = benchmark.hard_macro_indices[i]
        for slot, (ox, oy) in enumerate(off):
            pin = plc.modules_w_pins[<pin index for slot>]
            pin.set_offset(ox, oy)
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_CONGESTION = True
"""

from __future__ import annotations

import numpy as np


ORIENT_NAMES = ("N", "FN", "S", "FS")


def _flip_offsets_np(orig: np.ndarray, orient: int) -> np.ndarray:
    if orig.size == 0:
        return orig
    if orient == 0:
        return orig
    if orient == 1:
        return np.column_stack([-orig[:, 0], orig[:, 1]])
    if orient == 2:
        return -orig
    if orient == 3:
        return np.column_stack([orig[:, 0], -orig[:, 1]])
    raise ValueError(orient)


def orient_flip_optimize(benchmark,
                         hard_pos: np.ndarray,
                         soft_pos: np.ndarray | None = None,
                         rounds: int = 2,
                         verbose: bool = False):
    """Greedy orientation selection.

    Args:
        benchmark: Benchmark with .net_pin_nodes, .macro_pin_offsets,
                   .port_positions, .num_hard_macros, .num_macros.
        hard_pos: np.ndarray [n_hard, 2] — positions of hard macros (centers).
        soft_pos: np.ndarray [n_soft, 2] or None.
        rounds:   round-robin passes; early-exit if a pass changes 0.
        verbose:  print per-round HPWL/changes.

    Returns:
        orientations: list[int] of length n_hard, values 0..3 (N/FN/S/FS)
        adjusted_offsets: list[np.ndarray [n_pins_i, 2]] — new offsets after flip
    """
    n_hard = int(benchmark.num_hard_macros)
    n_total = int(benchmark.num_macros)
    n_soft = n_total - n_hard

    if not benchmark.net_pin_nodes or n_hard == 0:
        empty_offsets = [
            (benchmark.macro_pin_offsets[i].cpu().numpy()
             if benchmark.macro_pin_offsets[i].numel() > 0
             else np.zeros((0, 2), dtype=np.float32))
            for i in range(n_hard)
        ]
        return [0] * n_hard, empty_offsets

    if soft_pos is None:
        if n_soft > 0:
            soft_pos = benchmark.macro_positions[n_hard:n_total].cpu().numpy()
        else:
            soft_pos = np.zeros((0, 2), dtype=np.float32)
    if benchmark.port_positions.numel() > 0:
        port_pos = benchmark.port_positions.cpu().numpy()
    else:
        port_pos = np.zeros((0, 2), dtype=np.float32)

    orig_offsets: list[np.ndarray] = []
    for i in range(n_hard):
        t = benchmark.macro_pin_offsets[i]
        if t.numel() > 0:
            orig_offsets.append(t.cpu().numpy().astype(np.float64))
        else:
            orig_offsets.append(np.zeros((0, 2), dtype=np.float64))

    num_nets = int(benchmark.num_nets)

    net_pin_owners: list[np.ndarray] = []
    net_pin_slots: list[np.ndarray] = []
    net_pin_x: list[np.ndarray] = []
    net_pin_y: list[np.ndarray] = []
    macro_to_nets: list[list[int]] = [[] for _ in range(n_hard)]
    macro_pin_in_net: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_hard)]

    for n_idx in range(num_nets):
        t = benchmark.net_pin_nodes[n_idx]
        owners = t[:, 0].cpu().numpy().astype(np.int64)
        slots = t[:, 1].cpu().numpy().astype(np.int64)
        net_pin_owners.append(owners)
        net_pin_slots.append(slots)

        xs = np.zeros(len(owners), dtype=np.float64)
        ys = np.zeros(len(owners), dtype=np.float64)

        hard_mask = owners < n_hard
        if hard_mask.any():
            for j_idx in np.nonzero(hard_mask)[0]:
                own = int(owners[j_idx])
                slot = int(slots[j_idx])
                if slot < len(orig_offsets[own]):
                    ox, oy = orig_offsets[own][slot]
                else:
                    ox, oy = 0.0, 0.0
                xs[j_idx] = float(hard_pos[own, 0]) + ox
                ys[j_idx] = float(hard_pos[own, 1]) + oy
        soft_mask = (owners >= n_hard) & (owners < n_total)
        if soft_mask.any():
            soft_idxs = owners[soft_mask] - n_hard
            xs[soft_mask] = soft_pos[soft_idxs, 0]
            ys[soft_mask] = soft_pos[soft_idxs, 1]
        port_mask = owners >= n_total
        if port_mask.any() and port_pos.shape[0] > 0:
            pidxs = owners[port_mask] - n_total
            xs[port_mask] = port_pos[pidxs, 0]
            ys[port_mask] = port_pos[pidxs, 1]

        net_pin_x.append(xs)
        net_pin_y.append(ys)

        unique_macros = np.unique(owners[hard_mask]) if hard_mask.any() else np.array([], dtype=np.int64)
        for mm in unique_macros:
            mm_int = int(mm)
            macro_to_nets[mm_int].append(n_idx)
            pin_indices = np.nonzero(owners == mm_int)[0]
            slots_for_mm = slots[pin_indices]
            macro_pin_in_net[mm_int].append((n_idx, pin_indices, slots_for_mm))

    net_x_min = np.array(
        [float(x.min()) if x.size > 0 else 0.0 for x in net_pin_x],
        dtype=np.float64,
    )
    net_x_max = np.array(
        [float(x.max()) if x.size > 0 else 0.0 for x in net_pin_x],
        dtype=np.float64,
    )
    net_y_min = np.array(
        [float(y.min()) if y.size > 0 else 0.0 for y in net_pin_y],
        dtype=np.float64,
    )
    net_y_max = np.array(
        [float(y.max()) if y.size > 0 else 0.0 for y in net_pin_y],
        dtype=np.float64,
    )

    orientations = [0] * n_hard

    if verbose:
        initial_hpwl = float(((net_x_max - net_x_min) + (net_y_max - net_y_min)).sum())
        print(f"[orient_flip] n_hard={n_hard} num_nets={num_nets} "
              f"initial HPWL={initial_hpwl:.0f}", flush=True)

    for r in range(rounds):
        changed = 0
        for m in range(n_hard):
            if orig_offsets[m].shape[0] == 0:
                continue
            entries = macro_pin_in_net[m]
            if not entries:
                continue
            cx = float(hard_pos[m, 0])
            cy = float(hard_pos[m, 1])

            best_orient = orientations[m]
            best_local = None

            for try_orient in range(4):
                new_off = _flip_offsets_np(orig_offsets[m], try_orient)
                local_hpwl = 0.0
                for (n_idx, pin_idx, slots_for_mm) in entries:
                    xs = net_pin_x[n_idx]
                    ys = net_pin_y[n_idx]
                    new_xs = xs.copy()
                    new_ys = ys.copy()
                    for jj, slot in zip(pin_idx, slots_for_mm):
                        if slot < new_off.shape[0]:
                            ox, oy = new_off[slot]
                        else:
                            ox, oy = 0.0, 0.0
                        new_xs[jj] = cx + ox
                        new_ys[jj] = cy + oy
                    local_hpwl += (new_xs.max() - new_xs.min()) + (new_ys.max() - new_ys.min())
                if best_local is None or local_hpwl < best_local - 1e-9:
                    best_local = local_hpwl
                    best_orient = try_orient

            if best_orient != orientations[m]:
                orientations[m] = best_orient
                changed += 1
                new_off = _flip_offsets_np(orig_offsets[m], best_orient)
                for (n_idx, pin_idx, slots_for_mm) in entries:
                    xs = net_pin_x[n_idx]
                    ys = net_pin_y[n_idx]
                    for jj, slot in zip(pin_idx, slots_for_mm):
                        if slot < new_off.shape[0]:
                            ox, oy = new_off[slot]
                        else:
                            ox, oy = 0.0, 0.0
                        xs[jj] = cx + ox
                        ys[jj] = cy + oy
                    net_x_min[n_idx] = float(xs.min())
                    net_x_max[n_idx] = float(xs.max())
                    net_y_min[n_idx] = float(ys.min())
                    net_y_max[n_idx] = float(ys.max())

        if verbose:
            cur_hpwl = float(((net_x_max - net_x_min) + (net_y_max - net_y_min)).sum())
            print(f"[orient_flip] round {r+1}: changed={changed} HPWL={cur_hpwl:.0f}",
                  flush=True)
        if changed == 0:
            break

    adjusted_offsets = [
        _flip_offsets_np(orig_offsets[i], orientations[i]).astype(np.float32)
        for i in range(n_hard)
    ]
    return orientations, adjusted_offsets


def apply_orientations_to_plc(plc, benchmark, orientations: list[int]) -> int:
    """Apply chosen orientations to plc by directly setting pin offsets.

    Returns number of pins updated. After this call, set
    plc.FLAG_UPDATE_WIRELENGTH = True and plc.FLAG_UPDATE_CONGESTION = True
    so subsequent get_cost() / get_congestion_cost() recompute.
    """
    n_hard = int(benchmark.num_hard_macros)
    if not benchmark.macro_pin_offsets:
        return 0

    hard_macros_to_inpins = getattr(plc, "hard_macros_to_inpins", None)
    if hard_macros_to_inpins is None:
        return 0

    updated = 0
    for i in range(n_hard):
        macro_idx = benchmark.hard_macro_indices[i]
        macro = plc.modules_w_pins[macro_idx]
        macro_name = macro.get_name()
        pin_names = hard_macros_to_inpins.get(macro_name, [])
        if not pin_names:
            continue
        macro.set_orientation(ORIENT_NAMES[orientations[i]])

        orig = benchmark.macro_pin_offsets[i]
        if orig.numel() == 0:
            continue
        new_off = _flip_offsets_np(orig.cpu().numpy().astype(np.float64),
                                   orientations[i])
        for slot, pin_name in enumerate(pin_names):
            if slot >= new_off.shape[0]:
                break
            pin_idx = plc.mod_name_to_indices.get(pin_name, -1)
            if pin_idx < 0:
                continue
            pin = plc.modules_w_pins[pin_idx]
            pin.set_offset(float(new_off[slot, 0]), float(new_off[slot, 1]))
            updated += 1
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_CONGESTION = True
    return updated


def reset_orientations_to_n(plc, benchmark) -> int:
    """Reset all hard macro pins back to original (orientation N) offsets."""
    n_hard = int(benchmark.num_hard_macros)
    if not benchmark.macro_pin_offsets:
        return 0
    hard_macros_to_inpins = getattr(plc, "hard_macros_to_inpins", None)
    if hard_macros_to_inpins is None:
        return 0
    updated = 0
    for i in range(n_hard):
        macro_idx = benchmark.hard_macro_indices[i]
        macro = plc.modules_w_pins[macro_idx]
        macro.set_orientation("N")
        macro_name = macro.get_name()
        pin_names = hard_macros_to_inpins.get(macro_name, [])
        orig = benchmark.macro_pin_offsets[i]
        if orig.numel() == 0:
            continue
        orig_np = orig.cpu().numpy()
        for slot, pin_name in enumerate(pin_names):
            if slot >= orig_np.shape[0]:
                break
            pin_idx = plc.mod_name_to_indices.get(pin_name, -1)
            if pin_idx < 0:
                continue
            pin = plc.modules_w_pins[pin_idx]
            pin.set_offset(float(orig_np[slot, 0]), float(orig_np[slot, 1]))
            updated += 1
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_CONGESTION = True
    return updated
