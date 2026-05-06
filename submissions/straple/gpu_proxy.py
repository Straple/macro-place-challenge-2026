"""GPU batched reproduction of Google's PlacementCost proxy.

Reproduces wirelength (exact HPWL), density (rectangular cell-overlap,
top-10% mean × 0.5) and congestion (Google-style 2-pin H/V routing demand
+ macro coverage demand, smoothed and abu-aggregated) per
``plc_client_os.PlacementCost``.  All three components match Google's
formulas to within float-precision rounding, vectorised across K placements.

API:
    proxy_K, components = gpu_proxy_batched(
        pos_K, sizes_t, macro_idx_p, offsets_p, mask_p,
        canvas_w, canvas_h, grid_rows, grid_cols, num_nets,
        n_hard, edges_pkg, smooth_matrices, routing_consts,
    )

``edges_pkg`` is the result of :func:`build_routing_edges`,
``smooth_matrices`` of :func:`build_smooth_matrices`, and
``routing_consts`` of :func:`build_routing_consts` — all computed once per
benchmark.
"""

from __future__ import annotations

import math
import torch


def build_wl_pkg_full(plc, name_to_global, n_total: int):
    """Build a wirelength-only net pin package that matches CPU exactly.

    CPU iterates ``plc.nets``: each net contributes weighted HPWL of pin
    positions of (driver + sinks).  Pins can be MACRO_PIN (offset within
    parent macro) or PORT (its own absolute position, fixed).

    To accommodate ports on GPU we append fixed port positions to the
    movable macro tensor by extending the index space:
        macro idx 0 .. n_total-1   -> movable macros (use pos_K[:, idx])
        macro idx n_total + p      -> port p, position = port_positions[p]

    Returns dict with:
        port_positions [Np, 2]
        macro_idx [N_nets, max_pins]
        offsets [N_nets, max_pins, 2]
        mask [N_nets, max_pins]
        weights [N_nets]
        net_cnt_total (CPU's plc.net_cnt)
    """
    port_pos_list = []
    port_to_idx = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == "PORT":
            port_to_idx[mod.get_name()] = len(port_pos_list)
            x, y = mod.get_pos()
            port_pos_list.append([float(x), float(y)])

    nets_macros = []
    nets_offsets = []
    nets_mask = []
    nets_weights = []
    max_pins = 0

    def _resolve(pin_name):
        parent = pin_name.split("/")[0]
        # First check PORT (named directly without slash sub-path).
        if pin_name in port_to_idx:
            return n_total + port_to_idx[pin_name], (0.0, 0.0)
        if parent in port_to_idx:
            return n_total + port_to_idx[parent], (0.0, 0.0)
        if parent in name_to_global:
            pin_idx = plc.mod_name_to_indices.get(pin_name, -1)
            if pin_idx < 0:
                return None
            offset = plc.modules_w_pins[pin_idx].get_offset()
            return name_to_global[parent], (float(offset[0]), float(offset[1]))
        return None

    for driver, sinks in plc.nets.items():
        macros = []
        offs = []
        for pin_name in [driver] + list(sinks):
            r = _resolve(pin_name)
            if r is None:
                continue
            macros.append(r[0])
            offs.append(list(r[1]))
        if len(macros) < 2:
            continue
        d_idx = plc.mod_name_to_indices.get(driver, -1)
        if d_idx >= 0:
            try:
                w = float(plc.modules_w_pins[d_idx].get_weight())
            except Exception:
                w = 1.0
        else:
            w = 1.0
        if w <= 0:
            w = 1.0
        nets_macros.append(macros)
        nets_offsets.append(offs)
        nets_weights.append(w)
        max_pins = max(max_pins, len(macros))

    N = len(nets_macros)
    macro_idx = torch.zeros((N, max_pins), dtype=torch.long)
    offsets = torch.zeros((N, max_pins, 2), dtype=torch.float32)
    mask = torch.zeros((N, max_pins), dtype=torch.bool)
    for n, (m, o) in enumerate(zip(nets_macros, nets_offsets)):
        p = len(m)
        macro_idx[n, :p] = torch.tensor(m, dtype=torch.long)
        offsets[n, :p] = torch.tensor(o, dtype=torch.float32)
        mask[n, :p] = True
    weights = torch.tensor(nets_weights, dtype=torch.float32)
    port_positions = torch.tensor(port_pos_list, dtype=torch.float32) if port_pos_list else \
        torch.zeros((0, 2), dtype=torch.float32)
    return {
        "macro_idx": macro_idx,
        "offsets": offsets,
        "mask": mask,
        "weights": weights,
        "port_positions": port_positions,
        "net_cnt_total": float(getattr(plc, "net_cnt", len(plc.nets))),
        "n_total": n_total,
    }


def build_net_weights_and_count(plc, name_to_global):
    """Return per-net driver weight tensor whose row order EXACTLY matches
    ``analytical_seed._build_net_pin_tensors_full`` (i.e., included iff
    >=2 of the net's pins have a known parent macro), plus the total
    ``len(plc.nets)`` used by CPU's wirelength normalisation.
    """
    weights = []
    for driver, sinks in plc.nets.items():
        valid_pins = 0
        for pin_name in [driver] + list(sinks):
            parent = pin_name.split("/")[0]
            if parent not in name_to_global:
                continue
            if plc.mod_name_to_indices.get(pin_name, -1) < 0:
                continue
            valid_pins += 1
        if valid_pins < 2:
            continue
        d_idx = plc.mod_name_to_indices.get(driver, -1)
        if d_idx >= 0:
            try:
                w = float(plc.modules_w_pins[d_idx].get_weight())
            except Exception:
                w = 1.0
        else:
            w = 1.0
        if w <= 0:
            w = 1.0
        weights.append(w)
    # Google's PlacementCost.net_cnt is the WEIGHTED sum of nets (each
    # weighted by its driver-pin weight, default 1).  This is the number
    # used in CPU's wirelength normalisation: get_wirelength()/((W+H)*net_cnt).
    plc_net_cnt = float(getattr(plc, "net_cnt", len(plc.nets)))
    return {
        "weights": torch.tensor(weights, dtype=torch.float32),
        "net_cnt_total": plc_net_cnt,
    }


def build_routing_edges_full(plc, name_to_global, n_total: int):
    """Like :func:`build_routing_edges` but also accepts PORTS as endpoints.

    Returns two edge groups (2-pin + 3-pin) plus port_positions and the
    combined index space n_total + n_ports.  Each macro/port idx i in
    pin_*_macro is interpreted as:
        i < n_total      → pos_K[:, i] (movable macro)
        i >= n_total     → port_positions[i - n_total] (fixed)

    The packs include a ``port_positions`` tensor for the caller to
    concat with pos_K when computing pin_xy.
    """
    port_pos_list = []
    port_to_idx = {}
    for idx, mod in enumerate(plc.modules_w_pins):
        if mod.get_type() == "PORT":
            port_to_idx[mod.get_name()] = len(port_pos_list)
            x, y = mod.get_pos()
            port_pos_list.append([float(x), float(y)])

    def _resolve(pin_name):
        if pin_name in port_to_idx:
            return n_total + port_to_idx[pin_name], (0.0, 0.0)
        parent = pin_name.split("/")[0]
        if parent in port_to_idx:
            return n_total + port_to_idx[parent], (0.0, 0.0)
        if parent in name_to_global:
            pin_idx = plc.mod_name_to_indices.get(pin_name, -1)
            if pin_idx < 0:
                return None
            offset = plc.modules_w_pins[pin_idx].get_offset()
            return name_to_global[parent], (float(offset[0]), float(offset[1]))
        return None

    p2_a_m, p2_a_off = [], []
    p2_b_m, p2_b_off = [], []
    p2_w = []
    p2_net_id = []          # legacy field, all -1 now
    p3_a_m, p3_a_off = [], []
    p3_b_m, p3_b_off = [], []
    p3_c_m, p3_c_off = [], []
    p3_w = []
    mp_pin_macros = []      # ≥4-pin nets: list[list[int]] (driver first)
    mp_pin_offsets = []     # list[list[(ox, oy)]]
    mp_weights = []

    for driver, sinks in plc.nets.items():
        d_idx = plc.mod_name_to_indices.get(driver, -1)
        if d_idx < 0:
            continue
        d_pin = plc.modules_w_pins[d_idx]
        try:
            w = float(d_pin.get_weight())
        except Exception:
            w = 1.0
        if w <= 0:
            w = 1.0
        d_res = _resolve(driver)
        if d_res is None:
            continue
        d_macro, d_offset = d_res
        valid = []
        for sink in sinks:
            r = _resolve(sink)
            if r is not None:
                valid.append(r)
        if not valid:
            continue
        n_pins = 1 + len(valid)
        if n_pins == 2:
            sm, soff = valid[0]
            p2_a_m.append(d_macro); p2_a_off.append(list(d_offset))
            p2_b_m.append(sm); p2_b_off.append(list(soff))
            p2_w.append(w)
            p2_net_id.append(-1)         # 2-pin nets need no dedup
        elif n_pins == 3:
            (m1, o1), (m2, o2) = valid
            p3_a_m.append(d_macro); p3_a_off.append(list(d_offset))
            p3_b_m.append(m1); p3_b_off.append(list(o1))
            p3_c_m.append(m2); p3_c_off.append(list(o2))
            p3_w.append(w)
        else:
            # ≥4 pins: store as multi-pin group (driver + sinks).  Routing
            # is dispatched per-K based on the count of distinct grid
            # cells the pins fall into, matching Google's set(node_gcells)
            # semantics in get_routing.
            mp_pin_macros.append([d_macro] + [m for m, _ in valid])
            mp_pin_offsets.append([list(d_offset)] + [list(o) for _, o in valid])
            mp_weights.append(w)

    def _zero2(): return torch.zeros((0, 2), dtype=torch.float32)

    edges_2pin = {
        "pin_a_macro": torch.tensor(p2_a_m, dtype=torch.long),
        "pin_a_offset": torch.tensor(p2_a_off, dtype=torch.float32) if p2_a_off else _zero2(),
        "pin_b_macro": torch.tensor(p2_b_m, dtype=torch.long),
        "pin_b_offset": torch.tensor(p2_b_off, dtype=torch.float32) if p2_b_off else _zero2(),
        "weight": torch.tensor(p2_w, dtype=torch.float32),
        "net_id": torch.tensor(p2_net_id, dtype=torch.long),
    }
    edges_3pin = {
        "pin_a_macro": torch.tensor(p3_a_m, dtype=torch.long),
        "pin_a_offset": torch.tensor(p3_a_off, dtype=torch.float32) if p3_a_off else _zero2(),
        "pin_b_macro": torch.tensor(p3_b_m, dtype=torch.long),
        "pin_b_offset": torch.tensor(p3_b_off, dtype=torch.float32) if p3_b_off else _zero2(),
        "pin_c_macro": torch.tensor(p3_c_m, dtype=torch.long),
        "pin_c_offset": torch.tensor(p3_c_off, dtype=torch.float32) if p3_c_off else _zero2(),
        "weight": torch.tensor(p3_w, dtype=torch.float32),
    }

    # ---- Multi-pin (>3 pins) nets: padded driver+sinks tensor for per-K
    # ---- distinct-cells dispatch (Google set(node_gcells) semantics).
    G = len(mp_weights)
    if G > 0:
        max_pins = max(len(p) for p in mp_pin_macros)
        mp_macros_t = torch.zeros((G, max_pins), dtype=torch.long)
        mp_offsets_t = torch.zeros((G, max_pins, 2), dtype=torch.float32)
        mp_pad_t = torch.zeros((G, max_pins), dtype=torch.bool)
        for gi, (mlist, olist) in enumerate(zip(mp_pin_macros, mp_pin_offsets)):
            p = len(mlist)
            mp_macros_t[gi, :p] = torch.tensor(mlist, dtype=torch.long)
            mp_offsets_t[gi, :p] = torch.tensor(olist, dtype=torch.float32)
            mp_pad_t[gi, :p] = True
    else:
        max_pins = 0
        mp_macros_t = torch.zeros((0, 0), dtype=torch.long)
        mp_offsets_t = torch.zeros((0, 0, 2), dtype=torch.float32)
        mp_pad_t = torch.zeros((0, 0), dtype=torch.bool)
    multi_pin_pkg = {
        "pin_macro": mp_macros_t,         # [G, max_pins]
        "pin_offset": mp_offsets_t,       # [G, max_pins, 2]
        "pin_pad": mp_pad_t,              # [G, max_pins]
        "weight": torch.tensor(mp_weights, dtype=torch.float32),
        "max_pins": max_pins,
    }

    port_positions = (torch.tensor(port_pos_list, dtype=torch.float32)
                       if port_pos_list else _zero2())
    return {
        "edges_2pin": edges_2pin,
        "edges_3pin": edges_3pin,
        "multi_pin": multi_pin_pkg,
        "port_positions": port_positions,
        "n_total": n_total,
        # Back-compat
        "pin_a_macro": edges_2pin["pin_a_macro"],
        "pin_a_offset": edges_2pin["pin_a_offset"],
        "pin_b_macro": edges_2pin["pin_b_macro"],
        "pin_b_offset": edges_2pin["pin_b_offset"],
        "weight": edges_2pin["weight"],
    }


def build_routing_edges(plc, name_to_global):
    """Decompose every net into pin-tuples by Google's rules.

    Returns dict with:
      ``edges_2pin`` — (driver, sink) pairs for 2-pin nets and >3-pin
        split-routed nets.  Each entry: pin_a_*, pin_b_*, weight.
      ``edges_3pin`` — 3-pin nets stored as (pin0, pin1, pin2) tuples
        (driver included, ordered as in plc.nets); pin_a_*, pin_b_*, pin_c_*.

    All offsets are pin-offset within macro (PORTS get (0,0)).
    """
    p2_a_m, p2_a_off = [], []
    p2_b_m, p2_b_off = [], []
    p2_w = []
    p3_a_m, p3_a_off = [], []
    p3_b_m, p3_b_off = [], []
    p3_c_m, p3_c_off = [], []
    p3_w = []
    for driver, sinks in plc.nets.items():
        d_idx = plc.mod_name_to_indices.get(driver, -1)
        if d_idx < 0:
            continue
        d_pin = plc.modules_w_pins[d_idx]
        d_parent = driver.split("/")[0]
        d_macro = name_to_global.get(d_parent, -1)
        if d_macro < 0:
            continue
        d_offset = d_pin.get_offset()
        try:
            w = float(d_pin.get_weight())
        except Exception:
            w = 1.0
        if w <= 0:
            w = 1.0
        # Resolve sinks
        valid = []
        for sink in sinks:
            s_idx = plc.mod_name_to_indices.get(sink, -1)
            if s_idx < 0:
                continue
            s_parent = sink.split("/")[0]
            s_macro = name_to_global.get(s_parent, -1)
            if s_macro < 0:
                continue
            s_off = plc.modules_w_pins[s_idx].get_offset()
            valid.append((s_macro, s_off))
        if not valid:
            continue
        n_pins = 1 + len(valid)
        if n_pins == 2:
            sm, soff = valid[0]
            p2_a_m.append(d_macro); p2_a_off.append(d_offset)
            p2_b_m.append(sm); p2_b_off.append(soff)
            p2_w.append(w)
        elif n_pins == 3:
            (m1, o1), (m2, o2) = valid
            p3_a_m.append(d_macro); p3_a_off.append(d_offset)
            p3_b_m.append(m1); p3_b_off.append(o1)
            p3_c_m.append(m2); p3_c_off.append(o2)
            p3_w.append(w)
        else:
            # >3-pin: split into (driver, sink_i) 2-pin segments per Google.
            for sm, soff in valid:
                p2_a_m.append(d_macro); p2_a_off.append(d_offset)
                p2_b_m.append(sm); p2_b_off.append(soff)
                p2_w.append(w)

    def _stack(arr, dtype):
        return (torch.tensor(arr, dtype=dtype) if arr
                else torch.zeros((0,) + ((2,) if dtype == torch.float32 else ()),
                                  dtype=dtype))

    edges_2pin = {
        "pin_a_macro": torch.tensor(p2_a_m, dtype=torch.long),
        "pin_a_offset": torch.tensor(p2_a_off, dtype=torch.float32)
            if p2_a_off else torch.zeros((0, 2), dtype=torch.float32),
        "pin_b_macro": torch.tensor(p2_b_m, dtype=torch.long),
        "pin_b_offset": torch.tensor(p2_b_off, dtype=torch.float32)
            if p2_b_off else torch.zeros((0, 2), dtype=torch.float32),
        "weight": torch.tensor(p2_w, dtype=torch.float32),
    }
    edges_3pin = {
        "pin_a_macro": torch.tensor(p3_a_m, dtype=torch.long),
        "pin_a_offset": torch.tensor(p3_a_off, dtype=torch.float32)
            if p3_a_off else torch.zeros((0, 2), dtype=torch.float32),
        "pin_b_macro": torch.tensor(p3_b_m, dtype=torch.long),
        "pin_b_offset": torch.tensor(p3_b_off, dtype=torch.float32)
            if p3_b_off else torch.zeros((0, 2), dtype=torch.float32),
        "pin_c_macro": torch.tensor(p3_c_m, dtype=torch.long),
        "pin_c_offset": torch.tensor(p3_c_off, dtype=torch.float32)
            if p3_c_off else torch.zeros((0, 2), dtype=torch.float32),
        "weight": torch.tensor(p3_w, dtype=torch.float32),
    }
    return {
        "edges_2pin": edges_2pin,
        "edges_3pin": edges_3pin,
        # Back-compat aliases (legacy split-only test code looks here)
        "pin_a_macro": edges_2pin["pin_a_macro"],
        "pin_a_offset": edges_2pin["pin_a_offset"],
        "pin_b_macro": edges_2pin["pin_b_macro"],
        "pin_b_offset": edges_2pin["pin_b_offset"],
        "weight": edges_2pin["weight"],
    }


def build_smooth_matrices(grid_rows: int, grid_cols: int, smooth_range: int,
                           device=None, dtype=torch.float32):
    """Pre-compute the dense smoothing matrices used in Google's
    __smooth_routing_cong (variable-window box average).

    Returns:
        M_cols [grid_cols, grid_cols] such that V_smoothed = V @ M_cols
            (over the columns axis).
        M_rows [grid_rows, grid_rows] such that
            H_smoothed = einsum("krc,rp->kpc", H, M_rows)
            (or equivalently H @ M_rows applied along the rows axis).
    """
    def _smooth_matrix(n: int) -> torch.Tensor:
        m = torch.zeros((n, n), dtype=dtype, device=device)
        for c in range(n):
            lp = max(c - smooth_range, 0)
            rp = min(c + smooth_range, n - 1)
            cnt = rp - lp + 1
            inv = 1.0 / cnt
            for p in range(lp, rp + 1):
                m[c, p] = inv
        return m

    return {
        "cols": _smooth_matrix(grid_cols),
        "rows": _smooth_matrix(grid_rows),
        "smooth_range": smooth_range,
    }


def build_routing_consts(plc, canvas_w: float, canvas_h: float,
                          grid_rows: int, grid_cols: int):
    """Pull routing-related constants off the PlacementCost object."""
    h_per_um, v_per_um = plc.get_routes_per_micron()
    h_alloc, v_alloc = plc.get_macro_routing_allocation()
    return {
        "hroutes_per_micron": float(h_per_um),
        "vroutes_per_micron": float(v_per_um),
        "hrouting_alloc": float(h_alloc),
        "vrouting_alloc": float(v_alloc),
        "grid_h_routes": (canvas_h / grid_rows) * float(h_per_um),
        "grid_v_routes": (canvas_w / grid_cols) * float(v_per_um),
        "smooth_range": int(getattr(plc, "smooth_range", 2)),
    }


def gpu_congestion_google(
    pos_K: torch.Tensor,
    sizes_t: torch.Tensor,
    n_hard: int,
    edges_pkg: dict,
    smooth_matrices: dict,
    routing_consts: dict,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    edge_chunk: int = 1024,
    macro_chunk: int = 32,
    abu_top_pct: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Reproduce ``PlacementCost.get_congestion_cost`` on the GPU for K placements.

    Steps (matching ``plc_client_os.PlacementCost.get_routing``):
      1. Net routing demand from 2-pin (driver, sink) decomposition: for each
         pair, deposit weight into the H row of the driver and the V column
         of the sink (between the row/col extents).
      2. Macro routing demand: each hard macro contributes ``x_dist *
         vrouting_alloc`` (V) and ``y_dist * hrouting_alloc`` (H) to every
         grid cell it physically overlaps.
      3. Normalise by grid_v_routes / grid_h_routes.
      4. Smooth the *net* demand with a variable-window box average of
         radius ``smooth_range`` (Google's __smooth_routing_cong).  Macros
         are NOT smoothed.
      5. Sum smoothed-net + macro per direction → total V/H grids.
      6. abu top-5% of (V ⊕ H) flattened.
    """
    K = pos_K.shape[0]
    dev = pos_K.device
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    grid_v_routes = float(routing_consts["grid_v_routes"])
    grid_h_routes = float(routing_consts["grid_h_routes"])
    vrouting_alloc = float(routing_consts["vrouting_alloc"])
    hrouting_alloc = float(routing_consts["hrouting_alloc"])

    e2 = edges_pkg.get("edges_2pin", edges_pkg)
    e3 = edges_pkg.get("edges_3pin", None)
    port_pos = edges_pkg.get("port_positions", None)
    if port_pos is not None and port_pos.shape[0] > 0:
        port_pos_t = port_pos.to(dev, pos_K.dtype)
        ports_K = port_pos_t.unsqueeze(0).expand(K, -1, -1)
        combined_pos = torch.cat([pos_K, ports_K], dim=1)
    else:
        combined_pos = pos_K
    pin_a_m = e2["pin_a_macro"].to(dev)
    pin_a_off = e2["pin_a_offset"].to(dev)
    pin_b_m = e2["pin_b_macro"].to(dev)
    pin_b_off = e2["pin_b_offset"].to(dev)
    weights = e2["weight"].to(dev)
    E = pin_a_m.shape[0]
    # 2-pin edges always carry their full weight (2-pin nets only).
    edge_w_K = weights[None, :].expand(K, E).contiguous()

    cols_arange = torch.arange(grid_cols, device=dev)
    rows_arange = torch.arange(grid_rows, device=dev)

    V_net = torch.zeros(K, grid_rows, grid_cols, dtype=pos_K.dtype, device=dev)
    H_net = torch.zeros_like(V_net)
    H_flat = H_net.view(K, grid_rows * grid_cols)
    V_flat = V_net.view(K, grid_rows * grid_cols)

    def _h_segment(row_idx, col_lo, col_hi, w_n):
        """Add weight w_n to H_flat[k, row_idx[k,n], col] for col in [col_lo,col_hi).

        row_idx, col_lo, col_hi: long [K, n]. w_n: float [n] or [K, n].
        """
        n = row_idx.shape[1]
        if n == 0:
            return
        h_mask = ((cols_arange.view(1, 1, -1) >= col_lo[:, :, None])
                  & (cols_arange.view(1, 1, -1) < col_hi[:, :, None]))
        if w_n.dim() == 1:
            w_b = w_n[None, :, None]
        else:
            w_b = w_n[:, :, None]
        contrib = h_mask.to(pos_K.dtype) * w_b
        idx = row_idx[:, :, None] * grid_cols + cols_arange.view(1, 1, -1)
        H_flat.scatter_add_(
            1, idx.expand(K, n, grid_cols).contiguous().view(K, -1),
            contrib.view(K, -1))

    def _v_segment(col_idx, row_lo, row_hi, w_n):
        n = col_idx.shape[1]
        if n == 0:
            return
        v_mask = ((rows_arange.view(1, 1, -1) >= row_lo[:, :, None])
                  & (rows_arange.view(1, 1, -1) < row_hi[:, :, None]))
        if w_n.dim() == 1:
            w_b = w_n[None, :, None]
        else:
            w_b = w_n[:, :, None]
        contrib = v_mask.to(pos_K.dtype) * w_b
        idx = (rows_arange.view(1, 1, -1) * grid_cols
               + col_idx[:, :, None])
        V_flat.scatter_add_(
            1, idx.expand(K, n, grid_rows).contiguous().view(K, -1),
            contrib.view(K, -1))

    # ---- 1a. Net routing demand (2-pin) ----
    for s in range(0, E, edge_chunk):
        e = min(s + edge_chunk, E)
        a_m = pin_a_m[s:e]
        b_m = pin_b_m[s:e]
        a_off = pin_a_off[s:e]
        b_off = pin_b_off[s:e]
        w_chunk = edge_w_K[:, s:e]   # [K, e_n] — already dedup-applied
        a_xy = combined_pos[:, a_m, :] + a_off[None, :, :]
        b_xy = combined_pos[:, b_m, :] + b_off[None, :, :]
        a_col = (a_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        a_row = (a_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()
        b_col = (b_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        b_row = (b_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()
        col_min = torch.minimum(a_col, b_col)
        col_max = torch.maximum(a_col, b_col)
        row_min = torch.minimum(a_row, b_row)
        row_max = torch.maximum(a_row, b_row)
        # H along source row (driver = a) from col_min to col_max-1
        _h_segment(a_row, col_min, col_max, w_chunk)
        # V along sink col (sink = b) from row_min to row_max-1
        _v_segment(b_col, row_min, row_max, w_chunk)

    # ---- 1b. Net routing demand (3-pin) — Google L/T/V/double-L cases ----
    # Note: Google collapses duplicate cells via set() before deciding the
    # routing scheme.  Three pins that land in 2 distinct cells become a
    # 2-pin net; three pins in 1 cell are skipped.  We detect this before
    # dispatching to L/T routing and route the collapsed nets as 2-pin.
    if e3 is not None and e3["pin_a_macro"].numel() > 0:
        a3 = e3["pin_a_macro"].to(dev)
        b3 = e3["pin_b_macro"].to(dev)
        c3 = e3["pin_c_macro"].to(dev)
        a3_off = e3["pin_a_offset"].to(dev)
        b3_off = e3["pin_b_offset"].to(dev)
        c3_off = e3["pin_c_offset"].to(dev)
        w3 = e3["weight"].to(dev)
        N3 = a3.shape[0]
        # Per-K cell coords
        a_xy = combined_pos[:, a3, :] + a3_off[None, :, :]
        b_xy = combined_pos[:, b3, :] + b3_off[None, :, :]
        c_xy = combined_pos[:, c3, :] + c3_off[None, :, :]
        a_col = (a_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        a_row = (a_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()
        b_col = (b_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        b_row = (b_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()
        c_col = (c_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        c_row = (c_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()

        # ----- Duplicate-cell collapse (mimics Google's set(node_gcells)) -----
        eq_ab = (a_col == b_col) & (a_row == b_row)
        eq_ac = (a_col == c_col) & (a_row == c_row)
        eq_bc = (b_col == c_col) & (b_row == c_row)
        # Number of distinct cells per net per K
        n_distinct = 3 - eq_ab.long() - (eq_ac & ~eq_ab).long() - (
            eq_bc & ~eq_ab & ~eq_ac).long()
        is_3 = (n_distinct == 3)
        is_2 = (n_distinct == 2)
        # For 2-distinct case, route as 2-pin (driver source = a) between the
        # two unique cells.  Pick the "other" cell:
        #   if eq_ab → other = c
        #   elif eq_ac → other = b
        #   elif eq_bc → other = b (a vs b)
        # Source row stays a_row (driver acts as source per Google's __split_net,
        # but here for 3-pin Google uses sorted gcells; collapsing to 2-pin
        # we follow Google's __two_pin_net_routing with a as source).
        other_col = torch.where(eq_ab, c_col,
                      torch.where(eq_ac, b_col, b_col))
        other_row = torch.where(eq_ab, c_row,
                      torch.where(eq_ac, b_row, b_row))
        col_min2 = torch.minimum(a_col, other_col)
        col_max2 = torch.maximum(a_col, other_col)
        row_min2 = torch.minimum(a_row, other_row)
        row_max2 = torch.maximum(a_row, other_row)
        w2_collapsed = w3[None, :] * is_2.to(pos_K.dtype)
        _h_segment(a_row, col_min2, col_max2, w2_collapsed)
        _v_segment(other_col, row_min2, row_max2, w2_collapsed)

        # Sort the 3 pins by (col, row): build sort key, argsort, gather rows/cols.
        # Key = col * (grid_rows+1) + row; ties broken by row.
        max_row = grid_rows + 1
        key = torch.stack([
            a_col * max_row + a_row,
            b_col * max_row + b_row,
            c_col * max_row + c_row,
        ], dim=-1)                                  # [K, N3, 3]
        rows_stack = torch.stack([a_row, b_row, c_row], dim=-1)
        cols_stack = torch.stack([a_col, b_col, c_col], dim=-1)
        order = key.argsort(dim=-1, stable=True)                 # [K, N3, 3]
        sorted_rows = torch.gather(rows_stack, -1, order)
        sorted_cols = torch.gather(cols_stack, -1, order)
        x1, x2, x3 = sorted_cols.unbind(-1)
        y1, y2, y3 = sorted_rows.unbind(-1)

        case_L = ((x1 < x2) & (x2 < x3)
                  & (torch.minimum(y1, y3) < y2)
                  & (torch.maximum(y1, y3) > y2))
        case_Vx2 = ((x2 == x3) & (x1 < x2)
                    & (y1 < torch.minimum(y2, y3)))
        case_doubleL = (y2 == y3) & ~case_L & ~case_Vx2
        case_T = ~ (case_L | case_Vx2 | case_doubleL)

        # Apply weights only where ALL 3 cells are distinct.
        only3 = is_3.to(pos_K.dtype)
        wL = w3[None, :] * case_L.to(pos_K.dtype) * only3
        wVx2 = w3[None, :] * case_Vx2.to(pos_K.dtype) * only3
        wdL = w3[None, :] * case_doubleL.to(pos_K.dtype) * only3
        wT = w3[None, :] * case_T.to(pos_K.dtype) * only3

        # Case L (Google __l_routing):
        #   H (x1, y1)..(x2, y1)        -> row=y1, col [x1, x2)
        #   H (x2, y2)..(x3, y2)        -> row=y2, col [x2, x3)
        #   V at x2, rows [min(y1,y2), max(y1,y2))
        #   V at x3, rows [min(y2,y3), max(y2,y3))
        _h_segment(y1, x1, x2, wL)
        _h_segment(y2, x2, x3, wL)
        _v_segment(x2, torch.minimum(y1, y2), torch.maximum(y1, y2), wL)
        _v_segment(x3, torch.minimum(y2, y3), torch.maximum(y2, y3), wL)

        # Case Vx2 (x2 == x3, vertical run at x2):
        #   H (x1, y1)..(x2, y1)
        #   V at x2, rows [y1, max(y2,y3))
        _h_segment(y1, x1, x2, wVx2)
        _v_segment(x2, y1, torch.maximum(y2, y3), wVx2)

        # Case double-L (y2 == y3):
        #   H (x1, y1)..(x2, y1)
        #   H (x2, y2)..(x3, y2)
        #   V at x2, rows [min(y1, y2), max(y1, y2))
        _h_segment(y1, x1, x2, wdL)
        _h_segment(y2, x2, x3, wdL)
        _v_segment(x2, torch.minimum(y1, y2), torch.maximum(y1, y2), wdL)

        # Case T (default __t_routing):
        #   sort by row first: tx1<=tx2<=tx3 with their rows: actually
        #   Google's __t_routing sorts by default (row, col) since
        #   list.sort() with no key uses tuple order = (y, x).
        # Recompute sort by (row, col) for t_routing.
        key_yx = torch.stack([
            a_row * (grid_cols + 1) + a_col,
            b_row * (grid_cols + 1) + b_col,
            c_row * (grid_cols + 1) + c_col,
        ], dim=-1)
        order_yx = key_yx.argsort(dim=-1, stable=True)
        sr = torch.gather(rows_stack, -1, order_yx)
        sc = torch.gather(cols_stack, -1, order_yx)
        ty1, ty2, ty3 = sr.unbind(-1)
        tx1, tx2, tx3 = sc.unbind(-1)
        xmin_T = torch.minimum(torch.minimum(tx1, tx2), tx3)
        xmax_T = torch.maximum(torch.maximum(tx1, tx2), tx3)
        # Only contributes for T-case rows.  We index using (k, n) but
        # only care where case_T (after re-sort the case_T mask is the
        # SAME bool over the same (k, n) pair because case is decided
        # geometrically — sort does not change the case decision).
        _h_segment(ty2, xmin_T, xmax_T, wT)
        _v_segment(tx1, torch.minimum(ty1, ty2), torch.maximum(ty1, ty2), wT)
        _v_segment(tx3, torch.minimum(ty2, ty3), torch.maximum(ty2, ty3), wT)

    # ---- 1c. Net routing demand (≥4 pins) — per-K dynamic dispatch
    # ---- following Google's `set(node_gcells)` semantics:
    # ----   distinct == 2 → 2-pin route between the 2 unique cells
    # ----   distinct == 3 → 3-pin L/T/V/double-L on the 3 unique cells
    # ----   distinct ≥ 4 → split-net (driver → each unique sink cell)
    mp = edges_pkg.get("multi_pin", None)
    if mp is not None and int(mp.get("max_pins", 0)) >= 2 and mp["weight"].numel() > 0:
        mp_macro = mp["pin_macro"].to(dev)         # [G, P]
        mp_off = mp["pin_offset"].to(dev)
        mp_pad = mp["pin_pad"].to(dev)
        mp_w = mp["weight"].to(dev)                # [G]
        G_mp, P_mp = mp_macro.shape

        # Per-K pin coords / cells.
        mp_xy = combined_pos[:, mp_macro.flatten(), :].view(K, G_mp, P_mp, 2) \
            + mp_off[None, :, :, :]                 # [K, G, P, 2]
        mp_col = (mp_xy[..., 0] / cell_w).floor().clamp(0, grid_cols - 1).long()
        mp_row = (mp_xy[..., 1] / cell_h).floor().clamp(0, grid_rows - 1).long()

        # Cell hash; pad pins get a sentinel so they never collide with real cells.
        SENTINEL = grid_rows * grid_cols
        cell_hash = mp_row * grid_cols + mp_col      # [K, G, P]
        pad_b = mp_pad[None, :, :].expand(K, -1, -1)
        cell_hash = torch.where(pad_b, cell_hash,
                                 torch.full_like(cell_hash, SENTINEL))

        # Sort cell hashes within each (K, G); pads (sentinel) end up last.
        sorted_hash, sort_idx = cell_hash.sort(dim=-1, stable=True)
        prev_hash = torch.cat([
            torch.full((K, G_mp, 1), -1, device=dev, dtype=sorted_hash.dtype),
            sorted_hash[..., :-1]], dim=-1)
        is_first = (sorted_hash != prev_hash) & (sorted_hash != SENTINEL)
        n_distinct = is_first.sum(dim=-1)            # [K, G]

        d_row = mp_row[..., 0]
        d_col = mp_col[..., 0]

        # Gather rows/cols by sort_idx for distinct-rep extraction.
        gathered_row = torch.gather(mp_row, -1, sort_idx)
        gathered_col = torch.gather(mp_col, -1, sort_idx)
        rank = is_first.cumsum(dim=-1)               # [K, G, P]; rank-of-each
        rep_max = min(4, P_mp)
        rep_row = torch.zeros((K, G_mp, rep_max), device=dev, dtype=torch.long)
        rep_col = torch.zeros((K, G_mp, rep_max), device=dev, dtype=torch.long)
        for j in range(rep_max):
            target_mask = (rank == (j + 1)) & is_first      # [K, G, P]
            slot = target_mask.long().argmax(dim=-1)         # [K, G]
            rep_row[..., j] = torch.gather(gathered_row, -1,
                                            slot.unsqueeze(-1)).squeeze(-1)
            rep_col[..., j] = torch.gather(gathered_col, -1,
                                            slot.unsqueeze(-1)).squeeze(-1)

        # ---- distinct == 2 ----
        case2 = (n_distinct == 2)
        if case2.any():
            w2 = mp_w[None, :] * case2.to(pos_K.dtype)
            r0_eq_d = ((rep_row[..., 0] == d_row) & (rep_col[..., 0] == d_col))
            other_row = torch.where(r0_eq_d, rep_row[..., 1], rep_row[..., 0])
            other_col = torch.where(r0_eq_d, rep_col[..., 1], rep_col[..., 0])
            cmin = torch.minimum(d_col, other_col)
            cmax = torch.maximum(d_col, other_col)
            rmin = torch.minimum(d_row, other_row)
            rmax = torch.maximum(d_row, other_row)
            _h_segment(d_row, cmin, cmax, w2)
            _v_segment(other_col, rmin, rmax, w2)

        # ---- distinct == 3 ----
        case3 = (n_distinct == 3)
        if case3.any():
            w3_mp = mp_w[None, :] * case3.to(pos_K.dtype)
            r0r, r0c = rep_row[..., 0], rep_col[..., 0]
            r1r, r1c = rep_row[..., 1], rep_col[..., 1]
            r2r, r2c = rep_row[..., 2], rep_col[..., 2]
            mr_max = grid_rows + 1
            keys_xy = torch.stack([
                r0c * mr_max + r0r,
                r1c * mr_max + r1r,
                r2c * mr_max + r2r,
            ], dim=-1)
            rows_stk = torch.stack([r0r, r1r, r2r], dim=-1)
            cols_stk = torch.stack([r0c, r1c, r2c], dim=-1)
            ord_xy = keys_xy.argsort(dim=-1, stable=True)
            sr_xy = torch.gather(rows_stk, -1, ord_xy)
            sc_xy = torch.gather(cols_stk, -1, ord_xy)
            xx1, xx2, xx3 = sc_xy.unbind(-1)
            yy1, yy2, yy3 = sr_xy.unbind(-1)
            cL = ((xx1 < xx2) & (xx2 < xx3)
                  & (torch.minimum(yy1, yy3) < yy2)
                  & (torch.maximum(yy1, yy3) > yy2))
            cVx2 = ((xx2 == xx3) & (xx1 < xx2)
                    & (yy1 < torch.minimum(yy2, yy3)))
            cdL = (yy2 == yy3) & ~cL & ~cVx2
            cT = ~ (cL | cVx2 | cdL)
            wL = w3_mp * cL.to(pos_K.dtype)
            wVx2 = w3_mp * cVx2.to(pos_K.dtype)
            wdL = w3_mp * cdL.to(pos_K.dtype)
            wT = w3_mp * cT.to(pos_K.dtype)

            keys_yx = torch.stack([
                r0r * (grid_cols + 1) + r0c,
                r1r * (grid_cols + 1) + r1c,
                r2r * (grid_cols + 1) + r2c,
            ], dim=-1)
            ord_yx = keys_yx.argsort(dim=-1, stable=True)
            sr_t = torch.gather(rows_stk, -1, ord_yx)
            sc_t = torch.gather(cols_stk, -1, ord_yx)
            ty1, ty2_, ty3 = sr_t.unbind(-1)
            tx1, tx2_, tx3 = sc_t.unbind(-1)

            _h_segment(yy1, xx1, xx2, wL)
            _h_segment(yy2, xx2, xx3, wL)
            _v_segment(xx2, torch.minimum(yy1, yy2),
                        torch.maximum(yy1, yy2), wL)
            _v_segment(xx3, torch.minimum(yy2, yy3),
                        torch.maximum(yy2, yy3), wL)
            _h_segment(yy1, xx1, xx2, wVx2)
            _v_segment(xx2, yy1, torch.maximum(yy2, yy3), wVx2)
            _h_segment(yy1, xx1, xx2, wdL)
            _h_segment(yy2, xx2, xx3, wdL)
            _v_segment(xx2, torch.minimum(yy1, yy2),
                        torch.maximum(yy1, yy2), wdL)
            xmin_T = torch.minimum(torch.minimum(tx1, tx2_), tx3)
            xmax_T = torch.maximum(torch.maximum(tx1, tx2_), tx3)
            _h_segment(ty2_, xmin_T, xmax_T, wT)
            _v_segment(tx1, torch.minimum(ty1, ty2_),
                        torch.maximum(ty1, ty2_), wT)
            _v_segment(tx3, torch.minimum(ty2_, ty3),
                        torch.maximum(ty2_, ty3), wT)

        # ---- distinct ≥ 4 ----
        case4 = (n_distinct >= 4)
        if case4.any():
            d_hash = d_row * grid_cols + d_col            # [K, G]
            sink_mask = (is_first
                         & (sorted_hash != d_hash.unsqueeze(-1))
                         & (sorted_hash != SENTINEL))     # [K, G, P]
            w4_mp = mp_w[None, :] * case4.to(pos_K.dtype)
            d_row_b = d_row.unsqueeze(-1).expand(-1, -1, P_mp).reshape(K, -1)
            d_col_b = d_col.unsqueeze(-1).expand(-1, -1, P_mp).reshape(K, -1)
            s_row_b = gathered_row.reshape(K, -1)
            s_col_b = gathered_col.reshape(K, -1)
            mw_b = (w4_mp.unsqueeze(-1).expand(-1, -1, P_mp).reshape(K, -1)
                     * sink_mask.reshape(K, -1).to(pos_K.dtype))
            cmin4 = torch.minimum(d_col_b, s_col_b)
            cmax4 = torch.maximum(d_col_b, s_col_b)
            rmin4 = torch.minimum(d_row_b, s_row_b)
            rmax4 = torch.maximum(d_row_b, s_row_b)
            # Chunk over edges to keep mask tensors [K, chunk, grid_dim]
            # under a reasonable memory footprint.
            n_e4 = d_row_b.shape[1]
            for s in range(0, n_e4, edge_chunk):
                e = min(s + edge_chunk, n_e4)
                _h_segment(d_row_b[:, s:e], cmin4[:, s:e],
                            cmax4[:, s:e], mw_b[:, s:e])
                _v_segment(s_col_b[:, s:e], rmin4[:, s:e],
                            rmax4[:, s:e], mw_b[:, s:e])

    # ---- 2. Macro routing demand (hard macros only) ----
    V_macro = torch.zeros_like(V_net)
    H_macro = torch.zeros_like(H_net)
    half_w = sizes_t[:n_hard, 0] / 2.0
    half_h = sizes_t[:n_hard, 1] / 2.0
    macro_x = pos_K[:, :n_hard, 0]
    macro_y = pos_K[:, :n_hard, 1]
    macro_x_lo = macro_x - half_w[None, :]
    macro_x_hi = macro_x + half_w[None, :]
    macro_y_lo = macro_y - half_h[None, :]
    macro_y_hi = macro_y + half_h[None, :]
    grid_x_lo = (cols_arange.to(pos_K.dtype) * cell_w)
    grid_x_hi = grid_x_lo + cell_w
    grid_y_lo = (rows_arange.to(pos_K.dtype) * cell_h)
    grid_y_hi = grid_y_lo + cell_h

    # Per-macro grid-cell extents (BL/UR rows and cols) for partial-overlap.
    bl_row = (macro_y_lo / cell_h).floor().clamp(0, grid_rows - 1).long()
    ur_row = (macro_y_hi / cell_h).floor().clamp(0, grid_rows - 1).long()
    bl_col = (macro_x_lo / cell_w).floor().clamp(0, grid_cols - 1).long()
    ur_col = (macro_x_hi / cell_w).floor().clamp(0, grid_cols - 1).long()
    multi_row = (ur_row != bl_row)
    multi_col = (ur_col != bl_col)
    eps_overlap = 1e-5
    # Partial-vertical: macro spans >1 rows AND its bl_row or ur_row is
    # not fully covered (y_dist < cell_h there).
    y_lo_partial = (macro_y_lo - bl_row.to(pos_K.dtype) * cell_h).abs() > eps_overlap
    y_hi_partial = ((ur_row.to(pos_K.dtype) + 1) * cell_h - macro_y_hi).abs() > eps_overlap
    partial_v = multi_row & (y_lo_partial | y_hi_partial)
    x_lo_partial = (macro_x_lo - bl_col.to(pos_K.dtype) * cell_w).abs() > eps_overlap
    x_hi_partial = ((ur_col.to(pos_K.dtype) + 1) * cell_w - macro_x_hi).abs() > eps_overlap
    partial_h = multi_col & (x_lo_partial | x_hi_partial)
    V_macro_flat = V_macro.view(K, grid_rows * grid_cols)
    H_macro_flat = H_macro.view(K, grid_rows * grid_cols)

    for s in range(0, n_hard, macro_chunk):
        e = min(s + macro_chunk, n_hard)
        ix_lo = macro_x_lo[:, s:e, None, None]
        ix_hi = macro_x_hi[:, s:e, None, None]
        iy_lo = macro_y_lo[:, s:e, None, None]
        iy_hi = macro_y_hi[:, s:e, None, None]
        ox = (torch.minimum(ix_hi, grid_x_hi[None, None, None, :])
              - torch.maximum(ix_lo, grid_x_lo[None, None, None, :]))
        oy = (torch.minimum(iy_hi, grid_y_hi[None, None, :, None])
              - torch.maximum(iy_lo, grid_y_lo[None, None, :, None]))
        ox_pos = ox.clamp_min(0.0)        # [K, c, 1, ncols]
        oy_pos = oy.clamp_min(0.0)        # [K, c, nrows, 1]
        touched_x = (ox_pos > 0).to(pos_K.dtype)
        touched_y = (oy_pos > 0).to(pos_K.dtype)
        touched_2d = touched_x * touched_y
        V_macro = V_macro + (ox_pos * vrouting_alloc * touched_2d).sum(dim=1)
        H_macro = H_macro + (oy_pos * hrouting_alloc * touched_2d).sum(dim=1)

    # Refresh views after .view() may have decoupled from updated tensors.
    V_macro_flat = V_macro.view(K, grid_rows * grid_cols)
    H_macro_flat = H_macro.view(K, grid_rows * grid_cols)

    # Partial-overlap subtraction for V_macro: in row=ur_row, col in [bl_col, ur_col],
    # subtract x_dist * vrouting_alloc (where x_dist is overlap of macro with cell col).
    for s in range(0, n_hard, macro_chunk):
        e = min(s + macro_chunk, n_hard)
        ix_lo = macro_x_lo[:, s:e, None]
        ix_hi = macro_x_hi[:, s:e, None]
        ox_cells = (torch.minimum(ix_hi, grid_x_hi[None, None, :])
                    - torch.maximum(ix_lo, grid_x_lo[None, None, :])).clamp_min(0.0)
        # mask cells in [bl_col, ur_col]
        bl_c = bl_col[:, s:e, None]
        ur_c = ur_col[:, s:e, None]
        mask_c_v = ((cols_arange.view(1, 1, -1) >= bl_c)
                    & (cols_arange.view(1, 1, -1) <= ur_c))
        pv = partial_v[:, s:e, None].to(pos_K.dtype)
        contrib_sub_v = -(ox_cells * vrouting_alloc
                           * mask_c_v.to(pos_K.dtype) * pv)
        ur_r = ur_row[:, s:e, None].expand(-1, -1, grid_cols)
        flat_idx_v_sub = (ur_r * grid_cols
                          + cols_arange.view(1, 1, -1).expand(K, e - s,
                                                              grid_cols))
        V_macro_flat.scatter_add_(
            1, flat_idx_v_sub.contiguous().view(K, -1),
            contrib_sub_v.view(K, -1))

    # Partial-overlap subtraction for H_macro: in col=ur_col, rows in [bl_row, ur_row],
    # subtract y_dist * hrouting_alloc.
    for s in range(0, n_hard, macro_chunk):
        e = min(s + macro_chunk, n_hard)
        iy_lo = macro_y_lo[:, s:e, None]
        iy_hi = macro_y_hi[:, s:e, None]
        oy_cells = (torch.minimum(iy_hi, grid_y_hi[None, None, :])
                    - torch.maximum(iy_lo, grid_y_lo[None, None, :])).clamp_min(0.0)
        bl_r = bl_row[:, s:e, None]
        ur_r = ur_row[:, s:e, None]
        mask_r_h = ((rows_arange.view(1, 1, -1) >= bl_r)
                    & (rows_arange.view(1, 1, -1) <= ur_r))
        ph = partial_h[:, s:e, None].to(pos_K.dtype)
        contrib_sub_h = -(oy_cells * hrouting_alloc
                           * mask_r_h.to(pos_K.dtype) * ph)
        ur_c = ur_col[:, s:e, None].expand(-1, -1, grid_rows)
        flat_idx_h_sub = (rows_arange.view(1, 1, -1).expand(K, e - s, grid_rows)
                          * grid_cols + ur_c)
        H_macro_flat.scatter_add_(
            1, flat_idx_h_sub.contiguous().view(K, -1),
            contrib_sub_h.view(K, -1))

    # ---- 3. Normalise ----
    V_net = V_net / grid_v_routes
    H_net = H_net / grid_h_routes
    V_macro = V_macro / grid_v_routes
    H_macro = H_macro / grid_h_routes

    # ---- 4. Smooth net demand only ----
    M_cols = smooth_matrices["cols"].to(dev, pos_K.dtype)
    M_rows = smooth_matrices["rows"].to(dev, pos_K.dtype)
    # V_smoothed[k, r, p] = sum_c V_net[k, r, c] * M_cols[c, p]
    V_smoothed = torch.einsum("krc,cp->krp", V_net, M_cols)
    # H_smoothed[k, p, c] = sum_r H_net[k, r, c] * M_rows[r, p]
    H_smoothed = torch.einsum("krc,rp->kpc", H_net, M_rows)

    # ---- 5. Combine smoothed net + macro ----
    V_total = V_smoothed + V_macro
    H_total = H_smoothed + H_macro

    # ---- 6. abu top-5% of (V ⊕ H) ----
    flat_cong = torch.cat([V_total.reshape(K, -1),
                            H_total.reshape(K, -1)], dim=1)
    n_total_cells = flat_cong.shape[1]
    top_n = max(1, math.floor(n_total_cells * abu_top_pct))
    top, _ = flat_cong.topk(top_n, dim=1)
    cong_K = top.mean(dim=1)
    return cong_K, {
        "V_total": V_total,
        "H_total": H_total,
        "V_net_smoothed": V_smoothed,
        "H_net_smoothed": H_smoothed,
        "V_macro": V_macro,
        "H_macro": H_macro,
    }


def gpu_proxy_batched(
    pos_K: torch.Tensor,
    sizes_t: torch.Tensor,
    macro_idx_p: torch.Tensor,
    offsets_p: torch.Tensor,
    mask_p: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    num_nets: int,
    n_hard: int = 0,
    edges_pkg: dict | None = None,
    smooth_matrices: dict | None = None,
    routing_consts: dict | None = None,
    net_weights: torch.Tensor | None = None,
    wl_norm_net_cnt: int | None = None,
    wl_pkg: dict | None = None,
    cong_smooth_sigma_frac: float = 0.5,
    cong_top_pct: float = 0.10,
    chunk_n: int = 96,
) -> tuple[torch.Tensor, dict]:
    """Compute approximate Google proxy_cost for a batch of K placements.

    Args:
        pos_K:        [K, n_total, 2] macro centers (movable + fixed).
        sizes_t:      [n_total, 2] (width, height).
        macro_idx_p:  [num_nets, max_pins] long, index of macro per pin
                      (padded with 0 where mask_p is False).
        offsets_p:    [num_nets, max_pins, 2] pin offset from macro center.
        mask_p:       [num_nets, max_pins] bool, valid pin mask.
        canvas_w/h:   canvas dimensions.
        grid_rows/cols: density/congestion grid resolution
                      (must match Google's PlacementCost grid for consistency).
        num_nets:     scalar, used in WL normalization.
        cong_smooth_sigma_frac: σ for smooth bbox-cell indicator (×cell size).
        cong_top_pct: top-percentage for density / congestion abu.
        chunk_n:      chunk over macros for density compute (keeps memory
                      footprint manageable on T4).

    Returns:
        proxy_K [K], components dict.
    """
    K, n_total, _ = pos_K.shape
    dev = pos_K.device
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows
    cell_area = cell_w * cell_h

    # ===== 1. Wirelength (exact HPWL) =====
    if wl_pkg is not None:
        wl_macro_idx = wl_pkg["macro_idx"].to(dev)
        wl_offsets = wl_pkg["offsets"].to(dev)
        wl_mask = wl_pkg["mask"].to(dev)
        wl_weights_t = wl_pkg["weights"].to(dev, pos_K.dtype)
        wl_port_pos = wl_pkg["port_positions"].to(dev, pos_K.dtype)
        wl_n_total = int(wl_pkg["n_total"])
        # Combined positions: pos_K extended with port positions (broadcast over K).
        if wl_port_pos.shape[0] > 0:
            ports_K = wl_port_pos.unsqueeze(0).expand(K, -1, -1)
            combined_pos = torch.cat([pos_K, ports_K], dim=1)
        else:
            combined_pos = pos_K
        pin_xy = combined_pos[:, wl_macro_idx, :] + wl_offsets[None, ...]
        mask_used = wl_mask[None, :, :]
        x = pin_xy[..., 0]
        y = pin_xy[..., 1]
        neg_inf = torch.finfo(x.dtype).min
        pos_inf = torch.finfo(x.dtype).max
        x_max = torch.where(mask_used, x, x.new_full((), neg_inf)).amax(dim=2)
        x_min = torch.where(mask_used, x, x.new_full((), pos_inf)).amin(dim=2)
        y_max = torch.where(mask_used, y, x.new_full((), neg_inf)).amax(dim=2)
        y_min = torch.where(mask_used, y, x.new_full((), pos_inf)).amin(dim=2)
        hpwl_per_net = ((x_max - x_min) + (y_max - y_min)) * wl_weights_t[None, :]
        hpwl_K = hpwl_per_net.sum(dim=1)
        norm_cnt = max(float(wl_pkg["net_cnt_total"]), 1.0)
        wl_K = hpwl_K / ((canvas_w + canvas_h) * norm_cnt)
        # Re-derive bounds for the (legacy) congestion fallback below.
        # We need x_max/x_min/y_max/y_min over the macro-only nets if it
        # is going to use them; recompute using macro_idx_p as before.
        pin_xy_legacy = pos_K[:, macro_idx_p, :] + offsets_p[None, ...]
        xl = pin_xy_legacy[..., 0]
        yl = pin_xy_legacy[..., 1]
        mlk = mask_p[None, :, :]
        x_max = torch.where(mlk, xl, xl.new_full((), neg_inf)).amax(dim=2)
        x_min = torch.where(mlk, xl, xl.new_full((), pos_inf)).amin(dim=2)
        y_max = torch.where(mlk, yl, yl.new_full((), neg_inf)).amax(dim=2)
        y_min = torch.where(mlk, yl, yl.new_full((), pos_inf)).amin(dim=2)
    else:
        # pin_xy [K, num_nets, max_pins, 2] = pos[K, macro_idx_p, :] + offsets_p
        pin_xy = pos_K[:, macro_idx_p, :] + offsets_p[None, ...]
        x = pin_xy[..., 0]
        y = pin_xy[..., 1]
        neg_inf = torch.finfo(x.dtype).min
        pos_inf = torch.finfo(x.dtype).max
        mask_K = mask_p[None, :, :]
        x_max = torch.where(mask_K, x, x.new_full((), neg_inf)).amax(dim=2)
        x_min = torch.where(mask_K, x, x.new_full((), pos_inf)).amin(dim=2)
        y_max = torch.where(mask_K, y, x.new_full((), neg_inf)).amax(dim=2)
        y_min = torch.where(mask_K, y, x.new_full((), pos_inf)).amin(dim=2)
        hpwl_per_net = (x_max - x_min) + (y_max - y_min)
        if net_weights is not None:
            w_net = net_weights.to(pos_K.device, pos_K.dtype)
            if w_net.shape[0] == hpwl_per_net.shape[1]:
                hpwl_per_net = hpwl_per_net * w_net[None, :]
        hpwl_K = hpwl_per_net.sum(dim=1)
        norm_cnt = (wl_norm_net_cnt if wl_norm_net_cnt is not None
                     else max(num_nets, 1))
        norm_cnt = max(float(norm_cnt), 1.0)
        wl_K = hpwl_K / ((canvas_w + canvas_h) * norm_cnt)

    # ===== 2. Density (exact rectangular overlap per cell) =====
    half_w = sizes_t[:, 0] / 2.0
    half_h = sizes_t[:, 1] / 2.0
    macro_x_lo = pos_K[:, :, 0] - half_w[None, :]                  # [K, n]
    macro_x_hi = pos_K[:, :, 0] + half_w[None, :]
    macro_y_lo = pos_K[:, :, 1] - half_h[None, :]
    macro_y_hi = pos_K[:, :, 1] + half_h[None, :]

    grid_x_lo = (torch.arange(grid_cols, device=dev,
                               dtype=pos_K.dtype) * cell_w)        # [ncols]
    grid_x_hi = grid_x_lo + cell_w
    grid_y_lo = (torch.arange(grid_rows, device=dev,
                               dtype=pos_K.dtype) * cell_h)        # [nrows]
    grid_y_hi = grid_y_lo + cell_h

    cell_overlap = torch.zeros(K, grid_rows, grid_cols,
                                dtype=pos_K.dtype, device=dev)
    for i0 in range(0, n_total, chunk_n):
        i1 = min(i0 + chunk_n, n_total)
        ix_lo = macro_x_lo[:, i0:i1, None, None]                   # [K, c, 1, 1]
        ix_hi = macro_x_hi[:, i0:i1, None, None]
        iy_lo = macro_y_lo[:, i0:i1, None, None]
        iy_hi = macro_y_hi[:, i0:i1, None, None]
        ox = (torch.minimum(ix_hi, grid_x_hi[None, None, None, :])
              - torch.maximum(ix_lo, grid_x_lo[None, None, None, :])).clamp_min(0.0)
        oy = (torch.minimum(iy_hi, grid_y_hi[None, None, :, None])
              - torch.maximum(iy_lo, grid_y_lo[None, None, :, None])).clamp_min(0.0)
        cell_overlap += (ox * oy).sum(dim=1)
    cell_density = cell_overlap / cell_area                         # [K, nrows, ncols]

    # density_cost = 0.5 * mean(top floor(N*0.1) cells) per Google
    grid_total = grid_rows * grid_cols
    top_n_density = max(1, math.floor(grid_total * 0.1))
    flat_density = cell_density.reshape(K, -1)
    top_d, _ = flat_density.topk(top_n_density, dim=-1)
    density_K = 0.5 * top_d.mean(dim=-1)                            # [K]

    # ===== 3. Congestion =====
    if edges_pkg is not None and smooth_matrices is not None and routing_consts is not None:
        # Google's exact L/V routing reproduction.
        congestion_K, _ = gpu_congestion_google(
            pos_K, sizes_t,
            n_hard=n_hard if n_hard > 0 else int(macro_idx_p.shape[0]),
            edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
            routing_consts=routing_consts,
            canvas_w=canvas_w, canvas_h=canvas_h,
            grid_rows=grid_rows, grid_cols=grid_cols,
        )
    else:
        # Fallback: smooth bbox-demand top-10% mean (NOT Google's formula —
        # use only when edges_pkg / routing_consts are unavailable).
        cong_sigma = (cell_w + cell_h) * 0.5 * cong_smooth_sigma_frac
        grid_cx = (torch.arange(grid_cols, device=dev,
                                  dtype=pos_K.dtype) * cell_w + cell_w / 2)
        grid_cy = (torch.arange(grid_rows, device=dev,
                                  dtype=pos_K.dtype) * cell_h + cell_h / 2)
        in_x = (torch.sigmoid((x_max[..., None] - grid_cx[None, None, :])
                               / cong_sigma)
                * torch.sigmoid((grid_cx[None, None, :] - x_min[..., None])
                                 / cong_sigma))
        in_y = (torch.sigmoid((y_max[..., None] - grid_cy[None, None, :])
                               / cong_sigma)
                * torch.sigmoid((grid_cy[None, None, :] - y_min[..., None])
                                 / cong_sigma))
        cong_demand = torch.einsum("knr,knc->krc", in_y, in_x)
        flat_cong = cong_demand.reshape(K, -1)
        top_n_cong = max(1, math.floor(grid_total * cong_top_pct))
        top_c, _ = flat_cong.topk(top_n_cong, dim=-1)
        congestion_K = top_c.mean(dim=-1)

    # ===== 4. Combine via Google's weights: 1.0*wl + 0.5*density + 0.5*cong =====
    proxy_K = wl_K + 0.5 * density_K + 0.5 * congestion_K

    return proxy_K, {
        "wl": wl_K,
        "density": density_K,
        "congestion": congestion_K,
        "hpwl": hpwl_K,
    }


def gpu_proxy_for_snapshots(
    snapshots_pos: torch.Tensor,   # [n_snap, K, n_total, 2]
    sizes_t,
    macro_idx_p,
    offsets_p,
    mask_p,
    canvas_w: float,
    canvas_h: float,
    grid_rows: int,
    grid_cols: int,
    num_nets: int,
    chunk_n: int = 96,
) -> dict:
    """Run gpu_proxy_batched over each snapshot; returns [n_snap, K] arrays."""
    n_snap = snapshots_pos.shape[0]
    K = snapshots_pos.shape[1]
    proxy = torch.empty(n_snap, K, device=snapshots_pos.device,
                         dtype=snapshots_pos.dtype)
    wl = torch.empty_like(proxy)
    density = torch.empty_like(proxy)
    cong = torch.empty_like(proxy)
    for s in range(n_snap):
        p_K, comp = gpu_proxy_batched(
            snapshots_pos[s], sizes_t, macro_idx_p, offsets_p, mask_p,
            canvas_w, canvas_h, grid_rows, grid_cols, num_nets,
            chunk_n=chunk_n,
        )
        proxy[s] = p_K
        wl[s] = comp["wl"]
        density[s] = comp["density"]
        cong[s] = comp["congestion"]
    return {"proxy": proxy, "wl": wl, "density": density, "congestion": cong}
