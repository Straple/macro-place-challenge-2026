"""L-route diagnostic of routing congestion (matches TILOS attribution).

Per `gpu_proxy.gpu_congestion_google`, each 2-pin net deposits
  - H demand along the DRIVER ROW from col_min to col_max
  - V demand along the SINK  COL from row_min to row_max
This is L-shaped routing, NOT bbox-area RUDY. Long nets touch many
cells. Shortening long nets is the lever for cong.

This script
  1. computes per-net H + V cell footprints
  2. attributes the top-K hottest cells to nets that PHYSICALLY route
     through them (driver-row + sink-col membership), weighted by 1
  3. ranks contributors per cell and globally by total cell footprint
  4. ranks LONGEST nets (most cells touched) — direct lever target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submissions" / "straple"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=str)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--top-cells", type=int, default=20)
    parser.add_argument("--top-nets-per-cell", type=int, default=5)
    parser.add_argument("--top-long-nets", type=int, default=20)
    parser.add_argument("--bench-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import torch
    from macro_place.loader import load_benchmark_from_dir

    d = np.load(args.dump, allow_pickle=True)
    bench_name = str(d["bench_name"])
    if args.bench_dir is None:
        args.bench_dir = str(
            REPO / "external" / "MacroPlacement" / "Testcases"
            / "ICCAD04" / bench_name
        )
    benchmark, _plc = load_benchmark_from_dir(args.bench_dir)

    n_frames = int(d["frames_pos"].shape[0])
    frame = args.frame if args.frame >= 0 else n_frames + args.frame
    pos = d["frames_pos"][frame].astype(np.float64)
    cong_grid = d["frames_cong_grid"][frame].astype(np.float64)
    cluster_ids = (d["cluster_ids"] if "cluster_ids" in d.files
                   else -np.ones(int(d["n_total"]), dtype=np.int32))
    n_hard = int(d["n_hard"])
    n_total = int(d["n_total"])
    canvas_w = float(d["canvas_w"])
    canvas_h = float(d["canvas_h"])
    GR, GC = cong_grid.shape
    cell_w = canvas_w / GC
    cell_h = canvas_h / GR

    macro_pin_offsets = benchmark.macro_pin_offsets
    net_pin_nodes = benchmark.net_pin_nodes
    port_positions = (benchmark.port_positions.cpu().numpy()
                       if hasattr(benchmark, "port_positions") and
                       benchmark.port_positions is not None else None)

    print(f"[diag-lroute] computing L-route footprint for "
          f"{len(net_pin_nodes)} nets...", flush=True)

    nets_lroute = []
    for net_idx, npn in enumerate(net_pin_nodes):
        npn_np = npn.cpu().numpy() if hasattr(npn, "cpu") else np.asarray(npn)
        if npn_np.shape[0] < 2:
            continue
        owners = npn_np[:, 0]
        pin_idxs = npn_np[:, 1]
        pin_xys = np.zeros((len(owners), 2), dtype=np.float64)
        for k, (own, pi) in enumerate(zip(owners, pin_idxs)):
            own = int(own); pi = int(pi)
            if own < n_hard:
                offset = (macro_pin_offsets[own][pi].cpu().numpy()
                          if pi < len(macro_pin_offsets[own])
                          else np.zeros(2))
                pin_xys[k] = pos[own] + offset
            elif own < n_total:
                pin_xys[k] = pos[own]
            else:
                if port_positions is not None and own - n_total < len(port_positions):
                    pin_xys[k] = port_positions[own - n_total]
                else:
                    pin_xys[k] = (canvas_w * 0.5, canvas_h * 0.5)

        cols = np.clip((pin_xys[:, 0] / cell_w).astype(int), 0, GC - 1)
        rows = np.clip((pin_xys[:, 1] / cell_h).astype(int), 0, GR - 1)

        h_cells = set()
        v_cells = set()
        if len(owners) == 2:
            a_row, a_col = int(rows[0]), int(cols[0])
            b_row, b_col = int(rows[1]), int(cols[1])
            c_lo, c_hi = min(a_col, b_col), max(a_col, b_col)
            r_lo, r_hi = min(a_row, b_row), max(a_row, b_row)
            for c in range(c_lo, c_hi):
                h_cells.add((a_row, c))
            for r in range(r_lo, r_hi):
                v_cells.add((r, b_col))
        else:
            driver_row = int(rows[0])
            driver_col = int(cols[0])
            for k in range(1, len(owners)):
                s_row, s_col = int(rows[k]), int(cols[k])
                c_lo, c_hi = min(driver_col, s_col), max(driver_col, s_col)
                r_lo, r_hi = min(driver_row, s_row), max(driver_row, s_row)
                for c in range(c_lo, c_hi):
                    h_cells.add((driver_row, c))
                for r in range(r_lo, r_hi):
                    v_cells.add((r, s_col))

        all_cells = h_cells | v_cells
        if not all_cells:
            continue
        nets_lroute.append({
            "net_idx": net_idx,
            "n_pins": int(len(owners)),
            "owners": owners.astype(np.int64),
            "h_cells": h_cells,
            "v_cells": v_cells,
            "n_cells": len(all_cells),
        })

    cell_to_nets = {}
    for net in nets_lroute:
        for cell in net["h_cells"] | net["v_cells"]:
            cell_to_nets.setdefault(cell, []).append(net["net_idx"])

    flat = cong_grid.flatten()
    order = np.argsort(flat)[::-1][:args.top_cells]

    lines = []
    lines.append(f"# L-route congestion diagnosis: {Path(args.dump).name}")
    lines.append(f"# frame={frame}/{n_frames-1}  label={str(d['frames_label'][frame])!r}")
    m = d["frames_metrics"][frame]
    lines.append(f"# metrics: proxy={m[0]:.4f} WL={m[1]:.4f} "
                  f"dens={m[2]:.4f} cong={m[3]:.4f} ovl={int(m[4])}")
    lines.append(f"# grid: {GR}x{GC}  cell={cell_w:.3f}x{cell_h:.3f}μm")
    lines.append(f"# nets indexed: {len(nets_lroute)}/{len(net_pin_nodes)} "
                  f"(skipped: <2 pins or zero L-route)")
    lines.append(f"# cong grid: min={cong_grid.min():.3f} mean={cong_grid.mean():.3f} "
                  f"max={cong_grid.max():.3f} top1%={np.percentile(cong_grid, 99):.3f}")
    lines.append("")

    cluster_total = {}
    for rank, flat_idx in enumerate(order):
        r, c = divmod(int(flat_idx), GC)
        cong_val = float(cong_grid[r, c])
        contributing = cell_to_nets.get((r, c), [])
        contrib_pairs = []
        for net_idx in contributing:
            net = nets_lroute[net_idx if net_idx < len(nets_lroute) else 0]
            net = next((n for n in nets_lroute if n["net_idx"] == net_idx), None)
            if net is None:
                continue
            contrib_pairs.append((net_idx, net["n_cells"], net["n_pins"], net["owners"]))
        contrib_pairs.sort(key=lambda x: -x[1])

        lines.append(f"## rank {rank+1}: cell (r={r:2d}, c={c:2d})  cong={cong_val:.3f}")
        lines.append(f"  L-route nets through cell: {len(contrib_pairs)}")
        for net_idx, n_cells_val, n_pins, owners in contrib_pairs[:args.top_nets_per_cell]:
            owner_strs = []
            for own in owners[:6]:
                own = int(own)
                if 0 <= own < n_total:
                    cid = int(cluster_ids[own])
                    kind = "S" if own >= n_hard else "H"
                    owner_strs.append(f"{kind}#{own}(c{cid})")
                else:
                    owner_strs.append(f"P#{own}")
            if len(owners) > 6:
                owner_strs.append(f"+{len(owners)-6}")
            lines.append(f"    net#{net_idx:5d}  L_cells={n_cells_val:3d}  "
                          f"pins={n_pins}  owners=[{', '.join(owner_strs)}]")

        contrib_set = {p[0] for p in contrib_pairs}
        for net_idx in contrib_set:
            net = next((n for n in nets_lroute if n["net_idx"] == net_idx), None)
            if net is None:
                continue
            cluster_local = {}
            for own in net["owners"]:
                own = int(own)
                if 0 <= own < n_total:
                    cid = int(cluster_ids[own])
                    cluster_local[cid] = cluster_local.get(cid, 0) + 1
            if not cluster_local:
                continue
            dom = max(cluster_local.items(), key=lambda x: x[1])[0]
            cluster_total[dom] = cluster_total.get(dom, 0) + 1
        lines.append("")

    lines.append("# clusters by # of contributions to top hot cells (L-route):")
    for cid, total in sorted(cluster_total.items(), key=lambda x: -x[1])[:15]:
        n_in = int(np.sum(cluster_ids == cid))
        lines.append(f"  cluster {cid:3d}  contribs_in_top={total:5d}  size={n_in}")
    lines.append("")

    lines.append(f"# top {args.top_long_nets} LONGEST nets (most L-route cells):")
    sorted_long = sorted(nets_lroute, key=lambda x: -x["n_cells"])[:args.top_long_nets]
    for net in sorted_long:
        owner_strs = []
        for own in net["owners"][:6]:
            own = int(own)
            if 0 <= own < n_total:
                cid = int(cluster_ids[own])
                kind = "S" if own >= n_hard else "H"
                owner_strs.append(f"{kind}#{own}(c{cid})")
            else:
                owner_strs.append(f"P#{own}")
        if len(net["owners"]) > 6:
            owner_strs.append(f"+{len(net['owners'])-6}")
        cluster_local = {}
        for own in net["owners"]:
            own = int(own)
            if 0 <= own < n_total:
                cid = int(cluster_ids[own])
                cluster_local[cid] = cluster_local.get(cid, 0) + 1
        n_clusters_spanned = len(cluster_local)
        lines.append(f"  net#{net['net_idx']:5d}  L_cells={net['n_cells']:3d}  "
                      f"pins={net['n_pins']}  spans={n_clusters_spanned}clusters  "
                      f"owners=[{', '.join(owner_strs)}]")
    lines.append("")

    lines.append("# clusters by total L-route cells of nets they participate in:")
    cluster_lroute_mass = {}
    for net in nets_lroute:
        cluster_local = {}
        for own in net["owners"]:
            own = int(own)
            if 0 <= own < n_total:
                cid = int(cluster_ids[own])
                cluster_local[cid] = cluster_local.get(cid, 0) + 1
        if not cluster_local:
            continue
        dom = max(cluster_local.items(), key=lambda x: x[1])[0]
        cluster_lroute_mass[dom] = cluster_lroute_mass.get(dom, 0) + net["n_cells"]
    for cid, mass in sorted(cluster_lroute_mass.items(), key=lambda x: -x[1])[:15]:
        n_in = int(np.sum(cluster_ids == cid))
        lines.append(f"  cluster {cid:3d}  L_cells_total={mass:6d}  size={n_in}")

    out = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(out)
        print(f"[diag-lroute] wrote {args.output}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
