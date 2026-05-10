"""Diagnose congestion in a snapshot dump.

For the chosen frame, lists the top hot cells with: which nets contribute
the most (by RUDY = 1/bbox_area weighting), which macros sit in or near
the cell, and the cluster ranking aggregated over all top-K cells. Writes
a plain-text report so the user can compare configs.

Runs on the GPU server (requires `torch` to load the benchmark).

Usage:
  uv run python scripts/diagnose_congestion.py results/snapshots/ibm01_dump.npz
  uv run python scripts/diagnose_congestion.py <dump>.npz \\
      --frame -1 --top-cells 20 --top-nets-per-cell 5
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
    parser.add_argument("--frame", type=int, default=-1,
                        help="frame index (default: -1 = last)")
    parser.add_argument("--top-cells", type=int, default=20)
    parser.add_argument("--top-nets-per-cell", type=int, default=5)
    parser.add_argument("--bench-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="output txt file path")
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
    pos = d["frames_pos"][frame].astype(np.float64)            # [n_total, 2]
    cong_grid = d["frames_cong_grid"][frame].astype(np.float64)  # [GR, GC]
    cluster_ids = (d["cluster_ids"] if "cluster_ids" in d.files
                   else -np.ones(int(d["n_total"]), dtype=np.int32))
    n_hard = int(d["n_hard"])
    n_total = int(d["n_total"])
    canvas_w = float(d["canvas_w"])
    canvas_h = float(d["canvas_h"])
    GR, GC = cong_grid.shape
    cell_w = canvas_w / GC
    cell_h = canvas_h / GR
    label = str(d["frames_label"][frame])
    metrics = d["frames_metrics"][frame]

    macro_pin_offsets = benchmark.macro_pin_offsets
    net_pin_nodes = benchmark.net_pin_nodes
    port_positions = (benchmark.port_positions.cpu().numpy()
                       if hasattr(benchmark, "port_positions") and
                       benchmark.port_positions is not None else None)

    # Pre-compute per-net pin xy + bbox in grid units.
    print(f"[diag] computing {len(net_pin_nodes)} net bboxes...", flush=True)
    net_bboxes = []   # list of (rmin, cmin, rmax, cmax, n_cells, n_pins, owners_arr)
    for net_idx, npn in enumerate(net_pin_nodes):
        npn_np = npn.cpu().numpy() if hasattr(npn, "cpu") else np.asarray(npn)
        if npn_np.shape[0] == 0:
            continue
        owners = npn_np[:, 0]
        pin_idxs = npn_np[:, 1]
        pin_xys = np.zeros((len(owners), 2), dtype=np.float64)
        for k, (own, pi) in enumerate(zip(owners, pin_idxs)):
            own = int(own)
            pi = int(pi)
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
        bbox_min = pin_xys.min(axis=0)
        bbox_max = pin_xys.max(axis=0)
        c_min = int(np.clip(bbox_min[0] / cell_w, 0, GC - 1))
        c_max = int(np.clip(bbox_max[0] / cell_w, 0, GC - 1))
        r_min = int(np.clip(bbox_min[1] / cell_h, 0, GR - 1))
        r_max = int(np.clip(bbox_max[1] / cell_h, 0, GR - 1))
        n_cells = max(1, (r_max - r_min + 1) * (c_max - c_min + 1))
        net_bboxes.append((net_idx, r_min, c_min, r_max, c_max, n_cells,
                           int(len(owners)), owners.astype(np.int64)))
    print(f"[diag] {len(net_bboxes)} nets with bbox", flush=True)

    flat = cong_grid.flatten()
    order = np.argsort(flat)[::-1][:args.top_cells]

    lines = []
    lines.append(f"# Congestion diagnosis: {Path(args.dump).name}")
    lines.append(f"# frame={frame}/{n_frames-1}  label={label!r}")
    lines.append(f"# metrics: proxy={metrics[0]:.4f} WL={metrics[1]:.4f} "
                  f"dens={metrics[2]:.4f} cong={metrics[3]:.4f} "
                  f"ovl={int(metrics[4])}")
    lines.append(f"# grid: {GR}x{GC} cells, cell_size={cell_w:.3f}x{cell_h:.3f}μm")
    lines.append(f"# canvas: {canvas_w:.2f}x{canvas_h:.2f}μm")
    lines.append(f"# n_macros: {n_hard} hard + {n_total - n_hard} soft")
    lines.append(f"# n_nets: {len(net_bboxes)}")
    lines.append(f"# congestion grid stats: min={cong_grid.min():.3f} "
                  f"mean={cong_grid.mean():.3f} max={cong_grid.max():.3f} "
                  f"top1%={np.percentile(cong_grid, 99):.3f}")
    lines.append("")
    lines.append(f"# top {args.top_cells} hot cells, top "
                  f"{args.top_nets_per_cell} contributing nets each:")
    lines.append("")

    cluster_total_contrib = {}
    cluster_pair_pen = {}

    for rank, flat_idx in enumerate(order):
        r, c = divmod(int(flat_idx), GC)
        cong_val = float(cong_grid[r, c])
        cell_x_lo = c * cell_w
        cell_x_hi = (c + 1) * cell_w
        cell_y_lo = r * cell_h
        cell_y_hi = (r + 1) * cell_h

        contributing = []
        for entry in net_bboxes:
            net_idx, r_min, c_min, r_max, c_max, n_cells, n_pins, owners = entry
            if r_min <= r <= r_max and c_min <= c <= c_max:
                w = 1.0 / n_cells
                contributing.append((net_idx, w, n_pins, owners,
                                       (r_max - r_min + 1) *
                                       (c_max - c_min + 1)))
        contributing.sort(key=lambda x: -x[1])
        total_rudy = sum(c[1] for c in contributing)

        macros_in_cell = []
        radius_x = cell_w
        radius_y = cell_h
        for i in range(n_total):
            mx, my = pos[i]
            if (cell_x_lo - radius_x <= mx <= cell_x_hi + radius_x and
                    cell_y_lo - radius_y <= my <= cell_y_hi + radius_y):
                kind = "soft" if i >= n_hard else "hard"
                macros_in_cell.append((i, kind, int(cluster_ids[i])))

        cluster_count = {}
        for _, _, cid in macros_in_cell:
            if cid >= 0:
                cluster_count[cid] = cluster_count.get(cid, 0) + 1
        top_clusters = sorted(cluster_count.items(), key=lambda x: -x[1])[:3]

        lines.append(f"## rank {rank+1}: cell (r={r:2d}, c={c:2d})  "
                      f"cong={cong_val:.3f}")
        lines.append(f"  position: x=[{cell_x_lo:.2f},{cell_x_hi:.2f}]μm, "
                      f"y=[{cell_y_lo:.2f},{cell_y_hi:.2f}]μm")
        lines.append(f"  total nets through cell: {len(contributing)}, "
                      f"total RUDY={total_rudy:.3f}")
        for net_idx, w, n_pins, owners, area in contributing[:args.top_nets_per_cell]:
            owners_present = []
            for own in owners[:6]:
                own = int(own)
                if 0 <= own < n_total:
                    cid = int(cluster_ids[own])
                    kind = "S" if own >= n_hard else "H"
                    owners_present.append(f"{kind}#{own}(c{cid})")
                else:
                    owners_present.append(f"P#{own}")
            if len(owners) > 6:
                owners_present.append(f"+{len(owners)-6}")
            lines.append(f"    net#{net_idx:5d}  RUDY={w:.4f}  pins={n_pins}  "
                          f"bbox_cells={area:3d}  owners=[{', '.join(owners_present)}]")
        lines.append(f"  macros in cell±1: {len(macros_in_cell)} "
                      f"(hard={sum(1 for _,k,_ in macros_in_cell if k=='hard')}, "
                      f"soft={sum(1 for _,k,_ in macros_in_cell if k=='soft')})")
        lines.append(f"  top clusters here: " +
                      ", ".join(f"c{cid}({n})" for cid, n in top_clusters))
        lines.append("")

        for net_idx, w, n_pins, owners, area in contributing:
            cluster_local = {}
            for own in owners:
                own = int(own)
                if 0 <= own < n_total:
                    cid = int(cluster_ids[own])
                    cluster_local[cid] = cluster_local.get(cid, 0) + 1
            if not cluster_local:
                continue
            dom_cid = max(cluster_local.items(), key=lambda x: x[1])[0]
            cluster_total_contrib[dom_cid] = (
                cluster_total_contrib.get(dom_cid, 0) + w)

    lines.append("# clusters ranked by total RUDY contribution to top hot cells:")
    sorted_clusters = sorted(cluster_total_contrib.items(),
                              key=lambda x: -x[1])
    for cid, total in sorted_clusters[:15]:
        n_in = int(np.sum(cluster_ids == cid))
        lines.append(f"  cluster {cid:3d}  total_RUDY={total:.4f}  "
                      f"size={n_in} macros")

    out = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(out)
        print(f"[diag] wrote {args.output}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
