"""Per-frame snapshot dumper for placement progress visualization.

Captures pos + metrics + density grid + congestion grid at user-defined
points in the pipeline (gradient subsampled snapshots, post-legalize,
post-CD round, post-pair-swap round, post-triple-cycle round). Saves all
data to a single compressed .npz file so a standalone renderer can build
the HTML visualization without re-running the pipeline.

Gated by env STRAPLE_BATCH_DUMP_SNAPSHOTS=1; output dir via
STRAPLE_BATCH_DUMP_DIR (default: results/snapshots/).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class SnapshotDumper:
    def __init__(self, benchmark, plc, bench_name: str,
                 output_dir: Optional[str] = None):
        self.benchmark = benchmark
        self.plc = plc
        self.bench_name = bench_name
        if output_dir is None:
            output_dir = os.environ.get(
                "STRAPLE_BATCH_DUMP_DIR", "results/snapshots")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_pos: List[np.ndarray] = []
        self.frames_label: List[str] = []
        self.frames_metrics: List[np.ndarray] = []
        self.frames_density_grid: List[np.ndarray] = []
        self.frames_cong_grid: List[np.ndarray] = []

    def add(self, pos_full, label: str) -> None:
        """Capture a frame snapshot with TILOS-evaluated metrics + grids."""
        from macro_place.objective import compute_proxy_cost

        if isinstance(pos_full, torch.Tensor):
            pos_np = pos_full.detach().cpu().numpy().astype(np.float32)
            pos_t = pos_full.detach().cpu().to(torch.float32)
        else:
            pos_np = np.asarray(pos_full, dtype=np.float32)
            pos_t = torch.tensor(pos_np, dtype=torch.float32)
        try:
            cost = compute_proxy_cost(pos_t, self.benchmark, self.plc)
        except Exception as exc:
            print(f"[SNAPSHOT_DUMP] compute_proxy_cost failed at "
                  f"label={label}: {exc}", flush=True)
            return

        proxy = float(cost["proxy_cost"])
        wl = float(cost["wirelength_cost"])
        dens = float(cost["density_cost"])
        cong = float(cost["congestion_cost"])
        ovl = int(cost.get("overlap_count", -1))

        nrow = int(self.benchmark.grid_rows)
        ncol = int(self.benchmark.grid_cols)
        density_grid = np.asarray(
            self.plc.grid_cells, dtype=np.float32
        ).reshape(nrow, ncol).copy()
        h_cong = np.asarray(
            self.plc.H_routing_cong, dtype=np.float32
        ).reshape(nrow, ncol)
        v_cong = np.asarray(
            self.plc.V_routing_cong, dtype=np.float32
        ).reshape(nrow, ncol)
        cong_grid = np.maximum(h_cong, v_cong).astype(np.float32).copy()

        self.frames_pos.append(pos_np)
        self.frames_label.append(str(label))
        self.frames_metrics.append(
            np.array([proxy, wl, dens, cong, float(ovl)], dtype=np.float32)
        )
        self.frames_density_grid.append(density_grid)
        self.frames_cong_grid.append(cong_grid)

        print(f"[SNAPSHOT_DUMP frame={len(self.frames_pos)} label={label} "
              f"proxy={proxy:.4f} wl={wl:.4f} dens={dens:.4f} "
              f"cong={cong:.4f} ovl={ovl}]", flush=True)

    def save(self) -> Path:
        """Write all collected frames + benchmark metadata to .npz."""
        out_path = self.output_dir / f"{self.bench_name}_dump.npz"
        bench = self.benchmark
        n_total = bench.num_macros
        n_hard = bench.num_hard_macros
        macro_sizes = bench.macro_sizes.cpu().numpy().astype(np.float32)
        macro_fixed = np.asarray(
            [bool(bench.macro_fixed[i]) for i in range(n_total)], dtype=bool
        )
        macro_names = np.array(
            [str(getattr(bench, "macro_names", [f"macro_{i}"])[i])
             if hasattr(bench, "macro_names") else f"macro_{i}"
             for i in range(n_total)],
            dtype=object,
        )

        try:
            from clustering import cluster_macros
            cluster_target = max(15, n_total // 30)
            cluster_ids, num_clusters, _ = cluster_macros(
                bench, method="louvain", seed=42,
                max_net_size=20, target_num_clusters=cluster_target,
            )
            cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
        except Exception as exc:
            print(f"[SNAPSHOT_DUMP] cluster_macros failed: {exc} "
                  f"-- saving without cluster_ids", flush=True)
            cluster_ids = -np.ones(n_total, dtype=np.int32)
            num_clusters = 0

        if not self.frames_pos:
            print(f"[SNAPSHOT_DUMP] no frames collected for {self.bench_name}",
                  flush=True)
            return out_path

        frames_pos_arr = np.stack(self.frames_pos, axis=0)
        frames_label_arr = np.array(self.frames_label, dtype=object)
        frames_metrics_arr = np.stack(self.frames_metrics, axis=0)
        frames_density_grid_arr = np.stack(self.frames_density_grid, axis=0)
        frames_cong_grid_arr = np.stack(self.frames_cong_grid, axis=0)

        np.savez_compressed(
            out_path,
            frames_pos=frames_pos_arr,
            frames_label=frames_label_arr,
            frames_metrics=frames_metrics_arr,
            frames_density_grid=frames_density_grid_arr,
            frames_cong_grid=frames_cong_grid_arr,
            macro_sizes=macro_sizes,
            macro_fixed=macro_fixed,
            macro_names=macro_names,
            cluster_ids=cluster_ids,
            num_clusters=np.int64(num_clusters),
            n_total=np.int64(n_total),
            n_hard=np.int64(n_hard),
            canvas_w=np.float32(bench.canvas_width),
            canvas_h=np.float32(bench.canvas_height),
            grid_rows=np.int64(bench.grid_rows),
            grid_cols=np.int64(bench.grid_cols),
            bench_name=np.array(self.bench_name, dtype=object),
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[SNAPSHOT_DUMP] saved {out_path} "
              f"({len(self.frames_pos)} frames, {size_mb:.1f} MB)",
              flush=True)
        return out_path


def dump_enabled() -> bool:
    return os.environ.get("STRAPLE_BATCH_DUMP_SNAPSHOTS", "0") == "1"
