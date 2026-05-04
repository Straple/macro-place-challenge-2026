"""Placement progress recorder + 2x2 video renderer.

Activated via env var STRAPLE_VIS_VIDEO=<output_path.mp4>. Captures snapshots
during placement (initial / post-legalize / sampled LNS iters / post-refine)
and renders a 2x2 video:
    [placement]   [density heatmap]
    [congestion]  [score history]

The score panel plots WL / 0.5*density / 0.5*congestion / total proxy_cost
over snapshot index, with a vertical marker on the current frame.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class PlacementRecorder:
    def __init__(self, benchmark, plc, output_path: str, interval: int = 100,
                 max_frames: int = 60):
        self.benchmark = benchmark
        self.plc = plc
        self.output_path = output_path
        self.interval = max(1, interval)
        self.max_frames = max(4, max_frames)
        self.snapshots: List[Tuple[np.ndarray, str]] = []
        self._lns_counter = 0

    def add(self, hard_positions: np.ndarray, label: str):
        positions = np.ascontiguousarray(hard_positions, dtype=np.float64).copy()
        self.snapshots.append((positions, label))

    def maybe_add_lns(self, hard_positions: np.ndarray, lns_iter: int,
                      operator: str, accepted: bool, cost: float):
        self._lns_counter += 1
        if self._lns_counter == 1 or self._lns_counter % self.interval == 0:
            tag = "ACC" if accepted else "rej"
            label = f"LNS iter={lns_iter} op={operator} {tag} cost={cost:.4f}"
            self.add(hard_positions, label)

    def _subsample(self):
        if len(self.snapshots) <= self.max_frames:
            return self.snapshots
        n = len(self.snapshots)
        keep_first_last = 2
        budget = self.max_frames - keep_first_last
        middle = self.snapshots[1:-1]
        step = len(middle) / budget
        picked = [self.snapshots[0]]
        for i in range(budget):
            idx = int(round(i * step))
            picked.append(middle[min(idx, len(middle) - 1)])
        picked.append(self.snapshots[-1])
        print(f"[visualizer] subsampled {n} snapshots -> {len(picked)} frames",
              flush=True)
        return picked

    def render(self):
        if not self.snapshots:
            print("[visualizer] no snapshots, skipping", file=sys.stderr)
            return
        self.snapshots = self._subsample()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"[visualizer] matplotlib not available: {exc}", file=sys.stderr)
            return

        from macro_place.objective import compute_proxy_cost
        import torch

        breakdowns = []
        full_template = self.benchmark.macro_positions.clone()
        n_hard = self.benchmark.num_hard_macros

        print(f"[visualizer] computing proxy_cost for {len(self.snapshots)} snapshots...",
              flush=True)
        for idx, (pos, label) in enumerate(self.snapshots):
            full = full_template.clone()
            full[:pos.shape[0]] = torch.tensor(pos, dtype=torch.float32)
            costs = compute_proxy_cost(full, self.benchmark, self.plc)
            breakdowns.append(costs)
            if (idx + 1) % 10 == 0 or idx + 1 == len(self.snapshots):
                print(f"[visualizer]   snapshot {idx+1}/{len(self.snapshots)} "
                      f"proxy={costs['proxy_cost']:.4f} ovrlp={costs['overlap_count']}",
                      flush=True)

        history = {
            "wl": [b["wirelength_cost"] for b in breakdowns],
            "den": [b["density_cost"] for b in breakdowns],
            "cong": [b["congestion_cost"] for b in breakdowns],
            "proxy": [b["proxy_cost"] for b in breakdowns],
            "ovrlp": [b["overlap_count"] for b in breakdowns],
        }
        cost_max = max(max(history["wl"]), max(history["den"]), max(history["cong"]),
                       max(history["proxy"])) * 1.05

        out_path = Path(self.output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()

        if suffix in (".html", ".htm"):
            self._render_html(history, breakdowns, cost_max, out_path)
        else:
            self._render_video(history, breakdowns, cost_max, out_path, suffix)

    def _render_video(self, history, breakdowns, cost_max, out_path, suffix):
        import matplotlib.pyplot as plt
        from macro_place.objective import compute_proxy_cost
        import torch

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            print("[visualizer] ffmpeg not found, cannot create video; "
                  "use .html output instead", file=sys.stderr)
            return

        full_template = self.benchmark.macro_positions.clone()
        n_hard = self.benchmark.num_hard_macros

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for idx, (pos, label) in enumerate(self.snapshots):
                full = full_template.clone()
                full[:pos.shape[0]] = torch.tensor(pos, dtype=torch.float32)
                _ = compute_proxy_cost(full, self.benchmark, self.plc)

                fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                self._draw_placement(axes[0, 0], full, breakdowns[idx], label, idx)
                self._draw_density(axes[0, 1], full)
                self._draw_congestion(axes[1, 0], full)
                self._draw_score(axes[1, 1], history["wl"], history["den"],
                                 history["cong"], history["proxy"], history["ovrlp"],
                                 idx, cost_max)

                fig.suptitle(
                    f"{self.benchmark.name} · frame {idx+1}/{len(self.snapshots)} · "
                    f"proxy={history['proxy'][idx]:.4f}  "
                    f"overlaps={history['ovrlp'][idx]}",
                    fontsize=14, y=0.995,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                frame_path = tmp_dir / f"frame_{idx:05d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                if (idx + 1) % 10 == 0 or idx + 1 == len(self.snapshots):
                    print(f"[visualizer]   rendered frame {idx+1}/{len(self.snapshots)}",
                          flush=True)

            fps = int(os.environ.get("STRAPLE_VIS_FPS", "4"))
            if suffix == ".gif":
                cmd = [ffmpeg_bin, "-y", "-framerate", str(fps),
                       "-i", str(tmp_dir / "frame_%05d.png"), str(out_path)]
            else:
                cmd = [ffmpeg_bin, "-y", "-framerate", str(fps),
                       "-i", str(tmp_dir / "frame_%05d.png"),
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       str(out_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[visualizer] ffmpeg failed:\n{result.stderr[-1000:]}",
                      file=sys.stderr)
                return

        print(f"[visualizer] saved {out_path} "
              f"({len(self.snapshots)} frames @ {fps} fps)", flush=True)

    def _render_html(self, history, breakdowns, cost_max, out_path):
        import base64
        import io
        import json
        import matplotlib.pyplot as plt
        from macro_place.objective import compute_proxy_cost
        import torch

        full_template = self.benchmark.macro_positions.clone()
        bench = self.benchmark
        n_hard = bench.num_hard_macros

        macro_names = []
        for i in range(bench.num_macros):
            try:
                name = bench.macro_names[i]
            except (AttributeError, IndexError):
                name = f"macro_{i}"
            macro_names.append(name)

        sizes_list = bench.macro_sizes.tolist()
        fixed_list = [bool(bench.macro_fixed[i]) for i in range(bench.num_macros)]

        frames_data = []
        print(f"[visualizer] rendering {len(self.snapshots)} HTML frames...", flush=True)

        for idx, (pos, label) in enumerate(self.snapshots):
            full = full_template.clone()
            full[:pos.shape[0]] = torch.tensor(pos, dtype=torch.float32)
            _ = compute_proxy_cost(full, bench, self.plc)

            density_png = self._panel_to_base64(
                lambda ax: self._draw_density(ax, full), figsize=(7, 6))
            congestion_png = self._panel_to_base64(
                lambda ax: self._draw_congestion(ax, full), figsize=(7, 6))
            score_png = self._panel_to_base64(
                lambda ax: self._draw_score(ax, history["wl"], history["den"],
                                            history["cong"], history["proxy"],
                                            history["ovrlp"], idx, cost_max),
                figsize=(7, 6))

            macros = []
            full_list = full.tolist()
            for i in range(bench.num_macros):
                x, y = full_list[i]
                w, h = sizes_list[i]
                macros.append({
                    "i": i,
                    "name": macro_names[i],
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "w": round(w, 4),
                    "h": round(h, 4),
                    "soft": i >= n_hard,
                    "fixed": fixed_list[i],
                })

            frames_data.append({
                "label": str(label),
                "proxy": float(history["proxy"][idx]),
                "wl": float(history["wl"][idx]),
                "den": float(history["den"][idx]),
                "cong": float(history["cong"][idx]),
                "ovrlp": int(history["ovrlp"][idx]),
                "macros": macros,
                "density_png": density_png,
                "congestion_png": congestion_png,
                "score_png": score_png,
            })
            if (idx + 1) % 10 == 0 or idx + 1 == len(self.snapshots):
                print(f"[visualizer]   rendered HTML frame {idx+1}/{len(self.snapshots)}",
                      flush=True)

        html = self._build_html(frames_data, bench)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[visualizer] saved {out_path} "
              f"({len(self.snapshots)} frames, {size_mb:.1f} MB)", flush=True)

    def _panel_to_base64(self, draw_fn, figsize=(7, 6)):
        import base64
        import io
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        draw_fn(ax)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_html(self, frames_data, bench):
        import json
        bench_name = bench.name
        canvas_w = bench.canvas_width
        canvas_h = bench.canvas_height
        frames_json = json.dumps(frames_data)
        fps = int(os.environ.get("STRAPLE_VIS_FPS", "20"))

        return _HTML_TEMPLATE.format(
            bench_name=bench_name,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            num_hard=bench.num_hard_macros,
            num_total=bench.num_macros,
            num_soft=bench.num_macros - bench.num_hard_macros,
            fps=fps,
            frames_json=frames_json,
        )

    def _draw_placement(self, ax, full_placement, breakdown, label, idx):
        from matplotlib.patches import Rectangle

        bench = self.benchmark
        ax.add_patch(Rectangle((0, 0), bench.canvas_width, bench.canvas_height,
                               fill=False, edgecolor="black", linewidth=2))
        n_hard = bench.num_hard_macros
        for i in range(bench.num_macros):
            x, y = full_placement[i].tolist()
            w, h = bench.macro_sizes[i].tolist()
            is_soft = i >= n_hard
            if bench.macro_fixed[i]:
                color, alpha = "red", 0.5
            elif is_soft:
                color, alpha = "lightsteelblue", 0.25
            else:
                color, alpha = "blue", 0.5
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                                   fill=True, facecolor=color, alpha=alpha,
                                   edgecolor="black", linewidth=0.4,
                                   linestyle="dashed" if is_soft else "solid"))
        ax.set_xlim(0, bench.canvas_width)
        ax.set_ylim(0, bench.canvas_height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")
        ovrlp = breakdown["overlap_count"]
        title = f"Placement [{label}]"
        if ovrlp > 0:
            title += f"  ⚠ {ovrlp} overlaps"
        ax.set_title(title, fontsize=10)

    def _draw_density(self, ax, full_placement):
        bench = self.benchmark
        plc = self.plc
        n_hard = bench.num_hard_macros
        ax.add_patch(_rect((0, 0), bench.canvas_width, bench.canvas_height,
                           edge="black", lw=2))
        nrow, ncol = bench.grid_rows, bench.grid_cols
        dens = np.asarray(plc.grid_cells, dtype=float).reshape(nrow, ncol)
        vmax = max(float(np.max(dens)), 1e-9)
        ax.imshow(dens, origin="lower",
                  extent=(0, bench.canvas_width, 0, bench.canvas_height),
                  aspect="equal", cmap="Blues", alpha=0.7,
                  vmin=0.0, vmax=vmax, interpolation="nearest", zorder=0)
        for i in range(n_hard):
            x, y = full_placement[i].tolist()
            w, h = bench.macro_sizes[i].tolist()
            ax.add_patch(_rect((x - w / 2, y - h / 2), w, h,
                               edge="black", lw=0.5, zorder=3))
        ax.set_xlim(0, bench.canvas_width)
        ax.set_ylim(0, bench.canvas_height)
        ax.set_aspect("equal")
        ax.set_title("Density (grid cells)", fontsize=11)

    def _draw_congestion(self, ax, full_placement):
        bench = self.benchmark
        plc = self.plc
        n_hard = bench.num_hard_macros
        ax.add_patch(_rect((0, 0), bench.canvas_width, bench.canvas_height,
                           edge="black", lw=2))
        nrow, ncol = bench.grid_rows, bench.grid_cols
        h_cong = np.asarray(plc.H_routing_cong, dtype=float).reshape(nrow, ncol)
        v_cong = np.asarray(plc.V_routing_cong, dtype=float).reshape(nrow, ncol)
        cong = np.maximum(h_cong, v_cong)
        pos_vals = cong[cong > 0]
        vmax = float(np.percentile(pos_vals, 99)) if pos_vals.size else 1.0
        vmax = max(vmax, 1e-9)
        ax.imshow(cong, origin="lower",
                  extent=(0, bench.canvas_width, 0, bench.canvas_height),
                  aspect="equal", cmap="hot", alpha=0.7,
                  vmin=0.0, vmax=vmax, interpolation="nearest", zorder=0)
        for i in range(n_hard):
            x, y = full_placement[i].tolist()
            w, h = bench.macro_sizes[i].tolist()
            ax.add_patch(_rect((x - w / 2, y - h / 2), w, h,
                               edge="black", lw=0.5, zorder=3))
        ax.set_xlim(0, bench.canvas_width)
        ax.set_ylim(0, bench.canvas_height)
        ax.set_aspect("equal")
        ax.set_title("Congestion (max H/V routing)", fontsize=11)

    def _draw_score(self, ax, wl_h, den_h, cong_h, proxy_h, ovrlp_h, current_idx,
                    cost_max):
        xs = np.arange(len(wl_h))
        ax.plot(xs, proxy_h, color="black", lw=2.2, label="proxy_cost", zorder=5)
        ax.plot(xs, wl_h, color="green", lw=1.4, label="wirelength")
        ax.plot(xs, den_h, color="red", lw=1.4, label="density")
        ax.plot(xs, cong_h, color="blue", lw=1.4, label="congestion")
        ax.axvline(current_idx, color="orange", lw=2, alpha=0.7,
                   label=f"frame {current_idx}")
        ax.scatter([current_idx], [proxy_h[current_idx]], color="orange",
                   s=100, zorder=6, edgecolors="black", linewidths=1)

        if any(o > 0 for o in ovrlp_h):
            ax2 = ax.twinx()
            ax2.plot(xs, ovrlp_h, color="purple", lw=0.8, alpha=0.5,
                     label="overlaps")
            ax2.set_ylabel("overlaps", color="purple")
            ax2.tick_params(axis="y", labelcolor="purple")

        ax.set_xlabel("snapshot index")
        ax.set_ylabel("cost")
        ax.set_ylim(0, cost_max)
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Score history · proxy={proxy_h[current_idx]:.4f} "
            f"(WL={wl_h[current_idx]:.3f}, D={den_h[current_idx]:.3f}, "
            f"C={cong_h[current_idx]:.3f})",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)


def _rect(xy, w, h, edge="black", lw=1.0, zorder=2):
    from matplotlib.patches import Rectangle
    return Rectangle(xy, w, h, fill=False, edgecolor=edge,
                     linewidth=lw, zorder=zorder)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Placement progress · {bench_name}</title>
<style>
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1e1e1e;
    color: #e0e0e0;
  }}
  header {{
    padding: 12px 18px;
    background: #2a2a2a;
    border-bottom: 1px solid #404040;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }}
  h1 {{ font-size: 16px; margin: 0; font-weight: 500; }}
  .controls {{ display: flex; align-items: center; gap: 8px; }}
  button {{
    background: #3a3a3a;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }}
  button:hover {{ background: #4a4a4a; }}
  button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  input[type="range"] {{ width: 280px; }}
  .frame-info {{
    font-family: ui-monospace, monospace;
    font-size: 12px;
    color: #b0b0b0;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 8px;
    padding: 8px;
    height: calc(100vh - 130px);
  }}
  .panel {{
    background: #2a2a2a;
    border: 1px solid #404040;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .panel-title {{
    position: absolute;
    top: 6px;
    left: 8px;
    font-size: 11px;
    color: #888;
    background: rgba(30, 30, 30, 0.8);
    padding: 2px 6px;
    border-radius: 3px;
    z-index: 5;
  }}
  .panel img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  #placement-canvas {{
    width: 100%;
    height: 100%;
    cursor: crosshair;
  }}
  #tooltip {{
    position: absolute;
    background: rgba(20, 20, 20, 0.95);
    color: #fff;
    padding: 6px 10px;
    border-radius: 4px;
    font-family: ui-monospace, monospace;
    font-size: 12px;
    pointer-events: none;
    border: 1px solid #555;
    z-index: 10;
    display: none;
    white-space: pre;
  }}
  .ovrlp-warning {{ color: #ff6b6b; }}
  .help {{ font-size: 11px; color: #777; }}
</style>
</head>
<body>
<header>
  <h1>{bench_name} · {num_hard} hard + {num_soft} soft macros</h1>
  <div class="controls">
    <button id="prev">◀</button>
    <button id="play">▶ play</button>
    <button id="next">▶</button>
    <input type="range" id="slider" min="0" max="0" value="0" />
    <span class="frame-info" id="frame-info">frame 0/0</span>
  </div>
  <span class="help">← → keys • space = play/pause • hover macros for info</span>
</header>

<div class="grid">
  <div class="panel">
    <div class="panel-title" id="placement-title">Placement</div>
    <canvas id="placement-canvas"></canvas>
    <div id="tooltip"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Density</div>
    <img id="density-img" />
  </div>
  <div class="panel">
    <div class="panel-title">Congestion</div>
    <img id="congestion-img" />
  </div>
  <div class="panel">
    <div class="panel-title">Score history</div>
    <img id="score-img" />
  </div>
</div>

<script>
const FRAMES = {frames_json};
const CANVAS_W = {canvas_w};
const CANVAS_H = {canvas_h};
const NUM_HARD = {num_hard};
const NUM_TOTAL = {num_total};

let cur = 0;
let playing = false;
let lastT = 0;
const FPS = {fps};

const canvas = document.getElementById("placement-canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const slider = document.getElementById("slider");
const frameInfo = document.getElementById("frame-info");
const placementTitle = document.getElementById("placement-title");
slider.max = FRAMES.length - 1;

function colorFor(macro) {{
  if (macro.fixed) return "rgba(220, 60, 60, 0.55)";
  if (macro.soft)  return "rgba(176, 196, 222, 0.30)";
  return "rgba(50, 100, 200, 0.55)";
}}

function fitCanvas() {{
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * window.devicePixelRatio;
  canvas.height = r.height * window.devicePixelRatio;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  return r;
}}

function drawPlacement() {{
  const r = fitCanvas();
  const padding = 24;
  const aspectData = CANVAS_W / CANVAS_H;
  const aspectView = (r.width - 2 * padding) / (r.height - 2 * padding);
  let plotW, plotH;
  if (aspectView > aspectData) {{
    plotH = r.height - 2 * padding;
    plotW = plotH * aspectData;
  }} else {{
    plotW = r.width - 2 * padding;
    plotH = plotW / aspectData;
  }}
  const px = (r.width - plotW) / 2;
  const py = (r.height - plotH) / 2;

  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, r.width, r.height);
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(px, py, plotW, plotH);

  const sx = plotW / CANVAS_W;
  const sy = plotH / CANVAS_H;

  const f = FRAMES[cur];
  for (const m of f.macros) {{
    const left = px + m.x * sx - (m.w * sx) / 2;
    const top = py + plotH - (m.y * sy + (m.h * sy) / 2);
    const ww = m.w * sx;
    const hh = m.h * sy;
    ctx.fillStyle = colorFor(m);
    ctx.fillRect(left, top, ww, hh);
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.lineWidth = 0.4;
    ctx.strokeRect(left, top, ww, hh);
  }}

  canvas._plot = {{ px, py, plotW, plotH, sx, sy }};

  const ovrlp = f.ovrlp;
  let title = "Placement [" + f.label + "]";
  if (ovrlp > 0) title += "  ⚠ " + ovrlp + " overlaps";
  placementTitle.textContent = title;
  if (ovrlp > 0) placementTitle.classList.add("ovrlp-warning");
  else placementTitle.classList.remove("ovrlp-warning");
}}

function showFrame(i) {{
  cur = Math.max(0, Math.min(FRAMES.length - 1, i));
  slider.value = cur;
  const f = FRAMES[cur];
  document.getElementById("density-img").src = "data:image/png;base64," + f.density_png;
  document.getElementById("congestion-img").src = "data:image/png;base64," + f.congestion_png;
  document.getElementById("score-img").src = "data:image/png;base64," + f.score_png;
  drawPlacement();
  frameInfo.textContent =
    `frame ${{cur+1}}/${{FRAMES.length}} · proxy=${{f.proxy.toFixed(4)}} ` +
    `WL=${{f.wl.toFixed(3)}} D=${{f.den.toFixed(3)}} C=${{f.cong.toFixed(3)}} ` +
    `ovrlp=${{f.ovrlp}}`;
}}

document.getElementById("prev").onclick = () => showFrame(cur - 1);
document.getElementById("next").onclick = () => showFrame(cur + 1);
slider.oninput = (e) => showFrame(parseInt(e.target.value));

function tick(t) {{
  if (!playing) return;
  if (!lastT) lastT = t;
  const dt = t - lastT;
  if (dt >= 1000 / FPS) {{
    lastT = t;
    if (cur >= FRAMES.length - 1) {{
      playing = false;
      document.getElementById("play").textContent = "▶ play";
      return;
    }}
    showFrame(cur + 1);
  }}
  requestAnimationFrame(tick);
}}
function togglePlay() {{
  playing = !playing;
  document.getElementById("play").textContent = playing ? "❚❚ pause" : "▶ play";
  if (playing) {{
    if (cur >= FRAMES.length - 1) cur = 0;
    lastT = 0;
    requestAnimationFrame(tick);
  }}
}}
document.getElementById("play").onclick = togglePlay;

document.addEventListener("keydown", (e) => {{
  if (e.key === "ArrowLeft") {{ showFrame(cur - 1); e.preventDefault(); }}
  else if (e.key === "ArrowRight") {{ showFrame(cur + 1); e.preventDefault(); }}
  else if (e.key === " ") {{ togglePlay(); e.preventDefault(); }}
}});

canvas.addEventListener("mousemove", (e) => {{
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;
  const p = canvas._plot;
  if (!p) return;
  const dataX = (mx - p.px) / p.sx;
  const dataY = (p.plotH - (my - p.py)) / p.sy;
  if (dataX < 0 || dataX > CANVAS_W || dataY < 0 || dataY > CANVAS_H) {{
    tooltip.style.display = "none";
    return;
  }}
  const f = FRAMES[cur];
  let hit = null;
  let bestArea = Infinity;
  for (const m of f.macros) {{
    if (Math.abs(dataX - m.x) <= m.w / 2 && Math.abs(dataY - m.y) <= m.h / 2) {{
      const area = m.w * m.h;
      if (area < bestArea) {{ hit = m; bestArea = area; }}
    }}
  }}
  if (hit) {{
    const kind = hit.fixed ? "FIXED" : (hit.soft ? "soft" : "hard");
    tooltip.textContent =
      `[${{hit.i}}] ${{hit.name}}\\n` +
      `kind: ${{kind}}\\n` +
      `pos:  (${{hit.x.toFixed(3)}}, ${{hit.y.toFixed(3)}})\\n` +
      `size: ${{hit.w.toFixed(3)}} × ${{hit.h.toFixed(3)}}`;
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX - r.left + 14) + "px";
    tooltip.style.top = (e.clientY - r.top + 14) + "px";
  }} else {{
    tooltip.style.display = "none";
  }}
}});
canvas.addEventListener("mouseleave", () => {{ tooltip.style.display = "none"; }});

window.addEventListener("resize", () => drawPlacement());
showFrame(0);
</script>
</body>
</html>
"""
