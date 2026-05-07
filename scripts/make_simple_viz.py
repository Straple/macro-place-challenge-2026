"""Simple JS-only HTML viewer: every step animation, no matplotlib.

4-pane layout: placement / density heatmap / overlap heatmap / score history.
Default 2x2 grid, click any pane to expand fullscreen, ↑/↓ to cycle panes,
←/→ to navigate frames.

Usage:
    Called from gpu_run_one.py after gradient_batch + legalize.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Placement evolution · {bench_name}</title>
<style>
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #1a1a1a;
  color: #e0e0e0;
}}
header {{
  padding: 10px 18px;
  background: #2a2a2a;
  border-bottom: 1px solid #404040;
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}}
h1 {{ font-size: 14px; margin: 0; font-weight: 500; }}
button {{
  background: #3a3a3a;
  color: #e0e0e0;
  border: 1px solid #555;
  padding: 4px 9px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}}
button:hover {{ background: #4a4a4a; }}
input[type="range"] {{ width: 320px; }}
.frame-info {{
  font-family: ui-monospace, monospace;
  font-size: 12px;
  color: #b0b0b0;
}}
.help {{ font-size: 11px; color: #777; }}
#wrap {{
  padding: 10px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 10px;
  height: calc(100vh - 60px);
  box-sizing: border-box;
}}
.pane {{
  background: #1a1a1a;
  border: 1px solid #444;
  position: relative;
  overflow: hidden;
  cursor: zoom-in;
  display: flex;
  flex-direction: column;
}}
.pane.active {{
  border-color: #6c9;
}}
.pane.fullscreen {{
  grid-column: 1 / -1;
  grid-row: 1 / -1;
  cursor: zoom-out;
}}
.pane.hidden {{ display: none; }}
.pane-title {{
  position: absolute;
  top: 6px;
  left: 8px;
  font-size: 11px;
  color: #aaa;
  background: rgba(0,0,0,0.55);
  padding: 2px 6px;
  border-radius: 3px;
  z-index: 2;
  pointer-events: none;
  font-family: ui-monospace, monospace;
}}
canvas {{ display: block; flex: 1; width: 100%; height: 100%; }}
#tooltip {{
  position: absolute;
  background: rgba(20,20,20,0.95);
  color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  font-family: ui-monospace, monospace;
  font-size: 12px;
  pointer-events: none;
  border: 1px solid #555;
  display: none;
  white-space: pre;
  z-index: 100;
}}
</style>
</head>
<body>
<header>
  <h1>{bench_name} · {num_hard} hard + {num_soft} soft · K={num_clusters} clusters · best={best_proxy:.4f}</h1>
  <button id="prev">◀</button>
  <button id="play">▶ play</button>
  <button id="next">▶</button>
  <input type="range" id="slider" min="0" max="0" value="0" />
  <span class="frame-info" id="info">frame 0/0</span>
  <button id="speed">1×</button>
  <span class="help">← → frames • space play • ↑ ↓ pane • c color • s speed</span>
</header>
<div id="wrap">
  <div class="pane" data-pane="0"><div class="pane-title">placement</div><canvas id="cv0"></canvas></div>
  <div class="pane" data-pane="1"><div class="pane-title">density (macro footprint)</div><canvas id="cv1"></canvas></div>
  <div class="pane" data-pane="2"><div class="pane-title">overlap area heatmap</div><canvas id="cv2"></canvas></div>
  <div class="pane" data-pane="3"><div class="pane-title">score history</div><canvas id="cv3"></canvas></div>
</div>
<div id="tooltip"></div>
<script>
const SIZES = {sizes_json};
const COLORS = {colors_json};
const FIXED = {fixed_json};
const N_HARD = {num_hard};
const FRAMES = {frames_json};
const LABELS = {labels_json};
const PROXIES = {proxies_json};
const WL_COSTS = {wl_json};
const DEN_COSTS = {den_json};
const CONG_COSTS = {cong_json};
const OV_COUNTS = {ovc_json};
const OV_AREAS = {ova_json};
const STEPS = {steps_json};
const CANVAS_W = {canvas_w};
const CANVAS_H = {canvas_h};
const NUM_CLUSTERS = {num_clusters};
const GRID_ROWS = 48;
const GRID_COLS = Math.max(1, Math.round(GRID_ROWS * CANVAS_W / CANVAS_H));

const panes = Array.from(document.querySelectorAll(".pane"));
const canvases = panes.map(p => p.querySelector("canvas"));
const ctxs = canvases.map(c => c.getContext("2d"));
const slider = document.getElementById("slider");
const info = document.getElementById("info");
const speedBtn = document.getElementById("speed");
const tooltip = document.getElementById("tooltip");
slider.max = FRAMES.length - 1;

let cur = 0;
let playing = false;
let lastT = 0;
let speed = 1;
let colorMode = "cluster";
let activePane = -1; // -1 = grid, 0..3 = fullscreen
const SPEEDS = [1, 2, 4, 8, 0.5];
const FPS_BASE = 30;

function fitCanvas(idx) {{
  const c = canvases[idx];
  const rect = c.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.max(50, Math.floor(rect.width * dpr));
  c.height = Math.max(50, Math.floor(rect.height * dpr));
  c.style.width = rect.width + "px";
  c.style.height = rect.height + "px";
  const ctx = ctxs[idx];
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  return [rect.width, rect.height];
}}

function drawPlacement(idx) {{
  const [W, H] = fitCanvas(idx);
  const ctx = ctxs[idx];
  const sx = W / CANVAS_W;
  const sy = H / CANVAS_H;
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, W, H);
  const pos = FRAMES[cur];
  const ord = SIZES.map((s, i) => [s[0]*s[1], i]).sort((a,b) => b[0]-a[0]).map(x => x[1]);
  for (const i of ord) {{
    const x = pos[i*2], y = pos[i*2+1];
    const w = SIZES[i][0], h = SIZES[i][1];
    const c = COLORS[i];
    let color;
    if (FIXED[i]) color = "rgba(220,60,60,0.55)";
    else if (i >= N_HARD) {{
      color = colorMode === "cluster" && NUM_CLUSTERS > 0
        ? `rgba(${{c[0]}},${{c[1]}},${{c[2]}},0.30)` : "rgba(176,196,222,0.30)";
    }} else {{
      color = colorMode === "cluster" && NUM_CLUSTERS > 0
        ? `rgba(${{c[0]}},${{c[1]}},${{c[2]}},0.85)` : "rgba(50,100,200,0.65)";
    }}
    ctx.fillStyle = color;
    const px = x * sx - (w * sx) / 2;
    const py = H - (y * sy + (h * sy) / 2);
    const pw = w * sx, ph = h * sy;
    ctx.fillRect(px, py, pw, ph);
    if (i < N_HARD) {{
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.lineWidth = 0.4;
      ctx.strokeRect(px, py, pw, ph);
    }}
  }}
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(0, 0, W, H);
}}

function rasterizeFootprint(grid) {{
  for (let i = 0; i < grid.length; i++) grid[i] = 0;
  const cw = CANVAS_W / GRID_COLS;
  const ch = CANVAS_H / GRID_ROWS;
  const pos = FRAMES[cur];
  for (let i = 0; i < SIZES.length; i++) {{
    if (FIXED[i]) continue;
    const x = pos[i*2], y = pos[i*2+1];
    const w = SIZES[i][0], h = SIZES[i][1];
    const xl = x - w*0.5, xh = x + w*0.5;
    const yl = y - h*0.5, yh = y + h*0.5;
    const cl = Math.max(0, Math.floor(xl / cw));
    const ch_ = Math.min(GRID_COLS - 1, Math.floor(xh / cw));
    const rl = Math.max(0, Math.floor(yl / ch));
    const rh = Math.min(GRID_ROWS - 1, Math.floor(yh / ch));
    for (let r = rl; r <= rh; r++) {{
      const cyl = r * ch, cyh = cyl + ch;
      const ovh = Math.max(0, Math.min(yh, cyh) - Math.max(yl, cyl));
      if (ovh <= 0) continue;
      for (let c = cl; c <= ch_; c++) {{
        const cxl = c * cw, cxh = cxl + cw;
        const ovw = Math.max(0, Math.min(xh, cxh) - Math.max(xl, cxl));
        if (ovw <= 0) continue;
        grid[r * GRID_COLS + c] += ovw * ovh;
      }}
    }}
  }}
}}

function rasterizeOverlap(grid) {{
  for (let i = 0; i < grid.length; i++) grid[i] = 0;
  // Pairwise hard-hard overlap, distributed to grid by intersection.
  const cw = CANVAS_W / GRID_COLS;
  const ch = CANVAS_H / GRID_ROWS;
  const pos = FRAMES[cur];
  for (let i = 0; i < N_HARD; i++) {{
    const xi = pos[i*2], yi = pos[i*2+1];
    const wi = SIZES[i][0], hi = SIZES[i][1];
    for (let j = i + 1; j < N_HARD; j++) {{
      const xj = pos[j*2], yj = pos[j*2+1];
      const wj = SIZES[j][0], hj = SIZES[j][1];
      const ox = Math.max(0, Math.min(xi + wi*0.5, xj + wj*0.5) - Math.max(xi - wi*0.5, xj - wj*0.5));
      if (ox <= 0) continue;
      const oy = Math.max(0, Math.min(yi + hi*0.5, yj + hj*0.5) - Math.max(yi - hi*0.5, yj - hj*0.5));
      if (oy <= 0) continue;
      // Intersection rect
      const ixl = Math.max(xi - wi*0.5, xj - wj*0.5);
      const ixh = Math.min(xi + wi*0.5, xj + wj*0.5);
      const iyl = Math.max(yi - hi*0.5, yj - hj*0.5);
      const iyh = Math.min(yi + hi*0.5, yj + hj*0.5);
      const cl = Math.max(0, Math.floor(ixl / cw));
      const ch_ = Math.min(GRID_COLS - 1, Math.floor(ixh / cw));
      const rl = Math.max(0, Math.floor(iyl / ch));
      const rh = Math.min(GRID_ROWS - 1, Math.floor(iyh / ch));
      for (let r = rl; r <= rh; r++) {{
        const cyl = r * ch, cyh = cyl + ch;
        const ovh = Math.max(0, Math.min(iyh, cyh) - Math.max(iyl, cyl));
        if (ovh <= 0) continue;
        for (let c = cl; c <= ch_; c++) {{
          const cxl = c * cw, cxh = cxl + cw;
          const ovw = Math.max(0, Math.min(ixh, cxh) - Math.max(ixl, cxl));
          if (ovw <= 0) continue;
          grid[r * GRID_COLS + c] += ovw * ovh;
        }}
      }}
    }}
  }}
}}

function drawHeatmap(idx, grid, palette) {{
  const [W, H] = fitCanvas(idx);
  const ctx = ctxs[idx];
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, W, H);
  let maxV = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] > maxV) maxV = grid[i];
  if (maxV <= 0) maxV = 1;
  const cellW = W / GRID_COLS;
  const cellH = H / GRID_ROWS;
  for (let r = 0; r < GRID_ROWS; r++) {{
    for (let c = 0; c < GRID_COLS; c++) {{
      const v = grid[r * GRID_COLS + c] / maxV;
      if (v <= 0) continue;
      const col = palette(v);
      ctx.fillStyle = col;
      const px = c * cellW;
      const py = H - (r + 1) * cellH;
      ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
    }}
  }}
  ctx.strokeStyle = "#666";
  ctx.lineWidth = 1.0;
  ctx.strokeRect(0, 0, W, H);
}}

function viridis(t) {{
  // Approximate viridis colormap via 5 stops
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 144, 141],
    [94, 201, 98],
    [253, 231, 37]
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(stops.length - 1, i + 1)];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${{r}},${{g}},${{bl}})`;
}}

function hot(t) {{
  // Black -> red -> yellow
  const x = Math.max(0, Math.min(1, t));
  const r = Math.round(Math.min(1, x * 3) * 255);
  const g = Math.round(Math.max(0, Math.min(1, x * 3 - 1)) * 255);
  const b = Math.round(Math.max(0, Math.min(1, x * 3 - 2)) * 255);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

const densityGrid = new Float32Array(GRID_ROWS * GRID_COLS);
const overlapGrid = new Float32Array(GRID_ROWS * GRID_COLS);

function drawScoreHistory(idx) {{
  const [W, H] = fitCanvas(idx);
  const ctx = ctxs[idx];
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, W, H);
  const PADL = 50, PADR = 14, PADT = 18, PADB = 26;
  const plotW = W - PADL - PADR;
  const plotH = H - PADT - PADB;
  const N = PROXIES.length;
  if (N < 2 || plotW < 20 || plotH < 20) return;

  // Series we plot (each normalized to [0..1] within plot)
  const series = [
    {{ name: "proxy", data: PROXIES, color: "#88ccff" }},
    {{ name: "wl", data: WL_COSTS, color: "#ffb84d" }},
    {{ name: "density", data: DEN_COSTS, color: "#a5e9a8" }},
    {{ name: "cong", data: CONG_COSTS, color: "#ff9bd2" }},
    {{ name: "overlaps", data: OV_COUNTS, color: "#ff6b6b" }}
  ];

  // Per-series scaling to [0,1]
  const ranges = series.map(s => {{
    let lo = Infinity, hi = -Infinity;
    for (const v of s.data) {{
      if (v == null || isNaN(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }}
    if (!isFinite(lo)) lo = 0;
    if (!isFinite(hi)) hi = 1;
    if (hi - lo < 1e-9) hi = lo + 1;
    return [lo, hi];
  }});

  // Axes
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PADL, PADT);
  ctx.lineTo(PADL, PADT + plotH);
  ctx.lineTo(PADL + plotW, PADT + plotH);
  ctx.stroke();

  // Plot lines
  for (let s = 0; s < series.length; s++) {{
    const [lo, hi] = ranges[s];
    ctx.strokeStyle = series[s].color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {{
      const x = PADL + (i / (N - 1)) * plotW;
      const v = series[s].data[i];
      const yt = (v - lo) / (hi - lo);
      const y = PADT + plotH - yt * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Cursor (current frame)
  const cx = PADL + (cur / (N - 1)) * plotW;
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx, PADT);
  ctx.lineTo(cx, PADT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Legend with current values
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  let lx = PADL + 6;
  const ly = PADT + 4;
  const items = [
    [series[0].color, `proxy=${{(PROXIES[cur]||0).toFixed(3)}}`],
    [series[1].color, `wl=${{(WL_COSTS[cur]||0).toFixed(3)}}`],
    [series[2].color, `den=${{(DEN_COSTS[cur]||0).toFixed(3)}}`],
    [series[3].color, `cong=${{(CONG_COSTS[cur]||0).toFixed(3)}}`],
    [series[4].color, `ovrlap=${{OV_COUNTS[cur]||0}} (${{(OV_AREAS[cur]||0).toFixed(2)}})`]
  ];
  for (const [col, txt] of items) {{
    ctx.fillStyle = col;
    ctx.fillRect(lx, ly + 4, 10, 2);
    ctx.fillStyle = "#ddd";
    const tw = ctx.measureText(txt).width;
    ctx.fillText(txt, lx + 14, ly);
    lx += tw + 30;
  }}

  // X-axis label
  ctx.fillStyle = "#999";
  ctx.font = "10px ui-monospace, monospace";
  const stepStart = STEPS[0] != null ? STEPS[0] : 0;
  const stepEnd = STEPS[STEPS.length - 1] != null ? STEPS[STEPS.length - 1] : N;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(`step ${{stepStart}}`, PADL, H - 10);
  const endLabel = `step ${{stepEnd}} (final)`;
  ctx.fillText(endLabel, PADL + plotW - ctx.measureText(endLabel).width, H - 10);
  // Current label
  const curLabel = `frame ${{cur+1}}/${{N}} ${{LABELS[cur] || ""}}`;
  ctx.fillStyle = "#fff";
  ctx.fillText(curLabel, PADL + 2, H - 10 + (cur === N - 1 ? -16 : 0));
}}

function drawAll() {{
  // Skip hidden panes
  if (activePane === -1 || activePane === 0) {{
    drawPlacement(0);
  }}
  if (activePane === -1 || activePane === 1) {{
    rasterizeFootprint(densityGrid);
    drawHeatmap(1, densityGrid, viridis);
  }}
  if (activePane === -1 || activePane === 2) {{
    rasterizeOverlap(overlapGrid);
    drawHeatmap(2, overlapGrid, hot);
  }}
  if (activePane === -1 || activePane === 3) {{
    drawScoreHistory(3);
  }}
  // Update info bar (ALL components per frame)
  const ovc = OV_COUNTS[cur] || 0;
  const ova = OV_AREAS[cur] || 0;
  const proxy = PROXIES[cur] || 0;
  const wlc = WL_COSTS[cur] || 0;
  const denc = DEN_COSTS[cur] || 0;
  const conc = CONG_COSTS[cur] || 0;
  info.textContent = `frame ${{cur+1}}/${{FRAMES.length}} ${{LABELS[cur]}} · proxy=${{proxy.toFixed(4)}} · wl=${{wlc.toFixed(3)}} den=${{denc.toFixed(3)}} cong=${{conc.toFixed(3)}} · ovrlap=${{ovc}} area=${{ova.toFixed(2)}}`;
}}

function show(i) {{
  cur = Math.max(0, Math.min(FRAMES.length - 1, i));
  slider.value = cur;
  drawAll();
}}

function setActivePane(idx) {{
  activePane = idx;
  panes.forEach((p, i) => {{
    p.classList.remove("fullscreen", "hidden", "active");
    if (idx === -1) {{
      // grid view
    }} else if (i === idx) {{
      p.classList.add("fullscreen", "active");
    }} else {{
      p.classList.add("hidden");
    }}
  }});
  // Trigger reflow then redraw at new sizes
  requestAnimationFrame(drawAll);
}}

panes.forEach((p, i) => {{
  p.addEventListener("click", () => {{
    if (activePane === i) setActivePane(-1);
    else setActivePane(i);
  }});
}});

document.getElementById("prev").onclick = () => show(cur - 1);
document.getElementById("next").onclick = () => show(cur + 1);
slider.oninput = e => show(parseInt(e.target.value));

function tick(t) {{
  if (!playing) return;
  if (!lastT) lastT = t;
  const dt = t - lastT;
  if (dt >= 1000 / (FPS_BASE * speed)) {{
    lastT = t;
    if (cur >= FRAMES.length - 1) {{
      playing = false;
      document.getElementById("play").textContent = "▶ play";
      return;
    }}
    show(cur + 1);
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
function cycleSpeed() {{
  const idx = SPEEDS.indexOf(speed);
  speed = SPEEDS[(idx + 1) % SPEEDS.length];
  speedBtn.textContent = `${{speed}}×`;
}}
speedBtn.onclick = cycleSpeed;

function toggleColor() {{
  colorMode = colorMode === "cluster" ? "kind" : "cluster";
  drawAll();
}}

function cyclePane(dir) {{
  // -1 (grid) -> 0 -> 1 -> 2 -> 3 -> -1 ...
  const order = [-1, 0, 1, 2, 3];
  const i = order.indexOf(activePane);
  const ni = (i + dir + order.length) % order.length;
  setActivePane(order[ni]);
}}

document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft") {{ show(cur - 1); e.preventDefault(); }}
  else if (e.key === "ArrowRight") {{ show(cur + 1); e.preventDefault(); }}
  else if (e.key === "ArrowUp") {{ cyclePane(+1); e.preventDefault(); }}
  else if (e.key === "ArrowDown") {{ cyclePane(-1); e.preventDefault(); }}
  else if (e.key === " ") {{ togglePlay(); e.preventDefault(); }}
  else if (e.key === "s" || e.key === "S") {{ cycleSpeed(); }}
  else if (e.key === "c" || e.key === "C") {{ toggleColor(); }}
}});

window.addEventListener("resize", drawAll);
show(0);
</script>
</body>
</html>
"""


def render_simple_html(
    out_path: str,
    benchmark,
    snapshots_pos: np.ndarray,        # [num_snap, n_total, 2]
    snapshots_step: List[int],
    cluster_id: np.ndarray,           # [n_total]
    proxies: List[float],             # [num_snap+1] including final
    labels: List[str],                # [num_snap+1]
    final_pos: Optional[np.ndarray] = None,    # [n_total, 2] — legalized
    final_proxy: Optional[float] = None,
    final_label: Optional[str] = None,
    wl_costs: Optional[List[float]] = None,
    density_costs: Optional[List[float]] = None,
    cong_costs: Optional[List[float]] = None,
    overlap_counts: Optional[List[int]] = None,
    overlap_areas: Optional[List[float]] = None,
):
    bench_name = benchmark.name
    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.cpu().numpy().tolist()
    fixed = [bool(x) for x in benchmark.macro_fixed.cpu().numpy()]
    num_clusters = int(cluster_id.max()) + 1

    palette = []
    h = 0.137
    golden = 0.6180339887
    for c in range(max(num_clusters, 1)):
        h = (h + golden) % 1.0
        hp = h * 6
        cc = 0.9 * 0.78
        x = cc * (1 - abs((hp % 2) - 1))
        if hp < 1:
            r, g, b = cc, x, 0
        elif hp < 2:
            r, g, b = x, cc, 0
        elif hp < 3:
            r, g, b = 0, cc, x
        elif hp < 4:
            r, g, b = 0, x, cc
        elif hp < 5:
            r, g, b = x, 0, cc
        else:
            r, g, b = cc, 0, x
        m = 0.9 - cc
        palette.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
    macro_colors = [list(palette[cluster_id[i]]) for i in range(n_total)]

    frames = []
    for i in range(snapshots_pos.shape[0]):
        flat = []
        p = snapshots_pos[i]
        for j in range(n_total):
            flat.append(round(float(p[j, 0]), 3))
            flat.append(round(float(p[j, 1]), 3))
        frames.append(flat)
    if final_pos is not None:
        flat = []
        for j in range(n_total):
            flat.append(round(float(final_pos[j, 0]), 3))
            flat.append(round(float(final_pos[j, 1]), 3))
        frames.append(flat)

    n_frames = len(frames)
    if wl_costs is None:
        wl_costs = [0.0] * n_frames
    if density_costs is None:
        density_costs = [0.0] * n_frames
    if cong_costs is None:
        cong_costs = [0.0] * n_frames
    if overlap_counts is None:
        overlap_counts = [0] * n_frames
    if overlap_areas is None:
        overlap_areas = [0.0] * n_frames

    steps_full = list(snapshots_step)
    if final_pos is not None and len(steps_full) < n_frames:
        steps_full.append(steps_full[-1] + 1 if steps_full else 0)

    best_proxy = float(min(proxies)) if proxies else 0.0

    html = _HTML_TEMPLATE.format(
        bench_name=bench_name,
        num_hard=n_hard,
        num_soft=n_total - n_hard,
        num_clusters=num_clusters,
        canvas_w=float(benchmark.canvas_width),
        canvas_h=float(benchmark.canvas_height),
        sizes_json=json.dumps(sizes),
        colors_json=json.dumps(macro_colors),
        fixed_json=json.dumps(fixed),
        frames_json=json.dumps(frames, separators=(",", ":")),
        labels_json=json.dumps(labels),
        proxies_json=json.dumps([round(float(p), 5) for p in proxies]),
        wl_json=json.dumps([round(float(v), 5) for v in wl_costs]),
        den_json=json.dumps([round(float(v), 5) for v in density_costs]),
        cong_json=json.dumps([round(float(v), 5) for v in cong_costs]),
        ovc_json=json.dumps([int(v) for v in overlap_counts]),
        ova_json=json.dumps([round(float(v), 4) for v in overlap_areas]),
        steps_json=json.dumps([int(s) for s in steps_full]),
        best_proxy=best_proxy,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[simple_viz] saved {out_path} ({size_mb:.1f} MB, {len(frames)} frames)",
          flush=True)
