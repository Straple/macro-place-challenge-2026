"""Render placement snapshots .npz dump to a self-contained HTML viewer.

The HTML has 4 fixed-size panels in a 2x2 grid: placement (canvas),
density heatmap (viridis), congestion heatmap (inferno), score history
plot. A slider+play controls walk through frames; each frame is rendered
on demand from raw arrays in the embedded JSON (no pre-baked PNGs), so
re-rendering with different palettes/sizes is a code change in this
script only — no need to re-run the placement pipeline.

Usage:
  uv run python scripts/render_snapshots.py results/snapshots/ibm01_dump.npz
  # produces vis/ibm01_snapshots.html

Optional:
  --output PATH   override output html path
  --max-frames N  uniform-subsample to N frames (default: keep all)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=str, help="path to <bench>_dump.npz")
    parser.add_argument("--output", type=str, default=None,
                        help="output HTML path (default: vis/<bench>_snapshots.html)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="uniform-subsample to this many frames (0 = keep all)")
    args = parser.parse_args()

    dump_path = Path(args.dump).resolve()
    data = np.load(dump_path, allow_pickle=True)
    bench_name = str(data["bench_name"])
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        # Place HTML in repo-level vis/ regardless of where dump lives.
        repo_root = dump_path.parent
        while repo_root != repo_root.parent and not (repo_root / "scripts").is_dir():
            repo_root = repo_root.parent
        out_path = repo_root / "vis" / f"{bench_name}_snapshots.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_pos = data["frames_pos"]               # [N, n_total, 2]
    frames_label = data["frames_label"]           # [N] str
    frames_metrics = data["frames_metrics"]       # [N, 5] proxy/wl/dens/cong/ovl
    frames_density = data["frames_density_grid"]  # [N, nrow, ncol]
    frames_cong = data["frames_cong_grid"]        # [N, nrow, ncol]
    macro_sizes = data["macro_sizes"]             # [n_total, 2]
    macro_fixed = data["macro_fixed"]             # [n_total]
    n_hard = int(data["n_hard"])
    n_total = int(data["n_total"])
    canvas_w = float(data["canvas_w"])
    canvas_h = float(data["canvas_h"])
    grid_rows = int(data["grid_rows"])
    grid_cols = int(data["grid_cols"])

    n_frames = frames_pos.shape[0]
    if args.max_frames and args.max_frames < n_frames:
        idx = np.linspace(0, n_frames - 1, args.max_frames).astype(int)
        frames_pos = frames_pos[idx]
        frames_label = frames_label[idx]
        frames_metrics = frames_metrics[idx]
        frames_density = frames_density[idx]
        frames_cong = frames_cong[idx]
        n_frames = len(idx)
        print(f"[render_snapshots] subsampled to {n_frames} frames", flush=True)

    cluster_ids = data["cluster_ids"] if "cluster_ids" in data.files else (
        -np.ones(n_total, dtype=np.int32))
    num_clusters = int(data["num_clusters"]) if "num_clusters" in data.files else 0

    static_macros = []
    for i in range(n_total):
        static_macros.append({
            "i": i,
            "w": float(macro_sizes[i, 0]),
            "h": float(macro_sizes[i, 1]),
            "soft": bool(i >= n_hard),
            "fixed": bool(macro_fixed[i]),
            "c": int(cluster_ids[i]),
        })

    cong_max = float(np.percentile(frames_cong[frames_cong > 0], 99)) if (
        frames_cong > 0).any() else 1.0
    dens_max = float(frames_density.max())

    frames_data = []
    for i in range(n_frames):
        proxy, wl, dens, cong, ovl = (float(x) for x in frames_metrics[i])
        frame = {
            "label": str(frames_label[i]),
            "proxy": proxy, "wl": wl, "dens": dens, "cong": cong,
            "ovl": int(ovl),
            "pos": frames_pos[i].astype(np.float32).round(4).tolist(),
            "density": frames_density[i].astype(np.float32).round(5).tolist(),
            "cong_grid": frames_cong[i].astype(np.float32).round(5).tolist(),
        }
        frames_data.append(frame)

    payload = {
        "bench_name": bench_name,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "n_hard": n_hard,
        "n_total": n_total,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "num_clusters": num_clusters,
        "macros": static_macros,
        "frames": frames_data,
        "dens_max": dens_max,
        "cong_max": cong_max,
    }

    history_proxy = [float(x[0]) for x in frames_metrics]
    history_wl = [float(x[1]) for x in frames_metrics]
    history_dens = [float(x[2]) for x in frames_metrics]
    history_cong = [float(x[3]) for x in frames_metrics]
    history_ovl = [int(x[4]) for x in frames_metrics]
    payload["history"] = {
        "proxy": history_proxy, "wl": history_wl,
        "dens": history_dens, "cong": history_cong, "ovl": history_ovl,
    }

    html = _HTML_TEMPLATE.replace(
        "/*PAYLOAD*/", json.dumps(payload, separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[render_snapshots] saved {out_path} "
          f"({n_frames} frames, {size_mb:.1f} MB)", flush=True)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Placement snapshots</title>
<style>
  :root {
    --bg: #1a1a1a;
    --panel: #232323;
    --border: #3a3a3a;
    --text: #e0e0e0;
    --muted: #888;
    --accent: #ffae42;
  }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: var(--bg); color: var(--text);
         min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
  header { width: 100%; box-sizing: border-box;
           padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
           display: flex; gap: 12px; align-items: center; justify-content: center;
           flex-wrap: wrap; }
  h1 { font-size: 14px; margin: 0; font-weight: 500; }
  .controls { display: flex; align-items: center; gap: 6px; }
  button { background: #333; color: var(--text); border: 1px solid #555;
           padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }
  button:hover { background: #444; }
  button.active { background: #5a4020; border-color: var(--accent); }
  input[type="range"] { width: 360px; }
  .info { font-family: ui-monospace, "SF Mono", Menlo, monospace;
          font-size: 12px; color: #b0b0b0; }
  .grid {
    display: grid;
    grid-template-columns: 600px 600px;
    grid-template-rows: 480px 480px;
    gap: 8px;
    padding: 14px;
    margin: 0 auto;
  }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 3px;
           position: relative; overflow: hidden; }
  .panel-title { position: absolute; top: 6px; left: 8px;
                 font-family: ui-monospace, monospace; font-size: 11px;
                 color: var(--muted); background: rgba(20,20,20,0.85);
                 padding: 2px 6px; border-radius: 2px; z-index: 5; }
  .panel canvas { display: block; }
  .ovl-warn { color: #ff6b6b; }
  .label-line { font-family: ui-monospace, monospace; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="controls">
    <button id="prev">◀</button>
    <button id="play">▶ play</button>
    <button id="next">▶</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <button id="color-mode" title="Toggle color mode">color: cluster</button>
    <span class="info" id="info"></span>
  </div>
</header>
<div class="grid">
  <div class="panel"><div class="panel-title">placement</div>
    <canvas id="placement" width="600" height="480"></canvas></div>
  <div class="panel"><div class="panel-title">density (turbo · cool=empty red=packed)</div>
    <canvas id="density" width="600" height="480"></canvas></div>
  <div class="panel"><div class="panel-title">congestion (turbo · cool=ok red=hotspot)</div>
    <canvas id="congestion" width="600" height="480"></canvas></div>
  <div class="panel"><div class="panel-title">metrics</div>
    <canvas id="metrics" width="600" height="480"></canvas></div>
</div>
<script>
const PAYLOAD = /*PAYLOAD*/;
const FRAMES = PAYLOAD.frames;
const MACROS = PAYLOAD.macros;
const HISTORY = PAYLOAD.history;
const CW = PAYLOAD.canvas_w, CH = PAYLOAD.canvas_h;
const NHARD = PAYLOAD.n_hard, NTOT = PAYLOAD.n_total;
const GR = PAYLOAD.grid_rows, GC = PAYLOAD.grid_cols;
const DENS_MAX = PAYLOAD.dens_max || 1.0;
const CONG_MAX = PAYLOAD.cong_max || 1.0;
const NCLUST = PAYLOAD.num_clusters || 0;
const N = FRAMES.length;
let colorMode = NCLUST > 0 ? "cluster" : "kind";

function hsvToRgb(h, s, v) {
  const c = v * s;
  const hp = (h % 1) * 6;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let rp, gp, bp;
  if (hp < 1) { rp = c; gp = x; bp = 0; }
  else if (hp < 2) { rp = x; gp = c; bp = 0; }
  else if (hp < 3) { rp = 0; gp = c; bp = x; }
  else if (hp < 4) { rp = 0; gp = x; bp = c; }
  else if (hp < 5) { rp = x; gp = 0; bp = c; }
  else { rp = c; gp = 0; bp = x; }
  const m = v - c;
  return [Math.round((rp + m) * 255), Math.round((gp + m) * 255),
          Math.round((bp + m) * 255)];
}
const CLUSTER_COLORS = (function () {
  const arr = [];
  const n = Math.max(1, NCLUST);
  const golden = 0.6180339887;
  let h = 0.137;
  for (let k = 0; k < n; k++) {
    h = (h + golden) % 1;
    const [r, g, b] = hsvToRgb(h, 0.78, 0.92);
    arr.push([r, g, b]);
  }
  return arr;
})();
function macroFill(m) {
  if (m.fixed) return "rgba(220, 80, 80, 0.62)";
  if (colorMode === "cluster" && m.c >= 0 && NCLUST > 0) {
    const [r, g, b] = CLUSTER_COLORS[m.c % CLUSTER_COLORS.length];
    return m.soft ? `rgba(${r},${g},${b},0.32)` : `rgba(${r},${g},${b},0.72)`;
  }
  if (m.soft) return "rgba(170, 200, 235, 0.34)";
  return "rgba(95, 150, 235, 0.68)";
}

document.getElementById("title").textContent =
  PAYLOAD.bench_name + "  ·  " + NHARD + " hard + " + (NTOT-NHARD) + " soft  ·  " +
  N + " frames  ·  grid " + GR + "×" + GC;

function buildPalette(stops) {
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = stops[0], hi = stops[stops.length-1];
    for (let s = 0; s < stops.length-1; s++) {
      if (t >= stops[s][0] && t <= stops[s+1][0]) { lo = stops[s]; hi = stops[s+1]; break; }
    }
    const span = (hi[0]-lo[0]) || 1;
    const f = (t - lo[0]) / span;
    out[i*3+0] = Math.round(lo[1] + (hi[1]-lo[1])*f);
    out[i*3+1] = Math.round(lo[2] + (hi[2]-lo[2])*f);
    out[i*3+2] = Math.round(lo[3] + (hi[3]-lo[3])*f);
  }
  return out;
}
// Plasma (Matplotlib) — perceptually uniform, dark purple → yellow.
// Used for density: empty=dark, packed=bright.
const PLASMA = buildPalette([
  [0.000, 13, 8, 135],  [0.142, 75, 3, 161],   [0.285, 125, 3, 168],
  [0.428, 168, 34, 150],[0.571, 203, 70, 121], [0.714, 229, 107, 93],
  [0.857, 248, 148, 65],[1.000, 240, 249, 33]
]);
// Turbo (Google) — rainbow with strong contrast at hotspots.
// Used for congestion: low=dark blue/green, high=red.
const TURBO = buildPalette([
  [0.000, 48, 18, 59],  [0.142, 65, 70, 224],  [0.285, 25, 188, 188],
  [0.428, 90, 240, 86], [0.571, 220, 220, 50], [0.714, 250, 100, 30],
  [0.857, 200, 25, 25], [1.000, 122, 4, 3]
]);

let cur = 0;
let playing = false;
let lastT = 0;
const FPS = 6;
const slider = document.getElementById("slider");
slider.max = N - 1;

const elTitle = document.getElementById("title");
const elInfo = document.getElementById("info");

function planeFit(canvas, dataW, dataH, padding) {
  const cw = canvas.width, ch = canvas.height;
  const aspectData = dataW / dataH;
  const aspectView = (cw - 2*padding) / (ch - 2*padding);
  let plotW, plotH;
  if (aspectView > aspectData) { plotH = ch - 2*padding; plotW = plotH * aspectData; }
  else { plotW = cw - 2*padding; plotH = plotW / aspectData; }
  const px = (cw - plotW) / 2;
  const py = (ch - plotH) / 2;
  return { px, py, plotW, plotH, sx: plotW / dataW, sy: plotH / dataH };
}

function clearPanel(ctx, w, h) {
  ctx.fillStyle = "#161616"; ctx.fillRect(0, 0, w, h);
}

function drawPlacement() {
  const cv = document.getElementById("placement");
  const ctx = cv.getContext("2d");
  clearPanel(ctx, cv.width, cv.height);
  const fit = planeFit(cv, CW, CH, 24);
  ctx.strokeStyle = "#666"; ctx.lineWidth = 1;
  ctx.strokeRect(fit.px, fit.py, fit.plotW, fit.plotH);
  const f = FRAMES[cur];
  for (let i = 0; i < MACROS.length; i++) {
    const m = MACROS[i];
    const p = f.pos[i];
    const x = fit.px + p[0] * fit.sx - (m.w * fit.sx) / 2;
    const y = fit.py + fit.plotH - (p[1] * fit.sy + (m.h * fit.sy) / 2);
    const w = m.w * fit.sx, h = m.h * fit.sy;
    ctx.fillStyle = macroFill(m);
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(0,0,0,0.55)"; ctx.lineWidth = 0.4;
    ctx.strokeRect(x, y, w, h);
  }
}

const colorModeBtn = document.getElementById("color-mode");
function refreshColorBtn() {
  if (!colorModeBtn) return;
  colorModeBtn.textContent = "color: " + colorMode;
  colorModeBtn.classList.toggle("active", colorMode === "cluster");
}
function toggleColor() {
  if (NCLUST === 0) return;
  colorMode = colorMode === "cluster" ? "kind" : "cluster";
  refreshColorBtn();
  drawPlacement();
}
if (colorModeBtn) {
  colorModeBtn.onclick = toggleColor;
  refreshColorBtn();
}

function drawColorbar(ctx, palette, x, y, w, h, vMax, label) {
  for (let i = 0; i < h; i++) {
    const t = 1 - i / (h - 1);
    const ti = Math.max(0, Math.min(255, Math.round(t * 255)));
    ctx.fillStyle = `rgb(${palette[ti*3+0]},${palette[ti*3+1]},${palette[ti*3+2]})`;
    ctx.fillRect(x, y + i, w, 1);
  }
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1;
  ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
  ctx.fillStyle = "#ddd"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(vMax.toFixed(2), x + w + 4, y + 8);
  ctx.fillText("0.00", x + w + 4, y + h);
  if (label) {
    ctx.save(); ctx.translate(x - 6, y + h / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = "#aaa"; ctx.fillText(label, 0, 0); ctx.restore();
  }
}

function drawHeatmap(canvasId, gridArr, gridMax, palette, drawMacros, label) {
  const cv = document.getElementById(canvasId);
  const ctx = cv.getContext("2d");
  clearPanel(ctx, cv.width, cv.height);
  const cbarW = 14, cbarPad = 50;
  const fit = planeFit({width: cv.width - cbarPad, height: cv.height},
                        CW, CH, 24);
  const off = document.createElement("canvas");
  off.width = GC; off.height = GR;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(GC, GR);
  const denom = Math.max(gridMax, 1e-9);
  for (let r = 0; r < GR; r++) {
    for (let c = 0; c < GC; c++) {
      const v = gridArr[r][c];
      const t = Math.max(0, Math.min(1, v / denom));
      const ti = Math.round(t * 255);
      const dst_row = (GR - 1 - r);
      const idx = (dst_row * GC + c) * 4;
      img.data[idx + 0] = palette[ti*3+0];
      img.data[idx + 1] = palette[ti*3+1];
      img.data[idx + 2] = palette[ti*3+2];
      img.data[idx + 3] = 240;
    }
  }
  offCtx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, fit.px, fit.py, fit.plotW, fit.plotH);
  ctx.strokeStyle = "#aaa"; ctx.lineWidth = 1;
  ctx.strokeRect(fit.px, fit.py, fit.plotW, fit.plotH);
  if (drawMacros) {
    const f = FRAMES[cur];
    ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 0.5;
    for (let i = 0; i < NHARD; i++) {
      const m = MACROS[i];
      const p = f.pos[i];
      const x = fit.px + p[0] * fit.sx - (m.w * fit.sx) / 2;
      const y = fit.py + fit.plotH - (p[1] * fit.sy + (m.h * fit.sy) / 2);
      ctx.strokeRect(x, y, m.w * fit.sx, m.h * fit.sy);
    }
  }
  drawColorbar(ctx, palette, cv.width - cbarPad + 8, fit.py, cbarW,
               fit.plotH, gridMax, label);
}

function drawDensity() {
  drawHeatmap("density", FRAMES[cur].density, DENS_MAX, TURBO, true,
              "density");
}
function drawCongestion() {
  drawHeatmap("congestion", FRAMES[cur].cong_grid, CONG_MAX, TURBO, true,
              "max(H,V)");
}

function drawMetrics() {
  const cv = document.getElementById("metrics");
  const ctx = cv.getContext("2d");
  clearPanel(ctx, cv.width, cv.height);
  const padL = 60, padR = 60, padT = 30, padB = 36;
  const W = cv.width - padL - padR;
  const H = cv.height - padT - padB;
  const xs = HISTORY.proxy.length;
  const series = [
    {data: HISTORY.proxy, color: "#ffae42", lw: 2.4, label: "proxy"},
    {data: HISTORY.wl,    color: "#7fdcff", lw: 1.4, label: "WL"},
    {data: HISTORY.dens,  color: "#a3e2a4", lw: 1.4, label: "density"},
    {data: HISTORY.cong,  color: "#ff8a8a", lw: 1.4, label: "congestion"},
  ];
  let yMax = 0;
  for (const s of series) for (const v of s.data) if (v > yMax) yMax = v;
  yMax = yMax * 1.05 || 1;
  // axes
  ctx.fillStyle = "#161616"; ctx.fillRect(padL, padT, W, H);
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, W, H);
  ctx.fillStyle = "#888"; ctx.font = "11px ui-monospace, monospace";
  // y ticks
  for (let k = 0; k <= 4; k++) {
    const y = padT + (H * k / 4);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + W, y);
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.stroke();
    const v = yMax * (1 - k / 4);
    ctx.fillStyle = "#888"; ctx.fillText(v.toFixed(2), 8, y + 4);
  }
  // x ticks: frame indices
  for (let k = 0; k <= 5; k++) {
    const x = padL + (W * k / 5);
    const idx = Math.round((xs - 1) * k / 5);
    ctx.fillStyle = "#888"; ctx.fillText(String(idx), x - 8, padT + H + 18);
  }
  // series lines
  for (const s of series) {
    ctx.beginPath();
    for (let i = 0; i < xs; i++) {
      const x = padL + (W * i / Math.max(1, xs - 1));
      const y = padT + H - (H * s.data[i] / yMax);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = s.color; ctx.lineWidth = s.lw; ctx.stroke();
  }
  // current frame marker
  const xMark = padL + (W * cur / Math.max(1, xs - 1));
  ctx.beginPath(); ctx.moveTo(xMark, padT); ctx.lineTo(xMark, padT + H);
  ctx.strokeStyle = "rgba(255,255,255,0.5)"; ctx.lineWidth = 1.5; ctx.stroke();
  // legend
  let lx = padL + W + 8;
  let ly = padT + 14;
  for (const s of series) {
    ctx.fillStyle = s.color; ctx.fillRect(lx, ly - 8, 10, 10);
    ctx.fillStyle = "#ddd"; ctx.fillText(s.label, lx + 14, ly);
    ly += 18;
  }
  // y-axis label
  ctx.save();
  ctx.translate(14, padT + H/2); ctx.rotate(-Math.PI/2);
  ctx.fillStyle = "#888"; ctx.fillText("cost", 0, 0);
  ctx.restore();
  // x-axis label
  ctx.fillStyle = "#888";
  ctx.fillText("frame", padL + W/2 - 12, padT + H + 32);
  // current frame numeric callout
  const f = FRAMES[cur];
  ctx.fillStyle = "#ddd"; ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`proxy = ${f.proxy.toFixed(4)}`, padL + 8, padT + 18);
  ctx.fillText(`WL=${f.wl.toFixed(3)}  D=${f.dens.toFixed(3)}  C=${f.cong.toFixed(3)}`,
               padL + 8, padT + 36);
  ctx.fillStyle = f.ovl > 0 ? "#ff8585" : "#888";
  ctx.fillText(`overlaps=${f.ovl}`, padL + 8, padT + 54);
}

function showFrame(i) {
  cur = Math.max(0, Math.min(N - 1, i));
  slider.value = cur;
  const f = FRAMES[cur];
  elInfo.textContent =
    `frame ${cur+1}/${N}  ·  ${f.label}  ·  proxy=${f.proxy.toFixed(4)}  ` +
    `WL=${f.wl.toFixed(3)} D=${f.dens.toFixed(3)} C=${f.cong.toFixed(3)} ovl=${f.ovl}`;
  drawPlacement();
  drawDensity();
  drawCongestion();
  drawMetrics();
}

document.getElementById("prev").onclick = () => showFrame(cur-1);
document.getElementById("next").onclick = () => showFrame(cur+1);
slider.oninput = (e) => showFrame(parseInt(e.target.value));

function tick(t) {
  if (!playing) return;
  if (!lastT) lastT = t;
  if (t - lastT >= 1000 / FPS) {
    lastT = t;
    if (cur >= N - 1) { playing = false; document.getElementById("play").textContent = "▶ play"; return; }
    showFrame(cur + 1);
  }
  requestAnimationFrame(tick);
}
function togglePlay() {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "❚❚ pause" : "▶ play";
  if (playing) {
    if (cur >= N - 1) cur = 0;
    lastT = 0;
    requestAnimationFrame(tick);
  }
}
document.getElementById("play").onclick = togglePlay;
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowLeft") { showFrame(cur-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { showFrame(cur+1); e.preventDefault(); }
  else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  else if (e.key === "c" || e.key === "C") { toggleColor(); e.preventDefault(); }
});

showFrame(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
