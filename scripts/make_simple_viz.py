"""Simple JS-only HTML viewer: every step animation, no matplotlib.

Embeds ALL frames as JSON (positions + sizes + cluster colors), browser
canvas renders rectangles per-frame. ~5-10 MB for 250 frames × 1140 macros.
Renders 60+ FPS in browser.

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
  padding: 12px 18px;
  background: #2a2a2a;
  border-bottom: 1px solid #404040;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}}
h1 {{ font-size: 15px; margin: 0; font-weight: 500; }}
button {{
  background: #3a3a3a;
  color: #e0e0e0;
  border: 1px solid #555;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}}
button:hover {{ background: #4a4a4a; }}
input[type="range"] {{ width: 360px; }}
.frame-info {{
  font-family: ui-monospace, monospace;
  font-size: 12px;
  color: #b0b0b0;
}}
#wrap {{
  display: flex;
  justify-content: center;
  padding: 12px;
}}
#canvas {{
  background: #1a1a1a;
  border: 1px solid #444;
  cursor: crosshair;
}}
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
  z-index: 10;
}}
.help {{ font-size: 11px; color: #777; }}
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
  <span class="help">← → keys • space play/pause • s = speed • c = color</span>
</header>
<div id="wrap">
  <canvas id="canvas" width="900" height="900"></canvas>
  <div id="tooltip"></div>
</div>
<script>
const SIZES = {sizes_json};
const COLORS = {colors_json};
const FIXED = {fixed_json};
const N_HARD = {num_hard};
const FRAMES = {frames_json};
const LABELS = {labels_json};
const PROXIES = {proxies_json};
const CANVAS_W = {canvas_w};
const CANVAS_H = {canvas_h};
const NUM_CLUSTERS = {num_clusters};

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
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
const SPEEDS = [1, 2, 4, 8, 0.5];
const FPS_BASE = 30;

function fitCanvas() {{
  const W = Math.min(window.innerWidth - 50, 900);
  const H = Math.min(window.innerHeight - 100, 900);
  const ratio = CANVAS_W / CANVAS_H;
  let w, h;
  if (W / H > ratio) {{ h = H; w = h * ratio; }}
  else {{ w = W; h = w / ratio; }}
  canvas.width = w * window.devicePixelRatio;
  canvas.height = h * window.devicePixelRatio;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  return [w, h];
}}

function drawFrame() {{
  const [W, H] = fitCanvas();
  const sx = W / CANVAS_W;
  const sy = H / CANVAS_H;
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, W, H);
  const pos = FRAMES[cur];
  // Sort by area desc — large first, small overlay
  const ord = SIZES.map((s, i) => [s[0]*s[1], i]).sort((a,b) => b[0]-a[0]).map(x => x[1]);
  for (const i of ord) {{
    const x = pos[i*2], y = pos[i*2+1];
    const w = SIZES[i][0], h = SIZES[i][1];
    const c = COLORS[i];
    let color;
    if (FIXED[i]) color = "rgba(220,60,60,0.55)";
    else if (i >= N_HARD) {{
      // soft
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
  // border
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1.2;
  ctx.strokeRect(0, 0, W, H);
  const proxy = PROXIES[cur];
  info.textContent = `frame ${{cur+1}}/${{FRAMES.length}} · ${{LABELS[cur]}} · proxy=${{proxy.toFixed(4)}}`;
}}

function show(i) {{
  cur = Math.max(0, Math.min(FRAMES.length - 1, i));
  slider.value = cur;
  drawFrame();
}}

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
  drawFrame();
}}

document.addEventListener("keydown", e => {{
  if (e.key === "ArrowLeft") {{ show(cur - 1); e.preventDefault(); }}
  else if (e.key === "ArrowRight") {{ show(cur + 1); e.preventDefault(); }}
  else if (e.key === " ") {{ togglePlay(); e.preventDefault(); }}
  else if (e.key === "s" || e.key === "S") {{ cycleSpeed(); }}
  else if (e.key === "c" || e.key === "C") {{ toggleColor(); }}
}});

window.addEventListener("resize", drawFrame);
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
):
    bench_name = benchmark.name
    n_total = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.cpu().numpy().tolist()
    fixed = [bool(x) for x in benchmark.macro_fixed.cpu().numpy()]
    num_clusters = int(cluster_id.max()) + 1

    # Cluster palette: HSV golden ratio
    palette = []
    h = 0.137
    golden = 0.6180339887
    for c in range(max(num_clusters, 1)):
        h = (h + golden) % 1.0
        # HSV(h, 0.78, 0.9) -> RGB
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

    # Round positions to 3 decimals to keep JSON small
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
        proxies_json=json.dumps(proxies),
        best_proxy=best_proxy,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[simple_viz] saved {out_path} ({size_mb:.1f} MB, {len(frames)} frames)",
          flush=True)
