"""Generate a single PNG that explains the current state of the problem.

Reads results/snapshots/ibm01_dump.npz (baseline) and produces a 2x2
figure: placement, congestion heatmap, longest nets traced, summary
text. Intended as a one-shot visual explainer.

Usage:
  python3 scripts/explain_problem.py
  → vis/explain_problem.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO = Path(__file__).resolve().parent.parent


def hsv_palette(n):
    out = []
    h = 0.137
    golden = 0.6180339887
    for k in range(n):
        h = (h + golden) % 1
        c = 0.78
        v = 0.92
        i = int(h * 6)
        f = h * 6 - i
        p, q, t = v * (1 - c), v * (1 - f * c), v * (1 - (1 - f) * c)
        if i % 6 == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        out.append((r, g, b))
    return out


def main():
    dump = np.load(REPO / "results/snapshots/ibm01_dump.npz",
                    allow_pickle=True)
    pos = dump["frames_pos"][-1]
    cong = dump["frames_cong_grid"][-1]
    sizes = dump["macro_sizes"]
    fixed = dump["macro_fixed"]
    cluster_ids = dump["cluster_ids"]
    n_hard = int(dump["n_hard"])
    n_total = int(dump["n_total"])
    cw = float(dump["canvas_w"])
    ch = float(dump["canvas_h"])
    metrics = dump["frames_metrics"][-1]
    proxy, wl, dens, cong_val = float(metrics[0]), float(metrics[1]), float(metrics[2]), float(metrics[3])
    GR, GC = cong.shape
    cell_w, cell_h = cw / GC, ch / GR

    diag_path = REPO / "results/diag_lroute_dump.txt"
    long_nets = []
    if diag_path.exists():
        text = diag_path.read_text()
        in_long_section = False
        for line in text.splitlines():
            if "LONGEST nets" in line:
                in_long_section = True; continue
            if in_long_section and line.startswith("  net#"):
                import re
                m = re.match(r"\s*net#\s*(\d+)\s+L_cells=(\d+)\s+pins=(\d+)\s+spans=(\d+)clusters\s+owners=\[(.+)\]\s*$", line)
                if not m:
                    continue
                net_id = int(m.group(1))
                lcells = int(m.group(2))
                pins = int(m.group(3))
                owners = []
                for tok in m.group(5).split(","):
                    tok = tok.strip()
                    om = re.match(r"[SH]#(\d+)", tok)
                    if om:
                        owners.append(int(om.group(1)))
                long_nets.append({"id": net_id, "lcells": lcells,
                                   "pins": pins, "owners": owners})
            elif in_long_section and not line.startswith(" "):
                break

    cluster_palette = hsv_palette(max(40, int(cluster_ids.max()) + 1))

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20)

    # Panel 1: placement with cluster colors
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1.5))
    for i in range(n_total):
        x, y = pos[i]
        w, h = sizes[i]
        cid = int(cluster_ids[i])
        if fixed[i]:
            color = (0.86, 0.31, 0.31, 0.7)
        else:
            r, g, b = cluster_palette[cid % len(cluster_palette)]
            alpha = 0.32 if i >= n_hard else 0.72
            color = (r, g, b, alpha)
        ax1.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                                 facecolor=color, edgecolor="black", linewidth=0.3))
    ax1.set_xlim(0, cw); ax1.set_ylim(0, ch)
    ax1.set_aspect("equal")
    ax1.set_title(f"Placement — макросы по кластерам ({n_hard}H + {n_total-n_hard}S)", fontsize=12)
    ax1.set_xlabel("x (μm)"); ax1.set_ylabel("y (μm)")

    # Panel 2: congestion heatmap with hotspot markers
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(cong, origin="lower", extent=(0, cw, 0, ch),
                    cmap="turbo", aspect="equal")
    plt.colorbar(im, ax=ax2, fraction=0.046, label="cong (макс H,V)")
    flat = cong.flatten()
    top5 = np.argsort(flat)[::-1][:5]
    for rank, idx in enumerate(top5):
        r_idx, c_idx = divmod(int(idx), GC)
        cx_lo = c_idx * cell_w; cy_lo = r_idx * cell_h
        ax2.add_patch(Rectangle((cx_lo, cy_lo), cell_w, cell_h,
                                 fill=False, edgecolor="white", linewidth=1.8))
        ax2.annotate(f"#{rank+1}\ncong={cong[r_idx, c_idx]:.2f}",
                     xy=(cx_lo + cell_w/2, cy_lo + cell_h/2),
                     fontsize=8, color="white", ha="center", va="center",
                     fontweight="bold")
    ax2.set_xlim(0, cw); ax2.set_ylim(0, ch)
    ax2.set_title(f"Congestion heatmap — пики (cong={cong_val:.3f})", fontsize=12)
    ax2.set_xlabel("x (μm)"); ax2.set_ylabel("y (μm)")

    # Panel 3: long nets traced
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1))
    for i in range(n_total):
        x, y = pos[i]
        w, h = sizes[i]
        if i >= n_hard:
            ax3.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                                     facecolor="lightgray", edgecolor="none", alpha=0.4))
        else:
            ax3.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                                     facecolor="navy", edgecolor="black", linewidth=0.3, alpha=0.6))
    cmap_long = plt.cm.cool
    for rank, net in enumerate(long_nets[:8]):
        owners = net["owners"]
        if not owners: continue
        xs, ys = [], []
        for o in owners:
            if 0 <= o < n_total:
                xs.append(pos[o, 0]); ys.append(pos[o, 1])
        if len(xs) < 2: continue
        col = cmap_long(rank / 8)
        for k in range(1, len(xs)):
            ax3.plot([xs[0], xs[k]], [ys[0], ys[k]],
                     color=col, linewidth=1.5, alpha=0.85,
                     solid_capstyle="round")
        cx_mean = np.mean(xs); cy_mean = np.mean(ys)
        ax3.annotate(f"#{net['id']} ({net['pins']}p, {net['lcells']}c)",
                     xy=(cx_mean, cy_mean),
                     fontsize=7, color="black",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                     ha="center")
    ax3.set_xlim(0, cw); ax3.set_ylim(0, ch)
    ax3.set_aspect("equal")
    ax3.set_title("Top 8 длиннейших nets (by L-route cells touched)", fontsize=12)
    ax3.set_xlabel("x (μm)"); ax3.set_ylabel("y (μm)")

    # Panel 4: summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary = f"""ТЕКУЩИЕ МЕТРИКИ (ibm01, baseline trial9):
  proxy = {proxy:.4f}     цель ≤ 0.85, vmallela 0.7644
  WL    = {wl:.4f}     wirelength
  dens  = {dens:.4f}    плотность (макро на канвас)
  cong  = {cong_val:.4f}    congestion (узкие места routing)

ЧТО ЗА ПРОБЛЕМА (сверху-вправо):
  cong_max = {cong.max():.2f} в зоне ~5×5 μm.
  В цифрах: TILOS считает что в этих ячейках
  слишком много проводов хочет пройти. Если
  cong > 1.0 — DRC violation вероятна.

КТО ВИНОВАТ (снизу-слева):
  Длинные multi-pin nets (4-7 пинов через 3-6
  кластеров) проходят через эту зону.
  Top: net#3036 — 7 пинов через 6 кластеров, 135
  routing-cells. Тянет провод через canvas.

ЧТО МЫ ПЫТАЛИСЬ:
  ✗ Раздвинуть кластера → длинные провода ↑ → cong ↑
  ✗ FastPlace 1/(k-1) net weighting → noise
  ✗ Star-net для k≥4 → слабое улучшение cong
    ценой WL
  ✗ Cong inflate (push macros from hot cells)
    → катастрофа
  ✗ L-route loss bbox-center proxy → cong ↑
  ◐ L-BFGS finisher → marginal +0.6% (3/5 seeds)

ЧТО ХОЧУ ДАЛЬШЕ:
  L-BFGS — единственное направление, давшее
  positive signal, хоть и слабый. Tune
  hyperparams (alpha, from_step, добавить Wolfe
  line search) или verify на full K=384 1200s.

  Альтернативы: per-edge L-route loss (~6h
  refactor), HPO с L-BFGS в search space,
  принять текущий ~0.89 как floor."""
    ax4.text(0.02, 0.98, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffae0",
                        edgecolor="#888"))

    fig.suptitle("Macro Placement Challenge — состояние ibm01 на baseline",
                  fontsize=14, fontweight="bold", y=0.995)
    out = REPO / "vis/explain_problem.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}  ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
