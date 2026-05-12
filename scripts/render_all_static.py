"""Render static placement PNG for each ibm bench with available data.

For benches with snapshot dump (.npz): use final-frame placement + cluster
colors.
For benches with only seed pkl: scatter macro centers (no sizes available
locally) colored by hard/soft.

Outputs:
  vis/<bench>_placement.png  per-bench PNG
  vis/avg17_summary.png      4x4 grid of all 13 placements
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def hsv_palette(n):
    out = []
    h = 0.137
    golden = 0.6180339887
    for k in range(n):
        h = (h + golden) % 1
        c, v = 0.78, 0.92
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


def draw_placement_from_dump(ax, dump_path, bench_label):
    d = np.load(dump_path, allow_pickle=True)
    pos = d["frames_pos"][-1]
    sizes = d["macro_sizes"]
    fixed = d["macro_fixed"]
    cluster_ids = d["cluster_ids"] if "cluster_ids" in d.files else None
    n_hard = int(d["n_hard"])
    n_total = int(d["n_total"])
    cw = float(d["canvas_w"])
    ch = float(d["canvas_h"])
    metrics = d["frames_metrics"][-1]
    proxy = float(metrics[0])
    label = str(d["frames_label"][-1])

    nc = int(cluster_ids.max()) + 1 if cluster_ids is not None else 40
    palette = hsv_palette(max(40, nc))
    ax.add_patch(Rectangle((0, 0), cw, ch, fill=False, edgecolor="black", linewidth=1))
    for i in range(n_total):
        x, y = pos[i]
        w, h = sizes[i]
        if fixed[i]:
            color = (0.86, 0.31, 0.31, 0.7)
        else:
            cid = int(cluster_ids[i]) if cluster_ids is not None else 0
            r, g, b = palette[cid % len(palette)]
            alpha = 0.32 if i >= n_hard else 0.72
            color = (r, g, b, alpha)
        ax.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                                facecolor=color, edgecolor="black", linewidth=0.2))
    ax.set_xlim(0, cw); ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.set_title(f"{bench_label} · proxy={proxy:.4f} · {label}", fontsize=10)
    ax.tick_params(labelsize=7)


def draw_placement_from_pkl(ax, seed_pkl, bench_label, canvas_w, canvas_h):
    s = pickle.load(open(seed_pkl, "rb"))
    hard = s["hard"]
    soft = s["soft"]
    proxy = float(s.get("proxy", 0))
    n_hard = len(hard)
    n_soft = len(soft)
    ax.add_patch(Rectangle((0, 0), canvas_w, canvas_h, fill=False,
                            edgecolor="black", linewidth=1))
    ax.scatter(soft[:, 0], soft[:, 1], s=4, c="lightsteelblue", alpha=0.5,
                edgecolors="none")
    ax.scatter(hard[:, 0], hard[:, 1], s=10, c="navy", alpha=0.8,
                edgecolors="black", linewidths=0.3)
    ax.set_xlim(0, canvas_w); ax.set_ylim(0, canvas_h)
    ax.set_aspect("equal")
    ax.set_title(f"{bench_label} · proxy={proxy:.4f} · pre-pipeline",
                  fontsize=10)
    ax.tick_params(labelsize=7)


def main():
    proxies = {
        "ibm01": 0.8882, "ibm02": 1.4182, "ibm03": 1.2283, "ibm04": 1.1687,
        "ibm06": 1.5940, "ibm07": 1.2385, "ibm08": 1.4124, "ibm09": 1.0137,
        "ibm10": 1.3357, "ibm11": 1.0649, "ibm12": 1.5753, "ibm13": 1.1669,
        "ibm14": 1.4796,
    }

    # Inferred canvas size from ibm01_dump for fallback (close enough for
    # scatter visualization).
    d0 = np.load(REPO / "results/avg17/ibm01_dump.npz", allow_pickle=True)
    canvas_w = float(d0["canvas_w"])
    canvas_h = float(d0["canvas_h"])

    benches = list(proxies.keys())
    n = len(benches)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5.2))
    axes = axes.flatten()

    vis_dir = REPO / "vis"
    vis_dir.mkdir(exist_ok=True)

    for i, b in enumerate(benches):
        ax = axes[i]
        dump = REPO / f"results/avg17/{b}_dump.npz"
        pkl = REPO / f"results/avg17/seed_{b}.pkl"
        if dump.exists():
            draw_placement_from_dump(ax, dump, b)
        elif pkl.exists():
            draw_placement_from_pkl(ax, pkl, b, canvas_w, canvas_h)
        else:
            ax.text(0.5, 0.5, f"{b}\nproxy={proxies[b]:.4f}\n(no placement data)",
                     ha="center", va="center", transform=ax.transAxes,
                     fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

    for i in range(n, rows * cols):
        axes[i].axis("off")

    avg = np.mean(list(proxies.values()))
    fig.suptitle(
        f"AVG17 partial — 13/17 IBM benches done, average proxy = {avg:.4f}  "
        f"(target ≤ 1.4578)",
        fontsize=14, fontweight="bold")
    out = vis_dir / "avg17_summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"saved {out}  ({out.stat().st_size//1024} KB)")

    # Per-bench PNGs
    for b in benches:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        dump = REPO / f"results/avg17/{b}_dump.npz"
        pkl = REPO / f"results/avg17/seed_{b}.pkl"
        if dump.exists():
            draw_placement_from_dump(ax2, dump, b)
        elif pkl.exists():
            draw_placement_from_pkl(ax2, pkl, b, canvas_w, canvas_h)
        else:
            ax2.text(0.5, 0.5, f"{b}: no data", ha="center", va="center",
                      transform=ax2.transAxes)
        fig2.tight_layout()
        fig2.savefig(vis_dir / f"{b}_placement.png", dpi=100, bbox_inches="tight")
        plt.close(fig2)
    print(f"saved per-bench PNGs in {vis_dir}/")


if __name__ == "__main__":
    main()
