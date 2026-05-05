"""Рендер grid PNG из pos_K (все K финальных состояний от gradient_batch).

Каждый thumbnail: placement с cluster colors, под ним proxy + ovrlp метка.
Результат: vis/<bench>_gpu_grid.png — большая сетка (cols × rows).

Usage:
    uv run python scripts/render_grid.py --bench ibm01 --cols 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--cols", type=int, default=24)
    parser.add_argument("--cell-size", type=float, default=2.5,
                        help="inches per thumbnail")
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--show-soft", action="store_true",
                        help="render soft macros too (slow, uses 1140 instead of 246 patches)")
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    from clustering import cluster_macros

    bdir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.bench
    print(f"[grid] loading {args.bench}...", flush=True)
    benchmark, _ = load_benchmark_from_dir(str(bdir))
    n_hard = benchmark.num_hard_macros
    n_total = benchmark.num_macros

    npz_path = REPO_ROOT / "results" / f"gpu_pos_K_{args.bench}.npz"
    if not npz_path.exists():
        print(f"[grid] ERROR: {npz_path} not found. Run gpu_run_one.py first.",
              flush=True)
        sys.exit(1)
    data = np.load(npz_path)
    pos_K = data["pos_K"]
    proxies = data["proxies"]
    overlaps = data["overlaps"]
    best_idx = int(data["best_idx"][0])
    K = pos_K.shape[0]
    print(f"[grid] loaded K={K}, best_idx={best_idx}, "
          f"best_proxy={proxies[best_idx]:.4f}", flush=True)

    cluster_target = max(15, n_total // 30)
    cluster_id, num_clusters, _ = cluster_macros(
        benchmark, method="louvain", seed=42,
        max_net_size=20, target_num_clusters=cluster_target,
    )
    print(f"[grid] {num_clusters} clusters", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import colorsys

    # HSV golden ratio palette
    golden = 0.6180339887
    palette = []
    h = 0.137
    for c in range(num_clusters):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.9)
        palette.append((r, g, b))
    colors = np.array(palette, dtype=np.float32)
    macro_colors_hard = colors[cluster_id[:n_hard]]
    if args.show_soft:
        macro_colors_soft = colors[cluster_id[n_hard:]]

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0

    rows = (K + args.cols - 1) // args.cols
    print(f"[grid] grid {rows} rows × {args.cols} cols, "
          f"thumbnail {args.cell_size}x{args.cell_size}", flush=True)

    fig_w = args.cols * args.cell_size
    fig_h = rows * args.cell_size
    fig, axes = plt.subplots(rows, args.cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005,
                        wspace=0.03, hspace=0.06)

    sorted_proxy = sorted(proxies)
    p25 = sorted_proxy[len(sorted_proxy) // 4] if K >= 4 else sorted_proxy[0]
    p50 = sorted_proxy[len(sorted_proxy) // 2]

    t0 = time.time()
    for k in range(rows * args.cols):
        ax = axes[k // args.cols][k % args.cols]
        if k >= K:
            ax.axis("off")
            continue
        # Background
        proxy = float(proxies[k])
        ovrlp = int(overlaps[k])
        if k == best_idx:
            bg = (0.20, 0.55, 0.20)  # green for winner
        elif ovrlp > 0:
            bg = (0.30, 0.10, 0.10)
        elif proxy <= p25:
            bg = (0.10, 0.20, 0.10)
        elif proxy <= p50:
            bg = (0.13, 0.13, 0.13)
        else:
            bg = (0.10, 0.10, 0.10)
        ax.set_facecolor(bg)

        # Plot macros
        if args.show_soft:
            soft = pos_K[k, n_hard:]
            for i in range(n_total - n_hard):
                x, y = soft[i]
                w = sizes[n_hard + i, 0]
                h_ = sizes[n_hard + i, 1]
                col = macro_colors_soft[i]
                ax.add_patch(Rectangle((x - w / 2, y - h_ / 2), w, h_,
                                       facecolor=(*col, 0.35),
                                       edgecolor="none"))
        hard = pos_K[k, :n_hard]
        for i in range(n_hard):
            x, y = hard[i]
            w = sizes[i, 0]
            h_ = sizes[i, 1]
            col = macro_colors_hard[i]
            ax.add_patch(Rectangle((x - w / 2, y - h_ / 2), w, h_,
                                   facecolor=(*col, 0.85),
                                   edgecolor="black",
                                   linewidth=0.15))
        ax.set_xlim(0, canvas_w)
        ax.set_ylim(0, canvas_h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color((0.4, 0.4, 0.4))
            spine.set_linewidth(0.5)

        title = f"k={k} p={proxy:.3f}"
        if ovrlp > 0:
            title += f" ⚠{ovrlp}"
        if k == best_idx:
            title = f"★ {title}"
        ax.set_title(title, fontsize=7, color="white", pad=2)
        if (k + 1) % 32 == 0:
            print(f"[grid]   rendered {k+1}/{K}", flush=True)

    fig.patch.set_facecolor((0.05, 0.05, 0.05))
    out_path = REPO_ROOT / "vis" / f"{args.bench}_gpu_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[grid] saving {out_path} ...", flush=True)
    fig.savefig(out_path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[grid] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
