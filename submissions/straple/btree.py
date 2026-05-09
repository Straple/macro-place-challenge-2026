"""B*-tree representation for macro placement local search.

A B*-tree is a binary tree where each node stores a macro. Placement
follows DFS pre-order with a contour:
  - root at (0, 0)
  - left child of P at x = P.x + P.width (right of P)
  - right child of P at x = P.x (same column, but stacked above)
  - y = max contour height in [x, x + width)
  - contour updated after each placement

Resulting layout is overlap-free by construction.

Local moves:
  - swap_macros(i, j): exchange macro contents between two nodes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BNode:
    macro_idx: int
    width: float
    height: float
    left: int = -1
    right: int = -1
    parent: int = -1
    x: float = 0.0
    y: float = 0.0


@dataclass
class BTree:
    nodes: list = field(default_factory=list)
    root: int = -1


def _build_balanced(macro_indices: list, sizes_hard: np.ndarray) -> BTree:
    """Build a balanced B*-tree from a list of macro indices."""
    n = len(macro_indices)
    nodes = []
    for i, mi in enumerate(macro_indices):
        nodes.append(BNode(macro_idx=int(mi),
                            width=float(sizes_hard[mi, 0]),
                            height=float(sizes_hard[mi, 1])))
    if n == 0:
        return BTree(nodes=[], root=-1)

    def build(lo: int, hi: int, parent: int, is_left: bool) -> int:
        if lo > hi:
            return -1
        mid = (lo + hi) // 2
        nodes[mid].parent = parent
        nodes[mid].left = build(lo, mid - 1, mid, True)
        nodes[mid].right = build(mid + 1, hi, mid, False)
        if parent >= 0:
            if is_left:
                nodes[parent].left = mid
            else:
                nodes[parent].right = mid
        return mid

    root = build(0, n - 1, -1, False)
    return BTree(nodes=nodes, root=root)


def build_from_positions(pos_hard: np.ndarray,
                          sizes_hard: np.ndarray) -> BTree:
    """Build a B*-tree from current placement.

    Strategy: sort macros by x, then y. Use balanced binary tree on this
    ordered list. The resulting placement may not match input exactly,
    but it's a valid starting tree for local search.
    """
    n = len(pos_hard)
    if n == 0:
        return BTree(nodes=[], root=-1)
    order = np.lexsort((pos_hard[:, 1], pos_hard[:, 0]))
    return _build_balanced(list(order), sizes_hard)


def place_tree(tree: BTree, canvas_w: float, canvas_h: float
                ) -> tuple[np.ndarray, bool]:
    """Compute (x, y) for each node via DFS pre-order + contour.

    Returns (centers [n, 2], legal_flag). centers[i] = (cx, cy) of the
    macro at tree.nodes[i]. legal_flag = False if any node falls outside
    canvas.
    """
    n = len(tree.nodes)
    centers = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return centers, True

    contour: list = []
    legal = True
    stack: list = [(tree.root, True, 0.0)]
    placed = [False] * n

    def contour_height(x_lo: float, x_hi: float) -> float:
        h = 0.0
        for seg_lo, seg_hi, seg_h in contour:
            if seg_hi <= x_lo or seg_lo >= x_hi:
                continue
            if seg_h > h:
                h = seg_h
        return h

    def contour_insert(x_lo: float, x_hi: float, new_h: float) -> None:
        new_contour = []
        for seg_lo, seg_hi, seg_h in contour:
            if seg_hi <= x_lo or seg_lo >= x_hi:
                new_contour.append((seg_lo, seg_hi, seg_h))
            else:
                if seg_lo < x_lo:
                    new_contour.append((seg_lo, x_lo, seg_h))
                if seg_hi > x_hi:
                    new_contour.append((x_hi, seg_hi, seg_h))
        new_contour.append((x_lo, x_hi, new_h))
        new_contour.sort(key=lambda s: s[0])
        contour.clear()
        contour.extend(new_contour)

    while stack:
        ni, is_root, x_param = stack.pop()
        if ni < 0:
            continue
        node = tree.nodes[ni]
        if is_root:
            x = 0.0
        else:
            x = x_param
        x_end = x + node.width
        y = contour_height(x, x_end)
        node.x = x
        node.y = y
        centers[ni, 0] = x + node.width * 0.5
        centers[ni, 1] = y + node.height * 0.5
        placed[ni] = True
        if x_end > canvas_w + 1e-6 or y + node.height > canvas_h + 1e-6:
            legal = False
        contour_insert(x, x_end, y + node.height)
        if node.right >= 0:
            stack.append((node.right, False, node.x))
        if node.left >= 0:
            stack.append((node.left, False, node.x + node.width))

    if not all(placed):
        legal = False
    return centers, legal


def to_full_positions(centers: np.ndarray, tree: BTree,
                       full_template: np.ndarray) -> np.ndarray:
    """Map tree-node centers to a [n_total, 2] hard+soft array.

    full_template provides existing soft positions; only hard slots get
    overwritten.
    """
    out = full_template.copy()
    for ni, node in enumerate(tree.nodes):
        out[node.macro_idx, 0] = centers[ni, 0]
        out[node.macro_idx, 1] = centers[ni, 1]
    return out


def swap_macros(tree: BTree, ni: int, nj: int) -> None:
    """Swap macro contents (idx, width, height) between two nodes.

    Tree structure preserved.
    """
    a = tree.nodes[ni]
    b = tree.nodes[nj]
    a.macro_idx, b.macro_idx = b.macro_idx, a.macro_idx
    a.width, b.width = b.width, a.width
    a.height, b.height = b.height, a.height


def clone_tree(tree: BTree) -> BTree:
    return BTree(
        nodes=[BNode(
            macro_idx=n.macro_idx, width=n.width, height=n.height,
            left=n.left, right=n.right, parent=n.parent, x=n.x, y=n.y)
            for n in tree.nodes],
        root=tree.root)


def btree_polish(benchmark, plc, pos_full: np.ndarray,
                   n_rounds: int = 4,
                   verbose: bool = False,
                   time_budget: float = 0.0) -> tuple[np.ndarray, float]:
    """B*-tree local search polish.

    Pipeline:
      1. Build B*-tree from current hard macro positions.
      2. Place tree to obtain coordinates; verify legal & compute proxy.
         If proxy >= original → revert.
      3. Iterate: for each pair (i, j), try swap_macros, place, eval.
         Accept best swap of round if improves.
      4. Final TILOS verify.
    """
    import time
    import torch
    from macro_place.objective import compute_proxy_cost

    n_hard = int(benchmark.num_hard_macros)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    sizes_hard = sizes[:n_hard]

    pos = pos_full.astype(np.float64).copy()

    def _tilos(p_np: np.ndarray) -> tuple[float, int]:
        full = torch.tensor(p_np, dtype=torch.float32)
        c = compute_proxy_cost(full, benchmark, plc)
        return float(c["proxy_cost"]), int(c["overlap_count"])

    base_proxy_orig, base_ovrlp = _tilos(pos)
    if verbose:
        print(f"[btree] start proxy={base_proxy_orig:.4f} ovrlp={base_ovrlp}",
              flush=True)

    movable = [i for i in range(n_hard) if not fixed[i]]
    if len(movable) < 2:
        return pos, base_proxy_orig

    pos_hard = pos[:n_hard]
    tree = build_from_positions(pos_hard[movable], sizes_hard[movable])
    centers, legal = place_tree(tree, canvas_w, canvas_h)
    if not legal:
        if verbose:
            print(f"[btree] initial tree placement out-of-bounds, abort",
                  flush=True)
        return pos, base_proxy_orig

    base_pos = pos.copy()
    for li, mi in enumerate(movable):
        base_pos[mi, 0] = centers[li, 0]
        base_pos[mi, 1] = centers[li, 1]
    base_p, base_o = _tilos(base_pos)
    if verbose:
        print(f"[btree] tree-built proxy={base_p:.4f} ovrlp={base_o}",
              flush=True)
    if base_p >= base_proxy_orig - 1e-6 or base_o > base_ovrlp:
        if verbose:
            print(f"[btree] tree placement worse than orig -- abort",
                  flush=True)
        return pos, base_proxy_orig

    cur_tree = tree
    cur_pos = base_pos
    cur_proxy = base_p
    cur_ovrlp = base_o
    n_tree = len(cur_tree.nodes)
    t_start = time.time()

    for r in range(n_rounds):
        if time_budget > 0 and (time.time() - t_start) >= time_budget:
            break
        improvements = 0
        best_i = -1
        best_j = -1
        best_p = cur_proxy
        best_pos_round = cur_pos
        for i in range(n_tree):
            if time_budget > 0 and (time.time() - t_start) >= time_budget:
                break
            for j in range(i + 1, n_tree):
                trial = clone_tree(cur_tree)
                swap_macros(trial, i, j)
                trial_centers, trial_legal = place_tree(
                    trial, canvas_w, canvas_h)
                if not trial_legal:
                    continue
                trial_pos = pos.copy()
                for li, mi in enumerate(movable):
                    trial_pos[mi, 0] = trial_centers[li, 0]
                    trial_pos[mi, 1] = trial_centers[li, 1]
                tp, to_ = _tilos(trial_pos)
                if to_ > cur_ovrlp:
                    continue
                if tp < best_p - 1e-6:
                    best_p = tp
                    best_i = i
                    best_j = j
                    best_pos_round = trial_pos
        if best_i < 0:
            if verbose:
                print(f"[btree] round {r+1}: no improving swap", flush=True)
            break
        swap_macros(cur_tree, best_i, best_j)
        cur_pos = best_pos_round
        cur_proxy = best_p
        improvements += 1
        if verbose:
            print(f"[btree] round {r+1}/{n_rounds}: swap nodes "
                  f"({best_i},{best_j}) proxy {cur_proxy:.4f} "
                  f"elapsed {time.time()-t_start:.1f}s", flush=True)

    final_p, final_o = _tilos(cur_pos)
    if verbose:
        print(f"[btree] final TILOS proxy={final_p:.4f} ovrlp={final_o}",
              flush=True)
    if final_p >= base_proxy_orig - 1e-6 or final_o > base_ovrlp:
        if verbose:
            print(f"[btree] REVERT: not better than orig", flush=True)
        return pos_full, base_proxy_orig
    return cur_pos, final_p
