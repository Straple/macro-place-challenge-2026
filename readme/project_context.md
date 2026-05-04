---
name: Macro Placement Challenge 2026
description: Partcl/HRT macro placement competition context — team, dates, targets, strategy
type: project
originSessionId: 04d616cc-d794-45c3-b9ba-668b61000015
---
Team: **Straple**, repo `Straple/macro-place-challenge-2026`. Deadline: **2026-05-21 23:59 PT**.

Targets (lower is better, AVG proxy on 17 IBM benchmarks):
- 1.5336 — current best (will_seed reference from organizers)
- 1.4578 — RePlAce baseline = minimum threshold to clear
- 1.3479 — top-7 → Tier 2 / Grand Prize qualification
- 1.2224 — first place (Cezar)

Strategy: DREAMPlace seed → LNS refinement (Phase 1-3 in readme/todo.md). Currently in Phase 1 (LNS skeleton).

**Why:** Composite metric is `1.0 × wirelength + 0.5 × density + 0.5 × congestion`. Per will_seed decomposition, congestion is 66% of total cost — main bottleneck.

**How to apply:** When optimizing, focus on density+congestion (94% of cost combined), not WL (6%). LNS that spreads macros via centroid placement reduces both. SA on HPWL alone (will_seed approach) hits a ceiling.

Submission: `submissions/straple/placer.py`. Eval command:
```
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all
```

Hard rule from README: no benchmark-name hardcoding (DQ). Adaptive logic by structure (`if num_macros > 500: ...`) is allowed.

**🚨 Critical bug discovered 2026-05-05**: submitted LNS placer (`submissions/straple/placer.py`) optimizes ONLY hard macros (`full[:n_hard] = best_pos`), leaves soft positions at initial. Soft macros ARE movable per PROBLEM.md: constraint #4 forbids changing **size**, not position; `macro_fixed=0` for all 894 soft on ibm01. We ignore 78% of decision variables. Top placers (MTK DreamPlace++) move all macros — confirmed via their video showing rainbow-colored full-canvas placement. **Priority 0 next session**: extend LNS pipeline to update `pos[:n_total]` not `pos[:n_hard]`. Expected gain: -5..-20% on AVG17 just from this fix. `gradient_demo.py` already implements this via `STRAPLE_DEMO_PLACE_ALL=1` (not in submission).

Key files: `readme/todo.md` (plan + journal, sections 7/8/9), `readme/results.md` (cycle #25 = submission, #26-27 are session notes). Update both when running new variants.

---

## MTK / Billy Lee — DreamPlace++ (#3 leaderboard, score 1.2818)

Featured submission from Partcl. Quote from author Pei-Yu (Billy) Lee, Ph.D., Technical Manager and Senior Staff SoC Physical Design Architect at MediaTek (leads floorplan design and integration for flagship SoCs; 10+ years at Cadence, Synopsys, Maxeda; specializes in advanced node placement, STA, parallel computing, AI-driven EDA methodologies):

> "The core challenge of this contest lies in the massive search space and the severe geometric constraints of heterogeneous macros. Standard continuous models naturally struggle here. Our 'DreamPlace++' approach succeeded by introducing **structural constraints** and a dynamic, **multi-phase spatial optimization strategy**. The raw parallel power of the GPU was crucial — it allowed us to efficiently evaluate these complex boundaries and **smoothly steer the analytical engine out of local traps into a highly optimized, zero-overlap state**."

### Что отсюда вытащили (важные insights)

1. **Vanilla DREAMPlace недостаточен** — даже #3 место это надстройка ("DreamPlace++"), не голый DREAMPlace
2. **Structural constraints** — pre-clustering макросов по netlist topology, anchor per cluster, members компактно вокруг
3. **Multi-phase optimization** — несколько фаз (coarse → refinement → legalize-aware finishing), не один проход
4. **GPU критичен** — для evaluating "complex boundaries" (т.е. для вычисления реального overlap state часто)
5. **Zero-overlap state из аналитического engine** — выходят почти-legal, не нужен disruptive legalize
6. **Place ALL macros (hard + soft)** — подтверждено через их видео (`readme/mtk_dreamplace_plus_ibm01.mp4`): rainbow-colored placement где hard И soft равномерно распределены по canvas

### Их recipe (реверс-инжиниринг видео)

- **iter=0..5**: ANCHOR_SOFT init — все макросы (hard + soft) сгруппированы в одну точку (cluster anchor + members)
- **iter=185, proxy=2.05**: cluster начинает разворачиваться, видны паттерны течения
- **iter=280, proxy=1.39**: macros формируют узоры/линии, заполняя canvas
- **iter=470, proxy=0.91**: dense packing, rainbow colors, layout похож на настоящий chip floorplan

### Action items (для нашего placer'а если идём по аналитическому пути)

- ✅ `gradient_demo.py::STRAPLE_DEMO_PLACE_ALL=1` уже работает — нужен port в submitted LNS pipeline
- Реализовать adaptive density_weight + adaptive gamma schedule (DREAMPlace overflow algorithm)
- Cluster-aware init: METIS / spectral / connected components → anchor per cluster → spawn members near anchor
- Pre-clustering как **initial structural prior** (key MTK feature)
- Multi-phase scheduling с stop conditions per phase
