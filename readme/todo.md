# Macro Placement Challenge 2026 — План и журнал

> Команда **Straple** (приватный репо `Straple/macro-place-challenge-2026`).
> Дедлайн: **2026-05-21**. Сегодня: **2026-05-04** (17 дней).

---

## 🚦 Текущий статус (после 25 циклов оптимизации)

| | Значение |
|---|---|
| **Submitted AVG17 proxy** | **1.4445** (commit `c758df2`) |
| **Submitted AVG4** (ibm01/10/14/17) | 1.3886 (отчётный для итераций) |
| **Место** | ~16 (между ArzunPD 1.4421 и Pragnay 1.4427 / Convex 1.4556) |
| **vs RePlAce baseline (1.4578)** | **-0.91%** — пробили ✅ |
| **vs Top-10 (1.4076)** | +2.62% — нужно ещё **-2.6%** |
| **vs Top-7 (Гран-при, 1.3479)** | +7.17% — нужно ещё **-6.7%** |
| Submission status | ☑ Заполнено в Google form, репо приватный + access судьям |
| Wall time --all (parallel 12 cores) | 32.5 мин (max ibm17 = 28.7 мин — в пределах 1ч лимита) |
| Overlaps | **0 на всех 17** ✅ |

**Sub'd как**: "Adaptive LNS with Multi-Operator Search" — Pure C++ ALNS placer (4 destroy/repair operators, adaptive weights, multi-start, refine passes).

---

## 1. Анализ leaderboard (актуальный, 2026-05-04)

| # | Команда | Score | Подход | Open-source? |
|---|---|---|---|---|
| 1 | Cezar (ReFine) | 1.2224 | RL refinement над analytical seed (закрыто, disputed) | ❌ |
| 2 | RoRa (RoomPlace) | 1.2723 | "Rip and re-place" + analytical seed (pending verify) | ❌ |
| 3 | MTK (DreamPlace++) | 1.2818 | DREAMPlace + tuning, GPU 37s | ❌ |
| 4 | Electric Beatle (ePlace-Lite) | 1.3253 | Adam stochastic descent + multi-start hyperparam sweep, GPU 2000s | ❌ |
| 5 | UToronto MOSAIC | 1.3323 | Custom gradient + smooth WL/density/cong + joint hard+soft, **CPU** 24 мин | ❌ |
| 6 | V5 (TierPlace) | 1.3382 | GPU multi-density-formulation + phased optimization, 850s | ❌ |
| 7 | Shoom (MultiDREAMPlace) | 1.3381 | Multi-start DREAMPlace + min-disp legalize + SA, 350s | ❌ |
| 8 | Archgen (AutoDMP++) | 1.3479 | NVlabs AutoDMP fork: multi-start + Bayesian hyperparam + bounded refinement | based on [AutoDMP](https://github.com/NVlabs/AutoDMP) |
| 9 | Beatel (ePlace-Lite older) | 1.3913 | GPU 155s | ❌ |
| 10 | UT Austin AS | 1.4076 | **Plain DREAMPlace**, 17s | based on [DREAMPlace](https://github.com/limbo018/DREAMPlace) |
| 11 | ByteDancer | 1.4151 | Incremental Coordinate Descent, 38min/bench | ❌ |
| 12 | vmallela | 1.4152 | Pure Python+numpy CD+LNS, single-thread, 12h total | ❌ |
| 13 | TAISPlAce | 1.4321 | **ALNS + Thompson Sampling** (bandit operator selection) | ❌ |
| 14 | ArzunPD | 1.4421 | HyperPlace SA+LNS | ❌ |
| 15 | Pragnay | 1.4427 | SweepingBellPlacement | ❌ |
| **16** | **Мы (Straple ALNS)** | **1.4445** | **Pure C++ ALNS + multi-start + refine** | private |
| 17 | Convex Optimization (UWaterloo) | 1.4556 | — | — |
| — | RePlAce baseline | 1.4578 | — | TILOS published |

### Главные паттерны топ-10

1. **DREAMPlace или ePlace-style analytical placer** — фундамент 80% top-10. Это **наш главный пробел**.
2. **Recipe топа**: `analytical seed → min-displacement legalize → multi-start hyperparam sweep → SA/LNS polish`
3. **GPU помогает но не критично**: MOSAIC #5 = CPU 24 мин, мы тоже могли бы 24 мин.
4. **Чистый LNS застревает на ~1.42-1.45** (TAISPlAce, ArzunPD, vmallela, мы) — без gradient seed дальше не пробьёшь.
5. **Soft macro joint optimization** — MOSAIC явно делает.

### DQ-ловушки в которые попали другие
- `Mike Gao`: DREAMPlace silent fail → 47-189 overlaps на bench
- `BakaBobo`: import error в чистой среде
- `vmallela`: self-reported 1.1172 → verified 1.4152 (-27% на их железе)

**Урок**: тестировать в чистой среде, не полагаться на оптимизированные локальные настройки.

---

## 2. Что мы попробовали (25 циклов)

### Сработало (chronological)
| Cycle | Что | Эффект |
|---|---|---|
| #1 | Congestion-aware destroy в LNS | -0.11% |
| #3 | Adaptive LNS + чередование ops | -0.30% |
| #4 | **Congestion-aware repair** (spiral search threshold по cong-grid) | -0.89% |
| #7-8 | Multi-start N=3 → N=5 для больших | -1.11% |
| #10-11 | LNS budget 60 → 100 → 150 outer iters | -1.32% |
| **#12** | **🚨 BUG FIX: evaluator 1:1:1 → 1:0.5:0.5** | -1.48% (критично) |
| #13 | Skip SA для больших (n>=300) — SA там вреден | -1.99% |
| #14 | Vectorize `_smooth_hpwl` — 37× ускорение analytical | (для будущего) |
| #19 | 2-opt swap operator + LNS budget 8000 | -3.85% |
| #20 | ALNS adaptive weights + cluster destroy | **-4.93%** (пробили RePlAce) |
| #21 | 3 starts × 25000 LNS | -5.28% (ниже Top-10 на AVG4) |
| #22 | Shake-up + LNS 50000 | -5.85% |
| #24 | 3 refine passes intensification | **-6.42%** AVG4 = 1.3886 |
| #25 (--all) | Full 17 benchmarks с финальным конфигом | **AVG17 = 1.4445** |
| #26 | Parallel multi-start infrastructure (mp.Pool) | (готово, не использовано в submission) |

### Не сработало / откатили
- **Analytical seed (DREAMPlace-style)** — наш `analytical_seed.py` после legalize+LNS даёт **хуже** original initial. Initial benchmark.macro_positions уже хорош, наш analytical сходит в worse local minimum. (cycles #15-#18)
- **Adaptive SA budget** — больше SA = хуже proxy (SA optimize wrong objective HPWL). (cycle #9)
- **No alternation random/congested** — exploration важна. (cycle #2 регрессия +0.46%)
- **Tighter cong threshold P30** — слишком много fallback. (cycle #5)
- **Stronger swap (k=destroy)** — disruptive. (cycle #22)
- **Proximity-based swap** — хуже net-bias. (cycle #22)
- **Single start × 60000** — теряет diversity на ibm10. (cycle #20)
- **8 random-seed multi-start** — все сходят в один attractor (нужна structural diversity, не RNG)

---

## 3. Цели и план до дедлайна (17 дней)

### Цели по уровням ambition

| Уровень | AVG17 | Δ от 1.4445 | Что даёт |
|---|---|---|---|
| 🎯 Текущее | 1.4445 | 0% | ~16 место, выше RePlAce |
| 🎯 +1 место (Pragnay) | 1.4427 | -0.13% | trivial polishing |
| 🎯 Топ-13 (TAISPlAce) | 1.4321 | -0.86% | улучшение LNS / больше budget |
| 🎯 **Топ-10** | 1.4076 | **-2.55%** | **требует gradient seed** |
| 🎯 Топ-8 (Archgen) | 1.3479 | -6.69% | DREAMPlace-class + multi-start |
| 🎯 Гран-при ($20K) | ~1.30 | -10% | топ DREAMPlace + GPU + ORFS работает |

### План A: incremental wins (1-2 дня) — реалистичный для топ-13

1. **Parallel multi-start с N=16** на eval-машине (16 cores) с diverse seeds
   - Random RNG не работает (узнали). Нужна **structural diversity**:
     - Perturbed initial: `initial_pos + noise(σ=0.05·canvas)` для каждого старта
     - Или несколько hyperparameter configs (разные shake_threshold, destroy_size, etc.)
   - **Ожидание**: -0.5..-1.5% AVG17 → **1.42-1.43**
   - **Effort**: уже есть `STRAPLE_PARALLEL_STARTS` env var. Добавить perturbation в `_run_one_start`.

2. **Per-bench profiling на ibm06-09** где мы хуже RePlAce — найти что не так
   - На ibm06, ibm07, ibm08, ibm09 у нас +2..+5% от RePlAce. Возможно tuning per-size-class.

3. **Final --all run в чистой среде** для verification до сабмита

### План B: gradient seed (2-4 дня) — для топ-10

1. **Интегрировать [DREAMPlace](https://github.com/limbo018/DREAMPlace)** (BSD-3) как один из multi-start seeds
   - Subprocess wrapper или Python embedding
   - DREAMPlace requires: PyTorch, SCIP optional, CPU OR GPU build
   - Output → наш min-displacement legalize → LNS polish
   - **Ожидание**: -2..-5% AVG17 → **1.37-1.42**
   - **Effort**: 2-3 дня (build, integration, testing, dependency management)
   - **Риск**: сложности с deps в чистой среде (как у Mike Gao — silent fail)

2. **Альтернатива: фикс собственного analytical_seed.py**
   - Главная проблема: после legalize теряется structure
   - Fix: совместить analytical с min-displacement legalize что **не разрушает** layout
   - Может работать без external deps
   - **Effort**: 1-2 дня
   - **Ожидание**: -1..-3% если получится

### План C: реальный DREAMPlace + ORFS Tier 2 (5-7 дней) — для Гран-при

Только если получится топ-7 по proxy. Требует ORFS docker setup, NG45 designs.

---

## 4. Критичные риски до сабмита

- [ ] **Repo verification mismatch** (как у vmallela -27%). Мы детерминируемся, должно быть consistent. Но hardware у судей: 16-core EPYC vs наш 12-core M-series. Может быть minor differences.
- [ ] **Final pre-submit run в чистой среде** (новый clone в /tmp, чистый uv sync) — to catch dependency issues.
- [ ] **bounds-violations** на 15 из 17 бенчей (макросы вне canvas). Это **НЕ DQ** по checklist (только overlaps), но потенциальный риск если судьи усилят validation. Стоит закостыливать.

---

## 5. Полезные ссылки

- **DREAMPlace**: https://github.com/limbo018/DREAMPlace (BSD-3, GPU/CPU)
- **AutoDMP** (надстройка над DREAMPlace с MOBO): https://github.com/NVlabs/AutoDMP
- **ePlace 2014 paper**: https://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
- **Submission status**: monitor https://github.com/partcleda/partcl-macro-place-challenge — leaderboard updates через PRs от willpartcl
- **TILOS MacroPlacement**: https://github.com/TILOS-AI-Institute/MacroPlacement — benchmarks + RePlAce baseline

---

## 6. Решение что делать сейчас

**Рекомендация**: начать с **Плана A** (parallel + perturbed seeds) — quick win, может дать -0.5..-1.5% за 1-2 часа. Потом решить идти ли в План B (DREAMPlace).

Подробный журнал прогонов с per-bench разбивкой → [results.md](results.md).
