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

---

## 7. Сессия 2026-05-04 (попытка плана C, прервано)

### Что выяснили (важно для следующей сессии)

**1. analytical_seed.py не работает как seed — surrogate loss не отражает proxy.**

Прогнали `submissions/straple/analytical_seed.py` standalone на ibm17 (initial=1.7392):
| Конфиг | proxy после analytical | Δ от initial |
|---|---|---|
| 200 шагов lr=0.3 lambda=50000 (placer.py default) | 1.8816 | **+8.2%** |
| 300 шагов cold_start [(0,0),(0.2,100),(0.5,5000),(1,50000)] | 1.9440 | +11.8% |
| 20 шагов lr=0.5 lambda=1000 | 1.8335 | +5.4% |
| 50 шагов lr=0.3 lambda=5000 | 1.8396 | +5.8% |
| 50 шагов lr=0.1 lambda=1000 | 1.7850 | +2.6% (минимальный, всё равно хуже) |

**Корневая причина**: smooth HPWL surrogate (LSE с gamma~3.6) и Gaussian-bell density penalty НЕ КОРРЕЛИРУЮТ с реальным proxy_cost. Adam успешно минимизирует loss (8M → 4M), но реальный proxy растёт. Density goes 0.945→1.067, congestion goes 2.426→2.708. Конгестию вообще не моделируем — а это 66% метрики.

**Что делать чтобы починить (НЕ ДЕЛАЛИ — это 1-2 дня):**
- Добавить congestion-aware loss (bell-curve smoothing edge contributions to grid cells)
- Tune gamma/lambda через адаптивный update (как DREAMPlace's overflow-based subgradient ascent)
- Использовать Nesterov вместо Adam (DREAMPlace recipe)
- Возможно, проверить нет ли багов в density_penalty (target_util scale, normalization)

**2. Plan A: perturbed multi-start не помог в наивной форме.**

Изначально заменили start_idx=1,2 (same-seed multi-start) на perturbed σ=0.03, 0.05 of canvas. **Регрессия**: AVG4 1.3886 → **1.3962** (+0.55%). На ibm10 ушли с 1.2282 на 1.2588.

**Урок**: same-seed multi-start (3 starts с seeds 42,43,44) УЖЕ даёт diversity, особенно на ibm10. Менять её на perturbed = терять рабочую находку. Refactor на additive (3 orig + N perturbed extras) сделан в коде, **не измерен** (тест прерван).

**3. DREAMPlace интеграция через Docker возможна, но многодневная.**

- Склонили `external/DREAMPlace/` (BSD-3, github.com/limbo018/DREAMPlace)
- Запустили colima x86 VM на Mac arm64
- Скачали `limbo018/dreamplace:cuda` (20.2 GB) — оказалось это **только окружение**, DREAMPlace надо билдить внутри (~1 час компиляции из-за многих submodules: Limbo, Flute, OpenTimer, CUB, munkres-cpp)
- Без CUDA на Mac — только CPU mode, через Rosetta x86 — медленно
- TILOS уже имеет `external/MacroPlacement/CodeElements/FormatTranslators/src/ProtobufToLEFDEF.py` — конвертер protobuf → LEF/DEF (нужен для feed в DREAMPlace). Требует `tqdm` (не в наших deps).

### Изменения в коде (uncommitted, в submissions/straple/placer.py)

**Backward-compatible env var инфраструктура** (поведение по дефолту НЕ изменилось):
- `STRAPLE_PERTURB_EXTRA_STARTS` (default 0) — добавить N perturbed starts поверх 3 same-seed (только для n_movable >= 300)
- `STRAPLE_PERTURB_SCALES` — comma-sep σ factors для perturbed (default `0.05,0.10,0.15,0.20`)
- `STRAPLE_LNS_OUTER_CAP` (default 50000) — cap LNS outer iters
- `STRAPLE_LNS_OUTER_FACTOR` (default 60.0) — multiplier для adaptive_outer
- `STRAPLE_NUM_STARTS` — override num_orig_starts (был раньше)
- `STRAPLE_ANALYTICAL_PRESET=cold_start` — lambda/gamma schedule (мёртвый код, analytical не работает)

**Логика**: для analytical seed теперь `legalize_min_displacement(500)` + `legalize()` safety net вместо disruptive `legalize()`. Тоже мёртвый код пока.

`_perturb_initial(args, start_idx)` — Гауссов шум σ_factor * min(canvas_w, canvas_h), clamp в canvas, не двигает non-movable. Семя RNG = `self.seed + start_idx*1000 + 7`.

**Решение**: можно либо закоммитить инфраструктуру (она безопасна — поведение не меняет), либо `git checkout submissions/straple/placer.py` чтобы откатить.

### Рекомендованный план для следующей сессии

**Краткий путь к улучшению (1 день)**:
1. Закоммитить env-var инфраструктуру (`git add submissions/straple/placer.py`)
2. Тестировать `STRAPLE_LNS_OUTER_CAP=100000 STRAPLE_LNS_OUTER_FACTOR=120` — больше LNS итераций может дать -0.5..-1%. Wall ibm17 вырастет до ~50 мин (в 1ч лимита).
3. Если ок — measure full --all для AVG17.
4. Затем `STRAPLE_PERTURB_EXTRA_STARTS=2 STRAPLE_PERTURB_SCALES=0.05,0.10` — additive perturbation. Измерить best of 5 vs best of 3.

**Длинный путь (3-5 дней)**:
1. Билд DREAMPlace внутри docker контейнера (image уже скачан)
2. `pip install tqdm`, прогнать `ProtobufToLEFDEF.py` на ibm01 → LEF/DEF
3. Скормить в DREAMPlace, получить positions
4. Конвертировать обратно (DEF parsing + matching macro names)
5. Wire как один из multi-start seeds
6. Tune hyperparams (target_density, density_weight, gamma)

**Альтернатива (1-2 дня) — починить свой analytical**:
- Добавить congestion-aware loss
- Adaptive density_weight (как DREAMPlace's overflow-based)
- Nesterov instead of Adam
- Тест: должен ХОТЯ БЫ матчить initial proxy на ibm17, иначе бесполезно

### Инсайт от MTK (#3, DreamPlace++, score 1.2818)

Из публичного комментария Billy Lee (MediaTek):

> "Standard continuous models naturally struggle here. Our 'DreamPlace++' approach
> succeeded by introducing **structural constraints** and a **dynamic, multi-phase
> spatial optimization strategy**. The raw parallel power of the GPU was crucial —
> it allowed us to efficiently evaluate these complex boundaries and **smoothly
> steer the analytical engine out of local traps into a highly optimized,
> zero-overlap state**."

**Что отсюда можно вытащить:**

1. **Vanilla DREAMPlace недостаточен** — даже #3 место строилось НЕ на голом DREAMPlace, а на надстройке поверх. Подтверждает наше наблюдение, что наш analytical_seed.py с базовыми компонентами (smooth WL + density bell) обречён.

2. **Multi-phase optimization** — не один проход gradient → legalize, а несколько фаз с разными целями. Возможные интерпретации:
   - Phase 1: coarse global placement (пусть с overlap'ами)
   - Phase 2: пересборка density bins, узкие constraint'ы → refinement
   - Phase 3: legalize-aware finishing (макросы уже почти non-overlapping, нужно только snap)
   - Аналог DREAMPlace's overflow-based subgradient с stop_overflow=0.07: lambda растёт пока overflow > порог.

3. **Structural constraints** — НЕ просто WL + density. Что это может быть:
   - **Pre-clustering** макросов по netlist topology (mincut / spectral) → жёстко держим cluster в одном регионе
   - **Symmetry constraints** для регулярных IP (memory arrays)
   - **Border-affinity** — макросы с большим количеством IO-пинов прижимать к краю
   - **Region partitioning** — заранее разбить canvas на несколько регионов и распределить макросы по нагрузке

4. **"Smoothly steer out of local traps"** — escape от bad local minima ВНУТРИ analytical engine, не через рестарты:
   - Адаптивный density_weight (растёт когда overflow высокий, как DREAMPlace overflow algo)
   - Адаптивная gamma (decreasing schedule в зависимости от overflow, как у них)
   - Возможно momentum → Nesterov, чтобы не залипать в плоских регионах

5. **"Zero-overlap state"** из аналитического выхода — они выходят УЖЕ почти-legal. Значит multi-phase сходится к решению с минимумом overlap'ов, и финальный legalize_min_displacement не разрушает layout. Это решает нашу проблему "after legalize структура теряется".

**Action items для нашего placer (если идём по аналитическому пути):**
- Реализовать adaptive density_weight + adaptive gamma schedule (DREAMPlace overflow algorithm) в `analytical_seed.py`
- Добавить congestion-aware loss term
- Перед analytical: hierarchical clustering макросов (METIS / spectral) → использовать как **initial structural prior**
- Multi-phase scheduling с stop conditions per phase (overflow target, gradient norm threshold)
- На каждой фазе пересобирать density bins (увеличивать разрешение к концу)

---

## 8. Сессия 2026-05-04 (визуализация + gradient demo + GPU plans)

### Что сделано

**Visualizer (`submissions/straple/visualizer.py`)** — формат определяется по расширению output:
- `.mp4` — видео через ffmpeg (libx264)
- `.gif` — анимированный GIF
- `.html` — интерактивный (canvas + JS): hover на макросы → tooltip с именем, ←/→ переключают кадры, space play/pause, FPS configurable
- 2×2 layout: placement (canvas) / density heatmap / congestion heatmap / score history

**Force-directed demo (`submissions/straple/force_demo.py`)** — `STRAPLE_DEMO=force`:
- Random или center init
- Repulsion (overlap-axis push) + spring (по нетам) + spread (long-range Coulomb)
- Cooling damping (0.85→0.99), velocity cap decay
- Random ops: teleport / swap / pull / shake / scatter_cluster (каждые N шагов)
- Bounce от стен (bounce_factor=-0.7)
- Score history PNG (`STRAPLE_DEMO_SCORE_PNG=path.png`)

**Gradient demo (`submissions/straple/gradient_demo.py`)** — `STRAPLE_DEMO=gradient`, **наш аналог DREAMPlace для демо**:
- Adam optimizer
- Loss = WL_smooth (LSE) + λ_d × density_bell + λ_o × overlap_term
- Adaptive density_weight (растёт при overflow > stop_overflow)
- Cooling gamma (γ_start=1.5 × base → γ_end=0.3 × base)
- Cooling LR (lr × 0.05 to end)
- 6 overlap forms: linear / quadratic / cubic / huber / **gauss** / **gauss_overlap** / **coulomb**
- Optional Lagrangian update (`λ_o += ρ × overlap`)
- Optional finishing C++ legalize (`STRAPLE_DEMO_FINISH_LEGALIZE=1`)
- Score history с tracking loss components

**Best result (ibm01, 600 шагов, ~100с CPU)**:
- `OVERLAP_FORM=gauss_overlap` + `FINISH_LEGALIZE=0` → proxy=1.5800, **VALID 0 overlaps**, чисто PyTorch
- `OVERLAP_W=0` + `FINISH_LEGALIZE=1` → proxy=**1.4288**, 0 overlaps (best metrics)
- vs наш ALNS submission: 1.0584 (то есть demo гораздо хуже, но это только seed phase)

### План когда будет GPU (T4 + 16 vCPU + 64GB RAM, 100GB disk)

**Disk requirements**:
- DREAMPlace Docker image: 20 GB
- Native install + CUDA + PyTorch: ~10 GB
- Наш проект: 2 GB
- Builds + outputs + checkpoints: 50 GB
- **Минимум 50 GB, комфортно 100 GB**

**Phase 1: GPU-port текущего gradient_demo (1 день)**
1. Перевести все torch tensors на `device='cuda'` — 30 минут
2. Multi-start через batch dimension `[K, n, 2]` — добавить outer dim во все ops
3. Profile + tune для T4 (8.1 TFLOPS, 16GB)
4. Hyperparam sweep: 64-128 starts с разными (lr, λ_d_max, target_util, gamma_frac)
5. Best of K → seed для LNS polish

Ожидание GPU speedup:
- ibm01: 10× (мала задача)
- ibm17: ~40× (n=760, n²=580K pairs)
- Multi-start K=16 на T4: ~5 секунд на бенч (vs 60c CPU)
- 17 бенчей × 8 multi-start: **~2 минуты** на T4 vs ~2 часа CPU

**Phase 2: Congestion-aware loss (1-2 дня)**
- Реализовать smooth bbox через LSE на пинах нета (есть для HPWL)
- Routing demand grid: каждая ячейка получает sum bbox-ов проходящих
- Smooth top-10% via soft-max → дифференцируема
- Это закроет 66% метрики которую gradient сейчас не видит
- Ожидание: -10..-20% на C компоненту → proxy ниже 1.0 на ibm01

**Phase 3: DREAMPlace integration (3-5 дней)**
- Билд DREAMPlace внутри Docker (image скачан, ~1 час компиляции)
- Конвертер benchmark → LEF/DEF (есть `ProtobufToLEFDEF.py` от TILOS, нужен tqdm)
- Wrapper в нашем placer'е: subprocess → run DREAMPlace → parse output → multi-start seed
- Multi-start hyperparam sweep на T4 (target_density, density_weight, gamma — как Archgen)
- LNS polish после DREAMPlace seed

**Phase 4: Hybrid gradient + LNS (1 день)**
- Gradient seed (наш или DREAMPlace) → 0 overlaps
- ALNS polish на seed (наш существующий)
- Gradient даёт хороший basin, LNS делает локальную оптимизацию

### Ожидаемые улучшения по AVG17

| Подход | AVG17 | Ранг |
|---|---|---|
| Текущий submission | 1.4445 | ~16 |
| GPU-port + multi-start (Phase 1) | ~1.42 | ~13-14 |
| + Congestion-aware loss (Phase 2) | ~1.36-1.40 | ~9-12 |
| + DREAMPlace integration (Phase 3) | ~1.34-1.38 | ~7-10 |
| + Full hybrid (Phase 4) | ~1.30-1.34 | топ-5..7 |

**Реалистично топ-7 квалификация на Гран-при** (≤1.3479) при выполнении Phase 1-3 за ~1 неделю на T4.

### Optimizer / loss tuning candidates (когда будет GPU)

**Optimizer**:
- Nesterov accelerated gradient (DREAMPlace's choice)
- AdamW с weight decay
- Lookahead optimizer (slow + fast updates)

**Schedule**:
- DREAMPlace overflow-based gamma update: `coef = 10^((overflow - 0.1) × 20/9 - 1)`
- Density quadratic penalty (auto-Lagrangian)
- Reduce-on-plateau LR

**Loss components**:
- Congestion-aware (см. Phase 2)
- Per-region density (для structural constraints как у MTK)
- Pin-side affinity (макросы с >K IO-пинов прижимаются к краям)

### Параметры текущего gradient_demo (env vars)

```bash
STRAPLE_DEMO=gradient                    # mode: force | gradient
STRAPLE_DEMO_INIT=center                 # center | random
STRAPLE_DEMO_ITERS=600                   # шагов градиента
STRAPLE_DEMO_TIME_BUDGET=120             # ИЛИ time budget в секундах (вместо ITERS)
STRAPLE_DEMO_LR=0.3
STRAPLE_DEMO_LR_END_FACTOR=0.05          # LR cooling: 0.3 → 0.015
STRAPLE_DEMO_LAMBDA_START=0.01           # density_weight start
STRAPLE_DEMO_LAMBDA_GROWTH=1.05          # multiplicative growth per step
STRAPLE_DEMO_LAMBDA_MAX=200
STRAPLE_DEMO_GAMMA_FRAC=0.05             # WL_smooth gamma = canvas_min × frac
STRAPLE_DEMO_GAMMA_START=1.5             # cooling start factor
STRAPLE_DEMO_GAMMA_END=0.3               # cooling end factor
STRAPLE_DEMO_TARGET_UTIL=0.3             # density target
STRAPLE_DEMO_OVERLAP_W=20                # overlap penalty weight
STRAPLE_DEMO_OVERLAP_W_GROWTH=1.005
STRAPLE_DEMO_OVERLAP_W_MAX=2000
STRAPLE_DEMO_OVERLAP_FORM=gauss_overlap  # linear|quadratic|cubic|huber|gauss|gauss_overlap|coulomb
STRAPLE_DEMO_LAGRANGIAN=0                # use λ += ρ × overlap (additive) instead of multiplicative
STRAPLE_DEMO_RHO=5.0                     # Lagrangian rho
STRAPLE_DEMO_FINISH_LEGALIZE=0           # 1 = run C++ legalize at end
# Visualization:
STRAPLE_VIS_VIDEO=vis/output.html        # path with .html / .mp4 / .gif extension
STRAPLE_VIS_MAX_FRAMES=240
STRAPLE_VIS_FPS=24
STRAPLE_VIS_INTERVAL=100                 # for LNS mode
STRAPLE_DEMO_SCORE_PNG=vis/score.png     # static PNG of score history
STRAPLE_DEMO_SCORE_SAMPLE_S=1.0          # sampling interval for score
```

### Файлы (uncommitted)

- `submissions/straple/visualizer.py` — HTML/MP4/GIF visualizer
- `submissions/straple/force_demo.py` — physics demo
- `submissions/straple/gradient_demo.py` — gradient (DREAMPlace-style) demo
- Изменения в `submissions/straple/placer.py` — env-var dispatch для demo modes
- `vis/*.html`, `vis/*.png`, `vis/*.mp4` — артефакты экспериментов

Все три demo файла — **только для визуализации**, не используются submitted placer'ом. Default behavior `evaluate submissions/straple/placer.py` без env vars = baseline ALNS pipeline (1.4445 AVG17).

---

## 9. Сессия 2026-05-04 → 2026-05-05 (gradient deep dive + критический инсайт про soft macros)

### 🚨 ГЛАВНОЕ ОТКРЫТИЕ: soft macros можно (и нужно) двигать

Перечитали [PROBLEM.md](PROBLEM.md) — спецификация ясно говорит:
- Размещение `P ∈ ℝ^(n×2)` где **n = num_macros = ВСЕ макросы (hard + soft)**
- Constraint #4 запрещает менять **размеры** soft, не позиции
- Только `macro_fixed[i] = 1` блокирует движение
- Для ibm01 все 1140 макросов имеют `fixed=0` (никто не зафиксирован)

**Подтверждение** в `macro_place/objective.py::_set_placement` (lines 200-218): функция явно вызывает `node.set_pos(x, y)` и для hard, и для soft макросов из переданного `placement[num_macros, 2]`. Если мы передаём только обновлённые hard позиции — soft остаются на исходных. Это **наш bug**, а не constraint.

**Наш текущий submitted placer** (LNS-based) обновляет только `pos[:n_hard]`, оставляет soft на initial positions. **Мы 894 из 1140 макросов не оптимизировали** (78% переменных). Это критический пробел.

### Подтверждение через MTK видео

Пользователь поделился MP4 от MTK (3 место, 1.2818): кадры iter=5 → iter=185 → iter=470 показывают:
- **iter=5**: ВСЕ макросы (включая мелкие salt-зелёные = soft) сгруппированы в одну точку (anchor cluster)
- **iter=470**: финал, **rainbow colors** — hard и soft равномерно заполняют весь canvas, разные цвета = разные кластеры
- Промежуточные кадры — gradient unfolding кластеров

**Их recipe (из видео + публичного коммента)**:
1. **ANCHOR_SOFT init**: pre-clustering макросов по netlist topology, в каждом кластере anchor + members компактно вокруг
2. **Optimize ALL macros** (1140 для ibm01) — gradient двигает hard И soft
3. **Multi-phase**: spreading (high gamma, low lambda) → settling (low gamma, high lambda)
4. **Cluster-colored visualization** — видно сохранение структуры

### Реализованная инфраструктура

#### `submissions/straple/gradient_demo.py` (новое)

Полноценный gradient placer "a-la DREAMPlace":
- **Adam** optimizer с adaptive density_weight (растёт когда overflow > stop_overflow)
- **Cooling gamma** в WL_smooth (LSE)
- **Cooling LR** + опциональный `ReduceLROnPlateau` scheduler
- **Multiple overlap forms** (env `STRAPLE_DEMO_OVERLAP_FORM`):
  - `linear`: `Σ overlap_area` (sharp gradient, дёргается)
  - `quadratic`: `Σ overlap_area²` (smooth, не закрывает 0)
  - `gauss`: `Σ exp(-(dx²/σ_x² + dy²/σ_y²))` (бесконечно гладко, прилипает к стенам)
  - **`gauss_overlap`**: gauss + 5× boost по реальной overlap area (✅ best для visual + 0 overlaps)
  - `coulomb`: `Σ 1/(dist² + soft²)` (сильный push на близких)
  - `huber`: гибрид quadratic/linear
- **Lagrangian update**: `λ_o += ρ × overlap` (additive, теоретически convergent — но требует тонкой настройки ρ, иначе расходится)
- **Plateau-triggered ops**: после N steps без улучшения loss срабатывает random op (teleport / swap / shake / scatter_cluster) — заимствуется из force_demo
- **Multi-restart on plateau**: альтернатива ops — полный re-init с нового random seed, tracking global best across restarts
- **Congestion-aware loss** (env `STRAPLE_DEMO_CONG_W`): smooth bbox per net (LSE) → cell demand → top-10% mean. **Эмпирически не помог** — surrogate не коррелирует с TILOS L-shape routing model.
- **`STRAPLE_DEMO_PLACE_ALL=1`** (default ON): оптимизирует ВСЕ макросы, не только hard. Overlap penalty только между hard pairs (soft могут пересекаться). C++ legalize применяется только к первым n_hard.

#### `submissions/straple/visualizer.py` (новое)

Универсальный visualizer 2×2 layout:
- Формат определяется по расширению output: `.mp4` (ffmpeg), `.gif` (ffmpeg), `.html` (canvas + JS)
- **HTML mode**: hover на макрос → tooltip с именем/координатами, ←/→ переключают кадры, space play/pause, slider seek
- 4 панели: placement / density / congestion / score history с loss components
- Score chart показывает real cost (proxy/WL/D/C) И gradient loss (WL_smooth + λ_d×density + λ_o×overlap_term)
- Поддерживает variable pos size (как `[n_hard]`, так и `[n_total]`)

#### `submissions/straple/force_demo.py` (новое)

Force-directed физический демо (для сравнения с gradient):
- Random/center init
- Repulsion (overlap) + spring (нет) + spread (Coulomb)
- Random ops каждые N шагов: teleport / swap / pull / shake / scatter_cluster
- Bounce от стен с damping
- Cooling damping schedule

### Измеренные результаты (ibm01)

| Конфигурация | proxy | WL | D | C | overlaps | comment |
|---|---|---|---|---|---|---|
| Submitted ALNS (hard only) | 1.0584 | 0.072 | 0.840 | 1.133 | 0 ✅ | submission baseline |
| Force physics demo | 1.5582 | 0.133 | 1.046 | 1.804 | 0 ✅ | без attraction по нетам |
| Gradient (hard only) + legalize | 1.4288 | 0.098 | 1.026 | 1.635 | 0 ✅ | до открытия про soft |
| Gradient + Lagrangian (rho=0.05) | 1.6991 | 0.111 | 1.184 | 2.045 | 0 ✅ | без legalize, valid pure-gradient |
| Gradient + gauss_overlap | 1.5800 | 0.133 | 1.036 | 1.858 | 0 ✅ | gauss кастомный — visually smooth |
| **Gradient PLACE_ALL=1 + legalize** | **1.5288** | **0.142** | **0.774** | 2.000 | 0 ✅ | **+all macros, density -28%** |
| Gradient + restart (best of 23) | 1.5359 | 0.134 | 1.040 | 1.763 | 0 ✅ | multi-restart with legalize-each |
| MTK DreamPlace++ (видео) | ~0.91 | — | — | — | 0 ✅ | их submission, для референса |

**Ключевое наблюдение**: даже наивное PLACE_ALL=1 без других тюнов даёт density 0.774 vs 1.04 (-28%). Доступ к 894 дополнительным переменным даёт огромный manoeuvring room. Trade-off: WL и cong подросли, потому что soft пины тоже распределились — bbox-ы нетов стали шире.

### План для следующей сессии (важнейший)

**Приоритет 0: применить PLACE_ALL=1 в submitted placer** (не только в demo)
- Текущий `placer.py::place()` обновляет `full[:n_hard]` — заменить на `full[:n_total]`
- Расширить C++ pipeline (`PlacerState`) на soft макросы:
  - Overlap check только между hard
  - SA / LNS / refine двигают и hard, и soft
  - Edges включают пары с soft endpoints (через `_extract_edges_full`)
- **Ожидание**: -5..-20% на AVG17 ТОЛЬКО за счёт исправления этого пробела. Возможно сразу пробьёт топ-10.

**Приоритет 1: cluster-based init (anchor-soft)**
- Реализовать pre-clustering: METIS / spectral / connected components на edge graph
- Anchor per cluster: max-degree macro
- Initial spawn: anchors распределяются по canvas (k-means на target positions), members компактно вокруг своего anchor
- Это копирует MTK ANCHOR_SOFT → должно дать ещё -10..-20%

**Приоритет 2: GPU port + multi-start hyperparam sweep**
- Tensor ops уже vectorized — `pos.cuda()` минимум кода
- Multi-start через batch dim `[K, n, 2]`
- 16-64 starts на T4 → best by proxy
- Hyperparam sweep over (lr, λ_d_max, target_util, gamma_start, gamma_end, overlap_w)

**Приоритет 3: real DREAMPlace integration**
- Docker image скачан (limbo018/dreamplace:cuda 20GB)
- Build inside ~1 час
- ProtobufToLEFDEF.py для конвертации benchmark
- Subprocess wrapper в placer'е

### Уроки

1. **Читать спеку внимательно**. Мы потратили дни на оптимизацию hard-only pipeline, не заметив что soft тоже movable. Один внимательный read PROBLEM.md сэкономил бы недели.
2. **Смотреть видео топов**. MTK MP4 показал ANCHOR_SOFT init и multi-cluster colors за 30 секунд — то, что не смогли вычитать из публичного коммента.
3. **Surrogate ≠ true objective**. Naive smooth WL/density дают gradient в неправильную сторону для real proxy. DREAMPlace тратит 5 лет на правильные surrogate'ы (electric potential, pinrudy, etc.) — это не быстрая задача.
4. **Visualization >>> логи**. HTML с цветными кластерами + анимация показывает что происходит лучше чем тысячи логов. Score chart с loss components показывает почему gradient идёт куда идёт.
5. **Pure-PyTorch gradient вполне реализуем**. Наш `gradient_demo.py` показывает что без DREAMPlace мы можем построить рабочий analytical placer (хоть и хуже DREAMPlace в качестве). Достаточно для seed → LNS polish.

### Текущие env vars (полный список для gradient_demo)

```bash
# Mode
STRAPLE_DEMO=gradient                    # gradient | force
STRAPLE_DEMO_PLACE_ALL=1                 # 1 = optimize ALL macros, 0 = hard only
STRAPLE_DEMO_INIT=center                 # center | random
STRAPLE_DEMO_SPAWN_JITTER=0.005

# Time/iters
STRAPLE_DEMO_ITERS=600                   # ИЛИ
STRAPLE_DEMO_TIME_BUDGET=120             # секунд
STRAPLE_DEMO_SEED=42

# Optimizer
STRAPLE_DEMO_LR=0.3
STRAPLE_DEMO_LR_END_FACTOR=0.05          # cosine cooling end
STRAPLE_DEMO_LR_PLATEAU=1                # use ReduceLROnPlateau
STRAPLE_DEMO_PLATEAU_FACTOR=0.5
STRAPLE_DEMO_PLATEAU_PATIENCE=80
STRAPLE_DEMO_PLATEAU_MIN_LR=0.05

# Loss components
STRAPLE_DEMO_GAMMA_FRAC=0.05             # gamma base = canvas_min × frac
STRAPLE_DEMO_GAMMA_START=1.5
STRAPLE_DEMO_GAMMA_END=0.3
STRAPLE_DEMO_LAMBDA_START=0.05           # density weight start
STRAPLE_DEMO_LAMBDA_GROWTH=1.04
STRAPLE_DEMO_LAMBDA_MAX=200
STRAPLE_DEMO_TARGET_UTIL=0.4
STRAPLE_DEMO_STOP_OVERFLOW=0.07

# Overlap penalty
STRAPLE_DEMO_OVERLAP_W=15
STRAPLE_DEMO_OVERLAP_W_GROWTH=1.005
STRAPLE_DEMO_OVERLAP_W_MAX=2000
STRAPLE_DEMO_OVERLAP_FORM=gauss_overlap  # linear|quadratic|cubic|huber|gauss|gauss_overlap|coulomb
STRAPLE_DEMO_LAGRANGIAN=0                # use additive λ update
STRAPLE_DEMO_RHO=0.05                    # Lagrangian rho

# Congestion-aware loss (experimental, not effective on ibm01)
STRAPLE_DEMO_CONG_W=0
STRAPLE_DEMO_CONG_TOP_PCT=0.1

# Plateau-triggered ops
STRAPLE_DEMO_OP_ON_PLATEAU=0
STRAPLE_DEMO_OP_PATIENCE=150
STRAPLE_DEMO_OP_K=20
STRAPLE_DEMO_OP_WARMUP=0.1               # progress fraction
STRAPLE_DEMO_OP_MAX_PROGRESS=0.7
STRAPLE_DEMO_OPS=teleport,swap,shake,scatter_cluster
STRAPLE_DEMO_OP_EVERY=0                  # period; 0 = plateau-only

# Restart instead of ops
STRAPLE_DEMO_RESTART_ON_PLATEAU=0

# Finishing
STRAPLE_DEMO_FINISH_LEGALIZE=1           # C++ legalize at end (only first n_hard)

# Visualization
STRAPLE_VIS_VIDEO=vis/output.html        # .html / .mp4 / .gif
STRAPLE_VIS_MAX_FRAMES=240
STRAPLE_VIS_FPS=30
STRAPLE_DEMO_SCORE_PNG=vis/score.png
STRAPLE_DEMO_SCORE_SAMPLE_S=1.0
```

---

## 10. Сессия 2026-05-05 — DreamPlace++ recipe (cluster init + visualizer)

### Цель сессии
Воспроизвести MTK-style ANCHOR_SOFT pipeline (видео `mtk_dreamplace_plus_ibm01.mp4`):
кластеризовать макросы по netlist'у → spawn в одной anchor-точке на кластер → GP
раскручивает по canvas с сохранением структуры. Визуализировать процесс с
cluster colors как у MTK, экспортировать HTML в `vis/`.

### Baseline (точка отсчёта, ibm01)

| Что | proxy | wl | den | cong | overlaps | runtime |
|---|---|---|---|---|---|---|
| INITIAL (.plc по умолчанию) | 1.0385 | 0.064 | 0.812 | 1.137 | **69** ⚠ | — |
| Submitted ALNS (current) | 1.0584 | 0.072 | 0.840 | 1.133 | 0 ✅ | ~25 мин |
| gradient_demo PLACE_ALL=1 (default, 300 iter) | **1.6868** | 0.127 | 0.991 | 2.129 | 0 ✅ | 3.85с |
| MTK DreamPlace++ (видео target) | ~0.91 | — | — | — | 0 ✅ | ~37с |

Vanilla gradient_demo далеко от submitted ALNS и от MTK. Главные потери:
density 0.99 (оптимум ~0.5) и congestion 2.13 (оптимум ~0.7) — макросы кучкуются
в центре canvas вместо равномерного распределения.

### Поэтапный план

**Phase 1: clustering.py** (networkx Louvain)
- Hypergraph nets → undirected weighted graph (clique expansion, weight = 1/(k-1))
- Игнорировать nets с >K макросов (K=20) — supernets зашумляют
- Louvain → partition; fallback на label_propagation для очень больших
- Output: `cluster_id: tensor[n_total]`, `num_clusters: int`
- Success criteria: на ibm01 K кластеров в диапазоне 8-50, дисперсия размеров < 5×

**Phase 2: ANCHOR_SOFT init в gradient_demo**
- Новый env `STRAPLE_DEMO_INIT=anchor_soft`
- Anchors распределяются по canvas равномерным grid'ом по числу кластеров (или k-means)
- Members кластера spawn в радиусе R = anchor_jitter * canvas_min от anchor (Gaussian)
- Hard и soft трактуются одинаково
- `cluster_id` сохраняется в recorder для визуализации
- Success criteria: после init proxy >> baseline (все макросы в anchors), но после 200 iter
  GP распределяет их и proxy < 1.6868 (текущий gradient_demo с center init)

**Phase 3: cluster colors в visualizer.py**
- `PlacementRecorder.set_cluster_ids(arr)` — сохранить
- В HTML: каждый макрос получает поле `cluster` → цвет HSV(c/K, 0.7, 0.85)
- Hover tooltip показывает cluster id
- Сохранять выходной HTML в `vis/cluster_init_<bench>_<timestamp>.html`

**Phase 4: tune schedule**
- Больше iters (500-800)
- Adaptive λ_d schedule (DREAMPlace overflow rule)
- Multi-phase: spreading (low λ, high γ) → settling (high λ, low γ)
- Success criteria: proxy < 1.30 на ibm01 в pure-gradient (близко к ALNS submission 1.06)

**Phase 5: extend на 4-5 benchmarks**
- ibm01 (small), ibm04 (mid), ibm10 (mid), ibm14 (large), ibm17 (xlarge)
- Per-bench proxy + AVG → таблица в results.md
- Если AVG < ALNS — этот pipeline становится seed candidate для submitted placer

### Что готово до сессии (можно переиспользовать)
- `gradient_demo.py` Adam loop с adaptive λ + γ cooling — менять только init
- `_build_net_pin_tensors_full(benchmark, plc)` — net→macros mapping для hard+soft
- `visualizer.py` HTML с canvas+JS+score history — только цвета и tooltip добавить
- `STRAPLE_DEMO_PLACE_ALL=1` уже двигает все макросы

### Exit conditions сессии
- Минимум: cluster init работает, HTML с цветными кластерами в `vis/`, baseline для сравнения
- Норм: proxy на ibm01 < 1.30 (улучшение vs текущий gradient_demo на 22%+)
- Хорошо: pure-gradient pipeline matches ALNS на ibm01 (1.06)
- Идеально: pure-gradient AVG ≤ ALNS — становится seed для submitted placer

### Реализация и результаты (что получилось)

**Реализовано:**
- ✅ `submissions/straple/clustering.py` — Louvain на hypergraph clique-projection (180 LOC).
  Поддержка target_num_clusters через бинарный поиск resolution. На ibm01 (1140 макросов,
  5993 нета) кластеризация занимает 0.2с, K=18-19 (или target).
- ✅ `gradient_demo.py::init_mode=anchor_soft` + adaptive defaults (`auto` для K и target_util).
- ✅ `visualizer.py` cluster colors: HSV golden-ratio palette, кнопка переключения color-mode
  (`cluster ↔ kind`), hotkey `c`, tooltip показывает `cluster: c / K`.

**Метрики (best per bench, см. results.md cycle #28):**

| bench | proxy v1 (default) | proxy v2 (tuned) | RePlAce | vs RePlAce | runtime |
|---|---|---|---|---|---|
| ibm01 | 1.245 | **1.105** (K=40 r=0.05 util=auto λ=100 400i) | 0.998 | +10.7% | 14s |
| ibm04 | 1.727 | **1.485** (K=auto r=0.05 util=0.85 λ=1000 800i) | 1.302 | +14.0% | 38s |
| ibm10 | 2.179 | **1.873** (same v2 config) | 1.501 | +24.8% | 60s |
| ibm14 | 2.514 | **1.906** (same v2 config) | 1.544 | +23.5% | 80s |
| ibm17 | 2.290 | **1.886** (same v2 config) | 1.645 | +14.7% | 100s |
| **AVG5** | 1.991 | **1.651** | 1.398 | **+18.1%** | — |

Tuning v1→v2 = **−17%** AVG. Auto-defaults в коде: lambda_max=100/1000/2000 по
n_total<1500/<2500/else; target_util = actual_util × {0.95, 1.05} по тому же критерию.

**Главный win:** на **ibm01** pure-gradient за 14 секунд достигает proxy 1.105,
тогда как submitted ALNS требует **25 минут** для 1.058. То есть pure-gradient — отличный
кандидат на **seed для ALNS polish** (ALNS получит уже хороший начальный layout вместо
default initial с overlaps).

**HTML визуализации в `vis/`:**
- `ibm01_anchor_soft_best.html` (21.6 MB, 80 frames) — ⭐ best config
- `ibm01_anchor_soft_v1.html` (21 MB) — first run
- `ibm04_anchor_soft_v2.html`, `ibm10_anchor_soft.html`, `ibm14_anchor_soft.html`,
  `ibm17_anchor_soft.html`

В HTML видна эволюция MTK-style: iter=0 кластеры схлопнуты в anchor-точки, iter=200
"венозные" паттерны, finale — rainbow placement по всему canvas.

**Что НЕ закрыло:**
- Pure-gradient на больших бенчмарках (ibm04+) +14% от RePlAce. Density >0.85 даже с λmax=1000.
- Корневая причина: density penalty smooth-bell weak, soft-soft pair force отсутствует,
  congestion вообще не моделируется в loss (только косвенно через WL).
- Нужен либо DREAMPlace electrostatic potential model, либо congestion-aware loss term.

### Следующие шаги (для будущей сессии)
1. **Multi-seed averaging** прямо в gradient_demo (параллельный sweep сидов через mp.Pool)
2. **Hybrid: gradient seed → ALNS polish** в submitted placer.py (сменить initial_pos на gradient output)
3. **Per-cluster anchor displacement loss** — добавить loss term, удерживающий cluster в окрестности anchor (стабилизатор)
4. **Real congestion-aware loss** (smooth net bbox → cell demand → top-k mean) — проводился эксп, не сходился
5. **ePlace electrostatic FFT density** — переход с bell-curve density penalty на правильный physics-based

