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
