# 📊 Результаты замеров

> Полный журнал прогонов на нашем железе. Краткая сводка — в [todo.md](todo.md#5-журнал-экспериментов).

---

## 🏆 Текущий best score (FINAL для submission)

| | Значение |
|---|---|
| **AVG17 proxy** (все 17 IBM benches) | **1.4445** ⭐ (vs RePlAce -0.91%, vs Top-10 +2.62%) |
| **AVG4 proxy** (ibm01/10/14/17) | **1.3886** ⭐⭐⭐⭐ #24 (**vs RePlAce -2.19%, ниже Top-10 -1.35%!**) |
| Average runtime per bench | **8.8 мин** (528s) |
| Max per bench (ibm17) | 28.7 мин (в пределах 1ч лимита) |
| Total placer time (sum 17) | 149.6 мин (parallel wall: 32.5 мин с 12 workers) |
| Overlaps | **0 на ВСЕХ 17** ✅ |
| AVG17 proxy (--all) | 1.5181 (#4, перезамерить) |
| Placer | `submissions/straple/placer.py` (C++ + adaptive LNS + чередование congested/random) |
| Дата замера | 2026-05-03 |
| Wall time (fast_check) | ~90-115с на 4 бенчах parallel |
| Hardware | Mac ARM (aarch64-apple-darwin), Python 3.14.4, torch 2.10.0 |

**Гэп до целей:**

| Цель | Score | Δ от текущего | % улучшения нужно |
|---|---|---|---|
| RePlAce baseline (порог) | 1.4578 | -0.0603 | **-4.0%** |
| Топ-10 leaderboard | 1.4076 | -0.1105 | -7.3% |
| **Топ-7 → Гран-при $20K** | **1.3479** | **-0.1702** | **-11.2%** |
| Топ-3-5 | ~ 1.32 | -0.20 | -13% |
| Первое место (Cezar) | 1.2224 | -0.2957 | -19.5% |

---

## 📂 Журнал прогонов

### #27 · 2026-05-05 · 🚨 КРИТИЧЕСКОЕ ОТКРЫТИЕ: soft macros можно двигать + gradient demo

**Главное**: пересмотр спецификации показал что мы 894/1140 макросов (78%) не оптимизировали. Soft позиции movable, не зафиксированы, не запрещены к движению. Подтверждено в `_set_placement` (objective.py:200-218): функция явно записывает позиции и hard, и soft.

**Подтверждение через MTK видео** (3 место leaderboard): кадры показывают rainbow-colored placement где hard И soft равномерно заполняют весь canvas. Их рецепт включает **placement всех макросов**.

**Что не сделано в submitted placer** (commit `c758df2`, AVG17=1.4445):
- LNS pipeline updates только `pos[:n_hard]`
- Soft остаются на initial positions (`benchmark.macro_positions[n_hard:]`)
- Это критический пробел — потенциал улучшения только за исправление: **-5..-20% AVG17**

**Создана инфраструктура** (uncommitted, в `submissions/straple/`):
- `gradient_demo.py` — pure-PyTorch DREAMPlace-style placer (Adam + density bell + smooth HPWL + adaptive λ + cooling γ)
- `force_demo.py` — physics demo (force-directed + ops)
- `visualizer.py` — 2×2 HTML/MP4/GIF visualizer (canvas + JS, hover tooltips, ←/→ navigation)

**Эксперименты на ibm01** (gradient demo):

| Конфиг | proxy | WL | D | C | overlaps | примечание |
|---|---|---|---|---|---|---|
| Submitted ALNS (hard only) | 1.0584 | 0.072 | 0.840 | 1.133 | 0 ✅ | submission baseline |
| Force physics | 1.5582 | 0.133 | 1.046 | 1.804 | 0 ✅ | без netlist attraction |
| Gradient (hard only) + legalize | 1.4288 | 0.098 | 1.026 | 1.635 | 0 ✅ | до открытия про soft |
| Gradient gauss_overlap | 1.5800 | 0.133 | 1.036 | 1.858 | 0 ✅ | pure gradient, без legalize |
| Gradient + Lagrangian (ρ=0.05) | 1.6991 | 0.111 | 1.184 | 2.045 | 0 ✅ | augmented Lagrangian |
| **Gradient PLACE_ALL=1 + legalize** | **1.5288** | **0.142** | **0.752** | 2.000 | 0 ✅ | **density -28% за счёт soft** |
| Gradient + 23 restarts | 1.5359 | 0.134 | 1.040 | 1.763 | 0 ✅ | best of 23 random inits |
| MTK DreamPlace++ (видео) | ~0.91 | — | — | — | 0 ✅ | их submission, реф |

**Интерпретация**:
- Density упала с 1.04 → 0.752 (-28%) ТОЛЬКО за счёт включения soft макросов в оптимизацию
- WL и cong подросли — soft пины распределились, bbox-ы нетов стали шире
- Net trade-off на ibm01: proxy 1.4288 → 1.5288 в gradient_demo (хуже), но baseline для тестов
- В реальном LNS-pipeline с правильным balance ожидается -5..-20% на AVG17

**Не используется в submission** — gradient_demo это **демо/визуализация**, не submitted placer. Submitted placer (LNS) пока всё ещё hard-only. Перенос PLACE_ALL=1 в submitted pipeline — приоритет следующей сессии.

**Файлы артефактов**:
- `vis/all_macros_demo.html` — HTML с PLACE_ALL=1 (60 МБ, 240 кадров)
- `vis/restart_demo.html` — multi-restart (61 МБ)
- `vis/gauss_overlap_demo.html` — pure gradient gauss form (60 МБ)
- `readme/mtk_dreamplace_plus_ibm01.mp4` — MTK ANCHOR_SOFT reference (от пользователя)

**Команда воспроизведения** (PLACE_ALL=1):
```bash
STRAPLE_DEMO=gradient STRAPLE_DEMO_PLACE_ALL=1 \
  STRAPLE_DEMO_INIT=center STRAPLE_DEMO_ITERS=1500 \
  STRAPLE_DEMO_LR=0.5 STRAPLE_DEMO_LR_PLATEAU=1 \
  STRAPLE_DEMO_GAMMA_START=1.5 STRAPLE_DEMO_GAMMA_END=0.3 \
  STRAPLE_DEMO_LAMBDA_START=0.05 STRAPLE_DEMO_LAMBDA_MAX=200 \
  STRAPLE_DEMO_OVERLAP_FORM=gauss_overlap STRAPLE_DEMO_OVERLAP_W=15 \
  STRAPLE_DEMO_FINISH_LEGALIZE=1 \
  STRAPLE_VIS_VIDEO=vis/all_macros_demo.html \
  uv run evaluate submissions/straple/placer.py -b ibm01
```

**Подробнее** — [todo.md секция 9](todo.md#9-сессия-2026-05-04--2026-05-05-gradient-deep-dive--критический-инсайт-про-soft-macros).

---

### #26 · 2026-05-04 · попытка плана C (DREAMPlace + perturbation) — РЕГРЕССИЯ, прервано

**Контекст**: следующая сессия после submission #25 (AVG4=1.3886, AVG17=1.4445). Цель — пробить топ-7 (≤1.3479) для квалификации на Гран-при.

**Что попробовали:**

**(а) DREAMPlace native build на Mac arm64** — слишком долго setup (нужны GCC 7.5, Boost, Bison, кучу submodules). Через Docker x86 emulation (colima) — образ `limbo018/dreamplace:cuda` 20 ГБ скачан, но это только окружение, билд внутри 1+ час. Отложено.

**(б) Свой `analytical_seed.py` (Adam + smooth HPWL + density bell)** — проверили standalone на ibm17 (initial proxy=1.7392):
- 200 шагов lambda=50000 (placer.py default): proxy=**1.8816** (+8.2%)
- 300 шагов cold_start schedule: proxy=**1.9440** (+11.8%)
- 50 шагов lr=0.1 lambda=1000 (минимально-инвазивно): proxy=**1.7850** (+2.6%)

ВСЕ конфиги ХУЖЕ initial. Корневая причина: smooth WL surrogate (LSE) и density bell не коррелируют с реальным proxy. Adam успешно минимизирует loss (8M→4M), но реальный proxy_cost растёт. Конгестию вообще не моделируем (66% метрики).

Чтобы починить — переписать loss с congestion-aware moe + adaptive density_weight (как DREAMPlace). 1-2 дня работы.

**(в) Perturbed-initial multi-start (план A)** — добавили в `placer.py`:
- `_perturb_initial`: Гауссов шум σ_factor × min(canvas_w, canvas_h), clamp в canvas, non-movable не трогаем
- env vars: `STRAPLE_PERTURB_EXTRA_STARTS`, `STRAPLE_PERTURB_SCALES`, `STRAPLE_LNS_OUTER_CAP`, `STRAPLE_LNS_OUTER_FACTOR`

**Тест 1 (заменили start 1, 2 на perturbed σ=0.03, 0.05):** РЕГРЕССИЯ 1.3886 → **1.3962** (+0.55%).

| Bench | baseline #25 | perturb σ=0.03,0.05 | Δ |
|---|---|---|---|
| ibm01 | 1.0584 | 1.0584 | 0 (n<300, не trigger) |
| ibm10 | 1.2282 | **1.2588** | **+2.49%** ⚠ |
| ibm14 | 1.5454 | 1.5454 | 0 |
| ibm17 | 1.7223 | 1.7223 | 0 |
| **AVG4** | **1.3886** | **1.3962** | **+0.55%** ⚠ |
| Wall | 1797s | 1664s | -7% (быстрее, т.к. perturb path иногда раньше сходится) |

**Урок #1**: same-seed multi-start (3 starts с seeds 42, 43, 44) УЖЕ даёт diversity, особенно на ibm10. `_run_one_start` использует `self.seed + start_idx` для C++ RNG (state.rng) и Python ALNS (np.random.default_rng), разные start_idx → разные пути → разные минимумы. Заменять на perturbed = терять рабочую находку.

**Refactor (не измерен, fast_check прервал юзер)**: keep 3 same-seed orig + ADD `STRAPLE_PERTURB_EXTRA_STARTS` perturbed extras. Best-of-N ≤ best-of-3 в худшем случае. Стоит измерить.

**Что делать дальше** (см. [todo.md секция 7](todo.md#7-сессия-2026-05-04-попытка-плана-c-прервано)):
- Закоммитить env-var infrastructure (поведение по дефолту = baseline)
- `STRAPLE_LNS_OUTER_CAP=100000 STRAPLE_LNS_OUTER_FACTOR=120` — больше LNS budget, может дать -0.5..-1%
- `STRAPLE_PERTURB_EXTRA_STARTS=2` — additive perturbed starts, без потери same-seed
- DREAMPlace integration — отдельная многодневная задача

**Состояние кода**: изменения в `submissions/straple/placer.py` uncommitted (60 строк изменений). Backward-compatible — default behavior без env vars = #25 baseline. `git checkout submissions/straple/placer.py` для отката.

**Команды для воспроизведения регрессии:**
```bash
# Регрессия (заменили same-seed на perturb) — было до refactor:
# (Сейчас в коде refactor — заменено на additive)
# git diff покажет текущее состояние
```

---

### #25 · 2026-05-04 · 🏆 FULL `--all` РЕЗУЛЬТАТ для submission: AVG17=1.4445 (vs RePlAce -0.91%)

Запустили `scripts/fast_check.py --benches ibm01..ibm18 --workers 12` (parallel runner) на всех 17 IBM benchmarks с финальным конфигом:
- 3 starts + до 3 refine passes
- LNS budget: max(30, min(50000, ceil(60*N_movable)))
- 4 ALNS-операторa (rand/cong/swap/cluster) с adaptive weights
- Shake-up при stagnation

**Per-bench результаты** (overlaps=0 на всех):

| Bench | n_macros | Proxy | RePlAce | Δ vs RePlAce | Time |
|---|---|---|---|---|---|
| ibm01 | 246 | **1.0584** | 0.9976 | +6.1% | 96.7s |
| ibm02 | 254 | 1.4957 | 1.8370 | **-18.6%** ⭐ | 161.5s |
| ibm03 | 269 | 1.3666 | 1.5223 | **-10.2%** ⭐ | 110.1s |
| ibm04 | 285 | 1.3233 | 1.5786 | **-16.2%** ⭐ | 157.6s |
| ibm06 | 318 | 1.6748 | 1.6182 | +3.5% | 95.9s |
| ibm07 | 335 | 1.5511 | 1.4717 | +5.4% | 152.8s |
| ibm08 | 352 | 1.4590 | 1.4287 | +2.1% | 286.2s |
| ibm09 | 369 | 1.1434 | 1.1192 | +2.2% | 134.5s |
| ibm10 | 387 | **1.2282** | 1.4928 | **-17.7%** ⭐ | 1486.1s |
| ibm11 | 405 | 1.1939 | 1.1781 | +1.3% | 334.8s |
| ibm12 | 423 | 1.6066 | 1.7239 | **-6.8%** ⭐ | 1174.0s |
| ibm13 | 441 | 1.3669 | 1.3304 | +2.7% | 493.7s |
| ibm14 | 460 | 1.5454 | 1.5436 | +0.1% (parity) | 968.3s |
| ibm15 | 479 | 1.5730 | 1.5132 | +4.0% | 550.1s |
| ibm16 | 498 | 1.4687 | 1.4760 | -0.5% (parity) | 794.5s |
| ibm17 | 517 | 1.7223 | 1.6448 | +4.7% | 1723.6s (28.7 мин) |
| ibm18 | 537 | 1.7791 | 1.7722 | +0.4% (parity) | 256.9s |
| **AVG17** | — | **1.4445** | **1.4578** | **-0.91%** ⭐ | **avg=8.8 мин** |

- **vs RePlAce (1.4578): -0.91%** ⭐⭐⭐ — пробили baseline
- vs Top-10 (1.4076): +2.62% — между 14-17 местом
- vs Top-7 Гран-при (1.3479): +7.17% — далеко
- vs Straple #4 baseline (1.5181): -4.85%

**Где обходим RePlAce** (8 из 17 бенчей): ibm02 -18.6%, ibm03 -10.2%, ibm04 -16.2%, ibm10 -17.7%, ibm12 -6.8%, ibm14 parity, ibm16 parity, ibm18 parity.

**Где RePlAce лучше**: ibm01 (+6.1%), ibm06-09 (+2-5%), ibm07/15/17 (+4-5%).

**Time**: 32.5 мин wall (parallel 12 workers), max ibm17=28.7 мин — внутри 1ч/bench лимита ✅.

**Команда**: `uv run python scripts/fast_check.py --benches ibm01..ibm18 --workers 12`

---

### #24 · 2026-05-04 · Multiple refine passes (3) — FINAL BEST 🏆🏆 AVG4=1.3886 (vs RePlAce -2.19%)

**Идея**: вместо одного refine pass, делать до 3 passes (с early termination если refine не улучшил).

**Изменения**: цикл `for refine_iter in range(3)`, `break` если new cost ≥ best.

**Сводка** (1 fast_check):

| Bench | Proxy | vs #23 (1.3921) | vs Straple #4 baseline | vs RePlAce |
|---|---|---|---|---|
| ibm01 | **1.0584** ⭐ | -0.12% | -10.16% ⭐ | +6.1% |
| ibm10 | **1.2282** ⭐ | -0.84% | -11.27% ⭐ | **-17.7%** ⭐ |
| ibm14 | **1.5454** | -0.10% | -5.07% | +0.1% (parity!) |
| ibm17 | **1.7223** | -0.05% | -1.31% | +4.7% |
| **AVG4** | **1.3886** ⭐⭐⭐⭐ | **-0.25%** | **-6.42%** ⭐ | **-2.19%** ⭐⭐⭐ |

- **vs RePlAce: -2.19%**
- **vs Top-10 (1.4076): -1.35%**
- **ibm14 на parity с RePlAce (+0.1%)** — почти доделали!
- ibm10: **-17.7% от RePlAce**
- 0 overlaps, smoke 10/10
- wall 1657s (ibm17 1585s, ibm10 1280s) — близко к 1ч/bench limit

**Прогресс session за 24 цикла**:
- Старт: AVG4=1.4839 (vs RePlAce **+3.24%**)
- **Финал: AVG4=1.3886 (vs RePlAce -2.19%)**
- **Δ = -6.42% от старта**

**Команда**: `uv run python scripts/fast_check.py`

---

### #23 · 2026-05-04 · Refine pass на best (intensification phase) — best после cycle #23

**Идея**: после finishing all multi-start LNS, взять best position и сделать **дополнительный full LNS pass** с другим RNG seed. Это даёт второй "intensification" — ALNS weights пересоберутся с новым context, shake-up triggered более intense, поскольку basin уже глубокий.

**Изменения**: в `place()` после multi-start цикла, если `best_pos` найден, запускаем `_lns_loop` ещё раз на нём (state2, seed+9999).

**Сводка** (1 fast_check):

| Bench | Proxy | vs #22 (1.3970) | vs Straple #4 baseline | vs RePlAce |
|---|---|---|---|---|
| ibm01 | **1.0597** ⭐ | -0.47% | -10.05% ⭐ | +6.2% |
| ibm10 | **1.2386** ⭐ | -0.86% | -10.52% ⭐ | **-17.0%** ⭐ |
| ibm14 | **1.5469** ⭐ | -0.20% | -4.98% | +0.2% (parity++) |
| ibm17 | **1.7232** ⭐ | -0.05% | -1.25% | +4.8% |
| **AVG4** | **1.3921** ⭐⭐⭐ | **-0.35%** | **-6.19%** ⭐ | **-1.94%** ⭐⭐⭐ |

- **vs RePlAce: -1.94%** (раньше -1.60%)
- **vs Top-10 (1.4076): -1.10%**
- ibm10: **-17.0% от RePlAce**
- 0 overlaps, smoke 10/10
- wall 1116s (ibm17 1044s, ibm10 865s) — ещё в 1ч/bench budget

**Прогресс session за 23 цикла**:
- Старт: AVG4=1.4839 (vs RePlAce **+3.24%**)
- Финал: AVG4=1.3921 (vs RePlAce **-1.94%**)
- **Δ = -6.19%** — пробили на 5.18% от baseline

**Команда**: `uv run python scripts/fast_check.py`

---

### #22 · 2026-05-04 · Shake-up + LNS 50000 (3×N=60·movable) — best после cycle #22

**Изменения**:
- **Shake-up механизм**: при `no_improve_count >= shake_threshold` (300+) делаем большое perturbation: swap k=min(N/4, 50) макросов + destroy_and_repair тех же. До 5 раз. Сбрасываем op_weights к 1.0.
- LNS budget: 25000 → 50000 (scale 25·N → 60·N)
- N starts: 3 (как cycle #21)

**Сводка** (1 fast_check):

| Bench | Proxy | vs #21 (1.4055) | vs Straple #4 baseline | vs RePlAce |
|---|---|---|---|---|
| ibm01 | **1.0647** ⭐ | -0.94% | -9.63% ⭐ | +6.7% |
| ibm10 | **1.2494** ⭐ | -1.25% | -9.74% ⭐ | **-16.3%** ⭐ |
| ibm14 | **1.5500** | -0.40% | -4.79% | +0.4% (parity) |
| ibm17 | **1.7240** | -0.10% | -1.21% | +4.8% |
| **AVG4** | **1.3970** ⭐⭐⭐ | **-0.60%** | **-5.85%** ⭐ | **-1.60%** ⭐⭐⭐ |

- **vs RePlAce: -1.60%** (раньше -1.00%)
- **vs Top-10 (1.4076): -0.75%** (намного ниже!)
- ibm10: **-16.3% от RePlAce**
- 0 overlaps, smoke 10/10
- wall 850s (ibm17 778s), в 1ч/bench budget

**Прогресс session за 22 цикла**: 1.4839 → 1.3970 = **-5.85%**, vs RePlAce: +3.24% → **-1.60%**

**Команда**: `uv run python scripts/fast_check.py`

---

### #21 · 2026-05-04 · 3 starts × 25000 LNS iters — best после cycle #21

**Изменения**:
- num_orig_starts: 5 → 3 (меньше parallel starts, больше iterations per start)
- adaptive_outer: 15000 → 25000, scale 15·N → 25·N

**Идея**: deeper exploration per start вместо более широкого. Выигрыш от ALNS наибольший в long runs (operator weights успевают сойтись).

**Сводка** (детерминизм, 2 запуска идентичны):

| Bench | Proxy | vs #20 (1.4108) | vs Straple #4 baseline | vs RePlAce |
|---|---|---|---|---|
| ibm01 | **1.0748** ⭐ | -1.07% | -8.78% ⭐ | +7.7% |
| ibm10 | **1.2652** ⭐ | -0.47% | -8.59% ⭐ | **-15.2%** ⭐ |
| ibm14 | **1.5562** ⭐ | -0.15% | -4.41% | +0.8% (parity) |
| ibm17 | **1.7257** ⭐ | -0.08% | -1.11% | +4.9% |
| **AVG4** | **1.4055** ⭐⭐⭐ | **-0.38%** | **-5.28%** ⭐ | **-1.00%** ⭐⭐⭐ |

- **vs RePlAce: -1.00%** ⭐ (раньше -0.63%)
- **vs Top-10 (1.4076): -0.15% ⭐** — мы ниже Top-10 на AVG4!
- ibm10: **-15.2% от RePlAce**
- 0 overlaps, smoke 10/10
- wall 384s

**Прогресс session**:
- Старт: AVG4=1.4839 (vs RePlAce +3.24%)
- **Сейчас: AVG4=1.4055 (vs RePlAce -1.00%)**
- Δ = **-5.28%** за 21 цикл

**Команда**: `uv run python scripts/fast_check.py`

---

### #20 · 2026-05-04 · cluster destroy + smart swap + ALNS adaptive weights + LNS 15000 — best после cycle #20

**Что сделали (5 inc цик)**:
- **#20a**: `destroyClusterAndRepair(k)` в C++ — BFS из random seed по net-graph, выбирает k связанных макросов, репэйрит. Атакует tightly-coupled clusters.
- **#20b**: Smart swap — `swapTwoMacros` теперь с вероятностью 60% выбирает swap из net-neighbors (не random).
- **#20c**: **ALNS adaptive operator weights**:
  - 4 операторa: rand, cong, swap, cluster
  - Warmup 40 итер: round-robin
  - После: weighted random по `op_weights`, обновление `weight = decay·weight + reaction·(1+100·gain)` при accept; `weight *= decay` при reject (min 0.05)
- **#20d**: LNS budget 8000 → 15000, scale 8.0·N → 15.0·N

**Сводка** (детерминизм проверен — 2 запуска идентичны 1.4108):

| Bench | Proxy | vs #19 (1.4268) | vs Straple #4 baseline | vs RePlAce |
|---|---|---|---|---|
| ibm01 | **1.0864** ⭐ | -0.91% | -7.79% ⭐ | +8.9% |
| ibm10 | **1.2712** ⭐ | -2.48% | -8.16% ⭐ | **-14.8%** ⭐ |
| ibm14 | **1.5586** ⭐ | -1.12% | -4.27% | +1.0% (на parity) |
| ibm17 | **1.7270** ⭐ | -0.23% | -1.04% | +5.0% |
| **AVG4** | **1.4108** ⭐⭐⭐ | **-1.12%** | **-4.93%** ⭐ | **-0.63%** ⭐⭐⭐ |

- **vs RePlAce: -0.63% — 🎉 ПРОБИЛИ baseline!!!** (раньше +0.50%)
- ibm10: -14.8% от RePlAce — намного лучше
- 0 overlaps, smoke 10/10
- wall ~382s

**Где мы сейчас в leaderboard** (AVG17 экстраполяция от AVG4):
- RePlAce baseline: 1.4578 ✓ ниже
- Топ-10: 1.4076 — осталось ~0.23%
- Топ-7 (Гран-при): 1.3479

**Главный урок**:
- Heavy LNS budget + diverse operators (cluster, swap, cong, rand) + ALNS weights = win
- Smart swap (по net-neighbors) даёт реальный gain — random pair это шум
- Прогресс session: 1.4839 → **1.4108** = **-4.93% за 20 циклов**, vs RePlAce: +3.24% → **-0.63%**

**Команда**: `uv run python scripts/fast_check.py`

---

### #19 · 2026-05-04 · 2-opt swap operator + early termination + LNS 8000 — best после cycle #19

**Изменения**:
- `placer_core.cpp`: новый `swapTwoMacros(num_swaps)` — random pair swap с overlap check.
- `placer.py _lns_loop`: третий operator slot. Цикл по 3 операторам: rand, cong-aware, swap.
- LNS budget: 5000 → 8000 outer iters, scale 5.0·N → 8.0·N.
- Early termination: 500 consec rejects → break (избегает wasted compute).

**Сводка** (1 fast_check):

| Bench | Proxy | vs #18 (1.4410) | vs Straple #4 baseline | vs RePlAce | iters |
|---|---|---|---|---|---|
| ibm01 | **1.0963** ⭐ | -1.33% | -6.95% ⭐ | +9.9% | ~30 |
| ibm10 | **1.3035** ⭐ | -2.43% | -5.84% ⭐ | **-12.7%** ⭐ | ~3000 |
| ibm14 | **1.5763** | -0.28% | -3.18% | +2.1% | ~4000 |
| ibm17 | **1.7309** | -0.30% | -0.81% | +5.2% | ~6000 |
| **AVG4** | **1.4268** ⭐ | **-0.99%** | **-3.85%** ⭐ | **+0.50%** ⭐ | wall ~271s |

- **vs RePlAce: +0.50%** (раньше +1.50%) — теперь почти на уровне baseline RePlAce!
- ibm10: на **-12.7%** обходим RePlAce
- 0 overlaps

**Главные уроки**:
- **Swap operator работает** на больших (особенно ibm10): -2.43% за один цикл
- **Heavy LNS budget** — самый большой источник improvements за всю сессию (cycle 1-13: -2%; cycle 14-19: ещё -2.6% дополнительно)
- Early termination предотвращает wasted compute на стагнирующих starts

**Команда**: `uv run python scripts/fast_check.py`

---

### #18 · 2026-05-04 · Heavy LNS budget (1500 iters, scale 1.5·N) + analytical attempts + vectorize — best после cycle #18

**Контекст**: Пользователь попросил автономно добить E (DREAMPlace-style analytical placer) до улучшения метрики.

**Что попробовали (за ночь)**:

#### #14: Vectorize `_smooth_hpwl` (37× ускорение)
- Padded tensor approach с `mask` + `torch.logsumexp`
- 250ms/step → 6.8ms/step на ibm17. Open way к heavy analytical.

#### #15-#18: DREAMPlace-style analytical — провал
Серия попыток показала, что **analytical seed принципиально хуже initial benchmark.macro_positions** для нашего LNS pipeline:

1. const λ=1, 200 шагов: raw analytical proxy=1.62 → после legalize+LNS = 1.86 (vs original 1.59)
2. const λ=200, 1500 шагов: raw=1.79 (overlaps=662)
3. const λ=5000, 1500 шагов: raw=1.75 (overlaps=592)
4. **target_util=0.2 + λ=50000 + 2000 шагов**: raw=1.81 (**overlaps=46**!) — рекорд по чистоте seed
5. После legalize_min_displacement (force-directed push apart) + LNS на чистом seed: still 1.7641 на ibm14, vs original 1.5855

**Корневая причина (открытие)**: `benchmark.macro_positions` — это **уже хорошее placement** из IBM benchmark.
- ibm01 initial proxy = **1.0385** (наш best 1.1191!)
- ibm14 initial proxy = **1.5938** (наш best 1.5808)
- ibm17 initial proxy = **1.7392** (наш best 1.7362)

То есть **наш placer практически не улучшает initial**. На больших дизайнах initial — почти optimal local minimum, и любой analytical seed сходится в худший basin. Топовое решение DREAMPlace-style — это replacement, а не улучшение этих positions.

**Откат**: analytical_steps=0 by default. Код остаётся (`analytical_seed.py` с lambda/gamma schedule, `legalize_min_displacement` в C++) — может быть полезным позже.

#### #19: Heavy LNS budget — РЕАЛЬНЫЙ ПРОРЫВ ⭐

Понимание из profiling: initial уже ~1% от optimum. LNS делает 0.1-0.5% improvements per итерацию. Чтобы пробить — нужно МНОГО iterations.

Изменение: `adaptive_outer = max(self.lns_outer_iters, min(1500, ceil(1.5 · num_movable)))`. 
Раньше: `min(150, ceil(0.20·N))`.

**Сводка** (1 fast_check, детерминизм проверен):

| Bench | Proxy | vs #17 (1.4544) | vs Straple #4 baseline | vs RePlAce | iters |
|---|---|---|---|---|---|
| ibm01 | **1.1111** ⭐ | -1.51% | **-5.69%** ⭐ | +11.4% | 30 (n=246) |
| ibm10 | **1.3360** ⭐ | -2.10% | **-3.49%** ⭐ | **-10.5%** ⭐ | 581 |
| ibm14 | **1.5808** ⭐ | -0.30% | -2.90% | +2.4% | 921 |
| ibm17 | **1.7362** ⭐ | -0.17% | -0.51% | +5.6% | 1140 |
| **AVG4** | **1.4410** ⭐ | **-0.92%** | **-2.89%** ⭐ | **+1.50%** ⭐ | wall ~118s |

- vs Straple #4 baseline (1.4839): **-2.89%** — впервые превышаем -2.5%!
- vs RePlAce (на 4 бенчах AVG=1.4197): **+1.50%** (раньше +2.45%)
- ibm10 теперь **обходит RePlAce на 10.5%** (наш 1.3360 vs RePlAce 1.4928)
- 0 overlaps, smoke 10/10

**Главный урок**:
- **Initial benchmark positions УЖЕ хороши** — наш job в том чтобы найти marginal improvements от них через много LNS-итераций
- Тривиальный подход "1500 итераций вместо 150" дал **3× больше улучшения чем все предыдущие хитрые тюнинги cycle 1-13** (вместе)
- Analytical/DREAMPlace-style — фундаментально неприменим если initial уже хорош (без replacement)

**Время**: ibm10 48s, ibm17 44s — в 1-час budget огромный запас.

**Команда**: `uv run python scripts/fast_check.py`

---

### #17 · 2026-05-04 · skip SA для больших дизайнов + детальные SA stats — best после cycle #13

**Что нашли через SA-логирование** (новый `sa_refine_with_stats` в C++ возвращает accept/reject/boltzmann counts + WL trajectory):

На **ibm17** (n_movable=760):
- 3000 SA-итераций, **80% rejection — overlap** (`rej_overlap=2006/2496`). SA aggressive shift (`tStart=0.15·canvas`=10.9) сдвигает макросы в занятые места.
- WL улучшается на **~5.7%** (20074→18935), но **proxy ухудшается на +0.4%** (1.7432→1.7498) — SA tightly clustered макросы → density/cong растут.
- Аналогично на всех 5 starts. **SA активно ухудшает proxy на больших.**

На **ibm01** (n_movable=246):
- SA полезна (~~+1.4%~~ если выключить → регрессия). Меньше макросов = меньше overlap conflicts, WL gain не компенсируется density loss.

**Решение** (одна строка): запускать SA только для `num_movable < 300`. Для больших — skip.

**Изменения**:
- `submissions/straple/cpp/placer_core.cpp`: добавлен `simulatedAnnealingRefineWithStats(num_iters, snapshot_every)` → возвращает `py::dict` с counts (accepted/rejected/boltzmann/overlap) и WL trajectory. Поведение идентично оригинальному `sa_refine` (одинаковый seed = одинаковый результат) — это важно, тест предотвратит регресс если кто-то их разойдёт.
- `submissions/straple/placer.py`: `sa_iters_to_run = self.refine_iters if num_movable < 300 else 0`. Verbose использует `sa_refine_with_stats` для трассировки.

**Сводка** (медиана 3 запусков, детерминизм):

| Bench | Proxy | vs #16 (1.4619) | vs Straple #4 baseline | SA |
|---|---|---|---|---|
| ibm01 | 1.1282 | 0.00% | -4.23% | 3000 iter |
| ibm10 | **1.3647** | **-0.86%** ⭐ | **-1.41%** ⭐ | skip |
| ibm14 | **1.5855** | **-1.06%** ⭐ | **-2.61%** ⭐ | skip |
| ibm17 | **1.7392** | **-0.07%** | -0.34% | skip |
| **AVG4** | **1.4544** ⭐ | **-0.51%** | **-1.99%** ⭐ | wall ~108s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+2.45%** (раньше +2.98%, теперь ближе на пол-процента)
- vs прошлый best (#16: 1.4619): **-0.51%** ▼
- vs Straple #4 baseline (1.4839): **-1.99%** ▼ — впервые превышаем -2%
- Smoke 10/10, 0 overlaps

**Главный урок**:
- **SA optimize HPWL only**, на больших это **деструктивно** для density+cong.
- Логирование с stats показало истинную картину: 80% wasted iterations + WL/proxy disconnect.
- "Один параметр = один эффект" — иногда правильнее **отключить** что-то чем тюнить.

**Команда**: `STRAPLE_VERBOSE=1 uv run python -c "..."` для SA stats (counts + trajectory). 

---

### #16 · 2026-05-04 · 🚨 BUG FIX: ProxyEvaluator.evaluate() веса 1:1:1 → 1:0.5:0.5 + логирование

**Корневая проблема (найдена через детальное логирование)**: `_proxy_cost.cpp::evaluate()` возвращал `wl + density + congestion` (unweighted sum), а **реальный** `proxy_cost = 1·wl + 0.5·density + 0.5·congestion` (см. macro_place/objective.py:148-162).

Это значит:
- **Multi-start выбирал best по неправильной метрике** — мог выбрать вариант с лучшим `wl+den+cong`, но худшим реальным `proxy_cost`.
- **LNS accept/reject** работало с весами 1:1:1, переоценивая density/congestion в 2× и недооценивая wirelength в 2×.
- Все 11 предыдущих циклов оптимизировали wrong objective.

**Как нашли**: добавил подробное логирование (env `STRAPLE_VERBOSE=1`) — каждая фаза с таймингом, LNS прогресс с accept/reject, операторы random/cong, gain. На ibm17 заметил несовпадение: `evaluator.evaluate()` возвращает `3.4290`, а `compute_proxy_cost(...)["proxy_cost"]` = `1.7404`. Сложил `wl+den+cong = 0.054+0.949+2.424 = 3.427` ≈ 3.43.

**Изменения**:
- `submissions/straple/cpp/proxy_cost.cpp::evaluate()` — `return wl + 0.5*dens + 0.5*cong` (раньше `wl + dens + cong`).
- `submissions/straple/placer.py`: добавлен `verbose` флаг (env `STRAPLE_VERBOSE`) — печатает фазы (load_plc, _extract_edges, _build_proxy_evaluator, legalize/SA/LNS на каждый seed), LNS прогресс с операторами и accept counts.
- `submissions/straple/analytical_seed.py`: добавлены `log_every` и `log_proxy` параметры — печатает loss/grad/proxy на промежуточных шагах.
- `test/test_smoke.py`: новый `test_straple_evaluate_returns_proxy_cost` — проверяет `evaluator.evaluate()` совпадает с `compute_proxy_cost(...)["proxy_cost"]` И что aggregate = `wl + 0.5·den + 0.5·cong`. Регрессия предотвращена.

**Сводка** (1 fast_check, детерминизм подтверждался ранее):

| Bench | Proxy | vs #15 (1.4643) | vs Straple #4 baseline | Что |
|---|---|---|---|---|
| ibm01 | **1.1282** ⭐ | **-0.83%** | -4.23% ⭐ | density 0.911→0.910, cong 1.218→1.200 |
| ibm10 | 1.3765 | 0.00% | -0.56% | best и так был оптимальным по обеим метрикам |
| ibm14 | 1.6025 | 0.00% | -1.57% | то же |
| ibm17 | 1.7405 | +0.01% | -0.26% | в шуме |
| **AVG4** | **1.4619** ⭐ | **-0.16%** | **-1.48%** ⭐ | wall ~108s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+2.98%** (раньше +3.14%)

**Главный урок**: 
- **Логи раскрыли давний баг.** За 11 циклов мы tweakали алгоритм, который оптимизировал не то.
- Теперь LNS правильно расценивает trade-off между WL/density/congestion. На ibm01 это сразу дало -0.83% — там WL улучшения раньше не accept'ились (они "терялись" на фоне Δdensity×2).
- Тест навсегда зафиксирован — не повторим.

**Команда**: `STRAPLE_VERBOSE=1 uv run python -c "..."` для отладки; `uv run python scripts/fast_check.py` для измерения.

---

### #15 · 2026-05-03 · `submissions/straple/placer.py` (LNS outer up to 150, 0.20·N) — best после cycle #11

**Идея**: lift `adaptive_outer` upper bound 100→150, scale 0.15·N→0.20·N. Продолжение #14 — больше LNS budget даёт улучшения без overshoot.

**Изменения**: `placer.py` — `adaptive_outer = max(self.lns_outer_iters, min(150, math.ceil(0.20 * num_movable)))`.

**Результат** (1 fast_check; в предыдущих циклах 3/3 идентичны → детерминизм, одного хватает):

| Bench | Proxy | vs #14 (1.4656) | vs Straple #4 baseline | outer | time |
|---|---|---|---|---|---|
| ibm01 | 1.1376 | 0.00% | -3.44% | 30 (n=246) | 0.44s |
| ibm10 | **1.3765** | **-0.28%** ⭐ | -0.56% ⭐ | 78 (vs 59) | 7.67s |
| ibm14 | 1.6025 | -0.06% | -1.57% | 92 (vs 69) | 5.84s |
| ibm17 | 1.7404 | -0.03% | -0.27% | 104 (vs 78) | 8.73s |
| **AVG4** | **1.4643** ⭐ | **-0.09%** | **-1.32%** ⭐ | wall ~96s |

- vs RePlAce: +3.14% (-0.10% дополнительно от cycle #10)
- ibm10 ещё улучшилось на -0.28% (continued return от LNS budget)
- Время растёт пропорционально, но в budget (ibm17 8.73с << 1 час)

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #14 · 2026-05-03 · `submissions/straple/placer.py` (LNS outer up to 100, 0.15·N) — best после cycle #10

**Идея**: lift `adaptive_outer` upper bound с 60 до 100, scale с 0.10·N до 0.15·N. LNS использует full proxy_cost для accept (не подвержен SA-проблеме из cycle #9).

**Изменения**: `placer.py` `_lns_loop` — `adaptive_outer = max(self.lns_outer_iters, min(100, math.ceil(0.15 * num_movable)))`.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #12 (1.4674) | vs Straple #4 baseline | outer iters | time |
|---|---|---|---|---|---|
| ibm01 | **1.1376** ⭐ | -0.31% | -3.44% ⭐ | 30 (no change, n=246) | 0.42s |
| ibm10 | **1.3804** ⭐ | -0.17% | -0.28% ⭐ | 59 (vs 39) | 5.56s |
| ibm14 | 1.6035 | 0.00% | -1.50% ⭐ | 69 (vs 46) | 4.65s |
| ibm17 | **1.7410** ⭐ | -0.07% | -0.24% | 78 (vs 52) | 6.68s |
| **AVG4** | **1.4656** ⭐ | **-0.12%** | **-1.23%** ⭐ | wall ~103-109s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.24%** (продолжаем приближаться)
- vs прошлый best (#12: 1.4674): **-0.12%** ▼ (новый best)
- vs Straple #4 baseline (1.4839): **-1.23%** ▼
- Overlaps: 0, Smoke 9/9

**Что сработало**:
- ВСЕ 4 бенча улучшились — впервые за серию циклов.
- ibm01 -0.31%: density 0.914→0.911, cong 1.222→1.218. Больше LNS-итераций позволили глубже исследовать оптимум.
- ibm10 -0.17%: даже multi-start уже нашёл local optimum, но больше итераций позволили дальше его улучшить.
- ibm17 -0.07%: первое realистичное улучшение (хоть и маленькое).

**Что не сработало**:
- ibm14 — без изменений. Multi-start уже нашёл оптимум на этом basin.

**Главный урок**:
- LNS с full proxy_cost accept — robust, можно безопасно увеличивать budget.
- Comparison с cycle #9 (SA budget): SA optimize HPWL → больше = плохо. LNS optimize full proxy → больше = лучше (с убывающей отдачей).

**Время**: ibm17 6.68с — всё ещё << 1 час лимита. Запас огромный.

**Следующие шаги**:
- Cycle #11: попробовать N=8 multi-start вместе с большим LNS, или зафиксировать N=5 и попробовать другие направления (force-directed seed).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #13 · 2026-05-03 · попытка: adaptive SA budget (3000→up to 8000) — РЕГРЕССИЯ, откатили

**Идея**: lift sa_refine с фикс 3000 до `max(3000, min(8000, 15·N_movable))`. Гипотеза: больше SA-итераций для больших дизайнов даст лучше initial.

**Результат** (1 fast_check):

| Bench | Proxy | vs #12 (1.4674) | Заметка |
|---|---|---|---|
| ibm01 | 1.1431 | +0.18% ❌ | density 0.914→0.919 |
| ibm10 | 1.3952 | +0.90% ❌ | density 0.717→0.726, cong +0.7% |
| ibm14 | 1.6121 | +0.54% ❌ | density 0.980→0.986 |
| ibm17 | 1.7448 | +0.15% | cong 2.426→2.428 |
| **AVG4** | **1.4738** | **+0.43% ❌** | — |

**Что не сработало**: SA optimize ТОЛЬКО HPWL (через delta), а HPWL = ~4% от cost. Больше SA = глубже локальный минимум HPWL, который **хуже** для full proxy_cost (density/cong часто проседают). Это classical SA overshooting на wrong objective.

**Урок**: SA budget 3000 — already sweet spot. Дальше — менять SA target (например, добавить density penalty в SA delta), не количество итераций.

**Откат**: `git checkout placer.py`.

---

### #12 · 2026-05-03 · `submissions/straple/placer.py` (multi-start N=5 для больших) — best после cycle #8

**Идея**: lift `num_starts` с 3 до 5. На ibm14/17 N=3 не давал diversification (basin too deep) — попробовать больше N. Runtime budget огромный.

**Изменения**: `placer.py` — `num_starts = 5 if num_movable >= 300 else 1`.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #11 (1.4688) | vs Straple #4 baseline | starts | time |
|---|---|---|---|---|---|
| ibm01 | 1.1411 | 0.00% | -3.14% | 1 (skip) | 0.41s |
| ibm10 | 1.3828 | 0.00% | -0.11% | 5 | 4.10s |
| ibm14 | **1.6035** | **-0.35%** ⭐ | **-1.50%** ⭐ | 5 | 3.73s |
| ibm17 | 1.7422 | 0.00% | -0.17% | 5 | 5.25s |
| **AVG4** | **1.4674** ⭐ | **-0.10%** | **-1.11%** ⭐ | wall ~110-130s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.36%**
- vs прошлый best (#11: 1.4688): **-0.10%** ▼ (новый best)
- vs Straple #4 baseline (1.4839): **-1.11%** ▼
- Overlaps: 0, Smoke 9/9

**Что сработало**:
- ibm14: впервые multi-start дал улучшение, **-0.35%**. seed 4 или 5 нашёл alternative attractor. Density 0.980→0.980, Cong 2.130→2.117 (-0.6%).
- Время роста умеренное (×1.7 для больших vs N=3).

**Что не сработало**:
- ibm17 всё равно нет улучшения. Basin attraction ОЧЕНЬ глубокий — даже 5 разных seeds сходятся в ту же точку. Возможно нужен structurally разный initial (force-directed, grid-based), не просто разный random.
- ibm10 уже в N=3 нашёл best — N=5 не помог дальше.

**Главный урок**:
- Больше N помогает, но с убывающей отдачей. ibm14 потребовал N=5; ibm17 видимо нужно ещё больше или иной подход.
- Стоит остановиться на оптимальном N — для ibm14 N=5 хватает, ibm17 — другая проблема.

**Следующие шаги**:
- Cycle #9: возможно N=8 для ibm17, или попробовать **structurally разные initial**: force-directed seed как один из стартов.
- Альтернатива: **larger SA budget** (sa_refine 3000→6000 для больших).
- Альтернатива: **lift LNS outer upper bound** (60→80 для очень больших).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #11 · 2026-05-03 · `submissions/straple/placer.py` (multi-start N=3 для больших) — best после cycle #7

**Идея**: для больших дизайнов (`num_movable >= 300`) запустить 3 независимых старта с разными seed (42/43/44) полным циклом legalize→SA→LNS, выбрать минимальный по proxy_cost. Маленькие (ibm01) — один старт, как раньше (overhead не оправдан).

**Источник**: ALNS / SA literature — multi-start как стандарт diversification. На больших дизайнах разные seed → разные attractor'ы.

**Изменения**:
- `submissions/straple/placer.py` `StraplePlacer.place()`: цикл по `num_starts = 3 if num_movable >= 300 else 1`. `_build_proxy_evaluator` строится один раз и переиспользуется. Best по trial_cost.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #8 (1.4707) | vs Straple #4 baseline | starts |
|---|---|---|---|---|
| ibm01 | 1.1411 | 0.00% | -3.14% ⭐ | 1 (skip multi-start) |
| ibm10 | **1.3828** | **-0.53%** | -0.11% ⭐ | 3 (best of 3) |
| ibm14 | 1.6092 | 0.00% | -1.15% ⭐ | 3 (1st seed = best) |
| ibm17 | 1.7422 | 0.00% | -0.17% | 3 (1st seed = best) |
| **AVG4** | **1.4688** ⭐ | **-0.13%** | **-1.02%** ⭐ | wall ~88-96s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.46%** (хуже, продолжаем приближаться)
- vs прошлый best AVG4 (#8: 1.4707): **-0.13%** ▼ (новый best)
- vs Straple #4 baseline (1.4839): **-1.02%** ▼ — впервые превышаем -1%
- Overlaps: 0, Smoke 9/9

**Что сработало**:
- ibm10 (387 макросов) — впервые улучшение, **теперь меньше baseline**: 1.3828 vs RePlAce 1.4928 (-7.4% от RePlAce!) — обходим RePlAce на этом бенче.
- Реализация чистая: один общий evaluator, минимальный overhead.
- Время растёт пропорционально (ibm10/14/17 ×3). На fast_check всё ещё <100с total wall (parallel).

**Что не сработало**:
- ibm14 и ibm17: первый seed оказался best (или все 3 сходятся в одну точку). Multi-start не дал diversification. Гипотеза: на ibm14/17 локальный минимум очень глубокий (basin of attraction большой), все 3 seeds валятся в него.

**Главный урок**:
- Multi-start работает на ibm10, но не на ibm14/17. На очень больших дизайнах одного multi-start недостаточно — нужно больше N или structurally разные initial configurations (например, force-directed + grid-based).
- Размер basin of attraction зависит от netlist — на ibm10 он маленький, easy to escape.

**Следующие шаги**:
- Cycle #8: попробовать N=5 или больше (runtime budget огромный).
- Или: structurally разные seeds — random scatter vs grid vs centroid-based initial.
- Или: best result оставить + ещё короткий run проксированный с heavy LNS.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #10 · 2026-05-03 · попытка: wider congested_percent 5%→10% — НЕЙТРАЛ, откатили

**Идея**: расширить top-N% hot-cells с 5% до 10%, дать destroy-оператору больше таргетов.

**Результат**: AVG4 1.4712 (vs #8 1.4707, +0.03% — нейтрал, в шуме). ibm01-17 без значимых изменений.

**Урок**: ширина hot-cells не bottleneck — даже с 10% top хватает destroy-таргетов. Реальный bottleneck в чём-то другом (структурном). Откатили.

---

### #9 · 2026-05-03 · попытка: tighter threshold (P30 вместо median) — РЕГРЕССИЯ, откатили

**Идея**: В `repairMacroAware` заменить median (P50) на P30 — более жёсткий threshold. Гипотеза: на ibm10 (cong-grid гладкий) median всегда срабатывает, поведение = старое; tighter threshold заставит spiral искать действительно cool зоны.

**Результат** (1 fast_check):

| Bench | Proxy | vs #8 (1.4707) | Что |
|---|---|---|---|
| ibm01 | 1.1467 | +0.49% ❌ | density 0.914→0.921 |
| ibm10 | 1.3902 | 0.00% | как и было |
| ibm14 | 1.6149 | +0.35% ❌ | density 0.980→0.982 |
| ibm17 | 1.7436 | +0.08% | нейтрал |
| **AVG4** | **1.4739** | **+0.22% ❌** | — |

**Что не сработало**: P30 слишком жёсткий — fallback на старый repair срабатывает чаще, теряем aware-эффект. На ibm10 действительно ничего не изменилось (как и предсказывали), но на ibm01/14 — регрессия.

**Урок**: median (P50) — оптимальный threshold для текущей архитектуры. Жёстче = больше fallback. Реверт через `git checkout placer_core.cpp`.

**Гипотеза для cycle #6**: атаковать ibm10/17 не через threshold, а другим способом — например, **wider congested-percent** (top 10% вместо top 5%, больше hot-cells → больше destroy-таргетов). Или: **scale outer iterations specifically для ibm17** (lift upper bound 60→80).

---

### #8 · 2026-05-03 · `submissions/straple/placer.py` (congestion-aware repair) — НОВЫЙ BEST 🚀

**Идея**: На congested-destroy ветке LNS использовать **congestion-aware spiral search** в repair: принимать non-overlap позицию ТОЛЬКО если её max-cong по перекрытым cells < median(cong-grid). Если за spiral (r∈[1..20]) не нашли — fallback на старый «слепой» repair. Random-destroy ветка не трогается (exploration сохранена).

**Источник**: прямой follow-up из #5 («repair away from hot»). Теория ALNS: destroy + domain-aware repair > destroy + blind repair.

**Изменения**:
- `submissions/straple/cpp/proxy_cost.cpp` — новый pybind метод `getCongestionGrid()` экспонирует кэш cong-grid (hRouting+vRouting) как np.array shape [gridRows, gridCols].
- `submissions/straple/cpp/placer_core.cpp` — новый `repairMacroAware(state, idx, congGrid, threshold)` — клон существующего `repairMacro` со cong-проверкой trial-позиций. Spiral ограничен r∈[1..20] (вместо 80) для перформанса. Fallback на repairMacro при non-найдено. `destroyCongestedAndRepair` принимает новые параметры congGrid и зовёт repairMacroAware.
- `submissions/straple/placer.py` `_lns_loop` — на нечётной (congested) итерации передаёт `cong_grid` в C++; на чётной (random) не трогает.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #7 (1.4795) | vs Straple #4 (1.4839) | Что улучшилось |
|---|---|---|---|---|
| ibm01 | **1.1411** ⭐ | **-1.91%** | **-3.14%** ⭐ | density 0.923→0.914, cong 1.258→1.222 |
| ibm10 | 1.3902 | 0.00% | +0.43% | нейтрал (random ветка не менялась) |
| ibm14 | **1.6092** | **-0.71%** | **-1.15%** ⭐ | density 0.995→0.980, cong 2.140→2.130 |
| ibm17 | 1.7422 | -0.09% | -0.17% | cong 2.427→2.425 |
| **AVG4** | **1.4707** | **-0.59%** ⭐ | **-0.89%** ⭐ | wall ~88-91s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.59%** (хуже, но впервые ближе чем +4%)
- vs прошлый best AVG4 (#7: 1.4795): **-0.59%** ▼ — большой шаг
- Overlaps: 0, Smoke 9/9, Build OK без warnings

**Что сработало**:
- ibm01 -1.91% и ibm14 -0.71% — congestion-aware repair размещает макросы в зоны с реально низкой нагрузкой, не «первое же свободное место рядом». Параллельно падает density (макросы распределяются равномернее).
- Fallback архитектура работает: на ibm01 (где hot-зон мало, threshold редко срабатывает) репэйр в нужный момент откатывается на старое поведение.
- Минимальный runtime overhead (+1-3%).

**Что не сработало**:
- ibm10 0% (нейтрал): congested-итерации возможно вообще не находили threshold-удовлетворяющие позиции, fallback'или на старое. Гипотеза: cong-grid у ibm10 более «гладкий» (cong=1.92, низкий) — median high, threshold почти любой trial проходит → behavior идентичен старому.
- ibm17 -0.09% — почти ничего. Главный bottleneck: на ibm17 макросов 517, congested-destroy на 26 итерациях × 13 = 338 переразмещений; если каждое placeholder в hot-зоне даёт лишь 1-2% локального улучшения, на global proxy эффект -0.5-1%. Возможно нужны ДВЕ вещи: (1) больше congested-итераций на больших, (2) более узкий threshold (quantile 0.7 вместо median).

**Главный урок**:
- Domain-aware repair > blind repair (как в ALNS-литературе).
- Fallback архитектура ключевая для robustness — не ломает маленькие, помогает большим.
- ibm01 и ibm14 теперь **в зоне RePlAce-level** (на ibm01 наш 1.14, RePlAce 1.00 → +14%; на ibm14 наш 1.61, RePlAce 1.54 → +4.4%).

**Следующие шаги**:
- Cycle #5: попробовать **более узкий threshold** (quantile 0.7 / 0.6 вместо median) — для ibm10/17, где median не срабатывает.
- Или: scale congested-percent (топ 5% → топ 10%) для большей разрядки на ibm17.
- Или: multi-start для ibm17.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #7 · 2026-05-03 · `submissions/straple/placer.py` (adaptive LNS + чередование) — best после cycle #3

**Идея**: Минимальный фикс к #6. Вернуть чередование random/congested (exploration важна, особенно для ibm01), но СОХРАНИТЬ adaptive_destroy и adaptive_outer от #6.

**Источник**: прямой вывод из cycle #6 (где удаление чередования дало +2.5% регрессии на ibm01).

**Изменения**:
- `submissions/straple/placer.py` `_lns_loop`: восстановлено `if iteration % 2 == 1:` для congested branch, на чётных — random destroy. `adaptive_destroy=clamp(8, ceil(0.025·N), 16)` и `adaptive_outer=clamp(30, ceil(0.10·N), 60)` остались от #6.

**Сводка** (медиана 3 запусков, 3/3 идентичны — детерминизм):

| Bench | Proxy | vs #5 (1.4822) | vs Straple #4 baseline | params (k, outer) |
|---|---|---|---|---|
| ibm01 | **1.1633** | **-0.37%** | -1.26% ⭐ | k=8, outer=30 |
| ibm10 | 1.3902 | +0.09% | +0.43% | k=10, outer=39 |
| ibm14 | **1.6207** | **-0.45%** ⭐ | -0.45% ⭐ | k=12, outer=46 |
| ibm17 | 1.7437 | -0.03% | -0.08% | k=13, outer=52 |
| **AVG4** | **1.4795** | **-0.18%** | **-0.30%** ⭐ | wall=88-115s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+4.21%** (хуже, но ближе всех)
- vs прошлый best AVG4 (#5: 1.4822): **-0.18%** ▼ (новый best)
- vs Straple #4 baseline (AVG4=1.4839): **-0.30%** ▼ (превосходит исходный)
- Overlaps: 0
- Smoke 9/9 PASS, build OK без warnings

**Что сработало**:
- Чередование сохранило exploration на маленьких (ibm01: 1.1676 → 1.1633, ещё -0.37%).
- Adaptive params на больших дали реальный эффект **впервые на ibm14** (-0.45%): congestion 2.150 → 2.140. Большее число итераций (46 vs 30) позволяет глубже разрабатывать area.
- ibm17 нейтрал (-0.03%) — adaptive params не повредили, но и не помогли. Возможно нужен ещё больший budget, или проблема не в количестве итераций.
- Время растёт умеренно: ibm17 2.64s vs 2.59s в #5 (+1.9%) — wall ~88-115с (variability fork).

**Что не сработало**:
- ibm10 +0.09% (микро-регрессия, в шуме). Гипотеза: для ibm10 (387 макросов, k=10) лимит 0.025 даёт мало преимущества, а 0.10 outer = 39 уже близко к минимуму 30. Возможно стоит выше lower bound для outer.
- На ibm17 эффект минимальный — возможно нужен другой подход (repair-aware-of-hot, multi-start).

**Главный урок**:
- Чередование + adaptive — комбинация работает.
- Adaptive params **сами по себе** дают эффект ТОЛЬКО при сохранении exploration (random destroy).
- Подтверждена ALNS-литература: domain-aware operators хорошо работают в комбинации с random для diversification.

**Следующие шаги**:
- Cycle #4: попробовать **repair-aware-of-hot** — в spiral search избегать high-cong cells (адресует ibm17).
- Или: multi-start с 3-5 разных seed'ов на больших.
- Или: tune adaptive lower bounds (выше floor для outer на средних).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #6 · 2026-05-03 · `submissions/straple/placer.py` (adaptive LNS, без чередования) — РЕГРЕССИЯ

**Идея**: На больших дизайнах дать LNS реальную мощность по разгрузке congestion: (1) убрать чередование random/congested — всегда congested (random — fallback в C++ при пустом overlap); (2) `adaptive_destroy = clamp(8, ceil(0.025·N), 16)`; (3) `adaptive_outer = clamp(30, ceil(0.10·N), 60)`. Бонус: `nth_element` вместо `sort` в C++ для top-k.

**Источник**: собственная гипотеза + 2 из 3 follow-up'ов из #5.

**Результаты** (1 fast_check):

| Bench | Proxy | vs #5 (1.4822 baseline) | params (k, outer) |
|---|---|---|---|
| ibm01 | 1.1964 | **+2.47%** ⬆ ❌ | k=8, outer=30 |
| ibm10 | 1.3940 | +0.36% | k=10, outer=39 |
| ibm14 | 1.6286 | +0.03% | k=12, outer=46 |
| ibm17 | 1.7441 | -0.01% | k=13, outer=52 |
| **AVG4** | **1.4908** | **+0.58%** | — |

- vs Straple #4 baseline (AVG4=1.4839): **+0.46%** (хуже)
- vs RePlAce: +5.01% (хуже)

**Что сработало**:
- Реализация технически корректная, build OK, smoke 9/9, 0 overlaps.
- Даже на ibm17 60×13 итераций успевают за 2.82с (раньше было 30×8 за 2.59с) — масштабирование adekvatное.

**Что не сработало**:
- **ibm01: -0.89% улучшение #5 → +2.47% регрессия** (= откат на хуже-чем-baseline). Параметры на ibm01 не изменились (k=8, outer=30 — те же), значит **виновник — удаление чередования**. Random destroy служил exploration; congested-only застрял в локальном.
- На ibm10/14/17 increased k/outer **не помог** — гипотеза «больше work на больших → лучше» опровергнута. Скорее всего: (а) при k=12-13 разрушается слишком много макросов одновременно, repair (centroid+spiral) не успевает качественно переразместить; (б) congested-only без exploration легко зацикливается на одних и тех же hot-cells.

**Главный урок**:
- **Чередование random/congested оказалось важной диверсификацией** — нельзя убирать.
- Adaptive params БЕЗ exploration не дают эффекта.
- На больших нужны другие подходы — может быть multi-start или repair-aware-of-hot, не наращивание выборки.

**Следующие шаги**:
- Cycle #3: вернуть чередование, попробовать только adaptive params (k и outer).
- Альтернатива: repair-aware-of-hot (в spiral search избегать high-cong cells).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #5 · 2026-05-03 · `submissions/straple/placer.py` (congestion-aware destroy в LNS)

**Идея**: чередовать в LNS-петле два destroy-оператора — старый random и новый **congestion-aware**: выбирать k макросов, которые перекрывают top-5% самых перегруженных congestion-ячеек (взвешено по площади перекрытия и ценности ячейки).

**Источник**: собственная гипотеза, опираясь на декомпозицию cost (congestion = 66% AVG) и на принцип Ropke & Pisinger 2006 (ALNS — domain-aware destroy operators).

**Изменения**:
- `submissions/straple/cpp/proxy_cost.cpp` — кэш cong-grid после `evaluate()`, новый метод `getTopCongestedCells(percent)` возвращает Nx3 (row, col, weight).
- `submissions/straple/cpp/placer_core.cpp` — новый `destroyCongestedAndRepair(positions, hot_cells, gridRows, gridCols, k)`: считает overlap каждого movable макроса с hot-cells, выбирает top-k и репэйрит существующим weighted-centroid + spiral search. Fallback на random если ни один макрос не перекрывает hot-зону.
- `submissions/straple/placer.py` — `_lns_loop` чередует random/congested destroy по чётности итерации; congested_percent=0.05.

**Сводка** (медиана из 3 fast_check запусков, ibm01/10/14/17, parallel 4 workers):

| Bench | Proxy | WL | Density | Cong | Time | vs Straple #4 |
|---|---|---|---|---|---|---|
| ibm01 | **1.1676** ⭐ | 0.073 | 0.927 | 1.262 | 0.40s | **-0.89%** |
| ibm10 | 1.3890 | 0.070 | 0.717 | 1.921 | 1.74s | +0.34% |
| ibm14 | 1.6281 | 0.053 | 1.000 | 2.150 | 1.92s | +0.01% |
| ibm17 | 1.7442 | 0.054 | 0.952 | 2.428 | 2.59s | -0.05% |
| **AVG4** | **1.4822** | — | — | — | wall=105s | **-0.11%** |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+4.40%** (хуже)
- vs Straple #4 baseline (AVG4=1.4839): **-0.11%** ▼ (микроулучшение, в пределах шума 0.5%)
- Overlaps: 0
- Bounds-violations: pre-existing на ibm10/14/17 (есть и в baseline, не наша регрессия — см. "Известные проблемы" в [improve.md](improve.md))

**Что сработало**:
- Заметное улучшение на **ibm01** (-0.89%): congestion=1.284→1.262 (-1.7%), density=без изменения, WL=без изменения. Congestion действительно снижается на маленьком дизайне, где hot-cells более локализованы.
- Build чистый, smoke 9/9, 0 overlaps, fallback на random при отсутствии overlap корректно работает.
- Реализация **детерминирована** (3 запуска идентичны).

**Что не сработало**:
- На больших ibm10/14/17 эффект **в шуме (~±0.05%)**. Гипотеза: при большом количестве макросов (387-517) top-5% hot-cells не локализуют bottleneck — 30 LNS-итераций × 4 destroy = 120 макросов перемещено из ~500, недостаточно для глобальной разгрузки.
- Чередование с random может разводить улучшения (random destroy «возвращает» макросы обратно в hot-зону).

**Что попробовать дальше**:
- Чисто congestion destroy (не чередовать). Сравнить.
- Увеличить destroy_size для больших дизайнов (4 → 8-12).
- Larger LNS budget на больших (60-100 итераций вместо 30).
- "Repair away from hot": в spiral search избегать high-cong cells, не просто двигать в любую non-overlap позицию.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`
**Reviewer verdict**: ACK (5 minor notes — кэш macroCong неиспользуется, nth_element вместо sort, namedconst для 0.05, last_was_accepted флаг, унификация fallback shape).

---

### #4 · 2026-05-03 · `submissions/straple/placer.py` (pure C++: placer_core + proxy_cost)

**Что менялось от #3:**
- Реализован полный `compute_proxy_cost` на C++ (`cpp/proxy_cost.cpp`):
  - `get_wirelength`: HPWL по pin bbox каждого нета
  - `get_density_cost`: top-10% самых плотных grid-ячеек
  - `get_congestion_cost`: routing 2/3/N-pin + macro-route-over-grid + smoothing + ABU top-5%
- Python теперь только тонкая обёртка: извлекает pin connectivity из plc и передаёт в `_proxy_cost.ProxyEvaluator`
- LNS inner loop звонит `evaluator.evaluate(positions)` напрямую в C++ — без вызовов Python plc

**Результаты:**
- AVG proxy: **1.5181** (бит-в-бит совпадает с #3, т.е. наш C++ exact replicates plc.compute_proxy_cost)
- Total runtime: **16.19s** (vs 173.81s в #3 = **10.7× быстрее**)
- Best ibm01: **0.34s** (vs 3.4s)
- Slowest ibm17: 2.15s (vs 24.5s)

**Как это влияет на стратегию:**
- Раньше LNS ограничен ~30 itters (45с/bench overhead). Теперь можно гонять 300+ itters за то же время.
- Можно делать multi-restart с 5-10 сидов за разумное время.
- Можно делать proxy-aware SA (proxy_cost внутри inner SA loop).

**Команда:** `$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all`

---

### #3 · 2026-05-03 · `submissions/straple/placer.py` (C++ core via pybind11)

**Сводка:**
- AVG proxy: **1.5181**
- Best: **1.1781** на `ibm01` (246 макросов)
- Worst: **1.7944** на `ibm06` (318 макросов)
- Total runtime: **173.81s**
- Overlaps: 0 на всех ✅
- vs SA baseline: **+28.6%** (better)
- vs RePlAce baseline: **-4.1%** (хуже, но ближе чем will_seed)
- vs will_seed: **-1.01%** (lower is better)

**Архитектура:**
- C++ ядро (`submissions/straple/cpp/placer_core.cpp`) — legalize, SA refinement, LNS destroy/repair
- Python обёртка — extract edges из плк, оркестрирует пайплайн, оценивает proxy_cost через TILOS plc
- pybind11 bindings, компилируется через `cpp/build.sh`
- monkey-patch `plc.{soft,hard}_macro_pin_indices` → set (28× ускорение `compute_proxy_cost`)

**Параметры:**
- refine_iters=3000 (SA на HPWL)
- lns_outer_iters=30
- lns_destroy_size=8 (random destroy), repair=weighted-centroid + spiral search

**Per-benchmark детали:**

| Benchmark | Macros | Proxy | WL | Density | Congestion | Time | vs SA | vs RePlAce |
|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | **1.1781** ⭐ | 0.073 | 0.927 | 1.284 | 3.4s | +10.5% | -18.1% |
| ibm02 | 254 | 1.6316 | 0.078 | 0.809 | 2.299 | 5.1s | +14.5% | **+11.2%** ⭐ |
| ibm03 | 269 | 1.4148 | 0.081 | 0.841 | 1.827 | 4.6s | +18.7% | -7.0% |
| ibm04 | 285 | 1.4092 | 0.074 | 0.874 | 1.797 | 4.6s | +6.3% | -8.2% |
| ibm06 | 318 | 1.7944 | 0.068 | 0.834 | 2.619 | 4.5s | +28.4% | -10.9% |
| ibm07 | 335 | 1.5736 | 0.067 | 0.948 | 2.066 | 6.3s | +22.2% | -7.5% |
| ibm08 | 352 | 1.5037 | 0.069 | 0.871 | 1.998 | 7.0s | +21.8% | -5.3% |
| ibm09 | 369 | 1.1928 | 0.059 | 0.944 | 1.324 | 5.4s | +14.0% | -6.6% |
| ibm10 | 387 | 1.3843 | 0.069 | 0.716 | 1.915 | 18.1s | +34.4% | **+7.8%** ⭐ |
| ibm11 | 405 | 1.2855 | 0.056 | 0.965 | 1.495 | 7.6s | +24.9% | -9.2% |
| ibm12 | 423 | 1.6679 | 0.061 | 0.824 | 2.390 | 17.9s | +41.0% | **+3.4%** ⭐ |
| ibm13 | 441 | 1.4258 | 0.055 | 0.935 | 1.808 | 9.9s | +25.5% | -6.8% |
| ibm14 | 460 | 1.6280 | 0.053 | 1.000 | 2.150 | 16.6s | +28.4% | -5.5% |
| ibm15 | 479 | 1.6349 | 0.060 | 0.965 | 2.185 | 11.4s | +28.9% | -7.8% |
| ibm16 | 498 | 1.5458 | 0.050 | 0.895 | 2.097 | 15.6s | +30.8% | -4.6% |
| ibm17 | 517 | 1.7451 | 0.054 | 0.952 | 2.430 | 24.5s | +52.5% | -6.1% |
| ibm18 | 537 | 1.7916 | 0.054 | 1.041 | 2.435 | 11.3s | +35.5% | -1.1% |
| **AVG** | — | **1.5181** | — | — | — | **174s total** | **+28.6%** | **-4.1%** |

⭐ — обходит RePlAce baseline (3 из 17: ibm02, ibm10, ibm12). Will_seed обходил те же 3.

**Что менялось от will_seed:**
1. Та же архитектура (legalize + SA + опц. LNS) — но переписана на C++ (pybind11)
2. Добавлена outer LNS-фаза: 30 итераций × destroy 8 random + weighted-centroid repair + accept by full proxy_cost
3. Monkey-patch `pin_indices` → set: устранил O(N) lookup внутри `compute_proxy_cost`, дав ~28× ускорение

**Что сработало:**
- LNS post-processing спрятал ~1% улучшения над чистым SA (особенно сильно на ibm01: 1.2920 → 1.1781 = -8.8%)
- C++ переписывание SA: ~10× быстрее inner SA loop → можем себе позволить full 30 LNS iters везде

**Что не сработало (отвергнуто в ходе сессии):**
- SA-acceptance в LNS — ломает greedy улучшения
- Variable destroy + random repair — не консистентно
- Soft macro centroid update — массивная регрессия (WL)
- Spectral seed (Laplacian eigenvectors) — гораздо хуже initial
- `plc.optimize_stdcells()` — слишком медленно даже с маленьким budget

**Команда:**
```bash
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all
```

---

### #2 · 2026-05-03 · `submissions/will_seed/placer.py` (reference)

**Сводка:**
- AVG proxy: **1.5336**
- Best: **1.1625** на `ibm09` (369 макросов)
- Worst: **1.7921** на `ibm18` (537 макросов)
- Total runtime: **34.74s**
- Slowest benchmark: `ibm17` — 6.05s
- Overlaps: 0 на каждом бенчмарке ✅
- vs SA baseline: +27.8% (better)
- vs RePlAce baseline: -5.2% (хуже)

**Per-benchmark детали:**

| Benchmark | Macros | Proxy | WL | Density | Congestion | Time | vs SA | vs RePlAce |
|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 1.2920 | 0.074 | 1.047 | 1.390 | 3.23s | +1.9% | -29.5% |
| ibm02 | 254 | 1.6798 | 0.077 | 0.839 | 2.366 | 1.18s | +11.9% | **+8.6%** ⭐ |
| ibm03 | 269 | 1.4043 | 0.080 | 0.830 | 1.818 | 0.87s | +19.3% | -6.2% |
| ibm04 | 285 | 1.4478 | 0.073 | 0.907 | 1.842 | 1.54s | +3.7% | -11.2% |
| ibm06 | 318 | 1.7965 | 0.069 | 0.819 | 2.636 | 0.81s | +28.3% | -11.0% |
| ibm07 | 335 | 1.5903 | 0.067 | 0.978 | 2.069 | 0.84s | +21.4% | -8.7% |
| ibm08 | 352 | 1.5877 | 0.071 | 0.925 | 2.108 | 1.56s | +17.5% | -11.1% |
| ibm09 | 369 | **1.1625** ⭐ | 0.059 | 0.889 | 1.318 | 0.93s | +16.2% | -3.8% |
| ibm10 | 387 | 1.4116 | 0.071 | 0.752 | 1.929 | 4.06s | +33.1% | **+6.0%** ⭐ |
| ibm11 | 405 | 1.2547 | 0.055 | 0.921 | 1.479 | 1.24s | +26.7% | -6.6% |
| ibm12 | 423 | 1.6528 | 0.060 | 0.811 | 2.374 | 2.80s | +41.5% | **+4.2%** ⭐ |
| ibm13 | 441 | 1.4113 | 0.054 | 0.920 | 1.794 | 1.48s | +26.3% | -5.7% |
| ibm14 | 460 | 1.6515 | 0.053 | 1.013 | 2.184 | 2.97s | +27.4% | -7.0% |
| ibm15 | 479 | 1.6379 | 0.059 | 0.976 | 2.181 | 1.82s | +28.8% | -8.1% |
| ibm16 | 498 | 1.5484 | 0.050 | 0.893 | 2.103 | 1.91s | +30.7% | -4.8% |
| ibm17 | 517 | 1.7493 | 0.054 | 0.956 | 2.434 | **6.05s** ⬅ | +52.4% | -6.4% |
| ibm18 | 537 | 1.7921 | 0.054 | 1.042 | 2.435 | 1.44s | +35.4% | -1.1% |
| **AVG** | — | **1.5336** | — | — | — | **34.74s total** | **+27.8%** | **-5.2%** |

⭐ — обходит RePlAce baseline на этом конкретном бенчмарке (4 из 17).

**Декомпозиция cost (усреднено):**

| Компонент | Вклад в AVG | % от total |
|---|---|---|
| Wirelength (вес 1.0) | ~0.064 | ~4% |
| Density × 0.5 | ~0.92 × 0.5 = 0.46 | ~30% |
| Congestion × 0.5 | ~2.04 × 0.5 = **1.02** | **~66%** |

→ **Congestion — основной источник стоимости.** WL уже хорошо оптимизирован SA-фазой, density на среднем уровне, но congestion остаётся высоким. LNS должен фокусироваться на разгрузке congestion-hotspot'ов.

**Заметные паттерны:**
- На больших benchmark'ах (ibm12-ibm18, 423-537 макросов) — стабильно высокий congestion (>2.0), стабильно проигрываем RePlAce
- На маленьких (ibm01, 246 макросов) — proxy 1.29, разрыв с RePlAce 30% (RePlAce там 0.99)
- ibm02, ibm10, ibm12 — единственные где **обходим RePlAce** (на 4-9%) — стоит понять, почему именно эти
- ibm17 (517 макросов) самый дорогой по времени — 6 секунд из 34.74

**Чтение команды:**
```bash
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py --all
```

---

### #1 · 2026-05-03 · `submissions/examples/greedy_row_placer.py` (демо)

**Сводка:**
- AVG proxy: **2.2109** (точно как в leaderboard)
- Best: 1.6728 на `ibm09`
- Worst: 2.7696 на `ibm12`
- Total runtime: **0.05s**
- Overlaps: 0 на каждом бенчмарке ✅
- vs SA baseline: -4.0% (хуже)
- vs RePlAce baseline: -51.7% (сильно хуже)

**Per-benchmark детали:**

| Benchmark | Proxy | WL | Density | Congestion | vs SA | vs RePlAce | Overlaps |
|---|---|---|---|---|---|---|---|
| ibm01 | 2.0463 | 0.121 | 1.245 | 2.606 | -55.4% | -105.1% | 0 |
| ibm02 | 2.0431 | 0.092 | 1.085 | 2.818 | -7.1% | -11.2% | 0 |
| ibm03 | 2.1484 | 0.111 | 1.075 | 2.999 | -23.5% | -62.5% | 0 |
| ibm04 | 1.9321 | 0.090 | 0.973 | 2.711 | -28.5% | -48.3% | 0 |
| ibm06 | 2.1620 | 0.076 | 1.044 | 3.127 | +13.7% | -33.6% | 0 |
| ibm07 | 2.1144 | 0.084 | 1.124 | 2.936 | -4.5% | -44.5% | 0 |
| ibm08 | 2.4704 | 0.095 | 1.194 | 3.557 | -28.4% | -72.9% | 0 |
| ibm09 | **1.6728** ⭐ | 0.079 | 1.098 | 2.089 | -20.6% | -49.4% | 0 |
| ibm10 | 2.4523 | 0.100 | 1.026 | 3.678 | -16.2% | -63.4% | 0 |
| ibm11 | 2.0604 | 0.075 | 1.165 | 2.806 | -20.4% | -75.0% | 0 |
| ibm12 | **2.7696** ⬅ | 0.095 | 1.090 | 4.258 | +2.0% | -60.5% | 0 |
| ibm13 | 2.1004 | 0.074 | 1.102 | 2.950 | -9.7% | -57.3% | 0 |
| ibm14 | 2.4473 | 0.088 | 1.098 | 3.621 | -7.6% | -58.5% | 0 |
| ibm15 | 1.9488 | 0.077 | 1.050 | 2.694 | +15.3% | -28.6% | 0 |
| ibm16 | 2.5767 | 0.069 | 1.065 | 3.949 | -15.4% | -74.3% | 0 |
| ibm17 | 2.5411 | 0.067 | 1.093 | 3.856 | +30.8% | -54.5% | 0 |
| ibm18 | 2.0998 | 0.062 | 1.094 | 2.982 | +24.3% | -18.5% | 0 |
| **AVG** | **2.2109** | — | — | — | **-4.0%** | **-51.7%** | 0 |

**Что показывает:** Shelf-pack даёт самый маленький WL из всех (0.05-0.12, потому что компактно укладывает), но **density и congestion катастрофические** — макросы не учитывают netlist, всё запихивается в нижнюю часть канваса.

**Чтение команды:**
```bash
$HOME/.local/bin/uv run evaluate submissions/examples/greedy_row_placer.py --all
```

---

## 🏅 Где мы в leaderboard прямо сейчас

Если бы сабмитили текущий best (will_seed 1.5336):

```
...
| 24 | "#5 ubc cpen student" (Gene Pool Shuffle) | 1.5337 | ...
| 25 | Will Seed (Partcl) | 1.5336 | ⬅ МЫ
| 26 | "UT Austin" - RH (DREAMPlace) | 1.6037 | ...
...
| — | RePlAce (baseline) | 1.4578 | ⬅ хотим перепрыгнуть
...
| — | SA (baseline) | 2.1251 |
```

Нас обошли бы **24 команды**. Чтобы выйти в плюс относительно RePlAce — нужно улучшение хотя бы на **5.2%**.

### Ближайшие соседи и что они делают

| Место | Команда | Score | Подход (если известен) |
|---|---|---|---|
| 22 | SEVmakers | 1.5200 | Hybrid Legalization + SA |
| 23 | "CA" (congestion_aware) | 1.5247 | Congestion-aware |
| 24 | UBC student | 1.5337 | Gene Pool Shuffle |
| **25** | **Will Seed (мы)** | **1.5336** | SA refinement |
| 21 | oracleX | 1.5130 | — |
| 20 | UTAUSTIN-CT | 1.5062 | PLC-Exact Congestion-Aware SA |
| 19 | Jiangban Ya | 1.4943 | Spectral-Seed + Adaptive Legalizer |
| 18 | W3 Solutions | 1.4824 | GRACE |
| **—** | **RePlAce baseline** | **1.4578** | ⬅ цель минимум |

Чтобы достичь **топ-10 (1.4076)** — нужно улучшение на 8.9%.
Чтобы попасть в **топ-7 (Гран-при, ~1.348)** — на 12.7%.

---

## 🔬 Гипотезы / на что смотреть дальше

### 1. Congestion — главный bottleneck

В декомпозиции will_seed: ~66% AVG proxy идёт от congestion. SA в will_seed оптимизирует HPWL (wirelength), но congestion напрямую не таргетит → его не сильно понижает.

**Идея:** добавить в LNS-фазу destroy operator, который **атакует congested-области** — снимает макросы из routing-hotspot'ов и переразмещает их в менее загруженные зоны.

### 2. Большие benchmark'ы (ibm15-ibm18) — стабильное проседание

На больших дизайнах will_seed почти не догоняет RePlAce (-7..-1%), на маленьких сильнее (-29% на ibm01). Скорее всего, SA с 3000 итераций недостаточно для исследования пространства 500+ макросов.

**Идея:** на больших benchmark'ах — **больше итераций** или **multi-start**. У нас runtime запас огромный (1 час vs 6 секунд).

### 3. ibm02, ibm10, ibm12 — обходим RePlAce даже простым SA

Это маленькая статистика, но интересно понять, что в их структуре нетлиста делает SA эффективным.

**Идея:** проанализировать графы netlist'ов этих benchmark'ов — может быть, у них специфическая топология (мало больших нетов? кластеры?), и под неё подстроить наш алгоритм.

### 4. Runtime запас 600× — есть пространство для экспериментов

ibm17 (worst case) = 6 секунд из 3600 секунд лимита. Можем гонять multi-start с 100 разными seeds, или делать adaptive ALNS с долгой историей operator weights, или добавить RL-bandit на operator selection.

---

## 📐 Технические детали

**Окружение:**
- OS: macOS (darwin), aarch64 (Apple Silicon)
- Python: 3.14.4
- uv: 0.11.8
- torch: 2.10.0 (CPU build, MPS не используется will_seed'ом — он на numpy)
- numpy: 2.4.2
- macro-place: 0.1.0 (editable из репо)

**Нюансы железа:**
- Single-threaded numpy в will_seed → используется ~1 ядро
- Eval-машина (16 ядер EPYC) → если distribute through `multiprocessing` → можно ускорить linearly
- Mac ARM — нет CUDA → DRP интеграция нужна на Linux/CUDA-машине

**Команды для воспроизведения:**
```bash
# Текущий best (will_seed)
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py --all

# Демо (greedy)
$HOME/.local/bin/uv run evaluate submissions/examples/greedy_row_placer.py --all

# Один benchmark (быстрая отладка)
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py -b ibm01 --vis

# Smoke tests
$HOME/.local/bin/uv run pytest test/test_smoke.py -v
```

---

## 📋 Шаблон для будущих записей

```markdown
### #N · YYYY-MM-DD · `путь/к/placer.py` (короткое описание)

**Сводка:**
- AVG proxy: X.XXXX
- Best: X.XXXX на ibmXX
- Worst: X.XXXX на ibmXX
- Total runtime: X.XXs
- Overlaps: 0
- vs SA: ±X%
- vs RePlAce: ±X%

**Что менялось от прошлого прогона:**
- ...

**Что сработало:**
- ...

**Что не сработало:**
- ...

**Per-benchmark детали:** [таблица как у will_seed выше]

**Команда:** `uv run evaluate ... --all`

---

## Cycle #28 — DreamPlace++ recipe (ANCHOR_SOFT cluster init), 2026-05-05

### Что сделано (новое)

1. **`submissions/straple/clustering.py`** — netlist hypergraph clustering:
   - Clique expansion гиперсетей (вес 1/(k-1) на пару, nets > 20 пинов отброшены)
   - networkx Louvain partitioning, поддержка target_num_clusters через бинарный поиск resolution
   - Anchor distribution: uniform grid или centroid от initial pos
2. **`gradient_demo.py::init_mode=anchor_soft`** — все макросы spawn вокруг anchor своего кластера с малым Gaussian шумом.
3. **`visualizer.py`** — каждый макрос получает `cluster` ID, цвет по HSV-палитре (golden ratio шаг для разделения), кнопка переключения "color: cluster ↔ kind", hotkey `c`.
4. **Adaptive defaults**: `STRAPLE_DEMO_CLUSTER_TARGET=auto` → max(15, n//30); `STRAPLE_DEMO_TARGET_UTIL=auto` → actual_util * 0.95.

### Результаты (ibm01, multi-seed × K-grid sweep)

| Конфиг | proxy | wl | den | cong | runtime |
|---|---|---|---|---|---|
| INITIAL .plc | 1.0385 | 0.064 | 0.812 | 1.137 | — (69 ovrlp) |
| Submitted ALNS (best of 3 starts × 50K LNS iters) | 1.0584 | 0.072 | 0.840 | 1.133 | ~25 мин |
| gradient_demo center init (default, 300 iter) | 1.6868 | 0.127 | 0.991 | 2.129 | 3.85с |
| **gradient_demo anchor_soft K=40 r=0.05 (best of 6 seeds)** | **1.1050** | 0.084 | 0.745 | 1.297 | 14с |

**Ключевой результат**: pure-gradient pipeline за **14 секунд** vs ALNS submission **25 минут** — proxy всего на +4.4%, без всякого LNS polish'а.

K-grid sweep (seed=42, ibm01, 400 iters):
| K \ radius_frac | 0.01 | 0.04 | 0.05 | 0.06 | 0.10 |
|---|---|---|---|---|---|
| 8 | — | — | 1.31 | — | — |
| 10 | 1.47 | — | 1.28 | — | — |
| 20 | — | — | 1.25 | — | — |
| 30 | 1.17 | — | 1.14 | — | — |
| **40** | — | 1.11 | **1.10** | 1.13 | 1.16 |
| 50 | — | — | 1.22 | — | — |

Optimum: K=40 (n_total/30=38), spawn_radius=0.05*canvas_min. Шум ~±0.05 от seed.

Multi-seed для best config (K=40, r=0.05, ibm01, 400 iter):
seeds 7,13,21,42,100,999 → proxy 1.1093, 1.1328, 1.2164, 1.2400, 1.2582, 1.1445 (mean 1.183, **best 1.109**).

### Per-bench результаты (config: ITERS=800 lambda_max=1000 target_util=0.85, кроме ibm01)

| bench | n_hard | n_soft | proxy | RePlAce | vs RePlAce | runtime | HTML |
|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 894 | **1.1050** | 0.998 | +10.7% | 14s | [ibm01_anchor_soft_best.html](../vis/ibm01_anchor_soft_best.html) |
| ibm04 | 295 | 1085 | **1.4847** | 1.302 | +14.0% | 38s | [ibm04_anchor_soft_v2.html](../vis/ibm04_anchor_soft_v2.html) |
| ibm10 | 786 | 1982 | **1.8726** | 1.501 | +24.8% | 60s | [ibm10_anchor_soft_v2.html](../vis/ibm10_anchor_soft_v2.html) |
| ibm14 | 614 | 1529 | **1.9061** | 1.544 | +23.5% | 80s | [ibm14_anchor_soft_v2.html](../vis/ibm14_anchor_soft_v2.html) |
| ibm17 | 760 | 1844 | **1.8863** | 1.645 | +14.7% | 100s | [ibm17_anchor_soft_v2.html](../vis/ibm17_anchor_soft_v2.html) |
| **AVG5** | — | — | **1.6509** | **1.398** | +18.1% | — | — |

### Ключевые insights

1. **ANCHOR_SOFT клик**: pure-gradient на ibm01 за 14с матчит ALNS (25 мин) с +4.4%. Это **самый большой выигрыш сессии**.
2. **K = n_total / 30** оптимально на ibm01. Логично: средний размер кластера ~30 макросов = 1 anchor cell на canvas ~6×6 grid.
3. **Spawn radius 0.05 * canvas_min** оптимум: меньше — кучкуется, больше — теряется кластерная структура.
4. **target_util должен быть близко к actual_util** (~0.8 на ICCAD04). Default 0.4 даёт unsolvable loss → макросы кучкуются.
5. **Большие бенчмарки нужны более агрессивный λ schedule** (lambda_max=1000+) — иначе density penalty слишком слабый чтобы распределять макросы.
6. **Плохая новость**: для ibm04+ pure-gradient далек от RePlAce (+14%). Видимо нужен либо real congestion-aware loss, либо electrostatic potential model (ePlace), либо post-LNS polish.

### Сохранённые HTML-визуализации

В папке `vis/` (открыть в браузере, hover macros для tooltip, `c` переключает color cluster↔kind, ←/→ кадры):
- `ibm01_anchor_soft_best.html` — best K=40 r=0.05 (proxy 1.105)
- `ibm01_anchor_soft_v1.html` — first run K=20 (proxy 1.245)
- `ibm04_anchor_soft.html` / `_v2.html` — старая/новая конфиг
- `ibm10_anchor_soft.html` — first run (default config, proxy 2.18)
- `ibm14_anchor_soft.html` — proxy 2.51 (default)
- `ibm17_anchor_soft.html` — proxy 2.29 (default)

### Что попробовать дальше (не сделано в сессии)

1. **Multi-seed averaging внутри placer.py** — best-of-N seeds для submission
2. **post-gradient LNS polish** — gradient seed + наш ALNS на нём (combine best of both)
3. **Congestion-aware loss** — добавить smooth net bbox + grid demand penalty per net
4. **Electrostatic potential** (ePlace formulation) — заменить bell-curve density на FFT-based Poisson
5. **Per-cluster anchor displacement loss** — anchor remains stationary, members can drift; стабилизирует структуру

---

## Cycle #29 — GPU server (T4) + ALNS hybrid + drastic perturbation, 2026-05-05

### Setup
- GPU server: Intel Ice Lake + Tesla T4 (16 vCPU, 64GB RAM, 128GB disk).
- run_remote.sh переделан под новый сервер: bootstrap, push, eval, gpu, sweep.
- bootstrap ставит uv, deps, NVIDIA driver-535 (modprobe без reboot), torch CUDA 12.1.
- C++ build (placer_core, proxy_cost) — fix Linux compat (-undefined dynamic_lookup только для Mac).

### Что сделано (новое)

1. **`submissions/straple/gradient_demo.py`** — добавлен device='cuda' через STRAPLE_DEMO_DEVICE.
   Все tensors переносятся на GPU. Backward compat (cpu default).
2. **`submissions/straple/gradient_batch.py`** — true K-parallel batch [K, n, 2] для GPU.
   Все loss components (HPWL, density, overlap) vectorized over K dim. Один Adam.step
   обновляет все K seeds. **13× speedup** vs sequential CPU на ibm01 (583ms vs 7.8s per seed).
3. **`scripts/gpu_batch_search.py`** — runner для gradient_batch + per-K eval + C++ legalize
   для top-N candidates. Сохраняет best в `results/gpu_seed_<bench>.pkl`.
4. **`submissions/straple/placer.py::STRAPLE_GRADIENT_SEED_FILE`** — load external pre-computed
   gradient seed (от gpu_batch_search), feed в ALNS как extra start.
5. **`STRAPLE_PRESET=high_effort`** — adaptive preset с drastic perturbation (sigma 0.10..0.50),
   16 starts на small benches, 12-8 для big, big LNS budget.

### Метрики (GPU vs CPU, ibm01)

| Mode | Time | Notes |
|---|---|---|
| Sequential CPU 1 seed gradient_demo | 7.8s | baseline |
| Sequential GPU 1 seed gradient_demo | 11s | overhead launch >> compute, slower |
| **Batch GPU K=64 (gradient_batch)** | **37s** | **13× speedup vs CPU** (583ms/seed) |
| Batch GPU K=64 ibm17 | 600s | 9.5s/seed (n=2604, big benches scale linearly) |

GPU **рабочий** только для **batch** mode. Sequential GPU медленнее CPU из-за launch overhead.

### Pure-gradient sweep results (GPU batch K=64, with C++ legalize top-8)

| bench | INITIAL (with overlaps) | gpu_batch best | ALNS baseline | gap vs ALNS |
|---|---|---|---|---|
| ibm01 | 1.0385 | 1.1139 | 1.0483 | +6.3% |
| ibm04 | 1.3133 | 1.4273 | 1.2822 | +11.3% |
| ibm10 | 1.3397 | 1.4181 (tuned) | 1.2434 | +14.0% |

Pure-gradient + C++ legalize **не превосходит ALNS** на этих benches. ALNS уже хорошо
оптимизирован.

### ALNS hybrid (gradient seed → ALNS polish, ibm01)

ALNS на gradient seed = **1.0714** vs baseline ALNS = **1.0483** (хуже на 2.2%).
ALNS застряла в gradient'овском basin. На больших benches не помогло аналогично.

### 🏆 Главный win: PRESET=high_effort (drastic perturbation 16 starts)

| bench | baseline ALNS | high_effort | Δ |
|---|---|---|---|
| ibm01 | 1.0483 | **1.0432** | **−0.5%** |
| ibm04 | 1.2822 | **1.2643** | **−1.4%** |
| ibm10 | 1.2434 | TBD | — |
| ibm14 | TBD | TBD | — |
| ibm17 | TBD | TBD | — |

high_effort = `num_starts=4 + perturbed=12` (sigma scales 0.10..0.50) + `LNS factor=120 cap=80000`
+ adaptive по размеру bench. Runtime растёт ~3.5× (но в пределах 1ч лимита per bench).

### Insights

1. **GPU полезна только для batch ops**. Single-run = overhead launch dominates.
2. **gradient_batch** = единственная realistic GPU нагрузка. K=64 даёт реальный speedup.
3. **Pure-gradient still уступает ALNS** на этих benches. Surrogate ≠ true objective.
4. **ALNS уже near-optimal**. Gradient seeds не помогают (basin trap).
5. **Drastic perturbation work** — это путь к -1..-4% AVG17.

```
