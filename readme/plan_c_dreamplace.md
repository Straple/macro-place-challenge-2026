# План C: DREAMPlace + GPU + Tier 2 ORFS — путь к Гран-при $20K

> Промпт для следующей сессии. Цель — пробить **топ-7** (≤1.3479) по AVG17 proxy для квалификации на **Гран-при ($20K)**, затем выиграть Tier 2 на NG45 designs.

---

## ⚠️ Прочитать ПЕРЕД началом

1. **[results.md](results.md)** — полный журнал 25+ циклов оптимизации, включая что попробовали и не сработало.
2. **[todo.md](todo.md)** — анализ leaderboard, текущее состояние, выводы.
3. **[improve.md](improve.md)** — общие правила автономного цикла улучшений.
4. **[PROBLEM.md](PROBLEM.md)** — формальная постановка задачи.
5. **[SCORING.md](SCORING.md)** и **[../SCORING.md](../SCORING.md)** — правила Tier 2.
6. **[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)** — чеклист до сабмита.

**Код**:
- `submissions/straple/placer.py` — Python обёртка (текущая версия с ALNS + parallel multi-start через `STRAPLE_PARALLEL_STARTS` env var)
- `submissions/straple/cpp/placer_core.cpp` — C++ ядро (legalize, SA, LNS operators: rand/cong/swap/cluster, ALNS adaptive weights, shake-up)
- `submissions/straple/cpp/proxy_cost.cpp` — pure C++ replica `plc.compute_proxy_cost`
- `submissions/straple/analytical_seed.py` — наш собственный gradient placer (Python+torch). **Не работает** в текущем pipeline (после legalize теряется structure). Можно использовать в Plan C как fallback.
- `scripts/fast_check.py` — parallel runner для всех 17 IBM benchmarks (12 workers, ~32 мин wall)

**Текущее submission**: AVG17 = **1.4445** (commit `c758df2`), ~16 место в leaderboard.
- vs RePlAce (1.4578): -0.91% ✅
- vs Top-10 (1.4076): +2.62%
- vs Top-7/Гран-при (1.3479): **+7.17%** ← цель пробить

---

## 0. Контекст: почему План C

После 25 циклов чистого ALNS улучшения мы упёрлись в **~1.42-1.45** — это место где сидят все CPU-only LNS-based решения (TAISPlAce, ArzunPD, vmallela, мы). Дальше **без gradient-based seed не пробить**.

**80% top-10 используют DREAMPlace или ePlace-style analytical placer**. Recipe у всех топов:
```
DREAMPlace/ePlace seed → min-displacement legalize → multi-start hyperparam sweep → SA/LNS polish
```

**Наш собственный analytical_seed.py не работает** в текущем pipeline (после legalize структура разрушается). Реальный DREAMPlace — другая лига качества.

**План C цель**: пробить топ-7 (≤1.3479) для квалификации на Tier 2, затем выиграть Tier 2 на NG45 designs (Гран-при).

---

## 1. Цели и ожидания

| Уровень | AVG17 | Δ от 1.4445 | Что нужно |
|---|---|---|---|
| 🎯 Топ-13 | 1.4321 | -0.86% | parallel multi-start + diverse seeds |
| 🎯 Топ-10 | 1.4076 | -2.55% | **plain DREAMPlace** (UT Austin AS подход) |
| 🎯 Топ-8 (Archgen) | 1.3479 | -6.69% | DREAMPlace + multi-start hyperparam sweep + LNS polish |
| 🎯 **Топ-7 = Гран-при квалификация** | **≤1.3479** | **-6.69%** | DREAMPlace + tuning |
| 🎯 Win Гран-при | best WNS/TNS/Area на NG45 | — | + ORFS tuning |

Реалистичная цель плана C: **AVG17 ≈ 1.34-1.38** (топ-7..топ-9). Win Гран-при дополнительно требует ORFS calibration на NG45 (3-5 дней).

---

## 2. Архитектура целевого решения

```
┌─────────────────────────────────────────────────────────┐
│ StraplePlacer.place(benchmark) — финальная архитектура  │
└─────────────────────────────────────────────────────────┘
        │
        ├─→ Phase 1: Multi-start gradient placement
        │   ├─ Worker 1: DREAMPlace (config A)
        │   ├─ Worker 2: DREAMPlace (config B — другие hyperparams)
        │   ├─ Worker 3: DREAMPlace (config C)
        │   ├─ Worker 4: DREAMPlace (config D)
        │   ├─ Worker 5: original initial pos (как baseline diversity)
        │   ├─ ...
        │   └─ Worker 16: ...
        │
        ├─→ Phase 2: Min-displacement legalize каждого
        │
        ├─→ Phase 3: Эвалюатор → выбор best 1-3 кандидатов
        │
        ├─→ Phase 4: LNS polish на best (наш существующий ALNS)
        │   - 4 operators (rand/cong/swap/cluster)
        │   - adaptive weights, shake-up
        │   - refine passes
        │
        └─→ Return best placement
```

**Time budget на bench**: 1 час (3600s). Распределение:
- DREAMPlace seed: ~60-300s × N starts (parallel) = wall ~300s
- Legalize: ~10-60s per candidate
- LNS polish: остаток ~30-50 мин

---

## 3. Фазы работ

### Фаза 1: Setup DREAMPlace (Day 1, ~6-8 часов)

**Цель**: запустить vanilla DREAMPlace standalone на одном бенчмарке (ibm01) и получить proxy cost.

#### 1.1 Установка DREAMPlace

DREAMPlace — https://github.com/limbo018/DREAMPlace (BSD-3). Документация: README.md в их репо.

Build options:
- **Linux + CUDA** (предпочтительно для eval-машины с RTX 6000 Ada): полные функции
- **Linux + CPU** (для Mac dev — НЕ работает на macOS, нужен Docker или Linux VM)
- **macOS не поддерживается** — для разработки нужен Docker

Шаги (Linux):
```bash
# Зависимости: cmake, boost, eigen, flex, bison, gcc, python3-dev, cuda
git clone https://github.com/limbo018/DREAMPlace external/DREAMPlace
cd external/DREAMPlace
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/install -DPYTHON_EXECUTABLE=$(which python3)
make -j8 install
# После: export PYTHONPATH=$PWD/install:$PYTHONPATH
```

**Альтернатива: использовать Docker** — у DREAMPlace есть готовый image:
```bash
docker pull limbo018/dreamplace:cuda
```

**Для eval-машины Partcl**: они дают 16-core EPYC + RTX 6000 Ada + 100GB RAM. CUDA build предпочтителен.

**Для Mac dev**: CPU build через Docker — для проверки логики до push на eval-машину.

#### 1.2 Найти как DREAMPlace принимает наш формат бенчмарка

DREAMPlace работает с **Bookshelf, LEF/DEF, или ICCAD2015** форматами. Наши IBM benchmarks (ICCAD04) идут в формате `.pb.txt` (TILOS PlacementCost) и `.plc`.

**Варианты конвертации**:
1. **DEF → DREAMPlace**: уже есть `macro_place/def_writer.py` — экспорт нашего benchmark в DEF. DREAMPlace умеет читать LEF/DEF.
2. **Bookshelf**: TILOS submodule может иметь конвертер `pb.txt → bookshelf`. Проверить `external/MacroPlacement/`.
3. **Direct API**: DREAMPlace внутри строит граф из своих data structures. Можно вызывать его API программно с нашими данными.

Лучший путь — **DEF**, потому что у нас уже есть writer.

#### 1.3 Запустить DREAMPlace на ibm01

```python
# example: один бенчмарк через subprocess
import subprocess
subprocess.run(["python3", "external/DREAMPlace/dreamplace/Placer.py",
                "--config", "configs/ibm01.json"], check=True)
# Затем: парсить output (DREAMPlace pos файл) → naш Tensor
```

Ожидаемый output: позиции макросов после analytical placement. **Без legalize** (или с DREAMPlace's own legalize).

#### 1.4 Сравнить с нашим current best

```bash
# Конвертировать DREAMPlace pos → наш формат
# Запустить compute_proxy_cost
# Должно дать ~1.30-1.45 на ibm01 (vs наш 1.0584)
```

### Фаза 2: Integration в наш pipeline (Day 2-3, ~10-15 часов)

#### 2.1 DREAMPlace как один из multi-start seeds

Текущий `submissions/straple/placer.py::StraplePlacer.place()` имеет infrastructure для parallel multi-start (env var `STRAPLE_PARALLEL_STARTS`). Расширить:

```python
def _run_one_start(self, start_idx, args, evaluator, plc):
    if start_idx == 0 and self.use_dreamplace:
        # Worker 0: DREAMPlace seed
        seed_pos = run_dreamplace(benchmark, plc, config=self.dp_configs[0])
    elif start_idx < len(self.dp_configs) and self.use_dreamplace:
        # Worker 1..N: DREAMPlace с разными configs (multi-start hyperparam sweep)
        seed_pos = run_dreamplace(benchmark, plc, config=self.dp_configs[start_idx])
    else:
        # Original initial pos (diversity baseline)
        seed_pos = args["initial_pos"]
    
    # Дальше как сейчас:
    state.initialize(seed_pos, ...)
    state.legalize_min_displacement(...)  # новый minimum-displacement legalize
    # SA?  для маленьких
    # LNS polish (наш existing)
    return positions, cost
```

#### 2.2 Min-displacement legalize

У нас уже есть `legalize_min_displacement(max_iters)` в `placer_core.cpp` (cycle #18 был добавлен для analytical seed, но не использовался). Использовать его вместо `legalize()` (который sort-by-size + spiral search) для DREAMPlace seeds — он сохраняет analytical structure.

#### 2.3 Multi-start hyperparameter sweep

DREAMPlace имеет hyperparameters (target_density, lr, num_bins, density_weight, etc.). Один seed возможно не оптимален. Рекомендация Electric Beatle / Archgen:
- N=4-8 разных configs per benchmark
- Запускать параллельно
- Выбирать best по proxy_cost

### Фаза 3: Tuning + verify (Day 4-5)

- Прогон `--all` через `scripts/fast_check.py --workers 16`
- Сравнить AVG17 с нашим 1.4445
- Если **AVG17 ≤ 1.34**: ✅ топ-7 квалификация
- Если AVG17 > 1.4: что-то не так с интеграцией, debug

### Фаза 4: Submission update (Day 5)

- Финальный `--all` в чистой среде
- Update Google form с новым SHA
- Verify судьями (~1 неделя)

### Фаза 5: Tier 2 ORFS (Day 6-12) — только если квалифицировались на топ-7

**Tier 2 = OpenROAD flow** на 4 NG45 designs. Метрики: WNS, TNS, Area.

#### 5.1 Setup OpenROAD ORFS

```bash
# OR-Tools / OpenROAD docker
docker pull openroad/orfs
# Прогнать наш placement через OpenROAD на одном NG45 design
cd scripts && python evaluate_with_orfs.py submissions/straple/placer.py --design ariane133
```

См. `scripts/evaluate_with_orfs.py` (уже есть).

#### 5.2 NG45 designs

- ariane133, ariane136, mempool_tile, nvdla — все 4 публичные
- + 1-2 hidden designs (только у судей)
- В `external/MacroPlacement/Flows/NanGate45/` — netlist + lib + .plc

#### 5.3 Метрика Tier 2

Из [SCORING.md](SCORING.md):
```
weighted_geomean(WNS_improvement, TNS_improvement, Area_improvement) с весами 3:2:1

R_WNS = WNS_avg_baseline / WNS_sub  (после фикса в issue #66)
similarly for TNS, Area
```

Цель: ALL three improvements > 0 (feasibility gate), затем максимизировать weighted geomean.

#### 5.4 Hard macro orientation

**Разрешено**: `N`, `FN`, `FS`, `S` (Klein-4 flips). **Запрещено**: `R90`, `R270`, `FE`, `FW`.

Для Tier 2 — оптимизация orientation может дать +1-3% на pin access (ORFS более чувствителен).

Передать orientations через optional `orientations.pt` sidecar (см. issue #66 в official repo).

---

## 4. Технические детали интеграции DREAMPlace

### 4.1 Зависимости

DREAMPlace deps (для CUDA build):
- CMake ≥ 3.16
- boost-dev (system, graph)
- eigen3
- flex, bison
- python3-dev (≥3.8)
- CUDA Toolkit ≥ 11.0 (с nvcc)
- pytorch (с CUDA)

**Важно**: добавить deps в `pyproject.toml` чтобы воспроизвелось у судей. Если DREAMPlace не pip-installable — придётся бандлить как submodule + build script.

**DQ-риск (как у Mike Gao)**: если DREAMPlace silent fails на eval-машине → 47-189 overlaps на benchmark → DQ. **Обязательно тестировать в чистой среде**.

### 4.2 Альтернатива: bundled DREAMPlace binaries

Если build сложный — собрать на Linux, положить `.so` файлы в `submissions/straple/` рядом с placer.py. Но размер репо может вырасти > 100MB (у DREAMPlace тяжёлые binaries). Проверить лимит на submission size.

### 4.3 Mac developer workflow

Mac не поддерживает CUDA, и compile DREAMPlace на macOS difficult.
- **Вариант 1**: Docker для local dev — медленно но работает
- **Вариант 2**: разрабатывать на Mac без DREAMPlace (используя наш analytical_seed.py как fake DREAMPlace), а DREAMPlace запускать на eval-машине отдельно
- **Вариант 3**: rent Linux GPU machine (vast.ai, Lambda) для разработки

### 4.4 Subprocess vs Python embedding

**Subprocess** (рекомендую):
- ✅ Изоляция: deps не конфликтуют
- ✅ Crash protection: DREAMPlace fail → catch → fallback на наш seed
- ❌ Overhead на startup (~5-30s)

**Embedding** (через `import dreamplace`):
- ✅ Faster (no startup)
- ❌ Dep hell: DREAMPlace torch version vs наш torch version

Subprocess лучше для robustness.

### 4.5 Output parsing

DREAMPlace output: обычно `output.pl` (Bookshelf-style) или `output.def`. Парсер:
```python
def parse_dreamplace_output(path):
    # Read positions for each macro
    # Map back to our benchmark macro indices (через имена)
    # Return numpy [n_hard, 2] array
```

Mapping macro names может быть сложным — DREAMPlace может ренеймить. Сохранять mapping table при конвертации в input.

---

## 5. Risk management

### 5.1 DQ risks (избегать)

1. **Silent fail в eval environment** (как Mike Gao с DREAMPlace): тестировать в чистом docker до сабмита.
2. **Import errors** (как BakaBobo): все deps в `pyproject.toml`.
3. **Hardware mismatch** (как vmallela -27%): запустить на arch близком к EPYC если возможно.
4. **Hardcode под benchmark name**: НЕ делать `if benchmark.name == "ibm17"`.
5. **Forbidden orientations** (R90, R270, FE, FW): убедиться что DREAMPlace их не использует, либо clipping post-processing.
6. **Soft macro resize**: размер soft macros фиксирован, не менять.
7. **Runtime > 1ч на benchmark**: max ibm17 у нас сейчас 28.7 мин, есть запас. Но DREAMPlace + multi-start + LNS могут дольше — мониторить.

### 5.2 Operational risks

1. **Не сломать current submission**: работать в branch, не на main, пока не проверено.
2. **Сохранить fallback на ALNS-only**: если DREAMPlace fails — использовать наш текущий best 1.4445 как fallback.
3. **Verify в чистой среде** до push: clone в /tmp, fresh `uv sync`, прогон `--all`.

### 5.3 Time risks

- **17 дней** до дедлайна (сегодня 2026-05-04, дедлайн 2026-05-21).
- Setup DREAMPlace: 1 день (если без проблем) — 3 дня (если build issues)
- Integration: 2-3 дня
- Tuning + verify: 2 дня
- Tier 2 (если квалификация): 5-7 дней
- **Buffer**: 3-5 дней на непредвиденное.

Если за 5 дней DREAMPlace не интегрирован — переключиться на План A (perturbed multi-start) для финального улучшения current submission.

---

## 6. Submission protocol

### 6.1 До каждого update

- [ ] Smoke tests: `uv run pytest test/test_smoke.py -v` → 10/10 PASS
- [ ] `scripts/fast_check.py` (4 бенча, ~30 мин) → AVG4 не хуже текущего best 1.3886
- [ ] Full `--all` в чистой среде → AVG17 не хуже current 1.4445 (или лучше)

### 6.2 Update form

- Получить SHA: `git rev-parse HEAD`
- Update submission на https://forms.gle/YDRtYV5Vq68SZgKW9
- Verify судьями ~1 неделя

### 6.3 Если всё ломается

Submission rollback:
```bash
git revert HEAD  # откатить bad commit
git push origin main
# Update form с предыдущим SHA c758df2 (наш безопасный baseline 1.4445)
```

---

## 7. Полезные ресурсы

### Документация и статьи

- **DREAMPlace** repo: https://github.com/limbo018/DREAMPlace
- **DREAMPlace paper**: Lin et al. DAC 2019 "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement"
- **AutoDMP** (Bayesian hyperparam tuning over DREAMPlace): https://github.com/NVlabs/AutoDMP, ISPD 2023 paper
- **ePlace 2014** (foundation): https://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
- **MOSAIC inspiration** (UToronto, top-5 на CPU 24 мин): no public paper, но описание в leaderboard

### Существующие placer'ы в нашем репо

- `submissions/will_seed/placer.py` — reference от Partcl, AVG17 1.5336 (наш baseline)
- `submissions/examples/greedy_row_placer.py` — простой demo, 2.21
- `submissions/straple/placer.py` — наш текущий best, 1.4445

### TILOS submodule

- `external/MacroPlacement/` — bookshelf benchmarks, scripts, etc.
- `external/MacroPlacement/CodeElements/Plc_client/` — plc API
- `external/MacroPlacement/Flows/NanGate45/` — NG45 designs для Tier 2

### Challenge official repo

- https://github.com/partcleda/partcl-macro-place-challenge
- Leaderboard: README.md в этом репо
- Issues / PRs — следить за clarification и updates

---

## 8. План first session (ко началу)

**Шаг 1** (1 час): прочитать всё что в section "⚠️ Прочитать ПЕРЕД началом", + этот файл целиком.

**Шаг 2** (1 час): склонить DREAMPlace в `external/DREAMPlace/`, посмотреть README, понять build requirements. Установить deps (на Linux/Docker).

**Шаг 3** (2-3 часа): попытаться собрать DREAMPlace. Если на Mac — через Docker. Если на Linux — нативно. Цель: запустить `python3 dreamplace/Placer.py --help`.

**Шаг 4** (1-2 часа): создать минимальный конвертер benchmark → DEF (использовать `macro_place/def_writer.py`). Скормить DEF в DREAMPlace, получить output.

**Шаг 5** (1 час): парсить DREAMPlace output, конвертировать в numpy array, посчитать `compute_proxy_cost` на ibm01. Цель: получить число < 1.4 (показать что это лучше current 1.0584? — нет, ibm01 уже хорош у нас. Лучше тестировать на ibm17 где у нас 1.7223 и DREAMPlace должен дать ~1.4).

**Шаг 6** (1-2 часа): зафиксировать infrastructure, документировать, commit.

Затем по плану в Section 3 (фазы 2-5).

---

## 9. Что НЕ делать

- ❌ Не модифицировать `macro_place/` (evaluator) — DQ.
- ❌ Не trying писать свой DREAMPlace с нуля — это месяцы. Использовать готовый.
- ❌ Не делать hardcoding под имена бенчмарков.
- ❌ Не ломать текущий submission `c758df2` (1.4445) — это safe baseline. Все эксперименты в branch или с easy revert.
- ❌ Не push'ить непроверенное в main.
- ❌ Не использовать R90/R270/FE/FW orientations.
- ❌ Не resize'ить soft macros.

---

## 10. Критерии успеха плана C

- ✅ **Минимум**: AVG17 < 1.4445 (улучшить current submission хоть на немного)
- ✅ **Реалистично**: AVG17 < 1.4076 (топ-10)
- ✅ **Цель Гран-при**: AVG17 ≤ 1.3479 (топ-7 квалификация)
- ✅ **Идеально**: AVG17 ≤ 1.32 + Tier 2 win = $20K Гран-при

Удачи. Если застрянешь на DREAMPlace build > 1 дня — switch на план B (свой analytical_seed.py + better legalize) или план A (perturbed parallel multi-start). Не упереться в single approach.
