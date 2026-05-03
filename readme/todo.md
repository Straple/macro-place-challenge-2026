# Macro Placement Challenge 2026 — План

> Личный todo / журнал для участия в Partcl × HRT Macro Placement Challenge.
> Команда: **Straple** (репо: `Straple/macro-place-challenge-2026`).

---

## 🚦 Текущий статус

| Метрика | Значение | Источник |
|---|---|---|
| **Phase** | Phase 0 завершена → начинаем Phase 1 (LNS skeleton) | — |
| **Окружение** | uv 0.11.8, Python 3.14.4, torch 2.10.0, submodule подтянут | этот ноут (Mac ARM) |
| **Best AVG proxy** | **1.5336** (will_seed reference) | [results.md](results.md) |
| **Будем 25/30** в leaderboard | Если бы сабмитили сейчас | — |
| **Gap до RePlAce** | -5.2% (нужно улучшить) | — |
| **Gap до топ-7** | -12.7% (Tier 2/Гран-при) | — |
| **Gap до топ-1** | -25.3% (Cezar 1.2224) | — |

**Полные результаты замеров:** [results.md](results.md).

---

## 0. Контекст и цели

| Что | Значение |
|---|---|
| **Дедлайн** | 21 мая 2026, 23:59 PT |
| **Призы** | $20K Grand / $20K Proxy / $5K 2nd / $4K Innovation + swag |
| **Tier 1** (proxy) | Среднее по 17 IBM benchmarks. RePlAce baseline = **1.4578** |
| **Tier 2** (Grand) | Топ-7 по proxy → ORFS на 4 NG45 designs (+ 1-2 hidden) → WNS:TNS:Area = 3:2:1 |
| **Лимит** | 1 час на benchmark, 16 ядер + 100 GB RAM + RTX 6000 Ada 48 GB |
| **Лицензия победителя** | Apache 2.0 / MIT (open-source обязателен) |

**Цели по уровням** (текущий best = will_seed 1.5336):

| Уровень | Цель | Прирост от текущего | Что даёт |
|---|---|---|---|
| 🎯 **Минимум** | proxy ≤ **1.4578** | **-5.2%** | Выше RePlAce baseline → есть место в leaderboard выше baseline'а |
| 🎯 Топ-10 | proxy ≤ 1.4076 | -8.9% | Видимая позиция в топе |
| 🎯 **Реалистично** | proxy ≤ **1.3479** | **-12.7%** | Топ-7 → **квалификация на Tier 2 / Гран-при** |
| 🎯 Топ-3-5 | proxy ≤ 1.32 | -14% | Серьёзная борьба за призы |
| 🎯 Амбициозно | proxy ≤ **1.2224** | -25.3% | Первое место (как Cezar) |

---

## 1. Анализ leaderboard — что работает у топов

Извлечено из [README.md:233-265](../README.md#L233). Сортировка по proxy cost (ниже = лучше).

| # | Команда | Score | Что делают |
|---|---|---|---|
| 1 | Cezar **(ReFine)** | 1.2224 | Verified, оспаривает результат — детали скрыты |
| 2 | MTK **(DreamPlace++)** | 1.2818 | DREAMPlace + улучшения. **GPU, 37s/bench** |
| 3 | RoRa (RipPlace) | 1.3241 | 694s/bench |
| 4 | UToronto **(MOSAIC)** | 1.3323 | Gradient-based + smooth surrogates, hard+soft вместе |
| 5 | Shoom **(MultiDREAMPlace)** | 1.3381 | Multi-start DREAMPlace + min-displacement legalization + SA |
| 6 | V5 **(TierPlace)** | 1.3382 | GPU, multi-density-formulation pilot + phased optimization |
| 7 | Archgen **(AutoDMP++)** | 1.3479 | Multi-start + fast proxy screening + bounded refinement |
| 8 | Beatel (ePlace-Lite) | 1.3913 | GPU, 155s/bench |
| 9 | Varun (GRPlace) | 1.4017 | 27s/bench |
| 10 | UT Austin AS | 1.4076 | DREAMPlace Analytical, 17s/bench |
| 11 | ByteDancer | 1.4151 | Incremental CD, 38min/bench |
| 12 | vmallela | 1.4152 | **Pure Python+numpy single-threaded**, Incremental CD+LNS |
| 14 | ArzunPD | 1.4421 | HyperPlace SA+**LNS** |
| 13 | TAISPlAce | 1.4321 | A**LNS** + Thompson Sampling |
| **—** | **RePlAce baseline** | **1.4578** | **порог отсечки** |
| 22 | SEVmakers | 1.5200 | Hybrid Legalization + SA |
| **—** | **SA baseline** | **2.1251** | |
| **—** | Greedy Row demo | 2.2109 | |

### Главные выводы

1. **DREAMPlace (DRP) доминирует** в топ-10: места 2, 5, 7, 10 — все на нём. Это GPU-ускоренная версия RePlAce/ePlace; де-факто стандартный seed для современных решений.
2. **Multi-start работает**: Shoom (#5), Archgen (#7) — несколько запусков с разных random seeds → выбор лучшего.
3. **GPU даёт скорость**: топ-2 в **37 секунд**, топ-10 в **17 секунд**. CPU-only решения работают в десятки раз дольше.
4. **Чистый LNS застревает на ~1.42-1.45** (TAISPlAce, ArzunPD, vmallela). Без хорошего seed LNS только догоняет RePlAce.
5. **Soft macros важны**: MOSAIC (#4) явно оптимизирует hard+soft вместе. SA baseline тоже двигает soft через `plc.optimize_stdcells()`. Простые placer'ы оставляют soft на месте → теряют 5-15% по proxy.
6. **Tier 1 тесты ОТКРЫТЫЕ** ([README.md FAQ](../README.md#L281)). Те же 17 IBM benchmark файлы используются у нас и у судей. Self-reported = verified (с поправкой на железо). Это рычаг — можно глубоко анализировать каждый benchmark, тренировать ML на них. Но **хардкод под имя бенчмарка = DQ** ([README.md:88](../README.md#L88)). Адаптивная логика по структуре (`if num_macros > 500: ...`) — ок.
7. **DQ-ловушки** (важно избежать):
   - `Mike Gao`: DREAMPlace silent fail в eval environment → 47-189 overlaps на benchmark
   - `BakaBobo`: код импортирует несуществующий `macro_place.fast_proxy`
   - `vmallela`: self-reported 1.1172, на их железе **1.4152** — 27% хуже
   - **Урок:** тестировать в чистой среде до сабмита, не полагаться на свои оптимизированные компоненты, явно пинить зависимости

### Что выбираем мы

**Стратегия:** **DREAMPlace как seed → LNS-refinement сверху**.

Это даёт:
- DREAMPlace доводит до ~1.30-1.40 за минуты
- LNS-фаза дополнительно улучшает на 3-7% (видно у Shoom #5: DRP+SA → 1.3381)
- Возможность win по Innovation Award за гибрид (если достаточно оригинален)

Если LNS зайдёт хорошо — потенциальный target ~1.30-1.32 (топ-3-5).

---

## 2. API и инфраструктура (выжимка)

### Сигнатура submission

Файл: `submissions/straple/placer.py`

```python
import torch
from macro_place.benchmark import Benchmark

class StraplePlacer:
    def __init__(self):
        # evaluator зовёт без аргументов
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Возвращает [num_macros, 2] тензор центров (x, y) в микронах
        # Hard macros: индексы [0, num_hard_macros)
        # Soft macros: индексы [num_hard_macros, num_macros)
        # Оба двигать можно и НУЖНО (см. SETUP.md:127)
        ...
```

**Жёсткие требования** (см. [SETUP.md:152-157](../SETUP.md#L152)):
- Координаты — **центры** макросов (не углы!)
- Fixed macros (`benchmark.macro_fixed`) не двигаем
- Все макросы внутри canvas (учитывая половину размера)
- **Zero overlaps между hard macros** (soft могут перекрываться — это нормально, они абстракция кластеров)

### Ключевые модули

| Модуль | Что даёт |
|---|---|
| [macro_place/benchmark.py](../macro_place/benchmark.py) | `Benchmark` dataclass (canvas, macros, nets, fixed mask) |
| [macro_place/loader.py](../macro_place/loader.py) | `load_benchmark_from_dir(path)` → `(Benchmark, plc)` |
| [macro_place/objective.py](../macro_place/objective.py) | `compute_proxy_cost(placement, benchmark, plc)` → dict с `proxy_cost`, `wirelength_cost`, `density_cost`, `congestion_cost`, `overlap_count`, ... |
| [macro_place/utils.py](../macro_place/utils.py) | `validate_placement(placement, benchmark)` → `(is_valid, violations)`; `visualize_placement(...)` |
| [macro_place/evaluate.py](../macro_place/evaluate.py) | CLI `uv run evaluate <file>` |
| [macro_place/def_writer.py](../macro_place/def_writer.py) | Экспорт в DEF (для Tier 2 ORFS, опц.) |

### Формула proxy cost

```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

- **Wirelength** — нормализованный HPWL (half-perimeter wirelength)
- **Density** — top-10% самых плотных grid cells
- **Congestion** — top-5% самых перегруженных routing-сегментов

### Soft macro оптимизация

Если двигаем hard, **обязательно** перестраиваем soft:

```python
canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
plc.optimize_stdcells(
    use_current_loc=False, move_stdcells=True, move_macros=False,
    log_scale_conns=False, use_sizes=False, io_factor=1.0,
    num_steps=[100, 100, 100],
    max_move_distance=[canvas_size/100]*3,
    attract_factor=[100, 1.0e-3, 1.0e-5],
    repel_factor=[0, 1.0e6, 1.0e7],
)
```

⚠️ Это **медленно в Python** (минуты на вызов). SA baseline вызывает редко, между батчами hard moves. Альтернатива — свой быстрый force-directed на GPU.

### Бенчмарки

- **Pre-processed `.pt` уже есть** в [benchmarks/processed/public/](../benchmarks/processed/public/) — 17 IBM + 4 NG45 + ASAP7 версии. Можно работать **без** инициализации submodule.
- Submodule `external/MacroPlacement` нужен только для:
  - Прогона полного evaluator с `uv run evaluate ... -b ibm01` (он грузит из `external/MacroPlacement/Testcases/...`)
  - Tier 2 ORFS оценки (DEF/TCL → OpenROAD)

---

## 3. Roadmap (фазы)

### Phase 0 — Setup (Day 1-2) ✅ ЗАВЕРШЕНА

- [x] Инициализировать submodule: `git submodule update --init external/MacroPlacement` — 3.5GB, 17 IBM benchmarks
- [x] `uv sync` — установлено 24 пакета (torch 2.10.0, numpy 2.4.2, и т.д.)
- [x] Прогнать demo: `uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01` → proxy 2.0463
- [x] Прогнать `--all` для greedy_row → AVG 2.2109 (точно как в leaderboard), 0.05s total
- [x] Прогнать `--all` для will_seed → AVG **1.5336**, 34.74s total — **наша точка отсчёта**
- [x] Smoke tests: `pytest test/test_smoke.py` → 7/7 PASSED
- [ ] Прочитать [submissions/will_seed/placer.py](../submissions/will_seed/placer.py) целиком — детально, не через агента
- [ ] Создать `submissions/straple/placer.py` с заглушкой (random placer) — убедиться что evaluator его подхватывает

### Phase 1 — Реализация скелета LNS (Day 3-5) 👈 ТЕКУЩАЯ

- [ ] **Initial seed**: начать с GreedyRowPlacer, потом заменить на DREAMPlace
- [ ] **Cost cache**: кешировать proxy components чтобы не пересчитывать с нуля при destroy/repair
- [ ] **Destroy operator (random)**: убрать k случайных макросов
- [ ] **Repair operator (greedy bottom-left fill)**: восстановить по одному в позицию с минимальным delta_cost
- [ ] **Acceptance**: greedy improve-only на старте
- [ ] **Legalization**: либо встроить в repair (только swap-ы в legal позиции), либо post-pass spiral search (как у will_seed)
- [ ] **Замер**: на ibm01 — должен побить greedy_row (~1.7) минимум до ~1.5

### Phase 2 — Хороший seed (Day 6-10) ☐

DREAMPlace интеграция — это самое жирное улучшение по leaderboard.

- [ ] Изучить, можно ли прицепить `dreamplace` PyPI пакет / `dreamplace-fpga` / открытую реализацию
- [ ] Альтернатива: написать **свой analytical placer** (gradient descent на smooth bell-shaped density + log-sum-exp WL approximation) — это путь MOSAIC #4
- [ ] Альтернатива минимум: использовать **force-directed** placement на bipartite graph (быстрее писать, хуже DRP)
- [ ] Сравнить seed'ы на ibm01-ibm04: какой даёт лучший pre-LNS proxy cost?

### Phase 3 — LNS operators (Day 11-15) ☐

- [ ] **Adaptive operator selection** (как ALNS — TAISPlAce топ-13): набор destroy/repair операторов с весами, обновляющимися по успеху
- [ ] Destroy variants:
  - random k macros
  - spatially-clustered subset (соседи по координатам)
  - worst-cost subset (макросы с худшим вкладом в HPWL)
  - net-based (макросы из одной "плохой" сети)
- [ ] Repair variants:
  - greedy min-delta-cost insertion
  - mini-SA на subset
  - force-directed re-insertion
- [ ] **Simulated-annealing acceptance** (а не просто greedy): принимать худшее с убывающей температурой
- [ ] **Soft macro joint optimization**: вызывать `plc.optimize_stdcells()` периодически, ИЛИ написать свой быстрый soft updater

### Phase 4 — GPU acceleration (Day 16-20) ☐

Если не уложимся в час на ibm17/ibm18 — без GPU топ-7 не светит.

- [ ] Перенести cost-incremental update на CUDA (torch на GPU)
- [ ] Batched destroy/repair: пробовать сразу N кандидатов параллельно
- [ ] Profile: где именно узкое место — `plc.get_cost()`, overlap check, или legalization

### Phase 5 — Tuning + verify (Day 21-25) ☐

- [ ] Grid search по гиперпараметрам (k destroy, T_start/T_end, num_iters) на ibm01..ibm04
- [ ] Прогон `--all` много раз на разных seed'ах → stability check
- [ ] **Critical**: прогон в чистой среде (новая `uv sync`, чистый clone) — избежать ловушки `BakaBobo`/`Mike Gao`
- [ ] Опционально: ORFS прогон на ariane133 (3-8 часов) — проверить что Tier 2 не ломается
- [ ] Замерить runtime на каждом benchmark — должно быть ≤1 час

### Phase 6 — Submit (Day 26-28) ☐

- [ ] Финальный прогон `--all`, зафиксировать avg proxy cost и runtime
- [ ] Создать ветку `submission` в нашем `Straple/...` репо, default branch
- [ ] Проверить что в репо ничего лишнего (deps пинятся в `pyproject.toml`)
- [ ] Расшарить приватный репо с judges: `partclxhrtmacroplace@gmail.com`, `will@partcl.com`
- [ ] Заполнить Google form с SHA коммита в URL
- [ ] **До дедлайна**: можно пересабмитить новую версию

---

## 4. Идеи / Pool

### Конкретные техники для LNS
- [ ] **Critical net analysis**: находить нетлы с худшим HPWL и трогать только их макросы
- [ ] **Mirror/flip orientation**: разрешено `N`, `FN`, `FS`, `S` (Klein-4) — сохранять выбор в `orientations.pt`
- [ ] **Сохранение лучшего**: best_placement обновляется только при принятии хода
- [ ] **Restart**: если стагнация N итераций — большой destroy + greedy repair
- [ ] **Hybrid with quadratic placement**: сначала quadratic для seed, потом LNS

### Идеи seed-этапа
- [ ] **Spectral seed** (Jiangban Ya #19): eigenvectors of Laplacian → coordinates
- [ ] **Concentric layout** на основе netlist (центральные узлы внутри)
- [ ] **Min-cut bisection** rekursivно (классический partitioning approach)
- [ ] Использовать `benchmark.macro_positions` (initial placement) как seed — он от организаторов hand-crafted

### Идеи post-processing
- [ ] **Snap to grid** (Tier 2 это всё равно делает) — учесть на Tier 1, чтобы grid alignment был не случайным
- [ ] **Gap injection ≥12 μm** между hard macros — pre-paid защита от Tier 2 push-apart
- [ ] **Orientation optimization** в финале (только flip → можем выиграть pin access)

### Innovation Award идеи
- [ ] LNS с обучаемой выборкой destroy operators (RL-bandit)
- [ ] Predictive cost via small neural net trained per benchmark
- [ ] Multi-objective Pareto frontier между WL/density/congestion

### Идеи, использующие открытость Tier 1 тестов
> Tier 1 — все 17 IBM публичные. Можно использовать при тренировке/тюнинге, но **без хардкодинга под имя**.

- [ ] **Per-benchmark тюнинг гиперпараметров** через generic feature: вместо `if name == 'ibm17'` использовать `if num_macros > 500 and density > 0.50 then more_iterations`. Параметры можно подобрать grid search'ем, факторами выбирая структуру, а не имя.
- [ ] **GNN/RL обучение на 17 IBM** — leave-one-out cross-validation: тренируем на 16, проверяем на 17-м. Так ловим overfitting под конкретный benchmark.
- [ ] **Анализ ibm02/ibm10/ibm12** — на этих will_seed обходит RePlAce. Понять, в чём специфика их netlist/canvas структуры → попробовать использовать эти инсайты в general logic.
- [ ] **Pre-computed embeddings/seeds** для известной структуры (тип топологии netlist) — но генеришь их детектируя структуру в runtime, не по имени.
- [ ] **Адаптивный compute budget**: на маленьких (`ibm01`-`ibm04`) — мало итераций SA, на больших (`ibm15`-`ibm18`) — больше. Используется runtime-запас 600× от лимита.

---

## 5. Журнал экспериментов

> Краткий журнал. Полные таблицы по каждому прогону с per-benchmark разбивкой и декомпозицией cost — в [results.md](results.md).

| Дата | Что проверял | Score (avg) | Best/Worst | Runtime | Комментарий |
|---|---|---|---|---|---|
| _baseline_ | SA (TILOS reported) | 2.1251 | 1.3166 / 3.6726 | — | Из README |
| _baseline_ | RePlAce (TILOS reported) | **1.4578** | 0.9976 / 1.8370 | — | **Порог отсечки** |
| 2026-05-03 | greedy_row_placer (Mac ARM) | 2.2109 | 1.6728 (ibm09) / 2.7696 (ibm12) | 0.05s total | Точно как в leaderboard. Все 0 overlaps |
| 2026-05-03 | will_seed (Mac ARM) | **1.5336** | 1.1625 (ibm09) / 1.7921 (ibm18) | **34.74s total** (ibm17=6.05s ⬅ slowest) | Reference от организаторов. -5.2% от RePlAce, +27.8% от SA. Mac ARM = ~35s ↔ leaderboard 35s |
| | | | | | |

**Шаблон записи:** дата, что меняли, средний proxy на `--all`, лучший/худший benchmark, runtime, инсайты.

### Bottleneck-анализ (по will_seed)

Декомпозиция AVG proxy:
- Wirelength: ~0.06-0.08 (мало, SA уже хорошо это давит)
- Density: ~0.85-1.05 (средне)
- **Congestion: 1.3-2.6** ← основной источник стоимости

→ **Главная цель LNS — снизить congestion.** Просто двигать макросы для минимизации HPWL (как делает SA в will_seed) недостаточно.

---

## 6. Полезные команды

```bash
# Setup
git submodule update --init external/MacroPlacement
uv sync

# Запуск своего placer'а
uv run evaluate submissions/straple/placer.py -b ibm01            # один
uv run evaluate submissions/straple/placer.py --all               # все 17 IBM
uv run evaluate submissions/straple/placer.py --ng45              # 4 NG45
uv run evaluate submissions/straple/placer.py -b ibm01 --vis      # с визуализацией

# Подтянуть обновления от организаторов
git fetch upstream
git merge upstream/main
git push origin main

# Работа в submission ветке
git checkout -b submission
# ... edit submissions/straple/placer.py ...
git add submissions/straple/
git commit -m "LNS placer iteration N"
git push -u origin submission
git rev-parse HEAD                 # SHA для Google form

# Локально протестировать на pre-processed .pt (без submodule)
python -c "from macro_place.benchmark import Benchmark; b = Benchmark.load('benchmarks/processed/public/ibm01.pt'); print(b.num_hard_macros, b.canvas_width)"
```

---

## 7. Полезные файлы / референсы

### В репо
- [README.md](../README.md) — правила, leaderboard (или [русский перевод](README.md))
- [SCORING.md](../SCORING.md) — формула Tier 2 (geometric mean WNS:TNS:Area = 3:2:1) (или [русский перевод](SCORING.md))
- [SETUP.md](../SETUP.md) — API референс, soft macro helper, ORFS prerequisites (или [русский перевод](SETUP.md))
- [submissions/examples/greedy_row_placer.py](../submissions/examples/greedy_row_placer.py) — простой shelf-pack
- [submissions/examples/simple_random_placer.py](../submissions/examples/simple_random_placer.py) — тривиальный пример
- [submissions/will_seed/placer.py](../submissions/will_seed/placer.py) — **reference от организаторов**, гибрид легализации + SA refinement (изучить целиком)
- [macro_place/objective.py](../macro_place/objective.py) — формула proxy cost, overlap detection
- [macro_place/loader.py](../macro_place/loader.py) — `load_benchmark_from_dir`
- [scripts/evaluate_with_orfs.py](../scripts/evaluate_with_orfs.py) — Tier 2 локально (3-8 часов на NG45)

### Внешние
- Background paper #1 (key): [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304) — baseline numbers оттуда
- [TILOS MacroPlacement repo](https://github.com/TILOS-AI-Institute/MacroPlacement) — evaluator source, PlacementCost API
- [DREAMPlace repo](https://github.com/limbo018/DREAMPlace) — топ-2 базируется на нём
- [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) — Tier 2 evaluation flow
- [PlacementCost API в TILOS](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py) — для прямого доступа к нетлисту

---

## 8. Риски и подводные камни

| Риск | Митигация |
|---|---|
| **Float-precision overlaps** на границе | Добавлять gap ≥0.001 между макросами (как greedy_row делает) |
| **DQ из-за overlaps в eval env** (Mike Gao) | Прогон в чистой `uv sync` среде до сабмита |
| **Зависимость на нестандартный пакет** (BakaBobo, ArzunPD networkit) | Все deps в `pyproject.toml`, через `uv sync` ставится |
| **Self-reported ≠ verified** (vmallela 1.12 → 1.42) | Замерять на их железе (16 cores, 100GB) — или close to it |
| **Soft macros не двигаем** | Обязательно `plc.optimize_stdcells()` или свой soft updater |
| **Runtime > 1 hour на ibm17/ibm18** | Profile, GPU, или ослабить итерации на больших benchmarks |
| **Tier 2 push-apart портит размещение** | Оставлять ≥12 μm между hard macros в submission |
| **Hardcoding под benchmarks** (DQ) | Один алгоритм с параметрами, без if-ов на имя бенчмарка |
