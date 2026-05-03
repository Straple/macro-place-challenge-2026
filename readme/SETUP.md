# Setup и API Reference

> Русский перевод [SETUP.md](../SETUP.md). Оригинал на английском — источник истины при расхождениях.

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/partcleda/macro-place-challenge-2026.git
cd macro-place-challenge-2026

# Инициализировать TILOS MacroPlacement submodule (нужен для оценки)
git submodule update --init external/MacroPlacement

# Создать virtual environment и установить пакет (editable)
uv sync
```

## Структура проекта

```
├── macro_place/            # Устанавливаемый Python-пакет
│   ├── __init__.py
│   ├── benchmark.py        # Benchmark dataclass (PyTorch tensors)
│   ├── loader.py           # Загрузка бенчмарков из ICCAD04 формата
│   ├── objective.py        # Вычисление proxy cost
│   ├── utils.py            # Валидация и визуализация
│   └── def_writer.py       # Экспорт в DEF
├── submissions/
│   └── examples/           # Примеры placer'ов (greedy_row_placer.py, simple_random_placer.py)
├── external/
│   └── MacroPlacement/     # TILOS evaluator и ICCAD04 testcases
├── benchmarks/
│   └── processed/          # Pre-processed .pt benchmark файлы
├── pyproject.toml          # Конфигурация пакета и зависимостей (для uv)
└── SETUP.md                # Этот файл
```

## API Reference

### Загрузка бенчмарка

```python
from macro_place.loader import load_benchmark_from_dir

benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')
```

Возвращает:
- `benchmark`: dataclass `Benchmark` с PyTorch tensor'ами
- `plc`: объект `PlacementCost` (нужен для вычисления стоимости)

### Объект Benchmark

Dataclass `Benchmark` содержит:

| Поле | Тип | Описание |
|-------|------|-------------|
| `name` | `str` | Имя бенчмарка (например, "ibm01") |
| `canvas_width` | `float` | Ширина канваса в микронах |
| `canvas_height` | `float` | Высота канваса в микронах |
| `num_macros` | `int` | Всего макросов (hard + soft) |
| `num_hard_macros` | `int` | Количество hard macros (индексы `[0, num_hard)`) |
| `num_soft_macros` | `int` | Количество soft macros (индексы `[num_hard, num_macros)`) |
| `macro_positions` | `Tensor [N, 2]` | Координаты центров (x, y) (сначала hard, потом soft) |
| `macro_sizes` | `Tensor [N, 2]` | (width, height) каждого макро |
| `macro_fixed` | `Tensor [N]` | Boolean маска зафиксированных макросов |
| `macro_names` | `List[str]` | Имена для отладки |
| `num_nets` | `int` | Количество нетов |
| `grid_rows`, `grid_cols` | `int` | Размер сетки для density/congestion |
| `hard_macro_indices` | `List[int]` | Маппинг tensor index → PlacementCost module index (hard) |
| `soft_macro_indices` | `List[int]` | Маппинг tensor index → PlacementCost module index (soft) |

Вспомогательные методы:
- `benchmark.get_movable_mask()` — возвращает `~macro_fixed`
- `benchmark.get_hard_macro_mask()` — True для hard macros (первые `num_hard_macros` записей)
- `benchmark.get_soft_macro_mask()` — True для soft macros
- `benchmark.save(path)` / `Benchmark.load(path)` — сохранение/загрузка `.pt` файлов

### Вычисление Proxy Cost

```python
from macro_place.objective import compute_proxy_cost

costs = compute_proxy_cost(placement, benchmark, plc)
```

**Вход:** `placement` — тензор `[num_macros, 2]` с координатами центров (x, y).

**Выход:** Словарь:

| Ключ | Описание |
|-----|-------------|
| `proxy_cost` | Взвешенная сумма: 1.0 × WL + 0.5 × density + 0.5 × congestion |
| `wirelength_cost` | Нормализованный HPWL по всем нетам |
| `density_cost` | Density топ-10% grid cells |
| `congestion_cost` | Congestion топ-5% routing с сглаживанием |
| `overlap_count` | Количество пересекающихся пар макросов |
| `total_overlap_area` | Общая площадь overlap в μm² |
| `overlap_ratio` | Доля макросов, участвующих в пересечениях |

### Валидация placement

```python
from macro_place.utils import validate_placement

is_valid, violations = validate_placement(placement, benchmark)
```

Проверки:
- Корректная форма tensor
- Отсутствие NaN/Inf
- Все макросы внутри границ канваса
- Зафиксированные макросы на исходных позициях
- Ноль пересечений между макросами

### Визуализация placement

```python
from macro_place.utils import visualize_placement

visualize_placement(placement, benchmark, save_path='output.png')
```

## Написание placer'а

Твой placer принимает `Benchmark` и возвращает тензор `[num_macros, 2]` с позициями. Тензор содержит и hard macros (индексы `[0, num_hard_macros)`), и soft macros (индексы `[num_hard_macros, num_macros)`).

**Двигаются и hard, и soft макросы.** Hard macros — основная цель оптимизации (SRAM, IP-блоки и т.п.). Soft macros — кластеры стандартных ячеек. Совместная оптимизация их позиций вместе с hard макросами улучшит wirelength, density и congestion. SA baseline делает это, запуская force-directed размещение soft макросов после каждой пачки ходов hard макросов.

```python
import torch
from macro_place.benchmark import Benchmark

class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        # Hard macros: индексы [0, num_hard_macros)
        # Soft macros: индексы [num_hard_macros, num_macros)
        # Оба типа подвижны — оптимизируй оба для лучшего результата

        hard_movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(hard_movable)[0]

        # Здесь твой алгоритм
        # - Двигай hard макросы для оптимизации placement
        # - Опционально перепозиционируй soft макросы вслед за изменениями hard
        #   (SA baseline использует PlacementCost.optimize_stdcells() для этого)

        return placement
```

Ключевые ограничения:
- Координаты — **центры** (не углы)
- Зафиксированные макросы должны оставаться на исходных позициях
- Все макросы должны быть полностью внутри границ канваса
- **Ноль пересечений между hard macros** обязательно (soft macros могут пересекаться — они абстракции кластеров стандартных ячеек)
- Перемещение hard макросов без перепозиционирования soft ухудшит wirelength и density

См. [`submissions/examples/greedy_row_placer.py`](../submissions/examples/greedy_row_placer.py) для простого примера и [`submissions/will_seed/placer.py`](../submissions/will_seed/placer.py) для более полного подхода.

## Связность нетов (Net Connectivity)

Связность нетов хранится внутри объекта `PlacementCost` (`plc`), а не в тензорах `Benchmark`. Вычисление proxy cost использует её автоматически.

Если нужен прямой доступ к данным нетов для алгоритма (например, для GNN), его можно получить через PlacementCost API:

```python
# Количество нетов
print(plc.net_cnt)

# Доступ к отдельным модулям и их соединениям
for i, module in enumerate(plc.modules_w_pins):
    print(module.get_name(), module.get_pos())
```

Полный API PlacementCost: [TILOS MacroPlacement source](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py).

## Оптимизация Soft Macro

Soft macros (кластеры стандартных ячеек) соединены с hard macros через неты. При перемещении hard макросов оптимальные позиции soft макросов меняются. PlacementCost API предоставляет встроенный force-directed placer для soft макросов:

```python
# После установки позиций hard макросов, переоптимизировать soft:
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

Это то, что SA baseline делает между итерациями. **Внимание: это медленно в Python (~минуты на вызов).** Можно уменьшить `num_steps` для быстрой, но менее оптимальной работы — или реализовать свою оптимизацию soft макросов (например, совместно в GPU-оптимизаторе).

## Запуск бенчмарков

### IBM Benchmarks (Tier 1 — Proxy Cost)

17 IBM ICCAD04 бенчмарков лежат в `external/MacroPlacement/Testcases/ICCAD04/`. Запуск одного бенчмарка или всего набора через demo placer:

```bash
# Один бенчмарк
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01

# Все 17 бенчмарков с таблицей сравнения
uv run evaluate submissions/examples/greedy_row_placer.py --all
```

Чтобы оценить свой placer на всех бенчмарках, делай по тому же шаблону — цикл по директориям бенчмарков:

```python
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

BENCHMARKS = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06", "ibm07", "ibm08", "ibm09",
    "ibm10", "ibm11", "ibm12", "ibm13", "ibm14", "ibm15", "ibm16", "ibm17", "ibm18",
]

for name in BENCHMARKS:
    benchmark, plc = load_benchmark_from_dir(f'external/MacroPlacement/Testcases/ICCAD04/{name}')
    placement = my_placer.place(benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)
    print(f"{name}: proxy={costs['proxy_cost']:.4f}  overlaps={costs['overlap_count']}")
```

### NG45 Designs (Tier 2 — OpenROAD Flow)

Топ-7 сабмишенов по proxy score оцениваются через полный OpenROAD PnR flow на NanGate45 дизайнах. Эти дизайны в TILOS репозитории:

```
external/MacroPlacement/Flows/NanGate45/
├── ariane133/    # RISC-V core, 133 макроса
├── ariane136/    # RISC-V core, 136 макросов
├── mempool_tile/ # Memory pool, 20 макросов
└── nvdla/        # NVIDIA DLA, 128 макросов
```

Pre-processed `.pt` версии доступны в `benchmarks/processed/public/` для быстрой загрузки:

```python
from macro_place.benchmark import Benchmark

benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45_random.pt')
```

OpenROAD flow измеряет WNS (worst negative slack), TNS (total negative slack), Area. Участникам не нужно запускать OpenROAD самим — судьи сделают это для топовых сабмишенов.

#### Запуск ORFS локально (опционально)

Если хочешь протестировать своё placement через полный PnR flow локально, есть `scripts/evaluate_with_orfs.py`, автоматизирующий весь процесс.

**Предусловия**: установить [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) рядом с этим репо:

```bash
cd ..
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd macro-place-challenge-2026
```

**Запуск ORFS оценки:**

```bash
# Оценить один NG45 дизайн (использует дефолтное размещение)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker

# Оценить со своим размещением (сохранённым как [num_macros, 2] tensor)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker \
    --placement my_placement.pt

# Оценить все NG45 дизайны
python scripts/evaluate_with_orfs.py --all --no-docker

# Указать пользовательскую установку ORFS
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 \
    --orfs-root /path/to/OpenROAD-flow-scripts --no-docker
```

Скрипт:
1. Загружает бенчмарк и считает proxy cost.
2. Генерирует macro placement TCL скрипт (обрабатывает name mapping между protobuf и ODB форматами).
3. Копирует design config в ORFS с необходимыми патчами.
4. Запускает полный ORFS flow (synthesis → floorplan → placement → CTS → routing).
5. Парсит и рапортует WNS, TNS, Area и другие метрики.

Полный прогон ORFS занимает примерно **3-8 часов на дизайн** в зависимости от бенчмарка и машины.
