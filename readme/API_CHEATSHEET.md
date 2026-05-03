# API Cheatsheet — компактная шпаргалка

> Всё, что часто нужно при разработке placer'а, на одной странице. Полный референс — в [SETUP.md](SETUP.md). Сигнатуры взяты из [../macro_place/](../macro_place/).

## Импорты

```python
import torch
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost, compute_overlap_metrics
from macro_place.utils import validate_placement, visualize_placement
```

---

## Минимальный шаблон placer'а

```python
class MyPlacer:
    def __init__(self):  # вызывается без аргументов
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        # ... твой алгоритм ...
        return placement  # shape [num_macros, 2], координаты центров в μm
```

**Жёсткие требования:**
- Возвращаемый тензор: `[num_macros, 2]`, dtype `float32`/`float64`
- Координаты — **центры**, не углы
- Hard macros: индексы `[0, num_hard_macros)`
- Soft macros: индексы `[num_hard_macros, num_macros)`
- Fixed macros (`benchmark.macro_fixed[i] == True`) **не двигать**
- Все макросы в bounds: `width/2 ≤ x ≤ canvas_width - width/2` (то же для y)
- **Zero overlaps между hard macros** (soft могут пересекаться)

---

## Как evaluator находит твой класс

`uv run evaluate <file.py>` делает примерно так:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("placer", "submissions/.../placer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Ищет ПЕРВЫЙ класс в файле с методом .place()
placer_cls = next(cls for name, cls in vars(mod).items()
                  if isinstance(cls, type) and hasattr(cls, "place"))
placer = placer_cls()  # без аргументов!
placement = placer.place(benchmark)
```

→ Из этого следует:
1. Класс может называться как угодно, но должен идти **первым** в файле с методом `place()`.
2. `__init__()` не должен требовать аргументов (или иметь дефолты для всех).
3. Можно класть placer в собственный пакет (директорию с `__init__.py`), как `submissions/will_seed/`.

---

## Загрузка benchmark

### Из исходников (нужен submodule)

```python
benchmark, plc = load_benchmark_from_dir(
    "external/MacroPlacement/Testcases/ICCAD04/ibm01"
)
```

Возвращает `(Benchmark, PlacementCost)`. `plc` нужен для `compute_proxy_cost`.

### Из pre-processed `.pt` (без submodule, БЫСТРО)

```python
benchmark = Benchmark.load("benchmarks/processed/public/ibm01.pt")
# plc нет → compute_proxy_cost не получится
```

Хорошо для быстрой отладки самой логики placer'а.

### NG45 (Tier 2)

```python
benchmark, plc = load_benchmark(
    netlist_file="external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/netlist.pb.txt",
    plc_file="external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping/initial.plc",
    name="ariane133"
)
# или из .pt
benchmark = Benchmark.load("benchmarks/processed/public/ariane133_ng45.pt")
```

---

## Поля Benchmark (то, что чаще всего трогаем)

```python
benchmark.name                    # str, "ibm01"
benchmark.canvas_width            # float, μm
benchmark.canvas_height           # float, μm

benchmark.num_macros              # int, hard + soft
benchmark.num_hard_macros         # int
benchmark.num_soft_macros         # int

benchmark.macro_positions         # Tensor [N, 2], (x, y) центры
benchmark.macro_sizes             # Tensor [N, 2], (w, h)
benchmark.macro_fixed             # Tensor [N], bool — нельзя двигать
benchmark.macro_names             # List[str]

benchmark.num_nets                # int
benchmark.net_nodes               # List[Tensor], net → indices макросов
benchmark.net_pin_nodes           # List[Tensor [pins, 2]], net → (owner_idx, pin_slot)
benchmark.net_weights             # Tensor [num_nets]

benchmark.macro_pin_offsets       # List[Tensor], смещения пинов от центра
benchmark.port_positions          # Tensor [num_ports, 2]

benchmark.grid_rows               # int, для density/congestion
benchmark.grid_cols               # int

benchmark.hard_macro_indices      # List[int], в plc.modules_w_pins
benchmark.soft_macro_indices      # List[int]
```

### Маски (часто нужны)

```python
movable      = benchmark.get_movable_mask()       # ~macro_fixed
hard_mask    = benchmark.get_hard_macro_mask()    # True для индексов [0, num_hard)
soft_mask    = benchmark.get_soft_macro_mask()    # True для [num_hard, num_macros)

# Подвижные hard macros
hard_movable_idx = torch.where(movable & hard_mask)[0]

# Подвижные soft macros
soft_movable_idx = torch.where(movable & soft_mask)[0]
```

---

## Compute proxy cost

```python
costs = compute_proxy_cost(placement, benchmark, plc)
# → dict:
costs["proxy_cost"]              # float, ГЛАВНАЯ метрика для leaderboard
costs["wirelength_cost"]         # float, нормализованный HPWL
costs["density_cost"]            # float, top-10% densest grid cells
costs["congestion_cost"]         # float, top-5% routing congestion
costs["overlap_count"]           # int, число пар пересекающихся hard macros — должно быть 0
costs["total_overlap_area"]      # float, μm²
costs["max_overlap_area"]        # float, μm²
costs["num_macros_with_overlaps"] # int
costs["overlap_ratio"]           # float, доля затронутых макросов
```

**Формула:**
```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

### Только overlap-метрики (без полного cost'а)

```python
overlap_metrics = compute_overlap_metrics(placement, benchmark)
# Дешевле, чем полный compute_proxy_cost. Возвращает только overlap_*.
```

### Кастомные веса

```python
costs = compute_proxy_cost(placement, benchmark, plc, weights={
    "wirelength": 1.0,
    "density": 0.5,
    "congestion": 0.5,
})
```

---

## Validate placement

```python
is_valid, violations = validate_placement(placement, benchmark, check_overlaps=True)
# is_valid: bool
# violations: List[str], человекочитаемые ошибки

if not is_valid:
    print("Нарушения:", violations[:5])
```

Что проверяется:
- Shape тензора = `(num_macros, 2)`
- Нет NaN/Inf
- Все в bounds canvas
- Fixed макросы на исходных позициях
- Нет overlaps между hard macros (если `check_overlaps=True`)

---

## Visualize

```python
visualize_placement(placement, benchmark, save_path="out.png", plc=plc)
# Сохраняет PNG с 3 панелями: placement, density heatmap, congestion heatmap.
# Если plc=None — только placement без heatmap'ов.
```

---

## Soft macro optimization (важно!)

После каждой пачки ходов hard macros — **переоптимизируй soft**, иначе теряешь 5-15% по proxy:

```python
canvas_size = max(benchmark.canvas_width, benchmark.canvas_height)
plc.optimize_stdcells(
    use_current_loc=False,
    move_stdcells=True,
    move_macros=False,
    log_scale_conns=False,
    use_sizes=False,
    io_factor=1.0,
    num_steps=[100, 100, 100],
    max_move_distance=[canvas_size/100]*3,
    attract_factor=[100, 1.0e-3, 1.0e-5],
    repel_factor=[0, 1.0e6, 1.0e7],
)

# После — забрать обновлённые позиции:
for i, plc_idx in enumerate(benchmark.soft_macro_indices):
    pos = plc.modules_w_pins[plc_idx].get_pos()
    placement[benchmark.num_hard_macros + i] = torch.tensor([pos.x, pos.y])
```

⚠️ **Медленно** (минуты). Снижай `num_steps` для скорости в ущерб качеству. Или пиши свой быстрый force-directed на GPU.

---

## Прямой доступ к PlacementCost (для GNN/custom алгоритмов)

```python
plc.net_cnt                      # int, количество нетов

for i, module in enumerate(plc.modules_w_pins):
    name = module.get_name()      # str
    pos = module.get_pos()        # (x, y), доступ через .x, .y
    # ... pin info через module.get_pin_count(), и т.д.

# Установить позицию (но обычно через placement tensor → автоматически синкается):
module.set_pos(x, y)

# Получить cost напрямую:
plc.get_cost()                    # wirelength
plc.get_density_cost()            # density
plc.get_congestion_cost()         # congestion
```

Полный API: [TILOS plc_client_os.py](https://github.com/TILOS-AI-Institute/MacroPlacement/blob/main/CodeElements/Plc_client/plc_client_os.py).

---

## CLI команды

```bash
# Один benchmark
uv run evaluate submissions/.../placer.py -b ibm01

# Все 17 IBM
uv run evaluate submissions/.../placer.py --all

# 4 NG45
uv run evaluate submissions/.../placer.py --ng45

# С визуализацией (PNG в текущей директории)
uv run evaluate submissions/.../placer.py -b ibm01 --vis
uv run evaluate submissions/.../placer.py --all --vis

# Smoke tests
uv run pytest test/test_smoke.py -v

# Tier 2 (ORFS, требует OpenROAD-flow-scripts рядом)
uv run python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker
```

---

## Полный end-to-end пример

```python
import torch
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement

class MyPlacer:
    def place(self, benchmark):
        p = benchmark.macro_positions.clone()
        # ... алгоритм ...
        return p

benchmark, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
placer = MyPlacer()
placement = placer.place(benchmark)

ok, violations = validate_placement(placement, benchmark)
assert ok, violations[:5]

costs = compute_proxy_cost(placement, benchmark, plc)
print(f"proxy={costs['proxy_cost']:.4f}  WL={costs['wirelength_cost']:.4f}  "
      f"density={costs['density_cost']:.4f}  congest={costs['congestion_cost']:.4f}  "
      f"overlaps={costs['overlap_count']}")

visualize_placement(placement, benchmark, "out.png", plc=plc)
```

---

## Частые ошибки

| Ошибка | Причина | Фикс |
|---|---|---|
| `placement.shape == (num_hard_macros, 2)` | Забыл включить soft macros | Возвращай **все** `num_macros`, не только hard |
| `overlap_count > 0` после tiny ходов | Float-precision на границе | Добавляй gap ≥0.001 в legalization |
| `validate_placement` ругается на fixed | Случайно сдвинул зафиксированный | После всех ходов: `placement[fixed_mask] = benchmark.macro_positions[fixed_mask]` |
| `evaluator не находит класс` | Класс не первый в файле / нет метода `place` | Имя класса не важно, но порядок и метод важны |
| `compute_proxy_cost ОЧЕНЬ медленный` | `plc.optimize_stdcells` зовётся каждую итерацию | Снижай частоту вызова или `num_steps` |
| `NaN` в proxy cost | Макро вне bounds, или 0-площадь, или duplicates | `validate_placement` поймает; ищи в violations |
