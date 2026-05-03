# Quickstart — от чистой системы до первого прогона

> Пошаговое руководство: что установить, как запустить дефолтные решения, как протестировать, как создать свой placer. На каждом шаге — что должно получиться.

---

## TL;DR

```bash
# 1. Установить uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Подтянуть submodule с TILOS evaluator
git submodule update --init external/MacroPlacement

# 3. Создать venv и установить зависимости
uv sync

# 4. Прогнать smoke-тесты — убедиться что всё работает
uv run pytest test/test_smoke.py -v

# 5. Запустить дефолтное решение на одном бенчмарке
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01

# 6. Запустить на всех 17 IBM benchmarks (главный замер)
uv run evaluate submissions/examples/greedy_row_placer.py --all
```

Если все 6 шагов прошли — окружение готово, можно писать свой placer. Дальше — детали.

---

## Что должно быть установлено

- **Python 3.8+** (предпочтительно 3.10 или 3.11)
- **git** (для клонирования и subomodule)
- **uv** — современный Python package manager (поставим на шаге 1)
- **C++ toolchain** (для сборки PlacementCost из TILOS): на macOS — Xcode CLI tools (`xcode-select --install`), на Linux — build-essential
- **(опционально) CUDA 12.4** — если хочешь запускать на GPU. Без неё всё будет работать на CPU.

Проверка:
```bash
python3 --version       # >= 3.8
git --version
clang --version         # на macOS, или gcc --version на Linux
```

---

## Шаг 1 — Установить uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

После установки **перезапусти терминал** (или `source ~/.zshrc`), чтобы появилась команда `uv`.

Проверка:
```bash
uv --version            # должна появиться версия, например "uv 0.4.x"
```

> Зачем uv? Это быстрый установщик пакетов Python (на Rust). Делает то же что `pip + venv`, но на порядок быстрее. Это рекомендованный способ в [SETUP.md](SETUP.md) (его использует `evaluate` команда).

---

## Шаг 2 — Подтянуть TILOS submodule

В репо подключен сабмодуль с TILOS MacroPlacement — это и evaluator, и benchmark данные.

```bash
git submodule update --init external/MacroPlacement
```

Это **скачает ~500 MB** (один раз). Если канал медленный — добавь `--depth 1`:
```bash
git submodule update --init --depth 1 external/MacroPlacement
```

После завершения проверь:
```bash
ls external/MacroPlacement/Testcases/ICCAD04/
# Должны появиться директории ibm01, ibm02, ..., ibm18
```

> Если этот шаг упал по сети или подвис — попробуй `git submodule update --init --recursive` повторно. Можно работать **без submodule** на pre-processed `.pt` файлах (см. ниже), но тогда CLI `evaluate` не запустит свои бенчмарки.

---

## Шаг 3 — Установить зависимости

```bash
uv sync
```

Это:
- Создаст `.venv/` в корне репо
- Установит все зависимости из [pyproject.toml](../pyproject.toml) (torch, numpy, matplotlib, tqdm, absl-py)
- Установит сам пакет `macro_place` в editable-режиме

Проверка:
```bash
uv run python -c "import macro_place; print(macro_place.__file__)"
# Должно вывести путь типа /.../macro-place-challenge-2026/macro_place/__init__.py
```

> Если `uv sync` падает на установке `torch` — см. [TROUBLESHOOTING.md](TROUBLESHOOTING.md#uv-sync-падает-на-сборке-pytorch--cuda).

---

## Шаг 4 — Запустить smoke-тесты

В репо уже есть [test/test_smoke.py](../test/test_smoke.py) — pytest-тесты, проверяющие, что инфраструктура работает end-to-end.

```bash
uv run pytest test/test_smoke.py -v
```

Должно быть 7 тестов, все PASS:
- `test_load_benchmark_pt` — загрузка `.pt` файла
- `test_load_benchmark_from_dir` — загрузка из ICCAD04 директории
- `test_compute_proxy_cost` — вычисление proxy cost
- `test_validate_placement` — функция валидации
- `test_net_pin_nodes` — pin-level connectivity
- `test_benchmark_save_load_roundtrip` — сохранение/загрузка
- `test_greedy_row_placer` — end-to-end через greedy placer

Если что-то FAIL — разбираться **не двигаясь дальше**. Скорее всего:
- Submodule не подтянулся (см. шаг 2)
- Не собрался C++ binding в TILOS

---

## Шаг 5 — Запустить дефолтные решения (есть!)

В репо есть **3 готовых placer'а**, можно их сразу прогнать:

### 5.1. Тривиальный пример: random placer

[../submissions/examples/simple_random_placer.py](../submissions/examples/simple_random_placer.py) — кладёт макросы случайно. Результат плохой и с overlap'ами, нужен только для понимания API.

```bash
uv run evaluate submissions/examples/simple_random_placer.py -b ibm01
```

### 5.2. Greedy row packer ⭐ хороший reference

[../submissions/examples/greedy_row_placer.py](../submissions/examples/greedy_row_placer.py) — shelf-packing, сортирует макросы по высоте, заполняет рядами. Достигает **0 overlaps**, но не оптимизирует netlist.

```bash
# На одном бенчмарке (быстро)
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01

# Ожидаемый вывод:
#   Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
#      ibm01    ~1.85    1.3166    0.9976     -41%      -85%         0
```

```bash
# На всех 17 IBM (~30-60 секунд total)
uv run evaluate submissions/examples/greedy_row_placer.py --all

# В конце:
#      AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

### 5.3. Will seed (наиболее серьёзный пример) ⭐⭐

[../submissions/will_seed/placer.py](../submissions/will_seed/placer.py) — гибрид: spiral-search legalization + SA refinement. От организаторов как reference; в leaderboard заявлен как `Will Seed (Partcl)` со score 1.5338.

```bash
uv run evaluate submissions/will_seed/placer.py -b ibm01
# Должно быть значительно лучше greedy_row (~1.16 vs ~1.85 на ibm01)
```

```bash
uv run evaluate submissions/will_seed/placer.py --all
# Avg около 1.5338 — лучше SA, чуть хуже RePlAce
```

> **Изучи will_seed внимательно** — это лучший reference. Там видно, как делать legalization, как организовать SA-цикл с overlap-rejection, как использовать netlist для smart-ходов.

### 5.4. С визуализацией

```bash
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01 --vis
```

Создаст PNG в текущей директории — три панели: размещение макросов, density heatmap, congestion heatmap. Полезно для понимания, что happen.

---

## Шаг 6 — Понять, что выводит evaluator

Пример на `--all`:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    1.5338    1.3166    0.9976    -16.5%      -53.7%         0
   ibm02    2.1234    1.9072    1.8370    -11.3%      -15.6%         0
   ...
     AVG    1.7234    2.1251    1.4578     +18.9%     -18.2%         0
```

Колонки:
- **Benchmark** — имя
- **Proxy** — твой proxy cost (главная метрика)
- **SA, RePlAce** — baseline'ы из TILOS paper
- **vs SA / vs RePlAce** — на сколько % ты лучше/хуже (минус = ты хуже baseline'а)
- **Overlaps** — должно быть **0** на каждой строке, иначе DQ

Полная формула proxy cost — в [PROBLEM.md](PROBLEM.md#компоненты-целевой-функции):
```
proxy_cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion
```

---

## Шаг 7 — Создать свой placer

```bash
mkdir -p submissions/straple
```

Создай `submissions/straple/placer.py` с минимальным шаблоном:

```python
import torch
from macro_place.benchmark import Benchmark


class StraplePlacer:
    def __init__(self):
        # evaluator вызывает без аргументов
        pass

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # Стартуем от исходного размещения
        placement = benchmark.macro_positions.clone()

        # TODO: твой алгоритм здесь
        # - Двигай только movable hard macros
        # - Соблюдай canvas bounds
        # - Никаких overlaps между hard

        return placement
```

Проверь, что evaluator его подхватывает:
```bash
uv run evaluate submissions/straple/placer.py -b ibm01
```

> Поскольку placer пока ничего не делает (возвращает исходное), результат будет такой же, как initial placement из `.plc` файла организаторов. Это нормально — мы убедились, что pipeline работает.

---

## Шаг 8 — Цикл разработки (что делать каждый день)

```bash
# 1. Поправил submissions/straple/placer.py

# 2. Быстрая проверка на одном бенчмарке (5-30 сек)
uv run evaluate submissions/straple/placer.py -b ibm01

# 3. Если результат хороший — прогнать на всех (1-30 минут)
uv run evaluate submissions/straple/placer.py --all

# 4. Записать результат в журнал todo.md (секция "Журнал экспериментов")

# 5. Регулярно — smoke tests, чтоб не сломать инфру
uv run pytest test/test_smoke.py -v
```

---

## Шаг 9 — Свой dev-скрипт (для отладки и интроспекции)

CLI `evaluate` хорош для замеров, но не даёт breakpoints / детального вывода. Для отладки создай `scripts/dev_test.py`:

```python
"""Debug helper — запускать через `uv run python scripts/dev_test.py`."""
import time
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement, visualize_placement

from submissions.straple.placer import StraplePlacer


def main():
    # Можно работать без submodule из pre-processed .pt:
    # benchmark = Benchmark.load("benchmarks/processed/public/ibm01.pt")
    # plc = None   # без plc compute_proxy_cost не работает

    # Или из сабмодуля — полный pipeline:
    benchmark, plc = load_benchmark_from_dir(
        "external/MacroPlacement/Testcases/ICCAD04/ibm01"
    )

    placer = StraplePlacer()
    t0 = time.perf_counter()
    placement = placer.place(benchmark)
    runtime = time.perf_counter() - t0

    is_valid, violations = validate_placement(placement, benchmark)
    print(f"Valid: {is_valid}")
    if not is_valid:
        for v in violations[:5]:
            print(f"  ! {v}")

    if plc is not None:
        costs = compute_proxy_cost(placement, benchmark, plc)
        print(f"Proxy:    {costs['proxy_cost']:.4f}")
        print(f"  WL:      {costs['wirelength_cost']:.4f}")
        print(f"  Density: {costs['density_cost']:.4f}")
        print(f"  Congest: {costs['congestion_cost']:.4f}")
        print(f"Overlaps: {costs['overlap_count']} pairs, "
              f"area {costs['total_overlap_area']:.4f}")
        print(f"Runtime:  {runtime:.2f}s")

        visualize_placement(placement, benchmark, save_path="debug_ibm01.png", plc=plc)
        print("Saved: debug_ibm01.png")


if __name__ == "__main__":
    main()
```

Запуск:
```bash
uv run python scripts/dev_test.py
```

Преимущества над CLI:
- Видишь **разложение** proxy на компоненты — поймёшь, где слабая ось
- Можно ставить breakpoints через `import pdb; pdb.set_trace()`
- Можно держать benchmark в памяти и итеративно гонять placer

---

## Шаг 10 — Что дальше

1. **Прочитать reference**: [submissions/will_seed/placer.py](../submissions/will_seed/placer.py) полностью.
2. **Записать baseline в журнал**: запусти `--all` для greedy_row и will_seed, занеси в [todo.md](todo.md) секцию "Журнал экспериментов".
3. **Выбрать стратегию**: см. [ALGORITHMS.md](ALGORITHMS.md) для обзора подходов и [todo.md](todo.md) для нашего плана (DREAMPlace seed → LNS).
4. **Реализовать первую итерацию** placer'а: например, простой LNS с random destroy + greedy repair поверх will_seed legalization.
5. **Замерить + записать**.
6. **Итерировать**.

Перед сабмитом — обязательно [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).

---

## Если что-то сломалось

См. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — там типичные проблемы по этапам:
- Setup (uv не нашёлся, submodule пуст, sync падает на torch)
- Запуск placer'а (evaluator не находит класс)
- Validation / overlaps (float-precision, NaN в proxy)
- Runtime (>1 час)
- DQ-ловушки (Mike Gao / BakaBobo / vmallela кейсы)

---

## Альтернатива: работать без submodule

Если submodule не качается / не нужен — можно работать на pre-processed `.pt`:

```python
from macro_place.benchmark import Benchmark

# 17 IBM + 4 NG45 + ASAP7 версии — все pre-processed
benchmark = Benchmark.load("benchmarks/processed/public/ibm01.pt")

# Полная разработка алгоритма возможна, но:
# - Нет PlacementCost (плс) → нельзя считать proxy cost
# - CLI evaluate не запустит этот benchmark (он грузит из external/...)

# Зато можно валидировать и визуализировать:
from macro_place.utils import validate_placement
ok, viol = validate_placement(my_placement, benchmark)
```

Для полноценной разработки всё равно нужен submodule. Но для прототипирования логики и тестов — `.pt` файлов достаточно.
