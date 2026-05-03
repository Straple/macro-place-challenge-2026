# Troubleshooting — типичные проблемы и решения

> Что делать, когда сломалось. Сгруппировано по этапам.

## Setup / окружение

### `uv: command not found`

Установить:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Перезапустить терминал или: source ~/.zshrc
```

### `git submodule update` молчит, `external/MacroPlacement/` пуст

Проверь, есть ли URL в .gitmodules:
```bash
cat .gitmodules
git submodule status
git submodule update --init --recursive external/MacroPlacement
```

Если идёт долго (~500MB) — это норма. Если упал по сети — повтори с `--depth=1`:
```bash
git submodule update --init --recursive --depth 1 external/MacroPlacement
```

### `uv sync` падает на сборке pytorch / cuda

```bash
# Принудительно CPU-only torch (если CUDA не нужна):
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# С CUDA 12.4 (как в eval_docker):
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### `ImportError: cannot import name 'PlacementCost'`

Это C++ binding из TILOS submodule. Проверь:
```bash
ls external/MacroPlacement/CodeElements/Plc_client/
# Должен быть plc_client.so или plc_wrapper_main
```

Если нет — нужно собрать. См. [TILOS README](https://github.com/TILOS-AI-Institute/MacroPlacement/tree/main/CodeElements/Plc_client).

---

## Запуск placer'а

### `evaluator не находит мой класс`

`uv run evaluate <file>` берёт **первый** класс с методом `place()` в файле. Проверь:

```python
# ✅ Правильно
class MyPlacer:
    def place(self, benchmark): ...

# ❌ Неправильно (нет .place())
class MyPlacer:
    def run(self, benchmark): ...

# ❌ Неправильно (другой класс идёт раньше и тоже имеет .place())
class HelperWithPlace:
    def place(self, x): ...   # evaluator возьмёт ЭТО

class MyPlacer:
    def place(self, benchmark): ...
```

Решение: переименовать helper-метод, или поставить `MyPlacer` первым в файле.

### `__init__() требует аргументы`

Evaluator зовёт `placer_cls()` без аргументов. Если нужны параметры:

```python
class MyPlacer:
    def __init__(self, seed=42, iters=1000):  # ВСЕ с дефолтами
        self.seed = seed
        self.iters = iters
```

---

## Validation / overlaps

### `overlap_count > 0` хотя я вроде всё разнёс

**Причина 1: float-precision на границе.** Два макроса касаются друг друга, но из-за float arithmetic edge-coordinates чуть-чуть пересекаются.

Решение — добавлять gap при легализации:
```python
GAP = 1e-3
# После размещения макроса, следующий ставим на min_x + width + GAP
```

Так делает [submissions/examples/greedy_row_placer.py](../submissions/examples/greedy_row_placer.py).

### `overlap_count` падает, но не до 0

**Причина 2: legalization не успела сойтись.** Spiral search в will_seed имеет лимит радиусов — может не найти место в плотном dataset.

Решение:
- Увеличь max_radius
- Или начинай legalization не с Random initial, а с placement, у которого структура близка к легальной (DRP, например)
- Или используй детерминированный shelf-pack как fallback

### `validate_placement` ругается на bounds

```python
# ❌ Ошибка: координата = угол макроса, а не центр
placement[i] = torch.tensor([0.0, 0.0])  # макро вылазит за canvas (его центр в углу)

# ✅ Правильно: центр должен быть достаточно далеко от края
half_w, half_h = sizes[i] / 2
placement[i] = torch.tensor([half_w, half_h])  # макро прижат к (0,0) углом
```

### `validate_placement` ругается на NaN/Inf

Обычно: деление на 0 в твоём cost'e или проектирование в degenerate position. Найди источник:

```python
# Перед сложными расчётами:
assert not torch.isnan(placement).any(), "NaN в placement"
assert not torch.isinf(placement).any(), "Inf в placement"
```

### `validate_placement` говорит "Fixed macro moved"

Случайно сдвинул зафиксированный. После всех ходов — restore:
```python
fixed_mask = benchmark.macro_fixed
placement[fixed_mask] = benchmark.macro_positions[fixed_mask]
```

---

## Proxy cost

### `compute_proxy_cost` возвращает `inf` или `nan`

- Один из net'ов имеет 0 пинов или duplicate-positions → проверь nets
- Макрос точно на границе canvas → density grid считает 0 → деление на 0
- Какой-то макро имеет нулевой size

Лови на месте:
```python
costs = compute_proxy_cost(placement, benchmark, plc)
for k, v in costs.items():
    if isinstance(v, float) and (v != v or v == float("inf")):
        print(f"BAD: {k} = {v}")
```

### `compute_proxy_cost` ОЧЕНЬ медленный (минуты на вызов)

Скорее всего, ты неявно вызываешь `plc.optimize_stdcells()` много раз. Проверь, не зовёшь ли его на каждой итерации LNS.

Решение:
- Зови `optimize_stdcells` редко (раз в N итераций)
- Снижай `num_steps=[100, 100, 100]` → `[20, 20, 20]`
- Или пиши свой быстрый soft updater на GPU

### Proxy cost разный при одинаковом placement

Не должен. Если разный — наверняка `plc` хранит state между вызовами. Проверь, что не двигаешь soft внутри `compute_proxy_cost`.

---

## Runtime

### Runtime > 1 час на ibm17/ibm18

Лимит: **1 час на benchmark**. Большие benchmark'и (537 макросов) — узкое место.

Стратегии:
1. **Профайл**: `python -m cProfile -o p.out <script>; python -c "import pstats; pstats.Stats('p.out').sort_stats('cumulative').print_stats(30)"`
2. **GPU**: перенести cost-incremental update и overlap check на CUDA
3. **Adaptive iterations**: на больших benchmark'ах меньше итераций LNS
4. **Лучший seed**: чем лучше seed → тем меньше итераций нужно
5. **Параллелизм**: 16 ядер у evaluator'а — используй `torch.multiprocessing` для multi-start

### Один benchmark отваливается по timeout

Если конкретно один benchmark (например ibm14) идёт >>остальных — там может быть pathological case. Проверь:
- Не случилось ли infinite loop в legalization?
- Не растёт ли число операций > O(N²)?

---

## DQ-ловушки (избежать!)

### "В моём env работает 1.32, в eval env DQ" (как Mike Gao)

**Симптом:** Self-reported хороший, но в eval env — overlaps или ошибки.

**Причины:**
- DREAMPlace silently fails (CUDA не доступна / ABI mismatch) → возвращает unlegalized
- Какая-то твоя зависимость не установлена в чистом env
- Hardcoded path к моделям/данным, которых нет в репо

**Митигация:** прогон в чистом clone до сабмита:
```bash
cd /tmp
git clone https://github.com/Straple/macro-place-challenge-2026 fresh
cd fresh
git checkout submission
git submodule update --init external/MacroPlacement
uv sync
uv run pytest test/test_smoke.py -v
uv run evaluate submissions/straple/placer.py --all
```

### "Импортирует несуществующий модуль" (как BakaBobo)

**Симптом:** Код использует `from macro_place.fast_proxy import ...` — этого модуля нет.

**Причина:** забыл закоммитить свой fork evaluator'а / свой helper.

**Митигация:** все свои utilities кладём **внутрь** `submissions/straple/`. Не патчим `macro_place/`.

### "Self-reported 1.12, на их железе 1.42" (как vmallela)

**Симптом:** Замеры на твоём железе сильно лучше, чем на оценочном.

**Причины:**
- Single-threaded numpy на 16-core машине → 16× медленнее
- Pure Python в random hot path
- Не используешь `torch.compile` / multiprocessing

**Митигация:** замерять на железе, близком к eval:
- 16 ядер + 100GB RAM + RTX 6000 Ada (или хотя бы 16-core machine)
- При тестах ставить `OMP_NUM_THREADS=16`

### "Зависимость отсутствует" (как ArzunPD без `networkit`)

**Симптом:** Код требует `networkit` / `dreamplace` / etc, которых нет в стандартной install.

**Митигация:** все зависимости пиши в [pyproject.toml](../pyproject.toml):
```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",
    "tqdm>=4.65.0",
    "absl-py>=1.0.0",
    # Твои дополнительные:
    "networkit>=10.0",
    "scipy>=1.10",
]
```

И **тестируй в чистом clone**: `uv sync` подхватит твои deps.

### "Хардкод под benchmark" → DQ

**Симптом:** Код содержит ветвления по имени benchmark — `if benchmark.name == "ibm17": ...`.

**Причина:** Tier 1 тесты — открытые, **те же файлы** используются у тебя и у судей ([README.md FAQ](../README.md#L281)). Соблазн — посмотреть на каждый benchmark глазами и захардкодить под него лучшие параметры или даже готовое размещение.

**Это запрещено** ([README.md:88](../README.md#L88)):
> Hardcoding solutions for specific benchmarks (must be general algorithm)

**Что DQ:**
```python
# ❌ DQ
if benchmark.name == "ibm17":
    placement = load_precomputed("ibm17_solution.pt")

# ❌ DQ — по сути тот же хардкод, но через числа
SPECIFIC_PARAMS = {
    "ibm01": {"k": 5, "T": 0.1},
    "ibm02": {"k": 8, "T": 0.3},
    ...
}
params = SPECIFIC_PARAMS[benchmark.name]

# ❌ DQ — даже хеш не помогает
if hash(benchmark.name) == 1234567:
    ...
```

**Что разрешено:**
```python
# ✅ Адаптивная логика по структуре
if benchmark.num_hard_macros > 500:
    iters = 5000  # больше итераций на больших дизайнах
else:
    iters = 2000

# ✅ Гиперпараметры в зависимости от утилизации
util = total_macro_area / canvas_area
k_destroy = max(5, int(util * 20))

# ✅ Веса GNN/RL, обученные на 16 из 17 (leave-one-out)
self.model = load_pretrained("model_v3.pt")  # один и тот же файл для всех
```

**Граница тонкая:** судьи могут проводить аудит кода. Lookup-таблицы по специфическим magic numbers, имитирующие имена benchmark'ов через размеры → подозрительно. **Лучше избегать любых per-benchmark веток, если они не выводимы из публичной структуры дизайна.**

**Митигация:** перед сабмитом — `grep -rni "ibm0\|ibm1\|ariane\|nvdla\|mempool" submissions/straple/` — должны быть только в комментариях/тестах, не в логике.

### "Self-reported верифицируется?" — да

Так как тесты открытые (Tier 1), судьи прогоняют **те же файлы** что у тебя. Если у тебя AVG = 1.4 и нет проблем с зависимостями/threading'ом, у судей будет +- то же.

**Уроки из leaderboard:**
- `Cezar (ReFine)` — verified 1.2224 vs self-reported 1.0666: 12% разница (разные параметры? разное железо? оспаривает результат)
- `vmallela` — verified 1.4152 vs self-reported 1.1172: 27% разница (single-threaded numpy на 16-core eval-машине)
- `MTK` — verified лучше self-reported (1.282 vs 1.317): просто скромничал

Чтобы **избежать сюрпризов**:
- `torch.set_num_threads(16)` в начале placer'а
- `OMP_NUM_THREADS=16` в env
- Прогон в чистом clone (имитация eval-окружения) — см. [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)

---

## Tier 2 / ORFS

### `evaluate_with_orfs.py`: "ORFS not found"

```bash
# ORFS должен лежать рядом с репо:
cd ..
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd macro-place-challenge-2026
```

Или явно:
```bash
uv run python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 \
    --orfs-root /path/to/OpenROAD-flow-scripts --no-docker
```

### ORFS падает, но Tier 1 OK

Скорее всего, твоё placement имеет какие-то особенности, через которые ORFS не может маршрутизироваться:
- Зазор между hard macros < 12μm → PDN-routing не пройдёт
- Макрос вне core area
- Pin direction конфликт (если ты крутишь ориентации)

Митигация:
- Оставлять ≥12μm зазоры в submission ([SCORING.md:78](../SCORING.md#L78))
- Не использовать R90/R270 (запрещено)
- Использовать только N/FN/FS/S ориентации

### "WNS_sub < min(WNS_SA, WNS_RP)" → DQ из Гран-при

Feasibility gate провален (см. [SCORING.md](SCORING.md)).

Митигация:
- В этом случае всё равно остаёшься в Tier 1 ranking (Proxy Prize / 2nd Place)
- Чтобы пройти gate — нужно много итераций ORFS-проверок (3-8 часов каждая) → запускать только когда proxy уже в топ-7

---

## Submission

### "Form rejected" / репо не открывается у судей

- Проверь, что репо приватный, и что добавлены судьи как Collaborators (Read access):
  - `partclxhrtmacroplace@gmail.com`
  - `will@partcl.com`
- В URL формы дай ссылку на конкретный SHA: `https://github.com/Straple/macro-place-challenge-2026/tree/<SHA>`
- Default branch = `submission` (или ту, на которую ссылаешься)

### "Прошёл дедлайн, не могу пересабмитить"

Дедлайн **жёсткий**: 21 мая 2026, 23:59 PT. После него форма закрывается.

Митигация: подавать **early submission** за неделю до дедлайна, потом в последний день — финальный апдейт.
