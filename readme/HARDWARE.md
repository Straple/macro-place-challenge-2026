# Hardware — про железо для оценки и как его использовать

> Где будут оценивать, как написать код, который правильно использует CPU и GPU, как тестировать перед сабмитом.

---

## Где будут оценивать сабмишен

Из [README.md:73](../README.md#L73):

> **AMD EPYC 9655P с 16 ядрами + 100GB памяти и NVIDIA RTX 6000 Ada 48GB**

Это **серверы организаторов**. Ты лично туда не подключаешься. Когда судьи запускают твой код:

1. Они клонируют твой приватный репо (которым ты с ними поделился)
2. Запускают через docker (см. [../eval_docker/](../eval_docker/))
3. Внутри docker-контейнера — конкретное железо: 16 CPU, 100GB RAM, 1 GPU
4. Лимит на бенчмарк: **1 час**

### Важно понимать

- **Ты не имеешь доступа** к этому серверу. Тестируешь у себя на машине.
- **Твой код должен работать на этом железе** — учитывать 16 ядер, 100GB RAM, 1 GPU 48GB.
- **Self-reported результаты ≠ verified**. У `vmallela` self-reported 1.1172, а на eval железе — 1.4152 (на 27% хуже). Причина: single-threaded numpy на 16-core машине → 16× недогрузка. См. [TROUBLESHOOTING.md](TROUBLESHOOTING.md#self-reported-112-на-их-железе-142-как-vmallela).

---

## Спецификация eval-машины (детали)

### CPU: AMD EPYC 9655P
- **16 ядер** (Zen 5, 32 потока с SMT)
- Базовая частота: 2.6 GHz, boost до 4.5 GHz
- Большой L3 cache (~64 MB)
- **96 PCIe Gen5 lanes** (быстрый I/O)

### RAM: 100 GB
- Для большинства IBM-бенчмарков (246-537 макросов, 7k-16k нетов) — это **много**, забить сложно. Не страшись держать структуры в памяти, кешировать proxy components, держать копии placement.

### GPU: NVIDIA RTX 6000 Ada Generation 48 GB
- **48 GB GDDR6** памяти (огромный — большие тензоры/батчи)
- 18,176 CUDA cores
- 568 Tensor cores (5-й gen)
- Compute Capability 8.9 (Ada Lovelace)
- Поддерживает FP32, BF16, FP16, FP8, INT8

> Это **профессиональная карта**. По производительности ≈ RTX 4090, но с 48GB памяти. Ты дома такого, скорее всего, не имеешь — будет существенное расхождение в **скорости**, но не в **корректности**.

---

## Что значит "иметь дома меньше железа"

| У тебя | На eval | Что будет |
|---|---|---|
| 8 ядер CPU | 16 ядер | Твой замер времени **в 2× больше** реального. То, что у тебя 30 минут — у судей 15 минут. |
| 16 GB RAM | 100 GB RAM | Хорошо, у тебя ограниченнее → если влезает, у них тем более влезает. |
| Mac M-чип (нет CUDA) | RTX 6000 Ada | Если используешь GPU — у тебя через MPS, у них через CUDA. **Лучше тестировать на машине с CUDA.** |
| RTX 3060 12GB | RTX 6000 Ada 48GB | Корректность та же, скорость ~3× медленнее, не во всё влезаешь по памяти |
| Никакого GPU | RTX 6000 Ada | Если код только CPU — работает везде. Если есть GPU-путь — оба пути должны быть протестированы (`if torch.cuda.is_available(): ...`). |

### Если совсем нет железа близкого к eval

- **Облако**: AWS p4d/p4de/p5, Google Cloud A2, Lambda Labs — час A100/H100 стоит ~$1-3
- **Colab Pro+**: T4 / A100 (зависит от подписки)
- **Vast.ai / RunPod**: дешевле, по часам
- **Kaggle Notebooks**: бесплатные T4/P100 (30h/неделю)

Для финального verification **обязательно прогон на схожем железе** (16 cores + 1 GPU) — иначе риск self-reported≠verified.

---

## Как использовать GPU в своём коде

### Базовая проверка

```python
import torch

print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Памяти: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Универсальный device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Перенос данных на GPU

```python
# Исходный benchmark на CPU
positions = benchmark.macro_positions       # CPU tensor

# Переносим на GPU
positions_gpu = positions.to(device)
sizes_gpu = benchmark.macro_sizes.to(device)

# Все операции теперь на GPU
diffs = positions_gpu[:, None] - positions_gpu[None, :]   # [N, N, 2]
dists = diffs.norm(dim=-1)                                # [N, N]

# Возвращаем результат на CPU для evaluator'а
final_placement = positions_gpu.cpu()
return final_placement
```

> **Важно:** evaluator ожидает CPU tensor на возврате из `place()`. Не забудь `.cpu()` в конце.

### Когда GPU реально помогает

✅ **Помогает:**
- Большие батчевые операции на тензорах: pairwise distances, overlap detection между сотнями макросов
- Многократная свёртка / sliding window для density
- Multi-start: параллельно держать N размещений на GPU
- Свой analytical solver (gradient descent на smooth surrogates) — это путь DREAMPlace
- Любые операции на сетке `grid_rows × grid_cols`

❌ **Не помогает (или замедляет):**
- Маленькие операции: пара макросов, одно число — overhead на трансфер CPU↔GPU съедает выигрыш
- Скалярные циклы (Python `for` по макросам): GPU тут не используется, CPU быстрее
- Вызовы C++ функций из TILOS (`plc.get_cost()`) — они на CPU, GPU тут бесполезен
- Случайные выборки малого числа элементов

### Профилирование GPU memory

```python
import torch

torch.cuda.reset_peak_memory_stats()

# ... твой код ...

peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak:.2f} GB")
```

48 GB — это много, но если делаешь pairwise [N, N, 2] для N=10000 макросов в FP32 → 800 MB. Обычно влезает.

---

## Как использовать многоядерность CPU

### Контроль числа потоков (важно!)

PyTorch и numpy по умолчанию используют **один поток**. На 16-core машине это огромная потеря.

```python
import torch
import os

# Сообщи torch / OMP / MKL что у тебя 16 ядер
torch.set_num_threads(16)
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
```

> Иначе твоё `torch.matmul` будет работать на одном ядре, и proxy 1.40 на твоей машине превратится в proxy 1.40, но **в 16 раз медленнее** на eval-машине → DQ по runtime.

### Multiprocessing для multi-start

Запустить N независимых runs параллельно (multi-start placement):

```python
import torch.multiprocessing as mp

def run_one_seed(seed: int, benchmark) -> tuple[float, torch.Tensor]:
    placer = StraplePlacer(seed=seed)
    placement = placer.place(benchmark)
    cost = compute_proxy_cost(placement, benchmark, plc)["proxy_cost"]
    return cost, placement


def multi_start_place(benchmark, n_starts=8):
    seeds = list(range(n_starts))
    with mp.Pool(processes=8) as pool:
        results = pool.starmap(run_one_seed, [(s, benchmark) for s in seeds])
    best_cost, best_placement = min(results, key=lambda x: x[0])
    return best_placement
```

> Внимание: в Python нельзя сериализовать всё через multiprocessing — `plc` (PlacementCost из C++) может не быть pickleable. Тогда нужно загружать `plc` внутри каждого worker'а. Или использовать `concurrent.futures.ThreadPoolExecutor` (но GIL → это не для CPU-bound).

### `torch.compile` (PyTorch 2.x)

Если у тебя есть hot-path с torch операциями:

```python
@torch.compile
def compute_overlap(positions, sizes):
    # ... сложные тензорные операции ...
    return overlap_score

# Первый вызов медленный (компиляция), потом быстро
```

Иногда даёт 2-10× ускорение для горячих циклов.

---

## eval_docker — что внутри docker-окружения судей

Из [../eval_docker/](../eval_docker/):

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04
# ... ставит python 3.10, uv, копирует код, evaluator, бенчмарки
```

### run_eval.sh
```bash
docker run \
    --network none \                                # АИРГАП: нет интернета
    --gpus all \                                    # доступ к GPU
    --memory 64g \                                  # лимит RAM 64GB (не 100, осторожно!)
    --cpus 16 \                                     # лимит 16 CPU
    --rm \
    -v "${SUBMISSION_DIR}:/submission:ro" \
    -v "${RESULTS_DIR}:/results" \
    eval-image \
    --timeout 7200 \                                # 2 часа на весь --all
    python -m macro_place.evaluate /submission/placer.py --all
```

### Из этого следует

1. **Нет интернета** — твой код не может ничего скачать. Все веса нейронок, модели — должны быть в репо.
2. **Лимит 64 GB RAM** — не 100, как написано в README. Не хвастайся всеми ресурсами.
3. **2 часа на --all (7200 сек)** — это agregated лимит, плюс per-benchmark 1 час. Если уложишься в 1 час на 17 benchmarks по очереди (1 час × 17 = 17 часов) — точно превысишь, но если каждый тратит ~3-5 минут — норма.
4. **Все 16 ядер доступны** — используй их. CPU-quota в docker реальная.
5. **GPU доступен** — `--gpus all`.

---

## Что делать в `pyproject.toml` для GPU

Если твой код использует CUDA-специфичные пакеты (`dreamplace`, `cupy`):

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    # стандартное torch — на eval будет CUDA версия (т.к. в docker'е nvidia base)

    # Если нужен dreamplace:
    # "dreamplace @ git+https://github.com/limbo018/DREAMPlace.git",

    # Если нужен cupy с CUDA 12:
    # "cupy-cuda12x>=13.0",
]
```

> ⚠️ Проверяй на реально CUDA-machine, что `uv sync` ставит правильную torch + что код запускается. На macOS CUDA нет — на Mac тестируй CPU-путь, а финальную проверку с GPU делай на Linux.

---

## Тестирование производительности

### Замер времени per-benchmark

```python
import time

t0 = time.perf_counter()
placement = placer.place(benchmark)
t = time.perf_counter() - t0
print(f"{benchmark.name}: {t:.2f}s")
```

Запусти на всех 17 IBM, проверь, что все в **меньше 1 час** (с запасом — 30 мин).

### Профилирование hot-path

```bash
# cProfile (быстро + наглядно)
uv run python -m cProfile -o profile.out scripts/dev_test.py
uv run python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(30)"

# line_profiler (по строкам, требует @profile decorator)
uv pip install line_profiler
# Добавь @profile в hot-функцию
uv run kernprof -l scripts/dev_test.py
uv run python -m line_profiler dev_test.py.lprof
```

Типичные узкие места:
- `plc.get_cost()` (C++ wrapper) — основное время на больших benchmark'ах
- Pairwise overlap check O(N²) — заменяй на spatial hash / kd-tree
- `plc.optimize_stdcells()` — минуты, не вызывай каждую итерацию

### GPU memory profile

```python
import torch

torch.cuda.reset_peak_memory_stats()
placement = placer.place(benchmark)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU: {peak:.2f} GB")
```

Должно быть < 40 GB (запас на 48 GB карте).

### CPU utilization

На macOS:
```bash
# В отдельном терминале во время прогона:
top -pid $(pgrep -f evaluate) -stats cpu,mem,threads
```

На Linux:
```bash
htop -p $(pgrep -f evaluate)
```

Если видишь, что используется ~100% (одно ядро) — у тебя GIL/single-threaded. Если ~1600% (16 ядер × 100%) — отлично, ты задействовал всё.

---

## Reality check — какую часть ресурсов реально использует топ leaderboard

Из [README.md leaderboard](../README.md#L233):

| # | Команда | Score | Runtime | Hardware |
|---|---|---|---|---|
| 2 | MTK (DreamPlace++) | 1.2818 | **37s/bench** | GPU |
| 5 | Shoom (MultiDREAMPlace) | 1.3381 | 350s/bench | GPU (predict) |
| 8 | Beatel (ePlace-Lite) | 1.3913 | 155s/bench | GPU |
| 10 | UT Austin AS | 1.4076 | **17s/bench** | GPU |
| 11 | ByteDancer | 1.4151 | 38min/bench | CPU (медленный) |
| 12 | vmallela | 1.4152 | **12 hours total** | CPU single-threaded |

Уроки:
- **GPU + аналитический подход = 17-37 секунд/бенчмарк** → есть огромный запас по времени для multi-start или refinement
- **CPU multi-threaded SA/LNS = 5-30 минут/бенчмарк** → реально, но впритык
- **Single-threaded Python = 12 часов** на всё → точно DQ

> Если планируешь идти CPU-путём — обязательно multi-threading (`torch.set_num_threads(16)`). Если есть GPU — используй его, это ваш самый большой leverage.

---

## TL;DR — что делать прямо сейчас

1. Проверь, что у тебя есть: `python3 --version`, `git`, `uv` (см. [QUICKSTART.md](QUICKSTART.md)).
2. Проверь GPU: `python -c "import torch; print(torch.cuda.is_available())"`.
3. Если GPU есть — твой план: DREAMPlace seed на GPU + LNS refinement.
4. Если GPU нет — твой план: умный CPU-код + multi-threaded LNS, или арендовать облако на финальные замеры.
5. **Всегда** в начале placer'а: `torch.set_num_threads(16); os.environ["OMP_NUM_THREADS"] = "16"`.
6. Перед сабмитом — прогон на железе близком к eval (16 cores + GPU).
