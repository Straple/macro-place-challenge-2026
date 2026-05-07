# Промпт для агента: рандомизированный constructive initializer

Этот файл — self-contained брифинг для нового агента, чтобы он автономно
реализовал и протестировал альтернативный (рандомизированный, без оверлап-чека)
initial-pos generator для gradient-based placer'а.

---

## TL;DR

В проекте Macro Placement Challenge 2026 (команда **Straple**) уже работает
GPU pure-gradient placer (репо `~/Documents/Projects/macro-place-challenge-2026`,
основной entry point `submissions/straple/placer.py` с preset
`STRAPLE_PRESET=gradient_batch`). Ему нужно **разнообразное и качественное
начальное приближение** для K параллельных seeds. Сейчас инициализация —
наивная: для каждого Louvain-кластера выбирается anchor (centroid или
random grid) и макросы рассыпаются вокруг него gaussian-шумом.

Твоя задача — **реализовать рандомизированный constructive initializer**
без проверки overlap'ов, который даст K разных хороших по wirelength
placement'ов. Эти placement'ы будут использованы в gradient_batch.py как
стартовое `pos_init [K, n_total, 2]`.

---

## Контекст

### Где работаем
- Только на сервере: `ssh evyukhnevich@111.88.155.213` (T4 16 GB, 16 vCPU, 64 GB RAM,
  uv env в `~/macro-place/.venv`).
- Никаких локальных runs. Push кода через `./run_remote.sh push`.
- Long runs: `nohup ... > .remote_runs/$tag.log 2>&1 &` через ssh.

### Структура текущего пайплайна

```
evaluate placer.py --bench ibmXX
    │
    ▼
StraplePlacer.place(benchmark)            # submissions/straple/placer.py
    │  preset = "gradient_batch" (default)
    ▼
_place_gradient_batch(benchmark, plc):
    │
    │  1. Auto-K probe (Phase 1 shrink + Phase 1.5 grow + Phase 2 binary search)
    │     → optimal K aligned to multiple of 32, fits in fill_frac×VRAM.
    │
    │  2. Build proxy_pkgs:
    │       edges_pkg     = build_routing_edges_full(plc, name_to_global, n_total)
    │       smooth_matrices = build_smooth_matrices(rows, cols, smooth_range)
    │       routing_consts  = build_routing_consts(plc, ...)
    │       wl_pkg          = build_wl_pkg_full(plc, name_to_global, n_total)
    │
    │  3. gradient_batch(benchmark, plc, K=K, time_budget=3300, ...)
    │     ← *** initial pos sit inside this function, см. ниже ***
    │
    ▼
4. Pool 16 workers: legalize-only (C++ min-displacement) all K seeds.
5. Build pos_full_K [K, n_total, 2] on GPU + invalid_mask via pairwise overlap.
6. gpu_proxy_batched(pos_full_K, ...) → proxy_K [K]; argmin over valid → best.
7. Return best placement.
```

### Где сейчас initialization

`submissions/straple/gradient_batch.py`, около строк 195–245:

```python
# Cluster the netlist (Louvain over hyperedges)
cluster_id, num_clusters, _ = cluster_macros(
    benchmark, method="louvain", seed=seed,
    max_net_size=20, target_num_clusters=cluster_target)

# Anchors per K — three flavors (centroid / grid / shuffled grid) + jitter
anchors_K = ...  # [K, num_clusters, 2]

# spawn pos: each macro placed at its cluster anchor + Gaussian noise
sigma_per_macro = ...  # [n_active]
pos_init = np.zeros((K, n_active, 2), dtype=np.float32)
for k in range(K):
    anchor_pos_k = anchors_K[k][cluster_id_np]   # [n, 2]
    noise_k = rng.normal(0.0, 1.0, size=(n_active, 2)) * sigma_per_macro[:, None]
    pos_init[k] = anchor_pos_k + noise_k
# clamp в canvas, restore fixed (port positions)
pos = torch.tensor(pos_init, requires_grad=True, device=dev)
```

То есть сейчас init = "макрос ровно в anchor своего кластера + малый
gaussian jitter". Это **диверсифицированно по anchors**, но не учитывает
структуру netlist при размещении конкретного макроса.

### Главное

1. Initialize **должен возвращать `pos_init [K, n_active, 2]` numpy float32**
   с фиксированными macros на их original позициях. Movable mask:
   `benchmark.get_movable_mask()[:n_active]`. Fixed pos:
   `benchmark.macro_positions[:n_active]`.
2. **Никакого overlap check** — placement'ы могут пересекаться. Gradient'у
   (с overlap penalty) задача их разнести.
3. **Diversity per K**: разные seeds должны давать разные placement'ы
   (разные basins). Иначе все 384 seeds сойдутся в один и тот же optimum.
4. **Минимум WL**: placement должен иметь хорошие сетевые расстояния,
   чтобы gradient в первой фазе не тратил время на "разнесение наугад".

---

## Задача

Реализовать **четыре альтернативных initializer'а** в новом файле
`submissions/straple/init_strategies.py`. Каждый принимает benchmark + plc
+ K + seed → возвращает `pos_init [K, n_active, 2]`.

### Стратегия 1: Constructive greedy with Boltzmann sampling

Принцип: размещаем макросы по одному в порядке убывания связности, каждый
макрос — в "удачную" позицию (низкое предсказанное HPWL) с **рандомизацией
через temperature-controlled sampling**.

Алгоритм (pseudo):
```
order = sorted(macros by net-degree, desc)
for k in range(K):
    placed = {fixed_macros}
    pos[k] = fixed_pos
    for m in order if movable:
        # Find connected macros that are already placed
        connected = [n for n in neighbors_via_nets(m) if n in placed]
        if connected:
            # Build candidate score for each grid cell:
            #   score(cell) = -sum HPWL_increase(m at cell, n) for n in connected
            scores = compute_grid_scores(m, connected, pos[k])
        else:
            scores = uniform(grid)
        # Boltzmann sample from top-N cells
        cell = sample_softmax(scores, temperature=T)
        pos[k][m] = cell_center + small_jitter
        placed.add(m)
```

Параметры:
- `T` (temperature): высокая T → больше random; низкая T → greedy.
  Per-K vary T в `[0.5, 2.0]` для diversity.
- `top_n`: 5-20 cells.
- `grid_resolution`: 16×16 или 32×32 для оценки scores.

Bench size: ibm01 n=1140 макросов. Per-macro work ~ O(neighbors × grid_cells).
~5-10 ms на placement, K=384 → ~3 sec.

**Pros**: учитывает netlist при placement.
**Cons**: greedy bias может дать локально хорошие но глобально плохие
placement'ы. Нужно diversification через Boltzmann + per-K temperature jitter.

### Стратегия 2: Spectral placement (Quadratic optimization)

Решить задачу:
```
min Σ_(i,j ∈ net) w_n · ||pos_i - pos_j||^2
   s.t.  fixed_pos boundary
```

Это minimization quadratic form `xᵀLx` где L — графовая Laplacian (по
netlist hyperedges, веса 1/(net_size - 1) для clique reduction). Решение —
solve `L · x = b` (b = boundary contributions от fixed pins).

```python
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Build adjacency from nets (clique expansion or star)
A = build_adjacency_from_nets(plc, n_movable)   # sparse [n_movable, n_movable]
D = sp.diags(A.sum(axis=1))
L = (D - A).tocsr()

# Boundary: fixed macros / ports contribute b
b_x, b_y = compute_boundary_force(plc, fixed_pos)
x = spla.cg(L, b_x, atol=1e-5)[0]
y = spla.cg(L, b_y, atol=1e-5)[0]
```

Time: O(N · iters_CG) = ~50-200 ms на ibm01 (N=1140). Дает **один
deterministic placement** (без overlap, WL-optimal в quadratic sense).

**Randomization для K seeds**:
- (a) Perturb edge weights w_n с gaussian noise.
- (b) Perturb b с gaussian.
- (c) Add eigenvector к решению (random combination of top eigenvectors).

K placements за ~30 sec.

**Pros**: качественные WL-минимизированные solutions.
**Cons**: все K placements **похожие** без сильной randomization → low
diversity.

### Стратегия 3: Recursive min-cut bisection (Capo-style)

Алгоритм:
1. Cut netlist на 2 halves через `hMETIS` или spectral cut (Fiedler vector).
2. Assign each half в свою половину canvas (left/right или top/bottom).
3. Recurse on each half.

`hMETIS` доступен в `external/MacroPlacement/Flows/util/`. Spectral cut
проще: 2nd smallest eigenvector L даёт partition.

Pseudo:
```python
def bisect(macros, region):
    if len(macros) < threshold:
        place macros in region uniformly
        return
    cut_dir = random_choice(["horizontal", "vertical"])
    halves = spectral_cut(macros)
    sub_regions = split_region(region, cut_dir, halves)
    for half, sub_region in zip(halves, sub_regions):
        bisect(half, sub_region)
```

Random выбор cut direction + tie-break → diversity.

Time: O(N · log N · cut_cost). Per K ~ 1 sec.

**Pros**: классический подход, хорошее качество.
**Cons**: complex implementation, eigenvalue decomposition slowly без
tuning.

### Стратегия 4: Force-directed WL warmup

Тривиальная: random init + N итераций градиентного спуска **только
по WL** (overlap_w=0, density_w=0).

Это уже частично есть в `gradient_batch.py`! Можем включить через
изменение фаз:

```python
# Phase 0: pure-WL warmup (50-100 steps, overlap_w=0, density_w=0)
# Phase 1: spreading
# Phase 2: refining
# Phase 3: settling
```

Time: 50 steps × ~0.4 sec/step = 20 sec.  Per K — да.

**Pros**: trivial реализация, использует existing GPU code.
**Cons**: уже почти что мы делаем сейчас (Phase 1 имеет low overlap_w).
Marginal gain.

### Рекомендация

**Hybrid: Spectral + Constructive sampling**:
- Spectral даёт WL-optimal без overlap baseline.
- Constructive sampling даёт diversity через Boltzmann.

В `gradient_batch.py` подавать **смесь начальных placements**:
- 1/4 K seeds: spectral (с eigenvector noise).
- 1/4 K seeds: constructive с T=0.5 (greedy).
- 1/4 K seeds: constructive с T=1.0 (balanced).
- 1/4 K seeds: constructive с T=2.0 (random).

Получим K seeds с **уже хорошим WL и diverse basins**.

---

## Что нужно сделать

### Шаг 1: реализовать `init_strategies.py`

```python
# submissions/straple/init_strategies.py

def constructive_init(benchmark, plc, K: int, seed: int = 42,
                       temperature_K: np.ndarray = None,
                       grid_resolution: int = 16,
                       top_n: int = 10) -> np.ndarray:
    """K placements via greedy constructive + Boltzmann sampling.

    Returns: pos_init [K, n_active, 2] float32.
    """
    ...

def spectral_init(benchmark, plc, K: int, seed: int = 42,
                   eigvec_noise: float = 0.05) -> np.ndarray:
    """K placements via spectral solver with random perturbation per K.

    Returns: pos_init [K, n_active, 2] float32.
    """
    ...

def hybrid_init(benchmark, plc, K: int, seed: int = 42,
                 spectral_frac: float = 0.25,
                 temperature_range: tuple = (0.5, 2.0)) -> np.ndarray:
    """Combine spectral and constructive initial placements."""
    ...
```

### Шаг 2: подключить как опцию в `gradient_batch.py`

```python
init_method = os.environ.get("STRAPLE_BATCH_INIT", "louvain")
# louvain (default), constructive, spectral, hybrid
```

Передавать `pos_init` в существующую логику. Сохранить fallback на
текущий Louvain-anchor init.

### Шаг 3: smoke test

Тестировать через прямой вызов:

```python
# scripts/test_init_strategies.py
from init_strategies import constructive_init, spectral_init, hybrid_init

bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")

for fn in [constructive_init, spectral_init, hybrid_init]:
    t0 = time.time()
    pos = fn(bench, plc, K=64, seed=42)
    print(f"{fn.__name__}: {time.time()-t0:.2f}s, shape={pos.shape}")
    # Compute initial WL via gpu_proxy_batched
    from gpu_proxy import gpu_proxy_batched, build_routing_edges_full, ...
    proxy_K, comp = gpu_proxy_batched(...)
    print(f"  WL min/median/max: {comp['wl'].min():.4f} / "
          f"{comp['wl'].median():.4f} / {comp['wl'].max():.4f}")
```

Цель: показать что новые initializer'ы дают **lower WL** (или comparable
WL с большей diversity) чем Louvain init.

### Шаг 4: full submission run сравнить

Сравни `STRAPLE_BATCH_INIT=louvain` (default) vs `=hybrid` на ibm01 с
time_budget=600s. Проверь:
- Best proxy после legalize.
- Distribution stats (median, p25, p75).
- Gradient convergence speed (early steps should hit lower wl быстрее).

Если hybrid wins на ibm01 — повтори на ibm14 (medium) и ibm18 (large).

---

## Гипотеза: почему это должно помочь

Текущий init: макросы группируются по Louvain-кластерам. Кластер anchor —
random grid. Макросы в одном кластере **связаны** netlist'ами (this is the
whole point of Louvain), поэтому при random anchor эти связи **искажают**
оптимальное расположение кластера.

Spectral init: позиции макросов пропорциональны их связностям. Близко
связанные макросы оказываются рядом. WL автоматически low.

Constructive: каждый макрос размещается видя кого он "тянет" к себе.
Постепенно строится placement, где сильные связи удовлетворены.

Hybrid даёт **быстрее convergence** для gradient — он стартует из basin
где WL уже почти optimal, остаётся только разнести overlap'ы и
оптимизировать density. Это потенциально даёт **lower final proxy**
за тот же time_budget.

---

## Ограничения и риски

1. **Soft macros**: они подвижные но не имеют overlap с hard. Init должен
   корректно обрабатывать soft (тоже размещать через spectral или
   constructive).
2. **Ports / fixed**: не двигать! Их позиции в `benchmark.macro_positions`,
   movable mask = False.
3. **Net resolution**: некоторые pins могут быть orphan (нет в name_to_global).
   Skip как в существующем коде.
4. **Большие benches**: на ibm17/18 N=2700-3500 макросов. Spectral CG
   может быть медленный без preconditioner. Тестировать.
5. **Не overdo dev time**: цель — рабочий MVP за 2-3 дня. Если spectral
   дорого — focus на constructive.

---

## Команды сервера

```bash
# Push code
./run_remote.sh push

# Smoke test
ssh evyukhnevich@111.88.155.213 "cd macro-place && export PATH=\$HOME/.local/bin:\$PATH && \
    uv run --no-progress python scripts/test_init_strategies.py 2>&1 | tail -30"

# Long submission run (10 min smoke)
ssh evyukhnevich@111.88.155.213 'cd macro-place && export PATH=$HOME/.local/bin:$PATH && \
    nohup bash -c "STRAPLE_PRESET=gradient_batch STRAPLE_BATCH_TIME_BUDGET=600 \
    STRAPLE_BATCH_INIT=hybrid STRAPLE_VERBOSE=1 \
    uv run --no-progress evaluate submissions/straple/placer.py -b ibm01" \
    > .remote_runs/init_test.log 2>&1 < /dev/null & echo spawned'
```

---

## Что НЕ делать

- Не переписывать существующий Louvain init — оставить как fallback.
- Не тестировать локально — только на сервере (T4 GPU).
- Не делать legalize в init — это работа C++ legalize позже.
- Не оптимизировать density в init — это работа gradient.
- Не использовать matplotlib в init code — только raw numpy/torch.

Удачи!
