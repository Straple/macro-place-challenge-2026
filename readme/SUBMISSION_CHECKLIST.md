# Submission Checklist — что проверить перед сабмитом

> Запускать **за день до сабмита** и **в день сабмита**. Каждый пункт занимает минуты, но защищает от DQ.

## За неделю до дедлайна

- [ ] **Прогон `--all` на твоём железе** даёт ожидаемый proxy ≤ цели:
  ```bash
  uv run evaluate submissions/straple/placer.py --all
  ```
- [ ] Каждый из 17 IBM benchmarks: `Overlaps == 0`
- [ ] Runtime на самом большом (`ibm17` / `ibm18`) ≤ 1 час
- [ ] (Опционально) Прогон `--ng45` — Tier 2 preview

## За день до дедлайна — чистая среда

> Имитируем то, что сделают судьи в их docker'е.

- [ ] **Свежий clone в `/tmp` или `~`:**
  ```bash
  cd /tmp
  rm -rf fresh-test
  git clone https://github.com/Straple/macro-place-challenge-2026.git fresh-test
  cd fresh-test
  git checkout submission                          # твоя submission ветка
  git submodule update --init external/MacroPlacement
  uv sync
  ```
- [ ] **Smoke tests проходят:**
  ```bash
  uv run pytest test/test_smoke.py -v
  ```
- [ ] **`evaluate --all` в чистом env даёт тот же score**, что у тебя локально (±0.01):
  ```bash
  uv run evaluate submissions/straple/placer.py --all 2>&1 | tee /tmp/clean_run.log
  ```
- [ ] Никаких ошибок ImportError / FileNotFoundError / CUDA mismatch
- [ ] **Все зависимости в [pyproject.toml](../pyproject.toml):**
  ```bash
  uv pip list                                       # сверить со списком твоих imports
  ```

## За день — repo hygiene

- [ ] В [submissions/straple/](../submissions/straple/) нет лишнего: temp файлов, логов, datasets, моделей > 100MB
- [ ] Если есть pre-trained модель — лежит в репо или скачивается воспроизводимо (хеш!)
- [ ] **`.gitignore` в порядке** — нет случайно закоммиченных PNG/PT/логов
- [ ] **README в корне submission'a** ([submissions/straple/README.md](../submissions/straple/README.md)) содержит инструкцию для судей:
  ```markdown
  ## For judges
  Submission entry point: `submissions/straple/placer.py`

  Reproduce:
      git submodule update --init external/MacroPlacement
      uv sync
      uv run evaluate submissions/straple/placer.py --all
  ```
- [ ] **Никакого hardcoding под benchmark-имена** (DQ-причина):
  ```bash
  grep -rni "ibm0\|ibm1\|ariane\|nvdla\|mempool" submissions/straple/
  # Должны быть только в комментариях / тестах, не в логике
  ```

## За день — контроль ветки и доступа

- [ ] Создана/обновлена ветка `submission` в твоём приватном репо:
  ```bash
  git checkout submission
  git push -u origin submission
  ```
- [ ] **Default branch установлена** в GitHub Settings → General → Default branch = `submission`
- [ ] **Текущий SHA коммита**:
  ```bash
  git rev-parse HEAD
  # Скопировать — пойдёт в Google form URL
  ```
- [ ] **Судьи добавлены как Collaborators** (Settings → Collaborators):
  - [ ] `partclxhrtmacroplace@gmail.com` (или его GitHub username)
  - [ ] `will@partcl.com` (или его GitHub username)
  - Уровень доступа: **Read** достаточно

## День сабмита — заполнение Google form

[Submission form link](https://forms.gle/YDRtYV5Vq68SZgKW9)

- [ ] **Email** — твой
- [ ] **Team Name** — `Straple`
- [ ] **Name of Method** — короткое название (`LNS`, `DRP+ALNS`, и т.д.) (см. [todo.md](todo.md) для определения)
- [ ] **Short Description of Method** — 1-3 предложения (см. примеры в [todo.md](todo.md))
- [ ] **Link to Github repository** — формат:
  ```
  https://github.com/Straple/macro-place-challenge-2026/tree/<SHA>
  ```
  (зашитый SHA = stable снимок, даже если потом обновишь ветку)
- [ ] ☑ **Checkbox "Shared with Judges"** — поставить **только** если реально дал доступ выше
- [ ] **List of Team Members** — Full Name, Email, GitHub username
- [ ] **LinkedIn Profiles** — у каждого
- [ ] **Average proxy cost on IBM benchmarks** — взять из последнего прогона `--all`
- [ ] **Average runtime** — оттуда же
- [ ] **WNS на ariane133** (опционально, если делал ORFS прогон — не обязательно)
- [ ] **Area на ariane133** (опционально, если делал)
- [ ] ☑ **Assertion of Results** — что результаты accurate
- [ ] ☑ **Open-Source** — согласие на open-source при выигрыше

## После сабмита

- [ ] Сохранить screenshot заполненной формы (на случай вопросов)
- [ ] **Не трогать ветку `submission` до окончания судейства**, либо если делать апдейт — пересабмитить форму с новым SHA
- [ ] (Опционально) Watch [официального репо](https://github.com/partcleda/partcl-macro-place-challenge) на обновления leaderboard'а

---

## Критичные DQ-факторы (помни!)

| Что проверяю | Где упомянуто | Почему важно |
|---|---|---|
| `overlap_count == 0` на каждом benchmark | [README.md:91](../README.md#L91) | Жёсткое правило, **0 tolerance** |
| Runtime ≤ 1 час на benchmark | [README.md:72](../README.md#L72) | Hard timeout |
| Не модифицировал `macro_place/` evaluator | [README.md:87](../README.md#L87) | Запрещено модифицировать evaluation |
| Не хардкодил под benchmark-имена | [README.md:88](../README.md#L88) | DQ за hardcode |
| Только разрешённые ориентации (N/FN/FS/S) | [README.md:92](../README.md#L92) | R90/R270/FE/FW = DQ |
| Не resize'ил soft macros | [README.md:93](../README.md#L93) | Их размер фиксирован |
| Все зависимости в `pyproject.toml` | (BakaBobo case) | Иначе `import` упадёт у судей |
| Нет переменных среды / абс. путей | (общая практика) | Не воспроизведётся |

## Бонус — early submission

**Подай pre-final версию за 5-7 дней до дедлайна.** Так:
- Если форма обнаружит проблему — есть время фиксить
- Судьи могут начать предварительную верификацию раньше
- Финальный апдейт — за 1 день до дедлайна
- Безопаснее, чем сабмитить 21 мая в 23:55
