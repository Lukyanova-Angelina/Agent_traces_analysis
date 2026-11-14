# Agent Traces Analysis

Исследование поведения LLM-агентов на бенчмарке офисных задач (работа с Excel и Word). Траектории выполнения задач хранятся в PostgreSQL (БД `ata_logs`) и покрывают 5 харнессов (`opencode`, `hermes`, `openclaw`, `ouroboros`, `direct_mcp`) и несколько моделей (GigaChat, GLM, Kimi и др.).

Главная цель проекта — поведенческая фингерпринтика: по структуре цепочки действий предсказать пару `(harness, model_identifier)`, то есть определить, какая модель в каком харнессе работала.

Репозиторий содержит предобработку данных и четыре независимых пайплайна (таксономия действий, графовые признаки, CPA-аннотация, подготовка выборок). Пайплайны пока формируют признаки и разметку; обучение классификатора ещё не реализовано.

## Установка

- Python **3.13+** (`requires-python = ">=3.13"` в `pyproject.toml`)
- Менеджер пакетов: `uv` (в `Makefile` используется `uv sync`)

Зависимости из `pyproject.toml`:

- Основные: `pandas`, `numpy` (транзитивно), `networkx`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`, `requests`
- Объявлены, но в модулях пока не используются: `scikit-learn`, `transformers`, `torch`, `duckdb`, `jupyter`, `dotenv`
- Dev: `ruff`, `pytest`

Установка:

```bash
uv sync
```

### Переменные окружения

Файл `.env` (в `.gitignore`, в репозиторий не попадает) читается через `python-dotenv`:

| Переменная | Где используется | Назначение |
|---|---|---|
| `ATA_DB_URL` | `cpa.py`, `taxonomy.py`, `graphectory_pipeline.py` | Строка подключения SQLAlchemy к PostgreSQL |
| `CPA_API_KEY` | `cpa_opeai.py`, `cpa_consolidate.py` | Ключ OpenAI-совместимого API для CPA-разметки |
| `CPA_API_BASE` | `cpa_opeai.py`, `cpa_consolidate.py` | Базовый URL API |
| `CPA_MODEL` | `cpa_opeai.py`, `cpa_consolidate.py` | Имя модели (по умолчанию `gpt-4o-mini`) |

Пример `ATA_DB_URL`:

```
ATA_DB_URL=postgresql+psycopg2://<user>:<password>@127.0.0.1:5432/ata_logs_import
```

> Важно: в коде `taxonomy.py`, `cpa.py`, `graphectory_pipeline.py`, `extract_traces.py` и `src/agent_traces/agent_research_pipeline.py` встречаются захардкоженные строки подключения с паролем. Их следует убрать и оставить только `ATA_DB_URL` из окружения.

## Архитектура проекта

```
            PostgreSQL (ata_logs_import)
   v_chain_attempts  v_agent_trace_actions  v_agent_trace_full
          │                 │                     │
          │        ┌────────┴─────────┐           │
          │        ▼                  ▼           ▼
          │   taxonomy.py       cpa.py      graphectory_pipeline.py
          │   (Actonomy/CoALA)  (эпизоды)   (граф + ~45 метрик)
          │        │                  │           │
          │        ▼                  ▼           ▼
          │  actions_with_      cpa_episodes   chain_features_full.csv
          │  taxonomy.csv       .csv
          │                           │
          │      cpa_preparing_data.py (стратиф. выборка → cpa_sample.csv)
          │                           │
          │      cpa_opeai.py (LLM Open Induction → cpa_proposals.jsonl)
          │                           │
          │      cpa_analysis.py (статистика по proposals)
          │                           │
          │      cpa_consolidate.py (кодбук → codebook_v0.6.json,
          │                         cpa_labelled.csv)
          ▼
   extract_traces.py / src/agent_traces/agent_research_pipeline.py
   (ранние прототипы выгрузки и простой таксономии)
```

Четыре пайплайна:

1. **Таксономия действий** — `taxonomy.py`: категории Actonomy и уровни CoALA для каждого действия.
2. **Графовые признаки** — `graphectory_pipeline.py`: структурные метрики цепочки.
3. **CPA (Canonical Procedural Actions)** — `cpa.py` → `cpa_preparing_data.py` → `cpa_opeai.py` → `cpa_analysis.py` → `cpa_consolidate.py`.
4. **Служебные** — `diag.py` (проверка `chain_features_full.csv`), `extract_traces.py` (выгрузка для ML-прототипа).

## Модули

### `taxonomy.py` — таксономия действий
Присваивает каждому вызову инструмента категорию Actonomy (Retrieval, Grounding, Execution, Planning, Evaluating, Memory, Learning, Communication, Deciding, Other) и уровень CoALA (Internal / External / Memory).
- Ключевые функции: `refine_by_arguments`, `get_actonomy`, `get_coala`, `main`. Словарь `ACTION_TO_ACTONOMY` (~105 инструментов).
- Вход: `v_agent_trace_actions` (через `ATA_DB_URL`).
- Выход: `actions_with_taxonomy.csv`.

### `graphectory_pipeline.py` — графовые признаки траекторий
Строит `nx.MultiDiGraph` для каждой цепочки и извлекает структурные метрики: глубина, ветвление, энтропия переходов, объекты и повторы, XEPV-разметка (Explore/Execute/Plan/Verify), сессии, компоненты схемы сообщений.
- Ключевые функции: `build_graph`, `extract_metrics`, `xepv_sequence_features`, `classify_thought`, `empty_metrics`, `main`.
- Вход: `v_chain_attempts`, `v_agent_trace_actions`, `v_agent_trace_full` (через `ATA_DB_URL`).
- Выход: CSV (по умолчанию `chain_features.csv`, в репозитории — `chain_features_full.csv`). Поддерживает `--resume`, чанки и пропуск пустых цепочек.

### `cpa.py` — CPA, фаза 0 (эпизоды)
Группирует последовательные действия с одинаковыми `tool_name`, объектом и `step_index` в эпизоды.
- Ключевые функции: `extract_object`, `parse_args`, построение эпизодов в `main`.
- Вход: `v_agent_trace_actions` и `chain_features_full.csv` (для списка цепочек).
- Выход: `cpa_episodes.csv` (одна строка = один эпизод).

### `cpa_preparing_data.py` — CPA, стратифицированная выборка
Формирует выборку из эпизодов по пяти стратам: топ-30 пар `(tool_name, obj)`, средние и редкие пары, длинные эпизоды (>5 действий), эпизоды с ошибками.
- Ключевая функция: `stratify`.
- Вход: `cpa_episodes.csv`. Выход: `cpa_sample.csv` (200 эпизодов) и `cpa_sample_pilot.csv` (пилот).

### `cpa_opeai.py` — CPA, Open Induction через LLM
Отправляет эпизоды в OpenAI-совместимый API и получает название процедурной функции (4 поля: `proximal_goal`, `principal_output`, `state_change`, `preconditions`) либо отказ (`abstain`).
- Ключевые функции: `build_user_prompt`, `call_api`, `main`. Системный промпт — `SYSTEM_PROMPT`.
- Вход: `cpa_sample.csv`. Выход: `cpa_proposals.jsonl` (append-only, `--resume` пропускает уже обработанные).

### `cpa_analysis.py` — анализ предложений CPA
Печатает сводку по `cpa_proposals.jsonl`: статусы ok/abstain, распределение меток, разбивка по харнессам, причины отказов, группировка синонимов по первому слову.
- Вход: путь к JSONL как первый аргумент (по умолчанию `cpa_proposals.jsonl`). Выход: только текст в stdout.

### `cpa_consolidate.py` — консолидация кодбука CPA
Двухэтапная LLM-консолидация: кластеризация меток в кодбук, затем доразметка непокрытых меток.
- Ключевые функции: `stage1_consolidate`, `stage2_cover`, `main`. Промпты — `SYSTEM_PROMPT_1`, `SYSTEM_PROMPT_2`.
- Вход: `cpa_proposals.jsonl`. Выход: `codebook_v0.6.json` и `cpa_labelled.csv`.

### `extract_traces.py` — ранний прототип выгрузки
Тестовая выгрузка (ограничение 5 цепочек) событий и действий для ML. Объединяет действия шага с событиями по `step_attempt_id`.
- Ключевые функции: `to_text`, `to_dict`, `main`. Строка подключения захардкожена (не использует `ATA_DB_URL`).
- Выход: `traces_for_ml.csv`.

### `diag.py` — диагностика
Скрипт-проверка `chain_features_full.csv`: число строк (ожидается 11524), распределение `frac_revisited_actions` по харнессам.

### `src/agent_traces/agent_research_pipeline.py` — ранний прототип
Простой класс `TaxonomyMapper` (категории по именам инструментов, маппинг в CoALA) и `DataPipeline`. Читает `v_chain_attempts` и `NEW_v_action_categories`, сохраняет `agent_features_dataset.csv`. Строка подключения захардкожена. Заменён `taxonomy.py`.

### `src/agent_traces/__init__.py`
Заглушка: функция `main()` печатает приветствие. Указана как точка входа `agent-traces` в `pyproject.toml`, но реальной логики не содержит.

## Данные

Источник — PostgreSQL, БД `ata_logs_import`. Подробное описание столбцов: `data/raw/DATA_DICTIONARY.md`. Связи таблиц: `docs/table_relationships.txt` (статус «СЛОВАРЬ» / «ПРЕДПОЛОЖЕНИЕ» / «НЕ ПРОВЕРЕНО»; живая БД в этом документе не проверялась).

Представления, которые использует код:

- **`v_chain_attempts`** — одна строка = попытка цепочки (цепочка × прогон × среда × попытка). Ключевые поля: `chain_attempt_id`, `harness`, `model_identifier`, `model_family`, `status`, `valid_for_scoring`, `full_pass`, `task_score` (0..1), `completed_step_count`, `expected_step_count`, `end_to_end_seconds`, `input_tokens`, `output_tokens`, `is_canary`, `split`. NULL означает «неизвестно», а не 0.
- **`v_agent_trace_actions`** — одна строка = вызов инструмента. Ключевые поля: `chain_attempt_id`, `step_attempt_id`, `step_index`, `event_order`, `action`, `arguments` (JSON), `action_status`, `tool_call_id`, `trace_stream_id`, `agent_session_key`. Одна строка не означает успешного выполнения: проверяйте `action_status`.
- **`v_agent_trace_full`** — одна строка = событие потока (сообщения user/assistant/tool). Ключевые поля: `chain_attempt_id`, `step_attempt_id`, `trace_stream_id`, `event_order`, `role`, `content` (JSON), `tool_name`. Порядок внутри потока — по `event_order`.
- **`NEW_v_action_categories`** — собственное представление проекта (`new_categories_actions.sql`) с грубой категоризацией действий. Используется только `src/agent_traces/agent_research_pipeline.py`. Определение находится в `new_categories_actions.sql`; оно не является частью исходной БД.

Связи (1→N):

- `v_chain_attempts.chain_attempt_id` → `v_agent_trace_actions.chain_attempt_id` и `v_agent_trace_full.chain_attempt_id`
- `v_agent_trace_actions.step_attempt_id` = `v_agent_trace_full.step_attempt_id` (одна попытка шага → много событий и действий)
- `event_order` уникален только внутри `trace_stream_id`; не используйте его как ключ между потоками.

Ограничения данных:

- `v_agent_trace_full` содержит несколько потоков, которые могут описывать одно и то же действие (snapshots, native/tool_trace). Не суммируйте без дедупликации.
- `actual_model` сообщается на уровне попытки, а не события.

## Использование

Все скрипты читают `ATA_DB_URL` из `.env` (если аргумент не задан). Запускать из корня репозитория.

```bash
# 1. Таксономия действий → actions_with_taxonomy.csv
uv run python taxonomy.py

# 2. Графовые признаки → CSV (по умолчанию chain_features.csv)
uv run python graphectory_pipeline.py --out chain_features_full.csv --resume
#   полезные флаги: --limit N, --offset N, --overwrite, --no-full,
#   --thoughts (мысли агента, нужен --full), --include-empty, --include-canary,
#   --chunk-chains 200, --dry-run

# 3. CPA: эпизоды → cpa_episodes.csv
uv run python cpa.py --chains-csv chain_features_full.csv --out cpa_episodes.csv
#   --limit-chains N для пробного запуска

# 4. CPA: выборка → cpa_sample.csv и cpa_sample_pilot.csv
uv run python cpa_preparing_data.py --csv cpa_episodes.csv --out cpa_sample.csv

# 5. CPA: LLM-аннотация (требует CPA_API_KEY, CPA_API_BASE, CPA_MODEL)
uv run python cpa_opeai.py --csv cpa_sample.csv --out cpa_proposals.jsonl --resume
#   --limit N, --sleep 0.3

# 6. CPA: анализ предложений
uv run python cpa_analysis.py cpa_proposals.jsonl

# 7. CPA: консолидация кодбука
uv run python cpa_consolidate.py --proposals cpa_proposals.jsonl \
    --out-codebook codebook_v0.6.json --out-labelled cpa_labelled.csv

# Проверка результата
uv run python diag.py
```

Команды, указанные в `Makefile`: `make install`, `make lint` (`ruff check .`), `make format`, `make test` (`pytest`; тестов пока нет), `make notebook`.

> Скрипты запускаются из корня и используют относительные пути по умолчанию. Порядок шагов важен: каждый следующий берёт выход предыдущего.

## Формат выходных файлов

| Файл | Создаёт | Одна строка = | Ключевые колонки |
|---|---|---|---|
| `actions_with_taxonomy.csv` | `taxonomy.py` | одно действие инструмента | `chain_attempt_id`, `step_attempt_id`, `action`, `actonomy`, `coala` |
| `chain_features_full.csv` | `graphectory_pipeline.py` | одна попытка цепочки с графом | `chain_attempt_id`, `harness`, `model_identifier`, `status`, `n_nodes`, `n_edges`, `n_actions`, `frac_X/E/P/V`, `xepv_sequence`, `transition_entropy`, `n_cycles`, `expected_step_count`, `completed_step_count`, `task_score` и др. (~45 метрик) |
| `cpa_episodes.csv` | `cpa.py` | один эпизод: последовательные действия с одним `tool_name`, объектом и `step_index` | `chain_attempt_id`, `episode_id`, `harness`, `model_identifier`, `step_index`, `n_actions`, `tool_name`, `obj`, `statuses`, `unique_args`, `sample_args`, `tools_inside` |
| `cpa_sample.csv` | `cpa_preparing_data.py` | эпизод из выборки | то же + `stratum` (A–E) |
| `cpa_proposals.jsonl` | `cpa_opeai.py` | ответ LLM на один эпизод | `chain_attempt_id`, `episode_id`, `tool_name`, `obj`, `_status` (ok/abstain), `proposal` (`cpa_label`, 4 поля, `confidence` либо `reason`), `_prompt_hash` |
| `codebook_v0.6.json` | `cpa_consolidate.py` | — | `version`, `functions` (список процедурных функций) |
| `cpa_labelled.csv` | `cpa_consolidate.py` | эпизод с меткой | `chain_attempt_id`, `episode_id`, `original_label`, `canonical_label`, `confidence` |

Файлы `*.csv` в `.gitignore` и в репозиторий не попадают.


## Ограничения и известные нюансы

- **Мысли агента** (`thought`) по умолчанию не аннотируются: `graphectory_pipeline.py` добавляет их в граф только с флагом `--thoughts`, а классификация (`classify_thought`) — грубая, по ключевым словам.
- **`n_cycles` в `chain_features_full.csv` всегда 0** (все 7886 строк с графом). Расчёт есть в коде (`nx.simple_cycles` на проекции графа), но в этом выпуске он дал нули — причину не установлено; метрику не использовать без перепроверки.
- **Пустые цепочки**: 3638 цепочек без действий попадают в CSV только с `--include-empty`, все графовые метрики у них NULL.
- **Связь действий и событий не доказана**: `extract_traces.py` агрегирует действия по `step_attempt_id`, а не по событию; `event_order` уникален только внутри потока.
- **Захардкоженные пароли**: `extract_traces.py`, `cpa.py`, `taxonomy.py`, `graphectory_pipeline.py`, `src/agent_traces/agent_research_pipeline.py` содержат строку подключения с паролем (в `cpa.py`, `taxonomy.py`, `graphectory_pipeline.py` — как fallback). Перенести в `.env` и ротировать пароль.
- **CPA-разметка требует ручной проверки**: 134 ok / 66 abstain на 200 эпизодах; кодбук (`codebook_v0.5/v0.6`) получен LLM-консолидацией, качество обсуждается. Проверка синонимов и покрытия — вручную.
- **Модель LLM** по умолчанию `gpt-4o-mini`, фактически используемая модель задаётся `CPA_MODEL`; результаты зависят от неё и не воспроизводимы без кэша `cpa_proposals.jsonl`.
- **Классификатор не реализован**: зависимости `scikit-learn`, `torch`, `transformers` объявлены, но не используются.
- **`src/agent_traces/__init__.py`** — заглушка; `agent-traces` как точка входа не работает по смыслу.
- **Тестов нет**: каталог `tests/` содержит только `__init__.py`.
- **`cpa_analysis.py`** выполняется на импорте (без `main()`) и принимает путь первым аргументом.
- **`diag.py`** читает `chain_features_full.csv` из текущей директории и содержит ожидаемые значения в комментариях.
- **`extract_traces.py`** ограничен `TEST_LIMIT = 5` цепочками.
- **Актуальность словаря**: `v_chain_attempts` и др. в `DATA_DICTIONARY.md` описаны; `NEW_v_action_categories` — собственное представление, его нет в словаре.

## Ссылки

- **Graphectory / XEPV** — схема `xepv_*` и `frac_X/E/P/V` в `graphectory_pipeline.py`; ссылки на статью в коде нет.
- **arXiv:2608.06909** — execution chains и component schema (мысли/сообщения), `graphectory_pipeline.py` (комментарии v4).
- **arXiv:2609.24264** — CPA (Canonical Procedural Actions), трёхфазная аннотация; упоминается в описании проекта, в коде ссылки на статью нет.
- **CoALA** (Cognitive Architectures for Language Agents) — уровни Internal / External / Memory в `taxonomy.py`.
- **Actonomy** — категории действий (Retrieval, Grounding, …) в `taxonomy.py`.
- Библиотеки: [networkx](https://networkx.org/), [pandas](https://pandas.pydata.org/), [SQLAlchemy](https://www.sqlalchemy.org/), [python-dotenv](https://pypi.org/project/python-dotenv/).

## Статус

**Готово**
- Выгрузка и таксономия действий (`taxonomy.py`, 149 602 действия).
- Графовые признаки (`graphectory_pipeline.py`): 11 524 цепочки, из них 7 886 с графом.
- CPA фазы 0–2: эпизоды (81 082), выборка, LLM-разметка (200 эпизодов), анализ.

**В процессе**
- Консолидация кодбука CPA (`codebook_v0.5/v0.6`, ~17 функций, качество под вопросом).
- Чистка захардкоженных паролей и перевод всех скриптов на `ATA_DB_URL`.

**Запланировано**
- Использование признаков (графовых, таксономических, CPA) для классификации `(harness, model_identifier)`.
- Тесты в `tests/` и проверка `make lint` в CI.
- Исправление `n_cycles` и проверка `frac_revisited_actions`.


