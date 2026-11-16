# Graphectory: признаки по графу действий агента
 
Статус: **черновик**. Скрипт строит граф по каждой попытке цепочки и считает структурные признаки. Часть метрик пока работает неверно (см. раздел «Ограничения»), поэтому использовать их для выводов рано.
 
Источник идеи: *Process-Centric Analysis of Agentic Software Systems* (arXiv:2512.02393). Граф из статьи реализован частично.
 
## 1. Что делает
 
Для каждой попытки цепочки:
 
1. Берёт вызовы инструментов из `v_agent_trace_actions`.
2. По флагу `--full` добавляет результаты инструментов и реплики из `v_agent_trace_full`.
3. Строит ориентированный мультиграф (`networkx.MultiDiGraph`).
4. Считает около 60 признаков по графу и по последовательности действий.
5. Пишет одну строку в CSV.
Объект классификации здесь — попытка (цепочка действий). Признаки описывают её структуру, а не отдельные события.
 
## 2. Запуск
 
Нужен `.env` с `ATA_DB_URL` и установленные пакеты: `pandas`, `numpy`, `networkx`, `sqlalchemy`, `python-dotenv`.
 
```bash
python graphectory_pipeline.py                       # полный запуск, chain_features.csv
python graphectory_pipeline.py --dry-run             # сколько попыток будет обработано
python graphectory_pipeline.py --limit 20            # первые 20 попыток, для проверки
python graphectory_pipeline.py --resume              # продолжить прерванный запуск
python graphectory_pipeline.py --no-full             # только вызовы инструментов, без результатов и реплик
python graphectory_pipeline.py --thoughts            # добавить узлы реплик ассистента (нужен --full)
python graphectory_pipeline.py --include-empty       # писать и попытки без действий, граф-признаки будут пустыми
python graphectory_pipeline.py --overwrite           # начать файл заново
```
 
Остальные флаги: `--out` (путь к CSV), `--offset`, `--chunk-chains` (размер пачки для запросов к БД), `--include-canary`.
 
Цепочки берутся из `v_chain_attempts` с фильтром `is_canary = false` и непустыми `model_identifier` и `harness`.
 
## 3. Граф
 
**Узлы**
 
- `action`: один вызов инструмента. Ключевые поля: имя инструмента, статус (`success`, `error`, `running`, `other`, `unknown`), объект, аргументы.
- `result`: результат инструмента (роль `tool`), связан с вызовом по `tool_call_id`. Берётся один результат на вызов с приоритетом `native` > `tool_trace` > `bridge`.
- `thought`: реплика ассистента (только с `--thoughts`).
**Рёбра**
 
| Тип | Значение |
|---|---|
| `next` | следующее действие в том же потоке и шаге |
| `parallel` | действия с одинаковым `event_order` в одном шаге |
| `cross_stream` | переход между шагами или потоками |
| `same_object` | последовательные действия над одним объектом (файл или команда) |
| `retry` | повтор того же инструмента с теми же аргументами после ошибки |
| `call_result` | вызов → его результат |
| `thought_leads_to` | реплика ассистента → первое следующее действие |
 
Объект действия (`extract_object`): путь к файлу (с обрезкой до `/workspace/`), первые два слова команды, URL, запрос или имя листа.
 
## 4. Признаки
 
**Размер и состав**
`n_nodes`, `n_edges`, `n_actions`, `n_results`, `n_thoughts`, `n_unique_tools`.
 
**Статусы**
`n_errors`, `n_success`, `n_running`, `n_other_status`, доли `frac_*`. Действия без результата: `n_actions_without_result` (только с `--full`).
 
**Повторы**
- `n_repeat_pairs`, `n_repeat_excess`: пары «инструмент + объект», встреченные больше одного раза.
- `longest_same_tool_run`: самая длинная серия одного инструмента подряд.
- `n_objects`, `n_objects_revisited`, `max_actions_per_object`, `frac_revisited_actions`: возвраты к одному объекту.
**Структура графа**
- `graph_depth`: длина самого длинного пути по рёбрам `next`, `cross_stream`, `thought_leads_to`.
- `branching_mean`, `branching_max`: исходящие степени.
- `n_parallel_blocks`, `n_sessions`, `n_step_switches`.
- `n_cycles`: см. «Ограничения».
- `n_execution_chains`, `longest_execution_chain`, `avg_execution_chain_length`: компоненты связности по рёбрам `same_object` и `retry`.
**Последовательность**
- `transition_entropy`: энтропия переходов между инструментами.
- `xepv_sequence`, `xepv_hash`, `xepv_len`: строка из букв групп.
- `frac_E_to_V`, `frac_V_to_E`, `frac_X_to_E` и другие: доли биграмм групп.
- `xepv_entropy_bigram`, `n_xepv_bigrams`.
**Группы действий (X, E, P, V)**
- X — чтение и поиск (read, grep, view, search).
- E — изменение и выполнение (write, edit, bash, exec, create).
- P — планирование (todo, plan, task, skill).
- V — проверка (test, verify, validate, lint).
- other — всё остальное.
Правила сопоставления в `tool_to_group`: сначала точные имена, потом префиксы.
 
**Мысли ассистента** (только с `--full`)
`n_assistant_messages`, `n_user_messages`, `n_planning_thoughts`, `n_reflexive_thoughts`, `n_other_thoughts`, `frac_planning_thoughts`. Классификация по ключевым словам.
 
**Метаданные и исход** (из `v_chain_attempts`)
`harness`, `model_identifier`, `status`, `expected_step_count`, `completed_step_count`, `chain_completed_fraction`, `end_to_end_seconds`, токены, `full_pass`, `task_score`.
 
## 5. Ограничения
 
Эти пункты стоит исправить до использования признаков в классификации.
 
1. **`n_cycles` всегда 0.** Все рёбра ведут вперёд по времени, поэтому циклов нет. В статье циклы строятся по обратным рёбрам структуры директорий, а в этом коде структурных рёбер нет. Признак либо переименовать (например, `n_repeat_pairs`), либо заменить.
2. **Нет фильтра потоков.** Запрос `v_agent_trace_actions` не отбирает `stream_kind`. Если один вызов есть в нескольких потоках, он попадёт в граф несколько раз и завысит `n_actions` и все частоты. Нужен фильтр по `native` через `v_trace_index`.
3. **Нет фильтра валидности.** `valid_for_scoring` не проверяется, попытки с ошибками телеметрии идут в признаки.
4. **Группы X/E/P/V по именам.** Правила простые: `docx_write_document` попадает в E по префиксу, а не по смыслу. Для новых инструментов проверить вручную.
5. **Мысли на английском.** Ключевые слова английские. Для русских ответов классификация почти всегда даёт `other`.
6. **Ошибочные статусы.** Список статусов ошибок взят из наблюдений. Если в данных есть `model_error`, `invalid_telemetry` или другие значения, их стоит проверить: `model_error` сейчас попадает в `other`.
7. **Нет фаз из статьи.** Localization / Patch / Validation заменены группами X/E/P/V. Алгоритм разметки фаз из статьи (учёт тестовых файлов и факта патча) не реализован, поэтому `frac_*` не эквивалентны фазам статьи.
8. **Нет Langutory.** Строка фаз со сжатием повторов не строится. Ближайший аналог — `xepv_sequence`, но без run-length.
9. **Сиротские колонки.** В `FIELDS` есть строки-заглушки (`# ... существующие поля ...`), их можно удалить.
## 6. Связь со статьёй
 
| В статье | Здесь |
|---|---|
| Узлы — действия агента | есть, плюс результаты и реплики |
| Временные рёбра | есть (`next`, `cross_stream`, `parallel`) |
| Структурные рёбра (директория → файл → блок) | нет, заменены на `same_object` |
| Фазы L/P/V по алгоритму 1 | нет, заменены на группы X/E/P/V |
| Langutory с run-length | нет, есть `xepv_sequence` без сжатия |
| Метрики: NC, TEC, LC, ALL, SEC, SB | частично: `n_nodes`, `n_edges`, `n_repeat_pairs`, `frac_revisited_actions`, `n_same_object_edges`, `branching_max` |
| Эвристики H1–H4 онлайн-мониторинга | нет |
 
## 7. Вспомогательные скрипты
 
`extract_traces.py` и `build_features_taxonomy` (таблица «попытка × инструмент») в этом пайплайне не используются. Их можно удалить, если не нужны для ручной проверки данных. Категории инструментов из `TaxonomyMapper` стоит перенести в `tool_to_group` или в таблицу семейств из `taxonomy.py`, чтобы не держать три разных словаря.
 