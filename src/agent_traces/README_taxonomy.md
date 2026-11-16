# taxonomy.py

Гибридная таксономия для трейсов LLM-агентов: размечает **действия** и **мысли** по сохранённым попыткам цепочек в базе ATA.

## Что размечает

### 1. Действия (`v_agent_trace_actions`)

Каждый вызов инструмента (`action` + `arguments`) получает три уровня разметки:

| Колонка | Что это | Источник |
|---|---|---|
| `actonomy_legacy` | 10 категорий Act-ONOMY v1 (Retrieval / Grounding / Executing / ...) | `ACTION_TO_ACTONOMY` + refine по аргументам |
| `coala_legacy` | legacy CoALA (Internal / External / Memory) | `ACTONOMY_TO_COALA` |
| `actonomy` | Act-ONOMY-совместимая (Executing / Reflecting / ...) | `ACTION_TO_SUBACTION` + refine по аргументам |
| `subaction` | sub-action Act-ONOMY (`from_local_corpus`, `executing_plan`, `with_users`, ...) | то же |
| `coala` | CoALA v2: Memory → Internal, Retrieval → External, точечные override | `ACTONOMY_TO_coala` + `SUBACTION_TO_coala` |

«Обманчивые» инструменты уточняются по аргументам:
- `bash` / `exec` / `terminal`: `pip install` → Grounding, `cat`/`grep` → Retrieval, `pytest`/`--debug` → `executing_debug`, иначе `executing_plan`.
- `browser`: всегда Grounding (навигация = взаимодействие).
- `read` с полем `command`: трактуется как shell → Executing.
- `office_officecli`: `query`/`get` → Retrieval, иначе Grounding.

Результат: **`actions_with_taxonomy.csv`** + распределения + проверки консистентности (Communication должна уйти в Grounding, CoALA для Memory/from_open_web/from_relevant_context/from_skill_library/terminating). С `--compare` дополнительно печатается diff `coala_legacy → coala`.

### 2. Мысли (`v_agent_trace_full`, `role='assistant'`)

Из jsonb-поля `content` извлекается текст, затем rule-based по regex-паттернам относится к одному из:

- **Reflecting** — «previous attempt failed», «realized», «mistake»
- **Planning** — «first / next / then», «I'll», «plan:»
- **Reasoning** — «because», «therefore», «this means»
- **Other / Empty** — ничего не сработало / пусто

При равенстве score приоритет: Reflecting > Planning > Reasoning. Сохраняются `thought_type`, `matched` (сработавшие паттерны), `confidence` (доля совпадений лучшего типа), `first_sentence`.

Дедупликация по `(chain_attempt_id, step_attempt_id, event_order)` с приоритетом `stream_kind`: native > tool_trace > bridge (разные потоки могут дублировать одну историю).

Результат: **`thoughts_with_taxonomy.csv`** + распределение типов по harness и средняя confidence.

## Запуск

```bash
export ATA_DB_URL="postgresql://user:pass@host/db"   # или в .env

python taxonomy.py --mode actions             # только действия
python taxonomy.py --mode thoughts            # только мысли
python taxonomy.py --mode both                # оба (по умолчанию)
python taxonomy.py --mode actions --compare   # + diff coala_legacy → coala






## Таксономия: потенциальные риски и ошибки

### Критичные (влияют на результаты)
1. **Дубли потоков в действиях.** `run_actions_pipeline` не фильтрует `stream_kind`. Один вызов может попасть в native, tool_trace и bridge, и счётчики окажутся завышены. Нужен фильтр `stream_kind = 'native'`.
2. **Невалидные попытки не отсеиваются.** Фильтр по `chain_attempts.valid_for_scoring IS TRUE` отсутствует. `NULL` тоже не равен `true`.
3. **Дедупликация мыслей может терять события.** Ключ `(chain_attempt_id, step_attempt_id, event_order, role)` не содержит `trace_stream_id`, а `event_order` задан внутри потока. Разные события из разных потоков с одинаковым `event_order` схлопнутся.
4. **Regex для мыслей рассчитаны на английский.** Если ответы моделей на русском, большинство уйдёт в `Other`. Это ложный результат, а не признак отсутствия рефлексии.
5. **Ключи аргументов различаются по моделям.** Код читает только `command`. Для других моделей команда может лежать в `cmd`, `script` или в строке JSON. Такие действия молча попадут в `Executing` или `Other`.

### Методологические
6. **Подстрочные совпадения в правилах по командам.** `"cat "` срабатывает на `concat `, `"ls "` на `tools `, `"find "` на `refind `. Нужны границы слов.
7. **Расхождение legacy и новой разметки.** `browser` в legacy с `goto` даёт `Retrieval`, в новой всегда `Grounding`. `message` в legacy даёт `Communication`, в новой `Grounding/with_users`. Решения не записаны и не обоснованы.
8. **Классы не проверены на эталоне.** Маппинг инструментов в Act-ONOMY и CoALA выбран исследователем. Нет ни второго разметчика, ни проверки согласия.
9. **Неизвестные инструменты уходят в `Other`.** Полного списка покрытия по данным не снято, поэтому охват не измерен.
10. **Эвристика уверенности мыслей условна.** `confidence` считается как доля совпадений, а `Other` получает фиксированные 0.3. Это не вероятность и не сравнимо с LLM.
11. **`is_canary` заявлен в описании, но не фильтруется.** Канареечные задания попадают в выборку.

### Технические
12. **Два почти одинаковых словаря CoALA.** `ACTONOMY_TO_COALA` и `ACTONOMY_TO_coala` различаются регистром и содержанием. Легко использовать не тот.
13. **Пустой вход.** При пустом `v_agent_trace_actions` распаковка `t[0]` и `t[1]` упадёт. Нужен ранний выход.
14. **Порядок событий.** `ORDER BY event_order` без `trace_stream_id` не даёт строгого общего порядка, и это уже отмечено в комментарии кода.

### Не проверено
- Колонки `v_agent_trace_actions` (`event_order`, `action_block_index`, `action_status`, `arguments`) не сверены со словарём данных.
- Язык текстов мыслей по моделям не измерен.
- Покрытие инструментов по всем харнесам и моделям не снято.