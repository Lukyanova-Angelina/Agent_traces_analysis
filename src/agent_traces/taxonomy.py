#!/usr/bin/env python3
"""
taxonomy.py — гибридная таксономия для трейсов LLM-агентов.

Режимы: actions (legacy Act-ONOMY + Act-ONOMY sub-action + CoALA),
thoughts (rule-based из v_agent_trace_full), both.
Источники: Act-ONOMY (arXiv, SophiaGao/Act-ONOMY); CoALA (Sumers et al. 2023).
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()
DB_URL = os.environ.get("ATA_DB_URL")


# Плоский маппинг "имя инструмента → одна из 10 категорий Act-ONOMY".
# Используется legacy-разметкой get_actonomy; для новой разметки — ACTION_TO_SUBACTION.
ACTION_TO_ACTONOMY = {
    # Файловая система: чтение/листинг/поиск → Retrieval, запись/правка → Grounding
    "read": "Retrieval", "read_file": "Retrieval", "filesystem_read_file": "Retrieval",
    "list_files": "Retrieval", "dir_list": "Retrieval", "filesystem_list_directory": "Retrieval",
    "filesystem_list_directory_with_sizes": "Retrieval", "filesystem_directory_tree": "Retrieval",
    "filesystem_get_file_info": "Retrieval", "filesystem_list_allowed_directories": "Retrieval",
    "glob": "Retrieval", "grep": "Retrieval", "search_files": "Retrieval",
    "filesystem_search_files": "Retrieval", "file_fetch": "Retrieval",
    "write": "Grounding", "write_file": "Grounding", "filesystem_write_file": "Grounding",
    "edit": "Grounding", "edit_batch": "Grounding", "edit_text": "Grounding",
    "patch": "Grounding", "apply_patch": "Grounding", "filesystem_edit_file": "Grounding",
    "filesystem_move_file": "Grounding", "filesystem_create_directory": "Grounding",

    # Код и ОС: исполнение команд → Executing, поиск по коду → Retrieval
    "exec": "Executing", "bash": "Executing", "terminal": "Executing",
    "execute_code": "Executing", "run_command": "Executing", "run_script": "Executing",
    "process": "Executing",
    "query_code": "Retrieval", "search_code": "Retrieval",

    # Excel: чтение → Retrieval, запись/форматирование → Grounding
    "excel_read_data_from_excel": "Retrieval", "excel_get_workbook_metadata": "Retrieval",
    "excel_get_merged_cells": "Retrieval",
    "excel_validate_formula_syntax": "Evaluating",
    "excel_write_data_to_excel": "Grounding", "excel_create_workbook": "Grounding",
    "excel_create_worksheet": "Grounding", "excel_format_range": "Grounding",
    "excel_apply_formula": "Grounding", "excel_copy_range": "Grounding",
    "excel_copy_worksheet": "Grounding", "excel_create_pivot_table": "Grounding",
    "excel_delete_sheet_columns": "Grounding", "excel_delete_sheet_rows": "Grounding",
    "excel_delete_worksheet": "Grounding", "excel_insert_rows": "Grounding",
    "excel_rename_worksheet": "Grounding",

    # DOCX: чтение → Retrieval, запись → Grounding
    "docx_inspect_document": "Retrieval", "docx_read_table": "Retrieval", "docx_read_text": "Retrieval",
    "docx_write_document": "Grounding",
    "office_officecli": "Grounding",

    # Веб и зрение: загрузка/поиск/анализ → Retrieval, браузерные действия → Grounding
    "web_fetch": "Retrieval", "web_search": "Retrieval", "browse_page": "Retrieval",
    "vision_analyze": "Retrieval", "view_image": "Retrieval", "image": "Retrieval", "ocr_pdf": "Retrieval",
    "browser": "Grounding", "browser_click": "Grounding", "browser_back": "Grounding",
    "video_generate": "Grounding", "canvas": "Grounding",

    # Мета-инструменты: планирование задач → Planning
    "task": "Planning", "todo": "Planning", "todowrite": "Planning",
    "update_plan": "Planning", "update_goal": "Planning", "create_goal": "Planning",
    "delegate_task": "Planning", "subagents": "Planning", "schedule_subagent": "Planning",
    "sessions_spawn": "Planning", "sessions_yield": "Planning", "cron": "Planning",
    "wait_task": "Planning", "skill": "Planning", "skill_view": "Planning", "skills_list": "Planning",

    # Мета-инструменты: чтение контекста/результатов → Retrieval
    "peek_task": "Retrieval", "get_task_result": "Retrieval",
    "session_search": "Retrieval", "session_status": "Retrieval", "sessions_history": "Retrieval",

    # Мета-инструменты: оценка/решение/обучение
    "verify_and_record": "Evaluating", "task_acceptance_review": "Evaluating",
    "cancel_task": "Deciding", "discard_child_result": "Deciding",
    "skill_manage": "Learning", "skill_workshop": "Learning",

    # Коммуникация
    "message": "Communication", "send_file": "Communication", "sessions_send": "Communication",

    # Память
    "memory_get": "Memory", "memory_search": "Memory", "tree_read": "Memory", "tree_note": "Memory",

    # VCS и прочее
    "vcs_diff": "Retrieval", "vcs_status": "Retrieval",
    "gateway": "Grounding",
    "browser_navigate": "Retrieval", "browser_snapshot": "Retrieval", "browser_console": "Executing",
    "question": "Communication",
    "NULL": "Unknown", None: "Unknown"
}

# Legacy-маппинг Act-ONOMY → CoALA (Internal / External / Memory) для get_coala_legacy.
# Здесь Communication — отдельная категория, Memory — отдельный ярлык; результат
ACTONOMY_TO_COALA = {
    "Planning": "Internal", "Reasoning": "Internal", "Deciding": "Internal",
    "Evaluating": "Internal", "Reflecting": "Internal", "Learning": "Internal",
    "Communication": "External", "Retrieval": "External",
    "Grounding": "External", "Executing": "External",
    "Memory": "Memory", "Unknown": "Unknown", "Other": "Other"
}


# Детальный маппинг "имя инструмента → (actonomy, sub-action)" по Act-ONOMY.
# Используется новой разметкой get_actonomy_and_subaction.
# Ключ — значение поля action в v_agent_trace_actions (имя инструмента,
# не step_versions.action); arguments могут быть объектом/массивом/строкой.
ACTION_TO_SUBACTION = {
    # Файловая система
    "read":                     ("Retrieval", "from_local_corpus"),
    "read_file":                ("Retrieval", "from_local_corpus"),
    "filesystem_read_file":     ("Retrieval", "from_local_corpus"),
    "list_files":               ("Retrieval", "from_local_corpus"),
    "dir_list":                 ("Retrieval", "from_local_corpus"),
    "filesystem_list_directory": ("Retrieval", "from_local_corpus"),
    "filesystem_list_directory_with_sizes": ("Retrieval", "from_local_corpus"),
    "filesystem_directory_tree": ("Retrieval", "from_local_corpus"),
    "filesystem_get_file_info": ("Retrieval", "from_local_corpus"),
    "filesystem_list_allowed_directories": ("Retrieval", "from_local_corpus"),
    "glob":                     ("Retrieval", "from_local_corpus"),
    "grep":                     ("Retrieval", "from_local_corpus"),
    "search_files":             ("Retrieval", "from_local_corpus"),
    "filesystem_search_files":  ("Retrieval", "from_local_corpus"),
    "file_fetch":               ("Retrieval", "from_local_corpus"),
    "write":                    ("Grounding", "with_digital_environments"),
    "write_file":               ("Grounding", "with_digital_environments"),
    "filesystem_write_file":    ("Grounding", "with_digital_environments"),
    "edit":                     ("Grounding", "with_digital_environments"),
    "edit_batch":               ("Grounding", "with_digital_environments"),
    "edit_text":                ("Grounding", "with_digital_environments"),
    "patch":                    ("Grounding", "with_digital_environments"),
    "apply_patch":              ("Grounding", "with_digital_environments"),
    "filesystem_edit_file":     ("Grounding", "with_digital_environments"),
    "filesystem_move_file":     ("Grounding", "with_digital_environments"),
    "filesystem_create_directory": ("Grounding", "with_digital_environments"),

    # Код и ОС
    "exec":          ("Executing", "executing_plan"),
    "bash":          ("Executing", "executing_plan"),
    "terminal":      ("Executing", "executing_plan"),
    "execute_code":  ("Executing", "executing_plan"),
    "run_command":   ("Executing", "executing_plan"),
    "run_script":    ("Executing", "executing_plan"),
    "process":       ("Executing", "executing_plan"),
    "query_code":    ("Retrieval", "from_local_corpus"),
    "search_code":   ("Retrieval", "from_local_corpus"),
    "browser_console": ("Executing", "executing_plan"),

    # Excel
    "excel_read_data_from_excel":   ("Retrieval", "from_local_corpus"),
    "excel_get_workbook_metadata":  ("Retrieval", "from_local_corpus"),
    "excel_get_merged_cells":       ("Retrieval", "from_local_corpus"),
    "excel_validate_formula_syntax": ("Evaluating", "against_goals"),
    "excel_write_data_to_excel":    ("Grounding", "with_digital_environments"),
    "excel_create_workbook":        ("Grounding", "with_digital_environments"),
    "excel_create_worksheet":       ("Grounding", "with_digital_environments"),
    "excel_format_range":           ("Grounding", "with_digital_environments"),
    "excel_apply_formula":          ("Grounding", "with_digital_environments"),
    "excel_copy_range":             ("Grounding", "with_digital_environments"),
    "excel_copy_worksheet":         ("Grounding", "with_digital_environments"),
    "excel_create_pivot_table":     ("Grounding", "with_digital_environments"),
    "excel_delete_sheet_columns":   ("Grounding", "with_digital_environments"),
    "excel_delete_sheet_rows":      ("Grounding", "with_digital_environments"),
    "excel_delete_worksheet":       ("Grounding", "with_digital_environments"),
    "excel_insert_rows":            ("Grounding", "with_digital_environments"),
    "excel_rename_worksheet":       ("Grounding", "with_digital_environments"),

    # DOCX
    "docx_inspect_document": ("Retrieval", "from_local_corpus"),
    "docx_read_table":       ("Retrieval", "from_local_corpus"),
    "docx_read_text":        ("Retrieval", "from_local_corpus"),
    "docx_write_document":   ("Grounding", "with_digital_environments"),
    "office_officecli":      ("Grounding", "with_digital_environments"),

    # Веб и зрение
    "web_fetch":        ("Retrieval", "from_open_web"),
    "web_search":       ("Retrieval", "from_open_web"),
    "browse_page":      ("Retrieval", "from_open_web"),
    "browser_snapshot": ("Retrieval", "from_open_web"),
    "vision_analyze":   ("Grounding", "with_digital_environments"),
    "view_image":       ("Retrieval", "from_local_corpus"),
    "image":            ("Retrieval", "from_local_corpus"),
    "ocr_pdf":          ("Retrieval", "from_local_corpus"),
    "browser":          ("Grounding", "with_digital_environments"),
    "browser_click":    ("Grounding", "with_digital_environments"),
    "browser_back":     ("Grounding", "with_digital_environments"),
    "browser_navigate": ("Grounding", "with_digital_environments"),
    "video_generate":   ("Grounding", "with_digital_environments"),
    "canvas":           ("Grounding", "with_digital_environments"),

    # Планирование
    "task":              ("Planning", "decomposing_tasks"),
    "update_plan":       ("Planning", "modifying_plans"),
    "update_goal":       ("Planning", "formulating_workflows"),
    "create_goal":       ("Planning", "formulating_workflows"),
    "delegate_task":     ("Planning", "decomposing_tasks"),
    "subagents":         ("Planning", "decomposing_tasks"),
    "schedule_subagent": ("Planning", "decomposing_tasks"),
    "sessions_spawn":    ("Planning", "decomposing_tasks"),
    "sessions_yield":    ("Executing", "terminating"),
    "cron":              ("Planning", "formulating_workflows"),
    "wait_task":         ("Retrieval", "from_relevant_context"),

    # Библиотека навыков
    "skill":       ("Retrieval", "from_skill_library"),
    "skill_view":  ("Retrieval", "from_skill_library"),
    "skills_list": ("Retrieval", "from_skill_library"),

    # Контекст задачи
    "peek_task":       ("Retrieval", "from_relevant_context"),
    "get_task_result": ("Retrieval", "from_relevant_context"),

    # Память
    "session_search":   ("Memory", "reading_memory"),
    "session_status":   ("Memory", "reading_memory"),
    "sessions_history": ("Memory", "reading_memory"),
    "memory_get":       ("Memory", "reading_memory"),
    "memory_search":    ("Memory", "reading_memory"),
    "tree_read":        ("Memory", "reading_memory"),
    "tree_note":        ("Memory", "storing_information"),
    "todo":             ("Memory", "storing_information"),
    "todowrite":        ("Memory", "storing_information"),

    # Оценка / решение / обучение
    "verify_and_record":     ("Evaluating", "against_goals"),
    "task_acceptance_review": ("Evaluating", "against_gold_references"),
    "cancel_task":           ("Deciding", "terminating"),
    "discard_child_result":  ("Deciding", "accepting_rejecting"),
    "skill_manage":          ("Learning", "updating_reasoning"),
    "skill_workshop":        ("Learning", "updating_reasoning"),

    # Коммуникация: в Act-ONOMY относится к Grounding (with_users / with_other_agents)
    "message":       ("Grounding", "with_users"),
    "question":      ("Grounding", "with_users"),
    "send_file":     ("Grounding", "with_users"),
    "sessions_send": ("Grounding", "with_other_agents"),

    # VCS
    "vcs_diff":    ("Retrieval", "from_local_corpus"),
    "vcs_status":  ("Retrieval", "from_local_corpus"),
    "gateway":     ("Grounding", "with_digital_environments"),
}


# Адаптация CoALA: Memory — компонент (чтение/запись → Internal), Executing у нас —
# External (инструменты), Retrieval по умолчанию External (файл/веб = world state).
# Исключения для Retrieval/Executing — в SUBACTION_TO_coala.
ACTONOMY_TO_coala = {
    # Internal — операции над собственным состоянием/контекстом
    "Reasoning":  "Internal",
    "Planning":   "Internal",
    "Deciding":   "Internal",
    "Evaluating": "Internal",
    "Reflecting": "Internal",
    "Learning":   "Internal",
    "Memory":     "Internal",

    # External — взаимодействие с миром, пользователями, инструментами
    "Grounding":     "External",
    "Executing":     "External",
    "Retrieval":     "External",

    # Fallback для старых ярлыков
    "Communication": "External",

    "Unknown": "Unknown",
    "Other":   "Other",
}


# Точечные override поверх ACTONOMY_TO_coala: Retrieval из внутреннего контекста
# агента и meta-операция terminating → Internal. Остальные Retrieval (файл/веб)
# остаются External.
SUBACTION_TO_coala = {
    ("Retrieval", "from_relevant_context"): "Internal",  # peek_task / get_task_result
    ("Retrieval", "from_skill_library"):    "Internal",  # skill / skill_view / skills_list
    ("Executing", "terminating"):           "Internal",  # sessions_yield
}


def refine_by_arguments(action, arguments):
    """
    Уточняет actonomy (одна строка) для «обманчивых» инструментов по arguments.
    Возвращает новую actonomy или None. Только для legacy-разметки, не менять.
    """
    if action not in {"bash", "exec", "terminal", "run_command", "run_script", "process",
                      "office_officecli", "browser", "read"}:
        return None

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        return None

    if action in {"bash", "exec", "terminal", "run_command", "run_script", "process"}:
        cmd = str(arguments.get("command", "")).lower()
        # Установка пакетов меняет среду → Grounding
        if any(x in cmd for x in ["pip install", "apt-get", "apt install", "npm install"]):
            return "Grounding"
        # Чтение файлов через shell → Retrieval
        if any(x in cmd for x in ["ls ", "cat ", "head ", "tail ", "grep ", "find "]):
            return "Retrieval"
        return "Executing"

    if action == "office_officecli":
        cmd = str(arguments.get("command", "")).lower()
        if any(x in cmd for x in ["query", "get", "view", "read"]):
            return "Retrieval"
        return "Grounding"

    if action == "browser":
        act = str(arguments.get("action", "")).lower()
        if act in {"goto", "open", "navigate", "back"}:
            return "Retrieval"
        return "Grounding"

    if action == "read":
        # "command" в аргументах = shell-вызов, а не чтение файла
        if "command" in arguments:
            return "Executing"
        return "Retrieval"

    return None


def refine_v2_by_arguments(action, arguments):
    """
    Уточняет (actonomy, subaction) для «обманчивых» инструментов по arguments.
    Возвращает кортеж или None. Отличие от legacy: различает executing_plan /
    executing_debug и browser всегда Grounding (навигация = взаимодействие).
    """
    if action not in {"bash", "exec", "terminal", "run_command", "run_script",
                      "process", "office_officecli", "browser", "browser_console", "read"}:
        return None

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        return None

    if action in {"bash", "exec", "terminal", "run_command", "run_script", "process"}:
        cmd = str(arguments.get("command", "")).lower()
        if any(x in cmd for x in ["pip install", "apt-get", "apt install", "npm install"]):
            return ("Grounding", "with_digital_environments")
        if any(x in cmd for x in ["ls ", "cat ", "head ", "tail ", "grep ", "find "]):
            return ("Retrieval", "from_local_corpus")
        # Признаки отладки/тестов — отдельный sub-action
        if any(x in cmd for x in ["pytest", "npm test", "make test",
                                   "--debug", " -v ", "traceback"]):
            return ("Executing", "executing_debug")
        return ("Executing", "executing_plan")

    if action == "office_officecli":
        cmd = str(arguments.get("command", "")).lower()
        if any(x in cmd for x in ["query", "get", "view", "read"]):
            return ("Retrieval", "from_local_corpus")
        return ("Grounding", "with_digital_environments")

    if action == "browser":
        return ("Grounding", "with_digital_environments")

    # Консоль браузера — исполнение JS, не чтение
    if action == "browser_console":
        return ("Executing", "executing_plan")

    if action == "read":
        if "command" in arguments:
            return ("Executing", "executing_plan")
        return ("Retrieval", "from_local_corpus")

    return None


def get_actonomy(action, arguments):
    """
    Legacy-разметка: 10 категорий Act-ONOMY (одна строка).
    Порядок: refine_by_arguments → ACTION_TO_ACTONOMY → "Other".
    """
    if action is None or pd.isna(action):
        return "Unknown"
    refined = refine_by_arguments(action, arguments)
    if refined:
        return refined
    return ACTION_TO_ACTONOMY.get(action, "Other")


def get_coala_legacy(actonomy):
    """Legacy CoALA (Internal / External / Memory) поверх старой actonomy — для сравнения."""
    return ACTONOMY_TO_COALA.get(actonomy, "Other")


def get_actonomy_and_subaction(action, arguments):
    """
    Новая разметка: (actonomy, sub-action) по Act-ONOMY.
    Порядок: refine_v2_by_arguments → ACTION_TO_SUBACTION → fallback через legacy
    с переименованием Execution→Executing, Reflection→Reflecting, subaction=None.
    """
    if action is None or pd.isna(action):
        return ("Unknown", None)

    refined = refine_v2_by_arguments(action, arguments)
    if refined:
        return refined

    if action in ACTION_TO_SUBACTION:
        return ACTION_TO_SUBACTION[action]

    fallback_actonomy = ACTION_TO_ACTONOMY.get(action, "Other")
    rename = {"Execution": "Executing", "Reflection": "Reflecting"}
    fallback_actonomy = rename.get(fallback_actonomy, fallback_actonomy)
    return (fallback_actonomy, None)


def get_coala(actonomy, subaction):
    """
    CoALA-интерпретация (Internal / External / Other) для новой схемы.
    Override через SUBACTION_TO_coala (внутренний контекст и meta), иначе
    плоский ACTONOMY_TO_coala. Memory → Internal; Retrieval → External,
    кроме from_relevant_context / from_skill_library.
    """
    key = (actonomy, subaction)
    if key in SUBACTION_TO_coala:
        return SUBACTION_TO_coala[key]
    return ACTONOMY_TO_coala.get(actonomy, "Other")


# Regex-паттерны для rule-based классификации мыслей.
# При равенстве score приоритет: Reflecting > Planning > Reasoning.
THOUGHT_PATTERNS = {
    "Reflecting": [
        r"\b(previous|last|earlier)\s+(attempt|try|step|action)\b",
        r"\b(fail(ed|ure)|went wrong|didn'?t work|not work(ed)?)\b",
        r"\b(mistake|error|issue|problem|wrong|incorrect|bug)\b",
        r"\b(let me|i (should|need to)) (re-?try|try again|fix|correct)\b",
        r"\b(realize[ds]?|realiz(e|ing)|i was wrong)\b",
        r"\b(learned|insight|takeaway)\b",
    ],
    "Planning": [
        r"\b(first|next|then|finally|after that|step \d)\b",
        r"\b(i'?ll|i will|i'?m going to|let me|we need to)\b",
        r"\b(plan|strategy|approach|step(s)?)\b:",
        r"^\s*(1\.|2\.|-|\*)\s",
        r"\b(before|after) (doing|we|i)\b",
        r"\b(goal|objective|target)\b",
    ],
    "Reasoning": [
        r"\b(because|therefore|thus|hence|so)\b",
        r"\b(if|when) .{0,60}(then|we can|we should|it means)\b",
        r"\b(this|that|it) (means|implies|suggests|indicates)\b",
        r"\b(reason|logic|infer|conclude|deduce)\b",
        r"\b(given that|assuming|considering|since)\b",
    ],
    "Other": [],
}


def classify_thought(text):
    """
    Rule-based классификация мысли: Reasoning / Reflecting / Planning / Other.
    Возвращает dict с thought_type, matched (сработавшие паттерны),
    confidence (доля совпадений лучшего типа) и first_sentence (~200 символов).
    Ограничения: не учитывает отрицания и контекст — baseline для сравнения с LLM.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return {
            "thought_type": "Empty",
            "matched": [],
            "confidence": 0.0,
            "first_sentence": "",
        }

    text_lower = text.lower()
    scores = {k: 0 for k in THOUGHT_PATTERNS}
    matched = []
    for ttype, patterns in THOUGHT_PATTERNS.items():
        if ttype == "Other":
            continue
        for p in patterns:
            if re.search(p, text_lower):
                scores[ttype] += 1
                matched.append((ttype, p))

    priority = {"Reflecting": 3, "Planning": 2, "Reasoning": 1, "Other": 0}

    if all(v == 0 for v in scores.values()):
        thought_type = "Other"
        confidence = 0.3
    else:
        best = max(scores.items(), key=lambda kv: (kv[1], priority[kv[0]]))
        thought_type = best[0]
        total_hits = sum(scores.values())
        confidence = min(1.0, scores[thought_type] / max(total_hits, 1))

    first_sentence = re.split(r"[.!?]\s", text.strip(), maxsplit=1)[0][:200]

    return {
        "thought_type": thought_type,
        "matched": matched,
        "confidence": round(confidence, 2),
        "first_sentence": first_sentence,
    }


def _extract_text(content):
    """
    Извлекает текст из jsonb-поля content (v_agent_trace_full.content).
    content — читаемая проекция события: str, dict {"text": ...} / {"parts": [...]}
    / {"content": [...]}, либо список блоков ({"type": "text", ...}, tool_use и т.п.).
    При нераспознанной структуре — обрезанный json.dumps.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("parts"), list):
            return _extract_text(content["parts"])
        if isinstance(content.get("content"), (str, list, dict)):
            return _extract_text(content["content"])
        return json.dumps(content, ensure_ascii=False)[:2000]
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" and "text" in b:
                    parts.append(b["text"])
                elif "text" in b:
                    parts.append(str(b["text"]))
                else:
                    parts.append(json.dumps(b, ensure_ascii=False)[:300])
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(content)


def run_actions_pipeline(compare=False):
    """
    Читает v_agent_trace_actions, размечает действия и сохраняет
    actions_with_taxonomy.csv. Legacy-колонки (actonomy_legacy, coala_legacy)
    идут рядом с новыми (actonomy, subaction, coala) для сравнения.
    compare=True — печатает diff coala_legacy → coala.
    """
    engine = create_engine(DB_URL)

    print("Загрузка v_agent_trace_actions...")
    # event_order — внутренний номер события в trace_stream_id (не глобальный порядок);
    # ORDER BY по нему без trace_stream_id не даёт строгого общего порядка.
    sql = """
        SELECT chain_attempt_id, step_attempt_id, step_index,
               event_order, action, arguments, action_status,
               harness, model_identifier
        FROM v_agent_trace_actions
        ORDER BY chain_attempt_id, step_attempt_id, event_order
    """
    df = pd.read_sql(text(sql), engine)
    print(f"Загружено строк: {len(df)}")

    # Legacy-разметка (для сравнения)
    print("Разметка actonomy_legacy + coala_legacy...")
    df["actonomy_legacy"] = df.apply(
        lambda r: get_actonomy(r["action"], r["arguments"]), axis=1
    )
    df["coala_legacy"] = df["actonomy_legacy"].apply(get_coala_legacy)

    # Новая разметка
    print("Разметка actonomy + subaction...")
    v2 = df.apply(
        lambda r: get_actonomy_and_subaction(r["action"], r["arguments"]),
        axis=1,
    )
    df["actonomy"]  = [t[0] for t in v2]
    df["subaction"] = [t[1] for t in v2]

    print("Разметка coala (адаптированная)...")
    df["coala"] = df.apply(
        lambda r: get_coala(r["actonomy"], r["subaction"]), axis=1
    )

    out = "actions_with_taxonomy.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n✅ Сохранено в {out}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Строк:   {len(df)}")

    # Отчёты по распределениям
    print("\n=== actonomy_legacy (старая, 10 категорий) ===")
    print(df["actonomy_legacy"].value_counts())

    print("\n=== actonomy (новая, Act-ONOMY-совместимая) ===")
    print(df["actonomy"].value_counts())

    print("\n=== subaction (топ-20) ===")
    print(df["subaction"].value_counts().head(20))

    print("\n=== coala_legacy (старая) ===")
    print(df["coala_legacy"].value_counts())

    print("\n=== coala (новая, адаптированная) ===")
    print(df["coala"].value_counts())

    # Проверки консистентности
    print("\n=== ПРОВЕРКИ КОНСИСТЕНТНОСТИ ===")

    def check(label, series, expected):
        """Проверяет, что множество значений series совпадает с expected."""
        vals = set(series.unique())
        if not vals:
            print(f"{label}: (нет строк — проверка пропущена)")
            return
        ok = vals == expected
        mark = "✅ OK" if ok else f"⚠️ ожидалось {expected}"
        print(f"{label}: {sorted(vals)} ({mark})")

    comm_in_new = (df["actonomy"] == "Communication").sum()
    print(f"Communication в actonomy (новая): {comm_in_new} "
          f"({'✅ OK' if comm_in_new == 0 else '⚠️ должны были уйти в Grounding'})")

    check("CoALA для actonomy=Memory",
          df.loc[df["actonomy"] == "Memory", "coala"],
          {"Internal"})
    check("CoALA для from_open_web",
          df.loc[df["subaction"] == "from_open_web", "coala"],
          {"External"})
    check("CoALA для from_relevant_context",
          df.loc[df["subaction"] == "from_relevant_context", "coala"],
          {"Internal"})
    check("CoALA для from_skill_library",
          df.loc[df["subaction"] == "from_skill_library", "coala"],
          {"Internal"})
    check("CoALA для terminating",
          df.loc[df["subaction"] == "terminating", "coala"],
          {"Internal"})

    expected_map = {
        "Internal": "Internal cognition",
        "External": "External interaction",
        "Unknown":  "Unknown",
        "Other":    "Unknown",
    }
    # Diff coala_legacy → coala (диагностика переездов)
    if compare:
        diff = df[df["coala_legacy"] != df["coala"]]
        print(f"\n=== DIFF coala_legacy → coala ===")
        print(f"Строк с изменением: {len(diff)} "
              f"({100 * len(diff) / len(df):.1f}%)")
        print("\nТоп-20 переездов (action, coala_legacy → coala):")
        top = (diff.groupby(["action", "coala_legacy", "coala"])
                    .size()
                    .sort_values(ascending=False)
                    .head(20))
        print(top)

    # Какие инструменты попали в "Other" (в новой или legacy-разметке)
    other_new = df[df["actonomy"] == "Other"]["action"].unique()
    other_old = df[df["actonomy_legacy"] == "Other"]["action"].unique()
    all_other = sorted(set(other_new) | set(other_old))
    if all_other:
        print(f"\n⚠️ Инструменты в 'Other':")
        for t in all_other:
            print(f"  - {t}")
    else:
        print("\n✅ Все инструменты классифицированы!")


def run_thoughts_pipeline():
    """
    Читает v_agent_trace_full (role='assistant'), извлекает текст из jsonb-поля
    content и классифицирует мысли rule-based. Сохраняет thoughts_with_taxonomy.csv.

    Дедупликация по (chain_attempt_id, step_attempt_id, event_order, role) с
    приоритетом stream_kind native > tool_trace > bridge: разные потоки могут
    дублировать одну историю. is_canary исключает smoke-задания целиком.
    """
    engine = create_engine(DB_URL)

    print("Загрузка v_agent_trace_full (role='assistant')...")
    # is_canary (dataset_chains) — канареечный smoke-таск; исключаем целиком.
    # content IS NOT NULL — jsonb-проекция; пустой объект {} пройдёт фильтр и
    # отсеется позже по длине извлечённого текста.
    sql = """
        SELECT
          chain_attempt_id::text AS chain_attempt_id,
          step_attempt_id::text AS step_attempt_id,
          step_index,
          event_order,
          trace_stream_id::text AS trace_stream_id,
          stream_kind,
          harness,
          model_identifier,
          role,
          content,
          native_agent_session_key
        FROM v_agent_trace_full
        WHERE role = 'assistant'
          AND content IS NOT NULL
        ORDER BY chain_attempt_id, step_index, trace_stream_id, event_order
    """
    df = pd.read_sql(text(sql), engine)
    print(f"Загружено строк с role='assistant': {len(df)}")

    if df.empty:
        print("\n⚠️ НЕТ мыслей в v_agent_trace_full (role='assistant' пуст).")
        return

    # Приоритет источников: native — достовернее, bridge — резервный
    prio = {"native": 0, "tool_trace": 1, "bridge": 2}
    df["_prio"] = df["stream_kind"].map(prio).fillna(99)
    df = (df.sort_values("_prio")
            .drop_duplicates(
                subset=["chain_attempt_id", "step_attempt_id",
                        "event_order", "role"],
                keep="first")
            .drop(columns=["_prio"]))
    print(f"После дедупликации: {len(df)}")

    print("Извлечение текста из content...")
    df["text"] = df["content"].apply(_extract_text)
    df = df[df["text"].str.len() > 5].copy()
    print(f"После отсечения пустых: {len(df)}")

    if df.empty:
        print("\n⚠️ Все content пустые после извлечения текста.")
        return

    print("Rule-based классификация мыслей...")
    classified = df["text"].apply(classify_thought).apply(pd.Series)
    df = pd.concat([df.drop(columns=["content"]), classified], axis=1)

    out = "thoughts_with_taxonomy.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n✅ Сохранено в {out}")
    print(f"   Строк: {len(df)}")

    print("\n=== Распределение thought_type ===")
    print(df["thought_type"].value_counts())

    print("\n=== thought_type по harness ===")
    print(df.groupby(["harness", "thought_type"]).size().unstack(fill_value=0))

    print("\n=== Средняя confidence по типу ===")
    print(df.groupby("thought_type")["confidence"].mean().round(2))


def main():
    """
    CLI-точка входа. --mode actions | thoughts | both (по умолчанию both);
    --compare — diff coala_legacy → coala (только при --mode actions/both).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["actions", "thoughts", "both"],
        default="both",
        help="actions — разметка действий; thoughts — разбор мыслей; both — оба",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="вывести diff coala_legacy vs coala (для диагностики переездов)",
    )
    args = ap.parse_args()

    if not DB_URL:
        print("[error] ATA_DB_URL не задан в .env", file=sys.stderr)
        sys.exit(1)

    if args.mode in ("actions", "both"):
        print("=" * 60)
        print("PIPELINE: ACTIONS")
        print("=" * 60)
        run_actions_pipeline(compare=args.compare)

    if args.mode in ("thoughts", "both"):
        print("\n" + "=" * 60)
        print("PIPELINE: THOUGHTS")
        print("=" * 60)
        run_thoughts_pipeline()


if __name__ == "__main__":
    main()  