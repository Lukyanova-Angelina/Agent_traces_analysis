import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


load_dotenv()
DB_URL = os.environ.get("ATA_DB_URL", "postgresql+psycopg2://ata_logs:meow2006@127.0.0.1:5432/ata_logs_import")

# ─── ПОЛНЫЙ СЛОВАРЬ: action → actonomy (сверено с вашим списком) ───
ACTION_TO_ACTONOMY = {
    # --- File System ---
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

    # --- Code & OS ---
    "exec": "Execution", "bash": "Execution", "terminal": "Execution",
    "execute_code": "Execution", "run_command": "Execution", "run_script": "Execution",
    "process": "Execution", 
    "query_code": "Retrieval", "search_code": "Retrieval",

    # --- Office Excel ---
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

    # --- Office DOCX ---
    "docx_inspect_document": "Retrieval", "docx_read_table": "Retrieval", "docx_read_text": "Retrieval",
    "docx_write_document": "Grounding",
    "office_officecli": "Grounding", # Уточняется через аргументы

    # --- Web & Vision ---
    "web_fetch": "Retrieval", "web_search": "Retrieval", "browse_page": "Retrieval",
    "vision_analyze": "Retrieval", "view_image": "Retrieval", "image": "Retrieval", "ocr_pdf": "Retrieval",
    "browser": "Grounding", "browser_click": "Grounding", "browser_back": "Grounding",
    "video_generate": "Grounding", "canvas": "Grounding",

    # --- Agent Meta (Planning) ---
    "task": "Planning", "todo": "Planning", "todowrite": "Planning",
    "update_plan": "Planning", "update_goal": "Planning", "create_goal": "Planning",
    "delegate_task": "Planning", "subagents": "Planning", "schedule_subagent": "Planning",
    "sessions_spawn": "Planning", "sessions_yield": "Planning", "cron": "Planning",
    "wait_task": "Planning", "skill": "Planning", "skill_view": "Planning", "skills_list": "Planning",
    
    # --- Agent Meta (Retrieval) ---
    "peek_task": "Retrieval", "get_task_result": "Retrieval",
    "session_search": "Retrieval", "session_status": "Retrieval", "sessions_history": "Retrieval",

    # --- Agent Meta (Evaluating / Deciding / Learning) ---
    "verify_and_record": "Evaluating", "task_acceptance_review": "Evaluating",
    "cancel_task": "Deciding", "discard_child_result": "Deciding",
    "skill_manage": "Learning", "skill_workshop": "Learning",

    # --- Communication ---
    "message": "Communication", "send_file": "Communication", "sessions_send": "Communication",

    # --- Memory ---
    "memory_get": "Memory", "memory_search": "Memory", "tree_read": "Memory", "tree_note": "Memory",

    # --- VCS & Other ---
    "vcs_diff": "Retrieval", "vcs_status": "Retrieval",
    "gateway": "Grounding", 

    "browser_navigate": "Retrieval",  
    "browser_snapshot": "Retrieval",
    "browser_console": "Execution",

# --- Communication ---
    "question": "Communication",
    
    # --- Null / Unknown ---
    "NULL": "Unknown", None: "Unknown"
}

# ─── СЛОВАРЬ: actonomy → coala ───────────────────────────────
ACTONOMY_TO_COALA = {
    "Planning": "Internal", "Reasoning": "Internal", "Deciding": "Internal",
    "Evaluating": "Internal", "Reflection": "Internal", "Learning": "Internal",
    "Communication": "External", "Retrieval": "External",
    "Grounding": "External", "Execution": "External",
    "Memory": "Memory", "Unknown": "Unknown", "Other": "Other"
}

# ─── Уточнение для «обманчивых» инструментов через arguments ───
def refine_by_arguments(action, arguments):
    """Смотрит внутрь аргументов только для тех инструментов, где имя неоднозначно."""
    if action not in {"bash", "exec", "terminal", "run_command", "run_script", "process",
                      "office_officecli", "browser", "browser_console", "read"}:
        return None

    if isinstance(arguments, str):
        try: arguments = json.loads(arguments)
        except Exception: arguments = {}
    if not isinstance(arguments, dict):
        return None

    if action in {"bash", "exec", "terminal", "run_command", "run_script", "process"}:
        cmd = str(arguments.get("command", "")).lower()
        if any(x in cmd for x in ["pip install", "apt-get", "apt install", "npm install"]):
            return "Grounding"   # настройка среды
        if any(x in cmd for x in ["ls ", "cat ", "head ", "tail ", "grep ", "find "]):
            return "Retrieval"   # просто чтение
        return "Execution"       # запуск скрипта

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
        if "command" in arguments:  # OpenCode умеет read-как-bash
            return "Execution"
        return "Retrieval"

    return None

def get_actonomy(action, arguments):
    if action is None or pd.isna(action):
        return "Unknown"
    # 1. Уточнение по аргументам (если инструмент обманчивый)
    refined = refine_by_arguments(action, arguments)
    if refined:
        return refined
    # 2. По имени инструмента
    return ACTION_TO_ACTONOMY.get(action, "Other")

def get_coala(actonomy):
    return ACTONOMY_TO_COALA.get(actonomy, "Other")

# ─── ЗАГРУЗКА И РАЗМЕТКА ────────────────────────────────────
def main():
    engine = create_engine(DB_URL)

    print("Загрузка v_agent_trace_actions...")
    sql = """
        SELECT chain_attempt_id, step_attempt_id, step_index,
               event_order, action, arguments, action_status,
               harness, model_identifier
        FROM v_agent_trace_actions
        ORDER BY chain_attempt_id, step_attempt_id, event_order
    """
    df = pd.read_sql(text(sql), engine)
    print(f"Загружено строк: {len(df)}")

    # Добавляем ДВЕ НОВЫЕ КОЛОНКИ
    print("Разметка Actonomy...")
    df["actonomy"] = df.apply(lambda r: get_actonomy(r["action"], r["arguments"]), axis=1)

    print("Разметка CoALA...")
    df["coala"] = df["actonomy"].apply(get_coala)

    # Сохраняем
    out = "actions_with_taxonomy.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n✅ Сохранено в {out}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Строк:   {len(df)}")

    # Быстрый отчёт
    print("\n=== Распределение Actonomy ===")
    print(df["actonomy"].value_counts())
    print("\n=== Распределение CoALA ===")
    print(df["coala"].value_counts())

    # Проверка: не осталось ли 'Other' для известных инструментов
    other_tools = df[df["actonomy"] == "Other"]["action"].unique()
    if len(other_tools) > 0:
        print(f"\n⚠️ Внимание: эти инструменты не попали в словарь и ушли в 'Other':")
        for t in other_tools:
            print(f"  - {t}")
    else:
        print("\n✅ Все инструменты успешно классифицированы!")

if __name__ == "__main__":
    main()