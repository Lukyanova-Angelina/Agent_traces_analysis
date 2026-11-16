#!/usr/bin/env python3
"""
Graphectory pipeline v3 — исправлена ошибка UUID IN + восстановлены отступы.

Ключевые изменения v3:
  - build_in_sql_literal: UUID-литералы в SQL вместо bindparam(expanding=True).
    UUID берутся из БД, поэтому безопасно подставлять как литералы.
  - Пофикшено n_cycles = 0 внутри цикла (было NameError при пустом edges).
  - Восстановлены отступы в extract_metrics (блок pairs/obj_counts
    всегда выполняется, не только в else-ветке --no-full).

Что делает:
  1. Читает v_chain_attempts (y = harness + model_identifier)
  2. Чанками тянет v_agent_trace_actions (+ опц. v_agent_trace_full)
  3. Строит nx.MultiDiGraph для каждой цепочки
  4. Извлекает ~45 структурных метрик (включая revisited-объекты)
  5. Пишет chain_features.csv построчно (прерывание + --resume)
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


load_dotenv()
DEFAULT_DB_URL = os.environ.get(
    "ATA_DB_URL",
    "postgresql+psycopg2://ata_logs:meow2006@127.0.0.1:5432/ata_logs_import",
)


# ─── колонки результирующего CSV ───
FIELDS = [
    "chain_attempt_id", "harness", "model_identifier", "status",
    "n_nodes", "n_edges", "n_actions", "n_results", "n_thoughts",
    "n_unique_tools",
    "n_errors", "n_running", "n_success", "n_other_status",
    "frac_errors", "frac_running", "frac_success",
    "n_actions_without_result", "frac_actions_without_result",
    "n_repeat_pairs", "n_repeat_excess", "longest_same_tool_run",
    "n_objects", "n_objects_revisited", "max_actions_per_object",
    "n_same_object_edges", "frac_revisited_actions", "n_cycles",
    "graph_depth", "branching_mean", "branching_max",
    "n_parallel_blocks", "n_sessions", "n_step_switches",
    "transition_entropy",
        # ... существующие поля до n_other_tools ...
    "frac_X", "frac_E", "frac_P", "frac_V", "n_other_tools",

    # ─── v4: XEPV-последовательности ───
    "xepv_sequence", "xepv_hash", "xepv_len",
    "n_xepv_bigrams", "xepv_entropy_bigram",
    "frac_E_to_V", "frac_V_to_E", "frac_X_to_E", "frac_E_to_X",
    "frac_X_to_P", "frac_P_to_E", "frac_E_to_E",

    # ─── v4: execution chains (2608.06909) ───
    "n_execution_chains", "longest_execution_chain", "avg_execution_chain_length",

    # ─── v4: component schema (2608.06909) ───
    "n_user_messages", "n_assistant_messages",
    "n_planning_thoughts", "n_reflexive_thoughts", "n_other_thoughts",
    "frac_planning_thoughts",

    # ... существующие метаданные ...
    "expected_step_count", "completed_step_count", "chain_completed_fraction",
    # ...
    "full_pass", "task_score",
]


# ─── SQL ───
CHAIN_SQL = """
    SELECT
      chain_attempt_id::text AS chain_attempt_id,
      harness, model_identifier, status,
      full_pass, task_score, chain_completed_fraction,
      expected_step_count, completed_step_count,
      end_to_end_seconds, input_tokens, output_tokens, total_tokens
    FROM v_chain_attempts
    WHERE is_canary = :include_canary
      AND model_identifier IS NOT NULL
      AND harness IS NOT NULL
    ORDER BY chain_attempt_id
"""

ACTIONS_SQL_TEMPLATE = """
    SELECT
      chain_attempt_id::text AS chain_attempt_id,
      step_attempt_id::text AS step_attempt_id,
      step_index, event_order, action,
      arguments::text AS arguments_json,
      tool_call_id, action_status, action_block_index,
      agent_session_key, trace_stream_id::text AS trace_stream_id
    FROM v_agent_trace_actions
    WHERE action IS NOT NULL
      AND chain_attempt_id IN (/*IDS*/)
"""

FULL_SQL_TEMPLATE = """
    SELECT
      chain_attempt_id::text AS chain_attempt_id,
      step_attempt_id::text AS step_attempt_id,
      step_index, event_order, stream_kind, event_type, role,
      tool_name, tool_call_id,
      content::text AS content_json,
      trace_stream_id::text AS trace_stream_id,
      native_agent_session_key AS agent_session_key
    FROM v_agent_trace_full
    WHERE role IN ('tool', 'assistant', 'user')   -- ← добавили 'user'
      AND chain_attempt_id IN (/*IDS*/)
"""


def build_in_sql_literal(base_sql: str, id_list: list[str]) -> str:
    """
    Подставляет UUID-строки в SQL как литералы: 'uuid1', 'uuid2', ...

    Безопасно, потому что id_list получен из БД (валидные UUID).
    Решает проблему sqlalchemy.exc.ArgumentError (f405) при
    bindparam(expanding=True) + UUID в psycopg2.
    """
    if not id_list:
        # заведомо ложное условие, чтобы запрос вернул 0 строк
        return base_sql.replace("/*IDS*/", "NULL")
    quoted = ", ".join(
        "'" + str(uid).replace("'", "''") + "'" for uid in id_list
    )
    return base_sql.replace("/*IDS*/", quoted)


# ─── helpers: arguments ───
def parse_args(s):
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _canon_path(p):
    """Убрать /workspace/ и /workspaces/.../workspace/ — оставить относительный путь."""
    if not isinstance(p, str):
        return None
    p = p.strip()
    if "/workspace/" in p:
        p = p.rsplit("/workspace/", 1)[-1]
    return p or None


def extract_object(a):
    """Канонический 'объект' действия для same_object-рёбер и revisited-метрик."""
    if not isinstance(a, dict):
        return None

    # 1) файл — все реальные варианты ключей (read/filePath, excel/filepath, write/path)
    for k in ("file_path", "filePath", "filepath", "file", "filename",
              "path", "target"):
        v = a.get(k)
        if isinstance(v, str) and v:
            return f"file:{_canon_path(v)}"

    # 2) command — строка (bash) или список (office_officecli)
    cmd = a.get("command")
    if isinstance(cmd, str) and cmd:
        words = cmd.strip().split()[:2]
        return f"cmd:{' '.join(words)}"
    if isinstance(cmd, list) and cmd:
        sub = str(cmd[0]) if cmd else ""
        tgt = _canon_path(cmd[1]) if len(cmd) > 1 and isinstance(cmd[1], str) else ""
        return f"cmd:{sub}:{tgt}"

    # 3) url
    if isinstance(a.get("url"), str) and a["url"]:
        return f"url:{_canon_path(a['url'])}"

    # 4) query/pattern/sheet
    for k in ("query", "pattern", "sheet_name"):
        v = a.get(k)
        if isinstance(v, str) and v:
            return f"{k}:{v}"

    return None


# ─── helpers: status ───
def classify_status(s):
    """По реальным значениям action_status (см. diag.py)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "unknown"
    s = str(s).lower().strip()
    if s in ("completed", "ok", "success", "done", "finished"):
        return "success"
    if s in ("requested", "running", "pending", "started", "in_progress"):
        return "running"
    if s in ("error", "failed", "failure", "non_zero_exit", "shell_error",
             "workspace_blocked", "light_mode_blocked", "blocked",
             "timeout", "interrupted"):
        return "error"
    return "other"


# ─── XEPV-группы ───
_X_PREFIXES = ("read", "glob", "grep", "list", "search", "get", "view",
               "fetch", "cat", "ls", "browse", "navigate", "snapshot",
               "query", "inspect", "peek", "session_search", "session_status",
               "sessions_history", "memory_get", "memory_search", "tree_read",
               "vcs_diff", "vcs_status", "web_fetch", "web_search",
               "ocr_pdf", "vision_analyze", "view_image", "file_fetch")
_X_EXACT = {"excel_read_data_from_excel", "excel_get_workbook_metadata",
            "excel_get_merged_cells", "docx_inspect_document",
            "docx_read_table", "docx_read_text", "browser_navigate",
            "browser_snapshot", "question"}

_E_PREFIXES = ("write", "edit", "bash", "exec", "run", "terminal", "install",
               "apply", "patch", "create", "format", "copy", "delete",
               "insert", "rename", "move", "send", "message", "spawn",
               "video_generate", "canvas", "browser", "gateway", "skill_manage",
               "skill_workshop", "tree_note")
_E_EXACT = {"office_officecli", "excel_write_data_to_excel", "excel_create_workbook",
            "excel_create_worksheet", "excel_format_range", "excel_apply_formula",
            "excel_copy_range", "excel_copy_worksheet", "excel_create_pivot_table",
            "excel_delete_sheet_columns", "excel_delete_sheet_rows",
            "excel_delete_worksheet", "excel_insert_rows", "excel_rename_worksheet",
            "docx_write_document", "send_file", "sessions_send",
            "browser_click", "browser_back"}

_P_PREFIXES = ("plan", "task", "todo", "update_plan", "update_goal",
               "create_goal", "delegate", "subagents", "schedule",
               "sessions_spawn", "sessions_yield", "cron", "wait_task", "skill")
_P_EXACT = {"todowrite", "skills_list", "skill_view"}

_V_PREFIXES = ("pytest", "tsc", "lint", "check", "verify", "test", "validate")
_V_EXACT = {"excel_validate_formula_syntax", "verify_and_record",
            "task_acceptance_review"}


def tool_to_group(t):
    if not t:
        return "other"
    tl = str(t).lower()
    if tl in _V_EXACT or any(tl.startswith(p) for p in _V_PREFIXES):
        return "V"
    if tl in _P_EXACT or any(tl.startswith(p) for p in _P_PREFIXES):
        return "P"
    if tl in _E_EXACT or any(tl.startswith(p) for p in _E_PREFIXES):
        return "E"
    if tl in _X_EXACT or any(tl.startswith(p) for p in _X_PREFIXES):
        return "X"
    return "other"


# ─── построение графа ───
def _node_sort_key(r):
    return (
        r["step_index"] if r["step_index"] is not None else -1,
        r["trace_stream_id"] or "",
        r["event_order"] if r["event_order"] is not None else -1,
        int(r["action_block_index"]) if r["action_block_index"] is not None else 0,
    )

# ─── v4: классификация мыслей (2608.06909 component schema) ───
_PLANNING_KW = (
    "plan", "todo", "next step", "i will", "i should", "let me",
    "first", "then", "approach", "strategy", "step 1", "goal",
    "need to", "will need", "must",
)
_REFLEXIVE_KW = (
    "seems", "looks like", "error", "fail", "hmm", "wait",
    "actually", "unexpected", "incorrect", "wrong", "issue",
    "problem", "not what", "let me try", "instead", "however",
    "failed", "broken",
)


def _content_to_text(content_json):
    """Извлекает текст из content jsonb (dict / list / str)."""
    if not content_json:
        return ""
    try:
        v = json.loads(content_json)
    except Exception:
        return str(content_json).lower()
    if isinstance(v, str):
        return v.lower()
    if isinstance(v, dict):
        for k in ("text", "content", "message", "body"):
            if k in v and isinstance(v[k], str):
                return v[k].lower()
        return json.dumps(v).lower()
    if isinstance(v, list):
        parts = []
        for item in v:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return " ".join(parts).lower()
    return str(v).lower()


def classify_thought(content_json):
    s = _content_to_text(content_json)
    if not s:
        return "other"
    plan_score = sum(1 for kw in _PLANNING_KW if kw in s)
    refl_score = sum(1 for kw in _REFLEXIVE_KW if kw in s)
    if refl_score > plan_score:
        return "reflexive"
    if plan_score > 0:
        return "planning"
    return "other"


# ─── v4: XEPV-последовательности (point 1) ───
def xepv_sequence_features(tools_ordered):
    """tools_ordered: список имён инструментов в порядке (step, stream, event, block)."""
    import hashlib
    seq = [tool_to_group(t) for t in tools_ordered]
    seq_str = "".join(seq)
    seq_hash = hashlib.md5(seq_str.encode()).hexdigest()[:12] if seq_str else None

    bigrams = Counter(zip(seq, seq[1:]))
    total_bg = sum(bigrams.values())

    def _bfrac(a, b):
        return bigrams.get((a, b), 0) / total_bg if total_bg else 0.0

    if total_bg:
        arr = np.array(list(bigrams.values()), dtype=float) / total_bg
        xepv_ent = float(-np.sum(arr * np.log2(arr + 1e-12)))
    else:
        xepv_ent = 0.0

    return {
        "xepv_sequence": seq_str[:200],
        "xepv_hash": seq_hash,
        "xepv_len": len(seq),
        "n_xepv_bigrams": len(bigrams),
        "xepv_entropy_bigram": xepv_ent,
        "frac_E_to_V": _bfrac("E", "V"),
        "frac_V_to_E": _bfrac("V", "E"),
        "frac_X_to_E": _bfrac("X", "E"),
        "frac_E_to_X": _bfrac("E", "X"),
        "frac_X_to_P": _bfrac("X", "P"),
        "frac_P_to_E": _bfrac("P", "E"),
        "frac_E_to_E": _bfrac("E", "E"),
    }


# ─── v4: execution chains (point 4) ───
def execution_chain_metrics(g):
    """Цепочки = компоненты связности в подграфе same_object + retry рёбер."""
    chain_g = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        if d["kind"] in ("same_object", "retry"):
            chain_g.add_edge(u, v)

    if chain_g.number_of_nodes() == 0:
        return {
            "n_execution_chains": 0,
            "longest_execution_chain": 0,
            "avg_execution_chain_length": 0.0,
        }

    comps = list(nx.connected_components(chain_g.to_undirected()))
    sizes = [len(c) for c in comps]
    n_chains = len(comps)
    avg_len = sum(sizes) / len(sizes) if sizes else 0.0

    longest = 0
    try:
        order = list(nx.topological_sort(chain_g))
        dist = {n: 0 for n in chain_g.nodes()}
        for u in order:
            for v in chain_g.successors(u):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
        longest = max(dist.values()) if dist else 0
    except nx.NetworkXUnfeasible:
        longest = max(sizes) - 1 if sizes else 0

    return {
        "n_execution_chains": int(n_chains),
        "longest_execution_chain": int(longest),
        "avg_execution_chain_length": float(avg_len),
    }
def build_graph(action_rows, full_rows=None, include_thoughts=False):
    g = nx.MultiDiGraph()
    a = sorted(action_rows, key=_node_sort_key)

    action_nodes = []
    prev, prev_key = None, None
    for r in a:
        bidx = int(r["action_block_index"]) if r["action_block_index"] is not None else 0
        nid = f"act::{r['trace_stream_id']}::{r['event_order']}::{bidx}"
        args = parse_args(r["arguments_json"])
        g.add_node(
            nid, kind="action",
            tool_name=r["action"], tool_call_id=r["tool_call_id"],
            status=classify_status(r["action_status"]),
            step_index=r["step_index"],
            event_order=r["event_order"],
            block_index=bidx, session=r["agent_session_key"],
            args=args, obj=extract_object(args),
        )
        action_nodes.append(nid)

        k = _node_sort_key(r)
        if prev is not None:
            step_p, stream_p, eo_p, _ = prev_key
            step_c, stream_c, eo_c, _ = k
            if (step_c, stream_c) != (step_p, stream_p):
                g.add_edge(prev, nid, kind="cross_stream")
            elif eo_c > eo_p:
                g.add_edge(prev, nid, kind="next")
            else:
                g.add_edge(prev, nid, kind="parallel")
        prev, prev_key = nid, k

    if full_rows:
        # дедуп results по (step_attempt_id, tool_call_id), приоритет native > tool_trace > bridge
        results = {}
        for r in full_rows:
            if r["role"] != "tool" or r["tool_call_id"] is None:
                continue
            key = (r["step_attempt_id"], r["tool_call_id"])
            prio = {"native": 0, "tool_trace": 1, "bridge": 2}.get(r["stream_kind"], 99)
            if key not in results or prio < results[key][0]:
                results[key] = (prio, r)

        action_by_call = {}
        for an in action_nodes:
            tc = g.nodes[an].get("tool_call_id")
            if tc and tc not in action_by_call:
                action_by_call[tc] = an

        for _, r in results.values():
            nid = f"res::{r['trace_stream_id']}::{r['event_order']}::{r['tool_call_id']}"
            g.add_node(nid, kind="result",
                       tool_call_id=r["tool_call_id"],
                       event_order=r["event_order"],
                       step_index=r["step_index"],
                       content=r["content_json"])
            an = action_by_call.get(r["tool_call_id"])
            if an is not None:
                g.add_edge(an, nid, kind="call_result")

        if include_thoughts:
            seen = set()
            for r in sorted(full_rows,
                            key=lambda x: (x["step_attempt_id"] or "", x["event_order"])):
                if r["role"] != "assistant":
                    continue
                key = (r["step_attempt_id"], r["event_order"])
                if key in seen:
                    continue
                seen.add(key)
                nid = f"tht::{r['trace_stream_id']}::{r['event_order']}"
                g.add_node(nid, kind="thought",
                           event_order=r["event_order"],
                           step_index=r["step_index"],
                           content=r["content_json"])
                for an in action_nodes:
                    if g.nodes[an]["event_order"] >= r["event_order"]:
                        g.add_edge(nid, an, kind="thought_leads_to")
                        break

    # same_object рёбра (только вперёд по времени)
    by_obj = defaultdict(list)
    for n in action_nodes:
        o = g.nodes[n].get("obj")
        if o:
            by_obj[o].append(n)
    for o, ns in by_obj.items():
        for i in range(len(ns) - 1):
            g.add_edge(ns[i], ns[i + 1], kind="same_object", obj=o)

    # retry рёбра (тот же tool+args после error)
    for i, n in enumerate(action_nodes):
        if g.nodes[n]["status"] != "error":
            continue
        for m in action_nodes[i + 1:]:
            if (g.nodes[m]["tool_name"] == g.nodes[n]["tool_name"]
                    and g.nodes[m]["args"] == g.nodes[n]["args"]):
                g.add_edge(n, m, kind="retry")
                break

        # ─── v4: component schema (роли + классификация мыслей) ───
    import hashlib
    if full_rows:
        seen_user, seen_assistant = set(), set()
        n_user, n_assist = 0, 0
        n_plan, n_refl, n_oth = 0, 0, 0
        for r in full_rows:
            role = r["role"]
            ch = hashlib.md5(str(r["content_json"]).encode()).hexdigest()
            if role == "user":
                h = (r["step_attempt_id"], ch)
                if h not in seen_user:
                    seen_user.add(h)
                    n_user += 1
            elif role == "assistant":
                h = (r["step_attempt_id"], ch)
                if h not in seen_assistant:
                    seen_assistant.add(h)
                    n_assist += 1
                    tt = classify_thought(r["content_json"])
                    if tt == "planning":
                        n_plan += 1
                    elif tt == "reflexive":
                        n_refl += 1
                    else:
                        n_oth += 1
        g.graph["n_user_messages"] = n_user
        g.graph["n_assistant_messages"] = n_assist
        g.graph["n_planning_thoughts"] = n_plan
        g.graph["n_reflexive_thoughts"] = n_refl
        g.graph["n_other_thoughts"] = n_oth
    else:
        for k in ("n_user_messages", "n_assistant_messages",
                  "n_planning_thoughts", "n_reflexive_thoughts",
                  "n_other_thoughts"):
            g.graph[k] = None

    return g


# ─── метрики ───
def _longest_path_edges(dg):
    """Длина самого длинного пути в рёбрах через топологическую сортировку."""
    if dg.number_of_edges() == 0:
        return 0
    try:
        order = list(nx.topological_sort(dg))
    except nx.NetworkXUnfeasible:
        return 0
    dist = {n: 0 for n in dg.nodes()}
    for u in order:
        for v in dg.successors(u):
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1
    return max(dist.values()) if dist else 0


def extract_metrics(g, chain, has_full=False):
    nodes = list(g.nodes(data=True))
    edges = list(g.edges(data=True))
    kinds = Counter(d["kind"] for _, d in nodes)

    # ─── сначала actions/tools, потом всё, что от них зависит ───
    actions = [d for _, d in nodes if d["kind"] == "action"]
    tools = [d["tool_name"] for d in actions]
    n_actions = len(actions)
    denom = n_actions if n_actions else 1

    # ─── v4: execution chains ───
    exec_metrics = execution_chain_metrics(g)

    # ─── v4: XEPV-последовательности ───
    xepv_feats = xepv_sequence_features(tools)

    # статусы
    status_cnt = Counter(d["status"] for d in actions)
    n_err = status_cnt.get("error", 0)
    n_run = status_cnt.get("running", 0)
    n_ok = status_cnt.get("success", 0)
    n_oth = sum(v for k, v in status_cnt.items()
                if k not in ("error", "running", "success", "unknown"))

    # ---- зависимые от --full ----
    if has_full:
        call_with_result = {g.nodes[u].get("tool_call_id")
                            for u, _, d in edges if d["kind"] == "call_result"}
        no_result = sum(1 for d in actions
                        if d["tool_call_id"] and d["tool_call_id"] not in call_with_result)
        frac_no_result = no_result / denom
    else:
        no_result = None
        frac_no_result = None

    # ---- объекты и повторные визиты ----
    pairs = Counter((d["tool_name"], d["obj"]) for d in actions if d["obj"])
    n_repeat_pairs = sum(1 for c in pairs.values() if c > 1)
    n_repeat_excess = sum(c - 1 for c in pairs.values() if c > 1)

    obj_counts = Counter(d["obj"] for d in actions if d.get("obj"))
    n_objects = len(obj_counts)
    n_objects_revisited = sum(1 for c in obj_counts.values() if c > 1)
    max_actions_per_object = max(obj_counts.values()) if obj_counts else 0
    actions_revisiting = sum(c - 1 for c in obj_counts.values() if c > 1)
    frac_revisited_actions = actions_revisiting / n_actions if n_actions else 0.0
    n_same_object_edges = sum(1 for _, _, d in edges if d["kind"] == "same_object")

    # ---- серии одинаковых инструментов ----
    longest = cur = 0
    prev_t = None
    for t in tools:
        cur = cur + 1 if t == prev_t else 1
        prev_t = t
        longest = max(longest, cur)

    # ---- параллельные блоки и сессии/шаги ----
    eo_cnt = Counter((d["step_index"], d["event_order"]) for d in actions)
    n_parallel = sum(1 for c in eo_cnt.values() if c > 1)

    sessions = {d["session"] for d in actions if d["session"]}
    steps = [d["step_index"] for d in actions if d["step_index"] is not None]
    n_step_sw = sum(1 for x, y in zip(steps, steps[1:]) if x != y)

    # ---- depth/branching ----
    ng = nx.DiGraph()
    for u, v, d in edges:
        if d["kind"] in ("next", "cross_stream", "thought_leads_to"):
            ng.add_edge(u, v)

    depth = _longest_path_edges(ng)
    if ng.number_of_edges() > 0:
        out_deg = [dd for _, dd in ng.out_degree()]
        b_mean = float(np.mean(out_deg)) if out_deg else 0.0
        b_max = int(max(out_deg)) if out_deg else 0
    else:
        b_mean, b_max = 0.0, 0

    # ---- n_cycles ----
    g_proj = nx.DiGraph()
    for u, v, _d in edges:
        g_proj.add_edge(u, v)
    n_cycles = 0
    try:
        n_cycles = len(list(nx.simple_cycles(g_proj, length_bound=3)))
    except Exception:
        n_cycles = 0

    # ---- энтропия переходов ----
    trans = Counter()
    for u, v, d in edges:
        if d["kind"] in ("next", "cross_stream"):
            trans[(g.nodes[u]["tool_name"], g.nodes[v]["tool_name"])] += 1
    tot = sum(trans.values())
    if tot:
        arr = np.array(list(trans.values()), dtype=float) / tot
        ent = float(-np.sum(arr * np.log2(arr + 1e-12)))
    else:
        ent = 0.0

    # ---- XEPV доли ----
    grp = Counter(tool_to_group(t) for t in tools)

    return {
        "chain_attempt_id": chain["chain_attempt_id"],
        "harness": chain["harness"],
        "model_identifier": chain["model_identifier"],
        "status": chain["status"],
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_actions": n_actions,
        "n_results": kinds.get("result", 0),
        "n_thoughts": kinds.get("thought", 0),
        "n_unique_tools": len(set(tools)),
        "n_errors": n_err,
        "n_running": n_run,
        "n_success": n_ok,
        "n_other_status": n_oth,
        "frac_errors": n_err / denom,
        "frac_running": n_run / denom,
        "frac_success": n_ok / denom,
        "n_actions_without_result": no_result,
        "frac_actions_without_result": frac_no_result,
        "n_repeat_pairs": n_repeat_pairs,
        "n_repeat_excess": n_repeat_excess,
        "longest_same_tool_run": longest,
        "n_objects": n_objects,
        "n_objects_revisited": n_objects_revisited,
        "max_actions_per_object": max_actions_per_object,
        "n_same_object_edges": n_same_object_edges,
        "frac_revisited_actions": frac_revisited_actions,
        "n_cycles": n_cycles,
        "graph_depth": depth,
        "branching_mean": b_mean,
        "branching_max": b_max,
        "n_parallel_blocks": n_parallel,
        "n_sessions": len(sessions),
        "n_step_switches": n_step_sw,
        "transition_entropy": ent,
        "frac_X": grp.get("X", 0) / denom,
        "frac_E": grp.get("E", 0) / denom,
        "frac_P": grp.get("P", 0) / denom,
        "frac_V": grp.get("V", 0) / denom,
        "n_other_tools": grp.get("other", 0),
        **xepv_feats,
        **exec_metrics,
        "n_user_messages": g.graph.get("n_user_messages"),
        "n_assistant_messages": g.graph.get("n_assistant_messages"),
        "n_planning_thoughts": g.graph.get("n_planning_thoughts"),
        "n_reflexive_thoughts": g.graph.get("n_reflexive_thoughts"),
        "n_other_thoughts": g.graph.get("n_other_thoughts"),
        "frac_planning_thoughts": (
            g.graph["n_planning_thoughts"] / g.graph["n_assistant_messages"]
            if g.graph.get("n_assistant_messages") else None
        ),
        "expected_step_count": chain["expected_step_count"],
        "completed_step_count": chain["completed_step_count"],
        "chain_completed_fraction": chain["chain_completed_fraction"],
        "end_to_end_seconds": chain["end_to_end_seconds"],
        "input_tokens": chain["input_tokens"],
        "output_tokens": chain["output_tokens"],
        "total_tokens": chain["total_tokens"],
        "full_pass": chain["full_pass"],
        "task_score": chain["task_score"],
    }

def empty_metrics(chain):
    """Строка для цепочки без actions — все графовые признаки NA."""
    meta_keys = {
        "chain_attempt_id", "harness", "model_identifier", "status",
        "expected_step_count", "completed_step_count",
        "chain_completed_fraction", "end_to_end_seconds",
        "input_tokens", "output_tokens", "total_tokens",
        "full_pass", "task_score",
        # v4 — все остальные поля будут None, но метаданные сохраним
    }
    # остальные (не meta) автоматически получат None из первого row = {k: None ...}
    # так что просто убедись, что новые имена НЕ в meta_keys — они сами заполнятся None.
    row = {k: None for k in FIELDS if k not in meta_keys}
    row.update({
        "chain_attempt_id": chain["chain_attempt_id"],
        "harness": chain["harness"],
        "model_identifier": chain["model_identifier"],
        "status": chain["status"],
        "expected_step_count": chain["expected_step_count"],
        "completed_step_count": chain["completed_step_count"],
        "chain_completed_fraction": chain["chain_completed_fraction"],
        "end_to_end_seconds": chain["end_to_end_seconds"],
        "input_tokens": chain["input_tokens"],
        "output_tokens": chain["output_tokens"],
        "total_tokens": chain["total_tokens"],
        "full_pass": chain["full_pass"],
        "task_score": chain["task_score"],
    })
    return row


# ─── I/O ───
def load_done_ids(out_path):
    p = Path(out_path)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(out_path, usecols=["chain_attempt_id"], dtype=str)
        return set(df["chain_attempt_id"].dropna())
    except Exception as e:
        print(f"[warn] cannot read {out_path}: {e}", file=sys.stderr)
        return set()


def open_writer(out_path, fresh):
    f = open(out_path, "w" if fresh else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if fresh:
        w.writeheader()
    return f, w


# ─── main ───
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", default=DEFAULT_DB_URL)
    ap.add_argument("--out", default="chain_features.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--thoughts", action="store_true")
    ap.add_argument("--include-canary", action="store_true")
    ap.add_argument("--include-empty", action="store_true",
                    help="писать цепочки без actions (все граф-метрики = NA)")
    ap.add_argument("--chunk-chains", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.overwrite and Path(args.out).exists():
        Path(args.out).unlink()
    fresh = not Path(args.out).exists()

    print(f"[connect] {args.db_url.split('@')[-1]}")
    engine = create_engine(args.db_url)

    print("[chains] loading...")
    chains_df = pd.read_sql(
        text(CHAIN_SQL), engine,
        params={"include_canary": args.include_canary},
    )
    chains_df["chain_attempt_id"] = chains_df["chain_attempt_id"].astype(str)
    print(f"[chains] {len(chains_df)} after filters")

    if args.offset:
        chains_df = chains_df.iloc[args.offset:]
    if args.limit:
        chains_df = chains_df.iloc[: args.limit]

    if args.dry_run:
        print(f"[dry-run] would process {len(chains_df)} chains")
        engine.dispose()
        return

    done = load_done_ids(args.out) if args.resume else set()
    chains_df = chains_df[~chains_df["chain_attempt_id"].isin(done)]
    print(f"[resume] already done: {len(done)}, todo: {len(chains_df)}")

    if chains_df.empty:
        print("[done] nothing to do")
        engine.dispose()
        return

    use_full = not args.no_full
    include_thoughts = args.thoughts and use_full
    if args.thoughts and not use_full:
        print("[warn] --thoughts ignored (needs --full)", file=sys.stderr)

    f, writer = open_writer(args.out, fresh=fresh)
    t0 = time.time()
    n_done = 0
    n_empty = 0

    all_chains = chains_df.to_dict("records")
    try:
        for start in range(0, len(all_chains), args.chunk_chains):
            chunk = all_chains[start: start + args.chunk_chains]
            ids = [c["chain_attempt_id"] for c in chunk]
            print(f"[chunk {start // args.chunk_chains + 1}] {len(chunk)} chains...")

            sql_a = build_in_sql_literal(ACTIONS_SQL_TEMPLATE, ids)
            actions_df = pd.read_sql(text(sql_a), engine)
            if not actions_df.empty:
                actions_df["chain_attempt_id"] = actions_df["chain_attempt_id"].astype(str)
                actions_df["step_attempt_id"] = actions_df["step_attempt_id"].astype(str)
            print(f"  actions: {len(actions_df)}")

            full_df = None
            if use_full:
                sql_f = build_in_sql_literal(FULL_SQL_TEMPLATE, ids)
                full_df = pd.read_sql(text(sql_f), engine)
                if not full_df.empty:
                    full_df["chain_attempt_id"] = full_df["chain_attempt_id"].astype(str)
                    full_df["step_attempt_id"] = full_df["step_attempt_id"].astype(str)
                print(f"  full: {len(full_df)}")

            a_by_chain = defaultdict(list)
            if not actions_df.empty:
                for r in actions_df.to_dict("records"):
                    a_by_chain[r["chain_attempt_id"]].append(r)

            f_by_chain = defaultdict(list)
            if full_df is not None and not full_df.empty:
                for r in full_df.to_dict("records"):
                    f_by_chain[r["chain_attempt_id"]].append(r)

            for chain in chunk:
                cid = chain["chain_attempt_id"]
                a_rows = a_by_chain.get(cid, [])
                if not a_rows:
                    if args.include_empty:
                        writer.writerow(empty_metrics(chain))
                        n_empty += 1
                    continue
                f_rows = f_by_chain.get(cid, []) if use_full else None
                try:
                    g = build_graph(a_rows, f_rows, include_thoughts=include_thoughts)
                    writer.writerow(extract_metrics(g, chain, has_full=use_full))
                    n_done += 1
                except Exception as e:
                    print(f"[skip {cid}] {type(e).__name__}: {e}", file=sys.stderr)

            f.flush()
            elapsed = time.time() - t0
            rate = (n_done + n_empty) / elapsed if elapsed > 0 else 0
            print(f"[progress] {n_done}+{n_empty}empty / {len(all_chains)} "
                  f"({rate:.1f}/sec, {elapsed:.0f}s)")

    except KeyboardInterrupt:
        print("\n[interrupt] flushing...")
    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
        engine.dispose()

    print(f"[done] {n_done} chains + {n_empty} empty → {args.out}")


if __name__ == "__main__":
    main()