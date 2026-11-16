#!/usr/bin/env python3
"""
CPA: весь пайплайн в одном файле.

  python cpa.py extract     --chains chain_features_full.csv --out episodes.csv
  python cpa.py sample      --episodes episodes.csv --out sample.csv --pilot pilot.csv
  python cpa.py label       --sample pilot.csv --out proposals.jsonl --limit 25
  python cpa.py label       --sample sample.csv --out proposals.jsonl      # докачка пропущенных
  python cpa.py consolidate --proposals proposals.jsonl --codebook codebook.json --out labelled.csv
  python cpa.py report      --proposals proposals.jsonl
  python cpa.py agreement   --a runA.jsonl --b runB.jsonl

.env: ATA_DB_URL, CPA_API_KEY, CPA_API_BASE, CPA_MODEL, CPA_TEMPERATURE (по умолчанию 0)
"""
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

load_dotenv()
SEED = 42
FIELDS = ("proximal_goal", "principal_output", "state_change", "preconditions")
DONE = {"ok", "abstain"}
KEEP = ("chain_attempt_id", "episode_id", "harness", "model_identifier",
        "tool_name", "obj", "n_actions", "statuses")
API_KEY = os.environ.get("CPA_API_KEY")
API_BASE = os.environ.get("CPA_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("CPA_MODEL", "gpt-4o-mini")
TEMP = float(os.environ.get("CPA_TEMPERATURE", "0"))

LABEL_SYSTEM = """You annotate agent-trace episodes. An EPISODE is N consecutive tool calls
with the same tool_name and object. Identify the PROCEDURAL FUNCTION it performs,
defined by four fields: proximal_goal, principal_output, state_change, preconditions.
Rules:
- cpa_label is SCREAMING_SNAKE_CASE and must NOT contain the tool name (RETRIEVE_SECTION, not READ_FILE).
- n_actions is the true count. Do not call a multi-action episode "a single action".
- Abstain for pure navigation (cd, pwd, ls, git status), noise, or a single failed action with no retry.
- A single successful action with a clear goal IS labelled.
Output JSON only:
{"abstain": false, "cpa_label": "...", "proximal_goal": "...", "principal_output": "...",
 "state_change": "...", "preconditions": "...", "confidence": 0.0}
or {"abstain": true, "reason": "..."}"""

CONS_SYSTEM = """You consolidate candidate labels into a codebook of procedural functions.
Merge two labels only if proximal_goal AND state_change match. Keep them apart if state_change
differs (read vs write vs verify). Tool names, endpoints and wording never decide identity.
Every input label must appear in exactly one "aliases" list. Output JSON only:
{"functions": [{"canonical_label": "...", "aliases": ["..."],
  "definition": {"proximal_goal": "...", "principal_output": "...", "state_change": "...", "preconditions": "..."},
  "rationale": "..."}]}"""

REPAIR_SYSTEM = """Assign each UNCOVERED label to an EXISTING canonical function when its proximal_goal
and state_change match. Otherwise propose a new canonical name. Output JSON only:
{"assignments": [{"label": "...", "canonical": "..."}]}"""

ACTIONS_SQL = text("""
    SELECT chain_attempt_id, step_index, trace_stream_id, event_order,
           action_block_index, action, arguments, action_status
    FROM v_agent_trace_actions
    WHERE action IS NOT NULL AND chain_attempt_id IN :ids
    ORDER BY chain_attempt_id, step_index, trace_stream_id, event_order, action_block_index
""").bindparams(bindparam("ids", expanding=True))


# ───── утилиты ─────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_json(s):
    try:
        return json.loads(s)
    except ValueError:
        m = re.search(r"\{.*\}", s or "", flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def call_llm(messages, max_tokens):
    if not API_KEY:
        sys.exit("[error] CPA_API_KEY не задан")
    payload = {"model": MODEL, "messages": messages,
               "temperature": TEMP, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"}}
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(5):
        r = requests.post(url, json=payload, headers=headers, timeout=120)
        if r.status_code == 400 and "response_format" in payload:
            payload.pop("response_format")  # провайдер не поддерживает json_mode
            continue
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("LLM: исчерпаны попытки")


def llm_json(system, user, max_tokens):
    return parse_json(call_llm([{"role": "system", "content": system},
                                {"role": "user", "content": user}], max_tokens))


# ───── 1. эпизоды ─────
def parse_args(s):
    try:
        d = json.loads(s) if isinstance(s, str) and s.strip() else {}
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def canon_path(p):
    p = p.strip()
    return p.rsplit("/workspace/", 1)[-1] if "/workspace/" in p else p


def extract_object(a):
    """Канонический объект действия (как в исходном graphectory-пайплайне)."""
    for k in ("file_path", "filePath", "filepath", "file", "filename", "path", "target"):
        v = a.get(k)
        if isinstance(v, str) and v:
            return f"file:{canon_path(v)}"
    cmd = a.get("command")
    if isinstance(cmd, str) and cmd.strip():
        return "cmd:" + " ".join(cmd.split()[:2])
    if isinstance(cmd, list) and cmd:
        tgt = canon_path(cmd[1]) if len(cmd) > 1 and isinstance(cmd[1], str) else ""
        return f"cmd:{cmd[0]}:{tgt}"
    if isinstance(a.get("url"), str) and a["url"]:
        return f"url:{canon_path(a['url'])}"
    for k in ("query", "pattern", "sheet_name"):
        v = a.get(k)
        if isinstance(v, str) and v:
            return f"{k}:{v}"
    return ""


def build_episodes(acts):
    """Серии подряд идущих вызовов с одинаковыми (tool, obj, step)."""
    eps, cur, key = [], None, None
    for a in acts:
        obj = extract_object(parse_args(a["arguments"]))
        k = (a["action"], obj, a["step_index"])
        if k != key:
            cur = {"tool_name": a["action"], "obj": obj,
                   "step_index": a["step_index"], "actions": []}
            eps.append(cur)
            key = k
        cur["actions"].append(a)
    return eps


def episode_row(ep, idx, cid, harness, model):
    acts = ep["actions"]
    first = acts[0]["arguments"]
    return {
        "chain_attempt_id": cid, "harness": harness, "model_identifier": model,
        "episode_id": idx, "step_index": ep["step_index"], "n_actions": len(acts),
        "tool_name": ep["tool_name"], "obj": ep["obj"] or "",
        "statuses": "|".join(sorted({str(a["action_status"] or "unknown") for a in acts})),
        "unique_args": len({a["arguments"][:200] for a in acts}),
        "sample_args": first[:300] + ("..." if len(first) > 300 else ""),
    }


def cmd_extract(a):
    chains = pd.read_csv(a.chains, usecols=["chain_attempt_id", "harness",
                                            "model_identifier", "n_actions"])
    chains = chains[chains["n_actions"].fillna(0) > 0].copy()
    chains["chain_attempt_id"] = chains["chain_attempt_id"].astype(str)
    meta = chains.set_index("chain_attempt_id")[["harness", "model_identifier"]].to_dict("index")
    ids = list(meta)[: a.limit] if a.limit else list(meta)

    engine = create_engine(os.environ["ATA_DB_URL"])
    rows = []
    with engine.connect() as conn:
        for s in range(0, len(ids), a.chunk):
            df = pd.read_sql(ACTIONS_SQL, conn, params={"ids": ids[s:s + a.chunk]})
            if df.empty:
                continue
            df["chain_attempt_id"] = df["chain_attempt_id"].astype(str)
            df["arguments"] = df["arguments"].map(
                lambda v: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str))
            for cid, g in df.groupby("chain_attempt_id", sort=False):
                if cid not in meta:
                    continue
                m = meta[cid]
                for i, ep in enumerate(build_episodes(g.to_dict("records"))):
                    rows.append(episode_row(ep, i, cid, m["harness"], m["model_identifier"]))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"[done] {len(rows)} эпизодов → {a.out}")


# ───── 2. выборка ─────
def cmd_sample(a):
    df = pd.read_csv(a.episodes, dtype=str, keep_default_na=False)
    df["n_actions"] = df["n_actions"].astype(int)
    groups = {k: g for k, g in df.groupby(["tool_name", "obj"])}
    sizes = {k: len(g) for k, g in groups.items()}
    rnd = random.Random(SEED)

    def per_pair(keys, k, name):
        return pd.concat([groups[key].sample(min(k, len(groups[key])), random_state=SEED)
                          for key in keys]).assign(stratum=name)

    top = sorted(sizes, key=lambda x: (-sizes[x], x))[:30]
    mid = sorted(x for x, n in sizes.items() if 10 <= n <= 100)
    tail = sorted(x for x, n in sizes.items() if 2 <= n <= 9)
    long_ = df[df["n_actions"] > 5]
    errs = df[df["statuses"].str.contains("error")]
    parts = [
        per_pair(top, 2, "A_top_pairs"),
        per_pair(rnd.sample(mid, min(40, len(mid))), 1, "B_mid_pairs"),
        per_pair(rnd.sample(tail, min(30, len(tail))), 1, "C_tail_pairs"),
        long_.sample(min(40, len(long_)), random_state=SEED).assign(stratum="D_long"),
        errs.sample(min(30, len(errs)), random_state=SEED).assign(stratum="E_errors"),
    ]
    s = pd.concat(parts).drop_duplicates(["chain_attempt_id", "episode_id"])
    s.to_csv(a.out, index=False)
    s.groupby("stratum").head(a.pilot_per_stratum).to_csv(a.pilot, index=False)
    print(s["stratum"].value_counts().to_string())


# ───── 3. открытая индукция ─────
def validate(p):
    if p.get("abstain"):
        return "abstain"
    if not all(p.get(k) for k in ("cpa_label", *FIELDS)):
        return "missing_field"
    if not re.fullmatch(r"[A-Z][A-Z0-9_]+", p["cpa_label"]):
        return "bad_label"
    return "ok"


def build_user_prompt(r):
    return (f"harness: {r['harness']}\nmodel: {r['model_identifier']}\n"
            f"tool_name: {r['tool_name']}\nobject: {r['obj'] or '(none)'}\n"
            f"n_actions: {r['n_actions']}\nstatuses: {r['statuses']}\n\n"
            f"SAMPLE ARGUMENTS (first action):\n{r['sample_args'][:400]}\n\n"
            "Return JSON per the schema.")


def cmd_label(a):
    df = pd.read_csv(a.sample, dtype=str, keep_default_na=False)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["chain_attempt_id"], r["episode_id"])
                for r in load_jsonl(out) if r["_status"] in DONE}
    todo = [r for r in df.to_dict("records")
            if (r["chain_attempt_id"], r["episode_id"]) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[label] уже готово {len(done)}, в очереди {len(todo)}")

    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            try:
                prop = llm_json(LABEL_SYSTEM, build_user_prompt(r), max_tokens=600)
                if not isinstance(prop, dict):
                    raise ValueError("ответ не JSON-объект")
                status = validate(prop)
            except Exception as e:
                prop, status = {"_error": str(e)[:200]}, "error"
            rec = {**{k: r[k] for k in KEEP}, "_status": status, "proposal": prop}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {status:<12} {prop.get('cpa_label', '-')}")
            time.sleep(a.sleep)


# ───── 4. консолидация ─────
def card(label, recs):
    p = recs[0]["proposal"]
    return {"label": label, "n": len(recs), **{k: p.get(k, "") for k in FIELDS}}


def pick_def(recs):
    return {k: recs[0]["proposal"].get(k, "") for k in FIELDS}


def cmd_consolidate(a):
    recs = [r for r in load_jsonl(a.proposals) if r["_status"] == "ok"]
    groups = defaultdict(list)
    for r in recs:
        groups[r["proposal"]["cpa_label"]].append(r)
    labels = sorted(groups, key=lambda l: -len(groups[l]))
    print(f"[load] {len(recs)} ok, {len(labels)} уникальных меток")

    cb = Path(a.codebook)
    if cb.exists():
        funcs = json.loads(cb.read_text(encoding="utf-8"))["functions"]
        print(f"[codebook] используем существующий {cb} (правьте его вручную)")
    else:
        cards = [card(l, groups[l]) for l in labels]
        funcs = llm_json(CONS_SYSTEM, json.dumps(cards, ensure_ascii=False, indent=1),
                         max_tokens=6000)["functions"]

    # 1. чистим: только реальные метки, каждая метка — ровно у одной функции
    owner = {}
    for f in funcs:
        f["aliases"] = [x for x in dict.fromkeys(f.get("aliases", []))
                        if x in groups and x not in owner]
        for x in f["aliases"]:
            owner[x] = f["canonical_label"]

    # 2. непокрытые — LLM-назначение на существующие функции или новые
    uncovered = [l for l in labels if l not in owner]
    print(f"[cover] непокрыто: {len(uncovered)}")
    if uncovered:
        existing = [{"canonical_label": f["canonical_label"],
                     "proximal_goal": (f.get("definition") or {}).get("proximal_goal", ""),
                     "state_change": (f.get("definition") or {}).get("state_change", "")}
                    for f in funcs]
        payload = {"existing": existing,
                   "uncovered": [card(l, groups[l]) for l in uncovered]}
        data = llm_json(REPAIR_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=4000)
        canon = {f["canonical_label"]: f for f in funcs}
        for asg in data.get("assignments", []):
            lab, c = asg.get("label"), asg.get("canonical")
            if lab not in groups or lab in owner or not c:
                continue
            if c not in canon:
                canon[c] = {"canonical_label": c, "aliases": [],
                            "definition": pick_def(groups[lab]), "rationale": "new (repair)"}
                funcs.append(canon[c])
            canon[c]["aliases"].append(lab)
            owner[lab] = c

    # 3. остаток — одиночные функции, помеченные для ручной проверки.
    #    Fuzzy-merge по строке намеренно не используется.
    for lab in labels:
        if lab not in owner:
            funcs.append({"canonical_label": lab, "aliases": [lab],
                          "definition": pick_def(groups[lab]),
                          "rationale": "singleton: не покрыто LLM, проверить вручную"})
            owner[lab] = lab
            print(f"  singleton: {lab}")

    cb.write_text(json.dumps({"functions": funcs}, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{"chain_attempt_id": r["chain_attempt_id"], "episode_id": r["episode_id"],
             "harness": r["harness"], "model_identifier": r["model_identifier"],
             "tool_name": r["tool_name"], "obj": r["obj"], "n_actions": r["n_actions"],
             "original_label": r["proposal"]["cpa_label"],
             "canonical_label": owner[r["proposal"]["cpa_label"]],
             "confidence": r["proposal"].get("confidence")} for r in recs]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"[save] {cb} ({len(funcs)} функций), {a.out}")
    print(df["canonical_label"].value_counts().to_string())


# ───── 5. отчёт и согласие ─────
def cmd_report(a):
    recs = load_jsonl(a.proposals)
    ok = [r for r in recs if r["_status"] == "ok"]
    print("статусы:", dict(Counter(r["_status"] for r in recs)))
    print("\nтоп-30 меток:")
    for lab, n in Counter(r["proposal"]["cpa_label"] for r in ok).most_common(30):
        print(f"  {n:4d}  {lab}")
    leak = [r for r in ok
            if set(re.findall(r"[A-Z]+", r["tool_name"].upper()))
            & set(r["proposal"]["cpa_label"].split("_"))]
    print(f"\nметки с именем инструмента внутри: {len(leak)}")
    for r in leak[:10]:
        print(f"  {r['proposal']['cpa_label']}  (tool={r['tool_name']})")
    print("\nпричины abstain:")
    for reason, n in Counter((r["proposal"].get("reason") or "")[:80]
                             for r in recs if r["_status"] == "abstain").most_common(10):
        print(f"  {n:3d}  {reason}")
    conf = [float(r["proposal"].get("confidence") or 0) for r in ok]
    if conf:
        print(f"\nconfidence: mean={sum(conf)/len(conf):.2f} min={min(conf):.2f} max={max(conf):.2f}")


def cmd_agreement(a):
    """Согласие двух прогонов по формуле A из статьи, на уровне (чат, эпизод, метка)."""
    def counts(path):
        return Counter((r["chain_attempt_id"], r["episode_id"], r["proposal"]["cpa_label"])
                       for r in load_jsonl(path) if r["_status"] == "ok")
    ca, cb = counts(a.a), counts(a.b)
    keys = set(ca) | set(cb)
    m = sum(min(ca[k], cb[k]) for k in keys)
    p = sum(max(ca[k] - cb[k], 0) for k in keys)
    q = sum(max(cb[k] - ca[k], 0) for k in keys)
    print(f"прогон A: {sum(ca.values())}, прогон B: {sum(cb.values())}, совпало: {m}")
    print(f"A = {2 * m / (2 * m + p + q):.3f}")


# ───── main ─────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--chains", required=True)
    p.add_argument("--out", default="episodes.csv")
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=500)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("sample")
    p.add_argument("--episodes", default="episodes.csv")
    p.add_argument("--out", default="sample.csv")
    p.add_argument("--pilot", default="pilot.csv")
    p.add_argument("--pilot-per-stratum", type=int, default=5)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("label")
    p.add_argument("--sample", default="sample.csv")
    p.add_argument("--out", default="proposals.jsonl")
    p.add_argument("--limit", type=int)
    p.add_argument("--sleep", type=float, default=0.3)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("consolidate")
    p.add_argument("--proposals", default="proposals.jsonl")
    p.add_argument("--codebook", default="codebook.json")
    p.add_argument("--out", default="labelled.csv")
    p.set_defaults(fn=cmd_consolidate)

    p = sub.add_parser("report")
    p.add_argument("--proposals", default="proposals.jsonl")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("agreement")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.set_defaults(fn=cmd_agreement)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()