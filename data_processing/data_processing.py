# merged_clean_dedup builder + DB weak-type 2x upweight (UPWEIGHT GUARANTEED)
# - normalize
# - clean
# - dedup by hash key
# - (AFTER DEDUP) upweight DB weak types by duplicating 2x
# - shuffle
# - push_to_hub

from datasets import load_dataset, concatenate_datasets, Dataset
import re, json, random, hashlib
from typing import Dict, Any, List

SEED = 3407
random.seed(SEED)

ALF_DATASET = "u-10bei/sft_alfworld_trajectory_dataset_v5"
DB_DATASET  = "u-10bei/dbbench_sft_dataset_react_v4"

OUT_HF_DATASET_ID = "kochan13/mixed-agent-dataset-merged-clean-dedup-dbweak_2x_2"  # ★好きに変更

# ★ upweight 強度（2なら「弱点を2回追加」＝合計で弱点が3倍相当）
DB_WEAK_ADD_TIMES = 2  # 0ならupweightなし / 1なら+1回 / 2なら+2回

# -------------------------
# common utils
# -------------------------
def norm_ws(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -------------------------
# ALF normalize
# -------------------------
TAG_ALF = "[TASK=ALF]"
ALF_VERB = re.compile(
    r"^(go to|examine|look|open|close|take|pickup|pick up|put|drop|use|clean|heat|cool|slice|turn on|turn off)\b",
    re.I
)

def alf_extract_action(text: str) -> str:
    if not isinstance(text, str): return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        m = re.match(r"^(Act|Action)\s*:\s*(.+)$", ln, flags=re.I)
        if m: return m.group(2).strip()
    return lines[-1].strip() if lines else ""

NON_ACTION_PREFIX = re.compile(
    r"^(observation|admissible actions?|you (see|are)|task is to)\b",
    re.I
)

def alf_clean_action(a: str) -> str:
    if not isinstance(a, str):
        return ""
    a = norm_ws(a.strip())

    if NON_ACTION_PREFIX.match(a):
        return ""

    # コロン以降に説明が付くケースは前半のみ採用
    if ":" in a:
        a = a.split(":", 1)[0].strip()

    a = re.sub(r"[。\.]+$", "", a).strip()
    a = re.sub(r"\bin on\b", "in/on", a, flags=re.I)

    if not ALF_VERB.match(a):
        return ""
    return a

def strip_first_task_tag(text: str, tag: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lstrip()
    if t.startswith(tag):
        lines = t.splitlines()
        if lines and lines[0].strip() == tag:
            return "\n".join(lines[1:]).lstrip()
    return text

def normalize_alf(ex: Dict[str, Any]) -> Dict[str, Any]:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list):
        ex["messages"] = []
        return ex

    new = []
    first_user = True

    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")

        if role == "system":
            continue

        if role == "user":
            c = norm_ws(strip_first_task_tag(str(content), TAG_ALF))
            if first_user:
                c = f"{TAG_ALF}\n{c}".strip()
                first_user = False
            new.append({"role": "user", "content": c})
            continue

        if role == "assistant":
            a = alf_clean_action(alf_extract_action(str(content)))
            if not a:
                new.append({"role": "assistant", "content": ""})
            else:
                new.append({"role": "assistant", "content": f"Act: {a}"})
            continue

        new.append({"role": role, "content": norm_ws(str(content))})

    ex["messages"] = new
    return ex

def filter_alf(ex: Dict[str, Any]) -> bool:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        return False

    # assistant空は落とす + Act: 形式＆動詞チェック
    for m in msgs:
        if m.get("role") == "assistant":
            c = m.get("content","")
            if not isinstance(c, str) or c.strip() == "":
                return False
            if not c.strip().lower().startswith("act:"):
                return False
            act = c.split(":", 1)[-1].strip()
            if not alf_clean_action(act):
                return False
    return True

# -------------------------
# DB normalize
# -------------------------
TAG_DB = "[TASK=DB]"

def one_line_sql(sql: str) -> str:
    if not isinstance(sql, str): return ""
    s = re.sub(r"\s+", " ", sql).strip()
    s = s[:-1].strip() if s.endswith(";") else s
    return s

MYSQL_OUTPUT_PAT = re.compile(
    r"^\s*(Query OK|ERROR|Empty set|\(\d+ rows?\)|\d+ rows? in set|"
    r"Records:\s*\d+|Rows matched:\s*\d+|Changed:\s*\d+|Warnings:\s*\d+)",
    re.IGNORECASE
)

DBBENCH_INSTR_HINT = re.compile(
    r"(I will ask you a question|your operation should be like this|You MUST put SQL|"
    r"Every time you can only execute one SQL statement|Final Answer:|Action:\s*Operation)",
    re.IGNORECASE
)

def pick_db_question(msgs: List[Dict[str, Any]]) -> str:
    user_turns = [
        m.get("content", "")
        for m in msgs
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    for raw in reversed(user_turns):
        c = norm_ws(raw)
        if not c:
            continue
        if MYSQL_OUTPUT_PAT.match(c):
            continue
        if DBBENCH_INSTR_HINT.search(c):
            continue
        return c
    return norm_ws(user_turns[-1]) if user_turns else ""

def normalize_db(ex: Dict[str, Any]) -> Dict[str, Any]:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list):
        ex["messages"] = []
        return ex

    question = pick_db_question(msgs)
    if question and not question.startswith("[TASK="):
        question = f"{TAG_DB}\n{question}"

    md = ex.get("metadata", {}) or {}
    sql_raw = md.get("sql", "")
    sql = one_line_sql(sql_raw if isinstance(sql_raw, str) else "")

    ex["messages"] = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"Final: {sql}" if sql else ""},
    ]
    return ex

def filter_db(ex: Dict[str, Any]) -> bool:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False

    u = msgs[0].get("content", "")
    a = msgs[-1].get("content", "")

    if not (isinstance(u, str) and u.startswith(TAG_DB)):
        return False
    if not (isinstance(a, str) and a.strip().lower().startswith("final:")):
        return False

    sql = a.split(":", 1)[-1].strip()
    return sql != ""

# ★ DB弱点判定（metadata.type を見る）
def is_db_weak_type(ex: Dict[str, Any]) -> bool:
    md = ex.get("metadata", {}) or {}
    t = md.get("type", "") or ""
    if not isinstance(t, str):
        return False
    tl = t.lower()
    return ("aggregation" in tl) or ("counting" in tl) or ("insert" in tl) or ("update" in tl)

# -------------------------
# dedup key
# -------------------------
def canonicalize_for_key(msgs: List[Dict[str,str]]) -> str:
    parts = []
    for m in msgs:
        role = m.get("role", "")
        content = norm_ws(m.get("content", ""))
        parts.append(f"{role}::{content}")
    return "\n".join(parts)

def add_dedup_key(ex: Dict[str, Any]) -> Dict[str, Any]:
    msgs = ex.get("messages", [])
    key_src = canonicalize_for_key(msgs) if isinstance(msgs, list) else ""
    ex["dedup_key"] = sha1_text(key_src)
    return ex

def dedup_dataset(ds: Dataset, key_col: str = "dedup_key") -> Dataset:
    keys = ds[key_col]
    seen = set()
    keep_idx = []
    for i, k in enumerate(keys):
        if k in seen:
            continue
        seen.add(k)
        keep_idx.append(i)
    return ds.select(keep_idx)

# -------------------------
# run
# -------------------------
print("[INFO] load ALF/DB ...")
alf = load_dataset(ALF_DATASET, split="train")
db  = load_dataset(DB_DATASET,  split="train")

print("[INFO] normalize ALF ...")
alf2 = alf.map(normalize_alf, desc="normalize ALF")
alf2 = alf2.filter(filter_alf, desc="filter ALF")

print("[INFO] normalize DB ...")
db2 = db.map(normalize_db, desc="normalize DB")
db2 = db2.filter(filter_db, desc="filter DB")

print("[INFO] merge (pre-dedup) ...")
merged = concatenate_datasets([alf2, db2]).shuffle(seed=SEED)

print("[INFO] add dedup key ...")
merged = merged.map(add_dedup_key, desc="add dedup_key")

print("[INFO] dedup ...")
before = len(merged)
merged = dedup_dataset(merged, key_col="dedup_key")
after = len(merged)
print(f"[INFO] dedup: {before} -> {after} (removed {before-after})")

# =========================
# ★ ここが“最小差分の本体”：
#    dedup後に DB弱点を追加する（= upweightが確実に残る）
# =========================
def is_db_row(ex: Dict[str, Any]) -> bool:
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or not msgs:
        return False
    u = msgs[0].get("content", "")
    return isinstance(u, str) and u.startswith(TAG_DB)

print("[INFO] split ALF/DB after dedup ...")
db_dedup  = merged.filter(is_db_row, desc="split DB after dedup")
alf_dedup = merged.filter(lambda ex: not is_db_row(ex), desc="split ALF after dedup")
print(f"[INFO] after dedup -> ALF: {len(alf_dedup)}, DB: {len(db_dedup)}")

if DB_WEAK_ADD_TIMES > 0:
    print("[INFO] upweight DB weak types AFTER dedup ...")
    db_weak = db_dedup.filter(is_db_weak_type, desc="select DB weak types (deduped)")
    print(f"[INFO] DB weak rows: {len(db_weak)} / {len(db_dedup)}")

    extra = [db_weak] * DB_WEAK_ADD_TIMES
    db_final = concatenate_datasets([db_dedup] + extra)
    print(f"[INFO] DB after upweighting: {len(db_final)} (added {len(db_weak) * DB_WEAK_ADD_TIMES})")
else:
    db_final = db_dedup
    print("[INFO] DB upweighting skipped")

print("[INFO] final merge & shuffle ...")
final_ds = concatenate_datasets([alf_dedup, db_final]).shuffle(seed=SEED)

# push（Hubに要る列だけ）
if OUT_HF_DATASET_ID:
    keep_cols = ["messages"]
    if "metadata" in final_ds.column_names:
        keep_cols.append("metadata")
    if "id" in final_ds.column_names:
        keep_cols.append("id")
    if "tools" in final_ds.column_names:
        keep_cols.append("tools")

    push_ds = final_ds.select_columns(keep_cols)

    # dedup_key は最終成果物には不要
    if "dedup_key" in push_ds.column_names:
        push_ds = push_ds.remove_columns(["dedup_key"])

    print("[INFO] push_to_hub:", OUT_HF_DATASET_ID)
    push_ds.push_to_hub(OUT_HF_DATASET_ID)

print("[DONE]")
print("[SAMPLE]\n", json.dumps(final_ds[0]["messages"], ensure_ascii=False, indent=2))
