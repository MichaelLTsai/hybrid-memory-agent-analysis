#!/usr/bin/env python
"""HaluMem 的 P1 / P4 / P5：改用與 LongMemEval、LoCoMo 相同的充分性判準。

原本 HaluMem 只有 P4，且沿用官方的 RETRIEVAL_PRESENCE_PROMPT（逐條 evidence 判定
在不在）。那與 LongMemEval / LoCoMo 的 SUFFICIENCY_PROMPT（整批判定夠不夠）是兩把
不同的尺，導致：

  · HaluMem 的 P4 無法與其他基準並列；
  · 若 P1 改用充分性判準而 P4 不改，同一基準內 P1/P4 尺度不一，
    「RETRIEVAL = P4 失敗且 P1 通過」這個推論就不成立。

因此本程式對 HaluMem 重新判定 P4，並補上原本缺少的 P1，兩者共用同一支
SUFFICIENCY_PROMPT，與另外兩個基準逐字元相同。

範圍限縮（P1 用）
  HaluMem 的題目自帶 ssession_id，但實測顯示約三成題目的 evidence 並不在該
  session（跨會話題）。因此範圍鍵不取題目的 session，而是：

      evidence 的 memory_content
        → 接合到帶有 ssession_id 的 golden memory 記錄
        → 取得該 evidence 真正的來源 session

  接合表涵蓋所有帶 memory_content 或 reference_memory_content 的非 QA 記錄，
  實測接合率 100%。

歸因（與 LongMemEval 同義，新定義）
  P4 通過 且答對            → OK
  P4 通過 且答錯            → REASONING
  P4 失敗 且答對            → NOT_COUNTED   （記憶失敗但矇對，不計為失敗）
  P4 失敗 且答錯 且 S 為空  → SUMMARY
  P4 失敗 且答錯 且 P1 通過 → RETRIEVAL
  P4 失敗 且答錯 且 P1 失敗 → SUMMARY

  無 evidence 的題目（Memory Boundary）為拒答題，不進分母，只計 P5b。
  裁判判不出來回 None，排除於分母並記入 judge_failures。

用法：
  ../venv_memos/bin/python probe_halumem_unified.py --run amem-amem_cost_u2
  ../venv_memos/bin/python probe_halumem_unified.py --all --workers 3
"""

import argparse
import glob
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

from llms import llm_request_for_json  # noqa: E402

# 與 longmemeval_experiment/probe_longmem.py、locomo_experiment/probe_locomo.py 逐字元相同
SUFFICIENCY_PROMPT = """You are auditing an AI memory system. Decide whether a set of memories contains the information needed to answer a question correctly.

# Question
{question}

# Reference answer (the correct answer)
{answer}

# Evidence from the original conversation (what the system should have captured)
{evidence}

# Memories to check
{memories}

Do the Memories contain the information needed to produce the Reference answer?

- Answer "true" only if the needed facts are present (rewording is fine; for answers that must be
  computed — a count, a date difference — the raw components must be present).
- A merely related or topically-similar memory does NOT count.
- Ignore whether the memories are well-written; only their informational content matters.

Return strictly this JSON:
```json
{{"sufficient": true_or_false}}
```
"""

_LOCK = threading.Lock()
_FAILS = Counter()


def _note_failure(kind, exc):
    with _LOCK:
        _FAILS[f"{kind}:{type(exc).__name__}"] += 1


def _block(items):
    return "\n".join(f"- {x}" for x in items) if items else "(none)"


def judge_sufficient(question, answer, evidence, memories):
    """這批記憶夠不夠回答。True / False，判不出來回 None。"""
    if not memories:
        return False                       # 空集合是有效判定，不必問裁判
    prompt = SUFFICIENCY_PROMPT.format(
        question=question, answer=answer,
        evidence=_block(evidence), memories=_block(memories))
    try:
        v = llm_request_for_json(prompt).get("sufficient", None)
        if not isinstance(v, bool):
            raise ValueError(f"'sufficient' is not a bool: {v!r}")
        return v
    except Exception as e:
        _note_failure("sufficiency", e)
        return None


# ── 資料整理 ────────────────────────────────────────────────────────────────

def load_run(run_dir):
    """回傳 (qa_rows, store_by_session, golden_session_of)。"""
    frame = os.path.basename(run_dir).split("-")[0]
    path = os.path.join(run_dir, f"{frame}_eval_detail.jsonl")
    if not os.path.exists(path):
        cands = glob.glob(os.path.join(run_dir, "*_eval_detail.jsonl"))
        if not cands:
            return None, None, None
        path = cands[0]

    qa, store, golden = [], defaultdict(list), {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "question" in r:
                qa.append(r)
                continue
            if "is_included_in_golden_memories" in r:
                txt = (r.get("memory_content") or "").strip()
                if txt:
                    store[r.get("ssession_id")].append(txt)
                continue
            # 其餘皆為 golden memory 側的記錄，用來建立 evidence → session 的接合表
            for key in ("memory_content", "reference_memory_content"):
                v = (r.get(key) or "").strip()
                if v:
                    golden.setdefault(v, r.get("ssession_id"))
    return qa, store, golden


def evidence_texts(q):
    return [(e.get("memory_content") or "").strip()
            for e in (q.get("evidence") or []) if (e.get("memory_content") or "").strip()]


def retrieved_memories(q):
    """把 context 這個文字塊拆成條目；拆不出來就整塊當成一條。"""
    ctx = q.get("context") or ""
    if not ctx.strip():
        return []
    lines = [l.strip(' \t",') for l in ctx.splitlines()]
    # 只濾掉 JSON 陣列的裸括號與標頭；M4 的條目以「[CURRENT MEMORY | ...]」開頭，
    # 若以 startswith("[") 濾除會把整批記憶清空，使裁判只看到提示文字。
    items = [l for l in lines
             if l and l not in ("[", "]") and not l.startswith("Memories for")]
    return items or [ctx.strip()]


def scoped_memories(q, store, golden):
    """evidence 的來源 session 所產出的記憶。回傳 (memories, sessions, matched)。"""
    sids, matched = set(), 0
    for t in evidence_texts(q):
        if t in golden:
            sids.add(golden[t])
            matched += 1
    mems = []
    for sid in sids:
        mems.extend(store.get(sid, []))
    return mems, sorted(s for s in sids if s is not None), matched


# ── 單題歸因 ────────────────────────────────────────────────────────────────

def attribute_one(q, store, golden):
    ev = evidence_texts(q)
    question = q.get("question", "")
    answer = q.get("answer", "")
    correct = (q.get("result_type") == "Correct")

    rec = {
        "question": question[:300],
        "question_type": q.get("question_type"),
        "ssession_id": q.get("ssession_id"),
        "is_correct": correct,
        "n_evidence": len(ev),
        "P4": None, "P1": None, "verdict": None,
        "n_retrieved": 0, "n_scoped": 0, "scope_sessions": [],
    }

    if not ev:                                   # Memory Boundary：拒答題
        rec["verdict"] = "P5b_OK" if correct else "P5b_FAIL"
        return rec

    ctx = retrieved_memories(q)
    rec["n_retrieved"] = len(ctx)
    p4 = judge_sufficient(question, answer, ev, ctx)
    rec["P4"] = p4
    if p4 is None:
        rec["verdict"] = "UNADJUDICATED"
        return rec

    if p4:
        rec["verdict"] = "OK" if correct else "REASONING"
        return rec

    if correct:                                  # 階段失敗但答對，不計為失敗
        rec["verdict"] = "NOT_COUNTED"
        return rec

    mems, sids, matched = scoped_memories(q, store, golden)
    rec["n_scoped"] = len(mems)
    rec["scope_sessions"] = sids
    rec["evidence_matched"] = matched
    if not mems:
        rec["P1"] = False
        rec["verdict"] = "SUMMARY"                # 含 NO_WRITE
        return rec

    p1 = judge_sufficient(question, answer, ev, mems)
    rec["P1"] = p1
    if p1 is None:
        rec["verdict"] = "UNADJUDICATED"
        return rec
    rec["verdict"] = "RETRIEVAL" if p1 else "SUMMARY"
    return rec


# ── 執行 ────────────────────────────────────────────────────────────────────

def run_one(run_dir, workers=3, write=True):
    name = os.path.basename(run_dir)
    qa, store, golden = load_run(run_dir)
    if qa is None:
        print(f"  {name}: 找不到 eval_detail，略過")
        return None

    with _LOCK:
        _FAILS.clear()

    recs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(attribute_one, q, store, golden): q for q in qa}
        for fut in as_completed(futs):
            try:
                recs.append(fut.result())
            except Exception as e:
                _note_failure("worker", e)
                recs.append({"verdict": "UNADJUDICATED"})

    v = Counter(r["verdict"] for r in recs)
    n = sum(v[k] for k in ("OK", "REASONING", "RETRIEVAL", "SUMMARY", "NOT_COUNTED"))
    out = {
        "run": name,
        "n": n,
        "P1_fail": round(v["SUMMARY"] / n, 4) if n else None,
        "P4_fail": round(v["RETRIEVAL"] / n, 4) if n else None,
        "P5_fail": round(v["REASONING"] / n, 4) if n else None,
        "not_counted": v["NOT_COUNTED"],
        "abstain_n": v["P5b_OK"] + v["P5b_FAIL"],
        "unadjudicated": v["UNADJUDICATED"],
        "verdicts": dict(v),
        "judge_failures": dict(_FAILS),
    }
    err = (out["P1_fail"] or 0) + (out["P4_fail"] or 0) + (out["P5_fail"] or 0)
    out["attributable_error_rate"] = round(err, 4)

    if write:
        frame = name.split("-")[0]
        p = os.path.join(run_dir, f"{frame}_probe_unified.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(run_dir, f"{frame}_probe_unified_scores.json"),
                  "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"  {name:38s} N={n:4d}  P1={out['P1_fail']}  P4={out['P4_fail']}  "
          f"P5={out['P5_fail']}  錯誤率={out['attributable_error_rate']}  "
          f"矇對={out['not_counted']}  排除={out['unadjudicated']}")
    if _FAILS:
        print(f"      judge_failures: {dict(_FAILS)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="單一 run 名稱，例如 amem-amem_cost_u2")
    ap.add_argument("--all", action="store_true", help="跑所有有 eval_detail 的 run")
    ap.add_argument("--workers", type=int, default=3, help="並行度，預設 3")
    args = ap.parse_args()

    if args.run:
        dirs = [os.path.join(BASE, "results", args.run)]
    elif args.all:
        dirs = sorted({os.path.dirname(p)
                       for p in glob.glob(os.path.join(BASE, "results", "*", "*_eval_detail.jsonl"))})
    else:
        ap.error("需要 --run 或 --all")

    print(f"共 {len(dirs)} 個 run，並行度 {args.workers}\n")
    results = []
    for d in dirs:
        r = run_one(d, workers=args.workers)
        if r:
            results.append(r)

    with open(os.path.join(BASE, "probe_unified_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已寫入 {os.path.join(BASE, 'probe_unified_summary.json')}")


if __name__ == "__main__":
    main()
