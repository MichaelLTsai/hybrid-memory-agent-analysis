"""
LoCoMo P4 / P5 probes: value-level retrieval and reasoning attribution.

LoCoMo already ships turn-level recall@k and ndcg@k (matching dia_id), but those
ask whether the correct turn was retrieved, not whether that turn's content
survived into the context. Extraction-based backends make the two diverge: the
right source is retrieved, yet the value was already crushed during extraction.
The gap between the two numbers is exactly the amount of summary failure being
mis-recorded as retrieval success.

  P4  does the retrieved context suffice to answer (about 20 items, handed to the
      LLM as one batch, exhaustive)
  P5  passed P4 yet answered wrong, i.e. a reader problem for which the memory
      system bears no responsibility

P1 (whether the fact was remembered at all) lives in extraction_locomo.py, which
compares item by item against observations as golden memories, and is not
duplicated here. Attribution needs both read together:
    P4 fail and present in that session's memories -> RETRIEVAL
    P4 fail and absent                             -> SUMMARY

cat5 (adversarial) is an exception: its answer field is a lure, and the correct
behavior is to **avoid** it rather than produce it, so P4 and P5 are undefined
and only P5b_avoided is scored (did it dodge the trap).

Anything unadjudicable returns None, is excluded from the denominator, and its
cause is recorded in judge_failures.

Usage:
  python probe_locomo.py --run mem0-v1_31b
"""

import os
import sys
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

from llms import llm_request_for_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

_FAILS = Counter()
_LOCK = threading.Lock()
_WARNED = set()


def _note_failure(kind, exc):
    with _LOCK:
        _FAILS[kind] += 1
        first = kind not in _WARNED
        _WARNED.add(kind)
    if first:
        print(f"[probe_locomo] {kind} judge failure (further ones of this kind are not printed): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _block(mems):
    return "\n".join(f"- {m}" for m in mems) if mems else "(none)"


def judge_sufficient(question, answer, evidence, memories):
    if not memories:
        return False
    prompt = SUFFICIENCY_PROMPT.format(question=question, answer=answer,
                                       evidence=_block(evidence), memories=_block(memories))
    try:
        v = llm_request_for_json(prompt).get("sufficient", None)
        if not isinstance(v, bool):
            raise ValueError(f"'sufficient' is not a bool: {v!r}")
        return v
    except Exception as e:
        _note_failure("sufficiency", e)
        return None


def _ev_ids(qa):
    ev = qa.get("evidence")
    if isinstance(ev, str):
        try:
            ev = eval(ev)
        except Exception:
            ev = [ev]
    out = []
    for x in (ev or []):
        if isinstance(x, str) and x.startswith("D"):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            out += [y for y in x if isinstance(y, str) and y.startswith("D")]
    return out


def _sess_of(dia_id):
    import re
    m = re.match(r"^D(\d+):", str(dia_id or ""))
    return int(m.group(1)) if m else None


def attribute_one(qa, turn_text, dump_by_sess):
    cat = str(qa.get("category"))
    ev_ids = _ev_ids(qa)
    evidence = [turn_text.get(e, "") for e in ev_ids if turn_text.get(e)]
    # P4 is uniformly defined as what the reader can see at answering time. Letta
    # is an agent and sees the message history (agent_context) on top of the memory
    # snapshot; pipeline-style backends have no such field and fall back to retrieved.
    src = qa.get("agent_context") or qa.get("retrieved") or []
    retrieved = [r.get("text") if isinstance(r, dict) else str(r) for r in src]
    correct = bool(qa.get("is_correct"))

    rec = {"question": qa.get("question"), "category": cat, "is_correct": correct,
           "n_retrieved": len(retrieved), "evidence_ids": ev_ids}

    if cat == "5":
        # Lure questions: the correct behavior is to dodge the trap, so there is
        # no correct answer that ought to have been retrieved
        rec.update({"P4": None, "verdict": "OK" if correct else "P5b_FAIL",
                    "P5b_avoided": correct})
        return rec

    gold = qa.get("answer")
    # 1. P4: does the retrieved context suffice to answer
    p4 = judge_sufficient(qa.get("question"), gold, evidence, retrieved)
    rec["P4"] = p4
    if p4 is None:
        rec["verdict"] = "UNKNOWN"
        return rec
    if p4:
        rec["verdict"] = "OK" if correct else "REASONING"
        return rec

    # 2. Not retrieved -> is it present in the memories the evidence session produced
    if dump_by_sess is None:
        rec["verdict"] = "NO_DUMP"
        return rec
    sessions = {s for s in (_sess_of(e) for e in ev_ids) if s is not None}
    scoped = [m for s in sessions for m in dump_by_sess.get(s, [])]
    rec["n_scoped"] = len(scoped)
    if not scoped:
        rec["verdict"] = "NO_WRITE"
        return rec
    p1 = judge_sufficient(qa.get("question"), gold, evidence, scoped)
    rec["P1_scoped"] = p1
    rec["verdict"] = ("UNKNOWN" if p1 is None else ("RETRIEVAL" if p1 else "SUMMARY"))
    return rec


def run(run_name, frame=None, max_workers=8, write=True):
    frame = frame or run_name.split("-")[0]
    result_dir = os.path.join(BASE_DIR, "results", run_name)
    detail_f = os.path.join(result_dir, f"{frame}_locomo_detail.jsonl")
    results_f = os.path.join(result_dir, f"{frame}_locomo_results.jsonl")
    scores_f = os.path.join(result_dir, f"{frame}_locomo_scores.json")
    out_f = os.path.join(result_dir, f"{frame}_locomo_probe_detail.jsonl")

    with open(results_f, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    # is_correct only exists in the detail file (produced by evaluation)
    judged = {}
    if os.path.exists(detail_f):
        with open(detail_f, encoding="utf-8") as f:
            for l in f:
                d = json.loads(l)
                judged[(d.get("question"), str(d.get("category")))] = d.get("is_correct")

    with open(os.path.join(BASE_DIR, "data", "locomo10.json"), encoding="utf-8") as f:
        raws = {s["sample_id"]: s for s in json.load(f)}

    with _LOCK:
        _FAILS.clear(); _WARNED.clear()

    recs = []
    for s in samples:
        raw = raws.get(s.get("sample_id"))
        turn_text = {}
        if raw:
            for k, v in raw["conversation"].items():
                if k.startswith("session_") and isinstance(v, list):
                    for t in v:
                        turn_text[t.get("dia_id")] = f'{t.get("speaker")}: {t.get("text","")}'
        dump = s.get("memory_dump")
        by_sess = None
        if dump:
            by_sess = defaultdict(list)
            for m in dump:
                sid = _sess_of(m.get("dia_id"))
                if sid is not None:
                    by_sess[sid].append(m.get("text") or "")
        qas = s.get("qa") or []
        for qa in qas:
            qa["is_correct"] = judged.get((qa.get("question"), str(qa.get("category"))),
                                          qa.get("is_correct"))
        print(f"Probing {s.get('sample_id')}: {len(qas)} questions ...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(attribute_one, qa, turn_text, by_sess): qa for qa in qas}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except Exception as e:
                    _note_failure("worker", e)
                    r = {"question": futs[fut].get("question"), "verdict": "UNKNOWN"}
                r["sample_id"] = s.get("sample_id")
                recs.append(r)

    summary = _aggregate(recs)
    if write:
        with open(out_f, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scores = {}
        if os.path.exists(scores_f):
            with open(scores_f, encoding="utf-8") as f:
                scores = json.load(f)
        scores["probe"] = summary
        with open(scores_f, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"✅ wrote the \"probe\" block of {scores_f}; per-question -> {out_f}")

    print("\n=== LoCoMo probe ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"},
                     indent=2, ensure_ascii=False))
    return summary


def _aggregate(recs):
    v = Counter(r.get("verdict") for r in recs)
    by_cat = defaultdict(Counter)
    for r in recs:
        by_cat[r.get("category", "?")][r.get("verdict")] += 1

    ans = [r for r in recs if r.get("category") != "5"]
    adv = [r for r in recs if r.get("category") == "5"]
    p4_dec = [r for r in ans if isinstance(r.get("P4"), bool)]
    p1_dec = [r for r in ans if isinstance(r.get("P1_scoped"), bool)]
    p4_ok = [r for r in p4_dec if r["P4"]]

    total = len(recs)
    unknown = v.get("UNKNOWN", 0) + v.get("NO_DUMP", 0)
    return {
        "n_total": total,
        "n_answerable": len(ans),
        "n_adversarial": len(adv),
        "P4_sufficient": (sum(1 for r in p4_dec if r["P4"]) / len(p4_dec)) if p4_dec else None,
        "P4_n": len(p4_dec),
        "P1_scoped_sufficient": (sum(1 for r in p1_dec if r["P1_scoped"]) / len(p1_dec)) if p1_dec else None,
        "P1_scoped_n": len(p1_dec),
        # P5: passed P4 yet still wrong; same definition as HaluMem and LongMemEval.
        "P5_fail_given_P4": (sum(1 for r in p4_ok if not r.get("is_correct")) / len(p4_ok))
                            if p4_ok else None,
        "P5_n": len(p4_ok),
        "P5b_avoid_rate": (sum(1 for r in adv if r.get("P5b_avoided")) / len(adv)) if adv else None,
        "P5b_n": len(adv),
        "verdicts": dict(v),
        "attribution": {
            k: v.get(k, 0) / (total - unknown) if (total - unknown) else 0
            for k in ["OK", "SUMMARY", "RETRIEVAL", "REASONING", "NO_WRITE", "P5b_FAIL"]
        },
        "unknown_ratio": unknown / total if total else 0,
        "judge_failures": dict(_FAILS),
        "by_category": {k: dict(c) for k, c in by_cat.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    run(args.run, frame=args.frame, max_workers=args.max_workers)
