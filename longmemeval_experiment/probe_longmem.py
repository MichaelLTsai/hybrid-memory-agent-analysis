"""
LongMemEval P1 / P4 / P5 probes: attribute wrong answers to Summary / Retrieval / Reasoning.

LongMemEval has no golden memory (it labels only which sentence contains the
answer, plus the gold answer), so P1 is not an item-by-item comparison of
extracted memories against reference memories. It degenerates into an existence
check: does the information this question needs live in the memories that session
produced?

The same criterion is applied to two containers, and the order matters:
    P4  checks the retrieved context (about 20 items, handed to the LLM as one
        batch; exhaustive and reliable)
    P1  checks the full store, but **narrowed to that session's memories via
        answer_session_ids** (exhaustive, so "not found" means definitively
        absent rather than merely not surfaced by search)

Attribution takes the first stage that broke:
    P4 pass and answer correct        -> OK
    P4 pass and answer wrong          -> REASONING  (the material was right there)
    P4 fail and P1 pass               -> RETRIEVAL  (stored but not retrieved)
    P4 fail, P1 fail, session has memories -> SUMMARY (lost during extraction)
    P4 fail and the session produced 0 memories -> NO_WRITE (no write was triggered)

The criterion is "do these memories suffice to answer the question" rather than
"is the gold answer string present", because the derivational subsets
(multi-session counting, temporal day arithmetic) have answers whose literal text
never appears, so a string check would misjudge every one of them.

Abstention questions (question_id ending in _abs) have no evidence sentences, so
nothing should be remembered, P1 and P4 are undefined, and only P5b is scored
(did it say it did not know when it should have).

Anything unadjudicable returns None, is excluded from the denominator, and its
cause is recorded in judge_failures. No default is substituted, since that would
let LLM flakiness masquerade as a defect in the memory system.

Usage:
  python probe_longmem.py --run mem0-v1_31b
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
        print(f"[probe_longmem] {kind} judge failure (further ones of this kind are not printed): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def _block(mems):
    return "\n".join(f"- {m}" for m in mems) if mems else "(none)"


def judge_sufficient(question, answer, evidence, memories):
    """Whether these memories suffice to answer the question. True/False, or None if unadjudicable."""
    if not memories:
        return False                      # an empty set is a valid verdict; no need to ask the LLM
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


def evidence_turns(q, dataset_index):
    """Pull the sentences flagged has_answer for this question, as context for the judge."""
    raw = dataset_index.get(q.get("question_id"))
    if not raw:
        return []
    out = []
    for sid, sess in zip(raw.get("haystack_session_ids", []), raw.get("haystack_sessions", [])):
        if sid in set(raw.get("answer_session_ids", [])):
            for t in sess:
                if t.get("has_answer"):
                    out.append(f'[{t.get("role")}] {t.get("content","")}')
    return out


def scoped_memories(q):
    """Memories produced by this question's answer session. Returns (memories, has_provenance)."""
    allm = q.get("all_memories")
    if not allm:
        return [], False
    gold = set(q.get("answer_session_ids") or [])
    scoped, has_prov = [], False
    for m in allm:
        if isinstance(m, dict):
            sid = m.get("session_id")
            txt = m.get("text") or ""
            if sid is not None:
                has_prov = True
                if sid in gold:
                    scoped.append(txt)
        # Backends without provenance such as letta: memories are plain strings
    if has_prov:
        return scoped, True
    # No provenance -> fall back to the whole store (a wider scope, so "not found"
    # is less certain; this is flagged)
    return [(m.get("text") if isinstance(m, dict) else str(m)) for m in allm], False


def attribute_one(q, dataset_index):
    """Return the probe result dict for one question."""
    qid = str(q.get("question_id", ""))
    is_abs = qid.endswith("_abs")
    correct = bool(q.get("is_correct"))
    ev = evidence_turns(q, dataset_index)
    retrieved = [r.get("text") if isinstance(r, dict) else str(r)
                 for r in (q.get("retrieved") or [])]

    rec = {"question_id": qid, "question_type": q.get("question_type"),
           "is_correct": correct, "is_abstention": is_abs,
           "n_retrieved": len(retrieved)}

    if is_abs:
        # Nothing should have been remembered; only whether the reader abstained
        rec.update({"P1": None, "P4": None, "P5b_abstained": correct,
                    "verdict": "OK" if correct else "P5b_FAIL"})
        return rec

    scoped, has_prov = scoped_memories(q)
    rec["scope"] = "session" if has_prov else "global"
    rec["n_scoped"] = len(scoped)

    # 1. P4: does the retrieved context suffice (small, exhaustive, reliable)
    p4 = judge_sufficient(q.get("question"), q.get("answer"), ev, retrieved)
    rec["P4"] = p4
    if p4 is None:
        rec["verdict"] = "UNKNOWN"
        return rec
    if p4:
        rec["P1"] = None                      # retrieved, so no need to check the store
        rec["verdict"] = "OK" if correct else "REASONING"
        return rec

    # 2. Not retrieved -> did that session write anything at all
    if q.get("all_memories") is None:
        rec.update({"P1": None, "verdict": "NO_DUMP"})   # this run did not dump the store
        return rec
    if not scoped:
        rec.update({"P1": False, "verdict": "NO_WRITE"}) # the session produced no memories
        return rec

    # 3. P1: does the store (narrowed to that session) suffice to answer
    p1 = judge_sufficient(q.get("question"), q.get("answer"), ev, scoped)
    rec["P1"] = p1
    if p1 is None:
        rec["verdict"] = "UNKNOWN"
    else:
        rec["verdict"] = "RETRIEVAL" if p1 else "SUMMARY"
    return rec


def run(run_name, frame=None, max_workers=8, write=True):
    frame = frame or run_name.split("-")[0]
    result_dir = os.path.join(BASE_DIR, "results", run_name)
    detail_f = os.path.join(result_dir, f"{frame}_lme_detail.jsonl")
    results_f = os.path.join(result_dir, f"{frame}_lme_results.jsonl")
    scores_f = os.path.join(result_dir, f"{frame}_lme_scores.json")
    out_f = os.path.join(result_dir, f"{frame}_lme_probe_detail.jsonl")

    src = detail_f if os.path.exists(detail_f) else results_f
    with open(src, encoding="utf-8") as f:
        qs = [json.loads(l) for l in f if l.strip()]
    # The detail file may omit all_memories; recover it from results
    if src == detail_f and os.path.exists(results_f):
        with open(results_f, encoding="utf-8") as f:
            byid = {json.loads(l).get("question_id"): json.loads(l) for l in f if l.strip()}
        for q in qs:
            if q.get("all_memories") is None:
                q["all_memories"] = (byid.get(q.get("question_id")) or {}).get("all_memories")

    with open(os.path.join(BASE_DIR, "data", "longmemeval_s.json"), encoding="utf-8") as f:
        dataset_index = {x["question_id"]: x for x in json.load(f)}

    with _LOCK:
        _FAILS.clear(); _WARNED.clear()

    print(f"Probing {len(qs)} questions ...")
    recs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(attribute_one, q, dataset_index): q for q in qs}
        for fut in as_completed(futs):
            try:
                recs.append(fut.result())
            except Exception as e:
                _note_failure("worker", e)
                recs.append({"question_id": futs[fut].get("question_id"), "verdict": "UNKNOWN"})

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

    print("\n=== LongMemEval probe ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_question_type"},
                     indent=2, ensure_ascii=False))
    return summary


def _aggregate(recs):
    v = Counter(r.get("verdict") for r in recs)
    by_type = defaultdict(Counter)
    for r in recs:
        by_type[r.get("question_type", "?")][r.get("verdict")] += 1

    ans = [r for r in recs if not r.get("is_abstention")]
    # The P1 and P4 denominators hold only genuinely adjudicated questions
    p4_dec = [r for r in ans if isinstance(r.get("P4"), bool)]
    p1_dec = [r for r in ans if isinstance(r.get("P1"), bool)]
    p4_ok = [r for r in p4_dec if r["P4"]]
    abst = [r for r in recs if r.get("is_abstention")]

    total = len(recs)
    unknown = v.get("UNKNOWN", 0) + v.get("NO_DUMP", 0)
    return {
        "n_total": total,
        "n_answerable": len(ans),
        "n_abstention": len(abst),
        # P4: does the retrieved context suffice
        "P4_sufficient": (sum(1 for r in p4_dec if r["P4"]) / len(p4_dec)) if p4_dec else None,
        "P4_n": len(p4_dec),
        # P1: do that session's memories suffice (judged only where P4 failed)
        "P1_sufficient": (sum(1 for r in p1_dec if r["P1"]) / len(p1_dec)) if p1_dec else None,
        "P1_n": len(p1_dec),
        # P5: passed P4 (the answer really was in the context) yet still wrong,
        # which is the reader's responsibility. The denominator is the P4-passing
        # questions, the same definition as HaluMem's P5_fail_given_P4.
        "P5_fail_given_P4": (sum(1 for r in p4_ok if not r.get("is_correct")) / len(p4_ok))
                            if p4_ok else None,
        "P5_n": len(p4_ok),
        # P5b: did the abstention questions correctly abstain
        "P5b_abstain_rate": (sum(1 for r in abst if r.get("P5b_abstained")) / len(abst)) if abst else None,
        "P5b_n": len(abst),
        "verdicts": dict(v),
        # Attribution shares, with UNKNOWN and NO_DUMP excluded from the denominator
        "attribution": {
            k: v.get(k, 0) / (total - unknown) if (total - unknown) else 0
            for k in ["OK", "SUMMARY", "RETRIEVAL", "REASONING", "NO_WRITE", "P5b_FAIL"]
        },
        "unknown_ratio": unknown / total if total else 0,
        "judge_failures": dict(_FAILS),
        "scope": sorted({r.get("scope") for r in ans if r.get("scope")}),
        "by_question_type": {k: dict(c) for k, c in by_type.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory name under results/, e.g. mem0-v1_31b")
    ap.add_argument("--frame", default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    run(args.run, frame=args.frame, max_workers=args.max_workers)
