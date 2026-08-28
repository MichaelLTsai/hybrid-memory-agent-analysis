"""
HaluMem P4 / P5 / P5b probes: the stage-capability measurements the official
metrics do not provide.

HaluMem's official metrics cover Summary (integrity / accuracy) and Storage
(memory_update) but include nothing for the retrieval stage. qa_attribution.py
does produce a retrieval_ratio, but that is an **attribution share** whose
denominator is only the wrong answers, not a measure of retrieval ability:

    retrieval_ratio = P(failed at retrieval | answered wrong)
                      conditional; two systems can match here yet differ tenfold
    P4              = P(retrieved context suffices to answer)
                      absolute; comparable across datasets

This script runs P4 over **all questions**, which makes HaluMem's P4 mean the
same thing as LoCoMo's and LongMemEval's.

  P4   whether the retrieved context contains all of the question's evidence
       (all questions, not only the wrong ones)
  P5   passed P4 yet answered wrong, i.e. a reader-side problem
  P5b  Memory Boundary (no evidence; the answer is that it was never mentioned):
       whether the system correctly abstained

Adjudication reuses RETRIEVAL_PRESENCE_PROMPT from qa_attribution.py, so it
shares the definition with the existing attribution. Anything unadjudicable
returns None, is excluded from the denominator, and its cause is recorded in
judge_failures.

Usage:
  python probe_halumem.py --run mem0_oss-v1_31b_u2
"""

import os
import sys
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from qa_attribution import (
    parse_context_memories, RETRIEVAL_PRESENCE_PROMPT,
)
from llms import llm_request_for_json

BASE = os.path.dirname(os.path.abspath(__file__))

_FAILS = Counter()
_LOCK = threading.Lock()
_WARNED = set()


def _note_failure(kind, exc):
    with _LOCK:
        _FAILS[kind] += 1
        first = kind not in _WARNED
        _WARNED.add(kind)
    if first:
        print(f"[probe_halumem] {kind} judge failure (further ones of this kind are not printed): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


def judge_all_present(context_mems, evidences):
    """Whether all of the question's evidence is in the context. True/False, or None if unadjudicable."""
    if not evidences:
        return None
    if not context_mems:
        return False                      # an empty context is a valid verdict
    ev_block = "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidences))
    ctx_block = "\n".join(f"- {m}" for m in context_mems)
    prompt = RETRIEVAL_PRESENCE_PROMPT.format(context=ctx_block, evidence=ev_block)
    try:
        present = llm_request_for_json(prompt).get("present", [])
        if not isinstance(present, list) or len(present) != len(evidences):
            raise ValueError(f"'present' has the wrong length: {present!r}")
        return all(bool(x) for x in present)
    except Exception as e:
        _note_failure("retrieval", e)
        return None


def probe_one(qa):
    ev = [e.get("memory_content", "") for e in (qa.get("evidence") or [])
          if e.get("memory_content")]
    correct = qa.get("result_type") == "Correct"
    rec = {"question_type": qa.get("question_type"),
           "result_type": qa.get("result_type"), "is_correct": correct}

    if not ev:
        # Memory Boundary: the answer is that it was never mentioned, so nothing
        # should be retrieved; only correct abstention matters
        rec.update({"P4": None, "P5b_abstained": correct, "kind": "abstention",
                    "verdict": "OK" if correct else "P5b_FAIL"})
        return rec

    rec["kind"] = "answerable"
    # P4 is uniformly defined as what the reader can see at answering time.
    # Letta's agent_context includes the message history (recall memory);
    # pipeline-style backends have only the context field.
    ctx = qa.get("agent_context")
    ctx = list(ctx) if ctx else parse_context_memories(qa.get("context", ""))
    p4 = judge_all_present(ctx, ev)
    rec["P4"] = p4
    if p4 is None:
        rec["verdict"] = "UNKNOWN"
    elif p4:
        rec["verdict"] = "OK" if correct else "REASONING"     # P5
    else:
        rec["verdict"] = "NOT_RETRIEVED"                       # going further back needs qa_attribution
    return rec


def run(run_name, frame=None, max_workers=6, write=True):
    frame = frame or ("mem0_oss" if run_name.startswith("mem0_oss") else run_name.split("-")[0])
    rd = os.path.join(BASE, "results", run_name)
    detail_f = os.path.join(rd, f"{frame}_eval_detail.jsonl")
    scores_f = os.path.join(rd, f"{frame}_scores.json")
    out_f = os.path.join(rd, f"{frame}_probe_detail.jsonl")

    with open(detail_f, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    qas = [r for r in rows if "result_type" in r and "question" in r]
    print(f"Probing {len(qas)} questions ...")

    with _LOCK:
        _FAILS.clear(); _WARNED.clear()

    recs = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(probe_one, qa): qa for qa in qas}
        for fut in as_completed(futs):
            try:
                recs.append(fut.result())
            except Exception as e:
                _note_failure("worker", e)
                recs.append({"verdict": "UNKNOWN"})

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

    print("\n=== HaluMem probe ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_question_type"},
                     indent=2, ensure_ascii=False))
    return summary


def _aggregate(recs):
    ans = [r for r in recs if r.get("kind") == "answerable"]
    abst = [r for r in recs if r.get("kind") == "abstention"]
    p4_dec = [r for r in ans if isinstance(r.get("P4"), bool)]
    p4_ok = [r for r in p4_dec if r["P4"]]

    by_type = defaultdict(Counter)
    for r in recs:
        by_type[r.get("question_type", "?")][r.get("verdict")] += 1

    return {
        "P4_sufficient": (len(p4_ok) / len(p4_dec)) if p4_dec else None,
        "P4_n": len(p4_dec),
        # P5: among P4-passing questions, how many are still answered wrong
        # (the reader's responsibility)
        "P5_fail_given_P4": (sum(1 for r in p4_ok if not r["is_correct"]) / len(p4_ok))
                            if p4_ok else None,
        "P5_n": len(p4_ok),
        "P5b_abstain_rate": (sum(1 for r in abst if r.get("P5b_abstained")) / len(abst))
                            if abst else None,
        "P5b_n": len(abst),
        "verdicts": dict(Counter(r.get("verdict") for r in recs)),
        "unknown_ratio": sum(1 for r in recs if r.get("verdict") == "UNKNOWN") / len(recs)
                         if recs else 0,
        "judge_failures": dict(_FAILS),
        "by_question_type": {k: dict(c) for k, c in by_type.items()},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()
    run(args.run, frame=args.frame, max_workers=args.max_workers)
