"""
LoCoMo evaluation (independent from halumem_experiment).

Metrics computed per run:
  ① LLM-judge accuracy (category-aware) — the modern LoCoMo main metric
  ② Token-level F1 (SQuAD-style) on answerable questions — for comparison with
     token-overlap papers
  ③ Retrieval Recall@k / NDCG@k (k=3,5,10) — how well retrieval covered the gold
     evidence turns.

     Each stored memory carries its source dia_id (the adapter feeds turn-by-turn
     with metadata={"dia_id": ...}), so Recall@k / NDCG@k are computed by EXACT
     dia_id matching between the ranked retrieved memories and the gold evidence
     dia_ids — no embedding approximation.

Outputs:
  results/mem0-{version}/mem0_locomo_scores.json
  results/mem0-{version}/mem0_locomo_detail.jsonl
"""

import os
import re
import sys
import json
import string
from collections import Counter, defaultdict

import numpy as np

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from llms import llm_request_for_json   # noqa: E402

CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
RETRIEVAL_KS = [3, 5, 10]


# ── ① LLM judge (category-aware) ──────────────────────────────────────────────

JUDGE_ANSWERABLE = """You are grading a question-answering system on a long-conversation memory task.
Question: {question}
Reference answer: {answer}
System response: {response}
Is the system response semantically correct (conveys the reference answer)? Minor wording differences are fine.
Return strictly:
```json
{{"correct": true_or_false}}
```"""

JUDGE_ADVERSARIAL = """You are grading a system on an ADVERSARIAL question. The conversation does NOT support
this tempting-but-wrong answer.
Question: {question}
Tempting-but-WRONG answer (trap): {answer}
System response: {response}
The system is CORRECT if it AVOIDS the trap (does not assert the wrong answer / says no info).
Return strictly:
```json
{{"correct": true_or_false}}
```"""


def _judge(qa) -> bool:
    if qa.get("category") == 5:
        p = JUDGE_ADVERSARIAL.format(question=qa["question"],
                                     answer=qa.get("adversarial_answer", ""),
                                     response=qa.get("system_response", ""))
    else:
        p = JUDGE_ANSWERABLE.format(question=qa["question"],
                                    answer=qa.get("answer", ""),
                                    response=qa.get("system_response", ""))
    try:
        return bool(llm_request_for_json(p).get("correct", False))
    except Exception:
        return False


# ── ② Token-level F1 (SQuAD-style) ────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


# ── ③ Retrieval Recall@k / NDCG@k (exact dia_id matching) ─────────────────────

def retrieval_metrics(retrieved_dia_ids: list, gold_evidence: list) -> dict:
    """Recall@k / NDCG@k by exact dia_id match of ranked retrieved memories vs gold."""
    out = {f"recall@{k}": 0.0 for k in RETRIEVAL_KS}
    out.update({f"ndcg@{k}": 0.0 for k in RETRIEVAL_KS})
    gold = {d for d in gold_evidence if d}
    seen = set(); retrieved_dia_ids = [d for d in retrieved_dia_ids
                                       if d and not (d in seen or seen.add(d))]
    if not gold or not retrieved_dia_ids:
        return out
    for k in RETRIEVAL_KS:
        topk = retrieved_dia_ids[:k]
        covered = len(gold & {d for d in topk if d})
        out[f"recall@{k}"] = covered / len(gold)
        # NDCG: relevance 1 if the retrieved dia_id at position i is a gold evidence
        dcg   = sum((1.0 if topk[i] in gold else 0.0) / np.log2(i + 2) for i in range(len(topk)))
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gold))))
        out[f"ndcg@{k}"] = (dcg / ideal) if ideal > 0 else 0.0
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(version="default", result_dir=None, frame="mem0",
                   extraction=True, max_workers=8, max_obs=None, max_mem=None):
    if result_dir is None:
        result_dir = f"./results/{frame}-{version}/"
    input_file  = os.path.join(result_dir, f"{frame}_locomo_results.jsonl")
    scores_file = os.path.join(result_dir, f"{frame}_locomo_scores.json")
    detail_file = os.path.join(result_dir, f"{frame}_locomo_detail.jsonl")

    with open(input_file, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    tot = defaultdict(int); cor = defaultdict(int)
    f1_sum = defaultdict(float); f1_n = defaultdict(int)
    retr_sum = defaultdict(float); retr_n = 0
    details = []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_items = [qa for s in samples for qa in s["qa"]]
    print(f"Evaluating {len(all_items)} QA...")

    # LLM judge (parallel)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_judge, qa): qa for qa in all_items}
        judged = {}
        for fut in as_completed(futs):
            judged[id(futs[fut])] = fut.result()

    # F1 + retrieval
    for qa in all_items:
        cat = qa.get("category")
        ok  = judged.get(id(qa), False)
        tot[cat] += 1; tot["all"] += 1
        if ok:
            cor[cat] += 1; cor["all"] += 1
        qa["is_correct"] = ok

        # token-F1 (answerable only)
        if cat != 5 and qa.get("answer") is not None:
            f1 = token_f1(qa.get("system_response", ""), str(qa["answer"]))
            qa["token_f1"] = round(f1, 4)
            f1_sum[cat] += f1; f1_n[cat] += 1
            f1_sum["all"] += f1; f1_n["all"] += 1

        # retrieval metrics (exact dia_id match) — only when the backend provides
        # turn-level dia_id provenance (else null, e.g. session-level Letta/Graphiti)
        gold_ev = qa.get("evidence", [])
        retrieved = qa.get("retrieved", [])
        if gold_ev and any(r.get("dia_id") for r in retrieved):
            ranked_dia = [r.get("dia_id") for r in retrieved]
            rm = retrieval_metrics(ranked_dia, gold_ev)
            qa["retrieval"] = {k: round(v, 4) for k, v in rm.items()}
            for kk, vv in rm.items():
                retr_sum[kk] += vv
            retr_n += 1

        details.append(qa)

    scores = {
        "qa_accuracy_all": cor["all"] / tot["all"] if tot["all"] else 0,   # LLM judge
        "token_f1_all":    f1_sum["all"] / f1_n["all"] if f1_n["all"] else 0,
        "qa_num": tot["all"],
        # None (not 0) when no run had turn-level dia_id provenance (e.g. session-level Graphiti)
        "retrieval": {k: (retr_sum[k] / retr_n if retr_n else None)
                      for k in [f"recall@{x}" for x in RETRIEVAL_KS] + [f"ndcg@{x}" for x in RETRIEVAL_KS]},
        "retrieval_note": "exact dia_id matching (each memory carries its source dia_id)",
        "per_category": {},
    }
    for c in sorted(k for k in tot if k != "all"):
        name = CATEGORY_NAMES.get(c, str(c))
        scores["per_category"][name] = {
            "accuracy": cor[c] / tot[c] if tot[c] else 0,
            "token_f1": f1_sum[c] / f1_n[c] if f1_n[c] else None,
            "num": tot[c],
        }

    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    with open(detail_file, "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 4. Extraction stage (a metric LoCoMo does not ship), which needs
    #    results.jsonl to carry a memory_dump. Written to the "extraction" block of
    #    scores.json; without a dump it is marked skipped rather than invented.
    if extraction:
        try:
            from extraction_locomo import run as run_extraction_eval
            print("\n--- Extraction-stage evaluation (integrity / accuracy / F1) ---")
            scores["extraction"] = run_extraction_eval(
                os.path.basename(os.path.normpath(result_dir)),
                frame=frame, max_workers=max_workers,
                max_obs=max_obs, max_mem=max_mem)
        except Exception as e:
            print(f"⚠️  extraction evaluation failed, skipped: {type(e).__name__}: {e}")

    print("\n=== LoCoMo scores ===")
    print(json.dumps({k: v for k, v in scores.items() if k != "per_category"}, indent=2, ensure_ascii=False))
    print(f"\nSaved → {scores_file}")
    return scores
