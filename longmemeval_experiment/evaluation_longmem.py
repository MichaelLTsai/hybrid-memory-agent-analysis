"""
LongMemEval-S evaluation (independent folder; imports halumem shared LLM judge).

Metrics per run:
  ① LLM-judge accuracy (overall + per question_type)   — main LongMemEval metric
  ② Token-level F1 (SQuAD-style)
  ③ Retrieval Recall@k / NDCG@k (SESSION-level, exact via session_id provenance)
     matching retrieved memories' session_id against answer_session_ids.

Abstention questions (question_id ends with "_abs"): correct = the system abstains
(says it doesn't know / no info), since the answer is not in the haystack.
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

RETRIEVAL_KS = [3, 5, 10]

JUDGE = """You are grading a long-term memory QA system.
Question: {question}
Reference answer: {answer}
System response: {response}
Is the system response semantically correct (conveys the reference answer)? Minor wording differences are fine.
Return strictly:
```json
{{"correct": true_or_false}}
```"""

JUDGE_ABS = """You are grading an ABSTENTION question: the information is NOT in the user's history,
so the correct behavior is to abstain (say it doesn't know / no information).
Question: {question}
System response: {response}
Is the system correctly abstaining (not fabricating an answer)?
Return strictly:
```json
{{"correct": true_or_false}}
```"""


def _judge(q) -> bool:
    resp = q.get("system_response", "")
    if str(q.get("question_id", "")).endswith("_abs"):
        p = JUDGE_ABS.format(question=q["question"], response=resp)
    else:
        p = JUDGE.format(question=q["question"], answer=q.get("answer", ""), response=resp)
    try:
        return bool(llm_request_for_json(p).get("correct", False))
    except Exception:
        return False


def _normalize(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def retrieval_metrics(ranked_sids, gold_sids):
    out = {f"recall@{k}": 0.0 for k in RETRIEVAL_KS}
    out.update({f"ndcg@{k}": 0.0 for k in RETRIEVAL_KS})
    gold = {s for s in gold_sids if s}
    # dedupe the ranked list to distinct sessions (many memories share one session)
    seen = set(); ranked_sids = [s for s in ranked_sids
                                 if s and not (s in seen or seen.add(s))]
    if not gold or not ranked_sids:
        return out
    for k in RETRIEVAL_KS:
        topk = ranked_sids[:k]
        covered = len(gold & {s for s in topk if s})
        out[f"recall@{k}"] = covered / len(gold)
        dcg   = sum((1.0 if topk[i] in gold else 0.0) / np.log2(i + 2) for i in range(len(topk)))
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gold))))
        out[f"ndcg@{k}"] = (dcg / ideal) if ideal > 0 else 0.0
    return out


def run_evaluation(version="default", result_dir=None, frame="mem0"):
    if result_dir is None:
        result_dir = f"./results/{frame}-{version}/"
    input_file  = os.path.join(result_dir, f"{frame}_lme_results.jsonl")
    scores_file = os.path.join(result_dir, f"{frame}_lme_scores.json")
    detail_file = os.path.join(result_dir, f"{frame}_lme_detail.jsonl")

    with open(input_file, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"Evaluating {len(items)} questions...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_judge, q): q for q in items}
        judged = {}
        for fut in as_completed(futs):
            judged[id(futs[fut])] = fut.result()

    tot = defaultdict(int); cor = defaultdict(int)
    f1_sum = 0.0; f1_n = 0
    retr_sum = defaultdict(float); retr_n = 0
    details = []
    for q in items:
        qt = q["question_type"]; ok = judged.get(id(q), False)
        tot[qt] += 1; tot["all"] += 1
        if ok:
            cor[qt] += 1; cor["all"] += 1
        q["is_correct"] = ok
        if q.get("answer") and not str(q.get("question_id","")).endswith("_abs"):
            f1 = token_f1(q.get("system_response", ""), str(q["answer"]))
            q["token_f1"] = round(f1, 4); f1_sum += f1; f1_n += 1
        gold = q.get("answer_session_ids", [])
        retrieved = q.get("retrieved", [])
        if gold and any(r.get("session_id") for r in retrieved):
            rm = retrieval_metrics([r.get("session_id") for r in retrieved], gold)
            q["retrieval"] = {k: round(v, 4) for k, v in rm.items()}
            for kk, vv in rm.items():
                retr_sum[kk] += vv
            retr_n += 1
        details.append(q)

    scores = {
        "qa_accuracy_all": cor["all"] / tot["all"] if tot["all"] else 0,
        "token_f1_all":    f1_sum / f1_n if f1_n else 0,
        "qa_num": tot["all"],
        "retrieval": {k: (retr_sum[k] / retr_n if retr_n else None)
                      for k in [f"recall@{x}" for x in RETRIEVAL_KS] + [f"ndcg@{x}" for x in RETRIEVAL_KS]},
        "per_type": {qt: {"accuracy": cor[qt] / tot[qt], "num": tot[qt]}
                     for qt in sorted(k for k in tot if k != "all")},
    }
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    with open(detail_file, "w", encoding="utf-8") as f:
        for d in details:
            d.pop("retrieved_memories", None)   # keep detail small
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n=== LongMemEval-S scores ===")
    print(json.dumps({k: v for k, v in scores.items() if k != "per_type"}, indent=2, ensure_ascii=False))
    print("per_type:", json.dumps(scores["per_type"], ensure_ascii=False))
    print(f"\nSaved → {scores_file}")
    return scores
