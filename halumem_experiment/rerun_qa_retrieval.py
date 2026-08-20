"""
Retrieval ablation — re-run ONLY the QA-time retrieval on an existing Mem0 run,
reusing its stored memories (Qdrant store). Everything else is held fixed, so
memory_integrity / interference / update scores stay identical to the baseline;
only QA retrieval (context) and the resulting answers change.

Methods:
  ann     : baseline behavior (vector top-k) — sanity check, should ~= baseline
  mmr     : retrieve top-N candidates, then Maximal Marginal Relevance picks a
            relevant-but-diverse top-k (reduces redundant memories in context)
  rerank  : retrieve top-N candidates, then a cross-encoder re-scores (query, mem)
            and picks the top-k most relevant

Usage (run in the MAIN venv with mem0 2.0.x + sentence-transformers):
  python rerun_qa_retrieval.py --baseline mem0_oss-full_user1_gemma431b \
         --method mmr --version full_user1_gemma431b_mmr
  python rerun_qa_retrieval.py --baseline mem0_oss-full_user1_gemma431b \
         --method rerank --version full_user1_gemma431b_rerank

Then evaluate (also main venv):
  python run.py --backend mem0 --version full_user1_gemma431b_mmr --eval-only
"""

import os
import re
import json
import time
import copy
import argparse

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

import token_tracker as _tk
from prompts import PROMPT_MEMZERO
from llms import llm_request
from eval_mem0_oss import build_mem0_config, TEMPLATE_MEM0, _format_search_results, mem_search

from mem0 import Memory

load_dotenv()

RESULTS_DIR = "./results"
_EMBED = None
_CROSS = None


def _embed_model():
    global _EMBED
    if _EMBED is None:
        from sentence_transformers import SentenceTransformer
        _EMBED = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED


def _cross_encoder():
    global _CROSS
    if _CROSS is None:
        from sentence_transformers import CrossEncoder
        _CROSS = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS


# ── Retrieval methods (operate on a candidate list of memory texts) ────────────

def select_ann(query, cand_texts, k):
    """Baseline: candidates already come back ranked by vector similarity."""
    return cand_texts[:k]


def select_mmr(query, cand_texts, k, lambda_=0.5):
    """Maximal Marginal Relevance: relevance to query vs redundancy among picks."""
    if len(cand_texts) <= k:
        return cand_texts
    model = _embed_model()
    cv = model.encode(cand_texts, normalize_embeddings=True)
    qv = model.encode([query], normalize_embeddings=True)[0]
    sim_q = cv @ qv                      # relevance of each candidate to the query
    selected, remaining = [], list(range(len(cand_texts)))
    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda i: sim_q[i])
        else:
            def mmr_score(i):
                redundancy = max(float(cv[i] @ cv[j]) for j in selected)
                return lambda_ * sim_q[i] - (1 - lambda_) * redundancy
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return [cand_texts[i] for i in selected]


def select_rerank(query, cand_texts, k):
    """Cross-encoder reranking: re-score each (query, memory) pair."""
    if len(cand_texts) <= k:
        return cand_texts
    ce = _cross_encoder()
    scores = ce.predict([(query, t) for t in cand_texts])
    order = np.argsort(-scores)[:k]
    return [cand_texts[i] for i in order]


SELECTORS = {"ann": select_ann, "mmr": select_mmr, "rerank": select_rerank}


# ── Per-user QA re-run ─────────────────────────────────────────────────────────

def rerun_user(baseline_dir, uuid, method, n_candidates, top_k, out_tmp_dir):
    with open(os.path.join(baseline_dir, "tmp", f"{uuid}.json"), encoding="utf-8") as f:
        user_data = json.load(f)
    user_name = user_data["user_name"]

    # Open the EXISTING store read-only (reuse stored memories; no re-extraction)
    collection_name = f"halumem_{uuid.replace('-', '_')[:40]}"
    qdrant_path = os.path.join(baseline_dir, "qdrant_data")
    config = build_mem0_config(collection_name, qdrant_path)
    mem = Memory.from_config(config)

    selector = SELECTORS[method]

    for session in user_data["sessions"]:
        if "questions" not in session:
            continue
        for qa in session["questions"]:
            # Retrieve a LARGER candidate set, then re-select with the chosen method
            t0 = time.time()
            res = mem_search(mem, qa["question"], user_name, n_candidates)
            cand = _format_search_results(res.get("results", []))
            chosen = selector(qa["question"], cand, top_k)
            search_ms = (time.time() - t0) * 1000

            context = TEMPLATE_MEM0.format(
                user_id=user_name,
                memories=json.dumps(chosen, indent=4),
            )
            prompt = PROMPT_MEMZERO.format(context=context, question=qa["question"])
            t0 = time.time()
            response = llm_request(prompt)
            response_ms = (time.time() - t0) * 1000

            qa["context"] = context
            qa["search_duration_ms"] = round(search_ms, 1)
            qa["system_response"] = response
            qa["response_duration_ms"] = round(response_ms, 1)

    with open(os.path.join(out_tmp_dir, f"{uuid}.json"), "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=2)
    return user_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="e.g. mem0_oss-full_user1_gemma431b")
    ap.add_argument("--method", required=True, choices=list(SELECTORS))
    ap.add_argument("--version", required=True, help="new run version tag")
    ap.add_argument("--candidates", type=int, default=50, help="candidate pool size before re-selection")
    ap.add_argument("--top-k", type=int, default=20, help="final memories passed to QA LLM")
    args = ap.parse_args()

    baseline_dir = os.path.join(RESULTS_DIR, args.baseline)
    out_dir = os.path.join(RESULTS_DIR, f"mem0_oss-{args.version}")
    out_tmp = os.path.join(out_dir, "tmp")
    os.makedirs(out_tmp, exist_ok=True)

    uuids = [f[:-5] for f in os.listdir(os.path.join(baseline_dir, "tmp"))
             if f.endswith(".json") and not f.endswith("_error.json")]

    print(f"\n{'='*60}")
    print(f"  Retrieval ablation: {args.method.upper()}")
    print(f"  Baseline   : {args.baseline}  (reusing stored memories)")
    print(f"  Candidates : {args.candidates} → top-k {args.top_k}")
    print(f"  Users      : {len(uuids)}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    for uuid in tqdm(uuids, desc=f"{args.method} re-retrieval"):
        rerun_user(baseline_dir, uuid, args.method, args.candidates, args.top_k, out_tmp)

    # Merge tmp → eval_results.jsonl
    out_file = os.path.join(out_dir, "mem0_oss_eval_results.jsonl")
    with open(out_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(out_tmp)):
            if fn.endswith(".json"):
                with open(os.path.join(out_tmp, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(out_dir, "mem0_oss")
    print(f"\n✅ Done → {out_file}")
    print(f"   Next: python run.py --backend mem0 --version {args.version} --eval-only")


if __name__ == "__main__":
    main()
