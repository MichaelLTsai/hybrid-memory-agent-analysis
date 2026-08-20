"""
QA error attribution: assign every wrongly answered question to one of three stages.
  STORAGE     : the needed memory was never stored (extraction or deduplication failed)
  RETRIEVAL   : the memory was stored but was not retrieved for this question
  GENERATION  : the memory was retrieved yet the LLM answered wrong
                (answer LLM or prompt failure)

Method: embedding similarity between evidence and the stored pool or retrieved
context. Mem0 rewrites sentences, so exact string matching cannot be used.

Usage:
  python analyze_qa_attribution.py --run mem0_oss-full_user1_atomic_31b
  python analyze_qa_attribution.py --run mem0_oss-full_user1_atomic_31b --sim 0.6
"""

import os
import re
import json
import glob
import argparse
from collections import Counter, defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Similarity threshold at which one evidence item counts as a hit in a memory pool
DEFAULT_SIM = 0.60


def load_stored_pool(run_dir: str) -> list[str]:
    """Collect every memory the system actually stored, across sessions, from tmp/*.json."""
    pool = []
    tmp_dir = os.path.join(run_dir, "tmp")
    for fpath in glob.glob(os.path.join(tmp_dir, "*.json")):
        if fpath.endswith("_error.log"):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for s in data.get("sessions", []):
            for m in s.get("extracted_memories", []):
                if isinstance(m, str) and m.strip():
                    pool.append(m.strip())
    return pool


def load_failed_qa(run_dir: str, frame: str) -> list[dict]:
    """Read eval_detail and pull the wrong (non-Correct) QA records carrying evidence and context."""
    detail = os.path.join(run_dir, f"{frame}_eval_detail.jsonl")
    out = []
    with open(detail, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "question" not in d or "result_type" not in d:
                continue
            if d["result_type"] == "Correct":
                continue
            ev = d.get("evidence") or []
            if not ev:                      # no evidence (Memory Boundary); unattributable
                continue
            out.append(d)
    return out


def strip_timestamp(s: str) -> str:
    """Strip the ISO timestamp prefix from context or memory text, leaving only content."""
    return re.sub(r"^\s*\S*\d{4}-\d{2}-\d{2}\S*[:\s]+", "", s).strip()


def parse_context_memories(context: str) -> list[str]:
    """Extract each retrieved memory from a QA record's context field."""
    mems = []
    for m in re.findall(r'"((?:[^"\\]|\\.)*)"', context):
        m = strip_timestamp(m)
        if len(m) > 10:
            mems.append(m)
    return mems


def max_sim(evi_vec, pool_vecs) -> float:
    if pool_vecs is None or len(pool_vecs) == 0:
        return 0.0
    sims = pool_vecs @ evi_vec
    return float(np.max(sims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. mem0_oss-full_user1_atomic_31b")
    ap.add_argument("--sim", type=float, default=DEFAULT_SIM, help="similarity threshold for a hit")
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_DIR, args.run)
    frame   = args.run.split("-")[0]              # mem0_oss

    print(f"Loading model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    stored_pool = load_stored_pool(run_dir)
    failed_qa   = load_failed_qa(run_dir, frame)
    print(f"Stored memories: {len(stored_pool)}")
    print(f"Failed QA (with evidence): {len(failed_qa)}")
    print(f"Similarity threshold: {args.sim}\n")

    # Pre-encode the entire stored pool
    pool_vecs = model.encode(stored_pool, normalize_embeddings=True,
                             show_progress_bar=True) if stored_pool else np.zeros((0, 384))

    # Attribute question by question. A question may carry several evidence items;
    # classify by the most fundamental failure:
    #   any evidence item never stored     -> STORAGE
    #   else any stored but not retrieved  -> RETRIEVAL
    #   else (all retrieved yet wrong)     -> GENERATION
    attribution = Counter()
    by_type = defaultdict(Counter)
    examples = defaultdict(list)

    for qa in failed_qa:
        evidences = [e.get("memory_content", "") for e in qa["evidence"] if e.get("memory_content")]
        if not evidences:
            continue
        ctx_mems = parse_context_memories(qa.get("context", ""))
        ctx_vecs = model.encode(ctx_mems, normalize_embeddings=True) if ctx_mems else np.zeros((0, 384))
        evi_vecs = model.encode(evidences, normalize_embeddings=True)

        verdict = "GENERATION"   # default: everything retrieved yet still wrong
        for evi_vec in evi_vecs:
            in_storage = max_sim(evi_vec, pool_vecs) >= args.sim
            in_context = max_sim(evi_vec, ctx_vecs) >= args.sim
            if not in_storage:
                verdict = "STORAGE"
                break
            if not in_context:
                verdict = "RETRIEVAL"
                # Do not break: a later unstored item outranks this, as STORAGE wins
        attribution[verdict] += 1
        by_type[qa["question_type"]][verdict] += 1
        if len(examples[verdict]) < 3:
            examples[verdict].append(qa)

    total = sum(attribution.values())
    print("="*60)
    print("QA failure attribution")
    print("="*60)
    for k in ["STORAGE", "RETRIEVAL", "GENERATION"]:
        n = attribution[k]
        print(f"  {k:12s}: {n:3d}  ({n/total*100:.1f}%)")
    print(f"  {'TOTAL':12s}: {total}")

    print("\nBreakdown by question_type:")
    for qt, c in sorted(by_type.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values())
        print(f"  {qt:28s} (n={tot:3d}): "
              f"STORAGE={c['STORAGE']:2d} RETRIEVAL={c['RETRIEVAL']:2d} GENERATION={c['GENERATION']:2d}")

    print("\nExamples per category:")
    for k in ["STORAGE", "RETRIEVAL", "GENERATION"]:
        print(f"\n--- {k} ---")
        for qa in examples[k]:
            print(f"  Q: {qa['question'][:80]}")
            print(f"  gold  : {qa['answer'][:60]}")
            print(f"  system: {qa['system_response'][:60]}  [{qa['result_type']}]")


if __name__ == "__main__":
    main()
