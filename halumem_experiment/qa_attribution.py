"""
QA error attribution: assign every wrongly answered question to one of three stages.

  STORAGE     : the memory needed to answer was never stored (extraction or
                deduplication failed)
  RETRIEVAL   : the memory was stored but was not retrieved for this question
  GENERATION  : the memory was retrieved yet the LLM still answered wrong
                (answer LLM or prompt failure)

Method (robust; entirely LLM-judged, with no embedding threshold):
  Call 1  retrieval presence: is each piece of evidence in the retrieved context?
            all present -> GENERATION
            any missing -> proceed to Call 2
  Call 2  storage presence: is the missing evidence in the full pool of stored
            memories? (embeddings only narrow the pool to top-K candidates; the
            verdict is still the LLM's)
            present -> RETRIEVAL
            absent  -> STORAGE
  Priority: STORAGE > RETRIEVAL > GENERATION
            (if even one required fact was never stored, that is the most
            fundamental failure)

Anything unadjudicable is labelled UNKNOWN and never replaced by a default. If a
judge failure defaulted to "not retrieved" or "not stored", the bias would all
flow into STORAGE, letting LLM flakiness masquerade as a defect in the memory
system. The count and causes of UNKNOWN are reported separately in the summary.

  NO_EVIDENCE : the question has no evidence to check against (Memory Boundary,
                for instance), so attribution is structurally impossible
  UNKNOWN     : judge failure (LLM error or embedding failure); no verdict this time

Integrated into evaluation.py: after the QA evaluation finishes, wrong answers
are attributed automatically, the result is written to the "qa_attribution" block
of scores.json, and per-question labels go into the detail file.
"""

import os
import re
import sys
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from llms import llm_request_for_json

STORAGE_TOPK = 25          # candidates pulled from the pool for the LLM to inspect
_EMBED_MODEL = None        # lazy-loaded sentence-transformer

# Judge-failure counters (thread-safe), used to expose the causes of UNKNOWN
_FAILS = Counter()
_FAILS_LOCK = threading.Lock()
_WARNED = set()


def _note_failure(kind: str, exc: Exception) -> None:
    """Record one judge failure; print the full message to stderr only once per cause."""
    with _FAILS_LOCK:
        _FAILS[kind] += 1
        first = kind not in _WARNED
        _WARNED.add(kind)
    if first:
        print(f"[qa_attribution] {kind} judge failure (further ones of this kind are not printed): "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)


# ── Prompts ─────────────────────────────────────────────────────────────────

RETRIEVAL_PRESENCE_PROMPT = """You are auditing an AI memory system. Your job is to check whether the information needed to answer a question was successfully RETRIEVED into the provided context.

# Retrieved Memories (what the system pulled up for this question)
{context}

# Required Evidence Facts (needed to answer correctly)
{evidence}

For EACH required evidence fact, decide whether that same fact is semantically present in the Retrieved Memories (the identical fact, even if reworded or rephrased counts as present; a merely related or topically-similar memory does NOT count).

Return strictly this JSON:
```json
{{"present": [true_or_false, ...]}}
```
The list must have exactly {n} boolean entries, in the same order as the evidence facts.
"""

STORAGE_PRESENCE_PROMPT = """You are auditing an AI memory system. Your job is to check whether a specific fact was ever STORED in the system's memory.

# Candidate Stored Memories (the most similar memories found in the entire store)
{candidates}

# Fact to check
{fact}

Is this fact semantically present among the candidate stored memories (the identical fact, even if reworded counts as present; a merely related or topically-similar memory does NOT count)?

Return strictly this JSON:
```json
{{"present": true_or_false}}
```
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

_EMBED_LOCK = threading.Lock()
_POOL_VECS = {}          # id(pool) -> encoded pool matrix


def _get_embed_model():
    """Lazy singleton. Locked: several worker threads hit this at once on startup and
    loading the model concurrently either races or blows memory."""
    global _EMBED_MODEL
    with _EMBED_LOCK:
        if _EMBED_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBED_MODEL


def _pool_vectors(pool: list[str]):
    """
    Encode the memory pool ONCE per pool and cache it.

    Without this every _judge_storage_presence call re-encodes the whole pool —
    for one HaluMem user that is ~120 questions x ~670 memories = ~80k redundant
    encodings, which under a ThreadPoolExecutor exhausts memory and makes the
    embedding step fail wholesale (it showed up as 82/122 UNKNOWN attributions).
    """
    key = id(pool)
    with _EMBED_LOCK:
        cached = _POOL_VECS.get(key)
    if cached is not None:
        return cached
    vecs = _get_embed_model().encode(pool, normalize_embeddings=True,
                                     batch_size=64, show_progress_bar=False)
    with _EMBED_LOCK:
        _POOL_VECS[key] = vecs
    return vecs


def _strip_ts(s: str) -> str:
    """Strip the ISO timestamp prefix from a memory, leaving only its content."""
    return re.sub(r"^\s*\S*\d{4}-\d{2}-\d{2}\S*[:\s]+", "", s).strip()


def parse_context_memories(context: str) -> list[str]:
    """
    Extract each retrieved memory from a QA record's context field.

    Two formats must be handled:
      - mem0 / amem / memos / structmem -> a JSON array with memories in double quotes
      - letta                           -> a memory block, newline separated, unquoted

    A quote-only regex parses zero memories for letta, so every question would be
    judged "not retrieved", systematically inflating retrieval and storage while
    pinning GENERATION at zero. When the quoted form yields nothing, fall back to
    splitting on newlines.
    """
    ctx = context or ""
    mems = []
    for m in re.findall(r'"((?:[^"\\]|\\.)*)"', ctx):
        m = _strip_ts(m)
        if len(m) > 10:
            mems.append(m)
    if len(mems) >= 2:
        return mems

    # Newline format: drop the leading description line and any bullet markers
    out = []
    for line in ctx.splitlines():
        line = _strip_ts(line.strip().lstrip("-•*").strip())
        if len(line) > 10 and not line.lower().startswith(("memories for", "no memories")):
            out.append(line)
    return out if len(out) > len(mems) else mems


def build_stored_pools(users: list[dict]) -> dict:
    """uuid -> every memory the system actually stored for that user (across sessions, deduplicated)."""
    pools = {}
    for u in users:
        uuid = u.get("uuid")
        seen, pool = set(), []
        for s in u.get("sessions", []):
            for m in s.get("extracted_memories", []):
                if isinstance(m, str):
                    t = _strip_ts(m)
                    if len(t) > 5 and t not in seen:
                        seen.add(t)
                        pool.append(t)
        pools[uuid] = pool
    return pools


# ── LLM judges ────────────────────────────────────────────────────────────────

def _judge_retrieval_presence(context_mems: list[str], evidences: list[str]) -> list[bool | None]:
    """True/False per evidence item; None when unadjudicable (never defaulted to False)."""
    ev_block = "\n".join(f"{i+1}. {e}" for i, e in enumerate(evidences))
    ctx_block = "\n".join(f"- {m}" for m in context_mems) if context_mems else "(none retrieved)"
    prompt = RETRIEVAL_PRESENCE_PROMPT.format(context=ctx_block, evidence=ev_block, n=len(evidences))
    try:
        result = llm_request_for_json(prompt)
        present = result.get("present", [])
        if not isinstance(present, list):
            raise ValueError(f"'present' is not a list: {present!r}")
        if len(present) != len(evidences):       # model misaligned -> void the whole batch
            raise ValueError(f"'present' length {len(present)} != evidence count {len(evidences)}")
        return [bool(x) for x in present]
    except Exception as e:
        _note_failure("retrieval", e)
        return [None] * len(evidences)


def _judge_storage_presence(fact: str, pool: list[str]) -> bool | None:
    """True/False, or None when unadjudicable. An empty pool is a valid verdict (nothing was stored)."""
    if not pool:
        return False
    # Embeddings narrow the candidates. On failure do not degrade to pool[:K]:
    # that is equivalent to random candidates and would misjudge as not stored
    try:
        import numpy as np
        pv = _pool_vectors(pool)                       # cached per pool
        fv = _get_embed_model().encode([fact], normalize_embeddings=True)[0]
        order = np.argsort(-(pv @ fv))[:STORAGE_TOPK]
        candidates = [pool[i] for i in order]
    except Exception as e:
        _note_failure("embedding", e)
        return None
    cand_block = "\n".join(f"- {c}" for c in candidates)
    prompt = STORAGE_PRESENCE_PROMPT.format(candidates=cand_block, fact=fact)
    try:
        result = llm_request_for_json(prompt).get("present", None)
        if not isinstance(result, bool):
            raise ValueError(f"'present' is not a bool: {result!r}")
        return result
    except Exception as e:
        _note_failure("storage", e)
        return None


def _attribute_one(qa: dict, pool: list[str]) -> str:
    evidences = [e.get("memory_content", "") for e in qa.get("evidence", []) if e.get("memory_content")]
    if not evidences:
        return "NO_EVIDENCE"           # no evidence (Memory Boundary); structurally unattributable
    ctx_mems = parse_context_memories(qa.get("context", ""))

    present = _judge_retrieval_presence(ctx_mems, evidences)
    if any(p is None for p in present):
        return "UNKNOWN"               # retrieval verdict failed; do not guess
    if all(present):
        return "GENERATION"            # everything was retrieved yet the answer is wrong

    # For evidence that was not retrieved, check item by item whether it was stored
    for evi, in_ctx in zip(evidences, present):
        if in_ctx:
            continue
        stored = _judge_storage_presence(evi, pool)
        if stored is None:
            return "UNKNOWN"           # storage verdict failed; do not guess
        if not stored:
            return "STORAGE"           # one item was never stored; the deepest failure
    return "RETRIEVAL"                 # everything missing is in the store; a pure retrieval miss


# ── Public entry ──────────────────────────────────────────────────────────────

def attribute_failures(qa_records: list[dict], pools: dict, max_workers: int = 8) -> dict:
    """
    Attribute every QA record with result_type != Correct, adding an 'attribution'
    field to the record in place.

    Returns a summary dict: totals, the share of each of the three categories, and
    a breakdown by question_type.
    """
    wrong = [qa for qa in qa_records if qa.get("result_type") != "Correct"]

    with _FAILS_LOCK:
        _FAILS.clear()
        _WARNED.clear()

    counts = Counter()
    by_type = defaultdict(Counter)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_attribute_one, qa, pools.get(qa.get("uuid"), [])): qa
            for qa in wrong
        }
        for fut in as_completed(futures):
            qa = futures[fut]
            try:
                verdict = fut.result()
            except Exception as e:
                _note_failure("worker", e)
                verdict = "UNKNOWN"
            qa["attribution"] = verdict
            counts[verdict] += 1
            by_type[qa.get("question_type", "?")][verdict] += 1

    total = sum(counts.values())
    unknown = counts.get("UNKNOWN", 0)
    no_evidence = counts.get("NO_EVIDENCE", 0)
    # The denominator holds only genuinely adjudicated questions, excluding both
    # structurally unattributable ones and judge failures
    attributable = total - unknown - no_evidence
    summary = {
        "total_wrong":         total,
        "attributable":        attributable,
        "storage_num":         counts.get("STORAGE", 0),
        "retrieval_num":       counts.get("RETRIEVAL", 0),
        "generation_num":      counts.get("GENERATION", 0),
        "unknown_num":         unknown,
        "no_evidence_num":     no_evidence,
        "storage_ratio":       counts.get("STORAGE", 0) / attributable if attributable else 0,
        "retrieval_ratio":     counts.get("RETRIEVAL", 0) / attributable if attributable else 0,
        "generation_ratio":    counts.get("GENERATION", 0) / attributable if attributable else 0,
        # The UNKNOWN share must stay visible, otherwise the reliability of the
        # attribution cannot be assessed
        "unknown_ratio":       unknown / total if total else 0,
        "judge_failures":      dict(_FAILS),
        "storage_topk":        STORAGE_TOPK,
        "by_question_type":    {k: dict(v) for k, v in by_type.items()},
    }
    return summary


# ── CLI: backfill attribution into an existing run's scores.json ───────────────

def _backfill(run: str, max_workers: int = 6):
    """Attribute an existing run and write back to {frame}_scores.json and eval_detail."""
    base = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(base, "results", run)
    frame   = run.split("-")[0]

    detail_path = os.path.join(run_dir, f"{frame}_eval_detail.jsonl")
    extract_path = os.path.join(run_dir, f"{frame}_eval_results.jsonl")
    scores_path = os.path.join(run_dir, f"{frame}_scores.json")

    # Read the already computed QA records and extraction results
    qa_records, other_records = [], []
    with open(detail_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            (qa_records if ("question" in d and "result_type" in d) else other_records).append(d)

    users = []
    with open(extract_path, encoding="utf-8") as f:
        for line in f:
            users.append(json.loads(line))

    pools = build_stored_pools(users)
    print(f"QA records: {len(qa_records)} | wrong: "
          f"{sum(1 for q in qa_records if q.get('result_type')!='Correct')}")
    print("Attributing...")
    summary = attribute_failures(qa_records, pools, max_workers=max_workers)

    # Write back to scores.json
    with open(scores_path, encoding="utf-8") as f:
        scores = json.load(f)
    scores["qa_attribution"] = summary
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    # Write back to detail (qa records already carry the attribution field)
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in other_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in qa_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["unknown_num"]:
        print(f"\n⚠️  {summary['unknown_num']}/{summary['total_wrong']} questions unadjudicable "
              f"({summary['unknown_ratio']:.1%}); causes: {summary['judge_failures']}."
              f"\n    The three ratios already exclude these; when this share is high "
              f"the results should not be cited directly.")
    print(f"\n✅ Written to {scores_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. mem0_oss-full_user1_atomic_31b")
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()
    _backfill(args.run, args.max_workers)
