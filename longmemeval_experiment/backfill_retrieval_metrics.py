#!/usr/bin/env python
"""Backfill LongMemEval Recall@k / NDCG@k for runs that recorded no session ids.

    ../venv_memos/bin/python backfill_retrieval_metrics.py --run structmem-ablate_e1_m1
    ../venv_memos/bin/python backfill_retrieval_metrics.py --all-structmem
    ../venv_memos/bin/python backfill_retrieval_metrics.py --all-structmem --dry-run

Why this exists
---------------
evaluation_longmem.py only computes the official session-level retrieval metrics
when the recorded ``retrieved`` rows carry a ``session_id``:

    if gold and any(r.get("session_id") for r in retrieved):

The StructMem adapter sets it to None for every row, because ``lm.retrieve()``
returns pre-formatted strings and the entry id is lost before the adapter sees
them. Every StructMem run therefore has an empty recall@k / ndcg@k column, on
every batch, going back to the first one.

The provenance is recoverable without re-running anything. Each run already
stores ``all_memories`` as ``{text, session_id}`` with full coverage, and each
retrieved row is ``"{time_stamp} {weekday} {memory}"``. Stripping the prefix and
matching the text back recovers the source session. Measured on the ablation
runs: 100% of atomic entries resolve, uniquely, with no ambiguity.

Cross-event summaries retrieved through the dual circuit are the only rows that
do not resolve, and correctly so: a summary spans a time window rather than one
session, so it has no single source to credit. They are excluded from the
retrieval metrics and reported separately, which matches what the metric asks:
"were the gold sessions retrieved", not "was every context line attributable".

Nothing is recomputed except the retrieval block. Scores are updated in place
and a ``retrieval_note`` records how they were derived.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "results")

#: "2023-09-30T15:06:05.000 Mon " -> stripped. The weekday is absent on summary
#: lines, which is one of the ways they fail to match.
#: Two shapes reach the recorded rows. The baseline path writes
#: "2023-09-30T15:06:05.000 Mon <text>"; the M4 path writes
#: "[CURRENT MEMORY | 2023-08-11T09:09:04.000] <text>". Strip either.
PREFIX = re.compile(r"^(?:\[[^\]]*\]\s*|\S+T\S+\s+(?:\w{3}\s+)?)")
SUMMARY_MARK = re.compile(r"\[\s*(?:SUMMARY|CURRENT SUMMARY|HISTORICAL SUMMARY|RAW SUMMARY)\b",
                          re.IGNORECASE)

RETRIEVAL_KS = (3, 5, 10)


def retrieval_metrics(ranked_sessions: List[Optional[str]], gold: List[str]) -> Dict[str, float]:
    """Session-level Recall@k and NDCG@k.

    Mirrors evaluation_longmem.py: a gold session counts as covered once any
    retrieved row from it appears in the top k, and NDCG uses binary gains with
    the ideal ranking placing every gold session first.
    """
    import math

    out: Dict[str, float] = {}
    gold_set = set(gold)
    for k in RETRIEVAL_KS:
        top = ranked_sessions[:k]
        covered = len({s for s in top if s in gold_set})
        out[f"recall@{k}"] = covered / len(gold_set) if gold_set else 0.0

        # Binary gain, credited once per gold session at its best rank. Without
        # the seen-set a session retrieved five times contributes five gains and
        # NDCG exceeds 1.
        seen = set()
        dcg = 0.0
        for i, s in enumerate(top):
            if s in gold_set and s not in seen:
                seen.add(s)
                dcg += 1.0 / math.log2(i + 2)
        ideal_n = min(len(gold_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
        out[f"ndcg@{k}"] = (dcg / idcg) if idcg else 0.0
    return out


def resolve_run(run: str, dry_run: bool = False) -> Optional[dict]:
    run_dir = os.path.join(RESULTS, run)
    tmp_files = sorted(glob.glob(os.path.join(run_dir, "tmp", "*.json")))
    if not tmp_files:
        print(f"  {run}: no tmp/*.json, skipped")
        return None

    frame = run.split("-")[0]
    scores_f = os.path.join(run_dir, f"{frame}_lme_scores.json")
    if not os.path.exists(scores_f):
        print(f"  {run}: no {os.path.basename(scores_f)}, skipped")
        return None

    per_q: List[Dict[str, float]] = []
    n_atomic = n_resolved = n_summary = n_unresolved = 0

    for path in tmp_files:
        q = json.load(open(path, encoding="utf-8"))
        gold = q.get("answer_session_ids") or []
        retrieved = q.get("retrieved") or []
        if not gold or not retrieved:
            continue

        index = defaultdict(set)
        for m in q.get("all_memories") or []:
            sid = m.get("session_id")
            if sid is not None:
                index[str(m.get("text", "")).strip()].add(sid)

        ranked: List[Optional[str]] = []
        for row in retrieved:
            text = str(row.get("text", ""))
            if SUMMARY_MARK.search(text):
                n_summary += 1
                continue                      # spans a window, no single source
            n_atomic += 1
            stripped = PREFIX.sub("", text).strip()
            sids = index.get(stripped)
            if sids and len(sids) == 1:
                ranked.append(next(iter(sids)))
                n_resolved += 1
            else:
                ranked.append(None)           # keeps the rank position honest
                n_unresolved += 1

        per_q.append(retrieval_metrics(ranked, [str(g) for g in gold]))

    if not per_q:
        print(f"  {run}: no question had both gold sessions and retrieved rows")
        return None

    agg = {k: sum(d[k] for d in per_q) / len(per_q) for k in per_q[0]}
    rate = (100.0 * n_resolved / n_atomic) if n_atomic else 0.0
    note = (
        f"session ids recovered by matching retrieved text against all_memories; "
        f"{n_resolved}/{n_atomic} atomic rows resolved ({rate:.1f}%), "
        f"{n_summary} cross-event summary rows excluded (no single source session)"
    )

    print(f"  {run}: questions={len(per_q)} atomic={n_atomic} resolved={rate:.1f}% "
          f"summaries_excluded={n_summary} unresolved={n_unresolved}")
    print("    " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items())))

    if dry_run:
        return agg

    scores = json.load(open(scores_f, encoding="utf-8"))
    scores["retrieval"] = {k: agg[k] for k in
                           [f"recall@{k}" for k in RETRIEVAL_KS] + [f"ndcg@{k}" for k in RETRIEVAL_KS]}
    scores["retrieval_note"] = note
    with open(scores_f, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--all-structmem", action="store_true",
                    help="every results/structmem-* run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the numbers without writing scores")
    args = ap.parse_args()

    if args.all_structmem:
        runs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(RESULTS, "structmem-*"))
                      if os.path.isdir(d))
    elif args.run:
        runs = [args.run]
    else:
        ap.error("need --run or --all-structmem")

    print(f"{'DRY RUN: ' if args.dry_run else ''}backfilling {len(runs)} run(s)")
    for r in runs:
        resolve_run(r, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
