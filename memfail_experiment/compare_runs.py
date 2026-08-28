"""MemFail cross-run comparison.

Covers the pipeline writers (Mem0 v1/v2, A-MEM, StructMem), the same Mem0 v1
with a weaker extraction LLM, and agent-managed Letta -- the two variants added
to test whether the storage stage is reachable at all.

Aggregates the official analyze_errors.py output. Attribution comes from that
pipeline's `error_type` column verbatim -- this script never re-judges.

analyze_errors.py assigns each question exactly one error_type by walking the
stages in order and stopping at the first failure, so the counts partition the
incorrect answers rather than overlapping.
"""

import csv
import glob
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("results_5q_mem0v1", "Mem0 v1"),
    ("results_5q_mem0v1_e4b", "v1 (E4B ext)"),
    ("results_5q_letta", "Letta"),
    ("results_5q_mem0v2", "Mem0 v2"),
    ("results_5q_amem", "A-MEM"),
    ("results_5q_structmem", "StructMem"),
    # 第二批(2026-08-19 起)：同樣的資料與參數，adapter 已接上分階段成本量測。
    ("results_cost_mem0v1", "Mem0 v1 cost"),
    ("results_cost_mem0v2", "Mem0 v2 cost"),
    ("results_cost_structmem", "StructMem cost"),
    ("results_cost_amem", "A-MEM cost"),
    ("results_cost_letta", "Letta cost"),
]

DATASETS = [
    ("coexisting_facts", "coexist"),
    ("conditional_easy", "cond-easy"),
    ("conditional_hard", "cond-hard"),
    ("long_hop", "long-hop"),
    ("persona", "persona"),
]

# The four MemFail stages, in the order analyze_errors.py checks them.
STAGES = ["summary", "storage", "retrieval", "reasoning"]

# Column values vary by task; map them onto the shared stage names.
ERR_MAP = {
    "summary_error": "summary",
    "not_stored": "storage",
    "storage_error": "storage",
    "not_retrieved": "retrieval",
    "retrieval_error": "retrieval",
    "reasoning_error": "reasoning",
}


def _latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def read_dataset(root, key):
    """Return dict with n / correct / per-stage counts / store size, or None."""
    an = _latest(os.path.join(BASE_DIR, root, key, "analysis", "analysis_*.csv"))
    if not an:
        return None
    rows = list(csv.DictReader(open(an)))
    if not rows:
        return None

    out = {"n": len(rows), "correct": 0, **{s: 0 for s in STAGES}, "other": 0}
    for r in rows:
        if (r.get("judge_result") or "").strip().lower() == "correct":
            out["correct"] += 1
            continue
        et = (r.get("error_type") or "").strip().lower()
        stage = ERR_MAP.get(et)
        if stage:
            out[stage] += 1
        else:
            out["other"] += 1

    gt = _latest(os.path.join(BASE_DIR, root, key, "*", "graded_traces_*.json"))
    out["store"] = len(json.load(open(gt)).get("all_memories_at_time_of_questions") or []) if gt else None
    return out


def main():
    present = [(r, lab) for r, lab in RUNS if os.path.isdir(os.path.join(BASE_DIR, r))]
    W = 24

    print("acc  +  Summ/Stor/Retr/Reas  (stage counts partition the wrong answers)\n")
    print(f"{'dataset':<11}" + "".join(f"{lab:>{W}}" for _, lab in present))
    print("-" * (11 + W * len(present)))

    tot = {lab: {"n": 0, "correct": 0, "other": 0, **{s: 0 for s in STAGES}} for _, lab in present}

    for key, dlabel in DATASETS:
        line = f"{dlabel:<11}"
        for root, lab in present:
            d = read_dataset(root, key)
            if not d:
                line += f"{'(pending)':>{W}}"
                continue
            cell = f"{d['correct']/d['n']:.2f}  " + "/".join(str(d[s]) for s in STAGES)
            line += f"{cell:>{W}}"
            for k in ("n", "correct", "other", *STAGES):
                tot[lab][k] += d[k]
        print(line)

    print("-" * (11 + W * len(present)))
    line = f"{'TOTAL':<11}"
    for _, lab in present:
        t = tot[lab]
        cell = (f"{t['correct']/t['n']:.2f}  " + "/".join(str(t[s]) for s in STAGES)) if t["n"] else "(pending)"
        line += f"{cell:>{W}}"
    print(line)
    line = f"{'  n':<11}"
    for _, lab in present:
        line += f"{tot[lab]['n']:>{W}}"
    print(line)

    print("\nstore size (memories held at question time)")
    print(f"{'dataset':<11}" + "".join(f"{lab:>{W}}" for _, lab in present))
    for key, dlabel in DATASETS:
        line = f"{dlabel:<11}"
        for root, lab in present:
            gt = _latest(os.path.join(BASE_DIR, root, key, "*", "graded_traces_*.json"))
            v = len(json.load(open(gt)).get("all_memories_at_time_of_questions") or []) if gt else "-"
            line += f"{v:>{W}}"
        print(line)

    stray = {lab: tot[lab]["other"] for _, lab in present if tot[lab]["other"]}
    if stray:
        print(f"\nunmapped error_type values: {stray}")


if __name__ == "__main__":
    main()
