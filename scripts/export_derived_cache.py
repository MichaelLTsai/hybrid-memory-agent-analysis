"""Extract the values the matrix needs that can only be computed from large raw outputs.

`build_matrix_excel.py` has two families of columns that require the
per-question raw output:

  1. Memory-store size: how many entries each run's final store holds, read from
     memory_dump / all_memories / store_after in *_results.jsonl.
  2. HaluMem per-question-type accuracy: the official scores.json reports only
     the overall correct_qa_ratio, so the per-type breakdown must be recomputed
     from *_eval_detail.jsonl.

Those raw files run to tens of megabytes each and are not tracked in git, while
the values derived from them are only a few kilobytes. Run this once on a
machine holding the complete outputs to produce results_derived_cache.json;
build_matrix_excel.py falls back to that cache whenever the raw files are
absent, so a plain clone can still rebuild the complete matrix.

    python scripts/export_derived_cache.py
"""

import json
import os
import statistics
import sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from build_matrix_excel import BACKENDS  # noqa: E402

OUT = os.path.join(BASE, "results_derived_cache.json")


def store_size_raw(dataset, run):
    """Same definition as build_matrix_excel.store_size, but reads raw files only."""
    if dataset == "locomo":
        frame = run.split("-")[0]
        f = os.path.join(BASE, "locomo_experiment", "results", run,
                         f"{frame}_locomo_results.jsonl")
        d = json.loads(open(f, encoding="utf-8").readline()).get("memory_dump")
        return [len(d), None] if d else [None, None]

    if dataset == "longmem":
        frame = run.split("-")[0]
        f = os.path.join(BASE, "longmemeval_experiment", "results", run,
                         f"{frame}_lme_results.jsonl")
        ns = [len(json.loads(l).get("all_memories") or [])
              for l in open(f, encoding="utf-8") if l.strip()]
        return [sum(ns), statistics.median(ns)] if ns else [None, None]

    if dataset == "halumem":
        frame = "mem0_oss" if run.startswith("mem0_oss") else run.split("-")[0]
        f = os.path.join(BASE, "halumem_experiment", "results", run,
                         f"{frame}_eval_results.jsonl")
        us = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        total = 0
        for u in us:
            last = None
            for sess in u["sessions"]:
                if sess.get("store_after") is not None:
                    last = sess["store_after"]
            if last is not None:
                total += len(last)
            else:
                total += sum(len(sess.get("extracted_memories") or [])
                             for sess in u["sessions"])
        return [total, None]

    return [None, None]


def halumem_per_type_raw(run):
    """HaluMem per-question-type accuracy, recomputed from the per-question detail file."""
    frame = "mem0_oss" if run.startswith("mem0_oss") else run.split("-")[0]
    path = os.path.join(BASE, "halumem_experiment", "results", run,
                        f"{frame}_eval_detail.jsonl")
    tot, cor = Counter(), Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "result_type" not in r or "question" not in r:
                continue
            tot[r.get("question_type")] += 1
            if r.get("result_type") == "Correct":
                cor[r.get("question_type")] += 1
    return {t: (cor[t] / tot[t] if tot[t] else None) for t in tot}


def main():
    cache = {"store_size": {}, "halumem_per_type": {}}
    missing = []

    for _name, hal, loc, lme, _mf, _llm in BACKENDS:
        for dataset, run in (("halumem", hal), ("locomo", loc), ("longmem", lme)):
            if not run:
                continue
            key = f"{dataset}/{run}"
            if key in cache["store_size"]:
                continue
            try:
                cache["store_size"][key] = store_size_raw(dataset, run)
            except Exception as e:
                missing.append(f"store_size {key}: {type(e).__name__}")

        if hal and hal not in cache["halumem_per_type"]:
            try:
                cache["halumem_per_type"][hal] = halumem_per_type_raw(hal)
            except Exception as e:
                missing.append(f"per_type {hal}: {type(e).__name__}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1, sort_keys=True)

    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size_kb:.1f} KB)")
    print(f"  store sizes        {len(cache['store_size'])}")
    print(f"  HaluMem per-type   {len(cache['halumem_per_type'])}")
    if missing:
        print(f"  skipped {len(missing)} item(s), raw files unreadable:")
        for m in missing[:10]:
            print(f"    {m}")


if __name__ == "__main__":
    main()
