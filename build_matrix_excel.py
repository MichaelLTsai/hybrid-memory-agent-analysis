#!/usr/bin/env python3
"""
Aggregate the failure-stage matrix over four datasets x five backends
into memory_failure_matrix.xlsx.

Column groups follow the stage decomposition:
    Summary / Storage / Retrieval / Reasoning / Memory Performance
Each group is further split by dataset. A blank cell means either that the run
has not finished or that the cell does not exist structurally; see the
"Definitions" sheet.

Everything is read from each experiment's own scores.json and never recomputed,
so after changing a run it is enough to re-run this script.

    ./venv_memos/bin/python build_matrix_excel.py
"""

import os
import re
import json
from datetime import datetime

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "memory_failure_matrix.xlsx")

# ── Which run corresponds to each backend in each of the four experiments ───
BACKENDS = [
    # (display name, halumem run, locomo run, longmemeval run, memfail run, LLM)
    ("Mem0 v1",   "mem0_oss-v1_31b_u2",              "mem0-v1_31b",      "mem0-v1_31b",      "results_5q_mem0v1",    "gemma-4-31B-it"),
    ("Mem0 v2",   "mem0_oss-v2_31b_u2",              "mem0-v2_31b",      "mem0-v2_31b",      "results_5q_mem0v2",    "gemma-4-31B-it"),
    ("StructMem", "structmem-user2nd_gemma431b_probe","structmem-sm_31b", "structmem-sm_31b", "results_5q_structmem", "gemma-4-31B-it"),
    ("A-MEM",     "amem-user2nd_gemma431b_probe",    "amem-amem_31b",    "amem-amem_31b",    "results_5q_amem",      "gemma-4-31B-it"),
    ("Letta",     "letta-user2nd_gemma431b_probe",   "letta-letta_31b",  "letta-letta_31b",  "results_5q_letta",     "openai-proxy/gemma-4-31B-it"),
    # Same runs, but P4 counts the message history visible at answering time
    # (recall memory) instead of only core blocks plus archival. Letta has three
    # memory tiers, and counting only the latter two underestimates what it can
    # actually see. Both rows are kept side by side: the gap between them is
    # itself a finding. No matching version was run for LongMemEval or MemFail,
    # so those cells stay empty.
    ("Letta (with history)", "letta-u2_ctx",                "letta-ctx_31b",    "",                 "",                     "openai-proxy/gemma-4-31B-it"),

    # ── Batch 2 (2026-08-19) ────────────────────────────────────────────────
    # Same backends and same sampling parameters; the only difference is that the
    # adapters now carry staged cost instrumentation (token_tracker's ingest / qa /
    # other buckets), which is why only this batch has Cost columns. It sits
    # alongside batch 1 rather than replacing it: the score gap between the two is
    # the run-to-run variance under identical parameters, which is itself a result.
    ("Mem0 v1 · 0819",   "mem0_oss-v1_cost_u2",   "mem0-v1_cost",      "mem0-v1_cost",      "results_cost_mem0v1",    "gemma-4-31B-it"),
    ("Mem0 v2 · 0819",   "mem0_oss-v2_cost_u2",   "mem0-v2_cost",      "mem0-v2_cost",      "results_cost_mem0v2",    "gemma-4-31B-it"),
    ("StructMem · 0819", "structmem-sm_cost_u2",  "structmem-sm_cost", "structmem-sm_cost", "results_cost_structmem", "gemma-4-31B-it"),
    ("A-MEM · 0819",     "amem-amem_cost_u2",     "amem-amem_cost",    "amem-amem_cost",    "results_cost_amem",      "gemma-4-31B-it"),
    ("Letta · 0819",     "letta-letta_cost_u2",   "letta-letta_cost",  "letta-letta_cost",  "results_cost_letta",     "openai-proxy/gemma-4-31B-it"),

    # ── Batch 3 (2026-08-19): additional HaluMem samples ────────────────────
    # HaluMem users #3 and #4 only. Motivation: the Dynamic Update question type
    # is very unevenly distributed across users. User #1, used by the first two
    # batches, has only 4 such questions, while #3 has 22 and #4 has 11. This
    # raises the end-to-end evidence for "can it answer correctly after an update"
    # from 4 questions to 37. The memory-point-level memory_update metric already
    # had 123 to 168 samples per user and was never short.
    ("Mem0 v1 · u34",   "mem0_oss-v1_cost_u34",  "", "", "", "gemma-4-31B-it"),
    ("Mem0 v2 · u34",   "mem0_oss-v2_cost_u34",  "", "", "", "gemma-4-31B-it"),
    ("StructMem · u34", "structmem-sm_cost_u34", "", "", "", "gemma-4-31B-it"),
    ("A-MEM · u34",     "amem-amem_cost_u34",    "", "", "", "gemma-4-31B-it"),
    ("Letta · u34",     "letta-letta_cost_u34",  "", "", "", "openai-proxy/gemma-4-31B-it"),
]

# Which batch each row belongs to. Batch 2 uses exactly the same sampling as
# batch 1 (LoCoMo conv-26, the first 3 questions of each LongMemEval type, and
# HaluMem's 2nd user), so the two batches can be compared directly.
BATCH = {"Mem0 v1": "① 08-14", "Mem0 v2": "① 08-14", "StructMem": "① 08-14",
         "A-MEM": "① 08-14", "Letta": "① 08-14", "Letta (with history)": "① 08-14",
         "Mem0 v1 · 0819": "② 08-19", "Mem0 v2 · 0819": "② 08-19",
         "StructMem · 0819": "② 08-19", "A-MEM · 0819": "② 08-19",
         "Letta · 0819": "② 08-19",
         "Mem0 v1 · u34": "③ u3+u4", "Mem0 v2 · u34": "③ u3+u4",
         "StructMem · u34": "③ u3+u4", "A-MEM · u34": "③ u3+u4",
         "Letta · u34": "③ u3+u4"}

FRAME = {"mem0_oss": "mem0_oss", "mem0": "mem0", "structmem": "structmem",
         "amem": "amem", "letta": "letta"}


# Ingestion granularity per backend per dataset: whether one add call covers a
# single turn or a whole session. This is an adapter choice, not an architectural
# property, and the same backend can differ across datasets (mem0 is turn-level on
# LoCoMo but session-level on LongMemEval and HaluMem). It directly affects both
# memory volume and extraction quality, so it must be read alongside any /turn ratio.
GRANULARITY = {
    #  backend      LoCoMo     LongMemEval  HaluMem    MemFail
    "Mem0 v1":   ("turn",    "session",   "session", "turn"),
    "Mem0 v2":   ("turn",    "session",   "session", "turn"),
    "StructMem": ("session", "session",   "session", "turn"),
    "A-MEM":     ("turn",    "turn",      "turn",    "turn"),
    "Letta":     ("session", "session",   "session", "turn"),
    "Letta (with history)": ("session", "session", "session", "turn"),
    "Mem0 v1 · 0819":   ("turn",    "session",   "session", "turn"),
    "Mem0 v2 · 0819":   ("turn",    "session",   "session", "turn"),
    "StructMem · 0819": ("session", "session",   "session", "turn"),
    "A-MEM · 0819":     ("turn",    "turn",      "turn",    "turn"),
    "Letta · 0819":     ("session", "session",   "session", "turn"),
    "Mem0 v1 · u34":   ("turn",    "session",   "session", "turn"),
    "Mem0 v2 · u34":   ("turn",    "session",   "session", "turn"),
    "StructMem · u34": ("session", "session",   "session", "turn"),
    "A-MEM · u34":     ("turn",    "turn",      "turn",    "turn"),
    "Letta · u34":     ("session", "session",   "session", "turn"),
}


# Turns actually fed into the memory systems in this round, used to normalize
# memory volume. Calibration: A-MEM writes exactly one entry per turn, so its
# entries-per-turn should come out to exactly 1.00. It does on all four datasets,
# which confirms these turn counts are right.
TURNS = {
    "locomo":  419,     # conv-26:19 sessions
    "halumem": 3242,    # user #1 / 77 sessions (batches 1 and 2)
    "longmem": 485,     # median haystack turns over the 18 questions (each builds its own store)
    "memfail": 45,      # fact sentences in the storage phase across the 35 questions
}

# Batch 3 runs HaluMem users #3 and #4, totalling 3,210 + 2,960 = 6,170 messages,
# unlike user #1's 3,242. Reusing the old denominator would inflate entries-per-turn
# by a factor of 1.9.
TURNS_BY_BATCH = {"③ u3+u4": {"halumem": 6170}}


def turns_for(batch: str, dataset: str) -> int:
    return TURNS_BY_BATCH.get(batch, {}).get(dataset, TURNS[dataset])


# Entry counts and HaluMem per-question-type accuracy can only be computed from
# the per-question raw output, and those files run to tens of megabytes each, so
# they are not tracked in git. scripts/export_derived_cache.py extracts the computed
# values into a cache of a few kilobytes: when the raw files are present this
# recomputes as before, and when they are absent (a plain clone, for instance) it
# reads the cache. Both paths yield the same numbers.
_CACHE_PATH = os.path.join(BASE, "results_derived_cache.json")
try:
    with open(_CACHE_PATH, encoding="utf-8") as _f:
        DERIVED_CACHE = json.load(_f)
except Exception:
    DERIVED_CACHE = {"store_size": {}, "halumem_per_type": {}}


def store_size(dataset: str, run: str):
    """How many entries this run stored on this dataset. Returns (total, per-question median).

    LongMemEval builds a separate store per question, so the total is a sum over
    18 questions and only the per-question median is a comparable scale.
    """
    import statistics
    try:
        if dataset == "locomo":
            frame = run.split("-")[0]
            f = os.path.join(BASE, "locomo_experiment", "results", run,
                             f"{frame}_locomo_results.jsonl")
            d = json.loads(open(f, encoding="utf-8").readline()).get("memory_dump")
            return (len(d), None) if d else (None, None)
        if dataset == "longmem":
            frame = run.split("-")[0]
            f = os.path.join(BASE, "longmemeval_experiment", "results", run,
                             f"{frame}_lme_results.jsonl")
            ns = [len(json.loads(l).get("all_memories") or [])
                  for l in open(f, encoding="utf-8") if l.strip()]
            return (sum(ns), statistics.median(ns)) if ns else (None, None)
        if dataset == "halumem":
            # Use store_after from the last session (the full store at that point)
            # rather than summing each session's extracted_memories:
            #   - Letta's extracted_memories is a cumulative snapshot, so summing
            #     would double-count by an order of magnitude
            #   - mem0 issues UPDATE/DELETE, so summing additions also overstates
            #     the final store size
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
                else:   # only backends without store_after fall back to summing
                    total += sum(len(sess.get("extracted_memories") or [])
                                 for sess in u["sessions"])
            return (total, None)
        if dataset == "memfail":
            return (memfail(run).get("mf_store"), None)
    except Exception:
        pass
    cached = DERIVED_CACHE.get("store_size", {}).get(f"{dataset}/{run}")
    if cached:
        return tuple(cached)
    return (None, None)


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def halumem(run):
    d = _load(os.path.join(BASE, "halumem_experiment", "results", run, "mem0_oss_scores.json"))
    if d is None:
        frame = run.split("-")[0]
        d = _load(os.path.join(BASE, "halumem_experiment", "results", run, f"{frame}_scores.json"))
    if not d:
        return {}
    mu = d.get("memory_update", {})
    qa = d.get("question_answering", {})
    at = d.get("qa_attribution") or {}
    mi = d.get("memory_integrity", {})
    ma = d.get("memory_accuracy", {})
    pr = d.get("probe") or {}
    return {
        "hal_f1":        d.get("memory_extraction_f1"),
        "hal_integrity": mi.get("recall(all)"),
        "hal_target":    ma.get("target_accuracy(all)"),
        "hal_interf":    ma.get("interference_accuracy(all)"),
        "hal_update":    mu.get("correct_update_memory_ratio(all)"),
        "hal_upd_omis":  mu.get("omission_update_memory_ratio(all)"),
        "hal_upd_hall":  mu.get("hallucination_update_memory_ratio(all)"),
        "hal_p4":        at.get("retrieval_ratio"),
        "hal_p5":        at.get("generation_ratio"),
        "hal_storage":   at.get("storage_ratio"),
        "hal_unknown":   at.get("unknown_ratio"),
        "hal_qa":        qa.get("correct_qa_ratio(all)"),
        "hal_qa_hall":   qa.get("hallucination_qa_ratio(all)"),
        "hal_qa_omis":   qa.get("omission_qa_ratio(all)"),
        "hal_n":         qa.get("qa_num"),
        # The four update categories are denominated over all update points, but
        # only some are successfully adjudicated, so the (all) figures sum to less
        # than 1. The (valid) figures give the distribution over adjudicated ones.
        "hal_upd_valid":   mu.get("correct_update_memory_ratio(valid)"),
        "hal_upd_judged":  (mu.get("update_memory_valid_num") / mu["update_memory_num"]
                            if mu.get("update_memory_num") else None),
        "hal_upd_n":       mu.get("update_memory_num"),
        # Stage-capability probes (probe_halumem.py, denominator = all questions)
        "hal_p4_suff":   pr.get("P4_sufficient"),
        "hal_p5_fail":   pr.get("P5_fail_given_P4"),
        "hal_p5_n":      pr.get("P5_n"),
        "hal_probe_unk": pr.get("unknown_ratio"),
    }


def locomo(run):
    frame = run.split("-")[0]
    d = _load(os.path.join(BASE, "locomo_experiment", "results", run, f"{frame}_locomo_scores.json"))
    if not d:
        return {}
    e = d.get("extraction") or {}
    p = d.get("probe") or {}
    r = d.get("retrieval") or {}
    if "skipped" in e:
        e = {}
    return {
        "loc_p1":      e.get("memory_integrity_recall"),
        "loc_acc":     e.get("memory_accuracy_precision"),
        "loc_f1":      e.get("memory_extraction_f1"),
        "loc_spk":     e.get("speaker_confusion_ratio"),
        "loc_recall":  r.get("recall@5"),
        "loc_ndcg":    r.get("ndcg@5"),
        "loc_p4":      p.get("P4_sufficient"),
        "loc_p1scope": p.get("P1_scoped_sufficient"),
        "loc_p5":      p.get("P5_fail_given_P4"),
        "loc_p5_n":    p.get("P5_n"),
        "loc_qa":      d.get("qa_accuracy_all"),
        "loc_tf1":     d.get("token_f1_all"),
        "loc_n":       d.get("qa_num"),
        "loc_attr_sum":  (p.get("attribution") or {}).get("SUMMARY"),
        "loc_attr_retr": (p.get("attribution") or {}).get("RETRIEVAL"),
        "loc_attr_reas": (p.get("attribution") or {}).get("REASONING"),
    }


def longmem(run):
    frame = run.split("-")[0]
    d = _load(os.path.join(BASE, "longmemeval_experiment", "results", run, f"{frame}_lme_scores.json"))
    if not d:
        return {}
    p = d.get("probe") or {}
    r = d.get("retrieval") or {}
    per = d.get("per_type") or {}
    ku = (per.get("knowledge-update") or {}).get("accuracy")
    return {
        "lme_p1":     p.get("P1_sufficient"),
        "lme_p4":     p.get("P4_sufficient"),
        "lme_p5":     p.get("P5_fail_given_P4"),
        "lme_p5_n":   p.get("P5_n"),
        "lme_ku":     ku,
        "lme_recall": r.get("recall@5"),
        "lme_ndcg":   r.get("ndcg@5"),
        "lme_qa":     d.get("qa_accuracy_all"),
        "lme_n":      d.get("qa_num"),
        "lme_attr_sum":  (p.get("attribution") or {}).get("SUMMARY"),
        "lme_attr_retr": (p.get("attribution") or {}).get("RETRIEVAL"),
        "lme_attr_reas": (p.get("attribution") or {}).get("REASONING"),
    }


# ── Cost and latency ───────────────────────────────────────────────────────
# Source is each run's {frame}_token_usage.json, measured in place by
# halumem_experiment/token_tracker.py while the experiment runs (not estimated
# afterwards). It sorts LLM calls into three buckets:
#     ingest  memory writes (extraction and update as each turn/session is fed in)
#     qa      answering (retrieval + generation)
#     other   judge / probes / warm-up, explicitly excluded, so the cost figures
#             carry no evaluation overhead
#
# Runs executed before 2026-08-18 recorded a single total_tokens with no
# bucketing. That legacy format always returns None and the table shows
# "legacy format" as a reminder that a re-run is needed to get figures.
_COST_FILE = {
    "longmem": ("longmemeval_experiment", "{frame}_lme"),
    "locomo":  ("locomo_experiment",      "{frame}_locomo"),
    "halumem": ("halumem_experiment",     "{frame}"),
}


def cost(dataset: str, run: str, prefix: str):
    if not run:
        return {}
    try:
        if dataset == "memfail":
            # MemFail writes output_dir under playground as results/<memory>/
            import glob
            hits = glob.glob(os.path.join(BASE, "memfail_experiment", "**",
                                          "memfail_token_usage.json"), recursive=True)
            if not hits:
                return {}
            path = max(hits, key=os.path.getmtime)
        else:
            sub, pat = _COST_FILE[dataset]
            frame = ("mem0_oss" if run.startswith("mem0_oss") else run.split("-")[0])
            path = os.path.join(BASE, sub, "results", run,
                                pat.format(frame=frame) + "_token_usage.json")
        d = _load(path)
        if not d:
            return {}
        ing, qa = d.get("ingest"), d.get("qa")
        if not isinstance(ing, dict) or not isinstance(qa, dict):
            return {f"{prefix}_cost_old": "legacy format"}       # legacy run, no buckets
        return {
            f"{prefix}_in_calls": ing.get("calls_per_unit"),
            f"{prefix}_in_tok":   ing.get("tokens_per_unit"),
            f"{prefix}_in_sec":   ing.get("sec_per_unit_median"),
            f"{prefix}_qa_calls": qa.get("calls_per_unit"),
            f"{prefix}_qa_tok":   qa.get("tokens_per_unit"),
            f"{prefix}_qa_sec":   qa.get("sec_per_unit_median"),
        }
    except Exception:
        return {}


def memfail(run):
    """Read this run's TOTAL row from the experiment_results.md that compare_runs writes.

    Column order: Run | Dataset | Q# | Correct | Acc | Storage | Summary | Retr | Reason | Store
    """
    md = os.path.join(BASE, "memfail_experiment", "experiment_results.md")
    try:
        for line in open(md, encoding="utf-8"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 10 and c[0] == run and c[1] == "TOTAL":
                n = int(c[2])
                return {
                    "mf_n":       n,
                    "mf_correct": float(c[4]),
                    "mf_storage": int(c[5]) / n if n else None,
                    "mf_summary": int(c[6]) / n if n else None,
                    "mf_retr":    int(c[7]) / n if n else None,
                    "mf_reason":  int(c[8]) / n if n else None,
                    "mf_store":   int(c[9]),
                }
    except Exception:
        pass
    return {}


# ── Column definitions: (group, header, key, source description) ────────────
COLUMNS = [
    ("",          "Run name",   "run_name", "Row label in this table"),
    ("",          "Backend",    "backend",  "Memory architecture"),
    ("",          "LLM",        "llm",      "Extraction LLM (judge is always gemma-4-E4B-it)"),
    ("",          "Batch",       "batch",    "(1) = original batch, 2026-08-14; (2) = re-run batch, 2026-08-19 (only this one has cost columns)"),

    # Within every group the dataset order is fixed: LongMemEval, LoCoMo,
    # HaluMem, MemFail. Header markers: ↑ higher is better, ↓ lower is better.
    # Differences in denominators are documented in the Definitions sheet.
    ("Summary",   "LongMemEval P1 ↑",      "lme_p1",  "probe_longmem.py; judged only on questions where P4 failed, so the denominator is small"),
    ("Summary",   "LoCoMo P1 recall ↑",    "loc_p1",  "extraction_locomo.py; observations treated as golden memories, strict=2"),
    ("Summary",   "LoCoMo precision ↑",    "loc_acc", "Precision counterpart of the above (whether what was recorded is correct)"),
    ("Summary",   "LoCoMo F1 ↑",           "loc_f1",  "Harmonic mean of the two above"),
    ("Summary",   "HaluMem recall ↑",      "hal_integrity", "Official memory_integrity recall (all)"),
    ("Summary",   "HaluMem precision ↑",   "hal_target",    "Official target_accuracy (all); this is the precision used by f1"),
    ("Summary",   "HaluMem F1 ↑",          "hal_f1",  "Official memory_extraction_f1"),
    ("Summary",   "HaluMem interference ↑","hal_interf","Official interference_accuracy (all); correct-rejection rate on distractors, not part of f1"),
    ("Summary",   "MemFail summary_error ↓","mf_summary","Official analyze_errors summary_error as a share of all questions"),

    ("Storage",   "LongMemEval KU acc ↑",  "lme_ku",     "QA accuracy on the knowledge-update subset (end-to-end, not a storage-specific metric)"),
    ("Storage",   "HaluMem update ↑",      "hal_update", "Official correct_update_memory_ratio (all); omission and hallucination are its subcategories and are not listed separately"),
    ("Storage",   "MemFail storage_error ↓","mf_storage","Official analyze_errors not_stored as a share of all questions"),

    ("Retrieval", "LongMemEval P4 ↑",      "lme_p4",    "probe_longmem.py; whether the retrieved context suffices to answer (value level)"),
    ("Retrieval", "LongMemEval Recall@5 ↑","lme_recall","Official retrieval metric (session level)"),
    ("Retrieval", "LongMemEval NDCG@5 ↑",  "lme_ndcg",  "Official retrieval metric (session level)"),
    ("Retrieval", "LoCoMo P4 ↑",           "loc_p4",    "probe_locomo.py; same definition"),
    ("Retrieval", "LoCoMo Recall@5 ↑",     "loc_recall","Official retrieval metric (turn level, exact dia_id match)"),
    ("Retrieval", "LoCoMo NDCG@5 ↑",       "loc_ndcg",  "Official retrieval metric (turn level)"),
    ("Retrieval", "HaluMem P4 ↑",          "hal_p4_suff","probe_halumem.py; same definition, run over all 188 questions"),
    ("Retrieval", "MemFail retr_error ↓",  "mf_retr",   "Official analyze_errors not_retrieved as a share of all questions"),

    # P5 = share of questions that passed P4 (the answer really was in the
    # retrieved context) yet were still answered wrong; the denominator is the
    # P4-passing questions. The definition is the same across all four datasets.
    # Denominator sizes vary a lot (LongMemEval has only 2 to 14), so read these
    # against the question counts in the Scale group. Per-metric denominators are
    # documented in the Definitions sheet.
    ("Reasoning", "LongMemEval P5 fail ↓", "lme_p5",     "probe_longmem.py; share of P4-passing questions still answered wrong"),
    ("Reasoning", "LoCoMo P5 fail ↓",      "loc_p5",     "probe_locomo.py; same definition"),
    ("Reasoning", "HaluMem P5 fail ↓",     "hal_p5_fail","probe_halumem.py; same definition"),
    ("Reasoning", "MemFail reason_error ↓","mf_reason",  "Official analyze_errors reasoning_error as a share of all questions"),

    ("Memory Performance", "LongMemEval QA ↑", "lme_qa", "Official LLM-judge accuracy"),
    ("Memory Performance", "LoCoMo QA ↑",      "loc_qa", "Official LLM-judge accuracy"),
    ("Memory Performance", "LoCoMo token F1 ↑","loc_tf1","Metric from the original LoCoMo paper"),
    ("Memory Performance", "HaluMem QA ↑",     "hal_qa", "Official correct_qa_ratio (all)"),
    ("Memory Performance", "MemFail correct ↑","mf_correct","Official overall accuracy"),

    ("Memory volume", "LME granularity",      "lme_gran",      "Whether one add call covers a turn or a whole session"),
    ("Memory volume", "LME entries/question",   "lme_store_med", "Each question builds its own store; median over the 18 questions"),
    ("Memory volume", "LME /turn",       "lme_store_pt",  "Median entry count / 485 (median haystack turns)"),
    ("Memory volume", "LME entries total",  "lme_store",     "Sum over the 18 per-question stores (scale reference only, not comparable across backends)"),
    ("Memory volume", "LoCoMo granularity",   "loc_gran",      "Whether one add call covers a turn or a whole session"),
    ("Memory volume", "LoCoMo entries",   "loc_store",     "Entries written for conv-26 (419 turns)"),
    ("Memory volume", "LoCoMo /turn",    "loc_store_pt",  "Entries / 419"),
    ("Memory volume", "HaluMem granularity",  "hal_gran",      "Whether one add call covers a turn or a whole session"),
    ("Memory volume", "HaluMem entries",  "hal_store",     "Total entries written for 1 user / 77 sessions (3,242 turns)"),
    ("Memory volume", "HaluMem /turn",   "hal_store_pt",  "Entries / 3,242"),
    ("Memory volume", "MemFail granularity",  "mf_gran",       "Whether one add call covers a turn or a session (fixed by the benchmark harness)"),
    ("Memory volume", "MemFail entries",  "mf_store",      "Store column from the official compare_runs output"),
    ("Memory volume", "MemFail /turn",   "mf_store_pt",   "Entries / 45"),

    # Cost: measured in place by token_tracker.py while the experiment runs, with
    # judge and probe overhead excluded. A "unit" is the span covered by one add
    # call (turn or session, see the granularity columns), so calls/unit must be
    # read together with granularity: A-MEM's 1.0 is per turn, StructMem's is per
    # session.
    ("Cost", "LME ingest calls/unit",  "lme_in_calls", "LLM calls needed to ingest one unit"),
    ("Cost", "LME ingest tok/unit",    "lme_in_tok",   "Tokens to ingest one unit (prompt + completion)"),
    ("Cost", "LME ingest sec/unit",     "lme_in_sec",   "End-to-end seconds to ingest one unit (median)"),
    ("Cost", "LME answer calls/question",    "lme_qa_calls", "LLM calls to answer one question (retrieval side included)"),
    ("Cost", "LME answer tok/question",      "lme_qa_tok",   "Tokens to answer one question"),
    ("Cost", "LME answer sec/question",       "lme_qa_sec",   "End-to-end seconds to answer one question (median)"),
    ("Cost", "LoCoMo ingest calls/unit","loc_in_calls", "As above"),
    ("Cost", "LoCoMo ingest tok/unit",  "loc_in_tok",   "As above"),
    ("Cost", "LoCoMo ingest sec/unit",   "loc_in_sec",   "As above"),
    ("Cost", "LoCoMo answer calls/question",  "loc_qa_calls", "As above"),
    ("Cost", "LoCoMo answer tok/question",    "loc_qa_tok",   "As above"),
    ("Cost", "LoCoMo answer sec/question",     "loc_qa_sec",   "As above"),
    ("Cost", "HaluMem ingest calls/unit","hal_in_calls","As above"),
    ("Cost", "HaluMem ingest tok/unit",  "hal_in_tok",  "As above"),
    ("Cost", "HaluMem ingest sec/unit",   "hal_in_sec",  "As above"),
    ("Cost", "HaluMem answer calls/question",  "hal_qa_calls","As above"),
    ("Cost", "HaluMem answer tok/question",    "hal_qa_tok",  "As above"),
    ("Cost", "HaluMem answer sec/question",     "hal_qa_sec",  "As above"),
    ("Cost", "MemFail ingest calls/unit","mf_in_calls", "MemFail is always turn level"),
    ("Cost", "MemFail ingest tok/unit",  "mf_in_tok",   "As above"),
    ("Cost", "MemFail ingest sec/unit",   "mf_in_sec",   "As above"),
    ("Cost", "MemFail answer calls/question",  "mf_qa_calls", "As above"),
    ("Cost", "MemFail answer tok/question",    "mf_qa_tok",   "As above"),
    ("Cost", "MemFail answer sec/question",     "mf_qa_sec",   "As above"),

    ("Scale", "LME questions",     "lme_n",     "3 questions per type"),
    ("Scale", "LME haystack", "lme_scale", "Haystack size per question"),
    ("Scale", "LoCoMo",      "loc_scale", "conversation / session / turn"),
    ("Scale", "LoCoMo questions",  "loc_n",     "All QA in that conversation"),
    ("Scale", "HaluMem",     "hal_scale", "user / session / turn"),
    ("Scale", "HaluMem questions", "hal_n",     "All QA for that user"),
    ("Scale", "HaluMem update points","hal_upd_n","Number of updated memory points"),
    ("Scale", "MemFail questions", "mf_n",      "Question count after sampling the 5 subsets"),

    ("Quality", "LoCoMo speaker confusion ↓","loc_spk","Fact recorded correctly but attributed to the wrong speaker (specific to LoCoMo two-party dialogue)"),
    ("Quality", "HaluMem update judged rate ↑","hal_upd_judged","Successfully judged / all update points; below 1 means update is underestimated"),
    ("Quality", "HaluMem attribution unknown ↓","hal_unknown","Share that qa_attribution could not adjudicate"),
    ("Quality", "HaluMem probe unknown ↓","hal_probe_unk","Share that probe_halumem could not adjudicate"),
]

# ── Subset scores ───────────────────────────────────────────────────────────
# Each dataset's official metric broken out by subset. Subsets are ordered by
# purpose group so that members of a group sit adjacent, and the group code is
# printed in the column header: 1 single-point recall / 2a chained / 2b parallel /
# 2c mixed / 3 temporal / 4 post-update / 5 abstention. The grouping dimension is
# what kind of memory a correct answer demands, not the shape of the answer.
PURPOSE_GROUPS = [
    ("1",  "Single-point recall",
     "One memory entry suffices; the answer is a value held directly in it."),
    ("2a", "Multi-memory, chained",
     "Entries depend on one another (A->B->C) and a missing link breaks the chain. "
     "MemFail's conditional_hard belongs here: the generator deliberately splits "
     "each rule across three non-adjacent sentences, so the condition can only be "
     "recovered by combining them."),
    ("2b", "Multi-memory, parallel",
     "Several mutually independent entries must all be present; missing one leaves "
     "the answer incomplete. The typical failure is extraction merging them into a "
     "single superordinate concept, or deduplication treating them as one entry so "
     "that they overwrite each other."),
    ("2c", "Multi-memory, mixed",
     "Within one official subset, chained, parallel, and redundant relations all "
     "coexist, and the official labels do not distinguish them, so no finer split "
     "is possible. HaluMem's Multi-hop Inference and Generalization & Application "
     "cannot even be separated from each other: their evidence-count distributions "
     "overlap and their answer vocabulary coverage differs by only 6 percentage "
     "points. Note: redundant means two or more evidence items are attached even "
     "though one would suffice."),
    ("3",  "Temporal reasoning",
     "A correct answer requires the memory to retain timestamps (date differences, "
     "ordering, most recent occurrence). Dropping time information during "
     "extraction guarantees failure."),
    ("4",  "Post-update value",
     "The same fact has been updated and the latest value must be returned. This "
     "sits at the Storage stage: whether the stale value was removed or superseded."),
    ("5",  "Abstention and correction",
     "The correct behavior is not to answer but to admit the information is absent "
     "from memory, or to point out that the question rests on a false premise. The "
     "scoring logic differs from the other groups: it judges whether the system "
     "correctly avoided answering, not whether it produced some value, so scores in "
     "this group should not be pooled with the others."),
]

_SUBSETS = [
    # (key prefix, subset name, display name, purpose group code)
    ("lme_sub_", "single-session-user",          "single-session-user",      "1"),
    ("lme_sub_", "single-session-assistant",     "single-session-assistant", "1"),
    ("lme_sub_", "single-session-preference",    "single-session-preference","1"),
    ("lme_sub_", "multi-session",                "multi-session",            "2b"),
    ("lme_sub_", "temporal-reasoning",           "temporal-reasoning",       "3"),
    ("lme_sub_", "knowledge-update",             "knowledge-update",         "4"),
    ("loc_sub_", "single_hop",                   "cat4 single_hop",          "1"),
    ("loc_sub_", "multi_hop",                    "cat1 multi_hop",           "2c"),
    ("loc_sub_", "open_domain",                  "cat3 open_domain",         "2c"),
    ("loc_sub_", "temporal",                     "cat2 temporal",            "3"),
    ("loc_sub_", "adversarial",                  "cat5 adversarial",         "5"),
    ("hal_sub_", "Basic Fact Recall",            "Basic Fact Recall",        "1"),
    ("hal_sub_", "Multi-hop Inference",          "Multi-hop Inference",      "2c"),
    ("hal_sub_", "Generalization & Application", "Generalization & App.",    "2c"),
    ("hal_sub_", "Dynamic Update",               "Dynamic Update",           "4"),
    ("hal_sub_", "Memory Boundary",              "Memory Boundary",          "5"),
    ("hal_sub_", "Memory Conflict",              "Memory Conflict",          "5"),
    ("mf_sub_",  "persona_retrieval",            "persona_retrieval",        "1"),
    ("mf_sub_",  "conditional_easy",             "conditional_easy",         "1"),
    ("mf_sub_",  "long_hop",                     "long_hop",                 "2a"),
    ("mf_sub_",  "conditional_hard",             "conditional_hard",         "2a"),
    ("mf_sub_",  "coexisting_facts",             "coexisting_facts",         "2b"),
]
_SRC = {"lme_sub_": "Official per_type accuracy",
        "loc_sub_": "Official per_category accuracy",
        "hal_sub_": "Official output gives only the overall value; per-type is recomputed from the per-question file (the overall value matches the official one)",
        "mf_sub_":  "Acc column from the official compare_runs output"}
for _p, _k, _disp, _g in _SUBSETS:
    COLUMNS.append(("Subset scores", f"[{_g}] {_disp} \u2191", _p + _k, _SRC[_p]))
# LoCoMo token F1: the original paper's headline metric, also present in the
# official per_category output, listed separately below
for _p, _k, _disp, _g in _SUBSETS:
    if _p == "loc_sub_":
        COLUMNS.append(("Subset scores", f"[{_g}] {_disp} F1 \u2191", "loc_subf1_" + _k,
                        "Official per_category token_f1 (blank for cat5, which has no answer column)"))

GROUP_FILL = {
    "":                    "F2F5F8",
    "Summary":             "E8F3F0",
    "Storage":             "E8EEF9",
    "Retrieval":           "F8F0DE",
    "Reasoning":           "F5E8F0",
    "Memory Performance":  "E6E6E6",
    "Memory volume":               "E8F0FA",
    "Cost":                 "FDF2E9",
    "Subset scores":             "EDF3EA",
    "Scale":                 "F0F2F5",
    "Quality":                 "FBEEE6",
}
GROUP_FONT = {
    "Summary": "16685A", "Storage": "2A4A8E", "Retrieval": "87590C",
    "Reasoning": "782D58", "Memory Performance": "333333", "": "141E27",
    "Memory volume": "1A4F8A", "Scale": "5C6B7A", "Quality": "8A5A2B", "Cost": "9C4221", "Subset scores": "3D6B2E",
}


def run_done(dataset: str, run: str) -> bool:
    """Whether this run's scores.json exists. Absent means the run has not finished
    (queued or in progress). An empty run name means the combination has no
    corresponding run at all; that counts as complete and renders as a dash.
    """
    if not run:
        return True
    if dataset == "halumem":
        rd = os.path.join(BASE, "halumem_experiment", "results", run)
        frame = "mem0_oss" if run.startswith("mem0_oss") else run.split("-")[0]
        return os.path.exists(os.path.join(rd, f"{frame}_scores.json"))
    if dataset == "locomo":
        frame = run.split("-")[0]
        return os.path.exists(os.path.join(BASE, "locomo_experiment", "results", run,
                                           f"{frame}_locomo_scores.json"))
    if dataset == "longmem":
        frame = run.split("-")[0]
        return os.path.exists(os.path.join(BASE, "longmemeval_experiment", "results", run,
                                           f"{frame}_lme_scores.json"))
    if dataset == "memfail":
        return bool(memfail(run))
    return False


# Column-key prefix to dataset, used to tell a blank cell that is "still running"
# from one that does not exist structurally
KEY_DATASET = {"hal_": "halumem", "loc_": "locomo", "lme_": "longmem", "mf_": "memfail"}


def collect():
    rows = []
    for name, hal, loc, lme, mf, llm in BACKENDS:
        r = {"run_name": f"{name} · gemma4-31B", "backend": name, "llm": llm,
             "batch": BATCH.get(name, "")}
        r.update(halumem(hal)); r.update(locomo(loc))
        r.update(longmem(lme)); r.update(memfail(mf))
        r["_done"] = {"halumem": run_done("halumem", hal), "locomo": run_done("locomo", loc),
                      "longmem": run_done("longmem", lme), "memfail": run_done("memfail", mf)}
        # Store size: necessary context for reading recall and precision, since
        # writing more entries makes high recall easier by construction
        r["loc_store"], _                  = store_size("locomo", loc)
        r["lme_store"], r["lme_store_med"] = store_size("longmem", lme)
        r["hal_store"], _                  = store_size("halumem", hal)
        r["mf_store"], _                   = store_size("memfail", mf)
        # Entries per turn: dataset scales differ, so only this ratio is
        # comparable across datasets
        for k, ds in [("loc_store", "locomo"), ("hal_store", "halumem"),
                      ("mf_store", "memfail")]:
            v = r.get(k)
            r[k + "_pt"] = (v / turns_for(r["batch"], ds)) if isinstance(v, (int, float)) else None
        r["lme_store_pt"] = (r["lme_store_med"] / TURNS["longmem"]
                             if isinstance(r.get("lme_store_med"), (int, float)) else None)
        # Subset scores: the official metric broken out per subset (column
        # definitions in SUBSET_COLS)
        r.update(subset_scores(name, hal, loc, lme, mf))
        # Cost and latency: only runs executed after 2026-08-18 carry staged data
        r.update(cost("longmem", lme, "lme")); r.update(cost("locomo", loc, "loc"))
        r.update(cost("halumem", hal, "hal")); r.update(cost("memfail", mf, "mf"))
        g = GRANULARITY.get(name, ("?",) * 4)
        r["loc_gran"], r["lme_gran"], r["hal_gran"], r["mf_gran"] = g
        # Scale: question counts alone do not convey workload, so user / session /
        # turn counts are added
        r["hal_scale"] = "1 user / 77 sessions / 3,242 turns"
        r["loc_scale"] = "1 conv (conv-26) / 19 sessions / 419 turns"
        r["lme_scale"] = "about 50 sessions, 491 turns per question"
        rows.append(r)
    return rows


# Key prefix to dataset, used to split out the per-dataset sheets
KEY_PREFIX = {"lme_": "LongMemEval", "loc_": "LoCoMo", "hal_": "HaluMem", "mf_": "MemFail"}
SHEET_ORDER = ["LongMemEval", "LoCoMo", "HaluMem", "MemFail"]


# ── Subset breakdown ────────────────────────────────────────────────────────
# The 22 official subsets mapped to purpose groups by what kind of memory a
# correct answer demands. The grouping dimension is uniformly the memory
# requirement (how many entries, whether timestamps are needed, whether the latest
# value is needed) rather than the shape of the answer; whether external world
# knowledge or arithmetic is required are cross-cutting attributes and do not
# define groups.
#   1   single-point recall      one entry suffices
#   2a  multi-memory, chained    entries depend on one another; a missing link breaks it
#   2b  multi-memory, parallel   several independent entries must all be present
#   2c  multi-memory, mixed      chained/parallel/redundant coexist within one subset
#                                and the official labels cannot separate them
#   3   temporal reasoning       memory must retain timestamps
#   4   post-update value        the stale value must be removed or superseded
#   5   abstention and correction  absent from memory, or the question's premise is false
SUBSET_GROUP = {
    # LongMemEval: all three single-session-* types have answer_session == 1,
    # so all are single-point
    "single-session-user": "1 single-point recall",
    "single-session-assistant": "1 single-point recall",
    "single-session-preference": "1 single-point recall",
    "multi-session": "2b multi-memory parallel",
    "temporal-reasoning": "3 temporal reasoning",
    "knowledge-update": "4 post-update value",
    # LoCoMo
    "single_hop": "1 single-point recall",
    "multi_hop": "2c multi-memory mixed",
    "open_domain": "2c multi-memory mixed",
    "temporal": "3 temporal reasoning",
    "adversarial": "5 abstention and correction",
    # HaluMem: the official Multi-hop and Generalization labels cannot be
    # separated from each other (their evidence counts overlap and answer
    # vocabulary coverage differs by only 6 percentage points), so both go to 2c
    "Basic Fact Recall": "1 single-point recall",
    "Multi-hop Inference": "2c multi-memory mixed",
    "Generalization & Application": "2c multi-memory mixed",
    "Dynamic Update": "4 post-update value",
    "Memory Boundary": "5 abstention and correction",
    "Memory Conflict": "5 abstention and correction",
    # MemFail: the generator deliberately splits each conditional_hard rule across
    # three non-adjacent sentences, so it requires cross-sentence composition and
    # belongs to the chained group rather than single-point
    "persona_retrieval": "1 single-point recall",
    "conditional_easy": "1 single-point recall",
    "long_hop": "2a multi-memory chained",
    "conditional_hard": "2a multi-memory chained",
    "coexisting_facts": "2b multi-memory parallel",
}

# The experiments name their verdicts differently; normalize to five categories
# before comparing
VERDICT_MAP = {"OK": "OK", "SUMMARY": "SUMMARY", "NO_WRITE": "STORAGE",
               "RETRIEVAL": "RETRIEVAL", "NOT_RETRIEVED": "RETRIEVAL",
               "REASONING": "REASONING"}
VERDICT_COLS = ["OK", "SUMMARY", "STORAGE", "RETRIEVAL", "REASONING", "Other"]

LOCOMO_CAT = {"1": "multi_hop", "2": "temporal", "3": "open_domain",
              "4": "single_hop", "5": "adversarial"}


def _norm_verdicts(counts):
    from collections import Counter
    o = Counter()
    for k, v in (counts or {}).items():
        o[VERDICT_MAP.get(k, "Other")] += v
    return o


def subset_scores(name, hal, loc, lme, mf):
    """Official metrics for this backend on every subset of the four datasets.

    Returns {column_key: score}.

    Official metrics only: QA accuracy for LongMemEval and HaluMem, accuracy plus
    token F1 for LoCoMo, accuracy for MemFail. Retrieval (recall@k), extraction
    (P1), and update metrics are stored as overall values only in all four
    experiments, with no subset-level breakdown available.
    """
    out = {}

    # ── LongMemEval ── official per_type.accuracy
    if lme:
        frame = lme.split("-")[0]
        d = _load(os.path.join(BASE, "longmemeval_experiment", "results", lme,
                               f"{frame}_lme_scores.json")) or {}
        for t, st in (d.get("per_type") or {}).items():
            out[f"lme_sub_{t}"] = st.get("accuracy")

    # ── LoCoMo ── official per_category carries both accuracy and token_f1
    if loc:
        frame = loc.split("-")[0]
        d = _load(os.path.join(BASE, "locomo_experiment", "results", loc,
                               f"{frame}_locomo_scores.json")) or {}
        for t, st in (d.get("per_category") or {}).items():
            out[f"loc_sub_{t}"] = st.get("accuracy")
            out[f"loc_subf1_{t}"] = st.get("token_f1")

    # ── HaluMem ── the official output gives only the overall correct_qa_ratio,
    #    so per-type figures are recomputed from the per-question file. The
    #    recomputed overall value has been verified to match the official one.
    if hal:
        from collections import Counter
        frame = "mem0_oss" if hal.startswith("mem0_oss") else hal.split("-")[0]
        tot, cor = Counter(), Counter()
        try:
            path = os.path.join(BASE, "halumem_experiment", "results", hal,
                                f"{frame}_eval_detail.jsonl")
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
        except Exception:
            pass
        if tot:
            for t in tot:
                out[f"hal_sub_{t}"] = cor[t] / tot[t] if tot[t] else None
        else:   # per-question detail absent (a plain clone) -> read the cache
            for t, acc in (DERIVED_CACHE.get("halumem_per_type", {}).get(hal) or {}).items():
                out[f"hal_sub_{t}"] = acc

    # ── MemFail ── the md table compare_runs writes, one row per subset
    if mf:
        try:
            md = os.path.join(BASE, "memfail_experiment", "experiment_results.md")
            for line in open(md, encoding="utf-8"):
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                if len(c) >= 10 and c[0] == mf and c[1] != "TOTAL":
                    out[f"mf_sub_{c[1]}"] = float(c[4])
        except Exception:
            pass
    return out

def dataset_of(key: str):
    """Which dataset a column key belongs to. The first three columns
    (run_name / backend / llm) belong to none and return None."""
    for pre, ds in KEY_PREFIX.items():
        if key.startswith(pre):
            return ds
    return None


def columns_for(ds: str):
    """Columns for a per-dataset sheet: the shared columns plus that dataset's own,
    in the order they appear in COLUMNS."""
    return [c for c in COLUMNS if dataset_of(c[2]) in (None, ds)]


def write_sheet(ws, columns, rows):
    """Write one set of columns as a sheet: two header rows, merged group headers,
    best value in bold, and RUNNING markers."""
    thin = Side(style="thin", color="C8D0D8")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Two header rows: group, then column name
    for ci, (grp, title, key, _) in enumerate(columns, 1):
        g = ws.cell(row=1, column=ci, value=grp)
        g.fill = PatternFill("solid", fgColor=GROUP_FILL.get(grp, "FFFFFF"))
        g.font = Font(bold=True, size=10, color=GROUP_FONT.get(grp, "000000"))
        g.alignment = Alignment(horizontal="center", vertical="center")
        g.border = border

        h = ws.cell(row=2, column=ci, value=title)
        h.fill = PatternFill("solid", fgColor=GROUP_FILL.get(grp, "FFFFFF"))
        h.font = Font(bold=True, size=9, color=GROUP_FONT.get(grp, "000000"))
        h.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        h.border = border
        ws.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = \
            max(11, min(20, len(title) + 3))

    # Merge the group header cells
    start = 1
    for ci in range(2, len(columns) + 2):
        cur = columns[ci - 1][0] if ci <= len(columns) else None
        prev = columns[start - 1][0]
        if cur != prev:
            if ci - 1 > start:
                ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=ci - 1)
            start = ci

    # Best value per column: max for ↑ columns, min for ↓ columns. Ties are all
    # bolded. Best values are only compared within a comparable group. Batches 1
    # and 2 ran the same data (HaluMem user #1, LoCoMo conv-26) so comparing them
    # is meaningful. Batch 3 ran users #3 and #4, a completely different set of
    # 360 questions, so ranking it against batches 1 and 2 would be wrong and it
    # forms its own group.
    def _cmp_group(row):
        return "u34" if str(row.get("batch", "")).startswith("③") else "main"

    best = {}
    for ci, (grp, title, key, _) in enumerate(columns, 1):
        if "↑" not in title and "↓" not in title:
            continue
        for g in {_cmp_group(r) for r in rows}:
            vals = [row.get(key) for row in rows
                    if _cmp_group(row) == g and isinstance(row.get(key), (int, float))]
            if vals:
                best[(ci, g)] = max(vals) if "↑" in title else min(vals)

    for ri, row in enumerate(rows, 3):
        for ci, (grp, title, key, _) in enumerate(columns, 1):
            v = row.get(key)
            c = ws.cell(row=ri, column=ci, value=v)
            c.border = border
            c.alignment = Alignment(horizontal="center", vertical="center")
            if isinstance(v, float):
                c.number_format = "0.0000"
            if ci <= 3:
                c.font = Font(bold=(ci <= 2), size=9)
                c.alignment = Alignment(horizontal="left", vertical="center")
            elif (isinstance(v, (int, float))
                  and best.get((ci, _cmp_group(row))) == v):
                c.font = Font(bold=True, size=9, color="1A4F8A")
            if v is None:
                # A blank cell means one of two things: the run has not finished,
                # or the metric does not exist structurally (Letta has no turn
                # provenance, so LoCoMo recall@k simply cannot be computed)
                ds = {"LongMemEval": "longmem", "LoCoMo": "locomo",
                      "HaluMem": "halumem", "MemFail": "memfail"}.get(dataset_of(key))
                if ds and not row.get("_done", {}).get(ds, True):
                    c.value = "RUNNING"
                    c.font = Font(color="C05621", size=9, italic=True, bold=True)
                else:
                    c.value = "—"
                    c.font = Font(color="B0B8C0", size=9)

    ws.freeze_panes = "D3"
    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 34


def build():
    rows = collect()
    wb = openpyxl.Workbook()

    # Master sheet: every column
    ws = wb.active
    ws.title = "Failure Matrix"
    write_sheet(ws, COLUMNS, rows)

    # One sheet per dataset: same structure, only that dataset's columns
    for ds in SHEET_ORDER:
        write_sheet(wb.create_sheet(ds), columns_for(ds), rows)

    # ── Definitions sheet ───────────────────────────────────────────────────
    ws2 = wb.create_sheet("Definitions")
    for ci, h in enumerate(["Stage", "Column", "Source / definition"], 1):
        c = ws2.cell(row=1, column=ci, value=h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="F2F5F8")
    for ri, (grp, title, key, desc) in enumerate(COLUMNS, 2):
        ws2.cell(row=ri, column=1, value=grp or "—")
        ws2.cell(row=ri, column=2, value=title)
        ws2.cell(row=ri, column=3, value=desc).alignment = Alignment(wrap_text=True)

    # ── What the [code] prefix on subset columns means ──────────────────────
    ri = len(COLUMNS) + 3
    t = ws2.cell(row=ri, column=1, value="Purpose group codes")
    t.font = Font(bold=True, size=11, color="3D6B2E")
    ws2.cell(row=ri, column=3,
             value="The [1] [2a] ... prefix on each Subset scores column names the "
                   "purpose group that subset belongs to. The grouping dimension is "
                   "what kind of memory a correct answer demands: how many entries, "
                   "whether timestamps are needed, whether the latest value is needed. "
                   "Whether external world knowledge is required, whether arithmetic is "
                   "involved, and whether the answer is open-ended are cross-cutting "
                   "attributes and do not define groups. Subsets in the same group sit "
                   "adjacent in the sheet so comparable abilities can be read across."
             ).alignment = Alignment(wrap_text=True)
    for ci, h in enumerate(["Code", "Group", "Definition"], 1):
        c = ws2.cell(row=ri + 1, column=ci, value=h)
        c.font = Font(bold=True, size=10)
        c.fill = PatternFill("solid", fgColor="EDF3EA")
    for k, (code, gname, gdesc) in enumerate(PURPOSE_GROUPS, ri + 2):
        ws2.cell(row=k, column=1, value=code).font = Font(bold=True, size=10)
        ws2.cell(row=k, column=2, value=gname)
        ws2.cell(row=k, column=3, value=gdesc).alignment = Alignment(wrap_text=True)

    ws2.column_dimensions["A"].width = 18
    ws2.column_dimensions["B"].width = 24
    ws2.column_dimensions["C"].width = 80

    # ── Run parameters sheet ────────────────────────────────────────────────
    ws3 = wb.create_sheet("Run parameters")
    meta = [
        ["Generated at", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Extraction LLM", "gemma-4-31B-it (openai-proxy/gemma-4-31B-it for Letta)"],
        ["Judge LLM", "gemma-4-E4B-it (OPENAI_MODEL)"],
        ["Judge temperature", "0 (pinned from this batch onward; earlier runs were not pinned)"],
        ["API endpoint", "NCHC GenAI portal"],
        ["", ""],
        ["LongMemEval scale", "18 questions (3 per type), haystack of about 50 sessions each"],
        ["LoCoMo scale", "1 conversation (conv-26), 19 sessions / 419 turns / 199 QA"],
        ["HaluMem scale", "1 user (the 2nd), 77 sessions / 188 QA"],
        ["MemFail scale", "35 questions (5 per subset, 15 for persona)"],
        ["", ""],
        ["top-k", "20 (identical across all four experiments)"],
        ["", ""],
        ["", "-- Cost and latency columns --"],
        ["Instrumentation", "halumem_experiment/token_tracker.py intercepts LLM calls in place while the experiment runs"],
        ["Buckets", "ingest (memory writes) / qa (retrieval + generation) / other (judge, probes, warm-up)"],
        ["Excluded", "Cost figures exclude the judge and the probes; those land in the other bucket and are not counted"],
        ["Definition of unit", "The span covered by one add call, i.e. the granularity column: turn or session"],
        ["Letta call counts", "Taken from usage.step_count; one messages.create may run several agent steps server-side"],
        ["Seconds per unit", "End-to-end median including retrieval and non-LLM overhead, not pure API time"],
        ["Why cells are blank", "Runs executed before 2026-08-18 recorded only total_tokens with no bucketing; they must be re-run to produce figures"],
        ["", ""],
        ["", "-- Run mapped to each backend --"],
    ]
    for name, hal, loc, lme, mf, llm in BACKENDS:
        meta.append([name, f"HaluMem={hal} | LoCoMo={loc} | LongMemEval={lme} | MemFail={mf}"])
    meta += [
        ["", ""],
        ["", "-- Known limitations --"],
        ["LongMemEval", "Not finished in this batch, so the column is blank"],
        ["HaluMem StructMem/A-MEM/Letta", "Reuses user2nd_gemma431b_probe from 2026-08-08 to 08-09, where judge temperature was not pinned"],
        ["LoCoMo Letta P1-scoped", "Letta has no turn provenance, so the scope falls back to the whole store and is not directly comparable to the other backends"],
        ["MemFail Letta", "results_5q_letta; the letta_b2 and letta_flush variants were not adopted"],
    ]
    for ri, (a, b) in enumerate(meta, 1):
        ws3.cell(row=ri, column=1, value=a).font = Font(bold=bool(a and not b.startswith("HaluMem=")))
        ws3.cell(row=ri, column=2, value=b).alignment = Alignment(wrap_text=True)
    ws3.column_dimensions["A"].width = 32
    ws3.column_dimensions["B"].width = 100

    wb.save(OUT)
    print(f"✅ {OUT}")
    filled = sum(1 for r in rows for k, v in r.items() if v is not None)
    print(f"   {len(rows)} rows x {len(COLUMNS)} columns, {filled} cells filled")
    return OUT


if __name__ == "__main__":
    build()
