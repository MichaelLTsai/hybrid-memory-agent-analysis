"""
StructMem adapter for LongMemEval-S (independent folder; imports halumem shared code).

StructMem is not a standalone package — it is the LightMemory engine run in
`extraction_mode="event"` with periodic cross-event consolidation. This file reuses
halumem_experiment/eval_structmem.py's config builder and store dump WITHOUT
modifying them.

LongMemEval structure (per question):
  question, question_type, answer, question_date
  answer_session_ids   : evidence sessions (SESSION-level provenance)
  haystack_sessions    : ~48 sessions, each a list of {role, content} turns
  haystack_session_ids : ids parallel to haystack_sessions

Per question we build a FRESH StructMem store from that question's haystack (each
question has its own haystack — no sharing), then answer the single question.

Provenance: StructMem extracts events across a whole session, so each new store entry
is attributed to the session that produced it — matching LongMemEval's session-level
evidence, and letting probe_longmem.py scope P1 to the answer session.

NOTE: per-question haystacks are expensive (~48 session-adds/question), so runs use a
subset via --max-questions.
"""

import os
import re
import sys
import json
import time
import shutil
import logging
import traceback

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk
from llms import llm_request
from eval_structmem import _build_config, _store_dump, SM_LLM, SUMMARIZE_EVERY

# Scratch root for the per-question Qdrant stores. The per-question tag is keyed
# on question id and ablation arm, not on --version, so two runs of the same arm
# would reuse (and rmtree) the same directories. Overriding this per batch keeps
# an earlier batch's stores intact.
QDRANT_BASE = os.getenv(
    "LME_QDRANT_BASE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_structmem"),
)

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a user's long chat history.

Memories:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the memories above. If they do not contain enough information, say "No information available".
Answer:"""


def _sm_time(s: str) -> str:
    """LongMemEval date like '2023/05/30 (Tue) 23:40' → LightMemory's expected format.
    LightMemory parses time_stamp strictly and silently drops the whole session on a
    mismatch, so normalise before add_memory."""
    from datetime import datetime
    if s:
        try:
            from dateutil import parser as dp
            return dp.parse(re.sub(r"\([A-Za-z]{3}\)", "", s)).strftime("%Y/%m/%d (%a) %H:%M")
        except Exception:
            pass
    return datetime.now().strftime("%Y/%m/%d (%a) %H:%M")


# M4 QA conditioning. Only reaches the prompt when enable_m4_state_qa is on.
_STATE_QA_RULES = {
    "current": ("- CURRENT MEMORY controls the present-state answer.\n"
                "  - HISTORICAL MEMORY is context only, never the current value."),
    "historical": ("- Answer with the past state the question asks about.\n"
                   "  - Do not let a newer CURRENT MEMORY override the historical target."),
    "transition": ("- Describe the change: the before value and the after value.\n"
                   "  - Use TRANSITION evidence for when and how it changed."),
    "neutral": ("- Answer by ordinary relevance.\n"
                "  - Do not force a temporal narrative."),
}


def _state_ablation(trace_dir=None):
    """Resolve the M1/M3/M4 arm from STRUCTMEM_EXPERIMENT (default E0)."""
    from lightmem.memory.state import config as _state_config

    cfg = _state_config.from_env()
    if trace_dir:
        cfg.trace_dir = trace_dir
    return cfg


def _stratified_sample(data, n):
    """Sample ~n questions evenly across question_type (data is grouped by type).

    The LME_TYPE_QUOTA environment variable overrides the quota for individual
    question types, for example
        LME_TYPE_QUOTA="knowledge-update=10"
    which gives knowledge-update 10 questions while every other type keeps its
    even share of n. The first k questions of a type are always the ones taken, so
    an enlarged sample still fully contains the original one and the two batches
    remain directly comparable. When a quota is set the total exceeds n and is no
    longer truncated, since exceeding it is the whole point.
    """
    import os
    from collections import defaultdict
    by_type = defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    per = max(1, n // len(by_type))
    quota = {}
    for item in os.getenv("LME_TYPE_QUOTA", "").split(","):
        if "=" in item:
            k, _, v = item.partition("=")
            try:
                quota[k.strip()] = int(v)
            except ValueError:
                pass
    out = []
    # Question types are otherwise processed in data-file order, which puts
    # knowledge-update last. A run that dies partway then has none of the
    # questions the state ablation is actually about. LME_TYPE_ORDER promotes
    # the named types to the front so the critical ones land first; it only
    # reorders, never changes the selection, so results stay identical.
    order = [t.strip() for t in os.getenv("LME_TYPE_ORDER", "").split(",") if t.strip()]
    types = ([t for t in order if t in by_type]
             + [t for t in by_type if t not in order])
    for t in types:
        qs = by_type[t]
        out.extend(qs[:quota.get(t, per)])
    if quota:
        return out
    return out[:n] if len(out) >= n else out


def setup_logger(log_dir, qid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"lme_structmem.{qid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{qid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def process_question(q, top_k, save_path, log_dir, session_limit=None, llm_model=None):
    from lightmem.memory.lightmem import LightMemory

    qid    = q["question_id"]
    logger = setup_logger(log_dir, qid)
    logger.info(f"=== Start {qid} ({q['question_type']}) | StructMem | LLM={llm_model or SM_LLM} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    # Each ablation arm gets its own collections so no two arms share a store.
    state_cfg = _state_ablation(trace_dir=os.path.join(save_path, "traces"))
    tag = re.sub(r"[^a-zA-Z0-9]", "", qid)[:28] + state_cfg.collection_suffix()
    tag = re.sub(r"[^a-zA-Z0-9_]", "", tag)
    shutil.rmtree(os.path.join(QDRANT_BASE, tag), ignore_errors=True)          # fresh store
    shutil.rmtree(os.path.join(QDRANT_BASE, tag + "_sum"), ignore_errors=True)
    logger.info(f"Ablation arm: {state_cfg.to_manifest()}")

    sessions = q.get("haystack_sessions", [])
    sess_ids = q.get("haystack_session_ids", [])
    dates    = q.get("haystack_dates", [""] * len(sessions))
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]
        dates = dates[:session_limit]

    lm = None
    try:
        cfg = _build_config(tag, llm_model)
        cfg["state_ablation"] = state_cfg
        cfg["embedding_retriever"]["configs"]["path"] = os.path.join(QDRANT_BASE, tag)
        cfg["summary_retriever"]["configs"]["path"]   = os.path.join(QDRANT_BASE, tag + "_sum")
        lm = LightMemory.from_config(cfg)

        # ── ① Feed each haystack session; track which session produced each entry ──
        t0 = time.time()
        seen, id2sid = set(), {}
        n_ok = 0
        for sess, sid, date in zip(sessions, sess_ids, dates):
            stamp = _sm_time(date)
            msgs = [{"role": t.get("role", "user"), "content": t.get("content", ""),
                     "time_stamp": stamp}
                    for t in sess if t.get("content")]
            if not msgs:
                continue
            # One session is one ingest unit.
            with _tk.unit("ingest"):
                try:
                    lm.add_memory(msgs, force_extract=True)
                    n_ok += 1
                except Exception as e:
                    logger.warning(f"{sid} add error: {e}")

            # Consolidation is part of the architecture's write cost, so it goes into
            # the ingest bucket without counting a unit.
            if n_ok and n_ok % SUMMARIZE_EVERY == 0:
                with _tk.phase("ingest"):
                    # LightMem's offline update was never wired into this
                    # adapter, so knowledge-update was previously measured with
                    # no update mechanism running at all. It is enabled here for
                    # every arm: E0 gets the original update/delete/ignore, the
                    # M1 arms get the state commit. Set STRUCTMEM_OFFLINE_UPDATE=0
                    # to restore the old no-update condition.
                    def _run_update():
                        if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") != "1":
                            return
                        try:
                            lm.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                            lm.offline_update_all_entries(score_threshold=0.9)
                        except Exception as e:
                            logger.warning(f"{sid} offline_update error: {e}")

                    def _run_summarize():
                        try:
                            lm.summarize(process_all=True, enable_cross_event=True,
                                         retrieval_scope="global", top_k_seeds=15)
                        except Exception as e:
                            logger.warning(f"{sid} summarize error: {e}")

                    # M3 needs the state commit to land before summaries are
                    # written; without it the original order is preserved.
                    if lm.state_ablation.enable_m3_summary_sync:
                        _run_update()
                        _run_summarize()
                    else:
                        _run_summarize()
                        _run_update()

            for entry in _store_dump(lm):
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    id2sid[entry["id"]] = sid
        # Final flush: the loop above only fires every SUMMARIZE_EVERY sessions,
        # so without this the tail of the haystack would never be consolidated
        # or state-audited. Applied identically to every arm.
        with _tk.phase("ingest"):
            def _final_update():
                if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") != "1":
                    return
                try:
                    lm.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                    lm.offline_update_all_entries(score_threshold=0.9)
                except Exception as e:
                    logger.warning(f"final offline_update error: {e}")

            def _final_summarize():
                try:
                    lm.summarize(process_all=True, enable_cross_event=True,
                                 retrieval_scope="global", top_k_seeds=15)
                except Exception as e:
                    logger.warning(f"final summarize error: {e}")

            # Same ordering rule as the in-loop flush, so the E1/E3 stale-summary
            # condition stays genuine.
            if lm.state_ablation.enable_m3_summary_sync:
                _final_update()
                _final_summarize()
            else:
                _final_summarize()
                _final_update()

        logger.info(f"Fed {n_ok}/{len(sessions)} sessions in {(time.time()-t0):.0f}s")

        # ── ② Full store dump → probe_longmem.py (P1: was it ever stored?) ──
        try:
            all_memories = [{"text": e["memory"], "session_id": id2sid.get(e["id"])}
                            for e in _store_dump(lm)]
            logger.info(f"Dumped {len(all_memories)} memories")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            all_memories = None

        # ── ③ Answer the single question ─────────────────────────────────
        # Each question builds its own store, so one question is one qa unit.
        with _tk.unit("qa"):
            packet = None
            try:
                # Dual-circuit retrieval (entries + cross-event summaries);
                # state-aware ordering and labels only when M4 is on.
                mems, packet = lm.retrieve_for_qa(q["question"], limit=top_k)
            except Exception as e:
                logger.warning(f"retrieve error: {e}")
                mems = []
            mems = [m if isinstance(m, str) else str(m) for m in mems]
            # session-level extraction → no per-memory session id at retrieval time
            retrieved = [{"text": m, "session_id": None} for m in mems]

            context = "\n".join(f"  - {m}" for m in mems) or "  (none)"
            if packet is not None:
                context = (
                    f"  QUERY STATE VIEW: {packet.query_view.upper()}\n"
                    + context
                    + "\n  State rules:\n  "
                    + _STATE_QA_RULES.get(packet.query_view, _STATE_QA_RULES["neutral"])
                )
            prompt = QA_PROMPT.format(context=context,
                                      date=q.get("question_date", ""), question=q["question"])
            t0 = time.time()
            response = llm_request(prompt)

        new_q = {k: q.get(k) for k in ["question_id", "question_type", "question", "answer",
                                       "question_date", "answer_session_ids"]}
        new_q.update({"retrieved": retrieved, "retrieved_memories": mems,
                      "all_memories": all_memories, "system_response": response,
                      "response_ms": round((time.time() - t0) * 1000, 1)})
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        logger.info(f"{qid} complete → {tmp_file}")
        return {"qid": qid, "status": "ok"}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{qid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"qid": qid, "status": "error", "path": err}
    finally:
        try:
            lm.embedding_retriever.client.close()
        except Exception:
            pass


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "structmem"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "structmem_lme_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        data = data[:2]
    elif max_questions:
        data = _stratified_sample(data, max_questions)

    print(f"\n{'='*60}")
    print(f"  LongMemEval-S × StructMem (event mode)")
    print(f"  LLM        : {llm_model or SM_LLM}")
    print(f"  questions  : {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    start = time.time()
    for idx, q in enumerate(data, 1):
        qid = q["question_id"]
        if qid in done:
            print(f"⏭️  [{idx}/{len(data)}] {qid} already done"); continue
        r = process_question(q, top_k, save_path, log_dir,
                             session_limit=5 if smoke else None, llm_model=llm_model)
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['qid']} ({q['question_type']}) → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(save_path, "structmem_lme")

    meta = {
        "extraction_llm": llm_model or SM_LLM,
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    "all-MiniLM-L6-v2",
        "granularity":    "session",
    }
    with open(os.path.join(save_path, "structmem_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
