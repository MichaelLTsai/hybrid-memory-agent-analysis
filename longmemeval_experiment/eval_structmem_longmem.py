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

QDRANT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_structmem")

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
    for t, qs in by_type.items():
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

    tag = re.sub(r"[^a-zA-Z0-9]", "", qid)[:28]
    shutil.rmtree(os.path.join(QDRANT_BASE, tag), ignore_errors=True)          # fresh store
    shutil.rmtree(os.path.join(QDRANT_BASE, tag + "_sum"), ignore_errors=True)

    sessions = q.get("haystack_sessions", [])
    sess_ids = q.get("haystack_session_ids", [])
    dates    = q.get("haystack_dates", [""] * len(sessions))
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]
        dates = dates[:session_limit]

    lm = None
    try:
        cfg = _build_config(tag, llm_model)
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
                    try:
                        lm.summarize(process_all=True, enable_cross_event=True,
                                     retrieval_scope="global", top_k_seeds=15)
                    except Exception as e:
                        logger.warning(f"{sid} summarize error: {e}")

            for entry in _store_dump(lm):
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    id2sid[entry["id"]] = sid
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
            try:
                mems = lm.retrieve(q["question"], limit=top_k)
            except Exception as e:
                logger.warning(f"retrieve error: {e}")
                mems = []
            mems = [m if isinstance(m, str) else str(m) for m in mems]
            # session-level extraction → no per-memory session id at retrieval time
            retrieved = [{"text": m, "session_id": None} for m in mems]

            prompt = QA_PROMPT.format(context="\n".join(f"  - {m}" for m in mems) or "  (none)",
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
