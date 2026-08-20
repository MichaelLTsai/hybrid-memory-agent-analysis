"""
Mem0 adapter for LongMemEval-S (independent folder; imports halumem shared code).

LongMemEval structure (per question):
  question, question_type, answer, question_date
  answer_session_ids : evidence sessions (SESSION-level provenance)
  haystack_sessions  : ~48 sessions, each a list of {role, content} turns
  haystack_session_ids : ids parallel to haystack_sessions

Per question we build a FRESH Mem0 store from that question's haystack (each
question has its own haystack — no sharing), then answer the single question.
Each session is fed with metadata={"session_id": ...} so retrieved memories carry
their source session → exact SESSION-level Recall@k / NDCG@k (matches the
session-level evidence).

NOTE: per-question haystacks are expensive (~48 session-adds/question), so runs
use a subset via --max-questions.
"""

import os
import sys
import json
import time
import copy
import logging
import traceback

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk
from llms import llm_request
from eval_mem0_oss import build_mem0_config, mem_search, MEM0_MAJOR
from mem0 import Memory

QDRANT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a user's long chat history.

Memories:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the memories above. If they do not contain enough information, say "No information available".
Answer:"""


def _mem_get_all(mem, user_name: str):
    """Version-aware full-store dump. Param names differ between Mem0 majors:
        v1.x : get_all(user_id=..., limit=N)
        v2.x : get_all(filters={'user_id':...}, limit=N)   # top-level user_id rejected
    NOTE: the default limit is 100 in both — must be raised or the dump truncates."""
    if MEM0_MAJOR < 2:
        return mem.get_all(user_id=user_name, limit=100000)
    # v2 caps at top_k (default 20) — `limit` silently lands in **kwargs and is ignored,
    # which truncates the dump to 20 and makes P1 look catastrophically low.
    return mem.get_all(filters={"user_id": user_name}, top_k=100000)


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
    logger = logging.getLogger(f"lme_mem0.{qid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{qid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def process_question(q, top_k, save_path, log_dir, session_limit=None, llm_model=None):
    qid    = q["question_id"]
    logger = setup_logger(log_dir, qid)
    logger.info(f"=== Start question {qid} ({q['question_type']}) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    user_id    = f"lme_{qid.replace('-', '_')}"
    collection = f"lme_{qid.replace('-', '_')[:40]}"
    qdrant_path = os.path.join(QDRANT_BASE, qid.replace('-', '_'))
    mem = Memory.from_config(build_mem0_config(collection, qdrant_path, llm_model=llm_model))

    sessions  = q["haystack_sessions"]
    sess_ids  = q["haystack_session_ids"]
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]

    try:
        # ── Feed each haystack session (metadata=session_id for provenance) ──
        t0 = time.time()
        for sess, sid in zip(sessions, sess_ids):
            messages = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                        for t in sess if t.get("content")]
            if messages:
                # mem0 works at session level on LongMemEval, so one session is one ingest unit.
                with _tk.unit("ingest"):
                    try:
                        mem.add(messages, user_id=user_id, metadata={"session_id": sid})
                    except Exception as e:
                        logger.warning(f"add {sid} error: {e}")
        add_ms = (time.time() - t0) * 1000
        logger.info(f"Fed {len(sessions)} sessions in {add_ms:.0f}ms")

        # ── Full memory-store dump → P1 (was the value ever stored at all?) ──
        #    Without this, a value missing from `retrieved` cannot be separated into
        #    SUMMARY (never extracted) vs RETRIEVAL (stored but not surfaced).
        #    Taken right after ingest; session_id provenance lets P1 scope the search
        #    to the answer session instead of a lossy top-k over the whole store.
        #    NOTE: get_all() defaults to limit=100 — must raise it or the dump truncates.
        try:
            snap = _mem_get_all(mem, user_id)
            items = snap.get("results", snap) if isinstance(snap, dict) else snap
            all_memories = [{"text": it.get("memory", ""),
                             "session_id": (it.get("metadata") or {}).get("session_id")}
                            for it in items if it.get("memory")]
            logger.info(f"Dumped {len(all_memories)} memories")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            all_memories = None

        # ── Answer the single question ───────────────────────────────────
        # LongMemEval builds a store per question, so one question is one qa unit
        # (outside the loop).
        with _tk.unit("qa"):
            res = mem_search(mem, q["question"], user_id, top_k)
            retrieved = [{"text": it.get("memory", ""),
                          "session_id": (it.get("metadata") or {}).get("session_id")}
                         for it in res.get("results", [])]
            mems = [r["text"] for r in retrieved]
            context = json.dumps(mems, indent=2)
            prompt  = QA_PROMPT.format(context=context, date=q.get("question_date", ""),
                                       question=q["question"])
            t0 = time.time()
            response = llm_request(prompt)

        new_q = {
            "question_id":        qid,
            "question_type":      q["question_type"],
            "question":           q["question"],
            "answer":             q.get("answer"),
            "question_date":      q.get("question_date"),
            "answer_session_ids": q.get("answer_session_ids", []),
            "retrieved":          retrieved,
            "retrieved_memories": mems,
            "all_memories":       all_memories,
            "system_response":    response,
            "response_ms":        round((time.time() - t0) * 1000, 1),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        try:
            mem.vector_store.client.close()
        except Exception:
            pass
        return {"qid": qid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{qid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        try:
            mem.vector_store.client.close()
        except Exception:
            pass
        return {"qid": qid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "mem0"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "mem0_lme_results.jsonl")
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
    print(f"  LongMemEval-S × Mem0")
    print(f"  MEM0 major : {MEM0_MAJOR}")
    print(f"  LLM        : {llm_model or os.getenv('MEM0_LLM_MODEL')}")
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

    _tk.save(save_path, "mem0_lme")
    meta = {
        "extraction_llm": llm_model or os.getenv("MEM0_LLM_MODEL", "unknown"),
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    os.getenv("MEM0_EMBED_MODEL", "unknown"),
        "granularity":    "session",
    }
    with open(os.path.join(save_path, "mem0_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
