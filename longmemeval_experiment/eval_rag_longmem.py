"""
RAG / raw-turn baseline for LongMemEval-S (independent folder; imports halumem shared code).

The "no memory management" lower bound: for each question, build a FRESH in-memory
vector store from that question's haystack — every raw turn is embedded verbatim and
tagged with its SESSION id (LongMemEval provenance is session-level: answer_session_ids).
At QA time retrieve top-k by cosine and answer.

Because RAG stores raw turns tagged with session_id, retrieval Recall@k / NDCG@k are
exact and natural (no extraction, no provenance guessing) — directly comparable to
Mem0 (session-level metadata) and A-MEM (turn→session mapping).

CRUD framing:
  ① Extraction : NONE (raw turns, only embedded for search)
  ② CUD        : NONE (pure accumulate)
  ③ Storage    : raw turn text + embedding + session_id
  ④ Retrieve   : cosine top-k

Embeddings use the same Ollama model as Mem0 (bge-m3) for a fair comparison.
"""

import os
import sys
import json
import time
import logging
import traceback

import numpy as np
import requests

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk
from llms import llm_request

OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RAG_EMBED_MODEL = os.getenv("MEM0_EMBED_MODEL", "bge-m3:latest")

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a user's long chat history.

Memories:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the memories above. If they do not contain enough information, say "No information available".
Answer:"""


def _embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": RAG_EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    v = np.asarray(r.json()["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TurnStore:
    """In-memory vector store of raw turns (text + session_id)."""
    def __init__(self):
        self.texts, self.sids, self.vecs = [], [], []

    def add(self, text, sid):
        self.texts.append(text); self.sids.append(sid); self.vecs.append(_embed(text))

    def search(self, query, top_k):
        if not self.texts:
            return []
        sims = np.stack(self.vecs) @ _embed(query)
        order = np.argsort(-sims)[:top_k]
        return [{"text": self.texts[i], "session_id": self.sids[i]} for i in order]


def _stratified_sample(data, n):
    """Sample ~n questions evenly across question_type (data is grouped by type)."""
    from collections import defaultdict
    by = defaultdict(list)
    for q in data:
        by[q["question_type"]].append(q)
    per = max(1, n // len(by))
    out = []
    for qs in by.values():
        out.extend(qs[:per])
    return out[:n] if len(out) >= n else out


def setup_logger(log_dir, qid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"lme_rag.{qid}")
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
    logger.info(f"=== Start question {qid} ({q['question_type']}) (RAG) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    store    = TurnStore()
    sessions = q["haystack_sessions"]
    sess_ids = q["haystack_session_ids"]
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]

    try:
        # ① Store raw turns (tagged with session_id)
        t0 = time.time()
        n = 0
        for sess, sid in zip(sessions, sess_ids):
            for turn in sess:
                content = turn.get("content", "")
                if not content:
                    continue
                # RAG performs no extraction; ingest is embeddings only and costs
                # zero LLM calls (the control baseline).
                with _tk.unit("ingest"):
                    store.add(f'{turn.get("role", "user")}: {content}', sid)
                n += 1
        logger.info(f"Stored {n} raw turns ({len(sessions)} sessions) in {(time.time()-t0):.1f}s")

        # 2. Answer the single question. Each question builds its own store, so one
        #    question is one qa unit.
        with _tk.unit("qa"):
            retrieved = store.search(q["question"], top_k)
            mems = [r["text"] for r in retrieved]
            prompt = QA_PROMPT.format(context=json.dumps(mems, indent=2),
                                      date=q.get("question_date", ""), question=q["question"])
            t0 = time.time()
            response = llm_request(prompt)

        new_q = {k: q.get(k) for k in ["question_id", "question_type", "question", "answer",
                                       "question_date", "answer_session_ids"]}
        new_q.update({"retrieved": retrieved, "retrieved_memories": mems,
                      "system_response": response, "response_ms": round((time.time()-t0)*1000, 1)})
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        return {"qid": qid, "status": "ok", "path": tmp_file}
    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{qid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"qid": qid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "rag"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "rag_lme_results.jsonl")
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
    print(f"  LongMemEval-S × RAG baseline (no extraction, raw turns)")
    print(f"  EMBED  : {RAG_EMBED_MODEL}")
    print(f"  QA LLM : {llm_model or os.getenv('OPENAI_MODEL')}")
    print(f"  questions : {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
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

    _tk.save(save_path, "rag_lme")
    meta = {"extraction_llm": "none (raw turns)", "judge_llm": os.getenv("OPENAI_MODEL", "unknown"),
            "embed_model": RAG_EMBED_MODEL, "granularity": "turn"}
    with open(os.path.join(save_path, "rag_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
