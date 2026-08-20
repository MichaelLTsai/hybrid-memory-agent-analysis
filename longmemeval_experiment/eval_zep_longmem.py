"""
Zep (Cloud / Platform) adapter for LongMemEval-S.

Zep is a MANAGED cloud memory platform (data is uploaded to Zep's servers). It
builds a temporal knowledge graph per user from conversation messages: facts
become graph EDGES carrying valid_at / invalid_at / expired_at — i.e. Zep models
knowledge UPDATES natively (a superseded fact is marked invalid). This is why
Zep is a strong baseline for the knowledge-update question type.

Paradigm (mechanical, like Mem0/MemOS — not an autonomous agent):
  Per QUESTION → a FRESH Zep user. Each haystack SESSION → one Zep THREAD
  (thread_id ← per-session tag), messages ingested there. Zep asynchronously
  builds the graph; we poll episodes until processed. Then:
    • ANSWER  : graph.search(scope="edges")  → temporal facts as context,
                answered by the SHARED judge LLM (gemma-4-E4B, same as others)
    • RECALL  : graph.search(scope="episodes") → each Episode carries thread_id
                → map back to session_id → exact SESSION-level Recall@k / NDCG@k

Tokens: the ANSWER-side LLM calls go through halumem's token_tracker (captured
in zep_lme_token_usage.json). Zep's server-side graph-building LLM usage is NOT
token-exposed (it is a managed platform, billed as API usage) — noted in meta.

env: ZEP_API_KEY   (+ halumem .env for the shared answer/judge LLM)

Run EXTRACTION under venv_zep with --skip-eval; EVALUATION under the main venv
with --eval-only (the judge in evaluation_longmem.py needs halumem llms).
"""

import os
import re
import sys
import json
import time
import logging
import traceback

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk
from llms import llm_request  # shared answer LLM (gemma-4-E4B via NCHC), same as Mem0/MemOS/RAG

ZEP_API_KEY = os.getenv("ZEP_API_KEY", "")
PROCESS_TIMEOUT = int(os.getenv("ZEP_PROCESS_TIMEOUT", "14400"))  # cap (~4h); returns early when graph is done

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a user's long chat history.

Memories:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the memories above. If they do not contain enough information, say "No information available".
Answer:"""


def _client():
    from zep_cloud.client import Zep
    return Zep(api_key=ZEP_API_KEY)


def _stratified_sample(data, n):
    from collections import defaultdict
    by_type = defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    per = max(1, n // len(by_type))
    out = []
    for qs in by_type.values():
        out.extend(qs[:per])
    return out[:n] if len(out) >= n else out


def setup_logger(log_dir, qid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"lme_zep.{qid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{qid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _wait_processed(client, user_id, expected, logger, timeout):
    """Poll episodes until the graph is fully built (Zep processes async).

    Robust to episode-count != message-count: 'done' when every episode Zep has
    created is processed AND the total episode count has stopped growing
    (stable across consecutive polls) — not a comparison to the message count.
    """
    t0 = time.time()
    prev_total = -1; stable = 0
    while time.time() - t0 < timeout:
        try:
            eps = client.graph.episode.get_by_user_id(user_id=user_id, lastn=max(expected + 100, 500))
            items = getattr(eps, "episodes", None) or getattr(eps, "items", None) or []
            total = len(items)
            done  = sum(1 for e in items if getattr(e, "processed", False))
            stable = stable + 1 if total == prev_total else 0
            prev_total = total
            logger.info(f"  processing… {done}/{total} episodes (stable={stable}, target≈{expected}, {(time.time()-t0):.0f}s)")
            # done: something ingested, all processed, count settled, and near the
            # expected volume (guards against declaring done mid-ingest)
            if total > 0 and done == total and stable >= 2 and total >= expected * 0.9:
                logger.info(f"processing complete: {done} episodes in {(time.time()-t0):.0f}s")
                return True
        except Exception as e:
            logger.warning(f"poll error: {e}")
        time.sleep(20)
    logger.warning(f"processing wait TIMED OUT after {timeout}s ({prev_total} episodes seen)")
    return False


def _facts_with_time(edges):
    """Zep-recommended temporal handling for answering: drop expired/superseded
    facts, tag each surviving fact with its valid-from date, newest first -- so
    the answer LLM prefers the CURRENT value on knowledge-update questions
    (instead of an outdated one flattened into a contradictory list)."""
    rows = []
    for e in edges:
        f = getattr(e, "fact", None)
        if not f:
            continue
        if getattr(e, "expired_at", None) or getattr(e, "invalid_at", None):
            continue  # superseded / no longer valid -> drop
        v = getattr(e, "valid_at", None)
        rows.append((v or "", (f + " (as of " + v[:10] + ")") if v else f))
    rows.sort(key=lambda x: x[0], reverse=True)  # newest first
    return [r[1] for r in rows]


def process_question(q, top_k, save_path, log_dir, session_limit=None, llm_model=None):
    qid    = q["question_id"]
    logger = setup_logger(log_dir, qid)
    logger.info(f"=== Start question {qid} ({q['question_type']}) | Zep cloud ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    sessions = q["haystack_sessions"]
    sess_ids = q["haystack_session_ids"]
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]

    from zep_cloud.types.message import Message
    client  = _client()
    tag     = re.sub(r"[^a-zA-Z0-9]", "", qid)[:24]
    user_id = f"lme_{tag}"
    thread_of_session = {}   # thread_id → real session_id (provenance map)

    try:
        # Isolation: wipe any leftover graph for this user_id first, so each
        # question gets a clean, independent Zep graph (no cross-question mixing).
        try:
            client.user.delete(user_id=user_id)
            time.sleep(2)
        except Exception:
            pass
        try:
            client.user.add(user_id=user_id, first_name="lme")
        except Exception:
            pass

        # ① Ingest each haystack session as its own thread (thread ↔ session) ──
        t0 = time.time(); n_msg = 0
        for idx, (sess, sid) in enumerate(zip(sessions, sess_ids)):
            thread_id = f"{tag}_s{idx}"
            thread_of_session[thread_id] = sid
            try:
                client.thread.create(thread_id=thread_id, user_id=user_id)
            except Exception:
                pass
            # Zep caps a single message at 4096 chars → split long turns (keep role
            # + thread, so no content is dropped and provenance is preserved).
            msgs = []
            for t in sess:
                c = t.get("content", "")
                if not c:
                    continue
                role = t.get("role") or "user"
                for j in range(0, len(c), 4000):
                    msgs.append(Message(role=role, content=c[j:j+4000]))
            if not msgs:
                continue
            for i in range(0, len(msgs), 30):                # Zep caps messages/add
                try:
                    client.thread.add_messages(thread_id=thread_id, messages=msgs[i:i+30])
                    n_msg += len(msgs[i:i+30])
                except Exception as e:
                    logger.warning(f"add_messages {thread_id} error: {e}")
        logger.info(f"Ingested {len(sessions)} sessions / {n_msg} messages in {(time.time()-t0):.0f}s")

        # ② Wait for Zep to build the graph ───────────────────────────────────
        _wait_processed(client, user_id, expected=n_msg, logger=logger, timeout=PROCESS_TIMEOUT)

        # ③ Retrieve — facts (answer) + episodes (session provenance) ─────────
        facts = []
        try:
            er = client.graph.search(query=q["question"], user_id=user_id,
                                     scope="edges", limit=top_k, reranker="cross_encoder")
            facts = _facts_with_time(getattr(er, "edges", None) or [])
        except Exception as e:
            logger.warning(f"edge search error: {e}")

        retrieved = []
        try:
            pr = client.graph.search(query=q["question"], user_id=user_id,
                                     scope="episodes", limit=top_k, reranker="cross_encoder")
            for ep in (getattr(pr, "episodes", None) or []):
                tid = getattr(ep, "thread_id", None)
                retrieved.append({"text": getattr(ep, "content", ""),
                                  "session_id": thread_of_session.get(tid)})
        except Exception as e:
            logger.warning(f"episode search error: {e}")
        n_sid = sum(1 for r in retrieved if r["session_id"])
        logger.info(f"retrieved {len(facts)} facts / {len(retrieved)} episodes ({n_sid} with provenance)")

        # ④ Answer with the SHARED judge LLM (facts as context) ───────────────
        context = json.dumps(facts or [r["text"] for r in retrieved], indent=2, ensure_ascii=False)
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
            "retrieved":          retrieved,                 # episodes → session_id (for Recall@k)
            "retrieved_memories": facts,                     # Zep facts used to answer
            "system_response":    response,
            "response_ms":        round((time.time() - t0) * 1000, 1),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        logger.info(f"Question {qid} complete → {tmp_file}")
        try:
            client.user.delete(user_id=user_id)             # cleanup cloud graph
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
            client.user.delete(user_id=user_id)
        except Exception:
            pass
        return {"qid": qid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "zep"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "zep_lme_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    if not ZEP_API_KEY:
        print("❌ ZEP_API_KEY not set"); return output_file

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    # smoke = ONE knowledge-update question with FULL haystack (needs the update)
    if smoke:
        ku = [q for q in data if q["question_type"] == "knowledge-update"]
        data = ku[:1] if ku else data[:1]
    elif max_questions:
        data = _stratified_sample(data, max_questions)

    print(f"\n{'='*60}")
    print(f"  LongMemEval-S × Zep (cloud platform, temporal knowledge graph)")
    print(f"  ANSWER LLM : {os.getenv('OPENAI_MODEL', 'gemma-4-E4B-it')}  (shared judge LLM)")
    print(f"  questions  : {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    start = time.time()
    for idx, q in enumerate(data, 1):
        qid = q["question_id"]
        if qid in done:
            print(f"⏭️  [{idx}/{len(data)}] {qid} already done"); continue
        r = process_question(q, top_k, save_path, log_dir,
                             session_limit=None, llm_model=llm_model)  # full haystack
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['qid']} ({q['question_type']}) → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(save_path, "zep_lme")   # ANSWER-side tokens → zep_lme_token_usage.json
    meta = {"extraction_llm": "zep-cloud (server-side graph; tokens not exposed)",
            "judge_llm": os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"),
            "embed_model": "zep-cloud (managed)", "granularity": "session",
            "backend_detail": "Zep Cloud temporal knowledge graph; answer=shared LLM over graph facts"}
    with open(os.path.join(save_path, "zep_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
