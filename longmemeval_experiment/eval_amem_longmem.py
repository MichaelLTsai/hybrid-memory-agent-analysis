"""
A-MEM adapter for LongMemEval-S (independent folder; imports halumem shared code).

Per question: fresh A-MEM store built from that question's haystack, then answer.
A-MEM is turn-level (one note per turn); we map note_id → session_id so retrieval
resolves to SESSION-level provenance (matches answer_session_ids) → exact Recall@k.

⚠️ Turn-level over a ~48-session haystack = hundreds of add_note calls (each an LLM
keyword/tag/evolution pass) → slow. Use --max-questions for a small subset.
"""

import os
import sys
import json
import time
import logging
import traceback

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

# ── Force LOCAL Ollama for A-MEM ONLY (this adapter) ──────────────────────────
# A-MEM is turn-level with O(n²) note-evolution: over a ~48-session LongMemEval
# haystack that is hundreds of add_note calls, each an LLM pass. Via NCHC each
# call is a ~1min+ network round-trip → infeasible. gemma3n:e4b on the local
# M2 Ultra runs ~1s/call. We override the shared .env's A-MEM backend HERE only,
# before importing eval_amem (which reads these at module import). Other backends
# and the halumem .env are untouched.
os.environ["AMEM_BACKEND"]   = "ollama"
os.environ["AMEM_LLM_MODEL"] = "gemma3n:e4b"           # A-MEM → litellm ollama_chat/gemma3n:e4b
os.environ.setdefault("OLLAMA_API_BASE", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

import token_tracker as _tk
from eval_amem import _make_amem_instance, AMEM_LLM_MODEL, AMEM_EMBED_MODEL, AMEM_BACKEND
from llms import llm_request
_tk.patch_litellm()

QA_PROMPT = """You are answering a question using ONLY the memory notes retrieved from a user's long chat history.

Notes:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the notes above. If they do not contain enough information, say "No information available".
Answer:"""


def _parse_amem_time(s):
    try:
        from dateutil import parser as dp
        return dp.parse(str(s).split("(")[0].strip()).strftime("%Y%m%d%H%M")
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d%H%M")


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
    logger = logging.getLogger(f"lme_amem.{qid}")
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

    memory = _make_amem_instance(llm_model=llm_model)
    note2sid = {}
    sessions = q["haystack_sessions"]
    sess_ids = q["haystack_session_ids"]
    dates    = q.get("haystack_dates", [""] * len(sessions))
    if session_limit:
        sessions, sess_ids, dates = sessions[:session_limit], sess_ids[:session_limit], dates[:session_limit]

    try:
        t0 = time.time()
        for sess, sid, date in zip(sessions, sess_ids, dates):
            atime = _parse_amem_time(date)
            for turn in sess:
                content = turn.get("content", "")
                if not content:
                    continue
                with _tk.unit("ingest"):      # A-MEM is turn level: one turn is one unit
                    try:
                        nid = memory.add_note(content=f'{turn.get("role","user")}: {content}', time=atime)
                        if nid:
                            note2sid[nid] = sid
                    except Exception as e:
                        logger.warning(f"add_note error: {e}")
        logger.info(f"Fed {len(sessions)} sessions ({len(note2sid)} notes) in {(time.time()-t0):.0f}s")

        # ── Full memory-store dump → P1 (was the value ever stored at all?) ──
        #    A-MEM keeps every note in .memories (id → MemoryNote); note2sid carries
        #    the source session, so P1 can scope to the answer session instead of a
        #    lossy top-k over the whole store.
        try:
            all_memories = [{"text": getattr(n, "content", "") or "",
                             "session_id": note2sid.get(mid)}
                            for mid, n in (getattr(memory, "memories", None) or {}).items()
                            if getattr(n, "content", "")]
            logger.info(f"Dumped {len(all_memories)} memories")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            all_memories = None

        # Each question builds its own store, so one question is one qa unit;
        # search_agentic calls the LLM as well.
        with _tk.unit("qa"):
            results   = memory.search_agentic(q["question"], k=top_k)
            retrieved = [{"text": r.get("content", ""), "session_id": note2sid.get(r.get("id"))}
                         for r in results]
            mems = [r["text"] for r in retrieved]
            prompt = QA_PROMPT.format(context="\n".join(f"  - {m}" for m in mems) or "  (none)",
                                      date=q.get("question_date", ""), question=q["question"])
            t0 = time.time()
            response = llm_request(prompt)

        new_q = {k: q.get(k) for k in ["question_id", "question_type", "question", "answer",
                                       "question_date", "answer_session_ids"]}
        new_q.update({"retrieved": retrieved, "retrieved_memories": mems,
                      "all_memories": all_memories,
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
    frame     = "amem"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "amem_lme_results.jsonl")
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
    print(f"  LongMemEval-S × A-MEM (Zettelkasten, turn-level)")
    print(f"  LLM   : {llm_model or AMEM_LLM_MODEL} ({AMEM_BACKEND})  |  EMBED: {AMEM_EMBED_MODEL}")
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

    _tk.save(save_path, "amem_lme")
    meta = {"extraction_llm": llm_model or AMEM_LLM_MODEL, "judge_llm": os.getenv("OPENAI_MODEL", "unknown"),
            "embed_model": AMEM_EMBED_MODEL, "granularity": "turn"}
    with open(os.path.join(save_path, "amem_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
