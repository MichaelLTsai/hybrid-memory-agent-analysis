"""
MemOS (MemTensor) adapter for LongMemEval-S — tree/graph memory (the real MemOS).

Paradigm (same family as Mem0, opposite of Letta):
  MemOS is a MECHANICAL memory OS, not an autonomous agent. A mem_reader LLM
  distills each session into topic/concept/fact nodes stored in a Neo4j graph
  (vectors offloaded to Qdrant, because Neo4j Community has no native vector
  index). Retrieval is a graph + vector hybrid search — it does NOT depend on
  the chat model choosing to call a tool, so it does not collapse under a weak
  model the way Letta does.

Per QUESTION we build a FRESH MemCube from that question's haystack (each
question has its own haystack — no sharing) with a unique user_name (Neo4j
logical isolation) + unique Qdrant collection, feed each session tagged with
session_id, then search + answer with the SHARED judge LLM (same answer model
as Mem0/RAG → cross-backend comparable). session_id rides on each memory's
metadata → exact SESSION-level Recall@k / NDCG@k.

Infra (all self-hosted in $HOME, see LETTA_SETUP.md style):
  Neo4j 5.26 community  : bolt://localhost:7687  (neo4j / memos12345)
  Qdrant server         : localhost:6333         (vec store for graph nodes)
  Ollama                : localhost:11434        (bge-m3 embedder)
  NCHC openai-proxy     : gemma-4-E4B-it         (extractor + dispatcher + answer)

Run EXTRACTION under venv_memos with --skip-eval; EVALUATION under the main
venv with --eval-only (the judge in evaluation_longmem.py needs halumem llms).
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

from llms import llm_request  # shared answer LLM (gemma-4-E4B via NCHC), same as Mem0/RAG

# ── endpoints / models ───────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("MEMOS_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("MEMOS_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("MEMOS_NEO4J_PASSWORD", "memos12345")
QDRANT_HOST    = os.getenv("MEMOS_QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("MEMOS_QDRANT_PORT", "6333"))
EMBED_MODEL    = os.getenv("MEMOS_EMBED_MODEL", "bge-m3:latest")   # ollama, 1024-dim
EMBED_DIM      = int(os.getenv("MEMOS_EMBED_DIM", "1024"))
NCHC_BASE      = os.getenv("NCHC_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1"))
NCHC_KEY       = os.getenv("NCHC_API_KEY",  os.getenv("OPENAI_API_KEY", ""))
MEMOS_LLM      = os.getenv("MEMOS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"))

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a user's long chat history.

Memories:
{context}

Current date: {date}
Question: {question}

Answer concisely and factually based only on the memories above. If they do not contain enough information, say "No information available".
Answer:"""


def _llm_cfg(model=None):
    return {"backend": "openai", "config": {
        "model_name_or_path": model or MEMOS_LLM, "temperature": 0.0,
        "max_tokens": 2048, "api_key": NCHC_KEY, "api_base": NCHC_BASE}}


def _embed_cfg():
    return {"backend": "ollama", "config": {"model_name_or_path": EMBED_MODEL}}


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
    logger = logging.getLogger(f"lme_memos.{qid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{qid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _build_mos(qid, top_k, llm_model):
    """Fresh MOS + tree-memory MemCube isolated to this question."""
    from memos.configs.mem_os import MOSConfig
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.mem_os.main import MOS
    from memos.mem_cube.general import GeneralMemCube

    tag     = qid.replace("-", "_")[:40]
    uid     = f"memos_{tag}"
    cube_id = f"cube_{tag}"
    uname   = f"u_{tag}"                     # Neo4j logical-isolation key
    coll    = f"lme_memos_{tag}"             # Qdrant collection
    llm     = _llm_cfg(llm_model)
    emb     = _embed_cfg()

    mos_cfg = MOSConfig(
        user_id=uid, chat_model=llm,
        mem_reader={"backend": "simple_struct", "config": {
            "llm": llm, "embedder": emb,
            "chunker": {"backend": "sentence", "config": {
                "tokenizer_or_token_counter": "gpt2", "chunk_size": 512,
                "chunk_overlap": 128, "min_sentences_per_chunk": 1}}}},
        top_k=top_k, enable_textual_memory=True,
        enable_activation_memory=False, enable_parametric_memory=False)
    mos = MOS(mos_cfg)
    try:
        mos.create_user(user_id=uid)
    except Exception:
        pass

    cube_cfg = GeneralMemCubeConfig.model_validate({
        "user_id": uid, "cube_id": cube_id,
        "text_mem": {"backend": "tree_text", "config": {
            "extractor_llm": llm, "dispatcher_llm": llm, "embedder": emb,
            "graph_db": {"backend": "neo4j-community", "config": {
                "uri": NEO4J_URI, "user": NEO4J_USER, "password": NEO4J_PASSWORD,
                "db_name": "neo4j", "user_name": uname,
                "use_multi_db": False, "auto_create": False,
                "embedding_dimension": EMBED_DIM,
                "vec_config": {"backend": "qdrant", "config": {
                    "collection_name": coll, "vector_dimension": EMBED_DIM,
                    "distance_metric": "cosine", "host": QDRANT_HOST, "port": QDRANT_PORT}}}},
            "reorganize": False}},
        "act_mem": {"backend": "uninitialized"},
        "para_mem": {"backend": "uninitialized"}})
    cube = GeneralMemCube(cube_cfg)
    mos.register_mem_cube(cube, mem_cube_id=cube_id, user_id=uid)
    return mos, uid, uname, coll


def _parse_results(res, top_k):
    """MOSSearchResult.text_mem → [{text, session_id}] preserving rank order."""
    out = []
    for cube_res in res.get("text_mem", []):
        for m in cube_res.get("memories", []):
            txt = getattr(m, "memory", None)
            meta = getattr(m, "metadata", None)
            sid = getattr(meta, "session_id", None) if meta is not None else None
            if txt is None and isinstance(m, dict):
                txt = m.get("memory")
                sid = (m.get("metadata") or {}).get("session_id")
            if txt:
                out.append({"text": txt, "session_id": sid})
    return out[:top_k]


def _cleanup(uname, coll, logger):
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with drv.session() as s:
            s.run("MATCH (n {user_name:$u}) DETACH DELETE n", u=uname)
        drv.close()
    except Exception as e:
        logger.warning(f"neo4j cleanup failed: {e}")
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT).delete_collection(coll)
    except Exception as e:
        logger.warning(f"qdrant cleanup failed: {e}")


def process_question(q, top_k, save_path, log_dir, session_limit=None, llm_model=None):
    qid    = q["question_id"]
    logger = setup_logger(log_dir, qid)
    logger.info(f"=== Start question {qid} ({q['question_type']}) | MemOS tree | LLM={llm_model or MEMOS_LLM} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    sessions = q["haystack_sessions"]
    sess_ids = q["haystack_session_ids"]
    if session_limit:
        sessions, sess_ids = sessions[:session_limit], sess_ids[:session_limit]

    mos = uname = coll = None
    try:
        mos, uid, uname, coll = _build_mos(qid, top_k, llm_model)

        # ① Feed each haystack session (session_id → provenance) ──────────────
        t0 = time.time()
        for sess, sid in zip(sessions, sess_ids):
            messages = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                        for t in sess if t.get("content")]
            if messages:
                try:
                    mos.add(messages=messages, user_id=uid, session_id=sid)
                except Exception as e:
                    logger.warning(f"add {sid} error: {e}")
        logger.info(f"Fed {len(sessions)} sessions in {(time.time()-t0):.0f}s")

        # ② Retrieve (graph+vector hybrid) — NO session filter ────────────────
        res = mos.search(query=q["question"], user_id=uid, top_k=top_k, mode="fine")
        retrieved = _parse_results(res, top_k)
        n_sid = sum(1 for r in retrieved if r["session_id"])
        logger.info(f"retrieved {len(retrieved)} memories, {n_sid} with session provenance")

        # ③ Answer with the SHARED judge LLM (same as Mem0/RAG) ───────────────
        mems    = [r["text"] for r in retrieved]
        context = json.dumps(mems, indent=2, ensure_ascii=False)
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
            "system_response":    response,
            "response_ms":        round((time.time() - t0) * 1000, 1),
        }
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        logger.info(f"Question {qid} complete → {tmp_file}")
        _cleanup(uname, coll, logger)
        return {"qid": qid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{qid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        if uname:
            _cleanup(uname, coll, logger)
        return {"qid": qid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "memos"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "memos_lme_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        data = data[:1]
    elif max_questions:
        data = _stratified_sample(data, max_questions)

    print(f"\n{'='*60}")
    print(f"  LongMemEval-S × MemOS (tree/graph memory)")
    print(f"  LLM     : {llm_model or MEMOS_LLM}  (extractor+dispatcher+answer)")
    print(f"  EMBED   : ollama/{EMBED_MODEL}  ({EMBED_DIM}-dim)")
    print(f"  GRAPH   : {NEO4J_URI}  |  VEC: qdrant {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"  questions : {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

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

    meta = {"extraction_llm": llm_model or MEMOS_LLM,
            "judge_llm": os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"),
            "embed_model": f"ollama/{EMBED_MODEL}", "granularity": "session",
            "backend_detail": "MemOS tree_text (Neo4j community + Qdrant vec)"}
    with open(os.path.join(save_path, "memos_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
