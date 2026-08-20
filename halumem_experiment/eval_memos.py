"""
MemOS (MemTensor) adapter for HaluMem -- tree/graph memory.

MemOS is a MECHANICAL memory OS: a mem_reader LLM distills each session into
topic/concept/fact nodes in a Neo4j graph (vectors in Qdrant, since Neo4j
Community has no native vector index). Retrieval is graph+vector hybrid --
it does not depend on the chat model deciding to call a tool.

HaluMem is PROGRESSIVE: per session we ingest the dialogue, snapshot the NEW
memories (extracted_memories), then answer that session's questions from the
memory accumulated so far.

Update-probe instrumentation (P2/P3):
  MemOS reports no ADD/UPDATE/DELETE events, so resolution is recovered by
  SNAPSHOT DIFF -- after every session we dump the user's whole graph
  (`store_after`: id + text). Comparing consecutive snapshots yields, per
  memory: still present / text changed / disappeared -> DESTROYED /
  UNTOUCHED / COEXIST classification for the old value.

Infra (self-hosted in $HOME):
  Neo4j 5.26 community : bolt://localhost:7687
  Qdrant server        : localhost:6333
  Ollama               : localhost:11434 (bge-m3 embedder, 1024-dim)
  NCHC openai-proxy    : extraction/dispatcher LLM (--llm-model)

Run EXTRACTION under venv_memos with --skip-eval; EVALUATION under the main venv.
"""

import os
import re
import json
import time
import copy
import logging
import traceback

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import token_tracker as _tk
from prompts import PROMPT_MEMZERO
from llms import llm_request

NEO4J_URI      = os.getenv("MEMOS_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("MEMOS_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("MEMOS_NEO4J_PASSWORD", "memos12345")
QDRANT_HOST    = os.getenv("MEMOS_QDRANT_HOST", "localhost")
QDRANT_PORT    = int(os.getenv("MEMOS_QDRANT_PORT", "6333"))
EMBED_MODEL    = os.getenv("MEMOS_EMBED_MODEL", "bge-m3:latest")
EMBED_DIM      = int(os.getenv("MEMOS_EMBED_DIM", "1024"))
NCHC_BASE      = os.getenv("NCHC_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
NCHC_KEY       = os.getenv("NCHC_API_KEY",  os.getenv("OPENAI_API_KEY", ""))
MEMOS_LLM      = os.getenv("MEMOS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"))

TEMPLATE_MEMOS = """Memories for user {user_id}:
{memories}"""


def extract_user_name(persona_info):
    m = re.search(r"Name:\s*(.*?); Gender:", persona_info or "")
    return m.group(1).strip().replace(" ", "_") if m else "user"


def setup_logger(log_dir, uuid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("halumem_memos." + uuid)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, uuid + ".log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _llm_cfg(model=None):
    return {"backend": "openai", "config": {
        "model_name_or_path": model or MEMOS_LLM, "temperature": 0.0,
        "max_tokens": 2048, "api_key": NCHC_KEY, "api_base": NCHC_BASE}}


def _embed_cfg():
    return {"backend": "ollama", "config": {"model_name_or_path": EMBED_MODEL}}


def _build_mos(tag, top_k, llm_model):
    from memos.configs.mem_os import MOSConfig
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.mem_os.main import MOS
    from memos.mem_cube.general import GeneralMemCube

    uid     = "halumem_" + tag
    cube_id = "cube_" + tag
    uname   = "u_" + tag
    coll    = "halumem_memos_" + tag
    llm = _llm_cfg(llm_model); emb = _embed_cfg()

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


def _store_dump(uname):
    """Full graph snapshot for this user: [{id, memory}] -- feeds P2/P3 by diff."""
    out = []
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with drv.session() as s:
            for r in s.run("MATCH (n:Memory) WHERE n.user_name=$u RETURN n.id AS id, n.memory AS m", u=uname):
                if r["m"]:
                    out.append({"id": r["id"], "memory": r["m"]})
        drv.close()
    except Exception:
        pass
    return out


def _parse_results(res, top_k):
    out = []
    for cube_res in res.get("text_mem", []):
        for m in cube_res.get("memories", []):
            txt = getattr(m, "memory", None)
            if txt is None and isinstance(m, dict):
                txt = m.get("memory")
            if txt:
                out.append(txt)
    return out[:top_k]


def _cleanup(uname, coll, logger):
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with drv.session() as s:
            s.run("MATCH (n {user_name:$u}) DETACH DELETE n", u=uname)
        drv.close()
    except Exception as e:
        logger.warning("neo4j cleanup failed: " + str(e))
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT).delete_collection(coll)
    except Exception as e:
        logger.warning("qdrant cleanup failed: " + str(e))


def process_user(user_data, top_k, save_path, log_dir, smoke_session_limit=None,
                 llm_model=None, **kwargs):
    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    logger.info("=== Start user " + user_name + " (" + uuid + ") | MemOS tree | LLM=" + str(llm_model or MEMOS_LLM) + " ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, uuid + ".json")

    tag = re.sub(r"[^a-zA-Z0-9]", "", uuid)[:28]
    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {"uuid": uuid, "user_name": user_name, "sessions": []}
    mos = uname = coll = None
    seen_ids = set()

    try:
        mos, uid, uname, coll = _build_mos(tag, top_k, llm_model)
        _cleanup(uname, coll, logger)              # isolation: start from a clean graph
        mos, uid, uname, coll = _build_mos(tag, top_k, llm_model)

        for sid, session in enumerate(tqdm(sessions, desc="User " + user_name)):
            new_session = {"memory_points": session["memory_points"],
                           "dialogue":      session["dialogue"]}
            messages = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                        for t in session["dialogue"] if t.get("content")]

            t0 = time.time()
            try:
                mos.add(messages=messages, user_id=uid, session_id="s" + str(sid))
            except Exception as e:
                logger.warning("S" + str(sid) + " add error: " + str(e))
            add_ms = (time.time() - t0) * 1000

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]; del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            # ── Update-probe instrumentation ────────────────────────────────
            # MemOS emits no events; snapshot diff recovers what happened to
            # old values (P3) and whether v1 survives beside v2 (P2).
            snap = _store_dump(uname)
            new_session["store_after"] = snap
            new_ids = [x for x in snap if x["id"] not in seen_ids]
            seen_ids |= {x["id"] for x in snap}
            extracted_memories = [x["memory"] for x in new_ids]

            new_session["extracted_memories"]       = extracted_memories
            new_session["add_dialogue_duration_ms"] = add_ms
            logger.info("S" + str(sid) + ": +" + str(len(extracted_memories)) +
                        " new (" + str(len(snap)) + " total) in " + str(round(add_ms)) + "ms")

            for mp in new_session["memory_points"]:
                if mp["is_update"] == "False" or not mp.get("original_memories"):
                    continue
                try:
                    r = mos.search(query=mp["memory_content"], user_id=uid, top_k=10, mode="fine")
                    mp["memories_from_system"] = _parse_results(r, 10)
                except Exception:
                    mp["memories_from_system"] = []

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                t0 = time.time()
                try:
                    res = mos.search(query=qa["question"], user_id=uid, top_k=top_k, mode="fine")
                    mems = _parse_results(res, top_k)
                except Exception:
                    mems = []
                search_ms = (time.time() - t0) * 1000
                context = TEMPLATE_MEMOS.format(user_id=user_name,
                            memories=json.dumps(mems, indent=4, ensure_ascii=False))
                prompt  = PROMPT_MEMZERO.format(context=context, question=qa["question"])
                t0 = time.time()
                response = llm_request(prompt)
                new_qa = copy.deepcopy(qa)
                new_qa["context"]              = context
                new_qa["search_duration_ms"]   = round(search_ms, 1)
                new_qa["system_response"]      = response
                new_qa["response_duration_ms"] = round((time.time() - t0) * 1000, 1)
                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        logger.info("User " + user_name + " complete -> " + tmp_file)
        _cleanup(uname, coll, logger)
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, uuid + "_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error("FAILED:\n" + tb)
        if uname:
            _cleanup(uname, coll, logger)
        return {"uuid": uuid, "status": "error", "path": err}


def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   smoke_session_limit=1, llm_model=None, prompt_template=None,
                   prompt_params=None, max_users=None, skip_users=0):
    frame     = "memos"
    save_path = "./results/" + frame + "-" + version + "/"
    log_dir   = "./logs/" + frame + "-" + version + "/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, frame + "_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    print("\n" + "="*60)
    print("  HaluMem x MemOS (tree/graph memory)")
    print("  LLM     : " + str(llm_model or MEMOS_LLM) + "  (extractor+dispatcher)")
    print("  ANSWER  : " + os.getenv("OPENAI_MODEL", "gemma-4-E4B-it") + "  (shared)")
    print("  EMBED   : ollama/" + EMBED_MODEL)
    print("  DATA    : " + str(data_path) + " | VERSION: " + version + " | SMOKE: " + str(smoke))
    print("="*60 + "\n")

    _tk.tracker.reset()
    start = time.time()
    users = list(iter_jsonl(data_path))
    if skip_users:
        users = users[skip_users:]
    if smoke:
        users = users[:1]
    elif max_users:
        users = users[:max_users]

    for idx, user_data in enumerate(users, 1):
        uuid = user_data["uuid"]
        if uuid in done_uuids:
            print("[" + str(idx) + "/" + str(len(users)) + "] " + uuid + " already done"); continue
        r = process_user(user_data, top_k, save_path, log_dir,
                         smoke_session_limit=smoke_session_limit if smoke else None,
                         llm_model=llm_model)
        icon = "OK" if r["status"] == "ok" else "ERR"
        print(icon + " [" + str(idx) + "/" + str(len(users)) + "] " + r["uuid"] + " -> " + r["status"])

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    with open(os.path.join(save_path, frame + "_token_usage.json"), "w") as f:
        json.dump({"total_tokens": _tk.tracker.total,
                   "note": "answer-side LLM; MemOS extraction LLM goes through the same NCHC proxy"}, f)
    print("Answer-LLM tokens: " + str(_tk.tracker.total))
    print("\nExtraction done in " + str(round(time.time()-start, 1)) + "s -> " + output_file)
    return output_file
