"""
StructMem adapter for HaluMem -- structure-enriched hierarchical memory.

StructMem (Zhejiang Univ. + Ant Group) is not a standalone package: it is
LightMem run in `extraction_mode="event"` plus periodic cross-event
consolidation via summarize(). Two levels:

  1. Event-level bindings   -- dual-perspective extraction per utterance:
     FACTUAL entries (who/what/when/where) + RELATIONAL entries (interpersonal
     dynamics, causal influence, temporal dependency), each anchored to its
     originating timestamp. No rigid schema, unlike graph memory.
  2. Cross-event consolidation -- summarize() periodically retrieves similar
     historical entries as seeds, rebuilds their event context, and synthesises
     higher-level relational hypotheses across time.

Design choices for a fair cross-architecture comparison:
  · pre_compress / topic_segment OFF -- the reference config needs LLMLingua-2
    on CUDA; disabling keeps the comparison about MEMORY, not compression, and
    lets it run on this machine.
  · messages_use = "hybrid"  -- feeds user AND assistant turns, matching the
    other backends (HaluMem ground truth includes assistant-sourced memories).
  · memory_manager = openai backend pointed at the NCHC proxy, so extraction
    uses the same gemma-4-31B as Mem0/MemOS/Letta/A-MEM.

Update-probe instrumentation: StructMem reports no ADD/UPDATE/DELETE events,
so P2/P3 come from snapshot diff over embedding_retriever.get_all().

Runs under ~/structmem_env (LightMem needs Python <3.12).
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

NCHC_KEY  = os.getenv("NCHC_API_KEY",  os.getenv("OPENAI_API_KEY", ""))
NCHC_BASE = os.getenv("NCHC_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
SM_LLM    = os.getenv("STRUCTMEM_LLM_MODEL", os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"))
EMBED     = os.getenv("STRUCTMEM_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_structmem")
# Cross-event consolidation is expensive; run it every N sessions.
SUMMARIZE_EVERY = int(os.getenv("STRUCTMEM_SUMMARIZE_EVERY", "10"))

TEMPLATE_SM = """Memories for user {user_id}:
{memories}"""

DATE_FORMAT = "%b %d, %Y, %H:%M:%S"


def extract_user_name(persona_info):
    m = re.search(r"Name:\s*(.*?); Gender:", persona_info or "")
    return m.group(1).strip().replace(" ", "_") if m else "user"


def setup_logger(log_dir, uuid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("halumem_structmem." + uuid)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, uuid + ".log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _build_config(tag, llm_model):
    return {
        "pre_compress":    False,
        "topic_segment":   False,
        "messages_use":    "hybrid",
        "metadata_generate": True,
        "text_summary":    True,
        "extraction_mode": "event",          # <- StructMem
        "memory_manager": {
            "model_name": "openai",
            "configs": {"model": llm_model or SM_LLM, "api_key": NCHC_KEY,
                        "openai_base_url": NCHC_BASE,
                        "max_tokens": 8192, "temperature": 0.0},
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {"model_name": "huggingface",
            "configs": {"model": EMBED, "embedding_dims": 384,
                        "model_kwargs": {"device": "cpu"}}},
        "retrieve_strategy": "embedding",
        "embedding_retriever": {"model_name": "qdrant",
            "configs": {"collection_name": "sm_" + tag,
                        "embedding_model_dims": 384,
                        "path": os.path.join(QDRANT_BASE, tag)}},
        "summary_retriever": {"model_name": "qdrant",
            "configs": {"collection_name": "smsum_" + tag,
                        "embedding_model_dims": 384,
                        "path": os.path.join(QDRANT_BASE, tag + "_sum")}},
    }


def _store_dump(lm):
    """All entries as [{id, memory}] -- feeds P2/P3 by snapshot diff."""
    out = []
    try:
        for e in (lm.embedding_retriever.get_all() or []):
            pl = getattr(e, "payload", None) or (e.get("payload") if isinstance(e, dict) else {}) or {}
            txt = pl.get("memory") or pl.get("text") or pl.get("content")
            eid = getattr(e, "id", None) or (e.get("id") if isinstance(e, dict) else None)
            if txt:
                out.append({"id": str(eid), "memory": txt})
    except Exception:
        pass
    return out


def _to_messages(dialogue, base_time):
    msgs = []
    for t in dialogue:
        if not t.get("content"):
            continue
        msgs.append({"role": t.get("role", "user"), "content": t["content"],
                     "time_stamp": base_time})
    return msgs


def process_user(user_data, top_k, save_path, log_dir, smoke_session_limit=None,
                 llm_model=None, **kwargs):
    from datetime import datetime
    from lightmem.memory.lightmem import LightMemory

    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    logger.info("=== Start user " + user_name + " (" + uuid + ") | StructMem (event mode) | LLM=" +
                str(llm_model or SM_LLM) + " ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, uuid + ".json")

    tag = re.sub(r"[^a-zA-Z0-9]", "", uuid)[:28]
    import shutil
    shutil.rmtree(os.path.join(QDRANT_BASE, tag), ignore_errors=True)          # isolation
    shutil.rmtree(os.path.join(QDRANT_BASE, tag + "_sum"), ignore_errors=True)

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {"uuid": uuid, "user_name": user_name, "sessions": []}
    seen_ids = set()
    lm = None

    try:
        lm = LightMemory.from_config(_build_config(tag, llm_model))

        for sid, session in enumerate(tqdm(sessions, desc="User " + user_name)):
            new_session = {"memory_points": session["memory_points"],
                           "dialogue":      session["dialogue"]}
            try:
                dt = datetime.strptime(session["start_time"], DATE_FORMAT)
                ts = dt.strftime("%Y/%m/%d (%a) %H:%M")
            except Exception:
                ts = session.get("start_time", "")

            t0 = time.time()
            n_fail = 0
            # One session is one ingest unit.
            with _tk.unit("ingest"):
                try:
                    lm.add_memory(_to_messages(session["dialogue"], ts), force_extract=True)
                except Exception as e:
                    n_fail = 1
                    logger.warning("S" + str(sid) + " add error: " + str(e))
            add_ms = (time.time() - t0) * 1000

            # Cross-event consolidation: the second half of StructMem.
            # This block runs only once every SUMMARIZE_EVERY sessions, but it is
            # part of StructMem's own write cost, so it goes into the ingest bucket
            # without counting as a new unit. That way calls_per_unit reflects the
            # real cost amortized across every session.
            did_sum = False
            if (sid + 1) % SUMMARIZE_EVERY == 0:
                with _tk.phase("ingest"):
                    try:
                        lm.summarize(process_all=True, enable_cross_event=True,
                                     retrieval_scope="global", top_k_seeds=15)
                        did_sum = True
                    except Exception as e:
                        logger.warning("S" + str(sid) + " summarize error: " + str(e))
                    # Conflict resolution. summarize() only consolidates across
                    # events and never replaces an old value with a new one.
                    # LightMem's update mechanism is these two offline batch
                    # methods, which earlier runs never invoked, so HaluMem's low
                    # memory_update score measured a disabled feature rather than a
                    # poor mechanism. Set STRUCTMEM_OFFLINE_UPDATE=0 to disable it
                    # again as a control.
                    if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") == "1":
                        try:
                            lm.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                            lm.offline_update_all_entries(score_threshold=0.9)
                        except Exception as e:
                            logger.warning(f"offline_update error: {e}")

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]; del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            snap = _store_dump(lm)
            new_ids = [x for x in snap if x["id"] not in seen_ids]
            seen_ids |= {x["id"] for x in snap}
            extracted = [x["memory"] for x in new_ids]

            new_session["store_after"]              = snap
            new_session["extracted_memories"]       = extracted
            new_session["add_dialogue_duration_ms"] = add_ms
            new_session["probe"] = {
                "new_entries":     len(extracted),
                "triggered_write": bool(extracted),
                "store_size":      len(snap),
                "consolidated":    did_sum,
                "add_failures":    n_fail,
            }
            logger.info("S" + str(sid) + ": +" + str(len(extracted)) + " new (" +
                        str(len(snap)) + " total)" + (" [consolidated]" if did_sum else ""))

            for mp in new_session["memory_points"]:
                if mp["is_update"] == "False" or not mp.get("original_memories"):
                    continue
                # A probing retrieval for evaluation, not part of the architecture's
                # write path, so it goes to other.
                with _tk.phase("other"):
                    try:
                        mp["memories_from_system"] = lm.retrieve(mp["memory_content"], limit=10)
                    except Exception:
                        mp["memories_from_system"] = []

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                with _tk.unit("qa"):              # one question is one qa unit
                    t0 = time.time()
                    try:
                        mems = lm.retrieve(qa["question"], limit=top_k)
                    except Exception:
                        mems = []
                    search_ms = (time.time() - t0) * 1000
                    context = TEMPLATE_SM.format(user_id=user_name,
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
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, uuid + "_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error("FAILED:\n" + tb)
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
    frame     = "structmem"
    save_path = "./results/" + frame + "-" + version + "/"
    log_dir   = "./logs/" + frame + "-" + version + "/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, frame + "_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    print("\n" + "="*60)
    print("  HaluMem x StructMem (LightMem event mode + cross-event consolidation)")
    print("  EXTRACT LLM : " + str(llm_model or SM_LLM))
    print("  ANSWER  LLM : " + os.getenv("OPENAI_MODEL", "gemma-4-E4B-it") + "  (shared)")
    print("  EMBED       : " + EMBED + "  (cpu)")
    print("  consolidate : every " + str(SUMMARIZE_EVERY) + " sessions")
    print("  DATA        : " + str(data_path) + " | VERSION: " + version + " | SMOKE: " + str(smoke))
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

    _tk.save(save_path, frame, extra={
        "note": "StructMem routes both extraction and answering through the same "
                "NCHC proxy; the ingest bucket covers add_memory plus the "
                "summarize/offline_update that runs once every "
                + str(SUMMARIZE_EVERY) + " sessions (amortized into calls_per_unit)",
        "granularity": "session",
    })
    print("\nExtraction done in " + str(round(time.time()-start, 1)) + "s -> " + output_file)
    return output_file
