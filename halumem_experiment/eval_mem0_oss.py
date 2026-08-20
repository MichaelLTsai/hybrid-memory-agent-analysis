"""
Mem0 OSS adapter for HaluMem evaluation.

Replaces eval_memzero.py (which uses Mem0 cloud API) with the
open-source Mem0 library backed by a local Ollama model.

Swap models by editing .env:
  MEM0_LLM_MODEL=qwen2.5:4b        # memory extraction LLM
  MEM0_EMBED_MODEL=nomic-embed-text # embedding model
  OPENAI_MODEL=qwen2.5:4b           # QA / evaluation judge LLM
"""

import os
import re
import json
import time
import copy
import logging
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm
import mem0
from mem0 import Memory
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Mem0 major version — controls additive (v2) vs LLM-CRUD (v1) behavior & config keys
MEM0_MAJOR = int(mem0.__version__.split(".")[0])

# Mem0 v1.x runs _add_to_vector_store in a worker thread (parallel with graph),
# but Qdrant-local's internal SQLite connection is bound to the creating thread,
# which raises "SQLite objects created in a thread can only be used in that same
# thread". Graph is disabled here, so only one thread ever writes — making it safe
# to relax check_same_thread globally. (v2.x doesn't need this.)
if MEM0_MAJOR < 2:
    import sqlite3 as _sqlite3
    _orig_sqlite_connect = _sqlite3.connect

    def _thread_safe_connect(*args, **kwargs):
        # Force-override: Qdrant-local passes check_same_thread=True explicitly,
        # so setdefault wouldn't help — we must overwrite it.
        kwargs["check_same_thread"] = False
        return _orig_sqlite_connect(*args, **kwargs)

    _sqlite3.connect = _thread_safe_connect

import token_tracker as _tk          # applies OpenAI patch immediately
from prompts import PROMPT_MEMZERO
from llms import llm_request

load_dotenv()

OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MEM0_LLM_MODEL   = os.getenv("MEM0_LLM_MODEL", "gemma-4-E4B-it")
MEM0_EMBED_MODEL = os.getenv("MEM0_EMBED_MODEL", "mxbai-embed-large:latest")
QDRANT_PATH      = os.getenv("QDRANT_PATH", "./qdrant_data")
NCHC_API_KEY     = os.getenv("NCHC_API_KEY", "")
NCHC_BASE_URL    = os.getenv("NCHC_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def render_prompt_template(template_file: str, params: dict) -> str:
    """Render a Jinja2 template from prompts/ directory."""
    env = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_file)
    return template.render(**params)

# Custom instructions passed to Mem0 so it produces richer, named memories
MEM0_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure
"""

TEMPLATE_MEM0 = """Memories for user {user_id}:

    {memories}
"""

DATE_FORMAT = "%b %d, %Y, %H:%M:%S"


def build_mem0_config(
    collection_name: str,
    qdrant_path: str,
    llm_model: str = None,
    custom_prompt: str = None,
) -> dict:
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model or MEM0_LLM_MODEL,
                "api_key": NCHC_API_KEY,
                "openai_base_url": NCHC_BASE_URL,
                "temperature": 0,
                "max_tokens": int(os.getenv("MEM0_MAX_TOKENS", "2000")),
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": MEM0_EMBED_MODEL,
                "ollama_base_url": OLLAMA_URL,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "path": qdrant_path,
                "embedding_model_dims": int(os.getenv("MEM0_EMBED_DIMS", "1024")),
            },
        },
    }

    if MEM0_MAJOR >= 2:
        # v2.x: additive extraction; custom_prompt is appended to the ADD-only prompt.
        config["custom_prompt"] = custom_prompt or MEM0_CUSTOM_INSTRUCTIONS
    else:
        # v1.x: LLM-driven ADD/UPDATE/DELETE. The fact-extraction prompt key differs,
        # and it REPLACES the default extraction prompt (which carries JSON format +
        # few-shot examples). Only override it when a custom prompt is explicitly given;
        # otherwise keep v1's robust defaults. Leave custom_update_memory_prompt at
        # default so the canonical v1 CRUD logic runs.
        if custom_prompt:
            config["custom_fact_extraction_prompt"] = custom_prompt

    return config


def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        # mem0 v2 does not allow whitespace in user_id
        return match.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name from persona_info: {persona_info[:100]}")


def mem_search(mem, query: str, user_name: str, limit: int):
    """Version-aware search. Param names differ between Mem0 majors:
        v1.x : search(query, user_id=..., limit=N)
        v2.x : search(query, filters={'user_id':...}, top_k=N)   # 'limit' is ignored!
    """
    if MEM0_MAJOR < 2:
        return mem.search(query, user_id=user_name, limit=limit)
    return mem.search(query, filters={"user_id": user_name}, top_k=limit)


def _format_search_results(results: list) -> list[str]:
    """Convert mem0 search results to timestamped strings."""
    out = []
    for item in results:
        ts = item.get("created_at") or item.get("metadata", {}).get("timestamp", "")
        memory = item.get("memory", "")
        out.append(f"{ts}: {memory}" if ts else memory)
    return out


def setup_logger(log_dir: str, uuid: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger



# ── Update-probe instrumentation helpers ────────────────────────────────────
def _attach_mem0_logger(logger):
    """Pipe Mem0's internal logger into our per-user log. Mem0 swallows LLM
    failures (logger.error/warning) and silently returns nothing, so without
    this an empty add() looks identical to correct deduplication."""
    import logging as _lg
    for name in ("mem0", "mem0.memory.main"):
        ml = _lg.getLogger(name)
        ml.setLevel(_lg.DEBUG)
        if not any(getattr(h, "_probe", False) for h in ml.handlers):
            for h in logger.handlers:
                h._probe = True
                ml.addHandler(h)


def _record_llm(mem, sink):
    """Wrap the Mem0 LLM so every raw response is captured.

    Mem0 returns only ADD/UPDATE/DELETE memories from add(); NONE decisions
    ("already known") never surface, so an empty result conflates successful
    dedup with an LLM failure. Recording the raw decision JSON recovers the
    complete action set, NONE included.
    """
    orig = mem.llm.generate_response

    def wrapped(*args, **kwargs):
        resp = orig(*args, **kwargs)
        sink.append(resp if isinstance(resp, str) else str(resp))
        return resp

    mem.llm.generate_response = wrapped
    return mem


def _parse_llm_calls(raw_calls):
    """Split recorded raw responses into the extraction step (facts) and the
    decision step (per-memory actions incl. NONE), and classify failures."""
    import json as _j, re as _re
    facts, actions, failures = [], [], []
    for r in raw_calls:
        txt = (r or "").strip()
        if not txt:
            failures.append("empty_llm_response"); continue
        body = _re.sub(r"^```(?:json)?|```$", "", txt, flags=_re.M).strip()
        try:
            obj = _j.loads(body, strict=False)
        except Exception:
            m = _re.search(r"\{.*\}", body, _re.S)
            try:
                obj = _j.loads(m.group(0), strict=False) if m else None
            except Exception:
                obj = None
        if obj is None:
            failures.append("unparseable_json"); continue
        if isinstance(obj, dict) and "facts" in obj:
            facts.extend(obj.get("facts") or [])
        elif isinstance(obj, dict) and "memory" in obj:
            for a in (obj.get("memory") or []):
                actions.append({"id": a.get("id"), "event": a.get("event"),
                                "text": a.get("text"), "old_memory": a.get("old_memory")})
    return facts, actions, failures


def process_user(
    user_data: dict,
    top_k: int,
    save_path: str,
    log_dir: str,
    qdrant_path: str,
    smoke_session_limit: int = None,
    llm_model: str = None,
    custom_prompt: str = None,
) -> dict:
    uuid = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger = setup_logger(log_dir, uuid)
    effective_model = llm_model or MEM0_LLM_MODEL
    logger.info(f"=== Start user {user_name} ({uuid}) | MEM0_LLM={effective_model} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")
    session_log_file = os.path.join(log_dir, f"{uuid}_sessions.jsonl")

    # One Mem0 instance per user, isolated collection
    collection_name = f"halumem_{uuid.replace('-', '_')[:40]}"
    config = build_mem0_config(collection_name, qdrant_path,
                               llm_model=llm_model, custom_prompt=custom_prompt)
    mem = Memory.from_config(config)
    _attach_mem0_logger(logger)
    _llm_raw = []
    _record_llm(mem, _llm_raw)

    # Clear any leftover memories from previous runs
    try:
        mem.delete_all(user_id=user_name)
        logger.info(f"Cleared existing memories for {user_name}")
    except Exception as e:
        logger.warning(f"delete_all failed (may be empty): {e}")

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {
        "uuid": uuid,
        "user_name": user_name,
        "sessions": [],
    }

    try:
        for sid, session in enumerate(tqdm(sessions, desc=f"User {user_name}")):
            session_wall_start = time.time()

            new_session = {
                "memory_points": session["memory_points"],
                "dialogue": session["dialogue"],
            }

            # Parse session timestamp
            dt = datetime.strptime(session["start_time"], DATE_FORMAT).replace(tzinfo=timezone.utc)

            # Format dialogue for Mem0
            formatted_dialogue = [
                {"role": t["role"], "content": t["content"]}
                for t in session["dialogue"]
            ]

            # --- Add dialogue to Mem0 ---
            # One session is one ingest unit. token_tracker derives from this how
            # many LLM calls, tokens, and seconds writing one session costs.
            t0 = time.time()
            _llm_raw.clear()
            with _tk.unit("ingest"):
                result = mem.add(
                    formatted_dialogue,
                    user_id=user_name,
                    metadata={"timestamp": dt.isoformat()},
                )
            add_ms = (time.time() - t0) * 1000
            _facts, _actions, _failures = _parse_llm_calls(list(_llm_raw))

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"] = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            # Extract memories returned by Mem0
            extracted_memories = [
                item["memory"] for item in result.get("results", [])
            ]
            new_session["extracted_memories"] = extracted_memories
            new_session["add_dialogue_duration_ms"] = add_ms

            # ── Update-probe instrumentation ────────────────────────────────
            # P3 (resolve): Mem0 reports its own decision per memory --
            #   ADD / UPDATE (+previous_memory = the overwritten old value) / DELETE / NONE
            # P2 (align):   full store snapshot lets us check whether the old
            #               value survived alongside the new one.
            from collections import Counter as _C
            new_session["probe"] = {
                "facts_extracted":  len(_facts),          # step 1 output
                "actions_all":      _actions,             # step 2: incl. NONE
                "action_counts":    dict(_C(a["event"] for a in _actions if a.get("event"))),
                "llm_calls":        len(_llm_raw),
                "llm_failures":     _failures,            # empty response / bad JSON
                "returned_n":       len(result.get("results", [])),
            }
            new_session["add_events"] = [
                {k: it.get(k) for k in ("id", "memory", "event", "previous_memory")}
                for it in result.get("results", [])
            ]
            try:
                # NOTE: get_all() defaults to limit=100 -- must raise it or the
                # snapshot silently truncates once the store exceeds 100 memories.
                snap = mem.get_all(user_id=user_name, limit=100000)
                snap_items = snap.get("results", snap) if isinstance(snap, dict) else snap
                new_session["store_after"] = [
                    {"id": it.get("id"), "memory": it.get("memory")}
                    for it in (snap_items or [])
                ]
            except Exception as e:
                logger.warning(f"Session {sid} | store snapshot failed: {e}")
            logger.info(
                f"Session {sid} | add_memory: {len(extracted_memories)} memories extracted in {add_ms:.0f}ms"
            )

            # --- Search for update memories ---
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory.get("original_memories"):
                    continue
                # This search exists for evaluation (producing memories_from_system
                # for the update metric), not something the memory system does while
                # processing a session, so it goes to other and not into ingest cost.
                t0 = time.time()
                with _tk.phase("other"):
                    search_res = mem_search(mem, memory["memory_content"], user_name, 10)
                search_ms = (time.time() - t0) * 1000
                memories_from_system = _format_search_results(
                    search_res.get("results", [])
                )
                memory["memories_from_system"] = memories_from_system
                logger.debug(
                    f"Session {sid} | update_search '{memory['memory_content'][:50]}...' "
                    f"→ {len(memories_from_system)} results in {search_ms:.0f}ms"
                )

            # --- QA ---
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            qa_log_entries = []

            for qa in session["questions"]:
                # One question is one qa unit, covering end-to-end retrieval plus generation cost.
                with _tk.unit("qa"):
                    t0 = time.time()
                    search_res = mem_search(mem, qa["question"], user_name, top_k)
                    search_ms = (time.time() - t0) * 1000

                    memories_list = _format_search_results(search_res.get("results", []))
                    context = TEMPLATE_MEM0.format(
                        user_id=user_name,
                        memories=json.dumps(memories_list, indent=4),
                    )

                    prompt = PROMPT_MEMZERO.format(context=context, question=qa["question"])

                    t0 = time.time()
                    response = llm_request(prompt)
                    response_ms = (time.time() - t0) * 1000

                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = search_ms
                new_qa["system_response"] = response
                new_qa["response_duration_ms"] = response_ms
                new_session["questions"].append(new_qa)

                qa_log_entries.append({
                    "question": qa["question"],
                    "expected_answer": qa["answer"],
                    "system_response": response,
                    "question_type": qa.get("question_type"),
                    "difficulty": qa.get("difficulty"),
                    "retrieved_memory_count": len(memories_list),
                    "search_ms": round(search_ms, 1),
                    "response_ms": round(response_ms, 1),
                })
                logger.debug(
                    f"Session {sid} | QA '{qa['question'][:60]}' → '{response[:80]}'"
                )

            new_user_data["sessions"].append(new_session)

            # Write per-session log entry
            session_elapsed_ms = (time.time() - session_wall_start) * 1000
            session_log = {
                "session_id": sid,
                "start_time": session["start_time"],
                "extracted_memory_count": len(extracted_memories),
                "extracted_memories": extracted_memories,
                "question_count": len(new_session["questions"]),
                "qa_results": qa_log_entries,
                "session_elapsed_ms": round(session_elapsed_ms, 1),
            }
            with open(session_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_log, ensure_ascii=False) + "\n")

            logger.info(
                f"Session {sid} done | {len(new_session['questions'])} QA pairs | "
                f"elapsed {session_elapsed_ms:.0f}ms"
            )

        # Save intermediate result for this user
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)

        logger.info(f"User {user_name} complete → {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"uuid": uuid, "status": "error", "path": error_path}


def iter_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_extraction(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    smoke: bool = False,
    smoke_session_limit: int = 1,
    llm_model: str = None,
    prompt_template: str = None,
    prompt_params: dict = None,
    max_users: int = None,
    skip_users: int = 0,
) -> str:
    """
    Run the Mem0 OSS memory extraction pipeline on HaluMem data.

    Returns the path to the merged output JSONL file.
    """
    frame = "mem0_oss"
    save_path = f"./results/{frame}-{version}/"
    log_dir = f"./logs/{frame}-{version}/"
    qdrant_path = os.path.join(save_path, "qdrant_data")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Render prompt template if provided, otherwise use default
    if prompt_template and prompt_params is not None:
        custom_prompt = render_prompt_template(prompt_template, prompt_params)
    else:
        custom_prompt = None  # will use MEM0_CUSTOM_INSTRUCTIONS

    # Scan already-completed users so we can resume after interruption
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}
    error_uuids = {f[:-10] for f in os.listdir(tmp_dir) if f.endswith("_error.log")}

    effective_model = llm_model or MEM0_LLM_MODEL
    print(f"\n{'='*60}")
    print(f"  HaluMem × Mem0 OSS Extraction")
    print(f"  MEM0_LLM  : {effective_model}")
    print(f"  EMBED     : {MEM0_EMBED_MODEL}")
    print(f"  PROMPT    : {prompt_template or 'default'} {prompt_params or ''}")
    print(f"  DATA      : {data_path}")
    print(f"  VERSION   : {version}")
    print(f"  SMOKE     : {smoke}")
    if done_uuids:
        print(f"  RESUME    : {len(done_uuids)} users already done, skipping")
    if error_uuids:
        print(f"  RETRY     : {len(error_uuids)} previously errored users will be retried")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    start_time = time.time()
    users = list(iter_jsonl(data_path))
    if skip_users:
        users = users[skip_users:]
    if smoke:
        users = users[:1]
    elif max_users:
        users = users[:max_users]

    total = len(users)
    for idx, user_data in enumerate(users, 1):
        uuid = user_data["uuid"]

        # Skip already-completed users (resume support)
        if uuid in done_uuids and uuid not in error_uuids:
            print(f"⏭️  [{idx}/{total}] {uuid} already done, skipping")
            continue

        # Remove stale error log before retrying
        error_log = os.path.join(tmp_dir, f"{uuid}_error.log")
        if os.path.exists(error_log):
            os.remove(error_log)

        result = process_user(
            user_data=user_data,
            top_k=top_k,
            save_path=save_path,
            log_dir=log_dir,
            qdrant_path=qdrant_path,
            smoke_session_limit=smoke_session_limit if smoke else None,
            llm_model=llm_model,
            custom_prompt=custom_prompt,
        )
        status_icon = "✅" if result["status"] == "ok" else "❌"
        print(f"{status_icon} [{idx}/{total}] {result['uuid']} → {result['status']}")

    # Merge tmp files into final JSONL
    with open(output_file, "w", encoding="utf-8") as f_out:
        for fname in sorted(os.listdir(tmp_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(tmp_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f_in:
                    data = json.load(f_in)
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️  Skipped {fname}: {e}")

    elapsed = time.time() - start_time
    _tk.save(save_path, frame)
    print(f"\n✅ Extraction done in {elapsed:.1f}s")
    print(f"✅ Results → {output_file}")
    return output_file
