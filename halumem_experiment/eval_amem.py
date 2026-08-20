"""
A-MEM adapter for HaluMem evaluation — turn-level memory processing.

Key difference from eval_mem0_oss.py:
  Mem0:  entire session dialogue passed at once → auto-extracts facts
  A-MEM: each user turn fed individually → organizes as Zettelkasten notes
         (one LLM call per user turn for keyword/context extraction)

Install A-MEM first:
  pip install git+https://github.com/agiresearch/A-mem.git

.env keys used:
  AMEM_BACKEND      = ollama | openai   (default: ollama)
  AMEM_LLM_MODEL    = model name        (default: gemma-4-E4B-it)
  AMEM_EMBED_MODEL  = sentence-transformer embedding model
  OLLAMA_BASE_URL   = http://localhost:11434 (used when AMEM_BACKEND=ollama)
  NCHC_API_KEY      = ...               (used when AMEM_BACKEND=openai)
  NCHC_BASE_URL     = ...               (used when AMEM_BACKEND=openai)
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

import token_tracker as _tk          # applies OpenAI patch immediately
from prompts import PROMPT_MEMZERO
from llms import llm_request

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
AMEM_BACKEND     = os.getenv("AMEM_BACKEND",     "ollama")
AMEM_LLM_MODEL   = os.getenv("AMEM_LLM_MODEL",   "gemma-4-E4B-it")
AMEM_EMBED_MODEL = os.getenv("AMEM_EMBED_MODEL",  "all-MiniLM-L6-v2")
AMEM_API_KEY     = os.getenv("NCHC_API_KEY",      "")
AMEM_BASE_URL    = os.getenv("NCHC_BASE_URL",     "https://portal.genai.nchc.org.tw/api/v1")
OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")

DATE_FORMAT = "%b %d, %Y, %H:%M:%S"

TEMPLATE_AMEM = """Memories for user {user_id}:

    {memories}
"""


# ── Patches ───────────────────────────────────────────────────────────────────

def _apply_patches():
    """
    Apply minimal patches to A-MEM before importing AgenticMemorySystem.

    Ollama backend: A-MEM uses litellm under the hood. Setting OLLAMA_API_BASE
    env var is all that's needed — no code-level patch required.

    OpenAI backend (NCHC): A-MEM's OpenAIController hardcodes the default
    OpenAI endpoint. We patch its __init__ to accept a custom base_url so
    it works with any OpenAI-compatible API.
    """
    # Ollama: litellm reads OLLAMA_API_BASE automatically
    if AMEM_BACKEND == "ollama" and OLLAMA_URL != "http://localhost:11434":
        os.environ["OLLAMA_API_BASE"] = OLLAMA_URL

    # OpenAI-compatible (NCHC): patch base_url into OpenAIController
    if AMEM_BACKEND == "openai":
        from agentic_memory.llm_controller import OpenAIController
        from openai import OpenAI as _OpenAI

        def _patched_init(self, model="gpt-4", api_key=None):
            self.model = model
            self.client = _OpenAI(
                api_key=api_key or AMEM_API_KEY,
                base_url=AMEM_BASE_URL,
            )

        OpenAIController.__init__ = _patched_init


_apply_patches()

from agentic_memory.memory_system import AgenticMemorySystem  # noqa: E402

# Patch litellm AFTER A-MEM is importable (A-MEM pulls in litellm on import)
_tk.patch_litellm()


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        return match.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name from persona_info: {persona_info[:100]}")


def setup_logger(log_dir: str, uuid: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem_amem.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _format_search_results(results: list) -> list[str]:
    """Convert A-MEM search results to timestamped strings."""
    out = []
    for item in results:
        content = item.get("content", "")
        ts      = item.get("timestamp", "")
        out.append(f"{ts}: {content}" if ts else content)
    return out


def _make_amem_instance(llm_model: str = None) -> AgenticMemorySystem:
    """
    Create a fresh, isolated A-MEM instance.

    Each call creates a new in-memory ChromaDB client (A-MEM resets the
    collection in __init__), so per-user isolation holds as long as users
    are processed sequentially in the same process.
    """
    return AgenticMemorySystem(
        model_name=AMEM_EMBED_MODEL,
        llm_backend=AMEM_BACKEND,
        llm_model=llm_model or AMEM_LLM_MODEL,
        api_key=AMEM_API_KEY if AMEM_BACKEND == "openai" else None,
    )


# ── Core processing ───────────────────────────────────────────────────────────

def process_user(
    user_data: dict,
    top_k: int,
    save_path: str,
    log_dir: str,
    smoke_session_limit: int = None,
    llm_model: str = None,
) -> dict:
    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    effective_model = llm_model or AMEM_LLM_MODEL
    logger.info(
        f"=== Start user {user_name} ({uuid}) | AMEM_LLM={effective_model} | "
        f"BACKEND={AMEM_BACKEND} ==="
    )

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file         = os.path.join(tmp_dir, f"{uuid}.json")
    session_log_file = os.path.join(log_dir, f"{uuid}_sessions.jsonl")

    # One A-MEM instance per user — fresh isolated memory store
    memory = _make_amem_instance(llm_model=llm_model)

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {
        "uuid":      uuid,
        "user_name": user_name,
        "sessions":  [],
    }

    try:
        for sid, session in enumerate(tqdm(sessions, desc=f"User {user_name}")):
            session_wall_start = time.time()

            new_session = {
                "memory_points": session["memory_points"],
                "dialogue":      session["dialogue"],
            }

            # A-MEM time format: YYYYMMDDHHMM
            dt = datetime.strptime(
                session["start_time"], DATE_FORMAT
            ).replace(tzinfo=timezone.utc)
            amem_time = dt.strftime("%Y%m%d%H%M")

            # ── Turn-level memory addition ───────────────────────────────
            # Feed each user turn to A-MEM individually.
            # A-MEM calls its LLM once per turn to extract keywords/context.
            session_note_ids = []
            add_total_ms = 0.0

            # Feed BOTH roles: the other backends ingest the whole dialogue and
            # HaluMem's ground truth includes assistant-sourced memories, so
            # user-only ingestion would handicap A-MEM non-architecturally.
            _turn_results = []
            for turn in session["dialogue"]:
                if not turn.get("content"):
                    continue
                # A-MEM writes at turn level, so one turn is one ingest unit and
                # its calls_per_unit is not directly comparable to session-level
                # backends. Use a normalizable quantity such as tokens per turn.
                t0 = time.time()
                with _tk.unit("ingest"):
                    note_id = memory.add_note(
                        content=f'{turn.get("role","user")}: {turn["content"]}',
                        time=amem_time,
                    )
                add_total_ms += (time.time() - t0) * 1000
                _turn_results.append({"role": turn.get("role"), "note_id": note_id})
                if note_id:
                    session_note_ids.append(note_id)

            # QA-only sessions: just track timing and move on
            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_total_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            # Collect stored note contents as "extracted memories"
            # read() returns a MemoryNote object; access .content attribute
            extracted_memories = []
            for nid in session_note_ids:
                note = memory.read(nid)
                if note is not None and note.content:
                    extracted_memories.append(note.content)

            # ── Update-probe instrumentation ────────────────────────────
            # A-MEM emits no ADD/UPDATE/DELETE events, so P2/P3 come from
            # snapshot diff over the whole note store.
            try:
                snap = [{"id": k, "memory": getattr(v, "content", None) or ""}
                        for k, v in (getattr(memory, "memories", {}) or {}).items()]
            except Exception as e:
                logger.warning(f"Session {sid} | snapshot failed: {e}")
                snap = []
            new_session["store_after"] = snap
            new_session["probe"] = {
                "turns_fed":       len(_turn_results),
                "notes_created":   sum(1 for r in _turn_results if r["note_id"]),
                "triggered_write": bool(session_note_ids),
                "store_size":      len(snap),
            }
            new_session["extracted_memories"]       = extracted_memories
            new_session["add_dialogue_duration_ms"] = add_total_ms

            user_turns = [t for t in session["dialogue"] if t.get("content")]
            logger.info(
                f"Session {sid} | {len(user_turns)} user turns → "
                f"{len(session_note_ids)} notes in {add_total_ms:.0f}ms"
            )

            # ── Update memory search ─────────────────────────────────────
            for memory_point in new_session["memory_points"]:
                if (memory_point["is_update"] == "False"
                        or not memory_point.get("original_memories")):
                    continue
                t0 = time.time()
                results   = memory.search_agentic(memory_point["memory_content"], k=10)
                search_ms = (time.time() - t0) * 1000
                memory_point["memories_from_system"] = _format_search_results(results)
                logger.debug(
                    f"Session {sid} | update_search "
                    f"'{memory_point['memory_content'][:50]}' "
                    f"→ {len(results)} results in {search_ms:.0f}ms"
                )

            # ── QA ───────────────────────────────────────────────────────
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            qa_log_entries = []

            for qa in session["questions"]:
                # search_agentic itself calls the LLM (A-MEM's agentic retrieval),
                # so its qa-bucket calls_per_unit exceeds 1. That is precisely the
                # architectural difference being measured.
                with _tk.unit("qa"):
                    t0 = time.time()
                    results   = memory.search_agentic(qa["question"], k=top_k)
                    search_ms = (time.time() - t0) * 1000

                    memories_list = _format_search_results(results)
                    context = TEMPLATE_AMEM.format(
                        user_id=user_name,
                        memories=json.dumps(memories_list, indent=4),
                    )
                    prompt = PROMPT_MEMZERO.format(context=context, question=qa["question"])

                    t0 = time.time()
                    response    = llm_request(prompt)
                    response_ms = (time.time() - t0) * 1000

                new_qa = copy.deepcopy(qa)
                new_qa["context"]              = context
                new_qa["search_duration_ms"]   = search_ms
                new_qa["system_response"]      = response
                new_qa["response_duration_ms"] = response_ms
                new_session["questions"].append(new_qa)

                qa_log_entries.append({
                    "question":               qa["question"],
                    "expected_answer":        qa["answer"],
                    "system_response":        response,
                    "question_type":          qa.get("question_type"),
                    "difficulty":             qa.get("difficulty"),
                    "retrieved_memory_count": len(memories_list),
                    "search_ms":              round(search_ms, 1),
                    "response_ms":            round(response_ms, 1),
                })
                logger.debug(
                    f"Session {sid} | QA '{qa['question'][:60]}' → '{response[:80]}'"
                )

            new_user_data["sessions"].append(new_session)

            session_elapsed_ms = (time.time() - session_wall_start) * 1000
            session_log = {
                "session_id":         sid,
                "start_time":         session["start_time"],
                "user_turn_count":    len(user_turns),
                "note_count":         len(session_note_ids),
                "extracted_memories": extracted_memories,
                "question_count":     len(new_session["questions"]),
                "qa_results":         qa_log_entries,
                "session_elapsed_ms": round(session_elapsed_ms, 1),
            }
            with open(session_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_log, ensure_ascii=False) + "\n")

            logger.info(
                f"Session {sid} done | {len(new_session['questions'])} QA | "
                f"elapsed {session_elapsed_ms:.0f}ms"
            )

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)

        logger.info(f"User {user_name} complete → {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb         = traceback.format_exc()
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"uuid": uuid, "status": "error", "path": error_path}


# ── Dataset iteration ─────────────────────────────────────────────────────────

def iter_jsonl(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_extraction(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    smoke: bool = False,
    smoke_session_limit: int = 1,
    llm_model: str = None,
    prompt_template: str = None,   # unused — A-MEM handles prompting internally
    prompt_params: dict = None,    # unused
    max_users: int = None,
    skip_users: int = 0,
) -> str:
    frame     = "amem"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_uuids  = {f[:-5]  for f in os.listdir(tmp_dir) if f.endswith(".json")}
    error_uuids = {f[:-10] for f in os.listdir(tmp_dir) if f.endswith("_error.log")}

    effective_model = llm_model or AMEM_LLM_MODEL
    print(f"\n{'='*60}")
    print(f"  HaluMem × A-MEM Extraction (turn-level)")
    print(f"  AMEM_LLM  : {effective_model}")
    print(f"  EMBED     : {AMEM_EMBED_MODEL}")
    print(f"  BACKEND   : {AMEM_BACKEND}")
    print(f"  DATA      : {data_path}")
    print(f"  VERSION   : {version}")
    print(f"  SMOKE     : {smoke}")
    if done_uuids:
        print(f"  RESUME    : {len(done_uuids)} users already done, skipping")
    if error_uuids:
        print(f"  RETRY     : {len(error_uuids)} errored users will be retried")
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

        if uuid in done_uuids and uuid not in error_uuids:
            print(f"⏭️  [{idx}/{total}] {uuid} already done, skipping")
            continue

        error_log = os.path.join(tmp_dir, f"{uuid}_error.log")
        if os.path.exists(error_log):
            os.remove(error_log)

        result = process_user(
            user_data=user_data,
            top_k=top_k,
            save_path=save_path,
            log_dir=log_dir,
            smoke_session_limit=smoke_session_limit if smoke else None,
            llm_model=llm_model,
        )
        icon = "✅" if result["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{total}] {result['uuid']} → {result['status']}")

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
