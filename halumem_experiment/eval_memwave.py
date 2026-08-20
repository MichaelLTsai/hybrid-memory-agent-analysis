"""
MemWeave adapter for HaluMem evaluation — session-level memory processing.

Architecture comparison:
  Mem0 (session-level):     entire session → LLM extraction → Qdrant (vector DB)
  MemWeave (session-level): entire session → LLM flush extraction → Markdown files + SQLite
                            Retrieval: hybrid BM25 + vector (sqlite-vec) + temporal decay + MMR

Install:
  pip install memweave

.env keys used:
  MEMWEAVE_LLM_MODEL   = LiteLLM model for flush extraction  (default: gemma-4-E4B-it)
  MEMWEAVE_EMBED_MODEL = LiteLLM embedding model             (default: bge-m3:latest)
  NCHC_API_KEY         = API key for NCHC endpoint
  NCHC_BASE_URL        = Base URL for NCHC endpoint
  OLLAMA_BASE_URL      = Ollama base URL (for embeddings)
"""

import os
import re
import json
import time
import asyncio
import copy
import logging
import traceback

from dotenv import load_dotenv
from tqdm import tqdm

import token_tracker as _tk          # applies OpenAI patch immediately
from prompts import PROMPT_MEMZERO
from llms import llm_request

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

NCHC_API_KEY  = os.getenv("NCHC_API_KEY", "")
NCHC_BASE_URL = os.getenv("NCHC_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1")
OLLAMA_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# MemWeave uses LiteLLM internally.
# For NCHC (OpenAI-compatible): "openai/model-name" + OPENAI_API_BASE env var
_raw_llm = os.getenv("MEMWEAVE_LLM_MODEL", "gemma-4-E4B-it")
MEMWEAVE_LLM_MODEL = f"openai/{_raw_llm}" if "/" not in _raw_llm else _raw_llm

_raw_embed = os.getenv("MEMWEAVE_EMBED_MODEL", "bge-m3:latest")
MEMWEAVE_EMBED_MODEL = f"ollama/{_raw_embed}" if "/" not in _raw_embed else _raw_embed

# Route LiteLLM to NCHC for flush LLM calls
os.environ.setdefault("OPENAI_API_BASE", NCHC_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY",  NCHC_API_KEY)

DATE_FORMAT = "%b %d, %Y, %H:%M:%S"

TEMPLATE_MEMWEAVE = """Memories for user {user_id}:

    {memories}
"""

# Extraction system prompt — instructs LLM to extract personal facts about the user
# (overrides MemWeave's default code-context prompt)
EXTRACTION_SYSTEM_PROMPT = (
    "You are a personal memory extraction assistant.\n"
    "Extract durable personal facts about the user from the conversation below.\n"
    "Write each fact as a separate bullet point (- fact).\n"
    "Focus on: personal details, preferences, relationships, events, and goals.\n"
    "Ignore assistant turns that don't contain user information.\n"
    "If there is nothing worth remembering, reply with @@SILENT_REPLY@@.\n"
    "Do NOT include timestamps or source labels."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        return match.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name from persona_info: {persona_info[:100]}")


def setup_logger(log_dir: str, uuid: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem_memweave.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _parse_flush_result(extracted_text: str | None) -> list[str]:
    """Parse LLM flush output into individual memory strings."""
    if not extracted_text or extracted_text.strip() == "@@SILENT_REPLY@@":
        return []
    facts = []
    for line in extracted_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip common list prefixes: "- ", "* ", "• ", "1. "
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        if line:
            facts.append(line)
    return facts


def _format_search_results(results: list) -> list[str]:
    """Convert MemWeave SearchResult list to text strings."""
    return [r.snippet for r in results if r.snippet]


def _make_memweave_config(workspace_dir: str, llm_model: str = None):
    from memweave import MemoryConfig
    from memweave.config import EmbeddingConfig, FlushConfig

    effective_llm = llm_model or MEMWEAVE_LLM_MODEL

    return MemoryConfig(
        workspace_dir=workspace_dir,
        embedding=EmbeddingConfig(
            model=MEMWEAVE_EMBED_MODEL,
            api_base=OLLAMA_URL if MEMWEAVE_EMBED_MODEL.startswith("ollama/") else None,
        ),
        flush=FlushConfig(
            enabled=True,
            model=effective_llm,
            max_tokens=1024,
            temperature=0.0,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        ),
        progress=False,
    )


# ── Async core (runs in a single event loop per user) ─────────────────────────

async def _process_user_async(
    user_data: dict,
    top_k: int,
    save_path: str,
    log_dir: str,
    store_base: str,
    smoke_session_limit: int,
    llm_model: str,
) -> dict:
    from memweave import MemWeave

    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)

    effective_llm = llm_model or MEMWEAVE_LLM_MODEL
    logger.info(
        f"=== Start user {user_name} ({uuid}) | LLM={effective_llm} | "
        f"EMBED={MEMWEAVE_EMBED_MODEL} ==="
    )

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file         = os.path.join(tmp_dir, f"{uuid}.json")
    session_log_file = os.path.join(log_dir, f"{uuid}_sessions.jsonl")

    # Per-user isolated workspace (each user gets their own .md files + SQLite)
    workspace = os.path.join(store_base, uuid)
    config    = _make_memweave_config(workspace, llm_model=effective_llm)

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {
        "uuid":      uuid,
        "user_name": user_name,
        "sessions":  [],
    }

    async with MemWeave(config) as mem:
        for sid, session in enumerate(tqdm(sessions, desc=f"User {user_name}")):
            session_wall_start = time.time()

            new_session = {
                "memory_points": session["memory_points"],
                "dialogue":      session["dialogue"],
            }

            # ── Session-level flush ──────────────────────────────────────
            # Build conversation list and pass to flush().
            # flush() calls LLM to extract durable facts → writes to memory/YYYY-MM-DD.md
            # Returns the extracted text (or None if nothing to store).
            conversation = [
                {"role": t["role"], "content": t["content"]}
                for t in session["dialogue"]
            ]

            t0 = time.time()
            extracted_text = await mem.flush(conversation)
            flush_ms = (time.time() - t0) * 1000

            extracted_memories = _parse_flush_result(extracted_text)

            # QA-only sessions: flush memory but skip extraction evaluation
            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = flush_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            new_session["extracted_memories"]       = extracted_memories
            new_session["add_dialogue_duration_ms"] = flush_ms

            logger.info(
                f"Session {sid} | flush → {len(extracted_memories)} facts in {flush_ms:.0f}ms"
            )

            # ── Update memory search ─────────────────────────────────────
            for memory_point in new_session["memory_points"]:
                if (memory_point["is_update"] == "False"
                        or not memory_point.get("original_memories")):
                    continue
                t0 = time.time()
                results   = await mem.search(memory_point["memory_content"], max_results=10)
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
                t0 = time.time()
                results   = await mem.search(qa["question"], max_results=top_k)
                search_ms = (time.time() - t0) * 1000

                memories_list = _format_search_results(results)
                context = TEMPLATE_MEMWEAVE.format(
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
                "extracted_memories": extracted_memories,
                "question_count":     len(new_session.get("questions", [])),
                "qa_results":         qa_log_entries,
                "session_elapsed_ms": round(session_elapsed_ms, 1),
            }
            with open(session_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_log, ensure_ascii=False) + "\n")

            logger.info(
                f"Session {sid} done | {len(new_session.get('questions', []))} QA | "
                f"elapsed {session_elapsed_ms:.0f}ms"
            )

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(new_user_data, f, ensure_ascii=False, indent=2)

    logger.info(f"User {user_name} complete → {tmp_file}")
    return {"uuid": uuid, "status": "ok", "path": tmp_file}


# ── Sync wrapper ──────────────────────────────────────────────────────────────

def process_user(
    user_data: dict,
    top_k: int,
    save_path: str,
    log_dir: str,
    store_base: str,
    smoke_session_limit: int = None,
    llm_model: str = None,
) -> dict:
    uuid = user_data["uuid"]
    try:
        return asyncio.run(
            _process_user_async(
                user_data=user_data,
                top_k=top_k,
                save_path=save_path,
                log_dir=log_dir,
                store_base=store_base,
                smoke_session_limit=smoke_session_limit,
                llm_model=llm_model,
            )
        )
    except Exception:
        tb         = traceback.format_exc()
        tmp_dir    = os.path.join(save_path, "tmp")
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(tb)
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
    prompt_template: str = None,   # unused — MemWeave handles prompting internally
    prompt_params: dict = None,    # unused
    max_users: int = None,
) -> str:
    frame      = "memwave"
    save_path  = f"./results/{frame}-{version}/"
    log_dir    = f"./logs/{frame}-{version}/"
    store_base = f"./results/{frame}-{version}/memory_store/"
    os.makedirs(save_path,  exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(store_base, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_uuids  = {f[:-5]  for f in os.listdir(tmp_dir) if f.endswith(".json")}
    error_uuids = {f[:-10] for f in os.listdir(tmp_dir) if f.endswith("_error.log")}

    effective_llm = f"openai/{llm_model}" if (llm_model and "/" not in llm_model) else (llm_model or MEMWEAVE_LLM_MODEL)

    print(f"\n{'='*60}")
    print(f"  HaluMem × MemWeave Extraction (session-level)")
    print(f"  FLUSH_LLM : {effective_llm}")
    print(f"  EMBED     : {MEMWEAVE_EMBED_MODEL}")
    print(f"  STORE     : {store_base}{{uuid}}/memory/")
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
            store_base=store_base,
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
