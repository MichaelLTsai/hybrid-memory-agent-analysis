"""
Letta (MemGPT) adapter for HaluMem — stateful agent with self-managed memory.

Paradigm difference from the other backends:
  Mem0/RAG/etc : we store memories, then retrieve + answer with a separate LLM
  Letta        : an AGENT self-manages memory (core blocks + archival) via tools,
                 and ANSWERS QUESTIONS ITSELF using its own LLM

CRUD framing:
  ① Extraction : agent decides what to write to memory (core_memory_append / archival_insert)
  ② CUD        : agent-driven — LLM edits core memory blocks, inserts archival
  ③ Storage    : core memory blocks (in-context) + archival passages (pgvector)
  ④ Retrieve   : agent calls tools to search archival; core memory always in context

Requires the Letta server running (see LETTA_SETUP.md). This adapter is a thin
HTTP client (letta_client) and can run in the main venv; the heavy server runs
in venv_letta talking to a home-dir postgres.

Feeding granularity: SESSION-level for the first pass — each non-QA session's
user turns are sent as one message so the agent updates its memory (turn-level
would be ~1400 agent calls/user = hours). extracted_memories = a snapshot of the
agent's core (human block) + archival memory after each session.

.env / env keys:
  LETTA_BASE_URL   = http://localhost:8283
  LETTA_LLM_MODEL  = openai-proxy/gemma-4-E4B-it   (agent LLM = answer LLM)
  LETTA_EMBED_MODEL= letta/letta-free
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

# Letta's LLM runs server-side and is invisible to the in-process OpenAI patch,
# so _tally() reports it manually from response.usage into the same counter,
# preserving identical phase and unit semantics.
import token_tracker as _tk

load_dotenv()

LETTA_BASE_URL    = os.getenv("LETTA_BASE_URL",   "http://localhost:8283")
LETTA_LLM_MODEL   = os.getenv("LETTA_LLM_MODEL",  "openai-proxy/gemma-4-E4B-it")
LETTA_EMBED_MODEL = os.getenv("LETTA_EMBED_MODEL", "letta/letta-free")

PERSONA = "I am a helpful assistant with long-term memory. I remember important facts the user shares."


def extract_user_name(persona_info: str) -> str:
    m = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if m:
        return m.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name from persona_info: {persona_info[:100]}")


def setup_logger(log_dir, uuid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem_letta.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _get_client():
    from letta_client import Letta
    return Letta(base_url=LETTA_BASE_URL)


# Letta runs its LLM calls server-side, so the eval-process token_tracker can't
# see them. Instead we accumulate the usage Letta reports on each response.
_LETTA_TOKENS = {"total": 0}


def _tally(resp, seconds: float):
    """Feed the usage the Letta server reports into token_tracker under the current phase.

    Letta is an agent: one messages.create may run several LLM steps server-side
    (reason, call a memory tool, heartbeat and think again). usage.step_count is
    the real number of LLM calls, not 1. This is its largest cost difference from
    pipeline-style backends and has to be captured.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        _tk.record_external(seconds=seconds, calls=1)
        return
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    tot = getattr(u, "total_tokens", None) or (pt + ct)
    steps = getattr(u, "step_count", None) or 1
    _LETTA_TOKENS["total"] += tot
    _tk.record_external(prompt_tokens=pt, completion_tokens=ct, total_tokens=tot,
                        seconds=seconds, calls=steps)


def _agent_answer(client, agent_id, message: str) -> str:
    """Send a message, return the agent's assistant_message text (and tally tokens)."""
    _t0 = time.time()
    resp = client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": message}],
    )
    _tally(resp, time.time() - _t0)
    answer = ""
    for m in resp.messages:
        if getattr(m, "message_type", "") == "assistant_message":
            c = getattr(m, "content", None)
            if c:
                answer = c if isinstance(c, str) else str(c)
    return answer


MEMORY_TOOLS = ("core_memory_append", "core_memory_replace", "archival_memory_insert",
                "memory_insert", "memory_replace", "memory_rethink")


def _dump_agent_context(client, agent_id, limit: int = 200) -> list:
    """
    The message history (recall memory) the agent can actually see at answering time.

    Letta has three memory tiers (core / recall / archival), but _dump_memory()
    covers only core blocks and archival passages, leaving the message history out
    entirely. A pipeline-style backend answers from its retrieved top-k alone,
    while Letta can read the raw text it was just fed, which makes the two P4
    figures incomparable. Recording the message history here lets P4 be defined
    uniformly as what the reader can see at answering time.

    This tier is dynamic: once the context window fills, older messages are
    evicted, so it can only be measured at each answering moment and is not a
    fixed store size.
    """
    out = []
    try:
        for m in client.agents.messages.list(agent_id=agent_id, limit=limit, order="desc"):
            c = getattr(m, "content", None)
            if not c:
                continue
            txt = c if isinstance(c, str) else str(c)
            role = getattr(m, "role", None) or getattr(m, "message_type", "")
            if len(txt) > 5:
                out.append(f"[{role}] {txt}")
    except Exception:
        pass
    return out


def _agent_answer_traced(client, agent_id, message: str):
    """Same as _agent_answer, but also reports which MEMORY TOOLS the agent
    invoked. For a self-managed agent this is the true P0 signal: a write only
    happens if the agent decides to call one -- unlike pipeline backends where
    writing is unconditional."""
    _t0 = time.time()
    resp = client.agents.messages.create(
        agent_id=agent_id, messages=[{"role": "user", "content": message}])
    _tally(resp, time.time() - _t0)
    answer, calls = "", []
    for m in resp.messages:
        mt = getattr(m, "message_type", "")
        if mt == "assistant_message":
            c = getattr(m, "content", None)
            if c:
                answer = c if isinstance(c, str) else str(c)
        elif mt == "tool_call_message":
            tc = getattr(m, "tool_call", None)
            name = getattr(tc, "name", None) if tc else None
            if name:
                calls.append(name)
    return answer, calls


def _store_dump(client, agent_id):
    """Snapshot as [{id, memory}] so P2/P3 can be recovered by diffing.
    Letta emits no ADD/UPDATE/DELETE events, so snapshot diff is the only way
    to see what happened to an old value."""
    out = []
    try:
        for b in client.agents.blocks.list(agent_id=agent_id):
            if b.label == "human" and b.value:
                for line in b.value.splitlines():
                    line = line.strip("-\u2022 \t")
                    if len(line) > 8:
                        out.append({"id": "core::" + line[:80], "memory": line})
    except Exception:
        pass
    try:
        for pg in client.agents.passages.list(agent_id=agent_id):
            t = getattr(pg, "text", None) or getattr(pg, "content", None)
            if t and getattr(pg, "id", None):
                out.append({"id": pg.id, "memory": t})
    except Exception:
        pass
    return out


def _dump_memory(client, agent_id) -> list[str]:
    """Snapshot the agent's memory: core (human block) + archival passages."""
    mems = []
    try:
        for b in client.agents.blocks.list(agent_id=agent_id):
            if b.label == "human" and b.value:
                # human block may be a multi-line summary; split into lines
                for line in b.value.splitlines():
                    line = line.strip("-• \t")
                    if len(line) > 8:
                        mems.append(line)
    except Exception:
        pass
    try:
        for p in client.agents.passages.list(agent_id=agent_id):
            t = getattr(p, "text", None) or getattr(p, "content", None)
            if t:
                mems.append(t)
    except Exception:
        pass
    return mems


def process_user(user_data, top_k, save_path, log_dir, smoke_session_limit=None, **kwargs):
    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    logger.info(f"=== Start user {user_name} ({uuid}) | LETTA_LLM={LETTA_LLM_MODEL} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

    client = _get_client()

    # One fresh agent per user (empty human block → learns from scratch)
    agent = client.agents.create(
        model=LETTA_LLM_MODEL,
        embedding=LETTA_EMBED_MODEL,
        memory_blocks=[
            {"label": "human",   "value": ""},
            {"label": "persona", "value": PERSONA},
        ],
    )
    agent_id = agent.id
    logger.info(f"Created agent {agent_id}")

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {"uuid": uuid, "user_name": user_name, "sessions": []}

    try:
        for sid, session in enumerate(tqdm(sessions, desc=f"User {user_name}")):
            new_session = {
                "memory_points": session["memory_points"],
                "dialogue":      session["dialogue"],
            }

            # ── Feed session (user turns) to the agent ───────────────────
            # Feed BOTH roles. The other backends (Mem0/MemOS/Zep) ingest the whole
            # dialogue, and HaluMem's ground truth includes memories sourced from
            # assistant turns -- user-only ingestion would handicap Letta for a
            # reason that is not architectural.
            convo = "\n".join(f'{t.get("role","user")}: {t.get("content","")}'
                               for t in session["dialogue"] if t.get("content"))
            feed_msg = (
                "Here is what I want to tell you in this session. "
                "Please remember the important, durable facts about me:\n\n" + convo
            )
            t0 = time.time()
            _tool_calls = []
            with _tk.unit("ingest"):          # one session is one ingest unit
                try:
                    _, _tool_calls = _agent_answer_traced(client, agent_id, feed_msg)
                except Exception as e:
                    logger.warning(f"Session {sid} feed error: {e}")
            add_ms = (time.time() - t0) * 1000

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            # ── Snapshot memory as extracted_memories ────────────────────
            extracted = _dump_memory(client, agent_id)
            snap = _store_dump(client, agent_id)
            mem_calls = [c for c in _tool_calls if c in MEMORY_TOOLS]
            new_session["store_after"] = snap
            new_session["probe"] = {
                "tool_calls":        _tool_calls,          # everything the agent called
                "memory_tool_calls": mem_calls,            # the write attempts -> P0
                "triggered_write":   bool(mem_calls),
                "store_size":        len(snap),
            }
            new_session["extracted_memories"]       = extracted
            new_session["add_dialogue_duration_ms"] = add_ms
            logger.info(f"Session {sid} | memory snapshot: {len(extracted)} items in {add_ms:.0f}ms")

            # ── Update memory search (agent-based) ───────────────────────
            for mp in new_session["memory_points"]:
                if mp["is_update"] == "False" or not mp.get("original_memories"):
                    continue
                mp["memories_from_system"] = extracted  # agent memory snapshot

            # ── QA (agent answers itself) ────────────────────────────────
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                t0 = time.time()
                with _tk.unit("qa"):          # one question is one qa unit
                    try:
                        response = _agent_answer(client, agent_id, qa["question"])
                    except Exception as e:
                        response = f"[agent error: {e}]"
                response_ms = (time.time() - t0) * 1000

                new_qa = copy.deepcopy(qa)
                # core blocks + archival (the older definition)
                new_qa["context"]              = "\n".join(extracted)
                # plus the message history at answering time, which is what makes
                # P4 mean the same thing as it does for the other backends
                new_qa["agent_context"]        = _dump_agent_context(client, agent_id)
                new_qa["search_duration_ms"]   = 0.0
                new_qa["system_response"]      = response
                new_qa["response_duration_ms"] = round(response_ms, 1)
                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        logger.info(f"User {user_name} complete → {tmp_file}")

        # Clean up the agent (keep the server tidy)
        try:
            client.agents.delete(agent_id=agent_id)
        except Exception:
            pass
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"uuid": uuid, "status": "error", "path": err}


def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   smoke_session_limit=1, llm_model=None, prompt_template=None,
                   prompt_params=None, max_users=None, skip_users=0) -> str:
    frame     = "letta"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    global LETTA_LLM_MODEL
    if llm_model:
        LETTA_LLM_MODEL = llm_model if "/" in llm_model else f"openai-proxy/{llm_model}"

    print(f"\n{'='*60}")
    print(f"  HaluMem × Letta (stateful agent, self-managed memory)")
    print(f"  AGENT LLM : {LETTA_LLM_MODEL}")
    print(f"  EMBED     : {LETTA_EMBED_MODEL}")
    print(f"  SERVER    : {LETTA_BASE_URL}")
    print(f"  DATA      : {data_path}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    # Verify server is up
    try:
        import requests
        requests.get(f"{LETTA_BASE_URL}/v1/health/", timeout=10).raise_for_status()
    except Exception as e:
        print(f"❌ Letta server not reachable at {LETTA_BASE_URL} — start it first (see LETTA_SETUP.md)\n{e}")
        return output_file

    _tk.reset()
    _LETTA_TOKENS["total"] = 0
    start = time.time()
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
        if uuid in done_uuids:
            print(f"⏭️  [{idx}/{total}] {uuid} already done, skipping")
            continue
        result = process_user(user_data, top_k, save_path, log_dir,
                              smoke_session_limit=smoke_session_limit if smoke else None)
        icon = "✅" if result["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{total}] {result['uuid']} → {result['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    # Letta reports usage per response (server-side LLM); we accumulated it.
    _tk.save(save_path, frame, extra={
        "note": "server-side LLM; figures come from Letta's response.usage, and "
                "calls uses usage.step_count (one messages.create may contain "
                "several agent steps)",
        "granularity": "session",
    })

    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
