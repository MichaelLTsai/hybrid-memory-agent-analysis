"""
Letta (MemGPT) adapter for LoCoMo — stateful agent, self-managed memory.

Talks to the Letta server over HTTP (letta_client). The heavy server runs in
venv_letta against a home-dir postgres (see ../halumem_experiment/LETTA_SETUP.md).

Paradigm: an AGENT self-manages memory and ANSWERS the questions itself using its
own LLM (openai-proxy/gemma-4-E4B-it — same family as the other backends).

Feeding: SESSION-level (one message per session; turn-level would be thousands of
agent calls). No turn-level dia_id provenance → retrieval Recall@k is null.
Tokens are summed from Letta's per-response usage (server-side LLM).
"""

import os
import sys
import json
import time
import copy
import logging
import traceback

# Letta's LLM runs server-side and is invisible to the in-process OpenAI patch,
# so _tally() reports it manually from response.usage into the same counter,
# preserving identical phase and unit semantics.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")))
import token_tracker as _tk

LETTA_BASE_URL    = os.getenv("LETTA_BASE_URL",   "http://localhost:8283")
LETTA_LLM_MODEL   = os.getenv("LETTA_LLM_MODEL",  "openai-proxy/gemma-4-E4B-it")
LETTA_EMBED_MODEL = os.getenv("LETTA_EMBED_MODEL", "letta/letta-free")
PERSONA = "I am a helpful assistant with long-term memory. I remember important facts people share."

_LETTA_TOKENS = {"total": 0}


def _client():
    from letta_client import Letta
    return Letta(base_url=LETTA_BASE_URL)


def _tally(resp, seconds: float):
    """Feed the usage the Letta server reports into token_tracker under the current phase.

    One messages.create may run several LLM steps server-side (reason, call a
    memory tool, heartbeat), and usage.step_count is the real call count.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        _tk.record_external(seconds=seconds, calls=1)
        return
    pt = getattr(u, "prompt_tokens", 0) or 0
    ct = getattr(u, "completion_tokens", 0) or 0
    tot = getattr(u, "total_tokens", None) or (pt + ct)
    _LETTA_TOKENS["total"] += tot
    _tk.record_external(prompt_tokens=pt, completion_tokens=ct, total_tokens=tot,
                        seconds=seconds, calls=getattr(u, "step_count", None) or 1)


def _agent_answer(client, agent_id, message: str) -> str:
    _t0 = time.time()
    resp = client.agents.messages.create(agent_id=agent_id, messages=[{"role": "user", "content": message}])
    _tally(resp, time.time() - _t0)
    answer = ""
    for m in resp.messages:
        if getattr(m, "message_type", "") == "assistant_message":
            c = getattr(m, "content", None)
            if c:
                answer = c if isinstance(c, str) else str(c)
    return answer


def _dump_memory(client, agent_id) -> list[str]:
    mems = []
    try:
        for b in client.agents.blocks.list(agent_id=agent_id):
            if b.label == "human" and b.value:
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

def _session_keys(conv):
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_letta.{cid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{cid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def process_conversation(sample, top_k, save_path, log_dir, session_limit=None, qa_limit=None, llm_model=None):
    cid    = sample["sample_id"]
    conv   = sample["conversation"]
    logger = setup_logger(log_dir, cid)
    logger.info(f"=== Start conversation {cid} (Letta) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    client = _client()
    agent = client.agents.create(
        model=llm_model or LETTA_LLM_MODEL,
        embedding=LETTA_EMBED_MODEL,
        memory_blocks=[{"label": "human", "value": ""},
                       {"label": "persona", "value": PERSONA}],
    )
    agent_id = agent.id
    new_sample = {"sample_id": cid, "speaker_a": conv.get("speaker_a"),
                  "speaker_b": conv.get("speaker_b"), "qa": []}

    try:
        # ① Feed each session to the agent (one message/session)
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        for sk in sess_keys:
            body = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in conv[sk])
            msg = ("Here is a conversation session. Please remember the important, durable "
                   "facts about the people in it:\n\n" + body)
            with _tk.unit("ingest"):          # one session is one ingest unit
                try:
                    _agent_answer(client, agent_id, msg)
                except Exception as e:
                    logger.warning(f"feed {sk} error: {e}")
        logger.info(f"Fed {len(sess_keys)} sessions in {(time.time()-t0):.0f}s")

        mem_snapshot = _dump_memory(client, agent_id)

        # ── Full memory-store dump → extraction_locomo.py (integrity/accuracy/F1) ──
        #    Taken right after ingest, BEFORE the QA phase: Letta answers as an agent
        #    and may write new memories while answering, which would contaminate a
        #    dump taken later. No turn-level provenance (fed session-by-session and
        #    the agent rewrites its own blocks) → dia_id is None, so extraction_locomo
        #    falls back to scope="global" and marks the run as not directly comparable
        #    with session-scoped backends.
        new_sample["memory_dump"] = [{"text": m, "dia_id": None} for m in mem_snapshot]
        logger.info(f"Dumped {len(new_sample['memory_dump'])} memories (no dia_id provenance)")

        # ② QA — the agent answers itself
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            t0 = time.time()
            with _tk.unit("qa"):              # one question is one qa unit
                try:
                    response = _agent_answer(client, agent_id, qa["question"])
                except Exception as e:
                    response = f"[agent error: {e}]"
            new_qa = copy.deepcopy(qa)
            new_qa["retrieved"]          = [{"text": m, "dia_id": None} for m in mem_snapshot]
            new_qa["retrieved_memories"] = mem_snapshot
            # The message history at answering time (recall memory). Pipeline-style
            # backends have only their retrieved top-k, while Letta can also see the
            # raw text it was fed. Without recording this, P4 would underestimate it
            # and would not mean the same thing as it does for the other backends.
            new_qa["agent_context"]      = _dump_agent_context(client, agent_id)
            new_qa["system_response"]    = response
            new_qa["response_ms"]        = round((time.time() - t0) * 1000, 1)
            new_sample["qa"].append(new_qa)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_sample, f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation {cid} complete → {tmp_file}")
        try:
            client.agents.delete(agent_id=agent_id)
        except Exception:
            pass
        return {"cid": cid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{cid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"cid": cid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_convs=None, llm_model=None) -> str:
    frame     = "letta"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "letta_locomo_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        data = data[:1]
    elif max_convs:
        data = data[:max_convs]

    print(f"\n{'='*60}")
    print(f"  LoCoMo × Letta (stateful agent, self-managed memory)")
    print(f"  AGENT LLM : {llm_model or LETTA_LLM_MODEL}")
    print(f"  SERVER    : {LETTA_BASE_URL}")
    print(f"  conversations: {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    try:
        import requests
        requests.get(f"{LETTA_BASE_URL}/v1/health/", timeout=10).raise_for_status()
    except Exception as e:
        print(f"❌ Letta server not reachable — start it first (see halumem LETTA_SETUP.md)\n{e}")
        return output_file

    _LETTA_TOKENS["total"] = 0
    _tk.reset()
    start = time.time()
    for idx, sample in enumerate(data, 1):
        cid = sample["sample_id"]
        if cid in done:
            print(f"⏭️  [{idx}/{len(data)}] {cid} already done"); continue
        r = process_conversation(sample, top_k, save_path, log_dir,
                                 session_limit=3 if smoke else None,
                                 qa_limit=5 if smoke else None, llm_model=llm_model)
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['cid']} → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(save_path, "letta_locomo", extra={
        "note": "server-side LLM; figures come from Letta's response.usage, and "
                "calls uses usage.step_count (one messages.create may contain "
                "several agent steps)",
        "granularity": "session",
    })
    meta = {
        "extraction_llm": (llm_model or LETTA_LLM_MODEL).replace("openai-proxy/", ""),
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    LETTA_EMBED_MODEL,
        "granularity":    "session",
    }
    with open(os.path.join(save_path, "letta_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s | tokens: {_LETTA_TOKENS['total']:,} → {output_file}")
    return output_file
