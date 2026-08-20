"""
Letta (MemGPT) adapter for LongMemEval-S — stateful agent, self-managed memory.
Ported from halumem_experiment/eval_letta.py (same proven paradigm).

Paradigm (same as HaluMem):
  Mem0/RAG : we store memories, then retrieve + answer with a separate LLM
  Letta    : an AGENT self-manages memory (core blocks + archival) via tools,
             and ANSWERS THE QUESTION ITSELF using its own LLM.

Per QUESTION we spin up a FRESH agent, feed that question's haystack (one message
per session; turn-level would be thousands of agent calls), then the agent answers
the single question. extracted_memories = snapshot of the agent's core (human block)
+ archival passages at QA time.

Retrieval metrics (Recall@k / NDCG@k) — how we get session provenance out of a
system that self-manages its memory:
  ① PROVENANCE is exact, not inferred. We feed sessions ONE AT A TIME, so after
     each session we diff the agent's passage-id set and human-block lines; whatever
     is new was written while digesting that session → memory_id ➜ session_id.
     Same fidelity as Mem0's metadata={"session_id": ...}, just recovered by
     write-order instead of being handed to the store.
  ② RANKING is the agent's own retrieval, not a whole-memory dump. At QA time we
     call Letta's archival search (GET /v1/agents/{id}/archival-memory/search?query=)
     with the question — that returns passages ranked by semantic similarity, which
     is exactly what Recall@k / NDCG@k are meant to score.
  ③ Core (human-block) memories rank FIRST: in Letta core memory sits in the
     context window unconditionally, so it is always "retrieved" at answer time.

Differences vs the HaluMem adapter (LongMemEval-specific):
  • feed BOTH user & assistant turns (single-session-assistant answers live in
    assistant turns), not just user turns
  • prepend each session's date (temporal-reasoning questions need it)

⚠️ Dependencies: letta_client + requests only. Run EXTRACTION under venv_letta with
   --skip-eval, then EVALUATION under the main venv with --eval-only (the judge in
   evaluation_longmem.py needs the halumem `llms` module).

env keys (same as halumem):
  LETTA_BASE_URL   = http://localhost:8283
  LETTA_LLM_MODEL  = openai-proxy/gemma-4-E4B-it   (agent LLM = answer LLM)
  LETTA_EMBED_MODEL= letta/letta-free              (superseded — see EMBEDDING below)

EMBEDDING (why an explicit embedding_config, not the LETTA_EMBED_MODEL handle):
  Letta's semantic archival memory NEEDS a real embedding endpoint. On this box:
    • letta/letta-free    → needs Letta Cloud, falls back to NCHC → NCHC has no
                            /embeddings endpoint → 404
    • ollama/bge-m3       → Letta's built-in ollama handle POSTs to
                            {base}/embeddings, but ollama serves /v1/embeddings
                            (OpenAI-compat) → 404
  Fix: give the agent an explicit embedding_config of type "openai" pointing at
  ollama's OpenAI-compatible endpoint (…:11434/v1). Letta then POSTs to
  …/v1/embeddings → 200. This is the SAME bge-m3 the other backends use, so
  Letta's archival retrieval is embedding-comparable to Mem0 / RAG.
"""

import os
import sys
import json
import time
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

# Explicit embedding_config → ollama bge-m3 over its OpenAI-compatible endpoint.
# (See module docstring for why the plain LETTA_EMBED_MODEL handle 404s here.)
OLLAMA_OPENAI_BASE = os.getenv("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")
LETTA_EMBED_CONFIG = {
    "embedding_endpoint_type": "openai",
    "embedding_endpoint":      OLLAMA_OPENAI_BASE,
    "embedding_model":         os.getenv("LETTA_EMBED_OLLAMA_MODEL", "bge-m3:latest"),
    "embedding_dim":           int(os.getenv("LETTA_EMBED_DIM", "1024")),
    "embedding_chunk_size":    300,
}
PERSONA = "I am a helpful assistant with long-term memory. I remember important facts the user shares."

# Letta runs its LLM calls server-side, so the eval-process token_tracker can't see
# them. Instead we accumulate the usage Letta reports on each response.
_LETTA_TOKENS = {"total": 0}


def _get_client():
    from letta_client import Letta
    return Letta(base_url=LETTA_BASE_URL)


def _tally(resp, seconds: float):
    """Feed the usage the Letta server reports into token_tracker under the current phase.

    One messages.create may run several LLM steps server-side, and
    usage.step_count is the real call count.
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
    """Send a message, return the agent's assistant_message text (and tally tokens)."""
    _t0 = time.time()
    resp = client.agents.messages.create(
        agent_id=agent_id, messages=[{"role": "user", "content": message}])
    _tally(resp, time.time() - _t0)
    answer = ""
    for m in resp.messages:
        if getattr(m, "message_type", "") == "assistant_message":
            c = getattr(m, "content", None)
            if c:
                answer = c if isinstance(c, str) else str(c)
    return answer


def _core_lines(client, agent_id) -> list:
    """Current lines of the agent's core 'human' block (always in context)."""
    out = []
    try:
        for b in client.agents.blocks.list(agent_id=agent_id):
            if b.label == "human" and b.value:
                for line in b.value.splitlines():
                    line = line.strip("-• \t")
                    if len(line) > 8:
                        out.append(line)
    except Exception:
        pass
    return out


def _passages(client, agent_id) -> dict:
    """Current archival passages as {passage_id: text}."""
    out = {}
    try:
        for p in client.agents.passages.list(agent_id=agent_id):
            t = getattr(p, "text", None) or getattr(p, "content", None)
            if t and getattr(p, "id", None):
                out[p.id] = t
    except Exception:
        pass
    return out


def _archival_search(agent_id, query, top_k) -> list:
    """The agent's OWN ranked retrieval for this query → [{id, content}, ...].

    Goes over REST because the pinned letta_client does not expose this endpoint;
    `requests` is already a dependency of this adapter.
    """
    import requests
    r = requests.get(
        f"{LETTA_BASE_URL}/v1/agents/{agent_id}/archival-memory/search",
        params={"query": query, "top_k": top_k}, timeout=120)
    r.raise_for_status()
    return [{"id": x.get("id"), "content": x.get("content", "")}
            for x in (r.json().get("results") or [])]


def _dump_memory(client, agent_id) -> list:
    """Whole-memory snapshot (core + archival), unranked. Kept for logging/back-compat."""
    return _core_lines(client, agent_id) + list(_passages(client, agent_id).values())


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
    logger = logging.getLogger(f"lme_letta.{qid}")
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
    logger.info(f"=== Start question {qid} ({q['question_type']}) | LETTA_LLM={LETTA_LLM_MODEL} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{qid}.json")

    client = _get_client()
    # Letta needs a provider/model handle; a bare model name must get the proxy prefix.
    model_handle = llm_model or LETTA_LLM_MODEL
    if "/" not in model_handle:
        model_handle = f"openai-proxy/{model_handle}"
    # One fresh agent per question (empty human block → learns this haystack from scratch)
    agent = client.agents.create(
        model=model_handle,
        embedding_config=LETTA_EMBED_CONFIG,
        memory_blocks=[{"label": "human", "value": ""},
                       {"label": "persona", "value": PERSONA}],
    )
    agent_id = agent.id
    logger.info(f"Created agent {agent_id}")

    sessions = q["haystack_sessions"]
    dates    = q.get("haystack_dates", [""] * len(sessions))
    sess_ids = q.get("haystack_session_ids", [None] * len(sessions))
    if session_limit:
        sessions, dates = sessions[:session_limit], dates[:session_limit]
        sess_ids = sess_ids[:session_limit]

    try:
        # ① Feed each haystack session, one at a time, recording provenance ───
        # After each session we diff the agent's memory: anything newly written
        # was written while digesting THAT session → exact memory ➜ session_id.
        t0 = time.time()
        prov = {}                     # passage_id | "core::<line>"  ➜  session_id
        seen_pids, seen_core = set(), set()
        for sess, date, sid in zip(sessions, dates, sess_ids):
            body = "\n".join(f'{t.get("role","user")}: {t.get("content","")}'
                             for t in sess if t.get("content"))
            if not body:
                continue
            feed_msg = (f"[Session date: {date}] Here is a past conversation session. Please remember "
                        f"the important, durable facts the user shares:\n\n{body}")
            with _tk.unit("ingest"):          # one session is one ingest unit
                try:
                    _agent_answer(client, agent_id, feed_msg)
                except Exception as e:
                    logger.warning(f"feed session error: {e}")
            for pid, _t in _passages(client, agent_id).items():
                if pid not in seen_pids:
                    seen_pids.add(pid); prov[pid] = sid
            for line in _core_lines(client, agent_id):
                if line not in seen_core:
                    seen_core.add(line); prov[f"core::{line}"] = sid
        logger.info(f"Fed {len(sessions)} sessions in {(time.time()-t0):.0f}s | "
                    f"provenance: {len(prov)} memories over {len(set(prov.values()))} sessions")

        # ② Build the RANKED retrieval list for this question ─────────────────
        # core memory first (unconditionally in context), then the agent's own
        # semantic search over archival memory.
        ranked = [{"text": l, "session_id": prov.get(f"core::{l}"), "source": "core"}
                  for l in _core_lines(client, agent_id)]
        try:
            hits = _archival_search(agent_id, q["question"], top_k)
        except Exception as e:
            logger.warning(f"archival search failed, falling back to unranked passages: {e}")
            hits = [{"id": pid, "content": t} for pid, t in _passages(client, agent_id).items()][:top_k]
        ranked += [{"text": h["content"], "session_id": prov.get(h["id"]), "source": "archival"}
                   for h in hits]
        extracted = [r["text"] for r in ranked]
        n_attr = sum(1 for r in ranked if r["session_id"])
        logger.info(f"retrieved {len(ranked)} memories ({len(ranked)-len(hits)} core / {len(hits)} archival), "
                    f"{n_attr} with session provenance")

        # ③ QA — the agent answers itself ─────────────────────────────────────
        t0 = time.time()
        prompt = (f"Today's date is {q.get('question_date','')}. Using only what you remember from our "
                  f"past conversations, answer concisely. If you do not know, say so.\n\n"
                  f"Question: {q['question']}")
        with _tk.unit("qa"):                  # each question builds its own store, so one question is one qa unit
            try:
                response = _agent_answer(client, agent_id, prompt)
            except Exception as e:
                response = f"[agent error: {e}]"
        response_ms = (time.time() - t0) * 1000

        new_q = {k: q.get(k) for k in ["question_id", "question_type", "question", "answer",
                                       "question_date", "answer_session_ids"]}
        new_q.update({
            "retrieved":          ranked,
            "retrieved_memories": extracted,
            "all_memories":       _dump_memory(client, agent_id),
            "system_response":    response,
            "response_ms":        round(response_ms, 1),
        })
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_q, f, ensure_ascii=False, indent=2)
        logger.info(f"Question {qid} complete → {tmp_file}")
        try:
            client.agents.delete(agent_id=agent_id)
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
            client.agents.delete(agent_id=agent_id)
        except Exception:
            pass
        return {"qid": qid, "status": "error", "path": err}


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_questions=None, llm_model=None) -> str:
    frame     = "letta"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "letta_lme_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    global LETTA_LLM_MODEL
    if llm_model:
        LETTA_LLM_MODEL = llm_model if "/" in llm_model else f"openai-proxy/{llm_model}"

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    if smoke:
        data = data[:2]
    elif max_questions:
        data = _stratified_sample(data, max_questions)

    print(f"\n{'='*60}")
    print(f"  LongMemEval-S × Letta (stateful agent, self-managed memory)")
    print(f"  AGENT LLM : {LETTA_LLM_MODEL}")
    print(f"  EMBED     : {LETTA_EMBED_MODEL}")
    print(f"  SERVER    : {LETTA_BASE_URL}")
    print(f"  questions : {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    try:
        import requests
        requests.get(f"{LETTA_BASE_URL}/v1/health/", timeout=10).raise_for_status()
    except Exception as e:
        print(f"❌ Letta server not reachable at {LETTA_BASE_URL} — start it first (see LETTA_SETUP.md)\n{e}")
        return output_file

    _LETTA_TOKENS["total"] = 0
    _tk.reset()
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

    _tk.save(save_path, "letta_lme", extra={
        "note": "server-side LLM; figures come from Letta's response.usage, and "
                "calls uses usage.step_count (one messages.create may contain "
                "several agent steps)",
        "granularity": "session",
    })
    meta = {"extraction_llm": (llm_model or LETTA_LLM_MODEL).replace("openai-proxy/", ""),
            "judge_llm": os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"),
            "embed_model": f"ollama/{LETTA_EMBED_CONFIG['embedding_model']}", "granularity": "session",
            "provenance": "write-order diff (exact); ranking = Letta archival-memory search"}
    with open(os.path.join(save_path, "letta_lme_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s | tokens: {_LETTA_TOKENS['total']:,} → {output_file}")
    return output_file
