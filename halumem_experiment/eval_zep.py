
"""
Zep (Cloud / Platform) adapter for HaluMem - temporal knowledge graph memory.

Zep is a MANAGED cloud platform: dialogue is uploaded to Zep, which builds a
per-user temporal knowledge graph (facts = edges with valid_at / invalid_at, so
knowledge UPDATES are modelled natively). Mechanical, like Mem0 - not an agent.

HaluMem is PROGRESSIVE: for each session we ingest its dialogue, then answer
THAT session's questions using memory accumulated so far. So we ingest session i,
wait for Zep to finish building the graph, snapshot the NEW facts (extracted_memories
for session i = graph edges that appeared during session i), then answer.

PACKED ingestion (chosen for feasibility): HaluMem sessions are dense (~43 turns).
One Zep message per turn would be ~2800 episodes/user (~15-20h async build). Instead
each session's turns are concatenated as "role: content" lines split into <=4096-char
messages -> ~200-350 episodes/user (~2-3h). Provenance stays session-level.

Isolation: one Zep user per HaluMem user (deleted before + after) -> independent graph.

Tokens: answer-side LLM (gemma-4-E4B via NCHC, shared) counted by token_tracker ->
zep_token_usage.json. Zep server-side graph-building tokens are not exposed.

env: ZEP_API_KEY  (+ .env for the shared answer LLM)
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

ZEP_API_KEY     = os.getenv("ZEP_API_KEY", "")
PROCESS_TIMEOUT = int(os.getenv("ZEP_PROCESS_TIMEOUT", "1200"))
PACK_LIMIT      = 4000

TEMPLATE_ZEP = """Memories for user {user_id}:
{memories}"""


def extract_user_name(persona_info):
    m = re.search(r"Name:\s*(.*?); Gender:", persona_info or "")
    return m.group(1).strip().replace(" ", "_") if m else "user"


def _client():
    from zep_cloud.client import Zep
    return Zep(api_key=ZEP_API_KEY)


def setup_logger(log_dir, uuid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("halumem_zep." + uuid)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, uuid + ".log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _pack_dialogue(dialogue):
    text = "\n".join((t.get("role", "user") + ": " + t.get("content", ""))
                     for t in dialogue if t.get("content"))
    return [text[i:i+PACK_LIMIT] for i in range(0, len(text), PACK_LIMIT)]


def _episode_counts(client, user_id):
    try:
        eps = client.graph.episode.get_by_user_id(user_id=user_id, lastn=5000)
        items = getattr(eps, "episodes", None) or []
        return len(items), sum(1 for e in items if getattr(e, "processed", False))
    except Exception:
        return 0, 0


def _wait_processed(client, user_id, baseline, logger, timeout):
    t0 = time.time(); prev = -1; stable = 0
    while time.time() - t0 < timeout:
        total, done = _episode_counts(client, user_id)
        if total > baseline:
            stable = stable + 1 if total == prev else 0
            prev = total
            if done == total and stable >= 2:
                return total
        time.sleep(12)
    logger.warning("  wait timeout after " + str(timeout) + "s (episodes seen=" + str(prev) + ")")
    return prev


def _all_edges_status(client, user_id):
    """Paginated full-store dump WITH temporal status -- Zep marks superseded
    facts via invalid_at/expired_at, so P3 can read INVALIDATED_KEPT directly."""
    out = []
    cursor = None
    for _ in range(60):
        try:
            page = client.graph.edge.get_by_user_id(user_id=user_id, limit=100,
                                                     uuid_cursor=cursor) or []
        except Exception:
            break
        if not page:
            break
        before = len(out)
        for e in page:
            if getattr(e, "fact", None) and getattr(e, "uuid_", None):
                out.append({"id": e.uuid_, "memory": e.fact,
                            "valid_at":   str(getattr(e, "valid_at", None) or ""),
                            "invalid_at": str(getattr(e, "invalid_at", None) or ""),
                            "expired_at": str(getattr(e, "expired_at", None) or "")})
        cursor = getattr(page[-1], "uuid_", None)
        if len(out) == before or not cursor:
            break
    return out


def _all_edges(client, user_id):
    """All graph facts {edge_uuid: fact}, paginated (the API caps a page ~50)."""
    out = {}
    cursor = None
    for _ in range(60):                     # safety cap -> up to ~6000 edges
        try:
            page = client.graph.edge.get_by_user_id(user_id=user_id, limit=100,
                                                     uuid_cursor=cursor) or []
        except Exception:
            break
        if not page:
            break
        before = len(out)
        for e in page:
            if getattr(e, "fact", None) and getattr(e, "uuid_", None):
                out[e.uuid_] = e.fact
        cursor = getattr(page[-1], "uuid_", None)
        if len(out) == before or not cursor:   # no progress -> stop
            break
    return out


def _facts_with_time(edges):
    """Zep-recommended temporal handling: drop expired/superseded facts, tag each
    surviving fact with its valid-from date, newest first -- so the answer LLM
    prefers the CURRENT value on knowledge-update / conflict questions."""
    rows = []
    for e in edges:
        f = getattr(e, "fact", None)
        if not f:
            continue
        if getattr(e, "expired_at", None) or getattr(e, "invalid_at", None):
            continue  # superseded / no longer valid -> drop
        v = getattr(e, "valid_at", None)
        rows.append((v or "", (f + " (as of " + v[:10] + ")") if v else f))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows]


def _search_facts(client, user_id, query, limit):
    try:
        r = client.graph.search(query=query, user_id=user_id, scope="edges",
                                limit=limit, reranker="cross_encoder")
        return _facts_with_time(getattr(r, "edges", None) or [])
    except Exception:
        return []


def process_user(user_data, top_k, save_path, log_dir, smoke_session_limit=None, **kwargs):
    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    logger.info("=== Start user " + user_name + " (" + uuid + ") | Zep cloud (packed, progressive) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, uuid + ".json")

    from zep_cloud.types.message import Message
    client  = _client()
    tag     = re.sub(r"[^a-zA-Z0-9]", "", uuid)[:28]
    user_id = "halumem_" + tag

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {"uuid": uuid, "user_name": user_name, "sessions": []}
    seen_edges = set()

    try:
        try:
            client.user.delete(user_id=user_id); time.sleep(2)
        except Exception:
            pass
        try:
            client.user.add(user_id=user_id, first_name=user_name[:40])
        except Exception:
            pass

        for sid, session in enumerate(tqdm(sessions, desc="User " + user_name)):
            new_session = {"memory_points": session["memory_points"],
                           "dialogue":      session["dialogue"]}
            thread_id = tag + "_s" + str(sid)
            try:
                client.thread.create(thread_id=thread_id, user_id=user_id)
            except Exception:
                pass

            base_total, _ = _episode_counts(client, user_id)
            t0 = time.time()
            for chunk in _pack_dialogue(session["dialogue"]):
                try:
                    client.thread.add_messages(thread_id=thread_id,
                                               messages=[Message(role="user", content=chunk)])
                except Exception as e:
                    logger.warning("S" + str(sid) + " add error: " + str(e))
            add_ms = (time.time() - t0) * 1000

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]; del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            _wait_processed(client, user_id, base_total, logger, PROCESS_TIMEOUT)

            # One paginated dump serves both the delta and the update probes:
            # Zep marks superseded facts with invalid_at/expired_at, so P3 can
            # read INVALIDATED_KEPT straight off the store snapshot.
            snap = _all_edges_status(client, user_id)
            edges = {e["id"]: e["memory"] for e in snap}
            new_facts = [f for u, f in edges.items() if u not in seen_edges]
            seen_edges |= set(edges.keys())
            new_session["extracted_memories"]       = new_facts
            new_session["store_after"]              = snap
            new_session["add_dialogue_duration_ms"] = add_ms
            logger.info("S" + str(sid) + ": +" + str(len(new_facts)) + " facts (" + str(len(edges)) + " total)")

            for mp in new_session["memory_points"]:
                if mp["is_update"] == "False" or not mp.get("original_memories"):
                    continue
                mp["memories_from_system"] = _search_facts(client, user_id, mp["memory_content"], 10)

            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                t0 = time.time()
                facts = _search_facts(client, user_id, qa["question"], top_k)
                search_ms = (time.time() - t0) * 1000
                context = TEMPLATE_ZEP.format(user_id=user_name,
                                              memories=json.dumps(facts, indent=4, ensure_ascii=False))
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
        try:
            client.user.delete(user_id=user_id)
        except Exception:
            pass
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, uuid + "_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error("FAILED:\n" + tb)
        try:
            client.user.delete(user_id=user_id)
        except Exception:
            pass
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
    frame     = "zep"
    save_path = "./results/" + frame + "-" + version + "/"
    log_dir   = "./logs/" + frame + "-" + version + "/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, frame + "_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    if not ZEP_API_KEY:
        print("ZEP_API_KEY not set"); return output_file

    print("\n" + "="*60)
    print("  HaluMem x Zep (cloud temporal graph, PACKED + progressive)")
    print("  ANSWER LLM : " + os.getenv("OPENAI_MODEL", "gemma-4-E4B-it") + "  (shared)")
    print("  DATA       : " + str(data_path) + "  | VERSION: " + version + " | SMOKE: " + str(smoke))
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
                         smoke_session_limit=smoke_session_limit if smoke else None)
        icon = "OK" if r["status"] == "ok" else "ERR"
        print(icon + " [" + str(idx) + "/" + str(len(users)) + "] " + r["uuid"] + " -> " + r["status"])

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    with open(os.path.join(save_path, frame + "_token_usage.json"), "w") as f:
        json.dump({"total_tokens": _tk.tracker.total,
                   "note": "answer-side LLM only; Zep server-side graph-building not exposed"}, f)
    print("Answer-LLM tokens: " + str(_tk.tracker.total))
    print("\nExtraction done in " + str(round(time.time()-start, 1)) + "s -> " + output_file)
    return output_file
