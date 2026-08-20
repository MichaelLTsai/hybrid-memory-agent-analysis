"""
Graphiti (temporal knowledge graph) adapter for LoCoMo.

Reuses halumem_experiment's Graphiti construction (build_graphiti) WITHOUT
modifying it. Graphiti extracts entity-relation triples with temporal validity —
the key backend for LoCoMo's TEMPORAL questions.

dia_id provenance: each turn is fed as an episode named with its dia_id; we map
episode.uuid → dia_id. Retrieved edges carry `.episodes` (source episode uuids),
so we resolve each retrieved fact back to its dia_id → exact Recall@k / NDCG@k.

Config (from halumem .env): GRAPHITI_LLM_MODEL (Llama-Scout, JSON-capable) +
GRAPHITI_EMBED_MODEL (mxbai via Ollama). Note: this differs from Mem0/RAG
(E4B + bge-m3) because Graphiti needs a schema-following LLM — recorded in meta.
"""

import os
import re
import sys
import json
import time
import copy
import asyncio
import logging
import traceback
from datetime import datetime, timezone

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

from eval_graphiti import (  # reuse, no modify
    build_graphiti, create_kuzu_fts_indexes, GRAPHITI_LLM_MODEL, GRAPHITI_EMBED_MODEL,
)
from graphiti_core.nodes import EpisodeType
from llms import llm_request

KUZU_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kuzu_data")

QA_PROMPT = """You are answering a question using ONLY the facts retrieved from a temporal knowledge graph of a long conversation between two people.

Facts (with event times):
{context}

Question: {question}

Answer concisely and factually based only on the facts above. If they do not contain enough information, say "No information available".
Answer:"""


def _parse_date(s: str) -> datetime:
    """LoCoMo date like '1:56 pm on 8 May, 2023' → datetime (UTC)."""
    if not s:
        return datetime.now(timezone.utc)
    try:
        from dateutil import parser as dp
        return dp.parse(s.replace(" on ", " ")).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _session_keys(conv):
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_graphiti.{cid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{cid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


async def _search_dia(graphiti, group_id, query, top_k, ep2dia):
    """Search KG; return ranked [{text, dia_id}] resolving edge → source episode dia_id."""
    try:
        edges = await graphiti.search(query=query, group_ids=[group_id], num_results=top_k)
    except Exception:
        return []
    out = []
    for e in edges:
        fact = getattr(e, "fact", None) or getattr(e, "name", "")
        va   = getattr(e, "valid_at", "")
        # session-level feeding → no turn-level dia_id provenance
        out.append({"text": f"{va}: {fact}" if va else fact, "dia_id": None})
    return out


async def _process_async(sample, top_k, save_path, log_dir, session_limit, qa_limit):
    cid    = sample["sample_id"]
    conv   = sample["conversation"]
    logger = setup_logger(log_dir, cid)
    logger.info(f"=== Start conversation {cid} (Graphiti) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    group_id  = f"locomo_{cid.replace('-', '_')}"
    os.makedirs(KUZU_BASE, exist_ok=True)
    kuzu_path = os.path.join(KUZU_BASE, cid.replace('-', '_') + ".kz")   # Kuzu creates this (not a dir)
    graphiti = build_graphiti(kuzu_path, group_id)
    create_kuzu_fts_indexes(graphiti, logger)   # Kuzu FTS indexes (build_indices is a no-op)

    ep2dia = {}
    new_sample = {"sample_id": cid, "speaker_a": conv.get("speaker_a"),
                  "speaker_b": conv.get("speaker_b"), "qa": []}

    try:
        # ① Feed each SESSION as one episode (better extraction than turn-by-turn;
        #    Graphiti needs enough context per episode to extract facts). Trade-off:
        #    no turn-level dia_id provenance → exact retrieval Recall@k not available.
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        n = 0
        for sk in sess_keys:
            dt = _parse_date(conv.get(f"{sk}_date_time", ""))
            body = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in conv[sk])
            try:
                await graphiti.add_episode(
                    name=f"{cid}_{sk}",
                    episode_body=body,
                    source=EpisodeType.text,
                    source_description=f"LoCoMo {cid} {sk}",
                    reference_time=dt,
                    group_id=group_id,
                )
                n += 1
            except Exception as e:
                logger.warning(f"add_episode {sk} error: {e}")
        logger.info(f"Fed {n} sessions as episodes in {(time.time()-t0):.0f}s")

        # ── Full memory-store dump → extraction_locomo.py (integrity/accuracy/F1) ──
        #    One episode per session, so an edge's source episode recovers its session.
        #    That is enough for session-scoped extraction scoring, so the dump emits a
        #    synthetic "D{session}:0" — extraction_locomo only parses the session number
        #    out of a dia_id. Turn-level provenance genuinely does not exist here.
        try:
            episodes = await graphiti.retrieve_episodes(
                reference_time=datetime.now(timezone.utc),
                last_n=10000, group_ids=[group_id])
            uuid2sess = {}
            for ep in episodes:
                m = re.search(r"session_(\d+)$", getattr(ep, "name", "") or "")
                if m:
                    uuid2sess[ep.uuid] = int(m.group(1))
            res = await graphiti.get_nodes_and_edges_by_episode([ep.uuid for ep in episodes])
            dump, seen = [], set()
            for e in res.edges:
                fact = getattr(e, "fact", None) or getattr(e, "name", "")
                if not fact or fact in seen:
                    continue
                seen.add(fact)
                sess = next((uuid2sess[u] for u in (getattr(e, "episodes", None) or [])
                             if u in uuid2sess), None)
                dump.append({"text": fact,
                             "dia_id": f"D{sess}:0" if sess is not None else None})
            new_sample["memory_dump"] = dump
            logger.info(f"Dumped {len(dump)} facts "
                        f"({sum(1 for d in dump if d['dia_id'])} with session provenance)")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            new_sample["memory_dump"] = None

        # ② QA
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            retrieved = await _search_dia(graphiti, group_id, qa["question"], top_k, ep2dia)
            mems = [r["text"] for r in retrieved]
            prompt = QA_PROMPT.format(context="\n".join(f"  - {m}" for m in mems) or "  (none)",
                                      question=qa["question"])
            t0 = time.time()
            response = llm_request(prompt)

            new_qa = copy.deepcopy(qa)
            new_qa["retrieved"]          = retrieved
            new_qa["retrieved_memories"] = mems
            new_qa["system_response"]    = response
            new_qa["response_ms"]        = round((time.time() - t0) * 1000, 1)
            new_sample["qa"].append(new_qa)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_sample, f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation {cid} complete → {tmp_file}")
        status = {"cid": cid, "status": "ok", "path": tmp_file}
    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{cid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        status = {"cid": cid, "status": "error", "path": err}
    finally:
        try:
            await graphiti.close()
        except Exception:
            pass
    return status


def process_conversation(sample, top_k, save_path, log_dir, session_limit=None, qa_limit=None):
    return asyncio.run(_process_async(sample, top_k, save_path, log_dir, session_limit, qa_limit))


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_convs=None, llm_model=None) -> str:
    frame     = "graphiti"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "graphiti_locomo_results.jsonl")
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
    print(f"  LoCoMo × Graphiti (temporal knowledge graph)")
    print(f"  LLM   : {GRAPHITI_LLM_MODEL}")
    print(f"  EMBED : {GRAPHITI_EMBED_MODEL}")
    print(f"  conversations: {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    start = time.time()
    for idx, sample in enumerate(data, 1):
        cid = sample["sample_id"]
        if cid in done:
            print(f"⏭️  [{idx}/{len(data)}] {cid} already done"); continue
        r = process_conversation(sample, top_k, save_path, log_dir,
                                 session_limit=3 if smoke else None,
                                 qa_limit=5 if smoke else None)
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['cid']} → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    meta = {
        "extraction_llm": GRAPHITI_LLM_MODEL,
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    GRAPHITI_EMBED_MODEL,
        "granularity":    "session",   # session-level episodes (no turn dia_id provenance)
    }
    with open(os.path.join(save_path, "graphiti_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
