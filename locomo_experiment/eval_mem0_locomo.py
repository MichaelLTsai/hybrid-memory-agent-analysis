"""
Mem0 adapter for the LoCoMo dataset (independent from halumem_experiment).

References (imports) shared utilities from ../halumem_experiment WITHOUT modifying
them: build_mem0_config, mem_search, _format_search_results, llm_request, token_tracker.

LoCoMo structure (per conversation sample):
  conversation: speaker_a / speaker_b + session_N (dated) with turns {speaker, dia_id, text}
  qa: [{question, answer|adversarial_answer, evidence:[dia_id], category}]

Pipeline (per conversation):
  ① feed each session's dialogue to Mem0 (LLM extraction into vector store)
  ② for each QA: mem.search(question) → top-k → build context → LLM answer
Output: results/mem0-{version}/mem0_locomo_results.jsonl
        (one line per conversation, with qa[].system_response filled)
"""

import os
import sys
import json
import time
import copy
import logging
import traceback

# ── Reference halumem_experiment's shared code (import only, never modify) ──────
HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))   # reuse NCHC / Mem0 config

from tqdm import tqdm
import token_tracker as _tk                       # applies OpenAI patch on import
from llms import llm_request
from eval_mem0_oss import (
    build_mem0_config, mem_search, _format_search_results, MEM0_MAJOR,
)
from mem0 import Memory

QDRANT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a long conversation between two people.

Memories:
{context}

Question: {question}

Answer concisely and factually based only on the memories above. If the memories do not contain enough information to answer, say "No information available".
Answer:"""


def _mem_get_all(mem, user_name: str):
    """Version-aware full-store dump. Param names differ between Mem0 majors:
        v1.x : get_all(user_id=..., limit=N)
        v2.x : get_all(filters={'user_id':...}, limit=N)   # top-level user_id rejected
    NOTE: the default limit is 100 in both — must be raised or the dump truncates."""
    if MEM0_MAJOR < 2:
        return mem.get_all(user_id=user_name, limit=100000)
    # v2 caps at top_k (default 20) — `limit` silently lands in **kwargs and is ignored,
    # which truncates the dump to 20 and makes P1 look catastrophically low.
    return mem.get_all(filters={"user_id": user_name}, top_k=100000)


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_mem0.{cid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{cid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _session_keys(conv: dict):
    """Ordered session keys: session_1, session_2, ... (skip *_date_time)."""
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k
            and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def process_conversation(sample, top_k, save_path, log_dir, llm_model=None,
                         session_limit=None, qa_limit=None):
    cid    = sample["sample_id"]
    conv   = sample["conversation"]
    logger = setup_logger(log_dir, cid)
    logger.info(f"=== Start conversation {cid} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    user_id = f"locomo_{cid.replace('-', '_')}"
    collection = f"locomo_{cid.replace('-', '_')[:40]}"
    # Per-conversation AND per-run Qdrant path. The run tag comes from save_path
    # (results/mem0-{version}/), which keeps Mem0 v1 and v2 — different store
    # schemas — from sharing a directory: without this the second version to run
    # inherits the first one's vectors, and the two cannot run concurrently.
    run_tag = os.path.basename(os.path.normpath(save_path)) or "default"
    qdrant_path = os.path.join(QDRANT_BASE, run_tag, cid.replace('-', '_'))
    mem = Memory.from_config(build_mem0_config(collection, qdrant_path, llm_model=llm_model))
    try:
        mem.delete_all(user_id=user_id)
    except Exception:
        pass

    speaker_a = conv.get("speaker_a", "SpeakerA")
    speaker_b = conv.get("speaker_b", "SpeakerB")

    new_sample = {"sample_id": cid, "speaker_a": speaker_a, "speaker_b": speaker_b, "qa": []}

    try:
        # ── ① Feed TURN-BY-TURN so each memory carries its source dia_id ─
        #    (enables exact retrieval Recall@k / NDCG@k, no embedding approx)
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        n_turns = 0
        for sk in sess_keys:
            for turn in conv[sk]:
                content = f'{turn["speaker"]}: {turn["text"]}'
                # mem0 writes at turn level on LoCoMo, so one turn is one ingest unit.
                with _tk.unit("ingest"):
                    try:
                        mem.add([{"role": "user", "content": content}], user_id=user_id,
                                metadata={"dia_id": turn.get("dia_id")})
                        n_turns += 1
                    except Exception as e:
                        logger.warning(f"add {turn.get('dia_id')} error: {e}")
        add_ms = (time.time() - t0) * 1000
        logger.info(f"Fed {n_turns} turns ({len(sess_keys)} sessions) in {add_ms:.0f}ms")

        # ── Full memory-store dump → extraction_locomo.py (integrity/accuracy/F1) ──
        #    Taken right after ingest so it reflects what the WRITE path produced.
        #    NOTE: get_all() defaults to limit=100 — must raise it or the dump truncates.
        try:
            snap = _mem_get_all(mem, user_id)
            items = snap.get("results", snap) if isinstance(snap, dict) else snap
            new_sample["memory_dump"] = [
                {"text": it.get("memory", ""),
                 "dia_id": (it.get("metadata") or {}).get("dia_id")}
                for it in items if it.get("memory")
            ]
            logger.info(f"Dumped {len(new_sample['memory_dump'])} memories")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            new_sample["memory_dump"] = None

        # ── ② QA ─────────────────────────────────────────────────────────
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            with _tk.unit("qa"):              # one question is one qa unit
                t0 = time.time()
                res = mem_search(mem, qa["question"], user_id, top_k)
                # keep ranked (text, dia_id) for exact retrieval metrics
                retrieved = [{"text": it.get("memory", ""),
                              "dia_id": (it.get("metadata") or {}).get("dia_id")}
                             for it in res.get("results", [])]
                mems = [r["text"] for r in retrieved]
                search_ms = (time.time() - t0) * 1000

                context = json.dumps(mems, indent=2)
                prompt  = QA_PROMPT.format(context=context, question=qa["question"])
                t0 = time.time()
                response = llm_request(prompt)
                response_ms = (time.time() - t0) * 1000

            new_qa = copy.deepcopy(qa)
            new_qa["retrieved"]          = retrieved   # ranked {text, dia_id}
            new_qa["retrieved_memories"] = mems         # text only (for context)
            new_qa["system_response"]    = response
            new_qa["search_ms"]          = round(search_ms, 1)
            new_qa["response_ms"]        = round(response_ms, 1)
            new_sample["qa"].append(new_qa)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_sample, f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation {cid} complete → {tmp_file}")
        return {"cid": cid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{cid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"cid": cid, "status": "error", "path": err}
    finally:
        # Release the Qdrant local lock so the next conversation / run is clean
        try:
            mem.vector_store.client.close()
        except Exception:
            pass


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_convs=None, llm_model=None) -> str:
    frame     = "mem0"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    output_file = os.path.join(save_path, "mem0_locomo_results.jsonl")
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
    print(f"  LoCoMo × Mem0")
    print(f"  MEM0 major : {MEM0_MAJOR}")
    print(f"  LLM        : {llm_model or os.getenv('MEM0_LLM_MODEL')}")
    print(f"  conversations: {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    start = time.time()
    for idx, sample in enumerate(data, 1):
        cid = sample["sample_id"]
        if cid in done:
            print(f"⏭️  [{idx}/{len(data)}] {cid} already done"); continue
        r = process_conversation(
            sample, top_k, save_path, log_dir, llm_model=llm_model,
            session_limit=3 if smoke else None,
            qa_limit=5 if smoke else None,
        )
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['cid']} → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(save_path, "mem0_locomo")

    # Record run config (read by update_excel_locomo)
    meta = {
        "extraction_llm": llm_model or os.getenv("MEM0_LLM_MODEL", "unknown"),
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    os.getenv("MEM0_EMBED_MODEL", "unknown"),
        "granularity":    "turn",   # turn-by-turn feed → each memory carries source dia_id
    }
    with open(os.path.join(save_path, "mem0_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
