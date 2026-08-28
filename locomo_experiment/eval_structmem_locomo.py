"""
StructMem adapter for the LoCoMo dataset (independent from halumem_experiment).

StructMem (Zhejiang Univ. + Ant Group) is not a standalone package — it is the
LightMemory engine run in `extraction_mode="event"` with cross-event consolidation.
This file reuses halumem_experiment/eval_structmem.py's config builder and store
dump WITHOUT modifying it.

LoCoMo structure (per conversation sample):
  conversation: speaker_a / speaker_b + session_N (dated) with turns {speaker, dia_id, text}
  qa: [{question, answer|adversarial_answer, evidence:[dia_id], category}]

Pipeline (per conversation):
  ① feed each SESSION as one add_memory call (StructMem extracts events from it),
     consolidating every SUMMARIZE_EVERY sessions
  ② dump the whole store → extraction_locomo.py can score P1 (integrity/accuracy/F1)
  ③ for each QA: lm.retrieve(question) → top-k → build context → LLM answer

Provenance: StructMem extracts events across a whole session, so a memory maps to a
SESSION, not a turn. The dump therefore emits a synthetic "D{session}:0" dia_id —
extraction_locomo only parses the session number out of it. Turn-level retrieval
Recall@k is genuinely not available, so retrieved dia_id is left None (same as the
session-level Graphiti adapter).

Output: results/structmem-{version}/structmem_locomo_results.jsonl
"""

import os
import re
import sys
import json
import time
import copy
import shutil
import logging
import traceback

# ── Reference halumem_experiment's shared code (import only, never modify) ──────
HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))

from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))   # reuse NCHC config

from tqdm import tqdm
import token_tracker as _tk                       # applies OpenAI patch on import
from llms import llm_request
from eval_structmem import _build_config, _store_dump, SM_LLM, SUMMARIZE_EVERY

QDRANT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_structmem")

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a long conversation between two people.

Memories:
{context}

Question: {question}

Answer concisely and factually based only on the memories above. If the memories do not contain enough information to answer, say "No information available".
Answer:"""


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_structmem.{cid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{cid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _session_keys(conv: dict):
    """Ordered session keys: session_1, session_2, ... (skip *_date_time / *_summary)."""
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k
            and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _sm_time(s: str) -> str:
    """LoCoMo '1:56 pm on 8 May, 2023' → LightMemory's '2023/05/08 (Mon) 13:56'.
    LightMemory parses time_stamp strictly and silently drops the whole session on a
    format mismatch, so the conversion has to happen before add_memory."""
    from datetime import datetime
    if s:
        try:
            from dateutil import parser as dp
            return dp.parse(s.replace(" on ", " ")).strftime("%Y/%m/%d (%a) %H:%M")
        except Exception:
            pass
    return datetime.now().strftime("%Y/%m/%d (%a) %H:%M")


def _to_messages(turns, ts):
    """LoCoMo turns → LightMemory message dicts. Speaker name is kept inline so the
    extractor can attribute facts to the right person (LoCoMo is a 2-party dialogue)."""
    stamp = _sm_time(ts)
    return [{"role": "user", "content": f'{t.get("speaker")}: {t.get("text","")}',
             "time_stamp": stamp}
            for t in turns if t.get("text")]


# M4 QA conditioning. Only reaches the prompt when enable_m4_state_qa is on.
_STATE_QA_RULES = {
    "current": ("- CURRENT MEMORY controls the present-state answer.\n"
                "- HISTORICAL MEMORY is context only, never the current value."),
    "historical": ("- Answer with the past state the question asks about.\n"
                   "- Do not let a newer CURRENT MEMORY override the historical target."),
    "transition": ("- Describe the change: the before value and the after value.\n"
                   "- Use TRANSITION evidence for when and how it changed."),
    "neutral": ("- Answer by ordinary relevance.\n"
                "- Do not force a temporal narrative."),
}


def _state_ablation(trace_dir=None):
    """Resolve the M1/M3/M4 arm from STRUCTMEM_EXPERIMENT (default E0)."""
    from lightmem.memory.state import config as _state_config

    cfg = _state_config.from_env()
    if trace_dir:
        cfg.trace_dir = trace_dir
    return cfg


def process_conversation(sample, top_k, save_path, log_dir, llm_model=None,
                         session_limit=None, qa_limit=None):
    from lightmem.memory.lightmem import LightMemory

    cid    = sample["sample_id"]
    conv   = sample["conversation"]
    logger = setup_logger(log_dir, cid)
    logger.info(f"=== Start conversation {cid} (StructMem, event mode) | LLM={llm_model or SM_LLM} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    # Each ablation arm owns its own collections so no two arms share a store.
    state_cfg = _state_ablation(trace_dir=os.path.join(save_path, "traces"))
    tag = re.sub(r"[^a-zA-Z0-9]", "", cid)[:28] + state_cfg.collection_suffix()
    tag = re.sub(r"[^a-zA-Z0-9_]", "", tag)
    logger.info(f"Ablation arm: {state_cfg.to_manifest()}")
    shutil.rmtree(os.path.join(QDRANT_BASE, tag), ignore_errors=True)          # isolation
    shutil.rmtree(os.path.join(QDRANT_BASE, tag + "_sum"), ignore_errors=True)

    new_sample = {"sample_id": cid, "speaker_a": conv.get("speaker_a"),
                  "speaker_b": conv.get("speaker_b"), "qa": []}
    lm = None

    try:
        cfg = _build_config(tag, llm_model)
        cfg["state_ablation"] = state_cfg
        # Point the retrievers at this experiment's own qdrant dir, not halumem's.
        cfg["embedding_retriever"]["configs"]["path"] = os.path.join(QDRANT_BASE, tag)
        cfg["summary_retriever"]["configs"]["path"]   = os.path.join(QDRANT_BASE, tag + "_sum")
        lm = LightMemory.from_config(cfg)

        # ── ① Feed each session as one event-extraction call ─────────────
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        # id → session number, so the store dump can carry session provenance
        seen, id2sess = set(), {}
        n = 0
        for sk in tqdm(sess_keys, desc=f"{cid}"):
            snum = int(sk.split("_")[1])
            ts   = conv.get(f"{sk}_date_time", "")
            # One session is one ingest unit.
            with _tk.unit("ingest"):
                try:
                    lm.add_memory(_to_messages(conv[sk], ts), force_extract=True)
                    n += 1
                except Exception as e:
                    logger.warning(f"{sk} add error: {e}")

            # Consolidation runs only once every SUMMARIZE_EVERY sessions but is
            # part of the architecture's write cost, so it goes into the ingest
            # bucket without counting a unit (amortized into calls_per_unit).
            if n and n % SUMMARIZE_EVERY == 0:
                with _tk.phase("ingest"):
                    # Conflict resolution. summarize() only consolidates across
                    # events and never replaces an old value with a new one.
                    # LightMem's update mechanism is these two offline batch
                    # methods. Set STRUCTMEM_OFFLINE_UPDATE=0 to disable it
                    # again as a control.
                    #
                    # Ordering is the M3 switch: with summary sync on, the state
                    # commit must land BEFORE the summaries are written so the
                    # summariser sees active/superseded labels. With it off the
                    # original order is kept and summaries may go stale, which is
                    # exactly the E1/E3 condition.
                    def _run_update():
                        if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") != "1":
                            return
                        try:
                            lm.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                            lm.offline_update_all_entries(score_threshold=0.9)
                        except Exception as e:
                            logger.warning(f"offline_update error: {e}")

                    def _run_summarize():
                        try:
                            lm.summarize(process_all=True, enable_cross_event=True,
                                         retrieval_scope="global", top_k_seeds=15)
                        except Exception as e:
                            logger.warning(f"{sk} summarize error: {e}")

                    if lm.state_ablation.enable_m3_summary_sync:
                        _run_update()
                        _run_summarize()
                    else:
                        _run_summarize()
                        _run_update()

            # Entries new since the previous session belong to this session.
            for entry in _store_dump(lm):
                if entry["id"] not in seen:
                    seen.add(entry["id"])
                    id2sess[entry["id"]] = snum
        # Final flush: the loop only fires every SUMMARIZE_EVERY sessions, so
        # without this the tail would never be consolidated or state-audited.
        with _tk.phase("ingest"):
            def _final_update():
                if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") != "1":
                    return
                try:
                    lm.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                    lm.offline_update_all_entries(score_threshold=0.9)
                except Exception as e:
                    logger.warning(f"final offline_update error: {e}")

            def _final_summarize():
                try:
                    lm.summarize(process_all=True, enable_cross_event=True,
                                 retrieval_scope="global", top_k_seeds=15)
                except Exception as e:
                    logger.warning(f"final summarize error: {e}")

            if lm.state_ablation.enable_m3_summary_sync:
                _final_update(); _final_summarize()
            else:
                _final_summarize(); _final_update()

        logger.info(f"Fed {n} sessions in {(time.time()-t0):.0f}s")

        # ── ② Full memory-store dump → extraction_locomo.py (P1) ─────────
        #    Session-level provenance only; emitted as synthetic "D{session}:0".
        try:
            dump = _store_dump(lm)
            new_sample["memory_dump"] = [
                {"text": e["memory"],
                 "dia_id": (f'D{id2sess[e["id"]]}:0' if e["id"] in id2sess else None)}
                for e in dump
            ]
            logger.info(f"Dumped {len(new_sample['memory_dump'])} memories "
                        f"({sum(1 for d in new_sample['memory_dump'] if d['dia_id'])} with session provenance)")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            new_sample["memory_dump"] = None

        # ── ③ QA ─────────────────────────────────────────────────────────
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            with _tk.unit("qa"):              # one question is one qa unit
                t0 = time.time()
                packet = None
                try:
                    # Dual-circuit retrieval (entries + cross-event summaries);
                    # state-aware ordering and labels only when M4 is on.
                    mems, packet = lm.retrieve_for_qa(qa["question"], limit=top_k)
                except Exception as e:
                    logger.warning(f"retrieve error: {e}")
                    mems = []
                mems = [m if isinstance(m, str) else str(m) for m in mems]
                search_ms = (time.time() - t0) * 1000

                context = "\n".join(f"  - {m}" for m in mems) or "  (none)"
                if packet is not None:
                    context = (f"  QUERY STATE VIEW: {packet.query_view.upper()}\n"
                               + context + "\n  State rules:\n  "
                               + _STATE_QA_RULES.get(packet.query_view, _STATE_QA_RULES["neutral"]))
                prompt = QA_PROMPT.format(context=context, question=qa["question"])
                t0 = time.time()
                response = llm_request(prompt)

            new_qa = copy.deepcopy(qa)
            # session-level extraction → no turn-level dia_id provenance
            new_qa["retrieved"]          = [{"text": m, "dia_id": None} for m in mems]
            new_qa["retrieved_memories"] = mems
            new_qa["system_response"]    = response
            new_qa["search_ms"]          = round(search_ms, 1)
            new_qa["response_ms"]        = round((time.time() - t0) * 1000, 1)
            new_sample["qa"].append(new_qa)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_sample, f, ensure_ascii=False, indent=2)
        logger.info(f"Conversation {cid} complete → {tmp_file}")
        return {"cid": cid, "status": "ok"}

    except Exception as e:
        logger.error(f"{cid} failed: {e}\n{traceback.format_exc()}")
        return {"cid": cid, "status": f"error: {e}"}
    finally:
        try:
            lm.embedding_retriever.client.close()
        except Exception:
            pass


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   max_convs=None, llm_model=None) -> str:
    frame     = "structmem"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    output_file = os.path.join(save_path, "structmem_locomo_results.jsonl")
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
    print(f"  LoCoMo × StructMem (event mode)")
    print(f"  LLM        : {llm_model or SM_LLM}")
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

    _tk.save(save_path, "structmem_locomo")

    meta = {
        "extraction_llm": llm_model or SM_LLM,
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    "all-MiniLM-L6-v2",
        "granularity":    "session",   # event extraction spans a session → no turn dia_id
    }
    with open(os.path.join(save_path, "structmem_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
