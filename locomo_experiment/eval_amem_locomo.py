"""
A-MEM (Agentic Memory / Zettelkasten) adapter for LoCoMo.

Reuses halumem_experiment's A-MEM instance construction (_make_amem_instance,
which applies the NCHC patches on import) WITHOUT modifying it.

A-MEM is turn-level by design (one note per turn), so each note maps to one
dia_id → exact retrieval Recall@k / NDCG@k via a note_id → dia_id map.

CRUD framing:
  ① Extraction : LLM per turn → Zettelkasten note (keywords/tags/context)
  ② CUD        : link + evolve neighbours (no delete)
  ③ Storage    : notes in ChromaDB (all-MiniLM embeddings)
  ④ Retrieve   : semantic search over notes
"""

import os
import sys
import json
import time
import copy
import logging
import traceback

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk                       # patches OpenAI on import
from eval_amem import _make_amem_instance, AMEM_LLM_MODEL, AMEM_EMBED_MODEL, AMEM_BACKEND  # reuse
from llms import llm_request

_tk.patch_litellm()   # A-MEM may route LLM calls via litellm — capture those too

QA_PROMPT = """You are answering a question using ONLY the memory notes retrieved from a long conversation between two people.

Notes:
{context}

Question: {question}

Answer concisely and factually based only on the notes above. If they do not contain enough information, say "No information available".
Answer:"""


def _parse_amem_time(s: str) -> str:
    """LoCoMo date '1:56 pm on 8 May, 2023' → A-MEM time 'YYYYMMDDHHMM'."""
    try:
        from dateutil import parser as dp
        return dp.parse(s.replace(" on ", " ")).strftime("%Y%m%d%H%M")
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d%H%M")


def _session_keys(conv):
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_amem.{cid}")
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
    logger.info(f"=== Start conversation {cid} (A-MEM) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    memory = _make_amem_instance(llm_model=llm_model)   # fresh isolated ChromaDB
    note2dia = {}
    new_sample = {"sample_id": cid, "speaker_a": conv.get("speaker_a"),
                  "speaker_b": conv.get("speaker_b"), "qa": []}

    try:
        # ① Feed each turn as a note (turn-level → dia_id provenance)
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        n = 0
        for sk in sess_keys:
            amem_time = _parse_amem_time(conv.get(f"{sk}_date_time", ""))
            for turn in conv[sk]:
                content = f'{turn["speaker"]}: {turn["text"]}'
                with _tk.unit("ingest"):      # A-MEM is turn level: one turn is one unit
                    try:
                        note_id = memory.add_note(content=content, time=amem_time)
                        if note_id:
                            note2dia[note_id] = turn.get("dia_id")
                        n += 1
                    except Exception as e:
                        logger.warning(f"add_note {turn.get('dia_id')} error: {e}")
        logger.info(f"Fed {n} turns as notes in {(time.time()-t0):.0f}s")

        # ── Full memory-store dump → extraction_locomo.py (integrity/accuracy/F1) ──
        #    A-MEM keeps every note in .memories (id → MemoryNote); note2dia carries
        #    the source turn, so the dump keeps turn-level provenance.
        try:
            new_sample["memory_dump"] = [
                {"text": getattr(note, "content", "") or "", "dia_id": note2dia.get(mid)}
                for mid, note in (getattr(memory, "memories", None) or {}).items()
                if getattr(note, "content", "")
            ]
            logger.info(f"Dumped {len(new_sample['memory_dump'])} memories")
        except Exception as e:
            logger.warning(f"memory dump failed: {e}")
            new_sample["memory_dump"] = None

        # ② QA
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            with _tk.unit("qa"):            # search_agentic calls the LLM too
                results = memory.search_agentic(qa["question"], k=top_k)
                retrieved = [{"text": r.get("content", ""),
                              "dia_id": note2dia.get(r.get("id"))} for r in results]
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
    frame     = "amem"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "amem_locomo_results.jsonl")
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
    print(f"  LoCoMo × A-MEM (Zettelkasten, turn-level)")
    print(f"  LLM   : {llm_model or AMEM_LLM_MODEL}  ({AMEM_BACKEND})")
    print(f"  EMBED : {AMEM_EMBED_MODEL}")
    print(f"  conversations: {len(data)}  | VERSION: {version} | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
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

    meta = {
        "extraction_llm": llm_model or AMEM_LLM_MODEL,
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    AMEM_EMBED_MODEL,
        "granularity":    "turn",
    }
    with open(os.path.join(save_path, "amem_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _tk.save(save_path, "amem_locomo")
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
