"""
RAG / Full-context baseline for LoCoMo (independent from halumem_experiment).

The "no memory management" lower bound: store each raw turn (verbatim) with its
dia_id in an in-memory vector store; retrieve top-k by cosine at QA time.

Because RAG stores RAW turns, each stored item IS a single dia_id — retrieval
Recall@k / NDCG@k are exact and natural (no extraction, no provenance guessing).

CRUD framing:
  ① Extraction : NONE (raw turns, only embedded for search)
  ② CUD        : NONE (pure accumulate)
  ③ Storage    : raw turn text + embedding + dia_id
  ④ Retrieve   : cosine top-k

Embeddings use the same Ollama model as Mem0 (bge-m3) for a fair comparison.
"""

import os
import sys
import json
import time
import copy
import logging
import traceback

import numpy as np
import requests

HALUMEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "halumem_experiment")
sys.path.insert(0, os.path.abspath(HALUMEM_DIR))
from dotenv import load_dotenv
load_dotenv(os.path.join(HALUMEM_DIR, ".env"))

import token_tracker as _tk
from llms import llm_request

OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RAG_EMBED_MODEL = os.getenv("MEM0_EMBED_MODEL", "bge-m3:latest")

QA_PROMPT = """You are answering a question using ONLY the memories retrieved from a long conversation between two people.

Memories:
{context}

Question: {question}

Answer concisely and factually based only on the memories above. If the memories do not contain enough information to answer, say "No information available".
Answer:"""


def _embed(text: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": RAG_EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    v = np.asarray(r.json()["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TurnStore:
    """In-memory vector store of raw turns (text + dia_id)."""
    def __init__(self):
        self.texts, self.dia_ids, self.vecs = [], [], []

    def add(self, text, dia_id):
        self.texts.append(text); self.dia_ids.append(dia_id); self.vecs.append(_embed(text))

    def search(self, query, top_k):
        if not self.texts:
            return []
        sims = np.stack(self.vecs) @ _embed(query)
        order = np.argsort(-sims)[:top_k]
        return [{"text": self.texts[i], "dia_id": self.dia_ids[i]} for i in order]


def setup_logger(log_dir, cid):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"locomo_rag.{cid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{cid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


def _session_keys(conv):
    keys = [k for k in conv if k.startswith("session") and "date_time" not in k and "summary" not in k]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def process_conversation(sample, top_k, save_path, log_dir, session_limit=None, qa_limit=None):
    cid    = sample["sample_id"]
    conv   = sample["conversation"]
    logger = setup_logger(log_dir, cid)
    logger.info(f"=== Start conversation {cid} (RAG) ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{cid}.json")

    store = TurnStore()
    new_sample = {"sample_id": cid, "speaker_a": conv.get("speaker_a"),
                  "speaker_b": conv.get("speaker_b"), "qa": []}

    try:
        # ① Store raw turns (with dia_id)
        t0 = time.time()
        sess_keys = _session_keys(conv)
        if session_limit:
            sess_keys = sess_keys[:session_limit]
        n = 0
        for sk in sess_keys:
            for turn in conv[sk]:
                # RAG performs no extraction, so ingest is embeddings only. Zero LLM
                # calls is exactly its value as a control: every difference in ingest
                # cost is attributable to the extraction mechanism itself.
                with _tk.unit("ingest"):
                    store.add(f'{turn["speaker"]}: {turn["text"]}', turn.get("dia_id"))
                n += 1
        logger.info(f"Stored {n} raw turns in {(time.time()-t0)*1000:.0f}ms")

        # ── Full memory-store dump → extraction_locomo.py (integrity/accuracy/F1) ──
        #    RAG performs NO extraction — each "memory" is a verbatim turn. Its
        #    extraction scores are therefore the ceiling/sanity check for the metric,
        #    not a comparable memory system: recall should be near 1 (nothing is
        #    compressed away) and precision near 1 (raw text is always supported).
        new_sample["memory_dump"] = [
            {"text": t, "dia_id": d} for t, d in zip(store.texts, store.dia_ids)
        ]
        logger.info(f"Dumped {len(new_sample['memory_dump'])} memories (raw turns)")

        # ② QA
        qa_list = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
        for qa in qa_list:
            with _tk.unit("qa"):              # one question is one qa unit
                t0 = time.time()
                retrieved = store.search(qa["question"], top_k)
                mems = [r["text"] for r in retrieved]
                search_ms = (time.time() - t0) * 1000

                prompt = QA_PROMPT.format(context=json.dumps(mems, indent=2), question=qa["question"])
                t0 = time.time()
                response = llm_request(prompt)

            new_qa = copy.deepcopy(qa)
            new_qa["retrieved"]          = retrieved
            new_qa["retrieved_memories"] = mems
            new_qa["system_response"]    = response
            new_qa["search_ms"]          = round(search_ms, 1)
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
    frame     = "rag"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    output_file = os.path.join(save_path, "rag_locomo_results.jsonl")
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
    print(f"  LoCoMo × RAG baseline (no extraction, raw turns)")
    print(f"  EMBED  : {RAG_EMBED_MODEL}")
    print(f"  QA LLM : {os.getenv('OPENAI_MODEL')}")
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
                                 qa_limit=5 if smoke else None)
        icon = "✅" if r["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{len(data)}] {r['cid']} → {r['status']}")

    with open(output_file, "w", encoding="utf-8") as fo:
        for fn in sorted(os.listdir(tmp_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(tmp_dir, fn), encoding="utf-8") as fi:
                    fo.write(json.dumps(json.load(fi), ensure_ascii=False) + "\n")

    _tk.save(save_path, "rag_locomo")
    meta = {
        "extraction_llm": "none (raw turns)",
        "judge_llm":      os.getenv("OPENAI_MODEL", "unknown"),
        "embed_model":    RAG_EMBED_MODEL,
        "granularity":    "turn",   # each stored chunk = one turn (dia_id)
    }
    with open(os.path.join(save_path, "rag_locomo_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
