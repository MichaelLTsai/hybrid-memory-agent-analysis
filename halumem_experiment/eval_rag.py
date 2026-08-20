"""
RAG / Full-context baseline adapter for HaluMem — the "no memory management" lower bound.

Architecture (vs the extraction-based backends):
  Mem0/A-MEM/etc : dialogue → LLM extraction → managed memory store → retrieve
  RAG (this file): dialogue → store RAW user turns (no extraction, no CUD) → ANN top-k

CRUD framing:
  ① Extraction : NONE — user turns stored verbatim
  ② CUD        : NONE — pure accumulate (Create only, never compare/update/delete)
  ③ Storage    : raw turn text + embedding, in-memory per-user vector store
  ④ Retrieve   : cosine similarity top-k

Embeddings use the same Ollama model as Mem0 (bge-m3) for a fair retrieval comparison.
The QA answer LLM is OPENAI_MODEL from .env (no extraction LLM is involved at all).

.env keys used:
  MEM0_EMBED_MODEL = embedding model on Ollama (default: bge-m3:latest)
  OLLAMA_BASE_URL  = Ollama base URL
  OPENAI_MODEL     = QA answer + judge LLM
"""

import os
import re
import json
import time
import copy
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import token_tracker as _tk          # applies OpenAI patch immediately
from prompts import PROMPT_MEMZERO
from llms import llm_request

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RAG_EMBED_MODEL  = os.getenv("MEM0_EMBED_MODEL", "bge-m3:latest")
DATE_FORMAT      = "%b %d, %Y, %H:%M:%S"

TEMPLATE_RAG = """Memories for user {user_id}:

    {memories}
"""


# ── Embedding (Ollama) ─────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """Single embedding via Ollama; L2-normalized for cosine via dot product."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": RAG_EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    v = np.asarray(r.json()["embedding"], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TurnStore:
    """In-memory per-user vector store of raw turns (accumulate-only)."""
    def __init__(self):
        self.texts: list[str] = []
        self.vecs:  list[np.ndarray] = []

    def add(self, text: str):
        self.texts.append(text)
        self.vecs.append(_embed(text))

    def search(self, query: str, top_k: int) -> list[str]:
        if not self.texts:
            return []
        qv = _embed(query)
        sims = np.stack(self.vecs) @ qv
        order = np.argsort(-sims)[:top_k]
        return [self.texts[i] for i in order]


# ── Helpers ─────────────────────────────────────────────────────────────────

def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        return match.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name from persona_info: {persona_info[:100]}")


def setup_logger(log_dir: str, uuid: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem_rag.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


# ── Core processing ───────────────────────────────────────────────────────────

def process_user(user_data, top_k, save_path, log_dir, smoke_session_limit=None, **kwargs):
    uuid      = user_data["uuid"]
    user_name = extract_user_name(user_data["persona_info"])
    logger    = setup_logger(log_dir, uuid)
    logger.info(f"=== Start user {user_name} ({uuid}) | RAG | EMBED={RAG_EMBED_MODEL} ===")

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")

    store = TurnStore()                      # fresh per-user, accumulate-only
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

            # ── Store raw USER turns (no extraction, no CUD) ─────────────
            t0 = time.time()
            session_turns = []
            # RAG performs no extraction, so ingest is embeddings only and costs
            # zero LLM calls (the control baseline). Units are counted per session
            # to align with the other HaluMem backends.
            with _tk.unit("ingest"):
                for turn in session["dialogue"]:
                    if turn["role"] != "user":
                        continue
                    store.add(turn["content"])
                    session_turns.append(turn["content"])
            add_ms = (time.time() - t0) * 1000

            if session.get("is_generated_qa_session", False):
                new_session["add_dialogue_duration_ms"] = add_ms
                new_session["is_generated_qa_session"]  = True
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            new_session["extracted_memories"]       = session_turns
            new_session["add_dialogue_duration_ms"] = add_ms
            logger.info(f"Session {sid} | stored {len(session_turns)} raw user turns in {add_ms:.0f}ms")

            # ── Update memory search ─────────────────────────────────────
            for mp in new_session["memory_points"]:
                if mp["is_update"] == "False" or not mp.get("original_memories"):
                    continue
                mp["memories_from_system"] = store.search(mp["memory_content"], top_k=10)

            # ── QA ───────────────────────────────────────────────────────
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                with _tk.unit("qa"):          # one question is one qa unit
                    t0 = time.time()
                    mems = store.search(qa["question"], top_k=top_k)
                    search_ms = (time.time() - t0) * 1000

                    context = TEMPLATE_RAG.format(
                        user_id=user_name,
                        memories=json.dumps(mems, indent=4),
                    )
                    prompt = PROMPT_MEMZERO.format(context=context, question=qa["question"])

                    t0 = time.time()
                    response = llm_request(prompt)
                    response_ms = (time.time() - t0) * 1000

                new_qa = copy.deepcopy(qa)
                new_qa["context"]              = context
                new_qa["search_duration_ms"]   = round(search_ms, 1)
                new_qa["system_response"]      = response
                new_qa["response_duration_ms"] = round(response_ms, 1)
                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)
        logger.info(f"User {user_name} complete → {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        err = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(err, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        return {"uuid": uuid, "status": "error", "path": err}


# ── Entry point ───────────────────────────────────────────────────────────────

def iter_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_extraction(data_path, version="default", top_k=20, smoke=False,
                   smoke_session_limit=1, llm_model=None, prompt_template=None,
                   prompt_params=None, max_users=None) -> str:
    frame     = "rag"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    done_uuids  = {f[:-5] for f in os.listdir(tmp_dir) if f.endswith(".json")}

    print(f"\n{'='*60}")
    print(f"  HaluMem × RAG baseline (no extraction, no CUD)")
    print(f"  EMBED   : {RAG_EMBED_MODEL}")
    print(f"  QA LLM  : {os.getenv('OPENAI_MODEL')}")
    print(f"  DATA    : {data_path}")
    print(f"  VERSION : {version}  | SMOKE: {smoke}")
    print(f"{'='*60}\n")

    _tk.tracker.reset()
    start = time.time()
    users = list(iter_jsonl(data_path))
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

    _tk.save(save_path, frame)
    print(f"\n✅ Extraction done in {time.time()-start:.1f}s → {output_file}")
    return output_file
