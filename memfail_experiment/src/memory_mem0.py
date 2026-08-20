"""
Mem0 memory system for the MemFail harness.

MemFail ships the datasets, the evaluation loop and the failure-mode attribution
scripts, but NOT the memory-system adapters the paper evaluates -- `src/` only
contains the two trivial baselines. This module supplies the Mem0 one.

Contract (src/types.py MemorySystem protocol):
    get_memories(prompt, conversation) -> str      # build the context slice
    update_memory(prompt, response, history)       # write the turn into memory
    get_all_memories() -> list[str]                # whole store (drives attribution)

Probe instrumentation (subset that applies to MemFail's current tasks):
    P0  trigger   -- did a store_conversation call change the store at all?
                     Mem0 is a pipeline, so this is ~1.0 by construction; it only
                     discriminates once an autonomous backend (Letta) is added.
    P1  capture   -- MemFail's own storage_check reads get_all_memories().
    P4  retrieval -- retrieved slice is recorded per query.
    Update probes (P2 align / P3 resolve / P4b stale) are NOT instrumented here:
    none of MemFail's four tasks contains a supersession, so there is no v1->v2
    pair to measure. They arrive with the update dataset.

Mem0 notes carried over from the HaluMem/LongMemEval runs:
  · max_tokens must be generous -- the v1 update-decision prompt echoes the whole
    retrieved memory set, and 2000 silently truncates it into an empty response.
  · get_all() defaults to limit=100 and silently truncates a larger store.
"""

import os
import time
from typing import List, Optional

from .types import Conversation, LLMResponse, Prompt

_DEF_MAX_TOKENS = int(os.getenv("MEM0_MAX_TOKENS", "8000"))

def _mem0_major() -> int:
    try:
        import mem0
        return int(str(mem0.__version__).split(".")[0])
    except Exception:
        return 1
_GET_ALL_LIMIT = 100000          # get_all() default is 100 and truncates silently


_KNOWN_DIMS = {
    "bge-m3:latest": 1024, "bge-m3": 1024,
    "all-minilm-l6-v2": 384, "nomic-embed-text:latest": 768,
    "mxbai-embed-large:latest": 1024, "qwen3-embedding:0.6b": 1024,
    "text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
}


def _allow_cross_thread_qdrant():
    """Let the on-disk qdrant store be used from a thread other than the one
    that opened it.

    macOS ships SQLite compiled THREADSAFE=2, so qdrant's local persistence
    sets check_same_thread=True. Mem0 v1 runs _add_to_vector_store inside a
    ThreadPoolExecutor, so writes land on a worker thread and sqlite rejects
    them -- silently, as "Error processing memory action" on stderr while add()
    still returns. Mem0 v2 writes on the calling thread and never trips this.

    Access stays serialised: only the vector-store future touches qdrant (the
    graph store is not configured), so this relaxes the thread *identity* check
    without introducing concurrent use of the connection.
    """
    try:
        from qdrant_client.local.persistence import CollectionPersistence
        CollectionPersistence.CHECK_SAME_THREAD = False
    except Exception:
        pass


def _resolve_dims(provider, model, explicit):
    if explicit:
        return int(explicit)
    if model and model.lower() in _KNOWN_DIMS:
        return _KNOWN_DIMS[model.lower()]
    return 1536          # Mem0/OpenAI default


class Mem0MemorySystem:
    """Mem0 (OSS) backed memory with per-call probe accounting."""

    def __init__(
        self,
        num_memories: int = 10,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4.1-mini",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        ollama_base_url: Optional[str] = None,
        shared_user_id: str = "memfail_user",
        run_dir: Optional[str] = None,
        collection_name: str = "memfail",
    ):
        from mem0 import Memory

        self.num_memories = num_memories
        self.user_id = shared_user_id or "memfail_user"
        base = os.path.join(run_dir, "qdrant") if run_dir else "./qdrant_memfail"

        llm_cfg = {
            "model": llm_model,
            "temperature": 0.0,
            "max_tokens": _DEF_MAX_TOKENS,
        }
        if llm_api_key:
            llm_cfg["api_key"] = llm_api_key
        if llm_base_url:
            llm_cfg["openai_base_url"] = llm_base_url

        config = {
            "llm": {"provider": llm_provider, "config": llm_cfg},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "path": os.path.join(base, collection_name),
                    "on_disk": True,
                    # Must match the embedder: Mem0 defaults the collection to
                    # 1536 (OpenAI); bge-m3 is 1024 and the mismatch only shows
                    # up as a shape error deep inside qdrant on the first add().
                    "embedding_model_dims": _resolve_dims(embedding_provider, embedding_model, embedding_dims),
                },
            },
        }
        if embedding_provider:
            emb_cfg = {}
            if embedding_model:
                emb_cfg["model"] = embedding_model
            if embedding_provider == "ollama" and ollama_base_url:
                emb_cfg["ollama_base_url"] = ollama_base_url
            emb_cfg["embedding_dims"] = _resolve_dims(embedding_provider, embedding_model, embedding_dims)
            if embedding_provider == "openai":
                if llm_api_key:
                    emb_cfg["api_key"] = llm_api_key
                if llm_base_url:
                    emb_cfg["openai_base_url"] = llm_base_url
            config["embedder"] = {"provider": embedding_provider, "config": emb_cfg}

        self.mem0_major = _mem0_major()
        _allow_cross_thread_qdrant()
        self.memory = Memory.from_config(config)

        # ── probe accounting ────────────────────────────────────────────────
        self.probe = {
            "writes_attempted": 0,
            "writes_that_changed_store": 0,   # P0 trigger numerator
            "add_events": [],                 # ADD / UPDATE / DELETE + previous_memory
            "add_failures": 0,
            "retrievals": [],                 # per query: {query, k, returned}
            "store_size_trace": [],
        }

    # ── read path ───────────────────────────────────────────────────────────
    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        try:
            if self.mem0_major >= 2:
                res = self.memory.search(
                    query=str(prompt), filters={"user_id": self.user_id},
                    top_k=self.num_memories,
                )
            else:
                res = self.memory.search(
                    query=str(prompt), user_id=self.user_id, limit=self.num_memories
                )
            items = res.get("results", res) if isinstance(res, dict) else res
            mems = [it.get("memory", "") for it in (items or []) if it.get("memory")]
        except Exception:
            mems = []
        self.probe["retrievals"].append(
            {"query": str(prompt)[:300], "k": self.num_memories, "returned": mems}
        )
        return "\n".join(f"- {m}" for m in mems)

    # ── write path ──────────────────────────────────────────────────────────
    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        before = len(self._all_raw())
        self.probe["writes_attempted"] += 1
        messages = [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
        try:
            res = self.memory.add(messages, user_id=self.user_id)
            events = [
                {k: it.get(k) for k in ("id", "memory", "event", "previous_memory")}
                for it in (res.get("results", []) if isinstance(res, dict) else [])
            ]
            self.probe["add_events"].extend(events)
        except Exception:
            self.probe["add_failures"] += 1
        after = self._all_raw()
        if len(after) != before:
            self.probe["writes_that_changed_store"] += 1
        self.probe["store_size_trace"].append(len(after))

    # ── whole store (drives MemFail's attribution + P1) ─────────────────────
    def _all_raw(self) -> list:
        """Whole store. v1 takes user_id/limit; v2 takes filters/top_k."""
        try:
            if self.mem0_major >= 2:
                res = self.memory.get_all(
                    filters={"user_id": self.user_id}, top_k=_GET_ALL_LIMIT
                )
            else:
                res = self.memory.get_all(user_id=self.user_id, limit=_GET_ALL_LIMIT)
            items = res.get("results", res) if isinstance(res, dict) else res
            return items or []
        except Exception:
            return []

    def get_all_memories(self) -> List[str]:
        return [it.get("memory", "") for it in self._all_raw() if it.get("memory")]

    def finalize_conversation(self, conversation: Conversation) -> None:
        pass

    # ── probe export ────────────────────────────────────────────────────────
    def probe_summary(self) -> dict:
        from collections import Counter
        p = self.probe
        n = max(p["writes_attempted"], 1)
        return {
            "P0_trigger": round(p["writes_that_changed_store"] / n, 4),
            "writes_attempted": p["writes_attempted"],
            "add_failures": p["add_failures"],
            "event_counts": dict(Counter(e["event"] for e in p["add_events"] if e.get("event"))),
            "final_store_size": len(self.get_all_memories()),
            "store_size_trace": p["store_size_trace"],
            "n_retrievals": len(p["retrievals"]),
            "avg_retrieved": round(
                sum(len(r["returned"]) for r in p["retrievals"]) / max(len(p["retrievals"]), 1), 2
            ),
        }
