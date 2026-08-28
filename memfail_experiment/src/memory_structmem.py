"""
StructMem memory system for the MemFail harness.

StructMem (Zhejiang Univ. + Ant Group) is not a standalone package: it is
LightMem run with `extraction_mode="event"` plus periodic cross-event
consolidation via summarize(). Two levels:
  1. event-level bindings -- dual-perspective extraction (factual + relational
     entries), each anchored to its originating timestamp;
  2. cross-event consolidation -- summarize() synthesises higher-level
     relations across time.

Not shipped in MemFail's repo, so this supplies it against the same
three-function contract.

Config choices, mirroring the HaluMem adapter so the two are comparable:
  · pre_compress / topic_segment OFF -- the reference config needs LLMLingua-2
    on CUDA; disabling keeps the comparison about MEMORY, not compression.
  · messages_use = "hybrid" -- ingests user AND assistant turns.
  · memory_manager = openai backend, pointed at whatever base_url is given.

Requires LightMem, which pins python <3.12 -- run this backend from the 3.11
environment (~/structmem_env), not venv_memfail.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional

from .types import Conversation, LLMResponse, Prompt


class StructMemMemorySystem:
    def __init__(
        self,
        num_memories: int = 10,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        segmenter_model: Optional[str] = None,
        segmenter_device: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
        embedding_device: str = "cpu",
        qdrant_path: Optional[str] = None,
        collection_name: str = "structmem_memfail",
        enable_summary: bool = True,
        summarize_every: Optional[int] = None,
        run_dir: Optional[str] = None,
    ):
        from lightmem.memory.lightmem import LightMemory

        self.num_memories = num_memories
        self.enable_summary = enable_summary
        # Cross-event consolidation is expensive; run it every N writes.
        self.summarize_every = summarize_every or int(os.getenv("STRUCTMEM_SUMMARIZE_EVERY", "10"))

        # Each ablation arm owns its own collections so no two arms share a store.
        from lightmem.memory.state import config as _state_config
        self.state_cfg = _state_config.from_env()
        if run_dir:
            self.state_cfg.trace_dir = os.path.join(run_dir, "traces")
        collection_name = collection_name + self.state_cfg.collection_suffix()

        base = qdrant_path or (os.path.join(run_dir, "structmem_qdrant") if run_dir
                               else "./structmem_qdrant")
        shutil.rmtree(os.path.join(base, collection_name), ignore_errors=True)
        shutil.rmtree(os.path.join(base, collection_name + "_sum"), ignore_errors=True)

        # LightMem's factory schema is model_name/configs, and its OpenAI manager
        # reads config.model + config.openai_base_url (not model_name_or_path).
        llm = {"model_name": "openai", "configs": {
            "model": model, "api_key": api_key, "openai_base_url": base_url,
            "max_tokens": 8192, "temperature": 0.0}}
        emb = {"model_name": "huggingface", "configs": {
            "model": embedding_model, "embedding_dims": embedding_dimension,
            "model_kwargs": {"device": embedding_device}}}

        cfg = {
            "state_ablation": self.state_cfg,
            "pre_compress": False,
            "topic_segment": False,
            "messages_use": "hybrid",
            "metadata_generate": True,
            "text_summary": True,
            "extraction_mode": "event",          # <- StructMem
            "memory_manager": llm,
            "extract_threshold": 0.1,
            "index_strategy": "embedding",
            "text_embedder": emb,
            "retrieve_strategy": "embedding",
            "embedding_retriever": {"model_name": "qdrant", "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dimension,
                "path": os.path.join(base, collection_name)}},
            "summary_retriever": {"model_name": "qdrant", "configs": {
                "collection_name": collection_name + "_sum",
                "embedding_model_dims": embedding_dimension,
                "path": os.path.join(base, collection_name + "_sum")}},
        }
        self.memory = LightMemory.from_config(cfg)

        self.probe = {
            "writes_attempted": 0,
            "writes_that_changed_store": 0,
            "add_failures": 0,
            "consolidations": 0,
            "retrievals": [],
            "store_size_trace": [],
        }

    # ── read path ───────────────────────────────────────────────────────────
    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        try:
            # Dual-circuit retrieval; state-aware ordering only when M4 is on.
            mems, _packet = self.memory.retrieve_for_qa(str(prompt), limit=self.num_memories)
            mems = mems or []
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
        before = len(self._entries())
        self.probe["writes_attempted"] += 1
        ts = datetime.now().strftime("%Y/%m/%d (%a) %H:%M")
        msgs = [{"role": "user", "content": str(prompt), "time_stamp": ts},
                {"role": "assistant", "content": str(response), "time_stamp": ts}]
        try:
            self.memory.add_memory(msgs, force_extract=True)
        except Exception:
            self.probe["add_failures"] += 1

        if self.enable_summary and self.probe["writes_attempted"] % self.summarize_every == 0:
            # LightMem's offline update was never wired into this adapter, so
            # MemFail was previously measured with no update mechanism running.
            # Enabled here for every arm: E0 gets the original
            # update/delete/ignore, the M1 arms get the state commit.
            def _run_update():
                if os.getenv("STRUCTMEM_OFFLINE_UPDATE", "1") != "1":
                    return
                try:
                    self.memory.construct_update_queue_all_entries(top_k=20, keep_top_n=10)
                    self.memory.offline_update_all_entries(score_threshold=0.9)
                except Exception:
                    pass

            def _run_summarize():
                try:
                    self.memory.summarize(process_all=True, enable_cross_event=True,
                                          retrieval_scope="global", top_k_seeds=15)
                    self.probe["consolidations"] += 1
                except Exception:
                    pass

            # M3 needs the state commit before summaries are written.
            if self.state_cfg.enable_m3_summary_sync:
                _run_update(); _run_summarize()
            else:
                _run_summarize(); _run_update()

        after = len(self._entries())
        if after != before:
            self.probe["writes_that_changed_store"] += 1
        self.probe["store_size_trace"].append(after)

    # ── whole store ─────────────────────────────────────────────────────────
    def _entries(self) -> list:
        try:
            return self.memory.embedding_retriever.get_all() or []
        except Exception:
            return []

    def get_all_memories(self) -> List[str]:
        out = []
        for e in self._entries():
            pl = getattr(e, "payload", None) or (e.get("payload") if isinstance(e, dict) else {}) or {}
            txt = pl.get("memory") or pl.get("text") or pl.get("content")
            if txt:
                out.append(txt)
        return out

    def finalize_conversation(self, conversation: Conversation) -> None:
        pass

    def probe_summary(self) -> dict:
        p = self.probe
        n = max(p["writes_attempted"], 1)
        return {
            "P0_trigger": round(p["writes_that_changed_store"] / n, 4),
            "writes_attempted": p["writes_attempted"],
            "add_failures": p["add_failures"],
            "consolidations": p["consolidations"],
            "final_store_size": len(self.get_all_memories()),
            "store_size_trace": p["store_size_trace"],
            "n_retrievals": len(p["retrievals"]),
            "avg_retrieved": round(
                sum(len(r["returned"]) for r in p["retrievals"]) / max(len(p["retrievals"]), 1), 2
            ),
        }
