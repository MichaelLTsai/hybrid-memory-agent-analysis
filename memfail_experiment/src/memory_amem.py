"""
A-MEM memory system for the MemFail harness.

A-MEM (agiresearch/A-mem) writes one LLM-authored *note* per input, links notes
to semantic neighbours, and can "evolve" existing notes when new ones arrive.
Retrieval is embedding search over the note store.

Not shipped in MemFail's repo (only the two trivial baselines are), so this
supplies it against the same three-function contract.

Probe instrumentation (the subset MemFail's current tasks can exercise):
    P0 trigger -- did this write actually create a note?
    store trace / retrieval log -- feed P1 and P4.
Update probes (P2/P3/P4b) need a supersession dataset and are added with it.
"""

import os
from typing import List, Optional

from .types import Conversation, LLMResponse, Prompt


class AMEMMemorySystem:
    def __init__(
        self,
        num_memories: int = 10,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        run_dir: Optional[str] = None,
    ):
        from agentic_memory.memory_system import AgenticMemorySystem

        self.num_memories = num_memories
        # A-MEM reads the OpenAI creds from the environment.
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_BASE"] = base_url

        self.memory = AgenticMemorySystem(
            model_name=embedding_model,
            llm_backend=llm_backend,
            llm_model=llm_model,
            evo_threshold=evo_threshold,
            api_key=api_key,
        )

        self.probe = {
            "writes_attempted": 0,
            "writes_that_changed_store": 0,
            "note_ids": [],
            "add_failures": 0,
            "retrievals": [],
            "store_size_trace": [],
        }

    # ── read path ───────────────────────────────────────────────────────────
    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        mems: List[str] = []
        try:
            for hit in (self.memory.search_agentic(str(prompt), k=self.num_memories) or []):
                txt = hit.get("content") if isinstance(hit, dict) else getattr(hit, "content", None)
                if txt:
                    mems.append(txt)
        except Exception:
            pass
        self.probe["retrievals"].append(
            {"query": str(prompt)[:300], "k": self.num_memories, "returned": mems}
        )
        return "\n".join(f"- {m}" for m in mems)

    # ── write path ──────────────────────────────────────────────────────────
    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        before = len(self._notes())
        self.probe["writes_attempted"] += 1
        try:
            nid = self.memory.add_note(content=str(prompt))
            if nid:
                self.probe["note_ids"].append(nid)
        except Exception:
            self.probe["add_failures"] += 1
        after = len(self._notes())
        if after != before:
            self.probe["writes_that_changed_store"] += 1
        self.probe["store_size_trace"].append(after)

    # ── whole store ─────────────────────────────────────────────────────────
    def _notes(self) -> dict:
        return getattr(self.memory, "memories", {}) or {}

    def get_all_memories(self) -> List[str]:
        out = []
        for note in self._notes().values():
            txt = getattr(note, "content", None)
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
            "final_store_size": len(self.get_all_memories()),
            "store_size_trace": p["store_size_trace"],
            "n_retrievals": len(p["retrievals"]),
            "avg_retrieved": round(
                sum(len(r["returned"]) for r in p["retrievals"]) / max(len(p["retrievals"]), 1), 2
            ),
        }
