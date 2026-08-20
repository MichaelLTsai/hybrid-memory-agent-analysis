"""Letta (MemGPT) memory system for the MemFail harness.

Letta differs from every other backend here in *who decides what to store*. Mem0,
A-MEM and StructMem run a fixed extraction pipeline on each input; Letta hands
the conversation to an agent that chooses, via tool calls, whether to write to
its in-context core blocks or to archival storage -- and may choose to write
nothing at all.

That is the reason this adapter exists. MemFail's storage stage never fires for
pipeline writers (they always write something), so the taxonomy's storage bucket
has been empty across every run. An agent-managed writer with bounded core
blocks can genuinely drop a fact, which is what makes the stage reachable.

Store snapshot = core "human" block lines + archival passages, matching the
convention used by the HaluMem Letta evaluation so numbers stay comparable.

Probe instrumentation:
    P0 trigger      -- did this write change the store at all?
    tool_calls      -- which memory tools the agent actually invoked
    silent_writes   -- turns where the agent invoked no memory tool
"""

import json
import os
import re
from typing import List, Optional

from .types import Conversation, LLMResponse, Prompt

# Tool names across Letta versions; core_* is the older API, memory_* the newer.
MEMORY_TOOLS = (
    "core_memory_append", "core_memory_replace", "archival_memory_insert",
    "memory_insert", "memory_replace", "memory_rethink",
)

PERSONA = (
    "I am a helpful assistant with long-term memory. I remember important facts "
    "the user shares, storing them so I can recall them later."
)

# Letta caps a single message; MemFail conversations can exceed it.
_MAX_MSG_CHARS = 12000


class LettaMemorySystem:
    def __init__(
        self,
        num_memories: int = 10,
        base_url: str = "http://localhost:8283",
        llm_model: str = "gemma-4-31B-it",
        embedding_model: str = "letta/letta-free",
        agent_name: Optional[str] = None,
        run_dir: Optional[str] = None,
        flush: bool = False,
    ):
        from letta_client import Letta

        self.num_memories = num_memories
        self.flush = flush
        self.client = Letta(base_url=base_url)
        # Letta routes by "<provider>/<model>"; bare names 400 on the server.
        self.llm_model = llm_model if "/" in llm_model else f"openai-proxy/{llm_model}"
        self.embedding_model = embedding_model

        agent = self.client.agents.create(
            model=self.llm_model,
            embedding=self.embedding_model,
            memory_blocks=[
                {"label": "human", "value": ""},
                {"label": "persona", "value": PERSONA},
            ],
        )
        self.agent_id = agent.id

        # MemFail's trace writer keeps no probe data, so a run's per-turn write
        # decisions are unrecoverable afterwards -- which is exactly what has to
        # be inspected when a run stores nothing. Append each decision as it
        # happens so a completed run can be explained rather than guessed at.
        self.write_log = os.path.join(run_dir or ".", "letta_writes.jsonl")

        self.probe = {
            "writes_attempted": 0,
            "writes_that_changed_store": 0,
            "silent_writes": 0,          # agent invoked no memory tool
            "tool_calls": [],            # which memory tools fired, per write
            "add_failures": 0,
            "retrievals": [],
            "store_size_trace": [],
        }

    # ── read path ───────────────────────────────────────────────────────────
    def get_memories(self, prompt: Prompt, conversation: Conversation) -> str:
        """Retrieve without letting the agent answer.

        MemFail supplies its own reader LLM, so the agent must not do the
        reasoning here -- otherwise Letta would be scored on a different task
        than every other backend.
        """
        q = str(prompt)[:1000]

        # Core blocks are in the agent's context on every turn by construction --
        # that is the whole point of MemGPT-style core memory -- so they are not
        # subject to top-k retrieval and must all be returned. Truncating them to
        # k turns "always visible" into "arbitrary k of n in insertion order",
        # which scores Letta on a retrieval step its architecture does not have.
        core = self._core_lines()

        # Archival is the part that genuinely needs retrieval; k applies there.
        archival: List[str] = []
        try:
            for pg in self.client.agents.passages.list(
                agent_id=self.agent_id, search=q, limit=self.num_memories
            ):
                t = getattr(pg, "text", None) or getattr(pg, "content", None)
                if t and t not in core:
                    archival.append(t)
        except Exception:
            pass

        mems = core + archival[: self.num_memories]
        self.probe["retrievals"].append(
            {"query": q[:300], "k": self.num_memories, "returned": mems}
        )
        return "\n".join(f"- {m}" for m in mems)

    # ── write path ──────────────────────────────────────────────────────────
    def update_memory(
        self, prompt: Prompt, response: LLMResponse, conversation_history: Conversation
    ) -> None:
        before = self._store_size()
        self.probe["writes_attempted"] += 1

        text = str(prompt)
        if len(text) > _MAX_MSG_CHARS:
            text = text[:_MAX_MSG_CHARS]

        tools_used = []
        try:
            resp = self.client.agents.messages.create(
                agent_id=self.agent_id,
                messages=[{"role": "user", "content": text}],
            )
            for m in getattr(resp, "messages", []) or []:
                if getattr(m, "message_type", "") == "tool_call_message":
                    tc = getattr(m, "tool_call", None)
                    name = getattr(tc, "name", None) if tc else None
                    if name:
                        tools_used.append(name)
        except Exception:
            self.probe["add_failures"] += 1

        mem_tools = [t for t in tools_used if t in MEMORY_TOOLS]
        self.probe["tool_calls"].append(mem_tools)
        if not mem_tools:
            self.probe["silent_writes"] += 1

        after = self._store_size()
        if after != before:
            self.probe["writes_that_changed_store"] += 1
        self.probe["store_size_trace"].append(after)

        try:
            with open(self.write_log, "a") as fh:
                fh.write(json.dumps({
                    "turn": self.probe["writes_attempted"],
                    "input": text[:200],
                    "tools": mem_tools,
                    "all_tools": tools_used,
                    "store_before": before,
                    "store_after": after,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── whole store ─────────────────────────────────────────────────────────
    def _core_lines_strict(self) -> List[str]:
        out = []
        for b in self.client.agents.blocks.list(agent_id=self.agent_id):
            if getattr(b, "label", "") == "human" and getattr(b, "value", ""):
                for line in b.value.splitlines():
                    line = re.sub(r"^[-•*\s]+", "", line).strip()
                    if len(line) > 8:
                        out.append(line)
        return out

    def _passages_strict(self) -> List[str]:
        out = []
        for p in self.client.agents.passages.list(agent_id=self.agent_id, limit=10000):
            t = getattr(p, "text", None) or getattr(p, "content", None)
            if t:
                out.append(t)
        return out

    def _core_lines(self) -> List[str]:
        try:
            return self._core_lines_strict()
        except Exception:
            return []

    def _passages(self) -> List[str]:
        try:
            return self._passages_strict()
        except Exception:
            return []

    def get_all_memories(self) -> List[str]:
        """Whole-store snapshot, used by the judge to decide storage errors.

        _core_lines/_passages swallow transport errors and return []. That is
        fine mid-run, but here an empty result is indistinguishable from "the
        agent stored nothing" and would be scored as a storage failure the
        system did not commit -- so a failure at this call is recorded rather
        than silently passed off as an empty store.
        """
        try:
            core = self._core_lines_strict()
            passages = self._passages_strict()
        except Exception as e:
            self.probe.setdefault("snapshot_failures", []).append(repr(e)[:200])
            return self._core_lines() + self._passages()
        return core + passages

    def _store_size(self) -> int:
        return len(self.get_all_memories())

    def finalize_conversation(self, conversation: Conversation) -> None:
        """Optional flush barrier between a conversation and the questions.

        Letta writes lazily: it buffers turns in the agent's context and commits
        them to memory in a batch whenever the agent decides to, which may be
        several turns later or not at all before the run ends. A store that is
        empty at question time therefore conflates "never stored" with "not yet
        flushed" -- two different failures that MemFail's taxonomy scores
        identically.

        The flush only ever fires on a turn, so waiting cannot force it; the
        barrier has to spend one. This is an intervention on the system under
        test and is off by default, so the unflushed condition stays available
        as the control.
        """
        if not self.flush:
            return
        before = self._store_size()
        try:
            self.client.agents.messages.create(
                agent_id=self.agent_id,
                messages=[{"role": "user", "content": (
                    "Before we move on: store to memory any facts from this "
                    "conversation you have not saved yet. Reply with nothing else."
                )}],
            )
        except Exception:
            self.probe["add_failures"] += 1
        after = self._store_size()
        self.probe.setdefault("flushes", []).append({"before": before, "after": after})

    def probe_summary(self) -> dict:
        p = self.probe
        n = max(p["writes_attempted"], 1)
        flat = [t for call in p["tool_calls"] for t in call]
        return {
            "P0_trigger": round(p["writes_that_changed_store"] / n, 4),
            "silent_write_rate": round(p["silent_writes"] / n, 4),
            "writes_attempted": p["writes_attempted"],
            "add_failures": p["add_failures"],
            "tool_call_counts": {t: flat.count(t) for t in sorted(set(flat))},
            "final_store_size": self._store_size(),
        }
