"""
Graphiti (temporal KG) adapter for HaluMem evaluation.

Replaces mem0 with Graphiti's knowledge graph memory system.
Uses Kuzu as the embedded graph DB — no Neo4j server needed.

Swap models by editing .env:
  GRAPHITI_LLM_MODEL=gemma-4-E4B-it  # entity/relation extraction LLM
  GRAPHITI_EMBED_MODEL=mxbai-embed-large  # embedding model via Ollama
"""

import os
import re
import json
import time
import copy
import asyncio
import logging
import traceback
from datetime import datetime, timezone
import kuzu

from dotenv import load_dotenv
from tqdm import tqdm

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.client import DEFAULT_MAX_TOKENS, ModelSize, Message
import pydantic

from prompts import PROMPT_ZEP
from llms import llm_request

load_dotenv()

NCHC_API_KEY       = os.getenv("NCHC_API_KEY", "")
NCHC_BASE_URL      = os.getenv("NCHC_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1")
GRAPHITI_LLM_MODEL = os.getenv("GRAPHITI_LLM_MODEL", os.getenv("MEM0_LLM_MODEL", "gemma-4-E4B-it"))
OLLAMA_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GRAPHITI_EMBED_MODEL = os.getenv("GRAPHITI_EMBED_MODEL", "mxbai-embed-large")
GRAPHITI_EMBED_DIM   = int(os.getenv("GRAPHITI_EMBED_DIMS", "1024"))

DATE_FORMAT = "%b %d, %Y, %H:%M:%S"

CONTEXT_TEMPLATE = """FACTS and ENTITIES represent relevant context to the current conversation.

# Facts with timestamps
<FACTS>
{facts}
</FACTS>

# Entities
<ENTITIES>
{entities}
</ENTITIES>
"""


# Common aliases that LLMs use for Graphiti's expected field names
_FIELD_ALIASES: dict[str, list[str]] = {
    "extracted_entities": ["entities", "nodes", "extracted_nodes", "entity_list", "entity_nodes"],
    "entity_resolutions": ["entities", "nodes", "resolutions", "duplicates", "resolved"],
    "edges": ["relationships", "relations", "extracted_edges", "edge_list", "facts"],
    "extracted_edges": ["edges", "relationships", "relations", "facts"],
    "nodes": ["entities", "extracted_entities", "node_list"],
    "communities": ["community_list", "groups"],
    "summaries": ["summary_list", "entity_summaries"],
}


def _fix_edge_facts(data: dict) -> dict:
    """
    Normalize each edge dict from Gemma's output to match Graphiti's Edge schema:
    - source/target → source_entity_name/target_entity_name
    - relation → relation_type (SCREAMING_SNAKE_CASE)
    - fact: true/false → natural language string
    """
    _SRC_KEYS = {"source_entity_name", "source", "from", "src"}
    _TGT_KEYS = {"target_entity_name", "target", "to", "tgt", "destination"}
    _REL_KEYS = {"relation_type", "relation", "type", "relationship", "rel"}
    _FACT_KEYS = {"fact", "description", "text", "statement"}

    def normalize_edge(e: dict) -> dict:
        if not isinstance(e, dict):
            return e
        src = next((e[k] for k in _SRC_KEYS if k in e), "")
        tgt = next((e[k] for k in _TGT_KEYS if k in e), "")
        rel = next((e[k] for k in _REL_KEYS if k in e), "RELATES_TO")
        fact = next((e[k] for k in _FACT_KEYS if k in e and isinstance(e[k], str)), None)
        if not fact:
            fact = f"{src} {rel} {tgt}."
        # Normalize relation_type to SCREAMING_SNAKE_CASE
        rel = str(rel).upper().replace(" ", "_").replace("-", "_")
        return {
            **e,
            "source_entity_name": src,
            "target_entity_name": tgt,
            "relation_type": rel,
            "fact": fact,
        }

    for key in ("edges", "extracted_edges"):
        if key in data and isinstance(data[key], list):
            data = {**data, key: [normalize_edge(e) for e in data[key]]}
    return data


def _replace_first_person(data: dict, user_name: str) -> dict:
    """
    Replace first-person references (I, me, my name, User, the user) with the
    actual user entity name so edge lookups succeed.
    """
    _FIRST_PERSON = {"i", "me", "the user", "user", "my name", "myself"}

    def fix(name: str) -> str:
        if name.lower().strip() in _FIRST_PERSON:
            return user_name
        return name

    for edge_key in ("edges", "extracted_edges"):
        if edge_key not in data:
            continue
        data = {**data, edge_key: [
            {**e,
             "source_entity_name": fix(e.get("source_entity_name", "")),
             "target_entity_name": fix(e.get("target_entity_name", ""))}
            if isinstance(e, dict) else e
            for e in data[edge_key]
        ]}
    return data


def _normalize_edge_entity_names(data: dict, messages: list) -> dict:
    """
    Normalize edge entity names to match KG node names.

    Handles Gemma's inconsistencies:
    - "ThomasSusan" (entity node) vs "Thomas" or "Susan" (edge reference)
    - "MartinezDaniel" (entity node) vs "Daniel Martinez" (edge reference)

    Matching strategy (in order):
    1. Exact match
    2. Fingerprint (sorted chars) — handles word-order swaps
    3. Prefix/suffix match — handles partial names
    4. Substring match — handles sub-word references
    """
    node_names = set()
    for msg in messages:
        content = getattr(msg, 'content', '') or ''
        for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', content):
            node_names.add(m.group(1))

    if not node_names:
        return data

    def norm(s: str) -> str:
        return s.replace(' ', '').lower()

    # Pre-build lookup structures
    fp_map   = {''.join(sorted(norm(n))): n for n in node_names}
    norm_map = {norm(n): n for n in node_names}

    def best_match(name: str) -> str:
        if name in node_names:
            return name
        n = norm(name)
        # fingerprint
        fp = ''.join(sorted(n))
        if fp in fp_map:
            return fp_map[fp]
        # exact norm
        if n in norm_map:
            return norm_map[n]
        # prefix / suffix / substring
        best = None
        best_len = 0
        for node in node_names:
            nn = norm(node)
            if nn.startswith(n) or nn.endswith(n) or n in nn:
                if len(n) > best_len:
                    best, best_len = node, len(n)
        if best:
            return best
        return name

    for edge_key in ("edges", "extracted_edges"):
        if edge_key not in data:
            continue
        data = {**data, edge_key: [
            {**e,
             "source_entity_name": best_match(e.get("source_entity_name", "")),
             "target_entity_name": best_match(e.get("target_entity_name", ""))}
            if isinstance(e, dict) else e
            for e in data[edge_key]
        ]}

    return data


def _fix_edge_duplicate(data: dict) -> dict:
    """
    EdgeDuplicate expects duplicate_facts: list[int] and contradicted_facts: list[int].
    LLMs sometimes return null, strings, wrong key names, or omit the fields entirely.
    """
    _DUP_KEYS = {"duplicate_facts", "duplicates", "duplicate_indices", "duplicate_ids"}
    _CON_KEYS = {"contradicted_facts", "contradictions", "contradiction_indices", "contradicted"}

    def to_int_list(val) -> list:
        if val is None:
            return []
        if isinstance(val, list):
            result = []
            for v in val:
                try:
                    result.append(int(v))
                except (TypeError, ValueError):
                    pass
            return result
        if isinstance(val, str):
            val = val.strip().strip("[]")
            if not val or val.lower() in ("none", "null", ""):
                return []
            try:
                return [int(x.strip()) for x in val.split(",") if x.strip().lstrip("-").isdigit()]
            except Exception:
                return []
        return []

    # Find duplicate_facts value using known aliases
    dup_val = next((data[k] for k in _DUP_KEYS if k in data), [])
    con_val = next((data[k] for k in _CON_KEYS if k in data), [])

    # Always set both fields (ensures they exist even if missing)
    data = {
        **data,
        "duplicate_facts":    to_int_list(dup_val),
        "contradicted_facts": to_int_list(con_val),
    }
    return data


def _remap_to_schema(data, model: type[pydantic.BaseModel]) -> dict:
    """
    Remap dict keys to match a Pydantic model's required fields.
    Handles common aliases and bare JSON arrays that NCHC models return.
    """
    # If Gemma returns a bare JSON array, wrap it in the model's list field
    if isinstance(data, list):
        for field_name, field_info in model.model_fields.items():
            if "list" in str(field_info.annotation).lower():
                return {field_name: data}
        return {}

    if not isinstance(data, dict):
        return data

    # If the dict looks like a JSON schema (LLM returned the schema itself), provide defaults
    if "$defs" in data or "properties" in data:
        defaults = {}
        for field_name, field_info in model.model_fields.items():
            annotation = str(field_info.annotation)
            if "list" in annotation.lower():
                defaults[field_name] = []
            elif "str" in annotation.lower():
                defaults[field_name] = ""
            elif "int" in annotation.lower():
                defaults[field_name] = 0
        return defaults

    required_fields = set(model.model_fields.keys())
    present_keys    = set(data.keys())

    # Already correct
    if required_fields.issubset(present_keys):
        return data

    result = dict(data)
    for field_name in required_fields:
        if field_name in result:
            continue
        # Try known aliases
        for alias in _FIELD_ALIASES.get(field_name, []):
            if alias in data:
                result[field_name] = data[alias]
                break
        # Fallback: try any key whose name contains the field name or vice versa
        if field_name not in result:
            for key in present_keys:
                if key.lower() in field_name.lower() or field_name.lower() in key.lower():
                    result[field_name] = data[key]
                    break
        # Last resort: provide empty defaults for list fields
        if field_name not in result:
            field_info = model.model_fields[field_name]
            annotation = str(field_info.annotation)
            if "list" in annotation.lower():
                result[field_name] = []

    return result


class NHCGraphitiLLMClient(OpenAIGenericClient):
    """
    Drop-in replacement for OpenAIGenericClient that:
    1. Keeps response_format=json_object (NCHC supports it)
    2. Strips markdown fences NCHC Gemma adds to responses
    3. Post-hoc remaps field names to match Graphiti's Pydantic schemas
    """

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[pydantic.BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict:
        result = await super().generate_response(
            messages, response_model, max_tokens, model_size, group_id, prompt_name
        )
        if response_model is not None and isinstance(result, dict):
            result = _fix_edge_facts(result)
            result = _fix_edge_duplicate(result)
            result = _remap_to_schema(result, response_model)

            if prompt_name == "extract_edges.edge":
                # Normalize entity names (concatenated names, word-order swaps)
                result = _normalize_edge_entity_names(result, messages)
                # Replace first-person references with actual user entity name
                if group_id:
                    user_name = group_id.replace("_", " ")
                    result = _replace_first_person(result, user_name)

        return result

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[pydantic.BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict:
        import openai as _openai

        openai_messages = []
        for m in messages:
            content = self._clean_input(m.content)
            role = "user" if m.role == "user" else "system"
            openai_messages.append({"role": role, "content": content})

        # Inject schema into the last user message so the model uses exact field names
        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_note = (
                f"\n\nYou MUST return a JSON object that EXACTLY matches this schema "
                f"(use these exact key names): {json.dumps(schema, ensure_ascii=False)}"
            )
            if openai_messages and openai_messages[-1]["role"] == "user":
                openai_messages[-1]["content"] += schema_note

        try:
            response = await self.client.chat.completions.create(
                model=self.model or "Llama-3.1-8B-Instruct",
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content or ""

            # Strip markdown fences that NCHC Gemma adds (e.g. ```json\n{...}\n```)
            stripped = re.sub(r"^```(?:json)?\s*", "", result.strip())
            stripped = re.sub(r"\s*```$", "", stripped.strip())

            # Try strict JSON
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

            # json5 fallback (handles single quotes, trailing commas)
            try:
                import json5
                return json5.loads(stripped)
            except Exception:
                pass

            # Last resort: extract first JSON block with greedy match
            m = re.search(r"(\{.*\})", stripped, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    try:
                        import json5
                        return json5.loads(m.group(1))
                    except Exception:
                        pass

            raise ValueError(f"Cannot parse JSON: {result[:200]}")

        except _openai.RateLimitError as e:
            from graphiti_core.exceptions import RateLimitError
            raise RateLimitError from e
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).error(f"LLM response error: {e}")
            raise


def create_kuzu_fts_indexes(graphiti: Graphiti, logger=None):
    """
    Kuzu's build_indices_and_constraints() is a no-op.
    We manually create FTS indexes after the first episode is added
    (Kuzu requires data to exist before creating FTS indexes).
    """
    driver = graphiti.driver
    if not hasattr(driver, "db"):
        return
    try:
        conn = kuzu.Connection(driver.db)
        fts_queries = [
            "CALL CREATE_FTS_INDEX('Episodic', 'episode_content', ['content', 'source', 'source_description']);",
            "CALL CREATE_FTS_INDEX('Entity', 'node_name_and_summary', ['name', 'summary']);",
            "CALL CREATE_FTS_INDEX('Community', 'community_name', ['name']);",
            "CALL CREATE_FTS_INDEX('RelatesToNode_', 'edge_name_and_fact', ['name', 'fact']);",
        ]
        for q in fts_queries:
            try:
                conn.execute(q)
            except Exception:
                pass  # index may already exist
        conn.close()
        if logger:
            logger.info("Kuzu FTS indexes created successfully")
    except Exception as e:
        if logger:
            logger.warning(f"FTS index creation warning: {e}")


def build_graphiti(kuzu_path: str, group_id: str) -> Graphiti:
    llm_client = NHCGraphitiLLMClient(
        config=LLMConfig(
            api_key=NCHC_API_KEY,
            model=GRAPHITI_LLM_MODEL,
            base_url=NCHC_BASE_URL,
            temperature=0,
            max_tokens=2000,
        )
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key="ollama",
            base_url=f"{OLLAMA_URL}/v1",
            embedding_model=GRAPHITI_EMBED_MODEL,
            embedding_dim=GRAPHITI_EMBED_DIM,
        )
    )
    # KuzuDriver is missing _database attribute that Graphiti base code expects.
    # Setting it to group_id prevents Graphiti from trying to clone the driver.
    driver = KuzuDriver(db=kuzu_path)
    driver._database = group_id
    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
    )


async def get_all_memories(graphiti: Graphiti, group_id: str) -> list[str]:
    """Return ALL facts stored in KG for this user by traversing episodes.

    Replaces the buggy approach of using user_name as a semantic query,
    which always returned the same top-2 facts regardless of session content.
    """
    from datetime import datetime, timezone
    try:
        episodes = await graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=10000,
            group_ids=[group_id],
        )
        if not episodes:
            return []
        episode_uuids = [ep.uuid for ep in episodes]
        results = await graphiti.get_nodes_and_edges_by_episode(episode_uuids)
        seen = set()
        memories = []
        for e in results.edges:
            fact     = getattr(e, "fact", None) or getattr(e, "name", "")
            valid_at = getattr(e, "valid_at", "")
            if fact and fact not in seen:
                seen.add(fact)
                memories.append(f"{valid_at}: {fact}" if valid_at else fact)
        return memories
    except Exception as e:
        return []


async def search_memories(graphiti: Graphiti, group_id: str, query: str, top_k: int = 20):
    """Search KG and return (context_str, memories_list)."""
    try:
        edges = await graphiti.search(
            query=query,
            group_ids=[group_id],
            num_results=top_k,
        )
    except Exception as e:
        return "", []

    facts    = []
    memories = []
    for e in edges:
        fact     = getattr(e, "fact", None) or getattr(e, "name", "")
        valid_at = getattr(e, "valid_at", "")
        if fact:
            facts.append(f"  - {fact} (event_time: {valid_at})")
            memories.append(f"{valid_at}: {fact}" if valid_at else fact)

    context = CONTEXT_TEMPLATE.format(
        facts="\n".join(facts) if facts else "  (none)",
        entities="  (see facts above)",
    )
    return context, memories


def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if match:
        return match.group(1).strip().replace(" ", "_")
    raise ValueError(f"Cannot parse name: {persona_info[:100]}")


def setup_logger(log_dir: str, uuid: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"halumem.graphiti.{uuid}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, f"{uuid}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    return logger


async def process_user_async(
    user_data: dict,
    top_k: int,
    save_path: str,
    log_dir: str,
    kuzu_root: str,
    smoke_session_limit: int = None,
) -> dict:
    uuid           = user_data["uuid"]
    user_name      = extract_user_name(user_data["persona_info"])  # "Martin_Mark"
    display_name   = user_name.replace("_", " ")                   # "Martin Mark" for episode text
    group_id       = user_name                                      # "Martin_Mark" for KG user_id

    logger = setup_logger(log_dir, uuid)
    logger.info(f"=== Start {user_name} ({uuid}) | LLM={GRAPHITI_LLM_MODEL} ===")

    tmp_dir  = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{uuid}.json")
    session_log_file = os.path.join(log_dir, f"{uuid}_sessions.jsonl")

    # Kuzu db path — one file per user (Kuzu creates its own internal structure)
    kuzu_path = os.path.join(kuzu_root, uuid)

    graphiti = build_graphiti(kuzu_path, group_id)
    # Create FTS indexes immediately — Kuzu tables exist after setup_schema(), data not required
    create_kuzu_fts_indexes(graphiti, logger)
    _indices_built = True

    # Compact persona line prepended to every episode so LLM always knows who the user is
    persona_line = f"[User Profile: {user_data['persona_info']}]"

    # Custom extraction instruction reused across all add_episode calls
    extraction_instruction = (
        f"The speaker of this conversation is '{display_name}'. "
        f"The user profile above describes {display_name}. "
        f"Always extract '{display_name}' as an entity when 'I', 'me', or 'my' appears. "
        "Keep full person names as a single entity (e.g. 'John Smith', not split). "
        "When extracting edges, use the speaker's full name as source, not 'I' or 'User'. "
        "Use ONLY entity names exactly as listed in the ENTITIES section."
    )

    sessions = user_data["sessions"]
    if smoke_session_limit:
        sessions = sessions[:smoke_session_limit]

    new_user_data = {"uuid": uuid, "user_name": user_name, "sessions": []}

    try:
        for sid, session in enumerate(tqdm(sessions, desc=f"User {user_name}")):
            session_wall_start = time.time()

            new_session = {
                "memory_points": session["memory_points"],
                "dialogue":      session["dialogue"],
            }

            if session.get("is_generated_qa_session", False):
                new_session["is_generated_qa_session"] = True
                new_session["add_dialogue_duration_ms"] = 0
                del new_session["dialogue"]
                del new_session["memory_points"]
                new_user_data["sessions"].append(new_session)
                continue

            # ── Turn-level processing ────────────────────────────────────────
            # Group dialogue messages by dialogue_turn number (user + assistant share same turn)
            from collections import defaultdict
            turns: dict[int, list] = defaultdict(list)
            for msg in session["dialogue"]:
                turns[msg.get("dialogue_turn", 0)].append(msg)

            t0_session = time.time()
            for turn_idx in sorted(turns.keys()):
                turn_msgs = turns[turn_idx]

                # Use turn timestamp (first message of the turn)
                raw_ts = turn_msgs[0].get("timestamp", session["start_time"])
                try:
                    dt = datetime.strptime(raw_ts, DATE_FORMAT).replace(tzinfo=timezone.utc)
                except ValueError:
                    dt = datetime.strptime(session["start_time"], DATE_FORMAT).replace(tzinfo=timezone.utc)

                # Format: mark each message with [USER] or [ASSISTANT]
                turn_text = "\n".join(
                    f"[{'USER' if m['role'] == 'user' else 'ASSISTANT'}]: {m['content']}"
                    for m in turn_msgs
                )

                # Prepend persona + speaker identity so LLM always knows who the user is.
                episode_text = (
                    f"{persona_line}\n"
                    f"[Speaker: {display_name}]\n"
                    f"{turn_text}"
                )

                try:
                    t_turn = time.time()
                    await graphiti.add_episode(
                        name=f"{user_name}_s{sid}_t{turn_idx}",
                        episode_body=episode_text,
                        source=EpisodeType.text,
                        source_description=f"HaluMem — {display_name} session {sid} turn {turn_idx}",
                        reference_time=dt,
                        group_id=group_id,
                        custom_extraction_instructions=extraction_instruction,
                    )
                    logger.debug(f"Session {sid} turn {turn_idx} OK ({(time.time()-t_turn)*1000:.0f}ms)")
                except Exception as e:
                    logger.warning(f"Session {sid} turn {turn_idx} add_episode error: {e}")

            add_ms = (time.time() - t0_session) * 1000
            # ────────────────────────────────────────────────────────────────

            # extracted_memories = ALL facts in KG for this user after this session
            # Uses episode traversal instead of semantic search to avoid always
            # returning the same top-k facts matched to the user's name.
            extracted_memories = await get_all_memories(graphiti, group_id)
            new_session["extracted_memories"]       = extracted_memories
            new_session["add_dialogue_duration_ms"] = add_ms
            logger.info(f"Session {sid} | {len(extracted_memories)} facts in KG | {add_ms:.0f}ms")

            # Search for update-related memories
            for memory in new_session["memory_points"]:
                if memory["is_update"] == "False" or not memory.get("original_memories"):
                    continue
                _, memories_from_system = await search_memories(
                    graphiti, group_id, memory["memory_content"], top_k=10
                )
                memory["memories_from_system"] = memories_from_system

            # QA
            if "questions" not in session:
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            qa_log_entries = []

            for qa in session["questions"]:
                t0 = time.time()
                context, _ = await search_memories(graphiti, group_id, qa["question"], top_k=top_k)
                search_ms = (time.time() - t0) * 1000

                prompt = PROMPT_ZEP.format(context=context, question=qa["question"])

                t0 = time.time()
                response = llm_request(prompt)
                response_ms = (time.time() - t0) * 1000

                new_qa = copy.deepcopy(qa)
                new_qa["context"]           = context
                new_qa["search_duration_ms"]  = search_ms
                new_qa["system_response"]     = response
                new_qa["response_duration_ms"] = response_ms
                new_session["questions"].append(new_qa)

                qa_log_entries.append({
                    "question":        qa["question"],
                    "expected_answer": qa["answer"],
                    "system_response": response,
                    "question_type":   qa.get("question_type"),
                    "difficulty":      qa.get("difficulty"),
                    "search_ms":       round(search_ms, 1),
                    "response_ms":     round(response_ms, 1),
                })
                logger.debug(f"Session {sid} | Q: {qa['question'][:60]} → {response[:80]}")

            new_user_data["sessions"].append(new_session)

            session_elapsed_ms = (time.time() - session_wall_start) * 1000
            session_log = {
                "session_id":            sid,
                "start_time":            session["start_time"],
                "extracted_memory_count": len(extracted_memories),
                "extracted_memories":    extracted_memories,
                "question_count":        len(new_session["questions"]),
                "qa_results":            qa_log_entries,
                "session_elapsed_ms":    round(session_elapsed_ms, 1),
            }
            with open(session_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_log, ensure_ascii=False) + "\n")

            logger.info(f"Session {sid} done | {len(new_session['questions'])} QA | {session_elapsed_ms:.0f}ms")

        await graphiti.close()

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False, indent=2)

        logger.info(f"User {user_name} complete → {tmp_file}")
        return {"uuid": uuid, "status": "ok", "path": tmp_file}

    except Exception:
        tb = traceback.format_exc()
        error_path = os.path.join(tmp_dir, f"{uuid}_error.log")
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(tb)
        logger.error(f"FAILED:\n{tb}")
        try:
            await graphiti.close()
        except Exception:
            pass
        return {"uuid": uuid, "status": "error", "path": error_path}


def process_user(user_data, top_k, save_path, log_dir, kuzu_root, smoke_session_limit=None):
    """Sync wrapper — asyncio.run() is safe here since each user is independent."""
    return asyncio.run(
        process_user_async(user_data, top_k, save_path, log_dir, kuzu_root, smoke_session_limit)
    )


def iter_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_extraction(
    data_path: str,
    version: str = "default",
    top_k: int = 20,
    smoke: bool = False,
    smoke_session_limit: int = 1,
    llm_model: str = None,
    prompt_template: str = None,
    prompt_params: dict = None,
    max_users: int = None,
) -> str:
    frame     = "graphiti"
    save_path = f"./results/{frame}-{version}/"
    log_dir   = f"./logs/{frame}-{version}/"
    kuzu_root = os.path.join(save_path, "kuzu_data")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(kuzu_root, exist_ok=True)

    output_file = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir     = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    done_uuids  = {f[:-5]  for f in os.listdir(tmp_dir) if f.endswith(".json")}
    error_uuids = {f[:-10] for f in os.listdir(tmp_dir) if f.endswith("_error.log")}

    print(f"\n{'='*60}")
    print(f"  HaluMem × Graphiti (Kuzu embedded)")
    print(f"  GRAPHITI_LLM  : {GRAPHITI_LLM_MODEL}")
    print(f"  EMBED         : {GRAPHITI_EMBED_MODEL}")
    print(f"  DATA          : {data_path}")
    print(f"  VERSION       : {version}")
    print(f"  SMOKE         : {smoke}")
    if done_uuids:
        print(f"  RESUME        : {len(done_uuids)} users already done, skipping")
    if error_uuids:
        print(f"  RETRY         : {len(error_uuids)} errored users will be retried")
    print(f"{'='*60}\n")

    start_time = time.time()
    users = list(iter_jsonl(data_path))
    if smoke:
        users = users[:1]
    elif max_users:
        users = users[:max_users]

    total = len(users)
    for idx, user_data in enumerate(users, 1):
        uuid = user_data["uuid"]

        if uuid in done_uuids and uuid not in error_uuids:
            print(f"⏭️  [{idx}/{total}] {uuid} already done, skipping")
            continue

        error_log = os.path.join(tmp_dir, f"{uuid}_error.log")
        if os.path.exists(error_log):
            os.remove(error_log)

        result = process_user(
            user_data=user_data,
            top_k=top_k,
            save_path=save_path,
            log_dir=log_dir,
            kuzu_root=kuzu_root,
            smoke_session_limit=smoke_session_limit if smoke else None,
        )
        icon = "✅" if result["status"] == "ok" else "❌"
        print(f"{icon} [{idx}/{total}] {result['uuid']} → {result['status']}")

    # Merge tmp → final JSONL
    with open(output_file, "w", encoding="utf-8") as f_out:
        for fname in sorted(os.listdir(tmp_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(tmp_dir, fname), encoding="utf-8") as f_in:
                    f_out.write(json.dumps(json.load(f_in), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️  Skipped {fname}: {e}")

    elapsed = time.time() - start_time
    print(f"\n✅ Extraction done in {elapsed:.1f}s → {output_file}")
    return output_file
