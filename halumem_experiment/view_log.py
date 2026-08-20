"""
Log viewer for HaluMem experiments (mem0 and Graphiti).

Usage:
    # Default backend = mem0
    python view_log.py --version smoke_nchc_gemma4

    # Graphiti backend
    python view_log.py --version smoke_graphiti --backend graphiti

    # Per-session evaluation scores
    python view_log.py --version smoke_graphiti --backend graphiti --eval-per-session

    # Detailed eval for session 0
    python view_log.py --version smoke_graphiti --backend graphiti --eval-session 0

    # Full dialogue + memories + QA for session 0
    python view_log.py --version smoke_graphiti --backend graphiti --session 0

    # Show QA retrieved context (what KG facts were used)
    python view_log.py --version smoke_graphiti --backend graphiti --session 0 --show-context

    # Only extracted memories
    python view_log.py --version smoke_graphiti --backend graphiti --memories-only
"""

import argparse
import json
import os

FRAME_MAP = {"mem0": "mem0_oss", "graphiti": "graphiti"}


def load_eval_detail(version: str, frame: str = "mem0_oss"):
    """Load evaluation detail records, grouped by (uuid, session_id)."""
    detail_file = f"./results/{frame}-{version}/{frame}_eval_detail.jsonl"
    if not os.path.exists(detail_file):
        return {}
    from collections import defaultdict
    data = defaultdict(lambda: {"integrity": [], "accuracy": [], "update": [], "qa": []})
    with open(detail_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("uuid", ""), r.get("ssession_id", 0))
            if "memory_integrity_score" in r:
                data[key]["integrity"].append(r)
            elif "memory_accuracy_score" in r:
                data[key]["accuracy"].append(r)
            elif "memory_update_type" in r:
                data[key]["update"].append(r)
            elif "result_type" in r:
                data[key]["qa"].append(r)
    return data


def print_eval_per_session(users, eval_detail):
    """Per-session evaluation breakdown."""
    SCORE_SYM = {2: "✅", 1: "△", 0: "❌", None: "?"}
    UPDATE_SYM = {"Correct": "✅", "Hallucination": "🔴", "Omission": "🟡", "Other": "⚪", None: "?"}
    QA_SYM    = {"Correct": "✅", "Hallucination": "🔴", "Omission": "🟡", None: "?"}

    for user in users:
        uuid = user["uuid"]
        name = user.get("user_name", uuid[:8])
        sessions = [s for s in user.get("sessions", []) if not s.get("is_generated_qa_session")]

        print_divider()
        print(f"  USER: {name}  ({len(sessions)} sessions)")
        print_divider()

        for sid, session in enumerate(sessions):
            key = (uuid, sid)
            ev  = eval_detail.get(key, {"integrity": [], "accuracy": [], "update": [], "qa": []})

            integrity_recs = ev["integrity"]
            update_recs    = ev["update"]
            qa_recs        = ev["qa"]

            # Ground truth counts
            gt_all         = [m for m in session.get("memory_points", []) if m.get("memory_source") != "interference"]
            gt_update      = [m for m in gt_all if m.get("is_update") == "True"]
            gt_new         = [m for m in gt_all if m.get("is_update") == "False"]
            n_extracted    = len(session.get("extracted_memories", []))

            # Recall from integrity records (non-interference)
            int_non_interf = [r for r in integrity_recs if r.get("memory_source") != "interference"]
            recalled       = sum(1 for r in int_non_interf if r.get("memory_integrity_score") == 2)
            recall_str     = f"{recalled}/{len(int_non_interf)}" if int_non_interf else f"0/{len(gt_new)}"

            # Update results
            upd_str = ""
            if update_recs:
                upd_parts = [f"{UPDATE_SYM[r.get('memory_update_type')]} {r.get('memory_update_type','?')}" for r in update_recs]
                upd_str = "  |  Updates: " + ", ".join(upd_parts)
            elif gt_update:
                upd_str = f"  |  Updates: {len(gt_update)} expected (not evaluated)"

            # QA results
            qa_str = ""
            if qa_recs:
                qa_parts = [f"{QA_SYM[r.get('result_type')]}" for r in qa_recs]
                qa_str = "  |  QA: " + " ".join(qa_parts)
            elif session.get("questions"):
                qa_str = f"  |  QA: {len(session['questions'])} (not evaluated)"

            print(f"  Sess {sid:>3}  GT={len(gt_all):>3}(new={len(gt_new)},upd={len(gt_update)})  "
                  f"Extracted={n_extracted:>3}  Recall={recall_str}"
                  f"{upd_str}{qa_str}")

        print()

    print_divider()
    print("Legend: ✅ Correct/Full  △ Partial  ❌ Missing  🔴 Hallucination  🟡 Omission")
    print_divider()


def print_eval_session_detail(users, eval_detail, sid: int):
    """Detailed per-memory and per-QA evaluation for one session across all users."""
    SCORE_LABEL = {2: "FULL ✅", 1: "PART △", 0: "MISS ❌", None: "ERR ?"}
    UPDATE_SYM  = {"Correct": "✅ Correct", "Hallucination": "🔴 Hallucination",
                   "Omission": "🟡 Omission", "Other": "⚪ Other", None: "?"}
    QA_SYM      = {"Correct": "✅", "Hallucination": "🔴", "Omission": "🟡", None: "?"}

    for user in users:
        uuid = user["uuid"]
        name = user.get("user_name", uuid[:8])
        sessions = [s for s in user.get("sessions", []) if not s.get("is_generated_qa_session")]
        if sid >= len(sessions):
            continue
        session = sessions[sid]
        key = (uuid, sid)
        ev  = eval_detail.get(key, {"integrity": [], "accuracy": [], "update": [], "qa": []})

        print_divider()
        print(f"  USER: {name}  |  SESSION: {sid}  |  {session.get('start_time','')}")
        print_divider()

        # --- Memory Points: new ---
        gt_new = [m for m in session.get("memory_points", [])
                  if m.get("memory_source") != "interference" and m.get("is_update") == "False"]
        if gt_new:
            print(f"\n  ── Ground Truth Memory Points (NEW, {len(gt_new)}) ──")
            int_map = {r["memory_content"]: r for r in ev["integrity"]}
            for m in gt_new:
                rec   = int_map.get(m["memory_content"])
                score = rec.get("memory_integrity_score") if rec else None
                label = SCORE_LABEL.get(score, "?")
                src   = m.get("memory_source", "?")
                print(f"  [{label}][{src}] {m['memory_content']}")

        # --- Memory Points: update ---
        gt_upd = [m for m in session.get("memory_points", [])
                  if m.get("memory_source") != "interference" and m.get("is_update") == "True"]
        if gt_upd:
            print(f"\n  ── Ground Truth Memory Points (UPDATE, {len(gt_upd)}) ──")
            upd_map = {r["memory_content"]: r for r in ev["update"]}
            for m in gt_upd:
                rec    = upd_map.get(m["memory_content"])
                result = rec.get("memory_update_type") if rec else None
                label  = UPDATE_SYM.get(result, "? (not evaluated)")
                print(f"  [{label}] {m['memory_content']}")
                if m.get("original_memories"):
                    for orig in m["original_memories"]:
                        print(f"     ↑ was: {orig}")

        # --- Interference ---
        gt_interf = [m for m in session.get("memory_points", []) if m.get("memory_source") == "interference"]
        if gt_interf:
            print(f"\n  ── Interference (planted false info, {len(gt_interf)}) ──")
            for m in gt_interf:
                print(f"  [PLANT] {m['memory_content']}")

        # --- Extracted memories ---
        extracted = session.get("extracted_memories", [])
        print(f"\n  ── Extracted Memories ({len(extracted)}) ──")
        for i, mem in enumerate(extracted, 1):
            print(f"  [{i:02d}] {mem}")

        # --- QA ---
        questions = session.get("questions", [])
        if questions:
            print(f"\n  ── QA Results ({len(questions)} questions) ──")
            qa_map = {}
            for r in ev["qa"]:
                qa_map[r["question"]] = r
            for i, q in enumerate(questions, 1):
                rec    = qa_map.get(q["question"])
                result = rec.get("result_type") if rec else None
                sym    = QA_SYM.get(result, "?")
                print(f"  [{sym}][{q.get('question_type','?')}][{q.get('difficulty','?')}]")
                print(f"       Q: {q['question']}")
                print(f"  Expected: {q['answer']}")
                print(f"  Response: {q.get('system_response', '(not answered)')}")
                print()

        print_divider()


def load_results(version: str, frame: str = "mem0_oss"):
    result_file = f"./results/{frame}-{version}/{frame}_eval_results.jsonl"
    if not os.path.exists(result_file):
        print(f"No results found: {result_file}")
        return []
    users = []
    with open(result_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                users.append(json.loads(line))
    return users


def load_session_logs(version: str, frame: str = "mem0_oss"):
    log_dir = f"./logs/{frame}-{version}"
    logs = {}
    if not os.path.exists(log_dir):
        return logs
    for fname in os.listdir(log_dir):
        if fname.endswith("_sessions.jsonl"):
            uuid = fname.replace("_sessions.jsonl", "")
            sessions = []
            with open(os.path.join(log_dir, fname), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sessions.append(json.loads(line))
            logs[uuid] = sessions
    return logs


def print_divider(char="=", width=70):
    print(char * width)


def print_session_summary(users, session_logs):
    """Print a one-line-per-session summary table."""
    print_divider()
    print(f"{'User':<20} {'Sess':>4} {'Turns':>5} {'Extracted':>9} {'QA':>4} {'Time(s)':>8}")
    print_divider("-")
    for user in users:
        uuid = user["uuid"]
        name = user.get("user_name", uuid[:8])
        slogs = session_logs.get(uuid, [])
        for slog in slogs:
            sid    = slog["session_id"]
            n_mem  = slog["extracted_memory_count"]
            n_qa   = slog["question_count"]
            t_s    = slog["session_elapsed_ms"] / 1000
            # Count dialogue turns from full results
            sessions = user.get("sessions", [])
            if sid < len(sessions):
                n_turns = len(sessions[sid].get("dialogue", []))
            else:
                n_turns = "?"
            print(f"{name:<20} {sid:>4} {n_turns:>5} {n_mem:>9} {n_qa:>4} {t_s:>8.1f}")
    print_divider()


def print_session_detail(user, sid: int, session_log=None):
    """Print full detail: dialogue → memories → QA."""
    name = user.get("user_name", user["uuid"][:8])
    sessions = user.get("sessions", [])
    if sid >= len(sessions):
        print(f"Session {sid} not found (only {len(sessions)} sessions)")
        return

    s = sessions[sid]
    print_divider()
    print(f"  USER: {name}  |  SESSION: {sid}")
    if session_log:
        print(f"  Time: {session_log['session_elapsed_ms']/1000:.1f}s  |  "
              f"Extracted: {session_log['extracted_memory_count']} memories  |  "
              f"QA: {session_log['question_count']} questions")
    print_divider()

    # --- Dialogue ---
    dialogue = s.get("dialogue", [])
    if dialogue:
        print(f"\n【 DIALOGUE  ({len(dialogue)} turns) 】")
        print_divider("-")
        for turn in dialogue:
            role   = turn["role"].upper()
            ts     = turn.get("timestamp", "")
            content = turn["content"]
            prefix = "  👤 USER  " if turn["role"] == "user" else "  🤖 ASST  "
            print(f"{prefix}[{ts}]")
            # Wrap long content
            for chunk in [content[i:i+100] for i in range(0, len(content), 100)]:
                print(f"      {chunk}")
            print()

    # --- Extracted Memories ---
    extracted = s.get("extracted_memories", [])
    print(f"\n【 EXTRACTED MEMORIES ({len(extracted)} total) 】")
    print_divider("-")
    if extracted:
        for i, mem in enumerate(extracted, 1):
            print(f"  [{i:02d}] {mem}")
    else:
        print("  (none extracted)")

    # --- Ground Truth Memory Points ---
    gts = [m for m in s.get("memory_points", []) if m.get("memory_source") != "interference"]
    interference = [m for m in s.get("memory_points", []) if m.get("memory_source") == "interference"]
    print(f"\n【 GROUND TRUTH MEMORY POINTS ({len(gts)} real + {len(interference)} interference) 】")
    print_divider("-")
    for m in gts:
        flag = "↺ UPDATE" if m.get("is_update") == "True" else "  NEW   "
        src  = m.get("memory_source", "?")
        print(f"  [{flag}][{src}] {m['memory_content']}")
    if interference:
        print(f"\n  ── Interference (planted false memories in dialogue) ──")
        for m in interference:
            print(f"  [INTERFERE] {m['memory_content']}")

    # --- QA ---
    questions = s.get("questions", [])
    if questions:
        print(f"\n【 QA  ({len(questions)} questions) 】")
        print_divider("-")
        for i, q in enumerate(questions, 1):
            qt    = q.get("question_type", "")
            diff  = q.get("difficulty", "")
            print(f"  [{i}] [{qt}][{diff}]")
            print(f"       Q: {q['question']}")
            print(f"  Expected: {q['answer']}")
            print(f"  Response: {q.get('system_response', '(not answered)')}")
            print()

    print_divider()


def print_memories_only(users, session_logs):
    """Print just extracted memories per session."""
    for user in users:
        name = user.get("user_name", user["uuid"][:8])
        print_divider()
        print(f"USER: {name}")
        print_divider("-")
        for s in user.get("sessions", []):
            if s.get("is_generated_qa_session"):
                continue
            sid = user["sessions"].index(s)
            extracted = s.get("extracted_memories", [])
            print(f"\n  Session {sid} ({len(extracted)} memories extracted):")
            for m in extracted:
                print(f"    • {m}")
        print()


def print_kg_structure(version: str, frame: str = "graphiti", session_filter: int = None):
    """Show the actual KG entity-relation structure stored in Kuzu."""
    import glob
    kuzu_root = f"./results/{frame}-{version}/kuzu_data"
    if not os.path.exists(kuzu_root):
        print(f"No Kuzu data found at {kuzu_root}")
        print("  (KG view only available for Graphiti backend)")
        return

    try:
        import kuzu
    except ImportError:
        print("pip install graphiti-core[kuzu]")
        return

    for uuid_dir in sorted(os.listdir(kuzu_root)):
        db_path = os.path.join(kuzu_root, uuid_dir)
        if not os.path.isdir(db_path) and not os.path.exists(db_path):
            continue
        try:
            db   = kuzu.Database(db_path)
            conn = kuzu.Connection(db)

            # --- Resolve session filter to episode UUID ---
            episode_uuid_filter = None
            episode_label = "ALL sessions"
            if session_filter is not None:
                # Find episode UUID for this session index
                # Episode name format: {user_name}_s{sid}
                r = conn.execute(
                    "MATCH (e:Episodic) RETURN e.uuid, e.name ORDER BY e.created_at"
                )
                episodes = []
                while r.has_next():
                    episodes.append(r.get_next())
                if session_filter < len(episodes):
                    episode_uuid_filter = episodes[session_filter][0]
                    episode_label = f"Session {session_filter} ({episodes[session_filter][1]})"
                else:
                    print(f"  Session {session_filter} not found (only {len(episodes)} sessions)")
                    conn.close()
                    continue

            print_divider()
            print(f"  KG: {uuid_dir[:8]}...  |  {episode_label}")
            print_divider()

            # --- Relationships (edges filtered by episode) ---
            if episode_uuid_filter:
                rel_query = f"""
                    MATCH (src:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(tgt:Entity)
                    WHERE list_contains(e.episodes, '{episode_uuid_filter}')
                    RETURN src.name, src.summary, e.name, tgt.name, tgt.summary, e.fact, e.valid_at
                    ORDER BY e.valid_at
                """
            else:
                rel_query = """
                    MATCH (src:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(tgt:Entity)
                    RETURN src.name, src.summary, e.name, tgt.name, tgt.summary, e.fact, e.valid_at
                    ORDER BY e.valid_at
                """

            r = conn.execute(rel_query)
            rel_rows = []
            while r.has_next():
                rel_rows.append(r.get_next())

            # Collect unique entities from these edges
            seen_entities = {}
            for src_name, src_sum, _, tgt_name, tgt_sum, _, _ in rel_rows:
                if src_name not in seen_entities:
                    seen_entities[src_name] = src_sum
                if tgt_name not in seen_entities:
                    seen_entities[tgt_name] = tgt_sum

            print(f"\n  ENTITY NODES ({len(seen_entities)} entities involved)")
            print_divider("-")
            for name, summary in sorted(seen_entities.items()):
                print(f"  [{name}]")
                if summary:
                    print(f"    {(summary or '')[:100]}")

            print(f"\n  RELATIONSHIPS ({len(rel_rows)} edges)")
            print_divider("-")
            for src, _, rel, tgt, _, fact, valid_at in rel_rows:
                print(f"  ({src})  --[{rel or 'RELATES_TO'}]-->  ({tgt})")
                print(f"    fact: {fact}")
                if valid_at:
                    print(f"    at  : {valid_at}")
                print()

            # --- Updates: expired edges (old facts replaced by new ones) ---
            r_exp = conn.execute("""
                MATCH (src:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(tgt:Entity)
                WHERE e.expired_at IS NOT NULL
                RETURN src.name, e.name, tgt.name, e.fact, e.valid_at, e.expired_at
                ORDER BY e.expired_at
            """)
            exp_rows = []
            while r_exp.has_next():
                exp_rows.append(r_exp.get_next())

            if exp_rows:
                print(f"\n  UPDATES — EXPIRED EDGES ({len(exp_rows)} facts replaced)")
                print_divider("-")
                for src, rel, tgt, fact, valid_at, expired_at in exp_rows:
                    print(f"  ❌ WAS:  ({src}) --[{rel}]--> ({tgt})")
                    print(f"         fact: {fact}")
                    print(f"         valid {valid_at} → expired {expired_at}")
                    print()
            else:
                print(f"\n  UPDATES — no expired edges (no contradictions detected)")

            conn.close()
        except Exception as e:
            print(f"  Error reading KG for {uuid_dir}: {e}")

    print_divider()


def print_qa_context(users, sid: int):
    """Show the KG context (retrieved facts) for each QA in a session."""
    for user in users:
        name = user.get("user_name", user["uuid"][:8])
        sessions = [s for s in user.get("sessions", []) if not s.get("is_generated_qa_session")]
        if sid >= len(sessions):
            continue
        session = sessions[sid]
        questions = session.get("questions", [])
        if not questions:
            print(f"No QA in session {sid}")
            continue

        print_divider()
        print(f"  USER: {name}  |  SESSION: {sid}  —  QA Retrieved Context")
        print_divider()

        for i, q in enumerate(questions, 1):
            print(f"\n  [{i}] [{q.get('question_type','?')}] Q: {q['question']}")
            print(f"  Expected : {q['answer']}")
            print(f"  Response : {q.get('system_response','(none)')}")

            context = q.get("context", "")
            if context:
                print(f"\n  ── Retrieved Context ──")
                for line in context.strip().splitlines():
                    if line.strip():
                        print(f"  {line}")
            print()
        print_divider()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version",  required=True,
                        help="Experiment version, e.g. smoke_graphiti")
    parser.add_argument("--backend",  default="mem0", choices=["mem0", "graphiti"],
                        help="Memory backend (default: mem0)")
    parser.add_argument("--session",  default=None,
                        help="Session ID for full detail (int or 'all')")
    parser.add_argument("--memories-only",   action="store_true",
                        help="Show only extracted memories per session")
    parser.add_argument("--eval-per-session", action="store_true",
                        help="Per-session recall / update / QA scores")
    parser.add_argument("--eval-session", default=None, type=int,
                        help="Detailed eval for a specific session ID")
    parser.add_argument("--show-context", action="store_true",
                        help="Show the KG/memory context retrieved for each QA")
    parser.add_argument("--show-kg", action="store_true",
                        help="Show full KG structure (entities + relations) — Graphiti only")
    args = parser.parse_args()

    frame = FRAME_MAP[args.backend]
    users = load_results(args.version, frame)
    if not users:
        print(f"No results found for version '{args.version}' backend '{args.backend}'")
        print(f"Expected: ./results/{frame}-{args.version}/{frame}_eval_results.jsonl")
        return
    session_logs = load_session_logs(args.version, frame)
    eval_detail  = load_eval_detail(args.version, frame)

    if args.show_kg:
        sid_filter = int(args.session) if args.session and args.session != "all" else None
        print_kg_structure(args.version, frame, session_filter=sid_filter)
        return

    if args.memories_only:
        print_memories_only(users, session_logs)
        return

    if args.eval_per_session:
        print_eval_per_session(users, eval_detail)
        return

    if args.eval_session is not None:
        print_eval_session_detail(users, eval_detail, args.eval_session)
        return

    if args.show_context:
        sid = int(args.session) if args.session and args.session != "all" else 0
        print_qa_context(users, sid)
        return

    if args.session is None:
        # Summary table
        print_session_summary(users, session_logs)
        print(f"\nBackend: {args.backend}  |  Version: {args.version}")
        print("\nTips:")
        print("  --session 0                        # dialogue + memories + QA")
        print("  --session 0 --show-context         # also show retrieved KG facts per QA")
        print("  --eval-per-session                 # recall/update/QA scores per session")
        print("  --eval-session 0                   # detailed eval for session 0")
        print("  --memories-only                    # extracted memories only")
        return

    # Full detail view
    for user in users:
        uuid = user["uuid"]
        slogs = session_logs.get(uuid, [])
        if args.session == "all":
            for sid in range(len(user.get("sessions", []))):
                slog = next((s for s in slogs if s["session_id"] == sid), None)
                print_session_detail(user, sid, slog)
        else:
            sid  = int(args.session)
            slog = next((s for s in slogs if s["session_id"] == sid), None)
            print_session_detail(user, sid, slog)


if __name__ == "__main__":
    main()
