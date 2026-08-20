"""
Entry point for HaluMem experiments.

Usage:
  # Smoke test — mem0 (default)
  python run.py --smoke

  # Smoke test — Graphiti
  python run.py --smoke --backend graphiti

  # Full run — mem0
  python run.py --data ./data/HaluMem-Medium.jsonl --version medium_nchc_gemma4

  # Full run — Graphiti
  python run.py --data ./data/HaluMem-Medium.jsonl --version medium_graphiti --backend graphiti

  # Extraction only (no eval)
  python run.py --smoke --skip-eval --backend graphiti

  # Eval only on existing results
  python run.py --eval-only --version medium_graphiti --backend graphiti

Backend options:
  mem0      → uses eval_mem0_oss.py  (Mem0 OSS + NCHC LLM)
  graphiti  → uses eval_graphiti.py  (Graphiti KG + Kuzu, no server needed)
  amem      → uses eval_amem.py      (A-MEM turn-level + ChromaDB)
  memwave   → uses eval_memwave.py   (MemWeave session-level + Markdown files + SQLite)
"""

import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

FRAME_MAP = {
    "mem0":     "mem0_oss",
    "graphiti": "graphiti",
    "amem":     "amem",
    "memwave":  "memwave",
    "rag":      "rag",
    "letta":    "letta",
    "zep":      "zep",
    "memos":    "memos",
    "structmem": "structmem",
}


def main():
    parser = argparse.ArgumentParser(description="HaluMem experiment runner")
    parser.add_argument("--data",    default="./data/HaluMem-Medium.jsonl")
    parser.add_argument("--version", default="default",
                        help="Experiment version tag (used in output folder names)")
    parser.add_argument("--backend", default="mem0", choices=["mem0", "graphiti", "amem", "memwave", "rag", "letta", "zep", "memos", "structmem"],
                        help="Memory backend: mem0 (default), graphiti, or amem")
    parser.add_argument("--top-k",   type=int, default=20)
    parser.add_argument("--smoke",   action="store_true",
                        help="Run only 1 session from 1 user")
    parser.add_argument("--smoke-sessions", type=int, default=1)
    parser.add_argument("--skip-users", type=int, default=0,
                        help="skip the first N users (use with --max-users to pick a specific user)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit number of users to process (e.g. 2 for a 2-user full run)")
    parser.add_argument("--skip-eval",  action="store_true")
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--eval-workers", type=int, default=4)
    # Mem0-specific overrides
    parser.add_argument("--llm-model", default=None,
                        help="Override MEM0_LLM_MODEL from .env (mem0 backend only)")
    parser.add_argument("--prompt-template", default=None,
                        help="Jinja template file under prompts/ (mem0 backend only)")
    parser.add_argument("--prompt-params", default=None,
                        help='JSON string of template params, e.g. \'{"granularity":"atomic"}\'')
    args = parser.parse_args()

    frame      = FRAME_MAP[args.backend]
    result_dir = f"./results/{frame}-{args.version}/"

    # ── Extraction ──────────────────────────────────────────────
    if not args.eval_only:
        if args.backend == "mem0":
            from eval_mem0_oss import run_extraction
        elif args.backend == "amem":
            from eval_amem import run_extraction
        elif args.backend == "memwave":
            from eval_memwave import run_extraction
        elif args.backend == "rag":
            from eval_rag import run_extraction
        elif args.backend == "letta":
            from eval_letta import run_extraction
        elif args.backend == "zep":
            from eval_zep import run_extraction
        elif args.backend == "memos":
            from eval_memos import run_extraction
        elif args.backend == "structmem":
            from eval_structmem import run_extraction
        else:
            from eval_graphiti import run_extraction

        prompt_params = json.loads(args.prompt_params) if args.prompt_params else None
        run_extraction(
            data_path=args.data,
            version=args.version,
            top_k=args.top_k,
            smoke=args.smoke,
            smoke_session_limit=args.smoke_sessions,
            llm_model=args.llm_model,
            prompt_template=args.prompt_template,
            prompt_params=prompt_params,
            max_users=args.max_users,
            # mem0 / memos support skip_users; don't break the other adapters
            **({"skip_users": args.skip_users} if args.skip_users else {}),
        )

    # ── Evaluation ──────────────────────────────────────────────
    if not args.skip_eval:
        output_file = os.path.join(result_dir, f"{frame}_eval_results.jsonl")
        if not os.path.exists(output_file):
            print(f"❌ Extraction output not found: {output_file}")
            print("   Run without --eval-only first.")
            return

        print(f"\n{'='*60}")
        print(f"  Running HaluMem Evaluation")
        print(f"  Backend : {args.backend}")
        print(f"  Input   : {output_file}")
        print(f"  Frame   : {frame} | Version: {args.version}")
        print(f"{'='*60}\n")

        from evaluation import run_evaluation
        scores = run_evaluation(
            frame=frame,
            version=args.version,
            result_dir=result_dir,
            max_workers=args.eval_workers,
        )

        print("\n=== Final Scores ===")
        print(json.dumps(scores, indent=2, ensure_ascii=False))

        # Auto-update experiment_results.xlsx + .md
        try:
            from update_excel import scan_runs, build_excel, build_markdown
            runs = scan_runs()
            build_excel(runs)
            build_markdown(runs)
        except Exception as e:
            print(f"⚠️  Results file update skipped: {e}")


if __name__ == "__main__":
    main()
