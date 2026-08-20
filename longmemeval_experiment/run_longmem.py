"""
Entry point for LongMemEval-S experiments (independent from other datasets).

Usage:
  python run_longmem.py --backend mem0 --smoke                    # 2 questions, 5 sessions each
  python run_longmem.py --backend mem0 --version v1 --max-questions 20
  python run_longmem.py --backend mem0 --version v1 --eval-only
"""

import argparse
import os

DATA_DEFAULT = "./data/longmemeval_s.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mem0", choices=["mem0", "amem", "rag", "letta", "memos", "zep", "structmem"])
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--version", default="default")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="2 questions, 5 sessions each")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    frame = args.backend
    result_dir = f"./results/{frame}-{args.version}/"

    if not args.eval_only:
        if args.backend == "mem0":
            from eval_mem0_longmem import run_extraction
        elif args.backend == "amem":
            from eval_amem_longmem import run_extraction
        elif args.backend == "rag":
            from eval_rag_longmem import run_extraction
        elif args.backend == "letta":
            from eval_letta_longmem import run_extraction
        elif args.backend == "memos":
            from eval_memos_longmem import run_extraction
        elif args.backend == "zep":
            from eval_zep_longmem import run_extraction
        elif args.backend == "structmem":
            from eval_structmem_longmem import run_extraction
        run_extraction(
            data_path=args.data, version=args.version, top_k=args.top_k,
            smoke=args.smoke, max_questions=args.max_questions, llm_model=args.llm_model,
        )

    if not args.skip_eval:
        out = os.path.join(result_dir, f"{frame}_lme_results.jsonl")
        if not os.path.exists(out):
            print(f"❌ results not found: {out} (run without --eval-only first)")
            return
        from evaluation_longmem import run_evaluation
        run_evaluation(version=args.version, result_dir=result_dir, frame=frame)
        try:
            from update_excel_longmem import build_excel, scan_runs
            build_excel(scan_runs())
        except Exception as e:
            print(f"⚠️  Excel update skipped: {e}")


if __name__ == "__main__":
    main()
