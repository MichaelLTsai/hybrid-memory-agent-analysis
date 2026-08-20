"""
Entry point for LoCoMo experiments (independent from halumem_experiment).

Usage:
  python run_locomo.py --backend mem0 --smoke --skip-eval          # smoke, extraction only
  python run_locomo.py --backend mem0 --version v1_gemma4          # full run + eval + excel
  python run_locomo.py --backend mem0 --version v1 --max-convs 2   # first 2 conversations
  python run_locomo.py --backend mem0 --version v1 --eval-only     # re-eval existing results
"""

import argparse
import json
import os

DATA_DEFAULT = "./data/locomo10.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mem0", choices=["mem0", "rag", "graphiti", "amem", "letta", "structmem"])
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--version", default="default")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="1 conversation")
    ap.add_argument("--max-convs", type=int, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--llm-model", default=None)
    # Extraction-stage evaluation (integrity / accuracy / F1). Each observation and
    # each system memory costs one LLM call, which adds up, so --max-obs and
    # --max-mem sample evenly to control cost.
    ap.add_argument("--skip-extraction-eval", action="store_true",
                    help="skip the extraction-stage metrics (integrity / accuracy / F1)")
    ap.add_argument("--max-obs", type=int, default=None,
                    help="max observations judged per conversation (evenly sampled)")
    ap.add_argument("--max-mem", type=int, default=None,
                    help="max system memories judged per conversation (evenly sampled)")
    args = ap.parse_args()

    frame = args.backend
    result_dir = f"./results/{frame}-{args.version}/"

    # ── Extraction ──────────────────────────────────────────────
    if not args.eval_only:
        if args.backend == "mem0":
            from eval_mem0_locomo import run_extraction
        elif args.backend == "rag":
            from eval_rag_locomo import run_extraction
        elif args.backend == "graphiti":
            from eval_graphiti_locomo import run_extraction
        elif args.backend == "amem":
            from eval_amem_locomo import run_extraction
        elif args.backend == "letta":
            from eval_letta_locomo import run_extraction
        elif args.backend == "structmem":
            from eval_structmem_locomo import run_extraction
        run_extraction(
            data_path=args.data, version=args.version, top_k=args.top_k,
            smoke=args.smoke, max_convs=args.max_convs, llm_model=args.llm_model,
        )

    # ── Evaluation ──────────────────────────────────────────────
    if not args.skip_eval:
        out = os.path.join(result_dir, f"{frame}_locomo_results.jsonl")
        if not os.path.exists(out):
            print(f"❌ results not found: {out} (run without --eval-only first)")
            return
        from evaluation_locomo import run_evaluation
        run_evaluation(version=args.version, result_dir=result_dir, frame=frame,
                       extraction=not args.skip_extraction_eval,
                       max_obs=args.max_obs, max_mem=args.max_mem)

        # Auto-update the LoCoMo experiment_results.xlsx
        try:
            from update_excel_locomo import build_excel, scan_runs
            build_excel(scan_runs())
        except Exception as e:
            print(f"⚠️  Excel update skipped: {e}")


if __name__ == "__main__":
    main()
