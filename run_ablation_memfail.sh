#!/usr/bin/env bash
# MemFail arm of the M1/M3/M4 state ablation.
#
#   ./run_ablation_memfail.sh                    # all five arms
#   ARMS="E1_m1 E4_full" ./run_ablation_memfail.sh
#
# Why this does not use the subset .sh drivers: those call `uv run`, which
# resolves to memfail_experiment/.venv (Python 3.14). LightMem pins
# python >=3.10,<3.12 and cannot be installed there, and src/__init__.py
# swallows the ImportError so StructMemMemorySystem would silently become None.
# The baseline StructMem run therefore bypassed the .sh entirely, and so does
# this: the evaluate and analyze scripts are called directly under
# ~/structmem_env (3.11), exactly as the recorded run_metadata shows.
#
# Every parameter below is copied from the baseline run's own metadata
# (results_5q_structmem/*/structmem/traces_*.json), so an arm differs from the
# baseline only by the ablation flags. In particular the _5q datasets are
# reused verbatim: no regeneration, same 35 questions.
#
# What this buys: coexisting_facts is the only guardrail in the study against
# M1 over-superseding preferences that can legitimately hold at once.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/memfail_experiment"
cd "$ROOT" || exit 1

PY="$HOME/structmem_env/bin/python"
IFS=' ' read -r -a ARMS <<< "${ARMS:-E0_baseline E1_m1 E2_m1_m3 E3_m1_m4 E4_full}"

export STRUCTMEM_M1_AUDIT_WORKERS="${STRUCTMEM_M1_AUDIT_WORKERS:-2}"

# Matches the baseline run_metadata.
LLM_MODEL="${MEMFAIL_LLM_MODEL:-gemma-4-E4B-it}"
# The grader must be 31B. response_format is advisory on this proxy, and
# gemma-4-E4B-it ignores the json_schema and answers "Yes" in prose, so every
# judge call fails to parse and the whole subset scores 0. 31B honours it.
JUDGE_MODEL="${MEMFAIL_JUDGE_MODEL:-gemma-4-31B-it}"
NUM_MEMORIES=5
SEED=42
ANALYSIS_WORKERS="${MEMFAIL_ANALYSIS_WORKERS:-3}"

# Credentials. memfail_experiment has no .env of its own; the NCHC settings
# live with the other benchmarks. The answering LLM and the grader read
# OPENAI_API_KEY / OPENAI_BASE_URL from the environment (src/llm.py), while
# StructMem's extraction LLM takes them as explicit flags.
set -a; [ -f ../halumem_experiment/.env ] && . ../halumem_experiment/.env; set +a
API_KEY="${NCHC_API_KEY:-${OPENAI_API_KEY:-}}"
BASE_URL="${NCHC_BASE_URL:-${OPENAI_BASE_URL:-}}"
export OPENAI_API_KEY="$API_KEY"
export OPENAI_BASE_URL="$BASE_URL"
if [ -z "$API_KEY" ] || [ -z "$BASE_URL" ]; then
  echo "!! NCHC_API_KEY / NCHC_BASE_URL not found; aborting" >&2; exit 1
fi

# subset | evaluate script dir | script | dataset | subset-specific args
run_subset () {
  local arm="$1" label="$2" dir="$3" script="$4" dataset="$5"; shift 5
  local lower; lower="$(echo "$arm" | tr '[:upper:]' '[:lower:]')"
  local out="results_ablate_${lower}/${label}"
  mkdir -p "$out"

  echo "=== ${arm} / ${label} ==="
  STRUCTMEM_EXPERIMENT="$arm" "$PY" "playground/${dir}/${script}" \
      --dataset "$dataset" \
      --memory structmem \
      --output-dir "$out" \
      --run-dir "$out" \
      --llm-model "$LLM_MODEL" \
      --num-memories "$NUM_MEMORIES" \
      --shared-user-id "${label}_eval_user" \
      --seed "$SEED" \
      --structmem-model "$LLM_MODEL" \
      --structmem-api-key "$API_KEY" \
      --structmem-base-url "$BASE_URL" \
      --structmem-embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
      --structmem-qdrant-path "${out}/structmem_qdrant" \
      --structmem-collection-name "structmem_${lower}" \
      "$@" \
      > "logs_memfail_${lower}_${label}.out" 2>&1
  echo "    evaluate exit=$?"

  # Grade the traces the evaluate step just wrote.
  local traces
  traces=$(ls -t "${out}/structmem/graded_traces_"*.json 2>/dev/null | head -1)
  if [ -z "$traces" ]; then
    traces=$(ls -t "${out}/structmem/traces_"*.json 2>/dev/null | head -1)
  fi
  if [ -n "$traces" ]; then
    "$PY" "playground/${dir}/analyze_errors.py" \
        --traces "$traces" \
        --output-dir "${out}/analysis" \
        --model "$JUDGE_MODEL" \
        --api-key "$API_KEY" \
        --workers "$ANALYSIS_WORKERS" \
        >> "logs_memfail_${lower}_${label}.out" 2>&1
    echo "    analyze  exit=$?  -> ${out}/analysis"
  else
    echo "    !! no traces produced, skipping analysis"
  fi
}

for arm in "${ARMS[@]}"; do
  run_subset "$arm" coexisting_facts coexisting_facts        evaluate_coexisting_facts.py \
             datasets/_5q/coexisting_5.csv        --facts-per-group 10
  run_subset "$arm" conditional_easy conditional_facts       evaluate_conditional_facts.py \
             datasets/_5q/conditional_easy_5.csv  --facts-per-group 1
  run_subset "$arm" conditional_hard conditional_facts       evaluate_conditional_facts.py \
             datasets/_5q/conditional_hard_5.csv  --facts-per-group 1
  run_subset "$arm" long_hop         long_hop                evaluate_long_hop.py \
             datasets/_5q/long_hop_5.csv          --hop-counts 1,2,3 --num-facts 1
  run_subset "$arm" persona          custom_persona_retrieval evaluate_persona_retrieval.py \
             datasets/_5q/persona_5.csv           --facts-per-group 1
  echo "=== ${arm} DONE ==="
done

echo "=== ALL MEMFAIL ARMS FINISHED ==="
