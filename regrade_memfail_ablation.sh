#!/usr/bin/env bash
# Re-grade the MemFail ablation traces with a judge that can actually follow the
# schema.
#
#   ./regrade_memfail_ablation.sh
#
# The sweep graded with gemma-4-E4B-it. response_format is advisory on the NCHC
# proxy, and E4B ignores the json_schema and answers in prose ("Yes"), so every
# judge call raised "Expecting value: line 1 column 1" and every subset scored
# 0/5 with error_type=None. gemma-4-31B-it honours the schema.
#
# Only the grading is redone. The evaluate step (memory construction, retrieval,
# answering) is untouched and its traces are reused, so this costs judge calls
# only.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/memfail_experiment"
cd "$ROOT" || exit 1

PY="$HOME/structmem_env/bin/python"
set -a; [ -f ../halumem_experiment/.env ] && . ../halumem_experiment/.env; set +a
export OPENAI_API_KEY="${NCHC_API_KEY:-}"
export OPENAI_BASE_URL="${NCHC_BASE_URL:-}"
JUDGE_MODEL="${MEMFAIL_JUDGE_MODEL:-gemma-4-31B-it}"
WORKERS="${MEMFAIL_ANALYSIS_WORKERS:-2}"

# macOS ships bash 3.2, which has no associative arrays; a case is portable.
script_dir_for () {
  case "$1" in
    coexisting_facts) echo coexisting_facts ;;
    conditional_easy|conditional_hard) echo conditional_facts ;;
    long_hop) echo long_hop ;;
    persona) echo custom_persona_retrieval ;;
    *) echo "" ;;
  esac
}

for arm_dir in results_ablate_*; do
  [ -d "$arm_dir" ] || continue
  for label in coexisting_facts conditional_easy conditional_hard long_hop persona; do
    out="${arm_dir}/${label}"
    [ -d "$out" ] || continue
    traces=$(ls -t "${out}/structmem/graded_traces_"*.json 2>/dev/null | head -1)
    [ -n "$traces" ] || traces=$(ls -t "${out}/structmem/traces_"*.json 2>/dev/null | head -1)
    if [ -z "$traces" ]; then
      echo "!! ${out}: no traces"; continue
    fi
    echo "=== regrading ${out} ==="
    # Old analysis files are left in place; the new one carries a later
    # timestamp and every reader picks the latest.
    "$PY" "playground/$(script_dir_for "$label")/analyze_errors.py" \
        --traces "$traces" \
        --output-dir "${out}/analysis" \
        --model "$JUDGE_MODEL" \
        --api-key "$OPENAI_API_KEY" \
        --workers "$WORKERS" \
        > "logs_regrade_$(basename "$arm_dir")_${label}.out" 2>&1
    echo "    exit=$?"
  done
done
echo "=== REGRADE FINISHED ==="
