#!/usr/bin/env bash
# LoCoMo arm of the M1/M3/M4 ablation: conv-26, 199 questions, four arms in
# parallel. This is the benchmark that carries the temporal-reasoning guardrail
# (StructMem 0.7027, first of five), which the other two benchmarks cannot test.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
export STRUCTMEM_M1_AUDIT_WORKERS="${STRUCTMEM_M1_AUDIT_WORKERS:-2}"
cd "$ROOT/locomo_experiment" || exit 1
for arm in E1_m1 E2_m1_m3 E3_m1_m4 E4_full; do
  ver="ablate_$(echo "$arm" | tr '[:upper:]' '[:lower:]')"
  ( STRUCTMEM_EXPERIMENT="$arm" "$PY" run_locomo.py --backend structmem \
      --version "$ver" --max-convs 1 > "logs_${ver}.out" 2>&1
    echo "LoCoMo ${arm} exit=$?" ) &
  echo "=== LoCoMo ${arm} started (pid $!) ==="
  sleep 3
done
wait
echo "=== LOCOMO ARMS FINISHED ==="
