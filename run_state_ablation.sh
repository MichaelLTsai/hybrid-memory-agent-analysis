#!/usr/bin/env bash
# Drive the M1 / M3 / M4 state ablation across its five arms.
#
#   ./run_state_ablation.sh lme       # LongMemEval, knowledge-update x5
#   ./run_state_ablation.sh halumem   # HaluMem, 1 user
#   ./run_state_ablation.sh both      # both, every arm in parallel
#
# Arms run CONCURRENTLY, one process each. The NCHC endpoint was measured at
# roughly linear scaling up to ~12 concurrent requests and collapsing at 16
# (154 out-tok/s at 12, 70 at 16), so ten ingest lines plus a small audit burst
# sits inside the usable band. STRUCTMEM_M1_AUDIT_WORKERS caps the burst the
# M1 audit phase adds on top of each line.
#
# Every arm gets its own --version, hence its own results directory, log
# directory, trace directory and Qdrant collections. Nothing writable is shared.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
# Arms to run. E0_baseline is the control; override with ARMS="..." to skip it,
# but note that dropping E0 means there is no in-batch reference for "how much
# better than original StructMem" -- only E1..E4 remain mutually comparable.
IFS=' ' read -r -a ARMS <<< "${ARMS:-E0_baseline E1_m1 E2_m1_m3 E3_m1_m4 E4_full}"

export STRUCTMEM_M1_AUDIT_WORKERS="${STRUCTMEM_M1_AUDIT_WORKERS:-2}"

# LongMemEval: knowledge-update only, 5 questions. Every other type quota 0.
# --max-questions is what activates the stratified sampler; the quota then
# decides the selection. Without the flag all 500 questions would run.
LME_QUOTA="knowledge-update=5,multi-session=0,temporal-reasoning=0,single-session-user=0,single-session-assistant=0,single-session-preference=0"

pids=()

start_lme () {
  for arm in "${ARMS[@]}"; do
    ver="ablate_$(echo "$arm" | tr '[:upper:]' '[:lower:]')"
    ( cd "$ROOT/longmemeval_experiment" && \
      STRUCTMEM_EXPERIMENT="$arm" LME_TYPE_QUOTA="$LME_QUOTA" \
      "$PY" run_longmem.py --backend structmem --version "$ver" --max-questions 5 \
        > "logs_${ver}.out" 2>&1
      echo "LME ${arm} exit=$?" ) &
    pids+=($!)
    echo "=== LME ${arm} started (pid $!) ==="
    sleep 3   # stagger model loading so five processes do not thrash at once
  done
}

start_halumem () {
  for arm in "${ARMS[@]}"; do
    ver="ablate_$(echo "$arm" | tr '[:upper:]' '[:lower:]')"
    ( cd "$ROOT/halumem_experiment" && \
      STRUCTMEM_EXPERIMENT="$arm" \
      "$PY" run.py --backend structmem --version "$ver" --max-users 1 \
        > "logs_${ver}.out" 2>&1
      echo "HaluMem ${arm} exit=$?" ) &
    pids+=($!)
    echo "=== HaluMem ${arm} started (pid $!) ==="
    sleep 3
  done
}

case "${1:-both}" in
  lme)     start_lme ;;
  halumem) start_halumem ;;
  both)    start_lme; start_halumem ;;
  *) echo "usage: $0 [lme|halumem|both]" >&2; exit 2 ;;
esac

echo "=== ${#pids[@]} lines running, waiting ==="
wait
echo "=== ALL ARMS FINISHED ==="
