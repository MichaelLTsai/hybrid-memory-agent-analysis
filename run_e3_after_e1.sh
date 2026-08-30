#!/usr/bin/env bash
# Repair LongMemEval E1, then run E3. One endpoint line throughout.
#
# Busy-detection note: the previous version waited on
#   pgrep -f "run_ablation_31b.sh|ensure_halumem.sh"
# which also matched the monitoring shell, because that shell's command line
# quotes those very script names. The chain then blocked for seven hours on a
# process that never exits. Patterns here are anchored to the interpreter, so a
# shell that merely mentions a script name cannot satisfy them.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
SELF=$$

busy () {
  # Real python workers are the only things that consume the endpoint.
  pgrep -f "structmem_env/bin/python .*--backend structmem" >/dev/null 2>&1 && return 0
  # Sibling drivers, matched as "bash <path>/<script>.sh", never as a mention.
  pgrep -f "^bash .*/(run_ablation_31b|ensure_halumem)\.sh$" >/dev/null 2>&1 && return 0
  return 1
}

echo "[chain] $(date '+%F %T') waiting for the endpoint line to be free"
n=0
while busy; do sleep 30; n=$((n+1)); [ $((n % 20)) = 0 ] && echo "[chain]   still busy after $((n/2)) min"; done
echo "[chain] $(date '+%F %T') line free"

LME_E1="$ROOT/longmemeval_experiment/results/structmem-ablate_e1_m1_31b/tmp/6a1eabeb.json"
if [ ! -f "$LME_E1" ]; then
  echo "[chain] $(date '+%F %T') repairing LongMemEval E1 (6a1eabeb)"
  ( cd "$ROOT/longmemeval_experiment" && \
    STRUCTMEM_EXPERIMENT="E1_m1" \
    LME_TYPE_QUOTA="knowledge-update=6,multi-session=3,temporal-reasoning=3,single-session-user=3,single-session-assistant=3,single-session-preference=3" \
    LME_TYPE_ORDER="knowledge-update,temporal-reasoning,single-session-assistant" \
    LME_QDRANT_BASE="$ROOT/longmemeval_experiment/qdrant_structmem_31b" \
    "$PY" run_longmem.py --backend structmem --version ablate_e1_m1_31b \
      --max-questions 21 --llm-model gemma-4-31B-it \
      > logs_ablate_e1_m1_31b_repair.out 2>&1 )
  echo "[chain]   repair exit=$?"
  ( cd "$ROOT/longmemeval_experiment" && "$PY" probe_longmem.py \
      --run structmem-ablate_e1_m1_31b > probe_ablate_e1_m1_31b.out 2>&1 )
  echo "[chain]   re-probe exit=$?"
fi
echo "[chain]   E1 LME now $(ls "$ROOT/longmemeval_experiment/results/structmem-ablate_e1_m1_31b/tmp/"*.json 2>/dev/null | wc -l | tr -d ' ')/21"

echo "[chain] $(date '+%F %T') E3 HaluMem via ensure_halumem"
VER=ablate_e3_m1_m4_31b SCOPE="--skip-users 2 --max-users 1" \
  "$ROOT/ensure_halumem.sh" >> "$ROOT/ensure_halumem_e3.out" 2>&1
echo "[chain]   ensure_halumem exit=$?"

echo "[chain] $(date '+%F %T') E3 LongMemEval + LoCoMo via driver"
PHASES="E3_m1_m4" "$ROOT/run_ablation_31b.sh" >> "$ROOT/ablation_e3.out" 2>&1
echo "[chain]   driver exit=$?"
echo "[chain] ALL DONE $(date '+%F %T')"
