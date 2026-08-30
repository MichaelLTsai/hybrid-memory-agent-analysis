#!/usr/bin/env bash
# Run HaluMem for one arm, retrying until it produces non-empty scores.
#
#   VER=ablate_e1_m1_31b SCOPE="--skip-users 2 --max-users 1" ./ensure_halumem.sh
#
# Why this exists: halumem_experiment/run.py has no user-level retry. A single
# HTTP 503 that escapes the request-level backoff aborts that user, yet the
# process still exits 0 and writes structmem_scores.json with memory_num=0.
# That silent-empty-success has now cost two runs: E0 died to a SIGSEGV at the
# end of evaluation, E1 to a transient 503 two minutes in, and both looked like
# clean exits to the driver.
#
# Each attempt starts from a clean results dir, because a leftover
# <user>.json.partial makes the next attempt look like it has state it does not.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
VER="${VER:?set VER}"
SCOPE="${SCOPE:---skip-users 2 --max-users 1}"
# Must match EXPERIMENT_PRESETS exactly (E0_baseline / E1_m1 / E3_m1_m4).
case "$VER" in
  *e0_baseline*) STRUCTMEM_ARM="E0_baseline" ;;
  *e1_m1*)       STRUCTMEM_ARM="E1_m1" ;;
  *e3_m1_m4*)    STRUCTMEM_ARM="E3_m1_m4" ;;
  *) echo "!! cannot derive arm from VER=$VER" >&2; exit 2 ;;
esac
MAX="${MAX_ATTEMPTS:-4}"
cd "$ROOT/halumem_experiment" || exit 1

memory_num () {
  "$PY" -c "import json,sys
try: print(json.load(open(sys.argv[1]))['memory_integrity']['memory_num'])
except Exception: print(0)" "results/structmem-$VER/structmem_scores.json" 2>/dev/null
}

# Wait for any other 31B line; one at a time is what this endpoint sustains.
# Anchored to the interpreter: an unanchored pattern also matches a monitoring
# shell whose command line quotes these script names, which deadlocked the E3
# chain for seven hours.
while pgrep -f "structmem_env/bin/python .*--backend structmem" >/dev/null 2>&1; do
  sleep 30
done

for a in $(seq 1 "$MAX"); do
  echo "[ensure] attempt $a/$MAX for $VER  $(date '+%F %T')"
  if [ -d "results/structmem-$VER" ]; then
    mkdir -p failed_503
    mv "results/structmem-$VER" "failed_503/structmem-${VER}_try${a}_$(date +%H%M%S)" 2>/dev/null
  fi
  STRUCTMEM_EXPERIMENT="$STRUCTMEM_ARM" \
    "$PY" run.py --backend structmem --version "$VER" $SCOPE \
      --llm-model "${EXTRACT_LLM:-gemma-4-31B-it}" \
      > "logs_${VER}_try${a}.out" 2>&1
  code=$?
  n=$(memory_num)
  echo "[ensure] attempt $a exit=$code memory_num=$n"
  if [ "${n:-0}" -gt 0 ]; then
    echo "[ensure] SUCCESS, running unified probe"
    "$PY" probe_halumem_unified.py --run "structmem-$VER" > "probe_${VER}.out" 2>&1
    echo "[ensure] probe exit=$? $(date '+%F %T')"
    exit 0
  fi
  # A segfault after a full ingest is recoverable without re-ingesting.
  if [ "$code" = "139" ] && ls "results/structmem-$VER/tmp/"*.json >/dev/null 2>&1; then
    echo "[ensure] segfault with intact ingest, trying --eval-only"
    "$PY" run.py --backend structmem --version "$VER" --eval-only \
      > "logs_${VER}_try${a}_evalonly.out" 2>&1
    if [ "$(memory_num)" -gt 0 ] 2>/dev/null; then
      "$PY" probe_halumem_unified.py --run "structmem-$VER" > "probe_${VER}.out" 2>&1
      echo "[ensure] recovered via eval-only"; exit 0
    fi
  fi
  echo "[ensure] attempt $a produced empty scores, backing off 120s"
  sleep 120
done
echo "[ensure] FAILED after $MAX attempts"; exit 1
