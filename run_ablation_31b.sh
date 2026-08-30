#!/usr/bin/env bash
# StructMem state ablation on gemma-4-31B-it, phased so E0 lands first.
#
#   ./run_ablation_31b.sh              # E0 phase, then E1, then E3
#   PHASES="E0_baseline" ./run_ablation_31b.sh
#   SKIP_LME=1 ./run_ablation_31b.sh   # LME already running elsewhere
#
# Why phased by ARM rather than by benchmark
# ------------------------------------------
# The ask is "get E0 out as fast as possible". The endpoint is decode-saturated,
# so total wall time is fixed no matter how the work is ordered; what ordering
# controls is WHICH results exist early. Running all of E0's benchmarks first
# means the complete E0 row (LongMemEval + HaluMem + LoCoMo) is usable after
# roughly a third of the batch, instead of three arms all finishing at the end.
# Inside a phase the benchmarks run SEQUENTIALLY. Three concurrent lines was
# fine on E4B but collapsed the endpoint on 31B (998x HTTP 503), so one line at
# a time is the only setting that completes.
#
# Model per benchmark, matched to what the other four architectures used
# ---------------------------------------------------------------------
#   LongMemEval  extraction 31B
#   HaluMem      extraction 31B
#   LoCoMo       extraction 31B, judge E4B (OPENAI_MODEL from .env)
#   MemFail      NOT RUN. All five architectures ingested MemFail with
#                gemma-4-E4B-it, and the existing E0/E1/E3 MemFail runs already
#                use E4B, so they are correct as they stand. Re-running them on
#                31B would put StructMem on a different model from every
#                comparator, which is the exact defect this batch exists to fix.
#
# HaluMem user scope
# ------------------
#   E0       users #3 and #4  -- the slice Mem0 v1/v2, A-MEM and Letta ran, so
#                               E0 is comparable across architectures.
#   E1, E3   user #3 only     -- half the ingest, and inside E0's user set. Note
#                               the residual asymmetry: the E1/E3-vs-E0 delta on
#                               HaluMem compares one user against two, so read
#                               it as indicative, not exact.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
SUFFIX="${SUFFIX:-31b}"

IFS=' ' read -r -a PHASES <<< "${PHASES:-E0_baseline E1_m1 E3_m1_m4}"

# Never inherited: eval_structmem.py defaults SM_LLM to gemma-4-E4B-it.
EXTRACT_LLM="${EXTRACT_LLM:-gemma-4-31B-it}"
export STRUCTMEM_M1_AUDIT_WORKERS="${STRUCTMEM_M1_AUDIT_WORKERS:-2}"
export LME_TYPE_ORDER="${LME_TYPE_ORDER:-knowledge-update,temporal-reasoning,single-session-assistant}"

LME_QUOTA="knowledge-update=6,multi-session=3,temporal-reasoning=3"
LME_QUOTA="$LME_QUOTA,single-session-user=3,single-session-assistant=3"
LME_QUOTA="$LME_QUOTA,single-session-preference=3"

[ -x "$PY" ] || { echo "!! interpreter not found: $PY" >&2; exit 1; }

ver_for () { echo "ablate_$(echo "$1" | tr '[:upper:]' '[:lower:]')_${SUFFIX}"; }

# HaluMem: E0 gets both comparison users, the ablation arms get the first of them.
halumem_scope () {
  case "$1" in
    E0_baseline) echo "--skip-users 2 --max-users 2" ;;
    *)           echo "--skip-users 2 --max-users 1" ;;
  esac
}

# Benchmarks run ONE AT A TIME. Three concurrent lines was survivable on
# gemma-4-E4B-it but is not on 31B: the first attempt at three-way concurrency
# drew 998 HTTP 503s plus 500 "系統繁忙" from the NCHC endpoint, failed 5 of 21
# LongMemEval questions, and killed BOTH HaluMem users outright, leaving
# memory_num=0. The endpoint, not the client, is the constraint, and it is
# already saturated by a single 31B line.
#
# A benchmark that already has scores is skipped, so a repair run only redoes
# what is actually missing.
has_scores () {
  case "$1" in
    lme)     ls "$ROOT/longmemeval_experiment/results/structmem-$2/"*_lme_scores.json >/dev/null 2>&1 ;;
    halumem) [ -s "$ROOT/halumem_experiment/results/structmem-$2/structmem_scores.json" ] &&
             [ "$("$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('memory_integrity',{}).get('memory_num',0))" \
                  "$ROOT/halumem_experiment/results/structmem-$2/structmem_scores.json" 2>/dev/null)" != "0" ] ;;
    locomo)  [ -s "$ROOT/locomo_experiment/results/structmem-$2/structmem_locomo_scores.json" ] ;;
  esac
}

run_phase () {
  local arm="$1" ver; ver="$(ver_for "$arm")"
  echo "=== PHASE ${arm} starting $(date '+%F %T') ==="

  # HaluMem first: it is the longest and the one that was destroyed, and the
  # ask is to get a complete E0 as early as possible.
  if has_scores halumem "$ver"; then
    echo "    HaluMem ${arm} already has scores, skipping"
  else
    echo "    HaluMem ${arm} starting  scope: $(halumem_scope "$arm")"
    ( cd "$ROOT/halumem_experiment" && \
      STRUCTMEM_EXPERIMENT="$arm" \
      "$PY" run.py --backend structmem --version "$ver" \
        $(halumem_scope "$arm") --llm-model "$EXTRACT_LLM" \
        > "logs_${ver}.out" 2>&1 )
    echo "    HaluMem ${arm} exit=$?  $(date '+%H:%M')"
  fi

  if [ "${SKIP_LME:-0}" != "1" ]; then
    echo "    LME ${arm} starting"
    ( cd "$ROOT/longmemeval_experiment" && \
      STRUCTMEM_EXPERIMENT="$arm" LME_TYPE_QUOTA="$LME_QUOTA" \
      LME_QDRANT_BASE="$ROOT/longmemeval_experiment/qdrant_structmem_${SUFFIX}" \
      "$PY" run_longmem.py --backend structmem --version "$ver" \
        --max-questions 21 --llm-model "$EXTRACT_LLM" \
        > "logs_${ver}.out" 2>&1 )
    echo "    LME ${arm} exit=$?  $(date '+%H:%M')"
  fi

  if has_scores locomo "$ver"; then
    echo "    LoCoMo ${arm} already has scores, skipping"
  else
    echo "    LoCoMo ${arm} starting"
    ( cd "$ROOT/locomo_experiment" && \
      STRUCTMEM_EXPERIMENT="$arm" \
      "$PY" run_locomo.py --backend structmem --version "$ver" \
        --max-convs 1 --llm-model "$EXTRACT_LLM" \
        > "logs_${ver}.out" 2>&1 )
    echo "    LoCoMo ${arm} exit=$?  $(date '+%H:%M')"
  fi
  echo "=== PHASE ${arm} ingest done $(date '+%F %T') ==="

  # Stage attribution, sequential: the probes are judge-bound and would only
  # contend with each other.
  if [ "${SKIP_PROBE:-0}" != "1" ]; then
    [ "${SKIP_LME:-0}" != "1" ] && \
      ( cd "$ROOT/longmemeval_experiment" && "$PY" probe_longmem.py \
          --run "structmem-${ver}" > "probe_${ver}.out" 2>&1; echo "    probe LME exit=$?" )
    ( cd "$ROOT/halumem_experiment" && "$PY" probe_halumem_unified.py \
        --run "structmem-${ver}" > "probe_${ver}.out" 2>&1; echo "    probe HaluMem exit=$?" )
    ( cd "$ROOT/locomo_experiment" && "$PY" probe_locomo.py \
        --run "structmem-${ver}" > "probe_${ver}.out" 2>&1; echo "    probe LoCoMo exit=$?" )
  fi
  echo "=== PHASE ${arm} COMPLETE $(date '+%F %T') ==="
}

echo "=== StructMem ablation on ${EXTRACT_LLM} ==="
echo "    phases : ${PHASES[*]}"
echo "    suffix : ${SUFFIX}"
echo "    MemFail: not run (already correct on E4B, matching all five architectures)"
echo

for arm in "${PHASES[@]}"; do
  run_phase "$arm"
done
echo "=== ALL PHASES FINISHED $(date '+%F %T') ==="
