#!/usr/bin/env bash
# Re-run the StructMem state ablation on a wider LongMemEval slice.
#
#   ./run_lme21_ablation.sh            # E0, then E1, then E3; each probed on finish
#   ARMS="E1_m1" ./run_lme21_ablation.sh
#   SKIP_PROBE=1 ./run_lme21_ablation.sh
#
# What changed against run_state_ablation.sh
# ------------------------------------------
# Slice. The earlier ablation ran knowledge-update x5 (221 haystack sessions).
# This one runs knowledge-update x6 plus 3 of every other type: 21 questions,
# 983 sessions, 4.45x the ingest. The sampler takes the first k of each type, so
# the six knowledge-update questions are a strict superset of the old five
# (6a1eabeb 6aeb4375 830ce83f 852ce960 945e3d21, plus d7c942c3) and the two
# batches stay directly comparable on that type.
#
# Arms. E0 / E1 / E3 only. E2 and E4 are dropped because both carry M3, which is
# not in scope for this batch.
#
# Isolation. New --version per arm, so results/, logs/ and traces/ are fresh and
# nothing from the previous batch is touched. LME_QDRANT_BASE moves the
# per-question Qdrant scratch stores to their own root as well: those are keyed
# on question id and arm rather than on --version, so without the override this
# run would rmtree the earlier batch's stores.
#
# Extraction LLM. gemma-4-31B-it, matching every other architecture in the
# study. Set explicitly, never inherited from eval_structmem.py's E4B default.
#
# Order. Arms run sequentially, E0 first, each probed as soon as it finishes.
# The endpoint is decode-saturated, so three concurrent lines only split a fixed
# pie: sequential gets a complete, usable baseline in a third of the time and
# leaves a finished arm rather than three partial ones if anything interrupts.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LME="$ROOT/longmemeval_experiment"
PY="$HOME/structmem_env/bin/python"

IFS=' ' read -r -a ARMS <<< "${ARMS:-E0_baseline E1_m1 E3_m1_m4}"
SUFFIX="${SUFFIX:-lme21}"

# Extraction LLM. eval_structmem.py defaults SM_LLM to gemma-4-E4B-it, which is
# NOT what the rest of the study uses: Mem0 v1/v2, A-MEM and Letta all ingest
# with gemma-4-31B-it. The first attempt inherited that default and produced a
# batch that could not be compared across architectures; it is archived under
# longmemeval_experiment/discarded_e4b_*. Set explicitly here so the model can
# never again be decided by a default.
export STRUCTMEM_LLM_MODEL="${STRUCTMEM_LLM_MODEL:-gemma-4-31B-it}"

export STRUCTMEM_M1_AUDIT_WORKERS="${STRUCTMEM_M1_AUDIT_WORKERS:-2}"
export LME_QDRANT_BASE="${LME_QDRANT_BASE:-$LME/qdrant_structmem_$SUFFIX}"
mkdir -p "$LME_QDRANT_BASE"

# knowledge-update 6, every other type 3. --max-questions only has to be
# non-zero to activate the sampler; once a quota is set it is the quota that
# decides, and the total is not truncated back down to this number.
QUOTA="knowledge-update=6,multi-session=3,temporal-reasoning=3"
QUOTA="$QUOTA,single-session-user=3,single-session-assistant=3"
QUOTA="$QUOTA,single-session-preference=3"

# Process order, not selection: the same 21 questions either way. The first
# attempt died at 10/21 with every knowledge-update question still queued,
# because types otherwise run in data-file order and knowledge-update is last.
# Those are the questions the state ablation exists to measure, so they go
# first and a second interruption cannot cost all of them.
export LME_TYPE_ORDER="${LME_TYPE_ORDER:-knowledge-update,temporal-reasoning,single-session-assistant}"

[ -x "$PY" ] || { echo "!! interpreter not found: $PY" >&2; exit 1; }

version_for () { echo "ablate_$(echo "$1" | tr '[:upper:]' '[:lower:]')_${SUFFIX}"; }

echo "=== LongMemEval ablation, ${SUFFIX} ==="
echo "    arms        : ${ARMS[*]}"
echo "    quota       : $QUOTA"
echo "    qdrant base : $LME_QDRANT_BASE"
echo

# Arms run ONE AT A TIME, in the order given, E0 first. Running them
# concurrently splits a decode-saturated endpoint three ways, so the baseline
# only lands when the whole batch lands. Sequential means E0 is complete and
# usable in about a third of the total time, and if the batch is interrupted
# there is a finished arm rather than three partial ones. Total wall time is
# roughly the same either way.
for arm in "${ARMS[@]}"; do
  ver="$(version_for "$arm")"
  echo "=== ${arm} starting $(date '+%F %T') -> results/structmem-${ver}/ ==="
  ( cd "$LME" && \
    STRUCTMEM_EXPERIMENT="$arm" LME_TYPE_QUOTA="$QUOTA" \
    "$PY" run_longmem.py --backend structmem --version "$ver" --max-questions 21 \
      > "logs_${ver}.out" 2>&1 )
  echo "=== ${arm} finished $(date '+%F %T') exit=$? ==="

  # Probe each arm as soon as it is done, so its stage attribution is available
  # while the next arm ingests, instead of waiting for the whole batch.
  if [ "${SKIP_PROBE:-0}" != "1" ]; then
    echo "=== probing ${arm} ==="
    ( cd "$LME" && "$PY" probe_longmem.py --run "structmem-${ver}" \
        > "probe_${ver}.out" 2>&1
      echo "    probe exit=$?" )
  fi
done
echo "=== all arms finished $(date '+%F %T') ==="

echo "=== BATCH FINISHED $(date '+%F %T') ==="
