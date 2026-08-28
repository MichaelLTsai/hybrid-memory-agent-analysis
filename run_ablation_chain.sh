#!/usr/bin/env bash
# Wait for the in-flight HaluMem arms to finish, then run the LongMemEval arms.
#
# The NCHC endpoint is decode-saturated: measured aggregate throughput rose only
# ~15% going from 2 to 10 concurrent lines, so running both benchmarks at once
# just splits a fixed pie and delays every result. Sequencing them means a
# complete HaluMem comparison lands hours earlier, and LongMemEval then gets the
# whole endpoint to itself.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

# E3 was restarted late (a hung request with no client timeout took its first
# attempt down at session 48), so it is deliberately NOT a gate here: waiting
# for it would idle the endpoint for hours. It runs alongside LongMemEval.
echo "[chain] $(date '+%F %T') waiting for HaluMem E1/E2/E4 to finish"
while pgrep -f "run.py --backend structmem --version ablate_e1_m1" >/dev/null \
   || pgrep -f "run.py --backend structmem --version ablate_e2_m1_m3" >/dev/null \
   || pgrep -f "run.py --backend structmem --version ablate_e4_full" >/dev/null; do
  sleep 60
done
echo "[chain] $(date '+%F %T') HaluMem done, starting LongMemEval"

ARMS="E1_m1 E2_m1_m3 E3_m1_m4 E4_full" ./run_state_ablation.sh lme
echo "[chain] $(date '+%F %T') LongMemEval done"
