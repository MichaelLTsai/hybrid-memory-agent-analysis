#!/usr/bin/env bash
# Recover HaluMem E0 scores after the segfault, without redoing the ingest.
#
# The E0 HaluMem run ingested both users cleanly (52 MB of extracted memories in
# results/structmem-ablate_e0_baseline_31b/tmp/), completed Integrity, Accuracy
# and Update evaluation for both, then died with SIGSEGV (exit 139) before
# writing structmem_scores.json. The extraction is intact; only the evaluation
# needs redoing.
#
# run.py --eval-only skips run_extraction and evaluates the existing tmp/*.json,
# so this costs judge calls only, not the ~48 min of ingest.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
VER="${VER:-ablate_e0_baseline_31b}"

# One 31B line at a time: two lines is what drew 998 HTTP 503s earlier.
while pgrep -f "run_ablation_31b.sh|run_longmem.py --backend structmem" >/dev/null 2>&1; do
  sleep 60
done

cd "$ROOT/halumem_experiment" || exit 1
echo "[repair] $(date '+%F %T') starting HaluMem eval-only for $VER"
"$PY" run.py --backend structmem --version "$VER" --eval-only \
  > "logs_${VER}_evalonly.out" 2>&1
echo "[repair] eval-only exit=$? $(date '+%F %T')"

if [ -s "results/structmem-${VER}/structmem_scores.json" ]; then
  echo "[repair] scores written, running unified probe"
  "$PY" probe_halumem_unified.py --run "structmem-${VER}" > "probe_${VER}.out" 2>&1
  echo "[repair] probe exit=$? $(date '+%F %T')"
else
  echo "[repair] !! still no scores; eval-only did not succeed"
fi
echo "[repair] DONE $(date '+%F %T')"
