#!/usr/bin/env bash
# Re-judge HaluMem E0. Its user #3 has result_type=None on all 169 QA records
# and memory_update_type=None on all 123 update records: the answers were
# generated correctly but never judged, because the run segfaulted mid-evaluation
# and the earlier --eval-only recovery did not re-judge that user. The (all)
# denominators count those as failures, which is what dragged E0's HaluMem
# numbers down.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
VER=ablate_e0_baseline_31b
while pgrep -f "structmem_env/bin/python .*--backend structmem" >/dev/null 2>&1; do sleep 30; done
cd "$ROOT/halumem_experiment" || exit 1
cp "results/structmem-$VER/structmem_scores.json" \
   "results/structmem-$VER/structmem_scores.before_rejudge.json" 2>/dev/null
echo "[e0-eval] $(date '+%F %T') re-judging"
"$PY" run.py --backend structmem --version "$VER" --eval-only \
  > "logs_${VER}_rejudge.out" 2>&1
echo "[e0-eval] exit=$? $(date '+%F %T')"
"$PY" -c "
import json
d=json.load(open('results/structmem-$VER/structmem_scores.json'))
q=d['question_answering']; m=d['memory_update']
print(f\"[e0-eval] qa_num={q['qa_num']} valid={q['qa_valid_num']}  update_num={m['update_memory_num']} valid={m['update_memory_valid_num']}\")
"
"$PY" probe_halumem_unified.py --run "structmem-$VER" > "probe_${VER}.out" 2>&1
echo "[e0-eval] probe exit=$? DONE $(date '+%F %T')"
