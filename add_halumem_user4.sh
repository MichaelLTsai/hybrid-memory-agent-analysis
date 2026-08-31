#!/usr/bin/env bash
# Widen HaluMem for E1 and E3 from user #3 to users #3 and #4, so all three
# arms sit on the slice the four comparator architectures ran.
#
# Deliberately NOT ensure_halumem.sh: that script moves the results directory
# aside before every attempt, which here would delete user #3's completed
# ingest and force it to be redone. eval_structmem.py already resumes at user
# granularity ("<uuid> already done"), so simply re-invoking with the wider
# --max-users ingests only the missing user. Retries therefore keep the
# directory and let that resume do the work.
#
# Success is judged on qa_valid_num, not exit code: run.py exits 0 even when a
# user dies or goes unjudged, which is how E0 shipped 169 unjudged QA records.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/structmem_env/bin/python"
cd "$ROOT/halumem_experiment" || exit 1

idle () { ! pgrep -f "structmem_env/bin/python .*--backend structmem" >/dev/null 2>&1; }

check () {   # ver -> "qa_num qa_valid_num n_users"
  "$PY" -c "
import json,glob,sys
v=sys.argv[1]
try:
    d=json.load(open(f'results/structmem-{v}/structmem_scores.json'))
    q=d['question_answering']
    print(q['qa_num'], q['qa_valid_num'], len(glob.glob(f'results/structmem-{v}/tmp/*.json')))
except Exception: print(0,0,0)" "$1"
}

for spec in "ablate_e1_m1_31b:E1_m1" "ablate_e3_m1_m4_31b:E3_m1_m4"; do
  VER="${spec%%:*}"; ARM="${spec##*:}"
  echo "[add4] ==== $ARM ===="
  for try in 1 2 3; do
    while ! idle; do sleep 30; done
    echo "[add4] $(date '+%F %T') $ARM attempt $try, scope users #3+#4"
    STRUCTMEM_EXPERIMENT="$ARM" \
      "$PY" run.py --backend structmem --version "$VER" \
        --skip-users 2 --max-users 2 --llm-model gemma-4-31B-it \
        > "logs_${VER}_add4_try${try}.out" 2>&1
    echo "[add4]   exit=$?"
    read -r qn qv nu <<< "$(check "$VER")"
    echo "[add4]   qa_num=$qn qa_valid=$qv users_ingested=$nu"
    # Both users present and every QA record judged.
    if [ "${nu:-0}" -ge 2 ] && [ "${qv:-0}" -gt 0 ] && [ "${qn:-0}" = "${qv:-0}" ]; then
      echo "[add4]   OK, probing"
      "$PY" probe_halumem_unified.py --run "structmem-$VER" > "probe_${VER}.out" 2>&1
      echo "[add4]   probe exit=$?"
      break
    fi
    echo "[add4]   incomplete, retrying after 120s"
    sleep 120
  done
done
echo "[add4] ALL DONE $(date '+%F %T')"
