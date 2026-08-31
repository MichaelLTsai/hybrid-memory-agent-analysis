#!/usr/bin/env bash
# Wait for the E0 re-judge, then widen E1 and E3 to users #3 and #4.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[chain2] $(date '+%F %T') waiting for the E0 re-judge"
while pgrep -f "^bash .*/repair_e0_eval\.sh$" >/dev/null 2>&1; do sleep 30; done
while pgrep -f "structmem_env/bin/python .*--backend structmem" >/dev/null 2>&1; do sleep 30; done
echo "[chain2] $(date '+%F %T') E0 re-judge finished"
tail -3 "$ROOT/repair_e0_eval.out" 2>/dev/null | sed 's/^/[chain2]   /'
echo "[chain2] $(date '+%F %T') widening E1 and E3 to users #3+#4"
"$ROOT/add_halumem_user4.sh" >> "$ROOT/add_halumem_user4.out" 2>&1
echo "[chain2]   exit=$?"
echo "[chain2] ALL DONE $(date '+%F %T')"
