#!/usr/bin/env bash
# =============================================================================
#  Memory failure attribution experiments: unified entry point
#
#      bash run.sh check       environment check, makes no API calls
#      bash run.sh matrix      rebuild the aggregate tables (~10s, no API key)
#      bash run.sh smoke       minimal end-to-end validation (~10 min, needs a key)
#      bash run.sh reproduce   full reproduction (~8 hours, needs a key)
#
#  To verify the published results, run check then matrix. Neither needs an API
#  key nor the datasets, and together they rebuild the complete results table.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
bad()  { echo "  ${RED}✗${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }
head1() { echo; echo "${BOLD}$*${RESET}"; }

# ── Virtual environment required by each backend ─────────────────────────────
# Mem0 v1 and v2 differ only in the mem0ai version installed: the adapter reads
# mem0.__version__ and branches on it, so the code is identical.
venv_for() {
    case "$1" in
        mem0v1|amem|rag) echo "$ROOT/venv_mem0v1/bin/python" ;;
        mem0v2)          echo "$ROOT/venv_memfail/bin/python" ;;
        structmem)       echo "$HOME/structmem_env/bin/python" ;;
        letta)           echo "$ROOT/venv_letta/bin/python" ;;
        memos)           echo "$ROOT/venv_memos/bin/python" ;;
        zep)             echo "$ROOT/venv_zep/bin/python" ;;
        tools)           echo "$ROOT/venv_memos/bin/python" ;;
        *)               echo "" ;;
    esac
}

# Locate a python suitable for the aggregation tools (needs openpyxl)
tools_python() {
    for p in "$ROOT/venv_memos/bin/python" "$ROOT/venv_mem0v1/bin/python" \
             "$ROOT/venv_memfail/bin/python" python3; do
        if "$p" -c "import openpyxl" >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}

# =============================================================================
#  check
# =============================================================================
cmd_check() {
    local fail=0

    head1 "Virtual environments"
    for name in mem0v1 mem0v2 structmem letta memos zep; do
        local py; py="$(venv_for "$name")"
        if [ -x "$py" ]; then
            ok "$(printf '%-10s' "$name") $("$py" -V 2>&1)  ${DIM}$py${RESET}"
        else
            warn "$(printf '%-10s' "$name") missing (this backend cannot run)  ${DIM}$py${RESET}"
        fi
    done

    head1 "Aggregation tools"
    if TOOLS_PY=$(tools_python); then
        ok "openpyxl available  ${DIM}$TOOLS_PY${RESET}"
    else
        bad "no python with openpyxl found; 'matrix' cannot run"; fail=1
    fi

    head1 "Environment variables"
    if [ -f "$ROOT/halumem_experiment/.env" ]; then
        ok "halumem_experiment/.env present"
        # Only checks whether values are filled in; never prints them
        local missing=""
        for k in OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL; do
            local v; v=$(grep -E "^${k}=" "$ROOT/halumem_experiment/.env" 2>/dev/null | head -1 | cut -d= -f2-)
            if [ -z "$v" ] || [[ "$v" == *"your"* ]]; then missing="$missing $k"; fi
        done
        if [ -n "$missing" ]; then
            warn "not filled in:$missing"
        else
            ok "primary LLM settings complete"
        fi
    else
        warn "halumem_experiment/.env missing ('matrix' does not need it; smoke / reproduce do)"
        echo "      cp .env.example halumem_experiment/.env   then fill in your key"
    fi

    head1 "Local services"
    local ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
    if curl -fsS --max-time 3 "$ollama_url/api/tags" >/dev/null 2>&1; then
        ok "Ollama running ($ollama_url)"
        if curl -fsS --max-time 3 "$ollama_url/api/tags" | grep -q "bge-m3"; then
            ok "embedding model bge-m3 ready"
        else
            warn "bge-m3 not found; run: ollama pull bge-m3"
        fi
    else
        warn "Ollama not running ($ollama_url). Not needed for 'matrix'; needed to run experiments"
    fi
    if curl -fsS --max-time 3 "http://localhost:8283/v1/health/" >/dev/null 2>&1; then
        ok "Letta server running"
    else
        warn "Letta server not running (affects the letta backend only; see halumem_experiment/LETTA_SETUP.md)"
    fi

    head1 "Datasets"
    local d
    for d in "locomo_experiment/data/locomo10.json" \
             "longmemeval_experiment/data/longmemeval_s.json" \
             "halumem_experiment/data/HaluMem-Medium.jsonl"; do
        if [ -f "$ROOT/$d" ]; then
            ok "$(printf '%-46s' "$d") $(du -h "$ROOT/$d" | cut -f1)"
        else
            warn "$(printf '%-46s' "$d") missing; run: bash scripts/download_data.sh"
        fi
    done
    [ -d "$ROOT/memfail_experiment/datasets" ] && ok "memfail_experiment/datasets/ (ships with the repo)"

    head1 "Result summaries"
    local n
    n=$(find "$ROOT"/*_experiment/results -name "*_scores.json" 2>/dev/null | wc -l | tr -d ' ')
    ok "$n scores.json files ship with the repo; 'matrix' rebuilds the tables from them"

    echo
    if [ "$fail" -eq 0 ]; then
        echo "${GREEN}Environment check complete.${RESET} Next: bash run.sh matrix"
    else
        echo "${RED}Required components are missing; resolve the items marked ✗ above.${RESET}"
        return 1
    fi
}

# =============================================================================
#  matrix: rebuild the tables from the bundled scores.json; no key, no datasets
# =============================================================================
cmd_matrix() {
    local py
    if ! py=$(tools_python); then
        bad "no python with openpyxl found"
        echo "      run: pip install openpyxl"
        return 1
    fi
    head1 "Rebuilding the failure matrix"
    echo "  ${DIM}$py build_matrix_excel.py${RESET}"
    "$py" build_matrix_excel.py || return 1

    for sub in halumem_experiment locomo_experiment longmemeval_experiment memfail_experiment; do
        local script="$ROOT/$sub/update_excel"
        [ "$sub" = "locomo_experiment" ]      && script="$ROOT/$sub/update_excel_locomo"
        [ "$sub" = "longmemeval_experiment" ] && script="$ROOT/$sub/update_excel_longmem"
        [ "$sub" = "memfail_experiment" ]     && script="$ROOT/$sub/update_excel_memfail"
        [ -f "${script}.py" ] || continue
        ( cd "$ROOT/$sub" && "$py" -c "
import sys
sys.path.insert(0, '.')
mod = __import__('$(basename "$script")')
mod.build_excel(mod.scan_runs())
" >/dev/null 2>&1 ) && ok "$sub/experiment_results.xlsx updated" \
                    || warn "$sub/experiment_results.xlsx skipped"
    done

    echo
    echo "${GREEN}Done.${RESET} Produced:"
    echo "  memory_failure_matrix.xlsx                     failure-stage matrix across datasets and architectures (main table)"
    echo "  halumem_experiment/experiment_results.xlsx     HaluMem per-run detail"
    echo "  locomo_experiment/experiment_results.xlsx      LoCoMo per-run detail"
    echo "  longmemeval_experiment/experiment_results.xlsx LongMemEval per-run detail"
    echo "  memfail_experiment/experiment_results.xlsx     MemFail per-run detail"
}

# =============================================================================
#  smoke: minimal end-to-end check that the key and environment actually work
# =============================================================================
cmd_smoke() {
    local py; py="$(venv_for mem0v1)"
    if [ ! -x "$py" ]; then bad "venv_mem0v1 is missing"; return 1; fi

    head1 "Connectivity test"
    ( cd "$ROOT/halumem_experiment" && "$py" test.py 2>&1 | head -5 ) \
        || { bad "LLM endpoint unreachable; check .env"; return 1; }

    head1 "LoCoMo x RAG baseline (1 conversation)"
    echo "  ${DIM}Lightest end-to-end path: no Mem0 extraction, exercises data, embeddings, and scoring${RESET}"
    ( cd "$ROOT/locomo_experiment" && \
      "$py" run_locomo.py --backend rag --version smoke_check --smoke --skip-extraction-eval )

    echo
    echo "${GREEN}Smoke test complete.${RESET} Results in locomo_experiment/results/rag-smoke_check/"
}

# =============================================================================
#  reproduce: full reproduction of batch ② (the one carrying cost instrumentation)
# =============================================================================
#  Sampling matches the published runs:
#    HaluMem      user #2, 77 sessions / 188 QA     --skip-users 1 --max-users 1
#    LoCoMo       conv-26, 19 sessions / 419 turns  --smoke
#    LongMemEval  3 questions per type, 18 total    --max-questions 18
#    MemFail      5 per subset (15 for persona)
#    top-k 20 throughout
# =============================================================================
cmd_reproduce() {
    local tag="${1:-repro}"

    cat <<EOF

${BOLD}Full reproduction${RESET}
  Version tag  : ${tag}
  Est. runtime : about 8 hours (four datasets x five backends)
  Est. cost    : depends on your LLM pricing; call volume is recorded in each
                 run's *_token_usage.json

${YELLOW}Important${RESET}: always use a fresh version tag. Mem0's Qdrant store is
persisted per tag, so reusing one inherits the previous vectors, doubles the
memory store, and invalidates the results.

EOF
    read -r -p "Proceed? [y/N] " yn
    [[ "$yn" =~ ^[Yy]$ ]] || { echo "Cancelled."; return 0; }

    local failed=()
    run_step() {  # run_step <description> <workdir> <python> <args...>
        local desc="$1" dir="$2" py="$3"; shift 3
        head1 "$desc"
        if [ ! -x "$py" ]; then warn "venv missing, skipping: $py"; failed+=("$desc (missing venv)"); return; fi
        echo "  ${DIM}cd $dir && $py $*${RESET}"
        if ( cd "$ROOT/$dir" && "$py" "$@" ); then ok "$desc complete"
        else bad "$desc failed"; failed+=("$desc"); fi
    }

    # ── HaluMem ──
    run_step "HaluMem x Mem0 v1" halumem_experiment "$(venv_for mem0v1)" \
        run.py --backend mem0 --version "v1_${tag}" --skip-users 1 --max-users 1 --top-k 20
    run_step "HaluMem x Mem0 v2" halumem_experiment "$(venv_for mem0v2)" \
        run.py --backend mem0 --version "v2_${tag}" --skip-users 1 --max-users 1 --top-k 20
    run_step "HaluMem x A-MEM" halumem_experiment "$(venv_for amem)" \
        run.py --backend amem --version "amem_${tag}" --skip-users 1 --max-users 1 --top-k 20
    run_step "HaluMem x StructMem" halumem_experiment "$(venv_for structmem)" \
        run.py --backend structmem --version "sm_${tag}" --skip-users 1 --max-users 1 --top-k 20
    run_step "HaluMem x Letta" halumem_experiment "$(venv_for letta)" \
        run.py --backend letta --version "letta_${tag}" --skip-users 1 --max-users 1 --top-k 20

    # ── LoCoMo ──
    run_step "LoCoMo x Mem0 v1" locomo_experiment "$(venv_for mem0v1)" \
        run_locomo.py --backend mem0 --version "v1_${tag}" --smoke --top-k 20
    run_step "LoCoMo x Mem0 v2" locomo_experiment "$(venv_for mem0v2)" \
        run_locomo.py --backend mem0 --version "v2_${tag}" --smoke --top-k 20
    run_step "LoCoMo x A-MEM" locomo_experiment "$(venv_for amem)" \
        run_locomo.py --backend amem --version "amem_${tag}" --smoke --top-k 20
    run_step "LoCoMo x StructMem" locomo_experiment "$(venv_for structmem)" \
        run_locomo.py --backend structmem --version "sm_${tag}" --smoke --top-k 20
    run_step "LoCoMo x Letta" locomo_experiment "$(venv_for letta)" \
        run_locomo.py --backend letta --version "letta_${tag}" --smoke --top-k 20

    # ── LongMemEval ──
    run_step "LongMemEval x Mem0 v1" longmemeval_experiment "$(venv_for mem0v1)" \
        run_longmem.py --backend mem0 --version "v1_${tag}" --max-questions 18 --top-k 20
    run_step "LongMemEval x Mem0 v2" longmemeval_experiment "$(venv_for mem0v2)" \
        run_longmem.py --backend mem0 --version "v2_${tag}" --max-questions 18 --top-k 20
    run_step "LongMemEval x A-MEM" longmemeval_experiment "$(venv_for amem)" \
        run_longmem.py --backend amem --version "amem_${tag}" --max-questions 18 --top-k 20
    run_step "LongMemEval x StructMem" longmemeval_experiment "$(venv_for structmem)" \
        run_longmem.py --backend structmem --version "sm_${tag}" --max-questions 18 --top-k 20
    run_step "LongMemEval x Letta" longmemeval_experiment "$(venv_for letta)" \
        run_longmem.py --backend letta --version "letta_${tag}" --max-questions 18 --top-k 20

    # ── MemFail ──
    head1 "MemFail: four task pipelines"
    echo "  ${DIM}MemFail ships its own shell pipelines; run them in turn${RESET}"
    for task in conditional_facts coexisting_facts long_hop personal_retrieval; do
        if ( cd "$ROOT/memfail_experiment" && ./${task}.sh --run-id "${tag}" ); then
            ok "MemFail $task complete"
        else
            bad "MemFail $task failed"; failed+=("MemFail $task")
        fi
    done

    head1 "Rebuilding the tables"
    warn "The new runs are tagged '${tag}', which does not match the run names registered"
    echo "      in build_matrix_excel.py. To fold them into the matrix, edit the BACKENDS"
    echo "      table at the top of that file, then run:"
    echo "        python scripts/export_derived_cache.py"
    echo "        bash run.sh matrix"

    echo
    if [ ${#failed[@]} -eq 0 ]; then
        echo "${GREEN}All steps complete.${RESET}"
    else
        echo "${RED}${#failed[@]} step(s) did not succeed:${RESET}"
        printf '  - %s\n' "${failed[@]}"
        return 1
    fi
}

# =============================================================================
case "${1:-}" in
    check)     cmd_check ;;
    matrix)    cmd_matrix ;;
    smoke)     cmd_smoke ;;
    reproduce) shift; cmd_reproduce "$@" ;;
    *)
        sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
