#!/usr/bin/env bash
# =============================================================================
#  Download the four benchmark datasets
#
#  These are public third-party datasets totalling about 400 MB, with individual
#  files exceeding GitHub's per-file limit, so they are not tracked in the
#  repository. Run this once.
#
#      bash scripts/download_data.sh              # everything
#      bash scripts/download_data.sh locomo       # a single dataset
#
#  Requires python3 and curl. The HuggingFace-hosted datasets need the
#  huggingface_hub package; the script prints the install command if it is missing.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(locomo longmemeval halumem memfail)

want() { for t in "${TARGETS[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }

# Locate a python that has huggingface_hub installed
hf_python() {
    for p in "$ROOT"/venv_*/bin/python python3; do
        command -v "$p" >/dev/null 2>&1 || [ -x "$p" ] || continue
        if "$p" -c "import huggingface_hub" >/dev/null 2>&1; then echo "$p"; return 0; fi
    done
    return 1
}

# ── LoCoMo: direct download from GitHub, 2.7 MB ──────────────────────────────
if want locomo; then
    echo "==> LoCoMo"
    mkdir -p "$ROOT/locomo_experiment/data"
    if [ -f "$ROOT/locomo_experiment/data/locomo10.json" ]; then
        echo "    already present, skipping"
    else
        curl -fL --progress-bar \
            -o "$ROOT/locomo_experiment/data/locomo10.json" \
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" \
            && echo "    done" || echo "    !! download failed"
    fi
fi

# ── LongMemEval-S: HuggingFace, about 265 MB ─────────────────────────────────
if want longmemeval; then
    echo "==> LongMemEval-S"
    mkdir -p "$ROOT/longmemeval_experiment/data"
    if [ -f "$ROOT/longmemeval_experiment/data/longmemeval_s.json" ]; then
        echo "    already present, skipping"
    elif PY=$(hf_python); then
        "$PY" "$ROOT/scripts/_hf_download.py" xiaowu0162/longmemeval longmemeval_s \
            "$ROOT/longmemeval_experiment/data/longmemeval_s.json" \
            || echo "    !! download failed; check your network and HuggingFace access"
    else
        echo "    !! no python with huggingface_hub found; run: pip install huggingface_hub"
    fi
fi

# ── HaluMem: HuggingFace, gated; accept the license and log in first ─────────
if want halumem; then
    echo "==> HaluMem-Medium"
    mkdir -p "$ROOT/halumem_experiment/data"
    if [ -f "$ROOT/halumem_experiment/data/HaluMem-Medium.jsonl" ]; then
        echo "    already present, skipping"
    elif PY=$(hf_python); then
        if ! "$PY" "$ROOT/scripts/_hf_download.py" MemTensor/HaluMem HaluMem-Medium.jsonl \
                "$ROOT/halumem_experiment/data/HaluMem-Medium.jsonl"; then
            echo "    !! Download failed. HaluMem is a gated dataset; two steps are required:"
            echo "       1. Accept the license at https://huggingface.co/datasets/MemTensor/HaluMem"
            echo "       2. Run: huggingface-cli login"
            echo "       Then re-run this script."
        fi
    else
        echo "    !! no python with huggingface_hub found; run: pip install huggingface_hub"
    fi
fi

# ── MemFail: ships with the repository ───────────────────────────────────────
if want memfail; then
    echo "==> MemFail"
    echo "    Already bundled under memfail_experiment/datasets/, no download needed"
    echo "    Upstream: https://huggingface.co/datasets/ishirgarg/MemFail"
fi

echo
echo "Dataset preparation complete."
