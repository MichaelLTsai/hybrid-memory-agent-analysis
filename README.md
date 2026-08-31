# Hybrid Memory Architectures for Performance and Behavioral Analysis of Intelligent Agents in Multi-Objective Tasks

Experiments for my master's thesis at NCCU (Dept. of Computer Science).

Five LLM long-term memory architectures on four benchmarks, with every wrong
answer attributed to a stage: summarization, storage, retrieval, or reasoning.

Aggregate QA accuracy hides which stage broke. Two systems scoring 60 can fail
for opposite reasons, one never extracting the fact and the other extracting but
failing to retrieve it, and the fixes point in opposite directions. So each stage
gets its own instrumentation. The stage-level profiles turn out to differ a lot
between architectures, which is what makes a hybrid worth considering; the hybrid
itself is not in this repo.

## Checking the results

Every run's scores ship with the repo, so the tables rebuild from a clone. No API
key, no datasets, about a minute.

```bash
git clone https://github.com/MichaelLTsai/hybrid-memory-agent-analysis.git
cd hybrid-memory-agent-analysis
python3 -m venv venv_memos
./venv_memos/bin/pip install -r requirements-tools.txt
bash run.sh check
bash run.sh matrix
```

`memory_failure_matrix.xlsx` comes out at 16 rows by 109 columns. Every cell
traces to a `*/results/<run>/*_scores.json` written when that experiment ran;
`build_matrix_excel.py` aggregates and never recomputes.

To actually run experiments you need an OpenAI-compatible endpoint and the
datasets:

```bash
cp .env.example halumem_experiment/.env    # fill in your key
bash scripts/download_data.sh locomo       # 2.7 MB
bash run.sh smoke                          # ~10 min end-to-end check

bash scripts/download_data.sh              # all four, ~400 MB
bash run.sh reproduce my_tag               # ~8 h
```

Use a fresh `my_tag` every time. Mem0's Qdrant store persists per tag, so
reusing one inherits the old vectors and doubles the store.

## Results

Batch 1 (2026-08-14), the only batch where all five architectures finished all
four datasets. Full detail in `memory_failure_matrix.xlsx`.

| | LongMemEval QA | LoCoMo QA | HaluMem QA | MemFail acc. | LoCoMo P4 | HaluMem P4 |
|---|---|---|---|---|---|---|
| Mem0 v1 | 0.389 | 0.457 | 0.372 | 0.714 | 0.480 | 0.353 |
| Mem0 v2 | 0.389 | 0.613 | 0.351 | 0.771 | 0.599 | 0.354 |
| StructMem | 0.056 | 0.673 | 0.473 | 0.686 | 0.704 | 0.221 |
| A-MEM | 0.444 | 0.482 | 0.553 | 0.829 | 0.520 | 0.688 |
| Letta | 0.556 | 0.588 | 0.580 | 0.543 | 0.539 | 0.311 |

P4 asks whether what was retrieved suffices to answer, so it only means
something read against QA. StructMem tops both on LoCoMo, so retrieval is not its
bottleneck there; on HaluMem its P4 falls to 0.221 while QA holds at 0.473, so
that score is coming from somewhere else. Nothing leads everywhere.

## Setup

The architectures, and the venv each needs:

- Mem0 v1, `venv_mem0v1`: LLM fact extraction plus vector retrieval, `mem0ai` 1.x
- Mem0 v2, `venv_memfail`: same adapter, `mem0ai` 2.x. `eval_mem0_oss.py` reads
  `mem0.__version__` and branches, so switching versions means switching venvs
- StructMem, `~/structmem_env`: event-level dual-perspective extraction plus
  cross-event consolidation, built on LightMem
- A-MEM, `venv_mem0v1`: Zettelkasten notes per turn, evolving links to existing notes
- Letta (MemGPT), `venv_letta`: stateful agent managing core and archival memory
  through tools

`rag`, `memos`, `zep`, and `graphiti` adapters also run but sit outside the main
comparison. `rag` stores raw turns with no memory management, as a retrieval
lower bound.

Dependencies conflict across backends, most obviously `mem0ai` 1.x against 2.x,
hence one venv each. Everything installs under `$HOME` or the project directory;
no sudo, Docker, or Homebrew.

```bash
ollama pull bge-m3        # shared embeddings, https://ollama.com

python3.12 -m venv venv_mem0v1
./venv_mem0v1/bin/pip install -r halumem_experiment/requirements.txt
./venv_mem0v1/bin/pip install "mem0ai==1.0.11" git+https://github.com/agiresearch/A-mem.git

python3.12 -m venv venv_memfail
./venv_memfail/bin/pip install -r halumem_experiment/requirements.txt
./venv_memfail/bin/pip install "mem0ai>=2.0"

python3.11 -m venv ~/structmem_env        # LightMem needs Python < 3.12
git clone https://github.com/zjunlp/LightMem.git ~/LightMem
~/structmem_env/bin/pip install -e ~/LightMem

python3.12 -m venv venv_letta             # server + PostgreSQL,
./venv_letta/bin/pip install letta        # see halumem_experiment/LETTA_SETUP.md

python3.12 -m venv venv_memos && ./venv_memos/bin/pip install MemoryOS   # optional
python3.12 -m venv venv_zep   && ./venv_zep/bin/pip install zep-cloud    # optional
```

`bash run.sh check` reports what is missing and skips only the affected backends.

## What was run

| Dataset | Sampling |
|---|---|
| HaluMem | user #2, 77 sessions / 3,242 turns / 188 QA |
| LoCoMo | conv-26, 19 sessions / 419 turns / 199 QA |
| LongMemEval-S | 3 questions per type, 18 total |
| MemFail | 5 per subset (15 for persona), 35 total |

Samples are small on purpose. Every question also needs the P1/P4/P5 probe
judgments, which cost several times a plain QA pass, and under a fixed budget
five architectures on identical questions with stage-level evidence beats a
larger sample yielding one number.

Models are pinned across all four experiments so differences fall on the
architectures: `gemma-4-31B-it` for extraction (`openai-proxy/gemma-4-31B-it` for
Letta), `gemma-4-E4B-it` as judge at temperature 0 from batch 2 on, `bge-m3`
embeddings on local Ollama, top-k 20, served through the NCHC GenAI portal. Any
OpenAI-compatible endpoint works; change three variables in `.env`. A different
model shifts absolute scores but the comparison is relative under one fixed model.

## Running one experiment

```bash
cd halumem_experiment
../venv_mem0v1/bin/python run.py --backend mem0 --version v1_repro \
    --skip-users 1 --max-users 1 --top-k 20

cd locomo_experiment
~/structmem_env/bin/python run_locomo.py --backend structmem \
    --version sm_repro --smoke --top-k 20

cd longmemeval_experiment
../venv_mem0v1/bin/python run_longmem.py --backend amem \
    --version amem_repro --max-questions 18 --top-k 20

cd memfail_experiment
./coexisting_facts.sh --run-id repro
```

`--eval-only` rescores existing results and `--skip-eval` stops after extraction,
so interrupted runs resume. Probes run separately:

```bash
../venv_memos/bin/python probe_halumem.py --run mem0_oss-v1_repro
```

Then add the new tags to `BACKENDS` at the top of `build_matrix_excel.py`, run
`python scripts/export_derived_cache.py`, and `bash run.sh matrix`.

## Metrics

Columns group by stage; the `Definitions` sheet in the workbook has all of them.

- Summary: P1, HaluMem F1. Was it recorded, and recorded correctly
- Storage: HaluMem update, LongMemEval KU. Did a new value supersede the stale one
- Retrieval: P4, Recall@5, NDCG@5. Does what came back suffice to answer
- Reasoning: P5. Passed P4 and still got it wrong
- Memory Performance: QA accuracy

P1, P4, and P5 are ours; the official metrics cover no retrieval stage. Two
things that look alike and are not:

```
retrieval_ratio = P(failed at retrieval | answered wrong)   denominator: wrong answers
P4              = P(retrieved context suffices)             denominator: all questions
```

The first is an attribution share. Two systems can post the same
`retrieval_ratio` and differ tenfold in retrieval ability. P4 runs over every
question, which is what lets it mean the same thing on all three datasets.

Subset columns carry a group code for what a correct answer demands of memory:
single-point recall, multi-memory chained, parallel, or mixed, temporal
reasoning, post-update value, and abstention. Comparable subsets from different
datasets sit adjacent.

## Repository contents

```
run.sh                       check / matrix / smoke / reproduce
build_matrix_excel.py        aggregates scores.json into the failure-stage matrix
memory_failure_matrix.xlsx   main results table
results_derived_cache.json   values derived from the untracked raw output
scripts/                     dataset download, cache regeneration
halumem_experiment/          runner, adapters, scoring, probes, results
locomo_experiment/           same shape
longmemeval_experiment/      same shape
memfail_experiment/          four .sh pipelines plus the src/ harness
```

Tracked: source, and per run the scores, token usage, metadata, and probe detail.

Not tracked: `venv_*/` (5.9 GB); datasets (~400 MB across four, individual files
over GitHub's 100 MB limit, fetch with `scripts/download_data.sh`);
`*_eval_results.jsonl` and `*_detail.jsonl` (2.5 GB, largest single file 77 MB);
qdrant and kuzu state; `.env`.

Two column families, per-run store size and HaluMem per-type accuracy, could only
be computed from that raw output. They are cached in `results_derived_cache.json`
(about 5 KB), which `build_matrix_excel.py` falls back to when the raw files are
absent. Both paths produce identical matrices apart from the timestamp. Rerun
`scripts/export_derived_cache.py` after adding or changing a run.

## Limitations

- Small samples: one LoCoMo conversation (199 QA), one HaluMem user (188 QA), 18
  LongMemEval questions, 35 MemFail questions. The ordering across architectures
  comes from identical questions and holds, but confidence intervals on absolute
  values are wide and a small gap in one cell means little.
- Batches 2 and 3 are unfinished; cells marked `RUNNING` say so. Batch 1 is complete.
- Cost columns exist only for batch 2. Staged token and latency instrumentation
  landed 2026-08-18; earlier runs recorded only `total_tokens`.
- Letta is not strictly comparable. It answers with its own agent rather than
  handing memories to a shared LLM, and has three memory tiers. Both a `Letta`
  row and a `Letta (with history)` row are kept, differing on whether the message
  history visible at answering time counts as retrieved context. The gap is a finding.
- Judge temperature was unpinned in batch 1. The HaluMem rows for StructMem,
  A-MEM, and Letta reuse those earlier runs.
- StructMem runs with `pre_compress` and `topic_segment` off; the reference config
  wants LLMLingua-2 on CUDA, unavailable here. The comparison covers the memory
  mechanism, not the compression module.

## Sources

HaluMem ([gated](https://huggingface.co/datasets/MemTensor/HaluMem)) ·
[LoCoMo](https://github.com/snap-research/locomo) ·
[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval) ·
[MemFail](https://huggingface.co/datasets/ishirgarg/MemFail) ·
[Mem0](https://github.com/mem0ai/mem0) ·
[A-MEM](https://github.com/agiresearch/A-mem) ·
[LightMem](https://github.com/zjunlp/LightMem) ·
[Letta](https://github.com/letta-ai/letta)

`memfail_experiment/` is adapted from the MemFail repo; its original README is
kept in that directory.
