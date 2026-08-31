# Hybrid Memory Architectures for Performance and Behavioral Analysis of Intelligent Agents in Multi-Objective Tasks

Experiments for my master's thesis at NCCU (Dept. of Computer Science).

Five LLM long-term memory architectures on four benchmarks, with every wrong
answer attributed to a stage: summarization, storage, retrieval, or reasoning.

Aggregate QA accuracy hides which stage broke. Two systems scoring 60 can fail
for opposite reasons, one never extracting the fact and the other extracting but
failing to retrieve it, and the fixes point in opposite directions. So each stage
gets its own instrumentation. The stage profiles differ enough between
architectures that a hybrid becomes worth considering; the hybrid itself is not
in this repo.

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

`memory_failure_matrix.xlsx` comes out at 26 rows by 88 columns across twelve
sheets: `Failure Matrix` is the master, then per-dataset views, three pivots
(`By Sub-dataset`, `By Category`, `By Group`), a `Mem0 v1 vs v2` comparison, a
`Subset Map`, `Definitions`, and `Run parameters`. Every cell traces to a
`*/results/<run>/*_scores.json` written when that experiment ran;
`build_matrix_excel.py` aggregates and never recomputes.

To run experiments you need an OpenAI-compatible endpoint and the datasets:

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

Batch 6, the comparable intersection: every row is `gemma-4-31B-it`, LoCoMo is
conv-26's 199 questions, MemFail the same 35 for all five. Earlier batches mix
models or slices, which is why they cannot be ranked against each other. QA is
end-to-end accuracy, higher better; P4 and P5 are failure rates, lower better.

| Batch 6 | LongMemEval | LoCoMo | HaluMem | MemFail |
|---|---|---|---|---|
| Mem0 v1 QA | 0.455 | 0.442 | 0.297 | 0.743 |
| Mem0 v2 QA | 0.409 | 0.618 | 0.353 | 0.771 |
| StructMem QA | 0.429 | 0.709 | 0.252 | 0.857 |
| A-MEM QA | 0.500 | 0.492 | 0.489 | 0.771 |
| Letta QA | 0.409 | 0.608 | 0.506 | 0.514 |

P4 asks whether what was retrieved suffices to answer; P5 is the share that
passed P4 and still got it wrong. Read together they place the failure:

| Batch 6 | LongMemEval P4 / P5 | LoCoMo P4 / P5 | HaluMem P4 / P5 |
|---|---|---|---|
| Mem0 v1 | 0.091 / 0.364 | 0.107 / 0.183 | 0.221 / 0.390 |
| Mem0 v2 | 0.136 / 0.364 | 0.045 / 0.081 | 0.231 / 0.390 |
| StructMem | 0.000 / 0.238 | 0.057 / 0.175 | 0.144 / 0.598 |
| A-MEM | 0.000 / 0.273 | 0.128 / 0.144 | 0.110 / 0.459 |
| Letta | 0.000 / 0.045 | 0.000 / 0.151 | 0.086 / 0.362 |

StructMem tops LoCoMo QA at 0.709 with a low P4 failure rate, so retrieval is not
its bottleneck there. On HaluMem it is last at 0.252, and P4 failure stays low
(0.144) while P5 failure is the highest of the five (0.598): the material comes
back and the answer still goes wrong. An aggregate score would have shown only
that StructMem is inconsistent. Nothing leads everywhere.

Batch 6 also carries a StructMem ablation (E0 baseline through E4 full, arms
M1/M3/M4), reported on the 31B runs.

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
| HaluMem | one user's sessions, 188 QA in the early batches, 360 in batch 6 |
| LoCoMo | conv-26, 19 sessions / 419 turns / 199 QA |
| LongMemEval-S | 3 questions per type, 18 to 22 depending on batch |
| MemFail | 5 per subset (15 for persona), 35 total |

Exact per-batch scale is in the `Run parameters` sheet. Samples are small on
purpose: every question also needs the P1/P4/P5 probe judgments, which cost
several times a plain QA pass, and under a fixed budget five architectures on
identical questions with stage-level evidence beats a larger sample yielding one
number.

Models are pinned so differences fall on the architectures: `gemma-4-31B-it` for
extraction (`openai-proxy/gemma-4-31B-it` for Letta), `gemma-4-E4B-it` as judge
at temperature 0 from batch 2 on, `bge-m3` embeddings on local Ollama, top-k 20,
served through the NCHC GenAI portal. Any OpenAI-compatible endpoint works;
change three variables in `.env`. A different model shifts absolute scores but
the comparison is relative under one fixed model.

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
../venv_memos/bin/python probe_halumem_unified.py --run mem0_oss-v1_repro
```

Then add the new tags to `BACKENDS` at the top of `build_matrix_excel.py`, run
`python scripts/export_derived_cache.py`, and `bash run.sh matrix`.

## Metrics

Columns group by stage; the `Definitions` sheet has all of them.

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
single-point recall, multi-hop composition, multi-memory parallel, temporal
reasoning, post-update value, abstention and correction, and application and
extrapolation. The `Subset Map` sheet lists which subset falls where and why.

## Repository contents

```
run.sh                       check / matrix / smoke / reproduce
build_matrix_excel.py        aggregates scores.json into the failure-stage matrix
memory_failure_matrix.xlsx   main results table
results_derived_cache.json   values derived from the untracked raw output
scripts/                     dataset download, cache regeneration
halumem_experiment/          runner, adapters, scoring, probe_halumem_unified.py, results
locomo_experiment/           same shape, probe_locomo.py
longmemeval_experiment/      same shape, probe_longmem.py
memfail_experiment/          four .sh pipelines plus the src/ harness
```

Tracked: source, and per run the scores, token usage, metadata, and probe detail.

Not tracked: `venv_*/` (5.9 GB); datasets (~400 MB across four, individual files
over GitHub's 100 MB limit, fetch with `scripts/download_data.sh`);
`*_eval_results.jsonl` and `*_detail.jsonl` (2.5 GB, largest single file 77 MB);
qdrant and kuzu state; `.env`.

Two column families, per-run store size and HaluMem per-type accuracy, could only
be computed from that raw output. They are cached in `results_derived_cache.json`,
which `build_matrix_excel.py` falls back to when the raw files are absent. Both
paths produce identical matrices apart from the timestamp. Rerun
`scripts/export_derived_cache.py` after adding or changing a run.

## Limitations

- Small samples: one LoCoMo conversation (199 QA), one HaluMem user, 18 to 22
  LongMemEval questions, 35 MemFail questions. The ordering across architectures
  comes from identical questions and holds, but confidence intervals on absolute
  values are wide and a small gap in one cell means little.
- Batch 6's denominators are close but not identical. StructMem E0 answered 21
  LongMemEval and 353 HaluMem questions against the comparators' 22 and 360.
  Every metric is a rate, so the gap does not bias them.
- Cost columns start at batch 2. Staged token and latency instrumentation landed
  2026-08-18; batch 1 recorded only `total_tokens`.
- Letta is not strictly comparable. It answers with its own agent rather than
  handing memories to a shared LLM, and has three memory tiers. Batch 1 keeps
  both a `Letta` row and a `Letta (with history)` row, differing on whether the
  message history visible at answering time counts as retrieved context. The gap
  is a finding.
- Judge temperature was unpinned in batch 1.
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
