# Hybrid Memory Architectures for Performance and Behavioral Analysis of Intelligent Agents in Multi-Objective Tasks

Master's thesis research code and results, Department of Computer Science,
National Chengchi University.

## What this repository contains

This stage of the work builds the **measurement foundation** for the thesis:
a comparison of five LLM long-term memory architectures across four benchmark
datasets, where every wrong answer is attributed to one of four stages,
**summarization / storage / retrieval / reasoning**, rather than being folded
into a single aggregate score.

Conventional memory benchmarks report only end-to-end QA accuracy. Two systems
both scoring 60 can fail for opposite reasons: one never extracted the fact,
the other extracted it but could not retrieve it. Those two failures call for
opposite fixes. This project decomposes a memory system into
`extract → write → retrieve → answer` and instruments each stage separately,
so every number maps to a concrete architectural component.

This is also what motivates the hybrid direction in the thesis title. As the
[results](#results) below show, no single architecture leads across all four
datasets: each one's strengths and weaknesses are **distributed across stages**,
some strong at retrieval, others at extraction. That stage-level evidence is
what defines the design space for a hybrid memory architecture. The hybrid
architecture itself is out of scope for this repository.

---

## Contents

- [Quick start](#quick-start)
- [Experimental setup](#experimental-setup)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Full environment setup](#full-environment-setup)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Metric definitions](#metric-definitions)
- [Known limitations](#known-limitations)

---

## Quick start

Three tiers, ordered by the cost of running them.

### Tier 1: verify the results (about 1 minute, no API key, no datasets)

Every run's scoring summary ships with the repository, so the aggregate matrix
can be rebuilt from repository contents alone. No experiment needs to be re-run
and no API key is required. **To check the research results, this tier is enough.**

```bash
git clone https://github.com/MichaelLTsai/hybrid-memory-agent-analysis.git
cd hybrid-memory-agent-analysis

python3 -m venv venv_memos
./venv_memos/bin/pip install -r requirements-tools.txt

bash run.sh check      # environment check, lists whatever is missing
bash run.sh matrix     # rebuild the aggregate tables
```

This produces `memory_failure_matrix.xlsx`, 16 rows by 109 columns, covering
four datasets by five architectures. Every cell traces back to
`*/results/<run>/*_scores.json`, the raw scores written by the scoring code at
the time each experiment ran. The script only aggregates; it never recomputes.

### Tier 2: smoke test (about 10 minutes, API key required)

Confirms that your LLM endpoint, embedding service, and scoring chain all work.

```bash
cp .env.example halumem_experiment/.env
$EDITOR halumem_experiment/.env        # fill in OPENAI_API_KEY and friends

bash scripts/download_data.sh locomo   # LoCoMo only, 2.7 MB
bash run.sh smoke
```

### Tier 3: full reproduction (about 8 hours, API key required)

```bash
bash scripts/download_data.sh          # all four datasets, about 400 MB
bash run.sh reproduce my_tag
```

`my_tag` is a version tag, and **a fresh one is required for every re-run**.
Mem0's Qdrant store is persisted in a per-tag directory, so reusing a tag
inherits the previous vectors, doubles the memory store, and invalidates
the results.

---

## Experimental setup

### Five memory architectures

| Architecture | Mechanism | Virtual environment |
|---|---|---|
| **Mem0 v1** | LLM fact extraction plus vector retrieval, `mem0ai` 1.x | `venv_mem0v1` |
| **Mem0 v2** | Same adapter, `mem0ai` 2.x (revised extraction and update policy) | `venv_memfail` |
| **StructMem** | Event-level dual-perspective extraction plus cross-event consolidation, built on LightMem | `~/structmem_env` |
| **A-MEM** | Zettelkasten-style per-turn notes that evolve links to existing notes | `venv_mem0v1` |
| **Letta** (MemGPT) | Stateful agent managing its own core and archival memory through tools | `venv_letta` |

Mem0 v1 and v2 run **the same adapter code**: `eval_mem0_oss.py` reads
`mem0.__version__` and branches internally, so switching versions means
switching virtual environments, not editing code.

Four further backends serve as controls: `rag` (no memory management at all,
raw conversation turns stored verbatim, the retrieval lower bound), `memos`,
`zep`, and `graphiti`. These are excluded from the main comparison table but
their adapters are runnable.

### Four datasets

| Dataset | What it probes | Sampling used here |
|---|---|---|
| **HaluMem** | Extraction completeness and hallucination, with official memory-point-level annotations | User #2, 77 sessions / 3,242 turns / 188 QA |
| **LoCoMo** | Multi-hop and temporal reasoning over very long two-party dialogue | conv-26, 19 sessions / 419 turns / 199 QA |
| **LongMemEval-S** | Six question types, each with its own haystack of roughly 50 sessions | 3 questions per type, 18 total |
| **MemFail** | Adversarial datasets hand-designed to elicit specific failure modes | 5 per subset (15 for persona), 35 total |

The small sample sizes are a deliberate trade-off. The object of study is
**stage-level attribution**, and every question additionally requires the
P1 / P4 / P5 probe judgments, costing several times a plain QA pass. Under a
fixed budget, running all five architectures on exactly the same questions with
stage-level evidence is worth more than a larger sample yielding one number.

### Model configuration

All four experiments pin the same models, so cross-architecture differences can
be attributed to the architectures themselves:

- Extraction LLM: `gemma-4-31B-it` (`openai-proxy/gemma-4-31B-it` for Letta)
- Judge LLM: `gemma-4-E4B-it`, temperature pinned to 0 from batch ② onward
  (see [known limitations](#known-limitations), item 5)
- Embeddings: `bge-m3` on a local Ollama instance, shared by all architectures
- top-k: 20 throughout
- API endpoint: NCHC GenAI portal (OpenAI-compatible)

The endpoint can be swapped for any OpenAI-compatible service by changing three
variables in `.env`; no code changes are needed. Changing the model shifts the
absolute scores, but the conclusions here rest on relative differences between
architectures under a single fixed model.

---

## Results

The table below is batch ① (2026-08-14), the only batch in which all five
architectures completed on all four datasets. The full 109 columns are in
`memory_failure_matrix.xlsx`.

| Architecture | LongMemEval QA | LoCoMo QA | HaluMem QA | MemFail acc. | LoCoMo P4 | HaluMem P4 |
|---|---|---|---|---|---|---|
| Mem0 v1 | 0.389 | 0.457 | 0.372 | 0.714 | 0.480 | 0.353 |
| Mem0 v2 | 0.389 | 0.613 | 0.351 | 0.771 | 0.599 | 0.354 |
| StructMem | 0.056 | 0.673 | 0.473 | 0.686 | 0.704 | 0.221 |
| A-MEM | 0.444 | 0.482 | 0.553 | 0.829 | 0.520 | 0.688 |
| Letta | 0.556 | 0.588 | 0.580 | 0.543 | 0.539 | 0.311 |

QA is end-to-end accuracy; P4 is the retrieval probe, asking whether what was
retrieved suffices to answer the question. The pair is only informative read
together. StructMem has the highest LoCoMo P4 (0.704) and the highest LoCoMo QA,
so retrieval is not its bottleneck there. On HaluMem its P4 drops to 0.221 while
QA is 0.473, meaning that score is not carried by retrieval. No architecture
leads across all four datasets, and each one's profile is **distributed across
stages**, which is exactly what a single aggregate score conceals.

---

## Repository layout

```
.
├── run.sh                       entry point: check / matrix / smoke / reproduce
├── build_matrix_excel.py        aggregates every scores.json into the failure-stage matrix
├── memory_failure_matrix.xlsx   main results table (16 rows x 109 columns)
├── results_derived_cache.json   cached values derived from large raw outputs, see below
├── requirements-tools.txt       minimal dependencies for rebuilding the tables
├── .env.example                 environment variable template
│
├── scripts/
│   ├── download_data.sh         fetch the four datasets
│   └── export_derived_cache.py  regenerate the cache on a machine holding full outputs
│
├── halumem_experiment/          HaluMem
│   ├── run.py                   entry point
│   ├── eval_*.py                per-backend adapters
│   ├── evaluation.py            official metric scoring
│   ├── probe_halumem.py         P4 / P5 / P5b stage probes
│   ├── qa_attribution.py        per-question failure attribution
│   ├── token_tracker.py         staged cost and latency instrumentation
│   └── results/<run>/           per-run scoring summaries
│
├── locomo_experiment/           LoCoMo (run_locomo.py / probe_locomo.py)
├── longmemeval_experiment/      LongMemEval-S (run_longmem.py / probe_longmem.py)
└── memfail_experiment/          MemFail (four .sh pipelines plus the src/ harness)
```

### What is and is not tracked

**Tracked**: all source code, each run's `*_scores.json`, `*_token_usage.json`,
`*_meta.json`, and `*_probe_detail.jsonl`, plus the four `experiment_results.xlsx`
files.

**Not tracked**:

| Excluded | Reason | How to obtain |
|---|---|---|
| `venv_*/` | 5.9 GB | see the setup section below |
| `*/data/` datasets | about 400 MB, individual files exceed GitHub's 100 MB limit | `bash scripts/download_data.sh` |
| `*_eval_results.jsonl`, `*_detail.jsonl` | per-question raw output, 2.5 GB total, largest single file 77 MB | re-run the experiments |
| `qdrant_*/`, `kuzu_data/` | vector and graph store runtime state, rebuildable | re-run the experiments |
| `.env` | contains API keys | copy `.env.example` and fill it in |

Two families of columns in `build_matrix_excel.py` (each run's memory-store size,
and HaluMem's per-question-type accuracy) could originally only be computed from
those large per-question outputs. So that anyone cloning the repository can still
rebuild the complete matrix, those values are extracted into
`results_derived_cache.json` (about 5 KB): when the raw files are present the
script recomputes as before, and when they are absent it falls back to the cache.
Both paths have been verified to produce identical matrices apart from the
generation timestamp.

---

## Full environment setup

The backends have mutually incompatible dependencies, most obviously `mem0ai`
1.x versus 2.x, so each gets its own virtual environment. Everything installs
under the home or project directory; no sudo, Docker, or Homebrew is required.

```bash
# Shared: Ollama provides embeddings
# Install from https://ollama.com
ollama pull bge-m3

# Mem0 v1 + A-MEM + RAG
python3.12 -m venv venv_mem0v1
./venv_mem0v1/bin/pip install -r halumem_experiment/requirements.txt
./venv_mem0v1/bin/pip install "mem0ai==1.0.11" git+https://github.com/agiresearch/A-mem.git

# Mem0 v2
python3.12 -m venv venv_memfail
./venv_memfail/bin/pip install -r halumem_experiment/requirements.txt
./venv_memfail/bin/pip install "mem0ai>=2.0"

# StructMem: LightMem requires Python < 3.12
python3.11 -m venv ~/structmem_env
git clone https://github.com/zjunlp/LightMem.git ~/LightMem
~/structmem_env/bin/pip install -e ~/LightMem

# Letta: server plus PostgreSQL, several more steps
python3.12 -m venv venv_letta
./venv_letta/bin/pip install letta
# full procedure in halumem_experiment/LETTA_SETUP.md

# Optional: MemOS (needs Neo4j and a Qdrant server), Zep (cloud service)
python3.12 -m venv venv_memos && ./venv_memos/bin/pip install MemoryOS
python3.12 -m venv venv_zep   && ./venv_zep/bin/pip install zep-cloud
```

`bash run.sh check` verifies each of these and skips only the backends whose
environment is missing.

---

## Reproducing the experiments

`bash run.sh reproduce <tag>` works through 15 dataset-by-architecture
combinations (five architectures each on HaluMem, LoCoMo, and LongMemEval)
plus MemFail's four task pipelines. To run a single one:

```bash
# HaluMem x Mem0 v1
cd halumem_experiment
../venv_mem0v1/bin/python run.py --backend mem0 --version v1_repro \
    --skip-users 1 --max-users 1 --top-k 20

# LoCoMo x StructMem
cd locomo_experiment
~/structmem_env/bin/python run_locomo.py --backend structmem \
    --version sm_repro --smoke --top-k 20

# LongMemEval x A-MEM
cd longmemeval_experiment
../venv_mem0v1/bin/python run_longmem.py --backend amem \
    --version amem_repro --max-questions 18 --top-k 20

# MemFail
cd memfail_experiment
./coexisting_facts.sh --run-id repro
```

Every runner supports `--eval-only` (rescore existing results) and `--skip-eval`
(extraction only), so an interrupted run can be resumed.

Probe metrics are computed separately afterwards:

```bash
cd halumem_experiment
../venv_memos/bin/python probe_halumem.py --run mem0_oss-v1_repro
```

Finally, add the new run tags to the `BACKENDS` table at the top of
`build_matrix_excel.py`, regenerate the derived cache with
`python scripts/export_derived_cache.py`, and run `bash run.sh matrix` to fold
them into the matrix.

---

## Metric definitions

Matrix columns are grouped by failure stage. Full definitions live in the
`Definitions` sheet of `memory_failure_matrix.xlsx`.

| Stage | Representative metrics | Meaning |
|---|---|---|
| **Summary** | P1, HaluMem F1 | Was what should be remembered actually recorded, and recorded correctly |
| **Storage** | HaluMem update, LongMemEval KU | After a fact is updated, was the stale value properly superseded |
| **Retrieval** | **P4**, Recall@5, NDCG@5 | Does what was retrieved suffice to answer the question |
| **Reasoning** | **P5** | Share of questions that passed P4 yet were still answered wrong |
| **Memory Performance** | QA accuracy | End-to-end outcome |

P1, P4, and P5 are probes added by this work, because the official metrics do
not cover the retrieval stage. Two quantities must be kept apart:

```
retrieval_ratio = P(failed at retrieval | answered wrong)   conditional; denominator is wrong answers only
P4              = P(retrieved context suffices)             absolute; denominator is all questions
```

The former is an attribution share: two systems can post the same
`retrieval_ratio` while differing tenfold in actual retrieval ability. P4 runs
over **all questions**, which is what makes P4 mean the same thing across the
three datasets and therefore directly comparable.

The subset-score columns are additionally grouped into seven categories by what
kind of memory a correct answer demands (single-point recall; multi-memory
chained; multi-memory parallel; multi-memory mixed; temporal reasoning;
post-update value; abstention and correction), so that comparable subsets from
different datasets can be read side by side. Those groupings are also documented
in the `Definitions` sheet.

---

## Known limitations

These are recorded in the results workbook itself; summarized here:

1. **Small samples.** LoCoMo contributes one conversation (199 QA), HaluMem one
   user (188 QA), LongMemEval 18 questions, MemFail 35 questions. The relative
   ordering across architectures was obtained on exactly the same questions and
   is comparable, but confidence intervals on absolute values are wide, and
   small gaps in a single cell should not be read as architectural superiority.

2. **Batches ② and ③ are incomplete.** Cells marked `RUNNING` indicate a run
   that has not finished. Batch ① (2026-08-14) is the complete one.

3. **Cost columns exist only for batch ②.** Staged token and latency
   instrumentation (the ingest / qa / other buckets in `token_tracker.py`) was
   added on 2026-08-18; earlier runs recorded only `total_tokens`, so their cost
   cells are blank.

4. **Letta is not strictly comparable to the others.** Letta answers using its
   own agent rather than handing retrieved memories to a shared LLM, and it has
   three memory tiers. The matrix therefore carries both a `Letta` row and a
   `Letta (with history)` row, differing in whether the message history visible
   at answering time counts as retrieved context. The gap between the two rows
   is itself a finding.

5. **Judge temperature was not pinned in batch ①**, only from batch ② onward.
   The HaluMem rows for StructMem, A-MEM, and Letta reuse earlier runs and are
   affected by this.

6. **StructMem runs with `pre_compress` and `topic_segment` disabled.** The
   reference configuration requires LLMLingua-2 on CUDA, which this machine
   cannot provide. With them off, the comparison covers the memory mechanism
   itself and excludes the compression module.

---

## Datasets and upstream projects

- HaluMem: https://huggingface.co/datasets/MemTensor/HaluMem (gated, requires accepting the license)
- LoCoMo: https://github.com/snap-research/locomo
- LongMemEval: https://huggingface.co/datasets/xiaowu0162/longmemeval
- MemFail: https://huggingface.co/datasets/ishirgarg/MemFail
  (`memfail_experiment/` is adapted from the official repository; its original README is kept in that directory)
- Mem0: https://github.com/mem0ai/mem0
- A-MEM: https://github.com/agiresearch/A-mem
- LightMem / StructMem: https://github.com/zjunlp/LightMem
- Letta: https://github.com/letta-ai/letta
