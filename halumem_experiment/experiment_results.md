# HaluMem Experiment Results

_Last updated: 2026-08-20 22:17_

## Configuration

| Run | Date | Backend | Extraction LLM | Embed Model | Judge LLM | Users | Sessions |
|-----|------|---------|----------------|-------------|-----------|------:|--------:|
| `mem0_oss-full_user1_atomic` | 2026-05-25 | Mem0 OSS | `gemma-4-E4B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-full_user1_atomic_31b` | 2026-05-25 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-full_user1_gemma431b` | 2026-05-25 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-full_user1_gemma431b_mmr` | 2026-06-26 | Mem0 OSS | `unknown` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-full_user1_gemma431b_rerank` | 2026-06-26 | Mem0 OSS | `unknown` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-full_user1_gemma431b_top5` | 2026-06-26 | Mem0 OSS | `unknown` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 65 |
| `mem0_oss-medium_nchc_gemma4` | 2026-04-27 | Mem0 OSS | `gemma-4-E4B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 20 | 1387 |
| `mem0_oss-smoke_gemma431b` | 2026-05-25 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 1 |
| `mem0_oss-smoke_nchc_gemma4` | 2026-04-27 | Mem0 OSS | `gemma-4-E4B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 1 |
| `mem0_oss-user2_v1_31b` | 2026-06-08 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 2 | 142 |
| `mem0_oss-user2nd_gemma431b_probe` | 2026-08-08 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-user2nd_gemma431b_probe_BROKEN_maxtok2000` | 2026-08-08 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-v1_31b_u2` | 2026-08-14 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-v1_cost_u2` | 2026-08-19 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-v1_cost_u34` | 2026-08-19 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 2 | 139 |
| `mem0_oss-v2_31b_u2` | 2026-08-14 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-v2_cost_u2` | 2026-08-19 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 1 | 77 |
| `mem0_oss-v2_cost_u34` | 2026-08-19 | Mem0 OSS | `gemma-4-31B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 2 | 139 |
| `graphiti-full_user1_gemma31b` | 2026-05-05 | Graphiti | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 65 |
| `graphiti-full_user1_graphiti` | 2026-05-04 | Graphiti | `Llama-3.3-70B-Instruct` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 65 |
| `graphiti-smoke_3sess` | 2026-04-29 | Graphiti | `Llama-3.3-70B-Instruct` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 3 |
| `graphiti-smoke_graphiti` | 2026-04-28 | Graphiti | `gemma-4-E4B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 1 |
| `graphiti-user2_graphiti_fix` | 2026-05-29 | Graphiti | `Llama-4-Scout-17B-16E-Instruct-FP8` | `mxbai-embed-large` | `gemma-4-E4B-it` | 2 | 142 |
| `amem-amem_cost_u2` | 2026-08-20 | A-MEM | `gemma-4-31B-it` | `all-MiniLM-L6-v2` | `gemma-4-E4B-it` | 1 | 77 |
| `amem-full_user1_amem` | 2026-05-26 | A-MEM | `gemma-4-E4B-it` | `all-MiniLM-L6-v2` | `gemma-4-E4B-it` | 1 | 65 |
| `amem-full_user1_amem_31b` | 2026-05-26 | A-MEM | `gemma-4-31B-it` | `all-MiniLM-L6-v2` | `gemma-4-E4B-it` | 1 | 65 |
| `amem-smoke_amem` | 2026-05-26 | A-MEM | `gemma-4-E4B-it` | `all-MiniLM-L6-v2` | `gemma-4-E4B-it` | 1 | 1 |
| `amem-user2nd_gemma431b_probe` | 2026-08-08 | A-MEM | `gemma-4-31B-it` | `all-MiniLM-L6-v2` | `gemma-4-E4B-it` | 1 | 77 |
| `memwave-user2_memwave` | 2026-05-28 | MemWave | `openai/gemma-4-E4B-it` | `bge-m3:latest` | `gemma-4-E4B-it` | 2 | 142 |
| `rag-user1` | 2026-07-07 | RAG | `none (raw turns)` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 65 |
| `letta-u2_ctx` | 2026-08-17 | Letta | `openai-proxy/gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `letta-user1` | 2026-07-07 | Letta | `openai-proxy/gemma-4-E4B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 65 |
| `letta-user2nd_gemma431b_probe` | 2026-08-08 | Letta | `openai-proxy/gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `zep-user0-65s` | 2026-07-31 | Zep Cloud | `unknown` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 65 |
| `zep-user2nd_gemma431b_probe` | 2026-08-08 | Zep Cloud | `unknown` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `memos-user2nd_gemma431b_probe` | 2026-08-08 | MemOS | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `structmem-sm_cost_u2` | 2026-08-19 | StructMem | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `structmem-sm_cost_u34` | 2026-08-20 | StructMem | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 2 | 139 |
| `structmem-u2_upd` | 2026-08-17 | StructMem | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |
| `structmem-user2nd_gemma431b_probe` | 2026-08-09 | StructMem | `gemma-4-31B-it` | `mxbai-embed-large` | `gemma-4-E4B-it` | 1 | 77 |

## Results

| Run | Total Tokens | MP Recall | MP Precision | MP F1 | Interference Acc | Weighted Acc | Upd Correct | Upd Omission | QA Correct | QA Halluc | QA Omission |
|-----|-------------:|----------:|-------------:|------:|-----------------:|-------------:|------------:|-------------:|-----------:|----------:|------------:|
| `mem0_oss-full_user1_atomic` | — | 0.623 | 0.998 | 0.767 | 0.544 | 0.749 | 0.761 | 0.127 | 0.409 | 0.122 | 0.470 |
| `mem0_oss-full_user1_atomic_31b` | — | 0.829 | 0.998 | 0.906 | 0.424 | 0.930 | **0.866** | 0.042 | 0.427 | 0.098 | 0.476 |
| `mem0_oss-full_user1_gemma431b` | — | 0.803 | 0.999 | 0.890 | 0.408 | 0.930 | 0.838 | 0.056 | 0.427 | 0.104 | 0.470 |
| `mem0_oss-full_user1_gemma431b_mmr` | 286,796 | 0.796 | 0.999 | 0.886 | 0.448 | 0.936 | **0.866** | 0.085 | 0.311 | 0.152 | 0.537 |
| `mem0_oss-full_user1_gemma431b_rerank` | 289,288 | 0.789 | 0.995 | 0.880 | 0.472 | 0.934 | 0.859 | 0.077 | 0.329 | 0.152 | 0.518 |
| `mem0_oss-full_user1_gemma431b_top5` | 133,439 | 0.803 | 0.999 | 0.890 | 0.448 | 0.930 | 0.859 | 0.056 | 0.323 | 0.104 | 0.573 |
| `mem0_oss-medium_nchc_gemma4` | — | 0.520 | 0.999 | 0.684 | 0.545 | 0.756 | 0.672 | 0.203 | 0.390 | 0.103 | 0.508 |
| `mem0_oss-smoke_gemma431b` | — | 0.733 | **1.000** | 0.846 | 0.000 | **1.000** | 0.000 | **0.000** | **1.000** | **0.000** | **0.000** |
| `mem0_oss-smoke_nchc_gemma4` | — | 0.867 | **1.000** | 0.929 | 0.000 | **1.000** | 0.000 | **0.000** | 0.667 | **0.000** | 0.333 |
| `mem0_oss-user2_v1_31b` | 1,671,316 | 0.330 | 0.993 | 0.495 | 0.724 | 0.887 | 0.336 | 0.345 | 0.338 | 0.114 | 0.548 |
| `mem0_oss-user2nd_gemma431b_probe` | 947,868 | 0.496 | 0.993 | 0.661 | 0.558 | 0.805 | 0.444 | 0.241 | 0.367 | 0.096 | 0.537 |
| `mem0_oss-user2nd_gemma431b_probe_BROKEN_maxtok2000` | 941,152 | 0.251 | 0.995 | 0.401 | 0.857 | 0.916 | 0.179 | 0.481 | 0.282 | 0.170 | 0.548 |
| `mem0_oss-v1_31b_u2` | 944,385 | 0.486 | 0.998 | 0.654 | 0.565 | 0.818 | 0.370 | 0.321 | 0.372 | 0.085 | 0.543 |
| `mem0_oss-v1_cost_u2` | 956,959 | 0.484 | 0.998 | 0.652 | 0.592 | 0.809 | 0.426 | 0.284 | 0.362 | 0.085 | 0.553 |
| `mem0_oss-v1_cost_u34` | 1,711,606 | 0.513 | 0.997 | 0.678 | 0.569 | 0.870 | 0.525 | 0.211 | 0.297 | 0.142 | 0.561 |
| `mem0_oss-v2_31b_u2` | 1,256,333 | 0.729 | 0.998 | 0.843 | 0.333 | 0.919 | 0.716 | 0.142 | 0.351 | 0.144 | 0.505 |
| `mem0_oss-v2_cost_u2` | 1,258,857 | 0.711 | 0.998 | 0.830 | 0.367 | 0.919 | 0.741 | 0.136 | 0.351 | 0.101 | 0.548 |
| `mem0_oss-v2_cost_u34` | 2,277,985 | 0.792 | 0.998 | 0.883 | 0.365 | 0.932 | 0.766 | 0.090 | 0.353 | 0.153 | 0.494 |
| `graphiti-full_user1_gemma31b` | — | 0.075 | 0.974 | 0.140 | 0.904 | 0.052 | 0.310 | 0.507 | 0.463 | 0.140 | 0.396 |
| `graphiti-full_user1_graphiti` | — | 0.071 | 0.969 | 0.132 | 0.952 | 0.046 | 0.239 | 0.606 | 0.360 | 0.134 | 0.506 |
| `graphiti-smoke_3sess` | — | 0.060 | 0.917 | 0.113 | 0.000 | 0.261 | 0.000 | **0.000** | 0.444 | **0.000** | 0.556 |
| `graphiti-smoke_graphiti` | — | 0.333 | **1.000** | 0.500 | 0.000 | **1.000** | 0.000 | **0.000** | 0.333 | **0.000** | 0.667 |
| `graphiti-user2_graphiti_fix` | — | 0.006 | **1.000** | 0.013 | 0.978 | 0.015 | 0.000 | 0.987 | 0.224 | 0.009 | 0.767 |
| `amem-amem_cost_u2` | 5,624,886 | 0.829 | 0.975 | 0.896 | 0.109 | 0.659 | 0.790 | 0.117 | 0.590 | 0.176 | 0.234 |
| `amem-full_user1_amem` | — | 0.712 | 0.995 | 0.830 | 0.600 | 0.826 | 0.507 | 0.176 | 0.555 | 0.177 | 0.268 |
| `amem-full_user1_amem_31b` | 2,076,257 | 0.692 | 0.994 | 0.816 | 0.624 | 0.838 | 0.549 | 0.162 | 0.561 | 0.165 | 0.274 |
| `amem-smoke_amem` | — | **1.000** | **1.000** | **1.000** | 0.000 | 0.833 | 0.000 | **0.000** | 0.667 | 0.333 | **0.000** |
| `amem-user2nd_gemma431b_probe` | 5,642,742 | 0.823 | 0.966 | 0.889 | 0.082 | 0.657 | 0.735 | 0.123 | 0.553 | 0.213 | 0.234 |
| `memwave-user2_memwave` | 175,647 | 0.357 | 0.998 | 0.526 | 0.651 | 0.873 | 0.000 | **0.000** | 0.230 | 0.009 | 0.761 |
| `rag-user1` | 182,068 | 0.710 | 0.994 | 0.828 | 0.616 | 0.823 | 0.479 | 0.134 | 0.360 | 0.110 | 0.530 |
| `letta-u2_ctx` | 5,782,036 | 0.629 | 0.991 | 0.769 | 0.367 | 0.063 | 0.543 | 0.222 | 0.580 | 0.399 | 0.021 |
| `letta-user1` | — | 0.488 | 0.953 | 0.645 | 0.544 | 0.056 | 0.289 | 0.592 | 0.537 | 0.293 | 0.171 |
| `letta-user2nd_gemma431b_probe` | 5,475,397 | 0.673 | 0.975 | 0.796 | 0.286 | 0.059 | 0.556 | 0.235 | 0.580 | 0.372 | 0.048 |
| `zep-user0-65s` | 139,210 | 0.661 | 0.989 | 0.792 | 0.472 | 0.602 | 0.415 | 0.169 | 0.378 | 0.116 | 0.506 |
| `zep-user2nd_gemma431b_probe` | 152,841 | 0.621 | 0.993 | 0.764 | 0.381 | 0.675 | 0.261 | 0.298 | 0.351 | 0.064 | 0.585 |
| `memos-user2nd_gemma431b_probe` | 1,941,768 | 0.823 | 0.978 | 0.894 | 0.265 | 0.633 | 0.660 | 0.099 | 0.330 | 0.096 | 0.574 |
| `structmem-sm_cost_u2` | 1,229,104 | 0.064 | **1.000** | 0.121 | **1.000** | 0.932 | 0.019 | 0.901 | 0.234 | 0.122 | 0.644 |
| `structmem-sm_cost_u34` | 2,230,806 | 0.049 | **1.000** | 0.093 | 0.996 | 0.893 | 0.067 | 0.839 | 0.206 | 0.064 | 0.731 |
| `structmem-u2_upd` | 1,225,702 | 0.078 | **1.000** | 0.145 | **1.000** | 0.932 | 0.006 | 0.889 | 0.234 | 0.133 | 0.633 |
| `structmem-user2nd_gemma431b_probe` | 1,536,018 | 0.811 | 0.992 | 0.893 | 0.211 | 0.740 | 0.525 | 0.204 | 0.473 | 0.090 | 0.436 |

> **Bold** = best value in each column.
