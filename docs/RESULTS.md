# Results: Inkling 2 Mini on one DGX Spark, 2026-11-21

One DGX Spark (GB10, 128 GB unified, 242 GB/s measured copy bandwidth, 2,190 MHz under load), every
other GPU workload stopped, the client on the same machine. Method and definitions are in the README;
every run, kept or rejected, is in [PLAN.md](PLAN.md); raw lines and boot logs are in [raw/](raw/).

## Users at once, chat workload

Capacity = the most simultaneous users (closed loop, back-to-back requests, ~1K-token replies at the
model's default reasoning effort, temperature 1.0) whose **slowest 10% of replies** still decode at the
floor, with TTFT p95 ≤ 5 s and no failed request.

| Engine, config | ≥ 20 tok/s | **≥ 10 tok/s** | ≥ 5 tok/s | Peak aggregate tok/s | TTFT p95 at the 10 tok/s edge | GSM8K (250, T=0) |
|---|---:|---:|---:|---:|---:|---:|
| **vLLM**, DSpark 7, 0.70 memory | 1 (4 at 18.9) | **28** (32 on the line) | 64 | 445 at 64 | 2.7 s | 97.2% |
| vLLM, no speculation | - | 16 | 96 | 474 at 96 | 4.8 s | not run |
| **SGLang**, DSpark 8, 0.65 static memory | not run (levels 16+) | **24** | 48 | 419 at 64 | 0.75 s | 97.2% |
| SGLang, no speculation | - | 16 | 16 | 163 at 16 (higher levels not run) | 0.49 s | not run |
| TokenSpeed, PR #3682 source build + sm_12x patch, DSpark, 32 seqs | 1 (16.9 at 1) | 1 | 1 | 66 at 16 | 0.58 s | 96.8% |
| llama.cpp b12600, IQ4_NL GGUF, no spec | 1 | 1 | 16 | 138 at 32 | 0.27 s | 97.2% |
| llama.cpp Unsloth fork, UD-Q4_K_XL + DSpark 3 | 1 | 1 | 1 | 43 at 8 (stopped) | - | not run |
| TensorFold | cannot serve Inkling 2 Mini yet ([ENGINES.md](ENGINES.md)) | | | | | |

- **vLLM is the default.** Four more users than SGLang at the 10 tok/s floor (p10 10.4 vs 9.5 at 28),
  and the highest throughput with speculation on.
- **SGLang starts replies fastest:** TTFT p95 under 1 s at every chat level up to 40 users, vs 2-3 s for
  vLLM. Pick it when time-to-first-token matters more than four extra users.
- **Speculation decides the answer.** Without DSpark both engines fall under 10 tok/s past 16 users;
  with it they reach 24-28. Without it vLLM has the highest raw throughput (474 tok/s at 96 users, 5.1
  tok/s each): the choice for batch jobs, not for people waiting on a stream.
- **Quality is identical:** 97.2% GSM8K on vLLM with BF16 KV, vLLM with FP8 KV and SGLang.

## Users at once, long prompts (8K tokens each, all arriving together)

| Engine | Users within SLO | TTFT p50 / p95 at 1 user | at 4 users | Aggregate at 8 users |
|---|---:|---:|---:|---:|
| **SGLang** | **2** | 2.5 / 2.5 s | 7.3 / 9.6 s | 49.5 tok/s, p10 9.7 |
| vLLM | 0 | 7.9 / 11.6 s | 4.6 / 13.5 s | 28.6 tok/s, p10 3.0 |
| vLLM, FP8 KV | 0 | not run | not run | 33.3 tok/s |

Prefill binds here, not memory or decode: vLLM reaches ~3,200 prompt tok/s batched (~1,000 for a lone
request), SGLang ~5,800. Every new 8K prompt pauses the running streams while it is chunked in.
Retrieval was 100% correct in every long run. **Run SGLang for agent and RAG traffic with long prompts.**
FP8 KV grows vLLM's pool from 858K to 1.27M tokens with no GSM8K loss but no speed gain at 8K.

## Admission cap

`MAX_SEQS` (vLLM `--max-num-seqs`, SGLang `--max-running-requests`) is the number of requests decoded at
once; the rest wait in the queue. Measured on vLLM with the cap at 32 (`vllm-cap32`):

| Users offered | Cap | Aggregate tok/s | Per user p10 | TTFT p50 / p95 s |
|---:|---:|---:|---:|---:|
| 48 | 64 (no cap in effect) | 416 | 8.2 | 2.0 / 3.2 |
| 48 | 32 | 329 | 9.7 | 31.0 / 75.6 |
| 32 | 32 | 315 | 9.4 | 3.5 / 3.7 |

With a cap, admitted users keep their streaming speed and the others wait for a slot. Three runs at 32
users gave p10 9.4, 9.7 and 10.0, so 32 sits just under the floor and **28 is the shipped cap**
(`engines/vllm/defaults.env`; 24 for SGLang). Raise it when total throughput matters more than per-user
speed.

## What did not work

| Tried | Result |
|---|---|
| DSpark with fewer drafts (vLLM 3 or 5, SGLang block 4) | refused by both engines: the checkpoint's DSpark serves only its trained block (8) |
| Loop share-cache / fewer loop passes | not allowed: the checkpoint is fixed-loop (`is_varloop: false`); both engines refuse it |
| vLLM `--max-num-batched-tokens 8192` | inside noise of 4096 (204.8 vs 204.4 at 16, 328.7 vs 329.5 at 32) |
| SGLang cookbook memory 0.45 | sliding-window pool of 37K tokens fills at 48 users: TTFT p95 6.5 s, 22 s at 64 |
| TokenSpeed PR as a Python overlay on the nightly image | `No module named 'tokenspeed_engine.scheduler'`; rebuilt from source instead |
| TokenSpeed with the image's kernels / unpatched / 64 requests | no sm_121 kernel image; Triton GEMM needs 147 KB shared memory (GB10: 99 KB); KV pool cannot admit one token. Fixed in `engines/tokenspeed/` |
| TokenSpeed once serving | 19.4 tok/s at 1 user, 65.7 at 16 (p10 3.6): about a third of vLLM on sm_121 |
| llama.cpp mainline with Unsloth's DSpark GGUF | `markov_w1.weight` shape mismatch (vocab 201024 vs 200058) |
| llama.cpp Unsloth fork with DSpark | works, but 42.7 tok/s at 8 users vs 74.6 without speculation |
| TensorFold | no Inkling 2 Mini model yet (plan PR ashhart/TensorFold#4274) |

## Against the ceiling

At 32 users vLLM with DSpark reaches 66% of the bandwidth ceiling, falling to 53% at 64 as the 8 verify
rows per user make compute co-bind. Plain decode slightly beats the uniform-routing ceiling (routing is
skewed). See [THEORY.md](THEORY.md#measured-against-the-ceiling).

## Not verified

- One run per configuration and level (2 rounds each, warm caches, warm-up level first); no reboot between
  engines, so later runs may have lost a few percent to memory fragmentation. Differences under ~3% are
  noise.
- Burst arrivals only; staggered arrivals and prefix-cache reuse (shared system prompts) were not measured.
- The client ran on the Spark (~1% CPU).
- Contexts beyond 8K, and vision inputs, were not benchmarked.
