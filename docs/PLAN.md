# Hill-climb log

Goal: the most simultaneous users one DGX Spark can serve Inkling 2 Mini to, at a usable speed.
Every experiment changes one thing from the current best. Numbers come from `bench/sweep.py`
(raw lines in `raw/sweep.jsonl`); boot logs are in `raw/`. Failures stay in this log.

## Targets and constraints

- **Metric:** the largest concurrency N (closed loop, N users each sending back-to-back requests) whose
  slowest users (P10 per-request decode) stay at or above **10 tok/s**, with TTFT p95 ≤ 5 s and no failed
  requests. Also reported: the 20 tok/s and 5 tok/s tiers, aggregate tok/s, TTFT and the longest pause.
- **Workload `chat`:** a fixed panel of 16 long-form instructions, the model's default reasoning effort,
  temperature 1.0 (the checkpoint's `generation_config.json`), natural EOS, replies capped at 1,024
  tokens; reasoning tokens count as output. Draft acceptance depends on content and temperature, so the
  panel never changes. Workload `long`: a unique ~8K-token document per request plus a retrieval question.
- **Lossless only:** the trained NVFP4 / MXFP8 weights as released. No loop-pass or shared-KV tricks (the
  checkpoint is fixed-loop, `is_varloop: false`; both vLLM and SGLang refuse those options for it).
- **Machine:** one DGX Spark (GB10, 128 GB unified LPDDR5X), a shared development box. Its other
  services keep about 28 GB of CPU memory in use; every GPU workload was stopped for these runs.

## Machine

| Item | Value |
|---|---|
| GPU | NVIDIA GB10, sm_121, driver 580.178.04 (CUDA 13.0) |
| Memory | 121 GiB unified, ~92 GiB available with the GPU workloads stopped |
| Copy bandwidth (read + write, 2 GiB buffers) | 242 GB/s |
| Read bandwidth (sum over 2 GiB BF16) | 250 GB/s |
| BF16 GEMM 4096³ / 8192³ | 93.7 / 96.5 TFLOPS |
| SM clock under load | 2,190 MHz (not the degraded 513 MHz state) |

Measured with `bench/bandwidth.py` in the vLLM image.

## Log

Numbers: aggregate tok/s / per-user p50 / per-user p10 (slowest 10%) at each concurrency, workload
`chat`. "Cap@10" is the largest measured level whose p10 ≥ 10 tok/s with TTFT p95 ≤ 5 s.

### 2026-11-21 08:00, vLLM baseline (`vllm-base-s64-spec7`): DSpark 7 drafts, 64 seqs, 0.70 memory

The recipe page's command plus `--max-num-seqs 64 --gpu-memory-utilization 0.70`. Startup 317 s
(weights 91 s through mmap on ext4, FlashInfer autotune and graph capture ~2 min). Boot log: attention
on FlashAttention 2, KV cache **858,464 tokens** (57.3 GiB), maximum concurrency 3.27× at 262,144.

| Users | 1 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aggregate | 40 | 79 | 126 | 206 | 326 | 427 | 440 |
| Per user p50 | 41.7 | 21.3 | 17.5 | 14.4 | 12.2 | 10.0 | 8.0 |
| Per user p10 | 36.8 | 18.9 | 14.3 | 12.1 | 10.0 | 8.3 | 6.4 |

DSpark acceptance 2.6-2.7 tokens per step at temperature 1.0, falling fast by position
(0.64, 0.39, 0.25, 0.16, 0.11, 0.08, 0.05). c1 is at the ceiling for that acceptance (41.4); c32 is 65%
of it, c64 53% (512 verify rows per step: compute co-binds). TTFT was inflated by Triton kernels JIT-
compiling for each new batch shape during the run (13 "JIT compilation during inference" warnings), so
the sweep now warms each level up before timing it.

### 08:07, control rerun (`vllm-ctl-spec7`), warm caches and per-level warm-up

c16 204 / 14.5 / 11.7, c32 330 / 11.8 / 9.7, c48 416 / 10.2 / 8.2, c64 445 / 7.9 / 6.5. Within 1-3% of
the baseline: run-to-run noise here is about ±3%, and **cap@10 sits at 32 users** (p10 9.7-10.0).
TTFT p50 stays 2-2.5 s under load although an idle request's first token arrives in 0.17 s.

### 08:27, DSpark with 3 or 5 drafts (`vllm-spec3`, `vllm-spec5`): refused

The image refuses any other draft count: `ValueError: Inkling2 DSpark requires num_speculative_tokens
= 7 (the checkpoint's block size - 1)`. Rejected without a measurement; the lever exists only in
llama.cpp (`--spec-draft-n-max`) and SGLang (`--speculative-num-draft-tokens`).

### 08:31, speculation off (`vllm-nospec`, 128 seqs)

| Users | 8 | 16 | 32 | 64 | 96 |
|---|---:|---:|---:|---:|---:|
| Aggregate | 114 | 163 | 242 | 370 | 474 |
| Per user p50 | 14.8 | 10.7 | 8.1 | 6.3 | 5.3 |
| Per user p10 | 14.4 | 10.4 | 7.8 | 6.0 | 5.1 |

c8 is 97% of the plain-decode ceiling and c16-c96 exceed it: routing is not uniform, so fewer experts
are touched per step than the uniform model assumes. **Rejected for the 10 tok/s target** (cap 16 against
32), but it is the better choice for raw throughput past ~64 users (474 tok/s at 96 users against DSpark's
445 at 64) and for the 5 tok/s tier, and its TTFT is lower.

### 09:04, scheduler budget 8192 (`vllm-spec7-bt8192`)

The boot warning suggests raising `--max-num-batched-tokens` for the draft slots. c16 205 / 14.7 / 12.4,
c32 329 / 12.3 / 9.8, c48 413 / 10.1 / 8.4: inside noise. **Rejected**; 4096 stays.

### 09:20, SGLang control (`sglang-ctl-dspark8`): the cookbook's verified Spark DSpark recipe, 64 running

`--tp 1 --ep 1 --attention-backend fa4 --page-size 128 --mem-fraction-static 0.45`, DSpark with an
8-token verify block and the four `SGLANG_OPT_*` DSpark switches, `--max-running-requests 64`. Startup
5.8 min (prefill and verify CUDA graph capture ~3 min). Pools: full-attention KV 374,400 tokens, but the
**sliding-window pool only 37,376 tokens** (`--swa-full-tokens-ratio` default), which at a 1,024-token
window is room for ~36 requests.

| Users | 16 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|
| Aggregate | 195 | 309 | 381 | 390 |
| Per user p50 | 14.2 | 11.1 | 8.9 | 7.5 |
| Per user p10 | 11.8 | 9.0 | 7.4 | 5.6 |
| TTFT p50 / p95 s | 0.74 / 0.78 | 1.03 / 1.04 | 1.03 / 6.5 | 1.25 / 21.9 |
| Longest pause p95 s | 0.31 | 0.59 | 7.6 | 24.4 |

Close to vLLM at 16 users (195 vs 204) with a far better TTFT (0.74 s vs 2.5 s), 6% behind at 32, and at
48-64 the window pool fills (`window_1024 token usage: 0.85` at 48) and requests stall for up to 24 s.

### 09:44, SGLang speculation off (`sglang-nospec`)

c8 112 / 14.4 / 14.4, c16 163 / 10.7 / 10.4: identical to vLLM without speculation (114 and 163), so plain
decode is bandwidth-bound on both engines. The higher levels were skipped (they would repeat vLLM's curve).

### 09:55, SGLang 4-token verify block (`sglang-draft4`): refused

`ValueError: DSpark speculative_num_draft_tokens must equal gamma + 1 (= 8 for gamma=7)`. Like vLLM, SGLang
serves the checkpoint's DSpark only at its trained block. Rejected without a measurement.

### 09:55, SGLang static pool 0.65 (`sglang-mem065`)

Pools double: full KV 715,904 tokens, window pool 71,552. c16 194 / 13.7 / 11.4, c32 304 / 10.5 / 8.8,
c48 384 / 9.1 / 7.3 with **TTFT p95 1.0 s and longest pause 0.58 s (control: 6.5 s and 7.6 s)**, c64 below.
Throughput is unchanged; the stalls are gone. **Kept** (SGLang's best).

### 10:15, SGLang window pool ratio 0.3 at the cookbook's 0.45 (`sglang-swa03`)

`--swa-full-tokens-ratio 0.3`: window pool 73,088 tokens, full pool 243,712 (control: 37,376 and 374,400).
c32 308 / 10.6 / 9.1, c48 378 / 8.7 / 7.3 with TTFT p95 1.1 s and longest pause 0.52 s. The same cure as
0.65 memory with 19 GB less static memory, at the cost of a smaller full-attention pool (243K tokens,
~7 users at 32K). Both kept as options: 0.65 is the default (more long-context room on a dedicated
Spark), the ratio is the fix where memory is shared. c64 426.5 / p10 6.4, matching 0.65 memory (419).

### 10:32, vLLM finalist, finer steps (`vllm-best-fine`): DSpark 7, 64 seqs, 0.70, chunk 4096

| Users | 20 | 24 | 28 | 36 | 40 |
|---|---:|---:|---:|---:|---:|
| Aggregate | 237 | 279 | 306 | 353 | 369 |
| Per user p50 | 13.5 | 13.3 | 12.3 | 11.1 | 10.6 |
| Per user p10 | 11.1 | 10.8 | 10.4 | 9.2 | 9.0 |
| TTFT p50 / p95 s | 2.56 / 3.36 | 0.63 / 0.84 | 1.80 / 2.72 | 2.41 / 3.00 | 1.87 / 2.97 |

The 10 tok/s edge sits between 28 (p10 10.4) and 32 (9.7-10.0 in two earlier runs). **28 users** is the
headline; 32 is on the line. GSM8K gate (250 questions, temperature 0, 32 at once): **97.2%**, 1
truncated, 0 errors.

### 11:01, vLLM long-context workload (`vllm-best-long8k`)

~8K-token unique document per request, 256-token answer, all users arriving at once.

| Users | 8 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|
| Aggregate output tok/s | 28.6 | see raw | 46.8 | 46.3 |
| TTFT p50 / p95 s | 7.4 / 32.1 | 7.5 / 43.0 | 23.8 / 68.3 | 48.2 / 89.4 |
| Retrieval correct | 16/16 | 32/32 | 48/48 | 64/64 |

Prefill, not memory, binds: the engine log shows **~3,200 prompt tok/s** at most, so each new 8K prompt
costs ~2.5 s of the whole GPU and every running stream pauses while it is chunked in. No level meets the
5 s TTFT limit with a burst arrival; see the low-concurrency follow-up below for the usable range.

### 11:16, vLLM FP8 KV cache (`vllm-fp8kv-long8k`)

`--kv-cache-dtype fp8`: pool 1,274,526 tokens (BF16: 858K, +49%). Long workload c8 33.3, c16 45.6, c24
43.4, c32 48.9 tok/s with the same TTFTs as BF16, retrieval 100%. GSM8K **97.2%** (same as BF16).
Lossless on these checks but no speed gain at 8K, where prefill binds. Kept as an option (`KV=fp8`) for
many users at very long context; not the default.

### 11:37, SGLang finalist, finer steps (`sglang-best-fine`): 0.65 static memory

| Users | 20 | 24 | 28 | 36 |
|---|---:|---:|---:|---:|
| Aggregate | 230 | 262 | 285 | 320 |
| Per user p10 | 11.0 | 10.1 | 9.5 | 8.2 |
| TTFT p50 / p95 s | 0.68 / 0.72 | 0.66 / 0.75 | 0.74 / 0.79 | 0.80 / 0.93 |

**24 users** at the 10 tok/s floor, 4 fewer than vLLM, with steady sub-second TTFT at every level.

### 12:04, SGLang long-context workload (`sglang-best-long8k`), 0.65 static memory

| Users | 8 | 16 | 32 |
|---|---:|---:|---:|
| Aggregate output tok/s | 49.5 | 53.1 | 56.4 |
| Per user p10 | 9.7 | 5.7 | 2.7 |
| TTFT p50 / p95 s | 11.0 / 19.2 | 23.1 / 38.5 | see raw |
| Retrieval correct | 16/16 | 32/32 | 64/64 |

**SGLang handles 8K prompts much better than vLLM:** at 8 users 49.5 vs 28.6 tok/s, p10 9.7 vs 3.0 and
TTFT p95 19 s vs 32 s. Its prefill is roughly twice as fast (eight 8K prompts, 64K tokens, cleared in
~11 s: ~5,800 tok/s).

### 12:40, both engines at 1, 2 and 4 users on 8K prompts (`*-long8k-low`)

| Users | vLLM TTFT p50 / p95 s | vLLM p10 tok/s | SGLang TTFT p50 / p95 s | SGLang p10 tok/s |
|---:|---:|---:|---:|---:|
| 1 | 7.85 / 11.6 | 52.5 | 2.52 / 2.54 | 51.7 |
| 2 | 5.04 / 7.08 | 26.9 | 3.72 / 4.81 | 26.6 |
| 4 | 4.59 / 13.5 | 10.9 | 7.34 / 9.64 | 19.5 |

vLLM needs 7.9 s to start one 8K reply (~1,000 prompt tok/s for a lone request); SGLang 2.5 s. With every
user arriving at once, SGLang meets the SLO up to **2 users** and vLLM at none. Steady traffic with
staggered arrivals and a shared system prompt (prefix cache) does much better than this burst test;
those were not measured. **For long-prompt (agent, RAG) traffic, run SGLang.**

### 12:17, TokenSpeed PR #3682 as a Python overlay (`ts-pr3682-dspark`): failed to start

`ModuleNotFoundError: No module named 'tokenspeed_engine.scheduler'`. The PR is 156 commits newer than the
newest published image (nightly-c47e7a) and its runtime needs the newer Rust/C++ engine. Rebuilt from the
PR's source instead (`engines/tokenspeed/build.sh`), see below.

### 12:17, llama.cpp b12600 with Unsloth's DSpark drafter (`llama-iq4-dspark3-s32`): refused

`check_tensor_dims: tensor 'markov_w1.weight' has wrong shape; expected 256, 200058, got 256, 201024`.
Unsloth's drafter is laid out for Unsloth's fork; mainline's DSpark support predates `inkling2_loop`.

### 12:18, llama.cpp b12600, IQ4_NL, no speculation, 32 slots x 16K (`llama-iq4-nospec-s32`)

| Users | 1 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| Aggregate | 35.8 | 74.6 | 109.7 | 137.7 |
| Per user p10 | 35.9 | 9.3 | 7.0 | 3.9 |
| TTFT p50 / p95 s | 0.20 / 0.27 | 1.06 / 1.23 | 0.94 / 2.39 | 1.34 / 4.05 |

Under the 10 tok/s floor from 8 users; a third of vLLM's aggregate at 32.

### 12:50, Unsloth llama.cpp fork b12588-mix + UD-Q4_K_XL + DSpark, 3 drafts (`llama-ud4-dspark3-s32`)

The drafter loads and accepts (2.4-3.0 tokens per step at one user), but c1 28.2 tok/s (mainline without
speculation: 35.8) and c8 **42.7 tok/s, p10 4.9** (mainline: 74.6). Speculation costs throughput as soon
as slots run in parallel (not profiled; the draft and verify batches of 8 slots are the likely cost).
Part of the c1 gap may be the different quant (UD-Q4_K_XL vs IQ4_NL). **Rejected**; the 16/32-user levels and the
7-draft variant were skipped (they could not reach the floor).

### 13:00, GSM8K gate for llama.cpp b12600 IQ4_NL (`llama-iq4-nospec`)

**97.2%** (243/250), mean 487 completion tokens, 1 truncated, 0 errors: the same score as vLLM and SGLang
on the NVFP4 weights, so the IQ4_NL requantisation and mainline's looped-layer path (without #32704) hold
up on this check. Wall time 1,477 s at 32 parallel, against 337-397 s on vLLM and SGLang.

### 13:24, TokenSpeed source build, first boot (`ts-pr3682-dspark`): no sm_121 kernels

The source-built engine imports and the model loads (`Inkling2ForConditionalGenerationMTP`, 54 GB free
after weights), then the first forward fails: `BatchQKApplyRotaryPosIdsCosSinCache failed with error code
no kernel image is available for execution on the device`. The image's precompiled tokenspeed-kernel
targets sm_90a/100a/103a only. `build.sh` now also compiles tokenspeed-kernel for sm_121a.

### 13:35, admission cap 32 on vLLM (`vllm-cap32`)

c32 315 tok/s, p10 9.4, TTFT p95 3.7 s; c48 (16 users queued) 329 tok/s, p10 **9.7** (uncapped 8.2),
TTFT p50 / p95 31 / 76 s. The cap holds the admitted users' speed and makes the rest wait. Three runs at
32 users (10.0, 9.7, 9.4) sit on or under the floor, so the shipped caps are **28 (vLLM)** and **24 (SGLang)**.

### 13:55, the shipped default, `./start.sh` with no overrides (`vllm-default`, MAX_SEQS=28)

c28 **297 tok/s, p50 12.0, p10 10.4, TTFT p50 / p95 3.1 / 4.4 s, SLO met**, matching the finalist sweep
(306 tok/s, p10 10.4).

### 14:12, TokenSpeed with sm_121a kernels: Triton shared memory

The CUDA kernels now run (RoPE passes), then the DSpark drafter's `triton_skinny.skinny_mm` fails:
`OutOfResources: shared memory, Required: 147480, Hardware limit: 101376`. Its tiles are sized for
datacenter Blackwell (228 KB per SM). `engines/tokenspeed/sm12x-skinny-mm.patch` makes
`skinny_mm_supported()` false on sm_12x, which sends those calls to the generic GEMM the PR already uses
on other platforms.

### 14:24, TokenSpeed with the patch, 64 requests, 262K context: KV pool too small

Warm-up and CUDA graph capture (batches 64-8) pass. Then `num_lcm_blocks=3628 cannot admit one token
with the configured inkling cache scheduler limits`: the pool left at 0.70 memory cannot hold the
per-request working set the scheduler reserves for 64 requests. Retried with 32 requests, 64K context
and 0.80 memory.

### 14:32, TokenSpeed serving (`ts-pr3682-s32`): 32 requests, 64K context, 0.80 memory

Starts in 564 s and passes the smoke test: pool 424,832 tokens, CUDA graphs for 32-4, DSpark acceptance
2.65-2.70 (the same as vLLM). c1 **19.4 tok/s** (vLLM 39.8), c16 **65.7 tok/s, p10 3.6** (vLLM 204).
The engine log shows ~80 tok/s of generation at 16 requests. Its kernels are tuned for datacenter
Blackwell and AMD; on sm_121 the patched GEMM path and untuned Triton configs run at about a third of
vLLM's speed (not profiled further). 24 and 32 users were skipped. **Rejected for this target**;
kept as a working fourth engine (`./start.sh tokenspeed`).

GSM8K gate (`ts-pr3682`): **96.8%** (242/250), mean 444 completion tokens, 1 truncated, 0 errors. One
question below the other engines' 97.2%, inside the ±1.1% sampling noise of 250 questions, so the
sm_12x GEMM fallback does not visibly change outputs.

## Status

Plateau reached for this machine and these engines.

- **Shipped default:** vLLM, DSpark 7, 0.70 memory, chunk 4096, admission cap 28: 28 users at ≥ 10 tok/s
  (verified with the shipped settings: 297 tok/s, p10 10.4), GSM8K 97.2%.
- **Long prompts:** SGLang (0.65 static memory, cap 24), the only engine within the SLO on 8K prompts.
- **Remaining levers, not tried:** prefix caching with a shared system prompt (expected to help long prompts
  most), staggered arrivals, FP8 KV on SGLang, and a reboot before each engine to remove fragmentation
  effects. DSpark draft length, loop sharing and TensorFold are blocked by the checkpoint or the engine.
