# Ceilings: how many users can one Spark serve?

Decode on GB10 is memory-bandwidth bound. A decode step reads the weights it needs once for every
stream in the batch, plus each stream's KV cache. `bench/ceiling_inkling.py` does the arithmetic below;
numbers here use the measured 242 GB/s copy bandwidth (`docs/PLAN.md`).

## Bytes read per decode step

| Part | Bytes | Why |
|---|---:|---|
| Attention, gates, norms (BF16) on 48 layer visits | ~1.8 GB | layers 6-17 run three passes, and their weights are re-read on each pass (24 MB L2) |
| LM head, 201,024 × 2,304 BF16 | 0.93 GB | read every step |
| DSpark draft (3 layers + Markov head) | 0.64 GB | one block forward per step, shared by the batch |
| Routed experts | up to 29.4 GB | each visit reads the union of its rows' top-8 of 128 experts: 44 NVFP4 visits × 0.573 GB + 4 MXFP8 visits × 1.05 GB when all are touched |
| KV per stream | 36 KiB × context + 0.17 GiB | 8 full-attention caches (layers 5, 23 once; 11, 17 three times) + the draft's; 42 sliding-window caches of 1,024 tokens |

The expert term is what makes this model different from a dense one at concurrency. One stream at
batch 1 touches 8 of 128 experts per visit, so a step reads ~4.5 GB. With r rows per step the touched
share is 1 − (120/128)^r under uniform routing: 64% at 16 rows, 87% at 32, 98% at 64. Plain decode
therefore stops scaling early: at 32 streams a step already reads 30 GB and each user gets ~6.6 tok/s.

Speculative decoding changes the arithmetic. DSpark verifies a block of 8 tokens per stream per step
(rows = 8 × streams), so experts saturate at only 8 streams, but each step then emits the accepted
length `a` (measured 2.6-2.7 at temperature 1.0 with 7 drafts) for every stream, and the per-step KV
and weight reads are shared by `a` tokens. Above ~64 streams the 512+ verify rows make compute
co-bind with bandwidth, so the real curve bends below this table.

## Decode ceiling at 2K context, a = 2.65 (measured acceptance)

| Streams | Plain decode, aggregate (per user) | DSpark, aggregate (per user) |
|---:|---:|---:|
| 1 | 50 (50.2) | 41 (41.4) |
| 4 | 93 (23.2) | 85 (21.3) |
| 8 | 117 (14.6) | 150 (18.7) |
| 16 | 151 (9.4) | 279 (17.4) |
| 32 | 212 (6.6) | 503 (15.7) |
| 64 | 324 (5.1) | 840 (13.1) |
| 96 | 413 (4.3) | 1,081 (11.3) |
| 128 | 481 (3.8) | 1,263 (9.9) |

**What this predicts for the 10 tok/s-per-user target:** plain decode cannot reach it above about 16
users at any efficiency; DSpark can in principle go past 96 users at short context, and in practice to
whatever fraction of the ceiling the kernels reach (35-70% is typical). At 8K context the DSpark
ceiling for 64 users falls to 648 (10.1 per user), so long contexts cost concurrency through KV reads,
not only memory.

## KV capacity

vLLM's boot log on this machine (0.70 memory utilisation, BF16 KV): **858,464 tokens** in 57.3 GiB,
"maximum concurrency 3.27× at 262,144 tokens". That is ~100 streams at 8K or ~26 at 32K of live
context, so at chat lengths memory is not the binding limit; decode bandwidth is.

## Prefill

Prefill is compute bound: ~2 × 5.4 B active weight-visits ≈ 11 GFLOP per token plus attention. At
50% of the measured 95 TFLOPS BF16 (NVFP4 experts run faster), the practical ceiling is about 4,000
tokens/s at short context.

## Measured against the ceiling

| Streams | Plain decode: measured / ceiling | DSpark (vLLM): measured / ceiling |
|---:|---:|---:|
| 8 | 114 / 117 (97%) | 126 / 150 (84%) |
| 16 | 163 / 151 (108%) | 204 / 279 (73%) |
| 32 | 242 / 212 (114%) | 330 / 503 (66%) |
| 64 | 370 / 324 (114%) | 445 / 840 (53%) |

- **Plain decode beats the uniform-routing ceiling.** Real routing is skewed, so a batch touches fewer
  distinct experts than 1 − (120/128)^r predicts, and each step reads fewer bytes. The plain-decode
  ceiling here is therefore a slight underestimate; the hard bound with every expert touched is lower.
- **DSpark falls further below its ceiling as streams grow:** 84% at 8, 53% at 64. That is the compute
  co-bind predicted above (8 verify rows per stream: 512 rows at 64 streams), plus the draft's own
  forward. It is why the 10 tok/s edge lands near 28-32 users instead of ~100.
- **Prefill measured ~3,200 tokens/s** (vLLM engine log, 8K prompts), 80% of the estimate. An 8K prompt
  costs ~2.5 s of the whole GPU, which is what limits the long-context workload.
