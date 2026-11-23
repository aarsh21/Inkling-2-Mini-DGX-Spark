# Time to first token (TTFT)

Measured on 2026-11-23 on one DGX Spark that also runs a full Senpiper appliance (about 50 GB free for the
model). The workload resembles an assistant like Ask: a shared ~6,200-token system prompt (rules plus tool
descriptions) and a short unique question, 400 output tokens at most, temperature 0.7, default reasoning effort
(`high`, 0.9). TTFT is the first streamed delta of any kind, so a reasoning token counts. "Answer" is the first
`content` token, after the reasoning. Each level runs N users twice. Client: [`bench/probe.py`](../bench/probe.py).
Raw lines: [`raw/ttft/`](raw/ttft/).

| Config | TTFT p50 at 1 / 8 / 16 / 24 users | Aggregate tok/s at 8 / 16 / 24 | Cold 8K prompt, TTFT at 1 / 4 / 8 users |
|---|---|---|---|
| vLLM, DSpark, mem 0.40, chunk 4096, prefix caching | 5.4 / 1.85 / 3.9 s / - | 154 / 213 / - | 7.9 s / - / - |
| SGLang stock, DSpark 8, mem 0.45, chunk 8192 | 0.27 / 0.48 / 0.83 / 1.02 s | 177 / 211 / 271 | 2.3 / 8.9 / 17.6 s |
| **SGLang + knapcio overlays**, same flags | **0.28 / 0.37 / 0.73 / 0.88 s** | **149 / 245 / 327** | **1.4 / 5.4 / 10.8 s** |

The overlay row passed the quality gate: GSM8K, first 250 questions, temperature 0: 97.2% (243/250), the same
as the stock engines.

## What made the difference

1. **Prefill speed.** On GB10, vLLM's inkling-2-mini engine prefills about 3,200 tok/s batched and about
   1,000 tok/s for one request. SGLang prefills about 3,700 tok/s stock and about 6,100 tok/s with the
   overlays (from the server's own `input throughput` log lines on 8K chunks).
2. **Prefix cache that hits.** SGLang's radix cache served 6,144 of the 6,224 prompt tokens from cache on every
   request (`--enable-cache-report` puts `cached_tokens` in `usage`). vLLM showed seconds of TTFT on the same
   shared prompt at 1 and 16 users.
3. **[knapcio's GB10 overlays](https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark)** (MIT, v2.4, commit
   `5b42d3f`): 13 exact import-time patches for this exact engine build (`lmsysorg/sglang:inkling-2-mini`, arm64
   `sha256:bc811ac7…`, SGLang `g03fde9cc3`). They cut the bytes read per decode step and the work per prefill
   chunk. `./start.sh sglang` fetches them at that commit and mounts them (`OVERLAY=0` runs the stock engine).
   The first start compiles their CUDA extensions, about 5 more minutes; the builds are cached in `CACHE_DIR`.

## Reasoning effort is the other half of perceived latency

The model thinks before it answers, and the chat template's default effort is `high` (0.9). With the same
prompt on vLLM, the first answer token came after:

| Effort (`chat_template_kwargs.reasoning_effort`) | Answer at 1 user | Answer at 8 users | Mean output tokens |
|---|---:|---:|---:|
| `high` (default) | 10.7 s | 6.0 s | 165-310 |
| `medium` | 2.1 s | 3.8 s | 131-142 |
| `low` | 3.4 s | 2.6 s | 102-137 |

These are 2-request samples, so treat them as indicative. A server-wide default is one flag,
`--default-chat-template-kwargs '{"reasoning_effort":"medium"}'` (SGLang and vLLM), and requests can still ask for
more. The recipe keeps the model's default because lower effort trades away answer quality.

## Still slow: bursts of cold long prompts

SGLang prefills prompts one after another, but every request's first token came at about the same time: 8 cold
8K documents sent together all started after 10.8 s. The 64K tokens are prefilled at about 6,100 tok/s before
any of them decode. With a cached system prompt this does not show (prefill is ~100 tokens per request), so it
matters only for bursts of fresh documents.
