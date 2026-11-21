# Inkling 2 Mini on one DGX Spark, tuned for the most users at once

Serving recipe for [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(looped MoE, 26.4B total / 3.4B active, NVFP4 + MXFP8 as trained, 1M context, bundled DSpark draft) on
**one NVIDIA DGX Spark** (GB10, 128 GB unified memory), measured with four engines (TensorFold was checked too; it cannot serve the model yet) and hill-climbed for
**concurrency**: how many people can use the model at the same time and still get a usable stream.

```shell
./start.sh            # vLLM, the measured best: check the Spark, fetch the weights, start, smoke-test
./start.sh sglang     # or: tokenspeed, llamacpp
./stop.sh             # drain, stop, keep the server log in logs/
DRY_RUN=1 ./start.sh  # print the docker command and change nothing
```

The server speaks the OpenAI API on `http://<spark>:8000/v1`, model `inkling-2-mini`, with reasoning
(`reasoning_content`) and tool calls parsed.

## Results on one DGX Spark

How many people can stream from one Spark at once, with the **slowest 10% of replies still at ≥ 10 tok/s**
and TTFT p95 ≤ 5 s (chat workload, ~1K-token replies, default reasoning, temperature 1.0):

| Engine (best config) | Users at ≥ 10 tok/s | Users at ≥ 5 tok/s | Peak aggregate | TTFT p95 at the edge | GSM8K |
|---|---:|---:|---:|---:|---:|
| **vLLM**, DSpark (default) | **28** | 64 | 445 tok/s (64 admitted) | 2.7 s | 97.2% |
| SGLang, DSpark | 24 | 48 | 419 tok/s (64 admitted) | 0.75 s | 97.2% |
| vLLM, no speculation | 16 | 96 | 474 tok/s (96 admitted) | 4.8 s | not run |
| TokenSpeed (PR #3682, source build) | 1 | 1 | 66 tok/s (16 measured) | 0.58 s | 96.8% |
| llama.cpp, IQ4_NL GGUF | 1 | 16 | 138 tok/s (32 slots) | 0.27 s | 97.2% |
| TensorFold | does not support Inkling 2 Mini yet | | | | |

- **Default: vLLM with DSpark and an admission cap of 28** (`MAX_SEQS=28`). User 29 waits in the queue
  instead of slowing everyone below 10 tok/s; `MAX_SEQS=64` admits 64, the slowest at ~6.5 tok/s, for 445 tok/s in total.
- **Long prompts (8K tokens, agents, RAG): use `./start.sh sglang`.** It starts an 8K reply in 2.5 s vs
  7.9 s on vLLM and prefills about twice as fast; prefill, not memory, limits long-prompt traffic.
- **Batch jobs:** `SPEC=none` on vLLM gives the highest throughput (474 tok/s at 96 users, ~5 tok/s each).
- DSpark is the biggest lever: without it no engine holds 10 tok/s past 16 users. It only runs at the
  checkpoint's trained block (7 drafts); shorter drafts and loop sharing are refused for this checkpoint.

Full tables, long-context results and every rejected experiment: [docs/RESULTS.md](docs/RESULTS.md),
[docs/PLAN.md](docs/PLAN.md). Engine pins and status: [docs/ENGINES.md](docs/ENGINES.md).

## How "users at once" is measured

`bench/sweep.py` runs N closed-loop users: each sends a request, streams the reply, and sends the next.
A level passes when the **slowest 10% of replies still decode at ≥ 10 tok/s**, TTFT p95 ≤ 5 s and no
request fails. The 20 and 5 tok/s tiers are reported too.

- **Workload `chat`:** 16 fixed long-form instructions (code, essays, explanations), the model's default
  reasoning effort, temperature 1.0 (the checkpoint's default), natural EOS, replies capped at 1,024
  tokens. Reasoning tokens count. Draft acceptance depends on content, so the panel never changes.
- **Workload `long`:** a unique ~8K-token document per request and a question whose answer is inside it
  (retrieval is scored), 256-token replies.
- Token counts come from the server's `usage` block, never from streamed chunks (DSpark sends several
  tokens per chunk). Each level warms up first, so JIT compiles of new batch shapes are not timed.
- The client runs on the Spark itself (stdlib Python, ~1% CPU).

`docs/THEORY.md` computes the bandwidth ceiling for this model at each concurrency, `docs/PLAN.md` is the
experiment log (every run, kept or rejected), `docs/raw/` holds every sweep line and server boot log.

## Requirements

- One DGX Spark (GB10, CUDA 13 driver; measured on 580.178.04) with Docker and the NVIDIA runtime.
- About 40 GiB free memory for the default settings (`MIN_AVAIL_GIB`), and nothing else on the GPU.
- Disk: 20 GiB for the weights; 27 GB (vLLM), 39 GB (SGLang), 19 + 21 GB (TokenSpeed base and build) or
  5 GB (llama.cpp, plus 18 GB of GGUF) for the image.
- `curl`, `python3`, `sha256sum`, `ss`.

## Settings

Layered: `scripts/defaults.env`, then `engines/<engine>/defaults.env`, then `.env` (see `.env.example`),
then exported variables.

| Setting | Default | Meaning |
|---|---|---|
| `PORT`, `HOST`, `SERVED_NAME` | `8000`, `0.0.0.0`, `inkling-2-mini` | API address and model id |
| `MODEL_DIR`, `MODEL_REVISION` | `~/models/Inkling-2-Mini`, `ed40e7da` | weights; fetched and sha256-checked when missing |
| `MAX_SEQS` | engine default | **the admission cap**: requests past it queue instead of slowing everyone |
| `CONTEXT` | `262144` | longest request (the model takes 1M; the KV pool, not this, bounds users) |
| `SPEC` | `dspark` | `dspark` or `none` |
| `GPU_UTIL` / `MEM_FRACTION` | `0.70` (TokenSpeed `0.80`) / `0.65` | vLLM, TokenSpeed / SGLang share of the unified pool |
| `CACHE_DIR` | `~/.cache/inkling2mini-spark` | JIT, autotune and graph caches kept across restarts |
| `RESTART` | `no` | `unless-stopped` for a standing deployment |
| `EXTRA_ARGS` | empty | more engine flags, e.g. `--api-key ...` |

## Credits

See [CREDITS.md](CREDITS.md). Method: [inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill).
Layout inspired by [GLM-5.3-Flash-DGX-Sparks-1104](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104).

## License

Apache 2.0, see [LICENSE](LICENSE). Downloaded software and weights keep their own terms.
