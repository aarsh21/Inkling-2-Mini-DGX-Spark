# Credits

- **Inkling 2 Mini** and its DSpark draft are by Thinking Machines Lab, Apache 2.0
  (<https://huggingface.co/thinkingmachines/Inkling-2-Mini>).
- **vLLM** (Apache 2.0) and its `vllm/vllm-openai:inkling-2-mini` image; the Inkling 2 Mini support is by
  Inferact, Thinking Machines and NVIDIA (<https://vllm.ai/blog/2026-11-20-inkling2-mini>).
- **SGLang** (Apache 2.0) and its `lmsysorg/sglang:inkling-2-mini` image (sgl-project/sglang#51926). The
  DGX Spark launch flags start from the SGLang cookbook's verified Spark recipe.
- **TokenSpeed** (lightseekorg, Apache 2.0); Inkling 2 Mini support is lightseekorg/tokenspeed#3682.
- **llama.cpp** (MIT) by ggml-org; GGUF conversions by ggml-org (`Inkling-2-Mini-GGUF`) and the DSpark
  GGUF drafter by Unsloth (`unsloth/Inkling-2-Mini-GGUF`).
- **The method:** the hill-climb procedure, `bench/bench.py`, `bench/ceiling.py`, `bench/mem.sh` and
  `bench/queue.sh` come from sybil-solutions/inference-optimisation-skill (MIT, commit `fda968e7`, license
  in `bench/LICENSE.inference-optimisation-skill`).
- **The layout:** modelled on MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104 (a measured, settings-driven
  launcher with every default backed by a log entry).
