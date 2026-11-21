#!/usr/bin/env bash
# Finalists: finer concurrency steps around the 10 tok/s edge, the long-context workload, FP8 KV for
# long contexts, and the GSM8K quality gate on the same server.
set -u
cd "$(dirname "$0")/.."
gate() { python3 bench/gsm8k.py --name "$1" --n 250 --concurrency 32 | tee -a "docs/raw/$1.gsm8k.log"; }
KEEP=1 LEVELS=20,24,28,36,40 bench/experiment.sh vllm-best-fine vllm && gate vllm-best
KEEP=1 WORKLOAD=long LEVELS=8,16,24,32 bench/experiment.sh vllm-best-long8k vllm
KEEP=1 WORKLOAD=long LEVELS=8,16,24,32 bench/experiment.sh vllm-fp8kv-long8k vllm KV=fp8 && gate vllm-fp8kv
KEEP=1 LEVELS=20,24,28,36,40 bench/experiment.sh sglang-best-fine sglang MEM_FRACTION=0.65 && gate sglang-best
KEEP=1 WORKLOAD=long LEVELS=8,16,24,32 bench/experiment.sh sglang-best-long8k sglang MEM_FRACTION=0.65
./stop.sh
echo QUEUE_DONE
