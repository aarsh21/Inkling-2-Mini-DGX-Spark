#!/usr/bin/env bash
# Admission cap check: vLLM with MAX_SEQS at the 10 tok/s edge, offered more users than it admits.
set -u
cd "$(dirname "$0")/.."
LEVELS=32,48 bench/experiment.sh vllm-cap32 vllm MAX_SEQS=32
echo QUEUE_DONE
