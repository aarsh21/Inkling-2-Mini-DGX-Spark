#!/usr/bin/env bash
# vLLM round 1: draft length, speculation off, scheduler token budget. One change each from the control.
set -u
cd "$(dirname "$0")/.."
export GPU_UTIL=0.70 LEVELS=16,32,48,64
bench/experiment.sh vllm-ctl-spec7 vllm
bench/experiment.sh vllm-spec3 vllm SPEC_TOKENS=3
bench/experiment.sh vllm-spec5 vllm SPEC_TOKENS=5
LEVELS=8,16,32,64,96 bench/experiment.sh vllm-nospec vllm SPEC=none MAX_SEQS=128
bench/experiment.sh vllm-spec7-bt8192 vllm CHUNK=8192
echo QUEUE_DONE
