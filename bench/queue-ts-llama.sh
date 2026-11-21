#!/usr/bin/env bash
# TokenSpeed (PR #3682 overlaid on the newest image) and llama.cpp (GGUF IQ4_NL + DSpark GGUF drafter).
set -u
cd "$(dirname "$0")/.."
LEVELS=1,16,32 bench/experiment.sh ts-pr3682-dspark tokenspeed
LEVELS=1,8,16,32 bench/experiment.sh llama-iq4-dspark3-s32 llamacpp SLOTS=32 CTX_PER_SLOT=16384
LEVELS=1,8,16,32 bench/experiment.sh llama-iq4-nospec-s32 llamacpp SLOTS=32 CTX_PER_SLOT=16384 SPEC=none
WORKLOAD=long LEVELS=1,2,4 bench/experiment.sh vllm-best-long8k-low vllm
WORKLOAD=long LEVELS=1,2,4 bench/experiment.sh sglang-best-long8k-low sglang MEM_FRACTION=0.65
./stop.sh
echo QUEUE_DONE
