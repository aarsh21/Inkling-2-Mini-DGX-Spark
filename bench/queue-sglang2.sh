#!/usr/bin/env bash
# SGLang round 2: the sliding-window pool (37,376 tokens at the cookbook's 0.45) caps running requests
# near 36; give it a larger share of the KV pool. One change from the control.
set -u
cd "$(dirname "$0")/.."
export LEVELS=32,48,64
bench/experiment.sh sglang-swa03 sglang MEM_FRACTION=0.45 EXTRA_ARGS="--swa-full-tokens-ratio 0.3"
echo QUEUE_DONE
