#!/usr/bin/env bash
# Final check of the shipped default (./start.sh with no overrides) at its admission cap.
set -u
cd "$(dirname "$0")/.."
LEVELS=28 bench/experiment.sh vllm-default vllm
echo QUEUE_DONE
