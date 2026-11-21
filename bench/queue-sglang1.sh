#!/usr/bin/env bash
# SGLang round 1: the cookbook's verified DGX Spark DSpark recipe with an explicit admission cap,
# then one change each: speculation off, a shorter verify block, a larger static memory pool.
set -u
cd "$(dirname "$0")/.."
export LEVELS=16,32,48,64
bench/experiment.sh sglang-ctl-dspark8 sglang
LEVELS=8,16,32,64,96 bench/experiment.sh sglang-nospec sglang SPEC=none MAX_SEQS=128
bench/experiment.sh sglang-draft4 sglang SPEC_DRAFT_TOKENS=4
bench/experiment.sh sglang-mem065 sglang MEM_FRACTION=0.65
echo QUEUE_DONE
