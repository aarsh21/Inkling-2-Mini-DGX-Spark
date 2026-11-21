#!/usr/bin/env bash
# TokenSpeed with a pool that fits its per-request reservation: 32 requests, 64K context, 0.80 memory.
set -u
cd "$(dirname "$0")/.."
gate() { python3 bench/gsm8k.py --name "$1" --n 250 --concurrency 32 | tee -a "docs/raw/$1.gsm8k.log"; }
KEEP=1 LEVELS=1,16,24,32 bench/experiment.sh ts-pr3682-s32 tokenspeed MAX_SEQS=32 CONTEXT=65536 GPU_UTIL=0.80 \
  && gate ts-pr3682
./stop.sh
echo QUEUE_DONE
