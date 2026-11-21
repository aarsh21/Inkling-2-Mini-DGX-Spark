#!/usr/bin/env bash
# Third pass, revised: the Unsloth DSpark build was rejected at 8 users (42.7 tok/s vs 74.6 without
# speculation), so its remaining levels and variants are skipped. GSM8K gate for mainline llama.cpp, then
# TokenSpeed rebuilt from PR #3682 source.
set -u
cd "$(dirname "$0")/.."
gate() { python3 bench/gsm8k.py --name "$1" --n 250 --concurrency 32 | tee -a "docs/raw/$1.gsm8k.log"; }
./stop.sh
KEEP=1 LEVELS=1 bench/experiment.sh llama-iq4-nospec-gate llamacpp SLOTS=32 CTX_PER_SLOT=16384 SPEC=none \
  && gate llama-iq4-nospec
./stop.sh
docker image inspect tokenspeed-inkling2mini:pr3682-52fdeee2 >/dev/null 2>&1 \
  && docker image rm tokenspeed-inkling2mini:pr3682-52fdeee2
engines/tokenspeed/build.sh > docs/raw/tokenspeed-build.log 2>&1 || echo "[E_BUILD] tokenspeed build failed"
KEEP=1 LEVELS=1,16,24,32 bench/experiment.sh ts-pr3682-dspark tokenspeed && gate ts-pr3682
./stop.sh
echo QUEUE_DONE
