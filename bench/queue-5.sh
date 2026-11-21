#!/usr/bin/env bash
# TokenSpeed rebuilt from PR #3682 source, then its sweep and GSM8K gate.
set -u
cd "$(dirname "$0")/.."
gate() { python3 bench/gsm8k.py --name "$1" --n 250 --concurrency 32 | tee -a "docs/raw/$1.gsm8k.log"; }
engines/tokenspeed/build.sh > docs/raw/tokenspeed-build.log 2>&1 || echo "[E_BUILD] tokenspeed build failed"
KEEP=1 LEVELS=1,16,24,32 bench/experiment.sh ts-pr3682-dspark tokenspeed && gate ts-pr3682
./stop.sh
echo QUEUE_DONE
