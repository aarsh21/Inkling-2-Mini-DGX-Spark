#!/usr/bin/env bash
# TokenSpeed, second source build: tokenspeed-kernel compiled for sm_121a. Sweep and GSM8K gate if it serves.
set -u
cd "$(dirname "$0")/.."
gate() { python3 bench/gsm8k.py --name "$1" --n 250 --concurrency 32 | tee -a "docs/raw/$1.gsm8k.log"; }
docker image rm tokenspeed-inkling2mini:pr3682-52fdeee2 2>/dev/null
engines/tokenspeed/build.sh > docs/raw/tokenspeed-build.log 2>&1 || echo "[E_BUILD] tokenspeed build failed"
docker image inspect tokenspeed-inkling2mini:pr3682-52fdeee2 >/dev/null 2>&1 \
  && KEEP=1 LEVELS=1,16,24,32 bench/experiment.sh ts-pr3682-sm121 tokenspeed && gate ts-pr3682
./stop.sh
echo QUEUE_DONE
