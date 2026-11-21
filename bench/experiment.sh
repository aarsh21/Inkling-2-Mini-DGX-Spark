#!/usr/bin/env bash
# One hill-climb experiment: start ENGINE with the given overrides, run the concurrency sweep, stop.
#   bench/experiment.sh NAME ENGINE [VAR=value ...]
#   LEVELS=1,8,32,64 WORKLOAD=chat bench/experiment.sh vllm-spec7-s64 vllm MAX_SEQS=64 SPEC_TOKENS=7
# Writes docs/raw/NAME.boot.log (the server's startup log) and appends to docs/raw/sweep.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
name=${1:?name}; engine=${2:?engine}; shift 2
for kv in "$@"; do export "${kv?}"; done
mkdir -p docs/raw
./start.sh "$engine" 2>&1 | tee "docs/raw/$name.start.log"
. scripts/common.sh; load_settings "$engine"
docker logs "$CONTAINER" > "docs/raw/$name.boot.log" 2>&1
python3 bench/sweep.py --base-url "http://127.0.0.1:$PORT" --model "$SERVED_NAME" --name "$name" \
  --levels "${LEVELS:-1,8,16,32,48,64}" --workload "${WORKLOAD:-chat}" --rounds "${ROUNDS:-2}" \
  ${SWEEP_ARGS:-} | tee -a "docs/raw/$name.sweep.log"
docker logs "$CONTAINER" > "docs/raw/$name.server.log" 2>&1
[ "${KEEP:-0}" = 1 ] || ./stop.sh
