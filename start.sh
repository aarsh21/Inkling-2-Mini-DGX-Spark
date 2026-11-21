#!/usr/bin/env bash
# Start Inkling 2 Mini on one DGX Spark with the chosen engine.
#   ./start.sh [vllm|sglang|tokenspeed|llamacpp]     (default: ENGINE from .env, else vllm)
#   DRY_RUN=1 ./start.sh sglang                      print the docker command and change nothing
set -euo pipefail
. "$(dirname "$0")/scripts/common.sh"
ENGINE=${1:-${ENGINE:-vllm}}
[ -f "$RECIPE_DIR/engines/$ENGINE/run.sh" ] || die E_CONFIG "unknown engine '$ENGINE' (vllm, sglang, tokenspeed, llamacpp)"
load_settings "$ENGINE"
. "$RECIPE_DIR/engines/$ENGINE/run.sh"   # defines the docker_args array

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' docker run "${docker_args[@]}"; echo; exit 0; fi
if docker ps -aq -f "label=$LABEL" | grep -q .; then
  say "stopping the running recipe container first"; "$RECIPE_DIR/stop.sh"
fi
check_host
[ "$ENGINE" = llamacpp ] || ensure_weights
declare -F engine_prepare >/dev/null && engine_prepare   # optional per-engine hook (downloads)
mkdir -p "$CACHE_DIR"
say "starting $ENGINE as $CONTAINER on :$PORT"
docker run "${docker_args[@]}" >/dev/null
wait_ready
smoke
say "serving '$SERVED_NAME' at http://$(hostname -I | awk '{print $1}'):$PORT/v1   (logs: docker logs -f $CONTAINER)"
