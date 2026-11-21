#!/usr/bin/env bash
# Stop the recipe's container: give in-flight requests DRAIN_S seconds, then stop with a timeout.
# Only ever touches containers labelled recipe=inkling-2-mini-spark.
set -euo pipefail
. "$(dirname "$0")/scripts/common.sh"
load_settings vllm
ids=$(docker ps -aq -f "label=$LABEL")
[ -n "$ids" ] || { say "nothing running"; exit 0; }
mkdir -p "$RECIPE_DIR/logs"
for id in $ids; do
  name=$(docker inspect -f '{{.Name}}' "$id" | tr -d /)
  docker logs "$id" > "$RECIPE_DIR/logs/$name-$(date +%Y%m%d-%H%M%S).log" 2>&1 || true
  say "draining $name for ${DRAIN_S:-10}s"; sleep "${DRAIN_S:-10}"
  docker stop -t 60 "$id" >/dev/null; docker rm "$id" >/dev/null 2>&1 || true
  say "stopped $name"
done
