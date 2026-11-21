# Shared by start.sh and stop.sh. Sourced, not executed.
# Settings are layered: engines/<engine>/defaults.env, then .env, then exported variables.

RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LABEL=recipe=inkling-2-mini-spark

die() { echo "[$1] $2" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }

load_settings() {
  local engine=$1 f exported
  exported=$(export -p)  # exported variables win over the files
  set -a
  for f in "$RECIPE_DIR/scripts/defaults.env" "$RECIPE_DIR/engines/$engine/defaults.env" "$RECIPE_DIR/.env"; do
    # shellcheck disable=SC1090
    [ -f "$f" ] && . "$f"
  done
  set +a
  eval "$(sed 's/^declare -x /declare -gx /' <<<"$exported")" 2>/dev/null
}

check_host() {
  [ "$(uname -m)" = aarch64 ] || die E_HOST "expected an aarch64 DGX Spark, got $(uname -m)"
  command -v docker >/dev/null || die E_HOST "docker is not installed"
  docker info --format '{{json .Runtimes}}' | grep -q nvidia || die E_HOST "the NVIDIA container runtime is missing"
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q GB10 || say "warning: no GB10 found; this recipe was measured on a DGX Spark"
  local avail_gib
  avail_gib=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$avail_gib" -ge "${MIN_AVAIL_GIB:-40}" ] || die E_MEMORY "only ${avail_gib} GiB available, need MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-40} (stop other GPU work first)"
  if ss -ltn "( sport = :$PORT )" | grep -q LISTEN; then die E_PORT "port $PORT is in use"; fi
}

ensure_weights() {
  local f
  for f in config.json model-00001-of-00001.safetensors mtp.safetensors tokenizer.json; do
    [ -f "$MODEL_DIR/$f" ] && continue
    say "weights missing in $MODEL_DIR, downloading thinkingmachines/Inkling-2-Mini@$MODEL_REVISION"
    "$RECIPE_DIR/scripts/fetch-weights.sh" "$MODEL_DIR" "$MODEL_REVISION"
    return
  done
}

wait_ready() {
  local t0=$SECONDS
  say "waiting for /v1/models on :$PORT (up to ${WAIT_TIMEOUT}s)"
  # /v1/models alone is not enough: TokenSpeed's gateway answers it before the engine has loaded.
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 &&
        curl -sf -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
          -d "{\"model\": \"$SERVED_NAME\", \"max_tokens\": 1, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}" \
          >/dev/null 2>&1; do
    docker ps -q -f "name=^${CONTAINER}$" | grep -q . || { docker logs --tail 60 "$CONTAINER" 2>&1; die E_START "the container exited"; }
    [ $((SECONDS - t0)) -lt "$WAIT_TIMEOUT" ] || die E_START "not ready after ${WAIT_TIMEOUT}s; see docker logs $CONTAINER"
    sleep 5
  done
  say "ready after $((SECONDS - t0))s"
}

smoke() {
  local out
  out=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$SERVED_NAME\", \"max_tokens\": 1024, \"temperature\": 0,
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Reply with the number only.\"}]}") \
    || die E_SMOKE "the smoke request failed"
  echo "$out" | grep -q 391 || die E_SMOKE "unexpected smoke reply: $(echo "$out" | head -c 400)"
  say "smoke test passed (17 * 23 = 391)"
}
