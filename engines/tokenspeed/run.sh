# shellcheck shell=bash
spec=()
[ "$SPEC" = dspark ] && spec=(--speculative-algorithm DSPARK)
# shellcheck disable=SC2206
docker_args=(-d --name "$CONTAINER" --label "$LABEL" --restart "$RESTART" --gpus all --ipc=host --network host --shm-size 16g
  -e TRITON_USE_SLP_VECTORIZER=1
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/tokenspeed:/root/.cache"
  "$IMAGE" ts serve /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --tp 1 --max-model-len "$CONTEXT" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL"
  --chunked-prefill-size "$CHUNK" --kv-cache-dtype "$KV" --sampling-backend triton --enable-prefix-caching
  "${spec[@]}" $EXTRA_ARGS)
engine_prepare() { docker image inspect "$IMAGE" >/dev/null 2>&1 || "$RECIPE_DIR/engines/tokenspeed/build.sh"; }
