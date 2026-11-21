# shellcheck shell=bash
spec=()
if [ "$SPEC" = dspark ]; then
  spec=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_TOKENS,\"draft_sample_method\":\"probabilistic\"}")
fi
# shellcheck disable=SC2206
docker_args=(-d --name "$CONTAINER" --label "$LABEL" --restart "$RESTART" --gpus all --ipc=host --network host
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/vllm:/root/.cache"
  "$IMAGE" /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --tensor-parallel-size 1 --max-model-len "$CONTEXT" --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype "$KV" --max-num-batched-tokens "$CHUNK"
  --enable-prefix-caching --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
  "${spec[@]}" $EXTRA_ARGS)
