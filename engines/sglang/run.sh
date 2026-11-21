# shellcheck shell=bash
spec=(); env_spec=()
if [ "$SPEC" = dspark ]; then
  spec=(--speculative-algorithm DSPARK --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS")
  env_spec=(-e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
    -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1)
fi
# shellcheck disable=SC2206
docker_args=(-d --name "$CONTAINER" --label "$LABEL" --restart "$RESTART" --gpus all --ipc=host --network host --shm-size 16g
  -e MAX_JOBS="$MAX_JOBS" "${env_spec[@]}"
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/sglang:/root/.cache"
  "$IMAGE" sglang serve --model-path /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128
  --context-length "$CONTEXT" --max-running-requests "$MAX_SEQS" --mem-fraction-static "$MEM_FRACTION"
  --chunked-prefill-size "$CHUNK" --kv-cache-dtype "$KV"
  --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
  "${spec[@]}" $EXTRA_ARGS)
