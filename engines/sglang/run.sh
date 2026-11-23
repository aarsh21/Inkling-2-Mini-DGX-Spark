# shellcheck shell=bash
spec=(); env_spec=()
if [ "$SPEC" = dspark ]; then
  spec=(--speculative-algorithm DSPARK --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS")
  env_spec=(-e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
    -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1)
fi
# knapcio's GB10 speed overlays (MIT, docs/TTFT.md): exact import-time patches, fetched at a pinned commit
overlay=()
if [ "$OVERLAY" = 1 ]; then
  overlay=(-v "$CACHE_DIR/knapcio-$OVERLAY_COMMIT/overlay:/opt/inkling-overlay:ro" -e PYTHONPATH=/opt/inkling-overlay
    --env-file "$RECIPE_DIR/engines/sglang/overlay.env")
  [ "$SPEC" = dspark ] && spec+=(--speculative-use-block-verification)
fi
engine_prepare() {
  [ "$OVERLAY" = 1 ] || return 0
  local dir="$CACHE_DIR/knapcio-$OVERLAY_COMMIT"
  [ -f "$dir/overlay/sitecustomize.py" ] && return 0
  say "fetching knapcio's overlays at $OVERLAY_COMMIT"
  rm -rf "$dir" && git clone -q https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark "$dir" &&
    git -C "$dir" checkout -q "$OVERLAY_COMMIT" || die E_CONFIG "could not fetch the overlays (OVERLAY=0 runs without them)"
}
# shellcheck disable=SC2206
docker_args=(-d --name "$CONTAINER" --label "$LABEL" --restart "$RESTART" --gpus all --ipc=host --network host --shm-size 16g
  -e MAX_JOBS="$MAX_JOBS" "${env_spec[@]}" "${overlay[@]}"
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR/sglang:/root/.cache"
  "$IMAGE" sglang serve --model-path /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128
  --context-length "$CONTEXT" --max-running-requests "$MAX_SEQS" --mem-fraction-static "$MEM_FRACTION"
  --chunked-prefill-size "$CHUNK" --kv-cache-dtype "$KV"
  --grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2
  "${spec[@]}" $EXTRA_ARGS)
