# shellcheck shell=bash
# llama-server splits --ctx-size evenly over --parallel slots.
spec=()
[ "$SPEC" = dspark ] && spec=(-md "/gguf/$DRAFT_GGUF" --spec-type draft-dspark --spec-draft-n-max "$SPEC_TOKENS" -ngld 999)
# shellcheck disable=SC2206
docker_args=(-d --name "$CONTAINER" --label "$LABEL" --restart "$RESTART" --gpus all --network host
  -v "$GGUF_DIR:/gguf:ro"
  "$IMAGE" --model "/gguf/$GGUF" --alias "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --jinja --n-gpu-layers 999 --flash-attn on --parallel "$SLOTS" --ctx-size $((SLOTS * CTX_PER_SLOT))
  --cont-batching --metrics "${spec[@]}" $EXTRA_ARGS)
engine_prepare() {
  "$RECIPE_DIR/scripts/fetch-gguf.sh" "$GGUF_DIR" "$GGUF" ${spec:+"$DRAFT_GGUF"}
  docker image inspect "$IMAGE" >/dev/null 2>&1 || "$RECIPE_DIR/engines/llamacpp/build.sh"
}
