#!/usr/bin/env bash
# Build the llama.cpp image named by $IMAGE from release binaries placed on the CUDA 13 runtime of the
# published ghcr.io/ggml-org/llama.cpp:server-cuda13 image (built 2026-11-17, before inkling2_loop).
#   llama-cpp-inkling2mini:b12600         mainline b12600 (29b52d39, "model : Inkling 2 Mini", #32694).
#                                         Reads the ggml-org GGUFs. Its DSpark drafter does not load the
#                                         Unsloth DSpark GGUF (markov_w1 vocab 201024 vs 200058).
#   llama-cpp-inkling2mini:unsloth-b12588 Unsloth's fork b12588-mix-c0c3de8 (88328607), arm64 CUDA 13
#                                         portable build (sm_90-sm_121). Reads the Unsloth root UD GGUFs
#                                         and loads the Unsloth DSpark drafter.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); work=$(mktemp -d)
image=${IMAGE:-llama-cpp-inkling2mini:b12600}
case $image in
  *:b12600)
    url=https://github.com/ggml-org/llama.cpp/releases/download/b12600/llama-b12600-bin-ubuntu-cuda-13.5-arm64.tar.gz
    sum=8435712a783efb71ddb0c1d92b13cf8ef32ff8df1b58ec62f6a1d2a63873cc38 ;;
  *:unsloth-b12588)
    url=https://github.com/unslothai/llama.cpp/releases/download/b12588-mix-c0c3de8/app-b12588-mix-c0c3de8-linux-arm64-cuda13-portable.tar.gz
    sum=1ae89924180ec0ce8a18714737e591bf171db70f4fc5f5bd9fc160c7455004c7 ;;
  *) echo "build.sh: no recipe for $image" >&2; exit 1 ;;
esac
python3 "$here/../../scripts/pget.py" "$url" "$work/llama.tar.gz" 8
echo "$sum  $work/llama.tar.gz" | sha256sum -c -
mkdir "$work/x" && tar -xzf "$work/llama.tar.gz" -C "$work/x"
# The archives nest the binaries one directory down; take the directory that holds llama-server.
srv=$(find "$work/x" -name llama-server -type f | head -1)
mv "$(dirname "$srv")" "$work/bin"
cat > "$work/Dockerfile" <<'DOCKERFILE'
FROM ghcr.io/ggml-org/llama.cpp:server-cuda13@sha256:c5576f106d1ef86fe31148640281ae82fe327e9562b2abc7399dc94c82aa2b3d
COPY bin/ /app/
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
DOCKERFILE
docker build -t "$image" "$work"
rm -rf "$work"
