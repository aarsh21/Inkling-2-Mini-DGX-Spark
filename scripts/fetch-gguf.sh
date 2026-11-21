#!/usr/bin/env bash
# Fetch the GGUF files llama.cpp serves, at pinned revisions, and check sha256.
#   scripts/fetch-gguf.sh DIR FILE...
set -euo pipefail
dir=${1:?dir}; shift
here=$(cd "$(dirname "$0")" && pwd)
ggml=https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/db61c5b492a6a034d986328d28274af0a439aa6c
unsloth=https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/ea0c79b2ecd73669e774c8519b79bfabfddbc027
for f in "$@"; do
  [ -f "$dir/$f" ] && continue
  case $f in dspark-*|*-UD-*) base=$unsloth ;; *) base=$ggml ;; esac
  python3 "$here/pget.py" "$base/$f" "$dir/$f" 16
done
cd "$dir" && grep -E " ($(IFS='|'; echo "$*"))\$" "$here/gguf.sha256" | sha256sum -c -
