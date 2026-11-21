#!/usr/bin/env bash
# Download thinkingmachines/Inkling-2-Mini at a pinned revision into a plain directory and check sha256.
#   scripts/fetch-weights.sh DEST REVISION
# Uses parallel ranged HTTP over IPv4 (scripts/pget.py): on the measured Spark, IPv6 to the Hub hung and
# the Xet client ran at under 1 MB/s while plain ranged GETs filled the link.
set -euo pipefail
dest=${1:?dest}; rev=${2:?revision}
here=$(cd "$(dirname "$0")" && pwd)
base=https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/$rev
for f in config.json generation_config.json hf_quant_config.json chat_template.jinja model.safetensors.index.json \
         processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json tiktoken/tokenizer.model \
         mtp.safetensors model-00001-of-00001.safetensors; do
  [ -f "$dest/$f" ] || python3 "$here/pget.py" "$base/$f" "$dest/$f" 16
done
cd "$dest"
sha256sum -c "$here/weights.sha256"
