#!/usr/bin/env bash
# Convert the merged model to GGUF and quantise it for CPU / laptop inference with llama.cpp.
#
#   bash scripts/export_gguf.sh                     # artifacts/merged -> artifacts/gguf/model-{f16,Q4_K_M,Q8_0}.gguf
#   QUANTS="Q4_K_M" bash scripts/export_gguf.sh
#
# Q4_K_M: ~2.5 GB for a 4B model, runs on a laptop CPU; Q8_0: near-lossless reference.
set -euo pipefail

MERGED=${MERGED:-artifacts/merged}
OUT=${OUT:-artifacts/gguf}
QUANTS=${QUANTS:-"Q4_K_M Q8_0"}
LLAMA=${LLAMA:-third_party/llama.cpp}

mkdir -p "$OUT" third_party
if [ ! -d "$LLAMA" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
pip install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"

if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$LLAMA/build" --target llama-quantize llama-cli -j "$(nproc)" >/dev/null
fi

python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outtype f16 --outfile "$OUT/model-f16.gguf"
for q in $QUANTS; do
  "$LLAMA/build/bin/llama-quantize" "$OUT/model-f16.gguf" "$OUT/model-$q.gguf" "$q"
done
ls -lh "$OUT"
echo
echo "Try it:  $LLAMA/build/bin/llama-cli -m $OUT/model-Q4_K_M.gguf -cnv"
