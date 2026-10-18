#!/usr/bin/env bash
# ignis on the llama.cpp leaf (ADR 0052), e.g. on a Jetson Orin:
#
#   IGNIS_LLAMA_CPP_DIR=~/llama.cpp scripts/llama/serve.sh build
#   IGNIS_LLAMA_CPP_DIR=~/llama.cpp scripts/llama/serve.sh run ~/models/qwen3-vl-4b [extra ignis-server flags]
#
# The model directory holds ignis-llama.json, the GGUF weights, the mmproj and
# the six frontend files of the model's Hugging Face repository
# (crates/llama/src/manifest.rs). llama.cpp is a shared-library CUDA build
# (BUILD_SHARED_LIBS=ON, GGML_CUDA=ON); its libraries are found through
# LD_LIBRARY_PATH, which a dependency's build script cannot set for the binary.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/../.." && pwd)
: "${IGNIS_LLAMA_CPP_DIR:?set IGNIS_LLAMA_CPP_DIR to the llama.cpp source tree}"
LIB=${IGNIS_LLAMA_CPP_LIB:-$IGNIS_LLAMA_CPP_DIR/build/bin}
TARGET=${CARGO_BUILD_TARGET:-$(uname -m)-unknown-linux-gnu}
BIN=$HERE/target/$TARGET/release/ignis-server

case "${1:-}" in
  build)
    cd "$HERE"
    IGNIS_LLAMA_CPP_DIR=$IGNIS_LLAMA_CPP_DIR IGNIS_LLAMA_CPP_LIB=$LIB \
      cargo build --release --target "$TARGET" -p ignis-server --features llama
    ;;
  run)
    model=${2:?model directory}
    shift 2
    export LD_LIBRARY_PATH=$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    exec "$BIN" --model-artifact "$model" --spec-backend off --vision-enabled \
      --model-max-context "${IGNIS_LLAMA_CONTEXT:-16384}" "$@"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 2
    ;;
esac
