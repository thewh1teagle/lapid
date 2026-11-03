#!/usr/bin/env bash
# Regenerate the lapid-quant i-quant golden fixtures from ggml's reference implementation.
# Out of the cargo build (lapid stays pure Rust): needs a C compiler and a llama.cpp checkout.
#   tools/iquant_golden/gen.sh [path/to/llama.cpp]
# Compiled without floating-point contraction, as lapid's Rust ports assume.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
llama="${1:-$here/../../../llama.cpp}"
g="$llama/ggml"
out="$here/../../crates/lapid-quant/tests/fixtures/iquant"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cc -O2 -ffp-contract=off -std=gnu11 -I"$g/include" -I"$g/src" -I"$g/src/ggml-cpu" \
    "$here/golden.c" "$g/src/ggml-quants.c" -lm -o "$tmp/golden"
mkdir -p "$out"
"$tmp/golden" "$out"
echo "llama.cpp $(git -C "$llama" rev-parse --short HEAD 2>/dev/null || echo unknown)" > "$out/SOURCE"
ls -l "$out"
