#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/jchacker5/Documents/ornith-mlx-research"
VENV="$ROOT/.venv/bin"

# Ornith-35B MoE: use mixed-bit quant to preserve gate precision.
"$VENV/mlx_lm.convert" \
  --hf-path deepreinforce-ai/Ornith-1.0-35B \
  --mlx-path "$ROOT/models/ornith-35b-4bit-mixed" \
  -q \
  --quant-predicate mixed_3_4 \
  --trust-remote-code

echo "Converted to $ROOT/models/ornith-35b-4bit-mixed"