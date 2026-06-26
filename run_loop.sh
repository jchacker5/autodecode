#!/usr/bin/env bash
# Kick off an autonomous agent with the autoresearch program.
# Usage: ./run_loop.sh [agent]
#   agent: opencode (default) | claude | manual
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

AGENT="${1:-opencode}"

case "$AGENT" in
  opencode)
    opencode run "Read program.md in $ROOT. Do setup (branch autoresearch/jun25 if fresh), run baseline uv run optimize.py, then loop forever optimizing decode_tok_s. Only edit optimize.py. Log to results.tsv."
    ;;
  manual)
    echo "Open program.md and point your agent at:"
    echo "  $ROOT"
    echo ""
    echo "Baseline:"
    echo "  uv run optimize.py > run.log 2>&1"
    echo "  grep '^decode_tok_s:' run.log"
    ;;
  *)
    echo "Unknown agent: $AGENT (use opencode|manual)"
    exit 1
    ;;
esac