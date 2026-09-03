#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "${1:-}" in
  run|doctor|report|compare)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark.py" "$@"
    ;;
  *)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/benchmark.py" run "$@"
    ;;
esac
