#!/usr/bin/env bash
set -euo pipefail
model_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$model_dir/../build.py" "$(basename -- "$model_dir")" "$@"
