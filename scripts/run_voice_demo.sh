#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -x "$PROJECT_DIR/.venv-demo/bin/python" ]]; then
  DEMO_PYTHON="$PROJECT_DIR/.venv-demo/bin/python"
else
  DEMO_PYTHON=python3
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "$DEMO_PYTHON" -m mediqa_qwen_eval.demo "$@"
