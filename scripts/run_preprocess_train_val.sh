#!/usr/bin/env bash
# dualhead/ 에서 train → val 전처리 (.venv 고정)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "error: $PY 없음. cd $ROOT && python3 -m venv .venv && pip install ..." >&2
  exit 1
fi
echo "[$(date -Iseconds)] train start"
"$PY" datasets/preprocess_data.py --root datasets/argoverse_v1 --split train --num_workers 24
echo "[$(date -Iseconds)] val start"
"$PY" datasets/preprocess_data.py --root datasets/argoverse_v1 --split val --num_workers 24
echo "[$(date -Iseconds)] done"
