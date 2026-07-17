#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SKATE_BFM_DATA:-/63data1/hwh_data/Skate-bfm}"
ENV="$DATA/envs/skate-bfm"
CACHE="$DATA/cache/uv"
PYTHON="$ENV/bin/python"

mkdir -p "$DATA/envs" "$DATA/models" "$CACHE"
if [[ ! -x "$PYTHON" ]]; then
  uv venv "$ENV" --python 3.12
fi

run_uv() {
  if [[ "${USE_PROXYCHAINS:-0}" == "1" ]]; then
    proxychains uv "$@"
  else
    uv "$@"
  fi
}

export UV_CACHE_DIR="$CACHE"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

run_uv pip install --python "$PYTHON" \
  --index https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
run_uv pip install --python "$PYTHON" \
  -e "$ROOT/husky_sim" \
  "joblib==1.5.3" "moviepy==2.2.1" "safetensors==0.8.0" "pytest>=8,<9"

echo "Environment ready: $ENV"
echo "Place the BFM0 model under: $DATA/models/bfm0"
