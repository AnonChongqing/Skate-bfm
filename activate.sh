#!/usr/bin/env bash

SKATE_BFM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SKATE_BFM_ROOT
export SKATE_BFM_DATA="${SKATE_BFM_DATA:-/63data1/hwh_data/Skate-bfm}"
export VIRTUAL_ENV="$SKATE_BFM_DATA/envs/skate-bfm"

if [[ ! -x "$VIRTUAL_ENV/bin/python" ]]; then
  echo "Missing environment: $VIRTUAL_ENV" >&2
  echo "Run: bash $SKATE_BFM_ROOT/setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$SKATE_BFM_ROOT/03_latent_flow:$SKATE_BFM_ROOT/husky_sim/src:$SKATE_BFM_ROOT/01_bfm0_motion_husky:$SKATE_BFM_ROOT/01_bfm0_motion_husky/bfm0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SKATE_BFM_DATA/cache/huggingface"
export TORCH_HOME="$SKATE_BFM_DATA/cache/torch"
export UV_CACHE_DIR="$SKATE_BFM_DATA/cache/uv"
export XDG_CACHE_HOME="$SKATE_BFM_DATA/cache"
export WARP_CACHE_PATH="/tmp/skate_bfm_warp_${USER}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "$SKATE_BFM_DATA/runs/latent_flow"
RESULTS_LINK="$SKATE_BFM_ROOT/03_latent_flow/results/runs"
if [[ ! -e "$RESULTS_LINK" && ! -L "$RESULTS_LINK" ]]; then
  ln -s "$SKATE_BFM_DATA/runs/latent_flow" "$RESULTS_LINK"
fi

mkdir -p "$SKATE_BFM_DATA/checkpoints/latent_flow"
CHECKPOINT_LINK="$SKATE_BFM_ROOT/03_latent_flow/checkpoint"
if [[ ! -e "$CHECKPOINT_LINK" && ! -L "$CHECKPOINT_LINK" ]]; then
  ln -s "$SKATE_BFM_DATA/checkpoints/latent_flow" "$CHECKPOINT_LINK"
fi

hash -r
