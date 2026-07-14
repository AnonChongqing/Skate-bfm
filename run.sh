#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/activate.sh"
exec python "$ROOT/01_bfm0_motion_husky/scripts/rollout.py" "$@"
