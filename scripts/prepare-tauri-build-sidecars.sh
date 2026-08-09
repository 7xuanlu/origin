#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${TAURI_ENV_DEBUG:-}" == "true" ]]; then
  exec bash "$SCRIPT_DIR/prepare-sidecars.sh" "$@"
fi

exec bash "$SCRIPT_DIR/prepare-sidecars.sh" --release "$@"
