#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Release/CI flow: daemon sidecars come from the pinned RELEASED asset
# (prepare-sidecars.sh --download), not a local backend compile. Tauri's
# beforeBuildCommand runs THIS wrapper, so the download must happen here -- else
# it falls through to the --release compile path and dies on the missing backend
# checkout. The release workflow sets WENLAN_DOWNLOAD_SIDECARS=1.
if [[ "${WENLAN_DOWNLOAD_SIDECARS:-}" == "1" ]]; then
  exec bash "$SCRIPT_DIR/prepare-sidecars.sh" --download
fi

if [[ "${TAURI_ENV_DEBUG:-}" == "true" ]]; then
  exec bash "$SCRIPT_DIR/prepare-sidecars.sh" "$@"
fi

exec bash "$SCRIPT_DIR/prepare-sidecars.sh" --release "$@"
