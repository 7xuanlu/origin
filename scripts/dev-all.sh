#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DAEMON_STARTED_BY_THIS_RUN=0
export WENLAN_DEV_PRESERVE_DAEMON_ON_QUIT=0

while IFS='=' read -r key value; do
  case "$key" in
    WENLAN_PORT|WENLAN_DEV_UI_PORT|WENLAN_DEV_REMOTE_PORT_START|WENLAN_DEV_APP_ID|WENLAN_DEV_TAURI_MCP_SOCKET|WENLAN_DATA_DIR|WENLAN_DEV_STATE_DIR)
      export "$key=$value"
      ;;
  esac
done < <(bash "$SCRIPT_DIR/dev-runtime.sh" print-config)

remove_tauri_mcp_socket() {
  if [[ -S "$WENLAN_DEV_TAURI_MCP_SOCKET" ]]; then
    rm -f "$WENLAN_DEV_TAURI_MCP_SOCKET"
  fi
}

cleanup() {
  trap - EXIT HUP INT TERM
  remove_tauri_mcp_socket
  if (( DAEMON_STARTED_BY_THIS_RUN == 1 )); then
    bash "$SCRIPT_DIR/dev-runtime.sh" stop
  fi
}
trap cleanup EXIT HUP INT TERM

cd "$REPO_ROOT"
remove_tauri_mcp_socket
pnpm prepare:sidecars --force-build
if bash "$SCRIPT_DIR/dev-runtime.sh" start-for-session; then
  DAEMON_STARTED_BY_THIS_RUN=1
else
  status=$?
  if (( status != 10 )); then
    exit "$status"
  fi
  export WENLAN_DEV_PRESERVE_DAEMON_ON_QUIT=1
fi
pnpm tauri dev --config "{\"identifier\":\"$WENLAN_DEV_APP_ID\",\"build\":{\"devUrl\":\"http://localhost:$WENLAN_DEV_UI_PORT\"}}"
