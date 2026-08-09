#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE=""
FORCE_BUILD=false
PRINT_PATHS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      PROFILE="release"
      ;;
    --force-build)
      FORCE_BUILD=true
      ;;
    --print-paths)
      PRINT_PATHS=true
      ;;
    *)
      echo "usage: $0 [--release] [--force-build] [--print-paths]" >&2
      exit 2
      ;;
  esac
  shift
done

BIN_DIR="$REPO_ROOT/app/binaries"
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ { print $2 }')"

BACKEND_DIR="$(bash "$SCRIPT_DIR/resolve-backend-dir.sh" "$REPO_ROOT")"

REQUESTED_TRIPLE="${TARGET_TRIPLE:-${TAURI_ENV_TARGET_TRIPLE:-}}"
TRIPLE="${REQUESTED_TRIPLE:-$HOST_TRIPLE}"
USE_TARGET_DIR=false
if [[ -n "${TARGET_TRIPLE:-}" || ( -n "${TAURI_ENV_TARGET_TRIPLE:-}" && "${TAURI_ENV_TARGET_TRIPLE:-}" != "$HOST_TRIPLE" ) ]]; then
  USE_TARGET_DIR=true
fi
if [[ -z "$PROFILE" ]]; then
  if [[ "${TAURI_ENV_DEBUG:-}" == "false" ]]; then
    PROFILE="release"
  else
    PROFILE="debug"
  fi
fi
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BACKEND_DIR/target}"
if [[ "$USE_TARGET_DIR" == "true" ]]; then
  SOURCE_DIR="$CARGO_TARGET_DIR/$TRIPLE/$PROFILE"
else
  SOURCE_DIR="$CARGO_TARGET_DIR/$PROFILE"
fi
case "$TRIPLE" in
  *windows*) EXE_SUFFIX=".exe" ;;
  *) EXE_SUFFIX="" ;;
esac
SERVER_SRC="$SOURCE_DIR/wenlan-server$EXE_SUFFIX"
MCP_SRC="$SOURCE_DIR/wenlan-mcp$EXE_SUFFIX"
CLI_SRC="$SOURCE_DIR/wenlan$EXE_SUFFIX"
SERVER_DEST="$BIN_DIR/wenlan-server-$TRIPLE$EXE_SUFFIX"
MCP_DEST="$BIN_DIR/wenlan-mcp-$TRIPLE$EXE_SUFFIX"
CLI_DEST="$BIN_DIR/wenlan-$TRIPLE$EXE_SUFFIX"
CLOUDFLARED_DEST="$BIN_DIR/cloudflared-$TRIPLE$EXE_SUFFIX"
CLOUDFLARED_SRC=""

if [[ -n "${CLOUDFLARED_BIN:-}" ]]; then
  if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
    echo "error: CLOUDFLARED_BIN is set but not executable: $CLOUDFLARED_BIN" >&2
    exit 1
  fi
  CLOUDFLARED_SRC="$CLOUDFLARED_BIN"
elif [[ "$TRIPLE" == "$HOST_TRIPLE" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED_SRC="$(command -v cloudflared)"
  fi
elif [[ "$PRINT_PATHS" != "true" ]]; then
  echo "error: CLOUDFLARED_BIN is required for cross-target sidecar prep (TARGET_TRIPLE=$TRIPLE)" >&2
  echo "       Refusing to copy host cloudflared for non-host target $HOST_TRIPLE -> $TRIPLE." >&2
  exit 1
fi

if [[ "$PRINT_PATHS" == "true" ]]; then
  printf 'server_src=%s\n' "$SERVER_SRC"
  printf 'mcp_src=%s\n' "$MCP_SRC"
  printf 'cli_src=%s\n' "$CLI_SRC"
  printf 'server_dest=%s\n' "$SERVER_DEST"
  printf 'mcp_dest=%s\n' "$MCP_DEST"
  printf 'cli_dest=%s\n' "$CLI_DEST"
  printf 'cloudflared_src=%s\n' "${CLOUDFLARED_SRC:-<CLOUDFLARED_BIN required for cross-target>}"
  printf 'cloudflared_dest=%s\n' "$CLOUDFLARED_DEST"
  exit 0
fi

if [[ "$FORCE_BUILD" == "true" || ! -x "$SERVER_SRC" || ! -x "$MCP_SRC" || ! -x "$CLI_SRC" ]]; then
  if [[ "$USE_TARGET_DIR" == "true" && "$PROFILE" == "release" ]]; then
    cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" --target "$TRIPLE" --release -p wenlan-server -p wenlan-mcp -p wenlan
  elif [[ "$USE_TARGET_DIR" == "true" ]]; then
    cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" --target "$TRIPLE" -p wenlan-server -p wenlan-mcp -p wenlan
  elif [[ "$PROFILE" == "release" ]]; then
    cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" --release -p wenlan-server -p wenlan-mcp -p wenlan
  else
    cargo build --manifest-path "$BACKEND_DIR/Cargo.toml" -p wenlan-server -p wenlan-mcp -p wenlan
  fi
else
  echo "Using existing backend sidecars from $SOURCE_DIR"
fi

mkdir -p "$BIN_DIR"

install -m 755 "$SERVER_SRC" "$SERVER_DEST"
install -m 755 "$MCP_SRC" "$MCP_DEST"
install -m 755 "$CLI_SRC" "$CLI_DEST"

if [[ -n "$CLOUDFLARED_SRC" ]]; then
  install -m 755 "$CLOUDFLARED_SRC" "$CLOUDFLARED_DEST"
else
  echo "error: cloudflared not found in PATH; install cloudflared or set CLOUDFLARED_BIN" >&2
  echo "       Required by Tauri externalBin: binaries/cloudflared-$TRIPLE" >&2
  exit 1
fi

echo "Prepared sidecars in $BIN_DIR for $TRIPLE"
