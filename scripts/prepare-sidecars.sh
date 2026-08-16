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

# Checked before the backend build, not after the install step. cloudflared is a
# hard Tauri externalBin requirement with no fallback, and finding it missing at
# install time means having waited out a full backend build first.
if [[ -z "$CLOUDFLARED_SRC" && "$PRINT_PATHS" != "true" ]]; then
  echo "error: cloudflared not found in PATH; install cloudflared or set CLOUDFLARED_BIN" >&2
  echo "       Required by Tauri externalBin: binaries/cloudflared-$TRIPLE" >&2
  echo "       Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
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

install -m 755 "$CLOUDFLARED_SRC" "$CLOUDFLARED_DEST"

# Windows needs two runtime DLLs beside the sidecars. The daemon loads ONNX
# Runtime dynamically there (fastembed's ort-load-dynamic feature, selected in
# crates/wenlan-core/Cargo.toml) and links the Vulkan loader for llama.cpp, and
# neither ships inside the executables. app/tauri.windows.conf.json bundles them
# out of this directory, so `tauri build` fails with ResourcePathNotFound if they
# are absent. Both staging scripts pin a version and verify a SHA-256; the Vulkan
# one also checks LunarG's Authenticode signature and drops VulkanRT-License.txt
# next to the loader, which the bundle ships for redistribution.
#
# The check below covers every file the bundle names, not just the two DLLs: a
# tree holding the DLLs but not the license would otherwise take the reuse
# branch and fail later inside `tauri build` with the same opaque
# ResourcePathNotFound this block exists to prevent. Reuse trusts whatever is
# already there, so bumping a pinned version in either staging script means
# deleting these files by hand on a machine that has already staged them; CI
# always starts from a clean checkout and is unaffected.
if [[ "$TRIPLE" == *windows* ]]; then
  WINDOWS_RUNTIME_FILES=(onnxruntime.dll vulkan-1.dll VulkanRT-License.txt)
  WINDOWS_RUNTIME_MISSING=false
  for name in "${WINDOWS_RUNTIME_FILES[@]}"; do
    if [[ ! -f "$BIN_DIR/$name" ]]; then
      WINDOWS_RUNTIME_MISSING=true
      break
    fi
  done
  if [[ "$WINDOWS_RUNTIME_MISSING" == "true" ]]; then
    if ! command -v powershell.exe >/dev/null 2>&1; then
      echo "error: the Windows runtime DLLs are missing from $BIN_DIR and powershell.exe is unavailable" >&2
      echo "       Stage them with scripts/stage-onnxruntime-windows.ps1 and scripts/stage-vulkan-loader-windows.ps1" >&2
      exit 1
    fi
    if command -v cygpath >/dev/null 2>&1; then
      PS_BIN_DIR="$(cygpath -w "$BIN_DIR")"
      PS_SCRIPT_DIR="$(cygpath -w "$SCRIPT_DIR")"
    else
      PS_BIN_DIR="$BIN_DIR"
      PS_SCRIPT_DIR="$SCRIPT_DIR"
    fi
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$PS_SCRIPT_DIR\\stage-onnxruntime-windows.ps1" \
      -DestinationDirectory "$PS_BIN_DIR"
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$PS_SCRIPT_DIR\\stage-vulkan-loader-windows.ps1" \
      -DestinationDirectory "$PS_BIN_DIR"
    for name in "${WINDOWS_RUNTIME_FILES[@]}"; do
      if [[ ! -f "$BIN_DIR/$name" ]]; then
        echo "error: staging finished but $BIN_DIR/$name is still missing" >&2
        exit 1
      fi
    done
  else
    echo "Using existing Windows runtime files in $BIN_DIR"
  fi
fi

echo "Prepared sidecars in $BIN_DIR for $TRIPLE"
