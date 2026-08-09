#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROFILE=""
FORCE_BUILD=false
PRINT_PATHS=false
DOWNLOAD=false
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
    --download)
      DOWNLOAD=true
      ;;
    *)
      echo "usage: $0 [--release] [--force-build] [--print-paths] [--download]" >&2
      exit 2
      ;;
  esac
  shift
done

BIN_DIR="$REPO_ROOT/app/binaries"
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ { print $2 }')"

if [[ "$DOWNLOAD" == "true" ]]; then
  # --download mode: install the pinned RELEASED daemon asset from
  # 7xuanlu/wenlan instead of building from a local checkout. The asset is
  # arm64/darwin-only today, so the triple is fixed rather than host-derived.
  # This branch does NOT resolve a backend checkout or run cargo.
  TRIPLE="aarch64-apple-darwin"
  SERVER_DEST="$BIN_DIR/wenlan-server-$TRIPLE"
  MCP_DEST="$BIN_DIR/wenlan-mcp-$TRIPLE"
  CLI_DEST="$BIN_DIR/wenlan-$TRIPLE"
  CLOUDFLARED_DEST="$BIN_DIR/cloudflared-$TRIPLE"

  PIN_FILE="$REPO_ROOT/.wenlan-backend-version"
  if [[ ! -f "$PIN_FILE" ]]; then
    echo "error: $PIN_FILE not found; cannot resolve pinned backend release" >&2
    exit 1
  fi
  PIN_TAG="$(sed -n '1p' "$PIN_FILE" | tr -d '[:space:]')"
  PIN_SHA="$(sed -n '2p' "$PIN_FILE" | tr -d '[:space:]')"
  if [[ -z "$PIN_TAG" || -z "$PIN_SHA" ]]; then
    echo "error: $PIN_FILE is malformed; expected line 1 = release tag, line 2 = sha256" >&2
    exit 1
  fi

  ASSET="wenlan-darwin-arm64.tar.gz"
  TMP_DOWNLOAD="$(mktemp -d -p "${TMPDIR:-/tmp}")"
  trap 'rm -rf "$TMP_DOWNLOAD"' EXIT

  # Cross-repo read of 7xuanlu/wenlan, authenticated by this repo's own
  # GITHUB_TOKEN/GH_TOKEN -- works only because that repo is PUBLIC; if it
  # ever goes private this 404s and fails loud by design, no silent fallback.
  if ! gh release download "$PIN_TAG" --repo 7xuanlu/wenlan --pattern "$ASSET" --dir "$TMP_DOWNLOAD" --clobber; then
    echo "error: failed to download $ASSET from 7xuanlu/wenlan release $PIN_TAG" >&2
    exit 1
  fi

  TARBALL="$TMP_DOWNLOAD/$ASSET"
  if [[ ! -f "$TARBALL" ]]; then
    echo "error: $ASSET was not present after downloading release $PIN_TAG" >&2
    exit 1
  fi

  ACTUAL_SHA="$(shasum -a 256 "$TARBALL" | awk '{ print $1 }')"
  if [[ "$ACTUAL_SHA" != "$PIN_SHA" ]]; then
    echo "error: sha256 mismatch for $ASSET (pinned $PIN_SHA, got $ACTUAL_SHA)" >&2
    exit 1
  fi

  EXTRACT_DIR="$TMP_DOWNLOAD/extracted"
  mkdir -p "$EXTRACT_DIR"
  tar xzf "$TARBALL" -C "$EXTRACT_DIR"

  CLI_SRC="$(find "$EXTRACT_DIR" -type f -name "wenlan" -print -quit)"
  SERVER_SRC="$(find "$EXTRACT_DIR" -type f -name "wenlan-server" -print -quit)"
  MCP_SRC="$(find "$EXTRACT_DIR" -type f -name "wenlan-mcp" -print -quit)"
  if [[ -z "$CLI_SRC" || -z "$SERVER_SRC" || -z "$MCP_SRC" ]]; then
    echo "error: $ASSET did not contain all of wenlan, wenlan-server, wenlan-mcp" >&2
    exit 1
  fi

  mkdir -p "$BIN_DIR"

  # Signing: these downloaded binaries are re-signed AD-HOC by Tauri's macOS
  # bundle step (app/tauri.conf.json bundle.macOS.signingIdentity = "-"; the
  # app ships ad-hoc signed, not Developer-ID notarized). `xattr -cr` below
  # strips the quarantine attribute so Gatekeeper doesn't reject them --
  # identical treatment to the previously CI-compiled sidecars and to the
  # cloudflared sidecar. No Developer-ID re-sign is needed or performed.
  install -m 755 "$SERVER_SRC" "$SERVER_DEST"
  install -m 755 "$MCP_SRC" "$MCP_DEST"
  install -m 755 "$CLI_SRC" "$CLI_DEST"

  if command -v xattr >/dev/null 2>&1; then
    xattr -cr "$SERVER_DEST"
    xattr -cr "$MCP_DEST"
    xattr -cr "$CLI_DEST"
  fi

  # Cloudflared is OPTIONAL in download mode: the release workflow supplies it
  # via CLOUDFLARED_BIN, but the CI download-smoke only needs the daemon
  # binaries, so a missing cloudflared here is a warning -- NOT the hard error
  # the build path below raises (a full Tauri bundle does require it).
  CLOUDFLARED_SRC=""
  if [[ -n "${CLOUDFLARED_BIN:-}" ]]; then
    if [[ ! -x "$CLOUDFLARED_BIN" ]]; then
      echo "error: CLOUDFLARED_BIN is set but not executable: $CLOUDFLARED_BIN" >&2
      exit 1
    fi
    CLOUDFLARED_SRC="$CLOUDFLARED_BIN"
  elif command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED_SRC="$(command -v cloudflared)"
  fi

  if [[ -n "$CLOUDFLARED_SRC" ]]; then
    install -m 755 "$CLOUDFLARED_SRC" "$CLOUDFLARED_DEST"
  else
    echo "warning: cloudflared not found in PATH; raw Tauri builds may still fail on binaries/cloudflared-$TRIPLE" >&2
  fi

  echo "Prepared sidecars in $BIN_DIR for $TRIPLE (downloaded $PIN_TAG)"
  exit 0
fi

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
