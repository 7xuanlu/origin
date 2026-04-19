#!/usr/bin/env bash
# Downloads origin-mcp and cloudflared binaries for Tauri sidecar bundling.
# Usage: bash scripts/download-sidecars.sh
#
# Binaries are placed in app/binaries/ with Tauri's naming convention:
#   {name}-{target_triple}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARIES_DIR="$SCRIPT_DIR/../app/binaries"
mkdir -p "$BINARIES_DIR"

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64) TARGET="aarch64-apple-darwin" ;;
  x86_64)        TARGET="x86_64-apple-darwin" ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

echo "Target: $TARGET"
echo "Binaries dir: $BINARIES_DIR"

# ── origin-server (daemon) ─────────────────────────────────────────
ORIGIN_SERVER_BIN="$BINARIES_DIR/origin-server-$TARGET"
REPO_ROOT="$SCRIPT_DIR/.."
if [ -f "$REPO_ROOT/target/release/origin-server" ]; then
  echo "Copying origin-server from target/release/"
  cp "$REPO_ROOT/target/release/origin-server" "$ORIGIN_SERVER_BIN"
elif [ -f "$REPO_ROOT/target/debug/origin-server" ]; then
  echo "Copying origin-server from target/debug/ (debug build)"
  cp "$REPO_ROOT/target/debug/origin-server" "$ORIGIN_SERVER_BIN"
else
  echo "origin-server not found. Build it first:"
  echo "  cargo build --release -p origin-server"
  exit 1
fi
chmod +x "$ORIGIN_SERVER_BIN"
echo "✓ origin-server → $ORIGIN_SERVER_BIN"

# ── origin-mcp ──────────────────────────────────────────────────────
ORIGIN_MCP_BIN="$BINARIES_DIR/origin-mcp-$TARGET"
if [ -f "$HOME/.cargo/bin/origin-mcp" ]; then
  echo "Copying origin-mcp from ~/.cargo/bin/"
  cp "$HOME/.cargo/bin/origin-mcp" "$ORIGIN_MCP_BIN"
elif [ -f "$HOME/Repos/origin-mcp/target/release/origin-mcp" ]; then
  echo "Copying origin-mcp from local repo build"
  cp "$HOME/Repos/origin-mcp/target/release/origin-mcp" "$ORIGIN_MCP_BIN"
elif command -v origin-mcp &>/dev/null; then
  echo "Copying origin-mcp from PATH"
  cp "$(which origin-mcp)" "$ORIGIN_MCP_BIN"
else
  echo "origin-mcp not found. Install it first:"
  echo "  cargo install --git https://github.com/7xuanlu/origin-mcp"
  exit 1
fi
chmod +x "$ORIGIN_MCP_BIN"
echo "✓ origin-mcp → $ORIGIN_MCP_BIN"

# ── cloudflared ─────────────────────────────────────────────────────
CLOUDFLARED_BIN="$BINARIES_DIR/cloudflared-$TARGET"
if command -v cloudflared &>/dev/null; then
  echo "Copying cloudflared from PATH"
  cp "$(which cloudflared)" "$CLOUDFLARED_BIN"
else
  echo "Downloading cloudflared..."
  case "$ARCH" in
    arm64|aarch64) CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz" ;;
    x86_64)        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz" ;;
  esac
  TMPDIR_CF="$(mktemp -d)"
  curl -sL "$CF_URL" | tar xz -C "$TMPDIR_CF"
  cp "$TMPDIR_CF/cloudflared" "$CLOUDFLARED_BIN"
  rm -rf "$TMPDIR_CF"
fi
chmod +x "$CLOUDFLARED_BIN"
echo "✓ cloudflared → $CLOUDFLARED_BIN"

echo ""
echo "Done. Both binaries ready in $BINARIES_DIR"
ls -lh "$BINARIES_DIR"
