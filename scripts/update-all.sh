#!/usr/bin/env bash
# Update Origin desktop app, daemon, and origin-mcp to latest main.
# Idempotent: safe to run repeatedly.
set -euo pipefail

TS=""
trap 'if [ -n "$TS" ] && [ -d "/tmp/Origin.app.bak-$TS" ] && [ ! -d /Applications/Origin.app ]; then mv "/tmp/Origin.app.bak-$TS" /Applications/Origin.app; echo "==> Restored backup after failure"; fi' ERR

ORIGIN_DIR="${ORIGIN_DIR:-$HOME/Repos/origin}"
ORIGIN_MCP_DIR="${ORIGIN_MCP_DIR:-$HOME/Repos/origin-mcp}"

echo "==> Pulling origin..."
git -C "$ORIGIN_DIR" pull --ff-only

echo "==> Pulling origin-mcp..."
git -C "$ORIGIN_MCP_DIR" pull --ff-only

echo "==> Upgrading origin-mcp via brew..."
brew upgrade 7xuanlu/tap/origin-mcp

echo "==> Building Origin .app..."
cd "$ORIGIN_DIR"
CXXFLAGS="-std=c++17" pnpm release

echo "==> Replacing /Applications/Origin.app..."
TS=$(date +%Y%m%d-%H%M%S)
if [ -d /Applications/Origin.app ]; then
  mv /Applications/Origin.app "/tmp/Origin.app.bak-$TS"
fi
cp -R "$ORIGIN_DIR/target/release/bundle/macos/Origin.app" /Applications/
xattr -cr /Applications/Origin.app

VERSION=$(/usr/bin/defaults read /Applications/Origin.app/Contents/Info.plist CFBundleShortVersionString)
echo "==> Installed Origin v$VERSION (backup at /tmp/Origin.app.bak-$TS)"

if lsof -ti :7878 >/dev/null 2>&1; then
  echo "==> Note: a daemon is running on :7878. Restart it to pick up the new binary:"
  echo "    lsof -ti :7878 | xargs kill -9 && open -a Origin"
fi
