#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin"
cat > "$tmp/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" > "$CARGO_LOG"
target=""
while (($#)); do
  if [[ "$1" == "--target" ]]; then
    target="$2"
    break
  fi
  shift
done
[[ -n "$target" ]]
suffix=""
[[ "$target" == "x86_64-pc-windows-msvc" ]] && suffix=".exe"
mkdir -p "$CARGO_TARGET_DIR/$target/release"
for binary in wenlan wenlan-server wenlan-mcp; do
  if [[ "${OMIT_MCP:-0}" == "1" && "$binary" == "wenlan-mcp" ]]; then
    continue
  fi
  cat > "$CARGO_TARGET_DIR/$target/release/$binary$suffix" <<'BIN'
#!/usr/bin/env bash
[[ "${1:-}" == "--help" ]]
BIN
  chmod +x "$CARGO_TARGET_DIR/$target/release/$binary$suffix"
done
EOF
chmod +x "$tmp/bin/cargo"

export PATH="$tmp/bin:$PATH"
export CARGO_LOG="$tmp/cargo.log"
export CARGO_TARGET_DIR="$tmp/target"

bash "$repo_root/scripts/build-release-binaries.sh" x86_64-unknown-linux-gnu
expected="build --release -p wenlan -p wenlan-server -p wenlan-mcp --target x86_64-unknown-linux-gnu"
[[ "$(cat "$CARGO_LOG")" == "$expected" ]]

mkdir -p "$tmp/release-tools/scripts"
cp "$repo_root/scripts/build-release-binaries.sh" \
  "$repo_root/scripts/release_targets.py" \
  "$tmp/release-tools/scripts/"
WENLAN_REPO_ROOT="$repo_root" \
  bash "$tmp/release-tools/scripts/build-release-binaries.sh" aarch64-apple-darwin
expected="build --release -p wenlan -p wenlan-server -p wenlan-mcp --target aarch64-apple-darwin"
[[ "$(cat "$CARGO_LOG")" == "$expected" ]]

rm -rf "$CARGO_TARGET_DIR/x86_64-unknown-linux-gnu"
if OMIT_MCP=1 \
  bash "$repo_root/scripts/build-release-binaries.sh" x86_64-unknown-linux-gnu; then
  echo "release smoke unexpectedly accepted a missing wenlan-mcp binary" >&2
  exit 1
fi

rm "$CARGO_LOG"
if bash "$repo_root/scripts/build-release-binaries.sh" x86_64-apple-darwin; then
  echo "unsupported release target unexpectedly succeeded" >&2
  exit 1
fi
[[ ! -e "$CARGO_LOG" ]]
