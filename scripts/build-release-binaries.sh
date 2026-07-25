#!/usr/bin/env bash
set -euo pipefail

tools_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${WENLAN_REPO_ROOT:-$(cd "$tools_root/.." && pwd)}"
target="${1:?usage: build-release-binaries.sh <target>}"

if command -v python3 >/dev/null 2>&1; then
  python_bin=python3
else
  python_bin=python
fi
"$python_bin" "$tools_root/release_targets.py" check --target "$target" >/dev/null

cd "$repo_root"
cargo build --release \
  -p wenlan \
  -p wenlan-server \
  -p wenlan-mcp \
  --target "$target"

suffix=""
if [[ "$target" == "x86_64-pc-windows-msvc" ]]; then
  suffix=".exe"
fi
target_root="${CARGO_TARGET_DIR:-target}"
binary_dir="$target_root/$target/release"
"$binary_dir/wenlan$suffix" --help
"$binary_dir/wenlan-server$suffix" --help
"$binary_dir/wenlan-mcp$suffix" --help
