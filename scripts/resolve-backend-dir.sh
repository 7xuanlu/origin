#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${1:-$DEFAULT_REPO_ROOT}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

is_wenlan_backend_dir() {
  local dir="$1"
  [[ -f "$dir/Cargo.toml" && -d "$dir/crates/wenlan-server" && -d "$dir/crates/wenlan-mcp" && -d "$dir/crates/wenlan-cli" ]]
}

if [[ -n "${WENLAN_BACKEND_DIR:-}" ]]; then
  candidate="$WENLAN_BACKEND_DIR"
  if [[ "$candidate" != /* ]]; then
    candidate="$REPO_ROOT/$candidate"
  fi
  if ! is_wenlan_backend_dir "$candidate"; then
    echo "error: WENLAN_BACKEND_DIR is not a Wenlan backend checkout: $candidate" >&2
    exit 1
  fi
  (cd "$candidate" && pwd)
  exit 0
fi

for candidate in \
  "$REPO_ROOT/../wenlan" \
  "$REPO_ROOT/../../../wenlan" \
  "$REPO_ROOT/../.."
do
  if is_wenlan_backend_dir "$candidate"; then
    (cd "$candidate" && pwd)
    exit 0
  fi
done

echo "error: could not find Wenlan backend checkout; set WENLAN_BACKEND_DIR" >&2
exit 1
