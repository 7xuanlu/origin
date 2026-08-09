#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
HELPER="$REPO_ROOT/scripts/stabilize-rust-cache-toolchains.sh"
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

STATE_FILE="$TEST_DIR/toolchains"
UNINSTALL_LOG="$TEST_DIR/uninstalled"
MOCK_RUSTUP="$TEST_DIR/rustup"
PINNED="1.95.0-x86_64-pc-windows-msvc"
UNRELATED="stable-x86_64-pc-windows-msvc"

cat > "$STATE_FILE" <<EOF
$UNRELATED
$PINNED
EOF

cat > "$MOCK_RUSTUP" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

case "$1 $2" in
    "show active-toolchain")
        printf '%s (default)\n' "$MOCK_ACTIVE_TOOLCHAIN"
        ;;
    "toolchain list")
        cat "$MOCK_TOOLCHAIN_STATE"
        ;;
    "toolchain uninstall")
        toolchain=$3
        printf '%s\n' "$toolchain" >> "$MOCK_UNINSTALL_LOG"
        grep -Fvx "$toolchain" "$MOCK_TOOLCHAIN_STATE" > "$MOCK_TOOLCHAIN_STATE.next"
        mv "$MOCK_TOOLCHAIN_STATE.next" "$MOCK_TOOLCHAIN_STATE"
        ;;
    *)
        printf 'unexpected rustup invocation: %s\n' "$*" >&2
        exit 64
        ;;
esac
EOF
chmod +x "$MOCK_RUSTUP"

if [[ ! -f "$HELPER" ]]; then
    echo "FAIL: cache-toolchain helper is missing"
    exit 1
fi

MISSING_ACTIVE_STATE="$TEST_DIR/missing-active-toolchains"
MISSING_ACTIVE_LOG="$TEST_DIR/missing-active-uninstalled"
printf '%s\n' "$UNRELATED" > "$MISSING_ACTIVE_STATE"
if MOCK_ACTIVE_TOOLCHAIN="$PINNED" \
    MOCK_TOOLCHAIN_STATE="$MISSING_ACTIVE_STATE" \
    MOCK_UNINSTALL_LOG="$MISSING_ACTIVE_LOG" \
    RUSTUP_BIN="$MOCK_RUSTUP" \
        bash "$HELPER" >/dev/null 2>&1; then
    echo "FAIL: helper accepted an active toolchain absent from the installed set"
    exit 1
fi
if [[ -s "$MISSING_ACTIVE_LOG" ]]; then
    echo "FAIL: helper mutated toolchains before validating the active pin"
    cat "$MISSING_ACTIVE_LOG"
    exit 1
fi
echo "PASS: missing active toolchain fails before mutation"

MOCK_ACTIVE_TOOLCHAIN="$PINNED" \
MOCK_TOOLCHAIN_STATE="$STATE_FILE" \
MOCK_UNINSTALL_LOG="$UNINSTALL_LOG" \
RUSTUP_BIN="$MOCK_RUSTUP" \
    bash "$HELPER"

if [[ $(cat "$STATE_FILE") != "$PINNED" ]]; then
    echo "FAIL: helper did not leave only the active pinned toolchain"
    cat "$STATE_FILE"
    exit 1
fi

if [[ $(cat "$UNINSTALL_LOG") != "$UNRELATED" ]]; then
    echo "FAIL: helper removed the wrong toolchain"
    cat "$UNINSTALL_LOG"
    exit 1
fi

echo "PASS: unrelated runner toolchain removed; active pinned toolchain preserved"
