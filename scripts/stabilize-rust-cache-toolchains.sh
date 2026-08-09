#!/usr/bin/env bash
set -euo pipefail

RUSTUP_BIN=${RUSTUP_BIN:-rustup}
active_output=$("$RUSTUP_BIN" show active-toolchain)
active_toolchain=${active_output%% *}

if [[ -z "$active_toolchain" ]]; then
    echo "ERROR: rustup did not report an active toolchain" >&2
    exit 1
fi

installed_lines=()
while IFS= read -r line; do
    installed_lines+=("$line")
done < <("$RUSTUP_BIN" toolchain list --quiet)
active_found=false
for line in "${installed_lines[@]}"; do
    toolchain=${line%% *}
    if [[ "$toolchain" == "$active_toolchain" ]]; then
        active_found=true
        break
    fi
done

if [[ "$active_found" != true ]]; then
    echo "ERROR: active Rust toolchain is not installed: $active_toolchain" >&2
    exit 1
fi

for line in "${installed_lines[@]}"; do
    toolchain=${line%% *}
    if [[ "$toolchain" != "$active_toolchain" ]]; then
        echo "Removing unrelated Rust toolchain from cache inputs: $toolchain"
        "$RUSTUP_BIN" toolchain uninstall "$toolchain"
    fi
done

remaining_lines=()
while IFS= read -r line; do
    remaining_lines+=("$line")
done < <("$RUSTUP_BIN" toolchain list --quiet)
if [[ ${#remaining_lines[@]} -ne 1 || ${remaining_lines[0]%% *} != "$active_toolchain" ]]; then
    echo "ERROR: Rust cache inputs are not limited to the active toolchain" >&2
    printf '  %s\n' "${remaining_lines[@]}" >&2
    exit 1
fi

echo "Rust cache toolchain input stabilized: $active_toolchain"
