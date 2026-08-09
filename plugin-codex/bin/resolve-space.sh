#!/usr/bin/env bash
# Resolve client-local Wenlan Space context.
# Precedence: strict pin > explicit arg > default env > cwd mapping >
# registered repo basename > unscoped. The daemon owns the save default.

set -u

cwd=""
arg=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cwd) cwd="${2:-}"; shift $(( $# > 1 ? 2 : 1 )) ;;
        --arg) arg="${2:-}"; shift $(( $# > 1 ? 2 : 1 )) ;;
        --topic) shift $(( $# > 1 ? 2 : 1 )) ;;
        *) shift ;;
    esac
done

trim() {
    printf '%s' "$1" | awk '{
        sub(/^[[:space:]]+/, "")
        sub(/[[:space:]]+$/, "")
        print
    }'
}

locked="$(trim "${WENLAN_SPACE:-}")"
if [ -n "$locked" ]; then
    printf '%s\tlocked-env\n' "$locked"
    exit 0
fi

explicit="$(trim "$arg")"
if [ -n "$explicit" ]; then
    printf '%s\targ\n' "$explicit"
    exit 0
fi

fallback="$(trim "${WENLAN_DEFAULT_SPACE:-}")"
if [ -n "$fallback" ]; then
    printf '%s\tdefault-env\n' "$fallback"
    exit 0
fi

spaces_file="${SPACES_FILE:-${HOME}/.wenlan/spaces.toml}"
if [ -n "$cwd" ] && [ -f "$spaces_file" ]; then
    pairs="$(mktemp)"
    awk '
        BEGIN { in_mapping = 0; prefix = ""; space = "" }
        /^\[\[mapping\]\]/ {
            if (in_mapping && prefix != "" && space != "") print prefix "\t" space
            in_mapping = 1; prefix = ""; space = ""; next
        }
        in_mapping && /^prefix[ \t]*=/ {
            sub(/^prefix[ \t]*=[ \t]*/, ""); sub(/[ \t]*$/, ""); gsub(/"/, "")
            prefix = $0
        }
        in_mapping && /^space[ \t]*=/ {
            sub(/^space[ \t]*=[ \t]*/, ""); sub(/[ \t]*$/, ""); gsub(/"/, "")
            space = $0
        }
        END {
            if (in_mapping && prefix != "" && space != "") print prefix "\t" space
        }
    ' "$spaces_file" 2>/dev/null > "$pairs"

    best_prefix=""
    best_space=""
    while IFS=$'\t' read -r prefix space; do
        case "$prefix" in
            "~/"*) expanded="${HOME}/${prefix#~/}" ;;
            "~") expanded="$HOME" ;;
            *) expanded="$prefix" ;;
        esac
        case "$cwd" in
            "$expanded"|"$expanded/"*)
                if [ ${#expanded} -gt ${#best_prefix} ]; then
                    best_prefix="$expanded"
                    best_space="$space"
                fi
                ;;
        esac
    done < "$pairs"
    rm -f "$pairs"

    if [ -n "$best_space" ]; then
        printf '%s\tcwd-config\n' "$best_space"
        exit 0
    fi
fi

if [ -n "$cwd" ] && [ -d "$cwd" ]; then
    repo_root="$(cd "$cwd" && git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$repo_root" ]; then
        common="$(cd "$cwd" && git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        case "$common" in
            */.git) repo_name="$(basename "$(dirname "$common")")" ;;
            *) repo_name="$(basename "$repo_root")" ;;
        esac

        wenlan_bin="${WENLAN_BIN:-wenlan}"
        if "$wenlan_bin" spaces show "$repo_name" --format json >/dev/null 2>&1; then
            printf '%s\tcwd-repo\n' "$repo_name"
            exit 0
        fi
    fi
fi

printf '\tunscoped\n'
