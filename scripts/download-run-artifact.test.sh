#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
helper="$repo_root/scripts/download-run-artifact.sh"
test_dir=$(mktemp -d)
trap 'rm -rf -- "$test_dir"' EXIT

mock_gh="$test_dir/gh"
cat > "$mock_gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
attempts=$(cat "$MOCK_ATTEMPTS" 2>/dev/null || printf 0)
attempts=$((attempts + 1))
printf '%s' "$attempts" > "$MOCK_ATTEMPTS"
if (( attempts < MOCK_SUCCEED_ON )); then
    exit 1
fi
while [[ $# -gt 0 ]]; do
    if [[ $1 == --dir ]]; then
        mkdir -p "$2"
        printf 'fixture artifact' > "$2/model.bin"
        exit 0
    fi
    shift
done
exit 64
EOF
chmod +x "$mock_gh"

attempts="$test_dir/attempts"
destination="$test_dir/output"
GH_BIN="$mock_gh" \
GH_TOKEN=fixture \
GITHUB_REPOSITORY=owner/repo \
MOCK_ATTEMPTS="$attempts" \
MOCK_SUCCEED_ON=2 \
WENLAN_ARTIFACT_DOWNLOAD_ATTEMPTS=3 \
WENLAN_ARTIFACT_DOWNLOAD_RETRY_DELAY_SECONDS=0 \
    bash "$helper" 123 artifact "$destination"

[[ $(cat "$attempts") == 2 ]]
[[ $(cat "$destination/model.bin") == 'fixture artifact' ]]

rm -f "$attempts"
if GH_BIN="$mock_gh" \
    GH_TOKEN=fixture \
    GITHUB_REPOSITORY=owner/repo \
    MOCK_ATTEMPTS="$attempts" \
    MOCK_SUCCEED_ON=9 \
    WENLAN_ARTIFACT_DOWNLOAD_ATTEMPTS=2 \
    WENLAN_ARTIFACT_DOWNLOAD_RETRY_DELAY_SECONDS=0 \
        bash "$helper" 123 artifact "$test_dir/never"; then
    echo "FAIL: exhausted artifact download unexpectedly succeeded" >&2
    exit 1
fi
[[ $(cat "$attempts") == 2 ]]

if GH_BIN="$mock_gh" GH_TOKEN=fixture GITHUB_REPOSITORY=owner/repo \
    bash "$helper" invalid artifact "$test_dir/invalid"; then
    echo "FAIL: non-numeric run id was accepted" >&2
    exit 1
fi

echo "download-run-artifact tests PASS"
