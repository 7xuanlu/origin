#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMPDIR_TEST=$(mktemp -d)
trap "rm -rf $TMPDIR_TEST" EXIT

mkdir -p \
    "$TMPDIR_TEST/crates/wenlan-mcp/npm" \
    "$TMPDIR_TEST/crates/wenlan-cli/npm" \
    "$TMPDIR_TEST/plugin/.claude-plugin" \
    "$TMPDIR_TEST/plugin-codex/.codex-plugin" \
    "$TMPDIR_TEST/plugin-codex/bin" \
    "$TMPDIR_TEST/plugin-codex/skills/setup"
echo "0.5.0" > "$TMPDIR_TEST/version.txt"
cat > "$TMPDIR_TEST/Cargo.toml" <<EOF
[workspace.package]
version = "0.5.0"   # x-release-please-version

[workspace.dependencies]
wenlan-types = { path = "crates/wenlan-types", version = "0.5.0" }
wenlan-core  = { path = "crates/wenlan-core",  version = "0.5.0" }
EOF
cat > "$TMPDIR_TEST/Cargo.lock" <<EOF
[[package]]
name = "wenlan"
version = "0.5.0"

[[package]]
name = "wenlan-core"
version = "0.5.0"

[[package]]
name = "wenlan-mcp"
version = "0.5.0"

[[package]]
name = "wenlan-server"
version = "0.5.0"

[[package]]
name = "wenlan-types"
version = "0.5.0"
EOF
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/crates/wenlan-mcp/npm/package.json"
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/crates/wenlan-cli/npm/package.json"
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/plugin/.claude-plugin/plugin.json"
echo '{"version": "0.5.0+codex"}' > "$TMPDIR_TEST/plugin-codex/.codex-plugin/plugin.json"
cat > "$TMPDIR_TEST/plugin-codex/bin/wenlan-mcp-runner.sh" <<EOF
exec npx -y wenlan-mcp@^0.5.0 --agent-name "\${agent_name}" "\$@"
EOF
cat > "$TMPDIR_TEST/plugin-codex/README.md" <<EOF
Fallbacks to npx -y wenlan-mcp@^0.5.0 when no local runtime exists.
EOF
cat > "$TMPDIR_TEST/plugin-codex/skills/setup/SKILL.md" <<EOF
curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v0.5.0/install.sh | bash
EOF

# Test 1: all match → exit 0
(cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh")
echo "PASS test 1: all matching"

# Test 2: mismatch → exit 1
echo '{"version": "0.4.9"}' > "$TMPDIR_TEST/plugin/.claude-plugin/plugin.json"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 2: should have detected drift"
    exit 1
fi
echo "PASS test 2: drift detected"

# Test 3: internal workspace dependency mismatch → exit 1
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/plugin/.claude-plugin/plugin.json"
perl -0pi -e 's/wenlan-core  = \{ path = "crates\/wenlan-core",  version = "0\.5\.0" \}/wenlan-core  = { path = "crates\/wenlan-core",  version = "0.4.9" }/' "$TMPDIR_TEST/Cargo.toml"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 3: should have detected internal dependency drift"
    exit 1
fi
echo "PASS test 3: internal dependency drift detected"

# Test 4: Cargo.lock mismatch → exit 1
perl -0pi -e 's/wenlan-core  = \{ path = "crates\/wenlan-core",  version = "0\.4\.9" \}/wenlan-core  = { path = "crates\/wenlan-core",  version = "0.5.0" }/' "$TMPDIR_TEST/Cargo.toml"
perl -0pi -e 's/name = "wenlan-core"\nversion = "0\.5\.0"/name = "wenlan-core"\nversion = "0.4.9"/' "$TMPDIR_TEST/Cargo.lock"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 4: should have detected Cargo.lock drift"
    exit 1
fi
echo "PASS test 4: Cargo.lock drift detected"

perl -0pi -e 's/name = "wenlan-core"\nversion = "0\.4\.9"/name = "wenlan-core"\nversion = "0.5.0"/' "$TMPDIR_TEST/Cargo.lock"
echo '{"version": "0.4.9+codex"}' > "$TMPDIR_TEST/plugin-codex/.codex-plugin/plugin.json"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 5: should have detected Codex plugin manifest drift"
    exit 1
fi
echo "PASS test 5: Codex plugin manifest drift detected"

echo '{"version": "0.5.0+codex"}' > "$TMPDIR_TEST/plugin-codex/.codex-plugin/plugin.json"
perl -0pi -e 's/wenlan-mcp@\^0\.5\.0/wenlan-mcp@^0.4.9/g' "$TMPDIR_TEST/plugin-codex/bin/wenlan-mcp-runner.sh"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 6: should have detected Codex runner pin drift"
    exit 1
fi
echo "PASS test 6: Codex runner pin drift detected"

perl -0pi -e 's/wenlan-mcp@\^0\.4\.9/wenlan-mcp@^0.5.0/g' "$TMPDIR_TEST/plugin-codex/bin/wenlan-mcp-runner.sh"
perl -0pi -e 's|/v0\.5\.0/install\.sh|/v0.4.9/install.sh|g' "$TMPDIR_TEST/plugin-codex/skills/setup/SKILL.md"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 7: should have detected Codex setup install tag drift"
    exit 1
fi
echo "PASS test 7: Codex setup install tag drift detected"

perl -0pi -e 's|/v0\.4\.9/install\.sh|/v0.5.0/install.sh|g' "$TMPDIR_TEST/plugin-codex/skills/setup/SKILL.md"
perl -0pi -e 's/wenlan-mcp@\^0\.5\.0/wenlan-mcp@^0.4.9/g' "$TMPDIR_TEST/plugin-codex/README.md"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 8: should have detected Codex README runner pin drift"
    exit 1
fi
echo "PASS test 8: Codex README runner pin drift detected"

cat > "$TMPDIR_TEST/plugin-codex/README.md" <<EOF
No package fallback is documented here.
EOF
if output=$(cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh" 2>&1); then
    echo "FAIL test 9: should have detected missing Codex README runner pin"
    exit 1
fi
if ! printf '%s\n' "$output" | grep -q "Codex plugin release pin missing"; then
    echo "FAIL test 9: missing pin error was not reported"
    printf '%s\n' "$output"
    exit 1
fi
echo "PASS test 9: Codex README runner pin missing detected"

assert_release_job_pins_tag() {
    local workflow="$1"
    local job="$2"
    python3 - "$workflow" "$job" <<'PY'
import re
import sys
from pathlib import Path

workflow = Path(sys.argv[1]).read_text()
job = re.escape(sys.argv[2])
match = re.search(rf"^  {job}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", workflow, re.MULTILINE | re.DOTALL)
if not match:
    raise SystemExit(1)
body = match.group("body")
checkout = re.search(r"^      - (?:name: Checkout\n        )?uses: actions/checkout@[^\n]+\n(?P<body>.*?)(?=^      - |\Z)", body, re.MULTILINE | re.DOTALL)
if not checkout or not re.search(r"^          ref: refs/tags/\$\{\{ env\.RELEASE_TAG \}\}\s*$", checkout.group("body"), re.MULTILINE):
    raise SystemExit(1)
verify = re.search(r"^      - name: Verify release checkout\n(?P<body>.*?)(?=^      - |\Z)", body, re.MULTILINE | re.DOTALL)
if not verify or not re.search(r"^        shell: bash\s*$", verify.group("body"), re.MULTILINE):
    raise SystemExit(1)
if 'git rev-parse HEAD' not in verify.group("body") or 'git rev-list -n1 "refs/tags/$RELEASE_TAG"' not in verify.group("body"):
    raise SystemExit(1)
PY
}

for job in prepare-release publish-crates publish-npm; do
    if ! assert_release_job_pins_tag "$REPO_ROOT/.github/workflows/release.yml" "$job"; then
        echo "FAIL test 10: $job must checkout and verify RELEASE_TAG"
        exit 1
    fi
done
if ! python3 - "$REPO_ROOT/.github/workflows/release.yml" <<'PY'
import re
import sys
from pathlib import Path

workflow = Path(sys.argv[1]).read_text()
match = re.search(
    r"^  resolve-promotion:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
    workflow,
    re.MULTILINE | re.DOTALL,
)
if not match:
    raise SystemExit(1)
body = match.group("body")
for marker in [
    "ref: refs/tags/${{ env.RELEASE_TAG }}",
    'tag_sha="$(git rev-list -n1 "refs/tags/$RELEASE_TAG")"',
    "scripts/release-promotion.py consume-main-receipt",
    "Tag does not match the validated release version.",
]:
    if marker not in body:
        raise SystemExit(1)
PY
then
    echo "FAIL test 10: release resolver must bind RELEASE_TAG to the main promotion receipt"
    exit 1
fi
echo "PASS test 10: release jobs pin the receipt-derived RELEASE_TAG"

python3 "$REPO_ROOT/scripts/release-workflow-contract.test.py"
echo "PASS test 11: release promotion and public install contracts"
