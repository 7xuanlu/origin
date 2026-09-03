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
    "$TMPDIR_TEST/plugin-codex/skills/setup" \
    "$TMPDIR_TEST/app"
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
name = "wenlan-app"
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
cat > "$TMPDIR_TEST/app/Cargo.toml" <<EOF
[package]
name = "wenlan-app"
version = "0.5.0" # x-release-please-version
EOF
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/app/tauri.conf.json"
echo '{"name": "wenlan-app", "version": "0.5.0"}' > "$TMPDIR_TEST/package.json"
cat > "$TMPDIR_TEST/plugin-codex/bin/wenlan-mcp-runner.sh" <<EOF
exec npx -y wenlan-mcp@^0.5.0 --agent-name "\${agent_name}" "\$@"
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

# Test 2b: app trio version mismatch → exit 1
echo '{"version": "0.4.9"}' > "$TMPDIR_TEST/app/tauri.conf.json"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 2b: should have detected app/tauri.conf.json drift"
    exit 1
fi
echo "PASS test 2b: app/tauri.conf.json drift detected"
echo '{"version": "0.5.0"}' > "$TMPDIR_TEST/app/tauri.conf.json"

# Test 2c: app/Cargo.toml version mismatch → exit 1
cat > "$TMPDIR_TEST/app/Cargo.toml" <<EOF
[package]
name = "wenlan-app"
version = "0.4.9" # x-release-please-version
EOF
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 2c: should have detected app/Cargo.toml drift"
    exit 1
fi
echo "PASS test 2c: app/Cargo.toml drift detected"
cat > "$TMPDIR_TEST/app/Cargo.toml" <<EOF
[package]
name = "wenlan-app"
version = "0.5.0" # x-release-please-version
EOF

# Test 2d: package.json version mismatch → exit 1
echo '{"name": "wenlan-app", "version": "0.4.9"}' > "$TMPDIR_TEST/package.json"
if (cd "$TMPDIR_TEST" && RELEASE_TAG="v0.5.0" bash "$OLDPWD/scripts/validate-versions.sh") 2>/dev/null; then
    echo "FAIL test 2d: should have detected package.json drift"
    exit 1
fi
echo "PASS test 2d: package.json drift detected"
echo '{"name": "wenlan-app", "version": "0.5.0"}' > "$TMPDIR_TEST/package.json"

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

assert_release_job_pins_release_sha() {
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
if not checkout or not re.search(r"^          ref: \$\{\{ env\.RELEASE_SHA \}\}\s*$", checkout.group("body"), re.MULTILINE):
    raise SystemExit(1)
verify = re.search(r"^      - name: Verify release checkout\n(?P<body>.*?)(?=^      - |\Z)", body, re.MULTILINE | re.DOTALL)
if not verify or not re.search(r"^        shell: bash\s*$", verify.group("body"), re.MULTILINE):
    raise SystemExit(1)
if any(marker not in verify.group("body") for marker in ['git rev-parse HEAD', 'RELEASE_SHA', '/git/ref/tags/$RELEASE_TAG']):
    raise SystemExit(1)
PY
}

for job in prepare-release publish-crates publish-npm; do
    if ! assert_release_job_pins_release_sha "$REPO_ROOT/.github/workflows/release.yml" "$job"; then
        echo "FAIL test 10: $job must checkout the resolved release SHA and verify RELEASE_TAG"
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
if "ref: ${{ github.sha }}" not in body or "ref: ${{ env.RELEASE_SHA }}" in body:
    raise SystemExit(1)
for marker in [
    'git rev-parse HEAD)" == "$GITHUB_SHA',
    '/git/ref/tags/$RELEASE_TAG',
    '--sha "$RELEASE_SHA"',
    "scripts/release-promotion.py gate-main",
    "scripts/release-promotion.py consume-main-receipt",
    '"$GITHUB_REF" == "refs/heads/main"',
    ".main_sha == $sha and .main_run == null",
    ".receipt.run_id == $source_run_id",
    ".receipt.run_attempt == $source_run_attempt",
    "Tag does not match the validated release version.",
]:
    if marker not in body:
        raise SystemExit(1)
PY
then
    echo "FAIL test 10: release resolver must keep main control-plane code while binding RELEASE_TAG to the exact receipt"
    exit 1
fi
if ! python3 - "$REPO_ROOT/.github/workflows/release.yml" <<'PY'
import re
import sys
from pathlib import Path

workflow = Path(sys.argv[1]).read_text()


def job_body(job):
    match = re.search(
        rf"^  {re.escape(job)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise SystemExit(1)
    return match.group("body")


# promote-assets runs the same promotion resolver as resolve-promotion, so it
# is control plane: pinning it to the release commit would run the release's
# own copy of the tooling, and a resolver fix could never reach a recovery of
# that release. It still revalidates the receipt-derived tag.
# The docker job is the same shape: its checkout supplies only the runtime
# Dockerfile and the image verifier, while the bytes placed in the image come
# from the receipt-verified artifact.
for job in ["promote-assets", "docker"]:
    packaging = job_body(job)
    if (
        "ref: ${{ github.sha }}" not in packaging
        or "ref: ${{ env.RELEASE_SHA }}" in packaging
    ):
        raise SystemExit(1)
    if "/git/ref/tags/$RELEASE_TAG" not in packaging or "RELEASE_SHA" not in packaging:
        raise SystemExit(1)

for job in ["prepare-release", "publish-crates", "publish-npm"]:
    body = job_body(job)
    if "ref: ${{ env.RELEASE_SHA }}" not in body or "ref: ${{ github.sha }}" in body:
        raise SystemExit(1)
    if "/git/ref/tags/$RELEASE_TAG" not in body or "RELEASE_SHA" not in body:
        raise SystemExit(1)
PY
then
    echo "FAIL test 10: every release source/publish job must use RELEASE_SHA and revalidate the live tag"
    exit 1
fi
echo "PASS test 10: promotion keeps immutable main control code; source/publish jobs pin RELEASE_SHA and verify RELEASE_TAG"

python3 "$REPO_ROOT/scripts/release-workflow-contract.test.py"
echo "PASS test 11: release promotion and public install contracts"

bash "$REPO_ROOT/scripts/bump-version.test.sh"
echo "PASS test 12: release version sync disables npm lifecycle scripts"
