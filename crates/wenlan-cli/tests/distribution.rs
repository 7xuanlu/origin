// SPDX-License-Identifier: Apache-2.0
//! Distribution and packaging contract tests for Wenlan's user-facing setup paths.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("wenlan-cli is nested under crates/")
        .to_path_buf()
}

fn read_json(relative: &str) -> Value {
    let path = repo_root().join(relative);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn assert_file(relative: &str) {
    let path = repo_root().join(relative);
    assert!(path.is_file(), "missing file: {}", path.display());
}

fn json_string<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string field `{key}` in {value}"))
}

#[test]
fn plugin_distribution_contains_required_files() {
    for path in [
        "plugin/.claude-plugin/plugin.json",
        "plugin/.claude-plugin/README.md",
        "plugin/.mcp.json",
        "plugin/bin/wenlan-mcp-runner.sh",
        "plugin/hooks/hooks.json",
        "plugin/hooks/check-daemon.sh",
        "plugin/skills/brief/SKILL.md",
        "plugin/skills/capture/SKILL.md",
        "plugin/skills/handoff/SKILL.md",
        "plugin/skills/setup/SKILL.md",
        "plugin/skills/distill/SKILL.md",
    ] {
        assert_file(path);
    }
}

#[test]
fn plugin_manifest_and_mcp_launcher_stay_in_sync() {
    let plugin = read_json("plugin/.claude-plugin/plugin.json");
    assert_eq!(json_string(&plugin, "name"), "wenlan");
    assert_eq!(json_string(&plugin, "license"), "Apache-2.0");
    assert_eq!(json_string(&plugin, "category"), "productivity");

    let keywords = plugin["keywords"].as_array().expect("keywords array");
    for keyword in ["claude-code", "memory", "mcp", "local-first"] {
        assert!(
            keywords.iter().any(|value| value == keyword),
            "missing plugin keyword {keyword}"
        );
    }

    let mcp = read_json("plugin/.mcp.json");
    let server = &mcp["mcpServers"]["wenlan"];
    assert_eq!(
        json_string(server, "command"),
        "${CLAUDE_PLUGIN_ROOT}/bin/wenlan-mcp-runner.sh"
    );
}

#[test]
fn npm_package_allowlists_match_release_generated_files() {
    // The wenlan CLI wrapper currently ships a macOS-arm64-only
    // `run.js`. Linux/Windows users install via the Docker image, the tar/zip
    // release archives, or `cargo install`, so the npm allowlist stays narrow
    // on purpose.
    let setup_pkg = read_json("crates/wenlan-cli/npm/package.json");
    assert_eq!(json_string(&setup_pkg, "name"), "wenlan");
    assert_eq!(setup_pkg["bin"]["wenlan"], "run.js");
    assert_eq!(setup_pkg["license"], "Apache-2.0");
    assert_eq!(setup_pkg["os"], serde_json::json!(["darwin"]));
    assert_eq!(setup_pkg["cpu"], serde_json::json!(["arm64"]));
    assert_eq!(
        setup_pkg["files"],
        serde_json::json!(["run.js", "README.md", "LICENSE"])
    );

    // wenlan-mcp ships prebuilt binaries for every release-matrix target
    // (darwin arm64, linux arm64/x64, windows x64) via its npm postinstall. The
    // allowlist must include each platform the matrix uploads or `npm
    // install` rejects the package on those hosts.
    let mcp_pkg = read_json("crates/wenlan-mcp/npm/package.json");
    assert_eq!(json_string(&mcp_pkg, "name"), "wenlan-mcp");
    assert_eq!(mcp_pkg["bin"]["wenlan-mcp"], "run.js");
    assert_eq!(mcp_pkg["scripts"]["postinstall"], "node install.js");
    assert_eq!(mcp_pkg["license"], "Apache-2.0");
    assert_eq!(
        mcp_pkg["os"],
        serde_json::json!(["darwin", "linux", "win32"])
    );
    assert_eq!(mcp_pkg["cpu"], serde_json::json!(["arm64", "x64"]));
    assert_eq!(
        mcp_pkg["files"],
        serde_json::json!(["install.js", "run.js", "README.md", "LICENSE"])
    );
}

#[test]
fn release_workflow_promotes_target_inventory_and_publishes_npm_packages() {
    let workflow = fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
        .expect("read release workflow");
    for needle in [
        "Promote exact validated release assets",
        "scripts/release-promotion.py consume-main-receipt",
        "scripts/release-promotion.py download-assets",
        "name: docker-runtime-inputs",
        "Publish wenlan-mcp",
        "Publish wenlan",
        "cp README.md crates/wenlan-mcp/npm/README.md",
        "cp README.md crates/wenlan-cli/npm/README.md",
        "wenlan-mcp-darwin-arm64.tar.gz",
    ] {
        assert!(
            workflow.contains(needle),
            "release workflow missing `{needle}`"
        );
    }

    let inventory = Command::new("python3")
        .arg(repo_root().join("scripts/release_targets.py"))
        .arg("matrix")
        .output()
        .expect("run canonical release-target inventory");
    assert!(
        inventory.status.success(),
        "release-target inventory failed:\n{}",
        String::from_utf8_lossy(&inventory.stderr)
    );
    let matrix: Value =
        serde_json::from_slice(&inventory.stdout).expect("parse canonical release-target matrix");
    let artifact_names = matrix["include"]
        .as_array()
        .expect("release matrix include array")
        .iter()
        .map(|entry| json_string(entry, "artifact_name"))
        .collect::<Vec<_>>();
    assert_eq!(
        artifact_names,
        [
            "wenlan-darwin-arm64",
            "wenlan-linux-arm64",
            "wenlan-linux-x64",
            "wenlan-windows-x64",
        ],
        "canonical release artifact inventory drift"
    );
}

#[test]
#[ignore = "manual smoke: requires real codex CLI and may download wenlan-mcp through npx"]
fn smoke_codex_mcp_add_uses_temp_home() {
    let runtime = TempDir::new().expect("temp home");
    let origin = assert_cmd::cargo::cargo_bin("wenlan");

    let output = Command::new(origin)
        .env("HOME", runtime.path())
        .env("WENLAN_HOST", "http://127.0.0.1:9")
        .args(["mcp", "add", "codex"])
        .output()
        .expect("run origin mcp add codex");

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let config = runtime.path().join(".codex/config.toml");
    assert!(
        config.exists(),
        "expected Codex config at {}",
        config.display()
    );
    let text = fs::read_to_string(config).expect("read Codex config");
    assert!(text.contains("wenlan"), "{text}");
    assert!(text.contains("wenlan-mcp"), "{text}");
}
