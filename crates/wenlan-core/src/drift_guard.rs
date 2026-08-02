//! Fail-loud drift guards (test-only). Each `#[test]` here is a CI + pre-push gate
//! that makes a class of doc/flag/config drift structurally hard. Mirrors the
//! `seed_contract.rs` teeth pattern. See docs/superpowers/specs/2026-06-19-drift-defense-system-design.md.
//!
//! Failure messages teach the fix: each gate's assert states the invariant that
//! broke, where to fix it, and the escape hatch when the contract itself changed
//! on purpose (update the `*_violations` fn AND its positive control). New teeth
//! follow the same form.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Expr, Lit, Pat, Token};

#[cfg(test)]
#[path = "drift_guard/r4_test_support_test.rs"]
mod r4_test_support_test;

#[cfg(test)]
#[path = "drift_guard/post_write_structure_test.rs"]
mod post_write_structure_test;

/// Repo root, resolved at compile time from this crate's manifest dir
/// (crates/wenlan-core -> ../.. == repo root).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve repo root")
}

/// Tracked files matching a git pathspec, relative to repo root.
fn git_ls_files(root: &Path, pattern: &str) -> Vec<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", pattern])
        .output()
        .expect("run git ls-files");
    assert!(out.status.success(), "git ls-files failed for {pattern}");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.to_string())
        .collect()
}

// ── Teeth #3: version-file byte-identical assert ──

/// The version string carried by each release-please-managed source of truth.
/// The 4 daemon crates use `version.workspace = true`, so the only Cargo version
/// is the root workspace one. Plus the CC plugin manifest (`plugin.json`), kept on
/// the same release train via `release-please-config.json` `extra-files` so the
/// plugin can't silently lag the daemon (the recurring version-drift nag). 4 sources.
fn version_sources() -> Vec<(String, String)> {
    let root = repo_root();
    let mut out = Vec::new();

    let vt = std::fs::read_to_string(root.join("version.txt")).expect("read version.txt");
    out.push(("version.txt".to_string(), vt.trim().to_string()));

    let mf =
        std::fs::read_to_string(root.join(".release-please-manifest.json")).expect("read manifest");
    let mfj: serde_json::Value = serde_json::from_str(&mf).expect("parse manifest json");
    out.push((
        ".release-please-manifest.json".to_string(),
        mfj["."].as_str().expect("manifest \".\" key").to_string(),
    ));

    let ct = std::fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");
    let line = ct
        .lines()
        .find(|l| l.contains("x-release-please-version"))
        .expect("workspace version line with x-release-please-version marker");
    let re = regex::Regex::new(r#""([0-9]+\.[0-9]+\.[0-9]+[^"]*)""#).unwrap();
    let v = re.captures(line).expect("version literal on marker line")[1].to_string();
    out.push(("Cargo.toml".to_string(), v));

    let pj = std::fs::read_to_string(root.join("plugin/.claude-plugin/plugin.json"))
        .expect("read plugin.json");
    let pjj: serde_json::Value = serde_json::from_str(&pj).expect("parse plugin.json");
    out.push((
        "plugin/.claude-plugin/plugin.json".to_string(),
        pjj["version"]
            .as_str()
            .expect("plugin.json \"version\" key")
            .to_string(),
    ));

    out
}

#[test]
fn version_files_are_in_sync() {
    let sources = version_sources();
    let (_, first) = &sources[0];
    let mismatched: Vec<&(String, String)> = sources.iter().filter(|(_, v)| v != first).collect();
    assert!(
        mismatched.is_empty(),
        "version drift across release-please files: {sources:?} (expected all == {first}). \
         Fix: set the SAME version string in version.txt, .release-please-manifest.json, and \
         the workspace Cargo.toml version line — a manual bump must touch all three (teeth #3)"
    );
}

#[test]
fn version_sync_detects_mismatch() {
    // Pure-logic guard: a hand-built mismatched set must be flagged.
    let sources = [
        ("a".to_string(), "0.8.4".to_string()),
        ("b".to_string(), "0.8.5".to_string()),
    ];
    let (_, first) = &sources[0];
    let mismatched: Vec<_> = sources.iter().filter(|(_, v)| v != first).collect();
    assert_eq!(mismatched.len(), 1, "mismatch must be detected");
}

// ── Teeth #5: FastEmbed CI distribution contract ──

fn fastembed_ci_cache_violations(workflow: &str) -> Vec<String> {
    const DOWNLOAD_STEP: &str = "Download portable FastEmbed model";
    const CACHE_DIR: &str = "${{ github.workspace }}/.fastembed_cache";
    const CACHE_PATH: &str = ".fastembed_cache";
    const CACHE_KEY: &str = "fastembed-bge-base-en-v1.5-q-v3-portable";
    const ARTIFACT_NAME: &str = "fastembed-bge-base-en-v1.5-q-v3-portable-${{ github.run_id }}";
    const JOBS: &[(&str, &[&str])] = &[
        ("linux-nextest", &["Run workspace-lib archive partition"]),
        (
            "test",
            &[
                "Workspace lib tests (macOS)",
                "Integration tests wenlan-cli + wenlan-server (Windows)",
            ],
        ),
        (
            "canonical-acceptance",
            &["Integration tests wenlan-cli + wenlan-server (Linux)"],
        ),
        (
            "contract-integration",
            &["Affected wenlan-mcp + wenlan-types integrations"],
        ),
        (
            "release-preflight",
            &["Native ORT smoke (Windows release preflight)"],
        ),
    ];

    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse ci.yml");
    let mut violations = Vec::new();

    for (job_name, consumer_names) in JOBS {
        let actual_cache_dir = parsed["jobs"][*job_name]["env"]["FASTEMBED_CACHE_DIR"].as_str();
        if actual_cache_dir != Some(CACHE_DIR) {
            violations.push(format!(
                "job {job_name} sets FASTEMBED_CACHE_DIR={actual_cache_dir:?}, expected {CACHE_DIR:?}"
            ));
        }

        let Some(steps) = parsed["jobs"][*job_name]["steps"].as_sequence() else {
            violations.push(format!("job {job_name} has no steps"));
            continue;
        };
        if steps.iter().any(|step| {
            step["run"]
                .as_str()
                .is_some_and(|run| run.contains("WENLAN_TEST_FASTEMBED_CACHE"))
        }) {
            violations.push(format!(
                "job {job_name} overrides FASTEMBED_CACHE_DIR with WENLAN_TEST_FASTEMBED_CACHE"
            ));
        }
        let download_indexes: Vec<usize> = steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                (step["name"].as_str() == Some(DOWNLOAD_STEP)).then_some(index)
            })
            .collect();
        if download_indexes.len() != 1 {
            violations.push(format!(
                "job {job_name} has {} {DOWNLOAD_STEP:?} steps, expected 1",
                download_indexes.len()
            ));
            continue;
        }

        let download_index = download_indexes[0];
        let download = &steps[download_index];
        if download["uses"]
            .as_str()
            .is_none_or(|uses| !uses.starts_with("actions/download-artifact@"))
        {
            violations.push(format!(
                "job {job_name} does not download the prepared FastEmbed artifact"
            ));
        }
        let actual_path = download["with"]["path"].as_str();
        if actual_path != Some(CACHE_PATH) {
            violations.push(format!(
                "job {job_name} downloads FastEmbed to {actual_path:?}, expected {CACHE_PATH:?}"
            ));
        }
        let actual_name = download["with"]["name"].as_str();
        if actual_name != Some(ARTIFACT_NAME) {
            violations.push(format!(
                "job {job_name} uses FastEmbed artifact name {actual_name:?}, expected {ARTIFACT_NAME:?}"
            ));
        }
        if download["id"].as_str() != Some("fastembed-download")
            || download["continue-on-error"].as_bool() != Some(true)
        {
            violations.push(format!(
                "job {job_name} does not expose the official artifact action outcome for bounded retry"
            ));
        }
        let retry_indexes = steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                (step["name"].as_str() == Some("Retry portable FastEmbed model download"))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let expected_retry_if = if *job_name == "release-preflight" {
            "matrix.target == 'x86_64-pc-windows-msvc' && steps.fastembed-download.outcome == 'failure'"
        } else {
            "steps.fastembed-download.outcome == 'failure'"
        };
        let retry_valid = retry_indexes.as_slice() == [download_index + 1]
            && retry_indexes.first().is_some_and(|index| {
                let retry = &steps[*index];
                retry["if"].as_str() == Some(expected_retry_if)
                    && retry["env"]["GH_TOKEN"].as_str() == Some("${{ github.token }}")
                    && retry["run"].as_str()
                        == Some(
                            "bash scripts/download-run-artifact.sh \"$GITHUB_RUN_ID\" \"fastembed-bge-base-en-v1.5-q-v3-portable-${GITHUB_RUN_ID}\" .fastembed_cache",
                        )
            });
        if !retry_valid {
            violations.push(format!(
                "job {job_name} lacks the exact fail-closed FastEmbed artifact retry fallback"
            ));
        }
        if steps.iter().any(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with("actions/cache/restore@"))
        }) {
            violations.push(format!(
                "job {job_name} still performs a concurrent FastEmbed cache restore"
            ));
        }
        if *job_name == "test" && !download["if"].is_null() {
            violations.push("test downloads the model only on a subset of matrix OSes".into());
        }
        if *job_name == "release-preflight"
            && download["if"].as_str() != Some("matrix.target == 'x86_64-pc-windows-msvc'")
        {
            violations
                .push("release-preflight does not scope the FastEmbed download to Windows".into());
        }

        for consumer_name in *consumer_names {
            let consumer_index = steps
                .iter()
                .position(|step| step["name"].as_str() == Some(consumer_name));
            match consumer_index {
                Some(index) if download_index < index => {}
                Some(index) => violations.push(format!(
                    "job {job_name} downloads FastEmbed at step {download_index} after consumer step {index}"
                )),
                None => violations.push(format!(
                    "job {job_name} is missing consumer step {consumer_name:?}"
                )),
            }
            if *job_name == "release-preflight" {
                let cache_override = consumer_index
                    .and_then(|index| steps.get(index))
                    .and_then(|step| step["env"]["WENLAN_TEST_FASTEMBED_CACHE"].as_str());
                if cache_override != Some("${{ env.FASTEMBED_CACHE_DIR }}") {
                    violations.push(format!(
                        "release-preflight consumer {consumer_name:?} does not use the prepared FastEmbed cache"
                    ));
                }
            }
        }
    }

    let detect_steps = parsed["jobs"]["detect-changes"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let detect_index = |name: &str| {
        detect_steps
            .iter()
            .position(|step| step["name"].as_str() == Some(name))
    };
    let prep_order = [
        detect_index("Test FastEmbed cache preparation"),
        detect_index("Restore portable FastEmbed model"),
        detect_index("Restore legacy Linux FastEmbed model"),
        detect_index("Prepare portable FastEmbed model"),
        detect_index("Publish portable FastEmbed model for this run"),
        detect_index("Save portable FastEmbed model"),
    ];
    if prep_order.iter().any(Option::is_none)
        || !prep_order
            .windows(2)
            .all(|pair| pair[0].is_some_and(|left| pair[1].is_some_and(|right| left < right)))
    {
        violations.push(
            "detect-changes does not test, restore, prepare, publish, then save the portable model"
                .into(),
        );
    }
    let retry_test = detect_index("Test artifact download retry")
        .and_then(|index| detect_steps.get(index).copied());
    if retry_test.and_then(|step| step["run"].as_str())
        != Some("bash scripts/download-run-artifact.test.sh")
    {
        violations.push("detect-changes does not test the bounded artifact retry helper".into());
    }
    let prepare = detect_index("Prepare portable FastEmbed model")
        .and_then(|index| detect_steps.get(index).copied());
    if prepare.and_then(|step| step["run"].as_str())
        != Some("python3 scripts/prepare-fastembed-cache.py")
    {
        violations.push("detect-changes does not run the pinned FastEmbed preparer".into());
    }
    let rust_ci_condition = "steps.release-proof.outputs.verified-release-merge != 'true' && steps.test-plan.outputs.rust-ci-required == 'true'";
    for name in [
        "Restore portable FastEmbed model",
        "Prepare portable FastEmbed model",
        "Publish portable FastEmbed model for this run",
    ] {
        let condition = detect_index(name)
            .and_then(|index| detect_steps.get(index).copied())
            .and_then(|step| step["if"].as_str());
        if condition != Some(rust_ci_condition) {
            violations.push(format!(
                "detect-changes {name:?} is not gated by the canonical rust-ci-required planner output"
            ));
        }
    }
    let cache_miss_condition = format!(
        "{rust_ci_condition} && steps.fastembed-portable-restore.outputs.cache-hit != 'true'"
    );
    for name in [
        "Restore legacy Linux FastEmbed model",
        "Save portable FastEmbed model",
    ] {
        let condition = detect_index(name)
            .and_then(|index| detect_steps.get(index).copied())
            .and_then(|step| step["if"].as_str());
        if condition != Some(cache_miss_condition.as_str()) {
            violations.push(format!(
                "detect-changes {name:?} is not gated by rust-ci-required plus the portable-cache miss"
            ));
        }
    }
    for name in [
        "Restore portable FastEmbed model",
        "Save portable FastEmbed model",
    ] {
        let Some(step) = detect_index(name).and_then(|index| detect_steps.get(index).copied())
        else {
            continue;
        };
        if step["with"]["path"].as_str() != Some(CACHE_PATH)
            || step["with"]["key"].as_str() != Some(CACHE_KEY)
            || step["with"]["enableCrossOsArchive"].as_str() != Some("true")
        {
            violations.push(format!(
                "detect-changes {name:?} does not use the portable cache contract"
            ));
        }
    }
    let restore = detect_index("Restore portable FastEmbed model")
        .and_then(|index| detect_steps.get(index).copied());
    if restore.and_then(|step| step["with"]["fail-on-cache-miss"].as_str()) == Some("true") {
        violations
            .push("detect-changes treats a FastEmbed cache miss as a correctness failure".into());
    }
    let save = detect_index("Save portable FastEmbed model")
        .and_then(|index| detect_steps.get(index).copied());
    if save.and_then(|step| step["continue-on-error"].as_bool()) != Some(true) {
        violations.push("detect-changes treats a FastEmbed cache save failure as fatal".into());
    }
    let publish = detect_index("Publish portable FastEmbed model for this run")
        .and_then(|index| detect_steps.get(index).copied());
    if publish
        .and_then(|step| step["uses"].as_str())
        .is_none_or(|uses| !uses.starts_with("actions/upload-artifact@"))
        || publish.and_then(|step| step["with"]["name"].as_str()) != Some(ARTIFACT_NAME)
        || publish.and_then(|step| step["with"]["path"].as_str()) != Some(CACHE_PATH)
        || publish.and_then(|step| step["with"]["include-hidden-files"].as_bool()) != Some(true)
        || publish.and_then(|step| step["with"]["compression-level"].as_u64()) != Some(0)
        || publish.and_then(|step| step["with"]["retention-days"].as_u64()) != Some(1)
        || publish.and_then(|step| step["with"]["if-no-files-found"].as_str()) != Some("error")
        || publish.and_then(|step| step["with"]["overwrite"].as_bool()) != Some(true)
    {
        violations.push(
            "detect-changes does not publish one run-scoped verified FastEmbed artifact".into(),
        );
    }

    violations
}

fn coverage_fastembed_cache_violations(workflow: &str) -> Vec<String> {
    const CACHE_STEP: &str = "Restore portable FastEmbed model";
    const CACHE_DIR: &str = "${{ github.workspace }}/.fastembed_cache";
    const CACHE_PATH: &str = "${{ env.FASTEMBED_CACHE_DIR }}";
    const CACHE_KEY: &str = "fastembed-bge-base-en-v1.5-q-v3-portable";

    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse coverage.yml");
    let mut violations = Vec::new();
    let actual_cache_dir = parsed["jobs"]["coverage"]["env"]["FASTEMBED_CACHE_DIR"].as_str();
    if actual_cache_dir != Some(CACHE_DIR) {
        violations.push(format!(
            "coverage sets FASTEMBED_CACHE_DIR={actual_cache_dir:?}, expected {CACHE_DIR:?}"
        ));
    }

    let Some(steps) = parsed["jobs"]["coverage"]["steps"].as_sequence() else {
        violations.push("coverage job has no steps".into());
        return violations;
    };
    let cache_steps: Vec<&serde_yaml::Value> = steps
        .iter()
        .filter(|step| step["name"].as_str() == Some(CACHE_STEP))
        .collect();
    if cache_steps.len() != 1 {
        violations.push(format!(
            "coverage has {} {CACHE_STEP:?} steps, expected 1",
            cache_steps.len()
        ));
        return violations;
    }
    let cache_step = cache_steps[0];
    let actual_path = cache_step["with"]["path"].as_str();
    if actual_path != Some(CACHE_PATH) {
        violations.push(format!(
            "coverage caches {actual_path:?}, expected {CACHE_PATH:?}"
        ));
    }
    let actual_key = cache_step["with"]["key"].as_str();
    if actual_key != Some(CACHE_KEY) {
        violations.push(format!(
            "coverage uses FastEmbed cache key {actual_key:?}, expected {CACHE_KEY:?}"
        ));
    }
    if cache_step["uses"]
        .as_str()
        .is_none_or(|uses| !uses.starts_with("actions/cache/restore@"))
        || cache_step["with"]["enableCrossOsArchive"].as_str() != Some("true")
    {
        violations
            .push("coverage does not restore the cross-OS portable FastEmbed v3 cache".into());
    }
    if steps.iter().any(|step| {
        step["uses"].as_str().is_some_and(|uses| {
            uses.starts_with("actions/cache/save@") || uses == "actions/cache@v4"
        })
    }) {
        violations.push("coverage contains a FastEmbed cache writer".into());
    }

    violations
}

fn coverage_single_test_execution_violations(workflow: &str) -> Vec<String> {
    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse coverage.yml");
    let main_owned_cache = "${{ github.ref == 'refs/heads/main' }}";
    let rust_cache = job_step_using(&parsed, "coverage", "Swatinem/rust-cache");
    let mut violations = Vec::new();
    if rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some(main_owned_cache) {
        violations.push("coverage cache writes are not restricted to main".into());
    }
    if rust_cache.and_then(|step| step["with"]["cache-all-crates"].as_str()) != Some("false")
        || rust_cache.and_then(|step| step["with"]["cache-targets"].as_str()) != Some("false")
    {
        violations.push(
            "coverage rust-cache footprint must exclude crates.io and instrumented targets".into(),
        );
    }
    let Some(run) = parsed["jobs"]["coverage"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .find(|step| {
            step["name"].as_str() == Some("Run Rust coverage (wenlan-core + wenlan-server)")
        })
        .and_then(|step| step["run"].as_str())
    else {
        violations.push("coverage job is missing its Rust coverage step".into());
        return violations;
    };

    let lines = run.lines().collect::<Vec<_>>();
    let starts = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            line.trim_start()
                .starts_with("cargo llvm-cov")
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let commands = starts
        .iter()
        .enumerate()
        .map(|(position, start)| {
            let end = starts.get(position + 1).copied().unwrap_or(lines.len());
            lines[*start..end].join(" ")
        })
        .collect::<Vec<_>>();
    let test_commands = commands
        .iter()
        .filter(|command| !command.trim_start().starts_with("cargo llvm-cov report"))
        .collect::<Vec<_>>();
    let report_commands = commands
        .iter()
        .filter(|command| command.trim_start().starts_with("cargo llvm-cov report"))
        .collect::<Vec<_>>();

    if test_commands.len() != 1 || !test_commands[0].contains("--no-report") {
        violations.push(format!(
            "coverage must execute instrumented tests exactly once with --no-report; found {} test commands",
            test_commands.len()
        ));
    }
    if report_commands.len() != 2
        || !report_commands.iter().any(|command| {
            command.contains("--summary-only")
                && command.contains("--json")
                && command.contains("--output-path rust-coverage.json")
        })
        || !report_commands
            .iter()
            .any(|command| command.trim() == "cargo llvm-cov report")
    {
        violations.push(
            "coverage must render JSON and text summaries with two report-only commands".into(),
        );
    }

    violations
}

fn release_rust_cache_violations(workflow: &str) -> Vec<String> {
    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse release.yml");
    let mut violations = Vec::new();
    if parsed["jobs"].get("release").is_some() {
        violations.push("release workflow retains the duplicate tag build matrix".into());
    }
    for forbidden in [
        "cargo build",
        "build-release-binaries",
        "Swatinem/rust-cache",
        "sccache-action",
    ] {
        if workflow.contains(forbidden) {
            violations.push(format!(
                "release workflow can rebuild or cache PR-validated binaries via {forbidden:?}"
            ));
        }
    }
    let promote = job_step(
        &parsed,
        "promote-assets",
        "Download exact validated wrapper once",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !promote.contains("scripts/release-promotion.py download-assets") {
        violations
            .push("release workflow does not consume the receipt-bound archive bundle".into());
    }
    violations
}

fn nextest_whole_core_serialization_violations(config: &str) -> Vec<String> {
    let parsed: toml::Value = toml::from_str(config).expect("parse nextest.toml");
    let mut violations = Vec::new();
    let Some(overrides) = parsed["profile"]["default"]["overrides"].as_array() else {
        return violations;
    };
    for override_ in overrides {
        if override_["filter"].as_str() != Some("package(wenlan-core)") {
            continue;
        }
        let Some(group) = override_["test-group"].as_str() else {
            continue;
        };
        let max_threads = parsed["test-groups"][group]["max-threads"].as_integer();
        if max_threads == Some(1) {
            violations.push(format!(
                "nextest serializes the entire wenlan-core package through group {group:?}"
            ));
        }
    }

    violations
}

fn nextest_drift_priority_violations(config: &str) -> Vec<String> {
    let parsed: toml::Value = toml::from_str(config).expect("parse nextest.toml");
    let Some(overrides) = parsed["profile"]["default"]["overrides"].as_array() else {
        return vec!["nextest has no default-profile overrides".into()];
    };
    if overrides.iter().any(|override_| {
        override_["filter"].as_str() == Some("test(/^drift_guard::/)")
            && override_["priority"].as_integer() == Some(100)
    }) {
        Vec::new()
    } else {
        vec!["nextest does not run drift guards at highest priority".into()]
    }
}

fn text_embedding_initializer_sites(path: &str, source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || !trimmed.contains("TextEmbedding::try_new(") {
                return None;
            }
            Some(format!("{path}:{trimmed}"))
        })
        .collect()
}

#[test]
fn fastembed_ci_artifact_is_prepared_once_before_model_consumers() {
    let workflow =
        std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read ci.yml");
    let violations = fastembed_ci_cache_violations(&workflow);
    assert!(
        violations.is_empty(),
        "FastEmbed CI distribution contract drift — ci.yml must prepare the FastEmbed model \
         cache ONCE and every model-consuming job must download that prepared artifact. Fix \
         the steps named below in .github/workflows/ci.yml; an intentional contract change \
         also updates fastembed_ci_cache_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn coverage_and_ci_share_the_fastembed_cache_contract() {
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/coverage.yml"))
        .expect("read coverage.yml");
    let violations = coverage_fastembed_cache_violations(&workflow);
    assert!(
        violations.is_empty(),
        "Coverage FastEmbed cache contract drift — coverage.yml must share ci.yml's prepared \
         FastEmbed cache contract instead of downloading models independently. Fix \
         .github/workflows/coverage.yml; an intentional contract change also updates \
         coverage_fastembed_cache_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn coverage_fastembed_cache_contract_detects_non_sharing_path() {
    let workflow = r#"
jobs:
  coverage:
    env: {}
    steps:
      - name: Restore portable FastEmbed model
        uses: actions/cache@v4
        with:
          path: ~/.local/share/wenlan/memorydb/fastembed_cache
          key: fastembed-bge-base-en-v1.5-q-v1
"#;
    let violations = coverage_fastembed_cache_violations(workflow);
    for expected in [
        "FASTEMBED_CACHE_DIR",
        "coverage caches",
        "cache key",
        "cross-OS portable FastEmbed v3",
        "FastEmbed cache writer",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn coverage_executes_instrumented_tests_once_then_reports_twice() {
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/coverage.yml"))
        .expect("read coverage.yml");
    let violations = coverage_single_test_execution_violations(&workflow);
    assert!(
        violations.is_empty(),
        "Coverage execution contract drift — coverage.yml must run the instrumented test suite \
         ONCE and derive both reports from that single run (a second pass doubles CI time). \
         Fix .github/workflows/coverage.yml; an intentional contract change also updates \
         coverage_single_test_execution_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn coverage_execution_contract_rejects_two_test_runs_fixture() {
    let workflow = r#"
jobs:
  coverage:
    steps:
      - name: Run Rust coverage (wenlan-core + wenlan-server)
        run: |
          cargo llvm-cov --package wenlan-core --summary-only --json
          cargo llvm-cov --package wenlan-core --summary-only
"#;
    let violations = coverage_single_test_execution_violations(workflow);
    for expected in [
        "cache writes",
        "rust-cache footprint",
        "exactly once",
        "report-only commands",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn release_reuses_receipt_bound_archives_without_compiling() {
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
        .expect("read release.yml");
    let violations = release_rust_cache_violations(&workflow);
    assert!(
        violations.is_empty(),
        "Release artifact-reuse contract drift — release.yml must consume receipt-bound \
         archives without compiling or restoring compiler caches. Fix \
         .github/workflows/release.yml; an intentional contract change also updates \
         release_rust_cache_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_artifact_reuse_contract_rejects_build_fixture() {
    let workflow = r#"
jobs:
  release:
    steps:
      - name: Set up sccache
        uses: mozilla-actions/sccache-action@sha
      - name: Build and smoke shipped release binaries
        env:
          SCCACHE_GHA_ENABLED: "true"
          RUSTC_WRAPPER: sccache
        run: cargo build --release
"#;
    let violations = release_rust_cache_violations(workflow);
    for expected in [
        "duplicate tag build matrix",
        "rebuild or cache",
        "receipt-bound",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

fn release_please_trigger_violations(workflow: &str) -> Vec<String> {
    let parsed: serde_yaml::Value =
        serde_yaml::from_str(workflow).unwrap_or(serde_yaml::Value::Null);
    let mut violations = Vec::new();
    let trigger_names = parsed["on"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let workflows = parsed["on"]["workflow_run"]["workflows"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let types = parsed["on"]["workflow_run"]["types"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let branches = parsed["on"]["workflow_run"]["branches"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    if trigger_names != BTreeSet::from(["workflow_run"])
        || workflows != ["CI"]
        || types != ["completed"]
        || branches != ["main"]
    {
        violations.push("release-please trigger is not exact completed main CI".into());
    }
    let jobs = parsed["jobs"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    if jobs != BTreeSet::from(["route-main", "maintain-release-pr", "create-validated-tag"]) {
        violations.push("release-please hybrid route job inventory drifted".into());
    }
    let route = &parsed["jobs"]["route-main"];
    let route_condition = route["if"].as_str().unwrap_or_default();
    for required in [
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.conclusion == 'success'",
    ] {
        if !route_condition.contains(required) {
            violations.push(format!(
                "release-please route omits main CI identity {required:?}"
            ));
        }
    }
    let revalidate = job_step(
        &parsed,
        "route-main",
        "Consume the exact main CI promotion receipt",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !revalidate.contains("scripts/release-promotion.py consume-main-receipt") {
        violations.push("release-please route does not consume the main promotion receipt".into());
    }
    let maintain = &parsed["jobs"]["maintain-release-pr"];
    if maintain["if"].as_str() != Some("needs.route-main.outputs.state == 'ordinary'")
        || job_step_using(
            &parsed,
            "maintain-release-pr",
            "googleapis/release-please-action",
        )
        .and_then(|step| step["with"]["skip-github-release"].as_bool())
            != Some(true)
    {
        violations.push("ordinary main is not restricted to PR-only release-please".into());
    }
    let create = &parsed["jobs"]["create-validated-tag"];
    let create_run = job_step(
        &parsed,
        "create-validated-tag",
        "Require pending release lifecycle and create exact tag",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if create["if"].as_str() != Some("needs.route-main.outputs.state == 'validated'")
        || !create_run.contains("refs/tags/$RELEASE_TAG")
        || !create_run.contains("sha=\"$MAIN_SHA\"")
    {
        violations
            .push("validated release route does not create only the exact receipt tag".into());
    }
    violations
}

#[test]
fn release_please_runs_only_after_successful_main_push_ci() {
    let workflow =
        std::fs::read_to_string(repo_root().join(".github/workflows/release-please.yml"))
            .expect("read release-please.yml");
    let violations = release_please_trigger_violations(&workflow);
    assert!(
        violations.is_empty(),
        "release-please trigger drift — automatic release bookkeeping must run only after the \
         successful CI workflow for a main push. Fix .github/workflows/release-please.yml; an \
         intentional contract change also updates release_please_trigger_violations() and its \
         positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_please_trigger_rejects_direct_or_failed_pushes() {
    let workflow = r#"
on:
  push:
    branches: [main]
jobs:
  release-please:
    if: github.ref == 'refs/heads/main'
"#;
    let violations = release_please_trigger_violations(workflow);
    for expected in [
        "exact completed main CI",
        "hybrid route job",
        "main promotion receipt",
    ] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn nextest_does_not_serialize_the_entire_core_package() {
    let config = std::fs::read_to_string(repo_root().join(".config/nextest.toml"))
        .expect("read nextest.toml");
    let violations = nextest_whole_core_serialization_violations(&config);
    assert!(
        violations.is_empty(),
        "nextest whole-package serialization contract drift — .config/nextest.toml may \
         serialize named test groups but never the entire wenlan-core package. Fix \
         .config/nextest.toml; an intentional contract change also updates \
         nextest_whole_core_serialization_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn nextest_runs_drift_guards_before_database_heavy_tests() {
    let config = std::fs::read_to_string(repo_root().join(".config/nextest.toml"))
        .expect("read nextest.toml");
    let violations = nextest_drift_priority_violations(&config);
    assert!(
        violations.is_empty(),
        "nextest drift-guard priority contract failed:\n{}",
        violations.join("\n")
    );
}

#[test]
fn nextest_drift_priority_contract_rejects_default_order() {
    let violations = nextest_drift_priority_violations(
        r#"
[[profile.default.overrides]]
filter = 'test(/^drift_guard::/)'
priority = 0
"#,
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("highest priority")),
        "fixture must reject default drift-guard order: {violations:?}"
    );
}

#[test]
fn nextest_parallelism_contract_rejects_whole_core_serialization() {
    let config = r#"
[test-groups]
wenlan-core = { max-threads = 1 }

[[profile.default.overrides]]
filter = 'package(wenlan-core)'
test-group = 'wenlan-core'
"#;
    let violations = nextest_whole_core_serialization_violations(config);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("entire wenlan-core package")),
        "fixture must exercise whole-package serialization: {violations:?}"
    );
}

#[test]
fn all_text_embedding_initializers_use_the_cross_process_lock() {
    let root = repo_root();
    let mut sites = Vec::new();
    for path in git_ls_files(&root, "*.rs").into_iter().filter(|path| {
        path.starts_with("crates/wenlan-core/src/")
            && path != "crates/wenlan-core/src/drift_guard.rs"
    }) {
        let source = std::fs::read_to_string(root.join(&path)).expect("read Rust source");
        sites.extend(text_embedding_initializer_sites(&path, &source));
    }
    assert_eq!(
        sites,
        ["crates/wenlan-core/src/db.rs:TextEmbedding::try_new(options)"],
        "FastEmbed text initialization bypasses db::init_text_embedding: {sites:?}"
    );
}

#[test]
fn text_embedding_initializer_guard_detects_a_direct_call() {
    let sites = text_embedding_initializer_sites(
        "crates/wenlan-core/src/new_model.rs",
        "let model = fastembed::TextEmbedding::try_new(options)?;",
    );
    assert_eq!(sites.len(), 1, "positive control must detect direct init");
}

fn sentence_delimiter_sites(path: &str, source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || !trimmed.contains(r"[.!?]+\s+") {
                return None;
            }
            Some(format!("{path}:{trimmed}"))
        })
        .collect()
}

/// The sentence-boundary rule decides where one claim ends and the next
/// begins, so a second copy silently forks the claim corpus: two call sites
/// that agree today drift apart on the next abbreviation fix, and "sentence
/// n" stops meaning the same prose in the two paths. M5 claim identity is
/// content-addressed over that text, so the fork is not cosmetic — it mints
/// divergent revisions for the same page. See
/// `docs/plans/2026-07-27-m5-claim-extractor-spec.md` §1.
#[test]
fn the_sentence_delimiter_rule_has_exactly_one_definition() {
    let root = repo_root();
    let mut sites = Vec::new();
    for path in git_ls_files(&root, "*.rs").into_iter().filter(|path| {
        path.starts_with("crates/")
            && path.contains("/src/")
            && path != "crates/wenlan-core/src/drift_guard.rs"
    }) {
        let source = std::fs::read_to_string(root.join(&path)).expect("read Rust source");
        sites.extend(sentence_delimiter_sites(&path, &source));
    }
    assert_eq!(
        sites,
        [concat!(
            "crates/wenlan-core/src/faithfulness.rs:",
            r#"let re = regex::Regex::new(r"(?m)[.!?]+\s+").expect("static regex");"#
        )],
        "the sentence-boundary rule must live in exactly one place \
         (faithfulness::sentence_spans); callers needing offsets call it \
         rather than re-deriving the split: {sites:?}"
    );
}

#[test]
fn sentence_delimiter_guard_detects_a_second_copy() {
    let sites = sentence_delimiter_sites(
        "crates/wenlan-core/src/somewhere_else.rs",
        "    let delim = regex::Regex::new(r\"(?m)[.!?]+\\s+\").unwrap();",
    );
    assert_eq!(sites.len(), 1, "positive control must detect a second copy");
}

#[test]
fn clippy_syntax_guard_forbids_direct_text_embedding_initializers() {
    let config =
        std::fs::read_to_string(repo_root().join("clippy.toml")).expect("read clippy.toml");
    let parsed: toml::Value = toml::from_str(&config).expect("parse clippy.toml");
    let guarded = parsed["disallowed-methods"]
        .as_array()
        .is_some_and(|methods| {
            methods.iter().any(|method| {
                method["path"].as_str() == Some("fastembed::TextEmbedding::try_new")
                    && method["replacement"].as_str() == Some("crate::db::init_text_embedding")
            })
        });
    assert!(
        guarded,
        "clippy.toml must syntax-check every TextEmbedding initializer"
    );
}

#[test]
fn fastembed_ci_cache_contract_detects_wrong_path_and_order() {
    let workflow = r#"
jobs:
  test:
    env:
      FASTEMBED_CACHE_DIR: /tmp/wrong-fastembed-cache
    steps:
      - name: Workspace lib tests (macOS)
        run: export WENLAN_TEST_FASTEMBED_CACHE=/tmp/stale-cache
      - name: Download portable FastEmbed model
        uses: actions/download-artifact@bad
        with:
          path: ~/.local/share/wenlan/memorydb/fastembed_cache
          name: stale
  canonical-acceptance:
    env:
      FASTEMBED_CACHE_DIR: ${{ github.workspace }}/.fastembed_cache
    steps:
      - name: Download portable FastEmbed model
        uses: actions/download-artifact@bad
        with:
          path: .fastembed_cache
          name: stale
      - name: Integration tests wenlan-cli + wenlan-server (Linux)
  contract-integration:
    env:
      FASTEMBED_CACHE_DIR: ${{ github.workspace }}/.fastembed_cache
    steps:
      - name: Download portable FastEmbed model
        uses: actions/download-artifact@bad
        with:
          path: ${{ env.FASTEMBED_CACHE_DIR }}
          name: stale
      - name: Affected wenlan-mcp + wenlan-types integrations
"#;
    let violations = fastembed_ci_cache_violations(workflow);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("FASTEMBED_CACHE_DIR")),
        "fixture must violate the explicit cache directory: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("after consumer")),
        "fixture must violate restore ordering: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("WENLAN_TEST_FASTEMBED_CACHE")),
        "fixture must reject per-step cache overrides: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("artifact name")),
        "fixture must reject a missing or stale artifact name: {violations:?}"
    );
}

// ── Teeth #7: Windows ONNX Runtime release contract ──

// Compatibility pair grounded in ort commit 2de34065983a5c034f5afcc072b23b99479f465b:
// ort-sys/build/download/dist.txt pins the Windows x64 CPU build to ms@1.23.2,
// and ort-sys/src/version.rs exposes ORT_API_VERSION = 23.
const ORT_CRATE_VERSION: &str = "2.0.0-rc.11";
const WINDOWS_ORT_VERSION: &str = "1.23.2";
const WINDOWS_ORT_ZIP_SHA256: &str =
    "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd";

fn dependency_features<'a>(
    manifest: &'a toml::Value,
    path: &[&str],
    dependency: &str,
) -> Option<Vec<&'a str>> {
    let mut table = manifest;
    for key in path {
        table = table.get(*key)?;
    }
    table
        .get(dependency)?
        .get("features")?
        .as_array()
        .map(|features| features.iter().filter_map(toml::Value::as_str).collect())
}

fn windows_ort_contract_violations(
    workspace_manifest: &str,
    core_manifest: &str,
    cargo_lock: &str,
    stage_script: &str,
) -> Vec<String> {
    let workspace: toml::Value =
        toml::from_str(workspace_manifest).expect("parse workspace Cargo.toml");
    let core: toml::Value = toml::from_str(core_manifest).expect("parse wenlan-core Cargo.toml");
    let lock: toml::Value = toml::from_str(cargo_lock).expect("parse Cargo.lock");
    let mut violations = Vec::new();

    let base_features =
        dependency_features(&workspace, &["workspace", "dependencies"], "fastembed")
            .unwrap_or_default();
    if base_features
        .iter()
        .any(|feature| feature.starts_with("ort-"))
    {
        violations.push(
            "workspace FastEmbed features select an ORT linkage mode for every target".to_string(),
        );
    }

    if core["dependencies"].get("fastembed").is_some() {
        violations.push("wenlan-core declares FastEmbed outside target-specific sections".into());
    }

    let windows_features = dependency_features(
        &core,
        &["target", "cfg(windows)", "dependencies"],
        "fastembed",
    )
    .unwrap_or_default();
    if !windows_features.contains(&"ort-load-dynamic")
        || windows_features
            .iter()
            .any(|feature| feature.starts_with("ort-download-binaries"))
    {
        violations.push(
            "Windows FastEmbed must use ort-load-dynamic without downloaded static binaries".into(),
        );
    }

    let non_windows_features = dependency_features(
        &core,
        &["target", "cfg(not(windows))", "dependencies"],
        "fastembed",
    )
    .unwrap_or_default();
    if !non_windows_features.contains(&"ort-download-binaries-native-tls")
        || non_windows_features.contains(&"ort-load-dynamic")
    {
        violations.push(
            "non-Windows FastEmbed must retain downloaded static ORT without dynamic loading"
                .into(),
        );
    }

    let ort_versions: Vec<&str> = lock["package"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|package| matches!(package["name"].as_str(), Some("ort" | "ort-sys")))
        .filter_map(|package| package["version"].as_str())
        .collect();
    if ort_versions != [ORT_CRATE_VERSION, ORT_CRATE_VERSION] {
        violations.push(format!(
            "Cargo.lock must pin ort and ort-sys to verified version {ORT_CRATE_VERSION}, got {ort_versions:?}"
        ));
    }

    if !stage_script.contains(&format!("$OrtVersion = \"{WINDOWS_ORT_VERSION}\"")) {
        violations.push(format!(
            "Windows ORT stager must use version {WINDOWS_ORT_VERSION}"
        ));
    }
    if !stage_script.contains(&format!(
        "$ExpectedZipSha256 = \"{WINDOWS_ORT_ZIP_SHA256}\""
    )) || !stage_script.contains("Get-FileHash")
        || !stage_script.contains("$ActualZipSha256 -ne $ExpectedZipSha256")
    {
        violations.push("Windows ORT archive must be verified against its pinned SHA-256".into());
    }

    violations
}

#[test]
fn windows_ort_release_contract_is_dynamic_and_version_matched() {
    let root = repo_root();
    let workspace =
        std::fs::read_to_string(root.join("Cargo.toml")).expect("read workspace Cargo.toml");
    let core = std::fs::read_to_string(root.join("crates/wenlan-core/Cargo.toml"))
        .expect("read wenlan-core Cargo.toml");
    let lock = std::fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let stage_script = std::fs::read_to_string(root.join("scripts/stage-onnxruntime-windows.ps1"))
        .unwrap_or_default();
    let violations = windows_ort_contract_violations(&workspace, &core, &lock, &stage_script);
    assert!(
        violations.is_empty(),
        "Windows ONNX Runtime release contract drift — the Windows ORT dependency stays \
         dynamic-loading and version-matched to the pinned ort compatibility pair (teeth #7 \
         header comment). Fix the manifests/workflow named below; an intentional contract \
         change also updates windows_ort_contract_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn windows_ort_release_contract_rejects_static_unverified_mismatch() {
    let workspace = r#"
[workspace.dependencies]
fastembed = { version = "5", features = ["ort-download-binaries-native-tls"] }
"#;
    let core = r#"
[dependencies]
fastembed = { workspace = true }
"#;
    let lock = r#"
[[package]]
name = "ort"
version = "2.0.0-rc.10"

[[package]]
name = "ort-sys"
version = "2.0.0-rc.10"
"#;
    let stage_script = r#"
$OrtVersion = "1.20.0"
Invoke-WebRequest -Uri "https://example.invalid/onnxruntime.zip"
"#;
    let violations = windows_ort_contract_violations(workspace, core, lock, stage_script);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("every target")),
        "fixture must reject target-independent static ORT: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("ort-load-dynamic")),
        "fixture must require dynamic ORT on Windows: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains(ORT_CRATE_VERSION)),
        "fixture must reject an unverified ort crate version: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains(WINDOWS_ORT_VERSION)),
        "fixture must reject a mismatched ORT DLL version: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("SHA-256")),
        "fixture must reject an unverified ORT archive: {violations:?}"
    );
}

fn workflow_step_run<'a>(workflow: &'a serde_yaml::Value, step_name: &str) -> Option<&'a str> {
    workflow["jobs"]
        .as_mapping()?
        .values()
        .filter_map(|job| job["steps"].as_sequence())
        .flat_map(|steps| steps.iter())
        .find(|step| step["name"].as_str() == Some(step_name))
        .and_then(|step| step["run"].as_str())
}

fn windows_ort_distribution_violations(
    ci_workflow: &str,
    release_workflow: &str,
    smoke_script: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let release: serde_yaml::Value =
        serde_yaml::from_str(release_workflow).expect("parse release.yml");
    let mut violations = Vec::new();
    let stage =
        workflow_step_run(&ci, "Stage Windows release runtimes before smoke").unwrap_or_default();
    let build =
        workflow_step_run(&ci, "Build and smoke shipped release binaries").unwrap_or_default();
    let smoke =
        workflow_step_run(&ci, "Native ORT smoke (Windows release preflight)").unwrap_or_default();
    if !stage.contains("scripts/stage-onnxruntime-windows.ps1")
        || !stage.contains("scripts/stage-vulkan-loader-windows.ps1")
        || !build.contains("scripts/build-release-binaries.sh")
        || !smoke.contains("scripts/smoke-windows.ps1")
    {
        violations
            .push("Release PR CI does not stage, package, and smoke exact Windows runtimes".into());
    }
    if release["jobs"].get("release").is_some()
        || release_workflow.contains("cargo build")
        || release_workflow.contains("build-release-binaries")
    {
        violations.push(
            "tag release rebuilds instead of promoting the smoke-tested Windows archive".into(),
        );
    }
    let promote = job_step(
        &release,
        "promote-assets",
        "Download exact validated wrapper once",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !promote.contains("scripts/release-promotion.py download-assets") {
        violations
            .push("tag release does not consume the observer-validated Windows archive".into());
    }
    if !smoke_script.contains("Get-Process -Id $proc.Id -Module")
        || !smoke_script.contains("onnxruntime.dll")
        || !smoke_script.contains("vulkan-1.dll")
        || !smoke_script.contains("/api/memory/store")
        || !smoke_script.contains("chunks_created")
        || smoke_script.contains("$env:ORT_DYLIB_PATH")
    {
        violations.push(
            "Windows smoke does not exercise vector inference through the exact packaged DLLs"
                .into(),
        );
    }
    violations
}

#[test]
fn windows_ort_distribution_stages_packages_and_exercises_exact_dll() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release.yml");
    let smoke = std::fs::read_to_string(root.join("scripts/smoke-windows.ps1"))
        .expect("read smoke-windows.ps1");
    let violations = windows_ort_distribution_violations(&ci, &release, &smoke);
    assert!(
        violations.is_empty(),
        "Windows ORT distribution proof drift — the Windows release must stage, package, and \
         smoke-test the EXACT onnxruntime.dll it ships. Fix .github/workflows/release.yml; an \
         intentional contract change also updates windows_ort_distribution_violations() and \
         its positive controls:\n{}",
        violations.join("\n")
    );
}

#[test]
fn windows_ort_distribution_contract_rejects_unexercised_archive() {
    let workflow = r#"
jobs:
  test:
    steps:
      - name: Package
        run: 7z a dist/wenlan.zip wenlan-server.exe
"#;
    let violations = windows_ort_distribution_violations(workflow, workflow, "health only");
    for expected in [
        "stage, package, and smoke",
        "observer-validated",
        "vector inference",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn windows_ort_distribution_contract_rejects_unscoped_archive_payloads() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release.yml")
        .replace(
            "scripts/release-promotion.py download-assets",
            "scripts/build-release-binaries.sh",
        );
    let smoke = std::fs::read_to_string(root.join("scripts/smoke-windows.ps1"))
        .expect("read smoke-windows.ps1");
    let violations = windows_ort_distribution_violations(&ci, &release, &smoke);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("tag release rebuilds")),
        "fixture must reject tag-side Windows rebuilding: {violations:?}"
    );
}

#[test]
fn windows_ort_distribution_contract_rejects_late_or_wrong_os_test_bootstrap() {
    let workflow = r#"
jobs:
  test:
    steps:
      - name: Integration tests wenlan-cli + wenlan-server
        run: cargo nextest run
      - name: Stage ONNX Runtime for Windows tests
        if: matrix.os == 'macos-14'
        run: |
          scripts/stage-onnxruntime-windows.ps1
          "ORT_DYLIB_PATH=x" | Out-File $env:GITHUB_ENV
"#;
    let violations = windows_ort_distribution_violations(workflow, workflow, "health only");
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("stage, package, and smoke")),
        "fixture must reject an incomplete Windows candidate proof: {violations:?}"
    );
}

// ── Teeth #8: differential CI routing stays fail-closed ──

fn detect_change_filter_paths(workflow: &serde_yaml::Value, filter_name: &str) -> BTreeSet<String> {
    let Some(steps) = workflow["jobs"]["detect-changes"]["steps"].as_sequence() else {
        return BTreeSet::new();
    };
    let Some(filters_yaml) = steps
        .iter()
        .find(|step| step["id"].as_str() == Some("filter"))
        .and_then(|step| step["with"]["filters"].as_str())
    else {
        return BTreeSet::new();
    };
    let Ok(filters) = serde_yaml::from_str::<serde_yaml::Value>(filters_yaml) else {
        return BTreeSet::new();
    };
    filters[filter_name]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .map(str::to_string)
        .collect()
}

fn job_needs(workflow: &serde_yaml::Value, job_name: &str) -> Vec<String> {
    let needs = &workflow["jobs"][job_name]["needs"];
    if let Some(single) = needs.as_str() {
        return vec![single.to_string()];
    }
    needs
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .map(str::to_string)
        .collect()
}

fn required_job_closure(workflow: &serde_yaml::Value) -> BTreeSet<String> {
    let mut required = BTreeSet::new();
    let mut pending = vec!["conclusion".to_string()];
    while let Some(job_name) = pending.pop() {
        if required.insert(job_name.clone()) {
            pending.extend(job_needs(workflow, &job_name));
        }
    }
    required
}

fn required_jobs_contain(workflow: &serde_yaml::Value, needle: &str) -> bool {
    required_job_closure(workflow).iter().any(|job_name| {
        workflow["jobs"][job_name]["steps"]
            .as_sequence()
            .into_iter()
            .flatten()
            .any(|step| {
                step["run"].as_str().is_some_and(|run| run.contains(needle))
                    || step["uses"]
                        .as_str()
                        .is_some_and(|uses| uses.contains(needle))
            })
    })
}

fn job_step<'a>(
    workflow: &'a serde_yaml::Value,
    job_name: &str,
    step_name: &str,
) -> Option<&'a serde_yaml::Value> {
    workflow["jobs"][job_name]["steps"]
        .as_sequence()?
        .iter()
        .find(|step| step["name"].as_str() == Some(step_name))
}

fn job_step_using<'a>(
    workflow: &'a serde_yaml::Value,
    job_name: &str,
    action: &str,
) -> Option<&'a serde_yaml::Value> {
    workflow["jobs"][job_name]["steps"]
        .as_sequence()?
        .iter()
        .find(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains(action))
        })
}

fn native_platform_markers() -> Vec<(&'static str, &'static str, regex::Regex)> {
    vec![
        (
            "windows",
            "windows",
            regex::Regex::new(r#"\b(?:_WIN32|_WIN64|WIN32)\b"#).unwrap(),
        ),
        (
            "macos",
            "macos",
            regex::Regex::new(r#"\b(?:__APPLE__|__MACH__|TARGET_OS_OSX)\b"#).unwrap(),
        ),
        (
            "linux",
            "rust",
            regex::Regex::new(r#"\b__linux__\b"#).unwrap(),
        ),
        (
            "unix",
            "macos",
            regex::Regex::new(r#"\b(?:__unix__|__unix)\b"#).unwrap(),
        ),
        (
            "unix",
            "windows",
            regex::Regex::new(r#"\b(?:__unix__|__unix)\b"#).unwrap(),
        ),
    ]
}

fn rust_cfg_expression_ranges(contents: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut search_from = 0;
    while let Some(relative) = contents[search_from..].find("cfg") {
        let start = search_from + relative;
        let previous_is_identifier = contents[..start]
            .bytes()
            .next_back()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
        if previous_is_identifier {
            search_from = start + 3;
            continue;
        }

        let mut cursor = start + 3;
        if contents[cursor..].starts_with("_attr") {
            cursor += "_attr".len();
        } else if contents[cursor..].starts_with('!') {
            cursor += 1;
        }
        while contents
            .as_bytes()
            .get(cursor)
            .is_some_and(u8::is_ascii_whitespace)
        {
            cursor += 1;
        }
        if contents.as_bytes().get(cursor) != Some(&b'(') {
            search_from = start + 3;
            continue;
        }

        let expression_start = cursor + 1;
        let mut depth = 0_u32;
        let mut in_string = false;
        let mut escaped = false;
        let mut expression_end = None;
        for (relative, character) in contents[cursor..].char_indices() {
            let absolute = cursor + relative;
            if in_string {
                if escaped {
                    escaped = false;
                } else if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    in_string = false;
                }
                continue;
            }
            if character == '"' {
                in_string = true;
                continue;
            }
            match character {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        expression_end = Some(absolute);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = expression_end {
            ranges.push((expression_start, end));
            search_from = end + 1;
        } else {
            search_from = start + 3;
        }
    }
    ranges
}

fn cfg_expression_has_word(expression: &str, expected: &str) -> bool {
    expression
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|word| word == expected)
}

fn source_platform_routes(
    path: &str,
    contents: &str,
    markers: &[(&'static str, &'static str, regex::Regex)],
) -> BTreeSet<(&'static str, &'static str)> {
    let mut routes = BTreeSet::new();
    if [".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"]
        .iter()
        .any(|extension| path.ends_with(extension))
    {
        routes.insert(("native", "macos"));
        routes.insert(("native", "windows"));
    }
    if path.ends_with(".m") || path.ends_with(".mm") {
        routes.insert(("macos", "macos"));
    }
    for (start, end) in rust_cfg_expression_ranges(contents) {
        let expression = &contents[start..end];
        if cfg_expression_has_word(expression, "windows") {
            routes.insert(("windows", "windows"));
        }
        if cfg_expression_has_word(expression, "macos") {
            routes.insert(("macos", "macos"));
        }
        if cfg_expression_has_word(expression, "linux") {
            routes.insert(("linux", "rust"));
        }
        if cfg_expression_has_word(expression, "unix") {
            routes.insert(("unix", "macos"));
            routes.insert(("unix", "windows"));
        }
    }
    for (platform, filter, path_marker) in [
        ("windows", "windows", "std::os::windows"),
        ("macos", "macos", "std::os::macos"),
        ("linux", "rust", "std::os::linux"),
        ("unix", "macos", "std::os::unix"),
        ("unix", "windows", "std::os::unix"),
    ] {
        if contents.contains(path_marker) {
            routes.insert((platform, filter));
        }
    }
    for (platform, filter, marker) in markers {
        if marker.is_match(contents) {
            routes.insert((*platform, *filter));
        }
    }
    routes
}

#[test]
fn platform_source_markers_cover_native_positive_controls() {
    let markers = native_platform_markers();
    let nested_rust_cfg = source_platform_routes(
        "crates/platform.rs",
        r#"
#[cfg(all(not(feature = "portable"), windows))]
fn windows_only() {}
#[cfg(all(not(feature = "portable"), macos))]
fn macos_only() {}
#[cfg(all(not(feature = "portable"), linux))]
fn linux_only() {}
#[cfg(all(not(feature = "portable"), unix))]
fn unix_only() {}
"#,
        &markers,
    );
    assert!(nested_rust_cfg.contains(&("windows", "windows")));
    assert!(nested_rust_cfg.contains(&("macos", "macos")));
    assert!(nested_rust_cfg.contains(&("linux", "rust")));
    assert!(nested_rust_cfg.contains(&("unix", "macos")));
    assert!(nested_rust_cfg.contains(&("unix", "windows")));

    let windows = source_platform_routes("crates/native.c", "#if defined(_WIN32)", &markers);
    assert!(windows.contains(&("windows", "windows")));

    let shared_native = source_platform_routes(
        "crates/native.cpp",
        "void platform_neutral(void);",
        &markers,
    );
    assert!(shared_native.contains(&("native", "macos")));
    assert!(shared_native.contains(&("native", "windows")));

    let alternate_cpp = source_platform_routes(
        "crates/native.cxx",
        "void platform_neutral(void);",
        &markers,
    );
    assert!(alternate_cpp.contains(&("native", "macos")));
    assert!(alternate_cpp.contains(&("native", "windows")));

    let apple = source_platform_routes("crates/native.h", "#ifdef __APPLE__", &markers);
    assert!(apple.contains(&("macos", "macos")));

    let linux = source_platform_routes("crates/native.cc", "#ifdef __linux__", &markers);
    assert!(linux.contains(&("linux", "rust")));

    let unix = source_platform_routes("crates/native.cpp", "#ifdef __unix__", &markers);
    assert!(unix.contains(&("unix", "macos")));
    assert!(unix.contains(&("unix", "windows")));

    let objective_c =
        source_platform_routes("crates/native.m", "void platform_neutral(void);", &markers);
    assert!(objective_c.contains(&("macos", "macos")));
}

fn platform_sensitive_paths(root: &Path) -> Vec<(String, &'static str, &'static str)> {
    let markers = native_platform_markers();
    let mut paths = BTreeSet::new();
    let mut native_sources = BTreeSet::new();
    for pattern in [
        "*.rs", "*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx", "*.m", "*.mm",
    ] {
        native_sources.extend(git_ls_files(root, pattern));
    }
    for path in native_sources
        .into_iter()
        .filter(|path| path.starts_with("crates/"))
    {
        let contents = std::fs::read_to_string(root.join(&path)).unwrap_or_default();
        for (platform, filter) in source_platform_routes(&path, &contents, &markers) {
            paths.insert((path.clone(), platform, filter));
        }
    }

    for path in git_ls_files(root, "scripts/*") {
        let lower = path.to_ascii_lowercase();
        if lower.contains("windows") || lower.ends_with(".ps1") {
            paths.insert((path.clone(), "windows", "windows"));
        }
        if lower.contains("macos") {
            paths.insert((path.clone(), "macos", "macos"));
        }
        if lower.contains("linux") {
            paths.insert((path, "linux", "rust"));
        }
    }
    for path in git_ls_files(root, "docker/*") {
        paths.insert((path, "linux", "rust"));
    }

    paths.into_iter().collect()
}

fn release_profile_sensitive_paths(root: &Path) -> Vec<String> {
    let core_lib =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/lib.rs")).expect("read core lib");
    assert!(
        core_lib.contains("#[cfg(test)]\nmod drift_guard;"),
        "drift_guard.rs exclusion is valid only while the whole module is test-only"
    );
    git_ls_files(root, "*.rs")
        .into_iter()
        .filter(|path| {
            path.starts_with("crates/")
                && path.contains("/src/")
                && path != "crates/wenlan-core/src/drift_guard.rs"
        })
        .filter(|path| {
            std::fs::read_to_string(root.join(path))
                .is_ok_and(|contents| has_production_release_marker(&contents))
        })
        .collect()
}

fn has_production_release_marker(contents: &str) -> bool {
    rust_cfg_expression_ranges(contents)
        .into_iter()
        .filter(|(start, end)| cfg_expression_has_word(&contents[*start..*end], "debug_assertions"))
        .any(|(start, end)| {
            let line_start = contents[..start]
                .rfind('\n')
                .map_or(0, |newline| newline + 1);
            let line_end = contents[end..]
                .find('\n')
                .map_or(contents.len(), |newline| end + newline);
            let adjacent_attributes = contents[..line_start]
                .lines()
                .rev()
                .take_while(|line| line.trim_start().starts_with("#["))
                .chain(
                    contents[line_end..]
                        .lines()
                        .skip(1)
                        .take_while(|line| line.trim_start().starts_with("#[")),
                );
            !adjacent_attributes.into_iter().any(|line| {
                let attribute = line.trim();
                attribute == "#[test]" || attribute.contains("::test")
            })
        })
}

#[test]
fn release_profile_marker_scan_is_fail_closed_after_test_modules() {
    let test_only = r#"
#[test]
#[cfg(debug_assertions)]
fn debug_only_assertion() {}
"#;
    assert!(!has_production_release_marker(test_only));

    let nonterminal_test_module = r#"
#[cfg(test)]
mod tests {
    #[test]
    #[cfg(debug_assertions)]
    fn debug_only_assertion() {}
}

#[cfg(not(debug_assertions))]
pub fn release_only_runtime() {}
"#;
    assert!(has_production_release_marker(nonterminal_test_module));

    let nested_release_predicate = r#"
#[cfg(all(not(feature = "portable"), debug_assertions))]
pub fn nested_release_sensitive_runtime() {}
"#;
    assert!(has_production_release_marker(nested_release_predicate));
}

fn filter_routes_path(patterns: &BTreeSet<String>, path: &str) -> bool {
    patterns.contains(path)
        || patterns.iter().any(|pattern| {
            pattern
                .strip_prefix("*.")
                .filter(|_| !path.contains('/'))
                .is_some_and(|extension| path.ends_with(&format!(".{extension}")))
        })
        || patterns.iter().any(|pattern| {
            pattern
                .strip_prefix("**/*.")
                .is_some_and(|extension| path.ends_with(&format!(".{extension}")))
        })
        || patterns.iter().any(|pattern| {
            pattern
                .strip_suffix("/**")
                .is_some_and(|prefix| path.starts_with(&format!("{prefix}/")))
        })
        || patterns.iter().any(|pattern| {
            pattern
                .split_once("/**/")
                .and_then(|(prefix, suffix)| {
                    suffix
                        .strip_suffix("/**")
                        .map(|directory| (prefix, directory))
                })
                .is_some_and(|(prefix, directory)| {
                    path.starts_with(&format!("{prefix}/"))
                        && path.contains(&format!("/{directory}/"))
                })
        })
        || patterns.iter().any(|pattern| {
            pattern
                .strip_prefix("crates/**/*.")
                .is_some_and(|extension| {
                    path.starts_with("crates/") && path.ends_with(&format!(".{extension}"))
                })
        })
}

fn ci_routing_contract_violations(
    workflow: &str,
    planner_source: &str,
    platform_sensitive_paths: &[(String, &'static str, &'static str)],
    release_profile_sensitive_paths: &[String],
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse ci.yml");
    let mut violations = Vec::new();

    for output in [
        "plugin-rust-contract",
        "plugin-version-contract",
        "macos",
        "windows",
        "windows-lint",
        "macos-m4",
        "windows-llm-probe",
        "release-preflight",
        "mcp-platform",
        "workspace-platform",
        "test-plan",
        "workspace-lib-required",
        "cli-server-integration-required",
        "core-integration-required",
        "contract-integration-required",
        "canonical-smokes-required",
        "canonical-acceptance-required",
        "rust-ci-required",
        "platform-test-plan",
        "platform-workspace-lib-required",
        "platform-cli-server-integration-required",
    ] {
        if ci["jobs"]["detect-changes"]["outputs"][output]
            .as_str()
            .is_none()
        {
            violations.push(format!("detect-changes does not expose {output} routing"));
        }
    }

    let filter_step = ci["jobs"]["detect-changes"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .find(|step| step["id"].as_str() == Some("filter"));
    if filter_step.and_then(|step| step["with"]["list-files"].as_str()) != Some("json") {
        violations.push("detect-changes does not expose the changed-file inventory as JSON".into());
    }
    let impact_paths = detect_change_filter_paths(&ci, "impact");
    if !impact_paths.contains("**") {
        violations.push("impact routing is not a fail-closed repository catch-all".into());
    }
    let universal_planner_condition =
        "steps.release-proof.outputs.verified-release-merge != 'true'";
    let planner_setup = job_step(
        &ci,
        "detect-changes",
        "Set up Rust for affected-test planning",
    );
    let planner_test_step = job_step(&ci, "detect-changes", "Test CI impact planner");
    let planner_test = planner_test_step
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if planner_test != "python3 scripts/ci_test_plan.test.py" {
        violations.push("detect-changes does not test the impact planner before use".into());
    }
    let planner = job_step(&ci, "detect-changes", "Plan affected Rust tests");
    let planner_run = planner
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    let changed_files = planner
        .and_then(|step| step["env"]["CHANGED_FILES_JSON"].as_str())
        .unwrap_or_default();
    let event_name = planner
        .and_then(|step| step["env"]["CI_EVENT_NAME"].as_str())
        .unwrap_or_default();
    let planner_condition = planner
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let detect_steps = ci["jobs"]["detect-changes"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let detect_step_index = |name: &str| {
        detect_steps
            .iter()
            .position(|step| step["name"].as_str() == Some(name))
    };
    let planner_order = (
        detect_step_index("Set up Rust for affected-test planning"),
        detect_step_index("Test CI impact planner"),
        detect_step_index("Plan affected Rust tests"),
    );
    if planner_setup.and_then(|step| step["if"].as_str()) != Some(universal_planner_condition)
        || planner_setup
            .and_then(|step| step["uses"].as_str())
            .is_none_or(|uses| !uses.starts_with("dtolnay/rust-toolchain@"))
        || planner_setup.and_then(|step| step["with"]["toolchain"].as_str()) != Some("1.95.0")
        || planner_test_step.and_then(|step| step["if"].as_str())
            != Some(universal_planner_condition)
        || planner_condition != universal_planner_condition
        || planner_setup.is_some_and(|step| step.get("continue-on-error").is_some())
        || planner_test_step.is_some_and(|step| step.get("continue-on-error").is_some())
        || planner.is_some_and(|step| step.get("continue-on-error").is_some())
        || !matches!(planner_order, (Some(setup), Some(test), Some(plan)) if setup < test && test < plan)
    {
        violations.push(
            "affected-test planner setup/tests/plan do not run in order on every non-reused CI run"
                .into(),
        );
    }
    if planner.and_then(|step| step["id"].as_str()) != Some("test-plan")
        || !planner_run.contains("cargo metadata --format-version 1 --locked --no-deps")
        || !planner_run.contains("python3 scripts/ci_test_plan.py plan")
        || !planner_run.contains("--changed-files-json \"$CHANGED_FILES_JSON\"")
        || !planner_run.contains("--event-name \"$CI_EVENT_NAME\"")
        || !planner_run.contains("--github-output \"$GITHUB_OUTPUT\"")
        || changed_files != "${{ steps.filter.outputs.impact_files }}"
        || event_name != "${{ github.event_name }}"
    {
        violations.push(
            "detect-changes does not derive its test plan from Cargo metadata and the complete changed-file inventory"
                .into(),
        );
    }
    for output in [
        "test-plan",
        "workspace-lib-required",
        "cli-server-integration-required",
        "core-integration-required",
        "contract-integration-required",
        "canonical-smokes-required",
        "canonical-acceptance-required",
        "rust-ci-required",
    ] {
        let expected = format!("${{{{ steps.test-plan.outputs.{output} }}}}");
        if ci["jobs"]["detect-changes"]["outputs"][output].as_str() != Some(expected.as_str()) {
            violations.push(format!(
                "detect-changes output {output} does not come from the canonical test planner"
            ));
        }
    }

    let platform_planner = job_step(
        &ci,
        "detect-changes",
        "Plan affected platform behavior tests",
    );
    let platform_planner_run = platform_planner
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    let platform_event = platform_planner
        .and_then(|step| step["env"]["CI_EVENT_NAME"].as_str())
        .unwrap_or_default();
    let platform_macos_files = platform_planner
        .and_then(|step| step["env"]["MACOS_FILES_JSON"].as_str())
        .unwrap_or_default();
    let platform_macos_m4_files = platform_planner
        .and_then(|step| step["env"]["MACOS_M4_FILES_JSON"].as_str())
        .unwrap_or_default();
    let platform_windows_files = platform_planner
        .and_then(|step| step["env"]["WINDOWS_FILES_JSON"].as_str())
        .unwrap_or_default();
    let platform_order = (
        detect_step_index("Plan affected Rust tests"),
        detect_step_index("Plan affected platform behavior tests"),
    );
    if platform_planner.and_then(|step| step["id"].as_str()) != Some("platform-test-plan")
        || platform_planner.and_then(|step| step["if"].as_str())
            != Some(universal_planner_condition)
        || platform_planner.is_some_and(|step| step.get("continue-on-error").is_some())
        || !matches!(platform_order, (Some(canonical), Some(platform)) if canonical < platform)
        || !platform_planner_run.contains("python3 scripts/ci_test_plan.py plan")
        || !platform_planner_run.contains("--scope platform")
        || !platform_planner_run.contains("--changed-files-json \"$CHANGED_FILES_JSON\"")
        || !platform_planner_run.contains("--metadata-file \"$RUNNER_TEMP/cargo-metadata.json\"")
        || !platform_planner_run.contains("--event-name \"$CI_EVENT_NAME\"")
        || !platform_planner_run.contains("--github-output \"$GITHUB_OUTPUT\"")
        || !platform_planner_run.contains("--argjson macos \"$MACOS_FILES_JSON\"")
        || !platform_planner_run.contains("--argjson macos_m4 \"$MACOS_M4_FILES_JSON\"")
        || !platform_planner_run.contains("--argjson windows \"$WINDOWS_FILES_JSON\"")
        || !platform_planner_run
            .contains("(($macos - $macos_m4) + $windows) | unique")
        || platform_macos_files != "${{ steps.filter.outputs.macos_files }}"
        || platform_macos_m4_files
            != "${{ steps.filter.outputs['macos-m4_files'] }}"
        || platform_windows_files != "${{ steps.filter.outputs.windows_files }}"
        || platform_planner_run.contains("impact_files")
        || platform_event
            != "${{ startsWith(github.head_ref, 'release-please--branches--') && 'release-please' || github.event_name }}"
    {
        violations.push(
            "detect-changes does not derive its platform behavior plan from the exact generic \
             macOS plus Windows filter inventories"
                .into(),
        );
    }
    for (output, planner_output) in [
        ("platform-test-plan", "test-plan"),
        ("platform-workspace-lib-required", "workspace-lib-required"),
        (
            "platform-cli-server-integration-required",
            "cli-server-integration-required",
        ),
    ] {
        let expected = format!("${{{{ steps.platform-test-plan.outputs.{planner_output} }}}}");
        if ci["jobs"]["detect-changes"]["outputs"][output].as_str() != Some(expected.as_str()) {
            violations.push(format!(
                "detect-changes output {output} does not come from the platform test planner"
            ));
        }
    }
    if ci["jobs"]["detect-changes"]["outputs"]["nextest-config"].as_str()
        != Some("${{ steps.filter.outputs.nextest-config }}")
    {
        violations.push("detect-changes does not expose the nextest platform-smoke route".into());
    }
    let nextest_platform = job_step(&ci, "test", "Validate nextest config on platform");
    if nextest_platform.and_then(|step| step["if"].as_str())
        != Some("needs.detect-changes.outputs.nextest-config == 'true'")
        || nextest_platform.and_then(|step| step["run"].as_str())
            != Some(
                "cargo nextest run -p wenlan-types --lib -E 'test(/^brand::tests::brand_is_wenlan$/)' --no-tests=fail",
            )
        || !detect_change_filter_paths(&ci, "nextest-config").contains(".config/nextest.toml")
    {
        violations.push(
            "nextest config changes do not execute one bounded real test on each routed platform"
                .into(),
        );
    }
    for contract in [
        "def build_platform_plan(",
        "if not relevant:",
        "no platform behavioral inputs changed",
        "PLATFORM_CONTRACT_ONLY_PATHS = {",
        "path not in PLATFORM_CONTRACT_ONLY_PATHS",
        "behavioral_packages = {\"wenlan-core\", \"wenlan-server\", \"wenlan\"}",
        "choices=(\"canonical\", \"platform\")",
    ] {
        if !planner_source.contains(contract) {
            violations.push(format!(
                "platform test planner lost required contract {contract:?}"
            ));
        }
    }

    let rust_ci_formula = r#"    required["rust-ci-required"] = (
        required["workspace-lib-required"]
        or required["canonical-acceptance-required"]
        or required["contract-integration-required"]
    )"#;
    if !planner_source.contains(rust_ci_formula) {
        violations.push(
            "planner rust-ci-required output is not the union of every required Rust suite".into(),
        );
    }
    let plugin_owner = planner_source.find("if _plugin_job_owns(path):");
    let npm_owner = planner_source.find("if _npm_job_owns(path):");
    let docs_owner = planner_source.find("if _docs_job_owns(path):");
    let cargo_owner = planner_source.find("owner = _owner(path, directories)");
    let fast_owner_order = matches!(
        (plugin_owner, npm_owner, docs_owner, cargo_owner),
        (Some(plugin), Some(npm), Some(docs), Some(cargo))
            if plugin < npm && npm < docs && docs < cargo
    );
    for required in [
        "PLUGIN_JOB_PREFIXES = (",
        "PLUGIN_JOB_FILES = {",
        "DOCS_JOB_PREFIXES = (\"docs\",)",
        "DOCS_JOB_FILES = {",
        "def _npm_job_owns(path: str) -> bool:",
        "return len(parts) >= 4 and parts[0] == \"crates\" and parts[2] == \"npm\"",
        "def _rust_fixture_owns(path: str) -> bool:",
        "return path.endswith(\".md\") and not _rust_fixture_owns(path)",
    ] {
        if !planner_source.contains(required) {
            violations.push(format!(
                "planner non-Rust fast-owner contract omits {required:?}"
            ));
        }
    }
    if !fast_owner_order {
        violations
            .push("planner non-Rust fast owners do not run before Cargo ownership fallback".into());
    }
    let unknown_owner = planner_source.find("if owner is None:");
    let unknown_full = planner_source.find("return _full_plan(f\"unowned changed path: {path}\")");
    if !matches!(
        (cargo_owner, unknown_owner, unknown_full),
        (Some(owner), Some(unknown), Some(full)) if owner < unknown && unknown < full
    ) {
        violations.push("planner unknown paths do not fail closed into the full Rust plan".into());
    }

    let rust_paths = detect_change_filter_paths(&ci, "rust");
    if !rust_paths.contains("crates/**/*.rs") {
        violations.push(
            "Linux canonical routing is not a fail-closed catch-all for tracked Rust sources"
                .into(),
        );
    }
    if !rust_paths.contains("crates/**/tests/**") {
        violations.push(
            "Linux canonical routing omits non-Rust test fixtures under crates/**/tests/**".into(),
        );
    }
    if !rust_paths.contains(".github/workflows/coverage.yml") {
        violations.push(
            "coverage workflow cannot bootstrap its FastEmbed cache contract through rust".into(),
        );
    }
    if !rust_paths.contains(".github/workflows/release-please.yml") {
        violations.push(
            "release-please workflow cannot bootstrap its post-CI trigger contract through rust"
                .into(),
        );
    }
    if !rust_paths.contains(".github/workflows/release-pr-maintenance.yml") {
        violations.push(
            "release PR maintenance workflow cannot bootstrap its push-only contract through rust"
                .into(),
        );
    }
    if !rust_paths.contains("clippy.toml") {
        violations.push(
            "clippy configuration cannot bootstrap its syntax-aware FastEmbed guard through rust"
                .into(),
        );
    }
    for plugin_tree in [
        "plugin/**",
        "plugin-codex/**",
        ".agents/plugins/**",
        ".claude-plugin/**",
        "plugin-contract.json",
    ] {
        if rust_paths.contains(plugin_tree) {
            violations.push(format!(
                "broad rust routing still claims plugin-only tree {plugin_tree}"
            ));
        }
    }
    let plugin_rust_contract = detect_change_filter_paths(&ci, "plugin-rust-contract");
    let expected_plugin_rust_contract = BTreeSet::from([
        ".agents/plugins/**".to_string(),
        ".claude-plugin/**".to_string(),
        "plugin-contract.json".to_string(),
        "plugin-codex/**".to_string(),
        "plugin/**".to_string(),
    ]);
    if plugin_rust_contract != expected_plugin_rust_contract {
        violations.push(format!(
            "plugin-rust-contract routing drifted: {plugin_rust_contract:?}"
        ));
    }
    let expected_plugin = BTreeSet::from([
        ".agents/plugins/**".to_string(),
        ".claude-plugin/**".to_string(),
        "plugin-contract.json".to_string(),
        "plugin-codex/**".to_string(),
        "plugin/**".to_string(),
        "scripts/validate-codex-plugin-slice.py".to_string(),
        "scripts/validate-plugin-contract.py".to_string(),
        "scripts/validate-plugin-contract.test.sh".to_string(),
    ]);
    if detect_change_filter_paths(&ci, "plugin") != expected_plugin {
        violations.push("plugin fast-owner routing is not exact".into());
    }
    if detect_change_filter_paths(&ci, "plugin-version-contract")
        != BTreeSet::from(["plugin/.claude-plugin/plugin.json".to_string()])
    {
        violations.push("plugin version routing is not exact".into());
    }
    if detect_change_filter_paths(&ci, "npm") != BTreeSet::from(["crates/*/npm/**".to_string()]) {
        violations.push("npm fast-owner routing is not exact".into());
    }
    let windows_paths = detect_change_filter_paths(&ci, "windows");
    let macos_paths = detect_change_filter_paths(&ci, "macos");
    let mcp_platform = detect_change_filter_paths(&ci, "mcp-platform");
    for (path, platform, filter) in platform_sensitive_paths {
        let routed =
            if path.starts_with("crates/wenlan-mcp/") || path.starts_with("crates/wenlan-types/") {
                &mcp_platform
            } else {
                match *filter {
                    "macos" => &macos_paths,
                    "windows" => &windows_paths,
                    _ => &rust_paths,
                }
            };
        if !filter_routes_path(routed, path) {
            violations.push(format!(
                "platform-sensitive {platform} path is not fail-closed through its owner: {path}"
            ));
        }
        if !filter_routes_path(&rust_paths, path) {
            violations.push(format!(
                "platform-sensitive {platform} path cannot bootstrap the CI contract through rust: {path}"
            ));
        }
    }

    let release_sensitive = detect_change_filter_paths(&ci, "release-preflight");
    for path in release_profile_sensitive_paths {
        if !filter_routes_path(&release_sensitive, path) {
            violations.push(format!(
                "release-profile-sensitive source is not routed through release-preflight: {path}"
            ));
        }
    }
    for path in [
        "Cargo.toml",
        "rust-toolchain.toml",
        "crates/wenlan-core/Cargo.toml",
        "crates/**/build.rs",
        "crates/*/npm/**",
        "install.sh",
        "scripts/stage-onnxruntime-windows.ps1",
        "scripts/setup-vulkan-sdk-windows.ps1",
        "scripts/build-release-binaries.sh",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    ] {
        if !release_sensitive.contains(path) {
            violations.push(format!(
                "release-preflight routing omits native/build-sensitive path {path}"
            ));
        }
    }

    for (filter, paths) in [
        ("rust", &rust_paths),
        ("macos", &macos_paths),
        ("windows", &windows_paths),
    ] {
        if !paths.contains(".config/nextest.toml") {
            violations.push(format!(
                "{filter} routing omits nextest config that guards core-test parallelism"
            ));
        }
    }
    for extension in ["c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx"] {
        let path = format!("crates/**/*.{extension}");
        for (filter, paths) in [
            ("rust", &rust_paths),
            ("macos", &macos_paths),
            ("windows", &windows_paths),
        ] {
            if !paths.contains(&path) {
                violations.push(format!(
                    "{filter} routing omits shared native source glob {path}"
                ));
            }
        }
        if !release_sensitive.contains(&path) {
            violations.push(format!(
                "release-preflight routing omits shared native source glob {path}"
            ));
        }
    }
    for extension in ["m", "mm"] {
        let path = format!("crates/**/*.{extension}");
        if !release_sensitive.contains(&path) {
            violations.push(format!(
                "release-preflight routing omits Apple native source glob {path}"
            ));
        }
    }
    for (platform, paths) in [("macos", &macos_paths), ("windows", &windows_paths)] {
        if !filter_routes_path(paths, "install.sh") {
            violations.push(format!(
                "{platform} routing omits the root installer install.sh"
            ));
        }
    }
    let windows_lint = detect_change_filter_paths(&ci, "windows-lint");
    for path in [
        "crates/wenlan-core/src/lint/**",
        "scripts/lint-scale-gate.sh",
        "scripts/stage-vulkan-loader-windows.ps1",
        "scripts/stage-vulkan-loader-windows.test.ps1",
    ] {
        if !windows_lint.contains(path) {
            violations.push(format!(
                "windows-lint routing omits lint-sensitive path {path}"
            ));
        }
    }
    for path in &windows_lint {
        if !filter_routes_path(&windows_paths, path) {
            violations.push(format!(
                "lint-sensitive Windows path does not also schedule the Windows job: {path}"
            ));
        }
    }
    let macos_m4 = detect_change_filter_paths(&ci, "macos-m4");
    let expected_macos_m4: BTreeSet<String> = [
        "crates/wenlan-core/src/community_grouping.rs",
        "crates/wenlan-core/src/community_partition.rs",
        "crates/wenlan-core/src/edge_grounding.rs",
        "crates/wenlan-core/src/provenance.rs",
        "crates/wenlan-core/src/db.rs",
        "crates/wenlan-core/src/db/community_grouping_state.rs",
        "crates/wenlan-core/src/refinery/mod.rs",
        "crates/wenlan-core/tests/m4_community_gates.rs",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    for path in expected_macos_m4.difference(&macos_m4) {
        violations.push(format!("macos-m4 routing omits M4 contract input {path}"));
    }
    for path in macos_m4.difference(&expected_macos_m4) {
        violations.push(format!(
            "macos-m4 focused-owner allowlist has unreviewed path {path}"
        ));
    }
    for path in &macos_m4 {
        if !filter_routes_path(&macos_paths, path) {
            violations.push(format!(
                "M4-sensitive path does not also schedule the macOS job: {path}"
            ));
        }
    }
    let m4_platform_target = "crates/wenlan-core/tests/m4_community_gates.rs";
    if !platform_sensitive_paths
        .iter()
        .any(|(path, platform, _)| path == m4_platform_target && *platform == "macos")
    {
        violations.push("the focused M4 target lost its real macOS cfg branch".into());
    }
    for (path, platform, _) in platform_sensitive_paths {
        if macos_m4.contains(path) && path != m4_platform_target {
            violations.push(format!(
                "macos-m4 focused source {path} acquired {platform}-specific cfg; retain it in \
                 the generic platform plan or move that proof into the focused target"
            ));
        }
    }

    let windows_llm = detect_change_filter_paths(&ci, "windows-llm-probe");
    for path in [
        "crates/wenlan-core/src/engine.rs",
        "crates/wenlan-core/src/llm_provider.rs",
        "crates/wenlan-core/src/bin/model_probe.rs",
        "crates/wenlan-core/Cargo.toml",
        "crates/**/build.rs",
        "scripts/stage-onnxruntime-windows.ps1",
        "scripts/smoke-windows-llm.ps1",
        "scripts/smoke-windows-llm.test.ps1",
        "scripts/setup-vulkan-sdk-windows.ps1",
        "scripts/setup-vulkan-sdk-windows.test.ps1",
        "scripts/stage-vulkan-loader-windows.ps1",
        "scripts/stage-vulkan-loader-windows.test.ps1",
        "scripts/setup-msvc-ninja-windows.ps1",
        "scripts/setup-msvc-ninja-windows.test.ps1",
    ] {
        if !windows_llm.contains(path) {
            violations.push(format!(
                "windows-llm-probe routing omits LLM contract input {path}"
            ));
        }
    }
    for path in &windows_llm {
        if !filter_routes_path(&windows_paths, path) {
            violations.push(format!(
                "LLM-sensitive path does not also schedule the Windows job: {path}"
            ));
        }
    }
    for path in [
        "crates/wenlan-mcp/src/**",
        "crates/wenlan-mcp/Cargo.toml",
        "crates/wenlan-mcp/build.rs",
        "crates/wenlan-types/src/**",
        "crates/wenlan-types/Cargo.toml",
        "crates/wenlan-types/build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    ] {
        if !mcp_platform.contains(path) {
            violations.push(format!(
                "mcp-platform routing omits platform-compile-sensitive path {path}"
            ));
        }
    }

    for job in ["lint", "test"] {
        let actual = job_needs(&ci, job);
        if actual != ["detect-changes"] {
            violations.push(format!(
                "{job} is unnecessarily serialized; needs={actual:?}, expected [\"detect-changes\"]"
            ));
        }
    }
    let differential_timeout =
        "${{ (matrix.os == 'windows-2022' || github.event_name != 'pull_request') && 60 || 45 }}";
    if ci["jobs"]["test"]["timeout-minutes"].as_str() != Some(differential_timeout) {
        violations.push(
            "test does not enforce the 45-minute non-Windows PR budget while allowing a 60-minute Windows/non-PR backstop".into(),
        );
    }
    let release_preflight_condition = ci["jobs"]["release-preflight"]["if"]
        .as_str()
        .unwrap_or_default();
    if job_needs(&ci, "release-preflight") != ["detect-changes"]
        || !release_preflight_condition.contains("github.event_name != 'pull_request'")
        || !release_preflight_condition
            .contains("startsWith(github.head_ref, 'release-please--branches--')")
        || !release_preflight_condition
            .contains("needs.detect-changes.outputs.release-preflight == 'true'")
    {
        violations.push(
            "release-preflight is not isolated to release-sensitive PRs and release backstops"
                .into(),
        );
    }
    for profile in ["DEV", "TEST"] {
        let key = format!("CARGO_PROFILE_{profile}_DEBUG");
        let actual = ci["jobs"]["test"]["env"][&key].as_str();
        if actual != Some("0") {
            violations.push(format!(
                "test job sets {key}={actual:?}, expected \"0\" for dev/test artifact reuse"
            ));
        }
    }
    if job_step_using(&ci, "test", "sccache-action").is_some()
        || ci["jobs"]["test"]["env"]["RUSTC_WRAPPER"].as_str() == Some("sccache")
        || ci["jobs"]["test"]["env"]["SCCACHE_GHA_RW_MODE"]
            .as_str()
            .is_some()
    {
        violations.push("platform test matrix still owns a compiler-cache lane".into());
    }
    let main_owned_cache = "${{ github.ref == 'refs/heads/main' }}";
    for job in ["lint", "test", "mcp-platform", "release-preflight"] {
        let rust_cache = job_step_using(&ci, job, "Swatinem/rust-cache");
        if rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some(main_owned_cache) {
            violations.push(format!("{job} cache writes are not restricted to main"));
        }
    }
    let test_rust_cache = job_step_using(&ci, "test", "Swatinem/rust-cache");
    if test_rust_cache.and_then(|step| step["with"]["cache-targets"].as_str()) != Some("true") {
        violations.push("platform test matrix does not retain its reusable target cache".into());
    }
    let contract_rust_cache = job_step_using(&ci, "contract-integration", "Swatinem/rust-cache");
    if contract_rust_cache.and_then(|step| step["with"]["shared-key"].as_str()) != Some("test")
        || contract_rust_cache.and_then(|step| step["with"]["cache-targets"].as_str())
            != Some("false")
        || contract_rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some("false")
    {
        violations.push(
            "contract-integration does not reuse test inputs through a restore-only target-free rust-cache"
                .into(),
        );
    }
    for job in [
        "canonical-acceptance",
        "contract-integration",
        "plugin-rust-contract",
    ] {
        if ci["jobs"][job]["env"]["SCCACHE_GHA_RW_MODE"].as_str() != Some("READ_ONLY") {
            violations.push(format!("{job} sccache mode is not read-only"));
        }
    }
    let rust_ci_job_condition = "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.rust-ci-required == 'true'";
    for job in ["fmt", "lint"] {
        if ci["jobs"][job]["if"].as_str() != Some(rust_ci_job_condition) {
            violations.push(format!(
                "required Rust job {job} does not derive exactly from rust-ci-required"
            ));
        }
    }
    let contract_condition = "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.contract-integration-required == 'true'";
    if ci["jobs"]["contract-integration"]["if"].as_str() != Some(contract_condition) {
        violations.push(
            "required contract integration job does not derive exactly from its planner output"
                .into(),
        );
    }
    let platform_condition = ci["jobs"]["test"]["if"].as_str().unwrap_or_default();
    let expected_platform_condition = "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.rust-ci-required == 'true' && (needs.detect-changes.outputs.macos == 'true' || needs.detect-changes.outputs.windows == 'true' || startsWith(github.head_ref, 'release-please--branches--') || github.event_name != 'pull_request')";
    if platform_condition != expected_platform_condition {
        violations.push(
            "platform test job does not derive from rust-ci-required plus its platform owner"
                .into(),
        );
    }
    for required in [
        "needs.detect-changes.outputs.rust-ci-required",
        "needs.detect-changes.outputs.macos",
        "needs.detect-changes.outputs.windows",
        "startsWith(github.head_ref, 'release-please--branches--')",
        "github.event_name != 'pull_request'",
    ] {
        if !platform_condition.contains(required) {
            violations.push(format!(
                "test condition omits platform scheduling trigger {required}"
            ));
        }
    }
    if platform_condition.contains("needs.detect-changes.outputs.rust == 'true'") {
        violations.push("platform test matrix is still scheduled by broad Rust ownership".into());
    }

    let matrix_run = ci["jobs"]["detect-changes"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .find(|step| step["id"].as_str() == Some("matrix"))
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    for required in [
        "github.event_name",
        "pull_request",
        "startsWith(github.head_ref, 'release-please--branches--')",
        "steps.filter.outputs.macos",
        "steps.filter.outputs.windows",
    ] {
        if !matrix_run.contains(required) {
            violations.push(format!(
                "dynamic OS matrix is missing differential/backstop routing marker {required:?}"
            ));
        }
    }
    if matrix_run.contains("ubuntu-24.04") {
        violations.push("dynamic platform matrix still contains a duplicate Linux compiler".into());
    }

    let conclusion_run =
        workflow_step_run(&ci, "Aggregate expected CI results").unwrap_or_default();
    if !conclusion_run.contains("expect_job") || conclusion_run.contains("success|skipped") {
        violations.push(
            "conclusion has no expected-vs-actual contract and accepts skipped jobs generically"
                .into(),
        );
    }
    for required in [
        "needs.detect-changes.outputs.rust-ci-required",
        "needs.detect-changes.outputs.macos",
        "needs.detect-changes.outputs.windows",
        "startsWith(github.head_ref, 'release-please--branches--')",
        "github.event_name != 'pull_request'",
    ] {
        if !conclusion_run.contains(required) {
            violations.push(format!(
                "conclusion expectation omits CI scheduling trigger {required}"
            ));
        }
    }
    let run_rust = "run_rust='${{ needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.rust-ci-required == 'true' }}'";
    let run_platform = "run_platform='${{ needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.rust-ci-required == 'true' && (needs.detect-changes.outputs.macos == 'true' || needs.detect-changes.outputs.windows == 'true' || startsWith(github.head_ref, 'release-please--branches--') || github.event_name != 'pull_request') }}'";
    for (name, expected) in [("run_rust", run_rust), ("run_platform", run_platform)] {
        if !conclusion_run
            .lines()
            .map(str::trim)
            .any(|line| line == expected)
        {
            violations.push(format!(
                "conclusion {name} does not derive exactly from rust-ci-required"
            ));
        }
    }
    if conclusion_run.contains("github.event.head_commit.message") {
        violations.push(
            "conclusion can accept skipped non-PR backstops based on the head commit message"
                .into(),
        );
    }
    let conclusion_needs = job_needs(&ci, "conclusion");
    for job in [
        "linux-nextest-build",
        "linux-nextest",
        "mcp-platform",
        "canonical-acceptance",
        "contract-integration",
        "release-preflight",
        "plugin-rust-contract",
    ] {
        if !conclusion_needs.iter().any(|candidate| candidate == job) {
            violations.push(format!("conclusion.needs omits {job}"));
        }
    }
    for (job, expected) in [
        ("fmt", "\"$run_rust\""),
        ("lint", "\"$run_rust\""),
        ("linux-nextest-build", "\"$run_workspace_lib\""),
        ("linux-nextest", "\"$run_workspace_lib\""),
        ("test", "\"$run_platform\""),
        (
            "mcp-platform",
            "needs.detect-changes.outputs.workspace-platform",
        ),
        ("canonical-acceptance", "\"$run_canonical\""),
        ("contract-integration", "\"$run_contract\""),
        ("release-preflight", "startsWith(github.head_ref"),
        ("docs", "needs.detect-changes.outputs.docs"),
        ("plugin", "needs.detect-changes.outputs.plugin"),
        (
            "plugin-rust-contract",
            "needs.detect-changes.outputs.plugin-rust-contract",
        ),
        ("npm", "needs.detect-changes.outputs.npm"),
    ] {
        let result = format!("needs.{job}.result");
        if !conclusion_run.lines().any(|line| {
            line.contains(&format!("expect_job {job}"))
                && line.contains(expected)
                && line.contains(&result)
        }) {
            violations.push(format!(
                "conclusion does not compare expected-vs-actual result for {job}"
            ));
        }
    }

    for step_name in [
        "Build and smoke shipped release binaries",
        "Native ORT smoke (Windows release preflight)",
    ] {
        if job_step(&ci, "test", step_name).is_some() {
            violations.push(format!(
                "{step_name} still serializes release proof inside the Windows test matrix"
            ));
        }
    }
    let windows_linker = job_step(&ci, "test", "Configure rust-lld linker (Windows tests)");
    let windows_linker_condition = windows_linker
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let windows_linker_run = windows_linker
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !windows_linker_condition.contains("matrix.os == 'windows-2022'")
        || !windows_linker_run.contains("rust-lld.exe")
        || !windows_linker_run.contains("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=")
        || !windows_linker_run.contains("RUSTFLAGS=")
        || !windows_linker_run.contains("$env:GITHUB_ENV")
    {
        violations.push("test does not configure rust-lld for Windows".into());
    }

    let windows_lint_condition = job_step(&ci, "test", "Page lint scale gate (Windows functional)")
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    if !windows_lint_condition.contains("needs.detect-changes.outputs.windows-lint == 'true'")
        || !windows_lint_condition.contains("github.event_name != 'pull_request'")
    {
        violations.push(
            "Windows page-lint proof is not gated to lint-sensitive PRs plus non-PR backstops"
                .into(),
        );
    }

    let m4_condition = "matrix.os == 'macos-14' && (github.event_name != 'pull_request' || startsWith(github.head_ref, 'release-please--branches--') || needs.detect-changes.outputs.macos-m4 == 'true')";
    if job_step(&ci, "test", "M4 community gates (macOS-owned)")
        .and_then(|step| step["if"].as_str())
        != Some(m4_condition)
    {
        violations.push(
            "macOS M4 gate is not focused to community/provenance inputs plus backstops".into(),
        );
    }
    let windows_llm_condition = "matrix.os == 'windows-2022' && (github.event_name != 'pull_request' || startsWith(github.head_ref, 'release-please--branches--') || needs.detect-changes.outputs.windows-llm-probe == 'true')";
    for step_name in [
        "Validate Windows smoke harness",
        "Validate Windows LLM probe",
    ] {
        if job_step(&ci, "test", step_name).and_then(|step| step["if"].as_str())
            != Some(windows_llm_condition)
        {
            violations.push(format!(
                "{step_name} is not focused to Windows LLM inputs plus backstops"
            ));
        }
    }

    let debug_build = job_step(&ci, "test", "Build Windows contract binaries");
    if debug_build.and_then(|step| step["if"].as_str())
        != Some("matrix.os == 'windows-2022' && needs.detect-changes.outputs.platform-cli-server-integration-required == 'true'")
        || debug_build
        .and_then(|step| step["run"].as_str())
        .is_none_or(|run| {
            !run.contains("cargo build -p wenlan -p wenlan-server")
                || !run.contains("needs.detect-changes.outputs.workspace-platform")
                || !run.contains("scripts/stage-onnxruntime-windows.ps1")
                || !run.contains("scripts/stage-vulkan-loader-windows.ps1")
                || !run.contains("target\\debug")
                || run.contains("--release")
        })
    {
        violations.push(
            "ordinary Windows contract does not build and stage adjacent ONNX/Vulkan debug runtime artifacts".into(),
        );
    }
    let schtasks_step = job_step(
        &ci,
        "test",
        "E2E wenlan background on/off round-trip (Windows; schtasks)",
    );
    let schtasks = schtasks_step
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if schtasks_step.and_then(|step| step["if"].as_str())
        != Some("matrix.os == 'windows-2022' && needs.detect-changes.outputs.platform-cli-server-integration-required == 'true'")
        || !schtasks.contains(r"target\debug\wenlan.exe")
    {
        violations.push("ordinary Windows schtasks contract does not use debug binaries".into());
    }

    let mcp_compile = job_step(&ci, "mcp-platform", "Compile platform-owned MCP runtime");
    let mcp_condition = ci["jobs"]["mcp-platform"]["if"]
        .as_str()
        .unwrap_or_default();
    let mcp_run = mcp_compile
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    let mcp_rust_cache = job_step_using(&ci, "mcp-platform", "Swatinem/rust-cache");
    let test_owner_formula = "${{ github.event_name != 'pull_request' || startsWith(github.head_ref, 'release-please--branches--') || (matrix.os == 'macos-14' && needs.detect-changes.outputs.macos == 'true') || (matrix.os == 'windows-2022' && needs.detect-changes.outputs.windows == 'true') }}";
    let mcp_windows_linker = job_step(
        &ci,
        "mcp-platform",
        "Configure rust-lld linker (Windows MCP compile)",
    );
    let mcp_oses = ci["jobs"]["mcp-platform"]["strategy"]["matrix"]["os"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<BTreeSet<_>>();
    if mcp_oses != BTreeSet::from(["macos-14", "windows-2022"])
        || !mcp_condition.contains("needs.detect-changes.outputs.workspace-platform == 'true'")
        || !mcp_condition.contains("needs.detect-changes.outputs.mcp-platform == 'true'")
        || !mcp_condition.contains("startsWith(github.head_ref, 'release-please--branches--')")
        || !mcp_condition.contains("github.event_name != 'pull_request'")
        || !mcp_run.contains("cargo check -p wenlan-mcp --lib --bins")
        || mcp_run.contains("--all-targets")
        || mcp_compile.and_then(|step| step["if"].as_str())
            != Some("env.TEST_OWNS_PLATFORM == 'true' || needs.detect-changes.outputs.workspace-platform != 'true'")
        || ci["jobs"]["mcp-platform"]["env"]["CARGO_PROFILE_DEV_DEBUG"].as_str() != Some("0")
        || ci["jobs"]["mcp-platform"]["env"]["CARGO_PROFILE_TEST_DEBUG"].as_str() != Some("0")
        || ci["jobs"]["mcp-platform"]["env"]["TEST_OWNS_PLATFORM"].as_str()
            != Some(test_owner_formula)
        || mcp_rust_cache.and_then(|step| step["with"]["shared-key"].as_str())
            != Some("mcp-platform")
        || mcp_rust_cache.and_then(|step| step["with"]["cache-all-crates"].as_str())
            != Some("false")
        || mcp_rust_cache.and_then(|step| step["with"]["cache-targets"].as_str()) != Some("false")
        || mcp_rust_cache.is_some_and(|step| step.get("if").is_some())
        || mcp_windows_linker.and_then(|step| step["if"].as_str())
            != Some("matrix.os == 'windows-2022'")
        || mcp_windows_linker
            .and_then(|step| step["run"].as_str())
            .is_none_or(|run| !run.contains("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"))
    {
        violations.push(
            "independent macOS/Windows ownership does not differentially compile every wenlan-mcp target with a bounded cache footprint"
                .into(),
        );
    }

    let transferred_workspace = job_step(
        &ci,
        "test",
        "Build CLI/server contract binaries (test owner)",
    );
    if job_step(
        &ci,
        "test",
        "Compile platform-owned MCP runtime (test owner)",
    )
    .is_some()
        || transferred_workspace.and_then(|step| step["if"].as_str())
            != Some("needs.detect-changes.outputs.workspace-platform == 'true' && !(matrix.os == 'macos-14' && needs.detect-changes.outputs.release-preflight == 'true')")
        || transferred_workspace
            .and_then(|step| step["run"].as_str())
            .is_none_or(|run| {
                !run.contains("cargo build -p wenlan -p wenlan-server --bins")
                    || run.contains("wenlan-mcp")
            })
    {
        violations.push(
            "CLI/server platform proof is not transferred without pulling MCP into the critical test owner"
                .into(),
        );
    }

    for (job, step_name, suite, plan_argument, expected_plan) in [
        (
            "test",
            "Workspace lib tests (macOS)",
            "workspace-lib",
            "--plan-json \"$CI_TEST_PLAN\"",
            "${{ needs.detect-changes.outputs.platform-test-plan }}",
        ),
        (
            "test",
            "Integration tests wenlan-cli + wenlan-server (macOS)",
            "cli-server-integration",
            "--plan-json \"$CI_TEST_PLAN\"",
            "${{ needs.detect-changes.outputs.platform-test-plan }}",
        ),
        (
            "test",
            "Integration tests wenlan-cli + wenlan-server (Windows)",
            "cli-server-integration",
            "--plan-env CI_TEST_PLAN",
            "${{ needs.detect-changes.outputs.platform-test-plan }}",
        ),
        (
            "canonical-acceptance",
            "Integration tests wenlan-cli + wenlan-server (Linux)",
            "cli-server-integration",
            "--plan-json \"$CI_TEST_PLAN\"",
            "${{ needs.detect-changes.outputs.test-plan }}",
        ),
        (
            "canonical-acceptance",
            "Run integration tests (core) (Linux)",
            "core-integration",
            "--plan-json \"$CI_TEST_PLAN\"",
            "${{ needs.detect-changes.outputs.test-plan }}",
        ),
    ] {
        let step = job_step(&ci, job, step_name);
        let run = step
            .and_then(|candidate| candidate["run"].as_str())
            .unwrap_or_default();
        let plan = step
            .and_then(|candidate| candidate["env"]["CI_TEST_PLAN"].as_str())
            .unwrap_or_default();
        if !run.contains("python3 scripts/ci_test_plan.py run")
            || !run.contains(&format!("--suite {suite}"))
            || !run.contains(plan_argument)
            || plan != expected_plan
        {
            violations.push(format!(
                "{job} {step_name} does not execute the validated impacted-test plan"
            ));
        }
    }
    for (step_name, output) in [
        (
            "Integration tests wenlan-cli + wenlan-server (Linux)",
            "cli-server-integration-required",
        ),
        (
            "Run integration tests (core) (Linux)",
            "core-integration-required",
        ),
    ] {
        let expected = format!("needs.detect-changes.outputs.{output} == 'true'");
        if job_step(&ci, "canonical-acceptance", step_name).and_then(|step| step["if"].as_str())
            != Some(expected.as_str())
        {
            violations.push(format!(
                "canonical acceptance {step_name} is not gated by planner output {output}"
            ));
        }
    }
    for (step_name, expected_condition) in [
        (
            "Workspace lib tests (macOS)",
            "matrix.os == 'macos-14' && needs.detect-changes.outputs.platform-workspace-lib-required == 'true'",
        ),
        (
            "Integration tests wenlan-cli + wenlan-server (macOS)",
            "matrix.os == 'macos-14' && needs.detect-changes.outputs.platform-cli-server-integration-required == 'true'",
        ),
        (
            "Integration tests wenlan-cli + wenlan-server (Windows)",
            "matrix.os == 'windows-2022' && needs.detect-changes.outputs.platform-cli-server-integration-required == 'true'",
        ),
    ] {
        if job_step(&ci, "test", step_name).and_then(|step| step["if"].as_str())
            != Some(expected_condition)
        {
            violations.push(format!(
                "platform step {step_name} is not gated by its focused platform plan"
            ));
        }
    }

    let Some(test_steps) = ci["jobs"]["test"]["steps"].as_sequence() else {
        violations.push("test job has no steps for fail-fast ordering".into());
        return violations;
    };
    let step_index = |name: &str| {
        test_steps
            .iter()
            .position(|step| step["name"].as_str() == Some(name))
    };
    let integration_index = step_index("Integration tests wenlan-cli + wenlan-server (Windows)");
    match (
        step_index("Validate Windows smoke harness"),
        integration_index,
    ) {
        (Some(harness), Some(integration)) if harness < integration => {}
        _ => violations
            .push("Validate Windows smoke harness does not fail fast before integration".into()),
    }
    match (
        integration_index,
        step_index("Build Windows contract binaries"),
    ) {
        (Some(integration), Some(debug)) if integration < debug => {}
        _ => violations
            .push("Windows steps do not fail fast from integration into the debug build".into()),
    }

    violations
}

#[test]
fn ci_routing_is_fail_closed_and_differential() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let planner = std::fs::read_to_string(root.join("scripts/ci_test_plan.py"))
        .expect("read CI test planner");
    let platform_sensitive_paths = platform_sensitive_paths(&root);
    let release_profile_sensitive_paths = release_profile_sensitive_paths(&root);
    let violations = ci_routing_contract_violations(
        &workflow,
        &planner,
        &platform_sensitive_paths,
        &release_profile_sensitive_paths,
    );
    assert!(
        violations.is_empty(),
        "CI routing contract drift — differential CI routing stays fail-closed: every tracked \
         source class routes to a required owner and unknown paths hit a catch-all. Fix the \
         detect-changes/impact routing in .github/workflows/ci.yml; an intentional contract \
         change also updates ci_routing_contract_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ci_planner_routing_rejects_optional_and_fail_open_mutations() {
    let root = repo_root();
    let workflow = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("read ci.yml")
        .replacen(
            "if: steps.release-proof.outputs.verified-release-merge != 'true'",
            "if: false",
            1,
        )
        .replacen(
            "if: needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.rust-ci-required == 'true'",
            "if: needs.detect-changes.outputs.rust == 'true'",
            1,
        )
        .replacen(
            "steps.test-plan.outputs.rust-ci-required == 'true'",
            "steps.filter.outputs.rust == 'true'",
            1,
        )
        .replacen(
            "(($macos - $macos_m4) + $windows) | unique",
            "($macos + $windows) | unique",
            1,
        )
        .replacen(
            "            macos-m4:\n              - 'crates/wenlan-core/src/community_grouping.rs'",
            "            macos-m4:\n              - 'crates/wenlan-core/src/community_grouping.rs'\n              - 'crates/wenlan-core/src/unreviewed_m4.rs'",
            1,
        );
    let planner = std::fs::read_to_string(root.join("scripts/ci_test_plan.py"))
        .expect("read CI test planner")
        .replacen("        if _docs_job_owns(path):", "        if False:", 1)
        .replacen(
            "return len(parts) >= 4 and parts[0] == \"crates\" and parts[2] == \"npm\"",
            "return False",
            1,
        )
        .replacen(
            "return _full_plan(f\"unowned changed path: {path}\")",
            "return {\"version\": 1, \"mode\": \"differential\"}",
            1,
        )
        .replacen(
            "        or required[\"canonical-acceptance-required\"]",
            "        and required[\"canonical-acceptance-required\"]",
            1,
        )
        .replacen("    if not relevant:", "    if False:", 1);
    let mut platform_sensitive_paths = platform_sensitive_paths(&root);
    platform_sensitive_paths.push(("crates/wenlan-core/src/db.rs".into(), "macos", "macos"));
    let release_profile_sensitive_paths = release_profile_sensitive_paths(&root);
    let mut violations = ci_routing_contract_violations(
        &workflow,
        &planner,
        &platform_sensitive_paths,
        &release_profile_sensitive_paths,
    );
    violations.extend(fastembed_ci_cache_violations(&workflow));
    for expected in [
        "every non-reused CI run",
        "required Rust job fmt",
        "canonical rust-ci-required planner output",
        "rust-ci-required output is not the union",
        "non-Rust fast-owner",
        "unknown paths",
        "platform test planner lost required contract",
        "exact generic macOS plus Windows filter inventories",
        "macos-m4 focused-owner allowlist has unreviewed path",
        "macos-m4 focused source crates/wenlan-core/src/db.rs acquired macos-specific cfg",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "mutation must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn documentation_only_changes_take_the_docs_lane_without_runtime_backstops() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci: serde_yaml::Value = serde_yaml::from_str(&workflow).expect("parse ci.yml");
    let docs = detect_change_filter_paths(&ci, "docs");
    let runtime_filters = [
        "rust",
        "macos",
        "windows",
        "release-preflight",
        "mcp-platform",
        "workspace-platform",
    ]
    .map(|name| (name, detect_change_filter_paths(&ci, name)));

    for path in [
        "AGENTS.md",
        "CLAUDE.md",
        "CHANGELOG.md",
        "README.es-ES.md",
        "app/eval/AGENTS.md",
        "crates/wenlan-core/AGENTS.md",
        "docs/technical-foundations.md",
        "LICENSE",
    ] {
        assert!(
            filter_routes_path(&docs, path),
            "documentation-only path must schedule docs checks: {path}"
        );
        for (filter, patterns) in &runtime_filters {
            assert!(
                !filter_routes_path(patterns, path),
                "documentation-only path must not schedule {filter}: {path}"
            );
        }
    }

    for path in [
        "scripts/check-readme-translations.py",
        "scripts/check-readme-translations.test.sh",
        "scripts/update-readme-eval.py",
        "scripts/update-readme-eval.test.py",
        "scripts/validate-versions.sh",
        "scripts/validate-versions.test.sh",
    ] {
        assert!(
            filter_routes_path(&docs, path),
            "documentation validator must schedule docs checks: {path}"
        );
    }

    let markdown_test_fixture = "crates/wenlan-core/tests/fixtures/folder/notes/linked.md";
    assert!(
        filter_routes_path(&docs, markdown_test_fixture),
        "Markdown test fixtures must still receive docs checks"
    );
    let rust = runtime_filters
        .iter()
        .find(|(name, _)| *name == "rust")
        .map(|(_, patterns)| patterns)
        .expect("rust routing");
    assert!(
        filter_routes_path(rust, markdown_test_fixture),
        "Markdown under crates/**/tests/** must retain Rust test coverage"
    );
}

#[test]
fn ordinary_pr_required_path_excludes_release_and_unowned_platform_backstops() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci: serde_yaml::Value = serde_yaml::from_str(&workflow).expect("parse ci.yml");

    let release_condition = ci["jobs"]["release-preflight"]["if"]
        .as_str()
        .expect("release-preflight condition");
    assert!(
        release_condition.contains("github.event_name != 'pull_request'")
            && release_condition
                .contains("startsWith(github.head_ref, 'release-please--branches--')")
            && release_condition
                .contains("needs.detect-changes.outputs.release-preflight == 'true'"),
        "release preflight must run for release-sensitive PRs, release-please PRs, and non-PR \
         backstops: \
         {release_condition}"
    );

    let matrix = ci["jobs"]["detect-changes"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .find(|step| step["id"].as_str() == Some("matrix"))
        .and_then(|step| step["run"].as_str())
        .expect("dynamic OS matrix");
    assert!(
        matrix.contains("startsWith(github.head_ref, 'release-please--branches--')"),
        "release-please PRs must retain the full OS matrix"
    );

    for filter in ["macos", "windows"] {
        let paths = detect_change_filter_paths(&ci, filter);
        for shared_path in [
            "crates/wenlan-cli/src/**",
            "crates/wenlan-cli/tests/**",
            "crates/wenlan-server/src/**",
            "crates/wenlan-server/tests/**",
            "crates/wenlan-mcp/src/**",
            "crates/wenlan-types/src/**",
            "Cargo.toml",
            "Cargo.lock",
            "crates/**/Cargo.toml",
            "version.txt",
            ".release-please-manifest.json",
            "CHANGELOG.md",
        ] {
            assert!(
                !paths.contains(shared_path),
                "{filter} full-platform routing must not claim shared path {shared_path}"
            );
        }
    }

    let mcp_platform_paths = detect_change_filter_paths(&ci, "mcp-platform");
    let platform_paths = platform_sensitive_paths(&root);
    for (path, platform, filter) in platform_paths {
        if path.starts_with("crates/wenlan-mcp/") || path.starts_with("crates/wenlan-types/") {
            assert!(
                filter_routes_path(&mcp_platform_paths, &path),
                "MCP platform compile does not own {platform}-sensitive path {path}"
            );
        } else {
            let routed = detect_change_filter_paths(&ci, filter);
            assert!(
                filter_routes_path(&routed, &path),
                "{platform}-sensitive path is not routed through {filter}: {path}"
            );
        }
    }

    // PR #392 was a representative shared-code change: it touched the global
    // lockfile, MCP/wire code, ordinary server code, and one macOS-owned
    // scheduler. It must keep the macOS proof and focused MCP compile without
    // paying for the unrelated full Windows CLI/server contract.
    let pr_392_paths = [
        "Cargo.lock",
        "crates/wenlan-core/src/db.rs",
        "crates/wenlan-mcp/src/tools.rs",
        "crates/wenlan-server/Cargo.toml",
        "crates/wenlan-server/src/memory_routes.rs",
        "crates/wenlan-server/src/scheduler.rs",
        "crates/wenlan-server/tests/route_convergence.rs",
        "crates/wenlan-types/src/requests.rs",
    ];
    let macos_paths = detect_change_filter_paths(&ci, "macos");
    let windows_paths = detect_change_filter_paths(&ci, "windows");
    let release_paths = detect_change_filter_paths(&ci, "release-preflight");
    assert!(
        pr_392_paths
            .iter()
            .any(|path| filter_routes_path(&macos_paths, path)),
        "PR #392 must retain its macOS-owned scheduler proof"
    );
    assert!(
        !pr_392_paths
            .iter()
            .any(|path| filter_routes_path(&windows_paths, path)),
        "PR #392 must not schedule the unrelated full Windows contract"
    );
    assert!(
        !pr_392_paths
            .iter()
            .any(|path| filter_routes_path(&release_paths, path)),
        "PR #392 must not schedule the unrelated release-profile matrix"
    );
    assert!(
        pr_392_paths
            .iter()
            .any(|path| filter_routes_path(&mcp_platform_paths, path)),
        "PR #392 must retain the focused MCP platform compile"
    );

    assert!(
        required_job_closure(&ci).contains("mcp-platform"),
        "the independent MCP platform compile must gate conclusion"
    );
    assert_eq!(
        ci["jobs"]["mcp-platform"]["timeout-minutes"].as_u64(),
        Some(20),
        "the focused MCP platform compile must keep a 20-minute ceiling"
    );
}

#[test]
fn ci_release_reuse_and_linux_shards_are_fail_closed() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let planner = std::fs::read_to_string(root.join("scripts/ci_test_plan.py"))
        .expect("read CI test planner");
    let ci: serde_yaml::Value = serde_yaml::from_str(&workflow).expect("parse ci.yml");
    let proof_output = "verified-release-merge";
    let proof_ref = "needs.detect-changes.outputs.verified-release-merge != 'true'";

    let permissions = &ci["jobs"]["detect-changes"]["permissions"];
    assert_eq!(
        permissions["contents"].as_str(),
        Some("read"),
        "release proof needs read-only repository contents"
    );
    assert_eq!(
        permissions["checks"].as_str(),
        Some("read"),
        "release proof must read the required conclusion check"
    );
    assert_eq!(
        permissions["pull-requests"].as_str(),
        Some("read"),
        "release proof must read the associated PR and its file inventory"
    );
    assert!(
        ci["permissions"]["checks"].is_null() && ci["permissions"]["pull-requests"].is_null(),
        "proof-only permissions must not be granted workflow-wide"
    );
    assert!(
        ci["jobs"]["detect-changes"]["outputs"][proof_output]
            .as_str()
            .is_some(),
        "detect-changes must expose the verified release proof"
    );
    let proof = job_step(&ci, "detect-changes", "Verify reusable release merge")
        .expect("release proof step");
    let proof_test = job_step(&ci, "detect-changes", "Test release merge proof")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    assert_eq!(
        proof_test,
        "python3 scripts/verify-release-merge.test.py\npython3 scripts/release-promotion.test.py\n",
        "release proof tests must run before routing"
    );
    assert_eq!(
        proof["id"].as_str(),
        Some("release-proof"),
        "release proof output owner"
    );
    let proof_run = proof["run"].as_str().unwrap_or_default();
    for required in [
        "python3 scripts/release-promotion.py gate-main",
        "--repository \"$GITHUB_REPOSITORY\"",
        "--sha \"$GITHUB_SHA\"",
        "--wait-seconds 720",
        "--main-run-id \"$GITHUB_RUN_ID\"",
        "--main-run-attempt \"$GITHUB_RUN_ATTEMPT\"",
        "--github-output \"$GITHUB_OUTPUT\"",
        "--plan-output \"$RUNNER_TEMP/main-release-promotion-receipt.json\"",
    ] {
        assert!(
            proof_run.contains(required),
            "release proof step omits {required:?}: {proof_run}"
        );
    }

    for job in [
        "fmt",
        "lint",
        "linux-nextest-build",
        "linux-nextest",
        "test",
        "mcp-platform",
        "canonical-acceptance",
        "contract-integration",
        "release-preflight",
        "docs",
        "plugin",
        "plugin-rust-contract",
        "npm",
    ] {
        let condition = ci["jobs"][job]["if"].as_str().unwrap_or_default();
        assert!(
            condition.contains(proof_ref),
            "{job} can repeat a verified release merge: {condition}"
        );
    }

    let archive_condition = "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.workspace-lib-required == 'true'";
    assert_eq!(
        job_needs(&ci, "linux-nextest-build"),
        ["detect-changes"],
        "Linux archive producer must depend only on the planner"
    );
    assert_eq!(
        ci["jobs"]["linux-nextest-build"]["if"].as_str(),
        Some(archive_condition),
        "Linux archive producer must use the canonical workspace-lib gate"
    );
    let archive = job_step(
        &ci,
        "linux-nextest-build",
        "Build workspace-lib nextest archive",
    )
    .expect("single Linux workspace-lib archive producer");
    let archive_run = archive["run"].as_str().unwrap_or_default();
    for required in [
        "python3 scripts/ci_test_plan.py archive",
        "--plan-json \"$CI_TEST_PLAN\"",
        "--archive-file \"$RUNNER_TEMP/wenlan-workspace-lib-${GITHUB_RUN_ID}.tar.zst\"",
    ] {
        assert!(
            archive_run.contains(required),
            "Linux archive producer omits {required:?}: {archive_run}"
        );
    }
    let publish = job_step(
        &ci,
        "linux-nextest-build",
        "Publish workspace-lib nextest archive",
    )
    .expect("Linux archive upload");
    assert_eq!(
        publish["with"]["name"].as_str(),
        Some("wenlan-workspace-lib-nextest-${{ github.run_id }}")
    );
    assert_eq!(publish["with"]["compression-level"].as_u64(), Some(0));
    assert_eq!(publish["with"]["retention-days"].as_u64(), Some(1));
    assert_eq!(publish["with"]["if-no-files-found"].as_str(), Some("error"));
    let producer_count = ci["jobs"]
        .as_mapping()
        .into_iter()
        .flatten()
        .flat_map(|(_, job)| job["steps"].as_sequence().into_iter().flatten())
        .filter(|step| {
            step["with"]["name"].as_str()
                == Some("wenlan-workspace-lib-nextest-${{ github.run_id }}")
                && step["uses"]
                    .as_str()
                    .is_some_and(|uses| uses.starts_with("actions/upload-artifact@"))
        })
        .count();
    assert_eq!(
        producer_count, 1,
        "the workspace-lib archive must have exactly one producer"
    );

    assert_eq!(
        job_needs(&ci, "linux-nextest"),
        ["detect-changes", "linux-nextest-build"],
        "Linux partition consumers must wait for the one archive producer"
    );
    assert_eq!(
        ci["jobs"]["linux-nextest"]["if"].as_str(),
        Some(archive_condition),
        "Linux partition consumers must use the canonical workspace-lib gate"
    );
    let partitions = ci["jobs"]["linux-nextest"]["strategy"]["matrix"]["partition"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(partitions, BTreeSet::from(["slice:1/2", "slice:2/2"]));
    assert_eq!(
        planner.matches("--no-tests=pass").count(),
        2,
        "a globally non-empty filterset may still leave one nextest slice empty; the run must allow that shard while list validation strips the run-only flag"
    );
    let consumer_steps = ci["jobs"]["linux-nextest"]["steps"]
        .as_sequence()
        .expect("Linux archive consumer steps");
    assert!(
        consumer_steps
            .iter()
            .filter_map(|step| step["uses"].as_str())
            .all(|uses| {
                !uses.contains("rust-cache")
                    && !uses.contains("sccache")
                    && !uses.starts_with("actions/cache")
            }),
        "Linux archive consumers must be artifact-only and cache-free"
    );
    assert!(
        ci["jobs"]["linux-nextest"]["env"]["RUSTC_WRAPPER"].is_null()
            && ci["jobs"]["linux-nextest"]["env"]["SCCACHE_GHA_ENABLED"].is_null()
            && ci["jobs"]["linux-nextest"]["env"]["CARGO_PROFILE_DEV_DEBUG"].is_null()
            && ci["jobs"]["linux-nextest"]["env"]["CARGO_PROFILE_TEST_DEBUG"].is_null(),
        "Linux archive consumers must not configure a compiler"
    );
    let archive_download = job_step(
        &ci,
        "linux-nextest",
        "Download workspace-lib nextest archive",
    )
    .expect("Linux archive download");
    assert_eq!(
        archive_download["with"]["name"].as_str(),
        Some("wenlan-workspace-lib-nextest-${{ github.run_id }}")
    );
    let linux = job_step(&ci, "linux-nextest", "Run workspace-lib archive partition")
        .expect("Linux workspace archive partition");
    let linux_run = linux["run"].as_str().unwrap_or_default();
    for required in [
        "--suite workspace-lib",
        "--archive-file \"$RUNNER_TEMP/wenlan-nextest-archive/wenlan-workspace-lib-${GITHUB_RUN_ID}.tar.zst\"",
        "--workspace-remap \"$GITHUB_WORKSPACE\"",
        "--partition \"$CI_TEST_PARTITION\"",
    ] {
        assert!(
            linux_run.contains(required),
            "Linux archive consumer omits {required:?}: {linux_run}"
        );
    }
    assert!(
        consumer_steps
            .iter()
            .filter_map(|step| step["run"].as_str())
            .all(|run| !run.contains("cargo build")
                && !run.contains("cargo check")
                && !run.contains("cargo test")
                && !run.contains("nextest archive")),
        "Linux archive consumers must not compile"
    );
    let macos_run = job_step(&ci, "test", "Workspace lib tests (macOS)")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    assert!(
        !macos_run.contains("--partition"),
        "macOS must retain its single complete test run"
    );

    let conclusion =
        workflow_step_run(&ci, "Aggregate expected CI results").expect("conclusion script");
    assert!(
        conclusion.contains(proof_ref),
        "conclusion can accept skipped main jobs without the verified release proof"
    );
    for job in ["linux-nextest-build", "linux-nextest"] {
        assert!(
            conclusion.lines().any(|line| {
                line.contains(&format!("expect_job {job} \"$run_workspace_lib\""))
                    && line.contains(&format!("needs.{job}.result"))
            }),
            "conclusion does not fail closed on {job}"
        );
    }
}

#[test]
fn ci_manifest_and_lockfile_changes_get_focused_platform_compile_proof() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci: serde_yaml::Value = serde_yaml::from_str(&workflow).expect("parse ci.yml");

    assert!(
        ci["jobs"]["detect-changes"]["outputs"]["workspace-platform"]
            .as_str()
            .is_some(),
        "detect-changes must expose workspace platform dependency routing"
    );
    let paths = detect_change_filter_paths(&ci, "workspace-platform");
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "crates/**/Cargo.toml",
        "rust-toolchain.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    ] {
        assert!(
            paths.contains(path),
            "workspace platform compile omits dependency-sensitive path {path}"
        );
    }
    let condition = ci["jobs"]["mcp-platform"]["if"]
        .as_str()
        .unwrap_or_default();
    assert!(
        condition.contains("needs.detect-changes.outputs.workspace-platform == 'true'"),
        "focused platform job does not schedule workspace dependency changes"
    );
    let workspace = job_step(&ci, "mcp-platform", "Build workspace contract binaries")
        .expect("workspace platform build step");
    assert_eq!(
        workspace["if"].as_str(),
        Some("needs.detect-changes.outputs.workspace-platform == 'true' && env.TEST_OWNS_PLATFORM != 'true'"),
        "workspace build must be limited to dependency-sensitive changes without a test owner"
    );
    let run = workspace["run"].as_str().unwrap_or_default();
    assert!(
        run.contains("cargo build -p wenlan -p wenlan-server -p wenlan-mcp --bins"),
        "workspace platform proof does not link every shipped binary"
    );
    assert!(
        !run.contains("--release"),
        "ordinary dependency PRs must not pay release-preflight cost"
    );

    let platform_steps = ci["jobs"]["mcp-platform"]["steps"]
        .as_sequence()
        .expect("mcp-platform steps");
    let build_index = platform_steps
        .iter()
        .position(|step| step["name"].as_str() == Some("Build workspace contract binaries"))
        .expect("workspace platform build step index");
    let windows_condition = "matrix.os == 'windows-2022' && needs.detect-changes.outputs.workspace-platform == 'true' && env.TEST_OWNS_PLATFORM != 'true'";
    for (step_name, required_command) in [
        (
            "Install sqlite3 (Windows platform build)",
            "vcpkg install sqlite3",
        ),
        (
            "Set up Vulkan SDK (Windows platform build)",
            "scripts/setup-vulkan-sdk-windows.ps1",
        ),
        (
            "Configure MSVC Ninja (Windows platform build)",
            "scripts/setup-msvc-ninja-windows.ps1",
        ),
    ] {
        let step = job_step(&ci, "mcp-platform", step_name)
            .unwrap_or_else(|| panic!("missing workspace platform prerequisite {step_name}"));
        assert_eq!(
            step["if"].as_str(),
            Some(windows_condition),
            "{step_name} must run only for Windows dependency-sensitive changes"
        );
        assert!(
            step["run"]
                .as_str()
                .unwrap_or_default()
                .contains(required_command),
            "{step_name} does not run {required_command}"
        );
        let prerequisite_index = platform_steps
            .iter()
            .position(|candidate| candidate["name"].as_str() == Some(step_name))
            .expect("workspace platform prerequisite index");
        assert!(
            prerequisite_index < build_index,
            "{step_name} must run before the workspace platform build"
        );
    }
}

#[test]
fn ci_fans_out_one_prepared_fastembed_artifact_per_run() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci: serde_yaml::Value = serde_yaml::from_str(&workflow).expect("parse ci.yml");
    let artifact_name = "fastembed-bge-base-en-v1.5-q-v3-portable-${{ github.run_id }}";

    let producer = job_step(
        &ci,
        "detect-changes",
        "Publish portable FastEmbed model for this run",
    )
    .expect("detect-changes must publish the prepared FastEmbed snapshot");
    assert!(
        producer["uses"]
            .as_str()
            .is_some_and(|action| action.starts_with("actions/upload-artifact@")),
        "FastEmbed producer must use upload-artifact"
    );
    assert_eq!(
        producer["with"]["name"].as_str(),
        Some(artifact_name),
        "FastEmbed artifact producer name"
    );
    assert_eq!(
        producer["with"]["path"].as_str(),
        Some(".fastembed_cache"),
        "FastEmbed artifact producer path"
    );
    assert_eq!(
        producer["with"]["include-hidden-files"].as_bool(),
        Some(true),
        "the hidden FastEmbed cache directory must be included"
    );
    assert_eq!(
        producer["with"]["overwrite"].as_bool(),
        Some(true),
        "re-running the full workflow must replace the run-scoped artifact"
    );

    for job in [
        "linux-nextest",
        "test",
        "canonical-acceptance",
        "contract-integration",
    ] {
        assert!(
            job_step_using(&ci, job, "actions/cache/restore").is_none(),
            "{job} must not make a concurrent cache-service restore"
        );
        let consumer = job_step(&ci, job, "Download portable FastEmbed model")
            .unwrap_or_else(|| panic!("{job} must download the prepared FastEmbed artifact"));
        assert!(
            consumer["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with("actions/download-artifact@")),
            "{job} FastEmbed consumer must use the artifact download action"
        );
        assert_eq!(
            consumer["with"]["name"].as_str(),
            Some(artifact_name),
            "{job} FastEmbed artifact name"
        );
        assert_eq!(
            consumer["with"]["path"].as_str(),
            Some(".fastembed_cache"),
            "{job} FastEmbed artifact path"
        );
    }
}

fn plugin_rust_contract_violations(ci_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let mut violations = Vec::new();
    let condition = ci["jobs"]["plugin-rust-contract"]["if"]
        .as_str()
        .unwrap_or_default();
    if job_needs(&ci, "plugin-rust-contract") != ["detect-changes"]
        || condition
            != "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.plugin-rust-contract == 'true'"
        || ci["jobs"]["plugin-rust-contract"]["continue-on-error"].as_bool() == Some(true)
        || ci["jobs"]["plugin-rust-contract"]["runs-on"].as_str() != Some("ubuntu-24.04")
        || ci["jobs"]["plugin-rust-contract"]["timeout-minutes"].as_u64() != Some(20)
    {
        violations.push("plugin-rust-contract job is not an exact required bounded owner".into());
    }

    let contracts = [
        (
            "Validate Claude plugin distribution",
            "cargo nextest run -p wenlan --test distribution --no-tests=fail",
            None,
        ),
        (
            "Validate cross-surface plugin distribution",
            "cargo nextest run -p wenlan-types --test plugin_distribution --no-tests=fail",
            None,
        ),
        (
            "Validate skill MCP tool references",
            "cargo nextest run -p wenlan-mcp --lib -E 'test(/^tools::tests::skills_reference_only_live_tools$/)' --no-tests=fail",
            None,
        ),
        (
            "Validate plugin release version sync",
            "cargo nextest run -p wenlan-core --lib -E 'test(/^drift_guard::version_files_are_in_sync$/)' --no-tests=fail",
            Some("needs.detect-changes.outputs.plugin-version-contract == 'true'"),
        ),
    ];
    for (name, command, condition) in contracts {
        let step = job_step(&ci, "plugin-rust-contract", name);
        if step.and_then(|step| step["run"].as_str()) != Some(command)
            || step.and_then(|step| step["if"].as_str()) != condition
        {
            violations.push(format!(
                "plugin-rust-contract lost exact consumer contract {name:?}"
            ));
        }
    }
    let cargo_contract_count = ci["jobs"]["plugin-rust-contract"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(|step| step["run"].as_str())
        .filter(|run| run.trim_start().starts_with("cargo nextest run"))
        .count();
    if cargo_contract_count != 4 {
        violations.push(format!(
            "plugin-rust-contract has {cargo_contract_count} Cargo consumer contracts, expected four"
        ));
    }
    let conclusion_needs = job_needs(&ci, "conclusion");
    let conclusion = workflow_step_run(&ci, "Aggregate expected CI results").unwrap_or_default();
    if !conclusion_needs
        .iter()
        .any(|job| job == "plugin-rust-contract")
        || !conclusion.lines().any(|line| {
            line.contains("expect_job plugin-rust-contract")
                && line.contains("needs.detect-changes.outputs.plugin-rust-contract")
                && line.contains("needs.plugin-rust-contract.result")
        })
    {
        violations.push("conclusion does not fail closed on plugin-rust-contract".into());
    }
    violations
}

#[test]
fn plugin_trees_use_four_required_rust_consumer_contracts() {
    let workflow =
        std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read ci.yml");
    let violations = plugin_rust_contract_violations(&workflow);
    assert!(
        violations.is_empty(),
        "plugin Rust contract drift — plugin trees stay out of broad Rust ownership while their \
         four exact Rust consumers remain required. Fix .github/workflows/ci.yml; an intentional \
         contract change also updates plugin_rust_contract_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn plugin_rust_contract_rejects_optional_or_incomplete_fixture() {
    let workflow = r#"
jobs:
  plugin-rust-contract:
    needs: docs
    if: false
    runs-on: windows-2022
    timeout-minutes: 60
    continue-on-error: true
    steps:
      - name: Validate Claude plugin distribution
        run: cargo test
  conclusion:
    needs: [docs]
"#;
    let violations = plugin_rust_contract_violations(workflow);
    for expected in [
        "exact required bounded owner",
        "exact consumer contract",
        "expected four",
        "conclusion does not fail closed",
    ] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn ci_routing_contract_rejects_fail_open_fixture() {
    let workflow = r#"
jobs:
  detect-changes:
    outputs:
      windows: ${{ steps.filter.outputs.windows }}
      release-preflight: ${{ steps.filter.outputs.release-preflight }}
    steps:
      - id: filter
        with:
          filters: |
            rust:
              - 'crates/**/*.rs'
            windows: []
            release-preflight:
              - 'Cargo.toml'
      - id: matrix
        run: echo 'json=["ubuntu-24.04", "macos-14"]'
  lint:
    needs: [detect-changes, fmt]
  test:
    needs: [detect-changes, fmt, lint]
    steps:
      - name: Build Windows release binaries
        if: matrix.os == 'windows-2022'
        run: cargo build --release
      - name: Native ORT smoke (Windows; release profile)
        if: matrix.os == 'windows-2022'
        run: smoke
      - name: Workspace lib tests (macOS, single-threaded)
        if: matrix.os == 'macos-14'
        run: cargo nextest run --workspace --lib --test-threads=1
      - name: Compile platform-owned MCP runtime
        if: matrix.os == 'macos-14' || matrix.os == 'windows-2022'
        run: cargo check -p wenlan-mcp --all-targets
  conclusion:
    steps:
      - name: Aggregate
        run: |
          case "$result" in
            success|skipped) ;;
          esac
"#;
    let platform_sensitive_paths = vec![
        ("crates/new_windows.rs".to_string(), "windows", "windows"),
        ("crates/new_macos.rs".to_string(), "macos", "macos"),
        ("scripts/new-windows.ps1".to_string(), "windows", "windows"),
    ];
    let release_profile_sensitive_paths = vec!["crates/release_only.rs".to_string()];
    let violations = ci_routing_contract_violations(
        workflow,
        "",
        &platform_sensitive_paths,
        &release_profile_sensitive_paths,
    );
    for expected in [
        "platform-sensitive windows",
        "platform-sensitive macos",
        "bootstrap the CI contract",
        "expected-vs-actual",
        "unnecessarily serialized",
        "dev/test artifact reuse",
        "cache writes",
        "reusable target cache",
        "restore-only target-free rust-cache",
        "sccache mode is not read-only",
        "required Rust job",
        "coverage workflow",
        "release-please workflow",
        "clippy configuration",
        "non-Rust test fixtures",
        "nextest config",
        "release-profile-sensitive",
        "native/build-sensitive",
        "45-minute non-Windows PR budget",
        "release-sensitive PRs and release backstops",
        "rust-lld",
        "debug runtime artifacts",
        "differentially compile every wenlan-mcp target",
        "mcp-platform routing",
        "CLI/server platform proof",
        "ordinary Windows contract",
        "fail fast before integration",
        "fail fast from integration",
        "root installer",
        "changed-file inventory as JSON",
        "repository catch-all",
        "test the impact planner",
        "every non-reused CI run",
        "derive its test plan",
        "rust-ci-required output",
        "non-Rust fast-owner",
        "unknown paths",
        "plugin fast-owner",
        "npm fast-owner",
        "validated impacted-test plan",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

// ── Teeth #9: release preflight preserves event-bounded shipped-target proof ──

fn release_preflight_contract_violations(ci_workflow: &str, release_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let release: serde_yaml::Value =
        serde_yaml::from_str(release_workflow).expect("parse release.yml");
    let mut violations = Vec::new();
    let preflight = &ci["jobs"]["release-preflight"];
    if job_needs(&ci, "release-preflight") != ["detect-changes"]
        || preflight["if"].as_str().is_none_or(|condition| {
            !condition.contains("github.event_name != 'pull_request'")
                || !condition.contains("startsWith(github.head_ref, 'release-please--branches--')")
                || !condition.contains("needs.detect-changes.outputs.release-preflight == 'true'")
        })
    {
        violations.push(
            "release-preflight is not isolated to release-sensitive PRs and backstops".into(),
        );
    }
    if preflight["strategy"]["fail-fast"].as_bool() != Some(true)
        || preflight["strategy"]["matrix"].as_str()
            != Some("${{ fromJSON(needs.detect-changes.outputs.release-preflight-targets) }}")
    {
        violations.push("release-preflight is not a fail-fast canonical target matrix".into());
    }
    let matrix_plan = job_step(
        &ci,
        "detect-changes",
        "Emit event-bounded release preflight matrix",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    for marker in [
        "scripts/release_targets.py preflight-matrix",
        "--event-name \"$GITHUB_EVENT_NAME\"",
        "--ref \"$GITHUB_REF\"",
        "--head-ref \"$GITHUB_HEAD_REF\"",
        "--github-output \"$GITHUB_OUTPUT\"",
    ] {
        if !matrix_plan.contains(marker) {
            violations.push(format!(
                "release-preflight event matrix omits fail-closed planner input {marker:?}"
            ));
        }
    }
    let inventory_test = job_step(&ci, "detect-changes", "Test release target inventory")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !inventory_test.contains("bash scripts/release_preflight_matrix.test.sh") {
        violations.push(
            "release-preflight event matrix is not exercised through its shell contract".into(),
        );
    }
    let build = job_step(
        &ci,
        "release-preflight",
        "Build and smoke shipped release binaries",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if build != "bash scripts/build-release-binaries.sh \"${{ matrix.target }}\"" {
        violations
            .push("release-preflight does not build and smoke the canonical archives once".into());
    }
    let cache = job_step_using(&ci, "release-preflight", "Swatinem/rust-cache");
    if cache.and_then(|step| step["with"]["workspaces"].as_str()) != Some(". -> target") {
        violations.push("release-preflight cache has overlapping cache roots".into());
    }
    for step in preflight["steps"].as_sequence().into_iter().flatten() {
        let name = step["name"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let run = step["run"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let allowed_candidate_step = matches!(
            name.as_str(),
            "package canonical release candidate archives"
                | "smoke exact release candidate archives"
                | "write untrusted release candidate claim manifest"
                | "upload immutable release candidate artifact"
        );
        if name.contains("publish")
            || run.contains("gh release")
            || run.contains("npm publish")
            || (name.contains("package") && !allowed_candidate_step)
        {
            violations
                .push("release-preflight contains an unauthorized publishing side effect".into());
        }
    }
    if release["on"].get("workflow_dispatch").is_some() {
        violations.push("tag release retains an unbound manual dispatch path".into());
    }
    if release["jobs"].get("release").is_some()
        || release_workflow.contains("cargo build")
        || release_workflow.contains("build-release-binaries")
    {
        violations
            .push("tag release retains duplicate compilation of PR-validated binaries".into());
    }
    if job_needs(&release, "prepare-release") != ["resolve-promotion"]
        || job_needs(&release, "promote-assets") != ["resolve-promotion", "prepare-release"]
    {
        violations.push("tag release artifact-promotion DAG bypasses receipt resolution".into());
    }
    let download = job_step(
        &release,
        "promote-assets",
        "Download exact validated wrapper once",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !download.contains("scripts/release-promotion.py download-assets") {
        violations.push("tag release does not download the exact validated asset wrapper".into());
    }
    if !job_needs(&ci, "conclusion")
        .iter()
        .any(|need| need == "release-preflight")
    {
        violations.push("conclusion.needs omits release-preflight".into());
    }
    violations
}

#[test]
fn release_preflight_is_release_gated_and_non_publishing() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let violations = release_preflight_contract_violations(&ci, &release);
    assert!(
        violations.is_empty(),
        "release-preflight contract drift — ordinary PR/main matrices must remain bounded, \
         release-please/manual matrices must mirror every shipped target, and no lane may publish \
         beyond immutable candidate data. \
         Fix .github/workflows/ci.yml and release.yml; an intentional contract change also \
         updates release_preflight_contract_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_preflight_contract_rejects_drift_and_side_effects() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let ci = ci
        .replace(
            "      - name: Reject truncated PR file inventory",
            "      - name: Accept truncated PR file inventory",
        )
        .replace(
            "startsWith(github.head_ref, 'release-please--branches--')",
            "false",
        )
        .replace("      fail-fast: true", "      fail-fast: false")
        .replace(
            "          save-if: ${{ github.ref == 'refs/heads/main' }}",
            "          save-if: \"true\"",
        )
        .replace(
            "          shared-key: release-v3-${{ matrix.target }}",
            "          shared-key: release-v3-ci-only-${{ matrix.target }}",
        )
        .replace(
            "        run: bash scripts/build-release-binaries.sh \"${{ matrix.target }}\"",
            "        run: gh release create unsafe",
        )
        .replace("preflight-matrix", "matrix")
        .replace(
            "          bash scripts/release_preflight_matrix.test.sh",
            "          echo release preflight matrix shell contract removed",
        )
        .replace(
            "          expect_job release-preflight '${{ github.event_name != 'pull_request' || startsWith(github.head_ref, 'release-please--branches--') || needs.detect-changes.outputs.release-preflight == 'true' }}' '${{ needs.release-preflight.result }}'",
            "          echo release-preflight skipped",
        )
        .replace(
            "      - name: Native ORT smoke (Windows release preflight)",
            "      - name: Native ORT smoke removed",
        )
        .replace(
            "      - name: Stage Windows release runtimes before smoke",
            "      - name: Stage Windows release runtimes removed",
        )
        .replace(
            "      - name: Select native Perl for vendored OpenSSL",
            "      - name: Native Perl removed",
        )
        .replace(
            "      - name: Mark Windows explicit target as nested Cargo cache",
            "      - name: Nested Cargo target marker removed",
        )
        .replace(
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=",
            "REMOVED_WINDOWS_LINKER=",
        );
    let release = release
        .replace(
            "      matrix: ${{ fromJSON(needs.prepare-release.outputs.release-targets) }}",
            "      matrix: {}",
        )
        .replace(
            "      - name: Require main workflow for manual release",
            "      - name: Manual release ref guard removed",
        )
        .replace(
            "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=",
            "REMOVED_WINDOWS_LINKER=",
        )
        .replace(
            "    timeout-minutes: ${{ matrix.target == 'x86_64-pc-windows-msvc' && 90 || 60 }}",
            "    timeout-minutes: 900",
        )
        .replace("    needs: prepare-release", "    needs: release")
        .replace(
            "    needs: [docker, release, publish-crates, publish-npm, update-homebrew]",
            "    needs: [docker, release]",
        )
        .replace("    needs: docker-manifest", "    needs: release")
        .replace(
            "          save-if: \"false\"",
            "          save-if: \"true\"",
        );
    let release = release
        .replace("  push:\n", "  workflow_dispatch:\n  push:\n")
        .replace(
            "scripts/release-promotion.py download-assets",
            "scripts/build-release-binaries.sh",
        )
        .replace("    needs: resolve-promotion", "    needs: publish-crates");
    let violations = release_preflight_contract_violations(&ci, &release);
    for expected in [
        "release-sensitive PRs and backstops",
        "fail-fast canonical target matrix",
        "event matrix omits fail-closed planner input",
        "event matrix is not exercised through its shell contract",
        "canonical archives once",
        "unauthorized publishing side effect",
        "unbound manual dispatch",
        "duplicate compilation",
        "artifact-promotion DAG",
        "exact validated asset wrapper",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "mutation must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn release_preflight_contract_rejects_overlapping_cache_roots() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("read ci.yml")
        .replace(
            "          workspaces: . -> target",
            "          workspaces: |\n            . -> target\n            . -> target/${{ matrix.target }}",
        );
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let violations = release_preflight_contract_violations(&ci, &release);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("overlapping cache roots")),
        "mutation must reject overlapping cache roots: {violations:?}"
    );
}

// ── Teeth #10: canonical acceptance runs beside the long workspace-lib lane ──

fn release_candidate_observer_contract_violations(
    ci_workflow: &str,
    observer_workflow: &str,
    validator_script: &str,
    archive_script: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let observer: serde_yaml::Value =
        serde_yaml::from_str(observer_workflow).unwrap_or(serde_yaml::Value::Null);
    let mut violations = Vec::new();
    let workflows = observer["on"]["workflow_run"]["workflows"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let types = observer["on"]["workflow_run"]["types"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let branches = observer["on"]["workflow_run"]["branches"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    if workflows != ["CI"]
        || types != ["completed"]
        || branches != ["release-please--branches--main"]
        || observer["on"].as_mapping().is_none_or(|on| on.len() != 1)
    {
        violations.push(
            "release candidate observer trigger is not exact release-branch completed CI".into(),
        );
    }
    if observer["permissions"]["actions"].as_str() != Some("read")
        || observer["permissions"]["contents"].as_str() != Some("read")
        || observer["permissions"]["pull-requests"].as_str() != Some("read")
        || observer["permissions"]
            .as_mapping()
            .is_none_or(|values| values.len() != 3)
    {
        violations.push("release candidate observer permissions are not exactly read-only".into());
    }
    if observer["jobs"]
        .as_mapping()
        .is_none_or(|jobs| jobs.len() != 1)
        || observer["jobs"]["validate"]["runs-on"].as_str() != Some("ubuntu-24.04")
        || observer["jobs"]["validate"]["timeout-minutes"].as_u64() != Some(15)
    {
        violations
            .push("release candidate observer is not one bounded hosted validation job".into());
    }
    let steps = observer["jobs"]["validate"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let uses = steps
        .iter()
        .filter_map(|step| step["uses"].as_str())
        .collect::<Vec<_>>();
    if steps.len() != 5
        || uses
            != [
                "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
                "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
                "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
            ]
    {
        violations
            .push("release candidate observer must have exactly five closed-receipt steps".into());
    }
    let checkout = steps.first().copied();
    if checkout.and_then(|step| step["uses"].as_str())
        != Some("actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803")
        || checkout
            .and_then(|step| step.as_mapping())
            .is_none_or(|step| step.len() != 2)
        || checkout
            .and_then(|step| step["with"].as_mapping())
            .is_none_or(|with| with.len() != 3)
        || checkout.and_then(|step| step["with"]["ref"].as_str()) != Some("${{ github.sha }}")
        || checkout.and_then(|step| step["with"]["fetch-depth"].as_u64()) != Some(1)
        || checkout.and_then(|step| step["with"]["persist-credentials"].as_bool()) != Some(false)
    {
        violations.push(
            "release candidate observer checkout is not the exact trusted default-branch step"
                .into(),
        );
    }
    let validator = steps.get(1).copied();
    let expected_run = r#"python3 scripts/validate-release-candidate.py \
  --event "$GITHUB_EVENT_PATH" \
  --repository "$GITHUB_REPOSITORY" \
  --temp-root "$RUNNER_TEMP" \
  --summary "$GITHUB_STEP_SUMMARY" \
  --validated-assets-dir "$RUNNER_TEMP/validated-release-assets" \
  --receipt "$RUNNER_TEMP/validated-release-receipt.json""#;
    if validator.and_then(|step| step["name"].as_str())
        != Some("Validate release candidate as untrusted data")
        || validator
            .and_then(|step| step.as_mapping())
            .is_none_or(|step| step.len() != 3)
        || validator
            .and_then(|step| step["env"].as_mapping())
            .is_none_or(|env| env.len() != 1)
        || validator.and_then(|step| step["env"]["GITHUB_TOKEN"].as_str())
            != Some("${{ github.token }}")
        || validator
            .and_then(|step| step["run"].as_str())
            .map(str::trim_end)
            != Some(expected_run)
    {
        violations.push("release candidate observer validator env or command is not exact".into());
    }
    let validated_upload = steps.get(2).copied();
    if validated_upload.and_then(|step| step["name"].as_str())
        != Some("Upload exact validated release assets")
        || validated_upload.and_then(|step| step["id"].as_str()) != Some("validated-assets")
        || validated_upload.and_then(|step| step["uses"].as_str())
            != Some("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a")
        || validated_upload.and_then(|step| step["with"]["name"].as_str())
            != Some(
                "validated-release-assets-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}-${{ github.run_id }}-${{ github.run_attempt }}",
            )
        || validated_upload.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ runner.temp }}/validated-release-assets/*")
        || validated_upload.and_then(|step| step["with"]["compression-level"].as_u64())
            != Some(0)
        || validated_upload.and_then(|step| step["with"]["retention-days"].as_u64())
            != Some(30)
        || validated_upload.and_then(|step| step["with"]["if-no-files-found"].as_str())
            != Some("error")
        || validated_upload.and_then(|step| step["with"]["overwrite"].as_bool())
            != Some(false)
    {
        violations.push("validated assets upload is not immutable and source-run scoped".into());
    }
    let close = steps.get(3).copied();
    let close_run = close
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if close.and_then(|step| step["name"].as_str())
        != Some("Close receipt over validated assets artifact")
        || !close_run.contains("close-receipt")
        || close.and_then(|step| step["env"]["GH_TOKEN"].as_str()) != Some("${{ github.token }}")
        || !close_run.contains("/actions/runs/$GITHUB_RUN_ID")
        || !close_run.contains("steps.validated-assets.outputs.artifact-id")
        || !close_run.contains("steps.validated-assets.outputs.artifact-digest")
        || !close_run.contains("--observer-run-id \"$GITHUB_RUN_ID\"")
        || !close_run.contains("--observer-run-attempt \"$GITHUB_RUN_ATTEMPT\"")
        || !close_run.contains("--observer-workflow-id \"$observer_workflow_id\"")
        || !close_run.contains("--observer-code-sha \"$GITHUB_WORKFLOW_SHA\"")
    {
        violations.push("receipt close step omits validated artifact identity".into());
    }
    let closed_upload = steps.get(4).copied();
    if closed_upload.and_then(|step| step["name"].as_str())
        != Some("Upload closed validation receipt")
        || closed_upload.and_then(|step| step["uses"].as_str())
            != Some("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a")
        || closed_upload.and_then(|step| step["with"]["name"].as_str())
            != Some(
                "validated-release-receipt-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
            )
        || closed_upload.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ runner.temp }}/validated-release-receipt.json")
        || closed_upload.and_then(|step| step["with"]["compression-level"].as_u64())
            != Some(0)
        || closed_upload.and_then(|step| step["with"]["retention-days"].as_u64())
            != Some(30)
        || closed_upload.and_then(|step| step["with"]["if-no-files-found"].as_str())
            != Some("error")
        || closed_upload.and_then(|step| step["with"]["overwrite"].as_bool())
            != Some(true)
    {
        violations.push("closed receipt upload is not a retry-safe source-run locator".into());
    }
    for forbidden in [
        "workflow_dispatch",
        "id-token",
        "attestations",
        "actions/cache",
        "rust-cache",
        "download-artifact",
        "actions/attest",
        "${{ secrets.",
    ] {
        if observer_workflow.contains(forbidden) {
            violations.push(format!(
                "release candidate observer contains forbidden capability {forbidden:?}"
            ));
        }
    }
    if !detect_change_filter_paths(&ci, "rust")
        .contains(".github/workflows/release-candidate-observer.yml")
    {
        violations.push("Rust routing omits the release candidate observer teeth".into());
    }
    let exact_gate = "github.event_name == 'pull_request' && github.event.pull_request.base.ref == 'main' && github.event.pull_request.head.ref == 'release-please--branches--main' && github.event.pull_request.head.repo.full_name == github.repository && github.event.pull_request.head.repo.fork == false && github.event.pull_request.draft == false && github.event.pull_request.user.login == '7xuanlu'";
    for step_name in [
        "Package canonical release candidate archives",
        "Smoke exact release candidate archives",
        "Write untrusted release candidate claim manifest",
        "Upload immutable release candidate artifact",
    ] {
        if job_step(&ci, "release-preflight", step_name).and_then(|step| step["if"].as_str())
            != Some(exact_gate)
        {
            violations.push(format!(
                "release candidate producer step {step_name:?} lacks the exact same-repository gate"
            ));
        }
    }
    let upload = job_step(
        &ci,
        "release-preflight",
        "Upload immutable release candidate artifact",
    );
    if upload.and_then(|step| step["uses"].as_str())
        != Some("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a")
        || upload.and_then(|step| step["with"]["name"].as_str())
            != Some(
                "release-candidate-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.target }}",
            )
        || upload.and_then(|step| step["with"]["path"].as_str()) != Some("dist/*")
        || upload.and_then(|step| step["with"]["compression-level"].as_u64()) != Some(0)
        || upload.and_then(|step| step["with"]["retention-days"].as_u64()) != Some(14)
        || upload.and_then(|step| step["with"]["if-no-files-found"].as_str()) != Some("error")
        || upload.and_then(|step| step["with"]["overwrite"].as_bool()) != Some(false)
    {
        violations
            .push("release candidate producer artifact is not immutable and run-attempt scoped".into());
    }
    let producer_names = ci["jobs"]["release-preflight"]["steps"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(|step| step["name"].as_str())
        .filter(|name| name.contains("release candidate"))
        .collect::<Vec<_>>();
    if producer_names
        != [
            "Package canonical release candidate archives",
            "Smoke exact release candidate archives",
            "Write untrusted release candidate claim manifest",
            "Upload immutable release candidate artifact",
        ]
    {
        violations.push(
            "release candidate producer does not package, smoke, manifest, then upload".into(),
        );
    }
    for required in [
        "CI_WORKFLOW_PATH = \".github/workflows/ci.yml\"",
        "REQUIRED_RELEASE_PATHS = RELEASE_MANAGED_PATHS",
        "class _CredentialStrippingRedirect",
        "redirected.remove_header(\"Authorization\")",
        "/actions/runs/{run_id}",
        "/actions/workflows/{workflow_id}",
        "/commits/{head_sha}/pulls",
        "/actions/runs/{run_id}/artifacts",
        "/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
        "/actions/artifacts/{artifact_id}/zip",
        "payload.get(\"total_count\")",
        "safe_extract_zip(wrapper, extracted, outer_names)",
        "safe_extract_archive(",
        "base_records, head_records = _validate_release_tree_modes(",
        "new != old.replace(old_version, new_version)",
        "_release_version_policy(",
        "Darwin standalone archive bytes differ",
        "merged candidate tree differs",
        "/issues/{pr_number}/events",
        "| Canonical inner asset |",
        "def write_validated_receipt(",
        "def close_receipt(",
        "document[\"receipt_state\"] = \"closed\"",
        "validated_assets_artifact",
        "This receipt is observe-only",
        "def _latest_release_target_attempts(",
        "job.get(\"run_attempt\") != attempt",
        "def _latest_candidate_artifact_attempts(",
        "release-preflight job for {target!r} in artifact attempt {attempt}",
        "def _candidate_artifacts_for_attempts(",
    ] {
        if !validator_script.contains(required) {
            violations.push(format!(
                "release candidate validator omits evidence {required:?}"
            ));
        }
    }
    for required in [
        "def _validate_zip_structure(",
        "compression_method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}",
        "def _decompress_canonical_gzip(",
        "zlib.decompressobj(wbits=31)",
        "expanded_size = _decompress_canonical_gzip(path, raw_path)",
        "decompressor.unused_data",
        "def _validate_tar_termination(",
        "def _validate_raw_tar(",
        "raw_records = _validate_raw_tar(raw_path)",
        "tar extensions and special entries are forbidden",
        "tar.gz has hidden data after its last member",
    ] {
        if !archive_script.contains(required) {
            violations.push(format!(
                "release archive parser omits hostile-input bound {required:?}"
            ));
        }
    }
    for forbidden in [
        "subprocess.",
        ".chmod(",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_OUTPUT",
    ] {
        if validator_script.contains(forbidden) {
            violations.push(format!(
                "release candidate validator may execute or route untrusted bytes via {forbidden:?}"
            ));
        }
    }
    violations
}

#[test]
fn release_candidate_observer_is_read_only_and_observe_only() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/release-candidate-observer.yml"))
            .unwrap_or_default();
    let validator = std::fs::read_to_string(root.join("scripts/validate-release-candidate.py"))
        .unwrap_or_default();
    let archive =
        std::fs::read_to_string(root.join("scripts/release_archive.py")).unwrap_or_default();
    let violations =
        release_candidate_observer_contract_violations(&ci, &observer, &validator, &archive);
    assert!(
        violations.is_empty(),
        "release candidate observe-only contract drift:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_candidate_observer_contract_rejects_privilege_and_identity_drift() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("read ci")
        .replace(
            "github.event.pull_request.head.repo.full_name == github.repository",
            "true",
        )
        .replace("          overwrite: false", "          overwrite: true");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/release-candidate-observer.yml"))
            .expect("read observer")
            .replace("  actions: read", "  actions: write")
        .replace(
            "ref: ${{ github.sha }}",
            "ref: ${{ github.event.workflow_run.head_sha }}",
        )
        .replace("  workflow_run:", "  workflow_dispatch:\n  workflow_run:")
        .replace(
            "    branches: [release-please--branches--main]",
            "    branches: [main]",
        )
        .replace(
            "      - name: Validate release candidate as untrusted data",
            "      - name: Unexpected third step\n        run: echo no\n      - name: Validate release candidate as untrusted data",
        )
        .replace(
            "          GITHUB_TOKEN: ${{ github.token }}",
            "          EXTRA: value\n          GITHUB_TOKEN: ${{ github.token }}",
        )
        .replace(
            "--summary \"$GITHUB_STEP_SUMMARY\"",
            "--summary \"$RUNNER_TEMP/summary\"",
        )
        .replace("          retention-days: 30", "          retention-days: 14")
        .replace(
            "name: validated-release-receipt-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
            "name: validated-release-receipt-latest",
        )
        .replace(
            "--observer-workflow-id \"$observer_workflow_id\"",
            "--observer-workflow-id removed",
        );
    let validator = std::fs::read_to_string(root.join("scripts/validate-release-candidate.py"))
        .expect("read validator")
        .replace(
            "safe_extract_zip(wrapper, extracted, outer_names)",
            "# outer validation removed",
        )
        .replace(
            "base_records, head_records = _validate_release_tree_modes(",
            "base_records, head_records = _release_tree_modes_removed(",
        )
        .replace("new != old.replace(old_version, new_version)", "new == old")
        .replace(
            "| Canonical inner asset |",
            "| Missing inner asset receipt |",
        );
    let archive = std::fs::read_to_string(root.join("scripts/release_archive.py"))
        .expect("read archive")
        .replace(
            "expanded_size = _decompress_canonical_gzip(path, raw_path)",
            "expanded_size = 0 # gzip preflight removed",
        )
        .replace(
            "raw_records = _validate_raw_tar(raw_path)",
            "raw_records = Vec::new() # hostile preflight removed",
        );
    let violations =
        release_candidate_observer_contract_violations(&ci, &observer, &validator, &archive);
    for expected in [
        "trigger is not exact",
        "exactly read-only",
        "exact trusted default-branch",
        "exactly five closed-receipt steps",
        "env or command is not exact",
        "exact same-repository gate",
        "not immutable",
        "omits evidence",
        "hostile-input bound",
    ] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "mutation must exercise {expected:?}: {violations:?}"
        );
    }
}

fn release_promotion_contract_violations(
    ci_workflow: &str,
    release_please_workflow: &str,
    release_please_config: &str,
    fast_maintenance_workflow: &str,
    release_workflow: &str,
    promotion_script: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let release_please: serde_yaml::Value =
        serde_yaml::from_str(release_please_workflow).expect("parse release-please.yml");
    let release_please_config: serde_json::Value =
        serde_json::from_str(release_please_config).unwrap_or(serde_json::Value::Null);
    let fast_maintenance: serde_yaml::Value =
        serde_yaml::from_str(fast_maintenance_workflow).expect("parse release-pr-maintenance.yml");
    let release: serde_yaml::Value =
        serde_yaml::from_str(release_workflow).expect("parse release.yml");
    let mut violations = Vec::new();
    if release_please_config["packages"]["."]["always-update"].as_bool() != Some(true) {
        violations.push("release-please package always-update is not exact true".into());
    }
    let main_gate = job_step(&ci, "detect-changes", "Verify reusable release merge")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    for required in [
        "scripts/release-promotion.py gate-main",
        "--wait-seconds 720",
        "--main-run-id \"$GITHUB_RUN_ID\"",
        "--main-run-attempt \"$GITHUB_RUN_ATTEMPT\"",
        "--plan-output \"$RUNNER_TEMP/main-release-promotion-receipt.json\"",
    ] {
        if !main_gate.contains(required) {
            violations.push(format!("main release gate omits {required:?}"));
        }
    }
    let main_receipt = job_step(
        &ci,
        "detect-changes",
        "Upload main release promotion receipt",
    );
    if main_receipt.and_then(|step| step["if"].as_str())
        != Some("steps.release-proof.outputs.release-gate-state == 'validated'")
        || main_receipt.and_then(|step| step["with"]["retention-days"].as_u64()) != Some(30)
        || main_receipt.and_then(|step| step["with"]["overwrite"].as_bool()) != Some(false)
    {
        violations.push(
            "main promotion receipt is not validated-only, immutable, and retained 30 days".into(),
        );
    }
    let trigger_names = release_please["on"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    if trigger_names != BTreeSet::from(["workflow_run"])
        || release_please["jobs"].as_mapping().is_none_or(|jobs| {
            jobs.keys()
                .filter_map(serde_yaml::Value::as_str)
                .collect::<BTreeSet<_>>()
                != BTreeSet::from(["route-main", "maintain-release-pr", "create-validated-tag"])
        })
    {
        violations.push("release-please hybrid trigger or job inventory drifted".into());
    }
    if release_please["concurrency"]["group"].as_str() != Some("release-please-main")
        || release_please["concurrency"]["cancel-in-progress"].as_bool() != Some(false)
    {
        violations.push("release-please does not serialize receipt and tag operations".into());
    }
    if fast_maintenance["concurrency"]["group"].as_str() != Some("release-pr-maintenance-main")
        || fast_maintenance["concurrency"]["cancel-in-progress"].as_bool() != Some(false)
        || fast_maintenance["concurrency"]["queue"].as_str() != Some("max")
        || release_please["jobs"]["maintain-release-pr"]["concurrency"]["group"].as_str()
            != Some("release-pr-maintenance-main")
        || release_please["jobs"]["maintain-release-pr"]["concurrency"]["cancel-in-progress"]
            .as_bool()
            != Some(false)
        || release_please["jobs"]["maintain-release-pr"]["concurrency"]["queue"].as_str()
            != Some("max")
    {
        violations.push("release PR branch writers do not share the fixed maintenance lock".into());
    }
    let fast_trigger_names = fast_maintenance["on"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let fast_branches = fast_maintenance["on"]["push"]["branches"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let fast_jobs = fast_maintenance["jobs"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    if fast_trigger_names != BTreeSet::from(["push"])
        || fast_branches != ["main"]
        || fast_jobs != BTreeSet::from(["maintain-release-pr"])
    {
        violations.push("fast release maintenance is not exact main-push PR-only routing".into());
    }
    let fast_job = &fast_maintenance["jobs"]["maintain-release-pr"];
    let fast_action = job_step_using(
        &fast_maintenance,
        "maintain-release-pr",
        "googleapis/release-please-action",
    );
    let fast_push = job_step(
        &fast_maintenance,
        "maintain-release-pr",
        "Commit synced versions to release PR branch",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    let push_commands = fast_push
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("git push"))
        .collect::<Vec<_>>();
    if fast_job["if"].as_str()
        != Some("github.event_name == 'push' && github.ref == 'refs/heads/main'")
        || fast_job["timeout-minutes"].as_u64() != Some(5)
        || fast_maintenance["permissions"]["contents"].as_str() != Some("read")
        || fast_maintenance["permissions"]["pull-requests"].as_str() != Some("read")
        || fast_job["permissions"]["contents"].as_str() != Some("write")
        || fast_job["permissions"]["pull-requests"].as_str() != Some("write")
        || fast_action.and_then(|step| step["uses"].as_str())
            != Some("googleapis/release-please-action@0dfd8538845b8e92600d271a895a5372865d4062")
        || fast_action.and_then(|step| step["with"]["skip-github-release"].as_bool()) != Some(true)
        || push_commands != ["git push origin HEAD"]
    {
        violations.push(
            "fast release maintenance is not pinned, least-privilege, PR-only, and non-force"
                .into(),
        );
    }
    for forbidden in [
        "workflow_run:",
        "workflow_dispatch:",
        "pull_request:",
        "create-validated-tag",
        "release-promotion.py",
        "refs/tags/",
        "/git/refs",
        "git tag",
        "gh release",
        "autorelease:",
        "/issues/",
        "--method POST",
    ] {
        if fast_maintenance_workflow.contains(forbidden) {
            violations.push(format!(
                "fast release maintenance contains publishing or lifecycle mutation {forbidden:?}"
            ));
        }
    }
    let route = job_step(
        &release_please,
        "route-main",
        "Consume the exact main CI promotion receipt",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !route.contains("scripts/release-promotion.py consume-main-receipt") {
        violations.push("release-please does not consume the exact main receipt".into());
    }
    let route_condition = release_please["jobs"]["route-main"]["if"]
        .as_str()
        .unwrap_or_default();
    let tag_job = &release_please["jobs"]["create-validated-tag"];
    let tag_step = job_step(
        &release_please,
        "create-validated-tag",
        "Require pending release lifecycle and create exact tag",
    );
    if !route_condition.contains("github.event.workflow_run.event == 'push'")
        || !route_condition.contains("github.event.workflow_run.head_branch == 'main'")
        || !route_condition.contains("github.event.workflow_run.conclusion == 'success'")
        || job_needs(&release_please, "create-validated-tag") != ["route-main"]
        || tag_job["if"].as_str() != Some("needs.route-main.outputs.state == 'validated'")
        || tag_step.and_then(|step| step["env"]["MAIN_SHA"].as_str())
            != Some("${{ github.event.workflow_run.head_sha }}")
    {
        violations.push(
            "validated tag is not exclusively bound to the observed successful main receipt".into(),
        );
    }
    if job_step_using(
        &release_please,
        "maintain-release-pr",
        "googleapis/release-please-action",
    )
    .and_then(|step| step["with"]["skip-github-release"].as_bool())
        != Some(true)
    {
        violations.push("ordinary main release-please is not PR-only".into());
    }
    if release["on"].get("workflow_dispatch").is_some()
        || release["jobs"].get("release").is_some()
        || release_workflow.contains("cargo build")
        || release_workflow.contains("build-release-binaries")
    {
        violations.push("tag release has an unbound trigger or duplicate compilation lane".into());
    }
    if release_workflow.contains("cargo publish -p wenlan-types --dry-run") {
        violations.push("tag release duplicates Cargo publish verification".into());
    }
    let crates_text = serde_yaml::to_string(&release["jobs"]["publish-crates"]).unwrap_or_default();
    for required in [
        "CARGO_REGISTRY_TOKEN is required because wenlan-types",
        "CARGO_REGISTRY_TOKEN is required because wenlan-mcp",
        "Require wenlan-mcp on crates.io",
        "wenlan-mcp ${VERSION} not visible on sparse index after 10 min",
    ] {
        if !crates_text.contains(required) {
            violations.push(format!(
                "crates.io publication omits fail-closed proof {required:?}"
            ));
        }
    }
    if crates_text.contains("if: env.CARGO_REGISTRY_TOKEN != ''") {
        violations.push("crates.io publication can silently skip a missing credential".into());
    }
    let mut checkout_count = 0;
    for job in release["jobs"].as_mapping().into_iter().flatten() {
        for step in job
            .1
            .get("steps")
            .and_then(serde_yaml::Value::as_sequence)
            .into_iter()
            .flatten()
        {
            if step["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with("actions/checkout@"))
            {
                checkout_count += 1;
                if step["with"]["ref"].as_str() != Some("${{ github.sha }}") {
                    violations.push(
                        "tag release checkout is not pinned to the immutable event SHA".into(),
                    );
                }
            }
        }
    }
    if checkout_count == 0 || release_workflow.contains("ref: refs/tags/${{ env.RELEASE_TAG }}") {
        violations.push("tag release retains a mutable tag-ref checkout".into());
    }
    for job_name in [
        "prepare-release",
        "resolve-promotion",
        "promote-assets",
        "docker",
        "docker-manifest",
        "publish-crates",
        "publish-npm",
        "update-homebrew",
        "finalize-release",
    ] {
        let job_text = serde_yaml::to_string(&release["jobs"][job_name]).unwrap_or_default();
        if !job_text.contains("/git/ref/tags/$RELEASE_TAG") || !job_text.contains("GITHUB_SHA") {
            violations.push(format!(
                "publication job {job_name:?} does not revalidate the immutable event tag"
            ));
        }
    }
    let prepare_text =
        serde_yaml::to_string(&release["jobs"]["prepare-release"]).unwrap_or_default();
    if !prepare_text
        .contains("Existing stable release cannot enter an incremental publication rerun.")
        || !prepare_text.contains("isPrerelease")
    {
        violations.push("existing stable release can enter incremental publication".into());
    }
    let resolve = job_step(
        &release,
        "resolve-promotion",
        "Resolve tag to the trusted observer receipt",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !resolve.contains("scripts/release-promotion.py consume-main-receipt")
        || job_needs(&release, "prepare-release") != ["resolve-promotion"]
        || job_needs(&release, "promote-assets") != ["resolve-promotion", "prepare-release"]
    {
        violations
            .push("tag release does not resolve the main receipt before asset promotion".into());
    }
    let plan_upload = job_step(&release, "resolve-promotion", "Upload small promotion plan");
    let plan_download = job_step_using(&release, "promote-assets", "actions/download-artifact");
    if plan_upload.and_then(|step| step["with"]["name"].as_str())
        != Some("release-promotion-plan-${{ github.run_id }}")
        || plan_upload.and_then(|step| step["with"]["overwrite"].as_bool()) != Some(true)
        || plan_download.and_then(|step| step["with"]["name"].as_str())
            != Some("release-promotion-plan-${{ github.run_id }}")
    {
        violations.push(
            "tag promotion plan is not a retry-safe same-run locator with fresh receipt validation"
                .into(),
        );
    }
    for (job, step_name, artifact_name) in [
        (
            "promote-assets",
            "Publish exact Homebrew inputs for this run",
            "homebrew-artifacts",
        ),
        (
            "promote-assets",
            "Publish exact Docker runtime inputs for this run",
            "docker-runtime-inputs",
        ),
        (
            "docker",
            "Publish immutable image digest receipt",
            "docker-image-digest-${{ matrix.tag-suffix }}",
        ),
    ] {
        let upload = job_step(&release, job, step_name);
        if upload.and_then(|step| step["with"]["name"].as_str()) != Some(artifact_name)
            || upload.and_then(|step| step["with"]["overwrite"].as_bool()) != Some(true)
        {
            violations.push(format!(
                "retryable internal artifact {artifact_name:?} is not run-scoped and overwrite-safe"
            ));
        }
    }
    let docker = &release["jobs"]["docker"];
    let docker_text = serde_yaml::to_string(docker).unwrap_or_default();
    if job_needs(&release, "docker") != ["promote-assets"]
        || !docker_text.contains("docker/Dockerfile.release-runtime")
        || !docker_text.contains("scripts/verify-release-runtime-image.py")
        || docker_text.contains("docker/Dockerfile.daemon")
    {
        violations.push(
            "Docker release does not reuse and smoke the exact validated Linux binary".into(),
        );
    }
    for required in [
        "MAX_RECEIPT_CANDIDATES = 20",
        "latest trusted observer attempt",
        "observer reruns produced conflicting release semantics",
        "main-release-promotion-receipt-{run_id}-",
        "/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
        "main promotion receipt claims a future run attempt",
        "main promotion receipt reruns produced conflicting semantics",
        "subparsers.add_parser(\"consume-main-receipt\")",
        "subparsers.add_parser(\"download-assets\")",
        "safe_extract_zip(wrapper, output_dir, expected)",
    ] {
        if !promotion_script.contains(required) {
            violations.push(format!("promotion resolver omits {required:?}"));
        }
    }
    violations
}

#[test]
fn release_promotion_reuses_exact_archives_and_fails_closed() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci");
    let release_please = std::fs::read_to_string(root.join(".github/workflows/release-please.yml"))
        .expect("read release-please");
    let release_please_config = std::fs::read_to_string(root.join("release-please-config.json"))
        .expect("read release-please config");
    let fast_maintenance =
        std::fs::read_to_string(root.join(".github/workflows/release-pr-maintenance.yml"))
            .expect("read fast release maintenance");
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let promotion = std::fs::read_to_string(root.join("scripts/release-promotion.py"))
        .expect("read promotion resolver");
    let violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &release,
        &promotion,
    );
    assert!(
        violations.is_empty(),
        "hybrid release promotion contract drift:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_promotion_contract_rejects_rebuild_and_unbounded_receipts() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci");
    let release_please = std::fs::read_to_string(root.join(".github/workflows/release-please.yml"))
        .expect("read release-please");
    let release_please_config = std::fs::read_to_string(root.join("release-please-config.json"))
        .expect("read release-please config");
    let fast_maintenance =
        std::fs::read_to_string(root.join(".github/workflows/release-pr-maintenance.yml"))
            .expect("read fast release maintenance");
    let release = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release")
        .replace(
            "scripts/release-promotion.py download-assets",
            "cargo build --release",
        );
    let promotion = std::fs::read_to_string(root.join("scripts/release-promotion.py"))
        .expect("read promotion resolver")
        .replace(
            "MAX_RECEIPT_CANDIDATES = 20",
            "MAX_RECEIPT_CANDIDATES = 1000",
        );
    let violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &release,
        &promotion,
    );
    for expected in ["duplicate compilation", "MAX_RECEIPT_CANDIDATES"] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "mutation must exercise {expected:?}: {violations:?}"
        );
    }
    for (label, mutated_config) in [
        (
            "false",
            release_please_config.replace("\"always-update\": true", "\"always-update\": false"),
        ),
        (
            "missing",
            release_please_config.replace("\"always-update\"", "\"always-update-missing\""),
        ),
    ] {
        let config_violations = release_promotion_contract_violations(
            &ci,
            &release_please,
            &mutated_config,
            &fast_maintenance,
            &release,
            &promotion,
        );
        assert!(
            config_violations
                .iter()
                .any(|item| item.contains("always-update is not exact true")),
            "{label} always-update mutation must fail closed: {config_violations:?}"
        );
    }
    let duplicate_publish_verification =
        std::fs::read_to_string(root.join(".github/workflows/release.yml"))
            .expect("read release")
            .replace(
                "cargo publish -p wenlan-types",
                "cargo publish -p wenlan-types --dry-run",
            );
    let duplicate_publish_violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &duplicate_publish_verification,
        &std::fs::read_to_string(root.join("scripts/release-promotion.py"))
            .expect("read promotion resolver"),
    );
    assert!(
        duplicate_publish_violations
            .iter()
            .any(|item| item.contains("duplicates Cargo publish verification")),
        "mutation must reject duplicate Cargo publish verification: {duplicate_publish_violations:?}"
    );
    let retry_unsafe_release = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release")
        .replace(
            "name: release-promotion-plan-${{ github.run_id }}",
            "name: release-promotion-plan-${{ github.run_id }}-${{ github.run_attempt }}",
        );
    let retry_violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &retry_unsafe_release,
        &std::fs::read_to_string(root.join("scripts/release-promotion.py"))
            .expect("read promotion resolver"),
    );
    assert!(
        retry_violations
            .iter()
            .any(|item| item.contains("retry-safe")),
        "mutation must exercise retry-safe tag promotion: {retry_violations:?}"
    );
    let mutable_tag_release = retry_unsafe_release.replace(
        "ref: ${{ github.sha }}",
        "ref: refs/tags/${{ env.RELEASE_TAG }}",
    );
    let mutable_tag_violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &mutable_tag_release,
        &std::fs::read_to_string(root.join("scripts/release-promotion.py"))
            .expect("read promotion resolver"),
    );
    assert!(
        mutable_tag_violations
            .iter()
            .any(|item| item.contains("mutable tag")),
        "mutation must reject mutable tag checkout: {mutable_tag_violations:?}"
    );
    let stable_incremental_release = retry_unsafe_release.replace(
        "Existing stable release cannot enter an incremental publication rerun.",
        "Continuing existing stable release.",
    );
    let stable_violations = release_promotion_contract_violations(
        &ci,
        &release_please,
        &release_please_config,
        &fast_maintenance,
        &stable_incremental_release,
        &std::fs::read_to_string(root.join("scripts/release-promotion.py"))
            .expect("read promotion resolver"),
    );
    assert!(
        stable_violations
            .iter()
            .any(|item| item.contains("stable release")),
        "mutation must reject incremental stable release: {stable_violations:?}"
    );

    let unsafe_fast_maintenance = fast_maintenance
        .replace("skip-github-release: true", "skip-github-release: false")
        .replace(
            "group: release-pr-maintenance-main",
            "group: release-please-main",
        )
        .replace("queue: max", "queue: single")
        .replace("run: bash scripts/bump-version.sh", "run: gh api /git/refs")
        .replace("git push origin HEAD", "git push --force origin HEAD");
    let unsafe_release_please = release_please
        .replace(
            "github.event.workflow_run.conclusion == 'success'",
            "github.event.workflow_run.conclusion != 'success'",
        )
        .replace(
            "MAIN_SHA: ${{ github.event.workflow_run.head_sha }}",
            "MAIN_SHA: ${{ github.sha }}",
        );
    let routing_violations = release_promotion_contract_violations(
        &ci,
        &unsafe_release_please,
        &release_please_config,
        &unsafe_fast_maintenance,
        &std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release"),
        &std::fs::read_to_string(root.join("scripts/release-promotion.py"))
            .expect("read promotion resolver"),
    );
    for expected in [
        "fixed maintenance lock",
        "pinned, least-privilege, PR-only, and non-force",
        "publishing or lifecycle mutation",
        "observed successful main receipt",
    ] {
        assert!(
            routing_violations
                .iter()
                .any(|item| item.contains(expected)),
            "mutation must exercise {expected:?}: {routing_violations:?}"
        );
    }
}

fn windows_native_parallelism_violations(
    ci_workflow: &str,
    _release_workflow: &str,
    msvc_setup: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let mut violations = Vec::new();
    let step = job_step(
        &ci,
        "release-preflight",
        "Bound native build concurrency (Windows)",
    );
    let run = step
        .and_then(|candidate| candidate["run"].as_str())
        .unwrap_or_default();
    if step.and_then(|candidate| candidate["if"].as_str())
        != Some("matrix.target == 'x86_64-pc-windows-msvc'")
        || !run.contains("CARGO_BUILD_JOBS=2")
        || run.contains("CARGO_BUILD_JOBS=1")
    {
        violations
            .push("release preflight does not cap outer Windows Cargo at exactly two jobs".into());
    }
    if !msvc_setup.contains("$env:CMAKE_BUILD_PARALLEL_LEVEL = \"1\"")
        || msvc_setup.contains("$env:CMAKE_BUILD_PARALLEL_LEVEL = \"2\"")
        || msvc_setup.contains("CARGO_BUILD_JOBS")
    {
        violations.push("MSVC Ninja setup does not keep nested CMake serialized".into());
    }
    violations
}

#[test]
fn windows_cargo_uses_two_jobs_while_nested_cmake_stays_serial() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let setup = std::fs::read_to_string(root.join("scripts/setup-msvc-ninja-windows.ps1"))
        .expect("read Windows MSVC Ninja setup");
    let violations = windows_native_parallelism_violations(&ci, &release, &setup);
    assert!(
        violations.is_empty(),
        "Windows native parallelism drift — Cargo may run exactly two outer jobs while nested \
         CMake remains serialized at one worker. Fix the CI/release caps or the MSVC Ninja setup; \
         an intentional change also updates this guard and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn windows_native_parallelism_rejects_reversed_limits() {
    let ci = r#"
jobs:
  release-preflight:
    steps:
      - name: Bound native build concurrency (Windows)
        if: matrix.target == 'x86_64-pc-windows-msvc'
        run: CARGO_BUILD_JOBS=1
      - name: Build and smoke shipped release binaries
        run: build
"#;
    let release = r#"
jobs:
  release:
    steps:
      - name: Build and smoke shipped release binaries
        run: build
      - name: Cap Windows Cargo parallelism
        if: matrix.target == 'x86_64-pc-windows-msvc'
        run: CARGO_BUILD_JOBS=1
"#;
    let setup = "$env:CMAKE_BUILD_PARALLEL_LEVEL = \"2\"\nCARGO_BUILD_JOBS=2";
    let violations = windows_native_parallelism_violations(ci, release, setup);
    for expected in ["exactly two jobs", "nested CMake"] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

fn canonical_acceptance_contract_violations(ci_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let mut violations = Vec::new();
    let job = &ci["jobs"]["canonical-acceptance"];

    if job_needs(&ci, "canonical-acceptance") != ["detect-changes"] {
        violations.push("Canonical acceptance is serialized behind another required job".into());
    }
    if job["if"].as_str()
        != Some(
            "needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.canonical-acceptance-required == 'true'",
        )
    {
        violations.push(
            "Canonical acceptance does not use the planner's canonical aggregate gate".into(),
        );
    }
    if job["runs-on"].as_str() != Some("ubuntu-24.04")
        || job["timeout-minutes"].as_str()
            != Some("${{ github.event_name == 'pull_request' && 30 || 60 }}")
    {
        violations.push("Canonical acceptance is not bounded on the canonical Linux runner".into());
    }
    for (name, expected) in [
        ("CARGO_PROFILE_DEV_DEBUG", "0"),
        ("CARGO_PROFILE_TEST_DEBUG", "0"),
        ("SCCACHE_GHA_RW_MODE", "READ_ONLY"),
        (
            "FASTEMBED_CACHE_DIR",
            "${{ github.workspace }}/.fastembed_cache",
        ),
    ] {
        if job["env"][name].as_str() != Some(expected) {
            violations.push(format!(
                "Canonical acceptance env {name} is not pinned to {expected}"
            ));
        }
    }

    for step_name in [
        "Page lint scale gate (Linux time + RSS)",
        "Upload Page lint scale receipt (Linux)",
        "Integration tests wenlan-cli + wenlan-server (Linux)",
        "Run integration tests (core) (Linux)",
        "E2E wenlan background on/off (Linux user-systemd)",
        "E2E folder ingest over HTTP (Linux)",
    ] {
        if job_step(&ci, "canonical-acceptance", step_name).is_none() {
            violations.push(format!(
                "Canonical acceptance does not own required step {step_name}"
            ));
        }
        if job_step(&ci, "test", step_name).is_some() {
            violations.push(format!(
                "required step {step_name} remains serialized in the workspace-lib matrix"
            ));
        }
    }
    for (step_name, expected_run, expected_condition, violation) in [
        (
            "Page lint scale gate (Linux time + RSS)",
            r#"bash scripts/lint-scale-gate.sh "$RUNNER_TEMP/task-19-memory-lint-debugger-linux.txt""#,
            "needs.detect-changes.outputs.canonical-smokes-required == 'true'",
            "Canonical acceptance page lint command is not executable",
        ),
        (
            "Integration tests wenlan-cli + wenlan-server (Linux)",
            "python3 scripts/ci_test_plan.py run --suite cli-server-integration --plan-json \"$CI_TEST_PLAN\"",
            "needs.detect-changes.outputs.cli-server-integration-required == 'true'",
            "Canonical acceptance CLI/server integration command is not executable",
        ),
        (
            "E2E folder ingest over HTTP (Linux)",
            "bash scripts/smoke-folder-ingest.sh",
            "needs.detect-changes.outputs.canonical-smokes-required == 'true'",
            "Canonical acceptance folder ingest smoke is not planner-gated",
        ),
    ] {
        let step = job_step(&ci, "canonical-acceptance", step_name);
        if step.and_then(|step| step["run"].as_str()) != Some(expected_run)
            || step.and_then(|step| step["if"].as_str()) != Some(expected_condition)
        {
            violations.push(violation.into());
        }
    }
    let core_integration = job_step(
        &ci,
        "canonical-acceptance",
        "Run integration tests (core) (Linux)",
    );
    for step_name in [
        "Integration tests wenlan-cli + wenlan-server (Linux)",
        "Run integration tests (core) (Linux)",
    ] {
        if job_step(&ci, "canonical-acceptance", step_name)
            .and_then(|step| step["env"]["CI_TEST_PLAN"].as_str())
            != Some("${{ needs.detect-changes.outputs.test-plan }}")
        {
            violations.push(format!(
                "Canonical acceptance {step_name} does not consume the validated test plan"
            ));
        }
    }
    if core_integration.and_then(|step| step["if"].as_str())
        != Some("needs.detect-changes.outputs.core-integration-required == 'true'")
    {
        violations.push("Linux core integration coverage is not planner-gated".into());
    }
    let systemd = job_step(
        &ci,
        "canonical-acceptance",
        "E2E wenlan background on/off (Linux user-systemd)",
    );
    let systemd_run = systemd
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    let systemd_active_lines = systemd_run
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<BTreeSet<_>>();
    let manager_probe_offset = systemd_run.find(
        "run_bounded manager-probe 20s bash -c 'systemctl --user show-environment >/dev/null'",
    );
    let build_offset = systemd_run.find("run_bounded build 12m cargo build");
    let trap_offset = systemd_run.find("trap cleanup EXIT");
    let start_attempt_offset = systemd_run.find("start_attempted=true");
    let start_offset = systemd_run.find("run_bounded start 45s");
    if systemd.and_then(|step| step["if"].as_str())
        != Some("needs.detect-changes.outputs.canonical-smokes-required == 'true'")
        || systemd_run.contains("loginctl enable-linger")
        || systemd_run.matches(r#"return "$status""#).count() < 2
        || !matches!((manager_probe_offset, build_offset), (Some(probe), Some(build)) if probe < build)
        || !matches!(
            (trap_offset, start_attempt_offset, start_offset),
            (Some(trap), Some(attempt), Some(start)) if trap < attempt && attempt < start
        )
        || [
            r#"if timeout --signal=TERM --kill-after=10s "$limit" "$@"; then"#,
            r#"if timeout --signal=TERM --kill-after=5s 20s "$@"; then"#,
            "return \"$status\"",
            "run_bounded manager-probe 20s bash -c 'systemctl --user show-environment >/dev/null'",
            "run_bounded build 12m cargo build -p wenlan -p wenlan-server",
            "trap cleanup EXIT",
            "start_attempted=false",
            "start_attempted=true",
            "run_bounded start 45s \"$STAGE/wenlan\" background on",
            "run_bounded probe-enabled-after-start 20s systemctl --user is-enabled wenlan-server",
            "if ! run_bounded health 90s bash -c '",
            r#"until curl --connect-timeout 1 --max-time 2 -sf \"#,
            "run_bounded stop 45s \"$STAGE/wenlan\" background off",
            "run_bounded probe-enabled-after-stop 20s systemctl --user is-enabled wenlan-server",
            r#"active_state="$(timeout --signal=TERM --kill-after=5s 20s \"#,
            "systemctl --user show wenlan-server --property=ActiveState --value)\"",
            "test \"$active_state\" = \"inactive\"",
            r#"if ! cleanup_bounded cleanup-stop "$STAGE/wenlan" background off; then"#,
            r#"if [ "$start_attempted" = true ] && [ -x "$STAGE/wenlan" ]; then"#,
            "if ! cleanup_bounded cleanup-disable systemctl --user disable --now wenlan-server.service; then",
            r#"if ! cleanup_bounded cleanup-unit-remove bash -c 'rm -f -- "$1"' _ "$UNIT_PATH"; then"#,
            "if ! cleanup_bounded cleanup-reload systemctl --user daemon-reload; then",
            "local cleanup_status=0",
            "cleanup_status=1",
            "receipt cleanup fail",
            "trap - EXIT",
            "if [ \"$main_status\" -ne 0 ]; then",
            "exit \"$main_status\"",
            "exit \"$cleanup_status\"",
        ]
        .iter()
        .any(|command| !systemd_active_lines.contains(command))
    {
        violations
            .push("Linux systemd acceptance command lost a bounded lifecycle assertion".into());
    }
    let lint_upload = job_step(
        &ci,
        "canonical-acceptance",
        "Upload Page lint scale receipt (Linux)",
    );
    let lint_preserve = job_step(
        &ci,
        "canonical-acceptance",
        "Preserve Page lint diagnostic on prerequisite failure",
    );
    let preserve_run = lint_preserve
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if lint_preserve.and_then(|step| step["if"].as_str())
        != Some("always() && needs.detect-changes.outputs.canonical-smokes-required == 'true'")
        || !preserve_run.contains("if [[ ! -f \"$receipt\" ]]")
        || !preserve_run.contains("${{ steps.page-lint-scale.outcome }}")
        || lint_upload.and_then(|step| step["if"].as_str())
            != Some("always() && needs.detect-changes.outputs.canonical-smokes-required == 'true'")
        || lint_upload
            .and_then(|step| step["uses"].as_str())
            .is_none_or(|uses| !uses.starts_with("actions/upload-artifact@"))
        || lint_upload.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ runner.temp }}/task-19-memory-lint-debugger-linux.txt")
    {
        violations.push("Linux page lint receipt is not always preserved".into());
    }
    let macos_integration = job_step(
        &ci,
        "test",
        "Integration tests wenlan-cli + wenlan-server (macOS)",
    );
    if macos_integration.and_then(|step| step["if"].as_str())
        != Some("matrix.os == 'macos-14' && needs.detect-changes.outputs.platform-cli-server-integration-required == 'true'")
    {
        violations.push("macOS lost its shared CLI/server integration owner".into());
    }
    for step_name in [
        "E2E CLI surface smoke (Linux)",
        "E2E MCP stdio surface smoke (Linux)",
    ] {
        if job_step(&ci, "canonical-acceptance", step_name).and_then(|step| step["if"].as_str())
            != Some(
                "needs.detect-changes.outputs.canonical-smokes-required == 'true' && github.event_name != 'pull_request'",
            )
        {
            violations.push(format!(
                "Canonical acceptance surface smoke {step_name} is not backstop-only and planner-gated"
            ));
        }
    }

    let acceptance_steps = job["steps"]
        .as_sequence()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let rust_caches = acceptance_steps
        .iter()
        .filter(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
        })
        .collect::<Vec<_>>();
    let rust_cache = rust_caches.first().copied();
    if rust_caches.len() != 1
        || rust_cache.and_then(|step| step["with"]["shared-key"].as_str()) != Some("test")
        || rust_cache.and_then(|step| step["with"]["cache-targets"].as_str()) != Some("false")
        || rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some("false")
    {
        violations
            .push("Canonical acceptance needs exactly one restore-only rust-cache action".into());
    }
    let sccache_actions = acceptance_steps
        .iter()
        .filter(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("sccache-action"))
        })
        .count();
    if sccache_actions != 1 {
        violations.push("Canonical acceptance needs exactly one read-only sccache action".into());
    }
    let fastembed_downloads = acceptance_steps
        .iter()
        .filter(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with("actions/download-artifact@"))
        })
        .collect::<Vec<_>>();
    let fastembed = fastembed_downloads.first().copied();
    if fastembed_downloads.len() != 1
        || fastembed.and_then(|step| step["with"]["path"].as_str()) != Some(".fastembed_cache")
        || fastembed.and_then(|step| step["with"]["name"].as_str())
            != Some("fastembed-bge-base-en-v1.5-q-v3-portable-${{ github.run_id }}")
    {
        violations.push("Canonical acceptance FastEmbed artifact is not run-scoped".into());
    }
    if acceptance_steps
        .iter()
        .filter_map(|step| step["uses"].as_str())
        .any(|uses| uses.starts_with("actions/cache"))
    {
        violations.push("Canonical acceptance contains a FastEmbed cache action".into());
    }

    if !job_needs(&ci, "conclusion")
        .iter()
        .any(|need| need == "canonical-acceptance")
    {
        violations.push("conclusion.needs omits canonical acceptance".into());
    }
    let conclusion = workflow_step_run(&ci, "Aggregate expected CI results").unwrap_or_default();
    if !conclusion.lines().map(str::trim).any(|line| {
        line.starts_with("expect_job canonical-acceptance ")
            && line.contains("\"$run_canonical\"")
            && line.contains("needs.canonical-acceptance.result")
    }) {
        violations.push("conclusion does not fail closed on canonical acceptance".into());
    }

    violations
}

#[test]
fn canonical_acceptance_is_parallel_required_and_artifact_only() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let violations = canonical_acceptance_contract_violations(&ci);
    assert!(
        violations.is_empty(),
        "Canonical acceptance critical-path contract drift — canonical acceptance stays \
         parallel, required, and artifact-only beside the long workspace-lib lane. Fix \
         .github/workflows/ci.yml; an intentional contract change also updates \
         canonical_acceptance_contract_violations() and its positive controls:\n{}",
        violations.join("\n")
    );
}

#[test]
fn canonical_acceptance_contract_rejects_serialized_or_optional_fixture() {
    let ci = r#"
jobs:
  test:
    if: run-rust
    steps:
      - name: E2E folder ingest over HTTP (Linux)
        run: old
  canonical-acceptance:
    needs: [test]
    if: other
    runs-on: windows-2022
    timeout-minutes: 90
    env:
      SCCACHE_GHA_RW_MODE: READ_WRITE
    steps:
      - uses: Swatinem/rust-cache@v2
        with:
          save-if: "true"
      - uses: actions/cache@v4
  conclusion:
    needs: [test]
    steps:
      - name: Aggregate expected CI results
        run: echo success
"#;
    let violations = canonical_acceptance_contract_violations(ci);
    for expected in [
        "serialized",
        "canonical aggregate gate",
        "canonical Linux runner",
        "env",
        "does not own required step",
        "remains serialized",
        "macOS lost",
        "exactly one restore-only rust-cache",
        "FastEmbed artifact is not run-scoped",
        "FastEmbed cache action",
        "conclusion.needs",
        "conclusion does not fail closed",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn canonical_acceptance_contract_rejects_semantic_noops_and_secondary_writers() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci = ci
        .replace(
            "      SCCACHE_GHA_RW_MODE: READ_ONLY",
            "      SCCACHE_GHA_RW_MODE: READ_WRITE",
        )
        .replace(
            "        run: python3 scripts/ci_test_plan.py run --suite cli-server-integration --plan-json \"$CI_TEST_PLAN\"",
            "        run: \"true\"",
        )
        .replace(
            "        if: needs.detect-changes.outputs.canonical-smokes-required == 'true'\n        run: bash scripts/smoke-folder-ingest.sh",
            "        if: \"false\"\n        run: bash scripts/smoke-folder-ingest.sh",
        )
        .replace(
            "          run_bounded stop 45s \"$STAGE/wenlan\" background off",
            "          \"$STAGE/wenlan\" background off",
        )
        .replace(
            "      - name: Install cargo-nextest",
            "      - uses: Swatinem/rust-cache@v2\n        with:\n          save-if: \"true\"\n      - name: Install cargo-nextest",
        )
        .replace(
            "          expect_job canonical-acceptance \"$run_canonical\" '${{ needs.canonical-acceptance.result }}'",
            "          # expect_job canonical-acceptance \"$run_canonical\" '${{ needs.canonical-acceptance.result }}'",
        );
    let violations = canonical_acceptance_contract_violations(&ci);
    for expected in [
        "SCCACHE_GHA_RW_MODE",
        "CLI/server integration command",
        "folder ingest smoke",
        "systemd acceptance command",
        "exactly one restore-only rust-cache",
        "conclusion does not fail closed",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "semantic mutation must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn canonical_acceptance_contract_rejects_missing_trap_or_fail_open_cleanup() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    for (mutation, expected) in [
        (
            ci.replace(
                "          trap cleanup EXIT",
                "          # trap cleanup EXIT",
            ),
            "missing EXIT trap",
        ),
        (
            ci.replace(
                "            exit \"$cleanup_status\"",
                "            exit 0 # cleanup failure ignored",
            ),
            "fail-open cleanup",
        ),
    ] {
        let violations = canonical_acceptance_contract_violations(&mutation);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("systemd acceptance command")),
            "fixture must reject {expected}: {violations:?}"
        );
    }
}

// ── Teeth #11: every normal core integration target has a required owner ──

fn core_integration_contract_violations(ci_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let Some(step) = job_step(
        &ci,
        "canonical-acceptance",
        "Run integration tests (core) (Linux)",
    ) else {
        return vec!["required Linux core integration step is missing".into()];
    };

    let expected = "python3 scripts/ci_test_plan.py run --suite core-integration --plan-json \"$CI_TEST_PLAN\"";
    let mut violations = Vec::new();
    if step["run"].as_str() != Some(expected)
        || step["env"]["CI_TEST_PLAN"].as_str()
            != Some("${{ needs.detect-changes.outputs.test-plan }}")
        || step["if"].as_str()
            != Some("needs.detect-changes.outputs.core-integration-required == 'true'")
    {
        violations.push(
            "required Linux core integration step does not execute the validated fail-closed planner"
                .into(),
        );
    }
    violations
}

#[test]
fn every_core_integration_target_has_a_required_or_manual_owner() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let violations = core_integration_contract_violations(&ci);
    assert!(
        violations.is_empty(),
        "core integration planner wiring drift — every normal core integration target keeps a \
         required (or explicitly manual) CI owner; no direct-command bypass, no dead text \
         coverage. Fix .github/workflows/ci.yml; an intentional contract change also updates \
         core_integration_contract_violations() and its positive controls:\n{}",
        violations.join("\n")
    );
}

#[test]
fn core_integration_inventory_rejects_direct_command_bypass() {
    let ci = r#"
jobs:
  canonical-acceptance:
    steps:
      - name: Run integration tests (core) (Linux)
        run: cargo nextest run -p wenlan-core --features eval-harness --test known
"#;
    let violations = core_integration_contract_violations(ci);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("fail-closed planner")),
        "a hand-maintained target list must not bypass the fail-closed planner: {violations:?}"
    );
}

#[test]
fn core_integration_inventory_rejects_dead_text_coverage() {
    let ci = r#"
jobs:
  canonical-acceptance:
    steps:
      - name: Run integration tests (core) (Linux)
        run: |
          # cargo nextest run -p wenlan-core --features eval-harness --test commented_out
          echo --test echoed_only
          cargo nextest run -p wenlan-core --features eval-harness --test actually_run # --test inline_commented
"#;
    let violations = core_integration_contract_violations(ci);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("fail-closed planner")),
        "dead command text must not satisfy required integration ownership: {violations:?}"
    );
}

#[test]
fn core_integration_inventory_rejects_shell_operator_dead_text() {
    for operator in ["|", "&"] {
        let ci = format!(
            r#"
jobs:
  canonical-acceptance:
    steps:
      - name: Run integration tests (core) (Linux)
        env:
          CI_TEST_PLAN: ${{{{ needs.detect-changes.outputs.test-plan }}}}
        run: python3 scripts/ci_test_plan.py run --suite core-integration --plan-json "$CI_TEST_PLAN" {operator} true
"#
        );
        let violations = core_integration_contract_violations(&ci);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("fail-closed planner")),
            "shell operator {operator:?} must not weaken the exact planner invocation: {violations:?}"
        );
    }
}

// ── Teeth #12: the main eval canary stays off the required CI path ──

fn main_canary_contract_violations(ci_workflow: &str, canary_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let canary: serde_yaml::Value =
        serde_yaml::from_str(canary_workflow).unwrap_or(serde_yaml::Value::Null);
    let mut violations = Vec::new();

    let mut required_jobs = BTreeSet::new();
    let mut pending_jobs = vec!["conclusion".to_string()];
    while let Some(job_name) = pending_jobs.pop() {
        if required_jobs.insert(job_name.clone()) {
            pending_jobs.extend(job_needs(&ci, &job_name));
        }
    }
    for job_name in &required_jobs {
        for step in ci["jobs"][job_name]["steps"]
            .as_sequence()
            .into_iter()
            .flatten()
        {
            let run = step["run"].as_str().unwrap_or_default();
            if run.contains("eval::retrieval") && run.contains("--run-ignored=only") {
                violations.push(format!(
                    "required CI test critical path contains the embedding eval in job {job_name}"
                ));
            }
        }
    }
    for step_name in [
        "Run embedding-only eval (main only, Linux)",
        "Upload eval canary baseline (with env schema)",
    ] {
        if job_step(&ci, "test", step_name).is_some() {
            violations.push(format!(
                "{step_name} still extends the required CI test critical path"
            ));
        }
    }
    if job_needs(&ci, "conclusion")
        .iter()
        .any(|job| job == "main-canary")
    {
        violations.push("conclusion.needs includes the non-blocking main canary".into());
    }
    if !detect_change_filter_paths(&ci, "rust").contains(".github/workflows/main-canary.yml") {
        violations.push("Rust routing omits the main canary workflow contract".into());
    }

    let push_branches = canary["on"]["push"]["branches"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    if push_branches != ["main"] {
        violations.push(format!(
            "main canary push trigger is not limited to main: {push_branches:?}"
        ));
    }
    if canary["on"]["push"].get("paths").is_some()
        || canary["on"]["push"].get("paths-ignore").is_some()
    {
        violations.push(
            "main canary filters main pushes by path instead of proving every accepted push".into(),
        );
    }
    if !canary["on"]["workflow_dispatch"].is_null() {
        // A null mapping value is how `workflow_dispatch:` is represented.
    } else if canary["on"].get("workflow_dispatch").is_none() {
        violations.push("main canary has no manual workflow_dispatch trigger".into());
    }
    if canary["on"].get("pull_request").is_some() {
        violations.push("main canary runs on pull requests".into());
    }
    if !canary["concurrency"].is_null() {
        violations.push(
            "main canary uses concurrency that can discard an accepted main push proof".into(),
        );
    }

    let job = &canary["jobs"]["main-canary"];
    if job["runs-on"].as_str() != Some("ubuntu-24.04") {
        violations.push("main canary does not run on the canonical Linux platform".into());
    }
    if job["timeout-minutes"].as_u64() != Some(60) {
        violations.push("main canary does not retain a 60-minute cold-cache budget".into());
    }
    if !job_needs(&canary, "main-canary").is_empty() {
        violations.push("main canary is not an independent job".into());
    }
    if job["env"]["SCCACHE_GHA_RW_MODE"].as_str() != Some("READ_ONLY") {
        violations.push("main canary sccache is not read-only".into());
    }
    if job["env"]["FASTEMBED_CACHE_DIR"].as_str()
        != Some("${{ github.workspace }}/.fastembed_cache")
    {
        violations.push("main canary does not pin the FastEmbed cache directory".into());
    }

    let canary_steps = job["steps"].as_sequence().into_iter().flatten();
    let rust_cache_steps = canary_steps
        .clone()
        .filter(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
        })
        .collect::<Vec<_>>();
    if rust_cache_steps.len() != 1
        || rust_cache_steps.iter().any(|step| {
            step["with"]["shared-key"].as_str() != Some("test")
                || step["with"]["cache-targets"].as_str() != Some("false")
                || step["with"]["save-if"].as_str() != Some("false")
        })
    {
        violations.push("main canary rust-cache is not restore-only".into());
    }
    if rust_cache_steps
        .iter()
        .any(|step| step["with"]["save-if"].as_str() != Some("false"))
    {
        violations.push("main canary contains a rust-cache writer".into());
    }
    if canary_steps.clone().any(|step| {
        step["env"]["SCCACHE_GHA_RW_MODE"]
            .as_str()
            .is_some_and(|mode| mode != "READ_ONLY")
            || step["run"].as_str().is_some_and(|run| {
                run.contains("SCCACHE_GHA_RW_MODE") && run.contains("READ_WRITE")
            })
    }) {
        violations.push("main canary step overrides sccache read-only mode".into());
    }
    let fastembed_restore = job_step_using(&canary, "main-canary", "actions/cache/restore");
    if fastembed_restore
        .and_then(|step| step["uses"].as_str())
        .is_none_or(|uses| !uses.contains("actions/cache/restore@"))
        || fastembed_restore.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ env.FASTEMBED_CACHE_DIR }}")
        || fastembed_restore.and_then(|step| step["with"]["key"].as_str())
            != Some("fastembed-bge-base-en-v1.5-q-v3-portable")
        || fastembed_restore.and_then(|step| step["with"]["enableCrossOsArchive"].as_str())
            != Some("true")
    {
        violations.push(
            "main canary FastEmbed cache is not the restore-only cross-OS portable v3 cache".into(),
        );
    }
    if canary_steps
        .clone()
        .filter_map(|step| step["uses"].as_str())
        .any(|uses| uses.starts_with("actions/cache@") || uses.contains("actions/cache/save@"))
    {
        violations.push("main canary contains a FastEmbed cache writer".into());
    }

    let eval = job_step(&canary, "main-canary", "Run embedding-only eval");
    if eval.and_then(|step| step["run"].as_str())
        != Some("cargo nextest run -p wenlan-core --lib --run-ignored=only eval::retrieval")
        || eval.and_then(|step| step["env"]["EVAL_BASELINES_DIR"].as_str())
            != Some("${{ runner.temp }}/origin-eval-canary")
    {
        violations.push("main canary does not run the exact embedding-only eval contract".into());
    }
    let upload = job_step(&canary, "main-canary", "Upload eval canary baseline");
    if upload
        .and_then(|step| step["uses"].as_str())
        .is_none_or(|uses| !uses.contains("actions/upload-artifact@"))
        || upload.and_then(|step| step["if"].as_str()) != Some("always()")
        || upload.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ runner.temp }}/origin-eval-canary/*.json")
        || upload.and_then(|step| step["with"]["if-no-files-found"].as_str()) != Some("warn")
    {
        violations
            .push("main canary does not preserve its always-uploaded baseline receipt".into());
    }

    violations
}

#[test]
fn main_canary_is_independent_and_read_only() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let canary =
        std::fs::read_to_string(root.join(".github/workflows/main-canary.yml")).unwrap_or_default();
    let violations = main_canary_contract_violations(&ci, &canary);
    assert!(
        violations.is_empty(),
        "main canary contract drift — the main eval canary stays an independent, read-only \
         job off the required CI path. Fix .github/workflows/main-canary.yml (and its ci.yml \
         references); an intentional contract change also updates \
         main_canary_contract_violations() and its positive controls:\n{}",
        violations.join("\n")
    );
}

#[test]
fn main_canary_contract_rejects_embedded_or_writing_fixture() {
    let ci = r#"
jobs:
  detect-changes:
    steps:
      - id: filter
        with:
          filters: |
            rust:
              - 'crates/**/*.rs'
  test:
    steps:
      - name: Run embedding-only eval (main only, Linux)
        run: cargo nextest run
      - name: Upload eval canary baseline (with env schema)
        run: upload
  conclusion:
    needs: [test, main-canary]
"#;
    let canary = r#"
on:
  pull_request:
  push:
    branches: [feature]
    paths-ignore: [docs/**]
concurrency:
  group: main-canary
  cancel-in-progress: true
jobs:
  main-canary:
    needs: test
    runs-on: windows-2022
    timeout-minutes: 15
    env:
      SCCACHE_GHA_RW_MODE: READ_WRITE
      FASTEMBED_CACHE_DIR: /tmp/other
    steps:
      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: canary
          cache-targets: "true"
          save-if: "true"
      - uses: actions/cache@v4
        with:
          path: /tmp/other
          key: mutable
      - name: Run embedding-only eval
        run: cargo test
"#;
    let violations = main_canary_contract_violations(ci, canary);
    for expected in [
        "required CI test critical path",
        "conclusion.needs",
        "Rust routing",
        "limited to main",
        "filters main pushes by path",
        "manual workflow_dispatch",
        "pull requests",
        "concurrency",
        "canonical Linux",
        "60-minute cold-cache budget",
        "independent job",
        "sccache is not read-only",
        "FastEmbed cache directory",
        "rust-cache is not restore-only",
        "portable v3 cache",
        "FastEmbed cache writer",
        "exact embedding-only eval",
        "always-uploaded baseline receipt",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn main_canary_contract_rejects_semantic_reinsertion_and_secondary_writers() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci = ci.replace(
        "      - name: E2E wenlan background on/off (Linux user-systemd)",
        r#"      - name: Renamed retrieval regression suite
        if: matrix.os == 'ubuntu-24.04'
        run: cargo nextest run -p wenlan-core --lib --run-ignored=only eval::retrieval
      - name: E2E wenlan background on/off (Linux user-systemd)"#,
    );
    let canary = std::fs::read_to_string(root.join(".github/workflows/main-canary.yml"))
        .expect("read main-canary.yml");
    let canary = canary
        .replace(
            "      - name: Install cargo-nextest",
            r#"      - uses: Swatinem/rust-cache@v2
        with:
          shared-key: hidden-writer
          cache-targets: "true"
          save-if: "true"
      - name: Install cargo-nextest"#,
        )
        .replace(
            "      - name: Run embedding-only eval\n        env:\n",
            "      - name: Run embedding-only eval\n        env:\n          SCCACHE_GHA_RW_MODE: READ_WRITE\n",
        );
    let violations = main_canary_contract_violations(&ci, &canary);
    for expected in [
        "required CI test critical path",
        "rust-cache writer",
        "step overrides sccache read-only mode",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn main_canary_contract_rejects_eval_inside_conclusion_itself() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let ci = ci.replace(
        "      - name: Aggregate expected CI results",
        r#"      - name: Renamed canary inside the required summary
        run: cargo nextest run -p wenlan-core --lib --run-ignored=only eval::retrieval
      - name: Aggregate expected CI results"#,
    );
    let canary = std::fs::read_to_string(root.join(".github/workflows/main-canary.yml"))
        .expect("read main-canary.yml");
    let violations = main_canary_contract_violations(&ci, &canary);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("required CI test critical path")),
        "fixture must reject an eval step inside conclusion itself: {violations:?}"
    );
}

// ── Teeth #13: CI measurements stay read-only and off the required path ──

fn ci_observer_contract_violations(
    ci_workflow: &str,
    observer_workflow: &str,
    observer_script: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let observer: serde_yaml::Value =
        serde_yaml::from_str(observer_workflow).unwrap_or(serde_yaml::Value::Null);
    let mut violations = Vec::new();

    let workflows = observer["on"]["workflow_run"]["workflows"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    let types = observer["on"]["workflow_run"]["types"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    if workflows != ["CI"] || types != ["completed"] {
        violations.push("CI observer is not triggered only after completed CI runs".into());
    }
    if observer["on"].as_mapping().is_none_or(|on| on.len() != 1) {
        violations.push("CI observer has triggers beyond workflow_run".into());
    }
    if observer["concurrency"]["group"].as_str()
        != Some(
            "ci-observer-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
        )
        || observer["concurrency"]["cancel-in-progress"].as_bool() != Some(false)
    {
        violations.push("CI observer does not preserve every run-attempt receipt".into());
    }
    if observer["permissions"]["actions"].as_str() != Some("read")
        || observer["permissions"]["contents"].as_str() != Some("read")
        || observer["permissions"]
            .as_mapping()
            .is_none_or(|permissions| permissions.len() != 2)
    {
        violations
            .push("CI observer does not have exact read-only Actions/content permissions".into());
    }
    if observer_workflow.contains("${{ secrets.") {
        violations.push("CI observer reads repository secrets".into());
    }
    if required_job_closure(&ci)
        .iter()
        .any(|job| job.contains("observer"))
        || required_jobs_contain(&ci, ".github/workflows/ci-observer.yml")
        || required_jobs_contain(&ci, "scripts/ci-observer.py")
    {
        violations.push("required CI closure depends on the out-of-band CI observer".into());
    }
    if !detect_change_filter_paths(&ci, "rust").contains(".github/workflows/ci-observer.yml") {
        violations.push("Rust routing omits the CI observer contract".into());
    }
    let required_script_step = job_step(
        &ci,
        "detect-changes",
        "Verify CI observer and ORT contracts",
    );
    let required_script_lines = required_script_step
        .and_then(|step| step["run"].as_str())
        .into_iter()
        .flat_map(str::lines)
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    let expected_script_lines = [
        "python3 scripts/ci-cache-maintenance.test.py",
        "python3 scripts/ci-observer.test.py",
        "python3 scripts/ci-timed-command.test.py",
        "python3 scripts/verify-ort-source-pin.test.py",
        "python3 scripts/verify-ort-source-pin.py",
    ];
    if !required_job_closure(&ci).contains("detect-changes")
        || ci["jobs"]["detect-changes"]
            .get("continue-on-error")
            .is_some()
        || required_script_step.is_some_and(|step| step.get("if").is_some())
        || required_script_step.is_some_and(|step| step.get("continue-on-error").is_some())
        || required_script_lines != expected_script_lines
    {
        violations.push(
            "required Linux CI measurement contracts are not in the exact executable test step"
                .into(),
        );
    }

    let observer_jobs = observer["jobs"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _job)| name.as_str())
        .collect::<Vec<_>>();
    if observer_jobs != ["collect"] {
        violations.push("CI observer contains jobs beyond the single bounded collector".into());
    }
    let job = &observer["jobs"]["collect"];
    if job["runs-on"].as_str() != Some("ubuntu-24.04") || job["timeout-minutes"].as_u64() != Some(5)
    {
        violations.push("CI observer is not bounded to the canonical hosted runner".into());
    }
    if job.get("environment").is_some() || job.get("permissions").is_some() {
        violations.push("CI observer adds environment or job-level permissions".into());
    }
    if observer["env"]["CI_CACHE_BUDGET_GB"].as_str() != Some("10") {
        violations
            .push("CI observer does not use the current 10 GB repository cache policy".into());
    }
    if observer["env"]["CI_REQUIRED_GATE_TARGET_MINUTES"].as_str() != Some("20") {
        violations.push("CI observer does not encode the 20-minute required gate target".into());
    }
    let steps = job["steps"].as_sequence().into_iter().flatten();
    let checkouts = steps
        .clone()
        .filter(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("actions/checkout@"))
        })
        .collect::<Vec<_>>();
    let checkout = checkouts.first().copied();
    if checkouts.len() != 1 {
        violations.push("CI observer does not contain exactly one trusted checkout".into());
    }
    if checkout.and_then(|step| step["with"]["ref"].as_str()) != Some("${{ github.sha }}")
        || checkout.and_then(|step| step["with"]["persist-credentials"].as_bool()) != Some(false)
    {
        violations.push(
            "CI observer does not checkout trusted default-branch code without credentials".into(),
        );
    }
    if checkout
        .and_then(|step| step["with"]["ref"].as_str())
        .is_some_and(|reference| reference.contains("head_sha"))
    {
        violations.push("CI observer executes code from the measured untrusted head SHA".into());
    }
    let uses = steps
        .clone()
        .filter_map(|step| step["uses"].as_str())
        .collect::<Vec<_>>();
    if uses.iter().any(|action| {
        action.contains("actions/cache")
            || action.contains("rust-cache")
            || action.contains("sccache-action")
            || action.contains("download-artifact")
    }) {
        violations.push("CI observer restores untrusted artifacts or build caches".into());
    }
    let action_pin = regex::Regex::new(r"^[^@]+@[0-9a-f]{40}$").unwrap();
    if uses.iter().any(|action| !action_pin.is_match(action)) {
        violations.push("CI observer uses an action without an immutable SHA pin".into());
    }
    let allowed_actions = [
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f",
    ];
    if uses.len() != allowed_actions.len()
        || uses.iter().any(|action| !allowed_actions.contains(action))
    {
        violations.push("CI observer uses an action beyond checkout and receipt upload".into());
    }
    let run = steps
        .clone()
        .filter_map(|step| step["run"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    for forbidden in [
        "cargo ",
        "npm ",
        "git ",
        "eval ",
        "source ",
        "/logs",
        "/artifacts",
        "cache delete",
        "--method POST",
        "--method PUT",
        "--method PATCH",
        "--method DELETE",
        "${{ github.event.workflow_run.",
    ] {
        if run.contains(forbidden) {
            violations.push(format!(
                "CI observer can execute or mutate untrusted state through {forbidden:?}"
            ));
        }
    }
    for required in [
        "/actions/runs/$RUN_ID/attempts/$RUN_ATTEMPT/jobs?per_page=100",
        "/actions/cache/usage",
        "--method GET",
        "--paginate --slurp",
        "X-GitHub-Api-Version: 2026-03-10",
        "scripts/ci-observer.py",
        "--event \"$GITHUB_EVENT_PATH\"",
        "--cache-budget-gb \"$CI_CACHE_BUDGET_GB\"",
        "--required-gate-target-minutes \"$CI_REQUIRED_GATE_TARGET_MINUTES\"",
    ] {
        if !run.contains(required) {
            violations.push(format!("CI observer omits required evidence {required:?}"));
        }
    }
    for forbidden in [
        "/actions/cache/storage-limit",
        "cache-limit.json",
        "--cache-limit",
    ] {
        if run.contains(forbidden) || observer_script.contains(forbidden) {
            violations.push(format!(
                "CI observer still depends on removed cache-limit evidence {forbidden:?}"
            ));
        }
    }
    if run.matches("X-GitHub-Api-Version: 2026-03-10").count() != 2 {
        violations.push(
            "CI observer does not pin both GitHub metadata reads to API version 2026-03-10".into(),
        );
    }
    let receipt_builder = job_step(&observer, "collect", "Build timing and cache receipt");
    if receipt_builder.and_then(|step| step["if"].as_str()) != Some("always()") {
        violations.push("CI observer receipt builder does not run after metadata failures".into());
    }
    let upload = steps.clone().find(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|uses| uses.contains("actions/upload-artifact@"))
    });
    if upload.and_then(|step| step["if"].as_str()) != Some("always()")
        || upload.and_then(|step| step["with"]["path"].as_str())
            != Some("${{ runner.temp }}/ci-observer/receipt.json")
        || upload.and_then(|step| step["with"]["if-no-files-found"].as_str()) != Some("error")
        || upload.and_then(|step| step["with"]["retention-days"].as_u64()) != Some(30)
    {
        violations.push(
            "CI observer receipt is not always uploaded from runner.temp with missing files fatal"
                .into(),
        );
    }

    for required in [
        "CACHE_BUDGET_SOURCE = \"repository_policy\"",
        "parser.add_argument(\"--cache-budget-gb\", required=True)",
        "parser.add_argument(\"--required-gate-target-minutes\", required=True)",
        "\"schema_version\": 2",
        "\"required_gate_target\"",
        "\"scope\": \"ordinary_pull_request\"",
        "\"applicable\": ordinary_pr",
        "SLO_EXEMPT_JOB_PREFIXES = (\"release-preflight (\",)",
        "\"CI required gate target exceeded\"",
        "\"budget_gb\": budget_gb",
        "\"budget_source\": CACHE_BUDGET_SOURCE",
        "def warning(title, message):",
        "::warning title={title}",
        "write_receipt(args.output, error_receipt(args, error))",
    ] {
        if !observer_script.contains(required) {
            violations.push(format!(
                "CI observer schema-v2 warning receipt omits {required:?}"
            ));
        }
    }
    if observer_script.contains("\"schema_version\": 1")
        || observer_script.contains("raise SystemExit(1)")
        || observer_script.contains("raise SystemExit(2)")
        || observer_script.matches("raise SystemExit(0)").count() < 2
    {
        violations
            .push("CI observer receipt can gate CI or regress from warning-only schema v2".into());
    }
    let invalid_write =
        observer_script.find("write_receipt(args.output, error_receipt(args, error))");
    let invalid_warning = observer_script.find("warning(\"CI observer invalid input\"");
    let over_budget_warning = observer_script.find("\"CI cache budget exceeded\"");
    if !matches!((invalid_write, invalid_warning), (Some(write), Some(warning)) if write < warning)
        || over_budget_warning.is_none()
    {
        violations.push(
            "CI observer does not persist invalid/over-budget receipts before non-gating warnings"
                .into(),
        );
    }

    violations
}

#[test]
fn ci_observer_is_out_of_band_read_only_and_never_executes_measured_code() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/ci-observer.yml")).unwrap_or_default();
    let script = std::fs::read_to_string(root.join("scripts/ci-observer.py")).unwrap_or_default();
    let violations = ci_observer_contract_violations(&ci, &observer, &script);
    assert!(
        violations.is_empty(),
        "CI observer contract drift — the observer stays out-of-band, read-only, and never \
         executes the code it measures. Fix .github/workflows/ci-observer.yml; an intentional \
         contract change also updates ci_observer_contract_violations() and its positive \
         controls:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ci_observer_contract_rejects_privilege_and_untrusted_inputs() {
    let ci = "jobs:\n  conclusion:\n    needs: [ci-observer]\n";
    let observer = r#"
on:
  pull_request:
  workflow_run:
    workflows: [Other]
    types: [requested]
permissions:
  actions: write
  contents: write
jobs:
  collect:
    runs-on: self-hosted
    timeout-minutes: 60
    environment: production
    env:
      TOKEN: ${{ secrets.RELEASE_TOKEN }}
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
          persist-credentials: true
      - uses: Swatinem/rust-cache@v2
      - uses: actions/download-artifact@v4
      - run: cargo test && gh api --method DELETE /actions/caches && gh api /logs
      - uses: actions/upload-artifact@v4
        with:
          path: target/
  extra:
    permissions:
      contents: write
    steps:
      - run: echo extra
"#;
    let violations = ci_observer_contract_violations(ci, observer, "");
    for expected in [
        "completed CI runs",
        "triggers beyond",
        "read-only Actions/content permissions",
        "reads repository secrets",
        "required CI closure",
        "Rust routing",
        "measurement contract",
        "single bounded collector",
        "canonical hosted runner",
        "environment or job-level permissions",
        "trusted default-branch code",
        "untrusted head SHA",
        "untrusted artifacts or build caches",
        "immutable SHA pin",
        "beyond checkout and receipt upload",
        "execute or mutate untrusted state",
        "required evidence",
        "receipt builder",
        "missing files fatal",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "observer fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn ci_observer_contract_rejects_short_circuited_or_optional_measurement_tests() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/ci-observer.yml")).unwrap_or_default();
    let script = std::fs::read_to_string(root.join("scripts/ci-observer.py")).unwrap_or_default();
    let ci = ci
        .replace(
            "      - name: Verify CI observer and ORT contracts\n        run: |",
            "      - name: Verify CI observer and ORT contracts\n        if: \"false\"\n        run: |\n          exit 0",
        );
    let violations = ci_observer_contract_violations(&ci, &observer, &script);
    assert!(
        violations
            .iter()
            .any(|violation| { violation.contains("exact executable test step") }),
        "optional or short-circuited measurement tests must fail: {violations:?}"
    );
}

#[test]
fn ci_observer_contract_rejects_non_blocking_measurement_tests() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/ci-observer.yml")).unwrap_or_default();
    let script = std::fs::read_to_string(root.join("scripts/ci-observer.py")).unwrap_or_default();
    let ci = ci.replace(
        "      - name: Verify CI observer and ORT contracts\n        run: |",
        "      - name: Verify CI observer and ORT contracts\n        continue-on-error: true\n        run: |",
    );
    let violations = ci_observer_contract_violations(&ci, &observer, &script);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("exact executable test step")),
        "non-blocking measurement tests must fail: {violations:?}"
    );
}

// ── Teeth #14: hosted optimization experiments stay manual and restore-only ──

#[test]
fn ci_observer_schema_v2_contract_rejects_gating_limit_metadata() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let observer =
        std::fs::read_to_string(root.join(".github/workflows/ci-observer.yml")).unwrap_or_default();
    let script = std::fs::read_to_string(root.join("scripts/ci-observer.py")).unwrap_or_default();
    let observer = observer
        .replace("CI_CACHE_BUDGET_GB: \"10\"", "CI_CACHE_BUDGET_GB: \"20\"")
        .replace(
            "CI_REQUIRED_GATE_TARGET_MINUTES: \"20\"",
            "CI_REQUIRED_GATE_TARGET_MINUTES: \"30\"",
        )
        .replace("2026-03-10", "2022-11-28")
        .replace("/actions/cache/usage\"", "/actions/cache/storage-limit\"")
        .replace(
            "ci-observer-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
            "ci-observer",
        )
        .replace("cancel-in-progress: false", "cancel-in-progress: true")
        .replace("retention-days: 30", "retention-days: 7");
    let script = script
        .replace("\"schema_version\": 2", "\"schema_version\": 1")
        .replace("raise SystemExit(0)", "raise SystemExit(2)")
        .replace("--cache-budget-gb", "--cache-limit");
    let violations = ci_observer_contract_violations(&ci, &observer, &script);
    for expected in [
        "current 10 GB",
        "20-minute required gate target",
        "preserve every run-attempt receipt",
        "receipt is not always uploaded",
        "API version 2026-03-10",
        "removed cache-limit evidence",
        "schema-v2 warning receipt",
        "warning-only schema v2",
    ] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "mutation must exercise {expected:?}: {violations:?}"
        );
    }
}

fn ci_benchmark_contract_violations(ci_workflow: &str, benchmark_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let benchmark: serde_yaml::Value =
        serde_yaml::from_str(benchmark_workflow).unwrap_or(serde_yaml::Value::Null);
    let mut violations = Vec::new();

    if benchmark["on"].get("workflow_dispatch").is_none()
        || benchmark["on"].as_mapping().is_none_or(|on| on.len() != 1)
    {
        violations.push("CI benchmark is not workflow_dispatch-only".into());
    }
    if benchmark["permissions"]["contents"].as_str() != Some("read")
        || benchmark["permissions"]
            .as_mapping()
            .is_none_or(|permissions| permissions.len() != 1)
    {
        violations.push("CI benchmark does not have exact read-only contents permission".into());
    }
    if benchmark_workflow.contains("${{ secrets.") {
        violations.push("CI benchmark reads repository secrets".into());
    }
    if required_job_closure(&ci)
        .iter()
        .any(|job| job.contains("benchmark"))
        || required_jobs_contain(&ci, "ci-benchmark")
        || required_jobs_contain(&ci, "ci-timed-command.py")
    {
        violations.push("required CI closure depends on a benchmark job".into());
    }
    if !detect_change_filter_paths(&ci, "rust").contains(".github/workflows/ci-benchmark.yml") {
        violations.push("Rust routing omits the CI benchmark contract".into());
    }

    for required in [
        "p4-runners",
        "p5-test-engine",
        "p5-windows-drive",
        "p6-release",
        "ubuntu-24.04",
        "ubuntu-latest",
        "macos-14",
        "macos-15",
        "macos-26",
        "macos-latest",
        "windows-2022",
        "windows-2025",
        "windows-latest",
        "cargo nextest run --workspace --lib",
        "cargo test --workspace --lib --no-fail-fast",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_LTO",
        "scripts/ci-timed-command.py",
    ] {
        if !benchmark_workflow.contains(required) {
            violations.push(format!(
                "CI benchmark omits experiment control {required:?}"
            ));
        }
    }
    for (job_name, step_name) in [
        ("p4-runners", "Configure rust-lld (Windows)"),
        ("p5-windows-drive", "Configure rust-lld"),
        ("p6-release", "Configure rust-lld (Windows)"),
    ] {
        let linker = job_step(&benchmark, job_name, step_name)
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        if !linker.contains("rust-lld.exe")
            || !linker.contains("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=")
            || !linker.contains("RUSTFLAGS=")
            || !linker.contains("$env:GITHUB_ENV")
        {
            violations.push(format!(
                "benchmark job {job_name} does not configure rust-lld for target and host build artifacts"
            ));
        }
    }
    if benchmark["jobs"]["p5-test-engine"]["env"]["SCCACHE_GHA_RW_MODE"].as_str()
        != Some("READ_ONLY")
    {
        violations.push("P5 benchmark does not enforce read-only sccache".into());
    }
    let p6_entries = benchmark["jobs"]["p6-release"]["strategy"]["matrix"]["include"]
        .as_sequence()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if p6_entries.is_empty()
        || p6_entries.iter().any(|entry| {
            entry.get("cache").is_some()
                || !matches!(
                    entry["cache_mode"].as_str(),
                    Some("cold" | "production-restore")
                )
        })
        || !p6_entries
            .iter()
            .any(|entry| entry["cache_mode"].as_str() == Some("cold"))
        || !p6_entries
            .iter()
            .any(|entry| entry["cache_mode"].as_str() == Some("production-restore"))
    {
        violations.push(
            "P6 cache_mode must use both exact cold and production-restore vocabulary".into(),
        );
    }
    let p6_cache_receipt = job_step(&benchmark, "p6-release", "Record release cache evidence")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    let active_receipt_lines = p6_cache_receipt
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();
    let expected_cache_logic = [
        r#"requested_mode = "${{ matrix.cache_mode }}""#,
        r#"profile = "${{ matrix.profile }}""#,
        "effective_cache = (",
        r#""profile-invalidated-cold""#,
        r#"if requested_mode == "production-restore" and profile != "current""#,
        "else requested_mode",
        ")",
    ];
    let has_exact_cache_logic = active_receipt_lines
        .windows(expected_cache_logic.len())
        .any(|window| window == expected_cache_logic);
    let effective_cache_assignments = active_receipt_lines
        .iter()
        .filter(|line| line.starts_with("effective_cache ="))
        .count();
    if !has_exact_cache_logic
        || effective_cache_assignments != 1
        || !active_receipt_lines.contains(&r#""effective_cache": effective_cache,"#)
    {
        violations
            .push("P6 cache receipt does not expose profile-invalidated restores as cold".into());
    }
    let drives = benchmark["jobs"]["p5-windows-drive"]["strategy"]["matrix"]["drive"]
        .as_sequence()
        .into_iter()
        .flatten()
        .filter_map(serde_yaml::Value::as_str)
        .collect::<Vec<_>>();
    if drives != ["C", "D"] || !benchmark_workflow.contains("/wenlan-benchmark/") {
        violations.push("CI benchmark does not compare explicit Windows C and D roots".into());
    }

    let benchmark_jobs = benchmark["jobs"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _job)| name.as_str())
        .collect::<BTreeSet<_>>();
    let expected_jobs = BTreeSet::from([
        "p4-runners",
        "p5-test-engine",
        "p5-windows-drive",
        "p6-release",
    ]);
    if benchmark_jobs != expected_jobs {
        violations.push("CI benchmark contains jobs beyond the four orthogonal suites".into());
    }

    let jobs = benchmark["jobs"].as_mapping().into_iter().flatten();
    for (job_name, job) in jobs {
        let job_name = job_name.as_str().unwrap_or("<non-string>");
        if job.get("environment").is_some() || job.get("permissions").is_some() {
            violations.push(format!(
                "benchmark job {job_name} adds environment or job-level permissions"
            ));
        }
        if serde_yaml::to_string(job)
            .is_ok_and(|yaml| yaml.contains("SCCACHE_GHA_RW_MODE: READ_WRITE"))
        {
            violations.push(format!(
                "benchmark job {job_name} enables sccache writes in env"
            ));
        }
        if job["strategy"]["fail-fast"].as_bool() != Some(false) {
            violations.push(format!("benchmark job {job_name} is not fail-fast false"));
        }
        if job["timeout-minutes"]
            .as_u64()
            .is_none_or(|timeout| timeout > 90)
        {
            violations.push(format!("benchmark job {job_name} lacks a bounded timeout"));
        }
        let steps = job["steps"].as_sequence().into_iter().flatten();
        let upload = steps.clone().find(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("actions/upload-artifact@"))
        });
        if upload.and_then(|step| step["if"].as_str()) != Some("always()")
            || upload
                .and_then(|step| step["with"]["path"].as_str())
                .is_none_or(|path| path.contains("target"))
        {
            violations.push(format!(
                "benchmark job {job_name} does not always upload receipt-only evidence"
            ));
        }
    }

    let steps = benchmark["jobs"]
        .as_mapping()
        .into_iter()
        .flat_map(|jobs| jobs.values())
        .flat_map(|job| job["steps"].as_sequence().into_iter().flatten());
    let action_pin = regex::Regex::new(r"^[^@]+@[0-9a-f]{40}$").unwrap();
    for step in steps {
        let uses = step["uses"].as_str().unwrap_or_default();
        if !uses.is_empty() && !action_pin.is_match(uses) {
            violations.push("CI benchmark uses an action without an immutable SHA pin".into());
        }
        if uses.contains("Swatinem/rust-cache") && step["with"]["save-if"].as_str() != Some("false")
        {
            violations.push("CI benchmark contains a rust-cache writer".into());
        }
        if uses.contains("actions/cache@") || uses.contains("actions/cache/save@") {
            violations.push("CI benchmark contains a generic cache writer".into());
        }
        if step["run"]
            .as_str()
            .is_some_and(|run| run.contains("SCCACHE_GHA_RW_MODE") && run.contains("READ_WRITE"))
        {
            violations.push("CI benchmark contains an sccache writer".into());
        }
    }

    violations
}

#[test]
fn ci_benchmark_is_manual_restore_only_and_outside_required_ci() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let benchmark = std::fs::read_to_string(root.join(".github/workflows/ci-benchmark.yml"))
        .unwrap_or_default();
    let violations = ci_benchmark_contract_violations(&ci, &benchmark);
    assert!(
        violations.is_empty(),
        "CI benchmark contract drift — hosted benchmarks stay manual, restore-only, and \
         outside required CI. Fix .github/workflows/ci-benchmark.yml; an intentional \
         contract change also updates ci_benchmark_contract_violations() and its positive \
         control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ci_benchmark_contract_rejects_automatic_or_writing_fixture() {
    let ci = "jobs:\n  conclusion:\n    needs: [ci-benchmark]\n";
    let benchmark = r#"
on:
  push:
permissions:
  contents: write
env:
  TOKEN: ${{ secrets.RELEASE_TOKEN }}
jobs:
  bad:
    timeout-minutes: 120
    permissions:
      contents: write
    env:
      SCCACHE_GHA_RW_MODE: READ_WRITE
    strategy:
      fail-fast: true
    steps:
      - uses: Swatinem/rust-cache@v2
      - uses: actions/cache@v4
      - run: SCCACHE_GHA_RW_MODE=READ_WRITE cargo test
      - uses: actions/upload-artifact@v4
        with:
          path: target/
"#;
    let violations = ci_benchmark_contract_violations(ci, benchmark);
    for expected in [
        "workflow_dispatch-only",
        "read-only contents permission",
        "reads repository secrets",
        "required CI closure",
        "Rust routing",
        "omits experiment control",
        "profile-invalidated restores",
        "four orthogonal suites",
        "environment or job-level permissions",
        "sccache writes in env",
        "fail-fast false",
        "bounded timeout",
        "receipt-only evidence",
        "rust-cache writer",
        "generic cache writer",
        "sccache writer",
        "immutable SHA pin",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "benchmark fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

#[test]
fn ci_benchmark_contract_requires_read_only_sccache_and_truthful_restore_modes() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let benchmark = std::fs::read_to_string(root.join(".github/workflows/ci-benchmark.yml"))
        .expect("read benchmark workflow");
    let benchmark = benchmark
        .replace("      SCCACHE_GHA_RW_MODE: READ_ONLY\n", "")
        .replace("cache_mode: production-restore", "cache_mode: warm")
        .replace("profile-invalidated-cold", "production-restore");
    let violations = ci_benchmark_contract_violations(&ci, &benchmark);
    for expected in [
        "read-only sccache",
        "production-restore",
        "profile-invalidated restores",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "benchmark guard must reject missing {expected}: {violations:?}"
        );
    }
}

#[test]
fn ci_benchmark_contract_requires_target_and_host_linkers_on_windows() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let benchmark = std::fs::read_to_string(root.join(".github/workflows/ci-benchmark.yml"))
        .expect("read benchmark workflow");
    let benchmark = benchmark.replace(
        "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=",
        "REMOVED_WINDOWS_LINKER=",
    );
    let violations = ci_benchmark_contract_violations(&ci, &benchmark);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("target and host build artifacts")),
        "missing Cargo target linker must fail the benchmark contract: {violations:?}"
    );
}

#[test]
fn ci_benchmark_contract_rejects_dead_profile_cache_logic() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let benchmark = std::fs::read_to_string(root.join(".github/workflows/ci-benchmark.yml"))
        .expect("read benchmark workflow");
    let benchmark = benchmark.replace(
        "if requested_mode == \"production-restore\" and profile != \"current\"",
        "if False",
    );
    let violations = ci_benchmark_contract_violations(&ci, &benchmark);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("profile-invalidated restores")),
        "dead truthfulness logic must fail the benchmark contract: {violations:?}"
    );
}

#[test]
fn ci_benchmark_contract_rejects_unused_effective_cache_value() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let benchmark = std::fs::read_to_string(root.join(".github/workflows/ci-benchmark.yml"))
        .expect("read benchmark workflow");
    let benchmark = benchmark.replace(
        r#""effective_cache": effective_cache,"#,
        r#""effective_cache": requested_mode,"#,
    );
    let violations = ci_benchmark_contract_violations(&ci, &benchmark);
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("profile-invalidated restores")),
        "unused effective cache logic must fail the benchmark contract: {violations:?}"
    );
}

fn workflow_action_pin_violations(workflow_name: &str, workflow: &str) -> Vec<String> {
    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse workflow");
    let action_pin = regex::Regex::new(r"^[^@\s]+@[0-9a-f]{40}$").unwrap();
    let mut violations = Vec::new();
    for (job_name, job) in parsed["jobs"].as_mapping().into_iter().flatten() {
        let job_name = job_name.as_str().unwrap_or("<non-string>");
        let mut check = |location: &str, uses: Option<&str>| {
            if let Some(uses) = uses {
                if !uses.starts_with("./") && !action_pin.is_match(uses) {
                    violations.push(format!(
                        "{workflow_name} {location} uses action {uses:?} without an immutable SHA pin"
                    ));
                }
            }
        };
        check(
            &format!("job {job_name}"),
            job.get("uses").and_then(serde_yaml::Value::as_str),
        );
        for (index, step) in job["steps"].as_sequence().into_iter().flatten().enumerate() {
            let step_name = step["name"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| format!("#{index}"));
            check(
                &format!("job {job_name} step {step_name}"),
                step.get("uses").and_then(serde_yaml::Value::as_str),
            );
        }
    }
    violations
}

#[test]
fn ci_evidence_workflows_pin_every_action_by_sha() {
    let root = repo_root();
    let mut violations = Vec::new();
    for path in [
        ".github/workflows/ci.yml",
        ".github/workflows/main-canary.yml",
        ".github/workflows/ci-observer.yml",
        ".github/workflows/ci-benchmark.yml",
    ] {
        let workflow = std::fs::read_to_string(root.join(path)).expect("read workflow");
        violations.extend(workflow_action_pin_violations(path, &workflow));
    }
    assert!(
        violations.is_empty(),
        "CI evidence workflow action pin drift — third-party actions in the evidence \
         workflows must be pinned to immutable commit SHAs. Fix the workflow file named \
         below; an intentional contract change also updates workflow_action_pin_violations() \
         and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ci_evidence_action_pin_contract_rejects_mutable_refs() {
    let workflow = r#"
on: push
jobs:
  reusable:
    uses: owner/repo/.github/workflows/reusable.yml@main
  test:
    steps:
      - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
      - uses: actions/upload-artifact@v4
      - uses: ./local-action
"#;
    let violations = workflow_action_pin_violations("fixture.yml", workflow);
    assert_eq!(
        violations.len(),
        2,
        "mutable job and step action refs must fail while SHA and local refs pass: {violations:?}"
    );
    for mutable_ref in ["@main", "@v4"] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(mutable_ref)),
            "missing violation for {mutable_ref}: {violations:?}"
        );
    }
}

// ── Teeth #1: repo pointer/path resolver ──

const REPO_TOP_DIRS: &[&str] = &["crates/", "docs/", "app/", "scripts/", ".github/"];

/// Extract candidate in-repo path references from one markdown file's text.
/// Ignores code fences, URLs, prose, and `<!-- drift-ok -->` lines.
fn extract_repo_path_refs(md: &str) -> Vec<String> {
    let token = regex::Regex::new(r"[A-Za-z0-9_./\-]+").unwrap();
    let mut refs = Vec::new();
    let mut in_fence = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || line.contains("<!-- drift-ok -->") {
            continue;
        }
        for m in token.find_iter(line) {
            let t = m.as_str();
            if t.starts_with("http") {
                continue;
            }
            if t.contains('/') && REPO_TOP_DIRS.iter().any(|p| t.starts_with(p)) {
                let path = t
                    .split(':')
                    .next()
                    .unwrap()
                    .trim_end_matches(['.', ',', ')', '`']);
                if !path.is_empty() {
                    refs.push(path.to_string());
                }
            }
        }
    }
    refs
}

#[test]
fn path_extractor_finds_real_and_ignores_noise() {
    let md = "\
See `crates/wenlan-core/src/db.rs` for details.
Visit https://docs/example.com for nothing.
```
docs/in/a/fence.rs should be ignored
```
This crates/wenlan-core/src/eval/seed_contract.rs:42 line ref.
A made-up path crates/does/not/exist.rs here. <!-- drift-ok -->
";
    let refs = extract_repo_path_refs(md);
    assert!(refs.contains(&"crates/wenlan-core/src/db.rs".to_string()));
    assert!(refs.contains(&"crates/wenlan-core/src/eval/seed_contract.rs".to_string()));
    assert!(
        !refs.iter().any(|r| r.contains("fence")),
        "fenced path leaked"
    );
    assert!(
        !refs.iter().any(|r| r.contains("does/not/exist")),
        "drift-ok line leaked"
    );
    assert!(!refs.iter().any(|r| r.starts_with("http")), "url leaked");
}

#[test]
fn doc_path_references_resolve() {
    let root = repo_root();
    let mut dangling = Vec::new();
    for f in git_ls_files(&root, "*.md") {
        // Skip docs that legitimately reference aspirational / moved / extracted paths:
        // plan & design docs (not-yet-created), and AUDIT.md historical audits (may
        // reference code since extracted to other repos, e.g. the Tauri app -> wenlan-app).
        if f.starts_with("docs/plans/")
            || f.starts_with("docs/superpowers/")
            || f.ends_with("AUDIT.md")
        {
            continue;
        }
        let txt = std::fs::read_to_string(root.join(&f)).unwrap_or_default();
        for r in extract_repo_path_refs(&txt) {
            // Only resolve file-like refs (have an extension); skip directory and
            // glob-stem references, which aren't precise enough to verify.
            if Path::new(&r).extension().is_none() {
                continue;
            }
            if !root.join(&r).exists() {
                dangling.push(format!("{f} -> {r}"));
            }
        }
    }
    assert!(
        dangling.is_empty(),
        "dangling in-repo path references (fix the doc or add <!-- drift-ok -->):\n{}",
        dangling.join("\n")
    );
}

// ── Teeth #2: retrieval/eval-flag doc contract (fail-closed) ──

/// Infra/transport/path flags exempt from the documentation requirement.
/// Extend deliberately, each with a one-line reason.
const FLAG_ALLOWLIST: &[&str] = &[
    "WENLAN_PORT",                        // transport
    "WENLAN_HOST",                        // transport
    "WENLAN_BIND_ADDR",                   // transport
    "WENLAN_DATA_DIR",                    // path
    "WENLAN_PORT_FILE",                   // path
    "WENLAN_LISTENING_ON",                // runtime status
    "WENLAN_GIT_SHA",                     // build stamp
    "WENLAN_MCP_CACHE_DIR",               // path
    "WENLAN_MIGRATIONS_HASH",             // build stamp
    "WENLAN_TEST_LINT_EPOCH",             // process-only lint test clock
    "WENLAN_DATA_LOCK_CHILD_ROOT",        // test-only child-process lock root
    "WENLAN_DATA_LOCK_CHILD_READY",       // test-only child-process ready signal
    "WENLAN_DATA_LOCK_CHILD_RELEASE",     // test-only child-process release signal
    "WENLAN_TEST_STARTUP_SIGNAL_BARRIER", // test-only startup signal synchronization
    "WENLAN_RB01_PROFILE",                // test-only target-Mac profiler opt-in
    "WENLAN_RB01_LANE",                   // test-only target-Mac profiler lane
    "WENLAN_BATCH_LOG",                   // debug logging
    "WENLAN_CHATGPT_ZIP",                 // import path
];

/// BASELINE: behavioral flags undocumented when this contract was introduced
/// (2026-06-19). Grandfathered so the gate lands green on a repo with an existing
/// backlog; a NEW undocumented flag still fails fail-closed. BURN DOWN by
/// documenting each in an AGENTS.md and deleting it from this list. (Pure test/infra
/// flags — e.g. WENLAN_TEST_FASTEMBED_CACHE — should instead move to FLAG_ALLOWLIST.)
const BASELINE_UNDOCUMENTED: &[&str] = &[
    "WENLAN_COT_MAX_ITER",
    "WENLAN_COT_ROUND_TIMEOUT_SECS",
    "WENLAN_ENABLE_CONTEXT_COMPRESS",
    "WENLAN_ENABLE_COT_RETRIEVAL",
    "WENLAN_ENABLE_DUAL_POOL_RESOLVE",
    "WENLAN_ENABLE_ENTITY_MINHASH",
    "WENLAN_ENABLE_EPISODE_CHANNEL",
    "WENLAN_ENABLE_EVICTION",
    "WENLAN_ENABLE_FACT_CHANNEL",
    "WENLAN_ENABLE_FTS_HARDENING",
    "WENLAN_ENABLE_GLOBAL_PRELUDE",
    "WENLAN_ENABLE_GRAPH_GATE",
    "WENLAN_ENABLE_GRAPH_SEED",
    "WENLAN_ENABLE_REFLECTION_DEBOUNCE",
    "WENLAN_ENABLE_RERANK_BLEND",
    "WENLAN_ENABLE_SALIENCE_PRIOR",
    "WENLAN_ENABLE_SESSION_DIVERSITY",
    "WENLAN_ENABLE_TEMPORAL_GROUNDING",
    "WENLAN_EPISODE_CHANNEL_LIMIT",
    "WENLAN_EPISODE_WORD_GATE",
    "WENLAN_EVAL_ANSWER_PROMPT_V2",
    "WENLAN_EXPAND_TEMP",
    "WENLAN_FACT_CHANNEL_LIMIT",
    "WENLAN_GRAPH_FRONTIER_CAP",
    "WENLAN_GRAPH_HOP_DEPTH",
    "WENLAN_GRAPH_HUB_CAP",
    "WENLAN_GRAPH_KHOP_DEPTH",
    "WENLAN_GRAPH_KHOP_MAX_NODES",
    "WENLAN_GRAPH_SEED_TOP_K",
    "WENLAN_GRAPH_SURFACE_BUDGET",
    // Helper-read LLM batching flags (parse_clamped_*_env call sites in llm_provider.rs),
    // surfaced by the broadened read-detector. Pre-existing + undocumented at contract intro.
    "WENLAN_LLM_COALESCE_MS",
    "WENLAN_LLM_PARALLEL_SEQS",
    "WENLAN_LLM_WORKERS",
    "WENLAN_MAGNITUDE_FUSION",
    "WENLAN_MERGE_SHRINK_GUARD",
    "WENLAN_PAGE_CHANNEL_LIMIT",
    "WENLAN_PRELUDE_BUCKET_K",
    "WENLAN_PRELUDE_MIN_MEMBERS",
    "WENLAN_PRF_ROUNDS",
    "WENLAN_QUERY_DECOMP_MAX_SUBQUERIES",
    "WENLAN_QUERY_INTENT_FTS_BOOST",
    "WENLAN_SESSION_DIVERSITY_MAX",
    "WENLAN_SPACE",
    "WENLAN_TEST_FASTEMBED_CACHE",
];

/// Every WENLAN_* flag read in production source (`crates/*/src`). Matches the flag
/// name as a string-literal argument to an env reader — `env::var("…")`, `var_os("…")`,
/// or any `*_env("…")` helper (e.g. the `parse_clamped_*_env` idiom, whose name arg is
/// a literal at the call site) — so indirect reads through a helper aren't silently
/// missed. Whitespace-tolerant so multi-line call sites (name on its own line) match.
fn flags_read_in_code(root: &Path) -> BTreeSet<String> {
    let re = regex::Regex::new(r#"(?:var_os|var|_env)\s*\(\s*"(WENLAN_[A-Z0-9_]+)""#).unwrap();
    let mut flags = BTreeSet::new();
    for f in git_ls_files(root, "*.rs") {
        if !f.starts_with("crates/") || !f.contains("/src/") {
            continue; // production source only
        }
        let txt = std::fs::read_to_string(root.join(&f)).unwrap_or_default();
        for c in re.captures_iter(&txt) {
            flags.insert(c[1].to_string());
        }
    }
    flags
}

/// Every WENLAN_* flag mentioned in any tracked AGENTS.md (the prose flag docs).
fn documented_flags(root: &Path) -> BTreeSet<String> {
    let re = regex::Regex::new(r"WENLAN_[A-Z0-9_]+").unwrap();
    let mut flags = BTreeSet::new();
    for f in git_ls_files(root, "*AGENTS.md") {
        let txt = std::fs::read_to_string(root.join(&f)).unwrap_or_default();
        for m in re.find_iter(&txt) {
            flags.insert(m.as_str().to_string());
        }
    }
    flags
}

/// Fail-closed set-difference: flags read in code but neither documented nor exempt.
/// Extracted so the gate AND a positive-control test exercise the same logic.
fn undocumented_flags(
    read: &BTreeSet<String>,
    documented: &BTreeSet<String>,
    exempt: &BTreeSet<String>,
) -> Vec<String> {
    read.iter()
        .filter(|f| !documented.contains(*f) && !exempt.contains(*f))
        .cloned()
        .collect()
}

#[test]
fn flag_collectors_basic() {
    let root = repo_root();
    let doc = documented_flags(&root);
    assert!(
        doc.contains("WENLAN_GRAPH_MEMORY_STREAM"),
        "expected a known documented flag to be found"
    );
    let read = flags_read_in_code(&root);
    assert!(
        read.contains("WENLAN_GRAPH_HUB_CAP"),
        "expected a known code-read flag to be found"
    );
}

#[test]
fn behavioral_flags_are_documented() {
    let root = repo_root();
    let read = flags_read_in_code(&root);
    let documented = documented_flags(&root);
    // Exempt = explicit infra allowlist ∪ the grandfathered burn-down baseline.
    let exempt: BTreeSet<String> = FLAG_ALLOWLIST
        .iter()
        .chain(BASELINE_UNDOCUMENTED.iter())
        .map(|s| s.to_string())
        .collect();

    let missing = undocumented_flags(&read, &documented, &exempt);

    assert!(
        missing.is_empty(),
        "NEW undocumented behavioral WENLAN_* flag(s). Fix: document in an *AGENTS.md* \
         (only AGENTS.md files are scanned for docs — docs/ and READMEs do NOT count), \
         or add to FLAG_ALLOWLIST / BASELINE_UNDOCUMENTED with a reason:\n{}",
        missing.join("\n")
    );
}

#[test]
fn flag_doc_contract_detects_undocumented() {
    // Positive control: the SAME set-difference the gate uses must flag a
    // read-but-undocumented flag while leaving a documented one alone. Proves the
    // tooth bites (the failure path), not just that the live repo happens to be green.
    let read: BTreeSet<String> = ["WENLAN_REAL", "WENLAN_FAKE_UNDOCUMENTED"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let documented: BTreeSet<String> = ["WENLAN_REAL"].iter().map(|s| s.to_string()).collect();
    let exempt: BTreeSet<String> = BTreeSet::new();
    let missing = undocumented_flags(&read, &documented, &exempt);
    assert_eq!(missing, vec!["WENLAN_FAKE_UNDOCUMENTED".to_string()]);
}

#[test]
fn flag_default_mismatch_warns() {
    // Best-effort, non-blocking: report (never fail on) same-line `unwrap_or(<lit>)`
    // code defaults for human cross-check against the doc bullet. Multi-line
    // defaults are skipped (warn-by-omission).
    let root = repo_root();
    let read_re = regex::Regex::new(
        r#"env::var\("(WENLAN_[A-Z0-9_]+)"\).*unwrap_or\(([0-9]+(?:\.[0-9]+)?|true|false)\)"#,
    )
    .unwrap();
    let mut code_defaults = BTreeMap::new();
    for f in git_ls_files(&root, "*.rs") {
        if !f.starts_with("crates/") || !f.contains("/src/") {
            continue;
        }
        let txt = std::fs::read_to_string(root.join(&f)).unwrap_or_default();
        for c in read_re.captures_iter(&txt) {
            code_defaults.insert(c[1].to_string(), c[2].to_string());
        }
    }
    for (flag, def) in &code_defaults {
        eprintln!(
            "[drift-guard] {flag} same-line code default = {def} — verify the doc bullet matches."
        );
    }
}

#[test]
fn root_agents_md_stays_lean() {
    // Teeth #4 — size budget on the ONE always-loaded instruction file.
    // Root AGENTS.md (which CLAUDE.md re-imports) is paid in full context EVERY
    // session; subtree AGENTS.md load on-demand. It silently accreted 39.9KB ->
    // 57.3KB as each retrieval/engine PR appended its flag wall to the path of
    // least resistance (the file it was already editing). This gate makes the
    // agents.md hierarchical convention the DEFAULT-BY-FORCE: exceed the budget
    // and the only green path is moving crate-specific reference into the owning
    // crate's AGENTS.md, not raising this number. No verifier control needed —
    // the check is a byte comparison, not parsing logic.
    const BUDGET: u64 = 44_000; // ~11k tok. Today ~39.8KB after the 2026-06-23 extraction.
    let path = repo_root().join("AGENTS.md");
    let bytes = std::fs::metadata(&path).expect("stat root AGENTS.md").len();
    assert!(
        bytes <= BUDGET,
        "root AGENTS.md is {bytes}B > {BUDGET}B budget. It loads in FULL every session. \
         Push crate-specific reference (env-flag docs, deep internals) into the owning crate's \
         subtree AGENTS.md — they load on-demand and still satisfy the teeth-#2 flag-doc contract \
         (it scans every tracked *AGENTS.md). Raising BUDGET is the wrong fix."
    );
}

// ── Teeth #6: quoted AGENTS.md section-heading resolver ──
//
// Teeth #1 verifies a referenced *path* exists, but a cross-reference like
//   See `crates/wenlan-core/AGENTS.md` "Eval seed + eval read: ONE route, ONE contract".
// also names a *section heading* inside that file. When a doc-tiering refactor moves or
// renames a section, the path stays valid while the quoted heading silently dangles —
// the failure a Codex review caught on the index-and-pointer refactor. This tooth
// resolves each quoted heading against the target file's actual headings
// (case-insensitively, since prose sometimes lowercases the title).

/// Parse `<…AGENTS.md> "<heading>"` cross-references from one markdown file's text.
/// Returns (target_relative_to_root, quoted_heading); a bare `AGENTS.md` (no `/`)
/// resolves to the root AGENTS.md. Only a quote immediately following the AGENTS.md
/// mention (one optional backtick + whitespace) counts, which keeps unrelated quotes
/// out. Skips code fences and `<!-- drift-ok -->` lines, mirroring teeth #1.
fn extract_section_refs(md: &str) -> Vec<(String, String)> {
    let re = regex::Regex::new(r#"`?([A-Za-z0-9_./\-]*AGENTS\.md)`?\s+"([^"]{3,})""#).unwrap();
    let mut refs = Vec::new();
    let mut in_fence = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence || line.contains("<!-- drift-ok -->") {
            continue;
        }
        for c in re.captures_iter(line) {
            let token = &c[1];
            let target = if token.contains('/') {
                token.to_string()
            } else {
                "AGENTS.md".to_string() // bare/`root` reference => root file
            };
            refs.push((target, c[2].to_string()));
        }
    }
    refs
}

/// ATX headings (`#`..`######`) of a markdown file, heading text only, fences skipped.
fn md_headings(md: &str) -> Vec<String> {
    let re = regex::Regex::new(r"^\s*#{1,6}\s+(.*?)\s*$").unwrap();
    let mut headings = Vec::new();
    let mut in_fence = false;
    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(c) = re.captures(line) {
            headings.push(c[1].to_string());
        }
    }
    headings
}

#[test]
fn section_ref_extractor_parses_forms_and_skips_noise() {
    let md = "\
See `crates/wenlan-core/AGENTS.md` \"Eval seed contract\".
Also root `AGENTS.md` \"Eval Citation Discipline\" applies.
```
`crates/x/AGENTS.md` \"fenced ref\" must be ignored
```
An unquoted root AGENTS.md Some Heading must not match.
A suppressed `app/eval/AGENTS.md` \"skip me\" line. <!-- drift-ok -->
";
    let refs = extract_section_refs(md);
    assert!(refs.contains(&(
        "crates/wenlan-core/AGENTS.md".to_string(),
        "Eval seed contract".to_string()
    )));
    assert!(refs.contains(&(
        "AGENTS.md".to_string(),
        "Eval Citation Discipline".to_string()
    )));
    assert!(
        !refs.iter().any(|(_, h)| h == "fenced ref"),
        "fenced ref leaked"
    );
    assert!(
        !refs.iter().any(|(_, h)| h == "Some Heading"),
        "unquoted heading matched"
    );
    assert!(
        !refs.iter().any(|(_, h)| h == "skip me"),
        "drift-ok line leaked"
    );
}

#[test]
fn doc_section_references_resolve() {
    let root = repo_root();
    let mut dangling = Vec::new();
    for f in git_ls_files(&root, "*.md") {
        // Same aspirational/historical skips as teeth #1.
        if f.starts_with("docs/plans/")
            || f.starts_with("docs/superpowers/")
            || f.ends_with("AUDIT.md")
        {
            continue;
        }
        let txt = std::fs::read_to_string(root.join(&f)).unwrap_or_default();
        for (target, heading) in extract_section_refs(&txt) {
            let Ok(target_txt) = std::fs::read_to_string(root.join(&target)) else {
                // A missing target *path* is teeth #1's job for slash refs; only flag
                // here for the root file, which teeth #1's '/'-gated extractor skips.
                if !target.contains('/') {
                    dangling.push(format!(
                        "{f} -> {target} unreadable (heading \"{heading}\")"
                    ));
                }
                continue;
            };
            let want = heading.to_lowercase();
            let found = md_headings(&target_txt)
                .iter()
                .any(|h| h.to_lowercase() == want);
            if !found {
                dangling.push(format!("{f} -> {target} has no section \"{heading}\""));
            }
        }
    }
    assert!(
        dangling.is_empty(),
        "quoted AGENTS.md section references that don't resolve to a heading \
         (fix the pointer, fix the heading, or add <!-- drift-ok -->):\n{}",
        dangling.join("\n")
    );
}

#[test]
fn section_resolver_detects_moved_heading() {
    // Positive control: a quoted heading absent from the target must be flagged,
    // and a present one (case-insensitively) must be accepted.
    let src = "See `crates/wenlan-core/AGENTS.md` \"Gone Section\" for details.";
    let target = "# Title\n\n## Present Section\n\nbody\n### another one\n";
    let refs = extract_section_refs(src);
    assert_eq!(
        refs,
        vec![(
            "crates/wenlan-core/AGENTS.md".to_string(),
            "Gone Section".to_string()
        )]
    );
    let headings = md_headings(target);
    let want = refs[0].1.to_lowercase();
    assert!(
        !headings.iter().any(|h| h.to_lowercase() == want),
        "resolver must flag a heading absent from the target"
    );
    assert!(
        headings
            .iter()
            .any(|h| h.to_lowercase() == "present section"),
        "resolver must accept a heading present in the target"
    );
}

#[test]
fn m5_reader_inventory_matches_current_tree() {
    let root = repo_root();
    let output = std::process::Command::new("python3")
        .arg("scripts/m5-reader-sweep.py")
        .arg("--check")
        .current_dir(&root)
        .output()
        .expect("run scripts/m5-reader-sweep.py --check");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "M5 reader inventory drifted from the current source tree.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("reader inventory check: ok"),
        "reader sweep check mode did not emit its success receipt.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

// ── R1: keep the giant DB test module external and census-invisible ──

const DB_TEST_MODULE_PATH: &str = "db/main_tests.rs";

fn db_test_module_layout_violations(
    db_source: &str,
    external_path: &str,
    external_exists: bool,
) -> Vec<String> {
    let mut violations = Vec::new();
    let declaration = format!("#[cfg(test)]\n#[path = \"{external_path}\"]\npub(crate) mod tests;");
    if db_source.matches(&declaration).count() != 1 {
        violations.push(format!(
            "db.rs must declare its test module exactly once through {external_path:?}"
        ));
    }
    if regex::Regex::new(r"(?m)^\s*pub\(crate\)\s+mod\s+tests\s*\{")
        .unwrap()
        .is_match(db_source)
    {
        violations.push("db.rs still contains an inline pub(crate) mod tests body".into());
    }
    if !external_path.ends_with("_test.rs") && !external_path.ends_with("_tests.rs") {
        violations.push(format!(
            "{external_path} is visible to scripts/m5-reader-sweep.py; use an _test.rs or _tests.rs suffix"
        ));
    }
    if !external_exists {
        violations.push(format!(
            "external DB test module is missing: {external_path}"
        ));
    }
    violations
}

#[test]
fn db_main_tests_live_outside_db_rs() {
    let root = repo_root();
    let db_source =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/db.rs")).expect("read db.rs");
    let external = Path::new("crates/wenlan-core/src").join(DB_TEST_MODULE_PATH);
    let violations = db_test_module_layout_violations(
        &db_source,
        DB_TEST_MODULE_PATH,
        root.join(external).is_file(),
    );

    assert!(
        violations.is_empty(),
        "R1 DB test-module layout drifted:\n{}",
        violations.join("\n")
    );
}

#[test]
fn db_test_module_layout_guard_rejects_inline_and_census_visible_shapes() {
    let violations = db_test_module_layout_violations(
        "#[cfg(test)]\npub(crate) mod tests {\n    #[test]\n    fn still_inline() {}\n}\n",
        "db/tests.rs",
        false,
    );
    assert_eq!(
        violations,
        vec![
            "db.rs must declare its test module exactly once through \"db/tests.rs\"",
            "db.rs still contains an inline pub(crate) mod tests body",
            "db/tests.rs is visible to scripts/m5-reader-sweep.py; use an _test.rs or _tests.rs suffix",
            "external DB test module is missing: db/tests.rs",
        ],
        "the guard must reject the pre-R1 inline shape and the natural but census-visible filename"
    );
}

// ── R0: exact direct DB connection access baseline ──

const EXTERNAL_CONN_ACCESS_BASELINE: &[(&str, usize)] = &[];

fn direct_conn_access_count(source: &str) -> usize {
    regex::Regex::new(r"\.conn\s*\.\s*lock\s*\(\s*\)\s*\.\s*await")
        .unwrap()
        .find_iter(source)
        .count()
}

fn is_legacy_conn_census_path(path: &str) -> bool {
    path != "crates/wenlan-core/src/db.rs"
        && !path.starts_with("crates/wenlan-core/src/db/")
        && path != "crates/wenlan-core/src/drift_guard/r4_test_support_test.rs"
}

fn current_external_conn_access(root: &Path) -> BTreeMap<String, usize> {
    git_ls_files(root, "crates/wenlan-core/src")
        .into_iter()
        .filter(|path| path.ends_with(".rs"))
        .filter(|path| is_legacy_conn_census_path(path))
        .filter_map(|path| {
            let source = std::fs::read_to_string(root.join(&path)).expect("read Rust source");
            let count = direct_conn_access_count(&source);
            (count > 0).then_some((path, count))
        })
        .collect()
}

fn external_conn_access_violations(
    baseline: &BTreeMap<String, usize>,
    current: &BTreeMap<String, usize>,
) -> Vec<String> {
    let mut violations: Vec<String> = current
        .iter()
        .filter_map(|(path, count)| match baseline.get(path) {
            Some(allowed) if count == allowed => None,
            Some(allowed) if count > allowed => Some(format!(
                "{path}: direct .conn.lock() access increased {allowed} -> {count}; \
                 replace it with a bounded MemoryDB method"
            )),
            Some(allowed) => Some(format!(
                "{path}: direct .conn.lock() access decreased {allowed} -> {count}; \
                 lower the baseline in this diff"
            )),
            None => Some(format!(
                "{path}: {count} new direct .conn.lock() access{}",
                if *count == 1 { "" } else { "es" }
            )),
        })
        .collect();
    violations.extend(
        baseline
            .keys()
            .filter(|path| !current.contains_key(*path))
            .map(|path| {
                format!(
                    "{path}: stale direct .conn.lock() baseline row; remove it from the baseline"
                )
            }),
    );
    violations.sort();
    violations
}

#[test]
fn external_conn_access_matches_exact_baseline() {
    let baseline: BTreeMap<String, usize> = EXTERNAL_CONN_ACCESS_BASELINE
        .iter()
        .map(|(path, count)| ((*path).to_string(), *count))
        .collect();
    let current = current_external_conn_access(&repo_root());
    let violations = external_conn_access_violations(&baseline, &current);

    assert!(
        violations.is_empty(),
        "direct MemoryDB connection access outside db.rs/db/** must match the exact baseline; \
         replace new locks with bounded MemoryDB methods and update the baseline in the same diff \
         after removals:\n{}",
        violations.join("\n")
    );
}

#[test]
fn legacy_conn_census_excludes_only_the_syntax_aware_parser_fixture() {
    let parser_fixture = "crates/wenlan-core/src/drift_guard/r4_test_support_test.rs";
    assert!(!is_legacy_conn_census_path(parser_fixture));
    assert!(is_legacy_conn_census_path(
        "crates/wenlan-core/src/ordinary_new_test.rs"
    ));

    let current = current_external_conn_access(&repo_root());
    assert!(
        !current.contains_key(parser_fixture),
        "the syntax-aware R4 parser owns the synthetic raw-access controls in its fixture file"
    );

    let ordinary_path = "crates/wenlan-core/src/ordinary_new_test.rs".to_string();
    let violations = external_conn_access_violations(
        &BTreeMap::new(),
        &BTreeMap::from([(ordinary_path.clone(), 1)]),
    );
    assert_eq!(
        violations,
        vec![format!("{ordinary_path}: 1 new direct .conn.lock() access")],
        "an ordinary new source path must remain inside the legacy census"
    );
}

#[test]
fn external_conn_access_exact_baseline_rejects_all_drift() {
    let baseline = BTreeMap::from([
        ("crates/wenlan-core/src/a.rs".to_string(), 2),
        ("crates/wenlan-core/src/middle.rs".to_string(), 3),
        ("crates/wenlan-core/src/ok.rs".to_string(), 2),
        ("crates/wenlan-core/src/removed.rs".to_string(), 1),
    ]);
    let current = BTreeMap::from([
        ("crates/wenlan-core/src/a.rs".to_string(), 3),
        ("crates/wenlan-core/src/middle.rs".to_string(), 1),
        ("crates/wenlan-core/src/new.rs".to_string(), 1),
        ("crates/wenlan-core/src/ok.rs".to_string(), 2),
    ]);

    assert_eq!(
        external_conn_access_violations(&baseline, &current),
        vec![
            "crates/wenlan-core/src/a.rs: direct .conn.lock() access increased 2 -> 3; replace it with a bounded MemoryDB method",
            "crates/wenlan-core/src/middle.rs: direct .conn.lock() access decreased 3 -> 1; lower the baseline in this diff",
            "crates/wenlan-core/src/new.rs: 1 new direct .conn.lock() access",
            "crates/wenlan-core/src/removed.rs: stale direct .conn.lock() baseline row; remove it from the baseline",
        ],
        "the ratchet must accept exact matches and reject count drift, new files, and stale baseline rows in deterministic order"
    );
}

#[test]
fn external_conn_access_matcher_catches_formatted_await_chains() {
    let source = concat!(
        "let one = db.",
        "conn.lock().await;\n",
        "let two = db\n    .",
        "conn\n    .lock()\n    .await;\n",
        "let not_an_access = \".conn.lock()\";\n",
    );
    assert_eq!(
        direct_conn_access_count(source),
        2,
        "the matcher must catch one-line and rustfmt-split access chains"
    );
}

// ── R2: bounded historical migration modules ──

const MIGRATIONS_V004_V009_PATH: &str = "crates/wenlan-core/src/db/migrations_v004_v009.rs";
const MIGRATIONS_V004_V009: &[(i64, &str)] = &[
    (4, "migrate_4_refinement_pipeline"),
    (5, "migrate_5_session_tables"),
    (6, "migrate_6_access_tracking"),
    (7, "migrate_7_briefing_cache"),
    (8, "migrate_8_narrative_cache"),
    (9, "migrate_9_agent_activity"),
];

fn migrations_v004_v009_layout_violations(
    db_source: &str,
    module_source: &str,
    module_exists: bool,
) -> Vec<String> {
    let mut violations = Vec::new();
    if !module_exists {
        violations.push(format!(
            "historical migration module is missing: {MIGRATIONS_V004_V009_PATH}"
        ));
    }
    if db_source.matches("mod migrations_v004_v009;").count() != 1 {
        violations.push("db.rs must declare mod migrations_v004_v009 exactly once".into());
    }

    let dispatcher = db_source
        .find("// Migration 4:")
        .zip(db_source.find("// Migration 10:"))
        .and_then(|(start, end)| (start < end).then_some(&db_source[start..end]));
    match dispatcher {
        Some(dispatcher) => {
            let mut previous = 0;
            for (version, method) in MIGRATIONS_V004_V009 {
                let call = format!("self.{method}().await?;");
                match dispatcher.find(&call) {
                    Some(position) if position > previous => previous = position,
                    Some(_) => violations
                        .push(format!("migration dispatcher call is out of order: {call}")),
                    None => violations.push(format!(
                        "migration dispatcher is missing ordered call: {call}"
                    )),
                }
                if !dispatcher.contains(&format!("if version < {version}")) {
                    violations.push(format!(
                        "migration dispatcher is missing version guard {version}"
                    ));
                }
            }
            if dispatcher.contains("conn.execute") || dispatcher.contains("ALTER TABLE") {
                violations.push(
                    "run_migrations still contains SQL bodies for migrations 4 through 9".into(),
                );
            }
        }
        None => violations.push(
            "could not isolate the run_migrations segment from migration 4 through migration 9"
                .into(),
        ),
    }

    let mut previous = 0;
    for (version, method) in MIGRATIONS_V004_V009 {
        let definition = format!("async fn {method}(");
        match module_source.find(&definition) {
            Some(position) if position > previous => previous = position,
            Some(_) => violations.push(format!(
                "historical migration method is out of order: {method}"
            )),
            None => violations.push(format!(
                "historical migration module is missing method: {method}"
            )),
        }
        if !module_source.contains(&format!("PRAGMA user_version = {version}")) {
            violations.push(format!(
                "historical migration {version} lost its user_version stamp"
            ));
        }
    }
    for protected in [
        "migrate_98",
        "migrate_99",
        "truth_cutover",
        "page_truth",
        "claim_identity",
    ] {
        if module_source.contains(protected) {
            violations.push(format!(
                "historical migration module crossed the M5 boundary: {protected}"
            ));
        }
    }
    violations
}

#[test]
fn migrations_4_through_9_live_in_one_bounded_module() {
    let root = repo_root();
    let db_source =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/db.rs")).expect("read db.rs");
    let module_path = root.join(MIGRATIONS_V004_V009_PATH);
    let module_source = std::fs::read_to_string(&module_path).unwrap_or_default();
    let violations =
        migrations_v004_v009_layout_violations(&db_source, &module_source, module_path.is_file());

    assert!(
        violations.is_empty(),
        "R2 historical migration boundary drifted:\n{}",
        violations.join("\n")
    );
}

#[test]
fn historical_migration_guard_rejects_inline_sql_and_m5_scope_creep() {
    let db_source = concat!(
        "mod migrations_v004_v009;\n",
        "// Migration 4: bad inline body\n",
        "if version < 4 { conn.execute(\"ALTER TABLE pages\", ()).await?; }\n",
        "// Migration 10: boundary\n",
    );
    let module_source =
        "impl MemoryDB { async fn migrate_98_claim_identity() {} } // truth_cutover";
    let violations = migrations_v004_v009_layout_violations(db_source, module_source, true);

    assert!(
        violations
            .iter()
            .any(|item| item.contains("still contains SQL bodies")),
        "positive control must reject an inline migration body"
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("crossed the M5 boundary")),
        "positive control must reject M5 migration scope creep"
    );
}

// ── R3: bounded DB domain modules ──

const SOURCE_SYNC_MODULE_PATH: &str = "crates/wenlan-core/src/db/source_sync.rs";
const SOURCE_SYNC_METHODS: &[&str] = &[
    "upsert_sync_state",
    "get_sync_state",
    "list_sync_state_paths",
    "delete_sync_state",
    "delete_all_sync_state",
];
const ONBOARDING_MILESTONES_MODULE_PATH: &str =
    "crates/wenlan-core/src/db/onboarding_milestones.rs";
const ONBOARDING_MILESTONES_METHODS: &[&str] = &[
    "record_milestone",
    "list_milestones",
    "acknowledge_milestone",
    "increment_milestone_shown_count",
    "reset_onboarding_milestones",
];

fn db_domain_module_layout_violations(
    db_source: &str,
    module_source: &str,
    module_exists: bool,
    module_name: &str,
    module_path: &str,
    methods: &[&str],
) -> Vec<String> {
    let mut violations = Vec::new();
    if !module_exists {
        violations.push(format!("DB domain module is missing: {module_path}"));
    }

    let declaration = format!("mod {module_name};");
    if db_source.matches(&declaration).count() != 1 {
        violations.push(format!("db.rs must declare {declaration} exactly once"));
    }
    if module_source.matches("impl MemoryDB").count() != 1 {
        violations.push(format!(
            "{module_name} must contain exactly one MemoryDB implementation"
        ));
    }

    for method in methods {
        let definition = format!("pub async fn {method}(");
        if db_source.contains(&definition) {
            violations.push(format!(
                "db.rs still contains the {module_name} method body: {method}"
            ));
        }
        if !module_source.contains(&definition) {
            violations.push(format!("{module_name} module is missing method: {method}"));
        }
        if module_source.matches(&definition).count() > 1 {
            violations.push(format!(
                "{module_name} module defines {method} more than once"
            ));
        }
    }

    let expected_visible_methods: BTreeSet<String> =
        methods.iter().map(|method| (*method).to_string()).collect();
    let actual_visible_methods: BTreeSet<String> = module_source
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let after_visibility = if let Some(rest) = line.strip_prefix("pub ") {
                rest
            } else if let Some(rest) = line.strip_prefix("pub(") {
                let close = rest.find(')')?;
                rest[close + 1..].trim_start()
            } else {
                return None;
            };
            let after_async = after_visibility
                .strip_prefix("async ")
                .unwrap_or(after_visibility);
            let after_fn = after_async.strip_prefix("fn ")?;
            let name = after_fn
                .split(|character: char| {
                    character == '(' || character == '<' || character.is_whitespace()
                })
                .next()?;
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect();
    if actual_visible_methods != expected_visible_methods {
        violations.push(format!(
            "{module_name} visible method set drifted: expected {expected_visible_methods:?}, \
             found {actual_visible_methods:?}"
        ));
    }
    violations
}

#[test]
fn source_sync_methods_live_in_one_bounded_domain_module() {
    let root = repo_root();
    let db_source =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/db.rs")).expect("read db.rs");
    let module_path = root.join(SOURCE_SYNC_MODULE_PATH);
    let module_source = std::fs::read_to_string(&module_path).unwrap_or_default();
    let violations = db_domain_module_layout_violations(
        &db_source,
        &module_source,
        module_path.is_file(),
        "source_sync",
        SOURCE_SYNC_MODULE_PATH,
        SOURCE_SYNC_METHODS,
    );

    assert!(
        violations.is_empty(),
        "R3 source-sync boundary drifted:\n{}",
        violations.join("\n")
    );
}

#[test]
fn onboarding_milestone_methods_live_in_one_bounded_domain_module() {
    let root = repo_root();
    let db_source =
        std::fs::read_to_string(root.join("crates/wenlan-core/src/db.rs")).expect("read db.rs");
    let module_path = root.join(ONBOARDING_MILESTONES_MODULE_PATH);
    let module_source = std::fs::read_to_string(&module_path).unwrap_or_default();
    let violations = db_domain_module_layout_violations(
        &db_source,
        &module_source,
        module_path.is_file(),
        "onboarding_milestones",
        ONBOARDING_MILESTONES_MODULE_PATH,
        ONBOARDING_MILESTONES_METHODS,
    );

    assert!(
        violations.is_empty(),
        "R3 onboarding-milestones boundary drifted:\n{}",
        violations.join("\n")
    );
}

#[test]
fn db_domain_guard_rejects_missing_duplicate_inline_and_visible_scope_drift() {
    let db_source = concat!(
        "mod source_sync;\n",
        "mod source_sync;\n",
        "impl MemoryDB { pub async fn upsert_sync_state(&self) {} }\n",
    );
    let module_source = concat!(
        "impl MemoryDB {\n",
        "pub async fn upsert_sync_state(&self) {}\n",
        "pub async fn upsert_sync_state(&self) {}\n",
        "pub async fn get_sync_state(&self) {}\n",
        "pub async fn list_sync_state_paths(&self) {}\n",
        "pub async fn delete_sync_state(&self) {}\n",
        "pub fn unrelated_domain_method(&self) {}\n",
        "pub(crate) async fn unrelated_crate_method(&self) {}\n",
        "pub(super) fn unrelated_parent_method(&self) {}\n",
        "}\n",
        "impl MemoryDB {}\n",
    );
    let violations = db_domain_module_layout_violations(
        db_source,
        module_source,
        false,
        "source_sync",
        SOURCE_SYNC_MODULE_PATH,
        SOURCE_SYNC_METHODS,
    );

    for expected in [
        "DB domain module is missing",
        "must declare mod source_sync; exactly once",
        "exactly one MemoryDB implementation",
        "still contains",
        "missing method: delete_all_sync_state",
        "defines upsert_sync_state more than once",
        "visible method set drifted",
    ] {
        assert!(
            violations.iter().any(|item| item.contains(expected)),
            "positive control did not trigger {expected:?}: {violations:?}"
        );
    }
    let visible_drift = violations
        .iter()
        .find(|item| item.contains("visible method set drifted"))
        .expect("positive control must reject unrelated visible methods");
    for unexpected in [
        "unrelated_domain_method",
        "unrelated_crate_method",
        "unrelated_parent_method",
    ] {
        assert!(
            visible_drift.contains(unexpected),
            "positive control did not detect visible method {unexpected}: {visible_drift}"
        );
    }
}

// ── Teeth #15: every production `INSERT INTO pages` names `kind` ──

/// Overwrite `[from, to)` with spaces, leaving newlines alone so reported
/// offsets still line up with the file.
fn blank_span(out: &mut [u8], from: usize, to: usize) {
    for byte in &mut out[from..to] {
        if *byte != b'\n' {
            *byte = b' ';
        }
    }
}

/// Blank comments — and, when `keep_strings` is false, string and char literals
/// too — to spaces, preserving byte length and every newline.
///
/// Two callers want opposite halves of the same walk. Reading Rust block
/// structure needs literals blanked, so a brace inside a string cannot move a
/// depth counter. Reading the SQL a file executes needs literals *kept* but
/// still recognised, so a `//` or a brace inside a query is not mistaken for
/// Rust, while prose in a comment stops being scannable text.
///
/// Handles the three shapes that actually bite: Rust block comments nest, raw
/// strings (`r#"…"#`) do not honour backslash escapes, and a `'` opens a char
/// literal only when it closes like one — otherwise it is a lifetime, which
/// never carries braces and must be left alone.
fn mask_lexical(source: &str, keep_strings: bool) -> String {
    let bytes = source.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let blank = |byte: u8| if byte == b'\n' { b'\n' } else { b' ' };
    let lit = |byte: u8| if keep_strings { byte } else { blank(byte) };
    let mut i = 0usize;
    while i < bytes.len() {
        let byte = bytes[i];

        if byte == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }

        if byte == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while i < bytes.len() {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                    continue;
                }
                if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth = depth.saturating_sub(1);
                    out.extend_from_slice(b"  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                    continue;
                }
                out.push(blank(bytes[i]));
                i += 1;
            }
            continue;
        }

        if byte == b'r' {
            let mut hashes = 0usize;
            let mut open = i + 1;
            while bytes.get(open) == Some(&b'#') {
                hashes += 1;
                open += 1;
            }
            if bytes.get(open) == Some(&b'"') {
                while i <= open {
                    out.push(lit(bytes[i]));
                    i += 1;
                }
                while i < bytes.len() {
                    if bytes[i] == b'"' {
                        let mut close = i + 1;
                        let mut seen = 0usize;
                        while seen < hashes && bytes.get(close) == Some(&b'#') {
                            seen += 1;
                            close += 1;
                        }
                        if seen == hashes {
                            while i < close {
                                out.push(lit(bytes[i]));
                                i += 1;
                            }
                            break;
                        }
                    }
                    out.push(lit(bytes[i]));
                    i += 1;
                }
                continue;
            }
        }

        if byte == b'"' {
            out.push(lit(byte));
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    out.push(lit(bytes[i]));
                    i += 1;
                    if i < bytes.len() {
                        out.push(lit(bytes[i]));
                        i += 1;
                    }
                    continue;
                }
                let closing = bytes[i] == b'"';
                out.push(lit(bytes[i]));
                i += 1;
                if closing {
                    break;
                }
            }
            continue;
        }

        if byte == b'\'' {
            // `'x'` and `'\n'` close within a few bytes; `'static` never does.
            let closes = if bytes.get(i + 1) == Some(&b'\\') {
                (2..8).find(|&n| bytes.get(i + n) == Some(&b'\''))
            } else if bytes.get(i + 2) == Some(&b'\'') {
                Some(2)
            } else {
                None
            };
            if let Some(n) = closes {
                for offset in 0..=n {
                    out.push(lit(bytes[i + offset]));
                }
                i += n + 1;
                continue;
            }
        }

        out.push(byte);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| source.to_string())
}

/// Collapse a SQL column list lifted out of Rust source: line continuations,
/// string-literal punctuation, and indentation all become single spaces, so
/// the reported site is stable under reformatting.
fn normalized_column_list(columns: &str) -> String {
    columns
        .split_whitespace()
        .map(|word| word.trim_matches(|c| c == '\\' || c == '"'))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// True when the text immediately preceding an `INTO pages` is the INSERT
/// keyword, allowing for a conflict clause: `INSERT INTO`, `INSERT OR IGNORE
/// INTO`, `INSERT OR REPLACE INTO`. Anything else that merely mentions the
/// table (`DELETE FROM pages`, a doc comment) is not a writer.
fn insert_keyword_precedes(head: &str) -> bool {
    // `ends_with` rather than word equality: the keyword is embedded in a Rust
    // string literal, so the token carrying it is `conn.execute("INSERT`.
    let head = head.trim_end();
    if head.ends_with("INSERT") {
        return true;
    }
    let mut words = head.split_whitespace().rev();
    let _conflict_action = words.next();
    words.next() == Some("OR") && words.next().is_some_and(|word| word.ends_with("INSERT"))
}

/// Production inserts into `pages` whose column list does not name `kind`,
/// reported as `path: <column list>`.
///
/// `pages.kind` (migration 89) is `NOT NULL DEFAULT 'concept'`, so omitting it
/// does not fail the insert — it silently asserts the row is a concept page.
/// That default is exactly how every page written after migration 89 came to
/// lie about what it is, the reserved Overview singleton included. The column
/// list is the only place the omission is visible, so that is what is read.
///
/// Matching is deliberately loose on shape and strict on meaning: whitespace is
/// collapsed first (so a statement split across source lines reads like a
/// compact one) and keywords are compared case-insensitively against an
/// ASCII-uppercase twin, because a guard that only recognises the one spelling
/// the current code happens to use is a guard that passes the day someone
/// writes `INSERT OR IGNORE` — a form this very file's migration 89 already
/// uses on another table.
fn page_insert_sites_without_kind(path: &str, source: &str) -> Vec<String> {
    let production = strip_cfg_test_items(source);
    let collapsed = production.split_whitespace().collect::<Vec<_>>().join(" ");
    // `to_ascii_uppercase` is length-preserving, so one offset indexes both.
    let upper = collapsed.to_ascii_uppercase();
    let mut sites = Vec::new();
    let mut cursor = 0usize;
    while let Some(found) = upper[cursor..].find("INTO PAGES") {
        let start = cursor + found;
        cursor = start + "INTO PAGES".len();
        let after = &collapsed[cursor..];
        // `pages_fts`, `pages_pre49`, … are other tables entirely.
        if after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue;
        }
        if !insert_keyword_precedes(&upper[..start]) {
            continue;
        }
        let Some(open) = after.find('(') else {
            continue;
        };
        let Some(close) = after[open..].find(')') else {
            continue;
        };
        let columns = normalized_column_list(&after[open + 1..open + close]);
        if columns.split(',').any(|column| column.trim() == "kind") {
            continue;
        }
        sites.push(format!("{path}: {columns}"));
    }
    sites
}

/// Migration 46 rebuilds `pages` from the legacy `concepts` table at
/// `user_version = 46`, forty-three migrations before `kind` exists. Naming the
/// column there would be a syntax error against the schema of its own era, so
/// it is the one production insert that legitimately omits it — pinned by its
/// exact column list rather than a line number so the exemption cannot quietly
/// widen to cover a new writer.
const PRE_KIND_SCHEMA_REBUILD: &str = concat!(
    "crates/wenlan-core/src/db.rs: ",
    "id, title, summary, content, entity_id, domain, source_memory_ids, version, status, ",
    "embedding, created_at, last_compiled, last_modified, sources_updated_count, ",
    "stale_reason, user_edited"
);

/// Test modules in this crate live in their own files, gated by a
/// `#[cfg(test)] mod …;` declaration in the parent — a convention
/// `db_main_tests_live_outside_db_rs` already enforces. The gate is therefore
/// in the parent file, not in the module, so the `#[cfg(test)]` strip cannot
/// see it and the scan has to recognise these by name.
fn is_test_only_module(path: &str) -> bool {
    let stem = path.strip_suffix(".rs").unwrap_or(path);
    stem.ends_with("_test")
        || stem.ends_with("_tests")
        || stem.contains("_test/")
        || stem.contains("_tests/")
        || stem.contains("_test_support")
}

#[test]
fn every_production_page_insert_names_kind() {
    let root = repo_root();
    let mut sites = Vec::new();
    for path in git_ls_files(&root, "*.rs").into_iter().filter(|path| {
        path.starts_with("crates/")
            && path.contains("/src/")
            && path != "crates/wenlan-core/src/drift_guard.rs"
            && !is_test_only_module(path)
    }) {
        let source = std::fs::read_to_string(root.join(&path)).expect("read Rust source");
        sites.extend(page_insert_sites_without_kind(&path, &source));
    }
    assert_eq!(
        sites,
        [PRE_KIND_SCHEMA_REBUILD],
        "a production INSERT INTO pages omits `kind`, so the row it writes takes the \
         NOT NULL DEFAULT 'concept' and claims to be a concept page whatever it really is. \
         Name the column and stamp it from crate::pages::page_kind_for"
    );
}

#[test]
fn page_insert_kind_guard_detects_omission_and_ignores_lookalikes() {
    let omitted = page_insert_sites_without_kind(
        "crates/wenlan-core/src/somewhere.rs",
        "conn.execute(\"INSERT INTO pages (id, title, creation_kind) VALUES (?1, ?2, ?3)\", ())",
    );
    assert_eq!(
        omitted,
        ["crates/wenlan-core/src/somewhere.rs: id, title, creation_kind"],
        "positive control must catch an omission and must not read `creation_kind` as `kind`"
    );

    assert!(
        page_insert_sites_without_kind(
            "crates/wenlan-core/src/somewhere.rs",
            "conn.execute(\"INSERT INTO pages (id, title, kind, creation_kind) \
             VALUES (?1, ?2, ?3, ?4)\", ())",
        )
        .is_empty(),
        "a column list naming kind is not a violation"
    );

    assert!(
        page_insert_sites_without_kind(
            "crates/wenlan-core/src/somewhere.rs",
            "conn.execute(\"INSERT INTO pages_fts(rowid, title) VALUES (?1, ?2)\", ())",
        )
        .is_empty(),
        "pages_fts is a different table"
    );

    // Spellings the first cut of this guard was blind to. Each one is a real
    // way to write the same insert, so each one must still be caught.
    for (label, source) in [
        (
            "conflict clause",
            "conn.execute(\"INSERT OR IGNORE INTO pages (id, title, creation_kind) \
             VALUES (?1, ?2, ?3)\", ())",
        ),
        (
            "replace clause",
            "conn.execute(\"INSERT OR REPLACE INTO pages (id, title, creation_kind) \
             VALUES (?1, ?2, ?3)\", ())",
        ),
        (
            "lowercase keywords",
            "conn.execute(\"insert into pages (id, title, creation_kind) \
             values (?1, ?2, ?3)\", ())",
        ),
        (
            "split across lines",
            "conn.execute(\"INSERT INTO\n    pages\n    (id, title, creation_kind)\n \
             VALUES (?1, ?2, ?3)\", ())",
        ),
    ] {
        assert_eq!(
            page_insert_sites_without_kind("crates/wenlan-core/src/somewhere.rs", source),
            ["crates/wenlan-core/src/somewhere.rs: id, title, creation_kind"],
            "a production insert written as `{label}` must still be caught"
        );
    }

    assert!(
        page_insert_sites_without_kind(
            "crates/wenlan-core/src/somewhere.rs",
            "conn.execute(\"DELETE FROM pages WHERE id = ?1\", ())",
        )
        .is_empty(),
        "a statement that merely names the table is not an insert"
    );
}

#[test]
fn page_insert_kind_guard_is_fail_closed_after_test_items() {
    let source = concat!(
        "#[cfg(test)]\n",
        "mod tests {\n",
        "    fn fixture() {\n",
        "        conn.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n",
        "    }\n",
        "}\n",
        "async fn create(&self) {\n",
        "    #[cfg(test)]\n",
        "    if validate {\n",
        "        hooks::after_validation(id).await;\n",
        "    }\n",
        "    tx.execute(\"INSERT INTO pages (id, summary) VALUES (?1, ?2)\", ());\n",
        "}\n",
    );
    assert_eq!(
        page_insert_sites_without_kind("crates/wenlan-core/src/somewhere.rs", source),
        ["crates/wenlan-core/src/somewhere.rs: id, summary"],
        "the gated fixture must be ignored, and a production insert after an inline \
         #[cfg(test)] item must still be caught"
    );
}

/// A cfg attribute gates the syntactic node that OWNS it, not the next braced
/// item or semicolon-terminated statement. The old lexical delimiter applied
/// one item-shaped heuristic everywhere, so an attribute on a parameter, arm,
/// or field could consume later production code and make both teeth pass on
/// code they never scanned.
#[test]
fn cfg_stripping_preserves_production_after_nested_attribute_positions() {
    let path = "crates/wenlan-core/src/somewhere.rs";
    let fixtures = [
        (
            "function parameter",
            "fn create(#[cfg(test)] _: (), page: &Page) {\n\
                 if page.kind.as_str() == \"source\" {}\n\
                 tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
             }",
        ),
        (
            "closure parameter",
            "fn create(page: &Page) {\n\
                 let _probe = |#[cfg(test)] _: (), value: u8| value;\n\
                 if page.kind.as_str() == \"source\" {}\n\
                 tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
             }",
        ),
        (
            "match arm",
            "fn create(page: &Page, tag: &str) {\n\
                 let _ = match tag {\n\
                     #[cfg(test)] \"probe\" => 0,\n\
                     _ => {\n\
                         if page.kind.as_str() == \"source\" {}\n\
                         tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
                         1\n\
                     }\n\
                 };\n\
             }",
        ),
        (
            "struct field",
            "struct Probe { #[cfg(test)] only: (), value: u8 }\n\
             fn create(page: &Page) {\n\
                 if page.kind.as_str() == \"source\" {}\n\
                 tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
             }",
        ),
    ];

    for (label, source) in fixtures {
        assert_eq!(
            page_kind_routing_sites(path, source),
            [format!("{path}: kind == \"source\"")],
            "a cfg-gated {label} must not blank the production comparison after it"
        );
        assert_eq!(
            page_insert_sites_without_kind(path, source),
            [format!("{path}: id, title")],
            "a cfg-gated {label} must not blank the production INSERT after it"
        );
    }

    let token_whitespace = "# [cfg(test)]\n\
         fn probe(page: &Page) {\n\
             if page.kind.as_str() == \"source\" {}\n\
             tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
         }\n\
         fn create(page: &Page) {\n\
             if page.kind.as_str() == \"source\" {}\n\
             tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n\
         }";
    assert_eq!(
        page_kind_routing_sites(path, token_whitespace),
        [format!("{path}: kind == \"source\"")],
        "whitespace between `#` and `[` must not expose a test-only comparison"
    );
    assert_eq!(
        page_insert_sites_without_kind(path, token_whitespace),
        [format!("{path}: id, title")],
        "whitespace between `#` and `[` must not expose a test-only INSERT"
    );

    let utf8_before_routing = "fn create(page: &Page) { let 測試測試測試測試測試測試 = 0; \
         if page.kind.as_str() == \"source\" {} #[cfg(test)] let _probe = [0; 64]; }";
    syn::parse_file(utf8_before_routing).expect("the UTF-8 routing control must be valid Rust");
    assert_eq!(
        page_kind_routing_sites(path, utf8_before_routing),
        [format!("{path}: kind == \"source\"")],
        "UTF-8 before a cfg span must not move blanking onto an earlier production comparison"
    );

    let utf8_before_insert =
        "fn create() { let 測試測試測試測試測試測試測試測試測試測試測試測試 = 0; \
         tx.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ()); \
         #[cfg(test)] let _probe = [0; 64]; }";
    syn::parse_file(utf8_before_insert).expect("the UTF-8 INSERT control must be valid Rust");
    assert_eq!(
        page_insert_sites_without_kind(path, utf8_before_insert),
        [format!("{path}: id, title")],
        "UTF-8 before a cfg span must not move blanking onto an earlier production INSERT"
    );
}

/// A brace that is only *text* must not move the depth counter. Both fixtures
/// below put an unmatched `{` inside the `#[cfg(test)]` region — once in a
/// string, once in a comment — so a counter reading raw bytes never sees the
/// region balance, blanks the rest of the file, and reports nothing. Silence
/// is the worst possible answer from a fail-closed guard, so each fixture is
/// followed by a production insert the scan is required to still catch.
#[test]
fn page_insert_kind_guard_counts_only_structural_braces() {
    for (label, text) in [
        ("string", "        let brace = \"{\";\n"),
        ("comment", "        // an unmatched { , written in prose\n"),
    ] {
        let source = format!(
            concat!(
                "#[cfg(test)]\n",
                "mod tests {{\n",
                "    fn fixture() {{\n",
                "{}",
                "        conn.execute(\"INSERT INTO pages (id, title) VALUES (?1, ?2)\", ());\n",
                "    }}\n",
                "}}\n",
                "async fn create() {{\n",
                "    tx.execute(\"INSERT INTO pages (id, summary) VALUES (?1, ?2)\", ());\n",
                "}}\n",
            ),
            text
        );
        syn::parse_file(&source)
            .unwrap_or_else(|error| panic!("the {label} control must be valid Rust: {error}"));
        assert_eq!(
            page_insert_sites_without_kind("crates/wenlan-core/src/somewhere.rs", &source),
            ["crates/wenlan-core/src/somewhere.rs: id, summary"],
            "an unmatched brace in a {label} must not swallow the production insert after it"
        );
    }
}

// ── Teeth #16: no production read routes on a non-`entity` page kind ──

/// `pages.kind` values a production read may not discriminate on.
///
/// `'entity'` is absent on purpose. The `kind='entity'` shadow fence predates
/// this repair and is the one value every writer has always stamped
/// deliberately, so it is the one value a reader can trust today.
const UNROUTABLE_PAGE_KINDS: [&str; 4] = ["authored", "concept", "overview", "source"];

/// SQL clauses that bind a statement to a table. A literal carrying none of
/// them names no table, which makes it a fragment rather than a statement.
const SQL_TABLE_CLAUSES: [&str; 5] = ["FROM ", "JOIN ", "UPDATE ", "INTO ", "TABLE "];

/// Keywords a SQL predicate opens with. A fragment names no table, so this is
/// the only thing separating `" AND kind = 'overview'"` from a log line that
/// happens to mention `kind='concept'` — migration 107 ships exactly such a
/// line, and without this the tooth cries wolf on its own repair.
const SQL_PREDICATE_LEADS: [&str; 5] = ["AND", "OR", "WHERE", "HAVING", "ON"];

/// Production reads that filter or route on a `pages.kind` value other than
/// `'entity'`, reported as `path: kind <op> <literal>`.
///
/// Writing the column truthfully is not the same as making it trustworthy to
/// read. Two gaps survive this change on purpose. Rename, archive, and replace
/// paths never re-derive `kind`, so a page renamed to `Overview` — or away from
/// it — keeps whatever it was stamped with. And migration 89 folded only
/// `creation_kind='imported'` onto `'source'`, never `'source'` itself, so a
/// page imported before 89 still sits at `'concept'`; migration 107's ledger
/// guard skips exactly those rows, because 89 already wrote them an entry. A
/// reader routing on those values today is routing on stale data, which is why
/// the fence stays up until M6 re-derives `kind` on every mutation path.
///
/// A write is not a read. A SQL predicate whose own literal names a table is
/// excused unless that literal *selects* from `pages` (`FROM pages` / `JOIN
/// pages`), which is what keeps migration 89's `CHECK` constraint, migration
/// 107's `UPDATE pages … WHERE kind = 'concept'` repair guard, and every other
/// table's `kind` column out of the scan without an exemption list that would
/// have to be maintained — and widened — by hand. A literal
/// naming *no* table proves nothing either way, and is counted rather than
/// excused when it opens on a SQL connective: that is the only reading under
/// which `sql.push_str(" AND kind = 'overview'")` is visible at all, since the
/// `FROM pages` it belongs to sits in a different literal. The connective is
/// what keeps prose out — migration 107 logs the phrase `kind='concept'`, and
/// counting every literal that merely contains the column would make the
/// repair's own success message a finding. The residual cost is that a
/// fragment appended to some *other* table's query trips the tooth; naming
/// that table in the fragment clears it.
///
/// The operator and the quote style are matched as a pair: SQL spells equality
/// `=` around single quotes, Rust spells it `==` around double quotes. Nothing
/// else is a comparison. That pairing is what keeps a Rust assignment and the
/// `kind = "concept"` line inside an eval TOML fixture from reading as reads.
/// Accessors between the column and the operator are stepped over, so
/// `page.kind.as_str() == "overview"` and `page.kind() != "concept"` are the
/// same site as `page.kind == "overview"`.
///
/// Control flow is the other half and lives in `page_kind_match_sites`, which
/// this calls. That half is parsed rather than scanned. What being parsed buys
/// is narrower than it first looks, and worth stating exactly. Punctuation
/// stopped being load-bearing: comma-optional bodies, turbofish commas and
/// closure commas are structure now.
///
/// Nesting is reached in four positions, which is the honest enumeration this
/// paragraph owed and did not give: a `match` written inside an ARM BODY, inside
/// an ARM GUARD, inside a `matches!` SCRUTINEE, and inside a `matches!` GUARD.
/// The last of those was parsed and thrown away until round 7, so
/// `matches!(flag, true if match page.kind.as_str() { … })` was invisible while
/// this very paragraph claimed guards could not hide shape. Reaching a position
/// is not the same as recording in it: the routed literal is taken from
/// `arm.pat` only, because a `"source"` in a body or a guard is a value the code
/// computes rather than a pattern the column is matched against, and recording
/// either would make the tooth cry wolf. A guard that compares directly —
/// `if page.kind.as_str() == "source"` — is still caught, but by the lexical
/// comparison scan above, not by either visitor.
///
/// What a reader NAMES can still hide, and that is where the list below lives.
///
/// What still gets past this, each verified to escape rather than assumed to:
///
/// - a comparison spelled as a method call — `kind.eq("overview")`,
///   `kind.starts_with("over")`. The accessor walk steps onto the call and
///   finds no operator behind it.
/// - a literal reached indirectly, on either half: `const OVERVIEW: &str =
///   "overview"` then `kind == OVERVIEW`, and equally `match page.kind.as_str()
///   { OVERVIEW => … }`, where the arm is a path pattern and not a literal one.
///   Both compare an identifier against nothing.
/// - a value that is not a literal at all — `" AND kind = ?1"` with the kind
///   bound at runtime, or `format!(" AND kind = '{wanted}'")`. A bound value is
///   not knowable from the source, so this one is a limit of static reading
///   rather than of this tooth.
/// - the column reached through an intermediate binding: `let k =
///   normalise(page.kind.as_str()); match k { "source" => … }`. The scrutinee
///   names `k`, and this scan does not follow assignments. Direct reads and
///   one-expression compounds — `(page.kind.as_str(), active)`,
///   `normalise(page.kind.as_str())`, `{ page.kind.as_str() }` — are covered;
///   a named hop is not.
/// - a SQL fragment that opens on something other than `AND`, `OR`, `WHERE`,
///   `HAVING`, or `ON`: a bare `" kind = 'overview'"` relying on the base
///   ending in `WHERE`, a leading `"("`, or a `" WHEN kind = 'overview' THEN"`
///   inside a CASE. This is the price of the connective gate, and the gate is
///   what keeps migration 107's log line out; widening the lead set trades
///   reach for false findings on prose.
/// - a `match` inside a macro body that is not a sequence of statements, and a
///   routing decision hidden behind a macro of this repository's own —
///   `is_source!(page)`. Macro tokens are not parsed on the way in, so the scan
///   re-parses what it can (`my_wrapper!(let l = match page.kind … ;)` is
///   reached) and skips bodies with their own grammar. Narrower than the old
///   text scan, which saw every macro body: the one place going structural
///   costs reach.
/// - a kind that is not a string. If `kind` ever becomes an enum,
///   `match page.kind { PageKind::Source => … }` is a variant pattern, not a
///   string literal, and this scan goes quiet. Nothing in the tree is written
///   that way today; it is named because migrating the column would silently
///   retire the tooth rather than redden it.
///
/// The shape defects three review rounds found — the guard, the comma boundary,
/// the turbofish, the enumerated scrutinee list — are gone by construction
/// rather than patched, and `if let`/`while let` moved from disclosed hole to
/// coverage.
///
/// Two things err the other way, toward a visible false finding rather than a
/// silent miss, which is the direction a fail-closed guard should lean: `cfg`
/// predicates are skipped only when they genuinely REQUIRE `test`, so an item
/// excluded from production builds by some other means is still scanned; and a
/// scrutinee is searched as a whole tree, so a `match` over an unrelated value
/// computed near anything named `kind` can report. A file that does not parse,
/// and a `matches!` whose pattern cannot be recovered, are likewise reported
/// rather than passed.
///
/// WHERE a `cfg` is read is its own ledger, and it is a cry-wolf ledger rather
/// than a hole ledger: an unread `cfg` position can only ADD a finding for code
/// that never ships, never hide a reader. Both halves of the tooth now ask the
/// same question through `compiles_outside_test` — the lexical scan judged the
/// literal text `#[cfg(test)]` until round 7, so `#[cfg(all(test, feature =
/// "slow"))]` produced a production finding the visitor gates had no power to
/// retract, since a text-scan finding is recorded before the parser runs.
///
/// The positions are no longer a list anyone curates. EVERY node in the pinned
/// syn 2.0.117 that owns a `Vec<Attribute>` is gated — 19 of them, enumerated on
/// `skip_cfg_test_nodes!`. That is a derived claim, and this is the derivation:
///
/// ```text
/// grep -rn 'pub attrs: Vec<Attribute>' "$(cargo metadata --format-version 1 \
///   | jq -r '.packages[]|select(.name=="syn" and .version=="2.0.117")|.manifest_path' \
///   | xargs dirname)/src"
/// ```
///
/// 94 hits, 93 distinct nodes (`ItemStatic` appears twice — once for real, once
/// inside a token.rs doc comment). One is excluded: `DeriveInput`, which a file
/// walk never reaches, because items parse as `Item`. The other 92 are each
/// reached by at least one of the 19 gates.
///
/// It reads as over-coverage — nothing in this tree writes a `cfg` on a bare-fn
/// argument — and that is deliberate. Rounds 6, 7 and 8 each shipped a
/// hand-picked subset, each with an argument for why the omissions were
/// harmless, and each argument was wrong for a different reason: `Item::Type`
/// because a const position can hold a MACRO and `visit_macro` re-parses macro
/// input as statements; closure parameters and struct-pattern fields because
/// `Pat::Macro` reaches the same re-parse. Three rounds of that is enough
/// evidence that the subset is not knowable in advance. Closing the list costs
/// nine lines and ends the class.
///
/// Two things this does NOT claim. It is exhaustive over syn's ATTRIBUTE
/// positions, not over ways to hide a routing decision — a match assembled by a
/// build script, or behind a proc macro, is not in this file at all. And it is
/// pinned to 2.0.117: a syn bump that adds an attribute-bearing node reopens the
/// gap silently, which is why the derivation is written down here rather than
/// left as something the next reader has to reconstruct.
///
/// None of the above is characterised as evasion; they are ordinary ways to
/// write the same decision. It is a drift guard over the shapes this repository
/// actually writes.
fn page_kind_routing_sites(path: &str, source: &str) -> Vec<String> {
    let production = strip_cfg_test_items(source);
    // Literals kept, comments blanked: the SQL this looks for lives inside
    // string literals, while prose *about* `kind='concept'` is not a reader.
    let visible = mask_lexical(&production, true);
    let collapsed = visible.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut sites = Vec::new();
    let mut cursor = 0usize;
    while let Some(found) = collapsed[cursor..].find("kind") {
        let start = cursor + found;
        cursor = start + "kind".len();
        // `creation_kind`, `assigned_kind`, `prior_creation_kind` are other
        // columns and carry no fence.
        if collapsed[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }
        let mut rest = collapsed[cursor..].trim_start();
        // `COALESCE(kind, 'concept') != 'entity'` is how this codebase already
        // spells the entity fence, at every one of its two dozen live sites, so
        // a reader that grows here will be a copy of one of them with the
        // literal changed. Step over the default and read the operator that
        // follows, or the guard is blind to its single most likely spelling.
        let head = &collapsed[..start];
        if head
            .len()
            .checked_sub("COALESCE(".len())
            .and_then(|from| head.get(from..))
            .is_some_and(|word| word.eq_ignore_ascii_case("COALESCE("))
        {
            let Some(close) = rest.find(')') else {
                continue;
            };
            rest = rest[close + 1..].trim_start();
        }
        // An accessor is transparent. `page.kind.as_str() == "overview"` and
        // `page.kind() != "concept"` route on the value `page.kind ==
        // "overview"` routes on; only the spelling differs, and `Page.kind` is
        // a public `String`, so the accessor forms are the ones a reader would
        // actually reach for. Step the chain until the operator is reachable.
        // Nothing is decided here — the literal check below still gates every
        // hit — so stepping past a `.len()` or a `.clone()` costs only reach.
        loop {
            let stepped = if let Some(after) = rest.strip_prefix("()") {
                after
            } else if let Some(name) = rest.strip_prefix('.') {
                let len = name
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(name.len());
                if len == 0 {
                    break;
                }
                &name[len..]
            } else {
                break;
            };
            rest = stepped.trim_start();
        }
        // Which quote styles each operator may legally carry. SQLite spells
        // inequality both `<>` and `!=`, but only Rust spells equality `==`,
        // and a bare `=` beside a double-quoted string is a Rust *assignment*,
        // which is what keeps the `kind = "concept"` line inside an eval TOML
        // fixture from reading as a comparison.
        let (op, single, double, tail) = if let Some(tail) = rest.strip_prefix("==") {
            ("==", false, true, tail)
        } else if let Some(tail) = rest.strip_prefix("!=") {
            ("!=", true, true, tail)
        } else if let Some(tail) = rest.strip_prefix("<>") {
            ("<>", true, false, tail)
        } else if let Some(tail) = rest.strip_prefix('=') {
            ("=", true, false, tail)
        } else if rest
            .get(..2)
            .is_some_and(|word| word.eq_ignore_ascii_case("in"))
            && !rest[2..].starts_with(|c: char| c.is_alphanumeric() || c == '_')
        {
            ("IN", true, false, &rest[2..])
        } else {
            continue;
        };
        let operand = tail.trim_start();
        let (scope, list) = match operand.strip_prefix('(') {
            Some(items) => (items.split(')').next().unwrap_or_default(), true),
            None => (operand, false),
        };
        let quote = match scope.trim_start().chars().next() {
            Some('\'') if single => '\'',
            Some('"') if double => '"',
            // A column, a CASE, a bind parameter, an enum path: not a literal.
            _ => continue,
        };
        let literals: Vec<&str> = if list {
            scope.split(quote).skip(1).step_by(2).collect()
        } else {
            scope.split(quote).nth(1).into_iter().collect()
        };
        let Some(routed) = literals
            .iter()
            .find(|literal| UNROUTABLE_PAGE_KINDS.contains(literal))
        else {
            continue;
        };
        if quote == '\'' {
            let literal_start = collapsed[..start].rfind('"').map_or(0, |open| open + 1);
            let statement = collapsed[literal_start..start].to_ascii_uppercase();
            // A literal naming no table at all is a fragment, not a statement:
            // `sql.push_str(" AND kind = 'overview'")` appended to a base built
            // somewhere else. Demanding the proof sit in the same literal is
            // what let that shape through, so a fragment is read as one and
            // counted, rather than excused for failing to prove itself — but
            // only when it OPENS on a predicate keyword, or every log line
            // mentioning `kind='concept'` becomes a finding. The opening token
            // is the test because everything between it and the column is
            // punctuation a reader is free to write: `AND p.kind`,
            // `AND COALESCE(kind, 'concept')` — the live fence's own spelling —
            // and `AND (kind` are all the same predicate.
            let fragment_predicate = !SQL_TABLE_CLAUSES
                .iter()
                .any(|clause| statement.contains(clause))
                && statement
                    .split_whitespace()
                    .next()
                    .is_some_and(|word| SQL_PREDICATE_LEADS.contains(&word));
            if !fragment_predicate
                && !statement.contains("FROM PAGES")
                && !statement.contains("JOIN PAGES")
            {
                continue;
            }
        }
        sites.push(format!("{path}: kind {op} {quote}{routed}{quote}"));
    }
    // The match scan gets the ORIGINAL source, not the stripped twin. Exact
    // span blanking need not leave delimiters parseable, and the parser already
    // skips test-only syntax structurally with the same gate list.
    sites.extend(page_kind_match_sites(path, source));
    sites
}

/// The same scan over a statement fragment rather than a whole file, so the
/// controls below can stay the one-liners they are.
///
/// This is the ONLY place a snippet is wrapped and re-parsed. Keeping it out of
/// [`page_kind_routing_sites`] is what makes the production path strictly
/// fail-closed: a real file that no longer parses becomes a finding instead of
/// being silently retried as a function body until something sticks.
#[cfg(test)]
fn page_kind_routing_sites_in_snippet(path: &str, snippet: &str) -> Vec<String> {
    page_kind_routing_sites(path, &format!("fn drift_guard_probe() {{\n{snippet}\n}}"))
}

/// Rust `match` / `matches!` routing on a page kind — the shape the comparison
/// scan structurally cannot see, because an arm is a pattern and never writes
/// an operator.
///
/// Parsed with `syn` rather than walked as text. Three review rounds found
/// three separate defects in a hand-rolled arm walker (a literal read by the
/// punctuation that followed it, an arm boundary read off commas, an
/// enumerated list of scrutinee wrappers), and the second had no textual fix:
/// `make::<u8, u16>("source")` and `a < b, c > d` are the same character
/// sequence at the depth a bracket counter can see, so only a parser can say
/// whether that comma separates arms. `drift_guard` is `#[cfg(test)]`
/// (lib.rs:29) and `syn` is already a dev-dependency of this crate with `full`
/// + `visit`, so this costs nothing at production-compile time.
///
/// With an AST the three regions of `pattern if guard => body` arrive already
/// separated, so guards, comma-optional block bodies, turbofish commas, closure
/// commas, and nested matches all stop being special cases. The routed literal
/// is taken from `arm.pat` ALONE — a `"source"` in a body is prose and one in a
/// guard is an expression, and recording either would make the tooth cry wolf.
/// That is a rule about where a literal is RECORDED, not about where this walks:
/// arm bodies, arm guards, `matches!` scrutinees and `matches!` guards are all
/// traversed, because a `match` written inside any of them is its own site.
///
/// A production file parses as a FILE or it is a finding. There is deliberately
/// no retry as a function body here: substituting something the parser *can*
/// read for the thing it cannot is how a checker reports OK on what it never
/// checked. Snippet parsing exists, but only behind the test-only entry point
/// `page_kind_routing_sites_in_snippet`.
fn page_kind_match_sites(path: &str, production: &str) -> Vec<String> {
    let Ok(file) = syn::parse_file(production) else {
        return vec![format!(
            "{path}: could not be parsed as Rust, so the page-kind match scan could not run"
        )];
    };
    let mut visitor = MatchVisitor {
        path,
        sites: Vec::new(),
    };
    visitor.visit_file(&file);
    visitor.sites
}

/// Whether code carrying this `cfg` can still be compiled into a NON-test
/// build — the question that decides whether the scan may skip it.
///
/// Unknown predicates (`feature = "..."`, `target_os = "..."`) count as "yes,
/// it can", so a subtree is skipped only when `test` is genuinely REQUIRED.
/// That direction is the whole point: `#[cfg(not(test))]` and
/// `#[cfg(any(target_os = "macos", test))]` are production code — the second is
/// live style in this repository (crates/wenlan-server/src/main.rs:156) — and
/// an over-eager skip is a silent hole, while an over-eager scan is at worst a
/// visible false finding. Reading any nested `test` ident as "test-only" got
/// exactly this backwards.
fn compiles_outside_test(meta: &syn::Meta) -> bool {
    match meta {
        syn::Meta::Path(path) => !path.is_ident("test"),
        syn::Meta::NameValue(_) => true,
        syn::Meta::List(list) => {
            let Ok(nested) =
                list.parse_args_with(Punctuated::<syn::Meta, Token![,]>::parse_terminated)
            else {
                return true;
            };
            if list.path.is_ident("all") {
                nested.iter().all(compiles_outside_test)
            } else if list.path.is_ident("any") {
                nested.iter().any(compiles_outside_test)
            } else {
                // `not(..)`, and anything unrecognised: assume it compiles, so
                // the subtree is scanned rather than skipped.
                true
            }
        }
    }
}

/// True only for attributes that make the code they carry test-only.
fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg")
            && attr
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| !compiles_outside_test(&meta))
    })
}

/// Every `Item` variant that carries attributes. Enumerated in full rather than
/// sampled, for the third time in this scan's history: a partial list of
/// scrutinee wrappers, then a partial list of `Expr` variants, then this.
///
/// The 7-of-15 list this replaces was defended as consequence-free on the
/// grounds that the omitted containers hold expressions only in const position,
/// where no runtime `page` exists. That reasoning was wrong, and the
/// counter-example is short: a const position can hold a MACRO, and
/// `visit_macro` re-parses macro input as statements, so
/// `#[cfg(test)] type T = [u8; ignored!(match page.kind.as_str() { … })];`
/// reported a test-only match as production. Const position bounds what
/// COMPILES, not what this scan reads.
///
/// Checked against the pinned syn (=2.0.117): of its 16 `Item` variants, 15
/// carry `attrs` and all 15 are listed. `Item::Verbatim` is the one exclusion —
/// it wraps a raw `TokenStream` and has no attributes to read.
fn item_attrs(item: &syn::Item) -> &[syn::Attribute] {
    match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

/// All 16 attribute-bearing `Pat` variants of the pinned syn; `Verbatim` is the
/// exclusion. Five of them (`Const`, `Lit`, `Macro`, `Path`, `Range`) are
/// `Expr` structs reused in pattern position, and carry the expression's own
/// attribute list.
///
/// Round 8 found this whole family unread. A closure parameter and a
/// struct-pattern field both accept a `cfg` — checked with rustc, not assumed —
/// and both descend through `Pat::Macro` into `visit_macro`, where macro input
/// is re-parsed as statements. That is the same route the `Item::Type` omission
/// took, which is why it is closed here as a family rather than one variant at
/// a time.
fn pat_attrs(pattern: &Pat) -> &[syn::Attribute] {
    match pattern {
        Pat::Const(pattern) => &pattern.attrs,
        Pat::Ident(pattern) => &pattern.attrs,
        Pat::Lit(pattern) => &pattern.attrs,
        Pat::Macro(pattern) => &pattern.attrs,
        Pat::Or(pattern) => &pattern.attrs,
        Pat::Paren(pattern) => &pattern.attrs,
        Pat::Path(pattern) => &pattern.attrs,
        Pat::Range(pattern) => &pattern.attrs,
        Pat::Reference(pattern) => &pattern.attrs,
        Pat::Rest(pattern) => &pattern.attrs,
        Pat::Slice(pattern) => &pattern.attrs,
        Pat::Struct(pattern) => &pattern.attrs,
        Pat::Tuple(pattern) => &pattern.attrs,
        Pat::TupleStruct(pattern) => &pattern.attrs,
        Pat::Type(pattern) => &pattern.attrs,
        Pat::Wild(pattern) => &pattern.attrs,
        _ => &[],
    }
}

/// All 3 `GenericParam` variants of the pinned syn — every one carries an
/// attribute list, so there is no exclusion here.
///
/// A gated type or const parameter reaches code through its DEFAULT, which is a
/// const position, and a const position can hold a macro.
fn generic_param_attrs(param: &syn::GenericParam) -> &[syn::Attribute] {
    match param {
        syn::GenericParam::Lifetime(param) => &param.attrs,
        syn::GenericParam::Type(param) => &param.attrs,
        syn::GenericParam::Const(param) => &param.attrs,
    }
}

/// All 4 attribute-bearing `ImplItem` variants of the pinned syn; `Verbatim` is
/// the exclusion. `Type` was the omission — an associated type's array length is
/// a const position, and a const position can hold a macro.
fn impl_item_attrs(item: &syn::ImplItem) -> &[syn::Attribute] {
    match item {
        syn::ImplItem::Const(item) => &item.attrs,
        syn::ImplItem::Fn(item) => &item.attrs,
        syn::ImplItem::Macro(item) => &item.attrs,
        syn::ImplItem::Type(item) => &item.attrs,
        _ => &[],
    }
}

/// All 4 attribute-bearing `TraitItem` variants of the pinned syn; `Verbatim` is
/// the exclusion.
fn trait_item_attrs(item: &syn::TraitItem) -> &[syn::Attribute] {
    match item {
        syn::TraitItem::Const(item) => &item.attrs,
        syn::TraitItem::Fn(item) => &item.attrs,
        syn::TraitItem::Macro(item) => &item.attrs,
        syn::TraitItem::Type(item) => &item.attrs,
        _ => &[],
    }
}

/// All 4 attribute-bearing `ForeignItem` variants of the pinned syn; `Verbatim`
/// is the exclusion. An `extern` block is reached through `Item::ForeignMod`,
/// and `ForeignItem::Macro` is another macro this scan re-parses.
fn foreign_item_attrs(item: &syn::ForeignItem) -> &[syn::Attribute] {
    match item {
        syn::ForeignItem::Fn(item) => &item.attrs,
        syn::ForeignItem::Macro(item) => &item.attrs,
        syn::ForeignItem::Static(item) => &item.attrs,
        syn::ForeignItem::Type(item) => &item.attrs,
        _ => &[],
    }
}

/// `Stmt::Item` and `Stmt::Expr` keep their attributes on the item or the
/// expression, where the item and expression gates already read them.
fn stmt_attrs(statement: &syn::Stmt) -> &[syn::Attribute] {
    match statement {
        syn::Stmt::Local(local) => &local.attrs,
        syn::Stmt::Macro(statement) => &statement.attrs,
        _ => &[],
    }
}

/// Every `Expr` variant that carries attributes, so a `#[cfg(test)]` in
/// expression position is read wherever it is written. Enumerated rather than
/// sampled: a partial list here is the same defect shape as the partial
/// scrutinee list this scan already had to fix once.
///
/// Checked against the pinned syn (=2.0.117): of its 40 `Expr` variants, 39
/// carry `attrs` and all 39 are listed. `Expr::Verbatim` is the one exclusion —
/// it wraps a raw `TokenStream` and has no attributes to read — so the `_` arm
/// is unreachable for any attribute-bearing node rather than a silent catch-all.
fn expr_attrs(expr: &Expr) -> &[syn::Attribute] {
    macro_rules! attrs_of {
        ($($variant:ident),+ $(,)?) => {
            match expr {
                $(Expr::$variant(node) => &node.attrs,)+
                _ => &[],
            }
        };
    }
    attrs_of!(
        Array, Assign, Async, Await, Binary, Block, Break, Call, Cast, Closure, Const, Continue,
        Field, ForLoop, Group, If, Index, Infer, Let, Lit, Loop, Macro, Match, MethodCall, Paren,
        Path, Range, RawAddr, Reference, Repeat, Return, Struct, Try, TryBlock, Tuple, Unary,
        Unsafe, While, Yield,
    )
}

/// The `cfg` gates on non-expression nodes, written once and mixed into both
/// visitors in this scan.
///
/// Eighteen positions here, plus each visitor's own `visit_expr` for the
/// nineteenth: items, impl items, trait items, foreign items, statements, match
/// arms, struct-literal fields, field declarations, enum variants, the file
/// itself, patterns, struct-pattern fields, typed patterns (which is where a
/// function parameter's attributes live), receivers, generic parameters,
/// variadics, bare-fn arguments, and bare variadics.
///
/// That list is not taste. It is every node in the pinned syn 2.0.117 that owns
/// a `Vec<Attribute>`, derived from the vendored source rather than recalled —
/// the derivation is recorded in the ceiling below. Rounds 6, 7 and 8 each
/// shipped a hand-picked subset and each was found short by a shape nobody had
/// thought of; a closed list is the only version of this that stops being wrong.
///
/// [`MatchVisitor`] walks a file looking for routing sites; `KindReader` walks a
/// single scrutinee asking whether it names the column. They have to agree on
/// which nodes are production code, because the second one runs on a subtree the
/// first has already entered: a scrutinee whose only `kind` read sits behind
/// `#[cfg(test)]` is not a production reader, and reporting it is the same
/// cry-wolf the item and statement gates fixed one level up. Sharing the gates
/// is what stops the two from drifting apart again.
///
/// Each visitor still writes its own `visit_expr`: `KindReader` stops walking
/// the moment it has found the column, and `MatchVisitor` does not. The callback
/// is a no-op for those readers and records an exact source span for the text
/// stripper below, so all three consumers share one closed gate list.
macro_rules! return_if_cfg_test_node {
    ($visitor:expr, $node:expr, $attrs:expr) => {{
        let attrs = $attrs;
        if is_cfg_test(attrs) {
            $visitor.record_cfg_test_node($node, attrs);
            return;
        }
    }};
}

macro_rules! skip_cfg_test_nodes {
    () => {
        fn visit_item(&mut self, node: &'ast syn::Item) {
            return_if_cfg_test_node!(self, node, item_attrs(node));
            visit::visit_item(self, node);
        }

        fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
            return_if_cfg_test_node!(self, node, impl_item_attrs(node));
            visit::visit_impl_item(self, node);
        }

        fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
            return_if_cfg_test_node!(self, node, trait_item_attrs(node));
            visit::visit_trait_item(self, node);
        }

        fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
            return_if_cfg_test_node!(self, node, stmt_attrs(node));
            visit::visit_stmt(self, node);
        }

        fn visit_arm(&mut self, node: &'ast syn::Arm) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_arm(self, node);
        }

        fn visit_foreign_item(&mut self, node: &'ast syn::ForeignItem) {
            return_if_cfg_test_node!(self, node, foreign_item_attrs(node));
            visit::visit_foreign_item(self, node);
        }

        // A struct-literal field keeps its attributes on the `FieldValue` and
        // not on the expression underneath, so an expression gate alone reads
        // an empty attribute list and descends. This is live style here
        // (crates/wenlan-core/src/lint/runner.rs:132).
        fn visit_field_value(&mut self, node: &'ast syn::FieldValue) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_field_value(self, node);
        }

        // A field DECLARATION and an enum variant reach code through their
        // type's array length and their discriminant. Both are const positions,
        // and a const position can hold a macro this scan re-parses.
        fn visit_field(&mut self, node: &'ast syn::Field) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_field(self, node);
        }

        fn visit_variant(&mut self, node: &'ast syn::Variant) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_variant(self, node);
        }

        // A file's own inner `#![cfg(test)]` gates every item in it.
        fn visit_file(&mut self, node: &'ast syn::File) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_file(self, node);
        }

        // Patterns. `visit_pat` covers closure parameters and every nested
        // pattern; `visit_field_pat` covers a struct pattern's fields, which
        // syn reaches directly without passing through `visit_pat`.
        fn visit_pat(&mut self, node: &'ast Pat) {
            return_if_cfg_test_node!(self, node, pat_attrs(node));
            visit::visit_pat(self, node);
        }

        fn visit_field_pat(&mut self, node: &'ast syn::FieldPat) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_field_pat(self, node);
        }

        // A function parameter is a `PatType`, which `visit_fn_arg` reaches
        // WITHOUT going through `visit_pat` — so this is not redundant.
        fn visit_pat_type(&mut self, node: &'ast syn::PatType) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_pat_type(self, node);
        }

        fn visit_receiver(&mut self, node: &'ast syn::Receiver) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_receiver(self, node);
        }

        fn visit_generic_param(&mut self, node: &'ast syn::GenericParam) {
            return_if_cfg_test_node!(self, node, generic_param_attrs(node));
            visit::visit_generic_param(self, node);
        }

        fn visit_variadic(&mut self, node: &'ast syn::Variadic) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_variadic(self, node);
        }

        fn visit_bare_fn_arg(&mut self, node: &'ast syn::BareFnArg) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_bare_fn_arg(self, node);
        }

        fn visit_bare_variadic(&mut self, node: &'ast syn::BareVariadic) {
            return_if_cfg_test_node!(self, node, &node.attrs);
            visit::visit_bare_variadic(self, node);
        }
    };
}

struct CfgTestSpanCollector {
    spans: Vec<(usize, usize)>,
}

impl CfgTestSpanCollector {
    fn record_cfg_test_node<T: Spanned>(&mut self, node: &T, attrs: &[syn::Attribute]) {
        let start = attrs.first().map_or_else(
            || node.span().byte_range().start,
            |attr| attr.span().byte_range().start,
        );
        self.spans.push((start, node.span().byte_range().end));
    }
}

impl<'ast> Visit<'ast> for CfgTestSpanCollector {
    skip_cfg_test_nodes!();

    fn visit_expr(&mut self, node: &'ast Expr) {
        return_if_cfg_test_node!(self, node, expr_attrs(node));
        visit::visit_expr(self, node);
    }
}

/// Blank exactly the parsed syntax nodes that a test-only cfg attribute owns.
/// On a parse failure the original text remains visible, which can cry wolf
/// but cannot make a production writer or routing comparison disappear.
fn strip_cfg_test_items(source: &str) -> String {
    let Ok(file) = syn::parse_file(source) else {
        return source.to_string();
    };

    let mut out = source.as_bytes().to_vec();
    if is_cfg_test(&file.attrs) {
        blank_span(&mut out, 0, source.len());
        return String::from_utf8(out).expect("blanking writes only ASCII spaces");
    }

    let mut collector = CfgTestSpanCollector { spans: Vec::new() };
    collector.visit_file(&file);
    for (from, to) in collector.spans {
        if from <= to
            && to <= source.len()
            && source.is_char_boundary(from)
            && source.is_char_boundary(to)
        {
            blank_span(&mut out, from, to);
        }
    }

    String::from_utf8(out).expect("blanking writes only ASCII spaces")
}

struct MatchVisitor<'a> {
    path: &'a str,
    sites: Vec<String>,
}

impl MatchVisitor<'_> {
    fn record_cfg_test_node<T: Spanned>(&mut self, _node: &T, _attrs: &[syn::Attribute]) {}

    /// Records the pattern if it routes on a forbidden kind, and says whether
    /// it did, so a `match` can stop at its first offending arm.
    fn record(&mut self, pattern: &Pat) -> bool {
        let Some(kind) = pattern_kind(pattern) else {
            return false;
        };
        self.sites
            .push(format!("{}: match on kind, arm \"{kind}\"", self.path));
        true
    }
}

impl<'ast> Visit<'ast> for MatchVisitor<'_> {
    // The shared gates. Items and impl items alone left cfg-gated statements,
    // trait items, match arms, expressions, struct-literal fields, field
    // declarations and enum variants unread.
    skip_cfg_test_nodes!();

    fn visit_expr(&mut self, node: &'ast Expr) {
        return_if_cfg_test_node!(self, node, expr_attrs(node));
        visit::visit_expr(self, node);
    }

    fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
        if expr_names_kind(&node.expr) {
            for arm in &node.arms {
                if is_cfg_test(&arm.attrs) {
                    continue;
                }
                if self.record(&arm.pat) {
                    break;
                }
            }
        }
        visit::visit_expr_match(self, node);
    }

    /// `if let "source" = page.kind.as_str()` and `while let` route on the
    /// column exactly as a `match` arm does; the pattern simply precedes its
    /// scrutinee. Free to cover once the shape is an AST node rather than text.
    fn visit_expr_let(&mut self, node: &'ast syn::ExprLet) {
        if expr_names_kind(&node.expr) {
            self.record(&node.pat);
        }
        visit::visit_expr_let(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "matches")
        {
            match node.parse_body::<MatchesInvocation>() {
                Ok(invocation) => {
                    if expr_names_kind(&invocation.scrutinee) {
                        self.record(&invocation.pattern);
                    }
                    // The scrutinee and the guard are both ordinary
                    // expressions, and either can contain a `match` of its own.
                    self.visit_expr(&invocation.scrutinee);
                    if let Some(guard) = &invocation.guard {
                        self.visit_expr(guard);
                    }
                }
                // Unreadable is not clean. A `matches!` whose pattern cannot be
                // recovered is a decision on unknown terms, and swallowing the
                // parse error is what let a trailing comma erase the call.
                Err(_) => self.sites.push(format!(
                    "{}: a matches! invocation could not be parsed, so its pattern is unknown",
                    self.path
                )),
            }
            return;
        }
        // A `match` written inside some other macro's body is still a reader.
        // Token streams are not parsed by `syn` on the way in, so recover the
        // ones that happen to be statements; the rest (`vec![1, 2]`, a format
        // string) simply do not parse and are skipped.
        if let Ok(statements) = node.parse_body_with(syn::Block::parse_within) {
            for statement in &statements {
                self.visit_stmt(statement);
            }
        }
    }
}

/// `matches!(scrutinee, pattern | pattern if guard $(,)?)`.
///
/// The guard is KEPT rather than discarded. Its own literals are still not
/// routing — they are an expression's, not a pattern's, which is why the guard
/// is never passed to `record` — but it is a whole expression, so a `match` on
/// the column can be written inside it. Parsing it and throwing it away made
/// `matches!(flag, true if match page.kind.as_str() { … })` invisible.
struct MatchesInvocation {
    scrutinee: Expr,
    pattern: Pat,
    guard: Option<Expr>,
}

impl Parse for MatchesInvocation {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let scrutinee = input.parse()?;
        input.parse::<Token![,]>()?;
        let pattern = Pat::parse_multi_with_leading_vert(input)?;
        let guard = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        // core's `matches!` ends in `$(,)?`, and `parse_body` refuses leftover
        // tokens — so an ordinary trailing comma used to fail the parse.
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }
        Ok(MatchesInvocation {
            scrutinee,
            pattern,
            guard,
        })
    }
}

/// True when a scrutinee READS the page-kind column anywhere inside it:
/// `page.kind`, `page.kind.as_str()`, `page.kind()`, a bare `kind` binding, and
/// equally the compound forms a reader actually writes —
/// `(page.kind.as_str(), active)`, `normalize(page.kind.as_str())`,
/// `{ page.kind.as_str() }`.
///
/// Searched as a tree, not as an enumerated list of wrapper nodes. The list was
/// the defect: it named `Reference`/`Paren`/`Unary`/`Try`/`Cast`/`Group` and so
/// missed `Tuple`, `Call` and `Block`, and any such list is one syntax away
/// from the next miss. Identifiers are still compared whole, so `creation_kind`
/// — a different column, carrying no fence — never matches at any depth.
///
/// The tree is walked under the SAME `cfg` gates as the file above it. A
/// scrutinee is ordinary code and can carry its own test-only branch, so
/// without them a `match` whose production scrutinee reads some other column
/// reported purely because a `#[cfg(test)]` sibling named `kind`.
fn expr_names_kind(expr: &Expr) -> bool {
    struct KindReader {
        found: bool,
    }

    impl KindReader {
        fn record_cfg_test_node<T: Spanned>(&mut self, _node: &T, _attrs: &[syn::Attribute]) {}
    }

    impl<'ast> Visit<'ast> for KindReader {
        skip_cfg_test_nodes!();

        fn visit_expr(&mut self, node: &'ast Expr) {
            if self.found {
                return;
            }
            return_if_cfg_test_node!(self, node, expr_attrs(node));
            self.found = names_kind_here(node);
            if !self.found {
                visit::visit_expr(self, node);
            }
        }
    }

    let mut reader = KindReader { found: false };
    reader.visit_expr(expr);
    reader.found
}

/// Does THIS node name `kind`, ignoring its children?
fn names_kind_here(expr: &Expr) -> bool {
    match expr {
        Expr::Field(field) => {
            matches!(&field.member, syn::Member::Named(name) if name == "kind")
        }
        Expr::MethodCall(call) => call.method == "kind",
        Expr::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "kind"),
        _ => false,
    }
}

/// The page kind a pattern routes on, at any depth of an or-pattern, a
/// reference, or a tuple — every position where the literal is still matched
/// against rather than evaluated.
fn pattern_kind(pattern: &Pat) -> Option<String> {
    match pattern {
        Pat::Lit(literal) => match &literal.lit {
            Lit::Str(text) => {
                let value = text.value();
                UNROUTABLE_PAGE_KINDS
                    .contains(&value.as_str())
                    .then_some(value)
            }
            _ => None,
        },
        Pat::Or(cases) => cases.cases.iter().find_map(pattern_kind),
        Pat::Paren(paren) => pattern_kind(&paren.pat),
        Pat::Reference(reference) => pattern_kind(&reference.pat),
        Pat::Ident(ident) => ident
            .subpat
            .as_ref()
            .and_then(|(_, subpattern)| pattern_kind(subpattern)),
        Pat::TupleStruct(tuple) => tuple.elems.iter().find_map(pattern_kind),
        Pat::Tuple(tuple) => tuple.elems.iter().find_map(pattern_kind),
        Pat::Slice(slice) => slice.elems.iter().find_map(pattern_kind),
        _ => None,
    }
}

#[test]
fn no_production_read_routes_on_a_non_entity_page_kind() {
    let root = repo_root();
    let mut sites = Vec::new();
    for path in git_ls_files(&root, "*.rs").into_iter().filter(|path| {
        path.starts_with("crates/")
            && path.contains("/src/")
            && path != "crates/wenlan-core/src/drift_guard.rs"
            && !is_test_only_module(path)
    }) {
        let source = std::fs::read_to_string(root.join(&path)).expect("read Rust source");
        sites.extend(page_kind_routing_sites(&path, &source));
    }
    assert!(
        sites.is_empty(),
        "production code now reads `pages.kind` to decide something, on a value other than \
         'entity': {sites:?}. The column is written truthfully as of migration 107, but it is \
         not yet trustworthy to read: rename, archive, and replace paths never re-derive it, \
         and migration 89 folded only creation_kind='imported' onto 'source', so a page \
         imported before 89 still carries 'concept'. Resolve by title or creation_kind the way \
         synthesis::overview does, and retire this tooth in the PR that re-derives kind on \
         every mutation path"
    );
}

#[test]
fn page_kind_routing_guard_separates_reads_from_writes() {
    let path = "crates/wenlan-core/src/somewhere.rs";

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "conn.query(\"SELECT id FROM pages WHERE kind = 'source' AND status = 'active'\", ())",
        ),
        [format!("{path}: kind = 'source'")],
        "a SELECT that filters pages on a non-entity kind is exactly what the fence forbids"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "conn.query(\"SELECT id FROM pages p JOIN page_map m ON m.page_id = p.id \
             WHERE p.kind IN ('source','overview')\", ())",
        ),
        [format!("{path}: kind IN 'source'")],
        "an IN list is a filter too, and a qualified `p.kind` is still the column"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(path, "if page.kind == \"overview\" { return Ok(()); }"),
        [format!("{path}: kind == \"overview\"")],
        "routing in Rust on a deserialized Page is the same fence"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "conn.query(\"SELECT id FROM pages WHERE status = 'active' \
             AND COALESCE(kind, 'concept') = 'source'\", ())",
        ),
        [format!("{path}: kind = 'source'")],
        "the fence's own COALESCE spelling with the literal swapped is the likeliest \
         way a reader lands here, so it must not be the one shape that slips through"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if page.kind.as_str() == \"overview\" { return Ok(()); }"
        ),
        [format!("{path}: kind == \"overview\"")],
        "`Page.kind` is a public String, so `.as_str()` is the form a reader would \
         actually write; an accessor between the column and the operator changes \
         the spelling and nothing else"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if page.kind() != \"concept\" { return Ok(()); }"
        ),
        [format!("{path}: kind != \"concept\"")],
        "a getter is the same read as the field"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let mut sql = String::from(\"SELECT id FROM pages WHERE space = ?1\"); \
             sql.push_str(\" AND kind = 'overview'\");",
        ),
        [format!("{path}: kind = 'overview'")],
        "a predicate appended to a base built elsewhere is still a predicate, and \
         requiring the `FROM pages` proof in the same literal is exactly what hid it"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(path, "sql.push_str(\" AND p.kind = 'overview'\");"),
        [format!("{path}: kind = 'overview'")],
        "a fragment is recognised by the connective it opens with, so qualifying \
         the column with its table alias cannot hide it"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "sql.push_str(\" AND COALESCE(kind, 'concept') = 'source'\");",
        ),
        [format!("{path}: kind = 'source'")],
        "the live entity fence is spelled `AND COALESCE(kind, 'concept') != 'entity'` \
         (db.rs:41949, db.rs:42055), so the appended form of it with the literal \
         swapped is the likeliest reader to grow here"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(path, "sql.push_str(\" AND (kind = 'overview')\");"),
        [format!("{path}: kind = 'overview'")],
        "parenthesising a predicate changes its punctuation, not what it decides"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let label = match page.kind.as_str() { \"source\" => \"doc\", _ => \"note\" };",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "a match arm decides on the column without ever writing an operator"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let label = match page.kind() { \"overview\" => \"home\", _ => \"note\" };",
        ),
        [format!("{path}: match on kind, arm \"overview\"")],
        "the scrutinee is matched as a whole word, so an accessor call reads the \
         same as a bare binding"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if matches!(page.kind.as_str(), \"overview\" | \"authored\") { return Ok(()); }",
        ),
        [format!("{path}: match on kind, arm \"overview\"")],
        "matches! is the same decision spelled as a macro"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let l = match page.kind.as_str() { \"source\" if active => 1, _ => 0 };",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "a guard narrows when the arm fires, so a guarded pattern routes on the \
         column at least as much as a bare one"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if matches!(page.kind.as_str(), \"source\" if active) { return Ok(()); }",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "the macro takes a guard too"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let l = match page.kind.as_str() {\n    \
             \"entity\" => if active { 0 } else { 1 }\n    \
             \"source\" => 2,\n    _ => 0,\n};",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "Rust lets an arm whose body is an expression-with-block drop its \
         trailing comma, and this repository writes that style, so an arm \
         boundary read off commas loses every arm after the first such body"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if let \"source\" = page.kind.as_str() { skip(); }"
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "an `if let` is a match arm with the pattern written first, and reading \
         the AST makes it the same site rather than a disclosed hole"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "match (page.kind.as_str(), active) { (\"source\", true) => route(), _ => {} }",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "pairing the column with a second value is ordinary Rust and this \
         repository's own style, so a scrutinee list that stopped at single-\
         value wrappers let the commonest compound form straight through"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "let l = match normalize(page.kind.as_str()) { \"overview\" => 1, _ => 0 };",
        ),
        [format!("{path}: match on kind, arm \"overview\"")],
        "passing the column through a helper does not stop the arm deciding on it"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if matches!(page.kind.as_str(), \"source\",) { return Ok(()); }",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "`matches!` ends in `$(,)?`, so a trailing comma is valid Rust — and a \
         parser that refused it while the caller swallowed the error erased \
         the whole invocation"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "if matches!(flag, true if match page.kind.as_str() { \"source\" => true, \
             _ => false }) { return Ok(()); }",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "a `matches!` guard is a whole expression and can hold a `match` of its \
         own. The guard was parsed only to be discarded, so the invocation's own \
         scrutinee — `flag`, naming nothing — was the only thing ever looked at"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "#[cfg(not(test))]\nfn f(page: &Page) -> u8 {\n    \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n}",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "`cfg(not(test))` is production-ONLY code; reading any nested `test` \
         ident as test-only skipped exactly the code this tooth exists to read"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "#[cfg(any(target_os = \"macos\", test))]\nfn f(page: &Page) -> u8 {\n    \
             match page.kind.as_str() { \"overview\" => 1, _ => 0 }\n}",
        ),
        [format!("{path}: match on kind, arm \"overview\"")],
        "this is live style in this repository (crates/wenlan-server/src/main.rs:156): \
         the function ships on macOS, and `test` only widens when it also compiles"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "#[cfg(feature = \"test-utils\")]\nfn f(page: &Page) -> u8 {\n    \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n}",
        ),
        [format!("{path}: match on kind, arm \"source\"")],
        "a feature merely NAMED after tests is compiled into production builds. \
         This control used to read `creation_kind`, so it passed whether the \
         item was scanned or skipped — it asserted nothing at all"
    );

    // The exhaustiveness claim on `skip_cfg_test_nodes!` is derived from ONE
    // pinned syn version. A bump can add an attribute-bearing node and reopen
    // the gap in silence, so the pin itself is a tooth: this fails on the bump,
    // pointing at the derivation to re-run rather than letting the claim rot.
    let lock = std::fs::read_to_string(repo_root().join("Cargo.lock"))
        .expect("Cargo.lock must be readable to check the syn pin");
    let pinned = lock
        .split("[[package]]")
        .find(|block| block.contains("\nname = \"syn\"\n"))
        .and_then(|block| {
            block
                .lines()
                .find_map(|line| line.strip_prefix("version = ").map(|v| v.trim_matches('"')))
        });
    assert_eq!(
        pinned,
        Some("2.0.117"),
        "syn moved. The gate list on `skip_cfg_test_nodes!` claims to cover every \
         attribute-bearing node in syn 2.0.117, derived by grepping the vendored \
         source for `pub attrs: Vec<Attribute>` (94 hits, 93 distinct nodes, \
         `DeriveInput` excluded as unreachable from a file walk). Re-run that \
         derivation against the new version, add any new node to the macro, and \
         update this pin — three rounds of review found this list short by a \
         shape nobody predicted, so it is not safe to assume a bump added none"
    );

    // The positions round 7 DISCLOSED as unread, now READ. Round 7 pinned them
    // as known false findings; two of those pins turned out not to compile, or
    // not to be caused by the position they named — a disclosure resting on a
    // fixture nobody built is worth no more than the argument it replaced.
    // Every fixture below was compiled with `rustc --edition 2021` in BOTH
    // configurations (with and without `--cfg test`, exit 0 each), so each is
    // real Rust whose test-only half genuinely leaves a production build.
    //
    // Each is CAUSAL: the macro sits inside the node the attribute gates, so
    // removing that gate is what brings the finding back.
    for (label, source) in [
        (
            "a function parameter, reached through `PatType` rather than `Pat`",
            "macro_rules! ignored_const { ($($tt:tt)*) => { 1 }; }\n\
             fn f(#[cfg(test)] p: [u8; ignored_const!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 })]) {}",
        ),
        (
            "a generic TYPE parameter, gated together with the field that uses it \
             so the struct compiles either way",
            "macro_rules! ignored_const { ($($tt:tt)*) => { 1 }; }\n\
             struct S<#[cfg(test)] T = [u8; ignored_const!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 })]> {\n    \
             #[cfg(test)]\n    marker: std::marker::PhantomData<T>,\n}",
        ),
        (
            "a CONST generic parameter, with the macro inside the parameter's own \
             default — round 7 put it in the return type, where no parameter gate \
             could ever have silenced it",
            "macro_rules! ignored_const { ($($tt:tt)*) => { 1 }; }\n\
             struct C<#[cfg(test)] const N: usize = { ignored_const!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 }) }> {\n    \
             #[cfg(test)]\n    bytes: [u8; N],\n}",
        ),
        (
            "a CLOSURE parameter",
            "macro_rules! ignored_const { ($($tt:tt)*) => { 1 }; }\n\
             let _c = |#[cfg(test)] p: [u8; ignored_const!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 })]| ();",
        ),
        (
            "a STRUCT-PATTERN field",
            "macro_rules! ignored_pat { ($($tt:tt)*) => { _ }; }\n\
             let S { #[cfg(test)] a: ignored_pat!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 }), .. } = s;",
        ),
    ] {
        assert_eq!(
            page_kind_routing_sites_in_snippet(path, source),
            Vec::<String>::new(),
            "{label} is a cfg-gated position holding a macro, and macro input is \
             re-parsed as statements — so leaving the position ungated reported \
             this test-only match as production. Every fixture here compiled \
             under `rustc --edition 2021` both with and without `--cfg test`; \
             two of round 7's did not, which is why they proved nothing"
        );
    }

    assert_eq!(
        page_kind_routing_sites(
            path,
            "#![cfg(test)]\nfn f(page: &Page) -> u8 {\n    \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n}",
        ),
        Vec::<String>::new(),
        "a file's own inner attribute gates every item in it, and it is the one \
         position reached through the production entry rather than a snippet. No \
         file in this tree carries one, which is why round 7 disclosed it; \
         reading it costs four lines, which is less than the disclosure did"
    );

    assert_eq!(
        page_kind_routing_sites_in_snippet(
            path,
            "macro_rules! ignored_pat { ($($tt:tt)*) => { _ }; }\n\
             let (#[cfg(test)] ignored_pat!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 }), y) = t;",
        ),
        [format!(
            "{path}: could not be parsed as Rust, so the page-kind match scan could not run"
        )],
        "an attribute on a TUPLE-pattern element is not a shape any gate could \
         reach, because it is not Rust: rustc rejects it with `expected pattern, \
         found #` in both configurations. syn refusing it mirrors the compiler, \
         and this scan reports the file as unreadable rather than passing it \
         clean. Round 7 read this single fixture as proof that PATTERN position \
         at large was unparseable; the closure-parameter and struct-pattern-field \
         controls above are the counterexamples it was missing"
    );

    // The production entry point, not the snippet one: these prove that a file
    // this scan cannot read reports itself rather than passing clean.
    assert_eq!(
        page_kind_routing_sites(path, "let x = 5;"),
        [format!(
            "{path}: could not be parsed as Rust, so the page-kind match scan could not run"
        )],
        "a fragment is not a file. This is the exact false clean the fallback \
         bought: statements parse once they are wrapped in a function, so a \
         tracked file that stopped being a file returned an empty finding list \
         — indistinguishable from a file that was read and found innocent"
    );

    assert_eq!(
        page_kind_routing_sites(path, "fn truncated(page: &Page) -> u8 { match page.kind"),
        [format!(
            "{path}: could not be parsed as Rust, so the page-kind match scan could not run"
        )],
        "a file whose readers cannot be enumerated must never read as having none"
    );

    for (label, source) in [
        (
            "the entity fence itself",
            "conn.query(\"SELECT id FROM pages WHERE kind = 'entity' AND status = 'active'\", ())",
        ),
        (
            "the entity fence in its live COALESCE spelling",
            "conn.query(\"SELECT id FROM pages WHERE status = 'active' \
             AND COALESCE(kind, 'concept') != 'entity'\", ())",
        ),
        (
            "a COALESCE projection, which decides nothing",
            "conn.query(\"SELECT COALESCE(kind, 'concept') FROM pages WHERE id = ?1\", ())",
        ),
        (
            "migration 89's CHECK constraint",
            "conn.execute(\"ALTER TABLE pages ADD COLUMN kind TEXT NOT NULL DEFAULT 'concept' \
             CHECK(kind IN ('entity','concept','source','overview','authored'))\", ())",
        ),
        (
            "migration 107's repair guard",
            "tx.execute(\"UPDATE pages SET kind = CASE WHEN creation_kind = 'authored' \
             THEN 'authored' ELSE 'concept' END WHERE kind = 'concept'\", ())",
        ),
        (
            "a different column",
            "conn.query(\"SELECT id FROM pages WHERE creation_kind = 'source'\", ())",
        ),
        (
            "a different table's kind",
            "conn.query(\"SELECT id FROM entity_links WHERE kind = 'source'\", ())",
        ),
        (
            "a fragment that names its own table",
            "sql.push_str(\" JOIN entity_links e ON e.page_id = p.id AND e.kind = 'source'\");",
        ),
        (
            "migration 107's own log line, which is prose and not a predicate",
            "log::info!(\"[migration] Migration 107 applied: {repaired} page(s) moved off the \
             silent kind='concept' default\");",
        ),
        (
            "a Rust assignment, not a comparison",
            "let mut expected = Fixture::default(); expected.kind = \"concept\";",
        ),
        (
            "prose in a comment",
            "// pre-89 imports still carry kind='concept' until M6 re-derives them",
        ),
        (
            "a bind parameter",
            "conn.query(\"SELECT id FROM pages WHERE kind = ?1\", params![kind])",
        ),
        (
            "a match on the entity fence",
            "let shadow = match page.kind.as_str() { \"entity\" => true, _ => false };",
        ),
        (
            "a match on a different column",
            "let l = match page.creation_kind.as_str() { \"source\" => 1, _ => 0 };",
        ),
        (
            "prose in an arm body, which decides nothing",
            "let l = match page.kind.as_str() { \"entity\" => bail!(\"source missing\"), \
             _ => 0 };",
        ),
        (
            "a literal in a guard expression, which is not the arm's pattern",
            "let l = match page.kind.as_str() { \"entity\" if tag == \"source\" => 1, \
             _ => 0 };",
        ),
        (
            "the same guard literal inside the macro",
            "if matches!(page.kind.as_str(), \"entity\" if tag == \"source\") { return Ok(()); }",
        ),
        (
            "a forbidden literal in an arm body, reached past a turbofish comma",
            "let l = match page.kind.as_str() { \"entity\" => make::<u8, u16>(\"source\"), \
             _ => 0 };",
        ),
        (
            "the same shape with a two-argument closure",
            "let l = match page.kind.as_str() { \"entity\" => fold(|a, b| pick(a, b, \"source\")), \
             _ => 0 };",
        ),
        (
            "a test fixture, which the scan now skips by attribute rather than by \
             blanking its lines",
            "#[cfg(test)]\nmod tests {\n    fn f(page: &Page) -> u8 {\n        \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n    }\n}",
        ),
        (
            "a cfg that REQUIRES test, whatever else it also requires",
            "#[cfg(all(test, feature = \"slow\"))]\nfn f(page: &Page) -> u8 {\n    \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n}",
        ),
        (
            "the same cfg over a COMPARISON. The lexical half judged the literal \
             text `#[cfg(test)]` while the AST half was already evaluating the \
             predicate, so one tooth gave two answers — and a text-scan finding \
             is recorded before the parser runs, so the visitor gates could not \
             retract it",
            "#[cfg(all(test, feature = \"slow\"))]\nfn f(page: &Page) {\n    \
             if page.kind.as_str() == \"source\" {}\n}",
        ),
        (
            "a cfg-gated type alias whose array length holds a macro. A const \
             position bounds what COMPILES, not what this scan reads: macro input \
             is re-parsed as statements, so the omitted `Item::Type` reported a \
             test-only match as production",
            "macro_rules! ignored_const { ($($tt:tt)*) => { 1 }; }\n\
             #[cfg(test)]\ntype T = [u8; ignored_const!(\
             match page.kind.as_str() { \"source\" => 1, _ => 0 })];",
        ),
        (
            "the same route through a cfg-gated extern block, so the fix reads as \
             a class rather than one patched variant",
            "#[cfg(test)]\nextern \"C\" {\n    \
             ignored_items!(match page.kind.as_str() { \"source\" => 1, _ => 0 });\n}",
        ),
        (
            "a cfg-gated statement, an attribute position items alone never reached",
            "#[cfg(test)]\nlet l = match page.kind.as_str() { \"source\" => 1, _ => 0 };",
        ),
        (
            "a cfg-gated trait item, likewise",
            "trait Router {\n    #[cfg(test)]\n    fn route(page: &Page) -> u8 {\n        \
             match page.kind.as_str() { \"source\" => 1, _ => 0 }\n    }\n}",
        ),
        (
            "a cfg-gated match arm, likewise",
            "let l = match tag {\n    #[cfg(test)]\n    \
             \"probe\" => match page.kind.as_str() { \"source\" => 1, _ => 0 },\n    \
             _ => 0,\n};",
        ),
        (
            "a compound scrutinee over a different column",
            "let l = match (page.creation_kind.as_str(), active) { \
             (\"source\", true) => 1, _ => 0 };",
        ),
        (
            "a cfg-gated struct-literal field, whose attribute syn keeps on the \
             field and not on the expression under it",
            "let built = Built {\n    #[cfg(test)]\n    \
             route: match page.kind.as_str() { \"source\" => 1, _ => 0 },\n    \
             name,\n};",
        ),
        (
            "a scrutinee whose only `kind` read is cfg-gated, so the value the \
             production build matches on is a different column entirely",
            "let l = match ({\n    #[cfg(test)]\n    let k = page.kind.as_str();\n    \
             #[cfg(not(test))]\n    let k = page.slug.as_str();\n    k\n}) { \
             \"source\" => 1, _ => 0 };",
        ),
        (
            "a cfg attribute written across SEVERAL LINES. The lexical stripper \
             asked whether ONE line both opened and closed an attribute until \
             round 9, so rustfmt's own wrapping of a long predicate turned a \
             test-only comparison into a production finding — and a text-scan \
             finding is recorded before the parser runs, so no visitor gate \
             could retract it",
            "#[cfg(\n    all(test, feature = \"slow\")\n)]\nfn f(page: &Page) {\n    \
             if page.kind.as_str() == \"source\" {}\n}",
        ),
        (
            "a cfg attribute SHARING ITS LINE with the item it gates, which the \
             line-shaped stripper also missed: the line started with `#[cfg(` \
             but ended with the item's brace",
            "#[cfg(all(test, feature = \"slow\"))] fn f(page: &Page) { \
             if page.kind.as_str() == \"source\" {} }",
        ),
        (
            "the same gate spelled with the whitespace Rust permits. Formatting \
             is not a semantic signal, and a stripper that reads it as one is \
             deciding what ships by how it was typed",
            "#[ cfg ( all ( test , feature = \"slow\" ) ) ]\nfn f(page: &Page) {\n    \
             if page.kind.as_str() == \"source\" {}\n}",
        ),
    ] {
        assert!(
            page_kind_routing_sites_in_snippet(path, source).is_empty(),
            "{label} is not a read routing on kind: {:?}",
            page_kind_routing_sites_in_snippet(path, source)
        );
    }
}
