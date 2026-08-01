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
    let Some(steps) = parsed["jobs"]["release"]["steps"].as_sequence() else {
        return vec!["release job has no steps".into()];
    };

    if steps.iter().any(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|uses| uses.contains("sccache-action"))
    }) {
        violations.push(
            "release tag builds install sccache despite near-zero cross-tag cache reuse".into(),
        );
    }
    let build = steps
        .iter()
        .find(|step| step["name"].as_str() == Some("Build and smoke shipped release binaries"));
    if build.is_none_or(|step| {
        step["env"]["RUSTC_WRAPPER"].as_str() == Some("sccache")
            || step["env"]["SCCACHE_GHA_ENABLED"].as_str() == Some("true")
    }) {
        violations.push("release shipped-binary build still depends on sccache GHA state".into());
    }
    let rust_cache = steps.iter().find(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
    });
    if rust_cache.is_none() {
        violations.push("release job removed its target-level rust-cache fallback".into());
    }
    let windows_condition = "matrix.target == 'x86_64-pc-windows-msvc'";
    let windows_only = "${{ matrix.target == 'x86_64-pc-windows-msvc' }}";
    if rust_cache.and_then(|step| step["with"]["shared-key"].as_str())
        != Some("release-v3-${{ matrix.target }}")
        || rust_cache.and_then(|step| step["with"]["workspaces"].as_str()) != Some(". -> target")
        || rust_cache.and_then(|step| step["with"]["cache-all-crates"].as_str())
            != Some(windows_only)
        || rust_cache.and_then(|step| step["with"]["cache-workspace-crates"].as_str())
            != Some("false")
        || rust_cache.and_then(|step| step["with"]["cache-targets"].as_str()) != Some(windows_only)
        || rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some("false")
    {
        violations.push(
            "release target cache is not restore-only, host+target coherent, and capacity-bounded to Windows"
                .into(),
        );
    }
    let marker_name = "Mark Windows explicit target as nested Cargo cache";
    let marker = steps
        .iter()
        .find(|step| step["name"].as_str() == Some(marker_name));
    let marker_run = marker
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if marker.and_then(|step| step["if"].as_str()) != Some(windows_condition)
        || !marker_run.contains("CACHEDIR.TAG")
        || !marker_run.contains("Signature: 8a477f597d28d172789f06886806bc55")
    {
        violations.push(
            "release does not create a valid nested Cargo target marker before cache restore"
                .into(),
        );
    }
    let stabilizer_index = steps.iter().position(|step| {
        step["name"].as_str() == Some("Stabilize Windows Rust cache toolchain inputs")
    });
    let marker_index = steps
        .iter()
        .position(|step| step["name"].as_str() == Some(marker_name));
    let cache_index = steps.iter().position(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
    });
    if !matches!(
        (stabilizer_index, marker_index, cache_index),
        (Some(stabilizer), Some(marker), Some(cache)) if stabilizer < marker && marker < cache
    ) {
        violations.push(
            "release does not stabilize Windows toolchain inputs and mark the nested target before cache restore"
                .into(),
        );
    }
    if parsed["jobs"]["release"]["timeout-minutes"].as_str()
        != Some("${{ matrix.target == 'x86_64-pc-windows-msvc' && 90 || 60 }}")
    {
        violations.push("release build matrix has no bounded 90/60-minute timeout".into());
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
fn release_uses_target_cache_without_tag_scoped_sccache_writes() {
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
        .expect("read release.yml");
    let violations = release_rust_cache_violations(&workflow);
    assert!(
        violations.is_empty(),
        "Release cache contract drift — release.yml must use the capacity-bounded rust-cache \
         target cache and must not depend on tag-scoped sccache writes. Fix \
         .github/workflows/release.yml; an intentional contract change also updates \
         release_rust_cache_violations() and its positive control:\n{}",
        violations.join("\n")
    );
}

#[test]
fn release_cache_contract_rejects_sccache_only_fixture() {
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
        "install sccache",
        "depends on sccache",
        "rust-cache fallback",
        "capacity-bounded",
        "stabilize Windows toolchain inputs",
        "nested Cargo target marker",
        "90/60-minute timeout",
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
    if workflows != ["CI"] || types != ["completed"] || branches != ["main"] {
        violations.push(
            "release-please is not triggered by completed main-branch CI workflow runs".into(),
        );
    }
    let trigger_names = parsed["on"]
        .as_mapping()
        .into_iter()
        .flatten()
        .filter_map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    if trigger_names != BTreeSet::from(["workflow_dispatch", "workflow_run"])
        || parsed["on"].get("push").is_some()
    {
        violations.push("release-please retains a direct push trigger or an extra trigger".into());
    }
    let condition = parsed["jobs"]["release-please"]["if"]
        .as_str()
        .unwrap_or_default();
    for required in [
        "github.event_name == 'workflow_dispatch'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.conclusion == 'success'",
    ] {
        if !condition.contains(required) {
            violations.push(format!(
                "release-please job condition omits successful main push CI proof {required:?}"
            ));
        }
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
        "completed main-branch CI",
        "direct push trigger",
        "successful main push CI proof",
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

    let release_stage =
        workflow_step_run(&release, "Bundle onnxruntime.dll (Windows)").unwrap_or_default();
    if !release_stage.contains("scripts/stage-onnxruntime-windows.ps1") {
        violations.push("release workflow does not use the pinned Windows ORT stager".into());
    }

    for (workflow, job, step_name, required_condition, owner) in [
        (
            &ci,
            "test",
            "Configure MSVC Ninja (Windows tests)",
            "matrix.os == 'windows-2022'",
            "Windows test CI",
        ),
        (
            &ci,
            "release-preflight",
            "Configure MSVC Ninja (Windows release proof)",
            "",
            "Windows release proof",
        ),
        (
            &release,
            "release",
            "Configure MSVC Ninja (Windows release)",
            "matrix.target == 'x86_64-pc-windows-msvc'",
            "Windows release workflow",
        ),
    ] {
        let step = job_step(workflow, job, step_name);
        let condition = step
            .and_then(|candidate| candidate["if"].as_str())
            .unwrap_or_default();
        let run = step
            .and_then(|candidate| candidate["run"].as_str())
            .unwrap_or_default();
        if (!required_condition.is_empty() && !condition.contains(required_condition))
            || !run.contains("scripts/setup-msvc-ninja-windows.ps1")
        {
            violations.push(format!(
                "{owner} does not configure the shared x64 MSVC Ninja environment"
            ));
        }
    }

    for (workflow, job, required_condition, owner) in [
        (
            &ci,
            "test",
            "matrix.os == 'windows-2022'",
            "Windows test CI",
        ),
        (&ci, "release-preflight", "", "Windows release proof"),
        (
            &release,
            "release",
            "matrix.os == 'windows-2022'",
            "Windows release workflow",
        ),
    ] {
        let step = job_step(workflow, job, "Set up Vulkan SDK (Windows only)");
        let condition = step
            .and_then(|candidate| candidate["if"].as_str())
            .unwrap_or_default();
        let run = step
            .and_then(|candidate| candidate["run"].as_str())
            .unwrap_or_default();
        if (!required_condition.is_empty() && !condition.contains(required_condition))
            || !run.contains("scripts/setup-vulkan-sdk-windows.ps1")
        {
            violations.push(format!(
                "{owner} does not configure the pinned Vulkan SDK in its required scope"
            ));
        }
    }
    for (workflow, job, step_name, required_condition, required_destination, owner) in [
        (
            &ci,
            "test",
            "Set up Vulkan SDK (Windows only)",
            "matrix.os == 'windows-2022'",
            "$env:RUNNER_TEMP",
            "Windows test CI",
        ),
        (
            &ci,
            "release-preflight",
            "Stage Windows release runtimes before smoke",
            "matrix.target == 'x86_64-pc-windows-msvc'",
            r#"target\${{ matrix.target }}\release"#,
            "Windows release proof",
        ),
        (
            &release,
            "release",
            "Set up Vulkan SDK (Windows only)",
            "matrix.os == 'windows-2022'",
            r#"target\${{ matrix.target }}\release"#,
            "Windows release workflow",
        ),
    ] {
        let step = job_step(workflow, job, step_name);
        let condition = step
            .and_then(|candidate| candidate["if"].as_str())
            .unwrap_or_default();
        let run = step
            .and_then(|candidate| candidate["run"].as_str())
            .unwrap_or_default();
        if !condition.contains(required_condition)
            || !run.contains("scripts/stage-vulkan-loader-windows.ps1")
            || !run.contains(required_destination)
            || run.contains("SkipAuthenticodeValidationForFixture")
        {
            violations.push(format!(
                "{owner} does not stage the pinned Vulkan loader in its required scope"
            ));
        }
    }

    let package = workflow_step_run(&release, "Package").unwrap_or_default();
    if !package.contains("wenlan-server.exe")
        || !package.contains("onnxruntime.dll")
        || !package.contains("vulkan-1.dll")
        || !package.contains("VulkanRT-License.txt")
    {
        violations.push(
            "release archive does not include the server, runtimes, and Vulkan license together"
                .into(),
        );
    }

    let universal_archive_verify = workflow_step_run(
        &release,
        "Verify release archive contains wenlan, wenlan-server, wenlan-mcp",
    )
    .unwrap_or_default();
    if !universal_archive_verify.contains("for bin in wenlan wenlan-server wenlan-mcp")
        || universal_archive_verify.contains("onnxruntime.dll")
        || universal_archive_verify.contains("vulkan-1.dll")
        || universal_archive_verify.contains("VulkanRT-License.txt")
    {
        violations.push(
            "cross-platform archive verification mixes Windows-only runtime payloads into every target"
                .into(),
        );
    }
    let windows_archive_verify = job_step(
        &release,
        "release",
        "Verify Windows release archive runtimes",
    );
    let windows_archive_verify_run = windows_archive_verify
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if windows_archive_verify.and_then(|step| step["if"].as_str())
        != Some("matrix.target == 'x86_64-pc-windows-msvc'")
        || !windows_archive_verify_run.contains("unzip -l")
        || !windows_archive_verify_run.contains("onnxruntime.dll")
        || !windows_archive_verify_run.contains("vulkan-1.dll")
        || !windows_archive_verify_run.contains("VulkanRT-License.txt")
    {
        violations.push(
            "Windows archive runtime verification is not restricted to the shipped Windows zip"
                .into(),
        );
    }

    let packaged_smoke =
        workflow_step_run(&release, "Smoke packaged Windows release").unwrap_or_default();
    if !packaged_smoke.contains("Expand-Archive")
        || !packaged_smoke.contains("Test-Path")
        || !packaged_smoke.contains("scripts/smoke-windows.ps1")
        || !packaged_smoke.contains("vulkan-1.dll")
        || !packaged_smoke.contains("VulkanRT-License.txt")
    {
        violations.push("release workflow does not smoke the extracted Windows archive".into());
    }

    let pr_build =
        workflow_step_run(&ci, "Build and smoke shipped release binaries").unwrap_or_default();
    let pr_runtime_stage =
        workflow_step_run(&ci, "Stage Windows release runtimes before smoke").unwrap_or_default();
    let pr_smoke =
        workflow_step_run(&ci, "Native ORT smoke (Windows release preflight)").unwrap_or_default();
    let windows_test_bootstrap =
        workflow_step_run(&ci, "Stage ONNX Runtime for Windows tests").unwrap_or_default();
    if !windows_test_bootstrap.contains("scripts/stage-onnxruntime-windows.ps1")
        || !windows_test_bootstrap.contains("ORT_DYLIB_PATH=")
        || !windows_test_bootstrap.contains("$env:GITHUB_ENV")
        || !windows_test_bootstrap.contains("$env:GITHUB_PATH")
    {
        violations.push(
            "Windows tests do not pin the verified ORT build path and DLL search path before inference"
                .into(),
        );
    }
    let test_steps = ci["jobs"]["test"]["steps"].as_sequence();
    let bootstrap_step = test_steps.and_then(|steps| {
        steps
            .iter()
            .find(|step| step["name"].as_str() == Some("Stage ONNX Runtime for Windows tests"))
    });
    if !bootstrap_step
        .and_then(|step| step["if"].as_str())
        .is_some_and(|condition| condition.contains("matrix.os == 'windows-2022'"))
    {
        violations.push("Windows ORT test bootstrap is not guarded for windows-2022".into());
    }
    let bootstrap_index = test_steps.and_then(|steps| {
        steps
            .iter()
            .position(|step| step["name"].as_str() == Some("Stage ONNX Runtime for Windows tests"))
    });
    let bootstrap_precedes_consumers = test_steps.is_some_and(|steps| {
        let Some(bootstrap_index) = bootstrap_index else {
            return false;
        };
        [
            "Page lint scale gate (Windows functional)",
            "Integration tests wenlan-cli + wenlan-server",
        ]
        .iter()
        .filter_map(|name| {
            steps
                .iter()
                .position(|step| step["name"].as_str() == Some(*name))
        })
        .all(|consumer_index| bootstrap_index < consumer_index)
    });
    if !bootstrap_precedes_consumers {
        violations
            .push("Windows ORT test bootstrap must run before inference-capable tests".into());
    }
    if !pr_build.contains("scripts/build-release-binaries.sh")
        || !pr_runtime_stage.contains("scripts/stage-onnxruntime-windows.ps1")
        || !pr_runtime_stage.contains("scripts/stage-vulkan-loader-windows.ps1")
        || !pr_smoke.contains("scripts/smoke-windows.ps1")
    {
        violations.push("PR CI does not build, stage, and exercise dynamic ORT on Windows".into());
    }

    let source_pin =
        workflow_step_run(&ci, "Verify CI observer and ORT contracts").unwrap_or_default();
    if !source_pin.contains("scripts/verify-ort-source-pin.py") {
        violations.push("PR CI does not verify the actual crates.io ort-sys source pin".into());
    }
    if !ci_workflow.contains("'crates/wenlan-core/Cargo.toml'") {
        violations.push("Windows CI path filter omits wenlan-core's ORT feature manifest".into());
    }

    if !smoke_script.contains("Get-Process -Id $proc.Id -Module")
        || !smoke_script.contains("onnxruntime.dll")
        || !smoke_script.contains("vulkan-1.dll")
        || !smoke_script.contains("Resolve-Path")
        || !smoke_script.contains("/api/memory/store")
        || !smoke_script.contains("chunks_created")
        || !smoke_script.contains("blue lamp adjusts ocean timepieces")
        || smoke_script.contains("$env:ORT_DYLIB_PATH")
    {
        violations.push(
            "Windows smoke does not force vector inference through the exact default-loaded ORT module"
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
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("stager")),
        "fixture must reject a missing ORT stager: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("DLL search path")),
        "fixture must reject Windows tests that can load a runner DLL: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("Vulkan loader")),
        "fixture must reject a release without a scoped Vulkan loader: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("extracted")),
        "fixture must reject an untested archive: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("vector inference")),
        "fixture must reject a smoke with no module proof: {violations:?}"
    );
}

#[test]
fn windows_ort_distribution_contract_rejects_unscoped_archive_payloads() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release.yml")
        .replace(
            "      - name: Verify Windows release archive runtimes\n        if: matrix.target == 'x86_64-pc-windows-msvc'",
            "      - name: Verify Windows release archive runtimes\n        if: always()",
        );
    let smoke = std::fs::read_to_string(root.join("scripts/smoke-windows.ps1"))
        .expect("read smoke-windows.ps1");
    let violations = windows_ort_distribution_violations(&ci, &release, &smoke);
    assert!(
        violations.iter().any(|violation| {
            violation.contains(
                "Windows archive runtime verification is not restricted to the shipped Windows zip",
            )
        }),
        "fixture must reject Windows-only payload checks on non-Windows archives: {violations:?}"
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
            .any(|violation| violation.contains("guarded for windows-2022")),
        "fixture must reject the wrong bootstrap OS gate: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|violation| violation.contains("before inference-capable tests")),
        "fixture must reject a late ORT bootstrap: {violations:?}"
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
        || platform_event
            != "${{ startsWith(github.head_ref, 'release-please--branches--') && 'release-please' || github.event_name }}"
    {
        violations.push(
            "detect-changes does not emit a separate fail-closed platform behavior plan"
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
    for contract in [
        "def build_platform_plan(",
        "if not relevant:",
        "no platform behavioral inputs changed",
        "path != \"crates/wenlan-core/src/drift_guard.rs\"",
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
    for path in [
        "crates/wenlan-core/src/community_grouping.rs",
        "crates/wenlan-core/src/community_partition.rs",
        "crates/wenlan-core/src/edge_grounding.rs",
        "crates/wenlan-core/src/provenance.rs",
        "crates/wenlan-core/src/db.rs",
        "crates/wenlan-core/src/db/community_grouping_state.rs",
        "crates/wenlan-core/src/refinery/mod.rs",
        "crates/wenlan-core/tests/m4_community_gates.rs",
        ".config/nextest.toml",
    ] {
        if !macos_m4.contains(path) {
            violations.push(format!("macos-m4 routing omits M4 contract input {path}"));
        }
    }
    for path in &macos_m4 {
        if !filter_routes_path(&macos_paths, path) {
            violations.push(format!(
                "M4-sensitive path does not also schedule the macOS job: {path}"
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
            != Some("needs.detect-changes.outputs.workspace-platform == 'true'")
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
    let platform_sensitive_paths = platform_sensitive_paths(&root);
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
        proof_test, "python3 scripts/verify-release-merge.test.py",
        "release proof tests must run before routing"
    );
    assert_eq!(
        proof["id"].as_str(),
        Some("release-proof"),
        "release proof output owner"
    );
    let proof_run = proof["run"].as_str().unwrap_or_default();
    for required in [
        "python3 scripts/verify-release-merge.py",
        "--github-output \"$GITHUB_OUTPUT\"",
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

    let producer = job_step_using(&ci, "detect-changes", "actions/upload-artifact")
        .expect("detect-changes must publish the prepared FastEmbed snapshot");
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

// ── Teeth #9: release preflight mirrors every shipped target without publishing ──

fn release_preflight_contract_violations(ci_workflow: &str, release_workflow: &str) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let release: serde_yaml::Value =
        serde_yaml::from_str(release_workflow).expect("parse release.yml");
    let mut violations = Vec::new();

    let inventory_guard = job_step(&ci, "detect-changes", "Reject truncated PR file inventory");
    let inventory_guard_run = inventory_guard
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if inventory_guard.and_then(|step| step["if"].as_str())
        != Some("github.event_name == 'pull_request'")
        || inventory_guard.and_then(|step| step["env"]["CHANGED_FILE_COUNT"].as_str())
            != Some("${{ github.event.pull_request.changed_files }}")
        || !inventory_guard_run.contains("-gt 3000")
        || !inventory_guard_run.contains("cannot route fail-closed")
    {
        violations.push(
            "detect-changes does not reject the REST API's truncated PR file inventory".into(),
        );
    }
    let dispatch_guard = job_step(
        &release,
        "prepare-release",
        "Require main workflow for manual release",
    );
    let dispatch_guard_run = dispatch_guard
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if dispatch_guard.and_then(|step| step["if"].as_str())
        != Some("github.event_name == 'workflow_dispatch'")
        || dispatch_guard.and_then(|step| step["env"]["WORKFLOW_REF"].as_str())
            != Some("${{ github.ref }}")
        || !dispatch_guard_run.contains("$WORKFLOW_REF")
        || !dispatch_guard_run.contains("refs/heads/main")
        || !dispatch_guard_run.contains("exit 1")
    {
        violations
            .push("manual release dispatch does not require the current main workflow ref".into());
    }

    if ci["jobs"]["detect-changes"]["outputs"]["release-targets"]
        .as_str()
        .is_none()
        || ci["jobs"]["detect-changes"]["outputs"]["release-preflight-targets"].as_str()
            != Some("${{ steps.release-preflight-targets.outputs.release-preflight-targets }}")
    {
        violations.push(
            "detect-changes does not expose canonical and bounded PR release matrices".into(),
        );
    }
    let bounded_matrix = job_step(
        &ci,
        "detect-changes",
        "Emit bounded PR release preflight matrix",
    );
    let bounded_run = bounded_matrix
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if bounded_matrix.and_then(|step| step["id"].as_str()) != Some("release-preflight-targets")
        || !bounded_run.contains("--output-name release-preflight-targets")
        || !bounded_run.contains("--exclude-target x86_64-pc-windows-msvc")
        || !bounded_run.contains("$GITHUB_EVENT_NAME")
        || !bounded_run.contains("$GITHUB_HEAD_REF")
        || !bounded_run.contains("release-please--branches--*")
    {
        violations.push(
            "ordinary PR release matrix does not exclude only the duplicate Windows release profile"
                .into(),
        );
    }
    for (job, test_name) in [
        ("detect-changes", "Test release target inventory"),
        ("prepare-release", "Test release target inventory"),
    ] {
        let workflow = if job == "detect-changes" {
            &ci
        } else {
            &release
        };
        let run = job_step(workflow, job, test_name)
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        let prefix = if job == "detect-changes" {
            "scripts"
        } else {
            ".release-tools/scripts"
        };
        if !run.contains(&format!("python3 {prefix}/release_targets.test.py"))
            || !run.contains(&format!("bash {prefix}/build-release-binaries.test.sh"))
        {
            violations.push(format!("{job} does not test the shared release inventory"));
        }
    }
    for (workflow, job, expected_output) in [
        (
            &ci,
            "detect-changes",
            "${{ steps.release-targets.outputs.release-targets }}",
        ),
        (
            &release,
            "prepare-release",
            "${{ steps.release-targets.outputs.release-targets }}",
        ),
    ] {
        if workflow["jobs"][job]["outputs"]["release-targets"].as_str() != Some(expected_output) {
            violations.push(format!("{job} does not expose the shared release matrix"));
        }
        let step = job_step(workflow, job, "Emit release target matrix");
        let command = if job == "detect-changes" {
            "python3 scripts/release_targets.py matrix --github-output \"$GITHUB_OUTPUT\""
        } else {
            "python3 .release-tools/scripts/release_targets.py matrix --github-output \"$GITHUB_OUTPUT\""
        };
        if step.and_then(|candidate| candidate["id"].as_str()) != Some("release-targets")
            || step.and_then(|candidate| candidate["run"].as_str()) != Some(command)
        {
            violations.push(format!("{job} does not emit the shared release matrix"));
        }
    }
    for job in ["prepare-release", "release"] {
        let tooling = job_step(&release, job, "Checkout release tooling");
        if tooling
            .and_then(|step| step["uses"].as_str())
            .is_none_or(|uses| !uses.starts_with("actions/checkout@"))
            || tooling.and_then(|step| step["with"]["ref"].as_str())
                != Some("${{ github.workflow_sha }}")
            || tooling.and_then(|step| step["with"]["path"].as_str()) != Some(".release-tools")
        {
            violations.push(format!(
                "{job} cannot rerun historical tags with workflow-pinned release tooling"
            ));
        }
    }

    let job = &ci["jobs"]["release-preflight"];
    if job_needs(&ci, "release-preflight") != ["detect-changes"]
        || job["if"].as_str().is_none_or(|condition| {
            !condition.contains("github.event_name != 'pull_request'")
                || !condition.contains("startsWith(github.head_ref, 'release-please--branches--')")
                || !condition.contains("needs.detect-changes.outputs.release-preflight == 'true'")
        })
    {
        violations.push(
            "release-preflight is not isolated to release-sensitive PRs and release backstops"
                .into(),
        );
    }
    if job["runs-on"].as_str() != Some("${{ matrix.os }}")
        || job["timeout-minutes"].as_str()
            != Some(
                "${{ matrix.target == 'x86_64-pc-windows-msvc' && 90 || (github.event_name != 'pull_request' && 60 || 45) }}",
            )
        || job["strategy"]["fail-fast"].as_bool() != Some(true)
        || job["strategy"]["matrix"].as_str()
            != Some("${{ fromJSON(needs.detect-changes.outputs.release-preflight-targets) }}")
    {
        violations.push(
            "release-preflight is not a fail-fast bounded ordinary-PR matrix with a full four-target backstop and cold-cache safety ceiling"
                .into(),
        );
    }
    if release["jobs"]["release"]["strategy"]["matrix"].as_str()
        != Some("${{ fromJSON(needs.prepare-release.outputs.release-targets) }}")
    {
        violations.push("tag release does not consume the canonical release matrix".into());
    }
    if release["jobs"]["release"]["timeout-minutes"].as_str()
        != Some("${{ matrix.target == 'x86_64-pc-windows-msvc' && 90 || 60 }}")
    {
        violations
            .push("tag release does not enforce the bounded 90/60-minute matrix timeout".into());
    }
    if job_needs(&release, "docker") != ["prepare-release"]
        || job_needs(&release, "docker-manifest")
            != [
                "docker",
                "release",
                "publish-crates",
                "publish-npm",
                "update-homebrew",
            ]
        || job_needs(&release, "finalize-release") != ["docker-manifest"]
    {
        violations.push(
            "Docker DAG does not build per-arch images after prepare-release, gate public manifests on every publish channel, and gate final release promotion on that manifest barrier"
                .into(),
        );
    }

    for (workflow, job_name, expected) in [
        (
            &ci,
            "release-preflight",
            "bash scripts/build-release-binaries.sh \"${{ matrix.target }}\"",
        ),
        (
            &release,
            "release",
            "bash .release-tools/scripts/build-release-binaries.sh \"${{ matrix.target }}\"",
        ),
    ] {
        let build = job_step(
            workflow,
            job_name,
            "Build and smoke shipped release binaries",
        )
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
        if build != expected {
            violations.push(format!(
                "{job_name} does not use the shared shipped-binary build contract"
            ));
        }
    }
    if job_step(
        &release,
        "release",
        "Build and smoke shipped release binaries",
    )
    .and_then(|step| step["env"]["WENLAN_REPO_ROOT"].as_str())
        != Some("${{ github.workspace }}")
    {
        violations.push(
            "tag release does not run workflow-pinned tooling against the tag checkout".into(),
        );
    }

    let toolchain = job_step_using(&ci, "release-preflight", "dtolnay/rust-toolchain");
    if toolchain.and_then(|step| step["with"]["toolchain"].as_str()) != Some("1.95.0")
        || toolchain.and_then(|step| step["with"]["targets"].as_str())
            != Some("${{ matrix.target }}")
    {
        violations.push("release-preflight does not install the matrix target".into());
    }
    let windows_condition = "matrix.target == 'x86_64-pc-windows-msvc'";
    for step_name in [
        "Stabilize Windows Rust cache toolchain inputs",
        "Configure rust-lld linker (Windows release preflight)",
        "Mark Windows explicit target as nested Cargo cache",
        "Install sqlite3 (Windows only)",
        "Stage Windows release runtimes before smoke",
        "Native ORT smoke (Windows release preflight)",
    ] {
        if job_step(&ci, "release-preflight", step_name).and_then(|step| step["if"].as_str())
            != Some(windows_condition)
        {
            violations.push(format!(
                "{step_name} is not restricted to the shipped Windows target"
            ));
        }
    }
    for (workflow, job_name, step_name, owner) in [
        (
            &ci,
            "release-preflight",
            "Configure rust-lld linker (Windows release preflight)",
            "release-preflight",
        ),
        (
            &release,
            "release",
            "Configure rust-lld linker (Windows)",
            "tag release",
        ),
    ] {
        let linker = job_step(workflow, job_name, step_name)
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        if !linker.contains("rust-lld.exe")
            || !linker.contains("CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=")
            || !linker.contains("RUSTFLAGS=")
            || !linker.contains("$env:GITHUB_ENV")
        {
            violations.push(format!(
                "{owner} does not configure rust-lld for target and host build artifacts on Windows"
            ));
        }
    }
    let sqlite = job_step(&ci, "release-preflight", "Install sqlite3 (Windows only)")
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !sqlite.contains("vcpkg install sqlite3:x64-windows-static-md")
        || !sqlite.contains("$env:GITHUB_ENV")
    {
        violations.push("release-preflight does not link the Windows sqlite dependency".into());
    }
    for (workflow, job_name) in [(&ci, "release-preflight"), (&release, "release")] {
        let native_perl = job_step(
            workflow,
            job_name,
            "Select native Perl for vendored OpenSSL",
        );
        let native_perl_run = native_perl
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        if native_perl.and_then(|step| step["if"].as_str()) != Some(windows_condition)
            || native_perl.and_then(|step| step["shell"].as_str()) != Some("pwsh")
            || !native_perl_run.contains("Get-Command perl.exe")
            || !native_perl_run.contains("-All")
            || !native_perl_run.contains("candidate.Source -match")
            || !native_perl_run.contains("[\\\\/]Git[\\\\/]")
            || !native_perl_run.contains("Locale::Maketext::Simple")
            || !native_perl_run.contains("OPENSSL_SRC_PERL=")
            || !native_perl_run.contains("$env:GITHUB_ENV")
        {
            violations.push(format!(
                "{job_name} does not select and validate native Windows Perl before vendored OpenSSL"
            ));
        }
        let steps = workflow["jobs"][job_name]["steps"].as_sequence();
        let native_perl_index = steps.and_then(|items| {
            items.iter().position(|step| {
                step["name"].as_str() == Some("Select native Perl for vendored OpenSSL")
            })
        });
        let build_index = steps.and_then(|items| {
            items.iter().position(|step| {
                step["name"].as_str() == Some("Build and smoke shipped release binaries")
            })
        });
        if !matches!(
            (native_perl_index, build_index),
            (Some(native_perl_index), Some(build_index)) if native_perl_index < build_index
        ) {
            violations.push(format!(
                "{job_name} selects native Windows Perl after the release build"
            ));
        }
    }
    let marker_name = "Mark Windows explicit target as nested Cargo cache";
    let mut marker_runs = Vec::new();
    for (workflow, job_name, owner) in [
        (&ci, "release-preflight", "release-preflight"),
        (&release, "release", "tag release"),
    ] {
        let marker = job_step(workflow, job_name, marker_name);
        let marker_run = marker
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        if marker.and_then(|step| step["if"].as_str()) != Some(windows_condition)
            || !marker_run.contains("CACHEDIR.TAG")
            || !marker_run.contains("Signature: 8a477f597d28d172789f06886806bc55")
        {
            violations.push(format!(
                "{owner} does not create a valid nested Cargo target marker"
            ));
        }
        let steps = workflow["jobs"][job_name]["steps"].as_sequence();
        let marker_index = steps.and_then(|items| {
            items
                .iter()
                .position(|step| step["name"].as_str() == Some(marker_name))
        });
        let cache_index = steps.and_then(|items| {
            items.iter().position(|step| {
                step["uses"]
                    .as_str()
                    .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
            })
        });
        if !matches!(
            (marker_index, cache_index),
            (Some(marker), Some(cache)) if marker < cache
        ) {
            violations.push(format!(
                "{owner} creates the nested target marker after cache restore"
            ));
        }
        marker_runs.push(marker_run);
    }
    if marker_runs.len() != 2 || marker_runs[0] != marker_runs[1] {
        violations.push("producer/consumer nested target marker parity has drifted".into());
    }
    let cache = job_step_using(&ci, "release-preflight", "Swatinem/rust-cache");
    if cache.and_then(|step| step["with"]["shared-key"].as_str())
        != Some("release-v3-${{ matrix.target }}")
        || cache.and_then(|step| step["with"]["cache-all-crates"].as_str()) != Some("true")
        || cache.and_then(|step| step["with"]["cache-workspace-crates"].as_str()) != Some("false")
        || cache.and_then(|step| step["with"]["cache-targets"].as_str())
            != Some("${{ matrix.target == 'x86_64-pc-windows-msvc' }}")
        || cache.and_then(|step| step["with"]["save-if"].as_str())
            != Some("${{ github.ref == 'refs/heads/main' }}")
    {
        violations.push(
            "release-preflight cache is not host+target coherent, capacity-bounded, and main-owned"
                .into(),
        );
    }
    if cache.and_then(|step| step["with"]["workspaces"].as_str()) != Some(". -> target") {
        violations.push(
            "release-preflight cache must use one top-level target root; overlapping cache roots are forbidden"
                .into(),
        );
    }
    let release_cache = job_step_using(&release, "release", "Swatinem/rust-cache");
    let windows_cache = "${{ matrix.target == 'x86_64-pc-windows-msvc' }}";
    if release_cache.and_then(|step| step["with"]["shared-key"].as_str())
        != cache.and_then(|step| step["with"]["shared-key"].as_str())
        || release_cache.and_then(|step| step["with"]["cache-all-crates"].as_str())
            != Some(windows_cache)
        || release_cache.and_then(|step| step["with"]["cache-workspace-crates"].as_str())
            != Some("false")
        || release_cache.and_then(|step| step["with"]["cache-targets"].as_str())
            != Some(windows_cache)
        || release_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some("false")
    {
        violations.push(
            "tag release cache does not restore the main-owned preflight key with a Windows-only footprint"
                .into(),
        );
    }
    if release_cache.and_then(|step| step["with"]["workspaces"].as_str()) != Some(". -> target") {
        violations.push(
            "tag release cache must use one top-level target root; overlapping cache roots are forbidden"
                .into(),
        );
    }
    if release_cache.and_then(|step| step["with"]["shared-key"].as_str())
        != cache.and_then(|step| step["with"]["shared-key"].as_str())
        || release_cache.and_then(|step| step["with"]["workspaces"].as_str())
            != cache.and_then(|step| step["with"]["workspaces"].as_str())
    {
        violations.push("release cache producer/consumer parity has drifted".into());
    }
    let windows_runtime_stage = job_step(
        &ci,
        "release-preflight",
        "Stage Windows release runtimes before smoke",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !windows_runtime_stage.contains("scripts/stage-onnxruntime-windows.ps1")
        || !windows_runtime_stage.contains("scripts/stage-vulkan-loader-windows.ps1")
        || !windows_runtime_stage.contains(r"target\${{ matrix.target }}\release")
    {
        violations.push(
            "Windows release preflight omits adjacent runtime DLL staging before executable smoke"
                .into(),
        );
    }
    let windows_smoke = job_step(
        &ci,
        "release-preflight",
        "Native ORT smoke (Windows release preflight)",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !windows_smoke.contains("scripts/smoke-windows.ps1")
        || !windows_smoke.contains(r"target\${{ matrix.target }}\release")
    {
        violations.push("Windows release preflight omits the native ORT smoke".into());
    }
    let ci_steps = job["steps"].as_sequence();
    let runtime_stage_index = ci_steps.and_then(|steps| {
        steps.iter().position(|step| {
            step["name"].as_str() == Some("Stage Windows release runtimes before smoke")
        })
    });
    let build_index = ci_steps.and_then(|steps| {
        steps.iter().position(|step| {
            step["name"].as_str() == Some("Build and smoke shipped release binaries")
        })
    });
    let smoke_index = ci_steps.and_then(|steps| {
        steps.iter().position(|step| {
            step["name"].as_str() == Some("Native ORT smoke (Windows release preflight)")
        })
    });
    if !matches!(
        (runtime_stage_index, build_index, smoke_index),
        (Some(runtime_stage_index), Some(build_index), Some(smoke_index))
            if runtime_stage_index < build_index && build_index < smoke_index
    ) {
        violations.push(
            "Windows release preflight does not stage runtime DLLs before executable smoke".into(),
        );
    }
    let release_steps = release["jobs"]["release"]["steps"].as_sequence();
    let release_ort_index = release_steps.and_then(|steps| {
        steps
            .iter()
            .position(|step| step["name"].as_str() == Some("Bundle onnxruntime.dll (Windows)"))
    });
    let release_vulkan_index = release_steps.and_then(|steps| {
        steps
            .iter()
            .position(|step| step["name"].as_str() == Some("Set up Vulkan SDK (Windows only)"))
    });
    let release_build_index = release_steps.and_then(|steps| {
        steps.iter().position(|step| {
            step["name"].as_str() == Some("Build and smoke shipped release binaries")
        })
    });
    if !matches!(
        (release_ort_index, release_vulkan_index, release_build_index),
        (Some(ort_index), Some(vulkan_index), Some(build_index))
            if ort_index < build_index && vulkan_index < build_index
    ) {
        violations.push("tag release does not stage runtime DLLs before executable smoke".into());
    }

    for step in job["steps"].as_sequence().into_iter().flatten() {
        let name = step["name"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let run = step["run"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let uses = step["uses"]
            .as_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if name.contains("package")
            || name.contains("publish")
            || run.contains("gh release")
            || run.contains("npm publish")
            || uses.contains("upload-artifact")
        {
            violations
                .push("release-preflight contains a publishing or packaging side effect".into());
        }
    }

    if !job_needs(&ci, "conclusion")
        .iter()
        .any(|need| need == "release-preflight")
    {
        violations.push("conclusion.needs omits release-preflight".into());
    }
    let conclusion = workflow_step_run(&ci, "Aggregate expected CI results").unwrap_or_default();
    if !conclusion.lines().map(str::trim).any(|line| {
        line.starts_with("expect_job release-preflight ")
            && line.contains("startsWith(github.head_ref, 'release-please--branches--')")
            && line.contains("needs.detect-changes.outputs.release-preflight")
            && line.contains("needs.release-preflight.result")
    }) {
        violations.push("conclusion does not fail closed on release-preflight".into());
    }

    violations
}

#[test]
fn release_preflight_is_release_gated_and_read_only() {
    let root = repo_root();
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let release =
        std::fs::read_to_string(root.join(".github/workflows/release.yml")).expect("read release");
    let violations = release_preflight_contract_violations(&ci, &release);
    assert!(
        violations.is_empty(),
        "release-preflight contract drift — preflight must mirror every shipped release \
         target without publishing side effects, and run only for release-sensitive changes. \
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
    let violations = release_preflight_contract_violations(&ci, &release);
    for expected in [
        "release-sensitive PRs and release backstops",
        "bounded ordinary-PR matrix",
        "canonical release matrix",
        "bounded 90/60-minute",
        "Docker DAG",
        "shared shipped-binary",
        "current main workflow ref",
        "host+target coherent, capacity-bounded, and main-owned",
        "Windows-only footprint",
        "nested Cargo target marker",
        "producer/consumer",
        "truncated PR file inventory",
        "native Windows Perl",
        "target and host build artifacts",
        "runtime DLLs before executable smoke",
        "native ORT smoke",
        "publishing or packaging side effect",
        "conclusion does not fail closed",
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

fn windows_native_parallelism_violations(
    ci_workflow: &str,
    release_workflow: &str,
    msvc_setup: &str,
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(ci_workflow).expect("parse ci.yml");
    let release: serde_yaml::Value =
        serde_yaml::from_str(release_workflow).expect("parse release.yml");
    let mut violations = Vec::new();
    let windows_condition = "matrix.target == 'x86_64-pc-windows-msvc'";
    for (workflow, job, step_name, owner) in [
        (
            &ci,
            "release-preflight",
            "Bound native build concurrency (Windows)",
            "release preflight",
        ),
        (
            &release,
            "release",
            "Cap Windows Cargo parallelism",
            "tag release",
        ),
    ] {
        let step = job_step(workflow, job, step_name);
        let run = step
            .and_then(|candidate| candidate["run"].as_str())
            .unwrap_or_default();
        if step.and_then(|candidate| candidate["if"].as_str()) != Some(windows_condition)
            || !run.contains("CARGO_BUILD_JOBS=2")
            || run.contains("CARGO_BUILD_JOBS=1")
        {
            violations.push(format!(
                "{owner} does not cap outer Windows Cargo work at exactly two jobs"
            ));
        }
        let steps = workflow["jobs"][job]["steps"].as_sequence();
        let cap_index = steps.and_then(|items| {
            items
                .iter()
                .position(|candidate| candidate["name"].as_str() == Some(step_name))
        });
        let build_index = steps.and_then(|items| {
            items.iter().position(|candidate| {
                candidate["name"].as_str() == Some("Build and smoke shipped release binaries")
            })
        });
        if !matches!((cap_index, build_index), (Some(cap), Some(build)) if cap < build) {
            violations.push(format!("{owner} applies its Cargo cap after the build"));
        }
    }
    if msvc_setup
        .matches("$env:CMAKE_BUILD_PARALLEL_LEVEL = \"1\"")
        .count()
        != 1
        || !msvc_setup.contains("\"CMAKE_BUILD_PARALLEL_LEVEL\"")
        || msvc_setup.contains("$env:CMAKE_BUILD_PARALLEL_LEVEL = \"2\"")
        || msvc_setup.contains("CARGO_BUILD_JOBS")
    {
        violations.push(
            "MSVC Ninja setup does not keep nested CMake at one worker independently of Cargo"
                .into(),
        );
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
    for expected in ["exactly two jobs", "after the build", "nested CMake"] {
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
    if lint_upload.and_then(|step| step["if"].as_str())
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

/// Blank out `#[cfg(test)]`-gated items so a test fixture sitting beside
/// production code is not mistaken for a production writer. Brace-balanced
/// rather than truncating at the first marker, because the gate also appears
/// as a statement *inside* production functions (`page_drafts.rs` gates a test
/// hook two statements above the real Page-draft INSERT) — truncating there
/// would blind the scan to the very write it exists to check. Line count is
/// preserved so reported offsets still line up with the file.
fn strip_cfg_test_items(source: &str) -> String {
    let mut kept: Vec<&str> = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "#[cfg(test)]" {
            kept.push(line);
            continue;
        }
        kept.push("");
        let mut depth = 0usize;
        let mut opened = false;
        for gated in lines.by_ref() {
            kept.push("");
            depth += gated.matches('{').count();
            depth -= gated.matches('}').count().min(depth);
            opened |= gated.contains('{');
            // A braced item ends when its braces balance; an attribute on a
            // plain `use`/`const` statement ends at the first `;`.
            if (opened && depth == 0) || (!opened && gated.trim_end().ends_with(';')) {
                break;
            }
        }
    }
    kept.join("\n")
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

/// Production `INSERT INTO pages` statements whose column list does not name
/// `kind`, reported as `path: <column list>`.
///
/// `pages.kind` (migration 89) is `NOT NULL DEFAULT 'concept'`, so omitting it
/// does not fail the insert — it silently asserts the row is a concept page.
/// That default is exactly how every page written after migration 89 came to
/// lie about what it is, the reserved Overview singleton included. The column
/// list is the only place the omission is visible, so that is what is read.
fn page_insert_sites_without_kind(path: &str, source: &str) -> Vec<String> {
    const NEEDLE: &str = "INSERT INTO pages";
    let production = strip_cfg_test_items(source);
    let mut sites = Vec::new();
    let mut rest = production.as_str();
    while let Some(offset) = rest.find(NEEDLE) {
        let after = &rest[offset + NEEDLE.len()..];
        rest = after;
        // `pages_fts`, `pages_pre49`, … are other tables entirely.
        if after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
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
/// `#[cfg(test)] mod …;` declaration in the parent — a convention teeth #16
/// (`db_main_tests_live_outside_db_rs`) already enforces. The gate is therefore
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
