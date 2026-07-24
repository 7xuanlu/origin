//! Fail-loud drift guards (test-only). Each `#[test]` here is a CI + pre-push gate
//! that makes a class of doc/flag/config drift structurally hard. Mirrors the
//! `seed_contract.rs` teeth pattern. See docs/superpowers/specs/2026-06-19-drift-defense-system-design.md.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
        "version drift across release-please files: {sources:?} (expected all == {first})"
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

// ── Teeth #5: FastEmbed CI cache contract ──

fn fastembed_ci_cache_violations(workflow: &str) -> Vec<String> {
    const CACHE_STEP: &str = "Cache fastembed model (Linux)";
    const CACHE_DIR: &str = "${{ github.workspace }}/.fastembed_cache";
    const CACHE_PATH: &str = "${{ env.FASTEMBED_CACHE_DIR }}";
    const CACHE_KEY: &str = "fastembed-bge-base-en-v1.5-q-v2";
    const JOBS: &[(&str, &str)] = &[
        ("test", "Workspace lib tests (Linux)"),
        (
            "test-quarantine",
            "Quarantined tests (wenlan-mcp + wenlan-types)",
        ),
    ];

    let parsed: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse ci.yml");
    let mut violations = Vec::new();

    for (job_name, consumer_name) in JOBS {
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
        let cache_indexes: Vec<usize> = steps
            .iter()
            .enumerate()
            .filter_map(|(index, step)| {
                (step["name"].as_str() == Some(CACHE_STEP)).then_some(index)
            })
            .collect();
        if cache_indexes.len() != 1 {
            violations.push(format!(
                "job {job_name} has {} {CACHE_STEP:?} steps, expected 1",
                cache_indexes.len()
            ));
            continue;
        }

        let cache_index = cache_indexes[0];
        let actual_path = steps[cache_index]["with"]["path"].as_str();
        if actual_path != Some(CACHE_PATH) {
            violations.push(format!(
                "job {job_name} caches {actual_path:?}, expected {CACHE_PATH:?}"
            ));
        }
        let actual_key = steps[cache_index]["with"]["key"].as_str();
        if actual_key != Some(CACHE_KEY) {
            violations.push(format!(
                "job {job_name} uses FastEmbed cache key {actual_key:?}, expected {CACHE_KEY:?}"
            ));
        }

        let consumer_index = steps
            .iter()
            .position(|step| step["name"].as_str() == Some(consumer_name));
        match consumer_index {
            Some(index) if cache_index < index => {}
            Some(index) => violations.push(format!(
                "job {job_name} restores FastEmbed at step {cache_index} after consumer step {index}"
            )),
            None => violations.push(format!(
                "job {job_name} is missing consumer step {consumer_name:?}"
            )),
        }
    }

    violations
}

fn coverage_fastembed_cache_violations(workflow: &str) -> Vec<String> {
    const CACHE_STEP: &str = "Cache fastembed model (Linux)";
    const CACHE_DIR: &str = "${{ github.workspace }}/.fastembed_cache";
    const CACHE_PATH: &str = "${{ env.FASTEMBED_CACHE_DIR }}";
    const CACHE_KEY: &str = "fastembed-bge-base-en-v1.5-q-v2";

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
        .find(|step| step["name"].as_str() == Some("Build"));
    if build.is_none_or(|step| {
        step["env"]["RUSTC_WRAPPER"].as_str() == Some("sccache")
            || step["env"]["SCCACHE_GHA_ENABLED"].as_str() == Some("true")
    }) {
        violations.push("release Build still depends on sccache GHA state".into());
    }
    if !steps.iter().any(|step| {
        step["uses"]
            .as_str()
            .is_some_and(|uses| uses.contains("Swatinem/rust-cache"))
    }) {
        violations.push("release job removed its target-level rust-cache fallback".into());
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
fn fastembed_ci_cache_is_restored_before_model_consumers() {
    let workflow =
        std::fs::read_to_string(repo_root().join(".github/workflows/ci.yml")).expect("read ci.yml");
    let violations = fastembed_ci_cache_violations(&workflow);
    assert!(
        violations.is_empty(),
        "FastEmbed CI cache contract drift:\n{}",
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
        "Coverage FastEmbed cache contract drift:\n{}",
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
      - name: Cache fastembed model (Linux)
        with:
          path: ~/.local/share/wenlan/memorydb/fastembed_cache
          key: fastembed-bge-base-en-v1.5-q-v1
"#;
    let violations = coverage_fastembed_cache_violations(workflow);
    for expected in ["FASTEMBED_CACHE_DIR", "coverage caches", "cache key"] {
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
        "Coverage execution contract drift:\n{}",
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
    for expected in ["cache writes", "exactly once", "report-only commands"] {
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
        "Release cache contract drift:\n{}",
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
      - name: Build
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
fn nextest_does_not_serialize_the_entire_core_package() {
    let config = std::fs::read_to_string(repo_root().join(".config/nextest.toml"))
        .expect("read nextest.toml");
    let violations = nextest_whole_core_serialization_violations(&config);
    assert!(
        violations.is_empty(),
        "nextest whole-package serialization contract drift:\n{}",
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
      - name: Workspace lib tests (Linux)
        run: export WENLAN_TEST_FASTEMBED_CACHE=/tmp/stale-cache
      - name: Cache fastembed model (Linux)
        with:
          path: ~/.local/share/wenlan/memorydb/fastembed_cache
  test-quarantine:
    env:
      FASTEMBED_CACHE_DIR: ${{ github.workspace }}/.fastembed_cache
    steps:
      - name: Cache fastembed model (Linux)
        with:
          path: ${{ env.FASTEMBED_CACHE_DIR }}
      - name: Quarantined tests (wenlan-mcp + wenlan-types)
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
            .any(|violation| violation.contains("cache key")),
        "fixture must reject a missing or stale cache key: {violations:?}"
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
        "Windows ONNX Runtime release contract drift:\n{}",
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

    let package = workflow_step_run(&release, "Package").unwrap_or_default();
    if !package.contains("wenlan-server.exe") || !package.contains("onnxruntime.dll") {
        violations.push("release archive does not include the server and ORT DLL together".into());
    }

    let packaged_smoke =
        workflow_step_run(&release, "Smoke packaged Windows release").unwrap_or_default();
    if !packaged_smoke.contains("Expand-Archive")
        || !packaged_smoke.contains("Test-Path")
        || !packaged_smoke.contains("scripts/smoke-windows.ps1")
    {
        violations.push("release workflow does not smoke the extracted Windows archive".into());
    }

    let pr_build = workflow_step_run(&ci, "Build Windows release binaries").unwrap_or_default();
    let pr_smoke =
        workflow_step_run(&ci, "Native ORT smoke (Windows; release profile)").unwrap_or_default();
    let windows_test_bootstrap =
        workflow_step_run(&ci, "Stage ONNX Runtime for Windows tests").unwrap_or_default();
    if !windows_test_bootstrap.contains("scripts/stage-onnxruntime-windows.ps1")
        || !windows_test_bootstrap.contains("ORT_DYLIB_PATH=")
        || !windows_test_bootstrap.contains("$env:GITHUB_ENV")
    {
        violations.push(
            "Windows tests do not pin ORT_DYLIB_PATH to the verified runtime before inference"
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
    if !pr_build.contains("cargo build --release")
        || !pr_smoke.contains("scripts/stage-onnxruntime-windows.ps1")
        || !pr_smoke.contains("scripts/smoke-windows.ps1")
    {
        violations.push("PR CI does not build, stage, and exercise dynamic ORT on Windows".into());
    }

    let source_pin = workflow_step_run(&ci, "Verify ort-sys source pin").unwrap_or_default();
    if !source_pin.contains("scripts/verify-ort-source-pin.py") {
        violations.push("PR CI does not verify the actual crates.io ort-sys source pin".into());
    }
    if !ci_workflow.contains("'crates/wenlan-core/Cargo.toml'") {
        violations.push("Windows CI path filter omits wenlan-core's ORT feature manifest".into());
    }

    if !smoke_script.contains("Get-Process -Id $proc.Id -Module")
        || !smoke_script.contains("onnxruntime.dll")
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
        "Windows ORT distribution proof drift:\n{}",
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
            .any(|violation| violation.contains("ORT_DYLIB_PATH")),
        "fixture must reject Windows tests that can load a runner DLL: {violations:?}"
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
                // The whole module is imported behind #[cfg(test)] in lib.rs.
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
                .strip_suffix("/**")
                .is_some_and(|prefix| path.starts_with(&format!("{prefix}/")))
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
    platform_sensitive_paths: &[(String, &'static str, &'static str)],
    release_profile_sensitive_paths: &[String],
) -> Vec<String> {
    let ci: serde_yaml::Value = serde_yaml::from_str(workflow).expect("parse ci.yml");
    let mut violations = Vec::new();

    for output in [
        "macos",
        "windows",
        "windows-lint",
        "windows-release",
        "mcp-platform",
    ] {
        if ci["jobs"]["detect-changes"]["outputs"][output]
            .as_str()
            .is_none()
        {
            violations.push(format!("detect-changes does not expose {output} routing"));
        }
    }

    let rust_paths = detect_change_filter_paths(&ci, "rust");
    if !rust_paths.contains("crates/**/*.rs") {
        violations.push(
            "Linux canonical routing is not a fail-closed catch-all for tracked Rust sources"
                .into(),
        );
    }
    if !rust_paths.contains(".github/workflows/coverage.yml") {
        violations.push(
            "coverage workflow cannot bootstrap its FastEmbed cache contract through rust".into(),
        );
    }
    if !rust_paths.contains("clippy.toml") {
        violations.push(
            "clippy configuration cannot bootstrap its syntax-aware FastEmbed guard through rust"
                .into(),
        );
    }
    for (path, platform, filter) in platform_sensitive_paths {
        let routed = detect_change_filter_paths(&ci, filter);
        if !filter_routes_path(&routed, path) {
            violations.push(format!(
                "platform-sensitive {platform} path is not fail-closed through {filter}: {path}"
            ));
        }
        if !filter_routes_path(&rust_paths, path) {
            violations.push(format!(
                "platform-sensitive {platform} path cannot bootstrap the CI contract through rust: {path}"
            ));
        }
    }

    let release_sensitive = detect_change_filter_paths(&ci, "windows-release");
    for path in release_profile_sensitive_paths {
        if !filter_routes_path(&release_sensitive, path) {
            violations.push(format!(
                "release-profile-sensitive source is not routed through windows-release: {path}"
            ));
        }
    }
    let windows_paths = detect_change_filter_paths(&ci, "windows");
    let macos_paths = detect_change_filter_paths(&ci, "macos");
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
            ("windows-release", &release_sensitive),
        ] {
            if !paths.contains(&path) {
                violations.push(format!(
                    "{filter} routing omits shared native source glob {path}"
                ));
            }
        }
    }
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "crates/**/Cargo.toml",
        "crates/**/build.rs",
        "crates/**/*.c",
        "crates/**/*.cc",
        "crates/**/*.cpp",
        "crates/**/*.cxx",
        "crates/**/*.h",
        "crates/**/*.hh",
        "crates/**/*.hpp",
        "crates/**/*.hxx",
        "crates/*/npm/**",
        "install.sh",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "scripts/stage-onnxruntime-windows.ps1",
        "scripts/smoke-windows.ps1",
        "version.txt",
        ".release-please-manifest.json",
        "CHANGELOG.md",
    ] {
        if !release_sensitive.contains(path) {
            violations.push(format!(
                "windows-release routing omits release-sensitive path {path}"
            ));
        }
    }
    for (platform, paths) in [("macos", &macos_paths), ("windows", &windows_paths)] {
        if !filter_routes_path(paths, "install.sh") {
            violations.push(format!(
                "{platform} routing omits the root installer install.sh"
            ));
        }
        for path in [
            "crates/wenlan-server/src/**",
            "crates/wenlan-server/tests/**",
        ] {
            if !paths.contains(path) {
                violations.push(format!(
                    "{platform} routing omits shared server runtime boundary {path}"
                ));
            }
        }
        for path in [
            "version.txt",
            ".release-please-manifest.json",
            "CHANGELOG.md",
        ] {
            if !paths.contains(path) {
                violations.push(format!(
                    "{platform} routing omits pre-release full-platform trigger {path}"
                ));
            }
        }
    }
    let windows_lint = detect_change_filter_paths(&ci, "windows-lint");
    for path in [
        "crates/wenlan-core/src/lint/**",
        "scripts/lint-scale-gate.sh",
    ] {
        if !windows_lint.contains(path) {
            violations.push(format!(
                "windows-lint routing omits lint-sensitive path {path}"
            ));
        }
    }
    for (child_name, child_paths) in [
        ("release-sensitive", &release_sensitive),
        ("lint-sensitive", &windows_lint),
    ] {
        for path in child_paths {
            if !filter_routes_path(&windows_paths, path) {
                violations.push(format!(
                    "{child_name} Windows path does not also schedule the Windows job: {path}"
                ));
            }
        }
    }
    let mcp_platform = detect_change_filter_paths(&ci, "mcp-platform");
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
    for path in &mcp_platform {
        for (platform, platform_paths) in [("macos", &macos_paths), ("windows", &windows_paths)] {
            if !filter_routes_path(platform_paths, path) {
                violations.push(format!(
                    "mcp-platform path does not also schedule {platform}: {path}"
                ));
            }
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
    let differential_timeout = "${{ github.event_name == 'pull_request' && 30 || 60 }}";
    for job in ["test", "windows-release-proof"] {
        if ci["jobs"][job]["timeout-minutes"].as_str() != Some(differential_timeout) {
            violations.push(format!(
                "{job} does not enforce the 30-minute PR budget while allowing a 60-minute non-PR backstop"
            ));
        }
    }
    let windows_release_condition = ci["jobs"]["windows-release-proof"]["if"]
        .as_str()
        .unwrap_or_default();
    if job_needs(&ci, "windows-release-proof") != ["detect-changes"]
        || !windows_release_condition
            .contains("needs.detect-changes.outputs.windows-release == 'true'")
        || !windows_release_condition.contains("github.event_name != 'pull_request'")
    {
        violations.push(
            "Windows release proof is not an independent differential job after detect-changes"
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
    let setup_sccache = job_step(&ci, "test", "Set up sccache");
    let setup_condition = setup_sccache
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let enable_sccache = job_step(&ci, "test", "Enable sccache (Linux)");
    let enable_condition = enable_sccache
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let enable_run = enable_sccache
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if setup_condition != "matrix.os == 'ubuntu-24.04'"
        || enable_condition != "matrix.os == 'ubuntu-24.04'"
        || !enable_run.contains("SCCACHE_GHA_ENABLED=true")
        || !enable_run.contains("RUSTC_WRAPPER=sccache")
        || ci["jobs"]["test"]["env"]["RUSTC_WRAPPER"].as_str() == Some("sccache")
    {
        violations.push(
            "test matrix does not restrict sccache reads/writes to the proven Linux lane".into(),
        );
    }
    let main_owned_cache = "${{ github.ref == 'refs/heads/main' }}";
    for job in ["lint", "test", "test-quarantine", "windows-release-proof"] {
        let rust_cache = job_step_using(&ci, job, "Swatinem/rust-cache");
        if rust_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some(main_owned_cache) {
            violations.push(format!("{job} cache writes are not restricted to main"));
        }
    }
    let test_rust_cache = job_step_using(&ci, "test", "Swatinem/rust-cache");
    if test_rust_cache.and_then(|step| step["with"]["cache-targets"].as_str())
        != Some("${{ matrix.os != 'ubuntu-24.04' }}")
    {
        violations.push("Linux sccache lane duplicates target artifacts in rust-cache".into());
    }
    let quarantine_rust_cache = job_step_using(&ci, "test-quarantine", "Swatinem/rust-cache");
    if quarantine_rust_cache.and_then(|step| step["with"]["cache-targets"].as_str())
        != Some("false")
    {
        violations.push("test-quarantine duplicates sccache target artifacts in rust-cache".into());
    }
    let sccache_mode = "${{ github.ref == 'refs/heads/main' && 'READ_WRITE' || 'READ_ONLY' }}";
    for job in ["test", "test-quarantine"] {
        if ci["jobs"][job]["env"]["SCCACHE_GHA_RW_MODE"].as_str() != Some(sccache_mode) {
            violations.push(format!("{job} sccache PR mode is not read-only"));
        }
    }
    for job in ["fmt", "lint", "test", "test-quarantine"] {
        let condition = ci["jobs"][job]["if"].as_str().unwrap_or_default();
        for required in [
            "needs.detect-changes.outputs.rust",
            "needs.detect-changes.outputs.macos",
            "needs.detect-changes.outputs.windows",
            "github.event_name != 'pull_request'",
        ] {
            if !condition.contains(required) {
                violations.push(format!(
                    "{job} condition omits CI scheduling trigger {required}"
                ));
            }
        }
        if condition.contains("github.event.head_commit.message") {
            violations.push(format!(
                "{job} can skip a non-PR full backstop based on the head commit message"
            ));
        }
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
        "ubuntu-24.04",
        "steps.filter.outputs.macos",
        "steps.filter.outputs.windows",
    ] {
        if !matrix_run.contains(required) {
            violations.push(format!(
                "dynamic OS matrix is missing differential/backstop routing marker {required:?}"
            ));
        }
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
        "needs.detect-changes.outputs.rust",
        "needs.detect-changes.outputs.macos",
        "needs.detect-changes.outputs.windows",
        "github.event_name != 'pull_request'",
    ] {
        if !conclusion_run.contains(required) {
            violations.push(format!(
                "conclusion expectation omits CI scheduling trigger {required}"
            ));
        }
    }
    if conclusion_run.contains("github.event.head_commit.message") {
        violations.push(
            "conclusion can accept skipped non-PR backstops based on the head commit message"
                .into(),
        );
    }
    if !job_needs(&ci, "conclusion")
        .iter()
        .any(|job| job == "windows-release-proof")
    {
        violations.push("conclusion.needs omits the Windows release proof".into());
    }
    for (job, expected) in [
        ("fmt", "\"$run_rust\""),
        ("lint", "\"$run_rust\""),
        ("test", "\"$run_rust\""),
        (
            "windows-release-proof",
            "needs.detect-changes.outputs.windows-release",
        ),
        ("docs", "needs.detect-changes.outputs.docs"),
        ("plugin", "needs.detect-changes.outputs.plugin"),
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
        "Build Windows release binaries",
        "Native ORT smoke (Windows; release profile)",
    ] {
        if job_step(&ci, "test", step_name).is_some() {
            violations.push(format!(
                "{step_name} still serializes release proof inside the Windows test matrix"
            ));
        }
    }
    let windows_release_build = job_step(
        &ci,
        "windows-release-proof",
        "Build Windows release binaries",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    let release_build_commands = windows_release_build
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("cargo build --release"))
        .collect::<Vec<_>>();
    if release_build_commands.len() != 1 {
        violations.push(format!(
            "Windows release proof uses {} Cargo release build invocations, expected a single Cargo invocation",
            release_build_commands.len()
        ));
    }
    let release_build_args = release_build_commands
        .first()
        .map(|command| command.split_whitespace().collect::<Vec<_>>())
        .unwrap_or_default();
    let has_arg_pair = |flag: &str, value: &str| {
        release_build_args
            .windows(2)
            .any(|args| args == [flag, value])
    };
    if !has_arg_pair("-p", "wenlan")
        || !has_arg_pair("-p", "wenlan-server")
        || !has_arg_pair("-p", "wenlan-mcp")
        || !has_arg_pair("-p", "wenlan-core")
        || !has_arg_pair("--bin", "wenlan")
        || !has_arg_pair("--bin", "wenlan-server")
        || !has_arg_pair("--bin", "wenlan-mcp")
        || !has_arg_pair("--bin", "model_probe")
    {
        violations.push("Windows release proof omits a release artifact or model probe".into());
    }
    let windows_release_smoke = job_step(
        &ci,
        "windows-release-proof",
        "Native ORT smoke (Windows; release profile)",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !windows_release_smoke.contains("scripts/stage-onnxruntime-windows.ps1")
        || !windows_release_smoke.contains("scripts/smoke-windows.ps1")
        || !windows_release_smoke.contains(r"target\release")
    {
        violations.push("Windows release proof omits the native ORT smoke".into());
    }
    let release_cache = job_step_using(&ci, "windows-release-proof", "Swatinem/rust-cache");
    if release_cache
        .and_then(|step| step["uses"].as_str())
        .is_none_or(|uses| !uses.contains("Swatinem/rust-cache"))
        || release_cache.and_then(|step| step["with"]["shared-key"].as_str())
            != Some("windows-release")
        || release_cache.and_then(|step| step["with"]["cache-all-crates"].as_str()) != Some("true")
        || release_cache.and_then(|step| step["with"]["save-if"].as_str()) != Some(main_owned_cache)
    {
        violations
            .push("Windows release cache is not main-owned under a dedicated release key".into());
    }
    for (job, step_name) in [
        ("test", "Configure rust-lld linker (Windows tests)"),
        (
            "windows-release-proof",
            "Configure rust-lld linker (Windows release)",
        ),
    ] {
        let linker = job_step(&ci, job, step_name);
        let condition = linker
            .and_then(|step| step["if"].as_str())
            .unwrap_or_default();
        let run = linker
            .and_then(|step| step["run"].as_str())
            .unwrap_or_default();
        if (job == "test" && !condition.contains("matrix.os == 'windows-2022'"))
            || !run.contains("rust-lld.exe")
            || !run.contains("RUSTFLAGS=")
            || !run.contains("$env:GITHUB_ENV")
        {
            violations.push(format!("{job} does not configure rust-lld for Windows"));
        }
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

    let debug_build = job_step(&ci, "test", "Build Windows contract binaries");
    if debug_build
        .and_then(|step| step["run"].as_str())
        .is_none_or(|run| {
            !run.contains("cargo build -p wenlan -p wenlan-server")
                || !run.contains("target\\debug")
                || run.contains("--release")
        })
    {
        violations.push(
            "ordinary Windows contract does not build and stage debug runtime artifacts".into(),
        );
    }
    let schtasks = job_step(
        &ci,
        "test",
        "E2E wenlan background on/off round-trip (Windows; schtasks)",
    )
    .and_then(|step| step["run"].as_str())
    .unwrap_or_default();
    if !schtasks.contains(r"target\debug\wenlan.exe") {
        violations.push("ordinary Windows schtasks contract does not use debug binaries".into());
    }

    let mcp_compile = job_step(&ci, "test", "Compile platform-owned MCP runtime");
    let mcp_condition = mcp_compile
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let mcp_run = mcp_compile
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !mcp_condition.contains("matrix.os == 'macos-14'")
        || !mcp_condition.contains("matrix.os == 'windows-2022'")
        || !mcp_condition.contains("needs.detect-changes.outputs.mcp-platform == 'true'")
        || !mcp_condition.contains("github.event_name != 'pull_request'")
        || !mcp_run.contains("cargo check -p wenlan-mcp --lib --bins")
        || mcp_run.contains("--all-targets")
    {
        violations.push(
            "macOS/Windows ownership does not differentially compile every wenlan-mcp target"
                .into(),
        );
    }

    let macos_lib = job_step(&ci, "test", "Workspace lib tests (macOS)");
    let macos_lib_run = macos_lib
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !macos_lib_run.contains("cargo nextest run --workspace --lib")
        || macos_lib_run.contains("--test-threads=1")
    {
        violations.push("macOS nextest workspace proof is unnecessarily single-threaded".into());
    }
    let shared_integration = job_step(
        &ci,
        "test",
        "Integration tests wenlan-cli + wenlan-server (shared integration only)",
    );
    let shared_integration_condition = shared_integration
        .and_then(|step| step["if"].as_str())
        .unwrap_or_default();
    let shared_integration_run = shared_integration
        .and_then(|step| step["run"].as_str())
        .unwrap_or_default();
    if !shared_integration_condition.contains("matrix.os != 'windows-2022'")
        || !shared_integration_run.contains("-E 'kind(test)'")
    {
        violations
            .push("Linux/macOS integration step duplicates wenlan CLI/server lib tests".into());
    }
    let windows_integration = job_step(
        &ci,
        "test",
        "Integration tests wenlan-cli + wenlan-server (Windows)",
    );
    if windows_integration
        .and_then(|step| step["if"].as_str())
        .is_none_or(|condition| !condition.contains("matrix.os == 'windows-2022'"))
        || windows_integration
            .and_then(|step| step["run"].as_str())
            .is_none_or(|run| {
                !run.contains("cargo nextest run -p wenlan -p wenlan-server")
                    || run.contains("kind(test)")
            })
    {
        violations.push("Windows does not retain its full CLI/server platform contract".into());
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
        step_index("Compile platform-owned MCP runtime"),
        step_index("Build Windows contract binaries"),
    ) {
        (Some(integration), Some(mcp), Some(debug)) if integration < mcp && mcp < debug => {}
        _ => violations
            .push("Windows steps do not fail fast from integration into later compilation".into()),
    }

    violations
}

#[test]
fn ci_routing_is_fail_closed_and_differential() {
    let root = repo_root();
    let workflow =
        std::fs::read_to_string(root.join(".github/workflows/ci.yml")).expect("read ci.yml");
    let platform_sensitive_paths = platform_sensitive_paths(&root);
    let release_profile_sensitive_paths = release_profile_sensitive_paths(&root);
    let violations = ci_routing_contract_violations(
        &workflow,
        &platform_sensitive_paths,
        &release_profile_sensitive_paths,
    );
    assert!(
        violations.is_empty(),
        "CI routing contract drift:\n{}",
        violations.join("\n")
    );
}

#[test]
fn ci_routing_contract_rejects_fail_open_fixture() {
    let workflow = r#"
jobs:
  detect-changes:
    outputs:
      windows: ${{ steps.filter.outputs.windows }}
      windows-release: ${{ steps.filter.outputs.windows-release }}
    steps:
      - id: filter
        with:
          filters: |
            rust:
              - 'crates/**/*.rs'
            windows: []
            windows-release:
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
        "proven Linux lane",
        "cache writes",
        "duplicates target artifacts",
        "duplicates sccache target artifacts",
        "sccache PR mode",
        "condition omits CI scheduling trigger",
        "coverage workflow",
        "clippy configuration",
        "nextest config",
        "release-profile-sensitive",
        "release-sensitive",
        "30-minute PR budget",
        "independent differential job",
        "single Cargo invocation",
        "release artifact or model probe",
        "native ORT smoke",
        "release cache is not main-owned",
        "rust-lld",
        "does not also schedule",
        "debug runtime artifacts",
        "differentially compile every wenlan-mcp target",
        "mcp-platform routing",
        "single-threaded",
        "duplicates wenlan CLI/server lib tests",
        "full CLI/server platform contract",
        "fail fast before integration",
        "fail fast from integration",
        "root installer",
    ] {
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains(expected)),
            "fixture must exercise {expected:?}: {violations:?}"
        );
    }
}

// ── Teeth #9: the main eval canary stays off the required CI path ──

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
            != Some("fastembed-bge-base-en-v1.5-q-v2")
    {
        violations.push("main canary FastEmbed cache is not restore-only".into());
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
        "main canary contract drift:\n{}",
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
        "manual workflow_dispatch",
        "pull requests",
        "concurrency",
        "canonical Linux",
        "60-minute cold-cache budget",
        "independent job",
        "sccache is not read-only",
        "FastEmbed cache directory",
        "rust-cache is not restore-only",
        "FastEmbed cache is not restore-only",
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
