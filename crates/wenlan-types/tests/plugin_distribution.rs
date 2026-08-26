// SPDX-License-Identifier: Apache-2.0
//! Lightweight plugin distribution contract tests.

use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("wenlan-types is nested under crates/")
        .to_path_buf()
}

fn read_text(relative: &str) -> String {
    let path = repo_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn frontmatter_value(text: &str, key: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next()? != "---" {
        return None;
    }
    let prefix = format!("{key}:");
    for line in lines {
        if line == "---" {
            return None;
        }
        if let Some(value) = line.strip_prefix(&prefix) {
            return Some(value.trim().trim_matches('"').to_string());
        }
    }
    None
}

#[test]
fn plugin_setup_repairs_stale_daemon_versions() {
    let setup = read_text("plugin/skills/setup/SKILL.md");
    let codex_setup = read_text("plugin-codex/skills/setup/SKILL.md");
    let hook = read_text("plugin/hooks/check-daemon.sh");

    for text in [&setup, &codex_setup] {
        for needle in [
            "Compare daemon version vs plugin manifest version",
            "If mismatch, check the direction before repairing.",
            "Only if the daemon release is older than the plugin release, repair the",
            "curl -fsSL https://raw.githubusercontent.com/7xuanlu/wenlan/v${RELEASE_VER}/install.sh | bash",
            "wenlan setup --basic",
            "wenlan background on",
            "Runtime and plugin versions still differ after repair",
            "update the Wenlan plugin",
        ] {
            assert!(
                text.contains(needle),
                "/setup missing stale-daemon repair contract: {needle}"
            );
        }
        assert!(
            !text.contains("wenlan install"),
            "/setup should use `wenlan background on`, not `wenlan install`"
        );
        assert!(
            !text.contains("EXPECTED_VER=\"0.9.5\""),
            "/setup must not hardcode the old Codex plugin runtime version"
        );
        assert!(
            !text.contains("/init"),
            "/setup skill should not advertise the removed /init command"
        );
    }

    assert!(
        hook.contains("Run /wenlan:setup to repair"),
        "SessionStart hook should direct version mismatches to the self-healing /setup entrypoint"
    );
    assert!(
        !hook.contains("/wenlan:init"),
        "SessionStart hook should not mention the removed /init entrypoint"
    );
    assert!(
        !hook.contains("Otherwise upgrade: curl -fsSL"),
        "SessionStart hook should not print raw upgrade commands when /setup owns repair"
    );
    assert!(
        hook.contains("for i in 1 2 3") && hook.contains("curl -fsS -m 3"),
        "SessionStart hook should retry daemon health checks to avoid false down reports"
    );
}

#[test]
fn plugin_skill_inventory_uses_setup_and_no_deprecated_aliases() {
    for path in [
        "plugin/skills/setup/SKILL.md",
        "plugin-codex/skills/setup/SKILL.md",
    ] {
        assert!(
            repo_root().join(path).is_file(),
            "missing setup skill: {path}"
        );
    }

    for path in [
        "plugin/skills/init/SKILL.md",
        "plugin/skills/debrief/SKILL.md",
        "plugin-codex/skills/init/SKILL.md",
        "plugin-codex/skills/debrief/SKILL.md",
    ] {
        assert!(
            !repo_root().join(path).exists(),
            "removed skill should not remain as an alias: {path}"
        );
    }
}

#[test]
fn pages_skill_replaces_read() {
    // `/read` was renamed to `/pages` (browse + preview). The pages skill
    // must exist, the read skill must be gone, and no user-facing doc may
    // still advertise `/read`.
    let pages_path = repo_root().join("plugin/skills/pages/SKILL.md");
    assert!(
        pages_path.is_file(),
        "missing pages skill: {}",
        pages_path.display()
    );
    let read_path = repo_root().join("plugin/skills/read/SKILL.md");
    assert!(
        !read_path.exists(),
        "read skill should be deleted (renamed to /pages): {}",
        read_path.display()
    );

    for doc in [
        "plugin/skills/help/SKILL.md",
        "plugin/skills/README.md",
        "plugin/.claude-plugin/README.md",
    ] {
        let text = read_text(doc);
        assert!(
            !text.contains("/read"),
            "{doc} still advertises the removed /read command"
        );
        assert!(
            text.contains("/pages"),
            "{doc} should advertise the /pages command"
        );
    }
}

// The MCP surface hands the agent raw snake_case codes; the plain sentence the
// CLI prints is not in that payload. Both lint skills therefore carry the
// mapping in prose -- and it has to stay the *same* prose, so this test reads
// the sentences straight off the enums rather than from a second hardcoded
// copy.
#[test]
fn both_lint_skills_spell_out_every_check_code_in_plain_english() {
    use wenlan_types::lint::{LintActionCode, LintRecommendationCode, LintSummaryCode};

    let expected: Vec<(&str, &str)> = vec![
        ("check_passed", LintSummaryCode::CheckPassed.meaning()),
        (
            "finding_detected",
            LintSummaryCode::FindingDetected.meaning(),
        ),
        (
            "prerequisite_unavailable",
            LintSummaryCode::PrerequisiteUnavailable.meaning(),
        ),
        (
            "snapshot_inconsistent",
            LintSummaryCode::SnapshotInconsistent.meaning(),
        ),
        (
            "execution_failed",
            LintSummaryCode::ExecutionFailed.meaning(),
        ),
        ("expected_empty", LintSummaryCode::ExpectedEmpty.meaning()),
        (
            "review_finding",
            LintRecommendationCode::ReviewFinding.action(),
        ),
        (
            "restore_prerequisite",
            LintRecommendationCode::RestorePrerequisite.action(),
        ),
        (
            "rerun_after_snapshot_stabilizes",
            LintRecommendationCode::RerunAfterSnapshotStabilizes.action(),
        ),
        (
            "inspect_runtime",
            LintRecommendationCode::InspectRuntime.action(),
        ),
        (
            "choose_model_source",
            LintActionCode::ChooseModelSource.action(),
        ),
    ];

    for path in [
        "plugin/skills/lint/SKILL.md",
        "plugin-codex/skills/lint/SKILL.md",
    ] {
        let text = read_text(path);
        for (code, sentence) in &expected {
            assert!(
                text.contains(&format!("`{code}`")),
                "{path} never names the `{code}` code"
            );
            assert!(
                text.contains(sentence),
                "{path} does not say what `{code}` means: expected {sentence:?}"
            );
        }
        assert!(
            text.contains("does not count toward actionable"),
            "{path} must say a passing check with an action is not a finding"
        );
    }
}

#[test]
fn lint_is_the_only_public_repair_flow_on_both_surfaces() {
    for path in [
        "plugin/skills/lint/SKILL.md",
        "plugin-codex/skills/lint/SKILL.md",
    ] {
        let text = read_text(path);
        for needle in [
            "/lint repair",
            "Plain `/lint`, `/lint deep`, the lint MCP tool, and `/api/lint` are fully",
            "If zero or multiple",
            "types remain plausible, do not prepare",
            "only one target",
            "Lint creates durable Review Items for choices that are not yet exact.",
            "turns exactly one Review Item into a separately approved manifest",
            "`lint_repair_review` generic accept remains rejected and non-mutating",
            "Never call `apply_lint_repair` in the same turn as `prepare_lint_repair`.",
            "prepare_lint_repair",
            "apply repair <manifest-id> <manifest-digest>",
            "Never call apply_lint_repair in the same turn as prepare_lint_repair",
            "applied_unverified",
            "CLI fallback: `wenlan lint --profile deep --agent-assist`",
        ] {
            assert!(text.contains(needle), "{path} missing guardrail: {needle}");
        }
        assert_eq!(
            frontmatter_value(&text, "argument-hint").as_deref(),
            Some("[deep|repair] [global|uncategorized|space:<name>]"),
            "{path} exposes the wrong public argument grammar",
        );
        assert!(
            !text.contains("profile:deep"),
            "{path} exposes profile:deep"
        );
        assert!(
            !text.contains("profile:general"),
            "{path} exposes profile:general"
        );
    }
    for removed in [
        "plugin/skills/lint-repair/SKILL.md",
        "plugin-codex/skills/lint-repair/SKILL.md",
    ] {
        assert!(
            !std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(removed)
                .exists(),
            "deprecated public skill still exists: {removed}"
        );
    }
}
