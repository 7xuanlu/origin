use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const PHASE_FILE: &str = "crates/wenlan-core/src/drift_guard/post_write_structure_phase.txt";
const POST_WRITE_ROOT: &str = "crates/wenlan-core/src/post_write.rs";
const POST_WRITE_DIR: &str = "crates/wenlan-core/src/post_write";
const CHILDREN: &[&str] = &[
    "entity_graph",
    "page_create",
    "page_dispatch",
    "page_revision",
    "page_update",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Monolith,
    TestsExternalized,
    LowerRiskFlows,
    CreateDispatch,
    Final,
}

impl Phase {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "monolith" => Ok(Self::Monolith),
            "tests-externalized" => Ok(Self::TestsExternalized),
            "lower-risk-flows" => Ok(Self::LowerRiskFlows),
            "create-dispatch" => Ok(Self::CreateDispatch),
            "final" => Ok(Self::Final),
            other => Err(format!("unknown post_write structure phase: {other:?}")),
        }
    }

    fn external_tests(self) -> bool {
        self != Self::Monolith
    }

    fn children(self) -> &'static [&'static str] {
        match self {
            Self::Monolith | Self::TestsExternalized => &[],
            Self::LowerRiskFlows => &["entity_graph", "page_revision"],
            Self::CreateDispatch => &[
                "entity_graph",
                "page_create",
                "page_dispatch",
                "page_revision",
            ],
            Self::Final => CHILDREN,
        }
    }

    fn reexports(self) -> BTreeSet<String> {
        let mut expected = BTreeSet::new();
        if matches!(
            self,
            Self::LowerRiskFlows | Self::CreateDispatch | Self::Final
        ) {
            for item in [
                "create_entity",
                "create_relation",
                "create_relation_with_span",
                "add_observation",
            ] {
                expected.insert(format!("pub:entity_graph:{item}"));
            }
            for item in [
                "accept_pending_revision",
                "accept_pending_revision_with_knowledge_path",
                "dismiss_pending_revision",
                "dismiss_contradiction",
            ] {
                expected.insert(format!("pub:page_revision:{item}"));
            }
        }
        if matches!(self, Self::CreateDispatch | Self::Final) {
            for item in [
                "create_page",
                "create_page_with_floor",
                "create_page_with_tuning",
                "update_page",
                "update_page_preserving_sources",
            ] {
                expected.insert(format!("pub:page_dispatch:{item}"));
            }
            for item in ["PageWrite", "page_write"] {
                expected.insert(format!("pub:page_dispatch:{item}"));
            }
            for item in [
                "update_page_at_source_revision",
                "update_page_growth_at_versions",
            ] {
                expected.insert(format!("pub(crate):page_dispatch:{item}"));
            }
        }
        if self == Self::Final {
            for item in [
                "PipelineStage",
                "Writer",
                "page_is_human_owned",
                "stage_page_revision_card",
            ] {
                expected.insert(format!("pub:page_update:{item}"));
            }
            for item in ["PRE_WRITE_GATE", "merge_shrink_threshold"] {
                expected.insert(format!("pub(crate):page_update:{item}"));
            }
        }
        expected
    }
}

fn mask_rust_non_code(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut index = 0;
    let mut block_depth = 0usize;
    let mut string = false;
    let mut escaped = false;
    let mut character = false;
    let mut raw_end: Option<Vec<u8>> = None;

    while index < bytes.len() {
        if let Some(end) = raw_end.as_ref() {
            if bytes[index..].starts_with(end) {
                for offset in 0..end.len() {
                    out[index + offset] = b' ';
                }
                index += end.len();
                raw_end = None;
            } else {
                if bytes[index] != b'\n' {
                    out[index] = b' ';
                }
                index += 1;
            }
            continue;
        }
        if block_depth > 0 {
            out[index] = b' ';
            if bytes[index..].starts_with(b"/*") {
                block_depth += 1;
                if index + 1 < out.len() {
                    out[index + 1] = b' ';
                }
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                block_depth -= 1;
                if index + 1 < out.len() {
                    out[index + 1] = b' ';
                }
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if string {
            if bytes[index] != b'\n' {
                out[index] = b' ';
            }
            if escaped {
                escaped = false;
            } else if bytes[index] == b'\\' {
                escaped = true;
            } else if bytes[index] == b'"' {
                string = false;
            }
            index += 1;
            continue;
        }
        if character {
            if bytes[index] != b'\n' {
                out[index] = b' ';
            }
            if escaped {
                escaped = false;
            } else if bytes[index] == b'\\' {
                escaped = true;
            } else if bytes[index] == b'\'' {
                character = false;
            }
            index += 1;
            continue;
        }
        if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                out[index] = b' ';
                index += 1;
            }
            continue;
        }
        if bytes[index..].starts_with(b"/*") {
            block_depth = 1;
            out[index] = b' ';
            if index + 1 < out.len() {
                out[index + 1] = b' ';
            }
            index += 2;
            continue;
        }
        if bytes[index] == b'r' {
            let mut cursor = index + 1;
            while cursor < bytes.len() && bytes[cursor] == b'#' {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'"' {
                let hashes = cursor - index - 1;
                for offset in 0..=(cursor - index) {
                    out[index + offset] = b' ';
                }
                raw_end = Some(
                    std::iter::once(b'"')
                        .chain(std::iter::repeat_n(b'#', hashes))
                        .collect(),
                );
                index = cursor + 1;
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                string = true;
                out[index] = b' ';
            }
            b'\'' if index + 2 < bytes.len() && bytes[index + 2] == b'\'' => {
                character = true;
                out[index] = b' ';
            }
            b'\''
                if index + 3 < bytes.len()
                    && bytes[index + 1] == b'\\'
                    && bytes[index + 3] == b'\'' =>
            {
                character = true;
                out[index] = b' ';
            }
            _ => {}
        }
        index += 1;
    }

    String::from_utf8(out).expect("mask preserves UTF-8 byte widths")
}

fn child_declarations(source: &str) -> BTreeSet<String> {
    let code = mask_rust_non_code(source);
    let declaration = regex::Regex::new(
        r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:;|\{)",
    )
    .unwrap();
    declaration
        .captures_iter(&code)
        .filter_map(|capture| {
            let name = capture.get(1).expect("module capture").as_str();
            (name != "tests").then(|| name.to_string())
        })
        .collect()
}

fn facade_reexports(source: &str) -> BTreeSet<String> {
    let code = mask_rust_non_code(source);
    let grouped = regex::Regex::new(
        r"(?s)(pub(?:\([^)]*\))?)\s+use\s+self::(entity_graph|page_create|page_dispatch|page_revision|page_update)::\{([^}]*)\};",
    )
    .unwrap();
    let single = regex::Regex::new(
        r"(pub(?:\([^)]*\))?)\s+use\s+self::(entity_graph|page_create|page_dispatch|page_revision|page_update)::([A-Za-z_][A-Za-z0-9_]*)\s*;",
    )
    .unwrap();
    let mut found = BTreeSet::new();
    for capture in grouped.captures_iter(&code) {
        for item in capture[3]
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            found.insert(format!("{}:{}:{item}", &capture[1], &capture[2]));
        }
    }
    for capture in single.captures_iter(&code) {
        found.insert(format!("{}:{}:{}", &capture[1], &capture[2], &capture[3]));
    }
    found
}

fn top_level_function_count(source: &str, name: &str) -> usize {
    let code = mask_rust_non_code(source);
    let declaration = regex::Regex::new(&format!(
        r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+{}\b",
        regex::escape(name)
    ))
    .unwrap();
    declaration
        .find_iter(&code)
        .filter(|found| {
            code[..found.start()]
                .bytes()
                .fold(0isize, |depth, byte| match byte {
                    b'{' => depth + 1,
                    b'}' => depth - 1,
                    _ => depth,
                })
                == 0
        })
        .count()
}

fn structure_violations(
    source: &str,
    phase: Phase,
    existing_children: &BTreeSet<String>,
    external_test_exists: bool,
    external_sources: &[(&str, &str)],
) -> Vec<String> {
    let mut violations = Vec::new();
    let expected_children: BTreeSet<String> = phase
        .children()
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let declared_children = child_declarations(source);
    if declared_children != expected_children {
        violations.push(format!(
            "private child declarations are {declared_children:?}, expected {expected_children:?} for {phase:?}"
        ));
    }
    if existing_children != &expected_children {
        violations.push(format!(
            "private child files are {existing_children:?}, expected {expected_children:?} for {phase:?}"
        ));
    }

    let code = mask_rust_non_code(source);
    for child in CHILDREN {
        let public =
            regex::Regex::new(&format!(r"(?m)^\s*pub(?:\([^)]*\))?\s+mod\s+{child}\s*;")).unwrap();
        if public.is_match(&code) {
            violations.push(format!(
                "post_write child module must stay private: {child}"
            ));
        }
    }

    let external_declaration =
        "#[cfg(test)]\n#[path = \"post_write/post_write_tests.rs\"]\nmod tests;";
    let inline_tests = regex::Regex::new(r"(?m)^\s*mod\s+tests\s*\{")
        .unwrap()
        .is_match(&code);
    if phase.external_tests() {
        if source.matches(external_declaration).count() != 1 {
            violations.push(
                "post_write.rs must declare the external post_write_tests.rs module exactly once"
                    .into(),
            );
        }
        if inline_tests {
            violations.push("post_write.rs still contains an inline tests module".into());
        }
        if !external_test_exists {
            violations.push("post_write/post_write_tests.rs is missing".into());
        }
    } else {
        if !inline_tests {
            violations.push("monolith phase must retain the inline tests module".into());
        }
        if source.contains(external_declaration) || external_test_exists {
            violations.push("monolith phase unexpectedly externalized post_write tests".into());
        }
    }

    let actual_reexports = facade_reexports(source);
    let expected_reexports = phase.reexports();
    if actual_reexports != expected_reexports {
        violations.push(format!(
            "facade child re-exports are {actual_reexports:?}, expected {expected_reexports:?} for {phase:?}"
        ));
    }

    let activity_helpers = top_level_function_count(source, "log_activity_best_effort");
    if activity_helpers != 1 {
        violations.push(format!(
            "post_write root must own exactly one top-level log_activity_best_effort helper, found {activity_helpers}"
        ));
    }

    if phase == Phase::Final {
        let private_impl = regex::Regex::new(
            r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*_impl\s*(?:<|\()",
        )
        .unwrap();
        let names: Vec<&str> = private_impl
            .find_iter(&code)
            .map(|found| found.as_str().trim())
            .collect();
        if !names.is_empty() {
            violations.push(format!(
                "final post_write facade still owns private *_impl bodies: {names:?}"
            ));
        }
    }

    let child_path = regex::Regex::new(
        r"(?:crate::|wenlan_core::)?post_write::(?:entity_graph|page_create|page_dispatch|page_revision|page_update)::",
    )
    .unwrap();
    for (path, external) in external_sources {
        if child_path.is_match(&mask_rust_non_code(external)) {
            violations.push(format!(
                "external caller names a private post_write child path: {path}"
            ));
        }
    }

    violations.sort();
    violations
}

#[test]
fn post_write_structure_matches_expected_phase() {
    let root = super::repo_root();
    let phase_source = std::fs::read_to_string(root.join(PHASE_FILE)).expect("read phase marker");
    let phase = Phase::parse(&phase_source).expect("valid post_write structure phase");
    let source = std::fs::read_to_string(root.join(POST_WRITE_ROOT)).expect("read post_write.rs");
    let post_write_dir = root.join(POST_WRITE_DIR);
    let existing_children: BTreeSet<String> = if post_write_dir.is_dir() {
        std::fs::read_dir(&post_write_dir)
            .expect("read post_write child directory")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("rs"))
            .filter_map(|path| {
                let stem = path.file_stem()?.to_str()?;
                (stem != "post_write_tests").then(|| stem.to_string())
            })
            .collect()
    } else {
        BTreeSet::new()
    };
    let external_test = root
        .join(POST_WRITE_DIR)
        .join("post_write_tests.rs")
        .is_file();
    fn collect_rust_files(path: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(path).expect("read Rust source directory") {
            let path = entry.expect("read Rust source entry").path();
            if path.is_dir() {
                collect_rust_files(&path, out);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    let mut rust_files = Vec::new();
    collect_rust_files(&root.join("crates"), &mut rust_files);
    let external_sources: Vec<(String, String)> = rust_files
        .into_iter()
        .map(|path| {
            path.strip_prefix(&root)
                .expect("crate source below repo root")
                .to_string_lossy()
                .to_string()
        })
        .filter(|path| path.ends_with(".rs"))
        .filter(|path| path != POST_WRITE_ROOT)
        .filter(|path| !path.starts_with(&format!("{POST_WRITE_DIR}/")))
        .map(|path| {
            let source = std::fs::read_to_string(root.join(&path)).unwrap_or_default();
            (path, source)
        })
        .collect();
    let borrowed: Vec<(&str, &str)> = external_sources
        .iter()
        .map(|(path, source)| (path.as_str(), source.as_str()))
        .collect();
    let violations =
        structure_violations(&source, phase, &existing_children, external_test, &borrowed);
    assert!(
        violations.is_empty(),
        "R6 post_write structure drifted at phase {phase:?}; advance \
         post_write_structure_phase.txt only in the slice that establishes the next exact shape:\n{}",
        violations.join("\n")
    );
}

#[test]
fn post_write_structure_guard_rejects_missing_or_public_final_boundary() {
    let valid = r#"mod entity_graph;
mod page_create;
mod page_dispatch;
mod page_revision;
mod page_update;
#[cfg(test)]
#[path = "post_write/post_write_tests.rs"]
mod tests;
pub use self::entity_graph::{create_entity, create_relation, create_relation_with_span, add_observation};
pub use self::page_dispatch::{PageWrite, page_write, create_page, create_page_with_floor, create_page_with_tuning, update_page, update_page_preserving_sources};
pub(crate) use self::page_dispatch::{update_page_at_source_revision, update_page_growth_at_versions};
pub use self::page_revision::{accept_pending_revision, accept_pending_revision_with_knowledge_path, dismiss_pending_revision, dismiss_contradiction};
pub use self::page_update::{PipelineStage, Writer, page_is_human_owned, stage_page_revision_card};
pub(crate) use self::page_update::{PRE_WRITE_GATE, merge_shrink_threshold};
pub(crate) async fn log_activity_best_effort() {}
"#;
    let children: BTreeSet<String> = CHILDREN.iter().map(|name| (*name).to_string()).collect();
    assert!(
        structure_violations(valid, Phase::Final, &children, true, &[]).is_empty(),
        "synthetic final facade must satisfy the exact boundary"
    );

    let widened_reexport =
        format!("{valid}\npub(super) use self::page_create::create_page_impl;\n");
    assert!(
        structure_violations(&widened_reexport, Phase::Final, &children, true, &[])
            .iter()
            .any(|item| item.contains("facade child re-exports")),
        "positive control must reject a pub(super) facade re-export"
    );

    let broken = format!(
        "{}\nmod hidden_inline_child {{}}\nasync fn update_page_impl() {{}}\n",
        valid
            .replace("mod page_update;", "pub mod page_update;")
            .replace("update_page_preserving_sources", "update_page_impl")
    );
    let violations = structure_violations(
        &broken,
        Phase::Final,
        &children,
        true,
        &[(
            "crates/wenlan-server/src/bad.rs",
            "crate::post_write::page_update::update_page();",
        )],
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("private child declarations")),
        "positive control must reject an undeclared inline child"
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("must stay private: page_update")),
        "positive control must reject a public child"
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("facade child re-exports")),
        "positive control must reject facade re-export drift"
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("private *_impl bodies")),
        "positive control must reject private implementation bodies in the final facade"
    );
    assert!(
        violations
            .iter()
            .any(|item| item.contains("private post_write child path")),
        "positive control must reject an external child-path caller"
    );

    let missing_root_helper = valid.replace(
        "pub(crate) async fn log_activity_best_effort() {}",
        "pub(crate) async fn moved_activity_helper() {}",
    );
    assert!(
        structure_violations(&missing_root_helper, Phase::Final, &children, true, &[])
            .iter()
            .any(|item| item.contains("root must own exactly one")),
        "positive control must reject moving the shared activity helper out of the root facade"
    );
}

#[test]
fn post_write_structure_phase_parser_fails_closed() {
    assert_eq!(Phase::parse("monolith").unwrap(), Phase::Monolith);
    assert!(
        Phase::parse("almost-final").is_err(),
        "an unknown staged phase must not silently weaken the structure tooth"
    );
}

#[test]
fn post_write_facade_contract_compiles_and_preserves_result_shape() {
    let _ = crate::post_write::create_entity;
    let _ = crate::post_write::create_page;
    let _ = crate::post_write::update_page;
    let _ = crate::post_write::accept_pending_revision;
    let _ = std::mem::size_of::<Option<crate::post_write::PageWrite<'static>>>();

    let result = crate::post_write::WriteResult {
        id: "page_1".into(),
        attached_to: None,
        warnings: Vec::new(),
        wrote: false,
        revision_card_id: None,
        gated: false,
        acknowledged: false,
        outcome: crate::post_write::WriteOutcome::Unchanged,
    };
    let serialized = serde_json::to_value(result).expect("serialize stable WriteResult facade");
    assert_eq!(
        serialized,
        serde_json::json!({
            "id": "page_1",
            "warnings": [],
            "wrote": false,
            "outcome": "unchanged",
        }),
        "the representative write-result envelope must not drift during decomposition"
    );

    let legacy: crate::post_write::WriteResult = serde_json::from_value(serde_json::json!({
        "id": "legacy",
        "warnings": [],
        "wrote": true,
    }))
    .expect("deserialize the pre-outcome WriteResult envelope");
    assert_eq!(legacy.outcome, crate::post_write::WriteOutcome::Wrote);
    assert!(!legacy.gated);
    assert!(!legacy.acknowledged);
}
