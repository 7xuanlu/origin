// SPDX-License-Identifier: Apache-2.0
//! KG-faithfulness benchmark — measures whether the extractor produces entities
//! and relations grounded in the source memory text.

use serde::{Deserialize, Serialize};

use crate::eval::report::ReportEnv;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KgCaseResult {
    pub fixture_path: String,
    pub case_id: String,
    pub entity_precision: f64,
    pub entity_recall: f64,
    pub entity_f1: f64,
    pub relation_precision: f64,
    pub relation_recall: f64,
    pub relation_f1: f64,
    pub unfaithful_entities: Vec<String>,
    pub unfaithful_relations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KgFaithfulnessReport {
    pub fixture_count: usize,
    pub case_count: usize,
    pub entity_precision: f64,
    pub entity_recall: f64,
    pub entity_f1: f64,
    pub relation_precision: f64,
    pub relation_recall: f64,
    pub relation_f1: f64,
    pub per_case: Vec<KgCaseResult>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub env: Option<ReportEnv>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub latency: Option<crate::eval::latency::LatencySummary>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KgFixture {
    pub description: String,
    pub case: Vec<KgFixtureCase>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KgFixtureCase {
    pub id: String,
    pub source_text: String,
    pub expected_entities: Vec<KgExpectedEntity>,
    pub expected_relations: Vec<KgExpectedRelation>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KgExpectedEntity {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KgExpectedRelation {
    pub from: String,
    pub to: String,
    pub relation_type: String,
}

/// Returns true if `name` appears as a whole-word substring in `source`
/// (case-insensitive). Uses regex word boundaries (`\b`) to avoid false
/// positives like "Rust" matching "Trustworthy". Falls back to plain
/// substring contains() if regex compilation fails (e.g. on pathological
/// input), preserving the previous behavior.
pub fn check_entity_faithful_string(name: &str, source: &str) -> bool {
    let needle = name.trim();
    if needle.is_empty() {
        return false;
    }
    let pattern = format!(r"(?i)\b{}\b", regex::escape(needle));
    match regex::Regex::new(&pattern) {
        Ok(re) => re.is_match(source),
        Err(_) => {
            let lo_needle = needle.to_ascii_lowercase();
            let lo_source = source.to_ascii_lowercase();
            lo_source.contains(&lo_needle)
        }
    }
}

pub fn check_relation_faithful_string(
    rel: &KgExpectedRelation,
    source: &str,
    canonical: &[&str],
) -> bool {
    check_entity_faithful_string(&rel.from, source)
        && check_entity_faithful_string(&rel.to, source)
        && canonical
            .iter()
            .any(|c| c.eq_ignore_ascii_case(&rel.relation_type))
}

pub fn f1(precision: f64, recall: f64) -> f64 {
    if precision == 0.0 && recall == 0.0 {
        return 0.0;
    }
    2.0 * precision * recall / (precision + recall)
}

/// Canonical relation vocabulary. Sourced from `crate::extract::RELATION_VOCABULARY`
/// which mirrors the production seed at `db.rs:3907-3925`. Aliases are coerced
/// at write time so the bench checks canonical names only.
pub fn canonical_relation_types() -> Vec<&'static str> {
    crate::extract::RELATION_VOCABULARY.to_vec()
}

pub fn score_case(
    fixture_path: &str,
    case: &KgFixtureCase,
    extracted: &crate::extract::KgExtractionResult,
) -> KgCaseResult {
    let canonical = canonical_relation_types();

    // Entity precision: extracted entities that are faithful to source.
    let mut faithful_extracted = 0usize;
    let mut unfaithful_entities: Vec<String> = Vec::new();
    for e in &extracted.entities {
        if check_entity_faithful_string(&e.name, &case.source_text) {
            faithful_extracted += 1;
        } else {
            unfaithful_entities.push(e.name.clone());
        }
    }
    let entity_precision = if extracted.entities.is_empty() {
        0.0
    } else {
        faithful_extracted as f64 / extracted.entities.len() as f64
    };

    // Entity recall: expected entities present in extraction.
    let extracted_names: std::collections::HashSet<String> = extracted
        .entities
        .iter()
        .map(|e| e.name.to_ascii_lowercase())
        .collect();
    let recalled_count = case
        .expected_entities
        .iter()
        .filter(|e| extracted_names.contains(&e.name.to_ascii_lowercase()))
        .count();
    let entity_recall = if case.expected_entities.is_empty() {
        0.0
    } else {
        recalled_count as f64 / case.expected_entities.len() as f64
    };

    // Relation precision: extracted relations faithful to source + canonical type.
    let mut faithful_extracted_rels = 0usize;
    let mut unfaithful_relations: Vec<String> = Vec::new();
    for r in &extracted.relations {
        let as_expected = KgExpectedRelation {
            from: r.from.clone(),
            to: r.to.clone(),
            relation_type: r.relation_type.clone(),
        };
        if check_relation_faithful_string(&as_expected, &case.source_text, &canonical) {
            faithful_extracted_rels += 1;
        } else {
            unfaithful_relations.push(format!("{} --{}-> {}", r.from, r.relation_type, r.to));
        }
    }
    let relation_precision = if extracted.relations.is_empty() {
        0.0
    } else {
        faithful_extracted_rels as f64 / extracted.relations.len() as f64
    };

    // Relation recall: expected relations present in extraction.
    let extracted_triples: std::collections::HashSet<(String, String, String)> = extracted
        .relations
        .iter()
        .map(|r| {
            (
                r.from.to_ascii_lowercase(),
                r.to.to_ascii_lowercase(),
                r.relation_type.to_ascii_lowercase(),
            )
        })
        .collect();
    let recalled_rel_count = case
        .expected_relations
        .iter()
        .filter(|er| {
            extracted_triples.contains(&(
                er.from.to_ascii_lowercase(),
                er.to.to_ascii_lowercase(),
                er.relation_type.to_ascii_lowercase(),
            ))
        })
        .count();
    let relation_recall = if case.expected_relations.is_empty() {
        0.0
    } else {
        recalled_rel_count as f64 / case.expected_relations.len() as f64
    };

    KgCaseResult {
        fixture_path: fixture_path.to_string(),
        case_id: case.id.clone(),
        entity_precision,
        entity_recall,
        entity_f1: f1(entity_precision, entity_recall),
        relation_precision,
        relation_recall,
        relation_f1: f1(relation_precision, relation_recall),
        unfaithful_entities,
        unfaithful_relations,
    }
}

/// Run the full KG-faithfulness benchmark over every fixture under `fixture_dir`.
pub async fn run_kg_faithfulness_eval(
    extractor: &crate::engine::LlmEngine,
    fixture_dir: &std::path::Path,
) -> KgFaithfulnessReport {
    let mut report = KgFaithfulnessReport::default();
    if !fixture_dir.exists() {
        return report;
    }
    let fixtures: Vec<std::path::PathBuf> = std::fs::read_dir(fixture_dir)
        .ok()
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("toml"))
                .collect()
        })
        .unwrap_or_default();
    for path in &fixtures {
        let fx = match load_kg_fixture(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[kg_faith] skip {}: {}", path.display(), e);
                continue;
            }
        };
        report.fixture_count += 1;
        let memories: Vec<(usize, String)> = fx
            .case
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.source_text.clone()))
            .collect();
        let extracted = extractor.extract_kg_batch(&memories);
        for (case_idx, case) in fx.case.iter().enumerate() {
            let ext = extracted.iter().find(|r| r.index == case_idx);
            let Some(ext) = ext else {
                continue;
            };
            let path_str = path.to_string_lossy().to_string();
            let case_result = score_case(&path_str, case, ext);
            report.per_case.push(case_result);
            report.case_count += 1;
        }
    }
    // Macro-averages.
    if !report.per_case.is_empty() {
        let n = report.per_case.len() as f64;
        report.entity_precision = report
            .per_case
            .iter()
            .map(|c| c.entity_precision)
            .sum::<f64>()
            / n;
        report.entity_recall = report.per_case.iter().map(|c| c.entity_recall).sum::<f64>() / n;
        report.entity_f1 = report.per_case.iter().map(|c| c.entity_f1).sum::<f64>() / n;
        report.relation_precision = report
            .per_case
            .iter()
            .map(|c| c.relation_precision)
            .sum::<f64>()
            / n;
        report.relation_recall = report
            .per_case
            .iter()
            .map(|c| c.relation_recall)
            .sum::<f64>()
            / n;
        report.relation_f1 = report.per_case.iter().map(|c| c.relation_f1).sum::<f64>() / n;
    }
    report
}

pub fn load_kg_fixture(path: &std::path::Path) -> Result<KgFixture, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kg_faithfulness_report_default_is_empty() {
        let r = KgFaithfulnessReport::default();
        assert_eq!(r.fixture_count, 0);
        assert_eq!(r.entity_f1, 0.0);
        assert!(r.per_case.is_empty());
        assert!(r.env.is_none());
    }

    #[test]
    fn case_result_holds_per_fixture_metrics() {
        let c = KgCaseResult {
            fixture_path: "test.toml".to_string(),
            case_id: "c1".to_string(),
            entity_precision: 0.8,
            entity_recall: 0.6,
            entity_f1: 0.686,
            relation_precision: 0.5,
            relation_recall: 0.5,
            relation_f1: 0.5,
            unfaithful_entities: vec!["fake_entity".to_string()],
            unfaithful_relations: vec![],
        };
        assert!((c.entity_f1 - 0.686).abs() < 0.01);
    }

    #[test]
    fn load_kg_fixture_parses_minimal_toml() {
        let toml_str = r#"
            description = "smoke"

            [[case]]
            id = "c1"
            source_text = "Rust is related_to memory safety."
            expected_entities = [
                { name = "Rust", kind = "language" },
                { name = "memory safety", kind = "concept" },
            ]
            expected_relations = [
                { from = "Rust", to = "memory safety", relation_type = "related_to" },
            ]
        "#;
        let f: KgFixture = toml::from_str(toml_str).expect("toml parses");
        assert_eq!(f.case.len(), 1);
        assert_eq!(f.case[0].expected_entities.len(), 2);
        assert_eq!(f.case[0].expected_relations[0].relation_type, "related_to");
    }

    #[test]
    fn load_kg_fixture_from_path_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(
            &path,
            r#"
            description = "file load"
            [[case]]
            id = "c1"
            source_text = "X."
            expected_entities = []
            expected_relations = []
        "#,
        )
        .unwrap();
        let f = load_kg_fixture(&path).expect("loads");
        assert_eq!(f.description, "file load");
    }

    #[test]
    fn check_entity_faithful_string_match_lowercase() {
        let src = "Rust is a systems programming language.";
        assert!(check_entity_faithful_string("Rust", src));
        assert!(check_entity_faithful_string("rust", src));
        assert!(check_entity_faithful_string("systems programming", src));
        assert!(!check_entity_faithful_string("Python", src));
    }

    #[test]
    fn check_relation_faithful_requires_both_endpoints_and_canonical_type() {
        let src = "Rust is related_to memory safety.";
        let canonical = canonical_relation_types();
        let r_ok = KgExpectedRelation {
            from: "Rust".into(),
            to: "memory safety".into(),
            relation_type: "related_to".into(),
        };
        let r_bad_endpoint = KgExpectedRelation {
            from: "Python".into(),
            to: "memory safety".into(),
            relation_type: "related_to".into(),
        };
        let r_noncanonical_type = KgExpectedRelation {
            from: "Rust".into(),
            to: "memory safety".into(),
            relation_type: "totally_made_up_type".into(),
        };
        assert!(check_relation_faithful_string(&r_ok, src, &canonical));
        assert!(!check_relation_faithful_string(
            &r_bad_endpoint,
            src,
            &canonical
        ));
        assert!(!check_relation_faithful_string(
            &r_noncanonical_type,
            src,
            &canonical
        ));
    }

    #[test]
    fn f1_handles_edge_cases() {
        assert_eq!(f1(0.0, 0.0), 0.0);
        assert!((f1(1.0, 1.0) - 1.0).abs() < 1e-9);
        assert!((f1(0.5, 0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn score_case_perfect_extraction_yields_f1_1() {
        let case = KgFixtureCase {
            id: "c1".into(),
            source_text: "Rust is related_to memory safety.".into(),
            expected_entities: vec![
                KgExpectedEntity {
                    name: "Rust".into(),
                    kind: "language".into(),
                },
                KgExpectedEntity {
                    name: "memory safety".into(),
                    kind: "concept".into(),
                },
            ],
            expected_relations: vec![KgExpectedRelation {
                from: "Rust".into(),
                to: "memory safety".into(),
                relation_type: "related_to".into(),
            }],
        };
        let extracted = crate::extract::KgExtractionResult {
            index: 0,
            entities: vec![
                crate::extract::ExtractedEntity {
                    name: "Rust".into(),
                    entity_type: "language".into(),
                },
                crate::extract::ExtractedEntity {
                    name: "memory safety".into(),
                    entity_type: "concept".into(),
                },
            ],
            observations: vec![],
            relations: vec![crate::extract::ExtractedRelation {
                from: "Rust".into(),
                to: "memory safety".into(),
                relation_type: "related_to".into(),
                confidence: None,
                explanation: None,
                span: None,
            }],
        };
        let r = score_case("test.toml", &case, &extracted);
        assert!((r.entity_f1 - 1.0).abs() < 1e-9);
        assert!((r.relation_f1 - 1.0).abs() < 1e-9);
        assert!(r.unfaithful_entities.is_empty());
    }

    #[test]
    fn check_entity_faithful_string_rejects_substring_false_positives() {
        // "Rust" must not match "Trustworthy" / "Rustaceans" / "intrust"
        assert!(!check_entity_faithful_string(
            "Rust",
            "Trustworthy code matters."
        ));
        assert!(!check_entity_faithful_string(
            "Rust",
            "Rustaceans love trust."
        ));
        // But it should still match when the word appears as a token
        assert!(check_entity_faithful_string(
            "Rust",
            "We chose Rust for safety."
        ));
        assert!(check_entity_faithful_string(
            "rust",
            "We chose Rust for safety."
        ));
        // Multi-word names still work (word boundaries around the full phrase)
        assert!(check_entity_faithful_string(
            "memory safety",
            "Rust provides memory safety."
        ));
    }

    #[test]
    fn score_case_hallucinated_entity_lowers_precision() {
        let case = KgFixtureCase {
            id: "c2".into(),
            source_text: "Rust is fast.".into(),
            expected_entities: vec![KgExpectedEntity {
                name: "Rust".into(),
                kind: "language".into(),
            }],
            expected_relations: vec![],
        };
        let extracted = crate::extract::KgExtractionResult {
            index: 0,
            entities: vec![
                crate::extract::ExtractedEntity {
                    name: "Rust".into(),
                    entity_type: "language".into(),
                },
                crate::extract::ExtractedEntity {
                    name: "Python".into(),
                    entity_type: "language".into(),
                },
            ],
            observations: vec![],
            relations: vec![],
        };
        let r = score_case("test.toml", &case, &extracted);
        assert!((r.entity_precision - 0.5).abs() < 1e-9);
        assert!((r.entity_recall - 1.0).abs() < 1e-9);
        assert_eq!(r.unfaithful_entities, vec!["Python".to_string()]);
    }
}
