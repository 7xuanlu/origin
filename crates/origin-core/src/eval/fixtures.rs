// SPDX-License-Identifier: AGPL-3.0-only
//! TOML fixture loader for eval test cases.

use crate::error::OriginError;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixtureFile {
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalCase {
    pub query: String,
    pub domain: Option<String>,
    #[serde(default)]
    pub seeds: Vec<SeedMemory>,
    #[serde(default)]
    pub negative_seeds: Vec<SeedMemory>,
    #[serde(default)]
    pub entities: Vec<SeedEntity>,
    /// When true, this case has NO relevant results — all seeds are distractors.
    /// Runner uses empty-set metrics instead of standard IR metrics.
    #[serde(default)]
    pub empty_set: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SeedMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub domain: Option<String>,
    #[serde(default)]
    pub relevance: u8,
    pub structured_fields: Option<String>,
    /// Explicit confidence override. When None, auto-derived from memory_type tier.
    pub confidence: Option<f32>,
    /// Whether this memory has been user-confirmed (activates 1.5x boost).
    #[serde(default)]
    pub confirmed: Option<bool>,
    /// Quality assessment: "high", "medium", "low", or None (defaults to 0.9x).
    pub quality: Option<String>,
    /// Explicit is_recap flag. When None, auto-derived from memory_type == "recap".
    pub is_recap: Option<bool>,
    /// Source agent name (for trust-level scoring).
    pub source_agent: Option<String>,
    /// Simulated age in days (0 = now, 30 = one month old).
    /// Runner subtracts age_days from current timestamp when seeding.
    #[serde(default)]
    pub age_days: Option<u32>,
    /// Source ID this memory supersedes (for temporal update chains).
    pub supersedes: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SeedEntity {
    pub name: String,
    pub entity_type: String,
    pub domain: Option<String>,
    #[serde(default)]
    pub observations: Vec<SeedObservation>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SeedObservation {
    /// Simple string observation (backward-compatible)
    Simple(String),
    /// Observation with relevance grading for eval scoring
    Graded { content: String, relevance: u8 },
}

impl SeedObservation {
    pub fn content(&self) -> &str {
        match self {
            Self::Simple(s) => s,
            Self::Graded { content, .. } => content,
        }
    }
    pub fn relevance(&self) -> u8 {
        match self {
            Self::Simple(_) => 0, // ungraded = don't count in scoring
            Self::Graded { relevance, .. } => *relevance,
        }
    }
}

/// Load all fixture files from a directory and its subdirectories.
pub fn load_fixtures(dir: &Path) -> Result<Vec<EvalCase>, OriginError> {
    let mut cases = Vec::new();
    if !dir.exists() {
        return Ok(cases);
    }
    load_fixtures_recursive(dir, &mut cases)?;
    Ok(cases)
}

fn load_fixtures_recursive(dir: &Path, cases: &mut Vec<EvalCase>) -> Result<(), OriginError> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| OriginError::Generic(format!("read fixture dir {}: {}", dir.display(), e)))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            load_fixtures_recursive(&path, cases)?;
        } else if path.extension().is_some_and(|ext| ext == "toml") {
            let content = std::fs::read_to_string(&path).map_err(|e| {
                OriginError::Generic(format!("read fixture {}: {}", path.display(), e))
            })?;
            let fixture: FixtureFile = toml::from_str(&content).map_err(|e| {
                OriginError::Generic(format!("parse fixture {}: {}", path.display(), e))
            })?;
            cases.extend(fixture.cases);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fixture_toml() {
        let toml_str = r#"
[[cases]]
query = "test query"
domain = "test"

[[cases.seeds]]
id = "mem_1"
content = "Test memory content"
memory_type = "fact"
domain = "test"
relevance = 3

[[cases.seeds]]
id = "mem_2"
content = "Structured memory"
memory_type = "preference"
domain = "test"
relevance = 2
structured_fields = '{"preference":"dark mode","applies_when":"editors"}'

[[cases.negative_seeds]]
id = "mem_neg"
content = "Irrelevant content"
memory_type = "identity"
domain = "personal"

[[cases.entities]]
name = "Rust"
entity_type = "technology"
domain = "test"
observations = [
  "Rust is a systems programming language",
  { content = "Rust has zero-cost abstractions", relevance = 2 },
]
"#;
        let fixture: FixtureFile = toml::from_str(toml_str).unwrap();
        assert_eq!(fixture.cases.len(), 1);
        assert_eq!(fixture.cases[0].query, "test query");
        assert_eq!(fixture.cases[0].seeds.len(), 2);
        assert_eq!(fixture.cases[0].seeds[0].relevance, 3);
        assert_eq!(
            fixture.cases[0].seeds[1].structured_fields.as_deref(),
            Some(r#"{"preference":"dark mode","applies_when":"editors"}"#)
        );
        assert_eq!(fixture.cases[0].negative_seeds.len(), 1);
        assert_eq!(fixture.cases[0].entities.len(), 1);
        assert_eq!(fixture.cases[0].entities[0].name, "Rust");
        assert_eq!(fixture.cases[0].entities[0].observations.len(), 2);
        // First is simple string (ungraded), second is graded
        assert_eq!(fixture.cases[0].entities[0].observations[0].relevance(), 0);
        assert_eq!(fixture.cases[0].entities[0].observations[1].relevance(), 2);
    }

    #[test]
    fn test_fixture_serialize_roundtrip() {
        let fixture = FixtureFile {
            cases: vec![EvalCase {
                query: "test query".to_string(),
                domain: Some("test".to_string()),
                seeds: vec![SeedMemory {
                    id: "mem_1".to_string(),
                    content: "Test content".to_string(),
                    memory_type: "fact".to_string(),
                    domain: Some("test".to_string()),
                    relevance: 3,
                    structured_fields: None,
                    confidence: None,
                    confirmed: Some(true),
                    quality: None,
                    is_recap: None,
                    source_agent: None,
                    age_days: None,
                    supersedes: None,
                }],
                negative_seeds: vec![SeedMemory {
                    id: "neg_1".to_string(),
                    content: "Negative content".to_string(),
                    memory_type: "fact".to_string(),
                    domain: Some("other".to_string()),
                    relevance: 0,
                    structured_fields: None,
                    confidence: None,
                    confirmed: None,
                    quality: None,
                    is_recap: None,
                    source_agent: None,
                    age_days: None,
                    supersedes: None,
                }],
                entities: vec![],
                empty_set: false,
            }],
        };

        let toml_str = toml::to_string_pretty(&fixture).unwrap();
        let parsed: FixtureFile = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.cases.len(), 1);
        assert_eq!(parsed.cases[0].query, "test query");
        assert_eq!(parsed.cases[0].seeds[0].relevance, 3);
        assert_eq!(parsed.cases[0].negative_seeds[0].id, "neg_1");
    }

    #[test]
    fn test_load_fixtures_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        let fixture_path = dir.path().join("test.toml");
        std::fs::write(
            &fixture_path,
            r#"
[[cases]]
query = "hello"

[[cases.seeds]]
id = "m1"
content = "world"
memory_type = "fact"
domain = "test"
relevance = 2
"#,
        )
        .unwrap();

        let cases = load_fixtures(dir.path()).unwrap();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].query, "hello");
    }

    #[test]
    fn test_parse_temporal_fields() {
        let toml_str = r#"
[[cases]]
query = "what database"
empty_set = false

[[cases.seeds]]
id = "old_1"
content = "Chose PostgreSQL"
memory_type = "decision"
domain = "project"
relevance = 1
age_days = 90

[[cases.seeds]]
id = "new_1"
content = "Migrated to libSQL"
memory_type = "decision"
domain = "project"
relevance = 3
age_days = 5
supersedes = "old_1"
"#;
        let fixture: FixtureFile = toml::from_str(toml_str).unwrap();
        assert!(!fixture.cases[0].empty_set);
        assert_eq!(fixture.cases[0].seeds[0].age_days, Some(90));
        assert_eq!(fixture.cases[0].seeds[0].supersedes, None);
        assert_eq!(fixture.cases[0].seeds[1].age_days, Some(5));
        assert_eq!(
            fixture.cases[0].seeds[1].supersedes,
            Some("old_1".to_string())
        );
    }

    #[test]
    fn test_parse_empty_set_case() {
        let toml_str = r#"
[[cases]]
query = "best Italian restaurants"
empty_set = true

[[cases.negative_seeds]]
id = "neg_1"
content = "User prefers cooking at home"
memory_type = "preference"
domain = "personal"
"#;
        let fixture: FixtureFile = toml::from_str(toml_str).unwrap();
        assert!(fixture.cases[0].empty_set);
        assert!(fixture.cases[0].seeds.is_empty());
        assert_eq!(fixture.cases[0].negative_seeds.len(), 1);
    }

    #[test]
    fn test_existing_fixtures_parse_without_new_fields() {
        let toml_str = r#"
[[cases]]
query = "test"

[[cases.seeds]]
id = "m1"
content = "content"
memory_type = "fact"
domain = "test"
relevance = 2
"#;
        let fixture: FixtureFile = toml::from_str(toml_str).unwrap();
        assert!(!fixture.cases[0].empty_set);
        assert_eq!(fixture.cases[0].seeds[0].age_days, None);
        assert_eq!(fixture.cases[0].seeds[0].supersedes, None);
    }

    #[test]
    fn test_load_fixtures_from_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("gen").join("regression");
        std::fs::create_dir_all(&sub).unwrap();

        std::fs::write(
            sub.join("test.toml"),
            r#"
[[cases]]
query = "from subdir"

[[cases.seeds]]
id = "m1"
content = "sub content"
memory_type = "fact"
domain = "test"
relevance = 2
"#,
        )
        .unwrap();

        std::fs::write(
            dir.path().join("top.toml"),
            r#"
[[cases]]
query = "from top"

[[cases.seeds]]
id = "m2"
content = "top content"
memory_type = "fact"
domain = "test"
relevance = 3
"#,
        )
        .unwrap();

        let cases = load_fixtures(dir.path()).unwrap();
        assert_eq!(cases.len(), 2);
        let queries: Vec<&str> = cases.iter().map(|c| c.query.as_str()).collect();
        assert!(queries.contains(&"from subdir"));
        assert!(queries.contains(&"from top"));
    }

    #[test]
    fn test_all_real_fixtures_parse() {
        // Verify all fixture TOML files in eval/fixtures/ parse without errors
        // and have no duplicate seed IDs within the same case.
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
        if !fixture_dir.exists() {
            return; // skip if fixtures dir not present (e.g. in CI without fixtures)
        }
        let cases = load_fixtures(&fixture_dir).unwrap();
        assert!(!cases.is_empty(), "Expected at least one fixture case");

        // Check no duplicate IDs within any single case
        for case in &cases {
            let mut seen = std::collections::HashSet::new();
            for seed in &case.seeds {
                assert!(
                    seen.insert(&seed.id),
                    "Duplicate seed ID '{}' in case '{}'",
                    seed.id,
                    case.query
                );
            }
            for neg in &case.negative_seeds {
                assert!(
                    seen.insert(&neg.id),
                    "Duplicate negative ID '{}' in case '{}'",
                    neg.id,
                    case.query
                );
            }
        }
    }
}
