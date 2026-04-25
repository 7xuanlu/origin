// SPDX-License-Identifier: AGPL-3.0-only
use crate::cache::EmbeddingCache;
use crate::chunker::ChunkingEngine;
use crate::concepts::Concept;
use crate::error::OriginError;
use crate::events::EventEmitter;
use crate::privacy::redact_pii;
use crate::sources::{stability_tier, RawDocument};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;

#[derive(Clone, serde::Serialize)]
pub struct MigrationProgress {
    pub current: usize,
    pub total: usize,
    pub phase: String,
}

/// Embedding dimension — must match the model (GTE-Base-EN-v1.5-Q = 768).
pub const EMBEDDING_DIM: usize = 768;

/// Known-client registry — maps canonical technical IDs (what clients send in
/// `x-agent-name`) to human-friendly display names (what users see in UI).
///
/// **This is an OPEN registry, not an enum.** Unknown agents fall through to
/// their raw canonical ID. The registry exists only to make well-known MCP
/// clients display nicely out of the box — users don't have to manually
/// register `openai-mcp` to see it labeled `ChatGPT`.
///
/// User-set overrides (via `agent_connections.display_name`) always win over
/// entries here — see `row_to_agent`.
///
/// Keep this sorted by canonical ID.
pub const KNOWN_CLIENTS: &[(&str, &str)] = &[
    ("chatgpt", "ChatGPT"),
    ("claude-code", "Claude Code"),
    ("claude-desktop", "Claude Desktop"),
    ("codex", "Codex"),
    ("continue", "Continue"),
    ("cursor", "Cursor"),
    ("gemini-cli", "Gemini CLI"),
    ("obsidian", "Obsidian"),
    ("openai-mcp", "ChatGPT"),
    ("raycast", "Raycast"),
    ("vscode", "VS Code"),
    ("windsurf", "Windsurf"),
    ("zed", "Zed"),
];

/// Look up the friendly display name for a known canonical agent ID.
pub fn known_client_display_name(canonical: &str) -> Option<&'static str> {
    KNOWN_CLIENTS
        .iter()
        .find(|(k, _)| *k == canonical)
        .map(|(_, v)| *v)
}

/// Normalize an agent name into its canonical technical form.
///
/// - Trims whitespace.
/// - Lowercases.
/// - Collapses spaces, underscores, and dots into hyphens.
/// - Collapses runs of hyphens.
///
/// Idempotent: `canonicalize_agent_id(canonicalize_agent_id(x)) == canonicalize_agent_id(x)`.
///
/// Examples:
/// - `"Claude Code"` → `"claude-code"`
/// - `"CLAUDE_CODE"` → `"claude-code"`
/// - `"openai.mcp"`  → `"openai-mcp"`
pub fn canonicalize_agent_id(s: &str) -> String {
    let trimmed = s.trim().to_lowercase();
    let mut out = String::with_capacity(trimmed.len());
    let mut prev_dash = false;
    for ch in trimmed.chars() {
        let is_sep = ch == ' ' || ch == '_' || ch == '.' || ch == '-';
        if is_sep {
            if !prev_dash && !out.is_empty() {
                out.push('-');
                prev_dash = true;
            }
        } else {
            out.push(ch);
            prev_dash = false;
        }
    }
    // Trim trailing hyphen if any.
    while out.ends_with('-') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod agent_id_tests {
    use super::*;

    #[test]
    fn canonicalize_handles_case_and_separators() {
        assert_eq!(canonicalize_agent_id("Claude Code"), "claude-code");
        assert_eq!(canonicalize_agent_id("CLAUDE_CODE"), "claude-code");
        assert_eq!(canonicalize_agent_id("claude.code"), "claude-code");
        assert_eq!(canonicalize_agent_id("  Claude   Code  "), "claude-code");
        assert_eq!(canonicalize_agent_id("claude-code"), "claude-code");
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let once = canonicalize_agent_id("Open AI MCP");
        assert_eq!(canonicalize_agent_id(&once), once);
    }

    #[test]
    fn known_clients_are_canonical() {
        for (id, _) in KNOWN_CLIENTS {
            assert_eq!(
                *id,
                canonicalize_agent_id(id),
                "KNOWN_CLIENTS entry {id} is not canonical"
            );
        }
    }

    #[test]
    fn known_clients_lookup() {
        assert_eq!(known_client_display_name("openai-mcp"), Some("ChatGPT"));
        assert_eq!(
            known_client_display_name("claude-code"),
            Some("Claude Code")
        );
        assert_eq!(known_client_display_name("unknown-tool"), None);
    }
}

/// Embedding model configuration — used by eval 2x2 (model × prefix) ablations.
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub model: EmbeddingModel,
    pub dim: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::BGEBaseENV15Q,
            dim: 768,
        }
    }
}

impl EmbedConfig {
    pub fn bge_small() -> Self {
        Self {
            model: EmbeddingModel::BGESmallENV15,
            dim: 384,
        }
    }
    pub fn bge_small_q() -> Self {
        Self {
            model: EmbeddingModel::BGESmallENV15Q,
            dim: 384,
        }
    }
    pub fn bge_base() -> Self {
        Self {
            model: EmbeddingModel::BGEBaseENV15,
            dim: 768,
        }
    }
    pub fn bge_base_q() -> Self {
        Self {
            model: EmbeddingModel::BGEBaseENV15Q,
            dim: 768,
        }
    }
    pub fn gte_base() -> Self {
        Self {
            model: EmbeddingModel::GTEBaseENV15,
            dim: 768,
        }
    }
    pub fn gte_base_q() -> Self {
        Self {
            model: EmbeddingModel::GTEBaseENV15Q,
            dim: 768,
        }
    }
    pub fn nomic_v15() -> Self {
        Self {
            model: EmbeddingModel::NomicEmbedTextV15,
            dim: 768,
        }
    }
    pub fn nomic_v15_q() -> Self {
        Self {
            model: EmbeddingModel::NomicEmbedTextV15Q,
            dim: 768,
        }
    }
    pub fn snowflake_m() -> Self {
        Self {
            model: EmbeddingModel::SnowflakeArcticEmbedM,
            dim: 768,
        }
    }
    pub fn snowflake_m_q() -> Self {
        Self {
            model: EmbeddingModel::SnowflakeArcticEmbedMQ,
            dim: 768,
        }
    }
    pub fn mpnet_base() -> Self {
        Self {
            model: EmbeddingModel::AllMpnetBaseV2,
            dim: 768,
        }
    }
    // 1024d models
    pub fn bge_large() -> Self {
        Self {
            model: EmbeddingModel::BGELargeENV15,
            dim: 1024,
        }
    }
    pub fn bge_large_q() -> Self {
        Self {
            model: EmbeddingModel::BGELargeENV15Q,
            dim: 1024,
        }
    }
    pub fn gte_large() -> Self {
        Self {
            model: EmbeddingModel::GTELargeENV15,
            dim: 1024,
        }
    }
    pub fn gte_large_q() -> Self {
        Self {
            model: EmbeddingModel::GTELargeENV15Q,
            dim: 1024,
        }
    }
    pub fn mxbai_large() -> Self {
        Self {
            model: EmbeddingModel::MxbaiEmbedLargeV1,
            dim: 1024,
        }
    }
    pub fn mxbai_large_q() -> Self {
        Self {
            model: EmbeddingModel::MxbaiEmbedLargeV1Q,
            dim: 1024,
        }
    }
    pub fn modernbert_large() -> Self {
        Self {
            model: EmbeddingModel::ModernBertEmbedLarge,
            dim: 1024,
        }
    }
    pub fn snowflake_l() -> Self {
        Self {
            model: EmbeddingModel::SnowflakeArcticEmbedL,
            dim: 1024,
        }
    }
}

/// Resolve the directory FastEmbed should load the ONNX model from.
///
/// Resolution order (first hit wins):
///   1. The per-DB cache `<db_path>/fastembed_cache` when that directory
///      exists and is non-empty. This is what the running daemon uses
///      at `~/Library/Application Support/origin/memorydb/fastembed_cache`.
///   2. `ORIGIN_TEST_FASTEMBED_CACHE` env var (escape hatch for CI).
///   3. `dirs::data_dir().join("origin/memorydb/fastembed_cache")` —
///      matches the production daemon's conventional path on the same
///      host, so tests that use a tempdir DB still pick up the already-
///      downloaded model instead of re-fetching from HuggingFace.
///   4. `None` — let FastEmbed use its own default (`~/.fastembed_cache/`)
///      and download the model if it isn't there. Fine for a dev box
///      with internet, fails noisily on hosts where the HuggingFace TLS
///      bundle misbehaves (see the `OSStatus -26276` symptom from
///      2026-04-16 that blocked ~240 tests from `test_db()`).
pub fn resolve_fastembed_cache_dir(db_path: &std::path::Path) -> Option<std::path::PathBuf> {
    // 1. Per-DB cache — production daemon populates this.
    let per_db = db_path.join("fastembed_cache");
    if per_db.exists()
        && std::fs::read_dir(&per_db)
            .map(|mut it| it.next().is_some())
            .unwrap_or(false)
    {
        return Some(per_db);
    }

    // 2. Explicit env override.
    if let Ok(env_cache) = std::env::var("ORIGIN_TEST_FASTEMBED_CACHE") {
        let p = std::path::PathBuf::from(env_cache);
        if p.exists() {
            return Some(p);
        }
    }

    // 3. Shared host cache — matches the daemon's conventional path even
    //    when the caller supplied a tempdir.
    if let Some(shared) = dirs::data_dir().map(|d| d.join("origin/memorydb/fastembed_cache")) {
        if shared.exists()
            && std::fs::read_dir(&shared)
                .map(|mut it| it.next().is_some())
                .unwrap_or(false)
        {
            return Some(shared);
        }
    }

    None
}

/// Bigram Jaccard similarity between two strings (0.0–1.0).
/// Used for content-based deduplication in search results.
fn bigram_jaccard(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> HashSet<(char, char)> {
        let lower: Vec<char> = s.chars().flat_map(|c| c.to_lowercase()).collect();
        lower.windows(2).map(|w| (w[0], w[1])).collect()
    }
    let ba = bigrams(a);
    let bb = bigrams(b);
    let union = ba.union(&bb).count();
    if union == 0 {
        return 1.0;
    }
    ba.intersection(&bb).count() as f64 / union as f64
}

/// Row data for clustering algorithm
struct ClusterMemRow {
    source_id: String,
    content: String,
    entity_id: Option<String>,
    entity_name: Option<String>,
    community_id: Option<u32>,
    domain: Option<String>,
    embedding: Vec<f32>,
}

/// Cosine similarity between two embedding vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Greedy clustering: attach each memory to the first cluster where
/// cosine similarity with the centroid exceeds threshold.
fn cluster_by_similarity(
    memories: &[ClusterMemRow],
    indices: &[usize],
    threshold: f64,
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();

    for &idx in indices {
        let emb = &memories[idx].embedding;
        let mut assigned = false;
        for (group, centroid) in clusters.iter_mut() {
            if cosine_similarity(emb, centroid) >= threshold {
                group.push(idx);
                // Update centroid as running average
                for (j, val) in emb.iter().enumerate() {
                    centroid[j] += (val - centroid[j]) / group.len() as f32;
                }
                assigned = true;
                break;
            }
        }
        if !assigned {
            clusters.push((vec![idx], emb.clone()));
        }
    }

    clusters.into_iter().map(|(g, _)| g).collect()
}

fn build_distillation_cluster(
    memories: &[ClusterMemRow],
    indices: &[usize],
) -> DistillationCluster {
    let source_ids: Vec<String> = indices
        .iter()
        .map(|&i| memories[i].source_id.clone())
        .collect();
    let contents: Vec<String> = indices
        .iter()
        .map(|&i| memories[i].content.clone())
        .collect();
    let entity_id = memories[indices[0]].entity_id.clone();
    let entity_name = memories[indices[0]].entity_name.clone();
    let domain = memories[indices[0]].domain.clone();
    let estimated_tokens = contents.iter().map(|c| c.len() / 4 + 15).sum::<usize>() + 100;
    DistillationCluster {
        source_ids,
        contents,
        entity_id,
        entity_name,
        domain,
        estimated_tokens,
    }
}

/// Split a cluster that exceeds the LLM's token limit.
/// Uses farthest-first seed selection + nearest-centroid assignment.
/// No memory count cap — conceptual coherence is an LLM judgment, not a number.
fn sub_cluster_by_tokens(
    memories: &[ClusterMemRow],
    cluster: DistillationCluster,
    token_limit: usize,
) -> Vec<DistillationCluster> {
    if cluster.estimated_tokens <= token_limit {
        return vec![cluster];
    }

    let k = (cluster.estimated_tokens as f64 / token_limit as f64).ceil() as usize;
    let k = k.max(2);

    // Map source_ids back to indices in the memories array
    let indices: Vec<usize> = cluster
        .source_ids
        .iter()
        .filter_map(|sid| memories.iter().position(|m| m.source_id == *sid))
        .collect();

    if indices.len() <= k {
        return indices
            .iter()
            .map(|&i| build_distillation_cluster(memories, &[i]))
            .collect();
    }

    // Farthest-first seed selection
    let mut seeds: Vec<usize> = Vec::with_capacity(k);
    seeds.push(indices[0]);

    for _ in 1..k {
        let mut best_idx = indices[0];
        let mut best_min_dist = f64::NEG_INFINITY;
        for &idx in &indices {
            if seeds.contains(&idx) {
                continue;
            }
            let min_dist = seeds
                .iter()
                .map(|&s| 1.0 - cosine_similarity(&memories[idx].embedding, &memories[s].embedding))
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = idx;
            }
        }
        seeds.push(best_idx);
    }

    // Assign each memory to nearest seed
    let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k];
    for &idx in &indices {
        let mut best_k = 0;
        let mut best_sim = f64::NEG_INFINITY;
        for (ki, &seed) in seeds.iter().enumerate() {
            let sim = cosine_similarity(&memories[idx].embedding, &memories[seed].embedding);
            if sim > best_sim {
                best_sim = sim;
                best_k = ki;
            }
        }
        assignments[best_k].push(idx);
    }

    assignments
        .into_iter()
        .filter(|group| !group.is_empty())
        .map(|group| build_distillation_cluster(memories, &group))
        .collect()
}

// ===== Public Types =====
//
// Most DTOs now live in `origin-types` so the server crate and downstream
// consumers can depend on them without pulling in libSQL/FastEmbed. They are
// re-exported here so existing `crate::db::SearchResult`-style imports keep
// working.

pub use origin_types::{
    AgentActivityRow, AgentConnection, DomainInfo, Entity, EntityDetail, EntitySearchResult,
    HomeStats, IndexedFileInfo, MemoryItem, MemoryStats, MemoryVersionItem, Observation, Profile,
    RejectionRecord, Relation, RelationWithEntity, SearchResult, Space, TopMemory, TypeBreakdown,
};

// --- Types still unique to the DB layer ----------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDetail {
    pub id: String,
    pub content: String,
    pub title: String,
    pub source_id: String,
    pub chunk_index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_start: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_end: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DistillationCluster {
    pub source_ids: Vec<String>,
    pub contents: Vec<String>,
    pub entity_id: Option<String>,
    pub entity_name: Option<String>,
    pub domain: Option<String>,
    pub estimated_tokens: usize,
}

/// A pending revision waiting for human approval (Protected tier supersede).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRevision {
    pub source_id: String,
    pub content: String,
    pub source_agent: Option<String>,
}

/// A memory chunk that needs its embedding refreshed.
#[derive(Debug, Clone)]
pub struct PendingReembed {
    /// The `id` primary key of the chunk row (passed to `reembed_memory`).
    pub chunk_id: String,
    /// The `source_id` of the parent memory (used for look-ups and assertions).
    pub source_id: String,
    /// The text to embed (source_text when available, otherwise content).
    pub embed_text: String,
}

/// An activity row from the consolidated activities table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityRow {
    pub id: String,
    pub started_at: i64,
    pub ended_at: i64,
}

/// A capture reference row from the consolidated capture_refs table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRefRow {
    pub source_id: String,
    pub activity_id: String,
    pub snapshot_id: Option<String>,
    pub app_name: String,
    pub window_title: String,
    pub timestamp: i64,
    pub source: String,
}

/// A session snapshot row from the consolidated session_snapshots table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshotRow {
    pub id: String,
    pub activity_id: String,
    pub started_at: i64,
    pub ended_at: i64,
    pub primary_apps: Vec<String>,
    pub summary: String,
    pub tags: Vec<String>,
    pub capture_count: usize,
}

/// A refinement queue proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementProposal {
    pub id: String,
    pub action: String,
    pub source_ids: Vec<String>,
    pub payload: Option<String>,
    pub confidence: f64,
    pub status: String,
    pub created_at: String,
}

/// A consolidation candidate: a domain with N+ decayed memories.
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub domain: Option<String>,
    pub count: i64,
}

/// Sync state for a file tracked by a knowledge source.
#[derive(Debug, Clone)]
pub struct FileSyncState {
    pub source_id: String,
    pub file_path: String,
    pub mtime_ns: i64,
    pub content_hash: String,
    pub last_synced_at: i64,
}

// ===== Schema =====

const SCHEMA: &str = "
-- Memories (primary memory storage)
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    chunk_index INTEGER NOT NULL,
    last_modified INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,
    language TEXT,
    byte_start INTEGER,
    byte_end INTEGER,
    semantic_unit TEXT,
    memory_type TEXT,
    domain TEXT,
    source_agent TEXT,
    confidence REAL,
    confirmed INTEGER,
    supersedes TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    pending_revision INTEGER DEFAULT 0,
    word_count INTEGER NOT NULL DEFAULT 0,
    entity_id TEXT,
    enrichment_status TEXT NOT NULL DEFAULT 'enriched',
    quality TEXT CHECK(quality IN ('low', 'medium', 'high')),
    is_recap INTEGER NOT NULL DEFAULT 0,
    supersede_mode TEXT NOT NULL DEFAULT 'hide',
    structured_fields TEXT,
    retrieval_cue TEXT,
    source_text TEXT,
    created_at INTEGER,
    stability TEXT NOT NULL DEFAULT 'new',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed INTEGER,
    refinement_status TEXT,
    effective_confidence REAL,
    embedding F32_BLOB(768),
    version INTEGER DEFAULT 1,
    changelog TEXT DEFAULT '[]'
);

-- Access log: per-source access events for recency/frequency tracking
CREATE TABLE IF NOT EXISTS access_log (
    source_id TEXT NOT NULL,
    accessed_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(accessed_at);
CREATE INDEX IF NOT EXISTS idx_access_log_source ON access_log(source_id);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_source_id ON memories(source_id);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_last_modified ON memories(last_modified);
CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type) WHERE memory_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id) WHERE entity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_enrichment_status ON memories(enrichment_status) WHERE enrichment_status != 'enriched';
CREATE INDEX IF NOT EXISTS idx_memories_is_recap ON memories(is_recap) WHERE is_recap = 1;
CREATE INDEX IF NOT EXISTS idx_memories_pending_revision ON memories(pending_revision) WHERE pending_revision != 0;
CREATE INDEX IF NOT EXISTS idx_memories_stability ON memories(stability) WHERE source = 'memory';

-- Knowledge graph: Entities
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    domain TEXT,
    source_agent TEXT,
    confidence REAL,
    confirmed INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    embedding F32_BLOB(768)
);

-- Knowledge graph: Observations
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    source_agent TEXT,
    confidence REAL,
    confirmed INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL
);

-- Knowledge graph: Relations
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    from_entity TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_entity TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    source_agent TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity);

-- User profile (single row)
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    display_name TEXT,
    email TEXT,
    bio TEXT,
    avatar_path TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Agent registry
CREATE TABLE IF NOT EXISTS agent_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    display_name TEXT,
    agent_type TEXT NOT NULL DEFAULT 'api',
    description TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    trust_level TEXT NOT NULL DEFAULT 'full',
    last_seen_at INTEGER,
    memory_count INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_connections_name ON agent_connections(name);

-- Onboarding milestones: fire-once events that drive post-wizard UX
CREATE TABLE IF NOT EXISTS onboarding_milestones (
    id TEXT PRIMARY KEY,
    first_triggered_at INTEGER NOT NULL,
    acknowledged_at INTEGER,
    payload TEXT
);

-- Spaces: project-scoped memory containers
CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    suggested INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain) WHERE domain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain) WHERE domain IS NOT NULL;

-- Rejection log: quality gate diagnostics
CREATE TABLE IF NOT EXISTS rejected_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_agent TEXT,
    rejection_reason TEXT NOT NULL,
    rejection_detail TEXT,
    similarity_score REAL,
    similar_to_source_id TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejected_reason ON rejected_memories(rejection_reason);
CREATE INDEX IF NOT EXISTS idx_rejected_agent ON rejected_memories(source_agent);
";

// FTS5 and vector indexes created separately (may not support IF NOT EXISTS)
const FTS_SCHEMA: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, title,
    content=memories,
    content_rowid=rowid
);
";

const FTS_TRIGGERS: &str = "
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, title) VALUES (new.rowid, new.content, new.title);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, title) VALUES('delete', old.rowid, old.content, old.title);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content, title ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, title) VALUES('delete', old.rowid, old.content, old.title);
    INSERT INTO memories_fts(rowid, content, title) VALUES (new.rowid, new.content, new.title);
END;
";

// ===== MemoryDB =====

pub struct MemoryDB {
    _db: libsql::Database,
    pub(crate) conn: tokio::sync::Mutex<libsql::Connection>,
    embedder: Arc<std::sync::Mutex<TextEmbedding>>,
    chunker: ChunkingEngine,
    embedding_cache: std::sync::Mutex<EmbeddingCache>,
}

/// Returns true when a memory title looks like it would make a poor snippet —
/// e.g. too short, a raw code statement, or a bare URL.  Used in
/// `list_recent_retrievals` to prefer the content field over a garbage title.
fn title_looks_garbage(title: &str) -> bool {
    // Too short to be meaningful
    if title.chars().count() < 10 {
        return true;
    }
    // Starts with common code prefixes (rough heuristic — good enough for the home card)
    const CODE_PREFIXES: &[&str] = &[
        "const ",
        "let ",
        "var ",
        "function ",
        "async function",
        "await ",
        "import ",
        "export ",
        "return ",
        "class ",
        "def ",
        "http://",
        "https://",
    ];
    let lower = title.to_ascii_lowercase();
    CODE_PREFIXES.iter().any(|p| lower.starts_with(p))
}

impl MemoryDB {
    /// Convert a multi-word query into FTS5 OR query.
    /// "vector embedding model" → "vector OR embedding OR model"
    fn fts_or_query(query: &str) -> String {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() <= 1 {
            return query.to_string();
        }
        words.join(" OR ")
    }

    pub async fn new(db_path: &Path, emitter: Arc<dyn EventEmitter>) -> Result<Self, OriginError> {
        std::fs::create_dir_all(db_path)?;
        let db_file = db_path.join("origin_memory.db");
        log::warn!("[memory_db] opening DB at {}", db_file.display());

        let db = libsql::Builder::new_local(db_file.to_str().unwrap_or("origin_memory.db"))
            .build()
            .await
            .map_err(|e| OriginError::VectorDb(format!("libsql open: {}", e)))?;
        log::warn!("[memory_db] DB opened, connecting...");

        let conn = db
            .connect()
            .map_err(|e| OriginError::VectorDb(format!("libsql connect: {}", e)))?;
        log::warn!("[memory_db] connected, running PRAGMA...");

        // Enable WAL mode and foreign keys
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .await
            .map_err(|e| OriginError::VectorDb(format!("pragma: {}", e)))?;
        log::warn!("[memory_db] PRAGMA done, checking chunks cleanup...");

        // (chunks cleanup block follows)
        // After cleanup, log before schema:
        // log is added after the cleanup block below

        // Clean up legacy `chunks` table artifacts from pre-migration DBs.
        // Migration 24 renamed chunks→memories but may have left behind the old
        // table, indexes, FTS virtual table, and triggers. These must be removed
        // before SCHEMA creation to avoid conflicts.
        {
            let has_chunks =
                {
                    let mut rows = conn.query(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks'",
                    libsql::params![],
                ).await.map_err(|e| OriginError::VectorDb(format!("check chunks: {e}")))?;
                    rows.next()
                        .await
                        .ok()
                        .flatten()
                        .and_then(|r| r.get::<i64>(0).ok())
                        .unwrap_or(0)
                        > 0
                };

            if has_chunks {
                log::info!("[memory_db] cleaning up legacy 'chunks' table artifacts");
                // Drop triggers first (reference chunks table)
                let _ = conn
                    .execute("DROP TRIGGER IF EXISTS chunks_fts_insert", ())
                    .await;
                let _ = conn
                    .execute("DROP TRIGGER IF EXISTS chunks_fts_delete", ())
                    .await;
                let _ = conn
                    .execute("DROP TRIGGER IF EXISTS chunks_fts_update", ())
                    .await;
                // Drop FTS virtual table
                let _ = conn.execute("DROP TABLE IF EXISTS chunks_fts", ()).await;
                // Drop indexes
                for idx in &[
                    "idx_chunks_source_id",
                    "idx_chunks_source",
                    "idx_chunks_last_modified",
                    "idx_chunks_memory_type",
                    "idx_chunks_supersedes",
                    "idx_chunks_entity_id",
                    "idx_chunks_enrichment_status",
                    "idx_chunks_is_recap",
                    "idx_chunks_domain",
                    "idx_chunks_pending_revision",
                ] {
                    let _ = conn
                        .execute(&format!("DROP INDEX IF EXISTS {}", idx), ())
                        .await;
                }
                // Drop vector index shadow table
                let _ = conn
                    .execute("DROP TABLE IF EXISTS chunks_vec_idx_shadow", ())
                    .await;
                // Drop the empty chunks table itself
                let _ = conn.execute("DROP TABLE IF EXISTS chunks", ()).await;
                log::info!("[memory_db] legacy 'chunks' cleanup complete");
            }
        }

        log::warn!("[memory_db] creating schema...");
        // Create core tables (IF NOT EXISTS — no-op for existing DBs, creates for new)
        // Note: SCHEMA includes all columns (including migration 10 reconciliation columns).
        // For existing DBs, run_migrations() handles ALTER TABLE additions.
        conn.execute_batch(SCHEMA)
            .await
            .map_err(|e| OriginError::VectorDb(format!("schema: {}", e)))?;
        log::warn!("[memory_db] schema created");

        log::warn!("[memory_db] creating FTS...");
        // Create FTS5 virtual table
        if let Err(e) = conn.execute_batch(FTS_SCHEMA).await {
            log::warn!(
                "[memory_db] FTS5 creation failed (may already exist): {}",
                e
            );
        }
        log::warn!("[memory_db] FTS done, creating triggers...");

        // Create FTS sync triggers
        if let Err(e) = conn.execute_batch(FTS_TRIGGERS).await {
            log::warn!("[memory_db] FTS triggers failed (may already exist): {}", e);
        }
        // Create vector indexes only for fresh databases.
        // For existing DBs, indexes already exist and CREATE INDEX IF NOT EXISTS
        // with libsql_vector_idx triggers a full DiskANN rebuild that blocks for minutes.
        // Detect fresh DB by checking if this is a new file (no memories yet).
        let is_fresh_db = {
            let mut rows = conn
                .query("SELECT COUNT(*) FROM memories", libsql::params![])
                .await
                .map_err(|e| OriginError::VectorDb(format!("count memories: {e}")))?;
            rows.next()
                .await
                .ok()
                .flatten()
                .and_then(|r| r.get::<i64>(0).ok())
                .unwrap_or(0)
                == 0
        };
        if is_fresh_db {
            log::warn!("[memory_db] fresh DB, creating vector indexes...");
            if let Err(e) = conn.execute(
                "CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )", (),
            ).await {
                log::warn!("[memory_db] memories vector index creation failed: {}", e);
            }
            if let Err(e) = conn.execute(
                "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )", (),
            ).await {
                log::warn!("[memory_db] entities vector index creation failed: {}", e);
            }
        } else {
            log::warn!("[memory_db] existing DB, skipping vector index creation");
        }

        // Initialize embedding model — use absolute cache path so model is found
        // regardless of CWD (pnpm tauri dev vs cargo test have different CWDs).
        //
        // IMPORTANT: TextEmbedding::try_new() is synchronous and blocks for 10-30s
        // while loading the 140MB ONNX model + graph optimization.
        // Run on a dedicated std::thread (completely outside tokio) and return
        // the result via a oneshot channel. This avoids starving the tokio
        // runtime — spawn_blocking still occupies runtime-managed threads and
        // its JoinHandle needs worker threads to poll, which caused deadlocks
        // when combined with block_on calls during startup.
        let embed_cache_dir = resolve_fastembed_cache_dir(db_path);
        log::info!(
            "[memory_db] starting embedder init (std::thread), cache={}",
            embed_cache_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<default>".into())
        );
        let (embed_tx, embed_rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("embedder-init".into())
            .spawn(move || {
                log::info!("[memory_db] embedder thread: loading ONNX model...");
                let mut opts = InitOptions::new(EmbeddingModel::BGEBaseENV15Q)
                    .with_show_download_progress(true);
                if let Some(cache) = embed_cache_dir {
                    opts = opts.with_cache_dir(cache);
                }
                let result = TextEmbedding::try_new(opts);
                let _ = embed_tx.send(result);
            })
            .map_err(|e| OriginError::Embedding(format!("spawn embedder thread: {}", e)))?;
        let embedder = embed_rx
            .await
            .map_err(|_| OriginError::Embedding("embedder thread panicked".into()))?
            .map_err(|e| OriginError::Embedding(format!("init embedder: {}", e)))?;

        log::info!("[memory_db] initialized at {}", db_file.display());

        let instance = Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            embedder: Arc::new(std::sync::Mutex::new(embedder)),
            chunker: ChunkingEngine::new(),
            embedding_cache: std::sync::Mutex::new(EmbeddingCache::new(200)),
        };

        // Run schema migrations for existing databases
        instance.run_migrations(emitter.as_ref()).await?;

        // Ensure a default profile always exists
        instance.bootstrap_profile().await?;

        Ok(instance)
    }

    /// Fast constructor for eval loops: reuses a pre-loaded embedder.
    /// Creates a fresh ephemeral DB with schema + migrations but skips the 10-30s
    /// embedding model load. Use `create_shared_embedder()` to create the embedder once.
    pub async fn new_with_shared_embedder(
        db_path: &Path,
        emitter: Arc<dyn EventEmitter>,
        embedder: Arc<std::sync::Mutex<TextEmbedding>>,
    ) -> Result<Self, OriginError> {
        std::fs::create_dir_all(db_path)?;
        let db_file = db_path.join("origin_memory.db");

        let db = libsql::Builder::new_local(db_file.to_str().unwrap_or("origin_memory.db"))
            .build()
            .await
            .map_err(|e| OriginError::VectorDb(format!("libsql open: {}", e)))?;

        let conn = db
            .connect()
            .map_err(|e| OriginError::VectorDb(format!("libsql connect: {}", e)))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .await
            .map_err(|e| OriginError::VectorDb(format!("pragma: {}", e)))?;

        conn.execute_batch(SCHEMA)
            .await
            .map_err(|e| OriginError::VectorDb(format!("schema: {}", e)))?;

        if let Err(e) = conn.execute_batch(FTS_SCHEMA).await {
            log::warn!("[memory_db] FTS5 creation failed: {}", e);
        }
        if let Err(e) = conn.execute_batch(FTS_TRIGGERS).await {
            log::warn!("[memory_db] FTS triggers failed: {}", e);
        }

        // Vector indexes for fresh DB
        let _ = conn.execute(
            "CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories (
                libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
            )", (),
        ).await;
        let _ = conn.execute(
            "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities (
                libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
            )", (),
        ).await;

        let instance = Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            embedder,
            chunker: ChunkingEngine::new(),
            embedding_cache: std::sync::Mutex::new(EmbeddingCache::new(200)),
        };

        instance.run_migrations(emitter.as_ref()).await?;
        instance.bootstrap_profile().await?;

        Ok(instance)
    }

    /// Create a shared embedder that can be passed to `new_with_shared_embedder`.
    /// Loads the BGE-Base-EN-v1.5-Q model once (10-30s), then reuse across DB instances.
    pub async fn create_shared_embedder(
    ) -> Result<Arc<std::sync::Mutex<TextEmbedding>>, OriginError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("embedder-init-shared".into())
            .spawn(move || {
                let opts = InitOptions::new(EmbeddingModel::BGEBaseENV15Q)
                    .with_show_download_progress(true);
                let result = TextEmbedding::try_new(opts);
                let _ = tx.send(result);
            })
            .map_err(|e| OriginError::Embedding(format!("spawn embedder: {}", e)))?;
        let embedder = rx
            .await
            .map_err(|_| OriginError::Embedding("embedder thread panicked".into()))?
            .map_err(|e| OriginError::Embedding(format!("init embedder: {}", e)))?;
        Ok(Arc::new(std::sync::Mutex::new(embedder)))
    }

    /// Lightweight constructor for eval ablation tests (model × prefix 2x2).
    /// Creates a fresh ephemeral DB with the given embedding model and dimension.
    /// Skips migrations (fresh DB only) and profile bootstrap.
    pub async fn new_with_embed_config(
        db_path: &Path,
        config: EmbedConfig,
    ) -> Result<Self, OriginError> {
        std::fs::create_dir_all(db_path)?;
        let db_file = db_path.join("origin_memory.db");

        let db = libsql::Builder::new_local(db_file.to_str().unwrap_or("origin_memory.db"))
            .build()
            .await
            .map_err(|e| OriginError::VectorDb(format!("libsql open: {}", e)))?;

        let conn = db
            .connect()
            .map_err(|e| OriginError::VectorDb(format!("libsql connect: {}", e)))?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .await
            .map_err(|e| OriginError::VectorDb(format!("pragma: {}", e)))?;

        // Build schema with configurable vector dimension
        let schema = SCHEMA.replace("F32_BLOB(768)", &format!("F32_BLOB({})", config.dim));
        conn.execute_batch(&schema)
            .await
            .map_err(|e| OriginError::VectorDb(format!("schema: {}", e)))?;

        if let Err(e) = conn.execute_batch(FTS_SCHEMA).await {
            log::warn!("[memory_db] FTS5 creation failed: {}", e);
        }
        if let Err(e) = conn.execute_batch(FTS_TRIGGERS).await {
            log::warn!("[memory_db] FTS triggers failed: {}", e);
        }

        // Vector indexes for fresh DB
        let _ = conn.execute(
                "CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )", (),
        ).await;
        let _ = conn.execute(
                "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )", (),
        ).await;

        // Init embedding model — use default cache dir (persistent across runs)
        let model = config.model;
        let (embed_tx, embed_rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("embedder-init-eval".into())
            .spawn(move || {
                let result = TextEmbedding::try_new(
                    InitOptions::new(model).with_show_download_progress(true),
                );
                let _ = embed_tx.send(result);
            })
            .map_err(|e| OriginError::Embedding(format!("spawn embedder thread: {}", e)))?;
        let embedder = embed_rx
            .await
            .map_err(|_| OriginError::Embedding("embedder thread panicked".into()))?
            .map_err(|e| OriginError::Embedding(format!("init embedder: {}", e)))?;

        Ok(Self {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            embedder: Arc::new(std::sync::Mutex::new(embedder)),
            chunker: ChunkingEngine::new(),
            embedding_cache: std::sync::Mutex::new(EmbeddingCache::new(200)),
        })
    }

    // ===== Migrations =====

    /// Get the column names for a given table via PRAGMA table_info.
    async fn get_table_columns(&self, table: &str) -> Result<HashSet<String>, OriginError> {
        let conn = self.conn.lock().await;
        // table name is hardcoded by callers (not user input), safe to format
        let sql = format!("PRAGMA table_info({})", table);
        let mut rows = conn
            .query(&sql, ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_table_columns({}): {}", table, e)))?;
        let mut columns = HashSet::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            // PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
            if let Ok(name) = row.get::<String>(1) {
                columns.insert(name);
            }
        }
        Ok(columns)
    }

    /// Apply incremental migrations based on PRAGMA user_version.
    /// Idempotent: checks column existence before ALTER TABLE.
    pub async fn run_migrations(&self, emitter: &dyn EventEmitter) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;

        // Read current user_version
        let mut rows = conn
            .query("PRAGMA user_version", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("read user_version: {}", e)))?;
        let version: i64 = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            0
        };
        drop(rows);

        if version < 1 {
            // Migration 1: reserved (initial schema — no-op since CREATE TABLE IF NOT EXISTS handles it)
            conn.execute("PRAGMA user_version = 1", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=1: {}", e)))?;
            log::info!("[memory_db] migration: set user_version = 1 (baseline)");
        }

        // Release lock for get_table_columns (which also acquires it)
        drop(conn);

        if version < 2 {
            // Migration 2: Add profile extended fields + memories pinned column + unpin trigger

            // --- profiles: email, bio, avatar_path ---
            let profile_cols = self.get_table_columns("profiles").await?;
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration2 begin: {}", e)))?;

            if !profile_cols.contains("email") {
                conn.execute("ALTER TABLE profiles ADD COLUMN email TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("alter profiles add email: {}", e))
                    })?;
            }
            if !profile_cols.contains("bio") {
                conn.execute("ALTER TABLE profiles ADD COLUMN bio TEXT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("alter profiles add bio: {}", e)))?;
            }
            if !profile_cols.contains("avatar_path") {
                conn.execute("ALTER TABLE profiles ADD COLUMN avatar_path TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("alter profiles add avatar_path: {}", e))
                    })?;
            }

            // --- memories: pinned column ---
            drop(conn);
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;

            if !chunk_cols.contains("pinned") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("alter memories add pinned: {}", e)))?;
            }

            // --- unpin trigger ---
            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS unpin_on_unconfirm AFTER UPDATE OF confirmed ON memories WHEN NEW.confirmed = 0 BEGIN UPDATE memories SET pinned = 0 WHERE source_id = NEW.source_id; END",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create unpin trigger: {}", e)))?;

            conn.execute("PRAGMA user_version = 2", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=2: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration2 commit: {}", e)))?;

            log::info!("[memory_db] migration 2: added profile fields (email, bio, avatar_path), memories.pinned, unpin trigger");
        }

        // Re-read version for migration 3 (migration 2 may have just run)
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("PRAGMA user_version", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("read user_version m3: {}", e)))?;
        let version: i64 = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            0
        };
        drop(rows);
        drop(conn);

        if version < 3 {
            // Migration 3: Add pending_revision column to memories
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration3 begin: {}", e)))?;

            if !chunk_cols.contains("pending_revision") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN pending_revision INTEGER DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("alter memories add pending_revision: {}", e))
                })?;
            }

            conn.execute(
                "UPDATE memories SET pending_revision = 0 WHERE pending_revision IS NULL",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("backfill pending_revision: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_pending_revision ON memories(pending_revision) WHERE pending_revision != 0",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create pending_revision index: {}", e)))?;

            conn.execute("PRAGMA user_version = 3", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=3: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration3 commit: {}", e)))?;

            log::info!("[memory_db] migration 3: added memories.pending_revision column + index");
        }

        // Migration 4: Refinement pipeline columns + queue table
        if version < 4 {
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration4 begin: {}", e)))?;

            if !chunk_cols.contains("access_count") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("alter memories add access_count: {}", e))
                })?;
            }
            if !chunk_cols.contains("last_accessed") {
                conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("alter memories add last_accessed: {}", e))
                    })?;
            }
            if !chunk_cols.contains("refinement_status") {
                conn.execute("ALTER TABLE memories ADD COLUMN refinement_status TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!(
                            "alter memories add refinement_status: {}",
                            e
                        ))
                    })?;
            }
            if !chunk_cols.contains("effective_confidence") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN effective_confidence REAL",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("alter memories add effective_confidence: {}", e))
                })?;
            }

            conn.execute(
                "CREATE TABLE IF NOT EXISTS refinement_queue (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    source_ids TEXT NOT NULL,
                    payload TEXT,
                    confidence REAL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT (datetime('now')),
                    resolved_at TEXT
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create refinement_queue: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_refinement_status ON refinement_queue(status)
                 WHERE status IN ('pending', 'awaiting_review')",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create refinement_status index: {}", e)))?;

            conn.execute("PRAGMA user_version = 4", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=4: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration4 commit: {}", e)))?;

            log::info!("[memory_db] migration 4: added refinement columns + queue table");
        }

        // Migration 5: Session tables consolidated from session_db
        if version < 5 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration5 begin: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY,
                    started_at INTEGER NOT NULL,
                    ended_at INTEGER NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create activities: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS capture_refs (
                    source_id TEXT PRIMARY KEY,
                    activity_id TEXT NOT NULL,
                    snapshot_id TEXT,
                    app_name TEXT NOT NULL,
                    window_title TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    source TEXT NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create capture_refs: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS session_snapshots (
                    id TEXT PRIMARY KEY,
                    activity_id TEXT NOT NULL,
                    started_at INTEGER NOT NULL,
                    ended_at INTEGER NOT NULL,
                    primary_apps TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    capture_count INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create session_snapshots: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_captures_activity ON capture_refs(activity_id)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create capture_refs idx: {}", e)))?;
            conn.execute("CREATE INDEX IF NOT EXISTS idx_captures_unpackaged ON capture_refs(activity_id) WHERE snapshot_id IS NULL", ())
                .await.map_err(|e| OriginError::VectorDb(format!("create unpackaged idx: {}", e)))?;
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON session_snapshots(started_at DESC)", ())
                .await.map_err(|e| OriginError::VectorDb(format!("create snapshots time idx: {}", e)))?;

            conn.execute("PRAGMA user_version = 5", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=5: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration5 commit: {}", e)))?;

            log::info!("[memory_db] migration 5: session tables consolidated from session_db");
        }

        // Migration 6: word_count column on memories + access_log table
        if version < 6 {
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration6 begin: {}", e)))?;

            if !chunk_cols.contains("word_count") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN word_count INTEGER NOT NULL DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("alter memories add word_count: {}", e))
                })?;
            }

            conn.execute(
                "UPDATE memories SET word_count = LENGTH(content) - LENGTH(REPLACE(content, ' ', '')) + 1 WHERE content IS NOT NULL AND content != ''",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("backfill word_count: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS access_log (source_id TEXT NOT NULL, accessed_at INTEGER NOT NULL)",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("create access_log: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(accessed_at)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create access_log time index: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_access_log_source ON access_log(source_id)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create access_log source index: {}", e)))?;

            conn.execute("PRAGMA user_version = 6", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=6: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration6 commit: {}", e)))?;

            log::info!("[memory_db] migration 6: added word_count column + access_log table");
        }

        // Migration 7: Briefing cache table
        if version < 7 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration7 begin: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS briefing_cache (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    content TEXT NOT NULL,
                    generated_at INTEGER NOT NULL,
                    memory_count INTEGER NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create briefing_cache: {}", e)))?;

            conn.execute("PRAGMA user_version = 7", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=7: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration7 commit: {}", e)))?;

            log::info!("[memory_db] migration 7: added briefing_cache table");
        }

        // Migration 8: Narrative cache table
        if version < 8 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration8 begin: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS narrative_cache (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    content TEXT NOT NULL,
                    generated_at INTEGER NOT NULL,
                    memory_count INTEGER NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create narrative_cache: {}", e)))?;

            conn.execute("PRAGMA user_version = 8", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=8: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration8 commit: {}", e)))?;

            log::info!("[memory_db] migration 8: added narrative_cache table");
        }

        // Migration 9: Agent activity table (impact tracking)
        if version < 9 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration9 begin: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    agent_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    memory_ids TEXT,
                    query TEXT,
                    detail TEXT
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create agent_activity: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_time ON agent_activity(timestamp)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create idx_activity_time: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_agent ON agent_activity(agent_name)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create idx_activity_agent: {}", e)))?;

            conn.execute("PRAGMA user_version = 9", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=9: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration9 commit: {}", e)))?;

            log::info!("[memory_db] migration 9: added agent_activity table");
        }

        // Migration 10: Memory reconciliation schema — new columns on memories
        if version < 10 {
            // ALTER TABLE must run outside a transaction in libSQL
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;

            if !chunk_cols.contains("entity_id") {
                conn.execute("ALTER TABLE memories ADD COLUMN entity_id TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration10 add entity_id: {}", e))
                    })?;
            }
            if !chunk_cols.contains("enrichment_status") {
                conn.execute("ALTER TABLE memories ADD COLUMN enrichment_status TEXT NOT NULL DEFAULT 'enriched'", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration10 add enrichment_status: {}", e)))?;
            }
            if !chunk_cols.contains("quality") {
                conn.execute("ALTER TABLE memories ADD COLUMN quality TEXT CHECK(quality IN ('low', 'medium', 'high'))", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration10 add quality: {}", e)))?;
            }
            if !chunk_cols.contains("is_recap") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN is_recap INTEGER NOT NULL DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration10 add is_recap: {}", e)))?;
            }
            if !chunk_cols.contains("supersede_mode") {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN supersede_mode TEXT NOT NULL DEFAULT 'hide'",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration10 add supersede_mode: {}", e))
                })?;
            }

            // Data migration + indexes inside a transaction
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration10 begin: {}", e)))?;

            // Migrate existing recap-type memories to is_recap flag
            conn.execute(
                "UPDATE memories SET is_recap = 1, memory_type = 'fact' WHERE memory_type = 'recap'",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("migration10 recap→is_recap: {}", e)))?;

            // Migrate correction → fact
            conn.execute(
                "UPDATE memories SET memory_type = 'fact' WHERE memory_type = 'correction'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration10 correction→fact: {}", e)))?;

            // Migrate custom → fact
            conn.execute(
                "UPDATE memories SET memory_type = 'fact' WHERE memory_type = 'custom'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration10 custom→fact: {}", e)))?;

            // Create new indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id) WHERE entity_id IS NOT NULL",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("migration10 idx_entity_id: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_enrichment_status ON memories(enrichment_status) WHERE enrichment_status != 'enriched'",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("migration10 idx_enrichment: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_is_recap ON memories(is_recap) WHERE is_recap = 1",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("migration10 idx_is_recap: {}", e)))?;

            conn.execute(
                "UPDATE memories SET supersede_mode = 'archive' WHERE memory_type = 'decision'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration10 decision→archive: {}", e)))?;

            conn.execute("PRAGMA user_version = 10", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=10: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration10 commit: {}", e)))?;

            log::info!("[memory_db] migration 10: added entity_id, enrichment_status, quality, is_recap, supersede_mode columns; migrated recap/correction/custom → fact");
        }

        // Migration 11: Structured memory fields + retrieval cues (from main)
        if version < 11 {
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;

            if !chunk_cols.contains("structured_fields") {
                conn.execute("ALTER TABLE memories ADD COLUMN structured_fields TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration11 add structured_fields: {}", e))
                    })?;
            }
            if !chunk_cols.contains("retrieval_cue") {
                conn.execute("ALTER TABLE memories ADD COLUMN retrieval_cue TEXT", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration11 add retrieval_cue: {}", e))
                    })?;
            }

            conn.execute("PRAGMA user_version = 11", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=11: {}", e)))?;

            log::info!(
                "[memory_db] migration 11: added structured_fields and retrieval_cue columns"
            );
        }

        // Migration 12: Spaces table + domain indexes
        if version < 12 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration12 begin: {}", e)))?;

            conn.execute(
                "CREATE TABLE IF NOT EXISTS spaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    suggested INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("create spaces: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain) WHERE domain IS NOT NULL",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("idx_memories_domain: {}", e)))?;

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain) WHERE domain IS NOT NULL",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("idx_entities_domain: {}", e)))?;

            conn.execute(
                "INSERT OR IGNORE INTO spaces (id, name, description, suggested, created_at, updated_at)
                 SELECT lower(hex(randomblob(16))), domain, NULL, 1, unixepoch('now'), unixepoch('now')
                 FROM (
                   SELECT DISTINCT domain FROM memories WHERE domain IS NOT NULL AND domain != ''
                   UNION
                   SELECT DISTINCT domain FROM entities WHERE domain IS NOT NULL AND domain != ''
                 )",
                (),
            ).await.map_err(|e| OriginError::VectorDb(format!("seed spaces: {}", e)))?;

            conn.execute("PRAGMA user_version = 12", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=12: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration12 commit: {}", e)))?;

            log::info!("[memory_db] migration 12: spaces table + domain indexes");
        }

        // Migration 13: Clean up empty-string domains and spaces
        if version < 13 {
            let conn = self.conn.lock().await;
            conn.execute("DELETE FROM spaces WHERE name = ''", ())
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration13 delete empty space: {}", e))
                })?;
            conn.execute("UPDATE memories SET domain = NULL WHERE domain = ''", ())
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration13 null empty chunk domains: {}", e))
                })?;
            conn.execute("UPDATE entities SET domain = NULL WHERE domain = ''", ())
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration13 null empty entity domains: {}", e))
                })?;
            conn.execute("PRAGMA user_version = 13", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=13: {}", e)))?;
            log::info!("[memory_db] migration 13: cleaned up empty-string domains and spaces");
        }

        // Migration 14: Add sort_order to spaces
        if version < 14 {
            let space_cols = self.get_table_columns("spaces").await?;
            let conn = self.conn.lock().await;
            if !space_cols.contains("sort_order") {
                conn.execute(
                    "ALTER TABLE spaces ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration14 add sort_order: {}", e)))?;
                // Backfill: assign order by name alphabetically
                conn.execute(
                    "UPDATE spaces SET sort_order = (SELECT COUNT(*) FROM spaces s2 WHERE s2.name < spaces.name)",
                    (),
                ).await.map_err(|e| OriginError::VectorDb(format!("migration14 backfill sort_order: {}", e)))?;
            }
            conn.execute("PRAGMA user_version = 14", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=14: {}", e)))?;
            log::info!("[memory_db] migration 14: added sort_order to spaces");
        }

        // Migration 15: Add starred to spaces
        if version < 15 {
            let space_cols = self.get_table_columns("spaces").await?;
            let conn = self.conn.lock().await;
            if !space_cols.contains("starred") {
                conn.execute(
                    "ALTER TABLE spaces ADD COLUMN starred INTEGER NOT NULL DEFAULT 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration15 add starred: {}", e)))?;
            }
            conn.execute("PRAGMA user_version = 15", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=15: {}", e)))?;
            log::info!("[memory_db] migration 15: added starred to spaces");
        }

        // Migration 16: Mark all non-starred, non-manually-described spaces as suggested
        // (migration 12 seeded domain-derived spaces as confirmed, but they should be suggested)
        if version < 16 {
            let conn = self.conn.lock().await;
            conn.execute(
                "UPDATE spaces SET suggested = 1 WHERE description IS NULL AND starred = 0",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration16 mark suggested: {}", e)))?;
            conn.execute("PRAGMA user_version = 16", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=16: {}", e)))?;
            log::info!("[memory_db] migration 16: marked domain-derived spaces as suggested");
        }

        // Migration 17: Eval system tables (from eval branch)
        if version < 17 {
            let conn = self.conn.lock().await;
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS eval_signals (
                    id TEXT PRIMARY KEY,
                    signal_type TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    query_context TEXT,
                    rank_position INTEGER,
                    created_at INTEGER NOT NULL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_eval_signals_created ON eval_signals(created_at);
                CREATE INDEX IF NOT EXISTS idx_eval_signals_memory ON eval_signals(memory_id);

                CREATE TABLE IF NOT EXISTS eval_judgments (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    rank_position INTEGER,
                    llm_score INTEGER NOT NULL,
                    llm_reason TEXT,
                    search_mode TEXT,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_eval_judgments_created ON eval_judgments(created_at);

                CREATE TABLE IF NOT EXISTS eval_tune_log (
                    id TEXT PRIMARY KEY,
                    tuned_at INTEGER NOT NULL,
                    signals_count INTEGER,
                    knob_changes TEXT NOT NULL,
                    mrr_before REAL,
                    mrr_after REAL,
                    ndcg_before REAL,
                    ndcg_after REAL
                );",
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration 17 eval tables: {}", e)))?;
            conn.execute("PRAGMA user_version = 17", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=17: {}", e)))?;
            log::info!("[memory_db] migration 17: created eval_signals, eval_judgments, eval_tune_log tables");
        }

        // Migration 18: Add source_text column for structured content model
        if version < 18 {
            let has_col = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("PRAGMA table_info(memories)", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                let mut found = false;
                while let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    let name: String = row.get(1).unwrap_or_default();
                    if name == "source_text" {
                        found = true;
                        break;
                    }
                }
                found
            };
            if !has_col {
                let conn = self.conn.lock().await;
                conn.execute("ALTER TABLE memories ADD COLUMN source_text TEXT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("add source_text: {}", e)))?;
            }
            let conn = self.conn.lock().await;
            conn.execute(
                "UPDATE memories SET source_text = content
                 WHERE source = 'memory' AND structured_fields IS NOT NULL AND source_text IS NULL",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("backfill source_text: {}", e)))?;
            conn.execute("PRAGMA user_version = 18", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=18: {}", e)))?;
            log::info!(
                "[memory_db] migration 18: added source_text column + backfilled from content"
            );
        }

        // Migration 19: Flatten existing structured_fields into content, mark for re-embedding
        // Reads each memory with structured_fields, flattens to pipe-delimited string,
        // moves original prose to source_text, and marks enrichment_status = 'reembed_pending'
        // (migration 43 backfills this into needs_reembed later).
        if version < 19 {
            let conn = self.conn.lock().await;
            // Fetch all memories with structured_fields that haven't been flattened yet
            let mut rows = conn
                .query(
                    "SELECT source_id, content, structured_fields FROM memories
                     WHERE source = 'memory' AND structured_fields IS NOT NULL
                     AND chunk_index = 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 19 read: {}", e)))?;

            let mut updates: Vec<(String, String, String)> = Vec::new(); // (source_id, new_content, original_content)
            while let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let source_id: String = row.get(0).unwrap_or_default();
                let content: String = row.get(1).unwrap_or_default();
                let sf: String = row.get(2).unwrap_or_default();
                if let Some(flat) = crate::schema::flatten_structured_fields(&sf) {
                    // Only update if flattened content differs from current content
                    // (skip if already flattened from a previous partial run)
                    if flat != content {
                        updates.push((source_id, flat, content));
                    }
                }
            }
            drop(rows);

            let count = updates.len();
            if count > 0 {
                conn.execute("BEGIN", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                for (source_id, new_content, original_content) in &updates {
                    conn.execute(
                        "UPDATE memories SET content = ?1, source_text = ?2, enrichment_status = 'reembed_pending'
                         WHERE source_id = ?3 AND source = 'memory'",
                        libsql::params![new_content.clone(), original_content.clone(), source_id.clone()],
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 19 update {}: {}", source_id, e)))?;
                }
                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                log::info!(
                    "[memory_db] migration 19: flattened {} memories with structured fields",
                    count
                );
            }

            conn.execute("PRAGMA user_version = 19", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=19: {}", e)))?;
            log::info!(
                "[memory_db] migration 19: structured content flatten + reembed marking complete"
            );
        }

        // Migration 20: Re-run flatten for DBs that ran old migration 19 (reembed-only)
        // before the flatten logic was added. Idempotent — skips already-flattened rows.
        if version < 20 {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source_id, content, structured_fields FROM memories
                     WHERE source = 'memory' AND structured_fields IS NOT NULL
                     AND chunk_index = 0",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 20 read: {}", e)))?;

            let mut updates: Vec<(String, String, String)> = Vec::new();
            while let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let source_id: String = row.get(0).unwrap_or_default();
                let content: String = row.get(1).unwrap_or_default();
                let sf: String = row.get(2).unwrap_or_default();
                if let Some(flat) = crate::schema::flatten_structured_fields(&sf) {
                    if flat != content {
                        updates.push((source_id, flat, content));
                    }
                }
            }
            drop(rows);

            let count = updates.len();
            if count > 0 {
                conn.execute("BEGIN", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                for (source_id, new_content, original_content) in &updates {
                    conn.execute(
                        "UPDATE memories SET content = ?1, source_text = ?2, enrichment_status = 'reembed_pending'
                         WHERE source_id = ?3 AND source = 'memory'",
                        libsql::params![new_content.clone(), original_content.clone(), source_id.clone()],
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 20 update {}: {}", source_id, e)))?;
                }
                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                log::info!(
                    "[memory_db] migration 20: flattened {} existing structured memories",
                    count
                );
            }

            conn.execute("PRAGMA user_version = 20", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=20: {}", e)))?;
        }

        if version < 21 {
            // Migration 21: Add created_at column to memories.
            // created_at is immutable (set once at insert), unlike last_modified which
            // gets updated by enrichment/entity extraction. Recaps use created_at to
            // avoid pulling old memories that were recently re-processed.
            let chunk_cols = self.get_table_columns("memories").await?;
            let conn = self.conn.lock().await;

            if !chunk_cols.contains("created_at") {
                conn.execute("ALTER TABLE memories ADD COLUMN created_at INTEGER", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration 21 add created_at: {}", e))
                    })?;

                // Backfill from last_modified for existing rows
                conn.execute(
                    "UPDATE memories SET created_at = last_modified WHERE created_at IS NULL",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 21 backfill: {}", e)))?;

                log::info!("[memory_db] migration 21: added memories.created_at, backfilled from last_modified");
            }

            conn.execute("PRAGMA user_version = 21", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=21: {}", e)))?;
        }

        // Migration 22: Add stability column (new/learned/confirmed) replacing boolean confirmed
        if version < 22 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 22 begin: {}", e)))?;

            // Add stability column
            let has_stability = conn
                .query(
                    "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'stability'",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 22 check: {}", e)))?
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
                .map(|r| r.get::<i64>(0).unwrap_or(0))
                .unwrap_or(0);

            if has_stability == 0 {
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN stability TEXT DEFAULT 'new'",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 22 add stability: {}", e)))?;
            }

            // Migrate existing data: confirmed=1 → 'confirmed', else → 'new'
            conn.execute(
                "UPDATE memories SET stability = CASE WHEN confirmed = 1 THEN 'confirmed' ELSE 'new' END WHERE source = 'memory'", ()
            ).await.map_err(|e| OriginError::VectorDb(format!("migration 22 migrate data: {}", e)))?;

            // Create index on stability for efficient queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_stability ON memories(stability) WHERE source = 'memory'", ()
            ).await.map_err(|e| OriginError::VectorDb(format!("migration 22 stability idx: {}", e)))?;

            conn.execute("PRAGMA user_version = 22", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("set user_version=22: {}", e)))?;
            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 22 commit: {}", e)))?;

            log::info!(
                "[memory_db] migration 22: added stability column, migrated confirmed→stability"
            );
        }

        // Migration 23: Stop pipe-delimited flattening — restore prose to content.
        // Previously, flatten_structured_fields() overwrote content with pipe-delimited
        // text and saved original to source_text. Restore prose for better display/FTS/embedding.
        if version < 23 {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 23 begin: {}", e)))?;

            conn.execute(
                "UPDATE memories SET content = source_text
                 WHERE source = 'memory' AND source_text IS NOT NULL",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("restore prose content: {}", e)))?;

            // Mark affected memories for re-embedding since content changed
            conn.execute(
                "UPDATE memories SET enrichment_status = 'reembed_pending'
                 WHERE source = 'memory' AND source_text IS NOT NULL",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("mark reembed: {}", e)))?;

            conn.execute("PRAGMA user_version = 23", ())
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 23 commit: {}", e)))?;

            log::info!("[memory_db] migration 23: restored prose content, marked for re-embedding");
        }

        // Re-read version for migration 24 — scoped to release the lock
        // before the crash recovery pass (which also needs the lock).
        let version: i64 = {
            let conn = self.conn.lock().await;
            let mut rows = conn.query("PRAGMA user_version", ()).await.map_err(|e| {
                OriginError::VectorDb(format!("read version for migration 24: {}", e))
            })?;
            let v = if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                row.get::<i64>(0).unwrap_or(0)
            } else {
                0
            };
            v
        };

        if version < 24 {
            // Migration 24: Rename chunks→memories + upgrade embeddings to 768-dim
            //
            // Check if old chunks table exists (existing DB) vs fresh DB (memories already created by SCHEMA)
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 check chunks: {}", e)))?;
            let has_chunks = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
                .is_some();
            drop(rows);

            if has_chunks {
                // Phase 1: Create new tables, copy non-vector data from old chunks table
                log::info!(
                    "[memory_db] migration 24: starting chunks→memories rename + embedding upgrade"
                );

                conn.execute("PRAGMA foreign_keys = OFF", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 fk off: {}", e)))?;

                conn.execute("BEGIN", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 begin: {}", e)))?;

                // Create memories table with 768-dim embedding
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT,
                        url TEXT,
                        chunk_index INTEGER NOT NULL,
                        last_modified INTEGER NOT NULL,
                        chunk_type TEXT NOT NULL,
                        language TEXT,
                        byte_start INTEGER,
                        byte_end INTEGER,
                        semantic_unit TEXT,
                        memory_type TEXT,
                        domain TEXT,
                        source_agent TEXT,
                        confidence REAL,
                        confirmed INTEGER,
                        supersedes TEXT,
                        pinned INTEGER NOT NULL DEFAULT 0,
                        pending_revision INTEGER DEFAULT 0,
                        word_count INTEGER NOT NULL DEFAULT 0,
                        entity_id TEXT,
                        enrichment_status TEXT NOT NULL DEFAULT 'enriched',
                        quality TEXT CHECK(quality IN ('low', 'medium', 'high')),
                        is_recap INTEGER NOT NULL DEFAULT 0,
                        supersede_mode TEXT NOT NULL DEFAULT 'hide',
                        structured_fields TEXT,
                        retrieval_cue TEXT,
                        source_text TEXT,
                        created_at INTEGER,
                        stability TEXT NOT NULL DEFAULT 'new',
                        access_count INTEGER NOT NULL DEFAULT 0,
                        last_accessed INTEGER,
                        refinement_status TEXT,
                        effective_confidence REAL,
                        embedding F32_BLOB(768)
                    )",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 create memories: {}", e)))?;

                // Copy all data from chunks (embedding = NULL, will re-embed in Phase 2)
                conn.execute(
                    "INSERT OR IGNORE INTO memories (
                        id, content, source, source_id, title, summary, url,
                        chunk_index, last_modified, chunk_type, language,
                        byte_start, byte_end, semantic_unit, memory_type, domain,
                        source_agent, confidence, confirmed, supersedes, pinned,
                        pending_revision, word_count, entity_id, enrichment_status,
                        quality, is_recap, supersede_mode, structured_fields,
                        retrieval_cue, source_text, created_at, stability,
                        access_count, last_accessed, refinement_status, effective_confidence
                    )
                    SELECT
                        id, content, source, source_id, title, summary, url,
                        chunk_index, last_modified, chunk_type, language,
                        byte_start, byte_end, semantic_unit, memory_type, domain,
                        source_agent, confidence, confirmed, supersedes, pinned,
                        pending_revision, word_count, entity_id, enrichment_status,
                        quality, is_recap, supersede_mode, structured_fields,
                        retrieval_cue, source_text, created_at, stability,
                        access_count, last_accessed, refinement_status, effective_confidence
                    FROM chunks",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 copy chunks: {}", e)))?;

                // Create entities_new with 768-dim embedding
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS entities_new (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        domain TEXT,
                        source_agent TEXT,
                        confidence REAL,
                        confirmed INTEGER DEFAULT 0,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        embedding F32_BLOB(768)
                    )",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 create entities_new: {}", e)))?;

                // Copy entity data
                conn.execute(
                    "INSERT OR IGNORE INTO entities_new (
                        id, name, entity_type, domain, source_agent,
                        confidence, confirmed, created_at, updated_at
                    )
                    SELECT
                        id, name, entity_type, domain, source_agent,
                        confidence, confirmed, created_at, updated_at
                    FROM entities",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 copy entities: {}", e)))?;

                // Drop old tables and swap
                conn.execute("DROP TABLE IF EXISTS chunks_fts", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 drop chunks_fts: {}", e)))?;
                conn.execute("DROP TABLE IF EXISTS chunks", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 drop chunks: {}", e)))?;

                // observations and relations reference entities(id) — drop and recreate references
                conn.execute("DROP TABLE IF EXISTS entities", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 drop entities: {}", e)))?;
                conn.execute("ALTER TABLE entities_new RENAME TO entities", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 rename entities: {}", e)))?;

                // Recreate indexes on memories
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_source_id ON memories(source_id)",
                    (),
                )
                .await
                .ok();
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source)",
                    (),
                )
                .await
                .ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_modified ON memories(last_modified)", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type) WHERE memory_type IS NOT NULL", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes) WHERE supersedes IS NOT NULL", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_entity_id ON memories(entity_id) WHERE entity_id IS NOT NULL", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_enrichment_status ON memories(enrichment_status) WHERE enrichment_status != 'enriched'", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_is_recap ON memories(is_recap) WHERE is_recap = 1", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain) WHERE domain IS NOT NULL", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_pending_revision ON memories(pending_revision) WHERE pending_revision != 0", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_stability ON memories(stability) WHERE source = 'memory'", ()).await.ok();
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain) WHERE domain IS NOT NULL", ()).await.ok();
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations(entity_id)",
                    (),
                )
                .await
                .ok();
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity)",
                    (),
                )
                .await
                .ok();
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity)",
                    (),
                )
                .await
                .ok();

                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 commit: {}", e)))?;

                conn.execute("PRAGMA foreign_keys = ON", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 fk on: {}", e)))?;

                log::info!(
                    "[memory_db] migration 24 phase 1 complete: tables renamed, data copied"
                );
            } else {
                log::info!("[memory_db] migration 24: fresh DB, memories table already exists — skipping table rename");
            }

            conn.execute("PRAGMA user_version = 24", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("m24 set version: {}", e)))?;

            // Phase 2: Re-embed all memories with new model (outside transaction)
            drop(conn); // release lock for re-embedding

            let total_memories = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("SELECT COUNT(*) FROM memories WHERE embedding IS NULL", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 count: {}", e)))?;
                let count: i64 = if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                };
                count as usize
            };

            if total_memories > 0 {
                log::info!(
                    "[memory_db] migration 24 phase 2: re-embedding {} memories",
                    total_memories
                );
                let mut embedded = 0usize;
                let batch_size = 64;

                loop {
                    // Fetch a batch of un-embedded memories
                    let batch: Vec<(String, String, Option<String>)> = {
                        let conn = self.conn.lock().await;
                        let mut rows = conn.query(
                            "SELECT id, COALESCE(source_text, content), domain FROM memories WHERE embedding IS NULL LIMIT ?1",
                            libsql::params![batch_size as i64],
                        ).await.map_err(|e| OriginError::VectorDb(format!("m24 batch: {}", e)))?;
                        let mut batch = Vec::new();
                        while let Some(row) = rows
                            .next()
                            .await
                            .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        {
                            let id: String = row.get(0).unwrap_or_default();
                            let text: String = row.get(1).unwrap_or_default();
                            let domain: Option<String> = row.get(2).unwrap_or(None);
                            batch.push((id, text, domain));
                        }
                        batch
                    };

                    if batch.is_empty() {
                        break;
                    }

                    // Embed the batch (prepend domain prefix for better contextual embeddings)
                    let texts: Vec<String> = batch
                        .iter()
                        .map(|(_, t, d)| {
                            if let Some(ref domain) = d {
                                format!("[{}] {}", domain, t)
                            } else {
                                t.clone()
                            }
                        })
                        .collect();
                    let embeddings = self.generate_embeddings(&texts)?;

                    // Update each row
                    let conn = self.conn.lock().await;
                    if let Err(e) = conn.execute("BEGIN", ()).await {
                        log::warn!("[memory_db] m24 re-embed batch begin: {}", e);
                    }
                    for (idx, (id, _, _)) in batch.iter().enumerate() {
                        if idx < embeddings.len() {
                            if let Err(e) = conn
                                .execute(
                                    "UPDATE memories SET embedding = vector32(?1) WHERE id = ?2",
                                    libsql::params![Self::vec_to_sql(&embeddings[idx]), id.clone()],
                                )
                                .await
                            {
                                log::warn!("[memory_db] m24 re-embed update id={}: {}", id, e);
                            }
                        }
                    }
                    if let Err(e) = conn.execute("COMMIT", ()).await {
                        log::warn!("[memory_db] m24 re-embed batch commit: {}", e);
                    }
                    drop(conn);

                    embedded += batch.len();
                    if let Ok(payload) = serde_json::to_string(&MigrationProgress {
                        current: embedded,
                        total: total_memories,
                        phase: "Upgrading memory quality...".into(),
                    }) {
                        let _ = emitter.emit("migration-progress", &payload);
                    }
                    log::info!(
                        "[memory_db] migration 24: re-embedded {}/{} memories",
                        embedded,
                        total_memories
                    );
                }
            }

            // Re-embed entities
            let total_entities = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("SELECT COUNT(*) FROM entities WHERE embedding IS NULL", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m24 entity count: {}", e)))?;
                let count: i64 = if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                };
                count as usize
            };

            if total_entities > 0 {
                log::info!(
                    "[memory_db] migration 24: re-embedding {} entities",
                    total_entities
                );
                let entity_batch_size = 64;
                let mut entity_embedded = 0usize;

                loop {
                    let batch: Vec<(String, String)> = {
                        let conn = self.conn.lock().await;
                        let mut rows = conn
                            .query(
                                "SELECT id, name FROM entities WHERE embedding IS NULL LIMIT ?1",
                                libsql::params![entity_batch_size as i64],
                            )
                            .await
                            .map_err(|e| {
                                OriginError::VectorDb(format!("m24 entity batch: {}", e))
                            })?;
                        let mut batch = Vec::new();
                        while let Some(row) = rows
                            .next()
                            .await
                            .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        {
                            let id: String = row.get(0).unwrap_or_default();
                            let name: String = row.get(1).unwrap_or_default();
                            batch.push((id, name));
                        }
                        batch
                    };

                    if batch.is_empty() {
                        break;
                    }

                    let texts: Vec<String> = batch.iter().map(|(_, n)| n.clone()).collect();
                    let embeddings = self.generate_embeddings(&texts)?;

                    let conn = self.conn.lock().await;
                    if let Err(e) = conn.execute("BEGIN", ()).await {
                        log::warn!("[memory_db] m24 entity re-embed begin: {}", e);
                    }
                    for (idx, (id, _)) in batch.iter().enumerate() {
                        if idx < embeddings.len() {
                            if let Err(e) = conn
                                .execute(
                                    "UPDATE entities SET embedding = vector32(?1) WHERE id = ?2",
                                    libsql::params![Self::vec_to_sql(&embeddings[idx]), id.clone()],
                                )
                                .await
                            {
                                log::warn!("[memory_db] m24 entity re-embed id={}: {}", id, e);
                            }
                        }
                    }
                    if let Err(e) = conn.execute("COMMIT", ()).await {
                        log::warn!("[memory_db] m24 entity re-embed commit: {}", e);
                    }
                    drop(conn);

                    entity_embedded += batch.len();
                    log::info!(
                        "[memory_db] migration 24: re-embedded {}/{} entities",
                        entity_embedded,
                        total_entities
                    );
                }
            }

            // Phase 3: Recreate FTS and vector indexes
            {
                let conn = self.conn.lock().await;

                // Drop existing FTS table and triggers to avoid duplicates
                // (the insert trigger fires during Phase 1 INSERT, so FTS already has entries)
                conn.execute("DROP TRIGGER IF EXISTS memories_fts_insert", ())
                    .await
                    .ok();
                conn.execute("DROP TRIGGER IF EXISTS memories_fts_delete", ())
                    .await
                    .ok();
                conn.execute("DROP TRIGGER IF EXISTS memories_fts_update", ())
                    .await
                    .ok();
                conn.execute("DROP TABLE IF EXISTS memories_fts", ())
                    .await
                    .ok();

                // FTS5
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, title, content=memories, content_rowid=rowid)",
                    ()
                ).await.ok();

                // Populate FTS from existing data
                conn.execute(
                    "INSERT INTO memories_fts(rowid, content, title) SELECT rowid, content, title FROM memories",
                    ()
                ).await.ok();

                // FTS triggers
                conn.execute("CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid, content, title) VALUES (new.rowid, new.content, new.title); END", ()).await.ok();
                conn.execute("CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, content, title) VALUES('delete', old.rowid, old.content, old.title); END", ()).await.ok();
                conn.execute("CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content, title ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, content, title) VALUES('delete', old.rowid, old.content, old.title); INSERT INTO memories_fts(rowid, content, title) VALUES (new.rowid, new.content, new.title); END", ()).await.ok();

                // DiskANN vector indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories (libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32'))",
                    ()
                ).await.ok();

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities (libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32'))",
                    ()
                ).await.ok();

                // Unpin trigger
                conn.execute(
                    "CREATE TRIGGER IF NOT EXISTS unpin_on_unconfirm AFTER UPDATE OF confirmed ON memories WHEN NEW.confirmed = 0 BEGIN UPDATE memories SET pinned = 0 WHERE source_id = NEW.source_id; END",
                    ()
                ).await.ok();
            }

            log::info!("[memory_db] migration 24 complete: memories table active, {} memories + {} entities re-embedded with GTE-Base 768d", total_memories, total_entities);
        }

        // Migration 25: Embedding model swap GTE-Base-Q → BGE-Base-Q (same 768d).
        // NULL all embeddings so the crash recovery pass re-embeds with the new model.
        {
            let version: i64 = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("PRAGMA user_version", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("read version for m25: {e}")))?;
                if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                }
            };

            if version == 24 {
                log::info!(
                    "[memory_db] migration 25: switching embeddings GTE-Base-Q → BGE-Base-Q"
                );
                let conn = self.conn.lock().await;
                conn.execute(
                    "UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m25 null memories: {e}")))?;
                conn.execute(
                    "UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m25 null entities: {e}")))?;
                conn.execute("PRAGMA user_version = 25", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m25 set version: {e}")))?;
                drop(conn);
                log::info!("[memory_db] migration 25: embeddings cleared, crash recovery will re-embed with BGE-Base-Q");
            }
        }

        // Crash recovery: if app crashed during Phase 2 re-embedding, some memories
        // may have NULL embeddings permanently (user_version is already 24 so migration
        // won't re-run). This unconditional pass is a no-op if all embeddings exist.
        {
            let null_count = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("SELECT COUNT(*) FROM memories WHERE embedding IS NULL", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("null embed check: {}", e)))?;
                let count: i64 = if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                };
                count as usize
            };

            if null_count > 0 {
                log::warn!(
                    "[memory_db] found {} memories with NULL embeddings, re-embedding...",
                    null_count
                );
                let batch_size = 64;
                let mut recovered = 0usize;

                loop {
                    let batch: Vec<(String, String, Option<String>)> = {
                        let conn = self.conn.lock().await;
                        let mut rows = conn.query(
                            "SELECT id, COALESCE(source_text, content), domain FROM memories WHERE embedding IS NULL LIMIT ?1",
                            libsql::params![batch_size as i64],
                        ).await.map_err(|e| OriginError::VectorDb(format!("null embed batch: {}", e)))?;
                        let mut batch = Vec::new();
                        while let Some(row) = rows
                            .next()
                            .await
                            .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        {
                            let id: String = row.get(0).unwrap_or_default();
                            let text: String = row.get(1).unwrap_or_default();
                            let domain: Option<String> = row.get(2).unwrap_or(None);
                            batch.push((id, text, domain));
                        }
                        batch
                    };

                    if batch.is_empty() {
                        break;
                    }

                    let texts: Vec<String> = batch
                        .iter()
                        .map(|(_, t, d)| {
                            if let Some(ref domain) = d {
                                format!("[{}] {}", domain, t)
                            } else {
                                t.clone()
                            }
                        })
                        .collect();
                    let embeddings = self.generate_embeddings(&texts)?;

                    let conn = self.conn.lock().await;
                    if let Err(e) = conn.execute("BEGIN", ()).await {
                        log::warn!("[memory_db] null embed recovery begin: {}", e);
                    }
                    for (idx, (id, _, _)) in batch.iter().enumerate() {
                        if idx < embeddings.len() {
                            if let Err(e) = conn
                                .execute(
                                    "UPDATE memories SET embedding = vector32(?1) WHERE id = ?2",
                                    libsql::params![Self::vec_to_sql(&embeddings[idx]), id.clone()],
                                )
                                .await
                            {
                                log::warn!(
                                    "[memory_db] null embed recovery update id={}: {}",
                                    id,
                                    e
                                );
                            }
                        }
                    }
                    if let Err(e) = conn.execute("COMMIT", ()).await {
                        log::warn!("[memory_db] null embed recovery commit: {}", e);
                    }
                    drop(conn);

                    recovered += batch.len();
                    if let Ok(payload) = serde_json::to_string(&MigrationProgress {
                        current: recovered,
                        total: null_count,
                        phase: "Recovering embeddings...".into(),
                    }) {
                        let _ = emitter.emit("migration-progress", &payload);
                    }
                }
                log::info!("[memory_db] null embedding recovery complete");
            }

            // Entity crash recovery: same pattern for entities with NULL embeddings
            let entity_null_count = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("SELECT COUNT(*) FROM entities WHERE embedding IS NULL", ())
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("null entity embed check: {}", e))
                    })?;
                let count: i64 = if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                };
                count as usize
            };

            if entity_null_count > 0 {
                log::warn!(
                    "[memory_db] found {} entities with NULL embeddings, re-embedding...",
                    entity_null_count
                );
                let batch_size = 64;
                let mut recovered = 0usize;

                loop {
                    let batch: Vec<(String, String)> = {
                        let conn = self.conn.lock().await;
                        let mut rows = conn
                            .query(
                                "SELECT id, name FROM entities WHERE embedding IS NULL LIMIT ?1",
                                libsql::params![batch_size as i64],
                            )
                            .await
                            .map_err(|e| {
                                OriginError::VectorDb(format!("null entity embed batch: {}", e))
                            })?;
                        let mut batch = Vec::new();
                        while let Some(row) = rows
                            .next()
                            .await
                            .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        {
                            let id: String = row.get(0).unwrap_or_default();
                            let name: String = row.get(1).unwrap_or_default();
                            batch.push((id, name));
                        }
                        batch
                    };

                    if batch.is_empty() {
                        break;
                    }

                    let texts: Vec<String> = batch.iter().map(|(_, n)| n.clone()).collect();
                    let embeddings = self.generate_embeddings(&texts)?;

                    let conn = self.conn.lock().await;
                    if let Err(e) = conn.execute("BEGIN", ()).await {
                        log::warn!("[memory_db] null entity embed recovery begin: {}", e);
                    }
                    for (idx, (id, _)) in batch.iter().enumerate() {
                        if idx < embeddings.len() {
                            if let Err(e) = conn
                                .execute(
                                    "UPDATE entities SET embedding = vector32(?1) WHERE id = ?2",
                                    libsql::params![Self::vec_to_sql(&embeddings[idx]), id.clone()],
                                )
                                .await
                            {
                                log::warn!(
                                    "[memory_db] null entity embed recovery id={}: {}",
                                    id,
                                    e
                                );
                            }
                        }
                    }
                    if let Err(e) = conn.execute("COMMIT", ()).await {
                        log::warn!("[memory_db] null entity embed recovery commit: {}", e);
                    }
                    drop(conn);

                    recovered += batch.len();
                    log::info!(
                        "[memory_db] entity embedding recovery: {}/{}",
                        recovered,
                        entity_null_count
                    );
                }
                log::info!(
                    "[memory_db] null entity embedding recovery complete: {} entities",
                    recovered
                );
            }
        }

        let _ = emitter.emit("migration-complete", "{}");

        // Migration 26: Concepts table
        {
            let version: i64 = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("PRAGMA user_version", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("read version for m26: {e}")))?;
                if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                }
            };

            if version < 26 {
                let conn = self.conn.lock().await;
                conn.execute_batch("
                    CREATE TABLE IF NOT EXISTS concepts (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        summary TEXT,
                        content TEXT NOT NULL,
                        entity_id TEXT,
                        domain TEXT,
                        source_memory_ids TEXT NOT NULL DEFAULT '[]',
                        version INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'active',
                        embedding F32_BLOB(768),
                        created_at TEXT NOT NULL,
                        last_compiled TEXT NOT NULL,
                        last_modified TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_concepts_entity_id ON concepts(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_concepts_domain ON concepts(domain);
                    CREATE INDEX IF NOT EXISTS idx_concepts_status ON concepts(status);
                    CREATE INDEX IF NOT EXISTS idx_concepts_embedding ON concepts (libsql_vector_idx(embedding));

                    CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
                        title, summary, content, content='concepts', content_rowid='rowid'
                    );

                    CREATE TRIGGER IF NOT EXISTS concepts_fts_insert AFTER INSERT ON concepts BEGIN
                        INSERT INTO concepts_fts(rowid, title, summary, content)
                        VALUES (NEW.rowid, NEW.title, NEW.summary, NEW.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS concepts_fts_delete AFTER DELETE ON concepts BEGIN
                        INSERT INTO concepts_fts(concepts_fts, rowid, title, summary, content)
                        VALUES ('delete', OLD.rowid, OLD.title, OLD.summary, OLD.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS concepts_fts_update AFTER UPDATE OF title, summary, content ON concepts BEGIN
                        INSERT INTO concepts_fts(concepts_fts, rowid, title, summary, content)
                        VALUES ('delete', OLD.rowid, OLD.title, OLD.summary, OLD.content);
                        INSERT INTO concepts_fts(rowid, title, summary, content)
                        VALUES (NEW.rowid, NEW.title, NEW.summary, NEW.content);
                    END;
                ").await.map_err(|e| OriginError::VectorDb(format!("migration 26: {}", e)))?;

                conn.execute("PRAGMA user_version = 26", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=26: {}", e)))?;

                log::info!("[memory_db] migration 26: concepts table + FTS + indexes");
            }
        }

        // Migration 27: Add community_id column to entities for graph community detection
        {
            let version: i64 = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("PRAGMA user_version", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("read version for m27: {e}")))?;
                if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                }
            };

            if version < 27 {
                let conn = self.conn.lock().await;
                conn.execute_batch(
                    "
                    ALTER TABLE entities ADD COLUMN community_id INTEGER;
                    CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(community_id);
                ",
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 27: {}", e)))?;

                conn.execute("PRAGMA user_version = 27", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=27: {}", e)))?;

                log::info!("[memory_db] migration 27: community_id column on entities");
            }

            // Migration 28: Source sync state tracking
            if version < 28 {
                let conn = self.conn.lock().await;
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS source_sync_state (
                        source_id TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        mtime_ns INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        last_synced_at INTEGER NOT NULL,
                        PRIMARY KEY (source_id, file_path)
                    );
                    CREATE INDEX IF NOT EXISTS idx_sync_state_source ON source_sync_state(source_id);"
                ).await.map_err(|e| OriginError::VectorDb(format!("migration 28 source_sync_state: {}", e)))?;
                conn.execute("PRAGMA user_version = 28", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=28: {}", e)))?;
                log::info!("[memory_db] migration 28: created source_sync_state table");
            }

            // Migration 29a: import_state table (from PR #61, merged into main).
            // For fresh DBs: creates the table at version 29. For user's DB
            // that already ran our branch (version >= 29): SKIPS, but migration
            // 34 below ensures the table exists via CREATE TABLE IF NOT EXISTS.
            if version < 29 {
                let conn = self.conn.lock().await;
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS import_state (
                        id                       TEXT PRIMARY KEY,
                        vendor                   TEXT NOT NULL,
                        source_path              TEXT NOT NULL,
                        total_conversations      INTEGER,
                        processed_conversations  INTEGER NOT NULL DEFAULT 0,
                        stage                    TEXT NOT NULL,
                        error_message            TEXT,
                        started_at               TEXT NOT NULL,
                        updated_at               TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_import_state_stage ON import_state(stage);",
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 29a import_state: {}", e)))?;
                // Don't bump user_version here — migration 29b below handles it.
            }

            // Migration 29b: case-normalize `agent_activity.agent_name` so
            // `Claude Code` and `claude-code` collapse into one filter entry.
            // Paired with case normalization on the write path in
            // `log_agent_activity`.
            if version < 29 {
                let conn = self.conn.lock().await;
                conn.execute(
                    "UPDATE agent_activity SET agent_name = LOWER(TRIM(agent_name))",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration 29 normalize agent_name: {}", e))
                })?;
                conn.execute("PRAGMA user_version = 29", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=29: {}", e)))?;
                log::info!("[memory_db] migration 29: case-normalized agent_activity.agent_name");
            }

            // Migration 30: add `display_name` column to agent_connections +
            // canonicalize existing `name` values. Pre-existing rows written by
            // the SetupWizard used human labels like `"Claude Code"` as the
            // technical ID — this migration splits that into:
            //   name          = canonical technical ID ("claude-code")
            //   display_name  = original human label ("Claude Code")
            // On collision (both `Claude Code` and `claude-code` exist), we
            // keep the freshest row (highest `last_seen_at`) and drop the others.
            // After rewriting, backfill `display_name` from `KNOWN_CLIENTS`
            // for any row that still has a null display_name.
            if version < 30 {
                // Check if display_name column exists; some earlier DBs may
                // already have it from SCHEMA, others need ALTER.
                let (needs_alter, rows_to_process): (bool, Vec<(String, String, Option<i64>)>) = {
                    let conn = self.conn.lock().await;

                    let mut has_col = conn
                        .query(
                            "SELECT COUNT(*) FROM pragma_table_info('agent_connections') WHERE name = 'display_name'",
                            (),
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m30 pragma: {e}")))?;
                    let has_display_name: i64 = has_col
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        .and_then(|r| r.get(0).ok())
                        .unwrap_or(0);
                    drop(has_col);

                    if has_display_name == 0 {
                        conn.execute(
                            "ALTER TABLE agent_connections ADD COLUMN display_name TEXT",
                            (),
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m30 add display_name: {e}")))?;
                    }

                    // Snapshot rows we need to examine.
                    let mut rows = conn
                        .query("SELECT id, name, last_seen_at FROM agent_connections", ())
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m30 select: {e}")))?;
                    let mut buf: Vec<(String, String, Option<i64>)> = Vec::new();
                    while let Some(row) = rows
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                    {
                        buf.push((
                            row.get::<String>(0).unwrap_or_default(),
                            row.get::<String>(1).unwrap_or_default(),
                            row.get::<Option<i64>>(2).unwrap_or(None),
                        ));
                    }
                    drop(rows);
                    (has_display_name == 0, buf)
                };
                let _ = needs_alter;

                for (id, name, _last_seen) in rows_to_process {
                    let canonical = canonicalize_agent_id(&name);
                    let conn = self.conn.lock().await;
                    if canonical == name {
                        // Already canonical — just backfill display_name if
                        // we have a known friendly name.
                        if let Some(display) = known_client_display_name(&canonical) {
                            conn.execute(
                                "UPDATE agent_connections SET display_name = COALESCE(display_name, ?1) WHERE id = ?2",
                                libsql::params![display, id],
                            )
                            .await
                            .map_err(|e| {
                                OriginError::VectorDb(format!("m30 backfill display: {e}"))
                            })?;
                        }
                        continue;
                    }

                    // Name needs canonicalization. Check if another row already
                    // owns the canonical form.
                    let mut collision = conn
                        .query(
                            "SELECT id FROM agent_connections WHERE name = ?1 AND id != ?2 LIMIT 1",
                            libsql::params![canonical.clone(), id.clone()],
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m30 collision check: {e}")))?;
                    let collision_id = collision
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                        .and_then(|r| r.get::<String>(0).ok());
                    drop(collision);

                    if let Some(surviving_id) = collision_id {
                        // Another row already owns `canonical`. Preserve the
                        // human label on the surviving row (prefer the human
                        // label as `display_name`), then delete this row.
                        let display = known_client_display_name(&canonical)
                            .map(|s| s.to_string())
                            .unwrap_or(name.clone());
                        conn.execute(
                            "UPDATE agent_connections SET display_name = COALESCE(display_name, ?1) WHERE id = ?2",
                            libsql::params![display, surviving_id],
                        )
                        .await
                        .map_err(|e| {
                            OriginError::VectorDb(format!("m30 merge display: {e}"))
                        })?;
                        conn.execute(
                            "DELETE FROM agent_connections WHERE id = ?1",
                            libsql::params![id],
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m30 drop duplicate: {e}")))?;
                    } else {
                        // No collision — rename to canonical, promote the
                        // original label to display_name.
                        let display = known_client_display_name(&canonical)
                            .map(|s| s.to_string())
                            .unwrap_or(name.clone());
                        conn.execute(
                            "UPDATE agent_connections SET name = ?1, display_name = COALESCE(display_name, ?2) WHERE id = ?3",
                            libsql::params![canonical, display, id],
                        )
                        .await
                        .map_err(|e| {
                            OriginError::VectorDb(format!("m30 canonicalize: {e}"))
                        })?;
                    }
                }

                let conn = self.conn.lock().await;
                conn.execute("PRAGMA user_version = 30", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=30: {}", e)))?;
                log::info!(
                    "[memory_db] migration 30: added display_name + canonicalized agent_connections.name"
                );
            }

            // Migration 31: re-canonicalize `agent_activity.agent_name`.
            //
            // Migration 29 used SQLite's `LOWER(TRIM(x))` which lowercased but
            // left spaces alone — so `"Claude Code"` became `"claude code"`
            // (space, not hyphen) and never matched the `"claude-code"` form
            // the CLI sends. This migration re-runs the full canonicalization
            // in Rust (SQLite has no replace-runs-of-separators), collapsing
            // `"claude code"` + `"claude-code"` into one set.
            if version < 31 {
                // Snapshot distinct agent names.
                let distinct_names: Vec<String> = {
                    let conn = self.conn.lock().await;
                    let mut rows = conn
                        .query("SELECT DISTINCT agent_name FROM agent_activity", ())
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m31 distinct select: {e}")))?;
                    let mut buf = Vec::new();
                    while let Some(row) = rows
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                    {
                        buf.push(row.get::<String>(0).unwrap_or_default());
                    }
                    buf
                };

                // Only rewrite the rows whose current value isn't already canonical.
                let mut rewrites = 0usize;
                for current in distinct_names {
                    let canonical = canonicalize_agent_id(&current);
                    if canonical == current {
                        continue;
                    }
                    let conn = self.conn.lock().await;
                    conn.execute(
                        "UPDATE agent_activity SET agent_name = ?1 WHERE agent_name = ?2",
                        libsql::params![canonical.clone(), current.clone()],
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m31 rewrite {current}: {e}")))?;
                    rewrites += 1;
                }

                let conn = self.conn.lock().await;
                conn.execute("PRAGMA user_version = 31", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=31: {}", e)))?;
                log::info!(
                    "[memory_db] migration 31: re-canonicalized {rewrites} distinct agent_activity.agent_name values"
                );
            }

            // Migration 32: default registered agents to `"full"` trust.
            //
            // The old default (`"review"`) silently gated Tier 1 chat-context
            // (identity, preferences, narrative brief) for every connected
            // client — which meant no user-registered agent ever got those,
            // regardless of whether the user actually trusted it. For a
            // single-user local server, registration IS the trust gesture:
            // if you ran the SetupWizard for this agent, you want it to see
            // Tier 1. Upgrade every existing `"review"` row to `"full"`.
            // Agents you want to downgrade can still be edited via the API.
            if version < 32 {
                let conn = self.conn.lock().await;
                conn.execute(
                    "UPDATE agent_connections SET trust_level = 'full' WHERE trust_level = 'review'",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration 32 upgrade trust: {e}"))
                })?;
                conn.execute("PRAGMA user_version = 32", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=32: {}", e)))?;
                log::info!(
                    "[memory_db] migration 32: upgraded agent_connections.trust_level review → full"
                );
            }

            // Migration 33: map legacy `'untrusted'` trust_level to `'unknown'`.
            //
            // The Settings page trust dropdown changed its option set from
            // `full | review | untrusted` to `full | review | unknown` as part
            // of the vocabulary cleanup. `describeTrustLevel()` falls back
            // correctly for unknown values, but the native `<select>` won't
            // display a value that has no matching `<option>` — it renders
            // the first option as selected while the underlying row keeps its
            // old value. Clicking the dropdown would then silently upgrade
            // the agent from "untrusted" to whatever's picked. Remap now so
            // the UI and DB stay in sync.
            //
            // Idempotent: a re-run finds no `'untrusted'` rows and does nothing.
            if version < 33 {
                let conn = self.conn.lock().await;
                conn.execute(
                    "UPDATE agent_connections SET trust_level = 'unknown' WHERE trust_level = 'untrusted'",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration 33 remap untrusted: {e}"))
                })?;
                conn.execute("PRAGMA user_version = 33", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=33: {}", e)))?;
                log::info!(
                    "[memory_db] migration 33: remapped agent_connections.trust_level untrusted → unknown"
                );
            }

            // Migration 34: ensure import_state table exists for DBs where
            // migration 29 was skipped (the user's DB had version >= 29 from
            // this branch's earlier run, so it never ran main's import_state
            // migration). CREATE TABLE IF NOT EXISTS is idempotent.
            if version < 34 {
                let conn = self.conn.lock().await;
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS import_state (
                        id                       TEXT PRIMARY KEY,
                        vendor                   TEXT NOT NULL,
                        source_path              TEXT NOT NULL,
                        total_conversations      INTEGER,
                        processed_conversations  INTEGER NOT NULL DEFAULT 0,
                        stage                    TEXT NOT NULL,
                        error_message            TEXT,
                        started_at               TEXT NOT NULL,
                        updated_at               TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_import_state_stage ON import_state(stage);",
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 34 import_state: {}", e)))?;
                conn.execute("PRAGMA user_version = 34", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=34: {}", e)))?;
                log::info!("[memory_db] migration 34: ensured import_state table exists");
            }

            // Migration 35: App metadata key-value store (scheduler persistence)
            // Used by the event-driven steep scheduler to persist last_daily_steep_ts
            // across restarts. CREATE TABLE IF NOT EXISTS for safety against migration
            // number collisions with parallel branches.
            if version < 35 {
                let conn = self.conn.lock().await;
                conn.execute_batch(
                    "
                    CREATE TABLE IF NOT EXISTS app_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                ",
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration35 app_metadata: {}", e)))?;

                conn.execute("PRAGMA user_version = 35", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("set user_version=35: {}", e)))?;
                log::info!("[memory_db] migration 35: app_metadata table created");
            }

            if version < 36 {
                let conn = self.conn.lock().await;
                let now = chrono::Utc::now().to_rfc3339();
                conn.execute(
                    "UPDATE concepts SET status = 'archived', last_modified = ?1 \
                     WHERE status = 'active' AND ( \
                        title = 'general' \
                        OR title GLOB '*[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]-*' \
                        OR (LENGTH(title) BETWEEN 7 AND 12 AND title GLOB '*[0-9a-f]*' AND NOT (title GLOB '* *' OR title GLOB '*[g-zG-Z]*')) \
                        OR title LIKE 'const %' \
                        OR title LIKE 'let %' \
                        OR title LIKE 'await %' \
                        OR title LIKE 'function %' \
                        OR title LIKE 'import %' \
                        OR title LIKE 'fn %' \
                        OR title LIKE '[obs/%' \
                        OR title LIKE '[/%' \
                     )",
                    libsql::params![now],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 36: {e}")))?;
                conn.execute("PRAGMA user_version = 36", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 36 bump: {e}")))?;
                log::info!("[migration] Migration 36 applied: archived ugly-title concepts");
            }

            // Migration 37: archive any 'general'-titled concepts that slipped through
            // after migration 36 ran (generated between daemon restarts). Idempotent —
            // LOWER(title) = 'general' covers any capitalisation variant.
            if version < 37 {
                let conn = self.conn.lock().await;
                let now = chrono::Utc::now().to_rfc3339();
                conn.execute(
                    "UPDATE concepts SET status = 'archived', last_modified = ?1 \
                     WHERE status = 'active' AND LOWER(title) IN \
                       ('general', 'untitled', 'topic', 'concept', 'cluster', 'misc', 'other', 'unknown')",
                    libsql::params![now],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 37: {e}")))?;
                conn.execute("PRAGMA user_version = 37", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 37 bump: {e}")))?;
                log::info!("[migration] Migration 37 applied: archived generic-title concepts");
            }

            // Migration 38: archive concepts whose titles are commit-message-format,
            // obsidian path fragments, or absolute filesystem paths that slipped through
            // before the extended pattern filters were added.
            if version < 38 {
                let conn = self.conn.lock().await;
                let now = chrono::Utc::now().to_rfc3339();
                conn.execute(
                    "UPDATE concepts SET status = 'archived', last_modified = ?1 \
                     WHERE status = 'active' AND ( \
                        title LIKE 'feat:%' OR title LIKE 'fix:%' OR title LIKE 'chore:%' \
                        OR title LIKE 'docs:%' OR title LIKE 'refactor:%' OR title LIKE 'test:%' \
                        OR title LIKE 'style:%' OR title LIKE 'perf:%' OR title LIKE 'ci:%' \
                        OR title LIKE 'build:%' OR title LIKE 'revert:%' \
                        OR title LIKE '[obs/%' OR title LIKE '[mem_%' \
                        OR title LIKE '[import_%' OR title LIKE '[file_%' \
                        OR title LIKE '/Users/%' OR title LIKE '/home/%' OR title LIKE '~/%' \
                        OR title LIKE '%/second-brain/%' OR title LIKE '%/inbox/%' \
                     )",
                    libsql::params![now],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 38: {e}")))?;
                conn.execute("PRAGMA user_version = 38", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 38 bump: {e}")))?;
                log::info!("[migration] Migration 38 applied: archived commit-message and path-fragment concepts");
            }

            // Migration 39: archive remaining ugly-titled concepts that survived
            // migrations 36-38, plus dedup concepts with identical titles.
            if version < 39 {
                let conn = self.conn.lock().await;
                let now = chrono::Utc::now().to_rfc3339();
                let now2 = now.clone();

                // Archive concepts with git-diff-stats titles, markdown bold, UUIDs,
                // code snippets, or very thin bodies (<100 chars)
                conn.execute(
                    "UPDATE concepts SET status = 'archived', last_modified = ?1 \
                     WHERE status = 'active' AND ( \
                        title LIKE '% files changed,%' \
                        OR title LIKE '**%' \
                        OR title LIKE 'const %' \
                        OR title LIKE 'let %' \
                        OR title LIKE 'await %' \
                        OR (LENGTH(title) >= 32 AND title GLOB '[0-9a-f]*-[0-9a-f]*-[0-9a-f]*') \
                        OR LENGTH(content) < 100 \
                     )",
                    libsql::params![now],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 39 cleanup: {e}")))?;

                // Dedup: for concepts with identical titles, keep the one with
                // the longest content and archive the rest.
                conn.execute(
                    "UPDATE concepts SET status = 'archived', last_modified = ?1 \
                     WHERE status = 'active' AND id NOT IN ( \
                        SELECT id FROM ( \
                            SELECT id, ROW_NUMBER() OVER ( \
                                PARTITION BY title ORDER BY LENGTH(content) DESC \
                            ) AS rn FROM concepts WHERE status = 'active' \
                        ) WHERE rn = 1 \
                     )",
                    libsql::params![now2],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 39 dedup: {e}")))?;

                conn.execute("PRAGMA user_version = 39", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 39 bump: {e}")))?;
                log::info!("[migration] Migration 39 applied: archived ugly remnants + deduped identical titles");
            }

            // Migration 40: topic-key upsert schema + concept_sources join table.
            // Adds version/changelog columns to memories, staleness columns to concepts,
            // and the concept_sources join table (with backfill from source_memory_ids JSON).
            if version < 40 {
                let conn = self.conn.lock().await;

                // Add version + changelog to memories (topic-key upsert tracking)
                let _ = conn
                    .execute(
                        "ALTER TABLE memories ADD COLUMN version INTEGER DEFAULT 1",
                        (),
                    )
                    .await;
                let _ = conn
                    .execute(
                        "ALTER TABLE memories ADD COLUMN changelog TEXT DEFAULT '[]'",
                        (),
                    )
                    .await;

                // Add staleness-tracking columns to concepts
                let _ = conn
                    .execute(
                        "ALTER TABLE concepts ADD COLUMN sources_updated_count INTEGER DEFAULT 0",
                        (),
                    )
                    .await;
                let _ = conn
                    .execute("ALTER TABLE concepts ADD COLUMN stale_reason TEXT", ())
                    .await;
                let _ = conn
                    .execute(
                        "ALTER TABLE concepts ADD COLUMN user_edited INTEGER DEFAULT 0",
                        (),
                    )
                    .await;

                // Create concept_sources join table (replaces source_memory_ids JSON column)
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS concept_sources (
                        concept_id        TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                        memory_source_id  TEXT NOT NULL,
                        linked_at         INTEGER NOT NULL,
                        link_reason       TEXT,
                        PRIMARY KEY (concept_id, memory_source_id)
                    )",
                    (),
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration 40 create concept_sources: {e}"))
                })?;

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_concept_sources_memory ON concept_sources(memory_source_id)",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 40 idx: {e}")))?;

                // Backfill concept_sources from existing source_memory_ids JSON
                {
                    conn.execute("BEGIN", ()).await.map_err(|e| {
                        OriginError::VectorDb(format!("migration 40 backfill begin: {e}"))
                    })?;

                    let mut rows = conn
                        .query(
                            "SELECT id, source_memory_ids, created_at FROM concepts WHERE source_memory_ids != '[]' AND source_memory_ids IS NOT NULL",
                            (),
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("migration 40 backfill query: {e}")))?;

                    let mut backfill_rows: Vec<(String, String, String)> = Vec::new();
                    while let Some(row) = rows.next().await.map_err(|e| {
                        OriginError::VectorDb(format!("migration 40 backfill next: {e}"))
                    })? {
                        let concept_id: String = row.get(0).map_err(|e| {
                            OriginError::VectorDb(format!("migration 40 backfill id: {e}"))
                        })?;
                        let json_str: String = row.get(1).unwrap_or_else(|_| "[]".to_string());
                        let created_at_str: String = row
                            .get(2)
                            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
                        backfill_rows.push((concept_id, json_str, created_at_str));
                    }

                    for (concept_id, json_str, created_at_str) in backfill_rows {
                        let linked_at = chrono::DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.timestamp())
                            .unwrap_or_else(|_| chrono::Utc::now().timestamp());
                        let source_ids: Vec<String> =
                            serde_json::from_str(&json_str).unwrap_or_default();
                        for sid in &source_ids {
                            let _ = conn
                                .execute(
                                    "INSERT OR IGNORE INTO concept_sources (concept_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, ?3, 'backfill')",
                                    libsql::params![concept_id.clone(), sid.clone(), linked_at],
                                )
                                .await;
                        }
                    }

                    conn.execute("COMMIT", ()).await.map_err(|e| {
                        OriginError::VectorDb(format!("migration 40 backfill commit: {e}"))
                    })?;
                }

                conn.execute("PRAGMA user_version = 40", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 40 bump: {e}")))?;
                log::info!("[migration] Migration 40 applied: topic-key upsert columns + concept_sources join table");
            }

            // Migration 41: KG quality — alias table, relation vocabulary, new columns,
            // entity/relation deduplication, and unique index on relations.
            if version < 41 {
                {
                    let conn = self.conn.lock().await;
                    conn.execute_batch(
                        "
                        -- 1. Entity aliases table
                        CREATE TABLE IF NOT EXISTS entity_aliases (
                            alias_name TEXT NOT NULL,
                            canonical_entity_id TEXT NOT NULL REFERENCES entities(id),
                            created_at INTEGER NOT NULL,
                            source TEXT DEFAULT 'auto'
                        );
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_alias_name ON entity_aliases(alias_name);

                        -- 2. Relation type vocabulary
                        CREATE TABLE IF NOT EXISTS relation_type_vocabulary (
                            canonical TEXT PRIMARY KEY,
                            aliases TEXT,
                            category TEXT,
                            count INTEGER DEFAULT 0
                        );

                        -- Seed vocabulary (18 entries)
                        INSERT OR IGNORE INTO relation_type_vocabulary (canonical, aliases, category, count) VALUES
                            ('works_on', '[\"working_at\",\"works_at\",\"working_on\"]', 'professional', 0),
                            ('leads', '[\"leading\",\"manages\",\"heads\"]', 'professional', 0),
                            ('member_of', '[\"belongs_to\",\"part_of_team\"]', 'professional', 0),
                            ('authored', '[\"wrote\",\"created_doc\"]', 'professional', 0),
                            ('knows', '[\"familiar_with\",\"met\"]', 'personal', 0),
                            ('located_in', '[\"lives_in\",\"based_in\"]', 'personal', 0),
                            ('uses', '[\"utilizes\",\"leverages\",\"using\"]', 'technical', 0),
                            ('depends_on', '[\"requires\",\"needs\"]', 'technical', 0),
                            ('created', '[\"built\",\"made\",\"developed\"]', 'technical', 0),
                            ('part_of', '[\"component_of\",\"subset_of\"]', 'structural', 0),
                            ('prefers', '[\"favors\",\"likes\",\"chooses\"]', 'personal', 0),
                            ('decided', '[\"chose\",\"selected\",\"committed_to\"]', 'personal', 0),
                            ('learned_from', '[\"discovered_via\",\"taught_by\"]', 'personal', 0),
                            ('contradicts', '[\"conflicts_with\",\"opposes\"]', 'structural', 0),
                            ('replaced_by', '[\"superseded_by\",\"deprecated_by\"]', 'structural', 0),
                            ('blocked_by', '[\"waiting_on\",\"stuck_on\"]', 'structural', 0),
                            ('discussed_in', '[\"mentioned_in\",\"referenced_in\"]', 'structural', 0),
                            ('related_to', '[\"associated_with\",\"connected_to\"]', 'structural', 0);

                        -- 3. New columns on relations (idempotent via INSERT trick)
                        -- SQLite lacks IF NOT EXISTS for ALTER TABLE, so we check the schema.
                        ",
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 41 DDL: {e}")))?;

                    // Add columns idempotently -- check if they exist first.
                    for (table, col, col_type) in [
                        ("relations", "confidence", "REAL"),
                        ("relations", "explanation", "TEXT"),
                        ("relations", "source_memory_id", "TEXT"),
                        ("entities", "embedding_updated_at", "INTEGER"),
                    ] {
                        let has_col: bool = {
                            let mut rows = conn
                                .query(
                                    &format!("SELECT COUNT(*) FROM pragma_table_info('{}') WHERE name = ?1", table),
                                    libsql::params![col.to_string()],
                                )
                                .await
                                .map_err(|e| OriginError::VectorDb(format!("migration 41 col check: {e}")))?;
                            match rows.next().await {
                                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0) > 0,
                                _ => false,
                            }
                        };
                        if !has_col {
                            conn.execute(
                                &format!("ALTER TABLE {} ADD COLUMN {} {}", table, col, col_type),
                                (),
                            )
                            .await
                            .map_err(|e| {
                                OriginError::VectorDb(format!(
                                    "migration 41 add {}.{}: {e}",
                                    table, col
                                ))
                            })?;
                        }
                    }
                }
                // conn guard is dropped here so dedup helpers can re-acquire

                // 5. Populate entity_aliases with one self-alias per existing entity,
                //    then deduplicate entities with identical lowercase names.
                self.migrate_41_dedup_entities().await?;

                {
                    let conn = self.conn.lock().await;

                    // 6. Deduplicate relations: for each (from_entity, to_entity,
                    //    relation_type) group keep the row with the smallest rowid and
                    //    delete the rest.
                    conn.execute(
                        "DELETE FROM relations
                         WHERE rowid NOT IN (
                             SELECT MIN(rowid)
                             FROM relations
                             GROUP BY from_entity, to_entity, relation_type
                         )",
                        (),
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration 41 dedup relations: {e}"))
                    })?;

                    // 7. Unique index on relations
                    conn.execute_batch(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_relations_unique
                             ON relations(from_entity, to_entity, relation_type);",
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("migration 41 relations unique idx: {e}"))
                    })?;

                    conn.execute("PRAGMA user_version = 41", ())
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("migration 41 bump: {e}")))?;
                    log::info!("[migration] Migration 41 applied: alias table, relation vocabulary, dedup, unique index");
                }
            }

            // Migration 42: Recreate concepts vector index with tuning params +
            // backfill NULL concept embeddings.
            if version < 42 {
                {
                    let conn = self.conn.lock().await;

                    // Drop the untuned index (created in migration 26 with no params)
                    conn.execute("DROP INDEX IF EXISTS idx_concepts_embedding", ())
                        .await
                        .ok(); // tolerate if already dropped

                    // Recreate with cosine metric + float8 compression (matches memories_vec_idx)
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_concepts_embedding ON concepts (\
                         libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32'))",
                        (),
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("migration 42 concepts idx: {e}")))?;

                    conn.execute("PRAGMA user_version = 42", ())
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("migration 42 bump: {e}")))?;
                }

                // Backfill embeddings for concepts that have NULL embedding
                let backfilled = self.backfill_concept_embeddings().await.unwrap_or(0);
                log::info!(
                    "[migration] Migration 42 applied: concepts vector index with cosine+float8, backfilled {} embeddings",
                    backfilled
                );
            }
        }

        // Migration 43: enrichment_steps table + needs_reembed column + summary view
        {
            let version: i64 = {
                let conn = self.conn.lock().await;
                let mut rows = conn
                    .query("PRAGMA user_version", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("read version for m43: {e}")))?;
                if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    row.get(0).unwrap_or(0)
                } else {
                    0
                }
            };

            if version < 43 {
                let conn = self.conn.lock().await;

                // Create enrichment_steps table if not exists (idempotent)
                let table_exists: bool = {
                    let mut rows = conn
                        .query(
                            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='enrichment_steps'",
                            (),
                        )
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m43 table check: {e}")))?;
                    if let Some(row) = rows
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                    {
                        let count: i64 = row.get(0).unwrap_or(0);
                        count > 0
                    } else {
                        false
                    }
                };

                conn.execute("BEGIN", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 begin: {e}")))?;

                if !table_exists {
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS enrichment_steps (
                            source_id TEXT NOT NULL,
                            step_name TEXT NOT NULL,
                            status TEXT NOT NULL,
                            error TEXT,
                            attempts INTEGER NOT NULL DEFAULT 1,
                            updated_at INTEGER NOT NULL,
                            PRIMARY KEY (source_id, step_name)
                        )",
                        (),
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 create table: {e}")))?;

                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_enrichment_steps_failed ON enrichment_steps(status) WHERE status = 'failed'",
                        (),
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 create index: {e}")))?;
                }

                // Add needs_reembed column if missing
                let col_exists: bool = {
                    let mut rows = conn
                        .query("PRAGMA table_info(memories)", ())
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("m43 pragma: {e}")))?;
                    let mut found = false;
                    while let Ok(Some(row)) = rows.next().await {
                        let col_name: String = row.get(1).unwrap_or_default();
                        if col_name == "needs_reembed" {
                            found = true;
                            break;
                        }
                    }
                    found
                };

                if !col_exists {
                    conn.execute(
                        "ALTER TABLE memories ADD COLUMN needs_reembed INTEGER NOT NULL DEFAULT 0",
                        (),
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 add column: {e}")))?;

                    // Backfill: mark existing reembed_pending memories
                    conn.execute(
                        "UPDATE memories SET needs_reembed = 1 WHERE enrichment_status = 'reembed_pending'",
                        (),
                    )
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 backfill: {e}")))?;
                }

                // Create summary view
                conn.execute(
                    "CREATE VIEW IF NOT EXISTS memory_enrichment_summary AS
                     SELECT
                         m.source_id,
                         CASE
                             WHEN COUNT(s.step_name) = 0 THEN 'raw'
                             WHEN SUM(CASE WHEN s.status IN ('failed','abandoned') THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                             WHEN SUM(CASE WHEN s.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                             ELSE 'enrichment_partial'
                         END AS summary
                     FROM (SELECT DISTINCT source_id FROM memories) m
                     LEFT JOIN enrichment_steps s ON s.source_id = m.source_id
                     GROUP BY m.source_id",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("m43 create view: {e}")))?;

                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 commit: {e}")))?;

                conn.execute("PRAGMA user_version = 43", ())
                    .await
                    .map_err(|e| OriginError::VectorDb(format!("m43 bump: {e}")))?;

                log::info!("[migration] Migration 43 applied: enrichment_steps table, needs_reembed column, memory_enrichment_summary view");
            }
        }

        Ok(())
    }

    /// Migration 41 helper: seed one self-alias per entity, then collapse
    /// entities that share the same lowercase name by keeping the
    /// most-observed one and redirecting the others as aliases.
    async fn migrate_41_dedup_entities(&self) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;

        // Seed self-aliases for all existing entities (idempotent via INSERT OR IGNORE)
        conn.execute_batch(
            "INSERT OR IGNORE INTO entity_aliases (alias_name, canonical_entity_id, created_at, source)
             SELECT LOWER(name), id, created_at, 'migration' FROM entities;",
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("migration 41 seed aliases: {e}")))?;

        // Find all groups of entities that share the same lowercase name
        // and have more than one member.
        let mut dup_rows = conn
            .query(
                "SELECT LOWER(name) AS lname
                 FROM entities
                 GROUP BY LOWER(name)
                 HAVING COUNT(*) > 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration 41 find dups: {e}")))?;

        let mut dup_names: Vec<String> = Vec::new();
        while let Some(row) = dup_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("migration 41 dup row: {e}")))?
        {
            dup_names.push(row.get(0).unwrap_or_default());
        }
        drop(dup_rows);

        // For each duplicated lowercase name, keep the entity with the most
        // observations (tie-break: most recent updated_at), then:
        //   - redirect all aliases from losers to the winner
        //   - redirect all relations from losers to the winner
        //   - delete losers
        for lname in dup_names {
            // Find winner (most observations, then latest update)
            let mut winner_rows = conn
                .query(
                    "SELECT e.id
                     FROM entities e
                     LEFT JOIN observations o ON o.entity_id = e.id
                     WHERE LOWER(e.name) = ?1
                     GROUP BY e.id
                     ORDER BY COUNT(o.id) DESC, e.updated_at DESC
                     LIMIT 1",
                    libsql::params![lname.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 winner: {e}")))?;
            let winner_id: String = match winner_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 winner row: {e}")))?
            {
                Some(row) => row.get(0).unwrap_or_default(),
                None => continue,
            };
            drop(winner_rows);

            // Collect loser ids
            let mut loser_rows = conn
                .query(
                    "SELECT id FROM entities WHERE LOWER(name) = ?1 AND id != ?2",
                    libsql::params![lname.clone(), winner_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 losers: {e}")))?;
            let mut loser_ids: Vec<String> = Vec::new();
            while let Some(row) = loser_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 loser row: {e}")))?
            {
                loser_ids.push(row.get(0).unwrap_or_default());
            }
            drop(loser_rows);

            for loser_id in &loser_ids {
                // Redirect aliases from loser to winner
                conn.execute(
                    "UPDATE OR IGNORE entity_aliases SET canonical_entity_id = ?1 WHERE canonical_entity_id = ?2",
                    libsql::params![winner_id.clone(), loser_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 redir alias: {e}")))?;

                // Redirect observations from loser to winner
                conn.execute(
                    "UPDATE observations SET entity_id = ?1 WHERE entity_id = ?2",
                    libsql::params![winner_id.clone(), loser_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 redir obs: {e}")))?;

                // Redirect relations: from_entity references
                conn.execute(
                    "UPDATE OR IGNORE relations SET from_entity = ?1 WHERE from_entity = ?2",
                    libsql::params![winner_id.clone(), loser_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 redir rel from: {e}")))?;

                // Redirect relations: to_entity references
                conn.execute(
                    "UPDATE OR IGNORE relations SET to_entity = ?1 WHERE to_entity = ?2",
                    libsql::params![winner_id.clone(), loser_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 redir rel to: {e}")))?;

                // Clean up any remaining aliases pointing to loser (no CASCADE on FK)
                conn.execute(
                    "DELETE FROM entity_aliases WHERE canonical_entity_id = ?1",
                    libsql::params![loser_id.clone()],
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("migration 41 del loser aliases: {e}"))
                })?;

                // Delete the loser entity
                conn.execute(
                    "DELETE FROM entities WHERE id = ?1",
                    libsql::params![loser_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("migration 41 del loser: {e}")))?;
            }
        }

        Ok(())
    }

    // ===== Space CRUD Methods =====

    pub async fn list_spaces(&self) -> Result<Vec<Space>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT s.id, s.name, s.description, s.suggested, s.created_at, s.updated_at,
                        (SELECT COUNT(DISTINCT c.source_id) FROM memories c WHERE c.domain = s.name AND c.source = 'memory' AND c.pending_revision = 0 AND COALESCE(c.is_recap, 0) = 0 AND c.source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)) as mem_count,
                        (SELECT COUNT(*) FROM entities e WHERE e.domain = s.name) as ent_count,
                        s.sort_order, s.starred
                 FROM spaces s
                 ORDER BY s.starred DESC, s.sort_order, s.name",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_spaces: {}", e)))?;

        let mut spaces = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            spaces.push(Space {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                description: row.get::<Option<String>>(2).unwrap_or(None),
                suggested: row.get::<i32>(3).unwrap_or(0) != 0,
                created_at: row.get::<f64>(4).unwrap_or(0.0),
                updated_at: row.get::<f64>(5).unwrap_or(0.0),
                memory_count: row.get::<u64>(6).unwrap_or(0),
                entity_count: row.get::<u64>(7).unwrap_or(0),
                sort_order: row.get::<i64>(8).unwrap_or(0),
                starred: row.get::<i32>(9).unwrap_or(0) != 0,
            });
        }
        Ok(spaces)
    }

    pub async fn get_space(&self, name: &str) -> Result<Option<Space>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT s.id, s.name, s.description, s.suggested, s.created_at, s.updated_at,
                        (SELECT COUNT(DISTINCT c.source_id) FROM memories c WHERE c.domain = s.name AND c.source = 'memory' AND c.pending_revision = 0 AND COALESCE(c.is_recap, 0) = 0 AND c.source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)) as mem_count,
                        (SELECT COUNT(*) FROM entities e WHERE e.domain = s.name) as ent_count,
                        s.sort_order
                 FROM spaces s WHERE s.name = ?1",
                libsql::params![name],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_space: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(Some(Space {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                description: row.get::<Option<String>>(2).unwrap_or(None),
                suggested: row.get::<i32>(3).unwrap_or(0) != 0,
                created_at: row.get::<f64>(4).unwrap_or(0.0),
                updated_at: row.get::<f64>(5).unwrap_or(0.0),
                memory_count: row.get::<u64>(6).unwrap_or(0),
                entity_count: row.get::<u64>(7).unwrap_or(0),
                sort_order: row.get::<i64>(8).unwrap_or(0),
                starred: row.get::<i32>(9).unwrap_or(0) != 0,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn create_space(
        &self,
        name: &str,
        description: Option<&str>,
        suggested: bool,
    ) -> Result<Space, OriginError> {
        let conn = self.conn.lock().await;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp() as f64;
        // New spaces go to the end
        let mut rows = conn
            .query("SELECT COALESCE(MAX(sort_order), -1) + 1 FROM spaces", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("max_sort_order: {}", e)))?;
        let next_order: i64 = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            0
        };
        drop(rows);

        conn.execute(
            "INSERT INTO spaces (id, name, description, suggested, sort_order, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            libsql::params![id.clone(), name, description, suggested as i32, next_order, now, now],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("create_space: {}", e)))?;

        Ok(Space {
            id,
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            suggested,
            sort_order: next_order,
            starred: false,
            memory_count: 0,
            entity_count: 0,
            created_at: now,
            updated_at: now,
        })
    }

    pub async fn update_space(
        &self,
        name: &str,
        new_name: &str,
        description: Option<&str>,
    ) -> Result<Space, OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp() as f64;

        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("update_space begin: {}", e)))?;

        let txn_result = async {
            conn.execute(
                "UPDATE spaces SET name = ?1, description = ?2, updated_at = ?3 WHERE name = ?4",
                libsql::params![new_name, description, now, name],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("update_space: {}", e)))?;

            if name != new_name {
                conn.execute(
                    "UPDATE memories SET domain = ?1 WHERE domain = ?2",
                    libsql::params![new_name, name],
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("update_space cascade memories: {}", e))
                })?;

                conn.execute(
                    "UPDATE entities SET domain = ?1 WHERE domain = ?2",
                    libsql::params![new_name, name],
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("update_space cascade entities: {}", e))
                })?;
            }

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("update_space commit: {}", e)))?;

            Ok::<(), OriginError>(())
        }
        .await;

        if let Err(e) = txn_result {
            let _ = conn.execute("ROLLBACK", ()).await;
            return Err(e);
        }

        drop(conn);
        self.get_space(new_name)
            .await?
            .ok_or_else(|| OriginError::VectorDb("space not found after update".into()))
    }

    /// Delete a space. `memory_action`:
    /// - "keep" = memories keep their domain tag (orphaned, re-adopted if space recreated)
    /// - "unassign" = memories have domain set to NULL
    /// - "delete" = memories are deleted entirely
    /// - "move:target" = memories moved to target space
    pub async fn delete_space(&self, name: &str, memory_action: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;

        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_space begin: {}", e)))?;

        let txn_result = async {
            match memory_action {
                "keep" => { /* do nothing — orphan memories with domain intact */ }
                "unassign" => {
                    conn.execute(
                        "UPDATE memories SET domain = NULL WHERE domain = ?1",
                        libsql::params![name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space unassign memories: {}", e))
                    })?;
                    conn.execute(
                        "UPDATE entities SET domain = NULL WHERE domain = ?1",
                        libsql::params![name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space unassign entities: {}", e))
                    })?;
                }
                "delete" => {
                    conn.execute(
                        "DELETE FROM memories WHERE domain = ?1 AND source = 'memory'",
                        libsql::params![name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space delete memories: {}", e))
                    })?;
                    conn.execute(
                        "DELETE FROM entities WHERE domain = ?1",
                        libsql::params![name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space delete entities: {}", e))
                    })?;
                }
                other if other.starts_with("move:") => {
                    let target = &other[5..];
                    conn.execute(
                        "UPDATE memories SET domain = ?1 WHERE domain = ?2",
                        libsql::params![target, name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space move memories: {}", e))
                    })?;
                    conn.execute(
                        "UPDATE entities SET domain = ?1 WHERE domain = ?2",
                        libsql::params![target, name],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("delete_space move entities: {}", e))
                    })?;
                }
                _ => { /* unknown action — treat as keep */ }
            }

            conn.execute("DELETE FROM spaces WHERE name = ?1", libsql::params![name])
                .await
                .map_err(|e| OriginError::VectorDb(format!("delete_space: {}", e)))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("delete_space commit: {}", e)))?;
            Ok::<(), OriginError>(())
        }
        .await;

        if let Err(e) = txn_result {
            let _ = conn.execute("ROLLBACK", ()).await;
            return Err(e);
        }
        Ok(())
    }

    pub async fn confirm_space(&self, name: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp() as f64;
        conn.execute(
            "UPDATE spaces SET suggested = 0, updated_at = ?1 WHERE name = ?2",
            libsql::params![now, name],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("confirm_space: {}", e)))?;
        Ok(())
    }

    /// Move a space to a new position. Reorders other spaces to fill the gap.
    pub async fn reorder_space(&self, name: &str, new_order: i64) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        // Get current order
        let mut rows = conn
            .query(
                "SELECT sort_order FROM spaces WHERE name = ?1",
                libsql::params![name],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("reorder get: {}", e)))?;
        let old_order: i64 = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            return Ok(());
        };
        drop(rows);

        if old_order == new_order {
            return Ok(());
        }

        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("reorder begin: {}", e)))?;

        if new_order < old_order {
            // Moving up: shift items in [new_order, old_order) down by 1
            conn.execute(
                "UPDATE spaces SET sort_order = sort_order + 1 WHERE sort_order >= ?1 AND sort_order < ?2",
                libsql::params![new_order, old_order],
            ).await.map_err(|e| OriginError::VectorDb(format!("reorder shift down: {}", e)))?;
        } else {
            // Moving down: shift items in (old_order, new_order] up by 1
            conn.execute(
                "UPDATE spaces SET sort_order = sort_order - 1 WHERE sort_order > ?1 AND sort_order <= ?2",
                libsql::params![old_order, new_order],
            ).await.map_err(|e| OriginError::VectorDb(format!("reorder shift up: {}", e)))?;
        }

        conn.execute(
            "UPDATE spaces SET sort_order = ?1 WHERE name = ?2",
            libsql::params![new_order, name],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("reorder set: {}", e)))?;

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("reorder commit: {}", e)))?;
        Ok(())
    }

    pub async fn toggle_space_starred(&self, name: &str) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE spaces SET starred = 1 - starred, updated_at = ?1 WHERE name = ?2",
            libsql::params![chrono::Utc::now().timestamp() as f64, name],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("toggle_space_starred: {}", e)))?;

        // Return new value
        let mut rows = conn
            .query(
                "SELECT starred FROM spaces WHERE name = ?1",
                libsql::params![name],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        let starred = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i32>(0).unwrap_or(0) != 0
        } else {
            false
        };
        Ok(starred)
    }

    pub async fn auto_create_space_if_needed(&self, domain: &str) -> Result<(), OriginError> {
        if domain.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().await;
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp() as f64;
        conn.execute(
            "INSERT OR IGNORE INTO spaces (id, name, description, suggested, created_at, updated_at)
             VALUES (?1, ?2, NULL, 1, ?3, ?4)",
            libsql::params![id, domain, now, now],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("auto_create_space: {}", e)))?;
        Ok(())
    }

    // ===== Private Helpers =====

    /// Convert a Vec<f32> to a JSON array string for vector32() SQL function.
    fn vec_to_sql(v: &[f32]) -> String {
        let mut s = String::with_capacity(v.len() * 10);
        s.push('[');
        for (i, f) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            // 6 decimal places is plenty for 32-bit float precision
            use std::fmt::Write;
            let _ = write!(s, "{:.6}", f);
        }
        s.push(']');
        s
    }

    /// Get embedding from cache, or compute and cache it.
    fn get_or_compute_embedding(&self, text: &str) -> Result<Vec<f32>, OriginError> {
        {
            let mut cache = self.embedding_cache.lock().unwrap();
            if let Some(cached) = cache.get(text) {
                return Ok(cached);
            }
        }
        let mut embedder = self.embedder.lock().unwrap();
        let embeddings = embedder
            .embed(vec![text], None)
            .map_err(|e| OriginError::Embedding(e.to_string()))?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| OriginError::Embedding("No embedding generated".into()))?;
        drop(embedder);
        self.embedding_cache
            .lock()
            .unwrap()
            .put(text, embedding.clone());
        Ok(embedding)
    }

    /// Generate embeddings for a batch of texts.
    pub fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, OriginError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let mut embedder = self.embedder.lock().unwrap();
        embedder
            .embed(text_refs, None)
            .map_err(|e| OriginError::Embedding(e.to_string()))
    }

    /// Get memories that need re-embedding (structured content was set but embedding is stale).
    pub async fn get_reembed_candidates(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, content, source_text, structured_fields FROM memories
                 WHERE needs_reembed = 1 AND source = 'memory'
                 ORDER BY last_modified DESC LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_reembed_candidates: {}", e)))?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let id: String = row.get(0).unwrap_or_default();
            let content: String = row.get(1).unwrap_or_default();
            let source_text: Option<String> = row.get::<Option<String>>(2).unwrap_or(None);
            let _sf: Option<String> = row.get::<Option<String>>(3).unwrap_or(None);
            // Embed from prose when available (natural language embeds better)
            let embed_text = source_text.unwrap_or(content);
            results.push((id, embed_text));
        }
        Ok(results)
    }

    /// Get memories that need re-embedding, keyed by source_id (needs_reembed = 1).
    pub async fn get_pending_reembeds(
        &self,
        limit: usize,
    ) -> Result<Vec<PendingReembed>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, source_id, content, source_text FROM memories
                 WHERE needs_reembed = 1 AND source = 'memory'
                 ORDER BY last_modified DESC LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_pending_reembeds: {}", e)))?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let chunk_id: String = row.get(0).unwrap_or_default();
            let source_id: String = row.get(1).unwrap_or_default();
            let content: String = row.get(2).unwrap_or_default();
            let source_text: Option<String> = row.get::<Option<String>>(3).unwrap_or(None);
            let embed_text = source_text.unwrap_or_else(|| content.clone());
            results.push(PendingReembed {
                chunk_id,
                source_id,
                embed_text,
            });
        }
        Ok(results)
    }

    /// Re-embed a single memory with fresh embedding from its current content.
    pub async fn reembed_memory(&self, chunk_id: &str, content: &str) -> Result<(), OriginError> {
        let embeddings = self.generate_embeddings(&[content.to_string()])?;
        if embeddings.is_empty() {
            return Err(OriginError::VectorDb("empty embedding result".into()));
        }
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET embedding = vector32(?1), enrichment_status = 'legacy', needs_reembed = 0 WHERE id = ?2",
            libsql::params![Self::vec_to_sql(&embeddings[0]), chunk_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("reembed_memory: {}", e)))?;
        Ok(())
    }

    /// Get imported memories that have not yet been reclassified (memory_type IS NULL).
    /// Returns (source_id, content) pairs grouped by source_id so each import is
    /// processed exactly once, even when chunked into multiple rows.
    pub async fn get_unclassified_imports(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, GROUP_CONCAT(content, ' ') as combined_content
                 FROM memories
                 WHERE source = 'memory'
                   AND source_id LIKE 'import_%'
                   AND memory_type IS NULL
                 GROUP BY source_id
                 ORDER BY MAX(last_modified) DESC
                 LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((source_id, content));
        }
        Ok(results)
    }

    /// Count import memories that still need classification.
    pub async fn count_unclassified_imports(&self) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories
                 WHERE source = 'memory'
                   AND source_id LIKE 'import_%'
                   AND memory_type IS NULL",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let count: i64 = row.get(0).unwrap_or(0);
            Ok(count as usize)
        } else {
            Ok(0)
        }
    }

    /// Update the memory_type for all memories with a given source_id.
    pub async fn update_memory_type(
        &self,
        source_id: &str,
        memory_type: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET memory_type = ?1 WHERE source_id = ?2",
            libsql::params![memory_type, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        Ok(())
    }

    pub async fn update_domain(&self, source_id: &str, domain: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET domain = ?1 WHERE source_id = ?2",
            libsql::params![domain, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        Ok(())
    }

    pub async fn update_quality(&self, source_id: &str, quality: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET quality = ?1 WHERE source_id = ?2",
            libsql::params![quality, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        Ok(())
    }

    pub async fn update_title(&self, source_id: &str, title: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET title = ?1 WHERE source_id = ?2",
            libsql::params![title, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        Ok(())
    }

    /// Falls back gracefully if the index has no data yet.
    /// Check if a memory with this content already exists.
    /// Uses prefix matching (first 200 chars) so long memories that were
    /// chunked by `upsert_documents` can still be detected as duplicates.
    pub async fn has_memory_content(&self, content: &str) -> Result<bool, OriginError> {
        let prefix: String = content.chars().take(200).collect();
        let pattern = format!("{}%", prefix.replace('%', "\\%").replace('_', "\\_"));
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM memories WHERE source = 'memory' AND chunk_index = 0
                 AND (content LIKE ?1 ESCAPE '\\' OR source_text LIKE ?1 ESCAPE '\\') LIMIT 1",
                libsql::params![pattern],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        Ok(rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .is_some())
    }

    pub async fn find_similar_chunk(
        &self,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let vec_str = Self::vec_to_sql(embedding);

        // Use vector_distance for a reliable brute-force scan (small table is fine).
        // DiskANN index may not be immediately consistent after insert.
        let mut rows = conn
            .query(
                "SELECT vector_distance_cos(embedding, vector32(?1)) AS dist
                 FROM memories
                 ORDER BY dist ASC
                 LIMIT 1",
                libsql::params![vec_str],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let distance: f64 = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            Ok(distance < threshold as f64)
        } else {
            Ok(false)
        }
    }

    /// Check if similar content already exists in the memory store.
    /// Returns Some((source_id, cosine_similarity)) if a match is found,
    /// None otherwise. The caller compares similarity against their threshold.
    /// Batch version of `check_novelty` — embeds every content in one
    /// FastEmbed call, then runs the per-content vector_top_k query
    /// sequentially (libSQL's single connection serializes anyway). Used
    /// by the ingest coalescer to amortize the 50–100ms-per-request
    /// embedding cost across concurrent `/api/memory/store` calls.
    ///
    /// Returns a parallel `Vec` — `results[i]` is the novelty result for
    /// `contents[i]`. `None` means no neighbor was found in the DB.
    /// Embedding failures return `Err` (fail closed).
    pub async fn check_novelty_batch(
        &self,
        contents: &[String],
    ) -> Result<Vec<Option<(String, f64)>>, OriginError> {
        if contents.is_empty() {
            return Ok(vec![]);
        }
        let embeddings = self.generate_embeddings(contents).map_err(|e| {
            log::error!(
                "[quality_gate] batch embedding failed for {} docs (fail closed): {e}",
                contents.len()
            );
            e
        })?;
        let conn = self.conn.lock().await;
        let mut results = Vec::with_capacity(contents.len());
        for embedding in embeddings {
            let vec_str = Self::vec_to_sql(&embedding);
            let mut rows = conn
                .query(
                    "SELECT c.source_id, vector_distance_cos(c.embedding, vector32(?1))
                     FROM vector_top_k('memories_vec_idx', vector32(?1), 5) AS vt
                     JOIN memories c ON c.rowid = vt.id
                     WHERE c.source = 'memory' AND c.pending_revision = 0",
                    libsql::params![vec_str],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("check_novelty_batch: {e}")))?;
            match rows.next().await {
                Ok(Some(row)) => {
                    let source_id: String = row.get(0).unwrap_or_default();
                    let distance: f64 = row.get(1).unwrap_or(1.0);
                    let similarity = (1.0 - distance).max(0.0);
                    results.push(Some((source_id, similarity)));
                }
                Ok(None) => results.push(None),
                Err(e) => {
                    return Err(OriginError::VectorDb(format!(
                        "check_novelty_batch row: {e}"
                    )))
                }
            }
        }
        Ok(results)
    }

    pub async fn check_novelty(&self, content: &str) -> Result<Option<(String, f64)>, OriginError> {
        let embedding = self.get_or_compute_embedding(content).map_err(|e| {
            log::error!("[quality_gate] embedding failed (fail closed): {e}");
            e
        })?;
        let vec_str = Self::vec_to_sql(&embedding);

        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT c.source_id, vector_distance_cos(c.embedding, vector32(?1))
             FROM vector_top_k('memories_vec_idx', vector32(?1), 5) AS vt
             JOIN memories c ON c.rowid = vt.id
             WHERE c.source = 'memory' AND c.pending_revision = 0",
                libsql::params![vec_str],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("check_novelty: {e}")))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("check_novelty row: {e}")))?
        {
            let source_id: String = row.get(0).unwrap_or_default();
            let distance: f64 = row.get(1).unwrap_or(1.0);
            let similarity = (1.0 - distance).max(0.0);
            Ok(Some((source_id, similarity)))
        } else {
            Ok(None)
        }
    }

    /// Parse a row from memories table into a SearchResult.
    /// Convert a database row to SearchResult.
    /// Columns: 0=id, 1=content, 2=source, 3=source_id, 4=title, 5=summary,
    /// 6=url, 7=chunk_index, 8=last_modified, 9=chunk_type, 10=language,
    /// 11=byte_start, 12=byte_end, 13=semantic_unit, 14=memory_type, 15=domain,
    /// 16=source_agent, 17=confidence, 18=confirmed, 19=stability, 20=supersedes,
    /// 21=entity_id, 22=quality, 23=is_recap, 24=supersede_mode,
    /// 25=structured_fields, 26=retrieval_cue, 27=source_text, 28=score/distance/rank
    fn row_to_search_result(row: &libsql::Row, score: f32) -> Result<SearchResult, OriginError> {
        Ok(SearchResult {
            id: row
                .get::<String>(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            content: row
                .get::<String>(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            source: row
                .get::<String>(2)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            source_id: row
                .get::<String>(3)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            title: row
                .get::<String>(4)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            url: row.get::<Option<String>>(6).unwrap_or(None),
            chunk_index: row.get::<i32>(7).unwrap_or(0),
            last_modified: row.get::<i64>(8).unwrap_or(0),
            score,
            chunk_type: row.get::<Option<String>>(9).unwrap_or(None),
            language: row.get::<Option<String>>(10).unwrap_or(None),
            semantic_unit: row.get::<Option<String>>(13).unwrap_or(None),
            memory_type: row.get::<Option<String>>(14).unwrap_or(None),
            domain: row.get::<Option<String>>(15).unwrap_or(None),
            source_agent: row.get::<Option<String>>(16).unwrap_or(None),
            confidence: row.get::<Option<f64>>(17).unwrap_or(None).map(|v| v as f32),
            confirmed: row.get::<Option<i64>>(18).unwrap_or(None).map(|v| v != 0),
            stability: row.get::<Option<String>>(19).unwrap_or(None),
            supersedes: row.get::<Option<String>>(20).unwrap_or(None),
            summary: row.get::<Option<String>>(5).unwrap_or(None),
            entity_id: row.get::<Option<String>>(21).unwrap_or(None),
            entity_name: None, // Populated separately via entity lookup
            quality: row.get::<Option<String>>(22).unwrap_or(None),
            is_recap: row
                .get::<Option<i64>>(23)
                .unwrap_or(None)
                .is_some_and(|v| v != 0),
            is_archived: false, // Set by supersede logic after row read
            structured_fields: row.get::<Option<String>>(25).unwrap_or(None),
            retrieval_cue: row.get::<Option<String>>(26).unwrap_or(None),
            source_text: row.get::<Option<String>>(27).unwrap_or(None),
            raw_score: 0.0, // Set later during normalization
        })
    }

    // ===== Chunk Methods (matching VectorDB API) =====

    /// Chunk, embed, and upsert documents. Returns the number of memory rows created.
    pub async fn upsert_documents(&self, docs: Vec<RawDocument>) -> Result<usize, OriginError> {
        if docs.is_empty() {
            return Ok(0);
        }

        // Collect all memory rows across all documents
        struct MemoryRow {
            id: String,
            content: String,
            source: String,
            source_id: String,
            title: String,
            summary: Option<String>,
            url: Option<String>,
            chunk_index: i32,
            last_modified: i64,
            chunk_type: String,
            language: Option<String>,
            byte_start: Option<i64>,
            byte_end: Option<i64>,
            semantic_unit: Option<String>,
            memory_type: Option<String>,
            domain: Option<String>,
            source_agent: Option<String>,
            confidence: Option<f32>,
            confirmed: Option<bool>,
            stability: Option<String>,
            supersedes: Option<String>,
            pending_revision: bool,
            word_count: i64,
            entity_id: Option<String>,
            // Retired column: kept in struct for compatibility with RawDocument
            // but ignored on INSERT (always written as "legacy"). Status is now
            // derived from the enrichment_steps table.
            #[allow(dead_code)]
            enrichment_status: String,
            quality: Option<String>,
            is_recap: bool,
            supersede_mode: String,
            structured_fields: Option<String>,
            retrieval_cue: Option<String>,
            source_text: Option<String>,
        }

        let mut memory_rows: Vec<MemoryRow> = Vec::new();
        let mut chunk_texts: Vec<String> = Vec::new();
        let mut source_ids_to_delete: HashSet<(String, String)> = HashSet::new();

        for doc in &docs {
            let content = redact_pii(&doc.content);
            let metadata = doc.metadata.clone();

            // Content is always the original natural language prose.
            // structured_fields JSON holds the structured representation for display.
            // (Previously this flattened structured_fields into pipe-delimited content
            // and saved original to source_text — that caused redundant storage and
            // made content worse for display, FTS, and embedding.)
            let (final_content, derived_source_text) = (content.clone(), None::<String>);

            let chunks = self
                .chunker
                .chunk(&final_content, &doc.title, &doc.source_id, &metadata);

            source_ids_to_delete.insert((doc.source.clone(), doc.source_id.clone()));

            for (i, chunk) in chunks.iter().enumerate() {
                let mut hasher = Sha256::new();
                hasher.update(doc.source_id.as_bytes());
                hasher.update(i.to_string().as_bytes());
                let hash = format!("{:x}", hasher.finalize());
                let chunk_id = hash[..16].to_string();

                // Contextual enrichment: for memory rows, prepend domain as embedding
                // prefix. Autoresearch ablation (2026-04-03) showed [domain]-only
                // is +25% NDCG over [type | domain] — type labels add noise that
                // dilutes semantic signal. Retrieval cue follows on its own line.
                let embedding_text = if doc.source == "memory" {
                    let prefix = if let Some(ref d) = doc.domain {
                        format!("[{}] ", d)
                    } else {
                        String::new()
                    };
                    let cue_prefix = doc.retrieval_cue.as_deref().unwrap_or("");
                    let embed_content = derived_source_text.as_deref().unwrap_or(&chunk.content);
                    if cue_prefix.is_empty() {
                        format!("{}{}", prefix, embed_content)
                    } else {
                        format!("{}{}\n{}", prefix, cue_prefix, embed_content)
                    }
                } else {
                    chunk.content.clone()
                };

                chunk_texts.push(embedding_text);

                memory_rows.push(MemoryRow {
                    id: chunk_id,
                    content: chunk.content.clone(),
                    source: doc.source.clone(),
                    source_id: doc.source_id.clone(),
                    title: doc.title.clone(),
                    summary: doc.summary.clone(),
                    url: doc.url.clone(),
                    chunk_index: i as i32,
                    last_modified: doc.last_modified,
                    chunk_type: chunk.chunk_type.clone(),
                    language: chunk.language.clone(),
                    byte_start: chunk.byte_range.map(|(s, _)| s as i64),
                    byte_end: chunk.byte_range.map(|(_, e)| e as i64),
                    semantic_unit: chunk.semantic_unit.clone(),
                    memory_type: doc.memory_type.clone(),
                    domain: doc.domain.clone(),
                    source_agent: doc.source_agent.clone(),
                    confidence: doc.confidence,
                    confirmed: doc.confirmed,
                    stability: doc.stability.clone(),
                    supersedes: doc.supersedes.clone(),
                    pending_revision: doc.pending_revision,
                    word_count: chunk.content.split_whitespace().count() as i64,
                    entity_id: doc.entity_id.clone(),
                    enrichment_status: doc.enrichment_status.clone(),
                    quality: doc.quality.clone(),
                    is_recap: doc.is_recap,
                    supersede_mode: doc.supersede_mode.clone(),
                    structured_fields: doc.structured_fields.clone(),
                    retrieval_cue: doc.retrieval_cue.clone(),
                    // derived_source_text wins when we promoted structured_fields to content.
                    // Fall back to doc.source_text if no promotion happened.
                    source_text: derived_source_text
                        .clone()
                        .or_else(|| doc.source_text.clone()),
                });
            }
        }

        if memory_rows.is_empty() {
            return Ok(0);
        }

        // Generate embeddings for all memory rows
        let embeddings = self.generate_embeddings(&chunk_texts)?;
        if embeddings.len() != memory_rows.len() {
            return Err(OriginError::Embedding(format!(
                "Expected {} embeddings, got {}",
                memory_rows.len(),
                embeddings.len()
            )));
        }

        let conn = self.conn.lock().await;

        // Wrap deletes + inserts in a transaction for atomicity and performance
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("begin transaction: {}", e)))?;

        // Delete existing rows for these source_ids
        for (source, source_id) in &source_ids_to_delete {
            conn.execute(
                "DELETE FROM memories WHERE source = ?1 AND source_id = ?2",
                libsql::params![source.clone(), source_id.clone()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete old memories: {}", e)))?;
        }

        // Insert new memory rows with proper NULL handling for optional fields
        let total = memory_rows.len();
        for (row, embedding) in memory_rows.into_iter().zip(embeddings.iter()) {
            let vec_str = Self::vec_to_sql(embedding);
            let confirmed_int: Option<i64> = row.confirmed.map(|b| if b { 1 } else { 0 });

            let summary_val: libsql::Value =
                row.summary.map(|s| s.into()).unwrap_or(libsql::Value::Null);
            let url_val: libsql::Value = row.url.map(|s| s.into()).unwrap_or(libsql::Value::Null);
            let language_val: libsql::Value = row
                .language
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let byte_start_val: libsql::Value = row
                .byte_start
                .map(|v| v.into())
                .unwrap_or(libsql::Value::Null);
            let byte_end_val: libsql::Value = row
                .byte_end
                .map(|v| v.into())
                .unwrap_or(libsql::Value::Null);
            let semantic_unit_val: libsql::Value = row
                .semantic_unit
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let memory_type_val: libsql::Value = row
                .memory_type
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let domain_val: libsql::Value =
                row.domain.map(|s| s.into()).unwrap_or(libsql::Value::Null);
            let source_agent_val: libsql::Value = row
                .source_agent
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let confidence_val: libsql::Value = row
                .confidence
                .map(|v| (v as f64).into())
                .unwrap_or(libsql::Value::Null);
            let confirmed_val: libsql::Value = confirmed_int
                .map(|v| v.into())
                .unwrap_or(libsql::Value::Null);
            let stability_val: libsql::Value = row
                .stability
                .clone()
                .map(libsql::Value::Text)
                .unwrap_or(libsql::Value::Text("new".to_string()));
            let supersedes_val: libsql::Value = row
                .supersedes
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let pending_revision_val: i64 = if row.pending_revision { 1 } else { 0 };
            let entity_id_val: libsql::Value = row
                .entity_id
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let quality_val: libsql::Value =
                row.quality.map(|s| s.into()).unwrap_or(libsql::Value::Null);
            let is_recap_val: i64 = if row.is_recap { 1 } else { 0 };
            let structured_fields_val: libsql::Value = row
                .structured_fields
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let retrieval_cue_val: libsql::Value = row
                .retrieval_cue
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);
            let source_text_val: libsql::Value = row
                .source_text
                .map(|s| s.into())
                .unwrap_or(libsql::Value::Null);

            conn.execute(
                "INSERT INTO memories (id, content, source, source_id, title, summary, url,
                    chunk_index, last_modified, chunk_type, language, byte_start, byte_end,
                    semantic_unit, memory_type, domain, source_agent, confidence, confirmed,
                    stability, supersedes, pending_revision, word_count,
                    entity_id, enrichment_status, quality, is_recap, supersede_mode,
                    structured_fields, retrieval_cue, source_text,
                    embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                    ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23,
                    ?24, ?25, ?26, ?27, ?28,
                    ?29, ?30, ?31,
                    vector32(?32), ?33)",
                libsql::params![
                    row.id,
                    row.content,
                    row.source,
                    row.source_id,
                    row.title,
                    summary_val,
                    url_val,
                    row.chunk_index as i64,
                    row.last_modified,
                    row.chunk_type,
                    language_val,
                    byte_start_val,
                    byte_end_val,
                    semantic_unit_val,
                    memory_type_val,
                    domain_val,
                    source_agent_val,
                    confidence_val,
                    confirmed_val,
                    stability_val,
                    supersedes_val,
                    pending_revision_val,
                    row.word_count,
                    entity_id_val,
                    "legacy", // column retired; status derived from enrichment_steps
                    quality_val,
                    is_recap_val,
                    row.supersede_mode,
                    structured_fields_val,
                    retrieval_cue_val,
                    source_text_val,
                    vec_str,
                    row.last_modified // created_at = last_modified at insert time
                ],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("insert memory: {}", e)))?;
        }

        // Soft-suppress superseded memories (skip when pending_revision — human hasn't approved yet)
        for doc in &docs {
            if let Some(ref superseded_id) = doc.supersedes {
                if !doc.pending_revision {
                    conn.execute(
                        "UPDATE memories SET confirmed = 0 WHERE source_id = ?1 AND source = 'memory'",
                        libsql::params![superseded_id.to_string()],
                    )
                    .await
                    .ok(); // Best effort
                }
            }
        }

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("commit transaction: {}", e)))?;

        log::info!("[memory_db] upserted {} memories", total);
        Ok(total)
    }

    /// Hybrid search: vector similarity + FTS, combined with Reciprocal Rank Fusion.
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);
        let fetch_limit = (limit * 3) as i64;

        let conn = self.conn.lock().await;

        // --- Vector search ---
        let mut vector_results: Vec<SearchResult> = Vec::new();
        {
            let sql = if source_filter.is_some() {
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        vector_distance_cos(c.embedding, vector32(?1))
                 FROM vector_top_k('memories_vec_idx', vector32(?1), ?2) AS vt
                 JOIN memories c ON c.rowid = vt.id
                 WHERE c.pending_revision = 0 AND c.source = ?3"
            } else {
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        vector_distance_cos(c.embedding, vector32(?1))
                 FROM vector_top_k('memories_vec_idx', vector32(?1), ?2) AS vt
                 JOIN memories c ON c.rowid = vt.id
                 WHERE c.pending_revision = 0"
            };

            let rows_result = if let Some(filter) = source_filter {
                conn.query(
                    sql,
                    libsql::params![vec_str.clone(), fetch_limit, filter.to_string()],
                )
                .await
            } else {
                conn.query(sql, libsql::params![vec_str, fetch_limit]).await
            };

            match rows_result {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        let distance: f64 = row.get(28).unwrap_or(1.0);
                        if let Ok(result) = Self::row_to_search_result(&row, distance as f32) {
                            vector_results.push(result);
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "[memory_db] vector search failed (index may not exist): {}",
                        e
                    );
                    // Fall through to FTS-only results
                }
            }
        }

        // --- FTS search ---
        let mut fts_results: Vec<SearchResult> = Vec::new();
        {
            let fts_sql = if source_filter.is_some() {
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        fts.rank
                 FROM memories_fts fts
                 JOIN memories c ON fts.rowid = c.rowid
                 WHERE memories_fts MATCH ?1 AND c.pending_revision = 0 AND c.source = ?3
                 ORDER BY fts.rank
                 LIMIT ?2"
            } else {
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        fts.rank
                 FROM memories_fts fts
                 JOIN memories c ON fts.rowid = c.rowid
                 WHERE memories_fts MATCH ?1 AND c.pending_revision = 0
                 ORDER BY fts.rank
                 LIMIT ?2"
            };

            let fts_result = if let Some(filter) = source_filter {
                conn.query(
                    fts_sql,
                    libsql::params![query.to_string(), fetch_limit, filter.to_string()],
                )
                .await
            } else {
                conn.query(fts_sql, libsql::params![query.to_string(), fetch_limit])
                    .await
            };

            match fts_result {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        let rank: f64 = row.get(28).unwrap_or(0.0);
                        if let Ok(result) = Self::row_to_search_result(&row, rank as f32) {
                            fts_results.push(result);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[memory_db] FTS search failed: {}", e);
                }
            }
        }

        // --- Reciprocal Rank Fusion (distance-weighted for vector signal) ---
        let mut score_map: HashMap<String, f32> = HashMap::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        for (rank, result) in vector_results.into_iter().enumerate() {
            let distance = result.score;
            let similarity = (1.0 - distance).max(0.01);
            let rrf_score = similarity / (60.0 + rank as f32);
            *score_map.entry(result.id.clone()).or_default() += rrf_score;
            result_map.entry(result.id.clone()).or_insert(result);
        }

        for (rank, result) in fts_results.into_iter().enumerate() {
            let rrf_score = 1.0 / (60.0 + rank as f32);
            *score_map.entry(result.id.clone()).or_default() += rrf_score;
            result_map.entry(result.id.clone()).or_insert(result);
        }

        let mut final_results: Vec<SearchResult> = result_map
            .into_values()
            .map(|mut r| {
                r.score = *score_map.get(&r.id).unwrap_or(&0.0);
                r
            })
            .collect();

        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(limit);

        Ok(final_results)
    }

    /// Normalize old memory type names to the 5-type system.
    /// Maps: correction→fact, custom→fact, recap→fact. Passes others through.
    fn normalize_type_filter(type_str: &str) -> String {
        type_str
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| match s {
                "correction" => "fact",
                "custom" => "fact",
                "recap" => "fact",
                other => other,
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Hybrid search (vector + FTS + RRF) with memory-specific filters.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_memory(
        &self,
        query: &str,
        limit: usize,
        memory_type: Option<&str>,
        domain: Option<&str>,
        source_agent: Option<&str>,
        confirmation_boost: Option<f32>,
        recap_penalty: Option<f32>,
        scoring: Option<&crate::tuning::SearchScoringConfig>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        let t_embed = std::time::Instant::now();
        let embedding = self.get_or_compute_embedding(query)?;
        let embed_ms = t_embed.elapsed().as_millis();
        log::warn!(
            "[memory_db] timing: embed={}ms ({})",
            embed_ms,
            if embed_ms > 5 { "miss" } else { "hit" }
        );
        let vec_str = Self::vec_to_sql(&embedding);
        let fetch_limit = (limit * 3) as i64;

        let t_lock = std::time::Instant::now();
        let conn = self.conn.lock().await;
        log::warn!(
            "[memory_db] timing: conn_lock={}ms",
            t_lock.elapsed().as_millis()
        );

        // Build memory-specific filter conditions (shared by vector and FTS paths)
        let mut filter_conditions: Vec<String> = Vec::new();
        let mut filter_values: Vec<libsql::Value> = Vec::new();

        if let Some(mt) = memory_type {
            // Normalize old type names and support comma-separated types
            let normalized = Self::normalize_type_filter(mt);
            let types: Vec<&str> = normalized
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if types.len() == 1 {
                filter_conditions.push("c.memory_type = ?".to_string());
                filter_values.push(libsql::Value::Text(types[0].to_string()));
            } else if types.len() > 1 {
                let placeholders: Vec<String> = types.iter().map(|_| "?".to_string()).collect();
                filter_conditions.push(format!("c.memory_type IN ({})", placeholders.join(",")));
                for t in &types {
                    filter_values.push(libsql::Value::Text(t.to_string()));
                }
            }
        }
        if let Some(d) = domain {
            if d == "uncategorized" {
                filter_conditions.push("c.domain IS NULL".to_string());
            } else {
                filter_conditions.push("c.domain = ?".to_string());
                filter_values.push(libsql::Value::Text(d.to_string()));
            }
        }
        if let Some(sa) = source_agent {
            filter_conditions.push("c.source_agent = ?".to_string());
            filter_values.push(libsql::Value::Text(sa.to_string()));
        }

        // Supersedes exclusion: only hide memories superseded with mode='hide'.
        // Memories with supersede_mode='archive' remain visible (marked is_archived post-query).
        let supersedes_exclusion = "AND c.pending_revision = 0 AND c.source_id NOT IN (\
            SELECT supersedes FROM memories \
            WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' \
            AND supersede_mode = 'hide' \
            GROUP BY supersedes\
        )";

        // --- Vector search ---
        let mut vector_results: Vec<SearchResult> = Vec::new();
        {
            // Renumber placeholders: ?1 = vector, ?2 = limit, ?3.. = filters
            // Each condition may have multiple ? placeholders (e.g. IN (?,?,?))
            let mut vec_where_parts: Vec<String> = Vec::new();
            let mut param_idx = 3usize;
            for cond in &filter_conditions {
                let mut numbered = String::new();
                for ch in cond.chars() {
                    if ch == '?' {
                        numbered.push_str(&format!("?{}", param_idx));
                        param_idx += 1;
                    } else {
                        numbered.push(ch);
                    }
                }
                vec_where_parts.push(numbered);
            }
            let vec_filter = if vec_where_parts.is_empty() {
                String::new()
            } else {
                format!("AND {}", vec_where_parts.join(" AND "))
            };

            let sql = format!(
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        vector_distance_cos(c.embedding, vector32(?1))
                 FROM vector_top_k('memories_vec_idx', vector32(?1), ?2) AS vt
                 JOIN memories c ON c.rowid = vt.id
                 WHERE 1=1 {} {}",
                supersedes_exclusion, vec_filter
            );

            let mut params: Vec<libsql::Value> = vec![
                libsql::Value::Text(vec_str),
                libsql::Value::Integer(fetch_limit),
            ];
            params.extend(filter_values.clone());

            match conn.query(&sql, params).await {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        let distance: f64 = row.get(28).unwrap_or(1.0);
                        if let Ok(result) = Self::row_to_search_result(&row, distance as f32) {
                            vector_results.push(result);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[memory_db] memory vector search failed: {}", e);
                }
            }
        }

        // --- FTS search ---
        let mut fts_results: Vec<SearchResult> = Vec::new();
        {
            // Renumber placeholders: ?1 = query, ?2 = limit, ?3.. = filters
            let mut fts_where_parts: Vec<String> = Vec::new();
            let mut param_idx = 3usize;
            for cond in &filter_conditions {
                let mut numbered = String::new();
                for ch in cond.chars() {
                    if ch == '?' {
                        numbered.push_str(&format!("?{}", param_idx));
                        param_idx += 1;
                    } else {
                        numbered.push(ch);
                    }
                }
                fts_where_parts.push(numbered);
            }
            let fts_extra = if fts_where_parts.is_empty() {
                String::new()
            } else {
                format!(" AND {}", fts_where_parts.join(" AND "))
            };

            let fts_sql = format!(
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        fts.rank
                 FROM memories_fts fts
                 JOIN memories c ON fts.rowid = c.rowid
                 WHERE memories_fts MATCH ?1 {} {}
                 ORDER BY fts.rank
                 LIMIT ?2",
                supersedes_exclusion, fts_extra
            );

            // Try AND matching first (implicit FTS5 default), fall back to OR
            // if no results. OR matching broadens recall for multi-word queries
            // where no single document contains all terms.
            let fts_queries = vec![query.to_string(), Self::fts_or_query(query)];

            for fts_query in &fts_queries {
                let mut params: Vec<libsql::Value> = vec![
                    libsql::Value::Text(fts_query.clone()),
                    libsql::Value::Integer(fetch_limit),
                ];
                params.extend(filter_values.clone());

                match conn.query(&fts_sql, params).await {
                    Ok(mut rows) => {
                        while let Ok(Some(row)) = rows.next().await {
                            let rank: f64 = row.get(28).unwrap_or(0.0);
                            if let Ok(result) = Self::row_to_search_result(&row, rank as f32) {
                                fts_results.push(result);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("[memory_db] memory FTS search failed: {}", e);
                    }
                }
                if !fts_results.is_empty() {
                    break; // AND matched, no need for OR fallback
                }
            }
        }

        // --- Reciprocal Rank Fusion (distance-weighted for vector signal) ---
        // Standard RRF uses 1/(k+rank) which barely differentiates in small pools.
        // Weight the vector signal by cosine similarity so genuinely close matches
        // score proportionally higher than items that barely made the top-k.
        let mut score_map: HashMap<String, f32> = HashMap::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        let rrf_k = scoring.map(|s| s.rrf_k).unwrap_or(60.0);

        for (rank, result) in vector_results.into_iter().enumerate() {
            let distance = result.score; // cosine distance from vector_distance_cos
            let similarity = (1.0 - distance).max(0.01);
            let rrf_score = similarity / (rrf_k + rank as f32);
            *score_map.entry(result.id.clone()).or_default() += rrf_score;
            result_map.entry(result.id.clone()).or_insert(result);
        }

        // FTS weight: downweight keyword signal to reduce noise from keyword-heavy
        // negatives that share surface terms but are semantically wrong.
        let fts_weight = scoring.map(|s| s.fts_weight).unwrap_or(0.2);

        for (rank, result) in fts_results.into_iter().enumerate() {
            let rrf_score = fts_weight / (rrf_k + rank as f32);
            *score_map.entry(result.id.clone()).or_default() += rrf_score;
            result_map.entry(result.id.clone()).or_insert(result);
        }

        // Collect source_ids that are superseded by archive-mode memories
        let archived_ids: HashSet<String> = {
            let mut ids = HashSet::new();
            let mut rows = conn.query(
                "SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' AND supersede_mode = 'archive'",
                libsql::params![],
            ).await.map_err(|e| OriginError::VectorDb(format!("archived_ids query: {}", e)))?;
            while let Ok(Some(row)) = rows.next().await {
                if let Ok(sid) = row.get::<String>(0) {
                    ids.insert(sid);
                }
            }
            ids
        };

        let now = chrono::Utc::now().timestamp();

        // Domain boost: collect unique domains from results, check if query mentions any.
        // If it does, results matching that domain get a 1.5x boost.
        let query_lower = query.to_lowercase();
        let query_domains: HashSet<String> = result_map
            .values()
            .filter_map(|r| r.domain.clone())
            .filter(|d| query_lower.contains(&d.to_lowercase()))
            .collect();

        let mut final_results: Vec<SearchResult> = result_map
            .into_values()
            .map(|mut r| {
                let rrf = *score_map.get(&r.id).unwrap_or(&0.0);

                // Tiered retrieval: weight by confidence and recency decay
                let conf = r.confidence.unwrap_or(0.5);
                let tier = stability_tier(r.memory_type.as_deref());
                // Inline decay rates (match TuningConfig defaults) — search doesn't hold config ref
                let dr = match tier {
                    crate::sources::StabilityTier::Protected => 0.001,
                    crate::sources::StabilityTier::Standard => 0.01,
                    crate::sources::StabilityTier::Ephemeral => 0.05,
                };
                let age_days = ((now - r.last_modified) as f64 / 86400.0).max(0.0);
                let recency = (-dr * age_days).exp() as f32;

                // Quality multiplier: low=0.7, medium/NULL=0.9, high=1.0
                let quality_mult = match r.quality.as_deref() {
                    Some("high") => 1.0f32,
                    Some("low") => 0.7,
                    _ => 0.9, // medium or NULL
                };

                // Stability boost: confirmed > learned > new
                let eff_confirm_boost = confirmation_boost
                    .unwrap_or_else(|| scoring.map(|s| s.confirmation_boost).unwrap_or(2.5));
                let confirm_mult = match r.stability.as_deref() {
                    Some("confirmed") => eff_confirm_boost,
                    Some("learned") => 1.2,
                    _ => {
                        if r.confirmed == Some(true) {
                            eff_confirm_boost
                        } else {
                            1.0
                        }
                    }
                };

                // Recap penalty: recaps are summaries, originals should rank higher
                let eff_recap_penalty = recap_penalty
                    .unwrap_or_else(|| scoring.map(|s| s.recap_penalty).unwrap_or(0.3));
                let recap_mult = if r.is_recap {
                    eff_recap_penalty
                } else {
                    1.0f32
                };

                // Domain relevance boost: if query mentions a domain name,
                // memories from that domain get 1.5x to surface above cross-project noise
                let eff_domain_boost = scoring.map(|s| s.domain_boost).unwrap_or(1.5);
                let domain_mult = if !query_domains.is_empty() {
                    if let Some(ref d) = r.domain {
                        if query_domains.contains(d) {
                            eff_domain_boost
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                r.score =
                    rrf * conf * recency * quality_mult * confirm_mult * recap_mult * domain_mult;

                // Mark archived decisions (superseded with mode='archive')
                if archived_ids.contains(&r.source_id) {
                    r.is_archived = true;
                    r.score *= 0.5; // Halve score for archived results
                }

                r
            })
            .collect();
        // Sort by score descending, then by source_id ascending for deterministic tie-breaking
        final_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source_id.cmp(&b.source_id))
        });

        // --- Graph augmentation (knowledge graph as third RRF signal) ---
        // Drop the conn lock first — augment_with_graph acquires it internally.
        drop(conn);
        final_results = self.augment_with_graph(query, final_results, limit).await?;

        // Deduplicate: keep only the best-scoring chunk per source document.
        // This prevents recap entries and multi-chunk documents from flooding results.
        {
            let mut seen_sources: HashSet<String> = HashSet::new();
            final_results.retain(|r| seen_sources.insert(r.source_id.clone()));
        }

        // Content-based dedup: if a recap overlaps heavily with an original, drop the recap.
        // Uses bigram Jaccard similarity (0.5 threshold — lower than capture dedup because
        // recaps reformulate content rather than copy it verbatim).
        {
            let mut drop_indices: HashSet<usize> = HashSet::new();
            for i in 0..final_results.len() {
                if drop_indices.contains(&i) {
                    continue;
                }
                for j in (i + 1)..final_results.len() {
                    if drop_indices.contains(&j) {
                        continue;
                    }
                    // Only dedup if at least one is a recap
                    if !final_results[i].is_recap && !final_results[j].is_recap {
                        continue;
                    }
                    let sim = bigram_jaccard(&final_results[i].content, &final_results[j].content);
                    if sim > 0.5 {
                        // Drop the lower-scoring one (j, since results are sorted by score desc)
                        drop_indices.insert(j);
                    }
                }
            }
            if !drop_indices.is_empty() {
                let mut idx = 0;
                final_results.retain(|_| {
                    let keep = !drop_indices.contains(&idx);
                    idx += 1;
                    keep
                });
            }
        }

        // Near-duplicate dedup: catch near-identical content (Jaccard > 0.92) regardless of
        // recap status. This is a safety net for eval seeds and scenarios where post-ingest
        // dedup was bypassed. Keeps only the highest-ranked of each cluster.
        {
            let mut drop_indices: HashSet<usize> = HashSet::new();
            for i in 0..final_results.len() {
                if drop_indices.contains(&i) {
                    continue;
                }
                for j in (i + 1)..final_results.len() {
                    if drop_indices.contains(&j) {
                        continue;
                    }
                    let sim = bigram_jaccard(&final_results[i].content, &final_results[j].content);
                    if sim > 0.92 {
                        // Drop the lower-scoring one (j, since results are sorted by score desc)
                        drop_indices.insert(j);
                    }
                }
            }
            if !drop_indices.is_empty() {
                let mut idx = 0;
                final_results.retain(|_| {
                    let keep = !drop_indices.contains(&idx);
                    idx += 1;
                    keep
                });
            }
        }

        // Supersede-aware dedup: if a distilled (superseding) memory is in results, remove the
        // raw source it supersedes. This ensures MCP recall surfaces the enriched version only,
        // not both the original and its distillation.
        //
        // Exception: decisions use archive mode for history (you want to see "we chose MongoDB,
        // then switched to PostgreSQL"). Only non-decision types prefer the distilled version.
        {
            let mut superseded_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for result in &final_results {
                if let Some(ref supersedes) = result.supersedes {
                    superseded_ids.insert(supersedes.clone());
                }
            }
            if !superseded_ids.is_empty() {
                final_results.retain(|r| {
                    // Always keep if not superseded
                    if !superseded_ids.contains(&r.source_id) {
                        return true;
                    }
                    // Keep decisions even when superseded — they show historical context
                    r.memory_type.as_deref() == Some("decision")
                });
            }
        }

        // Strip KG observations from output — they're internal scaffolding for score
        // boosting via RRF merge, not user-facing content. Without this filter, low-value
        // observations like "Settings" consume result slots and push real memories out.
        final_results.retain(|r| r.source != "knowledge_graph");

        log::warn!(
            "[memory_db] timing: db_queries={}ms",
            t_lock.elapsed().as_millis()
        );
        final_results.truncate(limit);

        // Preserve raw scores before normalization (for absolute relevance gating).
        for r in &mut final_results {
            r.raw_score = r.score;
        }

        // Score normalization: scale by theoretical maximum so absolute quality is preserved.
        // Min-max normalization (dividing by actual max) always pushes the top result to 1.0,
        // even when it's irrelevant garbage. Instead, divide by the theoretical best-case RRF
        // score so genuinely good matches score high and poor matches score low.
        {
            let theoretical_max_rrf = (1.0 + fts_weight) / rrf_k;
            let eff_confirm = scoring.map(|s| s.confirmation_boost).unwrap_or(2.5);
            let eff_domain = scoring.map(|s| s.domain_boost).unwrap_or(1.5);
            let peak_multiplier = eff_confirm * eff_domain;
            let reference_max = theoretical_max_rrf * peak_multiplier;
            for r in &mut final_results {
                r.score = (r.score / reference_max).min(1.0);
            }
        }

        // Populate entity_name for results that have entity_id
        // Re-acquire conn (was dropped before augment_with_graph to avoid deadlock)
        let entity_ids: Vec<String> = final_results
            .iter()
            .filter_map(|r| r.entity_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        if !entity_ids.is_empty() {
            let placeholders: Vec<String> = entity_ids.iter().map(|_| "?".to_string()).collect();
            let sql = format!(
                "SELECT id, name FROM entities WHERE id IN ({})",
                placeholders.join(",")
            );
            let params: Vec<libsql::Value> = entity_ids
                .iter()
                .map(|id| libsql::Value::Text(id.clone()))
                .collect();
            let mut entity_names: HashMap<String, String> = HashMap::new();
            let conn = self.conn.lock().await;
            if let Ok(mut rows) = conn.query(&sql, params).await {
                while let Ok(Some(row)) = rows.next().await {
                    if let (Ok(id), Ok(name)) = (row.get::<String>(0), row.get::<String>(1)) {
                        entity_names.insert(id, name);
                    }
                }
            }
            drop(conn);
            for r in &mut final_results {
                if let Some(eid) = &r.entity_id {
                    r.entity_name = entity_names.get(eid).cloned();
                }
            }
        }

        Ok(final_results)
    }

    /// Hybrid search with graph augmentation and optional LLM reranking.
    /// This is the quality-focused search path: adds knowledge graph observations
    /// as a third RRF signal, then optionally reranks with the on-device LLM.
    /// Falls back gracefully if graph search or reranking fails.
    pub async fn search_memory_reranked(
        &self,
        query: &str,
        limit: usize,
        memory_type: Option<&str>,
        domain: Option<&str>,
        source_agent: Option<&str>,
        llm: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        // Fetch more candidates than needed for graph merging + reranking
        let fetch_pool = (limit * 2).min(10).max(limit);

        let mut results = self
            .search_memory(
                query,
                fetch_pool,
                memory_type,
                domain,
                source_agent,
                None,
                None,
                None,
            )
            .await?;

        // Note: graph augmentation is now done inside search_memory, no double-augment here.

        // --- LLM reranking ---
        if let Some(llm) = llm {
            if results.len() > 1 {
                let mut candidate_list = String::new();
                for (i, r) in results.iter().enumerate() {
                    let truncated: String = r.content.chars().take(200).collect();
                    candidate_list.push_str(&format!("{}. {}\n", i + 1, truncated));
                }

                let rerank_result = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    llm.generate(crate::llm_provider::LlmRequest {
                        system_prompt: Some(
                            "Rate each result's relevance to the query on a scale of 0-10.\n\
                             Output ONLY a JSON array of integer scores, e.g. [8, 3, 7]."
                                .into(),
                        ),
                        user_prompt: format!("Query: {}\n\n{}", query, candidate_list),
                        max_tokens: 128,
                        temperature: 0.1,
                        label: None,
                    }),
                )
                .await;

                let reranked: Option<Vec<(String, f32)>> = match rerank_result {
                    Ok(Ok(output)) => {
                        // Extract JSON array from output
                        let start_idx = output.find('[');
                        let end_idx = output.rfind(']');
                        if let (Some(si), Some(ei)) = (start_idx, end_idx) {
                            if ei > si {
                                let json_str = &output[si..=ei];
                                match serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
                                    Ok(vals) => {
                                        let scores: Vec<(String, f32)> = results
                                            .iter()
                                            .enumerate()
                                            .map(|(i, r)| {
                                                let score = vals
                                                    .get(i)
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0)
                                                    as f32;
                                                (r.id.clone(), score)
                                            })
                                            .collect();
                                        if scores.is_empty() {
                                            None
                                        } else {
                                            Some(scores)
                                        }
                                    }
                                    Err(e) => {
                                        log::warn!("[memory_db] rerank JSON parse failed: {e}");
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            log::warn!("[memory_db] rerank: no JSON array found in output");
                            None
                        }
                    }
                    Ok(Err(e)) => {
                        log::warn!("[memory_db] rerank LLM failed: {e}");
                        None
                    }
                    Err(_) => {
                        log::warn!("[memory_db] rerank timed out");
                        None
                    }
                };

                if let Some(scores) = reranked {
                    let score_map: HashMap<String, f32> =
                        scores.into_iter().map(|(id, s)| (id, s / 10.0)).collect();
                    for r in &mut results {
                        if let Some(&rerank_score) = score_map.get(&r.id) {
                            r.score = rerank_score;
                        }
                    }
                    results.sort_by(|a, b| {
                        b.score
                            .partial_cmp(&a.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }

        results.truncate(limit);
        Ok(results)
    }

    /// Hybrid search with LLM-based query expansion BEFORE search.
    /// Generates 2-3 alternative phrasings of the query, runs search_memory for each,
    /// then merges all result sets via RRF. Falls back to plain search_memory if LLM
    /// call fails or is absent.
    pub async fn search_memory_expanded(
        &self,
        query: &str,
        limit: usize,
        memory_type: Option<&str>,
        domain: Option<&str>,
        source_agent: Option<&str>,
        llm: Option<Arc<dyn crate::llm_provider::LlmProvider>>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        // Build expanded query list starting with the original
        let mut queries: Vec<String> = vec![query.to_string()];

        if let Some(ref llm) = llm {
            let expand_result = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                llm.generate(crate::llm_provider::LlmRequest {
                    system_prompt: Some(
                        "Rewrite this memory search query into 2-3 alternative phrasings that would match different vocabulary. Output ONLY a JSON array of strings.".into(),
                    ),
                    user_prompt: query.to_string(),
                    max_tokens: 256,
                    temperature: 0.3,
                    label: None,
                }),
            )
            .await;

            match expand_result {
                Ok(Ok(output)) => {
                    // Extract JSON array from output
                    let start_idx = output.find('[');
                    let end_idx = output.rfind(']');
                    if let (Some(si), Some(ei)) = (start_idx, end_idx) {
                        if ei > si {
                            let json_str = &output[si..=ei];
                            match serde_json::from_str::<Vec<String>>(json_str) {
                                Ok(expansions) if !expansions.is_empty() => {
                                    queries.extend(expansions.into_iter().take(3));
                                }
                                Ok(_) => {
                                    log::warn!("[memory_db] expand: empty expansion list, using original query only");
                                }
                                Err(e) => {
                                    log::warn!("[memory_db] expand JSON parse failed: {e}");
                                }
                            }
                        }
                    } else {
                        log::warn!("[memory_db] expand: no JSON array found in output");
                    }
                }
                Ok(Err(e)) => {
                    log::warn!("[memory_db] expand LLM failed: {e}");
                }
                Err(_) => {
                    log::warn!("[memory_db] expand timed out");
                }
            }
        }

        // Fetch a pool per query; we'll merge via RRF
        let fetch_pool = limit * 2;

        // Run search_memory for each query (original + expansions)
        let mut all_ranked: Vec<Vec<SearchResult>> = Vec::with_capacity(queries.len());
        for q in &queries {
            match self
                .search_memory(
                    q,
                    fetch_pool,
                    memory_type,
                    domain,
                    source_agent,
                    None,
                    None,
                    None,
                )
                .await
            {
                Ok(results) => all_ranked.push(results),
                Err(e) => {
                    log::warn!("[memory_db] expand search failed for query '{q}': {e}");
                }
            }
        }

        // If all searches failed, fall back to a plain search on the original query
        if all_ranked.is_empty() {
            return self
                .search_memory(
                    query,
                    limit,
                    memory_type,
                    domain,
                    source_agent,
                    None,
                    None,
                    None,
                )
                .await;
        }

        // Merge via RRF: for each result list, score each item as 1/(60 + rank)
        let mut score_map: HashMap<String, f32> = HashMap::new();
        let mut result_map: HashMap<String, SearchResult> = HashMap::new();

        for ranked in all_ranked {
            for (rank, result) in ranked.into_iter().enumerate() {
                let rrf_score = 1.0 / (60.0 + rank as f32);
                *score_map.entry(result.id.clone()).or_default() += rrf_score;
                result_map.entry(result.id.clone()).or_insert(result);
            }
        }

        // Apply accumulated RRF scores and sort
        let mut merged: Vec<SearchResult> = result_map
            .into_values()
            .map(|mut r| {
                r.score = *score_map.get(&r.id).unwrap_or(&0.0);
                r
            })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        merged.truncate(limit);
        Ok(merged)
    }

    /// Augment search results with knowledge graph observations via RRF merge.
    /// Graph observations boost scores of related memories but are stripped from
    /// final output by search_memory (KG is internal scaffolding, not user-facing).
    /// Returns the merged + re-sorted results. If no entities exist, returns input unchanged.
    pub async fn augment_with_graph(
        &self,
        query: &str,
        results: Vec<SearchResult>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, OriginError> {
        let entity_hits = match self.search_entities_by_vector(query, 5).await {
            Ok(hits) if !hits.is_empty() => hits,
            _ => return Ok(results),
        };

        let entity_ids: Vec<String> = entity_hits.iter().map(|r| r.entity.id.clone()).collect();
        let graph_results = match self.get_observations_for_entities(&entity_ids, limit).await {
            Ok(r) if !r.is_empty() => r,
            _ => return Ok(results),
        };

        // Build RRF scores from existing results
        // Preserve existing multiplied scores (confidence × recency × quality etc.)
        // Graph observations start at 0.0 and get only their RRF contribution
        let mut score_map: HashMap<String, f32> =
            results.iter().map(|r| (r.id.clone(), r.score)).collect();
        let mut result_map: HashMap<String, SearchResult> =
            results.into_iter().map(|r| (r.id.clone(), r)).collect();

        // Merge graph observations as third RRF signal
        for (rank, result) in graph_results.into_iter().enumerate() {
            let rrf_score = 1.0 / (60.0 + rank as f32);
            *score_map.entry(result.id.clone()).or_default() += rrf_score;
            result_map.entry(result.id.clone()).or_insert(result);
        }

        let mut merged: Vec<SearchResult> = result_map
            .into_values()
            .map(|mut r| {
                r.score = *score_map.get(&r.id).unwrap_or(&0.0);
                r
            })
            .collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source_id.cmp(&b.source_id))
        });
        Ok(merged)
    }

    /// Vector-only search bypassing hybrid FTS+RRF pipeline — used as NaiveRag baseline.
    /// Embeds the query, queries the DiskANN index, and returns results scored by
    /// cosine similarity (1.0 - distance). No FTS, no RRF, no scoring adjustments.
    #[allow(dead_code)] // Used by eval module (token_efficiency.rs)
    pub(crate) async fn naive_vector_search(
        &self,
        query: &str,
        limit: usize,
        domain: Option<&str>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);
        // Fetch more candidates when domain filtering (some will be filtered out)
        let fetch_limit = if domain.is_some() {
            (limit * 3) as i64
        } else {
            limit as i64
        };

        let conn = self.conn.lock().await;

        let (sql, params): (String, Vec<libsql::Value>) = if let Some(d) = domain {
            (
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        vector_distance_cos(c.embedding, vector32(?1))
                 FROM vector_top_k('memories_vec_idx', vector32(?1), ?2) AS vt
                 JOIN memories c ON c.rowid = vt.id
                 WHERE c.pending_revision = 0 AND c.domain = ?3"
                    .to_string(),
                vec![
                    libsql::Value::Text(vec_str.clone()),
                    libsql::Value::Integer(fetch_limit),
                    libsql::Value::Text(d.to_string()),
                ],
            )
        } else {
            (
                "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                        c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                        c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                        c.confidence, c.confirmed, c.stability, c.supersedes,
                        c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                        c.structured_fields, c.retrieval_cue, c.source_text,
                        vector_distance_cos(c.embedding, vector32(?1))
                 FROM vector_top_k('memories_vec_idx', vector32(?1), ?2) AS vt
                 JOIN memories c ON c.rowid = vt.id
                 WHERE c.pending_revision = 0"
                    .to_string(),
                vec![
                    libsql::Value::Text(vec_str.clone()),
                    libsql::Value::Integer(fetch_limit),
                ],
            )
        };

        let mut results = Vec::new();
        match conn.query(&sql, params).await {
            Ok(mut rows) => {
                while let Ok(Some(row)) = rows.next().await {
                    let distance: f64 = row.get(28).unwrap_or(1.0);
                    let score = (1.0 - distance).max(0.0) as f32;
                    if let Ok(result) = Self::row_to_search_result(&row, score) {
                        results.push(result);
                    }
                }
            }
            Err(e) => {
                log::warn!("[naive_vector_search] vector index query failed: {}", e);
            }
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    /// FTS-only search (no vector, no RRF) — used as ablation baseline.
    /// Runs BM25 FTS5 matching with AND-then-OR fallback; score = negated rank.
    /// Eval-only: not used in production search paths.
    #[allow(dead_code)] // Used by eval module (token_efficiency.rs)
    pub(crate) async fn fts_only_search(
        &self,
        query: &str,
        limit: usize,
        domain: Option<&str>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        let fetch_limit = if domain.is_some() {
            (limit * 3) as i64
        } else {
            limit as i64
        };

        let conn = self.conn.lock().await;

        let (domain_clause, domain_param): (String, Option<libsql::Value>) = if let Some(d) = domain
        {
            (
                "AND c.domain = ?3".to_string(),
                Some(libsql::Value::Text(d.to_string())),
            )
        } else {
            (String::new(), None)
        };

        let fts_sql = format!(
            "SELECT c.id, c.content, c.source, c.source_id, c.title, c.summary, c.url,
                    c.chunk_index, c.last_modified, c.chunk_type, c.language, c.byte_start,
                    c.byte_end, c.semantic_unit, c.memory_type, c.domain, c.source_agent,
                    c.confidence, c.confirmed, c.stability, c.supersedes,
                    c.entity_id, c.quality, c.is_recap, c.supersede_mode,
                    c.structured_fields, c.retrieval_cue, c.source_text,
                    fts.rank
             FROM memories_fts fts
             JOIN memories c ON fts.rowid = c.rowid
             WHERE memories_fts MATCH ?1
               AND c.pending_revision = 0 {}
             ORDER BY fts.rank
             LIMIT ?2",
            domain_clause
        );

        // AND match first, fall back to OR for multi-word queries
        let fts_queries = vec![query.to_string(), Self::fts_or_query(query)];
        let mut results: Vec<SearchResult> = Vec::new();

        for fts_query in &fts_queries {
            let mut params: Vec<libsql::Value> = vec![
                libsql::Value::Text(fts_query.clone()),
                libsql::Value::Integer(fetch_limit),
            ];
            if let Some(ref dp) = domain_param {
                params.push(dp.clone());
            }

            match conn.query(&fts_sql, params).await {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        // FTS5 rank is negative BM25; negate so higher = better
                        let rank: f64 = row.get(28).unwrap_or(0.0);
                        let score = (-rank) as f32;
                        if let Ok(result) = Self::row_to_search_result(&row, score) {
                            results.push(result);
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[fts_only_search] FTS search failed: {}", e);
                }
            }
            if !results.is_empty() {
                break;
            }
        }

        // Dedup by id (AND and OR passes shouldn't overlap, but guard anyway)
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.id.clone()));

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Vector + FTS merged by max score per document (no RRF) — ablation baseline.
    /// Merges both signal lists by taking the max score for any document appearing
    /// in both, then sorts descending and truncates to limit.
    /// Eval-only: not used in production search paths.
    #[allow(dead_code)] // Used by eval module (token_efficiency.rs)
    pub(crate) async fn vector_plus_fts_search(
        &self,
        query: &str,
        limit: usize,
        domain: Option<&str>,
    ) -> Result<Vec<SearchResult>, OriginError> {
        // Run both searches independently; score normalization is done after merge.
        let vec_results = self.naive_vector_search(query, limit, domain).await?;
        let fts_results = self.fts_only_search(query, limit, domain).await?;

        // Normalise vector scores (already in [0,1] as cosine similarity).
        // Normalise FTS scores to [0,1] range so both signals are comparable.
        let fts_max = fts_results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let fts_norm = if fts_max > 0.0 { fts_max } else { 1.0 };

        let mut merged: std::collections::HashMap<String, SearchResult> =
            std::collections::HashMap::new();

        for r in vec_results {
            // Vector scores are already cosine similarity in [0,1]
            let prev_score = merged
                .get(&r.id)
                .map(|e| e.score)
                .unwrap_or(f32::NEG_INFINITY);
            if r.score > prev_score {
                merged.insert(r.id.clone(), r);
            }
        }

        for mut r in fts_results {
            let normalised = r.score / fts_norm;
            let prev_score = merged
                .get(&r.id)
                .map(|e| e.score)
                .unwrap_or(f32::NEG_INFINITY);
            if normalised > prev_score {
                r.score = normalised;
                merged.insert(r.id.clone(), r);
            }
        }

        let mut results: Vec<SearchResult> = merged.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source_id.cmp(&b.source_id))
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Search entities by vector similarity. Returns EntitySearchResult with full Entity data.
    /// Tries DiskANN index first, falls back to brute-force cosine similarity.
    pub async fn search_entities_by_vector(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<EntitySearchResult>, OriginError> {
        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);

        let conn = self.conn.lock().await;

        // Try DiskANN index first
        let sql = "SELECT e.id, e.name, e.entity_type, e.domain, e.source_agent, e.confidence, e.confirmed, e.created_at, e.updated_at, vector_distance_cos(e.embedding, vector32(?1))
                   FROM vector_top_k('entities_vec_idx', vector32(?1), ?2) AS vt
                   JOIN entities e ON e.rowid = vt.id";

        let mut results = Vec::new();
        match conn
            .query(sql, libsql::params![vec_str.clone(), limit as i64])
            .await
        {
            Ok(mut rows) => {
                while let Ok(Some(row)) = rows.next().await {
                    let entity = Entity {
                        id: row.get::<String>(0).unwrap_or_default(),
                        name: row.get::<String>(1).unwrap_or_default(),
                        entity_type: row.get::<String>(2).unwrap_or_default(),
                        domain: row.get::<Option<String>>(3).unwrap_or(None),
                        source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                        confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                        confirmed: row.get::<i64>(6).unwrap_or(0) != 0,
                        created_at: row.get::<i64>(7).unwrap_or(0),
                        updated_at: row.get::<i64>(8).unwrap_or(0),
                    };
                    let distance: f64 = row.get::<f64>(9).unwrap_or(1.0);
                    results.push(EntitySearchResult {
                        entity,
                        distance: distance as f32,
                    });
                }
            }
            Err(e) => {
                log::debug!(
                    "[memory_db] entity DiskANN search failed, trying brute-force: {}",
                    e
                );
            }
        }

        // Fallback: brute-force cosine distance when DiskANN index is unavailable or empty
        if results.is_empty() {
            let fallback_sql =
                "SELECT id, name, entity_type, domain, source_agent, confidence, confirmed, created_at, updated_at, vector_distance_cos(embedding, vector32(?1)) as distance
                 FROM entities
                 WHERE embedding IS NOT NULL
                 ORDER BY distance ASC
                 LIMIT ?2";
            match conn
                .query(fallback_sql, libsql::params![vec_str, limit as i64])
                .await
            {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        let entity = Entity {
                            id: row.get::<String>(0).unwrap_or_default(),
                            name: row.get::<String>(1).unwrap_or_default(),
                            entity_type: row.get::<String>(2).unwrap_or_default(),
                            domain: row.get::<Option<String>>(3).unwrap_or(None),
                            source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                            confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                            confirmed: row.get::<i64>(6).unwrap_or(0) != 0,
                            created_at: row.get::<i64>(7).unwrap_or(0),
                            updated_at: row.get::<i64>(8).unwrap_or(0),
                        };
                        let distance: f64 = row.get::<f64>(9).unwrap_or(1.0);
                        results.push(EntitySearchResult {
                            entity,
                            distance: distance as f32,
                        });
                    }
                }
                Err(e) => {
                    log::warn!("[memory_db] entity brute-force search also failed: {}", e);
                }
            }
        }

        Ok(results)
    }

    /// Get observations for a list of entity IDs, returned as SearchResult items
    /// with `source = "knowledge_graph"` so they can be merged into hybrid search results.
    pub async fn get_observations_for_entities(
        &self,
        entity_ids: &[String],
        limit: usize,
    ) -> Result<Vec<SearchResult>, OriginError> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().await;

        let placeholders: Vec<String> = (1..=entity_ids.len()).map(|i| format!("?{}", i)).collect();
        let limit_param = entity_ids.len() + 1;
        let sql = format!(
            "SELECT o.id, o.content, e.name, o.created_at, o.source_agent, o.confidence
             FROM observations o
             JOIN entities e ON o.entity_id = e.id
             WHERE o.entity_id IN ({})
             ORDER BY o.created_at DESC
             LIMIT ?{}",
            placeholders.join(","),
            limit_param
        );

        let mut params: Vec<libsql::Value> = entity_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        params.push(libsql::Value::Integer(limit as i64));

        let mut results = Vec::new();
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_observations_for_entities: {}", e)))?;

        while let Ok(Some(row)) = rows.next().await {
            let id: String = row.get(0).unwrap_or_default();
            let content: String = row.get(1).unwrap_or_default();
            let entity_name: String = row.get(2).unwrap_or_default();
            let created_at: i64 = row.get(3).unwrap_or(0);
            let source_agent: Option<String> = row.get::<Option<String>>(4).unwrap_or(None);
            let confidence: Option<f64> = row.get::<Option<f64>>(5).unwrap_or(None);
            let source_id = format!("obs_{}", id);

            results.push(SearchResult {
                id,
                content,
                source: "knowledge_graph".to_string(),
                source_id,
                title: entity_name.clone(),
                url: None,
                chunk_index: 0,
                last_modified: created_at,
                score: 0.0,
                chunk_type: None,
                language: None,
                semantic_unit: None,
                memory_type: None,
                domain: None,
                source_agent,
                confidence: confidence.map(|v| v as f32),
                confirmed: None,
                stability: None,
                supersedes: None,
                summary: None,
                entity_id: None,
                entity_name: Some(entity_name),
                quality: None,
                is_archived: false,
                is_recap: false,
                structured_fields: None,
                retrieval_cue: None,
                source_text: None,
                raw_score: 0.0,
            });
        }

        Ok(results)
    }

    /// Insert an eval signal (fire-and-forget, INSERT OR IGNORE for dedup).
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_eval_signal(
        &self,
        id: &str,
        signal_type: &str,
        memory_id: &str,
        query_context: Option<&str>,
        rank_position: Option<i32>,
        created_at: i64,
        metadata: Option<&str>,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR IGNORE INTO eval_signals (id, signal_type, memory_id, query_context, rank_position, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            libsql::params![id, signal_type, memory_id, query_context, rank_position, created_at, metadata],
        ).await.map_err(|e| OriginError::VectorDb(format!("insert eval_signal: {}", e)))?;
        Ok(())
    }

    /// Delete all rows for a given source + source_id.
    pub async fn delete_by_source_id(
        &self,
        source: &str,
        source_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM memories WHERE source = ?1 AND source_id = ?2",
            libsql::params![source.to_string(), source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_by_source_id: {}", e)))?;
        // Clean up orphaned enrichment_steps (no FK cascade)
        conn.execute(
            "DELETE FROM enrichment_steps WHERE source_id = ?1",
            libsql::params![source_id.to_string()],
        )
        .await
        .ok();
        Ok(())
    }

    /// Update summary for all rows of a document.
    pub async fn update_document_summary(
        &self,
        source_id: &str,
        _source_type: &str,
        summary: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET summary = ?1 WHERE source_id = ?2",
            libsql::params![summary.to_string(), source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_summary: {}", e)))?;
        Ok(())
    }

    /// Update a single column by source_id (whitelisted columns only).
    pub async fn update_column_by_source_id(
        &self,
        source: &str,
        source_id: &str,
        column: &str,
        value: &str,
    ) -> Result<(), OriginError> {
        // Whitelist allowed column names to prevent SQL injection
        const ALLOWED: &[&str] = &[
            "content",
            "title",
            "summary",
            "chunk_type",
            "language",
            "semantic_unit",
            "memory_type",
            "domain",
            "source_agent",
            "supersedes",
            "confirmed",
        ];
        if !ALLOWED.contains(&column) {
            return Err(OriginError::VectorDb(format!(
                "column '{}' not allowed for update",
                column
            )));
        }

        let conn = self.conn.lock().await;
        let sql = format!(
            "UPDATE memories SET {} = ?1 WHERE source_id = ?2 AND source = ?3",
            column
        );
        conn.execute(
            &sql,
            libsql::params![value.to_string(), source_id.to_string(), source.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_column: {}", e)))?;
        Ok(())
    }

    /// Update last_modified timestamp for all rows of a source_id.
    pub async fn update_timestamp_by_source_id(
        &self,
        source_id: &str,
        new_timestamp: i64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET last_modified = ?1 WHERE source_id = ?2",
            libsql::params![new_timestamp, source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_timestamp: {}", e)))?;
        Ok(())
    }

    /// Total number of memories.
    pub async fn count(&self) -> Result<u64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM memories", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("count: {}", e)))?;

        if let Ok(Some(row)) = rows.next().await {
            let count: i64 = row.get(0).unwrap_or(0);
            Ok(count as u64)
        } else {
            Ok(0)
        }
    }

    /// List all indexed files (grouped by source_id).
    pub async fn list_indexed_files(&self) -> Result<Vec<IndexedFileInfo>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, MAX(title) as title, MAX(source) as source,
                        MAX(url) as url, COUNT(*) as chunk_count,
                        MAX(last_modified) as last_modified, MAX(summary) as summary,
                        MAX(memory_type), MAX(domain), MAX(source_agent),
                        MAX(CAST(confidence AS REAL)), MAX(confirmed), MAX(pinned)
                 FROM memories
                 GROUP BY source_id
                 ORDER BY MAX(last_modified) DESC",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_indexed: {}", e)))?;

        let mut files = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            files.push(IndexedFileInfo {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                source: row.get::<String>(2).unwrap_or_default(),
                url: row.get::<Option<String>>(3).unwrap_or(None),
                chunk_count: row.get::<i64>(4).unwrap_or(0) as u64,
                last_modified: row.get::<i64>(5).unwrap_or(0),
                summary: row.get::<Option<String>>(6).unwrap_or(None),
                processing: false,
                memory_type: row.get::<Option<String>>(7).unwrap_or(None),
                domain: row.get::<Option<String>>(8).unwrap_or(None),
                source_agent: row.get::<Option<String>>(9).unwrap_or(None),
                confidence: row.get::<Option<f64>>(10).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<Option<i64>>(11).unwrap_or(None).map(|v| v != 0),
                stability: None, // not fetched in list_indexed_files aggregate query
                pinned: row.get::<i64>(12).unwrap_or(0) != 0,
            });
        }
        Ok(files)
    }

    /// Get all memories for a document.
    pub async fn get_memories_by_source_id(
        &self,
        source: &str,
        source_id: &str,
    ) -> Result<Vec<MemoryDetail>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, content, title, source_id, chunk_index, chunk_type, language,
                        semantic_unit, byte_start, byte_end, summary
                 FROM memories
                 WHERE source = ?1 AND source_id = ?2
                 ORDER BY chunk_index ASC",
                libsql::params![source.to_string(), source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memories: {}", e)))?;

        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            results.push(MemoryDetail {
                id: row.get::<String>(0).unwrap_or_default(),
                content: row.get::<String>(1).unwrap_or_default(),
                title: row.get::<String>(2).unwrap_or_default(),
                source_id: row.get::<String>(3).unwrap_or_default(),
                chunk_index: row.get::<i32>(4).unwrap_or(0),
                chunk_type: row.get::<Option<String>>(5).unwrap_or(None),
                language: row.get::<Option<String>>(6).unwrap_or(None),
                semantic_unit: row.get::<Option<String>>(7).unwrap_or(None),
                byte_start: row.get::<Option<i64>>(8).unwrap_or(None),
                byte_end: row.get::<Option<i64>>(9).unwrap_or(None),
                summary: row.get::<Option<String>>(10).unwrap_or(None),
            });
        }
        Ok(results)
    }

    /// Delete memories within a time range. Returns the number deleted.
    pub async fn delete_by_time_range(&self, start: i64, end: i64) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute(
                "DELETE FROM memories WHERE last_modified >= ?1 AND last_modified <= ?2",
                libsql::params![start, end],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_time_range: {}", e)))?;
        Ok(deleted as usize)
    }

    /// Bulk delete by (source, source_id) pairs.
    pub async fn delete_bulk(&self, items: &[(String, String)]) -> Result<(), OriginError> {
        if items.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().await;
        for (source, source_id) in items {
            conn.execute(
                "DELETE FROM memories WHERE source = ?1 AND source_id = ?2",
                libsql::params![source.clone(), source_id.clone()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_bulk: {}", e)))?;
        }
        Ok(())
    }

    /// Get details for a specific memory.
    pub async fn get_memory_details(
        &self,
        source: &str,
        source_id: &str,
        chunk_index: i32,
    ) -> Result<Option<MemoryDetail>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, content, title, source_id, chunk_index, chunk_type, language,
                        semantic_unit, byte_start, byte_end, summary
                 FROM memories
                 WHERE source = ?1 AND source_id = ?2 AND chunk_index = ?3",
                libsql::params![
                    source.to_string(),
                    source_id.to_string(),
                    chunk_index as i64
                ],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_details: {}", e)))?;

        if let Ok(Some(row)) = rows.next().await {
            Ok(Some(MemoryDetail {
                id: row.get::<String>(0).unwrap_or_default(),
                content: row.get::<String>(1).unwrap_or_default(),
                title: row.get::<String>(2).unwrap_or_default(),
                source_id: row.get::<String>(3).unwrap_or_default(),
                chunk_index: row.get::<i32>(4).unwrap_or(0),
                chunk_type: row.get::<Option<String>>(5).unwrap_or(None),
                language: row.get::<Option<String>>(6).unwrap_or(None),
                semantic_unit: row.get::<Option<String>>(7).unwrap_or(None),
                byte_start: row.get::<Option<i64>>(8).unwrap_or(None),
                byte_end: row.get::<Option<i64>>(9).unwrap_or(None),
                summary: row.get::<Option<String>>(10).unwrap_or(None),
            }))
        } else {
            Ok(None)
        }
    }

    /// Update a single memory's content (and re-embed).
    pub async fn update_memory(&self, id: &str, new_content: &str) -> Result<(), OriginError> {
        let embedding = self.get_or_compute_embedding(new_content)?;
        let vec_str = Self::vec_to_sql(&embedding);

        let conn = self.conn.lock().await;
        // Match on source_id (the external identifier passed from API routes)
        // and chunk_index = 0 (primary chunk holds the canonical content).
        conn.execute(
            "UPDATE memories SET content = ?1, embedding = vector32(?2) WHERE source_id = ?3 AND chunk_index = 0",
            libsql::params![new_content.to_string(), vec_str, id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_memory: {}", e)))?;
        Ok(())
    }

    /// Replace entire document content: re-chunk, re-embed, re-insert.
    pub async fn update_document_content(
        &self,
        source_id: &str,
        new_content: &str,
    ) -> Result<(), OriginError> {
        // Get existing doc metadata from current memories
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source, title, url, summary, memory_type, domain, source_agent,
                        confidence, confirmed, supersedes, last_modified
                 FROM memories WHERE source_id = ?1 LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("update_doc_content: {}", e)))?;

        let meta = if let Ok(Some(row)) = rows.next().await {
            Some((
                row.get::<String>(0).unwrap_or_default(),
                row.get::<String>(1).unwrap_or_default(),
                row.get::<Option<String>>(2).unwrap_or(None),
                row.get::<Option<String>>(3).unwrap_or(None),
                row.get::<Option<String>>(4).unwrap_or(None),
                row.get::<Option<String>>(5).unwrap_or(None),
                row.get::<Option<String>>(6).unwrap_or(None),
                row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                row.get::<Option<i64>>(8).unwrap_or(None).map(|v| v != 0),
                row.get::<Option<String>>(9).unwrap_or(None),
                row.get::<i64>(10).unwrap_or(0),
            ))
        } else {
            None
        };
        drop(rows);
        drop(conn);

        if let Some((
            source,
            title,
            url,
            summary,
            memory_type,
            domain,
            source_agent,
            confidence,
            confirmed,
            supersedes,
            last_modified,
        )) = meta
        {
            let doc = RawDocument {
                source,
                source_id: source_id.to_string(),
                title,
                summary,
                content: new_content.to_string(),
                url,
                last_modified,
                metadata: HashMap::new(),
                memory_type,
                domain,
                source_agent,
                confidence,
                confirmed,
                supersedes,
                pending_revision: false,
                ..Default::default()
            };
            self.upsert_documents(vec![doc]).await?;
        }

        Ok(())
    }

    /// Count memories per source.
    pub async fn count_by_source(&self) -> Result<HashMap<String, u64>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT source, COUNT(*) FROM memories GROUP BY source", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_by_source: {}", e)))?;

        let mut map = HashMap::new();
        while let Ok(Some(row)) = rows.next().await {
            let source: String = row.get(0).unwrap_or_default();
            let count: i64 = row.get(1).unwrap_or(0);
            map.insert(source, count as u64);
        }
        Ok(map)
    }

    /// List memories with optional filters, returning rich MemoryItem structs for the UI.
    pub async fn list_memories(
        &self,
        domain: Option<&str>,
        memory_type: Option<&str>,
        confirmed: Option<bool>,
        pinned: Option<bool>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, OriginError> {
        let conn = self.conn.lock().await;

        let mut sql = String::from(
            "SELECT source_id, title,
                    GROUP_CONCAT(content, '\n') as content,
                    MAX(summary) as summary,
                    MAX(memory_type) as memory_type,
                    MAX(domain) as domain,
                    MAX(source_agent) as source_agent,
                    MAX(confidence) as confidence,
                    MAX(confirmed) as confirmed,
                    MAX(stability) as stability,
                    MAX(pinned) as pinned,
                    MAX(supersedes) as supersedes,
                    MAX(last_modified) as last_modified,
                    COUNT(*) as chunk_count,
                    MAX(entity_id) as entity_id,
                    MAX(quality) as quality,
                    MAX(is_recap) as is_recap,
                    (SELECT CASE
                        WHEN COUNT(es.source_id) = 0 THEN 'raw'
                        WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                        WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                        ELSE 'enrichment_partial'
                    END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status,
                    MAX(supersede_mode) as supersede_mode,
                    MAX(structured_fields) as structured_fields,
                    MAX(retrieval_cue) as retrieval_cue,
                    SUM(access_count) as access_count,
                    MAX(source_text) as source_text
             FROM memories
             WHERE source = 'memory'
               AND pending_revision = 0
               AND source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)"
        );
        let mut params: Vec<libsql::Value> = Vec::new();

        if let Some(d) = domain {
            if d == "uncategorized" {
                sql.push_str(" AND domain IS NULL");
            } else {
                params.push(d.into());
                sql.push_str(&format!(" AND domain = ?{}", params.len()));
            }
        }
        if let Some(mt) = memory_type {
            params.push(mt.into());
            sql.push_str(&format!(" AND memory_type = ?{}", params.len()));
        }
        if let Some(c) = confirmed {
            params.push(if c { 1i64.into() } else { 0i64.into() });
            sql.push_str(&format!(" AND confirmed = ?{}", params.len()));
        }
        if let Some(p) = pinned {
            params.push(if p { 1i64.into() } else { 0i64.into() });
            sql.push_str(&format!(" AND pinned = ?{}", params.len()));
        }

        sql.push_str(" GROUP BY source_id ORDER BY last_modified DESC");
        params.push((limit as i64).into());
        sql.push_str(&format!(" LIMIT ?{}", params.len()));

        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_memories: {}", e)))?;

        let mut items = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            items.push(MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: row.get::<u64>(13).unwrap_or(0),
                entity_id: row.get::<Option<String>>(14).unwrap_or(None),
                quality: row.get::<Option<String>>(15).unwrap_or(None),
                is_recap: row.get::<i64>(16).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(17).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(18).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(19).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(20).unwrap_or(None),
                access_count: row.get::<u64>(21).unwrap_or(0),
                source_text: row.get::<Option<String>>(22).unwrap_or(None),
                version: 1,
                changelog: None,
            });
        }
        Ok(items)
    }

    /// Fetch a single memory by source_id, aggregating its memories.
    pub async fn get_memory_detail(
        &self,
        source_id: &str,
    ) -> Result<Option<MemoryItem>, OriginError> {
        let conn = self.conn.lock().await;
        let sql = "SELECT source_id, title,
                    GROUP_CONCAT(content, '\n') as content,
                    MAX(summary) as summary,
                    MAX(memory_type) as memory_type,
                    MAX(domain) as domain,
                    MAX(source_agent) as source_agent,
                    MAX(confidence) as confidence,
                    MAX(confirmed) as confirmed,
                    MAX(stability) as stability,
                    MAX(pinned) as pinned,
                    MAX(supersedes) as supersedes,
                    MAX(last_modified) as last_modified,
                    COUNT(*) as chunk_count,
                    MAX(entity_id) as entity_id,
                    MAX(quality) as quality,
                    MAX(is_recap) as is_recap,
                    (SELECT CASE
                        WHEN COUNT(es.source_id) = 0 THEN 'raw'
                        WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                        WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                        ELSE 'enrichment_partial'
                    END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status,
                    MAX(supersede_mode) as supersede_mode,
                    MAX(structured_fields) as structured_fields,
                    MAX(retrieval_cue) as retrieval_cue,
                    SUM(access_count) as access_count,
                    MAX(source_text) as source_text
             FROM memories
             WHERE pending_revision = 0
               AND source_id = ?1
             GROUP BY source_id";

        let mut rows = conn
            .query(sql, libsql::params![source_id])
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_detail: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(Some(MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: row.get::<u64>(13).unwrap_or(0),
                entity_id: row.get::<Option<String>>(14).unwrap_or(None),
                quality: row.get::<Option<String>>(15).unwrap_or(None),
                is_recap: row.get::<i64>(16).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(17).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(18).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(19).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(20).unwrap_or(None),
                access_count: row.get::<u64>(21).unwrap_or(0),
                source_text: row.get::<Option<String>>(22).unwrap_or(None),
                version: 1,
                changelog: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Fetch multiple memories by source_id in a single query.
    /// Returns only the source_ids that exist; missing ids are silently omitted.
    /// Preserves input order.
    pub async fn get_memories_by_source_ids(
        &self,
        source_ids: &[String],
    ) -> Result<Vec<MemoryItem>, OriginError> {
        if source_ids.is_empty() {
            return Ok(vec![]);
        }
        let placeholders = source_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT source_id, title,
                GROUP_CONCAT(content, '\n') as content,
                MAX(summary) as summary,
                MAX(memory_type) as memory_type,
                MAX(domain) as domain,
                MAX(source_agent) as source_agent,
                MAX(confidence) as confidence,
                MAX(confirmed) as confirmed,
                MAX(stability) as stability,
                MAX(pinned) as pinned,
                MAX(supersedes) as supersedes,
                MAX(last_modified) as last_modified,
                COUNT(*) as chunk_count,
                MAX(entity_id) as entity_id,
                MAX(quality) as quality,
                MAX(is_recap) as is_recap,
                (SELECT CASE
                    WHEN COUNT(es.source_id) = 0 THEN 'raw'
                    WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                    WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                    ELSE 'enrichment_partial'
                END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status,
                MAX(supersede_mode) as supersede_mode,
                MAX(structured_fields) as structured_fields,
                MAX(retrieval_cue) as retrieval_cue,
                SUM(access_count) as access_count,
                MAX(source_text) as source_text,
                MAX(version) as version,
                MAX(changelog) as changelog
             FROM memories
             WHERE pending_revision = 0
               AND source_id IN ({placeholders})
             GROUP BY source_id"
        );
        let conn = self.conn.lock().await;
        let params: Vec<libsql::Value> = source_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memories_by_source_ids: {e}")))?;

        // Build a map keyed by source_id, then re-order to match input order.
        let mut map: std::collections::HashMap<String, MemoryItem> =
            std::collections::HashMap::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let item = MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: row.get::<u64>(13).unwrap_or(0),
                entity_id: row.get::<Option<String>>(14).unwrap_or(None),
                quality: row.get::<Option<String>>(15).unwrap_or(None),
                is_recap: row.get::<i64>(16).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(17).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(18).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(19).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(20).unwrap_or(None),
                access_count: row.get::<u64>(21).unwrap_or(0),
                source_text: row.get::<Option<String>>(22).unwrap_or(None),
                version: row.get::<i64>(23).unwrap_or(1),
                changelog: row.get::<Option<String>>(24).unwrap_or(None),
            };
            map.insert(item.source_id.clone(), item);
        }
        // Return in input order, skipping missing ids.
        let items = source_ids.iter().filter_map(|id| map.remove(id)).collect();
        Ok(items)
    }

    /// Load confirmed memories of a specific type, excluding superseded ones.
    /// Optionally filter by domain when `domain_filter` is `Some`.
    pub async fn load_memories_by_type(
        &self,
        memory_type: &str,
        limit: usize,
        domain_filter: Option<&str>,
    ) -> Result<Vec<MemoryItem>, OriginError> {
        let conn = self.conn.lock().await;
        let supersedes_exclusion = "AND pending_revision = 0 AND source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)";

        let domain_clause = if domain_filter.is_some() {
            "AND domain = ?3"
        } else {
            ""
        };

        let sql = format!(
            "SELECT source_id, MAX(title) as title,
                    GROUP_CONCAT(content, '\n') as content,
                    MAX(summary) as summary,
                    MAX(memory_type) as memory_type,
                    MAX(domain) as domain,
                    MAX(source_agent) as source_agent,
                    MAX(confidence) as confidence,
                    MAX(confirmed) as confirmed,
                    MAX(stability) as stability,
                    MAX(pinned) as pinned,
                    MAX(supersedes) as supersedes,
                    MAX(last_modified) as last_modified,
                    COUNT(*) as chunk_count,
                    MAX(entity_id) as entity_id,
                    MAX(quality) as quality,
                    MAX(is_recap) as is_recap,
                    (SELECT CASE
                        WHEN COUNT(es.source_id) = 0 THEN 'raw'
                        WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                        WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                        ELSE 'enrichment_partial'
                    END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status,
                    MAX(supersede_mode) as supersede_mode,
                    MAX(structured_fields) as structured_fields,
                    MAX(retrieval_cue) as retrieval_cue,
                    SUM(access_count) as access_count,
                    MAX(source_text) as source_text
             FROM memories
             WHERE source = 'memory' AND memory_type = ?1 AND confirmed != 0
             {} {}
             GROUP BY source_id
             ORDER BY last_modified DESC
             LIMIT ?2",
            supersedes_exclusion, domain_clause
        );

        let mut rows = if let Some(domain) = domain_filter {
            conn.query(
                &sql,
                libsql::params![memory_type.to_string(), limit as i64, domain.to_string()],
            )
            .await
        } else {
            conn.query(&sql, libsql::params![memory_type.to_string(), limit as i64])
                .await
        }
        .map_err(|e| OriginError::VectorDb(format!("load_memories_by_type: {}", e)))?;

        let mut items = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            items.push(MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: row.get::<u64>(13).unwrap_or(0),
                entity_id: row.get::<Option<String>>(14).unwrap_or(None),
                quality: row.get::<Option<String>>(15).unwrap_or(None),
                is_recap: row.get::<i64>(16).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(17).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(18).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(19).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(20).unwrap_or(None),
                access_count: row.get::<u64>(21).unwrap_or(0),
                source_text: row.get::<Option<String>>(22).unwrap_or(None),
                version: 1,
                changelog: None,
            });
        }
        Ok(items)
    }

    /// Load decision memories for the timeline view.
    /// Filters to `source = 'memory' AND memory_type = 'decision' AND chunk_index = 0`.
    /// Returns full `MemoryItem` rows with `structured_fields` for expanded view.
    pub async fn load_decisions(
        &self,
        domain: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryItem>, OriginError> {
        let conn = self.conn.lock().await;
        let supersedes_exclusion = "AND pending_revision = 0 AND source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)";

        let domain_clause = if domain.is_some() {
            "AND domain = ?2"
        } else {
            ""
        };

        let sql = format!(
            "SELECT source_id, MAX(title) as title,
                    GROUP_CONCAT(content, '\n') as content,
                    MAX(summary) as summary,
                    MAX(memory_type) as memory_type,
                    MAX(domain) as domain,
                    MAX(source_agent) as source_agent,
                    MAX(confidence) as confidence,
                    MAX(confirmed) as confirmed,
                    MAX(stability) as stability,
                    MAX(pinned) as pinned,
                    MAX(supersedes) as supersedes,
                    MAX(last_modified) as last_modified,
                    COUNT(*) as chunk_count,
                    MAX(entity_id) as entity_id,
                    MAX(quality) as quality,
                    MAX(is_recap) as is_recap,
                    (SELECT CASE
                        WHEN COUNT(es.source_id) = 0 THEN 'raw'
                        WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                        WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                        ELSE 'enrichment_partial'
                    END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status,
                    MAX(supersede_mode) as supersede_mode,
                    MAX(structured_fields) as structured_fields,
                    MAX(retrieval_cue) as retrieval_cue,
                    SUM(access_count) as access_count,
                    MAX(source_text) as source_text
             FROM memories
             WHERE source = 'memory' AND memory_type = 'decision' AND chunk_index = 0 AND confirmed != 0
             {} {}
             GROUP BY source_id
             ORDER BY last_modified DESC
             LIMIT ?1",
            supersedes_exclusion, domain_clause
        );

        let mut rows = if let Some(d) = domain {
            conn.query(&sql, libsql::params![limit as i64, d.to_string()])
                .await
        } else {
            conn.query(&sql, libsql::params![limit as i64]).await
        }
        .map_err(|e| OriginError::VectorDb(format!("load_decisions: {}", e)))?;

        let mut items = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            items.push(MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: row.get::<u64>(13).unwrap_or(0),
                entity_id: row.get::<Option<String>>(14).unwrap_or(None),
                quality: row.get::<Option<String>>(15).unwrap_or(None),
                is_recap: row.get::<i64>(16).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(17).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(18).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(19).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(20).unwrap_or(None),
                access_count: row.get::<u64>(21).unwrap_or(0),
                source_text: row.get::<Option<String>>(22).unwrap_or(None),
                version: 1,
                changelog: None,
            });
        }
        Ok(items)
    }

    /// List distinct domains that have decision memories.
    pub async fn list_decision_domains(&self) -> Result<Vec<String>, OriginError> {
        let conn = self.conn.lock().await;
        let sql = "SELECT DISTINCT domain FROM memories WHERE source = 'memory' AND memory_type = 'decision' AND domain IS NOT NULL ORDER BY domain";
        let mut rows = conn
            .query(sql, libsql::params![])
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_decision_domains: {}", e)))?;

        let mut domains = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            if let Ok(d) = row.get::<String>(0) {
                domains.push(d);
            }
        }
        Ok(domains)
    }

    /// List documents matching structured filters, deduped by source_id.
    /// All filter values are passed as parameterized queries to prevent SQL injection.
    pub async fn list_filtered(
        &self,
        source: Option<&str>,
        memory_type: Option<&str>,
        domain: Option<&str>,
        limit: usize,
    ) -> Result<Vec<IndexedFileInfo>, OriginError> {
        let conn = self.conn.lock().await;

        let mut conditions = Vec::new();
        let mut params: Vec<libsql::Value> = Vec::new();
        let mut idx = 1;

        if let Some(s) = source {
            conditions.push(format!("source = ?{}", idx));
            params.push(s.to_string().into());
            idx += 1;
        }
        if let Some(mt) = memory_type {
            conditions.push(format!("memory_type = ?{}", idx));
            params.push(mt.to_string().into());
            idx += 1;
        }
        if let Some(d) = domain {
            conditions.push(format!("domain = ?{}", idx));
            params.push(d.to_string().into());
            idx += 1;
        }

        // Exclude superseded memories and pending revisions
        conditions.push("pending_revision = 0".to_string());
        conditions.push("source_id NOT IN (SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND pending_revision = 0 AND source = 'memory' GROUP BY supersedes)".to_string());

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        params.push((limit as i64).into());
        let limit_param = idx;

        let sql = format!(
            "SELECT source_id, MAX(title) as title, MAX(source) as source,
                    MAX(url) as url, COUNT(*) as chunk_count,
                    MAX(last_modified) as last_modified, MAX(summary) as summary,
                    MAX(memory_type), MAX(domain), MAX(source_agent),
                    MAX(CAST(confidence AS REAL)), MAX(confirmed), MAX(pinned)
             FROM memories
             {}
             GROUP BY source_id
             ORDER BY MAX(last_modified) DESC
             LIMIT ?{}",
            where_clause, limit_param
        );

        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_filtered: {}", e)))?;

        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            results.push(IndexedFileInfo {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                source: row.get::<String>(2).unwrap_or_default(),
                url: row.get::<Option<String>>(3).unwrap_or(None),
                chunk_count: row.get::<i64>(4).unwrap_or(0) as u64,
                last_modified: row.get::<i64>(5).unwrap_or(0),
                summary: row.get::<Option<String>>(6).unwrap_or(None),
                processing: false,
                memory_type: row.get::<Option<String>>(7).unwrap_or(None),
                domain: row.get::<Option<String>>(8).unwrap_or(None),
                source_agent: row.get::<Option<String>>(9).unwrap_or(None),
                confidence: row.get::<Option<f64>>(10).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<Option<i64>>(11).unwrap_or(None).map(|v| v != 0),
                stability: None, // not fetched in list_filtered aggregate query
                pinned: row.get::<i64>(12).unwrap_or(0) != 0,
            });
        }

        Ok(results)
    }

    // ===== Knowledge Graph Methods =====

    /// Create a new entity in the knowledge graph (simplified — no embedding, no source_agent/confidence).
    pub async fn create_entity(
        &self,
        name: &str,
        entity_type: &str,
        domain: Option<&str>,
    ) -> Result<String, OriginError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, domain, confirmed, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 0, ?5, ?5)",
            libsql::params![
                id.clone(),
                name.to_string(),
                entity_type.to_string(),
                domain.map(|d| d.to_string()),
                now
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("create_entity: {}", e)))?;

        Ok(id)
    }

    /// Store a new entity in the knowledge graph.
    pub async fn store_entity(
        &self,
        name: &str,
        entity_type: &str,
        domain: Option<&str>,
        source_agent: Option<&str>,
        confidence: Option<f32>,
    ) -> Result<String, OriginError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();

        let embedding = self.get_or_compute_embedding(name)?;
        let vec_str = Self::vec_to_sql(&embedding);

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, domain, source_agent, confidence,
                confirmed, created_at, updated_at, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7, ?7, vector32(?8))",
            libsql::params![
                id.clone(),
                name.to_string(),
                entity_type.to_string(),
                domain
                    .map(|s| libsql::Value::Text(s.to_string()))
                    .unwrap_or(libsql::Value::Null),
                source_agent
                    .map(|s| libsql::Value::Text(s.to_string()))
                    .unwrap_or(libsql::Value::Null),
                confidence
                    .map(|v| libsql::Value::Real(v as f64))
                    .unwrap_or(libsql::Value::Null),
                now,
                vec_str
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("store_entity: {}", e)))?;

        // Auto-create a self-alias for the entity name (lowercase).
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias_name, canonical_entity_id, created_at, source) VALUES (?1, ?2, unixepoch(), 'auto')",
            libsql::params![name.to_lowercase(), id.clone()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("store_entity alias: {}", e)))?;

        Ok(id)
    }

    /// Resolve an entity ID from an alias (case-insensitive).
    pub async fn resolve_entity_by_alias(&self, name: &str) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT canonical_entity_id FROM entity_aliases WHERE alias_name = ?1",
                libsql::params![name.to_lowercase()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("alias lookup: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("alias row: {}", e)))?
        {
            Ok(Some(row.get::<String>(0).unwrap()))
        } else {
            Ok(None)
        }
    }

    /// Add an alias entry for an entity.
    pub async fn add_entity_alias(
        &self,
        alias: &str,
        entity_id: &str,
        source: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias_name, canonical_entity_id, created_at, source) VALUES (?1, ?2, unixepoch(), ?3)",
            libsql::params![alias.to_lowercase(), entity_id.to_string(), source.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("add alias: {}", e)))?;
        Ok(())
    }

    /// Resolve a relation type string against the vocabulary (case-insensitive).
    /// Returns the canonical form if the input matches a canonical or an alias,
    /// otherwise returns the input unchanged (lowercased).
    pub async fn resolve_relation_type(&self, relation_type: &str) -> Result<String, OriginError> {
        let lower = relation_type.to_lowercase();
        let conn = self.conn.lock().await;

        // Check if input is already a canonical key (canonicals are lowercase).
        let mut rows = conn
            .query(
                "SELECT canonical FROM relation_type_vocabulary WHERE canonical = ?1",
                libsql::params![lower.clone()],
            )
            .await
            .map_err(|e| {
                OriginError::VectorDb(format!("resolve_relation_type canonical: {}", e))
            })?;
        if rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("resolve_relation_type row: {}", e)))?
            .is_some()
        {
            return Ok(lower);
        }
        drop(rows);

        // Scan aliases JSON arrays for a case-insensitive match.
        let mut rows = conn
            .query(
                "SELECT canonical, aliases FROM relation_type_vocabulary WHERE aliases IS NOT NULL",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("resolve_relation_type aliases: {}", e)))?;
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("resolve_relation_type alias row: {}", e)))?
        {
            let canonical = row.get::<String>(0).unwrap_or_default();
            let aliases_json = row.get::<String>(1).unwrap_or_default();
            if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(&aliases_json) {
                for v in &arr {
                    if let Some(alias) = v.as_str() {
                        if alias.to_lowercase() == lower {
                            return Ok(canonical);
                        }
                    }
                }
            }
        }

        // Not found — return lowercased input unchanged.
        Ok(lower)
    }

    /// Increment the usage count for a canonical relation type.
    pub async fn increment_relation_type_count(&self, canonical: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE relation_type_vocabulary SET count = count + 1 WHERE canonical = ?1",
            libsql::params![canonical.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("increment_relation_type_count: {}", e)))?;
        Ok(())
    }

    /// Search entities by exact name (case-insensitive).
    pub async fn search_entities_by_name(&self, name: &str) -> Result<Vec<Entity>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, name, entity_type, domain, source_agent, confidence, confirmed, created_at, updated_at
                 FROM entities WHERE LOWER(name) = LOWER(?1)",
                libsql::params![name.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("search_entities_by_name: {}", e)))?;

        let mut entities = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            entities.push(Entity {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                entity_type: row.get::<String>(2).unwrap_or_default(),
                domain: row.get::<Option<String>>(3).unwrap_or(None),
                source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(6).unwrap_or(0) != 0,
                created_at: row.get::<i64>(7).unwrap_or(0),
                updated_at: row.get::<i64>(8).unwrap_or(0),
            });
        }
        Ok(entities)
    }

    /// Refresh an entity's embedding by recomputing from the provided text.
    /// Also updates `embedding_updated_at` and `updated_at` timestamps.
    pub async fn refresh_entity_embedding(
        &self,
        entity_id: &str,
        text: &str,
    ) -> Result<(), OriginError> {
        let embeddings = self.generate_embeddings(&[text.to_string()])?;
        if embeddings.is_empty() {
            return Err(OriginError::VectorDb(
                "refresh_entity_embedding: empty embedding result".into(),
            ));
        }
        let vec_str = Self::vec_to_sql(&embeddings[0]);
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE entities SET embedding = vector32(?1), embedding_updated_at = ?2, updated_at = ?2 WHERE id = ?3",
            libsql::params![vec_str, now, entity_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("refresh_entity_embedding: {}", e)))?;
        Ok(())
    }

    /// Add an observation to an entity.
    pub async fn add_observation(
        &self,
        entity_id: &str,
        content: &str,
        source_agent: Option<&str>,
        confidence: Option<f32>,
    ) -> Result<String, OriginError> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, source_agent, confidence, confirmed, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
            libsql::params![
                id.clone(),
                entity_id.to_string(),
                content.to_string(),
                source_agent.map(|s| libsql::Value::Text(s.to_string())).unwrap_or(libsql::Value::Null),
                confidence.map(|v| libsql::Value::Real(v as f64)).unwrap_or(libsql::Value::Null),
                now
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("add_observation: {}", e)))?;

        // Update entity's updated_at
        conn.execute(
            "UPDATE entities SET updated_at = ?1 WHERE id = ?2",
            libsql::params![now, entity_id.to_string()],
        )
        .await
        .ok(); // Best effort

        Ok(id)
    }

    /// Create a relation between two entities.
    /// The relation type is normalized against the vocabulary via `resolve_relation_type`
    /// before insertion. On conflict (same from/to/type), updates confidence if higher
    /// and fills in explanation/source_memory_id if previously null.
    /// Returns the ID of the inserted or existing relation.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_relation(
        &self,
        from_entity: &str,
        to_entity: &str,
        relation_type: &str,
        source_agent: Option<&str>,
        confidence: Option<f64>,
        explanation: Option<&str>,
        source_memory_id: Option<&str>,
    ) -> Result<String, OriginError> {
        // Normalize relation type against vocabulary.
        // NOTE: resolve_relation_type acquires the conn lock, so we must not hold it here.
        let canonical = self.resolve_relation_type(relation_type).await?;

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();

        let conn = self.conn.lock().await;

        // Upsert: insert new or update existing if new confidence is higher.
        let affected = conn.execute(
            "INSERT INTO relations (id, from_entity, to_entity, relation_type, source_agent, confidence, explanation, source_memory_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(from_entity, to_entity, relation_type) DO UPDATE SET
                 confidence = CASE
                     WHEN EXCLUDED.confidence IS NOT NULL AND (confidence IS NULL OR EXCLUDED.confidence > confidence)
                     THEN EXCLUDED.confidence ELSE confidence END,
                 explanation = CASE
                     WHEN EXCLUDED.confidence IS NOT NULL AND (confidence IS NULL OR EXCLUDED.confidence > confidence)
                     THEN COALESCE(EXCLUDED.explanation, explanation) ELSE explanation END,
                 source_memory_id = COALESCE(EXCLUDED.source_memory_id, source_memory_id)",
            libsql::params![
                id.clone(),
                from_entity.to_string(),
                to_entity.to_string(),
                canonical.clone(),
                source_agent.map(|s| libsql::Value::Text(s.to_string())).unwrap_or(libsql::Value::Null),
                confidence.map(libsql::Value::Real).unwrap_or(libsql::Value::Null),
                explanation.map(|s| libsql::Value::Text(s.to_string())).unwrap_or(libsql::Value::Null),
                source_memory_id.map(|s| libsql::Value::Text(s.to_string())).unwrap_or(libsql::Value::Null),
                now
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("create_relation: {}", e)))?;

        // Only increment vocabulary count for genuinely new relations.
        if affected > 0 {
            // Check if this was an insert (the id we generated exists) vs an update.
            let mut rows = conn
                .query(
                    "SELECT id FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![from_entity.to_string(), to_entity.to_string(), canonical.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("create_relation check: {}", e)))?;
            let existing_id = match rows.next().await {
                Ok(Some(row)) => row.get::<String>(0).unwrap_or(id.clone()),
                _ => id.clone(),
            };
            drop(rows);
            drop(conn);

            // Only count new inserts (our generated id matches the stored id).
            if existing_id == id {
                self.increment_relation_type_count(&canonical).await.ok();
            }

            return Ok(existing_id);
        }

        drop(conn);
        Ok(id)
    }

    pub async fn list_entities(
        &self,
        entity_type: Option<&str>,
        domain: Option<&str>,
    ) -> Result<Vec<Entity>, OriginError> {
        let conn = self.conn.lock().await;

        let mut sql = String::from(
            "SELECT id, name, entity_type, domain, source_agent, confidence, confirmed, created_at, updated_at
             FROM entities WHERE 1=1"
        );
        let mut params: Vec<libsql::Value> = Vec::new();

        if let Some(et) = entity_type {
            params.push(et.into());
            sql.push_str(&format!(" AND entity_type = ?{}", params.len()));
        }
        if let Some(d) = domain {
            if d == "uncategorized" {
                sql.push_str(" AND domain IS NULL");
            } else {
                params.push(d.into());
                sql.push_str(&format!(" AND domain = ?{}", params.len()));
            }
        }

        sql.push_str(" ORDER BY updated_at DESC");

        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_entities: {}", e)))?;

        let mut entities = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            entities.push(Entity {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                entity_type: row.get::<String>(2).unwrap_or_default(),
                domain: row.get::<Option<String>>(3).unwrap_or(None),
                source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(6).unwrap_or(0) != 0,
                created_at: row.get::<i64>(7).unwrap_or(0),
                updated_at: row.get::<i64>(8).unwrap_or(0),
            });
        }
        Ok(entities)
    }

    pub async fn get_entity_detail(&self, entity_id: &str) -> Result<EntityDetail, OriginError> {
        let conn = self.conn.lock().await;

        // Fetch entity
        let mut rows = conn
            .query(
                "SELECT id, name, entity_type, domain, source_agent, confidence, confirmed, created_at, updated_at
                 FROM entities WHERE id = ?1",
                libsql::params![entity_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_entity: {}", e)))?;

        let entity = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Entity {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                entity_type: row.get::<String>(2).unwrap_or_default(),
                domain: row.get::<Option<String>>(3).unwrap_or(None),
                source_agent: row.get::<Option<String>>(4).unwrap_or(None),
                confidence: row.get::<Option<f64>>(5).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(6).unwrap_or(0) != 0,
                created_at: row.get::<i64>(7).unwrap_or(0),
                updated_at: row.get::<i64>(8).unwrap_or(0),
            }
        } else {
            return Err(OriginError::VectorDb(format!(
                "Entity not found: {}",
                entity_id
            )));
        };

        // Fetch observations
        let mut obs_rows = conn
            .query(
                "SELECT id, entity_id, content, source_agent, confidence, confirmed, created_at
                 FROM observations WHERE entity_id = ?1 ORDER BY created_at DESC",
                libsql::params![entity_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_observations: {}", e)))?;

        let mut observations = Vec::new();
        while let Some(row) = obs_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            observations.push(Observation {
                id: row.get::<String>(0).unwrap_or_default(),
                entity_id: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                source_agent: row.get::<Option<String>>(3).unwrap_or(None),
                confidence: row.get::<Option<f64>>(4).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(5).unwrap_or(0) != 0,
                created_at: row.get::<i64>(6).unwrap_or(0),
            });
        }

        // Fetch relations (both directions) with entity names
        let mut rel_rows = conn
            .query(
                "SELECT r.id, r.relation_type, r.source_agent, r.created_at,
                        'outgoing' as direction, r.to_entity as entity_id,
                        e.name as entity_name, e.entity_type as entity_type
                 FROM relations r JOIN entities e ON e.id = r.to_entity
                 WHERE r.from_entity = ?1
                 UNION ALL
                 SELECT r.id, r.relation_type, r.source_agent, r.created_at,
                        'incoming' as direction, r.from_entity as entity_id,
                        e.name as entity_name, e.entity_type as entity_type
                 FROM relations r JOIN entities e ON e.id = r.from_entity
                 WHERE r.to_entity = ?1
                 ORDER BY 4 DESC",
                libsql::params![entity_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_relations: {}", e)))?;

        let mut relations = Vec::new();
        while let Some(row) = rel_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            relations.push(RelationWithEntity {
                id: row.get::<String>(0).unwrap_or_default(),
                relation_type: row.get::<String>(1).unwrap_or_default(),
                source_agent: row.get::<Option<String>>(2).unwrap_or(None),
                created_at: row.get::<i64>(3).unwrap_or(0),
                direction: row.get::<String>(4).unwrap_or_default(),
                entity_id: row.get::<String>(5).unwrap_or_default(),
                entity_name: row.get::<String>(6).unwrap_or_default(),
                entity_type: row.get::<String>(7).unwrap_or_default(),
            });
        }

        Ok(EntityDetail {
            entity,
            observations,
            relations,
        })
    }

    pub async fn update_observation(
        &self,
        observation_id: &str,
        content: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE observations SET content = ?1 WHERE id = ?2",
            libsql::params![content, observation_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_observation: {}", e)))?;
        Ok(())
    }

    pub async fn delete_observation(&self, observation_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM observations WHERE id = ?1",
            libsql::params![observation_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_observation: {}", e)))?;
        Ok(())
    }

    pub async fn delete_entity(&self, entity_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        // Null out entity_id references in memories (manual cascade since no FK)
        conn.execute(
            "UPDATE memories SET entity_id = NULL WHERE entity_id = ?1",
            libsql::params![entity_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_entity cascade memories: {}", e)))?;
        // Remove aliases referencing this entity before deleting it (FK constraint)
        conn.execute(
            "DELETE FROM entity_aliases WHERE canonical_entity_id = ?1",
            libsql::params![entity_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_entity cascade aliases: {}", e)))?;
        conn.execute(
            "DELETE FROM entities WHERE id = ?1",
            libsql::params![entity_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_entity: {}", e)))?;
        Ok(())
    }

    /// Resolve entity by name: exact match → LIKE substring → vector search → None.
    pub async fn resolve_entity_by_name(&self, name: &str) -> Result<Option<String>, OriginError> {
        {
            let conn = self.conn.lock().await;
            // Step A: exact case-insensitive match
            let mut rows = conn
                .query(
                    "SELECT id FROM entities WHERE LOWER(name) = LOWER(?1) LIMIT 1",
                    libsql::params![name],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("resolve_entity exact: {}", e)))?;
            if let Ok(Some(row)) = rows.next().await {
                let id: String = row.get(0).unwrap();
                return Ok(Some(id));
            }
            // Step B: fuzzy LIKE substring match
            let mut rows = conn
                .query(
                    "SELECT id FROM entities WHERE name LIKE '%' || ?1 || '%' LIMIT 1",
                    libsql::params![name],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("resolve_entity fuzzy: {}", e)))?;
            if let Ok(Some(row)) = rows.next().await {
                let id: String = row.get(0).unwrap();
                return Ok(Some(id));
            }
        } // Drop conn lock before vector search (which acquires its own lock)
          // Step C: vector similarity match (cosine distance < 0.15 ≈ similarity > 0.85)
        if let Ok(hits) = self.search_entities_by_vector(name, 1).await {
            if let Some(hit) = hits.first() {
                if hit.distance < 0.15 {
                    return Ok(Some(hit.entity.id.clone()));
                }
            }
        }
        Ok(None)
    }

    /// Update a memory's entity_id (for post-ingest entity linking).
    pub async fn update_memory_entity_id(
        &self,
        source_id: &str,
        entity_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET entity_id = ?1 WHERE source_id = ?2",
            libsql::params![entity_id, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_memory_entity_id: {}", e)))?;
        Ok(())
    }

    /// Get the entity_id for a memory (by source_id, chunk_index=0).
    pub async fn get_memory_entity_id(
        &self,
        source_id: &str,
    ) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT entity_id FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_entity_id: {}", e)))?;
        if let Ok(Some(row)) = rows.next().await {
            Ok(row.get::<Option<String>>(0).unwrap_or(None))
        } else {
            Ok(None)
        }
    }

    /// Get the source_agent for a memory (by source_id, chunk_index=0).
    pub async fn get_memory_source_agent(
        &self,
        source_id: &str,
    ) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_agent FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_source_agent: {}", e)))?;
        if let Ok(Some(row)) = rows.next().await {
            Ok(row.get::<Option<String>>(0).unwrap_or(None))
        } else {
            Ok(None)
        }
    }

    /// Find recent memories from the same source_agent within a time window, for batched extraction.
    /// Returns `Vec<(source_id, content)>` ordered by last_modified ASC, max 10.
    pub async fn find_recent_batch(
        &self,
        source_agent: &str,
        window_secs: i64,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let cutoff = now - window_secs;

        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories \
                 WHERE source = 'memory' AND chunk_index = 0 \
                   AND source_agent = ?1 \
                   AND entity_id IS NULL \
                   AND last_modified > ?2 \
                 ORDER BY last_modified ASC \
                 LIMIT 10",
                libsql::params![source_agent, cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("find_recent_batch: {}", e)))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((source_id, content));
        }
        Ok(results)
    }

    /// Find memories that have no entity extraction (entity_id IS NULL).
    /// Used by the refinery's entity backfill phase to gradually self-heal.
    /// Returns `Vec<(source_id, content)>` ordered by last_modified DESC (newest first).
    pub async fn find_memories_without_entities(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories
                 WHERE (entity_id IS NULL)
                   AND content IS NOT NULL AND content != ''
                 ORDER BY last_modified DESC
                 LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("find_memories_without_entities: {e}")))?;
        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((source_id, content));
        }
        Ok(results)
    }

    /// Get enrichment status for a memory, derived from per-step outcomes.
    /// Delegates to `get_enrichment_summary` so callers get a live view of
    /// what actually ran rather than a stale flag.
    pub async fn get_enrichment_status(
        &self,
        source_id: &str,
    ) -> Result<Option<String>, OriginError> {
        let summary = self.get_enrichment_summary(source_id).await?;
        Ok(Some(summary))
    }

    /// Record (or upsert) a single enrichment step outcome for a memory.
    /// If a row for (source_id, step_name) already exists, increments attempts and updates status/error.
    pub async fn record_enrichment_step(
        &self,
        source_id: &str,
        step_name: &str,
        status: &str,
        error: Option<&str>,
    ) -> Result<(), OriginError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO enrichment_steps (source_id, step_name, status, error, attempts, updated_at)
             VALUES (?1, ?2, ?3, ?4, 1, ?5)
             ON CONFLICT(source_id, step_name) DO UPDATE SET
                status = ?3, error = ?4, attempts = enrichment_steps.attempts + 1, updated_at = ?5",
            libsql::params![source_id, step_name, status, error, now],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("record_enrichment_step: {e}")))?;
        Ok(())
    }

    /// Return all enrichment step records for a memory, ordered by insertion.
    pub async fn get_enrichment_steps(
        &self,
        source_id: &str,
    ) -> Result<Vec<origin_types::EnrichmentStepStatus>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT step_name, status, error, attempts FROM enrichment_steps WHERE source_id = ?1 ORDER BY rowid",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_enrichment_steps: {e}")))?;
        let mut steps = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            steps.push(origin_types::EnrichmentStepStatus {
                step: row.get::<String>(0).unwrap_or_default(),
                status: row.get::<String>(1).unwrap_or_default(),
                error: row.get::<Option<String>>(2).ok().flatten(),
                attempts: row.get::<u32>(3).unwrap_or(0),
            });
        }
        Ok(steps)
    }

    /// Derive a summary string from the enrichment_steps for a memory.
    /// Returns: "raw" (no steps), "enriched" (all ok/skipped), "enrichment_failed" (all failed/abandoned),
    /// or "enrichment_partial" (mixed).
    pub async fn get_enrichment_summary(&self, source_id: &str) -> Result<String, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) as total,
                        SUM(CASE WHEN status = 'failed' OR status = 'abandoned' THEN 1 ELSE 0 END) as failed_count,
                        SUM(CASE WHEN status IN ('ok','skipped') THEN 1 ELSE 0 END) as ok_count
                 FROM enrichment_steps WHERE source_id = ?1",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_enrichment_summary: {e}")))?;
        if let Ok(Some(row)) = rows.next().await {
            let total: i64 = row.get(0).unwrap_or(0);
            let failed: i64 = row.get(1).unwrap_or(0);
            let ok: i64 = row.get(2).unwrap_or(0);
            Ok(if total == 0 {
                "raw".to_string()
            } else if failed == 0 {
                "enriched".to_string()
            } else if ok == 0 {
                "enrichment_failed".to_string()
            } else {
                "enrichment_partial".to_string()
            })
        } else {
            Ok("raw".to_string())
        }
    }

    /// Reset any "abandoned" steps for a memory back to "failed" with attempts = 0.
    /// Returns the number of rows updated.
    pub async fn reset_abandoned_steps(&self, source_id: &str) -> Result<u64, OriginError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let rows_affected = conn
            .execute(
                "UPDATE enrichment_steps SET status = 'failed', attempts = 0, updated_at = ?1
                 WHERE source_id = ?2 AND status = 'abandoned'",
                libsql::params![now, source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("reset_abandoned_steps: {e}")))?;
        Ok(rows_affected)
    }

    /// Return memories with at least one `failed` enrichment step that hasn't
    /// exceeded `max_attempts`, ordered oldest-first. Returns (source_id, step_name, content).
    pub async fn get_failed_enrichment_memories(
        &self,
        max_attempts: usize,
        limit: usize,
    ) -> Result<Vec<(String, String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT es.source_id, es.step_name, m.content
                 FROM enrichment_steps es
                 JOIN (SELECT source_id, content FROM memories WHERE chunk_index = 0 AND source = 'memory') m
                    ON m.source_id = es.source_id
                 WHERE es.status = 'failed' AND es.attempts < ?1
                 ORDER BY es.updated_at ASC LIMIT ?2",
                libsql::params![max_attempts as i64, limit as i64],
            )
            .await
            .map_err(|e| {
                OriginError::VectorDb(format!("get_failed_enrichment_memories: {e}"))
            })?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            results.push((
                row.get::<String>(0).unwrap_or_default(),
                row.get::<String>(1).unwrap_or_default(),
                row.get::<String>(2).unwrap_or_default(),
            ));
        }
        Ok(results)
    }

    pub async fn confirm_entity(
        &self,
        entity_id: &str,
        confirmed: bool,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE entities SET confirmed = ?1 WHERE id = ?2",
            libsql::params![if confirmed { 1i64 } else { 0i64 }, entity_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("confirm_entity: {}", e)))?;
        Ok(())
    }

    pub async fn confirm_observation(
        &self,
        observation_id: &str,
        confirmed: bool,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE observations SET confirmed = ?1 WHERE id = ?2",
            libsql::params![if confirmed { 1i64 } else { 0i64 }, observation_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("confirm_observation: {}", e)))?;
        Ok(())
    }

    /// Run label propagation community detection on the entity-relationship graph.
    /// Returns the number of communities found and updates community_id on each entity.
    pub async fn detect_communities(&self) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;

        // 1. Load all entity IDs
        let mut entity_rows = conn
            .query("SELECT id FROM entities", ())
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;

        let mut entity_ids: Vec<String> = Vec::new();
        while let Some(row) = entity_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            entity_ids.push(
                row.get::<String>(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            );
        }
        drop(entity_rows);

        if entity_ids.is_empty() {
            return Ok(0);
        }

        // 2. Build adjacency list from relations table (undirected, weighted by edge count)
        let mut edge_rows = conn
            .query("SELECT from_entity, to_entity FROM relations", ())
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;

        let mut adjacency: std::collections::HashMap<
            String,
            std::collections::HashMap<String, u32>,
        > = std::collections::HashMap::new();

        while let Some(row) = edge_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let from: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let to: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            *adjacency
                .entry(from.clone())
                .or_default()
                .entry(to.clone())
                .or_insert(0) += 1;
            *adjacency.entry(to).or_default().entry(from).or_insert(0) += 1;
        }
        drop(edge_rows);

        // 3. Initialize labels: each entity gets its own index as label
        let id_to_idx: std::collections::HashMap<String, u32> = entity_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i as u32))
            .collect();
        let mut labels: Vec<u32> = (0..entity_ids.len() as u32).collect();

        // 4. Label propagation (max 100 iterations)
        for iteration in 0..100 {
            let mut changed = false;
            for (i, entity_id) in entity_ids.iter().enumerate() {
                if let Some(neighbors) = adjacency.get(entity_id) {
                    // Count neighbor labels weighted by edge count
                    let mut label_counts: std::collections::HashMap<u32, u32> =
                        std::collections::HashMap::new();
                    for (neighbor_id, weight) in neighbors {
                        if let Some(&neighbor_idx) = id_to_idx.get(neighbor_id) {
                            *label_counts
                                .entry(labels[neighbor_idx as usize])
                                .or_insert(0) += weight;
                        }
                    }
                    // Adopt most common label (ties broken by smallest label for determinism)
                    if let Some((&best_label, _)) = label_counts
                        .iter()
                        .max_by(|(l1, c1), (l2, c2)| c1.cmp(c2).then(l2.cmp(l1)))
                    {
                        if labels[i] != best_label {
                            labels[i] = best_label;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                log::info!("[community] converged after {} iterations", iteration + 1);
                break;
            }
        }

        // 5. Normalize labels to sequential community IDs
        let mut label_to_community: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        let mut next_community = 0u32;
        for &label in &labels {
            label_to_community.entry(label).or_insert_with(|| {
                let id = next_community;
                next_community += 1;
                id
            });
        }

        // 6. Update community_id on each entity (batch in transaction)
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        for (i, entity_id) in entity_ids.iter().enumerate() {
            let community_id = label_to_community[&labels[i]];
            conn.execute(
                "UPDATE entities SET community_id = ?1 WHERE id = ?2",
                libsql::params![community_id, entity_id.clone()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;
        }
        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?;

        log::info!(
            "[community] detected {} communities across {} entities",
            next_community,
            entity_ids.len()
        );
        Ok(next_community as usize)
    }

    /// Pin a confirmed memory. Returns an error if the memory is not confirmed
    /// or if the maximum number of pinned memories (12) has been reached.
    pub async fn pin_memory(&self, source_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;

        // Check the memory exists
        let mut rows = conn
            .query(
                "SELECT source_id FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pin_memory check: {}", e)))?;

        rows.next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .ok_or_else(|| OriginError::VectorDb(format!("memory '{}' not found", source_id)))?;

        // Check if already pinned (skip count check if so)
        let mut already_rows = conn
            .query(
                "SELECT pinned FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pin_memory already check: {}", e)))?;

        if let Some(r) = already_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            if r.get::<i64>(0).unwrap_or(0) != 0 {
                // Already pinned, nothing to do
                return Ok(());
            }
        }

        // Check count of distinct pinned source_ids
        let mut count_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE pinned = 1 AND source = 'memory'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pin_memory count: {}", e)))?;

        let pinned_count = count_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<i64>(0).unwrap_or(0))
            .unwrap_or(0);

        if pinned_count >= 12 {
            return Err(OriginError::VectorDb(
                "maximum of 12 pinned memories reached".to_string(),
            ));
        }

        conn.execute(
            "UPDATE memories SET pinned = 1 WHERE source_id = ?1 AND source = 'memory'",
            libsql::params![source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("pin_memory update: {}", e)))?;

        Ok(())
    }

    /// Unpin a memory.
    pub async fn unpin_memory(&self, source_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET pinned = 0 WHERE source_id = ?1 AND source = 'memory'",
            libsql::params![source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("unpin_memory: {}", e)))?;
        Ok(())
    }

    /// List all pinned memories.
    pub async fn list_pinned_memories(&self) -> Result<Vec<MemoryItem>, OriginError> {
        self.list_memories(None, None, None, Some(true), 100).await
    }

    /// Hybrid search filtered to fact-type memories (formerly corrections).
    /// Returns fact memories relevant to the topic, used as guardrails in context assembly.
    pub async fn search_corrections_by_topic(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, OriginError> {
        self.search_memory(query, limit, Some("fact"), None, None, None, None, None)
            .await
    }

    pub async fn get_memory_stats(&self) -> Result<MemoryStats, OriginError> {
        let conn = self.conn.lock().await;

        // Total distinct memories
        let mut total_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats total: {}", e)))?;
        let total = if let Some(row) = total_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        // New today (last 24h)
        let since = chrono::Utc::now().timestamp() - 86400;
        let mut new_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND last_modified > ?1",
                libsql::params![since],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats new: {}", e)))?;
        let new_today = if let Some(row) = new_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        // Confirmed count
        let mut conf_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND confirmed = 1",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats confirmed: {}", e)))?;
        let confirmed = if let Some(row) = conf_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        // Domain breakdown
        let mut domain_rows = conn
            .query(
                "SELECT COALESCE(domain, 'uncategorized') as d, COUNT(DISTINCT source_id) as c
                 FROM memories WHERE source = 'memory'
                 GROUP BY d ORDER BY c DESC",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats domains: {}", e)))?;

        let mut domains = Vec::new();
        while let Some(row) = domain_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            domains.push(DomainInfo {
                name: row.get::<String>(0).unwrap_or_default(),
                count: row.get::<u64>(1).unwrap_or(0),
            });
        }

        // Type breakdown
        let mut type_rows = conn
            .query(
                "SELECT COALESCE(memory_type, 'unknown') as mt, COUNT(DISTINCT source_id) as c
                 FROM memories WHERE source = 'memory'
                 GROUP BY mt ORDER BY c DESC",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats types: {}", e)))?;

        let mut by_type = Vec::new();
        while let Some(row) = type_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            by_type.push(TypeBreakdown {
                memory_type: row.get::<String>(0).unwrap_or_default(),
                count: row.get::<u64>(1).unwrap_or(0),
            });
        }

        // Entity-linked count
        let mut el_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND entity_id IS NOT NULL",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats entity_linked: {}", e)))?;
        let entity_linked = if let Some(row) = el_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        // Enrichment pending count
        let mut ep_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND source_id NOT IN (SELECT DISTINCT source_id FROM enrichment_steps)",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("memory_stats enrichment: {}", e)))?;
        let enrichment_pending = if let Some(row) = ep_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        Ok(MemoryStats {
            total,
            new_today,
            confirmed,
            domains,
            by_type,
            entity_linked,
            enrichment_pending,
        })
    }

    /// Aggregate impact metrics for the homepage dashboard.
    /// Combines access_log (today/week granularity) with access_count (all-time).
    pub async fn get_home_stats(&self) -> Result<HomeStats, OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        let today_start = now - 86400;
        let week_start = now - 7 * 86400;

        // ---- Reuse same queries as get_memory_stats for total/new_today/confirmed ----

        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats total: {e}")))?;
        let total = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND last_modified > ?1 AND source_id NOT LIKE 'merged_%'",
                libsql::params![today_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats new_today: {e}")))?;
        let new_today = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND confirmed = 1",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats confirmed: {e}")))?;
        let confirmed = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Distillation stats ----

        let mut ingested_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("distillation ingested: {}", e)))?;
        let total_ingested = ingested_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // Count memories actually superseded by another memory (distilled away)
        let mut superseded_rows = conn
            .query(
                "SELECT COUNT(DISTINCT c.source_id) FROM memories c \
                 INNER JOIN memories s ON s.supersedes = c.source_id AND s.source = 'memory' \
                 WHERE c.source = 'memory'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("distillation superseded: {}", e)))?;
        let superseded_count = superseded_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);
        let active_insights = total_ingested.saturating_sub(superseded_count);

        // Count distilled memories created today (merged_* source_ids created in last 24h)
        let mut distilled_today_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND source_id LIKE 'merged_%' AND supersede_mode <> 'archive' AND last_modified > ?1",
                libsql::params![today_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("distilled_today: {}", e)))?;
        let distilled_today = distilled_today_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // Count all distilled memories (merged_* all time)
        let mut distilled_all_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND source_id LIKE 'merged_%' AND supersede_mode <> 'archive'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("distilled_all: {}", e)))?;
        let distilled_all = distilled_all_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // Count source memories consumed by distillation:
        // memories referenced by a merged_* memory's supersedes chain + other sources archived during apply_merge.
        // Use: count archived non-merged, non-decision memories (decisions use archive for history, not distillation).
        let mut archived_rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND supersede_mode = 'archive' \
                   AND source_id NOT LIKE 'merged_%' AND source_id NOT LIKE 'recap_%' \
                   AND memory_type <> 'decision'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("sources_archived: {}", e)))?;
        let sources_archived = archived_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Times served today (from access_log, memory sources only) ----

        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE a.accessed_at > ?1 AND c.source = 'memory'",
                libsql::params![today_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats served_today: {e}")))?;
        let times_served_today = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Words saved today (memory sources only) ----

        let mut rows = conn
            .query(
                "SELECT COALESCE(SUM(c.word_count), 0) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE a.accessed_at > ?1 AND c.source = 'memory'",
                libsql::params![today_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats words_today: {e}")))?;
        let words_saved_today = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Times served this week (from access_log, memory sources only) ----

        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE a.accessed_at > ?1 AND c.source = 'memory'",
                libsql::params![week_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats served_week: {e}")))?;
        let times_served_week = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Words saved this week (memory sources only) ----

        let mut rows = conn
            .query(
                "SELECT COALESCE(SUM(c.word_count), 0) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE a.accessed_at > ?1 AND c.source = 'memory'",
                libsql::params![week_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats words_week: {e}")))?;
        let words_saved_week = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Times served all-time (from access_log, consistent with today/week) ----

        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE c.source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats served_all: {e}")))?;
        let times_served_all = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Words saved all-time (from access_log, consistent with today/week) ----

        let mut rows = conn
            .query(
                "SELECT COALESCE(SUM(c.word_count), 0) FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE c.source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats words_all: {e}")))?;
        let words_saved_all = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Key insights (identity + preference + decision + goal + fact) ----

        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND confirmed = 1 \
                   AND memory_type IN ('identity', 'preference', 'decision', 'goal', 'fact')",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats insights: {e}")))?;
        let corrections_active = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<u64>(0).unwrap_or(0))
            .unwrap_or(0);

        // ---- Top 3 memories this week (from access_log) ----

        let mut top_rows = conn
            .query(
                "SELECT a.source_id, SUBSTR(c.content, 1, 200), c.memory_type, c.domain, COUNT(*) as cnt \
                 FROM access_log a \
                 JOIN memories c ON c.source_id = a.source_id AND c.chunk_index = 0 \
                 WHERE a.accessed_at > ?1 \
                 GROUP BY a.source_id ORDER BY cnt DESC LIMIT 3",
                libsql::params![week_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("home_stats top_week: {e}")))?;

        let mut top_memories = Vec::new();
        while let Some(row) = top_rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            top_memories.push(TopMemory {
                source_id: row.get::<String>(0).unwrap_or_default(),
                content: row.get::<String>(1).unwrap_or_default(),
                memory_type: match row.get_value(2) {
                    Ok(libsql::Value::Text(s)) => Some(s),
                    _ => None,
                },
                domain: match row.get_value(3) {
                    Ok(libsql::Value::Text(s)) => Some(s),
                    _ => None,
                },
                times_retrieved: row.get::<u64>(4).unwrap_or(0),
            });
        }

        // ---- Fallback: top 3 all-time if no weekly data ----

        if top_memories.is_empty() {
            let mut fallback_rows = conn
                .query(
                    "SELECT source_id, SUBSTR(content, 1, 200), memory_type, domain, access_count \
                     FROM memories WHERE source = 'memory' AND access_count > 0 AND chunk_index = 0 \
                     ORDER BY access_count DESC LIMIT 3",
                    libsql::params![],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("home_stats top_alltime: {e}")))?;

            while let Some(row) = fallback_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                top_memories.push(TopMemory {
                    source_id: row.get::<String>(0).unwrap_or_default(),
                    content: row.get::<String>(1).unwrap_or_default(),
                    memory_type: match row.get_value(2) {
                        Ok(libsql::Value::Text(s)) => Some(s),
                        _ => None,
                    },
                    domain: match row.get_value(3) {
                        Ok(libsql::Value::Text(s)) => Some(s),
                        _ => None,
                    },
                    times_retrieved: row.get::<u64>(4).unwrap_or(0),
                });
            }
        }

        Ok(HomeStats {
            total,
            new_today,
            confirmed,
            total_ingested,
            active_insights,
            distilled_today,
            distilled_all,
            sources_archived,
            times_served_today,
            words_saved_today,
            times_served_week,
            words_saved_week,
            times_served_all,
            words_saved_all,
            corrections_active,
            top_memories,
        })
    }

    // ==================== Version Chain ====================

    /// Walk the version chain for a memory, returning all versions ordered oldest->newest.
    pub async fn get_version_chain(
        &self,
        source_id: &str,
    ) -> Result<Vec<MemoryVersionItem>, OriginError> {
        let conn = self.conn.lock().await;

        // Step 1: Walk backward to find root
        let mut current = source_id.to_string();
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(current.clone()) {
                break; // cycle detection
            }

            // Get the supersedes field for current source_id
            let mut rows = conn
                .query(
                    "SELECT supersedes FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                    libsql::params![current.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain backward: {e}")))?;

            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain row: {e}")))?
            {
                // supersedes may be NULL — get_value lets us check
                match row.get_value(0) {
                    Ok(libsql::Value::Text(prev)) if !prev.is_empty() => {
                        current = prev;
                        continue;
                    }
                    _ => break,
                }
            } else {
                break; // source_id not found
            }
        }

        let root = current;

        // Step 2: Walk forward from root
        let mut chain = Vec::new();
        let mut current = root;
        let mut visited = std::collections::HashSet::new();
        loop {
            if !visited.insert(current.clone()) {
                break; // cycle detection
            }

            // Load this version's data (aggregate across rows)
            let mut rows = conn
                .query(
                    "SELECT source_id, MAX(title), MAX(content), MAX(memory_type),
                            MAX(CAST(confirmed AS INTEGER)), MAX(supersedes), MAX(last_modified)
                     FROM memories WHERE source_id = ?1 AND source = 'memory' GROUP BY source_id",
                    libsql::params![current.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain forward: {e}")))?;

            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain row: {e}")))?
            {
                chain.push(MemoryVersionItem {
                    source_id: row.get::<String>(0).unwrap_or_default(),
                    title: row.get::<String>(1).unwrap_or_default(),
                    content: row.get::<String>(2).unwrap_or_default(),
                    memory_type: row.get::<Option<String>>(3).unwrap_or(None),
                    confirmed: row.get::<i64>(4).unwrap_or(0) != 0,
                    supersedes: row.get::<Option<String>>(5).unwrap_or(None),
                    last_modified: row.get::<i64>(6).unwrap_or(0),
                });
            } else {
                break; // source_id not found in DB
            }

            // Find the next version (who supersedes current?)
            let mut next_rows = conn
                .query(
                    "SELECT source_id FROM memories WHERE supersedes = ?1 AND source = 'memory' GROUP BY source_id LIMIT 1",
                    libsql::params![current.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain next: {e}")))?;

            if let Some(row) = next_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("version_chain next row: {e}")))?
            {
                current = row.get::<String>(0).unwrap_or_default();
            } else {
                break; // end of chain
            }
        }

        Ok(chain)
    }

    // ==================== Profile CRUD ====================

    pub async fn bootstrap_profile(&self) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM profiles", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("bootstrap_profile count: {}", e)))?;
        let count: i64 = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            0
        };
        if count == 0 {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().timestamp();
            conn.execute(
                "INSERT INTO profiles (id, name, display_name, created_at, updated_at) VALUES (?1, ?2, NULL, ?3, ?3)",
                libsql::params![id, "User".to_string(), now],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("bootstrap_profile insert: {}", e)))?;
        }
        Ok(())
    }

    pub async fn get_profile(&self) -> Result<Option<Profile>, OriginError> {
        self.bootstrap_profile().await?;
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT id, name, display_name, email, bio, avatar_path, created_at, updated_at FROM profiles LIMIT 1", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_profile: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(Some(Profile {
                id: row.get::<String>(0).unwrap_or_default(),
                name: row.get::<String>(1).unwrap_or_default(),
                display_name: row.get::<Option<String>>(2).unwrap_or(None),
                email: row.get::<Option<String>>(3).unwrap_or(None),
                bio: row.get::<Option<String>>(4).unwrap_or(None),
                avatar_path: row.get::<Option<String>>(5).unwrap_or(None),
                created_at: row.get::<i64>(6).unwrap_or(0),
                updated_at: row.get::<i64>(7).unwrap_or(0),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn update_profile(
        &self,
        id: &str,
        name: Option<&str>,
        display_name: Option<&str>,
        email: Option<&str>,
        bio: Option<&str>,
        avatar_path: Option<&str>,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();

        let mut sets = vec!["updated_at = ?1".to_string()];
        let mut params: Vec<libsql::Value> = vec![now.into()];

        if let Some(n) = name {
            params.push(n.into());
            sets.push(format!("name = ?{}", params.len()));
        }

        // Helper: empty string → NULL, non-empty → value
        let nullable_fields: &[(&str, Option<&str>)] = &[
            ("display_name", display_name),
            ("email", email),
            ("bio", bio),
            ("avatar_path", avatar_path),
        ];
        for (col, val) in nullable_fields {
            if let Some(v) = val {
                if v.is_empty() {
                    params.push(libsql::Value::Null);
                } else {
                    params.push((*v).into());
                }
                sets.push(format!("{} = ?{}", col, params.len()));
            }
        }

        params.push(id.into());
        let sql = format!(
            "UPDATE profiles SET {} WHERE id = ?{}",
            sets.join(", "),
            params.len()
        );

        conn.execute(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("update_profile: {}", e)))?;
        Ok(())
    }

    // ==================== Agent Connection CRUD ====================

    pub async fn register_agent(&self, name: &str) -> Result<AgentConnection, OriginError> {
        // Canonicalize on the way in so `"Claude Code"` and `"claude-code"`
        // collapse to one row. The caller's original label — if it differs —
        // is promoted to `display_name` for UI.
        let canonical = canonicalize_agent_id(name);
        let original_label = if canonical != name.trim().to_lowercase() || canonical != name {
            Some(name.to_string())
        } else {
            None
        };
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, name, display_name, agent_type, description, enabled, trust_level, last_seen_at, memory_count, created_at, updated_at
                 FROM agent_connections WHERE name = ?1",
                libsql::params![canonical.clone()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("register_agent select: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            return Ok(Self::row_to_agent(&row));
        }
        drop(rows);

        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();
        let display_name_to_store = original_label
            .clone()
            .or_else(|| known_client_display_name(&canonical).map(|s| s.to_string()));
        // Default new registrations to `"full"`. Rationale: on a single-user
        // local server, registration IS the trust gesture — if you ran the
        // SetupWizard for this agent, you want it to see your identity and
        // preferences. The old default (`"review"`) silently gated Tier 1
        // chat-context for every agent, which nobody noticed or wanted.
        // Unregistered callers still fall through to `"unknown"` in
        // `handle_chat_context` and only see Tier 3 (search results).
        conn.execute(
            "INSERT INTO agent_connections (id, name, display_name, agent_type, description, enabled, trust_level, last_seen_at, memory_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'api', NULL, 1, 'full', NULL, 0, ?4, ?4)",
            libsql::params![id.clone(), canonical.clone(), display_name_to_store.clone(), now],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("register_agent insert: {}", e)))?;

        Ok(AgentConnection {
            id,
            name: canonical,
            display_name: display_name_to_store,
            agent_type: "api".to_string(),
            description: None,
            enabled: true,
            trust_level: "full".to_string(),
            last_seen_at: None,
            memory_count: 0,
            created_at: now,
            updated_at: now,
        })
    }

    pub async fn get_agent(&self, name: &str) -> Result<Option<AgentConnection>, OriginError> {
        let canonical = canonicalize_agent_id(name);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, name, display_name, agent_type, description, enabled, trust_level, last_seen_at, memory_count, created_at, updated_at
                 FROM agent_connections WHERE name = ?1",
                libsql::params![canonical],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_agent: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(Some(Self::row_to_agent(&row)))
        } else {
            Ok(None)
        }
    }

    pub async fn list_agents(&self) -> Result<Vec<AgentConnection>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, name, display_name, agent_type, description, enabled, trust_level, last_seen_at, memory_count, created_at, updated_at
                 FROM agent_connections ORDER BY COALESCE(last_seen_at, 0) DESC, created_at DESC",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_agents: {}", e)))?;
        let mut agents = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            agents.push(Self::row_to_agent(&row));
        }
        Ok(agents)
    }

    pub async fn update_agent(
        &self,
        name: &str,
        agent_type: Option<&str>,
        description: Option<&str>,
        enabled: Option<bool>,
        trust_level: Option<&str>,
        display_name: Option<&str>,
    ) -> Result<(), OriginError> {
        let canonical = canonicalize_agent_id(name);
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        let mut sets = vec!["updated_at = ?1".to_string()];
        let mut params: Vec<libsql::Value> = vec![now.into()];
        if let Some(at) = agent_type {
            params.push(at.into());
            sets.push(format!("agent_type = ?{}", params.len()));
        }
        if let Some(d) = description {
            params.push(d.into());
            sets.push(format!("description = ?{}", params.len()));
        }
        if let Some(e) = enabled {
            params.push((e as i64).into());
            sets.push(format!("enabled = ?{}", params.len()));
        }
        if let Some(tl) = trust_level {
            params.push(tl.into());
            sets.push(format!("trust_level = ?{}", params.len()));
        }
        if let Some(dn) = display_name {
            // Allow explicit clearing via empty string.
            let val: libsql::Value = if dn.is_empty() {
                libsql::Value::Null
            } else {
                dn.into()
            };
            params.push(val);
            sets.push(format!("display_name = ?{}", params.len()));
        }
        params.push(canonical.into());
        let sql = format!(
            "UPDATE agent_connections SET {} WHERE name = ?{}",
            sets.join(", "),
            params.len()
        );
        conn.execute(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("update_agent: {}", e)))?;
        Ok(())
    }

    pub async fn delete_agent(&self, name: &str) -> Result<(), OriginError> {
        let canonical = canonicalize_agent_id(name);
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM agent_connections WHERE name = ?1",
            libsql::params![canonical],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_agent: {}", e)))?;
        Ok(())
    }

    pub async fn touch_agent(&self, name: &str) -> Result<(), OriginError> {
        let canonical = canonicalize_agent_id(name);
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "UPDATE agent_connections SET memory_count = memory_count + 1, last_seen_at = ?1, updated_at = ?1 WHERE name = ?2",
            libsql::params![now, canonical],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("touch_agent: {}", e)))?;
        Ok(())
    }

    /// Check if an agent is allowed to write. Auto-registers if unknown.
    /// Returns the agent's trust_level on success, or OriginError::AgentDisabled if disabled.
    pub async fn check_agent_for_write(&self, agent_name: &str) -> Result<String, OriginError> {
        let agent = self.register_agent(agent_name).await?;
        if !agent.enabled {
            return Err(OriginError::AgentDisabled(format!(
                "Agent '{}' is disabled",
                agent_name
            )));
        }
        self.touch_agent(agent_name).await?;
        Ok(agent.trust_level)
    }

    /// Convenience wrapper for Option<&str> source_agent.
    /// Returns "full" for None (local/first-party writes).
    pub async fn check_agent_for_write_optional(
        &self,
        agent_name: Option<&str>,
    ) -> Result<String, OriginError> {
        match agent_name {
            Some(name) => self.check_agent_for_write(name).await,
            None => Ok("full".to_string()),
        }
    }

    fn row_to_agent(row: &libsql::Row) -> AgentConnection {
        let raw_name = row.get::<String>(1).unwrap_or_default();
        // Column 2 is display_name (nullable). If the DB doesn't set one,
        // fall back to the KNOWN_CLIENTS registry so well-known technical IDs
        // (e.g. `openai-mcp`) surface with their friendly name (`ChatGPT`)
        // without requiring the user to explicitly register them.
        let display_name = row
            .get::<Option<String>>(2)
            .unwrap_or(None)
            .or_else(|| known_client_display_name(&raw_name).map(|s| s.to_string()));
        AgentConnection {
            id: row.get::<String>(0).unwrap_or_default(),
            name: raw_name,
            display_name,
            agent_type: row.get::<String>(3).unwrap_or_default(),
            description: row.get::<Option<String>>(4).unwrap_or(None),
            enabled: row.get::<i64>(5).unwrap_or(1) != 0,
            trust_level: row
                .get::<String>(6)
                .unwrap_or_else(|_| "review".to_string()),
            last_seen_at: row.get::<Option<i64>>(7).unwrap_or(None),
            memory_count: row.get::<i64>(8).unwrap_or(0),
            created_at: row.get::<i64>(9).unwrap_or(0),
            updated_at: row.get::<i64>(10).unwrap_or(0),
        }
    }

    // ==================== Onboarding Milestones ====================

    /// Record a milestone. Returns `Some(record)` if newly fired, `None` if
    /// already fired. Race-safe against concurrent callers via
    /// `ON CONFLICT ... DO NOTHING RETURNING` — two simultaneous
    /// evaluator checks cannot both fire the same milestone.
    pub async fn record_milestone(
        &self,
        id: crate::onboarding::MilestoneId,
        payload: Option<serde_json::Value>,
    ) -> Result<Option<crate::onboarding::MilestoneRecord>, OriginError> {
        let now = chrono::Utc::now().timestamp();
        let payload_str =
            match &payload {
                Some(v) => Some(serde_json::to_string(v).map_err(|e| {
                    OriginError::Generic(format!("record_milestone payload: {}", e))
                })?),
                None => None,
            };
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "INSERT INTO onboarding_milestones (id, first_triggered_at, payload) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(id) DO NOTHING \
                 RETURNING id, first_triggered_at, acknowledged_at, payload",
                libsql::params![id.as_str(), now, payload_str],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("record_milestone insert: {}", e)))?;
        match rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("record_milestone next: {}", e)))?
        {
            Some(row) => {
                let id_str: String = row.get(0).map_err(|e| {
                    OriginError::VectorDb(format!("record_milestone row.id: {}", e))
                })?;
                let parsed_id = crate::onboarding::MilestoneId::from_str(&id_str)
                    .map_err(OriginError::Generic)?;
                let first_triggered_at: i64 = row.get(1).map_err(|e| {
                    OriginError::VectorDb(format!("record_milestone row.first_triggered_at: {}", e))
                })?;
                let acknowledged_at: Option<i64> = row.get(2).map_err(|e| {
                    OriginError::VectorDb(format!("record_milestone row.acknowledged_at: {}", e))
                })?;
                let payload_str_out: Option<String> = row.get(3).map_err(|e| {
                    OriginError::VectorDb(format!("record_milestone row.payload: {}", e))
                })?;
                let payload_val = payload_str_out.and_then(|s| serde_json::from_str(&s).ok());
                Ok(Some(crate::onboarding::MilestoneRecord {
                    id: parsed_id,
                    first_triggered_at,
                    acknowledged_at,
                    payload: payload_val,
                }))
            }
            None => Ok(None),
        }
    }

    /// Return all milestone rows ordered by trigger time ascending. Called
    /// by the `/api/onboarding/milestones` endpoint and the cold-start
    /// toast replay logic in `MilestoneToaster`.
    pub async fn list_milestones(
        &self,
    ) -> Result<Vec<crate::onboarding::MilestoneRecord>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, first_triggered_at, acknowledged_at, payload \
                 FROM onboarding_milestones ORDER BY first_triggered_at ASC",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_milestones: {}", e)))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_milestones next: {}", e)))?
        {
            let id_str: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(format!("list_milestones row.id: {}", e)))?;
            let parsed_id = match crate::onboarding::MilestoneId::from_str(&id_str) {
                Ok(id) => id,
                Err(e) => {
                    log::warn!(
                        "list_milestones: skipping unknown milestone id '{}': {}",
                        id_str,
                        e
                    );
                    continue;
                }
            };
            let first_triggered_at: i64 = row.get(1).map_err(|e| {
                OriginError::VectorDb(format!("list_milestones row.first_triggered_at: {}", e))
            })?;
            let acknowledged_at: Option<i64> = row.get(2).map_err(|e| {
                OriginError::VectorDb(format!("list_milestones row.acknowledged_at: {}", e))
            })?;
            let payload_str: Option<String> = row.get(3).map_err(|e| {
                OriginError::VectorDb(format!("list_milestones row.payload: {}", e))
            })?;
            let payload_val = payload_str.and_then(|s| serde_json::from_str(&s).ok());
            out.push(crate::onboarding::MilestoneRecord {
                id: parsed_id,
                first_triggered_at,
                acknowledged_at,
                payload: payload_val,
            });
        }
        Ok(out)
    }

    /// Set `acknowledged_at = now()` for the given milestone, but only if
    /// it has not already been acknowledged (so we don't overwrite the
    /// original ack timestamp on repeated clicks).
    pub async fn acknowledge_milestone(
        &self,
        id: crate::onboarding::MilestoneId,
    ) -> Result<(), OriginError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE onboarding_milestones SET acknowledged_at = ?1 \
             WHERE id = ?2 AND acknowledged_at IS NULL",
            libsql::params![now, id.as_str()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("acknowledge_milestone: {}", e)))?;
        Ok(())
    }

    /// Update the `payload.shown_count` counter for a milestone. Used by
    /// `FirstConceptModal` to track non-acknowledging dismissals so the
    /// modal self-retires after 3 shows (see spec §3.2). Returns the new
    /// count. If the milestone is not yet recorded, this is a no-op and
    /// returns 0.
    pub async fn increment_milestone_shown_count(
        &self,
        id: crate::onboarding::MilestoneId,
    ) -> Result<u32, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT payload FROM onboarding_milestones WHERE id = ?1",
                libsql::params![id.as_str()],
            )
            .await
            .map_err(|e| {
                OriginError::VectorDb(format!("increment_milestone_shown_count select: {}", e))
            })?;
        let current: Option<String> = match rows.next().await.map_err(|e| {
            OriginError::VectorDb(format!("increment_milestone_shown_count next: {}", e))
        })? {
            Some(r) => r.get(0).map_err(|e| {
                OriginError::VectorDb(format!("increment_milestone_shown_count row: {}", e))
            })?,
            None => return Ok(0),
        };
        let mut payload_val: serde_json::Value = current
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        let count = payload_val
            .get("shown_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
            + 1;
        payload_val["shown_count"] = serde_json::json!(count);
        let new_str = serde_json::to_string(&payload_val).map_err(|e| {
            OriginError::Generic(format!("increment_milestone_shown_count serialize: {}", e))
        })?;
        conn.execute(
            "UPDATE onboarding_milestones SET payload = ?1 WHERE id = ?2",
            libsql::params![new_str, id.as_str()],
        )
        .await
        .map_err(|e| {
            OriginError::VectorDb(format!("increment_milestone_shown_count update: {}", e))
        })?;
        Ok(count)
    }

    /// Clear all milestone rows. Dev/demo-only — exposed via
    /// `POST /api/onboarding/reset` and gated to `import.meta.env.DEV` on
    /// the frontend side.
    pub async fn reset_onboarding_milestones(&self) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM onboarding_milestones", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("reset_onboarding_milestones: {}", e)))?;
        Ok(())
    }

    // ===== Count helpers for MilestoneEvaluator =====

    /// Count active (i.e. not archived/superseded) concepts.
    pub async fn count_active_concepts(&self) -> Result<i64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM concepts WHERE status = 'active'", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_active_concepts query: {}", e)))?;
        let row = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_active_concepts next: {}", e)))?
            .ok_or_else(|| OriginError::Generic("count_active_concepts: no rows".into()))?;
        row.get::<i64>(0)
            .map_err(|e| OriginError::VectorDb(format!("count_active_concepts get: {}", e)))
    }

    /// Return the most-recently-modified active concept, if any. Used by
    /// `MilestoneEvaluator::check_after_refinery_pass` to build the
    /// `first-concept` payload.
    pub async fn first_active_concept(
        &self,
    ) -> Result<Option<crate::concepts::Concept>, OriginError> {
        let mut list = self.list_concepts("active", 1, 0).await?;
        Ok(list.pop())
    }

    /// Return the genuinely first-compiled active concept (ordered by
    /// `created_at` ascending). Used by `MilestoneEvaluator::check_after_refinery_pass`
    /// so the `first-concept` milestone payload references the concept that
    /// was actually compiled first, not the most recently edited one.
    pub async fn oldest_active_concept(
        &self,
    ) -> Result<Option<crate::concepts::Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE status = 'active' ORDER BY created_at ASC LIMIT 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("oldest_active_concept: {e}")))?;
        match rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("oldest_active_concept next: {e}")))?
        {
            Some(row) => Ok(Some(Self::row_to_concept(&row)?)),
            None => Ok(None),
        }
    }

    /// Count rows in the `entities` table (knowledge-graph nodes).
    pub async fn count_entities(&self) -> Result<i64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM entities", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_entities query: {}", e)))?;
        let row = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_entities next: {}", e)))?
            .ok_or_else(|| OriginError::Generic("count_entities: no rows".into()))?;
        row.get::<i64>(0)
            .map_err(|e| OriginError::VectorDb(format!("count_entities get: {}", e)))
    }

    /// Count rows in the `relations` table (knowledge-graph edges).
    pub async fn count_relations(&self) -> Result<i64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM relations", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_relations query: {}", e)))?;
        let row = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_relations next: {}", e)))?
            .ok_or_else(|| OriginError::Generic("count_relations: no rows".into()))?;
        row.get::<i64>(0)
            .map_err(|e| OriginError::VectorDb(format!("count_relations get: {}", e)))
    }

    /// Return the most recent knowledge-graph relations with entity names resolved.
    ///
    /// `since_ms` filters by `created_at` (unix seconds). Rows where either
    /// entity name is missing are excluded so the UI always shows readable text.
    pub async fn list_recent_relations(
        &self,
        limit: usize,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentRelation>, OriginError> {
        let conn = self.conn.lock().await;
        let sql = "SELECT r.id, r.from_entity, r.relation_type, r.to_entity, \
                   e1.name AS from_entity_name, e2.name AS to_entity_name, \
                   r.created_at \
                   FROM relations r \
                   JOIN entities e1 ON r.from_entity = e1.id \
                   JOIN entities e2 ON r.to_entity = e2.id \
                   WHERE (?1 IS NULL OR r.created_at >= ?1) \
                   AND e1.name IS NOT NULL AND e1.name != '' \
                   AND e2.name IS NOT NULL AND e2.name != '' \
                   ORDER BY r.created_at DESC LIMIT ?2";
        let mut rows = conn
            .query(sql, libsql::params![since_ms, limit as i64])
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_recent_relations query: {}", e)))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_recent_relations next: {}", e)))?
        {
            out.push(origin_types::RecentRelation {
                id: row.get::<String>(0).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 0: {}", e))
                })?,
                from_entity_id: row.get::<String>(1).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 1: {}", e))
                })?,
                relation_type: row.get::<String>(2).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 2: {}", e))
                })?,
                to_entity_id: row.get::<String>(3).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 3: {}", e))
                })?,
                from_entity_name: row.get::<String>(4).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 4: {}", e))
                })?,
                to_entity_name: row.get::<String>(5).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 5: {}", e))
                })?,
                created_at_ms: row.get::<i64>(6).map_err(|e| {
                    OriginError::VectorDb(format!("list_recent_relations col 6: {}", e))
                })?,
            });
        }
        Ok(out)
    }

    /// Count agent connections that have recorded at least one memory write.
    /// Used to detect when a second agent starts contributing (the
    /// `second-agent` milestone fires at ≥2).
    pub async fn count_agents_with_writes(&self) -> Result<i64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM agent_connections WHERE memory_count >= 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_agents_with_writes query: {}", e)))?;
        let row = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("count_agents_with_writes next: {}", e)))?
            .ok_or_else(|| OriginError::Generic("count_agents_with_writes: no rows".into()))?;
        row.get::<i64>(0)
            .map_err(|e| OriginError::VectorDb(format!("count_agents_with_writes get: {}", e)))
    }

    // ===== Tiered Stability Methods =====

    pub async fn get_memory_type(&self, source_id: &str) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT memory_type FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_type: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(row.get::<Option<String>>(0).unwrap_or(None))
        } else {
            Ok(None)
        }
    }

    pub async fn get_memory_domain(&self, source_id: &str) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT domain FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_domain: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(row.get::<Option<String>>(0).unwrap_or(None))
        } else {
            Ok(None)
        }
    }

    pub async fn source_id_exists(&self, source_id: &str) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM memories WHERE source_id = ?1 LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("source_id_exists: {}", e)))?;
        Ok(rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .is_some())
    }

    pub async fn get_stability(&self, source_id: &str) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT stability FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_stability: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(row.get::<Option<String>>(0).unwrap_or(None))
        } else {
            Ok(None)
        }
    }

    pub async fn set_stability(&self, source_id: &str, stability: &str) -> Result<(), OriginError> {
        if !matches!(stability, "new" | "learned" | "confirmed") {
            return Err(OriginError::VectorDb(format!(
                "invalid stability: {}",
                stability
            )));
        }
        let conn = self.conn.lock().await;
        let confirmed_int: i64 = if stability == "confirmed" { 1 } else { 0 };
        if stability == "confirmed" {
            conn.execute(
                "UPDATE memories SET stability = ?1, confirmed = ?2, confidence = 1.0 WHERE source_id = ?3 AND source = 'memory'",
                libsql::params![stability.to_string(), confirmed_int, source_id.to_string()],
            ).await.map_err(|e| OriginError::VectorDb(format!("set_stability: {}", e)))?;
        } else {
            conn.execute(
                "UPDATE memories SET stability = ?1, confirmed = ?2 WHERE source_id = ?3 AND source = 'memory'",
                libsql::params![stability.to_string(), confirmed_int, source_id.to_string()],
            ).await.map_err(|e| OriginError::VectorDb(format!("set_stability: {}", e)))?;
        }
        Ok(())
    }

    pub async fn confirm_memory(&self, source_id: &str) -> Result<(), OriginError> {
        self.set_stability(source_id, "confirmed").await
    }

    /// Returns up to `limit` memories that need human review, ranked by priority:
    /// 1. Low quality from untrusted agents
    /// 2. Low confidence
    /// 3. Oldest unreviewed new memories
    pub async fn get_nurture_cards(
        &self,
        limit: usize,
        domain_filter: Option<&str>,
    ) -> Result<Vec<MemoryItem>, OriginError> {
        let conn = self.conn.lock().await;
        let domain_clause = if domain_filter.is_some() {
            "AND c.domain = ?2"
        } else {
            ""
        };
        let sql = format!(
            "SELECT c.source_id, c.title, c.content, c.summary, c.memory_type, c.domain,
                    c.source_agent, c.confidence, c.confirmed, c.stability,
                    c.pinned, c.supersedes, c.last_modified,
                    c.entity_id, c.quality, COALESCE(c.is_recap, 0) as is_recap,
                    (SELECT CASE
                        WHEN COUNT(es.source_id) = 0 THEN 'raw'
                        WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched'
                        WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed'
                        ELSE 'enrichment_partial'
                    END FROM enrichment_steps es WHERE es.source_id = c.source_id) AS enrichment_status, c.supersede_mode, c.structured_fields,
                    c.retrieval_cue, c.access_count, c.source_text
             FROM memories c
             WHERE c.source = 'memory'
               AND c.stability = 'new'
               AND COALESCE(c.is_recap, 0) = 0
               AND c.source_id NOT IN (
                   SELECT supersedes FROM memories WHERE supersedes IS NOT NULL AND source = 'memory'
               )
               {}
             ORDER BY
               c.pending_revision DESC,
               CASE c.quality WHEN 'low' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END ASC,
               c.confidence ASC,
               c.last_modified DESC
             LIMIT ?1",
            domain_clause
        );

        let mut rows = if let Some(domain) = domain_filter {
            conn.query(&sql, libsql::params![limit as i64, domain.to_string()])
                .await
        } else {
            conn.query(&sql, libsql::params![limit as i64]).await
        }
        .map_err(|e| OriginError::VectorDb(format!("get_nurture_cards: {}", e)))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            // Column order: 0=source_id, 1=title, 2=content, 3=summary, 4=memory_type,
            // 5=domain, 6=source_agent, 7=confidence, 8=confirmed, 9=stability, 10=pinned,
            // 11=supersedes, 12=last_modified, 13=entity_id, 14=quality, 15=is_recap,
            // 16=enrichment_status, 17=supersede_mode, 18=structured_fields, 19=retrieval_cue,
            // 20=access_count, 21=source_text
            results.push(MemoryItem {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                summary: row.get::<Option<String>>(3).unwrap_or(None),
                memory_type: row.get::<Option<String>>(4).unwrap_or(None),
                domain: row.get::<Option<String>>(5).unwrap_or(None),
                source_agent: row.get::<Option<String>>(6).unwrap_or(None),
                confidence: row.get::<Option<f64>>(7).unwrap_or(None).map(|v| v as f32),
                confirmed: row.get::<i64>(8).unwrap_or(0) != 0,
                stability: row.get::<Option<String>>(9).unwrap_or(None),
                pinned: row.get::<i64>(10).unwrap_or(0) != 0,
                supersedes: row.get::<Option<String>>(11).unwrap_or(None),
                last_modified: row.get::<i64>(12).unwrap_or(0),
                chunk_count: 1,
                entity_id: row.get::<Option<String>>(13).unwrap_or(None),
                quality: row.get::<Option<String>>(14).unwrap_or(None),
                is_recap: row.get::<i64>(15).unwrap_or(0) != 0,
                enrichment_status: row.get::<String>(16).unwrap_or_else(|_| "raw".to_string()),
                supersede_mode: row.get::<String>(17).unwrap_or_else(|_| "hide".to_string()),
                structured_fields: row.get::<Option<String>>(18).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(19).unwrap_or(None),
                access_count: row.get::<u64>(20).unwrap_or(0),
                source_text: row.get::<Option<String>>(21).unwrap_or(None),
                version: 1,
                changelog: None,
            });
        }
        Ok(results)
    }

    /// Promote 'new' memories to 'learned' if they've survived min_age_days without contradiction.
    /// Protected tier (identity, preference) never auto-promotes.
    pub async fn promote_uncontradicted(&self, min_age_days: i64) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;
        let cutoff = chrono::Utc::now().timestamp() - (min_age_days * 86400);
        // TODO: Add contradiction check — exclude memories flagged in refinement_queue
        // Currently promotes all qualifying memories regardless of contradiction status.
        // When refinement_queue.action = 'detect_contradiction' is reliably populated,
        // add: AND source_id NOT IN (SELECT target_id FROM refinement_queue WHERE action = 'detect_contradiction' AND status = 'pending')
        let changed = conn
            .execute(
                "UPDATE memories SET stability = 'learned', confirmed = 0
             WHERE source = 'memory'
               AND stability = 'new'
               AND memory_type NOT IN ('identity', 'preference')
               AND COALESCE(quality, 'medium') != 'low'
               AND last_modified < ?1",
                libsql::params![cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("promote_uncontradicted: {}", e)))?;
        Ok(changed as usize)
    }

    /// Accept a pending revision for a target memory. The `target_source_id` is the
    /// original memory being superseded. This finds the pending revision that supersedes it,
    /// activates it, and suppresses the original.
    pub async fn accept_pending_revision(&self, target_source_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;

        // Find the pending revision that supersedes this target
        let mut rows = conn
            .query(
                "SELECT source_id FROM memories WHERE supersedes = ?1 AND pending_revision = 1 AND source = 'memory' LIMIT 1",
                libsql::params![target_source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("accept_pending_revision query: {}", e)))?;

        let revision_source_id: String = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<String>(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
        } else {
            return Err(OriginError::VectorDb(format!(
                "No pending revision found for source_id: {}",
                target_source_id
            )));
        };

        // Activate the revision
        conn.execute(
            "UPDATE memories SET pending_revision = 0, confirmed = 1, stability = 'confirmed', confidence = 1.0 WHERE source_id = ?1",
            libsql::params![revision_source_id.clone()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("accept_pending_revision activate: {}", e)))?;

        // Suppress the original
        conn.execute(
            "UPDATE memories SET confirmed = 0, stability = 'new' WHERE source_id = ?1",
            libsql::params![target_source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("accept_pending_revision suppress: {}", e)))?;

        Ok(())
    }

    /// Dismiss a pending revision for a target memory. Deletes the pending revision,
    /// leaving the original unchanged.
    pub async fn dismiss_pending_revision(
        &self,
        target_source_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let rows_affected = conn
            .execute(
                "DELETE FROM memories WHERE supersedes = ?1 AND pending_revision = 1 AND source = 'memory'",
                libsql::params![target_source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("dismiss_pending_revision: {}", e)))?;
        if rows_affected == 0 {
            return Err(OriginError::VectorDb(format!(
                "No pending revision found for source_id: {}",
                target_source_id
            )));
        }
        Ok(())
    }

    pub async fn get_pending_revision_for(
        &self,
        target_source_id: &str,
    ) -> Result<Option<PendingRevision>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content, source_agent FROM memories WHERE supersedes = ?1 AND pending_revision = 1 AND source = 'memory' LIMIT 1",
                libsql::params![target_source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_pending_revision_for: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(Some(PendingRevision {
                source_id: row
                    .get::<String>(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                content: row
                    .get::<String>(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                source_agent: row.get::<Option<String>>(2).unwrap_or(None),
            }))
        } else {
            Ok(None)
        }
    }

    // ==================== Session / Activity Methods ====================

    /// Insert or update an activity record.
    pub async fn upsert_activity(
        &self,
        id: &str,
        started_at: i64,
        ended_at: i64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO activities (id, started_at, ended_at) VALUES (?1, ?2, ?3) ON CONFLICT(id) DO UPDATE SET ended_at = ?3",
            libsql::params![id, started_at, ended_at],
        ).await.map_err(|e| OriginError::VectorDb(format!("upsert_activity: {}", e)))?;
        Ok(())
    }

    /// Get completed activities with ended_at set.
    pub async fn get_completed_activities(
        &self,
        _gap_secs: i64,
    ) -> Result<Vec<ActivityRow>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, started_at, ended_at FROM activities ORDER BY started_at DESC",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_completed_activities: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(ActivityRow {
                id: row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                started_at: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                ended_at: row
                    .get(2)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            });
        }
        Ok(results)
    }

    /// Insert a capture reference.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_capture_ref(
        &self,
        source_id: &str,
        activity_id: &str,
        snapshot_id: Option<&str>,
        app_name: &str,
        window_title: &str,
        timestamp: i64,
        source: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO capture_refs (source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            libsql::params![source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source],
        ).await.map_err(|e| OriginError::VectorDb(format!("insert_capture_ref: {}", e)))?;
        Ok(())
    }

    /// Get unpackaged captures for a given activity.
    pub async fn get_unpackaged_captures(
        &self,
        activity_id: &str,
    ) -> Result<Vec<CaptureRefRow>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source FROM capture_refs WHERE activity_id = ?1 AND snapshot_id IS NULL ORDER BY timestamp",
            [activity_id],
        ).await.map_err(|e| OriginError::VectorDb(format!("get_unpackaged_captures: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(CaptureRefRow {
                source_id: row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                activity_id: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                snapshot_id: row.get(2).unwrap_or(None),
                app_name: row
                    .get(3)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                window_title: row
                    .get(4)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                timestamp: row
                    .get(5)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                source: row
                    .get(6)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            });
        }
        Ok(results)
    }

    /// Mark captures as packaged into a snapshot.
    pub async fn mark_captures_packaged(
        &self,
        source_ids: &[&str],
        snapshot_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("mark_packaged begin: {}", e)))?;
        for source_id in source_ids {
            conn.execute(
                "UPDATE capture_refs SET snapshot_id = ?1 WHERE source_id = ?2",
                libsql::params![snapshot_id, *source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("mark_packaged update: {}", e)))?;
        }
        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("mark_packaged commit: {}", e)))?;
        Ok(())
    }

    /// Check if an activity has any unpackaged captures.
    pub async fn has_unpackaged_captures(&self, activity_id: &str) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM capture_refs WHERE activity_id = ?1 AND snapshot_id IS NULL LIMIT 1",
                [activity_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("has_unpackaged: {}", e)))?;
        let has = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .is_some();
        Ok(has)
    }

    /// Get captures belonging to a snapshot.
    pub async fn get_captures_for_snapshot(
        &self,
        snapshot_id: &str,
    ) -> Result<Vec<CaptureRefRow>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source FROM capture_refs WHERE snapshot_id = ?1 ORDER BY timestamp",
            [snapshot_id],
        ).await.map_err(|e| OriginError::VectorDb(format!("get_captures_for_snapshot: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(CaptureRefRow {
                source_id: row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                activity_id: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                snapshot_id: row.get(2).unwrap_or(None),
                app_name: row
                    .get(3)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                window_title: row
                    .get(4)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                timestamp: row
                    .get(5)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                source: row
                    .get(6)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            });
        }
        Ok(results)
    }

    // ==================== Session Snapshots ====================

    /// Insert a session snapshot.
    pub async fn insert_snapshot(&self, snap: &SessionSnapshotRow) -> Result<(), OriginError> {
        let now = chrono::Utc::now().timestamp();
        let apps_json = serde_json::to_string(&snap.primary_apps).unwrap_or_default();
        let tags_json = serde_json::to_string(&snap.tags).unwrap_or_default();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO session_snapshots (id, activity_id, started_at, ended_at, primary_apps, summary, tags, capture_count, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            libsql::params![snap.id.clone(), snap.activity_id.clone(), snap.started_at, snap.ended_at, apps_json, snap.summary.clone(), tags_json, snap.capture_count as i64, now],
        ).await.map_err(|e| OriginError::VectorDb(format!("insert_snapshot: {}", e)))?;
        Ok(())
    }

    /// Get recent session snapshots.
    pub async fn get_recent_snapshots(
        &self,
        limit: usize,
    ) -> Result<Vec<SessionSnapshotRow>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id, activity_id, started_at, ended_at, primary_apps, summary, tags, capture_count FROM session_snapshots ORDER BY started_at DESC LIMIT ?1",
            [limit as i64],
        ).await.map_err(|e| OriginError::VectorDb(format!("get_recent_snapshots: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let apps_json: String = row
                .get(4)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let tags_json: String = row
                .get(6)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push(SessionSnapshotRow {
                id: row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                activity_id: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                started_at: row
                    .get(2)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                ended_at: row
                    .get(3)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                primary_apps: serde_json::from_str(&apps_json).unwrap_or_default(),
                summary: row
                    .get(5)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                capture_count: row
                    .get::<i64>(7)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                    as usize,
            });
        }
        Ok(results)
    }

    /// Delete a session snapshot and unlink its captures.
    pub async fn delete_snapshot(&self, snapshot_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE capture_refs SET snapshot_id = NULL WHERE snapshot_id = ?1",
            [snapshot_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_snapshot unlink: {}", e)))?;
        conn.execute("DELETE FROM session_snapshots WHERE id = ?1", [snapshot_id])
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_snapshot: {}", e)))?;
        Ok(())
    }

    /// Update a snapshot's summary and tags (called by LLM after synthesis).
    pub async fn update_snapshot_summary(
        &self,
        snapshot_id: &str,
        summary: &str,
        tags: &[String],
    ) -> Result<(), OriginError> {
        let tags_json = serde_json::to_string(tags).unwrap_or_default();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE session_snapshots SET summary = ?1, tags = ?2 WHERE id = ?3",
            libsql::params![summary, tags_json, snapshot_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_snapshot_summary: {}", e)))?;
        Ok(())
    }

    /// Delete session data (activities, capture_refs, snapshots) overlapping a time range.
    pub async fn delete_session_by_time_range(
        &self,
        start: i64,
        end: i64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        // Find overlapping activity IDs
        let mut rows = conn
            .query(
                "SELECT id FROM activities WHERE started_at <= ?2 AND ended_at >= ?1",
                libsql::params![start, end],
            )
            .await
            .map_err(|e| {
                OriginError::VectorDb(format!("delete_session_by_time_range query: {}", e))
            })?;
        let mut ids = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            ids.push(id);
        }
        drop(rows);

        for id in &ids {
            conn.execute(
                "UPDATE capture_refs SET snapshot_id = NULL WHERE activity_id = ?1 AND snapshot_id IS NOT NULL",
                [id.as_str()],
            ).await.map_err(|e| OriginError::VectorDb(format!("delete_session unlink: {}", e)))?;
            conn.execute(
                "DELETE FROM session_snapshots WHERE activity_id = ?1",
                [id.as_str()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_session snapshots: {}", e)))?;
            conn.execute(
                "DELETE FROM capture_refs WHERE activity_id = ?1",
                [id.as_str()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_session captures: {}", e)))?;
            conn.execute("DELETE FROM activities WHERE id = ?1", [id.as_str()])
                .await
                .map_err(|e| OriginError::VectorDb(format!("delete_session activity: {}", e)))?;
        }
        Ok(())
    }

    // ==================== Briefing Cache ====================

    /// Get the cached briefing (content, generated_at, memory_count).
    pub async fn get_cached_briefing(&self) -> Result<Option<(String, i64, u64)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT content, generated_at, memory_count FROM briefing_cache WHERE id = 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_cached_briefing: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let content: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let generated_at: i64 = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let memory_count: u64 = row.get::<u64>(2).unwrap_or(0);
            Ok(Some((content, generated_at, memory_count)))
        } else {
            Ok(None)
        }
    }

    /// Insert or replace the single briefing cache row.
    pub async fn upsert_briefing_cache(
        &self,
        content: &str,
        memory_count: u64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO briefing_cache (id, content, generated_at, memory_count) VALUES (1, ?1, ?2, ?3)",
            libsql::params![content, now, memory_count as i64],
        ).await.map_err(|e| OriginError::VectorDb(format!("upsert_briefing_cache: {}", e)))?;
        Ok(())
    }

    // ===== Narrative Cache =====

    /// Fetch confirmed memories of profile types (identity, preference, goal) for narrative.
    pub async fn get_memories_for_narrative(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::narrative::NarrativeMemory>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, title, content, memory_type FROM memories \
                 WHERE source = 'memory' AND confirmed = 1 AND chunk_index = 0 \
                   AND memory_type IN ('identity', 'preference', 'goal') \
                 ORDER BY last_modified DESC LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memories_for_narrative: {}", e)))?;

        let mut memories = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            memories.push(crate::narrative::NarrativeMemory {
                source_id: row.get::<String>(0).unwrap_or_default(),
                title: row.get::<String>(1).unwrap_or_default(),
                content: row.get::<String>(2).unwrap_or_default(),
                memory_type: row.get::<String>(3).unwrap_or_default(),
            });
        }
        Ok(memories)
    }

    /// Count confirmed narrative-eligible memories.
    pub async fn get_narrative_memory_count(&self) -> Result<u64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND confirmed = 1 AND chunk_index = 0 \
                   AND memory_type IN ('identity', 'preference', 'decision', 'goal')",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_narrative_memory_count: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(row.get::<u64>(0).unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    /// Read cached narrative.
    pub async fn get_cached_narrative(&self) -> Result<Option<(String, i64, u64)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT content, generated_at, memory_count FROM narrative_cache WHERE id = 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_cached_narrative: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let content: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let generated_at: i64 = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let memory_count: u64 = row.get::<u64>(2).unwrap_or(0);
            Ok(Some((content, generated_at, memory_count)))
        } else {
            Ok(None)
        }
    }

    /// Insert or replace the single narrative cache row.
    pub async fn upsert_narrative_cache(
        &self,
        content: &str,
        memory_count: u64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR REPLACE INTO narrative_cache (id, content, generated_at, memory_count) VALUES (1, ?1, ?2, ?3)",
            libsql::params![content, now, memory_count as i64],
        ).await.map_err(|e| OriginError::VectorDb(format!("upsert_narrative_cache: {}", e)))?;
        Ok(())
    }

    /// Count total distinct memory source_ids.
    pub async fn get_memory_count(&self) -> Result<u64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memory_count: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Ok(row.get::<u64>(0).unwrap_or(0))
        } else {
            Ok(0)
        }
    }

    /// Get briefing stats: dominant domain, primary agent, new-today count.
    pub async fn get_briefing_stats(
        &self,
        cutoff: i64,
    ) -> Result<crate::briefing::BriefingStats, OriginError> {
        let conn = self.conn.lock().await;
        let today_start = chrono::Utc::now()
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();

        // New memories today
        let mut rows = conn
            .query(
                "SELECT COUNT(DISTINCT source_id) FROM memories \
                 WHERE source = 'memory' AND last_modified > ?1 AND chunk_index = 0",
                libsql::params![today_start],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("briefing_stats new_today: {}", e)))?;
        let new_today = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<u64>(0).unwrap_or(0)
        } else {
            0
        };

        // Dominant domain (most frequent in last 48h)
        let mut rows = conn
            .query(
                "SELECT domain, COUNT(*) as cnt FROM memories \
                 WHERE source = 'memory' AND last_modified > ?1 AND chunk_index = 0 \
                   AND domain IS NOT NULL AND domain != '' \
                 GROUP BY domain ORDER BY cnt DESC LIMIT 1",
                libsql::params![cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("briefing_stats domain: {}", e)))?;
        let dominant_domain = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<String>(0).ok()
        } else {
            None
        };

        // Primary agent (most frequent in last 48h)
        let mut rows = conn
            .query(
                "SELECT source_agent, COUNT(*) as cnt FROM memories \
                 WHERE source = 'memory' AND last_modified > ?1 AND chunk_index = 0 \
                   AND source_agent IS NOT NULL AND source_agent != '' \
                 GROUP BY source_agent ORDER BY cnt DESC LIMIT 1",
                libsql::params![cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("briefing_stats agent: {}", e)))?;
        let primary_agent = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            row.get::<String>(0).ok()
        } else {
            None
        };

        Ok(crate::briefing::BriefingStats {
            dominant_domain,
            primary_agent,
            new_today,
        })
    }

    /// Get recent memories for briefing generation (all types, sorted by recency).
    pub async fn get_recent_memories_for_briefing(
        &self,
        cutoff: i64,
        limit: usize,
    ) -> Result<Vec<crate::briefing::BriefingMemory>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT title, content, COALESCE(memory_type, 'observation'), domain, last_modified
             FROM memories
             WHERE source = 'memory' AND chunk_index = 0 AND last_modified > ?1
               AND is_recap = 0
             ORDER BY last_modified DESC
             LIMIT ?2",
                libsql::params![cutoff, limit as i64],
            )
            .await
            .map_err(|e| {
                OriginError::VectorDb(format!("get_recent_memories_for_briefing: {}", e))
            })?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(crate::briefing::BriefingMemory {
                title: row.get(0).unwrap_or_default(),
                content: row.get(1).unwrap_or_default(),
                memory_type: row.get(2).unwrap_or_else(|_| "observation".to_string()),
                domain: row.get::<Option<String>>(3).unwrap_or(None),
                last_modified: row.get(4).unwrap_or(0),
            });
        }
        Ok(results)
    }

    /// Get confirmed memories of a specific type within the cutoff window.
    pub async fn get_memories_by_type_confirmed(
        &self,
        memory_type: &str,
        cutoff: i64,
    ) -> Result<Vec<crate::briefing::BriefingMemory>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT title, content, COALESCE(memory_type, 'observation'), domain, last_modified
             FROM memories
             WHERE source = 'memory' AND chunk_index = 0
               AND memory_type = ?1 AND confirmed = 1
               AND last_modified > ?2
             ORDER BY last_modified DESC
             LIMIT 10",
                libsql::params![memory_type, cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_memories_by_type_confirmed: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(crate::briefing::BriefingMemory {
                title: row.get(0).unwrap_or_default(),
                content: row.get(1).unwrap_or_default(),
                memory_type: row.get(2).unwrap_or_else(|_| memory_type.to_string()),
                domain: row.get::<Option<String>>(3).unwrap_or(None),
                last_modified: row.get(4).unwrap_or(0),
            });
        }
        Ok(results)
    }

    /// Get pending contradiction items from the refinement queue.
    pub async fn get_pending_contradiction_items(
        &self,
    ) -> Result<Vec<crate::briefing::ContradictionItem>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id, source_ids FROM refinement_queue WHERE action = 'detect_contradiction' AND status = 'awaiting_review'",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("get_pending_contradictions: {}", e)))?;

        let mut items = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let source_ids_json: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let source_ids: Vec<String> =
                serde_json::from_str(&source_ids_json).unwrap_or_default();

            if source_ids.len() < 2 {
                continue;
            }

            let new_source_id = source_ids[0].clone();
            let existing_source_id = source_ids[1].clone();

            // Fetch content for both source memories — must drop conn first since get_chunk_content needs it
            // Instead, inline the queries here to keep the same connection lock
            let mut new_rows = conn
                .query(
                    "SELECT content FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                    [new_source_id.as_str()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("contradiction new content: {}", e)))?;
            let new_content = if let Some(r) = new_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                r.get::<String>(0).unwrap_or_default()
            } else {
                continue; // skip if source memory not found
            };

            let mut existing_rows = conn
                .query(
                    "SELECT content FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                    [existing_source_id.as_str()],
                )
                .await
                .map_err(|e| {
                    OriginError::VectorDb(format!("contradiction existing content: {}", e))
                })?;
            let existing_content = if let Some(r) = existing_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                r.get::<String>(0).unwrap_or_default()
            } else {
                continue; // skip if source memory not found
            };

            items.push(crate::briefing::ContradictionItem {
                id,
                new_content,
                existing_content,
                new_source_id,
                existing_source_id,
            });
        }
        Ok(items)
    }

    /// Return the subset of `candidate_ids` that currently have an unresolved
    /// contradiction in `refinement_queue`.
    ///
    /// Signal used by the home's NeedsReview badge.
    pub async fn pending_review_memory_ids(
        &self,
        candidate_ids: &[String],
    ) -> Result<std::collections::HashSet<String>, OriginError> {
        use std::collections::HashSet;
        if candidate_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let conn = self.conn.lock().await;
        let stmt = conn
            .prepare(
                "SELECT source_ids FROM refinement_queue \
                 WHERE action = 'detect_contradiction' AND status = 'awaiting_review'",
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pending_review prepare: {e}")))?;
        let mut rows = stmt
            .query(())
            .await
            .map_err(|e| OriginError::VectorDb(format!("pending_review query: {e}")))?;
        let candidate_set: HashSet<&String> = candidate_ids.iter().collect();
        let mut flagged: HashSet<String> = HashSet::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("pending_review next: {e}")))?
        {
            let raw: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(format!("pending_review get: {e}")))?;
            let ids: Vec<String> = serde_json::from_str(&raw).unwrap_or_default();
            for id in ids {
                if candidate_set.contains(&id) {
                    flagged.insert(id);
                }
            }
        }
        Ok(flagged)
    }

    // ==================== Same-Type Memory Lookup ====================

    /// Find confirmed memories of the same type and domain.
    /// Returns (source_id, structured_fields, content) tuples.
    /// Excludes recaps and the given source_id (self).
    pub async fn find_same_type_memories(
        &self,
        exclude_source_id: &str,
        memory_type: &str,
        domain: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, Option<String>, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let (sql, params): (String, Vec<libsql::Value>) = if let Some(d) = domain {
            (
                "SELECT source_id, structured_fields, content FROM memories
                 WHERE source = 'memory' AND memory_type = ?1 AND domain = ?2
                 AND source_id != ?3 AND is_recap = 0
                 AND chunk_index = 0
                 ORDER BY last_modified DESC LIMIT ?4"
                    .to_string(),
                vec![
                    libsql::Value::Text(memory_type.to_string()),
                    libsql::Value::Text(d.to_string()),
                    libsql::Value::Text(exclude_source_id.to_string()),
                    libsql::Value::Integer(limit as i64),
                ],
            )
        } else {
            (
                "SELECT source_id, structured_fields, content FROM memories
                 WHERE source = 'memory' AND memory_type = ?1 AND domain IS NULL
                 AND source_id != ?2 AND is_recap = 0
                 AND chunk_index = 0
                 ORDER BY last_modified DESC LIMIT ?3"
                    .to_string(),
                vec![
                    libsql::Value::Text(memory_type.to_string()),
                    libsql::Value::Text(exclude_source_id.to_string()),
                    libsql::Value::Integer(limit as i64),
                ],
            )
        };
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("find_same_type_memories: {}", e)))?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let source_id: String = row.get(0).unwrap_or_default();
            let sf: Option<String> = row.get::<Option<String>>(1).unwrap_or(None);
            let content: String = row.get(2).unwrap_or_default();
            results.push((source_id, sf, content));
        }
        Ok(results)
    }

    // ==================== Refinement Queue ====================

    /// Get source_ids from active proposals (pending/applied) to prevent re-queuing.
    /// Dismissed proposals are excluded so memories can be re-tried with improved prompts.
    pub async fn get_all_proposal_source_ids(
        &self,
    ) -> Result<std::collections::HashSet<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT source_ids FROM refinement_queue WHERE status IN ('pending', 'awaiting_review', 'auto_applied')",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("get_proposal_ids: {}", e)))?;
        let mut ids = std::collections::HashSet::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let json: String = row.get(0).unwrap_or_default();
            if let Ok(source_ids) = serde_json::from_str::<Vec<String>>(&json) {
                ids.extend(source_ids);
            }
        }
        Ok(ids)
    }

    /// Insert a refinement proposal into the queue.
    pub async fn insert_refinement_proposal(
        &self,
        id: &str,
        action: &str,
        source_ids: &[String],
        payload: Option<&str>,
        confidence: f64,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let source_ids_json = serde_json::to_string(source_ids).unwrap_or_default();
        conn.execute(
            "INSERT OR REPLACE INTO refinement_queue (id, action, source_ids, payload, confidence) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![id, action, source_ids_json, payload, confidence],
        ).await.map_err(|e| OriginError::VectorDb(format!("insert_refinement: {}", e)))?;
        Ok(())
    }

    /// Get all pending/awaiting_review refinement proposals.
    pub async fn get_pending_refinements(&self) -> Result<Vec<RefinementProposal>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id, action, source_ids, payload, confidence, status, created_at FROM refinement_queue WHERE status IN ('pending', 'awaiting_review') ORDER BY created_at",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("get_pending: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_ids_json: String = row
                .get(2)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let source_ids: Vec<String> =
                serde_json::from_str(&source_ids_json).unwrap_or_default();
            results.push(RefinementProposal {
                id: row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                action: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                source_ids,
                payload: row.get(3).unwrap_or(None),
                confidence: row
                    .get(4)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                status: row
                    .get(5)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
                created_at: row
                    .get(6)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            });
        }
        Ok(results)
    }

    /// One-time cleanup: remove stale dedup_merge proposals from the v1 pipeline.
    pub async fn clear_dedup_merge_proposals(&self) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM refinement_queue WHERE action = 'dedup_merge'",
            (),
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("clear_dedup_merge_proposals: {}", e)))?;
        Ok(())
    }

    /// Resolve a refinement proposal with a new status.
    pub async fn resolve_refinement(&self, id: &str, status: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE refinement_queue SET status = ?1, resolved_at = datetime('now') WHERE id = ?2",
            libsql::params![status, id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("resolve_refinement: {}", e)))?;
        Ok(())
    }

    /// Dismiss all `detect_contradiction / awaiting_review` refinement queue rows that
    /// reference `source_id` in their `source_ids` JSON array.
    ///
    /// Used by the home-screen "dismiss contradiction" inline action.
    pub async fn dismiss_contradiction_for_source(
        &self,
        source_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        // JSON_EACH lets SQLite iterate the source_ids array so we don't need
        // application-side filtering, and we avoid the substring false-positive
        // risk of a plain LIKE search.
        conn.execute(
            "UPDATE refinement_queue
             SET status = 'dismissed', resolved_at = datetime('now')
             WHERE action = 'detect_contradiction'
               AND status = 'awaiting_review'
               AND id IN (
                   SELECT rq.id FROM refinement_queue rq, json_each(rq.source_ids) je
                   WHERE je.value = ?1
               )",
            libsql::params![source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("dismiss_contradiction_for_source: {}", e)))?;
        Ok(())
    }

    /// Find domains with N+ decayed memories below a confidence threshold (consolidation candidates).
    pub async fn get_consolidation_candidates(
        &self,
        _threshold: f64,
        min_count: i64,
    ) -> Result<Vec<ConsolidationCandidate>, OriginError> {
        let conn = self.conn.lock().await;
        // Gate: unconfirmed + unpinned only. No confidence threshold —
        // consolidation turns multiple okay memories into one good one,
        // no need to wait for decay first.
        let mut rows = conn.query(
            "SELECT domain, COUNT(*) as cnt FROM memories
             WHERE source = 'memory' AND (confirmed = 0 OR confirmed IS NULL) AND (pinned = 0 OR pinned IS NULL) AND domain IS NOT NULL AND chunk_index = 0
             AND supersede_mode != 'archive' AND source_id NOT LIKE 'merged_%'
             GROUP BY domain HAVING cnt >= ?1",
            libsql::params![min_count],
        ).await.map_err(|e| OriginError::VectorDb(format!("consolidation candidates: {}", e)))?;
        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(ConsolidationCandidate {
                domain: row.get(0).unwrap_or(None),
                count: row
                    .get(1)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            });
        }
        Ok(results)
    }

    // ==================== Distillation Clustering ====================

    /// Find clusters of semantically related memories for distillation.
    /// Groups by entity_id first, then sub-clusters by vector similarity.
    pub async fn find_distillation_clusters(
        &self,
        similarity_threshold: f64,
        min_size: usize,
        max_clusters: usize,
        token_limit: usize,
    ) -> Result<Vec<DistillationCluster>, OriginError> {
        let conn = self.conn.lock().await;

        // No covered_ids exclusion — memories can participate in multiple concepts.
        // Dedup happens in distill_concepts via Jaccard overlap check.

        let mut rows = conn.query(
            "SELECT m.source_id, m.content, m.entity_id, m.domain, m.embedding, e.community_id, e.name \
             FROM memories m \
             LEFT JOIN entities e ON m.entity_id = e.id \
             WHERE m.source = 'memory' AND m.chunk_index = 0 \
               AND (m.confirmed = 0 OR m.confirmed IS NULL) \
               AND (m.pinned = 0 OR m.pinned IS NULL) \
               AND m.supersede_mode <> 'archive' \
               AND m.source_id NOT LIKE 'merged_%' \
               AND m.source_id NOT LIKE 'recap_%' \
               AND m.is_recap = 0 \
               AND m.embedding IS NOT NULL \
             ORDER BY m.entity_id, m.domain, m.last_modified DESC",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("distillation fetch: {}", e)))?;

        let mut memories: Vec<ClusterMemRow> = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let entity_id: Option<String> = row.get(2).unwrap_or(None);
            let domain: Option<String> = row.get(3).unwrap_or(None);
            // Parse embedding from F32_BLOB — stored as little-endian f32 bytes
            let emb_bytes: Vec<u8> = row
                .get(4)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let embedding: Vec<f32> = emb_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let community_id: Option<u32> = row.get::<u32>(5).ok();
            let entity_name: Option<String> = row.get(6).unwrap_or(None);
            memories.push(ClusterMemRow {
                source_id,
                content,
                entity_id,
                entity_name,
                community_id,
                domain,
                embedding,
            });
        }
        drop(rows);
        drop(conn);

        log::info!(
            "[distill] found {} eligible memories for clustering",
            memories.len()
        );
        if memories.is_empty() {
            return Ok(vec![]);
        }

        // Group by community_id (preferred) or entity_id (fallback)
        let mut community_groups: std::collections::HashMap<u32, Vec<usize>> =
            std::collections::HashMap::new();
        let mut entity_groups: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        let mut unlinked: Vec<usize> = Vec::new();

        for (i, mem) in memories.iter().enumerate() {
            if let Some(cid) = mem.community_id {
                // Group by community_id (preferred — graph-aware grouping)
                community_groups.entry(cid).or_default().push(i);
            } else if let Some(ref eid) = mem.entity_id {
                // Fallback: group by entity_id
                entity_groups.entry(eid.clone()).or_default().push(i);
            } else {
                unlinked.push(i);
            }
        }

        let mut clusters: Vec<DistillationCluster> = Vec::new();

        // Sub-cluster within each community group by vector similarity
        for indices in community_groups.values() {
            let sub = cluster_by_similarity(&memories, indices, similarity_threshold);
            for group in sub {
                if group.len() >= min_size {
                    let cluster = build_distillation_cluster(&memories, &group);
                    let split = sub_cluster_by_tokens(&memories, cluster, token_limit);
                    clusters.extend(split);
                }
            }
        }

        // Sub-cluster within each entity group by vector similarity (no community_id)
        for indices in entity_groups.values() {
            let sub = cluster_by_similarity(&memories, indices, similarity_threshold);
            for group in sub {
                if group.len() >= min_size {
                    let cluster = build_distillation_cluster(&memories, &group);
                    let split = sub_cluster_by_tokens(&memories, cluster, token_limit);
                    clusters.extend(split);
                }
            }
        }

        // Cluster unlinked memories by vector similarity
        if unlinked.len() >= min_size {
            let sub = cluster_by_similarity(&memories, &unlinked, similarity_threshold);
            for group in sub {
                if group.len() >= min_size {
                    let cluster = build_distillation_cluster(&memories, &group);
                    let split = sub_cluster_by_tokens(&memories, cluster, token_limit);
                    clusters.extend(split);
                }
            }
        }

        log::info!(
            "[distill] community_groups={}, entity_groups={}, unlinked={}, clusters_found={}",
            community_groups.len(),
            entity_groups.len(),
            unlinked.len(),
            clusters.len()
        );

        // Sort by size descending (larger clusters = more value)
        clusters.sort_by_key(|c| std::cmp::Reverse(c.source_ids.len()));
        clusters.truncate(max_clusters);

        Ok(clusters)
    }

    // ==================== Recent Memories (for recap generation) ====================

    /// Get recent non-recap memories since a given epoch timestamp.
    /// Returns (source_id, content, domain, last_modified) tuples. Used by the refinery to synthesize recaps.
    pub async fn get_recent_memories_for_recap(
        &self,
        since_epoch: i64,
        limit: usize,
    ) -> Result<Vec<(String, String, Option<String>, i64)>, OriginError> {
        let conn = self.conn.lock().await;
        // Use created_at (immutable, set at insert) instead of last_modified
        // (updated by enrichment) to avoid pulling old memories into recaps.
        let mut rows = conn.query(
            "SELECT source_id, content, domain, COALESCE(created_at, last_modified) FROM memories
             WHERE source = 'memory'
               AND is_recap = 0
               AND source_id NOT LIKE 'merged_%'
               AND chunk_index = 0
               AND COALESCE(created_at, last_modified) > ?1
             ORDER BY COALESCE(created_at, last_modified) DESC
             LIMIT ?2",
            libsql::params![since_epoch, limit as i64],
        ).await.map_err(|e| OriginError::VectorDb(format!("get_recent_for_recap: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row.get::<String>(1).unwrap_or_default();
            let domain: Option<String> = row.get(2).unwrap_or(None);
            let last_modified: i64 = row
                .get(3)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((source_id, content, domain, last_modified));
        }
        Ok(results)
    }

    /// Check if a recap already exists covering a time window (prevents duplicate recaps).
    pub async fn has_recap_since(&self, since_epoch: i64) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM memories
             WHERE source = 'memory'
               AND is_recap = 1
               AND COALESCE(created_at, last_modified) > ?1
             LIMIT 1",
                [since_epoch],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("has_recap_since: {}", e)))?;
        let has = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .is_some();
        Ok(has)
    }

    /// Check if a recap already exists covering a specific burst range.
    /// A recap covers a burst if its last_modified falls within [burst_start, burst_end + 60].
    pub async fn has_recap_covering_range(
        &self,
        burst_start: i64,
        burst_end: i64,
    ) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT 1 FROM memories WHERE source = 'memory' AND is_recap = 1
               AND COALESCE(created_at, last_modified) >= ?1 AND COALESCE(created_at, last_modified) <= ?2 + 60 LIMIT 1",
            libsql::params![burst_start, burst_end],
        ).await.map_err(|e| OriginError::VectorDb(format!("has_recap_covering_range: {}", e)))?;
        let has = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .is_some();
        Ok(has)
    }

    /// Get memories that have no entity_id link (for reweave phase).
    pub async fn get_unlinked_memories(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories
             WHERE source = 'memory'
               AND entity_id IS NULL
               AND is_recap = 0
               AND chunk_index = 0
             ORDER BY last_modified DESC
             LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_unlinked_memories: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row.get::<String>(1).unwrap_or_default();
            results.push((source_id, content));
        }
        Ok(results)
    }

    /// Get recent decision memories for decision log generation.
    pub async fn get_recent_decisions_for_recap(
        &self,
        since_epoch: i64,
        limit: usize,
    ) -> Result<Vec<(String, String, Option<String>, i64)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT source_id, content, domain, COALESCE(created_at, last_modified) FROM memories
             WHERE source = 'memory'
               AND memory_type = 'decision'
               AND is_recap = 0
               AND source_id NOT LIKE 'merged_%'
               AND chunk_index = 0
               AND COALESCE(created_at, last_modified) > ?1
             ORDER BY COALESCE(created_at, last_modified) DESC
             LIMIT ?2",
            libsql::params![since_epoch, limit as i64],
        ).await.map_err(|e| OriginError::VectorDb(format!("get_recent_decisions: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row.get::<String>(1).unwrap_or_default();
            let domain: Option<String> = row.get(2).unwrap_or(None);
            let last_modified: i64 = row
                .get(3)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((source_id, content, domain, last_modified));
        }
        Ok(results)
    }

    /// Get decision memories that lack structured_fields (for backfill).
    pub async fn get_decisions_without_structured_fields(
        &self,
        limit: usize,
    ) -> Result<Vec<(String, String)>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories \
                 WHERE source = 'memory' AND memory_type = 'decision' AND chunk_index = 0 \
                   AND confirmed != 0 AND (structured_fields IS NULL OR structured_fields = '') \
                 ORDER BY last_modified DESC LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_decisions_without_sf: {}", e)))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row.get(0).unwrap_or_default();
            let content: String = row.get(1).unwrap_or_default();
            results.push((source_id, content));
        }
        Ok(results)
    }

    /// Update structured_fields for an existing memory.
    pub async fn update_structured_fields(
        &self,
        source_id: &str,
        structured_fields_json: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE memories SET structured_fields = ?1, needs_reembed = 1 \
             WHERE source_id = ?2 AND chunk_index = 0",
            libsql::params![structured_fields_json, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_structured_fields: {}", e)))?;
        Ok(())
    }

    /// Apply deferred classify + extract results to a memory in a minimal
    /// number of SQL statements. Caller passes whatever fields the LLM
    /// produced; `None` values leave the existing column untouched.
    ///
    /// Used by the async enrichment path in `handle_store_memory` after the
    /// memory has been persisted with placeholder values (memory_type="fact",
    /// no domain/quality, no extracted fields). Runs as two UPDATEs — one
    /// across all chunks for row-level metadata (memory_type/domain/quality/
    /// supersede_mode), one targeted at chunk_index=0 for extraction fields.
    #[allow(clippy::too_many_arguments)]
    pub async fn apply_enrichment(
        &self,
        source_id: &str,
        memory_type: &str,
        domain: Option<&str>,
        quality: Option<&str>,
        supersede_mode: &str,
        structured_fields: Option<&str>,
        retrieval_cue: Option<&str>,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        // Row-level metadata — mirrored across all chunks. `COALESCE(?, col)`
        // keeps the existing column when the caller passes `None`, so agents
        // that supplied e.g. a domain at store time don't get it overwritten.
        conn.execute(
            "UPDATE memories SET
                memory_type = ?1,
                domain = COALESCE(?2, domain),
                quality = COALESCE(?3, quality),
                supersede_mode = ?4
             WHERE source_id = ?5",
            libsql::params![memory_type, domain, quality, supersede_mode, source_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("apply_enrichment: {}", e)))?;

        // Extraction fields live on chunk_index=0 only. Bump enrichment_status
        // so the re-embed pass picks up the richer structured content.
        if structured_fields.is_some() || retrieval_cue.is_some() {
            conn.execute(
                "UPDATE memories SET
                    structured_fields = COALESCE(?1, structured_fields),
                    retrieval_cue = COALESCE(?2, retrieval_cue),
                    needs_reembed = CASE WHEN ?1 IS NOT NULL THEN 1 ELSE needs_reembed END
                 WHERE source_id = ?3 AND chunk_index = 0",
                libsql::params![structured_fields, retrieval_cue, source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("apply_enrichment (chunk 0): {}", e)))?;
        }
        Ok(())
    }

    // ==================== Merge / Queue Processing ====================

    /// Get content for multiple memories by source_id.
    pub async fn get_memory_contents(
        &self,
        source_ids: &[String],
    ) -> Result<Vec<String>, OriginError> {
        if source_ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().await;
        let mut results = vec![];
        for source_id in source_ids {
            let mut rows = conn
                .query(
                    "SELECT content FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                    [source_id.as_str()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("get_memory_contents: {}", e)))?;
            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let content: String = row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                results.push(content);
            }
        }
        Ok(results)
    }

    /// Get the highest stability tier among a set of memories.
    pub async fn get_highest_tier(
        &self,
        source_ids: &[String],
    ) -> Result<crate::sources::StabilityTier, OriginError> {
        let conn = self.conn.lock().await;
        let mut highest = crate::sources::StabilityTier::Ephemeral;
        for source_id in source_ids {
            let mut rows = conn
                .query(
                    "SELECT memory_type FROM memories WHERE source_id = ?1 LIMIT 1",
                    [source_id.as_str()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("get_highest_tier: {}", e)))?;
            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let mt: Option<String> = row.get(0).unwrap_or(None);
                let tier = crate::sources::stability_tier(mt.as_deref());
                // Protected > Standard > Ephemeral
                match (&highest, &tier) {
                    (_, crate::sources::StabilityTier::Protected) => highest = tier,
                    (
                        crate::sources::StabilityTier::Ephemeral,
                        crate::sources::StabilityTier::Standard,
                    ) => highest = tier,
                    _ => {}
                }
            }
        }
        Ok(highest)
    }

    /// Apply a merge: create a new memory with merged content that supersedes the originals.
    pub async fn apply_merge(
        &self,
        source_ids: &[String],
        merged_content: &str,
    ) -> Result<String, OriginError> {
        self.apply_merge_with_title(source_ids, merged_content, None)
            .await
    }

    /// Apply a merge with an optional short title. If title is None, uses first 80 chars of content.
    pub async fn apply_merge_with_title(
        &self,
        source_ids: &[String],
        merged_content: &str,
        title: Option<&str>,
    ) -> Result<String, OriginError> {
        // Get metadata from the first source memory
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT memory_type, domain, source_agent FROM memories WHERE source_id = ?1 LIMIT 1",
            [source_ids[0].as_str()],
        ).await.map_err(|e| OriginError::VectorDb(format!("apply_merge query: {}", e)))?;

        let (memory_type, domain, source_agent) = if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            (
                row.get::<Option<String>>(0).unwrap_or(None),
                row.get::<Option<String>>(1).unwrap_or(None),
                row.get::<Option<String>>(2).unwrap_or(None),
            )
        } else {
            (None, None, None)
        };
        drop(rows);
        drop(conn);

        // Create merged document
        let merged_id = format!("merged_{}", uuid::Uuid::new_v4());
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: merged_id.clone(),
            title: title
                .map(|t| t.to_string())
                .unwrap_or_else(|| merged_content.chars().take(80).collect()),
            summary: None,
            content: merged_content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
            memory_type,
            domain,
            source_agent: Some(source_agent.unwrap_or_else(|| "refinery".to_string())),
            confidence: Some(0.8), // Merged memories get a confidence boost
            confirmed: None,
            supersedes: Some(source_ids[0].clone()), // supersede chain to first original
            pending_revision: false,
            ..Default::default()
        };

        self.upsert_documents(vec![doc]).await?;

        // Distilled memories are atomic — must be exactly 1 chunk.
        // The chunker sometimes produces duplicates; clean up any extras.
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM memories WHERE source_id = ?1 AND source = 'memory' AND chunk_index > 0",
            [merged_id.as_str()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("apply_merge dedup memories: {}", e)))?;
        drop(conn);

        // Archive the original source memories so they're hidden from stream
        // Skip archiving merged memories — they're handled via supersedes chain
        let conn = self.conn.lock().await;
        for sid in source_ids {
            if !sid.starts_with("merged_") {
                conn.execute(
                    "UPDATE memories SET supersede_mode = 'archive' WHERE source_id = ?1 AND source = 'memory'",
                    [sid.as_str()],
                ).await.map_err(|e| OriginError::VectorDb(format!("apply_merge archive: {}", e)))?;
            }
        }
        drop(conn);

        Ok(merged_id)
    }

    // ==================== Decay Engine ====================

    /// Decay pass: update the effective_confidence column for all memories.
    /// Called by the refinery steep cycle (backstop every 2 hours).
    /// Returns the number of memories updated.
    pub async fn decay_update_confidence(&self) -> Result<u64, OriginError> {
        let now_epoch = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;

        // Fetch all memory rows with their decay parameters.
        // Skip 'confirmed' stability — those never decay.
        let mut rows = conn.query(
            "SELECT source_id, confidence, memory_type, confirmed, pinned, access_count, last_accessed, last_modified, stability
             FROM memories WHERE source = 'memory' AND COALESCE(stability, 'new') != 'confirmed'",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("decay query: {}", e)))?;

        let mut updates: Vec<(String, f64)> = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let confidence: f64 = row.get::<f64>(1).unwrap_or(0.5);
            let memory_type: Option<String> = row.get(2).unwrap_or(None);
            let confirmed: bool = row.get::<i64>(3).unwrap_or(0) != 0;
            let pinned: bool = row.get::<i64>(4).unwrap_or(0) != 0;
            let access_count: u64 = row.get::<i64>(5).unwrap_or(0) as u64;
            let last_accessed: Option<String> = row.get(6).unwrap_or(None);
            let last_modified_int: Option<i64> = row.get(7).unwrap_or(None);
            let stability: Option<String> = row.get(8).unwrap_or(None);

            let tier = crate::sources::stability_tier(memory_type.as_deref());
            let cfg = crate::tuning::ConfidenceConfig::default();
            let mut rate = crate::decay::decay_rate_for(&tier, confirmed, pinned, &cfg);

            // 'learned' stability decays at half the normal rate
            if stability.as_deref() == Some("learned") {
                rate *= 0.5;
            }

            // Use last_accessed if available, otherwise fall back to last_modified (as string)
            let days = if let Some(ref ts) = last_accessed {
                crate::decay::days_since(Some(ts.as_str()), now_epoch)
            } else if let Some(lm) = last_modified_int {
                let lm_str = lm.to_string();
                crate::decay::days_since(Some(&lm_str), now_epoch)
            } else {
                0.0
            };

            let eff = crate::decay::effective_confidence(confidence, rate, days, access_count);
            updates.push((source_id, eff));
        }
        drop(rows);

        // Batch update
        let count = updates.len() as u64;
        if !updates.is_empty() {
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("decay begin: {}", e)))?;
            for (source_id, eff) in &updates {
                conn.execute(
                    "UPDATE memories SET effective_confidence = ?1 WHERE source_id = ?2",
                    libsql::params![*eff, source_id.as_str()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("decay update: {}", e)))?;
            }
            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("decay commit: {}", e)))?;
        }

        Ok(count)
    }

    // ==================== Access Tracking ====================

    /// Flush buffered access counts to the database.
    /// Called by the access tracker timer every 60 seconds.
    pub async fn flush_access_counts(&self, source_ids: &[String]) -> Result<(), OriginError> {
        if source_ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("flush_access begin: {}", e)))?;
        for source_id in source_ids {
            conn.execute(
                "UPDATE memories SET access_count = COALESCE(access_count, 0) + 1, last_accessed = datetime('now') WHERE source_id = ?1",
                [source_id.as_str()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("flush_access update: {}", e)))?;
        }
        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("flush_access commit: {}", e)))?;
        Ok(())
    }

    /// Log individual access events for time-granular stats (today/week).
    pub async fn log_accesses(&self, source_ids: &[String]) -> Result<(), OriginError> {
        if source_ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("log_accesses begin: {}", e)))?;
        for sid in source_ids {
            conn.execute(
                "INSERT INTO access_log (source_id, accessed_at) VALUES (?1, ?2)",
                libsql::params![sid.as_str(), now],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("log_accesses insert: {}", e)))?;
        }
        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("log_accesses commit: {}", e)))?;
        Ok(())
    }

    /// Log an agent activity event (read, search, or refine).
    pub async fn log_agent_activity(
        &self,
        agent_name: &str,
        action: &str,
        memory_ids: &[String],
        query: Option<&str>,
        detail: &str,
    ) -> Result<(), OriginError> {
        // Canonicalize the agent name on write so `Claude Code`, `claude-code`,
        // `claude_code`, and `Claude code` all collapse into one filter entry
        // in the Activity view. Previously used `to_lowercase()` alone, which
        // left `"Claude Code"` as `"claude code"` (space, not hyphen) — not
        // matching the hyphenated form the CLI sends. Migration 31 backfilled
        // the history; this keeps new writes aligned with `agent_connections.name`.
        let agent_name_norm = canonicalize_agent_id(agent_name);
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        let ids_str = if memory_ids.is_empty() {
            None
        } else {
            Some(memory_ids.join(","))
        };
        conn.execute(
            "INSERT INTO agent_activity (timestamp, agent_name, action, memory_ids, query, detail)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![
                now,
                agent_name_norm,
                action,
                ids_str.as_deref().unwrap_or(""),
                query.unwrap_or(""),
                detail
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("log_agent_activity insert: {}", e)))?;
        Ok(())
    }

    /// Return up to `limit` most recent retrieval events joined to concept titles.
    /// Reads from `agent_activity` (populated by search/read handlers) and resolves
    /// comma-separated `memory_ids` to concept titles via `concepts.source_memory_ids`
    /// (JSON array; matched with quoted LIKE to avoid substring collisions —
    /// e.g. `mem_1` vs `mem_10`).
    pub async fn list_recent_retrievals(
        &self,
        limit: i64,
    ) -> Result<Vec<origin_types::RetrievalEvent>, OriginError> {
        // TODO(home-v2 #retrieval-perf): Mutex is held across the entire events
        // resolve loop (concept titles + memory snippets). At limit=10 events
        // with up to ~5 memory_ids each this is tolerable, but every concurrent
        // MemoryDB caller blocks for the duration. Consider: release lock between
        // events, or bulk-resolve all concept_ids + memory_snippets for all events
        // in 2 queries total.
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT timestamp, agent_name, query, memory_ids
                 FROM agent_activity
                 WHERE action IN ('search','read') AND memory_ids != ''
                 ORDER BY timestamp DESC LIMIT ?1",
                libsql::params![limit],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_recent_retrievals scan: {e}")))?;

        let mut raw: Vec<(i64, String, Option<String>, Vec<String>)> = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let ts_s: i64 = row.get(0).unwrap_or(0);
            let agent: String = row.get(1).unwrap_or_default();
            let q: String = row.get(2).unwrap_or_default();
            let ids_str: String = row.get(3).unwrap_or_default();
            let ids: Vec<String> = ids_str
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            let query = if q.is_empty() { None } else { Some(q) };
            raw.push((ts_s, agent, query, ids));
        }

        let mut events: Vec<origin_types::RetrievalEvent> = Vec::with_capacity(raw.len());
        for (ts_s, agent, query, ids) in raw {
            let mut titles: Vec<String> = Vec::new();
            let mut concept_ids: Vec<String> = Vec::new();
            // Track by concept id to avoid duplicates (title dedup could miss
            // concepts whose titles changed since the event was recorded).
            let mut seen_concept_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for id in &ids {
                let pattern = format!("%\"{}\"%", id);
                let mut t_rows = conn
                    .query(
                        "SELECT id, title FROM concepts
                         WHERE status = 'active' AND source_memory_ids LIKE ?1
                         LIMIT 5",
                        libsql::params![pattern],
                    )
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("list_recent_retrievals resolve: {e}"))
                    })?;
                while let Ok(Some(tr)) = t_rows.next().await {
                    let cid: String = tr.get(0).unwrap_or_default();
                    let title: String = tr.get(1).unwrap_or_default();
                    if !cid.is_empty() && !title.is_empty() && seen_concept_ids.insert(cid.clone())
                    {
                        titles.push(title);
                        concept_ids.push(cid);
                    }
                }
            }
            // Build memory_snippets in the same order as memory_ids.
            // One batch query per event — fetches the lowest chunk_index row for
            // each source_id in the set (avoids N+1 queries).
            // Prefer title if non-empty (truncated to 80 chars); otherwise use
            // the first 100 chars of content.  Missing ids are skipped silently
            // (same behaviour as the concept-title lookup above).
            // NOTE: snippets are populated BEFORE the skip guard so that events
            // with no matching concept but a valid memory are still surfaced via
            // memory_snippets (UI fallback).
            let mut snippets: Vec<String> = Vec::with_capacity(ids.len());
            if !ids.is_empty() {
                let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                let snippet_sql = format!(
                    "SELECT m.source_id, COALESCE(m.title, ''), COALESCE(m.content, '') \
                     FROM memories m \
                     WHERE m.source_id IN ({placeholders}) \
                       AND m.chunk_index = ( \
                           SELECT MIN(chunk_index) FROM memories m2 WHERE m2.source_id = m.source_id \
                       )"
                );
                let snippet_params: Vec<libsql::Value> = ids
                    .iter()
                    .map(|id| libsql::Value::Text(id.clone()))
                    .collect();
                let mut m_rows = conn
                    .query(&snippet_sql, snippet_params)
                    .await
                    .map_err(|e| {
                        OriginError::VectorDb(format!("list_recent_retrievals snippets: {e}"))
                    })?;
                let mut snippet_map: std::collections::HashMap<String, String> =
                    std::collections::HashMap::new();
                while let Ok(Some(mr)) = m_rows.next().await {
                    let source_id: String = mr.get(0).unwrap_or_default();
                    let title: String = mr.get(1).unwrap_or_default();
                    let content: String = mr.get(2).unwrap_or_default();
                    let snippet = if !title.is_empty() && !title_looks_garbage(&title) {
                        title.chars().take(80).collect::<String>()
                    } else if !content.is_empty() {
                        content.chars().take(100).collect::<String>()
                    } else if !title.is_empty() {
                        // Title is garbage AND content is empty — still show the title as a last resort.
                        title.chars().take(80).collect::<String>()
                    } else {
                        continue;
                    };
                    snippet_map.insert(source_id, snippet);
                }
                // Iterate ids in original order, preserving order and skipping unknown ids.
                for id in &ids {
                    if let Some(snippet) = snippet_map.remove(id) {
                        snippets.push(snippet);
                    }
                }
            }
            // Skip events where neither a concept title nor a memory snippet
            // could be resolved — nothing useful to show in the UI.
            if titles.is_empty() && snippets.is_empty() {
                continue;
            }
            events.push(origin_types::RetrievalEvent {
                timestamp_ms: ts_s.saturating_mul(1000),
                agent_name: agent,
                query,
                concept_titles: titles,
                concept_ids,
                memory_snippets: snippets,
            });
        }
        Ok(events)
    }

    /// Return up to `limit` most recent concept changes (created or revised),
    /// ordered by `last_modified` DESC. Only `status = 'active'` rows — archived
    /// or otherwise non-active concepts are excluded.
    ///
    /// Classification:
    /// - `Created` — `version == 1` AND `created_at == last_modified`
    ///   (no edits since creation).
    /// - `Revised` — `version > 1`, OR `version == 1` with `created_at !=
    ///   last_modified` (edited without a version bump).
    ///
    /// `Merged` is currently never emitted: the `concepts` schema has no
    /// merge-tracking column (e.g. `merged_into` / `superseded_by`). The
    /// `ConceptChangeKind::Merged` variant remains defined in `origin-types`
    /// so a later task can extend this method once such a column exists.
    pub async fn list_recent_changes(
        &self,
        limit: i64,
    ) -> Result<Vec<origin_types::ConceptChange>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, version, created_at, last_modified
                 FROM concepts
                 WHERE status = 'active'
                 ORDER BY last_modified DESC LIMIT ?1",
                libsql::params![limit],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_recent_changes scan: {e}")))?;

        let mut out = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let id: String = row.get(0).unwrap_or_default();
            let title: String = row.get(1).unwrap_or_default();
            let version: i64 = row.get(2).unwrap_or(1);
            let created_at: String = row.get(3).unwrap_or_default();
            let last_modified: String = row.get(4).unwrap_or_default();

            let change_kind = if version > 1 {
                origin_types::ConceptChangeKind::Revised
            } else if created_at == last_modified {
                origin_types::ConceptChangeKind::Created
            } else {
                origin_types::ConceptChangeKind::Revised
            };

            let changed_at_ms = chrono::DateTime::parse_from_rfc3339(&last_modified)
                .map(|dt| dt.timestamp_millis())
                .unwrap_or(0);

            out.push(origin_types::ConceptChange {
                concept_id: id,
                title,
                change_kind,
                changed_at_ms,
            });
        }
        Ok(out)
    }

    /// Return up to `limit` most-recently-modified memories, ordered newest first.
    ///
    /// `since_ms` is the caller's "last seen" watermark in **milliseconds**. The DB
    /// stores `created_at` / `last_modified` as Unix **seconds**, so we convert at
    /// the boundary.  `since_ms` drives badge derivation only — it never filters
    /// rows out.  The returned `timestamp_ms` is always in milliseconds.
    pub async fn list_recent_memories(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentActivityItem>, OriginError> {
        use origin_types::{ActivityKind, RecentActivityItem};

        // --- Phase A: fetch rows (drop conn guard before second async call) ---
        #[allow(clippy::type_complexity)]
        let rows_raw: Vec<(
            String,
            String,
            Option<String>,
            String,
            i64,
            Option<i64>,
            String,
            Option<String>,
        )> = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source_id, title, summary, content, \
                            created_at, last_modified, \
                            (SELECT CASE \
                                WHEN COUNT(es.source_id) = 0 THEN 'raw' \
                                WHEN SUM(CASE WHEN es.status = 'failed' OR es.status = 'abandoned' THEN 1 ELSE 0 END) = 0 THEN 'enriched' \
                                WHEN SUM(CASE WHEN es.status IN ('ok','skipped') THEN 1 ELSE 0 END) = 0 THEN 'enrichment_failed' \
                                ELSE 'enrichment_partial' \
                            END FROM enrichment_steps es WHERE es.source_id = memories.source_id) AS enrichment_status, \
                            entity_id \
                     FROM memories \
                     WHERE source = 'memory' AND chunk_index = 0 \
                       AND (supersede_mode IS NULL OR supersede_mode != 'archive') \
                     ORDER BY COALESCE(last_modified, created_at) DESC \
                     LIMIT ?1",
                    libsql::params![limit],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("list_recent_memories scan: {e}")))?;
            let mut out = Vec::new();
            while let Ok(Some(row)) = rows.next().await {
                let source_id: String = row.get(0).unwrap_or_default();
                let title: String = row.get(1).unwrap_or_default();
                let summary: Option<String> = row.get::<Option<String>>(2).unwrap_or(None);
                let content: String = row.get(3).unwrap_or_default();
                let created_at: i64 = row.get(4).unwrap_or(0);
                let last_modified: Option<i64> = row.get::<Option<i64>>(5).unwrap_or(None);
                let enrichment_status: String =
                    row.get(6).unwrap_or_else(|_| "enriched".to_string());
                let entity_id: Option<String> = row.get::<Option<String>>(7).unwrap_or(None);
                out.push((
                    source_id,
                    title,
                    summary,
                    content,
                    created_at,
                    last_modified,
                    enrichment_status,
                    entity_id,
                ));
            }
            out
        };
        // conn guard dropped here — safe to call pending_review_memory_ids next

        // --- Phase B: resolve NeedsReview flags ---
        let candidate_ids: Vec<String> = rows_raw.iter().map(|(id, ..)| id.clone()).collect();
        let flagged = self.pending_review_memory_ids(&candidate_ids).await?;

        // --- Phase C + D: derive badge + build output ---
        // Convert since_ms (milliseconds) → since_s (seconds) for DB-column comparisons.
        let since_s = since_ms.map(|ms| ms / 1000);

        let mut items = Vec::with_capacity(rows_raw.len());
        for (
            source_id,
            title,
            summary,
            content,
            created_at,
            last_modified,
            enrichment_status,
            entity_id,
        ) in rows_raw
        {
            let badge = derive_memory_badge(
                &source_id,
                created_at,
                last_modified,
                &enrichment_status,
                entity_id.as_deref(),
                since_s,
                &flagged,
            );

            // Snippet: prefer summary; fall back to first 100 chars of content (UTF-8 safe).
            let snippet: Option<String> = summary.filter(|s| !s.is_empty()).or_else(|| {
                let truncated: String = content.chars().take(100).collect();
                if truncated.is_empty() {
                    None
                } else {
                    Some(truncated)
                }
            });

            // timestamp_ms: DB stores seconds → convert to milliseconds.
            let ts_s = last_modified.unwrap_or(created_at);
            let timestamp_ms = (ts_s as u64).saturating_mul(1000);

            items.push(RecentActivityItem {
                kind: ActivityKind::Memory,
                id: source_id,
                title,
                snippet,
                timestamp_ms,
                badge,
            });
        }
        Ok(items)
    }

    /// Return up to `limit` most-recently-modified *unconfirmed* memories,
    /// ordered newest first.
    ///
    /// An "unconfirmed" memory is one whose `confirmed` column is 0 or NULL —
    /// i.e. the user has not explicitly approved it yet. These rows are
    /// surfaced on the home page's Worth-a-glance strip as items that benefit
    /// from a quick confirm/edit pass.
    ///
    /// Every returned item carries `badge = NeedsReview`: the whole point of
    /// this feed is that the caller is already asking "what needs review?"
    /// Using the existing `NeedsReview` variant (rather than adding a new
    /// `Unconfirmed`) keeps the Badge component + Worth-a-glance filter
    /// simple and consistent across contradiction-flagged items and
    /// unconfirmed-memory items.
    pub async fn list_unconfirmed_memories(
        &self,
        limit: i64,
    ) -> Result<Vec<origin_types::RecentActivityItem>, OriginError> {
        use origin_types::{ActivityBadge, ActivityKind, RecentActivityItem};

        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, title, summary, content, \
                        created_at, last_modified \
                 FROM memories \
                 WHERE source = 'memory' AND chunk_index = 0 \
                   AND (supersede_mode IS NULL OR supersede_mode != 'archive') \
                   AND (confirmed = 0 OR confirmed IS NULL) \
                   AND (is_recap IS NULL OR is_recap != 1) \
                 ORDER BY COALESCE(last_modified, created_at) DESC \
                 LIMIT ?1",
                libsql::params![limit],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_unconfirmed_memories scan: {e}")))?;

        let mut items = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let source_id: String = row.get(0).unwrap_or_default();
            let title: String = row.get(1).unwrap_or_default();
            let summary: Option<String> = row.get::<Option<String>>(2).unwrap_or(None);
            let content: String = row.get(3).unwrap_or_default();
            let created_at: i64 = row.get(4).unwrap_or(0);
            let last_modified: Option<i64> = row.get::<Option<i64>>(5).unwrap_or(None);

            // Snippet: prefer summary; fall back to first 100 chars of content (UTF-8 safe).
            let snippet: Option<String> = summary.filter(|s| !s.is_empty()).or_else(|| {
                let truncated: String = content.chars().take(100).collect();
                if truncated.is_empty() {
                    None
                } else {
                    Some(truncated)
                }
            });

            // timestamp_ms: DB stores seconds → convert to milliseconds.
            let ts_s = last_modified.unwrap_or(created_at);
            let timestamp_ms = (ts_s as u64).saturating_mul(1000);

            items.push(RecentActivityItem {
                kind: ActivityKind::Memory,
                id: source_id,
                title,
                snippet,
                timestamp_ms,
                badge: ActivityBadge::NeedsReview,
            });
        }
        Ok(items)
    }

    /// Return up to `limit` most-recently-modified active concepts, ordered newest first.
    ///
    /// `since_ms` is the caller's "last seen" watermark in **milliseconds**.  The
    /// `concepts` table stores timestamps as RFC 3339 **strings**; comparisons use
    /// lexicographic ordering which is correct for UTC timestamps with the same offset
    /// format (`+00:00`).  `since_ms` drives badge derivation only — it never filters
    /// rows out.  The returned `timestamp_ms` is always in milliseconds.
    ///
    /// Badge precedence (high → low):
    ///   NeedsReview > New > Growing > Revised > None
    pub async fn list_recent_concepts_with_badges(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentActivityItem>, OriginError> {
        use origin_types::{ActivityBadge, ActivityKind, RecentActivityItem};

        // Convert since_ms → RFC3339 string for TEXT-column comparisons, and
        // → Unix seconds for INTEGER memories.created_at comparisons.
        let since_rfc = since_ms.and_then(|ms| {
            chrono::DateTime::from_timestamp(ms / 1000, 0).map(|dt| dt.to_rfc3339())
        });
        let since_s: Option<i64> = since_ms.map(|ms| ms / 1000);

        // --- Phase A: fetch top-N active concepts (scoped guard) ---
        let concept_rows: Vec<(String, String, String, String, i64, String, String)> = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT id, title, COALESCE(summary, ''), COALESCE(source_memory_ids, '[]'), \
                            version, created_at, last_modified \
                     FROM concepts \
                     WHERE status = 'active' \
                     ORDER BY last_modified DESC \
                     LIMIT ?1",
                    libsql::params![limit],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("list_recent_concepts scan: {e}")))?;
            let mut out = Vec::new();
            while let Ok(Some(row)) = rows.next().await {
                let id: String = row.get(0).unwrap_or_default();
                let title: String = row.get(1).unwrap_or_default();
                let summary: String = row.get(2).unwrap_or_default();
                let src_json: String = row.get(3).unwrap_or_else(|_| "[]".to_string());
                let version: i64 = row.get(4).unwrap_or(1);
                let created_at: String = row.get(5).unwrap_or_default();
                let last_modified: String = row.get(6).unwrap_or_default();
                out.push((
                    id,
                    title,
                    summary,
                    src_json,
                    version,
                    created_at,
                    last_modified,
                ));
            }
            out
            // conn guard dropped here
        };

        // --- Phase B: collect all source_memory_ids + call pending_review once ---
        let mut all_source_ids: Vec<String> = Vec::new();
        let mut per_concept_members: Vec<Vec<String>> = Vec::with_capacity(concept_rows.len());
        for (_, _, _, src_json, _, _, _) in &concept_rows {
            let members: Vec<String> = serde_json::from_str(src_json).unwrap_or_default();
            all_source_ids.extend(members.iter().cloned());
            per_concept_members.push(members);
        }
        let flagged = self.pending_review_memory_ids(&all_source_ids).await?;

        // --- Phase C+D: badge derivation + output ---
        // TODO(home-v2 #2): coalesce the per-concept growth count into a single
        // JOIN query when `limit` grows beyond ~10. Current N+1 pattern re-acquires
        // the mutex per concept; fine at limit=10 but it's the home hot path.
        // Sketch:
        //   SELECT c.id, COUNT(*) FROM concepts c, memories m
        //   WHERE c.id IN (...) AND m.source_id IN (json_each(c.source_memory_ids))
        //     AND m.created_at >= ?since_s
        //   GROUP BY c.id
        let mut items = Vec::with_capacity(concept_rows.len());
        for ((id, title, summary, _src_json, version, created_at, last_modified), members) in
            concept_rows.into_iter().zip(per_concept_members)
        {
            // Determine if any member is pending review (NeedsReview wins over all).
            let needs_review = members.iter().any(|m| flagged.contains(m));

            let badge = if needs_review {
                ActivityBadge::NeedsReview
            } else if let Some(ref since) = since_rfc {
                // New: concept.created_at (RFC3339 string) >= since (RFC3339 string).
                if created_at >= *since {
                    ActivityBadge::New
                } else {
                    // Growing: count source members whose created_at (INTEGER seconds) >= since_s.
                    let growing_count = if !members.is_empty() {
                        let since_s_val = since_s.unwrap_or(0);
                        let placeholders: String = members
                            .iter()
                            .enumerate()
                            .map(|(i, _)| format!("?{}", i + 2))
                            .collect::<Vec<_>>()
                            .join(", ");
                        let sql = format!(
                            "SELECT COUNT(*) FROM memories \
                             WHERE source_id IN ({}) AND created_at >= ?1",
                            placeholders
                        );
                        let mut params: Vec<libsql::Value> =
                            vec![libsql::Value::Integer(since_s_val)];
                        for m in &members {
                            params.push(libsql::Value::Text(m.clone()));
                        }
                        let count: i64 = {
                            let conn = self.conn.lock().await;
                            let mut rows = conn
                                .query(&sql, libsql::params_from_iter(params))
                                .await
                                .map_err(|e| {
                                    OriginError::VectorDb(format!(
                                        "list_recent_concepts growth count: {e}"
                                    ))
                                })?;
                            let row_opt = rows.next().await.map_err(|e| {
                                OriginError::VectorDb(format!("concept growth next: {e}"))
                            })?;
                            match row_opt {
                                Some(r) => r.get::<i64>(0).map_err(|e| {
                                    OriginError::VectorDb(format!("concept growth count: {e}"))
                                })?,
                                None => 0,
                            }
                            // conn guard dropped here
                        };
                        count
                    } else {
                        0
                    };

                    if growing_count > 0 {
                        ActivityBadge::Growing {
                            added: growing_count as u32,
                        }
                    } else if version > 1 && last_modified >= *since {
                        ActivityBadge::Revised
                    } else {
                        ActivityBadge::None
                    }
                }
            } else {
                // No since_ms provided — no badge.
                ActivityBadge::None
            };

            // snippet: prefer non-empty summary; fall back to None.
            let snippet: Option<String> = if summary.is_empty() {
                None
            } else {
                Some(summary)
            };

            // timestamp_ms: parse RFC3339 last_modified → milliseconds.
            let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&last_modified)
                .map(|dt| dt.timestamp_millis() as u64)
                .unwrap_or(0);

            items.push(RecentActivityItem {
                kind: ActivityKind::Concept,
                id,
                title,
                snippet,
                timestamp_ms,
                badge,
            });
        }
        Ok(items)
    }

    /// Pipeline diagnostic: enrichment status, entity linking, refinement queue.
    pub async fn pipeline_status(&self) -> Result<serde_json::Value, OriginError> {
        let conn = self.conn.lock().await;

        // Enrichment status breakdown — derived from enrichment_steps table
        let mut rows = conn
            .query(
                "SELECT status, COUNT(*) FROM enrichment_steps GROUP BY status",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pipeline_status enrichment: {e}")))?;
        let mut enrichment = serde_json::Map::new();
        while let Ok(Some(row)) = rows.next().await {
            let status: String = row.get(0).unwrap_or_default();
            let count: i64 = row.get(1).unwrap_or(0);
            enrichment.insert(status, serde_json::Value::Number(count.into()));
        }
        // Count memories with no enrichment steps at all (raw)
        let mut raw_rows = conn.query(
            "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND source_id NOT IN (SELECT DISTINCT source_id FROM enrichment_steps)",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("pipeline_status raw count: {e}")))?;
        if let Ok(Some(row)) = raw_rows.next().await {
            let raw_count: i64 = row.get(0).unwrap_or(0);
            if raw_count > 0 {
                enrichment.insert(
                    "raw".to_string(),
                    serde_json::Value::Number(raw_count.into()),
                );
            }
        }

        // Entity linking
        let mut rows = conn
            .query(
                "SELECT \
               COUNT(*) FILTER (WHERE entity_id IS NOT NULL) as linked, \
               COUNT(*) FILTER (WHERE entity_id IS NULL) as unlinked \
             FROM memories WHERE source = 'memory'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pipeline_status entity: {e}")))?;
        let (linked, unlinked) = if let Ok(Some(row)) = rows.next().await {
            (
                row.get::<i64>(0).unwrap_or(0),
                row.get::<i64>(1).unwrap_or(0),
            )
        } else {
            (0, 0)
        };

        // Refinement queue
        let mut rows = conn
            .query(
                "SELECT action, status, COUNT(*) FROM refinement_queue GROUP BY action, status",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pipeline_status queue: {e}")))?;
        let mut queue = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let action: String = row.get(0).unwrap_or_default();
            let status: String = row.get(1).unwrap_or_default();
            let count: i64 = row.get(2).unwrap_or(0);
            queue.push(serde_json::json!({"action": action, "status": status, "count": count}));
        }

        // Recaps
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND is_recap = 1",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("pipeline_status recaps: {e}")))?;
        let recap_count: i64 = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
            .map(|r| r.get::<i64>(0).unwrap_or(0))
            .unwrap_or(0);

        // Type breakdown
        let mut rows = conn.query(
            "SELECT memory_type, COUNT(*) FROM memories WHERE source = 'memory' GROUP BY memory_type ORDER BY COUNT(*) DESC",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("pipeline_status types: {e}")))?;
        let mut types = serde_json::Map::new();
        while let Ok(Some(row)) = rows.next().await {
            let mt: String = row.get(0).unwrap_or_else(|_| "null".to_string());
            let count: i64 = row.get(1).unwrap_or(0);
            types.insert(mt, serde_json::Value::Number(count.into()));
        }

        // Quality breakdown
        let mut rows = conn.query(
            "SELECT COALESCE(quality, 'unclassified'), COUNT(*) FROM memories WHERE source = 'memory' GROUP BY quality",
            (),
        ).await.map_err(|e| OriginError::VectorDb(format!("pipeline_status quality: {e}")))?;
        let mut quality = serde_json::Map::new();
        while let Ok(Some(row)) = rows.next().await {
            let q: String = row.get(0).unwrap_or_default();
            let count: i64 = row.get(1).unwrap_or(0);
            quality.insert(q, serde_json::Value::Number(count.into()));
        }

        Ok(serde_json::json!({
            "enrichment": enrichment,
            "entity_linking": {"linked": linked, "unlinked": unlinked},
            "refinement_queue": queue,
            "recaps": recap_count,
            "types": types,
            "quality": quality,
        }))
    }

    // NOTE: `get_most_recent_agent` used to live here as the fallback for paths
    // that couldn't carry an agent name (DELETE, header-less calls). It was
    // removed because it's the loose-observation footgun — attributing a request
    // to whoever last did anything can silently assign a destructive action to
    // the wrong agent. Since the switch to "x-agent-name header is canonical",
    // unknown callers become `"unknown"` (hidden from UI filters) and the fallback
    // is gone. See research doc and mem0 issues #3218 / #3998 for precedent.

    /// List recent agent activity with memory titles resolved.
    pub async fn list_agent_activity(
        &self,
        limit: usize,
        agent_name: Option<&str>,
        since: Option<i64>,
    ) -> Result<Vec<AgentActivityRow>, OriginError> {
        let conn = self.conn.lock().await;

        let mut sql = String::from(
            "SELECT id, timestamp, agent_name, action, memory_ids, query, detail
             FROM agent_activity WHERE 1=1",
        );
        let mut params: Vec<libsql::Value> = Vec::new();
        let mut param_idx = 1;

        if let Some(agent) = agent_name {
            sql.push_str(&format!(" AND agent_name = ?{}", param_idx));
            params.push(libsql::Value::Text(agent.to_string()));
            param_idx += 1;
        }

        if let Some(ts) = since {
            sql.push_str(&format!(" AND timestamp >= ?{}", param_idx));
            params.push(libsql::Value::Integer(ts));
            param_idx += 1;
        }
        let _ = param_idx; // suppress unused warning

        sql.push_str(&format!(" ORDER BY timestamp DESC LIMIT {}", limit));

        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_agent_activity query: {}", e)))?;

        let mut activities: Vec<AgentActivityRow> = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            activities.push(AgentActivityRow {
                id: row.get::<i64>(0).unwrap_or(0),
                timestamp: row.get::<i64>(1).unwrap_or(0),
                agent_name: row.get::<String>(2).unwrap_or_default(),
                action: row.get::<String>(3).unwrap_or_default(),
                memory_ids: row.get::<String>(4).ok().filter(|s| !s.is_empty()),
                query: row.get::<String>(5).ok().filter(|s| !s.is_empty()),
                detail: row.get::<String>(6).ok().filter(|s| !s.is_empty()),
                memory_titles: Vec::new(), // resolved below
            });
        }
        drop(rows);

        // Resolve memory titles from memory_ids
        for activity in &mut activities {
            if let Some(ref ids_csv) = activity.memory_ids {
                let ids: Vec<&str> = ids_csv.split(',').filter(|s| !s.is_empty()).collect();
                if !ids.is_empty() {
                    let placeholders: Vec<String> = ids
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("?{}", i + 1))
                        .collect();
                    let title_sql = format!(
                        "SELECT DISTINCT title FROM memories WHERE source_id IN ({}) GROUP BY source_id",
                        placeholders.join(",")
                    );
                    let title_params: Vec<libsql::Value> = ids
                        .iter()
                        .map(|id| libsql::Value::Text(id.to_string()))
                        .collect();
                    let mut title_rows = conn
                        .query(&title_sql, libsql::params_from_iter(title_params))
                        .await
                        .map_err(|e| OriginError::VectorDb(format!("resolve titles: {}", e)))?;
                    while let Some(trow) = title_rows
                        .next()
                        .await
                        .map_err(|e| OriginError::VectorDb(e.to_string()))?
                    {
                        if let Ok(title) = trow.get::<String>(0) {
                            if !title.is_empty() {
                                activity.memory_titles.push(title);
                            }
                        }
                    }
                }
            }
        }

        Ok(activities)
    }

    // ===== Rejection Log =====

    /// Insert a rejection record into the quality gate rejection log.
    #[allow(clippy::too_many_arguments)]
    pub async fn log_rejection(
        &self,
        id: &str,
        content: &str,
        source_agent: Option<&str>,
        rejection_reason: &str,
        rejection_detail: Option<&str>,
        similarity_score: Option<f64>,
        similar_to_source_id: Option<&str>,
    ) -> Result<(), OriginError> {
        let truncated_content: String = content.chars().take(500).collect();
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT OR IGNORE INTO rejected_memories (id, content, source_agent, rejection_reason, rejection_detail, similarity_score, similar_to_source_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![id, truncated_content, source_agent, rejection_reason, rejection_detail, similarity_score, similar_to_source_id, now],
        ).await.map_err(|e| OriginError::VectorDb(format!("log_rejection: {e}")))?;
        Ok(())
    }

    /// Query rejection records, optionally filtered by reason. Returns newest first.
    pub async fn get_rejections(
        &self,
        limit: usize,
        reason: Option<&str>,
    ) -> Result<Vec<RejectionRecord>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = if let Some(r) = reason {
            conn.query(
                "SELECT id, content, source_agent, rejection_reason, rejection_detail, similarity_score, similar_to_source_id, created_at
                 FROM rejected_memories WHERE rejection_reason = ?1 ORDER BY created_at DESC LIMIT ?2",
                libsql::params![r, limit as i64],
            ).await
        } else {
            conn.query(
                "SELECT id, content, source_agent, rejection_reason, rejection_detail, similarity_score, similar_to_source_id, created_at
                 FROM rejected_memories ORDER BY created_at DESC LIMIT ?1",
                libsql::params![limit as i64],
            ).await
        }.map_err(|e| OriginError::VectorDb(format!("get_rejections: {e}")))?;

        let mut records = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_rejections row: {e}")))?
        {
            records.push(RejectionRecord {
                id: row.get::<String>(0).unwrap_or_default(),
                content: row.get::<String>(1).unwrap_or_default(),
                source_agent: row.get::<String>(2).ok(),
                rejection_reason: row.get::<String>(3).unwrap_or_default(),
                rejection_detail: row.get::<String>(4).ok(),
                similarity_score: row.get::<f64>(5).ok(),
                similar_to_source_id: row.get::<String>(6).ok(),
                created_at: row.get::<i64>(7).unwrap_or(0),
            });
        }
        Ok(records)
    }

    /// Delete rejection records older than `max_age_days`. Returns the number deleted.
    pub async fn prune_rejections(&self, max_age_days: i64) -> Result<usize, OriginError> {
        let cutoff = chrono::Utc::now().timestamp() - (max_age_days * 86400);
        let conn = self.conn.lock().await;
        let deleted = conn
            .execute(
                "DELETE FROM rejected_memories WHERE created_at < ?1",
                libsql::params![cutoff],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("prune_rejections: {e}")))?;
        Ok(deleted as usize)
    }

    // ==================== Concepts ====================

    /// Insert a new concept. Generates an embedding from `title + summary` if available.
    /// Embedding failures are logged but do not prevent insertion.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_concept(
        &self,
        id: &str,
        title: &str,
        summary: Option<&str>,
        content: &str,
        entity_id: Option<&str>,
        domain: Option<&str>,
        source_memory_ids: &[&str],
        now: &str,
    ) -> Result<(), OriginError> {
        let source_ids_json = serde_json::to_string(&source_memory_ids)
            .map_err(|e| OriginError::VectorDb(format!("serialize source_memory_ids: {e}")))?;

        // Generate embedding from title + summary (before acquiring conn lock)
        let embed_text = match summary {
            Some(s) => format!("{} {}", title, s),
            None => title.to_string(),
        };
        let embedding_sql = match self.generate_embeddings(&[embed_text]) {
            Ok(vecs) if !vecs.is_empty() => Some(Self::vec_to_sql(&vecs[0])),
            Ok(_) => {
                log::warn!("insert_concept: empty embedding result for {id}");
                None
            }
            Err(e) => {
                log::warn!("insert_concept: embedding failed for {id}: {e}");
                None
            }
        };

        let conn = self.conn.lock().await;
        match &embedding_sql {
            Some(emb) => {
                conn.execute(
                    "INSERT INTO concepts (id, title, summary, content, entity_id, domain, source_memory_ids, version, status, embedding, created_at, last_compiled, last_modified)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, 'active', vector32(?8), ?9, ?9, ?9)",
                    libsql::params![id, title, summary, content, entity_id, domain, source_ids_json, emb.as_str(), now],
                ).await.map_err(|e| OriginError::VectorDb(format!("insert_concept: {e}")))?;
            }
            None => {
                conn.execute(
                    "INSERT INTO concepts (id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, 'active', ?8, ?8, ?8)",
                    libsql::params![id, title, summary, content, entity_id, domain, source_ids_json, now],
                ).await.map_err(|e| OriginError::VectorDb(format!("insert_concept: {e}")))?;
            }
        }
        Ok(())
    }

    /// Retrieve a concept by id. Returns None if not found.
    pub async fn get_concept(&self, id: &str) -> Result<Option<Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE id = ?1",
                libsql::params![id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_concept: {e}")))?;
        match rows.next().await {
            Ok(Some(row)) => Ok(Some(Self::row_to_concept(&row)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(OriginError::VectorDb(format!("get_concept row: {e}"))),
        }
    }

    /// Find an active concept linked to a specific entity. Returns None if not found.
    pub async fn get_concept_by_entity(
        &self,
        entity_id: &str,
    ) -> Result<Option<Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE entity_id = ?1 AND status = 'active' LIMIT 1",
                libsql::params![entity_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_concept_by_entity: {e}")))?;
        match rows.next().await {
            Ok(Some(row)) => Ok(Some(Self::row_to_concept(&row)?)),
            Ok(None) => Ok(None),
            Err(e) => Err(OriginError::VectorDb(format!(
                "get_concept_by_entity row: {e}"
            ))),
        }
    }

    /// Find an active concept that includes a specific source memory.
    pub async fn find_concept_by_source_memory(
        &self,
        source_id: &str,
    ) -> Result<Option<Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn.query(
            "SELECT id FROM concepts WHERE status = 'active' AND source_memory_ids LIKE ?1 LIMIT 1",
            libsql::params![format!("%\"{}\"%" , source_id)],
        ).await.map_err(|e| OriginError::VectorDb(format!("find_concept_by_source_memory: {e}")))?;

        match rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            Some(row) => {
                let id: String = row.get(0).unwrap_or_default();
                drop(rows);
                drop(conn);
                self.get_concept(&id).await
            }
            None => Ok(None),
        }
    }

    /// List concepts filtered by status, ordered by last_modified descending.
    pub async fn list_concepts(
        &self,
        status: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE status = ?1 ORDER BY last_modified DESC LIMIT ?2 OFFSET ?3",
                libsql::params![status, limit, offset],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_concepts: {e}")))?;
        let mut results = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            results.push(Self::row_to_concept(&row)?);
        }
        Ok(results)
    }

    /// List concepts, optionally filtered by domain.
    pub async fn list_concepts_by_domain(
        &self,
        status: &str,
        domain: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let (sql, params): (String, Vec<libsql::Value>) = if let Some(d) = domain {
            (
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE status = ?1 AND domain = ?2 ORDER BY last_modified DESC LIMIT ?3 OFFSET ?4".to_string(),
                vec![
                    libsql::Value::Text(status.to_string()),
                    libsql::Value::Text(d.to_string()),
                    libsql::Value::Integer(limit as i64),
                    libsql::Value::Integer(offset as i64),
                ],
            )
        } else {
            (
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0)
                 FROM concepts WHERE status = ?1 ORDER BY last_modified DESC LIMIT ?2 OFFSET ?3".to_string(),
                vec![
                    libsql::Value::Text(status.to_string()),
                    libsql::Value::Integer(limit as i64),
                    libsql::Value::Integer(offset as i64),
                ],
            )
        };
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("list concepts by domain: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            results.push(Self::row_to_concept(&row)?);
        }
        Ok(results)
    }

    /// Update a concept's content and source memory ids. Increments version, updates timestamps.
    /// Dual-writes: updates the JSON `source_memory_ids` column (backward compat) AND
    /// inserts any new links into the `concept_sources` join table.
    pub async fn update_concept_content(
        &self,
        id: &str,
        content: &str,
        source_memory_ids: &[&str],
        link_reason: &str,
    ) -> Result<(), OriginError> {
        let source_ids_json = serde_json::to_string(&source_memory_ids)
            .map_err(|e| OriginError::VectorDb(format!("serialize source_memory_ids: {e}")))?;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        // Dual-write: update JSON column (backward compat) + increment version + timestamps
        conn.execute(
            "UPDATE concepts SET content = ?1, source_memory_ids = ?2, version = version + 1, last_compiled = ?3, last_modified = ?3 WHERE id = ?4",
            libsql::params![content, source_ids_json, now, id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_concept_content: {e}")))?;
        // Dual-write: insert any new source links into the join table (idempotent)
        for sid in source_memory_ids {
            let _ = conn
                .execute(
                    "INSERT OR IGNORE INTO concept_sources (concept_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, ?3, ?4)",
                    libsql::params![id, sid, now_ts, link_reason],
                )
                .await;
        }
        Ok(())
    }

    /// Find a matching concept for a new memory — by entity_id first, then embedding similarity.
    pub async fn find_matching_concept(
        &self,
        entity_id: Option<&str>,
        memory_embedding: &[f32],
        similarity_threshold: f64,
    ) -> Result<Option<Concept>, OriginError> {
        // First: try entity_id match
        if let Some(eid) = entity_id {
            if let Some(concept) = self.get_concept_by_entity(eid).await? {
                return Ok(Some(concept));
            }
        }

        // Second: try embedding similarity against concept embeddings
        let emb_sql = Self::vec_to_sql(memory_embedding);
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT c.id, c.title, c.summary, c.content, c.entity_id, c.domain,
                        c.source_memory_ids, c.version, c.status, c.created_at, c.last_compiled, c.last_modified,
                        COALESCE(c.sources_updated_count, 0), c.stale_reason, COALESCE(c.user_edited, 0),
                        vector_distance_cos(c.embedding, vector32(?1)) as dist
                 FROM concepts c
                 WHERE c.status = 'active' AND c.embedding IS NOT NULL
                 ORDER BY dist ASC LIMIT 1",
                libsql::params![emb_sql],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("concept similarity: {}", e)))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let dist: f64 = row.get(15).unwrap_or(1.0);
            let similarity = 1.0 - dist;
            if similarity >= similarity_threshold {
                return Ok(Some(Self::row_to_concept(&row)?));
            }
        }

        Ok(None)
    }

    /// Check if a concept's inputs have changed since last compilation.
    /// True if source memories were modified or new memories linked to same entity.
    pub async fn has_concept_sources_changed(
        &self,
        concept: &Concept,
    ) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;

        // Check 1: Any source memory modified after last_compiled?
        if !concept.source_memory_ids.is_empty() {
            for sid in &concept.source_memory_ids {
                let mut rows = conn.query(
                    "SELECT last_modified FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                    libsql::params![sid.clone()],
                ).await.map_err(|e| OriginError::VectorDb(format!("concept change check: {e}")))?;
                if let Some(row) = rows
                    .next()
                    .await
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?
                {
                    let modified: i64 = row.get(0).unwrap_or(0);
                    // Compare unix epoch to RFC3339 last_compiled
                    if let Ok(compiled) =
                        chrono::DateTime::parse_from_rfc3339(&concept.last_compiled)
                    {
                        if modified > compiled.timestamp() {
                            return Ok(true);
                        }
                    }
                }
            }
        }

        // Check 2: New memories linked to same entity but not in source_memory_ids?
        if let Some(ref entity_id) = concept.entity_id {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM memories \
                 WHERE entity_id = ?1 AND chunk_index = 0 AND source = 'memory' \
                   AND is_recap = 0 AND source_id NOT LIKE 'merged_%'",
                    libsql::params![entity_id.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("concept new mem check: {e}")))?;
            if let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let total: u64 = row.get(0).unwrap_or(0);
                if total as usize > concept.source_memory_ids.len() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get memory contents by source_ids (for recompilation).
    pub async fn get_memory_contents_by_ids(
        &self,
        source_ids: &[String],
    ) -> Result<Vec<(String, String)>, OriginError> {
        if source_ids.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().await;
        let placeholders = source_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT source_id, content FROM memories WHERE source_id IN ({}) AND chunk_index = 0 \
             AND source_id NOT IN (\
                 SELECT supersedes FROM memories \
                 WHERE supersedes IS NOT NULL AND pending_revision = 0 \
                 AND source = 'memory' AND supersede_mode = 'hide' \
                 GROUP BY supersedes\
             )",
            placeholders
        );
        let params: Vec<libsql::Value> = source_ids
            .iter()
            .map(|id| libsql::Value::Text(id.clone()))
            .collect();
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| OriginError::VectorDb(format!("get memories by ids: {}", e)))?;

        let mut results = vec![];
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            let content: String = row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            results.push((id, content));
        }
        Ok(results)
    }

    /// Archive a concept (set status to 'archived').
    pub async fn archive_concept(&self, id: &str) -> Result<(), OriginError> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE concepts SET status = 'archived', last_modified = ?1 WHERE id = ?2",
            libsql::params![now, id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("archive_concept: {e}")))?;
        Ok(())
    }

    /// Check maximum Jaccard overlap between a set of source_ids and any existing concept.
    /// Returns 0.0-1.0 (0 = no overlap, 1 = identical source set).
    pub async fn max_concept_overlap(&self, source_ids: &[String]) -> Result<f64, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_memory_ids FROM concepts WHERE status = 'active'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("concept overlap: {e}")))?;

        let input_set: std::collections::HashSet<&str> =
            source_ids.iter().map(|s| s.as_str()).collect();
        let mut max_overlap = 0.0f64;

        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let json: String = row.get(0).unwrap_or_default();
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(&json) {
                let concept_set: std::collections::HashSet<&str> =
                    ids.iter().map(|s| s.as_str()).collect();
                let intersection = input_set.intersection(&concept_set).count();
                let union = input_set.union(&concept_set).count();
                if union > 0 {
                    let jaccard = intersection as f64 / union as f64;
                    if jaccard > max_overlap {
                        max_overlap = jaccard;
                    }
                }
            }
        }
        Ok(max_overlap)
    }

    /// Get all memory source_ids that are already covered by active concepts.
    pub async fn get_covered_memory_ids(
        &self,
    ) -> Result<std::collections::HashSet<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_memory_ids FROM concepts WHERE status = 'active'",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("covered ids: {e}")))?;
        let mut covered = std::collections::HashSet::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let json: String = row.get(0).unwrap_or_default();
            if let Ok(ids) = serde_json::from_str::<Vec<String>>(&json) {
                covered.extend(ids);
            }
        }
        Ok(covered)
    }

    pub async fn delete_concept(&self, id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM concepts WHERE id = ?1", libsql::params![id])
            .await
            .map_err(|e| OriginError::VectorDb(format!("delete_concept: {e}")))?;
        Ok(())
    }

    /// Search concepts via vector similarity + FTS5 with RRF fusion.
    ///
    /// Embeds the query, runs DiskANN vector search on concept embeddings
    /// (title+summary), runs FTS5 MATCH on concept content, then merges
    /// results with Reciprocal Rank Fusion (same pattern as search_memory).
    pub async fn search_concepts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Concept>, OriginError> {
        // Embed query before acquiring conn lock (same pattern as search_memory)
        let embedding = self.get_or_compute_embedding(query)?;
        let vec_str = Self::vec_to_sql(&embedding);
        let fetch_limit = (limit * 3) as i64;

        let conn = self.conn.lock().await;

        let concept_select = "c.id, c.title, c.summary, c.content, c.entity_id, c.domain, \
                              c.source_memory_ids, c.version, c.status, c.created_at, \
                              c.last_compiled, c.last_modified, \
                              COALESCE(c.sources_updated_count, 0), c.stale_reason, \
                              COALESCE(c.user_edited, 0)";

        // --- Vector search via DiskANN index ---
        let mut vector_results: Vec<(String, f64, Concept)> = Vec::new();
        let vec_sql = format!(
            "SELECT {}, vector_distance_cos(c.embedding, vector32(?1)) AS dist \
             FROM vector_top_k('idx_concepts_embedding', vector32(?1), ?2) AS vt \
             JOIN concepts c ON c.rowid = vt.id \
             WHERE c.status = 'active'",
            concept_select,
        );
        match conn
            .query(&vec_sql, libsql::params![vec_str, fetch_limit])
            .await
        {
            Ok(mut rows) => {
                while let Ok(Some(row)) = rows.next().await {
                    match Self::row_to_concept(&row) {
                        Ok(concept) => {
                            let distance: f64 = row.get(15).unwrap_or(1.0);
                            let id = concept.id.clone();
                            vector_results.push((id, distance, concept));
                        }
                        Err(e) => {
                            log::warn!("[search_concepts] skipping malformed vector row: {e}")
                        }
                    }
                }
            }
            Err(e) => {
                log::warn!("[search_concepts] vector search failed: {e}");
            }
        }

        // --- FTS search with AND-then-OR fallback ---
        let mut fts_results: Vec<(String, Concept)> = Vec::new();
        let fts_sql = format!(
            "SELECT {} \
             FROM concepts c \
             JOIN concepts_fts f ON c.rowid = f.rowid \
             WHERE concepts_fts MATCH ?1 AND c.status = 'active' \
             ORDER BY rank LIMIT ?2",
            concept_select,
        );
        let fts_queries = vec![query.to_string(), Self::fts_or_query(query)];
        for fts_q in &fts_queries {
            match conn
                .query(&fts_sql, libsql::params![fts_q.clone(), fetch_limit])
                .await
            {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        if let Ok(concept) = Self::row_to_concept(&row) {
                            let id = concept.id.clone();
                            fts_results.push((id, concept));
                        }
                    }
                }
                Err(e) => {
                    log::debug!("[search_concepts] FTS query failed ({}): {e}", fts_q);
                }
            }
            if !fts_results.is_empty() {
                break; // AND matched, skip OR fallback
            }
        }

        drop(conn);

        // --- RRF fusion (distance-weighted, same as search_memory) ---
        let rrf_k = 60.0f32;
        let fts_weight = 0.2f32;
        let mut score_map: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut concept_map: std::collections::HashMap<String, Concept> =
            std::collections::HashMap::new();

        for (rank, (id, distance, concept)) in vector_results.into_iter().enumerate() {
            let similarity = (1.0 - distance as f32).max(0.01);
            let rrf_score = similarity / (rrf_k + rank as f32);
            *score_map.entry(id.clone()).or_default() += rrf_score;
            concept_map.entry(id).or_insert(concept);
        }

        for (rank, (id, concept)) in fts_results.into_iter().enumerate() {
            let rrf_score = fts_weight / (rrf_k + rank as f32);
            *score_map.entry(id.clone()).or_default() += rrf_score;
            concept_map.entry(id).or_insert(concept);
        }

        // Sort by combined RRF score, return top limit
        let mut final_results: Vec<Concept> = concept_map.into_values().collect();
        final_results.sort_by(|a, b| {
            let sa = score_map.get(&a.id).unwrap_or(&0.0);
            let sb = score_map.get(&b.id).unwrap_or(&0.0);
            sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        final_results.truncate(limit);
        Ok(final_results)
    }

    /// Backfill embeddings for concepts with NULL embedding column.
    ///
    /// Called by migration 42 and can be run manually for maintenance.
    /// Computes embeddings from title + summary (same as insert_concept).
    pub async fn backfill_concept_embeddings(&self) -> Result<usize, OriginError> {
        let needs_embed: Vec<(String, String)> = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT id, title, summary FROM concepts WHERE embedding IS NULL",
                    (),
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("backfill concepts fetch: {e}")))?;

            let mut out = Vec::new();
            while let Ok(Some(row)) = rows.next().await {
                let id: String = row.get(0).unwrap_or_default();
                let title: String = row.get(1).unwrap_or_default();
                let summary: Option<String> = row.get::<String>(2).ok();
                let embed_text = match summary {
                    Some(s) if !s.is_empty() => format!("{} {}", title, s),
                    _ => title,
                };
                out.push((id, embed_text));
            }
            out
        }; // conn lock dropped

        if needs_embed.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = needs_embed.iter().map(|(_, t)| t.clone()).collect();
        let embeddings = self.generate_embeddings(&texts)?;

        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("backfill begin: {e}")))?;

        let mut count = 0usize;
        for ((id, _), emb) in needs_embed.iter().zip(embeddings.iter()) {
            let emb_sql = Self::vec_to_sql(emb);
            if conn
                .execute(
                    "UPDATE concepts SET embedding = vector32(?1) WHERE id = ?2",
                    libsql::params![emb_sql, id.clone()],
                )
                .await
                .is_ok()
            {
                count += 1;
            }
        }

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("backfill commit: {e}")))?;
        Ok(count)
    }

    /// Parse a row into a Concept. Column order must match the SELECT used in concept queries.
    fn row_to_concept(row: &libsql::Row) -> Result<Concept, OriginError> {
        let source_ids_json: String = row.get::<String>(6).unwrap_or_else(|_| "[]".to_string());
        let source_memory_ids: Vec<String> =
            serde_json::from_str(&source_ids_json).unwrap_or_default();
        Ok(Concept {
            id: row
                .get::<String>(0)
                .map_err(|e| OriginError::VectorDb(format!("concept id: {e}")))?,
            title: row
                .get::<String>(1)
                .map_err(|e| OriginError::VectorDb(format!("concept title: {e}")))?,
            summary: row.get::<Option<String>>(2).unwrap_or(None),
            content: row
                .get::<String>(3)
                .map_err(|e| OriginError::VectorDb(format!("concept content: {e}")))?,
            entity_id: row.get::<Option<String>>(4).unwrap_or(None),
            domain: row.get::<Option<String>>(5).unwrap_or(None),
            source_memory_ids,
            version: row.get::<i64>(7).unwrap_or(1),
            status: row
                .get::<String>(8)
                .unwrap_or_else(|_| "active".to_string()),
            created_at: row
                .get::<String>(9)
                .map_err(|e| OriginError::VectorDb(format!("concept created_at: {e}")))?,
            last_compiled: row
                .get::<String>(10)
                .map_err(|e| OriginError::VectorDb(format!("concept last_compiled: {e}")))?,
            last_modified: row
                .get::<String>(11)
                .map_err(|e| OriginError::VectorDb(format!("concept last_modified: {e}")))?,
            sources_updated_count: row.get::<i64>(12).unwrap_or(0),
            stale_reason: row.get::<Option<String>>(13).unwrap_or(None),
            user_edited: row.get::<i64>(14).unwrap_or(0) != 0,
        })
    }

    // ===== Topic Match Helpers =====

    /// Fetch lightweight candidate memories for topic matching.
    /// Prefers same domain + memory_type but does not require them.
    /// Returns candidates with domain/type metadata so the caller can compute
    /// tiered thresholds (exact match → lower threshold, no match → higher).
    pub async fn topic_match_candidates(
        &self,
        domain: Option<&str>,
        memory_type: Option<&str>,
        max_candidates: usize,
    ) -> Result<Vec<crate::topic_match::TopicMatchCandidate>, OriginError> {
        let conn = self.conn.lock().await;

        // Build a flexible query: prefer same domain+type, but include all recent
        // chunk_index=0 memories as candidates. ORDER BY gives priority to exact
        // domain+type matches, then partial, then everything else.
        let sql = "SELECT source_id, title, content, entity_id, embedding, domain, memory_type
                   FROM memories
                   WHERE chunk_index = 0 AND pending_revision = 0
                   ORDER BY
                     CASE WHEN domain = ?1 AND memory_type = ?2 THEN 0
                          WHEN domain = ?1 OR memory_type = ?2 THEN 1
                          ELSE 2 END,
                     last_modified DESC
                   LIMIT ?3";
        let mut rows = conn
            .query(
                sql,
                libsql::params![
                    domain.unwrap_or(""),
                    memory_type.unwrap_or(""),
                    max_candidates as i64
                ],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("topic_match_candidates: {e}")))?;

        let mut candidates = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let source_id: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(format!("topic_cand source_id: {e}")))?;
            let title: String = row.get::<String>(1).unwrap_or_default();
            let content: String = row
                .get(2)
                .map_err(|e| OriginError::VectorDb(format!("topic_cand content: {e}")))?;
            let entity_id: Option<String> = row.get::<Option<String>>(3).unwrap_or(None);
            // Decode F32_BLOB (little-endian f32 bytes)
            let embedding: Vec<f32> = row
                .get::<Vec<u8>>(4)
                .unwrap_or_default()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let cand_domain: Option<String> = row.get::<Option<String>>(5).unwrap_or(None);
            let cand_type: Option<String> = row.get::<Option<String>>(6).unwrap_or(None);
            candidates.push(crate::topic_match::TopicMatchCandidate {
                source_id,
                title,
                content,
                entity_id,
                embedding,
                domain: cand_domain,
                memory_type: cand_type,
            });
        }
        Ok(candidates)
    }

    /// FTS5-based title matching for topic matching.
    ///
    /// Queries `memories_fts` with a `title:` column filter and returns the
    /// `source_id` values that match. The caller intersects this set with
    /// the pre-fetched candidates in Rust.
    ///
    /// Words shorter than 2 chars are skipped so that tokens like "SQL",
    /// "API", "Go" are retained while noise particles like "a" are dropped.
    pub async fn topic_match_title_fts(
        &self,
        title: &str,
        candidate_source_ids: &[&str],
    ) -> Result<Vec<String>, OriginError> {
        // Build list of significant words (2+ char, alphanumeric only).
        let words: Vec<String> = title
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 2)
            .map(|w| format!("title:{w}"))
            .collect();

        if words.is_empty() || candidate_source_ids.is_empty() {
            return Ok(Vec::new());
        }

        let fts_query = words.join(" OR ");

        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT c.source_id
                 FROM memories_fts fts
                 JOIN memories c ON fts.rowid = c.rowid
                 WHERE memories_fts MATCH ?1
                   AND c.chunk_index = 0
                   AND c.pending_revision = 0",
                libsql::params![fts_query],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("topic_match_title_fts: {e}")))?;

        // Collect matching source_ids and intersect with candidate set.
        let candidate_set: std::collections::HashSet<&str> =
            candidate_source_ids.iter().copied().collect();
        let mut matched = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let sid: String = row
                .get(0)
                .map_err(|e| OriginError::VectorDb(format!("topic_title_fts sid: {e}")))?;
            if candidate_set.contains(sid.as_str()) {
                matched.push(sid);
            }
        }
        Ok(matched)
    }

    // ===== Concept Sources Join Table Methods =====

    /// Link a memory to a concept in the concept_sources join table.
    /// Idempotent: INSERT OR IGNORE on the composite primary key.
    pub async fn link_concept_source(
        &self,
        concept_id: &str,
        memory_source_id: &str,
        link_reason: &str,
    ) -> Result<(), OriginError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR IGNORE INTO concept_sources (concept_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, ?3, ?4)",
            libsql::params![concept_id, memory_source_id, now, link_reason],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("link_concept_source: {e}")))?;
        Ok(())
    }

    /// Get all source memories linked to a concept, ordered by linked_at ascending.
    pub async fn get_concept_sources(
        &self,
        concept_id: &str,
    ) -> Result<Vec<origin_types::ConceptSource>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT concept_id, memory_source_id, linked_at, link_reason FROM concept_sources WHERE concept_id = ?1 ORDER BY linked_at ASC",
                libsql::params![concept_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_concept_sources: {e}")))?;
        let mut result = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            result.push(origin_types::ConceptSource {
                concept_id: row.get(0).map_err(|e| {
                    OriginError::VectorDb(format!("concept_sources concept_id: {e}"))
                })?,
                memory_source_id: row.get(1).map_err(|e| {
                    OriginError::VectorDb(format!("concept_sources memory_source_id: {e}"))
                })?,
                linked_at: row.get(2).unwrap_or(0),
                link_reason: row.get::<Option<String>>(3).unwrap_or(None),
            });
        }
        Ok(result)
    }

    /// Reverse lookup: find all active concepts that reference a given memory source_id.
    pub async fn get_concepts_for_memory(
        &self,
        memory_source_id: &str,
    ) -> Result<Vec<crate::concepts::Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT c.id, c.title, c.summary, c.content, c.entity_id, c.domain,
                        c.source_memory_ids, c.version, c.status, c.created_at, c.last_compiled, c.last_modified,
                        COALESCE(c.sources_updated_count, 0), c.stale_reason, COALESCE(c.user_edited, 0)
                 FROM concepts c
                 INNER JOIN concept_sources cs ON c.id = cs.concept_id
                 WHERE cs.memory_source_id = ?1 AND c.status = 'active'",
                libsql::params![memory_source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_concepts_for_memory: {e}")))?;
        let mut result = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            result.push(Self::row_to_concept(&row)?);
        }
        Ok(result)
    }

    /// Remove concept_sources rows where the referenced memory no longer exists.
    pub async fn cleanup_orphaned_concept_sources(&self) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;
        let rows_affected = conn
            .execute(
                "DELETE FROM concept_sources WHERE memory_source_id NOT IN (SELECT DISTINCT source_id FROM memories)",
                (),
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("cleanup_orphaned_concept_sources: {e}")))?;
        Ok(rows_affected as usize)
    }

    /// Check if a memory is protected from in-place upsert (confirmed or high-stability).
    pub async fn is_memory_protected(&self, source_id: &str) -> Result<bool, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT confirmed, stability FROM memories WHERE source_id = ?1 AND chunk_index = 0 LIMIT 1",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("is_memory_protected: {e}")))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            let confirmed: i64 = row.get(0).unwrap_or(0);
            let stability: Option<String> = row.get::<Option<String>>(1).unwrap_or(None);
            Ok(confirmed != 0
                || matches!(stability.as_deref(), Some("learned") | Some("confirmed")))
        } else {
            Ok(false)
        }
    }

    /// Mark a concept as stale with a specific reason.
    pub async fn set_concept_stale(
        &self,
        concept_id: &str,
        reason: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE concepts SET stale_reason = ?1 WHERE id = ?2",
            libsql::params![reason, concept_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("set_concept_stale: {e}")))?;
        Ok(())
    }

    /// Increment a concept's sources_updated_count (for trivial/non-conflicting source changes).
    pub async fn increment_concept_sources_updated(
        &self,
        concept_id: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE concepts SET sources_updated_count = sources_updated_count + 1 WHERE id = ?1",
            libsql::params![concept_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("increment_concept_sources_updated: {e}")))?;
        Ok(())
    }

    /// List active concepts with the given stale_reason, up to `limit` rows.
    pub async fn list_stale_concepts(
        &self,
        reason: &str,
    ) -> Result<Vec<crate::concepts::Concept>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, title, summary, content, entity_id, domain, source_memory_ids, version, status, created_at, last_compiled, last_modified, COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0) FROM concepts WHERE stale_reason = ?1 AND status = 'active' LIMIT 10",
                libsql::params![reason],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_stale_concepts: {e}")))?;
        let mut result = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(e.to_string()))?
        {
            result.push(Self::row_to_concept(&row)?);
        }
        Ok(result)
    }

    /// Clear staleness fields after successful re-distillation.
    pub async fn clear_concept_staleness(&self, concept_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE concepts SET stale_reason = NULL, sources_updated_count = 0 WHERE id = ?1",
            libsql::params![concept_id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("clear_concept_staleness: {e}")))?;
        Ok(())
    }

    /// Update a memory's content in-place for topic-key upsert.
    ///
    /// Workflow:
    /// 1. Reads existing row metadata (outside transaction, outside lock).
    /// 2. Computes new embedding (sync, CPU-bound, outside lock).
    /// 3. Inside a single transaction: deletes all chunks for source_id, re-inserts
    ///    one chunk with new content + embedding + incremented version + changelog.
    ///
    /// Preserves: title, source, source_id, memory_type, domain, entity_id, confirmed,
    /// stability, quality, structured_fields, created_at, and all other metadata.
    /// Updates: content, embedding, version, changelog, last_modified, word_count.
    pub async fn upsert_memory_in_place(
        &self,
        source_id: &str,
        new_content: &str,
        content_embedding: &[f32],
        source_agent: Option<&str>,
        incoming_source_id: Option<&str>,
        changelog_cap: usize,
    ) -> Result<(), OriginError> {
        // ---- Step 1: read existing row metadata (need lock briefly) ----
        struct SavedMeta {
            source: String,
            title: String,
            summary: Option<String>,
            url: Option<String>,
            chunk_type: String,
            language: Option<String>,
            memory_type: Option<String>,
            domain: Option<String>,
            confidence: Option<f64>,
            confirmed: i64,
            stability: String,
            entity_id: Option<String>,
            // Retired: see MemoryRow.enrichment_status above.
            #[allow(dead_code)]
            enrichment_status: String,
            quality: Option<String>,
            is_recap: i64,
            structured_fields: Option<String>,
            retrieval_cue: Option<String>,
            source_text: Option<String>,
            created_at: i64,
            version: i64,
            changelog: String,
        }

        let saved = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source, title, summary, url, chunk_type, language,
                            memory_type, domain, confidence, confirmed, stability,
                            entity_id, enrichment_status, quality, is_recap,
                            structured_fields, retrieval_cue, source_text,
                            COALESCE(created_at, last_modified),
                            COALESCE(version, 1), COALESCE(changelog, '[]')
                     FROM memories
                     WHERE source_id = ?1 AND chunk_index = 0
                     LIMIT 1",
                    libsql::params![source_id],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("upsert_in_place read: {e}")))?;

            let row = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
                .ok_or_else(|| {
                    OriginError::VectorDb(format!(
                        "upsert_memory_in_place: source_id {source_id} not found"
                    ))
                })?;

            SavedMeta {
                source: row
                    .get::<String>(0)
                    .unwrap_or_else(|_| "memory".to_string()),
                title: row.get::<String>(1).unwrap_or_default(),
                summary: row.get::<Option<String>>(2).unwrap_or(None),
                url: row.get::<Option<String>>(3).unwrap_or(None),
                chunk_type: row.get::<String>(4).unwrap_or_else(|_| "prose".to_string()),
                language: row.get::<Option<String>>(5).unwrap_or(None),
                memory_type: row.get::<Option<String>>(6).unwrap_or(None),
                domain: row.get::<Option<String>>(7).unwrap_or(None),
                confidence: row.get::<Option<f64>>(8).unwrap_or(None),
                confirmed: row.get::<i64>(9).unwrap_or(0),
                stability: row.get::<String>(10).unwrap_or_else(|_| "new".to_string()),
                entity_id: row.get::<Option<String>>(11).unwrap_or(None),
                enrichment_status: row
                    .get::<String>(12)
                    .unwrap_or_else(|_| "enriched".to_string()),
                quality: row.get::<Option<String>>(13).unwrap_or(None),
                is_recap: row.get::<i64>(14).unwrap_or(0),
                structured_fields: row.get::<Option<String>>(15).unwrap_or(None),
                retrieval_cue: row.get::<Option<String>>(16).unwrap_or(None),
                source_text: row.get::<Option<String>>(17).unwrap_or(None),
                created_at: row
                    .get::<i64>(18)
                    .unwrap_or_else(|_| chrono::Utc::now().timestamp()),
                version: row.get::<i64>(19).unwrap_or(1),
                changelog: row.get::<String>(20).unwrap_or_else(|_| "[]".to_string()),
            }
        };

        // ---- Step 2: use the caller-supplied embedding (already computed for topic matching) ----
        let vec_sql = Self::vec_to_sql(content_embedding);

        // ---- Step 3: build new changelog entry and version ----
        let now_ts = chrono::Utc::now().timestamp();
        let new_version = saved.version + 1;
        let entry = serde_json::json!({
            "version": new_version,
            "at": now_ts,
            "delta": "",   // placeholder — async LLM can fill this later
            "source_agent": source_agent,
            "incoming_source_id": incoming_source_id,
        });
        let mut changelog: Vec<serde_json::Value> =
            serde_json::from_str(&saved.changelog).unwrap_or_default();
        changelog.push(entry);
        if changelog.len() > changelog_cap {
            changelog.drain(..changelog.len() - changelog_cap);
        }
        let changelog_json = serde_json::to_string(&changelog)
            .map_err(|e| OriginError::VectorDb(format!("serialize changelog: {e}")))?;

        // Recount words for the new content
        let new_word_count = new_content.split_whitespace().count() as i64;

        // Generate a fresh chunk id (old id is deleted; avoids PK edge-cases during replay)
        let new_chunk_id = format!(
            "mem_{}",
            uuid::Uuid::new_v4()
                .to_string()
                .replace('-', "")
                .chars()
                .take(12)
                .collect::<String>()
        );

        // ---- Step 4: transaction — delete old chunks, insert new chunk ----
        // TODO(multi-chunk-upsert): For most memories content fits in a single chunk.
        // If content exceeds the chunk size limit (~2000 chars / 512 tokens), this
        // currently stores the full content as chunk_index=0 only. A future improvement
        // should reuse the chunker module to split `new_content` and insert multiple
        // chunks with sequential chunk_index values, mirroring how `upsert_documents`
        // handles multi-chunk RawDocuments.
        {
            let conn = self.conn.lock().await;
            conn.execute("BEGIN", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("upsert_in_place BEGIN: {e}")))?;

            conn.execute(
                "DELETE FROM memories WHERE source_id = ?1",
                libsql::params![source_id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("upsert_in_place DELETE: {e}")))?;

            let insert_sql = "INSERT INTO memories (
                    id, content, source, source_id, title, summary, url,
                    chunk_index, last_modified, chunk_type, language, byte_start, byte_end,
                    semantic_unit, memory_type, domain, source_agent, confidence, confirmed,
                    stability, supersedes, pending_revision, word_count,
                    entity_id, enrichment_status, quality, is_recap, supersede_mode,
                    structured_fields, retrieval_cue, source_text,
                    embedding, created_at, version, changelog
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7,
                    0, ?8, ?9, ?10, NULL, NULL,
                    NULL, ?11, ?12, ?13, ?14, ?15,
                    ?16, NULL, 0, ?17,
                    ?18, ?19, ?20, ?21, 'hide',
                    ?22, ?23, ?24,
                    vector32(?25), ?26, ?27, ?28
                )";

            conn.execute(
                insert_sql,
                libsql::params![
                    new_chunk_id,
                    new_content,
                    saved.source,
                    source_id,
                    saved.title,
                    saved.summary,
                    saved.url,
                    now_ts,
                    saved.chunk_type,
                    saved.language,
                    saved.memory_type,
                    saved.domain,
                    source_agent,
                    saved.confidence,
                    saved.confirmed,
                    saved.stability,
                    new_word_count,
                    saved.entity_id,
                    "legacy", // column retired; status derived from enrichment_steps
                    saved.quality,
                    saved.is_recap,
                    saved.structured_fields,
                    saved.retrieval_cue,
                    saved.source_text,
                    vec_sql,
                    saved.created_at,
                    new_version,
                    changelog_json
                ],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("upsert_in_place INSERT: {e}")))?;

            conn.execute("COMMIT", ())
                .await
                .map_err(|e| OriginError::VectorDb(format!("upsert_in_place COMMIT: {e}")))?;
        }

        log::info!(
            "[db] upsert_memory_in_place: source_id={source_id} v{} → v{new_version}",
            saved.version
        );
        Ok(())
    }

    // ===== Source Sync State Methods =====

    /// Insert or update sync state for a file tracked by a knowledge source.
    pub async fn upsert_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
        mtime_ns: i64,
        content_hash: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        conn.execute(
            "INSERT INTO source_sync_state (source_id, file_path, mtime_ns, content_hash, last_synced_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(source_id, file_path) DO UPDATE SET
                mtime_ns = excluded.mtime_ns,
                content_hash = excluded.content_hash,
                last_synced_at = excluded.last_synced_at",
            libsql::params![
                source_id.to_string(),
                file_path.to_string(),
                mtime_ns,
                content_hash.to_string(),
                now
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("upsert_sync_state: {}", e)))?;
        Ok(())
    }

    /// Get sync state for a specific file in a source.
    pub async fn get_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
    ) -> Result<Option<FileSyncState>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, file_path, mtime_ns, content_hash, last_synced_at
                 FROM source_sync_state WHERE source_id = ?1 AND file_path = ?2",
                libsql::params![source_id.to_string(), file_path.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_sync_state: {}", e)))?;
        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_sync_state row: {}", e)))?
        {
            Ok(Some(FileSyncState {
                source_id: row
                    .get::<String>(0)
                    .map_err(|e| OriginError::VectorDb(format!("sync source_id: {e}")))?,
                file_path: row
                    .get::<String>(1)
                    .map_err(|e| OriginError::VectorDb(format!("sync file_path: {e}")))?,
                mtime_ns: row
                    .get::<i64>(2)
                    .map_err(|e| OriginError::VectorDb(format!("sync mtime_ns: {e}")))?,
                content_hash: row
                    .get::<String>(3)
                    .map_err(|e| OriginError::VectorDb(format!("sync content_hash: {e}")))?,
                last_synced_at: row
                    .get::<i64>(4)
                    .map_err(|e| OriginError::VectorDb(format!("sync last_synced_at: {e}")))?,
            }))
        } else {
            Ok(None)
        }
    }

    /// List all tracked file paths for a source.
    pub async fn list_sync_state_paths(&self, source_id: &str) -> Result<Vec<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT file_path FROM source_sync_state WHERE source_id = ?1",
                libsql::params![source_id.to_string()],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_sync_state_paths: {}", e)))?;
        let mut paths = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_sync_state_paths row: {}", e)))?
        {
            paths.push(
                row.get::<String>(0)
                    .map_err(|e| OriginError::VectorDb(format!("sync path: {e}")))?,
            );
        }
        Ok(paths)
    }

    /// Delete sync state for a specific file in a source.
    pub async fn delete_sync_state(
        &self,
        source_id: &str,
        file_path: &str,
    ) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM source_sync_state WHERE source_id = ?1 AND file_path = ?2",
            libsql::params![source_id.to_string(), file_path.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_sync_state: {}", e)))?;
        Ok(())
    }

    /// Delete all sync state entries for a source.
    pub async fn delete_all_sync_state(&self, source_id: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "DELETE FROM source_sync_state WHERE source_id = ?1",
            libsql::params![source_id.to_string()],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("delete_all_sync_state: {}", e)))?;
        Ok(())
    }

    /// Check which of the given source_ids already exist in the memories
    /// table. Used for skip-existing dedup during bulk chat import.
    ///
    /// Returns a HashSet of source_ids that are present.
    pub async fn check_existing_import_source_ids(
        &self,
        candidates: &[String],
    ) -> Result<std::collections::HashSet<String>, OriginError> {
        use std::collections::HashSet;

        if candidates.is_empty() {
            return Ok(HashSet::new());
        }

        let conn = self.conn.lock().await;

        // Query in batches to avoid SQL parameter limit (SQLite default is 999).
        const BATCH: usize = 500;
        let mut found: HashSet<String> = HashSet::new();

        for chunk in candidates.chunks(BATCH) {
            let placeholders = std::iter::repeat_n("?", chunk.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "SELECT DISTINCT source_id FROM memories WHERE source_id IN ({placeholders})"
            );
            let params: Vec<libsql::Value> = chunk
                .iter()
                .map(|s| libsql::Value::Text(s.clone()))
                .collect();
            let mut rows = conn
                .query(&sql, params)
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            while let Some(row) = rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
            {
                let sid: String = row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(e.to_string()))?;
                found.insert(sid);
            }
        }

        Ok(found)
    }

    /// Store a single memory from a bulk chat import. Skips classification,
    /// extraction, and event emission — those happen later via the refinery
    /// steep cycle. Uses `source = 'memory'` and `memory_type = NULL` so the row
    /// matches the existing `import_%` filter in `get_unclassified_imports`.
    pub async fn store_raw_import_memory(
        &self,
        source_id: &str,
        content: &str,
        title: Option<&str>,
        created_at: Option<chrono::DateTime<chrono::Utc>>,
        chunk_index: i64,
    ) -> Result<String, OriginError> {
        let memory_id = format!("mem_{}", uuid::Uuid::new_v4());
        let now_ts = chrono::Utc::now().timestamp();
        let created_ts = created_at.map(|ts| ts.timestamp()).unwrap_or(now_ts);
        let title_str = title.unwrap_or("");
        let word_count = content.split_whitespace().count() as i64;

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, word_count, created_at)
             VALUES (?1, ?2, 'memory', ?3, ?4, ?5, ?6, 'text', ?7, ?8)",
            libsql::params![
                memory_id.clone(),
                content.to_string(),
                source_id.to_string(),
                title_str.to_string(),
                chunk_index,
                now_ts,
                word_count,
                created_ts,
            ],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("store_raw_import_memory: {}", e)))?;
        Ok(memory_id)
    }

    /// Store multiple raw import memories inside a single transaction.
    ///
    /// Each entry is `(source_id, content, title, created_at, chunk_index)`.
    /// On error the entire batch is rolled back.
    #[allow(clippy::type_complexity)]
    pub async fn store_raw_import_memories_batch(
        &self,
        entries: &[(
            String,
            String,
            Option<String>,
            Option<chrono::DateTime<chrono::Utc>>,
            i64,
        )],
    ) -> Result<usize, OriginError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("batch import begin: {e}")))?;

        let mut count = 0usize;
        for (source_id, content, title, created_at, chunk_index) in entries {
            let memory_id = format!("mem_{}", uuid::Uuid::new_v4());
            let now_ts = chrono::Utc::now().timestamp();
            let created_ts = created_at.map(|ts| ts.timestamp()).unwrap_or(now_ts);
            let title_str = title.as_deref().unwrap_or("");
            let word_count = content.split_whitespace().count() as i64;

            if let Err(e) = conn.execute(
                "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, word_count, created_at)
                 VALUES (?1, ?2, 'memory', ?3, ?4, ?5, ?6, 'text', ?7, ?8)",
                libsql::params![
                    memory_id,
                    content.clone(),
                    source_id.clone(),
                    title_str.to_string(),
                    *chunk_index,
                    now_ts,
                    word_count,
                    created_ts,
                ],
            ).await {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(OriginError::VectorDb(format!("batch import insert: {e}")));
            }
            count += 1;
        }

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("batch import commit: {e}")))?;
        Ok(count)
    }

    // ===== ImportState CRUD =====

    /// Create a new import-state row in the `parsing` stage.
    pub async fn start_import_state(
        &self,
        id: &str,
        vendor: crate::chat_import::types::Vendor,
        source_path: &str,
    ) -> Result<(), OriginError> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO import_state (id, vendor, source_path, processed_conversations, stage, started_at, updated_at)
             VALUES (?, ?, ?, 0, 'parsing', ?, ?)",
            libsql::params![id, vendor.as_str(), source_path, now.clone(), now],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("start_import_state: {}", e)))?;
        Ok(())
    }

    /// Update the stage of an existing import, optionally setting total, processed counts, and error message.
    pub async fn update_import_state_stage(
        &self,
        id: &str,
        stage: crate::chat_import::bulk_ingest::ImportStage,
        total: Option<i64>,
        processed: Option<i64>,
    ) -> Result<(), OriginError> {
        self.update_import_state_stage_with_error(id, stage, total, processed, None)
            .await
    }

    /// Update the stage of an existing import, with an optional error message.
    pub async fn update_import_state_stage_with_error(
        &self,
        id: &str,
        stage: crate::chat_import::bulk_ingest::ImportStage,
        total: Option<i64>,
        processed: Option<i64>,
        error_message: Option<&str>,
    ) -> Result<(), OriginError> {
        let now = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE import_state SET stage = ?, updated_at = ?,
             total_conversations = COALESCE(?, total_conversations),
             processed_conversations = COALESCE(?, processed_conversations),
             error_message = COALESCE(?, error_message)
             WHERE id = ?",
            libsql::params![stage.as_str(), now, total, processed, error_message, id],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("update_import_state_stage: {}", e)))?;
        Ok(())
    }

    /// Load an import-state row by ID.
    pub async fn load_import_state(
        &self,
        id: &str,
    ) -> Result<Option<crate::chat_import::bulk_ingest::ImportState>, OriginError> {
        use crate::chat_import::bulk_ingest::{ImportStage, ImportState};
        use crate::chat_import::types::Vendor;

        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, vendor, source_path, total_conversations, processed_conversations, stage, error_message, started_at, updated_at
                 FROM import_state WHERE id = ?",
                libsql::params![id],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("load_import_state query: {}", e)))?;

        let row = match rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("load_import_state next: {}", e)))?
        {
            Some(r) => r,
            None => return Ok(None),
        };

        let row_id: String = row
            .get(0)
            .map_err(|e| OriginError::VectorDb(format!("load_import_state col id: {}", e)))?;
        let vendor_str: String = row
            .get(1)
            .map_err(|e| OriginError::VectorDb(format!("load_import_state col vendor: {}", e)))?;
        let source_path: String = row.get(2).map_err(|e| {
            OriginError::VectorDb(format!("load_import_state col source_path: {}", e))
        })?;
        let total_conversations: Option<i64> = row.get(3).ok();
        let processed_conversations: i64 = row.get(4).map_err(|e| {
            OriginError::VectorDb(format!("load_import_state col processed: {}", e))
        })?;
        let stage_str: String = row
            .get(5)
            .map_err(|e| OriginError::VectorDb(format!("load_import_state col stage: {}", e)))?;
        let error_message: Option<String> = row.get(6).ok();
        let started_at_str: String = row.get(7).map_err(|e| {
            OriginError::VectorDb(format!("load_import_state col started_at: {}", e))
        })?;
        let updated_at_str: String = row.get(8).map_err(|e| {
            OriginError::VectorDb(format!("load_import_state col updated_at: {}", e))
        })?;

        let vendor = Vendor::from_str(&vendor_str).ok_or_else(|| {
            OriginError::VectorDb(format!(
                "load_import_state: unknown vendor '{}'",
                vendor_str
            ))
        })?;
        let stage = ImportStage::from_str(&stage_str).ok_or_else(|| {
            OriginError::VectorDb(format!("load_import_state: unknown stage '{}'", stage_str))
        })?;
        let started_at = chrono::DateTime::parse_from_rfc3339(&started_at_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| {
                OriginError::VectorDb(format!("load_import_state parse started_at: {}", e))
            })?;
        let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map_err(|e| {
                OriginError::VectorDb(format!("load_import_state parse updated_at: {}", e))
            })?;

        Ok(Some(ImportState {
            id: row_id,
            vendor,
            source_path,
            total_conversations,
            processed_conversations,
            stage,
            error_message,
            started_at,
            updated_at,
        }))
    }

    /// Return all imports whose stage is not terminal (done/error).
    pub async fn list_pending_imports(
        &self,
    ) -> Result<Vec<crate::chat_import::bulk_ingest::ImportState>, OriginError> {
        use crate::chat_import::bulk_ingest::{ImportStage, ImportState};
        use crate::chat_import::types::Vendor;

        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, vendor, source_path, total_conversations, processed_conversations, stage, error_message, started_at, updated_at
                 FROM import_state
                 WHERE stage NOT IN ('done', 'error')
                   AND NOT (stage = 'stage_b' AND total_conversations IS NOT NULL
                            AND processed_conversations >= total_conversations)
                 ORDER BY started_at DESC",
                libsql::params![],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_pending_imports query: {}", e)))?;

        let mut result = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("list_pending_imports next: {}", e)))?
        {
            let row_id: String = row.get(0).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col id: {}", e))
            })?;
            let vendor_str: String = row.get(1).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col vendor: {}", e))
            })?;
            let source_path: String = row.get(2).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col source_path: {}", e))
            })?;
            let total_conversations: Option<i64> = row.get(3).ok();
            let processed_conversations: i64 = row.get(4).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col processed: {}", e))
            })?;
            let stage_str: String = row.get(5).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col stage: {}", e))
            })?;
            let error_message: Option<String> = row.get(6).ok();
            let started_at_str: String = row.get(7).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col started_at: {}", e))
            })?;
            let updated_at_str: String = row.get(8).map_err(|e| {
                OriginError::VectorDb(format!("list_pending_imports col updated_at: {}", e))
            })?;

            let vendor = Vendor::from_str(&vendor_str).ok_or_else(|| {
                OriginError::VectorDb(format!(
                    "list_pending_imports: unknown vendor '{}'",
                    vendor_str
                ))
            })?;
            let stage = ImportStage::from_str(&stage_str).ok_or_else(|| {
                OriginError::VectorDb(format!(
                    "list_pending_imports: unknown stage '{}'",
                    stage_str
                ))
            })?;
            let started_at = chrono::DateTime::parse_from_rfc3339(&started_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| {
                    OriginError::VectorDb(format!("list_pending_imports parse started_at: {}", e))
                })?;
            let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| {
                    OriginError::VectorDb(format!("list_pending_imports parse updated_at: {}", e))
                })?;

            result.push(ImportState {
                id: row_id,
                vendor,
                source_path,
                total_conversations,
                processed_conversations,
                stage,
                error_message,
                started_at,
                updated_at,
            });
        }

        Ok(result)
    }

    // ==================== App Metadata ====================

    /// Get a value from the app_metadata key-value store.
    pub async fn get_app_metadata(&self, key: &str) -> Result<Option<String>, OriginError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT value FROM app_metadata WHERE key = ?1",
                libsql::params![key],
            )
            .await
            .map_err(|e| OriginError::VectorDb(format!("get_app_metadata: {}", e)))?;
        match rows.next().await {
            Ok(Some(row)) => {
                let value: String = row
                    .get(0)
                    .map_err(|e| OriginError::VectorDb(format!("get_app_metadata read: {}", e)))?;
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(OriginError::VectorDb(format!(
                "get_app_metadata next: {}",
                e
            ))),
        }
    }

    /// Set a value in the app_metadata key-value store (upsert).
    pub async fn set_app_metadata(&self, key: &str, value: &str) -> Result<(), OriginError> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO app_metadata (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            libsql::params![key, value],
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("set_app_metadata: {}", e)))?;
        Ok(())
    }
}

/// Derive the activity badge for a single memory row.
///
/// All time values (`created_at`, `last_modified`, `since`) are in **Unix seconds**
/// (the unit used by the `memories` table).
///
/// Badge precedence:
/// 1. `NeedsReview` - memory is in `refinement_queue/awaiting_review` (wins over all)
/// 2. `New` - `created_at >= since`
/// 3. `Refined` - modified after `since`, with a >60s grace period past `created_at`,
///    and either `enrichment_status='enriched'` or an entity link present
/// 4. `None` - none of the above
pub fn derive_memory_badge(
    source_id: &str,
    created_at: i64,
    last_modified: Option<i64>,
    enrichment_status: &str,
    entity_id: Option<&str>,
    since: Option<i64>,
    flagged: &std::collections::HashSet<String>,
) -> origin_types::ActivityBadge {
    use origin_types::ActivityBadge;

    // Rule 1: NeedsReview overrides everything.
    if flagged.contains(source_id) {
        return ActivityBadge::NeedsReview;
    }

    let Some(since) = since else {
        return ActivityBadge::None;
    };

    // Rule 2: created after the watermark → New.
    if created_at >= since {
        return ActivityBadge::New;
    }

    // Rule 3: Refined — modified after watermark, outside the 60-second post-ingest grace.
    let lm = last_modified.unwrap_or(created_at);
    let has_link = entity_id.map(|s| !s.is_empty()).unwrap_or(false);
    const GRACE_S: i64 = 60; // seconds — matches the 60_000 ms spec value
    if lm >= since && lm > created_at + GRACE_S && (enrichment_status == "enriched" || has_link) {
        return ActivityBadge::Refined;
    }

    ActivityBadge::None
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::OnceLock;
    use tempfile::tempdir;

    /// Shared embedder singleton so the ONNX model is loaded exactly once
    /// across all tests (avoids concurrent download/read race).
    fn shared_embedder() -> Arc<std::sync::Mutex<TextEmbedding>> {
        static EMBEDDER: OnceLock<Arc<std::sync::Mutex<TextEmbedding>>> = OnceLock::new();
        EMBEDDER
            .get_or_init(|| {
                // Reuse the daemon's FastEmbed cache so tests don't try to
                // re-download the 140 MB ONNX model (which fails on hosts
                // with restricted network / HuggingFace TLS bundle issues,
                // e.g. the `OSStatus -26276` symptom we hit 2026-04-16).
                // We pass a bogus `db_path` — `resolve_fastembed_cache_dir`
                // will skip the per-DB branch (its tempdir-style db_path is
                // empty) and walk the env + shared-host candidates.
                let mut opts = InitOptions::new(EmbeddingModel::BGEBaseENV15Q)
                    .with_show_download_progress(false);
                if let Some(cache) =
                    resolve_fastembed_cache_dir(std::path::Path::new(".nonexistent"))
                {
                    opts = opts.with_cache_dir(cache);
                }
                let emb = TextEmbedding::try_new(opts).expect(
                    "failed to init embedder for tests (set ORIGIN_TEST_FASTEMBED_CACHE or run the daemon once to populate the shared cache)",
                );
                Arc::new(std::sync::Mutex::new(emb))
            })
            .clone()
    }

    /// Helper: create a MemoryDB backed by a temp directory, reusing
    /// the shared embedder so tests can run in parallel safely.
    pub async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempdir().expect("failed to create temp dir");
        let db_path = dir.path();
        std::fs::create_dir_all(db_path).unwrap();
        let db_file = db_path.join("origin_memory.db");

        let db = libsql::Builder::new_local(db_file.to_str().unwrap())
            .build()
            .await
            .expect("libsql open");
        let conn = db.connect().expect("libsql connect");
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .await
            .unwrap();
        conn.execute_batch(SCHEMA).await.unwrap();
        let _ = conn.execute_batch(FTS_SCHEMA).await;
        let _ = conn.execute_batch(FTS_TRIGGERS).await;

        // Create vector indexes (same as MemoryDB::new does)
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS memories_vec_idx ON memories (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )",
                (),
            )
            .await;
        let _ = conn
            .execute(
                "CREATE INDEX IF NOT EXISTS entities_vec_idx ON entities (
                    libsql_vector_idx(embedding, 'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32')
                )",
                (),
            )
            .await;

        let memory_db = MemoryDB {
            _db: db,
            conn: tokio::sync::Mutex::new(conn),
            embedder: shared_embedder(),
            chunker: ChunkingEngine::new(),
            embedding_cache: std::sync::Mutex::new(EmbeddingCache::new(200)),
        };
        memory_db
            .run_migrations(&crate::events::NoopEmitter)
            .await
            .unwrap();
        (memory_db, dir)
    }

    /// Helper: build a minimal RawDocument for testing.
    fn make_doc(source: &str, source_id: &str, title: &str, content: &str) -> RawDocument {
        RawDocument {
            source: source.to_string(),
            source_id: source_id.to_string(),
            title: title.to_string(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
            memory_type: None,
            domain: None,
            source_agent: None,
            confidence: None,
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            ..Default::default()
        }
    }

    /// Helper: build a RawDocument with memory-layer fields set.
    fn make_memory_doc(
        source_id: &str,
        content: &str,
        memory_type: &str,
        domain: &str,
        source_agent: &str,
    ) -> RawDocument {
        let supersede_mode = if memory_type == "decision" {
            "archive".to_string()
        } else {
            "hide".to_string()
        };
        RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: format!("memory-{}", source_id),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: HashMap::new(),
            memory_type: Some(memory_type.to_string()),
            domain: Some(domain.to_string()),
            source_agent: Some(source_agent.to_string()),
            confidence: Some(0.9),
            confirmed: Some(false),
            supersedes: None,
            pending_revision: false,
            supersede_mode,
            ..Default::default()
        }
    }

    // ==================== MemoryDB::new ====================

    #[tokio::test]
    async fn test_new_creates_db_file() {
        let dir = tempdir().unwrap();
        let db = MemoryDB::new(dir.path(), Arc::new(crate::events::NoopEmitter)).await;
        assert!(db.is_ok(), "MemoryDB::new should succeed");
        let db_file = dir.path().join("origin_memory.db");
        assert!(db_file.exists(), "DB file should be created on disk");
    }

    #[tokio::test]
    async fn test_new_creates_tables() {
        let (db, _dir) = test_db().await;
        // Verify tables exist by querying them
        let count = db.count().await.expect("count should work on empty db");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_new_idempotent() {
        let dir = tempdir().unwrap();
        // Create twice on the same path — should not error
        let _db1 = MemoryDB::new(dir.path(), Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();
        drop(_db1);
        let _db2 = MemoryDB::new(dir.path(), Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();
    }

    // ==================== upsert_documents ====================

    #[tokio::test]
    async fn test_upsert_single_document() {
        let (db, _dir) = test_db().await;
        let doc = make_doc(
            "local_files",
            "file1",
            "test.txt",
            "Hello world, this is a test document with some content for chunking.",
        );
        let count = db.upsert_documents(vec![doc]).await.unwrap();
        assert!(count >= 1, "should create at least 1 memory");

        let total = db.count().await.unwrap();
        assert_eq!(total, count as u64);
    }

    #[tokio::test]
    async fn test_upsert_multiple_documents() {
        let (db, _dir) = test_db().await;
        let docs = vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "First document about Rust programming language features.",
            ),
            make_doc(
                "local_files",
                "f2",
                "b.txt",
                "Second document about Python machine learning libraries.",
            ),
        ];
        let count = db.upsert_documents(docs).await.unwrap();
        assert!(count >= 2, "should create at least 2 memories (1 per doc)");
    }

    #[tokio::test]
    async fn test_upsert_empty_vec() {
        let (db, _dir) = test_db().await;
        let count = db.upsert_documents(vec![]).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_upsert_replaces_existing() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("local_files", "f1", "a.txt", "Original content about Rust.");
        db.upsert_documents(vec![doc]).await.unwrap();
        let count1 = db.count().await.unwrap();

        // Upsert the same source_id with new content — old memories should be deleted first
        let doc2 = make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Updated content about Python.",
        );
        db.upsert_documents(vec![doc2]).await.unwrap();
        let count2 = db.count().await.unwrap();

        // Should have roughly the same number of memories, not double
        assert_eq!(count1, count2, "upsert should replace, not accumulate");
    }

    #[tokio::test]
    async fn test_upsert_preserves_memory_fields() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem1",
            "User prefers dark mode in all editors.",
            "preference",
            "personal",
            "claude-code",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        let memories = db
            .get_memories_by_source_id("memory", "mem1")
            .await
            .unwrap();
        assert!(!memories.is_empty(), "should have memories");

        // Verify the data round-tripped via list_filtered
        let listed = db
            .list_filtered(Some("memory"), None, None, 10)
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].source_id, "mem1");
    }

    #[tokio::test]
    async fn test_store_memory_uses_structured_content() {
        let (db, _dir) = test_db().await;
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "mem_struct".to_string(),
            title: "dark mode preference".to_string(),
            content: "I prefer dark mode in all editors".to_string(),
            memory_type: Some("preference".to_string()),
            structured_fields: Some(
                r#"{"preference":"dark mode","applies_when":"editors, terminals"}"#.to_string(),
            ),
            last_modified: chrono::Utc::now().timestamp(),
            enrichment_status: "enriched".to_string(),
            supersede_mode: "hide".to_string(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT content, source_text FROM memories WHERE source_id = 'mem_struct'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let content: String = row.get(0).unwrap();
        let source_text: Option<String> = row.get::<Option<String>>(1).unwrap();

        // Content should always be the original natural language prose (no flattening)
        assert_eq!(
            content, "I prefer dark mode in all editors",
            "content should be prose: {}",
            content
        );
        // source_text should be NULL — we no longer overwrite it with a copy of the prose
        assert!(
            source_text.is_none(),
            "source_text should be NULL when content is prose: {:?}",
            source_text
        );
    }

    // ==================== search (FTS path) ====================

    #[tokio::test]
    async fn test_search_returns_results() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "rust.txt",
                "Rust is a systems programming language focused on safety and performance.",
            ),
            make_doc(
                "local_files",
                "f2",
                "python.txt",
                "Python is a high-level programming language known for its simplicity.",
            ),
        ])
        .await
        .unwrap();

        let results = db.search("Rust programming", 10, None).await.unwrap();
        // Should find at least the Rust document (via FTS even if vector index is absent)
        assert!(!results.is_empty(), "search should return results");
    }

    #[tokio::test]
    async fn test_search_with_source_filter() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "Cats are wonderful pets that bring joy.",
            ),
            make_doc(
                "memory",
                "m1",
                "b.txt",
                "Dogs are loyal companions and great friends.",
            ),
        ])
        .await
        .unwrap();

        let results = db
            .search("pets animals", 10, Some("local_files"))
            .await
            .unwrap();
        for r in &results {
            assert_eq!(r.source, "local_files", "source filter should be applied");
        }
    }

    #[tokio::test]
    async fn test_search_empty_query() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Some content about databases and SQL.",
        )])
        .await
        .unwrap();
        // Empty or very short query — should not panic
        let results = db.search("", 10, None).await;
        // May return results or error, but should not panic
        assert!(results.is_ok() || results.is_err());
    }

    #[tokio::test]
    async fn test_search_respects_limit() {
        let (db, _dir) = test_db().await;
        let mut docs = Vec::new();
        for i in 0..10 {
            docs.push(make_doc(
                "local_files",
                &format!("f{}", i),
                &format!("doc{}.txt", i),
                &format!(
                    "Programming language number {} is interesting for different reasons.",
                    i
                ),
            ));
        }
        db.upsert_documents(docs).await.unwrap();

        let results = db.search("programming language", 3, None).await.unwrap();
        assert!(results.len() <= 3, "should respect limit");
    }

    // ==================== search_memory ====================

    #[tokio::test]
    async fn test_search_memory_basic() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc(
                "m1",
                "Favorite color is blue for UI elements",
                "preference",
                "personal",
                "claude-code",
            ),
            make_memory_doc(
                "m2",
                "Project deadline is next Friday for release",
                "fact",
                "work",
                "chatgpt",
            ),
        ])
        .await
        .unwrap();

        // Hybrid search (vector + FTS) should find results via at least FTS
        let results = db
            .search_memory("color blue", 10, None, None, None, None, None, None)
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "search_memory should find results via hybrid search"
        );
        assert!(
            results[0].content.contains("color"),
            "top result should match query"
        );
    }

    #[tokio::test]
    async fn test_search_memory_with_filters() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc(
                "m1",
                "Favorite color is blue for UI elements",
                "preference",
                "personal",
                "claude-code",
            ),
            make_memory_doc(
                "m2",
                "Project deadline is next Friday for release",
                "fact",
                "work",
                "chatgpt",
            ),
        ])
        .await
        .unwrap();

        // Filter by memory_type should narrow results
        let results = db
            .search_memory(
                "deadline Friday",
                10,
                Some("fact"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "should find fact memory");
        assert!(
            results
                .iter()
                .all(|r| r.memory_type.as_deref() == Some("fact")),
            "all results should have memory_type=fact"
        );

        // Filter by source_agent
        let results = db
            .search_memory(
                "color blue",
                10,
                None,
                None,
                Some("claude-code"),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "should find claude-code memory");
        assert!(
            results
                .iter()
                .all(|r| r.source_agent.as_deref() == Some("claude-code")),
            "all results should have source_agent=claude-code"
        );
    }

    // ==================== search_memory: archive-visible, quality, type normalization ====================

    #[tokio::test]
    async fn test_superseded_decision_archive_visible() {
        let (db, _dir) = test_db().await;

        // Store old decision — decisions auto-get supersede_mode='archive'
        let old_doc = make_memory_doc(
            "mem_old",
            "We chose MongoDB for the database system",
            "decision",
            "engineering",
            "claude",
        );
        db.upsert_documents(vec![old_doc]).await.unwrap();

        // Store new decision that supersedes old one
        let mut new_doc = make_memory_doc(
            "mem_new",
            "We chose PostgreSQL for the database system instead",
            "decision",
            "engineering",
            "claude",
        );
        new_doc.supersedes = Some("mem_old".to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        // Both decisions should still appear in search (archive-visible mode for decisions)
        let results = db
            .search_memory("database system", 10, None, None, None, None, None, None)
            .await
            .unwrap();
        let source_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        assert!(
            source_ids.contains(&"mem_new"),
            "new decision should appear"
        );
        assert!(
            source_ids.contains(&"mem_old"),
            "old decision should appear (archive-visible)"
        );

        // Old decision should be marked as archived
        let old_result = results.iter().find(|r| r.source_id == "mem_old").unwrap();
        assert!(
            old_result.is_archived,
            "old decision should be marked archived"
        );
    }

    #[tokio::test]
    async fn test_superseded_hide_mode_excluded() {
        let (db, _dir) = test_db().await;

        // Store old fact — facts auto-get supersede_mode='hide'
        let old_doc = make_memory_doc(
            "hide_old",
            "Favorite color is green for dashboards",
            "fact",
            "personal",
            "claude",
        );
        db.upsert_documents(vec![old_doc]).await.unwrap();

        // Store new fact that supersedes old one (hide mode)
        let mut new_doc = make_memory_doc(
            "hide_new",
            "Favorite color is blue for dashboards",
            "fact",
            "personal",
            "claude",
        );
        new_doc.supersedes = Some("hide_old".to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        let results = db
            .search_memory(
                "favorite color dashboards",
                10,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let source_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        assert!(source_ids.contains(&"hide_new"), "new fact should appear");
        assert!(
            !source_ids.contains(&"hide_old"),
            "old fact should be hidden (supersede_mode=hide)"
        );
    }

    #[tokio::test]
    async fn test_get_memory_contents_by_ids_excludes_superseded_hide() {
        let (db, _dir) = test_db().await;

        // Store old fact
        let old_doc = make_memory_doc(
            "recompile_old",
            "I prefer Python for all projects",
            "fact",
            "personal",
            "claude",
        );
        db.upsert_documents(vec![old_doc]).await.unwrap();

        // Store new fact that supersedes old one (hide mode)
        let mut new_doc = make_memory_doc(
            "recompile_new",
            "I prefer Rust for all projects",
            "fact",
            "personal",
            "claude",
        );
        new_doc.supersedes = Some("recompile_old".to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        // get_memory_contents_by_ids should exclude the superseded memory
        let results = db
            .get_memory_contents_by_ids(&["recompile_old".to_string(), "recompile_new".to_string()])
            .await
            .unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            !ids.contains(&"recompile_old"),
            "superseded (hide) memory should be excluded from concept recompilation"
        );
        assert!(
            ids.contains(&"recompile_new"),
            "current memory should be included"
        );
    }

    #[tokio::test]
    async fn test_get_memory_contents_by_ids_includes_superseded_archive() {
        let (db, _dir) = test_db().await;

        // Store old decision (decisions get supersede_mode='archive')
        let old_doc = make_memory_doc(
            "decision_old",
            "We chose PostgreSQL for the database",
            "decision",
            "work",
            "claude",
        );
        db.upsert_documents(vec![old_doc]).await.unwrap();

        // Store new decision that supersedes old one (archive mode)
        let mut new_doc = make_memory_doc(
            "decision_new",
            "We migrated from PostgreSQL to libSQL",
            "decision",
            "work",
            "claude",
        );
        new_doc.supersedes = Some("decision_old".to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        // get_memory_contents_by_ids should INCLUDE archived decisions
        let results = db
            .get_memory_contents_by_ids(&["decision_old".to_string(), "decision_new".to_string()])
            .await
            .unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"decision_old"),
            "archived (decision) memory should still be included"
        );
        assert!(
            ids.contains(&"decision_new"),
            "current memory should be included"
        );
    }

    #[tokio::test]
    async fn test_quality_multiplier_in_search() {
        let (db, _dir) = test_db().await;

        // Use very similar but not identical content so base RRF scores are close
        // but near-duplicate dedup (Jaccard > 0.92) doesn't collapse them
        let doc_high = make_memory_doc(
            "q_high",
            "Kubernetes orchestrates containers in production clusters for scalable workloads",
            "fact",
            "devops",
            "claude",
        );
        let doc_low = make_memory_doc(
            "q_low",
            "Kubernetes orchestrates containers in staging clusters for development workloads",
            "fact",
            "devops",
            "claude",
        );
        db.upsert_documents(vec![doc_high, doc_low]).await.unwrap();

        // Set quality: high for one, low for the other
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE memories SET quality = 'high' WHERE source_id = 'q_high'",
                libsql::params![],
            )
            .await
            .unwrap();
            conn.execute(
                "UPDATE memories SET quality = 'low' WHERE source_id = 'q_low'",
                libsql::params![],
            )
            .await
            .unwrap();
        }

        let results = db
            .search_memory(
                "Kubernetes container orchestration",
                10,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let high_result = results.iter().find(|r| r.source_id == "q_high");
        let low_result = results.iter().find(|r| r.source_id == "q_low");
        if let (Some(h), Some(l)) = (high_result, low_result) {
            assert!(
                h.score > l.score,
                "high quality ({}) should score higher than low quality ({})",
                h.score,
                l.score
            );
        } else {
            assert!(high_result.is_some(), "q_high should appear in results");
            assert!(low_result.is_some(), "q_low should appear in results");
        }
    }

    #[tokio::test]
    async fn test_normalize_type_filter() {
        assert_eq!(MemoryDB::normalize_type_filter("correction"), "fact");
        assert_eq!(MemoryDB::normalize_type_filter("custom"), "fact");
        assert_eq!(MemoryDB::normalize_type_filter("recap"), "fact");
        assert_eq!(MemoryDB::normalize_type_filter("identity"), "identity");
        assert_eq!(
            MemoryDB::normalize_type_filter("correction,custom"),
            "fact,fact"
        );
        assert_eq!(
            MemoryDB::normalize_type_filter("identity,correction,decision"),
            "identity,fact,decision"
        );
    }

    #[tokio::test]
    async fn test_search_memory_old_type_maps_to_new() {
        let (db, _dir) = test_db().await;

        // Store a fact memory with tokens that exactly match the search query
        db.upsert_documents(vec![make_memory_doc(
            "compat1",
            "Favorite color is blue for UI elements",
            "fact",
            "engineering",
            "claude",
        )])
        .await
        .unwrap();

        // Search with old type name "correction" should map to "fact" and still find results
        let results = db
            .search_memory(
                "color blue",
                10,
                Some("correction"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "old type 'correction' should map to 'fact' and find results"
        );
    }

    // ==================== delete_by_source_id ====================

    #[tokio::test]
    async fn test_delete_by_source_id() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "Content about Rust programming.",
            ),
            make_doc(
                "local_files",
                "f2",
                "b.txt",
                "Content about Python scripting.",
            ),
        ])
        .await
        .unwrap();

        let before = db.count().await.unwrap();
        assert!(before >= 2);

        db.delete_by_source_id("local_files", "f1").await.unwrap();

        let after = db.count().await.unwrap();
        assert!(after < before, "count should decrease after delete");

        // f2 should still exist
        let memories = db
            .get_memories_by_source_id("local_files", "f2")
            .await
            .unwrap();
        assert!(!memories.is_empty(), "f2 should survive");

        // f1 should be gone
        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(memories.is_empty(), "f1 should be deleted");
    }

    #[tokio::test]
    async fn test_delete_by_source_id_nonexistent() {
        let (db, _dir) = test_db().await;
        // Should not error on missing source_id
        let result = db.delete_by_source_id("local_files", "nonexistent").await;
        assert!(result.is_ok());
    }

    // ==================== delete_bulk ====================

    #[tokio::test]
    async fn test_delete_bulk() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "Content about apples and oranges.",
            ),
            make_doc(
                "local_files",
                "f2",
                "b.txt",
                "Content about bananas and grapes.",
            ),
            make_doc(
                "local_files",
                "f3",
                "c.txt",
                "Content about cherries and berries.",
            ),
        ])
        .await
        .unwrap();

        db.delete_bulk(&[
            ("local_files".to_string(), "f1".to_string()),
            ("local_files".to_string(), "f3".to_string()),
        ])
        .await
        .unwrap();

        // Only f2 should remain
        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(memories.is_empty());
        let memories = db
            .get_memories_by_source_id("local_files", "f2")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        let memories = db
            .get_memories_by_source_id("local_files", "f3")
            .await
            .unwrap();
        assert!(memories.is_empty());
    }

    #[tokio::test]
    async fn test_delete_bulk_empty() {
        let (db, _dir) = test_db().await;
        let result = db.delete_bulk(&[]).await;
        assert!(result.is_ok());
    }

    // ==================== delete_by_time_range ====================

    #[tokio::test]
    async fn test_delete_by_time_range() {
        let (db, _dir) = test_db().await;

        let mut doc1 = make_doc(
            "local_files",
            "f1",
            "old.txt",
            "Old content from the past era.",
        );
        doc1.last_modified = 1000;
        let mut doc2 = make_doc(
            "local_files",
            "f2",
            "new.txt",
            "New content from today's time.",
        );
        doc2.last_modified = 2000;
        let mut doc3 = make_doc(
            "local_files",
            "f3",
            "future.txt",
            "Future content coming soon tomorrow.",
        );
        doc3.last_modified = 3000;

        db.upsert_documents(vec![doc1, doc2, doc3]).await.unwrap();

        let deleted = db.delete_by_time_range(1500, 2500).await.unwrap();
        assert!(
            deleted >= 1,
            "should delete at least the doc at timestamp 2000"
        );

        // f1 (ts=1000) should survive
        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty(), "f1 (old) should survive");

        // f2 (ts=2000) should be deleted
        let memories = db
            .get_memories_by_source_id("local_files", "f2")
            .await
            .unwrap();
        assert!(memories.is_empty(), "f2 (in range) should be deleted");

        // f3 (ts=3000) should survive
        let memories = db
            .get_memories_by_source_id("local_files", "f3")
            .await
            .unwrap();
        assert!(!memories.is_empty(), "f3 (future) should survive");
    }

    // ==================== update_document_summary ====================

    #[tokio::test]
    async fn test_update_document_summary() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "notes.txt",
            "Meeting notes about the quarterly review and budget planning.",
        )])
        .await
        .unwrap();

        db.update_document_summary("f1", "local_files", "Summary of meeting notes")
            .await
            .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        for memory in &memories {
            assert_eq!(
                memory.summary.as_deref(),
                Some("Summary of meeting notes"),
                "all memories should have updated summary"
            );
        }
    }

    // ==================== update_column_by_source_id ====================

    #[tokio::test]
    async fn test_update_column_whitelist_allowed() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Some content for testing column updates.",
        )])
        .await
        .unwrap();

        // title is in the whitelist
        let result = db
            .update_column_by_source_id("local_files", "f1", "title", "New Title")
            .await;
        assert!(result.is_ok());

        // summary is in the whitelist
        let result = db
            .update_column_by_source_id("local_files", "f1", "summary", "A summary")
            .await;
        assert!(result.is_ok());

        // confirmed is in the whitelist
        let result = db
            .update_column_by_source_id("local_files", "f1", "confirmed", "1")
            .await;
        assert!(result.is_ok());

        // domain is in the whitelist
        let result = db
            .update_column_by_source_id("local_files", "f1", "domain", "work")
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_update_column_whitelist_blocked() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Content for blocked column test.",
        )])
        .await
        .unwrap();

        // "id" is NOT in the whitelist — should be rejected
        let result = db
            .update_column_by_source_id("local_files", "f1", "id", "hacked")
            .await;
        assert!(result.is_err(), "updating 'id' column should be rejected");

        // "embedding" is NOT in the whitelist
        let result = db
            .update_column_by_source_id("local_files", "f1", "embedding", "bad")
            .await;
        assert!(
            result.is_err(),
            "updating 'embedding' column should be rejected"
        );

        // arbitrary column name
        let result = db
            .update_column_by_source_id("local_files", "f1", "DROP TABLE", "oops")
            .await;
        assert!(result.is_err(), "SQL injection attempt should be rejected");
    }

    #[tokio::test]
    async fn test_update_column_actually_persists() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Original title test content for persistence check.",
        )])
        .await
        .unwrap();

        db.update_column_by_source_id("local_files", "f1", "title", "Updated Title")
            .await
            .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        assert_eq!(memories[0].title, "Updated Title");
    }

    // ==================== count ====================

    #[tokio::test]
    async fn test_count_empty() {
        let (db, _dir) = test_db().await;
        assert_eq!(db.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_count_after_inserts() {
        let (db, _dir) = test_db().await;
        let n = db
            .upsert_documents(vec![
                make_doc(
                    "local_files",
                    "f1",
                    "a.txt",
                    "First doc content about programming.",
                ),
                make_doc(
                    "local_files",
                    "f2",
                    "b.txt",
                    "Second doc content about databases.",
                ),
            ])
            .await
            .unwrap();
        assert_eq!(db.count().await.unwrap(), n as u64);
    }

    // ==================== count_by_source ====================

    #[tokio::test]
    async fn test_count_by_source() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "File system content about directories.",
            ),
            make_doc("memory", "m1", "b.txt", "Memory content about preferences."),
            make_doc("memory", "m2", "c.txt", "Memory content about decisions."),
        ])
        .await
        .unwrap();

        let counts = db.count_by_source().await.unwrap();
        assert!(counts.contains_key("local_files"));
        assert!(counts.contains_key("memory"));
        assert!(
            counts["memory"] >= 2,
            "memory source should have at least 2 memories"
        );
    }

    // ==================== list_indexed_files ====================

    #[tokio::test]
    async fn test_list_indexed_files_empty() {
        let (db, _dir) = test_db().await;
        let files = db.list_indexed_files().await.unwrap();
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn test_list_indexed_files() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "readme.md",
                "This is a readme with project documentation.",
            ),
            make_doc(
                "local_files",
                "f2",
                "main.rs",
                "fn main() { println!(\"hello world\"); }",
            ),
        ])
        .await
        .unwrap();

        let files = db.list_indexed_files().await.unwrap();
        assert_eq!(files.len(), 2);

        let source_ids: Vec<&str> = files.iter().map(|f| f.source_id.as_str()).collect();
        assert!(source_ids.contains(&"f1"));
        assert!(source_ids.contains(&"f2"));

        for file in &files {
            assert!(file.chunk_count >= 1);
            assert!(!file.title.is_empty());
        }
    }

    // ==================== list_filtered ====================

    #[tokio::test]
    async fn test_list_filtered_by_source() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "Local file content about testing.",
            ),
            make_doc(
                "memory",
                "m1",
                "b.txt",
                "Memory content about user preferences.",
            ),
        ])
        .await
        .unwrap();

        let results = db
            .list_filtered(Some("memory"), None, None, 100)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "memory");
    }

    #[tokio::test]
    async fn test_list_filtered_empty_filter() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_doc(
                "local_files",
                "f1",
                "a.txt",
                "Content A for empty filter test.",
            ),
            make_doc("memory", "m1", "b.txt", "Content B for empty filter test."),
        ])
        .await
        .unwrap();

        let results = db.list_filtered(None, None, None, 100).await.unwrap();
        assert_eq!(results.len(), 2, "empty filter should return all");
    }

    #[tokio::test]
    async fn test_list_filtered_respects_limit() {
        let (db, _dir) = test_db().await;
        let mut docs = Vec::new();
        for i in 0..5 {
            docs.push(make_doc(
                "local_files",
                &format!("f{}", i),
                &format!("doc{}.txt", i),
                &format!("Content for limit test number {} in filtering.", i),
            ));
        }
        db.upsert_documents(docs).await.unwrap();

        let results = db.list_filtered(None, None, None, 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    // ==================== Knowledge Graph: store_entity ====================

    #[tokio::test]
    async fn test_store_entity() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Rust", "language", Some("tech"), Some("claude"), Some(0.95))
            .await
            .unwrap();
        assert!(!id.is_empty(), "should return a UUID");
    }

    #[tokio::test]
    async fn test_store_entity_minimal() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        assert!(!id.is_empty());
    }

    // ==================== Knowledge Graph: add_observation ====================

    #[tokio::test]
    async fn test_add_observation() {
        let (db, _dir) = test_db().await;
        let entity_id = db
            .store_entity("Bob", "person", None, None, None)
            .await
            .unwrap();
        let obs_id = db
            .add_observation(&entity_id, "Bob likes coffee", Some("claude"), Some(0.8))
            .await
            .unwrap();
        assert!(!obs_id.is_empty());
    }

    #[tokio::test]
    async fn test_add_multiple_observations() {
        let (db, _dir) = test_db().await;
        let entity_id = db
            .store_entity("Project X", "project", Some("work"), None, None)
            .await
            .unwrap();

        let o1 = db
            .add_observation(&entity_id, "Uses React frontend", None, None)
            .await
            .unwrap();
        let o2 = db
            .add_observation(&entity_id, "Deploys to AWS Lambda", None, None)
            .await
            .unwrap();
        let o3 = db
            .add_observation(&entity_id, "Has 5 team members", None, None)
            .await
            .unwrap();

        // All should be unique IDs
        let ids: HashSet<_> = [o1, o2, o3].into_iter().collect();
        assert_eq!(ids.len(), 3, "all observation IDs should be unique");
    }

    // ==================== Knowledge Graph: create_relation ====================

    #[tokio::test]
    async fn test_create_relation() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("Project X", "project", None, None, None)
            .await
            .unwrap();

        let rel_id = db
            .create_relation(&e1, &e2, "works_on", Some("claude"), None, None, None)
            .await
            .unwrap();
        assert!(!rel_id.is_empty());
    }

    #[tokio::test]
    async fn test_create_multiple_relations() {
        let (db, _dir) = test_db().await;
        let alice = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let bob = db
            .store_entity("Bob", "person", None, None, None)
            .await
            .unwrap();
        let project = db
            .store_entity("Project Y", "project", None, None, None)
            .await
            .unwrap();

        let r1 = db
            .create_relation(&alice, &project, "leads", None, None, None, None)
            .await
            .unwrap();
        let r2 = db
            .create_relation(&bob, &project, "contributes_to", None, None, None, None)
            .await
            .unwrap();
        let r3 = db
            .create_relation(&alice, &bob, "manages", None, None, None, None)
            .await
            .unwrap();

        let ids: HashSet<_> = [r1, r2, r3].into_iter().collect();
        assert_eq!(ids.len(), 3, "all relation IDs should be unique");
    }

    // ==================== Knowledge Graph: full lifecycle ====================

    #[tokio::test]
    async fn test_knowledge_graph_full_lifecycle() {
        let (db, _dir) = test_db().await;

        // Create entities
        let rust = db
            .store_entity("Rust", "language", Some("tech"), Some("claude"), Some(0.99))
            .await
            .unwrap();
        let origin = db
            .store_entity(
                "Origin",
                "project",
                Some("tech"),
                Some("claude"),
                Some(0.95),
            )
            .await
            .unwrap();

        // Add observations
        let obs = db
            .add_observation(
                &rust,
                "Rust is a systems language",
                Some("claude"),
                Some(0.9),
            )
            .await
            .unwrap();
        assert!(!obs.is_empty());

        let obs2 = db
            .add_observation(
                &origin,
                "Origin uses Rust for backend",
                Some("claude"),
                None,
            )
            .await
            .unwrap();
        assert!(!obs2.is_empty());

        // Create relation
        let rel = db
            .create_relation(&origin, &rust, "uses", Some("claude"), None, None, None)
            .await
            .unwrap();
        assert!(!rel.is_empty());

        // Verify entities can be queried (via raw SQL to confirm data is in DB)
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM entities", ())
            .await
            .unwrap();
        if let Ok(Some(row)) = rows.next().await {
            let count: i64 = row.get(0).unwrap();
            assert_eq!(count, 2);
        }

        let mut rows = conn
            .query("SELECT COUNT(*) FROM observations", ())
            .await
            .unwrap();
        if let Ok(Some(row)) = rows.next().await {
            let count: i64 = row.get(0).unwrap();
            assert_eq!(count, 2);
        }

        let mut rows = conn
            .query("SELECT COUNT(*) FROM relations", ())
            .await
            .unwrap();
        if let Ok(Some(row)) = rows.next().await {
            let count: i64 = row.get(0).unwrap();
            assert_eq!(count, 1);
        }
    }

    // ==================== get_memories_by_source_id ====================

    #[tokio::test]
    async fn test_get_memories_by_source_id() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "notes.txt",
            "These are my personal notes about the project requirements and deadlines.",
        )])
        .await
        .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        assert_eq!(memories[0].source_id, "f1");
    }

    #[tokio::test]
    async fn test_get_memories_by_source_id_nonexistent() {
        let (db, _dir) = test_db().await;
        let memories = db
            .get_memories_by_source_id("local_files", "no_such_id")
            .await
            .unwrap();
        assert!(memories.is_empty());
    }

    #[tokio::test]
    async fn test_get_memories_ordered_by_chunk_index() {
        let (db, _dir) = test_db().await;
        // Use a longer content to potentially produce multiple memories
        let long_content = "Section one of the document. ".repeat(100);
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "long.txt",
            &long_content,
        )])
        .await
        .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        for (i, memory) in memories.iter().enumerate() {
            assert_eq!(
                memory.chunk_index, i as i32,
                "memories should be ordered by index"
            );
        }
    }

    // ==================== update_memory ====================

    #[tokio::test]
    async fn test_update_memory() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Original memory content about databases and queries.",
        )])
        .await
        .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty());

        let source_id = &memories[0].source_id;
        db.update_memory(
            source_id,
            "New updated memory content about machine learning.",
        )
        .await
        .unwrap();

        // Verify the content was updated
        let updated_memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert_eq!(
            updated_memories[0].content,
            "New updated memory content about machine learning."
        );
    }

    // ==================== update_timestamp_by_source_id ====================

    #[tokio::test]
    async fn test_update_timestamp() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Content for timestamp update test.",
        )])
        .await
        .unwrap();

        db.update_timestamp_by_source_id("f1", 99999).await.unwrap();

        let files = db.list_indexed_files().await.unwrap();
        let f1 = files.iter().find(|f| f.source_id == "f1").unwrap();
        assert_eq!(f1.last_modified, 99999);
    }

    // ==================== get_memory_details ====================

    #[tokio::test]
    async fn test_get_memory_details() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "details.txt",
            "Content to test get_memory_details functionality.",
        )])
        .await
        .unwrap();

        let detail = db.get_memory_details("local_files", "f1", 0).await.unwrap();
        assert!(detail.is_some());
        let detail = detail.unwrap();
        assert_eq!(detail.chunk_index, 0);
        assert_eq!(detail.source_id, "f1");
    }

    #[tokio::test]
    async fn test_get_memory_details_nonexistent() {
        let (db, _dir) = test_db().await;
        let detail = db
            .get_memory_details("local_files", "nope", 0)
            .await
            .unwrap();
        assert!(detail.is_none());
    }

    // ==================== update_document_content ====================

    #[tokio::test]
    async fn test_update_document_content() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "local_files",
            "f1",
            "a.txt",
            "Original document content about Rust.",
        )])
        .await
        .unwrap();

        db.update_document_content("f1", "Completely new replacement content about TypeScript.")
            .await
            .unwrap();

        let memories = db
            .get_memories_by_source_id("local_files", "f1")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        // The content should reflect the new text
        let all_content: String = memories
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            all_content.contains("TypeScript") || all_content.contains("replacement"),
            "updated content should reflect new text"
        );
    }

    #[tokio::test]
    async fn test_update_document_content_nonexistent() {
        let (db, _dir) = test_db().await;
        // Updating a nonexistent source_id should succeed silently (no rows matched)
        let result = db.update_document_content("no_such_id", "content").await;
        assert!(result.is_ok());
    }

    // ==================== vec_to_sql helper ====================

    #[test]
    fn test_vec_to_sql() {
        let v = vec![1.0_f32, 2.5, -0.3];
        let sql = MemoryDB::vec_to_sql(&v);
        assert!(sql.starts_with('['));
        assert!(sql.ends_with(']'));
        assert!(sql.contains("1.000000"));
        assert!(sql.contains("2.500000"));
        assert!(sql.contains("-0.300000"));
    }

    #[test]
    fn test_vec_to_sql_empty() {
        let v: Vec<f32> = vec![];
        let sql = MemoryDB::vec_to_sql(&v);
        assert_eq!(sql, "[]");
    }

    // ==================== Edge cases ====================

    #[tokio::test]
    async fn test_large_document_chunking() {
        let (db, _dir) = test_db().await;
        // A document large enough to be split into multiple memories
        let large_content = "This is paragraph about important topics. ".repeat(500);
        let doc = make_doc("local_files", "big", "large.txt", &large_content);
        let count = db.upsert_documents(vec![doc]).await.unwrap();
        assert!(
            count >= 2,
            "large document should be split into multiple memories, got {}",
            count
        );

        let memories = db
            .get_memories_by_source_id("local_files", "big")
            .await
            .unwrap();
        assert_eq!(memories.len(), count);
    }

    #[tokio::test]
    async fn test_concurrent_db_access() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);

        // Spawn multiple async tasks that write and read concurrently
        let mut handles = Vec::new();
        for i in 0..5 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                let doc = make_doc(
                    "local_files",
                    &format!("concurrent_{}", i),
                    &format!("c{}.txt", i),
                    &format!("Content for concurrent test number {} with unique data.", i),
                );
                db_clone.upsert_documents(vec![doc]).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let total = db.count().await.unwrap();
        assert!(
            total >= 5,
            "all concurrent upserts should succeed, got {}",
            total
        );
    }

    #[tokio::test]
    async fn test_special_characters_in_content() {
        let (db, _dir) = test_db().await;
        let doc = make_doc(
            "local_files",
            "special",
            "special.txt",
            "Content with 'quotes' and \"double quotes\" and even some SQL: SELECT * FROM table WHERE id = '1'; DROP TABLE; --",
        );
        let result = db.upsert_documents(vec![doc]).await;
        assert!(result.is_ok(), "special characters should not break insert");

        let memories = db
            .get_memories_by_source_id("local_files", "special")
            .await
            .unwrap();
        assert!(!memories.is_empty());
    }

    #[tokio::test]
    async fn test_unicode_content() {
        let (db, _dir) = test_db().await;
        let doc = make_doc(
            "local_files",
            "unicode",
            "unicode.txt",
            "Contenu en francais avec des accents. Japanische Zeichen: Tauri desktop application.",
        );
        db.upsert_documents(vec![doc]).await.unwrap();
        let memories = db
            .get_memories_by_source_id("local_files", "unicode")
            .await
            .unwrap();
        assert!(!memories.is_empty());
    }

    #[tokio::test]
    async fn test_list_entities() {
        let (db, _dir) = test_db().await;
        db.store_entity("Alice", "person", Some("work"), Some("claude"), Some(0.9))
            .await
            .unwrap();
        db.store_entity("Origin", "project", Some("work"), Some("claude"), Some(0.8))
            .await
            .unwrap();
        db.store_entity("Bob", "person", Some("personal"), None, None)
            .await
            .unwrap();

        // List all
        let all = db.list_entities(None, None).await.unwrap();
        assert_eq!(all.len(), 3);

        // Filter by type
        let people = db.list_entities(Some("person"), None).await.unwrap();
        assert_eq!(people.len(), 2);

        // Filter by domain
        let work = db.list_entities(None, Some("work")).await.unwrap();
        assert_eq!(work.len(), 2);

        // Filter by both
        let work_people = db
            .list_entities(Some("person"), Some("work"))
            .await
            .unwrap();
        assert_eq!(work_people.len(), 1);
        assert_eq!(work_people[0].name, "Alice");
    }

    #[tokio::test]
    async fn test_get_entity_detail() {
        let (db, _dir) = test_db().await;
        let alice_id = db
            .store_entity("Alice", "person", Some("work"), Some("claude"), Some(0.9))
            .await
            .unwrap();
        let origin_id = db
            .store_entity("Origin", "project", Some("work"), None, None)
            .await
            .unwrap();

        db.add_observation(&alice_id, "Speaks fluent Rust", Some("claude"), Some(0.8))
            .await
            .unwrap();
        db.add_observation(&alice_id, "Prefers TDD", Some("claude"), Some(0.95))
            .await
            .unwrap();
        db.create_relation(
            &alice_id,
            &origin_id,
            "works_on",
            Some("claude"),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let detail = db.get_entity_detail(&alice_id).await.unwrap();
        assert_eq!(detail.entity.name, "Alice");
        assert_eq!(detail.observations.len(), 2);
        assert_eq!(detail.relations.len(), 1);
        assert_eq!(detail.relations[0].relation_type, "works_on");
        assert_eq!(detail.relations[0].entity_name, "Origin");
        assert_eq!(detail.relations[0].direction, "outgoing");

        // Check from the other side
        let origin_detail = db.get_entity_detail(&origin_id).await.unwrap();
        assert_eq!(origin_detail.relations.len(), 1);
        assert_eq!(origin_detail.relations[0].direction, "incoming");
        assert_eq!(origin_detail.relations[0].entity_name, "Alice");
    }

    #[tokio::test]
    async fn test_update_observation() {
        let (db, _dir) = test_db().await;
        let eid = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let oid = db
            .add_observation(&eid, "Likes tea", Some("claude"), Some(0.7))
            .await
            .unwrap();

        db.update_observation(&oid, "Loves coffee").await.unwrap();

        let detail = db.get_entity_detail(&eid).await.unwrap();
        assert_eq!(detail.observations[0].content, "Loves coffee");
    }

    #[tokio::test]
    async fn test_delete_observation() {
        let (db, _dir) = test_db().await;
        let eid = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let oid1 = db
            .add_observation(&eid, "Fact one", None, None)
            .await
            .unwrap();
        let _oid2 = db
            .add_observation(&eid, "Fact two", None, None)
            .await
            .unwrap();

        db.delete_observation(&oid1).await.unwrap();

        let detail = db.get_entity_detail(&eid).await.unwrap();
        assert_eq!(detail.observations.len(), 1);
        assert_eq!(detail.observations[0].content, "Fact two");
    }

    #[tokio::test]
    async fn test_delete_entity_cascades() {
        let (db, _dir) = test_db().await;
        let alice_id = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let bob_id = db
            .store_entity("Bob", "person", None, None, None)
            .await
            .unwrap();
        db.add_observation(&alice_id, "Obs", None, None)
            .await
            .unwrap();
        db.create_relation(&alice_id, &bob_id, "knows", None, None, None, None)
            .await
            .unwrap();

        db.delete_entity(&alice_id).await.unwrap();

        // Entity gone
        let result = db.get_entity_detail(&alice_id).await;
        assert!(result.is_err());

        // Relation also gone (FK cascade)
        let bob_detail = db.get_entity_detail(&bob_id).await.unwrap();
        assert_eq!(bob_detail.relations.len(), 0);
    }

    #[tokio::test]
    async fn test_confirm_entity() {
        let (db, _dir) = test_db().await;
        let eid = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        assert!(!db.get_entity_detail(&eid).await.unwrap().entity.confirmed);

        db.confirm_entity(&eid, true).await.unwrap();
        assert!(db.get_entity_detail(&eid).await.unwrap().entity.confirmed);
    }

    #[tokio::test]
    async fn test_list_memories_rich() {
        let (db, _dir) = test_db().await;
        let doc1 = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_001".to_string(),
            title: "TDD preference".to_string(),
            content: "Prefers TDD workflow".to_string(),
            memory_type: Some("preference".to_string()),
            domain: Some("work".to_string()),
            source_agent: Some("claude".to_string()),
            confidence: Some(0.95),
            confirmed: Some(true),
            ..Default::default()
        };
        let doc2 = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_002".to_string(),
            title: "Morning person".to_string(),
            content: "Starts work before 8am".to_string(),
            memory_type: Some("fact".to_string()),
            domain: Some("identity".to_string()),
            source_agent: Some("claude".to_string()),
            confidence: Some(0.6),
            confirmed: Some(false),
            ..Default::default()
        };
        db.upsert_documents(vec![doc1, doc2]).await.unwrap();

        // List all
        let all = db.list_memories(None, None, None, None, 100).await.unwrap();
        assert_eq!(all.len(), 2);
        assert!(all[0].confidence.is_some());
        assert!(all[0].memory_type.is_some());

        // Filter by domain
        let work = db
            .list_memories(Some("work"), None, None, None, 100)
            .await
            .unwrap();
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].source_id, "mem_001");

        // Filter by confirmed
        let confirmed = db
            .list_memories(None, None, Some(true), None, 100)
            .await
            .unwrap();
        assert_eq!(confirmed.len(), 1);
    }

    #[tokio::test]
    async fn test_get_memory_stats() {
        let (db, _dir) = test_db().await;
        let doc1 = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_001".to_string(),
            title: "Fact 1".to_string(),
            content: "Content".to_string(),
            domain: Some("work".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        let doc2 = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_002".to_string(),
            title: "Fact 2".to_string(),
            content: "Content".to_string(),
            domain: Some("identity".to_string()),
            confirmed: Some(false),
            ..Default::default()
        };
        db.upsert_documents(vec![doc1, doc2]).await.unwrap();

        let stats = db.get_memory_stats().await.unwrap();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.confirmed, 1);
        assert_eq!(stats.domains.len(), 2);
    }

    // ==================== Profile CRUD ====================

    #[tokio::test]
    async fn test_profile_bootstrap() {
        let (db, _dir) = test_db().await;
        let profile = db.get_profile().await.unwrap();
        assert!(profile.is_some());
        let p = profile.unwrap();
        assert_eq!(p.name, "User");
        assert!(p.display_name.is_none());
    }

    #[tokio::test]
    async fn test_update_profile() {
        let (db, _dir) = test_db().await;
        db.bootstrap_profile().await.unwrap();
        let profile = db.get_profile().await.unwrap().unwrap();
        db.update_profile(&profile.id, Some("Lucian"), Some("Lu"), None, None, None)
            .await
            .unwrap();
        let updated = db.get_profile().await.unwrap().unwrap();
        assert_eq!(updated.name, "Lucian");
        assert_eq!(updated.display_name, Some("Lu".to_string()));
    }

    #[tokio::test]
    async fn test_bootstrap_profile_idempotent() {
        let (db, _dir) = test_db().await;
        db.bootstrap_profile().await.unwrap();
        db.bootstrap_profile().await.unwrap();
        let profile = db.get_profile().await.unwrap().unwrap();
        assert_eq!(profile.name, "User");
    }

    #[tokio::test]
    async fn test_profile_extended_fields() {
        let (db, _dir) = test_db().await;
        db.bootstrap_profile().await.unwrap();
        let profile = db.get_profile().await.unwrap().unwrap();

        // Update with new extended fields
        db.update_profile(
            &profile.id,
            Some("Lucian"),
            Some("Lu"),
            Some("lucian@example.com"),
            Some("Building cool stuff"),
            Some("/path/to/avatar.png"),
        )
        .await
        .unwrap();

        let updated = db.get_profile().await.unwrap().unwrap();
        assert_eq!(updated.name, "Lucian");
        assert_eq!(updated.display_name, Some("Lu".to_string()));
        assert_eq!(updated.email, Some("lucian@example.com".to_string()));
        assert_eq!(updated.bio, Some("Building cool stuff".to_string()));
        assert_eq!(updated.avatar_path, Some("/path/to/avatar.png".to_string()));
    }

    #[tokio::test]
    async fn test_profile_clear_optional_fields() {
        let (db, _dir) = test_db().await;
        db.bootstrap_profile().await.unwrap();
        let profile = db.get_profile().await.unwrap().unwrap();

        // First set the fields
        db.update_profile(
            &profile.id,
            Some("Lucian"),
            Some("Lu"),
            Some("lucian@example.com"),
            Some("A bio"),
            Some("/avatar.png"),
        )
        .await
        .unwrap();

        // Now clear them with empty strings (should become NULL)
        db.update_profile(&profile.id, None, Some(""), Some(""), Some(""), Some(""))
            .await
            .unwrap();

        let updated = db.get_profile().await.unwrap().unwrap();
        assert_eq!(updated.name, "Lucian", "name should not change (was None)");
        assert!(
            updated.display_name.is_none(),
            "empty string should become NULL"
        );
        assert!(updated.email.is_none(), "empty string should become NULL");
        assert!(updated.bio.is_none(), "empty string should become NULL");
        assert!(
            updated.avatar_path.is_none(),
            "empty string should become NULL"
        );
    }

    #[tokio::test]
    async fn test_migration_applies_to_existing_db() {
        let (db, _dir) = test_db().await;
        // After init (which test_db simulates), run_migrations should set user_version >= 2
        db.run_migrations(&crate::events::NoopEmitter)
            .await
            .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
        let version: i64 = if let Some(row) = rows.next().await.unwrap() {
            row.get::<i64>(0).unwrap_or(0)
        } else {
            0
        };
        assert!(
            version >= 3,
            "user_version should be >= 3 after migrations, got {}",
            version
        );
    }

    // ==================== Agent Connection CRUD ====================

    #[tokio::test]
    async fn test_register_agent() {
        let (db, _dir) = test_db().await;
        let agent = db.register_agent("claude-code").await.unwrap();
        assert_eq!(agent.name, "claude-code");
        assert_eq!(agent.agent_type, "api");
        // Default is `full` — registration is the trust gesture for a
        // single-user local server. See `register_agent`.
        assert_eq!(agent.trust_level, "full");
        assert!(agent.enabled);
        assert_eq!(agent.memory_count, 0);
    }

    #[tokio::test]
    async fn test_register_agent_idempotent() {
        let (db, _dir) = test_db().await;
        let a1 = db.register_agent("claude-code").await.unwrap();
        let a2 = db.register_agent("claude-code").await.unwrap();
        assert_eq!(a1.id, a2.id);
    }

    #[tokio::test]
    async fn test_list_agents() {
        let (db, _dir) = test_db().await;
        db.register_agent("claude-code").await.unwrap();
        db.register_agent("cursor").await.unwrap();
        let agents = db.list_agents().await.unwrap();
        assert_eq!(agents.len(), 2);
    }

    #[tokio::test]
    async fn test_update_agent() {
        let (db, _dir) = test_db().await;
        let agent = db.register_agent("claude-code").await.unwrap();
        db.update_agent(
            &agent.name,
            Some("cli"),
            Some("Claude Code CLI"),
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let updated = db.get_agent("claude-code").await.unwrap().unwrap();
        assert_eq!(updated.agent_type, "cli");
        assert_eq!(updated.description, Some("Claude Code CLI".to_string()));
    }

    #[tokio::test]
    async fn test_update_agent_trust_level() {
        let (db, _dir) = test_db().await;
        db.register_agent("trusted-bot").await.unwrap();
        db.update_agent("trusted-bot", None, None, None, Some("review"), None)
            .await
            .unwrap();
        let agent = db.get_agent("trusted-bot").await.unwrap().unwrap();
        assert_eq!(agent.trust_level, "review");
    }

    #[tokio::test]
    async fn test_delete_agent() {
        let (db, _dir) = test_db().await;
        db.register_agent("temp-agent").await.unwrap();
        db.delete_agent("temp-agent").await.unwrap();
        let agent = db.get_agent("temp-agent").await.unwrap();
        assert!(agent.is_none());
    }

    #[tokio::test]
    async fn test_touch_agent() {
        let (db, _dir) = test_db().await;
        let agent = db.register_agent("claude-code").await.unwrap();
        assert_eq!(agent.memory_count, 0);
        db.touch_agent("claude-code").await.unwrap();
        let updated = db.get_agent("claude-code").await.unwrap().unwrap();
        assert_eq!(updated.memory_count, 1);
        assert!(updated.last_seen_at.is_some());
    }

    #[tokio::test]
    async fn test_check_agent_enabled_allows_write() {
        let (db, _dir) = test_db().await;
        let result = db.check_agent_for_write("new-agent").await;
        assert!(result.is_ok());
        let trust = result.unwrap();
        // Default is `full` — registration implies trust.
        assert_eq!(trust, "full");
    }

    #[tokio::test]
    async fn test_check_agent_disabled_blocks_write() {
        let (db, _dir) = test_db().await;
        db.register_agent("bad-agent").await.unwrap();
        db.update_agent("bad-agent", None, None, Some(false), None, None)
            .await
            .unwrap();
        let result = db.check_agent_for_write("bad-agent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_check_agent_returns_trust_level() {
        let (db, _dir) = test_db().await;
        db.register_agent("trusted-bot").await.unwrap();
        db.update_agent("trusted-bot", None, None, None, Some("review"), None)
            .await
            .unwrap();
        let trust = db.check_agent_for_write("trusted-bot").await.unwrap();
        assert_eq!(trust, "review");
    }

    #[tokio::test]
    async fn test_check_agent_none_skips_check() {
        let (db, _dir) = test_db().await;
        let trust = db.check_agent_for_write_optional(None).await.unwrap();
        assert_eq!(trust, "full");
    }

    #[tokio::test]
    async fn test_agent_write_flow_integration() {
        let (db, _dir) = test_db().await;

        // Auto-register: new agent gets registered and returns trust level.
        // Default is `full` (per migration 32 / updated `register_agent`).
        let trust = db.check_agent_for_write("integration-bot").await.unwrap();
        assert_eq!(trust, "full");

        // Agent exists after auto-register
        let agent = db.get_agent("integration-bot").await.unwrap().unwrap();
        assert_eq!(agent.name, "integration-bot");
        assert_eq!(agent.memory_count, 1); // touch_agent was called
        assert!(agent.last_seen_at.is_some());

        // Upgrade trust
        db.update_agent("integration-bot", None, None, None, Some("full"), None)
            .await
            .unwrap();
        let trust = db.check_agent_for_write("integration-bot").await.unwrap();
        assert_eq!(trust, "full");

        // Disable blocks writes
        db.update_agent("integration-bot", None, None, Some(false), None, None)
            .await
            .unwrap();
        let result = db.check_agent_for_write("integration-bot").await;
        assert!(result.is_err());
        match result {
            Err(crate::error::OriginError::AgentDisabled(_)) => {} // expected
            other => panic!("Expected AgentDisabled, got: {:?}", other),
        }

        // None agent skips check (local writes)
        let trust = db.check_agent_for_write_optional(None).await.unwrap();
        assert_eq!(trust, "full");

        // Profile was bootstrapped during DB init
        let profile = db.get_profile().await.unwrap();
        assert!(profile.is_some());
    }

    // ==================== Phase 8: NULL vs empty-string regression tests ====================

    #[tokio::test]
    async fn test_store_entity_null_domain() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();

        // Verify domain is NULL, not empty string
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT domain, source_agent, confidence FROM entities WHERE id = ?1",
                libsql::params![id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let domain_val = row.get_value(0).unwrap();
        assert!(
            matches!(domain_val, libsql::Value::Null),
            "domain should be NULL, got {:?}",
            domain_val
        );
        let agent_val = row.get_value(1).unwrap();
        assert!(
            matches!(agent_val, libsql::Value::Null),
            "source_agent should be NULL, got {:?}",
            agent_val
        );
    }

    #[tokio::test]
    async fn test_store_entity_null_confidence() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Bob", "person", Some("work"), None, None)
            .await
            .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT confidence FROM entities WHERE id = ?1",
                libsql::params![id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let val = row.get_value(0).unwrap();
        assert!(
            matches!(val, libsql::Value::Null),
            "confidence should be NULL when None, got {:?}",
            val
        );
    }

    #[tokio::test]
    async fn test_add_observation_null_source_agent() {
        let (db, _dir) = test_db().await;
        let eid = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let oid = db
            .add_observation(&eid, "Likes coffee", None, None)
            .await
            .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_agent, confidence FROM observations WHERE id = ?1",
                libsql::params![oid],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let agent_val = row.get_value(0).unwrap();
        let conf_val = row.get_value(1).unwrap();
        assert!(
            matches!(agent_val, libsql::Value::Null),
            "source_agent should be NULL, got {:?}",
            agent_val
        );
        assert!(
            matches!(conf_val, libsql::Value::Null),
            "confidence should be NULL, got {:?}",
            conf_val
        );
    }

    // ==================== supersedes ====================

    #[tokio::test]
    async fn test_supersedes_soft_suppresses() {
        let (db, _dir) = test_db().await;

        // Store memory A
        let doc_a = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_a".to_string(),
            title: "Original".to_string(),
            content: "I prefer dark mode".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc_a]).await.unwrap();

        // Verify A is confirmed
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT confirmed FROM memories WHERE source_id = 'mem_a'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        assert_eq!(
            row.get::<i64>(0).unwrap(),
            1,
            "A should be confirmed before supersede"
        );
        drop(rows);
        drop(conn);

        // Store memory B superseding A
        let doc_b = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_b".to_string(),
            title: "Updated".to_string(),
            content: "I prefer light mode now".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            supersedes: Some("mem_a".to_string()),
            pending_revision: false,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_b]).await.unwrap();

        // Verify A is now soft-suppressed (confirmed = 0)
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT confirmed FROM memories WHERE source_id = 'mem_a'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        assert_eq!(
            row.get::<i64>(0).unwrap(),
            0,
            "A should be soft-suppressed after supersede"
        );
    }

    #[tokio::test]
    async fn test_search_excludes_superseded() {
        let (db, _dir) = test_db().await;

        // Store A and B (B supersedes A)
        let doc_a = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_search_a".to_string(),
            title: "V1".to_string(),
            content: "dark mode preference for coding".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc_a]).await.unwrap();

        let doc_b = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_search_b".to_string(),
            title: "V2".to_string(),
            content: "light mode preference for coding".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            supersedes: Some("mem_search_a".to_string()),
            pending_revision: false,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_b]).await.unwrap();

        // Search — A should be excluded
        let results = db
            .search_memory(
                "mode preference coding",
                10,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let source_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        assert!(
            !source_ids.contains(&"mem_search_a"),
            "superseded memory A should be excluded from search"
        );
        assert!(
            source_ids.contains(&"mem_search_b"),
            "superseding memory B should appear in search"
        );
    }

    #[tokio::test]
    async fn test_version_chain() {
        let (db, _dir) = test_db().await;

        // A → B → C chain
        let doc_a = RawDocument {
            source: "memory".to_string(),
            source_id: "chain_a".to_string(),
            title: "V1".to_string(),
            content: "First version".to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            last_modified: 1000,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_a]).await.unwrap();

        let doc_b = RawDocument {
            source: "memory".to_string(),
            source_id: "chain_b".to_string(),
            title: "V2".to_string(),
            content: "Second version".to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            supersedes: Some("chain_a".to_string()),
            pending_revision: false,
            last_modified: 2000,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_b]).await.unwrap();

        let doc_c = RawDocument {
            source: "memory".to_string(),
            source_id: "chain_c".to_string(),
            title: "V3".to_string(),
            content: "Third version".to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            supersedes: Some("chain_b".to_string()),
            pending_revision: false,
            last_modified: 3000,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_c]).await.unwrap();

        // Get chain from middle
        let chain = db.get_version_chain("chain_b").await.unwrap();
        assert_eq!(chain.len(), 3, "chain should have 3 items");
        assert_eq!(chain[0].source_id, "chain_a", "first should be root");
        assert_eq!(chain[1].source_id, "chain_b");
        assert_eq!(chain[2].source_id, "chain_c", "last should be newest");

        // Chain from root
        let chain = db.get_version_chain("chain_a").await.unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].source_id, "chain_a");

        // Chain from leaf
        let chain = db.get_version_chain("chain_c").await.unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[2].source_id, "chain_c");
    }

    #[tokio::test]
    async fn test_create_relation_null_source_agent() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("Alice", "person", None, None, None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("Bob", "person", None, None, None)
            .await
            .unwrap();
        let rid = db
            .create_relation(&e1, &e2, "knows", None, None, None, None)
            .await
            .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_agent FROM relations WHERE id = ?1",
                libsql::params![rid],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let val = row.get_value(0).unwrap();
        assert!(
            matches!(val, libsql::Value::Null),
            "source_agent should be NULL, got {:?}",
            val
        );
    }

    // ==================== Knowledge Graph: relation type normalization ====================

    #[tokio::test]
    async fn test_relation_type_normalized_at_insert() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Insert with alias type "working_at" -- should normalize to "works_on"
        db.create_relation(
            &e1,
            &e2,
            "working_at",
            Some("test"),
            Some(0.9),
            Some("she works there"),
            Some("mem_1"),
        )
        .await
        .unwrap();

        // Query the relation to verify normalization
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT relation_type, confidence, explanation, source_memory_id FROM relations WHERE from_entity = ?1",
                libsql::params![e1.clone()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let stored_type: String = row.get(0).unwrap();
        assert_eq!(
            stored_type, "works_on",
            "working_at should normalize to works_on"
        );
        let stored_conf: f64 = row.get(1).unwrap();
        assert!((stored_conf - 0.9).abs() < 0.01);
        let stored_explanation: String = row.get(2).unwrap();
        assert_eq!(stored_explanation, "she works there");
        let stored_source_id: String = row.get(3).unwrap();
        assert_eq!(stored_source_id, "mem_1");
    }

    // ==================== load_memories_by_type ====================

    #[tokio::test]
    async fn test_load_memories_by_type() {
        let (db, _dir) = test_db().await;

        // Store memories of different types
        let docs = vec![
            RawDocument {
                source: "memory".to_string(),
                source_id: "id_1".to_string(),
                title: "My name".to_string(),
                content: "I am a software engineer".to_string(),
                memory_type: Some("identity".to_string()),
                confirmed: Some(true),
                last_modified: 1000,
                ..Default::default()
            },
            RawDocument {
                source: "memory".to_string(),
                source_id: "pref_1".to_string(),
                title: "Code style".to_string(),
                content: "I prefer functional programming".to_string(),
                memory_type: Some("preference".to_string()),
                confirmed: Some(true),
                last_modified: 2000,
                ..Default::default()
            },
            RawDocument {
                source: "memory".to_string(),
                source_id: "id_2".to_string(),
                title: "Location".to_string(),
                content: "I live in San Francisco".to_string(),
                memory_type: Some("identity".to_string()),
                confirmed: Some(true),
                last_modified: 3000,
                ..Default::default()
            },
            RawDocument {
                source: "memory".to_string(),
                source_id: "id_unconfirmed".to_string(),
                title: "Unconfirmed".to_string(),
                content: "Maybe a student".to_string(),
                memory_type: Some("identity".to_string()),
                confirmed: Some(false), // unconfirmed
                last_modified: 4000,
                ..Default::default()
            },
        ];
        db.upsert_documents(docs).await.unwrap();

        // Load identity — should get 2 confirmed, newest first
        let identities = db
            .load_memories_by_type("identity", 10, None)
            .await
            .unwrap();
        assert_eq!(
            identities.len(),
            2,
            "should have 2 confirmed identity memories"
        );
        assert_eq!(identities[0].source_id, "id_2", "newest first");
        assert_eq!(identities[1].source_id, "id_1");

        // Load preference
        let prefs = db
            .load_memories_by_type("preference", 10, None)
            .await
            .unwrap();
        assert_eq!(prefs.len(), 1);

        // Load goal — none stored
        let goals = db.load_memories_by_type("goal", 10, None).await.unwrap();
        assert_eq!(goals.len(), 0);
    }

    #[tokio::test]
    async fn test_load_memories_by_type_excludes_superseded() {
        let (db, _dir) = test_db().await;

        let doc_a = RawDocument {
            source: "memory".to_string(),
            source_id: "id_old".to_string(),
            title: "V1".to_string(),
            content: "I am a junior developer".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc_a]).await.unwrap();

        let doc_b = RawDocument {
            source: "memory".to_string(),
            source_id: "id_new".to_string(),
            title: "V2".to_string(),
            content: "I am a senior developer".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(true),
            supersedes: Some("id_old".to_string()),
            pending_revision: false,
            ..Default::default()
        };
        db.upsert_documents(vec![doc_b]).await.unwrap();

        let identities = db
            .load_memories_by_type("identity", 10, None)
            .await
            .unwrap();
        let ids: Vec<&str> = identities.iter().map(|m| m.source_id.as_str()).collect();
        assert!(
            !ids.contains(&"id_old"),
            "superseded memory should be excluded"
        );
        assert!(ids.contains(&"id_new"), "current memory should be included");
    }

    // ==================== search_corrections_by_topic ====================

    #[tokio::test]
    async fn test_search_corrections_by_topic() {
        let (db, _dir) = test_db().await;

        // Corrections are now stored as memory_type = "fact" (migrated from "correction")
        let docs = vec![
            RawDocument {
                source: "memory".to_string(),
                source_id: "corr_1".to_string(),
                title: "Rust correction".to_string(),
                content:
                    "Never use unwrap in production Rust code, use expect or proper error handling"
                        .to_string(),
                memory_type: Some("fact".to_string()),
                confirmed: Some(true),
                ..Default::default()
            },
            RawDocument {
                source: "memory".to_string(),
                source_id: "corr_2".to_string(),
                title: "Python correction".to_string(),
                content:
                    "In Python, prefer list comprehensions over map and filter for readability"
                        .to_string(),
                memory_type: Some("fact".to_string()),
                confirmed: Some(true),
                ..Default::default()
            },
        ];
        db.upsert_documents(docs).await.unwrap();

        let results = db
            .search_corrections_by_topic("Rust error handling unwrap", 5)
            .await
            .unwrap();
        assert!(!results.is_empty(), "should find fact memory by topic");
        // Both corrections should appear (order depends on embedding similarity)
        let ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        assert!(
            ids.contains(&"corr_1"),
            "Rust correction should be in results"
        );
    }

    // ==================== Pin / Unpin ====================

    #[tokio::test]
    async fn test_pin_memory() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_001".to_string(),
            title: "Pinnable memory".to_string(),
            content: "This memory should be pinnable".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        db.pin_memory("pin_001").await.unwrap();

        let pinned = db.list_pinned_memories().await.unwrap();
        assert_eq!(pinned.len(), 1);
        assert_eq!(pinned[0].source_id, "pin_001");
    }

    #[tokio::test]
    async fn test_unpin_memory() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_002".to_string(),
            title: "Unpin test".to_string(),
            content: "This memory will be pinned then unpinned".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        db.pin_memory("pin_002").await.unwrap();
        let pinned = db.list_pinned_memories().await.unwrap();
        assert_eq!(pinned.len(), 1);

        db.unpin_memory("pin_002").await.unwrap();
        let pinned = db.list_pinned_memories().await.unwrap();
        assert!(pinned.is_empty(), "pinned list should be empty after unpin");
    }

    #[tokio::test]
    async fn test_pin_allows_unconfirmed() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_003".to_string(),
            title: "Unconfirmed memory".to_string(),
            content: "Unconfirmed memories should be pinnable".to_string(),
            confirmed: Some(false),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        db.pin_memory("pin_003").await.unwrap();
        let pinned = db.list_pinned_memories().await.unwrap();
        assert_eq!(pinned.len(), 1, "unconfirmed memory should be pinned");
    }

    #[tokio::test]
    async fn test_pin_max_limit() {
        let (db, _dir) = test_db().await;

        // Create and pin 12 confirmed memories (the maximum)
        for i in 0..12 {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: format!("pin_max_{:02}", i),
                title: format!("Pinned memory {}", i),
                content: format!("Content for pinned memory number {}", i),
                confirmed: Some(true),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
            db.pin_memory(&format!("pin_max_{:02}", i)).await.unwrap();
        }

        let pinned = db.list_pinned_memories().await.unwrap();
        assert_eq!(pinned.len(), 12);

        // 13th pin should fail
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_max_12".to_string(),
            title: "One too many".to_string(),
            content: "This 13th pin should be rejected".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let result = db.pin_memory("pin_max_12").await;
        assert!(result.is_err(), "13th pin should be rejected");
    }

    #[tokio::test]
    async fn test_unconfirm_auto_unpins() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_004".to_string(),
            title: "Auto-unpin test".to_string(),
            content: "This pinned memory will be unconfirmed".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        db.pin_memory("pin_004").await.unwrap();
        let pinned = db.list_pinned_memories().await.unwrap();
        assert_eq!(pinned.len(), 1);

        // Unconfirm should auto-unpin via the DB trigger
        db.update_column_by_source_id("memory", "pin_004", "confirmed", "0")
            .await
            .unwrap();

        let pinned = db.list_pinned_memories().await.unwrap();
        assert!(
            pinned.is_empty(),
            "unconfirming should auto-unpin via trigger"
        );
    }

    #[tokio::test]
    async fn test_list_memories_includes_pinned() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "pin_005".to_string(),
            title: "Pinned in list".to_string(),
            content: "This memory should show pinned=true in list_memories".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Before pinning, pinned should be false
        let all = db.list_memories(None, None, None, None, 100).await.unwrap();
        assert_eq!(all.len(), 1);
        assert!(!all[0].pinned, "should not be pinned initially");

        db.pin_memory("pin_005").await.unwrap();

        // After pinning, pinned should be true
        let all = db.list_memories(None, None, None, None, 100).await.unwrap();
        assert_eq!(all.len(), 1);
        assert!(all[0].pinned, "should be pinned after pin_memory");

        // Filter by pinned=true
        let pinned_only = db
            .list_memories(None, None, None, Some(true), 100)
            .await
            .unwrap();
        assert_eq!(pinned_only.len(), 1);
        assert_eq!(pinned_only[0].source_id, "pin_005");

        // Filter by pinned=false
        let unpinned_only = db
            .list_memories(None, None, None, Some(false), 100)
            .await
            .unwrap();
        assert!(unpinned_only.is_empty());
    }

    // ==================== Pending Revision ====================

    #[tokio::test]
    async fn test_pending_revision_excluded_from_search() {
        let (db, _dir) = test_db().await;

        // Insert a normal memory with distinctive content for embedding match
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "visible_mem".to_string(),
            title: "Visible".to_string(),
            content: "Kubernetes container orchestration deployment scaling pods services"
                .to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Insert a pending revision with similar content (should be hidden)
        let pending = RawDocument {
            source: "memory".to_string(),
            source_id: "pending_mem".to_string(),
            title: "Pending".to_string(),
            content: "Kubernetes container orchestration with Helm charts and namespaces"
                .to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(false),
            supersedes: Some("visible_mem".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![pending]).await.unwrap();

        // Search should find original but not the pending revision
        let results = db
            .search("kubernetes container orchestration", 10, None)
            .await
            .unwrap();
        let ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        assert!(
            ids.contains(&"visible_mem"),
            "original should be in search results"
        );
        assert!(
            !ids.contains(&"pending_mem"),
            "pending revision should be excluded from search"
        );
    }

    #[tokio::test]
    async fn test_pending_revision_excluded_from_list() {
        let (db, _dir) = test_db().await;

        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "listed_mem".to_string(),
            title: "Listed".to_string(),
            content: "My favorite IDE is Neovim".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let pending = RawDocument {
            source: "memory".to_string(),
            source_id: "pending_list_mem".to_string(),
            title: "Pending".to_string(),
            content: "My favorite IDE is VS Code".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(false),
            supersedes: Some("listed_mem".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![pending]).await.unwrap();

        let all = db.list_memories(None, None, None, None, 100).await.unwrap();
        let ids: Vec<&str> = all.iter().map(|m| m.source_id.as_str()).collect();
        assert!(ids.contains(&"listed_mem"), "original should be listed");
        assert!(
            !ids.contains(&"pending_list_mem"),
            "pending revision should be excluded from list"
        );
    }

    #[tokio::test]
    async fn test_pending_revision_does_not_suppress_target() {
        let (db, _dir) = test_db().await;

        // Original protected memory
        let original = RawDocument {
            source: "memory".to_string(),
            source_id: "target_mem".to_string(),
            title: "Original".to_string(),
            content: "My name is Lucian".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![original]).await.unwrap();

        // Pending revision that supersedes it — should NOT suppress the target
        let revision = RawDocument {
            source: "memory".to_string(),
            source_id: "revision_mem".to_string(),
            title: "Revision".to_string(),
            content: "My name is Lucas".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(false),
            supersedes: Some("target_mem".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![revision]).await.unwrap();

        // Target should still be visible (pending revision's supersedes doesn't count)
        let mems = db.list_memories(None, None, None, None, 100).await.unwrap();
        let ids: Vec<&str> = mems.iter().map(|m| m.source_id.as_str()).collect();
        assert!(
            ids.contains(&"target_mem"),
            "target must remain visible while revision is pending"
        );
        assert!(
            !ids.contains(&"revision_mem"),
            "pending revision must be hidden"
        );
    }

    #[tokio::test]
    async fn test_get_pending_revision_for() {
        let (db, _dir) = test_db().await;

        let original = RawDocument {
            source: "memory".to_string(),
            source_id: "pr_target".to_string(),
            title: "Original".to_string(),
            content: "I work at Acme Corp".to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![original]).await.unwrap();

        // No pending revision yet
        let none = db.get_pending_revision_for("pr_target").await.unwrap();
        assert!(none.is_none());

        // Add pending revision
        let revision = RawDocument {
            source: "memory".to_string(),
            source_id: "pr_revision".to_string(),
            title: "Revision".to_string(),
            content: "I work at Globex Corp".to_string(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test-agent".to_string()),
            confirmed: Some(false),
            supersedes: Some("pr_target".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![revision]).await.unwrap();

        let pr = db.get_pending_revision_for("pr_target").await.unwrap();
        assert!(pr.is_some());
        let pr = pr.unwrap();
        assert_eq!(pr.source_id, "pr_revision");
        assert_eq!(pr.content, "I work at Globex Corp");
        assert_eq!(pr.source_agent, Some("test-agent".to_string()));
    }

    #[tokio::test]
    async fn test_accept_pending_revision() {
        let (db, _dir) = test_db().await;

        let original = RawDocument {
            source: "memory".to_string(),
            source_id: "accept_target".to_string(),
            title: "Original".to_string(),
            content: "I am a junior developer".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![original]).await.unwrap();

        let revision = RawDocument {
            source: "memory".to_string(),
            source_id: "accept_revision".to_string(),
            title: "Revision".to_string(),
            content: "I am a senior developer".to_string(),
            memory_type: Some("identity".to_string()),
            confirmed: Some(false),
            supersedes: Some("accept_target".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![revision]).await.unwrap();

        // Accept the revision
        db.accept_pending_revision("accept_target").await.unwrap();

        // Revision should now be visible and confirmed
        let mems = db.list_memories(None, None, None, None, 100).await.unwrap();
        let ids: Vec<&str> = mems.iter().map(|m| m.source_id.as_str()).collect();
        assert!(
            ids.contains(&"accept_revision"),
            "accepted revision should now be visible"
        );
        assert!(
            !ids.contains(&"accept_target"),
            "original should be suppressed after accept"
        );

        // No more pending revision
        let pr = db.get_pending_revision_for("accept_target").await.unwrap();
        assert!(pr.is_none());
    }

    #[tokio::test]
    async fn test_dismiss_pending_revision() {
        let (db, _dir) = test_db().await;

        let original = RawDocument {
            source: "memory".to_string(),
            source_id: "dismiss_target".to_string(),
            title: "Original".to_string(),
            content: "I prefer tabs over spaces".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![original]).await.unwrap();

        let revision = RawDocument {
            source: "memory".to_string(),
            source_id: "dismiss_revision".to_string(),
            title: "Revision".to_string(),
            content: "I prefer spaces over tabs".to_string(),
            memory_type: Some("preference".to_string()),
            confirmed: Some(false),
            supersedes: Some("dismiss_target".to_string()),
            pending_revision: true,
            ..Default::default()
        };
        db.upsert_documents(vec![revision]).await.unwrap();

        // Dismiss the revision
        db.dismiss_pending_revision("dismiss_target").await.unwrap();

        // Original should still be visible
        let mems = db.list_memories(None, None, None, None, 100).await.unwrap();
        let ids: Vec<&str> = mems.iter().map(|m| m.source_id.as_str()).collect();
        assert!(
            ids.contains(&"dismiss_target"),
            "original should remain after dismiss"
        );
        assert!(
            !ids.contains(&"dismiss_revision"),
            "dismissed revision should be deleted"
        );

        // No more pending revision
        let pr = db.get_pending_revision_for("dismiss_target").await.unwrap();
        assert!(pr.is_none());
    }

    // ==================== search_entities_by_vector ====================

    #[tokio::test]
    async fn test_search_entities_by_vector() {
        let (db, _dir) = test_db().await;

        // Store entities with embeddings
        db.store_entity(
            "Rust programming language",
            "technology",
            Some("software"),
            None,
            None,
        )
        .await
        .unwrap();
        db.store_entity(
            "Python scripting language",
            "technology",
            Some("software"),
            None,
            None,
        )
        .await
        .unwrap();
        db.store_entity(
            "Machine learning algorithms",
            "concept",
            Some("ai"),
            None,
            None,
        )
        .await
        .unwrap();

        // Verify entities were stored
        let entities = db.list_entities(None, None).await.unwrap();
        assert_eq!(entities.len(), 3, "should have 3 entities stored");

        // Search should find semantically similar entities
        let results = db
            .search_entities_by_vector("Rust systems programming", 5)
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "should find at least one entity (got {} entities in db)",
            entities.len()
        );
        // The Rust entity should be in the results
        assert!(
            results.iter().any(|r| r.entity.name.contains("Rust")),
            "should find the Rust entity"
        );
    }

    #[tokio::test]
    async fn test_search_entities_by_vector_empty_db() {
        let (db, _dir) = test_db().await;

        // Should not panic on empty entity table
        let results = db.search_entities_by_vector("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // ==================== get_observations_for_entities ====================

    #[tokio::test]
    async fn test_get_observations_for_entities() {
        let (db, _dir) = test_db().await;

        // Create entities and add observations
        let eid1 = db
            .store_entity("Rust", "technology", None, None, None)
            .await
            .unwrap();
        let eid2 = db
            .store_entity("Python", "technology", None, None, None)
            .await
            .unwrap();

        db.add_observation(
            &eid1,
            "Rust is a systems programming language",
            Some("claude"),
            None,
        )
        .await
        .unwrap();
        db.add_observation(
            &eid1,
            "Rust has zero-cost abstractions",
            Some("claude"),
            None,
        )
        .await
        .unwrap();
        db.add_observation(
            &eid2,
            "Python is great for data science",
            Some("chatgpt"),
            None,
        )
        .await
        .unwrap();

        // Fetch observations for both entities
        let results = db
            .get_observations_for_entities(&[eid1.clone(), eid2.clone()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 3, "should return all 3 observations");
        // All results should have source = "knowledge_graph"
        assert!(
            results.iter().all(|r| r.source == "knowledge_graph"),
            "all results should have source=knowledge_graph"
        );
    }

    #[tokio::test]
    async fn test_get_observations_for_entities_empty_ids() {
        let (db, _dir) = test_db().await;
        let results = db.get_observations_for_entities(&[], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_get_observations_for_entities_respects_limit() {
        let (db, _dir) = test_db().await;
        let eid = db
            .store_entity("TestEntity", "thing", None, None, None)
            .await
            .unwrap();
        for i in 0..5 {
            db.add_observation(&eid, &format!("Observation number {}", i), None, None)
                .await
                .unwrap();
        }

        let results = db.get_observations_for_entities(&[eid], 2).await.unwrap();
        assert!(results.len() <= 2, "should respect limit");
    }

    // ==================== graph-augmented search_memory ====================

    // ==================== contextual enrichment ====================

    #[tokio::test]
    async fn test_memory_chunks_store_original_content() {
        let (db, _dir) = test_db().await;

        // Store a memory with rich metadata
        let doc = make_memory_doc(
            "ctx_test_1",
            "Always use parameterized SQL queries to prevent injection attacks",
            "fact",
            "software-engineering",
            "claude-code",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Retrieve the stored memory — content should be the ORIGINAL text,
        // not enriched (prefix is only in the embedding, not stored content)
        let memories = db
            .get_memories_by_source_id("memory", "ctx_test_1")
            .await
            .unwrap();
        assert!(!memories.is_empty(), "should have stored memories");

        let content = &memories[0].content;
        assert!(
            !content.starts_with("["),
            "stored content should NOT include metadata prefix (prefix is embedding-only)"
        );
        assert!(
            content.contains("parameterized SQL"),
            "stored content should contain the original text"
        );
    }

    #[tokio::test]
    async fn test_contextual_enrichment_stores_clean_content() {
        let (db, _dir) = test_db().await;

        // Store a memory with metadata that enriches the embedding
        let doc = make_memory_doc(
            "enrich_1",
            "Always double-check before deploying",
            "fact",
            "devops",
            "claude-code",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Content should be the original text (no prefix baked in)
        let memories = db
            .get_memories_by_source_id("memory", "enrich_1")
            .await
            .unwrap();
        assert!(!memories.is_empty());
        assert_eq!(memories[0].content, "Always double-check before deploying");
    }

    // ==================== search_memory_reranked ====================

    #[tokio::test]
    async fn test_search_memory_reranked_without_llm() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc(
                "m1",
                "Rust is a systems programming language",
                "fact",
                "software",
                "claude",
            ),
            make_memory_doc(
                "m2",
                "Python is great for data science work",
                "fact",
                "software",
                "claude",
            ),
        ])
        .await
        .unwrap();

        // Without LLM formatter, reranked search should behave like regular search
        let results = db
            .search_memory_reranked("Rust programming", 10, None, None, None, None)
            .await
            .unwrap();
        assert!(!results.is_empty(), "should return results without LLM");
    }

    #[tokio::test]
    async fn test_search_memory_reranked_respects_limit() {
        let (db, _dir) = test_db().await;
        for i in 0..5 {
            db.upsert_documents(vec![make_memory_doc(
                &format!("m{}", i),
                &format!("Programming topic number {} about software development", i),
                "fact",
                "software",
                "claude",
            )])
            .await
            .unwrap();
        }

        let results = db
            .search_memory_reranked("programming", 2, None, None, None, None)
            .await
            .unwrap();
        assert!(results.len() <= 2, "should respect limit");
    }

    #[tokio::test]
    async fn test_search_memory_reranked_includes_graph_observations() {
        let (db, _dir) = test_db().await;

        // Store a memory about dark mode (use words that match FTS query)
        db.upsert_documents(vec![make_memory_doc(
            "m1",
            "User prefers dark mode themes in all code editors and applications",
            "preference",
            "personal",
            "claude-code",
        )])
        .await
        .unwrap();

        // Store a knowledge graph entity with related observation
        let eid = db
            .store_entity(
                "User Interface Preferences",
                "concept",
                Some("personal"),
                None,
                None,
            )
            .await
            .unwrap();
        db.add_observation(
            &eid,
            "Dark themes reduce eye strain during late-night coding sessions",
            Some("claude-code"),
            None,
        )
        .await
        .unwrap();

        // search_memory_reranked inherits graph augmentation from search_memory, then adds LLM reranking
        let results = db
            .search_memory_reranked("dark mode", 10, None, None, None, None)
            .await
            .unwrap();
        assert!(!results.is_empty(), "should find results");

        // KG observations boost scores via RRF but are stripped from output
        assert!(
            results.iter().all(|r| r.source != "knowledge_graph"),
            "knowledge_graph observations should be filtered from reranked output"
        );
    }

    #[tokio::test]
    async fn test_get_memory_detail() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "mem_detail_001".to_string(),
            title: "Detail test".to_string(),
            content: "Full content for detail view".to_string(),
            memory_type: Some("fact".to_string()),
            domain: Some("work".to_string()),
            source_agent: Some("claude".to_string()),
            confidence: Some(0.85),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Should find existing memory
        let result = db.get_memory_detail("mem_detail_001").await.unwrap();
        assert!(result.is_some(), "should find memory by source_id");
        let item = result.unwrap();
        assert_eq!(item.source_id, "mem_detail_001");
        assert_eq!(item.content, "Full content for detail view");
        assert_eq!(item.memory_type.as_deref(), Some("fact"));
        assert_eq!(item.domain.as_deref(), Some("work"));
        assert!(item.confirmed);

        // Should return None for non-existent
        let missing = db.get_memory_detail("nonexistent").await.unwrap();
        assert!(
            missing.is_none(),
            "should return None for missing source_id"
        );
    }

    // ==================== Migration 4: Refinement Pipeline ====================

    #[tokio::test]
    async fn test_migration_4_refinement_columns() {
        let (db, _dir) = test_db().await;

        // Verify access_count column exists with default 0
        let conn = db.conn.lock().await;
        let _rows = conn
            .query("SELECT access_count, last_accessed, refinement_status, effective_confidence FROM memories LIMIT 1", ())
            .await
            .expect("refinement columns should exist after migration 4");
        // No rows is fine — just verifying columns exist
        drop(_rows);

        // Verify refinement_queue table exists with all columns
        let _rows = conn
            .query("SELECT id, action, source_ids, payload, confidence, status, created_at, resolved_at FROM refinement_queue LIMIT 1", ())
            .await
            .expect("refinement_queue table should exist after migration 4");
        drop(_rows);

        // Verify the partial index on status exists by inserting and querying
        conn.execute(
            "INSERT INTO refinement_queue (id, action, source_ids, confidence) VALUES ('test1', 'dedup_merge', '[\"a\",\"b\"]', 0.95)",
            (),
        ).await.expect("insert into refinement_queue should work");

        let mut rows = conn
            .query(
                "SELECT id FROM refinement_queue WHERE status = 'pending'",
                (),
            )
            .await
            .expect("query by status should use index");
        let row = rows.next().await.unwrap().unwrap();
        let id: String = row.get(0).unwrap();
        assert_eq!(id, "test1");
        drop(rows);
        drop(conn);
    }

    // ==================== Access Tracking ====================

    #[tokio::test]
    async fn test_flush_access_counts() {
        let (db, _dir) = test_db().await;

        // Insert a test document
        let doc = make_doc(
            "memory",
            "access_test_1",
            "Access Test",
            "Test memory for access tracking",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Verify initial access_count is 0
        let conn = db.conn.lock().await;
        let mut rows = conn.query(
            "SELECT access_count, last_accessed FROM memories WHERE source_id = 'access_test_1'",
            (),
        ).await.unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        let last_accessed: Option<String> = row.get(1).unwrap();
        assert_eq!(count, 0);
        assert!(last_accessed.is_none());
        drop(rows);
        drop(conn);

        // Flush access counts
        db.flush_access_counts(&["access_test_1".to_string()])
            .await
            .unwrap();

        // Verify access_count incremented
        let conn = db.conn.lock().await;
        let mut rows = conn.query(
            "SELECT access_count, last_accessed FROM memories WHERE source_id = 'access_test_1'",
            (),
        ).await.unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        let last_accessed: Option<String> = row.get(1).unwrap();
        assert_eq!(count, 1);
        assert!(
            last_accessed.is_some(),
            "last_accessed should be set after flush"
        );
        drop(rows);
        drop(conn);

        // Flush again — should increment to 2
        db.flush_access_counts(&["access_test_1".to_string()])
            .await
            .unwrap();
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT access_count FROM memories WHERE source_id = 'access_test_1'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(count, 2);
        drop(rows);
        drop(conn);
    }

    #[tokio::test]
    async fn test_flush_access_counts_empty() {
        let (db, _dir) = test_db().await;
        // Should not error on empty input
        db.flush_access_counts(&[]).await.unwrap();
    }

    // ==================== Migration 5: Session Tables ====================

    #[tokio::test]
    async fn test_migration_5_session_tables() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;

        // Verify activities table
        conn.query(
            "SELECT id, started_at, ended_at FROM activities LIMIT 1",
            (),
        )
        .await
        .expect("activities table should exist");

        // Verify capture_refs table
        conn.query("SELECT source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source FROM capture_refs LIMIT 1", ())
            .await.expect("capture_refs table should exist");
        drop(conn);
    }

    // ==================== Session CRUD Methods ====================

    #[tokio::test]
    async fn test_upsert_and_get_activity() {
        let (db, _dir) = test_db().await;
        db.upsert_activity("act1", 1000, 2000).await.unwrap();
        let activities = db.get_completed_activities(600).await.unwrap();
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].id, "act1");
        assert_eq!(activities[0].started_at, 1000);

        // Upsert should update, not duplicate
        db.upsert_activity("act1", 1000, 3000).await.unwrap();
        let activities = db.get_completed_activities(600).await.unwrap();
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].ended_at, 3000);
    }

    #[tokio::test]
    async fn test_insert_and_query_capture_ref() {
        let (db, _dir) = test_db().await;
        db.upsert_activity("act1", 1000, 2000).await.unwrap();
        db.insert_capture_ref("src1", "act1", None, "VS Code", "main.rs", 1500, "screen")
            .await
            .unwrap();
        let refs = db.get_unpackaged_captures("act1").await.unwrap();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].source_id, "src1");
        assert_eq!(refs[0].app_name, "VS Code");
    }

    #[tokio::test]
    async fn test_mark_captures_packaged() {
        let (db, _dir) = test_db().await;
        db.upsert_activity("act1", 1000, 2000).await.unwrap();
        db.insert_capture_ref("src1", "act1", None, "Chrome", "tab1", 1100, "screen")
            .await
            .unwrap();
        db.insert_capture_ref("src2", "act1", None, "Chrome", "tab2", 1200, "screen")
            .await
            .unwrap();
        db.mark_captures_packaged(&["src1", "src2"], "snap1")
            .await
            .unwrap();
        let refs = db.get_unpackaged_captures("act1").await.unwrap();
        assert!(refs.is_empty());
    }

    #[tokio::test]
    async fn test_has_unpackaged_captures() {
        let (db, _dir) = test_db().await;
        db.upsert_activity("act1", 1000, 2000).await.unwrap();
        assert!(!db.has_unpackaged_captures("act1").await.unwrap());
        db.insert_capture_ref("src1", "act1", None, "VS Code", "main.rs", 1100, "screen")
            .await
            .unwrap();
        assert!(db.has_unpackaged_captures("act1").await.unwrap());
    }

    #[tokio::test]
    async fn test_get_captures_for_snapshot() {
        let (db, _dir) = test_db().await;
        db.upsert_activity("act1", 1000, 2000).await.unwrap();
        db.insert_capture_ref(
            "src1",
            "act1",
            Some("snap1"),
            "Chrome",
            "tab1",
            1100,
            "screen",
        )
        .await
        .unwrap();
        db.insert_capture_ref(
            "src2",
            "act1",
            Some("snap1"),
            "Chrome",
            "tab2",
            1200,
            "screen",
        )
        .await
        .unwrap();
        db.insert_capture_ref(
            "src3",
            "act1",
            Some("snap2"),
            "VS Code",
            "main.rs",
            1300,
            "screen",
        )
        .await
        .unwrap();
        let refs = db.get_captures_for_snapshot("snap1").await.unwrap();
        assert_eq!(refs.len(), 2);
    }

    #[tokio::test]
    async fn test_update_snapshot_summary() {
        let (_db, _dir) = test_db().await;
        // update_snapshot_summary updates the memories table via update_document_summary
        // This is just a passthrough — tested via update_document_summary tests
    }

    // ==================== Distillation Clustering ====================

    #[tokio::test]
    async fn test_find_distillation_clusters() {
        let (db, _dir) = test_db().await;

        // Insert 4 memories: 2 about topic A (same entity), 2 about topic B (same entity)
        for (i, (entity, content)) in [
            (
                "entity_arch",
                "libSQL stores vectors and knowledge graph in one database",
            ),
            (
                "entity_arch",
                "Using libSQL for all storage simplifies deployment",
            ),
            (
                "entity_mcp",
                "MCP surface was redesigned from 12 tools to 4",
            ),
            (
                "entity_mcp",
                "The MCP tools are remember recall context forget",
            ),
        ]
        .iter()
        .enumerate()
        {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: format!("cluster_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                entity_id: Some(entity.to_string()),
                domain: Some("origin".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let clusters = db
            .find_distillation_clusters(0.5, 2, 20, 3500)
            .await
            .unwrap();
        // Should find at least 1 cluster (entity groups with 2+ members)
        assert!(!clusters.is_empty(), "should find at least one cluster");
        // Each cluster should have at least 2 members
        for cluster in &clusters {
            assert!(
                cluster.source_ids.len() >= 2,
                "cluster too small: {:?}",
                cluster.source_ids
            );
        }
    }

    // ==================== Recap Integration ====================

    #[tokio::test]
    async fn test_recap_searchable_via_search_memory() {
        let (db, _dir) = test_db().await;

        // Store a recap memory (is_recap flag, memory_type = fact)
        let doc = RawDocument {
            source_id: "recap_test_1".to_string(),
            content: "Spent 2 hours debugging Tauri IPC calls and reading Playwright docs"
                .to_string(),
            source: "memory".to_string(),
            title: "Afternoon coding session".to_string(),
            memory_type: Some("fact".to_string()),
            confidence: Some(0.5),
            last_modified: chrono::Utc::now().timestamp(),
            is_recap: true,
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Recap should be findable via search_memory
        let results = db
            .search_memory(
                "Tauri IPC debugging",
                10,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "recap should be searchable via search_memory"
        );
        assert_eq!(results[0].memory_type, Some("fact".to_string()));
    }

    #[tokio::test]
    async fn test_recap_included_in_count_by_source() {
        let (db, _dir) = test_db().await;

        // Store a recap
        let doc = RawDocument {
            source_id: "recap_count_1".to_string(),
            content: "Working on Origin memory system".to_string(),
            source: "memory".to_string(),
            title: "Morning session".to_string(),
            memory_type: Some("fact".to_string()),
            confidence: Some(0.5),
            last_modified: chrono::Utc::now().timestamp(),
            is_recap: true,
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let counts = db.count_by_source().await.unwrap();
        assert!(
            counts.get("memory").copied().unwrap_or(0) > 0,
            "recap should be counted under memory source"
        );
    }

    // ==================== Refinement Queue ====================

    #[tokio::test]
    async fn test_refinement_queue_crud() {
        let (db, _dir) = test_db().await;

        db.insert_refinement_proposal(
            "ref_1",
            "dedup_merge",
            &["mem1".to_string(), "mem2".to_string()],
            Some(r#"{"merged": "combined"}"#),
            0.95,
        )
        .await
        .unwrap();

        let pending = db.get_pending_refinements().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].action, "dedup_merge");
        assert_eq!(pending[0].source_ids, vec!["mem1", "mem2"]);

        db.resolve_refinement("ref_1", "auto_applied")
            .await
            .unwrap();
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(pending.is_empty());
    }

    #[tokio::test]
    async fn pending_review_memory_ids_returns_only_awaiting_review_contradiction_rows() {
        let (db, _dir) = test_db().await;
        // Seed two memories
        db.upsert_documents(vec![make_doc(
            "memory",
            "mem_flagged",
            "flagged memory",
            "content one",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_doc(
            "memory",
            "mem_clean",
            "clean memory",
            "content two",
        )])
        .await
        .unwrap();

        // Enqueue a detect_contradiction for mem_flagged, then set to awaiting_review
        db.insert_refinement_proposal(
            "ref_flagged",
            "detect_contradiction",
            &["mem_flagged".to_string()],
            None,
            0.9,
        )
        .await
        .unwrap();
        db.resolve_refinement("ref_flagged", "awaiting_review")
            .await
            .unwrap();

        // Enqueue a detect_contradiction for mem_clean, then set to resolved (must NOT match)
        db.insert_refinement_proposal(
            "ref_clean",
            "detect_contradiction",
            &["mem_clean".to_string()],
            None,
            0.9,
        )
        .await
        .unwrap();
        db.resolve_refinement("ref_clean", "resolved")
            .await
            .unwrap();

        let candidates = vec!["mem_flagged".to_string(), "mem_clean".to_string()];
        let flagged = db.pending_review_memory_ids(&candidates).await.unwrap();
        assert!(flagged.contains("mem_flagged"));
        assert!(!flagged.contains("mem_clean"));
    }

    #[tokio::test]
    async fn test_consolidation_candidates() {
        let (db, _dir) = test_db().await;

        // Insert 4 memories in same domain with low effective_confidence
        for i in 0..4 {
            let mut doc = make_memory_doc(
                &format!("low_{}", i),
                &format!("Rust coding fact variant {}", i),
                "fact",
                "engineering",
                "claude",
            );
            doc.confidence = Some(0.2);
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // Set low effective_confidence
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET effective_confidence = 0.1 WHERE source = 'memory'",
            (),
        )
        .await
        .unwrap();
        drop(conn);

        let candidates = db.get_consolidation_candidates(0.3, 3).await.unwrap();
        assert!(
            !candidates.is_empty(),
            "should find consolidation candidates"
        );
        assert!(candidates
            .iter()
            .any(|c| c.domain == Some("engineering".to_string())));
    }

    // ==================== Decay Engine ====================

    #[tokio::test]
    async fn test_decay_update_confidence() {
        let (db, _dir) = test_db().await;

        // Insert memories with different tiers
        let mut doc1 = make_memory_doc(
            "decay_ephemeral",
            "Some recap content about coding",
            "goal",
            "engineering",
            "system",
        );
        doc1.confidence = Some(0.5);
        let mut doc2 = make_memory_doc(
            "decay_protected",
            "My name is Lucian",
            "identity",
            "personal",
            "claude",
        );
        doc2.confidence = Some(0.9);
        doc2.confirmed = Some(true);
        db.upsert_documents(vec![doc1, doc2]).await.unwrap();

        let count = db.decay_update_confidence().await.unwrap();
        assert!(count >= 2, "should update at least 2 memories");

        // Protected+confirmed should retain full confidence (decay_rate=0.0)
        let conn = db.conn.lock().await;
        let mut rows = conn.query(
            "SELECT effective_confidence FROM memories WHERE source_id = 'decay_protected' LIMIT 1",
            (),
        ).await.unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let eff: f64 = row.get(0).unwrap();
        // Confirmed → decay_rate=0 → recency_boost=1.0, access_boost=1.0 → eff = confidence
        assert!(
            (eff - 0.9).abs() < 0.01,
            "confirmed memory should not decay, got {}",
            eff
        );
        drop(rows);
        drop(conn);
    }

    // ==================== Merge / Queue Processing ====================

    #[tokio::test]
    async fn test_get_memory_contents() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc("mc1", "Rust is fast", "fact", "eng", "claude"),
            make_memory_doc("mc2", "Rust is safe", "fact", "eng", "claude"),
        ])
        .await
        .unwrap();
        let contents = db
            .get_memory_contents(&["mc1".to_string(), "mc2".to_string()])
            .await
            .unwrap();
        assert_eq!(contents.len(), 2);
        assert!(contents.iter().any(|c| c.contains("fast")));
        assert!(contents.iter().any(|c| c.contains("safe")));
    }

    #[tokio::test]
    async fn test_apply_merge() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc(
                "merge1",
                "Rust is a systems language",
                "fact",
                "eng",
                "claude",
            ),
            make_memory_doc(
                "merge2",
                "Rust focuses on memory safety",
                "fact",
                "eng",
                "claude",
            ),
        ])
        .await
        .unwrap();

        db.apply_merge(
            &["merge1".to_string(), "merge2".to_string()],
            "Rust is a systems language focused on memory safety",
        )
        .await
        .unwrap();

        // Original memories should be superseded by the merged one
        let results = db
            .search_memory(
                "Rust systems safety",
                10,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            results
                .iter()
                .any(|r| r.content.contains("focused on memory safety")),
            "merged memory should be searchable"
        );
    }

    #[tokio::test]
    async fn test_migration_6_access_log_and_word_count() {
        let (db, _dir) = test_db().await;
        // access_log table should exist
        let result = db
            .conn
            .lock()
            .await
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='access_log'",
                (),
            )
            .await
            .unwrap();
        let mut rows = result;
        let row = rows.next().await.unwrap();
        assert!(row.is_some(), "access_log table should exist");
        // word_count column should exist on memories
        let result = db
            .conn
            .lock()
            .await
            .query("SELECT word_count FROM memories LIMIT 1", ())
            .await
            .unwrap();
        drop(result);
    }

    #[tokio::test]
    async fn test_word_count_set_on_ingest() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            source: "memory".into(),
            source_id: "test-wc".into(),
            title: "Test".into(),
            content: "hello world this is a test".into(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".into()),
            domain: None,
            source_agent: None,
            confidence: None,
            confirmed: None,
            supersedes: None,
            summary: None,
            metadata: std::collections::HashMap::new(),
            pending_revision: false,
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT word_count FROM memories WHERE source_id = 'test-wc' AND chunk_index = 0",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let wc: i64 = row.get(0).unwrap();
        assert_eq!(
            wc, 6,
            "word_count should be 6 for 'hello world this is a test'"
        );
    }

    #[tokio::test]
    async fn test_get_highest_tier() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory_doc("tier1", "Some fact", "fact", "eng", "claude"),
            make_memory_doc("tier2", "My preference", "preference", "personal", "claude"),
        ])
        .await
        .unwrap();

        let tier = db
            .get_highest_tier(&["tier1".to_string(), "tier2".to_string()])
            .await
            .unwrap();
        assert_eq!(
            tier,
            crate::sources::StabilityTier::Protected,
            "preference is Protected which is highest"
        );
    }

    // ==================== get_home_stats ====================

    #[tokio::test]
    async fn test_get_home_stats_empty() {
        let (db, _dir) = test_db().await;
        let stats = db.get_home_stats().await.unwrap();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.times_served_today, 0);
        assert_eq!(stats.times_served_week, 0);
        assert_eq!(stats.times_served_all, 0);
        assert_eq!(stats.words_saved_today, 0);
        assert_eq!(stats.words_saved_week, 0);
        assert_eq!(stats.words_saved_all, 0);
        assert_eq!(stats.corrections_active, 0);
        assert!(stats.top_memories.is_empty());
    }

    #[tokio::test]
    async fn test_get_home_stats_with_data() {
        let (db, _dir) = test_db().await;
        // Insert a memory with known word count
        let doc = make_memory_doc(
            "hs-1",
            "always use test driven development",
            "preference",
            "work",
            "claude",
        );
        db.upsert_documents(vec![doc]).await.unwrap();
        // Log 3 accesses (writes to access_log for today/week queries)
        db.log_accesses(&["hs-1".into(), "hs-1".into(), "hs-1".into()])
            .await
            .unwrap();

        let stats = db.get_home_stats().await.unwrap();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.times_served_today, 3);
        assert_eq!(stats.times_served_week, 3);
        // word_count for "always use test driven development" = 5 words, 3 accesses = 15
        assert_eq!(stats.words_saved_today, 15);
        assert_eq!(stats.words_saved_week, 15);
        // All-time stats also use access_log (consistent with today/week)
        assert_eq!(stats.times_served_all, 3);
        assert_eq!(stats.words_saved_all, 15);
        assert_eq!(stats.top_memories.len(), 1);
        assert_eq!(stats.top_memories[0].source_id, "hs-1");
        assert_eq!(stats.top_memories[0].times_retrieved, 3);
    }

    #[tokio::test]
    async fn test_get_home_stats_corrections() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "corr-1",
            "actually use tabs not spaces",
            "fact",
            "work",
            "claude",
        );
        db.upsert_documents(vec![doc]).await.unwrap();
        // Key insights query requires confirmed = 1
        db.confirm_memory("corr-1").await.unwrap();

        let stats = db.get_home_stats().await.unwrap();
        assert_eq!(stats.corrections_active, 1);
    }

    #[tokio::test]
    async fn test_get_home_stats_all_time_via_flush() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("at-1", "some important fact here", "fact", "work", "claude");
        db.upsert_documents(vec![doc]).await.unwrap();
        // All-time stats now use access_log (consistent with today/week)
        db.log_accesses(&["at-1".into(), "at-1".into()])
            .await
            .unwrap();

        let stats = db.get_home_stats().await.unwrap();
        // Each log_accesses entry counted from access_log
        assert_eq!(stats.times_served_all, 2);
        // "some important fact here" = 4 words, 2 accesses = 8
        assert_eq!(stats.words_saved_all, 8);
    }

    // ==================== log_accesses ====================

    #[tokio::test]
    async fn test_log_accesses() {
        let (db, _dir) = test_db().await;
        db.log_accesses(&["mem-1".into(), "mem-2".into(), "mem-1".into()])
            .await
            .unwrap();

        let row = db
            .conn
            .lock()
            .await
            .query("SELECT COUNT(*) FROM access_log", ())
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(count, 3, "should have 3 access_log rows");

        let row = db
            .conn
            .lock()
            .await
            .query(
                "SELECT COUNT(*) FROM access_log WHERE source_id = 'mem-1'",
                (),
            )
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let count: i64 = row.get(0).unwrap();
        assert_eq!(count, 2, "mem-1 should have 2 access_log rows");
    }

    // ==================== resolve_entity_by_name ====================

    #[tokio::test]
    async fn test_resolve_entity_by_name_exact() {
        let (db, _dir) = test_db().await;
        let entity_id = db.create_entity("Alice", "person", None).await.unwrap();
        let resolved = db.resolve_entity_by_name("Alice").await.unwrap();
        assert_eq!(resolved, Some(entity_id));
    }

    #[tokio::test]
    async fn test_resolve_entity_by_name_case_insensitive() {
        let (db, _dir) = test_db().await;
        let entity_id = db
            .create_entity("PostgreSQL", "technology", None)
            .await
            .unwrap();
        let resolved = db.resolve_entity_by_name("postgresql").await.unwrap();
        assert_eq!(resolved, Some(entity_id));
    }

    #[tokio::test]
    async fn test_resolve_entity_by_name_no_match() {
        let (db, _dir) = test_db().await;
        let resolved = db.resolve_entity_by_name("NonExistent").await.unwrap();
        assert_eq!(resolved, None);
    }

    #[tokio::test]
    async fn test_resolve_entity_by_name_substring() {
        let (db, _dir) = test_db().await;
        let entity_id = db
            .create_entity("Alice Johnson", "person", None)
            .await
            .unwrap();
        // Should match via LIKE substring when exact fails
        let resolved = db.resolve_entity_by_name("Johnson").await.unwrap();
        assert_eq!(resolved, Some(entity_id));
    }

    // ==================== update_memory_entity_id ====================

    #[tokio::test]
    async fn test_update_memory_entity_id() {
        let (db, _dir) = test_db().await;
        let entity_id = db.create_entity("Alice", "person", None).await.unwrap();
        let doc = make_memory_doc(
            "mem_link",
            "Alice prefers dark mode",
            "preference",
            "personal",
            "claude",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Initially no entity_id
        let row = db
            .conn
            .lock()
            .await
            .query(
                "SELECT entity_id FROM memories WHERE source_id = 'mem_link' LIMIT 1",
                (),
            )
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let initial: Option<String> = row.get(0).unwrap();
        assert!(initial.is_none());

        // Link it
        db.update_memory_entity_id("mem_link", &entity_id)
            .await
            .unwrap();

        let row = db
            .conn
            .lock()
            .await
            .query(
                "SELECT entity_id FROM memories WHERE source_id = 'mem_link' LIMIT 1",
                (),
            )
            .await
            .unwrap()
            .next()
            .await
            .unwrap()
            .unwrap();
        let linked: Option<String> = row.get(0).unwrap();
        assert_eq!(linked, Some(entity_id));
    }

    /// Contract test for `apply_enrichment` — the combined writeback path
    /// used by the async classify + extract flow in `handle_store_memory`.
    ///
    /// Invariants locked down:
    ///   * `memory_type` and `supersede_mode` overwrite unconditionally
    ///     (agent-refined classification wins over the initial placeholder).
    ///   * `domain` / `quality` use `COALESCE(?, col)` — `None` passes
    ///     preserve the caller-supplied original, `Some` replaces.
    ///   * `structured_fields` / `retrieval_cue` only touch `chunk_index = 0`
    ///     (those fields live on the lead chunk only).
    ///   * `needs_reembed` flips to `1` iff structured fields were provided
    ///     (so the re-embed pass picks it up).
    #[tokio::test]
    async fn test_apply_enrichment_writes_classification_and_extraction() {
        let (db, _dir) = test_db().await;
        let mut doc = make_memory_doc(
            "mem_apply",
            "Alice prefers dark mode in editors",
            "fact",
            "preferences",
            "claude",
        );
        // Caller-supplied domain should survive a `None` in apply_enrichment.
        doc.domain = Some("preferences".to_string());
        db.upsert_documents(vec![doc]).await.unwrap();

        // Apply enrichment with a refined memory_type, no domain/quality,
        // fresh supersede_mode, and a structured-fields payload.
        db.apply_enrichment(
            "mem_apply",
            "preference",
            None,
            Some("high"),
            "hide",
            Some(r#"{"preference":"dark mode","applies_when":"editors"}"#),
            Some("What editor theme does Alice use?"),
        )
        .await
        .unwrap();

        // Assert the row-level metadata flipped / preserved per spec.
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT memory_type, domain, quality, supersede_mode,
                        structured_fields, retrieval_cue, needs_reembed
                 FROM memories WHERE source_id = 'mem_apply' AND chunk_index = 0",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let memory_type: String = row.get(0).unwrap();
        let domain: Option<String> = row.get(1).unwrap();
        let quality: Option<String> = row.get(2).unwrap();
        let supersede_mode: String = row.get(3).unwrap();
        let sf: Option<String> = row.get(4).unwrap();
        let cue: Option<String> = row.get(5).unwrap();
        let needs_reembed: i64 = row.get(6).unwrap();

        assert_eq!(memory_type, "preference");
        assert_eq!(
            domain.as_deref(),
            Some("preferences"),
            "None domain input must preserve existing column value via COALESCE"
        );
        assert_eq!(quality.as_deref(), Some("high"));
        assert_eq!(supersede_mode, "hide");
        assert!(sf.is_some());
        assert!(sf.unwrap().contains("dark mode"));
        assert_eq!(cue.as_deref(), Some("What editor theme does Alice use?"));
        assert_eq!(
            needs_reembed, 1,
            "structured_fields update must set needs_reembed = 1 so re-embed picks it up"
        );
    }

    // ==================== Space CRUD ====================

    #[tokio::test]
    async fn test_list_spaces() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&crate::events::NoopEmitter)
            .await
            .unwrap();

        // No spaces initially
        let spaces = db.list_spaces().await.unwrap();
        assert!(spaces.is_empty());

        // Create a space
        let s = db
            .create_space("work", Some("Work stuff"), false)
            .await
            .unwrap();
        assert_eq!(s.name, "work");
        assert_eq!(s.description, Some("Work stuff".to_string()));
        assert!(!s.suggested);

        let spaces = db.list_spaces().await.unwrap();
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].name, "work");
        assert_eq!(spaces[0].memory_count, 0);
    }

    #[tokio::test]
    async fn test_space_crud() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&crate::events::NoopEmitter)
            .await
            .unwrap();

        // Create
        db.create_space("work", Some("Work things"), true)
            .await
            .unwrap();

        // Get
        let s = db.get_space("work").await.unwrap().unwrap();
        assert_eq!(s.name, "work");
        assert!(s.suggested);

        // Confirm
        db.confirm_space("work").await.unwrap();
        let s = db.get_space("work").await.unwrap().unwrap();
        assert!(!s.suggested);

        // Update (rename)
        db.update_space("work", "career", Some("Career stuff"))
            .await
            .unwrap();
        assert!(db.get_space("work").await.unwrap().is_none());
        let s = db.get_space("career").await.unwrap().unwrap();
        assert_eq!(s.description, Some("Career stuff".to_string()));

        // Delete
        db.delete_space("career", "keep").await.unwrap();
        assert!(db.get_space("career").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_auto_create_space_if_needed() {
        let (db, _dir) = test_db().await;
        db.run_migrations(&crate::events::NoopEmitter)
            .await
            .unwrap();

        // First call creates a suggested space
        db.auto_create_space_if_needed("health").await.unwrap();
        let s = db.get_space("health").await.unwrap().unwrap();
        assert!(s.suggested);
        assert_eq!(s.name, "health");

        // Second call is a no-op
        db.auto_create_space_if_needed("health").await.unwrap();
        let spaces = db.list_spaces().await.unwrap();
        assert_eq!(spaces.len(), 1);

        // Doesn't affect manually created spaces
        db.create_space("work", None, false).await.unwrap();
        db.auto_create_space_if_needed("work").await.unwrap();
        let s = db.get_space("work").await.unwrap().unwrap();
        assert!(!s.suggested); // Stays confirmed
    }

    // ==================== migration 11: structured_fields + retrieval_cue ====================

    #[tokio::test]
    async fn test_migration_11_structured_fields() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, word_count, structured_fields, retrieval_cue)
             VALUES ('test_sf', 'test', 'test', 'test_sf', 'test', 0, 0, 'text', 1, '{\"claim\": \"test\"}', 'What do I know about test?')",
            (),
        ).await.unwrap();

        let mut rows = conn
            .query(
                "SELECT structured_fields, retrieval_cue FROM memories WHERE id = 'test_sf'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let sf: Option<String> = row.get(0).unwrap();
        let rc: Option<String> = row.get(1).unwrap();
        assert_eq!(sf.unwrap(), "{\"claim\": \"test\"}");
        assert_eq!(rc.unwrap(), "What do I know about test?");
    }

    #[tokio::test]
    async fn test_store_and_retrieve_structured_fields() {
        let (db, _dir) = test_db().await;
        let mut doc = RawDocument {
            source: "memory".into(),
            source_id: "sf_test_1".into(),
            title: "Test structured".into(),
            content: "I am a Rust developer".into(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("identity".into()),
            ..Default::default()
        };
        doc.structured_fields = Some(
            serde_json::json!({"claim": "I am a Rust developer", "since": "2020"}).to_string(),
        );
        doc.retrieval_cue = Some("Who is the user in terms of Rust development?".into());

        db.upsert_documents(vec![doc]).await.unwrap();

        let memories = db
            .list_memories(None, Some("identity"), None, None, 10)
            .await
            .unwrap();
        let found = memories
            .iter()
            .find(|c| c.source_id == "sf_test_1")
            .unwrap();
        assert!(found.structured_fields.as_ref().unwrap().contains("claim"));
        assert!(found.retrieval_cue.as_ref().unwrap().contains("Rust"));
    }

    // ==================== augment_with_graph ====================

    #[tokio::test]
    async fn test_augment_with_graph_merges_observations() {
        let (db, _dir) = test_db().await;
        // store_entity handles embedding internally
        let eid = db
            .store_entity("Rust", "technology", None, None, None)
            .await
            .unwrap();
        db.add_observation(&eid, "Rust is a systems language", Some("test"), None)
            .await
            .unwrap();

        // Start with an empty result set
        let results: Vec<SearchResult> = vec![];
        let augmented = db
            .augment_with_graph("Rust programming", results, 10)
            .await
            .unwrap();
        assert!(!augmented.is_empty(), "should include graph observations");
        assert!(augmented.iter().any(|r| r.source == "knowledge_graph"));
    }

    // ==================== search_memory graph augmentation ====================

    #[tokio::test]
    async fn test_search_memory_includes_graph_observations() {
        let (db, _dir) = test_db().await;

        // Store a memory document so the DB isn't entirely empty
        let doc = RawDocument {
            source: "memory".into(),
            source_id: "graph_test_mem_1".into(),
            title: "Rust programming language".into(),
            content: "Rust is a systems programming language focused on safety".into(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".into()),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Create an entity with embedding (store_entity handles embedding internally)
        let eid = db
            .store_entity("Rust", "technology", None, Some("test-agent"), None)
            .await
            .unwrap();

        // Add an observation to that entity
        db.add_observation(
            &eid,
            "Rust is memory-safe without a GC",
            Some("test-agent"),
            None,
        )
        .await
        .unwrap();

        // Search — KG observations boost scores via RRF but are stripped from output
        let results = db
            .search_memory("Rust programming", 10, None, None, None, None, None, None)
            .await
            .unwrap();

        assert!(
            results.iter().all(|r| r.source != "knowledge_graph"),
            "knowledge_graph observations should be filtered from search output (used for score boosting only)"
        );
    }

    #[tokio::test]
    async fn test_search_memory_works_without_entities() {
        let (db, _dir) = test_db().await;

        // Store a memory but create NO entities
        let doc = RawDocument {
            source: "memory".into(),
            source_id: "no_entity_test_1".into(),
            title: "Python scripting".into(),
            content: "Python is a great scripting language".into(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".into()),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let results = db
            .search_memory("Python scripting", 10, None, None, None, None, None, None)
            .await
            .unwrap();

        assert!(
            !results.is_empty(),
            "should return results even without entities"
        );
        assert!(
            results.iter().all(|r| r.source != "knowledge_graph"),
            "without entities, no knowledge_graph results should appear"
        );
    }

    #[tokio::test]
    async fn test_observations_have_unique_source_ids() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        // Create entity
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
             VALUES ('e1', 'Rust', 'technology', 0, 0)",
            (),
        )
        .await
        .unwrap();
        // Create 3 observations
        for i in 1..=3 {
            conn.execute(
                &format!(
                    "INSERT INTO observations (id, entity_id, content, source_agent, created_at)
             VALUES ('obs{}', 'e1', 'observation {}', 'test', {})",
                    i, i, i
                ),
                (),
            )
            .await
            .unwrap();
        }
        drop(conn);

        let results = db
            .get_observations_for_entities(&["e1".to_string()], 10)
            .await
            .unwrap();
        assert_eq!(results.len(), 3);
        let source_ids: std::collections::HashSet<&str> =
            results.iter().map(|r| r.source_id.as_str()).collect();
        assert_eq!(
            source_ids.len(),
            3,
            "each observation must have a unique source_id"
        );
        assert!(results.iter().all(|r| r.source_id.starts_with("obs_")));
    }

    #[tokio::test]
    async fn test_get_reembed_candidates() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        // Insert a memory with source_text and needs_reembed = 1
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, word_count, source_text, needs_reembed, structured_fields)
             VALUES ('rc1', 'preference: dark mode', 'memory', 'mem_rc1', 'test', 0, 0,
             'text', 1, 'I prefer dark mode', 1, '{\"preference\":\"dark mode\"}')",
            (),
        ).await.unwrap();
        // Insert a normal enriched memory (should NOT be returned)
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, word_count, enrichment_status)
             VALUES ('rc2', 'normal content', 'memory', 'mem_rc2', 'test', 0, 0, 'text', 1, 'enriched')",
            (),
        ).await.unwrap();
        drop(conn);

        let candidates = db.get_reembed_candidates(10).await.unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, "rc1");
    }

    #[tokio::test]
    async fn test_reembed_memory() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        // Insert a memory that needs re-embedding (needs_reembed = 1)
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, word_count, needs_reembed, structured_fields)
             VALUES ('re1', 'preference: dark mode', 'memory', 'mem_re1', 'test', 0, 0,
             'text', 1, 1, '{\"preference\":\"dark mode\"}')",
            (),
        ).await.unwrap();
        drop(conn);

        db.reembed_memory("re1", "preference: dark mode")
            .await
            .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT needs_reembed FROM memories WHERE id = 're1'", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let needs_reembed: i64 = row.get(0).unwrap();
        assert_eq!(
            needs_reembed, 0,
            "needs_reembed should be cleared after re-embedding"
        );
    }

    #[tokio::test]
    async fn test_needs_reembed_replaces_reembed_pending() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_reembed_test",
            "dark mode preference",
            "preference",
            "tools",
            "test",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Memory starts with needs_reembed = 0
        let pending = db.get_pending_reembeds(10).await.unwrap();
        assert!(!pending.iter().any(|r| r.source_id == "mem_reembed_test"));

        // Mark needs_reembed = 1 (simulating a structured_fields update)
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE memories SET needs_reembed = 1 WHERE source_id = 'mem_reembed_test'",
                (),
            )
            .await
            .unwrap();
        }

        // Should now appear in pending reembeds
        let pending = db.get_pending_reembeds(10).await.unwrap();
        assert!(pending.iter().any(|r| r.source_id == "mem_reembed_test"));
    }

    #[tokio::test]
    async fn test_source_text_column_exists() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, word_count, source_text)
             VALUES ('st1', 'structured content', 'memory', 'mem_st1', 'test', 0, 0, 'text', 1, 'original prose content')",
            (),
        ).await.unwrap();
        let mut rows = conn
            .query("SELECT source_text FROM memories WHERE id = 'st1'", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let st: Option<String> = row.get(0).unwrap();
        assert_eq!(st.unwrap(), "original prose content");
    }

    #[tokio::test]
    async fn test_find_same_type_memories() {
        let (db, _dir) = test_db().await;
        // Store unconfirmed preference in "tools" domain (should still be found)
        let doc1 = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "mem_p1".to_string(),
            title: "dark mode pref".to_string(),
            content: "I prefer dark mode".to_string(),
            memory_type: Some("preference".to_string()),
            domain: Some("tools".to_string()),
            confirmed: Some(false),
            last_modified: chrono::Utc::now().timestamp(),
            enrichment_status: "enriched".to_string(),
            supersede_mode: "hide".to_string(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc1]).await.unwrap();

        // Store a fact in same domain (different type)
        let doc2 = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "mem_f1".to_string(),
            title: "Rust fact".to_string(),
            content: "Rust is fast".to_string(),
            memory_type: Some("fact".to_string()),
            domain: Some("tools".to_string()),
            confirmed: Some(true),
            last_modified: chrono::Utc::now().timestamp(),
            enrichment_status: "enriched".to_string(),
            supersede_mode: "hide".to_string(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc2]).await.unwrap();

        // Search for preferences in "tools" — should find mem_p1 only
        let results = db
            .find_same_type_memories("mem_new", "preference", Some("tools"), 3)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "mem_p1");
    }

    #[tokio::test]
    async fn test_home_stats_distillation() {
        let (db, _dir) = test_db().await;

        // Insert 5 memories
        for i in 0..5 {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: format!("distill_{}", i),
                title: format!("Memory {}", i),
                content: format!("Content {}", i),
                confirmed: Some(true),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // distill_0 is the distilled version that supersedes distill_1 and distill_2
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET supersedes = 'distill_1' WHERE source_id = 'distill_0' AND source = 'memory'",
            (),
        ).await.unwrap();
        // Also create a chain: distill_3 supersedes distill_4
        conn.execute(
            "UPDATE memories SET supersedes = 'distill_4' WHERE source_id = 'distill_3' AND source = 'memory'",
            (),
        ).await.unwrap();
        drop(conn);

        let stats = db.get_home_stats().await.unwrap();
        assert_eq!(stats.total_ingested, 5);
        // distill_1 and distill_4 are superseded by other memories → 5 - 2 = 3 active
        assert_eq!(stats.active_insights, 3);
    }

    // ==================== search_memory supersede-aware dedup ====================

    #[tokio::test]
    async fn test_search_prefers_distilled() {
        let (db, _dir) = test_db().await;

        // Insert original memory
        let original = RawDocument {
            source: "memory".to_string(),
            source_id: "search_orig".to_string(),
            title: "Use Redis".to_string(),
            content: "Use Redis for caching in the API layer".to_string(),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![original]).await.unwrap();

        // Insert distilled version that supersedes it
        let distilled = RawDocument {
            source: "memory".to_string(),
            source_id: "search_distilled".to_string(),
            title: "Use Redis for caching".to_string(),
            content: "Use Redis for caching in the API layer. Evaluated Memcached but Redis has richer data structures.".to_string(),
            confirmed: Some(true),
            supersedes: Some("search_orig".to_string()),
            ..Default::default()
        };
        db.upsert_documents(vec![distilled]).await.unwrap();

        // Archive the original (supersede_mode = 'archive' — still visible without dedup)
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET supersede_mode = 'archive' WHERE source_id = 'search_orig' AND source = 'memory'",
            (),
        ).await.unwrap();
        drop(conn);

        let results = db
            .search_memory("Redis caching", 10, None, None, None, None, None, None)
            .await
            .unwrap();
        let ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Distilled version should be present, original should not
        assert!(
            ids.contains(&"search_distilled"),
            "distilled should be in results"
        );
        assert!(
            !ids.contains(&"search_orig"),
            "archived original should not be in results when distilled is present"
        );
    }

    #[tokio::test]
    async fn test_log_rejection_and_query() {
        let (db, _dir) = test_db().await;
        db.log_rejection(
            "rej_001",
            "heartbeat ok",
            Some("claude-code"),
            "noise_pattern",
            Some("heartbeat"),
            None,
            None,
        )
        .await
        .unwrap();

        let rejections = db.get_rejections(10, None).await.unwrap();
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].content, "heartbeat ok");
        assert_eq!(rejections[0].rejection_reason, "noise_pattern");
        assert_eq!(rejections[0].source_agent.as_deref(), Some("claude-code"));
    }

    #[tokio::test]
    async fn test_rejection_filter_by_reason() {
        let (db, _dir) = test_db().await;
        db.log_rejection(
            "rej_001",
            "heartbeat",
            Some("agent-a"),
            "noise_pattern",
            None,
            None,
            None,
        )
        .await
        .unwrap();
        db.log_rejection(
            "rej_002",
            "sk-abc123",
            Some("agent-b"),
            "credential_leak",
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let noise_only = db.get_rejections(10, Some("noise_pattern")).await.unwrap();
        assert_eq!(noise_only.len(), 1);
        assert_eq!(noise_only[0].id, "rej_001");
    }

    #[tokio::test]
    async fn test_prune_old_rejections() {
        let (db, _dir) = test_db().await;
        // Insert with old timestamp (31 days ago)
        let old_ts = chrono::Utc::now().timestamp() - (31 * 86400);
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO rejected_memories (id, content, rejection_reason, created_at) VALUES (?1, ?2, ?3, ?4)",
            libsql::params!["old_rej", "old content", "noise_pattern", old_ts],
        ).await.unwrap();
        drop(conn);

        db.log_rejection(
            "new_rej",
            "new content",
            None,
            "too_short",
            None,
            None,
            None,
        )
        .await
        .unwrap();
        db.prune_rejections(30).await.unwrap();

        let all = db.get_rejections(100, None).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "new_rej");
    }

    // ==================== check_novelty ====================

    #[tokio::test]
    async fn test_check_novelty_empty_db() {
        let (db, _dir) = test_db().await;
        let result = db.check_novelty("User prefers dark mode").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_check_novelty_finds_similar() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            content: "User prefers dark mode for all IDEs".to_string(),
            source_id: "mem_existing".to_string(),
            source: "memory".to_string(),
            title: "Dark mode pref".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let result = db
            .check_novelty("User prefers dark mode for all code editors")
            .await
            .unwrap();
        assert!(result.is_some(), "Should find similar memory");
        let (source_id, similarity) = result.unwrap();
        assert_eq!(source_id, "mem_existing");
        assert!(
            similarity > 0.70,
            "Expected high similarity, got {similarity}"
        );
    }

    #[tokio::test]
    async fn test_check_novelty_different_content() {
        let (db, _dir) = test_db().await;
        let doc = RawDocument {
            content: "User prefers dark mode for all IDEs".to_string(),
            source_id: "mem_existing".to_string(),
            source: "memory".to_string(),
            title: "Dark mode pref".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        let result = db
            .check_novelty("The database uses libSQL with vector indexing for semantic search")
            .await
            .unwrap();
        match result {
            None => {} // No match — fine
            Some((_, sim)) => assert!(
                sim < 0.85,
                "Unrelated content should have low similarity, got {sim}"
            ),
        }
    }

    #[tokio::test]
    async fn test_concepts_table_exists() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='concepts'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap();
        assert!(row.is_some(), "concepts table should exist");
    }

    #[tokio::test]
    async fn test_insert_and_get_concept() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let id = "concept_test1";
        db.insert_concept(
            id,
            "Test Title",
            Some("A summary"),
            "## Key Facts\n- fact 1",
            Some("entity_1"),
            Some("engineering"),
            &["mem_1", "mem_2"],
            &now,
        )
        .await
        .unwrap();

        let concept = db.get_concept(id).await.unwrap().unwrap();
        assert_eq!(concept.title, "Test Title");
        assert_eq!(concept.version, 1);
        assert_eq!(concept.source_memory_ids, vec!["mem_1", "mem_2"]);
        assert_eq!(concept.status, "active");
    }

    #[tokio::test]
    async fn test_update_concept_content() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "concept_u1",
            "Title",
            None,
            "v1 content",
            None,
            None,
            &["m1"],
            &now,
        )
        .await
        .unwrap();

        db.update_concept_content("concept_u1", "v2 content", &["m1", "m2"], "concept_growth")
            .await
            .unwrap();
        let c = db.get_concept("concept_u1").await.unwrap().unwrap();
        assert_eq!(c.content, "v2 content");
        assert_eq!(c.version, 2);
        assert_eq!(c.source_memory_ids, vec!["m1", "m2"]);
    }

    #[tokio::test]
    async fn test_list_active_concepts() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept("c1", "Title A", None, "content", None, None, &["m1"], &now)
            .await
            .unwrap();
        db.insert_concept("c2", "Title B", None, "content", None, None, &["m2"], &now)
            .await
            .unwrap();
        db.archive_concept("c2").await.unwrap();

        let active = db.list_concepts("active", 100, 0).await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "c1");
    }

    #[tokio::test]
    async fn test_delete_concept() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_del",
            "To Delete",
            None,
            "content",
            None,
            None,
            &["m1"],
            &now,
        )
        .await
        .unwrap();

        // Verify it exists
        assert!(db.get_concept("c_del").await.unwrap().is_some());

        // Delete it
        db.delete_concept("c_del").await.unwrap();

        // Verify it's gone
        assert!(db.get_concept("c_del").await.unwrap().is_none());

        // Deleting non-existent ID should not error
        db.delete_concept("nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn test_get_concept_by_entity() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_ent",
            "Entity Concept",
            None,
            "content",
            Some("ent_123"),
            None,
            &["m1"],
            &now,
        )
        .await
        .unwrap();

        let found = db.get_concept_by_entity("ent_123").await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "c_ent");

        let missing = db.get_concept_by_entity("ent_999").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_find_matching_concept_by_entity() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_match",
            "libSQL Architecture",
            None,
            "## Key Facts\n- uses vectors",
            Some("entity_libsql"),
            Some("arch"),
            &["m1"],
            &now,
        )
        .await
        .unwrap();

        let found = db
            .find_matching_concept(Some("entity_libsql"), &[0.0; EMBEDDING_DIM], 0.75)
            .await
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "c_match");

        // No match by entity_id, and zero-vec won't be similar enough
        let missing = db
            .find_matching_concept(Some("no_such_entity"), &[0.0; EMBEDDING_DIM], 0.75)
            .await
            .unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_search_concepts_fts() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_search",
            "libSQL Vector Storage",
            Some("Stores vectors in F32_BLOB"),
            "## Key Facts\n- DiskANN indexing\n- 384-dim embeddings",
            None,
            Some("architecture"),
            &["m1", "m2"],
            &now,
        )
        .await
        .unwrap();

        let results = db.search_concepts("DiskANN vector", 10).await.unwrap();
        assert!(!results.is_empty(), "should find concept via FTS");
        assert_eq!(results[0].id, "c_search");
    }

    #[tokio::test]
    async fn test_search_concepts_vector() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_vec",
            "Database Architecture",
            Some("How data is stored and indexed in the system"),
            "## Storage\n- Uses libSQL (Turso fork)\n- DiskANN for vector indexing\n- FTS5 for keyword search",
            None,
            Some("architecture"),
            &["m1"],
            &now,
        )
        .await
        .unwrap();

        // Query with different words than the concept (no keyword overlap)
        let results = db
            .search_concepts("what databases does the project use", 10)
            .await
            .unwrap();
        assert!(
            !results.is_empty(),
            "should find concept via vector similarity even without keyword match"
        );
        assert_eq!(results[0].id, "c_vec");
    }

    #[tokio::test]
    async fn test_migration_24_renames_chunks_to_memories() {
        // This test verifies the migration works by checking that
        // after MemoryDB::new() on a fresh DB, the memories table exists
        // and chunks does not.
        let dir = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(dir.path(), Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();

        let conn = db.conn.lock().await;

        // memories table should exist
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'",
                (),
            )
            .await
            .unwrap();
        assert!(
            rows.next().await.unwrap().is_some(),
            "memories table should exist"
        );
        drop(rows);

        // chunks table should NOT exist
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'",
                (),
            )
            .await
            .unwrap();
        assert!(
            rows.next().await.unwrap().is_none(),
            "chunks table should not exist"
        );
        drop(rows);

        // user_version should be >= 24
        let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
        let version: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert!(
            version >= 24,
            "user_version should be >= 24, got {}",
            version
        );
    }

    #[tokio::test]
    async fn test_sub_cluster_by_tokens() {
        let memories: Vec<ClusterMemRow> = (0..6)
            .map(|i| {
                let mut emb = vec![0.0f32; 768];
                if i < 3 {
                    emb[0] = 1.0 + (i as f32 * 0.05);
                    emb[1] = 0.1 * i as f32;
                } else {
                    emb[0] = 0.1 * (i - 3) as f32;
                    emb[1] = 1.0 + ((i - 3) as f32 * 0.05);
                }
                ClusterMemRow {
                    source_id: format!("mem_{}", i),
                    content: "x".repeat(3985),
                    entity_id: Some("entity_test".to_string()),
                    entity_name: Some("Test Entity".to_string()),
                    community_id: None,
                    domain: Some("test".to_string()),
                    embedding: emb,
                }
            })
            .collect();

        let indices: Vec<usize> = (0..6).collect();
        let cluster = build_distillation_cluster(&memories, &indices);
        assert!(cluster.estimated_tokens > 3500);

        let result = sub_cluster_by_tokens(&memories, cluster, 3500);
        assert!(result.len() >= 2);
        for sub in &result {
            assert!(
                sub.estimated_tokens <= 3500,
                "sub-cluster too large: {}",
                sub.estimated_tokens
            );
            assert!(!sub.source_ids.is_empty());
        }
        let mut all_ids: Vec<String> = result.iter().flat_map(|c| c.source_ids.clone()).collect();
        all_ids.sort();
        let mut expected: Vec<String> = (0..6).map(|i| format!("mem_{}", i)).collect();
        expected.sort();
        assert_eq!(all_ids, expected);
    }

    #[tokio::test]
    async fn test_source_sync_state_crud() {
        let (db, _dir) = test_db().await;

        // Upsert
        db.upsert_sync_state("obsidian-main", "/vault/note.md", 1712678400_i64, "abc123")
            .await
            .unwrap();

        // Get
        let state = db
            .get_sync_state("obsidian-main", "/vault/note.md")
            .await
            .unwrap();
        assert!(state.is_some());
        let s = state.unwrap();
        assert_eq!(s.content_hash, "abc123");
        assert_eq!(s.mtime_ns, 1712678400);

        // Update
        db.upsert_sync_state("obsidian-main", "/vault/note.md", 1712678500_i64, "def456")
            .await
            .unwrap();
        let updated = db
            .get_sync_state("obsidian-main", "/vault/note.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.content_hash, "def456");

        // List all for source
        db.upsert_sync_state("obsidian-main", "/vault/other.md", 1712678600_i64, "ghi789")
            .await
            .unwrap();
        let all = db.list_sync_state_paths("obsidian-main").await.unwrap();
        assert_eq!(all.len(), 2);

        // Delete one
        db.delete_sync_state("obsidian-main", "/vault/note.md")
            .await
            .unwrap();
        let gone = db
            .get_sync_state("obsidian-main", "/vault/note.md")
            .await
            .unwrap();
        assert!(gone.is_none());

        // Delete all for source
        db.delete_all_sync_state("obsidian-main").await.unwrap();
        let empty = db.list_sync_state_paths("obsidian-main").await.unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn test_find_concept_by_source_memory_no_substring_false_positive() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        // Insert a concept whose only source memory is "mem_10"
        db.insert_concept(
            "concept_1",
            "Test concept",
            Some("A test concept"),
            "Content about something",
            None,
            Some("test"),
            &["mem_10"],
            &now,
        )
        .await
        .unwrap();

        // Searching for "mem_1" must NOT match "mem_10"
        let result = db.find_concept_by_source_memory("mem_1").await.unwrap();
        assert!(
            result.is_none(),
            "find_concept_by_source_memory('mem_1') should NOT match concept with only 'mem_10'"
        );

        // But searching for "mem_10" should match
        let result = db.find_concept_by_source_memory("mem_10").await.unwrap();
        assert!(
            result.is_some(),
            "find_concept_by_source_memory('mem_10') should find the concept"
        );
    }

    #[tokio::test]
    async fn test_sub_cluster_under_limit_returns_unchanged() {
        let memories: Vec<ClusterMemRow> = (0..3)
            .map(|i| ClusterMemRow {
                source_id: format!("mem_{}", i),
                content: "short".to_string(),
                entity_id: Some("entity_test".to_string()),
                entity_name: Some("Test Entity".to_string()),
                community_id: None,
                domain: Some("test".to_string()),
                embedding: vec![0.0f32; 768],
            })
            .collect();

        let indices: Vec<usize> = (0..3).collect();
        let cluster = build_distillation_cluster(&memories, &indices);
        assert!(cluster.estimated_tokens <= 3500);

        let result = sub_cluster_by_tokens(&memories, cluster, 3500);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source_ids.len(), 3);
    }

    #[tokio::test]
    async fn test_import_state_table_exists() {
        let (db, _tmp) = test_db().await;
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='import_state'",
                (),
            )
            .await
            .expect("query sqlite_master");
        let row = rows.next().await.expect("row stream");
        assert!(row.is_some(), "import_state table should exist");
    }

    #[tokio::test]
    async fn test_onboarding_milestones_table_exists() {
        let (db, _tmp) = test_db().await;
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='onboarding_milestones'",
                (),
            )
            .await
            .expect("query sqlite_master");
        let row = rows.next().await.expect("row stream");
        assert!(row.is_some(), "onboarding_milestones table was not created");
    }

    #[tokio::test]
    async fn test_record_milestone_is_idempotent() {
        use crate::onboarding::MilestoneId;
        let (db, _tmp) = test_db().await;

        let first = db
            .record_milestone(MilestoneId::FirstMemory, None)
            .await
            .unwrap();
        assert!(first.is_some(), "first record should return Some");

        let second = db
            .record_milestone(MilestoneId::FirstMemory, None)
            .await
            .unwrap();
        assert!(
            second.is_none(),
            "second record should return None (already fired)"
        );
    }

    #[tokio::test]
    async fn test_list_milestones_returns_all() {
        use crate::onboarding::MilestoneId;
        let (db, _tmp) = test_db().await;
        db.record_milestone(MilestoneId::FirstMemory, None)
            .await
            .unwrap();
        db.record_milestone(
            MilestoneId::FirstConcept,
            Some(serde_json::json!({"id":"c1"})),
        )
        .await
        .unwrap();

        let list = db.list_milestones().await.unwrap();
        assert_eq!(list.len(), 2);
        let has_concept = list
            .iter()
            .any(|m| m.id == MilestoneId::FirstConcept && m.payload.is_some());
        assert!(has_concept);
    }

    #[tokio::test]
    async fn test_acknowledge_milestone_sets_timestamp() {
        use crate::onboarding::MilestoneId;
        let (db, _tmp) = test_db().await;
        db.record_milestone(MilestoneId::FirstMemory, None)
            .await
            .unwrap();

        db.acknowledge_milestone(MilestoneId::FirstMemory)
            .await
            .unwrap();
        let list = db.list_milestones().await.unwrap();
        assert!(list[0].acknowledged_at.is_some());
    }

    #[tokio::test]
    async fn test_increment_milestone_shown_count() {
        use crate::onboarding::MilestoneId;
        let (db, _tmp) = test_db().await;
        db.record_milestone(MilestoneId::FirstConcept, None)
            .await
            .unwrap();

        let c1 = db
            .increment_milestone_shown_count(MilestoneId::FirstConcept)
            .await
            .unwrap();
        assert_eq!(c1, 1);
        let c2 = db
            .increment_milestone_shown_count(MilestoneId::FirstConcept)
            .await
            .unwrap();
        assert_eq!(c2, 2);
    }

    #[tokio::test]
    async fn test_reset_onboarding_milestones_clears_all() {
        use crate::onboarding::MilestoneId;
        let (db, _tmp) = test_db().await;
        db.record_milestone(MilestoneId::FirstMemory, None)
            .await
            .unwrap();
        db.record_milestone(MilestoneId::FirstConcept, None)
            .await
            .unwrap();

        db.reset_onboarding_milestones().await.unwrap();
        assert_eq!(db.list_milestones().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_check_existing_import_source_ids_returns_matches() {
        let (db, _tmp) = test_db().await;
        // Insert two pretend import memories with different source_ids.
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type)
             VALUES ('mem_a', 'x', 'memory', 'import_claude_conv-1', 'title', 0, 1712707200, 'text'),
                    ('mem_b', 'y', 'memory', 'import_claude_conv-2', 'title', 0, 1712707200, 'text')",
            (),
        )
        .await
        .unwrap();
        drop(conn);

        let candidates = vec![
            "import_claude_conv-1".to_string(),
            "import_claude_conv-3".to_string(),
            "import_claude_conv-2".to_string(),
        ];
        let existing = db
            .check_existing_import_source_ids(&candidates)
            .await
            .unwrap();
        assert_eq!(existing.len(), 2);
        assert!(existing.contains("import_claude_conv-1"));
        assert!(existing.contains("import_claude_conv-2"));
        assert!(!existing.contains("import_claude_conv-3"));
    }

    #[tokio::test]
    async fn test_import_state_insert_and_read() {
        let (db, _tmp) = test_db().await;
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO import_state (id, vendor, source_path, total_conversations, processed_conversations, stage, started_at, updated_at)
             VALUES ('imp_1', 'claude', '/tmp/test.zip', 10, 0, 'parsing', '2026-04-10T00:00:00Z', '2026-04-10T00:00:00Z')",
            (),
        )
        .await
        .expect("insert import_state row");
        let mut rows = conn
            .query(
                "SELECT id, vendor, stage FROM import_state WHERE id = 'imp_1'",
                (),
            )
            .await
            .expect("query");
        let row = rows.next().await.expect("row stream").expect("row present");
        let id: String = row.get(0).unwrap();
        let vendor: String = row.get(1).unwrap();
        let stage: String = row.get(2).unwrap();
        assert_eq!(id, "imp_1");
        assert_eq!(vendor, "claude");
        assert_eq!(stage, "parsing");
    }

    #[tokio::test]
    async fn test_get_app_metadata_missing_key_returns_none() {
        let (db, _dir) = test_db().await;
        let result = db.get_app_metadata("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_set_and_get_app_metadata() {
        let (db, _dir) = test_db().await;
        db.set_app_metadata("last_daily_steep_ts", "1712880000")
            .await
            .unwrap();
        let val = db.get_app_metadata("last_daily_steep_ts").await.unwrap();
        assert_eq!(val.as_deref(), Some("1712880000"));
    }

    #[tokio::test]
    async fn test_set_app_metadata_upserts() {
        let (db, _dir) = test_db().await;
        db.set_app_metadata("key", "old").await.unwrap();
        db.set_app_metadata("key", "new").await.unwrap();
        let val = db.get_app_metadata("key").await.unwrap();
        assert_eq!(val.as_deref(), Some("new"));
    }

    #[tokio::test]
    async fn list_recent_retrievals_resolves_memory_ids_to_concept_titles() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "concept_1",
            "Origin positioning",
            None,
            "content",
            None,
            None,
            &["mem_a", "mem_b"],
            &now,
        )
        .await
        .unwrap();

        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_a".into(), "mem_b".into()],
            Some("positioning"),
            "found 2",
        )
        .await
        .unwrap();
        db.log_agent_activity("claude-desktop", "read", &["mem_a".into()], None, "used 1")
            .await
            .unwrap();

        let events = db.list_recent_retrievals(10).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].agent_name, "claude-desktop");
        assert_eq!(events[0].query, None);
        assert_eq!(
            events[0].concept_titles,
            vec!["Origin positioning".to_string()]
        );
        assert_eq!(events[1].agent_name, "claude-code");
        assert_eq!(events[1].query.as_deref(), Some("positioning"));
        assert_eq!(
            events[1].concept_titles,
            vec!["Origin positioning".to_string()]
        );
    }

    #[tokio::test]
    async fn list_recent_retrievals_skips_events_without_concept_or_memory_matches() {
        // mem_orphan_gone does not exist in the memories table (deleted / never
        // stored), so neither a concept title nor a memory snippet can be resolved.
        // The event must still be skipped — nothing useful to show in the UI.
        let (db, _dir) = test_db().await;
        db.log_agent_activity(
            "claude-code",
            "read",
            &["mem_orphan_gone".into()],
            None,
            "used 1",
        )
        .await
        .unwrap();
        let events = db.list_recent_retrievals(10).await.unwrap();
        assert!(
            events.is_empty(),
            "events where memory_ids resolve to neither a concept nor a memory must be skipped"
        );
    }

    #[tokio::test]
    async fn list_recent_retrievals_includes_events_with_only_memory_snippets() {
        // Seed a memory (with content) and an agent_activity row, but do NOT
        // insert any concept referencing that memory.  After the fix the event
        // must be returned with concept_titles = [] and memory_snippets populated.
        let (db, _dir) = test_db().await;
        let content = "Rust ownership rules prevent data races at compile time.";
        db.upsert_documents(vec![make_doc("test", "mem_snippet_only", "", content)])
            .await
            .unwrap();
        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_snippet_only".into()],
            Some("ownership"),
            "found 1",
        )
        .await
        .unwrap();

        let events = db.list_recent_retrievals(10).await.unwrap();
        let ev = events
            .iter()
            .find(|e| {
                e.memory_snippets
                    .iter()
                    .any(|s| s.contains("Rust ownership"))
            })
            .expect("expected an event with the memory snippet");
        assert!(
            ev.concept_titles.is_empty(),
            "concept_titles must be empty when no concept references the memory"
        );
        assert!(
            ev.memory_snippets
                .iter()
                .any(|s| s.starts_with("Rust ownership")),
            "memory_snippets must contain the memory content, got {:?}",
            ev.memory_snippets
        );
    }

    #[tokio::test]
    async fn list_recent_retrievals_dedupes_concept_titles_within_event() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "concept_shared",
            "Shared",
            None,
            "content",
            None,
            None,
            &["mem_a", "mem_b"],
            &now,
        )
        .await
        .unwrap();
        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_a".into(), "mem_b".into()],
            Some("q"),
            "ok",
        )
        .await
        .unwrap();
        let events = db.list_recent_retrievals(10).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].concept_titles, vec!["Shared".to_string()]);
    }

    #[tokio::test]
    async fn list_recent_retrievals_populates_memory_snippets() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let content = "The quick brown fox jumps over the lazy dog. And more text that won't be in the snippet.";
        // Upsert a memory with a recognisable source_id and content.
        db.upsert_documents(vec![make_doc("test", "mem_snip_a", "", content)])
            .await
            .unwrap();
        // Also insert a concept to exercise the combined concept+snippet path.
        db.insert_concept(
            "concept_snip",
            "Snippet Concept",
            None,
            "content",
            None,
            None,
            &["mem_snip_a"],
            &now,
        )
        .await
        .unwrap();
        // Log an agent activity referencing mem_snip_a.
        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_snip_a".into()],
            Some("fox"),
            "found 1",
        )
        .await
        .unwrap();

        let events = db.list_recent_retrievals(10).await.unwrap();
        let e = events
            .iter()
            .find(|ev| !ev.memory_snippets.is_empty())
            .expect("expected at least one event with snippets");
        assert!(
            e.memory_snippets
                .iter()
                .any(|s| s.starts_with("The quick brown fox")),
            "expected snippet starting with 'The quick brown fox', got {:?}",
            e.memory_snippets
        );
    }

    #[tokio::test]
    async fn list_recent_retrievals_snippet_uses_first_chunk_for_multi_chunk_memory() {
        // Seed a multi-chunk memory by inserting two rows with the same source_id
        // but different chunk_index values directly into the DB.
        // The snippet must come from chunk_index=0, not chunk_index=1.
        let (db, _dir) = test_db().await;
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type)
                 VALUES
                   ('multi_0', 'First chunk content is the right snippet.', 'test', 'mem_multi', '', 0, 1712707200, 'text'),
                   ('multi_1', 'Second chunk should never appear in snippet.', 'test', 'mem_multi', '', 1, 1712707200, 'text')",
                (),
            )
            .await
            .unwrap();
        }
        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_multi".into()],
            Some("first chunk"),
            "found 1",
        )
        .await
        .unwrap();

        let events = db.list_recent_retrievals(10).await.unwrap();
        let ev = events
            .iter()
            .find(|e| e.memory_snippets.iter().any(|s| s.contains("First chunk")))
            .expect("expected event with snippet from first chunk");
        assert!(
            !ev.memory_snippets
                .iter()
                .any(|s| s.contains("Second chunk")),
            "snippet must not come from chunk_index=1; got {:?}",
            ev.memory_snippets
        );
        assert!(
            ev.memory_snippets
                .iter()
                .any(|s| s.starts_with("First chunk content")),
            "snippet must come from chunk_index=0; got {:?}",
            ev.memory_snippets
        );
    }

    #[tokio::test]
    async fn list_recent_retrievals_skips_garbage_title_and_uses_content() {
        // Seed a memory whose title is a code fragment (garbage) and whose
        // content has a recognisable sentence.  The snippet must use the content.
        let (db, _dir) = test_db().await;
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type)
                 VALUES ('garbage_title_1', 'Real useful content about Rust ownership.', 'test', 'mem_garbage_title', 'const x = await something()', 0, 1712707200, 'text')",
                (),
            )
            .await
            .unwrap();
        }
        db.log_agent_activity(
            "claude-code",
            "search",
            &["mem_garbage_title".into()],
            Some("ownership"),
            "found 1",
        )
        .await
        .unwrap();

        let events = db.list_recent_retrievals(10).await.unwrap();
        let ev = events
            .iter()
            .find(|e| e.memory_snippets.iter().any(|s| s.contains("Real useful")))
            .expect("expected event with content-based snippet");
        assert!(
            !ev.memory_snippets
                .iter()
                .any(|s| s.starts_with("const x = await")),
            "garbage title must not appear in snippet; got {:?}",
            ev.memory_snippets
        );
        assert!(
            ev.memory_snippets
                .iter()
                .any(|s| s.starts_with("Real useful content")),
            "snippet must come from content; got {:?}",
            ev.memory_snippets
        );
    }

    #[tokio::test]
    async fn list_recent_changes_classifies_created_vs_revised() {
        use origin_types::ConceptChangeKind;

        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let earlier = (chrono::Utc::now() - chrono::Duration::days(3)).to_rfc3339();

        // Created: version==1, created_at == last_modified
        db.insert_concept("c_new", "New idea", None, "content", None, None, &[], &now)
            .await
            .unwrap();

        // Revised: version > 1 (requires direct UPDATE — insert_concept writes version=1)
        db.insert_concept(
            "c_rev",
            "Older idea",
            None,
            "content",
            None,
            None,
            &[],
            &earlier,
        )
        .await
        .unwrap();
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE concepts SET version = 3, last_modified = ?1 WHERE id = ?2",
                libsql::params![now.clone(), "c_rev"],
            )
            .await
            .unwrap();
        }

        let changes = db.list_recent_changes(10).await.unwrap();
        let by_id: std::collections::HashMap<String, ConceptChangeKind> = changes
            .iter()
            .map(|c| (c.concept_id.clone(), c.change_kind))
            .collect();
        assert_eq!(by_id.get("c_new"), Some(&ConceptChangeKind::Created));
        assert_eq!(by_id.get("c_rev"), Some(&ConceptChangeKind::Revised));
    }

    #[tokio::test]
    async fn list_recent_changes_excludes_non_active() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_active",
            "Still here",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.insert_concept("c_gone", "Archived", None, "content", None, None, &[], &now)
            .await
            .unwrap();
        db.archive_concept("c_gone").await.unwrap();

        let changes = db.list_recent_changes(10).await.unwrap();
        let ids: std::collections::HashSet<String> =
            changes.iter().map(|c| c.concept_id.clone()).collect();
        assert!(ids.contains("c_active"));
        assert!(!ids.contains("c_gone"));
    }

    #[tokio::test]
    async fn list_recent_changes_orders_newest_first() {
        let (db, _dir) = test_db().await;
        let older = "2026-01-01T00:00:00Z".to_string();
        let newer = "2026-04-15T12:00:00Z".to_string();
        db.insert_concept("c_old", "Old", None, "content", None, None, &[], &older)
            .await
            .unwrap();
        db.insert_concept("c_new", "New", None, "content", None, None, &[], &newer)
            .await
            .unwrap();

        let changes = db.list_recent_changes(10).await.unwrap();
        assert!(changes.len() >= 2);
        assert_eq!(changes[0].concept_id, "c_new");
        assert_eq!(changes[1].concept_id, "c_old");
    }

    // ==================== list_recent_memories ====================

    /// Insert a memory row directly with explicit created_at and last_modified (Unix seconds).
    async fn insert_memory_at(
        db: &MemoryDB,
        source_id: &str,
        title: &str,
        content: &str,
        created_at: i64,
        last_modified: i64,
        enrichment_status: &str,
    ) {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO memories \
             (id, source, source_id, title, content, chunk_index, chunk_type, word_count, \
              last_modified, enrichment_status, created_at, stability, supersede_mode) \
             VALUES (?1, 'memory', ?2, ?3, ?4, 0, 'text', 10, ?5, ?6, ?7, 'new', 'hide')",
            libsql::params![
                format!("id_{}", source_id),
                source_id,
                title,
                content,
                last_modified,
                enrichment_status,
                created_at
            ],
        )
        .await
        .expect("insert_memory_at failed");
    }

    /// Insert a memory row with explicit `entity_id`, keeping other columns at defaults.
    /// Used to exercise the entity-link branch of `derive_memory_badge` Rule 3.
    #[allow(clippy::too_many_arguments)]
    async fn insert_memory_with_entity(
        db: &MemoryDB,
        source_id: &str,
        title: &str,
        content: &str,
        created_at: i64,
        last_modified: i64,
        enrichment_status: &str,
        entity_id: &str,
    ) {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO memories \
             (id, source, source_id, title, content, chunk_index, chunk_type, word_count, \
              last_modified, enrichment_status, entity_id, created_at, stability, supersede_mode) \
             VALUES (?1, 'memory', ?2, ?3, ?4, 0, 'text', 10, ?5, ?6, ?7, ?8, 'new', 'hide')",
            libsql::params![
                format!("id_{}", source_id),
                source_id,
                title,
                content,
                last_modified,
                enrichment_status,
                entity_id,
                created_at
            ],
        )
        .await
        .expect("insert_memory_with_entity failed");
    }

    #[tokio::test]
    async fn list_recent_memories_returns_top_n_ordered_by_recency() {
        let (db, _dir) = test_db().await;

        // Seed 3 memories with increasing last_modified timestamps (Unix seconds).
        insert_memory_at(
            &db,
            "mem_1",
            "Memory 1",
            "content 1",
            1000,
            1000,
            "enriched",
        )
        .await;
        insert_memory_at(
            &db,
            "mem_2",
            "Memory 2",
            "content 2",
            2000,
            2000,
            "enriched",
        )
        .await;
        insert_memory_at(
            &db,
            "mem_3",
            "Memory 3",
            "content 3",
            3000,
            3000,
            "enriched",
        )
        .await;

        // Request top-2 — should return mem_3 then mem_2 (newest first).
        let items = db.list_recent_memories(2, None).await.unwrap();
        assert_eq!(items.len(), 2, "expected exactly 2 items");
        assert_eq!(items[0].id, "mem_3");
        assert_eq!(items[1].id, "mem_2");
    }

    #[tokio::test]
    async fn list_recent_memories_badge_new_when_created_after_since_ms() {
        let (db, _dir) = test_db().await;

        // mem_old created at 1 s (Unix), mem_new at 5 s.
        insert_memory_at(
            &db,
            "mem_old",
            "Old Memory",
            "old content",
            1,
            1,
            "enriched",
        )
        .await;
        insert_memory_at(
            &db,
            "mem_new",
            "New Memory",
            "new content",
            5,
            5,
            "enriched",
        )
        .await;

        // since_ms = 3_000 ms  →  since_s = 3 s.
        // mem_new (created_at=5) >= 3 → New; mem_old (created_at=1) < 3 → None.
        let items = db.list_recent_memories(10, Some(3_000)).await.unwrap();
        let new_item = items
            .iter()
            .find(|i| i.id == "mem_new")
            .expect("mem_new missing");
        let old_item = items
            .iter()
            .find(|i| i.id == "mem_old")
            .expect("mem_old missing");
        assert!(
            matches!(new_item.badge, origin_types::ActivityBadge::New),
            "expected New, got {:?}",
            new_item.badge
        );
        assert!(
            matches!(old_item.badge, origin_types::ActivityBadge::None),
            "expected None, got {:?}",
            old_item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_memories_badge_refined_when_modified_after_grace_period_and_enriched() {
        let (db, _dir) = test_db().await;

        // created_at = 100_000 s (well before since_s = 400_000 s).
        // last_modified = 500_000 s (after since, and 400_000 s after created_at → >> 60 s grace).
        // enrichment_status derived as 'enriched' from enrichment_steps.
        // since_ms = 400_000_000 ms  →  since_s = 400_000 s.
        insert_memory_at(
            &db,
            "mem_ref",
            "Refined Memory",
            "refined content",
            100_000,
            500_000,
            "enriched",
        )
        .await;
        // Record enrichment steps so the derived status is 'enriched'
        db.record_enrichment_step("mem_ref", "dedup", "ok", None)
            .await
            .unwrap();
        db.record_enrichment_step("mem_ref", "entity_link", "ok", None)
            .await
            .unwrap();

        let items = db
            .list_recent_memories(10, Some(400_000_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "mem_ref")
            .expect("mem_ref missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::Refined),
            "expected Refined, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_memories_badge_none_when_modified_within_grace_period() {
        let (db, _dir) = test_db().await;

        // created_at = 10_000 s, last_modified = 10_030 s (delta = 30 s < 60 s grace).
        // since_ms = 35_000_000 ms  →  since_s = 35_000 s.
        // last_modified (10_030) < since_s (35_000) so Refined path not triggered; → None.
        insert_memory_at(
            &db,
            "mem_grace",
            "Grace Memory",
            "grace content",
            10_000,
            10_030,
            "enriched",
        )
        .await;

        let items = db.list_recent_memories(10, Some(35_000_000)).await.unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "mem_grace")
            .expect("mem_grace missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::None),
            "expected None, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_memories_badge_needs_review_overrides_new() {
        let (db, _dir) = test_db().await;

        // created_at = 5 s; since_ms = 1_000 ms → since_s = 1 s.
        // created_at (5) >= since_s (1) → normally New; NeedsReview should win.
        insert_memory_at(
            &db,
            "mem_nr",
            "NeedsReview Memory",
            "nr content",
            5,
            5,
            "enriched",
        )
        .await;

        // Enqueue detect_contradiction awaiting_review for mem_nr.
        db.insert_refinement_proposal(
            "ref_nr",
            "detect_contradiction",
            &["mem_nr".to_string()],
            None,
            0.9,
        )
        .await
        .unwrap();
        db.resolve_refinement("ref_nr", "awaiting_review")
            .await
            .unwrap();

        let items = db.list_recent_memories(10, Some(1_000)).await.unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "mem_nr")
            .expect("mem_nr missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::NeedsReview),
            "expected NeedsReview, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_memories_badge_refined_when_entity_linked_after_grace_period() {
        let (db, _dir) = test_db().await;

        // created_at = 100_000 s (well before since_s = 400_000 s).
        // last_modified = 500_000 s (after since, and 400_000 s after created_at → >> 60 s grace).
        // enrichment_status = 'pending' (NOT 'enriched') — entity_id branch must carry it alone.
        // entity_id = "ent_linked" (non-empty).
        // since_ms = 400_000_000 ms  →  since_s = 400_000 s.
        insert_memory_with_entity(
            &db,
            "mem_ent",
            "Entity Linked Memory",
            "entity content",
            100_000,
            500_000,
            "pending",
            "ent_linked",
        )
        .await;

        let items = db
            .list_recent_memories(10, Some(400_000_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "mem_ent")
            .expect("mem_ent missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::Refined),
            "expected Refined (entity-link branch), got {:?}",
            item.badge
        );
    }

    // ==================== list_unconfirmed_memories ====================

    #[tokio::test]
    async fn list_unconfirmed_memories_returns_only_unconfirmed() {
        let (db, _dir) = test_db().await;

        // Seed two memories — both default to unconfirmed (confirmed IS NULL after insert).
        insert_memory_at(
            &db,
            "mem_conf",
            "Confirmed",
            "confirmed content",
            1000,
            1000,
            "enriched",
        )
        .await;
        insert_memory_at(
            &db,
            "mem_unconf",
            "Unconfirmed",
            "unconfirmed content",
            2000,
            2000,
            "enriched",
        )
        .await;

        // Flip one to confirmed via the public API — mirrors the real confirm flow.
        db.confirm_memory("mem_conf").await.unwrap();

        let items = db.list_unconfirmed_memories(10).await.unwrap();

        // Confirmed memory must be absent.
        assert!(
            !items.iter().any(|i| i.id == "mem_conf"),
            "confirmed memory leaked into unconfirmed list: {:?}",
            items.iter().map(|i| &i.id).collect::<Vec<_>>()
        );

        // Unconfirmed memory present with badge = NeedsReview.
        let unconf = items
            .iter()
            .find(|i| i.id == "mem_unconf")
            .expect("mem_unconf missing");
        assert!(
            matches!(unconf.badge, origin_types::ActivityBadge::NeedsReview),
            "expected NeedsReview, got {:?}",
            unconf.badge
        );
        assert!(matches!(unconf.kind, origin_types::ActivityKind::Memory));
    }

    #[tokio::test]
    async fn list_unconfirmed_memories_orders_by_recency_desc() {
        let (db, _dir) = test_db().await;

        insert_memory_at(&db, "mem_old", "Old", "old", 1_000, 1_000, "enriched").await;
        insert_memory_at(&db, "mem_mid", "Mid", "mid", 2_000, 2_000, "enriched").await;
        insert_memory_at(&db, "mem_new", "New", "new", 3_000, 3_000, "enriched").await;

        let items = db.list_unconfirmed_memories(10).await.unwrap();
        // All three are unconfirmed; newest (highest COALESCE(last_modified,created_at)) first.
        let ids: Vec<_> = items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["mem_new", "mem_mid", "mem_old"],
            "unconfirmed list not ordered newest-first"
        );
    }

    #[tokio::test]
    async fn list_unconfirmed_memories_respects_limit() {
        let (db, _dir) = test_db().await;

        for i in 0..5 {
            insert_memory_at(
                &db,
                &format!("mem_{i}"),
                "t",
                "c",
                1000 + i as i64,
                1000 + i as i64,
                "enriched",
            )
            .await;
        }

        let items = db.list_unconfirmed_memories(2).await.unwrap();
        assert_eq!(items.len(), 2);
    }

    // ==================== list_recent_concepts_with_badges ====================

    /// Insert a concept row with explicit Unix-second timestamps and version.
    /// `source_memory_ids` is serialised to JSON and stored in the TEXT column.
    #[allow(clippy::too_many_arguments)]
    async fn insert_concept_at(
        db: &MemoryDB,
        id: &str,
        title: &str,
        summary: Option<&str>,
        source_memory_ids: &[&str],
        version: i64,
        created_at_s: i64,
        last_modified_s: i64,
    ) {
        let source_ids_json = serde_json::to_string(&source_memory_ids).unwrap();
        // RFC3339 from unix seconds — lexicographically comparable for UTC timestamps.
        let created_str = chrono::DateTime::from_timestamp(created_at_s, 0)
            .unwrap()
            .to_rfc3339();
        let modified_str = chrono::DateTime::from_timestamp(last_modified_s, 0)
            .unwrap()
            .to_rfc3339();
        let summary_val = summary.unwrap_or("");
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO concepts \
             (id, title, summary, content, entity_id, domain, source_memory_ids, version, status, \
              created_at, last_compiled, last_modified) \
             VALUES (?1, ?2, ?3, 'content', NULL, NULL, ?4, ?5, 'active', ?6, ?6, ?7)",
            libsql::params![
                id,
                title,
                summary_val,
                source_ids_json,
                version,
                created_str,
                modified_str
            ],
        )
        .await
        .expect("insert_concept_at failed");
    }

    #[tokio::test]
    async fn list_recent_concepts_with_badges_new_when_created_after_since_ms() {
        let (db, _dir) = test_db().await;

        // concept_a: created and last_modified at Unix second 5.
        insert_concept_at(&db, "concept_a", "Concept A", None, &[], 1, 5, 5).await;

        // since_ms = 1_000 ms → since_s = 1 s. concept_a.created_at (5s) >= 1s → New.
        let items = db
            .list_recent_concepts_with_badges(10, Some(1_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "concept_a")
            .expect("concept_a missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::New),
            "expected New, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_concepts_with_badges_growing_counts_only_post_since_ms_members() {
        let (db, _dir) = test_db().await;

        // Seed 3 memories at distinct Unix-second timestamps.
        insert_memory_at(&db, "mem_old", "Old", "old", 1, 1, "enriched").await;
        insert_memory_at(&db, "mem_fresh_1", "Fresh 1", "fresh 1", 2, 2, "enriched").await;
        insert_memory_at(&db, "mem_fresh_2", "Fresh 2", "fresh 2", 3, 3, "enriched").await;

        // concept_growing: created_at=1s (before since), last_modified=3s,
        // version=1 (not Revised), three source memories.
        insert_concept_at(
            &db,
            "concept_growing",
            "Growing Concept",
            None,
            &["mem_old", "mem_fresh_1", "mem_fresh_2"],
            1,
            1,
            3,
        )
        .await;

        // since_ms = 2_000 ms → since_s = 2.
        // concept created_at (1s) < since_s (2s) → not New.
        // version = 1 → not Revised.
        // mem_old (created_at=1) < 2 → doesn't qualify; mem_fresh_1 (2) >= 2 and mem_fresh_2 (3) >= 2
        // → added = 2 → Growing { added: 2 }.
        let items = db
            .list_recent_concepts_with_badges(10, Some(2_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "concept_growing")
            .expect("concept_growing missing");
        assert!(
            matches!(
                item.badge,
                origin_types::ActivityBadge::Growing { added: 2 }
            ),
            "expected Growing {{ added: 2 }}, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_concepts_with_badges_revised_when_version_gt_one_and_modified_after_since()
    {
        let (db, _dir) = test_db().await;

        // concept_rev: version=2, created_at=1s (before since), last_modified=5s (after since).
        // No source memories → growth = 0. Revised wins.
        insert_concept_at(&db, "concept_rev", "Revised Concept", None, &[], 2, 1, 5).await;

        // since_ms = 2_000 ms → since_s = 2.
        let items = db
            .list_recent_concepts_with_badges(10, Some(2_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "concept_rev")
            .expect("concept_rev missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::Revised),
            "expected Revised, got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_concepts_with_badges_needs_review_when_any_member_pending() {
        let (db, _dir) = test_db().await;

        // Seed a memory that will be flagged for review.
        insert_memory_at(
            &db,
            "mem_in_concept",
            "In-concept Memory",
            "content",
            5,
            5,
            "enriched",
        )
        .await;

        // Enqueue detect_contradiction awaiting_review for mem_in_concept.
        db.insert_refinement_proposal(
            "ref_concept_flag",
            "detect_contradiction",
            &["mem_in_concept".to_string()],
            None,
            0.9,
        )
        .await
        .unwrap();
        db.resolve_refinement("ref_concept_flag", "awaiting_review")
            .await
            .unwrap();

        // concept_flag: created_at=5s, source mem_in_concept, version=1.
        // since_ms=1_000 → would normally be New, but NeedsReview wins.
        insert_concept_at(
            &db,
            "concept_flag",
            "Flagged Concept",
            None,
            &["mem_in_concept"],
            1,
            5,
            5,
        )
        .await;

        let items = db
            .list_recent_concepts_with_badges(10, Some(1_000))
            .await
            .unwrap();
        let item = items
            .iter()
            .find(|i| i.id == "concept_flag")
            .expect("concept_flag missing");
        assert!(
            matches!(item.badge, origin_types::ActivityBadge::NeedsReview),
            "expected NeedsReview (overrides New), got {:?}",
            item.badge
        );
    }

    #[tokio::test]
    async fn list_recent_concepts_with_badges_returns_top_n_regardless_of_since_ms() {
        let (db, _dir) = test_db().await;

        // Seed 5 concepts at Unix seconds 0..4.
        for i in 0..5i64 {
            insert_concept_at(
                &db,
                &format!("concept_{i}"),
                &format!("Concept {i}"),
                None,
                &[],
                1,
                i,
                i,
            )
            .await;
        }

        // limit=3, since_ms=i64::MAX (no concept satisfies 'new/revised/growing').
        // Expect exactly 3 items, newest first, all badge None.
        let items = db
            .list_recent_concepts_with_badges(3, Some(i64::MAX))
            .await
            .unwrap();
        assert_eq!(items.len(), 3, "expected exactly 3 items");
        assert_eq!(items[0].id, "concept_4", "newest first");
        assert_eq!(items[1].id, "concept_3");
        assert_eq!(items[2].id, "concept_2");
        for item in &items {
            assert!(
                matches!(item.badge, origin_types::ActivityBadge::None),
                "expected None, got {:?}",
                item.badge
            );
        }
    }

    // ==================== Migration 41 (KG quality) ====================

    #[tokio::test]
    async fn test_migration_41_kg_quality_tables() {
        let (db, _dir) = test_db().await;
        let conn = db.conn.lock().await;

        // 1. entity_aliases table exists with correct columns
        let _rows = conn
            .query(
                "SELECT alias_name, canonical_entity_id, created_at, source FROM entity_aliases LIMIT 1",
                (),
            )
            .await
            .expect("entity_aliases table should exist after migration 40");
        drop(_rows);

        // 2. relation_type_vocabulary has >= 15 seed entries
        let mut rows = conn
            .query("SELECT COUNT(*) FROM relation_type_vocabulary", ())
            .await
            .expect("relation_type_vocabulary table should exist after migration 40");
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        assert!(
            count >= 15,
            "relation_type_vocabulary should have at least 15 seed entries, got {count}"
        );
        drop(rows);

        // 3. relations table has confidence, explanation, source_memory_id columns
        let _rows = conn
            .query(
                "SELECT id, confidence, explanation, source_memory_id FROM relations LIMIT 1",
                (),
            )
            .await
            .expect("relations should have confidence, explanation, source_memory_id columns after migration 40");
        drop(_rows);

        // 4. entities table has embedding_updated_at column
        let _rows = conn
            .query("SELECT id, embedding_updated_at FROM entities LIMIT 1", ())
            .await
            .expect("entities should have embedding_updated_at column after migration 40");
        drop(_rows);

        drop(conn);
    }

    #[tokio::test]
    async fn test_alias_resolution() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Alice Chen", "person", None, Some("test"), None)
            .await
            .unwrap();
        let resolved = db.resolve_entity_by_alias("alice chen").await.unwrap();
        assert_eq!(resolved, Some(id.clone()));
        let resolved2 = db.resolve_entity_by_alias("Alice Chen").await.unwrap();
        assert_eq!(resolved2, Some(id.clone()));
        let resolved3 = db.resolve_entity_by_alias("bob").await.unwrap();
        assert_eq!(resolved3, None);
    }

    #[tokio::test]
    async fn test_add_entity_alias() {
        let (db, _dir) = test_db().await;
        let id = db
            .store_entity("Alice Chen", "person", None, Some("test"), None)
            .await
            .unwrap();
        db.add_entity_alias("alice", &id, "auto").await.unwrap();
        let r1 = db.resolve_entity_by_alias("alice chen").await.unwrap();
        let r2 = db.resolve_entity_by_alias("alice").await.unwrap();
        assert_eq!(r1, Some(id.clone()));
        assert_eq!(r2, Some(id.clone()));
    }

    #[tokio::test]
    async fn test_resolve_relation_type() {
        let (db, _dir) = test_db().await;
        assert_eq!(
            db.resolve_relation_type("works_on").await.unwrap(),
            "works_on"
        );
        assert_eq!(
            db.resolve_relation_type("working_at").await.unwrap(),
            "works_on"
        );
        assert_eq!(
            db.resolve_relation_type("custom_type").await.unwrap(),
            "custom_type"
        );
    }

    // ==================== Topic-key upsert feature tests ====================

    // ---- link_concept_source + get_concept_sources ----

    #[tokio::test]
    async fn test_link_concept_source_idempotent() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        // Insert a concept and a memory to satisfy FK + existence expectations.
        db.insert_concept("c1", "Concept One", None, "content", None, None, &[], &now)
            .await
            .unwrap();
        let mem_doc = make_memory_doc(
            "src1",
            "Some content about topic.",
            "knowledge",
            "work",
            "agent",
        );
        db.upsert_documents(vec![mem_doc]).await.unwrap();

        // Link once.
        db.link_concept_source("c1", "src1", "initial_link")
            .await
            .unwrap();
        // Link again with the same (concept_id, memory_source_id) — idempotent INSERT OR IGNORE.
        db.link_concept_source("c1", "src1", "duplicate_link")
            .await
            .unwrap();

        let sources = db.get_concept_sources("c1").await.unwrap();
        assert_eq!(
            sources.len(),
            1,
            "idempotent insert should produce exactly 1 row"
        );
        assert_eq!(sources[0].concept_id, "c1");
        assert_eq!(sources[0].memory_source_id, "src1");
    }

    #[tokio::test]
    async fn test_get_concept_sources_ordered() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_ord",
            "Ordering test",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        // Insert two memories.
        db.upsert_documents(vec![make_memory_doc(
            "src_a",
            "Alpha content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "src_b",
            "Beta content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        // Manually insert with controlled linked_at values so order is deterministic.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT OR IGNORE INTO concept_sources (concept_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, ?3, ?4)",
                libsql::params!["c_ord", "src_b", 200i64, "later"],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT OR IGNORE INTO concept_sources (concept_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, ?3, ?4)",
                libsql::params!["c_ord", "src_a", 100i64, "earlier"],
            )
            .await
            .unwrap();
        }

        let sources = db.get_concept_sources("c_ord").await.unwrap();
        assert_eq!(sources.len(), 2);
        // Should be ordered by linked_at ASC: src_a first, src_b second.
        assert_eq!(sources[0].memory_source_id, "src_a");
        assert_eq!(sources[1].memory_source_id, "src_b");
    }

    // ---- get_concepts_for_memory ----

    #[tokio::test]
    async fn test_get_concepts_for_memory_reverse_lookup() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.upsert_documents(vec![make_memory_doc(
            "mem_rev",
            "Reverse lookup content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        db.insert_concept(
            "concept_x",
            "Concept X",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.insert_concept(
            "concept_y",
            "Concept Y",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        // Link the same memory to both concepts.
        db.link_concept_source("concept_x", "mem_rev", "reason_x")
            .await
            .unwrap();
        db.link_concept_source("concept_y", "mem_rev", "reason_y")
            .await
            .unwrap();

        let concepts = db.get_concepts_for_memory("mem_rev").await.unwrap();
        assert_eq!(concepts.len(), 2, "memory should be linked to 2 concepts");

        let ids: Vec<&str> = concepts.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"concept_x"), "concept_x should be in result");
        assert!(ids.contains(&"concept_y"), "concept_y should be in result");
    }

    #[tokio::test]
    async fn test_get_concepts_for_memory_excludes_archived() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.upsert_documents(vec![make_memory_doc(
            "mem_arc",
            "Archived concept test",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        db.insert_concept(
            "c_active",
            "Active concept",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.insert_concept(
            "c_archived",
            "Archived concept",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.archive_concept("c_archived").await.unwrap();

        db.link_concept_source("c_active", "mem_arc", "r")
            .await
            .unwrap();
        db.link_concept_source("c_archived", "mem_arc", "r")
            .await
            .unwrap();

        let concepts = db.get_concepts_for_memory("mem_arc").await.unwrap();
        // Only active concepts should be returned.
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].id, "c_active");
    }

    // ---- cleanup_orphaned_concept_sources ----

    #[tokio::test]
    async fn test_cleanup_orphaned_concept_sources() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_clean",
            "Cleanup test",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        // Insert a memory, link it, then delete it directly.
        db.upsert_documents(vec![make_memory_doc(
            "mem_ghost",
            "Ghost memory",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.link_concept_source("c_clean", "mem_ghost", "reason")
            .await
            .unwrap();

        // Delete the memory directly to simulate an orphan.
        {
            let conn = db.conn.lock().await;
            conn.execute("DELETE FROM memories WHERE source_id = 'mem_ghost'", ())
                .await
                .unwrap();
        }

        // Confirm orphan exists.
        let sources_before = db.get_concept_sources("c_clean").await.unwrap();
        assert_eq!(
            sources_before.len(),
            1,
            "orphan row should be present before cleanup"
        );

        let removed = db.cleanup_orphaned_concept_sources().await.unwrap();
        assert_eq!(removed, 1, "should remove exactly 1 orphaned row");

        let sources_after = db.get_concept_sources("c_clean").await.unwrap();
        assert_eq!(
            sources_after.len(),
            0,
            "orphan row should be gone after cleanup"
        );
    }

    #[tokio::test]
    async fn test_cleanup_orphaned_concept_sources_keeps_valid() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_keep",
            "Keep test",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "mem_valid",
            "Valid memory",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.link_concept_source("c_keep", "mem_valid", "reason")
            .await
            .unwrap();

        let removed = db.cleanup_orphaned_concept_sources().await.unwrap();
        assert_eq!(
            removed, 0,
            "no orphans should be removed when all memories exist"
        );

        let sources = db.get_concept_sources("c_keep").await.unwrap();
        assert_eq!(sources.len(), 1, "valid row should be intact");
    }

    // ---- topic_match_candidates ----

    #[tokio::test]
    async fn test_topic_match_candidates_filters() {
        let (db, _dir) = test_db().await;

        // Insert memories with different domain/type combinations.
        db.upsert_documents(vec![make_memory_doc(
            "m_work_know",
            "Work knowledge content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "m_work_pref",
            "Work preference content",
            "preference",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "m_personal_know",
            "Personal knowledge content",
            "knowledge",
            "personal",
            "agent",
        )])
        .await
        .unwrap();

        // Query for domain=work, type=knowledge — exact match should come first.
        let candidates = db
            .topic_match_candidates(Some("work"), Some("knowledge"), 100)
            .await
            .unwrap();

        let ids: Vec<&str> = candidates.iter().map(|c| c.source_id.as_str()).collect();
        assert!(
            ids.contains(&"m_work_know"),
            "m_work_know should be in candidates"
        );
        // Flexible matching: all candidates returned, exact match prioritized
        assert!(
            ids[0] == "m_work_know",
            "exact domain+type match should be first"
        );
        // Other memories are also returned (lower priority)
        assert!(ids.len() == 3, "all 3 memories should be candidates");
    }

    #[tokio::test]
    async fn test_topic_match_candidates_works_with_missing_filters() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory_doc(
            "m1",
            "Some content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        // Missing domain => still returns candidates (flexible matching).
        let candidates = db
            .topic_match_candidates(None, Some("knowledge"), 100)
            .await
            .unwrap();
        assert!(
            !candidates.is_empty(),
            "missing domain should still return candidates"
        );

        // Missing type => still returns candidates.
        let candidates = db
            .topic_match_candidates(Some("work"), None, 100)
            .await
            .unwrap();
        assert!(
            !candidates.is_empty(),
            "missing type should still return candidates"
        );
    }

    #[tokio::test]
    async fn test_topic_match_candidates_respects_max() {
        let (db, _dir) = test_db().await;

        for i in 0..5 {
            let sid = format!("m_max_{i}");
            let content = format!("Content item number {i} about some topic");
            db.upsert_documents(vec![make_memory_doc(
                &sid,
                &content,
                "knowledge",
                "work",
                "agent",
            )])
            .await
            .unwrap();
        }

        let candidates = db
            .topic_match_candidates(Some("work"), Some("knowledge"), 3)
            .await
            .unwrap();
        assert!(
            candidates.len() <= 3,
            "max_candidates limit should be respected"
        );
    }

    // ---- find_topic_match tiered thresholds ----

    /// Helper: store a memory and return its embedding for use in topic-match tests.
    async fn store_and_embed(
        db: &MemoryDB,
        source_id: &str,
        content: &str,
        memory_type: &str,
        domain: &str,
    ) -> Vec<f32> {
        db.upsert_documents(vec![make_memory_doc(
            source_id,
            content,
            memory_type,
            domain,
            "agent",
        )])
        .await
        .unwrap();
        db.generate_embeddings(&[content.to_string()])
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
    }

    #[tokio::test]
    async fn test_find_topic_match_exact_tier() {
        let (db, _dir) = test_db().await;
        let config = crate::tuning::TopicMatchConfig::default();

        // Store a memory about Rust async patterns
        let _ = store_and_embed(
            &db,
            "m_rust",
            "Rust async patterns use tokio runtime with spawn and select macros",
            "knowledge",
            "rust-project",
        )
        .await;

        // Query with SAME domain+type — should use exact tier (0.70)
        let query_emb = db
            .generate_embeddings(&[
                "Rust async patterns with tokio runtime and spawn for concurrency".to_string(),
            ])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let result = crate::topic_match::find_topic_match(
            &db,
            "Rust async patterns",
            Some("knowledge"),
            Some("rust-project"),
            None,
            &query_emb,
            &config,
        )
        .await
        .unwrap();

        assert!(
            result.matched_source_id.is_some(),
            "exact tier (domain+type match) should find a match"
        );
        assert_eq!(result.matched_source_id.as_deref(), Some("m_rust"));
    }

    #[tokio::test]
    async fn test_find_topic_match_partial_tier_different_type() {
        let (db, _dir) = test_db().await;
        let config = crate::tuning::TopicMatchConfig::default();

        let _ = store_and_embed(
            &db,
            "m_db",
            "PostgreSQL database with pgvector extension for vector search",
            "decision",
            "myproject",
        )
        .await;

        // Query with same domain but DIFFERENT type — partial tier (0.80)
        let query_emb = db
            .generate_embeddings(&[
                "PostgreSQL database using pgvector for vector similarity search".to_string(),
            ])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let result = crate::topic_match::find_topic_match(
            &db,
            "Database choice",
            Some("fact"),
            Some("myproject"),
            None,
            &query_emb,
            &config,
        )
        .await
        .unwrap();

        // Similarity should be high enough for partial tier
        assert!(
            result.matched_source_id.is_some(),
            "partial tier (same domain, different type) should match when similarity is high"
        );
    }

    #[tokio::test]
    async fn test_find_topic_match_no_domain_still_matches() {
        let (db, _dir) = test_db().await;
        let config = crate::tuning::TopicMatchConfig::default();

        let _ = store_and_embed(
            &db,
            "m_theme",
            "Dark mode theme preference for all editors and terminals",
            "preference",
            "personal",
        )
        .await;

        // Query with NO domain — should still find match if similarity is very high
        let query_emb = db
            .generate_embeddings(&[
                "Dark mode theme preference for all editors and terminals".to_string()
            ])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let result = crate::topic_match::find_topic_match(
            &db,
            "Theme preference",
            None,
            None,
            None,
            &query_emb,
            &config,
        )
        .await
        .unwrap();

        // Near-identical content should pass even the semantic-only tier (0.90)
        assert!(
            result.matched_source_id.is_some(),
            "semantic-only tier should match with very high similarity"
        );
    }

    #[tokio::test]
    async fn test_find_topic_match_different_topic_no_match() {
        let (db, _dir) = test_db().await;
        let config = crate::tuning::TopicMatchConfig::default();

        let _ = store_and_embed(
            &db,
            "m_frontend",
            "React 19 with server components for the frontend UI layer",
            "decision",
            "myproject",
        )
        .await;

        // Query about a completely different topic
        let query_emb = db
            .generate_embeddings(&[
                "Kubernetes deployment strategy using blue-green rollouts for zero downtime"
                    .to_string(),
            ])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let result = crate::topic_match::find_topic_match(
            &db,
            "Deployment strategy",
            Some("decision"),
            Some("myproject"),
            None,
            &query_emb,
            &config,
        )
        .await
        .unwrap();

        assert!(
            result.matched_source_id.is_none(),
            "completely different topic should not match even with same domain+type"
        );
    }

    #[tokio::test]
    async fn test_find_topic_match_below_partial_threshold_no_match() {
        let (db, _dir) = test_db().await;
        let config = crate::tuning::TopicMatchConfig::default();

        let _ = store_and_embed(
            &db,
            "m_auth",
            "JWT authentication with RS256 signing for API endpoints",
            "decision",
            "backend",
        )
        .await;

        // Somewhat related content but different domain+type — needs 0.80 for partial tier
        let query_emb = db
            .generate_embeddings(&[
                "OAuth2 authorization flow with PKCE for mobile app login".to_string()
            ])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let result = crate::topic_match::find_topic_match(
            &db,
            "Auth approach",
            Some("fact"),
            Some("mobile"),
            None,
            &query_emb,
            &config,
        )
        .await
        .unwrap();

        // Related but different enough that it should NOT match at the none tier (0.90)
        // This tests that the tiered thresholds actually prevent false positives
        assert!(
            result.matched_source_id.is_none(),
            "related-but-different content should not match across domain+type at high threshold"
        );
    }

    // ---- upsert_memory_in_place ----

    #[tokio::test]
    async fn test_upsert_memory_in_place_updates_version() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_uip",
            "Original content for upsert test.",
            "knowledge",
            "work",
            "agent",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Compute an embedding for the new content.
        let new_content = "Updated content after in-place upsert.";
        let embedding = db
            .generate_embeddings(&[new_content.to_string()])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        db.upsert_memory_in_place(
            "mem_uip",
            new_content,
            &embedding,
            Some("test-agent"),
            None,
            50,
        )
        .await
        .unwrap();

        // Verify content and version via direct DB query.
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT content, version FROM memories WHERE source_id = 'mem_uip' AND chunk_index = 0",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().expect("row should exist");
        let content: String = row.get(0).unwrap();
        let version: i64 = row.get(1).unwrap();

        assert_eq!(content, new_content, "content should be updated");
        assert_eq!(version, 2, "version should be incremented to 2");
    }

    #[tokio::test]
    async fn test_upsert_memory_in_place_changelog() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_cl", "Initial content.", "knowledge", "work", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();

        let new_content = "Content after first upsert.";
        let embedding = db
            .generate_embeddings(&[new_content.to_string()])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        db.upsert_memory_in_place(
            "mem_cl",
            new_content,
            &embedding,
            Some("my-agent"),
            Some("ext-src-1"),
            50,
        )
        .await
        .unwrap();

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT changelog FROM memories WHERE source_id = 'mem_cl' AND chunk_index = 0",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().expect("row should exist");
        let changelog_json: String = row.get(0).unwrap();
        let changelog: Vec<serde_json::Value> = serde_json::from_str(&changelog_json).unwrap();

        assert_eq!(changelog.len(), 1, "one changelog entry after first upsert");
        assert_eq!(changelog[0]["version"], 2);
        assert_eq!(changelog[0]["source_agent"], "my-agent");
        assert_eq!(changelog[0]["incoming_source_id"], "ext-src-1");
    }

    #[tokio::test]
    async fn test_upsert_memory_in_place_changelog_cap() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_cap", "Starting content.", "knowledge", "work", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();

        let cap = 50usize;
        // Upsert 55 times; changelog should be capped at `cap` entries.
        for i in 0..55 {
            let content = format!("Iteration {i} content for cap test.");
            let embedding = db
                .generate_embeddings(std::slice::from_ref(&content))
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            db.upsert_memory_in_place("mem_cap", &content, &embedding, None, None, cap)
                .await
                .unwrap();
        }

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT changelog FROM memories WHERE source_id = 'mem_cap' AND chunk_index = 0",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().expect("row should exist");
        let changelog_json: String = row.get(0).unwrap();
        let changelog: Vec<serde_json::Value> = serde_json::from_str(&changelog_json).unwrap();

        assert_eq!(
            changelog.len(),
            cap,
            "changelog should be capped at {cap} entries, got {}",
            changelog.len()
        );
    }

    // ---- is_memory_protected ----

    #[tokio::test]
    async fn test_is_memory_protected_confirmed() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_prot_conf",
            "Content to protect.",
            "knowledge",
            "work",
            "agent",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Initially not protected.
        assert!(!db.is_memory_protected("mem_prot_conf").await.unwrap());

        // Set confirmed=1 directly.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE memories SET confirmed = 1 WHERE source_id = 'mem_prot_conf'",
                (),
            )
            .await
            .unwrap();
        }

        assert!(
            db.is_memory_protected("mem_prot_conf").await.unwrap(),
            "confirmed memory should be protected"
        );
    }

    #[tokio::test]
    async fn test_is_memory_protected_stability_learned() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_prot_stab",
            "Content to protect by stability.",
            "knowledge",
            "work",
            "agent",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Initially not protected (stability='new' by default).
        assert!(!db.is_memory_protected("mem_prot_stab").await.unwrap());

        // Set stability='learned'.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE memories SET stability = 'learned' WHERE source_id = 'mem_prot_stab'",
                (),
            )
            .await
            .unwrap();
        }

        assert!(
            db.is_memory_protected("mem_prot_stab").await.unwrap(),
            "stability='learned' memory should be protected"
        );
    }

    #[tokio::test]
    async fn test_is_memory_protected_stability_confirmed() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_prot_stab2",
            "Content for stability confirmed.",
            "knowledge",
            "work",
            "agent",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE memories SET stability = 'confirmed' WHERE source_id = 'mem_prot_stab2'",
                (),
            )
            .await
            .unwrap();
        }

        assert!(
            db.is_memory_protected("mem_prot_stab2").await.unwrap(),
            "stability='confirmed' memory should be protected"
        );
    }

    #[tokio::test]
    async fn test_is_memory_protected_nonexistent() {
        let (db, _dir) = test_db().await;
        // Non-existent source_id should return false, not an error.
        let result = db.is_memory_protected("no_such_id").await.unwrap();
        assert!(!result, "non-existent memory should not be protected");
    }

    // ---- set_concept_stale / clear_concept_staleness / list_stale_concepts ----

    #[tokio::test]
    async fn test_stale_concepts_lifecycle() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_stale",
            "Stale concept",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        // Initially no stale concepts.
        let stale = db.list_stale_concepts("source_updated").await.unwrap();
        assert!(stale.is_empty(), "no stale concepts initially");

        // Mark stale.
        db.set_concept_stale("c_stale", "source_updated")
            .await
            .unwrap();

        let stale = db.list_stale_concepts("source_updated").await.unwrap();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].id, "c_stale");
        assert_eq!(stale[0].stale_reason.as_deref(), Some("source_updated"));

        // Clear staleness.
        db.clear_concept_staleness("c_stale").await.unwrap();

        let stale_after = db.list_stale_concepts("source_updated").await.unwrap();
        assert!(stale_after.is_empty(), "staleness should be cleared");

        // Verify stale_reason is NULL and sources_updated_count is 0 after clearing.
        let c = db.get_concept("c_stale").await.unwrap().unwrap();
        assert!(
            c.stale_reason.is_none(),
            "stale_reason should be None after clearing"
        );
        assert_eq!(c.sources_updated_count, 0);
    }

    #[tokio::test]
    async fn test_list_stale_concepts_excludes_archived() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_stale_arc",
            "Stale archived",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.set_concept_stale("c_stale_arc", "source_updated")
            .await
            .unwrap();
        db.archive_concept("c_stale_arc").await.unwrap();

        let stale = db.list_stale_concepts("source_updated").await.unwrap();
        assert!(
            stale.is_empty(),
            "archived concepts should not appear in stale list"
        );
    }

    // ---- increment_concept_sources_updated ----

    #[tokio::test]
    async fn test_increment_concept_sources_updated() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_concept(
            "c_inc",
            "Increment test",
            None,
            "content",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        // sources_updated_count starts at 0.
        let c = db.get_concept("c_inc").await.unwrap().unwrap();
        assert_eq!(c.sources_updated_count, 0);

        db.increment_concept_sources_updated("c_inc").await.unwrap();
        db.increment_concept_sources_updated("c_inc").await.unwrap();

        let c = db.get_concept("c_inc").await.unwrap().unwrap();
        assert_eq!(
            c.sources_updated_count, 2,
            "should be 2 after two increments"
        );
    }

    // ---- get_memories_by_source_ids ----

    #[tokio::test]
    async fn test_get_memories_by_source_ids() {
        let (db, _dir) = test_db().await;

        // Insert 3 memories.
        db.upsert_documents(vec![make_memory_doc(
            "sid_1",
            "First memory content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "sid_2",
            "Second memory content",
            "preference",
            "personal",
            "agent",
        )])
        .await
        .unwrap();
        db.upsert_documents(vec![make_memory_doc(
            "sid_3",
            "Third memory content",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        // Fetch only 2 of the 3.
        let ids = vec!["sid_1".to_string(), "sid_3".to_string()];
        let results = db.get_memories_by_source_ids(&ids).await.unwrap();

        assert_eq!(results.len(), 2, "should return exactly 2 memories");
        let result_ids: Vec<&str> = results.iter().map(|m| m.source_id.as_str()).collect();
        assert!(result_ids.contains(&"sid_1"));
        assert!(result_ids.contains(&"sid_3"));
        assert!(!result_ids.contains(&"sid_2"), "sid_2 was not requested");
    }

    #[tokio::test]
    async fn test_get_memories_by_source_ids_empty_input() {
        let (db, _dir) = test_db().await;

        let results = db.get_memories_by_source_ids(&[]).await.unwrap();
        assert!(results.is_empty(), "empty input should return empty result");
    }

    #[tokio::test]
    async fn test_get_memories_by_source_ids_missing_id() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory_doc(
            "sid_real",
            "Real memory",
            "knowledge",
            "work",
            "agent",
        )])
        .await
        .unwrap();

        // Request one real and one non-existent id.
        let ids = vec!["sid_real".to_string(), "sid_fake".to_string()];
        let results = db.get_memories_by_source_ids(&ids).await.unwrap();

        assert_eq!(results.len(), 1, "should only return the existing memory");
        assert_eq!(results[0].source_id, "sid_real");
    }

    // ---- topic_match_title_fts ----

    #[tokio::test]
    async fn test_topic_match_title_fts_basic() {
        let (db, _dir) = test_db().await;

        // Insert a memory whose title contains a searchable word.
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "fts_mem".to_string(),
            title: "Rust programming tips".to_string(),
            content: "Some content about Rust programming.".to_string(),
            memory_type: Some("knowledge".to_string()),
            domain: Some("work".to_string()),
            source_agent: Some("agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(false),
            supersede_mode: "hide".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Searching for "Rust" in title should match "fts_mem".
        let matched = db
            .topic_match_title_fts("Rust programming", &["fts_mem"])
            .await
            .unwrap();
        assert!(
            matched.contains(&"fts_mem".to_string()),
            "fts_mem should match title search for 'Rust'"
        );
    }

    #[tokio::test]
    async fn test_topic_match_title_fts_candidate_filter() {
        let (db, _dir) = test_db().await;

        // Insert two memories with similar titles.
        for sid in &["fts_a", "fts_b"] {
            let doc = crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: sid.to_string(),
                title: format!("Machine learning notes {sid}"),
                content: format!("Content for {sid}"),
                memory_type: Some("knowledge".to_string()),
                domain: Some("work".to_string()),
                source_agent: Some("agent".to_string()),
                confidence: Some(0.9),
                confirmed: Some(false),
                supersede_mode: "hide".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // Only pass "fts_a" as a candidate — "fts_b" should be excluded even if it matches FTS.
        let matched = db
            .topic_match_title_fts("Machine learning", &["fts_a"])
            .await
            .unwrap();
        assert!(matched.contains(&"fts_a".to_string()), "fts_a should match");
        assert!(
            !matched.contains(&"fts_b".to_string()),
            "fts_b not in candidate set, should be excluded"
        );
    }

    #[tokio::test]
    async fn test_topic_match_title_fts_empty_candidates() {
        let (db, _dir) = test_db().await;

        let matched = db.topic_match_title_fts("anything", &[]).await.unwrap();
        assert!(
            matched.is_empty(),
            "empty candidate set should yield empty result"
        );
    }

    // ==================== Migration 43: enrichment_steps ====================

    #[tokio::test]
    async fn test_record_and_get_enrichment_steps() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_step_test", "Rust is great", "fact", "tech", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();
        db.record_enrichment_step("mem_step_test", "dedup", "ok", None)
            .await
            .unwrap();
        db.record_enrichment_step("mem_step_test", "entity_link", "failed", Some("timeout"))
            .await
            .unwrap();
        db.record_enrichment_step("mem_step_test", "title_enrich", "skipped", None)
            .await
            .unwrap();
        let steps = db.get_enrichment_steps("mem_step_test").await.unwrap();
        assert_eq!(steps.len(), 3);
        let dedup = steps.iter().find(|s| s.step == "dedup").unwrap();
        assert_eq!(dedup.status, "ok");
        assert!(dedup.error.is_none());
        assert_eq!(dedup.attempts, 1);
        let entity = steps.iter().find(|s| s.step == "entity_link").unwrap();
        assert_eq!(entity.status, "failed");
        assert_eq!(entity.error.as_deref(), Some("timeout"));
        let title = steps.iter().find(|s| s.step == "title_enrich").unwrap();
        assert_eq!(title.status, "skipped");
    }

    #[tokio::test]
    async fn test_record_enrichment_step_upserts_on_retry() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_upsert_step", "test content", "fact", "tech", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();
        db.record_enrichment_step(
            "mem_upsert_step",
            "entity_extract",
            "failed",
            Some("LLM down"),
        )
        .await
        .unwrap();
        let steps = db.get_enrichment_steps("mem_upsert_step").await.unwrap();
        assert_eq!(steps[0].attempts, 1);
        db.record_enrichment_step(
            "mem_upsert_step",
            "entity_extract",
            "failed",
            Some("still down"),
        )
        .await
        .unwrap();
        let steps = db.get_enrichment_steps("mem_upsert_step").await.unwrap();
        assert_eq!(steps[0].attempts, 2);
        assert_eq!(steps[0].error.as_deref(), Some("still down"));
        db.record_enrichment_step("mem_upsert_step", "entity_extract", "ok", None)
            .await
            .unwrap();
        let steps = db.get_enrichment_steps("mem_upsert_step").await.unwrap();
        assert_eq!(steps[0].status, "ok");
        assert!(steps[0].error.is_none());
        assert_eq!(steps[0].attempts, 3);
    }

    #[tokio::test]
    async fn test_get_enrichment_summary() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_summary_test", "test", "fact", "tech", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();
        let summary = db.get_enrichment_summary("mem_summary_test").await.unwrap();
        assert_eq!(summary, "raw");
        db.record_enrichment_step("mem_summary_test", "dedup", "ok", None)
            .await
            .unwrap();
        db.record_enrichment_step("mem_summary_test", "entity_link", "skipped", None)
            .await
            .unwrap();
        let summary = db.get_enrichment_summary("mem_summary_test").await.unwrap();
        assert_eq!(summary, "enriched");
        db.record_enrichment_step("mem_summary_test", "title_enrich", "failed", Some("err"))
            .await
            .unwrap();
        let summary = db.get_enrichment_summary("mem_summary_test").await.unwrap();
        assert_eq!(summary, "enrichment_partial");
    }

    #[tokio::test]
    async fn test_get_enrichment_summary_all_failed() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_all_fail", "test", "fact", "tech", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();
        db.record_enrichment_step("mem_all_fail", "dedup", "failed", Some("err1"))
            .await
            .unwrap();
        db.record_enrichment_step("mem_all_fail", "entity_link", "failed", Some("err2"))
            .await
            .unwrap();
        let summary = db.get_enrichment_summary("mem_all_fail").await.unwrap();
        assert_eq!(summary, "enrichment_failed");
    }

    #[tokio::test]
    async fn test_reset_abandoned_steps() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc("mem_abandon_test", "test", "fact", "tech", "agent");
        db.upsert_documents(vec![doc]).await.unwrap();
        db.record_enrichment_step(
            "mem_abandon_test",
            "entity_extract",
            "abandoned",
            Some("gave up"),
        )
        .await
        .unwrap();
        db.record_enrichment_step("mem_abandon_test", "dedup", "ok", None)
            .await
            .unwrap();
        let reset = db.reset_abandoned_steps("mem_abandon_test").await.unwrap();
        assert_eq!(reset, 1);
        let steps = db.get_enrichment_steps("mem_abandon_test").await.unwrap();
        let extract = steps.iter().find(|s| s.step == "entity_extract").unwrap();
        assert_eq!(extract.status, "failed");
        assert_eq!(extract.attempts, 0);
    }

    #[tokio::test]
    async fn test_list_memories_derives_enrichment_status_from_steps() {
        let (db, _dir) = test_db().await;
        let doc = make_memory_doc(
            "mem_list_enrich",
            "Rust ownership model",
            "fact",
            "tech",
            "test",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Before any steps are recorded, status should be "raw"
        let items = db.list_memories(None, None, None, None, 10).await.unwrap();
        let item = items
            .iter()
            .find(|i| i.source_id == "mem_list_enrich")
            .unwrap();
        assert_eq!(item.enrichment_status, "raw", "no steps => raw");

        // Record one ok step and one failed step => partial
        db.record_enrichment_step("mem_list_enrich", "dedup", "ok", None)
            .await
            .unwrap();
        db.record_enrichment_step(
            "mem_list_enrich",
            "entity_link",
            "failed",
            Some("no entities"),
        )
        .await
        .unwrap();

        let items = db.list_memories(None, None, None, None, 10).await.unwrap();
        let item = items
            .iter()
            .find(|i| i.source_id == "mem_list_enrich")
            .unwrap();
        assert_eq!(
            item.enrichment_status, "enrichment_partial",
            "mix of ok and failed => partial"
        );

        // Also verify get_memory_detail derives from steps
        let detail = db
            .get_memory_detail("mem_list_enrich")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            detail.enrichment_status, "enrichment_partial",
            "detail also derives from steps"
        );
    }
}
