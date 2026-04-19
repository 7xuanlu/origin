// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Constants ───────────────────────────────────────────────────────────

pub const UNSORTED_SPACE_ID: &str = "unsorted";

/// Maximum number of closed activity streams to retain.
const MAX_CLOSED_STREAMS: usize = 500;

/// Seed spaces shipped with the app.
const SEED_SPACES: &[(&str, &str, &str, &str, &[&str])] = &[
    // (id, name, icon, color, app_patterns)
    (
        "code",
        "Code",
        "terminal",
        "#22c55e",
        &[
            "code",
            "terminal",
            "iterm",
            "warp",
            "alacritty",
            "xcode",
            "intellij",
            "webstorm",
            "pycharm",
            "neovim",
            "vim",
            "emacs",
            "cursor",
            "zed",
        ],
    ),
    (
        "communication",
        "Communication",
        "chat",
        "#3b82f6",
        &[
            "slack", "discord", "messages", "mail", "telegram", "whatsapp", "teams", "zoom",
        ],
    ),
    (
        "research",
        "Research",
        "globe",
        "#a855f7",
        &["safari", "firefox", "chrome", "arc", "brave", "edge"],
    ),
    (
        "writing",
        "Writing",
        "pencil",
        "#f59e0b",
        &[
            "notion",
            "obsidian",
            "notes",
            "pages",
            "word",
            "bear",
            "ia writer",
            "typora",
            "ulysses",
        ],
    ),
    (
        "design",
        "Design",
        "paintbrush",
        "#ec4899",
        &[
            "figma",
            "sketch",
            "pixelmator",
            "photoshop",
            "illustrator",
            "canva",
            "framer",
        ],
    ),
];

// ── Core structs ────────────────────────────────────────────────────────

/// The kind of rule used to match documents/captures to a space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SpaceRuleKind {
    App,
    Path,
    Keyword,
    UrlPattern,
}

/// A single matching rule within a space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceRule {
    pub kind: SpaceRuleKind,
    pub pattern: String,
}

/// A space groups related documents and captures by activity context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    pub id: String,
    pub name: String,
    pub icon: String,
    pub color: String,
    pub rules: Vec<SpaceRule>,
    pub pinned: bool,
    pub auto_detected: bool,
    pub created_at: u64,
}

impl Space {
    /// Create a new auto-detected space (discovered from an unknown app).
    pub fn new_auto(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            icon: "sparkles".to_string(),
            color: "#6b7280".to_string(),
            rules: vec![SpaceRule {
                kind: SpaceRuleKind::App,
                pattern: name.to_lowercase(),
            }],
            pinned: false,
            auto_detected: true,
            created_at: now_epoch_secs(),
        }
    }

    /// Create a new user-pinned (manually created) space.
    pub fn new_pinned(id: &str, name: &str, icon: &str, color: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            icon: icon.to_string(),
            color: color.to_string(),
            rules: vec![],
            pinned: true,
            auto_detected: false,
            created_at: now_epoch_secs(),
        }
    }
}

// ── ActivityStream ──────────────────────────────────────────────────────

/// An activity stream tracks a continuous period of work within a space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityStream {
    pub id: String,
    pub space_id: String,
    pub name: String,
    pub started_at: u64,
    pub ended_at: Option<u64>,
    pub app_sequence: Vec<String>,
}

impl ActivityStream {
    pub fn new(space_id: &str, name: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            space_id: space_id.to_string(),
            name: name.to_string(),
            started_at: now_epoch_secs(),
            ended_at: None,
            app_sequence: vec![],
        }
    }

    /// Close this stream, recording the end time.
    pub fn close(&mut self) {
        self.ended_at = Some(now_epoch_secs());
    }

    /// Append an app name to the activity sequence (deduplicates consecutive).
    pub fn add_app(&mut self, app_name: &str) {
        if self.app_sequence.last().map(|s| s.as_str()) != Some(app_name) {
            self.app_sequence.push(app_name.to_string());
        }
    }

    /// Duration in seconds (returns 0 for open streams).
    pub fn duration_secs(&self) -> u64 {
        match self.ended_at {
            Some(end) => end.saturating_sub(self.started_at),
            None => 0,
        }
    }
}

// ── SpaceStore ──────────────────────────────────────────────────────────

/// Persistent store for spaces, activity streams, document assignments, and tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceStore {
    pub spaces: Vec<Space>,
    pub activity_streams: Vec<ActivityStream>,
    pub document_spaces: HashMap<String, String>,
    pub document_tags: HashMap<String, BTreeSet<String>>,
    pub tags: BTreeSet<String>,
}

impl Default for SpaceStore {
    fn default() -> Self {
        let mut spaces = Vec::new();

        // Build seed spaces
        for &(id, name, icon, color, patterns) in SEED_SPACES {
            let rules = patterns
                .iter()
                .map(|p| SpaceRule {
                    kind: SpaceRuleKind::App,
                    pattern: p.to_string(),
                })
                .collect();
            spaces.push(Space {
                id: id.to_string(),
                name: name.to_string(),
                icon: icon.to_string(),
                color: color.to_string(),
                rules,
                pinned: true,
                auto_detected: false,
                created_at: 0,
            });
        }

        // Unsorted fallback — always present
        spaces.push(Space {
            id: UNSORTED_SPACE_ID.to_string(),
            name: "Unsorted".to_string(),
            icon: "inbox".to_string(),
            color: "#9ca3af".to_string(),
            rules: vec![],
            pinned: false,
            auto_detected: false,
            created_at: 0,
        });

        Self {
            spaces,
            activity_streams: vec![],
            document_spaces: HashMap::new(),
            document_tags: HashMap::new(),
            tags: BTreeSet::new(),
        }
    }
}

impl SpaceStore {
    // ── Helpers ──────────────────────────────────────────────────────

    /// Build a document key from source and source_id.
    pub fn doc_key(source: &str, source_id: &str) -> String {
        format!("{}::{}", source, source_id)
    }

    // ── Space CRUD ──────────────────────────────────────────────────

    pub fn get_space(&self, id: &str) -> Option<&Space> {
        self.spaces.iter().find(|s| s.id == id)
    }

    pub fn get_space_mut(&mut self, id: &str) -> Option<&mut Space> {
        self.spaces.iter_mut().find(|s| s.id == id)
    }

    pub fn add_space(&mut self, space: Space) {
        if self.get_space(&space.id).is_none() {
            self.spaces.push(space);
        }
    }

    /// Remove a space. Reassigns all its documents to "unsorted".
    /// Returns false if trying to remove "unsorted" (not allowed).
    pub fn remove_space(&mut self, id: &str) -> bool {
        if id == UNSORTED_SPACE_ID {
            return false;
        }
        self.spaces.retain(|s| s.id != id);
        // Reassign documents from the removed space to unsorted
        for space_id in self.document_spaces.values_mut() {
            if *space_id == id {
                *space_id = UNSORTED_SPACE_ID.to_string();
            }
        }
        true
    }

    pub fn rename_space(&mut self, id: &str, new_name: &str) {
        if let Some(space) = self.get_space_mut(id) {
            space.name = new_name.to_string();
        }
    }

    pub fn pin_space(&mut self, id: &str, pinned: bool) {
        if let Some(space) = self.get_space_mut(id) {
            space.pinned = pinned;
        }
    }

    // ── Document space assignment ───────────────────────────────────

    /// Assign a document to a space. Validates that the space exists.
    pub fn set_document_space(&mut self, source: &str, source_id: &str, space_id: &str) -> bool {
        if self.get_space(space_id).is_none() {
            return false;
        }
        let key = Self::doc_key(source, source_id);
        self.document_spaces.insert(key, space_id.to_string());
        true
    }

    pub fn get_document_space(&self, source: &str, source_id: &str) -> Option<&String> {
        let key = Self::doc_key(source, source_id);
        self.document_spaces.get(&key)
    }

    // ── Tags ────────────────────────────────────────────────────────

    /// Set tags for a document. Normalizes to lowercase and adds to the tag library.
    pub fn set_document_tags(
        &mut self,
        source: &str,
        source_id: &str,
        tags: Vec<String>,
    ) -> Vec<String> {
        let key = Self::doc_key(source, source_id);
        let tag_set: BTreeSet<String> = tags
            .into_iter()
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();

        // Add all tags to the library
        for tag in &tag_set {
            self.tags.insert(tag.clone());
        }

        if tag_set.is_empty() {
            self.document_tags.remove(&key);
        } else {
            self.document_tags.insert(key, tag_set.clone());
        }

        tag_set.into_iter().collect()
    }

    pub fn get_document_tags(&self, source: &str, source_id: &str) -> Vec<String> {
        let key = Self::doc_key(source, source_id);
        self.document_tags
            .get(&key)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Delete a tag from the library and all document assignments.
    pub fn delete_tag(&mut self, name: &str) {
        let normalized = name.trim().to_lowercase();
        self.tags.remove(&normalized);
        for tags in self.document_tags.values_mut() {
            tags.remove(&normalized);
        }
        self.document_tags.retain(|_, v| !v.is_empty());
    }

    /// Get all tags in the library as a sorted vector.
    pub fn list_all_tags(&self) -> Vec<String> {
        self.tags.iter().cloned().collect()
    }

    // ── Space resolution ────────────────────────────────────────────

    /// Resolve which space a capture belongs to based on app name, window title, and path.
    /// Matches against space rules, auto-detects unknown apps, falls back to "unsorted".
    pub fn resolve_space(
        &mut self,
        app_name: Option<&str>,
        window_title: Option<&str>,
        path: Option<&str>,
    ) -> String {
        // Try matching against existing space rules
        if let Some(matched) = self.match_rules(app_name, window_title, path) {
            return matched;
        }

        // Auto-detect: create a new space for an unknown app
        if let Some(app) = app_name {
            let app_lower = app.to_lowercase();
            // Check if we already auto-detected this app
            for space in &self.spaces {
                if space.auto_detected {
                    for rule in &space.rules {
                        if rule.kind == SpaceRuleKind::App && app_lower.contains(&rule.pattern) {
                            return space.id.clone();
                        }
                    }
                }
            }
            // Create new auto-detected space
            let id = format!("auto-{}", app_lower.replace(' ', "-"));
            let space = Space::new_auto(&id, app);
            let space_id = space.id.clone();
            self.spaces.push(space);
            log::info!("[spaces] auto-detected new space: {} (app: {})", id, app);
            return space_id;
        }

        UNSORTED_SPACE_ID.to_string()
    }

    /// Try to match against all space rules. Returns the first matching space ID.
    fn match_rules(
        &self,
        app_name: Option<&str>,
        window_title: Option<&str>,
        path: Option<&str>,
    ) -> Option<String> {
        let app_lower = app_name.map(|a| a.to_lowercase());
        let title_lower = window_title.map(|t| t.to_lowercase());
        let path_lower = path.map(|p| p.to_lowercase());

        for space in &self.spaces {
            for rule in &space.rules {
                let matched = match rule.kind {
                    SpaceRuleKind::App => {
                        if let Some(ref app) = app_lower {
                            app.contains(&rule.pattern)
                        } else {
                            false
                        }
                    }
                    SpaceRuleKind::Path => {
                        if let Some(ref p) = path_lower {
                            p.contains(&rule.pattern)
                        } else {
                            false
                        }
                    }
                    SpaceRuleKind::Keyword => {
                        let in_title = title_lower
                            .as_ref()
                            .is_some_and(|t| t.contains(&rule.pattern));
                        let in_path = path_lower
                            .as_ref()
                            .is_some_and(|p| p.contains(&rule.pattern));
                        in_title || in_path
                    }
                    SpaceRuleKind::UrlPattern => title_lower
                        .as_ref()
                        .is_some_and(|t| t.contains(&rule.pattern)),
                };
                if matched {
                    return Some(space.id.clone());
                }
            }
        }
        None
    }

    // ── Activity streams ────────────────────────────────────────────

    /// Get or create an activity stream for a space.
    /// Reuses the current open stream if the user wasn't AFK.
    /// Closes the current stream and creates a new one if AFK.
    pub fn get_or_create_stream(
        &mut self,
        space_id: &str,
        app_name: &str,
        was_afk: bool,
    ) -> &ActivityStream {
        // Find the most recent open stream for this space
        let open_idx = self
            .activity_streams
            .iter()
            .rposition(|s| s.space_id == space_id && s.ended_at.is_none());

        if let Some(idx) = open_idx {
            if was_afk {
                // Close the existing stream and create a new one
                self.activity_streams[idx].close();
                let mut stream = ActivityStream::new(space_id, app_name);
                stream.add_app(app_name);
                self.activity_streams.push(stream);
                let len = self.activity_streams.len();
                return &self.activity_streams[len - 1];
            } else {
                // Reuse existing stream
                self.activity_streams[idx].add_app(app_name);
                return &self.activity_streams[idx];
            }
        }

        // No open stream — create a new one
        let mut stream = ActivityStream::new(space_id, app_name);
        stream.add_app(app_name);
        self.activity_streams.push(stream);
        let len = self.activity_streams.len();
        &self.activity_streams[len - 1]
    }

    /// Prune closed activity streams, keeping only the most recent MAX_CLOSED_STREAMS.
    pub fn prune_streams(&mut self) {
        let closed_count = self
            .activity_streams
            .iter()
            .filter(|s| s.ended_at.is_some())
            .count();
        if closed_count > MAX_CLOSED_STREAMS {
            let to_remove = closed_count - MAX_CLOSED_STREAMS;
            let mut removed = 0;
            self.activity_streams.retain(|s| {
                if s.ended_at.is_some() && removed < to_remove {
                    removed += 1;
                    false
                } else {
                    true
                }
            });
        }
    }

    // ── Cleanup ─────────────────────────────────────────────────────

    /// Remove all data associated with a document.
    pub fn remove_document(&mut self, source: &str, source_id: &str) {
        let key = Self::doc_key(source, source_id);
        self.document_spaces.remove(&key);
        self.document_tags.remove(&key);
    }
}

// ── Persistence functions ───────────────────────────────────────────────

/// Path to the SQLite database file.
fn db_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
        .join("spaces.db")
}

/// Path to the legacy JSON file (for migration).
fn json_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
        .join("spaces.json")
}

// ── SpaceDb ─────────────────────────────────────────────────────────────

/// SQLite-backed persistence layer for the SpaceStore.
struct SpaceDb {
    conn: Connection,
}

impl SpaceDb {
    /// Open (or create) the spaces database at the given path.
    fn open(path: &std::path::Path) -> Result<Self, OriginError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)
            .map_err(|e| OriginError::Generic(format!("spaces db open: {e}")))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .map_err(|e| OriginError::Generic(format!("spaces db pragma: {e}")))?;
        let db = Self { conn };
        db.create_tables()?;
        Ok(db)
    }

    fn create_tables(&self) -> Result<(), OriginError> {
        self.conn
            .execute_batch(
                "
            CREATE TABLE IF NOT EXISTS spaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                icon TEXT NOT NULL DEFAULT '',
                color TEXT NOT NULL DEFAULT 'zinc',
                pinned INTEGER NOT NULL DEFAULT 0,
                auto_detected INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS space_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                pattern TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS activity_streams (
                id TEXT PRIMARY KEY,
                space_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                started_at INTEGER NOT NULL,
                ended_at INTEGER,
                app_sequence TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS document_spaces (
                doc_key TEXT PRIMARY KEY,
                space_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_tags (
                doc_key TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (doc_key, tag)
            );

            CREATE TABLE IF NOT EXISTS tags (
                name TEXT PRIMARY KEY
            );
            ",
            )
            .map_err(|e| OriginError::Generic(format!("spaces db create tables: {e}")))?;
        Ok(())
    }

    /// Write the full SpaceStore to SQLite, replacing all rows.
    fn save_all(&self, store: &SpaceStore) -> Result<(), OriginError> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| OriginError::Generic(format!("spaces db tx begin: {e}")))?;

        // Clear existing data
        tx.execute_batch(
            "DELETE FROM space_rules;
             DELETE FROM spaces;
             DELETE FROM activity_streams;
             DELETE FROM document_spaces;
             DELETE FROM document_tags;
             DELETE FROM tags;",
        )
        .map_err(|e| OriginError::Generic(format!("spaces db clear: {e}")))?;

        // ── Spaces + rules ──────────────────────────────────────────
        for space in &store.spaces {
            tx.execute(
                "INSERT INTO spaces (id, name, icon, color, pinned, auto_detected, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    space.id,
                    space.name,
                    space.icon,
                    space.color,
                    space.pinned as i32,
                    space.auto_detected as i32,
                    space.created_at as i64,
                ],
            )
            .map_err(|e| OriginError::Generic(format!("spaces db insert space: {e}")))?;

            for rule in &space.rules {
                let kind_str = match rule.kind {
                    SpaceRuleKind::App => "app",
                    SpaceRuleKind::Path => "path",
                    SpaceRuleKind::Keyword => "keyword",
                    SpaceRuleKind::UrlPattern => "url_pattern",
                };
                tx.execute(
                    "INSERT INTO space_rules (space_id, kind, pattern) VALUES (?1, ?2, ?3)",
                    params![space.id, kind_str, rule.pattern],
                )
                .map_err(|e| OriginError::Generic(format!("spaces db insert rule: {e}")))?;
            }
        }

        // ── Activity streams ────────────────────────────────────────
        for stream in &store.activity_streams {
            let app_seq_json =
                serde_json::to_string(&stream.app_sequence).unwrap_or_else(|_| "[]".to_string());
            tx.execute(
                "INSERT INTO activity_streams (id, space_id, name, started_at, ended_at, app_sequence)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    stream.id,
                    stream.space_id,
                    stream.name,
                    stream.started_at as i64,
                    stream.ended_at.map(|e| e as i64),
                    app_seq_json,
                ],
            )
            .map_err(|e| OriginError::Generic(format!("spaces db insert stream: {e}")))?;
        }

        // ── Document spaces ─────────────────────────────────────────
        for (doc_key, space_id) in &store.document_spaces {
            tx.execute(
                "INSERT INTO document_spaces (doc_key, space_id) VALUES (?1, ?2)",
                params![doc_key, space_id],
            )
            .map_err(|e| OriginError::Generic(format!("spaces db insert doc_space: {e}")))?;
        }

        // ── Document tags ───────────────────────────────────────────
        for (doc_key, tags) in &store.document_tags {
            for tag in tags {
                tx.execute(
                    "INSERT INTO document_tags (doc_key, tag) VALUES (?1, ?2)",
                    params![doc_key, tag],
                )
                .map_err(|e| OriginError::Generic(format!("spaces db insert doc_tag: {e}")))?;
            }
        }

        // ── Tags library ────────────────────────────────────────────
        for tag in &store.tags {
            tx.execute("INSERT INTO tags (name) VALUES (?1)", params![tag])
                .map_err(|e| OriginError::Generic(format!("spaces db insert tag: {e}")))?;
        }

        tx.commit()
            .map_err(|e| OriginError::Generic(format!("spaces db tx commit: {e}")))?;

        Ok(())
    }

    /// Load the full SpaceStore from SQLite.
    fn load_all(&self) -> Result<SpaceStore, OriginError> {
        // ── Spaces ──────────────────────────────────────────────────
        let mut spaces: Vec<Space> = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, name, icon, color, pinned, auto_detected, created_at FROM spaces",
                )
                .map_err(|e| OriginError::Generic(format!("spaces db select spaces: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok(Space {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        icon: row.get(2)?,
                        color: row.get(3)?,
                        rules: vec![], // filled below
                        pinned: row.get::<_, i32>(4)? != 0,
                        auto_detected: row.get::<_, i32>(5)? != 0,
                        created_at: row.get::<_, i64>(6)? as u64,
                    })
                })
                .map_err(|e| OriginError::Generic(format!("spaces db query spaces: {e}")))?;
            for row in rows {
                spaces.push(row.map_err(|e| OriginError::Generic(format!("spaces db row: {e}")))?);
            }
        }

        // ── Rules ───────────────────────────────────────────────────
        {
            let mut stmt = self
                .conn
                .prepare("SELECT space_id, kind, pattern FROM space_rules")
                .map_err(|e| OriginError::Generic(format!("spaces db select rules: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    let space_id: String = row.get(0)?;
                    let kind_str: String = row.get(1)?;
                    let pattern: String = row.get(2)?;
                    Ok((space_id, kind_str, pattern))
                })
                .map_err(|e| OriginError::Generic(format!("spaces db query rules: {e}")))?;
            for row in rows {
                let (space_id, kind_str, pattern) =
                    row.map_err(|e| OriginError::Generic(format!("spaces db rule row: {e}")))?;
                let kind = match kind_str.as_str() {
                    "app" => SpaceRuleKind::App,
                    "path" => SpaceRuleKind::Path,
                    "keyword" => SpaceRuleKind::Keyword,
                    "url_pattern" => SpaceRuleKind::UrlPattern,
                    _ => continue,
                };
                if let Some(space) = spaces.iter_mut().find(|s| s.id == space_id) {
                    space.rules.push(SpaceRule { kind, pattern });
                }
            }
        }

        // ── Activity streams ────────────────────────────────────────
        let mut activity_streams: Vec<ActivityStream> = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT id, space_id, name, started_at, ended_at, app_sequence
                     FROM activity_streams ORDER BY started_at",
                )
                .map_err(|e| OriginError::Generic(format!("spaces db select streams: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    let app_seq_json: String = row.get(5)?;
                    let app_sequence: Vec<String> =
                        serde_json::from_str(&app_seq_json).unwrap_or_default();
                    Ok(ActivityStream {
                        id: row.get(0)?,
                        space_id: row.get(1)?,
                        name: row.get(2)?,
                        started_at: row.get::<_, i64>(3)? as u64,
                        ended_at: row.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                        app_sequence,
                    })
                })
                .map_err(|e| OriginError::Generic(format!("spaces db query streams: {e}")))?;
            for row in rows {
                activity_streams.push(
                    row.map_err(|e| OriginError::Generic(format!("spaces db stream row: {e}")))?,
                );
            }
        }

        // ── Document spaces ─────────────────────────────────────────
        let mut document_spaces: HashMap<String, String> = HashMap::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT doc_key, space_id FROM document_spaces")
                .map_err(|e| OriginError::Generic(format!("spaces db select doc_spaces: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    let doc_key: String = row.get(0)?;
                    let space_id: String = row.get(1)?;
                    Ok((doc_key, space_id))
                })
                .map_err(|e| OriginError::Generic(format!("spaces db query doc_spaces: {e}")))?;
            for row in rows {
                let (k, v) =
                    row.map_err(|e| OriginError::Generic(format!("spaces db doc_space row: {e}")))?;
                document_spaces.insert(k, v);
            }
        }

        // ── Document tags ───────────────────────────────────────────
        let mut document_tags: HashMap<String, BTreeSet<String>> = HashMap::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT doc_key, tag FROM document_tags")
                .map_err(|e| OriginError::Generic(format!("spaces db select doc_tags: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    let doc_key: String = row.get(0)?;
                    let tag: String = row.get(1)?;
                    Ok((doc_key, tag))
                })
                .map_err(|e| OriginError::Generic(format!("spaces db query doc_tags: {e}")))?;
            for row in rows {
                let (k, t) =
                    row.map_err(|e| OriginError::Generic(format!("spaces db doc_tag row: {e}")))?;
                document_tags.entry(k).or_default().insert(t);
            }
        }

        // ── Tags library ────────────────────────────────────────────
        let mut tags: BTreeSet<String> = BTreeSet::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT name FROM tags")
                .map_err(|e| OriginError::Generic(format!("spaces db select tags: {e}")))?;
            let rows = stmt
                .query_map([], |row| row.get(0))
                .map_err(|e| OriginError::Generic(format!("spaces db query tags: {e}")))?;
            for row in rows {
                tags.insert(
                    row.map_err(|e| OriginError::Generic(format!("spaces db tag row: {e}")))?,
                );
            }
        }

        Ok(SpaceStore {
            spaces,
            activity_streams,
            document_spaces,
            document_tags,
            tags,
        })
    }
}

// ── Public persistence API (unchanged signatures) ───────────────────────

pub fn load_spaces() -> SpaceStore {
    let db_path = db_path();
    let json_path = json_path();

    // Try opening (or creating) the SQLite database
    let db = match SpaceDb::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            log::error!("[spaces] failed to open spaces.db: {e}");
            return SpaceStore::default();
        }
    };

    // If the DB has data, load from it
    match db.load_all() {
        Ok(store) if !store.spaces.is_empty() => return store,
        Ok(_) => { /* empty DB — fall through to migration or defaults */ }
        Err(e) => {
            log::error!("[spaces] failed to load from spaces.db: {e}");
            return SpaceStore::default();
        }
    }

    // Migration: if spaces.json exists, import it into SQLite
    if json_path.exists() {
        if let Ok(contents) = std::fs::read_to_string(&json_path) {
            if let Ok(store) = serde_json::from_str::<SpaceStore>(&contents) {
                log::info!(
                    "[spaces] migrating {} spaces from spaces.json to SQLite",
                    store.spaces.len()
                );
                if let Err(e) = db.save_all(&store) {
                    log::error!("[spaces] migration save failed: {e}");
                } else {
                    // Rename the old file so we don't re-migrate
                    let migrated = json_path.with_extension("json.migrated");
                    let _ = std::fs::rename(&json_path, &migrated);
                }
                return store;
            }
        }
    }

    // Fresh install — seed defaults into SQLite
    let store = SpaceStore::default();
    if let Err(e) = db.save_all(&store) {
        log::error!("[spaces] failed to seed defaults: {e}");
    }
    store
}

pub fn save_spaces(store: &SpaceStore) -> Result<(), OriginError> {
    let db = SpaceDb::open(&db_path())?;
    db.save_all(store)
}

// ── Utilities ───────────────────────────────────────────────────────────

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_store_has_seed_spaces() {
        let store = SpaceStore::default();
        // 5 seed spaces + unsorted = 6
        assert_eq!(store.spaces.len(), 6);
        assert!(store.get_space("code").is_some());
        assert!(store.get_space("communication").is_some());
        assert!(store.get_space("research").is_some());
        assert!(store.get_space("writing").is_some());
        assert!(store.get_space("design").is_some());
        assert!(store.get_space(UNSORTED_SPACE_ID).is_some());
    }

    #[test]
    fn test_resolve_space_matches_app_rules() {
        let mut store = SpaceStore::default();

        // Code editors
        assert_eq!(store.resolve_space(Some("Code"), None, None), "code");
        assert_eq!(
            store.resolve_space(Some("Visual Studio Code"), None, None),
            "code"
        );

        // Communication
        assert_eq!(
            store.resolve_space(Some("Slack"), None, None),
            "communication"
        );

        // Research (browsers)
        assert_eq!(store.resolve_space(Some("Safari"), None, None), "research");

        // Design
        assert_eq!(store.resolve_space(Some("Figma"), None, None), "design");

        // Writing
        assert_eq!(store.resolve_space(Some("Notion"), None, None), "writing");
    }

    #[test]
    fn test_resolve_space_auto_detects_unknown_app() {
        let mut store = SpaceStore::default();
        let initial_count = store.spaces.len();

        let space_id = store.resolve_space(Some("SuperNewApp"), None, None);
        assert_eq!(space_id, "auto-supernewapp");
        assert_eq!(store.spaces.len(), initial_count + 1);

        // Verify the auto-detected space
        let space = store.get_space("auto-supernewapp").unwrap();
        assert!(space.auto_detected);
        assert_eq!(space.name, "SuperNewApp");
    }

    #[test]
    fn test_resolve_space_reuses_auto_detected() {
        let mut store = SpaceStore::default();

        let id1 = store.resolve_space(Some("SuperNewApp"), None, None);
        let count_after_first = store.spaces.len();

        let id2 = store.resolve_space(Some("SuperNewApp"), None, None);
        assert_eq!(id1, id2);
        assert_eq!(store.spaces.len(), count_after_first);
    }

    #[test]
    fn test_remove_space_reassigns_to_unsorted() {
        let mut store = SpaceStore::default();

        // Assign a doc to "code"
        store.set_document_space("screen", "doc1", "code");

        // Remove "code" space
        assert!(store.remove_space("code"));

        // Doc should now be in unsorted
        let space = store.get_document_space("screen", "doc1").unwrap();
        assert_eq!(space, UNSORTED_SPACE_ID);
    }

    #[test]
    fn test_cannot_remove_unsorted() {
        let mut store = SpaceStore::default();
        assert!(!store.remove_space(UNSORTED_SPACE_ID));
        assert!(store.get_space(UNSORTED_SPACE_ID).is_some());
    }

    #[test]
    fn test_activity_stream_lifecycle() {
        let mut store = SpaceStore::default();

        // Create a new stream
        let stream = store.get_or_create_stream("code", "VS Code", false);
        let stream_id = stream.id.clone();
        assert!(stream.ended_at.is_none());
        assert_eq!(stream.app_sequence, vec!["VS Code"]);

        // Reuse the same stream (not AFK)
        let stream = store.get_or_create_stream("code", "Terminal", false);
        assert_eq!(stream.id, stream_id);
        assert_eq!(stream.app_sequence, vec!["VS Code", "Terminal"]);

        // AFK — should close old stream and create new one
        let stream = store.get_or_create_stream("code", "VS Code", true);
        assert_ne!(stream.id, stream_id);
        assert!(stream.ended_at.is_none());
        assert_eq!(stream.app_sequence, vec!["VS Code"]);

        // Old stream should be closed
        let old = store
            .activity_streams
            .iter()
            .find(|s| s.id == stream_id)
            .unwrap();
        assert!(old.ended_at.is_some());
    }

    #[test]
    fn test_document_tags() {
        let mut store = SpaceStore::default();

        // Set tags with mixed case
        let result = store.set_document_tags(
            "screen",
            "doc1",
            vec![
                "Rust".to_string(),
                "  TAURI  ".to_string(),
                "rust".to_string(), // duplicate after normalization
            ],
        );

        assert_eq!(result, vec!["rust", "tauri"]); // BTreeSet: sorted + deduped

        // Tags are in the library
        assert!(store.tags.contains("rust"));
        assert!(store.tags.contains("tauri"));

        // Retrieve document tags
        let tags = store.get_document_tags("screen", "doc1");
        assert_eq!(tags, vec!["rust", "tauri"]);

        // Delete a tag
        store.delete_tag("rust");
        assert!(!store.tags.contains("rust"));
        let tags = store.get_document_tags("screen", "doc1");
        assert_eq!(tags, vec!["tauri"]);
    }

    #[test]
    fn test_pin_space() {
        let mut store = SpaceStore::default();

        // Unsorted is not pinned by default
        let unsorted = store.get_space(UNSORTED_SPACE_ID).unwrap();
        assert!(!unsorted.pinned);

        // Pin it
        store.pin_space(UNSORTED_SPACE_ID, true);
        let unsorted = store.get_space(UNSORTED_SPACE_ID).unwrap();
        assert!(unsorted.pinned);

        // Unpin it
        store.pin_space(UNSORTED_SPACE_ID, false);
        let unsorted = store.get_space(UNSORTED_SPACE_ID).unwrap();
        assert!(!unsorted.pinned);
    }

    #[test]
    fn test_prune_streams() {
        let mut store = SpaceStore::default();

        // Create many closed streams
        for i in 0..(MAX_CLOSED_STREAMS + 50) {
            let mut stream = ActivityStream::new("code", &format!("app-{}", i));
            stream.close();
            store.activity_streams.push(stream);
        }

        assert_eq!(store.activity_streams.len(), MAX_CLOSED_STREAMS + 50);
        store.prune_streams();
        assert_eq!(store.activity_streams.len(), MAX_CLOSED_STREAMS);
    }

    #[test]
    fn test_set_document_space_validates() {
        let mut store = SpaceStore::default();

        // Valid space
        assert!(store.set_document_space("screen", "doc1", "code"));
        assert_eq!(
            store.get_document_space("screen", "doc1"),
            Some(&"code".to_string())
        );

        // Invalid space
        assert!(!store.set_document_space("screen", "doc2", "nonexistent"));
        assert_eq!(store.get_document_space("screen", "doc2"), None);
    }

    #[test]
    fn test_remove_document() {
        let mut store = SpaceStore::default();

        store.set_document_space("screen", "doc1", "code");
        store.set_document_tags("screen", "doc1", vec!["rust".to_string()]);

        // Both should exist
        assert!(store.get_document_space("screen", "doc1").is_some());
        assert!(!store.get_document_tags("screen", "doc1").is_empty());

        // Remove
        store.remove_document("screen", "doc1");
        assert!(store.get_document_space("screen", "doc1").is_none());
        assert!(store.get_document_tags("screen", "doc1").is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let store = SpaceStore::default();
        let json = serde_json::to_string(&store).unwrap();
        let deserialized: SpaceStore = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.spaces.len(), store.spaces.len());
    }

    // ── SQLite persistence tests ────────────────────────────────────

    /// Helper: open an in-memory SpaceDb for tests.
    fn test_db() -> SpaceDb {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .unwrap();
        let db = SpaceDb { conn };
        db.create_tables().unwrap();
        db
    }

    #[test]
    fn test_sqlite_roundtrip_default_store() {
        let db = test_db();
        let original = SpaceStore::default();
        db.save_all(&original).unwrap();
        let loaded = db.load_all().unwrap();

        assert_eq!(loaded.spaces.len(), original.spaces.len());
        for (orig, load) in original.spaces.iter().zip(loaded.spaces.iter()) {
            assert_eq!(orig.id, load.id);
            assert_eq!(orig.name, load.name);
            assert_eq!(orig.icon, load.icon);
            assert_eq!(orig.color, load.color);
            assert_eq!(orig.pinned, load.pinned);
            assert_eq!(orig.auto_detected, load.auto_detected);
            assert_eq!(orig.created_at, load.created_at);
            assert_eq!(orig.rules.len(), load.rules.len());
            for (or, lr) in orig.rules.iter().zip(load.rules.iter()) {
                assert_eq!(or.kind, lr.kind);
                assert_eq!(or.pattern, lr.pattern);
            }
        }
    }

    #[test]
    fn test_sqlite_roundtrip_with_data() {
        let db = test_db();
        let mut store = SpaceStore::default();

        // Add document assignments
        store.set_document_space("screen", "doc1", "code");
        store.set_document_space("file", "doc2", "writing");

        // Add tags
        store.set_document_tags(
            "screen",
            "doc1",
            vec!["rust".to_string(), "tauri".to_string()],
        );

        // Add activity streams
        let mut stream = ActivityStream::new("code", "VS Code");
        stream.add_app("VS Code");
        stream.add_app("Terminal");
        store.activity_streams.push(stream);

        let mut stream2 = ActivityStream::new("writing", "Notion");
        stream2.add_app("Notion");
        stream2.close();
        store.activity_streams.push(stream2);

        db.save_all(&store).unwrap();
        let loaded = db.load_all().unwrap();

        // Document spaces
        assert_eq!(loaded.document_spaces.len(), 2);
        assert_eq!(
            loaded.document_spaces.get("screen::doc1"),
            Some(&"code".to_string())
        );
        assert_eq!(
            loaded.document_spaces.get("file::doc2"),
            Some(&"writing".to_string())
        );

        // Document tags
        assert_eq!(loaded.document_tags.len(), 1);
        let doc1_tags = loaded.document_tags.get("screen::doc1").unwrap();
        assert!(doc1_tags.contains("rust"));
        assert!(doc1_tags.contains("tauri"));

        // Tags library
        assert!(loaded.tags.contains("rust"));
        assert!(loaded.tags.contains("tauri"));

        // Activity streams
        assert_eq!(loaded.activity_streams.len(), 2);
        let s1 = &loaded.activity_streams[0];
        assert_eq!(s1.space_id, "code");
        assert_eq!(s1.app_sequence, vec!["VS Code", "Terminal"]);
        assert!(s1.ended_at.is_none());

        let s2 = &loaded.activity_streams[1];
        assert_eq!(s2.space_id, "writing");
        assert!(s2.ended_at.is_some());
    }

    #[test]
    fn test_sqlite_save_overwrites_previous() {
        let db = test_db();

        // Save defaults
        let store1 = SpaceStore::default();
        db.save_all(&store1).unwrap();

        // Save a modified store with an extra space
        let mut store2 = SpaceStore::default();
        store2.add_space(Space::new_pinned("custom", "Custom", "star", "#ff0000"));
        db.save_all(&store2).unwrap();

        let loaded = db.load_all().unwrap();
        assert_eq!(loaded.spaces.len(), store2.spaces.len());
        assert!(loaded.spaces.iter().any(|s| s.id == "custom"));
    }

    #[test]
    fn test_sqlite_rule_kinds_roundtrip() {
        let db = test_db();
        let mut store = SpaceStore::default();

        // Add a space with all rule kinds
        let space = Space {
            id: "test-rules".to_string(),
            name: "Test Rules".to_string(),
            icon: "test".to_string(),
            color: "#000".to_string(),
            rules: vec![
                SpaceRule {
                    kind: SpaceRuleKind::App,
                    pattern: "myapp".to_string(),
                },
                SpaceRule {
                    kind: SpaceRuleKind::Path,
                    pattern: "/home/user".to_string(),
                },
                SpaceRule {
                    kind: SpaceRuleKind::Keyword,
                    pattern: "important".to_string(),
                },
                SpaceRule {
                    kind: SpaceRuleKind::UrlPattern,
                    pattern: "github.com".to_string(),
                },
            ],
            pinned: true,
            auto_detected: false,
            created_at: 1234567890,
        };
        store.add_space(space);

        db.save_all(&store).unwrap();
        let loaded = db.load_all().unwrap();

        let loaded_space = loaded.spaces.iter().find(|s| s.id == "test-rules").unwrap();
        assert_eq!(loaded_space.rules.len(), 4);
        assert_eq!(loaded_space.rules[0].kind, SpaceRuleKind::App);
        assert_eq!(loaded_space.rules[1].kind, SpaceRuleKind::Path);
        assert_eq!(loaded_space.rules[2].kind, SpaceRuleKind::Keyword);
        assert_eq!(loaded_space.rules[3].kind, SpaceRuleKind::UrlPattern);
    }

    #[test]
    fn test_sqlite_migration_from_json() {
        let dir = std::env::temp_dir().join(format!("origin_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let json_file = dir.join("spaces.json");
        let db_file = dir.join("spaces.db");

        // Write a JSON file with some data
        let mut store = SpaceStore::default();
        store.set_document_space("screen", "cap1", "code");
        store.set_document_tags("screen", "cap1", vec!["test".to_string()]);
        let json = serde_json::to_string_pretty(&store).unwrap();
        std::fs::write(&json_file, &json).unwrap();

        // Open the DB (no existing DB) and migrate
        let db = SpaceDb::open(&db_file).unwrap();
        // Simulate migration: load from JSON, save to SQLite
        let loaded_from_json: SpaceStore =
            serde_json::from_str(&std::fs::read_to_string(&json_file).unwrap()).unwrap();
        db.save_all(&loaded_from_json).unwrap();

        // Verify data survived migration
        let reloaded = db.load_all().unwrap();
        assert_eq!(reloaded.spaces.len(), store.spaces.len());
        assert_eq!(
            reloaded.document_spaces.get("screen::cap1"),
            Some(&"code".to_string())
        );
        assert!(reloaded.tags.contains("test"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
