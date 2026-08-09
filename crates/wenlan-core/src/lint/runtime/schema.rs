use crate::lint::context::LintContext;
use crate::lint::snapshot::LintReadSnapshot;
use std::collections::BTreeMap;

const TABLES: &[&str] = &[
    "activities",
    "agent_activity",
    "agent_connections",
    "app_metadata",
    "briefing_cache",
    "capture_refs",
    "child_vectors",
    "document_enrichment_queue",
    "document_tags",
    "memories",
    "memories_fts",
    "narrative_cache",
    "pages",
    "pages_fts",
    "profiles",
    "session_snapshots",
    "spaces",
    "summary_nodes",
    "summary_nodes_fts",
];
const SEARCH_OBJECTS: &[&str] = &[
    "child_vectors_vec_idx",
    "idx_pages_embedding",
    "idx_summary_nodes_embedding",
    "memories_fts",
    "memories_fts_delete",
    "memories_fts_insert",
    "memories_fts_update",
    "memories_vec_idx",
    "pages_fts",
    "pages_fts_delete",
    "pages_fts_insert",
    "pages_fts_update",
    "summary_nodes_fts",
    "summary_nodes_fts_delete",
    "summary_nodes_fts_insert",
    "summary_nodes_fts_update",
];

pub(crate) struct SchemaSnapshot {
    user_version: u64,
    missing_tables: Vec<String>,
    invalid_search_objects: Vec<String>,
}

impl SchemaSnapshot {
    pub(super) fn schema_population(&self) -> u64 {
        u64::try_from(TABLES.len() + 1).unwrap_or(u64::MAX)
    }

    pub(super) fn schema_affected(&self) -> u64 {
        u64::try_from(self.missing_tables.len()).unwrap_or(u64::MAX)
            + u64::from(self.user_version != u64::from(crate::db::SCHEMA_VERSION))
    }

    pub(super) fn search_population(&self) -> u64 {
        u64::try_from(SEARCH_OBJECTS.len()).unwrap_or(u64::MAX)
    }

    pub(super) fn search_affected(&self) -> u64 {
        u64::try_from(self.invalid_search_objects.len()).unwrap_or(u64::MAX)
    }

    pub(crate) const fn user_version(&self) -> u64 {
        self.user_version
    }

    pub(crate) fn missing_tables(&self) -> &[String] {
        &self.missing_tables
    }

    pub(crate) fn invalid_search_objects(&self) -> &[String] {
        &self.invalid_search_objects
    }
}

pub(super) async fn load(context: &LintContext<'_, '_>) -> Result<SchemaSnapshot, ()> {
    load_from_snapshot(context.snapshot()).await
}

pub(crate) async fn load_from_snapshot(
    snapshot: &LintReadSnapshot<'_>,
) -> Result<SchemaSnapshot, ()> {
    let user_version = scalar(snapshot, "PRAGMA user_version").await?;
    let objects = objects(snapshot).await?;
    let missing_tables = invalid_names(TABLES, |name| {
        objects.get(*name).is_some_and(|(kind, _)| kind == "table")
    });
    let invalid_search_objects = invalid_names(SEARCH_OBJECTS, |name| {
        objects
            .get(*name)
            .is_some_and(|(kind, sql)| valid_search_object(name, kind, sql))
    });
    Ok(SchemaSnapshot {
        user_version,
        missing_tables,
        invalid_search_objects,
    })
}

async fn objects(
    snapshot: &LintReadSnapshot<'_>,
) -> Result<BTreeMap<String, (String, String)>, ()> {
    let mut rows = snapshot
        .query(
            "SELECT name,type,COALESCE(sql,'') FROM sqlite_schema ORDER BY name",
            libsql::params::Params::None,
        )
        .await
        .map_err(|_| ())?;
    let mut objects = BTreeMap::new();
    while let Some(row) = rows.next().await.map_err(|_| ())? {
        objects.insert(
            row.get::<String>(0).map_err(|_| ())?,
            (
                row.get::<String>(1).map_err(|_| ())?,
                normalize(&row.get::<String>(2).map_err(|_| ())?),
            ),
        );
    }
    Ok(objects)
}

fn valid_search_object(name: &str, kind: &str, sql: &str) -> bool {
    if name.ends_with("_fts") {
        return kind == "table" && valid_fts(name, sql);
    }
    if name.contains("_fts_") {
        return kind == "trigger" && valid_trigger(name, sql);
    }
    kind == "index" && valid_vector_index(name, sql)
}

fn valid_fts(name: &str, sql: &str) -> bool {
    let (target, columns) = match name {
        "memories_fts" => ("memories", "content,title"),
        "pages_fts" => ("pages", "title,summary,content"),
        "summary_nodes_fts" => ("summary_nodes", "title,body"),
        _ => return false,
    };
    sql.contains("createvirtualtable")
        && sql.contains(&format!("{name}usingfts5({columns}"))
        && (sql.contains(&format!("content={target}"))
            || sql.contains(&format!("content='{target}'")))
        && (sql.contains("content_rowid=rowid") || sql.contains("content_rowid='rowid'"))
}

fn valid_trigger(name: &str, sql: &str) -> bool {
    let Some((family, action)) = name.rsplit_once('_') else {
        return false;
    };
    let Some(target) = family.strip_suffix("_fts") else {
        return false;
    };
    let columns = match target {
        "memories" => "content,title",
        "pages" => "title,summary,content",
        "summary_nodes" => "title,body",
        _ => return false,
    };
    let new_values = qualified_values(columns, "new");
    let old_values = qualified_values(columns, "old");
    let insert = format!("insertinto{family}(rowid,{columns})values(new.rowid,{new_values})");
    let delete = format!(
        "insertinto{family}({family},rowid,{columns})values('delete',old.rowid,{old_values})"
    );
    match action {
        "insert" => sql.contains(&format!("afterinserton{target}")) && sql.contains(&insert),
        "delete" => sql.contains(&format!("afterdeleteon{target}")) && sql.contains(&delete),
        "update" => {
            sql.contains(&format!("afterupdateof{columns}on{target}"))
                && sql.contains(&delete)
                && sql.contains(&insert)
        }
        _ => false,
    }
}

fn qualified_values(columns: &str, qualifier: &str) -> String {
    columns
        .split(',')
        .map(|column| format!("{qualifier}.{column}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn valid_vector_index(name: &str, sql: &str) -> bool {
    let target = match name {
        "memories_vec_idx" => "memories",
        "child_vectors_vec_idx" => "child_vectors",
        "idx_pages_embedding" => "pages",
        "idx_summary_nodes_embedding" => "summary_nodes",
        _ => return false,
    };
    sql.contains("createindex")
        && sql.contains(&format!(
            "on{target}(libsql_vector_idx(embedding,'metric=cosine','compress_neighbors=float8','max_neighbors=32'))"
        ))
}

fn normalize(sql: &str) -> String {
    sql.chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn invalid_names(names: &[&str], valid: impl Fn(&&str) -> bool) -> Vec<String> {
    names
        .iter()
        .filter(|name| !valid(name))
        .map(|name| (*name).to_string())
        .collect()
}

async fn scalar(snapshot: &LintReadSnapshot<'_>, sql: &str) -> Result<u64, ()> {
    let mut rows = snapshot
        .query(sql, libsql::params::Params::None)
        .await
        .map_err(|_| ())?;
    let value = rows
        .next()
        .await
        .map_err(|_| ())?
        .ok_or(())?
        .get::<i64>(0)
        .map_err(|_| ())?;
    u64::try_from(value).map_err(|_| ())
}
