use super::*;
use crate::db::test_support::TestDbSession;
use crate::db::tests::test_db;
use crate::lint::context::{
    AppliedScope, CancellationToken, ExecutionGate, LintClock, LintContext,
};
use crate::lint::pages::fs::scan_page_root;
#[cfg(unix)]
use crate::lint::runner::LintRunner;
use crate::lint::snapshot::LintReadSnapshot;
use std::fs;
use std::path::Path;
use tempfile::TempDir;
#[cfg(unix)]
use wenlan_types::lint::LintQuery;
use wenlan_types::lint::{
    LintApplicability, LintMetricCode, LintMetricValue, LintOpaqueId, LintOutcome, LintSeverity,
};

#[path = "link_checks_test/artifacts.rs"]
mod artifacts;
#[path = "link_checks_test/manifest.rs"]
mod manifest;
#[path = "link_checks_test/orphans.rs"]
mod orphans;

// M1 honest columns: `workspace` and `space` are NOT NULL and always
// mirrored (migration 80), so this writes one scope value into both --
// `None` seeds the reserved sentinel id rather than a NULL a NOT NULL
// column now rejects.
async fn insert_page(conn: &TestDbSession, id: &str, workspace: Option<&str>, status: &str) {
    let scope = workspace.unwrap_or(crate::db::UNFILED_SPACE_ID);
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, workspace, space, creation_kind, review_status) \
         VALUES (?1, ?1, 'body', '[]', 1, ?4, 'now', 'now', 'now', ?2, ?3, 'distilled', 'confirmed')",
        libsql::params![id, scope, scope, status],
    )
    .await
    .unwrap();
}

async fn link_row_count(conn: &TestDbSession) -> i64 {
    let mut rows = conn
        .query("SELECT COUNT(*) FROM page_links", ())
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

fn metric_value(result: &LintCheckResult, code: LintMetricCode) -> u64 {
    result
        .metrics()
        .iter()
        .find_map(|metric| {
            if metric.code() != code {
                return None;
            }
            match metric.value() {
                LintMetricValue::Count { value } => Some(*value),
                _ => None,
            }
        })
        .unwrap()
}

fn write(root: &Path, relative: &str, bytes: &[u8]) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, bytes).unwrap();
}
