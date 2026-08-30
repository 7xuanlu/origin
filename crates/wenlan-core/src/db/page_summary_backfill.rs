// SPDX-License-Identifier: Apache-2.0
//! One-time repair for page summaries written by the old "first bullet
//! anywhere" rule (fixed in #642 to "first prose sentence"). Pages distilled
//! before that fix keep their bullet, which on a page with an Open Questions
//! list is a question standing in for the page's claim. Nothing rewrites an
//! existing page's summary without an LLM pass, so the daemon reconciles
//! them once at start.

use super::MemoryDB;
use crate::error::WenlanError;
use crate::synthesis::distill::extract_page_summary;

impl MemoryDB {
    /// Set every distilled, machine-owned page's summary to the first prose
    /// sentence of its body when the stored summary differs. Pages a human
    /// edited (`user_edited = 1`) and pages of other kinds (authored, entity,
    /// source documents) are left alone: their summaries are not derived
    /// from the body by this rule. Idempotent; returns the number of pages
    /// changed. `last_modified` is untouched so the pass does not surface as
    /// recent activity.
    pub async fn backfill_page_summaries(&self) -> Result<usize, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, summary, content FROM pages
                  WHERE COALESCE(creation_kind, 'distilled') = 'distilled'
                    AND COALESCE(user_edited, 0) = 0",
                (),
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("summary backfill select: {error}")))?;
        let mut updates: Vec<(String, Option<String>)> = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("summary backfill row: {error}")))?
        {
            let id: String = row
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("summary backfill id: {error}")))?;
            let stored: Option<String> = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("summary backfill summary: {error}"))
            })?;
            let content: String = row.get(2).map_err(|error| {
                WenlanError::VectorDb(format!("summary backfill content: {error}"))
            })?;
            let derived = extract_page_summary(&content);
            if derived != stored {
                updates.push((id, derived));
            }
        }
        drop(rows);
        for (id, summary) in &updates {
            conn.execute(
                "UPDATE pages SET summary = ?1 WHERE id = ?2",
                libsql::params![summary.clone(), id.clone()],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("summary backfill update {id}: {error}"))
            })?;
        }
        Ok(updates.len())
    }
}

#[cfg(test)]
mod tests {
    use crate::db::tests::test_db;

    const BODY: &str = "Tally stores all data in one SQLite file next to the app [2][6]. \
The app process is the only writer [3].\n\n## Open Questions\n\
- Is there evidence that iCloud backups have ever failed?\n";

    #[tokio::test]
    async fn rewrites_a_first_bullet_summary_to_the_first_sentence() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "page_old_rule",
            "Tally",
            Some("Is there evidence that iCloud backups have ever failed?"),
            BODY,
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();

        assert_eq!(db.backfill_page_summaries().await.unwrap(), 1);
        let page = db.get_page("page_old_rule").await.unwrap().unwrap();
        assert_eq!(
            page.summary.as_deref(),
            Some("Tally stores all data in one SQLite file next to the app.")
        );

        // Idempotent: a second pass finds nothing to change.
        assert_eq!(db.backfill_page_summaries().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn leaves_human_edited_and_non_distilled_pages_alone() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        for id in ["page_human", "page_authored"] {
            db.insert_page(
                id,
                "Tally",
                Some("A summary someone chose."),
                BODY,
                None,
                None,
                &[],
                &now,
            )
            .await
            .unwrap();
        }
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE pages SET user_edited = 1 WHERE id = 'page_human'",
                (),
            )
            .await
            .unwrap();
            conn.execute(
                "UPDATE pages SET creation_kind = 'authored' WHERE id = 'page_authored'",
                (),
            )
            .await
            .unwrap();
        }

        assert_eq!(db.backfill_page_summaries().await.unwrap(), 0);
        for id in ["page_human", "page_authored"] {
            let page = db.get_page(id).await.unwrap().unwrap();
            assert_eq!(page.summary.as_deref(), Some("A summary someone chose."));
        }
    }
}
