// SPDX-License-Identifier: Apache-2.0

use super::WenlanError;

/// M3 stage F: the named adapter seam for `entity_id <-> page_id` translation
/// against the 1:1 `entity_page_map` (migration 90, both columns UNIQUE, FKs
/// ON DELETE CASCADE). Single-row lookups only -- the JOIN forms that hydrate
/// a full `Entity`/`EntityDetail` shape stay in `scoped_entities.rs`, which
/// this seam does not touch.
pub(crate) async fn page_id_for_entity(
    conn: &libsql::Connection,
    entity_id: &str,
) -> Result<Option<String>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT page_id FROM entity_page_map WHERE entity_id = ?1",
            libsql::params![entity_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("page_id_for_entity: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("page_id_for_entity row: {e}")))?
    {
        Some(row) => Ok(Some(row.get(0).map_err(|e| {
            WenlanError::VectorDb(format!("page_id_for_entity col: {e}"))
        })?)),
        None => Ok(None),
    }
}

/// The reverse direction of [`page_id_for_entity`].
#[allow(dead_code)] // no production call site among this stage's candidate
                    // refactor sites (all are forward-direction); proven by
                    // the round-trip bijection test.
pub(crate) async fn entity_id_for_page(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<Option<String>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT entity_id FROM entity_page_map WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("entity_id_for_page: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("entity_id_for_page row: {e}")))?
    {
        Some(row) => Ok(Some(row.get(0).map_err(|e| {
            WenlanError::VectorDb(format!("entity_id_for_page col: {e}"))
        })?)),
        None => Ok(None),
    }
}
