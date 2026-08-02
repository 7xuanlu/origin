// SPDX-License-Identifier: Apache-2.0
//! Uncontested migration-109 homes from the PR-A schema inventory.

use crate::WenlanError;

/// Install the additive J4-J7/J10 substrate in the caller's shared migration
/// transaction.
pub(crate) async fn ensure_remaining_substrate(
    tx: &libsql::Transaction,
) -> Result<(), WenlanError> {
    if !table_has_column(tx, "genesis_coverage_state", "m6_mutation_count").await? {
        tx.execute(
            "ALTER TABLE genesis_coverage_state
             ADD COLUMN m6_mutation_count INTEGER NOT NULL DEFAULT 0
                 CHECK (m6_mutation_count >= 0)",
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m109 mutation counter: {error}")))?;
    }
    if !table_has_column(tx, "genesis_coverage_state", "space_id").await? {
        tx.execute(
            "ALTER TABLE genesis_coverage_state ADD COLUMN space_id TEXT",
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m109 stable space id: {error}")))?;
    }
    tx.execute(
        "UPDATE genesis_coverage_state
            SET space_id = (
                SELECT spaces.id FROM spaces WHERE spaces.name = genesis_coverage_state.space
            )
          WHERE space_id IS NULL",
        (),
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m109 bind stable space ids: {error}")))?;
    if !genesis_space_bindings_are_valid(tx).await? {
        return Err(WenlanError::VectorDb(
            "m109 genesis coverage row does not resolve to its stable space id".to_string(),
        ));
    }

    tx.execute_batch(
        "
        CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_coverage_state_space_id
            ON genesis_coverage_state(space_id);

        CREATE TRIGGER IF NOT EXISTS m6_genesis_mutation_count_monotone
        BEFORE UPDATE OF m6_mutation_count ON genesis_coverage_state
        WHEN NEW.m6_mutation_count < OLD.m6_mutation_count
        BEGIN
            SELECT RAISE(ABORT, 'm6_mutation_count cannot decrease');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_mutation_count_identity_insert
        BEFORE INSERT ON genesis_coverage_state
        WHEN NEW.space IS NULL OR NEW.space_id IS NULL OR
             NOT EXISTS (
                 SELECT 1 FROM spaces
                  WHERE id = NEW.space_id AND name = NEW.space
             ) OR
             EXISTS (
                 SELECT 1 FROM genesis_coverage_state AS old
                  WHERE old.space = NEW.space OR old.space_id = NEW.space_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'genesis mutation proof requires one live stable space identity');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_mutation_count_replace_monotone
        BEFORE INSERT ON genesis_coverage_state
        WHEN EXISTS (
            SELECT 1 FROM genesis_coverage_state AS old
             WHERE old.space = NEW.space
               AND old.m6_mutation_count > NEW.m6_mutation_count
        )
        BEGIN
            SELECT RAISE(ABORT, 'm6_mutation_count cannot decrease');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_mutation_count_identity_update
        BEFORE UPDATE OF space, space_id ON genesis_coverage_state
        WHEN NEW.space IS NULL OR NEW.space_id IS NULL OR
             NEW.space_id <> OLD.space_id OR
             NOT EXISTS (
                 SELECT 1 FROM spaces
                  WHERE id = OLD.space_id AND name = NEW.space
             ) OR
             EXISTS (
                 SELECT 1 FROM genesis_coverage_state AS destination
                  WHERE destination.space = NEW.space
                    AND destination.space_id <> OLD.space_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'genesis mutation proof identity change is unauthorized');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_mutation_count_no_delete
        BEFORE DELETE ON genesis_coverage_state
        BEGIN
            SELECT RAISE(ABORT, 'm6_mutation_count cannot be erased');
        END;

        CREATE TABLE IF NOT EXISTS m6_deploy_attestation (
            attestation_id  TEXT    NOT NULL PRIMARY KEY
                                      CHECK (length(trim(attestation_id)) > 0),
            attested_at     INTEGER NOT NULL,
            signature       TEXT    NOT NULL CHECK (length(trim(signature)) > 0),
            binary_digest   TEXT    NOT NULL CHECK (length(trim(binary_digest)) > 0)
        );

        CREATE TABLE IF NOT EXISTS m6_genesis_provenance (
            page_id         TEXT    NOT NULL PRIMARY KEY,
            content_digest  TEXT    NOT NULL CHECK (length(trim(content_digest)) > 0),
            page_version    INTEGER NOT NULL CHECK (page_version >= 0),
            minted_at       INTEGER NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS m6_genesis_provenance_no_replace
        BEFORE INSERT ON m6_genesis_provenance
        WHEN EXISTS (
            SELECT 1 FROM m6_genesis_provenance AS old
             WHERE old.page_id = NEW.page_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'genesis provenance cannot be replaced');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_provenance_no_update
        BEFORE UPDATE ON m6_genesis_provenance
        BEGIN
            SELECT RAISE(ABORT, 'genesis provenance cannot be updated');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_genesis_provenance_no_delete
        BEFORE DELETE ON m6_genesis_provenance
        BEGIN
            SELECT RAISE(ABORT, 'genesis provenance cannot be deleted');
        END;

        CREATE TABLE IF NOT EXISTS page_projection_outbox (
            page_id         TEXT    NOT NULL,
            page_version    INTEGER NOT NULL CHECK (page_version >= 0),
            state           TEXT    NOT NULL
                                CHECK (state IN ('pending', 'handed_off', 'complete')),
            content_digest  TEXT    NOT NULL CHECK (length(trim(content_digest)) > 0),
            PRIMARY KEY(page_id, page_version)
        );
        CREATE INDEX IF NOT EXISTS idx_page_projection_outbox_state
            ON page_projection_outbox(state, page_id, page_version);

        CREATE TABLE IF NOT EXISTS m6_counters (
            space_id  TEXT    NOT NULL,
            space     TEXT    NOT NULL,
            name      TEXT    NOT NULL,
            value     INTEGER NOT NULL CHECK (value >= 0),
            PRIMARY KEY(space_id, name),
            UNIQUE(space, name)
        );
        CREATE TRIGGER IF NOT EXISTS m6_counters_monotone
        BEFORE UPDATE OF value ON m6_counters
        WHEN NEW.value < OLD.value
        BEGIN
            SELECT RAISE(ABORT, 'M6 counter cannot decrease');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_counters_replace_monotone
        BEFORE INSERT ON m6_counters
        WHEN EXISTS (
            SELECT 1 FROM m6_counters AS old
             WHERE old.space = NEW.space
               AND old.name = NEW.name
               AND old.value > NEW.value
        )
        BEGIN
            SELECT RAISE(ABORT, 'M6 counter cannot decrease');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_counters_identity_insert
        BEFORE INSERT ON m6_counters
        WHEN NOT EXISTS (
                 SELECT 1 FROM spaces
                  WHERE id = NEW.space_id AND name = NEW.space
             ) OR EXISTS (
                 SELECT 1 FROM m6_counters AS old
                  WHERE old.space = NEW.space
                    AND old.name = NEW.name
                    AND old.space_id <> NEW.space_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'M6 counter requires one live stable space identity');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_counters_identity_update
        BEFORE UPDATE OF space_id, space, name ON m6_counters
        WHEN NEW.space_id <> OLD.space_id OR NEW.name <> OLD.name OR
             NOT EXISTS (
                 SELECT 1 FROM spaces
                  WHERE id = OLD.space_id AND name = NEW.space
             ) OR
             EXISTS (
                 SELECT 1 FROM m6_counters AS destination
                  WHERE destination.space = NEW.space
                    AND destination.name = NEW.name
                    AND destination.space_id <> OLD.space_id
             )
        BEGIN
            SELECT RAISE(ABORT, 'M6 counter identity change is unauthorized');
        END;
        CREATE TRIGGER IF NOT EXISTS m6_counters_no_delete
        BEFORE DELETE ON m6_counters
        BEGIN
            SELECT RAISE(ABORT, 'M6 counter cannot be erased');
        END;
        ",
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m109 remaining substrate: {error}")))?;
    Ok(())
}

async fn genesis_space_bindings_are_valid(tx: &libsql::Transaction) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT 1
               FROM genesis_coverage_state AS coverage
              WHERE coverage.space_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM spaces
                     WHERE spaces.id = coverage.space_id
                       AND spaces.name = coverage.space
              )
              LIMIT 1",
            (),
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("m109 validate stable space ids: {error}"))
        })?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m109 read stable space ids: {error}")))?
        .is_none())
}

async fn table_has_column(
    tx: &libsql::Transaction,
    table: &str,
    expected_column: &str,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m109 inspect {table}: {error}")))?;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m109 inspect {table} row: {error}")))?
    {
        let column = row.get::<String>(1).map_err(|error| {
            WenlanError::VectorDb(format!("m109 inspect {table} column: {error}"))
        })?;
        if column == expected_column {
            return Ok(true);
        }
    }
    Ok(false)
}
