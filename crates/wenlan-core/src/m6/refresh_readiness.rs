// SPDX-License-Identifier: Apache-2.0
//! M6 refresh-job, dependency-snapshot, and readiness substrate.
//!
//! This module deliberately owns no transaction boundary. Migration 109 calls
//! [`ensure_refresh_readiness_tables`] from its shared transaction, refresh
//! finalization calls [`replace_refresh_dependencies`] from the existing outer
//! page transaction, and readiness checks/transitions use that same pattern.

/// Install the additive D4, D7, and D8 schema inside migration 109's shared
/// transaction.
pub async fn ensure_refresh_readiness_tables(
    tx: &libsql::Transaction,
) -> Result<(), libsql::Error> {
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS genesis_refresh_jobs (
             job_id                 INTEGER PRIMARY KEY,
             page_id                TEXT    NOT NULL,
             base_page_version      INTEGER NOT NULL CHECK(base_page_version >= 0),
             base_source_revision   INTEGER NOT NULL CHECK(base_source_revision >= 0),
             space                  TEXT    NOT NULL,
             dependency_generation  INTEGER NOT NULL CHECK(dependency_generation >= 0),
             active_root_digest     TEXT    NOT NULL,
             -- `queued` is the stale-page selection and `generated` is an
             -- in-memory value. Neither may become a durable state.
             state                  TEXT    NOT NULL CHECK(state IN (
                                        'leased', 'retry', 'finalized', 'revision_card')),
             lease_token            TEXT,
             lease_expires_at       INTEGER,
             attempt                INTEGER NOT NULL DEFAULT 0 CHECK(attempt >= 0),
             next_attempt_at        INTEGER,
             readiness_epoch        INTEGER NOT NULL CHECK(readiness_epoch >= 0),
             schema_version         INTEGER NOT NULL CHECK(schema_version > 0),
             reason                 TEXT    NOT NULL,
             created_at             INTEGER NOT NULL,
             updated_at             INTEGER NOT NULL
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_refresh_jobs_active
             ON genesis_refresh_jobs(page_id, base_page_version)
             WHERE state IN ('leased','retry');
         CREATE INDEX IF NOT EXISTS idx_genesis_refresh_jobs_space_compatibility
             ON genesis_refresh_jobs(space, readiness_epoch, schema_version, state);

         -- This is the CURRENT exact dependency snapshot for a page. A
         -- finalizer replaces every row for that page in its outer transaction.
         -- root_id intentionally has no FK: deleting a live provenance root
         -- must leave the missing identity observable to the readiness anti-join.
         CREATE TABLE IF NOT EXISTS m6_refresh_dependencies (
             space              TEXT    NOT NULL,
             page_id            TEXT    NOT NULL,
             page_version       INTEGER NOT NULL CHECK(page_version >= 0),
             claim_revision_id  TEXT    NOT NULL,
             root_id            TEXT    NOT NULL,
             created_at         INTEGER NOT NULL DEFAULT (unixepoch()),
             PRIMARY KEY(space, page_id, page_version, claim_revision_id, root_id)
         );
         CREATE INDEX IF NOT EXISTS idx_m6_refresh_dependencies_space_root
             ON m6_refresh_dependencies(space, root_id);

         -- `stage`, the epoch/phase fence, and operational disposition are
         -- independent axes. Composite diagram labels are presentation only.
         CREATE TABLE IF NOT EXISTS m6_readiness (
             space             TEXT    NOT NULL,
             stage             TEXT    NOT NULL CHECK(stage IN ('A','B','C','D','E')),
             signal            TEXT    NOT NULL,
             epoch             INTEGER NOT NULL CHECK(epoch >= 0),
             phase             TEXT    NOT NULL CHECK(phase IN ('off','preparing','committed')),
             operation_state   TEXT    NOT NULL DEFAULT 'active'
                                      CHECK(operation_state IN (
                                          'active','disabled','forward_fix')),
             operation_reason  TEXT,
             updated_at        INTEGER NOT NULL,
             PRIMARY KEY(space, stage, signal),
             CHECK(
                 (stage IN ('A','B','C','D') AND signal = '-') OR
                 (stage = 'E' AND signal IN (
                     'evidence-cluster', 'orphan-wikilink',
                     'community-overview', 'space-overview'))
             )
         );

         -- Soak is evidence about one readiness epoch, never a phase or stage.
         -- A passing row mechanically carries all three evidence components
         -- and the ratified 72h/20-turn/one-start minimum.
         CREATE TABLE IF NOT EXISTS m6_readiness_soak_receipts (
             space                   TEXT    NOT NULL,
             stage                   TEXT    NOT NULL CHECK(stage IN ('A','B','C','D','E')),
             signal                  TEXT    NOT NULL,
             readiness_epoch         INTEGER NOT NULL CHECK(readiness_epoch >= 0),
             window_started_at       INTEGER NOT NULL,
             window_ended_at         INTEGER NOT NULL,
             observed_turns          INTEGER NOT NULL CHECK(observed_turns >= 0),
             daemon_starts           INTEGER NOT NULL CHECK(daemon_starts >= 0),
             old_writer_mutations    INTEGER NOT NULL CHECK(old_writer_mutations >= 0),
             work_bound_violations   INTEGER NOT NULL CHECK(work_bound_violations >= 0),
             support_regressions     INTEGER NOT NULL CHECK(support_regressions >= 0),
             recorded_at             INTEGER NOT NULL,
             PRIMARY KEY(space, stage, signal, readiness_epoch),
             CHECK(
                 (stage IN ('A','B','C','D') AND signal = '-') OR
                 (stage = 'E' AND signal IN (
                     'evidence-cluster', 'orphan-wikilink',
                     'community-overview', 'space-overview'))
             ),
             CHECK(window_ended_at - window_started_at >= 259200),
             CHECK(observed_turns >= 20),
             CHECK(daemon_starts >= 1),
             CHECK(old_writer_mutations = 0),
             CHECK(work_bound_violations = 0),
             CHECK(support_regressions = 0)
         );
         CREATE TRIGGER IF NOT EXISTS m6_soak_receipt_fence_guard
         BEFORE INSERT ON m6_readiness_soak_receipts
         WHEN NOT EXISTS (
             SELECT 1 FROM m6_readiness r
              WHERE r.space = NEW.space
                AND r.stage = NEW.stage
                AND r.signal = NEW.signal
                AND r.epoch = NEW.readiness_epoch
         )
         BEGIN
             SELECT RAISE(ABORT, 'm6 soak receipt does not name the current readiness fence');
         END;",
    )
    .await
    .map(|_| ())
}

/// Everything the stale-page sweep has already selected and the lease
/// transaction must durably capture before any model work starts.
#[derive(Clone, Copy, Debug)]
pub struct RefreshLeaseRequest<'a> {
    pub page_id: &'a str,
    pub base_page_version: i64,
    pub base_source_revision: i64,
    pub space: &'a str,
    pub dependency_generation: i64,
    pub active_root_digest: &'a str,
    pub lease_token: &'a str,
    pub lease_expires_at: i64,
    pub readiness_epoch: i64,
    pub schema_version: i64,
    pub reason: &'a str,
    pub now: i64,
}

/// Atomically insert a new leased job, or claim a due retry, under the one
/// partial unique constraint. `true` is the caller's permission to begin model
/// work; `false` means another sweep owns this page/base-version.
pub async fn acquire_refresh_lease(
    tx: &libsql::Transaction,
    request: &RefreshLeaseRequest<'_>,
) -> Result<bool, libsql::Error> {
    let inserted = tx
        .execute(
            "INSERT OR IGNORE INTO genesis_refresh_jobs (
                 page_id, base_page_version, base_source_revision, space,
                 dependency_generation, active_root_digest, state, lease_token,
                 lease_expires_at, attempt, readiness_epoch, schema_version, reason,
                 created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'leased', ?7, ?8, 0, ?9, ?10, ?11,
                       ?12, ?12)",
            libsql::params![
                request.page_id,
                request.base_page_version,
                request.base_source_revision,
                request.space,
                request.dependency_generation,
                request.active_root_digest,
                request.lease_token,
                request.lease_expires_at,
                request.readiness_epoch,
                request.schema_version,
                request.reason,
                request.now,
            ],
        )
        .await?;
    if inserted == 1 {
        return Ok(true);
    }

    let claimed_retry = tx
        .execute(
            "UPDATE genesis_refresh_jobs
                SET base_source_revision = ?3,
                    space = ?4,
                    dependency_generation = ?5,
                    active_root_digest = ?6,
                    state = 'leased',
                    lease_token = ?7,
                    lease_expires_at = ?8,
                    next_attempt_at = NULL,
                    readiness_epoch = ?9,
                    schema_version = ?10,
                    reason = ?11,
                    updated_at = ?12
              WHERE page_id = ?1
                AND base_page_version = ?2
                AND state = 'retry'
                AND COALESCE(next_attempt_at, 0) <= ?12",
            libsql::params![
                request.page_id,
                request.base_page_version,
                request.base_source_revision,
                request.space,
                request.dependency_generation,
                request.active_root_digest,
                request.lease_token,
                request.lease_expires_at,
                request.readiness_epoch,
                request.schema_version,
                request.reason,
                request.now,
            ],
        )
        .await?;
    Ok(claimed_retry == 1)
}

/// One claim/root pair in the exact dependency snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RefreshDependency<'a> {
    pub claim_revision_id: &'a str,
    pub root_id: &'a str,
}

/// Replace one page's current dependency snapshot inside the caller's outer
/// genesis/refresh finalize transaction. This function never commits.
pub async fn replace_refresh_dependencies(
    tx: &libsql::Transaction,
    space: &str,
    page_id: &str,
    page_version: i64,
    dependencies: &[RefreshDependency<'_>],
) -> Result<(), libsql::Error> {
    tx.execute(
        "DELETE FROM m6_refresh_dependencies WHERE space = ?1 AND page_id = ?2",
        libsql::params![space, page_id],
    )
    .await?;
    for dependency in dependencies {
        tx.execute(
            "INSERT INTO m6_refresh_dependencies (
                 space, page_id, page_version, claim_revision_id, root_id
             ) VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                space,
                page_id,
                page_version,
                dependency.claim_revision_id,
                dependency.root_id,
            ],
        )
        .await?;
    }
    Ok(())
}

/// PR-D precondition 5. The anti-join deliberately treats both a missing root
/// and any non-active root as incompatible.
pub async fn space_refresh_dependencies_ready(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<bool, libsql::Error> {
    let mut rows = tx
        .query(
            "SELECT NOT EXISTS (
                 SELECT 1
                   FROM m6_refresh_dependencies d
                   LEFT JOIN provenance_roots r ON r.root_id = d.root_id
                  WHERE d.space = ?1
                    AND (r.root_id IS NULL OR r.status <> 'active')
             )",
            libsql::params![space],
        )
        .await?;
    let row = rows
        .next()
        .await?
        .ok_or(libsql::Error::QueryReturnedNoRows)?;
    Ok(row.get::<i64>(0)? != 0)
}

/// Durable readiness identity. The database CHECK is the final authority over
/// legal stage/signal combinations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadinessKey<'a> {
    pub space: &'a str,
    pub stage: &'a str,
    pub signal: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadinessPhase {
    Off,
    Preparing,
    Committed,
}

impl ReadinessPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Preparing => "preparing",
            Self::Committed => "committed",
        }
    }

    fn parse(value: &str) -> Result<Self, libsql::Error> {
        match value {
            "off" => Ok(Self::Off),
            "preparing" => Ok(Self::Preparing),
            "committed" => Ok(Self::Committed),
            other => Err(libsql::Error::Misuse(format!(
                "unreadable M6 readiness phase: {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReadinessFence {
    pub epoch: i64,
    pub phase: ReadinessPhase,
}

/// Create the initial `(epoch=0, phase=off)` fence. Returns false when the key
/// already exists, including D/`-` duplicate initialization.
pub async fn initialize_readiness(
    tx: &libsql::Transaction,
    key: ReadinessKey<'_>,
    now: i64,
) -> Result<bool, libsql::Error> {
    Ok(tx
        .execute(
            "INSERT OR IGNORE INTO m6_readiness (
                 space, stage, signal, epoch, phase, operation_state, updated_at
             ) VALUES (?1, ?2, ?3, 0, 'off', 'active', ?4)",
            libsql::params![key.space, key.stage, key.signal, now],
        )
        .await?
        == 1)
}

/// Read and parse the durable fence. An unparseable row is an error, never an
/// inert/default-permitting value.
pub async fn readiness_fence(
    tx: &libsql::Transaction,
    key: ReadinessKey<'_>,
) -> Result<Option<ReadinessFence>, libsql::Error> {
    let mut rows = tx
        .query(
            "SELECT epoch, phase FROM m6_readiness
              WHERE space = ?1 AND stage = ?2 AND signal = ?3",
            libsql::params![key.space, key.stage, key.signal],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    Ok(Some(ReadinessFence {
        epoch: row.get(0)?,
        phase: ReadinessPhase::parse(&row.get::<String>(1)?)?,
    }))
}

fn transition_is_legal(from: ReadinessPhase, to: ReadinessPhase) -> bool {
    matches!(
        (from, to),
        (ReadinessPhase::Off, ReadinessPhase::Preparing)
            | (ReadinessPhase::Preparing, ReadinessPhase::Off)
            | (ReadinessPhase::Preparing, ReadinessPhase::Committed)
    )
}

/// Compare-and-transition the independent epoch/phase fence. Every successful
/// transition increments the epoch, including a preparing->off abort.
pub async fn transition_readiness(
    tx: &libsql::Transaction,
    key: ReadinessKey<'_>,
    expected: ReadinessFence,
    next_phase: ReadinessPhase,
    now: i64,
) -> Result<Option<ReadinessFence>, libsql::Error> {
    if !transition_is_legal(expected.phase, next_phase) {
        return Err(libsql::Error::Misuse(format!(
            "illegal M6 readiness transition: {} -> {}",
            expected.phase.as_str(),
            next_phase.as_str()
        )));
    }
    let next_epoch = expected
        .epoch
        .checked_add(1)
        .ok_or_else(|| libsql::Error::Misuse("M6 readiness epoch overflow".to_string()))?;
    let changed = tx
        .execute(
            "UPDATE m6_readiness
                SET epoch = ?4, phase = ?5, updated_at = ?6
              WHERE space = ?1 AND stage = ?2 AND signal = ?3
                AND epoch = ?7 AND phase = ?8",
            libsql::params![
                key.space,
                key.stage,
                key.signal,
                next_epoch,
                next_phase.as_str(),
                now,
                expected.epoch,
                expected.phase.as_str(),
            ],
        )
        .await?;
    Ok((changed == 1).then_some(ReadinessFence {
        epoch: next_epoch,
        phase: next_phase,
    }))
}

/// The three-component soak proof plus its ratified activity minimum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SoakEvidence {
    pub window_started_at: i64,
    pub window_ended_at: i64,
    pub observed_turns: i64,
    pub daemon_starts: i64,
    pub old_writer_mutations: i64,
    pub work_bound_violations: i64,
    pub support_regressions: i64,
    pub recorded_at: i64,
}

/// Record a passing soak as separate durable evidence. Weak evidence is
/// rejected before the insert and again by the table CHECKs.
pub async fn record_soak_receipt(
    tx: &libsql::Transaction,
    key: ReadinessKey<'_>,
    readiness_epoch: i64,
    evidence: &SoakEvidence,
) -> Result<(), libsql::Error> {
    let duration = evidence
        .window_ended_at
        .checked_sub(evidence.window_started_at)
        .ok_or_else(|| libsql::Error::Misuse("M6 soak window overflow".to_string()))?;
    if duration < 259_200
        || evidence.observed_turns < 20
        || evidence.daemon_starts < 1
        || evidence.old_writer_mutations != 0
        || evidence.work_bound_violations != 0
        || evidence.support_regressions != 0
    {
        return Err(libsql::Error::Misuse(
            "M6 soak evidence does not satisfy the passing contract".to_string(),
        ));
    }
    tx.execute(
        "INSERT INTO m6_readiness_soak_receipts (
             space, stage, signal, readiness_epoch,
             window_started_at, window_ended_at, observed_turns, daemon_starts,
             old_writer_mutations, work_bound_violations, support_regressions,
             recorded_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        libsql::params![
            key.space,
            key.stage,
            key.signal,
            readiness_epoch,
            evidence.window_started_at,
            evidence.window_ended_at,
            evidence.observed_turns,
            evidence.daemon_starts,
            evidence.old_writer_mutations,
            evidence.work_bound_violations,
            evidence.support_regressions,
            evidence.recorded_at,
        ],
    )
    .await?;
    Ok(())
}
