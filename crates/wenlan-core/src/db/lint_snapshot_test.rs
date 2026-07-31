// SPDX-License-Identifier: Apache-2.0

const PINNED_FORWARDING: &str =
    "LintReadSnapshot::open_with_freshness(&self._db, Arc::clone(&self.lint_freshness)).await";
const UNPINNED_FORWARDING: &str =
    "LintReadSnapshot::open_unpinned(&self._db, Arc::clone(&self.lint_freshness), observer).await";

fn has_exact_forwarders(source: &str) -> bool {
    source.matches(PINNED_FORWARDING).count() == 1
        && source.matches(UNPINNED_FORWARDING).count() == 1
}

#[test]
fn r4_25a_lint_snapshot_entrypoints_are_db_owned() {
    let lint_snapshot = include_str!("../lint/snapshot.rs");
    let db_snapshot = include_str!("lint_snapshot.rs");
    let db = include_str!("../db.rs");

    assert_eq!(
        lint_snapshot.matches("self._db").count(),
        0,
        "production MemoryDB snapshot entrypoints still access _db outside db/**"
    );
    assert!(
        !lint_snapshot.contains("impl MemoryDB {"),
        "production MemoryDB snapshot entrypoints still live outside db/**"
    );
    assert_eq!(db.matches("mod lint_snapshot;").count(), 1);
    assert_eq!(db_snapshot.matches("impl MemoryDB {").count(), 1);
    assert_eq!(
        db_snapshot
            .matches("pub async fn open_lint_snapshot(&self) -> Result<LintReadSnapshot<'_>, SnapshotError>")
            .count(),
        1
    );
    assert_eq!(
        db_snapshot
            .matches("pub(crate) async fn open_unpinned_lint_snapshot(")
            .count(),
        1
    );
    assert_eq!(db_snapshot.matches("self._db").count(), 2);
    assert_eq!(
        db_snapshot
            .matches("Arc::clone(&self.lint_freshness)")
            .count(),
        2
    );
    assert!(
        has_exact_forwarders(db_snapshot),
        "DB-owned snapshot entrypoints no longer preserve both exact forwarding expressions"
    );

    let receiver_mutant = db_snapshot.replacen(
        UNPINNED_FORWARDING,
        "LintReadSnapshot::open_unpinned(&other._db, Arc::clone(&self.lint_freshness), observer).await",
        1,
    );
    assert_ne!(receiver_mutant, db_snapshot);
    assert!(
        !has_exact_forwarders(&receiver_mutant),
        "source tooth accepted a changed unpinned DB receiver"
    );

    let observer_mutant = db_snapshot.replacen(
        UNPINNED_FORWARDING,
        "LintReadSnapshot::open_unpinned(&self._db, Arc::clone(&self.lint_freshness), replacement_observer).await",
        1,
    );
    assert_ne!(observer_mutant, db_snapshot);
    assert!(
        !has_exact_forwarders(&observer_mutant),
        "source tooth accepted a changed unpinned observer"
    );
}
