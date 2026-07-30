// SPDX-License-Identifier: Apache-2.0

const ATOMIC_DECLARATION: &str = "pub(crate) async fn record_repair_verification_atomic(";
const RESULT_MATCH: &str = "    let receipt = match result {";
const ATOMIC_END: &str = "\n    Ok(receipt)\n}";
const RENAME_PERSISTENCE_BRANCH: &str =
    "        if let Some(session) = rename_projection_session {";
const PAGE_ROOT_PERSISTENCE_BRANCH: &str = "        } else if let Some(page_root) = page_root {";
const NO_PAGE_ROOT_PERSISTENCE_BRANCH: &str = "        } else {";
const PERSISTENCE_TAIL_END: &str = "\n        }\n    }\n    .await;";

fn bounded_atomic_source(child: &str) -> (&str, &str, &str) {
    let start = child
        .find(ATOMIC_DECLARATION)
        .expect("atomic operation declaration");
    let result_match = child[start..]
        .find(RESULT_MATCH)
        .map(|offset| start + offset)
        .expect("atomic result match");
    let end = child[result_match..]
        .find(ATOMIC_END)
        .map(|offset| result_match + offset + ATOMIC_END.len())
        .expect("atomic operation end");
    (
        &child[start..end],
        &child[start..result_match],
        &child[result_match..end],
    )
}

fn persistence_segments(precommit: &str) -> (&str, &str, &str) {
    let tail_start = precommit
        .rfind(RENAME_PERSISTENCE_BRANCH)
        .expect("rename persistence branch");
    let tail = &precommit[tail_start..];
    let page_root_start = tail
        .find(PAGE_ROOT_PERSISTENCE_BRANCH)
        .expect("page-root persistence branch");
    let no_page_root_start = tail[page_root_start..]
        .find(NO_PAGE_ROOT_PERSISTENCE_BRANCH)
        .map(|offset| page_root_start + offset)
        .expect("no-page-root persistence branch");
    let tail_end = tail[no_page_root_start..]
        .find(PERSISTENCE_TAIL_END)
        .map(|offset| no_page_root_start + offset + PERSISTENCE_TAIL_END.len())
        .expect("persistence tail end");
    (
        &tail[..page_root_start],
        &tail[page_root_start..no_page_root_start],
        &tail[no_page_root_start..tail_end],
    )
}

fn page_root_persistence_is_inside_projection_lock(segment: &str) -> bool {
    let Some(lock) = segment.find("KnowledgeProjectionWrite::with_projection_lock(") else {
        return false;
    };
    let Some(validation) = segment.find("validate_current_page_receipts_locked(") else {
        return false;
    };
    let Some(persist) = segment.find("store.persist_verification_receipt(&receipt)?;") else {
        return false;
    };
    let Some(closure_end) = segment.rfind("\n            })") else {
        return false;
    };
    segment
        .matches("KnowledgeProjectionWrite::with_projection_lock(")
        .count()
        == 1
        && segment
            .matches("store.persist_verification_receipt(&receipt)?;")
            .count()
            == 1
        && segment
            .matches("validate_current_page_receipts_locked(")
            .count()
            == 1
        && lock < validation
        && validation < persist
        && persist < closure_end
        && !segment.contains("drop(")
}

fn rename_persistence_follows_locked_validation(segment: &str) -> bool {
    let Some(validation_start) =
        segment.find("validate_current_page_receipts_on_repair_projection(")
    else {
        return false;
    };
    let Some(lock) = segment.find("session.locked()") else {
        return false;
    };
    let Some(validation_end) = segment[lock..]
        .find("\n            )?;")
        .map(|offset| lock + offset)
    else {
        return false;
    };
    let Some(persist) = segment.find("store.persist_verification_receipt(&receipt)?;") else {
        return false;
    };
    segment
        .matches("validate_current_page_receipts_on_repair_projection(")
        .count()
        == 1
        && segment.matches("session.locked()").count() == 1
        && segment
            .matches("store.persist_verification_receipt(&receipt)?;")
            .count()
            == 1
        && validation_start < lock
        && lock < validation_end
        && validation_end < persist
        && !segment.contains("drop(")
}

fn move_persist_before_validation(
    segment: &str,
    persist_line: &str,
    validation_line: &str,
) -> String {
    assert_eq!(segment.matches(persist_line).count(), 1);
    assert_eq!(segment.matches(validation_line).count(), 1);
    let without_persist = segment.replacen(persist_line, "", 1);
    without_persist.replacen(
        validation_line,
        &format!("{persist_line}{validation_line}"),
        1,
    )
}

#[test]
fn r4_24a_source_contract_is_atomic_bounded_and_caller_ordered() {
    let child = include_str!("repair_verification.rs");
    let repair = include_str!("../repair.rs");
    let db = include_str!("../db.rs");
    let (atomic, precommit, finalization) = bounded_atomic_source(child);

    assert_eq!(db.matches("pub(crate) mod repair_verification;").count(), 1);
    assert_eq!(child.matches(ATOMIC_DECLARATION).count(), 1);
    assert_eq!(
        repair.matches("record_repair_verification_atomic(").count(),
        1
    );

    for field in [
        "store:",
        "manifest:",
        "apply_receipt:",
        "request:",
        "prior_verified_tag_targets:",
        "rollback:",
        "rename_rollback:",
        "page_root:",
        "verified_at:",
        "rename_projection_session:",
    ] {
        assert!(
            child.contains(field),
            "verification input lost required field {field}"
        );
    }
    for forbidden in [
        "load_manifest(",
        "load_apply_receipt(",
        "load_rollback(",
        "load_rename_page_title_rollback(",
        "lock_manifest_operation(",
        "lock_tag_record_set(",
        "clear_pending_apply_receipt(",
        "libsql::Connection",
        "MutexGuard",
        "TransactionBehavior",
        "transaction_with_behavior",
        "after_projection_session",
        "FnOnce",
        "pending_path",
    ] {
        assert!(
            !atomic.contains(forbidden),
            "verification child gained forbidden capability {forbidden}"
        );
    }

    let ordered = |segment: &str, labels: &[&str]| {
        let mut previous = 0;
        for label in labels {
            let position = segment
                .find(label)
                .unwrap_or_else(|| panic!("lost ordered verification primitive {label}"));
            assert!(
                position >= previous,
                "verification primitive reordered: {label} at {position} before {previous}"
            );
            previous = position;
        }
    };
    assert_eq!(precommit.matches(".conn.lock().await").count(), 1);
    assert_eq!(precommit.matches("BEGIN IMMEDIATE").count(), 1);
    assert_eq!(precommit.matches("ROLLBACK").count(), 0);
    assert_eq!(precommit.matches("COMMIT").count(), 0);
    assert_eq!(finalization.matches("ROLLBACK").count(), 1);
    assert_eq!(finalization.matches("COMMIT").count(), 1);
    assert_eq!(
        precommit
            .matches("KnowledgeProjectionWrite::with_projection_lock(")
            .count(),
        3
    );
    assert_eq!(precommit.matches(".locked()").count(), 2);
    assert_eq!(precommit.matches("repair verify begin: {error}").count(), 1);
    assert_eq!(
        finalization
            .matches("repair verify commit: {error}")
            .count(),
        1
    );
    assert_eq!(
        precommit
            .matches("RepairTarget::PageProjection { page_id, .. }")
            .count(),
        3
    );
    ordered(
        precommit,
        &[
            ".conn.lock().await",
            "BEGIN IMMEDIATE",
            "validate_current_db_receipts(",
            "validate_tag_record_set_on_connection(",
            "validate_deterministic_target_resolved(",
            "RepairWriter::RenamePageTitle",
            "RepairWriter::QuarantineStalePageProjection",
            "projection_rollback_paths(",
            "repair_target_receipt_on_connection(",
            "RepairVerificationReceiptDraft",
            "persist_verification_receipt(",
        ],
    );
    ordered(finalization, &["ROLLBACK", "COMMIT"]);
    assert_eq!(
        precommit.matches("persist_verification_receipt(").count(),
        3
    );
    assert!(!finalization.contains("persist_verification_receipt("));

    let (rename_persist, page_root_persist, no_page_root_persist) = persistence_segments(precommit);
    assert!(
        rename_persistence_follows_locked_validation(rename_persist),
        "rename persistence must follow its locked final page validation"
    );

    assert!(
        page_root_persistence_is_inside_projection_lock(page_root_persist),
        "page-root persistence escaped its final projection-lock closure"
    );

    assert_eq!(
        no_page_root_persist
            .matches("store.persist_verification_receipt(&receipt)?;")
            .count(),
        1
    );
    for forbidden in [
        "KnowledgeProjectionWrite::with_projection_lock(",
        "session.locked()",
        "drop(",
    ] {
        assert!(
            !no_page_root_persist.contains(forbidden),
            "lock-free persistence branch gained {forbidden}"
        );
    }

    let caller_start = repair
        .find("async fn record_repair_verification_inner<F>(")
        .expect("verification caller");
    let caller_end = repair[caller_start..]
        .find("#[cfg(test)]\nasync fn record_repair_verification_with_projection_session_hook")
        .map(|offset| caller_start + offset)
        .expect("verification test facade");
    let caller = &repair[caller_start..caller_end];
    assert!(!caller.contains(".conn.lock().await"));
    let child_call = caller
        .find("record_repair_verification_atomic(")
        .expect("child call");
    let clear_pending = caller[child_call..]
        .find("clear_pending_apply_receipt(")
        .map(|offset| child_call + offset)
        .expect("post-commit pending cleanup");
    assert!(child_call < clear_pending);
    let between = &caller[child_call..clear_pending];
    assert!(!between.contains("drop("));
    assert_eq!(between.matches(".await").count(), 1);

    for seam in [
        "pub(crate) fn persist_verification_receipt(",
        "pub(crate) async fn validate_current_db_receipts(",
        "pub(crate) async fn validate_deterministic_target_resolved(",
        "pub(crate) fn validate_current_page_receipts_locked(",
        "pub(crate) fn validate_current_page_receipts_on_repair_projection(",
        "pub(crate) async fn capture_rename_page_title_on_connection(",
    ] {
        assert_eq!(
            repair.matches(seam).count(),
            1,
            "required verification seam has wrong visibility: {seam}"
        );
    }
}

#[test]
fn r4_24a_rejects_persistence_moved_outside_the_final_projection_lock() {
    let child = include_str!("repair_verification.rs");
    let (_, precommit, _) = bounded_atomic_source(child);
    let (_, page_root_persist, _) = persistence_segments(precommit);
    assert!(page_root_persistence_is_inside_projection_lock(
        page_root_persist
    ));

    let inside = concat!(
        "                store.persist_verification_receipt(&receipt)?;\n",
        "                Ok(receipt)\n",
        "            })",
    );
    let outside = concat!(
        "                Ok(receipt)\n",
        "            })?;\n",
        "            store.persist_verification_receipt(&receipt)?;\n",
        "            Ok(receipt)",
    );
    let mutant = page_root_persist.replacen(inside, outside, 1);
    assert_ne!(
        mutant, page_root_persist,
        "mutation fixture must move persistence"
    );
    assert!(
        !page_root_persistence_is_inside_projection_lock(&mutant),
        "source tooth accepted persistence moved outside the projection lock"
    );
}

#[test]
fn r4_24a_rejects_persistence_moved_before_final_page_validation() {
    let child = include_str!("repair_verification.rs");
    let (_, precommit, _) = bounded_atomic_source(child);
    let (rename_persist, page_root_persist, _) = persistence_segments(precommit);
    assert!(rename_persistence_follows_locked_validation(rename_persist));
    assert!(page_root_persistence_is_inside_projection_lock(
        page_root_persist
    ));

    let rename_mutant = move_persist_before_validation(
        rename_persist,
        "            store.persist_verification_receipt(&receipt)?;\n",
        "            validate_current_page_receipts_on_repair_projection(\n",
    );
    assert!(
        !rename_persistence_follows_locked_validation(&rename_mutant),
        "source tooth accepted rename persistence before final validation"
    );

    let page_root_mutant = move_persist_before_validation(
        page_root_persist,
        "                store.persist_verification_receipt(&receipt)?;\n",
        "                validate_current_page_receipts_locked(\n",
    );
    assert!(
        !page_root_persistence_is_inside_projection_lock(&page_root_mutant),
        "source tooth accepted page-root persistence before final validation"
    );
}
