// SPDX-License-Identifier: Apache-2.0

use super::overview_subscriptions::ensure_overview_subscription_tables;

async fn test_connection() -> (libsql::Database, libsql::Connection) {
    let db = libsql::Builder::new_local(":memory:")
        .build()
        .await
        .expect("build overview schema database");
    let conn = db.connect().expect("connect overview schema database");
    let tx = conn.transaction().await.expect("begin overview schema");
    ensure_overview_subscription_tables(&tx)
        .await
        .expect("install overview schema");
    tx.commit().await.expect("commit overview schema");
    (db, conn)
}

async fn insert_subscription(
    conn: &libsql::Connection,
    subscription_id: &str,
    scope_kind: &str,
    scope_id: &str,
    page_id: &str,
    space: Option<&str>,
    state: &str,
) -> Result<u64, libsql::Error> {
    conn.execute(
        "INSERT INTO m6_overview_subscriptions (
             subscription_id, scope_kind, scope_id, page_id, space, state,
             created_at, detached_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 100, NULL)",
        libsql::params![subscription_id, scope_kind, scope_id, page_id, space, state],
    )
    .await
}

#[tokio::test]
async fn install_scope_and_space_nullability_are_bidirectional() {
    let (_db, conn) = test_connection().await;

    insert_subscription(
        &conn,
        "sub-install",
        "install",
        "install",
        "page-install",
        None,
        "active",
    )
    .await
    .expect("an install subscription has no space");

    assert!(
        insert_subscription(
            &conn,
            "sub-install-with-space",
            "install",
            "install-2",
            "page-install-2",
            Some("space-a"),
            "active",
        )
        .await
        .is_err(),
        "install iff NULL must reject an install row carrying a space"
    );
    assert!(
        insert_subscription(
            &conn,
            "sub-space-without-space",
            "space",
            "space-a",
            "page-space-a",
            None,
            "active",
        )
        .await
        .is_err(),
        "install iff NULL must reject a non-install row without a space"
    );
}

#[tokio::test]
async fn retained_subscription_identity_rejects_null_rows() {
    let (_db, conn) = test_connection().await;

    for suffix in ["a", "b"] {
        assert!(
            conn.execute(
                "INSERT INTO m6_overview_subscriptions (
                     subscription_id, scope_kind, scope_id, page_id, space, state,
                     created_at, detached_at
                 ) VALUES (NULL, 'install', ?1, ?2, NULL, 'detached', 100, 100)",
                libsql::params![format!("install-{suffix}"), format!("page-{suffix}")],
            )
            .await
            .is_err(),
            "NULL must not create a second identity-less retained row"
        );
    }
}

#[tokio::test]
async fn detached_history_allows_resubscription_but_only_one_live_scope() {
    let (_db, conn) = test_connection().await;

    insert_subscription(
        &conn,
        "sub-old",
        "community",
        "community-a",
        "page-old",
        Some("space-a"),
        "active",
    )
    .await
    .expect("first active subscription");
    conn.execute(
        "UPDATE m6_overview_subscriptions
            SET state = 'detached', detached_at = 200
          WHERE subscription_id = 'sub-old'",
        (),
    )
    .await
    .expect("detach while retaining history");

    insert_subscription(
        &conn,
        "sub-new",
        "community",
        "community-a",
        "page-new",
        Some("space-a"),
        "active",
    )
    .await
    .expect("same scope can subscribe again after detach");
    assert!(
        insert_subscription(
            &conn,
            "sub-duplicate-live",
            "community",
            "community-a",
            "page-third",
            Some("space-a"),
            "active",
        )
        .await
        .is_err(),
        "a second simultaneous active subscription for one scope must fail"
    );

    let mut rows = conn
        .query(
            "SELECT state, COUNT(*)
               FROM m6_overview_subscriptions
              WHERE scope_kind = 'community' AND scope_id = 'community-a'
              GROUP BY state ORDER BY state",
            (),
        )
        .await
        .expect("read retained subscription history");
    let detached = rows.next().await.unwrap().unwrap();
    assert_eq!(detached.get::<String>(0).unwrap(), "active");
    assert_eq!(detached.get::<i64>(1).unwrap(), 1);
    let active = rows.next().await.unwrap().unwrap();
    assert_eq!(active.get::<String>(0).unwrap(), "detached");
    assert_eq!(active.get::<i64>(1).unwrap(), 1);
}

#[tokio::test]
async fn one_page_cannot_serve_two_live_scopes() {
    let (_db, conn) = test_connection().await;

    insert_subscription(
        &conn,
        "sub-a",
        "community",
        "community-a",
        "shared-page",
        Some("space-a"),
        "active",
    )
    .await
    .expect("first live page binding");
    assert!(
        insert_subscription(
            &conn,
            "sub-b",
            "community",
            "community-b",
            "shared-page",
            Some("space-a"),
            "active",
        )
        .await
        .is_err(),
        "the active-page partial unique index must reject a shared page"
    );
}
