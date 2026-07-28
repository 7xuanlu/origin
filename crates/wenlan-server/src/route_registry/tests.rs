use super::*;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::collections::BTreeMap;
use tower::ServiceExt;

#[test]
#[should_panic(expected = "unclassified router path")]
fn helper_registration_cannot_bypass_classification() {
    fn helper(router: TrackedRouter) -> TrackedRouter {
        router.route("/api/memory/dynamic-leak", super::get(|| async { "leak" }))
    }
    let _ = helper(TrackedRouter::new(Builder::Main));
}

#[test]
#[should_panic(expected = "unclassified router path")]
fn unknown_route_cannot_bypass_classification() {
    let _ = TrackedRouter::<()>::new(Builder::Main)
        .route("/api/unknown-read", super::get(|| async { "leak" }));
}

#[test]
#[should_panic(expected = "unclassified router path")]
fn wrong_method_cannot_satisfy_a_sensitive_registration() {
    let _ = TrackedRouter::<()>::new(Builder::Main)
        .route("/api/agents", super::post(|| async { "leak" }));
}

#[test]
#[should_panic]
fn duplicate_registration_fails_loud() {
    let router = TrackedRouter::<()>::new(Builder::Main)
        .route("/api/profile", super::get(|| async { "first" }))
        .route("/api/profile", super::get(|| async { "second" }));
    let _ = router.finish();
}

/// A path that IS in the truth manifest but carries no scope classification is
/// still refused. The two checks are independent, and this is the one that would
/// silently pass if truth coverage were derived from `sensitive_read_routes()`.
#[test]
#[should_panic(expected = "unclassified reader path")]
fn route_outside_the_truth_manifest_is_refused() {
    // `/api/health` is in `NON_SENSITIVE_PATHS`, so the scope check waves it
    // through -- but only GET is in the manifest, so the truth check must not.
    let _ = TrackedRouter::<()>::new(Builder::Main)
        .route("/api/health", super::put(|| async { "leak" }));
}

/// A route that belongs to the other builder does not satisfy this one.
#[test]
#[should_panic(expected = "unclassified reader path")]
fn main_only_route_is_refused_in_the_repair_builder() {
    let _ = TrackedRouter::<()>::new(Builder::Repair)
        .route("/api/profile", super::get(|| async { "leak" }));
}

/// Group the manifest's entries for one builder into per-path method routers.
///
/// `Get` and `Patch` have no chaining method on `TrackedMethodRouter`, so they
/// have to open the chain; everything else chains onto them. No path in the
/// manifest carries both, which this asserts rather than assumes.
fn register_builder(builder: Builder) -> TrackedRouter<()> {
    let mut by_path: BTreeMap<&'static str, Vec<ReaderMethod>> = BTreeMap::new();
    for (b, method, path) in truth_manifest::runtime_entries() {
        if b == builder {
            by_path.entry(path).or_default().push(method);
        }
    }

    let mut router = TrackedRouter::<()>::new(builder);
    for (path, mut methods) in by_path {
        let opener = |m: &ReaderMethod| !matches!(m, ReaderMethod::Get | ReaderMethod::Patch);
        methods.sort_by_key(opener);
        assert!(
            !(methods.contains(&ReaderMethod::Get) && methods.contains(&ReaderMethod::Patch)),
            "{path} carries both GET and PATCH; neither can chain onto the other"
        );

        let mut method_router = match methods[0] {
            ReaderMethod::Get => super::get(|| async { "read" }),
            ReaderMethod::Post => super::post(|| async { "read" }),
            ReaderMethod::Put => super::put(|| async { "read" }),
            ReaderMethod::Delete => super::delete(|| async { "read" }),
            ReaderMethod::Patch => super::patch(|| async { "read" }),
        };
        for method in &methods[1..] {
            method_router = match method {
                ReaderMethod::Post => method_router.post(|| async { "read" }),
                ReaderMethod::Put => method_router.put(|| async { "read" }),
                ReaderMethod::Delete => method_router.delete(|| async { "read" }),
                ReaderMethod::Get | ReaderMethod::Patch => {
                    unreachable!("{path}: {method:?} cannot chain, and sorting put it first")
                }
            };
        }
        router = router.route(path, method_router);
    }
    router
}

/// The manifest is registerable and satisfies both drift asserts exactly.
///
/// This is the positive control for `assert_truth_coverage`: it registers
/// precisely the manifest's rows and finishes, so a row for a route that cannot
/// be registered, or a manifest that no longer covers what `finish()` expects,
/// fails here rather than at daemon startup.
#[tokio::test]
async fn manifest_registers_and_seals_the_main_builder() {
    let app: AppRouter = register_builder(Builder::Main).finish().with_state(());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/profile")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}

#[test]
fn manifest_registers_and_seals_the_repair_builder() {
    let _ = register_builder(Builder::Repair).finish_restricted();
}

/// Dropping a single manifest row must fail `finish()`. Without this the
/// coverage assert could be vacuous -- a set-equality check whose two sides are
/// built from the same iterator proves nothing.
#[test]
#[should_panic(expected = "truth manifest registration drift")]
fn a_missing_registration_fails_the_coverage_assert() {
    let mut router = TrackedRouter::<()>::new(Builder::Repair);
    // The repair builder's six entries, minus `/api/status`.
    for (b, method, path) in truth_manifest::runtime_entries() {
        if b != Builder::Repair || path == "/api/status" {
            continue;
        }
        router = match method {
            ReaderMethod::Get => router.route(path, super::get(|| async { "read" })),
            ReaderMethod::Post => router.route(path, super::post(|| async { "read" })),
            _ => router,
        };
    }
    let _ = router.finish_restricted();
}
