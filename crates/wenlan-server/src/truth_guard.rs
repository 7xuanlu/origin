// SPDX-License-Identifier: Apache-2.0
//! The wire end of the M5 exposure contract.
//!
//! One middleware, layered over every registered route, that turns two request
//! headers into a [`TruthView`] the handlers downstream can act on -- and
//! refuses outright when a marker lands somewhere it may do nothing.
//!
//! It lives in middleware rather than in an extractor for one reason: the
//! refusal has to cover the 153 routes that have **no** truth adapter and never
//! will. An extractor only runs where a handler asks for it, so a `none`-shaped
//! route would silently ignore the marker -- which is the failure mode the
//! refusal exists to prevent. Middleware is the only seam that is total over the
//! router, and it stays total for routes added later without anyone remembering.
//!
//! `MatchedPath` is what makes the lookup honest: the manifest is keyed on the
//! route *template*, so the guard asks axum which route matched rather than
//! re-implementing routing against the concrete URI. Re-deriving the route would
//! be a second, weaker copy of `TrackedRouter` -- the hand-copied-canonical-thing
//! pattern that produced three wrong counts in Stage 0.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-reader-manifest.md` sections 4 and 6.

use axum::{
    extract::{FromRequestParts, MatchedPath, RawPathParams, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use wenlan_core::truth_contract::{
    self, MarkerOutcome, TruthDecision, TruthGrant, CONTRACT_HEADER, INTENT_HEADER,
};
use wenlan_core::truth_manifest::{self, Builder, MarkerShape, ReaderMethod};

use crate::state::SharedState;

/// The path parameters that name a page. Every `named_page`-shaped route spells
/// it `{id}`; `page_id` is accepted because `/api/distill/{page_id}` shows the
/// codebase does not spell it one way everywhere, and a grant that missed
/// because of a parameter name would fail *open* on the next route to use it.
const PAGE_ID_PARAMS: &[&str] = &["id", "page_id"];

/// What this call may see. Attached to every request as an extension.
///
/// Absent means automatic -- a handler that reads it out of a request that never
/// went through the guard (a unit test building a bare `Request`) gets the
/// conservative answer, not a panic and not a grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TruthView {
    pub grant: TruthGrant,
}

impl TruthView {
    pub fn automatic() -> Self {
        Self {
            grant: TruthGrant::Automatic,
        }
    }

    /// The view for a request, defaulting to automatic when none was attached.
    pub fn of(req: &Request) -> Self {
        req.extensions()
            .get::<Self>()
            .cloned()
            .unwrap_or_else(Self::automatic)
    }
}

/// Middleware state: the server state (for the audit write) plus which router
/// instance this is, since manifest identity is `(builder, method, path)`.
#[derive(Clone)]
pub struct TruthGuardState {
    pub state: SharedState,
    pub builder: Builder,
}

fn reader_method(method: &axum::http::Method) -> Option<ReaderMethod> {
    Some(match *method {
        axum::http::Method::GET => ReaderMethod::Get,
        axum::http::Method::POST => ReaderMethod::Post,
        axum::http::Method::PUT => ReaderMethod::Put,
        axum::http::Method::DELETE => ReaderMethod::Delete,
        axum::http::Method::PATCH => ReaderMethod::Patch,
        _ => return None,
    })
}

fn header(req: &Request, name: &str) -> Option<String> {
    req.headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Resolve every call's grant, refuse the ones that may not carry a marker, and
/// leave a durable row behind for the ones that do.
pub async fn guard(
    State(guard_state): State<TruthGuardState>,
    req: Request,
    next: Next,
) -> Response {
    let marker = truth_contract::marker_present(header(&req, INTENT_HEADER).as_deref());

    // The overwhelming majority of calls. Nothing to resolve, nothing to record,
    // no lookup worth doing -- an unmarked call is automatic on every route.
    if !marker {
        return run_with(req, next, TruthView::automatic()).await;
    }

    let contract = truth_contract::parse_contract(header(&req, CONTRACT_HEADER).as_deref());
    let caller = header(&req, "x-agent-name")
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let method_str = req.method().to_string();

    // A route we cannot identify gets the fail-closed shape. With `route_layer`
    // the matched path is always present, so this is belt-and-braces rather than
    // a live branch -- but the safe answer to "which route is this" being
    // unanswerable is that a marker may do nothing here.
    let shape = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_string())
        .zip(reader_method(req.method()))
        .and_then(|(path, method)| {
            truth_manifest::http_reader(guard_state.builder, method, &path)
                .map(|row| (row.marker_shape, path))
        });
    let (shape, path) = shape.unwrap_or((MarkerShape::None, req.uri().path().to_string()));

    let (named_pages, req) = named_pages(req).await;
    let decision = truth_contract::resolve(shape, contract, marker, &named_pages);

    // Every marked call, granted or refused. Best-effort: an audit write that
    // fails must not take the request with it, but it must be loud, because a
    // silently absent audit row is the compensating control quietly not being
    // there.
    let db = { guard_state.state.read().await.db.clone() };
    if let Some(db) = db {
        if let Err(error) = db
            .record_truth_marker(
                &caller,
                &method_str,
                &path,
                MarkerOutcome::of(&decision),
                &named_pages,
            )
            .await
        {
            tracing::error!("[truth] marker audit write failed for {method_str} {path}: {error}");
        }
    } else {
        // Same failure, different cause: no row was written, so the compensating
        // control for the conceded enumerate-then-fetch walk is not there for
        // this call. Silence here would be the exact shape the comment above
        // rules out. Latent -- no known window serves requests before `db` is
        // set -- which is why the request is still allowed to proceed.
        tracing::error!(
            "[truth] no database open; the marker audit row for {method_str} {path} from \
             {caller} (outcome {}, pages {named_pages:?}) was not written",
            MarkerOutcome::of(&decision).as_str(),
        );
    }

    match decision {
        TruthDecision::Refuse => {
            tracing::warn!(
                "[truth] refused a reader-intent marker at {method_str} {path} from {caller}"
            );
            refusal(&method_str, &path)
        }
        TruthDecision::Grant(grant) => run_with(req, next, TruthView { grant }).await,
    }
}

async fn run_with(mut req: Request, next: Next, view: TruthView) -> Response {
    req.extensions_mut().insert(view);
    next.run(req).await
}

/// The page IDs this call named, in path order.
///
/// Taken from the path, never from the client. A client-supplied list could
/// disagree with the route it was sent to, and then there are two answers to
/// "which page did this call name" -- one of them chosen by the caller.
async fn named_pages(req: Request) -> (Vec<String>, Request) {
    let (mut parts, body) = req.into_parts();
    let ids = match RawPathParams::from_request_parts(&mut parts, &()).await {
        Ok(params) => params
            .iter()
            .filter(|(name, _)| PAGE_ID_PARAMS.contains(name))
            .map(|(_, value)| value.to_string())
            .collect(),
        Err(_) => Vec::new(),
    };
    (ids, Request::from_parts(parts, body))
}

/// Refuse, rather than downgrade to automatic and serve the request anyway.
///
/// An ignored marker behaves correctly today and silently wrong after a
/// refactor; a refused one fails at the first integration test. The body names
/// the route and nothing else -- no page identity, no titles, nothing that would
/// make an error body into a disclosure channel.
fn refusal(method: &str, path: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({
            "error": format!(
                "reader-intent marker is not accepted at {method} {path}; \
                 this reader serves automatic callers only"
            ),
        })),
    )
        .into_response()
}

#[cfg(test)]
#[path = "truth_guard_test.rs"]
mod tests;
