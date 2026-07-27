use axum::extract::Request;
use axum::handler::Handler;
use axum::response::IntoResponse;
use axum::routing::MethodRouter;
use axum::routing::Route;
use axum::Router;
use std::collections::BTreeMap;
use std::convert::Infallible;
use tower::{Layer, Service};
use wenlan_core::lint::serving::routes::{self, Method};

#[path = "route_registry/app.rs"]
mod app;
pub use app::AppRouter;

pub(crate) struct TrackedRouter<S = ()> {
    inner: Router<S>,
    reads: BTreeMap<(Method, &'static str), usize>,
}

pub(crate) struct TrackedMethodRouter<S = ()> {
    inner: MethodRouter<S>,
    methods: Vec<RegisteredMethod>,
}

pub(crate) struct FinalizedRouter<S = ()> {
    inner: Router<S>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RegisteredMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

impl RegisteredMethod {
    const fn sensitive(self) -> Option<Method> {
        match self {
            Self::Get => Some(Method::Get),
            Self::Post => Some(Method::Post),
            Self::Put | Self::Delete | Self::Patch => None,
        }
    }
}

macro_rules! top_level_method {
    ($name:ident, $method:ident) => {
        pub(crate) fn $name<H, T, S>(handler: H) -> TrackedMethodRouter<S>
        where
            H: Handler<T, S>,
            T: 'static,
            S: Clone + Send + Sync + 'static,
        {
            TrackedMethodRouter {
                inner: axum::routing::$name(handler),
                methods: vec![RegisteredMethod::$method],
            }
        }
    };
}

top_level_method!(get, Get);
top_level_method!(post, Post);
top_level_method!(put, Put);
top_level_method!(delete, Delete);
top_level_method!(patch, Patch);

impl<S> TrackedMethodRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub(crate) fn post<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.inner = self.inner.post(handler);
        self.methods.push(RegisteredMethod::Post);
        self
    }

    pub(crate) fn put<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.inner = self.inner.put(handler);
        self.methods.push(RegisteredMethod::Put);
        self
    }

    pub(crate) fn delete<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.inner = self.inner.delete(handler);
        self.methods.push(RegisteredMethod::Delete);
        self
    }
}

impl<S> TrackedRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub(crate) fn new() -> Self {
        Self {
            inner: Router::new(),
            reads: BTreeMap::new(),
        }
    }

    pub(crate) fn route(mut self, path: &'static str, route: TrackedMethodRouter<S>) -> Self {
        let rows = route
            .methods
            .iter()
            .filter_map(|method| {
                method
                    .sensitive()
                    .and_then(|method| routes::route(method, path))
            })
            .collect::<Vec<_>>();
        assert!(
            rows.len() == route.methods.len()
                || NON_SENSITIVE_PATHS.contains(&path)
                || route.methods.iter().all(|method| {
                    NON_SENSITIVE_MIXED_ROUTES.contains(&(*method, path))
                        || method
                            .sensitive()
                            .is_some_and(|method| routes::route(method, path).is_some())
                }),
            "unclassified router path: {path}"
        );
        for row in rows {
            *self.reads.entry((row.method, row.path)).or_default() += 1;
        }
        self.inner = self.inner.route(path, route.inner);
        self
    }

    pub(crate) fn finish(self) -> FinalizedRouter<S> {
        let expected = routes::sensitive_read_routes()
            .map(|row| ((row.method, row.path), 1usize))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(self.reads, expected, "sensitive route registration drift");
        FinalizedRouter { inner: self.inner }
    }

    pub(crate) fn finish_restricted(self) -> FinalizedRouter<S> {
        let expected = routes::sensitive_read_routes()
            .map(|row| ((row.method, row.path), 1usize))
            .collect::<BTreeMap<_, _>>();
        assert!(
            self.reads
                .iter()
                .all(|(route, count)| expected.get(route) == Some(count)),
            "restricted router registered an unknown or duplicate sensitive read route"
        );
        FinalizedRouter { inner: self.inner }
    }
}

impl<S> FinalizedRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub(crate) fn layer<L>(self, layer: L) -> Self
    where
        L: Layer<Route> + Clone + Send + Sync + 'static,
        L::Service: Service<Request> + Clone + Send + Sync + 'static,
        <L::Service as Service<Request>>::Response: IntoResponse + 'static,
        <L::Service as Service<Request>>::Error: Into<Infallible> + 'static,
        <L::Service as Service<Request>>::Future: Send + 'static,
    {
        Self {
            inner: self.inner.layer(layer),
        }
    }

    pub(crate) fn with_state(self, state: S) -> AppRouter {
        AppRouter::new(self.inner.with_state(state))
    }
}

#[rustfmt::skip]
const NON_SENSITIVE_PATHS: &[&str] = &[
    "/api/health", "/api/status", "/api/ping", "/api/lint", "/api/repairs/plan", "/api/repairs/plan/entries", "/api/repairs/prepare", "/api/repairs/apply", "/api/repairs/verify", "/api/llm/test", "/api/shutdown", "/api/debug/pipeline",
    "/api/steep", "/api/distill", "/api/distill/{page_id}",
    "/api/ingest/text", "/api/ingest/webpage", "/api/ingest/memory", "/api/documents/{source}/{source_id}",
    "/api/import/memories", "/api/import/chat-export", "/api/memory/store", "/api/memory/confirm/{source_id}",
    "/api/memory/delete/{source_id}", "/api/memory/reclassify/{source_id}", "/api/memory/revision/{id}/accept",
    "/api/memory/revision/{id}/dismiss", "/api/memory/contradiction/{source_id}/dismiss", "/api/memory/entities",
    "/api/memory/relations", "/api/memory/observations", "/api/memory/link-entity", "/api/spaces/{name}",
    "/api/spaces/{from}/move-to/{to}", "/api/pages/{id}/archive",
    "/api/refinery/queue/{id}/reject", "/api/refinery/queue/{id}/accept", "/api/sources/{id}", "/api/sources/{id}/sync",
    "/api/config", "/api/config/skip-apps", "/api/config/routing", "/api/setup/status", "/api/setup/anthropic-key", "/api/on-device-model",
    "/api/on-device-model/download", "/api/chunks/{id}/update", "/api/chunks/time-range", "/api/chunks/delete-bulk",
    "/api/memory/entities/{id}/confirm", "/api/memory/entities/{id}/delete", "/api/memory/entities/{entity_id}/observations",
    "/api/memory/observations/{id}", "/api/memory/observations/{id}/confirm", "/api/spaces/{name}/pin",
    "/api/spaces/{name}/confirm", "/api/spaces/reorder", "/api/spaces/{name}/star", "/api/documents/{source_id}/space",
    "/api/tags/{name}", "/api/documents/{source_id}/tags", "/api/memory/{id}/update", "/api/memory/{id}/stability",
    "/api/memory/{id}/correct", "/api/profile/narrative/regenerate", "/api/memory/{id}/pin", "/api/memory/{id}/unpin",
    "/api/snapshots/{id}/delete", "/api/memory/{id}/update-page", "/api/knowledge/path",
    "/api/onboarding/milestones/{id}/acknowledge", "/api/onboarding/reset", "/ws/updates",
    "/api/pages/{id}/map/layout", "/api/pages/{id}/map/nodes", "/api/pages/{id}/map/edges",
    "/api/pages/{id}/map/nodes/{node_id}", "/api/pages/{id}/map/edges/{edge_id}",
    "/api/pages/{id}/map/improve",
    "/api/communities/proposals/{id}/accept", "/api/communities/proposals/{id}/reject",
];

const NON_SENSITIVE_MIXED_ROUTES: &[(RegisteredMethod, &str)] = &[
    (RegisteredMethod::Put, "/api/profile"),
    (RegisteredMethod::Put, "/api/agents/{name}"),
    (RegisteredMethod::Delete, "/api/agents/{name}"),
    (RegisteredMethod::Post, "/api/spaces"),
    (RegisteredMethod::Put, "/api/spaces/{name}"),
    (RegisteredMethod::Delete, "/api/spaces/{name}"),
    (RegisteredMethod::Post, "/api/pages"),
    (RegisteredMethod::Put, "/api/pages/{id}"),
    (RegisteredMethod::Delete, "/api/pages/{id}"),
    (RegisteredMethod::Post, "/api/sources"),
    (RegisteredMethod::Delete, "/api/pages/{id}/map"),
];

#[cfg(test)]
#[path = "route_registry/tests.rs"]
mod tests;
