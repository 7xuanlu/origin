use axum::extract::Request;
use axum::handler::Handler;
use axum::response::IntoResponse;
use axum::routing::MethodRouter;
use axum::routing::Route;
use axum::Router;
use std::any::type_name;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use tower::{Layer, Service};
use wenlan_core::lint::serving::routes::{self, Method};
use wenlan_core::truth_manifest::{self, Builder, ReaderMethod};

#[path = "route_registry/app.rs"]
mod app;
#[path = "route_registry/handler_manifest.rs"]
mod handler_manifest;
pub use app::AppRouter;

pub(crate) struct TrackedRouter<S = ()> {
    inner: Router<S>,
    reads: BTreeMap<(Method, &'static str), usize>,
    /// Which router instance this is. Truth classification is keyed on
    /// `(builder, method, path)` -- `/api/health` registers once per builder,
    /// and two call sites land in both -- so the builder cannot be inferred.
    builder: Builder,
    /// Every `(method, path)` registered here, for the M5 truth-manifest drift
    /// assert. Separate from `reads`, which tracks only the scope-sensitive
    /// subset: a route can be scope-insensitive and still serve page prose, so
    /// deriving truth coverage from `reads` would leave the opt-out lists
    /// unclassified. Truth classification is total over the router.
    truth: BTreeSet<(ReaderMethod, &'static str)>,
    /// Exact route-to-handler bindings. This is deliberately independent of
    /// the truth and scope manifests: those classify response behavior, while
    /// this catches a valid route being wired to the wrong function.
    handlers: BTreeMap<(ReaderMethod, &'static str, &'static str), usize>,
    check_handlers: bool,
}

pub(crate) struct TrackedMethodRouter<S = ()> {
    inner: MethodRouter<S>,
    handlers: Vec<RegisteredHandler>,
}

pub(crate) struct FinalizedRouter<S = ()> {
    inner: Router<S>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    /// Total, unlike `sensitive()`. Scope sensitivity is a read-path concern so
    /// it drops the mutating methods; truth classification may not, because a
    /// `PUT` that echoes a page title back is page-bearing all the same.
    const fn truth(self) -> ReaderMethod {
        match self {
            Self::Get => ReaderMethod::Get,
            Self::Post => ReaderMethod::Post,
            Self::Put => ReaderMethod::Put,
            Self::Delete => ReaderMethod::Delete,
            Self::Patch => ReaderMethod::Patch,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegisteredHandler {
    method: RegisteredMethod,
    handler: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct HandlerBinding {
    builder: Builder,
    method: ReaderMethod,
    path: &'static str,
    handler: &'static str,
}

macro_rules! top_level_method {
    ($name:ident, $method:ident) => {
        pub(crate) fn $name<H, T, S>(handler: H) -> TrackedMethodRouter<S>
        where
            H: Handler<T, S>,
            T: 'static,
            S: Clone + Send + Sync + 'static,
        {
            let handler_name = type_name::<H>();
            TrackedMethodRouter {
                inner: axum::routing::$name(handler),
                handlers: vec![RegisteredHandler {
                    method: RegisteredMethod::$method,
                    handler: handler_name,
                }],
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
        let handler_name = type_name::<H>();
        self.inner = self.inner.post(handler);
        self.handlers.push(RegisteredHandler {
            method: RegisteredMethod::Post,
            handler: handler_name,
        });
        self
    }

    pub(crate) fn put<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        let handler_name = type_name::<H>();
        self.inner = self.inner.put(handler);
        self.handlers.push(RegisteredHandler {
            method: RegisteredMethod::Put,
            handler: handler_name,
        });
        self
    }

    pub(crate) fn delete<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        let handler_name = type_name::<H>();
        self.inner = self.inner.delete(handler);
        self.handlers.push(RegisteredHandler {
            method: RegisteredMethod::Delete,
            handler: handler_name,
        });
        self
    }

    pub(crate) fn patch<H, T>(mut self, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        let handler_name = type_name::<H>();
        self.inner = self.inner.patch(handler);
        self.handlers.push(RegisteredHandler {
            method: RegisteredMethod::Patch,
            handler: handler_name,
        });
        self
    }
}

impl<S> TrackedRouter<S>
where
    S: Clone + Send + Sync + 'static,
{
    pub(crate) fn new(builder: Builder) -> Self {
        Self {
            inner: Router::new(),
            reads: BTreeMap::new(),
            builder,
            truth: BTreeSet::new(),
            handlers: BTreeMap::new(),
            check_handlers: true,
        }
    }

    #[cfg(test)]
    fn new_unbound_for_test(builder: Builder) -> Self {
        Self {
            check_handlers: false,
            ..Self::new(builder)
        }
    }

    pub(crate) fn route(mut self, path: &'static str, route: TrackedMethodRouter<S>) -> Self {
        let rows = route
            .handlers
            .iter()
            .filter_map(|registered| {
                registered
                    .method
                    .sensitive()
                    .and_then(|method| routes::route(method, path))
            })
            .collect::<Vec<_>>();
        assert!(
            rows.len() == route.handlers.len()
                || NON_SENSITIVE_PATHS.contains(&path)
                || route.handlers.iter().all(|registered| {
                    NON_SENSITIVE_MIXED_ROUTES.contains(&(registered.method, path))
                        || registered
                            .method
                            .sensitive()
                            .is_some_and(|method| routes::route(method, path).is_some())
                }),
            "unclassified router path: {path}"
        );
        for row in rows {
            *self.reads.entry((row.method, row.path)).or_default() += 1;
        }

        // Truth classification is TOTAL over the router, so this runs for every
        // method at every path -- including the ones the scope-sensitivity
        // check opts out of above. A route can be scope-insensitive and still
        // serve page prose (`/api/steep`, `/api/distill`, every `/api/repairs/*`
        // are all in `NON_SENSITIVE_PATHS` and all page-bearing), so deriving
        // truth coverage from that opt-out list would leave the page-bearing
        // half of it silently unclassified.
        for registered in &route.handlers {
            let method = registered.method.truth();
            assert!(
                truth_manifest::http_reader(self.builder, method, path).is_some(),
                "unclassified reader path: {method:?} {path} in {:?}. Add a row to \
                 wenlan_core::truth_manifest and regenerate \
                 crates/wenlan-core/contracts/m5-reader-manifest-inventory.md with \
                 scripts/m5-reader-sweep.py --update-inventory; a new route is \
                 page_bearing until its response type says otherwise.",
                self.builder
            );
            self.truth.insert((method, path));
            *self
                .handlers
                .entry((method, path, registered.handler))
                .or_default() += 1;
        }

        self.inner = self.inner.route(path, route.inner);
        self
    }

    /// Registered-versus-classified set equality for the truth manifest.
    ///
    /// Both directions matter. A route with no row is caught in `route()` at
    /// registration; this catches the other half -- a row for a route that is
    /// no longer registered, which would otherwise sit in the table looking
    /// like coverage of something that does not exist.
    fn assert_truth_coverage(&self) {
        let expected: BTreeSet<(ReaderMethod, &'static str)> = truth_manifest::runtime_entries()
            .filter(|(builder, _, _)| *builder == self.builder)
            .map(|(_, method, path)| (method, path))
            .collect();
        assert_eq!(
            self.truth, expected,
            "truth manifest registration drift in {:?}",
            self.builder
        );
    }

    fn assert_handler_coverage_against(&self, expected: impl IntoIterator<Item = HandlerBinding>) {
        let mut seen = BTreeSet::new();
        let mut expected_counts = BTreeMap::new();
        for row in expected {
            assert!(seen.insert(row), "duplicate handler manifest row: {row:?}");
            if row.builder == self.builder {
                *expected_counts
                    .entry((row.method, row.path, row.handler))
                    .or_default() += 1;
            }
        }
        assert_eq!(
            self.handlers, expected_counts,
            "route handler binding drift in {:?}",
            self.builder
        );
    }

    fn assert_handler_coverage(&self) {
        if self.check_handlers {
            self.assert_handler_coverage_against(handler_manifest::bindings());
        }
    }

    pub(crate) fn finish(self) -> FinalizedRouter<S> {
        let expected = routes::sensitive_read_routes()
            .map(|row| ((row.method, row.path), 1usize))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(self.reads, expected, "sensitive route registration drift");
        self.assert_truth_coverage();
        self.assert_handler_coverage();
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
        // Truth coverage is exact even here. `finish_restricted` is loose about
        // *sensitive* routes because the repair router deliberately serves a
        // subset of them, but the manifest enumerates the repair builder's six
        // entries exactly, so there is no reason to accept less.
        self.assert_truth_coverage();
        self.assert_handler_coverage();
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

    /// Like [`Self::layer`], but only for requests that matched a route.
    ///
    /// The M5 truth guard needs this rather than `layer`: it looks the route up
    /// in the manifest by `MatchedPath`, which only exists once routing has
    /// happened, and a request that matched nothing has no marker shape to
    /// enforce -- it is a 404 either way.
    pub(crate) fn route_layer<L>(self, layer: L) -> Self
    where
        L: Layer<Route> + Clone + Send + Sync + 'static,
        L::Service: Service<Request> + Clone + Send + Sync + 'static,
        <L::Service as Service<Request>>::Response: IntoResponse + 'static,
        <L::Service as Service<Request>>::Error: Into<Infallible> + 'static,
        <L::Service as Service<Request>>::Future: Send + 'static,
    {
        Self {
            inner: self.inner.route_layer(layer),
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
    "/api/spaces/{from}/move-to/{to}", "/api/pages/{id}/archive", "/api/pages/{id}/review",
    "/api/refinery/queue/{id}/reject", "/api/refinery/queue/{id}/accept", "/api/sources/{id}", "/api/sources/{id}/sync",
    "/api/config", "/api/config/skip-apps", "/api/config/routing", "/api/setup/status", "/api/setup/anthropic-key", "/api/on-device-model",
    "/api/on-device-model/download", "/api/outbox/drain", "/api/outbox/status", "/api/chunks/{id}/update", "/api/chunks/time-range", "/api/chunks/delete-bulk",
    "/api/memory/entities/{id}/confirm", "/api/memory/entities/{id}/delete", "/api/memory/entities/{entity_id}/observations",
    "/api/memory/entities/{id}/merge", "/api/memory/entities/{id}/aliases",
    "/api/memory/observations/{id}", "/api/memory/observations/{id}/confirm", "/api/spaces/{name}/pin",
    "/api/spaces/{name}/confirm", "/api/spaces/reorder", "/api/spaces/{name}/star", "/api/documents/{source_id}/space",
    "/api/tags/{name}", "/api/documents/{source_id}/tags", "/api/memory/{id}/update", "/api/memory/{id}/stability",
    "/api/memory/{id}/correct", "/api/profile/narrative/regenerate", "/api/memory/{id}/pin", "/api/memory/{id}/unpin",
    "/api/snapshots/{id}/delete", "/api/memory/{id}/update-page", "/api/knowledge/path",
    "/api/onboarding/milestones/{id}/acknowledge", "/api/onboarding/reset", "/ws/updates",
    "/api/pages/{id}/map/layout", "/api/pages/{id}/map/nodes", "/api/pages/{id}/map/edges",
    "/api/pages/{id}/map/nodes/{node_id}", "/api/pages/{id}/map/edges/{edge_id}",
    "/api/pages/{id}/map/improve",
];

const NON_SENSITIVE_MIXED_ROUTES: &[(RegisteredMethod, &str)] = &[
    (RegisteredMethod::Patch, "/api/brief"),
    (RegisteredMethod::Put, "/api/profile"),
    (RegisteredMethod::Put, "/api/agents/{name}"),
    (RegisteredMethod::Delete, "/api/agents/{name}"),
    (RegisteredMethod::Post, "/api/spaces"),
    (RegisteredMethod::Put, "/api/spaces/{name}"),
    (RegisteredMethod::Delete, "/api/spaces/{name}"),
    (RegisteredMethod::Put, "/api/spaces/default"),
    (RegisteredMethod::Delete, "/api/spaces/default"),
    (RegisteredMethod::Post, "/api/pages"),
    (RegisteredMethod::Put, "/api/pages/{id}"),
    (RegisteredMethod::Delete, "/api/pages/{id}"),
    (RegisteredMethod::Post, "/api/sources"),
    (RegisteredMethod::Delete, "/api/pages/{id}/map"),
];

#[cfg(test)]
#[path = "route_registry/tests.rs"]
mod tests;
