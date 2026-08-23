use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::{HeaderName, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use tower_http::cors::CorsLayer;

use crate::auth;
use crate::client::WenlanClient;
use crate::tools::{TransportMode, WenlanMcpServer};

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub port: u16,
    pub host: String,
    pub origin_url: String,
    pub token: Option<String>,
    pub agent_name: String,
    pub user_id: Option<String>,
    pub allowed_origins: Vec<String>,
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "status": "ok",
        "server": "wenlan-mcp",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

pub async fn run_serve(config: ServeConfig) -> anyhow::Result<()> {
    let client =
        WenlanClient::new(config.origin_url.clone()).with_agent_name(config.agent_name.clone());
    let agent_name = config.agent_name.clone();
    let user_id = config.user_id.clone();
    let token = config.token.clone();
    let allowed_origins = config.allowed_origins.clone();

    // rmcp's default allowed_hosts = [localhost, 127.0.0.1, ::1] is
    // DNS-rebinding protection for loopback deployments. Every serve
    // deployment we support is reached through a public tunnel
    // (cloudflared, ngrok) that forwards the tunnel hostname in Host,
    // whether or not a bearer token is used (Wenlan.app runs `serve
    // --no-auth --host 127.0.0.1` and fronts it with cloudflared; auth
    // deployments do the same with a token). Leaving the default in
    // place rejects every tunneled request with a plain-text 403 the
    // upstream MCP proxy cannot parse, surfacing to users as a bogus
    // "-32600 Invalid Request". MCP's custom-header requirement
    // already triggers CORS preflight, so the Origin allowlist catches
    // browser-driven DNS rebinding; a local non-browser attacker
    // bypasses Host checking anyway by hitting 127.0.0.1 directly.
    let mcp_config = StreamableHttpServerConfig::default().disable_allowed_hosts();

    let mcp_service = StreamableHttpService::new(
        move || {
            Ok(WenlanMcpServer::new(
                client.clone(),
                TransportMode::Http,
                agent_name.clone(),
                user_id.clone(),
            ))
        },
        Arc::new(LocalSessionManager::default()),
        mcp_config,
    );

    let cors = build_cors_layer(&config.allowed_origins);

    let mut router = Router::new()
        .nest_service("/mcp", mcp_service)
        .route("/health", get(health))
        .layer(cors);

    if let Some(ref expected_token) = token {
        let token_for_middleware = expected_token.clone();
        let origins_for_middleware = allowed_origins.clone();
        router = router.layer(middleware::from_fn(move |req: Request, next: Next| {
            let token = token_for_middleware.clone();
            let origins = origins_for_middleware.clone();
            async move { auth_and_origin_middleware(req, next, &token, &origins).await }
        }));
    }

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("wenlan-mcp HTTP server listening on {}", addr);

    if token.is_some() {
        tracing::info!("Bearer token authentication enabled");
    } else {
        tracing::warn!("Running without authentication — only safe on loopback");
    }

    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let ctrl_c = tokio::signal::ctrl_c();
            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => {},
                _ = sigterm.recv() => {},
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
        }
        tracing::info!("Shutting down wenlan-mcp HTTP server");
    };

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;

    Ok(())
}

fn build_cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            http::header::AUTHORIZATION,
            http::header::CONTENT_TYPE,
            http::header::ACCEPT,
            HeaderName::from_static("mcp-session-id"),
            HeaderName::from_static("mcp-protocol-version"),
        ]);

    if allowed_origins.iter().any(|o| o == "*") {
        cors.allow_origin(tower_http::cors::Any)
    } else {
        let origins: Vec<http::HeaderValue> = allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        cors.allow_origin(origins)
    }
}

/// Auth middleware: bearer token first (401), then Origin header (403).
async fn auth_and_origin_middleware(
    req: Request,
    next: Next,
    expected_token: &str,
    allowed_origins: &[String],
) -> axum::response::Response {
    let is_preflight = req.method() == Method::OPTIONS;
    let is_health = req.uri().path() == "/health";
    if is_preflight || is_health {
        return next.run(req).await;
    }

    // 1. Validate bearer token FIRST
    let auth_header = req.headers().get(http::header::AUTHORIZATION);
    match auth_header {
        Some(value) => {
            let value_str = value.to_str().unwrap_or("");
            match value_str.strip_prefix("Bearer ") {
                Some(provided) if auth::verify_token(provided, expected_token) => {}
                _ => return (StatusCode::UNAUTHORIZED, "Invalid bearer token").into_response(),
            }
        }
        None => return (StatusCode::UNAUTHORIZED, "Authorization header required").into_response(),
    }

    // 2. Validate Origin header AFTER auth
    if let Some(origin) = req.headers().get(http::header::ORIGIN) {
        if let Ok(origin_str) = origin.to_str() {
            if !auth::is_origin_allowed(origin_str, allowed_origins) {
                return (StatusCode::FORBIDDEN, "Origin not allowed").into_response();
            }
        }
    }

    next.run(req).await
}
