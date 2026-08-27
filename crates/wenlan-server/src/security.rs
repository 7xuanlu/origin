// SPDX-License-Identifier: Apache-2.0
//! Cross-origin request guard for the loopback daemon.
//!
//! The daemon has no auth and binds `127.0.0.1:7878` by default, so without a
//! guard any web page the user visits could drive it — cross-origin reads of
//! the whole memory store, CSRF writes, or DNS-rebinding. Native clients (the
//! desktop app, the `wenlan` CLI, `wenlan-mcp`) all talk to the daemon over
//! reqwest and send no `Origin` header, so they pass through untouched.
//! Browsers always attach `Origin` on cross-origin requests; a non-local one
//! is rejected. A non-local `Host` (DNS rebinding) is rejected too — unless the
//! operator deliberately exposed the daemon via `WENLAN_BIND_ADDR` (e.g. the
//! Docker image), which opts out of the Host check and owns its own access
//! control. A browser omits `Origin` on a no-cors GET — a plain `<img>`,
//! `<script>`, or a top-level navigation from a link — so a blind cross-site
//! GET would otherwise reach every route; browsers always attach
//! `Sec-Fetch-Site` to those same requests, so checking it closes that gap.
//! A local `Origin` still passes even when `Sec-Fetch-Site` says cross-site,
//! because the Tauri webview (`tauri://localhost`) and the Vite dev server
//! are cross-site to `127.0.0.1` by that header's definition yet are
//! legitimate local callers.

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};

/// True when a browser's `Sec-Fetch-Site` proves the request is cross-site
/// and no local `Origin` vouches for it. `sec_fetch_site` is compared
/// trimmed and ASCII-lowercased; `"cross-site"` and `"same-site"` are the
/// only values treated as cross-origin evidence — `"same-origin"`, `"none"`,
/// an absent header, or any other value all return `false`.
pub(crate) fn fetch_site_blocks(sec_fetch_site: Option<&str>, origin: Option<&str>) -> bool {
    let Some(site) = sec_fetch_site else {
        return false;
    };
    let site = site.trim().to_ascii_lowercase();
    if site != "cross-site" && site != "same-site" {
        return false;
    }
    !origin.is_some_and(origin_is_local)
}

/// Reject browser-driven cross-origin requests before they reach a handler.
pub async fn guard_local_only(req: Request, next: Next) -> Result<Response, StatusCode> {
    let headers = req.headers();

    if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        if !origin_is_local(origin) {
            return Err(StatusCode::FORBIDDEN);
        }
    }

    // A browser attaches `Sec-Fetch-Site` even when it omits `Origin` on a
    // no-cors GET, closing the blind cross-site GET gap the Origin-only check
    // above leaves open.
    let sec_fetch_site = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok());
    let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    if fetch_site_blocks(sec_fetch_site, origin) {
        return Err(StatusCode::FORBIDDEN);
    }

    // DNS-rebinding defense applies unless the operator deliberately bound the
    // daemon to a routable address (Docker/LAN), where the Host is legitimately
    // non-loopback and access control is their responsibility. A loopback bind
    // — and a bind value we cannot parse — keeps the check on.
    if bind_scope_from_env() != BindScope::External {
        if let Some(host) = headers.get(header::HOST).and_then(|v| v.to_str().ok()) {
            if !host_is_local(host) {
                return Err(StatusCode::FORBIDDEN);
            }
        }
    }

    Ok(next.run(req).await)
}

/// How the operator's `WENLAN_BIND_ADDR` value classifies for the two security
/// checks that key off it: this module's DNS-rebinding Host guard and the lint
/// external-egress gate (`lint_routes.rs`). Both treat an unparseable value
/// conservatively — it is neither a trusted loopback bind nor a deliberate
/// exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BindScope {
    /// Unset — the daemon uses its default loopback bind.
    Unset,
    /// Set to a loopback address.
    Loopback,
    /// A routable address or a hostname: deliberately exposed.
    External,
    /// Set, but not `host:port`.
    Unparseable,
}

pub(crate) fn bind_scope_from_env() -> BindScope {
    bind_scope(wenlan_core::env_compat::var_compat("WENLAN_BIND_ADDR"))
}

pub(crate) fn bind_scope(bind: Option<std::ffi::OsString>) -> BindScope {
    let Some(bind) = bind else {
        return BindScope::Unset;
    };
    let Ok(bind) = bind.into_string() else {
        return BindScope::Unparseable;
    };
    if let Ok(address) = bind.parse::<std::net::SocketAddr>() {
        return if address.ip().is_loopback() {
            BindScope::Loopback
        } else {
            BindScope::External
        };
    }
    // Not a `SocketAddr` literal (e.g. a hostname, which `SocketAddr::parse`
    // cannot resolve). `TcpListener::bind` resolves hostnames at bind time
    // (main.rs), so `host:port` with a real hostname is a valid, deliberate
    // exposure (Docker/LAN) — fall back to splitting it ourselves.
    let Some((host, port)) = bind.rsplit_once(':') else {
        return BindScope::Unparseable;
    };
    if host.is_empty() || port.parse::<u16>().is_err() {
        return BindScope::Unparseable;
    }
    if host == "localhost" {
        return BindScope::Loopback;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_loopback() => BindScope::Loopback,
        Ok(_) => BindScope::External,
        // A hostname we can't resolve here still resolves for `TcpListener::bind`.
        Err(_) => BindScope::External,
    }
}

/// True for `localhost` / `127.0.0.1` / `::1`, with or without a `:port`.
fn host_is_local(host: &str) -> bool {
    let hostname = if let Some(rest) = host.strip_prefix('[') {
        // Bracketed IPv6: "[::1]" or "[::1]:7878" — take up to the closing ']'.
        rest.split(']').next().unwrap_or(rest)
    } else if host.matches(':').count() == 1 {
        // "host:port" — strip the port (a single colon can't be bare IPv6).
        host.split(':').next().unwrap_or(host)
    } else {
        // Bare hostname / IPv4, or a bare IPv6 literal like "::1" (2+ colons).
        host
    };
    matches!(hostname, "localhost" | "127.0.0.1" | "::1")
}

/// True for a local `Origin` header value (or the Tauri webview origins).
pub(crate) fn origin_is_local(origin: &str) -> bool {
    if origin == "tauri://localhost" || origin == "http://tauri.localhost" {
        return true;
    }
    let after_scheme = origin
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(origin);
    host_is_local(after_scheme)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_local_variants() {
        for h in [
            "127.0.0.1",
            "127.0.0.1:7878",
            "localhost",
            "localhost:7878",
            "::1",
            "[::1]",
            "[::1]:7878",
        ] {
            assert!(host_is_local(h), "expected local: {h}");
        }
    }

    #[test]
    fn host_non_local_rejected() {
        for h in [
            "evil.com",
            "evil.com:7878",
            "192.168.1.5:7878",
            "wenlan.evil.com",
        ] {
            assert!(!host_is_local(h), "expected non-local: {h}");
        }
    }

    #[test]
    fn origin_local_and_tauri_allowed() {
        for o in [
            "http://localhost:1420",
            "http://127.0.0.1:7878",
            "https://localhost",
            "tauri://localhost",
            "http://tauri.localhost",
        ] {
            assert!(origin_is_local(o), "expected local origin: {o}");
        }
    }

    #[test]
    fn origin_cross_site_and_null_rejected() {
        for o in ["https://evil.com", "http://attacker.test:1420", "null"] {
            assert!(!origin_is_local(o), "expected rejected origin: {o}");
        }
    }

    #[test]
    fn fetch_site_blocks_cases() {
        for (sec_fetch_site, origin, expected) in [
            (Some("cross-site"), None, true),
            (Some("same-site"), None, true),
            (Some("Cross-Site"), None, true),
            (Some("cross-site"), Some("tauri://localhost"), false),
            (Some("cross-site"), Some("http://localhost:1420"), false),
            (Some("cross-site"), Some("https://evil.com"), true),
            (Some("same-origin"), None, false),
            (Some("none"), None, false),
            (None, None, false),
            (Some("garbage"), None, false),
        ] {
            assert_eq!(
                fetch_site_blocks(sec_fetch_site, origin),
                expected,
                "sec_fetch_site={sec_fetch_site:?} origin={origin:?}"
            );
        }
    }

    /// The Host (DNS-rebinding) check is skipped ONLY for a bind value that
    /// resolves to a routable address or a hostname (`TcpListener::bind`
    /// resolves both). A loopback bind — or a value we cannot parse at all —
    /// must keep the check on, because merely setting the variable is not
    /// evidence the daemon was deliberately exposed.
    #[test]
    fn bind_scope_classifies_host_check_exemption() {
        for (bind, expected) in [
            (None, BindScope::Unset),
            (Some("127.0.0.1:7878"), BindScope::Loopback),
            (Some("[::1]:7878"), BindScope::Loopback),
            (Some("0.0.0.0:7878"), BindScope::External),
            (Some("192.168.1.5:7878"), BindScope::External),
            (Some("not-a-socket"), BindScope::Unparseable),
            (Some(""), BindScope::Unparseable),
            (Some("localhost:7878"), BindScope::Loopback),
            (Some("wenlan:7878"), BindScope::External),
            (Some("10.0.0.5:7878"), BindScope::External),
            (Some("myhost"), BindScope::Unparseable),
        ] {
            let scope = bind_scope(bind.map(Into::into));
            assert_eq!(scope, expected, "bind={bind:?}");
            assert_eq!(
                scope == BindScope::External,
                matches!(
                    bind,
                    Some("0.0.0.0:7878")
                        | Some("192.168.1.5:7878")
                        | Some("wenlan:7878")
                        | Some("10.0.0.5:7878")
                ),
                "only a routable bind or hostname may skip the Host check: bind={bind:?}"
            );
        }
    }
}
