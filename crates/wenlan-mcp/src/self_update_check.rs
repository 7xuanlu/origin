use semver::Version;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

const CACHE_TTL: Duration = Duration::from_secs(24 * 3600);
const LATEST_RELEASE_URL: &str = "https://github.com/7xuanlu/wenlan/releases/latest";

/// Process-wide in-memory fallback for environments where on-disk cache writes
/// fail (locked-down sandboxes, missing dirs::cache_dir, etc). Without this,
/// `store_cache` would silently no-op and every invocation in the same
/// long-lived process (e.g. an MCP server hosting many sessions) would re-hit
/// the GitHub release page on every start.
static MEMORY_FALLBACK: Mutex<Option<CacheEntry>> = Mutex::new(None);

#[derive(Serialize, Deserialize, Debug, Clone)]
struct CacheEntry {
    latest_tag: String,
    checked_at_secs: u64,
}

fn cache_path() -> Option<PathBuf> {
    // WENLAN_MCP_CACHE_DIR override exists so tests can point at a temp dir
    // instead of polluting the user's real cache (~/Library/Caches/wenlan-mcp/...).
    let base = std::env::var_os("WENLAN_MCP_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| dirs::cache_dir().map(|d| d.join("wenlan-mcp")))?;
    std::fs::create_dir_all(&base).ok()?;
    Some(base.join("version-check.json"))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_cache() -> Option<CacheEntry> {
    if let Some(path) = cache_path() {
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(entry) = serde_json::from_slice::<CacheEntry>(&bytes) {
                if now_secs().saturating_sub(entry.checked_at_secs) < CACHE_TTL.as_secs() {
                    return Some(entry);
                }
            }
        }
    }
    // Fall back to the in-memory cache if disk read failed or was stale.
    let guard = MEMORY_FALLBACK.lock().ok()?;
    let entry = guard.as_ref()?;
    if now_secs().saturating_sub(entry.checked_at_secs) < CACHE_TTL.as_secs() {
        Some(entry.clone())
    } else {
        None
    }
}

fn store_cache(entry: &CacheEntry) {
    if let Some(path) = cache_path() {
        if let Ok(bytes) = serde_json::to_vec(entry) {
            if std::fs::write(&path, bytes).is_ok() {
                return;
            }
        }
    }
    // Disk write failed (no cache_dir, read-only FS, etc) — fall back to memory.
    if let Ok(mut guard) = MEMORY_FALLBACK.lock() {
        *guard = Some(entry.clone());
    }
}

/// The version a GitHub "latest release" redirect points at, without its `v`.
/// A repository with no releases redirects to the releases list instead, which
/// yields `None`.
fn version_from_location(location: &str) -> Option<String> {
    let tag = location.rsplit_once("/releases/tag/")?.1;
    let tag = tag.split(['?', '#']).next().unwrap_or(tag);
    if tag.is_empty() || tag.contains('/') {
        return None;
    }
    Some(tag.trim_start_matches('v').to_string())
}

/// The public releases page redirects to the latest tag. Unlike the API it has
/// no limit of 60 anonymous calls per hour per IP address, a budget the
/// install scripts share with this check on the user's network.
async fn fetch_latest_tag() -> Option<String> {
    let resp = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .ok()?
        .get(LATEST_RELEASE_URL)
        .header(
            "User-Agent",
            concat!("wenlan-mcp/", env!("CARGO_PKG_VERSION")),
        )
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?;
    let location = resp
        .headers()
        .get(reqwest::header::LOCATION)?
        .to_str()
        .ok()?;
    version_from_location(location)
}

/// Check for a newer published release. Returns Some(message) if behind,
/// None otherwise. Uses a 24h on-disk cache so this never adds startup latency
/// after the first run.
pub async fn check() -> Option<String> {
    let mcp_version = env!("CARGO_PKG_VERSION");
    let mcp = Version::parse(mcp_version).ok()?;

    let latest_tag = match load_cache() {
        Some(entry) => entry.latest_tag,
        None => {
            let tag = fetch_latest_tag().await?;
            store_cache(&CacheEntry {
                latest_tag: tag.clone(),
                checked_at_secs: now_secs(),
            });
            tag
        }
    };

    let latest = Version::parse(&latest_tag).ok()?;
    if latest > mcp {
        Some(format!(
            "A newer wenlan-mcp is available (v{latest}, you are on v{mcp}). \
             Run `brew upgrade wenlan-mcp`, `npm update -g wenlan-mcp`, or \
             `cargo install wenlan-mcp`, whichever installed it."
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Tests touch process-wide state (`WENLAN_MCP_CACHE_DIR` env var + the
    /// resulting on-disk cache file). Cargo runs tests in parallel by default,
    /// so we serialize the disk-touching tests through this lock. The env
    /// override is per-test (set inside the lock) so each disk-test gets its
    /// own temp dir — no pollution of the user's real cache.
    static CACHE_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn the_latest_version_comes_from_the_release_page_redirect() {
        assert_eq!(
            version_from_location("https://github.com/7xuanlu/wenlan/releases/tag/v0.17.0")
                .as_deref(),
            Some("0.17.0")
        );
        assert_eq!(
            version_from_location("/7xuanlu/wenlan/releases/tag/v0.18.0-rc.1?raw=1").as_deref(),
            Some("0.18.0-rc.1")
        );
        assert_eq!(
            version_from_location("https://github.com/7xuanlu/wenlan/releases"),
            None
        );
        assert_eq!(version_from_location(""), None);
    }

    fn set_temp_cache(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("wenlan-mcp-test-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("WENLAN_MCP_CACHE_DIR", &dir);
        dir
    }

    #[test]
    fn cache_path_under_user_cache_dir() {
        // No override → falls back to dirs::cache_dir().
        let _g = CACHE_LOCK.lock().unwrap();
        std::env::remove_var("WENLAN_MCP_CACHE_DIR");
        let p = cache_path().expect("cache dir should resolve on this platform");
        assert!(p.ends_with("wenlan-mcp/version-check.json"), "got {p:?}");
    }

    #[test]
    fn cache_round_trip_within_ttl() {
        let _g = CACHE_LOCK.lock().unwrap();
        let dir = set_temp_cache("round-trip");
        let entry = CacheEntry {
            latest_tag: "9.9.9".to_string(),
            checked_at_secs: now_secs(),
        };
        store_cache(&entry);
        let loaded = load_cache().expect("cache should load");
        assert_eq!(loaded.latest_tag, "9.9.9");
        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("WENLAN_MCP_CACHE_DIR");
    }

    #[test]
    fn cache_expires_after_ttl() {
        let _g = CACHE_LOCK.lock().unwrap();
        let dir = set_temp_cache("expires");
        let entry = CacheEntry {
            latest_tag: "9.9.9".to_string(),
            checked_at_secs: now_secs().saturating_sub(CACHE_TTL.as_secs() + 60),
        };
        store_cache(&entry);
        assert!(
            load_cache().is_none(),
            "expired entry should not be returned"
        );
        let _ = std::fs::remove_dir_all(&dir);
        std::env::remove_var("WENLAN_MCP_CACHE_DIR");
    }
}
