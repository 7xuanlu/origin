// SPDX-License-Identifier: AGPL-3.0-only
//! Exercises the actual daemon contract the whole Sources design rests on.
//! Requires a real daemon >= v0.10.0 on :7878 (build v0.11.0 locally from
//! ../wenlan). Ignored by default; run with:
//!   cargo test -p wenlan-app --test sources_integration -- --ignored --nocapture

#[tokio::test]
#[ignore = "requires a live daemon >= v0.10.0 on :7878"]
async fn directory_source_indexes_within_scheduler_window() {
    let client = reqwest::Client::new();
    let base = "http://127.0.0.1:7878";

    // Version floor.
    let health: serde_json::Value = client
        .get(format!("{base}/api/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let version = health["version"].as_str().unwrap();
    let parts: Vec<u32> = version.split('.').map(|n| n.parse().unwrap_or(0)).collect();
    assert!(
        parts[0] > 0 || parts[1] >= 10,
        "daemon {version} is below the v0.10.0 floor"
    );

    // Register a temp dir with a markdown file.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("note.md"), "# Hello\n\nintegration body").unwrap();
    let path = dir.path().to_string_lossy().to_string();
    let reg = client
        .post(format!("{base}/api/sources"))
        .json(&serde_json::json!({ "source_type": "directory", "path": path }))
        .send()
        .await
        .unwrap();
    assert!(
        reg.status().is_success(),
        "register failed: {}",
        reg.status()
    );

    // Poll until memory_count climbs and last_sync populates (scheduler is 30s).
    let mut indexed = false;
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let sources: serde_json::Value = client
            .get(format!("{base}/api/sources"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if let Some(s) = sources
            .as_array()
            .and_then(|a| a.iter().find(|s| s["path"] == path))
        {
            if s["last_sync"].as_i64().is_some() && s["memory_count"].as_i64().unwrap_or(0) > 0 {
                indexed = true;
                break;
            }
        }
    }
    assert!(indexed, "source did not index within the scheduler window");
}
