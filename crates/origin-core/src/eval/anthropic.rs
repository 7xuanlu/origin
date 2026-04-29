// SPDX-License-Identifier: AGPL-3.0-only
//! Anthropic Messages API and Batch API client for eval.

use std::collections::HashMap;

/// Call the Anthropic API directly via reqwest. Returns response text.
/// Much faster than `claude -p` (no process spawn overhead) and costs ~$0.001/call with Haiku.
pub async fn call_anthropic_api(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
) -> Result<String, String> {
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": messages
    });
    if let Some(sys) = system_prompt {
        body["system"] = serde_json::json!(sys);
    }

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {e}"))?;
    let answer = json["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("")
        .to_string();
    Ok(answer)
}

/// Estimate batch cost in USD (Haiku batch pricing: $0.50/MTok input, $2.50/MTok output).
pub fn estimate_batch_cost(prompts: &[(String, String, Option<String>, usize)]) -> f64 {
    let input_tokens: usize = prompts
        .iter()
        .map(|(_, prompt, sys, _)| {
            let sys_len = sys.as_ref().map(|s| s.len()).unwrap_or(0);
            (prompt.len() + sys_len) / 4 // rough estimate: 4 chars per token
        })
        .sum();
    let output_tokens: usize = prompts.iter().map(|(_, _, _, max_tok)| *max_tok).sum();
    let input_cost = input_tokens as f64 * 0.50 / 1_000_000.0;
    let output_cost = output_tokens as f64 * 2.50 / 1_000_000.0;
    input_cost + output_cost
}

/// Submit a batch of prompts to the Anthropic Batch API.
/// Returns the batch ID for polling.
///
/// Set `cost_cap_usd` to limit spending. Returns Err if estimated cost exceeds cap.
pub async fn submit_batch(
    client: &reqwest::Client,
    api_key: &str,
    requests: Vec<(String, String, Option<String>, usize)>, // (custom_id, prompt, system, max_tokens)
    model: &str,
    cost_cap_usd: f64,
) -> Result<String, String> {
    debug_assert!(
        cost_cap_usd <= 100.0,
        "cost_cap_usd suspiciously high: ${cost_cap_usd}"
    );
    let est_cost = estimate_batch_cost(&requests);
    eprintln!(
        "[batch] Estimated cost: ${:.3} ({} requests, cap: ${:.2})",
        est_cost,
        requests.len(),
        cost_cap_usd,
    );
    if est_cost > cost_cap_usd {
        return Err(format!(
            "Estimated cost ${:.3} exceeds cap ${:.2}. Reduce questions or raise cap.",
            est_cost, cost_cap_usd
        ));
    }
    let batch_requests: Vec<serde_json::Value> = requests
        .into_iter()
        .map(|(id, prompt, system, max_tokens)| {
            let mut params = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]
            });
            if let Some(sys) = system {
                params["system"] = serde_json::json!(sys);
            }
            serde_json::json!({
                "custom_id": id,
                "params": params
            })
        })
        .collect();

    let body = serde_json::json!({ "requests": batch_requests });

    let resp = client
        .post("https://api.anthropic.com/v1/messages/batches")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("batch submit failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("batch API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
    json["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "no batch id in response".to_string())
}

/// Poll a batch until it reaches "ended" status. Returns the results_url.
pub async fn poll_batch(
    client: &reqwest::Client,
    api_key: &str,
    batch_id: &str,
) -> Result<String, String> {
    let url = format!("https://api.anthropic.com/v1/messages/batches/{}", batch_id);
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;

        let resp = client
            .get(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| format!("poll failed: {e}"))?;

        let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
        let status = json["processing_status"].as_str().unwrap_or("unknown");

        let succeeded = json["request_counts"]["succeeded"].as_u64().unwrap_or(0);
        let processing = json["request_counts"]["processing"].as_u64().unwrap_or(0);
        let errored = json["request_counts"]["errored"].as_u64().unwrap_or(0);
        eprintln!(
            "[batch] status={}, succeeded={}, processing={}, errored={}",
            status, succeeded, processing, errored
        );

        if status == "ended" {
            return json["results_url"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "batch ended but no results_url".to_string());
        }
    }
}

/// Download batch results and return a map of custom_id -> response text.
pub async fn download_batch_results(
    client: &reqwest::Client,
    api_key: &str,
    results_url: &str,
) -> Result<HashMap<String, String>, String> {
    let resp = client
        .get(results_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;

    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;
    let mut results = HashMap::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let json: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("parse result line: {e}"))?;

        let custom_id = json["custom_id"].as_str().unwrap_or("").to_string();
        let result_type = json["result"]["type"].as_str().unwrap_or("");

        if result_type == "succeeded" {
            let answer = json["result"]["message"]["content"]
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|block| block["text"].as_str())
                .unwrap_or("")
                .to_string();
            results.insert(custom_id, answer);
        } else {
            eprintln!("[batch] {} result: {}", custom_id, result_type);
        }
    }

    Ok(results)
}
