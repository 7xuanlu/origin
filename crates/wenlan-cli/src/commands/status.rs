// SPDX-License-Identifier: Apache-2.0
//! `wenlan status` — show daemon, service, model, and key state.

use anyhow::Result;

use super::{service, setup};
use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

/// `format` is the resolved output format (Auto already collapsed in main).
/// `quiet` suppresses success output; errors still propagate via `?` to stderr.
/// We still hit the daemon under `quiet` to surface connection failures via exit code.
/// The JSON branch prints the same `"status": "unreachable"` body either way,
/// then returns the connect error so the process exits non-zero like every
/// other command does on a connect failure.
pub async fn run(client: &WenlanClient, format: ResolvedFormat, quiet: bool) -> Result<()> {
    if quiet {
        let _health = client.health().await?;
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => match client.health().await {
            Ok(health) => print_json(&health)?,
            Err(err) => {
                // The whole chain: the hint first, then the endpoint and the
                // transport cause, so a script reading this field sees both.
                let status = serde_json::json!({
                    "status": "unreachable",
                    "error": format!("{err:#}"),
                });
                print_json(&status)?;
                // Print the JSON scripts parse, but still exit non-zero like
                // every other command does on a connect failure.
                return Err(err);
            }
        },
        ResolvedFormat::Table => {
            println!("Wenlan runtime");
            service::print_status().await?;
            setup::print_runtime_status().await?;
        }
    }
    Ok(())
}
