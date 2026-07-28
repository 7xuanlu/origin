// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};
use wenlan_core::db::{LegacyBriefItem, MemoryDB};
use wenlan_types::{Brief, BriefItem, BriefItemState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedLegacyBrief {
    pub last_session_summary: String,
    pub last_handoff_at: Option<i64>,
    pub items: Vec<LegacyBriefItem>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LegacyImportReport {
    pub imported: usize,
    pub skipped_existing: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    LastSession,
    Active,
    Backlog,
}

pub fn parse_legacy_status(
    content: &str,
    fallback_added_at: i64,
) -> Result<ParsedLegacyBrief, String> {
    let mut section = None;
    let mut saw_known_section = false;
    let mut summary_lines = Vec::new();
    let mut items = Vec::new();
    let mut last_handoff_at = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("## Last session") {
            section = Some(Section::LastSession);
            saw_known_section = true;
            last_handoff_at = heading_date(trimmed).and_then(parse_date);
            continue;
        }
        if trimmed == "## Active" {
            section = Some(Section::Active);
            saw_known_section = true;
            continue;
        }
        if trimmed == "## Backlog" {
            section = Some(Section::Backlog);
            saw_known_section = true;
            continue;
        }
        if trimmed.starts_with("## ") {
            section = None;
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with("<!--") {
            continue;
        }

        match section {
            Some(Section::LastSession) => {
                let text = trimmed.strip_prefix("- ").unwrap_or(trimmed).trim();
                if !text.is_empty() {
                    summary_lines.push(text.to_string());
                }
            }
            Some(Section::Active) | Some(Section::Backlog) => {
                let Some(raw_item) = trimmed.strip_prefix("- ") else {
                    continue;
                };
                let state = if section == Some(Section::Active) {
                    BriefItemState::Active
                } else {
                    BriefItemState::Backlog
                };
                items.push(parse_legacy_item(raw_item, state, fallback_added_at));
            }
            None => {}
        }
    }

    if !saw_known_section {
        return Err("no Wenlan legacy Brief sections found".into());
    }

    Ok(ParsedLegacyBrief {
        last_session_summary: summary_lines.join("\n"),
        last_handoff_at,
        items,
    })
}

fn heading_date(heading: &str) -> Option<&str> {
    let start = heading.rfind('(')?;
    let date = heading.get(start + 1..heading.len().checked_sub(1)?)?;
    heading.ends_with(')').then_some(date)
}

fn parse_date(value: &str) -> Option<i64> {
    chrono::NaiveDate::parse_from_str(value.trim(), "%Y-%m-%d")
        .ok()?
        .and_hms_opt(0, 0, 0)
        .map(|date_time| date_time.and_utc().timestamp())
}

fn parse_legacy_item(
    raw_item: &str,
    state: BriefItemState,
    fallback_added_at: i64,
) -> LegacyBriefItem {
    let mut text = raw_item.trim().to_string();
    let gate = extract_suffix(&mut text, " (gated: ")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let added_at = match extract_suffix_candidate(&text, " (added ") {
        Some((prefix, value)) => match parse_date(value) {
            Some(timestamp) => {
                text = prefix.trim().to_string();
                timestamp
            }
            None => fallback_added_at,
        },
        None => fallback_added_at,
    };

    if text.is_empty() {
        text = raw_item.trim().to_string();
    }
    if text.is_empty() {
        text = "-".into();
    }

    LegacyBriefItem {
        text,
        state,
        added_at,
        gate,
    }
}

fn extract_suffix(text: &mut String, marker: &str) -> Option<String> {
    let (prefix, value) = extract_suffix_candidate(text, marker)?;
    let value = value.to_string();
    *text = prefix.trim().to_string();
    Some(value)
}

fn extract_suffix_candidate<'a>(text: &'a str, marker: &str) -> Option<(&'a str, &'a str)> {
    if !text.ends_with(')') {
        return None;
    }
    let start = text.rfind(marker)?;
    let prefix = text.get(..start)?;
    let value = text.get(start + marker.len()..text.len() - 1)?;
    Some((prefix, value))
}

pub async fn import_legacy_status_files(
    db: &MemoryDB,
    status_root: &Path,
) -> anyhow::Result<LegacyImportReport> {
    let mut report = LegacyImportReport::default();
    let entries = match std::fs::read_dir(status_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(report),
        Err(error) => return Err(error.into()),
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("md"))
        .collect::<Vec<_>>();
    paths.sort();

    let fallback_added_at = chrono::Utc::now().timestamp();
    for path in paths {
        let Some(space_name) = path.file_stem().and_then(|value| value.to_str()) else {
            report.warnings.push(format!(
                "ignored non-UTF-8 legacy status path {}",
                path.display()
            ));
            continue;
        };
        if space_name.is_empty() {
            continue;
        }
        if db.get_brief_by_space_name(space_name).await?.is_some() {
            report.skipped_existing += 1;
            continue;
        }

        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) => {
                report
                    .warnings
                    .push(format!("could not read {}: {error}", path.display()));
                continue;
            }
        };
        let parsed = match parse_legacy_status(&content, fallback_added_at) {
            Ok(parsed) => parsed,
            Err(error) => {
                report
                    .warnings
                    .push(format!("could not parse {}: {error}", path.display()));
                continue;
            }
        };
        if db
            .import_legacy_brief(
                space_name,
                &parsed.last_session_summary,
                parsed.last_handoff_at,
                &parsed.items,
            )
            .await?
        {
            report.imported += 1;
        } else {
            report.skipped_existing += 1;
        }
    }

    Ok(report)
}

pub fn render_brief_receipt(brief: &Brief, generated_at: i64) -> String {
    let generated = format_timestamp(generated_at);
    let last_session_date = brief
        .last_handoff_at
        .map(format_date)
        .unwrap_or_else(|| format_date(generated_at));
    let mut output = format!(
        "<!-- Generated by Wenlan from daemon-authoritative Brief state. -->\n\
         <!-- This file is a receipt for inspection, not an input. -->\n\
         # {} — Brief\n\n\
         Generated: {}\n\n\
         ## Last session ({})\n",
        brief.space, generated, last_session_date
    );

    if brief.last_session_summary.trim().is_empty() {
        output.push_str("_No session summary._\n");
    } else {
        for line in brief.last_session_summary.lines() {
            let line = line.trim();
            if !line.is_empty() {
                output.push_str("- ");
                output.push_str(line);
                output.push('\n');
            }
        }
    }

    output.push_str("\n## Active\n");
    render_items(&mut output, &brief.active);
    output.push_str("\n## Backlog\n");
    render_items(&mut output, &brief.backlog);
    output
}

fn render_items(output: &mut String, items: &[BriefItem]) {
    if items.is_empty() {
        output.push_str("_None._\n");
        return;
    }
    for item in items {
        output.push_str("- ");
        output.push_str(&single_line(&item.text));
        output.push_str(" (added ");
        output.push_str(&format_date(item.added_at));
        output.push(')');
        if let Some(gate) = item
            .gate
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            output.push_str(" (gated: ");
            output.push_str(&single_line(gate));
            output.push(')');
        }
        output.push('\n');
    }
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn format_date(timestamp: i64) -> String {
    chrono::DateTime::from_timestamp(timestamp, 0)
        .map(|value| value.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn format_timestamp(timestamp: i64) -> String {
    chrono::DateTime::from_timestamp(timestamp, 0)
        .map(|value| value.to_rfc3339())
        .unwrap_or_else(|| "unknown".into())
}

fn encoded_space_filename(space: &str) -> String {
    let mut encoded = String::new();
    for byte in space.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(*byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    if encoded.is_empty() {
        "space".into()
    } else {
        encoded
    }
}

pub fn project_brief_receipt(
    status_root: &Path,
    brief: &Brief,
    generated_at: i64,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(status_root)?;
    let filename = format!("{}.md", encoded_space_filename(&brief.space));
    let destination = status_root.join(&filename);
    let temporary = status_root.join(format!(".{filename}.{}.tmp", uuid::Uuid::new_v4()));
    std::fs::write(&temporary, render_brief_receipt(brief, generated_at))?;
    if let Err(error) = std::fs::rename(&temporary, &destination) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(destination)
}
