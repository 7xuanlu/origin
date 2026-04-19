// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const ACTIVITY_GAP_SECS: i64 = 1800; // 30-min inactivity → new activity

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    pub id: String,
    pub started_at: i64,
    pub ended_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivitySummary {
    pub id: String,
    pub started_at: i64,
    pub ended_at: i64,
    pub is_live: bool,
    pub app_names: Vec<String>,
}

impl Activity {
    pub fn new(now: i64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            started_at: now,
            ended_at: now,
        }
    }

    pub fn new_with_range(started_at: i64, ended_at: i64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            started_at,
            ended_at,
        }
    }

    pub fn to_summary(&self, is_live: bool) -> ActivitySummary {
        ActivitySummary {
            id: self.id.clone(),
            started_at: self.started_at,
            ended_at: self.ended_at,
            is_live,
            app_names: Vec::new(),
        }
    }
}

fn activities_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
        .join("activities.json")
}

pub fn load_activities() -> Vec<Activity> {
    let path = activities_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

pub fn save_activities(activities: &[Activity]) -> Result<(), OriginError> {
    let path = activities_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(activities)?;
    std::fs::write(&path, json)?;
    Ok(())
}
