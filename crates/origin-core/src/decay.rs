// SPDX-License-Identifier: AGPL-3.0-only
use crate::sources::StabilityTier;

/// Compute recency boost: 1.0 / (1.0 + decay_rate * days_since_last_access)
pub fn recency_boost(decay_rate: f64, days_since_last_access: f64) -> f64 {
    1.0 / (1.0 + decay_rate * days_since_last_access)
}

/// Compute access boost: 1.0 + ln(1 + access_count) * 0.1
pub fn access_boost(access_count: u64) -> f64 {
    1.0 + (1.0 + access_count as f64).ln() * 0.1
}

/// Compute effective score (search-time): base_score * recency * access
pub fn effective_score(
    base_score: f64,
    decay_rate: f64,
    days_since_last_access: f64,
    access_count: u64,
) -> f64 {
    base_score * recency_boost(decay_rate, days_since_last_access) * access_boost(access_count)
}

/// Compute effective confidence (stored): confidence * recency * access
pub fn effective_confidence(
    confidence: f64,
    decay_rate: f64,
    days_since_last_access: f64,
    access_count: u64,
) -> f64 {
    confidence * recency_boost(decay_rate, days_since_last_access) * access_boost(access_count)
}

/// Get decay rate for a stability tier. Confirmed/pinned memories return 0.0 (immune).
pub fn decay_rate_for(
    tier: &StabilityTier,
    confirmed: bool,
    pinned: bool,
    cfg: &crate::tuning::ConfidenceConfig,
) -> f64 {
    if confirmed || pinned {
        return 0.0;
    }
    crate::sources::decay_rate(tier, cfg)
}

/// Parse an ISO-8601 or integer timestamp and return days elapsed since now.
/// Returns 0.0 if the timestamp is None or unparseable.
pub fn days_since(timestamp: Option<&str>, now_epoch: i64) -> f64 {
    let epoch = match timestamp {
        Some(ts) => {
            // Try integer epoch first
            if let Ok(e) = ts.parse::<i64>() {
                e
            } else if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(ts) {
                dt.timestamp()
            } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S") {
                dt.and_utc().timestamp()
            } else {
                return 0.0;
            }
        }
        None => return 0.0,
    };
    let secs = (now_epoch - epoch).max(0) as f64;
    secs / 86400.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recency_boost_no_decay() {
        // Confirmed/pinned: rate=0 → boost=1.0 regardless of time
        assert_eq!(recency_boost(0.0, 365.0), 1.0);
    }

    #[test]
    fn test_recency_boost_ephemeral_14_days() {
        // Ephemeral after 14 days: 1/(1+0.05*14) = 1/1.7 ≈ 0.588
        let boost = recency_boost(0.05, 14.0);
        assert!((boost - 0.588).abs() < 0.01);
    }

    #[test]
    fn test_recency_boost_protected_365_days() {
        // Protected after 365 days: 1/(1+0.001*365) = 1/1.365 ≈ 0.733
        let boost = recency_boost(0.001, 365.0);
        assert!((boost - 0.733).abs() < 0.01);
    }

    #[test]
    fn test_access_boost_zero() {
        // 0 accesses: 1 + ln(1) * 0.1 = 1.0
        assert_eq!(access_boost(0), 1.0);
    }

    #[test]
    fn test_access_boost_ten() {
        // 10 accesses: 1 + ln(11) * 0.1 ≈ 1.2397
        let boost = access_boost(10);
        assert!((boost - 1.2397).abs() < 0.01);
    }

    #[test]
    fn test_effective_score_combined() {
        let score = effective_score(0.8, 0.05, 14.0, 5);
        let expected = 0.8 * recency_boost(0.05, 14.0) * access_boost(5);
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn test_effective_confidence_combined() {
        let conf = effective_confidence(0.7, 0.01, 30.0, 3);
        let expected = 0.7 * recency_boost(0.01, 30.0) * access_boost(3);
        assert!((conf - expected).abs() < 0.001);
    }

    #[test]
    fn test_decay_rate_for_confirmed_immune() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        assert_eq!(
            decay_rate_for(&StabilityTier::Ephemeral, true, false, &cfg),
            0.0
        );
        assert_eq!(
            decay_rate_for(&StabilityTier::Standard, false, true, &cfg),
            0.0
        );
    }

    #[test]
    fn test_decay_rate_for_by_tier() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        assert_eq!(
            decay_rate_for(&StabilityTier::Protected, false, false, &cfg),
            0.001
        );
        assert_eq!(
            decay_rate_for(&StabilityTier::Standard, false, false, &cfg),
            0.01
        );
        assert_eq!(
            decay_rate_for(&StabilityTier::Ephemeral, false, false, &cfg),
            0.05
        );
    }

    #[test]
    fn test_days_since_integer_timestamp() {
        let now = 1_000_000;
        let ts = "913600"; // 1 day ago = 86400 secs
        let days = days_since(Some(ts), now);
        assert!((days - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_days_since_none() {
        assert_eq!(days_since(None, 1_000_000), 0.0);
    }

    #[test]
    fn test_days_since_sqlite_datetime() {
        // SQLite datetime format: "2026-03-01 10:00:00"
        let ts = "2026-03-01 10:00:00";
        let now_dt =
            chrono::NaiveDateTime::parse_from_str("2026-03-15 10:00:00", "%Y-%m-%d %H:%M:%S")
                .unwrap();
        let now = now_dt.and_utc().timestamp();
        let days = days_since(Some(ts), now);
        assert!((days - 14.0).abs() < 0.01);
    }
}
