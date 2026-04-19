// SPDX-License-Identifier: AGPL-3.0-only
use regex::Regex;
use std::sync::LazyLock;

struct PiiPattern {
    regex: Regex,
    label: &'static str,
}

static PII_PATTERNS: LazyLock<Vec<PiiPattern>> = LazyLock::new(|| {
    vec![
        PiiPattern {
            regex: Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap(),
            label: "CREDIT_CARD",
        },
        PiiPattern {
            regex: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            label: "SSN",
        },
        PiiPattern {
            regex: Regex::new(r"(?i)AKIA[0-9A-Z]{16}").unwrap(),
            label: "AWS_KEY",
        },
        PiiPattern {
            regex: Regex::new(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----").unwrap(),
            label: "PRIVATE_KEY",
        },
        PiiPattern {
            regex: Regex::new(r#"(?i)(?:api_key|apikey|api-key|secret_key|secret|token|password|passwd)\s*[=:]\s*["']?[A-Za-z0-9\-_.]{8,}["']?"#).unwrap(),
            label: "API_KEY",
        },
        PiiPattern {
            regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(),
            label: "EMAIL",
        },
        PiiPattern {
            regex: Regex::new(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b").unwrap(),
            label: "PHONE",
        },
    ]
});

/// Redact PII patterns from text, replacing matches with [REDACTED].
pub fn redact_pii(text: &str) -> String {
    let mut result = text.to_string();
    for pattern in PII_PATTERNS.iter() {
        result = pattern
            .regex
            .replace_all(&result, &format!("[REDACTED:{}]", pattern.label))
            .to_string();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_card_redaction() {
        let input = "My card is 4111-1111-1111-1111 and 5500 0000 0000 0004";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:CREDIT_CARD]"));
        assert!(!result.contains("4111"));
        assert!(!result.contains("5500"));
    }

    #[test]
    fn test_ssn_redaction() {
        let input = "SSN: 123-45-6789";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:SSN]"));
        assert!(!result.contains("123-45-6789"));
    }

    #[test]
    fn test_aws_key_redaction() {
        let input = "aws_access_key = AKIAIOSFODNN7EXAMPLE";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:AWS_KEY]"));
    }

    #[test]
    fn test_api_key_redaction() {
        let input = r#"api_key="sk-abc123def456ghi789""#;
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:API_KEY]"));
    }

    #[test]
    fn test_private_key_redaction() {
        let input =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQ...\n-----END RSA PRIVATE KEY-----";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:PRIVATE_KEY]"));
    }

    #[test]
    fn test_no_false_positives() {
        let input = "Regular text with a number 12345 and no PII";
        let result = redact_pii(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_email_redaction() {
        let input = "Contact us at user@example.com or admin.team@company.co.uk for help";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:EMAIL]"));
        assert!(!result.contains("user@example.com"));
        assert!(!result.contains("admin.team@company.co.uk"));
    }

    #[test]
    fn test_phone_redaction() {
        let input = "Call me at (555) 123-4567 or +1-800-555-0199";
        let result = redact_pii(input);
        assert!(result.contains("[REDACTED:PHONE]"));
        assert!(!result.contains("555) 123-4567"));
        assert!(!result.contains("800-555-0199"));
    }

    #[test]
    fn test_phone_no_false_positive_on_short_numbers() {
        // 5-6 digit numbers should NOT be redacted as phone numbers
        let input = "Error code 12345 and version 1.2.3";
        let result = redact_pii(input);
        assert!(!result.contains("[REDACTED:PHONE]"));
    }
}
