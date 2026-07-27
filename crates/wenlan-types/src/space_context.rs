// SPDX-License-Identifier: Apache-2.0

use serde::de::{Error as _, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// Caller intent for a top-level write destination.
///
/// The wire shape stays ergonomic: an omitted field inherits context,
/// explicit JSON `null` selects Uncategorized, and a string names a Space.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WriteSpaceTarget {
    #[default]
    Inherit,
    Uncategorized,
    Named(String),
}

impl WriteSpaceTarget {
    pub fn is_inherit(&self) -> bool {
        matches!(self, Self::Inherit)
    }

    pub fn named(&self) -> Option<&str> {
        match self {
            Self::Named(name) => Some(name),
            Self::Inherit | Self::Uncategorized => None,
        }
    }

    pub fn as_deref(&self) -> Option<&str> {
        self.named()
    }
}

impl From<Option<String>> for WriteSpaceTarget {
    fn from(value: Option<String>) -> Self {
        match value {
            Some(name) => Self::Named(name),
            None => Self::Inherit,
        }
    }
}

impl From<String> for WriteSpaceTarget {
    fn from(value: String) -> Self {
        Self::Named(value)
    }
}

impl Serialize for WriteSpaceTarget {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Inherit | Self::Uncategorized => serializer.serialize_none(),
            Self::Named(name) => serializer.serialize_str(name),
        }
    }
}

impl<'de> Deserialize<'de> for WriteSpaceTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TargetVisitor;

        impl<'de> Visitor<'de> for TargetVisitor {
            type Value = WriteSpaceTarget;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("null or a non-empty Space name")
            }

            fn visit_none<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(WriteSpaceTarget::Uncategorized)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(WriteSpaceTarget::Uncategorized)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    return Err(D::Error::custom("Space name must not be empty"));
                }
                Ok(WriteSpaceTarget::Named(trimmed.to_string()))
            }
        }

        deserializer.deserialize_option(TargetVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteSpaceSource {
    Request,
    Header,
    Default,
    Uncategorized,
    Existing,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteOutcome {
    Created,
    ResolvedExisting,
    AttachedExisting,
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::{WriteOutcome, WriteSpaceSource, WriteSpaceTarget};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct WriteFixture {
        #[serde(default, skip_serializing_if = "WriteSpaceTarget::is_inherit")]
        space: WriteSpaceTarget,
    }

    #[test]
    fn write_target_distinguishes_missing_null_and_named_space() {
        let missing: WriteFixture = serde_json::from_str("{}").unwrap();
        let uncategorized: WriteFixture = serde_json::from_str(r#"{"space":null}"#).unwrap();
        let named: WriteFixture = serde_json::from_str(r#"{"space":"Work"}"#).unwrap();

        assert_eq!(missing.space, WriteSpaceTarget::Inherit);
        assert_eq!(uncategorized.space, WriteSpaceTarget::Uncategorized);
        assert_eq!(named.space, WriteSpaceTarget::Named("Work".to_string()));
        assert_eq!(serde_json::to_string(&missing).unwrap(), "{}");
        assert_eq!(
            serde_json::to_string(&uncategorized).unwrap(),
            r#"{"space":null}"#
        );
        assert_eq!(
            serde_json::to_string(&named).unwrap(),
            r#"{"space":"Work"}"#
        );
    }

    #[test]
    fn receipt_enums_accept_future_values() {
        assert_eq!(
            serde_json::from_str::<WriteSpaceSource>(r#""future_source""#).unwrap(),
            WriteSpaceSource::Unknown
        );
        assert_eq!(
            serde_json::from_str::<WriteOutcome>(r#""future_outcome""#).unwrap(),
            WriteOutcome::Unknown
        );
    }
}
