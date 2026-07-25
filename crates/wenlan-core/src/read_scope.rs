// SPDX-License-Identifier: Apache-2.0

use crate::{db::MemoryDB, WenlanError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadScope {
    Global,
    Space(String),
    Uncategorized,
}

impl ReadScope {
    pub fn matches(&self, binding: Option<&str>) -> bool {
        match self {
            Self::Global => true,
            Self::Space(space) => binding == Some(space.as_str()),
            Self::Uncategorized => {
                binding.is_none() || binding == Some(crate::db::UNFILED_SPACE_ID)
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReadScopeResolveError {
    #[error("unknown Space: {0}")]
    Unknown(String),
    #[error("uncategorized is ambiguous with a registered Space")]
    AmbiguousUncategorized,
    #[error(transparent)]
    Store(#[from] WenlanError),
}

pub async fn resolve_read_scope(
    db: &MemoryDB,
    raw: Option<&str>,
) -> Result<ReadScope, ReadScopeResolveError> {
    let Some(name) = raw.map(str::trim).filter(|name| !name.is_empty()) else {
        return Ok(ReadScope::Global);
    };

    let registered = db.get_space(name).await?;
    if name == "uncategorized" {
        return if registered.is_some() {
            Err(ReadScopeResolveError::AmbiguousUncategorized)
        } else {
            Ok(ReadScope::Uncategorized)
        };
    }

    if registered.is_some() {
        Ok(ReadScope::Space(name.to_string()))
    } else {
        Err(ReadScopeResolveError::Unknown(name.to_string()))
    }
}
