//! Wire types for the Wenlan HTTP API.
//!
//! All types live in the published `wenlan-types` crate. This module exists
//! only to provide a stable import path (`crate::types::...`) during the
//! cross-repo refactor; later PRs may remove the module entirely and import
//! `wenlan_types::*` at call sites directly.

pub use wenlan_types::memory::{RejectionRecord, SearchResult, Space};
pub use wenlan_types::requests::{
    ChatContextRequest, CreateConceptRequest, CreateEntityRequest, CreateRelationRequest,
    ListMemoriesRequest, SearchMemoryRequest, StoreMemoryRequest,
};
pub use wenlan_types::responses::{
    AcceptRefinementResponse, ChatContextResponse, ConfirmResponse, CreateEntityResponse,
    CreatePageResponse, CreateRelationResponse, DeleteResponse, ListMemoriesResponse,
    ListMemoryRevisionsResponse, ListPageRevisionsResponse, ListRefinementsResponse,
    RejectRefinementResponse, RevisionAcceptResponse, RevisionDismissResponse,
    SearchMemoryResponse, StoreMemoryResponse,
};
pub use wenlan_types::PageSourceWithMemory;
