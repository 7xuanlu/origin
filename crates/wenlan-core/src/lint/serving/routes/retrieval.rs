use super::{
    Capability::*, CrossScopePolicy::*, Method::*, ScopeBinding::*,
    SelectionGate::NotApplicable as NoGate, SelectorPrecedence::*, SensitiveReadRoute,
};

#[rustfmt::skip]
pub(super) const ROUTES: &[SensitiveReadRoute] = &[
    row!(Post,"/api/search","memory_search",BodyThenHeader,CallerAssertedAgentTrust,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Post,"/api/context","memory_context",BodyThenHeader,CallerAssertedAgentTrust,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Post,"/api/brief","space_brief",BodyThenHeader,CallerAssertedAgentTrust,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/retrievals/recent","retrieval_activity",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/recent","memory_activity",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/unconfirmed","memory_list",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Post,"/api/memory/search","memory_search",BodyThenHeader,CallerAssertedAgentTrust,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Post,"/api/memory/list","memory_list",BodyThenHeader,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/nurture","memory_cards",QueryThenHeader,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/pinned","memory_list",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
];
