use super::{
    Capability::*,
    CrossScopePolicy::*,
    Method::*,
    ScopeBinding::*,
    SelectionGate::{NotApplicable as NoGate, SingleId404},
    SelectorPrecedence::{
        BodyThenHeader, HeaderOnly, NotApplicable as NoSelector, QueryThenHeader,
    },
    SensitiveReadRoute,
};

#[rustfmt::skip]
pub(super) const ROUTES: &[SensitiveReadRoute] = &[
    row!(Get,"/api/profile","profile",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/agents","agent_list",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/agents/{name}","agent_detail",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Post,"/api/memory/entities/list","entity_list",BodyThenHeader,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
    row!(Post,"/api/memory/entities/search","entity_search",BodyThenHeader,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/entities/{entity_id}","entity_detail",HeaderOnly,UnauthenticatedLocal,EntitySpace,SingleId404,Rejected,Forbidden),
    row!(Get,"/api/memory/stats","memory_stats",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,AggregateOnly),
    row!(Get,"/api/home-stats","home_stats_with_memory_rows",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/memory/entity-suggestions","entity_suggestions",HeaderOnly,UnauthenticatedLocal,MemorySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/spaces","space_list",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/sources","source_list",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/profile/narrative","profile_narrative",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/knowledge/recent-relations","relation_list",HeaderOnly,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/knowledge/count","knowledge_count",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,AggregateOnly),
    row!(Get,"/api/onboarding/milestones","onboarding_state",NoSelector,UnauthenticatedLocal,Global,NoGate,NotApplicable,GlobalRead),
    row!(Get,"/api/communities","community_list",QueryThenHeader,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/communities/members","community_members",QueryThenHeader,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/communities/page-assignments","community_page_assignments",QueryThenHeader,UnauthenticatedLocal,PageWorkspace,NoGate,Rejected,Forbidden),
    row!(Get,"/api/communities/proposals","community_proposals",QueryThenHeader,UnauthenticatedLocal,EntitySpace,NoGate,Rejected,Forbidden),
];
