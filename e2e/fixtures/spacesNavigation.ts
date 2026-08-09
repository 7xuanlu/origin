// SPDX-License-Identifier: AGPL-3.0-only
import type {
  DistillReviewResponse,
  Entity,
  EntityDetail,
  MemoryItem,
  Page as KnowledgePage,
  RefinementProposalSummary,
  Space,
} from "../../src/lib/tauri";

export type SpacesNavigationFixture = {
  readonly spaces: readonly Space[];
  readonly pages: readonly KnowledgePage[];
  readonly entities: readonly Entity[];
  readonly entityDetails: readonly EntityDetail[];
  readonly memories: readonly MemoryItem[];
  readonly distillReview: DistillReviewResponse;
  readonly refinements: readonly RefinementProposalSummary[];
};

const NOW_SECONDS = 1_783_728_000;

function makePage(
  id: string,
  title: string,
  lastModified: string,
  staleReason?: string,
): KnowledgePage {
  return {
    id,
    title,
    summary: `${title} summary`,
    content: `# ${title}\n\nDeterministic content for the integrated Wenlan journey.`,
    entity_id: null,
    domain: "Wenlan",
    space: "Wenlan",
    source_memory_ids: [`memory-${id}`],
    version: 1,
    status: "active",
    created_at: "2026-07-01T00:00:00Z",
    last_compiled: lastModified,
    last_modified: lastModified,
    ...(staleReason ? { stale_reason: staleReason } : {}),
  };
}

function makeMemory(index: number): MemoryItem {
  return {
    source_id: `memory-${index}`,
    title: index === 0 ? "Fixture architecture" : `Fixture memory ${index}`,
    content: "Typed fixtures keep rendered browser journeys deterministic.",
    summary: null,
    memory_type: index % 3 === 0 ? "decision" : "fact",
    domain: "Wenlan",
    space: "Wenlan",
    source_agent: "codex",
    confidence: 0.9,
    confirmed: true,
    pinned: index === 0,
    supersedes: null,
    last_modified: NOW_SECONDS - index * 60,
    chunk_count: 1,
    access_count: index,
    is_recap: false,
    stability: "confirmed",
  };
}

const adaEntity: Entity = {
  id: "entity-ada", name: "Ada Lovelace", entity_type: "person", domain: "Wenlan",
  space: "Wenlan", source_agent: "research-agent", confidence: 0.87, confirmed: false,
  created_at: NOW_SECONDS - 604_800, updated_at: NOW_SECONDS - 259_200,
};

const entities: readonly Entity[] = [
  adaEntity,
  {
    id: "entity-babbage", name: "Charles Babbage", entity_type: "person", domain: "Wenlan",
    space: "Wenlan", source_agent: "research-agent", confidence: 0.91, confirmed: true,
    created_at: NOW_SECONDS - 691_200, updated_at: NOW_SECONDS - 345_600,
  },
  ...["Grace Hopper", "Alan Turing", "Edsger Dijkstra", "Margaret Hamilton", "Barbara Liskov"]
    .map((name, index): Entity => ({
      id: `entity-${index + 3}`, name, entity_type: "person", domain: "Wenlan", space: "Wenlan",
      source_agent: "research-agent", confidence: 0.8 - index * 0.03, confirmed: true,
      created_at: NOW_SECONDS - (index + 8) * 86_400, updated_at: NOW_SECONDS - (index + 5) * 86_400,
    })),
];

export function createSpacesNavigationFixture(): SpacesNavigationFixture {
  const pages = [
    makePage("page-architecture", "Fixture architecture", "2026-07-10T12:00:00Z"),
    { ...makePage("page-browser", "Ada Lovelace", "2026-07-10T11:00:00Z"), entity_id: "entity-ada" },
    { ...makePage("page-errors", "Why fixtures stay deterministic", "2026-07-10T10:00:00Z", "source_conflict"), content: "Decision: keep review fixtures deterministic." },
    makePage("page-keyboard", "Weekly fixture recap", "2026-07-09T12:00:00Z"),
    {
      ...makePage("page-cjk", "CJK layout", "2026-07-08T12:00:00Z", "source_updated"),
      creation_kind: "distilled",
      review_status: "unconfirmed",
    },
    makePage("page-history", "History semantics", "2026-07-07T12:00:00Z"),
    { ...makePage("page-independent", "Independent research", "2026-07-06T12:00:00Z"), domain: null, space: null },
  ];
  const memories = Array.from({ length: 205 }, (_, index) => makeMemory(index));
  return {
    spaces: [
      { id: "space-wenlan", name: "Wenlan", description: "Editorial memory system", suggested: false, starred: true, sort_order: 0, memory_count: 205, entity_count: entities.length, created_at: NOW_SECONDS - 604_800, updated_at: NOW_SECONDS },
      { id: "space-research", name: "Research", description: "Primary-source investigations", suggested: false, starred: true, sort_order: 1, memory_count: 18, entity_count: 4, created_at: NOW_SECONDS - 500_000, updated_at: NOW_SECONDS - 300 },
      { id: "space-personal", name: "Personal", description: "Durable preferences", suggested: false, starred: false, sort_order: 2, memory_count: 12, entity_count: 2, created_at: NOW_SECONDS - 400_000, updated_at: NOW_SECONDS - 600 },
      { id: "space-archive", name: "Archive", description: "Older material", suggested: false, starred: false, sort_order: 3, memory_count: 7, entity_count: 1, created_at: NOW_SECONDS - 300_000, updated_at: NOW_SECONDS - 900 },
      { id: "space-suggested", name: "AI Workflows", description: "Suggested from recent context", suggested: true, starred: false, sort_order: 4, memory_count: 3, entity_count: 1, created_at: NOW_SECONDS - 1_000, updated_at: NOW_SECONDS - 1_000 },
      { id: "space-suggested-2", name: "Product Signals", description: "Another deterministic suggestion", suggested: true, starred: false, sort_order: 5, memory_count: 2, entity_count: 1, created_at: NOW_SECONDS - 900, updated_at: NOW_SECONDS - 900 },
    ],
    pages,
    entities,
    entityDetails: entities.map((entity) => entity.id === adaEntity.id ? {
      entity,
      observations: [{ id: "obs-1", entity_id: "entity-ada", content: "Wrote the first published algorithm", source_agent: "research-agent", confidence: 0.8, confirmed: false, created_at: NOW_SECONDS - 600 }],
      relations: [
        { id: "relation-1", relation_type: "collaborated with", direction: "outgoing", entity_id: "entity-babbage", entity_name: "Charles Babbage", entity_type: "person", source_agent: "research-agent", created_at: NOW_SECONDS - 500 },
        { id: "relation-2", relation_type: "inspired", direction: "incoming", entity_id: "entity-3", entity_name: "Grace Hopper", entity_type: "person", source_agent: null, created_at: NOW_SECONDS - 400 },
      ],
    } : {
      entity,
      observations: [],
      relations: [],
    }),
    memories,
    distillReview: {
      pages_created: 0,
      scoped: false,
      created_ids: [],
      pending: [
        {
          source_ids: ["memory-31", "memory-32"],
          contents: [
            "Wenlan keeps review fixtures isolated from production data.",
            "The Review app resets its deterministic state on relaunch.",
          ],
          space: "Wenlan",
          estimated_tokens: 96,
          existing_page_id: null,
          existing_page_title: null,
          new_memory_count: 2,
        },
        {
          source_ids: ["memory-41"],
          contents: ["New evidence can extend the fixture architecture page."],
          space: "Wenlan",
          estimated_tokens: 48,
          existing_page_id: "page-architecture",
          existing_page_title: "Fixture architecture",
          new_memory_count: 1,
        },
      ],
      stale_pages: [],
      stale_truncated: false,
      orphan_topics: [],
    },
    refinements: [
      {
        id: "refinement-page-history-cleanup",
        action: "page_keep_or_archive",
        source_ids: ["memory-page-history"],
        payload: {
          action: "page_keep_or_archive",
          page_id: "page-history",
          source_count: 1,
        },
        confidence: 0.82,
        created_at: "2026-07-10T12:00:00Z",
      },
    ],
  };
}
