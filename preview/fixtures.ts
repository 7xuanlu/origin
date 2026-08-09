// SPDX-License-Identifier: AGPL-3.0-only
// Fixture data for the browser preview harness. Mirrors the shapes in
// PageDetail.citations.test.tsx so the preview shows the same states the
// tests assert.
import type {
  Page,
  PageCitation,
  PageLinksResponse,
  PageSourceWithMemory,
  ListPageRevisionsResponse,
  PendingRevisionItem,
  RecentActivityItem,
  RefinementProposalSummary,
  MemoryItem,
  Entity,
  EntityDetail,
  RelationWithEntity,
  DistillReviewResponse,
  PageChange,
  ListMemoryRevisionsResponse,
  RegisteredSource,
} from "../src/lib/tauri";

const cite = (
  occurrence: number,
  marker: number,
  over: Partial<PageCitation> = {},
): PageCitation => ({
  occurrence,
  marker,
  source_kind: "memory",
  locator: `mem-${marker}`,
  score: 0.9,
  status: "verified",
  scope: "sentence",
  ...over,
});

const CITED_CONTENT = `# Wenlan Daemon Architecture

The wenlan daemon runs entirely on the local machine, with no cloud dependency for core recall.[1] Storage is backed by libSQL with vector search extensions.[2]

## Storage layer

Distilled pages and raw memories share a single database file.[3] The schema versioning strategy follows the upstream design notes.[4]

## Distillation

Pages are compiled from source memories by the pipeline described in [[Memory Distillation]].[5] Authored sections survive re-distillation untouched.[6]

See also [[LibSQL Storage]].
`;

const base = {
  summary: null,
  entity_id: null,
  domain: "architecture",
  source_memory_ids: ["mem-1", "mem-2"],
  version: 3,
  status: "active",
  created_at: "2026-06-20T09:00:00+00:00",
  last_compiled: "2026-07-02T18:30:00+00:00",
  last_modified: "2026-07-02T18:30:00+00:00",
};

export const PAGES: Record<string, Page> = {
  "page-cited": {
    ...base,
    id: "page-cited",
    title: "Wenlan Daemon Architecture",
    content: CITED_CONTENT,
    citations: [
      cite(1, 1),
      cite(2, 2, { status: "unverified" }),
      cite(3, 3, { locator: "mem-9" }),
      cite(4, 4, {
        source_kind: "external_url",
        locator: "https://docs.turso.tech/libsql",
      }),
      cite(5, 5, {
        source_kind: "external_file",
        locator: "/Users/lucian/notes/distillation-design.md",
      }),
      cite(6, 6, { source_kind: "authored", locator: "authored" }),
    ],
  } as Page,
  "page-cleared": {
    ...base,
    id: "page-cleared",
    title: "Edit-Cleared Citations",
    content: CITED_CONTENT,
    citations: [],
  } as Page,
  "page-mismatch": {
    ...base,
    id: "page-mismatch",
    title: "Mismatched Citations",
    content:
      "# Mismatched Citations\n\nOnly two markers here.[1] But three citations arrive from the wire.[2]",
    citations: [cite(1, 1), cite(2, 2), cite(3, 3)],
  } as Page,
  "page-plain": {
    ...base,
    id: "page-plain",
    title: "Memory Distillation",
    content:
      "# Memory Distillation\n\nA page with no citation markers at all — the pre-citations rendering path.",
    citations: undefined,
  } as Page,
  // --- page_merge dossier pair 1: subset case (nothing lost) ---
  "page-threads-ranking": {
    id: "page-threads-ranking",
    title: "Threads Ranking Signals",
    summary: null,
    content: `# Threads Ranking Signals

Engagement velocity in the first 30 minutes matters more than raw follower count — a small account with a fast reply burst can outrank a big account's slower engagement. Accounts with a track record of quick, relevant replies get a small ranking boost on their own posts too.

Video and image posts consistently outperform text-only in the For You feed, but only when the caption stays under two sentences; longer captions suppress reach even on identical media.

Quote posts inherit part of the original post's velocity signal, which is why quoting a high-velocity post is a common growth tactic. The feed itself re-ranks every few minutes rather than staying static per session.`,
    entity_id: null,
    domain: "social-media",
    source_memory_ids: [
      "mem-rank-1",
      "mem-rank-2",
      "mem-rank-3",
      "mem-rank-4",
      "mem-rank-5",
      "mem-rank-6",
    ],
    version: 3,
    status: "active",
    created_at: "2026-06-15T10:00:00+00:00",
    last_compiled: "2026-07-08T16:45:00+00:00",
    last_modified: "2026-07-08T16:45:00+00:00",
  } as Page,
  "page-threads-ranking-draft": {
    id: "page-threads-ranking-draft",
    title: "Threads Algorithm Notes (draft)",
    summary: null,
    content: `# Threads Algorithm Notes (draft)

Early notes from digging into how Threads ranks posts in the For You feed, before I had all the details straight.

Engagement velocity in the first 30 minutes after posting seems to matter a lot more than raw follower count — a post from a small account that gets a burst of replies and reposts right away can outrank a big account's post that got the same total engagement spread over a day. Accounts that reply quickly and relevantly to others build up some kind of reputation signal that carries over and nudges their own posts too, though I don't have a clean way to measure how much yet.

Media matters: photo and video posts consistently do better than text-only posts in the For You feed. Caption length interacts with this though — once you go past roughly two sentences, reach drops noticeably even on the same media, so it is not just "add a photo and you are set."

One growth tactic that keeps coming up: quoting a post that already has high velocity passes along some fraction of its engagement signal to the quote post itself, which is presumably why quote-posting trending content is so common. Still need to check whether this decays with time since the original post.`,
    entity_id: null,
    domain: "social-media",
    source_memory_ids: ["mem-rank-1", "mem-rank-2", "mem-rank-3", "mem-rank-4", "mem-rank-5"],
    version: 1,
    status: "active",
    created_at: "2026-06-02T08:30:00+00:00",
    last_compiled: "2026-06-02T08:30:00+00:00",
    last_modified: "2026-06-02T08:30:00+00:00",
  } as Page,
  // --- page_merge dossier pair 2: transfer case (2 sources move) ---
  "page-threads-antipatterns": {
    id: "page-threads-antipatterns",
    title: "Threads Content Anti-Patterns",
    summary: null,
    content: `# Threads Content Anti-Patterns

Posting more than about four times an hour visibly suppresses reach on each post, even when every post performs well on its own. Explicit engagement-bait phrasing ("comment YES if you agree") gets flagged and suppressed rather than boosted.

Reposting the same content within 24 hours reads as spam and tanks both posts. More than two or three hashtags correlates with lower reach — the opposite of old Instagram-era advice. Replying to your own post repeatedly to "bump" it can suppress the original too.`,
    entity_id: null,
    domain: "social-media",
    source_memory_ids: [
      "mem-anti-1",
      "mem-anti-2",
      "mem-anti-3",
      "mem-anti-4",
      "mem-anti-5",
      "mem-anti-6",
    ],
    version: 3,
    status: "active",
    created_at: "2026-06-18T11:15:00+00:00",
    last_compiled: "2026-07-09T09:20:00+00:00",
    last_modified: "2026-07-09T09:20:00+00:00",
  } as Page,
  "page-threads-antipatterns-old": {
    id: "page-threads-antipatterns-old",
    title: "Threads Posting Mistakes (old)",
    summary: null,
    content: `# Threads Posting Mistakes (old)

Older, messier notes on things that seem to hurt reach on Threads — collected before I had a cleaner list.

Engagement-bait phrasing like "comment YES if you agree" or "tag someone who needs this" does not get boosted the way it might on other platforms; it gets flagged and actively suppressed instead, which surprised me at first.

Reposting the exact same content within 24 hours reads as spam to the ranking system and appears to tank the reach of both the original and the repost, not just the newer one. Hashtag stuffing is similarly counterproductive — more than two or three hashtags per post correlates with lower reach, the opposite of the old Instagram-era advice to tag everything.

Replying to your own post multiple times in a row to "bump" it back to the top of followers' feeds reads as spam behavior and can suppress the original post's distribution too.

Two things I have not confirmed yet: whether posting the exact same caption to Threads and Instagram at the same time hurts the Threads post specifically, and whether deleting an underperforming post and immediately reposting it resets its distribution to zero rather than giving it a second look.`,
    entity_id: null,
    domain: "social-media",
    source_memory_ids: [
      "mem-anti-3",
      "mem-anti-4",
      "mem-anti-5",
      "mem-anti-6",
      "mem-anti-7",
      "mem-anti-8",
    ],
    version: 1,
    status: "active",
    created_at: "2026-06-05T14:00:00+00:00",
    last_compiled: "2026-06-05T14:00:00+00:00",
    last_modified: "2026-06-05T14:00:00+00:00",
  } as Page,
};

// Pristine copies so the mock redistill_page can restore after update_page
// clears citations (mirrors the backend edit-path contract).
export const PRISTINE: Record<string, Page> = JSON.parse(JSON.stringify(PAGES));

const memory = (
  id: string,
  title: string,
  content: string,
  lastModified: number,
) => ({
  source: { page_id: "page-cited", memory_source_id: id, linked_at: 1_750_000_000 },
  memory: {
    source_id: id,
    title,
    content,
    summary: null,
    memory_type: "memory",
    domain: null,
    source_agent: "claude-code",
    confidence: 0.92,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: lastModified,
    chunk_count: 1,
  },
});

export const SOURCES: PageSourceWithMemory[] = [
  memory(
    "mem-1",
    "Local-first decision",
    "We keep the daemon local-first: recall never leaves the machine.",
    1_751_300_000,
  ),
  memory(
    "mem-2",
    "libSQL storage choice",
    "Chose libSQL over sqlite-vss for the built-in vector extension.",
    1_748_000_000,
  ),
  // mem-9 is deliberately absent: exercises the unresolvable-source fallback
  // in CitationChip and the count-vs-rows case in PageInfo.
];

export const LINKS: PageLinksResponse = {
  outbound: [
    { label: "Memory Distillation", target_page_id: "page-plain" },
    { label: "LibSQL Storage", target_page_id: null },
  ],
  inbound: [{ source_page_id: "page-plain", label: "Wenlan Daemon Architecture" }],
};

// --- review queue fixtures (DistillReviewPanel + ReviewDialog preview) ---

const reviewMemory = (
  id: string,
  title: string,
  content: string,
  lastModified: number,
): MemoryItem => ({
  source_id: id,
  title,
  content,
  summary: null,
  memory_type: "memory",
  domain: null,
  source_agent: "claude-code",
  confidence: 0.9,
  confirmed: true,
  pinned: false,
  supersedes: null,
  last_modified: lastModified,
  chunk_count: 1,
});

export const REVIEW_MEMORIES: Record<string, MemoryItem> = {
  "mem-pkg": reviewMemory(
    "mem-pkg",
    "JS package manager preference",
    "Lucian prefers npm for JavaScript package management across projects.",
    1_751_300_000,
  ),
  "mem-daemon-port": reviewMemory(
    "mem-daemon-port",
    "Daemon port convention",
    "The wenlan daemon listens on port 7878 and the dev server runs on 1420. Restarting the daemon requires killing the old process first before the port frees up.",
    1_751_100_000,
  ),
  "mem-review-flow": reviewMemory(
    "mem-review-flow",
    "Review workflow",
    "Review happens through the plugin /curate command, one item at a time in the terminal.",
    1_750_900_000,
  ),
  "mem-editor-a": reviewMemory(
    "mem-editor-a",
    "Editor choice",
    "Uses Visual Studio Code as the main editor.",
    1_750_800_000,
  ),
  "mem-editor-b": reviewMemory(
    "mem-editor-b",
    "Editor note",
    "Recently switched to Zed for Rust work.",
    1_751_200_000,
  ),
  "mem-preview-harness": reviewMemory(
    "mem-preview-harness",
    "Preview harness architecture",
    "The preview harness mocks Tauri's invoke() so review-queue UI can be exercised without the daemon running.",
    1_751_400_000,
  ),
};

// --- page_merge dossier fixtures (ReviewPageMerge preview) ---
// Two scenarios exercising deriveMergeLedger's two verdicts:
//  - "ranking" pair: retire's sources are a strict subset of keep's ->
//    onlyRetire = [] -> the green "nothing lost, safe" banner.
//  - "antipatterns" pair: retire has 2 sources keep doesn't -> onlyRetire
//    has 2 -> the indigo "2 sources move" banner.

const RANKING_MEMORIES: Record<string, MemoryItem> = {
  "mem-rank-1": reviewMemory(
    "mem-rank-1",
    "Engagement velocity beats follower count",
    "Threads posts get a ranking boost from replies and reposts in the first 30 minutes, more than from the poster's total follower count.",
    1_749_400_000,
  ),
  "mem-rank-2": reviewMemory(
    "mem-rank-2",
    "Reply reputation carries over",
    "Accounts with a track record of fast, relevant replies to others seem to get a small ranking boost on their own posts too.",
    1_749_500_000,
  ),
  "mem-rank-3": reviewMemory(
    "mem-rank-3",
    "Video and image posts outperform text",
    "Photo and video posts consistently outrank text-only posts in the Threads For You feed.",
    1_749_600_000,
  ),
  "mem-rank-4": reviewMemory(
    "mem-rank-4",
    "Caption length suppresses reach",
    "Captions longer than about two sentences reduce reach on Threads even when the underlying media is identical.",
    1_749_700_000,
  ),
  "mem-rank-5": reviewMemory(
    "mem-rank-5",
    "Quote posts inherit engagement signal",
    "Quoting a post with high engagement velocity passes along part of that signal to the quote post — a common growth tactic.",
    1_749_800_000,
  ),
  // Only in the keep page: observed after the draft page was written, which
  // is why the draft's 5 sources are a strict subset of these 6.
  "mem-rank-6": reviewMemory(
    "mem-rank-6",
    "For You feed re-ranks continuously",
    "The Threads For You feed appears to re-rank every few minutes rather than staying static per session; scrolling back up surfaces a new ordering.",
    1_751_950_000,
  ),
};

const ANTIPATTERN_MEMORIES: Record<string, MemoryItem> = {
  "mem-anti-1": reviewMemory(
    "mem-anti-1",
    "Overposting kills reach",
    "Posting more than about four times in an hour on Threads visibly suppresses reach on each individual post, even when each one performs well on its own.",
    1_750_100_000,
  ),
  "mem-anti-2": reviewMemory(
    "mem-anti-2",
    "External links throttled",
    "Posts containing an external link get noticeably less initial distribution than link-free posts on Threads.",
    1_750_150_000,
  ),
  "mem-anti-3": reviewMemory(
    "mem-anti-3",
    "Engagement bait phrasing",
    "Explicit engagement-bait phrasing like 'comment YES if you agree' gets flagged and suppressed on Threads rather than boosted.",
    1_750_200_000,
  ),
  "mem-anti-4": reviewMemory(
    "mem-anti-4",
    "Reposting own content too soon",
    "Reposting the same content within 24 hours reads as spam to Threads' ranking system and tanks both posts' reach.",
    1_750_250_000,
  ),
  "mem-anti-5": reviewMemory(
    "mem-anti-5",
    "Hashtag stuffing",
    "More than two or three hashtags per post correlates with lower reach on Threads, the opposite of the old Instagram-era advice.",
    1_750_300_000,
  ),
  "mem-anti-6": reviewMemory(
    "mem-anti-6",
    "Thread-bombing own replies",
    "Replying to your own post multiple times in a row to 'bump' it reads as spam and can suppress the original post's distribution.",
    1_750_350_000,
  ),
  // Only in the retired page: unconfirmed hypotheses the draft raised but
  // the canonical page hasn't absorbed yet — these 2 are what "moves".
  "mem-anti-7": reviewMemory(
    "mem-anti-7",
    "Cross-posting identical captions",
    "Posting the exact same caption simultaneously to Threads and Instagram measurably reduces the Threads post's own reach.",
    1_750_900_000,
  ),
  "mem-anti-8": reviewMemory(
    "mem-anti-8",
    "Deleting and reposting resets reach",
    "Deleting an underperforming post and reposting it immediately resets it to zero distribution instead of giving it a second chance.",
    1_750_950_000,
  ),
};

function mergeSource(
  pageId: string,
  mem: MemoryItem,
  linkedAt: number,
): PageSourceWithMemory {
  return { source: { page_id: pageId, memory_source_id: mem.source_id, linked_at: linkedAt }, memory: mem };
}

export const MERGE_SOURCES: Record<string, PageSourceWithMemory[]> = {
  "page-threads-ranking": [
    RANKING_MEMORIES["mem-rank-1"],
    RANKING_MEMORIES["mem-rank-2"],
    RANKING_MEMORIES["mem-rank-3"],
    RANKING_MEMORIES["mem-rank-4"],
    RANKING_MEMORIES["mem-rank-5"],
    RANKING_MEMORIES["mem-rank-6"],
  ].map((mem) => mergeSource("page-threads-ranking", mem, mem.last_modified + 1_800)),
  "page-threads-ranking-draft": [
    RANKING_MEMORIES["mem-rank-1"],
    RANKING_MEMORIES["mem-rank-2"],
    RANKING_MEMORIES["mem-rank-3"],
    RANKING_MEMORIES["mem-rank-4"],
    RANKING_MEMORIES["mem-rank-5"],
  ].map((mem) => mergeSource("page-threads-ranking-draft", mem, mem.last_modified + 900)),
  "page-threads-antipatterns": [
    ANTIPATTERN_MEMORIES["mem-anti-1"],
    ANTIPATTERN_MEMORIES["mem-anti-2"],
    ANTIPATTERN_MEMORIES["mem-anti-3"],
    ANTIPATTERN_MEMORIES["mem-anti-4"],
    ANTIPATTERN_MEMORIES["mem-anti-5"],
    ANTIPATTERN_MEMORIES["mem-anti-6"],
  ].map((mem) => mergeSource("page-threads-antipatterns", mem, mem.last_modified + 1_800)),
  "page-threads-antipatterns-old": [
    ANTIPATTERN_MEMORIES["mem-anti-3"],
    ANTIPATTERN_MEMORIES["mem-anti-4"],
    ANTIPATTERN_MEMORIES["mem-anti-5"],
    ANTIPATTERN_MEMORIES["mem-anti-6"],
    ANTIPATTERN_MEMORIES["mem-anti-7"],
    ANTIPATTERN_MEMORIES["mem-anti-8"],
  ].map((mem) => mergeSource("page-threads-antipatterns-old", mem, mem.last_modified + 900)),
};

const reviewRevisions = (): PendingRevisionItem[] => [
  {
    target_source_id: "mem-pkg",
    revision_source_id: "mem-pkg-rev",
    revision_content:
      "Lucian prefers pnpm for JavaScript package management across projects, keeping npm only for one-off npx invocations.",
    source_agent: "claude-code",
    last_modified: 1_752_000_000,
  },
  {
    target_source_id: "mem-daemon-port",
    revision_source_id: "mem-daemon-port-rev",
    revision_content:
      "The wenlan daemon listens on port 7878 and the dev server runs on 1420. The preview harness uses 1421.",
    source_agent: "codex",
    last_modified: 1_752_000_100,
  },
  {
    target_source_id: "mem-review-flow",
    revision_source_id: "mem-review-flow-rev",
    revision_content:
      "Review happens in the desktop app: a queue page with a before/after diff dialog, approve or dismiss per item.",
    source_agent: "claude-code",
    last_modified: 1_752_000_200,
  },
];

const reviewProposals = (): RefinementProposalSummary[] => [
  {
    id: "prop-merge-vscode",
    action: "entity_merge",
    source_ids: ["ent-vscode", "ent-vs-code"],
    payload: {
      action: "entity_merge",
      existing_id: "ent-vscode",
      new_id: "ent-vs-code",
      similarity: 0.93,
    } as RefinementProposalSummary["payload"],
    confidence: 0.93,
    created_at: "2026-07-08T14:20:00Z",
  },
  {
    id: "prop-contra-editor",
    action: "detect_contradiction",
    source_ids: ["mem-editor-a", "mem-editor-b"],
    payload: { action: "detect_contradiction" } as RefinementProposalSummary["payload"],
    confidence: 0.78,
    created_at: "2026-07-09T09:05:00Z",
  },
  {
    id: "prop-page-merge-arch",
    action: "page_merge",
    // Daemon order: source_ids[0] survives, source_ids[1] is absorbed.
    // Subset case: the draft's 5 sources are all present in the keep page's
    // 6, so the merge dossier renders the green "nothing lost" verdict.
    source_ids: ["page-threads-ranking", "page-threads-ranking-draft"],
    payload: {
      action: "page_merge",
      left_page_id: "page-threads-ranking",
      right_page_id: "page-threads-ranking-draft",
      source_overlap: 5,
      source_overlap_ratio: 1.0,
    } as RefinementProposalSummary["payload"],
    confidence: 1.0,
    created_at: "2026-07-10T08:26:41Z",
  },
  {
    id: "prop-page-merge-antipatterns",
    action: "page_merge",
    // Transfer case: the retired page has 2 sources (unconfirmed
    // hypotheses) not yet in the keep page, so the dossier renders the
    // indigo "2 sources move" verdict instead of the safe one.
    source_ids: ["page-threads-antipatterns", "page-threads-antipatterns-old"],
    payload: {
      action: "page_merge",
      left_page_id: "page-threads-antipatterns",
      right_page_id: "page-threads-antipatterns-old",
      source_overlap: 4,
      source_overlap_ratio: 0.67,
    } as RefinementProposalSummary["payload"],
    confidence: 0.88,
    created_at: "2026-07-10T11:05:00Z",
  },
  {
    id: "prop-rel-conflict-editor",
    action: "relation_conflict",
    // Daemon order: source_ids[0] is the new relation, [1] the existing one.
    source_ids: ["rel-uses-new", "rel-uses-old"],
    payload: {
      action: "relation_conflict",
      existing_id: "rel-uses-old",
      new_id: "rel-uses-new",
      from: "Lucian",
      to: "Zed",
      old_type: "EVALUATES",
      new_type: "USES_DAILY",
    } as RefinementProposalSummary["payload"],
    confidence: 0.82,
    created_at: "2026-07-09T18:40:00Z",
  },
  {
    id: "prop-keep-or-archive-scratch",
    action: "page_keep_or_archive",
    source_ids: ["page-cleared"],
    payload: {
      action: "page_keep_or_archive",
      page_id: "page-cleared",
      source_count: 1,
    } as RefinementProposalSummary["payload"],
    confidence: 1.0,
    created_at: "2026-07-10T07:12:00Z",
  },
  {
    id: "prop-suggest-entity-tauri",
    action: "suggest_entity",
    source_ids: ["mem-preview-harness", "mem-daemon-port"],
    payload: {
      action: "suggest_entity",
      name_hint: "Tauri",
    } as RefinementProposalSummary["payload"],
    confidence: 0.8,
    created_at: "2026-07-10T10:00:00Z",
  },
];

const reviewCaptures = (): RecentActivityItem[] => [
  {
    kind: "memory",
    id: "mem-capture-shortcuts",
    title: "Prefers keyboard-first review: Enter approves, D dismisses",
    snippet: "Observed while walking the review queue during the redesign session.",
    timestamp_ms: 1_752_000_300_000,
    badge: { kind: "needs_review" },
  },
  {
    kind: "memory",
    id: "mem-capture-worktree",
    title: "Feature work happens in git worktrees, never directly on main",
    snippet: "Stated as a standing rule for all wenlan-app development.",
    timestamp_ms: 1_752_000_400_000,
    badge: { kind: "needs_review" },
  },
];

export const REVIEW_STATE: {
  revisions: PendingRevisionItem[];
  proposals: RefinementProposalSummary[];
  captures: RecentActivityItem[];
} = {
  revisions: reviewRevisions(),
  proposals: reviewProposals(),
  captures: reviewCaptures(),
};

export function resetReviewFixtures(): void {
  REVIEW_STATE.revisions = reviewRevisions();
  REVIEW_STATE.proposals = reviewProposals();
  REVIEW_STATE.captures = reviewCaptures();
}

// Toggled by the preview harness's "Fail queue" button so problem 1's
// error-state UI (load failed / partial load) is reachable without a real
// daemon outage.
export const REVIEW_FAIL = { queue: false };

export const REVIEW_ENTITIES: Record<string, EntityDetail> = {
  "ent-vscode": {
    entity: {
      id: "ent-vscode",
      name: "Visual Studio Code",
      entity_type: "tool",
      domain: "tools",
      source_agent: "claude-code",
      confidence: 0.95,
      confirmed: true,
      created_at: 1_749_000_000,
      updated_at: 1_751_500_000,
    },
    observations: [
      {
        id: "obs-1",
        entity_id: "ent-vscode",
        content: "Primary editor for TypeScript work.",
        source_agent: "claude-code",
        confidence: 0.9,
        confirmed: true,
        created_at: 1_749_100_000,
      },
    ],
    relations: [],
  },
  "ent-vs-code": {
    entity: {
      id: "ent-vs-code",
      name: "VS Code",
      entity_type: "tool",
      domain: "tools",
      source_agent: "codex",
      confidence: 0.8,
      confirmed: false,
      created_at: 1_751_900_000,
      updated_at: 1_751_900_000,
    },
    observations: [
      {
        id: "obs-2",
        entity_id: "ent-vs-code",
        content: "Mentioned as the editor used for the wenlan-app repo.",
        source_agent: "codex",
        confidence: 0.8,
        confirmed: false,
        created_at: 1_751_900_000,
      },
    ],
    relations: [],
  },
};

// --- AtlasView / FocusGraph preview fixture ---
// 16 entities spanning the 5 validated palette slots (project/technology/
// organization/person/concept) plus place (folds to neutral), so every slot
// swatch has a rendered proof. gk-remote-office takes no relation below,
// exercising the empty-focus case. The four gk-z* entities form a second
// community (Zettelkasten is its own degree peak — Knowledge Graph ties it
// at 4 but never exceeds it), so the Atlas cartography layer renders two
// hulls, a minor region label, and the amber bridge edge.
const GRAPH_ENTITY_SEED: { id: string; name: string; entity_type: string }[] = [
  { id: "gk-wenlan", name: "Wenlan", entity_type: "project" },
  { id: "gk-desktop", name: "Wenlan Desktop", entity_type: "project" },
  { id: "gk-rust", name: "Rust", entity_type: "technology" },
  { id: "gk-typescript", name: "TypeScript", entity_type: "technology" },
  { id: "gk-tauri", name: "Tauri", entity_type: "technology" },
  { id: "gk-sqlite", name: "SQLite", entity_type: "technology" },
  { id: "gk-anthropic", name: "Anthropic", entity_type: "organization" },
  { id: "gk-github", name: "GitHub", entity_type: "organization" },
  { id: "gk-lucian", name: "Lucian", entity_type: "person" },
  { id: "gk-ada", name: "Ada", entity_type: "person" },
  { id: "gk-knowledge-graph", name: "Knowledge Graph", entity_type: "concept" },
  { id: "gk-remote-office", name: "Remote Office", entity_type: "place" },
  { id: "gk-zettel", name: "Zettelkasten", entity_type: "concept" },
  { id: "gk-znotes", name: "Field Notes", entity_type: "concept" },
  { id: "gk-zobsidian", name: "Obsidian", entity_type: "technology" },
  { id: "gk-zmochi", name: "Mochi", entity_type: "technology" },
];

// [relationId, fromId, verb, toId] — mirrored below onto both endpoints'
// detail.relations (outgoing on the source's list, incoming on the
// target's), the same way the real daemon returns one relation from either
// entity's perspective. 20 tuples, 11 distinct verbs, both directions.
const GRAPH_RELATION_SEED: [string, string, string, string][] = [
  ["gkr-1", "gk-wenlan", "uses", "gk-rust"],
  ["gkr-2", "gk-wenlan", "uses", "gk-typescript"],
  ["gkr-3", "gk-desktop", "uses", "gk-tauri"],
  ["gkr-4", "gk-desktop", "uses", "gk-sqlite"],
  ["gkr-5", "gk-desktop", "depends_on", "gk-wenlan"],
  ["gkr-6", "gk-lucian", "maintains", "gk-wenlan"],
  ["gkr-7", "gk-lucian", "maintains", "gk-desktop"],
  ["gkr-8", "gk-ada", "contributes_to", "gk-desktop"],
  ["gkr-9", "gk-wenlan", "hosted_on", "gk-github"],
  ["gkr-10", "gk-desktop", "hosted_on", "gk-github"],
  ["gkr-11", "gk-lucian", "works_with", "gk-anthropic"],
  ["gkr-12", "gk-wenlan", "implements", "gk-knowledge-graph"],
  ["gkr-13", "gk-desktop", "visualizes", "gk-knowledge-graph"],
  ["gkr-14", "gk-ada", "works_with", "gk-lucian"],
  ["gkr-15", "gk-knowledge-graph", "relates_to", "gk-rust"],
  ["gkr-16", "gk-zettel", "organizes", "gk-znotes"],
  ["gkr-17", "gk-zobsidian", "implements", "gk-zettel"],
  ["gkr-18", "gk-zmochi", "relates_to", "gk-zettel"],
  ["gkr-19", "gk-znotes", "hosted_on", "gk-zobsidian"],
  // The single inter-community edge — the cartography layer's bridge.
  ["gkr-20", "gk-knowledge-graph", "inspired_by", "gk-zettel"],
];

const graphEntitySeedById = new Map(GRAPH_ENTITY_SEED.map((e) => [e.id, e]));

export const GRAPH_ENTITIES: Entity[] = GRAPH_ENTITY_SEED.map((seed, i) => ({
  id: seed.id,
  name: seed.name,
  entity_type: seed.entity_type,
  // Two spaces mirroring the two communities — feeds the Atlas Space selector.
  domain: seed.id.startsWith("gk-z") ? "notes" : "wenlan-dev",
  source_agent: "claude-code",
  confidence: 0.9,
  confirmed: true,
  created_at: 1_752_000_000 + i * 1000,
  updated_at: 1_752_000_000 + i * 1000,
}));

const graphEntityById = new Map(GRAPH_ENTITIES.map((e) => [e.id, e]));

export const GRAPH_DETAILS: Record<string, EntityDetail> = Object.fromEntries(
  GRAPH_ENTITY_SEED.map((seed) => {
    const relations: RelationWithEntity[] = [];
    for (const [id, fromId, verb, toId] of GRAPH_RELATION_SEED) {
      const isSource = fromId === seed.id;
      const isTarget = toId === seed.id;
      if (!isSource && !isTarget) continue;
      const other = graphEntitySeedById.get(isSource ? toId : fromId)!;
      relations.push({
        id,
        relation_type: verb,
        direction: isSource ? "outgoing" : "incoming",
        entity_id: other.id,
        entity_name: other.name,
        entity_type: other.entity_type,
        source_agent: "claude-code",
        created_at: 1_752_000_000,
      });
    }
    return [seed.id, { entity: graphEntityById.get(seed.id)!, observations: [], relations }];
  }),
);

// --- Graph memories preview fixture (served by preview/mocks/core.ts) ---
// 8 memories tied to GRAPH_ENTITY_SEED ids: five resolve via entity_id, two
// (gm-6, gm-7) resolve only through the title-contains-entity-name fallback
// (entity_id unset), and one (gm-8) matches no entity at all — mirrors a
// stray memory the graph correctly drops rather than rendering as an orphan.
export const GRAPH_MEMORIES: MemoryItem[] = [
  {
    source_id: "gm-1",
    title: "Wenlan runs fully local, no cloud dependency for recall",
    content: "The daemon never sends memory content off-device for core recall.",
    summary: null,
    memory_type: "fact",
    domain: "architecture",
    source_agent: "claude-code",
    confidence: 0.95,
    confirmed: true,
    stability: "confirmed",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_100_000,
    chunk_count: 1,
    entity_id: "gk-wenlan",
  },
  {
    source_id: "gm-2",
    title: "Rust's ownership model prevents data races at compile time",
    content: "No garbage collector; the borrow checker enforces one mutable reference at a time.",
    summary: null,
    memory_type: "fact",
    domain: "engineering",
    source_agent: "claude-code",
    confidence: 0.92,
    confirmed: true,
    stability: "confirmed",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_105_000,
    chunk_count: 1,
    entity_id: "gk-rust",
  },
  {
    source_id: "gm-3",
    title: "Tauri IPC bridge marshals commands over a typed invoke() boundary",
    content: "Frontend calls invoke(cmd, args); Rust commands register via tauri::generate_handler!.",
    summary: null,
    memory_type: "fact",
    domain: "engineering",
    source_agent: "codex",
    confidence: 0.8,
    confirmed: false,
    stability: "learned",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_110_000,
    chunk_count: 1,
    entity_id: "gk-tauri",
  },
  {
    source_id: "gm-4",
    title: "Lucian prefers Zed for Rust work, VS Code for everything else",
    content: "Switched to Zed mid-2026 for the faster LSP experience on large Rust crates.",
    summary: null,
    memory_type: "preference",
    domain: null,
    source_agent: "claude-code",
    confidence: 0.9,
    confirmed: true,
    stability: "confirmed",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_115_000,
    chunk_count: 1,
    entity_id: "gk-lucian",
    is_recap: true,
  },
  {
    source_id: "gm-5",
    title: "Anthropic model routing prefers pinned sources over auto fallback",
    content: "A pinned_degraded result means the pin isn't configured yet and the chain fell back.",
    summary: null,
    memory_type: "fact",
    domain: "engineering",
    source_agent: "claude-code",
    confidence: 0.7,
    confirmed: false,
    stability: "new",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_120_000,
    chunk_count: 1,
    entity_id: "gk-anthropic",
  },
  {
    source_id: "gm-6",
    title: "GitHub Actions flakiness traced to a shared runner cache",
    content: "Intermittent CI failures cleared up after pinning the cache key to the lockfile hash.",
    summary: null,
    memory_type: "gotcha",
    domain: "engineering",
    source_agent: "codex",
    confidence: 0.75,
    confirmed: false,
    stability: "new",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_125_000,
    chunk_count: 1,
    // No entity_id — resolves via the title→entity-name fallback (GitHub).
  },
  {
    source_id: "gm-7",
    title: "SQLite vacuum reclaims space but blocks writers for its duration",
    content: "Run VACUUM during low-traffic windows; it rewrites the whole file.",
    summary: null,
    memory_type: "fact",
    domain: "engineering",
    source_agent: "claude-code",
    confidence: 0.88,
    confirmed: true,
    stability: "confirmed",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_130_000,
    chunk_count: 1,
    // No entity_id — resolves via the title→entity-name fallback (SQLite).
  },
  {
    source_id: "gm-8",
    title: "Stray note with no matching graph entity",
    content: "Never linked to any knowledge-graph entity — exercises the drop-if-unlinked path.",
    summary: null,
    memory_type: "fact",
    domain: null,
    source_agent: "claude-code",
    confidence: 0.5,
    confirmed: false,
    stability: "new",
    pinned: false,
    supersedes: null,
    last_modified: 1_752_135_000,
    chunk_count: 1,
  },
];

// 3 fixture pages: (1) anchored to gk-wenlan via entity_id, citing 2 memories;
// (2) anchored to gk-remote-office — the entity that otherwise has zero
// relations, so this page gives it its first graph link; (3) no anchor
// entity at all, citing only gm-1, so it appears only when both the Pages
// and Memories toggles are on (needs gm-1's node to exist to link to).
export const GRAPH_PAGES: Page[] = [
  {
    id: "page-wenlan-overview",
    title: "Wenlan Project Overview",
    summary: null,
    content: "# Wenlan\n\nLocal-first personal memory layer: daemon, desktop shell, and knowledge graph.",
    entity_id: "gk-wenlan",
    domain: "architecture",
    source_memory_ids: ["gm-1", "gm-2"],
    version: 1,
    status: "active",
    created_at: "2026-07-10T09:00:00+00:00",
    last_compiled: "2026-07-12T10:00:00+00:00",
    last_modified: "2026-07-12T10:00:00+00:00",
  },
  {
    id: "page-remote-office",
    title: "Remote Office Setup",
    summary: null,
    content: "# Remote Office\n\nNotes on the remote-office arrangement.",
    entity_id: "gk-remote-office",
    domain: null,
    source_memory_ids: [],
    version: 1,
    status: "active",
    created_at: "2026-07-11T09:00:00+00:00",
    last_compiled: "2026-07-11T09:00:00+00:00",
    last_modified: "2026-07-11T09:00:00+00:00",
  },
  {
    id: "page-distillation-pipeline",
    title: "Distillation Pipeline Overview",
    summary: null,
    content: "# Distillation Pipeline\n\nHow memories get synthesized into pages.",
    entity_id: null,
    domain: null,
    source_memory_ids: ["gm-1"],
    version: 1,
    status: "active",
    created_at: "2026-07-09T09:00:00+00:00",
    last_compiled: "2026-07-13T09:00:00+00:00",
    last_modified: "2026-07-13T09:00:00+00:00",
  },
];

export const REVIEW_DISTILL: DistillReviewResponse = {
  pages_created: 0,
  scoped: false,
  created_ids: [],
  pending: [
    {
      source_ids: ["mem-pkg", "mem-daemon-port"],
      contents: [
        "Lucian prefers npm for JavaScript package management across projects.",
        "The wenlan daemon listens on port 7878 and the dev server runs on 1420.",
      ],
      entity_id: "ent-tooling",
      entity_name: "Project tooling",
      space: "Engineering",
      estimated_tokens: 160,
      existing_page_id: "page-cited",
      existing_page_title: "Wenlan Daemon Architecture",
      new_memory_count: 2,
    },
  ],
  stale_pages: [
    {
      page_id: "page-cited",
      title: "Wenlan Daemon Architecture",
      summary: "Source memories changed after the page compiled.",
      source_memory_ids: ["mem-1", "mem-2"],
      sources_updated_count: 2,
      stale_reason: "source_updated",
      user_edited: false,
    },
  ],
  stale_truncated: false,
  orphan_topics: [{ label: "Preview harness", count: 3 }],
};

export const REVISIONS: ListPageRevisionsResponse = {
  page_id: "page-cited",
  current_version: 3,
  user_edited: false,
  stale_reason: null,
  entries: [
    {
      version: 3,
      at: 1_751_480_000,
      edited_by: "distiller",
      delta_summary: "Re-distilled after 2 new source memories",
      incoming_source_ids: ["mem-2"],
      citations_summary: "Citations: 6 (1 unverified)",
    },
    {
      version: 2,
      at: 1_750_400_000,
      edited_by: "user",
      delta_summary: "Manual edit of the storage section",
      incoming_source_ids: null,
      citations_summary: null,
    },
    {
      version: 1,
      at: 1_750_000_000,
      edited_by: "distiller",
      delta_summary: "Initial distillation from 2 memories",
      incoming_source_ids: ["mem-1", "mem-2"],
      citations_summary: "Citations: 4",
    },
  ],
};

// --- ReviewHistory preview: RecentRevisionsSection (list_recent_changes) ---

export const RECENT_CHANGES: PageChange[] = [
  {
    page_id: "page-threads-antipatterns",
    title: "Threads Content Anti-Patterns",
    change_kind: "merged",
    changed_at_ms: 1_752_142_800_000,
  },
  {
    page_id: "page-cited",
    title: "Wenlan Daemon Architecture",
    change_kind: "revised",
    changed_at_ms: 1_752_087_300_000,
  },
  {
    page_id: "page-threads-ranking",
    title: "Threads Ranking Signals",
    change_kind: "created",
    changed_at_ms: 1_752_019_800_000,
  },
  {
    page_id: "page-plain",
    title: "Memory Distillation",
    change_kind: "revised",
    changed_at_ms: 1_751_949_600_000,
  },
  {
    page_id: "page-mismatch",
    title: "Mismatched Citations",
    change_kind: "created",
    changed_at_ms: 1_751_800_200_000,
  },
];

// --- ReviewHistory preview: MemoryRevisionChain (get_memory_revisions) ---
// Keyed by the pending-revision items' target_source_id (reviewRevisions()
// above). "mem-pkg" gets a full 2-entry chain with a protected_revision
// entry; the others get the realistic default (shallow, nothing to show).

export const MEMORY_REVISIONS: Record<string, ListMemoryRevisionsResponse> = {
  "mem-pkg": {
    current_source_id: "mem-pkg",
    chain_depth: 3,
    entries: [
      {
        source_id: "mem-pkg-v1",
        depth: 1,
        title: "Standardize new repos on npm",
        content_preview:
          "New repos should default to npm; older repos already on yarn stay as-is until a deliberate migration.",
        last_modified: 1_750_400_000,
        source_agent: "claude-code",
        supersede_mode: null,
        delta_summary: "Narrowed from a blanket npm preference to new-repos-only",
      },
      {
        source_id: "mem-pkg-v0",
        depth: 2,
        title: "Package manager: leaning npm",
        content_preview:
          "Leaning toward npm since it ships with Node by default; still evaluating pnpm for monorepo speed.",
        last_modified: 1_749_150_000,
        source_agent: "codex",
        supersede_mode: "protected_revision",
        delta_summary: "Early exploratory note, protected before the npm preference firmed up",
      },
    ],
  },
  "mem-daemon-port": { current_source_id: "mem-daemon-port", chain_depth: 1, entries: [] },
  "mem-review-flow": { current_source_id: "mem-review-flow", chain_depth: 1, entries: [] },
};

// Connected sources for settings → Sources. The live page keeps this list
// empty on purpose (registered sources are app-local Tauri state a browser
// can't read), so without fixture rows the row/kebab/error-callout states
// could never be pixel-reviewed. last_sync is relative to load time so the
// "Last synced …ago" meta line stays plausible whenever the preview runs.
const nowSec = Math.floor(Date.now() / 1000);
export const REGISTERED_SOURCES: RegisteredSource[] = [
  {
    id: "fixture-src-vault",
    source_type: "obsidian",
    path: "/Users/lucian/Notes",
    status: "Active",
    last_sync: nowSec - 40 * 60,
    file_count: 214,
    memory_count: 187,
  },
  {
    id: "fixture-src-research",
    source_type: "directory",
    path: "/Users/lucian/Library/CloudStorage/GoogleDrive/research",
    status: "Active",
    last_sync: nowSec - 26 * 3600,
    file_count: 38,
    memory_count: 41,
    last_sync_errors: 3,
    last_sync_error_detail: "google_drive_offline",
  },
  {
    id: "fixture-src-clippings",
    source_type: "directory",
    path: "/Users/lucian/Archive/clippings",
    status: "Paused",
    last_sync: null,
    file_count: 0,
    memory_count: 0,
  },
];
