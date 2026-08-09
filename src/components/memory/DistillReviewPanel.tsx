// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  distillReview,
  type DistillReviewResponse,
} from "../../lib/tauri";
import ReviewDialog, {
  reviewKindLabel,
  reviewKindTone,
  useReviewItemSummary,
} from "./ReviewDialog";
import { RecentRevisionsSection } from "./ReviewHistory";
import { relativeMs } from "./page/format";
import {
  EXAMPLE_REVIEW_ITEMS,
  isExampleReviewItem,
  REVIEW_EXAMPLES_ENABLED,
  seedReviewExampleCaches,
} from "./reviewExamples";
import {
  reviewSuppressKey,
  useSuppressedReviewItems,
  type HiddenReviewEntry,
} from "./reviewSuppression";
import {
  reviewItemId,
  reviewItemSection,
  useReviewQueue,
  type ReviewItem,
  type ReviewSection,
} from "./useReviewQueue";
import {
  cacheDistillReviewSession,
  pageCandidateItems,
} from "./pages/pageReviewSignals";

interface DistillReviewPanelProps {
  onBack: () => void;
  onPageClick: (pageId: string) => void;
  onMemoryClick?: (sourceId: string) => void;
}

function truncateText(value: string, max: number): string {
  if (max <= 0) return "";
  if (value.length <= max) return value;
  if (max <= 3) return ".".repeat(max);
  return `${value.slice(0, max - 3).trimEnd()}...`;
}

// Captures never reach this panel (they surface on the home rail instead), so
// the filter has no "captures" chip — every other section gets one.
type ReviewFilterKey = "all" | Exclude<ReviewSection, "captures">;

// Mirrors useReviewQueue's SECTION_ORDER (not exported); keep in sync if that
// list changes.
const FILTER_ORDER: Exclude<ReviewFilterKey, "all">[] = [
  "revisions",
  "conflicts",
  "pages",
  "memory",
  "candidates",
  "topics",
];

const FILTER_LABEL_KEYS = {
  all: "review.filterAll",
  revisions: "review.filterRevisions",
  conflicts: "review.filterConflicts",
  pages: "review.filterPages",
  memory: "review.filterMemory",
  candidates: "review.filterCandidates",
  topics: "review.filterTopics",
} as const;

const panelTextStyle = {
  fontFamily: "var(--mem-font-body)",
  color: "var(--mem-text)",
};

const secondaryTextStyle = {
  fontFamily: "var(--mem-font-body)",
  color: "var(--mem-text-secondary)",
};

/** Mockup section-label: small uppercase eyebrow with a mono count beside it. */
const sectionTitleStyle = {
  margin: "0 0 10px",
  fontFamily: "var(--mem-font-body)",
  fontSize: "12px",
  fontWeight: 600,
  letterSpacing: "0.08em",
  textTransform: "uppercase" as const,
  color: "var(--mem-text-tertiary)",
  display: "flex",
  alignItems: "baseline",
  gap: 8,
};

const sectionCountStyle = {
  fontFamily: "var(--mem-font-mono)",
  fontVariantNumeric: "tabular-nums" as const,
  fontWeight: 400,
};

const emptyTitleStyle = {
  margin: "0 0 10px",
  fontFamily: "var(--mem-font-heading)",
  fontSize: "15px",
  fontWeight: 500,
  color: "var(--mem-text)",
};

const itemSurfaceStyle = {
  border: "1px solid var(--mem-border)",
  borderRadius: 8,
  backgroundColor: "var(--mem-surface)",
};

/** Dashed pill marking a dev-only sample card (see reviewExamples.ts) —
 * mirrors the recipe in ReviewDialog's header chip row. */
const examplePillStyle = {
  fontFamily: "var(--mem-font-mono)",
  fontSize: 10.5,
  letterSpacing: "0.06em",
  textTransform: "uppercase" as const,
  borderRadius: 5,
  padding: "1px 7px",
  color: "var(--mem-text-tertiary)",
  border: "1px dashed var(--mem-border)",
  whiteSpace: "nowrap" as const,
};

function QueueCard({
  item,
  onOpen,
  example,
}: {
  item: ReviewItem;
  onOpen: (id: string) => void;
  example?: boolean;
}) {
  const { t } = useTranslation();
  // Mockup card anatomy: chip + age on top, a real title (page/entity names,
  // never the bare kind label), then the evidence line (word delta, overlap,
  // similarity) — the reason a decision is being asked for.
  const { title, reason, delta } = useReviewItemSummary(item);
  const age = item.timestampMs != null ? relativeMs(item.timestampMs) : null;
  const tone = reviewKindTone(item);
  const preview =
    item.kind === "page_candidate"
      ? item.cluster.contents
          .map((content) => content.trim())
          .filter((content) => content.length > 0)
          .slice(0, 2)
      : [];
  return (
    <button
      type="button"
      aria-label={t("review.openItem", { title })}
      onClick={() => onOpen(reviewItemId(item))}
      className="text-left transition-[background-color,border-color,transform] duration-150 hover:bg-[var(--mem-hover)] hover:border-[var(--mem-accent-indigo)] active:scale-[0.99] focus-visible:outline-2 focus-visible:outline-[var(--mem-accent-indigo)] focus-visible:outline-offset-2"
      style={{
        ...itemSurfaceStyle,
        ...(example ? { border: "1px dashed var(--mem-border)" } : null),
        display: "grid",
        gap: 6,
        width: "100%",
        padding: "13px 14px",
        color: "var(--mem-text)",
        cursor: "pointer",
        fontFamily: "var(--mem-font-body)",
      }}
    >
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            fontSize: 11,
            letterSpacing: "0.04em",
            borderRadius: 5,
            padding: "2px 8px",
            color: tone.color,
            backgroundColor: tone.background,
          }}
        >
          {reviewKindLabel(t, item)}
        </span>
        {example && (
          <span style={examplePillStyle}>{t("review.exampleBadge")}</span>
        )}
        {age && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 12,
              color: "var(--mem-text-tertiary)",
              whiteSpace: "nowrap",
            }}
          >
            {age}
          </span>
        )}
      </span>
      <span
        style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: 15,
          fontWeight: 500,
          lineHeight: 1.35,
        }}
      >
        {title}
      </span>
      {(delta != null || reason) && (
        <span
          style={{
            ...secondaryTextStyle,
            display: "flex",
            gap: 10,
            alignItems: "baseline",
            flexWrap: "wrap",
            fontSize: 13,
            lineHeight: 1.45,
          }}
        >
          {delta != null && (
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: 11.5,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              <span style={{ color: "var(--mem-accent-sage)" }}>
                +{delta.added}
              </span>{" "}
              <span style={{ color: "var(--mem-status-danger-text)" }}>
                −{delta.removed}
              </span>
            </span>
          )}
          {reason}
        </span>
      )}
      {preview.map((content, index) => (
        <span
          key={index}
          style={{
            ...secondaryTextStyle,
            fontSize: 13,
            lineHeight: 1.45,
          }}
        >
          {truncateText(content, 140)}
        </span>
      ))}
    </button>
  );
}

/** Mirrors reviewKindTone/reviewKindLabel (ReviewDialog.tsx) for the three
 * locally-hideable kinds — those helpers key off a full ReviewItem, which a
 * HiddenReviewEntry (key/label/kind/at only) doesn't carry. */
function hiddenKindTone(kind: string): { color: string; background: string } {
  const mix = (token: string) => `color-mix(in srgb, ${token} 15%, transparent)`;
  switch (kind) {
    case "stale_page":
    case "page_candidate":
      return { color: "var(--mem-accent-warm)", background: mix("var(--mem-accent-warm)") };
    case "topic":
      return { color: "var(--mem-accent-amber)", background: mix("var(--mem-accent-amber)") };
    default:
      return { color: "var(--mem-text-tertiary)", background: "var(--mem-hover)" };
  }
}

function hiddenKindLabel(t: TFunction, kind: string): string {
  switch (kind) {
    case "stale_page":
      return t("review.kindPageRefresh");
    case "page_candidate":
      return t("review.kindPageCandidate");
    case "topic":
      return t("review.kindTopic");
    // Only example items land here with these two kinds — real revisions and
    // refinements dismiss through the daemon instead (see reviewSuppression.ts).
    case "revision":
      return t("review.kindRevision");
    case "refinement":
      return t("review.kindRefinement");
    default:
      return kind;
  }
}

/** Permanent, quiet escape hatch for items hidden via the dialog's "Hide"
 * button — no timed undo toast; Restore is always one click away. */
function HiddenFooter({
  entries,
  onRestore,
  onRestoreAll,
}: {
  entries: HiddenReviewEntry[];
  onRestore: (key: string) => void;
  onRestoreAll: () => void;
}) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);

  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: 12.5,
          color: "var(--mem-text-tertiary)",
          border: "none",
          background: "none",
          cursor: "pointer",
          padding: "6px 4px",
          margin: "-6px -4px",
        }}
      >
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {t("review.hiddenCount", { count: entries.length })}
        </span>
        {` · ${t("review.showHidden")}`}
      </button>
      {expanded && (
        <section style={{ marginTop: 8 }}>
          <h2 style={sectionTitleStyle}>
            {t("review.hiddenSectionTitle")}
            <span aria-hidden="true" style={sectionCountStyle}>
              {entries.length}
            </span>
            <button
              type="button"
              onClick={onRestoreAll}
              style={{
                marginLeft: "auto",
                fontFamily: "var(--mem-font-body)",
                fontSize: 12.5,
                fontWeight: 400,
                textTransform: "none",
                letterSpacing: "normal",
                color: "var(--mem-accent-indigo)",
                border: "none",
                background: "none",
                cursor: "pointer",
              }}
            >
              {t("review.restoreAll")}
            </button>
          </h2>
          <div className="grid gap-1.5">
            {entries.map((entry) => {
              const tone = hiddenKindTone(entry.kind);
              return (
                <div
                  key={entry.key}
                  style={{
                    ...itemSurfaceStyle,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 12px",
                  }}
                >
                  <span
                    style={{
                      fontSize: 11,
                      letterSpacing: "0.04em",
                      borderRadius: 5,
                      padding: "2px 8px",
                      color: tone.color,
                      backgroundColor: tone.background,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {hiddenKindLabel(t, entry.kind)}
                  </span>
                  <span
                    style={{
                      ...secondaryTextStyle,
                      flex: 1,
                      minWidth: 0,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontSize: 13,
                    }}
                  >
                    {truncateText(entry.label, 60)}
                  </span>
                  <button
                    type="button"
                    onClick={() => onRestore(entry.key)}
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      fontSize: 12.5,
                      color: "var(--mem-accent-indigo)",
                      border: "none",
                      background: "none",
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {t("review.restore")}
                  </button>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

/** Wrapping pill-chip row under the header, not a segmented control — the
 * category count is dynamic (2-7 chips), which reads wrong as segments. */
function FilterChipRow({
  filter,
  onSelect,
  counts,
  allCount,
}: {
  filter: ReviewFilterKey;
  onSelect: (filter: ReviewFilterKey) => void;
  counts: Partial<Record<ReviewSection, number>>;
  allCount: number;
}) {
  const { t } = useTranslation();
  // "All" and "Revisions" always render — Revisions existing even at 0 shows
  // the section exists. Every other chip only appears once it has items.
  const chips: ReviewFilterKey[] = [
    "all",
    ...FILTER_ORDER.filter((key) => key === "revisions" || (counts[key] ?? 0) > 0),
  ];
  return (
    <div
      role="group"
      aria-label={t("review.filterLabel")}
      style={{ display: "flex", flexWrap: "wrap", gap: 8, margin: "0 0 18px" }}
    >
      {chips.map((key) => {
        const selected = filter === key;
        const count = key === "all" ? allCount : counts[key] ?? 0;
        return (
          <button
            key={key}
            type="button"
            aria-pressed={selected}
            onClick={() => onSelect(key)}
            className="transition-colors duration-150 hover:bg-[var(--mem-hover)] focus-visible:outline-2 focus-visible:outline-[var(--mem-accent-indigo)] focus-visible:outline-offset-2"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 12,
              lineHeight: 1,
              padding: "6px 12px",
              borderRadius: 999,
              cursor: "pointer",
              border: selected
                ? "1px solid color-mix(in srgb, var(--mem-accent-indigo) 35%, transparent)"
                : "1px solid var(--mem-border)",
              color: selected ? "var(--mem-accent-indigo)" : "var(--mem-text-secondary)",
              backgroundColor: selected ? "var(--mem-indigo-bg)" : "transparent",
            }}
          >
            {t(FILTER_LABEL_KEYS[key])}
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontVariantNumeric: "tabular-nums",
                fontSize: 11,
                marginLeft: 6,
              }}
            >
              {count}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default function DistillReviewPanel({
  onBack,
  onPageClick,
  onMemoryClick,
}: DistillReviewPanelProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [lastResult, setLastResult] = useState<DistillReviewResponse | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  // Session-only — a review queue is a to-do list, so every mount starts at
  // the top instead of remembering the last category.
  const [filter, setFilter] = useState<ReviewFilterKey>("all");
  // Stale pages refreshed this session — a new distill result resets the set.
  const [resolvedStaleIds, setResolvedStaleIds] = useState<ReadonlySet<string>>(
    new Set(),
  );
  const didLoadInitialReview = useRef(false);
  const queue = useReviewQueue();
  const { hiddenKeys, hiddenEntries, hide, restore, restoreAll } =
    useSuppressedReviewItems();
  const review = useMutation({
    mutationFn: distillReview,
    retry: false,
    onSuccess: (result) => {
      setLastResult(result);
      cacheDistillReviewSession(queryClient, result);
      setResolvedStaleIds(new Set());
    },
  });
  // Mutation errors only — a failed queue fetch (queue.error) gets its own
  // dedicated UI in the caught-up slot / section list below, so it never
  // doubles up with this box.
  const rawError = review.error;
  const error = rawError instanceof Error
    ? rawError.message
    : rawError
      ? String(rawError)
      : null;

  // Deferred past StrictMode's setup→cleanup→setup cycle: mutating inside the
  // first effect setup detaches the mutation observer on cleanup, leaving the
  // hook stuck reporting pending after the mutation has already succeeded.
  useEffect(() => {
    const id = setTimeout(() => {
      if (didLoadInitialReview.current) return;
      didLoadInitialReview.current = true;
      review.mutate();
    }, 0);
    return () => clearTimeout(id);
  }, [review.mutate]);

  // Dev-only sample items (see reviewExamples.ts) read these caches through
  // the real ReviewDialog query hooks — seed once so no example ever fetches.
  useEffect(() => {
    if (REVIEW_EXAMPLES_ENABLED) seedReviewExampleCaches(queryClient);
  }, [queryClient]);

  // New-memory captures are inflow, not decisions — they surface as a count
  // on the home context rail instead of flooding this queue.
  const decisionItems = queue.items.filter((item) => item.kind !== "capture");

  // These three kinds have no daemon dismiss verb — "Hide" persists locally
  // instead (see reviewSuppression.ts); a hidden item drops out here so it
  // never reappears after a re-render or a fresh distill result.
  const isHiddenItem = (item: ReviewItem) => {
    const key = reviewSuppressKey(item);
    return key != null && hiddenKeys.has(key);
  };

  // Distill discovery rendered through the same card + dialog pattern as the
  // actionable queue, read-only until the daemon grows verbs for them.
  const candidateItems = pageCandidateItems(
    lastResult ?? undefined,
    t("review.untitledCluster"),
  ).filter((item) => !isHiddenItem(item));
  const topicItems = (lastResult?.orphan_topics ?? [])
    .map(
      (topic): ReviewItem => ({
        kind: "topic",
        id: topic.label,
        label: topic.label,
        count: topic.count,
        timestampMs: null,
      }),
    )
    .filter((item) => !isHiddenItem(item));
  // Compiled pages whose sources changed — actionable: approve re-distills.
  const stalePageItems = (lastResult?.stale_pages ?? [])
    .filter((page) => !resolvedStaleIds.has(page.page_id))
    .map(
      (page): ReviewItem => ({
        kind: "stale_page",
        id: page.page_id,
        title: page.title,
        summary: page.summary ?? null,
        sourcesUpdated: page.sources_updated_count ?? null,
        timestampMs: null,
      }),
    )
    .filter((item) => !isHiddenItem(item));
  // Dialog order mirrors the page: decisions, page refreshes, then discovery.
  // Also the source of chip counts — counted post-suppression, same as what
  // the page actually shows.
  const allVisible = [
    ...decisionItems,
    ...stalePageItems,
    ...candidateItems,
    ...topicItems,
  ];
  // Read-only distill discovery — surfaced separately from the pending-decision
  // count so the header reads "N to decide · +M discovered" (= the All total),
  // making the home badge (decisions only) an honest subset, not a mismatch.
  const discoveredCount =
    stalePageItems.length + candidateItems.length + topicItems.length;
  const sectionCounts = allVisible.reduce<Partial<Record<ReviewSection, number>>>(
    (acc, item) => {
      const section = reviewItemSection(item);
      acc[section] = (acc[section] ?? 0) + 1;
      return acc;
    },
    {},
  );
  const matchesFilter = (item: ReviewItem) =>
    filter === "all" || reviewItemSection(item) === filter;
  // Grouped under Revisions regardless of the filter chip they'd naturally
  // route to (a contradiction sample is "conflicts") — both examples live in
  // one teachable spot, so visibility follows that section's filter gate.
  const examplesVisible = filter === "all" || filter === "revisions";
  // Shown whenever no real revision items exist — the samples teach what a
  // revision review looks like, which the daemon otherwise never populates.
  // Gated on the revisions section (not the whole queue) so they still appear
  // alongside real work in other categories. Dev-only.
  const showExamples =
    REVIEW_EXAMPLES_ENABLED &&
    !queue.isLoading &&
    !queue.error &&
    (sectionCounts.revisions ?? 0) === 0;
  const exampleItems = showExamples
    ? EXAMPLE_REVIEW_ITEMS.filter((item) => !hiddenKeys.has(item.id))
    : [];
  // Arrow-key navigation in the dialog stays inside the active filter.
  const dialogItems = [
    ...allVisible.filter(matchesFilter),
    ...(examplesVisible ? exampleItems : []),
  ];

  // Stale-page refreshes resolve against the panel's distill result, not the
  // queue caches — drop them here after the daemon verb succeeds.
  const resolveItem = async (args: { item: ReviewItem; approve: boolean }) => {
    const { item, approve } = args;
    // Examples can never reach the daemon — dismiss hides locally through the
    // suppression store, accept is a silent no-op (see ReviewDialog's
    // reviewApproveBlocked, which already hides the Approve button).
    if (isExampleReviewItem(item)) {
      if (!approve) hide(item);
      return;
    }
    // The three read-only discovery kinds have no daemon dismiss verb — "no"
    // means hide it locally instead of calling the queue (which no-ops for
    // page_candidate/topic and has no dismiss path for stale_page anyway).
    if (
      !approve &&
      (item.kind === "stale_page" ||
        item.kind === "page_candidate" ||
        item.kind === "topic")
    ) {
      hide(item);
      return;
    }
    const result = await queue.resolve(args);
    if (item.kind === "stale_page" && approve) {
      setResolvedStaleIds((prev) => new Set(prev).add(item.id));
    }
    return result;
  };

  // Section order mirrors the queue's revisions > conflicts > pages ranking.
  const sections = [
    { key: "revisions", title: t("review.sectionRevisions") },
    { key: "conflicts", title: t("review.sectionConflicts") },
    { key: "pages", title: t("review.sectionPages") },
    { key: "memory", title: t("review.sectionRefinements") },
  ].map((section) => ({
    ...section,
    items: decisionItems.filter(
      (item) => reviewItemSection(item) === section.key,
    ),
  }));

  // "All caught up" must count the distill discovery too — an empty decision
  // queue with pending page work is not caught up.
  const distillHasWork =
    lastResult != null &&
    (candidateItems.length > 0 ||
      stalePageItems.length > 0 ||
      topicItems.length > 0);
  const allCaughtUp =
    !queue.isLoading &&
    !queue.error &&
    !review.isPending &&
    !review.error &&
    decisionItems.length === 0 &&
    !distillHasWork;

  const refresh = () => {
    review.mutate();
    queryClient.invalidateQueries({ queryKey: ["pending-revisions"] });
    queryClient.invalidateQueries({ queryKey: ["refinement-proposals"] });
    queryClient.invalidateQueries({ queryKey: ["unconfirmed-captures"] });
  };

  return (
    <div
      className="flex flex-col"
      style={{
        ...panelTextStyle,
        padding: "28px 36px",
        animation: "mem-fade-up 350ms cubic-bezier(0.16, 1, 0.3, 1) both",
      }}
    >
      <div className="flex items-center mb-5">
        <button
          type="button"
          onClick={onBack}
          aria-label="Back"
          className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{
            color: "var(--mem-text-tertiary)",
            background: "none",
            border: "none",
            cursor: "pointer",
            lineHeight: 0,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
        </button>
      </div>

      <div className="flex items-start justify-between gap-4">
        <div>
          <h1
            style={{
              margin: 0,
              fontFamily: "var(--mem-font-heading)",
              fontSize: "22px",
              fontWeight: 500,
              color: "var(--mem-text)",
            }}
          >
            {t("review.title")}
          </h1>
          <p style={{ ...secondaryTextStyle, margin: "5px 0 0", fontSize: "13px" }}>
            {t("review.subtitle")}
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          {decisionItems.length > 0 && (
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontVariantNumeric: "tabular-nums",
                fontSize: 12,
                color: "var(--mem-accent-indigo)",
                backgroundColor: "var(--mem-indigo-bg)",
                borderRadius: 999,
                padding: "4px 11px",
                whiteSpace: "nowrap",
              }}
            >
              {t("review.pendingCount", {
                count: queue.decisionsTruncated
                  ? `${decisionItems.length}+`
                  : decisionItems.length,
              })}
            </span>
          )}
          {discoveredCount > 0 && (
            <span
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontVariantNumeric: "tabular-nums",
                fontSize: 12,
                color: "var(--mem-text-secondary)",
                border: "1px solid var(--mem-border)",
                backgroundColor: "transparent",
                borderRadius: 999,
                padding: "3px 10px",
                whiteSpace: "nowrap",
              }}
            >
              {t("review.discoveredCount", { count: discoveredCount })}
            </span>
          )}
          <button
            type="button"
            onClick={refresh}
            disabled={review.isPending}
            className="rounded-md px-3 py-2 text-sm transition-colors duration-150 hover:bg-[var(--mem-hover)] disabled:opacity-60"
            style={{
              border: "1px solid var(--mem-border)",
              backgroundColor: "var(--mem-surface)",
              color: "var(--mem-text)",
              cursor: review.isPending ? "default" : "pointer",
              fontFamily: "var(--mem-font-body)",
            }}
          >
            {review.isPending ? t("review.refreshing") : t("review.refresh")}
          </button>
        </div>
      </div>

      <FilterChipRow
        filter={filter}
        onSelect={setFilter}
        counts={sectionCounts}
        allCount={allVisible.length}
      />

      {error && (
        <div
          role="alert"
          style={{
            ...itemSurfaceStyle,
            marginTop: 18,
            padding: "10px 12px",
            color: "var(--mem-text)",
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
          }}
        >
          {error}
        </div>
      )}

      <div className="grid gap-6" style={{ marginTop: 24 }}>
        {filter === "all" && allCaughtUp && (
          <section>
            <h2 style={emptyTitleStyle}>{t("review.allCaughtUp")}</h2>
            <p style={{ ...secondaryTextStyle, margin: 0, fontSize: "13px" }}>
              {t("review.allCaughtUpHint")}
            </p>
          </section>
        )}

        {queue.error && decisionItems.length === 0 && (
          <section>
            <h2 style={emptyTitleStyle}>{t("review.loadFailed")}</h2>
            <button
              type="button"
              onClick={refresh}
              className="rounded-md px-3 py-2 text-sm transition-colors duration-150 hover:bg-[var(--mem-hover)]"
              style={{
                border: "1px solid var(--mem-border)",
                backgroundColor: "var(--mem-surface)",
                color: "var(--mem-text)",
                cursor: "pointer",
                fontFamily: "var(--mem-font-body)",
              }}
            >
              {t("review.retry")}
            </button>
          </section>
        )}

        {queue.error && decisionItems.length > 0 && (
          <p style={{ ...secondaryTextStyle, margin: 0, fontSize: "12px" }}>
            {t("review.loadPartial")}
          </p>
        )}

        {/* Revisions keeps its own richer empty state below instead of this
         * generic box — see the "revisions" branch in the sections.map. No
         * auto-jump back to "all" when a filter empties; Show all is the
         * escape hatch. */}
        {filter !== "all" && filter !== "revisions" && (sectionCounts[filter] ?? 0) === 0 && (
          <div
            style={{
              ...itemSurfaceStyle,
              padding: "18px 16px",
              display: "grid",
              gap: 10,
              justifyItems: "start",
            }}
          >
            <p
              style={{
                margin: 0,
                fontFamily: "var(--mem-font-body)",
                fontSize: 12.5,
                color: "var(--mem-text-secondary)",
              }}
            >
              {t("review.filterEmpty")}
            </p>
            <button
              type="button"
              onClick={() => setFilter("all")}
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: 12,
                color: "var(--mem-accent-indigo)",
                background: "none",
                border: "none",
                padding: 0,
                cursor: "pointer",
              }}
            >
              {t("review.filterShowAll")}
            </button>
          </div>
        )}

        {sections.map((section) => {
          const hasItems = section.items.length > 0;
          if (section.key === "revisions") {
            // Always-on once the queue has settled — proves revisions exist
            // even when nothing is pending. Hidden during a real load
            // failure: the loadFailed/loadPartial blocks above already
            // cover that state. Hidden under any other filter — the generic
            // empty box above never applies to revisions (it keeps this
            // richer empty state instead).
            if (filter !== "all" && filter !== "revisions") return null;
            if (queue.isLoading || (!hasItems && queue.error)) return null;
            return (
              <section key={section.key}>
                <h2 style={sectionTitleStyle}>
                  {section.title}
                  <span aria-hidden="true" style={sectionCountStyle}>
                    {section.items.length}
                  </span>
                </h2>
                {hasItems ? (
                  <div className="grid gap-2.5">
                    {section.items.map((item) => (
                      <QueueCard
                        key={reviewItemId(item)}
                        item={item}
                        onOpen={setOpenId}
                      />
                    ))}
                  </div>
                ) : exampleItems.length > 0 ? (
                  <>
                    <p
                      style={{
                        margin: "0 0 10px",
                        fontFamily: "var(--mem-font-mono)",
                        fontSize: 11,
                        color: "var(--mem-text-tertiary)",
                      }}
                    >
                      {t("review.exampleHint")}
                    </p>
                    <div className="grid gap-2.5">
                      {exampleItems.map((item) => (
                        <QueueCard
                          key={reviewItemId(item)}
                          item={item}
                          onOpen={setOpenId}
                          example
                        />
                      ))}
                    </div>
                  </>
                ) : (
                  <div style={{ ...itemSurfaceStyle, padding: "13px 14px" }}>
                    <p style={{ ...secondaryTextStyle, margin: 0, fontSize: 13.5 }}>
                      {t("review.revisionsEmptyTitle")}
                    </p>
                    <p
                      style={{
                        ...secondaryTextStyle,
                        margin: "6px 0 0",
                        fontSize: 13,
                        lineHeight: 1.5,
                        color: "var(--mem-text-tertiary)",
                      }}
                    >
                      {t("review.revisionsEmptyBody")}
                    </p>
                  </div>
                )}
              </section>
            );
          }
          return (
            (filter === "all" || filter === section.key) &&
            hasItems && (
              <section key={section.key}>
                <h2 style={sectionTitleStyle}>
                  {section.title}
                  <span aria-hidden="true" style={sectionCountStyle}>
                    {section.items.length}
                  </span>
                </h2>
                <div className="grid gap-2.5">
                  {section.items.map((item) => (
                    <QueueCard
                      key={reviewItemId(item)}
                      item={item}
                      onOpen={setOpenId}
                    />
                  ))}
                </div>
              </section>
            )
          );
        })}

        {/* Stale pages share the "pages" filter with the refinement-driven
         * pages section above. Under "all" this is unchanged (shows even at
         * zero, via pagesCurrent); under any other filter it only renders
         * with real items — the generic empty box covers "pages" at zero
         * across both sources. */}
        {lastResult &&
          (filter === "all" || (filter === "pages" && stalePageItems.length > 0)) && (
          <section>
            <h2 style={sectionTitleStyle}>
              {t("review.sectionStalePages")}
              {stalePageItems.length > 0 && (
                <span aria-hidden="true" style={sectionCountStyle}>
                  {stalePageItems.length}
                </span>
              )}
            </h2>
            {lastResult.stale_truncated && (
              <p style={{ ...secondaryTextStyle, margin: "0 0 10px", fontSize: "12px" }}>
                {t("review.staleTruncated")}
              </p>
            )}
            {stalePageItems.length === 0 ? (
              <p style={{ ...secondaryTextStyle, margin: 0, fontSize: "13px" }}>
                {t("review.pagesCurrent")}
              </p>
            ) : (
              <div className="grid gap-2.5">
                {stalePageItems.map((item) => (
                  <QueueCard
                    key={reviewItemId(item)}
                    item={item}
                    onOpen={setOpenId}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {(filter === "all" || filter === "candidates") && candidateItems.length > 0 && (
          <section>
            <h2 style={sectionTitleStyle}>
              {t("review.sectionPageCandidates")}
              <span aria-hidden="true" style={sectionCountStyle}>
                {candidateItems.length}
              </span>
            </h2>
            <div className="grid gap-2.5">
              {candidateItems.map((item) => (
                <QueueCard
                  key={reviewItemId(item)}
                  item={item}
                  onOpen={setOpenId}
                />
              ))}
            </div>
          </section>
        )}

        {(filter === "all" || filter === "topics") &&
          lastResult &&
          topicItems.length > 0 && (
          <section>
            <h2 style={sectionTitleStyle}>
              {t("review.sectionOrphanTopics")}
              <span aria-hidden="true" style={sectionCountStyle}>
                {topicItems.length}
              </span>
            </h2>
            <div className="grid gap-2.5">
              {topicItems.map((item) => (
                <QueueCard
                  key={reviewItemId(item)}
                  item={item}
                  onOpen={setOpenId}
                />
              ))}
            </div>
          </section>
        )}

        {/* Changelog, not category work — noise under a category filter. */}
        {(filter === "all" || filter === "revisions") && (
          <RecentRevisionsSection onPageClick={onPageClick} />
        )}

        {/* Housekeeping, not category work. */}
        {filter === "all" && hiddenEntries.length > 0 && (
          <HiddenFooter
            entries={hiddenEntries}
            onRestore={restore}
            onRestoreAll={restoreAll}
          />
        )}
      </div>

      <ReviewDialog
        items={dialogItems}
        openId={openId}
        onOpenChange={setOpenId}
        onResolve={resolveItem}
        isResolving={queue.isResolving}
        onOpenMemory={onMemoryClick}
        onOpenPage={onPageClick}
      />
    </div>
  );
}
