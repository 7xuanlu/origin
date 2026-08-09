// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  getEntityDetail,
  getMemoryDetail,
  getPage,
  search,
} from "../../lib/tauri";
import { diffWords, diffWordCounts, type DiffSegment } from "../../lib/wordDiff";
import { isExampleReviewItem } from "./reviewExamples";
import { reviewItemId, type ReviewItem } from "./useReviewQueue";
import { PageMergeStripOff } from "./ReviewPageMerge";
import { MemoryRevisionChain } from "./ReviewHistory";

export function reviewKindLabel(t: TFunction, item: ReviewItem): string {
  if (item.kind === "revision") return t("review.kindRevision");
  if (item.kind === "capture") return t("review.kindCapture");
  if (item.kind === "stale_page") return t("review.kindPageRefresh");
  if (item.kind === "page_candidate") return t("review.kindPageCandidate");
  if (item.kind === "topic") return t("review.kindTopic");
  switch (item.action) {
    case "entity_merge":
      return t("review.kindEntityMerge");
    case "detect_contradiction":
      return t("review.kindContradiction");
    case "dedup_merge":
      return t("review.kindDuplicate");
    case "relation_conflict":
      return t("review.kindRelationConflict");
    case "suggest_entity":
      return t("review.kindEntitySuggestion");
    case "page_merge":
      return t("review.kindPageMerge");
    case "cross_space_discovery":
      return t("review.kindCrossSpace");
    case "page_keep_or_archive":
      return t("review.kindPageArchive");
  }
}

/** Distill discovery items have no daemon verb at all — the dialog shows them
 * without approve or dismiss; the daemon's compile pass consumes them. */
export function reviewReadOnly(item: ReviewItem): boolean {
  return item.kind === "page_candidate" || item.kind === "topic";
}

function pageKeepOrArchiveId(item: ReviewItem | null): string | null {
  if (
    item?.kind !== "refinement" ||
    item.action !== "page_keep_or_archive" ||
    item.payload?.action !== "page_keep_or_archive"
  ) {
    return null;
  }
  const pageId = item.payload.page_id.trim();
  return pageId.length > 0 ? pageId : null;
}

/** Actions the daemon rejects with 422 on accept — the dialog offers only
 * dismiss for these (suggest_entity/dedup_merge have no accept path;
 * cross_space_discovery needs a pick-space verb the app doesn't plumb yet). */
export function reviewApproveBlocked(item: ReviewItem): boolean {
  if (isExampleReviewItem(item)) return true;
  if (
    item.kind === "refinement" &&
    item.action === "page_keep_or_archive" &&
    pageKeepOrArchiveId(item) === null
  ) {
    return true;
  }
  return (
    reviewReadOnly(item) ||
    (item.kind === "refinement" &&
      (item.action === "suggest_entity" ||
        item.action === "dedup_merge" ||
        item.action === "cross_space_discovery"))
  );
}

/** Every kind now has a dismiss-side action: read-only discovery and
 * stale-page refreshes hide locally (see DistillReviewPanel's resolveItem
 * wrapper) rather than calling a daemon dismiss verb. */
export function reviewDismissBlocked(_item: ReviewItem): boolean {
  return false;
}

/** Per-kind chip tone: revisions indigo, page work warm, entity merges amber,
 * conflicts danger, new entities/topics sage. */
export function reviewKindTone(item: ReviewItem): {
  color: string;
  background: string;
} {
  const mix = (token: string) =>
    `color-mix(in srgb, ${token} 15%, transparent)`;
  if (item.kind === "revision" || item.kind === "capture") {
    return {
      color: "var(--mem-accent-indigo)",
      background: "var(--mem-indigo-bg)",
    };
  }
  if (item.kind === "page_candidate" || item.kind === "stale_page") {
    return {
      color: "var(--mem-accent-warm)",
      background: mix("var(--mem-accent-warm)"),
    };
  }
  if (item.kind === "topic") {
    return {
      color: "var(--mem-accent-amber)",
      background: mix("var(--mem-accent-amber)"),
    };
  }
  switch (item.action) {
    case "page_merge":
    case "page_keep_or_archive":
      return {
        color: "var(--mem-accent-warm)",
        background: mix("var(--mem-accent-warm)"),
      };
    case "entity_merge":
    case "dedup_merge":
      return {
        color: "var(--mem-accent-amber)",
        background: mix("var(--mem-accent-amber)"),
      };
    case "detect_contradiction":
    case "relation_conflict":
      return {
        color: "var(--mem-status-danger-text)",
        background: "var(--mem-status-danger-bg)",
      };
    default:
      // suggest_entity / cross_space_discovery — new entities and spaces.
      return {
        color: "var(--mem-accent-amber)",
        background: mix("var(--mem-accent-amber)"),
      };
  }
}

export function truncateReviewText(value: string, max: number): string {
  const trimmed = value.trim().replace(/\s+/g, " ");
  if (trimmed.length <= max) return trimmed;
  return `${trimmed.slice(0, max - 3).trimEnd()}...`;
}

type ReviewLookup = "page" | "entity" | "memory" | null;

/** Resolve which two ids a review item's evidence points at, if any. */
function reviewLookupRefs(item: ReviewItem | null): {
  lookup: ReviewLookup;
  aId: string | null;
  bId: string | null;
} {
  if (item?.kind === "revision") {
    return { lookup: "memory", aId: item.targetSourceId, bId: null };
  }
  if (item?.kind !== "refinement") return { lookup: null, aId: null, bId: null };
  switch (item.action) {
    case "page_merge":
      return {
        lookup: "page",
        aId: item.sourceIds[0] ?? null,
        bId: item.sourceIds[1] ?? null,
      };
    case "page_keep_or_archive": {
      const pageId = pageKeepOrArchiveId(item);
      return pageId !== null
        ? { lookup: "page", aId: pageId, bId: null }
        : { lookup: null, aId: null, bId: null };
    }
    case "entity_merge":
      return item.payload?.action === "entity_merge"
        ? {
            lookup: "entity",
            aId: item.payload.existing_id,
            bId: item.payload.new_id,
          }
        : { lookup: null, aId: null, bId: null };
    case "detect_contradiction":
    case "dedup_merge":
      return {
        lookup: "memory",
        aId: item.sourceIds[0] ?? null,
        bId: item.sourceIds[1] ?? null,
      };
    default:
      // relation_conflict / suggest_entity / cross_space_discovery carry
      // their evidence in the payload; their source_ids are not fetchable.
      return { lookup: null, aId: null, bId: null };
  }
}

async function fetchReviewName(
  lookup: Exclude<ReviewLookup, null>,
  id: string,
): Promise<{ name: string | null; text: string | null }> {
  if (lookup === "page") {
    const page = await getPage(id);
    return { name: page?.title ?? null, text: null };
  }
  if (lookup === "entity") {
    const detail = await getEntityDetail(id);
    return { name: detail?.entity.name ?? null, text: null };
  }
  const detail = await getMemoryDetail(id);
  return {
    name: detail?.title?.trim() || detail?.content || null,
    text: detail?.content ?? null,
  };
}

/**
 * Rich card/rail summary for a review item: fetches the names its ids point
 * at and surfaces the payload evidence (page names, overlap, similarity, word
 * delta) so an item never reads as just its kind label. Falls back to the
 * kind label while names load.
 */
export function useReviewItemSummary(item: ReviewItem | null): {
  title: string;
  reason: string | null;
  delta: { added: number; removed: number } | null;
} {
  const { t } = useTranslation();
  const { lookup, aId, bId } = reviewLookupRefs(item);
  const a = useQuery({
    queryKey: ["review-summary", lookup, aId],
    queryFn: () => fetchReviewName(lookup as Exclude<ReviewLookup, null>, aId as string),
    enabled: lookup != null && aId != null,
    staleTime: 60_000,
  });
  const b = useQuery({
    queryKey: ["review-summary", lookup, bId],
    queryFn: () => fetchReviewName(lookup as Exclude<ReviewLookup, null>, bId as string),
    enabled: lookup != null && bId != null,
    staleTime: 60_000,
  });
  const delta = useMemo(
    () =>
      item?.kind === "revision" && a.data?.text
        ? diffWordCounts(diffWords(a.data.text, item.content))
        : null,
    [item, a.data],
  );
  if (!item) return { title: "", reason: null, delta: null };

  const short = (value: string | null | undefined, max = 36): string | null => {
    const trimmed = value?.trim();
    return trimmed ? truncateReviewText(trimmed, max) : null;
  };
  const aName = short(a.data?.name);
  const bName = short(b.data?.name);
  let title: string | null = null;
  let reason: string | null = null;
  if (item.kind === "revision") {
    title = short(a.data?.name, 96) ?? short(item.content, 96);
    reason = item.agent ? t("review.proposedBy", { agent: item.agent }) : null;
  } else if (item.kind === "capture") {
    title = short(item.title, 96);
    reason = short(item.snippet, 96);
  } else if (item.kind === "page_candidate") {
    title = item.title;
    reason =
      (item.cluster.new_memory_count != null
        ? t("review.newSources", { count: item.cluster.new_memory_count })
        : t("review.sources", { count: item.cluster.source_ids.length })) +
      (item.cluster.existing_page_id
        ? ` · ${t("review.linkedExistingPage")}`
        : "");
  } else if (item.kind === "topic") {
    title = item.label;
    reason = t("review.topicReason", { count: item.count });
  } else if (item.kind === "stale_page") {
    title = item.title;
    reason =
      item.sourcesUpdated != null
        ? t("review.sourcesUpdated", { count: item.sourcesUpdated })
        : null;
  } else {
    const confidence = t("review.confidence", {
      percent: Math.round(item.confidence * 100),
    });
    reason = confidence;
    switch (item.action) {
      case "page_merge":
        if (aName && bName)
          title = t("review.pageMergeTitle", { keep: aName, absorb: bName });
        if (item.payload?.action === "page_merge")
          reason = t("review.mergeReason", {
            count: item.payload.source_overlap,
            percent: Math.round(item.payload.source_overlap_ratio * 100),
          });
        break;
      case "entity_merge":
        if (aName && bName)
          title = t("review.entityMergeTitle", { a: aName, b: bName });
        if (item.payload?.action === "entity_merge")
          reason = t("review.similarity", {
            percent: Math.round(item.payload.similarity * 100),
          });
        break;
      case "detect_contradiction":
        if (aName && bName)
          title = t("review.contradictionTitle", { a: aName, b: bName });
        break;
      case "dedup_merge":
        if (aName && bName)
          title = t("review.dedupTitle", { a: aName, b: bName });
        break;
      case "page_keep_or_archive":
        title = short(a.data?.name, 96);
        if (item.payload?.action === "page_keep_or_archive")
          reason = `${t("review.sources", { count: item.payload.source_count })} · ${confidence}`;
        break;
      case "relation_conflict":
        if (item.payload?.action === "relation_conflict")
          title = `${item.payload.from} → ${item.payload.to}`;
        break;
      case "suggest_entity":
        if (item.payload?.action === "suggest_entity")
          title = short(item.payload.name_hint, 96);
        break;
      case "cross_space_discovery":
        if (item.payload?.action === "cross_space_discovery")
          title = item.payload.spaces.join(" · ");
        break;
    }
  }
  return { title: title ?? reviewKindLabel(t, item), reason, delta };
}

const INS_STYLE: React.CSSProperties = {
  backgroundColor: "color-mix(in srgb, var(--mem-accent-sage) 24%, transparent)",
  textDecoration: "none",
  borderRadius: 3,
  padding: "0 2px",
};

const DEL_STYLE: React.CSSProperties = {
  backgroundColor: "var(--mem-status-danger-bg)",
  borderRadius: 3,
  padding: "0 2px",
};

const paneStyle: React.CSSProperties = {
  border: "1px solid var(--mem-border)",
  borderRadius: 10,
  backgroundColor: "var(--mem-detail-surface-raised)",
  padding: "13px 15px",
  fontFamily: "var(--mem-font-body)",
  fontSize: 14,
  lineHeight: 1.65,
  color: "var(--mem-text)",
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
};

const paneLabelStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 11,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
  margin: "0 0 7px",
};

const actionButtonStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 13.5,
  borderRadius: 8,
  padding: "8px 15px",
  cursor: "pointer",
  border: "1px solid var(--mem-border)",
  backgroundColor: "var(--mem-surface)",
  color: "var(--mem-text)",
};

/** Dashed pill marking a dev-only sample item (see reviewExamples.ts) — card
 * and dialog share this recipe. */
const examplePillStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-mono)",
  fontSize: 10.5,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  borderRadius: 5,
  padding: "1px 7px",
  color: "var(--mem-text-tertiary)",
  border: "1px dashed var(--mem-border)",
  whiteSpace: "nowrap",
};

/** On-open evidence for topic/suggest_entity cards: their daemon payloads
 * carry no source ids to look up, so the dialog runs a search for the topic
 * label / entity name hint instead — never per-card, only once the dialog is
 * actually open. */
function useReviewEvidence(query: string | null) {
  return useQuery({
    queryKey: ["review-evidence", query],
    queryFn: () => search(query as string, 4),
    enabled: query != null,
    staleTime: 60_000,
  });
}

const evidenceRowStyle: React.CSSProperties = {
  display: "block",
  width: "100%",
  textAlign: "left",
  border: "1px solid var(--mem-border)",
  borderRadius: 8,
  padding: "8px 12px",
  backgroundColor: "var(--mem-surface)",
  cursor: "pointer",
};

function ReviewEvidencePane({
  query,
  onOpenMemory,
}: {
  query: string | null;
  onOpenMemory?: (sourceId: string) => void;
}) {
  const { t } = useTranslation();
  const evidence = useReviewEvidence(query);
  if (query == null) return null;
  const results = (evidence.data ?? []).slice(0, 3);
  return (
    <div>
      <p style={paneLabelStyle}>{t("review.mentionedIn")}</p>
      {evidence.isLoading ? (
        <div style={paneStyle}>{t("review.loadingCurrent")}</div>
      ) : results.length === 0 ? (
        <div style={paneStyle}>{t("review.evidenceNone")}</div>
      ) : (
        <div style={{ display: "grid", gap: 8 }}>
          {results.map((result) => (
            <button
              key={result.id}
              type="button"
              onClick={() => onOpenMemory?.(result.source_id)}
              style={evidenceRowStyle}
            >
              <div
                style={{
                  fontFamily: "var(--mem-font-heading)",
                  fontWeight: 500,
                  fontSize: 13.5,
                  color: "var(--mem-text)",
                }}
              >
                {result.title || truncateReviewText(result.content, 60)}
              </div>
              <div
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: 12,
                  color: "var(--mem-text-tertiary)",
                  marginTop: 2,
                }}
              >
                {truncateReviewText(result.content, 100)}
              </div>
            </button>
          ))}
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              color: "var(--mem-text-tertiary)",
              fontSize: 12,
              margin: 0,
            }}
          >
            {t("review.evidenceBySearch")}
          </p>
        </div>
      )}
    </div>
  );
}

function DiffText({ segments }: { segments: DiffSegment[] }) {
  return (
    <>
      {segments.map((segment, index) =>
        segment.kind === "ins" ? (
          <ins key={index} style={INS_STYLE}>
            {segment.text}
          </ins>
        ) : segment.kind === "del" ? (
          <del key={index} style={DEL_STYLE}>
            {segment.text}
          </del>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </>
  );
}

interface ReviewDialogProps {
  items: ReviewItem[];
  openId: string | null;
  onOpenChange: (id: string | null) => void;
  onResolve: (args: { item: ReviewItem; approve: boolean }) => Promise<unknown>;
  isResolving: boolean;
  onOpenMemory?: (sourceId: string) => void;
  onOpenPage?: (pageId: string) => void;
}

export default function ReviewDialog({
  items,
  openId,
  onOpenChange,
  onResolve,
  isResolving,
  onOpenMemory,
  onOpenPage,
}: ReviewDialogProps) {
  const { t } = useTranslation();
  const [showDone, setShowDone] = useState(false);
  const [sideBySide, setSideBySide] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const [resolveError, setResolveError] = useState(false);
  const [resolvingLocally, setResolvingLocally] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const navigationVersionRef = useRef(0);
  const resolveRequestRef = useRef(0);

  const open = openId != null;
  const foundIndex = items.findIndex((entry) => reviewItemId(entry) === openId);
  const index = foundIndex >= 0 ? foundIndex : items.length > 0 ? 0 : -1;
  const item = showDone ? null : index >= 0 ? items[index] : null;
  const done = open && (showDone || items.length === 0);
  const resolving = isResolving || resolvingLocally;
  const summary = useReviewItemSummary(item);

  const detailSourceId =
    item?.kind === "revision"
      ? item.targetSourceId
      : item?.kind === "capture"
        ? item.id
        : null;
  const target = useQuery({
    queryKey: ["memory-detail", detailSourceId],
    queryFn: () => getMemoryDetail(detailSourceId as string),
    enabled: detailSourceId != null,
  });

  // Actions whose source_ids are memory ids; the rest point at entities,
  // pages, or relations and get dedicated panes below. suggest_entity's
  // source_ids aren't fetchable memory ids at all (see reviewLookupRefs
  // above) — it gets a search-backed evidence pane instead.
  const memoryPaneIds =
    item?.kind === "refinement" &&
    (item.action === "detect_contradiction" ||
      item.action === "dedup_merge" ||
      item.action === "cross_space_discovery")
      ? item.sourceIds.slice(0, 2)
      : [];
  const memoryPaneA = useQuery({
    queryKey: ["memory-detail", memoryPaneIds[0] ?? null],
    queryFn: () => getMemoryDetail(memoryPaneIds[0]),
    enabled: memoryPaneIds.length > 0,
  });
  const memoryPaneB = useQuery({
    queryKey: ["memory-detail", memoryPaneIds[1] ?? null],
    queryFn: () => getMemoryDetail(memoryPaneIds[1]),
    enabled: memoryPaneIds.length > 1,
  });

  const archivePageId = pageKeepOrArchiveId(item);
  const archivePage = useQuery({
    queryKey: ["page", archivePageId],
    queryFn: () => getPage(archivePageId as string),
    enabled: archivePageId != null,
  });

  const mergePayload =
    item?.kind === "refinement" && item.payload?.action === "entity_merge"
      ? item.payload
      : null;
  const mergeExisting = useQuery({
    queryKey: ["entity-detail", mergePayload?.existing_id ?? null],
    queryFn: () => getEntityDetail(mergePayload?.existing_id as string),
    enabled: mergePayload != null,
  });
  const mergeIncoming = useQuery({
    queryKey: ["entity-detail", mergePayload?.new_id ?? null],
    queryFn: () => getEntityDetail(mergePayload?.new_id as string),
    enabled: mergePayload != null,
  });

  const beforeContent = target.data?.content ?? "";
  const segments = useMemo(
    () =>
      item?.kind === "revision" && target.data
        ? diffWords(beforeContent, item.content)
        : [],
    [item, target.data, beforeContent],
  );
  const wordCounts = useMemo(() => diffWordCounts(segments), [segments]);

  const isContradiction =
    item?.kind === "refinement" && item.action === "detect_contradiction";
  // Daemon order: source_ids[0] is the new memory, source_ids[1] the existing
  // one — so pane A holds "after" and pane B holds "before".
  const contradictionSegments = useMemo(
    () =>
      isContradiction && memoryPaneA.data && memoryPaneB.data
        ? diffWords(memoryPaneB.data.content, memoryPaneA.data.content)
        : [],
    [isContradiction, memoryPaneA.data, memoryPaneB.data],
  );

  useEffect(() => {
    if (!open) return;
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    dialogRef.current?.focus();
    return () => {
      if (previouslyFocused?.isConnected) previouslyFocused.focus();
    };
  }, [open]);

  useEffect(() => {
    navigationVersionRef.current += 1;
    setResolveError(false);
  }, [openId]);

  const resolveCurrent = async (approve: boolean) => {
    if (!item || resolving) return;
    // reviewApproveBlocked already folds in reviewReadOnly — read-only kinds
    // (topic/page_candidate) can still dismiss (hide) even though they can't
    // approve.
    if (approve && reviewApproveBlocked(item)) return;
    if (!approve && reviewDismissBlocked(item)) return;
    const isCapture = item.kind === "capture";
    const isConflict =
      item.kind === "refinement" && item.action === "detect_contradiction";
    const isLocalHide =
      item.kind === "topic" ||
      item.kind === "page_candidate" ||
      item.kind === "stale_page";
    const next = items[index + 1] ?? (index > 0 ? items[index - 1] : null);
    const navigationVersion = navigationVersionRef.current;
    const requestId = ++resolveRequestRef.current;
    setResolveError(false);
    setResolvingLocally(true);
    try {
      await onResolve({ item, approve });
    } catch {
      if (
        resolveRequestRef.current === requestId &&
        navigationVersionRef.current === navigationVersion
      ) {
        setResolveError(true);
      }
      return;
    } finally {
      if (resolveRequestRef.current === requestId) {
        setResolvingLocally(false);
      }
    }
    if (
      resolveRequestRef.current !== requestId ||
      navigationVersionRef.current !== navigationVersion
    ) {
      return;
    }
    setFlash(
      approve
        ? t(
            isCapture
              ? "review.confirmed"
              : isConflict
                ? "review.resolved"
                : "review.approved",
          )
        : t(
            isCapture
              ? "review.forgotten"
              : isConflict
                ? "review.keptBoth"
                : isLocalHide
                  ? "review.hidden"
                  : "review.dismissed",
          ),
    );
    window.setTimeout(() => setFlash(null), 450);
    if (next) onOpenChange(reviewItemId(next));
    else setShowDone(true);
  };

  const goTo = (offset: number) => {
    if (items.length < 2) return;
    const nextIndex = (index + offset + items.length) % items.length;
    navigationVersionRef.current += 1;
    onOpenChange(reviewItemId(items[nextIndex]));
  };

  const close = () => {
    navigationVersionRef.current += 1;
    setShowDone(false);
    onOpenChange(null);
  };

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      const el = event.target as HTMLElement | null;
      if (event.key === "Tab") {
        const container = dialogRef.current;
        if (!container) return;
        const focusable = Array.from(
          container.querySelectorAll<HTMLElement>(
            'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
          ),
        );
        if (focusable.length === 0) {
          event.preventDefault();
          container.focus();
          return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = document.activeElement;
        const activeIsFocusable = active instanceof HTMLElement && focusable.includes(active);
        if (event.shiftKey && (!activeIsFocusable || active === first)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (!activeIsFocusable || active === last)) {
          event.preventDefault();
          first.focus();
        }
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        return;
      }
      if (
        el?.closest(
          'a[href], button, input, select, textarea, [contenteditable="true"], [role="button"]',
        )
      ) {
        return;
      }
      switch (event.key) {
        case "Enter":
          event.preventDefault();
          void resolveCurrent(true);
          break;
        case "d":
        case "D":
          void resolveCurrent(false);
          break;
        case "ArrowRight":
          goTo(1);
          break;
        case "ArrowLeft":
          goTo(-1);
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  if (!open) return null;

  const heading = done
    ? t("review.allCaughtUp")
    : item?.kind === "revision"
      ? (target.data?.title?.trim() ||
        truncateReviewText(item.content, 72))
      : item?.kind === "capture"
        ? truncateReviewText(item.title, 72)
        : item?.kind === "page_candidate" || item?.kind === "stale_page"
          ? item.title
          : item?.kind === "topic"
            ? item.label
            : item
              ? // Refinements: the resolved names ("A" and "B" look like the
                // same entity), falling back to the kind label while loading.
                summary.title
              : "";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t("review.title")}
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "center",
        padding: "6vh 16px 16px",
        backgroundColor: "rgba(0,0,0,0.45)",
        zIndex: 1100,
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        style={{
          position: "relative",
          width: "min(760px, 100%)",
          maxHeight: "86vh",
          overflowY: "auto",
          backgroundColor: "var(--mem-surface)",
          border: "1px solid var(--mem-border)",
          borderRadius: 16,
          boxShadow: "0 24px 48px rgba(0,0,0,0.35)",
          outline: "none",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {flash && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
              backgroundColor: "var(--mem-surface)",
              borderRadius: 16,
              zIndex: 3,
              fontFamily: "var(--mem-font-heading)",
              fontSize: 19,
              color: "var(--mem-status-success-text)",
            }}
          >
            {flash} ✓
          </div>
        )}

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "16px 20px 14px",
            borderBottom: "1px solid var(--mem-detail-divider)",
          }}
        >
          {item && (
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: 11,
                letterSpacing: "0.04em",
                borderRadius: 5,
                padding: "2px 8px",
                color: reviewKindTone(item).color,
                backgroundColor: reviewKindTone(item).background,
              }}
            >
              {reviewKindLabel(t, item)}
            </span>
          )}
          {item && isExampleReviewItem(item) && (
            <span style={examplePillStyle}>{t("review.exampleBadge")}</span>
          )}
          <span
            style={{
              marginLeft: "auto",
              fontFamily: "var(--mem-font-mono)",
              fontVariantNumeric: "tabular-nums",
              fontSize: 12,
              color: "var(--mem-text-tertiary)",
            }}
          >
            {item
              ? t("review.progress", {
                  position: index + 1,
                  total: items.length,
                })
              : ""}
          </span>
          <button
            type="button"
            aria-label={t("review.close")}
            onClick={close}
            style={{
              background: "none",
              border: "none",
              color: "var(--mem-text-tertiary)",
              cursor: "pointer",
              fontSize: 16,
              lineHeight: 1,
              padding: 6,
              borderRadius: 6,
            }}
          >
            ✕
          </button>
        </div>

        {done ? (
          <div style={{ textAlign: "center", padding: "40px 24px 44px" }}>
            <div
              style={{
                width: 52,
                height: 52,
                borderRadius: "50%",
                margin: "0 auto 16px",
                backgroundColor: "var(--mem-status-success-bg)",
                color: "var(--mem-status-success-text)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 24,
              }}
            >
              ✓
            </div>
            <h3
              style={{
                fontFamily: "var(--mem-font-heading)",
                fontWeight: 500,
                fontSize: 20,
                margin: "0 0 6px",
                color: "var(--mem-text)",
              }}
            >
              {t("review.allCaughtUp")}
            </h3>
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                color: "var(--mem-text-secondary)",
                fontSize: 13.5,
                margin: "0 0 20px",
              }}
            >
              {t("review.emptyQueueHint")}
            </p>
            <button
              type="button"
              onClick={close}
              style={{
                ...actionButtonStyle,
                backgroundColor: "var(--mem-accent-indigo)",
                borderColor: "var(--mem-accent-indigo)",
                color: "var(--mem-bg)",
                fontWeight: 600,
              }}
            >
              {t("review.backToReview")}
            </button>
          </div>
        ) : item ? (
          <>
            <div style={{ padding: "20px 22px 8px" }}>
              <h3
                style={{
                  fontFamily: "var(--mem-font-heading)",
                  fontWeight: 500,
                  fontSize: 19,
                  margin: "0 0 3px",
                  color: "var(--mem-text)",
                }}
              >
                {heading}
              </h3>
              <p
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-tertiary)",
                  fontSize: 12.5,
                  margin: "0 0 18px",
                }}
              >
                {item.kind === "revision"
                  ? item.agent
                    ? t("review.proposedBy", { agent: item.agent })
                    : ""
                  : item.kind === "capture"
                    ? t("review.captureHint")
                    : item.kind === "page_candidate"
                      ? (item.cluster.new_memory_count != null
                          ? t("review.newSources", {
                              count: item.cluster.new_memory_count,
                            })
                          : t("review.sources", {
                              count: item.cluster.source_ids.length,
                            })) +
                        (item.cluster.existing_page_id
                          ? ` · ${t("review.linkedExistingPage")}`
                          : "")
                      : item.kind === "topic"
                        ? t("review.topicReason", { count: item.count })
                        : item.kind === "stale_page"
                          ? (item.sourcesUpdated != null
                              ? t("review.sourcesUpdated", {
                                  count: item.sourcesUpdated,
                                })
                              : "")
                        : isContradiction
                          ? t("review.contradictionHint")
                      : item.action === "relation_conflict"
                        ? t("review.relationConflictHint")
                        : item.action === "page_keep_or_archive"
                          ? t("review.pageArchiveHint")
                          : item.action === "cross_space_discovery"
                            ? t("review.crossSpaceHint")
                            : item.action === "suggest_entity"
                              ? t("review.suggestEntityHint")
                              : item.action === "dedup_merge"
                                ? t("review.dedupHint")
                                : item.payload?.action === "page_merge"
                                  ? `${t("review.mergeReason", {
                                      count: item.payload.source_overlap,
                                      percent: Math.round(
                                        item.payload.source_overlap_ratio * 100,
                                      ),
                                    })} · ${t("review.confidence", {
                                      percent: Math.round(item.confidence * 100),
                                    })}`
                                  : t("review.confidence", {
                                      percent: Math.round(item.confidence * 100),
                                    })}
              </p>

              {item.kind === "revision" && (
                <>
                  <div
                    role="tablist"
                    style={{
                      display: "inline-flex",
                      border: "1px solid var(--mem-border)",
                      borderRadius: 8,
                      overflow: "hidden",
                      marginBottom: 14,
                    }}
                  >
                    {[
                      { side: false, label: t("review.unified") },
                      { side: true, label: t("review.sideBySide") },
                    ].map(({ side, label }) => (
                      <button
                        key={label}
                        type="button"
                        role="tab"
                        aria-selected={sideBySide === side}
                        onClick={() => setSideBySide(side)}
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          fontSize: 12.5,
                          padding: "5px 12px",
                          border: "none",
                          cursor: "pointer",
                          backgroundColor:
                            sideBySide === side
                              ? "var(--mem-indigo-bg)"
                              : "var(--mem-surface)",
                          color:
                            sideBySide === side
                              ? "var(--mem-accent-indigo)"
                              : "var(--mem-text-secondary)",
                        }}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {target.isLoading ? (
                    <div style={paneStyle}>{t("review.loadingCurrent")}</div>
                  ) : sideBySide ? (
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fit, minmax(240px, 1fr))",
                        gap: 12,
                      }}
                    >
                      <div>
                        <p style={paneLabelStyle}>{t("review.current")}</p>
                        <div style={paneStyle}>
                          <DiffText
                            segments={segments.filter(
                              (segment) => segment.kind !== "ins",
                            )}
                          />
                        </div>
                      </div>
                      <div>
                        <p style={paneLabelStyle}>{t("review.proposed")}</p>
                        <div style={paneStyle}>
                          <DiffText
                            segments={segments.filter(
                              (segment) => segment.kind !== "del",
                            )}
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={paneStyle}>
                      <DiffText segments={segments} />
                    </div>
                  )}

                  {!target.isLoading && (
                    <p
                      style={{
                        fontFamily: "var(--mem-font-body)",
                        color: "var(--mem-text-tertiary)",
                        fontSize: 12,
                        margin: "10px 2px 0",
                      }}
                    >
                      {t("review.wordDelta", {
                        added: wordCounts.added,
                        removed: wordCounts.removed,
                      })}
                      {" · "}
                      <del style={DEL_STYLE}>{t("review.stripped")}</del>
                      {" · "}
                      <ins style={INS_STYLE}>{t("review.added")}</ins>
                    </p>
                  )}

                  <MemoryRevisionChain
                    sourceId={item.targetSourceId}
                    onOpenMemory={(id) => onOpenMemory?.(id)}
                  />
                </>
              )}

              {item.kind === "capture" && (
                <div style={paneStyle}>
                  {target.isLoading
                    ? t("review.loadingCurrent")
                    : (target.data?.content ?? item.snippet ?? "")}
                </div>
              )}

              {item.kind === "page_candidate" && (
                <div style={{ display: "grid", gap: 12 }}>
                  <p
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      color: "var(--mem-text-secondary)",
                      fontSize: 13,
                      margin: 0,
                    }}
                  >
                    {t("review.candidateQueued")}
                  </p>
                  {item.cluster.contents
                    .map((content) => content.trim())
                    .filter((content) => content.length > 0)
                    .slice(0, 3)
                    .map((content, contentIndex) => (
                      <div key={contentIndex} style={paneStyle}>
                        {content}
                      </div>
                    ))}
                </div>
              )}

              {item.kind === "topic" && (
                <div style={{ display: "grid", gap: 12 }}>
                  <p
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      color: "var(--mem-text-secondary)",
                      fontSize: 13,
                      margin: 0,
                    }}
                  >
                    {t("review.topicHint")}
                  </p>
                  <ReviewEvidencePane
                    query={item.label}
                    onOpenMemory={onOpenMemory}
                  />
                </div>
              )}

              {item.kind === "stale_page" && (
                <div style={{ display: "grid", gap: 12 }}>
                  <p
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      color: "var(--mem-text-secondary)",
                      fontSize: 13,
                      margin: 0,
                    }}
                  >
                    {t("review.refreshHint")}
                  </p>
                  {item.summary && <div style={paneStyle}>{item.summary}</div>}
                </div>
              )}

              {item.kind === "refinement" &&
                item.action === "entity_merge" && (
                  <div style={{ display: "grid", gap: 12 }}>
                    <div>
                      <p style={paneLabelStyle}>{t("review.mergeKeep")}</p>
                      <div style={paneStyle}>
                        {mergeExisting.data?.entity.name ??
                          mergePayload?.existing_id}
                      </div>
                    </div>
                    <div>
                      <p style={paneLabelStyle}>{t("review.mergeFoldsIn")}</p>
                      <div style={paneStyle}>
                        {mergeIncoming.data?.entity.name ??
                          mergePayload?.new_id}
                      </div>
                    </div>
                  </div>
                )}

              {item.kind === "refinement" &&
                item.action === "detect_contradiction" && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(240px, 1fr))",
                      gap: 12,
                    }}
                  >
                    <div>
                      <p style={paneLabelStyle}>{t("review.existingMemory")}</p>
                      <div style={paneStyle}>
                        {memoryPaneB.isLoading ? (
                          t("review.loadingCurrent")
                        ) : contradictionSegments.length > 0 ? (
                          <DiffText
                            segments={contradictionSegments.filter(
                              (segment) => segment.kind !== "ins",
                            )}
                          />
                        ) : (
                          (memoryPaneB.data?.content ?? "")
                        )}
                      </div>
                    </div>
                    <div>
                      <p style={paneLabelStyle}>{t("review.newMemoryNewer")}</p>
                      <div style={paneStyle}>
                        {memoryPaneA.isLoading ? (
                          t("review.loadingCurrent")
                        ) : contradictionSegments.length > 0 ? (
                          <DiffText
                            segments={contradictionSegments.filter(
                              (segment) => segment.kind !== "del",
                            )}
                          />
                        ) : (
                          (memoryPaneA.data?.content ?? "")
                        )}
                      </div>
                    </div>
                  </div>
                )}

              {item.kind === "refinement" && item.action === "page_merge" && (
                <PageMergeStripOff
                  keepId={item.sourceIds[0]}
                  retireId={item.sourceIds[1]}
                  onOpenPage={(id) => onOpenPage?.(id)}
                  onOpenMemory={(id) => onOpenMemory?.(id)}
                  sourceOverlap={
                    item.payload?.action === "page_merge"
                      ? item.payload.source_overlap
                      : 0
                  }
                  sourceOverlapRatio={
                    item.payload?.action === "page_merge"
                      ? item.payload.source_overlap_ratio
                      : 0
                  }
                />
              )}

              {item.kind === "refinement" &&
                item.action === "relation_conflict" &&
                item.payload?.action === "relation_conflict" && (
                  <div style={{ display: "grid", gap: 12 }}>
                    <div>
                      <p style={paneLabelStyle}>{t("review.relationOld")}</p>
                      <div style={{ ...paneStyle, fontFamily: "var(--mem-font-mono)", fontSize: 13 }}>
                        <del style={DEL_STYLE}>
                          {item.payload.from} —{item.payload.old_type}→ {item.payload.to}
                        </del>
                      </div>
                    </div>
                    <div>
                      <p style={paneLabelStyle}>{t("review.relationNew")}</p>
                      <div style={{ ...paneStyle, fontFamily: "var(--mem-font-mono)", fontSize: 13 }}>
                        <ins style={INS_STYLE}>
                          {item.payload.from} —{item.payload.new_type}→ {item.payload.to}
                        </ins>
                      </div>
                    </div>
                  </div>
                )}

              {item.kind === "refinement" &&
                item.action === "page_keep_or_archive" && (
                  <div style={{ display: "grid", gap: 12 }}>
                    <div>
                      <p style={paneLabelStyle}>
                        {archivePage.data?.title ??
                          archivePageId ??
                          t("review.kindPageArchive")}
                      </p>
                      <div style={paneStyle}>
                        {archivePage.isLoading
                          ? t("review.loadingCurrent")
                          : (archivePage.data?.summary ??
                            archivePage.data?.content ??
                            "")}
                      </div>
                    </div>
                    {item.payload?.action === "page_keep_or_archive" && (
                      <p
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          color: "var(--mem-text-tertiary)",
                          fontSize: 12,
                          margin: 0,
                        }}
                      >
                        {t("review.sources", {
                          count: item.payload.source_count,
                        })}
                      </p>
                    )}
                  </div>
                )}

              {item.kind === "refinement" &&
                (item.action === "suggest_entity" ||
                  item.action === "dedup_merge" ||
                  item.action === "cross_space_discovery") && (
                  <div style={{ display: "grid", gap: 12 }}>
                    {item.payload?.action === "suggest_entity" &&
                      item.payload.name_hint && (
                        <div style={paneStyle}>{item.payload.name_hint}</div>
                      )}
                    {item.action === "suggest_entity" && (
                      <ReviewEvidencePane
                        query={
                          item.payload?.action === "suggest_entity"
                            ? (item.payload.name_hint ?? null)
                            : null
                        }
                        onOpenMemory={onOpenMemory}
                      />
                    )}
                    {item.payload?.action === "cross_space_discovery" && (
                      <p
                        style={{
                          fontFamily: "var(--mem-font-body)",
                          color: "var(--mem-text-secondary)",
                          fontSize: 13,
                          margin: 0,
                        }}
                      >
                        {t("review.crossSpaceSpaces", {
                          spaces: item.payload.spaces.join(" · "),
                          count: item.payload.memory_count,
                        })}
                      </p>
                    )}
                    {[memoryPaneA, memoryPaneB].map(
                      (pane, paneIndex) =>
                        memoryPaneIds[paneIndex] && (
                          <div key={paneIndex}>
                            <p style={paneLabelStyle}>
                              {pane.data?.title ?? ""}
                            </p>
                            <div style={paneStyle}>
                              {pane.isLoading
                                ? t("review.loadingCurrent")
                                : (pane.data?.content ?? "")}
                            </div>
                          </div>
                        ),
                    )}
                  </div>
                )}
            </div>

            {isExampleReviewItem(item) && (
              <p
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: 12,
                  color: "var(--mem-text-tertiary)",
                  margin: "0 20px 10px",
                }}
              >
                {t("review.exampleDialogNote")}
              </p>
            )}

            {resolveError && (
              <p
                role="alert"
                style={{
                  background: "var(--mem-status-danger-bg)",
                  border: "1px solid var(--mem-status-danger-border)",
                  borderRadius: 8,
                  color: "var(--mem-status-danger-text)",
                  font: "12px/1.5 var(--mem-font-body)",
                  margin: "4px 20px 0",
                  padding: "9px 11px",
                }}
              >
                {t("review.actionError")}
              </p>
            )}

            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "14px 20px 16px",
                borderTop: "1px solid var(--mem-detail-divider)",
                marginTop: 14,
              }}
            >
              {!reviewDismissBlocked(item) && (
                <button
                  type="button"
                  disabled={resolving}
                  onClick={() => void resolveCurrent(false)}
                  style={{
                    ...actionButtonStyle,
                    // Forget hard-deletes the capture; every other dismiss just
                    // clears the proposal, so only Forget wears the danger tone.
                    ...(item.kind === "capture"
                      ? {
                          color: "var(--mem-status-danger-text)",
                          borderColor: "var(--mem-status-danger-border)",
                        }
                      : { color: "var(--mem-text-secondary)" }),
                  }}
                >
                  {item.kind === "capture"
                    ? t("review.forget")
                    : isContradiction
                      ? t("review.keepBoth")
                      : item.kind === "refinement" &&
                          item.action === "page_keep_or_archive"
                        ? t("review.keepPage")
                        : item.kind === "topic" ||
                            item.kind === "page_candidate" ||
                            item.kind === "stale_page"
                          ? t("review.hide")
                          : t("review.dismiss")}
                </button>
              )}
              {item.kind === "page_candidate" &&
                item.cluster.existing_page_id &&
                onOpenPage && (
                  <button
                    type="button"
                    onClick={() =>
                      onOpenPage(item.cluster.existing_page_id as string)
                    }
                    style={actionButtonStyle}
                  >
                    {t("review.openPage")}
                  </button>
                )}
              {item.kind === "stale_page" && onOpenPage && (
                <button
                  type="button"
                  onClick={() => onOpenPage(item.id)}
                  style={actionButtonStyle}
                >
                  {t("review.openPage")}
                </button>
              )}
              {(item.kind === "revision" || item.kind === "capture") &&
                onOpenMemory && (
                  <button
                    type="button"
                    onClick={() =>
                      onOpenMemory(
                        item.kind === "revision" ? item.targetSourceId : item.id,
                      )
                    }
                    style={actionButtonStyle}
                  >
                    {t("review.openMemory")}
                  </button>
                )}
              <span style={{ flex: 1 }} />
              <button
                type="button"
                disabled={items.length < 2}
                onClick={() => goTo(1)}
                style={actionButtonStyle}
              >
                {t("review.skip")}
              </button>
              {!reviewApproveBlocked(item) && (
                <button
                  type="button"
                  disabled={resolving}
                  onClick={() => void resolveCurrent(true)}
                  style={{
                    ...actionButtonStyle,
                    backgroundColor: "var(--mem-accent-indigo)",
                    borderColor: "var(--mem-accent-indigo)",
                    color: "var(--mem-bg)",
                    fontWeight: 600,
                  }}
                >
                  {item.kind === "capture"
                    ? t("review.confirm")
                    : item.kind === "stale_page"
                      ? t("review.refreshPage")
                      : isContradiction
                        ? t("review.resolve")
                        : item.kind === "refinement" &&
                            item.action === "page_keep_or_archive"
                          ? t("review.archive")
                          : item.kind === "refinement" &&
                              item.action === "page_merge"
                            ? t("review.mergePages")
                            : t("review.approve")}
                </button>
              )}
            </div>

            {(item.kind === "topic" ||
              item.kind === "page_candidate" ||
              item.kind === "stale_page") && (
              <p
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-tertiary)",
                  fontSize: 12,
                  margin: "0 20px 14px",
                }}
              >
                {t("review.hideHint")}
              </p>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
