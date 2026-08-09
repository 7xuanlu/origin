// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import {
  getEntitySuggestions,
  getMemoryStats,
  listEntities,
  listRecentChanges,
  listRecentRetrievals,
  type MemoryStats,
  type Page,
} from "../../lib/tauri";
import { listAllActivePages } from "./pages/listAllPages";
import { Greeting } from "./Greeting";
import { useReviewQueue, reviewItemId, type ReviewItem } from "./useReviewQueue";
import ReviewDialog, {
  reviewKindLabel,
  useReviewItemSummary,
} from "./ReviewDialog";
import { RefiningList } from "./RefiningList";
import { ConnectionsList } from "./ConnectionsList";
import { RetrievalsList } from "./RetrievalsList";
import { WhatHappensNextCard, type HomePageState } from "../onboarding/WhatHappensNextCard";
import { GhostPagesRow } from "../onboarding/GhostPagesRow";
import { useMilestones } from "../onboarding/useMilestones";
import { FirstPageModal } from "../onboarding/FirstPageModal";
import { MilestoneHighlight } from "../onboarding/MilestoneHighlight";

interface HomePageProps {
  onNavigateMemory: (sourceId: string) => void;
  onNavigateStream: () => void;
  onNavigateLog: () => void;
  onNavigateGraph: () => void;
  onSelectPage?: (pageId: string) => void;
  onOpenDistillReview?: () => void;
}

const FIRST_CONCEPT_SHOWN_KEY = "onboarding:firstConceptShownCount";
const MAX_MODAL_SHOWS = 3;

function deriveHomePageState(params: {
  intelligenceReady: boolean;
  memoryCount: number;
  conceptCount: number;
}): HomePageState {
  // Pages take precedence over memories: if any page exists the home page is
  // "alive", even if memories were subsequently deleted. The local counter
  // still uses the legacy concept name because the daemon activity contract
  // has not been renamed yet.
  if (params.conceptCount > 0) return "alive";
  if (params.memoryCount > 0) return "gathering";
  if (params.intelligenceReady) return "listening";
  return "seed";
}

export default function HomePage({
  onNavigateMemory,
  onNavigateStream: _onNavigateStream,
  onNavigateLog: _onNavigateLog,
  onNavigateGraph: _onNavigateGraph,
  onSelectPage,
  onOpenDistillReview,
}: HomePageProps) {
  const { t } = useTranslation();
  const { data: retrievals = [] } = useQuery({
    queryKey: ["recentRetrievals"],
    queryFn: () => listRecentRetrievals(12),
    refetchInterval: 30_000,
  });

  const { data: changes = [] } = useQuery({
    queryKey: ["recentChanges"],
    queryFn: () => listRecentChanges(3),
    refetchInterval: 30_000,
  });

  const { data: recentConcepts = [], isLoading: recentConceptsLoading } = useQuery({
    queryKey: ["recent-concepts"],
    queryFn: listAllActivePages,
    refetchInterval: 10_000,
  });

  const { data: stats } = useQuery({
    queryKey: ["memoryStats"],
    queryFn: getMemoryStats,
    refetchInterval: 10_000,
  });

  const memoryCount = stats?.total ?? 0;
  const conceptCount = recentConcepts.length;
  const hasPages = recentConcepts.length > 0;
  const recentlyRefinedPages = useMemo(
    () =>
      [...recentConcepts]
        .sort((a, b) => Date.parse(b.last_modified) - Date.parse(a.last_modified))
        .slice(0, 6),
    [recentConcepts],
  );

  const { milestones, acknowledge } = useMilestones();
  const intelligenceReady = milestones.some(
    (m) => m.id === "intelligence-ready" && m.first_triggered_at != null,
  );
  const firstTriggeredAt = milestones.find((m) => m.id === "intelligence-ready")
    ?.first_triggered_at;
  const daysInListening = firstTriggeredAt
    ? Math.floor((Date.now() / 1000 - firstTriggeredAt) / 86_400)
    : 0;

  const firstConceptMs = milestones.find(
    (m) => m.id === "first-concept" && m.acknowledged_at == null,
  );
  const firstConceptData = firstConceptMs?.payload as
    | { page_id?: string; title?: string }
    | undefined;
  const firstConcept = firstConceptData?.page_id
    ? recentConcepts.find((c) => c.id === firstConceptData.page_id)
    : null;

  const homePageState = deriveHomePageState({
    intelligenceReady,
    memoryCount,
    conceptCount,
  });

  // Track whether we've incremented shown_count for the current first-concept
  // modal instance, so StrictMode double-invokes don't double-count.
  const incrementedOnMountRef = useRef(false);
  useEffect(() => {
    if (!firstConcept || !firstConceptMs) return;
    if (incrementedOnMountRef.current) return;
    incrementedOnMountRef.current = true;
    const current = parseInt(
      localStorage.getItem(FIRST_CONCEPT_SHOWN_KEY) || "0",
      10,
    );
    const next = current + 1;
    localStorage.setItem(FIRST_CONCEPT_SHOWN_KEY, String(next));
    if (next > MAX_MODAL_SHOWS) {
      acknowledge("first-concept");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [firstConcept?.id, firstConceptMs?.id]);

  const firstConceptShownCount = parseInt(
    localStorage.getItem(FIRST_CONCEPT_SHOWN_KEY) || "0",
    10,
  );
  const shouldShowFirstConceptModal =
    !!firstConcept &&
    !!firstConceptMs &&
    firstConceptShownCount <= MAX_MODAL_SHOWS;

  const isEmpty =
    retrievals.length === 0 &&
    changes.length === 0;

  const distillReviewEntry = onOpenDistillReview ? (
    <div style={{ display: "flex", justifyContent: "flex-end", margin: "-4px 0 10px" }}>
      <button
        type="button"
        onClick={onOpenDistillReview}
        style={{
          border: "1px solid var(--mem-border)",
          background: "var(--mem-surface)",
          color: "var(--mem-text)",
          borderRadius: 8,
          padding: "7px 11px",
          fontSize: 13,
          cursor: "pointer",
        }}
      >
        {t("home.reviewPageChanges")}
      </button>
    </div>
  ) : null;

  // The "recent-concepts" query has never resolved right after onboarding
  // (HomePage isn't mounted during the wizard), so its default `[]` would
  // otherwise make hasPages look like "no pages" and flash the never-
  // onboarded seed state before the real page list arrives.
  if (recentConceptsLoading) {
    return null;
  }

  if (hasPages) {
    return (
      <>
        <WikiHome
          allPages={recentConcepts}
          pages={recentlyRefinedPages}
          stats={stats}
          onSelectPage={onSelectPage}
          onOpenDistillReview={onOpenDistillReview}
          onOpenMemory={onNavigateMemory}
        />

        {shouldShowFirstConceptModal && firstConcept && (
          <FirstPageModal
            page={firstConcept}
            onOpen={(id) => {
              localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
              acknowledge("first-concept");
              onSelectPage?.(id);
            }}
            onDismiss={() => {
              localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
              acknowledge("first-concept");
            }}
          />
        )}
      </>
    );
  }

  return (
    <div className="flex flex-col gap-8 pb-16">
      <Greeting memoryCount={memoryCount} pageCount={conceptCount} />

      {isEmpty ? (
        <>
          <WhatHappensNextCard
            state={homePageState}
            memoryCount={memoryCount}
            daysInListening={daysInListening}
          />
          {distillReviewEntry}
          <MilestoneHighlight
            active={
              !shouldShowFirstConceptModal &&
              milestones.some(
                (m) => m.id === "first-concept" && m.acknowledged_at == null,
              )
            }
            onSeen={() => {}}
            intensity="full"
          >
            <section>
              <GhostPagesRow />
            </section>
          </MilestoneHighlight>
        </>
      ) : (
        <>
          {distillReviewEntry}
          <RefiningList changes={changes} pages={recentConcepts} onSelectPage={onSelectPage} />
          <RetrievalsList
            events={retrievals}
            onSelectPageById={(pageId) => onSelectPage?.(pageId)}
          />
          <ConnectionsList
            onSelectPage={onSelectPage}
            onSelectEntity={undefined}
          />
        </>
      )}

      {shouldShowFirstConceptModal && firstConcept && (
        <FirstPageModal
          page={firstConcept}
          onOpen={(id) => {
            localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
            acknowledge("first-concept");
            onSelectPage?.(id);
          }}
          onDismiss={() => {
            localStorage.removeItem(FIRST_CONCEPT_SHOWN_KEY);
            acknowledge("first-concept");
          }}
        />
      )}
    </div>
  );
}

function formatPagePath(page: Page): string {
  const domain = page.domain?.trim() || page.space?.trim();
  const title = page.title.replace(/\s+/g, "-");
  return `[[${domain ? `${domain}/` : ""}${title}]]`;
}

function formatSourceCount(t: TFunction, count: number): string {
  return t("home.counts.source", { count });
}

function updatedTodayCount(pages: Page[]): number {
  return pages.filter((page) => {
    const ms = Date.parse(page.last_modified);
    return !Number.isNaN(ms) && Math.floor((Date.now() - ms) / 86_400_000) <= 0;
  }).length;
}

function latestPageUpdate(t: TFunction, pages: Page[]): string {
  const latest = pages.reduce<string | null>((current, page) => {
    if (!current) return page.last_modified;
    return Date.parse(page.last_modified) > Date.parse(current) ? page.last_modified : current;
  }, null);
  return latest ? relativePageDate(t, latest) : t("home.relative.noUpdates");
}

function relativePageDate(t: TFunction, value: string): string {
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return t("home.relative.updatedRecently");
  const delta = Date.now() - ms;
  const days = Math.floor(delta / 86_400_000);
  if (days <= 0) return t("home.relative.today");
  if (days === 1) return t("home.relative.yesterday");
  if (days < 7) return t("home.relative.daysAgo", { count: days });
  const weeks = Math.floor(days / 7);
  return weeks === 1 ? t("home.relative.weekAgo") : t("home.relative.weeksAgo", { count: weeks });
}

function useElementMinWidth<T extends HTMLElement>(minWidth: number) {
  const ref = useRef<T | null>(null);
  const [matches, setMatches] = useState(false);

  function getMatches() {
    if (typeof window === "undefined") return false;
    const width = ref.current?.getBoundingClientRect().width ?? window.innerWidth;
    return width >= minWidth;
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    const element = ref.current;
    const update = () => setMatches(getMatches());
    update();

    if (element && "ResizeObserver" in window) {
      const observer = new ResizeObserver(update);
      observer.observe(element);
      return () => observer.disconnect();
    }

    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, [minWidth]);

  return [ref, matches] as const;
}

function WikiHome({
  allPages,
  pages,
  stats,
  onSelectPage,
  onOpenDistillReview,
  onOpenMemory,
}: {
  allPages: Page[];
  pages: Page[];
  stats?: MemoryStats;
  onSelectPage?: (pageId: string) => void;
  onOpenDistillReview?: () => void;
  onOpenMemory?: (sourceId: string) => void;
}) {
  const [containerRef, isWideLayout] = useElementMinWidth<HTMLDivElement>(820);
  const {
    items: reviewItems,
    isLoading: reviewLoading,
    error: reviewError,
    decisionsTruncated,
    resolve,
    isResolving,
    refetch: refetchReviewQueue,
  } = useReviewQueue();
  const [openReviewId, setOpenReviewId] = useState<string | null>(null);
  // New-memory captures are inflow, not decisions — they're unconfirmed but
  // already live (recalled, feeding pages), so they stay out of the rail.
  const decisionItems = reviewItems.filter((item) => item.kind !== "capture");
  return (
    <div
      data-testid="wiki-home"
      ref={containerRef}
      className="wiki-home"
      style={{
        display: "grid",
        gap: isWideLayout ? 24 : 22,
        gridTemplateColumns: "minmax(0, 1fr)",
        maxWidth: 1280,
        margin: "0 auto",
        width: "100%",
        paddingBottom: 64,
        alignItems: "start",
      }}
    >
      <section
        data-testid="wiki-daily-desk"
        className="wiki-daily-desk"
      >
        <TodayHeader pages={allPages} />

        <HomeContextRail
          pages={allPages}
          stats={stats}
        />
      </section>

      <div
        data-testid="wiki-content-grid"
        className="wiki-content-grid"
        style={{
          display: "grid",
          gap: isWideLayout ? 28 : 24,
          gridTemplateColumns: isWideLayout ? "minmax(0, 1fr) minmax(320px, 360px)" : "minmax(0, 1fr)",
          gridColumn: "1 / -1",
          alignItems: "start",
          minWidth: 0,
        }}
      >
        <PageList
          pages={pages}
          onSelectPage={onSelectPage}
          isWideLayout={isWideLayout}
        />
        <NeedsReviewRail
          items={decisionItems}
          isLoading={reviewLoading}
          error={reviewError}
          onRetry={refetchReviewQueue}
          isTruncated={decisionsTruncated}
          onOpenItem={setOpenReviewId}
          onOpenDistillReview={onOpenDistillReview}
          leadsColumn={!isWideLayout && decisionItems.length > 0}
        />
      </div>

      <ReviewDialog
        items={reviewItems}
        openId={openReviewId}
        onOpenChange={setOpenReviewId}
        onResolve={resolve}
        isResolving={isResolving}
        onOpenMemory={onOpenMemory}
      />
    </div>
  );
}

function SectionHeading({
  title,
  action,
  size = "default",
  level = 2,
}: {
  title: string;
  action?: React.ReactNode;
  size?: "default" | "compact" | "page";
  level?: 1 | 2;
}) {
  const Heading = level === 1 ? "h1" : "h2";
  const pageHeading = size === "page";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        gap: 12,
        marginBottom: pageHeading ? 16 : 12,
      }}
    >
      <Heading
        style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: size === "compact" ? 14 : pageHeading ? "var(--mem-destination-title-size)" : 18,
          fontWeight: 500,
          color: "var(--mem-text)",
          letterSpacing: pageHeading ? "-0.03em" : 0,
          lineHeight: pageHeading ? 1.12 : 1.2,
          margin: 0,
        }}
      >
        {title}
      </Heading>
      {action}
    </div>
  );
}

function TodayHeader({ pages }: { pages: Page[] }) {
  const { t } = useTranslation();
  return (
    <section data-testid="wiki-today-heading" className="wiki-today-heading">
      <SectionHeading
        title={t("home.todayInWenlan")}
        level={1}
        size="page"
        action={
          <span
            data-testid="wiki-context-latest"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: 11,
              color: "var(--mem-text-tertiary)",
              whiteSpace: "nowrap",
            }}
          >
            {latestPageUpdate(t, pages)}
          </span>
        }
      />
    </section>
  );
}

function PageList({
  pages,
  onSelectPage,
  isWideLayout,
}: {
  pages: Page[];
  onSelectPage?: (pageId: string) => void;
  isWideLayout: boolean;
}) {
  const { t } = useTranslation();
  if (!pages.length) return null;
  return (
    <div>
      <div
        data-testid="wiki-page-list"
        style={{
          display: "grid",
          gap: 0,
          borderTopStyle: "none",
          borderTopWidth: 0,
          borderTopColor: "transparent",
        }}
      >
        {pages.map((page) => (
          <button
            key={page.id}
            type="button"
            aria-label={t("home.openPage", { title: page.title })}
            className="transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{
              display: "grid",
              width: "100%",
              gap: isWideLayout ? 20 : 12,
              gridTemplateColumns: isWideLayout
                ? "minmax(240px, 1fr) minmax(128px, auto)"
                : "minmax(0, 1fr)",
              padding: isWideLayout ? "14px 4px" : "15px 4px",
              textAlign: "left",
              border: "none",
              borderBottom: "1px solid color-mix(in srgb, var(--mem-border) 70%, transparent)",
              background: "transparent",
              color: "inherit",
              cursor: onSelectPage ? "pointer" : "default",
            }}
            onClick={() => onSelectPage?.(page.id)}
          >
            <div style={{ display: "flex", minWidth: 0, gap: 12 }}>
              <PageIcon />
              <div className="min-w-0">
                <p
                  style={{
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                    fontFamily: "var(--mem-font-heading)",
                    fontSize: 16,
                    fontWeight: 500,
                    color: "var(--mem-text)",
                    lineHeight: 1.18,
                    margin: 0,
                  }}
                >
                  {page.title}
                </p>
                <p
                  className="truncate"
                  style={{
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: 11,
                    color: "var(--mem-text-tertiary)",
                    margin: "6px 0 0",
                  }}
                >
                  {formatPagePath(page)}
                </p>
                {page.summary && (
                  <p
                    style={{
                      display: isWideLayout ? "none" : "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                      fontFamily: "var(--mem-font-body)",
                      fontSize: 12,
                      color: "var(--mem-text-secondary)",
                      lineHeight: 1.45,
                      margin: "8px 0 0",
                    }}
                  >
                    {page.summary}
                  </p>
                )}
              </div>
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: isWideLayout ? "column" : "row",
                flexWrap: "wrap",
                alignItems: isWideLayout ? "flex-end" : "center",
                justifyContent: isWideLayout ? "center" : "flex-start",
                gap: isWideLayout ? 4 : 12,
                fontFamily: "var(--mem-font-body)",
                fontSize: 12,
                color: "var(--mem-text-tertiary)",
                textAlign: isWideLayout ? "right" : "left",
              }}
            >
              <span>{formatSourceCount(t, page.source_memory_ids.length)}</span>
              <span>{relativePageDate(t, page.last_modified)}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function HomeContextRail({
  pages,
  stats,
}: {
  pages: Page[];
  /** Aggregate memory counts; undefined while the stats query is loading. */
  stats?: MemoryStats;
}) {
  const { t } = useTranslation();
  // Same cache key/shape as AtlasView's entity fetch — shared cache,
  // no double-fetch.
  const entitiesQuery = useQuery({
    queryKey: ["constellation-entities"],
    queryFn: () => listEntities(),
    staleTime: 60_000,
  });
  // Same cache key/shape as EntitySuggestions' fetch — shared cache.
  const suggestionsQuery = useQuery({
    queryKey: ["entity-suggestions"],
    queryFn: getEntitySuggestions,
    staleTime: 60_000,
  });
  const pagesUpdatedToday = updatedTodayCount(pages);

  return (
    <div
      data-testid="wiki-context-rail"
      className="wiki-context-rail"
    >
      <section
        data-testid="wiki-index-summary"
        className="wiki-index-bar"
        aria-label={t("home.overview")}
      >
        <div
          data-testid="wiki-index-strip"
          className="wiki-index-strip"
        >
          <ContextMetric
            testId="pages"
            label={t("home.pages")}
            total={String(pages.length)}
            deltas={
              pagesUpdatedToday > 0
                ? [
                    {
                      text: t("home.deltaUpdated", { value: String(pagesUpdatedToday) }),
                      tone: "indigo",
                      testId: "wiki-context-updated-today",
                    },
                  ]
                : []
            }
          />
          <ContextMetric
            testId="memories"
            label={t("home.memories")}
            total={stats ? String(stats.total) : "—"}
            deltas={
              stats && stats.new_today > 0
                ? [
                    {
                      text: t("home.deltaToday", { value: String(stats.new_today) }),
                      tone: "sage" as const,
                      testId: "wiki-context-memories-delta",
                    },
                  ]
                : []
            }
          />
          <ContextMetric
            testId="entities"
            label={t("home.entities")}
            total={entitiesQuery.data ? String(entitiesQuery.data.length) : "—"}
            deltas={
              (suggestionsQuery.data?.length ?? 0) > 0
                ? [
                    {
                      text: t("home.deltaSuggested", { value: String(suggestionsQuery.data!.length) }),
                      tone: "sage",
                      testId: "wiki-context-entities-delta",
                    },
                  ]
                : []
            }
          />
        </div>
      </section>
    </div>
  );
}

interface ContextDelta {
  /** Rendered text, already localized, e.g. "+18 today". */
  text: string;
  tone: "sage" | "indigo" | "warm-pill";
  testId?: string;
}

function ContextMetric({
  testId,
  label,
  total,
  deltas,
}: {
  testId: string;
  label: string;
  /** Pre-formatted total; "—" while loading. */
  total: string;
  deltas: ContextDelta[];
}) {
  return (
    <div data-testid={`wiki-context-${testId}-cell`}>
      <p
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: 11,
          fontWeight: 600,
          color: "var(--mem-text-tertiary)",
          letterSpacing: "0.04em",
          lineHeight: 1.25,
          margin: "0 0 5px",
          textTransform: "uppercase",
        }}
      >
        {label}
      </p>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
        <span
          data-testid={`wiki-context-${testId}`}
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: 26,
            fontWeight: 500,
            lineHeight: 1.1,
            color: "var(--mem-text)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {total}
        </span>
        {deltas.map((delta, index) => (
          <span
            key={delta.testId ?? index}
            data-testid={delta.testId}
            style={
              delta.tone === "warm-pill"
                ? {
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: 11,
                    fontVariantNumeric: "tabular-nums",
                    borderRadius: 5,
                    padding: "2px 8px",
                    color: "var(--mem-accent-warm)",
                    backgroundColor: "color-mix(in srgb, var(--mem-accent-warm) 15%, transparent)",
                    whiteSpace: "nowrap",
                  }
                : {
                    fontFamily: "var(--mem-font-mono)",
                    fontSize: 11.5,
                    fontVariantNumeric: "tabular-nums",
                    color: delta.tone === "sage" ? "var(--mem-accent-sage)" : "var(--mem-accent-indigo)",
                    whiteSpace: "nowrap",
                  }
            }
          >
            {delta.text}
          </span>
        ))}
      </div>
    </div>
  );
}

function NeedsReviewRail({
  items,
  isLoading,
  error,
  onRetry,
  isTruncated = false,
  onOpenItem,
  onOpenDistillReview,
  leadsColumn = false,
}: {
  items: ReviewItem[];
  isLoading: boolean;
  /** Set when the queue fetch failed — an empty `items` then means "unknown", not "caught up". */
  error?: unknown;
  onRetry?: () => void;
  /** A source hit its fetch cap — show the count as "N+", never as exact. */
  isTruncated?: boolean;
  onOpenItem: (id: string) => void;
  onOpenDistillReview?: () => void;
  /** Single-column layout: surface the rail above the page list when it has items. */
  leadsColumn?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <section
      data-testid="wiki-page-updates"
      style={leadsColumn ? { order: -1 } : undefined}
    >
      <h2
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontFamily: "var(--mem-font-body)",
          fontSize: 11,
          fontWeight: 500,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--mem-text-tertiary)",
          margin: "0 0 8px",
        }}
      >
        {t("home.pageUpdates")}
        {items.length > 0 && (
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontVariantNumeric: "tabular-nums",
              fontSize: 11,
              fontWeight: 400,
              letterSpacing: 0,
              color: "var(--mem-accent-indigo)",
              backgroundColor: "var(--mem-indigo-bg)",
              borderRadius: 999,
              padding: "1px 8px",
            }}
          >
            {isTruncated ? `${items.length}+` : items.length}
          </span>
        )}
      </h2>
      <div
        style={{
          border: "1px solid var(--mem-border)",
          borderRadius: 10,
          padding: "11px 13px 8px",
          // Raised via border + ink-toned shadow, not a fill tint. Light stays
          // canvas-white (no smudge on the white home); dark lifts by fill
          // (--mem-home-card) where the shadow barely reads. Indigo identity
          // lives in the heading + count pill, not the fill.
          backgroundColor: "var(--mem-home-card)",
          boxShadow:
            "0 1px 2px rgba(26, 26, 46, 0.05), 0 4px 12px rgba(26, 26, 46, 0.06)",
        }}
      >
        <div data-testid="worth-a-glance" style={{ display: "grid" }}>
          {items.length === 0 ? (
            error ? (
              <p
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  fontFamily: "var(--mem-font-body)",
                  fontSize: 13,
                  color: "var(--mem-text-secondary)",
                  lineHeight: 1.5,
                  margin: "4px 0 6px",
                }}
              >
                {t("review.loadFailed")}
                {onRetry && (
                  <button
                    type="button"
                    onClick={onRetry}
                    style={{
                      border: "none",
                      background: "none",
                      padding: 0,
                      color: "var(--mem-accent-indigo)",
                      fontFamily: "inherit",
                      fontSize: "inherit",
                      cursor: "pointer",
                    }}
                  >
                    {t("review.retry")}
                  </button>
                )}
              </p>
            ) : (
              !isLoading && (
                <p
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 7,
                    fontFamily: "var(--mem-font-body)",
                    fontSize: 12.5,
                    color: "var(--mem-accent-sage)",
                    lineHeight: 1.5,
                    margin: "4px 0 6px",
                  }}
                >
                  ✓ {t("review.allCaughtUp")}
                </p>
              )
            )
          ) : (
            <>
              {items.slice(0, 3).map((item, itemIndex, shown) => (
                <ReviewRailItem
                  key={reviewItemId(item)}
                  item={item}
                  onOpenItem={onOpenItem}
                  isLast={itemIndex === shown.length - 1}
                />
              ))}
              {error && (
                <p
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: 12,
                    color: "var(--mem-text-tertiary)",
                    lineHeight: 1.4,
                    margin: "6px 0 0",
                  }}
                >
                  {t("review.loadPartial")}
                </p>
              )}
            </>
          )}
        </div>

        {onOpenDistillReview && (
          <button
            type="button"
            onClick={onOpenDistillReview}
            style={{
              display: "block",
              width: "100%",
              marginTop: 2,
              padding: "7px 2px",
              textAlign: "left",
              border: "0",
              backgroundColor: "transparent",
              color: "var(--mem-accent-indigo)",
              cursor: "pointer",
              fontFamily: "var(--mem-font-body)",
              fontSize: 12,
            }}
          >
            {items.length > 0
              ? `${t("review.reviewAll")} →`
              : t("home.reviewPageChanges")}
          </button>
        )}
      </div>
    </section>
  );
}

function reviewItemAge(ms: number): string {
  const diff = (Date.now() - ms) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ms).toLocaleDateString();
}

/** Mockup kind-dot palette: revision indigo, page-level the page accent,
 * capture warm, memory merges + suggestions amber, conflicts danger. Green
 * (sage) is reserved for success/added semantics, never a pending-review dot. */
function reviewDotColor(item: ReviewItem): string {
  if (item.kind === "revision") return "var(--mem-accent-indigo)";
  if (item.kind === "capture") return "var(--mem-accent-warm)";
  // Distill discovery never reaches the home rail; colors kept for totality.
  if (item.kind === "stale_page") return "var(--mem-accent-page)";
  if (item.kind === "page_candidate") return "var(--mem-accent-warm)";
  if (item.kind === "topic") return "var(--mem-accent-amber)";
  switch (item.action) {
    case "page_merge":
    case "page_keep_or_archive":
      return "var(--mem-accent-page)";
    case "detect_contradiction":
    case "relation_conflict":
      return "var(--mem-status-danger-text)";
    default:
      // suggest_entity and other refinements fold into amber.
      return "var(--mem-accent-amber)";
  }
}

function ReviewRailItem({
  item,
  onOpenItem,
  isLast = false,
}: {
  item: ReviewItem;
  onOpenItem: (id: string) => void;
  isLast?: boolean;
}) {
  const { t } = useTranslation();
  const kind = reviewKindLabel(t, item);
  // Same rich titles as the review-page cards — a rail row never reads as
  // just its kind label when the names its ids point at can be fetched.
  const { title } = useReviewItemSummary(item);
  // Bare confidence "%" read as an unlabeled number on this terse rail — the
  // dialog carries the precise, labeled metric (overlap / confidence) instead.
  const meta = [
    kind,
    item.timestampMs != null ? reviewItemAge(item.timestampMs) : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <button
      type="button"
      aria-label={t("review.openItem", { title })}
      onClick={() => onOpenItem(reviewItemId(item))}
      style={{
        display: "grid",
        gap: 3,
        width: "100%",
        textAlign: "left",
        border: "0",
        borderBottom: isLast
          ? "none"
          : "1px solid color-mix(in srgb, var(--mem-border) 68%, transparent)",
        borderRadius: 6,
        padding: "8px 2px 9px",
        backgroundColor: "transparent",
        color: "inherit",
        cursor: "pointer",
      }}
    >
      <span
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          minWidth: 0,
        }}
      >
        <span
          aria-hidden="true"
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            flex: "none",
            backgroundColor: reviewDotColor(item),
          }}
        />
        <span
          style={{
            overflow: "hidden",
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
            fontFamily: "var(--mem-font-heading)",
            fontSize: 13,
            fontWeight: 500,
            color: "var(--mem-text)",
            lineHeight: 1.25,
          }}
        >
          {title}
        </span>
      </span>
      <span
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: 11.5,
          color: "var(--mem-text-tertiary)",
          lineHeight: 1.35,
          paddingLeft: 14,
        }}
      >
        {meta}
      </span>
    </button>
  );
}

function PageIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true" className="mt-1.5 shrink-0" style={{ color: "var(--mem-page-icon)" }}>
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
