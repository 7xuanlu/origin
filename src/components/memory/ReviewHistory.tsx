// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  getMemoryRevisions,
  listRecentChanges,
  type MemoryRevisionEntry,
  type PageChange,
  type PageChangeKind,
} from "../../lib/tauri";
import { truncateReviewText } from "./ReviewDialog";
import { relativeMs } from "./page/format";

const secondaryTextStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  color: "var(--mem-text-secondary)",
};

/** Mirrors DistillReviewPanel's section-label recipe (not exported there). */
const sectionTitleStyle: React.CSSProperties = {
  margin: "0 0 10px",
  fontFamily: "var(--mem-font-body)",
  fontSize: "12px",
  fontWeight: 600,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
  display: "flex",
  alignItems: "baseline",
  gap: 8,
};

/** Mirrors ReviewDialog's pane-label recipe (not exported there). */
const paneLabelStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 11,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
  margin: "0 0 7px",
};

const itemSurfaceStyle: React.CSSProperties = {
  border: "1px solid var(--mem-border)",
  borderRadius: 8,
  backgroundColor: "var(--mem-surface)",
};

/** Read-only rows navigate on click; hover feedback only, never a hint of action. */
const rowInteractiveClassName =
  "text-left transition-colors duration-150 hover:bg-[var(--mem-hover)] focus-visible:outline-2 focus-visible:outline-[var(--mem-accent-indigo)] focus-visible:outline-offset-2";

function ShimmerRow({ height = 38 }: { height?: number }) {
  return (
    <div
      aria-hidden="true"
      className="spine-indexing"
      style={{
        ...itemSurfaceStyle,
        backgroundColor: "var(--mem-detail-surface-raised)",
        height,
      }}
    />
  );
}

const CHANGE_GLYPH: Record<PageChangeKind, string> = {
  revised: "✎",
  merged: "⇄",
  created: "+",
};

const CHANGE_TONE: Record<PageChangeKind, { color: string; background: string }> = {
  revised: {
    color: "var(--mem-accent-indigo)",
    background: "var(--mem-indigo-bg)",
  },
  merged: {
    color: "var(--mem-accent-warm)",
    background: "color-mix(in srgb, var(--mem-accent-warm) 15%, transparent)",
  },
  created: {
    color: "var(--mem-accent-sage)",
    background: "color-mix(in srgb, var(--mem-accent-sage) 15%, transparent)",
  },
};

function changeLabel(t: TFunction, kind: PageChangeKind): string {
  const keys = {
    revised: "review.changeRevised",
    merged: "review.changeMerged",
    created: "review.changeCreated",
  } as const;
  return t(keys[kind]);
}

function RecentRevisionRow({
  change,
  onClick,
}: {
  change: PageChange;
  onClick: (pageId: string) => void;
}) {
  const { t } = useTranslation();
  const tone = CHANGE_TONE[change.change_kind];
  return (
    <button
      type="button"
      onClick={() => onClick(change.page_id)}
      className={rowInteractiveClassName}
      style={{
        ...itemSurfaceStyle,
        display: "flex",
        alignItems: "center",
        gap: 8,
        width: "100%",
        padding: "10px 14px",
        cursor: "pointer",
        fontFamily: "var(--mem-font-body)",
      }}
    >
      <span aria-hidden="true" style={{ fontSize: 14, lineHeight: 1, color: tone.color }}>
        {CHANGE_GLYPH[change.change_kind]}
      </span>
      <span
        style={{
          flex: 1,
          minWidth: 0,
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          fontFamily: "var(--mem-font-heading)",
          fontSize: 14,
          fontWeight: 500,
          color: "var(--mem-text)",
        }}
      >
        {change.title}
      </span>
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
        {changeLabel(t, change.change_kind)}
      </span>
      <span
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontVariantNumeric: "tabular-nums",
          fontSize: 12,
          color: "var(--mem-text-tertiary)",
          whiteSpace: "nowrap",
        }}
      >
        {relativeMs(change.changed_at_ms)}
      </span>
    </button>
  );
}

/** The wiki's changelog — proof that revisions happen, even while the pending
 * queue is empty. Always renders; read-only, no retry affordance on error
 * (the panel-level Refresh already covers it). */
export function RecentRevisionsSection({
  onPageClick,
}: {
  onPageClick: (pageId: string) => void;
}) {
  const { t } = useTranslation();
  const query = useQuery({
    queryKey: ["recent-changes"],
    queryFn: () => listRecentChanges(8),
    staleTime: 60_000,
  });
  const changes = query.data ?? [];

  return (
    <section>
      <h2 style={sectionTitleStyle}>{t("review.sectionRecentRevisions")}</h2>
      <div className="grid gap-2.5">
        {query.isLoading ? (
          <>
            <ShimmerRow />
            <ShimmerRow />
            <ShimmerRow />
          </>
        ) : query.isError ? (
          <p style={{ ...secondaryTextStyle, margin: 0, fontSize: 12 }}>
            {t("review.recentRevisionsFailed")}
          </p>
        ) : changes.length === 0 ? (
          <p style={{ ...secondaryTextStyle, margin: 0, fontSize: 12 }}>
            {t("review.recentRevisionsEmpty")}
          </p>
        ) : (
          changes.map((change) => (
            <RecentRevisionRow
              key={`${change.page_id}-${change.changed_at_ms}`}
              change={change}
              onClick={onPageClick}
            />
          ))
        )}
      </div>
      <p style={{ ...secondaryTextStyle, margin: "8px 0 0", fontSize: 12 }}>
        {t("review.recentRevisionsHint")}
      </p>
    </section>
  );
}

function MemoryRevisionRow({
  entry,
  onClick,
}: {
  entry: MemoryRevisionEntry;
  onClick: (sourceId: string) => void;
}) {
  const { t } = useTranslation();
  // Only the most recent prior version (depth 1) carries a preview line —
  // deeper entries stay single-line so the chain doesn't overwhelm the diff.
  const preview =
    entry.depth === 1 ? truncateReviewText(entry.content_preview, 80) : null;
  return (
    <button
      type="button"
      onClick={() => onClick(entry.source_id)}
      className={rowInteractiveClassName}
      style={{
        ...itemSurfaceStyle,
        display: "grid",
        gap: 4,
        width: "100%",
        padding: "8px 12px",
        cursor: "pointer",
        fontFamily: "var(--mem-font-body)",
      }}
    >
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            fontSize: 12.5,
            color: "var(--mem-text-secondary)",
          }}
        >
          {entry.title}
        </span>
        {entry.supersede_mode === "protected_revision" && (
          <span
            style={{
              fontSize: 10.5,
              letterSpacing: "0.04em",
              borderRadius: 5,
              padding: "1px 7px",
              color: "var(--mem-accent-warm)",
              backgroundColor: "color-mix(in srgb, var(--mem-accent-warm) 15%, transparent)",
              whiteSpace: "nowrap",
            }}
          >
            {t("review.historyProtected")}
          </span>
        )}
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontVariantNumeric: "tabular-nums",
            fontSize: 11.5,
            color: "var(--mem-text-tertiary)",
            whiteSpace: "nowrap",
          }}
        >
          {relativeMs(entry.last_modified * 1000)}
        </span>
      </span>
      {preview && (
        <span
          style={{
            fontSize: 12,
            lineHeight: 1.4,
            color: "var(--mem-text-tertiary)",
          }}
        >
          {preview}
        </span>
      )}
    </button>
  );
}

/** Beneath a revision card's diff: proof the memory itself carries history.
 * Garnish, not decision — omits entirely on error or an empty chain rather
 * than compete with the accept/dismiss choice above it. */
export function MemoryRevisionChain({
  sourceId,
  onOpenMemory,
}: {
  sourceId: string | null;
  onOpenMemory: (sourceId: string) => void;
}) {
  const { t } = useTranslation();
  const query = useQuery({
    queryKey: ["memory-revisions", sourceId],
    queryFn: () => getMemoryRevisions(sourceId as string),
    enabled: sourceId != null,
    staleTime: 60_000,
  });

  if (sourceId == null) return null;
  if (query.isError) return null;

  if (query.isLoading) {
    return (
      <div style={{ marginTop: 18 }}>
        <ShimmerRow height={34} />
      </div>
    );
  }

  const entries = query.data?.entries ?? [];
  const chainDepth = query.data?.chain_depth ?? 0;
  if (chainDepth <= 1 && entries.length === 0) return null;

  return (
    <div style={{ marginTop: 18 }}>
      <p style={{ ...paneLabelStyle, fontVariantNumeric: "tabular-nums" }}>
        {t("review.historyLabel", { count: entries.length })}
      </p>
      <div className="grid gap-1.5">
        {entries.map((entry) => (
          <MemoryRevisionRow key={entry.source_id} entry={entry} onClick={onOpenMemory} />
        ))}
      </div>
    </div>
  );
}
