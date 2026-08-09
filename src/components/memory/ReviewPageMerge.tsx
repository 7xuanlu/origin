// SPDX-License-Identifier: AGPL-3.0-only
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  getPage,
  getPageSources,
  type Page,
  type PageSourceWithMemory,
} from "../../lib/tauri";
import { truncateReviewText } from "./ReviewDialog";

/** Reused verbatim from ReviewDialog's diff vocabulary. Not exported there
 * (ReviewDialog.tsx:307-318), so replicated here — the AFTER pane is the one
 * place this dossier borrows the ins/del language, applied to static text
 * rather than a word diff (see plan §3.3 "DiffText reuse decision"). */
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

/** Replicated from ReviewDialog.tsx:320-331 (not exported). */
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

/** Replicated from ReviewDialog.tsx:333-340 (not exported). */
const paneLabelStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: 11,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "var(--mem-text-tertiary)",
  margin: "0 0 7px",
};

/** Shimmer recipe matching MemoryCard.tsx's inline `mem-shimmer` usage — the
 * keyframes (index.css:1564-1567) are box-shadow-only, so pair with a raised
 * background to read as a loading surface. */
const shimmerStyle: React.CSSProperties = {
  backgroundColor: "var(--mem-detail-surface-raised)",
  animation: "mem-shimmer 1.6s ease-in-out infinite",
};

const dotBase: React.CSSProperties = {
  display: "inline-block",
  boxSizing: "border-box",
  width: 10,
  height: 10,
  borderRadius: "50%",
};

/** Set math for the merge dossier's source ledger — the client-computed
 * ground truth (plan §3.1). Membership by `source.memory_source_id`. */
export function deriveMergeLedger(
  keepSources: PageSourceWithMemory[],
  retireSources: PageSourceWithMemory[],
): {
  shared: PageSourceWithMemory[];
  onlyKeep: PageSourceWithMemory[];
  onlyRetire: PageSourceWithMemory[];
} {
  const keepIds = new Set(keepSources.map((entry) => entry.source.memory_source_id));
  const retireIds = new Set(
    retireSources.map((entry) => entry.source.memory_source_id),
  );
  return {
    shared: keepSources.filter((entry) =>
      retireIds.has(entry.source.memory_source_id),
    ),
    onlyKeep: keepSources.filter(
      (entry) => !retireIds.has(entry.source.memory_source_id),
    ),
    onlyRetire: retireSources.filter(
      (entry) => !keepIds.has(entry.source.memory_source_id),
    ),
  };
}

function sourceLabel(entry: PageSourceWithMemory): string {
  const label =
    entry.memory?.title ||
    truncateReviewText(entry.memory?.content ?? "", 40) ||
    entry.source.memory_source_id;
  return truncateReviewText(label, 40);
}

function VerdictBanner({
  loading,
  sourcesError,
  onlyRetireCount,
  sourceOverlap,
  sourceOverlapRatio,
}: {
  loading: boolean;
  sourcesError: boolean;
  onlyRetireCount: number;
  sourceOverlap: number;
  sourceOverlapRatio: number;
}) {
  const { t } = useTranslation();
  const base: React.CSSProperties = {
    borderRadius: 10,
    padding: "10px 14px",
    fontFamily: "var(--mem-font-body)",
    fontSize: 13.5,
  };

  if (loading) return <div style={{ ...base, height: 20, ...shimmerStyle }} />;

  if (sourcesError) {
    return (
      <div
        style={{
          ...base,
          backgroundColor: "var(--mem-detail-surface-raised)",
          color: "var(--mem-text-secondary)",
        }}
      >
        {t("review.mergeVerdictUnknown", {
          count: sourceOverlap,
          percent: Math.round(sourceOverlapRatio * 100),
        })}
      </div>
    );
  }

  if (onlyRetireCount === 0) {
    return (
      <div
        style={{
          ...base,
          backgroundColor: "var(--mem-status-success-bg)",
          color: "var(--mem-status-success-text)",
          border:
            "1px solid color-mix(in srgb, var(--mem-accent-sage) 36%, transparent)",
        }}
      >
        <span aria-hidden>✓ </span>
        {t("review.mergeVerdictSafe")}
      </div>
    );
  }

  // Transfer case — never a danger tone; a merge that moves sources is
  // normal, not alarming (plan §3.3).
  return (
    <div
      style={{
        ...base,
        backgroundColor: "var(--mem-indigo-bg)",
        color: "var(--mem-accent-indigo)",
        border:
          "1px solid color-mix(in srgb, var(--mem-accent-indigo) 36%, transparent)",
      }}
    >
      {t("review.mergeVerdictMoves", { count: onlyRetireCount })}
    </div>
  );
}

function BeforePane({
  eyebrow,
  pageId,
  page,
  isLoading,
  isError,
  sourceCount,
  borderColor,
  onOpenPage,
}: {
  eyebrow: string;
  pageId: string;
  page: Page | null | undefined;
  isLoading: boolean;
  isError: boolean;
  sourceCount: number;
  borderColor: string;
  onOpenPage: (pageId: string) => void;
}) {
  const { t } = useTranslation();
  const box: React.CSSProperties = { ...paneStyle, borderTop: `2px solid ${borderColor}` };

  return (
    <div>
      <p style={paneLabelStyle}>{eyebrow}</p>
      {isLoading ? (
        <div style={{ ...box, height: 200, ...shimmerStyle }} />
      ) : isError || !page ? (
        <div style={box}>
          <div style={{ fontFamily: "var(--mem-font-mono)", fontSize: 12, color: "var(--mem-text)" }}>
            {pageId}
          </div>
          <div
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: 12.5,
              color: "var(--mem-text-tertiary)",
              marginTop: 4,
            }}
          >
            {t("review.pageLoadFailed")}
          </div>
        </div>
      ) : (
        <div style={box}>
          <button
            type="button"
            onClick={() => onOpenPage(page.id)}
            className="hover:underline"
            style={{
              display: "block",
              width: "100%",
              textAlign: "left",
              fontFamily: "var(--mem-font-heading)",
              fontWeight: 500,
              fontSize: 15,
              color: "var(--mem-text)",
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
            }}
          >
            {page.title}
          </button>
          <div
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontVariantNumeric: "tabular-nums",
              fontSize: 11.5,
              color: "var(--mem-text-tertiary)",
              margin: "4px 0 0",
            }}
          >
            {t("review.mergeMeta", {
              version: page.version,
              sources: sourceCount,
              chars: page.content.length,
            })}
          </div>
          <div style={{ borderTop: "1px solid var(--mem-detail-divider)", margin: "8px 0" }} />
          <div
            style={{
              fontSize: 13.5,
              lineHeight: 1.6,
              maxHeight: 200,
              overflowY: "auto",
            }}
          >
            {page.content || page.summary || ""}
          </div>
        </div>
      )}
    </div>
  );
}

function SourceChip({
  entry,
  onOpenMemory,
}: {
  entry: PageSourceWithMemory;
  onOpenMemory: (memorySourceId: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onOpenMemory(entry.source.memory_source_id)}
      className="transition-colors duration-150 hover:bg-[var(--mem-hover)]"
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: 12,
        color: "var(--mem-text)",
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
        borderRadius: 999,
        padding: "3px 10px",
        cursor: "pointer",
      }}
    >
      {sourceLabel(entry)}
    </button>
  );
}

function SourceGroup({
  dotStyle,
  label,
  entries,
  emptyText,
  emptyTone,
  onOpenMemory,
}: {
  dotStyle: React.CSSProperties;
  label: string;
  entries: PageSourceWithMemory[];
  emptyText: string;
  emptyTone: "success" | "tertiary";
  onOpenMemory: (memorySourceId: string) => void;
}) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          fontFamily: "var(--mem-font-body)",
          fontSize: 12.5,
          color: "var(--mem-text-secondary)",
          marginBottom: 6,
        }}
      >
        <span style={dotStyle} aria-hidden />
        <span style={{ marginLeft: 6 }}>{label}</span>
        <span
          style={{
            marginLeft: 6,
            fontFamily: "var(--mem-font-mono)",
            fontVariantNumeric: "tabular-nums",
            color: "var(--mem-text-tertiary)",
          }}
        >
          {entries.length}
        </span>
      </div>
      {entries.length === 0 ? (
        <p
          style={{
            margin: 0,
            fontFamily: "var(--mem-font-body)",
            fontSize: 12.5,
            color:
              emptyTone === "success"
                ? "var(--mem-status-success-text)"
                : "var(--mem-text-tertiary)",
          }}
        >
          {emptyText}
        </p>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {entries.map((entry) => (
            <SourceChip
              key={entry.source.memory_source_id}
              entry={entry}
              onOpenMemory={onOpenMemory}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const ghostChipStyle: React.CSSProperties = {
  display: "inline-block",
  width: 64,
  height: 22,
  borderRadius: 999,
  border: "1px solid var(--mem-border)",
  animation: "mem-shimmer 1.6s ease-in-out infinite",
};

function LedgerSkeleton() {
  return (
    <div style={{ display: "grid", gap: 10 }} aria-hidden>
      {[0, 1, 2].map((row) => (
        <div key={row} style={{ display: "flex", gap: 6 }}>
          {[0, 1, 2, 3].map((chip) => (
            <span key={chip} style={ghostChipStyle} />
          ))}
        </div>
      ))}
    </div>
  );
}

export function PageMergeStripOff({
  keepId,
  retireId,
  onOpenPage,
  onOpenMemory,
  sourceOverlap = 0,
  sourceOverlapRatio = 0,
}: {
  keepId: string;
  retireId: string;
  onOpenPage: (pageId: string) => void;
  onOpenMemory: (memorySourceId: string) => void;
  /**
   * Verdict-banner fallback numbers for when the client-side source-list
   * fetch fails (plan §3.3 block A: "banner falls back to the payload
   * numbers"). Wire these from the page_merge proposal's
   * `item.payload.source_overlap` / `source_overlap_ratio` — the daemon's
   * numbers, not client-derived. Optional + defaulted so this component
   * still renders without them; the caller should pass real values for the
   * error-state message to be meaningful.
   */
  sourceOverlap?: number;
  sourceOverlapRatio?: number;
}) {
  const { t } = useTranslation();

  const keepPageQ = useQuery({
    queryKey: ["page", keepId],
    queryFn: () => getPage(keepId),
    enabled: Boolean(keepId),
    staleTime: 60_000,
  });
  const retirePageQ = useQuery({
    queryKey: ["page", retireId],
    queryFn: () => getPage(retireId),
    enabled: Boolean(retireId),
    staleTime: 60_000,
  });
  const keepSourcesQ = useQuery({
    queryKey: ["page-sources", keepId],
    queryFn: () => getPageSources(keepId),
    enabled: Boolean(keepId),
    staleTime: 60_000,
  });
  const retireSourcesQ = useQuery({
    queryKey: ["page-sources", retireId],
    queryFn: () => getPageSources(retireId),
    enabled: Boolean(retireId),
    staleTime: 60_000,
  });

  const sourcesLoading = keepSourcesQ.isLoading || retireSourcesQ.isLoading;
  const sourcesError = keepSourcesQ.isError || retireSourcesQ.isError;
  const pagesLoading = keepPageQ.isLoading || retirePageQ.isLoading;

  const ledger = useMemo(
    () => deriveMergeLedger(keepSourcesQ.data ?? [], retireSourcesQ.data ?? []),
    [keepSourcesQ.data, retireSourcesQ.data],
  );
  const union = ledger.shared.length + ledger.onlyKeep.length + ledger.onlyRetire.length;

  const keepSourceCount =
    keepSourcesQ.data?.length ?? keepPageQ.data?.source_memory_ids.length ?? 0;
  const retireSourceCount =
    retireSourcesQ.data?.length ?? retirePageQ.data?.source_memory_ids.length ?? 0;

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <VerdictBanner
        loading={sourcesLoading}
        sourcesError={sourcesError}
        onlyRetireCount={ledger.onlyRetire.length}
        sourceOverlap={sourceOverlap}
        sourceOverlapRatio={sourceOverlapRatio}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 12,
        }}
      >
        <BeforePane
          eyebrow={t("review.mergeKeptPage")}
          pageId={keepId}
          page={keepPageQ.data}
          isLoading={keepPageQ.isLoading}
          isError={keepPageQ.isError}
          sourceCount={keepSourceCount}
          borderColor="var(--mem-accent-sage)"
          onOpenPage={onOpenPage}
        />
        <BeforePane
          eyebrow={t("review.mergeRetiringPage")}
          pageId={retireId}
          page={retirePageQ.data}
          isLoading={retirePageQ.isLoading}
          isError={retirePageQ.isError}
          sourceCount={retireSourceCount}
          borderColor="var(--mem-accent-warm)"
          onOpenPage={onOpenPage}
        />
      </div>

      <div>
        <p style={paneLabelStyle}>{t("review.mergeSourcesLabel")}</p>
        {sourcesLoading ? (
          <LedgerSkeleton />
        ) : sourcesError ? (
          <p
            style={{
              margin: 0,
              fontFamily: "var(--mem-font-body)",
              fontSize: 12.5,
              color: "var(--mem-text-tertiary)",
            }}
          >
            {t("review.sourcesLoadFailed")}
          </p>
        ) : (
          <>
            <SourceGroup
              dotStyle={{ ...dotBase, backgroundColor: "var(--mem-accent-sage)" }}
              label={t("review.mergeShared")}
              entries={ledger.shared}
              emptyText="—"
              emptyTone="tertiary"
              onOpenMemory={onOpenMemory}
            />
            <SourceGroup
              dotStyle={{
                ...dotBase,
                backgroundColor: "transparent",
                border: "1px solid var(--mem-text-tertiary)",
              }}
              label={t("review.mergeOnlyKept")}
              entries={ledger.onlyKeep}
              emptyText="—"
              emptyTone="tertiary"
              onOpenMemory={onOpenMemory}
            />
            <SourceGroup
              dotStyle={{ ...dotBase, backgroundColor: "var(--mem-accent-warm)" }}
              label={t("review.mergeOnlyRetiring")}
              entries={ledger.onlyRetire}
              emptyText={t("review.mergeNoUnique")}
              emptyTone="success"
              onOpenMemory={onOpenMemory}
            />
          </>
        )}
      </div>

      {!sourcesError && (
        <div>
          <p style={paneLabelStyle}>{t("review.mergeAfterLabel")}</p>
          {pagesLoading || sourcesLoading ? (
            <div style={{ ...paneStyle, height: 72, ...shimmerStyle }} />
          ) : (
            <div style={paneStyle}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <span
                  style={{
                    fontFamily: "var(--mem-font-heading)",
                    fontWeight: 500,
                    fontSize: 14.5,
                    color: "var(--mem-text)",
                  }}
                >
                  {keepPageQ.data?.title ?? keepId}
                </span>
                <ins
                  style={{
                    ...INS_STYLE,
                    marginLeft: "auto",
                    fontFamily: "var(--mem-font-mono)",
                    fontVariantNumeric: "tabular-nums",
                    fontSize: 12,
                  }}
                >
                  {t("review.sources", { count: union })}
                </ins>
              </div>
              <div style={{ marginTop: 6, fontSize: 13.5 }}>
                <del style={DEL_STYLE}>{retirePageQ.data?.title ?? retireId}</del>{" "}
                {ledger.onlyRetire.length === 0
                  ? t("review.mergeAfterSafe", { count: ledger.shared.length })
                  : t("review.mergeAfterMoves", { count: ledger.onlyRetire.length })}
              </div>
              <p
                style={{
                  margin: "8px 0 0",
                  fontFamily: "var(--mem-font-body)",
                  fontSize: 12,
                  color: "var(--mem-text-tertiary)",
                }}
              >
                {t("review.mergeAfterNote")}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
