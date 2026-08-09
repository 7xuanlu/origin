// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  getMemoryDetail,
  getEnrichmentStatus,
  getMemoryRevisions,
  getVersionChain,
  listEntities,
  listAllTags,
  search,
  updateMemory,
  reclassifyMemory,
  pinMemory,
  unpinMemory,
  deleteFileChunks,
  clipboardWrite,
  getPendingRevision,
  acceptPendingRevision,
  dismissPendingRevision,
  listSpaces,
  agentDisplayName,
  MEMORY_FACETS,
  FACET_COLORS,
  STABILITY_TIERS,
  type MemoryType,
  type PendingRevision,
} from "../../lib/tauri";
import TagEditor from "../TagEditor";
import ContentRenderer from "./ContentRenderer";
import { DisclosureButton, PinIcon, RailPanelTitle } from "./MemoryDetailPrimitives";

interface MemoryDetailProps {
  sourceId: string;
  onBack: () => void;
  onNavigateEntity: (entityId: string) => void;
  onNavigateMemory: (sourceId: string) => void;
}

interface MemoryDetailStatusProps {
  readonly ariaLabel: string;
  readonly body: string;
  readonly onBack: () => void;
  readonly title: string;
  readonly backLabel: string;
}

function MemoryDetailStatus({ ariaLabel, body, onBack, title, backLabel }: MemoryDetailStatusProps) {
  return (
    <main className="memory-detail-dossier" aria-label={ariaLabel}>
      <header className="memory-detail-header">
        <button
          type="button"
          onClick={onBack}
          className="memory-detail-back"
          aria-label={backLabel}
          title={backLabel}
        >
          <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
      </header>
      <section className="memory-detail-status-card" aria-live="polite">
        <h2 className="memory-detail-status-title">{title}</h2>
        <p className="memory-detail-status-copy">{body}</p>
      </section>
    </main>
  );
}

function timeAgo(ts: number, locale: string): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto", style: "short" });
  if (diff < 60) return formatter.format(0, "second");
  if (diff < 3600) return formatter.format(-Math.floor(diff / 60), "minute");
  if (diff < 86400) return formatter.format(-Math.floor(diff / 3600), "hour");
  if (diff < 604800) return formatter.format(-Math.floor(diff / 86400), "day");
  return new Date(ts * 1000).toLocaleDateString(locale);
}

function absoluteDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseStructuredEntries(raw: string | null | undefined): Array<[string, string]> {
  if (!raw || raw === "{}") return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return [];
    return Object.entries(parsed).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string" && entry[1].trim().length > 0,
    );
  } catch (error) {
    if (error instanceof SyntaxError) return [];
    throw error;
  }
}

function parseSourceIds(raw: string | null | undefined): string[] {
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || !Array.isArray(parsed.source_ids)) return [];
    return parsed.source_ids.filter((sourceId): sourceId is string => typeof sourceId === "string");
  } catch (error) {
    if (error instanceof SyntaxError) return [];
    throw error;
  }
}

function isMemoryType(value: string | null | undefined): value is MemoryType {
  return MEMORY_FACETS.some((facet) => facet.type === value);
}

function displayTitle(content: string, title: string | null | undefined, fallbackTitle: string): string {
  return title || content.split("\n")[0]?.substring(0, 80) || fallbackTitle;
}

export default function MemoryDetail({
  sourceId,
  onBack,
  onNavigateEntity,
  onNavigateMemory,
}: MemoryDetailProps) {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [reclassifyOpen, setReclassifyOpen] = useState(false);
  const [spacePickerOpen, setSpacePickerOpen] = useState(false);
  const [expandedVersion, setExpandedVersion] = useState<string | null>(null);
  const [pendingRevision, setPendingRevision] = useState<PendingRevision | null>(null);
  const [copied, setCopied] = useState(false);
  const [editingTags, setEditingTags] = useState(false);
  const [sourceExpanded, setSourceExpanded] = useState(false);
  const [revisionHistoryExpanded, setRevisionHistoryExpanded] = useState(false);
  const [versionHistoryExpanded, setVersionHistoryExpanded] = useState(false);
  const [relatedEntitiesExpanded, setRelatedEntitiesExpanded] = useState(false);
  const [relatedMemoriesExpanded, setRelatedMemoriesExpanded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const spaceDropdownRef = useRef<HTMLDivElement>(null);

  const { data: spaces = [] } = useQuery({
    queryKey: ["spaces"],
    queryFn: listSpaces,
  });

  const {
    data: memory,
    isError: memoryLoadFailed,
    isPending: memoryPending,
  } = useQuery({
    queryKey: ["memoryDetail", sourceId],
    queryFn: () => getMemoryDetail(sourceId),
    refetchInterval: 5000,
  });

  const { data: enrichmentStatus } = useQuery({
    queryKey: ["enrichment-status", sourceId],
    queryFn: () => getEnrichmentStatus(sourceId),
    enabled: !!sourceId,
    staleTime: 30000,
    retry: false,
  });

  const { data: versionChain = [] } = useQuery({
    queryKey: ["version-chain", sourceId],
    queryFn: () => getVersionChain(sourceId),
    enabled: !!memory?.supersedes,
  });

  const { data: memoryRevisions } = useQuery({
    queryKey: ["memory-revisions", sourceId],
    queryFn: () => getMemoryRevisions(sourceId),
    enabled: !!sourceId,
    staleTime: 30_000,
    retry: false,
  });

  const { data: relatedEntities = [] } = useQuery({
    queryKey: ["entities", undefined, memory?.domain],
    queryFn: () => listEntities(undefined, memory?.domain ?? undefined),
    enabled: !!memory?.domain,
  });

  // For recaps: fetch exact source memories from structured_fields.source_ids
  // For regular memories: semantic search for related
  const recapSourceIds = memory?.is_recap ? parseSourceIds(memory.structured_fields) : [];

  const { data: relatedSearchResults = [] } = useQuery({
    queryKey: ["relatedMemories", sourceId, recapSourceIds],
    queryFn: async () => {
      if (!memory) return [];
      // Recaps: only show source memories when we have exact IDs
      if (memory.is_recap) {
        if (recapSourceIds.length === 0) return [];
        const results = await Promise.all(
          recapSourceIds.map((id) => getMemoryDetail(id))
        );
        return results.filter((r): r is NonNullable<typeof r> => r != null);
      }
      // Regular memories: semantic search for related
      const results = await search(memory.content.substring(0, 200), 6, "memory");
      return results.filter((r) => r.source_id !== sourceId);
    },
    enabled: !!memory,
    staleTime: 30000,
  });

  const { data: tagData } = useQuery({
    queryKey: ["tags"],
    queryFn: listAllTags,
    refetchInterval: 5000,
  });

  const docTagKey = `memory::${sourceId}`;
  const currentTags = tagData?.document_tags[docTagKey] ?? [];
  const allTags = tagData?.tags ?? [];

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["memoryDetail", sourceId] });
    queryClient.invalidateQueries({ queryKey: ["memory-revisions", sourceId] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
    queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
  };

  const updateMutation = useMutation({
    mutationFn: (content: string) => updateMemory(sourceId, content),
    onSuccess: invalidate,
  });

  const confirmMutation = useMutation({
    mutationFn: (confirmed: boolean) => updateMemory(sourceId, undefined, undefined, confirmed),
    onSuccess: invalidate,
  });

  const reclassifyMutation = useMutation({
    mutationFn: (memoryType: MemoryType) => reclassifyMemory(sourceId, memoryType),
    onSuccess: () => {
      setReclassifyOpen(false);
      invalidate();
    },
  });

  const pinMutation = useMutation({
    mutationFn: () => memory?.pinned ? unpinMemory(sourceId) : pinMemory(sourceId),
    onSuccess: invalidate,
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteFileChunks("memory", sourceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
      onBack();
    },
  });

  const changeSpaceMutation = useMutation({
    mutationFn: (newDomain: string | undefined) =>
      updateMemory(sourceId, undefined, newDomain, undefined, undefined),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memoryDetail", sourceId] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      setSpacePickerOpen(false);
    },
  });

  // Focus textarea on edit
  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  // Close reclassify dropdown on click outside
  useEffect(() => {
    if (!reclassifyOpen) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setReclassifyOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [reclassifyOpen]);

  // Close space picker on click outside
  useEffect(() => {
    if (!spacePickerOpen) return;
    const handler = (e: MouseEvent) => {
      if (spaceDropdownRef.current && !spaceDropdownRef.current.contains(e.target as Node)) {
        setSpacePickerOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [spacePickerOpen]);

  // Fetch pending revision for protected confirmed memories
  useEffect(() => {
    if (!memory) return;
    const facetType = memory.memory_type ?? "fact";
    const tier = STABILITY_TIERS[facetType] ?? "ephemeral";
    if (tier === "protected" && memory.confirmed) {
      getPendingRevision(memory.source_id).then(setPendingRevision).catch(() => {});
    }
  }, [memory]);

  const handleAcceptRevision = async () => {
    if (!pendingRevision) return;
    await acceptPendingRevision(sourceId);
    setPendingRevision(null);
    invalidate();
  };

  const handleDismissRevision = async () => {
    if (!pendingRevision) return;
    await dismissPendingRevision(sourceId);
    setPendingRevision(null);
  };

  const handleSave = () => {
    if (editContent.trim() && editContent !== memory?.content) {
      updateMutation.mutate(editContent.trim());
    }
    setEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === "Escape") {
      setEditContent(memory?.content ?? "");
      setEditing(false);
    }
  };

  if (!memory) {
    const titleKey = memoryLoadFailed
      ? "memoryDetail.loadFailedTitle"
      : memoryPending
        ? "memoryDetail.loadingTitle"
        : "memoryDetail.notFoundTitle";
    const bodyKey = memoryLoadFailed
      ? "memoryDetail.loadFailedBody"
      : memoryPending
        ? "memoryDetail.loadingBody"
        : "memoryDetail.notFoundBody";

    return (
      <MemoryDetailStatus
        ariaLabel={t("memoryDetail.dossierLabel")}
        backLabel={t("memoryDetail.backToMemories")}
        body={t(bodyKey)}
        onBack={onBack}
        title={t(titleKey)}
      />
    );
  }

  const facetType = isMemoryType(memory.memory_type) ? memory.memory_type : "fact";
  const isConfirmed = memory.confirmed;
  const formatTimeAgo = (ts: number) => timeAgo(ts, i18n.resolvedLanguage ?? i18n.language);
  const relatedMemories = relatedSearchResults.slice(0, 5);
  const revisionEntries = memoryRevisions?.entries ?? [];
  const visibleRevisionEntries = revisionHistoryExpanded ? revisionEntries : revisionEntries.slice(0, 1);
  const visibleVersionChain = versionHistoryExpanded ? versionChain : versionChain.slice(0, 3);
  const visibleRelatedEntities = relatedEntitiesExpanded ? relatedEntities : relatedEntities.slice(0, 4);
  const visibleRelatedMemories = relatedMemoriesExpanded ? relatedMemories : relatedMemories.slice(0, 3);
  const hasDaemonRevisionHistory = revisionEntries.length > 0;
  const title = displayTitle(memory.content, memory.title, t("memoryDetail.untitledMemory"));
  const structuredEntries = parseStructuredEntries(memory.structured_fields);
  const hasLegacyVersionHistory = !hasDaemonRevisionHistory && Boolean(memory.supersedes) && versionChain.length > 0;
  // Hero type scales with content length so short memories read as a statement
  // and long ones as an article body (keeps left/right visual balance).
  // Display serif holds up for ~4 lines max; past ~280 chars it reads as a
  // bloated headline, so longer content drops to body text with a lede.
  const contentLength = memory.content.trim().length;
  const heroScale = contentLength <= 160 ? "is-xl" : contentLength <= 280 ? "is-lg" : "is-body";
  const sourceText = memory.source_text?.trim() ?? "";
  const hasSourceExcerpt = sourceText.length > 0 && sourceText !== memory.content.trim();
  const sourceClipped = sourceText.length > 360;
  const visibleSourceText = sourceExpanded || !sourceClipped ? sourceText : `${sourceText.slice(0, 360)}…`;
  const hasConnections = relatedEntities.length > 0 || relatedMemories.length > 0;

  return (
    <main className="memory-detail-dossier" aria-label={t("memoryDetail.dossierLabel")}>
      {/* Topbar: back + state toggles + edit + delete */}
      <header className="memory-detail-header">
        <button
          type="button"
          onClick={onBack}
          className="memory-detail-back"
          aria-label={t("memoryDetail.backToMemories")}
          title={t("memoryDetail.backToMemories")}
        >
          <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>
      </header>

      {/* Dateline row: provenance left, record actions right — one full-frame header line */}
      <div className="memory-detail-eyebrow-row">
        <div className="memory-detail-eyebrow">
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setReclassifyOpen(!reclassifyOpen)}
              className={`memory-detail-facet-button ${FACET_COLORS[facetType]}`}
            >
              {facetType}
              <span className="ml-1 opacity-50">&#x25BE;</span>
            </button>
            {reclassifyOpen && (
              <div
                className="absolute left-0 top-7 z-50 rounded-lg shadow-xl py-1 min-w-[140px]"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                }}
              >
                {MEMORY_FACETS.map((facet) => (
                  <button
                    key={facet.type}
                    className="w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 hover:bg-[var(--mem-hover)]"
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      color: facetType === facet.type ? "var(--mem-text)" : "var(--mem-text-secondary)",
                    }}
                    onClick={() => reclassifyMutation.mutate(facet.type)}
                  >
                    <span className={`memory-detail-facet-dot ${FACET_COLORS[facet.type].split(" ")[0]}`} />
                    {facet.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="relative" ref={spaceDropdownRef}>
            <button
              onClick={() => setSpacePickerOpen(!spacePickerOpen)}
              className={`memory-detail-eyebrow-button ${memory.domain ? "" : "is-empty"}`}
            >
              {memory.domain ?? t("memoryDetail.assignSpace")}
            </button>
            {spacePickerOpen && (
              <div
                className="absolute left-0 top-8 z-50 rounded-lg shadow-xl py-1 min-w-[140px]"
                style={{
                  backgroundColor: "var(--mem-surface)",
                  border: "1px solid var(--mem-border)",
                }}
              >
                {memory.domain && (
                  <>
                    <button
                      onClick={() => changeSpaceMutation.mutate(undefined)}
                      className="w-full text-left px-3 py-1.5 text-xs hover:bg-[var(--mem-hover)]"
                      style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-tertiary)" }}
                    >
                      {t("memoryDetail.removeFromSpace")}
                    </button>
                    <div style={{ height: "1px", backgroundColor: "var(--mem-border)", margin: "2px 0" }} />
                  </>
                )}
                {spaces.filter((s) => s.name !== memory.domain).map((s) => (
                  <button
                    key={s.id}
                    onClick={() => changeSpaceMutation.mutate(s.name)}
                    className="w-full text-left px-3 py-1.5 text-xs hover:bg-[var(--mem-hover)]"
                    style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-secondary)" }}
                  >
                    <span className="capitalize">{s.name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {memory.source_agent && (
            <>
              <span className="memory-detail-eyebrow-sep">·</span>
              <span>{agentDisplayName(memory.source_agent)}</span>
            </>
          )}

          <span className="memory-detail-eyebrow-sep">·</span>
          <time title={absoluteDate(memory.last_modified)}>
            {formatTimeAgo(memory.last_modified)}
          </time>
        </div>

        <div className="memory-detail-actions">
          {/* Copy as context — recap only */}
          {memory.is_recap && (
            <button
              onClick={async () => {
                const space = memory.domain ? `**Space:** ${memory.domain}` : "";
                const time = `**Time:** ${absoluteDate(memory.last_modified)}`;
                const agent = memory.source_agent ? `**Generated by:** ${memory.source_agent}` : "";
                const meta = [space, time, agent].filter(Boolean).join("\n");
                const contentSection = `### Content\n${memory.content}`;
                const text = [
                  "## Activity Recap",
                  meta,
                  "",
                  contentSection,
                ].join("\n");
                await clipboardWrite(text);
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
              }}
              className={`memory-detail-text-button ${copied ? "success" : "accent"}`}
              style={copied ? undefined : { color: "var(--mem-accent-indigo)" }}
              title={t("memoryDetail.copyContextTitle")}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
              {copied ? t("memoryDetail.copied") : t("memoryDetail.copyAsContext")}
            </button>
          )}

          {/* Confirmed toggle */}
          <button
            onClick={() => confirmMutation.mutate(!isConfirmed)}
            className={`memory-detail-toggle ${isConfirmed ? "is-on" : ""}`}
            aria-label={t("memoryDetail.confirmedState", { state: isConfirmed ? t("memoryDetail.yes") : t("memoryDetail.no") })}
          >
            <span className={`memory-detail-state-dot ${isConfirmed ? "is-on" : ""}`} />
            {t("memoryDetail.confirmed")}
          </button>

          {/* Pin toggle */}
          <button
            onClick={() => pinMutation.mutate()}
            className={`memory-detail-toggle ${memory.pinned ? "is-on" : ""}`}
            aria-label={t("memoryDetail.pinnedState", { state: memory.pinned ? t("memoryDetail.yes") : t("memoryDetail.no") })}
            title={memory.pinned ? t("memoryDetail.unpin") : t("memoryDetail.pin")}
          >
            <span className={`memory-detail-state-icon ${memory.pinned ? "is-on" : ""}`}>
              <PinIcon filled={memory.pinned} size={12} />
            </span>
            {t("memoryDetail.pinned")}
          </button>

          {/* Edit */}
          {!editing && memory.memory_type !== "recap" && (
            <button
              onClick={() => { setEditContent(memory.content); setEditing(true); }}
              className="memory-detail-text-button"
              aria-label={t("memoryDetail.editMemory")}
            >
              {t("memoryDetail.edit")}
            </button>
          )}

          {/* Delete */}
          <button
            onClick={() => { if (window.confirm(t("memoryDetail.deleteConfirm"))) deleteMutation.mutate(); }}
            className="memory-detail-icon-button memory-detail-delete"
            aria-label={t("memoryDetail.deleteMemory")}
            title={t("memoryDetail.deleteMemory")}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
            </svg>
          </button>
        </div>
      </div>

      <div className="memory-detail-grid">
      <section className="memory-detail-reading" aria-label={t("memoryDetail.readingLabel")}>
        <h2 className="sr-only">{title}</h2>

        {/* The hero IS the memory — no card, no truncation */}
        {editing ? (
          <div className="memory-detail-editing-surface">
            <span className="memory-detail-editing-label">
              {t("memoryDetail.editing")}
            </span>
            <textarea
              ref={textareaRef}
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onKeyDown={handleKeyDown}
              className="memory-detail-editor"
              rows={Math.max(3, editContent.split("\n").length)}
            />
            <div className="memory-detail-editor-actions">
              <button
                onClick={handleSave}
                className="memory-detail-text-button primary"
              >
                {t("memoryDetail.save")}
              </button>
              <button
                onClick={() => { setEditContent(memory.content); setEditing(false); }}
                className="memory-detail-text-button"
              >
                {t("memoryDetail.cancel")}
              </button>
              <span className="memory-detail-shortcut">
                {t("memoryDetail.saveShortcut")}
              </span>
            </div>
          </div>
        ) : (
          <div className={`memory-detail-hero-text ${heroScale}`}>
            <ContentRenderer
              content={memory.content}
              structuredFields={memory.structured_fields}
              variant="detail"
            />
          </div>
        )}

        {/* Dossier strip: entity · quality · enrichment */}
        {(memory.entity_id || memory.quality === "low" || !!enrichmentStatus) && (
        <div className="memory-detail-strip">
          {memory.entity_id && (
            <span className="memory-detail-strip-item">
              <span className="memory-detail-strip-label">{t("memoryDetail.entity")}</span>
              <button
                onClick={() => {
                  if (memory.entity_id) onNavigateEntity(memory.entity_id);
                }}
                className="memory-detail-link-button"
              >
                {t("memoryDetail.linked")}
              </button>
            </span>
          )}

          {memory.quality && memory.quality === "low" && (
            <span className="memory-chip warning">
              {t("memoryDetail.lowQuality")}
            </span>
          )}

          {enrichmentStatus && (
            <span className="memory-detail-strip-item">
              <span className="memory-detail-strip-label">{t("memoryDetail.enrichment")}</span>
              <span
                className="memory-chip indigo"
                title={enrichmentStatus.steps.map((s) => `${s.step}: ${s.status}`).join("\n")}
              >
                {enrichmentStatus.summary}
              </span>
            </span>
          )}
        </div>
        )}

        {/* Tags: own wrapping line, "+" anchored */}
        <div className="memory-detail-tagline">
          <span className="memory-detail-strip-label">{t("memoryDetail.tags")}</span>
          {currentTags.map((tag) => (
            <span key={tag} className="memory-chip indigo">
              {tag}
            </span>
          ))}
          <div className="relative">
            <button
              onClick={() => setEditingTags(!editingTags)}
              className="memory-detail-tag-edit"
              aria-label={t("memoryDetail.editTags")}
              title={t("memoryDetail.editTags")}
            >
              {editingTags ? t("memoryDetail.done") : "+"}
            </button>
            {editingTags && (
              <TagEditor
                source="memory"
                sourceId={sourceId}
                lastModified={memory.last_modified}
                currentTags={currentTags}
                allTags={allTags}
                onClose={() => setEditingTags(false)}
                onTagsChanged={() => queryClient.invalidateQueries({ queryKey: ["tags"] })}
              />
            )}
          </div>
        </div>
      {structuredEntries.length > 0 && (
        <section className="memory-detail-structured">
          <h3 className="memory-detail-subsection">{t("memoryDetail.details")}</h3>
          <div className="memory-detail-field-list">
            {structuredEntries.map(([key, val]) => (
              <div key={key} className="memory-detail-field-row">
                <span className="memory-detail-field-label">{key.replace(/_/g, " ")}</span>
                <span className="memory-detail-field-value">{val}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Pending revision */}
      {pendingRevision && (
        <div className="memory-detail-pending">
          <div className="memory-detail-pending-title">
            {pendingRevision.source_agent
              ? t("memoryDetail.proposedUpdateFrom", { agent: pendingRevision.source_agent })
              : t("memoryDetail.proposedUpdate")}
          </div>
          <p className="memory-detail-pending-copy">
            {pendingRevision.content}
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleAcceptRevision}
              className="memory-detail-text-button primary"
            >
              {t("memoryDetail.accept")}
            </button>
            <button
              onClick={handleDismissRevision}
              className="memory-detail-text-button"
            >
              {t("memoryDetail.dismiss")}
            </button>
          </div>
        </div>
      )}

      {/* Source excerpt: the captured text behind this memory */}
      {hasSourceExcerpt && (
        <section className="memory-detail-source">
          <h3 className="memory-detail-subsection">{t("memoryDetail.sourceTitle")}</h3>
          <blockquote className="memory-detail-source-quote">
            {visibleSourceText}
          </blockquote>
          {sourceClipped && (
            <DisclosureButton
              ariaLabel={sourceExpanded ? t("memoryDetail.showLess") : t("memoryDetail.showFullSource")}
              onClick={() => setSourceExpanded(!sourceExpanded)}
            >
              {sourceExpanded ? t("memoryDetail.showLessCompact") : t("memoryDetail.showFullSource")}
            </DisclosureButton>
          )}
        </section>
      )}

      {/* Provenance: revision + legacy version history, quiet at the bottom */}
      {(hasDaemonRevisionHistory || hasLegacyVersionHistory) && (
        <div className="memory-detail-provenance-group">
          {/* Revision history from daemon supersession chain */}
          {hasDaemonRevisionHistory && (
            <section className="memory-detail-rail-panel memory-detail-secondary-panel">
              <div className="memory-detail-panel-heading">
                <RailPanelTitle>{t("memoryDetail.revisionHistory")}</RailPanelTitle>
                {revisionEntries.length > visibleRevisionEntries.length && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showAll", { count: revisionEntries.length })}
                    count={revisionEntries.length}
                    onClick={() => setRevisionHistoryExpanded(true)}
                  >
                    {t("memoryDetail.showAllCompact")}
                  </DisclosureButton>
                )}
                {revisionHistoryExpanded && revisionEntries.length > 1 && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showLess")}
                    onClick={() => setRevisionHistoryExpanded(false)}
                  >
                    {t("memoryDetail.showLessCompact")}
                  </DisclosureButton>
                )}
              </div>
              <div className="memory-detail-compact-list">
                {visibleRevisionEntries.map((entry) => {
                  const isCurrent = entry.source_id === memoryRevisions?.current_source_id;
                  const body = (
                    <div className="memory-detail-context-row-body">
                      <div className="memory-detail-context-row-meta">
                        <span className="memory-detail-context-row-title truncate">
                          {entry.title || t("memoryDetail.untitledMemory")}
                        </span>
                        {isCurrent && (
                          <span className="memory-detail-context-row-badge">
                            {t("memoryDetail.current")}
                          </span>
                        )}
                        <span className="memory-detail-context-row-meta-text">
                          {t("memoryDetail.depth", { depth: entry.depth })}
                        </span>
                        <span className="memory-detail-context-row-meta-text">
                          {formatTimeAgo(entry.last_modified)}
                        </span>
                      </div>
                      {entry.delta_summary && (
                        <p className="memory-detail-context-row-summary">
                          {entry.delta_summary}
                        </p>
                      )}
                      <p className="memory-detail-context-row-preview line-clamp-2">
                        {entry.content_preview}
                      </p>
                      {entry.source_agent && (
                        <p className="memory-detail-context-row-agent">
                          {entry.source_agent}
                        </p>
                      )}
                    </div>
                  );
                  if (isCurrent) {
                    return (
                      <div
                        key={entry.source_id}
                        className="memory-detail-context-row is-current"
                      >
                        {body}
                      </div>
                    );
                  }
                  return (
                    <button
                      key={entry.source_id}
                      onClick={() => onNavigateMemory(entry.source_id)}
                      className="memory-detail-context-row memory-detail-context-row-button"
                    >
                      {body}
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {/* Version history */}
          {hasLegacyVersionHistory && (
            <section className="memory-detail-rail-panel memory-detail-secondary-panel">
              <div className="memory-detail-panel-heading">
                <RailPanelTitle>{t("memoryDetail.versionHistory")}</RailPanelTitle>
                {versionChain.length > visibleVersionChain.length && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showAll", { count: versionChain.length })}
                    count={versionChain.length}
                    onClick={() => setVersionHistoryExpanded(true)}
                  >
                    {t("memoryDetail.showAllCompact")}
                  </DisclosureButton>
                )}
                {versionHistoryExpanded && versionChain.length > 3 && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showLess")}
                    onClick={() => setVersionHistoryExpanded(false)}
                  >
                    {t("memoryDetail.showLessCompact")}
                  </DisclosureButton>
                )}
              </div>
              <div className="memory-detail-compact-list">
                {visibleVersionChain.map((v, i) => (
                  <div key={v.source_id}>
                    <button
                      onClick={() => setExpandedVersion(expandedVersion === v.source_id ? null : v.source_id)}
                      className="memory-detail-context-row memory-detail-context-row-button"
                    >
                      <span className="memory-detail-version-index">
                        v{i + 1}
                      </span>
                      <span
                        className={`memory-detail-version-title ${v.source_id === sourceId ? "is-current" : ""} truncate`}
                      >
                        {v.title}
                      </span>
                      {v.confirmed && (
                        <span className="memory-detail-context-row-meta-text is-warm">
                          {t("memoryDetail.confirmedStatus")}
                        </span>
                      )}
                      <span className="memory-detail-context-row-meta-text">
                        {formatTimeAgo(v.last_modified)}
                      </span>
                      <svg
                        width="12" height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        className="memory-detail-version-chevron"
                        style={{
                          transform: expandedVersion === v.source_id ? "rotate(90deg)" : "rotate(0deg)",
                        }}
                      >
                        <path d="M9 18l6-6-6-6" />
                      </svg>
                    </button>
                    {expandedVersion === v.source_id && (
                      <div
                        className="memory-detail-version-preview"
                      >
                        {v.content}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

        </div>
      )}

      </section>

      {/* Marginalia rail: connections only */}
      {hasConnections && (
        <aside className="memory-detail-rail" aria-label={t("memoryDetail.contextLabel")}>
          <h3 className="memory-detail-rail-heading">{t("memoryDetail.connections")}</h3>

          {/* Related / Source memories */}
          {relatedMemories.length > 0 && (
            <section className="memory-detail-rail-section">
              <div className="memory-detail-panel-heading">
                <RailPanelTitle>
                  {memory.is_recap ? t("memoryDetail.sourceMemories") : t("memoryDetail.relatedMemories")}
                </RailPanelTitle>
                {relatedMemories.length > visibleRelatedMemories.length && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showAll", { count: relatedMemories.length })}
                    count={relatedMemories.length}
                    onClick={() => setRelatedMemoriesExpanded(true)}
                  >
                    {t("memoryDetail.showAllCompact")}
                  </DisclosureButton>
                )}
                {relatedMemoriesExpanded && relatedMemories.length > 3 && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showLess")}
                    onClick={() => setRelatedMemoriesExpanded(false)}
                  >
                    {t("memoryDetail.showLessCompact")}
                  </DisclosureButton>
                )}
              </div>
              <div className="memory-detail-related-grid">
                {visibleRelatedMemories.map((r) => {
                  const rFacet = r.memory_type ?? null;
                  const rColor = rFacet ? FACET_COLORS[rFacet] : null;
                  return (
                    <button
                      key={'id' in r ? r.id : r.source_id}
                      onClick={() => onNavigateMemory(r.source_id)}
                      className="memory-detail-related-card"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="memory-detail-related-copy line-clamp-2">
                          {r.content.length > 160 ? r.content.substring(0, 160) + "\u2026" : r.content}
                        </p>
                        <div className="memory-detail-related-meta">
                          {rFacet && rColor && (
                            <span className={`memory-detail-related-facet ${rColor}`}>
                              {rFacet}
                            </span>
                          )}
                          <span>{formatTimeAgo(r.last_modified)}</span>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>
          )}

          {/* Related entities */}
          {relatedEntities.length > 0 && (
            <section className="memory-detail-rail-section">
              <div className="memory-detail-panel-heading">
                <RailPanelTitle>{t("memoryDetail.relatedEntities")}</RailPanelTitle>
                {relatedEntities.length > visibleRelatedEntities.length && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showAll", { count: relatedEntities.length })}
                    count={relatedEntities.length}
                    onClick={() => setRelatedEntitiesExpanded(true)}
                  >
                    {t("memoryDetail.showAllCompact")}
                  </DisclosureButton>
                )}
                {relatedEntitiesExpanded && relatedEntities.length > 4 && (
                  <DisclosureButton
                    ariaLabel={t("memoryDetail.showLess")}
                    onClick={() => setRelatedEntitiesExpanded(false)}
                  >
                    {t("memoryDetail.showLessCompact")}
                  </DisclosureButton>
                )}
              </div>
              <div className="memory-detail-entity-chip-list">
                {visibleRelatedEntities.map((entity) => (
                  <button
                    key={entity.id}
                    onClick={() => onNavigateEntity(entity.id)}
                    className="memory-detail-entity-chip"
                  >
                    <span className="memory-detail-entity-name">
                      {entity.name}
                    </span>
                    <span className="memory-detail-entity-type">
                      {entity.entity_type}
                    </span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </aside>
      )}
      </div>
    </main>
  );
}
