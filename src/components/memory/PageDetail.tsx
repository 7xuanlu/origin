// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import { useQuery, useQueries, useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  getPage,
  getPageLinks,
  getPageRevisions,
  getEntityDetail,
  redistillPage,
  updatePage,
  deletePage,
  clipboardWrite,
  exportPageToObsidian,
  listRegisteredSources,
  getPageSources,
  type Entity,
  type Page,
} from "../../lib/tauri";
import ContentRenderer from "./ContentRenderer";
import RelatedPages from "./page/RelatedPages";
import PageInfo from "./page/PageInfo";
import { RailPanelTitle } from "./MemoryDetailPrimitives";
import { processCitations, stripCitationLinks } from "../../lib/pageCitations";
import CitationChip from "./page/CitationChip";

interface PageDetailProps {
  pageId: string;
  onBack: () => void;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
  onEntityClick?: (entityId: string) => void;
  onDismissAttachedPageNotice?: () => void;
  onPageLoaded?: (page: Pick<Page, "id" | "status" | "title">) => void;
  showAttachedPageNotice?: boolean;
}

function relativeTimeFromISO(iso: string, t: TFunction): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return t("pageDetail.dateline.relativeJustNow");
  if (diff < 3600) {
    return t("pageDetail.dateline.relativeMinutesAgo", {
      count: Math.floor(diff / 60),
    });
  }
  if (diff < 86400) {
    return t("pageDetail.dateline.relativeHoursAgo", {
      count: Math.floor(diff / 3600),
    });
  }
  return t("pageDetail.dateline.relativeDaysAgo", {
    count: Math.floor(diff / 86400),
  });
}

function normalizeLinkLabel(label: string): string {
  return label.trim().toLowerCase();
}

function parseWikilink(inner: string): { targetLabel: string; displayText: string } {
  const pipeIndex = inner.indexOf("|");
  const rawTarget = pipeIndex >= 0 ? inner.slice(0, pipeIndex) : inner;
  const headingIndex = rawTarget.indexOf("#");
  const targetLabel = (headingIndex >= 0 ? rawTarget.slice(0, headingIndex) : rawTarget).trim();
  const targetDisplay = targetLabel || rawTarget.trim();
  const alias = pipeIndex >= 0 ? inner.slice(pipeIndex + 1).trim() : "";
  return {
    targetLabel,
    displayText: alias || targetDisplay || inner.trim(),
  };
}

function folderName(path: string): string {
  return path.split("/").filter(Boolean).pop() || path;
}

const PAGE_LINK_ANCHOR_PREFIX = "#concept:";
type MenuInitialFocus = "first" | "last";

function enabledMenuItems(menu: HTMLDivElement | null): HTMLElement[] {
  if (!menu) return [];
  const items = Array.from(
    menu.querySelectorAll<HTMLElement>('[role="menuitem"]:not(:disabled)'),
  );
  const renderedItems = items.filter((item) => item.getClientRects().length > 0);
  return renderedItems.length > 0 ? renderedItems : items;
}

function focusMenuBoundary(
  menu: HTMLDivElement | null,
  boundary: MenuInitialFocus,
): void {
  const items = enabledMenuItems(menu);
  items[boundary === "first" ? 0 : items.length - 1]?.focus();
}

function handleMenuKeyDown(
  event: React.KeyboardEvent<HTMLDivElement>,
  menu: HTMLDivElement | null,
  closeMenu: () => void,
  trigger: HTMLButtonElement | null,
): void {
  if (!["ArrowDown", "ArrowUp", "Home", "End", "Escape"].includes(event.key)) {
    return;
  }

  event.preventDefault();
  event.stopPropagation();

  if (event.key === "Escape") {
    closeMenu();
    trigger?.focus();
    return;
  }

  const items = enabledMenuItems(menu);
  if (items.length === 0) return;

  const currentIndex = items.indexOf(document.activeElement as HTMLElement);
  let nextIndex = 0;
  if (event.key === "End") {
    nextIndex = items.length - 1;
  } else if (event.key === "ArrowDown") {
    nextIndex = currentIndex < 0 ? 0 : (currentIndex + 1) % items.length;
  } else if (event.key === "ArrowUp") {
    nextIndex = currentIndex < 0
      ? items.length - 1
      : (currentIndex - 1 + items.length) % items.length;
  }
  items[nextIndex]?.focus();
}

export default function PageDetail({
  pageId,
  onBack,
  onMemoryClick,
  onPageClick,
  onEntityClick,
  onDismissAttachedPageNotice,
  onPageLoaded,
  showAttachedPageNotice = false,
}: PageDetailProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [copied, setCopied] = useState(false);
  const [exported, setExported] = useState(false);
  const [copying, setCopying] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  const [redistillNotice, setRedistillNotice] = useState<{
    kind: "success" | "warning" | "error";
    message: string;
  } | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const exportMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const exportMenuInitialFocusRef = useRef<MenuInitialFocus>("first");
  const actionMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const actionMenuRef = useRef<HTMLDivElement>(null);
  const actionMenuListRef = useRef<HTMLDivElement>(null);
  const actionMenuInitialFocusRef = useRef<MenuInitialFocus>("first");
  const activePageIdRef = useRef(pageId);
  const saveInFlightRef = useRef(false);
  activePageIdRef.current = pageId;

  const {
    data: page,
    isLoading,
    isLoadingError: pageLoadFailed,
    isFetching: pageIsFetching,
    refetch: refetchPage,
  } = useQuery({
    queryKey: ["page", pageId],
    queryFn: () => getPage(pageId),
  });

  useEffect(() => {
    if (page == null) return;
    onPageLoaded?.({ id: page.id, status: page.status, title: page.title });
  }, [onPageLoaded, page?.id, page?.status, page?.title]);

  const { data: pageLinks } = useQuery({
    queryKey: ["page-links", pageId],
    queryFn: () => getPageLinks(pageId),
    enabled: !!pageId,
    staleTime: 30_000,
    retry: false,
  });

  const { data: pageRevisions } = useQuery({
    queryKey: ["page-revisions", pageId],
    queryFn: () => getPageRevisions(pageId),
    enabled: !!pageId,
    staleTime: 30_000,
    retry: false,
  });

  const outboundTargetByLabel = useMemo(() => {
    const map = new Map<string, string>();
    for (const link of pageLinks?.outbound ?? []) {
      if (link.target_page_id) {
        map.set(normalizeLinkLabel(link.label), link.target_page_id);
      }
    }
    return map;
  }, [pageLinks]);

  const { data: registeredSources = [] } = useQuery({
    queryKey: ["registeredSources"],
    queryFn: () => listRegisteredSources(),
    staleTime: 30000,
  });

  const obsidianSources = useMemo(
    () => registeredSources.filter((s) => s.source_type === "obsidian"),
    [registeredSources],
  );

  const { data: pageSources } = useQuery({
    queryKey: ["page-sources", pageId],
    queryFn: () => getPageSources(pageId),
    enabled: !!pageId,
  });

  // Entities on this page = the page's own anchor entity plus the anchor
  // entities of its source memories. These are enrichment links, not search.
  const pageEntityIds = useMemo(() => {
    const ids = new Set<string>();
    if (page?.entity_id) ids.add(page.entity_id);
    for (const s of pageSources ?? []) {
      if (s.memory?.entity_id) ids.add(s.memory.entity_id);
    }
    return [...ids];
  }, [page?.entity_id, pageSources]);

  const entityQueries = useQueries({
    queries: pageEntityIds.map((id) => ({
      queryKey: ["entityDetail", id],
      queryFn: () => getEntityDetail(id),
      staleTime: 60_000,
      retry: false,
    })),
  });
  const pageEntities = entityQueries
    .map((q) => q.data?.entity)
    .filter((e): e is Entity => !!e);

  useEffect(() => {
    setRedistillNotice(null);
    setActionError(null);
    setCopied(false);
    setExported(false);
    setEditing(false);
    setEditContent("");
  }, [pageId]);

  const updateMutation = useMutation({
    mutationFn: ({ id, content }: { id: string; content: string }) =>
      updatePage(id, content),
    onSuccess: (_result, { id }) => {
      queryClient.invalidateQueries({ queryKey: ["page", id] });
      queryClient.invalidateQueries({ queryKey: ["pages"] });
      queryClient.invalidateQueries({ queryKey: ["page-links", id] });
      queryClient.invalidateQueries({ queryKey: ["page-revisions", id] });
      if (activePageIdRef.current !== id) return;
      setActionError(null);
      setEditing(false);
    },
    onError: (_error, { id }) => {
      if (activePageIdRef.current !== id) return;
      setActionError(t("pageDetail.saveError"));
    },
    onSettled: () => {
      saveInFlightRef.current = false;
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deletePage(id),
    onSuccess: (_result, id) => {
      queryClient.invalidateQueries({ queryKey: ["page", id] });
      queryClient.invalidateQueries({ queryKey: ["pages"] });
      queryClient.invalidateQueries({ queryKey: ["page-links", id] });
      queryClient.invalidateQueries({ queryKey: ["page-revisions", id] });
      queryClient.invalidateQueries({ queryKey: ["page-sources", id] });
      if (activePageIdRef.current !== id) return;
      setActionError(null);
      onBack();
    },
    onError: (_error, id) => {
      if (activePageIdRef.current !== id) return;
      setActionError(t("pageDetail.deleteError"));
    },
  });

  const redistillMutation = useMutation({
    mutationFn: (id: string) => redistillPage(id),
    onSuccess: (result, id) => {
      queryClient.invalidateQueries({ queryKey: ["page", id] });
      queryClient.invalidateQueries({ queryKey: ["pages"] });
      queryClient.invalidateQueries({ queryKey: ["page-links", id] });
      queryClient.invalidateQueries({ queryKey: ["page-revisions", id] });
      queryClient.invalidateQueries({ queryKey: ["page-sources", id] });
      if (activePageIdRef.current !== id) return;
      if (result.status === "skipped") {
        setRedistillNotice({
          kind: "warning",
          message: result.hint || "Page re-distill skipped.",
        });
        return;
      }
      setRedistillNotice({
        kind: "success",
        message: result.updated ? "Page re-distilled." : "Page already up to date.",
      });
    },
    onError: (error, id) => {
      if (activePageIdRef.current !== id) return;
      setRedistillNotice({
        kind: "error",
        message: error instanceof Error ? error.message : "Page re-distill failed.",
      });
    },
  });

  const pageHasUserEdits = Boolean(page?.user_edited || pageRevisions?.user_edited);
  const handleRedistillClick = () => {
    if (
      pageHasUserEdits &&
      !confirm("Re-distill this edited page? The current version stays in page history for recovery.")
    ) {
      return;
    }
    redistillMutation.mutate(pageId);
  };

  const copyAsContext = useCallback(async () => {
    if (!page) return;
    const originPageId = page.id;
    const space = page.domain ? `**Space:** ${page.domain}` : "";
    const version = `**Version:** ${page.version}`;
    const compiled = `**Last compiled:** ${page.last_compiled}`;
    const meta = [space, version, compiled].filter(Boolean).join("\n");
    const text = [
      `## ${page.title}`,
      meta,
      "",
      page.content,
    ].join("\n");
    setActionError(null);
    setCopying(true);
    try {
      await clipboardWrite(text);
      if (activePageIdRef.current !== originPageId) return;
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      if (activePageIdRef.current !== originPageId) return;
      setActionError(t("pageDetail.copyError"));
    } finally {
      setCopying(false);
    }
  }, [page, t]);

  const handleExportToVault = useCallback(
    async (vaultPath: string) => {
      const originPageId = pageId;
      setExportMenuOpen(false);
      setActionError(null);
      setExporting(true);
      try {
        await exportPageToObsidian(originPageId, `${vaultPath}/Wenlan/pages`);
        if (activePageIdRef.current !== originPageId) return;
        setExported(true);
        setTimeout(() => setExported(false), 2000);
      } catch {
        if (activePageIdRef.current !== originPageId) return;
        setActionError(t("pageDetail.exportError"));
      } finally {
        setExporting(false);
      }
    },
    [pageId, t],
  );

  const handleSave = () => {
    if (saveInFlightRef.current || updateMutation.isPending) return;
    if (editContent.trim() && editContent !== page?.content) {
      setActionError(null);
      saveInFlightRef.current = true;
      updateMutation.mutate({ id: pageId, content: editContent.trim() });
    } else {
      setEditing(false);
    }
  };

  const beginEditing = () => {
    setActionError(null);
    setEditContent(page?.content ?? "");
    setEditing(true);
    setActionMenuOpen(false);
  };

  const requestDelete = () => {
    setActionMenuOpen(false);
    if (confirm(t("pageDetail.deleteConfirm"))) {
      setActionError(null);
      deleteMutation.mutate(pageId);
    }
  };

  const openExportMenu = (initialFocus: MenuInitialFocus) => {
    exportMenuInitialFocusRef.current = initialFocus;
    setActionMenuOpen(false);
    setExportMenuOpen(true);
  };

  const openActionMenu = (initialFocus: MenuInitialFocus) => {
    actionMenuInitialFocusRef.current = initialFocus;
    setExportMenuOpen(false);
    setActionMenuOpen(true);
  };

  const handleMenuTriggerKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    openMenu: (initialFocus: MenuInitialFocus) => void,
  ) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    event.stopPropagation();
    openMenu(event.key === "ArrowDown" ? "first" : "last");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === "Escape") {
      setEditing(false);
    }
  };

  // Focus textarea on edit
  useEffect(() => {
    if (editing && textareaRef.current) {
      textareaRef.current.focus();
      textareaRef.current.selectionStart = textareaRef.current.value.length;
    }
  }, [editing]);

  useEffect(() => {
    if (!exportMenuOpen) return;
    focusMenuBoundary(exportMenuRef.current, exportMenuInitialFocusRef.current);
  }, [exportMenuOpen]);

  useEffect(() => {
    if (!actionMenuOpen) return;
    focusMenuBoundary(actionMenuListRef.current, actionMenuInitialFocusRef.current);
  }, [actionMenuOpen]);

  useEffect(() => {
    if (!actionMenuOpen) return;
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!actionMenuRef.current?.contains(event.target as Node)) {
        setActionMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
    };
  }, [actionMenuOpen]);

  if (isLoading) return null;

  if (pageLoadFailed) {
    return (
      <div className="page-detail-load-state">
        <p role="alert">{t("pageDetail.loadError")}</p>
        <button
          disabled={pageIsFetching}
          onClick={() => void refetchPage()}
          type="button"
        >
          {t("pageDetail.retry")}
        </button>
      </div>
    );
  }

  if (!page) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-20">
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          Page not found
        </span>
        <button
          onClick={onBack}
          className="transition-colors text-sm"
          style={{ color: "var(--mem-text-secondary)" }}
        >
          Back
        </button>
      </div>
    );
  }

  const sourceCount = pageSources?.length ?? page.source_memory_ids.length;

  // Citations first (occurrence counting mirrors the backend and runs over the
  // raw stored body), then the existing display transforms.
  const processed = processCitations(page.content, page.citations);

  // Strip ## Sources (shown in Page info below)
  // Convert [[wikilinks]] to markdown links if they resolve to pages, else plain text
  const cleanedContent = processed.content
    .replace(/^#\s+.*\n+/, "") // Strip title heading (displayed separately by UI)
    .replace(/## Sources\n[\s\S]*?(?=\n## |\s*$)/, "")
    .replace(/\[\[([^\]]+)\]\]/g, (_match, inner) => {
      const link = parseWikilink(inner);
      const cid = outboundTargetByLabel.get(normalizeLinkLabel(link.targetLabel));
      if (cid) return `[${link.displayText}](${PAGE_LINK_ANCHOR_PREFIX}${cid})`;
      return link.displayText;
    })
    .trim();

  const sourceMemoryByLocator = new Map(
    (pageSources ?? [])
      .filter((cs) => cs.memory !== null)
      .map((cs) => [cs.source.memory_source_id, cs.memory]),
  );

  // Extract TLDR (first sentence) for native rendering under title.
  // Match first sentence ending with ". " or ".\n" — but not inside [[wikilinks]] or after abbreviations.
  // Scan starts after any leading heading lines so a body that opens with
  // "## Section" doesn't leak raw markdown into the plain-text lede.
  const leadingHeadings = cleanedContent.match(/^(?:#{1,6}[^\n]*\n+)+/)?.[0].length ?? 0;
  const bodyAfterHeadings = cleanedContent.slice(leadingHeadings);
  const sentenceEnd = bodyAfterHeadings.search(/\.\s/);
  const tldr = sentenceEnd > 0 && sentenceEnd < 400
    ? stripCitationLinks(bodyAfterHeadings.slice(0, sentenceEnd + 1).trim())
    : "";
  const displayContent = tldr
    ? (cleanedContent.slice(0, leadingHeadings) + bodyAfterHeadings.slice(sentenceEnd + 1).trimStart()).trim()
    : cleanedContent;

  // Intercept page/memory link clicks in rendered content (capture phase beats target="_blank")
  const handleContentClick = (e: React.MouseEvent) => {
    const anchor = (e.target as HTMLElement).closest("a");
    if (!anchor) return;
    const href = anchor.getAttribute("href") || "";
    if (href.startsWith(PAGE_LINK_ANCHOR_PREFIX)) {
      e.preventDefault();
      e.stopPropagation();
      onPageClick?.(href.replace(PAGE_LINK_ANCHOR_PREFIX, ""));
    } else if (href.startsWith("#memory:")) {
      e.preventDefault();
      e.stopPropagation();
      onMemoryClick(href.replace("#memory:", ""));
    }
  };

  const outboundLinks = pageLinks?.outbound ?? [];
  const inboundLinks = pageLinks?.inbound ?? [];
  const pageRevisionEntries = pageRevisions?.entries ?? [];

  const hasRail = pageEntities.length > 0 || outboundLinks.length > 0;

  return (
    <div className="page-detail">
      {/* Back + Header */}
      <div>
        <button
          aria-label={t("main.back")}
          onClick={onBack}
          className="mem-icon-action -ml-1.5"
          style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}
          type="button"
        >
          <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>

        <div className="page-detail-heading-row flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="page-detail-title">
              {page.title}
            </h1>
            <div className="page-detail-dateline">
              <span className="page-detail-dateline-item">
                {t("pageDetail.dateline.lastDistilled", {
                  time: relativeTimeFromISO(page.last_compiled, t),
                })}
              </span>
              <span className="page-detail-dateline-item">
                {t("pageDetail.dateline.sourceMemories", { count: sourceCount })}
              </span>
              {page.stale_reason && (
                <span
                  className="page-detail-dateline-item"
                  style={{
                    color:
                      page.stale_reason === "source_conflict"
                        ? "var(--mem-accent-amber)"
                        : "var(--mem-text-tertiary)",
                  }}
                >
                  {page.stale_reason === "source_conflict"
                    ? t("pageDetail.dateline.needsReview")
                    : t("pageDetail.dateline.updating")}
                </span>
              )}
            </div>
          </div>

          <div className="page-detail-header-actions">
            {!editing ? (
              <button
                type="button"
                className="page-detail-primary-action"
                onClick={beginEditing}
              >
                {t("pageDetail.editPage")}
              </button>
            ) : null}
            <div className="page-detail-icon-actions">
            {!editing && (
              <button
                aria-label={t("pageDetail.editPage")}
                onClick={beginEditing}
                className="mem-icon-action"
                title={t("pageDetail.editPage")}
                type="button"
              >
                <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
              </button>
            )}
            {!editing && (
              <button
                onClick={handleRedistillClick}
                disabled={redistillMutation.isPending}
                className="mem-icon-action"
                aria-label={
                  redistillMutation.isPending
                    ? t("pageDetail.redistillingPage")
                    : t("pageDetail.redistillPage")
                }
                title={
                  redistillMutation.isPending
                    ? t("pageDetail.redistillingPage")
                    : t("pageDetail.redistillPage")
                }
                type="button"
              >
                <svg
                  aria-hidden="true"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M21 12a9 9 0 11-2.64-6.36" />
                  <path d="M21 3v6h-6" />
                </svg>
              </button>
            )}
            <button
              onClick={copyAsContext}
              disabled={copying}
              className={`mem-icon-action ${copied ? "text-emerald-400" : ""}`}
              title={copied ? t("pageDetail.copied") : t("pageDetail.copyAsContext")}
              aria-label={copied ? t("pageDetail.copied") : t("pageDetail.copyAsContext")}
              type="button"
            >
              {copied ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
                </svg>
              )}
            </button>
            {/* Export button: 0 sources = disabled, 1 = direct, 2+ = popover */}
            <div className="relative">
              {obsidianSources.length === 0 ? (
                <button
                  disabled
                  className="mem-icon-action"
                  title={t("pageDetail.exportUnavailable")}
                  aria-label={t("pageDetail.exportUnavailable")}
                  type="button"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </button>
              ) : (
                <button
                  ref={exportMenuTriggerRef}
                  aria-expanded={obsidianSources.length >= 2 ? exportMenuOpen : undefined}
                  aria-haspopup={obsidianSources.length >= 2 ? "menu" : undefined}
                  onKeyDown={(event) => {
                    if (obsidianSources.length >= 2) {
                      handleMenuTriggerKeyDown(event, openExportMenu);
                    }
                  }}
                  onClick={() => {
                    if (obsidianSources.length === 1) {
                      handleExportToVault(obsidianSources[0].path);
                    } else if (exportMenuOpen) {
                      setExportMenuOpen(false);
                    } else {
                      openExportMenu("first");
                    }
                  }}
                  disabled={exporting}
                  className={`mem-icon-action ${exported ? "text-emerald-400" : ""}`}
                  title={exported ? t("pageDetail.exported") : t("pageDetail.exportToObsidian")}
                  aria-label={exported ? t("pageDetail.exported") : t("pageDetail.exportToObsidian")}
                  type="button"
                >
                  {exported ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                  )}
                </button>
              )}
              {exportMenuOpen && obsidianSources.length >= 2 && (
                <div
                  className="mem-popover-surface page-detail-export-menu absolute right-0 top-full mt-1 z-50"
                  onKeyDown={(event) => {
                    handleMenuKeyDown(
                      event,
                      exportMenuRef.current,
                      () => setExportMenuOpen(false),
                      exportMenuTriggerRef.current,
                    );
                  }}
                  ref={exportMenuRef}
                  role="menu"
                >
                  {obsidianSources.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleExportToVault(s.path)}
                      role="menuitem"
                      type="button"
                    >
                      {folderName(s.path)}
                    </button>
                  ))}
                </div>
              )}
            </div>
            </div>
            <div className="page-detail-actions-anchor" ref={actionMenuRef}>
              <button
                ref={actionMenuTriggerRef}
                type="button"
                className="mem-icon-action page-detail-actions-menu-trigger"
                aria-expanded={actionMenuOpen}
                aria-haspopup="menu"
                aria-label={t("pageDetail.actions")}
                title={t("pageDetail.actions")}
                onKeyDown={(event) => handleMenuTriggerKeyDown(event, openActionMenu)}
                onClick={() => {
                  if (actionMenuOpen) {
                    setActionMenuOpen(false);
                  } else {
                    openActionMenu("first");
                  }
                }}
              >
                <svg aria-hidden="true" width="16" height="4" viewBox="0 0 16 4" fill="currentColor">
                  <circle cx="2" cy="2" r="1.5" />
                  <circle cx="8" cy="2" r="1.5" />
                  <circle cx="14" cy="2" r="1.5" />
                </svg>
              </button>
              {actionMenuOpen ? (
                <div
                  aria-label={t("pageDetail.actions")}
                  className="mem-popover-surface page-detail-actions-menu"
                  onKeyDown={(event) => {
                    handleMenuKeyDown(
                      event,
                      actionMenuListRef.current,
                      () => setActionMenuOpen(false),
                      actionMenuTriggerRef.current,
                    );
                  }}
                  ref={actionMenuListRef}
                  role="menu"
                >
                  {!editing ? (
                    <button
                      className="page-detail-mobile-menu-item"
                      disabled={redistillMutation.isPending}
                      onClick={() => {
                        setActionMenuOpen(false);
                        handleRedistillClick();
                      }}
                      role="menuitem"
                      type="button"
                    >
                      {redistillMutation.isPending
                        ? t("pageDetail.redistillingPage")
                        : t("pageDetail.redistillPage")}
                    </button>
                  ) : null}
                  <button
                    className="page-detail-mobile-menu-item"
                    disabled={copying}
                    onClick={() => {
                      setActionMenuOpen(false);
                      void copyAsContext();
                    }}
                    role="menuitem"
                    type="button"
                  >
                    {copied ? t("pageDetail.copied") : t("pageDetail.copyAsContext")}
                  </button>
                  {obsidianSources.length === 0 ? (
                    <button
                      className="page-detail-mobile-menu-item"
                      disabled
                      role="menuitem"
                      type="button"
                    >
                      {t("pageDetail.exportToObsidian")}
                    </button>
                  ) : (
                    obsidianSources.map((source) => (
                      <button
                        className="page-detail-mobile-menu-item"
                        disabled={exporting}
                        key={source.id}
                        onClick={() => {
                          setActionMenuOpen(false);
                          void handleExportToVault(source.path);
                        }}
                        role="menuitem"
                        type="button"
                      >
                        {obsidianSources.length === 1
                          ? t("pageDetail.exportToObsidian")
                          : t("pageDetail.exportToVault", { vault: folderName(source.path) })}
                      </button>
                    ))
                  )}
                  <button
                    className="page-detail-menu-danger"
                    disabled={deleteMutation.isPending}
                    onClick={requestDelete}
                    role="menuitem"
                    type="button"
                  >
                    {t("pageDetail.deletePage")}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {showAttachedPageNotice && (
        <div
          aria-label={t("pages.composer.attachedNotice", { title: page.title })}
          aria-live="polite"
          className="page-detail-attached-notice"
          role="status"
        >
          <span>{t("pages.composer.attachedNotice", { title: page.title })}</span>
          <button onClick={onDismissAttachedPageNotice} type="button">
            {t("pages.composer.dismissNotice")}
          </button>
        </div>
      )}

      {redistillNotice && (
        <div
          role="status"
          aria-live="polite"
          className="rounded-lg px-3 py-2"
          style={{
            backgroundColor:
              redistillNotice.kind === "error"
                ? "rgba(239, 68, 68, 0.08)"
                : redistillNotice.kind === "warning"
                  ? "rgba(245, 158, 11, 0.08)"
                  : "rgba(16, 185, 129, 0.08)",
            border: "1px solid var(--mem-border)",
            color:
              redistillNotice.kind === "error"
                ? "#ef4444"
                : redistillNotice.kind === "warning"
                  ? "var(--mem-accent-amber)"
                  : "var(--mem-text-secondary)",
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            lineHeight: "1.5",
          }}
        >
          {redistillNotice.message}
        </div>
      )}

      {actionError && (
        <div
          aria-live="assertive"
          className="page-detail-action-error"
          role="alert"
        >
          {actionError}
        </div>
      )}

      {/* Content — edit mode or rendered */}
      {editing ? (
        <div className="flex flex-col gap-2">
          <textarea
            ref={textareaRef}
            disabled={updateMutation.isPending}
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            onKeyDown={handleKeyDown}
            className="w-full rounded-lg p-4 resize-y outline-none"
            style={{
              minHeight: "300px",
              backgroundColor: "var(--mem-surface)",
              border: "1px solid var(--mem-border)",
              color: "var(--mem-text)",
              fontFamily: "var(--mem-font-mono)",
              fontSize: "13px",
              lineHeight: "1.6",
            }}
          />
          <div className="flex items-center gap-2">
            <button
              onClick={handleSave}
              disabled={updateMutation.isPending}
              className="text-[11px] font-medium px-3 py-1.5 rounded-md transition-all"
              style={{ backgroundColor: "rgba(99, 102, 241, 0.15)", color: "var(--mem-accent-page)" }}
            >
              {updateMutation.isPending ? "Saving..." : "Save (Cmd+Enter)"}
            </button>
            <button
              onClick={() => setEditing(false)}
              className="text-[11px] font-medium px-3 py-1.5 rounded-md transition-all hover:bg-[var(--mem-hover-strong)]"
              style={{ color: "var(--mem-text-tertiary)" }}
            >
              Cancel (Esc)
            </button>
          </div>
        </div>
      ) : (
        <div className={hasRail ? "page-detail-grid" : undefined}>
          <div className="page-detail-prose" onClickCapture={handleContentClick}>
            {(page.summary || tldr) && (
              <div className="page-detail-lede">
                <p>{page.summary || tldr}</p>
              </div>
            )}
            <ContentRenderer
              content={displayContent}
              variant="detail"
              renderCitation={(k) => {
                const c = processed.byOccurrence.get(k);
                if (!c) return null;
                return (
                  <CitationChip
                    occurrence={k}
                    citation={c}
                    sourceMemory={sourceMemoryByLocator.get(c.locator) ?? null}
                    sourcesLoading={pageSources === undefined}
                    onOpenMemory={onMemoryClick}
                  />
                );
              }}
            />
          </div>
          {hasRail && (
            <aside className="memory-detail-rail page-detail-rail">
              {pageEntities.length > 0 && (
                <section className="memory-detail-rail-section">
                  <RailPanelTitle>{t("pageDetail.entities")}</RailPanelTitle>
                  <div className="memory-detail-entity-chip-list">
                    {pageEntities.map((e) => (
                      <button
                        key={e.id}
                        type="button"
                        onClick={() => onEntityClick?.(e.id)}
                        className="memory-detail-entity-chip"
                      >
                        <span className="memory-detail-entity-name">{e.name}</span>
                        <span className="memory-detail-entity-type">{e.entity_type}</span>
                      </button>
                    ))}
                  </div>
                </section>
              )}
              <RelatedPages outbound={outboundLinks} onPageClick={onPageClick} />
            </aside>
          )}
        </div>
      )}

      {!editing && (
        <PageInfo
          sourceCount={sourceCount}
          sources={pageSources}
          inbound={inboundLinks}
          revisions={pageRevisionEntries}
          citations={page.citations}
          citationState={processed.state}
          onMemoryClick={onMemoryClick}
          onPageClick={onPageClick}
        />
      )}
    </div>
  );
}
