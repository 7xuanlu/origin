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
  getDaemonVersion,
  getSystemInfo,
  daemonMeetsFloor,
  recordPageEditorDiagnostic,
  deletePage,
  clipboardWrite,
  exportPageToObsidian,
  listRegisteredSources,
  getPageSources,
  reviewPage,
  pageReviewSupported,
  type Entity,
  type Page,
  type PageReviewOutcome,
  type UpdatePageInput,
  type UpdatePageFailureKind,
} from "../../lib/tauri";
// Imported from its own module, not the lib/tauri barrel: tests mock that
// barrel, and a mocked module would shadow the value this must stay bound to.
import { PAGE_EDIT_DAEMON_FLOOR } from "../../lib/daemonVersion";
import { EXPLICIT_BROWSE_QUERY_POLICY, useTruthStatus } from "../../hooks/useTruthStatus";
import { PageTruthBadges } from "./PageTruthBadges";
import ContentRenderer from "./ContentRenderer";
import RelatedPages from "./page/RelatedPages";
import PageInfo from "./page/PageInfo";
import { pageReviewNotice, type PageReviewNotice } from "./page/pageReviewNotice";
import { RailPanelTitle } from "./MemoryDetailPrimitives";
import { processCitations, stripCitationLinks } from "../../lib/pageCitations";
import { stripLedeLabel } from "../../lib/pageLede";
import CitationChip from "./page/CitationChip";
import PageCanvas from "./PageCanvas";
import {
  prepareMarkdownSource,
  serializeMarkdownSource,
  type MarkdownSourceProfile,
  type PreparedMarkdownSource,
} from "./editor/markdownSourceContract";
import {
  beginPageSave,
  pageDraftChanged,
  pageVersionChanged,
  settlePageSave,
  type PageEditBaseline,
  type PageSaveCoordinatorState,
} from "./editor/pageSaveCoordinator";
import {
  MarkdownEditor,
  type MarkdownEditorHandle,
  type MarkdownEditorStatus,
} from "./editor/MarkdownEditor";
import {
  MarkdownEditorToolbar,
  type MarkdownEditorToolbarLabels,
} from "./editor/MarkdownEditorToolbar";
import { leadingMarkdownH1MatchesTitle } from "./editor/pageEditorPresentation";

interface PageDetailProps {
  pageId: string;
  onBack: () => void;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
  onEntityClick?: (entityId: string) => void;
  onDismissAttachedPageNotice?: () => void;
  onPageLoaded?: (page: Pick<Page, "id" | "status" | "title">) => void;
  onSavePendingChange?: (pending: boolean) => void;
  onEditDirtyChange?: (dirty: boolean) => void;
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

function createPageOperationId(): string {
  return globalThis.crypto.randomUUID();
}

function localShortcutModifier(): "Cmd" | "Ctrl" {
  if (typeof navigator === "undefined") return "Ctrl";
  return navigator.platform.toLowerCase().startsWith("mac") ? "Cmd" : "Ctrl";
}

const PAGE_LINK_ANCHOR_PREFIX = "#concept:";
type MenuInitialFocus = "first" | "last";

type PageEditGate =
  | { kind: "closed" }
  | { kind: "checking" }
  | { kind: "unsupported"; version: string | null }
  | { kind: "normalize"; prepared: PreparedMarkdownSource }
  | { kind: "editor" };

type ConflictLatestSource =
  | { kind: "idle" }
  | { kind: "loading"; operationId: string }
  | { kind: "loaded"; operationId: string; page: Page }
  | { kind: "error"; operationId: string };

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
  onSavePendingChange,
  onEditDirtyChange,
  showAttachedPageNotice = false,
}: PageDetailProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [copied, setCopied] = useState(false);
  const [exported, setExported] = useState(false);
  const [copying, setCopying] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editInitialDocument, setEditInitialDocument] = useState("");
  const [editHasMatchingTitle, setEditHasMatchingTitle] = useState(false);
  const [editGate, setEditGate] = useState<PageEditGate>({ kind: "closed" });
  const [editBaseline, setEditBaseline] = useState<PageEditBaseline | null>(null);
  const [sourceProfile, setSourceProfile] =
    useState<MarkdownSourceProfile | null>(null);
  const [saveState, setSaveState] = useState<PageSaveCoordinatorState>({
    phase: "idle",
  });
  const [conflictLatest, setConflictLatest] =
    useState<ConflictLatestSource>({ kind: "idle" });
  const [editValidation, setEditValidation] = useState<string | null>(null);
  const [editorFallbackSessionId, setEditorFallbackSessionId] =
    useState<string | null>(null);
  const [editorShortcutModifier, setEditorShortcutModifier] =
    useState<"Cmd" | "Ctrl">(localShortcutModifier);
  const [editorStatus, setEditorStatus] = useState<MarkdownEditorStatus>({
    sessionId: "",
    engine: "loading",
    ready: false,
    compositionActive: false,
    canUndo: false,
    canRedo: false,
  });
  const [editorSessionToken, setEditorSessionToken] = useState<{
    id: string;
    epoch: number;
  } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  const [redistillNotice, setRedistillNotice] = useState<{
    kind: "success" | "warning" | "error";
    message: string;
  } | null>(null);
  const [reviewNotice, setReviewNotice] = useState<PageReviewNotice | null>(null);
  // Reading is what a page is for, so it needs no control of its own — only
  // leaving for the canvas does. A two-tab row spent a whole band of the page
  // telling you that you were doing the obvious thing.
  const [canvasOpen, setCanvasOpen] = useState(false);
  // Editing happens in the reading column, so edit mode pins the view: nobody
  // gets to type into a page they cannot see.
  const showCanvas = canvasOpen && !editing;
  const editorRef = useRef<MarkdownEditorHandle>(null);
  const editDocumentRef = useRef("");
  const editDirtyRef = useRef(false);
  const editPageTitleRef = useRef("");
  const beginEditAttemptRef = useRef(0);
  const editorSessionEpochRef = useRef(0);
  const activeEditorSessionRef = useRef<{
    id: string;
    epoch: number;
  } | null>(null);
  const saveStateRef = useRef<PageSaveCoordinatorState>({ phase: "idle" });
  const exportMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const exportMenuRef = useRef<HTMLDivElement>(null);
  const exportMenuInitialFocusRef = useRef<MenuInitialFocus>("first");
  const actionMenuTriggerRef = useRef<HTMLButtonElement>(null);
  const actionMenuRef = useRef<HTMLDivElement>(null);
  const actionMenuListRef = useRef<HTMLDivElement>(null);
  const actionMenuInitialFocusRef = useRef<MenuInitialFocus>("first");
  const activePageIdRef = useRef(pageId);
  activePageIdRef.current = pageId;
  const editorSessionId = editBaseline
    ? `${editBaseline.pageId}:${editBaseline.version}`
    : null;
  const editorSessionEpoch = editorSessionToken?.epoch ?? 0;

  const {
    data: page,
    isLoading,
    isLoadingError: pageLoadFailed,
    isFetching: pageIsFetching,
    refetch: refetchPage,
  } = useQuery({
    queryKey: ["page", pageId],
    queryFn: () => getPage(pageId, "explicit"),
    ...EXPLICIT_BROWSE_QUERY_POLICY,
  });
  const { cutoverLive } = useTruthStatus();

  // Fails closed on purpose, exactly like `useDaemonVersion`: an unreachable
  // daemon leaves this undefined, and undefined must read as "not available"
  // rather than "unknown, offer it anyway". The backend answers with WHY it is
  // unavailable, because the two reasons send you to different places — one is
  // fixed by upgrading the daemon, the other cannot be fixed here at all.
  const { data: reviewAvailability } = useQuery({
    queryKey: ["page-review-supported"],
    queryFn: pageReviewSupported,
    staleTime: 60_000,
    retry: 1,
  });
  const reviewSupported = reviewAvailability === "ready";
  const reviewUnavailableReason =
    reviewAvailability === "platform_unsupported"
      ? t("pageDetail.reviewUnsupportedPlatform")
      : t("pageDetail.reviewUnsupported");

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

  // Canvas nodes arrive with label: null for memory/entity/page refs — the
  // daemon stores the reference, the client renders the backing object. Every
  // name below comes from a query this component already runs; the canvas
  // must not fetch anything of its own.
  const entitySignature = pageEntities
    .map((e) => `${e.id}\u0000${e.name}`)
    .join("|");
  const labelOverrides = useMemo(() => {
    const overrides = new Map<string, string>();
    if (page?.title) overrides.set(`page:${pageId}`, page.title);
    for (const cs of pageSources ?? []) {
      const memory = cs.memory;
      if (!memory) continue;
      const text = (memory.title || memory.summary || memory.content || "").trim();
      if (!text) continue;
      overrides.set(
        `memory:${memory.source_id}`,
        text.length > 64 ? `${text.slice(0, 64).trimEnd()}\u2026` : text,
      );
    }
    for (const entity of pageEntities) {
      overrides.set(`entity:${entity.id}`, entity.name);
    }
    return overrides;
    // pageEntities is rebuilt every render by useQueries; entitySignature is
    // the stable stand-in that actually tracks its content.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageId, page?.title, pageSources, entitySignature]);

  useEffect(() => {
    setRedistillNotice(null);
    setActionError(null);
    setCopied(false);
    setExported(false);
    setEditing(false);
    setEditGate({ kind: "closed" });
    setEditBaseline(null);
    setEditHasMatchingTitle(false);
    setSourceProfile(null);
    setEditValidation(null);
    setEditorFallbackSessionId(null);
    setEditorSessionToken(null);
    activeEditorSessionRef.current = null;
    beginEditAttemptRef.current += 1;
    editorSessionEpochRef.current += 1;
    editDocumentRef.current = "";
    editDirtyRef.current = false;
    editPageTitleRef.current = "";
    saveStateRef.current = { phase: "idle" };
    setSaveState({ phase: "idle" });
  }, [pageId]);

  const updateSaveState = useCallback((next: PageSaveCoordinatorState) => {
    saveStateRef.current = next;
    setSaveState(next);
  }, []);

  useEffect(() => {
    const blockPendingEscape = (event: KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        saveStateRef.current.phase === "pending"
      ) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    window.addEventListener("keydown", blockPendingEscape, true);
    return () => window.removeEventListener("keydown", blockPendingEscape, true);
  }, []);

  useEffect(() => {
    onSavePendingChange?.(saveState.phase === "pending");
  }, [onSavePendingChange, saveState.phase]);

  useEffect(
    () => () => {
      if (saveStateRef.current.phase !== "pending") {
        onSavePendingChange?.(false);
      }
    },
    [onSavePendingChange],
  );

  useEffect(() => {
    onEditDirtyChange?.(editDirtyRef.current);
    return () => onEditDirtyChange?.(false);
  }, [onEditDirtyChange]);

  const updateEditDirty = useCallback(
    (dirty: boolean) => {
      if (editDirtyRef.current === dirty) return;
      editDirtyRef.current = dirty;
      onEditDirtyChange?.(dirty);
    },
    [onEditDirtyChange],
  );

  const isCurrentSave = useCallback((input: UpdatePageInput) => {
    const current = saveStateRef.current;
    return (
      current.phase === "pending" &&
      current.pending.operationId === input.operationId &&
      current.pending.content === input.content &&
      current.pending.expectedVersion === input.expectedVersion
    );
  }, []);

  const fetchConflictLatest = useCallback(
    async (operationId: string): Promise<Page | null> => {
      setConflictLatest((currentLatest) =>
        currentLatest.kind === "loaded" &&
        currentLatest.operationId === operationId
          ? currentLatest
          : { kind: "loading", operationId },
      );
      try {
        const latest = await getPage(pageId, "explicit");
        const current = saveStateRef.current;
        if (
          current.phase !== "conflict" ||
          current.pending.operationId !== operationId
        ) {
          return null;
        }
        if (!latest) {
          setConflictLatest((currentLatest) =>
            currentLatest.kind === "loaded" &&
            currentLatest.operationId === operationId
              ? currentLatest
              : { kind: "error", operationId },
          );
          return null;
        }
        queryClient.setQueryData<Page | null>(["page", pageId], (currentPage) =>
          currentPage && currentPage.version >= latest.version
            ? currentPage
            : latest,
        );
        setConflictLatest((currentLatest) =>
          currentLatest.kind === "loaded" &&
          currentLatest.operationId === operationId &&
          currentLatest.page.version >= latest.version
            ? currentLatest
            : { kind: "loaded", operationId, page: latest },
        );
        return latest;
      } catch {
        const current = saveStateRef.current;
        if (
          current.phase === "conflict" &&
          current.pending.operationId === operationId
        ) {
          setConflictLatest((currentLatest) =>
            currentLatest.kind === "loaded" &&
            currentLatest.operationId === operationId
              ? currentLatest
              : { kind: "error", operationId },
          );
        }
        return null;
      }
    },
    [pageId, queryClient],
  );

  const updateMutation = useMutation({
    mutationFn: (input: UpdatePageInput) => updatePage(input),
    onSuccess: (outcome, input) => {
      if (!isCurrentSave(input)) return;
      const next = settlePageSave(saveStateRef.current, outcome);
      updateSaveState(next);
      if (outcome.outcome === "saved") {
        queryClient.invalidateQueries({ queryKey: ["page", input.id] });
        queryClient.invalidateQueries({ queryKey: ["pages"] });
        queryClient.invalidateQueries({ queryKey: ["page-links", input.id] });
        queryClient.invalidateQueries({ queryKey: ["page-revisions", input.id] });
        if (activePageIdRef.current !== input.id) return;
        setActionError(null);
        setEditing(false);
        setEditGate({ kind: "closed" });
        setEditBaseline(null);
        setSourceProfile(null);
        setEditValidation(null);
        setEditorFallbackSessionId(null);
        setEditorSessionToken(null);
        activeEditorSessionRef.current = null;
        editDocumentRef.current = "";
        updateEditDirty(false);
      } else if (outcome.outcome === "conflict") {
        void fetchConflictLatest(input.operationId);
      }
    },
    onError: (_error, input) => {
      if (!isCurrentSave(input)) return;
      updateSaveState(
        settlePageSave(saveStateRef.current, { outcome: "transport" }),
      );
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

  const reviewMutation = useMutation({
    // The content goes with the request because the mark is bound to the exact
    // text that was on screen, not to the page as a moving target. The backend
    // hashes what it is handed here; if the daemon's copy has since changed,
    // the answer is `stale` and the reader is asked to look again.
    mutationFn: ({ id, content }: { id: string; content: string }) => reviewPage(id, content),
    onSuccess: (outcome: PageReviewOutcome, { id }) => {
      if (outcome.kind === "applied") {
        queryClient.invalidateQueries({ queryKey: ["page", id] });
        // The wiki list renders the trust badges, so it is stale now too.
        queryClient.invalidateQueries({ queryKey: ["pages"] });
      }
      if (activePageIdRef.current !== id) return;
      setReviewNotice(pageReviewNotice(outcome, t));
    },
    onError: (_error, { id }) => {
      if (activePageIdRef.current !== id) return;
      setReviewNotice({
        kind: "error",
        message: t("pageDetail.reviewFailed"),
        offerReload: false,
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

  const closeEditor = () => {
    activeEditorSessionRef.current = null;
    beginEditAttemptRef.current += 1;
    editorSessionEpochRef.current += 1;
    editDocumentRef.current = "";
    editPageTitleRef.current = "";
    setEditing(false);
    setEditGate({ kind: "closed" });
    setEditBaseline(null);
    setEditHasMatchingTitle(false);
    setSourceProfile(null);
    setEditValidation(null);
    setEditorFallbackSessionId(null);
    setEditorSessionToken(null);
    setEditorStatus({
      sessionId: "",
      engine: "loading",
      ready: false,
      compositionActive: false,
      canUndo: false,
      canRedo: false,
    });
    setConflictLatest({ kind: "idle" });
    updateSaveState({ phase: "idle" });
    updateEditDirty(false);
  };

  const openPageSourceForEditing = (sourcePage: Page) => {
    if (activePageIdRef.current !== sourcePage.id) return;

    const prepared = prepareMarkdownSource(sourcePage.content);
    const sessionId = `${sourcePage.id}:${sourcePage.version}`;
    editorSessionEpochRef.current += 1;
    const sessionToken = {
      id: sessionId,
      epoch: editorSessionEpochRef.current,
    };
    activeEditorSessionRef.current = sessionToken;
    setEditorSessionToken(sessionToken);
    setEditorStatus({
      sessionId,
      engine: "loading",
      ready: false,
      compositionActive: false,
      canUndo: false,
      canRedo: false,
    });
    editDocumentRef.current = prepared.editorDocument;
    editPageTitleRef.current = sourcePage.title;
    updateEditDirty(false);
    setEditHasMatchingTitle(
      leadingMarkdownH1MatchesTitle(
        prepared.editorDocument,
        sourcePage.title,
      ),
    );
    setEditBaseline({
      pageId: sourcePage.id,
      content: sourcePage.content,
      version: sourcePage.version,
    });
    setEditValidation(null);
    setEditorFallbackSessionId(null);
    setConflictLatest({ kind: "idle" });
    updateSaveState({ phase: "idle" });
    if (!prepared.canEditLosslessly) {
      setSourceProfile(null);
      setEditGate({ kind: "normalize", prepared });
      return;
    }

    setEditInitialDocument(prepared.editorDocument);
    setSourceProfile(prepared.profile);
    setEditGate({ kind: "editor" });
  };

  const beginEditing = async () => {
    if (!page) return;
    const originPageId = page.id;
    const beginEditAttempt = ++beginEditAttemptRef.current;
    const isActiveBeginEdit = () =>
      activePageIdRef.current === originPageId &&
      beginEditAttemptRef.current === beginEditAttempt;

    setActionError(null);
    setEditing(true);
    setActionMenuOpen(false);
    setEditGate({ kind: "checking" });
    setEditValidation(null);
    updateSaveState({ phase: "idle" });

    let version: string;
    try {
      const [reportedVersion, systemInfo] = await Promise.all([
        getDaemonVersion(),
        getSystemInfo().catch(() => null),
      ]);
      if (!isActiveBeginEdit()) return;

      version = reportedVersion;
      if (systemInfo?.os) {
        setEditorShortcutModifier(
          systemInfo.os.toLowerCase() === "macos" ? "Cmd" : "Ctrl",
        );
      }
    } catch {
      if (!isActiveBeginEdit()) return;

      setEditGate({ kind: "unsupported", version: null });
      void recordPageEditorDiagnostic({
        event: "daemon_floor_blocked",
        reportedVersion: null,
        requiredFloor: PAGE_EDIT_DAEMON_FLOOR,
      }).catch(() => undefined);
      return;
    }
    if (!isActiveBeginEdit()) return;

    if (!daemonMeetsFloor(version, PAGE_EDIT_DAEMON_FLOOR)) {
      setEditGate({ kind: "unsupported", version });
      void recordPageEditorDiagnostic({
        event: "daemon_floor_blocked",
        reportedVersion: version,
        requiredFloor: PAGE_EDIT_DAEMON_FLOOR,
      }).catch(() => undefined);
      return;
    }

    if (!isActiveBeginEdit()) return;
    openPageSourceForEditing(page);
  };

  const handleNormalizeAndEdit = () => {
    if (editGate.kind !== "normalize") return;
    const normalizedProfile: MarkdownSourceProfile = {
      bom: editGate.prepared.profile.bom,
      lineEndings: { kind: "lf", separator: "\n" },
    };
    editDocumentRef.current = editGate.prepared.editorDocument;
    setEditInitialDocument(editGate.prepared.editorDocument);
    setSourceProfile(normalizedProfile);
    setEditGate({ kind: "editor" });
    updateEditDirty(
      editBaseline !== null &&
        serializeMarkdownSource(
          editGate.prepared.editorDocument,
          normalizedProfile,
        ) !== editBaseline.content,
    );
  };

  const saveDocument = (document: string) => {
    if (!editBaseline || !sourceProfile) return;

    editDocumentRef.current = document;
    const content = serializeMarkdownSource(document, sourceProfile);
    const attempt = beginPageSave(
      saveStateRef.current,
      editBaseline,
      content,
      createPageOperationId,
    );

    if (attempt.kind === "invalid") {
      setEditValidation(t("pageDetail.editor.empty"));
      return;
    }
    if (attempt.kind === "unchanged") {
      closeEditor();
      return;
    }
    if (attempt.kind !== "request") {
      return;
    }

    setEditValidation(null);
    updateSaveState(attempt.state);
    updateMutation.mutate(attempt.input);
  };

  const handleDocumentChange = (content: string) => {
    editDocumentRef.current = content;
    setEditHasMatchingTitle(
      leadingMarkdownH1MatchesTitle(content, editPageTitleRef.current),
    );
    setEditValidation(null);
    updateSaveState(pageDraftChanged(saveStateRef.current));
    updateEditDirty(
      editGate.kind === "editor" &&
        editBaseline !== null &&
        (sourceProfile
          ? serializeMarkdownSource(content, sourceProfile)
          : content) !== editBaseline.content,
    );
  };

  const currentDraftSource = () =>
    sourceProfile
      ? serializeMarkdownSource(editDocumentRef.current, sourceProfile)
      : editDocumentRef.current;

  const editorIsDirty = () =>
    editGate.kind === "editor" &&
    editBaseline !== null &&
    currentDraftSource() !== editBaseline.content;

  const editorToolbarLabels: MarkdownEditorToolbarLabels = {
    undo: t("pageDetail.editor.toolbar.undo"),
    redo: t("pageDetail.editor.toolbar.redo"),
    blockStyle: t("pageDetail.editor.toolbar.blockStyle"),
    paragraph: t("pageDetail.editor.toolbar.paragraph"),
    heading1: t("pageDetail.editor.toolbar.heading1"),
    heading2: t("pageDetail.editor.toolbar.heading2"),
    heading3: t("pageDetail.editor.toolbar.heading3"),
    bold: t("pageDetail.editor.toolbar.bold"),
    italic: t("pageDetail.editor.toolbar.italic"),
    inlineCode: t("pageDetail.editor.toolbar.inlineCode"),
    link: t("pageDetail.editor.toolbar.link"),
    blockquote: t("pageDetail.editor.toolbar.blockquote"),
    bulletList: t("pageDetail.editor.toolbar.bulletList"),
    numberedList: t("pageDetail.editor.toolbar.numberedList"),
    save:
      saveState.phase === "pending"
        ? t("pageDetail.editor.saving")
        : saveState.phase === "retryable"
          ? t("pageDetail.editor.retry")
          : t("pageDetail.editor.save"),
    cancel: t("pageDetail.editor.cancel"),
  };

  useEffect(() => {
    if (
      !editing ||
      editGate.kind !== "editor" ||
      !editBaseline ||
      !page ||
      page.id !== editBaseline.pageId
    ) {
      return;
    }

    const current = saveStateRef.current;
    const next = pageVersionChanged(
      current,
      editBaseline,
      currentDraftSource(),
      page.version,
      createPageOperationId,
    );
    if (next === current) {
      if (
        current.phase === "conflict" &&
        page.version > editBaseline.version
      ) {
        setConflictLatest((currentLatest) =>
          currentLatest.kind === "loaded" &&
          currentLatest.operationId === current.pending.operationId &&
          currentLatest.page.version >= page.version
            ? currentLatest
            : {
                kind: "loaded",
                operationId: current.pending.operationId,
                page,
              },
        );
      }
      return;
    }
    if (next.phase !== "conflict") return;

    updateSaveState(next);
    setConflictLatest({
      kind: "loaded",
      operationId: next.pending.operationId,
      page,
    });
  }, [
    editing,
    editGate.kind,
    editBaseline,
    page,
    saveState,
    sourceProfile,
    updateSaveState,
  ]);

  const requestCloseEditor = () => {
    if (saveStateRef.current.phase === "pending") return;
    if (
      editorIsDirty() &&
      !confirm(t("pageDetail.editor.discardConfirm"))
    ) {
      return;
    }
    closeEditor();
  };

  useEffect(() => {
    if (!editing) return;
    const captureUnfocusedEditorEscape = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.key !== "Escape"
      ) {
        return;
      }
      if (
        event.target instanceof Element &&
        event.target.closest(
          'input, textarea, select, [contenteditable]:not([contenteditable="false"])',
        )
      ) {
        return;
      }
      event.preventDefault();
      event.stopImmediatePropagation();
      requestCloseEditor();
    };
    window.addEventListener("keydown", captureUnfocusedEditorEscape, true);
    return () =>
      window.removeEventListener(
        "keydown",
        captureUnfocusedEditorEscape,
        true,
      );
  }, [editing, requestCloseEditor]);

  const requestBack = () => {
    if (saveStateRef.current.phase === "pending") return;
    if (
      editorIsDirty() &&
      !confirm(t("pageDetail.editor.discardConfirm"))
    ) {
      return;
    }
    if (editing) closeEditor();
    onBack();
  };

  const handleCopyDraft = async () => {
    await clipboardWrite(currentDraftSource());
  };

  const handleEditorFallback = (
    sessionId: string,
    sessionEpoch: number,
    reason: "load" | "construction",
  ) => {
    const active = activeEditorSessionRef.current;
    if (
      !active ||
      active.id !== sessionId ||
      active.epoch !== sessionEpoch
    ) {
      return;
    }
    setEditorFallbackSessionId(sessionId);
    void recordPageEditorDiagnostic({
      event: "editor_fallback",
      reason,
    }).catch(() => undefined);
  };

  const handleEditorStatus = (
    sessionId: string,
    sessionEpoch: number,
    status: MarkdownEditorStatus,
  ) => {
    const active = activeEditorSessionRef.current;
    if (
      !active ||
      active.id !== sessionId ||
      active.epoch !== sessionEpoch ||
      status.sessionId !== sessionId
    ) {
      return;
    }
    setEditorStatus(status);
  };

  const handleReloadLatest = async () => {
    if (saveStateRef.current.phase !== "conflict") return;
    if (
      editBaseline &&
      currentDraftSource() !== editBaseline.content &&
      !confirm(t("pageDetail.editor.reloadConfirm"))
    ) {
      return;
    }
    const operationId = saveStateRef.current.pending.operationId;
    const latest =
      conflictLatest.kind === "loaded" &&
      conflictLatest.operationId === operationId
        ? conflictLatest.page
        : await fetchConflictLatest(operationId);
    if (!latest) return;
    queryClient.setQueryData(["page", pageId], latest);
    openPageSourceForEditing(latest);
  };

  const failureMessage = (kind: UpdatePageFailureKind | "transport") => {
    switch (kind) {
      case "not_found":
        return t("pageDetail.editor.failure.notFound");
      case "auth_required":
        return t("pageDetail.editor.failure.authRequired");
      case "payload_too_large":
        return t("pageDetail.editor.failure.payloadTooLarge");
      case "validation":
        return t("pageDetail.editor.failure.validation");
      case "rate_limited":
        return t("pageDetail.editor.failure.rateLimited");
      case "server":
        return t("pageDetail.editor.failure.server");
      case "transport":
        return t("pageDetail.editor.failure.transport");
      case "other":
        return t("pageDetail.editor.failure.other");
    }
  };

  const requestDelete = () => {
    setActionMenuOpen(false);
    if (saveStateRef.current.phase === "pending") return;
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

  const handlePageDetailKeyDown = (e: React.KeyboardEvent) => {
    if (
      !editing ||
      e.defaultPrevented ||
      e.nativeEvent.isComposing ||
      e.key !== "Escape"
    ) {
      return;
    }
    if (
      e.target instanceof Element &&
      e.target.closest(
        'input, textarea, select, [contenteditable]:not([contenteditable="false"])',
      )
    ) {
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    requestCloseEditor();
  };

  useEffect(() => {
    if (editing && editGate.kind === "editor" && editorSessionId) {
      editorRef.current?.focus();
    }
  }, [editing, editGate.kind, editorSessionId]);

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

  // Extract the lede (first sentence) for native rendering under title.
  // Match first sentence ending with ". " or ".\n" — but not inside [[wikilinks]] or after abbreviations.
  // Scan starts after any leading heading lines so a body that opens with
  // "## Section" doesn't leak raw markdown into the plain-text lede.
  const leadingHeadings = cleanedContent.match(/^(?:#{1,6}[^\n]*\n+)+/)?.[0].length ?? 0;
  const bodyAfterHeadings = cleanedContent.slice(leadingHeadings);
  const sentenceEnd = bodyAfterHeadings.search(/\.\s/);
  // The first sentence in two forms: markdown, with its citation links intact
  // for the quote, and the plain text a reader sees. A leading "TLDR:" label
  // is not part of the sentence, so it comes off both. The length cap is
  // measured on the plain text, because the markers here have already been
  // rewritten to `[1](#citation:1)` links: a short sentence carrying a few of
  // them was pushed past the cap and dropped out of the lede, which left the
  // quote without its chips and repeated the sentence in the body.
  const firstSentenceMarkdown =
    sentenceEnd > 0 ? stripLedeLabel(bodyAfterHeadings.slice(0, sentenceEnd + 1).trim()) : "";
  const firstSentenceText = stripCitationLinks(firstSentenceMarkdown);
  const tldr =
    firstSentenceText.length > 0 && firstSentenceText.length < 400 ? firstSentenceText : "";
  // The lede shows the page summary when there is one; the first sentence is
  // cut from the body only when it is what the lede shows (no summary, or a
  // summary that repeats it). Cutting it under a different summary dropped
  // that sentence, and its citations, from the page.
  // Markers sit before the period ("... setup [1][2]."), so stripping them
  // leaves " ." behind; drop that space with the period before comparing.
  const normalizeSentence = (s: string) =>
    stripLedeLabel(s)
      .replace(/\[\d+\]/g, "").replace(/\s+/g, " ").trim().replace(/\s*\.$/, "").toLowerCase();
  const ledeText = page.summary ? stripLedeLabel(page.summary) : tldr;
  const ledeIsFirstSentence =
    !page.summary || (tldr !== "" && normalizeSentence(page.summary) === normalizeSentence(tldr));
  // When the lede is the first sentence, render that sentence with its
  // citation links so the chips move up with it instead of disappearing.
  const ledeMarkdown = tldr && ledeIsFirstSentence ? firstSentenceMarkdown : "";
  const displayContent = tldr && ledeIsFirstSentence
    ? (cleanedContent.slice(0, leadingHeadings) + bodyAfterHeadings.slice(sentenceEnd + 1).trimStart()).trim()
    : cleanedContent;

  const renderCitation = (k: number) => {
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
  };

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
  const hideOuterTitleWhileEditing =
    editing && editGate.kind === "editor" && editHasMatchingTitle;

  return (
    <div className="page-detail" onKeyDown={handlePageDetailKeyDown}>
      {/* Back + Header */}
      <div>
        <button
          aria-label={t("main.back")}
          onClick={requestBack}
          disabled={saveState.phase === "pending"}
          className="mem-icon-action -ml-1.5"
          style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0, marginBottom: "12px" }}
          type="button"
        >
          <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        </button>

        <div className="page-detail-heading-row flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1
              className={
                hideOuterTitleWhileEditing ? "sr-only" : "page-detail-title"
              }
            >
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
            <PageTruthBadges
              cutoverLive={cutoverLive}
              truth={page.truth}
              wrapperClassName="mt-2 flex flex-wrap items-center gap-2"
            />
          </div>

          {!editing && (
            <div className="page-detail-header-actions">
              <button
                type="button"
                className="page-detail-primary-action"
                onClick={beginEditing}
              >
                {t("pageDetail.editPage")}
              </button>
              <div className="page-detail-icon-actions">
                <button
                  aria-label={t("pageCanvas.tabCanvas")}
                  aria-pressed={showCanvas}
                  onClick={() => setCanvasOpen((open) => !open)}
                  className={`mem-icon-action page-detail-canvas-toggle${
                    showCanvas ? " is-active" : ""
                  }`}
                  title={t("pageCanvas.tabCanvas")}
                  type="button"
                >
                  {/* Boxes on a spine. Three dots joined by two diverging lines
                      is the share glyph — the same drawing Lucide ships as
                      share-2 — and on a page toolbar next to Copy and Export it
                      reads as Share, not as a map. Rectangles say "boxes", which
                      is what the canvas draws, and the sidebar's Graph keeps the
                      round nodes. */}
                  <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="1.5" y="9" width="7" height="6" rx="1.5" />
                    <rect x="15.5" y="2.5" width="7" height="6" rx="1.5" />
                    <rect x="15.5" y="15.5" width="7" height="6" rx="1.5" />
                    <path d="M8.5 12h3.5V5.5h3.5" />
                    <path d="M12 12v6.5h3.5" />
                  </svg>
                </button>
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
                  {/* Below 600px the icon row is hidden and this menu is the
                      whole toolbar, so the canvas needs a door here too — it
                      was reachable by icon only, which is no door at all on a
                      narrow window. */}
                  {!editing ? (
                    <button
                      className="page-detail-mobile-menu-item"
                      onClick={() => {
                        setActionMenuOpen(false);
                        setCanvasOpen((open) => !open);
                      }}
                      role="menuitem"
                      type="button"
                    >
                      {showCanvas
                        ? t("pageCanvas.closeCanvas")
                        : t("pageCanvas.tabCanvas")}
                    </button>
                  ) : null}
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
                  {/* Lives in the overflow menu at every width, like Delete,
                      rather than mirroring an icon-row button — this is a
                      low-frequency action that writes a durable record, and
                      the icon row is already full of things you reach for
                      constantly.

                      Visible but disabled before the daemon's truth cutover is
                      live, following the page editor's daemon-floor gate
                      rather than the hide-it convention used for provider
                      presets: this is an editorial action on the page in front
                      of you, and a control that silently disappears reads as a
                      feature that was taken away. The title says why.

                      Gone while editing, matching Canvas and Re-distill: the
                      mark attests the stored text, which is not what an open
                      editor is showing. (M5 App PR, D2/D7.) */}
                  {!editing ? (
                    <button
                      disabled={!reviewSupported || reviewMutation.isPending}
                      onClick={() => {
                        setActionMenuOpen(false);
                        reviewMutation.mutate({ id: pageId, content: page.content });
                      }}
                      role="menuitem"
                      title={
                        reviewSupported
                          ? t("pageDetail.markPageReviewed")
                          : reviewUnavailableReason
                      }
                      type="button"
                    >
                      {t("pageDetail.markPageReviewed")}
                    </button>
                  ) : null}
                  <button
                    className="page-detail-menu-danger"
                    disabled={
                      deleteMutation.isPending ||
                      saveState.phase === "pending"
                    }
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
          )}
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

      {reviewNotice && (
        <div
          role="status"
          aria-live="polite"
          className="rounded-lg px-3 py-2"
          data-testid="page-review-notice"
          style={{
            backgroundColor:
              reviewNotice.kind === "error"
                ? "rgba(239, 68, 68, 0.08)"
                : reviewNotice.kind === "warning"
                  ? "rgba(245, 158, 11, 0.08)"
                  : "rgba(16, 185, 129, 0.08)",
            border: "1px solid var(--mem-border)",
            color:
              reviewNotice.kind === "error"
                ? "#ef4444"
                : reviewNotice.kind === "warning"
                  ? "var(--mem-accent-amber)"
                  : "var(--mem-text-secondary)",
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            lineHeight: "1.5",
          }}
        >
          {reviewNotice.message}
          {reviewNotice.offerReload && (
            <button
              className="ml-2 underline"
              onClick={() => {
                // Clear only once the reload has actually landed. Dropping the
                // warning first and firing the refetch into the void leaves the
                // worst state on screen: the same stale text, no warning, and a
                // Review action that looks ready to approve content the daemon
                // has already refused once.
                void (async () => {
                  const reloaded = await refetchPage();
                  if (reloaded.isError) {
                    setReviewNotice({
                      kind: "error",
                      message: t("pageDetail.reviewReloadFailed"),
                      offerReload: true,
                    });
                    return;
                  }
                  setReviewNotice(null);
                })();
              }}
              type="button"
            >
              {t("pageDetail.reviewReload")}
            </button>
          )}
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

      {showCanvas ? (
        <div>
          <PageCanvas
            pageId={pageId}
            pageTitle={page.title}
            labelOverrides={labelOverrides}
            onMemoryClick={onMemoryClick}
            onPageClick={onPageClick}
            onEntityClick={onEntityClick}
          />
        </div>
      ) : editing ? (
        editGate.kind === "checking" ? (
          <div role="status" className="page-editor-notice">
            {t("pageDetail.editor.checking")}
          </div>
        ) : editGate.kind === "unsupported" ? (
          <div className="flex flex-col gap-3">
            <div role="alert" className="page-editor-notice">
              {t("pageDetail.editor.upgradeRequired", {
                floor: PAGE_EDIT_DAEMON_FLOOR,
                version:
                  editGate.version ??
                  t("pageDetail.editor.unavailableVersion"),
              })}
            </div>
            <button
              type="button"
              className="page-editor-action self-start"
              onClick={closeEditor}
            >
              {t("pageDetail.editor.cancel")}
            </button>
          </div>
        ) : editGate.kind === "normalize" ? (
          <div className="flex flex-col gap-3">
            <div role="alert" className="page-editor-notice">
              <strong>{t("pageDetail.editor.normalizeTitle")}</strong>
              <div>{t("pageDetail.editor.normalizeDescription")}</div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="page-editor-action"
                onClick={handleNormalizeAndEdit}
              >
                {t("pageDetail.editor.normalizeAction")}
              </button>
              <button
                type="button"
                className="page-editor-action"
                onClick={closeEditor}
              >
                {t("pageDetail.editor.cancel")}
              </button>
            </div>
          </div>
        ) : (
          <div className="page-editor-stack flex flex-col gap-2">
            {saveState.phase === "conflict" && (
              <div role="alert" className="page-editor-notice">
                <strong>{t("pageDetail.editor.conflictTitle")}</strong>
                <div>{t("pageDetail.editor.conflictBody")}</div>
                {conflictLatest.kind === "loading" && (
                  <div role="status">
                    {t("pageDetail.editor.latestLoading")}
                  </div>
                )}
                {conflictLatest.kind === "error" && (
                  <div>{t("pageDetail.editor.latestLoadFailed")}</div>
                )}
                {conflictLatest.kind === "loaded" && (
                  <details>
                    <summary>
                      {t("pageDetail.editor.latestSource", {
                        version: conflictLatest.page.version,
                      })}
                    </summary>
                    <pre>{conflictLatest.page.content}</pre>
                  </details>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    className="page-editor-action"
                    onClick={handleCopyDraft}
                  >
                    {t("pageDetail.editor.copyDraft")}
                  </button>
                  {conflictLatest.kind === "error" ? (
                    <button
                      type="button"
                      className="page-editor-action"
                      onClick={() => {
                        if (saveStateRef.current.phase === "conflict") {
                          void fetchConflictLatest(
                            saveStateRef.current.pending.operationId,
                          );
                        }
                      }}
                    >
                      {t("pageDetail.editor.retryLatest")}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="page-editor-action"
                      onClick={handleReloadLatest}
                      disabled={conflictLatest.kind !== "loaded"}
                    >
                      {t("pageDetail.editor.reloadLatest")}
                    </button>
                  )}
                  <button
                    type="button"
                    className="page-editor-action"
                    onClick={() => editorRef.current?.focus()}
                  >
                    {t("pageDetail.editor.keepEditing")}
                  </button>
                </div>
              </div>
            )}
            {saveState.phase === "retryable" && (
              <div role="alert" className="page-editor-notice">
                {failureMessage(
                  saveState.failure.outcome === "transport"
                    ? "transport"
                    : saveState.failure.kind,
                )}
                {saveState.failure.outcome !== "transport" &&
                  saveState.failure.message && (
                    <details>
                      <summary>
                        {t("pageDetail.editor.technicalDetails")}
                      </summary>
                      {saveState.failure.message}
                    </details>
                  )}
                <button
                  type="button"
                  className="page-editor-action"
                  onClick={handleCopyDraft}
                >
                  {t("pageDetail.editor.copyDraft")}
                </button>
              </div>
            )}
            {saveState.phase === "upgrade_required" && (
              <div role="alert" className="page-editor-notice">
                {t("pageDetail.editor.upgradeRequired", {
                  floor: saveState.requiredFloor,
                  version: saveState.reportedVersion,
                })}
                <button
                  type="button"
                  className="page-editor-action"
                  onClick={handleCopyDraft}
                >
                  {t("pageDetail.editor.copyDraft")}
                </button>
              </div>
            )}
            {saveState.phase === "failed" && (
              <div role="alert" className="page-editor-notice">
                {failureMessage(saveState.failure.kind)}
                {saveState.failure.message && (
                  <details>
                    <summary>{t("pageDetail.editor.technicalDetails")}</summary>
                    {saveState.failure.message}
                  </details>
                )}
                <button
                  type="button"
                  className="page-editor-action"
                  onClick={handleCopyDraft}
                >
                  {t("pageDetail.editor.copyDraft")}
                </button>
              </div>
            )}
            {editValidation && (
              <div role="alert" className="page-editor-notice">
                {editValidation}
              </div>
            )}
            {editorSessionId === editorFallbackSessionId && (
              <div role="alert" className="page-editor-notice">
                <strong>{t("pageDetail.editor.fallbackTitle")}</strong>
                <div>{t("pageDetail.editor.fallbackBody")}</div>
              </div>
            )}
            {editorSessionId && (
              <MarkdownEditorToolbar
                status={editorStatus}
                labels={editorToolbarLabels}
                saveDisabled={
                  saveState.phase === "pending" ||
                  saveState.phase === "conflict"
                }
                cancelDisabled={saveState.phase === "pending"}
                onCommand={(command) => editorRef.current?.runCommand(command)}
                onSave={() => editorRef.current?.requestSave()}
                onCancel={() => editorRef.current?.requestCancel()}
              />
            )}
            <p
              id="page-markdown-editor-description"
              className="sr-only"
            >
              {t(
                editorStatus.engine === "native"
                  ? "pageDetail.editor.descriptionFallback"
                  : "pageDetail.editor.description",
                { modifier: editorShortcutModifier },
              )}
            </p>
            {editorSessionId && (
              <MarkdownEditor
                ref={editorRef}
                initialDocument={editInitialDocument}
                sessionId={editorSessionId}
                disabled={saveState.phase === "pending"}
                ariaLabel={t("pageDetail.editor.label")}
                describedBy="page-markdown-editor-description"
                onDocumentChange={handleDocumentChange}
                onSave={saveDocument}
                onCancel={requestCloseEditor}
                onStatusChange={(status) =>
                  handleEditorStatus(
                    editorSessionId,
                    editorSessionEpoch,
                    status,
                  )
                }
                onFallback={(reason) =>
                  handleEditorFallback(
                    editorSessionId,
                    editorSessionEpoch,
                    reason,
                  )
                }
              />
            )}
          </div>
        )
      ) : (
        <div className={hasRail ? "page-detail-grid" : undefined}>
          <div className="page-detail-prose" onClickCapture={handleContentClick}>
            {ledeText && (
              <div className="page-detail-lede">
                {ledeMarkdown ? (
                  <ContentRenderer
                    content={ledeMarkdown}
                    variant="lede"
                    renderCitation={renderCitation}
                  />
                ) : (
                  <p>{ledeText}</p>
                )}
              </div>
            )}
            <ContentRenderer
              content={displayContent}
              variant="detail"
              renderCitation={renderCitation}
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

      {/* The canvas replaces the reading column rather than sitting above
          its apparatus: sources, citations and revisions are how a page is
          read, and the map is the other way of reading it. */}
      {!editing && !showCanvas && (
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
