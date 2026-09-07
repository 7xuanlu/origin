// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import {
  archiveEntities,
  confirmEntity,
  deleteEntity,
  queryEntities,
  restoreEntities,
  type Entity,
  type ListEntitiesRequest,
} from "../../../lib/tauri";
import { formatRelativeEntityTime } from "../entity-detail/formatEntityMetadata";
import {
  DEFAULT_FILTERS,
  entityListRequest,
  establishedByLabel,
  filterReadback,
  filtersActive,
  matchLabel,
  selectAllState,
  selectionSummary,
  subtitle,
  toggleSelectAll,
  toggleSelected,
  type EntityFilters,
  type EntityMemoriesFilter,
  type EntityTab,
  type EntityTypeFilter,
} from "./entitiesViewModel";
import "./EntitiesView.css";

interface EntitiesViewProps {
  /** Opens the entity dossier (the "Open" action on an established row). */
  readonly onEntityClick: (entityId: string) => void;
}

type Counts = { established: number; detected: number; archived: number };

type Dialog =
  | {
      kind: "archiveMatching";
      count: number;
      filter: ListEntitiesRequest;
      /** How many of the matched entities already have memories, shown as an
       * extra "Includes" line so archiving doesn't quietly take memories with
       * it. Null when the Memories chip already excludes them (filter is
       * "None") or the second dry-run came back at zero. */
      includesMemoriesCount: number | null;
      /** The filters the dry run was taken with, rendered as the dialog's
       * readback. A snapshot, not the live state: the search box debounces
       * into `filters`, so the live value can describe a different filter
       * than the one the dry run (and the apply) actually use. */
      readback: EntityFilters;
    }
  | { kind: "deletePermanently"; id: string; name: string };

const TAB_ORDER: readonly EntityTab[] = ["established", "detected", "archived"];

const TYPE_CHIPS: EntityTypeFilter[] = ["all", "concept", "person", "organization", "place"];
const MEMORIES_CHIPS: EntityMemoriesFilter[] = ["any", "none", "some"];

export function EntitiesView({ onEntityClick }: EntitiesViewProps) {
  const { i18n, t } = useTranslation();
  const locale = i18n.resolvedLanguage ?? i18n.language;

  const [tab, setTab] = useState<EntityTab>("detected");
  const [filters, setFilters] = useState<EntityFilters>(DEFAULT_FILTERS);
  const [queryInput, setQueryInput] = useState("");
  const [counts, setCounts] = useState<Counts>({ established: 0, detected: 0, archived: 0 });
  const [entities, setEntities] = useState<Entity[]>([]);
  const [total, setTotal] = useState(0);
  // A ref, not state: `loadPage` is memoized on [tab, filters] alone (see
  // below), so an `offset` dependency it doesn't declare would go stale after
  // the first page — "Load more" would keep refetching offset 0 forever.
  const offsetRef = useRef(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [dialog, setDialog] = useState<Dialog | null>(null);
  const [dialogPending, setDialogPending] = useState(false);
  const [archiveMatchingPending, setArchiveMatchingPending] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Monotonic request id: a list response only lands if it is still the
  // newest request. Without it a slow Detected response could overwrite the
  // Archived list after the user switched tabs, and the row actions would
  // then act on rows the current tab never offers.
  const requestSeqRef = useRef(0);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Debounce the search box into `filters.query` (Detected tab only).
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setFilters((current) => (current.query === queryInput ? current : { ...current, query: queryInput }));
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [queryInput]);

  const refreshCounts = useCallback(async () => {
    try {
      const [established, detected, archived] = await Promise.all([
        queryEntities({ status: "established", limit: 1, offset: 0 }),
        queryEntities({ status: "detected", limit: 1, offset: 0 }),
        queryEntities({ status: "archived", limit: 1, offset: 0 }),
      ]);
      setCounts({ established: established.total, detected: detected.total, archived: archived.total });
    } catch {
      // The tab counts are supplementary; a failure here does not block the
      // active tab's own list load, which surfaces its own error.
    }
  }, []);

  const loadPage = useCallback(
    async (reset: boolean) => {
      const nextOffset = reset ? 0 : offsetRef.current;
      const seq = ++requestSeqRef.current;
      if (reset) {
        setLoading(true);
        setLoadError(false);
      } else {
        setLoadingMore(true);
      }
      try {
        const response = await queryEntities(entityListRequest(tab, filters, nextOffset));
        if (seq !== requestSeqRef.current) return;
        setEntities((current) => (reset ? response.entities : [...current, ...response.entities]));
        setTotal(response.total);
        offsetRef.current = nextOffset + response.entities.length;
      } catch {
        if (seq !== requestSeqRef.current) return;
        if (reset) setLoadError(true);
        else toast.error(t("entities.error.load"));
      } finally {
        if (seq === requestSeqRef.current) {
          setLoading(false);
          setLoadingMore(false);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tab, filters],
  );

  useEffect(() => {
    setSelected(new Set());
    void loadPage(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, filters]);

  useEffect(() => {
    void refreshCounts();
  }, [refreshCounts]);

  useEffect(() => {
    if (!dialog) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        // The request is already in flight; closing now would only hide
        // its outcome.
        if (!dialogPending) setDialog(null);
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>("button:not([disabled])"),
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && (active === first || !dialogRef.current.contains(active))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || !dialogRef.current.contains(active))) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [dialog, dialogPending]);

  const handleTabListKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const index = TAB_ORDER.indexOf(tab);
    let next: EntityTab | undefined;
    if (event.key === "ArrowRight") next = TAB_ORDER[(index + 1) % TAB_ORDER.length];
    else if (event.key === "ArrowLeft") next = TAB_ORDER[(index + TAB_ORDER.length - 1) % TAB_ORDER.length];
    else if (event.key === "Home") next = TAB_ORDER[0];
    else if (event.key === "End") next = TAB_ORDER[TAB_ORDER.length - 1];
    if (!next) return;
    event.preventDefault();
    setTab(next);
    event.currentTarget.querySelector<HTMLButtonElement>(`#entities-tab-${next}`)?.focus();
  };

  const reload = useCallback(async () => {
    setSelected(new Set());
    await Promise.all([loadPage(true), refreshCounts()]);
  }, [loadPage, refreshCounts]);

  const withActionErrorHandling = async (action: () => Promise<void>) => {
    try {
      await action();
    } catch {
      toast.error(t("entities.error.action"));
    }
  };

  const handleEstablish = (ids: string[]) => withActionErrorHandling(async () => {
    await Promise.all(ids.map((id) => confirmEntity(id, true)));
    toast.success(t("entities.toast_established", { count: ids.length }));
    await reload();
  });

  const handleArchive = (ids: string[]) => withActionErrorHandling(async () => {
    await archiveEntities({ ids, dry_run: false });
    toast.success(t("entities.toast_archived", { count: ids.length }));
    await reload();
  });

  const handleRestore = (ids: string[]) => withActionErrorHandling(async () => {
    await restoreEntities({ ids, dry_run: false });
    toast.success(t("entities.toast_restored", { count: ids.length }));
    await reload();
  });

  const handleRestoreAll = () => withActionErrorHandling(async () => {
    const response = await restoreEntities({ filter: { status: "archived" }, dry_run: false });
    toast.success(t("entities.toast_restored", { count: response.count }));
    await reload();
  });

  const openArchiveMatchingDialog = async () => {
    setArchiveMatchingPending(true);
    try {
      // Snapshot what the user can see -- including a search term the 300 ms
      // debounce has not folded into `filters` yet -- and flush it, so the
      // dry run, the readback, and the apply all describe the same filter.
      const readback: EntityFilters = { ...filters, query: queryInput };
      if (readback.query !== filters.query) setFilters(readback);
      const filter = entityListRequest("detected", readback, 0);
      const dryRun = await archiveEntities({ filter, dry_run: true });
      // The Memories chip already excludes them when it's "None"; otherwise a
      // second dry-run (same filter, but only entities that have memories)
      // tells the dialog whether archiving would quietly take some with it.
      let includesMemoriesCount: number | null = null;
      if (readback.memories !== "none") {
        const withMemories = await archiveEntities({ filter: { ...filter, min_memories: 1 }, dry_run: true });
        includesMemoriesCount = withMemories.count > 0 ? withMemories.count : null;
      }
      setDialog({ kind: "archiveMatching", count: dryRun.count, filter, includesMemoriesCount, readback });
    } catch {
      toast.error(t("entities.error.action"));
    } finally {
      setArchiveMatchingPending(false);
    }
  };

  // One row at a time (spec): permanent delete has no undo, so it never
  // rides on a multi-select.
  const openDeletePermanentlyDialog = (entity: Entity) => {
    setDialog({ kind: "deletePermanently", id: entity.id, name: entity.name });
  };

  const confirmDialog = async () => {
    if (!dialog) return;
    setDialogPending(true);
    try {
      if (dialog.kind === "archiveMatching") {
        await archiveEntities({ filter: dialog.filter, dry_run: false });
        toast.success(t("entities.toast_archived", { count: dialog.count }));
      } else {
        await deleteEntity(dialog.id);
        toast.success(t("entities.toast_deleted", { count: 1 }));
      }
      setDialog(null);
      await reload();
    } catch {
      toast.error(t("entities.error.action"));
    } finally {
      setDialogPending(false);
    }
  };

  const ids = entities.map((entity) => entity.id);
  const selectAll = selectAllState(selected, ids);
  const hasMore = entities.length < total;
  const detectedFiltered = filtersActive(filters);

  return (
    <section aria-labelledby="entities-title" className="entities-view">
      <header className="entities-header">
        <h1 id="entities-title">{t("entities.title")}</h1>
        <span className="entities-subtitle">{subtitle(counts, t)}</span>
      </header>

      <div className="entities-tabs" role="tablist" aria-label={t("entities.title")} onKeyDown={handleTabListKeyDown}>
        {TAB_ORDER.map((key) => (
          <button
            key={key}
            aria-controls="entities-tabpanel"
            aria-selected={tab === key}
            className="entities-tab"
            id={`entities-tab-${key}`}
            onClick={() => setTab(key)}
            role="tab"
            tabIndex={tab === key ? 0 : -1}
            type="button"
          >
            {t(`entities.tabs.${key}`)}
            <span className="entities-tab-count">{counts[key]}</span>
          </button>
        ))}
      </div>

      <div aria-labelledby={`entities-tab-${tab}`} id="entities-tabpanel" role="tabpanel">
      {tab === "detected" && (
        <div className="entities-filters" aria-label={t("entities.filters.typeLabel")}>
          <input
            aria-label={t("entities.search.label")}
            className="entities-search"
            onChange={(event) => setQueryInput(event.target.value)}
            placeholder={t("entities.search.placeholder")}
            type="search"
            value={queryInput}
          />
          <div className="entities-chip-group" role="group" aria-label={t("entities.filters.typeLabel")}>
            <span className="entities-chip-label">{t("entities.filters.typeLabel")}</span>
            {TYPE_CHIPS.map((value) => (
              <button
                aria-pressed={filters.type === value}
                className="entities-chip"
                key={value}
                onClick={() => setFilters((current) => ({ ...current, type: value }))}
                type="button"
              >
                {value === "all" ? t("entities.filters.typeAny") : t(`entities.filters.type_${value}`)}
              </button>
            ))}
          </div>
          <div className="entities-chip-group" role="group" aria-label={t("entities.filters.memoriesLabel")}>
            <span className="entities-chip-label">{t("entities.filters.memoriesLabel")}</span>
            {MEMORIES_CHIPS.map((value) => (
              <button
                aria-pressed={filters.memories === value}
                className="entities-chip"
                key={value}
                onClick={() => setFilters((current) => ({ ...current, memories: value }))}
                type="button"
              >
                {value === "any"
                  ? t("entities.filters.memoriesAny")
                  : value === "none"
                    ? t("entities.filters.memoriesNone")
                    : t("entities.filters.memoriesSome")}
              </button>
            ))}
          </div>
        </div>
      )}

      {tab === "detected" && (
        <div className="entities-matchline">
          <span>{matchLabel("detected", total, detectedFiltered, t)}</span>
          <button
            className="entities-ghost-btn"
            disabled={total === 0 || archiveMatchingPending}
            onClick={() => void openArchiveMatchingDialog()}
            type="button"
          >
            {t("entities.archiveAllMatching")}
          </button>
        </div>
      )}

      {tab === "archived" && (
        <div className="entities-matchline">
          <span>{matchLabel("archived", total, false, t)}</span>
          <button
            className="entities-ghost-btn"
            disabled={total === 0}
            onClick={() => void handleRestoreAll()}
            type="button"
          >
            {t("entities.restoreAll")}
          </button>
        </div>
      )}

      {(tab === "detected" || tab === "archived") && selected.size > 0 && (
        <div className="entities-selection-bar">
          <span className="entities-selection-count">{selectionSummary(selected.size, t)}</span>
          {tab === "detected" ? (
            <>
              <button className="entities-ghost-btn" onClick={() => void handleEstablish([...selected])} type="button">
                {t("entities.selectionBar.establishSelected")}
              </button>
              <button className="entities-ghost-btn" onClick={() => void handleArchive([...selected])} type="button">
                {t("entities.selectionBar.archiveSelected")}
              </button>
            </>
          ) : (
            <>
              <button className="entities-ghost-btn" onClick={() => void handleRestore([...selected])} type="button">
                {t("entities.selectionBar.restoreSelected")}
              </button>
            </>
          )}
        </div>
      )}

      {loading ? (
        <p className="entities-state">{t("entities.loading")}</p>
      ) : loadError ? (
        <p className="entities-state" role="alert">{t("entities.error.load")}</p>
      ) : entities.length === 0 ? (
        <div className="entities-empty">
          <b>{t(`entities.empty.${tab}Title`)}</b>
          <p>{t(`entities.empty.${tab}Body`)}</p>
        </div>
      ) : (
        <table className="entities-table">
          <thead>
            <tr>
              {(tab === "detected" || tab === "archived") && (
                <th className="entities-check-col">
                  <input
                    aria-label={t("entities.actions.selectAll")}
                    checked={selectAll.checked}
                    onChange={(event) => setSelected((current) => toggleSelectAll(current, ids, event.target.checked))}
                    ref={(element) => {
                      if (element) element.indeterminate = selectAll.indeterminate;
                    }}
                    type="checkbox"
                  />
                </th>
              )}
              <th>{t("entities.columns.entity")}</th>
              <th>{t("entities.columns.type")}</th>
              <th style={{ textAlign: "right" }}>{t("entities.columns.memories")}</th>
              {tab === "detected" && <th style={{ textAlign: "right" }}>{t("entities.columns.detected")}</th>}
              {tab === "established" && <th>{t("entities.columns.establishedBy")}</th>}
              {tab === "archived" && <th style={{ textAlign: "right" }}>{t("entities.columns.archived")}</th>}
              <th />
            </tr>
          </thead>
          <tbody>
            {entities.map((entity) => (
              <tr key={entity.id}>
                {(tab === "detected" || tab === "archived") && (
                  <td className="entities-check-col">
                    <input
                      aria-label={t("entities.actions.selectEntity", { name: entity.name })}
                      checked={selected.has(entity.id)}
                      onChange={() => setSelected((current) => toggleSelected(current, entity.id))}
                      type="checkbox"
                    />
                  </td>
                )}
                <td>
                  {tab === "established" ? (
                    <button className="entities-name-link" onClick={() => onEntityClick(entity.id)} type="button">
                      {entity.name}
                    </button>
                  ) : entity.name}
                </td>
                <td>{entity.entity_type}</td>
                <td style={{ textAlign: "right" }}>{entity.memory_count}</td>
                {tab === "detected" && (
                  <td style={{ textAlign: "right" }}>{formatRelativeEntityTime(entity.created_at, locale)}</td>
                )}
                {tab === "established" && <td>{establishedByLabel(entity, t)}</td>}
                {tab === "archived" && (
                  <td style={{ textAlign: "right" }}>{formatRelativeEntityTime(entity.updated_at, locale)}</td>
                )}
                <td className="entities-row-actions">
                  {tab === "detected" && (
                    <>
                      <button className="entities-ghost-btn" onClick={() => void handleEstablish([entity.id])} type="button">
                        {t("entities.actions.establish")}
                      </button>
                      <button className="entities-ghost-btn" onClick={() => void handleArchive([entity.id])} type="button">
                        {t("entities.actions.archive")}
                      </button>
                    </>
                  )}
                  {tab === "established" && (
                    <>
                      <button className="entities-ghost-btn" onClick={() => onEntityClick(entity.id)} type="button">
                        {t("entities.actions.open")}
                      </button>
                      <button className="entities-ghost-btn" onClick={() => void handleArchive([entity.id])} type="button">
                        {t("entities.actions.archive")}
                      </button>
                    </>
                  )}
                  {tab === "archived" && (
                    <>
                      <button className="entities-ghost-btn" onClick={() => void handleRestore([entity.id])} type="button">
                        {t("entities.actions.restore")}
                      </button>
                      <button
                        aria-label={t("entities.actions.deletePermanentlyNamed", { name: entity.name })}
                        className="entities-ghost-btn entities-danger-btn"
                        onClick={() => openDeletePermanentlyDialog(entity)}
                        type="button"
                      >
                        {t("entities.actions.deletePermanently")}
                      </button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!loading && !loadError && hasMore && (
        <button
          className="entities-ghost-btn entities-load-more"
          disabled={loadingMore}
          onClick={() => void loadPage(false)}
          type="button"
        >
          {t("entities.actions.loadMore")}
        </button>
      )}
      </div>

      {dialog && (
        <div className="entities-scrim">
          <div
            aria-labelledby="entities-dialog-title"
            aria-modal="true"
            className="entities-dialog"
            ref={dialogRef}
            role="dialog"
          >
            <h2 id="entities-dialog-title">
              {dialog.kind === "archiveMatching"
                ? t("entities.dialogs.archiveMatching.title", { count: dialog.count })
                : t("entities.dialogs.deletePermanently.titleNamed", { name: dialog.name })}
            </h2>
            <p>
              {dialog.kind === "archiveMatching"
                ? t("entities.dialogs.archiveMatching.body")
                : t("entities.dialogs.deletePermanently.body")}
            </p>
            {dialog.kind === "archiveMatching" && (
              <div className="entities-dialog-scope">
                <span className="entities-dialog-scope-key">{t("entities.dialogs.archiveMatching.filterLabel")}</span>
                <span>{filterReadback(dialog.readback, t)}</span>
              </div>
            )}
            {dialog.kind === "archiveMatching" && dialog.includesMemoriesCount !== null && (
              <div className="entities-dialog-scope">
                <span className="entities-dialog-scope-key">{t("entities.dialogs.archiveMatching.includesLabel")}</span>
                <span className="entities-dialog-scope-value">
                  {t("entities.dialogs.archiveMatching.includesMemories")}
                  <span className="entities-dialog-scope-count">{dialog.includesMemoriesCount}</span>
                </span>
              </div>
            )}
            <p className="entities-dialog-hint">
              {dialog.kind === "archiveMatching"
                ? (dialog.includesMemoriesCount !== null
                  ? t("entities.dialogs.archiveMatching.reversibleHintWithMemories")
                  : t("entities.dialogs.archiveMatching.reversibleHint"))
                : t("entities.dialogs.deletePermanently.irreversibleHint")}
            </p>
            <div className="entities-dialog-actions">
              <button
                autoFocus
                className="entities-ghost-btn"
                disabled={dialogPending}
                onClick={() => setDialog(null)}
                type="button"
              >
                {dialog.kind === "archiveMatching"
                  ? t("entities.dialogs.archiveMatching.cancel")
                  : t("entities.dialogs.deletePermanently.cancel")}
              </button>
              <button
                className={dialog.kind === "deletePermanently" ? "entities-primary-btn entities-danger-btn" : "entities-primary-btn"}
                disabled={dialogPending}
                onClick={() => void confirmDialog()}
                type="button"
              >
                {dialog.kind === "archiveMatching"
                  ? t("entities.dialogs.archiveMatching.cta")
                  : t("entities.dialogs.deletePermanently.cta")}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
