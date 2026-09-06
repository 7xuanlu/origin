// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useEffect, useRef, useState } from "react";
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
    }
  | { kind: "deletePermanently"; ids: string[] };

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
      if (reset) {
        setLoading(true);
        setLoadError(false);
      } else {
        setLoadingMore(true);
      }
      try {
        const response = await queryEntities(entityListRequest(tab, filters, nextOffset));
        setEntities((current) => (reset ? response.entities : [...current, ...response.entities]));
        setTotal(response.total);
        offsetRef.current = nextOffset + response.entities.length;
      } catch {
        if (reset) setLoadError(true);
        else toast.error(t("entities.error.load"));
      } finally {
        setLoading(false);
        setLoadingMore(false);
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
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setDialog(null);
    };
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [dialog]);

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
      const filter = entityListRequest("detected", filters, 0);
      const dryRun = await archiveEntities({ filter, dry_run: true });
      // The Memories chip already excludes them when it's "None"; otherwise a
      // second dry-run (same filter, but only entities that have memories)
      // tells the dialog whether archiving would quietly take some with it.
      let includesMemoriesCount: number | null = null;
      if (filters.memories !== "none") {
        const withMemories = await archiveEntities({ filter: { ...filter, min_memories: 1 }, dry_run: true });
        includesMemoriesCount = withMemories.count > 0 ? withMemories.count : null;
      }
      setDialog({ kind: "archiveMatching", count: dryRun.count, filter, includesMemoriesCount });
    } catch {
      toast.error(t("entities.error.action"));
    } finally {
      setArchiveMatchingPending(false);
    }
  };

  const openDeletePermanentlyDialog = (ids: string[]) => {
    setDialog({ kind: "deletePermanently", ids });
  };

  const confirmDialog = async () => {
    if (!dialog) return;
    setDialogPending(true);
    try {
      if (dialog.kind === "archiveMatching") {
        await archiveEntities({ filter: dialog.filter, dry_run: false });
        toast.success(t("entities.toast_archived", { count: dialog.count }));
      } else {
        await Promise.all(dialog.ids.map((id) => deleteEntity(id)));
        toast.success(t("entities.toast_deleted", { count: dialog.ids.length }));
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

      <div className="entities-tabs" role="tablist" aria-label={t("entities.title")}>
        {(["established", "detected", "archived"] as const).map((key) => (
          <button
            key={key}
            aria-selected={tab === key}
            className="entities-tab"
            onClick={() => setTab(key)}
            role="tab"
            type="button"
          >
            {t(`entities.tabs.${key}`)}
            <span className="entities-tab-count">{counts[key]}</span>
          </button>
        ))}
      </div>

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
              <button
                className="entities-ghost-btn entities-danger-btn"
                onClick={() => openDeletePermanentlyDialog([...selected])}
                type="button"
              >
                {t("entities.actions.deletePermanently")}
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
                    <button className="entities-ghost-btn" onClick={() => void handleRestore([entity.id])} type="button">
                      {t("entities.actions.restore")}
                    </button>
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

      {dialog && (
        <div className="entities-scrim">
          <div aria-labelledby="entities-dialog-title" aria-modal="true" className="entities-dialog" role="dialog">
            <h2 id="entities-dialog-title">
              {dialog.kind === "archiveMatching"
                ? t("entities.dialogs.archiveMatching.title", { count: dialog.count })
                : t("entities.dialogs.deletePermanently.title", { count: dialog.ids.length })}
            </h2>
            <p>
              {dialog.kind === "archiveMatching"
                ? t("entities.dialogs.archiveMatching.body")
                : t("entities.dialogs.deletePermanently.body")}
            </p>
            {dialog.kind === "archiveMatching" && (
              <div className="entities-dialog-scope">
                <span className="entities-dialog-scope-key">{t("entities.dialogs.archiveMatching.filterLabel")}</span>
                <span>{filterReadback(filters, t)}</span>
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
              <button className="entities-ghost-btn" disabled={dialogPending} onClick={() => setDialog(null)} type="button">
                {dialog.kind === "archiveMatching"
                  ? t("entities.dialogs.archiveMatching.cancel")
                  : t("entities.dialogs.deletePermanently.cancel")}
              </button>
              <button
                autoFocus
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
