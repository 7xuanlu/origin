// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  confirmEntity,
  deleteEntity,
  getEntityDetail,
  search,
  type EntityDetail as EntityDetailRecord,
} from "../../lib/tauri";
import { slotForEntityType } from "../../lib/graph/palette";
import { EntityConnections } from "./entity-detail/EntityConnections";
import { EntityContextRail } from "./entity-detail/EntityContextRail";
import "./entity-detail/EntityDetail.css";
import {
  entityMonogram,
  formatAbsoluteTimestamp,
  formatRelativeEntityTime,
} from "./entity-detail/formatEntityMetadata";
import { EntityObservations } from "./entity-detail/EntityObservations";
import FocusGraph from "./FocusGraph";

const AtlasView = lazy(() => import("./AtlasView"));

interface EntityDetailProps {
  entityId: string;
  onBack: () => void;
  onEntityClick: (entityId: string) => void;
  onMemoryClick?: (sourceId: string) => void;
}

export default function EntityDetail({
  entityId,
  onBack,
  onEntityClick,
  onMemoryClick,
}: EntityDetailProps) {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [graphOpen, setGraphOpen] = useState(false);
  const locale = i18n.resolvedLanguage ?? i18n.language;
  const { data: detail, isError, refetch } = useQuery({
    queryKey: ["entityDetail", entityId],
    queryFn: () => getEntityDetail(entityId),
    refetchInterval: 5_000,
  });
  const { data: linkedMemories = [] } = useQuery({
    queryKey: ["entity-linked-memories", entityId, detail?.entity.name],
    queryFn: async () => {
      if (!detail?.entity.name) return [];
      const results = await search(detail.entity.name, 10, "memory");
      return results
        .filter((result) => result.entity_id === entityId || result.score > 0.7)
        .slice(0, 8);
    },
    enabled: Boolean(detail?.entity.name),
    staleTime: 30_000,
  });
  useEffect(() => {
    if (!graphOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setGraphOpen(false);
    };
    window.addEventListener("keydown", handleKeyDown, true);
    return () => window.removeEventListener("keydown", handleKeyDown, true);
  }, [graphOpen]);
  const invalidateEntityDetail = () => {
    queryClient.invalidateQueries({ queryKey: ["entityDetail", entityId] });
    queryClient.invalidateQueries({ queryKey: ["entities"] });
  };
  const confirmMutation = useMutation({
    mutationFn: (confirmed: boolean) => confirmEntity(entityId, confirmed),
    onSuccess: invalidateEntityDetail,
  });
  const deleteMutation = useMutation({
    mutationFn: () => deleteEntity(entityId),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ["entityDetail", entityId] });
      queryClient.removeQueries({ queryKey: ["entity-linked-memories", entityId] });
      queryClient.invalidateQueries({ queryKey: ["entities"] });
      queryClient.invalidateQueries({ queryKey: ["space-entities"] });
      queryClient.invalidateQueries({ queryKey: ["constellation-entities"] });
      queryClient.invalidateQueries({ queryKey: ["constellation-relations"] });
      queryClient.invalidateQueries({ queryKey: ["connections-entities"] });
      queryClient.invalidateQueries({ queryKey: ["searchEntities"] });
      onBack();
    },
  });

  if (!detail) {
    return (
      <div className="page-detail entity-detail-dossier" aria-label={t("entityDetail.dossierLabel")}>
        <header className="entity-dossier-header">
          <BackButton onBack={onBack} label={t("entityDetail.back")} />
        </header>
        <div className="entity-detail-status" role="status">
          {isError ? (
            <>
              <p className="entity-empty">{t("entityDetail.loadError")}</p>
              <button
                type="button"
                className="memory-detail-text-button"
                onClick={() => refetch()}
              >
                {t("entityDetail.retry")}
              </button>
            </>
          ) : (
            <p className="entity-empty">{t("entityDetail.loading")}</p>
          )}
        </div>
      </div>
    );
  }

  const { entity, observations, relations } = detail;
  const space = entity.space ?? entity.domain;
  const relativeTime = formatRelativeEntityTime(entity.updated_at, locale);
  const absoluteTime = formatAbsoluteTimestamp(entity.updated_at);
  const dateline = [entity.entity_type, space, relativeTime].filter(Boolean).join(" · ");

  return (
    <>
      <div
        className="page-detail entity-detail-dossier"
        aria-label={t("entityDetail.dossierLabel")}
      >
      <header className="entity-dossier-header">
        <BackButton onBack={onBack} label={t("entityDetail.back")} />
        <div className="entity-dossier-hero">
          <div className="entity-dossier-hero-row">
            <div className="entity-detail-head">
              <div className="entity-detail-seal" aria-hidden="true">
                {entityMonogram(entity.name)}
              </div>
              <div className="flex-1 min-w-0">
                <h1 className="page-detail-title">{entity.name}</h1>
                <p className="page-detail-dateline" title={absoluteTime ?? undefined}>
                  {dateline}
                </p>
              </div>
            </div>
            <div className="memory-detail-actions">
              {confirmMutation.isError || deleteMutation.isError ? (
                <span className="entity-error" role="alert">
                  {t("entityDetail.saveError")}
                </span>
              ) : null}
              <button
                type="button"
                disabled={confirmMutation.isPending}
                onClick={() => confirmMutation.mutate(!entity.confirmed)}
                className={`memory-detail-chip entity-status-chip ${entity.confirmed ? "success" : "warning"}`}
                title={
                  entity.confirmed
                    ? t("entityDetail.markUnconfirmed")
                    : t("entityDetail.confirmEntity")
                }
              >
                {entity.confirmed ? t("entityDetail.confirmed") : t("entityDetail.confirmEntity")}
              </button>
              {confirmDelete ? (
                <>
                  <span className="entity-delete-question">{t("entityDetail.deleteQuestion")}</span>
                  <button
                    type="button"
                    disabled={deleteMutation.isPending}
                    onClick={() => deleteMutation.mutate()}
                    className="entity-delete-confirm"
                  >
                    {t("entityDetail.delete")}
                  </button>
                  <button
                    type="button"
                    disabled={deleteMutation.isPending}
                    onClick={() => setConfirmDelete(false)}
                    className="memory-detail-text-button"
                  >
                    {t("entityDetail.cancel")}
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmDelete(true)}
                  className="memory-detail-icon-button memory-detail-delete"
                  aria-label={t("entityDetail.deleteEntity")}
                  title={t("entityDetail.deleteEntity")}
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
                    <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>
      <div className="page-detail-grid">
        <section
          className="page-detail-prose entity-detail-reading"
          aria-label={t("entityDetail.readingLabel")}
        >
          <EntityConnections
            name={entity.name}
            relations={relations}
            onEntityClick={onEntityClick}
            onExpand={() => setGraphOpen(true)}
          />
          <EntityObservations
            entityId={entityId}
            entityName={entity.name}
            observations={observations}
            onInvalidate={invalidateEntityDetail}
          />
        </section>
        <EntityContextRail entity={entity} locale={locale} onMemoryClick={onMemoryClick} />
      </div>
      </div>
      {graphOpen ? (
        <EntityGraphOverlay
          detail={detail}
          linkedMemoriesCount={linkedMemories.length}
          locale={locale}
          onClose={() => setGraphOpen(false)}
          onEntityClick={onEntityClick}
        />
      ) : null}
    </>
  );
}

function BackButton({ onBack, label }: { readonly onBack: () => void; readonly label: string }) {
  return (
    <button
      type="button"
      onClick={onBack}
      className="memory-detail-back"
      aria-label={label}
      title={label}
    >
      <svg
        aria-hidden="true"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M19 12H5M12 19l-7-7 7-7" />
      </svg>
    </button>
  );
}

type EntityGraphOverlayProps = {
  readonly detail: EntityDetailRecord;
  readonly linkedMemoriesCount: number;
  readonly locale: string;
  readonly onClose: () => void;
  readonly onEntityClick: (entityId: string) => void;
};

function EntityGraphOverlay({
  detail,
  linkedMemoriesCount,
  locale,
  onClose,
  onEntityClick,
}: EntityGraphOverlayProps) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"focus" | "map">("focus");
  const [showVerbs, setShowVerbs] = useState(true);
  const { entity, relations } = detail;
  const neighborCount = useMemo(
    () =>
      new Set(
        relations
          .filter((relation) => relation.entity_id !== entity.id)
          .map((relation) => relation.entity_id),
      ).size,
    [entity.id, relations],
  );
  const openEntity = (entityId: string) => {
    onClose();
    onEntityClick(entityId);
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t("entityDetail.expandGraph")}
      className="fixed inset-0 z-50"
      style={{
        background: "var(--mem-bg)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 16px",
          borderBottom: "1px solid var(--mem-border)",
          background: "var(--mem-surface)",
          fontFamily: "var(--mem-font-body)",
          flexWrap: "wrap",
        }}
      >
        <button
          type="button"
          autoFocus
          onClick={onClose}
          className="memory-detail-icon-button"
          aria-label={t("common.close")}
          title={t("common.close")}
        >
          <svg
            aria-hidden="true"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
        <span style={{ fontSize: 12, color: "var(--mem-text-tertiary)" }}>
          {t("focus.crumbAtlas")}
          {mode === "focus" ? (
            <>
              {" ▸ "}
              <b style={{ color: "var(--mem-text)", fontWeight: 500 }}>
                {t("focus.crumbFocus", { name: entity.name })}
              </b>
            </>
          ) : null}
        </span>
        {mode === "focus" ? (
          <button
            type="button"
            aria-pressed={showVerbs}
            onClick={() => setShowVerbs((value) => !value)}
            style={{
              fontSize: 12,
              color: showVerbs ? "var(--mem-text)" : "var(--mem-text-secondary)",
              border: `1px solid ${
                showVerbs ? "var(--mem-distilled-border)" : "var(--mem-border)"
              }`,
              borderRadius: "var(--mem-radius-full)",
              padding: "4px 12px",
              background: showVerbs ? "var(--mem-indigo-bg)" : "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            {t("focus.showVerbs")}
          </button>
        ) : null}
        {mode === "focus" ? (
          <span
            style={{
              marginLeft: "auto",
              font: "400 11px var(--mem-font-mono)",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {t("focus.neighbors", { count: neighborCount })}
            {linkedMemoriesCount > 0
              ? ` · ${t("focus.memoriesCount", { count: linkedMemoriesCount })}`
              : ""}
          </span>
        ) : null}
        <div
          role="group"
          aria-label={t("focus.viewSegmentLabel")}
          style={{
            display: "flex",
            border: "1px solid var(--mem-border)",
            borderRadius: "var(--mem-radius-md)",
            overflow: "hidden",
            marginLeft: mode === "focus" ? 0 : "auto",
          }}
        >
          {(["map", "focus"] as const).map((nextMode) => {
            const selected = mode === nextMode;
            return (
              <button
                key={nextMode}
                type="button"
                aria-pressed={selected}
                onClick={() => setMode(nextMode)}
                style={{
                  fontSize: 12,
                  padding: "4px 14px",
                  border: "none",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  color: selected ? "var(--mem-text)" : "var(--mem-text-tertiary)",
                  fontWeight: selected ? 500 : 400,
                  background: selected ? "var(--mem-hover-strong)" : "transparent",
                }}
              >
                {nextMode === "map" ? t("focus.segAtlas") : t("focus.segFocus")}
              </button>
            );
          })}
        </div>
      </div>
      <div
        style={{
          flex: 1,
          minHeight: 0,
          position: "relative",
          ...(mode === "focus"
            ? { display: "grid", gridTemplateColumns: "minmax(0, 1fr) 300px" }
            : {}),
        }}
      >
        {mode === "focus" ? (
          <>
            <div style={{ position: "relative", minWidth: 0 }}>
              <FocusGraph
                detail={detail}
                onEntityClick={openEntity}
                fill
                showVerbs={showVerbs}
                memoriesCount={linkedMemoriesCount}
              />
            </div>
            <aside
              aria-label={t("focus.panelLabel")}
              style={{
                borderLeft: "1px solid var(--mem-border)",
                background: "var(--mem-surface)",
                padding: 18,
                overflowY: "auto",
                fontFamily: "var(--mem-font-body)",
              }}
            >
              <div
                style={{
                  font: "500 10px var(--mem-font-mono)",
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                <i
                  aria-hidden="true"
                  style={{
                    color: `var(--kg-${slotForEntityType(entity.entity_type)})`,
                    fontStyle: "normal",
                  }}
                >
                  ●
                </i>
                {` ${entity.entity_type} · ${
                  entity.confirmed
                    ? t("focus.confirmedState")
                    : t("focus.unconfirmedState")
                }`}
              </div>
              <h4
                style={{
                  fontSize: 22,
                  fontWeight: 600,
                  margin: "6px 0 2px",
                  color: "var(--mem-text)",
                }}
              >
                {entity.name}
              </h4>
              <div style={{ fontSize: 12, color: "var(--mem-text-tertiary)" }}>
                {t("focus.observations", { count: detail.observations.length })}
                {` · ${t("focus.relations", { count: relations.length })}`}
                {` · ${t("focus.updatedAgo", {
                  ago: formatRelativeEntityTime(entity.updated_at, locale) ?? "",
                })}`}
              </div>
              <hr
                style={{
                  border: "none",
                  borderTop: "1px solid var(--mem-detail-divider)",
                  margin: "14px 0",
                }}
              />
              <div>
                {relations.map((relation) => (
                  <button
                    key={relation.id}
                    type="button"
                    onClick={() => openEntity(relation.entity_id)}
                    style={{
                      display: "flex",
                      alignItems: "baseline",
                      gap: 8,
                      padding: "5px 0",
                      fontSize: 12.5,
                      width: "100%",
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      textAlign: "left",
                      fontFamily: "inherit",
                    }}
                  >
                    <span
                      aria-hidden="true"
                      style={{
                        width: 7,
                        height: 7,
                        borderRadius: "var(--mem-radius-full)",
                        flex: "none",
                        alignSelf: "center",
                        background: `var(--kg-${slotForEntityType(relation.entity_type)})`,
                      }}
                    />
                    <code
                      style={{
                        font: "500 10px var(--mem-font-mono)",
                        color: "var(--mem-text-tertiary)",
                        letterSpacing: "0.04em",
                        minWidth: 96,
                      }}
                    >
                      {relation.direction === "incoming"
                        ? `${relation.relation_type} ←`
                        : `${relation.relation_type} →`}
                    </code>
                    <span style={{ color: "var(--mem-text)" }}>
                      {relation.entity_name}
                    </span>
                  </button>
                ))}
              </div>
              <hr
                style={{
                  border: "none",
                  borderTop: "1px solid var(--mem-detail-divider)",
                  margin: "14px 0",
                }}
              />
              <button
                type="button"
                onClick={onClose}
                style={{
                  display: "inline-block",
                  font: "500 12.5px var(--mem-font-body)",
                  color: "var(--mem-text)",
                  background: "var(--mem-indigo-bg)",
                  border: "1px solid var(--mem-distilled-border)",
                  borderRadius: "var(--mem-radius-md)",
                  padding: "7px 14px",
                  cursor: "pointer",
                }}
              >
                {t("focus.openEntity")}
              </button>
            </aside>
          </>
        ) : (
          <Suspense fallback={null}>
            <AtlasView focusEntityId={entity.id} onNodeClick={openEntity} />
          </Suspense>
        )}
      </div>
    </div>
  );
}
