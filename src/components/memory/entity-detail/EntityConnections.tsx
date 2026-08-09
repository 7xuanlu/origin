// SPDX-License-Identifier: AGPL-3.0-only
import { useMemo, useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";
import type { RelationWithEntity } from "../../../lib/tauri";
import {
  layoutGraphNeighbors,
  type GraphDirection,
  type GraphNeighbor,
} from "./entityGraphLayout";

type EntityConnectionsProps = {
  readonly name: string;
  readonly relations: readonly RelationWithEntity[];
  readonly onEntityClick: (entityId: string) => void;
  readonly onExpand?: () => void;
};

type EntityGraphProps = {
  readonly name: string;
  readonly neighbors: readonly GraphNeighbor[];
  readonly hiddenCount: number;
  readonly onEntityClick: (entityId: string) => void;
};

const GRAPH_NODE_CAP = 8;
const COMPACT_GRAPH_QUERY = "(max-width: 640px)";

function subscribeToCompactGraph(onChange: () => void): () => void {
  if (typeof window.matchMedia !== "function") return () => {};
  const media = window.matchMedia(COMPACT_GRAPH_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}

function compactGraphSnapshot(): boolean {
  return typeof window.matchMedia === "function" && window.matchMedia(COMPACT_GRAPH_QUERY).matches;
}

function EntityGraph({ name, neighbors, hiddenCount, onEntityClick }: EntityGraphProps) {
  const { t } = useTranslation();
  const compact = useSyncExternalStore(subscribeToCompactGraph, compactGraphSnapshot, () => false);
  const incoming = neighbors.filter((neighbor) => neighbor.direction === "incoming");
  const outgoing = neighbors.filter((neighbor) => neighbor.direction === "outgoing");
  const maxSide = Math.max(incoming.length, outgoing.length);
  const height = compact
    ? Math.min(600, 160 + neighbors.length * 54)
    : Math.min(280, 128 + maxSide * 30);
  const placed = layoutGraphNeighbors(neighbors, compact);

  return (
    <div
      className={`entity-graph ${compact ? "is-compact" : ""}`}
      style={{ height }}
      role="group"
      aria-label={t("entityDetail.graphLabel", { name })}
    >
      <svg
        className="entity-graph-edges"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {placed.map((neighbor) => (
          <line
            key={`edge-${neighbor.entityId}-${neighbor.direction}`}
            x1="50"
            y1="50"
            x2={neighbor.x}
            y2={neighbor.y}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      {incoming.length > 0 ? (
        <span className="entity-graph-caption is-left" aria-hidden="true">
          ← {t("entityDetail.incoming")}
        </span>
      ) : null}
      {outgoing.length > 0 ? (
        <span className="entity-graph-caption is-right" aria-hidden="true">
          {t("entityDetail.outgoing")} →
        </span>
      ) : null}
      {placed.map((neighbor) => (
        <span
          key={`verb-${neighbor.entityId}-${neighbor.direction}`}
          className={`entity-graph-verb is-${neighbor.direction}`}
          style={{
            left: `${neighbor.labelX}%`,
            top: `${neighbor.labelY}%`,
          }}
          aria-hidden="true"
        >
          {neighbor.verbs[0]}
          {neighbor.verbs.length > 1 ? ` +${neighbor.verbs.length - 1}` : ""}
        </span>
      ))}
      <div className="entity-graph-center" aria-hidden="true">
        <span className="entity-graph-center-dot" />
        <span className="entity-graph-center-name">{name}</span>
      </div>
      {placed.map((neighbor) => {
        const isIncoming = neighbor.direction === "incoming";
        const directionLabel = t(
          isIncoming ? "entityDetail.incoming" : "entityDetail.outgoing",
        );
        return (
          <button
            key={`node-${neighbor.entityId}-${neighbor.direction}`}
            type="button"
            className={`entity-graph-node ${isIncoming ? "is-left" : "is-right"}`}
            style={{ left: `${neighbor.x}%`, top: `${neighbor.y}%` }}
            onClick={() => onEntityClick(neighbor.entityId)}
            title={`${neighbor.verbs.join(", ")} — ${neighbor.name}`}
            aria-label={`${neighbor.name} (${neighbor.entityType}) · ${directionLabel} · ${neighbor.verbs.join(", ")}`}
          >
            <span className="entity-graph-node-dot" aria-hidden="true" />
            <span className="entity-graph-node-text">
              <span className="entity-graph-node-name">{neighbor.name}</span>
              <span className="entity-graph-node-type">{neighbor.entityType}</span>
            </span>
          </button>
        );
      })}
      {hiddenCount > 0 ? (
        <span className="entity-graph-more">
          {t("entityDetail.moreConnections", { count: hiddenCount })}
        </span>
      ) : null}
    </div>
  );
}

export function EntityConnections({
  name,
  relations,
  onEntityClick,
  onExpand,
}: EntityConnectionsProps) {
  const { t } = useTranslation();
  const neighbors = useMemo(() => {
    const byEntityAndDirection = new Map<string, GraphNeighbor>();
    for (const relation of relations) {
      const direction: GraphDirection = relation.direction === "incoming" ? "incoming" : "outgoing";
      const key = `${relation.entity_id}:${direction}`;
      const existing = byEntityAndDirection.get(key);
      if (existing) {
        if (!existing.verbs.includes(relation.relation_type)) {
          byEntityAndDirection.set(key, {
            ...existing,
            verbs: [...existing.verbs, relation.relation_type],
          });
        }
      } else {
        byEntityAndDirection.set(key, {
          entityId: relation.entity_id,
          name: relation.entity_name,
          entityType: relation.entity_type,
          verbs: [relation.relation_type],
          direction,
        });
      }
    }
    return Array.from(byEntityAndDirection.values()).sort((left, right) =>
      left.direction === right.direction
        ? left.name.localeCompare(right.name) || left.entityId.localeCompare(right.entityId)
        : left.direction.localeCompare(right.direction),
    );
  }, [relations]);
  const sortedRelations = useMemo(
    () =>
      [...relations].sort((left, right) =>
        left.direction === right.direction
          ? left.relation_type.localeCompare(right.relation_type)
          : left.direction === "outgoing"
            ? -1
            : 1,
      ),
    [relations],
  );
  const incoming = neighbors.filter((neighbor) => neighbor.direction === "incoming");
  const outgoing = neighbors.filter((neighbor) => neighbor.direction === "outgoing");
  const outgoingCount = Math.min(
    outgoing.length,
    GRAPH_NODE_CAP - Math.min(incoming.length, Math.ceil(GRAPH_NODE_CAP / 2)),
  );
  const shownNeighbors = [
    ...incoming.slice(0, GRAPH_NODE_CAP - outgoingCount),
    ...outgoing.slice(0, outgoingCount),
  ];
  const hiddenCount = neighbors.length - shownNeighbors.length;

  return (
    <section className="memory-detail-card" aria-labelledby="entity-connections-title">
      <div className="memory-detail-card-header">
        <h2 id="entity-connections-title" className="memory-detail-section-title">
          {t("entityDetail.connectionsTitle")}
        </h2>
        <div className="memory-detail-actions">
          {relations.length > 0 ? <span className="entity-count">{relations.length}</span> : null}
          {onExpand ? (
            <button
              type="button"
              onClick={onExpand}
              className="memory-detail-icon-button"
              aria-label={t("entityDetail.expandGraph")}
              title={t("entityDetail.expandGraph")}
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
                <path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3" />
              </svg>
            </button>
          ) : null}
        </div>
      </div>
      <div className="memory-detail-card-body">
        {relations.length === 0 ? (
          <p className="entity-empty">{t("entityDetail.emptyConnections")}</p>
        ) : (
          <>
            <EntityGraph
              name={name}
              neighbors={shownNeighbors}
              hiddenCount={hiddenCount}
              onEntityClick={onEntityClick}
            />
            <div
              className="entity-relation-list"
              role="group"
              aria-label={t("entityDetail.connectionsTitle")}
            >
              {sortedRelations.map((relation) => {
                const direction: GraphDirection =
                  relation.direction === "incoming" ? "incoming" : "outgoing";
                const directionLabel = t(
                  direction === "incoming" ? "entityDetail.incoming" : "entityDetail.outgoing",
                );
                return (
                  <button
                    key={relation.id}
                    type="button"
                    className="entity-relation-row"
                    onClick={() => onEntityClick(relation.entity_id)}
                    aria-label={`${relation.entity_name} (${relation.entity_type}) · ${directionLabel} · ${relation.relation_type}`}
                  >
                    <span className="entity-relation-verb">
                      {direction === "incoming"
                        ? `← ${relation.relation_type}`
                        : `${relation.relation_type} →`}
                    </span>
                    <span className="entity-relation-name">{relation.entity_name}</span>
                    <span className="entity-relation-type">{relation.entity_type}</span>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
