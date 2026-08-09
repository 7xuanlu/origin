// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { FACET_COLORS, search, type Entity } from "../../../lib/tauri";
import { MetadataRow, RailPanelTitle } from "../MemoryDetailPrimitives";
import {
  formatAbsoluteTimestamp,
  formatCalendarDate,
  formatConfidence,
  formatRelativeEntityTime,
} from "./formatEntityMetadata";

type EntityContextRailProps = {
  readonly entity: Entity;
  readonly locale: string;
  readonly onMemoryClick?: (sourceId: string) => void;
};

export function EntityContextRail({ entity, locale, onMemoryClick }: EntityContextRailProps) {
  const { t } = useTranslation();
  const space = entity.space ?? entity.domain;
  const confidence = formatConfidence(entity.confidence, 2);
  const createdDate = formatCalendarDate(entity.created_at, locale);
  const createdAbsolute = formatAbsoluteTimestamp(entity.created_at);
  const updatedRelative = formatRelativeEntityTime(entity.updated_at, locale);
  const updatedAbsolute = formatAbsoluteTimestamp(entity.updated_at);
  const { data: linkedMemories = [] } = useQuery({
    queryKey: ["entity-linked-memories", entity.id, entity.name],
    queryFn: async () => {
      const results = await search(entity.name, 10, "memory");
      return results
        .filter((result) => result.entity_id === entity.id || result.score > 0.7)
        .slice(0, 8);
    },
    enabled: Boolean(entity.name),
    staleTime: 30_000,
  });

  return (
    <aside
      className="memory-detail-rail page-detail-rail"
      aria-label={t("entityDetail.contextLabel")}
    >
      <section className="memory-detail-rail-section memory-detail-metadata-panel">
        <RailPanelTitle>{t("entityDetail.detailsTitle")}</RailPanelTitle>
        <div className="memory-detail-metadata-list">
          <MetadataRow label={t("entityDetail.typeLabel")}>
            <span className="entity-meta-mono">{entity.entity_type}</span>
          </MetadataRow>
          {space ? (
            <MetadataRow label={t("entityDetail.spaceLabel")}>
              <span className="entity-meta-mono">{space}</span>
            </MetadataRow>
          ) : null}
          <MetadataRow label={t("entityDetail.statusLabel")}>
            <span className={`memory-detail-chip ${entity.confirmed ? "success" : "warning"}`}>
              {entity.confirmed ? t("entityDetail.confirmed") : t("entityDetail.unconfirmed")}
            </span>
          </MetadataRow>
          {confidence ? (
            <MetadataRow label={t("entityDetail.confidenceLabel")}>
              <span className="entity-meta-mono">{confidence}</span>
            </MetadataRow>
          ) : null}
          {entity.source_agent ? (
            <MetadataRow label={t("entityDetail.sourceLabel")}>
              <span className="entity-meta-mono">{entity.source_agent}</span>
            </MetadataRow>
          ) : null}
          {createdDate ? (
            <MetadataRow label={t("entityDetail.firstSeen")}>
              <span className="entity-meta-mono" title={createdAbsolute ?? undefined}>
                {createdDate}
              </span>
            </MetadataRow>
          ) : null}
          {updatedRelative ? (
            <MetadataRow label={t("entityDetail.updatedLabel")}>
              <span className="entity-meta-mono" title={updatedAbsolute ?? undefined}>
                {updatedRelative}
              </span>
            </MetadataRow>
          ) : null}
        </div>
      </section>
      {linkedMemories.length > 0 ? (
        <section className="memory-detail-rail-section memory-detail-secondary-panel">
          <div className="memory-detail-panel-heading">
            <RailPanelTitle>{t("entityDetail.appearsTitle")}</RailPanelTitle>
            <span className="entity-appears-hint">{t("entityDetail.appearsHint")}</span>
          </div>
          <div className="memory-detail-related-grid">
            {linkedMemories.map((memory) => {
              const facet = memory.memory_type ?? null;
              const facetClass = facet ? FACET_COLORS[facet] : null;
              return (
                <button
                  key={memory.id}
                  type="button"
                  className="memory-detail-related-card"
                  onClick={() => onMemoryClick?.(memory.source_id)}
                >
                  <span className="memory-detail-context-row-body">
                    <span className="memory-detail-related-copy line-clamp-2">
                      {memory.content}
                    </span>
                    <span className="memory-detail-related-meta">
                      {facet && facetClass ? (
                        <span className={`memory-detail-related-facet border ${facetClass}`}>
                          {facet}
                        </span>
                      ) : null}
                      {memory.is_archived ? <span>{t("entityDetail.archived")}</span> : null}
                    </span>
                  </span>
                  <svg
                    className="memory-detail-related-chevron"
                    aria-hidden="true"
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                </button>
              );
            })}
          </div>
        </section>
      ) : null}
    </aside>
  );
}
