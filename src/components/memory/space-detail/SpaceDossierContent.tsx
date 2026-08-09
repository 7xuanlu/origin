// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import type { Entity, Page } from "../../../lib/tauri";
import type { SpaceDetailCopy } from "./copy";
import {
  KEY_ENTITY_LIMIT,
  formatLocalCalendarDate,
  pagesNeedingReview,
  recentlyRefinedPages,
  reviewReasonName,
  sortedKeyEntities,
} from "./model";

type SpaceDossierNavigation = {
  readonly onEntityClick: (entityId: string) => void;
  readonly onReviewAll?: () => void;
  readonly onSelectPage: (pageId: string) => void;
};

type SpaceDossierContentProps = {
  readonly copy: SpaceDetailCopy;
  readonly entities: readonly Entity[];
  readonly locale: string;
  readonly navigation: SpaceDossierNavigation;
  readonly pages: readonly Page[];
};

function PageIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M12 2 2 7l10 5 10-5-10-5ZM2 12l10 5 10-5M2 17l10 5 10-5" />
    </svg>
  );
}

export function SpaceDossierContent({
  copy,
  entities,
  locale,
  navigation,
  pages,
}: SpaceDossierContentProps) {
  const [showAllEntities, setShowAllEntities] = useState(false);
  const recentPages = recentlyRefinedPages(pages);
  const reviewPages = pagesNeedingReview(pages);
  const keyEntities = sortedKeyEntities(entities);
  const visibleEntities = showAllEntities
    ? keyEntities
    : keyEntities.slice(0, KEY_ENTITY_LIMIT);

  return (
    <div className="space-dossier-grid">
      <section aria-label={copy.recentlyRefined} className="space-dossier-recent">
        <h2>{copy.recentlyRefined}</h2>
        {recentPages.length === 0 ? (
          <p className="space-dossier-empty">{copy.noPages}</p>
        ) : (
          <div className="space-dossier-page-list">
            {recentPages.map((page) => {
              const timestamp = Date.parse(page.last_modified);
              return (
                <button key={page.id} onClick={() => navigation.onSelectPage(page.id)} type="button">
                  <PageIcon />
                  <span className="space-dossier-page-title">{page.title}</span>
                  <span className="space-dossier-page-meta">
                    <span>{copy.sourceCount(page.source_memory_ids.length)}</span>
                    <time dateTime={page.last_modified}>{formatLocalCalendarDate(timestamp, locale)}</time>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </section>

      <aside className="space-dossier-rail">
        <section aria-label={copy.needsReview} className="space-dossier-review">
          <h2>{copy.needsReview}</h2>
          {reviewPages.length === 0 ? (
            <p className="space-dossier-empty">{copy.noReview}</p>
          ) : (
            <div className="space-dossier-review-list">
              {reviewPages.map((page) => (
                <button key={page.id} onClick={() => navigation.onSelectPage(page.id)} type="button">
                  <PageIcon />
                  <span>{page.title}</span>
                  <small>{copy.reasons[reviewReasonName(page)]}</small>
                </button>
              ))}
            </div>
          )}
          <button
            className="space-dossier-text-action space-dossier-text-action-review"
            onClick={navigation.onReviewAll}
            type="button"
          >
            {copy.reviewAll}
          </button>
        </section>

        <section aria-label={copy.keyEntities} className="space-dossier-entities">
          <h2>{copy.keyEntities}</h2>
          {visibleEntities.length === 0 ? (
            <p className="space-dossier-empty">{copy.noEntities}</p>
          ) : (
            <div className="space-dossier-entity-list">
              {visibleEntities.map((entity) => (
                <button key={entity.id} onClick={() => navigation.onEntityClick(entity.id)} type="button">
                  {entity.name}
                </button>
              ))}
            </div>
          )}
          {keyEntities.length > KEY_ENTITY_LIMIT && (
            <button className="space-dossier-text-action" onClick={() => setShowAllEntities((current) => !current)} type="button">
              {showAllEntities
                ? copy.showLess
                : copy.viewAllEntities(keyEntities.length)}
            </button>
          )}
        </section>
      </aside>
    </div>
  );
}
