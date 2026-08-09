// SPDX-License-Identifier: AGPL-3.0-only
export type SpaceDetailCopy = {
  readonly backToSpaces: string;
  readonly cancel: string;
  readonly confirmDeleteSpace: string;
  readonly deleteSpace: string;
  readonly deleteWarning: string;
  readonly descriptionPlaceholder: string;
  readonly discard: string;
  readonly editDescription: string;
  readonly keep: string;
  readonly keyEntities: string;
  readonly loadError: string;
  readonly loading: string;
  readonly metrics: {
    readonly entities: string;
    readonly memories: string;
    readonly pages: string;
    readonly updated: string;
  };
  readonly mutationError: string;
  readonly needsReview: string;
  readonly noEntities: string;
  readonly noMemories: string;
  readonly noPages: string;
  readonly noReview: string;
  readonly notFound: string;
  readonly notUpdated: string;
  readonly rawMemories: string;
  readonly reasons: {
    readonly needsRefresh: string;
    readonly sourceConflict: string;
    readonly sourceUpdated: string;
  };
  readonly recentlyRefined: string;
  readonly relatedLoadError: string;
  readonly reviewAll: string;
  readonly save: string;
  readonly saveShortcut: string;
  readonly showLess: string;
  readonly showingLatest: (shown: number, total: number) => string;
  readonly sourceCount: (count: number) => string;
  readonly suggestedLabel: string;
  readonly viewAllEntities: (count: number) => string;
};

export const SPACE_DETAIL_KEY_COPY: SpaceDetailCopy = {
  backToSpaces: "spaceDetail.backToSpaces",
  cancel: "spaceDetail.cancel",
  confirmDeleteSpace: "spaceDetail.confirmDeleteSpace",
  deleteSpace: "spaceDetail.deleteSpace",
  deleteWarning: "spaceDetail.deleteWarning",
  descriptionPlaceholder: "spaceDetail.descriptionPlaceholder",
  discard: "spaceDetail.discard",
  editDescription: "spaceDetail.editDescription",
  keep: "spaceDetail.keep",
  keyEntities: "spaceDetail.keyEntities",
  loadError: "spaceDetail.loadError",
  loading: "spaceDetail.loading",
  metrics: {
    entities: "spaceDetail.metrics.entities",
    memories: "spaceDetail.metrics.memories",
    pages: "spaceDetail.metrics.pages",
    updated: "spaceDetail.metrics.updated",
  },
  mutationError: "spaceDetail.mutationError",
  needsReview: "spaceDetail.needsReview",
  noEntities: "spaceDetail.noEntities",
  noMemories: "spaceDetail.noMemories",
  noPages: "spaceDetail.noPages",
  noReview: "spaceDetail.noReview",
  notFound: "spaceDetail.notFound",
  notUpdated: "spaceDetail.notUpdated",
  rawMemories: "spaceDetail.rawMemories",
  reasons: {
    needsRefresh: "spaceDetail.reasons.needsRefresh",
    sourceConflict: "spaceDetail.reasons.sourceConflict",
    sourceUpdated: "spaceDetail.reasons.sourceUpdated",
  },
  recentlyRefined: "spaceDetail.recentlyRefined",
  relatedLoadError: "spaceDetail.relatedLoadError",
  reviewAll: "spaceDetail.reviewAll",
  save: "spaceDetail.save",
  saveShortcut: "spaceDetail.saveShortcut",
  showLess: "spaceDetail.showLess",
  showingLatest: () => "spaceDetail.showingLatest",
  sourceCount: () => "spaceDetail.sourceCount",
  suggestedLabel: "spaceDetail.suggestedLabel",
  viewAllEntities: () => "spaceDetail.viewAllEntities",
};
