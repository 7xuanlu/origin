// SPDX-License-Identifier: AGPL-3.0-only
import type { SpaceDetailCopy } from "./copy";

const TEST_COPY: Readonly<Record<string, string>> = {
  "spaceDetail.backToSpaces": "Spaces",
  "spaceDetail.cancel": "Cancel",
  "spaceDetail.confirmDeleteSpace": "Confirm delete space",
  "spaceDetail.deleteSpace": "Delete space",
  "spaceDetail.deleteWarning": "Memories keep their current space tag after this daemon removes the space.",
  "spaceDetail.descriptionPlaceholder": "Add a description...",
  "spaceDetail.discard": "Discard",
  "spaceDetail.editDescription": "Edit description",
  "spaceDetail.keep": "Keep",
  "spaceDetail.keyEntities": "Key entities",
  "spaceDetail.loadError": "Could not open space",
  "spaceDetail.loading": "Loading space",
  "spaceDetail.metrics.entities": "Entities",
  "spaceDetail.metrics.memories": "Memories",
  "spaceDetail.metrics.pages": "Pages",
  "spaceDetail.metrics.updated": "Updated",
  "spaceDetail.mutationError": "Could not save this change.",
  "spaceDetail.needsReview": "Needs review",
  "spaceDetail.noEntities": "No key entities yet",
  "spaceDetail.noMemories": "No memories in this space yet",
  "spaceDetail.noPages": "No refined pages yet",
  "spaceDetail.noReview": "No pages need review",
  "spaceDetail.notFound": "Space not found",
  "spaceDetail.notUpdated": "Not updated",
  "spaceDetail.rawMemories": "Raw memories",
  "spaceDetail.reasons.needsRefresh": "Needs page refresh",
  "spaceDetail.reasons.sourceConflict": "Source conflict",
  "spaceDetail.reasons.sourceUpdated": "New sources waiting",
  "spaceDetail.recentlyRefined": "Recently refined",
  "spaceDetail.relatedLoadError": "Some space details could not be loaded.",
  "spaceDetail.reviewAll": "Review all",
  "spaceDetail.save": "Save",
  "spaceDetail.saveShortcut": "Cmd+Enter",
  "spaceDetail.showLess": "Show less",
  "spaceDetail.showingLatest": "Showing the latest {{shown}} of {{total}} memories",
  "spaceDetail.sourceCount_one": "{{count}} source",
  "spaceDetail.sourceCount_other": "{{count}} sources",
  "spaceDetail.suggestedLabel": "Auto-created by an agent",
  "spaceDetail.viewAllEntities": "View all {{count}}",
};

type TranslationOptions = Readonly<Record<string, string | number>>;

export function translateSpaceDetailForTest(
  key: string,
  options: TranslationOptions = {},
): string {
  const pluralKey = typeof options.count === "number"
    ? `${key}_${options.count === 1 ? "one" : "other"}`
    : key;
  const template = TEST_COPY[pluralKey] ?? TEST_COPY[key] ?? key;
  return Object.entries(options).reduce(
    (copy, [name, value]) => copy.replaceAll(`{{${name}}}`, String(value)),
    template,
  );
}

export const SPACE_DETAIL_TEST_COPY: SpaceDetailCopy = {
  backToSpaces: translateSpaceDetailForTest("spaceDetail.backToSpaces"),
  cancel: translateSpaceDetailForTest("spaceDetail.cancel"),
  confirmDeleteSpace: translateSpaceDetailForTest("spaceDetail.confirmDeleteSpace"),
  deleteSpace: translateSpaceDetailForTest("spaceDetail.deleteSpace"),
  deleteWarning: translateSpaceDetailForTest("spaceDetail.deleteWarning"),
  descriptionPlaceholder: translateSpaceDetailForTest("spaceDetail.descriptionPlaceholder"),
  discard: translateSpaceDetailForTest("spaceDetail.discard"),
  editDescription: translateSpaceDetailForTest("spaceDetail.editDescription"),
  keep: translateSpaceDetailForTest("spaceDetail.keep"),
  keyEntities: translateSpaceDetailForTest("spaceDetail.keyEntities"),
  loadError: translateSpaceDetailForTest("spaceDetail.loadError"),
  loading: translateSpaceDetailForTest("spaceDetail.loading"),
  metrics: {
    entities: translateSpaceDetailForTest("spaceDetail.metrics.entities"),
    memories: translateSpaceDetailForTest("spaceDetail.metrics.memories"),
    pages: translateSpaceDetailForTest("spaceDetail.metrics.pages"),
    updated: translateSpaceDetailForTest("spaceDetail.metrics.updated"),
  },
  mutationError: translateSpaceDetailForTest("spaceDetail.mutationError"),
  needsReview: translateSpaceDetailForTest("spaceDetail.needsReview"),
  noEntities: translateSpaceDetailForTest("spaceDetail.noEntities"),
  noMemories: translateSpaceDetailForTest("spaceDetail.noMemories"),
  noPages: translateSpaceDetailForTest("spaceDetail.noPages"),
  noReview: translateSpaceDetailForTest("spaceDetail.noReview"),
  notFound: translateSpaceDetailForTest("spaceDetail.notFound"),
  notUpdated: translateSpaceDetailForTest("spaceDetail.notUpdated"),
  rawMemories: translateSpaceDetailForTest("spaceDetail.rawMemories"),
  reasons: {
    needsRefresh: translateSpaceDetailForTest("spaceDetail.reasons.needsRefresh"),
    sourceConflict: translateSpaceDetailForTest("spaceDetail.reasons.sourceConflict"),
    sourceUpdated: translateSpaceDetailForTest("spaceDetail.reasons.sourceUpdated"),
  },
  recentlyRefined: translateSpaceDetailForTest("spaceDetail.recentlyRefined"),
  relatedLoadError: translateSpaceDetailForTest("spaceDetail.relatedLoadError"),
  reviewAll: translateSpaceDetailForTest("spaceDetail.reviewAll"),
  save: translateSpaceDetailForTest("spaceDetail.save"),
  saveShortcut: translateSpaceDetailForTest("spaceDetail.saveShortcut"),
  showLess: translateSpaceDetailForTest("spaceDetail.showLess"),
  showingLatest: (shown, total) => translateSpaceDetailForTest("spaceDetail.showingLatest", { shown, total }),
  sourceCount: (count) => translateSpaceDetailForTest("spaceDetail.sourceCount", { count }),
  suggestedLabel: translateSpaceDetailForTest("spaceDetail.suggestedLabel"),
  viewAllEntities: (count) => translateSpaceDetailForTest("spaceDetail.viewAllEntities", { count }),
};
