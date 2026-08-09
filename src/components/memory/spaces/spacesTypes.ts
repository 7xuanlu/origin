import type { Space } from "../../../lib/tauri";

export type MoveDirection = "up" | "down";

export type SpacesOverviewLabels = {
  readonly title: string;
  readonly newSpace: string;
  readonly filterLabel: string;
  readonly filterPlaceholder: string;
  readonly suggestedHeading: string;
  readonly confirmedHeading: string;
  readonly pages: string;
  readonly memories: string;
  readonly updated: string;
  readonly nameLabel: string;
  readonly descriptionLabel: string;
  readonly descriptionPlaceholder: string;
  readonly create: string;
  readonly save: string;
  readonly cancel: string;
  readonly keep: string;
  readonly discard: string;
  readonly star: string;
  readonly unstar: string;
  readonly rename: string;
  readonly moveUp: string;
  readonly moveDown: string;
  readonly delete: string;
  readonly confirmDelete: string;
  readonly actionsFor: (name: string) => string;
  readonly dragSpace: (name: string) => string;
  readonly loading: string;
  readonly loadError: string;
  readonly mutationError: string;
  readonly retry: string;
  readonly noSuggestions: string;
  readonly noConfirmed: string;
  readonly noResults: string;
  readonly duplicateName: string;
  readonly nameRequired: string;
};

export type SpacesOverviewProps = {
  readonly labels: SpacesOverviewLabels;
  readonly onSelectSpace: (spaceName: string) => void;
  readonly createIntent?: boolean;
  readonly onCreateIntentHandled?: () => void;
  readonly onSpaceDeleted?: (spaceId: string) => void;
  readonly onSpaceRenamed?: (space: Pick<Space, "id" | "name">) => void;
};

export type SpaceEditorValue = {
  readonly name: string;
  readonly description?: string;
};

export type SpaceRowAction =
  | { readonly kind: "confirm"; readonly space: Space }
  | { readonly kind: "create"; readonly value: SpaceEditorValue }
  | { readonly kind: "delete"; readonly space: Space }
  | { readonly kind: "rename"; readonly space: Space; readonly value: SpaceEditorValue }
  | { readonly kind: "reorder"; readonly space: Space; readonly target: Space }
  | { readonly kind: "star"; readonly space: Space };
