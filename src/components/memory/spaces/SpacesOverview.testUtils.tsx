import type { ReactElement } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { Space } from "../../../lib/tauri";
import { SpacesOverview } from "./SpacesOverview";
import type { SpacesOverviewLabels, SpacesOverviewProps } from "./spacesTypes";

export const labels: SpacesOverviewLabels = {
  title: "Spaces",
  newSpace: "New space",
  filterLabel: "Filter spaces",
  filterPlaceholder: "Filter by name or description",
  suggestedHeading: "Suggested",
  confirmedHeading: "All spaces",
  pages: "Pages",
  memories: "Memories",
  updated: "Updated",
  nameLabel: "Name",
  descriptionLabel: "Description",
  descriptionPlaceholder: "Optional description",
  create: "Create",
  save: "Save",
  cancel: "Cancel",
  keep: "Keep",
  discard: "Discard",
  star: "Star",
  unstar: "Unstar",
  rename: "Rename",
  moveUp: "Move up",
  moveDown: "Move down",
  delete: "Delete",
  confirmDelete: "Confirm delete",
  actionsFor: (name) => `Actions for ${name}`,
  dragSpace: (name) => `Drag ${name}`,
  loading: "Loading spaces",
  loadError: "Spaces could not be loaded",
  mutationError: "The change could not be saved",
  retry: "Retry",
  noSuggestions: "No suggested spaces",
  noConfirmed: "No confirmed spaces",
  noResults: "No spaces match this filter",
  duplicateName: "A space with this name already exists",
  nameRequired: "Enter a space name",
};

export function makeSpace(overrides: Partial<Space> = {}): Space {
  return {
    id: "space-work",
    name: "Work",
    description: "Projects and planning",
    suggested: false,
    starred: false,
    sort_order: 10,
    memory_count: 4,
    entity_count: 2,
    created_at: 100,
    updated_at: 200,
    ...overrides,
  };
}

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
}

export function renderOverview(
  overrides: Partial<SpacesOverviewProps> = {},
  queryClient = createQueryClient(),
): ReturnType<typeof render> & { readonly queryClient: QueryClient } {
  const props: SpacesOverviewProps = {
    labels,
    onSelectSpace: () => undefined,
    ...overrides,
  };
  const ui: ReactElement = (
    <QueryClientProvider client={queryClient}>
      <SpacesOverview {...props} />
    </QueryClientProvider>
  );
  return { ...render(ui), queryClient };
}
