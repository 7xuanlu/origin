// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  confirmSpace,
  deleteSpace,
  getSpace,
  listEntities,
  listMemoriesRich,
  listPages,
  updateSpace,
  type Space,
} from "../../lib/tauri";
import { RawMemoriesSection } from "./space-detail/RawMemoriesSection";
import { SpaceDossierContent } from "./space-detail/SpaceDossierContent";
import { SpaceDossierHeader } from "./space-detail/SpaceDossierHeader";
import { SPACE_DETAIL_KEY_COPY, type SpaceDetailCopy } from "./space-detail/copy";
import {
  MEMORY_FETCH_LIMIT,
  PAGE_FETCH_LIMIT,
  formatLocalCalendarDate,
  latestDossierUpdate,
  pageCountLabel,
} from "./space-detail/model";
import "./space-detail/space-detail-header.css";
import "./space-detail/space-detail.css";
import "./pages/pageActions.css";

export type SpaceDetailProps = {
  readonly copy?: SpaceDetailCopy;
  readonly onBack: () => void;
  readonly onEntityClick: (entityId: string) => void;
  readonly onSpaceDeleted?: (spaceId: string) => void;
  readonly onSpaceLoaded?: (space: Space) => void;
  readonly onSpaceRenamed?: (space: Pick<Space, "id" | "name">) => void;
  readonly onReviewAll?: () => void;
  readonly onCreatePage?: (space: string) => void;
  readonly onSelectMemory: (sourceId: string) => void;
  readonly onSelectPage: (pageId: string) => void;
  readonly spaceName: string;
};

export default function SpaceDetail({
  copy = SPACE_DETAIL_KEY_COPY,
  onBack,
  onEntityClick,
  onSpaceDeleted,
  onSpaceLoaded,
  onSpaceRenamed,
  onReviewAll,
  onCreatePage,
  onSelectMemory,
  onSelectPage,
  spaceName,
}: SpaceDetailProps) {
  const { i18n } = useTranslation();
  const queryClient = useQueryClient();
  const spaceQuery = useQuery({
    queryKey: ["space", spaceName],
    queryFn: () => getSpace(spaceName),
    refetchInterval: 5_000,
  });
  const memoriesQuery = useQuery({
    queryKey: ["space-memories", spaceName],
    queryFn: () => listMemoriesRich(spaceName, undefined, undefined, MEMORY_FETCH_LIMIT),
    refetchInterval: 5_000,
  });
  const entitiesQuery = useQuery({
    queryKey: ["space-entities", spaceName],
    queryFn: () => listEntities(undefined, spaceName),
    refetchInterval: 10_000,
  });
  const pagesQuery = useQuery({
    queryKey: ["space-pages", spaceName],
    queryFn: () => listPages("active", spaceName, PAGE_FETCH_LIMIT),
    refetchInterval: 10_000,
  });
  const reportedSpaceIdRef = useRef<string | null>(null);

  useEffect(() => {
    const space = spaceQuery.data;
    if (!space || reportedSpaceIdRef.current === space.id) return;
    reportedSpaceIdRef.current = space.id;
    onSpaceLoaded?.(space);
  }, [onSpaceLoaded, spaceQuery.data]);

  const invalidateSpaceQueries = () => {
    queryClient.invalidateQueries({ queryKey: ["space", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["spaces"] });
    queryClient.invalidateQueries({ queryKey: ["space-memories", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["space-entities", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["space-pages", spaceName] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
  };

  const renameMutation = useMutation({
    mutationFn: ({ newName, description }: { readonly newName: string; readonly description?: string }) =>
      updateSpace(spaceName, newName, description),
    onSuccess: (updatedSpace, variables) => {
      invalidateSpaceQueries();
      if (variables.newName !== spaceName) {
        onSpaceRenamed?.({ id: updatedSpace.id, name: updatedSpace.name });
        onBack();
      }
    },
  });
  const deleteMutation = useMutation({
    mutationFn: () => deleteSpace(spaceName),
    onSuccess: () => {
      invalidateSpaceQueries();
      if (spaceQuery.data) onSpaceDeleted?.(spaceQuery.data.id);
      onBack();
    },
  });
  const confirmMutation = useMutation({
    mutationFn: () => confirmSpace(spaceName),
    onSuccess: invalidateSpaceQueries,
  });

  if (spaceQuery.isPending) {
    return <p className="space-dossier-state" role="status">{copy.loading}</p>;
  }
  if (spaceQuery.isError) {
    return <p className="space-dossier-state" role="alert">{copy.loadError}</p>;
  }
  if (!spaceQuery.data) {
    return <p className="space-dossier-state">{copy.notFound}</p>;
  }

  const space = spaceQuery.data;
  const memories = memoriesQuery.data ?? [];
  const entities = entitiesQuery.data ?? [];
  const pages = pagesQuery.data ?? [];
  const latestUpdate = latestDossierUpdate(space.updated_at, pages);
  const updatedLabel = latestUpdate === null
    ? copy.notUpdated
    : formatLocalCalendarDate(latestUpdate, i18n.language);
  const mutationError = renameMutation.isError || deleteMutation.isError || confirmMutation.isError;
  const relatedLoadError = memoriesQuery.isError || entitiesQuery.isError || pagesQuery.isError;

  return (
    <article className="space-dossier">
      <SpaceDossierHeader
        actions={{
          onBack,
          onDelete: () => deleteMutation.mutate(),
          onKeep: () => confirmMutation.mutate(),
          onCreatePage: () => onCreatePage?.(space.name),
          onSaveIdentity: ({ name, description }) => renameMutation.mutate({
            newName: name,
            description,
          }),
        }}
        copy={copy}
        mutationError={mutationError}
        pageCount={pageCountLabel(pages.length, i18n.language)}
        space={space}
        updatedLabel={updatedLabel}
      />

      {relatedLoadError && <p className="space-dossier-error" role="alert">{copy.relatedLoadError}</p>}

      <SpaceDossierContent
        copy={copy}
        entities={entities}
        locale={i18n.language}
        navigation={{ onEntityClick, onSelectPage, ...(onReviewAll ? { onReviewAll } : {}) }}
        pages={pages}
      />
      <RawMemoriesSection
        copy={copy}
        memories={memories}
        onSelectMemory={onSelectMemory}
        totalMemoryCount={space.memory_count}
      />
    </article>
  );
}
