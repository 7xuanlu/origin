import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  confirmSpace,
  createSpace,
  deleteSpace,
  listSpaces,
  reorderSpace,
  toggleSpaceStarred,
  updateSpace,
  type Space,
} from "../../../lib/tauri";
import { listAllActivePages } from "../pages/listAllPages";
import { ConfirmedSpaces } from "./ConfirmedSpaces";
import { SpaceEditor } from "./SpaceEditor";
import { SuggestedSpaces } from "./SuggestedSpaces";
import { filterSpaces, sortConfirmedSpaces, sortSuggestedSpaces } from "./spaceHelpers";
import type { SpaceRowAction, SpacesOverviewProps } from "./spacesTypes";
import "../pages/pageActions.css";
import "./spaces.css";
import "./spacesInventory.css";

const DEPENDENT_QUERY_KEYS = [
  "spaces",
  "spaces-page-counts",
  "pages",
  "sidebar-space-page-counts",
  "space",
  "space-pages",
  "space-memories",
  "space-entities",
  "memories",
] as const;

function assertNever(value: never): never {
  throw new TypeError(`Unsupported spaces action: ${String(value)}`);
}

async function runAction(action: SpaceRowAction): Promise<void> {
  switch (action.kind) {
    case "confirm":
      await confirmSpace(action.space.name);
      return;
    case "create":
      await createSpace(action.value.name, action.value.description);
      return;
    case "delete":
      await deleteSpace(action.space.name);
      return;
    case "rename":
      await updateSpace(action.space.name, action.value.name, action.space.description ?? undefined);
      return;
    case "reorder":
      await reorderSpace(action.space.name, action.target.sort_order);
      return;
    case "star":
      await toggleSpaceStarred(action.space.name);
      return;
    default:
      return assertNever(action);
  }
}

function actionKey(action: SpaceRowAction): string {
  return action.kind === "create" ? "create" : action.space.id;
}

export function SpacesOverview(props: SpacesOverviewProps) {
  const queryClient = useQueryClient();
  const [filter, setFilter] = useState("");
  const [creating, setCreating] = useState(false);
  const [lastGood, setLastGood] = useState<readonly Space[] | null>(null);
  const [pendingIds, setPendingIds] = useState<readonly string[]>([]);
  const [mutationFailed, setMutationFailed] = useState(false);
  const pendingRef = useRef(new Set<string>());
  const query = useQuery({ queryKey: ["spaces"], queryFn: listSpaces });
  const pagesQuery = useQuery({
    queryKey: ["spaces-page-counts"],
    queryFn: listAllActivePages,
  });
  const pageCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const page of pagesQuery.data ?? []) {
      const spaceName = page.space === undefined ? page.domain?.trim() : page.space?.trim();
      if (spaceName === undefined || spaceName.length === 0) continue;
      const key = spaceName.toLocaleLowerCase();
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
  }, [pagesQuery.data]);

  useEffect(() => {
    if (query.data !== undefined) setLastGood(query.data);
  }, [query.data]);

  useEffect(() => {
    if (props.createIntent !== true) return;
    setCreating(true);
    props.onCreateIntentHandled?.();
  }, [props.createIntent, props.onCreateIntentHandled]);

  const execute = async (action: SpaceRowAction, key: string): Promise<boolean> => {
    try {
      await runAction(action);
      switch (action.kind) {
        case "create":
          setCreating(false);
          break;
        case "delete":
          props.onSpaceDeleted?.(action.space.id);
          break;
        case "rename":
          props.onSpaceRenamed?.({ id: action.space.id, name: action.value.name });
          break;
        case "confirm":
        case "reorder":
        case "star":
          break;
        default:
          assertNever(action);
      }
      await Promise.all(DEPENDENT_QUERY_KEYS.map((key) => queryClient.invalidateQueries({ queryKey: [key] })));
      return true;
    } catch {
      setMutationFailed(true);
      return false;
    } finally {
      pendingRef.current.delete(key);
      setPendingIds(Array.from(pendingRef.current));
    }
  };

  const submit = (action: SpaceRowAction): Promise<boolean> => {
    const key = actionKey(action);
    if (pendingRef.current.has(key)) return Promise.resolve(false);
    pendingRef.current.add(key);
    setPendingIds(Array.from(pendingRef.current));
    setMutationFailed(false);
    return execute(action, key);
  };

  const allSpaces = query.data ?? lastGood ?? [];
  const suggested = filterSpaces(sortSuggestedSpaces(allSpaces), filter);
  const confirmed = filterSpaces(sortConfirmedSpaces(allSpaces), filter);
  const hasFilter = filter.trim().length > 0;
  const noResults = hasFilter && suggested.length === 0 && confirmed.length === 0;
  const pendingCreate = pendingIds.includes("create");

  return (
    <div className="spaces-overview">
      <header className="spaces-overview-header">
        <h1>{props.labels.title}</h1>
        <button type="button" className="page-create-action spaces-new-action" onClick={() => setCreating(true)}>
          {props.labels.newSpace}
        </button>
      </header>
      {creating ? (
        <SpaceEditor
          labels={props.labels}
          spaces={allSpaces}
          showDescription
          submitLabel={props.labels.create}
          pending={pendingCreate}
          onCancel={() => setCreating(false)}
          onSubmit={(value) => submit({ kind: "create", value })}
        />
      ) : null}
      {query.isPending && lastGood === null ? <p className="spaces-state">{props.labels.loading}</p> : null}
      {query.isError ? (
        <div className="spaces-error" role="alert">
          <span>{props.labels.loadError}</span>
          <button className="spaces-quiet-action" onClick={() => void query.refetch()}>{props.labels.retry}</button>
        </div>
      ) : null}
      {mutationFailed ? <p className="spaces-error" role="alert">{props.labels.mutationError}</p> : null}
      {query.isPending && lastGood === null ? null : (
        <>
          <SuggestedSpaces
            spaces={suggested}
            labels={props.labels}
            pendingIds={pendingIds}
            onSelect={props.onSelectSpace}
            onKeep={(space) => submit({ kind: "confirm", space })}
            onDiscard={(space) => submit({ kind: "delete", space })}
          />
          <ConfirmedSpaces
            spaces={confirmed}
            allSpaces={allSpaces}
            labels={props.labels}
            filter={filter}
            onFilterChange={setFilter}
            noResults={noResults}
            pageCounts={pageCounts}
            pendingIds={pendingIds}
            onSelect={props.onSelectSpace}
            onStar={(space) => submit({ kind: "star", space })}
            onRename={(space, value) => submit({ kind: "rename", space, value })}
            onReorder={(space, target) => submit({ kind: "reorder", space, target })}
            onDelete={(space) => submit({ kind: "delete", space })}
          />
        </>
      )}
    </div>
  );
}
