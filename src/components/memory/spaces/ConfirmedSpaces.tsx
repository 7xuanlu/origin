import { useCallback, useEffect, useRef } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Space } from "../../../lib/tauri";
import { canReorderTogether, findReorderTarget } from "./spaceHelpers";
import { SpaceRow } from "./SpaceRow";
import type { SpacesOverviewLabels, SpaceEditorValue } from "./spacesTypes";

type ConfirmedSpacesProps = {
  readonly spaces: readonly Space[];
  readonly allSpaces: readonly Space[];
  readonly labels: SpacesOverviewLabels;
  readonly filter: string;
  readonly onFilterChange: (value: string) => void;
  readonly noResults: boolean;
  readonly pageCounts: ReadonlyMap<string, number>;
  readonly pendingIds: readonly string[];
  readonly onSelect: (name: string) => void;
  readonly onStar: (space: Space) => void;
  readonly onRename: (space: Space, value: SpaceEditorValue) => Promise<boolean>;
  readonly onReorder: (space: Space, target: Space) => void;
  readonly onDelete: (space: Space) => void;
};

type ActivePointerDrag = {
  readonly source: Space;
  readonly pointerId: number;
};

export function ConfirmedSpaces(props: ConfirmedSpacesProps) {
  const activeDrag = useRef<ActivePointerDrag | null>(null);

  const cancelPointerMove = useCallback((pointerId: number) => {
    if (activeDrag.current?.pointerId === pointerId) activeDrag.current = null;
  }, []);

  useEffect(() => {
    const handlePointerEnd = (event: PointerEvent) => cancelPointerMove(event.pointerId);
    window.addEventListener("pointerup", handlePointerEnd);
    window.addEventListener("pointercancel", handlePointerEnd);
    return () => {
      window.removeEventListener("pointerup", handlePointerEnd);
      window.removeEventListener("pointercancel", handlePointerEnd);
      activeDrag.current = null;
    };
  }, [cancelPointerMove]);

  const requestMove = (space: Space, direction: "up" | "down") => {
    const target = findReorderTarget(props.spaces, space.id, direction);
    if (target !== null) props.onReorder(space, target);
  };

  const finishPointerMove = (event: ReactPointerEvent<HTMLDivElement>, target: Space) => {
    const drag = activeDrag.current;
    if (drag === null || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    activeDrag.current = null;
    if (canReorderTogether(drag.source, target)) props.onReorder(drag.source, target);
  };

  return (
    <section className="spaces-section spaces-confirmed-section" aria-label={props.labels.confirmedHeading}>
      <div className="spaces-inventory-header">
        <label className="spaces-filter">
          <span>{props.labels.filterLabel}</span>
          <input
            type="search"
            value={props.filter}
            placeholder={props.labels.filterPlaceholder}
            onChange={(event) => props.onFilterChange(event.currentTarget.value)}
          />
        </label>
      </div>
      {props.noResults ? (
        <p className="spaces-empty">{props.labels.noResults}</p>
      ) : props.spaces.length === 0 ? (
        <p className="spaces-empty">{props.labels.noConfirmed}</p>
      ) : (
        <div
          className="spaces-rows"
          onPointerCancel={(event) => cancelPointerMove(event.pointerId)}
          onLostPointerCapture={(event) => cancelPointerMove(event.pointerId)}
        >
          <div className="spaces-table-head" role="row">
            <span data-space-column="drag" aria-hidden="true" />
            <span data-space-column="name" role="columnheader">{props.labels.title}</span>
            <span data-space-column="pages" role="columnheader">{props.labels.pages}</span>
            <span data-space-column="memories" role="columnheader">{props.labels.memories}</span>
            <span data-space-column="updated" role="columnheader">{props.labels.updated}</span>
            <span data-space-column="menu" aria-hidden="true" />
          </div>
          {props.spaces.map((space) => {
            const canMoveUp = findReorderTarget(props.spaces, space.id, "up") !== null;
            const canMoveDown = findReorderTarget(props.spaces, space.id, "down") !== null;
            return (
              <div key={space.id} onPointerEnter={(event) => {
                if (activeDrag.current !== null) event.preventDefault();
              }} onPointerUp={(event) => finishPointerMove(event, space)}>
                <SpaceRow
                  space={space}
                  spaces={props.allSpaces}
                  labels={props.labels}
                  pageCount={props.pageCounts.get(space.name.toLocaleLowerCase()) ?? 0}
                  pending={props.pendingIds.includes(space.id)}
                  canMoveUp={canMoveUp}
                  canMoveDown={canMoveDown}
                  onSelect={props.onSelect}
                  onStar={props.onStar}
                  onRename={props.onRename}
                  onMoveUp={(source) => requestMove(source, "up")}
                  onMoveDown={(source) => requestMove(source, "down")}
                  onDelete={props.onDelete}
                  onDragStart={(source, pointerId) => {
                    if (activeDrag.current === null) activeDrag.current = { source, pointerId };
                  }}
                />
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
