import type { Space } from "../../../lib/tauri";
import type { SpacesOverviewLabels } from "./spacesTypes";

type SuggestedSpacesProps = {
  readonly spaces: readonly Space[];
  readonly labels: SpacesOverviewLabels;
  readonly pendingIds: readonly string[];
  readonly onSelect: (spaceName: string) => void;
  readonly onKeep: (space: Space) => void;
  readonly onDiscard: (space: Space) => void;
};

export function SuggestedSpaces(props: SuggestedSpacesProps) {
  if (props.spaces.length === 0) return null;

  return (
    <section className="spaces-section" aria-labelledby="suggested-spaces-heading">
      <h2 id="suggested-spaces-heading">
        {props.labels.suggestedHeading} ({props.spaces.length})
      </h2>
      <div className="spaces-rows">
        {props.spaces.map((space) => {
          const pending = props.pendingIds.includes(space.id);
          return (
            <div
              className="spaces-row spaces-row-suggested"
              data-testid={`space-row-${space.id}`}
              aria-busy={pending}
              key={space.id}
            >
              <button className="spaces-row-main" aria-label={space.name} onClick={() => props.onSelect(space.name)}>
                <span className="spaces-row-name">{space.name}</span>
                {space.description === null ? null : <span className="spaces-row-description">{space.description}</span>}
              </button>
              <div className="spaces-row-decisions">
                <button
                  type="button"
                  disabled={pending}
                  className="spaces-suggestion-action spaces-suggestion-keep"
                  onClick={() => props.onKeep(space)}
                >
                  {props.labels.keep}
                </button>
                <button
                  type="button"
                  disabled={pending}
                  className="spaces-suggestion-action spaces-suggestion-discard"
                  onClick={() => props.onDiscard(space)}
                >
                  {props.labels.discard}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
