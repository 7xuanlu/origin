import { useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import type { Space } from "../../../lib/tauri";
import { isDuplicateSpaceName } from "./spaceHelpers";
import type { SpaceEditorValue, SpacesOverviewLabels } from "./spacesTypes";

type SpaceEditorProps = {
  readonly labels: SpacesOverviewLabels;
  readonly spaces: readonly Space[];
  readonly initialName?: string;
  readonly initialDescription?: string;
  readonly showDescription: boolean;
  readonly submitLabel: string;
  readonly pending: boolean;
  readonly onSubmit: (value: SpaceEditorValue) => void;
  readonly onCancel: () => void;
};

export function SpaceEditor(props: SpaceEditorProps) {
  const [name, setName] = useState(props.initialName ?? "");
  const [description, setDescription] = useState(props.initialDescription ?? "");
  const [validation, setValidation] = useState<string | null>(null);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (props.pending) return;
    const trimmedName = name.trim();
    if (trimmedName.length === 0) {
      setValidation(props.labels.nameRequired);
      return;
    }
    if (isDuplicateSpaceName(props.spaces, trimmedName, props.initialName)) {
      setValidation(props.labels.duplicateName);
      return;
    }
    const trimmedDescription = description.trim();
    props.onSubmit({
      name: trimmedName,
      ...(trimmedDescription.length > 0 ? { description: trimmedDescription } : {}),
    });
  };

  const handleEditorKey = (event: KeyboardEvent<HTMLFormElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      props.onCancel();
      return;
    }
    if (event.key === "Enter" && event.target instanceof HTMLInputElement) {
      event.preventDefault();
      event.currentTarget.requestSubmit();
    }
  };

  return (
    <form className="spaces-editor" onSubmit={submit} onKeyDown={handleEditorKey}>
      <label className="spaces-field">
        <span>{props.labels.nameLabel}</span>
        <input
          autoFocus
          value={name}
          disabled={props.pending}
          onChange={(event) => {
            setName(event.currentTarget.value);
            setValidation(null);
          }}
        />
      </label>
      {props.showDescription ? (
        <label className="spaces-field">
          <span>{props.labels.descriptionLabel}</span>
          <textarea
            value={description}
            disabled={props.pending}
            placeholder={props.labels.descriptionPlaceholder}
            onChange={(event) => setDescription(event.currentTarget.value)}
          />
        </label>
      ) : null}
      {validation === null ? null : <p className="spaces-validation" role="alert">{validation}</p>}
      <div className="spaces-editor-actions">
        <button type="submit" className="spaces-primary-action" disabled={props.pending}>
          {props.submitLabel}
        </button>
        <button type="button" className="spaces-quiet-action" disabled={props.pending} onClick={props.onCancel}>
          {props.labels.cancel}
        </button>
      </div>
    </form>
  );
}
