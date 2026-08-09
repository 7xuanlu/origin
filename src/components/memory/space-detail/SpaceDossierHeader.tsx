// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import type { Space } from "../../../lib/tauri";
import type { SpaceDetailCopy } from "./copy";

type SpaceIdentityActions = {
  readonly onBack: () => void;
  readonly onCreatePage: () => void;
  readonly onDelete: () => void;
  readonly onKeep: () => void;
  readonly onSaveIdentity: (value: SpaceIdentityValue) => void;
};

export type SpaceIdentityValue = {
  readonly description: string;
  readonly name: string;
};

type SpaceDossierHeaderProps = {
  readonly actions: SpaceIdentityActions;
  readonly copy: SpaceDetailCopy;
  readonly mutationError: boolean;
  readonly pageCount: string;
  readonly space: Space;
  readonly updatedLabel: string;
};

type SubmittedIdentity = {
  readonly inputDescription: string;
  readonly inputName: string;
  readonly submittedDescription: string;
  readonly submittedName: string;
};

export function SpaceDossierHeader({
  actions,
  copy,
  mutationError,
  pageCount,
  space,
  updatedLabel,
}: SpaceDossierHeaderProps) {
  const { t } = useTranslation();
  const [editingIdentity, setEditingIdentity] = useState(false);
  const [nameValue, setNameValue] = useState(space.name);
  const [descriptionValue, setDescriptionValue] = useState(space.description ?? "");
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuTriggerRef = useRef<HTMLButtonElement>(null);
  const menuInitialFocusRef = useRef<"first" | "last">("first");
  const submittedIdentity = useRef<SubmittedIdentity | null>(null);

  useEffect(() => {
    const submitted = submittedIdentity.current;
    if (submitted === null) return;
    const updateLanded = space.name === submitted.submittedName
      && (space.description ?? "") === submitted.submittedDescription;
    if (updateLanded) {
      submittedIdentity.current = null;
      return;
    }
    if (!mutationError) return;
    setNameValue(submitted.inputName);
    setDescriptionValue(submitted.inputDescription);
    setEditingIdentity(true);
    submittedIdentity.current = null;
  }, [mutationError, space.description, space.name]);

  useEffect(() => {
    if (!menuOpen) return;
    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')
        ?? [],
    );
    items[menuInitialFocusRef.current === "first" ? 0 : items.length - 1]?.focus();
  }, [menuOpen]);

  const beginIdentityEdit = () => {
    submittedIdentity.current = null;
    setNameValue(space.name);
    setDescriptionValue(space.description ?? "");
    setMenuOpen(false);
    setEditingIdentity(true);
  };

  const cancelIdentityEdit = () => {
    submittedIdentity.current = null;
    setNameValue(space.name);
    setDescriptionValue(space.description ?? "");
    setEditingIdentity(false);
  };

  const saveIdentity = () => {
    const name = nameValue.trim();
    const description = descriptionValue.trim();
    if (name.length === 0) return;
    if (name === space.name && description === (space.description ?? "")) {
      setEditingIdentity(false);
      return;
    }
    submittedIdentity.current = {
      inputDescription: descriptionValue,
      inputName: nameValue,
      submittedDescription: description,
      submittedName: name,
    };
    actions.onSaveIdentity({ description, name });
    setEditingIdentity(false);
  };

  const handleEditorKeyDown = (event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      cancelIdentityEdit();
      return;
    }
    if (event.key === "Enter" && (event.currentTarget instanceof HTMLInputElement || event.metaKey)) {
      event.preventDefault();
      saveIdentity();
    }
  };

  const handleMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setMenuOpen(false);
      menuTriggerRef.current?.focus();
      return;
    }

    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)'),
    );
    if (items.length === 0) return;
    const currentIndex = items.indexOf(document.activeElement as HTMLButtonElement);
    if (event.key === "Home") {
      items[0].focus();
    } else if (event.key === "End") {
      items[items.length - 1].focus();
    } else if (event.key === "ArrowDown") {
      items[(currentIndex + 1) % items.length].focus();
    } else {
      items[(currentIndex - 1 + items.length) % items.length].focus();
    }
  };

  const openMenu = (initialFocus: "first" | "last") => {
    menuInitialFocusRef.current = initialFocus;
    setMenuOpen(true);
  };

  const handleMenuTriggerKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    event.stopPropagation();
    openMenu(event.key === "ArrowDown" ? "first" : "last");
  };

  return (
    <>
      <button className="space-dossier-parent" onClick={actions.onBack} type="button">
        <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
        <span>{copy.backToSpaces}</span>
      </button>

      {space.suggested && (
        <div className="space-dossier-suggested">
          <span>{copy.suggestedLabel}</span>
          <div>
            <button className="space-dossier-suggestion-action space-dossier-suggestion-keep" onClick={actions.onKeep} type="button">{copy.keep}</button>
            <button className="space-dossier-suggestion-action space-dossier-suggestion-discard" onClick={actions.onDelete} type="button">{copy.discard}</button>
          </div>
        </div>
      )}

      <header className="space-dossier-header">
        <div className="space-dossier-title-row">
          {editingIdentity ? (
            <div className="space-dossier-identity-editor">
              <input
                aria-label={space.name}
                autoFocus
                className="space-dossier-title-input"
                onChange={(event) => setNameValue(event.target.value)}
                onKeyDown={handleEditorKeyDown}
                value={nameValue}
              />
              <textarea
                aria-label={copy.editDescription}
                className="space-dossier-description-editor"
                onChange={(event) => setDescriptionValue(event.target.value)}
                onKeyDown={handleEditorKeyDown}
                placeholder={copy.descriptionPlaceholder}
                value={descriptionValue}
              />
              <div className="space-dossier-identity-editor-actions">
                <button onClick={saveIdentity} type="button">{copy.save}</button>
                <button onClick={cancelIdentityEdit} type="button">{copy.cancel}</button>
                <span>{copy.saveShortcut}</span>
              </div>
            </div>
          ) : (
            <>
              <div className="space-dossier-title-block">
                <div className="space-dossier-title-heading">
                  <h1>{space.name}</h1>
                  <button
                    aria-label={t("spaces.overview.editSpace")}
                    className="mem-icon-action space-dossier-title-edit"
                    onClick={beginIdentityEdit}
                    title={t("spaces.overview.editSpace")}
                    type="button"
                  >
                    <svg aria-hidden="true" viewBox="0 0 24 24"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" /></svg>
                  </button>
                </div>
                {space.description && <p className="space-dossier-description">{space.description}</p>}
              </div>

              {!space.suggested && (
                <div className="space-dossier-actions">
                  <button
                    className="page-create-action space-dossier-new-page"
                    onClick={actions.onCreatePage}
                    type="button"
                  >
                    {t("pages.overview.newPage")}
                  </button>
                  <button
                    ref={menuTriggerRef}
                    aria-expanded={menuOpen}
                    aria-haspopup="menu"
                    aria-label={t("spaces.overview.actionsFor", { name: space.name })}
                    className="mem-icon-action space-dossier-overflow-trigger"
                    onClick={() => {
                      if (menuOpen) {
                        setMenuOpen(false);
                      } else {
                        openMenu("first");
                      }
                    }}
                    onKeyDown={handleMenuTriggerKeyDown}
                    type="button"
                  >
                    <svg aria-hidden="true" className="space-dossier-overflow-icon" viewBox="0 0 24 24"><circle cx="5" cy="12" r="1.4" /><circle cx="12" cy="12" r="1.4" /><circle cx="19" cy="12" r="1.4" /></svg>
                  </button>
                  {menuOpen && (
                    <div
                      ref={menuRef}
                      className="mem-popover-surface space-dossier-menu"
                      onKeyDown={handleMenuKeyDown}
                      role="menu"
                    >
                      <button
                        className="space-dossier-menu-danger"
                        onClick={() => {
                          setConfirmingDelete(true);
                          setMenuOpen(false);
                        }}
                        role="menuitem"
                        type="button"
                      >
                        {copy.deleteSpace}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {confirmingDelete && (
          <div className="space-dossier-delete-confirmation">
            <span>{copy.deleteWarning}</span>
            <button aria-label={copy.confirmDeleteSpace} onClick={actions.onDelete} type="button">{copy.deleteSpace}</button>
            <button onClick={() => setConfirmingDelete(false)} type="button">{copy.cancel}</button>
          </div>
        )}

        {mutationError && <p className="space-dossier-error" role="alert">{copy.mutationError}</p>}

        <dl className="space-dossier-metrics">
          <div><dt>{copy.metrics.pages}</dt><dd>{pageCount}</dd></div>
          <div><dt>{copy.metrics.memories}</dt><dd>{new Intl.NumberFormat().format(space.memory_count)}</dd></div>
          <div><dt>{copy.metrics.entities}</dt><dd>{new Intl.NumberFormat().format(space.entity_count)}</dd></div>
          <div><dt>{copy.metrics.updated}</dt><dd>{updatedLabel}</dd></div>
        </dl>
      </header>
    </>
  );
}
