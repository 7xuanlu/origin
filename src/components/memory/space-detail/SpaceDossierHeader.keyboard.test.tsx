// SPDX-License-Identifier: AGPL-3.0-only
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { i18n } from "../../../i18n";
import type { Space } from "../../../lib/tauri";
import { SpaceDossierHeader } from "./SpaceDossierHeader";
import { SPACE_DETAIL_TEST_COPY } from "./testTranslation";

const space: Space = {
  id: "space-1",
  name: "Wenlan",
  description: "Editorial memory",
  suggested: false,
  starred: false,
  sort_order: 0,
  memory_count: 12,
  entity_count: 3,
  created_at: 1_000,
  updated_at: 2_000,
};

function renderHeader(mutationError = false) {
  const actions = {
    onBack: vi.fn(),
    onCreatePage: vi.fn(),
    onDelete: vi.fn(),
    onKeep: vi.fn(),
    onSaveIdentity: vi.fn(),
  };
  const result = render(
    <SpaceDossierHeader
      actions={actions}
      copy={SPACE_DETAIL_TEST_COPY}
      mutationError={mutationError}
      pageCount="4"
      space={space}
      updatedLabel="Jul 10, 2026"
    />,
  );
  return {
    ...result,
    actions,
    rerenderHeader: (nextMutationError: boolean) => result.rerender(
      <SpaceDossierHeader
        actions={actions}
        copy={SPACE_DETAIL_TEST_COPY}
        mutationError={nextMutationError}
        pageCount="4"
        space={space}
        updatedLabel="Jul 10, 2026"
      />,
    ),
  };
}

describe("SpaceDossierHeader identity editing", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it.each(["Enter", " "])("opens one name-and-description editor with %s", async (key) => {
    // Given the single semantic identity edit control beside the H1
    const user = userEvent.setup();
    renderHeader();
    const editTitle = screen.getByRole("button", { name: "Edit space" });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);

    // When the focused control is activated from the keyboard
    editTitle.focus();
    await user.keyboard(key === " " ? "[Space]" : `{${key}}`);

    // Then the one edit mode owns focus and contains both identity fields
    expect(screen.getByRole("textbox", { name: "Wenlan" })).toHaveFocus();
    expect(screen.getByRole("textbox", { name: "Edit description" })).toHaveValue("Editorial memory");
    expect(screen.queryByRole("button", { name: "Edit description" })).not.toBeInTheDocument();
  });

  it("contains identity-editor Escape before Main history navigation", async () => {
    // Given the combined editor and Main's window-level Escape navigation
    const user = userEvent.setup();
    const { actions } = renderHeader();
    const navigateBack = () => actions.onBack();
    window.addEventListener("keydown", navigateBack);
    await user.click(screen.getByRole("button", { name: "Edit space" }));

    // When Escape is dispatched from the title input
    const accepted = fireEvent.keyDown(screen.getByRole("textbox", { name: "Wenlan" }), { key: "Escape" });
    window.removeEventListener("keydown", navigateBack);

    // Then the whole editor cancels and the Space page remains
    expect(accepted).toBe(false);
    expect(actions.onBack).not.toHaveBeenCalled();
    expect(screen.getByRole("heading", { level: 1, name: "Wenlan" })).toBeInTheDocument();
  });

  it("saves name and description together through one identity action", async () => {
    // Given the combined identity editor
    const user = userEvent.setup();
    const { actions } = renderHeader();
    await user.click(screen.getByRole("button", { name: "Edit space" }));
    fireEvent.change(screen.getByRole("textbox", { name: "Wenlan" }), {
      target: { value: " Wenlan research " },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Edit description" }), {
      target: { value: " A retained description " },
    });

    // When the combined editor is saved
    await user.click(screen.getByRole("button", { name: "Save" }));

    // Then one mutation contract receives the normalized pair
    expect(actions.onSaveIdentity).toHaveBeenCalledWith({
      name: "Wenlan research",
      description: "A retained description",
    });
    expect(actions.onSaveIdentity).toHaveBeenCalledTimes(1);
  });

  it("retains a failed identity update in the combined editor", async () => {
    // Given both fields submitted from the combined editor
    const user = userEvent.setup();
    const { actions, rerenderHeader } = renderHeader();
    await user.click(screen.getByRole("button", { name: "Edit space" }));
    const input = screen.getByRole("textbox", { name: "Wenlan" });
    fireEvent.change(input, { target: { value: " Wenlan research " } });
    fireEvent.change(screen.getByRole("textbox", { name: "Edit description" }), {
      target: { value: " A retained description " },
    });
    await user.click(screen.getByRole("button", { name: "Save" }));

    // When the parent reports the mutation failure
    rerenderHeader(true);

    // Then both attempted values stay editable beside the existing alert
    expect(actions.onSaveIdentity).toHaveBeenCalledWith({
      name: "Wenlan research",
      description: "A retained description",
    });
    expect(screen.getByRole("textbox", { name: "Wenlan" })).toHaveValue(" Wenlan research ");
    expect(screen.getByRole("textbox", { name: "Edit description" })).toHaveValue(" A retained description ");
    expect(screen.getByRole("alert")).toHaveTextContent("Could not save this change.");
  });

  it("keeps Delete inside the overflow menu", async () => {
    // Given a confirmed Space header
    const user = userEvent.setup();
    renderHeader();

    // Then Delete is unavailable until overflow is opened
    expect(screen.queryByRole("menuitem", { name: "Delete space" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Actions for Wenlan" }));
    expect(screen.getByRole("menuitem", { name: "Delete space" })).toBeInTheDocument();

    // When Delete is chosen, the existing explicit confirmation still gates it
    await user.click(screen.getByRole("menuitem", { name: "Delete space" }));
    expect(screen.getByRole("button", { name: "Confirm delete space" })).toBeInTheDocument();
  });

  it("moves focus into the overflow menu and contains its navigation keys", async () => {
    // Given the overflow trigger
    const user = userEvent.setup();
    renderHeader();
    await user.click(screen.getByRole("button", { name: "Actions for Wenlan" }));
    const menu = screen.getByRole("menu");
    const deleteItem = screen.getByRole("menuitem", { name: "Delete space" });

    // Then opening follows the menu focus contract
    expect(deleteItem).toHaveFocus();

    // And the menu owns the standard composite navigation keys
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
      expect(fireEvent.keyDown(menu, { key })).toBe(false);
      expect(deleteItem).toHaveFocus();
    }
  });

  it("opens the overflow menu from its trigger with ArrowDown or ArrowUp", async () => {
    const user = userEvent.setup();
    renderHeader();
    const trigger = screen.getByRole("button", { name: "Actions for Wenlan" });

    trigger.focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("menuitem", { name: "Delete space" })).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(trigger).toHaveFocus();
    await user.keyboard("{ArrowUp}");
    expect(screen.getByRole("menuitem", { name: "Delete space" })).toHaveFocus();
  });

  it("contains menu Escape before Main history navigation and restores trigger focus", async () => {
    // Given the menu is open above Main's window-level Escape navigation
    const user = userEvent.setup();
    const { actions } = renderHeader();
    const navigateBack = () => actions.onBack();
    window.addEventListener("keydown", navigateBack);
    const trigger = screen.getByRole("button", { name: "Actions for Wenlan" });
    await user.click(trigger);

    // When Escape is dispatched from the focused menu item
    const accepted = fireEvent.keyDown(screen.getByRole("menuitem", { name: "Delete space" }), {
      key: "Escape",
    });
    window.removeEventListener("keydown", navigateBack);

    // Then only the menu closes, without changing Space history
    expect(accepted).toBe(false);
    expect(actions.onBack).not.toHaveBeenCalled();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
});
