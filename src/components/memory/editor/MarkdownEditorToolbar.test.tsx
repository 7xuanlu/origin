// SPDX-License-Identifier: AGPL-3.0-only
import { createEvent, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { MarkdownEditorCommand, MarkdownEditorStatus } from "./MarkdownEditor";
import {
  MarkdownEditorToolbar,
  type MarkdownEditorToolbarLabels,
  type MarkdownEditorToolbarProps,
} from "./MarkdownEditorToolbar";

const labels: MarkdownEditorToolbarLabels = {
  undo: "Undo",
  redo: "Redo",
  blockStyle: "Block style",
  paragraph: "Paragraph",
  heading1: "Heading 1",
  heading2: "Heading 2",
  heading3: "Heading 3",
  bold: "Bold",
  italic: "Italic",
  inlineCode: "Inline code",
  link: "Link",
  blockquote: "Quote",
  bulletList: "Bulleted list",
  numberedList: "Numbered list",
  save: "Save",
  cancel: "Cancel",
};

function status(overrides: Partial<MarkdownEditorStatus> = {}): MarkdownEditorStatus {
  return {
    sessionId: "page-1:1",
    engine: "codemirror",
    ready: true,
    compositionActive: false,
    canUndo: false,
    canRedo: false,
    ...overrides,
  };
}

function props(overrides: Partial<MarkdownEditorToolbarProps> = {}): MarkdownEditorToolbarProps {
  return {
    status: status(),
    labels,
    saveDisabled: false,
    cancelDisabled: false,
    onCommand: vi.fn(),
    onSave: vi.fn(),
    onCancel: vi.fn(),
    ...overrides,
  };
}

describe("MarkdownEditorToolbar", () => {
  it("does not expose Markdown presentation as a peer editing mode", () => {
    render(<MarkdownEditorToolbar {...props()} />);

    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(
      screen.getByRole("toolbar", { name: labels.blockStyle }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: labels.save })).toBeVisible();
    expect(
      screen.getByRole("group", { name: labels.blockStyle }),
    ).toBeVisible();
  });

  it("maps every format control to its exact command and follows undo/redo status", async () => {
    const user = userEvent.setup();
    const onCommand = vi.fn();
    render(
      <MarkdownEditorToolbar
        {...props({
          onCommand,
          status: status({ canUndo: true, canRedo: false }),
        })}
      />,
    );

    expect(screen.getByRole("button", { name: labels.undo })).toBeEnabled();
    expect(screen.getByRole("button", { name: labels.redo })).toBeDisabled();

    const buttonCommands: ReadonlyArray<readonly [string, MarkdownEditorCommand]> = [
      [labels.undo, "undo"],
      [labels.bold, "bold"],
      [labels.italic, "italic"],
      [labels.inlineCode, "inlineCode"],
      [labels.link, "link"],
      [labels.blockquote, "blockquote"],
      [labels.bulletList, "bulletList"],
      [labels.numberedList, "numberedList"],
    ];
    for (const [name] of buttonCommands) {
      await user.click(screen.getByRole("button", { name }));
    }

    const blockStyle = screen.getByRole("combobox", { name: labels.blockStyle });
    for (const command of ["paragraph", "heading1", "heading2", "heading3"] as const) {
      await user.selectOptions(blockStyle, command);
    }

    expect(onCommand.mock.calls.map(([command]) => command)).toEqual([
      ...buttonCommands.map(([, command]) => command),
      "paragraph",
      "heading1",
      "heading2",
      "heading3",
    ]);
  });

  it("renders familiar format actions as decorative 16px icons", () => {
    render(
      <MarkdownEditorToolbar
        {...props({ status: status({ canUndo: true, canRedo: true }) })}
      />,
    );

    const iconOnlyLabels = [
      labels.bold,
      labels.italic,
      labels.inlineCode,
      labels.link,
      labels.blockquote,
      labels.bulletList,
      labels.numberedList,
      labels.undo,
      labels.redo,
    ];

    for (const label of iconOnlyLabels) {
      const button = screen.getByRole("button", { name: label });
      expect(button.textContent).toBe("");
      expect(button).not.toHaveTextContent(label);
      expect(button).toHaveAttribute("title", label);
      const icons = button.querySelectorAll("svg");
      expect(icons).toHaveLength(1);
      expect(icons[0]).toHaveAttribute("aria-hidden", "true");
      expect(icons[0]).toHaveAttribute("width", "16");
      expect(icons[0]).toHaveAttribute("height", "16");
      expect(icons[0]).not.toHaveAttribute("tabindex");
    }
  });

  it("keeps block-style and persistence controls as visible text", () => {
    render(<MarkdownEditorToolbar {...props()} />);

    expect(
      screen.getByRole("combobox", { name: labels.blockStyle }),
    ).toHaveTextContent(
      `${labels.blockStyle}${labels.paragraph}${labels.heading1}${labels.heading2}${labels.heading3}`,
    );
    expect(screen.getByRole("button", { name: labels.save })).toHaveTextContent(
      labels.save,
    );
    expect(screen.getByRole("button", { name: labels.cancel })).toHaveTextContent(
      labels.cancel,
    );
  });

  it("keeps formatting and persistence in one compact toolbar row", () => {
    render(
      <MarkdownEditorToolbar
        {...props({ status: status({ canUndo: true, canRedo: true }) })}
      />,
    );

    const toolbar = screen.getByRole("toolbar", {
      name: labels.blockStyle,
    });
    const formatRow = screen.getByRole("group", {
      name: labels.blockStyle,
    });
    const persistenceActions = screen
      .getByRole("button", { name: labels.save })
      .closest(".page-editor-persistence-actions");
    expect(formatRow.parentElement).toBe(toolbar);
    expect(persistenceActions?.parentElement).toBe(toolbar);
    expect(toolbar.querySelector(".page-editor-toolbar-primary")).toBeNull();
    const groups = Array.from(
      formatRow.querySelectorAll<HTMLElement>("[data-toolbar-group]"),
    );
    expect(groups.map((group) => group.dataset.toolbarGroup)).toEqual([
      "inline",
      "block-style",
      "block",
      "history",
    ]);
    expect(
      groups.map((group) =>
        Array.from(group.querySelectorAll<HTMLElement>("button, select")).map(
          (control) => control.getAttribute("aria-label"),
        ),
      ),
    ).toEqual([
      [labels.bold, labels.italic, labels.inlineCode, labels.link],
      [labels.blockStyle],
      [labels.blockquote, labels.bulletList, labels.numberedList],
      [labels.undo, labels.redo],
    ]);
  });

  it("exposes which horizontal edge contains more formatting controls", () => {
    render(<MarkdownEditorToolbar {...props()} />);

    const formatGroup = screen.getByRole("group", { name: labels.blockStyle });
    Object.defineProperties(formatGroup, {
      clientWidth: { configurable: true, value: 300 },
      scrollWidth: { configurable: true, value: 700 },
      scrollLeft: { configurable: true, value: 0, writable: true },
    });

    fireEvent.scroll(formatGroup);
    expect(formatGroup).toHaveAttribute("data-overflow-cue", "end");

    formatGroup.scrollLeft = 400;
    fireEvent.scroll(formatGroup);
    expect(formatGroup).toHaveAttribute("data-overflow-cue", "start");
  });

  it("keeps the native recovery editor free of presentation controls and formatting actions", () => {
    render(
      <MarkdownEditorToolbar
        {...props({
          status: status({ engine: "native" }),
        })}
      />,
    );

    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("group", { name: labels.blockStyle }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: labels.save })).toBeEnabled();
    expect(screen.getByRole("button", { name: labels.cancel })).toBeEnabled();
  });

  it("keeps native disabled semantics while exposing a stable composition styling hook", () => {
    const composingProps = props({
      status: status({
        compositionActive: true,
        canUndo: false,
        canRedo: true,
      }),
    });
    const { rerender } = render(
      <MarkdownEditorToolbar
        {...composingProps}
      />,
    );

    const toolbar = screen
      .getByRole("button", { name: labels.save })
      .closest(".page-editor-toolbar");
    expect(toolbar).toHaveAttribute("data-composition-active", "true");
    expect(screen.getByRole("combobox", { name: labels.blockStyle })).toBeDisabled();
    for (const button of screen.getAllByRole("button")) expect(button).toBeDisabled();
    expect(screen.getByRole("button", { name: labels.bold })).toHaveAttribute(
      "data-composition-blocked",
      "true",
    );
    expect(screen.getByRole("button", { name: labels.redo })).toHaveAttribute(
      "data-composition-blocked",
      "true",
    );
    expect(screen.getByRole("button", { name: labels.undo })).not.toHaveAttribute(
      "data-composition-blocked",
    );

    rerender(
      <MarkdownEditorToolbar
        {...composingProps}
        status={status({ canUndo: false, canRedo: true })}
      />,
    );

    expect(toolbar).not.toHaveAttribute("data-composition-active");
    expect(screen.getByRole("button", { name: labels.bold })).toBeEnabled();
    expect(screen.getByRole("button", { name: labels.undo })).toBeDisabled();
    expect(screen.getByRole("button", { name: labels.redo })).toBeEnabled();
  });

  it("keeps editor focus on pointer-down and emits one command from keyboard activation", async () => {
    const user = userEvent.setup();
    const onCommand = vi.fn();
    render(<MarkdownEditorToolbar {...props({ onCommand })} />);
    const editor = document.createElement("textarea");
    document.body.prepend(editor);
    editor.focus();
    const bold = screen.getByRole("button", { name: labels.bold });
    const pointerDown = createEvent.pointerDown(bold, { bubbles: true, cancelable: true });

    fireEvent(bold, pointerDown);

    expect(pointerDown.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(editor);
    expect(onCommand).not.toHaveBeenCalled();

    bold.focus();
    await user.keyboard("[Enter]");
    expect(onCommand).toHaveBeenCalledOnce();
    expect(onCommand).toHaveBeenCalledWith("bold");
  });

  it("provides an accessible name and tooltip for every action button", () => {
    render(
      <MarkdownEditorToolbar
        {...props({ status: status({ canUndo: true, canRedo: true }) })}
      />,
    );

    for (const button of screen.getAllByRole("button")) {
      expect(button).toHaveAccessibleName();
      expect(button).toHaveAttribute("title", button.getAttribute("aria-label"));
    }
  });
});
