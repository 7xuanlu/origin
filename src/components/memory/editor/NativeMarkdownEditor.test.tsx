// SPDX-License-Identifier: AGPL-3.0-only
import { createRef } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { NativeMarkdownEditor } from "./NativeMarkdownEditor";
import type {
  MarkdownEditorHandle,
  MarkdownEditorProps,
  MarkdownEditorStatus,
} from "./MarkdownEditor";

const baseProps = (overrides: Partial<MarkdownEditorProps> = {}): MarkdownEditorProps => ({
  initialDocument: "original\n中文 😀 e\u0301",
  sessionId: "page-1:1",
  disabled: false,
  ariaLabel: "Native source",
  onDocumentChange: vi.fn(),
  onSave: vi.fn(),
  onCancel: vi.fn(),
  onFallback: vi.fn(),
  ...overrides,
});

describe("NativeMarkdownEditor", () => {
  it("reports Markdown-only status and saves the live document through its handle", () => {
    const ref = createRef<MarkdownEditorHandle>();
    const onSave = vi.fn();
    const onCancel = vi.fn();
    const onStatusChange = vi.fn<(status: MarkdownEditorStatus) => void>();
    render(
      <NativeMarkdownEditor
        ref={ref}
        {...baseProps({ onSave, onCancel })}
        onStatusChange={onStatusChange}
      />,
    );

    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "live\nsource\n" },
    });

    expect(ref.current?.requestSave).toBeTypeOf("function");
    expect(ref.current?.requestSave()).toBe(true);
    expect(onSave).toHaveBeenLastCalledWith("live\nsource\n");
    expect(ref.current?.runCommand("bold")).toBe(false);
    expect(ref.current?.requestCancel()).toBe(true);
    expect(onCancel).toHaveBeenCalledOnce();
    expect(onStatusChange).toHaveBeenLastCalledWith({
      sessionId: "page-1:1",
      engine: "native",
      ready: true,
      compositionActive: false,
      canUndo: false,
      canRedo: false,
    });
  });

  it("is an accessible auto-growing multiline textbox and emits exact logical-LF text", () => {
    const onDocumentChange = vi.fn();
    const props = baseProps({ onDocumentChange, describedBy: "help" });
    const { rerender } = render(<NativeMarkdownEditor {...props} />);
    const textbox = screen.getByRole("textbox", { name: "Native source" });

    expect(textbox).toHaveAttribute("aria-describedby", "help");
    expect(textbox).toHaveAttribute("aria-multiline", "true");
    expect(textbox).toHaveStyle({
      width: "100%",
      minHeight: "300px",
      resize: "none",
      overflow: "hidden",
    });
    expect(textbox).not.toHaveStyle({ maxHeight: "min(60vh, 36rem)" });
    fireEvent.change(textbox, { target: { value: "  exact\n中文 😀 e\u0301\n" } });
    expect(onDocumentChange).toHaveBeenCalledWith("  exact\n中文 😀 e\u0301\n");

    rerender(
      <NativeMarkdownEditor {...baseProps({ initialDocument: "remote", sessionId: props.sessionId })} />,
    );
    expect(textbox).toHaveValue("  exact\n中文 😀 e\u0301\n");
  });

  it("resets to the new initial document only when sessionId changes", () => {
    const { rerender } = render(<NativeMarkdownEditor {...baseProps()} />);
    const textbox = screen.getByRole("textbox");
    fireEvent.change(textbox, { target: { value: "draft" } });

    rerender(
      <NativeMarkdownEditor {...baseProps({ initialDocument: "next", sessionId: "page-1:2" })} />,
    );

    expect(screen.getByRole("textbox")).toHaveValue("next");
  });

  it("uses fresh callbacks, saves the synchronous document, and cancels", () => {
    const firstSave = vi.fn();
    const secondSave = vi.fn();
    const firstChange = vi.fn();
    const secondChange = vi.fn();
    const onCancel = vi.fn();
    const { rerender } = render(
      <NativeMarkdownEditor
        {...baseProps({ onSave: firstSave, onDocumentChange: firstChange, onCancel })}
      />,
    );
    const textbox = screen.getByRole("textbox");
    rerender(
      <NativeMarkdownEditor
        {...baseProps({ onSave: secondSave, onDocumentChange: secondChange, onCancel })}
      />,
    );

    fireEvent.change(textbox, { target: { value: "current" } });
    fireEvent.keyDown(textbox, { key: "s", ctrlKey: true });
    fireEvent.keyDown(textbox, { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(textbox, { key: "Escape" });

    expect(firstChange).not.toHaveBeenCalled();
    expect(secondChange).toHaveBeenCalledWith("current");
    expect(firstSave).not.toHaveBeenCalled();
    expect(secondSave).toHaveBeenNthCalledWith(1, "current");
    expect(secondSave).toHaveBeenNthCalledWith(2, "current");
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("consumes Mod-s but suppresses save and cancel during composition", () => {
    const ref = createRef<MarkdownEditorHandle>();
    const onSave = vi.fn();
    const onCancel = vi.fn();
    const onStatusChange = vi.fn<(status: MarkdownEditorStatus) => void>();
    render(
      <NativeMarkdownEditor
        ref={ref}
        {...baseProps({ onSave, onCancel, onStatusChange })}
      />,
    );
    const textbox = screen.getByRole("textbox");

    fireEvent.compositionStart(textbox);
    expect(onStatusChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ compositionActive: true }),
    );
    expect(ref.current?.requestSave()).toBe(false);
    expect(ref.current?.requestCancel()).toBe(false);
    expect(ref.current?.runCommand("bold")).toBe(false);
    const save = new KeyboardEvent("keydown", {
      bubbles: true,
      cancelable: true,
      key: "s",
      ctrlKey: true,
    });
    textbox.dispatchEvent(save);
    fireEvent.keyDown(textbox, { key: "Enter", ctrlKey: true });
    fireEvent.keyDown(textbox, { key: "Escape" });

    expect(save.defaultPrevented).toBe(true);
    expect(onSave).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();

    fireEvent.compositionEnd(textbox);
    expect(onStatusChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ compositionActive: false }),
    );
  });

  it("updates disabled and accessibility attributes and focuses through its handle", () => {
    const ref = createRef<MarkdownEditorHandle>();
    const { rerender } = render(<NativeMarkdownEditor ref={ref} {...baseProps()} />);
    const textbox = screen.getByRole("textbox");
    const focus = vi.spyOn(textbox, "focus");

    rerender(
      <NativeMarkdownEditor
        ref={ref}
        {...baseProps({ disabled: true, ariaLabel: "Disabled native", describedBy: "why" })}
      />,
    );
    act(() => ref.current?.focus());

    expect(screen.getByRole("textbox", { name: "Disabled native" })).toBeDisabled();
    expect(textbox).toHaveAttribute("aria-disabled", "true");
    expect(textbox).toHaveAttribute("aria-describedby", "why");
    expect(focus).toHaveBeenCalledOnce();
  });
});
