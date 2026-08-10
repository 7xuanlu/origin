// SPDX-License-Identifier: AGPL-3.0-only
import { StrictMode } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MarkdownEditor } from "./MarkdownEditor";
import { loadCodeMirrorEditor } from "./loadCodeMirrorEditor";
import type { MarkdownEditorProps } from "./MarkdownEditor";

vi.mock("./loadCodeMirrorEditor", () => ({ loadCodeMirrorEditor: vi.fn() }));

beforeEach(() => {
  vi.mocked(loadCodeMirrorEditor).mockRejectedValue(
    new Error("secret draft should never escape"),
  );
});

afterEach(() => vi.clearAllMocks());

const baseProps = (overrides: Partial<MarkdownEditorProps> = {}): MarkdownEditorProps => ({
  initialDocument: "original secret",
  sessionId: "page:1",
  disabled: false,
  ariaLabel: "Fallback source",
  onDocumentChange: vi.fn(),
  onSave: vi.fn(),
  onCancel: vi.fn(),
  onFallback: vi.fn(),
  ...overrides,
});

describe("MarkdownEditor load fallback", () => {
  it("uses the original session document and emits one content-free event under StrictMode", async () => {
    const onFallback = vi.fn();
    const props = baseProps({ onFallback });
    const { rerender } = render(
      <StrictMode>
        <MarkdownEditor {...props} />
      </StrictMode>,
    );

    const textbox = await screen.findByRole("textbox", { name: "Fallback source" });
    expect(textbox).toHaveValue("original secret");
    expect(screen.getByTestId("markdown-editor-fallback")).toBeEmptyDOMElement();
    expect(onFallback).toHaveBeenCalledTimes(1);
    expect(onFallback).toHaveBeenCalledWith("load");

    fireEvent.change(textbox, { target: { value: "local draft" } });
    rerender(
      <StrictMode>
        <MarkdownEditor
          {...props}
          initialDocument="ignored same-session refetch"
          onFallback={onFallback}
        />
      </StrictMode>,
    );
    expect(textbox).toHaveValue("local draft");
    expect(onFallback).toHaveBeenCalledTimes(1);
  });

  it("obeys current-document save, cancel, disabled, and accessibility contracts", async () => {
    const onSave = vi.fn();
    const onCancel = vi.fn();
    const { rerender } = render(
      <MarkdownEditor
        {...baseProps({
          describedBy: "fallback-help",
          onSave,
          onCancel,
        })}
      />,
    );
    const textbox = await screen.findByRole("textbox", { name: "Fallback source" });
    expect(textbox).toHaveAttribute("aria-describedby", "fallback-help");
    fireEvent.change(textbox, { target: { value: "current" } });
    fireEvent.keyDown(textbox, { key: "s", ctrlKey: true });
    fireEvent.keyDown(textbox, { key: "Escape" });
    expect(onSave).toHaveBeenCalledWith("current");
    expect(onCancel).toHaveBeenCalledOnce();

    rerender(
      <MarkdownEditor
        {...baseProps({ disabled: true, ariaLabel: "Disabled fallback", onSave, onCancel })}
      />,
    );
    expect(screen.getByRole("textbox", { name: "Disabled fallback" })).toBeDisabled();
  });
});
