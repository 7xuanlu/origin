import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createRef } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createPageDraft,
  getPage,
  listSpaces,
  publishPageDraft,
  updatePageDraft,
  type Page,
  type Space,
} from "../../../lib/tauri";
import {
  PageDraftEditor,
  type PageDraftEditorHandle,
} from "./PageDraftEditor";

vi.mock("../../../lib/tauri", () => ({
  createPageDraft: vi.fn(),
  discardPageDraft: vi.fn(),
  getPage: vi.fn(),
  listSpaces: vi.fn(),
  publishPageDraft: vi.fn(),
  updatePageDraft: vi.fn(),
}));

function page(overrides: Partial<Page> = {}): Page {
  return {
    id: "draft-1",
    title: "Draft title",
    summary: null,
    content: "Draft body",
    entity_id: null,
    domain: null,
    space: null,
    source_memory_ids: [],
    version: 3,
    status: "draft",
    creation_kind: "authored",
    review_status: "unconfirmed",
    created_at: "2026-07-16T00:00:00Z",
    last_compiled: "2026-07-16T00:00:00Z",
    last_modified: "2026-07-16T00:00:00Z",
    ...overrides,
  };
}

function space(name: string): Space {
  return {
    id: `space-${name}`,
    name,
    description: null,
    suggested: false,
    starred: false,
    sort_order: 0,
    memory_count: 0,
    entity_count: 0,
    created_at: 0,
    updated_at: 0,
  };
}

function renderEditor(props: Partial<React.ComponentProps<typeof PageDraftEditor>> = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onBack = vi.fn();
  const onPublished = vi.fn();
  const onOpenExisting = vi.fn();
  const result = render(
    <QueryClientProvider client={queryClient}>
      <PageDraftEditor
        onBack={onBack}
        onOpenExisting={onOpenExisting}
        onPublished={onPublished}
        space={null}
        {...props}
      />
    </QueryClientProvider>,
  );
  return { ...result, onBack, onOpenExisting, onPublished, queryClient };
}

describe("PageDraftEditor", () => {
  beforeEach(() => {
    vi.mocked(createPageDraft).mockReset();
    vi.mocked(getPage).mockReset();
    vi.mocked(listSpaces).mockReset().mockResolvedValue([space("Work")]);
    vi.mocked(publishPageDraft).mockReset();
    vi.mocked(updatePageDraft).mockReset();
  });

  it("autofocuses the title and starts Wiki drafts without a Space or Optional copy", async () => {
    renderEditor();

    expect(await screen.findByRole("textbox", { name: "Title" })).toHaveFocus();
    expect(screen.getByRole("combobox", { name: "Space" })).toHaveValue("");
    expect(screen.queryByText(/optional/i)).not.toBeInTheDocument();
  });

  it("reuses the existing Page detail object-title typography", () => {
    const css = readFileSync(
      resolve("src/components/memory/pages/pageDraftEditor.css"),
      "utf8",
    );
    const titleRule = css.match(/\.page-draft-title\s*\{(?<body>[^}]*)\}/)?.groups?.body;

    expect(titleRule).toContain("font-family: var(--mem-font-heading);");
    expect(titleRule).toContain("font-size: clamp(24px, 1.6vw + 14px, 30px);");
    expect(titleRule).toContain("font-weight: 500;");
    expect(titleRule).toContain("line-height: 1.22;");
    expect(titleRule).toContain("letter-spacing: -0.01em;");
  });

  it("preselects and can clear the current Space", async () => {
    renderEditor({ space: "Work" });
    const select = await screen.findByRole("combobox", { name: "Space" });

    expect(select).toHaveValue("Work");
    await userEvent.selectOptions(select, "");
    expect(select).toHaveValue("");
  });

  it("hydrates a resumed draft and synthesizes a stale Space option", async () => {
    vi.mocked(getPage).mockResolvedValue(page({ space: "Archived Space" }));
    renderEditor({ draftId: "draft-1" });

    expect(screen.getByText("Loading draft…")).toBeInTheDocument();
    expect(await screen.findByRole("textbox", { name: "Title" })).toHaveValue("Draft title");
    expect(screen.getByRole("textbox", { name: "Content" })).toHaveValue("Draft body");
    expect(screen.getByRole("combobox", { name: "Space" })).toHaveValue("Archived Space");
    expect(screen.getByRole("option", { name: "Archived Space" })).toBeInTheDocument();
  });

  it("renders explicit missing and load-error states", async () => {
    vi.mocked(getPage).mockResolvedValueOnce(null);
    const first = renderEditor({ draftId: "missing" });
    expect(await screen.findByText("This draft no longer exists.")).toBeInTheDocument();
    first.unmount();

    vi.mocked(getPage).mockRejectedValueOnce(new Error("offline"));
    renderEditor({ draftId: "broken" });
    expect(await screen.findByRole("alert")).toHaveTextContent("Draft couldn't be loaded.");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("does not hydrate an externally published Page as an editable draft", async () => {
    vi.mocked(getPage).mockResolvedValue(page({ status: "active" }));
    renderEditor({ draftId: "draft-1" });

    expect(await screen.findByText("This page is no longer a draft.")).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Title" })).not.toBeInTheDocument();
  });

  it("awaits the same explicit flush gate for Back, Escape, and the public handle", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    const ref = createRef<PageDraftEditorHandle>();
    const { onBack } = renderEditor({ ref });
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "A");

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(onBack).not.toHaveBeenCalled();
    await act(async () => {
      resolveCreate(page({ title: "A", content: "", version: 1 }));
    });
    await waitFor(() => expect(onBack).toHaveBeenCalledTimes(1));

    await act(async () => {
      await ref.current?.requestBack();
    });
    expect(onBack).toHaveBeenCalledTimes(2);

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(onBack).toHaveBeenCalledTimes(3));
  });

  it("coalesces concurrent Back, Escape, and public leave requests", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    const ref = createRef<PageDraftEditorHandle>();
    const { onBack } = renderEditor({ ref });
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "A");

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    fireEvent.keyDown(window, { key: "Escape" });
    const publicLeave = ref.current?.requestBack();
    expect(onBack).not.toHaveBeenCalled();

    await act(async () => {
      resolveCreate(page({ title: "A", content: "", version: 1 }));
      await publicLeave;
    });

    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("locks Publish while a parent leave autosave is in flight", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    const ref = createRef<PageDraftEditorHandle>();
    renderEditor({ ref });
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "A");
    await userEvent.type(screen.getByRole("textbox", { name: "Content" }), "B");

    let saving!: Promise<boolean>;
    act(() => {
      saving = ref.current!.flush();
    });
    expect(await screen.findByText("Saving…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Publish" })).toBeDisabled();

    await act(async () => {
      resolveCreate(page({ title: "A", content: "B", version: 1 }));
      await saving;
    });
  });

  it("keeps save failures above the writing surface", async () => {
    vi.mocked(createPageDraft).mockRejectedValue(new Error("offline"));
    renderEditor();
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "A");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));

    const alert = await screen.findByRole("alert");
    const content = screen.getByRole("textbox", { name: "Content" });
    expect(
      alert.compareDocumentPosition(content) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("publishes only meaningful drafts after flush and preserves the same id", async () => {
    vi.mocked(createPageDraft).mockResolvedValue(page({ version: 1 }));
    vi.mocked(publishPageDraft).mockResolvedValue(page({ status: "active", version: 2 }));
    const { onPublished } = renderEditor();
    const publish = await screen.findByRole("button", { name: "Publish" });
    expect(publish).toBeDisabled();

    await userEvent.type(screen.getByRole("textbox", { name: "Title" }), "Draft title");
    await userEvent.type(screen.getByRole("textbox", { name: "Content" }), "Draft body");
    expect(publish).toBeEnabled();
    await userEvent.click(publish);

    await waitFor(() => expect(publishPageDraft).toHaveBeenCalledWith({
      id: "draft-1",
      expectedVersion: 1,
    }));
    expect(onPublished).toHaveBeenCalledWith("draft-1");
  });

  it("reconciles a lost publish response when the draft is already active", async () => {
    vi.mocked(createPageDraft).mockImplementation(async (input) => page({
      id: input.clientDraftId,
      title: input.title,
      content: input.content,
      space: input.space,
      version: 1,
    }));
    vi.mocked(publishPageDraft).mockRejectedValue(new Error("response lost"));
    vi.mocked(getPage).mockImplementation(async (id) => page({
      id,
      status: "active",
      version: 2,
    }));
    const { onPublished } = renderEditor();
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "Draft title");
    await userEvent.type(screen.getByRole("textbox", { name: "Content" }), "Draft body");

    await userEvent.click(screen.getByRole("button", { name: "Publish" }));

    await waitFor(() => expect(getPage).toHaveBeenCalledWith(
      expect.stringMatching(/^page_[0-9a-f-]+$/),
    ));
    const getPageCalls = vi.mocked(getPage).mock.calls;
    const reconciledId = getPageCalls[getPageCalls.length - 1]?.[0];
    expect(onPublished).toHaveBeenCalledWith(reconciledId);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("opens a later active Page when it is newer than the attempted publish", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 5,
    });
    vi.mocked(getPage)
      .mockResolvedValueOnce(page())
      .mockResolvedValueOnce(page({ status: "active", version: 5 }));
    vi.mocked(publishPageDraft).mockRejectedValueOnce(conflict);
    const { onOpenExisting, onPublished, queryClient } = renderEditor({ draftId: "draft-1" });
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");

    await userEvent.click(await screen.findByRole("button", { name: "Publish" }));

    await waitFor(() => expect(getPage).toHaveBeenLastCalledWith("draft-1"));
    expect(onPublished).toHaveBeenCalledWith("draft-1");
    expect(onOpenExisting).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["pages", "active"] });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["pages", "draft"] });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps a structured publish conflict blocked when reconciliation fails", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 5,
    });
    vi.mocked(getPage)
      .mockResolvedValueOnce(page())
      .mockRejectedValueOnce(new Error("offline"));
    vi.mocked(publishPageDraft).mockRejectedValueOnce(conflict);
    renderEditor({ draftId: "draft-1" });

    const publish = await screen.findByRole("button", { name: "Publish" });
    await userEvent.click(publish);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This draft changed elsewhere.",
    );
    expect(screen.getByRole("button", { name: "Reload latest" })).toBeInTheDocument();
    expect(publish).toBeDisabled();
  });

  it("adopts a renamed Space returned by create replay without saving the obsolete name", async () => {
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(new Error("response lost"))
      .mockImplementationOnce(async (input) => page({
        id: input.clientDraftId,
        title: input.title,
        content: input.content,
        space: "Work renamed",
        version: 2,
      }));
    renderEditor({ space: "Work" });
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "Draft title");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    await screen.findByRole("alert");

    await userEvent.click(screen.getByRole("button", { name: "Retry save" }));

    await waitFor(() => expect(
      screen.getByRole("combobox", { name: "Space" }),
    ).toHaveValue("Work renamed"));
    expect(updatePageDraft).not.toHaveBeenCalled();
  });

  it("single-flights Publish from the first click and locks every editable field during flush", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    vi.mocked(publishPageDraft).mockResolvedValue(
      page({ status: "active", version: 2 }),
    );
    renderEditor();
    const title = await screen.findByRole("textbox", { name: "Title" });
    const content = screen.getByRole("textbox", { name: "Content" });
    const spaceSelect = screen.getByRole("combobox", { name: "Space" });
    await userEvent.type(title, "Draft title");
    await userEvent.type(content, "Draft body");
    const publish = screen.getByRole("button", { name: "Publish" });

    act(() => {
      publish.click();
      publish.click();
    });

    expect(title).toBeDisabled();
    expect(content).toBeDisabled();
    expect(spaceSelect).toBeDisabled();
    expect(screen.getByRole("button", { name: "Publishing…" })).toBeDisabled();
    await act(async () => {
      resolveCreate(page({ version: 1 }));
    });
    await waitFor(() => expect(publishPageDraft).toHaveBeenCalledTimes(1));
  });

  it("does not leave through Back, Escape, or the public handle while Publish is in flight", async () => {
    let resolvePublish!: (value: Page) => void;
    vi.mocked(getPage).mockResolvedValue(page());
    vi.mocked(publishPageDraft).mockReturnValue(new Promise((resolve) => {
      resolvePublish = resolve;
    }));
    const ref = createRef<PageDraftEditorHandle>();
    const { onBack, onPublished } = renderEditor({ draftId: "draft-1", ref });
    await screen.findByRole("textbox", { name: "Title" });

    fireEvent.click(screen.getByRole("button", { name: "Publish" }));
    const back = screen.getByRole("button", { name: "Back" });
    expect(back).toBeDisabled();
    fireEvent.click(back);
    fireEvent.keyDown(window, { key: "Escape" });
    const publicLeave = ref.current!.requestBack();

    expect(onBack).not.toHaveBeenCalled();
    await act(async () => {
      resolvePublish(page({ status: "active", version: 4 }));
      await publicLeave;
    });
    expect(onPublished).toHaveBeenCalledWith("draft-1");
    expect(onBack).not.toHaveBeenCalled();
  });

  it("makes a parent leave flush win over a Publish clicked in the same turn", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    vi.mocked(publishPageDraft).mockResolvedValue(
      page({ status: "active", version: 2 }),
    );
    const ref = createRef<PageDraftEditorHandle>();
    renderEditor({ ref });
    const title = await screen.findByRole("textbox", { name: "Title" });
    const content = screen.getByRole("textbox", { name: "Content" });
    await userEvent.type(title, "Draft title");
    await userEvent.type(content, "Draft body");
    const publish = screen.getByRole("button", { name: "Publish" });

    let leaving!: Promise<boolean>;
    act(() => {
      leaving = ref.current!.flush();
      publish.click();
    });

    expect(title).toBeDisabled();
    expect(content).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Space" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Publish" })).toBeDisabled();
    await act(async () => {
      resolveCreate(page({ version: 1 }));
      expect(await leaving).toBe(true);
    });
    expect(publishPageDraft).not.toHaveBeenCalled();
  });

  it("lets an Escape pre-leave handler close surrounding UI without flushing the draft", async () => {
    const onEscapeBeforeLeave = vi.fn(() => true);
    const { onBack } = renderEditor({ onEscapeBeforeLeave });
    await userEvent.type(await screen.findByRole("textbox", { name: "Title" }), "A");

    fireEvent.keyDown(window, { key: "Escape" });

    expect(onEscapeBeforeLeave).toHaveBeenCalledTimes(1);
    expect(createPageDraft).not.toHaveBeenCalled();
    expect(onBack).not.toHaveBeenCalled();
  });

  it("blocks stale writes and reloads the latest server snapshot explicitly", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 4,
    });
    vi.mocked(getPage)
      .mockResolvedValueOnce(page())
      .mockResolvedValueOnce(page({ content: "Latest body", version: 4 }));
    vi.mocked(updatePageDraft).mockRejectedValueOnce(conflict);
    renderEditor({ draftId: "draft-1" });
    const content = await screen.findByRole("textbox", { name: "Content" });
    await userEvent.clear(content);
    await userEvent.type(content, "My body");

    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This draft changed elsewhere.",
    );
    expect(screen.queryByRole("button", { name: "Retry save" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Reload latest" }));
    await waitFor(() => expect(content).toHaveValue("Latest body"));
    expect(updatePageDraft).toHaveBeenCalledTimes(1);
  });

  it("keeps a stale conflict blocked when reload-latest fails", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 4,
    });
    vi.mocked(getPage)
      .mockResolvedValueOnce(page())
      .mockRejectedValueOnce(new Error("offline"));
    vi.mocked(updatePageDraft).mockRejectedValueOnce(conflict);
    renderEditor({ draftId: "draft-1" });
    const content = await screen.findByRole("textbox", { name: "Content" });
    await userEvent.clear(content);
    await userEvent.type(content, "My body");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    await screen.findByText("This draft changed elsewhere.");

    await userEvent.click(screen.getByRole("button", { name: "Reload latest" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Latest draft couldn't be loaded.",
    );
    expect(content).toHaveValue("My body");
    expect(screen.getByRole("button", { name: "Reload latest" })).toBeInTheDocument();
    expect(updatePageDraft).toHaveBeenCalledTimes(1);
  });

  it("offers Open existing and Rename draft for an exact-title conflict", async () => {
    const conflict = Object.assign(new Error("title conflict"), {
      code: "page_title_conflict",
      existingPageId: "page-existing",
      existingPageTitle: "Existing",
    });
    vi.mocked(getPage).mockResolvedValue(page());
    vi.mocked(publishPageDraft).mockRejectedValue(conflict);
    const { onOpenExisting } = renderEditor({ draftId: "draft-1" });
    const title = await screen.findByRole("textbox", { name: "Title" });
    const publish = screen.getByRole("button", { name: "Publish" });

    await userEvent.click(publish);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "A page with this title already exists.",
    );
    expect(publish).toBeDisabled();

    const openExisting = screen.getByRole("button", { name: "Open existing" });
    const rename = screen.getByRole("button", { name: "Rename draft" });
    expect(openExisting).toHaveClass("page-draft-conflict-action");
    expect(rename).toHaveClass("page-draft-conflict-action");

    const css = readFileSync(
      resolve("src/components/memory/pages/pageDraftEditor.css"),
      "utf8",
    );
    const actionRule = css.match(
      /\.page-draft-conflict-action\s*\{(?<body>[^}]*)\}/,
    )?.groups?.body;
    expect(actionRule).toContain("background: var(--mem-indigo-bg);");
    expect(actionRule).toContain("border: 1px solid var(--mem-control-border);");
    expect(actionRule).toContain("color: var(--mem-accent-indigo);");

    await userEvent.click(openExisting);
    expect(onOpenExisting).toHaveBeenCalledWith("page-existing");

    await userEvent.click(rename);
    expect(title).toHaveFocus();
    expect(title).toHaveSelection();
    expect(screen.queryByText("A page with this title already exists.")).not.toBeInTheDocument();
    expect(publish).toBeEnabled();
  });

  it("blocks generic republish after a stale publish CAS until reload-latest succeeds", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 4,
    });
    vi.mocked(getPage)
      .mockResolvedValueOnce(page())
      .mockResolvedValueOnce(page({ version: 4 }))
      .mockResolvedValueOnce(page({ version: 4 }));
    vi.mocked(publishPageDraft).mockRejectedValueOnce(conflict);
    renderEditor({ draftId: "draft-1" });
    const publish = await screen.findByRole("button", { name: "Publish" });

    await userEvent.click(publish);
    expect(await screen.findByText("This draft changed elsewhere.")).toBeInTheDocument();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
    expect(publish).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "Reload latest" }));
    await waitFor(() => expect(publish).toBeEnabled());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(publishPageDraft).toHaveBeenCalledTimes(1);
  });
});
