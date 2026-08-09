import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { createElement, type PropsWithChildren } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createPageDraft,
  discardPageDraft,
  updatePageDraft,
  type Page,
} from "../../../lib/tauri";
import {
  usePageDraftAutosave,
  type PageDraftSnapshot,
} from "./usePageDraftAutosave";

vi.mock("../../../lib/tauri", () => ({
  createPageDraft: vi.fn(),
  discardPageDraft: vi.fn(),
  updatePageDraft: vi.fn(),
}));

const EMPTY: PageDraftSnapshot = { title: "", content: "", space: null };

function page(
  overrides: Partial<Page> & Pick<Page, "id" | "title" | "content" | "version">,
): Page {
  return {
    summary: null,
    entity_id: null,
    domain: null,
    space: null,
    source_memory_ids: [],
    status: "draft",
    creation_kind: "authored",
    review_status: "unconfirmed",
    created_at: "2026-07-16T00:00:00Z",
    last_compiled: "2026-07-16T00:00:00Z",
    last_modified: "2026-07-16T00:00:00Z",
    ...overrides,
  };
}

function wrapper({ children }: PropsWithChildren) {
  return createElement(
    QueryClientProvider,
    { client: new QueryClient({ defaultOptions: { queries: { retry: false } } }) },
    children,
  );
}

describe("usePageDraftAutosave", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(createPageDraft).mockReset();
    vi.mocked(updatePageDraft).mockReset();
    vi.mocked(discardPageDraft).mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not write an untouched or Space-only draft", async () => {
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { ...EMPTY, space: "Research" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(701);
    });

    expect(createPageDraft).not.toHaveBeenCalled();
    expect(result.current.draftId).toBeNull();
  });

  it("persists a genuine title-only draft after 700ms", async () => {
    vi.mocked(createPageDraft).mockResolvedValue(page({
      id: "draft-1",
      title: "Working title",
      content: "",
      version: 1,
    }));
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { title: "Working title", content: "", space: null } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(699);
    });
    expect(createPageDraft).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(createPageDraft).toHaveBeenCalledWith({
      clientDraftId: expect.stringMatching(/^page_[0-9a-f-]+$/),
      title: "Working title",
      content: "",
      space: null,
    });
    expect(result.current.draftId).toBe("draft-1");
    expect(result.current.status).toBe("saved");
  });

  it("flushes text typed inside the debounce window without waiting 700ms", async () => {
    vi.mocked(createPageDraft).mockResolvedValue(page({
      id: "draft-quit",
      title: "Last keystroke",
      content: "",
      version: 1,
    }));
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { title: "Last keystroke", content: "", space: null } });
    await act(async () => {
      expect(await result.current.flush()).toBe(true);
    });

    expect(createPageDraft).toHaveBeenCalledWith({
      clientDraftId: expect.stringMatching(/^page_[0-9a-f-]+$/),
      title: "Last keystroke",
      content: "",
      space: null,
    });
    expect(result.current.draftId).toBe("draft-quit");
  });

  it("persists a body-only draft while preserving raw body whitespace", async () => {
    vi.mocked(createPageDraft).mockResolvedValue(page({
      id: "draft-1",
      title: "",
      content: "  body\n",
      version: 1,
    }));
    const { rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { title: "", content: "  body\n", space: null } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });

    expect(createPageDraft).toHaveBeenCalledWith({
      clientDraftId: expect.stringMatching(/^page_[0-9a-f-]+$/),
      title: "",
      content: "  body\n",
      space: null,
    });
  });

  it("serializes an edit made during an in-flight create into one ordered update", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    vi.mocked(updatePageDraft).mockResolvedValue(page({
      id: "draft-1",
      title: "Second",
      content: "Body",
      version: 2,
    }));
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { title: "First", content: "Body", space: null } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(createPageDraft).toHaveBeenCalledTimes(1);

    rerender({ snapshot: { title: "Second", content: "Body", space: null } });
    await act(async () => {
      resolveCreate(page({
        id: "draft-1",
        title: "First",
        content: "Body",
        version: 1,
      }));
      await Promise.resolve();
    });

    expect(updatePageDraft).toHaveBeenCalledTimes(1);
    expect(updatePageDraft).toHaveBeenCalledWith({
      id: "draft-1",
      expectedVersion: 1,
      title: "Second",
      content: "Body",
      space: null,
    });
    expect(result.current.version).toBe(2);
  });

  it("discards a persisted draft when both title and body become empty", async () => {
    vi.mocked(discardPageDraft).mockResolvedValue();
    const initial = { title: "Draft", content: "Body", space: "Work" };
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({
        draftId: "draft-1",
        initial,
        initialVersion: 4,
        snapshot,
      }),
      { initialProps: { snapshot: initial }, wrapper },
    );

    rerender({ snapshot: { title: " ", content: "\n", space: "Work" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });

    expect(discardPageDraft).toHaveBeenCalledWith({
      id: "draft-1",
      expectedVersion: 4,
    });
    expect(result.current.draftId).toBeNull();
  });

  it("preserves fields after a network error and retries the latest snapshot", async () => {
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(page({
        id: "draft-1",
        title: "Keep me",
        content: "Still here",
        version: 1,
      }));
    const snapshot = { title: "Keep me", content: "Still here", space: null };
    const { result } = renderHook(
      () => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe("offline");

    await act(async () => {
      expect(await result.current.retry()).toBe(true);
    });
    expect(createPageDraft).toHaveBeenCalledTimes(2);
    const [firstAttempt, secondAttempt] = vi.mocked(createPageDraft).mock.calls.map(
      ([input]) => input,
    );
    expect(firstAttempt?.clientDraftId).toMatch(/^page_[0-9a-f-]+$/);
    expect(secondAttempt?.clientDraftId).toBe(firstAttempt?.clientDraftId);
    expect(result.current.status).toBe("saved");
  });

  it("stops autosaving after a Space validation rejection until explicitly retried", async () => {
    const validation = new Error('Space "missing" is not registered');
    vi.mocked(createPageDraft).mockRejectedValue(validation);
    const snapshot = {
      title: "Strictly scoped",
      content: "Body",
      space: "missing",
    };
    const { result } = renderHook(
      () => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_100);
    });

    expect(createPageDraft).toHaveBeenCalledTimes(1);
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe(validation);
  });

  it("updates newer edits after an idempotent retry returns the earlier committed snapshot", async () => {
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(new Error("response lost"))
      .mockImplementationOnce(async (input) => page({
        id: input.clientDraftId,
        title: "First title",
        content: "First body",
        space: "Wenlan Core",
        version: 1,
      }));
    vi.mocked(updatePageDraft).mockImplementation(async (input) => page({
      id: input.id,
      title: input.title,
      content: input.content,
      space: input.space,
      version: 2,
    }));
    const first: PageDraftSnapshot = {
      title: "First title",
      content: "First body",
      space: "Wenlan",
    };
    const second: PageDraftSnapshot = {
      title: "Newer title",
      content: "Newer body",
      space: "Research",
    };
    const onSpaceReconciled = vi.fn();
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({
        initial: EMPTY,
        onSpaceReconciled,
        snapshot,
      }),
      { initialProps: { snapshot: first }, wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("error");

    rerender({ snapshot: second });
    await act(async () => {
      expect(await result.current.retry()).toBe(true);
    });

    const [firstAttempt, secondAttempt] = vi.mocked(createPageDraft).mock.calls.map(
      ([input]) => input,
    );
    expect(secondAttempt?.clientDraftId).toBe(firstAttempt?.clientDraftId);
    expect(secondAttempt).toMatchObject(first);
    expect(updatePageDraft).toHaveBeenCalledWith({
      id: firstAttempt?.clientDraftId,
      expectedVersion: 1,
      ...second,
    });
    expect(result.current.status).toBe("saved");
    expect(result.current.version).toBe(2);
    expect(onSpaceReconciled).not.toHaveBeenCalled();
  });

  it("adopts a renamed server Space after create replay instead of PUTting the obsolete name", async () => {
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(new Error("response lost"))
      .mockImplementationOnce(async (input) => page({
        id: input.clientDraftId,
        title: input.title,
        content: input.content,
        space: "Wenlan Core",
        version: 2,
      }));
    const snapshot: PageDraftSnapshot = {
      title: "Replay identity",
      content: "Body",
      space: "Wenlan",
    };
    const onSpaceReconciled = vi.fn();
    const { result } = renderHook(
      () => usePageDraftAutosave({
        initial: EMPTY,
        onSpaceReconciled,
        snapshot,
      }),
      { wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("error");

    await act(async () => {
      expect(await result.current.retry()).toBe(true);
    });

    expect(updatePageDraft).not.toHaveBeenCalled();
    expect(onSpaceReconciled).toHaveBeenCalledWith("Wenlan Core");
    expect(result.current.status).toBe("saved");
    expect(result.current.version).toBe(2);
  });

  it("reconciles and discards an ambiguous create after the editor is cleared", async () => {
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(new Error("response lost"))
      .mockImplementationOnce(async (input) => page({
        id: input.clientDraftId,
        title: "Transient title",
        content: "Transient body",
        version: 1,
      }));
    vi.mocked(discardPageDraft).mockResolvedValue();
    const first: PageDraftSnapshot = {
      title: "Transient title",
      content: "Transient body",
      space: null,
    };
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: first }, wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("error");

    rerender({ snapshot: EMPTY });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });

    const recoveredId = vi.mocked(createPageDraft).mock.calls[0]?.[0].clientDraftId;
    expect(createPageDraft).toHaveBeenCalledTimes(2);
    expect(discardPageDraft).toHaveBeenCalledWith({
      id: recoveredId,
      expectedVersion: 1,
    });
    expect(result.current.draftId).toBeNull();
    expect(result.current.status).toBe("idle");
  });

  it("finishes an ambiguous discard before saving replacement content", async () => {
    const missing = Object.assign(new Error("gone"), {
      code: "page_draft_not_found",
    });
    vi.mocked(discardPageDraft)
      .mockRejectedValueOnce(new Error("response lost"))
      .mockRejectedValueOnce(missing);
    vi.mocked(createPageDraft).mockImplementation(async (input) => page({
      id: input.clientDraftId,
      title: input.title,
      content: input.content,
      version: 1,
    }));
    const initial: PageDraftSnapshot = {
      title: "Discard me",
      content: "Body",
      space: null,
    };
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({
        draftId: "draft-1",
        initial,
        initialVersion: 3,
        snapshot,
      }),
      { initialProps: { snapshot: initial }, wrapper },
    );

    rerender({ snapshot: EMPTY });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("error");
    expect(result.current.draftId).toBe("draft-1");

    rerender({ snapshot: { title: "Replacement", content: "", space: null } });
    await act(async () => {
      expect(await result.current.retry()).toBe(true);
    });

    expect(createPageDraft).toHaveBeenCalledWith({
      clientDraftId: expect.stringMatching(/^page_[0-9a-f-]+$/),
      title: "Replacement",
      content: "",
      space: null,
    });
    expect(updatePageDraft).not.toHaveBeenCalled();
    expect(result.current.status).toBe("saved");
    expect(result.current.draftId).toMatch(/^page_[0-9a-f-]+$/);
  });

  it("rotates a collided client draft id once before surfacing an error", async () => {
    const collision = Object.assign(new Error("collision"), {
      code: "page_draft_id_conflict",
    });
    vi.mocked(createPageDraft)
      .mockRejectedValueOnce(collision)
      .mockImplementationOnce(async (input) => page({
        id: input.clientDraftId,
        title: input.title,
        content: input.content,
        version: 1,
      }));
    const snapshot = { title: "Unique draft", content: "", space: null };
    const { result } = renderHook(
      () => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });

    const [collided, replacement] = vi.mocked(createPageDraft).mock.calls.map(
      ([input]) => input.clientDraftId,
    );
    expect(collided).toMatch(/^page_[0-9a-f-]+$/);
    expect(replacement).toMatch(/^page_[0-9a-f-]+$/);
    expect(replacement).not.toBe(collided);
    expect(result.current.status).toBe("saved");
  });

  it("surfaces save state after React StrictMode replays mount effects", async () => {
    vi.mocked(createPageDraft).mockRejectedValue(new Error("offline"));
    const snapshot = { title: "Keep me", content: "", space: null };
    const { result } = renderHook(
      () => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { reactStrictMode: true, wrapper },
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });

    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe("offline");
  });

  it("blocks a stale CAS generic retry until a remote snapshot is adopted", async () => {
    const conflict = Object.assign(new Error("stale"), {
      code: "draft_version_conflict",
      currentVersion: 8,
    });
    vi.mocked(updatePageDraft).mockRejectedValueOnce(conflict);
    const initial = { title: "Draft", content: "Old", space: null };
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({
        draftId: "draft-1",
        initial,
        initialVersion: 7,
        snapshot,
      }),
      { initialProps: { snapshot: initial }, wrapper },
    );

    rerender({ snapshot: { ...initial, content: "Mine" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    expect(result.current.status).toBe("conflict");

    await act(async () => {
      expect(await result.current.retry()).toBe(false);
    });
    expect(updatePageDraft).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.adoptRemote({
        draftId: "draft-1",
        version: 8,
        snapshot: { title: "Draft", content: "Latest", space: null },
      });
    });
    expect(result.current.status).toBe("saved");
    expect(result.current.version).toBe(8);
  });

  it("flush waits for the in-flight write and its ordered follow-up", async () => {
    let resolveCreate!: (value: Page) => void;
    vi.mocked(createPageDraft).mockReturnValue(new Promise((resolve) => {
      resolveCreate = resolve;
    }));
    vi.mocked(updatePageDraft).mockResolvedValue(page({
      id: "draft-1",
      title: "Second",
      content: "Body",
      version: 2,
    }));
    const { result, rerender } = renderHook(
      ({ snapshot }) => usePageDraftAutosave({ initial: EMPTY, snapshot }),
      { initialProps: { snapshot: EMPTY }, wrapper },
    );

    rerender({ snapshot: { title: "First", content: "Body", space: null } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(700);
    });
    rerender({ snapshot: { title: "Second", content: "Body", space: null } });

    let flushResult: boolean | undefined;
    await act(async () => {
      const flushing = result.current.flush().then((value) => {
        flushResult = value;
      });
      resolveCreate(page({
        id: "draft-1",
        title: "First",
        content: "Body",
        version: 1,
      }));
      await flushing;
    });

    expect(flushResult).toBe(true);
    expect(updatePageDraft).toHaveBeenCalledWith(expect.objectContaining({
      expectedVersion: 1,
      title: "Second",
    }));
  });
});
