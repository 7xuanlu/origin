// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../../i18n";

const mocks = vi.hoisted(() => ({
  openDialog: vi.fn(),
  detectVault: vi.fn(),
  addSource: vi.fn(),
  syncRegisteredSource: vi.fn(),
  listRegisteredSources: vi.fn(),
  detectObsidianVaults: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mocks.openDialog }));
vi.mock("../../../lib/vaultDetection", () => ({ detectVault: mocks.detectVault }));
vi.mock("../../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/tauri")>();
  return {
    ...actual,
    addSource: mocks.addSource,
    syncRegisteredSource: mocks.syncRegisteredSource,
    listRegisteredSources: mocks.listRegisteredSources,
    detectObsidianVaults: mocks.detectObsidianVaults,
  };
});

import VaultConnectCard from "./VaultConnectCard";

function renderCard(props: Partial<React.ComponentProps<typeof VaultConnectCard>> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <VaultConnectCard variant="wizard" {...props} />
    </QueryClientProvider>
  );
}

const SOURCE = {
  id: "s1",
  source_type: "obsidian",
  path: "/v",
  status: "Active",
  last_sync: null,
  file_count: 12,
  memory_count: 3,
};

describe("VaultConnectCard", () => {
  beforeEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
    mocks.openDialog.mockResolvedValue("/v");
    mocks.addSource.mockResolvedValue(SOURCE);
    mocks.syncRegisteredSource.mockResolvedValue({
      files_found: 12, ingested: 12, skipped: 0, errors: 0,
    });
    mocks.listRegisteredSources.mockResolvedValue([SOURCE]);
    mocks.detectObsidianVaults.mockResolvedValue([]);
  });

  // The wizard variant only records a pick — the "Setting up" step is the
  // one place that calls addSource/syncRegisteredSource now. There is no
  // "Connect" action left in this component at all in wizard mode.
  it("wizard variant never renders a Connect button — the wizard's own Continue is the only confirm", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: false, sourceType: "directory", docCount: 3,
      countCapped: false, hasValidDoc: true, unreadable: false,
    });
    renderCard();
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(screen.getByText(/3 supported files/)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Connect" })).not.toBeInTheDocument();
    expect(screen.getByText("Wenlan will import this when setup finishes.")).toBeInTheDocument();
  });

  it("a zero-count detection does not block the pick — it still reports up", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: false, sourceType: "directory", docCount: 0,
      countCapped: false, hasValidDoc: false, unreadable: false,
    });
    const onPick = vi.fn();
    renderCard({ onPick });
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(screen.getByText(/No notes found/)).toBeInTheDocument());
    await waitFor(() =>
      expect(onPick).toHaveBeenCalledWith({ sourceType: "directory", path: "/v", label: "v" }),
    );
  });

  it("browsing to a real Obsidian vault records the pick — it never calls addSource or syncRegisteredSource itself", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: true, sourceType: "obsidian", docCount: 12,
      countCapped: false, hasValidDoc: true, unreadable: false,
    });
    const onPick = vi.fn();
    renderCard({ onPick });
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(screen.getByText(/Detected \.obsidian\//)).toBeInTheDocument());
    await waitFor(() =>
      expect(onPick).toHaveBeenCalledWith({ sourceType: "obsidian", path: "/v", label: "v" }),
    );
    expect(mocks.addSource).not.toHaveBeenCalled();
    expect(mocks.syncRegisteredSource).not.toHaveBeenCalled();
  });

  it("renders Obsidian vault chips and clicking one populates the path input", async () => {
    mocks.detectObsidianVaults.mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);
    renderCard();
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());
    await userEvent.click(screen.getByText("Work Notes"));
    await waitFor(() =>
      expect(screen.getByDisplayValue("/Users/x/Vaults/Work Notes")).toBeInTheDocument()
    );
  });

  it("picking a chip reports onPick(obsidian, path) and never runs vault detection or addSource", async () => {
    mocks.detectObsidianVaults.mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);
    const onPick = vi.fn();
    renderCard({ onPick });
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());
    await userEvent.click(screen.getByText("Work Notes"));
    await waitFor(() =>
      expect(onPick).toHaveBeenCalledWith({
        sourceType: "obsidian",
        path: "/Users/x/Vaults/Work Notes",
        label: "Work Notes",
      }),
    );
    expect(mocks.detectVault).not.toHaveBeenCalled();
    expect(mocks.addSource).not.toHaveBeenCalled();
  });

  it("zero vaults: no chip row; browsing a folder still reports a pick", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: false, sourceType: "directory", docCount: 3,
      countCapped: false, hasValidDoc: true, unreadable: false,
    });
    const onPick = vi.fn();
    renderCard({ onPick });
    await userEvent.click(screen.getByText("Browse…"));
    // By the time detection resolves and re-renders, the (mocked, instantly
    // resolving) obsidian-vaults query has settled too — a reliable point to
    // assert its absence, unlike asserting right after render.
    await waitFor(() => expect(screen.getByText(/3 supported files/)).toBeInTheDocument());
    expect(screen.queryByText("Your Obsidian vaults")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(onPick).toHaveBeenCalledWith({ sourceType: "directory", path: "/v", label: "v" }),
    );
    expect(mocks.addSource).not.toHaveBeenCalled();
  });

  // Regression: handleBrowse is async — it awaits detectVault(selected) then
  // unconditionally setDetection(...). If the user picks a vault chip while
  // that scan is still in flight, the abandoned promise later resolves and
  // overwrites `detection` with results for the folder the user never ended
  // up choosing, contradicting the "Obsidian vault — <name>" line right
  // below it. Display-only — the pick itself keys off pickedVault/path.
  it("a browse scan that resolves after a vault chip was already picked does not overwrite the picked-vault display with stale info", async () => {
    mocks.detectObsidianVaults.mockResolvedValue([
      { name: "Work Notes", path: "/Users/x/Vaults/Work Notes" },
    ]);
    let resolveDetect!: (value: {
      isVault: boolean;
      sourceType: string;
      docCount: number;
      countCapped: boolean;
      hasValidDoc: boolean;
      unreadable: boolean;
    }) => void;
    mocks.detectVault.mockReturnValue(
      new Promise((resolve) => {
        resolveDetect = resolve;
      }),
    );

    renderCard();
    await waitFor(() => expect(screen.getByText("Work Notes")).toBeInTheDocument());

    // Start a folder browse — detectVault() is now in flight and won't
    // resolve until resolveDetect is called below.
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(mocks.detectVault).toHaveBeenCalledTimes(1));

    // Before that scan finishes, the user picks a real vault chip instead —
    // this is the discarding action.
    await userEvent.click(screen.getByText("Work Notes"));
    await waitFor(() =>
      expect(screen.getByText(/Obsidian vault — Work Notes/)).toBeInTheDocument()
    );

    // Now the abandoned browse-detection promise resolves, with results for
    // the folder the user never ended up choosing.
    resolveDetect({
      isVault: false, sourceType: "directory", docCount: 42,
      countCapped: false, hasValidDoc: false, unreadable: false,
    });
    await waitFor(() => expect(mocks.detectVault).toHaveResolvedTimes(1));

    // The picked-vault line is still the source of truth...
    expect(screen.getByText(/Obsidian vault — Work Notes/)).toBeInTheDocument();
    // ...and the stale detection result for the discarded folder must not
    // render alongside it.
    expect(screen.queryByText(/42 supported files/)).not.toBeInTheDocument();
  });

  // The chip row is conditional on detection, so it cannot be what tells a user
  // that Obsidian is supported — someone with Obsidian installed but no readable
  // registry entry would see no mention of it at all. Support is stated
  // unconditionally, and this pins the case the chips do NOT cover.
  it("states Obsidian support even when zero vaults are detected", async () => {
    renderCard();
    expect(
      await screen.findByText("Obsidian vaults — indexes your Markdown notes"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Any folder — indexes .md, .txt, and .pdf files"),
    ).toBeInTheDocument();
    // ...and it is genuinely the no-vault case, not the chip row in disguise.
    expect(screen.queryByText("Your Obsidian vaults")).not.toBeInTheDocument();
  });
});
