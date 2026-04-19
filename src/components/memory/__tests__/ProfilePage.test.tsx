import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ProfilePage from "../ProfilePage";

vi.mock("../../../lib/tauri", () => ({
  getProfile: vi.fn().mockResolvedValue({
    id: "1", name: "Lucian", display_name: "Lucian",
    email: "lucian@origin.dev", bio: "Building the knowledge layer",
    avatar_path: null, created_at: 1709251200,
  }),
  updateProfile: vi.fn().mockResolvedValue(null),
  setAvatar: vi.fn().mockResolvedValue(null),
  removeAvatar: vi.fn().mockResolvedValue(null),
  getProfileNarrative: vi.fn().mockResolvedValue({
    content: "You're a solo founder building Origin.",
    generated_at: 1744041600, is_stale: false, memory_count: 5,
  }),
  regenerateNarrative: vi.fn().mockResolvedValue(null),
  listMemoriesRich: vi.fn().mockImplementation((_domain, type) => {
    if (type === "preference") return Promise.resolve([
      { source_id: "p1", title: "Strict TDD", content: "Always TDD", memory_type: "preference", confirmed: true, pinned: false, last_modified: 1744041600 },
    ]);
    if (type === "goal") return Promise.resolve([
      { source_id: "g1", title: "Ship wiki", content: "Launch this weekend", memory_type: "goal", confirmed: true, pinned: false, last_modified: 1744041600 },
    ]);
    return Promise.resolve([]);
  }),
  MEMORY_FACETS: [],
  FACET_COLORS: {},
}));

function wrap(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe("ProfilePage (narrative-first)", () => {
  it("renders narrative brief", async () => {
    render(wrap(<ProfilePage onBack={() => {}} />));
    expect(await screen.findByText(/solo founder/)).toBeInTheDocument();
  });

  it("does not render Decision or Fact sections", async () => {
    render(wrap(<ProfilePage onBack={() => {}} />));
    await screen.findByText(/solo founder/);
    expect(screen.queryByText("Decision")).not.toBeInTheDocument();
    expect(screen.queryByText("Fact")).not.toBeInTheDocument();
  });

  it("shows Current Focus section", async () => {
    render(wrap(<ProfilePage onBack={() => {}} />));
    expect(await screen.findByText(/Current focus/)).toBeInTheDocument();
  });

  it("shows preference pills", async () => {
    render(wrap(<ProfilePage onBack={() => {}} />));
    expect(await screen.findByText(/Strict TDD/)).toBeInTheDocument();
  });
});
