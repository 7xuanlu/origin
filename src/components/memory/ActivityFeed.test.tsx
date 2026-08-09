// SPDX-License-Identifier: AGPL-3.0-only
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { i18n } from "../../i18n";
import ActivityFeed from "./ActivityFeed";
import type { AgentActivityItem, AgentConnection } from "../../lib/tauri";

const activityMock = vi.hoisted(() => vi.fn());
const agentsMock = vi.hoisted(() => vi.fn());

vi.mock("../../lib/tauri", () => ({
  listAgentActivity: activityMock,
  listAgents: agentsMock,
}));

const agent = (name: string, displayName: string | null): AgentConnection => ({
  id: name,
  name,
  display_name: displayName,
  agent_type: "mcp",
  description: null,
  enabled: true,
  trust_level: "full",
  last_seen_at: null,
  memory_count: 0,
  created_at: 0,
  updated_at: 0,
});

const activity = (item: Partial<AgentActivityItem>): AgentActivityItem => ({
  id: item.id ?? 1,
  timestamp: item.timestamp ?? Math.floor(Date.now() / 1000),
  agent_name: item.agent_name ?? "codex",
  action: item.action ?? "store",
  memory_ids: item.memory_ids ?? null,
  query: item.query ?? null,
  detail: item.detail ?? null,
  memory_titles: item.memory_titles ?? [],
});

function renderActivityFeed() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActivityFeed onNavigateMemory={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("ActivityFeed i18n", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    await i18n.changeLanguage("zh-Hant");
    agentsMock.mockResolvedValue([
      agent("codex", "Codex"),
      agent("claude-code", "Claude Code"),
    ]);
  });

  it("localizes the empty activity state", async () => {
    activityMock.mockResolvedValue([]);

    renderActivityFeed();

    expect(
      await screen.findByText("你的 AI 工具與記憶互動時，會出現在這裡。"),
    ).toBeInTheDocument();
  });

  it("localizes visible activity chrome and event copy", async () => {
    const now = Math.floor(Date.now() / 1000);
    activityMock.mockResolvedValue([
      activity({
        id: 1,
        timestamp: now - 30,
        agent_name: "codex",
        action: "store",
        memory_ids: "memory-1",
        memory_titles: ["Goal command context"],
      }),
      activity({
        id: 2,
        timestamp: now - 120,
        agent_name: "claude-code",
        action: "search",
        query: "i18n",
        memory_titles: [],
      }),
    ]);

    renderActivityFeed();

    expect(await screen.findByText("今天")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "動作" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "代理" })).toBeInTheDocument();
    expect(screen.getAllByText("記住").length).toBeGreaterThan(0);
    expect(screen.getAllByText("回憶").length).toBeGreaterThan(0);
    expect(screen.getByText("記住了 1 則記憶")).toBeInTheDocument();
    expect(screen.getByText("搜尋了你的記憶")).toBeInTheDocument();
    expect(screen.getByText("由 Codex")).toBeInTheDocument();
    expect(screen.getByText("剛剛")).toBeInTheDocument();
  });

  it("localizes backend refinement and steep activity details", async () => {
    const now = Math.floor(Date.now() / 1000);
    activityMock.mockResolvedValue([
      activity({
        id: 1,
        timestamp: now - 30,
        agent_name: "origin",
        action: "steep",
        detail: "Wenlan resolved 10 memory contradictions",
      }),
      activity({
        id: 2,
        timestamp: now - 90,
        agent_name: "daemon",
        action: "refinement_resolve",
        detail: JSON.stringify({
          action: "detect_contradiction",
          new_status: "dismissed",
          source_ids: ["mem_a", "mem_b"],
        }),
      }),
    ]);

    renderActivityFeed();

    expect(await screen.findByText("Wenlan 文瀾解決了 10 個記憶矛盾")).toBeInTheDocument();
    expect(screen.getAllByText("解決精煉").length).toBeGreaterThan(0);
    expect(screen.getByText("將記憶矛盾標記為已忽略")).toBeInTheDocument();
    expect(screen.queryByText("REFINEMENT_RESOLVE")).not.toBeInTheDocument();
    expect(screen.queryByText(/detect_contradiction/)).not.toBeInTheDocument();
  });

  it.each([
    {
      language: "zh-Hant",
      actionFilter: "動作",
      labels: ["略過頁面更新", "新增觀察", "新增實體", "新增關聯", "自動取代關聯"],
    },
    {
      language: "zh-Hans",
      actionFilter: "动作",
      labels: ["跳过页面更新", "新增观察", "新增实体", "新增关系", "自动替换关系"],
    },
  ])("localizes backend action ids in the $language action filter", async ({ language, actionFilter, labels }) => {
    await i18n.changeLanguage(language);
    const now = Math.floor(Date.now() / 1000);
    const rawActions = [
      "page_skip_user_edited",
      "observation_add",
      "entity_create",
      "relation_create",
      "relation_supersede_auto",
    ];

    activityMock.mockResolvedValue(
      rawActions.map((action, index) =>
        activity({
          id: index + 1,
          timestamp: now - index * 30,
          action,
        }),
      ),
    );

    renderActivityFeed();

    expect(await screen.findByRole("combobox", { name: actionFilter })).toBeInTheDocument();
    for (const label of labels) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    for (const rawAction of rawActions) {
      expect(screen.queryByText(rawAction)).not.toBeInTheDocument();
    }
  });

  it("maps legacy backend steep wording to current English page copy", async () => {
    await i18n.changeLanguage("en");
    activityMock.mockResolvedValue([
      activity({
        action: "steep",
        detail: "Wenlan refreshed 2 concepts with new information",
      }),
    ]);

    renderActivityFeed();

    expect(
      await screen.findByText("Wenlan refreshed 2 pages with new information"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/concept/i)).not.toBeInTheDocument();
  });
});
