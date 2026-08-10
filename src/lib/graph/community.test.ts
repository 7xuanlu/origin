// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { listCommunities, listCommunityMembers } from "../tauri";
import type { CommunityListResponse, CommunityMembersResponse } from "../tauri";
import {
  fetchSpaceCartography,
  fetchCartographyForSpaces,
  aggregateCartographyStatus,
} from "./community";

vi.mock("../tauri", async () => {
  const actual = await vi.importActual<typeof import("../tauri")>("../tauri");
  return {
    ...actual,
    listCommunities: vi.fn(),
    listCommunityMembers: vi.fn(),
  };
});

const mockListCommunities = vi.mocked(listCommunities);
const mockListCommunityMembers = vi.mocked(listCommunityMembers);

function communitiesPage(
  overrides: Partial<CommunityListResponse> = {},
): CommunityListResponse {
  return {
    schema_version: "community-read-v1",
    communities: [],
    next_cursor: null,
    ...overrides,
  };
}

function membersPage(overrides: Partial<CommunityMembersResponse> = {}): CommunityMembersResponse {
  return {
    schema_version: "community-read-v1",
    members: [],
    next_cursor: null,
    ...overrides,
  };
}

function summary(overrides: Partial<CommunityListResponse["communities"][number]> = {}) {
  return {
    community_id: "c1",
    space: "Work",
    display_name: null,
    member_count: 1,
    published_generation: 3,
    algo_version: "v1",
    projection_version: "v1",
    ...overrides,
  };
}

function member(overrides: Partial<CommunityMembersResponse["members"][number]> = {}) {
  return {
    space: "Work",
    node_id: "e1",
    node_kind: "entity",
    community_id: "c1",
    published_generation: 3,
    attachment: "core",
    ...overrides,
  };
}

describe("fetchSpaceCartography", () => {
  beforeEach(() => {
    mockListCommunities.mockReset();
    mockListCommunityMembers.mockReset();
  });

  it("parses a drained, generation-consistent read as ready with the member->community map", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ member_count: 2 })] }),
    );
    mockListCommunityMembers.mockResolvedValue(
      membersPage({ members: [member({ node_id: "e1" }), member({ node_id: "e2" })] }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result.status).toBe("ready");
    expect(result.memberCommunityId?.get("e1")).toBe("c1");
    expect(result.memberCommunityId?.get("e2")).toBe("c1");
  });

  it("flags a community whose drained member rows fall short of its summary member_count as a partial-error, not vacuously ready", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ member_count: 3 })] }),
    );
    // Pagination drains to next_cursor: null after only 2 rows — the daemon
    // said 3.
    mockListCommunityMembers.mockResolvedValue(
      membersPage({ members: [member({ node_id: "e1" }), member({ node_id: "e2" })] }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "member-count-mismatch" });
  });

  it("rejects a read that pads a community's member_count with duplicate rows instead of distinct members", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ member_count: 2 })] }),
    );
    // The same node twice across two pages. Counting raw rows reads 2 and
    // calls the read complete, while only ONE member is actually assigned.
    mockListCommunityMembers
      .mockResolvedValueOnce(
        membersPage({
          members: [member({ node_id: "e1" })],
          next_cursor: { space: "Work", node_id: "e1" },
        }),
      )
      .mockResolvedValueOnce(membersPage({ members: [member({ node_id: "e1" })] }));

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "member-count-mismatch" });
  });

  it("rejects a node the same read assigns to two different communities", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({
        communities: [
          summary({ community_id: "c1", member_count: 1 }),
          summary({ community_id: "c2", member_count: 1 }),
        ],
      }),
    );
    // Both summaries reconcile at one member each, so counting alone is happy
    // — but a node cannot be in two communities, and last-write-wins would
    // quietly hand the renderer c2.
    mockListCommunityMembers.mockResolvedValue(
      membersPage({
        members: [
          member({ node_id: "e1", community_id: "c1" }),
          member({ node_id: "e1", community_id: "c2" }),
        ],
      }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "conflicting-assignment" });
  });

  it("rejects a member row pointing at a community id no summary ever declared", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ community_id: "c1", member_count: 1 })] }),
    );
    // c1 reconciles exactly, so a per-summary count check alone never even
    // looks at the orphan row — but that row would still be handed to the
    // renderer as a real community assignment.
    mockListCommunityMembers.mockResolvedValue(
      membersPage({
        members: [
          member({ node_id: "e1", community_id: "c1" }),
          member({ node_id: "e2", community_id: "ghost" }),
        ],
      }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "unknown-community" });
  });

  it("returns ready when every community's drained member count matches its summary", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ member_count: 2 })] }),
    );
    mockListCommunityMembers.mockResolvedValue(
      membersPage({ members: [member({ node_id: "e1" }), member({ node_id: "e2" })] }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result.status).toBe("ready");
  });

  // Deliberate semantic, pinned so it can't be "fixed" by accident: a
  // community the daemon declares as empty is an EXACT match at zero members,
  // not a suspiciously short read. It stays ready (with nothing to assign) —
  // unlike a wholly empty response, which means no durable data was published
  // for this space at all and classifies as fallback.
  it("treats a community declared with member_count 0 and no member rows as ready", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ member_count: 0 })] }),
    );
    mockListCommunityMembers.mockResolvedValue(membersPage());

    const result = await fetchSpaceCartography("Work");

    expect(result.status).toBe("ready");
    expect(result.memberCommunityId?.size).toBe(0);
  });

  it("paginates both communities and members to exhaustion before deciding", async () => {
    mockListCommunities
      .mockResolvedValueOnce(
        communitiesPage({ communities: [summary({ community_id: "c1" })], next_cursor: "cursor-1" }),
      )
      .mockResolvedValueOnce(communitiesPage({ communities: [summary({ community_id: "c2" })] }));
    mockListCommunityMembers
      .mockResolvedValueOnce(
        membersPage({
          members: [member({ node_id: "e1", community_id: "c1" })],
          next_cursor: { space: "Work", node_id: "e1" },
        }),
      )
      .mockResolvedValueOnce(
        membersPage({ members: [member({ node_id: "e2", community_id: "c2" })] }),
      );

    const result = await fetchSpaceCartography("Work");

    expect(result.status).toBe("ready");
    expect(result.memberCommunityId?.get("e1")).toBe("c1");
    expect(result.memberCommunityId?.get("e2")).toBe("c2");
    expect(mockListCommunities).toHaveBeenCalledTimes(2);
    expect(mockListCommunities).toHaveBeenNthCalledWith(2, "Work", "cursor-1");
    expect(mockListCommunityMembers).toHaveBeenCalledTimes(2);
    expect(mockListCommunityMembers).toHaveBeenNthCalledWith(2, "Work", {
      space: "Work",
      node_id: "e1",
    });
  });

  it("flags a member row whose generation moved mid-pagination as a generation mismatch, not ready", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ published_generation: 3 })] }),
    );
    mockListCommunityMembers
      .mockResolvedValueOnce(
        membersPage({
          members: [member({ node_id: "e1", published_generation: 3 })],
          next_cursor: { space: "Work", node_id: "e1" },
        }),
      )
      // The daemon bumped the generation between the two member pages.
      .mockResolvedValueOnce(
        membersPage({ members: [member({ node_id: "e2", published_generation: 4 })] }),
      );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "generation-mismatch" });
  });

  it("treats an empty-but-successful read (no durable data published yet) as the ordinary fallback, not an error", async () => {
    mockListCommunities.mockResolvedValue(communitiesPage());
    mockListCommunityMembers.mockResolvedValue(membersPage());

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "fallback" });
  });

  it("surfaces a 404/405 (route missing on an old daemon) as a visible partial-error, not a silent fallback", async () => {
    mockListCommunities.mockRejectedValue(JSON.stringify({ status: 404, error: "no route" }));

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "old-daemon" });
  });

  it("reads a 422 (space name the daemon never registered) as the ordinary fallback", async () => {
    // The Atlas derives its space list from free-text entity space/domain, so
    // a name no Space registry knows is routine. The daemon answers 422; that
    // space simply has no durable community data. Calling it an error would
    // paint the whole worst-first badge red for an ordinary corpus.
    mockListCommunities.mockRejectedValue(
      JSON.stringify({ status: 422, error: "unknown Space: Scratch" }),
    );

    const result = await fetchSpaceCartography("Scratch");

    expect(result).toEqual({ status: "fallback" });
  });

  it("reads a 422 on the member read as fallback too", async () => {
    // The two reads race against Space deletion: the summaries can land just
    // before the Space goes away, leaving the member read to 422 on its own.
    mockListCommunities.mockResolvedValue(communitiesPage());
    mockListCommunityMembers.mockRejectedValue(
      JSON.stringify({ status: 422, error: "unknown Space: Scratch" }),
    );

    const result = await fetchSpaceCartography("Scratch");

    expect(result).toEqual({ status: "fallback" });
  });

  it("surfaces a plain transport failure distinctly from an old daemon", async () => {
    mockListCommunities.mockRejectedValue(new Error("connection reset"));

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "transport" });
  });

  it("surfaces a schema_version mismatch as its own partial-error reason", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ schema_version: "community-read-v2" }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "schema-mismatch" });
  });

  it("gives up as interrupted if a daemon never drains the cursor, instead of looping forever", async () => {
    mockListCommunities.mockResolvedValue(communitiesPage({ communities: [summary()] }));
    mockListCommunityMembers.mockResolvedValue(
      membersPage({ members: [member()], next_cursor: { space: "Work", node_id: "e1" } }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "interrupted" });
  });

  it("treats members reported without any community summary as a generation mismatch instead of vacuously ready", async () => {
    mockListCommunities.mockResolvedValue(communitiesPage());
    mockListCommunityMembers.mockResolvedValue(membersPage({ members: [member()] }));

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "generation-mismatch" });
  });

  it("flags a member row from a different space than requested as a partial-error instead of silently mixing spaces", async () => {
    mockListCommunities.mockResolvedValue(communitiesPage({ communities: [summary()] }));
    mockListCommunityMembers.mockResolvedValue(
      membersPage({ members: [member({ node_id: "e1", space: "OtherSpace" })] }),
    );

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "foreign-space" });
  });

  it("flags a community summary row from a different space than requested as a partial-error", async () => {
    mockListCommunities.mockResolvedValue(
      communitiesPage({ communities: [summary({ space: "OtherSpace" })] }),
    );
    mockListCommunityMembers.mockResolvedValue(membersPage());

    const result = await fetchSpaceCartography("Work");

    expect(result).toEqual({ status: "partial-error", reason: "foreign-space" });
  });
});

describe("fetchCartographyForSpaces / aggregateCartographyStatus", () => {
  beforeEach(() => {
    mockListCommunities.mockReset();
    mockListCommunityMembers.mockReset();
  });

  it("classifies each space independently — one space's failure never blocks another's", async () => {
    mockListCommunities.mockImplementation(async (space: string) =>
      communitiesPage({
        communities: space === "Ready" ? [summary({ space: "Ready" })] : [],
      }),
    );
    mockListCommunityMembers.mockImplementation(async (space: string) => {
      if (space === "Ready") return membersPage({ members: [member({ space: "Ready" })] });
      if (space === "Broken") throw new Error("boom");
      return membersPage();
    });

    const bySpace = await fetchCartographyForSpaces(["Ready", "Empty", "Broken"]);

    expect(bySpace.get("Ready")?.status).toBe("ready");
    expect(bySpace.get("Empty")?.status).toBe("fallback");
    expect(bySpace.get("Broken")).toEqual({ status: "partial-error", reason: "transport" });
  });

  it("aggregates worst-first: partial-error beats fallback beats ready", () => {
    const ready = new Map([["A", { status: "ready" as const, memberCommunityId: new Map() }]]);
    const fallback = new Map([
      ["A", { status: "ready" as const, memberCommunityId: new Map() }],
      ["B", { status: "fallback" as const }],
    ]);
    const partialError = new Map([
      ["A", { status: "fallback" as const }],
      ["B", { status: "partial-error" as const, reason: "transport" as const }],
    ]);

    expect(aggregateCartographyStatus(ready)).toBe("ready");
    expect(aggregateCartographyStatus(fallback)).toBe("fallback");
    expect(aggregateCartographyStatus(partialError)).toBe("partial-error");
    expect(aggregateCartographyStatus(new Map())).toBeNull();
  });

  it("never claims all-durable while a null-space (unscoped) node is present, even when every known space is ready", () => {
    const ready = new Map([["A", { status: "ready" as const, memberCommunityId: new Map() }]]);

    expect(aggregateCartographyStatus(ready, true)).toBe("fallback");
    expect(aggregateCartographyStatus(ready, false)).toBe("ready");
    // No known spaces at all, but an unscoped node still exists — the
    // aggregate must report its presence instead of null ("nothing to
    // report on").
    expect(aggregateCartographyStatus(new Map(), true)).toBe("fallback");
  });
});
