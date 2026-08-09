// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULTS, HANDLERS, liveInvoke } from "./live-invoke";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TAURI_TS_PATH = resolve(__dirname, "../../src/lib/tauri.ts");

/** Every command name passed to invoke(...) in src/lib/tauri.ts — both the
 *  plain `invoke("name", ...)` form and the generic-typed `invoke<T>("name")`
 *  form. */
function invokedCommands(source: string): Set<string> {
  const re = /\binvoke(?:<[^>]*>)?\(\s*"([^"]+)"/g;
  const out = new Set<string>();
  let m: RegExpExecArray | null;
  while ((m = re.exec(source))) out.add(m[1]);
  return out;
}

describe("liveInvoke list_all_tags", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("preserves the full TagData shape (tags + document_tags)", async () => {
    const payload = {
      tags: ["alpha", "beta"],
      document_tags: { "memory::m1": ["alpha"] },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 })),
    );

    const result = (await liveInvoke("list_all_tags")) as {
      tags: string[];
      document_tags: Record<string, string[]>;
    };

    expect(result.tags).toEqual(["alpha", "beta"]);
    expect(result.document_tags).toEqual({ "memory::m1": ["alpha"] });
  });
});

describe("liveInvoke authored Page preview", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates and reads a synthetic Page without writing to the live daemon", async () => {
    const fetch = vi.fn(() => Promise.reject(new Error("preview creation must stay local")));
    vi.stubGlobal("fetch", fetch);

    const created = (await liveInvoke("create_page", {
      title: "Preview-only Page",
      content: "A fixture-safe draft.",
      space: null,
    })) as { id: string; attached_to: string | null; warnings: string[] };
    const page = (await liveInvoke("get_page", { id: created.id })) as {
      id: string;
      title: string;
      creation_kind: string;
    };

    expect(created.attached_to).toBeNull();
    expect(page).toMatchObject({
      id: created.id,
      title: "Preview-only Page",
      creation_kind: "authored",
    });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("deletes a synthetic Page locally without mutating the live daemon", async () => {
    const fetch = vi.fn(() => Promise.reject(new Error("synthetic deletion must stay local")));
    vi.stubGlobal("fetch", fetch);
    const created = await liveInvoke("create_page", {
      title: "Delete locally",
      content: "Preview-only content.",
      space: null,
    }) as { id: string };

    await expect(liveInvoke("delete_page", { id: created.id })).resolves.toBeNull();
    await expect(liveInvoke("get_page", { id: created.id })).resolves.toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("maps daemon-backed Page deletion to DELETE rather than Archive", async () => {
    const fetch = vi.fn(async () =>
      new Response(JSON.stringify({ status: "deleted" }), { status: 200 })
    );
    vi.stubGlobal("fetch", fetch);

    await liveInvoke("delete_page", { id: "remote-page" });

    expect(fetch).toHaveBeenCalledWith(
      "/daemon/api/pages/remote-page",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("models partial draft snapshots, CAS updates, publish, discard, and title conflicts in memory", async () => {
    const fetch = vi.fn(async (input) =>
      new Response(JSON.stringify(
        String(input).endsWith("/api/spaces")
          ? [{ name: "Wenlan" }]
          : { pages: [] },
      ), { status: 200 })
    );
    vi.stubGlobal("fetch", fetch);

    const draft = (await liveInvoke("create_page_draft", {
      clientDraftId: "page_00000000-0000-4000-8000-000000000011",
      title: "",
      content: "  partial body  \n",
      space: " Wenlan ",
    })) as { id: string; version: number; status: string; content: string; space: string };
    expect(draft).toMatchObject({
      version: 1,
      status: "draft",
      content: "  partial body  \n",
      space: "Wenlan",
    });

    await expect(liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: 1,
      title: " ",
      content: "\n",
      space: "Wenlan",
    })).rejects.toThrow('"code":"invalid_page_draft"');

    await expect(liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: 0,
      title: "Wrong version",
      content: "Body",
      space: null,
    })).rejects.toThrow('"code":"draft_version_conflict"');

    const updated = (await liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: 1,
      title: "Preview draft",
      content: "  publish body  ",
      space: null,
    })) as { id: string; version: number; status: string };
    expect(updated).toMatchObject({ id: draft.id, version: 2, status: "draft" });

    const published = (await liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: 2,
    })) as { id: string; version: number; status: string; review_status: string };
    expect(published).toMatchObject({
      id: draft.id,
      version: 3,
      status: "active",
      review_status: "unconfirmed",
    });

    const disposable = (await liveInvoke("create_page_draft", {
      clientDraftId: "page_00000000-0000-4000-8000-000000000012",
      title: "Discard me",
      content: "",
      space: null,
    })) as { id: string; version: number };
    await expect(liveInvoke("discard_page_draft", {
      id: disposable.id,
      expectedVersion: disposable.version,
    })).resolves.toBeNull();
    await expect(liveInvoke("get_page", { id: disposable.id })).resolves.toBeNull();

    await liveInvoke("create_page", {
      title: "Collision title",
      content: "Existing active body",
      space: null,
    });
    const collision = (await liveInvoke("create_page_draft", {
      clientDraftId: "page_00000000-0000-4000-8000-000000000013",
      title: " Collision title ",
      content: "Draft body",
      space: null,
    })) as { id: string; version: number };
    await expect(liveInvoke("publish_page_draft", {
      id: collision.id,
      expectedVersion: collision.version,
    })).rejects.toThrow('"code":"page_title_conflict"');

    expect(fetch).toHaveBeenCalled();
    for (const [, init] of fetch.mock.calls) {
      expect(init).toMatchObject({ method: "GET" });
    }
  });

  it("deduplicates an ambiguous create retry by its client-generated draft id", async () => {
    const input = {
      clientDraftId: "page_22222222-2222-4222-8222-222222222222",
      title: "Preview retry",
      content: "The response may have been lost.",
      space: null,
    };

    const first = await liveInvoke("create_page_draft", input);
    const retried = await liveInvoke("create_page_draft", input);

    expect(retried).toEqual(first);
    expect(retried).toMatchObject({ id: input.clientDraftId, version: 1 });
  });

  it("requires client draft ids to use the canonical page_<uuid-v4> form", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    for (const clientDraftId of [
      "caller-controlled-primary-key",
      "page_00000000-0000-0000-8000-000000000001",
      "page_00000000-0000-4000-0000-000000000001",
      "page_00000000000040008000000000000001",
      "page_00000000-0000-4000-8000-000000000001-extra",
    ]) {
      await expect(liveInvoke("create_page_draft", {
        clientDraftId,
        title: "Invalid id",
        content: "Body",
        space: null,
      })).rejects.toThrow("Page draft id must use the page_<uuid-v4> format");
    }

    expect(fetch).not.toHaveBeenCalled();
  });

  it("replays only the immutable first create request", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify([{ name: "Wenlan" }]), { status: 200 })
      ),
    );
    const original = {
      clientDraftId: "page_33333333-3333-4333-8333-333333333333",
      title: "Replay identity",
      content: "Exact body  \n",
      space: "Wenlan",
    };

    const created = await liveInvoke("create_page_draft", original);
    const replayed = await liveInvoke("create_page_draft", original);

    expect(replayed).toEqual(created);
    expect(replayed).toMatchObject({ space: "Wenlan", version: 1 });
    await expect(liveInvoke("create_page_draft", {
      ...original,
      space: "missing-after-rename",
    })).rejects.toThrow('"code":"page_draft_id_conflict"');
    await expect(liveInvoke("create_page_draft", {
      ...original,
      content: "Different body",
      space: "missing-after-rename",
    })).rejects.toThrow('"code":"page_draft_id_conflict"');
  });

  it("replays an exact ambiguous update but rejects a divergent stale update", async () => {
    const draft = await liveInvoke("create_page_draft", {
      clientDraftId: "page_33333333-3333-4333-8333-333333333334",
      title: "Update replay",
      content: "Original",
      space: null,
    }) as { id: string; version: number };
    const request = {
      id: draft.id,
      expectedVersion: draft.version,
      title: "Update replay",
      content: "Committed",
      space: null,
    };

    const committed = await liveInvoke("update_page_draft", request);
    const replayed = await liveInvoke("update_page_draft", request);

    expect(replayed).toEqual(committed);
    await expect(liveInvoke("update_page_draft", {
      ...request,
      content: "Different",
    })).rejects.toThrow('"code":"draft_version_conflict"');
  });

  it("keeps a scrubbed UUID tombstone after discard", async () => {
    const request = {
      clientDraftId: "page_33333333-3333-4333-8333-333333333335",
      title: "Discarded",
      content: "Sensitive draft body",
      space: null,
    };
    const draft = await liveInvoke("create_page_draft", request) as {
      id: string;
      version: number;
    };
    await liveInvoke("discard_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });

    await expect(liveInvoke("create_page_draft", request))
      .rejects.toThrow('"code":"page_draft_id_conflict"');
    await expect(liveInvoke("create_page_draft", {
      ...request,
      title: "Different",
      content: "Request",
    })).rejects.toThrow('"code":"page_draft_id_conflict"');
    await expect(liveInvoke("get_page", { id: draft.id })).resolves.toBeNull();
  });

  it("rejects unregistered nonblank Spaces while treating whitespace and null as unscoped", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify([{ name: "Wenlan" }]), { status: 200 })
      ),
    );

    await expect(liveInvoke("create_page_draft", {
      clientDraftId: "page_44444444-4444-4444-8444-444444444441",
      title: "Strict scope",
      content: "Body",
      space: "missing",
    })).rejects.toThrow('Space "missing" is not registered');

    const whitespace = await liveInvoke("create_page_draft", {
      clientDraftId: "page_44444444-4444-4444-8444-444444444442",
      title: "Whitespace scope",
      content: "Body",
      space: " \n ",
    });
    const unscoped = await liveInvoke("create_page_draft", {
      clientDraftId: "page_44444444-4444-4444-8444-444444444443",
      title: "Null scope",
      content: "Body",
      space: null,
    });

    expect(whitespace).toMatchObject({ space: null, domain: null });
    expect(unscoped).toMatchObject({ space: null, domain: null });
  });

  it("reports stale versions before validating an updated Space", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify([{ name: "Wenlan" }]), { status: 200 })
      ),
    );
    const draft = await liveInvoke("create_page_draft", {
      clientDraftId: "page_55555555-5555-4555-8555-555555555555",
      title: "Update precedence",
      content: "Body",
      space: "Wenlan",
    }) as { id: string };

    await expect(liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: 0,
      title: "Stale",
      content: "Stale body",
      space: "missing",
    })).rejects.toThrow('"code":"draft_version_conflict"');

    await expect(liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: 1,
      title: "Current",
      content: "Current body",
      space: "missing",
    })).rejects.toThrow('Space "missing" is not registered');
  });

  it("replays only the immediately committed active Page after an ambiguous publish", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ pages: [] }), { status: 200 })
      ),
    );
    const draft = await liveInvoke("create_page_draft", {
      clientDraftId: "page_77777777-7777-4777-8777-777777777777",
      title: "Publish retry",
      content: "Body",
      space: null,
    }) as { id: string; version: number };

    const published = await liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });
    const replayed = await liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });

    expect(replayed).toEqual(published);
    expect(replayed).toMatchObject({ status: "active", version: draft.version + 1 });
    for (const expectedVersion of [draft.version - 1, draft.version + 2]) {
      await expect(liveInvoke("publish_page_draft", {
        id: draft.id,
        expectedVersion,
      })).rejects.toThrow(
        `"code":"draft_version_conflict","error":"Page draft changed since it was loaded","current_version":${draft.version + 1}`,
      );
    }
    await expect(liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version + 1,
    })).rejects.toThrow("is not a draft");

    await liveInvoke("update_page", {
      id: draft.id,
      content: "Later edit",
    });
    await expect(liveInvoke("get_page", { id: draft.id })).resolves.toMatchObject({
      content: "Later edit",
      version: draft.version + 2,
    });
    await expect(liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    })).rejects.toThrow(
      `"code":"draft_version_conflict","error":"Page draft changed since it was loaded","current_version":${draft.version + 2}`,
    );
  });

  it("publishes and stores the backend-canonical body without reserved Sources blocks", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ pages: [] }), { status: 200 })
      ),
    );
    const start = "<!-- origin:sources:start -->";
    const end = "<!-- origin:sources:end -->";
    const raw = `Lead prose  \n\n${start}\n## Sources\n- [[mem_1]]\n${end}\n\nTail prose  \n\n`;
    const expected = "Lead prose\n\nTail prose";
    const draft = await liveInvoke("create_page_draft", {
      clientDraftId: "page_99999999-9999-4999-8999-999999999999",
      title: "Canonical publish",
      content: raw,
      space: null,
    }) as { id: string; version: number };

    await expect(liveInvoke("get_page", { id: draft.id })).resolves.toMatchObject({
      content: raw,
      status: "draft",
    });

    const published = await liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });
    const stored = await liveInvoke("get_page", { id: draft.id });

    expect(published).toMatchObject({ content: expected });
    expect(stored).toMatchObject({ content: expected });
  });

  it("does not treat a reserved Sources block as meaningful draft content", async () => {
    const sourcesOnly = "<!-- origin:sources:start -->\n## Sources\n- [[mem_1]]\n<!-- origin:sources:end -->";

    await expect(liveInvoke("create_page_draft", {
      clientDraftId: "page_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      title: " ",
      content: sourcesOnly,
      space: null,
    })).rejects.toThrow('"code":"invalid_page_draft"');

    const draft = await liveInvoke("create_page_draft", {
      clientDraftId: "page_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
      title: "Temporary title",
      content: "",
      space: null,
    }) as { id: string; version: number };
    await expect(liveInvoke("update_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
      title: " ",
      content: sourcesOnly,
      space: null,
    })).rejects.toThrow('"code":"invalid_page_draft"');
  });

  it("rejects a Space-only empty draft instead of creating an impossible local row", async () => {
    const fetch = vi.fn();
    vi.stubGlobal("fetch", fetch);

    await expect(liveInvoke("create_page_draft", {
      clientDraftId: "page_00000000-0000-4000-8000-000000000014",
      title: "   ",
      content: "\n ",
      space: "Wenlan",
    })).rejects.toThrow('"code":"invalid_page_draft"');

    expect(fetch).not.toHaveBeenCalled();
  });

  it("checks daemon-backed active Pages before publishing a synthetic draft", async () => {
    const remotePage = {
      id: "remote-existing",
      title: "Remote collision",
      content: "Existing body",
      space: "Remote scope",
      domain: "Remote scope",
      status: "active",
      version: 1,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input) =>
        new Response(JSON.stringify(
          String(input).endsWith("/api/spaces")
            ? [{ name: "Remote scope" }]
            : { pages: [remotePage] },
        ), { status: 200 })
      ),
    );

    const draft = (await liveInvoke("create_page_draft", {
      clientDraftId: "page_00000000-0000-4000-8000-000000000015",
      title: " remote COLLISION ",
      content: "Synthetic body",
      space: "Remote scope",
    })) as { id: string; version: number };

    await expect(liveInvoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    })).rejects.toThrow(
      '"existing_page_id":"remote-existing"',
    );
  });

  it("matches backend Unicode lowercase independently of the host locale", async () => {
    const nativeLocaleLowercase = String.prototype.toLocaleLowerCase;
    const localeLowercase = vi
      .spyOn(String.prototype, "toLocaleLowerCase")
      .mockImplementation(function (this: string) {
        return nativeLocaleLowercase.call(this, "tr");
      });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input) =>
        new Response(JSON.stringify(
          String(input).endsWith("/api/spaces")
            ? [{ name: "Preview locale parity" }]
            : { pages: [] },
        ), { status: 200 })
      ),
    );

    try {
      await liveInvoke("create_page", {
        title: "I ΣΧΕΔΙΟ ПРОЕКТ",
        content: "Existing locale-sensitive title.",
        space: "Preview locale parity",
      });
      const draft = await liveInvoke("create_page_draft", {
        clientDraftId: "page_00000000-0000-4000-8000-000000000016",
        title: "i σχεδιο проект",
        content: "Draft body",
        space: "Preview locale parity",
      }) as { id: string; version: number };

      await expect(liveInvoke("publish_page_draft", {
        id: draft.id,
        expectedVersion: draft.version,
      })).rejects.toThrow('"code":"page_title_conflict"');
    } finally {
      localeLowercase.mockRestore();
    }
  });

  it("paginates the merged synthetic and daemon inventory without skipping the boundary row", async () => {
    const remotePages = Array.from({ length: 501 }, (_, index) => ({
      id: `remote-page-${index}`,
      title: `Remote ${index}`,
      content: "",
      space: "Pagination scope",
      domain: "Pagination scope",
      status: "active",
      version: 1,
    }));
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input) => {
        const url = new URL(String(input), "http://preview.test");
        const offset = Number(url.searchParams.get("offset") ?? 0);
        const limit = Number(url.searchParams.get("limit") ?? 500);
        return new Response(
          JSON.stringify({ pages: remotePages.slice(offset, offset + limit) }),
          { status: 200 },
        );
      }),
    );
    await liveInvoke("create_page", {
      title: "Local boundary Page",
      content: "Synthetic body",
      space: "Pagination scope",
    });

    const first = (await liveInvoke("list_pages", {
      status: "active",
      domain: "Pagination scope",
      limit: 500,
      offset: 0,
    })) as Array<{ id: string }>;
    const second = (await liveInvoke("list_pages", {
      status: "active",
      domain: "Pagination scope",
      limit: 500,
      offset: 500,
    })) as Array<{ id: string }>;

    expect(first).toHaveLength(500);
    expect(first[0]?.id).toMatch(/^preview-authored-page-/);
    expect(first.at(-1)?.id).toBe("remote-page-498");
    expect(second.map(({ id }) => id)).toEqual(["remote-page-499", "remote-page-500"]);
  });

  it("models a safe Space rename across local drafts and immutable create replay", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input) =>
        new Response(JSON.stringify(
          String(input).endsWith("/api/spaces")
            ? [
              { id: "space-source", name: "Rename Source", description: "Before" },
              { id: "space-collision", name: "Rename Collision", description: null },
            ]
            : { pages: [] },
        ), { status: 200 })
      ),
    );
    const original = {
      clientDraftId: "page_cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      title: "Rename replay",
      content: "Original",
      space: "Rename Source",
    };
    const created = await liveInvoke("create_page_draft", original) as {
      id: string;
      version: number;
    };
    const updated = await liveInvoke("update_page_draft", {
      id: created.id,
      expectedVersion: created.version,
      title: original.title,
      content: "Updated",
      space: original.space,
    }) as { last_modified: string; version: number };

    await expect(liveInvoke("update_space", {
      name: "Rename Source",
      newName: "Rename Collision",
      description: "Must not apply",
    })).rejects.toThrow('Space "Rename Collision" already exists');
    await expect(liveInvoke("get_page", { id: created.id })).resolves.toEqual(
      expect.objectContaining({
        content: "Updated",
        space: "Rename Source",
        version: updated.version,
      }),
    );

    const renamed = await liveInvoke("update_space", {
      name: "Rename Source",
      newName: "Rename Target",
    });
    const moved = await liveInvoke("get_page", { id: created.id }) as {
      last_modified: string;
      version: number;
    };

    expect(renamed).toMatchObject({ name: "Rename Target", description: null });
    expect(moved).toMatchObject({
      content: "Updated",
      space: "Rename Target",
      domain: "Rename Target",
      version: updated.version + 1,
    });
    expect(moved.last_modified).not.toBe(updated.last_modified);
    await expect(liveInvoke("create_page_draft", original)).resolves.toEqual(moved);
    await expect(liveInvoke("list_spaces")).resolves.toEqual(expect.arrayContaining([
      expect.objectContaining({ name: "Rename Target" }),
      expect.objectContaining({ name: "Rename Collision" }),
    ]));
  });
});

// The app-local DEFAULTS stand in for Rust commands that never return null. A
// stub that returns null (or the wrong keys) where the real command returns a
// struct does not render an empty panel — it white-screens the whole step,
// because consumers dereference the result unguarded. These pin the shapes that
// actually get dereferenced.
describe("liveInvoke app-local defaults match the real command contracts", () => {
  it("get_remote_access_status is a struct with a status kind, never null", async () => {
    // RemoteAccessPanel reads status.status unguarded; its `= { status: "off" }`
    // useQuery default only covers undefined, so null crashes the connect step.
    const result = await liveInvoke("get_remote_access_status");
    expect(result).not.toBeNull();
    expect(result).toHaveProperty("status");
  });

  it("get_on_device_model exposes a models array the card can map over", async () => {
    const result = (await liveInvoke("get_on_device_model")) as { models: unknown[] };
    expect(result).not.toBeNull();
    expect(Array.isArray(result.models)).toBe(true);
  });
});

// The setup wizard's "setting up" step has three long-running rows — daemon,
// model, import — and all three used to resolve to `null` (wire_state and
// add_source were entirely unmapped; download_on_device_model too), which
// either threw (`wire.daemon` on null) or left the row stuck forever. These
// pin the honest shapes so the rows actually resolve.
describe("liveInvoke wire_state (setup wizard daemon row)", () => {
  it("returns a reachable daemon, a resolved mcp_binary with its candidate trail, and 5 routed clients", async () => {
    const wire = (await liveInvoke("wire_state")) as {
      daemon: { reachable: boolean; base_url: string };
      mcp_binary: { command: string; candidates: { path: string; exists: boolean; source: string }[] };
      clients: { client_type: string; route: string; has_plugin: boolean }[];
    };

    expect(wire.daemon.reachable).toBe(true);
    expect(wire.mcp_binary.command.length).toBeGreaterThan(0);
    expect(wire.mcp_binary.candidates.length).toBeGreaterThan(0);

    expect(wire.clients).toHaveLength(5);
    const codex = wire.clients.find((c) => c.client_type === "codex_cli");
    const claudeCode = wire.clients.find((c) => c.client_type === "claude_code");
    // route_for (app/src/wire_state.rs): claude_code/codex_cli always route to
    // "plugin", never "config" — writing a raw MCP entry for either would
    // double-register the server since their plugins declare their own.
    expect(codex?.route).toBe("plugin");
    expect(claudeCode?.route).toBe("plugin");
    // Matches detect_mcp_clients_cmd's fixture: Codex CLI is the one already
    // configured client.
    expect(codex?.has_plugin).toBe(true);
  });
});

describe("liveInvoke setup wizard import row (add_source -> sync_registered_source)", () => {
  it("add_source echoes the requested type/path back as an Active RegisteredSource", async () => {
    const source = (await liveInvoke("add_source", {
      sourceType: "obsidian",
      path: "/Users/preview/Vaults/Work",
    })) as { source_type: string; path: string; status: string; id: string };

    expect(source.source_type).toBe("obsidian");
    expect(source.path).toBe("/Users/preview/Vaults/Work");
    expect(source.status).toBe("Active");
    expect(source.id.length).toBeGreaterThan(0);
  });

  it("sync_registered_source returns SyncStats with a non-zero ingest count", async () => {
    const stats = (await liveInvoke("sync_registered_source", { id: "preview-source-x" })) as {
      files_found: number;
      ingested: number;
      errors: number;
    };

    expect(stats.ingested).toBeGreaterThan(0);
    expect(stats.errors).toBe(0);
  });
});

// The on-device model row: get_on_device_model used to hardcode `cached: true`
// for both models, which meant `status === "running" && !modelEntry?.cached`
// could never be true — the download phase could never render. This proves
// the download actually stays "uncached" with a moving byte count until it
// resolves, then flips.
describe("liveInvoke on-device model download ramp", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("stays uncached with a ramping byte count until the download resolves, then flips to cached/loaded", async () => {
    expect(await liveInvoke("on_device_model_download_bytes")).toBeNull();
    const before = (await liveInvoke("get_on_device_model")) as {
      models: { id: string; cached: boolean }[];
    };
    expect(before.models.find((m) => m.id === "qwen3-4b")?.cached).toBe(true);

    const download = liveInvoke("download_on_device_model", { modelId: "qwen3-4b" });

    await vi.advanceTimersByTimeAsync(10_000);
    const midBytes = (await liveInvoke("on_device_model_download_bytes")) as number;
    expect(midBytes).toBeGreaterThan(900e6);
    expect(midBytes).toBeLessThan(1_100e6);

    const mid = (await liveInvoke("get_on_device_model")) as {
      loaded: string | null;
      models: { id: string; cached: boolean }[];
    };
    expect(mid.loaded).toBeNull();
    expect(mid.models.find((m) => m.id === "qwen3-4b")?.cached).toBe(false);

    await vi.advanceTimersByTimeAsync(20_000);
    await download;

    // The real Qwen3-4B blob is 2,497,281,120 bytes; the ramp caps there.
    const finalBytes = (await liveInvoke("on_device_model_download_bytes")) as number;
    expect(finalBytes).toBe(2_497_281_120);

    const after = (await liveInvoke("get_on_device_model")) as {
      loaded: string | null;
      models: { id: string; cached: boolean }[];
    };
    expect(after.loaded).toBe("qwen3-4b");
    expect(after.models.find((m) => m.id === "qwen3-4b")?.cached).toBe(true);
  });
});

// DEBT: 79 commands with no harness stub as of 2026-07-13, predating this fix
// and out of scope for it — spaces CRUD, entity/observation CRUD, snapshots,
// agent management, obsidian export/import, avatar, remote-access token
// rotation, native file/dialog commands, and a handful of others. None are
// reachable from a screen this fix touches, and liveInvoke's fallback (warn +
// return null) means they fail soft today rather than crash.
//
// This is a two-way ratchet, not a permanent excuse list: the coverage test
// below fails on any command outside this set with no stub (the bug class —
// ship an invoke() with nothing behind it), AND on any entry here that has
// since been stubbed or deleted from tauri.ts (so backfilling a gap, or
// removing a dead command, means deleting its name from here — the count can
// only go down). If this number ever goes up, that's a regression to justify
// in the PR, not a rebase footgun to silently resolve.
const UNSTUBBED_DEBT = new Set([
  "acknowledge_onboarding_milestone",
  "add_observation_cmd",
  "add_space",
  "add_watch_path",
  "confirm_entity_cmd",
  "confirm_observation_cmd",
  "confirm_space",
  "connect_source",
  "correct_memory_cmd",
  "create_entity_cmd",
  "create_space",
  "daemon_version",
  "delete_agent",
  "delete_bulk",
  "delete_by_time_range",
  "delete_entity_cmd",
  "delete_file_chunks",
  "delete_observation_cmd",
  "delete_snapshot",
  "delete_space",
  "delete_tag",
  "detect_obsidian_vaults",
  "disconnect_source",
  "dismiss_contradiction",
  "export_page_to_obsidian",
  "export_pages_to_obsidian",
  "get_agent",
  "get_chunks",
  "get_enrichment_status",
  "get_external_llm_key_configured",
  "get_pending_revision",
  "get_snapshot_captures",
  "get_snapshot_captures_with_content",
  "get_space",
  "import_chat_export",
  "import_memories_cmd",
  "ingest_webpage",
  "install_client_plugin",
  "list_external_models",
  "move_space",
  "open_file",
  "pin_space",
  "quick_capture",
  "read_source_dir",
  "read_text_file",
  "rebuild_activities",
  "reclassify_memory_cmd",
  "regenerate_narrative",
  "reindex",
  "remove_avatar",
  "remove_legacy_mcp_entry",
  "remove_raw_mcp_entry",
  "remove_source",
  "remove_space",
  "remove_watch_path",
  "rename_space",
  "reorder_space",
  "reset_onboarding_milestones",
  "rotate_remote_token",
  "save_temp_file",
  "set_api_key",
  "set_avatar",
  "set_document_space",
  "set_document_tags",
  "set_external_llm",
  "set_model_choice",
  "set_run_at_login",
  "set_stability_cmd",
  "sync_source",
  "test_external_llm",
  "test_remote_mcp_connection",
  "toggle_remote_access",
  "toggle_space_starred",
  "update_agent",
  "update_chunk",
  "update_observation_cmd",
  "update_profile",
  "upload_source_file",
  "write_mcp_config",
]);

// This is the test that would have caught all three commands the setup
// wizard's "setting up" step silently rendered against null (wire_state,
// download_on_device_model, add_source) — and fails the moment a future
// invoke() ships with no harness stub at all. One-directional on purpose: an
// unused extra HANDLERS/DEFAULTS entry is fine, a missing one is the bug.
describe("tauri.ts -> live-invoke.ts command coverage (parity)", () => {
  const source = readFileSync(TAURI_TS_PATH, "utf8");
  const commands = invokedCommands(source);
  const covered = new Set([...Object.keys(HANDLERS), ...Object.keys(DEFAULTS)]);

  it("extracted a realistic number of commands (extraction regex sanity floor)", () => {
    // If tauri.ts's invoke() calls ever stopped matching the regex above (a
    // refactor to a different call style, say), `commands` would silently
    // shrink toward empty and both assertions below would vacuously pass.
    expect(commands.size).toBeGreaterThan(150);
  });

  it("every invoke() command has a live-invoke entry, or is tracked as debt", () => {
    const uncovered = [...commands].filter((c) => !covered.has(c) && !UNSTUBBED_DEBT.has(c)).sort();
    expect(uncovered).toEqual([]);
  });

  // The other half of the ratchet: an entry that's been stubbed (covered) or
  // whose command no longer exists (deleted from tauri.ts) must be removed
  // from UNSTUBBED_DEBT — otherwise the list only ever grows and "debt" stops
  // meaning anything.
  it("UNSTUBBED_DEBT has no stale entries (already covered, or no longer invoked)", () => {
    const stale = [...UNSTUBBED_DEBT].filter((c) => covered.has(c) || !commands.has(c)).sort();
    expect(stale).toEqual([]);
  });

  it("the commands this fix targeted are real entries, not tracked as debt", () => {
    for (const cmd of [
      "wire_state",
      "download_on_device_model",
      "get_on_device_model",
      "on_device_model_download_bytes",
      "add_source",
      "sync_registered_source",
    ]) {
      expect(covered.has(cmd)).toBe(true);
      expect(UNSTUBBED_DEBT.has(cmd)).toBe(false);
    }
  });
});
