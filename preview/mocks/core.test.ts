// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PAGES, PRISTINE, REVIEW_STATE } from "../fixtures";
import { invoke } from "./core";
import { liveInvoke } from "./live-invoke";

vi.mock("./live-invoke", () => ({
  liveInvoke: vi.fn(async () => ({ deleted: true })),
}));

// The wizard's runtime row used to store a REAL memory in the maintainer's own
// daemon (store_memory fell through to liveInvoke, which proxied it) and then
// delete it. When this fixture owned every `delete_memory`, that delete was
// swallowed and each preview of the wizard left a real "Wenlan setup check"
// memory behind in the developer's database. store_memory is now synthesized
// in liveInvoke itself (PREVIEW_PROBES — see live-invoke.ts), so it no longer
// touches the daemon at all; this still covers the other half of the fix —
// the fixture may only delete what the fixture itself owns, and every other
// id (a real memory, or the wizard's synthesized probe id) still needs to
// reach delete_memory, which is what actually clears it.
describe("preview core.ts — delete_memory in fixture mode", () => {
  beforeEach(() => {
    vi.mocked(liveInvoke).mockClear();
    (window as { __PREVIEW_FIXTURES__?: boolean }).__PREVIEW_FIXTURES__ = true;
    REVIEW_STATE.captures = [
      { id: "cap_1" },
      { id: "cap_2" },
    ] as unknown as typeof REVIEW_STATE.captures;
  });

  it("deletes a review-queue capture from the fixture, without touching the daemon", async () => {
    await invoke("delete_memory", { sourceId: "cap_1" });

    expect(REVIEW_STATE.captures.map((c) => c.id)).toEqual(["cap_2"]);
    expect(liveInvoke).not.toHaveBeenCalled();
  });

  it("delegates a non-fixture id (the wizard's real probe memory) to the live daemon", async () => {
    const result = await invoke("delete_memory", { sourceId: "mem_real_probe" });

    expect(liveInvoke).toHaveBeenCalledWith("delete_memory", {
      sourceId: "mem_real_probe",
    });
    expect(result).toEqual({ deleted: true });
    expect(REVIEW_STATE.captures.map((c) => c.id)).toEqual(["cap_1", "cap_2"]);
  });
});

describe("preview core.ts — delete_page in fixture mode", () => {
  beforeEach(() => {
    (window as { __PREVIEW_FIXTURES__?: boolean }).__PREVIEW_FIXTURES__ = true;
    PAGES["page-cited"] = structuredClone(PRISTINE["page-cited"]);
  });

  it("removes the fixture Page instead of acknowledging a no-op", async () => {
    await expect(invoke("delete_page", { id: "page-cited" })).resolves.toBeNull();
    await expect(invoke("get_page", { id: "page-cited" })).resolves.toBeNull();
  });
});
