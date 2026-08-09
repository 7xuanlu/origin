// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from "@playwright/test";
import { installTauriMock } from "./tauriMock";

test("models detail reads and mutations while rejecting unknown commands", async ({ page }) => {
  // Given a fresh deterministic Tauri fixture runtime.
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  // When detail commands, a mutation, and an intentionally unknown command run.
  const results = await page.evaluate(async () => {
    const invoke = window.__TAURI_INTERNALS__?.invoke;
    if (!invoke) throw new Error("Tauri invoke is unavailable");

    const space = await invoke("get_space", { name: "Wenlan" });
    const entity = await invoke("get_entity_detail_cmd", { entityId: "entity-ada" });
    await invoke("create_space", { name: "Created in E2E", description: "Stateful" });
    const spaces = await invoke("list_spaces");

    let unknownError = "";
    try {
      await invoke("todo7_unknown_command");
    } catch (error) {
      unknownError = error instanceof Error ? error.message : String(error);
    }

    let argumentError = "";
    try {
      await invoke("create_space", { name: 42 });
    } catch (error) {
      argumentError = error instanceof Error ? error.message : String(error);
    }

    return { space, entity, spaces, unknownError, argumentError };
  });

  // Then typed details exist, the mutation is visible, and the unknown name is surfaced.
  expect(results.space).toMatchObject({ id: "space-wenlan", name: "Wenlan" });
  expect(results.entity).toMatchObject({ entity: { id: "entity-ada", name: "Ada Lovelace" } });
  expect(results.spaces).toEqual(expect.arrayContaining([
    expect.objectContaining({ name: "Created in E2E", description: "Stateful" }),
  ]));
  expect(results.unknownError).toContain("todo7_unknown_command");
  expect(results.argumentError).toContain("create_space");
  expect(results.argumentError).toContain("name");
});

test("models Page draft identity, replay, Space validation, and rename versioning", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  const results = await page.evaluate(async () => {
    const invoke = window.__TAURI_INTERNALS__?.invoke;
    if (!invoke) throw new Error("Tauri invoke is unavailable");
    const errorMessage = async (command: string, args: unknown) => {
      try {
        await invoke(command, args);
        return "";
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
    };

    const invalidIdError = await errorMessage("create_page_draft", {
      clientDraftId: "caller-controlled-primary-key",
      title: "Invalid",
      content: "Body",
      space: null,
    });
    const unregisteredSpaceError = await errorMessage("create_page_draft", {
      clientDraftId: "page_66666666-6666-4666-8666-666666666661",
      title: "Missing scope",
      content: "Body",
      space: "missing",
    });
    const whitespace = await invoke("create_page_draft", {
      clientDraftId: "page_66666666-6666-4666-8666-666666666662",
      title: "Whitespace scope",
      content: "Body",
      space: " \n ",
    });
    const original = {
      clientDraftId: "page_66666666-6666-4666-8666-666666666663",
      title: "Replay identity",
      content: "Exact body  \n",
      space: "Wenlan",
    };
    const created = await invoke("create_page_draft", original);
    const replayedBeforeRename = await invoke("create_page_draft", original);
    const conflictingSpaceReplayError = await errorMessage("create_page_draft", {
      ...original,
      space: "Research",
    });
    const divergentReplayError = await errorMessage("create_page_draft", {
      ...original,
      content: "Different body",
      space: "missing",
    });
    const staleVersionPrecedenceError = await errorMessage("update_page_draft", {
      id: original.clientDraftId,
      expectedVersion: 0,
      title: "Stale",
      content: "Stale body",
      space: "missing",
    });
    const unregisteredUpdateSpaceError = await errorMessage("update_page_draft", {
      id: original.clientDraftId,
      expectedVersion: 1,
      title: "Current",
      content: "Current body",
      space: "missing",
    });
    const updateReplayDraft = await invoke("create_page_draft", {
      clientDraftId: "page_66666666-6666-4666-8666-666666666664",
      title: "Update replay",
      content: "Original",
      space: null,
    }) as { id: string; version: number };
    const updateCommitted = await invoke("update_page_draft", {
      id: updateReplayDraft.id,
      expectedVersion: updateReplayDraft.version,
      title: "Update replay",
      content: "Committed",
      space: null,
    });
    const updateReplayed = await invoke("update_page_draft", {
      id: updateReplayDraft.id,
      expectedVersion: updateReplayDraft.version,
      title: "Update replay",
      content: "Committed",
      space: null,
    });
    const divergentUpdateReplayError = await errorMessage("update_page_draft", {
      id: updateReplayDraft.id,
      expectedVersion: updateReplayDraft.version,
      title: "Update replay",
      content: "Different",
      space: null,
    });
    const discarded = await invoke("create_page_draft", {
      clientDraftId: "page_66666666-6666-4666-8666-666666666665",
      title: "Discarded",
      content: "Sensitive draft body",
      space: null,
    }) as { id: string; version: number };
    await invoke("discard_page_draft", {
      id: discarded.id,
      expectedVersion: discarded.version,
    });
    const discardedRetryError = await errorMessage("create_page_draft", {
      clientDraftId: discarded.id,
      title: "Discarded",
      content: "Sensitive draft body",
      space: null,
    });

    const activeBeforeRename = await invoke("get_page", { id: "page-architecture" });
    await invoke("accept_refinement", { id: "refinement-page-history-cleanup" });
    const archivedBeforeRename = await invoke("get_page", { id: "page-history" });
    const renameClockDraft = await invoke("create_page_draft", {
      clientDraftId: "page_66666666-6666-4666-8666-666666666666",
      title: "Rename clock",
      content: "Before update",
      space: "Wenlan",
    }) as { id: string; version: number };
    const updatedBeforeRename = await invoke("update_page_draft", {
      id: renameClockDraft.id,
      expectedVersion: renameClockDraft.version,
      title: "Rename clock",
      content: "After update",
      space: "Wenlan",
    });
    const renameCollisionError = await errorMessage("update_space", {
      name: "Wenlan",
      newName: "Research",
      description: "Must not apply",
    });
    const draftAfterRenameCollision = await invoke("get_page", { id: renameClockDraft.id });
    const spacesAfterRenameCollision = await invoke("list_spaces", {});

    await invoke("update_space", {
      name: "Wenlan",
      newName: "Wenlan Core",
      description: "Renamed",
    });
    const moved = await invoke("get_page", { id: original.clientDraftId });
    const updatedAfterRename = await invoke("get_page", { id: renameClockDraft.id });
    const activeAfterRename = await invoke("get_page", { id: "page-architecture" });
    const archivedAfterRename = await invoke("get_page", { id: "page-history" });
    const replayedAfterRename = await invoke("create_page_draft", original);
    const staleAfterRenameError = await errorMessage("update_page_draft", {
      id: original.clientDraftId,
      expectedVersion: 1,
      title: original.title,
      content: original.content,
      space: "Wenlan",
    });
    await invoke("update_space", {
      name: "Wenlan Core",
      newName: "Wenlan Foundation",
      description: null,
    });
    const updatedAfterSecondRename = await invoke("get_page", { id: renameClockDraft.id });

    return {
      invalidIdError,
      unregisteredSpaceError,
      whitespace,
      created,
      replayedBeforeRename,
      conflictingSpaceReplayError,
      divergentReplayError,
      staleVersionPrecedenceError,
      unregisteredUpdateSpaceError,
      updateCommitted,
      updateReplayed,
      divergentUpdateReplayError,
      discardedRetryError,
      activeBeforeRename,
      activeAfterRename,
      archivedBeforeRename,
      archivedAfterRename,
      updatedBeforeRename,
      updatedAfterRename,
      updatedAfterSecondRename,
      renameCollisionError,
      draftAfterRenameCollision,
      spacesAfterRenameCollision,
      moved,
      replayedAfterRename,
      staleAfterRenameError,
    };
  });

  expect(results.invalidIdError).toContain("Page draft id must use the page_<uuid-v4> format");
  expect(results.unregisteredSpaceError).toContain('Space "missing" is not registered');
  expect(results.whitespace).toMatchObject({ space: null, domain: null });
  expect(results.replayedBeforeRename).toEqual(results.created);
  expect(results.conflictingSpaceReplayError).toContain('"code":"page_draft_id_conflict"');
  expect(results.divergentReplayError).toContain('"code":"page_draft_id_conflict"');
  expect(results.staleVersionPrecedenceError).toContain('"code":"draft_version_conflict"');
  expect(results.unregisteredUpdateSpaceError).toContain('Space "missing" is not registered');
  expect(results.updateReplayed).toEqual(results.updateCommitted);
  expect(results.divergentUpdateReplayError).toContain('"code":"draft_version_conflict"');
  expect(results.discardedRetryError).toContain('"code":"page_draft_id_conflict"');
  expect(results.renameCollisionError).toContain('Space "Research" already exists');
  expect(results.draftAfterRenameCollision).toEqual(results.updatedBeforeRename);
  expect(results.spacesAfterRenameCollision.filter(
    (space: { name: string }) => space.name === "Research",
  )).toHaveLength(1);
  for (const [before, after] of [
    [results.activeBeforeRename, results.activeAfterRename],
    [results.archivedBeforeRename, results.archivedAfterRename],
  ]) {
    expect(after).toMatchObject({
      space: "Wenlan Core",
      domain: "Wenlan Core",
      version: before.version,
      last_modified: before.last_modified,
    });
  }
  expect(results.updatedAfterRename).toMatchObject({
    space: "Wenlan Core",
    domain: "Wenlan Core",
    version: results.updatedBeforeRename.version + 1,
  });
  expect(results.updatedAfterRename.last_modified)
    .not.toBe(results.updatedBeforeRename.last_modified);
  expect(results.updatedAfterSecondRename).toMatchObject({
    space: "Wenlan Foundation",
    domain: "Wenlan Foundation",
    version: results.updatedAfterRename.version + 1,
  });
  expect(results.updatedAfterSecondRename.last_modified)
    .not.toBe(results.updatedAfterRename.last_modified);
  expect(results.moved).toMatchObject({ space: "Wenlan Core", domain: "Wenlan Core", version: 2 });
  expect(results.replayedAfterRename).toEqual(results.moved);
  expect(results.staleAfterRenameError).toContain('"code":"draft_version_conflict"');
  expect(results.staleAfterRenameError).toContain('"current_version":2');
});

test("replays only the immediately committed active Page after an ambiguous publish", async ({ page }) => {
  await installTauriMock(page, { locale: "en", rawActions: [] });
  await page.goto("/");

  const results = await page.evaluate(async () => {
    const invoke = window.__TAURI_INTERNALS__?.invoke;
    if (!invoke) throw new Error("Tauri invoke is unavailable");
    const commandError = async (command: string, args: unknown) => {
      try {
        await invoke(command, args);
        return "";
      } catch (error) {
        return error instanceof Error ? error.message : String(error);
      }
    };
    const publishError = (expectedVersion: number) => commandError("publish_page_draft", {
      id: "page_88888888-8888-4888-8888-888888888888",
      expectedVersion,
    });

    const start = "<!-- origin:sources:start -->";
    const end = "<!-- origin:sources:end -->";
    const rawBody = `Body  \n\n${start}\n## Sources\n- [[mem_1]]\n${end}\n\nTail  \n\n`;
    const sourcesOnly = `${start}\n## Sources\n- [[mem_1]]\n${end}`;
    const sourcesOnlyCreateError = await commandError("create_page_draft", {
      clientDraftId: "page_88888888-8888-4888-8888-888888888886",
      title: " ",
      content: sourcesOnly,
      space: null,
    });
    const updateTarget = await invoke("create_page_draft", {
      clientDraftId: "page_88888888-8888-4888-8888-888888888887",
      title: "Temporary title",
      content: "",
      space: null,
    }) as { id: string; version: number };
    const sourcesOnlyUpdateError = await commandError("update_page_draft", {
      id: updateTarget.id,
      expectedVersion: updateTarget.version,
      title: " ",
      content: sourcesOnly,
      space: null,
    });
    const draft = await invoke("create_page_draft", {
      clientDraftId: "page_88888888-8888-4888-8888-888888888888",
      title: "Publish retry",
      content: rawBody,
      space: null,
    }) as { id: string; version: number };
    const rawDraft = await invoke("get_page", { id: draft.id });
    const published = await invoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });
    const storedAfterPublish = await invoke("get_page", { id: draft.id });
    const replayed = await invoke("publish_page_draft", {
      id: draft.id,
      expectedVersion: draft.version,
    });
    const olderError = await publishError(draft.version - 1);
    const currentActiveError = await publishError(draft.version + 1);
    const newerError = await publishError(draft.version + 2);

    await invoke("update_page", {
      id: draft.id,
      content: "Later edit",
    });
    const mutated = await invoke("get_page", { id: draft.id });
    const mutatedReplayError = await publishError(draft.version);

    return {
      draft,
      rawDraft,
      rawBody,
      sourcesOnlyCreateError,
      sourcesOnlyUpdateError,
      published,
      storedAfterPublish,
      replayed,
      olderError,
      currentActiveError,
      newerError,
      mutated,
      mutatedReplayError,
    };
  });

  expect(results.replayed).toEqual(results.published);
  expect(results.rawDraft).toMatchObject({ content: results.rawBody, status: "draft" });
  expect(results.sourcesOnlyCreateError).toContain('"code":"invalid_page_draft"');
  expect(results.sourcesOnlyUpdateError).toContain('"code":"invalid_page_draft"');
  expect(results.replayed).toMatchObject({
    status: "active",
    version: results.draft.version + 1,
    content: "Body\n\nTail",
  });
  expect(results.storedAfterPublish).toMatchObject({ content: "Body\n\nTail" });
  for (const error of [results.olderError, results.newerError]) {
    expect(error).toContain('"code":"draft_version_conflict"');
    expect(error).toContain(`"current_version":${results.draft.version + 1}`);
  }
  expect(results.currentActiveError).toContain("is not a draft");
  expect(results.mutated).toMatchObject({
    content: "Later edit",
    version: results.draft.version + 2,
  });
  expect(results.mutatedReplayError).toContain('"code":"draft_version_conflict"');
  expect(results.mutatedReplayError).toContain(
    `"current_version":${results.draft.version + 2}`,
  );
});
