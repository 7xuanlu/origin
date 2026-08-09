// SPDX-License-Identifier: AGPL-3.0-only
import type {
  DistillReviewResponse,
  EntityDetail,
  MemoryItem,
  Page as KnowledgePage,
  RefinementProposalSummary,
  Space,
} from "../../src/lib/tauri";
import type { SpacesNavigationFixture } from "../fixtures/spacesNavigation";
import { baseResponse } from "./baseResponses";
import { ConfiguredTauriFailureError, TauriMockArgumentError } from "./errors";
import type { MockCommandCall, MockFailure } from "./types";

const CLIENT_PAGE_DRAFT_ID =
  /^page_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SOURCES_BLOCK_START = "<!-- origin:sources:start -->";
const SOURCES_BLOCK_END = "<!-- origin:sources:end -->";

type PageDraftCreateRequest = {
  readonly title: string;
  readonly content: string;
  readonly space: string | null;
};

function canonicalizePageBody(body: string): string {
  const start = body.indexOf(SOURCES_BLOCK_START);
  if (start < 0) return body.trimEnd();
  const afterStart = start + SOURCES_BLOCK_START.length;
  const relativeEnd = body.slice(afterStart).indexOf(SOURCES_BLOCK_END);
  if (relativeEnd < 0) return body.trimEnd();
  const end = afterStart + relativeEnd + SOURCES_BLOCK_END.length;
  const head = body.slice(0, start).trimEnd();
  const tail = body.slice(end).trimStart();
  return (tail ? `${head}\n\n${tail}` : head).trimEnd();
}

function sanitizeIngressContent(content: string): string {
  return canonicalizePageBody(content)
    .replaceAll(SOURCES_BLOCK_START, "")
    .replaceAll(SOURCES_BLOCK_END, "")
    .trimEnd();
}

function nextFixtureTimestamp(previous: string | null | undefined): string {
  const floor = Date.parse("2026-07-10T12:34:00Z");
  const previousMs = typeof previous === "string" ? Date.parse(previous) : Number.NaN;
  return new Date(Math.max(
    floor,
    Number.isFinite(previousMs) ? previousMs + 1 : 0,
  )).toISOString();
}

function optionalValue(args: unknown, key: string): unknown {
  return typeof args === "object" && args !== null ? Reflect.get(args, key) : undefined;
}

function requiredString(command: string, args: unknown, key: string): string {
  const value = optionalValue(args, key);
  if (typeof value !== "string" || value.trim() === "") throw new TauriMockArgumentError(command, key);
  return value;
}

function stringValue(command: string, args: unknown, key: string): string {
  const value = optionalValue(args, key);
  if (typeof value !== "string") throw new TauriMockArgumentError(command, key);
  return value;
}

function optionalString(args: unknown, key: string): string | null {
  const value = optionalValue(args, key);
  return typeof value === "string" ? value : null;
}

function requiredNumber(command: string, args: unknown, key: string): number {
  const value = optionalValue(args, key);
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TauriMockArgumentError(command, key);
  return value;
}

function requiredBoolean(command: string, args: unknown, key: string): boolean {
  const value = optionalValue(args, key);
  if (typeof value !== "boolean") throw new TauriMockArgumentError(command, key);
  return value;
}

function stringArray(args: unknown, key: string): readonly string[] {
  const value = optionalValue(args, key);
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

export class TauriMockRuntime {
  private spaces: Space[];
  private pages: KnowledgePage[];
  private entityDetails: EntityDetail[];
  private memories: MemoryItem[];
  private readonly distillReview: DistillReviewResponse;
  private refinements: RefinementProposalSummary[];
  private readonly callsLog: MockCommandCall[] = [];
  private readonly failures = new Map<string, string[]>();
  private readonly activityRows: readonly Record<string, unknown>[];
  private readonly pageDraftCreateRequests = new Map<string, PageDraftCreateRequest | null>();
  private pageSequence: number;

  constructor(
    fixture: SpacesNavigationFixture,
    failures: readonly MockFailure[] = [],
    rawActions: readonly string[] = [],
  ) {
    this.spaces = fixture.spaces.map((space) => ({ ...space }));
    this.pages = fixture.pages.map((page) => ({ ...page }));
    this.pageSequence = this.pages.length;
    this.entityDetails = fixture.entityDetails.map((detail) => ({
      entity: { ...detail.entity },
      observations: detail.observations.map((observation) => ({ ...observation })),
      relations: detail.relations.map((relation) => ({ ...relation })),
    }));
    this.memories = fixture.memories.map((memory) => ({ ...memory }));
    this.distillReview = structuredClone(fixture.distillReview);
    this.refinements = structuredClone([...fixture.refinements]);
    this.activityRows = rawActions.map((action, index) => ({
      id: index + 1,
      timestamp: 1_783_728_000 - index * 60,
      agent_name: "claude-code",
      action,
      memory_ids: null,
      query: null,
      detail: null,
      memory_titles: [],
    }));
    for (const failure of failures) this.failNext(failure.command, failure.message, failure.times ?? 1);
  }

  calls(): readonly MockCommandCall[] {
    return this.callsLog.map((call) => ({ ...call }));
  }

  failNext(command: string, message: string, times = 1): void {
    const queued = this.failures.get(command) ?? [];
    this.failures.set(command, [...queued, ...Array.from({ length: times }, () => message)]);
  }

  async invoke(command: string, args?: unknown): Promise<unknown> {
    this.callsLog.push({ command, args });
    const configured = this.failures.get(command)?.shift();
    if (configured) throw new ConfiguredTauriFailureError(command, configured);
    if (command.startsWith("plugin:")) return null;

    switch (command) {
      case "list_spaces": return this.spaces.map((space) => ({ ...space }));
      case "get_space": return this.spaces.find((space) => space.name === requiredString(command, args, "name")) ?? null;
      case "create_space": return this.createSpace(args);
      case "update_space": return this.updateSpace(args);
      case "delete_space": return this.deleteSpace(args);
      case "confirm_space": return this.confirmSpace(args);
      case "toggle_space_starred": return this.toggleSpace(args);
      case "reorder_space": return this.reorderSpace(args);
      case "list_pages": return this.listPages(args);
      case "get_page": return this.pages.find((page) => page.id === requiredString(command, args, "id")) ?? null;
      case "create_page": return this.createPage(args);
      case "update_page": return this.updatePage(args);
      case "delete_page": return this.deletePage(args);
      case "redistill_page": return this.redistillPage(args);
      case "create_page_draft": return this.createPageDraft(args);
      case "update_page_draft": return this.updatePageDraft(args);
      case "publish_page_draft": return this.publishPageDraft(args);
      case "discard_page_draft": return this.discardPageDraft(args);
      case "distill_review": return structuredClone(this.distillReview);
      case "list_refinements": return { proposals: structuredClone(this.refinements) };
      case "accept_refinement": return this.resolveRefinement(args, true);
      case "reject_refinement": return this.resolveRefinement(args, false);
      case "list_entities_cmd": return this.listEntities(args);
      case "get_entity_detail_cmd": return this.getEntityDetail(args);
      case "add_observation_cmd": return this.addObservation(args);
      case "update_observation_cmd": return this.updateObservation(args);
      case "delete_observation_cmd": return this.deleteObservation(args);
      case "confirm_observation_cmd": return this.confirmObservation(args);
      case "confirm_entity_cmd": return this.confirmEntity(args);
      case "delete_entity_cmd": return this.deleteEntity(args);
      case "list_memories_cmd": return this.listMemories(args);
      case "get_memory_detail": return this.memories.find((memory) => memory.source_id === requiredString(command, args, "sourceId")) ?? null;
      case "list_memories_by_ids": {
        const ids = new Set(stringArray(args, "ids"));
        return this.memories.filter((memory) => ids.has(memory.source_id));
      }
      case "list_indexed_files": return this.listIndexedFiles();
      case "update_memory_cmd": return this.updateMemory(args);
      case "reclassify_memory_cmd": return this.reclassifyMemory(args);
      case "set_stability_cmd": return this.setStability(args);
      case "delete_file_chunks": return this.deleteFileChunks(args);
      case "pin_memory": return this.setMemoryPinned(args, true);
      case "unpin_memory": return this.setMemoryPinned(args, false);
      case "confirm_memory": return this.confirmMemory(args);
      case "delete_memory": return this.deleteMemory(args);
      case "acknowledge_onboarding_milestone":
        requiredString(command, args, "id");
        return null;
      case "search": return this.search(args);
      default: return baseResponse(command, args, { activityRows: this.activityRows, memoryCount: this.memories.length });
    }
  }

  private createSpace(args: unknown): Space {
    const name = requiredString("create_space", args, "name");
    const space: Space = {
      id: `space-${name.toLowerCase().replaceAll(/[^a-z0-9]+/g, "-")}`,
      name, description: optionalString(args, "description"), suggested: false, starred: false,
      sort_order: this.spaces.length, memory_count: 0, entity_count: 0,
      created_at: 1_783_728_000, updated_at: 1_783_728_000,
    };
    this.spaces.push(space);
    return { ...space };
  }

  private updateSpace(args: unknown): Space {
    const name = requiredString("update_space", args, "name");
    const newName = requiredString("update_space", args, "newName");
    const index = this.spaces.findIndex((space) => space.name === name);
    if (index < 0) throw new TauriMockArgumentError("update_space", "name");
    const current = this.spaces[index];
    if (!current) throw new TauriMockArgumentError("update_space", "name");
    if (
      name !== newName
      && this.spaces.some((space, candidateIndex) =>
        candidateIndex !== index && space.name === newName
      )
    ) {
      throw new Error(`Space ${JSON.stringify(newName)} already exists`);
    }
    const updated = { ...current, name: newName, description: optionalString(args, "description") };
    if (name !== newName) {
      this.pages = this.pages.map((page) => {
        if (page.space !== name && page.domain !== name) return page;
        return {
          ...page,
          space: page.space === name ? newName : page.space,
          domain: page.domain === name ? newName : page.domain,
          version: page.status === "draft" ? page.version + 1 : page.version,
          last_modified: page.status === "draft"
            ? nextFixtureTimestamp(page.last_modified)
            : page.last_modified,
        };
      });
    }
    this.spaces[index] = updated;
    return { ...updated };
  }

  private deleteSpace(args: unknown): null {
    const name = requiredString("delete_space", args, "name");
    this.spaces = this.spaces.filter((space) => space.name !== name);
    return null;
  }

  private confirmSpace(args: unknown): null {
    const name = requiredString("confirm_space", args, "name");
    this.spaces = this.spaces.map((space) => space.name === name ? { ...space, suggested: false } : space);
    return null;
  }

  private toggleSpace(args: unknown): boolean {
    const name = requiredString("toggle_space_starred", args, "name");
    let starred = false;
    this.spaces = this.spaces.map((space) => {
      if (space.name !== name) return space;
      starred = !space.starred;
      return { ...space, starred };
    });
    return starred;
  }

  private reorderSpace(args: unknown): null {
    const name = requiredString("reorder_space", args, "name");
    const newOrder = requiredNumber("reorder_space", args, "newOrder");
    this.spaces = this.spaces.map((space) => space.name === name ? { ...space, sort_order: newOrder } : space);
    return null;
  }

  private listPages(args: unknown): readonly KnowledgePage[] {
    const status = optionalString(args, "status");
    const domain = optionalString(args, "domain");
    const limit = optionalValue(args, "limit");
    const offset = optionalValue(args, "offset");
    const filtered = this.pages.filter((page) => (!status || page.status === status) && (!domain || page.domain === domain || page.space === domain));
    return filtered.slice(typeof offset === "number" ? offset : 0, typeof limit === "number" ? (typeof offset === "number" ? offset : 0) + limit : undefined);
  }

  private createPage(args: unknown): { id: string; attached_to: string | null; warnings: string[] } {
    const title = requiredString("create_page", args, "title").trim();
    const content = requiredString("create_page", args, "content").trim();
    const space = optionalString(args, "space")?.trim() || null;
    if (space && !this.spaces.some((candidate) => candidate.name === space)) {
      throw new TauriMockArgumentError("create_page", "space");
    }
    const id = `page-authored-${++this.pageSequence}`;
    const now = "2026-07-10T12:30:00Z";
    this.pages = [{
      id,
      title,
      summary: null,
      content,
      entity_id: null,
      domain: space,
      space,
      source_memory_ids: [],
      version: 1,
      status: "active",
      creation_kind: "authored",
      review_status: "unconfirmed",
      created_at: now,
      last_compiled: now,
      last_modified: now,
    }, ...this.pages];
    return { id, attached_to: null, warnings: [] };
  }

  private deletePage(args: unknown): null {
    const id = requiredString("delete_page", args, "id");
    this.pages = this.pages.filter((page) => page.id !== id);
    return null;
  }

  private updatePage(args: unknown): null {
    const id = requiredString("update_page", args, "id");
    const content = requiredString("update_page", args, "content");
    const index = this.pages.findIndex((page) => page.id === id);
    const page = this.pages[index];
    if (index < 0 || !page) throw new TauriMockArgumentError("update_page", "id");
    this.pages[index] = {
      ...page,
      content,
      citations: [],
      user_edited: true,
      version: page.version + 1,
      last_modified: "2026-07-10T12:32:00Z",
    };
    return null;
  }

  private redistillPage(args: unknown): { status: "ok"; updated: true } {
    const id = requiredString("redistill_page", args, "pageId");
    const index = this.pages.findIndex((page) => page.id === id);
    const page = this.pages[index];
    if (index < 0 || !page) throw new TauriMockArgumentError("redistill_page", "pageId");
    this.pages[index] = {
      ...page,
      last_compiled: "2026-07-10T12:33:00Z",
      last_modified: "2026-07-10T12:33:00Z",
      stale_reason: null,
    };
    return { status: "ok", updated: true };
  }

  private resolveRefinement(
    args: unknown,
    approve: boolean,
  ): { id: string; action_applied: string } | { id: string } {
    const command = approve ? "accept_refinement" : "reject_refinement";
    const id = requiredString(command, args, "id");
    const index = this.refinements.findIndex((proposal) => proposal.id === id);
    const proposal = this.refinements[index];
    if (index < 0 || !proposal) throw new TauriMockArgumentError(command, "id");
    this.refinements.splice(index, 1);
    if (approve && proposal.payload?.action === "page_keep_or_archive") {
      const pageId = proposal.payload.page_id;
      this.pages = this.pages.map((page) =>
        page.id === pageId ? { ...page, status: "archived" } : page
      );
    }
    return approve ? { id, action_applied: proposal.action } : { id };
  }

  private createPageDraft(args: unknown): KnowledgePage {
    const id = requiredString("create_page_draft", args, "clientDraftId");
    if (!CLIENT_PAGE_DRAFT_ID.test(id)) {
      throw new Error("Page draft id must use the page_<uuid-v4> format");
    }
    const title = stringValue("create_page_draft", args, "title");
    const content = stringValue("create_page_draft", args, "content");
    if (!title.trim() && !sanitizeIngressContent(content).trim()) {
      throw new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "A Page draft needs a title or content",
      }));
    }
    const requestedSpace = this.normalizedDraftSpace(args);
    const existing = this.pages.find((page) => page.id === id);
    if (existing) {
      const request = this.pageDraftCreateRequests.get(id);
      if (
        existing.status === "draft"
        && request !== null
        && request !== undefined
        && request.title === title
        && request.content === content
        && request.space === requestedSpace
      ) {
        return { ...existing };
      }
      throw new Error(JSON.stringify({
        code: "page_draft_id_conflict",
        error: "Page draft id already belongs to another Page",
      }));
    }
    if (this.pageDraftCreateRequests.has(id)) {
      throw new Error(JSON.stringify({
        code: "page_draft_id_conflict",
        error: "Page draft id was already used",
      }));
    }
    const space = this.registeredDraftSpace(args);
    const now = "2026-07-10T12:30:00Z";
    const page: KnowledgePage = {
      id,
      title,
      summary: null,
      content,
      entity_id: null,
      domain: space,
      space,
      source_memory_ids: [],
      version: 1,
      status: "draft",
      creation_kind: "authored",
      review_status: "unconfirmed",
      created_at: now,
      last_compiled: now,
      last_modified: now,
    };
    this.pages = [page, ...this.pages];
    this.pageDraftCreateRequests.set(id, { title, content, space });
    return { ...page };
  }

  private normalizedDraftSpace(args: unknown): string | null {
    return optionalString(args, "space")?.trim() || null;
  }

  private registeredDraftSpace(args: unknown): string | null {
    const space = this.normalizedDraftSpace(args);
    if (space && !this.spaces.some((candidate) => candidate.name === space)) {
      throw new Error(`Space ${JSON.stringify(space)} is not registered`);
    }
    return space;
  }

  private draftFor(command: string, args: unknown): { page: KnowledgePage; index: number } {
    const id = requiredString(command, args, "id");
    const index = this.pages.findIndex((page) => page.id === id && page.status === "draft");
    const page = this.pages[index];
    if (index < 0 || !page) {
      throw new Error(JSON.stringify({
        code: "page_draft_not_found",
        error: "Page draft not found",
      }));
    }
    return { page, index };
  }

  private assertDraftVersion(
    command: string,
    args: unknown,
    page: KnowledgePage,
  ): void {
    const expectedVersion = requiredNumber(command, args, "expectedVersion");
    if (page.version !== expectedVersion) {
      throw new Error(JSON.stringify({
        code: "draft_version_conflict",
        error: "Page draft changed since it was loaded",
        current_version: page.version,
      }));
    }
  }

  private updatePageDraft(args: unknown): KnowledgePage {
    const command = "update_page_draft";
    const { page, index } = this.draftFor(command, args);
    const expectedVersion = requiredNumber(command, args, "expectedVersion");
    const title = stringValue(command, args, "title");
    const content = stringValue(command, args, "content");
    if (!title.trim() && !sanitizeIngressContent(content).trim()) {
      throw new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "A Page draft needs a title or content",
      }));
    }
    const requestedSpace = this.normalizedDraftSpace(args);
    if (
      page.version === expectedVersion + 1
      && page.title === title
      && page.content === content
      && page.space === requestedSpace
      && page.domain === requestedSpace
    ) {
      return { ...page };
    }
    this.assertDraftVersion(command, args, page);
    const space = this.registeredDraftSpace(args);
    const updated: KnowledgePage = {
      ...page,
      title,
      content,
      domain: space,
      space,
      version: page.version + 1,
      last_modified: "2026-07-10T12:31:00Z",
    };
    this.pages[index] = updated;
    return { ...updated };
  }

  private publishPageDraft(args: unknown): KnowledgePage {
    const command = "publish_page_draft";
    const id = requiredString(command, args, "id");
    const expectedVersion = requiredNumber(command, args, "expectedVersion");
    const publishedReplay = this.pages.find((page) => page.id === id);
    if (
      publishedReplay?.status === "active"
      && publishedReplay.version === expectedVersion + 1
    ) {
      return { ...publishedReplay };
    }
    if (publishedReplay && publishedReplay.version !== expectedVersion) {
      this.assertDraftVersion(command, args, publishedReplay);
    }
    if (publishedReplay?.status === "active") {
      throw new Error(`Page ${id} is not a draft`);
    }
    const { page, index } = this.draftFor(command, args);
    this.assertDraftVersion(command, args, page);
    const title = page.title.trim();
    const content = sanitizeIngressContent(page.content);
    if (!title || !content.trim()) {
      throw new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "Title and content are required",
      }));
    }
    const space = page.space?.trim() || null;
    const conflict = this.pages.find((candidate) =>
      candidate.id !== page.id
      && candidate.status === "active"
      && candidate.title.trim().toLowerCase() === title.toLowerCase()
      && (candidate.space?.trim() || null) === space
    );
    if (conflict) {
      throw new Error(JSON.stringify({
        code: "page_title_conflict",
        error: "A Page with this title already exists",
        existing_page_id: conflict.id,
        existing_page_title: conflict.title,
      }));
    }
    const published: KnowledgePage = {
      ...page,
      title,
      content,
      status: "active",
      review_status: "unconfirmed",
      version: page.version + 1,
      last_compiled: "2026-07-10T12:31:00Z",
      last_modified: "2026-07-10T12:31:00Z",
    };
    this.pages[index] = published;
    return { ...published };
  }

  private discardPageDraft(args: unknown): null {
    const command = "discard_page_draft";
    const { page, index } = this.draftFor(command, args);
    this.assertDraftVersion(command, args, page);
    this.pages.splice(index, 1);
    this.pageDraftCreateRequests.set(page.id, null);
    return null;
  }

  private listEntities(args: unknown) {
    const domain = optionalString(args, "domain");
    return this.entityDetails.map((detail) => detail.entity).filter((entity) => !domain || entity.domain === domain || entity.space === domain);
  }

  private getEntityDetail(args: unknown): EntityDetail {
    const entityId = requiredString("get_entity_detail_cmd", args, "entityId");
    const detail = this.entityDetails.find((candidate) => candidate.entity.id === entityId);
    if (!detail) throw new TauriMockArgumentError("get_entity_detail_cmd", "entityId");
    return detail;
  }

  private addObservation(args: unknown): string {
    const entityId = requiredString("add_observation_cmd", args, "entityId");
    const detail = this.entityDetails.find((candidate) => candidate.entity.id === entityId);
    if (!detail) throw new TauriMockArgumentError("add_observation_cmd", "entityId");
    const id = `obs-${detail.observations.length + 1}`;
    detail.observations.push({ id, entity_id: entityId, content: requiredString("add_observation_cmd", args, "content"), source_agent: optionalString(args, "sourceAgent"), confidence: null, confirmed: false, created_at: 1_783_728_000 });
    return id;
  }

  private updateObservation(args: unknown): null {
    const observationId = requiredString("update_observation_cmd", args, "observationId");
    const content = requiredString("update_observation_cmd", args, "content");
    for (const detail of this.entityDetails) detail.observations = detail.observations.map((item) => item.id === observationId ? { ...item, content } : item);
    return null;
  }

  private deleteObservation(args: unknown): null {
    const observationId = requiredString("delete_observation_cmd", args, "observationId");
    for (const detail of this.entityDetails) detail.observations = detail.observations.filter((item) => item.id !== observationId);
    return null;
  }

  private confirmObservation(args: unknown): null {
    const observationId = requiredString("confirm_observation_cmd", args, "observationId");
    const confirmed = requiredBoolean("confirm_observation_cmd", args, "confirmed");
    for (const detail of this.entityDetails) detail.observations = detail.observations.map((item) => item.id === observationId ? { ...item, confirmed } : item);
    return null;
  }

  private confirmEntity(args: unknown): null {
    const entityId = requiredString("confirm_entity_cmd", args, "entityId");
    const confirmed = requiredBoolean("confirm_entity_cmd", args, "confirmed");
    this.entityDetails = this.entityDetails.map((detail) => detail.entity.id === entityId ? { ...detail, entity: { ...detail.entity, confirmed } } : detail);
    return null;
  }

  private deleteEntity(args: unknown): null {
    const entityId = requiredString("delete_entity_cmd", args, "entityId");
    this.entityDetails = this.entityDetails.filter((detail) => detail.entity.id !== entityId);
    return null;
  }

  private listMemories(args: unknown): readonly MemoryItem[] {
    const domain = optionalString(args, "domain");
    const limit = optionalValue(args, "limit");
    const filtered = this.memories.filter((memory) => !domain || memory.domain === domain || memory.space === domain);
    return typeof limit === "number" ? filtered.slice(0, limit) : filtered;
  }

  private listIndexedFiles(): readonly Record<string, unknown>[] {
    return this.memories.map((memory) => ({
      source: "review-fixture.md",
      source_id: memory.source_id,
      title: memory.title,
      summary: memory.summary,
      chunk_count: memory.chunk_count,
      last_modified: memory.last_modified,
      processing: false,
      memory_type: memory.memory_type,
      domain: memory.domain,
      space: memory.space,
      source_agent: memory.source_agent,
      confidence: memory.confidence,
      confirmed: memory.confirmed,
      pinned: memory.pinned,
    }));
  }

  private updateMemory(args: unknown): null {
    const sourceId = requiredString("update_memory_cmd", args, "sourceId");
    const content = optionalValue(args, "content");
    const domain = optionalValue(args, "domain");
    const confirmed = optionalValue(args, "confirmed");
    const memoryType = optionalValue(args, "memoryType");
    if (content !== null && content !== undefined && typeof content !== "string") {
      throw new TauriMockArgumentError("update_memory_cmd", "content");
    }
    if (domain !== null && domain !== undefined && typeof domain !== "string") {
      throw new TauriMockArgumentError("update_memory_cmd", "domain");
    }
    if (confirmed !== null && confirmed !== undefined && typeof confirmed !== "boolean") {
      throw new TauriMockArgumentError("update_memory_cmd", "confirmed");
    }
    if (memoryType !== null && memoryType !== undefined && typeof memoryType !== "string") {
      throw new TauriMockArgumentError("update_memory_cmd", "memoryType");
    }
    if (!this.memories.some((memory) => memory.source_id === sourceId)) {
      throw new TauriMockArgumentError("update_memory_cmd", "sourceId");
    }
    this.memories = this.memories.map((memory) =>
      memory.source_id === sourceId
        ? {
            ...memory,
            ...(typeof content === "string" ? { content } : {}),
            ...(typeof domain === "string" ? { domain, space: domain } : {}),
            ...(typeof confirmed === "boolean" ? { confirmed } : {}),
            ...(typeof memoryType === "string" ? { memory_type: memoryType } : {}),
          }
        : memory
    );
    return null;
  }

  private reclassifyMemory(args: unknown): string {
    const sourceId = requiredString("reclassify_memory_cmd", args, "sourceId");
    const memoryType = requiredString("reclassify_memory_cmd", args, "memoryType");
    if (!this.memories.some((memory) => memory.source_id === sourceId)) {
      throw new TauriMockArgumentError("reclassify_memory_cmd", "sourceId");
    }
    this.memories = this.memories.map((memory) =>
      memory.source_id === sourceId ? { ...memory, memory_type: memoryType } : memory
    );
    return sourceId;
  }

  private setStability(args: unknown): null {
    const sourceId = requiredString("set_stability_cmd", args, "sourceId");
    const stability = requiredString("set_stability_cmd", args, "stability");
    if (!["new", "learned", "confirmed"].includes(stability)) {
      throw new TauriMockArgumentError("set_stability_cmd", "stability");
    }
    if (!this.memories.some((memory) => memory.source_id === sourceId)) {
      throw new TauriMockArgumentError("set_stability_cmd", "sourceId");
    }
    this.memories = this.memories.map((memory) =>
      memory.source_id === sourceId
        ? {
            ...memory,
            stability: stability as "new" | "learned" | "confirmed",
            confirmed: stability === "confirmed",
          }
        : memory
    );
    return null;
  }

  private setMemoryPinned(args: unknown, pinned: boolean): null {
    const command = pinned ? "pin_memory" : "unpin_memory";
    const sourceId = requiredString(command, args, "sourceId");
    if (!this.memories.some((memory) => memory.source_id === sourceId)) {
      throw new TauriMockArgumentError(command, "sourceId");
    }
    this.memories = this.memories.map((memory) =>
      memory.source_id === sourceId ? { ...memory, pinned } : memory
    );
    return null;
  }

  private deleteFileChunks(args: unknown): null {
    const source = requiredString("delete_file_chunks", args, "source");
    const sourceId = requiredString("delete_file_chunks", args, "sourceId");
    if (source !== "memory") {
      throw new TauriMockArgumentError("delete_file_chunks", "source");
    }
    this.memories = this.memories.filter((memory) => memory.source_id !== sourceId);
    return null;
  }

  private confirmMemory(args: unknown): null {
    const sourceId = requiredString("confirm_memory", args, "sourceId");
    const confirmed = requiredBoolean("confirm_memory", args, "confirmed");
    this.memories = this.memories.map((memory) =>
      memory.source_id === sourceId ? { ...memory, confirmed } : memory
    );
    return null;
  }

  private deleteMemory(args: unknown): null {
    const sourceId = requiredString("delete_memory", args, "sourceId");
    this.memories = this.memories.filter((memory) => memory.source_id !== sourceId);
    return null;
  }

  private search(args: unknown) {
    const query = requiredString("search", args, "query").toLowerCase();
    const limit = optionalValue(args, "limit");
    const results = this.memories.filter((memory) => `${memory.title} ${memory.content}`.toLowerCase().includes(query) || query.includes("ada"));
    return results.slice(0, typeof limit === "number" ? limit : 10).map((memory, index) => ({ id: `chunk-${index}`, content: memory.content, source: "fixture.md", source_id: memory.source_id, title: memory.title, url: null, chunk_index: 0, last_modified: memory.last_modified, score: 0.94, memory_type: memory.memory_type, entity_id: query.includes("ada") ? "entity-ada" : null, is_archived: false }));
  }

}
