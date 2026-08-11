// SPDX-License-Identifier: AGPL-3.0-only
// Live browser backend: maps Tauri invoke() commands to the daemon's HTTP API
// (proxied at /daemon → http://127.0.0.1:7878), mirroring app/src/search.rs.
// Commands with no daemon route get app-local defaults; unknown ones warn.

import { daemonMeetsFloor } from "../../src/lib/daemonVersion";

type Args = Record<string, unknown> | undefined;
const PAGE_EDIT_DAEMON_FLOOR = "0.14.1";

class HttpError extends Error {
  status: number;
  body: string;
  constructor(status: number, msg: string, body: string) {
    super(msg);
    this.status = status;
    this.body = body;
  }
}

async function http(
  method: string,
  path: string,
  body?: unknown,
  extraHeaders?: Record<string, string>,
): Promise<any> {
  const res = await fetch(`/daemon${path}`, {
    method,
    headers: {
      ...(body !== undefined ? { "content-type": "application/json" } : undefined),
      ...extraHeaders,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) {
    throw new HttpError(
      res.status,
      text || `${method} ${path} → ${res.status}`,
      text,
    );
  }
  return text ? JSON.parse(text) : null;
}
const get = (p: string, headers?: Record<string, string>) => http("GET", p, undefined, headers);
const post = (p: string, b: unknown = {}, headers?: Record<string, string>) => http("POST", p, b, headers);
const put = (p: string, b: unknown = {}) => http("PUT", p, b);
const del = (p: string) => http("DELETE", p);

// M5 truth axes: the same two-header marker app/src/api.rs attaches on its
// `*_explicit_browse` client methods. Only ever pass this to a route the
// daemon manifest marks Collection or NamedPage (`GET /api/pages`,
// `POST /api/pages/search`, `GET /api/pages/{id}`) — the daemon records
// every marked call, so a preview handler must not send it on an automatic
// read masquerading as a browse.
const EXPLICIT_BROWSE_HEADERS = {
  "x-wenlan-truth-contract": "1",
  "x-wenlan-reader-intent": "explicit",
};

const enc = encodeURIComponent;
const pageUpdateErrorMessage = (
  error: HttpError,
  fallback: string,
): string => {
  try {
    const body = JSON.parse(error.body) as { error?: unknown };
    return typeof body.error === "string" && body.error.trim()
      ? body.error
      : fallback;
  } catch {
    return fallback;
  }
};

const pageUpdateFailureKind = (
  status: number,
):
  | "not_found"
  | "auth_required"
  | "payload_too_large"
  | "validation"
  | "rate_limited"
  | "server"
  | "other" => {
  if (status === 404) return "not_found";
  if (status === 401 || status === 403) return "auth_required";
  if (status === 413) return "payload_too_large";
  if (status === 400 || status === 422) return "validation";
  if (status === 429) return "rate_limited";
  if (status >= 500) return "server";
  return "other";
};

const qs = (obj: Record<string, unknown>) => {
  const parts = Object.entries(obj)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${k}=${enc(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
};

// --- on-device model download simulation ---
// Real hf-hub downloads take minutes; the wizard's "setting up" row only
// needs to prove the honest state machine (running -> cached, loaded set)
// plays out, so this compresses it to ~25s. Module-level so `get_on_device_model`
// and `on_device_model_download_bytes` (a separate poll) can both read it.
const MODEL_DOWNLOAD_DURATION_MS = 25_000;
// The real Qwen3-4B GGUF blob size in bytes (hf-hub `.part` file), used to cap
// the simulated byte ramp just under the real total rather than at a round number.
const MODEL_BLOB_BYTES = 2_497_281_120;
let downloadStartedAt: number | null = null;
let downloadingModelId: string | null = null;

// --- setup wizard: daemon row probe memory (store_memory / get_memory_detail /
// delete_memory) ---
// The wizard's runtime row proves the write path by storing a probe memory,
// reading it straight back, and deleting it. Proxying store_memory to the
// real daemon (as this used to) meant every open of the wizard in a pixel
// review left a REAL memory in the maintainer's live knowledge base — seven
// of these leaked in before this was caught. Same precedent as `add_source`
// below: synthesize instead of mutating production data. Keyed by the
// synthesized id so get_memory_detail/delete_memory can find their own
// probes while still falling through to the live daemon for every other
// (real) memory id the rest of the app might ask about.
const PREVIEW_PROBES = new Map<string, unknown>();
let previewProbeSeq = 0;
const PREVIEW_AUTHORED_PAGES = new Map<string, Record<string, unknown>>();
type PreviewPageUpdateRequest = {
  readonly id: string;
  readonly content: string;
  readonly expectedVersion: number;
};
const PREVIEW_PAGE_UPDATE_RECEIPTS = new Map<
  string,
  { readonly input: PreviewPageUpdateRequest; readonly outcome: { outcome: "saved" } }
>();
const PREVIEW_DELETED_PAGE_IDS = new Set<string>();
const PREVIEW_SPACE_OVERRIDES = new Map<string, Record<string, unknown>>();
const PREVIEW_REMOVED_SPACE_NAMES = new Set<string>();
type PreviewPageDraftCreateRequest = {
  readonly title: string;
  readonly content: string;
  readonly space: string | null;
};
const PREVIEW_PAGE_DRAFT_CREATE_REQUESTS =
  new Map<string, PreviewPageDraftCreateRequest | null>();
let previewAuthoredPageSeq = 0;
const PAGE_BATCH_SIZE = 500;
const CLIENT_PAGE_DRAFT_ID =
  /^page_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const SOURCES_BLOCK_START = "<!-- origin:sources:start -->";
const SOURCES_BLOCK_END = "<!-- origin:sources:end -->";

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

function normalizedSpace(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function spaceRows(wire: any): Record<string, unknown>[] {
  const rows = Array.isArray(wire)
    ? wire
    : Array.isArray(wire?.spaces)
      ? wire.spaces
      : [];
  return rows.filter(
    (candidate): candidate is Record<string, unknown> =>
      candidate !== null && typeof candidate === "object",
  );
}

async function listPreviewSpaces(): Promise<Record<string, unknown>[]> {
  const remote = spaceRows(await get("/api/spaces")).filter((candidate) => {
    const name = typeof candidate.name === "string" ? candidate.name : "";
    return !PREVIEW_REMOVED_SPACE_NAMES.has(name)
      && !PREVIEW_SPACE_OVERRIDES.has(name);
  });
  return [...remote, ...PREVIEW_SPACE_OVERRIDES.values()].map((space) => ({ ...space }));
}

async function registeredPreviewDraftSpace(value: unknown): Promise<string | null> {
  const space = normalizedSpace(value);
  if (!space) return null;
  const spaces = await listPreviewSpaces();
  const registered = spaces.some(
    (candidate) =>
      candidate
      && typeof candidate === "object"
      && Reflect.get(candidate, "name") === space,
  );
  if (!registered) {
    throw new Error(`Space ${JSON.stringify(space)} is not registered`);
  }
  return space;
}

function nextPreviewTimestamp(previous: unknown): string {
  const previousMs = typeof previous === "string" ? Date.parse(previous) : Number.NaN;
  return new Date(Math.max(
    Date.now(),
    Number.isFinite(previousMs) ? previousMs + 1 : 0,
  )).toISOString();
}

function previewPage(
  id: string,
  title: string,
  content: string,
  space: string | null,
  status: "active" | "draft",
): Record<string, unknown> {
  const timestamp = new Date().toISOString();
  return {
    id,
    title,
    summary: null,
    content,
    entity_id: null,
    domain: space,
    space,
    source_memory_ids: [],
    version: 1,
    status,
    created_at: timestamp,
    last_compiled: timestamp,
    last_modified: timestamp,
    creation_kind: "authored",
    review_status: "unconfirmed",
    user_edited: true,
  };
}

function previewDraft(id: unknown): Record<string, unknown> {
  const draft = PREVIEW_AUTHORED_PAGES.get(String(id));
  if (!draft || draft.status !== "draft") {
    throw new Error(JSON.stringify({
      code: "page_draft_not_found",
      error: "Page draft not found",
    }));
  }
  return draft;
}

function assertPreviewDraftVersion(
  draft: Record<string, unknown>,
  expectedVersion: unknown,
): void {
  if (draft.version !== expectedVersion) {
    throw new Error(JSON.stringify({
      code: "draft_version_conflict",
      error: "Page draft changed since it was loaded",
      current_version: draft.version,
    }));
  }
}

function samePreviewPageScope(
  page: Record<string, unknown>,
  title: string,
  space: string | null,
): boolean {
  const pageTitle = String(page.title ?? "").trim().toLowerCase();
  return page.status === "active"
    && pageTitle === title.trim().toLowerCase()
    && normalizedSpace(page.space) === space;
}

async function listAllRemotePages(
  status: string,
  domain: string | null,
  headers?: Record<string, string>,
): Promise<Record<string, unknown>[]> {
  const pages: Record<string, unknown>[] = [];
  const seenIds = new Set<string>();
  let offset = 0;

  while (true) {
    const wire = await get(
      `/api/pages${qs({ status, domain, limit: PAGE_BATCH_SIZE, offset })}`,
      headers,
    );
    const batch = (Array.isArray(wire?.pages) ? wire.pages : wire) as unknown;
    if (!Array.isArray(batch)) return pages;

    let added = 0;
    for (const page of batch) {
      if (!page || typeof page !== "object") continue;
      const id = String(Reflect.get(page, "id") ?? "");
      if (!id || seenIds.has(id)) continue;
      seenIds.add(id);
      pages.push(page as Record<string, unknown>);
      added += 1;
    }
    if (batch.length < PAGE_BATCH_SIZE || added === 0) return pages;
    offset += batch.length;
  }
}

function downloadComplete(): boolean {
  return downloadStartedAt !== null && Date.now() - downloadStartedAt >= MODEL_DOWNLOAD_DURATION_MS;
}

async function listPagesVia(a: any, headers?: Record<string, string>): Promise<unknown> {
  const status = typeof a?.status === "string" ? a.status : "active";
  const domain = normalizedSpace(a?.domain);
  const remote = await listAllRemotePages(status, domain, headers);
  const local = [...PREVIEW_AUTHORED_PAGES.values()].filter((page) => {
    const statusMatches = page.status === status;
    const domainMatches = !domain || page.domain === domain || page.space === domain;
    return statusMatches && domainMatches;
  });
  const localIds = new Set(local.map((page) => page.id));
  const combined = [...local, ...remote.filter((page) => !localIds.has(page.id))];
  const offset = typeof a?.offset === "number" ? a.offset : 0;
  const limit = typeof a?.limit === "number" ? a.limit : undefined;
  return combined.slice(offset, limit === undefined ? undefined : offset + limit);
}

function searchPagesVia(a: any, headers?: Record<string, string>): Promise<unknown> {
  return post(
    "/api/pages/search",
    { query: a.query, limit: a.limit ?? null, page_type: null },
    headers,
  ).then((r) => r.pages ?? r);
}

async function getPageVia(a: any, headers?: Record<string, string>): Promise<unknown> {
  const id = String(a.id);
  const previewPage = PREVIEW_AUTHORED_PAGES.get(id);
  if (previewPage) return previewPage;
  if (PREVIEW_DELETED_PAGE_IDS.has(id)) return null;
  try {
    const wire = await get(`/api/pages/${enc(a.id)}`, headers);
    return wire?.page ?? null;
  } catch (e) {
    if (e instanceof HttpError && e.status === 404) return null;
    throw e;
  }
}

// Exported (not just module-local) so the parity test below can read the
// covered-command key sets without re-parsing this file.
export const HANDLERS: Record<string, (a: any) => Promise<unknown>> = {
  daemon_version: () =>
    get("/api/health").then((response) => String(response?.version ?? "")),
  // --- pages (mirrors search.rs exactly) ---
  get_page: (a) => getPageVia(a),
  // M5 truth axes: same lookup, marked as a human-initiated wiki browse so
  // the daemon attaches `Page.truth` once its cutover is live. Only
  // PageDetail (a page a person navigated to) calls this — see
  // app/src/search.rs::get_page_explicit_browse.
  get_page_explicit_browse: (a) => getPageVia(a, EXPLICIT_BROWSE_HEADERS),
  create_page: (a) => {
    const title = String(a?.title ?? "").trim();
    const content = String(a?.content ?? "").trim();
    const space = normalizedSpace(a?.space);
    if (!title || !content) return Promise.reject(new Error("Title and content are required"));
    const id = `preview-authored-page-${previewAuthoredPageSeq++}`;
    PREVIEW_AUTHORED_PAGES.set(id, previewPage(id, title, content, space, "active"));
    return Promise.resolve({ id, attached_to: null, warnings: [] });
  },
  create_page_draft: async (a) => {
    const id = typeof a?.clientDraftId === "string" ? a.clientDraftId : "";
    if (!CLIENT_PAGE_DRAFT_ID.test(id)) {
      throw new Error(JSON.stringify({
        error: "Page draft id must use the page_<uuid-v4> format",
      }));
    }
    const title = String(a?.title ?? "");
    const content = String(a?.content ?? "");
    if (!title.trim() && !sanitizeIngressContent(content).trim()) {
      throw new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "A Page draft needs a title or content",
      }));
    }
    const requestedSpace = normalizedSpace(a?.space);
    const existing = PREVIEW_AUTHORED_PAGES.get(id);
    if (existing) {
      const request = PREVIEW_PAGE_DRAFT_CREATE_REQUESTS.get(id);
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
    if (PREVIEW_PAGE_DRAFT_CREATE_REQUESTS.has(id)) {
      throw new Error(JSON.stringify({
        code: "page_draft_id_conflict",
        error: "Page draft id was already used",
      }));
    }
    const space = await registeredPreviewDraftSpace(a?.space);
    const draft = previewPage(
      id,
      title,
      content,
      space,
      "draft",
    );
    PREVIEW_AUTHORED_PAGES.set(id, draft);
    PREVIEW_PAGE_DRAFT_CREATE_REQUESTS.set(id, { title, content, space });
    return { ...draft };
  },
  update_page_draft: async (a) => {
    const draft = previewDraft(a?.id);
    const title = String(a?.title ?? "");
    const content = String(a?.content ?? "");
    if (!title.trim() && !sanitizeIngressContent(content).trim()) {
      throw new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "A Page draft needs a title or content",
      }));
    }
    const requestedSpace = normalizedSpace(a?.space);
    if (
      typeof a?.expectedVersion === "number"
      && draft.version === a.expectedVersion + 1
      && draft.title === title
      && draft.content === content
      && draft.space === requestedSpace
      && draft.domain === requestedSpace
    ) {
      return { ...draft };
    }
    assertPreviewDraftVersion(draft, a?.expectedVersion);
    const space = await registeredPreviewDraftSpace(a?.space);
    const updated = {
      ...draft,
      title,
      content,
      domain: space,
      space,
      version: Number(draft.version) + 1,
      last_modified: new Date().toISOString(),
    };
    PREVIEW_AUTHORED_PAGES.set(String(a.id), updated);
    return { ...updated };
  },
  publish_page_draft: async (a) => {
    const id = String(a?.id);
    const expectedVersion = a?.expectedVersion;
    const publishedReplay = PREVIEW_AUTHORED_PAGES.get(id);
    if (
      publishedReplay?.status === "active"
      && typeof expectedVersion === "number"
      && publishedReplay.version === expectedVersion + 1
    ) {
      return { ...publishedReplay };
    }
    if (
      publishedReplay
      && publishedReplay.version !== expectedVersion
    ) {
      assertPreviewDraftVersion(publishedReplay, expectedVersion);
    }
    if (publishedReplay?.status === "active") {
      throw new Error(`Page ${id} is not a draft`);
    }
    const draft = previewDraft(id);
    assertPreviewDraftVersion(draft, expectedVersion);
    const title = String(draft.title ?? "");
    const content = sanitizeIngressContent(String(draft.content ?? ""));
    if (!title.trim() || !content.trim()) {
      return Promise.reject(new Error(JSON.stringify({
        code: "invalid_page_draft",
        error: "Title and content are required",
      })));
    }
    const space = normalizedSpace(draft.space);
    const remotePages = await listAllRemotePages("active", space);
    const conflict = [...PREVIEW_AUTHORED_PAGES.values(), ...remotePages].find(
      (page) => page.id !== draft.id && samePreviewPageScope(page, title, space),
    );
    if (conflict) {
      return Promise.reject(new Error(JSON.stringify({
        code: "page_title_conflict",
        error: "A Page with this title already exists",
        existing_page_id: conflict.id,
        existing_page_title: conflict.title,
      })));
    }
    const published = {
      ...draft,
      title: title.trim(),
      content,
      status: "active",
      review_status: "unconfirmed",
      version: Number(draft.version) + 1,
      last_compiled: new Date().toISOString(),
      last_modified: new Date().toISOString(),
    };
    PREVIEW_AUTHORED_PAGES.set(String(a.id), published);
    return Promise.resolve({ ...published });
  },
  discard_page_draft: (a) => {
    const draft = previewDraft(a?.id);
    assertPreviewDraftVersion(draft, a?.expectedVersion);
    const id = String(a.id);
    PREVIEW_AUTHORED_PAGES.delete(id);
    PREVIEW_DELETED_PAGE_IDS.add(id);
    PREVIEW_PAGE_DRAFT_CREATE_REQUESTS.set(id, null);
    return Promise.resolve(null);
  },
  list_pages: (a) => listPagesVia(a),
  // M5 truth axes: PagesOverview is the one deliberate wiki-browse screen,
  // so it — and only it — calls these two marked siblings. Everything else
  // that lists pages (sidebar recents, the home feed, space counts) stays on
  // the automatic handlers above; see the `SpaceList.tsx` polling note in
  // crates/wenlan-core/src/truth_contract.rs for why that split matters.
  list_pages_explicit_browse: (a) => listPagesVia(a, EXPLICIT_BROWSE_HEADERS),
  search_pages: (a) => searchPagesVia(a),
  get_truth_status: () =>
    get("/api/status").then((r) => r?.truth ?? null),
  list_recent_pages: (a) =>
    get(`/api/pages/recent${qs({ limit: a?.limit, since_ms: a?.sinceMs })}`).then(
      (r) => r.pages ?? r,
    ),
  list_recent_changes: (a) =>
    get(`/api/pages/recent-changes${qs({ limit: a?.limit })}`).then((r) => r.changes ?? r),
  get_page_sources: (a) => get(`/api/pages/${enc(a.pageId)}/sources`).then((r) => r?.sources ?? r),
  get_page_links: (a) => get(`/api/pages/${enc(a.pageId)}/links`),

  // Page map (Canvas) — mirrors WenlanClient::page_map_call in app/src/api.rs.
  // The node DELETE carries a JSON body (base_revision), so it cannot use the
  // `del()` helper, which sends none.
  get_page_map: (a) => get(`/api/pages/${enc(a.pageId)}/map`),
  improve_page_map: (a) => post(`/api/pages/${enc(a.pageId)}/map/improve`, {}),
  create_page_map_node: (a) => post(`/api/pages/${enc(a.pageId)}/map/nodes`, a.body),
  patch_page_map_node: (a) =>
    http("PATCH", `/api/pages/${enc(a.pageId)}/map/nodes/${enc(a.nodeId)}`, a.body),
  delete_page_map_node: (a) =>
    http("DELETE", `/api/pages/${enc(a.pageId)}/map/nodes/${enc(a.nodeId)}`, a.body),
  put_page_map_layout: (a) => put(`/api/pages/${enc(a.pageId)}/map/layout`, a.body),

  // Communities (M6 cartography) — mirrors WenlanClient::list_communities /
  // list_community_members in app/src/api.rs.
  list_communities: (a) =>
    get(`/api/communities${qs({ space: a.space, cursor: a.cursor, limit: a.limit })}`),
  list_community_members: (a) => {
    const cursor = a.cursor as { space: string; node_id: string } | null | undefined;
    return get(
      `/api/communities/members${qs({
        space: a.space,
        cursor_space: cursor?.space,
        cursor_node_id: cursor?.node_id,
        limit: a.limit,
      })}`,
    );
  },
  get_page_revisions: (a) => get(`/api/pages/${enc(a.pageId)}/revisions`),
  redistill_page: (a) => post(`/api/distill/${enc(a.pageId)}`, {}),
  update_page: async (a) => {
    const health = await get("/api/health");
    const reportedVersion = String(health?.version ?? "");
    if (!daemonMeetsFloor(reportedVersion, PAGE_EDIT_DAEMON_FLOOR)) {
      return {
        outcome: "upgrade_required",
        reportedVersion,
        requiredFloor: PAGE_EDIT_DAEMON_FLOOR,
      };
    }

    const id = String(a.id);
    const local = PREVIEW_AUTHORED_PAGES.get(id);
    if (local?.status === "active") {
      const input: PreviewPageUpdateRequest = {
        id,
        content: String(a.content),
        expectedVersion: Number(a.expectedVersion),
      };
      const receiptKey = `${String(a.callerId)}\u0000${String(a.operationId)}`;
      const receipt = PREVIEW_PAGE_UPDATE_RECEIPTS.get(receiptKey);
      if (receipt) {
        const sameRequest =
          receipt.input.id === input.id
          && receipt.input.content === input.content
          && receipt.input.expectedVersion === input.expectedVersion;
        return sameRequest
          ? structuredClone(receipt.outcome)
          : {
              outcome: "conflict",
              message:
                "operation identity was already used for different page content",
            };
      }
      if (Number(local.version) !== input.expectedVersion) {
        return {
          outcome: "conflict",
          message: "This page changed elsewhere.",
        };
      }
      PREVIEW_AUTHORED_PAGES.set(id, {
        ...local,
        content: input.content,
        version: Number(local.version) + 1,
        last_modified: new Date().toISOString(),
      });
      const outcome = { outcome: "saved" } as const;
      PREVIEW_PAGE_UPDATE_RECEIPTS.set(receiptKey, {
        input: structuredClone(input),
        outcome: structuredClone(outcome),
      });
      return outcome;
    }
    try {
      const response = await post(`/api/memory/${enc(a.id)}/update-page`, {
        content: a.content,
        source_memory_ids: [],
        expected_version: a.expectedVersion,
        caller_id: a.callerId,
        operation_id: a.operationId,
      });
      if (!response?.ok || response?.gated) {
        return {
          outcome: "failure",
          kind: "other",
          status: 200,
          message: response?.gated
            ? "Daemon staged the page update instead of saving it."
            : "Daemon did not confirm the page update.",
        };
      }
      return { outcome: "saved" };
    } catch (error) {
      if (!(error instanceof HttpError)) throw error;
      if (error.status === 409) {
        return {
          outcome: "conflict",
          message: pageUpdateErrorMessage(
            error,
            "This page changed elsewhere.",
          ),
        };
      }
      return {
        outcome: "failure",
        kind: pageUpdateFailureKind(error.status),
        status: error.status,
        message: pageUpdateErrorMessage(error, error.message),
      };
    }
  },
  delete_page: (a) => {
    const id = String(a.id);
    if (PREVIEW_AUTHORED_PAGES.has(id)) {
      PREVIEW_AUTHORED_PAGES.delete(id);
      PREVIEW_DELETED_PAGE_IDS.add(id);
      return Promise.resolve(null);
    }
    return del(`/api/pages/${enc(id)}`);
  },
  archive_page: (a) => {
    const id = String(a.id);
    const local = PREVIEW_AUTHORED_PAGES.get(id);
    if (local) {
      PREVIEW_AUTHORED_PAGES.set(id, { ...local, status: "archived" });
      return Promise.resolve(null);
    }
    return post(`/api/pages/${enc(id)}/archive`);
  },
  list_orphan_links: (a) => get(`/api/pages/orphan-links${qs({ min_count: a?.minCount })}`),

  // --- memories ---
  list_recent_memories: (a) =>
    get(`/api/memory/recent${qs({ limit: a?.limit, since_ms: a?.sinceMs })}`).then(
      (r) => r.memories ?? r,
    ),
  get_memory_detail: (a) => {
    const sourceId = a.sourceId as string;
    if (PREVIEW_PROBES.has(sourceId)) return Promise.resolve(PREVIEW_PROBES.get(sourceId));
    return get(`/api/memory/${enc(sourceId)}/detail`).then((r) => r?.memory ?? null);
  },
  list_memories_by_ids: (a) =>
    get(`/api/memory/by-ids?ids=${(a.ids as string[]).map(enc).join(",")}`).then(
      (r) => r.memories ?? r,
    ),
  list_memories_cmd: (a) =>
    post("/api/memory/list", {
      memory_type: a?.memoryType ?? null,
      space: a?.domain ?? null,
      limit: a?.limit ?? 200,
      confirmed: a?.confirmed ?? null,
    }).then((r) =>
      (r.memories ?? []).map((info: any) => ({
        supersedes: null,
        entity_id: null,
        quality: null,
        is_recap: String(info.source_id ?? "").startsWith("recap_"),
        enrichment_status: "raw",
        supersede_mode: "hide",
        structured_fields: null,
        retrieval_cue: null,
        source_text: null,
        access_count: 0,
        version: 1,
        changelog: null,
        pending_revision: false,
        merged_from: null,
        ...info,
        confirmed: info.confirmed ?? false,
      })),
    ),
  list_pinned_memories: () => get("/api/memory/pinned").then((r) => r.memories ?? r),
  list_unconfirmed_memories: (a) =>
    get(`/api/memory/unconfirmed${qs({ limit: a?.limit ?? 50 })}`).then((r) => r.memories ?? r),
  get_memory_stats_cmd: () => get("/api/memory/stats"),
  get_memory_revisions: (a) => get(`/api/memory/${enc(a.sourceId)}/revisions`),
  get_version_chain_cmd: (a) => get(`/api/memory/${enc(a.sourceId)}/versions`),
  search: (a) =>
    post("/api/search", { query: a.query, limit: a?.limit ?? 20 }).then((r) => r.results ?? []),
  search_memory: (a) =>
    post("/api/memory/search", { query: a.query, limit: a?.limit ?? 20 }).then(
      (r) => r.results ?? r.memories ?? [],
    ),

  // --- entities / knowledge ---
  list_entities_cmd: (a) =>
    post("/api/memory/entities/list", { limit: a?.limit ?? 100 }).then((r) => r.entities ?? r),
  search_entities_cmd: (a) =>
    post("/api/memory/entities/search", { query: a.query }).then((r) => r.entities ?? r),
  get_entity_detail_cmd: (a) => get(`/api/memory/entities/${enc(a.entityId)}`),
  get_entity_suggestions_cmd: () => get("/api/memory/entity-suggestions"),
  list_recent_relations: (a) =>
    get(`/api/knowledge/recent-relations${qs({ limit: a?.limit ?? 50 })}`).then(
      (r) => r.relations ?? r,
    ),
  count_knowledge_files: () => get("/api/knowledge/count").then((r) => r.total ?? r.count ?? 0),

  // --- home / profile / spaces / misc ---
  get_home_stats: () => get("/api/home-stats"),
  get_briefing: () => get("/api/briefing"),
  get_profile: () => get("/api/profile"),
  get_profile_narrative: () => get("/api/profile/narrative"),
  list_spaces: () => listPreviewSpaces(),
  update_space: async (a) => {
    const name = normalizedSpace(a?.name);
    const newName = normalizedSpace(a?.newName);
    if (!name || !newName) throw new Error("A Space name is required");
    const spaces = await listPreviewSpaces();
    const current = spaces.find((space) => space.name === name);
    if (!current) throw new Error(`Space ${JSON.stringify(name)} is not registered`);
    if (name !== newName && spaces.some((space) => space.name === newName)) {
      throw new Error(`Space ${JSON.stringify(newName)} already exists`);
    }

    const updated = {
      ...current,
      name: newName,
      description: typeof a?.description === "string"
        ? a.description
        : null,
    };
    if (name !== newName) {
      PREVIEW_REMOVED_SPACE_NAMES.add(name);
      PREVIEW_SPACE_OVERRIDES.delete(name);
      for (const [id, page] of PREVIEW_AUTHORED_PAGES) {
        if (page.space !== name && page.domain !== name) continue;
        const draft = page.status === "draft";
        PREVIEW_AUTHORED_PAGES.set(id, {
          ...page,
          space: page.space === name ? newName : page.space,
          domain: page.domain === name ? newName : page.domain,
          version: draft ? Number(page.version) + 1 : page.version,
          last_modified: draft
            ? nextPreviewTimestamp(page.last_modified)
            : page.last_modified,
        });
      }
    }
    PREVIEW_SPACE_OVERRIDES.set(newName, updated);
    return { ...updated };
  },
  get_capture_stats: () => get("/api/capture-stats").then((r) => r.stats ?? r),
  // TagData shape: the UI reads both r.tags and r.document_tags — no unwrap.
  list_all_tags: () => get("/api/tags"),
  list_agents: () => get("/api/agents"),
  list_agent_activity: (a) => get(`/api/activities${qs({ limit: a?.limit ?? 50 })}`),
  list_activities: (a) => get(`/api/activities${qs({ limit: a?.limit ?? 50 })}`),
  list_recent_retrievals: (a) => get(`/api/retrievals/recent${qs({ limit: a?.limit ?? 20 })}`),
  list_pending_revisions: (a) =>
    get(`/api/memory/pending-revisions${qs({ limit: a?.limit })}`).then((r) => r.revisions ?? r),
  get_pending_contradictions: () =>
    get("/api/memory/contradictions").then((r) => r.contradictions ?? r),
  get_nurture_cards_cmd: () => get("/api/memory/nurture").then((r) => r.cards ?? r),
  list_refinements: () => get("/api/refinery/queue"),
  // Review-only pass: POST /api/distill with an empty body never creates pages.
  distill_review: () => post("/api/distill", {}),
  accept_pending_revision: (a) => post(`/api/memory/revision/${enc(a.sourceId)}/accept`),
  dismiss_pending_revision: (a) => post(`/api/memory/revision/${enc(a.sourceId)}/dismiss`),
  accept_refinement: (a) => post(`/api/refinery/queue/${enc(a.id)}/accept`),
  reject_refinement: (a) => post(`/api/refinery/queue/${enc(a.id)}/reject`),
  list_decisions_cmd: () => get("/api/decisions?limit=200").then((r) => r.decisions ?? r),
  list_decision_domains_cmd: () => get("/api/decisions/domains").then((r) => r.domains ?? r),
  get_working_memory: () => get("/api/memory/working").then((r) => r ?? null),
  pin_memory: (a) => post(`/api/memory/${enc(a.sourceId)}/pin`),
  unpin_memory: (a) => post(`/api/memory/${enc(a.sourceId)}/unpin`),
  confirm_memory: (a) => post(`/api/memory/confirm/${enc(a.sourceId)}`, { confirmed: true }),
  // The wizard's runtime row stores a probe memory, reads it straight back,
  // then deletes it — proving the write path exists for the row to render at
  // all. This USED to proxy to the real daemon on the theory that "the
  // wizard deletes its own probe immediately, so it's harmless" — but the
  // delete is best-effort (and, before this fix, silently swallowed on
  // failure), so a slow/failed cleanup left a REAL "Wenlan setup check"
  // memory behind in the maintainer's own knowledge base. Seven leaked in
  // this way. Same precedent as `add_source` below: synthesize a probe
  // (PREVIEW_PROBES, above) instead of writing anything real. See
  // get_memory_detail and delete_memory above/below for the other two legs
  // of this same round trip.
  store_memory: (a) => {
    const req = (a.req ?? {}) as { content?: string; source_agent?: string };
    const sourceId = `preview-probe-${previewProbeSeq++}`;
    const memory = {
      source_id: sourceId,
      title: (req.content ?? "Preview probe memory").slice(0, 60),
      content: req.content ?? "",
      summary: null,
      memory_type: null,
      domain: null,
      source_agent: req.source_agent ?? null,
      confidence: null,
      confirmed: false,
      pinned: false,
      supersedes: null,
      last_modified: Date.now(),
      chunk_count: 0,
    };
    PREVIEW_PROBES.set(sourceId, memory);
    return Promise.resolve({ source_id: sourceId, enrichment: "not_needed", hint: "" });
  },
  delete_memory: (a) => {
    const sourceId = a.sourceId as string;
    if (PREVIEW_PROBES.has(sourceId)) {
      PREVIEW_PROBES.delete(sourceId);
      return Promise.resolve({ deleted: true });
    }
    return del(`/api/memory/delete/${enc(sourceId)}`);
  },
  update_memory_cmd: (a) => put(`/api/memory/${enc(a.sourceId)}/update`, { content: a.content }),
  get_pipeline_status: () => get("/api/debug/pipeline"),
  list_onboarding_milestones: () =>
    get("/api/onboarding/milestones").then((r) => r.milestones ?? r),

  // --- setup wizard: import row (RegisteredSource / SyncStats shapes,
  // src/lib/tauri.ts) — synthesized rather than proxied to the real daemon,
  // since a preview click-through must not register a real source or ingest
  // real files into the maintainer's local database. ---
  add_source: (a) =>
    Promise.resolve({
      id: `preview-source-${enc(a.path)}`,
      source_type: a.sourceType,
      path: a.path,
      status: "Active",
      last_sync: null,
      file_count: 0,
      memory_count: 0,
    }),
  sync_registered_source: () =>
    Promise.resolve({
      files_found: 12,
      ingested: 12,
      skipped: 0,
      errors: 0,
    }),

  // --- setup wizard: on-device model row (OnDeviceModelResponse shape,
  // src/lib/tauri.ts). Dynamic, not a DEFAULTS literal: the download-progress
  // state below has to be visible through here for the ramp to render at all
  // — see MODEL_DOWNLOAD_DURATION_MS above. ---
  download_on_device_model: async (a) => {
    downloadStartedAt = Date.now();
    downloadingModelId = String(a.modelId);
    await new Promise((resolve) => setTimeout(resolve, MODEL_DOWNLOAD_DURATION_MS));
  },
  get_on_device_model: async () => {
    const complete = downloadComplete();
    const cachedFor = (id: string) => (id === downloadingModelId ? complete : true);
    return {
      loaded: complete ? downloadingModelId : null,
      selected: downloadingModelId,
      models: [
        { id: "qwen3-4b", display_name: "Qwen 3 4B", param_count: "4B", ram_required_gb: 3.0, file_size_gb: 2.7, cached: cachedFor("qwen3-4b") },
        { id: "qwen3.5-9b", display_name: "Qwen 3.5 9B", param_count: "9B", ram_required_gb: 6.0, file_size_gb: 5.5, cached: cachedFor("qwen3.5-9b") },
      ],
    };
  },
  // The byte count an in-flight hf-hub download has written so far (stats
  // `~/.cache/huggingface/hub/models--*/blobs/<etag>.part`), or null when
  // nothing is downloading. Ramps so a progress bar has something to show;
  // reaches the cap right around when the download above resolves.
  on_device_model_download_bytes: async () => {
    if (downloadStartedAt === null) return null;
    const elapsedSeconds = (Date.now() - downloadStartedAt) / 1000;
    return Math.min(elapsedSeconds * 100e6, MODEL_BLOB_BYTES);
  },
  // Feature detection against the real daemon: the live 0.13.2 daemon has no
  // routing endpoint and 404s, which maps to null (LEGACY mode) — exactly what
  // WenlanClient::get_resolved_routing does. A newer daemon returns the object.
  get_resolved_routing: async () => {
    try {
      return await get("/api/config/routing");
    } catch (e) {
      if (e instanceof HttpError && e.status === 404) return null;
      throw e;
    }
  },
  set_source_pin: (a) =>
    put("/api/config", {
      ...(a?.everydaySource != null ? { everyday_source: a.everydaySource } : {}),
      ...(a?.synthesisSource != null ? { synthesis_source: a.synthesisSource } : {}),
    }).then(() => undefined),
};

// App-local commands (no daemon route) → static defaults that route the UI
// to the main screen and render panels empty rather than crashing.
export const DEFAULTS: Record<string, unknown> = {
  should_show_wizard: false,
  get_setup_completed: true,
  // Shapes below mirror src/lib/tauri.ts exactly. A stub that returns null or the
  // wrong keys where the Rust command returns a struct doesn't just render empty —
  // it white-screens the step (RemoteAccessPanel reads status.status unguarded).
  set_traffic_lights_visible: null,
  set_setup_completed: null,
  is_run_at_login_enabled: false,
  list_watch_paths: [],
  list_sources: [],
  list_registered_sources: [],
  list_indexed_files: [],
  list_pending_imports: [],
  get_index_status: {
    indexing: false,
    total_chunks: 0,
    watched_paths: 0,
    last_indexed: null,
  },
  get_api_key: null,
  get_model_choice: [null, null],
  get_external_llm: [null, null],
  get_system_info: null,
  get_remote_access_status: { status: "off" },
  // The five clients a real machine actually reports (this is verbatim what
  // detect_mcp_clients_cmd returns on the maintainer's Mac, incl. one already
  // configured). This used to be `[]`, which meant the preview only ever showed
  // the wizard's EMPTY state — so the redesigned common case, which IS the
  // screen, could not be pixel-reviewed at all. An empty fixture doesn't render
  // "nothing to see"; it renders a different, misleading screen.
  detect_mcp_clients_cmd: [
    { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: true, already_configured: false },
    { name: "Claude Desktop", client_type: "claude_desktop", config_path: "~/Library/Application Support/Claude/claude_desktop_config.json", detected: true, already_configured: false },
    { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: true, already_configured: false },
    { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: true, already_configured: true },
    { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: true, already_configured: false },
  ],
  // WireState (src/lib/tauri.ts) — the "setting up" step's daemon row reads
  // `wire.daemon.reachable` unguarded (app/src/wire_state.rs never returns
  // null, so the frontend never null-checks it either); a stub that omits
  // the field crashes the step instead of rendering it. `mcp_binary` and
  // `clients` mirror a real resolved machine: an installed binary (plus the
  // rest of app/src/mcp_config.rs's candidate trail, missing paths included
  // per wire_state.rs's own docs), and the same 5 clients as
  // `detect_mcp_clients_cmd` above, routed the way app/src/wire_state.rs's
  // `route_for` actually routes them (`claude_code`/`codex_cli` always to
  // "plugin", the rest to "config" unless already configured).
  wire_state: {
    daemon: {
      base_url: "http://127.0.0.1:7878",
      reachable: true,
      version: "0.12.0",
      error: null,
    },
    mcp_binary: {
      command: "/Users/preview/.wenlan/bin/wenlan-mcp",
      args: [],
      candidates: [
        { path: "/Users/preview/.wenlan/bin/wenlan-mcp", exists: true, source: "installed" },
        { path: "/Applications/Wenlan.app/Contents/MacOS/wenlan-mcp", exists: false, source: "bundled" },
        { path: "/Users/preview/.cargo/bin/wenlan-mcp", exists: false, source: "cargo" },
      ],
    },
    clients: [
      // Cursor carries the raw+raw duplicate (both wenlan and legacy origin,
      // no plugin path) — verbatim the real ~/.cursor/mcp.json shape on this
      // machine, so the Diagnostics "remove the old entry" warnbox is
      // pixel-reviewable in preview.
      { client_type: "cursor", name: "Cursor", detected: true, config_path: "~/.cursor/mcp.json", has_raw_entry: true, has_raw_duplicate: true, has_plugin: false, route: "config" },
      { client_type: "claude_desktop", name: "Claude Desktop", detected: true, config_path: "~/Library/Application Support/Claude/claude_desktop_config.json", has_raw_entry: false, has_raw_duplicate: false, has_plugin: false, route: "config" },
      { client_type: "gemini_cli", name: "Gemini CLI", detected: true, config_path: "~/.gemini/settings.json", has_raw_entry: false, has_raw_duplicate: false, has_plugin: false, route: "config" },
      { client_type: "codex_cli", name: "Codex CLI", detected: true, config_path: "~/.codex/config.toml", has_raw_entry: false, has_raw_duplicate: false, has_plugin: true, route: "plugin" },
      { client_type: "claude_code", name: "Claude Code", detected: true, config_path: "~/.claude.json", has_raw_entry: false, has_raw_duplicate: false, has_plugin: false, route: "plugin" },
    ],
  },
  get_wenlan_mcp_entry: null,
  get_avatar_data_url: null,
  get_knowledge_path: null,
  get_session_snapshots: [],
  get_skip_apps: [],
  get_skip_title_patterns: [],
  suggest_tags: [],
  record_page_editor_diagnostic: null,
  // Both review commands are fixtures rather than daemon reads. Readiness was
  // briefly a live `/api/status` call, which made the Review action's state in
  // preview depend on whether the developer happened to have a current daemon
  // running — so the surface designers look at changed shape under them for
  // reasons that had nothing to do with the design. Submitting is a fixture for
  // the stronger reason that it cannot be done truthfully here at all: a real
  // review needs a minted capability, and preview has no Tauri backend to mint
  // one. Nothing is submitted; this is the success affordance, not a success.
  page_review_supported: "ready",
  review_page: { kind: "applied", reviewed_page_version: 1 },
  acknowledge_guarded_quit_request: true,
  cancel_guarded_quit_request: true,
  quit_wenlan_full: null,
  quit_origin_full: null,
  // Preview's daemon is reachable (wire_state above), so a Start would find it up.
  start_daemon_sidecar: { status: "already_running" },
};

export async function liveInvoke(cmd: string, args?: Args): Promise<unknown> {
  const handler = HANDLERS[cmd];
  if (handler) {
    try {
      return await handler(args);
    } catch (e) {
      console.warn(`[preview:live] ${cmd} failed:`, e);
      throw e;
    }
  }
  if (cmd in DEFAULTS) return DEFAULTS[cmd];
  console.warn(`[preview:live] unmapped invoke: ${cmd}`, args);
  return null;
}
