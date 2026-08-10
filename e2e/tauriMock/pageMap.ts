// SPDX-License-Identifier: AGPL-3.0-only
import type {
  PageMap,
  PageMapNode,
  PageMapNodeMutation,
  PageMapRefKind,
} from "../../src/lib/tauri";

/**
 * The canvas half of the Tauri mock.
 *
 * Canvas bugs live in pointer behaviour — a connector dragged into empty space,
 * a box that vanishes on release — which jsdom cannot produce and the Tauri
 * webview will not accept synthetic events for. Chromium can, so the map has to
 * exist here.
 */
function field(args: unknown, key: string): unknown {
  return typeof args === "object" && args !== null ? Reflect.get(args, key) : undefined;
}

function body(args: unknown): Record<string, unknown> {
  const value = field(args, "body");
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function str(source: Record<string, unknown> | unknown, key: string): string | null {
  const value = typeof source === "object" && source !== null ? Reflect.get(source, key) : undefined;
  return typeof value === "string" ? value : null;
}

export class PageMapStore {
  private readonly maps = new Map<string, PageMap>();
  private sequence = 0;

  /**
   * Every page has a map: the daemon seeds one from the page itself.
   *
   * Seeded with the shape the canvas actually has to handle — a centre, live
   * children to select and delete, and one of the daemon's suggestions, which is
   * hidden until asked for.
   */
  private mapFor(pageId: string): PageMap {
    const existing = this.maps.get(pageId);
    if (existing) return existing;
    const node = (
      id: string,
      parent: string | null,
      rank: number,
      refKind: PageMapRefKind,
      label: string | null,
      status: PageMapNode["status"] = "active",
    ): PageMapNode => ({
      id,
      parent_id: parent,
      rank,
      ref_kind: refKind,
      ref_id: refKind === "page" ? pageId : `${pageId}#${id}`,
      label,
      status,
      pinned: false,
      placed: false,
      collapsed: false,
      x: null,
      y: null,
      width: null,
      height: null,
      ref_state: "live",
    });
    const map: PageMap = {
      page_id: pageId,
      revision: 1,
      map_schema: 1,
      viewport: null,
      nodes: [
        node("n_root", null, 0, "page", null),
        node("n_storage", "n_root", 0, "section", "Storage layer"),
        node("n_query", "n_root", 1, "section", "Query path"),
        node("n_leaf", "n_storage", 0, "section", "Blob columns"),
        node("n_guess", "n_root", 2, "section", "Vector search", "suggested"),
      ],
      edges: [],
    };
    this.maps.set(pageId, map);
    return map;
  }

  get(args: unknown): PageMap {
    return structuredClone(this.mapFor(str(args, "pageId") ?? ""));
  }

  /** Improve is the daemon's own pass; here it only has to be idempotent. */
  improve(args: unknown): PageMap {
    return this.get(args);
  }

  create(args: unknown): PageMapNodeMutation {
    const map = this.mapFor(str(args, "pageId") ?? "");
    const request = body(args);
    const parentId = str(request, "parent_id") ?? "n_root";
    const node: PageMapNode = {
      id: `n_new_${++this.sequence}`,
      parent_id: parentId,
      rank: map.nodes.filter((n) => n.parent_id === parentId).length,
      ref_kind: (str(request, "ref_kind") as PageMapRefKind | null) ?? "section",
      ref_id: str(request, "ref_id") ?? `sec_${this.sequence}`,
      label: str(request, "label"),
      status: "active",
      pinned: false,
      placed: false,
      collapsed: false,
      x: null,
      y: null,
      width: null,
      height: null,
      ref_state: "live",
    };
    map.nodes.push(node);
    map.revision += 1;
    return { revision: map.revision, node: structuredClone(node) };
  }

  patch(args: unknown): PageMapNodeMutation {
    const map = this.mapFor(str(args, "pageId") ?? "");
    const nodeId = str(args, "nodeId") ?? "";
    const node = map.nodes.find((n) => n.id === nodeId);
    if (!node) throw new Error(`page map node not found: ${nodeId}`);
    const request = body(args);
    if ("label" in request) node.label = str(request, "label");
    if (typeof request.pinned === "boolean") node.pinned = request.pinned;
    const status = str(request, "status");
    if (status) node.status = status as PageMapNode["status"];
    map.revision += 1;
    return { revision: map.revision, node: structuredClone(node) };
  }

  /** Delete is a tombstone in the daemon, so it is one here too. */
  remove(args: unknown): PageMapNodeMutation {
    const map = this.mapFor(str(args, "pageId") ?? "");
    const nodeId = str(args, "nodeId") ?? "";
    const node = map.nodes.find((n) => n.id === nodeId);
    if (!node) throw new Error(`page map node not found: ${nodeId}`);
    node.status = "dismissed";
    map.revision += 1;
    return { revision: map.revision, node: structuredClone(node) };
  }

  layout(args: unknown): PageMap {
    const map = this.mapFor(str(args, "pageId") ?? "");
    const request = body(args);
    const positions = Array.isArray(request.positions) ? request.positions : [];
    for (const entry of positions) {
      const node = map.nodes.find((n) => n.id === str(entry, "node_id"));
      if (!node) continue;
      const x = Reflect.get(entry as object, "x");
      const y = Reflect.get(entry as object, "y");
      if (typeof x === "number") node.x = x;
      if (typeof y === "number") node.y = y;
      node.placed = true;
    }
    const viewport = request.viewport;
    map.viewport = (viewport ?? null) as PageMap["viewport"];
    map.revision += 1;
    return structuredClone(map);
  }
}
