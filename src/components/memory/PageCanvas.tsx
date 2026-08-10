// SPDX-License-Identifier: AGPL-3.0-only
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
  type SetStateAction,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  Background,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useNodesState,
  useReactFlow,
  type Edge,
  type FinalConnectionState,
  type Node,
  type EdgeTypes,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  asPageMapApiError,
  createPageMapNode,
  deletePageMapNode,
  getPage,
  getPageMap,
  improvePageMap,
  patchPageMapNode,
  putPageMapLayout,
  updatePage,
  type PageMapLayoutPosition,
  type PageMapNode,
  type PageMapStatus,
} from "../../lib/tauri";
import { layoutMap, nodeBoxSize, NODE_HEIGHT } from "../../lib/pageMap/tree";
import { slugify, withHeading } from "../../lib/pageMap/slug";
import { useGraphPalette } from "../../lib/graph/palette";
import CanvasNode, { type CanvasNodeData } from "./canvas/CanvasNode";
import SpineEdge from "./canvas/SpineEdge";

interface PageCanvasProps {
  pageId: string;
  pageTitle: string;
  /** "{ref_kind}:{ref_id}" -> display name, for nodes the daemon left label:null */
  labelOverrides: ReadonlyMap<string, string>;
  onMemoryClick: (sourceId: string) => void;
  onPageClick?: (pageId: string) => void;
  onEntityClick?: (entityId: string) => void;
}

// The map shape this build knows how to edit. A map stamped higher was written
// by a newer Wenlan: render it, never write to it (spec: degrade, never
// corrupt).
const SUPPORTED_MAP_SCHEMA = 1;
const LAYOUT_DEBOUNCE_MS = 600;

// A box the user is still naming. It lives only in local state and never
// reaches the daemon: a node's `ref_id` is fixed at creation (PATCH takes
// label/pinned/status/rank/parent_id and nothing else), so the name has to be
// settled *before* the create call rather than patched in afterwards.
const DRAFT_ID = "__draft__";
const DRAFT_SIZE = { width: 170, height: NODE_HEIGHT };
// Where a new box lands relative to its parent, until the next layout pass
// gives it a real radial slot.
const DRAFT_OFFSET = { x: 150, y: 64 };

// Key caps stay expressions rather than literal JSX text: they name physical
// keys, not copy, so they are not translated and must not read as new
// hardcoded strings.
// `as const` is load-bearing: i18n keys are a literal union here, and a
// `string` annotation would widen them past the point where t() type-checks.
// How far an arrow key moves the selection, and how far with Shift held.
const NUDGE_STEP = 8;
const NUDGE_STEP_COARSE = 40;

// The shortcut list behind the corner disclosure. Gestures lead because they are
// what someone tries in the first ten seconds; the key caps behind them are the
// same actions, faster.
// `as const` is load-bearing: i18n keys are a literal union here, and a
// `string` annotation would widen them past the point where t() type-checks.
const SHORTCUTS = [
  { cap: "Double-click", key: "pageCanvas.hintCreate" },
  { cap: "Right-click", key: "pageCanvas.hintMenu" },
  { cap: "Drag", key: "pageCanvas.hintMarquee" },
  { cap: "Tab", key: "pageCanvas.hintAddChild" },
  { cap: "Enter", key: "pageCanvas.hintAddSibling" },
  { cap: "F2", key: "pageCanvas.hintRename" },
  { cap: "Delete", key: "pageCanvas.hintDelete" },
  { cap: "+ -", key: "pageCanvas.hintZoom" },
  { cap: "Shift 0", key: "pageCanvas.hintActualSize" },
  { cap: "Shift 1", key: "pageCanvas.hintFit" },
  { cap: "Shift 2", key: "pageCanvas.hintZoomSelection" },
  { cap: "Shift /", key: "pageCanvas.hintHelp" },
] as const;

const HELP_PANEL_ID = "page-canvas-help-panel";

/**
 * How the map is framed when it opens.
 *
 * A bare `fitView` frames the extent and nothing else, which on a map this
 * shape is the wrong thing to optimise: a four-box page opened at 1.167, so
 * every box rendered larger than it was designed, and a thirteen-box page
 * opened at 0.563, which rasterizes a 12px label at under 7px. Legibility is
 * the constraint, extent is not — so the opening scale is clamped to a band
 * either side of actual size, and a map too big for the band opens readable and
 * partly off-frame. Seeing all of it is one key or one button away; reading it
 * is what the reader came for.
 *
 * The floor is derived rather than chosen: 12px labels at 0.75 render at 9px,
 * which is the smallest defensible size for UI text. It governs the cold open
 * and nothing else — "fit all" has no floor at all, by design.
 */
const OPENING_VIEW = { minZoom: 0.75, maxZoom: 1, padding: 0.15 } as const;

/**
 * How the map is framed by "fit all" — the key, the button, and the menu item,
 * which are one command and have to agree.
 *
 * They did not: each reached React Flow's bare defaults separately, so opening a
 * thirteen-box page at 0.812 and then pressing the fit key moved it to 0.846 —
 * a change with no cause a reader can see, which is exactly the kind of thing
 * that makes a canvas feel like it needs watching. Same padding as the opening
 * view fixes that.
 *
 * The ceiling is shared too, for a different reason: fitting a four-box page to
 * the frame means magnifying it to 1.167, drawing every box half again larger
 * than it was designed. Actual size is as large as anything should ever get.
 *
 * No floor, deliberately. A command that says it shows everything has to show
 * everything, however small it ends up.
 */
const FIT_ALL_VIEW = { maxZoom: 1, padding: 0.15, duration: 200 } as const;

// How much a key press zooms, and how long the move takes. React Flow's own
// step is 1.2x, which the Controls buttons use; matching it keeps the button
// and the key the same gesture.
const ZOOM_DURATION = 180;

/**
 * How a line arriving at a box on ring `depth` is drawn.
 *
 * A gentle arc along the spoke, drawn by `SpineEdge` — see that file for why
 * neither of React Flow's own edge types will do it. The short version: the
 * default bezier leaves from the bottom and arrives at the top whatever
 * direction the child actually lies in, which turns every branch running
 * sideways or upward into an S, and `straight` drew the map as a wire diagram.
 *
 * The two inks are the canvas's own (`--canvas-edge*`), not the graph palette's:
 * the knowledge-graph views composite their edges over a dark ground where a
 * fainter stroke still reads, and at this size those values sank into the page
 * surface. Named in CSS rather than resolved here so both themes stay in one
 * place — `var()` is legal in an SVG `stroke`.
 */
function spineLook(depth: number): Partial<Edge> {
  return {
    type: "spine",
    style:
      depth <= 1
        ? { stroke: "var(--canvas-edge-strong)", strokeWidth: 1.6 }
        : { stroke: "var(--canvas-edge)", strokeWidth: 1.1 },
  };
}

/** An open right-click menu, positioned in surface-local pixels. */
type CanvasMenu =
  | { kind: "pane"; x: number; y: number; clientX: number; clientY: number }
  | { kind: "node"; x: number; y: number; nodeId: string };

interface MenuItem {
  key: string;
  label: string;
  run: () => void;
  danger?: boolean;
}

const nodeTypes: NodeTypes = { pageMapNode: CanvasNode as NodeTypes[string] };
const edgeTypes: EdgeTypes = { spine: SpineEdge };

export default function PageCanvas(props: PageCanvasProps) {
  // useReactFlow (for the viewport we persist) only works under a provider.
  return (
    <ReactFlowProvider>
      <PageCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

function PageCanvasInner({
  pageId,
  pageTitle,
  labelOverrides,
  onMemoryClick,
  onPageClick,
  onEntityClick,
}: PageCanvasProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const palette = useGraphPalette();
  const { getViewport, screenToFlowPosition, fitView, zoomIn, zoomOut, zoomTo } =
    useReactFlow();
  const [notice, setNotice] = useState<string | null>(null);
  const [menu, setMenu] = useState<CanvasMenu | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  // Node state is local and React Flow owns it during a drag. This is what
  // makes dragging cheap: a pointer move rewrites one node's position instead
  // of re-deriving the whole tree, and every other node keeps its identity so
  // memo(CanvasNode) actually holds.
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [draft, setDraft] = useState<{ parentId: string; x: number; y: number } | null>(null);
  // The named box, held on screen while the daemon catches up. Creating one is
  // three round-trips against a real library — read the page, write the heading,
  // create the node — and the map refetch behind them takes another second or
  // two. Dropping the box at commit left that whole window blank, which reads as
  // the box having been thrown away.
  const [pending, setPending] = useState<
    { parentId: string; label: string; x: number; y: number } | null
  >(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  const {
    data: map,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["page-map", pageId],
    queryFn: () => getPageMap(pageId),
    retry: false,
  });

  const readOnly = !!map && map.map_schema > SUPPORTED_MAP_SCHEMA;

  const invalidate = useCallback(
    () => queryClient.invalidateQueries({ queryKey: ["page-map", pageId] }),
    [queryClient, pageId],
  );

  // Nodes the user has moved whose positions the daemon has not stored yet. A
  // resync must not overwrite these or the box snaps back to its old slot
  // mid-burst, which reads as the drag being ignored.
  //
  // Stamped per drop rather than kept as a plain set: a drag that starts while
  // the previous PUT is still in flight would otherwise be cleared by that
  // PUT's completion, and the refetch behind it would yank the box back to the
  // position the user just moved it away from.
  const dirtyRef = useRef(new Map<string, number>());
  const dragSeq = useRef(0);
  const nodesRef = useRef<Node[]>([]);
  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  useEffect(() => {
    dirtyRef.current.clear();
    setDraft(null);
    setPending(null);
    setEditingId(null);
    setNotice(null);
    setMenu(null);
    setShowSuggestions(false);
  }, [pageId]);

  const untitled = t("pageCanvas.untitled");

  // Suggestions are the daemon's guesses, not the user's map, so a page opens
  // on what the user actually put there. "Improve" is the one thing that asks
  // for them, and it un-hides them for the rest of the session.
  const visibleNodes = useMemo(() => {
    const all = map?.nodes ?? [];
    return showSuggestions ? all : all.filter((n) => n.status !== "suggested");
  }, [map?.nodes, showSuggestions]);
  const views = useMemo(
    () => layoutMap(visibleNodes, labelOverrides, untitled),
    [visibleNodes, labelOverrides, untitled],
  );

  // Suggestions outlive the session that hid them: `showSuggestions` resets on
  // every mount, so without a count to offer back, a reload would strand the
  // ones already in the map where nobody can accept or dismiss them.
  const hiddenSuggestions = useMemo(
    () =>
      showSuggestions
        ? 0
        : (map?.nodes ?? []).filter((n) => n.status === "suggested").length,
    [map?.nodes, showSuggestions],
  );

  // Box metrics and collapse state by id, so the layout PUT can be built from
  // local positions without walking the tree again.
  const metaRef = useRef(new Map<string, { width: number; height: number; collapsed: boolean }>());
  useEffect(() => {
    metaRef.current = new Map(
      views.map((v) => [
        v.node.id,
        { width: v.width, height: v.height, collapsed: v.node.collapsed },
      ]),
    );
  }, [views]);

  const revisionRef = useRef(map?.revision ?? 0);
  useEffect(() => {
    revisionRef.current = map?.revision ?? 0;
  }, [map?.revision]);

  // A losing race is not an error the user caused: say what happened, refresh
  // to the server's truth, and let them redo the one action.
  const handleMutationError = useCallback(
    (e: unknown) => {
      if (asPageMapApiError(e)?.status === 409) {
        setNotice(t("pageCanvas.conflict"));
        void invalidate();
        void refetch();
        return;
      }
      setNotice(t("pageCanvas.mutationError"));
    },
    [invalidate, refetch, t],
  );

  const onMutated = useCallback(() => {
    setNotice(null);
    void invalidate();
  }, [invalidate]);

  const improveMutation = useMutation({
    mutationFn: () => improvePageMap(pageId),
    onSuccess: () => {
      setShowSuggestions(true);
      onMutated();
    },
    onError: handleMutationError,
  });

  const statusMutation = useMutation({
    mutationFn: ({ nodeId, status }: { nodeId: string; status: PageMapStatus }) =>
      patchPageMapNode(pageId, nodeId, {
        base_revision: revisionRef.current,
        status,
      }),
    onSuccess: onMutated,
    onError: handleMutationError,
  });

  const renameMutation = useMutation({
    mutationFn: ({ nodeId, label }: { nodeId: string; label: string }) =>
      patchPageMapNode(pageId, nodeId, {
        base_revision: revisionRef.current,
        label,
      }),
    onSuccess: onMutated,
    onError: handleMutationError,
  });

  // The daemon's DELETE tombstones exactly one node — no cascade, no reparent —
  // and `buildSpine` drops any box whose parent is gone. Tombstoning a parent on
  // its own would therefore make its live children permanently invisible, so the
  // subtree comes down with it, deepest first.
  //
  // ponytail: sequential, because each tombstone bumps the map revision and the
  // DELETE response does not carry the new one. A leaf still costs exactly one
  // call; only a real subtree pays for the re-reads. Cascading daemon-side would
  // collapse this to a single request.
  const removeSubtree = useCallback(
    async (roots: string[]) => {
      const childrenOf = new Map<string, string[]>();
      for (const v of views) {
        const parent = v.node.parent_id;
        if (!parent) continue;
        const kids = childrenOf.get(parent);
        if (kids) kids.push(v.node.id);
        else childrenOf.set(parent, [v.node.id]);
      }
      const ordered: string[] = [];
      const seen = new Set<string>();
      const walk = (id: string) => {
        if (seen.has(id)) return;
        seen.add(id);
        for (const kid of childrenOf.get(id) ?? []) walk(kid);
        ordered.push(id); // post-order: every child precedes its parent
      };
      for (const id of roots) walk(id);

      let revision = revisionRef.current;
      for (let i = 0; i < ordered.length; i += 1) {
        await deletePageMapNode(pageId, ordered[i], { base_revision: revision });
        if (i + 1 < ordered.length) revision = (await getPageMap(pageId)).revision;
      }
      // What went, named while `views` still describes it — after the refetch
      // these nodes are gone and there is nothing left to read a label off.
      return {
        label: views.find((v) => v.node.id === roots[0])?.label ?? "",
        count: ordered.length,
      };
    },
    [views, pageId],
  );

  const removeMutation = useMutation({
    mutationFn: removeSubtree,
    // Say what was destroyed, and that it is destroyed. Deleting a box takes
    // its whole subtree and the daemon's tombstones are terminal — the section
    // cannot be re-added under the same name — so a bare Backspace on a
    // selected box is the most expensive keystroke on this surface. It used to
    // be the quietest too: `onMutated` clears the notice area, so success said
    // nothing at all.
    //
    // The notice goes after `onMutated` for exactly that reason; before it, the
    // clear would swallow the message.
    onSuccess: ({ label, count }) => {
      onMutated();
      setNotice(
        count > 1
          ? t("pageCanvas.deletedWith", { label, count: count - 1 })
          : t("pageCanvas.deletedOne", { label }),
      );
    },
    onError: handleMutationError,
  });

  // A new box is a new section of the page. The daemon recomputes a section
  // node's liveness from the page's own headings on every read
  // (`compute_ref_state`), so the heading has to exist before the node does —
  // otherwise the box the user just drew comes back marked as gone.
  const addMutation = useMutation({
    mutationFn: async ({
      parentId,
      label,
      x,
      y,
    }: {
      parentId: string;
      label: string;
      x: number;
      y: number;
    }) => {
      const page = await getPage(pageId);
      const content = page?.content ?? "";
      const next = withHeading(content, label);
      // A new box is a new section, so the page text gains the heading. The
      // write goes through the same guarded path the editor uses — it reports
      // refusals as a value rather than throwing, and a heading that silently
      // did not land would leave the box with nothing to point at, so anything
      // but `saved` becomes the error the notice already knows how to show.
      if (next !== content && page) {
        const outcome = await updatePage({
          id: pageId,
          content: next,
          expectedVersion: page.version,
          callerId: "wenlan-app",
          operationId: globalThis.crypto.randomUUID(),
        });
        if (outcome.outcome !== "saved") {
          throw new Error(`page update ${outcome.outcome}`);
        }
      }
      const created = await createPageMapNode(pageId, {
        base_revision: revisionRef.current,
        parent_id: parentId,
        ref_kind: "section",
        ref_id: `${pageId}#${slugify(label)}`,
        label,
      });
      // Where the box was let go of is the whole statement of where it belongs,
      // and create cannot carry a position — so write it here, before anything
      // refetches. A node that arrives unplaced is given a computed ring slot,
      // which is the box jumping out from under the pointer a beat after it
      // was made. Keep the center and take the real width for the label, so it
      // settles around the drop point instead of away from it.
      const size = nodeBoxSize(label);
      try {
        await putPageMapLayout(pageId, {
          base_revision: created.revision,
          positions: [
            {
              node_id: created.node.id,
              x: x + DRAFT_SIZE.width / 2,
              y: y + DRAFT_SIZE.height / 2,
              width: size.width,
              height: size.height,
              collapsed: false,
            },
          ],
        });
      } catch {
        // The box exists; only its position was lost. Landing in the computed
        // slot is worse than staying put, but it is not a failed create and
        // must not be reported as one.
      }
      return created;
    },
    onSuccess: async () => {
      setNotice(null);
      // The page body gained a heading, so the reading column and the revision
      // list are both stale now.
      void queryClient.invalidateQueries({ queryKey: ["page", pageId] });
      void queryClient.invalidateQueries({ queryKey: ["page-revisions", pageId] });
      // Held until the refetch lands: clearing first would blink the box out and
      // back in as the server copy takes its place.
      await invalidate();
      setPending(null);
    },
    onError: (error: unknown) => {
      setPending(null);
      handleMutationError(error);
    },
  });

  const layoutTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (layoutTimer.current) clearTimeout(layoutTimer.current);
    },
    [],
  );

  const flushLayout = useCallback(async () => {
    // Snapshot the stamps this PUT is answering for, so a drop that lands
    // while it is in flight stays dirty.
    const flushing = new Map(dirtyRef.current);
    const meta = metaRef.current;
    const positions: PageMapLayoutPosition[] = [];
    for (const n of nodesRef.current) {
      const m = meta.get(n.id);
      if (!m) continue; // the draft box, or a node the last refetch dropped
      // React Flow positions are top-left; the daemon stores box centers, so a
      // round-trip through layoutMap lands where the user let go.
      positions.push({
        node_id: n.id,
        x: n.position.x + m.width / 2,
        y: n.position.y + m.height / 2,
        width: m.width,
        height: m.height,
        collapsed: m.collapsed,
      });
    }
    if (positions.length === 0) return;
    const viewport = getViewport();
    const put = (base_revision: number) =>
      putPageMapLayout(pageId, { base_revision, viewport, positions });

    try {
      await put(revisionRef.current);
    } catch (e) {
      if (asPageMapApiError(e)?.status !== 409) {
        setNotice(t("pageCanvas.mutationError"));
        return;
      }
      // Where the user dropped a node IS the intent, so it wins the race:
      // refetch the revision and replay the same coordinates exactly once.
      const fresh = await refetch();
      const rev = fresh.data?.revision;
      if (rev === undefined) {
        setNotice(t("pageCanvas.conflict"));
        return;
      }
      try {
        await put(rev);
      } catch {
        setNotice(t("pageCanvas.conflict"));
        return;
      }
    }
    for (const [id, stamp] of flushing) {
      if (dirtyRef.current.get(id) === stamp) dirtyRef.current.delete(id);
    }
    setNotice(null);
    void invalidate();
  }, [getViewport, pageId, refetch, invalidate, t]);

  // One PUT per burst: every move restarts the timer, so dragging five nodes in
  // a row — or holding an arrow key down — sends one request carrying them all.
  const markMoved = useCallback(
    (ids: string[]) => {
      for (const id of ids) {
        dragSeq.current += 1;
        dirtyRef.current.set(id, dragSeq.current);
      }
      if (layoutTimer.current) clearTimeout(layoutTimer.current);
      layoutTimer.current = setTimeout(() => {
        layoutTimer.current = null;
        void flushLayout();
      }, LAYOUT_DEBOUNCE_MS);
    },
    [flushLayout],
  );

  const handleNodeDragStop = useCallback(
    (_event: unknown, node: Node) => markMoved([node.id]),
    [markMoved],
  );

  const openNode = useCallback(
    (node: PageMapNode) => {
      if (node.ref_kind === "memory") onMemoryClick(node.ref_id);
      else if (node.ref_kind === "page") onPageClick?.(node.ref_id);
      else if (node.ref_kind === "entity") onEntityClick?.(node.ref_id);
      // "section" is a pure grouping node — there is nothing behind it to open.
    },
    [onMemoryClick, onPageClick, onEntityClick],
  );

  const { mutate: mutateStatus } = statusMutation;
  const { mutate: mutateRename } = renameMutation;
  const { mutate: mutateRemove } = removeMutation;
  const { mutate: mutateAdd } = addMutation;

  // Server-derived nodes. Rebuilt only when the map itself changes, never on a
  // drag frame — every callback below closes over stable identities so the
  // `data` object can survive a whole drag untouched.
  const serverNodes = useMemo<Node[]>(
    () =>
      views.map((v) => {
        const data: CanvasNodeData = {
          label: v.label,
          refKind: v.node.ref_kind,
          status: v.node.status,
          dangling: v.node.ref_state === "dangling",
          isRoot: v.node.parent_id === null,
          depth: v.depth,
          // Layout puts the page at the origin, so the box's own center already
          // points away from it.
          outward: Math.atan2(v.y, v.x),
          readOnly,
          palette,
          width: v.width,
          height: v.height,
          editing: false,
          onOpen: () => openNode(v.node),
          onAccept: () => mutateStatus({ nodeId: v.node.id, status: "active" }),
          onDismiss: () => mutateStatus({ nodeId: v.node.id, status: "dismissed" }),
          onCommit: (label: string) => {
            setEditingId(null);
            const trimmed = label.trim();
            if (trimmed && trimmed !== v.label) {
              mutateRename({ nodeId: v.node.id, label: trimmed });
            }
          },
          onCancel: () => setEditingId(null),
        };
        return {
          id: v.node.id,
          type: "pageMapNode",
          draggable: !readOnly,
          position: { x: v.x - v.width / 2, y: v.y - v.height / 2 },
          data,
        };
      }),
    [views, readOnly, palette, openNode, mutateStatus, mutateRename],
  );

  // Fold the server's copy into local state, keeping what the user has done
  // since: an unflushed drag outranks the stored position, and selection must
  // survive a refetch or the map deselects itself every time it saves.
  useEffect(() => {
    setNodes((prev) => {
      const prevById = new Map(prev.map((n) => [n.id, n]));
      return serverNodes.map((n) => {
        const old = prevById.get(n.id);
        if (!old) return n;
        return {
          ...n,
          position: dirtyRef.current.has(n.id) ? old.position : n.position,
          selected: old.selected,
        };
      });
    });
  }, [serverNodes, setNodes]);

  const commitDraft = useCallback(
    (label: string) => {
      const pending = draft;
      if (!pending) return;
      const trimmed = label.trim();
      // An empty commit is never someone deciding against the box: the field
      // blurs on its own when React Flow takes focus back for the pane after a
      // connector drag, and clearing the draft here deleted the box a beat
      // after it appeared. Leave it standing — Escape is how you say no.
      if (!trimmed) return;
      setDraft(null);
      if (!slugify(trimmed)) {
        setNotice(t("pageCanvas.badSectionName"));
        return;
      }
      setPending({ ...pending, label: trimmed });
      mutateAdd({
        parentId: pending.parentId,
        label: trimmed,
        x: pending.x,
        y: pending.y,
      });
    },
    [draft, mutateAdd, t],
  );

  // One box, two moments: being named, then waiting for the daemon. It keeps the
  // same id and position through both so it never blinks out between them.
  const draftNode = useMemo<Node | null>(() => {
    const spot = draft ?? pending;
    if (!spot) return null;
    const naming = draft !== null;
    const data: CanvasNodeData = {
      label: naming ? "" : (pending?.label ?? ""),
      refKind: "section",
      status: "active",
      dangling: false,
      isRoot: false,
      // Dress it as what it is about to become, so it does not restyle itself
      // the instant the server's copy arrives.
      depth: (views.find((v) => v.node.id === spot.parentId)?.depth ?? 0) + 1,
      outward: Math.atan2(
        spot.y + DRAFT_SIZE.height / 2,
        spot.x + DRAFT_SIZE.width / 2,
      ),
      readOnly: false,
      palette,
      width: DRAFT_SIZE.width,
      height: DRAFT_SIZE.height,
      editing: naming,
      placeholder: t("pageCanvas.newSectionPlaceholder"),
      onOpen: () => {},
      onAccept: () => {},
      onDismiss: () => {},
      onCommit: commitDraft,
      onCancel: () => setDraft(null),
    };
    return {
      id: DRAFT_ID,
      type: "pageMapNode",
      draggable: false,
      selectable: false,
      position: { x: spot.x, y: spot.y },
      // The draft lives only in displayNodes, never in `nodes`, so the measure
      // React Flow reports for it is dropped by applyNodeChanges and it would
      // stay visibility:hidden forever — with an unfocusable name field inside
      // it. We already know the size, so say it and skip measuring.
      width: DRAFT_SIZE.width,
      height: DRAFT_SIZE.height,
      data,
    };
  }, [draft, pending, views, palette, commitDraft, t]);

  const rootId = useMemo(
    () => views.find((v) => v.node.parent_id === null)?.node.id ?? null,
    [views],
  );

  const startDraftAt = useCallback((parentId: string, x: number, y: number) => {
    setEditingId(null);
    setMenu(null);
    // Why the last thing failed has nothing to do with the box being drawn now.
    // Left up, "The center box is the page itself" sat over a brand-new box and
    // read as a complaint about that one.
    setNotice(null);
    setDraft({ parentId, x, y });
  }, []);

  const startDraft = useCallback(
    (mode: "child" | "sibling") => {
      if (!rootId) return;
      const selected = nodesRef.current.find((n) => n.selected);
      const anchorId = selected?.id ?? rootId;
      const parentId =
        !selected || mode === "child"
          ? anchorId
          : (views.find((v) => v.node.id === anchorId)?.node.parent_id ?? rootId);
      const base =
        nodesRef.current.find((n) => n.id === anchorId)?.position ?? { x: 0, y: 0 };
      startDraftAt(parentId, base.x + DRAFT_OFFSET.x, base.y + DRAFT_OFFSET.y);
    },
    [views, rootId, startDraftAt],
  );

  // A box drawn on empty canvas still has to join the tree — every node here
  // carries a parent — so it hangs off whatever is selected, or off the page
  // itself. It lands centered on the pointer, where the user aimed.
  const addBoxAt = useCallback(
    (clientX: number, clientY: number, parentId?: string) => {
      if (readOnly || !rootId) return;
      const point = screenToFlowPosition({ x: clientX, y: clientY });
      const selected = nodesRef.current.find((n) => n.selected && n.id !== DRAFT_ID);
      startDraftAt(
        parentId ?? selected?.id ?? rootId,
        point.x - DRAFT_SIZE.width / 2,
        point.y - DRAFT_SIZE.height / 2,
      );
    },
    [readOnly, rootId, screenToFlowPosition, startDraftAt],
  );

  // Drag a box's connector into empty space and let go: the new box is already
  // a child of the one you dragged from. Dropping onto another box does nothing
  // — re-parenting by drag is a separate feature.
  const handleConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      if (state.isValid) return;
      const from = state.fromNode?.id;
      if (!from || from === DRAFT_ID) return;
      // React Flow hands over the raw DOM event, which is the only thing here
      // carrying where the pointer was released.
      const point = "changedTouches" in event ? event.changedTouches[0] : event;
      if (!point) return;
      addBoxAt(point.clientX, point.clientY, from);
    },
    [addBoxAt],
  );

  const selectAll = useCallback(() => {
    setNodes((ns) =>
      ns.map((n) => (n.id === DRAFT_ID ? n : { ...n, selected: true })),
    );
  }, [setNodes]);

  const clearSelection = useCallback(() => {
    setNodes((ns) => ns.map((n) => (n.selected ? { ...n, selected: false } : n)));
  }, [setNodes]);

  const nudge = useCallback(
    (dx: number, dy: number) => {
      const ids = nodesRef.current
        .filter((n) => n.selected && n.id !== DRAFT_ID)
        .map((n) => n.id);
      if (ids.length === 0) return;
      const moving = new Set(ids);
      setNodes((ns) =>
        ns.map((n) =>
          moving.has(n.id)
            ? { ...n, position: { x: n.position.x + dx, y: n.position.y + dy } }
            : n,
        ),
      );
      markMoved(ids);
    },
    [setNodes, markMoved],
  );

  const deleteSelection = useCallback(() => {
    const selected = nodesRef.current
      .filter((n) => n.selected && n.id !== DRAFT_ID)
      .map((n) => n.id);
    if (selected.length === 0) return;
    const roots = new Set(
      views.filter((v) => v.node.parent_id === null).map((v) => v.node.id),
    );
    const deletable = selected.filter((id) => !roots.has(id));
    // Selecting everything and hitting Delete should still clear the map around
    // the center box, so this only complains when the center box was the *whole*
    // selection.
    if (deletable.length === 0) {
      setNotice(t("pageCanvas.rootUndeletable"));
      return;
    }
    setMenu(null);
    mutateRemove(deletable);
  }, [views, mutateRemove, t]);

  const localPoint = useCallback((clientX: number, clientY: number) => {
    const box = surfaceRef.current?.getBoundingClientRect();
    return { x: clientX - (box?.left ?? 0), y: clientY - (box?.top ?? 0) };
  }, []);

  const handlePaneContextMenu = useCallback(
    (event: ReactMouseEvent | MouseEvent) => {
      event.preventDefault(); // the browser's own menu has nothing to offer here
      const point = localPoint(event.clientX, event.clientY);
      setMenu({
        kind: "pane",
        x: point.x,
        y: point.y,
        clientX: event.clientX,
        clientY: event.clientY,
      });
    },
    [localPoint],
  );

  const handleNodeContextMenu = useCallback(
    (event: ReactMouseEvent, node: Node) => {
      event.preventDefault();
      if (node.id === DRAFT_ID) return;
      const point = localPoint(event.clientX, event.clientY);
      setMenu({ kind: "node", x: point.x, y: point.y, nodeId: node.id });
      // Right-clicking inside an existing multi-selection keeps it — that is the
      // whole point of "delete these four". Right-clicking outside one moves the
      // selection to the box under the pointer, so the menu's verbs have an
      // obvious referent.
      setNodes((ns) =>
        ns.some((n) => n.id === node.id && n.selected)
          ? ns
          : ns.map((n) => ({ ...n, selected: n.id === node.id })),
      );
    },
    [localPoint, setNodes],
  );

  // Double-click on empty canvas draws a box; on a box it renames. React Flow
  // has no pane-doubleclick callback, so the surface listens and defers to the
  // node handler when the pointer was over one.
  const handleDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      // The actions, the badge and the shortcut disclosure all float inside the
      // surface, so a double-click on any of them would otherwise draw a box
      // behind the control the user was aiming at.
      const inert =
        ".react-flow__node, .page-canvas-actions, .page-canvas-help, .page-canvas-badge";
      if ((event.target as HTMLElement).closest(inert)) return;
      addBoxAt(event.clientX, event.clientY);
    },
    [addBoxAt],
  );

  const handleNodeDoubleClick = useCallback(
    (_event: ReactMouseEvent, node: Node) => {
      if (readOnly || node.id === DRAFT_ID) return;
      setMenu(null);
      setEditingId(node.id);
    },
    [readOnly],
  );

  const menuItems = useMemo<MenuItem[]>(() => {
    if (!menu) return [];
    if (menu.kind === "pane") {
      const items: MenuItem[] = [];
      if (!readOnly) {
        items.push({
          key: "add",
          label: t("pageCanvas.menuAddHere"),
          run: () => addBoxAt(menu.clientX, menu.clientY),
        });
      }
      items.push(
        { key: "all", label: t("pageCanvas.menuSelectAll"), run: selectAll },
        {
          key: "fit",
          label: t("pageCanvas.menuFitView"),
          run: () => void fitView(FIT_ALL_VIEW),
        },
      );
      return items;
    }
    const view = views.find((v) => v.node.id === menu.nodeId);
    if (!view) return [];
    const items: MenuItem[] = [];
    // A section is a heading in the page body — there is nothing behind it to
    // open, so the verb is omitted rather than shown doing nothing.
    if (view.node.ref_kind !== "section") {
      items.push({
        key: "open",
        label: t("pageCanvas.menuOpen"),
        run: () => openNode(view.node),
      });
    }
    if (readOnly) return items;
    const anchor = nodesRef.current.find((n) => n.id === menu.nodeId)?.position;
    items.push(
      {
        key: "child",
        label: t("pageCanvas.menuAddChild"),
        run: () =>
          startDraftAt(
            menu.nodeId,
            (anchor?.x ?? 0) + DRAFT_OFFSET.x,
            (anchor?.y ?? 0) + DRAFT_OFFSET.y,
          ),
      },
      {
        key: "rename",
        label: t("pageCanvas.menuRename"),
        run: () => {
          setMenu(null);
          setEditingId(menu.nodeId);
        },
      },
    );
    if (view.node.parent_id !== null) {
      const kids = views.some((v) => v.node.parent_id === menu.nodeId);
      items.push({
        key: "delete",
        // Naming the cascade is the only warning the user gets — the boxes
        // underneath go too, and there is no undo yet.
        label: kids
          ? t("pageCanvas.menuDeleteSubtree")
          : t("pageCanvas.menuDelete"),
        run: deleteSelection,
        danger: true,
      });
    }
    return items;
  }, [
    menu,
    views,
    readOnly,
    t,
    addBoxAt,
    selectAll,
    fitView,
    openNode,
    startDraftAt,
    deleteSelection,
  ]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      // Typing a box's name is not a canvas shortcut.
      if (target && (target.tagName === "INPUT" || target.isContentEditable)) return;

      // Escape and select-all stay available on a read-only map: neither writes.
      //
      // It unwinds one layer at a time — the menu, then the shortcut sheet, then
      // the box being named, then the selection — and only reaches the window
      // handler that leaves the page once the canvas has nothing of its own left
      // to dismiss. Letting it through unconditionally meant closing a menu threw
      // the reader out of the page they were drawing.
      if (e.key === "Escape") {
        if (menu) setMenu(null);
        else if (helpOpen) setHelpOpen(false);
        else if (draft || editingId) {
          setDraft(null);
          setEditingId(null);
        } else if (nodesRef.current.some((n) => n.selected)) clearSelection();
        else return; // nothing of ours to close: Escape means "leave the page"
        // Claimed only once something was actually closed: the window handler
        // that leaves the page skips a press whose default is already
        // prevented, so preventing it up front swallowed the last Escape.
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && (e.key === "a" || e.key === "A")) {
        e.preventDefault();
        selectAll();
        return;
      }
      // Everything below is unmodified, and Cmd/Ctrl belongs to the window
      // rather than to the map: `zoomHotkeysEnabled` in tauri.conf.json gives
      // the webview its own Cmd+plus page zoom, so letting that press reach the
      // zoom keys below would resize the app UI and the map at once. Cmd+Tab and
      // Cmd+Arrow are the system's too, and were being swallowed here.
      if (e.metaKey || e.ctrlKey) return;

      // "?" is what GitHub, Linear and Slack all bind to the shortcut sheet.
      // e.code again: the glyph is Shift+/ on a US layout and something else
      // elsewhere, but the physical key is Slash either way.
      if (e.shiftKey && e.code === "Slash") {
        e.preventDefault();
        setHelpOpen((v) => !v);
        return;
      }
      // Bare +/- rather than the Cmd pair every browser uses, because Cmd is the
      // window's (above) — and bare is what Figma, Miro and tldraw bind, so it
      // is the gesture people arrive with.
      //
      // Matched by glyph AND by physical key, because neither alone is enough:
      // e.code binds the US position of "+", which on a German layout is a
      // different key entirely, so those readers would press the key printed
      // with a + and get nothing; e.key alone misses the numpad on layouts that
      // report it oddly. Shift is not excluded — "+" *is* Shift+Equal here.
      if (e.key === "+" || e.key === "=" || e.code === "Equal" || e.code === "NumpadAdd") {
        e.preventDefault();
        void zoomIn({ duration: ZOOM_DURATION });
        return;
      }
      if (e.key === "-" || e.code === "Minus" || e.code === "NumpadSubtract") {
        e.preventDefault();
        void zoomOut({ duration: ZOOM_DURATION });
        return;
      }
      // Figma's view keys — 0 for actual size, 1 for fit, 2 for fit selection —
      // which is also what the map opens at, so 0 is the way back.
      // e.code rather than e.key: Shift+1 reports "!" on a US layout, and a
      // different glyph again on most others.
      if (e.shiftKey && e.code === "Digit0") {
        e.preventDefault();
        void zoomTo(1, { duration: ZOOM_DURATION });
        return;
      }
      if (e.shiftKey && (e.code === "Digit1" || e.code === "Digit2")) {
        e.preventDefault();
        const picked =
          e.code === "Digit2" ? nodesRef.current.filter((n) => n.selected) : [];
        // Passing `nodes: undefined` is not the same as omitting it: React Flow
        // reads the key as "fit exactly these" and an empty list fits nothing.
        //
        // Zooming to a selection keeps no ceiling: filling the frame with one box
        // is the whole gesture, and Figma's Shift 2 magnifies for the same reason.
        void fitView(
          picked.length ? { duration: 200, nodes: picked } : FIT_ALL_VIEW,
        );
        return;
      }
      if (readOnly) return;
      // Every shortcut below moves, renames or removes something the open menu
      // is pointing at, so the menu stops being about anything.
      if (menu) setMenu(null);

      const selected = nodesRef.current.find((n) => n.selected);
      const step = e.shiftKey ? NUDGE_STEP_COARSE : NUDGE_STEP;
      switch (e.key) {
        case "Tab":
          e.preventDefault();
          startDraft("child");
          break;
        case "Enter":
          e.preventDefault();
          startDraft("sibling");
          break;
        case "F2":
          e.preventDefault();
          if (selected) setEditingId(selected.id);
          break;
        case "Delete":
        case "Backspace":
          e.preventDefault();
          deleteSelection();
          break;
        case "ArrowUp":
          e.preventDefault();
          nudge(0, -step);
          break;
        case "ArrowDown":
          e.preventDefault();
          nudge(0, step);
          break;
        case "ArrowLeft":
          e.preventDefault();
          nudge(-step, 0);
          break;
        case "ArrowRight":
          e.preventDefault();
          nudge(step, 0);
          break;
        default:
          break;
      }
    },
    [
      readOnly,
      startDraft,
      deleteSelection,
      nudge,
      selectAll,
      clearSelection,
      fitView,
      zoomIn,
      zoomOut,
      zoomTo,
      menu,
      draft,
      editingId,
      helpOpen,
    ],
  );

  // The array React Flow actually renders. In the common case — including
  // every frame of a drag — this IS the local state, so nothing is rebuilt and
  // memo(CanvasNode) holds for every box.
  const displayNodes = useMemo<Node[]>(() => {
    // First frame with data: state is still empty because the sync effect has
    // not run yet, and mounting React Flow on an empty array would make
    // fitView frame nothing.
    const base = nodes.length === 0 ? serverNodes : nodes;
    if (!editingId && !draftNode) return base;
    const out = editingId
      ? base.map((n) =>
          n.id === editingId ? { ...n, data: { ...n.data, editing: true } } : n,
        )
      : base.slice();
    if (draftNode) out.push(draftNode);
    return out;
  }, [nodes, serverNodes, editingId, draftNode]);

  const rfEdges = useMemo<Edge[]>(() => {
    const rendered = new Set(views.map((v) => v.node.id));
    const edges: Edge[] = [];
    for (const v of views) {
      const parent = v.node.parent_id;
      if (!parent || !rendered.has(parent)) continue;
      // "tree-" prefix keeps derived spine edges out of the server id space.
      edges.push({
        id: `tree-${v.node.id}`,
        source: parent,
        sourceHandle: "anchor",
        target: v.node.id,
        // The lines are the structure, so they thin out as the structure does:
        // spokes off the page carry weight, the twigs past them recede. Drawn in
        // one flat ink they were the faintest thing on a map they hold together.
        ...spineLook(v.depth),
      });
    }
    for (const e of map?.edges ?? []) {
      if (e.status === "dismissed") continue;
      if (!rendered.has(e.from_node) || !rendered.has(e.to_node)) continue;
      edges.push({
        id: e.id,
        source: e.from_node,
        sourceHandle: "anchor",
        target: e.to_node,
        label: e.label ?? undefined,
        style: {
          stroke: palette.bridge,
          strokeWidth: 1,
          strokeDasharray: "4 3",
        },
      });
    }
    // A box dragged out of a connector is a child of the box it came from, and
    // the connection line vanishes the moment the pointer lifts. Without this,
    // the new box hangs unattached until the daemon's copy arrives, which looks
    // exactly like the drag having failed.
    const spot = draft ?? pending;
    if (spot && rendered.has(spot.parentId)) {
      edges.push({
        id: `tree-${DRAFT_ID}`,
        source: spot.parentId,
        sourceHandle: "anchor",
        target: DRAFT_ID,
        ...spineLook((views.find((v) => v.node.id === spot.parentId)?.depth ?? 0) + 1),
      });
    }
    return edges;
  }, [views, map?.edges, palette, draft, pending]);

  // React Flow puts its own key handling on the document rather than on its
  // container, which is why its Shift-drag and Meta-click work wherever focus
  // happens to be. Ours hung off this surface's onKeyDown instead, so every
  // shortcut died the moment focus was anywhere else — opening the canvas
  // leaves focus on the button in the page header, and clicking the page title
  // is enough to lose it again — and Cmd-A fell through to the browser, which
  // select-alls the window and paints its highlight over the map. Same model as
  // the library now: one document listener, ignored while a field has focus.
  // Capture, not bubble: React runs its own handlers at the root container,
  // which is inside the document, so a bubble-phase listener here lands after
  // the page's Escape handler and Escape closed the whole canvas instead of
  // dropping the selection. Capture puts the canvas first, and the layered
  // Escape below still hands the key onward when it has nothing of its own left
  // to close.
  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown, true);
    return () => document.removeEventListener("keydown", handleKeyDown, true);
  }, [handleKeyDown]);

  // Focus still moves to the map when it opens, so Tab adds a box straight away
  // and a screen reader lands in the region it just asked for. The shortcuts no
  // longer depend on it.
  useEffect(() => {
    surfaceRef.current?.focus({ preventScroll: true });
  }, [isLoading]);

  // Rendering nothing here made every slow fetch look like a canvas that had
  // simply failed to arrive, with no way to tell waiting from broken.
  if (isLoading) {
    return <CanvasMessage title={t("pageCanvas.loading")} live />;
  }

  if (error) {
    const status = asPageMapApiError(error)?.status;
    if (status === 404 || status === 405) {
      return (
        <CanvasMessage
          title={t("pageCanvas.daemonOutdatedTitle")}
          body={t("pageCanvas.daemonOutdatedBody")}
        />
      );
    }
    return (
      <CanvasMessage
        title={t("pageCanvas.loadErrorTitle")}
        body={t("pageCanvas.loadErrorBody")}
        action={
          <button
            type="button"
            className="page-canvas-message-action"
            onClick={() => void refetch()}
          >
            {t("pageCanvas.retry")}
          </button>
        }
      />
    );
  }

  if (!map || map.revision === 0 || views.length === 0) {
    return (
      <CanvasMessage
        title={t("pageCanvas.emptyTitle")}
        body={t("pageCanvas.emptyBody")}
        action={
          <button
            type="button"
            className="page-canvas-message-action"
            onClick={() => improveMutation.mutate()}
            disabled={improveMutation.isPending || readOnly}
          >
            {improveMutation.isPending
              ? t("pageCanvas.generating")
              : t("pageCanvas.generate")}
          </button>
        }
      />
    );
  }

  return (
    <div className="page-canvas">
      {readOnly && (
        <p role="status" className="page-canvas-banner">
          {t("pageCanvas.readOnlyNotice")}
        </p>
      )}
      <div
        ref={surfaceRef}
        className="page-canvas-surface"
        role="region"
        // Focusable, or the shortcuts below only fire once a box has been
        // clicked — press Tab on a fresh canvas and nothing happens.
        tabIndex={0}
        aria-label={t("pageCanvas.regionLabel", { title: pageTitle })}
        onDoubleClick={handleDoubleClick}
      >
        <ReactFlow
          nodes={displayNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onNodeDragStop={handleNodeDragStop}
          onNodeDoubleClick={handleNodeDoubleClick}
          onNodeContextMenu={handleNodeContextMenu}
          onPaneContextMenu={handlePaneContextMenu}
          onPaneClick={() => {
            setMenu(null);
            setNotice(null);
          }}
          onNodeClick={() => setMenu(null)}
          onMoveStart={() => setMenu(null)}
          onConnectEnd={handleConnectEnd}
          nodesDraggable={!readOnly}
          nodesConnectable={!readOnly}
          edgesFocusable={false}
          // The Obsidian/Figma pointer model: drag empty canvas to rubber-band
          // a selection, two-finger scroll to pan, middle-drag to pan, and
          // pinch (or Cmd-scroll) to zoom.
          selectionOnDrag
          // Meta alone is the React Flow default; Shift is what someone
          // coming from Obsidian or Figma reaches for first.
          multiSelectionKeyCode={["Meta", "Control", "Shift"]}
          panOnDrag={[1]}
          panOnScroll
          // Double-click is how a box gets drawn here, so it must not also be
          // how the canvas zooms.
          zoomOnDoubleClick={false}
          // Delete is handled above, against the map's revision. Left to React
          // Flow it would drop the box from the canvas locally and leave the
          // daemon's copy untouched.
          deleteKeyCode={null}
          fitView
          fitViewOptions={OPENING_VIEW}
          // Below React Flow's default floor of 0.5, because the legibility
          // floor belongs to the opening view above and nowhere else. Left at
          // 0.5, a 28-box map answered "fit all" by clamping there and leaving
          // ten boxes outside the frame — a command that says it shows
          // everything has to show everything, however small.
          minZoom={0.25}
          // Where the zoom keys are discovered. React Flow puts these strings on
          // both the `title` and the `aria-label` of its own zoom buttons, so
          // hovering the control someone already reached for is what tells them
          // there is a key for it — no second surface to explain the first.
          ariaLabelConfig={{
            "controls.zoomIn.ariaLabel": `${t("pageCanvas.zoomIn")} (+)`,
            "controls.zoomOut.ariaLabel": `${t("pageCanvas.zoomOut")} (-)`,
            "controls.fitView.ariaLabel": `${t("pageCanvas.fitView")} (Shift 1)`,
          }}
          proOptions={{ hideAttribution: true }}
        >
          <Background color={palette.graticule} gap={24} />
          {/* React Flow's fit button calls `fitView` with the options passed
              here, not the ones on <ReactFlow>, so the button has to be handed
              the same object as the key or the two disagree. */}
          <Controls showInteractive={false} fitViewOptions={FIT_ALL_VIEW} />
        </ReactFlow>
        {menu && menuItems.length > 0 && (
          <ContextMenu
            x={menu.x}
            y={menu.y}
            items={menuItems}
            onClose={() => setMenu(null)}
          />
        )}
        {readOnly && (
          <span className="page-canvas-badge" title={t("pageCanvas.readOnlyNotice")}>
            {t("pageCanvas.readOnlyBadge")}
          </span>
        )}
        <div className="page-canvas-actions">
          {/* Improve is the only act here now. Adding a box had a button too,
              which duplicated three gestures that are already the fastest way to
              do it — double-click the canvas, Tab for a child, Enter for a
              sibling — and an empty page offers Generate in its own right, so
              nothing is only reachable by keyboard. */}
          <div className="page-canvas-action-cluster">
            <button
              type="button"
              className="page-canvas-action is-improve"
              onClick={() => improveMutation.mutate()}
              disabled={improveMutation.isPending || readOnly}
            >
              <span
                aria-hidden="true"
                className={
                  improveMutation.isPending
                    ? "page-canvas-action-glyph is-spinning"
                    : "page-canvas-action-glyph"
                }
              >
                {improveMutation.isPending ? "\u25CC" : "\u2726"}
              </span>
              {improveMutation.isPending
                ? t("pageCanvas.improving")
                : t("pageCanvas.improve")}
            </button>
          </div>
          {hiddenSuggestions > 0 && (
            <button
              type="button"
              className="page-canvas-reveal"
              onClick={() => setShowSuggestions(true)}
            >
              <span aria-hidden="true" className="page-canvas-reveal-dot">
                {"\u25CF"}
              </span>
              {t("pageCanvas.showSuggestions", { count: hiddenSuggestions })}
            </button>
          )}
          {notice && (
            <span role="status" aria-live="polite" className="page-canvas-notice">
              {notice}
            </span>
          )}
        </div>
        {!readOnly && <CanvasHelp open={helpOpen} setOpen={setHelpOpen} />}
      </div>
    </div>
  );
}

function ContextMenu({
  x,
  y,
  items,
  onClose,
}: {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}) {
  const { t } = useTranslation();
  return (
    <ul
      className="page-canvas-menu"
      role="menu"
      aria-label={t("pageCanvas.menuLabel")}
      style={{ left: x, top: y }}
      // The menu sits inside the canvas surface, so a click on it would
      // otherwise reach the pane underneath and fire the very handlers that
      // close it.
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item, i) => (
        <li key={item.key} role="none">
          <button
            type="button"
            role="menuitem"
            className={[
              "page-canvas-menu-item",
              item.danger ? "is-danger" : "",
              // A rule above the destructive verb, so the pointer travelling
              // down the list crosses a line before it reaches the one item
              // that takes boxes away and cannot be undone.
              item.danger && !items[i - 1]?.danger ? "is-separated" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => {
              item.run();
              onClose();
            }}
          >
            {item.label}
          </button>
        </li>
      ))}
    </ul>
  );
}

function CanvasHelp({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: Dispatch<SetStateAction<boolean>>;
}) {
  const { t } = useTranslation();
  const buttonRef = useRef<HTMLButtonElement | null>(null);

  return (
    <div
      className="page-canvas-help"
      // Escape is a canvas shortcut too (it clears the selection), so an open
      // panel has to swallow its own dismissal before the surface sees it.
      onKeyDown={(e) => {
        if (e.key !== "Escape" || !open) return;
        e.preventDefault();
        e.stopPropagation();
        setOpen(false);
        buttonRef.current?.focus();
      }}
    >
      {open && (
        <div
          id={HELP_PANEL_ID}
          className="page-canvas-help-panel"
          role="note"
          aria-label={t("pageCanvas.hintsLabel")}
        >
          {SHORTCUTS.map((s) => (
            <div key={s.cap} className="page-canvas-help-row">
              <kbd>{s.cap}</kbd>
              <span>{t(s.key)}</span>
            </div>
          ))}
          <p className="page-canvas-help-note">{t("pageCanvas.hintSectionNote")}</p>
        </div>
      )}
      <button
        ref={buttonRef}
        type="button"
        className="page-canvas-help-button"
        aria-label={t("pageCanvas.hintsLabel")}
        aria-expanded={open}
        aria-controls={open ? HELP_PANEL_ID : undefined}
        onClick={() => setOpen((v) => !v)}
      >
        ?
      </button>
    </div>
  );
}

/**
 * Waiting, empty, and failed — all three inside the frame the map would fill.
 *
 * Returned as bare text, they collapsed the page to a few lines floating where a
 * 620px canvas had been, which reads as a broken screen rather than as a canvas
 * that has not arrived.
 */
function CanvasMessage({
  title,
  body,
  action,
  live,
}: {
  title: string;
  body?: string;
  action?: ReactNode;
  /** Announce it: true while waiting, so the wait is not silent. */
  live?: boolean;
}) {
  return (
    <div className="page-canvas">
      <div
        className="page-canvas-surface is-message"
        role={live ? "status" : undefined}
        aria-live={live ? "polite" : undefined}
      >
        <div className="page-canvas-message">
          <p className="page-canvas-message-title">{title}</p>
          {body && <p className="page-canvas-message-body">{body}</p>}
          {action}
        </div>
      </div>
    </div>
  );
}
