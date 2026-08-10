// SPDX-License-Identifier: AGPL-3.0-only
import { memo, useEffect, useRef } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { useTranslation } from "react-i18next";
import {
  compositeOver,
  type GraphPalette,
  type GraphSlot,
} from "../../../lib/graph/palette";
import type { PageMapRefKind, PageMapStatus } from "../../../lib/tauri";

// Page-map ref kinds are not entity types, so they get their own mapping onto
// the validated 5-slot Atlas palette rather than going through
// slotForEntityType. Same inks, same theme reactivity, no new colors.
const SLOT_BY_REF_KIND: Record<PageMapRefKind, GraphSlot> = {
  page: "project",
  memory: "neutral",
  entity: "concept",
  section: "tool",
};

export interface CanvasNodeData extends Record<string, unknown> {
  label: string;
  refKind: PageMapRefKind;
  status: PageMapStatus;
  dangling: boolean;
  isRoot: boolean;
  /** Rings out from the page: 0 is the page, 1 its sections, 2+ their insides. */
  depth: number;
  /**
   * Direction away from the page at the middle, radians, screen axes. The grow
   * dot rides this side of the box.
   */
  outward: number;
  readOnly: boolean;
  palette: GraphPalette;
  width: number;
  height: number;
  /**
   * Show the name field instead of the label. True for the box being renamed
   * and for the unsaved draft box.
   */
  editing: boolean;
  /** Prompt for the name field; the draft box has no label to show yet. */
  placeholder?: string;
  onOpen: () => void;
  onAccept: () => void;
  onDismiss: () => void;
  onCommit: (label: string) => void;
  onCancel: () => void;
}

export type CanvasNodeType = Node<CanvasNodeData, "pageMapNode">;

// Both handles sit at the box center with zero opacity: a mind map reads as
// center-to-center spokes, and edge-anchored handles would kink every spoke
// on a radial layout.
const CENTER_HANDLE = {
  left: "50%",
  top: "50%",
  transform: "translate(-50%, -50%)",
  opacity: 0,
  pointerEvents: "none",
} as const;

// Which side of the box the grow dot sits on, from the direction the box faces.
// Quadrants clockwise from three o'clock, because screen y grows downward.
const SIDES = [Position.Right, Position.Bottom, Position.Left, Position.Top];

/**
 * A radial map grows outward, so the dot you grab belongs on the side facing
 * away from the page at the middle. Pinned to the bottom it pointed back at the
 * parent for every box above the center, sat between a box and its own children
 * for every box below it, and told everyone the map was a downward tree.
 */
function growSide(outward: number): Position {
  const quadrant = Math.round(outward / (Math.PI / 2));
  return SIDES[((quadrant % 4) + 4) % 4];
}

function CanvasNode({
  data,
  selected,
  isConnectable,
}: NodeProps<CanvasNodeType>) {
  const { t } = useTranslation();
  const {
    label,
    refKind,
    status,
    dangling,
    isRoot,
    depth,
    outward,
    readOnly,
    palette,
    width,
    height,
    editing,
    placeholder,
  } = data;

  const slotColor = palette[SLOT_BY_REF_KIND[refKind]];
  const suggested = status === "suggested";
  const showControls = suggested && !isRoot && !readOnly && !editing;

  // A mind map is read by its hierarchy, and until this ladder existed a leaf
  // was indistinguishable from a section: same fill, same border, same weight.
  // Three rungs — the page, its sections, what is inside them — spend no new
  // color, only weight, so the slot ink still means what it means everywhere
  // else in the app.
  const branch = depth <= 1;
  const fill = compositeOver(
    slotColor,
    palette.surface,
    isRoot ? 0.22 : branch ? 0.1 : 0.035,
  );

  return (
    <div
      style={{
        width,
        height,
        display: "flex",
        alignItems: "center",
        gap: 6,
        boxSizing: "border-box",
        padding: "0 10px",
        borderRadius: 10,
        border: `${isRoot ? 1.5 : 1}px ${suggested ? "dashed" : "solid"} ${
          branch ? slotColor : compositeOver(slotColor, palette.surface, 0.5)
        }`,
        backgroundColor: fill,
        opacity: dangling ? 0.45 : suggested ? 0.75 : 1,
        // Selection has to be visible or the keyboard shortcuts have no
        // referent — "Delete removes the selected box" is meaningless if
        // nothing on screen says which box that is.
        boxShadow: selected ? `0 0 0 2px ${slotColor}` : undefined,
      }}
    >
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={false}
        style={CENTER_HANDLE}
      />
      {editing ? (
        <NodeNameInput
          value={label}
          placeholder={placeholder}
          color={palette.label}
          onCommit={data.onCommit}
          onCancel={data.onCancel}
        />
      ) : (
        <button
          type="button"
          onClick={data.onOpen}
          title={dangling ? t("pageCanvas.dangling") : label}
          aria-label={
            dangling ? t("pageCanvas.danglingNode", { label }) : label
          }
          style={{
            flex: 1,
            minWidth: 0,
            textAlign: "left",
            background: "none",
            border: "none",
            // The box's selection ring is this canvas's focus indicator, and
            // Tab is bound to "add box inside" so nothing ever lands here by
            // keyboard. Left on, the browser drew a second ring tight around
            // the label the moment a shortcut was pressed after a click —
            // two rings in two colors, the inner one looking like a text field
            // the name could be typed into.
            outline: "none",
            padding: 0,
            cursor: "pointer",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: dangling ? palette.labelMuted : palette.label,
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            fontWeight: isRoot ? 600 : branch ? 500 : 400,
          }}
        >
          {label}
        </button>
      )}
      {showControls && (
        <>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              data.onAccept();
            }}
            aria-label={t("pageCanvas.accept")}
            title={t("pageCanvas.accept")}
            style={controlStyle(palette.label)}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" aria-hidden="true">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              data.onDismiss();
            }}
            aria-label={t("pageCanvas.dismiss")}
            title={t("pageCanvas.dismiss")}
            style={controlStyle(palette.labelMuted)}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" aria-hidden="true">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </>
      )}
      <Handle
        id="anchor"
        type="source"
        position={Position.Bottom}
        isConnectable={false}
        style={CENTER_HANDLE}
      />
      {/* The dot people actually grab, kept separate from the centered anchor
          above so edges keep routing center-to-center across the spine. */}
      <Handle
        id="grow"
        type="source"
        position={growSide(outward)}
        className="page-canvas-grow"
        isConnectable={isConnectable}
      />
    </div>
  );
}

/**
 * The name field, mounted only while a box is being named.
 *
 * Blur is the single commit path — Enter and clicking away both route through
 * it — so one name can never fire two creates. Escape records the intent, then
 * blurs. Because the field mounts fresh for each edit, `cancelled` starts false
 * on its own and nothing has to remember to reset it.
 */
function NodeNameInput({
  value,
  placeholder,
  color,
  onCommit,
  onCancel,
}: {
  value: string;
  placeholder?: string;
  color: string;
  onCommit: (label: string) => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const cancelled = useRef(false);
  const ref = useRef<HTMLInputElement>(null);

  // React Flow mounts a freshly added node with visibility:hidden while it
  // measures it, and a field that isn't visible cannot take focus — so React's
  // autoFocus lands on nothing and every keystroke goes to the canvas instead.
  // Ask again each frame until it sticks.
  useEffect(() => {
    let frame = 0;
    let tries = 0;
    const grab = () => {
      const el = ref.current;
      if (!el || document.activeElement === el) return;
      el.focus();
      el.select();
      if (document.activeElement !== el && ++tries < 30) {
        frame = requestAnimationFrame(grab);
      }
    };
    grab();
    return () => cancelAnimationFrame(frame);
  }, []);

  return (
    <input
      ref={ref}
      // Without these, React Flow drags the box and pans the canvas while the
      // user is trying to type into it.
      className="nodrag nopan nowheel"
      defaultValue={value}
      placeholder={placeholder}
      aria-label={t("pageCanvas.nameSectionLabel")}
      onKeyDown={(e) => {
        // Canvas shortcuts must not fire while a name is being typed.
        e.stopPropagation();
        if (e.key === "Enter") {
          e.preventDefault();
          e.currentTarget.blur();
        } else if (e.key === "Escape") {
          e.preventDefault();
          cancelled.current = true;
          e.currentTarget.blur();
        }
      }}
      onBlur={(e) =>
        cancelled.current ? onCancel() : onCommit(e.currentTarget.value)
      }
      style={{
        flex: 1,
        minWidth: 0,
        background: "none",
        border: "none",
        outline: "none",
        padding: 0,
        color,
        fontFamily: "var(--mem-font-body)",
        fontSize: 12,
      }}
    />
  );
}

function controlStyle(color: string): React.CSSProperties {
  return {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    width: 18,
    height: 18,
    borderRadius: 5,
    border: "none",
    background: "none",
    cursor: "pointer",
    color,
  };
}

export default memo(CanvasNode);
