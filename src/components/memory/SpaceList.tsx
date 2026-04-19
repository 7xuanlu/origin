// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useRef, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listSpaces, createSpace, deleteSpace, updateSpace, reorderSpace, toggleSpaceStarred, type Space } from "../../lib/tauri";

const SPACES_STORAGE_KEY = "sidebar:spacesCollapsed";
const TTL_MS = 7 * 86_400_000;

function useSpacesCollapsed() {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      const raw = localStorage.getItem(SPACES_STORAGE_KEY);
      if (!raw) return false;
      const { value, ts } = JSON.parse(raw);
      if (Date.now() - ts > TTL_MS) {
        localStorage.removeItem(SPACES_STORAGE_KEY);
        return false;
      }
      return !!value;
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(SPACES_STORAGE_KEY, JSON.stringify({ value: collapsed, ts: Date.now() }));
    } catch {
      // localStorage full or disabled. Non-fatal.
    }
  }, [collapsed]);
  return [collapsed, setCollapsed] as const;
}

interface SpaceListProps {
  onSelectSpace: (spaceName: string | null) => void;
}

interface ContextMenu {
  x: number;
  y: number;
  spaceName: string;
}

interface DragState {
  spaceName: string;
  startY: number;
  currentY: number;
  startIndex: number;
  overIndex: number;
}

export default function SpaceList({ onSelectSpace }: SpaceListProps) {
  const queryClient = useQueryClient();
  const [collapsed, setCollapsed] = useSpacesCollapsed();
  const [showForm, setShowForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [contextMenu, setContextMenu] = useState<ContextMenu | null>(null);
  const [renamingSpace, setRenamingSpace] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [drag, setDrag] = useState<DragState | null>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const rowHeight = 32; // px per row, matches py-1.5 + font

  const { data: spaces = [] } = useQuery({
    queryKey: ["spaces"],
    queryFn: listSpaces,
    refetchInterval: 10000,
  });

  const createMutation = useMutation({
    mutationFn: () => createSpace(newName.trim()),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      setNewName("");
      setShowForm(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: ({ name, memoryAction }: { name: string; memoryAction: string }) =>
      deleteSpace(name, memoryAction),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      setContextMenu(null);
    },
  });

  const renameMutation = useMutation({
    mutationFn: ({ oldName, newName: n }: { oldName: string; newName: string }) =>
      updateSpace(oldName, n),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      setRenamingSpace(null);
    },
  });

  const reorderMutation = useMutation({
    mutationFn: ({ name, newOrder }: { name: string; newOrder: number }) =>
      reorderSpace(name, newOrder),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
    },
  });

  const starMutation = useMutation({
    mutationFn: (name: string) => toggleSpaceStarred(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      setContextMenu(null);
    },
  });

  useEffect(() => {
    if (showForm && nameInputRef.current) nameInputRef.current.focus();
  }, [showForm]);

  // Close create form on click outside
  useEffect(() => {
    if (!showForm) return;
    const handler = (e: MouseEvent) => {
      if (formRef.current && !formRef.current.contains(e.target as Node)) {
        setShowForm(false);
        setNewName("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showForm]);

  useEffect(() => {
    if (renamingSpace && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingSpace]);

  // Close context menu
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setContextMenu(null);
    };
    const handleKey = (e: KeyboardEvent) => { if (e.key === "Escape") setContextMenu(null); };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKey);
    };
  }, [contextMenu]);

  const allSpaces = [...spaces.filter((s) => !s.suggested), ...spaces.filter((s) => s.suggested)];

  // ── Drag handlers ──
  const onPointerDown = useCallback((e: React.PointerEvent, spaceName: string, index: number) => {
    // Only start drag from the grip handle
    const target = e.target as HTMLElement;
    if (!target.closest("[data-grip]")) return;

    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ spaceName, startY: e.clientY, currentY: e.clientY, startIndex: index, overIndex: index });
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!drag) return;
    const dy = e.clientY - drag.startY;
    const newOverIndex = Math.max(0, Math.min(allSpaces.length - 1, drag.startIndex + Math.round(dy / rowHeight)));
    setDrag((d) => d ? { ...d, currentY: e.clientY, overIndex: newOverIndex } : null);
  }, [drag, allSpaces.length]);

  const onPointerUp = useCallback(() => {
    if (!drag) return;
    if (drag.overIndex !== drag.startIndex) {
      const targetSpace = allSpaces[drag.overIndex];
      if (targetSpace) {
        reorderMutation.mutate({ name: drag.spaceName, newOrder: targetSpace.sort_order });
      }
    }
    setDrag(null);
  }, [drag, allSpaces, reorderMutation]);

  const handleContextMenu = (e: React.MouseEvent, spaceName: string) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, spaceName });
  };

  const saveRename = (oldName: string) => {
    const trimmed = renameValue.trim();
    if (trimmed && trimmed !== oldName) {
      renameMutation.mutate({ oldName, newName: trimmed });
    } else {
      setRenamingSpace(null);
    }
  };

  // ── Render a space row ──
  const renderSpaceRow = (s: Space, index: number, isSuggested: boolean) => {
    const isDragging = drag?.spaceName === s.name;

    // Calculate visual offset during drag
    let translateY = 0;
    if (drag && !isDragging) {
      if (drag.startIndex < drag.overIndex) {
        // Dragging down: items between start+1..over shift up
        if (index > drag.startIndex && index <= drag.overIndex) translateY = -rowHeight;
      } else if (drag.startIndex > drag.overIndex) {
        // Dragging up: items between over..start-1 shift down
        if (index >= drag.overIndex && index < drag.startIndex) translateY = rowHeight;
      }
    }

    if (renamingSpace === s.name) {
      return (
        <div key={s.id} className="flex items-center gap-1 px-1 py-0.5">
          <input
            ref={renameInputRef}
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onBlur={() => saveRename(s.name)}
            onKeyDown={(e) => {
              if (e.key === "Enter") saveRename(s.name);
              if (e.key === "Escape") setRenamingSpace(null);
            }}
            className="flex-1 min-w-0 rounded-md px-2 py-1 bg-transparent outline-none"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              color: "var(--mem-text)",
              border: "1px solid var(--mem-accent-indigo)",
            }}
          />
        </div>
      );
    }

    return (
      <div
        key={s.id}
        className="group flex items-center rounded-md transition-transform"
        style={{
          transform: isDragging
            ? `translateY(${drag!.currentY - drag!.startY}px)`
            : `translateY(${translateY}px)`,
          transition: isDragging ? "none" : "transform 150ms ease",
          zIndex: isDragging ? 50 : 1,
          opacity: isDragging ? 0.9 : 1,
          position: "relative",
        }}
        onPointerDown={(e) => onPointerDown(e, s.name, index)}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {/* Grip handle: visible on hover */}
        <div
          data-grip
          className="shrink-0 flex items-center justify-center opacity-0 group-hover:opacity-40 transition-opacity duration-150"
          style={{
            width: 16,
            cursor: "grab",
            color: "var(--mem-text-tertiary)",
          }}
        >
          <svg width="8" height="14" viewBox="0 0 8 14" fill="currentColor">
            <circle cx="2" cy="2" r="1.2" />
            <circle cx="6" cy="2" r="1.2" />
            <circle cx="2" cy="7" r="1.2" />
            <circle cx="6" cy="7" r="1.2" />
            <circle cx="2" cy="12" r="1.2" />
            <circle cx="6" cy="12" r="1.2" />
          </svg>
        </div>

        {/* Space button */}
        <button
          onClick={() => !drag && onSelectSpace(s.name)}
          onContextMenu={(e) => handleContextMenu(e, s.name)}
          className="flex-1 flex items-center justify-between px-2 py-1.5 rounded-md text-left transition-colors duration-150 hover:bg-[var(--mem-hover)]"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: isSuggested ? "var(--mem-text-tertiary)" : "var(--mem-text-secondary)",
          }}
        >
          <span className="flex items-center gap-1 capitalize truncate">
            <span className="shrink-0 w-3.5 text-center" style={{ fontSize: "10px" }}>
              {s.starred
                ? <span style={{ color: "var(--mem-accent-amber)" }}>&#9733;</span>
                : isSuggested
                  ? <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: "var(--mem-accent-indigo)" }} />
                  : null}
            </span>
            {s.name}
          </span>
          <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", opacity: 0.5 }}>
            {s.memory_count || ""}
          </span>
        </button>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-0.5 relative">
      {/* Section label: clickable to collapse */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        aria-expanded={!collapsed}
        aria-controls="space-list-items"
        className="flex items-center justify-between px-2 mb-1 rounded hover:bg-[var(--mem-hover)] transition-colors duration-150 w-full"
        style={{ background: "none", border: "none", cursor: "pointer" }}
      >
        <span
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "10px",
            fontWeight: 600,
            letterSpacing: "0.05em",
            textTransform: "uppercase" as const,
            color: "var(--mem-text-tertiary)",
          }}
        >
          Spaces
        </span>
        <div className="flex items-center gap-2">
          <span
            role="button"
            title="New space"
            onClick={(e) => { e.stopPropagation(); setShowForm(true); }}
            className="flex items-center justify-center rounded transition-colors duration-150"
            style={{ color: "var(--mem-text-tertiary)", cursor: "pointer" }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--mem-text-secondary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--mem-text-tertiary)")}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden="true">
              <path d="M6 2 L6 10 M2 6 L10 6" />
            </svg>
          </span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              transform: collapsed ? "rotate(-90deg)" : "rotate(0deg)",
              transition: "transform 150ms",
              color: "var(--mem-text-tertiary)",
            }}
            aria-hidden="true"
          >
            <polyline points="3,4.5 6,7.5 9,4.5" />
          </svg>
        </div>
      </button>

      {!collapsed && (
        <div id="space-list-items">
          {/* Draggable space list */}
          <div ref={listRef}>
            {allSpaces.map((s, i) => renderSpaceRow(s, i, s.suggested))}
          </div>

          {/* New space form: triggered by + button in heading row */}
          {showForm && (
            <form
              ref={formRef}
              onSubmit={(e) => {
                e.preventDefault();
                if (newName.trim()) createMutation.mutate();
              }}
              className="flex items-center gap-1 mt-0.5 px-1"
            >
              <input
                ref={nameInputRef}
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Name"
                className="flex-1 min-w-0 rounded-md px-2 py-1 bg-transparent outline-none"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: "var(--mem-text)",
                  border: "1px solid var(--mem-border)",
                }}
                onKeyDown={(e) => {
                  if (e.key === "Escape") { setShowForm(false); setNewName(""); }
                }}
              />
              <button
                type="submit"
                disabled={!newName.trim() || createMutation.isPending}
                className="shrink-0 px-2 py-1 rounded-md text-xs font-medium transition-colors duration-150"
                style={{
                  backgroundColor: newName.trim() ? "var(--mem-accent-indigo)" : "transparent",
                  color: newName.trim() ? "white" : "var(--mem-text-tertiary)",
                  border: newName.trim() ? "none" : "1px solid var(--mem-border)",
                }}
              >
                Add
              </button>
            </form>
          )}
        </div>
      )}

      {/* Context menu: Rename + Delete only */}
      {contextMenu && (
        <div
          ref={menuRef}
          className="fixed z-50 rounded-lg shadow-xl py-1 min-w-[130px]"
          style={{
            left: contextMenu.x,
            top: contextMenu.y,
            backgroundColor: "var(--mem-surface)",
            border: "1px solid var(--mem-border)",
            animation: "mem-fade-up 120ms ease both",
          }}
        >
          <button
            onClick={() => starMutation.mutate(contextMenu.spaceName)}
            className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-secondary)" }}
          >
            {spaces.find((s) => s.name === contextMenu.spaceName)?.starred ? "Unstar" : "Star"}
          </button>
          <button
            onClick={() => {
              setRenameValue(contextMenu.spaceName);
              setRenamingSpace(contextMenu.spaceName);
              setContextMenu(null);
            }}
            className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-[var(--mem-hover)]"
            style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-secondary)" }}
          >
            Rename
          </button>
          <div style={{ height: "1px", backgroundColor: "var(--mem-border)", margin: "2px 0" }} />
          <button
            onClick={() => deleteMutation.mutate({ name: contextMenu.spaceName, memoryAction: "unassign" })}
            className="w-full text-left px-3 py-1.5 text-xs transition-colors duration-150 hover:bg-red-500/10"
            style={{ fontFamily: "var(--mem-font-body)", color: "#ef4444" }}
          >
            Delete space
          </button>
        </div>
      )}
    </div>
  );
}
