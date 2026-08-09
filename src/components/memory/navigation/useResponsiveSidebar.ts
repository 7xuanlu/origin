import { useEffect, useRef, useState, type RefObject } from "react";

const NARROW_QUERY = "(max-width: 899px)";

function initialNarrowViewport(): boolean {
  if (typeof window.matchMedia !== "function") return false;
  return window.matchMedia(NARROW_QUERY).matches;
}

export type ResponsiveSidebar = {
  readonly close: () => void;
  readonly collapsed: boolean;
  readonly isNarrow: boolean;
  readonly open: boolean;
  readonly presentation: "desktop" | "overlay";
  readonly toggle: () => void;
};

export function useResponsiveSidebar(
  desktopCollapsed: boolean,
  toggleDesktop: () => void,
  toggleRef: RefObject<HTMLButtonElement | null>,
): ResponsiveSidebar {
  const [isNarrow, setIsNarrow] = useState(initialNarrowViewport);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const drawerOpenRef = useRef(drawerOpen);

  useEffect(() => {
    drawerOpenRef.current = drawerOpen;
  }, [drawerOpen]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const media = window.matchMedia(NARROW_QUERY);
    const onChange = (event: MediaQueryListEvent) => {
      setIsNarrow(event.matches);
      if (!event.matches) {
        setDrawerOpen(false);
        if (desktopCollapsed && drawerOpenRef.current) toggleRef.current?.focus();
      }
    };
    setIsNarrow(media.matches);
    media.addEventListener?.("change", onChange);
    return () => media.removeEventListener?.("change", onChange);
  }, [desktopCollapsed, toggleRef]);

  const close = () => {
    setDrawerOpen(false);
    toggleRef.current?.focus();
  };

  useEffect(() => {
    if (!isNarrow || !drawerOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopImmediatePropagation();
      close();
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [drawerOpen, isNarrow]);

  return {
    close,
    collapsed: isNarrow ? !drawerOpen : desktopCollapsed,
    isNarrow,
    open: isNarrow ? drawerOpen : !desktopCollapsed,
    presentation: isNarrow ? "overlay" : "desktop",
    toggle: isNarrow ? () => setDrawerOpen((open) => !open) : toggleDesktop,
  };
}
