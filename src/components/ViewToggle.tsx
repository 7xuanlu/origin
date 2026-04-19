// SPDX-License-Identifier: AGPL-3.0-only
interface ViewToggleProps {
  active: "home" | "activity";
  onSwitch: (view: "home" | "activity") => void;
}

/**
 * Segmented control for switching between Home and Activity views.
 */
export default function ViewToggle({ active, onSwitch }: ViewToggleProps) {
  const btnStyle = (isActive: boolean): React.CSSProperties => ({
    fontFamily: "var(--mem-font-body)",
    fontSize: "12px",
    fontWeight: 500,
    minWidth: 56,
    textAlign: "center",
    color: isActive ? "var(--mem-text)" : "var(--mem-text-tertiary)",
    backgroundColor: isActive ? "var(--mem-surface)" : "transparent",
    boxShadow: isActive ? "0 1px 2px rgba(0,0,0,0.15)" : "none",
  });

  return (
    <div
      className="inline-flex items-center rounded-md p-0.5"
      style={{
        backgroundColor: "var(--mem-hover)",
        border: "1px solid var(--mem-border)",
      }}
    >
      <button
        onClick={() => onSwitch("home")}
        className="px-2.5 py-1 rounded transition-colors duration-150"
        style={btnStyle(active === "home")}
      >
        Home
      </button>
      <button
        onClick={() => onSwitch("activity")}
        className="px-2.5 py-1 rounded transition-colors duration-150"
        style={btnStyle(active === "activity")}
      >
        Activity
      </button>
    </div>
  );
}
