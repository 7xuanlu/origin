// SPDX-License-Identifier: AGPL-3.0-only
export function GhostConceptsRow() {
  return (
    <div>
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-tertiary)",
          margin: "0 0 10px 0",
        }}
      >
        Concepts will appear here as Origin finds patterns.
      </p>
      <div
        className="flex gap-3 pb-2"
        style={{ overflowX: "auto", scrollbarWidth: "none" }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            data-ghost-card
            className="rounded-xl shrink-0"
            style={{
              width: "280px",
              height: "110px",
              border: "1px solid var(--mem-border)",
              opacity: 0.4,
              backgroundColor: "transparent",
            }}
          />
        ))}
      </div>
    </div>
  );
}
