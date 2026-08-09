// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef } from "react";

interface Props {
  active: boolean;
  /** Fired after the wrapped content has been in view for >=1s. */
  onSeen: () => void;
  /** 'subtle' = border accent only; 'full' = glow + marker. */
  intensity?: "subtle" | "full";
  children: React.ReactNode;
}

export function MilestoneHighlight({
  active,
  onSeen,
  intensity = "full",
  children,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const seenRef = useRef(false);

  useEffect(() => {
    if (!active || seenRef.current) return;
    const el = ref.current;
    if (!el) return;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        timeout = setTimeout(() => {
          if (!seenRef.current) {
            seenRef.current = true;
            onSeen();
          }
        }, 1000);
      } else if (timeout) {
        clearTimeout(timeout);
        timeout = undefined;
      }
    });
    io.observe(el);
    return () => {
      io.disconnect();
      if (timeout) clearTimeout(timeout);
    };
  }, [active, onSeen]);

  return (
    <div
      ref={ref}
      style={{
        position: "relative",
        boxShadow:
          active && intensity === "full"
            ? "0 0 0 1px var(--mem-accent-warm), 0 0 24px rgba(251, 191, 36, 0.25)"
            : undefined,
        borderRadius: 12,
        transition: "box-shadow 1200ms ease",
      }}
    >
      {active && intensity === "full" && (
        <span
          aria-hidden
          style={{
            position: "absolute",
            top: -3,
            left: -3,
            width: 6,
            height: 6,
            borderRadius: "50%",
            backgroundColor: "var(--mem-accent-warm)",
          }}
        />
      )}
      {children}
    </div>
  );
}
