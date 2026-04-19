// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect, useRef } from "react";

interface MemoryStatusBarProps {
  message: string | null;
}

export default function MemoryStatusBar({ message }: MemoryStatusBarProps) {
  const [visible, setVisible] = useState(false);
  const [displayText, setDisplayText] = useState("");
  const timerRef = useRef<number>(0);

  useEffect(() => {
    if (!message) {
      setVisible(false);
      return;
    }

    setVisible(true);
    setDisplayText("");

    let i = 0;
    const typeTimer = setInterval(() => {
      i++;
      setDisplayText(message.slice(0, i));
      if (i >= message.length) clearInterval(typeTimer);
    }, 30);

    timerRef.current = window.setTimeout(() => setVisible(false), 5000);

    return () => {
      clearInterval(typeTimer);
      clearTimeout(timerRef.current);
    };
  }, [message]);

  if (!visible) return null;

  return (
    <div
      className="px-4 py-2 text-center transition-opacity duration-500"
      style={{
        fontFamily: "var(--mem-font-mono)",
        fontSize: "11px",
        color: "var(--mem-text-tertiary)",
        opacity: visible ? 1 : 0,
      }}
    >
      {displayText}
    </div>
  );
}
