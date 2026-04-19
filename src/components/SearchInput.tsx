// SPDX-License-Identifier: AGPL-3.0-only
import { useRef, useEffect } from "react";
import originIcon from "../assets/icon.png";

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  isLoading: boolean;
  onOpenMemory: () => void;
}

export default function SearchInput({
  value,
  onChange,
  isLoading,
  onOpenMemory,
}: SearchInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div style={{ paddingLeft: 18, paddingRight: 18, WebkitAppRegion: 'drag' } as React.CSSProperties} className="flex items-center gap-4 h-[58px] bg-[var(--bg-secondary)]">
      <svg
        className="w-[20px] h-[20px] text-[var(--text-tertiary)] shrink-0"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.8}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search..."
        className="flex-1 bg-transparent text-[var(--text-primary)] text-[17px] outline-none placeholder:text-[var(--text-tertiary)] font-light tracking-[-0.01em]"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
        spellCheck={false}
        autoComplete="off"
      />
      {isLoading && (
        <div className="w-4 h-4 border-2 border-[var(--text-tertiary)]/30 border-t-[var(--accent)] rounded-full animate-spin" />
      )}
      <button
        onClick={onOpenMemory}
        className="shrink-0 hover:opacity-80 transition-opacity duration-150"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <img src={originIcon} alt="Origin" className="w-5 h-5 rounded" />
      </button>
    </div>
  );
}
