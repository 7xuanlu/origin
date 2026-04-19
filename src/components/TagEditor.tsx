// SPDX-License-Identifier: AGPL-3.0-only
import { useState, useEffect, useRef } from "react";
import { setDocumentTags, suggestTags } from "../lib/tauri";

interface TagEditorProps {
  source: string;
  sourceId: string;
  lastModified: number;
  currentTags: string[];
  allTags: string[];
  onClose: () => void;
  onTagsChanged: () => void;
}

export default function TagEditor({
  source,
  sourceId,
  lastModified,
  currentTags,
  allTags,
  onClose,
  onTagsChanged,
}: TagEditorProps) {
  const [input, setInput] = useState("");
  const [tags, setTags] = useState<string[]>(currentTags);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Fetch suggestions on mount
  useEffect(() => {
    suggestTags(source, sourceId, lastModified).then(setSuggestions).catch(() => {});
  }, [source, sourceId, lastModified]);

  // Close on click-outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    // Delay listener so the opening click doesn't immediately close
    const id = setTimeout(() => document.addEventListener("mousedown", handleClick), 0);
    return () => {
      clearTimeout(id);
      document.removeEventListener("mousedown", handleClick);
    };
  }, [onClose]);

  // Close on Escape
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function addTag(tag: string) {
    const normalized = tag.trim().toLowerCase();
    if (!normalized || tags.includes(normalized)) return;
    const next = [...tags, normalized];
    setTags(next);
    setInput("");
    await setDocumentTags(source, sourceId, next);
    onTagsChanged();
  }

  async function removeTag(tag: string) {
    const next = tags.filter((t) => t !== tag);
    setTags(next);
    await setDocumentTags(source, sourceId, next);
    onTagsChanged();
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && input.trim()) {
      e.preventDefault();
      addTag(input);
    }
  }

  const query = input.trim().toLowerCase();

  // Filter suggestions: not already assigned, match query
  const filteredSuggestions = suggestions.filter(
    (s) => !tags.includes(s) && (!query || s.includes(query)),
  );

  // Filter all tags: not already assigned, not in suggestions, match query
  const filteredLibrary = allTags.filter(
    (t) => !tags.includes(t) && !suggestions.includes(t) && (!query || t.includes(query)),
  );

  // Show "create" option if query doesn't match any existing tag
  const showCreate =
    query &&
    !tags.includes(query) &&
    !allTags.includes(query) &&
    !suggestions.includes(query);

  return (
    <div
      ref={containerRef}
      className="absolute left-0 right-0 top-full mt-2 z-50 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.4)] p-3 space-y-2.5"
    >
      {/* Current tags */}
      {tags.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-[var(--accent)]/15 text-[var(--accent)]"
            >
              {tag}
              <button
                onClick={() => removeTag(tag)}
                className="hover:text-white transition-colors"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Input */}
      <input
        ref={inputRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Add a tag..."
        className="w-full text-[12px] bg-[var(--overlay-subtle)] border border-[var(--border)] rounded-lg px-2.5 py-1.5 text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] outline-none focus:border-[var(--accent)]/50"
      />

      {/* Dropdown options */}
      <div className="max-h-[140px] overflow-y-auto space-y-1">
        {/* Create new */}
        {showCreate && (
          <button
            onClick={() => addTag(query)}
            className="w-full text-left text-[11px] px-2 py-1.5 rounded-lg hover:bg-[var(--overlay-hover)] text-[var(--text-primary)] transition-colors"
          >
            Create &quot;{query}&quot;
          </button>
        )}

        {/* Suggested */}
        {filteredSuggestions.length > 0 && (
          <>
            <div className="text-[10px] font-medium text-[var(--text-tertiary)] uppercase tracking-wider px-2 pt-1">
              Suggested
            </div>
            {filteredSuggestions.map((s) => (
              <button
                key={s}
                onClick={() => addTag(s)}
                className="w-full text-left text-[11px] px-2 py-1.5 rounded-lg hover:bg-[var(--overlay-hover)] text-[var(--text-secondary)] transition-colors"
              >
                {s}
              </button>
            ))}
          </>
        )}

        {/* Library */}
        {filteredLibrary.length > 0 && (
          <>
            <div className="text-[10px] font-medium text-[var(--text-tertiary)] uppercase tracking-wider px-2 pt-1">
              Tags
            </div>
            {filteredLibrary.map((t) => (
              <button
                key={t}
                onClick={() => addTag(t)}
                className="w-full text-left text-[11px] px-2 py-1.5 rounded-lg hover:bg-[var(--overlay-hover)] text-[var(--text-secondary)] transition-colors"
              >
                {t}
              </button>
            ))}
          </>
        )}
      </div>
    </div>
  );
}
