import { useState, useRef, useEffect, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { importMemories, clipboardWrite, type ImportResult } from "../../lib/tauri";

type Source = "chatgpt" | "claude" | "other";

interface ImportViewProps {
  onBack: () => void;
  onComplete: (source: string, result: ImportResult) => void;
  /** When true, show "Continue" instead of "View memories" / "Import more" in summary. */
  wizardMode?: boolean;
  /** Called when internal phase changes. */
  onPhaseChange?: (phase: Phase) => void;
  /** When provided, shows a skip link in the footer. */
  onSkip?: () => void;
  /** Optional onboarding copy rendered above the import form. */
  wizardHint?: React.ReactNode;
}

type Phase = "input" | "progress" | "summary";

const PROGRESS_STEPS = [
  { label: "Parsing memories", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2", duration: 1500 },
  { label: "Generating embeddings", icon: "M13 10V3L4 14h7v7l9-11h-7z", duration: 4000 },
  { label: "Storing memories", icon: "M5 3v18l7-3 7 3V3l-7 3-7-3z", duration: 2000 },
];

function ImportProgress({ memoryCount }: { memoryCount: number }) {
  const [activeStep, setActiveStep] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setElapsed((e) => e + 100), 100);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Advance steps based on cumulative duration
    let cumulative = 0;
    for (let i = 0; i < PROGRESS_STEPS.length; i++) {
      cumulative += PROGRESS_STEPS[i].duration;
      if (elapsed < cumulative) {
        setActiveStep(i);
        return;
      }
    }
    setActiveStep(PROGRESS_STEPS.length - 1);
  }, [elapsed]);

  // Progress within current step
  let cumulativeBefore = 0;
  for (let i = 0; i < activeStep; i++) cumulativeBefore += PROGRESS_STEPS[i].duration;
  const stepElapsed = elapsed - cumulativeBefore;
  const stepProgress = Math.min(stepElapsed / PROGRESS_STEPS[activeStep].duration, 0.95);
  const totalDuration = PROGRESS_STEPS.reduce((s, p) => s + p.duration, 0);
  const overallProgress = Math.min(elapsed / totalDuration, 0.95);

  return (
    <div className="flex flex-col items-center max-w-md mx-auto py-16" style={{ gap: "32px" }}>
      {/* Count + source */}
      <div className="text-center" style={{ gap: "8px", display: "flex", flexDirection: "column" }}>
        <p style={{
          fontFamily: "var(--mem-font-heading)",
          fontSize: "28px",
          fontWeight: 400,
          color: "var(--mem-text)",
          letterSpacing: "-0.02em",
        }}>
          {memoryCount} memories
        </p>
        <p style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-tertiary)",
        }}>
          Processing your memories...
        </p>
      </div>

      {/* Progress bar */}
      <div style={{
        width: "100%",
        height: "3px",
        borderRadius: "2px",
        backgroundColor: "var(--mem-border)",
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          borderRadius: "2px",
          backgroundColor: "var(--mem-accent-indigo)",
          width: `${overallProgress * 100}%`,
          transition: "width 0.3s ease-out",
        }} />
      </div>

      {/* Steps */}
      <div style={{ width: "100%", display: "flex", flexDirection: "column", gap: "4px" }}>
        {PROGRESS_STEPS.map((step, i) => {
          const isActive = i === activeStep;
          const isDone = i < activeStep;

          return (
            <div
              key={step.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "14px",
                padding: "10px 14px",
                borderRadius: "10px",
                backgroundColor: isActive ? "var(--mem-surface)" : "transparent",
                border: isActive ? "1px solid var(--mem-border)" : "1px solid transparent",
                transition: "all 0.4s ease",
                opacity: isDone ? 0.5 : isActive ? 1 : 0.35,
              }}
            >
              {/* Icon */}
              <div style={{
                width: "32px",
                height: "32px",
                borderRadius: "8px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: isDone
                  ? "rgba(123, 123, 232, 0.15)"
                  : isActive
                    ? "rgba(123, 123, 232, 0.1)"
                    : "transparent",
                transition: "all 0.4s ease",
                flexShrink: 0,
              }}>
                {isDone ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke="var(--mem-accent-indigo)" strokeWidth="2.5"
                    strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                    stroke={isActive ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)"}
                    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                    style={isActive ? { animation: "pulse-subtle 2s ease-in-out infinite" } : undefined}>
                    <path d={step.icon} />
                  </svg>
                )}
              </div>

              {/* Label */}
              <span style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                fontWeight: isActive ? 500 : 400,
                color: isDone ? "var(--mem-text-secondary)" : isActive ? "var(--mem-text)" : "var(--mem-text-tertiary)",
                transition: "all 0.4s ease",
              }}>
                {step.label}
                {isDone && (
                  <span style={{ color: "var(--mem-text-tertiary)", fontWeight: 400, marginLeft: "6px" }}>
                    done
                  </span>
                )}
              </span>

              {/* Step progress for active */}
              {isActive && (
                <div style={{
                  marginLeft: "auto",
                  width: "48px",
                  height: "3px",
                  borderRadius: "2px",
                  backgroundColor: "var(--mem-border)",
                  overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%",
                    borderRadius: "2px",
                    backgroundColor: "var(--mem-accent-indigo)",
                    width: `${stepProgress * 100}%`,
                    transition: "width 0.3s ease-out",
                  }} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* CSS animation */}
      <style>{`
        @keyframes pulse-subtle {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

const EXPORT_PROMPT = `Export all of my stored memories and any context you've learned about me. Preserve my words verbatim where possible.

EVERY line MUST follow this exact format — no exceptions:
[TYPE] - content

TYPE must be exactly one of: identity, preference, decision, fact, goal

Where:
- identity = who I am (name, location, education, family, languages)
- preference = how I like things (opinions, tastes, working style, rules like "always do X" or "never do Y")
- decision = choices I made with rationale (tech, career, project directions)
- fact = things about my work, projects, skills, situation
- goal = what I'm working toward, aspirations

Example output:
[identity] - Lives in San Francisco, originally from Taiwan
[preference] - Prefers concise responses without trailing summaries
[decision] - Chose Rust + Tauri for the desktop app over Electron
[preference] - Never use emojis unless explicitly asked
[goal] - Launch Origin memory app by end of Q1
[fact] - Building a local-first AI memory layer called Origin

Rules:
- NO section headers, category labels, or grouping text
- NO explanations before or after — ONLY the tagged lines
- One memory per line
- Wrap entire output in a single code block`;

const SOURCE_LABELS: Record<Source, string> = {
  chatgpt: "ChatGPT",
  claude: "Claude",
  other: "Other",
};

// Aligned with Origin's --mem-accent-* palette (see FACET_COLORS in tauri.ts)
const TYPE_BADGE_STYLES: Record<string, { bg: string; text: string }> = {
  identity: { bg: "color-mix(in srgb, var(--mem-accent-indigo) 15%, transparent)", text: "var(--mem-accent-indigo)" },
  preference: { bg: "color-mix(in srgb, var(--mem-accent-warm) 15%, transparent)", text: "var(--mem-accent-warm)" },
  fact: { bg: "color-mix(in srgb, var(--mem-accent-glow) 12%, transparent)", text: "var(--mem-accent-glow)" },
  decision: { bg: "color-mix(in srgb, var(--mem-accent-amber) 15%, transparent)", text: "var(--mem-accent-amber)" },
  goal: { bg: "color-mix(in srgb, var(--mem-accent-sage) 15%, transparent)", text: "var(--mem-accent-sage)" },
};

export function ImportView({ onBack, onComplete, wizardMode, onPhaseChange, onSkip, wizardHint }: ImportViewProps) {
  const queryClient = useQueryClient();
  const [phase, setPhaseRaw] = useState<Phase>("input");
  const setPhase = useCallback((p: Phase) => {
    setPhaseRaw(p);
    onPhaseChange?.(p);
  }, [onPhaseChange]);
  const [source, setSource] = useState<Source>("chatgpt");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [promptCopied, setPromptCopied] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target?.result;
      if (typeof content === "string") {
        setText(content);
      }
    };
    reader.readAsText(file);
    // Reset so the same file can be re-selected
    e.target.value = "";
  };

  const handleImport = async () => {
    setError(null);
    setPhase("progress");
    try {
      const res = await importMemories(source, text);
      setResult(res);
      setPhase("summary");
      queryClient.invalidateQueries();
    } catch (err) {
      setError(String(err));
      setPhase("input");
    }
  };

  const handleReset = () => {
    setPhase("input");
    setText("");
    setError(null);
    setResult(null);
  };

  // ── Input form ──────────────────────────────────────────────────────
  if (phase === "input") {
    return (
      <div className="flex flex-col mx-auto py-4" style={{ height: "calc(100vh - 120px)", maxWidth: "672px" }}>
        {/* Header row: back + title + source pills */}
        <div className="flex items-center gap-4 mb-4 shrink-0">
          <button
            onClick={onBack}
            className="flex items-center gap-1 shrink-0 transition-colors duration-150"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              color: "var(--mem-text-secondary)",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </button>
          <h1
            className="shrink-0"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "18px",
              fontWeight: 500,
              color: "var(--mem-text)",
            }}
          >
            Import Memories
          </h1>
          <div className="flex gap-1 ml-auto shrink-0">
            {(["chatgpt", "claude", "other"] as Source[]).map((s) => (
              <button
                key={s}
                onClick={() => setSource(s)}
                className="px-2.5 py-1 rounded-md text-xs font-medium transition-colors duration-150"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  backgroundColor: source === s ? "var(--mem-accent-indigo)" : "transparent",
                  color: source === s ? "white" : "var(--mem-text-tertiary)",
                }}
              >
                {SOURCE_LABELS[s]}
              </button>
            ))}
          </div>
        </div>

        {wizardMode && wizardHint && (
          <div
            className="mb-4 rounded-lg px-3 py-2 shrink-0"
            style={{
              backgroundColor: "var(--mem-hover)",
              border: "1px solid var(--mem-border)",
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              color: "var(--mem-text-secondary)",
              lineHeight: "1.5",
            }}
          >
            {wizardHint}
          </div>
        )}

        {/* Two equal panels */}
        <div className="flex flex-col gap-3 flex-1 min-h-0">
          {/* Top panel: export prompt or instructions */}
          <div className="flex flex-col flex-1 min-h-0">
            <div className="flex items-center justify-between mb-1.5 shrink-0">
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", fontWeight: 500, color: "var(--mem-text-secondary)" }}>
                {source !== "other" ? "Export prompt" : "Instructions"}
              </span>
              {source !== "other" && (
                <button
                  onClick={() => {
                    clipboardWrite(EXPORT_PROMPT);
                    setPromptCopied(true);
                    setTimeout(() => setPromptCopied(false), 2000);
                  }}
                  className="px-2.5 py-0.5 rounded-md text-xs font-medium transition-colors"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    color: promptCopied ? "var(--mem-accent-warm)" : "var(--mem-accent-indigo)",
                    backgroundColor: promptCopied ? "rgba(251, 191, 36, 0.15)" : "rgba(123, 123, 232, 0.1)",
                  }}
                >
                  {promptCopied ? "Copied!" : "Copy prompt"}
                </button>
              )}
            </div>
            <pre
              className="rounded-lg px-3 py-2 overflow-auto text-[11px] leading-relaxed flex-1 min-h-0"
              style={{
                fontFamily: "var(--mem-font-mono)",
                backgroundColor: "var(--mem-hover)",
                color: "var(--mem-text-tertiary)",
                border: "1px solid var(--mem-border)",
                whiteSpace: "pre-wrap",
                margin: 0,
              }}
            >
              {source !== "other"
                ? `Copy this prompt, paste into ${SOURCE_LABELS[source]}, then paste the output below.\n\n${EXPORT_PROMPT}`
                : "Paste any list of facts or memories, one per line.\n\nEach line becomes a separate memory in Origin.\nEmpty lines and separators (---, ===) are skipped.\n\nOptionally prefix lines with a type tag:\n[identity] - Lives in San Francisco\n[preference] - Prefers concise responses\n[fact] - Building Origin, a local-first AI memory app\n[goal] - Launch by end of Q1\n\nLines without a tag are stored as facts."}
            </pre>
          </div>

          {/* Bottom panel: paste area */}
          <div className="flex flex-col flex-1 min-h-0">
            <div className="flex items-center justify-between mb-1.5 shrink-0">
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", fontWeight: 500, color: "var(--mem-text-secondary)" }}>
                Paste output
              </span>
              {text && (
                <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
                  {text.split("\n").filter((l) => l.trim()).length} lines
                </span>
              )}
            </div>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your memories here, one per line..."
              className="w-full rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-[var(--mem-accent-indigo)]/40 flex-1 min-h-0"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "13px",
                color: "var(--mem-text)",
                backgroundColor: "var(--mem-surface)",
                border: "1px solid var(--mem-border)",
                resize: "none",
                lineHeight: "1.6",
              }}
            />
          </div>
        </div>

        {/* Footer: upload + skip + error + import */}
        <div className="flex items-center gap-3 mt-3 shrink-0">
          <button
            onClick={() => fileRef.current?.click()}
            className="transition-colors"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "12px",
              color: "var(--mem-accent-indigo)",
            }}
          >
            Upload file
          </button>
          <input ref={fileRef} type="file" accept=".txt,.csv,.json" className="hidden" onChange={handleFileUpload} />
          {error && (
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "#ef4444" }}>
              {error}
            </span>
          )}
          <div className="flex items-center gap-3 ml-auto">
            {onSkip && (
              <button
                onClick={onSkip}
                className="transition-colors duration-150"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "13px",
                  color: "var(--mem-text-tertiary)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Skip
              </button>
            )}
            <button
              onClick={handleImport}
              disabled={!text.trim()}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                fontFamily: "var(--mem-font-body)",
                backgroundColor: "var(--mem-accent-indigo)",
                color: "white",
              }}
            >
              Import
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Progress ────────────────────────────────────────────────────────
  if (phase === "progress") {
    const lineCount = text.split("\n").filter((l) => l.trim()).length;
    return <ImportProgress memoryCount={lineCount} />;
  }

  // ── Summary ─────────────────────────────────────────────────────────
  if (phase === "summary" && result) {
    const breakdownEntries = Object.entries(result.breakdown).filter(
      ([, count]) => count > 0,
    );
    const kgTotal =
      result.entities_created +
      result.observations_added +
      result.relations_created;

    return (
      <div className="flex flex-col gap-6 max-w-2xl mx-auto py-4">
        {/* Summary card */}
        <div
          className="rounded-xl px-6 py-5"
          style={{
            backgroundColor: "var(--mem-surface)",
            border: "1px solid var(--mem-border)",
          }}
        >
          {/* Main stat */}
          <div className="flex items-center gap-3 mb-4">
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center"
              style={{ backgroundColor: "rgba(99, 102, 241, 0.15)" }}
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="rgb(99, 102, 241)"
                viewBox="0 0 24 24"
                strokeWidth="2"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <div>
              <p
                style={{
                  fontFamily: "var(--mem-font-heading)",
                  fontSize: "16px",
                  fontWeight: 500,
                  color: "var(--mem-text)",
                }}
              >
                {result.imported} memories imported from{" "}
                {SOURCE_LABELS[source]}
              </p>
              {result.skipped > 0 && (
                <p
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    color: "var(--mem-text-tertiary)",
                    marginTop: "2px",
                  }}
                >
                  {result.skipped} skipped (duplicates or empty)
                </p>
              )}
            </div>
          </div>

          {/* Type breakdown badges */}
          {breakdownEntries.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {breakdownEntries.map(([type, count]) => {
                const style = TYPE_BADGE_STYLES[type] ?? {
                  bg: "var(--mem-hover)",
                  text: "var(--mem-text-secondary)",
                };
                return (
                  <span
                    key={type}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium"
                    style={{
                      backgroundColor: style.bg,
                      color: style.text,
                      fontFamily: "var(--mem-font-body)",
                    }}
                  >
                    {type}
                    <span style={{ opacity: 0.7 }}>{count}</span>
                  </span>
                );
              })}
            </div>
          )}

          {/* KG stats */}
          {kgTotal > 0 && (
            <p
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {result.entities_created} entities, {result.observations_added}{" "}
              observations discovered
            </p>
          )}
        </div>

        {/* Actions */}
        <div className={`flex items-center gap-3 ${wizardMode ? "justify-center" : ""}`}>
          <button
            onClick={() => onComplete(source, result!)}
            className="px-6 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
            style={{
              fontFamily: "var(--mem-font-body)",
              backgroundColor: "var(--mem-accent-indigo)",
              color: "white",
            }}
          >
            {wizardMode ? "Continue" : "View memories"}
          </button>
          {!wizardMode && (
            <button
              onClick={handleReset}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
              style={{
                fontFamily: "var(--mem-font-body)",
                backgroundColor: "var(--mem-hover)",
                color: "var(--mem-text-secondary)",
                border: "1px solid var(--mem-border)",
              }}
            >
              Import more
            </button>
          )}
        </div>
      </div>
    );
  }

  return null;
}

export default ImportView;
