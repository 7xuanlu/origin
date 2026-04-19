import { useCallback, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";

interface DropZoneProps {
  /** Called when a file is drag-dropped (receives a File object). */
  onFileSelected: (file: File) => void;
  /** Called when a file is picked via the native dialog (receives a path string — no temp file needed). */
  onPathSelected?: (path: string) => void;
}

export function DropZone({ onFileSelected, onPathSelected }: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      setError(null);
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".zip")) {
        setError("Must be a .zip file exported from ChatGPT or Claude.");
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const handlePickFile = useCallback(async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{ name: "ZIP Archives", extensions: ["zip"] }],
      });
      if (!selected || typeof selected !== "string") return; // user cancelled
      const path = selected;
      setError(null);
      if (onPathSelected) {
        onPathSelected(path);
      } else {
        // Fallback: no onPathSelected handler — report as error
        setError("File picker returned a path but no handler is available.");
      }
    } catch (e) {
      setError(`Failed to open file picker: ${e}`);
    }
  }, [onPathSelected]);

  return (
    <div
      data-testid="chat-import-drop-zone"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        border: `1.5px dashed ${isDragging ? "var(--mem-accent-indigo)" : "var(--mem-border)"}`,
        borderRadius: "10px",
        padding: "28px 20px",
        textAlign: "center",
        transition: "all 0.2s ease",
        background: isDragging ? "var(--mem-indigo-bg)" : "transparent",
        cursor: "default",
      }}
    >
      {/* Upload icon */}
      <div
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 36,
          height: 36,
          borderRadius: 8,
          background: isDragging ? "var(--mem-accent-indigo)" : "var(--mem-hover-strong)",
          color: isDragging ? "white" : "var(--mem-text-tertiary)",
          transition: "all 0.2s ease",
          marginBottom: 12,
        }}
      >
        <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
        </svg>
      </div>

      <div
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          fontWeight: 500,
          color: "var(--mem-text)",
          marginBottom: 4,
        }}
      >
        Drop export ZIP here
      </div>

      <div
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "11px",
          color: "var(--mem-text-tertiary)",
          marginBottom: 14,
        }}
      >
        .zip file from ChatGPT or Claude export
      </div>

      <button
        onClick={handlePickFile}
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "12px",
          fontWeight: 500,
          padding: "6px 14px",
          borderRadius: "6px",
          backgroundColor: "var(--mem-accent-indigo)",
          color: "white",
          cursor: "pointer",
          transition: "opacity 0.15s ease",
          border: "none",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.85")}
        onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
      >
        Choose file
      </button>

      {error && (
        <p
          role="alert"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "11px",
            color: "#ef4444",
            marginTop: 12,
            lineHeight: "1.5",
          }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
