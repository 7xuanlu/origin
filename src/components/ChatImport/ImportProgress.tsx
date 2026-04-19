export type ImportStage = "parsing" | "stage_a" | "stage_b" | "done" | "error";

export interface EntityCounts {
  people: number;
  projects: number;
  concepts: number;
  decisions: number;
  tools: number;
}

interface ImportProgressProps {
  stage: ImportStage;
  memoriesProcessed: number;
  memoriesTotal: number;
  entityCounts: EntityCounts;
  conceptsWritten: number;
  conceptsTotal: number;
  conceptTitles: string[];
  errorMessage?: string;
}

export function ImportProgress({
  stage,
  memoriesProcessed,
  memoriesTotal,
  entityCounts,
  conceptsWritten,
  conceptsTotal,
  conceptTitles,
  errorMessage,
}: ImportProgressProps) {
  if (stage === "error") {
    return (
      <div className="p-6 bg-red-50 border border-red-300 rounded-lg">
        <h2 className="text-lg font-semibold text-red-800 mb-2">Import failed</h2>
        <p className="text-sm text-red-700">
          {errorMessage ?? "An unknown error occurred during import."}
        </p>
      </div>
    );
  }

  if (stage === "done") {
    return (
      <div className="p-6 bg-green-50 border border-green-300 rounded-lg">
        <h2 className="text-lg font-semibold text-green-800 mb-2">
          Your concepts are ready to explore
        </h2>
        <p className="text-sm text-green-700">
          Compiled {conceptsTotal} concepts from {memoriesTotal.toLocaleString()} memories.
        </p>
      </div>
    );
  }

  const pct = memoriesTotal > 0 ? Math.floor((memoriesProcessed / memoriesTotal) * 100) : 0;

  if (stage === "stage_a" || stage === "parsing") {
    return (
      <div className="p-6 border rounded-lg">
        <h2 className="text-lg font-semibold mb-2">Reading your conversations</h2>
        <div className="mb-4">
          <div className="h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-1">
            {memoriesProcessed.toLocaleString()} / {memoriesTotal.toLocaleString()} memories
          </p>
        </div>
        <p className="text-sm text-gray-700">
          So far: {entityCounts.people} people · {entityCounts.projects} projects ·{" "}
          {entityCounts.concepts} concepts · {entityCounts.decisions} decisions ·{" "}
          {entityCounts.tools} tools
        </p>
      </div>
    );
  }

  // stage_b
  const conceptPct = conceptsTotal > 0 ? Math.floor((conceptsWritten / conceptsTotal) * 100) : 0;
  return (
    <div className="p-6 border rounded-lg">
      <p className="text-sm text-gray-700 mb-2">
        &#x2713; Read {memoriesTotal.toLocaleString()} conversations
      </p>
      <h2 className="text-lg font-semibold mb-2">Compiling your concepts</h2>
      <div className="mb-4">
        <div className="h-2 bg-gray-200 rounded overflow-hidden">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${conceptPct}%` }}
          />
        </div>
        <p className="text-sm text-gray-600 mt-1">
          {conceptsWritten} / {conceptsTotal} concepts
        </p>
      </div>
      {conceptTitles.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium mb-1">Just compiled:</p>
          <ul className="text-sm text-blue-700">
            {conceptTitles.slice(0, 5).map((t) => (
              <li key={t}>{"\u2192"} {t}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
