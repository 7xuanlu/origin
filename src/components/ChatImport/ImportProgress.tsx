import { useTranslation } from "react-i18next";

export type ImportStage = "parsing" | "stage_a" | "stage_b" | "done" | "error";

export interface EntityCounts {
  people: number;
  projects: number;
  pages: number;
  decisions: number;
  tools: number;
}

interface ImportProgressProps {
  stage: ImportStage;
  memoriesProcessed: number;
  memoriesTotal: number;
  entityCounts: EntityCounts;
  pagesWritten: number;
  pagesTotal: number;
  pageTitles: string[];
  errorMessage?: string;
}

export function ImportProgress({
  stage,
  memoriesProcessed,
  memoriesTotal,
  entityCounts,
  pagesWritten,
  pagesTotal,
  pageTitles,
  errorMessage,
}: ImportProgressProps) {
  const { t } = useTranslation();

  if (stage === "error") {
    return (
      <div className="p-6 bg-red-50 border border-red-300 rounded-lg">
        <h2 className="text-lg font-semibold text-red-800 mb-2">
          {t("chatImport.progress.failedTitle")}
        </h2>
        <p className="text-sm text-red-700">
          {errorMessage ?? t("chatImport.progress.unknownError")}
        </p>
      </div>
    );
  }

  if (stage === "done") {
    return (
      <div className="p-6 bg-green-50 border border-green-300 rounded-lg">
        <h2 className="text-lg font-semibold text-green-800 mb-2">
          {t("chatImport.progress.readyTitle")}
        </h2>
        <p className="text-sm text-green-700">
          {t("chatImport.progress.compiledPages", {
            pages: pagesTotal,
            memories: memoriesTotal.toLocaleString(),
          })}
        </p>
      </div>
    );
  }

  const pct = memoriesTotal > 0 ? Math.floor((memoriesProcessed / memoriesTotal) * 100) : 0;

  if (stage === "stage_a" || stage === "parsing") {
    return (
      <div className="p-6 border rounded-lg">
        <h2 className="text-lg font-semibold mb-2">
          {t("chatImport.progress.readingTitle")}
        </h2>
        <div className="mb-4">
          <div className="h-2 bg-gray-200 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-1">
            {t("chatImport.progress.memoriesProgress", {
              processed: memoriesProcessed.toLocaleString(),
              total: memoriesTotal.toLocaleString(),
            })}
          </p>
        </div>
        <p className="text-sm text-gray-700">
          {t("chatImport.progress.soFar", {
            people: entityCounts.people,
            projects: entityCounts.projects,
            pages: entityCounts.pages,
            decisions: entityCounts.decisions,
            tools: entityCounts.tools,
          })}
        </p>
      </div>
    );
  }

  // stage_b
  const pagePct = pagesTotal > 0 ? Math.floor((pagesWritten / pagesTotal) * 100) : 0;
  return (
    <div className="p-6 border rounded-lg">
      <p className="text-sm text-gray-700 mb-2">
        {t("chatImport.progress.readConversations", {
          count: memoriesTotal.toLocaleString(),
        })}
      </p>
      <h2 className="text-lg font-semibold mb-2">
        {t("chatImport.progress.compilingTitle")}
      </h2>
      <div className="mb-4">
        <div className="h-2 bg-gray-200 rounded overflow-hidden">
          <div
            className="h-full bg-green-500 transition-all"
            style={{ width: `${pagePct}%` }}
          />
        </div>
        <p className="text-sm text-gray-600 mt-1">
          {t("chatImport.progress.pagesProgress", {
            written: pagesWritten,
            total: pagesTotal,
          })}
        </p>
      </div>
      {pageTitles.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium mb-1">
            {t("chatImport.progress.justCompiled")}
          </p>
          <ul className="text-sm text-blue-700">
            {pageTitles.slice(0, 5).map((t) => (
              <li key={t}>{"\u2192"} {t}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
