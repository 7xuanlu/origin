import { useState } from "react";
import { useTranslation } from "react-i18next";

interface UpgradePromptProps {
  memoryCount: number;
  estimatedLocalHours: number;
  onCloud: (apiKey: string) => void;
  onLocal: () => void;
}

export function UpgradePrompt({
  memoryCount,
  estimatedLocalHours,
  onCloud,
  onLocal,
}: UpgradePromptProps) {
  const { t } = useTranslation();
  const [apiKey, setApiKey] = useState("");
  return (
    <div className="p-6 border rounded-lg max-w-xl">
      <h2 className="text-lg font-semibold mb-2">
        {t("chatImport.upgrade.title")}
      </h2>
      <p className="text-sm text-gray-700 mb-4">
        {t("chatImport.upgrade.description", {
          memoryCount: memoryCount.toLocaleString(),
          hours: estimatedLocalHours,
        })}
      </p>
      <input
        type="password"
        placeholder={t("chatImport.upgrade.placeholder")}
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        className="w-full px-3 py-2 border rounded mb-4"
      />
      <div className="flex gap-3">
        <button
          onClick={() => onCloud(apiKey)}
          disabled={!apiKey}
          className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
        >
          {t("chatImport.upgrade.useCloud")}
        </button>
        <button onClick={onLocal} className="px-4 py-2 border rounded">
          {t("chatImport.upgrade.continueLocal")}
        </button>
      </div>
    </div>
  );
}
