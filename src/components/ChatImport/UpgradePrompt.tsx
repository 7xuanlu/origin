import { useState } from "react";

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
  const [apiKey, setApiKey] = useState("");
  return (
    <div className="p-6 border rounded-lg max-w-xl">
      <h2 className="text-lg font-semibold mb-2">This is a large archive</h2>
      <p className="text-sm text-gray-700 mb-4">
        This is a {memoryCount.toLocaleString()}-memory archive. Local processing will take
        approximately {estimatedLocalHours} hours. Add an Anthropic API key to process in
        ~15 minutes, or continue with local.
      </p>
      <input
        type="password"
        placeholder="Anthropic API key (sk-ant-...)"
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
          Use cloud (fast)
        </button>
        <button onClick={onLocal} className="px-4 py-2 border rounded">
          Continue with local
        </button>
      </div>
    </div>
  );
}
