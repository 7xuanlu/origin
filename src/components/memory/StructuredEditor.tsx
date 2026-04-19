// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";

const SCHEMAS: Record<string, { required: string[]; optional: string[]; labels: Record<string, string> }> = {
  identity:   { required: ["claim"], optional: ["evidence", "since"], labels: { claim: "I am / I do...", evidence: "How I know this", since: "Since when" } },
  preference: { required: ["preference", "applies_when"], optional: ["strength", "alternatives_rejected"], labels: { preference: "I prefer...", applies_when: "When this applies", strength: "How strongly", alternatives_rejected: "Over what alternatives" } },
  decision:   { required: ["decision", "context"], optional: ["alternatives_considered", "date", "reversible"], labels: { decision: "We decided to...", context: "Because...", alternatives_considered: "Instead of...", date: "When", reversible: "Reversible?" } },
  fact:       { required: ["claim"], optional: ["source", "verified", "domain"], labels: { claim: "The fact", source: "Where I learned this", verified: "Verified?", domain: "Topic area" } },
  goal:       { required: ["objective"], optional: ["target_date", "milestones", "status", "blockers"], labels: { objective: "I want to...", target_date: "By when", milestones: "Key milestones", status: "Current status", blockers: "What's blocking" } },
};

interface StructuredEditorProps {
  memoryType: string;
  initialFields?: Record<string, string>;
  onChange: (fields: Record<string, string>) => void;
}

export default function StructuredEditor({ memoryType, initialFields, onChange }: StructuredEditorProps) {
  const schema = SCHEMAS[memoryType] ?? SCHEMAS.fact;
  const [fields, setFields] = useState<Record<string, string>>(initialFields ?? {});

  const update = (key: string, value: string) => {
    const next = { ...fields, [key]: value };
    setFields(next);
    onChange(next);
  };

  const allFields = [...schema.required, ...schema.optional];

  return (
    <div className="flex flex-col gap-3">
      {allFields.map((f) => (
        <label key={f} className="flex flex-col gap-1">
          <span style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "11px",
            color: schema.required.includes(f) ? "var(--mem-text-secondary)" : "var(--mem-text-tertiary)",
            fontWeight: schema.required.includes(f) ? 500 : 400,
          }}>
            {schema.labels[f] ?? f}
            {schema.required.includes(f) && " *"}
          </span>
          <input
            type="text"
            value={fields[f] ?? ""}
            onChange={(e) => update(f, e.target.value)}
            placeholder={schema.labels[f] ?? f}
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
              padding: "6px 8px",
              borderRadius: "6px",
              border: "1px solid var(--mem-border)",
              backgroundColor: "var(--mem-surface)",
              color: "var(--mem-text)",
            }}
          />
        </label>
      ))}
    </div>
  );
}
