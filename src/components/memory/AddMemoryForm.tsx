// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { quickCapture, type Space } from "../../lib/tauri";
import { Button, Select } from "./settings/primitives";

interface AddMemoryFormProps {
  spaces: Space[];
  onClose: () => void;
}

export default function AddMemoryForm({ spaces, onClose }: AddMemoryFormProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [content, setContent] = useState("");
  const [domain, setDomain] = useState("");

  const mutation = useMutation({
    mutationFn: () =>
      quickCapture({
        content: content.trim(),
        title: content.trim().split("\n")[0].slice(0, 60),
        tags: [],
        domain: domain || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      queryClient.invalidateQueries({ queryKey: ["memoryStats"] });
      queryClient.invalidateQueries({ queryKey: ["spaces"] });
      onClose();
    },
  });

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.metaKey && content.trim()) {
      e.preventDefault();
      mutation.mutate();
    }
    if (e.key === "Escape") onClose();
  };

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{
        border: "1px solid var(--mem-accent-indigo)",
        backgroundColor: "var(--mem-surface)",
        animation: "mem-fade-up 300ms cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      <div className="p-4">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t("addMemory.placeholder")}
          autoFocus
          rows={3}
          className="w-full bg-transparent resize-none outline-none placeholder:text-[var(--mem-text-tertiary)]"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "14px",
            color: "var(--mem-text)",
            lineHeight: "1.6",
          }}
        />
        <div className="flex items-center justify-between mt-3 pt-3" style={{ borderTop: "1px solid var(--mem-border)" }}>
          <div className="flex items-center gap-2">
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)" }}>
              {t("addMemory.spaceLabel")}
            </span>
            <Select
              size="sm"
              aria-label={t("addMemory.spaceLabel")}
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            >
              <option value="">{t("addMemory.noSpace")}</option>
              {spaces.map((s) => (
                <option key={s.id} value={s.name}>{s.name}</option>
              ))}
            </Select>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>
              {t("addMemory.cancel")}
            </Button>
            <Button
              variant="primary"
              size="sm"
              disabled={!content.trim()}
              loading={mutation.isPending}
              onClick={() => content.trim() && mutation.mutate()}
            >
              {t("addMemory.save")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
