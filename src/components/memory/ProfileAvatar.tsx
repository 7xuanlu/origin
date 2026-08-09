// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState, type CSSProperties } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";

interface ProfileAvatarProps {
  avatarPath?: string | null;
  displayName: string;
  size: number;
  fontSize: number;
  className?: string;
  style?: CSSProperties;
}

function initialsFor(displayName: string): string {
  const initials = displayName
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  return initials || "?";
}

export default function ProfileAvatar({
  avatarPath,
  displayName,
  size,
  fontSize,
  className,
  style,
}: ProfileAvatarProps) {
  const [failedPath, setFailedPath] = useState<string | null>(null);

  useEffect(() => {
    setFailedPath(null);
  }, [avatarPath]);

  const showImage = Boolean(avatarPath && failedPath !== avatarPath);
  const baseStyle: CSSProperties = {
    width: size,
    height: size,
    borderRadius: "50%",
    ...style,
  };

  if (showImage && avatarPath) {
    return (
      <img
        src={convertFileSrc(avatarPath)}
        alt={displayName}
        className={className}
        style={{ ...baseStyle, objectFit: "cover" }}
        onError={() => setFailedPath(avatarPath)}
      />
    );
  }

  return (
    <div
      className={className}
      style={{
        ...baseStyle,
        background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--mem-font-heading)",
        fontSize,
        color: "white",
        fontWeight: 500,
      }}
      aria-label={displayName ? `${displayName} initials` : "Profile initials"}
    >
      {initialsFor(displayName)}
    </div>
  );
}
