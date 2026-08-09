import type { SettingsSection } from "../settings/SettingsSidebar";

export type View =
  | { readonly kind: "activity" }
  | { readonly kind: "connect-agent" }
  | { readonly kind: "decisions" }
  | { readonly kind: "distill-review" }
  | { readonly kind: "entity"; readonly entityId: string }
  | { readonly kind: "graph" }
  | { readonly kind: "home" }
  | { readonly kind: "import" }
  | { readonly kind: "memory"; readonly sourceId: string }
  | { readonly kind: "page"; readonly pageId: string }
  | { readonly kind: "page-draft"; readonly draftId?: string; readonly space: string | null }
  | { readonly kind: "pages" }
  | { readonly kind: "recaps" }
  | { readonly kind: "settings"; readonly section?: SettingsSection }
  | { readonly kind: "sources" }
  | { readonly kind: "space"; readonly spaceId: string | null; readonly spaceName: string }
  | { readonly kind: "spaces"; readonly create?: boolean }
  | { readonly kind: "stream" };

export type GlobalNavigation = "home" | "memories" | "pages" | "spaces" | "graph" | "sources";

function assertNever(value: never): never {
  throw new TypeError(`Unsupported view: ${String(value)}`);
}

export function activeNavigationForView(view: View): GlobalNavigation | null {
  switch (view.kind) {
    case "home":
      return "home";
    case "activity":
    case "decisions":
    case "distill-review":
    case "memory":
    case "recaps":
    case "stream":
      return "memories";
    case "entity":
    case "page":
    case "page-draft":
    case "pages":
      return "pages";
    case "space":
    case "spaces":
      return "spaces";
    case "graph":
      return "graph";
    case "import":
    case "sources":
      return "sources";
    case "connect-agent":
    case "settings":
      return null;
    default:
      return assertNever(view);
  }
}
