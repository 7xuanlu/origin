// SPDX-License-Identifier: AGPL-3.0-only
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  classifyContent,
  prepareForRender,
  extractPreview,
  normalizeContent,
} from "../../lib/contentClassifier";

interface ContentRendererProps {
  content: string;
  structuredFields?: string | null;
  variant: "card" | "detail";
  className?: string;
}

const markdownComponents = {
  p: ({ children }: { children?: React.ReactNode }) => (
    <p
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: "14px",
        lineHeight: "1.7",
        color: "var(--mem-text)",
        marginBottom: "0.75em",
      }}
    >
      {children}
    </p>
  ),
  strong: ({ children }: { children?: React.ReactNode }) => (
    <strong
      style={{
        color: "var(--mem-text)",
        fontWeight: 600,
      }}
    >
      {children}
    </strong>
  ),
  li: ({ children }: { children?: React.ReactNode }) => (
    <li
      style={{
        fontFamily: "var(--mem-font-body)",
        fontSize: "14px",
        lineHeight: "1.7",
        color: "var(--mem-text)",
        paddingLeft: "0.25em",
      }}
    >
      {children}
    </li>
  ),
  ul: ({ children }: { children?: React.ReactNode }) => (
    <ul
      style={{
        listStyleType: "disc",
        paddingLeft: "1.5em",
        color: "var(--mem-text-tertiary)",
        marginBottom: "0.75em",
      }}
    >
      {children}
    </ul>
  ),
  ol: ({ children }: { children?: React.ReactNode }) => (
    <ol
      style={{
        listStyleType: "decimal",
        paddingLeft: "1.5em",
        color: "var(--mem-text-tertiary)",
        marginBottom: "0.75em",
      }}
    >
      {children}
    </ol>
  ),
  a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        color: "var(--mem-accent-indigo)",
        textDecoration: "none",
      }}
      onMouseEnter={(e) => {
        (e.target as HTMLAnchorElement).style.textDecoration = "underline";
      }}
      onMouseLeave={(e) => {
        (e.target as HTMLAnchorElement).style.textDecoration = "none";
      }}
    >
      {children}
    </a>
  ),
  h1: ({ children }: { children?: React.ReactNode }) => (
    <h1
      style={{
        fontFamily: "var(--mem-font-heading)",
        fontSize: "18px",
        fontWeight: 500,
        color: "var(--mem-text)",
        marginTop: "1em",
        marginBottom: "0.5em",
      }}
    >
      {children}
    </h1>
  ),
  h2: ({ children }: { children?: React.ReactNode }) => (
    <h2
      style={{
        fontFamily: "var(--mem-font-heading)",
        fontSize: "16px",
        fontWeight: 500,
        color: "var(--mem-text)",
        marginTop: "1em",
        marginBottom: "0.5em",
      }}
    >
      {children}
    </h2>
  ),
  h3: ({ children }: { children?: React.ReactNode }) => (
    <h3
      style={{
        fontFamily: "var(--mem-font-heading)",
        fontSize: "14px",
        fontWeight: 600,
        color: "var(--mem-text)",
        marginTop: "0.75em",
        marginBottom: "0.25em",
      }}
    >
      {children}
    </h3>
  ),
  code: ({
    children,
    className,
  }: {
    children?: React.ReactNode;
    className?: string;
  }) => {
    if (!className) {
      return (
        <code
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "12px",
            backgroundColor: "var(--mem-hover)",
            padding: "0.15em 0.4em",
            borderRadius: "3px",
          }}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className={className}
        style={{
          fontFamily: "var(--mem-font-mono)",
          fontSize: "12px",
        }}
      >
        {children}
      </code>
    );
  },
};

function CardPreview({
  content,
  structuredFields,
}: {
  content: string;
  structuredFields?: string | null;
}) {
  if (!content.trim()) {
    return (
      <span style={{ color: "var(--mem-text-tertiary)", fontStyle: "italic" }}>
        No content
      </span>
    );
  }

  const normalized = normalizeContent(content);
  const shape = classifyContent(normalized, structuredFields);
  const preview = extractPreview(normalized, shape);

  if (preview === null) {
    return <>{content.split("\n")[0]}</>;
  }

  if (typeof preview === "object") {
    return (
      <>
        <strong style={{ fontWeight: 600 }}>{preview.key}:</strong>{" "}
        {preview.value}
      </>
    );
  }

  // For prose-like single-facts with multiple sentences, show only the first sentence
  if (shape === "single-fact" && preview.includes(". ")) {
    const sentenceEnd = preview.indexOf(". ");
    return <>{preview.slice(0, sentenceEnd + 1)}</>;
  }

  return <>{preview}</>;
}

export default function ContentRenderer({
  content,
  structuredFields,
  variant,
  className,
}: ContentRendererProps) {
  if (variant === "card") {
    return (
      <span className={className}>
        <CardPreview content={content} structuredFields={structuredFields} />
      </span>
    );
  }

  if (!content.trim()) {
    return (
      <p
        style={{
          color: "var(--mem-text-tertiary)",
          fontStyle: "italic",
          fontFamily: "var(--mem-font-body)",
          fontSize: "14px",
        }}
      >
        No content
      </p>
    );
  }

  const normalized = normalizeContent(content);
  const shape = classifyContent(normalized, structuredFields);
  const prepared = prepareForRender(normalized, shape);

  return (
    <div className={["content-renderer", className].filter(Boolean).join(" ")}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {prepared}
      </ReactMarkdown>
    </div>
  );
}
