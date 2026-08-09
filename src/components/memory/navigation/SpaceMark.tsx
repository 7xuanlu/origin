type SpaceMarkProps = {
  readonly active: boolean;
};

export function SpaceMark({ active }: SpaceMarkProps) {
  return (
    <svg
      aria-hidden="true"
      data-space-mark="self-contained-world"
      focusable="false"
      height="14"
      style={{ color: active ? "var(--mem-accent-indigo)" : "var(--mem-text-tertiary)" }}
      viewBox="0 0 24 24"
      width="14"
    >
      <path
        d="M18.816 13.58c2.292 2.138 3.546 4 3.092 4.9c-.745 1.46 -5.783 -.259 -11.255 -3.838c-5.47 -3.579 -9.304 -7.664 -8.56 -9.123c.464 -.91 2.926 -.444 5.803 .805"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M5 12a7 7 0 1 0 14 0a7 7 0 1 0 -14 0"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}
