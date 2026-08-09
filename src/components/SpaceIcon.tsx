// SPDX-License-Identifier: AGPL-3.0-only
/** Crisp monochrome SVG icons for Spaces — inherit color via currentColor */
export default function SpaceIcon({ icon, size = 16, className }: { icon: string; size?: number; className?: string }) {
  const s = { width: size, height: size };
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.5, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (icon) {
    case "terminal":
      return <svg viewBox="0 0 16 16" style={s} className={className}><rect x="1.5" y="2.5" width="13" height="11" rx="2" {...common} /><path d="M4.5 6l2.5 2-2.5 2" {...common} /><path d="M8.5 10.5h3" {...common} /></svg>;
    case "chat":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M2.5 3.5h11a1 1 0 011 1v6a1 1 0 01-1 1h-3l-2.5 2v-2h-5.5a1 1 0 01-1-1v-6a1 1 0 011-1z" {...common} /></svg>;
    case "globe":
      return <svg viewBox="0 0 16 16" style={s} className={className}><circle cx="8" cy="8" r="6" {...common} /><ellipse cx="8" cy="8" rx="2.8" ry="6" {...common} /><path d="M2 8h12" {...common} /><path d="M3 5h10M3 11h10" {...common} opacity={0.5} /></svg>;
    case "pencil":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M11.5 2.5l2 2-8.5 8.5-3 .5.5-3z" {...common} /><path d="M9.5 4.5l2 2" {...common} /></svg>;
    case "paintbrush":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M12.5 2.5c1 1 .5 2.5-.5 3.5l-4 4-3-3 4-4c1-1 2.5-1.5 3.5-.5z" {...common} /><path d="M5 10c-1.5.5-2.5 2-2.5 3.5 1.5 0 3-1 3.5-2.5" {...common} /></svg>;
    case "inbox":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M2.5 9.5l2-6h7l2 6" {...common} /><path d="M2.5 9.5v3a1 1 0 001 1h9a1 1 0 001-1v-3h-3.5l-1 1.5h-2l-1-1.5z" {...common} /></svg>;
    case "sparkles":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M8 1.5l1.2 3.8 3.8 1.2-3.8 1.2L8 11.5 6.8 7.7 3 6.5l3.8-1.2z" {...common} fill="currentColor" fillOpacity={0.15} /><path d="M12 10l.5 1.5 1.5.5-1.5.5-.5 1.5-.5-1.5L10 12l1.5-.5z" {...common} fill="currentColor" fillOpacity={0.15} /></svg>;
    case "test":
      return <svg viewBox="0 0 16 16" style={s} className={className}><path d="M6 2.5h4M7 2.5v4l-3.5 5.5a1 1 0 00.85 1.5h7.3a1 1 0 00.85-1.5L9 6.5v-4" {...common} /><path d="M5.5 10.5h5" {...common} opacity={0.4} /></svg>;
    default:
      return <span style={{ fontSize: size * 0.85, lineHeight: 1 }}>{icon}</span>;
  }
}
