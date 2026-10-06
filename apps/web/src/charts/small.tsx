import type { Point } from "../lib/api";
import { fmt } from "../lib/format";

/** Tiny SVG sparkline (no axes) for KPI tiles and table rows. */
export function Sparkline({ points, width = 140, height = 36, color }: { points: Point[]; width?: number; height?: number; color?: string }) {
  if (points.length < 2) return null;
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = width / (values.length - 1);
  const d = values.map((v, i) => `${i ? "L" : "M"}${(i * step).toFixed(1)},${(height - ((v - min) / span) * (height - 4) - 2).toFixed(1)}`).join("");
  const up = values[values.length - 1] >= values[0];
  const stroke = color ?? (up ? "var(--up)" : "var(--down)");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
      <path d={`${d}L${width},${height}L0,${height}Z`} fill={stroke} opacity={0.08} />
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.6} strokeLinejoin="round" />
    </svg>
  );
}

/** Horizontal bars with an optional reference marker (e.g. sector median). */
export function HBars({
  rows,
  max = 100,
  format = (v: number) => fmt.num(v, 0),
  markerLabel,
}: {
  rows: { label: string; value: number | null; marker?: number | null; color?: string }[];
  max?: number;
  format?: (v: number) => string;
  markerLabel?: string;
}) {
  return (
    <div>
      {rows.map((r) => {
        const v = r.value ?? 0;
        const color =
          r.color ?? (r.value === null ? "var(--text-faint)" : v / max >= 0.7 ? "var(--up)" : v / max >= 0.45 ? "var(--info)" : v / max >= 0.25 ? "var(--warn)" : "var(--down)");
        return (
          <div className="hbar-row" key={r.label}>
            <span className="muted">{r.label}</span>
            <div className="hbar-track">
              <div className="hbar-fill" style={{ width: `${Math.min(100, (v / max) * 100)}%`, background: color }} />
              {r.marker !== undefined && r.marker !== null && (
                <div className="hbar-marker" style={{ left: `calc(${Math.min(100, (r.marker / max) * 100)}% - 1px)` }} title={`${markerLabel ?? "Referencia"}: ${format(r.marker)}`} />
              )}
            </div>
            <span className="num" style={{ textAlign: "right", fontWeight: 600 }}>
              {r.value === null ? "—" : format(v)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Circular score gauge 0–100. */
export function Gauge({ value, label, color }: { value: number | null; label?: string; color: string }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className="gauge">
      <svg width={120} height={120} viewBox="0 0 120 120">
        <circle cx={60} cy={60} r={r} fill="none" stroke="var(--bg-elev-2)" strokeWidth={10} />
        <circle cx={60} cy={60} r={r} fill="none" stroke={color} strokeWidth={10} strokeLinecap="round" strokeDasharray={`${(v / 100) * c} ${c}`} style={{ transition: "stroke-dasharray .6s ease" }} />
      </svg>
      <div className="center">
        <div>
          <b className="num">{value === null ? "—" : fmt.num(value, 0)}</b>
          <span className="faint" style={{ fontSize: 11 }}>
            {label ?? "/ 100"}
          </span>
        </div>
      </div>
    </div>
  );
}
