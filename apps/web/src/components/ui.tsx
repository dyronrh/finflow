import type { CSSProperties, ReactNode } from "react";
import type { Alert, Decision } from "../lib/api";
import { decisionLabel } from "../lib/format";

export const decisionColor: Record<Decision, string> = {
  STRONG_LONG: "var(--d-strong-long)",
  LONG: "var(--d-long)",
  WATCH: "var(--d-watch)",
  NEUTRAL: "var(--d-neutral)",
  REDUCE: "var(--d-reduce)",
  AVOID: "var(--d-avoid)",
};

export function Card({
  title,
  action,
  children,
  bodyStyle,
  className = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  bodyStyle?: CSSProperties;
  className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      {(title || action) && (
        <header className="card-header">
          {title ? <h2 className="card-title">{title}</h2> : <span />}
          {action}
        </header>
      )}
      <div className="card-body" style={bodyStyle}>
        {children}
      </div>
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "up" | "down";
}) {
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value num ${tone ?? ""}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export function DecisionBadge({ decision, large }: { decision: Decision | null | undefined; large?: boolean }) {
  if (!decision) return <span className="badge muted">Sin score</span>;
  const color = decisionColor[decision];
  return (
    <span
      className={`badge ${large ? "badge-lg" : ""}`}
      style={{
        color,
        background: `color-mix(in srgb, ${color} 13%, transparent)`,
        borderColor: `color-mix(in srgb, ${color} 30%, transparent)`,
      }}
    >
      <span className="dot" />
      {decisionLabel[decision]}
    </span>
  );
}

export function ScoreBar({ value, color }: { value: number | null | undefined; color?: string }) {
  const v = value ?? 0;
  const c = color ?? (v >= 70 ? "var(--up)" : v >= 45 ? "var(--info)" : v >= 25 ? "var(--warn)" : "var(--down)");
  return (
    <div className="score-bar" title={value == null ? "Sin datos" : value.toFixed(1)}>
      <span style={{ width: `${Math.max(0, Math.min(100, v))}%`, background: c }} />
    </div>
  );
}

export function Skeleton({ height = 16, width = "100%", style }: { height?: number; width?: number | string; style?: CSSProperties }) {
  return <div className="skeleton" style={{ height, width, ...style }} />;
}

export function LoadingCard({ height = 240 }: { height?: number }) {
  return (
    <div className="card" style={{ padding: 18 }}>
      <Skeleton height={14} width={140} />
      <Skeleton height={height} style={{ marginTop: 14 }} />
    </div>
  );
}

export function ErrorBox({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="banner banner-danger">
      <Icon name="alert" />
      <div style={{ flex: 1 }}>
        <b>No se pudieron cargar los datos.</b> <span>{message}</span>
        <div className="faint" style={{ marginTop: 4 }}>
          ¿Está corriendo la API? <code>make api</code>
        </div>
      </div>
      {retry && (
        <button className="chip" onClick={retry}>
          Reintentar
        </button>
      )}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <Icon name="inbox" size={28} />
      <b style={{ color: "var(--text)" }}>{title}</b>
      {children && <div style={{ maxWidth: 460 }}>{children}</div>}
    </div>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="tab"
          aria-selected={o.value === value}
          className={o.value === value ? "active" : ""}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

const severityTone: Record<Alert["severity"], string> = {
  INFO: "var(--info)",
  WARNING: "var(--warn)",
  CRITICAL: "var(--down)",
};

export function AlertList({ alerts, empty = "Sin alertas activas" }: { alerts: Alert[]; empty?: string }) {
  if (!alerts.length)
    return (
      <div className="row muted" style={{ padding: "6px 0" }}>
        <Icon name="check" /> {empty}
      </div>
    );
  return (
    <div className="list">
      {alerts.map((a, i) => (
        <div className="list-item" key={`${a.code}-${a.security_id}-${i}`}>
          <span style={{ color: severityTone[a.severity], marginTop: 1 }}>
            <Icon name={a.severity === "INFO" ? "info" : "alert"} />
          </span>
          <div>
            <div>{a.message}</div>
            <div className="faint" style={{ fontSize: 11.5 }}>
              {a.severity} · {a.code}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

const paths: Record<string, string> = {
  dashboard: "M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z",
  list: "M4 6h16M4 12h16M4 18h10",
  chart: "M4 19V5M4 19h16M8 15l3-4 3 2 5-7",
  briefcase: "M4 8h16v11H4zM9 8V5h6v3M4 13h16",
  history: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 7v5l3 3",
  wallet: "M3 7h15a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H3zM3 7V5a2 2 0 0 1 2-2h11M16 14h2",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm10 2-5-5",
  sun: "M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7 1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2-1.4-1.4M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  alert: "M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  info: "M12 16v-4m0-4h.01M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
  check: "M20 6 9 17l-5-5",
  inbox: "M22 12h-6l-2 3h-4l-2-3H2M5.5 5h13L22 12v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6z",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  lock: "M6 11h12v10H6zM8 11V7a4 4 0 0 1 8 0v4",
};

export function Icon({ name, size = 16 }: { name: keyof typeof paths | string; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={paths[name] ?? paths.info} />
    </svg>
  );
}
