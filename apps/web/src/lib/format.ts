const nf = (opts: Intl.NumberFormatOptions) => new Intl.NumberFormat("es-CL", opts);

const pct1 = nf({ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
const pct2 = nf({ style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const num0 = nf({ maximumFractionDigits: 0 });
const num1 = nf({ minimumFractionDigits: 1, maximumFractionDigits: 1 });
const num2 = nf({ minimumFractionDigits: 2, maximumFractionDigits: 2 });
const usd = nf({ style: "currency", currency: "USD", maximumFractionDigits: 2 });
const compact = nf({ notation: "compact", maximumFractionDigits: 1 });

const ok = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v);

export const fmt = {
  pct: (v?: number | null, digits: 1 | 2 = 1) => (ok(v) ? (digits === 2 ? pct2 : pct1).format(v) : "—"),
  signedPct: (v?: number | null) => (ok(v) ? `${v > 0 ? "+" : ""}${pct1.format(v)}` : "—"),
  num: (v?: number | null, digits: 0 | 1 | 2 = 1) =>
    ok(v) ? (digits === 0 ? num0 : digits === 1 ? num1 : num2).format(v) : "—",
  usd: (v?: number | null) => (ok(v) ? usd.format(v) : "—"),
  compactUsd: (v?: number | null) => (ok(v) ? `$${compact.format(v)}` : "—"),
  compact: (v?: number | null) => (ok(v) ? compact.format(v) : "—"),
  date: (iso?: string | null) =>
    iso ? new Date(`${iso.slice(0, 10)}T12:00:00`).toLocaleDateString("es-CL", { day: "2-digit", month: "short", year: "numeric" }) : "—",
  datetime: (iso?: string | null) =>
    iso ? new Date(iso).toLocaleString("es-CL", { dateStyle: "medium", timeStyle: "short" }) : "—",
};

export const sectorLabel = (id?: string | null) =>
  ({
    information_technology: "Tecnología",
    health_care: "Salud",
    financials: "Financiero",
    consumer_discretionary: "Consumo discrecional",
    consumer_staples: "Consumo básico",
    industrials: "Industrial",
    communication_services: "Comunicaciones",
    energy: "Energía",
    materials: "Materiales",
    utilities: "Utilities",
    real_estate: "Inmobiliario",
  })[id ?? ""] ??
  (id ?? "—").replace(/_/g, " ");

export const decisionLabel: Record<string, string> = {
  STRONG_LONG: "Strong Long",
  LONG: "Long",
  WATCH: "Watch",
  NEUTRAL: "Neutral",
  REDUCE: "Reduce",
  AVOID: "Avoid",
};

export const familyLabel: Record<string, string> = {
  value: "Valoración",
  growth: "Crecimiento",
  profitability: "Rentabilidad",
  momentum: "Momentum",
  revisions: "Revisiones",
};
