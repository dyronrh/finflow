// Typed client for the FastAPI backend (apps/api). All endpoints are read-only.

export type Decision = "STRONG_LONG" | "LONG" | "WATCH" | "NEUTRAL" | "REDUCE" | "AVOID";
export const DECISIONS: Decision[] = ["STRONG_LONG", "LONG", "WATCH", "NEUTRAL", "REDUCE", "AVOID"];
export const FAMILIES = ["value", "growth", "profitability", "momentum", "revisions"] as const;
export type Family = (typeof FAMILIES)[number];
export type FactorScores = Record<Family, number | null>;

export interface RankRow {
  security_id: string;
  ticker: string;
  sector_id: string;
  decision: Decision;
  composite_score: number;
  percentile: number;
  factor_scores: FactorScores;
  risk_flags: string[];
  explanation?: string[];
  price?: number | null;
  market_cap_usd?: number | null;
  volatility_63d?: number | null;
}

export interface Overview {
  as_of_date: string;
  data_version: string;
  data_source: string;
  strategy_version: string;
  universe: string;
  warnings: string[];
  securities_in_dataset: number;
  eligible: number;
  decision_counts: Record<Decision, number>;
  sectors: { sector_id: string; count: number; positive: number }[];
  top: RankRow[];
  benchmark: { name: string; points: Point[] };
}

export interface Point {
  time: string;
  value: number;
}

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Rankings {
  as_of_date: string;
  strategy_version: string;
  data_version: string;
  eligible_count: number;
  items: RankRow[];
}

export interface SecurityAnalysis {
  as_of_date: string;
  security_id: string;
  ticker: string;
  sector_id: string;
  industry_id: string;
  eligible: boolean;
  ineligibility_reasons: string[];
  price: number;
  performance: Record<string, number | null>;
  decision: Decision | null;
  composite_score: number | null;
  composite_percentile: number | null;
  factor_scores: FactorScores | null;
  sector_median_scores: FactorScores | null;
  score_changes: { lookback: string; as_of_date: string; composite_score: number | null; delta: number | null }[];
  risk_flags: string[];
  missing_data: string[];
  explanation: string[];
}

export interface ScorePoint extends Partial<FactorScores> {
  date: string;
  composite_score: number | null;
  percentile?: number | null;
  decision: Decision | null;
}

export interface RiskSummary {
  gross_exposure: number;
  volatility_annual?: number;
  var_95_daily?: number;
  cvar_95_daily?: number;
  historical_var_95_daily?: number;
  historical_cvar_95_daily?: number;
  beta?: number | null;
  breaches?: string[];
}

export interface Alert {
  severity: "INFO" | "WARNING" | "CRITICAL";
  code: string;
  message: string;
  security_id?: string | null;
}

export interface ModelPortfolio {
  as_of_date: string;
  strategy_version: string;
  risk_enforced: boolean;
  risk_actions: string[];
  holdings: {
    security_id: string;
    ticker: string;
    sector_id: string;
    decision: Decision;
    composite_score: number;
    weight: number;
    risk_contribution: number | null;
    volatility_63d: number | null;
  }[];
  cash_weight: number;
  risk: RiskSummary;
  breaches: { code: string; severity: string; message: string; value: number; limit: number }[];
  sector_exposure: Record<string, number>;
  limits: Record<string, number | boolean>;
  max_sector_weight: number;
  alerts: Alert[];
}

export interface Backtest {
  start: string;
  end: string;
  strategy_version: string;
  series: Record<string, Point[]>;
  drawdown: Point[];
  benchmark_drawdown: Point[];
  summary: Record<string, number | null>;
  benchmark_summary: Record<string, number | null>;
  rebalances: { date: string; turnover: number | null; holdings: number | null; volatility: number | null }[];
  metadata: Record<string, unknown>;
}

export interface Paper {
  configured: boolean;
  ledger?: string;
  controls: Record<string, boolean>;
  account?: { cash: number; equity: number } | null;
  positions?: { security_id: string; qty: number; price: number | null; value: number | null; weight: number | null }[];
  proposals?: { id: string; created_at: string; as_of: string; status: string; nav: number; strategy_version: string; approval: { decided_by: string; decision: string; decided_at: string } | null }[];
  audit?: { at: string; actor: string; action: string; proposal_id: string | null; details: string }[];
  equity_history?: Point[];
}

export interface SecurityListItem {
  security_id: string;
  ticker: string;
  sector_id: string;
  has_prices: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, String(detail));
  }
  return response.json() as Promise<T>;
}

/**
 * Two data modes:
 * - live (default): the FastAPI backend (`make app`, `make api` + `make web-dev`);
 * - static (`vite build --mode static`): pre-generated JSON under /data, produced by
 *   `pipelines/export_static.py`, for serverless hosting such as Vercel.
 */
export const STATIC_MODE = import.meta.env.VITE_STATIC_DATA === "1";

const enc = encodeURIComponent;
const route = (live: string, file: string) => (STATIC_MODE ? `/data/${file}` : live);

/** Static snapshots store candles column-wise to halve their size. */
interface ColumnarCandles {
  time: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface SnapshotMeta {
  generated_at: string;
  data_source: string;
  data_version: string;
  strategy_version: string;
  price_history_years: number;
}

export const api = {
  meta: () => (STATIC_MODE ? get<SnapshotMeta>("/data/meta.json") : Promise.resolve(null)),
  overview: () => get<Overview>(route("/v1/overview", "overview.json")),
  rankings: (limit = 500) => get<Rankings>(route(`/v1/rankings?limit=${limit}`, "rankings.json")),
  securities: () => get<SecurityListItem[]>(route("/v1/securities", "securities.json")),
  analysis: (id: string) =>
    get<SecurityAnalysis>(route(`/v1/securities/${enc(id)}/analysis`, `securities/${enc(id)}/analysis.json`)),
  prices: async (id: string): Promise<{ candles: Candle[] }> => {
    if (!STATIC_MODE) return get<{ candles: Candle[] }>(`/v1/securities/${enc(id)}/prices`);
    const c = await get<ColumnarCandles>(`/data/securities/${enc(id)}/prices.json`);
    return {
      candles: c.time.map((time, i) => ({
        time,
        open: c.open[i],
        high: c.high[i],
        low: c.low[i],
        close: c.close[i],
        volume: c.volume[i],
      })),
    };
  },
  scoreHistory: (id: string, months = 24) =>
    get<{ points: ScorePoint[] }>(
      route(`/v1/securities/${enc(id)}/score-history?months=${months}`, `securities/${enc(id)}/history.json`),
    ),
  modelPortfolio: () => get<ModelPortfolio>(route("/v1/portfolio/model", "portfolio.json")),
  backtest: (years: number) => get<Backtest>(route(`/v1/backtest?years=${years}`, `backtest-${years}.json`)),
  paper: () => get<Paper>(route("/v1/paper", "paper.json")),
};
