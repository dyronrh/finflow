import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { CandleChart } from "../charts/CandleChart";
import { LineChart } from "../charts/LineChart";
import { Gauge, HBars } from "../charts/small";
import { Card, DecisionBadge, decisionColor, Empty, ErrorBox, Icon, LoadingCard, Skeleton } from "../components/ui";
import { api, ApiError, FAMILIES } from "../lib/api";
import { familyLabel, fmt, sectorLabel } from "../lib/format";

const FEATURE_LABELS: Record<string, string> = {
  earnings_yield: "Earnings yield",
  fcf_yield: "FCF yield",
  ev_ebitda: "EV/EBITDA",
  revenue_growth: "Crecimiento ventas",
  eps_growth: "Crecimiento EPS",
  roic: "ROIC",
  gross_margin: "Margen bruto",
  fcf_margin: "Margen FCF",
  net_debt_ebitda: "Deuda neta/EBITDA",
  mom_12_1: "Momentum 12-1",
  eps_rev_30d: "Revisiones 30d",
  eps_rev_90d: "Revisiones 90d",
  revisions_up_ratio: "% revisiones al alza",
};

const REASONS: Record<string, string> = {
  PRICE_BELOW_MIN: "Precio bajo el mínimo",
  MARKET_CAP_BELOW_MIN: "Market cap bajo el mínimo",
  ADV_BELOW_MIN: "Liquidez (ADV) insuficiente",
  MISSING_SECTOR: "Sin sector",
  NOT_IN_UNIVERSE: "No pertenece al índice en esta fecha",
};

export function Security() {
  const { id = "" } = useParams();
  const analysis = useQuery({ queryKey: ["analysis", id], queryFn: () => api.analysis(id), retry: (n, e) => !(e instanceof ApiError && e.status === 404) && n < 2 });
  const prices = useQuery({ queryKey: ["prices", id], queryFn: () => api.prices(id), staleTime: 5 * 60_000 });
  const history = useQuery({ queryKey: ["score-history", id], queryFn: () => api.scoreHistory(id, 24), staleTime: 5 * 60_000 });

  if (analysis.isError) {
    const notFound = analysis.error instanceof ApiError && analysis.error.status === 404;
    return (
      <div className="page">
        {notFound ? (
          <Card>
            <Empty title={`“${id}” no está en el dataset`}>
              Puede no pertenecer al universo (S&amp;P 500), no tener precios o reportes en la SEC, o no cotizar en bolsa (por ejemplo, empresas
              privadas). <Link to="/rankings" style={{ color: "var(--accent)" }}>Ver rankings</Link>
            </Empty>
          </Card>
        ) : (
          <ErrorBox error={analysis.error} retry={() => analysis.refetch()} />
        )}
      </div>
    );
  }

  const a = analysis.data;
  const candles = prices.data?.candles ?? [];
  const lastClose = candles.length ? candles[candles.length - 1].close : a?.price;
  const prevClose = candles.length > 1 ? candles[candles.length - 2].close : null;
  const dayChange = lastClose && prevClose ? lastClose / prevClose - 1 : null;
  const color = a?.decision ? decisionColor[a.decision] : "var(--text-faint)";
  const points = history.data?.points ?? [];

  return (
    <div className="page">
      <div className="security-head">
        <div>
          <div className="row" style={{ gap: 12 }}>
            <h1>{a?.ticker ?? id}</h1>
            {a ? <DecisionBadge decision={a.decision} large /> : <Skeleton width={90} height={24} />}
          </div>
          <div className="muted" style={{ marginTop: 4 }}>
            {a ? (
              <>
                {sectorLabel(a.sector_id)} · {a.industry_id.replace(/_/g, " ")} · datos al {fmt.date(a.as_of_date)}
              </>
            ) : (
              <Skeleton width={260} />
            )}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="price-big num">{fmt.usd(lastClose)}</div>
          <div className="row" style={{ justifyContent: "flex-end", gap: 14, fontSize: 13 }}>
            <span className={dayChange !== null && dayChange >= 0 ? "up" : "down"}>{fmt.signedPct(dayChange)} hoy</span>
            {a &&
              Object.entries(a.performance).map(([k, v]) => (
                <span key={k} className="num">
                  <span className="faint">{k.replace("return_", "")} </span>
                  <span className={v !== null && v >= 0 ? "up" : "down"}>{fmt.signedPct(v)}</span>
                </span>
              ))}
          </div>
        </div>
      </div>

      {a && !a.eligible && (
        <div className="banner banner-warn">
          <Icon name="alert" />
          <div>
            <b>Fuera del universo elegible:</b> {a.ineligibility_reasons.map((r) => REASONS[r] ?? r).join(" · ")}. El modelo no le asigna score.
          </div>
        </div>
      )}

      <div className="grid grid-main-side">
        <div className="stack">
          <section className="card" style={{ overflow: "hidden" }}>
            {prices.isLoading ? (
              <div style={{ padding: 18 }}>
                <Skeleton height={500} />
              </div>
            ) : prices.isError ? (
              <div style={{ padding: 18 }}>
                <ErrorBox error={prices.error} />
              </div>
            ) : (
              <CandleChart candles={candles} scoreHistory={points} height={500} />
            )}
          </section>

          <Card title="Historial del score compuesto · rebalanceos mensuales">
            {history.isLoading ? (
              <Skeleton height={220} />
            ) : (
              <LineChart
                height={220}
                format={(v) => fmt.num(v, 1)}
                series={[
                  {
                    key: "score",
                    label: "Score",
                    points: points.filter((p) => p.composite_score !== null).map((p) => ({ time: p.date, value: p.composite_score as number })),
                  },
                  {
                    key: "momentum",
                    label: familyLabel.momentum,
                    dashed: true,
                    points: points.filter((p) => p.momentum != null).map((p) => ({ time: p.date, value: p.momentum as number })),
                  },
                  {
                    key: "value",
                    label: familyLabel.value,
                    dashed: true,
                    points: points.filter((p) => p.value != null).map((p) => ({ time: p.date, value: p.value as number })),
                  },
                ]}
              />
            )}
          </Card>
        </div>

        <div className="stack">
          {!a ? (
            <LoadingCard height={380} />
          ) : (
            <>
              <Card title="Score del modelo">
                <div className="row" style={{ gap: 18, flexWrap: "nowrap" }}>
                  <Gauge value={a.composite_score} color={color} />
                  <dl className="kv" style={{ flex: 1 }}>
                    <dt>Percentil</dt>
                    <dd className="num">{fmt.pct(a.composite_percentile, 1)}</dd>
                    {a.score_changes.map((c) => (
                      <FragmentRow key={c.lookback} label={`Cambio ${c.lookback}`} value={c.delta} />
                    ))}
                  </dl>
                </div>
              </Card>

              <Card title="Factores vs mediana del sector">
                {a.factor_scores ? (
                  <>
                    <HBars
                      markerLabel="Mediana del sector"
                      rows={FAMILIES.map((f) => ({
                        label: familyLabel[f],
                        value: a.factor_scores?.[f] ?? null,
                        marker: a.sector_median_scores?.[f] ?? null,
                      }))}
                    />
                    <div className="faint" style={{ fontSize: 11.5, marginTop: 8 }}>
                      La línea vertical marca la mediana del sector. Scores 0–100, relativos al sector.
                    </div>
                  </>
                ) : (
                  <span className="muted">Sin score.</span>
                )}
              </Card>

              <Card title="Por qué esta decisión">
                {a.explanation.length ? (
                  <ul style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 6 }}>
                    {a.explanation.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                ) : (
                  <span className="muted">Sin explicación disponible.</span>
                )}
                {a.risk_flags.length > 0 && (
                  <div className="banner banner-danger" style={{ marginTop: 12 }}>
                    <Icon name="alert" />
                    <div>Alerta de riesgo: {a.risk_flags.join(", ")}</div>
                  </div>
                )}
                {a.missing_data.length > 0 && (
                  <div className="faint" style={{ fontSize: 12, marginTop: 12 }}>
                    Datos faltantes: {a.missing_data.map((m) => FEATURE_LABELS[m] ?? m).join(", ")}
                  </div>
                )}
              </Card>
            </>
          )}
          <div className="faint" style={{ fontSize: 11.5 }}>
            Señales generadas por un modelo cuantitativo sin ventaja demostrada fuera de muestra. No es una recomendación de inversión.
          </div>
        </div>
      </div>
    </div>
  );
}

function FragmentRow({ label, value }: { label: string; value: number | null }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className={`num ${value === null ? "faint" : value >= 0 ? "up" : "down"}`}>
        {value === null ? "—" : `${value > 0 ? "+" : ""}${fmt.num(value, 1)}`}
      </dd>
    </>
  );
}
