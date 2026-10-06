import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { HBars } from "../charts/small";
import { AlertList, Card, DecisionBadge, ErrorBox, Icon, LoadingCard, ScoreBar, Stat } from "../components/ui";
import { api } from "../lib/api";
import { fmt, sectorLabel } from "../lib/format";

export function Portfolio() {
  const navigate = useNavigate();
  const { data, isLoading, isError, error, refetch } = useQuery({ queryKey: ["model-portfolio"], queryFn: api.modelPortfolio });
  if (isError) return <div className="page"><ErrorBox error={error} retry={() => refetch()} /></div>;
  const r = data?.risk;
  const limits = data?.limits ?? {};

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Cartera modelo</h1>
          <p>Pesos objetivo que propondría la estrategia hoy, con su riesgo ex-ante. No incluye tus posiciones actuales.</p>
        </div>
        {data && (
          <span className="badge" style={{ color: data.risk_enforced ? "var(--up)" : "var(--warn)", background: "var(--bg-elev-2)" }}>
            <span className="dot" />
            {data.risk_enforced ? "Límites de riesgo activos" : "Límites solo informativos"} · {data.strategy_version}
          </span>
        )}
      </div>

      {data && data.risk_actions.length > 0 && (
        <div className="banner banner-info">
          <Icon name="info" />
          <div>
            <b>El risk engine ajustó la cartera:</b> {data.risk_actions.join(" · ")}
          </div>
        </div>
      )}

      <div className="grid grid-4">
        <Stat label="Posiciones" value={data ? data.holdings.length : "…"} sub={data ? `Caja ${fmt.pct(data.cash_weight)}` : undefined} />
        <Stat label="Volatilidad ex-ante" value={fmt.pct(r?.volatility_annual)} sub={`Límite ${fmt.pct(limits.max_portfolio_volatility as number)}`} />
        <Stat label="VaR 95% diario" value={fmt.pct(r?.var_95_daily, 2)} sub={`CVaR ${fmt.pct(r?.cvar_95_daily, 2)} · límite ${fmt.pct(limits.max_var_95_daily as number, 2)}`} />
        <Stat label="Beta vs benchmark" value={fmt.num(r?.beta ?? null, 2)} sub={`Límite ${fmt.num(limits.max_beta as number, 2)}`} />
      </div>

      <div className="grid grid-main-side">
        {isLoading || !data ? (
          <LoadingCard height={420} />
        ) : (
          <Card title="Posiciones objetivo" bodyStyle={{ padding: 0 }}>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Sector</th>
                    <th>Decisión</th>
                    <th>Score</th>
                    <th className="right">Peso</th>
                    <th style={{ width: 140 }}></th>
                    <th className="right">Aporte al riesgo</th>
                    <th className="right">Vol. 3M</th>
                  </tr>
                </thead>
                <tbody>
                  {data.holdings.map((h) => (
                    <tr key={h.security_id} className="clickable" onClick={() => navigate(`/security/${h.security_id}`)}>
                      <td className="ticker">{h.ticker}</td>
                      <td className="muted">{sectorLabel(h.sector_id)}</td>
                      <td>
                        <DecisionBadge decision={h.decision} />
                      </td>
                      <td className="num">{fmt.num(h.composite_score, 1)}</td>
                      <td className="right num" style={{ fontWeight: 600 }}>
                        {fmt.pct(h.weight)}
                      </td>
                      <td>
                        <ScoreBar value={(h.weight / 0.1) * 100} color="var(--accent)" />
                      </td>
                      <td
                        className="right num"
                        style={{ color: (h.risk_contribution ?? 0) > (limits.max_single_name_risk_contribution as number) ? "var(--down)" : undefined }}
                      >
                        {fmt.pct(h.risk_contribution)}
                      </td>
                      <td className="right num muted">{fmt.pct(h.volatility_63d)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        <div className="stack">
          <Card title="Exposición sectorial">
            {data ? (
              <>
                <HBars
                  max={Math.max(data.max_sector_weight * 1.2, ...Object.values(data.sector_exposure))}
                  format={(v) => fmt.pct(v)}
                  markerLabel="Límite sectorial"
                  rows={Object.entries(data.sector_exposure)
                    .sort((a, b) => b[1] - a[1])
                    .map(([s, v]) => ({
                      label: sectorLabel(s),
                      value: v,
                      marker: data.max_sector_weight,
                      color: v > data.max_sector_weight + 1e-6 ? "var(--down)" : "var(--accent)",
                    }))}
                />
                <div className="faint" style={{ fontSize: 11.5, marginTop: 8 }}>
                  La línea vertical marca el límite por sector ({fmt.pct(data.max_sector_weight, 1)}).
                </div>
              </>
            ) : (
              <LoadingCard height={200} />
            )}
          </Card>
          <Card title="Límites violados y alertas">
            {data ? (
              <AlertList
                alerts={[
                  ...data.breaches.map((b) => ({ severity: b.severity as "WARNING" | "CRITICAL", code: b.code, message: b.message })),
                  ...data.alerts.filter((a) => !data.breaches.some((b) => b.code === a.code)),
                ]}
                empty="Dentro de todos los límites de riesgo"
              />
            ) : (
              <LoadingCard height={80} />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
