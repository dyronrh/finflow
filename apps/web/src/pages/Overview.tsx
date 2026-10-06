import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { LineChart } from "../charts/LineChart";
import { HBars } from "../charts/small";
import { AlertList, Card, DecisionBadge, decisionColor, ErrorBox, Icon, LoadingCard, ScoreBar, Stat } from "../components/ui";
import { api, DECISIONS } from "../lib/api";
import { decisionLabel, fmt, sectorLabel } from "../lib/format";

export function Overview() {
  const navigate = useNavigate();
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const portfolio = useQuery({ queryKey: ["model-portfolio"], queryFn: api.modelPortfolio });

  if (overview.isError) return <div className="page"><ErrorBox error={overview.error} retry={() => overview.refetch()} /></div>;
  const o = overview.data;
  const total = o ? Object.values(o.decision_counts).reduce((a, b) => a + b, 0) : 0;
  const longs = o ? o.decision_counts.STRONG_LONG + o.decision_counts.LONG : 0;
  const bench = o?.benchmark.points ?? [];
  const benchReturn = bench.length > 1 ? bench[bench.length - 1].value / bench[0].value - 1 : null;

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Resumen</h1>
          <p>Estado del universo, señales del modelo y riesgo de la cartera modelo.</p>
        </div>
      </div>

      {o && o.data_source !== "real" && (
        <div className="banner banner-warn">
          <Icon name="alert" />
          <div>
            <b>Datos sintéticos.</b> La plataforma corre sobre un mercado simulado; los tickers y resultados no representan acciones reales. Para usar
            datos de mercado: <code>make fetch-data</code> y luego <code>FINFLOW_DATA_SOURCE=real make api</code>.
          </div>
        </div>
      )}
      {o?.warnings.map((w) => (
        <div className="banner banner-danger" key={w}>
          <Icon name="alert" />
          <div>{w}</div>
        </div>
      ))}

      <div className="grid grid-4">
        <Stat label="Universo con datos" value={o ? fmt.num(o.securities_in_dataset, 0) : "…"} sub={o?.universe} />
        <Stat label="Elegibles hoy" value={o ? fmt.num(o.eligible, 0) : "…"} sub="Pasan filtros de precio, tamaño y liquidez" />
        <Stat label="Señales Long" value={o ? fmt.num(longs, 0) : "…"} sub={o ? `${o.decision_counts.STRONG_LONG} Strong Long` : undefined} tone="up" />
        <Stat
          label={`${o?.benchmark.name ?? "Benchmark"} · 12 meses`}
          value={fmt.signedPct(benchReturn)}
          tone={benchReturn !== null && benchReturn >= 0 ? "up" : "down"}
          sub={o ? `Datos al ${fmt.date(o.as_of_date)}` : undefined}
        />
      </div>

      <div className="grid grid-main-side">
        <div className="stack">
          {o ? (
            <Card title={`${o.benchmark.name} · último año (base 100)`}>
              <LineChart series={[{ key: "b", label: o.benchmark.name, points: bench }]} height={240} />
            </Card>
          ) : (
            <LoadingCard height={260} />
          )}

          <Card
            title="Top 10 del ranking"
            action={
              <button className="chip" onClick={() => navigate("/rankings")}>
                Ver ranking completo <Icon name="arrow" size={14} />
              </button>
            }
            bodyStyle={{ padding: 0 }}
          >
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Ticker</th>
                    <th>Sector</th>
                    <th>Decisión</th>
                    <th style={{ width: 200 }}>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {o?.top.map((r, i) => (
                    <tr key={r.security_id} className="clickable" onClick={() => navigate(`/security/${r.security_id}`)}>
                      <td className="faint num">{i + 1}</td>
                      <td className="ticker">{r.ticker}</td>
                      <td className="muted">{sectorLabel(r.sector_id)}</td>
                      <td>
                        <DecisionBadge decision={r.decision} />
                      </td>
                      <td>
                        <div className="row" style={{ flexWrap: "nowrap" }}>
                          <div style={{ flex: 1 }}>
                            <ScoreBar value={r.composite_score} />
                          </div>
                          <span className="num" style={{ width: 36, textAlign: "right", fontWeight: 600 }}>
                            {fmt.num(r.composite_score, 1)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        <div className="stack">
          <Card title="Distribución de señales">
            {o ? (
              <>
                <div className="distribution">
                  {DECISIONS.map((d) => (
                    <span key={d} style={{ width: `${(o.decision_counts[d] / Math.max(total, 1)) * 100}%`, background: decisionColor[d] }} title={`${decisionLabel[d]}: ${o.decision_counts[d]}`} />
                  ))}
                </div>
                <div className="legend-row">
                  {DECISIONS.map((d) => (
                    <button key={d} className="chip" style={{ padding: "3px 8px" }} onClick={() => navigate(`/rankings?decision=${d}`)}>
                      <span className="legend-dot" style={{ background: decisionColor[d] }} />
                      {decisionLabel[d]} <b className="num">{o.decision_counts[d]}</b>
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <LoadingCard height={40} />
            )}
          </Card>

          <Card title="Alertas · cartera modelo">
            {portfolio.data ? <AlertList alerts={portfolio.data.alerts} /> : portfolio.isError ? <ErrorBox error={portfolio.error} /> : <LoadingCard height={80} />}
          </Card>

          <Card title="Señales positivas por sector">
            {o ? (
              <>
                <HBars
                  max={1}
                  format={(v) => fmt.pct(v, 1)}
                  rows={[...o.sectors]
                    .sort((a, b) => b.positive / b.count - a.positive / a.count)
                    .map((s) => ({ label: sectorLabel(s.sector_id), value: s.count ? s.positive / s.count : null }))}
                />
                <div className="faint" style={{ fontSize: 11.5, marginTop: 8 }}>
                  Proporción de acciones elegibles del sector con señal Strong Long, Long o Watch.
                </div>
              </>
            ) : (
              <LoadingCard height={160} />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
