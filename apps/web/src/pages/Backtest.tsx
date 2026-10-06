import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { LineChart, type LineSpec } from "../charts/LineChart";
import { Card, ErrorBox, Icon, Segmented, Skeleton, Stat } from "../components/ui";
import { api } from "../lib/api";
import { fmt } from "../lib/format";

const ROWS: { key: string; label: string; format: (v: number | null) => string }[] = [
  { key: "cagr", label: "CAGR", format: (v) => fmt.pct(v) },
  { key: "cumulative_return", label: "Retorno acumulado", format: (v) => fmt.pct(v) },
  { key: "annualized_volatility", label: "Volatilidad anual", format: (v) => fmt.pct(v) },
  { key: "sharpe_ratio", label: "Sharpe", format: (v) => fmt.num(v, 2) },
  { key: "sortino_ratio", label: "Sortino", format: (v) => fmt.num(v, 2) },
  { key: "max_drawdown", label: "Máximo drawdown", format: (v) => fmt.pct(v) },
  { key: "calmar_ratio", label: "Calmar", format: (v) => fmt.num(v, 2) },
  { key: "cvar_95_daily", label: "CVaR 95% diario", format: (v) => fmt.pct(v, 2) },
  { key: "hit_rate", label: "Días positivos", format: (v) => fmt.pct(v) },
];

export function Backtest() {
  const [years, setYears] = useState(3);
  const { data, isFetching, isError, error, refetch } = useQuery({
    queryKey: ["backtest", years],
    queryFn: () => api.backtest(years),
    staleTime: Infinity,
  });
  const s = data?.summary ?? {};
  const b = data?.benchmark_summary ?? {};
  const active = s.cagr != null && s.benchmark_cagr != null ? (s.cagr as number) - (s.benchmark_cagr as number) : null;

  const equity: LineSpec[] = data
    ? [
        { key: "strategy", label: "Estrategia", points: data.series.strategy },
        { key: "equal_weight", label: "Equal weight universo", points: data.series.equal_weight, dashed: true },
        ...(data.series.spy ? [{ key: "spy", label: "SPY", points: data.series.spy, dashed: true }] : []),
      ]
    : [];

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Backtest</h1>
          <p>Event-driven: señal al cierre de t, ejecución a la apertura de t+1, con costes, turnover y límites de la estrategia.</p>
        </div>
        <Segmented
          options={[1, 3, 5, 10].map((y) => ({ value: y, label: `${y} ${y === 1 ? "año" : "años"}` }))}
          value={years}
          onChange={setYears}
        />
      </div>

      {isError && <ErrorBox error={error} retry={() => refetch()} />}
      {isFetching && !data && (
        <div className="banner banner-info">
          <Icon name="history" />
          <div>
            <b>Calculando backtest…</b> La primera vez recalcula factores en cada rebalanceo; con datos reales puede tardar un par de minutos. Luego queda en
            caché.
          </div>
        </div>
      )}

      <div className="grid grid-4">
        <Stat label="CAGR estrategia" value={fmt.pct(s.cagr)} tone={(s.cagr ?? 0) >= 0 ? "up" : "down"} sub={`Benchmark ${fmt.pct(s.benchmark_cagr)}`} />
        <Stat label="Exceso anual vs benchmark" value={fmt.signedPct(active)} tone={(active ?? 0) >= 0 ? "up" : "down"} sub="Mismo universo, pesos iguales" />
        <Stat label="Sharpe" value={fmt.num(s.sharpe_ratio ?? null, 2)} sub={`Benchmark ${fmt.num(s.benchmark_sharpe ?? null, 2)}`} />
        <Stat label="Máximo drawdown" value={fmt.pct(s.max_drawdown)} tone="down" sub={`Benchmark ${fmt.pct(s.benchmark_max_drawdown)}`} />
      </div>

      <Card title="Curva de capital (base 100)" action={data && <span className="faint num" style={{ fontSize: 12 }}>{fmt.date(data.start)} → {fmt.date(data.end)}</span>}>
        {data ? <LineChart series={equity} height={340} /> : <Skeleton height={360} />}
      </Card>

      <div className="grid grid-2">
        <Card title="Drawdown">
          {data ? (
            <LineChart
              height={430}
              format={(v) => fmt.pct(v)}
              series={[
                { key: "dd", label: "Estrategia", points: data.drawdown, area: true, color: "#f2555a" },
                { key: "bdd", label: "Benchmark", points: data.benchmark_drawdown, dashed: true, color: "#8b95a7" },
              ]}
            />
          ) : (
            <Skeleton height={450} />
          )}
        </Card>
        <Card title="Métricas" bodyStyle={{ padding: 0 }}>
          <table className="data">
            <thead>
              <tr>
                <th>Métrica</th>
                <th className="right">Estrategia</th>
                <th className="right">Benchmark</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.key}>
                  <td className="muted">{r.label}</td>
                  <td className="right num" style={{ fontWeight: 600 }}>
                    {r.format((s[r.key] as number) ?? null)}
                  </td>
                  <td className="right num muted">{r.format((b[r.key] as number) ?? null)}</td>
                </tr>
              ))}
              <tr>
                <td className="muted">Costes totales</td>
                <td className="right num">{fmt.usd(s.total_costs_usd)}</td>
                <td className="right faint">—</td>
              </tr>
              <tr>
                <td className="muted">Turnover medio por rebalanceo</td>
                <td className="right num">{fmt.pct(s.avg_turnover_per_rebalance)}</td>
                <td className="right faint">—</td>
              </tr>
            </tbody>
          </table>
        </Card>
      </div>
      <div className="faint" style={{ fontSize: 12 }}>
        Los resultados pasados no garantizan resultados futuros. Revisa el sesgo de supervivencia y la fuente de datos antes de interpretar estos números.
      </div>
    </div>
  );
}
