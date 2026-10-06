import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Card, DecisionBadge, decisionColor, ErrorBox, LoadingCard, ScoreBar } from "../components/ui";
import { api, DECISIONS, FAMILIES, type Decision, type RankRow } from "../lib/api";
import { decisionLabel, familyLabel, fmt, sectorLabel } from "../lib/format";

const SHORT: Record<string, string> = { value: "Valor", growth: "Crec.", profitability: "Rent.", momentum: "Mom.", revisions: "Rev." };

type SortKey = "rank" | "ticker" | "composite_score" | "price" | "market_cap_usd" | "volatility_63d" | (typeof FAMILIES)[number];

export function Rankings() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const decision = params.get("decision") as Decision | null;
  const [sector, setSector] = useState("all");
  const [text, setText] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "rank", dir: 1 });
  const { data, isLoading, isError, error, refetch } = useQuery({ queryKey: ["rankings"], queryFn: () => api.rankings(500) });

  const items = data?.items ?? [];
  const sectors = useMemo(() => [...new Set(items.map((i) => i.sector_id))].sort(), [items]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    items.forEach((i) => (c[i.decision] = (c[i.decision] ?? 0) + 1));
    return c;
  }, [items]);

  const rows = useMemo(() => {
    const ranked = items.map((r, i) => ({ ...r, rank: i + 1 }));
    const q = text.trim().toUpperCase();
    const filtered = ranked.filter(
      (r) => (!decision || r.decision === decision) && (sector === "all" || r.sector_id === sector) && (!q || r.ticker.toUpperCase().includes(q)),
    );
    const value = (r: RankRow & { rank: number }): number | string => {
      if (sort.key === "rank") return r.rank;
      if (sort.key === "ticker") return r.ticker;
      if ((FAMILIES as readonly string[]).includes(sort.key)) return r.factor_scores[sort.key as (typeof FAMILIES)[number]] ?? -1;
      return (r[sort.key as keyof RankRow] as number | null) ?? -Infinity;
    };
    return [...filtered].sort((a, b) => {
      const va = value(a);
      const vb = value(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  }, [items, decision, sector, text, sort]);

  const header = (key: SortKey, label: string, right = false) => (
    <th
      className={`sortable ${right ? "right" : ""}`}
      onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : key === "rank" || key === "ticker" ? 1 : -1 }))}
      aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
    >
      {label} {sort.key === key ? (sort.dir === 1 ? "↑" : "↓") : ""}
    </th>
  );

  const setDecision = (d: Decision | null) => {
    const next = new URLSearchParams(params);
    if (d) next.set("decision", d);
    else next.delete("decision");
    setParams(next, { replace: true });
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Rankings</h1>
          <p>
            Score compuesto 0–100 relativo al sector{data ? ` · ${data.eligible_count} acciones elegibles al ${fmt.date(data.as_of_date)}` : ""}.
          </p>
        </div>
      </div>

      <div className="row">
        <button className={`chip ${!decision ? "active" : ""}`} onClick={() => setDecision(null)}>
          Todas <b className="num">{items.length}</b>
        </button>
        {DECISIONS.map((d) => (
          <button key={d} className={`chip ${decision === d ? "active" : ""}`} onClick={() => setDecision(decision === d ? null : d)}>
            <span className="legend-dot" style={{ background: decisionColor[d] }} />
            {decisionLabel[d]} <b className="num">{counts[d] ?? 0}</b>
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <select className="select" value={sector} onChange={(e) => setSector(e.target.value)} aria-label="Sector">
          <option value="all">Todos los sectores</option>
          {sectors.map((s) => (
            <option key={s} value={s}>
              {sectorLabel(s)}
            </option>
          ))}
        </select>
        <input className="select" placeholder="Filtrar ticker" value={text} onChange={(e) => setText(e.target.value)} style={{ width: 150 }} />
      </div>

      {isError ? (
        <ErrorBox error={error} retry={() => refetch()} />
      ) : isLoading ? (
        <LoadingCard height={480} />
      ) : (
        <Card bodyStyle={{ padding: 0 }}>
          <div className="table-wrap" style={{ maxHeight: "calc(100vh - 260px)" }}>
            <table className="data">
              <thead>
                <tr>
                  {header("rank", "#")}
                  {header("ticker", "Ticker")}
                  <th>Sector</th>
                  <th>Decisión</th>
                  {header("composite_score", "Score")}
                  {FAMILIES.map((f) => header(f, SHORT[f]))}
                  {header("price", "Precio", true)}
                  {header("market_cap_usd", "Market cap", true)}
                  {header("volatility_63d", "Vol. 3M", true)}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.security_id} className="clickable" onClick={() => navigate(`/security/${r.security_id}`)}>
                    <td className="faint num">{r.rank}</td>
                    <td className="ticker">{r.ticker}</td>
                    <td className="muted" style={{ maxWidth: 150, overflow: "hidden", textOverflow: "ellipsis" }}>
                      {sectorLabel(r.sector_id)}
                    </td>
                    <td>
                      <DecisionBadge decision={r.decision} />
                    </td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap", minWidth: 110 }}>
                        <div style={{ flex: 1 }}>
                          <ScoreBar value={r.composite_score} />
                        </div>
                        <b className="num" style={{ width: 34, textAlign: "right" }}>
                          {fmt.num(r.composite_score, 1)}
                        </b>
                      </div>
                    </td>
                    {FAMILIES.map((f) => (
                      <td key={f} title={`${familyLabel[f]}: ${fmt.num(r.factor_scores[f], 1)}`}>
                        <div style={{ width: 48 }}>
                          <ScoreBar value={r.factor_scores[f]} />
                        </div>
                      </td>
                    ))}
                    <td className="right num">{fmt.usd(r.price)}</td>
                    <td className="right num muted">{fmt.compactUsd(r.market_cap_usd)}</td>
                    <td className="right num" style={{ color: r.risk_flags.length ? "var(--down)" : "var(--text-muted)" }} title={r.risk_flags.length ? "Volatilidad en el 5% más alto: AVOID automático" : undefined}>
                      {fmt.pct(r.volatility_63d)}
                      {r.risk_flags.length ? " ⚠" : ""}
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={13} className="muted" style={{ textAlign: "center", padding: 30 }}>
                      Ninguna acción coincide con los filtros.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
