import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { LineChart } from "../charts/LineChart";
import { Card, Empty, ErrorBox, Icon, LoadingCard, Stat } from "../components/ui";
import { api, STATIC_MODE } from "../lib/api";
import { fmt } from "../lib/format";

const STATUS_COLOR: Record<string, string> = {
  PENDING_APPROVAL: "var(--warn)",
  APPROVED: "var(--info)",
  EXECUTING: "var(--info)",
  EXECUTED: "var(--up)",
  PARTIALLY_EXECUTED: "var(--warn)",
  REJECTED: "var(--text-faint)",
  EXPIRED: "var(--text-faint)",
  FAILED: "var(--down)",
};

export function Paper() {
  const navigate = useNavigate();
  const { data, isLoading, isError, error, refetch } = useQuery({ queryKey: ["paper"], queryFn: api.paper, refetchInterval: 30_000 });
  if (isError) return <div className="page"><ErrorBox error={error} retry={() => refetch()} /></div>;

  const c = data?.controls ?? {};
  const canTrade = c.global_trading_enabled && c.paper_trading_enabled && !c.kill_switch_active;

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Paper trading</h1>
          <p>Cuenta simulada, propuestas y auditoría. Vista de solo lectura.</p>
        </div>
        {data && (
          <div className="row">
            <span className="badge" style={{ color: c.kill_switch_active ? "var(--down)" : "var(--up)", background: "var(--bg-elev-2)" }}>
              <span className="dot" />
              {c.kill_switch_active ? "Kill switch ACTIVO" : "Kill switch inactivo"}
            </span>
            <span className="badge" style={{ color: canTrade ? "var(--up)" : "var(--warn)", background: "var(--bg-elev-2)" }}>
              <span className="dot" />
              {canTrade ? "Ejecución habilitada" : "Ejecución deshabilitada"}
            </span>
            <span className="badge" style={{ color: "var(--text-muted)", background: "var(--bg-elev-2)" }}>
              <Icon name="lock" size={12} /> Live trading bloqueado
            </span>
          </div>
        )}
      </div>

      <div className="banner banner-info">
        <Icon name="lock" />
        <div>
          {STATIC_MODE
            ? "Sitio estático: esta vista es una instantánea. Las propuestas, aprobaciones y ejecuciones se hacen en tu equipo con el CLI (python pipelines/paper_trade.py)."
            : "Aprobar y ejecutar se hace desde el CLI (python pipelines/paper_trade.py) hasta que la API tenga autenticación y roles. Así nadie puede enviar órdenes desde el navegador sin identificarse."}
        </div>
      </div>

      {isLoading || !data ? (
        <LoadingCard height={300} />
      ) : !data.configured ? (
        <Card>
          <Empty title="Aún no hay cuenta de paper trading">
            Crea la primera propuesta desde la terminal:
            <pre style={{ textAlign: "left", background: "var(--bg-elev-2)", padding: 12, borderRadius: 8, marginTop: 10 }}>
              python pipelines/paper_trade.py propose --by tu_nombre
            </pre>
          </Empty>
        </Card>
      ) : (
        <>
          <div className="grid grid-4">
            <Stat label="Patrimonio" value={fmt.usd(data.account?.equity)} />
            <Stat label="Caja" value={fmt.usd(data.account?.cash)} sub={data.account ? fmt.pct(data.account.cash / data.account.equity) : undefined} />
            <Stat label="Posiciones" value={data.positions?.length ?? 0} />
            <Stat label="Propuestas pendientes" value={data.proposals?.filter((p) => p.status === "PENDING_APPROVAL" || p.status === "APPROVED").length ?? 0} />
          </div>

          {data.equity_history && data.equity_history.length > 1 && (
            <Card title="Patrimonio (snapshots del broker)">
              <LineChart series={[{ key: "eq", label: "Patrimonio", points: data.equity_history }]} height={200} format={(v) => fmt.usd(v)} />
            </Card>
          )}

          <div className="grid grid-2">
            <Card title="Posiciones" bodyStyle={{ padding: 0 }}>
              <div className="table-wrap" style={{ maxHeight: 420 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th className="right">Cantidad</th>
                      <th className="right">Precio</th>
                      <th className="right">Valor</th>
                      <th className="right">Peso</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.positions?.map((p) => (
                      <tr key={p.security_id} className="clickable" onClick={() => navigate(`/security/${p.security_id}`)}>
                        <td className="ticker">{p.security_id}</td>
                        <td className="right num">{fmt.num(p.qty, 2)}</td>
                        <td className="right num">{fmt.usd(p.price)}</td>
                        <td className="right num">{fmt.usd(p.value)}</td>
                        <td className="right num">{fmt.pct(p.weight)}</td>
                      </tr>
                    ))}
                    {!data.positions?.length && (
                      <tr>
                        <td colSpan={5} className="muted" style={{ textAlign: "center", padding: 24 }}>
                          Sin posiciones.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Card>

            <Card title="Propuestas de rebalanceo" bodyStyle={{ padding: 0 }}>
              <div className="table-wrap" style={{ maxHeight: 420 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Creada</th>
                      <th>Estado</th>
                      <th>Aprobación</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.proposals?.map((p) => (
                      <tr key={p.id}>
                        <td className="num" style={{ fontSize: 12 }}>
                          {p.id}
                        </td>
                        <td className="muted">{fmt.datetime(p.created_at)}</td>
                        <td>
                          <span className="badge" style={{ color: STATUS_COLOR[p.status] ?? "var(--text-muted)", background: "var(--bg-elev-2)" }}>
                            <span className="dot" />
                            {p.status}
                          </span>
                        </td>
                        <td className="muted">{p.approval ? `${p.approval.decision} · ${p.approval.decided_by}` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <Card title="Auditoría (últimos 50 eventos)">
            <div className="list">
              {data.audit?.map((e, i) => (
                <div className="list-item" key={`${e.at}-${i}`}>
                  <span className="faint num" style={{ width: 150, flex: "none", fontSize: 12 }}>
                    {fmt.datetime(e.at)}
                  </span>
                  <span style={{ fontWeight: 600, width: 260, flex: "none" }}>{e.action}</span>
                  <span className="muted" style={{ width: 120, flex: "none" }}>
                    {e.actor}
                  </span>
                  <span className="faint" style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {e.proposal_id ?? ""} {e.details !== "{}" ? e.details : ""}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
