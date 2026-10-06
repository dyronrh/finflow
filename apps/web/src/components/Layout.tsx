import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { api, STATIC_MODE } from "../lib/api";
import { fmt, sectorLabel } from "../lib/format";
import { useTheme } from "../lib/theme";
import { Icon } from "./ui";

const NAV = [
  { to: "/", label: "Resumen", icon: "dashboard", end: true },
  { to: "/rankings", label: "Rankings", icon: "list" },
  { to: "/portfolio", label: "Cartera modelo", icon: "briefcase" },
  { to: "/backtest", label: "Backtest", icon: "history" },
  { to: "/paper", label: "Paper trading", icon: "wallet" },
];

function SearchBox() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const { data } = useQuery({ queryKey: ["securities"], queryFn: api.securities, staleTime: Infinity });

  const results = useMemo(() => {
    const q = query.trim().toUpperCase();
    if (!q || !data) return [];
    return data
      .filter((s) => s.ticker.toUpperCase().includes(q) || s.security_id.toUpperCase().includes(q))
      .sort((a, b) => Number(!a.ticker.toUpperCase().startsWith(q)) - Number(!b.ticker.toUpperCase().startsWith(q)) || a.ticker.localeCompare(b.ticker))
      .slice(0, 8);
  }, [query, data]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const go = (id: string) => {
    navigate(`/security/${encodeURIComponent(id)}`);
    setQuery("");
    setOpen(false);
    input.current?.blur();
  };

  return (
    <div className="search">
      <span className="icon">
        <Icon name="search" />
      </span>
      <input
        ref={input}
        value={query}
        placeholder="Buscar ticker…"
        aria-label="Buscar ticker"
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setActive(0);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((a) => Math.min(a + 1, results.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(a - 1, 0));
          } else if (e.key === "Enter" && results[active]) {
            go(results[active].security_id);
          } else if (e.key === "Escape") {
            setOpen(false);
            input.current?.blur();
          }
        }}
      />
      <kbd>/</kbd>
      {open && query && (
        <div className="search-results" role="listbox">
          {results.length === 0 ? (
            <div className="search-item muted">Sin resultados para “{query}”</div>
          ) : (
            results.map((r, i) => (
              <div
                key={r.security_id}
                role="option"
                aria-selected={i === active}
                className={`search-item ${i === active ? "active" : ""}`}
                onMouseDown={() => go(r.security_id)}
                onMouseEnter={() => setActive(i)}
              >
                <span className="ticker">{r.ticker}</span>
                <span className="faint" style={{ fontSize: 12 }}>
                  {sectorLabel(r.sector_id)}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function Layout() {
  const { theme, toggle } = useTheme();
  const { data } = useQuery({ queryKey: ["overview"], queryFn: api.overview, staleTime: 60_000 });
  const meta = useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: Infinity, enabled: STATIC_MODE });
  const synthetic = data?.data_source !== "real";
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Icon name="chart" />
          </div>
          <div>
            FinFlow
            <small>Quant Portfolio Intelligence</small>
          </div>
        </div>
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Icon name={n.icon} />
            <span>{n.label}</span>
          </NavLink>
        ))}
        <div className="sidebar-footer">
          Herramienta de investigación. No constituye asesoría financiera.
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <SearchBox />
          <div className="spacer" />
          {data && (
            <div className="row" style={{ gap: 8, fontSize: 12 }}>
              <span className={`badge ${synthetic ? "" : ""}`} style={{ color: synthetic ? "var(--warn)" : "var(--up)", background: synthetic ? "color-mix(in srgb, var(--warn) 12%, transparent)" : "color-mix(in srgb, var(--up) 12%, transparent)" }}>
                <span className="dot" />
                {synthetic ? "Datos sintéticos" : "Datos reales"}
              </span>
              <span className="faint num hide-mobile">al {fmt.date(data.as_of_date)}</span>
              <span className="faint hide-mobile">· {data.strategy_version}</span>
              {meta.data && (
                <span className="faint hide-mobile" title="Sitio estático: los datos se regeneran con pipelines/export_static.py">
                  · instantánea del {fmt.datetime(meta.data.generated_at)}
                </span>
              )}
            </div>
          )}
          <button className="icon-btn" onClick={toggle} aria-label="Cambiar tema" title="Cambiar tema">
            <Icon name={theme === "dark" ? "sun" : "moon"} />
          </button>
        </header>
        <Outlet />
      </div>
    </div>
  );
}
