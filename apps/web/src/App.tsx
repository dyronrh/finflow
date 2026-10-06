import { Link, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Card, Empty } from "./components/ui";
import { Backtest } from "./pages/Backtest";
import { Overview } from "./pages/Overview";
import { Paper } from "./pages/Paper";
import { Portfolio } from "./pages/Portfolio";
import { Rankings } from "./pages/Rankings";
import { Security } from "./pages/Security";

function NotFound() {
  return (
    <div className="page">
      <Card>
        <Empty title="Página no encontrada">
          <Link to="/" style={{ color: "var(--accent)" }}>
            Volver al resumen
          </Link>
        </Empty>
      </Card>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="rankings" element={<Rankings />} />
        <Route path="security/:id" element={<Security />} />
        <Route path="portfolio" element={<Portfolio />} />
        <Route path="backtest" element={<Backtest />} />
        <Route path="paper" element={<Paper />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
