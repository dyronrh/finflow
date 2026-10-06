import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Theme = "dark" | "light";

export interface ChartPalette {
  background: string;
  text: string;
  grid: string;
  border: string;
  up: string;
  down: string;
  upVolume: string;
  downVolume: string;
  accent: string;
  series: string[];
  crosshair: string;
}

const palettes: Record<Theme, ChartPalette> = {
  dark: {
    background: "#0f1420",
    text: "#8b95a7",
    grid: "rgba(139,149,167,0.08)",
    border: "rgba(139,149,167,0.18)",
    up: "#22c38e",
    down: "#f2555a",
    upVolume: "rgba(34,195,142,0.35)",
    downVolume: "rgba(242,85,90,0.35)",
    accent: "#2dd4bf",
    series: ["#2dd4bf", "#a78bfa", "#f59e0b", "#60a5fa"],
    crosshair: "rgba(226,232,240,0.35)",
  },
  light: {
    background: "#ffffff",
    text: "#5b6474",
    grid: "rgba(15,23,42,0.06)",
    border: "rgba(15,23,42,0.12)",
    up: "#0f9d6e",
    down: "#d93b42",
    upVolume: "rgba(15,157,110,0.30)",
    downVolume: "rgba(217,59,66,0.30)",
    accent: "#0d9488",
    series: ["#0d9488", "#7c3aed", "#d97706", "#2563eb"],
    crosshair: "rgba(15,23,42,0.35)",
  },
};

interface ThemeState {
  theme: Theme;
  toggle: () => void;
  chart: ChartPalette;
}

const ThemeContext = createContext<ThemeState | null>(null);

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem("finflow-theme");
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* storage unavailable */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("finflow-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  const value = useMemo(
    () => ({ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")), chart: palettes[theme] }),
    [theme],
  );
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeState {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme outside ThemeProvider");
  return ctx;
}
