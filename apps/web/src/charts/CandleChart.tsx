import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  PriceScaleMode,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Candle, ScorePoint } from "../lib/api";
import { decisionLabel, fmt } from "../lib/format";
import { useTheme } from "../lib/theme";
import { Segmented } from "../components/ui";

type Range = "1M" | "3M" | "6M" | "1A" | "3A" | "Todo";
const RANGE_DAYS: Record<Range, number | null> = { "1M": 31, "3M": 92, "6M": 183, "1A": 365, "3A": 1095, Todo: null };

const SMA_COLORS = { 20: "#60a5fa", 50: "#f59e0b", 200: "#a78bfa" } as const;
type SmaPeriod = keyof typeof SMA_COLORS;

const toTime = (iso: string) => (Date.parse(`${iso}T00:00:00Z`) / 1000) as UTCTimestamp;

function sma(candles: Candle[], period: number) {
  const out: { time: UTCTimestamp; value: number }[] = [];
  let sum = 0;
  for (let i = 0; i < candles.length; i++) {
    sum += candles[i].close;
    if (i >= period) sum -= candles[i - period].close;
    if (i >= period - 1) out.push({ time: toTime(candles[i].time), value: sum / period });
  }
  return out;
}

const BULLISH = new Set(["LONG", "STRONG_LONG"]);
const BEARISH = new Set(["REDUCE", "AVOID"]);

/** Markers where the strategy's decision changed between consecutive rebalances. */
function decisionMarkers(points: ScorePoint[], up: string, down: string, neutral: string): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];
  let previous: string | null = null;
  for (const p of points) {
    if (!p.decision || p.decision === previous) {
      previous = p.decision ?? previous;
      continue;
    }
    const bull = BULLISH.has(p.decision);
    const bear = BEARISH.has(p.decision);
    markers.push({
      time: toTime(p.date),
      position: bear ? "aboveBar" : "belowBar",
      shape: bull ? "arrowUp" : bear ? "arrowDown" : "circle",
      color: bull ? up : bear ? down : neutral,
      text: decisionLabel[p.decision],
      size: 1,
    });
    previous = p.decision;
  }
  return markers;
}

interface Legend {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number | null;
  sma: Partial<Record<SmaPeriod, number>>;
}

export function CandleChart({
  candles,
  scoreHistory = [],
  height = 460,
}: {
  candles: Candle[];
  scoreHistory?: ScorePoint[];
  height?: number;
}) {
  const { chart: palette } = useTheme();
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const smaRefs = useRef<Partial<Record<SmaPeriod, ISeriesApi<"Line">>>>({});
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  const [range, setRange] = useState<Range>("1A");
  const [smaOn, setSmaOn] = useState<Record<SmaPeriod, boolean>>({ 20: false, 50: true, 200: true });
  const [showVolume, setShowVolume] = useState(true);
  const [showSignals, setShowSignals] = useState(true);
  const [logScale, setLogScale] = useState(false);
  const [legend, setLegend] = useState<Legend | null>(null);

  const byTime = useMemo(() => {
    const map = new Map<number, { candle: Candle; prevClose: number | null }>();
    candles.forEach((c, i) => map.set(toTime(c.time), { candle: c, prevClose: i ? candles[i - 1].close : null }));
    return map;
  }, [candles]);
  const smaData = useMemo(
    () => ({ 20: sma(candles, 20), 50: sma(candles, 50), 200: sma(candles, 200) }),
    [candles],
  );

  const legendFor = (time: number | undefined): Legend | null => {
    const entry = time !== undefined ? byTime.get(time) : undefined;
    const last = entry ?? (candles.length ? { candle: candles[candles.length - 1], prevClose: candles.length > 1 ? candles[candles.length - 2].close : null } : null);
    if (!last) return null;
    const t = toTime(last.candle.time);
    const smaValues: Partial<Record<SmaPeriod, number>> = {};
    (Object.keys(SMA_COLORS) as unknown as SmaPeriod[]).forEach((p) => {
      const point = smaData[p].find((d) => d.time === t);
      if (point) smaValues[p] = point.value;
    });
    return {
      ...last.candle,
      change: last.prevClose ? last.candle.close / last.prevClose - 1 : null,
      sma: smaValues,
    };
  };

  // Create the chart once.
  useEffect(() => {
    if (!container.current) return;
    const chart = createChart(container.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: palette.text,
        fontSize: 12,
        attributionLogo: false,
        panes: { separatorColor: palette.border, separatorHoverColor: palette.grid },
      },
      grid: { vertLines: { color: palette.grid }, horzLines: { color: palette.grid } },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: palette.crosshair, style: LineStyle.Dashed, labelBackgroundColor: palette.background },
        horzLine: { color: palette.crosshair, style: LineStyle.Dashed, labelBackgroundColor: palette.background },
      },
      rightPriceScale: { borderColor: palette.border, scaleMargins: { top: 0.12, bottom: 0.08 } },
      timeScale: { borderColor: palette.border, rightOffset: 12, barSpacing: 7, minBarSpacing: 1.5 },
      localization: { locale: "es-CL" },
    });
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: palette.up,
      downColor: palette.down,
      borderUpColor: palette.up,
      borderDownColor: palette.down,
      wickUpColor: palette.up,
      wickDownColor: palette.down,
      priceLineStyle: LineStyle.Dotted,
    });
    const volume = chart.addSeries(
      HistogramSeries,
      { priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false },
      1,
    );
    chart.panes()[1]?.setStretchFactor(0.22);
    (Object.keys(SMA_COLORS) as unknown as SmaPeriod[]).forEach((p) => {
      smaRefs.current[p] = chart.addSeries(LineSeries, {
        color: SMA_COLORS[p],
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
    });
    markersRef.current = createSeriesMarkers(candle, []);
    chartRef.current = chart;
    candleRef.current = candle;
    volumeRef.current = volume;
    return () => {
      chart.remove();
      chartRef.current = null;
      smaRefs.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Theme.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.applyOptions({
      layout: { textColor: palette.text, panes: { separatorColor: palette.border, separatorHoverColor: palette.grid } },
      grid: { vertLines: { color: palette.grid }, horzLines: { color: palette.grid } },
      crosshair: {
        vertLine: { color: palette.crosshair, labelBackgroundColor: palette.background },
        horzLine: { color: palette.crosshair, labelBackgroundColor: palette.background },
      },
      rightPriceScale: { borderColor: palette.border },
      timeScale: { borderColor: palette.border },
    });
    candleRef.current?.applyOptions({
      upColor: palette.up,
      downColor: palette.down,
      borderUpColor: palette.up,
      borderDownColor: palette.down,
      wickUpColor: palette.up,
      wickDownColor: palette.down,
    });
  }, [palette]);

  // Data.
  useEffect(() => {
    if (!candleRef.current || !volumeRef.current) return;
    const data: CandlestickData<Time>[] = candles.map((c) => ({
      time: toTime(c.time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleRef.current.setData(data);
    volumeRef.current.setData(
      candles.map((c) => ({
        time: toTime(c.time),
        value: c.volume,
        color: c.close >= c.open ? palette.upVolume : palette.downVolume,
      })),
    );
    (Object.keys(SMA_COLORS) as unknown as SmaPeriod[]).forEach((p) => smaRefs.current[p]?.setData(smaData[p]));
    setLegend(legendFor(undefined));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candles, palette.upVolume, palette.downVolume, smaData]);

  // Indicators, volume, scale.
  useEffect(() => {
    (Object.keys(SMA_COLORS) as unknown as SmaPeriod[]).forEach((p) => smaRefs.current[p]?.applyOptions({ visible: smaOn[p] }));
  }, [smaOn]);
  useEffect(() => {
    volumeRef.current?.applyOptions({ visible: showVolume });
    chartRef.current?.panes()[1]?.setStretchFactor(showVolume ? 0.22 : 0.0001);
  }, [showVolume]);
  useEffect(() => {
    chartRef.current?.priceScale("right").applyOptions({ mode: logScale ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal });
  }, [logScale]);
  useEffect(() => {
    markersRef.current?.setMarkers(showSignals ? decisionMarkers(scoreHistory, palette.up, palette.down, palette.text) : []);
  }, [scoreHistory, showSignals, palette]);

  // Visible range.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !candles.length) return;
    const days = RANGE_DAYS[range];
    const last = toTime(candles[candles.length - 1].time);
    if (days === null) {
      chart.timeScale().fitContent();
      return;
    }
    const from = Math.max(toTime(candles[0].time), last - days * 86400) as UTCTimestamp;
    chart.timeScale().setVisibleRange({ from, to: last });
  }, [range, candles]);

  // Crosshair legend.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const handler = (param: MouseEventParams<Time>) => setLegend(legendFor(param.time as number | undefined));
    chart.subscribeCrosshairMove(handler);
    return () => chart.unsubscribeCrosshairMove(handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [byTime, smaData]);

  const tone = legend && legend.close >= legend.open ? "up" : "down";
  return (
    <div className="chart-shell">
      <div className="chart-toolbar">
        <Segmented
          options={(Object.keys(RANGE_DAYS) as Range[]).map((r) => ({ value: r, label: r }))}
          value={range}
          onChange={setRange}
        />
        <div className="row" style={{ gap: 14 }}>
          {(Object.keys(SMA_COLORS) as unknown as SmaPeriod[]).map((p) => (
            <label className="toggle" key={p}>
              <input type="checkbox" checked={smaOn[p]} onChange={(e) => setSmaOn((s) => ({ ...s, [p]: e.target.checked }))} />
              <span className="legend-dot" style={{ background: SMA_COLORS[p] }} />
              SMA {p}
            </label>
          ))}
          <label className="toggle">
            <input type="checkbox" checked={showVolume} onChange={(e) => setShowVolume(e.target.checked)} />
            Volumen
          </label>
          <label className="toggle" title="Cambios de decisión del modelo en cada rebalanceo">
            <input type="checkbox" checked={showSignals} onChange={(e) => setShowSignals(e.target.checked)} />
            Señales
          </label>
          <label className="toggle">
            <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
            Log
          </label>
        </div>
      </div>
      <div style={{ position: "relative" }}>
        {legend && (
          <div className="chart-legend num">
            <span className="muted">{fmt.date(legend.time)}</span>
            <span className="muted">
              O<b className={tone}>{fmt.num(legend.open, 2)}</b>
            </span>
            <span className="muted">
              H<b className={tone}>{fmt.num(legend.high, 2)}</b>
            </span>
            <span className="muted">
              L<b className={tone}>{fmt.num(legend.low, 2)}</b>
            </span>
            <span className="muted">
              C<b className={tone}>{fmt.num(legend.close, 2)}</b>
            </span>
            <span className={legend.change !== null && legend.change >= 0 ? "up" : "down"}>{fmt.signedPct(legend.change)}</span>
            <span className="muted">
              Vol<b>{fmt.compact(legend.volume)}</b>
            </span>
            {(Object.keys(legend.sma) as unknown as SmaPeriod[])
              .filter((p) => smaOn[p])
              .map((p) => (
                <span key={p} style={{ color: SMA_COLORS[p] }}>
                  SMA{p} <b>{fmt.num(legend.sma[p], 2)}</b>
                </span>
              ))}
          </div>
        )}
        <div ref={container} style={{ height }} />
      </div>
    </div>
  );
}
