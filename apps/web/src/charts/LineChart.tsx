import {
  AreaSeries,
  ColorType,
  createChart,
  CrosshairMode,
  LineSeries,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import type { Point } from "../lib/api";
import { fmt } from "../lib/format";
import { useTheme } from "../lib/theme";

export interface LineSpec {
  key: string;
  label: string;
  points: Point[];
  color?: string;
  dashed?: boolean;
  area?: boolean;
}

const toTime = (iso: string) => (Date.parse(`${iso}T00:00:00Z`) / 1000) as UTCTimestamp;

/** Multi-series line chart (equity curves, drawdowns) with a crosshair legend. */
export function LineChart({
  series,
  height = 320,
  format = (v: number) => fmt.num(v, 1),
}: {
  series: LineSpec[];
  height?: number;
  format?: (v: number) => string;
}) {
  const { chart: palette } = useTheme();
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRefs = useRef<Map<string, ISeriesApi<"Line"> | ISeriesApi<"Area">>>(new Map());
  const [hover, setHover] = useState<Record<string, number> | null>(null);
  const [hoverTime, setHoverTime] = useState<string | null>(null);

  useEffect(() => {
    if (!container.current) return;
    const chart = createChart(container.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: palette.text, fontSize: 12, attributionLogo: false },
      grid: { vertLines: { visible: false }, horzLines: { color: palette.grid } },
      crosshair: { mode: CrosshairMode.Magnet, vertLine: { color: palette.crosshair, style: LineStyle.Dashed }, horzLine: { visible: false } },
      rightPriceScale: { borderColor: palette.border },
      timeScale: { borderColor: palette.border },
      localization: { locale: "es-CL", priceFormatter: format },
      handleScale: { axisPressedMouseMove: false },
    });
    chartRef.current = chart;
    const handler = (param: MouseEventParams<Time>) => {
      if (!param.time) {
        setHover(null);
        setHoverTime(null);
        return;
      }
      const values: Record<string, number> = {};
      seriesRefs.current.forEach((s, key) => {
        const d = param.seriesData.get(s) as { value?: number } | undefined;
        if (d?.value !== undefined) values[key] = d.value;
      });
      setHover(values);
      setHoverTime(new Date((param.time as number) * 1000).toISOString().slice(0, 10));
    };
    chart.subscribeCrosshairMove(handler);
    return () => {
      chart.unsubscribeCrosshairMove(handler);
      chart.remove();
      chartRef.current = null;
      seriesRefs.current = new Map();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    chartRef.current?.applyOptions({
      layout: { textColor: palette.text },
      grid: { horzLines: { color: palette.grid } },
      rightPriceScale: { borderColor: palette.border },
      timeScale: { borderColor: palette.border },
    });
  }, [palette]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    seriesRefs.current.forEach((s) => chart.removeSeries(s));
    seriesRefs.current = new Map();
    series.forEach((spec, i) => {
      const color = spec.color ?? palette.series[i % palette.series.length];
      const common = {
        lineWidth: (i === 0 ? 2 : 1) as 1 | 2,
        lineStyle: spec.dashed ? LineStyle.Dashed : LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: i === 0,
      };
      const s = spec.area
        ? chart.addSeries(AreaSeries, {
            ...common,
            lineColor: color,
            topColor: `${color}00`,
            bottomColor: `${color}55`,
            invertFilledArea: true,
          })
        : chart.addSeries(LineSeries, { ...common, color });
      s.setData(spec.points.map((p) => ({ time: toTime(p.time), value: p.value })));
      seriesRefs.current.set(spec.key, s);
    });
    chart.timeScale().fitContent();
  }, [series, palette.series]);

  const last = (spec: LineSpec) => spec.points[spec.points.length - 1]?.value;
  return (
    <div>
      <div className="legend-row num" style={{ marginTop: 0, marginBottom: 8 }}>
        {hoverTime && <span className="muted">{fmt.date(hoverTime)}</span>}
        {series.map((spec, i) => {
          const color = spec.color ?? palette.series[i % palette.series.length];
          const value = hover ? hover[spec.key] : last(spec);
          return (
            <span key={spec.key}>
              <span className="legend-dot" style={{ background: color }} />
              <span className="muted">{spec.label}</span> <b>{value !== undefined ? format(value) : "—"}</b>
            </span>
          );
        })}
      </div>
      <div ref={container} style={{ height }} />
    </div>
  );
}
