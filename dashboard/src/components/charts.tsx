import { useEffect, useState, type ReactNode } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import type { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";
import { fmt, toRows, type Point, type Series } from "../lib/shape";

const TOKENS = ["--surface", "--ink", "--ink-2", "--muted", "--grid", "--axis", "--s1", "--s2", "--s3", "--s4",
  "--s5", "--s6", "--s7", "--s8"] as const;
type Theme = Record<(typeof TOKENS)[number], string>;

function readTheme(): Theme {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((t) => [t, style.getPropertyValue(t).trim()])) as Theme;
}

/** Resolved palette (SVG attributes cannot rely on CSS variables); follows OS and toggle changes. */
export function useTheme(): Theme {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const update = () => setTheme(readTheme());
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", update);
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      media.removeEventListener("change", update);
      observer.disconnect();
    };
  }, []);
  return theme;
}

/** Series color by fixed slot order of the entity (never by rank). */
export function seriesColor(theme: Theme, index: number): string {
  return theme[`--s${(index % 8) + 1}` as keyof Theme];
}

export function Card({ title, subtitle, children, table }: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  table?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <section className="rounded-xl border p-5" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <h3 className="text-[15px] font-semibold">{title}</h3>
          {subtitle && <p className="mt-0.5 text-[13px]" style={{ color: "var(--ink-2)" }}>{subtitle}</p>}
        </div>
        {table && (
          <button
            type="button"
            onClick={() => setShowTable(!showTable)}
            className="shrink-0 rounded-md border px-2 py-1 text-xs"
            style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}
            aria-pressed={showTable}
          >
            {showTable ? "Chart" : "Table"}
          </button>
        )}
      </div>
      {showTable && table ? table : children}
    </section>
  );
}

export function Legend({ names, kind = "line" }: { names: string[]; kind?: "line" | "rect" }) {
  const theme = useTheme();
  if (names.length < 2) return null;
  return (
    <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs" style={{ color: "var(--ink-2)" }}>
      {names.map((n, i) => (
        <li key={n} className="flex items-center gap-1.5">
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: kind === "line" ? 14 : 10,
              height: kind === "line" ? 2 : 10,
              borderRadius: kind === "line" ? 1 : 2,
              background: seriesColor(theme, i),
            }}
          />
          {n}
        </li>
      ))}
    </ul>
  );
}

function TooltipBox({ title, rows }: { title: string; rows: { name: string; value: string; color: string }[] }) {
  return (
    <div className="rounded-lg border px-3 py-2 text-xs shadow-sm"
      style={{ background: "var(--surface)", borderColor: "var(--border)", color: "var(--ink)" }}>
      <div className="mb-1" style={{ color: "var(--muted)" }}>{title}</div>
      {rows.map((r) => (
        <div key={r.name} className="flex items-center gap-2">
          <span aria-hidden style={{ width: 10, height: 2, background: r.color, display: "inline-block" }} />
          <span className="font-semibold">{r.value}</span>
          <span style={{ color: "var(--ink-2)" }}>{r.name}</span>
        </div>
      ))}
    </div>
  );
}

const axisProps = (theme: Theme) => ({
  stroke: theme["--axis"],
  tick: { fill: theme["--muted"], fontSize: 11 },
  tickLine: false,
});

/** Lines with markers and a 10% CI band per series; crosshair tooltip lists every series. */
export function LineCI({ series, xLabel, format = (v: number) => fmt(v), height = 240 }: {
  series: Series[];
  xLabel: string;
  format?: (v: number) => string;
  height?: number;
}) {
  const theme = useTheme();
  const rows = toRows(series);
  const names = series.map((s) => s.name);
  const content = ({ active, payload, label }: TooltipContentProps<ValueType, NameType>) => {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload as Record<string, unknown>;
    return (
      <TooltipBox
        title={`${xLabel} ${label}`}
        rows={names.filter((n) => typeof row[n] === "number").map((n) => {
          const [lo, hi] = row[`${n}_ci`] as [number, number];
          return { name: n, value: `${format(row[n] as number)} [${format(lo)}, ${format(hi)}]`,
            color: seriesColor(theme, names.indexOf(n)) };
        })}
      />
    );
  };
  return (
    <>
      <Legend names={names} />
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
          <CartesianGrid stroke={theme["--grid"]} vertical={false} />
          <XAxis dataKey="x" {...axisProps(theme)}
            label={{ value: xLabel, position: "insideBottom", offset: -10, fill: theme["--muted"], fontSize: 11 }} />
          <YAxis {...axisProps(theme)} axisLine={false} width={56} tickFormatter={(v: number) => format(v)} />
          <Tooltip content={content} cursor={{ stroke: theme["--axis"], strokeWidth: 1 }} />
          {names.map((n, i) => (
            <Area key={`${n}-ci`} dataKey={`${n}_ci`} stroke="none" fill={seriesColor(theme, i)} fillOpacity={0.1}
              isAnimationActive={false} connectNulls activeDot={false} />
          ))}
          {names.map((n, i) => (
            <Line key={n} dataKey={n} stroke={seriesColor(theme, i)} strokeWidth={2} isAnimationActive={false}
              connectNulls dot={{ r: 4, fill: seriesColor(theme, i), stroke: theme["--surface"], strokeWidth: 2 }}
              activeDot={{ r: 5, stroke: theme["--surface"], strokeWidth: 2 }} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </>
  );
}

/** Columns with 95% CI whiskers; `groups` > 1 draws side-by-side bars per category. */
export function BarCI({ groups, format = (v: number) => fmt(v), height = 220 }: {
  groups: { name: string; points: Point[] }[];
  format?: (v: number) => string;
  height?: number;
}) {
  const theme = useTheme();
  const categories = [...new Set(groups.flatMap((g) => g.points.map((p) => String(p.x))))];
  const rows = categories.map((c) => {
    const row: Record<string, unknown> = { x: c };
    for (const g of groups) {
      const p = g.points.find((q) => String(q.x) === c);
      if (p) {
        row[g.name] = p.mean;
        row[`${g.name}_err`] = [p.mean - p.lo, p.hi - p.mean];
        row[`${g.name}_ci`] = [p.lo, p.hi];
      }
    }
    return row;
  });
  const content = ({ active, payload, label }: TooltipContentProps<ValueType, NameType>) => {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload as Record<string, unknown>;
    return (
      <TooltipBox
        title={String(label)}
        rows={groups.filter((g) => typeof row[g.name] === "number").map((g) => {
          const [lo, hi] = row[`${g.name}_ci`] as [number, number];
          return { name: g.name, value: `${format(row[g.name] as number)} [${format(lo)}, ${format(hi)}]`,
            color: seriesColor(theme, groups.indexOf(g)) };
        })}
      />
    );
  };
  return (
    <>
      <Legend names={groups.map((g) => g.name)} kind="rect" />
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }} barGap={2}>
          <CartesianGrid stroke={theme["--grid"]} vertical={false} />
          <XAxis dataKey="x" {...axisProps(theme)} interval={0} />
          <YAxis {...axisProps(theme)} axisLine={false} width={56} tickFormatter={(v: number) => format(v)} />
          <Tooltip content={content} cursor={{ fill: theme["--grid"], fillOpacity: 0.4 }} />
          {groups.map((g, i) => (
            <Bar key={g.name} dataKey={g.name} fill={seriesColor(theme, i)} maxBarSize={24} radius={[4, 4, 0, 0]}
              isAnimationActive={false}>
              <ErrorBar dataKey={`${g.name}_err`} stroke={theme["--ink-2"]} strokeWidth={1} width={6} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

export function DataTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c} className="border-b px-2 py-1.5 text-left font-medium"
                style={{ borderColor: "var(--border)", color: "var(--ink-2)" }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((v, j) => (
                <td key={j} className="border-b px-2 py-1.5" style={{ borderColor: "var(--border)" }}>
                  {typeof v === "number" ? fmt(v) : v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function StatTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border p-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="text-[13px]" style={{ color: "var(--ink-2)" }}>{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {note && <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>{note}</div>}
    </div>
  );
}

/** Table rows for a sweep: one row per (series, x). */
export function sweepTable(series: Series[], xLabel: string, format = (v: number) => fmt(v)) {
  return (
    <DataTable
      columns={["Series", xLabel, "Mean", "95% CI", "n"]}
      rows={series.flatMap((s) => s.points.map((p) => [s.name, String(p.x), format(p.mean),
        `${format(p.lo)} – ${format(p.hi)}`, p.n]))}
    />
  );
}

export function barTable(groups: { name: string; points: Point[] }[], format = (v: number) => fmt(v)) {
  return (
    <DataTable
      columns={["Series", "Category", "Mean", "95% CI", "n"]}
      rows={groups.flatMap((g) => g.points.map((p) => [g.name, String(p.x), format(p.mean),
        `${format(p.lo)} – ${format(p.hi)}`, p.n]))}
    />
  );
}
