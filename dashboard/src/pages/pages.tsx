import type { ReactNode } from "react";
import { BarCI, Card, DataTable, LineCI, StatTile, barTable, sweepTable } from "../components/charts";
import type { Data } from "../data";
import { bars, fmt, gatewayByCell, param, sweep } from "../lib/shape";
import type { CellSummary, ExperimentResult } from "../types/result";

const LABELS: Record<string, string> = {
  vllm_metal: "vLLM (Metal)",
  llamacpp_metal: "llama.cpp (Metal)",
  vllm_apc_on: "vLLM, prefix cache on",
  vllm_apc_off: "vLLM, prefix cache off",
  llamacpp_cache_on: "llama.cpp, cache on",
  llamacpp_cache_off: "llama.cpp, cache off",
  local_only: "Local only",
  local_first: "Local first (fixed threshold)",
  local_first_slo: "Local first (SLO-aware)",
  remote_only: "Remote only",
  rag_crag: "CRAG RAG",
  rag_hotpot: "HotpotQA RAG",
  bfcl: "BFCL tool calls",
  agent: "Agentic HotpotQA",
  implicit: "Implicit cache",
  explicit: "Explicit cache",
  batch: "Batch API",
};
const label = (v: string) => LABELS[v] ?? v;
const ms = (v: number) => `${fmt(v, 0)} ms`;
const pct = (v: number) => `${fmt(100 * v, 0)}%`;
const usd = (v: number) => `$${fmt(v, 3)}`;

function relabel<T extends { name: string }>(series: T[]): T[] {
  return series.map((s) => ({ ...s, name: label(s.name) }));
}

function Missing({ what }: { what: string }) {
  return <p className="text-sm" style={{ color: "var(--muted)" }}>No published run for {what} yet.</p>;
}

function Section({ title, intro, children }: { title: string; intro?: string; children: ReactNode }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-semibold">{title}</h2>
        {intro && <p className="mt-1 max-w-3xl text-sm" style={{ color: "var(--ink-2)" }}>{intro}</p>}
      </div>
      {children}
    </div>
  );
}

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid gap-4 lg:grid-cols-2">{children}</div>;
}

function SweepCard({ r, title, subtitle, metric, seriesKey, format }: {
  r: ExperimentResult;
  title: string;
  subtitle?: string;
  metric: string;
  seriesKey: string | null;
  format?: (v: number) => string;
}) {
  const series = relabel(sweep(r, "workload.concurrency", seriesKey, metric));
  return (
    <Card title={title} subtitle={subtitle} table={sweepTable(series, "Concurrency", format)}>
      <LineCI series={series} xLabel="Concurrency" format={format} />
    </Card>
  );
}

function BarCard({ r, title, subtitle, metric, by, format }: {
  r: ExperimentResult;
  title: string;
  subtitle?: string;
  metric: string;
  by: (c: CellSummary) => string;
  format?: (v: number) => string;
}) {
  const groups = [{ name: title, points: bars(r, metric, (c) => label(by(c))) }];
  return (
    <Card title={title} subtitle={subtitle} table={barTable(groups, format)}>
      <BarCI groups={groups} format={format} />
    </Card>
  );
}

export function Overview({ data }: { data: Data }) {
  const r = data.results;
  const tiles: ReactNode[] = [];
  const e2 = r["e2-prefix-caching"];
  if (e2) {
    const t = (v: string) => e2.cells.find((c) => param(c, "target.variant") === v)?.metrics.ttft_p50_ms?.mean;
    const off = t("vllm_apc_off"), on = t("vllm_apc_on");
    if (off && on) tiles.push(<StatTile key="e2" label="vLLM prefix caching, TTFT p50" value={`${fmt(off / on, 1)}× faster`}
      note={`${ms(off)} → ${ms(on)} on multi-turn RAG (E2)`} />);
  }
  const e3 = r["e3-llmd-sim"];
  if (e3) {
    const best = [...e3.cells].sort((a, b) => b.metrics.goodput_rps.mean - a.metrics.goodput_rps.mean)[0];
    tiles.push(<StatTile key="e3" label="Best llm-d scorer profile" value={param(best, "target.scorer_profile")}
      note={`${fmt(best.metrics.goodput_rps.mean, 1)} req/s goodput at SLO (E3, simulated workers)`} />);
  }
  const e4 = r["e4-hybrid-gateway"];
  if (e4) {
    const at = (p: string) => e4.cells.find((c) => param(c, "target.policy") === p && param(c, "workload.concurrency") === "24");
    const lf = at("local_first"), lo = at("local_only");
    if (lf && lo) tiles.push(<StatTile key="e4" label="Overflow to Gemini at concurrency 24"
      value={`+${fmt(100 * (lf.metrics.goodput_rps.mean / lo.metrics.goodput_rps.mean - 1), 0)}% goodput`}
      note={`${pct(gatewayByCell(e4).get(lf.cell_id)?.remoteShare ?? NaN)} of requests sent remote (E4)`} />);
  }
  const e5 = r["e5-gemini"];
  if (e5) {
    for (const c of e5.cells) {
      tiles.push(<StatTile key={`e5-${c.cell_id}`} label={`Gemini Flash-Lite, ${label(c.cell_id)}`}
        value={pct(c.metrics.accuracy.mean)} note={`${usd(c.metrics.usd_per_correct?.mean ?? NaN)} per correct answer (E5)`} />);
    }
  }
  const e6 = r["e6-gemini-caching"];
  if (e6) {
    for (const c of e6.cells) {
      tiles.push(<StatTile key={`e6-${c.cell_id}`} label={`${label(c.cell_id)}, cost per 1k requests`}
        value={`$${fmt(c.metrics.usd_per_1k_requests.mean, 2)}`} note={`${pct(c.metrics.cached_token_ratio.mean)} of prompt tokens cached (E6)`} />);
    }
  }
  return (
    <Section title="Overview"
      intro="Serving experiments on one Apple M4 Mac with Qwen3 models, llm-d, a Go AI gateway, and Gemini 3.5 Flash-Lite. Every number comes from a saved result bundle; intervals are 95% CIs over repeats (or item shards for E5).">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{tiles}</div>
    </Section>
  );
}

export function Engines({ data }: { data: Data }) {
  const gpu = data.results["e1-engines-gpu"], cpu = data.results["e1-engines-cpu"];
  return (
    <Section title="Engines (E1)" intro="vLLM (vllm-metal) vs llama.cpp on the Mac GPU with the same Qwen3-0.6B Q8_0 file, closed-loop concurrency sweep, 128 output tokens, prefix caching off; plus llama.cpp CPU-only.">
      {gpu ? (
        <Grid>
          <SweepCard r={gpu} title="Output throughput" subtitle="Tokens per second, all clients" metric="output_tokens_per_s" seriesKey="target.variant" />
          <SweepCard r={gpu} title="Time to first token, p95" metric="ttft_p95_ms" seriesKey="target.variant" format={ms} />
          <SweepCard r={gpu} title="Time per output token, p50" metric="tpot_p50_ms" seriesKey="target.variant" format={ms} />
          {cpu && <SweepCard r={cpu} title="llama.cpp CPU-only throughput" subtitle="Tokens per second" metric="output_tokens_per_s" seriesKey={null} />}
        </Grid>
      ) : <Missing what="E1" />}
    </Section>
  );
}

export function Caching({ data }: { data: Data }) {
  const e2 = data.results["e2-prefix-caching"], e6 = data.results["e6-gemini-caching"];
  const variant = (c: CellSummary) => param(c, "target.variant");
  return (
    <Section title="Caching (E2, E6)" intro="Local prefix caching on multi-turn RAG sessions (E2), and Gemini implicit vs explicit context caching vs the Batch API on shared-document sessions (E6).">
      {e2 ? (
        <Grid>
          <BarCard r={e2} title="TTFT p50 by cache setting" metric="ttft_p50_ms" by={variant} format={ms} />
          <BarCard r={e2} title="Prefix cache hit ratio" metric="prefix_cache_hit_ratio" by={variant} format={pct} />
        </Grid>
      ) : <Missing what="E2" />}
      {e6 ? (
        <Grid>
          <BarCard r={e6} title="Gemini cost per 1k requests" subtitle="Explicit includes cache storage for the full TTL" metric="usd_per_1k_requests" by={(c) => c.cell_id} format={(v) => `$${fmt(v, 2)}`} />
          <BarCard r={e6} title="Share of prompt tokens served from cache" metric="cached_token_ratio" by={(c) => c.cell_id} format={pct} />
          <BarCard r={e6} title="TTFT p50 (interactive arms)" metric="ttft_p50_ms" by={(c) => c.cell_id} format={ms} />
          <BarCard r={e6} title="Batch job turnaround" subtitle="Seconds from submit to results" metric="turnaround_s" by={(c) => c.cell_id} format={(v) => `${fmt(v, 0)} s`} />
        </Grid>
      ) : <Missing what="E6" />}
    </Section>
  );
}

export function Scheduling({ data }: { data: Data }) {
  const sim = data.results["e3-llmd-sim"], metal = data.results["e3-llmd-metal"];
  const profile = (c: CellSummary) => param(c, "target.scorer_profile");
  return (
    <Section title="llm-d scheduling (E3)" intro="llm-d EPP scorer profiles over 8 simulated workers with a small KV cache (mode A); real vllm-metal replicas (mode C) when published.">
      {sim ? (
        <Grid>
          <BarCard r={sim} title="Goodput at SLO" subtitle="Requests per second meeting TTFT and E2E targets" metric="goodput_rps" by={profile} />
          <BarCard r={sim} title="TTFT p95" metric="ttft_p95_ms" by={profile} format={ms} />
          <BarCard r={sim} title="Prefix cache hit ratio" metric="prefix_cache_hit_ratio" by={profile} format={pct} />
        </Grid>
      ) : <Missing what="E3 mode A" />}
      {metal ? (
        <Grid>
          <BarCard r={metal} title="Goodput at SLO (real Metal replicas)" metric="goodput_rps" by={(c) => Object.values(c.params).join(" · ")} />
        </Grid>
      ) : <Missing what="E3 mode C (real Metal replicas)" />}
    </Section>
  );
}

function hybridCards(r: ExperimentResult) {
  const econ = gatewayByCell(r);
  const share = sweep(r, "workload.concurrency", "target.policy", "slo_attainment").map((s) => ({
    name: label(s.name),
    points: s.points.map((p) => {
      const cell = r.cells.find((c) => param(c, "target.policy") === s.name && param(c, "workload.concurrency") === String(p.x));
      const e = cell && econ.get(cell.cell_id);
      return { x: p.x, mean: e ? e.remoteShare : NaN, lo: e ? e.remoteShare : NaN, hi: e ? e.remoteShare : NaN, n: p.n };
    }).filter((p) => Number.isFinite(p.mean)),
  }));
  return (
    <Grid>
      <SweepCard r={r} title="SLO attainment" subtitle="Share of requests within TTFT 1 s and E2E 3 s" metric="slo_attainment" seriesKey="target.policy" format={pct} />
      <SweepCard r={r} title="Goodput at SLO" subtitle="Requests per second" metric="goodput_rps" seriesKey="target.policy" />
      <SweepCard r={r} title="TTFT p99" metric="ttft_p99_ms" seriesKey="target.policy" format={ms} />
      <Card title="Share of requests sent to Gemini" subtitle="Mean over repeats (from gateway route counters)" table={sweepTable(share, "Concurrency", pct)}>
        <LineCI series={share} xLabel="Concurrency" format={pct} />
      </Card>
    </Grid>
  );
}

export function Hybrid({ data }: { data: Data }) {
  const e4 = data.results["e4-hybrid-gateway"], e4b = data.results["e4b-slo-overflow"];
  return (
    <Section title="Hybrid serving and cost (E4)" intro="Go AI gateway in front of llm-d over simulated local workers, overflowing to Gemini 3.5 Flash-Lite under a hard spend cap. E4b compares the fixed in-flight threshold with SLO-aware overflow.">
      {e4 ? hybridCards(e4) : <Missing what="E4" />}
      <h3 className="pt-2 text-base font-semibold">E4b: SLO-aware overflow</h3>
      {e4b ? hybridCards(e4b) : <Missing what="E4b" />}
    </Section>
  );
}

export function Quality({ data }: { data: Data }) {
  const runs = Object.values(data.results).filter((r) => r.name.startsWith("e5-"));
  if (!runs.length) return <Section title="Quality and cost (E5)"><Missing what="E5" /></Section>;
  const groups = (metric: string) => runs.map((r) => ({
    name: r.name.replace("e5-", ""),
    points: bars(r, metric, (c) => label(c.cell_id)),
  }));
  const rows = runs.flatMap((r) => r.cells.map((c) => {
    const m = (k: string) => c.metrics[k]?.mean ?? NaN;
    return [r.name.replace("e5-", ""), label(c.cell_id), pct(m("accuracy")), ms(m("latency_p50_ms")), ms(m("latency_p95_ms")),
      Number.isFinite(m("usd_per_correct")) ? `$${fmt(1000 * m("usd_per_correct"), 2)}` : "–"];
  }));
  return (
    <Section title="Quality and cost (E5)" intro="Each model on CRAG and HotpotQA RAG (judged by Gemini with a rubric calibrated against 100 reference labels, κ = 0.96), BFCL v3 single-turn tool calls (AST grader), and agentic HotpotQA over an MCP search/read server. Requests at concurrency 1; CIs over 5 item shards.">
      <Grid>
        <Card title="Accuracy by suite" table={barTable(groups("accuracy"), pct)}>
          <BarCI groups={groups("accuracy")} format={pct} />
        </Card>
        <Card title="Latency p50 per request (agent: per task)" table={barTable(groups("latency_p50_ms"), ms)}>
          <BarCI groups={groups("latency_p50_ms")} format={ms} />
        </Card>
      </Grid>
      <Card title="Summary" subtitle="$ per 1k correct answers excludes judge cost">
        <DataTable columns={["Model", "Suite", "Accuracy", "Latency p50", "Latency p95", "$ / 1k correct"]} rows={rows} />
      </Card>
    </Section>
  );
}

export function Provenance({ data }: { data: Data }) {
  return (
    <Section title="Run provenance" intro="The exact run behind every page: git commit (dirty means uncommitted code was used), finish time, host, and trial counts.">
      <Card title="Published runs">
        <DataTable
          columns={["Experiment", "Page", "Run", "Commit", "Clean tree", "Trials", "Finished"]}
          rows={data.index.map((e) => {
            const r = data.results[e.name];
            const tools = r.provenance.tools as Record<string, unknown>;
            return [e.name, e.page, e.run_id, e.git_commit.slice(0, 7), e.git_dirty ? "no" : "yes",
              `${tools.trials_completed}/${tools.trials_planned}`, e.finished_at.replace("T", " ").slice(0, 19)];
          })}
        />
      </Card>
      <Card title="Hosts">
        <DataTable
          columns={["Experiment", "Host details"]}
          rows={data.index.map((e) => [e.name, hostSummary(data.results[e.name].provenance.host)])}
        />
      </Card>
    </Section>
  );
}

function hostSummary(host: Record<string, unknown>): string {
  const keys = ["platform", "machine", "cpu_brand", "cpu_count", "memory_total_gb", "python_version"];
  return keys.filter((k) => host[k] !== undefined).map((k) => `${k}: ${String(host[k])}`).join(" · ") || "–";
}

