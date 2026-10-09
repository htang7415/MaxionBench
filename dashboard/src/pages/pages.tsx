import type { ReactNode } from "react";
import { BarCI, Card, DataTable, LineCI, StatTile, barTable, sweepTable } from "../components/charts";
import type { Data } from "../data";
import { bars, byDraw, fmt, gatewayByCell, param, sweep } from "../lib/shape";
import type { CellSummary, ExperimentResult } from "../types/result";

const LABELS: Record<string, string> = {
  vllm_metal: "vLLM (Metal)",
  llamacpp_metal: "llama.cpp (Metal)",
  vllm_apc_on: "vLLM on",
  vllm_apc_off: "vLLM off",
  llamacpp_cache_on: "llama.cpp on",
  llamacpp_cache_off: "llama.cpp off",
  local_only: "Local only",
  local_first: "Local first (fixed threshold)",
  local_first_slo: "Local first (SLO-aware)",
  remote_only: "Remote only",
  "e1-engines-cpu": "llama.cpp (CPU)",
  rag_crag: "CRAG (given snippets)",
  rag_hotpot: "HotpotQA (gold context)",
  bfcl: "BFCL tool calls",
  agent: "Agentic HotpotQA",
  implicit: "Implicit cache",
  explicit: "Explicit cache",
  batch: "Batch API",
  "gateway-window+cache": "gateway window+cache",
};
const label = (v: string) => LABELS[v] ?? v;
const ms = (v: number) => `${fmt(v, 0)} ms`;
const pct = (v: number) => `${fmt(100 * v, 0)}%`;
const per1k = (v: number) => `$${fmt(1000 * v, 2)}`;
const usd = (v: number) => `$${fmt(v, 3)}`;
const kTok = (v: number) => `${fmt(v / 1000, 0)}k`;
const signedPct = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), 0)}%`;

function relabel<T extends { name: string }>(series: T[]): T[] {
  return series.map((s) => ({ ...s, name: label(s.name) }));
}

function Missing({ what }: { what: string }) {
  return <p className="text-sm" style={{ color: "var(--muted)" }}>No published run for {what} yet.</p>;
}

function Section({ title, eyebrow, intro, children }: { title: string; eyebrow?: string; intro?: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <div>
        {eyebrow && <div className="text-[12px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>{eyebrow}</div>}
        <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
        {intro && <p className="mt-2 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{intro}</p>}
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
      <LineCI series={series} xLabel="Concurrency" format={format} axisFormat={format === pct ? pct : undefined} />
    </Card>
  );
}

function BarCard({ r, title, subtitle, metric, by, format, axisFormat }: {
  r: ExperimentResult;
  title: string;
  subtitle?: string;
  metric: string;
  by: (c: CellSummary) => string;
  format?: (v: number) => string;
  axisFormat?: (v: number) => string;
}) {
  const groups = [{ name: title, points: bars(r, metric, (c) => label(by(c))) }];
  return (
    <Card title={title} subtitle={subtitle} table={barTable(groups, format)}>
      <BarCI groups={groups} format={format} axisFormat={axisFormat ?? (format === pct ? pct : undefined)} />
    </Card>
  );
}

/** K9: % change in recomputed prefill vs full history, per paired draw (window+cache first). */
function k9Recompute(data: Data, all: boolean) {
  const k9 = data.results["k9-gateway-context-qwen3-8b"], k9b = data.results["k9b-gateway-mask-min-growth-qwen3-8b"];
  if (!k9) return [];
  return byDraw([
    { result: k9, arms: { "window+cache": "window+cache" } },
    ...(all && k9b ? [{ result: k9b, arms: { "mask+cache": "mask+cache", "mask+cache+pause": "mask+cache, idle trim" } }] : []),
    ...(all ? [{ result: k9, arms: { "mask+cache": "mask, no min_growth" } }] : []),
  ], "recomputed_tokens_per_request", "arm", "day", "off");
}

function K9RecomputeCard({ data, all }: { data: Data; all: boolean }) {
  const draws = k9Recompute(data, all);
  // one arm: one bar per paired draw in a single color; several arms: grouped by arm, one color per draw
  const series = all ? draws : [{ name: "window+cache vs full history", points: draws.map((d) => ({ ...d.points[0], x: d.name })) }];
  return (
    <Card title="Prefill recomputed vs full history (K9)"
      subtitle="% change per paired draw, Copilot traffic through the gateway onto vllm-metal Qwen3-8B; recomputed tokens repeat exactly across reruns"
      table={barTable(series, signedPct)}>
      <BarCI groups={series} format={signedPct} axisFormat={signedPct} />
    </Card>
  );
}

const FINDINGS = [
  { stat: "57k tokens", head: "Production agents send long, tool-heavy prompts",
    body: "Median prompt across 9.1M GitHub Copilot coding-agent calls; tool output is 48% of prompt tokens, and only 8.6% of sessions ever trim.", href: "#/context" },
  { stat: "up to 2.6×", head: "Naive trimming fights the prefix cache",
    body: "Rewriting earlier messages makes the server recompute more prefill despite sending 50–70% fewer tokens (Copilot replays, ample KV memory).", href: "#/context" },
  { stat: "−53%", head: "Cache-aware trimming wins on both sides",
    body: "Append until a token budget, then trim once: less prefill under KV pressure (K6), 29% lower Gemini cost and no accuracy loss (C1).", href: "#/context" },
  { stat: "17% → 92%", head: "In the gateway, every client gets it",
    body: "Requests under 5 s to first token on an overloaded Qwen3-8B engine (K9); in front of Gemini, half the prompt tokens with accuracy 52% vs 44% (C2).", href: "#/gateway" },
];

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
        value={pct(c.metrics.accuracy.mean)} note={`${per1k(c.metrics.usd_per_correct?.mean ?? NaN)} per 1k correct answers (E5)`} />);
    }
  }
  const e6 = r["e6-gemini-caching"];
  if (e6) {
    for (const c of e6.cells) {
      tiles.push(<StatTile key={`e6-${c.cell_id}`} label={`${label(c.cell_id)}, cost per 1k requests`}
        value={`$${fmt(c.metrics.usd_per_1k_requests.mean, 2)}`} note={`${pct(c.metrics.cached_token_ratio.mean)} of prompt tokens cached (E6)`} />);
    }
  }
  const changes = k9Recompute(data, false).flatMap((x) => x.points.map((p) => p.mean));
  return (
    <div className="space-y-12">
      <section>
        <div className="text-[12px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>v0.5 · Latest release</div>
        <h2 className="mt-2 max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">Serving LLM agents efficiently</h2>
        <p className="mt-4 max-w-2xl text-base leading-relaxed" style={{ color: "var(--ink-2)" }}>
          Agents re-send a growing history every step, and engines are fast only when that history is already in the
          KV cache. MaxionBench measures the trade-off on production traces, real engines, and a paid API, and puts
          cache-aware context management into a Go gateway.
        </p>
        {changes.length > 0 && (
          <div className="mt-8 flex flex-wrap items-end gap-x-6 gap-y-2">
            <div className="text-5xl font-semibold tracking-tight md:text-6xl">
              {signedPct(Math.max(...changes))} to {signedPct(Math.min(...changes))}
            </div>
            <div className="max-w-sm pb-2 text-sm" style={{ color: "var(--ink-2)" }}>
              prefill recompute with gateway context management, in all {changes.length} paired runs of replayed GitHub
              Copilot traffic on Qwen3-8B (K9)
            </div>
          </div>
        )}
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Key findings</h3>
        <ol className="grid gap-3 sm:grid-cols-2">
          {FINDINGS.map((f, i) => (
            <li key={f.head}>
              <a href={f.href} className="block h-full rounded-xl border p-5"
                style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                <div className="text-xs font-medium" style={{ color: "var(--muted)" }}>{String(i + 1).padStart(2, "0")}</div>
                <div className="mt-2 text-3xl font-semibold tracking-tight">{f.stat}</div>
                <div className="mt-2 font-medium">{f.head}</div>
                <p className="mt-1 text-sm leading-relaxed" style={{ color: "var(--ink-2)" }}>{f.body}</p>
                <div className="mt-3 text-sm font-medium">See the data →</div>
              </a>
            </li>
          ))}
        </ol>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">The v0.5 result</h3>
        <K9RecomputeCard data={data} all={false} />
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold">Serving baselines (v0.3)</h3>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{tiles}</div>
      </section>
    </div>
  );
}

export function Engines({ data }: { data: Data }) {
  const gpu = data.results["e1-engines-gpu"], cpu = data.results["e1-engines-cpu"];
  return (
    <Section title="Engines (E1)" eyebrow="v0.3 · Serving" intro="vLLM (vllm-metal) vs llama.cpp on the Mac GPU with the same Qwen3-0.6B Q8_0 file, closed-loop concurrency sweep, 128 output tokens, prefix caching off; plus llama.cpp CPU-only.">
      {gpu ? (
        <Grid>
          <SweepCard r={gpu} title="Output throughput" subtitle="Tokens per second, all clients" metric="output_tokens_per_s" seriesKey="target.variant" />
          <SweepCard r={gpu} title="Time to first token, p95" subtitle="Milliseconds" metric="ttft_p95_ms" seriesKey="target.variant" format={ms} />
          <SweepCard r={gpu} title="Time per output token, p50" subtitle="Milliseconds" metric="tpot_p50_ms" seriesKey="target.variant" format={ms} />
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
    <Section title="Caching (E2, E6)" eyebrow="v0.3 · Serving" intro="Local prefix caching on multi-turn RAG sessions (E2), and Gemini implicit vs explicit context caching vs the Batch API on shared-document sessions (E6).">
      {e2 ? (
        <Grid>
          <BarCard r={e2} title="TTFT p50 by cache setting" subtitle="Milliseconds; vLLM automatic prefix caching and llama.cpp prompt cache, on vs off" metric="ttft_p50_ms" by={variant} format={ms} />
          <BarCard r={e2} title="Prefix cache hit ratio" metric="prefix_cache_hit_ratio" by={variant} format={pct} />
        </Grid>
      ) : <Missing what="E2" />}
      {e6 ? (
        <Grid>
          <BarCard r={e6} title="Gemini cost per 1k requests" subtitle="US dollars; explicit includes cache storage for the full TTL" metric="usd_per_1k_requests" by={(c) => c.cell_id} format={(v) => `$${fmt(v, 2)}`} />
          <BarCard r={e6} title="Share of prompt tokens served from cache" metric="cached_token_ratio" by={(c) => c.cell_id} format={pct} />
          <BarCard r={e6} title="TTFT p50 (interactive arms)" subtitle="Milliseconds" metric="ttft_p50_ms" by={(c) => c.cell_id} format={ms} />
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
    <Section title="llm-d scheduling (E3)" eyebrow="v0.3 · Serving" intro="llm-d EPP scorer profiles over 8 simulated workers with a small KV cache (mode A); real vllm-metal replicas (mode C) when published. Simulated latencies use vLLM's per-token TTFT at concurrency 4 plus the simulator's own load factor, so absolute latencies likely overstate load; compare profiles, not milliseconds.">
      {sim ? (
        <Grid>
          <BarCard r={sim} title="Goodput at SLO" subtitle="Requests per second meeting TTFT and E2E targets" metric="goodput_rps" by={profile} />
          <BarCard r={sim} title="TTFT p95" subtitle="Milliseconds" metric="ttft_p95_ms" by={profile} format={ms} />
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
      <SweepCard r={r} title="TTFT p99" subtitle="Milliseconds" metric="ttft_p99_ms" seriesKey="target.policy" format={ms} />
      <Card title="Share of requests sent to Gemini" subtitle="Mean over repeats (from gateway route counters)" table={sweepTable(share, "Concurrency", pct)}>
        <LineCI series={share} xLabel="Concurrency" format={pct} axisFormat={pct} />
      </Card>
    </Grid>
  );
}

export function Hybrid({ data }: { data: Data }) {
  const e4 = data.results["e4-hybrid-gateway"], e4b = data.results["e4b-slo-overflow"];
  return (
    <Section title="Hybrid serving and cost (E4)" eyebrow="v0.3 · Serving" intro="Go AI gateway in front of llm-d over simulated local workers, overflowing to Gemini 3.5 Flash-Lite under a hard spend cap. Simulated local latency likely counts load twice (loaded per-token rate plus the simulator's load factor), which favors overflow; Gemini latency and spend are real. E4b compares the fixed in-flight threshold with predicted-wait overflow (in-flight × recent time per completed request, an end-to-end estimate compared against the TTFT target): it triggers earlier, but gains stay within the CIs.">
      {e4 ? hybridCards(e4) : <Missing what="E4" />}
      <h3 className="pt-2 text-base font-semibold">E4b: predicted-wait (SLO-aware) overflow</h3>
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
    <Section title="Quality and cost (E5)" eyebrow="v0.3 · Serving" intro="Question answering with provided context — CRAG with the dataset's search snippets, HotpotQA with its gold paragraphs plus distractors (no retrieval is measured) — judged by Gemini against a rubric; BFCL v3 single-turn tool calls (AST grader); and agentic HotpotQA over an MCP search/read server, where the model must find the evidence itself. Judge vs 100 reference labels: κ = 0.96 overall, 0.86 on answered items; it missed 2 of 9 wrong answers, so judged accuracy is a slight upper bound. Requests at concurrency 1; CIs over 5 item shards.">
      <Grid>
        <Card title="Accuracy by suite" table={barTable(groups("accuracy"), pct)}>
          <BarCI groups={groups("accuracy")} format={pct} axisFormat={pct} />
        </Card>
        <Card title="Latency p50" subtitle="Milliseconds per request; agentic: per task (several model calls)" table={barTable(groups("latency_p50_ms"), ms)}>
          <BarCI groups={groups("latency_p50_ms")} format={ms} />
        </Card>
      </Grid>
      <Card title="Summary" subtitle="$ per 1k correct answers excludes judge cost">
        <DataTable columns={["Model", "Suite", "Accuracy", "Latency p50", "Latency p95", "$ / 1k correct"]} rows={rows} />
      </Card>
    </Section>
  );
}

export function ContextPolicies({ data }: { data: Data }) {
  const c1 = data.results["c1-context-policies"], k6 = data.results["k6-copilot-context-policies"];
  const k7 = data.results["k7-llmd-copilot-context-policies"], k8 = data.results["k8-vllm-metal-copilot-context-policies"];
  const policy = (c: CellSummary) => c.cell_id;
  const trace = (c: CellSummary) => param(c, "trace");
  const k6Groups = k6 ? [...new Set(k6.cells.map((c) => param(c, "capacity_tokens")))].map((cap) => ({
    name: `${fmt(Number(cap) / 1000, 0)}k tokens of KV per replica`,
    points: bars({ ...k6, cells: k6.cells.filter((c) => param(c, "capacity_tokens") === cap) }, "recomputed_tokens_per_request", trace),
  })) : [];
  return (
    <Section title="Context policies for agents (v0.4)" eyebrow="v0.4 · Agents" intro="What an agent sends the model each step: the whole history (full), or a trimmed view. C1: a Gemini 3.5 Flash-Lite agent on 50 BrowseComp-Plus tasks per policy, graded by the calibrated judge; CIs over 5 task shards. K6–K8: GitHub Copilot coding-agent sessions replayed under each policy, measuring prefill tokens the server recomputes.">
      {c1 ? (
        <Grid>
          <BarCard r={c1} title="Accuracy (C1)" subtitle="Judge-graded; at 50 tasks only summarize is significant vs full after Holm correction" metric="accuracy" by={policy} format={pct} />
          <BarCard r={c1} title="Gemini cost per task (C1)" subtitle="Billed: cached and uncached input, output, reasoning, summaries" metric="cost_usd_per_task" by={policy} format={usd} />
          <BarCard r={c1} title="Share of prompt tokens served from cache (C1)" subtitle="Rewriting earlier messages loses Gemini's implicit cache" metric="cached_share" by={policy} format={pct} />
          <BarCard r={c1} title="Cost per correct answer (C1)" metric="usd_per_correct" by={policy} format={usd} />
        </Grid>
      ) : <Missing what="C1" />}
      {k6 ? (
        <Card title="Prefill recomputed per request (K6, offline KV simulator)" subtitle="Tokens; 4 replicas, llm-d-style prefix+load routing, mean of 3 repeats"
          table={barTable(k6Groups, (v) => fmt(v, 0))}>
          <BarCI groups={k6Groups} format={(v) => fmt(v, 0)} />
        </Card>
      ) : <Missing what="K6" />}
      <Grid>
        {k7 ? <BarCard r={k7} title="Prefill recomputed per request (K7, live llm-d)" subtitle="Tokens; llm-d EPP + Envoy over 4 inference-sim workers, tight KV" metric="recomputed_tokens_per_request" by={trace} format={(v) => fmt(v, 0)} /> : <Missing what="K7" />}
        {k8 ? <BarCard r={k8} title="Prefill recomputed per request (K8, vllm-metal)" subtitle="Tokens; one Qwen3-0.6B replica on the Mac GPU, prompts at 1/32 scale" metric="recomputed_tokens_per_request" by={trace} format={(v) => fmt(v, 0)} /> : <Missing what="K8" />}
      </Grid>
    </Section>
  );
}

export function GatewayContext({ data }: { data: Data }) {
  const k9 = data.results["k9-gateway-context-qwen3-8b"];
  const c1 = data.results["c1-context-policies"], c2a = data.results["c2a-gateway-context"], c2b = data.results["c2b-gateway-context"];
  const slo = k9 ? byDraw([{ result: k9, arms: { off: "full history (off)", "window+cache": "window+cache" } }],
    "slo_attainment", "arm", "day") : [];
  const policy = (c: CellSummary) => label(c.cell_id);
  const c1Tasks = c1 && c2a ? [{
    name: "Accuracy on the C1 tasks",
    points: [...bars({ ...c1, cells: c1.cells.filter((c) => ["full", "window+cache"].includes(c.cell_id)) }, "accuracy",
      (c) => (c.cell_id === "full" ? "full" : "in-agent window+cache")), ...bars(c2a, "accuracy", policy)],
  }] : [];
  return (
    <Section title="Context management in the gateway (v0.5)" eyebrow="v0.5 · Gateway" intro="The Go gateway keeps each agent session's view append-only, so the prefix cache keeps hitting, and trims (keeps the last N exchanges, or masks old tool results) only past a token budget or when the history is new or rewritten. K9: Copilot sessions replayed as full chat histories through the gateway onto one vllm-metal Qwen3-8B replica; arms pair with full history (off) on the same schedule. C2: a Gemini agent behind the gateway.">
      {k9 ? (
        <Grid>
          <K9RecomputeCard data={data} all />
          <Card title="Requests with TTFT under 5 s (K9)" subtitle="Per draw; on the heavier Saturday draw (2) full history overloaded the engine (TTFT p50 80 s vs 1.0 s)"
            table={barTable(slo, pct)}>
            <BarCI groups={slo} format={pct} axisFormat={pct} />
          </Card>
        </Grid>
      ) : <Missing what="K9" />}
      {c2b ? (
        <Grid>
          <BarCard r={c2b} title="Accuracy, 50 new tasks (C2b)" subtitle="Judge-graded; with C2a, 52% vs 44% over 100 paired tasks (Holm p 0.51)" metric="accuracy" by={policy} format={pct} />
          <BarCard r={c2b} title="Prompt tokens per task (C2b)" subtitle="Thousands of tokens billed per task" metric="prompt_tokens_per_task" by={policy} format={kTok} axisFormat={kTok} />
          <BarCard r={c2b} title="Gemini cost per task (C2b)" metric="cost_usd_per_task" by={policy} format={usd} />
          <BarCard r={c2b} title="Share of prompt tokens served from cache (C2b)" subtitle="Each trim loses Gemini's implicit cache" metric="cached_share" by={policy} format={pct} />
        </Grid>
      ) : <Missing what="C2b" />}
      {c1Tasks.length > 0 && (
        <Card title="Gateway vs in-agent trimming on the C1 tasks (C1, C2a)" subtitle="Judge-graded accuracy; same 50 tasks and agent"
          table={barTable(c1Tasks, pct)}>
          <BarCI groups={c1Tasks} format={pct} axisFormat={pct} />
        </Card>
      )}
    </Section>
  );
}

export function Provenance({ data }: { data: Data }) {
  return (
    <Section title="Run provenance" eyebrow="Data" intro="The exact run behind every page: git commit (dirty means uncommitted code was used), finish time, host, and trial counts.">
      <Card title="Published runs">
        <DataTable
          columns={["Experiment", "Page", "Run", "Commit", "Clean tree", "Trials", "Finished"]}
          rows={data.index.map((e) => {
            const r = data.results[e.name];
            const tools = r.provenance.tools as Record<string, unknown>;
            return [e.name, e.page, e.run_id, e.git_commit.slice(0, 7), e.git_dirty ? "no" : "yes",
              tools.trials_planned === undefined ? String(tools.trials_completed ?? "–") : `${tools.trials_completed}/${tools.trials_planned}`, e.finished_at.replace("T", " ").slice(0, 19)];
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
  const parts = [
    host.apple_silicon_model,
    host.cpu_count_logical !== undefined ? `${host.cpu_count_logical} CPU threads` : undefined,
    typeof host.total_memory_bytes === "number" ? `${fmt(host.total_memory_bytes / 2 ** 30, 0)} GB` : undefined,
    host.macos_version !== undefined ? `macOS ${host.macos_version}` : host.platform,
    host.python_version !== undefined ? `Python ${host.python_version}` : undefined,
    host.docker_version,
  ];
  return parts.filter((v) => v !== undefined && v !== null).map(String).join(" · ") || "–";
}

