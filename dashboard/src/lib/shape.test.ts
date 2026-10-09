import { describe, expect, it } from "vitest";
import type { CellSummary, ExperimentResult, TrialResult } from "../types/result";
import { bars, byDraw, fmt, niceTicks, gatewayByCell, gatewayEconomics, sweep, toRows } from "./shape";

const ci = (mean: number) => ({ mean, ci_low: mean - 1, ci_high: mean + 1, std: 0.5, n: 3 });

function cell(id: string, params: Record<string, unknown>, metrics: Record<string, number>): CellSummary {
  return {
    cell_id: id,
    params,
    n_ok: 3,
    n_failed: 0,
    metrics: Object.fromEntries(Object.entries(metrics).map(([k, v]) => [k, ci(v)])),
  };
}

function trial(cellId: string, routes: Record<string, number>, spend: number): TrialResult {
  return {
    trial_id: `${cellId}-t`, cell_id: cellId, repeat: 0, seed: 1, status: "ok", started_at: "", duration_s: 1,
    host_load_1m_before: 0, quiet_host_ok: true, metrics: {}, requests_per_endpoint: [],
    target: { collected: { gateway: { route_decisions: routes, remote_spend_usd: spend } } }, error: null,
  };
}

const result = {
  name: "e4",
  cells: [
    cell("a", { "target.policy": "local_first", "workload.concurrency": 12 }, { goodput_rps: 11 }),
    cell("b", { "target.policy": "local_first", "workload.concurrency": 4 }, { goodput_rps: 7 }),
    cell("c", { "target.policy": "remote_only", "workload.concurrency": 4 }, { goodput_rps: 5 }),
  ],
  trials: [trial("a", { "local:capacity": 75, "remote:slo_predicted": 25 }, 0.02), trial("c", { "remote:policy": 50 }, 0.01)],
} as unknown as ExperimentResult;

describe("sweep", () => {
  it("groups by series and sorts numeric x", () => {
    const s = sweep(result, "workload.concurrency", "target.policy", "goodput_rps");
    expect(s.map((x) => x.name)).toEqual(["local_first", "remote_only"]);
    expect(s[0].points.map((p) => p.x)).toEqual([4, 12]);
    expect(s[0].points[0]).toMatchObject({ mean: 7, lo: 6, hi: 8 });
  });

  it("builds aligned rows with CI ranges", () => {
    const rows = toRows(sweep(result, "workload.concurrency", "target.policy", "goodput_rps"));
    expect(rows).toEqual([
      { x: 4, local_first: 7, local_first_ci: [6, 8], remote_only: 5, remote_only_ci: [4, 6] },
      { x: 12, local_first: 11, local_first_ci: [10, 12] },
    ]);
  });

  it("skips cells without the metric", () => {
    expect(sweep(result, "workload.concurrency", "target.policy", "missing")).toEqual([]);
  });
});

describe("gateway economics", () => {
  it("computes remote share and $ per 1k requests", () => {
    expect(gatewayEconomics(result.trials[0])).toEqual({ remoteShare: 0.25, usdPer1k: 0.2 });
    const byCell = gatewayByCell(result);
    expect(byCell.get("c")).toEqual({ remoteShare: 1, usdPer1k: 0.2 });
    expect(byCell.has("b")).toBe(false);
  });
});

describe("bars and fmt", () => {
  it("labels one bar per cell", () => {
    expect(bars(result, "goodput_rps", (c) => c.cell_id).map((p) => p.x)).toEqual(["a", "b", "c"]);
  });
  it("formats numbers compactly", () => {
    expect(fmt(1234.5)).toBe("1,235");
    expect(fmt(0.12345, 3)).toBe("0.123");
    expect(fmt(NaN)).toBe("–");
  });
});

describe("byDraw", () => {
  const run = (name: string, arms: string[], values: number[][]) => ({
    name,
    cells: arms.map((a) => cell(a, { day: "sat", arm: a }, {})),
    trials: values.flatMap((row, repeat) => row.map((v, i) => ({
      ...trial(arms[i], {}, 0), repeat, metrics: { recomputed: v },
    }))),
  }) as unknown as ExperimentResult;

  it("pairs arms with the baseline of the same repeat, across results", () => {
    const k9 = run("k9", ["off", "window"], [[200, 150], [100, 90]]);
    const k9b = run("k9b", ["mask"], [[260], [130]]);
    const series = byDraw([{ result: k9, arms: { window: "window" } }, { result: k9b, arms: { mask: "mask" } }],
      "recomputed", "arm", "day", "off");
    expect(series.map((s) => s.name)).toEqual(["sat, draw 1", "sat, draw 2"]);
    expect(series[0].points.map((p) => [p.x, Math.round(p.mean)])).toEqual([["window", -25], ["mask", 30]]);
    expect(series[1].points.map((p) => [p.x, Math.round(p.mean)])).toEqual([["window", -10], ["mask", 30]]);
  });

  it("returns raw values in the arms' order without a baseline", () => {
    const k9 = run("k9", ["window", "off"], [[150, 200]]);
    const series = byDraw([{ result: k9, arms: { off: "off", window: "window" } }], "recomputed", "arm", "day");
    expect(series[0].points.map((p) => [p.x, p.mean])).toEqual([["off", 200], ["window", 150]]);
  });
});

describe("niceTicks", () => {
  it("covers the data and zero with round steps", () => {
    expect(niceTicks(-38.6, -14.2)).toEqual([-40, -30, -20, -10, 0]);
    expect(niceTicks(0, 0.92)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(niceTicks(0, 414_000)).toEqual([0, 200_000, 400_000, 600_000]);
    expect(niceTicks(-41, 84)).toEqual([-50, 0, 50, 100]);
  });
});
