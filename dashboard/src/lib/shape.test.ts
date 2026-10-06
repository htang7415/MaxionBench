import { describe, expect, it } from "vitest";
import type { CellSummary, ExperimentResult, TrialResult } from "../types/result";
import { bars, fmt, gatewayByCell, gatewayEconomics, sweep, toRows } from "./shape";

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
