/* Generated from maxionbench/harness/result.schema.json by `npm run types`; do not edit. */

export interface ExperimentResult {
  cells: CellSummary[];
  description: string;
  name: string;
  provenance: Provenance;
  run_id: string;
  schema_version: "maxionbench-harness-result-v1";
  spec: {
    [k: string]: unknown;
  };
  trials: TrialResult[];
}
export interface CellSummary {
  cell_id: string;
  metrics: {
    [k: string]: MetricCI;
  };
  n_failed: number;
  n_ok: number;
  params: {
    [k: string]: unknown;
  };
}
export interface MetricCI {
  ci_high: number;
  ci_low: number;
  mean: number;
  n: number;
  std: number;
}
export interface Provenance {
  finished_at: string;
  git_commit: string;
  git_dirty: boolean;
  host: {
    [k: string]: unknown;
  };
  spec_fingerprint: string;
  started_at: string;
  tools: {
    [k: string]: unknown;
  };
}
export interface TrialResult {
  cell_id: string;
  duration_s: number;
  error: string | null;
  host_load_1m_before: number;
  metrics: {
    [k: string]: number;
  };
  quiet_host_ok: boolean;
  repeat: number;
  requests_per_endpoint: number[];
  seed: number;
  started_at: string;
  status: "ok" | "failed";
  target: {
    [k: string]: unknown;
  };
  trial_id: string;
}
