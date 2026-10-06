// Loads the static result files written by `npm run data` (maxionbench.harness.dashboard_export).
import { useEffect, useState } from "react";
import type { ExperimentResult } from "./types/result";

export interface IndexEntry {
  name: string;
  page: string;
  run_id: string;
  file: string;
  git_commit: string;
  git_dirty: boolean;
  finished_at: string;
}

export interface Data {
  index: IndexEntry[];
  results: Record<string, ExperimentResult>;
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path}: HTTP ${resp.status}`);
  return (await resp.json()) as T;
}

export async function loadData(base = "./data"): Promise<Data> {
  const { experiments } = await getJson<{ experiments: IndexEntry[] }>(`${base}/index.json`);
  const results = await Promise.all(experiments.map((e) => getJson<ExperimentResult>(`${base}/${e.file}`)));
  return { index: experiments, results: Object.fromEntries(experiments.map((e, i) => [e.name, results[i]])) };
}

export function useData(): { data: Data | null; error: string | null } {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    loadData().then(setData, (e: unknown) => setError(String(e)));
  }, []);
  return { data, error };
}
