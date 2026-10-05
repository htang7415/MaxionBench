# Architecture

MaxionBench is an installable benchmark CLI and Python package. It runs workloads against retrieval-engine adapters and writes auditable result bundles.

## Execution flow

```text
CLI
 └─ configuration and run matrix
     └─ scenario
         └─ adapter contract
             └─ retrieval engine

run artifacts
 ├─ schema validation
 ├─ promotion gates
 ├─ reporting
 └─ archive
```

## Boundaries

- Adapters translate the shared contract; they do not define benchmark policy.
- Scenarios define workload behavior; they do not format reports.
- Orchestration schedules work and records provenance; it does not implement engine behavior.
- Schemas are public compatibility boundaries.
- Reports consume validated artifacts and must not mutate source runs.

Large modules should be split only behind characterization tests. Preserve public entry points while moving implementation into smaller modules.

## v0.3 serving harness

```text
experiments/*.yaml
 └─ harness.spec ─► harness.planner ─► harness.runner
                                         ├─ harness.targets   (system under test: lifecycle, endpoints, picker)
                                         ├─ harness.workloads (request streams + load-generation settings)
                                         ├─ rag.loadgen       (open-loop Poisson, admission control, fallback)
                                         └─ harness.budget    (reserve/commit ledger for paid APIs)
                                       ─► results.json (+ requests.jsonl, spec.yaml, logs/)
                                       ─► harness.compare / dashboard
```

- Targets own lifecycle and expose OpenAI-compatible endpoints; they never define workload or SLO policy.
- Workloads produce request streams; they never know which engine serves them.
- `harness/result.schema.json` is generated from the result dataclasses and is the dashboard contract;
  `python -m maxionbench.harness schema --check` fails when it is stale.
- Latency is measured from the scheduled arrival time, so queueing delay is never hidden.
- Paid-API spend goes through `BudgetLedger.reserve` before any request; the ledger lives outside the
  repo so cleaning artifacts never resets spend.
- API keys are loaded only at runtime via `harness.secrets`; results record key presence, not value.

## Generated state

`dataset/`, `artifacts/`, `results/`, and `release/` contain local or generated state. Source, tests, configs, and documentation remain in Git.
