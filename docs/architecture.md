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

## Generated state

`dataset/`, `artifacts/`, `results/`, and `release/` contain local or generated state. Source, tests, configs, and documentation remain in Git.
