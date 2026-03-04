# FAISS CPU Behavior Card

## Engine
- name: FAISS CPU
- mode: in-process
- adapter: `maxionbench.adapters.faiss_cpu.FaissCpuAdapter`

## Visibility Semantics
- Writes are staged and become visible after `flush_or_commit`.

## Delete Semantics
- Delete is hard delete in adapter-visible state.

## Update Semantics
- Vector and payload updates are staged and applied at commit.

## Compaction / Optimization
- `optimize_or_compact` rebuilds the index from committed state.

## Persistence
- No built-in persistence in this adapter; state is process-local.

## Limitations
- Filtered queries use exact scoring fallback over committed records.
- Resource stats are estimated from in-memory footprint.
