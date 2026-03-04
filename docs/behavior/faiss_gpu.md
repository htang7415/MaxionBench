# FAISS GPU Behavior Card

## Engine
- name: FAISS GPU
- mode: in-process GPU
- adapter: `maxionbench.adapters.faiss_gpu.FaissGpuAdapter`

## Visibility Semantics
- Inherits staged-write semantics from FAISS CPU adapter (`flush_or_commit` visibility).

## Delete Semantics
- Hard delete in adapter-visible state.

## Update Semantics
- Vector/payload updates are staged and committed via `flush_or_commit`.

## Compaction / Optimization
- `optimize_or_compact` rebuilds and re-transfers index to GPU.

## Persistence
- Process-local state only.

## Limitations
- Requires FAISS build with GPU support.
- Filtered queries use exact scoring fallback over committed records.
