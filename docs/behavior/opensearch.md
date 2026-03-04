# OpenSearch Behavior Card

## Engine
- name: OpenSearch k-NN
- mode: server API
- adapter: `maxionbench.adapters.opensearch.OpenSearchAdapter`

## Visibility Semantics
- Adapter stages writes and applies them at `flush_or_commit`.
- Query path is commit-visible.

## Delete Semantics
- Delete is hard delete in adapter-visible state.
- Remote index is rewritten at commit in current implementation.

## Update Semantics
- Vector and payload updates are staged and materialized on commit.

## Compaction / Optimization
- `optimize_or_compact` maps to `_forcemerge` best-effort call.

## Persistence
- Persistence follows OpenSearch index and cluster durability settings.

## Limitations
- Filtered queries use exact fallback over committed adapter state.
- Current remote synchronization strategy favors deterministic semantics over incremental mutation efficiency.
