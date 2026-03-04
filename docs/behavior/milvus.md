# Milvus Behavior Card

## Engine
- name: Milvus
- mode: server API
- adapter: `maxionbench.adapters.milvus.MilvusAdapter`

## Visibility Semantics
- Adapter stages writes locally and applies them on `flush_or_commit`.
- Reads observe committed state from the adapter-visible snapshot.

## Delete Semantics
- Deletes are hard deletes in adapter-visible state.
- Remote collection is rewritten at commit in current implementation.

## Update Semantics
- Vector and payload updates are staged, then materialized on commit.

## Compaction / Optimization
- `optimize_or_compact` maps to collection flush behavior.
- No explicit blocking compaction command is issued by this adapter.

## Persistence
- Persistence depends on Milvus deployment/storage settings.
- Adapter-level visibility is commit-gated via `flush_or_commit`.

## Limitations
- Complex server-side filtering is not exposed; equality filtering is handled in exact fallback path.
- Current implementation prioritizes deterministic behavior over incremental remote mutation efficiency.
