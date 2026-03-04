# Weaviate Behavior Card

## Engine
- name: Weaviate
- mode: server API
- adapter: `maxionbench.adapters.weaviate.WeaviateAdapter`

## Visibility Semantics
- Writes are staged locally and become visible after `flush_or_commit`.
- Adapter query path reads committed adapter-visible state.

## Delete Semantics
- Delete is hard delete in adapter-visible state.
- Remote class objects are rebuilt on commit in current implementation.

## Update Semantics
- Vector and payload updates are staged and applied on commit.

## Compaction / Optimization
- `optimize_or_compact` is a no-op; Weaviate manages internal maintenance.

## Persistence
- Persistence depends on Weaviate deployment and storage configuration.

## Limitations
- Current query path is deterministic exact fallback over committed adapter state.
- Commit currently rewrites class objects for deterministic semantics.
