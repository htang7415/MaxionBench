# Qdrant Behavior Card

## Engine
- name: Qdrant
- mode: server API
- adapter: `maxionbench.adapters.qdrant.QdrantAdapter`

## Visibility Semantics
- Upserts/updates/deletes are issued with `wait=true`.
- In this adapter, operations are treated as visible after the request returns successfully.

## Delete Semantics
- API-level point deletion by explicit id list.
- Underlying storage cleanup/segment compaction is internal to Qdrant.

## Update Semantics
- Vector updates use `/points/vectors`.
- Payload updates use `/points/payload`.

## Compaction / Optimization
- No direct explicit `optimize_or_compact` endpoint is called by this adapter.
- Adapter exposes `optimize_or_compact` as a no-op and relies on engine background behavior.

## Persistence
- Persistence semantics depend on Qdrant deployment/storage configuration.
- This adapter does not alter cluster-level durability settings.

## Limitations
- Current filter translation supports equality matches only.
- Stats fields are mapped from available `/collections/{name}` response fields and may be partially zero depending on engine version/config.
- `deleted_count` is reported as `0` in this adapter because Qdrant does not expose a reliable deleted-points counter through the API path used here.
