# LanceDB Behavior Card

## Engine
- name: LanceDB
- adapter modules:
  - `maxionbench.adapters.lancedb_service.LanceDbServiceAdapter`
  - `maxionbench.adapters.lancedb_inproc.LanceDbInprocAdapter`

## Mode: lancedb-service

### Visibility Semantics
- HTTP mode delegates visibility semantics to the service implementation.
- Optional `inproc_uri` mode delegates to in-process adapter semantics.

### Delete Semantics
- Service mode depends on service contract implementation.
- In delegated inproc mode, delete is hard delete at commit.

### Update Semantics
- Service mode depends on service implementation.
- In delegated inproc mode, updates are staged until `flush_or_commit`.

### Compaction / Optimization
- Service mode calls `/optimize` endpoint.
- Delegated inproc mode uses flush-based maintenance behavior.

### Persistence
- Service mode persistence depends on service deployment/storage.
- Delegated inproc mode persists in configured LanceDB URI.

### Limitations
- HTTP mode expects MaxionBench adapter HTTP contract endpoints.

## Mode: lancedb-inproc

### Visibility Semantics
- Writes are staged locally and become visible on `flush_or_commit`.

### Delete Semantics
- Delete is hard delete in adapter-visible state.

### Update Semantics
- Vector/payload updates are staged and applied on commit.

### Compaction / Optimization
- `optimize_or_compact` currently maps to flush.

### Persistence
- Data is persisted at configured local LanceDB URI.

### Limitations
- Query path uses deterministic exact scoring over committed adapter state.
