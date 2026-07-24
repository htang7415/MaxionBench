# pgvector Behavior Card

## Engine
- name: PostgreSQL + pgvector
- mode: server database
- adapter: `maxionbench.adapters.pgvector.PgVectorAdapter`

## Visibility Semantics
- Writes are issued on a writer connection and remain invisible to reads until `flush_or_commit`.
- Reads use a separate reader connection and observe committed state only.

## Delete Semantics
- Delete is hard delete at row level (`DELETE FROM ... WHERE id = ...`).
- `deleted_count` combines adapter-tracked deletes with `pg_stat_user_tables.n_dead_tup`.

## Update Semantics
- Vector updates use `UPDATE ... SET embedding = ...`.
- Payload updates merge JSON with `payload = payload || <jsonb>`.

## Compaction / Optimization
- `optimize_or_compact` maps to `VACUUM ANALYZE`.
- Behavior is blocking for the session executing it and may increase latency.

## Persistence
- Persistence follows PostgreSQL durability settings (WAL/fsync configuration).
- `flush_or_commit` executes transaction commit on the writer connection.

## Limitations
- Filter translation currently supports equality-style JSON containment only.
- This adapter requires `psycopg` and a PostgreSQL instance with `pgvector` extension available.
