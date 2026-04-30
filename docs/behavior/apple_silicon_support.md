# Apple Silicon Support

Portable paper runs target a single Apple Silicon Mac mini. The native portable engine set is:

- `faiss-cpu`: host Python process, native arm64 when installed from an arm64 Python environment.
- `lancedb-inproc`: host Python process, native arm64 when installed from an arm64 Python environment.
- `lancedb-service`: local service mode; use `MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI` for the portable path.
- `pgvector`: Docker service, default image `pgvector/pgvector:0.8.2-pg16-trixie`.
- `qdrant`: Docker service, default image `qdrant/qdrant:v1.17.1`.

`maxionbench services up` validates Docker image manifests against the host architecture before starting containers. On Apple Silicon it requires `linux/arm64`; failures should be fixed with the matching `MAXIONBENCH_*_IMAGE` override instead of accepting QEMU emulation for reported runs.

Useful overrides:

```bash
export MAXIONBENCH_QDRANT_IMAGE=qdrant/qdrant:v1.17.1
export MAXIONBENCH_PGVECTOR_IMAGE=pgvector/pgvector:0.8.2-pg16-trixie
export MAXIONBENCH_LANCEDB_SERVICE_INPROC_URI="$PWD/artifacts/lancedb/service"
```

Use `--skip-arch-check` only for local debugging. It is not valid for the 24-hour portable paper run because x86_64 emulation can dominate runtime.
