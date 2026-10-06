# Apple Silicon Support

Portable benchmark runs target a single Apple Silicon Mac mini. The native portable engine set is:

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

Use `--skip-arch-check` only for local debugging. It is not valid for the 24-hour portable benchmark run because x86_64 emulation can dominate runtime.

## Serving engines (v0.3)

The v0.3 serving harness runs on the same Mac (Apple M4, 16 GB unified memory shared with other work):

- `vllm_metal`: vLLM on the Apple GPU through vllm-metal, installed in its own virtual environment
  (default `~/.venv-vllm-metal`). GGUF loading needs the `gguf` package, and only Q8_0, Q4_0, and Q4_1
  GGUF files load (Q4_K_M fails on its Q6_K embedding tensor). `--enable-prompt-tokens-details` is
  passed so responses report cached prompt tokens; the server exposes vLLM's full Prometheus metrics.
- `llamacpp_replicas`: `llama-server` with all layers on Metal (`-ngl 999`) or CPU only
  (`--device none -ngl 0 --no-op-offload`).
- `llmd`: the llm-d endpoint picker and Envoy run as Linux containers in Docker Desktop and reach
  host workers at Docker Desktop's host gateway (192.168.65.254), so workers stay bound to 127.0.0.1.
  `llm-d-inference-sim` workers need `POD_IP` set when `--enable-kvcache` is on.

Because memory is shared, the harness checks available memory before starting an engine
(`quiet_host.min_available_gb`), refuses a port that is already in use, and records foreign
inference processes (host or container) running during each trial. Run one large engine at a time.
