MaxionBench is an open, reproducible benchmark suite for retrieval backends and RAG pipeline components used in agentic LLM systems.

Implemented v0.1 harness components include:
- adapter contract + conformance suite
- scenarios `calibrate_d3`, `s1`..`s6`
- adapters: `mock`, `qdrant`, `pgvector`, `milvus`, `weaviate`, `opensearch`, `lancedb-service`, `lancedb-inproc`, `faiss-cpu`, `faiss-gpu`
- D4 local bundle loader for pinned BEIR subsets + CRAG slice
- report bundle generation for milestone/final figures and core paper tables (T1-T4)
- phased warmup/steady-state execution controls (`phase_timing_mode`, phase request caps)
- strict output validation plus legacy stage-timing backfill tooling (`maxionbench validate`, `maxionbench migrate-stage-timing`)
