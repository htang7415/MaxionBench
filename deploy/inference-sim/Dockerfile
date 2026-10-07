# llm-d-inference-sim v0.11.4 plus MaxionBench's CPU KV tier patch (cpu-kv-tier.patch):
# --cpu-kv-cache-size blocks of host-memory KV, written through when computed (like vLLM's
# OffloadingConnector), loaded back on a GPU miss after --cpu-kv-load-time-per-token each, announced
# with CPU-tier KV events, and counted in vllm:external_prefix_cache_hits_total. GPU eviction is O(1).
#   COPYFILE_DISABLE=1 tar -c --format ustar --no-mac-metadata --exclude "._*" -C deploy/inference-sim . \
#     | docker build -t maxionbench/llm-d-inference-sim:v0.11.4-cpukv -
FROM golang:1.26.6 AS builder
ARG TARGETOS
ARG TARGETARCH
ARG SIM_COMMIT=db3e12bdf8ccf500d93309dc73015fb811296404
WORKDIR /workspace
RUN git init -q . && git fetch -q --depth 1 https://github.com/llm-d/llm-d-inference-sim.git ${SIM_COMMIT} \
    && git checkout -q FETCH_HEAD
RUN go mod download
COPY cpu-kv-tier.patch /tmp/
RUN git apply /tmp/cpu-kv-tier.patch
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -o bin/llm-d-inference-sim cmd/llm-d-inference-sim/main.go

FROM gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=builder /workspace/bin/llm-d-inference-sim /app/llm-d-inference-sim
USER 65532:65532
ENTRYPOINT ["/app/llm-d-inference-sim"]
