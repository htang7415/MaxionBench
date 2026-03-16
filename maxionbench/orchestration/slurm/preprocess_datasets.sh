#!/usr/bin/env bash
set -euo pipefail
SLURM_DIR="${MAXIONBENCH_SLURM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=/dev/null
source "${SLURM_DIR}/common.sh"

DATASET_ROOT="${MAXIONBENCH_DATASET_ROOT:-${ROOT_DIR}/dataset}"
PROCESSED_ROOT="${MAXIONBENCH_PROCESSED_ROOT:-${DATASET_ROOT}/processed}"
D3_QUERY_SPLIT="${MAXIONBENCH_D3_QUERY_SPLIT:-public}"
D4_BEIR_SUBSETS="${MAXIONBENCH_D4_BEIR_SUBSETS:-scifact fiqa nfcorpus}"
CRAG_INPUT="${MAXIONBENCH_CRAG_INPUT:-${DATASET_ROOT}/D4/crag/crag_task_1_and_2_dev_v4.first_500.jsonl}"
CRAG_MAX_EXAMPLES="${MAXIONBENCH_CRAG_EXAMPLES:-500}"

mkdir -p "$(mb_resolve_host_path "${PROCESSED_ROOT}")"
mb_log "preprocessing datasets from ${DATASET_ROOT} into ${PROCESSED_ROOT}"

mb_python -m maxionbench.cli preprocess-datasets ann-hdf5 \
  --input "${DATASET_ROOT}/D1/glove-100-angular.hdf5" \
  --out "${PROCESSED_ROOT}/D1/glove-100-angular" \
  --family D1 \
  --name glove-100-angular \
  --metric angular \
  --json

mb_python -m maxionbench.cli preprocess-datasets ann-hdf5 \
  --input "${DATASET_ROOT}/D1/sift-128-euclidean.hdf5" \
  --out "${PROCESSED_ROOT}/D1/sift-128-euclidean" \
  --family D1 \
  --name sift-128-euclidean \
  --metric euclidean \
  --json

mb_python -m maxionbench.cli preprocess-datasets ann-hdf5 \
  --input "${DATASET_ROOT}/D1/gist-960-euclidean.hdf5" \
  --out "${PROCESSED_ROOT}/D1/gist-960-euclidean" \
  --family D1 \
  --name gist-960-euclidean \
  --metric euclidean \
  --json

mb_python -m maxionbench.cli preprocess-datasets ann-hdf5 \
  --input "${DATASET_ROOT}/D2/deep-image-96-angular.hdf5" \
  --out "${PROCESSED_ROOT}/D2/deep-image-96-angular" \
  --family D2 \
  --name deep-image-96-angular \
  --metric angular \
  --json

mb_python -m maxionbench.cli preprocess-datasets d3-yfcc \
  --input "${DATASET_ROOT}/D3/yfcc-10M" \
  --out "${PROCESSED_ROOT}/D3/yfcc-10M" \
  --query-split "${D3_QUERY_SPLIT}" \
  --json

for subset in ${D4_BEIR_SUBSETS}; do
  mb_python -m maxionbench.cli preprocess-datasets beir \
    --input "${DATASET_ROOT}/D4/beir/${subset}" \
    --out "${PROCESSED_ROOT}/D4/beir/${subset}" \
    --name "${subset}" \
    --json
done

mb_python -m maxionbench.cli preprocess-datasets crag \
  --input "${CRAG_INPUT}" \
  --out "${PROCESSED_ROOT}/D4/crag/small_slice" \
  --max-examples "${CRAG_MAX_EXAMPLES}" \
  --json
