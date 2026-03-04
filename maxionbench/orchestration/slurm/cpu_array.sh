#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

python -m maxionbench.orchestration.runner --config configs/scenarios/s1_ann_frontier.yaml
