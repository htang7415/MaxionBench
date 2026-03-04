#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Placeholder calibration entrypoint; full D3 calibration implementation is on
# the critical path after adapter integration.
python -m maxionbench.orchestration.runner --config configs/scenarios/calibrate_d3.yaml
