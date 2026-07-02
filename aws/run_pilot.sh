#!/usr/bin/env bash
# AWS pilot runner for one LW and one SW CKDMIP NLPQ band.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CONFIG_LW="${CONFIG_LW:-configs/aws_lw_band04_pilot.yaml}"
CONFIG_SW="${CONFIG_SW:-configs/aws_sw_band02_pilot.yaml}"

python scripts/aws_run_batch.py \
  --config "${CONFIG_LW}" \
  --config "${CONFIG_SW}" \
  --stages preflight download dev_tune final_train final_test plot report
