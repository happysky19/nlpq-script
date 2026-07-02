#!/usr/bin/env bash
# Bootstrap an Ubuntu AWS instance for ckdmip_nlpq_suite.
#
# This script installs system packages, Python dependencies, py2sess, and
# CKDMIP executables.  It does not download CKDMIP datasets.

set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NLPQ_AWS_ROOT="${NLPQ_AWS_ROOT:-/mnt/nlpq}"

echo "[bootstrap] project=${PROJECT_DIR}"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  gfortran \
  git \
  libhdf5-dev \
  libnetcdf-dev \
  libnetcdff-dev \
  netcdf-bin \
  pkg-config \
  python3-dev \
  python3-pip \
  python3-venv \
  unzip

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ "${SKIP_EXTERNAL_DEPS:-0}" != "1" ]]; then
  NLPQ_AWS_ROOT="${NLPQ_AWS_ROOT}" PROJECT_DIR="${PROJECT_DIR}" bash aws/install_external_deps.sh
fi

echo "[bootstrap] done. Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
