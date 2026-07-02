#!/usr/bin/env bash
# Install py2sess and build CKDMIP executables for AWS/local runs.

set -euo pipefail

NLPQ_AWS_ROOT="${NLPQ_AWS_ROOT:-/mnt/nlpq}"
PROJECT_DIR="${PROJECT_DIR:-${NLPQ_AWS_ROOT}/ckdmip_nlpq_suite}"
EXTERNAL_DIR="${EXTERNAL_DIR:-${NLPQ_AWS_ROOT}/external}"
PY2SESS_URL="${PY2SESS_URL:-https://github.com/happysky19/py2sess.git}"
PY2SESS_REF="${PY2SESS_REF:-main}"
PY2SESS_DIR="${PY2SESS_DIR:-${EXTERNAL_DIR}/py2sess}"
CKDMIP_SRC_DIR="${CKDMIP_SRC_DIR:-${EXTERNAL_DIR}/ckdmip-1.0-src}"
CKDMIP_LINK_DIR="${CKDMIP_LINK_DIR:-${EXTERNAL_DIR}/ckdmip}"
CKDMIP_TARBALL="${CKDMIP_TARBALL:-${EXTERNAL_DIR}/ckdmip-1.0.tar.gz}"
CKDMIP_SOURCE_URL="${CKDMIP_SOURCE_URL:-https://aux.ecmwf.int/ecpds/home/ckdmip/ckdmip-1.0.tar.gz}"

mkdir -p "${EXTERNAL_DIR}"

echo "[external] root=${NLPQ_AWS_ROOT}"
echo "[external] project=${PROJECT_DIR}"
echo "[external] external=${EXTERNAL_DIR}"

if [[ ! -d "${PY2SESS_DIR}/.git" ]]; then
  echo "[external] cloning py2sess from ${PY2SESS_URL}"
  git clone "${PY2SESS_URL}" "${PY2SESS_DIR}"
else
  echo "[external] py2sess checkout already exists: ${PY2SESS_DIR}"
fi

(
  cd "${PY2SESS_DIR}"
  git fetch --all --tags --prune
  git checkout "${PY2SESS_REF}"
)

python -m pip install -e "${PY2SESS_DIR}[torch]"

if [[ ! -d "${CKDMIP_SRC_DIR}" ]]; then
  if [[ ! -f "${CKDMIP_TARBALL}" ]]; then
    echo "[external] downloading CKDMIP source from ${CKDMIP_SOURCE_URL}"
    echo "[external] if this URL fails, set CKDMIP_SOURCE_URL to the official ckdmip-1.0 tarball URL"
    curl -fL "${CKDMIP_SOURCE_URL}" -o "${CKDMIP_TARBALL}"
  fi
  echo "[external] unpacking ${CKDMIP_TARBALL}"
  tmp_extract="${EXTERNAL_DIR}/_ckdmip_extract_tmp"
  rm -rf "${tmp_extract}"
  mkdir -p "${tmp_extract}"
  tar -xzf "${CKDMIP_TARBALL}" -C "${tmp_extract}"
  found_makefile="$(find "${tmp_extract}" -maxdepth 2 -type f -name Makefile -print -quit)"
  if [[ -z "${found_makefile}" ]]; then
    echo "[external] could not find CKDMIP Makefile after unpacking ${CKDMIP_TARBALL}" >&2
    exit 2
  fi
  found_dir="$(dirname "${found_makefile}")"
  mv "${found_dir}" "${CKDMIP_SRC_DIR}"
  rm -rf "${tmp_extract}"
else
  echo "[external] CKDMIP source already exists: ${CKDMIP_SRC_DIR}"
fi

(
  cd "${CKDMIP_SRC_DIR}"
  make
)

ln -sfn "${CKDMIP_SRC_DIR}" "${CKDMIP_LINK_DIR}"

for exe in ckdmip_lw ckdmip_sw; do
  if [[ ! -x "${CKDMIP_LINK_DIR}/bin/${exe}" ]]; then
    echo "[external] missing executable after build: ${CKDMIP_LINK_DIR}/bin/${exe}" >&2
    exit 2
  fi
done

python - <<'PY'
import importlib
module = importlib.import_module("py2sess")
solver = getattr(module, "TwoStreamEss")
if not hasattr(solver, "forward_flux"):
    raise SystemExit("py2sess TwoStreamEss.forward_flux is missing")
print(f"[external] py2sess ok: {getattr(module, '__version__', 'unknown')} {getattr(module, '__file__', '')}")
PY

echo "[external] CKDMIP ok: ${CKDMIP_LINK_DIR}/bin"
echo "[external] done"
