#!/bin/bash
#SBATCH --job-name=ckdmip-final
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

CONFIG="${1:?usage: sbatch final_test_band.sbatch path/to/config.yaml}"
LOG_DIR="$(python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from ckdmip_nlpq.config import build_output_paths, load_run_config
cfg = load_run_config(sys.argv[1])
print(build_output_paths(cfg, cfg.bands[0]).run_dir / "logs")
PY
)"
mkdir -p "${LOG_DIR}"
LOG_PREFIX="${LOG_DIR}/${SLURM_JOB_NAME:-final}-${SLURM_JOB_ID:-manual}"
exec > >(tee -a "${LOG_PREFIX}.out") 2> >(tee -a "${LOG_PREFIX}.err" >&2)

python scripts/run_ckdmip_nlpq.py --config "${CONFIG}" --stage final_test
