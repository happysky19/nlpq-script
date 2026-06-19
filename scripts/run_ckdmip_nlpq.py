#!/usr/bin/env python3
"""Run the YAML-controlled CKDMIP NLPQ workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ckdmip_nlpq.config import ALLOWED_STAGES, load_run_config, validate_run_config  # noqa: E402
from ckdmip_nlpq.workflow import run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=sorted(ALLOWED_STAGES), default="preflight")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_run_config(args.config)
    validate_run_config(config, stage=args.stage)
    run_stage(config, stage=args.stage, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
