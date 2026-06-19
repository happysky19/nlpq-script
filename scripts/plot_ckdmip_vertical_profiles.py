#!/usr/bin/env python3
"""Plot vertical flux and heating profiles for a YAML-defined CKDMIP NLPQ run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ckdmip_nlpq.config import load_run_config  # noqa: E402
from ckdmip_nlpq.plotting import plot_band_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_run_config(args.config)
    for band in config.bands:
        plot_path = plot_band_outputs(config, band=band)
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
