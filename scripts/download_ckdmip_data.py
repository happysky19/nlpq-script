#!/usr/bin/env python3
"""Download or plan official CKDMIP files for one YAML-defined run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ckdmip_nlpq.config import build_output_paths, load_run_config  # noqa: E402
from ckdmip_nlpq.data import build_download_plan, download_items, estimate_download_sizes, write_download_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--estimate-size", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    config = load_run_config(args.config)
    for band in config.bands:
        paths = build_output_paths(config, band)
        items = build_download_plan(config, band)
        if args.num_shards > 1:
            items = items[args.shard_index :: args.num_shards]
        if args.estimate_size:
            items = estimate_download_sizes(items)
        plan_path = paths.run_dir / "download_plan.csv"
        write_download_plan(plan_path, items)
        total = sum(item.estimated_bytes or 0 for item in items)
        print(f"wrote {plan_path}")
        if args.estimate_size:
            print(f"estimated bytes with known sizes: {total}")
        if not args.dry_run:
            download_items(items, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
