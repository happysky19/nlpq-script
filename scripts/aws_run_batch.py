#!/usr/bin/env python3
"""Run multiple CKDMIP NLPQ YAML jobs with AWS-friendly logs/manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts" / "run_ckdmip_nlpq.py"
PREFLIGHT = PROJECT / "scripts" / "aws_preflight.py"


@dataclass(frozen=True)
class JobResult:
    config: Path
    stage: str
    command: list[str]
    status: str
    returncode: int
    elapsed_s: float
    log_out: Path
    log_err: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": str(self.config),
            "stage": self.stage,
            "command": self.command,
            "status": self.status,
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "log_out": str(self.log_out),
            "log_err": str(self.log_err),
        }


def default_log_root() -> Path:
    return PROJECT / "runs" / "_aws_logs"


def run_command(command: list[str], *, log_out: Path, log_err: Path, env: dict[str, str]) -> tuple[int, float]:
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_err.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_out.open("w") as out, log_err.open("w") as err:
        proc = subprocess.run(command, cwd=PROJECT, env=env, stdout=out, stderr=err, check=False)
    return int(proc.returncode), time.perf_counter() - started


def stage_command(config: Path, stage: str, *, dry_run: bool) -> list[str]:
    command = [sys.executable, str(RUNNER), "--config", str(config), "--stage", stage]
    if dry_run:
        command.append("--dry-run")
    return command


def preflight_command(configs: list[Path], *, require_data: bool, json_out: Path) -> list[str]:
    command = [sys.executable, str(PREFLIGHT)]
    for config in configs:
        command.extend(["--config", str(config)])
    if require_data:
        command.append("--require-data")
    command.extend(["--json-out", str(json_out)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["preflight", "download", "dev_tune", "final_train", "final_test", "plot", "report"],
        help="Stages to run for each config. Use 'all' to delegate the whole workflow to run_ckdmip_nlpq.py.",
    )
    parser.add_argument("--log-root", type=Path, default=default_log_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aws-preflight", action="store_true")
    parser.add_argument("--require-data-in-preflight", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    configs = [path.expanduser().resolve() for path in args.config]
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_root = args.log_root.expanduser().resolve() / run_id
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else log_root / "aws_batch_manifest.json"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    results: list[JobResult] = []
    started = time.perf_counter()
    if not args.skip_aws_preflight:
        preflight_json = log_root / "aws_preflight.json"
        command = preflight_command(
            configs,
            require_data=bool(args.require_data_in_preflight),
            json_out=preflight_json,
        )
        rc, elapsed = run_command(
            command,
            log_out=log_root / "aws_preflight.out",
            log_err=log_root / "aws_preflight.err",
            env=env,
        )
        result = JobResult(
            config=Path("<all>"),
            stage="aws_preflight",
            command=command,
            status="OK" if rc == 0 else "FAILED",
            returncode=rc,
            elapsed_s=elapsed,
            log_out=log_root / "aws_preflight.out",
            log_err=log_root / "aws_preflight.err",
        )
        results.append(result)
        if rc != 0 and not args.keep_going:
            write_manifest(manifest_path, run_id, started, results)
            raise SystemExit(rc)

    for config in configs:
        for stage in args.stages:
            label = config.stem
            command = stage_command(config, stage, dry_run=bool(args.dry_run))
            rc, elapsed = run_command(
                command,
                log_out=log_root / f"{label}_{stage}.out",
                log_err=log_root / f"{label}_{stage}.err",
                env=env,
            )
            result = JobResult(
                config=config,
                stage=stage,
                command=command,
                status="OK" if rc == 0 else "FAILED",
                returncode=rc,
                elapsed_s=elapsed,
                log_out=log_root / f"{label}_{stage}.out",
                log_err=log_root / f"{label}_{stage}.err",
            )
            results.append(result)
            write_manifest(manifest_path, run_id, started, results)
            print(f"{label} {stage}: {result.status} ({elapsed:.1f}s)")
            if rc != 0 and not args.keep_going:
                raise SystemExit(rc)

    write_manifest(manifest_path, run_id, started, results)
    failed = [result for result in results if result.returncode != 0]
    if failed:
        raise SystemExit(1)


def write_manifest(path: Path, run_id: str, started: float, results: list[JobResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "project": str(PROJECT),
        "elapsed_s": time.perf_counter() - started,
        "status": "FAILED" if any(result.returncode != 0 for result in results) else "OK",
        "results": [result.as_dict() for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
