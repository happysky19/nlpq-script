#!/usr/bin/env python3
"""AWS/local readiness checks for CKDMIP NLPQ runs.

This script is intentionally conservative: it checks the software/runtime
surface needed by the YAML workflow without mutating datasets or models.  Use
``--require-data`` only after CKDMIP inputs have been downloaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ckdmip_nlpq.config import build_output_paths, load_run_config  # noqa: E402
from ckdmip_nlpq.data import missing_required_files  # noqa: E402
from ckdmip_nlpq.rt import (  # noqa: E402
    check_ckdmip_executable,
    check_py2sess_forward_flux_available,
)
from ckdmip_nlpq.workflow import is_rt_aware_method, primary_scenario  # noqa: E402


REQUIRED_IMPORTS = ("yaml", "h5py", "matplotlib", "netCDF4", "numpy", "torch")


def import_status(module_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - message is enough for CLI
        return {
            "name": module_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - started,
        }
    return {
        "name": module_name,
        "ok": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(getattr(module, "__file__", "")),
        "elapsed_s": time.perf_counter() - started,
    }


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - already reported by imports
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    status: dict[str, Any] = {
        "ok": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if torch.cuda.is_available():
        status["cuda_device_count"] = int(torch.cuda.device_count())
        status["cuda_devices"] = [
            {
                "index": idx,
                "name": str(torch.cuda.get_device_name(idx)),
                "capability": list(torch.cuda.get_device_capability(idx)),
            }
            for idx in range(torch.cuda.device_count())
        ]
    return status


def disk_status(paths: list[Path]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        resolved = probe.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            usage = shutil.disk_usage(resolved)
            out.append(
                {
                    "path": str(resolved),
                    "total_gb": usage.total / 1e9,
                    "used_gb": usage.used / 1e9,
                    "free_gb": usage.free / 1e9,
                }
            )
        except Exception as exc:
            out.append({"path": str(resolved), "error": f"{type(exc).__name__}: {exc}"})
    return out


def config_status(config_path: Path, *, require_data: bool) -> dict[str, Any]:
    item: dict[str, Any] = {"config": str(config_path)}
    errors: list[str] = []
    warnings: list[str] = []
    try:
        config = load_run_config(config_path)
    except Exception as exc:
        item["ok"] = False
        item["errors"] = [f"config_load: {type(exc).__name__}: {exc}"]
        return item

    item.update(
        {
            "ok": True,
            "domain": config.domain,
            "bands": config.bands,
            "run_id": config.run_id,
            "methods": config.methods,
            "q_values": config.q_values,
            "data_root": str(config.data_root),
            "run_root": str(config.run_root),
            "ckdmip_bin": str(config.ckdmip_bin),
            "py2sess_repo": None if config.py2sess_repo is None else str(config.py2sess_repo),
        }
    )
    try:
        scenario = primary_scenario(config)
        item["scenario"] = scenario
    except Exception as exc:
        errors.append(f"scenario: {type(exc).__name__}: {exc}")

    for band in config.bands:
        band_entry: dict[str, Any] = {"band": int(band)}
        paths = build_output_paths(config, band)
        band_entry["run_dir"] = str(paths.run_dir)
        try:
            check_ckdmip_executable(config, config.domain)
            band_entry["ckdmip_executable_ok"] = True
        except Exception as exc:
            band_entry["ckdmip_executable_ok"] = False
            errors.append(f"band {band}: ckdmip executable: {type(exc).__name__}: {exc}")
        if require_data:
            missing = missing_required_files(config, band, stage="preflight")
            band_entry["missing_required_file_count"] = len(missing)
            band_entry["missing_required_files_preview"] = [str(v) for v in missing[:12]]
            if missing:
                errors.append(f"band {band}: missing {len(missing)} CKDMIP files")
        else:
            missing = missing_required_files(config, band, stage="preflight")
            band_entry["missing_required_file_count"] = len(missing)
            if missing:
                warnings.append(f"band {band}: {len(missing)} CKDMIP files not present; run download stage first")
        item.setdefault("bands_status", []).append(band_entry)

    if any(is_rt_aware_method(method) for method in config.methods):
        try:
            version = check_py2sess_forward_flux_available(config.py2sess_repo)
            item["py2sess_forward_flux_ok"] = True
            item["py2sess_version"] = version
        except Exception as exc:
            item["py2sess_forward_flux_ok"] = False
            errors.append(f"py2sess forward_flux: {type(exc).__name__}: {exc}")

    item["warnings"] = warnings
    item["errors"] = errors
    item["ok"] = not errors
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    imports = [import_status(name) for name in REQUIRED_IMPORTS]
    configs = [config_status(path, require_data=bool(args.require_data)) for path in args.config]
    disk_paths: list[Path] = [PROJECT]
    for cfg in configs:
        for key in ("data_root", "run_root"):
            if cfg.get(key):
                disk_paths.append(Path(str(cfg[key])))
    payload = {
        "ok": all(item.get("ok") for item in imports) and all(item.get("ok") for item in configs),
        "project": str(PROJECT),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "env": {
            "AWS_REGION": os.environ.get("AWS_REGION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "imports": imports,
        "torch": torch_status(),
        "configs": configs,
        "disk": disk_status(disk_paths),
        "elapsed_s": time.perf_counter() - started,
    }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text)
    if not payload["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
