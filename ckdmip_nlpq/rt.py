"""Runtime checks and CKDMIP executable command construction."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    status: str
    returncode: int | None = None


def ckdmip_executable(config: RunConfig, domain: str) -> Path:
    name = "ckdmip_sw" if domain == "sw" else "ckdmip_lw"
    return config.ckdmip_bin / name


def check_ckdmip_executable(config: RunConfig, domain: str) -> None:
    path = ckdmip_executable(config, domain)
    if not path.exists():
        raise FileNotFoundError(f"missing CKDMIP executable: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"CKDMIP executable is not a file: {path}")


def check_py2sess_available(py2sess_repo: Path | None = None) -> str:
    inserted = False
    if py2sess_repo is not None:
        src = py2sess_repo / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
            inserted = True
        for name in list(sys.modules):
            if name == "py2sess" or name.startswith("py2sess."):
                del sys.modules[name]
    try:
        module = importlib.import_module("py2sess")
    except Exception as exc:
        raise ImportError("py2sess is required for rt-aware training") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(str(py2sess_repo / "src"))
            except ValueError:
                pass
    return str(getattr(module, "__version__", "unknown"))


def check_py2sess_forward_flux_available(py2sess_repo: Path | None = None) -> str:
    version = check_py2sess_available(py2sess_repo)
    try:
        module = importlib.import_module("py2sess")
        solver = getattr(module, "TwoStreamEss")
    except Exception as exc:
        raise ImportError("py2sess TwoStreamEss is required for rt-aware training") from exc
    if not hasattr(solver, "forward_flux"):
        location = str(getattr(module, "__file__", "unknown"))
        raise ImportError(
            "py2sess forward_flux is required for rt-aware training; "
            f"loaded py2sess from {location}"
        )
    return version


class CKDMIPRunner:
    def __init__(self, executable: Path, *, domain: str) -> None:
        self.executable = executable
        self.domain = domain

    def run(
        self,
        *,
        input_file: Path,
        output_file: Path,
        config_file: Path,
        scenario: str,
        dry_run: bool = False,
    ) -> CommandResult:
        command = [str(self.executable)]
        command.extend(["--config", str(config_file)])
        command.extend(["--scenario", str(scenario)])
        command.extend(["--ckd", str(input_file)])
        command.extend(["--output", str(output_file)])
        if dry_run:
            return CommandResult(command=command, status="DRY_RUN")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False)
        return CommandResult(command=command, status="OK" if completed.returncode == 0 else "FAILED", returncode=completed.returncode)


def write_ckdmip_namelist(path: Path, *, domain: str, raw_config: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rt = raw_config.get("rt", {})
    if domain == "lw":
        nangle = int(rt.get("lw_nangle", 4))
        text = "\n".join(
            [
                "&longwave_config",
                'optical_depth_name = "optical_depth",',
                'pressure_name = "pressure_hl",',
                'planck_name = "planck_hl",',
                'surf_emission_name = "surf_emission",',
                f"nangle = {nangle},",
                "do_write_planck = false,",
                "do_write_spectral_fluxes = true,",
                "do_write_optical_depth = false,",
                "input_planck_per_sterad = true,",
                "iverbose = 2",
                "/",
                "",
            ]
        )
    elif domain == "sw":
        mu_values = [float(v) for v in rt.get("mu_values", [0.1, 0.3, 0.5, 0.7, 0.9])]
        mu_text = ", ".join(f"{value:g}" for value in mu_values)
        surf_albedo = float(rt.get("surf_albedo", 0.15))
        text = "\n".join(
            [
                "&shortwave_config",
                'optical_depth_name = "optical_depth",',
                'rayleigh_optical_depth_name = "rayleigh_optical_depth",',
                'incoming_flux_name = "incoming_flux",',
                'pressure_name = "pressure_hl",',
                f"surf_albedo = {surf_albedo:g},",
                "iverbose = 3,",
                "use_mu0_dimension = true,",
                f"cos_solar_zenith_angle(1:{len(mu_values)}) = {mu_text},",
                "do_write_spectral_fluxes = true,",
                "/",
                "",
            ]
        )
    else:
        raise ValueError(f"unsupported domain: {domain}")
    path.write_text(text)
    return path


def write_command_manifest(path: Path, results: list[CommandResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"command": result.command, "status": result.status, "returncode": result.returncode}
        for result in results
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
