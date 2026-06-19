"""Vertical flux/heating plotting for CKDMIP NLPQ outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset

from .config import RunConfig, build_output_paths
from .data import flux_path


_cache_root = Path(tempfile.gettempdir()) / "ckdmip-nlpq-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import parse_profile_spec


GRAVITY_M_S2 = 9.80665
SECONDS_PER_DAY = 86400.0
CP_AIR_J_KG_K = 1004.0


def plot_band_outputs(config: RunConfig, *, band: int) -> Path:
    paths = build_output_paths(config, band)
    selected_path = paths.run_dir / "selected_settings.json"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected = json.loads(selected_path.read_text())["selected"]
    model_flux = paths.ckdmip_flux_path(str(selected["method"]), int(selected["q_value"]))
    if not model_flux.exists():
        raise FileNotFoundError(model_flux)
    scenario = str(config.raw.get("run", {}).get("scenarios", ["present"])[0])
    truth_flux = flux_path(config, config.domain, str(config.raw["split"]["final"]["test_dataset"]), scenario)
    test_profiles = parse_profile_spec(str(config.raw["split"]["final"].get("test_profiles", "0-49")))
    model = read_model_flux(model_flux, config.domain)
    if test_profiles:
        model = with_profile_ids(model, np.asarray(test_profiles, dtype=np.int64))
    plot_path = paths.plot_path()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_vertical_profiles(
        plot_path,
        domain=config.domain,
        band=int(band),
        model=model,
        truth=read_truth_flux(truth_flux, config.domain, int(band)) if truth_flux.exists() else None,
    )
    return plot_path


def read_model_flux(path: Path, domain: str) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        pressure = np.asarray(ds["pressure_hl"][:], dtype=np.float64)
        profile_ids = _profile_ids_from_netcdf(ds, pressure.shape[0])
        if domain == "lw":
            return {
                "profile_ids": profile_ids,
                "pressure_hl": pressure,
                "up": np.asarray(ds["flux_up_lw"][:], dtype=np.float64),
                "down": np.asarray(ds["flux_dn_lw"][:], dtype=np.float64),
            }
        down = np.asarray(ds["flux_dn_sw"][:], dtype=np.float64)
        up = np.asarray(ds["flux_up_sw"][:], dtype=np.float64)
        if down.ndim == 3:
            down = down[:, 2, :]
            up = up[:, 2, :]
        return {"profile_ids": profile_ids, "pressure_hl": pressure, "up": up, "down": down}


def read_truth_flux(path: Path, domain: str, band: int) -> dict[str, np.ndarray]:
    band_index = band - 1
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            pressure = np.asarray(handle["pressure_hl"], dtype=np.float64)
            profile_ids = _profile_ids_from_hdf5(handle, pressure.shape[0])
            if domain == "lw":
                up = np.asarray(handle["band_flux_up_lw"][:, :, band_index], dtype=np.float64)
                down = np.asarray(handle["band_flux_dn_lw"][:, :, band_index], dtype=np.float64)
            else:
                up = np.asarray(handle["band_flux_up_sw"][:, 2, :, band_index], dtype=np.float64)
                down = np.asarray(handle["band_flux_dn_sw"][:, 2, :, band_index], dtype=np.float64)
            return {"profile_ids": profile_ids, "pressure_hl": pressure, "up": up, "down": down}
    with Dataset(path) as ds:
        pressure = np.asarray(ds["pressure_hl"][:], dtype=np.float64)
        profile_ids = _profile_ids_from_netcdf(ds, pressure.shape[0])
        if domain == "lw":
            up = np.asarray(ds["band_flux_up_lw"][:, :, band_index], dtype=np.float64)
            down = np.asarray(ds["band_flux_dn_lw"][:, :, band_index], dtype=np.float64)
        else:
            up = np.asarray(ds["band_flux_up_sw"][:, 2, :, band_index], dtype=np.float64)
            down = np.asarray(ds["band_flux_dn_sw"][:, 2, :, band_index], dtype=np.float64)
        return {"profile_ids": profile_ids, "pressure_hl": pressure, "up": up, "down": down}


def with_profile_ids(flux: dict[str, np.ndarray], profile_ids: np.ndarray) -> dict[str, np.ndarray]:
    out = dict(flux)
    ids = np.asarray(profile_ids, dtype=np.int64)
    if ids.shape[0] != np.asarray(out["pressure_hl"]).shape[0]:
        raise ValueError("profile id count does not match flux columns")
    out["profile_ids"] = ids
    return out


def align_flux_profiles(
    model: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model_ids = np.asarray(model.get("profile_ids", np.arange(model["pressure_hl"].shape[0])), dtype=np.int64)
    truth_ids = np.asarray(truth.get("profile_ids", np.arange(truth["pressure_hl"].shape[0])), dtype=np.int64)
    if model_ids.shape[0] == truth_ids.shape[0] and np.array_equal(model_ids, truth_ids):
        return model, truth
    index = {int(profile_id): idx for idx, profile_id in enumerate(truth_ids.tolist())}
    missing = [int(profile_id) for profile_id in model_ids.tolist() if int(profile_id) not in index]
    if missing:
        raise ValueError(f"truth flux is missing profiles: {missing}")
    positions = np.asarray([index[int(profile_id)] for profile_id in model_ids.tolist()], dtype=np.int64)
    return model, _take_profiles(truth, positions, model_ids)


def _take_profiles(
    flux: dict[str, np.ndarray],
    positions: np.ndarray,
    profile_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"profile_ids": np.asarray(profile_ids, dtype=np.int64)}
    for key, value in flux.items():
        if key == "profile_ids":
            continue
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == np.asarray(flux["pressure_hl"]).shape[0]:
            out[key] = array[positions]
        else:
            out[key] = array
    return out


def _profile_ids_from_netcdf(ds: Dataset, ncol: int) -> np.ndarray:
    if "profile_id" in ds.variables:
        values = np.asarray(ds["profile_id"][:], dtype=np.int64)
        if values.shape[0] == ncol:
            return values
    return np.arange(ncol, dtype=np.int64)


def _profile_ids_from_hdf5(handle: h5py.File, ncol: int) -> np.ndarray:
    if "profile_id" in handle:
        values = np.asarray(handle["profile_id"], dtype=np.int64)
        if values.shape[0] == ncol:
            return values
    return np.arange(ncol, dtype=np.int64)


def plot_vertical_profiles(
    path: Path,
    *,
    domain: str,
    band: int,
    model: dict[str, np.ndarray],
    truth: dict[str, np.ndarray] | None,
) -> None:
    model_pressure = model["pressure_hl"] * 0.01
    if truth is not None:
        model, truth = align_flux_profiles(model, truth)
        model_pressure = model["pressure_hl"] * 0.01
    model_mid = 0.5 * (model_pressure[:, :-1] + model_pressure[:, 1:])
    model_heat = heating_rate(model["up"], model["down"], model["pressure_hl"])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6), constrained_layout=True)
    axes[0].plot(np.mean(model["down"], axis=0), np.mean(model_pressure, axis=0), color="#2f6fbb", label="model down")
    axes[0].plot(np.mean(model["up"], axis=0), np.mean(model_pressure, axis=0), color="#c73b3c", label="model up")
    axes[1].plot(np.mean(model_heat, axis=0), np.mean(model_mid, axis=0), color="#2f6fbb", label="model")
    if truth is not None:
        truth_pressure = truth["pressure_hl"] * 0.01
        truth_mid = 0.5 * (truth_pressure[:, :-1] + truth_pressure[:, 1:])
        truth_heat = heating_rate(truth["up"], truth["down"], truth["pressure_hl"])
        axes[0].plot(np.mean(truth["down"], axis=0), np.mean(truth_pressure, axis=0), "k--", label="truth down")
        axes[0].plot(np.mean(truth["up"], axis=0), np.mean(truth_pressure, axis=0), "k:", label="truth up")
        axes[1].plot(np.mean(truth_heat, axis=0), np.mean(truth_mid, axis=0), "k--", label="truth")
    for ax in axes:
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Pressure (hPa)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("Flux (W m$^{-2}$)")
    axes[1].set_xlabel("Heating rate (K day$^{-1}$)")
    fig.suptitle(f"{domain.upper()} band {band:02d}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def heating_rate(up: np.ndarray, down: np.ndarray, pressure_hl_pa: np.ndarray) -> np.ndarray:
    net = up - down
    dp = np.maximum(np.diff(pressure_hl_pa, axis=1), 1.0e-12)
    return (net[:, 1:] - net[:, :-1]) * GRAVITY_M_S2 * SECONDS_PER_DAY / CP_AIR_J_KG_K / dp
