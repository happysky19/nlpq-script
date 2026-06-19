"""Export frozen NLPQ optics in CKDMIP CKD NetCDF layouts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .data import infer_spectral_width
from .model import CompressedBatch, NativeBatch, compress_additive_spectral


PLANCK_C1 = 1.1910429723971884e-8
PLANCK_C2 = 1.4387768775039338


def write_ckdmip_netcdf(
    path: str | Path,
    native: NativeBatch,
    compressed: CompressedBatch,
    *,
    domain: str,
    band: int,
    rt_options: dict[str, Any] | None = None,
) -> None:
    if domain == "lw":
        write_lw_ckdmip_netcdf(path, native, compressed, band=band)
    elif domain == "sw":
        write_sw_ckdmip_netcdf(path, native, compressed, band=band, rt_options=rt_options or {})
    else:
        raise ValueError(f"unsupported domain: {domain}")


def write_lw_ckdmip_netcdf(
    path: str | Path,
    native: NativeBatch,
    compressed: CompressedBatch,
    *,
    band: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tau_q = _nonnegative_tau(compressed.tau_q)
    planck_hl_q = _compressed_planck(native, compressed.cluster_id, tau_q.shape[-1])
    surf_emission_q = np.pi * planck_hl_q[:, -1, :]
    gpt_width = _cluster_sums(infer_spectral_width(native.wavenumber), compressed.cluster_id, tau_q.shape[-1])

    with Dataset(path, "w") as ds:
        _define_common(ds, native.profile_ids.size, tau_q.shape[1], native.pressure_hl.shape[1])
        ds.createDimension("gpoint_lw", tau_q.shape[2])

        _write_common(ds, native)
        tau = ds.createVariable("optical_depth", "f8", ("column", "level", "gpoint_lw"))
        planck = ds.createVariable("planck_hl", "f8", ("column", "half_level", "gpoint_lw"))
        surf = ds.createVariable("surf_emission", "f8", ("column", "gpoint_lw"))
        weight = ds.createVariable("gpt_weight", "f8", ("gpoint_lw",))
        tau[:, :, :] = tau_q
        planck[:, :, :] = planck_hl_q
        surf[:, :] = surf_emission_q
        weight[:] = gpt_width
        tau.units = "1"
        planck.units = "W m-2 sr-1"
        surf.units = "W m-2"
        weight.units = "cm-1"
        ds.setncattr("title", "CKDMIP longwave CKD input from frozen NLPQ")
        ds.setncattr("band_id", int(band))


def write_sw_ckdmip_netcdf(
    path: str | Path,
    native: NativeBatch,
    compressed: CompressedBatch,
    *,
    band: int,
    rt_options: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tau_q = _nonnegative_tau(compressed.tau_q)
    rayleigh_tau_q = compressed.rayleigh_tau_q
    if rayleigh_tau_q is None:
        if not bool(rt_options.get("allow_zero_rayleigh", False)):
            raise ValueError("SW CKDMIP export requires Rayleigh optical depth or rt.allow_zero_rayleigh=true")
        rayleigh_tau_q = np.zeros_like(tau_q)
    incoming_flux_q = compressed.incoming_flux_q
    if incoming_flux_q is None:
        total_flux = rt_options.get("sw_total_incoming_flux_wm2")
        if total_flux is None:
            raise ValueError("SW CKDMIP export requires incoming_flux_native from ckdmip_ssi.h5")
        incoming_flux_q = np.asarray(compressed.weight_q, dtype=np.float64) * float(total_flux)
    if incoming_flux_q.ndim == 1:
        incoming_flux_q = np.broadcast_to(incoming_flux_q, (tau_q.shape[0], tau_q.shape[2]))
    gpt_width = _cluster_sums(infer_spectral_width(native.wavenumber), compressed.cluster_id, tau_q.shape[-1])

    with Dataset(path, "w") as ds:
        _define_common(ds, native.profile_ids.size, tau_q.shape[1], native.pressure_hl.shape[1])
        ds.createDimension("gpoint_sw", tau_q.shape[2])

        _write_common(ds, native)
        tau = ds.createVariable("optical_depth", "f8", ("column", "level", "gpoint_sw"))
        rayleigh = ds.createVariable("rayleigh_optical_depth", "f8", ("column", "level", "gpoint_sw"))
        incoming = ds.createVariable("incoming_flux", "f8", ("column", "gpoint_sw"))
        weight = ds.createVariable("gpt_weight", "f8", ("gpoint_sw",))
        tau[:, :, :] = tau_q
        rayleigh[:, :, :] = _nonnegative_tau(rayleigh_tau_q)
        incoming[:, :] = np.maximum(incoming_flux_q, 0.0)
        weight[:] = gpt_width
        tau.units = "1"
        rayleigh.units = "1"
        incoming.units = "W m-2"
        weight.units = "cm-1"
        ds.setncattr("title", "CKDMIP shortwave CKD input from frozen NLPQ")
        ds.setncattr("band_id", int(band))


def write_membership_csv(path: str | Path, cluster_id: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["native_index", "q_index"])
        writer.writeheader()
        for idx, q_index in enumerate(np.asarray(cluster_id, dtype=np.int64)):
            writer.writerow({"native_index": int(idx), "q_index": int(q_index)})


def _define_common(ds: Dataset, ncol: int, nlev: int, nhalf: int) -> None:
    ds.createDimension("column", ncol)
    ds.createDimension("level", nlev)
    ds.createDimension("half_level", nhalf)


def _write_common(ds: Dataset, native: NativeBatch) -> None:
    profile = ds.createVariable("profile_id", "i4", ("column",))
    pressure = ds.createVariable("pressure_hl", "f8", ("column", "half_level"))
    temperature = ds.createVariable("temperature_hl", "f8", ("column", "half_level"))
    profile[:] = native.profile_ids.astype(np.int32)
    pressure[:, :] = native.pressure_hl
    temperature[:, :] = native.temperature_hl
    pressure.units = "Pa"
    temperature.units = "K"


def _nonnegative_tau(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def _compressed_planck(native: NativeBatch, cluster_id: np.ndarray, q_value: int) -> np.ndarray:
    width = infer_spectral_width(native.wavenumber)
    wn = np.asarray(native.wavenumber, dtype=np.float64)
    temperature = np.asarray(native.temperature_hl, dtype=np.float64)
    exponent = PLANCK_C2 * wn[None, None, :] / np.maximum(temperature[:, :, None], 1.0)
    spectral = PLANCK_C1 * wn[None, None, :] ** 3 / np.expm1(np.clip(exponent, 1.0e-12, 700.0))
    return compress_additive_spectral(spectral * width[None, None, :], cluster_id, q_value)


def _cluster_sums(values: np.ndarray, cluster_id: np.ndarray, q_value: int) -> np.ndarray:
    out = np.zeros(q_value, dtype=np.float64)
    cluster = np.asarray(cluster_id, dtype=np.int64)
    for q in range(q_value):
        mask = cluster == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        out[q] = float(np.sum(np.asarray(values, dtype=np.float64)[mask]))
    return out
