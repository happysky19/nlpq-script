"""Flux and heating-rate metrics for CKDMIP NLPQ outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .plotting import align_flux_profiles, heating_rate, read_model_flux, read_truth_flux, with_profile_ids


def build_flux_metrics(
    *,
    domain: str,
    band: int,
    model_flux_path: Path,
    truth_flux_path: Path,
    profile_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    if not model_flux_path.exists():
        return {"metric_status": "missing_model_flux", "truth_flux": str(truth_flux_path)}
    if not truth_flux_path.exists():
        return {"metric_status": "missing_truth_flux", "truth_flux": str(truth_flux_path)}
    model = read_model_flux(model_flux_path, domain)
    if profile_ids is not None:
        model = with_profile_ids(model, np.asarray(profile_ids, dtype=np.int64))
    truth = read_truth_flux(truth_flux_path, domain, band)
    try:
        model, truth = align_flux_profiles(model, truth)
    except ValueError as exc:
        return {
            "metric_status": "profile_mismatch",
            "truth_flux": str(truth_flux_path),
            "error": str(exc),
        }
    if model["up"].shape != truth["up"].shape or model["down"].shape != truth["down"].shape:
        return {
            "metric_status": "shape_mismatch",
            "truth_flux": str(truth_flux_path),
            "model_up_shape": list(model["up"].shape),
            "truth_up_shape": list(truth["up"].shape),
        }
    model_heat = heating_rate(model["up"], model["down"], model["pressure_hl"])
    truth_heat = heating_rate(truth["up"], truth["down"], truth["pressure_hl"])
    pressure_mid_hpa = 0.5e-2 * (truth["pressure_hl"][:, :-1] + truth["pressure_hl"][:, 1:])
    upper = (np.mean(pressure_mid_hpa, axis=0) >= 0.02) & (np.mean(pressure_mid_hpa, axis=0) < 4.0)
    lower = (np.mean(pressure_mid_hpa, axis=0) >= 4.0) & (np.mean(pressure_mid_hpa, axis=0) <= 1100.0)
    toa = rmse(model["up"][:, 0] - truth["up"][:, 0])
    surface = rmse(model["down"][:, -1] - truth["down"][:, -1])
    return {
        "metric_status": "compared",
        "truth_flux": str(truth_flux_path),
        "toa_flux_rmse": toa,
        "surface_flux_rmse": surface,
        "toa_up_rmse": toa,
        "surface_down_rmse": surface,
        "level_up_rmse": rmse(model["up"] - truth["up"]),
        "level_down_rmse": rmse(model["down"] - truth["down"]),
        "heating_rmse_upper": rmse(model_heat[:, upper] - truth_heat[:, upper]) if np.any(upper) else "",
        "heating_rmse_lower": rmse(model_heat[:, lower] - truth_heat[:, lower]) if np.any(lower) else "",
    }


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
