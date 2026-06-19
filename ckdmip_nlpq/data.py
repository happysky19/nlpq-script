"""CKDMIP file planning and small native-batch loading helpers."""

from __future__ import annotations

import csv
import re
import shutil
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import RunConfig, parse_profile_spec


CKDMIP_BASE_URL = "https://aux.ecmwf.int/ecpds/home/ckdmip"
DEFAULT_PROFILE_BLOCKS = ("1-10", "11-20", "21-30", "31-40", "41-50")
DEFAULT_SCENARIOS = ("present",)
SW_SPECIES = (
    ("h2o", "present"),
    ("o3", "present"),
    ("n2", "constant"),
    ("o2", "constant"),
    ("co2", "present"),
    ("ch4", "present"),
    ("n2o", "present"),
    ("cfc11", "present-equivalent"),
    ("cfc12", "present"),
    ("rayleigh", "present"),
)
LW_SPECIES = (
    ("h2o", "present"),
    ("o3", "present"),
    ("co2", "present"),
    ("ch4", "present"),
    ("n2o", "present"),
    ("cfc11", "present-equivalent"),
    ("cfc12", "present"),
    ("n2", "constant"),
    ("o2", "constant"),
)
PRESENT_TRACE_GAS = {
    "co2": 415.0,
    "ch4": 1921.0,
    "n2o": 332.0,
    "cfc11": 861.0,
    "cfc12": 495.0,
}
SCENARIO_TRACE_GAS = {
    "present": {"co2": 415.0, "ch4": 1921.0, "n2o": 332.0, "cfc11": 861.0, "cfc12": 495.0},
    "preindustrial": {"co2": 280.0, "ch4": 700.0, "n2o": 270.0, "cfc11": 32.0, "cfc12": 0.0},
    "future": {"co2": 1120.0, "ch4": 3500.0, "n2o": 405.0, "cfc11": 2000.0, "cfc12": 200.0},
    "glacialmax": {"co2": 180.0, "ch4": 350.0, "n2o": 190.0, "cfc11": 32.0, "cfc12": 0.0},
}


@dataclass(frozen=True)
class DownloadItem:
    kind: str
    domain: str
    dataset: str
    scenario: str
    gas: str
    profile_block: str
    band: int
    url: str
    destination: Path
    estimated_bytes: int | None = None


@dataclass(frozen=True)
class RequiredFiles:
    spectra: tuple[Path, ...]
    fluxes: tuple[Path, ...]


@dataclass(frozen=True)
class CKDMIPNativeBatch:
    profile_ids: np.ndarray
    pressure_hl: np.ndarray
    temperature_hl: np.ndarray
    wavenumber: np.ndarray
    spectral_weight: np.ndarray
    tau_native: np.ndarray
    rayleigh_tau_native: np.ndarray | None = None
    incoming_flux_native: np.ndarray | None = None


def profile_blocks(config: RunConfig) -> list[str]:
    blocks = config.raw.get("run", {}).get("profile_blocks", list(DEFAULT_PROFILE_BLOCKS))
    return [str(block) for block in blocks]


def scenarios(config: RunConfig) -> list[str]:
    return [str(v) for v in config.raw.get("run", {}).get("scenarios", list(DEFAULT_SCENARIOS))]


def species_for_domain(config: RunConfig) -> list[tuple[str, str]]:
    raw_species = config.raw.get("run", {}).get("species")
    if raw_species:
        out: list[tuple[str, str]] = []
        for item in raw_species:
            if isinstance(item, str):
                out.append((item, "present"))
            else:
                out.append((str(item["name"]), str(item.get("tag", item.get("file_tag", "present")))))
        return out
    return list(SW_SPECIES if config.domain == "sw" else LW_SPECIES)


def spectrum_filename(domain: str, dataset: str, gas: str, tag: str, block: str) -> str:
    return f"ckdmip_{dataset}_{domain}_spectra_{gas}_{tag}_{block}.h5"


def flux_filename(domain: str, dataset: str, scenario: str) -> str:
    return f"ckdmip_{dataset}_{domain}_fluxes_{scenario}.h5"


def ssi_filename() -> str:
    return "ckdmip_ssi.h5"


def spectrum_path(config: RunConfig, domain: str, dataset: str, gas: str, tag: str, block: str) -> Path:
    return config.data_root / "raw" / "ckdmip" / f"{domain}_spectra" / dataset / spectrum_filename(
        domain, dataset, gas, tag, block
    )


def flux_path(config: RunConfig, domain: str, dataset: str, scenario: str) -> Path:
    return config.data_root / "raw" / "ckdmip" / f"{domain}_fluxes" / dataset / flux_filename(
        domain, dataset, scenario
    )


def ssi_path(config: RunConfig, dataset: str) -> Path:
    return config.data_root / "raw" / "ckdmip" / "sw_spectra" / dataset / ssi_filename()


def build_download_plan(config: RunConfig, band: int) -> list[DownloadItem]:
    """Build a file-granularity download plan for a requested band."""

    domain = config.domain
    datasets = [str(v) for v in config.raw.get("run", {}).get("datasets", ["evaluation1", "evaluation2"])]
    items: list[DownloadItem] = []
    for dataset in datasets:
        if domain == "sw":
            dest = ssi_path(config, dataset)
            items.append(
                DownloadItem(
                    kind="ssi",
                    domain=domain,
                    dataset=dataset,
                    scenario="",
                    gas="",
                    profile_block="",
                    band=int(band),
                    url=f"{CKDMIP_BASE_URL}/{domain}_spectra/{dataset}/{dest.name}",
                    destination=dest,
                )
            )
        for scenario in scenarios(config):
            flux_dest = flux_path(config, domain, dataset, scenario)
            items.append(
                DownloadItem(
                    kind="flux",
                    domain=domain,
                    dataset=dataset,
                    scenario=scenario,
                    gas="",
                    profile_block="",
                    band=int(band),
                    url=f"{CKDMIP_BASE_URL}/{domain}_fluxes/{dataset}/{flux_dest.name}",
                    destination=flux_dest,
                )
            )
        for block in profile_blocks(config):
            for gas, tag in species_for_domain(config):
                dest = spectrum_path(config, domain, dataset, gas, tag, block)
                items.append(
                    DownloadItem(
                        kind="spectra",
                        domain=domain,
                        dataset=dataset,
                        scenario=tag,
                        gas=gas,
                        profile_block=block,
                        band=int(band),
                        url=f"{CKDMIP_BASE_URL}/{domain}_spectra/{dataset}/{dest.name}",
                        destination=dest,
                    )
                )
    return items


def write_download_plan(path: Path, items: list[DownloadItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "domain",
                "dataset",
                "scenario",
                "gas",
                "profile_block",
                "band",
                "url",
                "destination",
                "estimated_bytes",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "kind": item.kind,
                    "domain": item.domain,
                    "dataset": item.dataset,
                    "scenario": item.scenario,
                    "gas": item.gas,
                    "profile_block": item.profile_block,
                    "band": item.band,
                    "url": item.url,
                    "destination": str(item.destination),
                    "estimated_bytes": "" if item.estimated_bytes is None else item.estimated_bytes,
                }
            )


def estimate_remote_size(url: str, *, timeout: float = 20.0) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
    except Exception:
        return None
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def estimate_download_sizes(items: list[DownloadItem]) -> list[DownloadItem]:
    return [replace(item, estimated_bytes=estimate_remote_size(item.url)) for item in items]


def download_items(items: list[DownloadItem], *, overwrite: bool = False) -> None:
    for item in items:
        destination = item.destination
        if destination.exists() and not overwrite:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        if overwrite and part.exists():
            part.unlink()
        expected_size = item.estimated_bytes
        if expected_size is None:
            expected_size = estimate_remote_size(item.url)
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {}
        mode = "wb"
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"
        request = urllib.request.Request(item.url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                response_mode = mode
                if resume_from > 0 and getattr(response, "status", None) == 200:
                    response_mode = "wb"
                with part.open(response_mode) as handle:
                    shutil.copyfileobj(response, handle)
        except Exception:
            if resume_from > 0:
                request = urllib.request.Request(item.url)
                with urllib.request.urlopen(request) as response, part.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            else:
                raise
        if expected_size is not None and part.stat().st_size != expected_size:
            raise IOError(f"incomplete download for {destination}: got {part.stat().st_size}, expected {expected_size}")
        part.replace(destination)


def required_files_for_stage(config: RunConfig, band: int, *, stage: str) -> RequiredFiles:
    if stage == "download":
        return RequiredFiles(spectra=(), fluxes=())
    spectra: list[Path] = []
    fluxes: list[Path] = []
    for item in build_download_plan(config, band):
        if item.kind == "spectra":
            spectra.append(item.destination)
        elif item.kind == "flux":
            fluxes.append(item.destination)
    return RequiredFiles(spectra=tuple(spectra), fluxes=tuple(fluxes))


def missing_required_files(config: RunConfig, band: int, *, stage: str) -> list[Path]:
    required = required_files_for_stage(config, band, stage=stage)
    return [path for path in (*required.spectra, *required.fluxes) if not path.exists()]


def block_start_index(block: str) -> int:
    match = re.match(r"^(\d+)-(\d+)$", block)
    if not match:
        raise ValueError(f"invalid profile block: {block}")
    return int(match.group(1)) - 1


def load_native_batch_from_npz(path: str | Path) -> CKDMIPNativeBatch:
    with np.load(path, allow_pickle=False) as data:
        rayleigh = np.asarray(data["rayleigh_tau_native"], dtype=np.float64) if "rayleigh_tau_native" in data.files else None
        incoming = np.asarray(data["incoming_flux_native"], dtype=np.float64) if "incoming_flux_native" in data.files else None
        return CKDMIPNativeBatch(
            profile_ids=np.asarray(data["profile_ids"], dtype=np.int64),
            pressure_hl=np.asarray(data["pressure_hl"], dtype=np.float64),
            temperature_hl=np.asarray(data["temperature_hl"], dtype=np.float64),
            wavenumber=np.asarray(data["wavenumber"], dtype=np.float64),
            spectral_weight=np.asarray(data["spectral_weight"], dtype=np.float64),
            tau_native=np.asarray(data["tau_native"], dtype=np.float64),
            rayleigh_tau_native=rayleigh,
            incoming_flux_native=incoming,
        )


def load_native_batch_from_ckdmip(
    config: RunConfig,
    *,
    band: int,
    dataset: str,
    profile_spec: str,
    scenario: str = "present",
) -> CKDMIPNativeBatch:
    """Load and sum official CKDMIP gas optical depths for a domain/band.

    The loader expects CKDMIP-like HDF5 variables named ``wavenumber``,
    ``optical_depth``, ``pressure_hl``, and ``temperature_hl``. It slices the
    requested band by wavenumber and sums all configured species.
    """

    profiles = parse_profile_spec(profile_spec)
    if not profiles:
        raise ValueError("profile selection is empty")

    tau_parts: list[np.ndarray] = []
    rayleigh_parts: list[np.ndarray] = []
    pressure_by_profile: dict[int, np.ndarray] = {}
    temperature_by_profile: dict[int, np.ndarray] = {}
    selected_wavenumber: np.ndarray | None = None

    for gas, tag in species_for_domain(config):
        gas_tau_by_profile: dict[int, np.ndarray] = {}
        for block in profile_blocks(config):
            path = spectrum_path(config, config.domain, dataset, gas, tag, block)
            if not path.exists():
                raise FileNotFoundError(path)
            start = block_start_index(block)
            with h5py.File(path, "r") as handle:
                wavenumber = np.asarray(handle["wavenumber"], dtype=np.float64)
                mask = band_mask(config.domain, int(band), wavenumber)
                if not np.any(mask):
                    raise ValueError(f"band {band} selects no wavenumbers in {path}")
                if selected_wavenumber is None:
                    selected_wavenumber = wavenumber[mask]
                elif selected_wavenumber.shape != wavenumber[mask].shape or not np.allclose(
                    selected_wavenumber, wavenumber[mask]
                ):
                    raise ValueError(f"wavenumber grid mismatch in {path}")
                optical_depth = np.asarray(handle["optical_depth"], dtype=np.float64)
                pressure = np.asarray(handle["pressure_hl"], dtype=np.float64)
                temperature = np.asarray(handle["temperature_hl"], dtype=np.float64)
                for local_idx in range(optical_depth.shape[0]):
                    profile_id = start + local_idx
                    if profile_id not in profiles:
                        continue
                    gas_tau_by_profile[profile_id] = optical_depth[local_idx, :, mask]
                    pressure_by_profile.setdefault(profile_id, pressure[local_idx])
                    temperature_by_profile.setdefault(profile_id, temperature[local_idx])
        if sorted(gas_tau_by_profile) != profiles:
            missing = sorted(set(profiles) - set(gas_tau_by_profile))
            raise FileNotFoundError(f"missing profiles for {gas}: {missing}")
        stacked = np.stack([gas_tau_by_profile[p] for p in profiles], axis=0)
        stacked = stacked * species_scale_for_scenario(gas, scenario)
        if config.domain == "sw" and gas == "rayleigh":
            rayleigh_parts.append(stacked)
        else:
            tau_parts.append(stacked)

    if selected_wavenumber is None:
        raise ValueError("no spectra loaded")
    if tau_parts:
        tau_native = np.sum(np.stack(tau_parts, axis=0), axis=0)
    elif rayleigh_parts:
        tau_native = np.zeros_like(rayleigh_parts[0])
    else:
        raise ValueError("no absorbing optical-depth spectra loaded")
    rayleigh_tau_native = None
    if rayleigh_parts:
        rayleigh_tau_native = np.sum(np.stack(rayleigh_parts, axis=0), axis=0)
    pressure_hl = np.stack([pressure_by_profile[p] for p in profiles], axis=0)
    temperature_hl = np.stack([temperature_by_profile[p] for p in profiles], axis=0)
    spectral_weight = infer_spectral_weight(selected_wavenumber)
    incoming_flux_native = None
    if config.domain == "sw":
        incoming_flux_native = load_incoming_flux(
            config,
            band=int(band),
            dataset=dataset,
            selected_wavenumber=selected_wavenumber,
        )
    return CKDMIPNativeBatch(
        profile_ids=np.asarray(profiles, dtype=np.int64),
        pressure_hl=pressure_hl,
        temperature_hl=temperature_hl,
        wavenumber=selected_wavenumber,
        spectral_weight=spectral_weight,
        tau_native=tau_native,
        rayleigh_tau_native=rayleigh_tau_native,
        incoming_flux_native=incoming_flux_native,
    )


def load_incoming_flux(
    config: RunConfig,
    *,
    band: int,
    dataset: str,
    selected_wavenumber: np.ndarray,
) -> np.ndarray:
    path = ssi_path(config, dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing CKDMIP solar irradiance file: {path}")
    with h5py.File(path, "r") as handle:
        if "solar_spectral_irradiance" not in handle:
            raise KeyError(f"solar_spectral_irradiance not found in {path}")
        incoming = np.asarray(handle["solar_spectral_irradiance"], dtype=np.float64)
        if "wavenumber" in handle:
            wavenumber = np.asarray(handle["wavenumber"], dtype=np.float64)
            mask = np.isin(np.round(wavenumber, 10), np.round(selected_wavenumber, 10))
            if np.count_nonzero(mask) != selected_wavenumber.size:
                mask = band_mask(config.domain, int(band), wavenumber)
            incoming = incoming[mask]
        if incoming.shape[-1] != selected_wavenumber.size:
            raise ValueError(f"SSI grid in {path} does not match selected spectra")
        return incoming


def species_scale_for_scenario(gas: str, scenario: str) -> float:
    if gas not in PRESENT_TRACE_GAS:
        if scenario == "n2-0" and gas == "n2":
            return 0.0
        if scenario == "o2-0" and gas == "o2":
            return 0.0
        return 1.0
    values = dict(SCENARIO_TRACE_GAS.get("present", {}))
    if scenario in SCENARIO_TRACE_GAS:
        values.update(SCENARIO_TRACE_GAS[scenario])
    else:
        parts = scenario.split("-")
        if len(parts) == 2 and parts[0] in values:
            values[parts[0]] = float(parts[1])
        elif len(parts) == 4:
            for gas_name, value in ((parts[0], parts[1]), (parts[2], parts[3])):
                if gas_name in values:
                    values[gas_name] = float(value)
    return float(values.get(gas, PRESENT_TRACE_GAS[gas])) / float(PRESENT_TRACE_GAS[gas])


def infer_spectral_weight(wavenumber: np.ndarray) -> np.ndarray:
    weight = infer_spectral_width(wavenumber)
    return weight / np.sum(weight)


def infer_spectral_width(wavenumber: np.ndarray) -> np.ndarray:
    if wavenumber.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(wavenumber.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (wavenumber[:-1] + wavenumber[1:])
    edges[0] = wavenumber[0] - (edges[1] - wavenumber[0])
    edges[-1] = wavenumber[-1] + (wavenumber[-1] - edges[-2])
    return np.diff(edges)


def band_mask(domain: str, band: int, wavenumber: np.ndarray) -> np.ndarray:
    bounds = default_band_bounds(domain)
    if band < 1 or band > len(bounds):
        raise ValueError(f"{domain} band {band} is outside configured bounds")
    left, right = bounds[band - 1]
    return (wavenumber >= left) & (wavenumber < right)


def default_band_bounds(domain: str) -> tuple[tuple[float, float], ...]:
    if domain == "sw":
        return (
            (250.0, 2600.0),
            (2600.0, 3250.0),
            (3250.0, 4000.0),
            (4000.0, 4650.0),
            (4650.0, 5150.0),
            (5150.0, 6150.0),
            (6150.0, 8050.0),
            (8050.0, 12850.0),
            (12850.0, 16000.0),
            (16000.0, 22650.0),
            (22650.0, 29000.0),
            (29000.0, 38000.0),
            (38000.0, 50000.0),
        )
    if domain == "lw":
        return (
            (0.0, 350.0),
            (350.0, 500.0),
            (500.0, 630.0),
            (630.0, 700.0),
            (700.0, 820.0),
            (820.0, 980.0),
            (980.0, 1080.0),
            (1080.0, 1180.0),
            (1180.0, 1390.0),
            (1390.0, 1480.0),
            (1480.0, 1800.0),
            (1800.0, 2080.0),
            (2080.0, 3260.0),
        )
    raise ValueError(f"unknown domain: {domain}")
