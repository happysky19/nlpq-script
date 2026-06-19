"""YAML configuration and domain/band-explicit path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_DOMAINS = {"sw", "lw"}
ALLOWED_METHODS = {"det", "rt-aware"}
ALLOWED_STAGES = {
    "preflight",
    "download",
    "dev_tune",
    "final_train",
    "final_test",
    "plot",
    "report",
    "all",
}


@dataclass(frozen=True)
class RunConfig:
    """Validated run configuration loaded from YAML."""

    path: Path
    raw: dict[str, Any]

    @property
    def domain(self) -> str:
        return str(self.raw["run"]["domain"])

    @property
    def bands(self) -> list[int]:
        return [int(v) for v in self.raw["run"]["bands"]]

    @property
    def run_id(self) -> str:
        return str(self.raw["run"]["run_id"])

    @property
    def methods(self) -> list[str]:
        return [str(v) for v in self.raw["nlpq"]["methods"]]

    @property
    def q_values(self) -> list[int]:
        return [int(v) for v in self.raw["nlpq"]["q_values"]]

    @property
    def data_root(self) -> Path:
        return Path(str(self.raw["paths"]["data_root"])).expanduser()

    @property
    def run_root(self) -> Path:
        return Path(str(self.raw["paths"]["run_root"])).expanduser()

    @property
    def ckdmip_bin(self) -> Path:
        return Path(str(self.raw["paths"]["ckdmip_bin"])).expanduser()

    @property
    def py2sess_repo(self) -> Path | None:
        value = self.raw.get("paths", {}).get("py2sess_repo")
        if value in (None, ""):
            return None
        return Path(str(value)).expanduser()


@dataclass(frozen=True)
class OutputPaths:
    """Output paths for one domain/band run."""

    domain: str
    band: int
    run_id: str
    run_dir: Path
    cache_dir: Path

    @property
    def band_label(self) -> str:
        return f"band{self.band:02d}"

    @property
    def prefix(self) -> str:
        return f"{self.domain}_{self.band_label}"

    def model_path(self, method: str, q_value: int, phase: str) -> Path:
        return self.run_dir / "models" / f"{self.prefix}_{method}_q{q_value}_{phase}.npz"

    def metric_path(self, name: str) -> Path:
        return self.run_dir / "metrics" / f"{self.prefix}_{name}.csv"

    def vertical_path(self, split: str) -> Path:
        return self.run_dir / "vertical" / f"{self.prefix}_{split}_vertical_outputs.npz"

    def ckdmip_input_path(self, method: str, q_value: int) -> Path:
        return self.run_dir / "ckdmip_inputs" / f"{self.prefix}_{method}_q{q_value}.nc"

    def ckdmip_flux_path(self, method: str, q_value: int) -> Path:
        return self.run_dir / "ckdmip_fluxes" / f"{self.prefix}_{method}_q{q_value}_fluxes.nc"

    def plot_path(self, name: str = "vertical_profiles") -> Path:
        return self.run_dir / "plots" / f"{self.prefix}_{name}.png"

    def report_path(self) -> Path:
        return self.run_dir / "reports" / f"{self.prefix}_final_report.md"

    def manifest_path(self) -> Path:
        return self.run_dir / f"manifest_{self.prefix}.json"


def load_run_config(path: str | Path) -> RunConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config YAML must contain a mapping")
    cfg = RunConfig(path=config_path, raw=raw)
    validate_run_config(cfg)
    return cfg


def validate_run_config(config: RunConfig, *, stage: str = "preflight") -> None:
    raw = config.raw
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    for section in ("paths", "run", "split", "nlpq", "training", "rt"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"missing required config section: {section}")
    for key in ("data_root", "run_root", "ckdmip_bin"):
        if not raw["paths"].get(key):
            raise ValueError(f"paths.{key} is required")

    domain = str(raw["run"].get("domain", ""))
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"run.domain must be one of {sorted(ALLOWED_DOMAINS)}")
    bands = raw["run"].get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("run.bands must be a non-empty list")
    if any(int(band) < 1 for band in bands):
        raise ValueError("run.bands must use one-based positive band ids")
    if not str(raw["run"].get("run_id", "")).strip():
        raise ValueError("run.run_id is required")

    methods = raw["nlpq"].get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("nlpq.methods must be a non-empty list")
    unknown_methods = sorted(set(str(v) for v in methods) - ALLOWED_METHODS)
    if unknown_methods:
        raise ValueError(f"unsupported NLPQ methods: {unknown_methods}")
    q_values = raw["nlpq"].get("q_values")
    if not isinstance(q_values, list) or not q_values:
        raise ValueError("nlpq.q_values must be a non-empty list")
    if any(int(value) < 1 for value in q_values):
        raise ValueError("nlpq.q_values must be positive")

    dev = raw["split"].get("dev", {})
    final = raw["split"].get("final", {})
    train = parse_profile_spec(str(dev.get("train_profiles", "")))
    val = parse_profile_spec(str(dev.get("val_profiles", "")))
    final_train = parse_profile_spec(str(final.get("train_profiles", "")))
    if not train or not val or not final_train:
        raise ValueError("split profiles must be non-empty")
    leakage = sorted(set(train) & set(val))
    if leakage:
        raise ValueError(f"train/val profile leakage: {leakage}")
    test_dataset = str(final.get("test_dataset", "")).strip()
    if not test_dataset:
        raise ValueError("split.final.test_dataset is required")
    tuning_datasets = [str(v) for v in raw.get("tuning", {}).get("datasets", ["evaluation1"])]
    if test_dataset in tuning_datasets:
        raise ValueError("final test dataset cannot be used for tuning")


def parse_profile_spec(value: str) -> list[int]:
    """Parse zero-based profile ids from comma/range syntax."""

    if not value.strip():
        return []
    out: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left_s, right_s = part.split("-", 1)
            left = int(left_s)
            right = int(right_s)
            if right < left:
                raise ValueError(f"invalid profile range: {part}")
            out.extend(range(left, right + 1))
        else:
            out.append(int(part))
    if min(out) < 0:
        raise ValueError("profile ids must be zero-based and nonnegative")
    return sorted(set(out))


def build_output_paths(config: RunConfig, band: int) -> OutputPaths:
    domain = config.domain
    band_label = f"band{int(band):02d}"
    run_dir = config.run_root / domain / band_label / config.run_id
    cache_dir = config.run_root.parent / "work" / "cache" / domain / band_label
    return OutputPaths(
        domain=domain,
        band=int(band),
        run_id=config.run_id,
        run_dir=run_dir,
        cache_dir=cache_dir,
    )
