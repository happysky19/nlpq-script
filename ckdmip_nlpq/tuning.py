"""Automatic dev-validation ranking and selected-setting export."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    method: str
    q_value: int
    lr: float
    steps: int
    train_pressure_min_hpa: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.method,
            "q_value": self.q_value,
            "lr": self.lr,
            "steps": self.steps,
            "train_pressure_min_hpa": self.train_pressure_min_hpa,
        }


def expand_candidates(raw_config: dict[str, Any]) -> list[Candidate]:
    methods = [str(v) for v in raw_config["nlpq"]["methods"]]
    q_values = [int(v) for v in raw_config["nlpq"]["q_values"]]
    grid = raw_config.get("tuning", {}).get("grid", {})
    lrs = [float(v) for v in grid.get("lr", [raw_config["training"].get("lr", 0.05)])]
    steps = [int(v) for v in grid.get("steps", [raw_config["training"].get("steps", 220)])]
    pmins = [
        float(v)
        for v in grid.get(
            "train_pressure_min_hpa",
            [raw_config["nlpq"].get("train_pressure_min_hpa", 0.001)],
        )
    ]
    out: list[Candidate] = []
    for idx, (method, q_value, lr, step_count, pmin) in enumerate(
        itertools.product(methods, q_values, lrs, steps, pmins)
    ):
        out.append(
            Candidate(
                candidate_id=idx,
                method=method,
                q_value=q_value,
                lr=lr,
                steps=step_count,
                train_pressure_min_hpa=pmin,
            )
        )
    return out


def score_candidate(row: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    weights = weights or {}
    terms = {
        "toa_flux_rmse": float(row.get("toa_flux_rmse", row.get("tau_rmse", 0.0))),
        "surface_flux_rmse": float(row.get("surface_flux_rmse", row.get("tau_rmse", 0.0))),
        "heating_rmse_upper": float(row.get("heating_rmse_upper", row.get("tau_rmse", 0.0))),
        "heating_rmse_lower": float(row.get("heating_rmse_lower", row.get("tau_rmse", 0.0))),
        "runtime_ms_per_profile": float(row.get("runtime_ms_per_profile", 0.0)),
    }
    return (
        weights.get("toa_flux", 1.0) * terms["toa_flux_rmse"]
        + weights.get("surface_flux", 1.0) * terms["surface_flux_rmse"]
        + weights.get("heating_upper", 1.0) * terms["heating_rmse_upper"]
        + weights.get("heating_lower", 1.0) * terms["heating_rmse_lower"]
        + weights.get("runtime", 0.0) * terms["runtime_ms_per_profile"]
    )


def rank_candidates(rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in rows:
        scored = dict(row)
        scored["score"] = score_candidate(scored, weights)
        ranked.append(scored)
    ranked.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["q_value"]),
            float(item.get("runtime_ms_per_profile", 0.0)),
            int(item["candidate_id"]),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selected_settings(yaml_path: Path, json_path: Path, ranked_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranked_rows:
        raise ValueError("ranked candidate rows are empty")
    selected = dict(ranked_rows[0])
    payload = {"selected": selected}
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=True))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
