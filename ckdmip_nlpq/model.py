"""Minimal frozen NLPQ model used by the CKDMIP workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1.0e-12
ALLOWED_METHODS = {"det", "rt-aware", "rt-aware-nn"}


@dataclass(frozen=True)
class NativeBatch:
    profile_ids: np.ndarray
    pressure_hl: np.ndarray
    temperature_hl: np.ndarray
    wavenumber: np.ndarray
    spectral_weight: np.ndarray
    tau_native: np.ndarray
    species_tau_native: np.ndarray | None = None
    species_names: tuple[str, ...] = ()
    rayleigh_tau_native: np.ndarray | None = None
    incoming_flux_native: np.ndarray | None = None


@dataclass(frozen=True)
class CompressedBatch:
    tau_q: np.ndarray
    weight_q: np.ndarray
    cluster_id: np.ndarray
    rayleigh_tau_q: np.ndarray | None = None
    incoming_flux_q: np.ndarray | None = None


class NLPQModel:
    """Small conservative pseudo-line model with a hard frozen assignment."""

    def __init__(
        self,
        *,
        domain: str,
        band: int,
        method: str,
        q_value: int,
        seed: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if method not in ALLOWED_METHODS:
            raise ValueError(f"method must be one of {sorted(ALLOWED_METHODS)}")
        if q_value < 1:
            raise ValueError("q_value must be positive")
        self.domain = domain
        self.band = int(band)
        self.method = method
        self.q_value = int(q_value)
        self.seed = int(seed)
        self.metadata = dict(metadata or {})
        self.cluster_id: np.ndarray | None = None
        self.weight_q: np.ndarray | None = None
        self.nn_state: dict[str, Any] | None = None
        self.frozen = False

    def fit(
        self,
        batch: NativeBatch,
        *,
        training_config: dict[str, Any] | None = None,
        py2sess_repo: Path | None = None,
    ) -> "NLPQModel":
        if self.frozen:
            raise RuntimeError("cannot fit a frozen model")
        spectral_count = int(batch.tau_native.shape[-1])
        if self.q_value > spectral_count:
            raise ValueError(f"Q={self.q_value} exceeds native spectral count M={spectral_count}")
        if self.q_value == spectral_count:
            cluster_id = np.arange(spectral_count, dtype=np.int64)
            weight_q = _cluster_weights(batch.spectral_weight, cluster_id, self.q_value)
            training_log: dict[str, Any] = {"assignment_training": "identity", "steps": 0}
        elif self.method in ("rt-aware", "rt-aware-nn"):
            from .rt_aware import train_rt_aware_assignment

            cluster_id, weight_q, training_log = train_rt_aware_assignment(
                batch,
                domain=self.domain,
                q_value=self.q_value,
                seed=self.seed,
                training_config=training_config,
                py2sess_repo=py2sess_repo,
            )
            if self.method == "rt-aware-nn" and self.q_value < spectral_count:
                from .nn_tau import train_neural_overlap_residual

                self.nn_state, nn_log = train_neural_overlap_residual(
                    batch,
                    cluster_id=cluster_id,
                    weight_q=weight_q,
                    q_value=self.q_value,
                    training_config=training_config,
                    seed=self.seed,
                )
                training_log = dict(training_log)
                training_log["neural_overlap"] = nn_log
        else:
            cluster_id = _contiguous_quantile_clusters(
                batch.spectral_weight,
                self.q_value,
                score=_spectral_score(batch, self.method),
            )
            weight_q = _cluster_weights(batch.spectral_weight, cluster_id, self.q_value)
            training_log = {"assignment_training": _assignment_training_name(self.method), "steps": 0}
        self.cluster_id = cluster_id
        self.weight_q = weight_q
        self.metadata.update(
            {
                "domain": self.domain,
                "band": self.band,
                "method": self.method,
                "q_value": self.q_value,
                "native_spectral_count": spectral_count,
                "compression_settings": compression_settings(self.domain, training_config),
                "fit_profile_ids": [int(v) for v in batch.profile_ids.tolist()],
                "online_policy": "frozen assignment; no eval/test fitting",
                "training_log": training_log,
            }
        )
        return self

    def freeze(self) -> "NLPQModel":
        if self.cluster_id is None or self.weight_q is None:
            raise RuntimeError("cannot freeze before fit/load")
        self.frozen = True
        return self

    def apply(self, batch: NativeBatch) -> CompressedBatch:
        if not self.frozen:
            raise RuntimeError("model must be frozen before apply")
        if self.cluster_id is None or self.weight_q is None:
            raise RuntimeError("model has no assignment")
        settings = dict(self.metadata.get("compression_settings", compression_settings(self.domain, None)))
        if self.domain == "sw":
            tau_q, rayleigh_tau_q, incoming_flux_q = compress_sw_optics(
                batch,
                self.cluster_id,
                self.q_value,
                settings=settings,
            )
        else:
            tau_q = compress_tau(batch.tau_native, batch.spectral_weight, self.cluster_id, self.q_value)
            rayleigh_tau_q = None
            incoming_flux_q = None
        if self.method == "rt-aware-nn" and self.nn_state is not None:
            from .nn_tau import apply_neural_overlap_residual

            tau_q = apply_neural_overlap_residual(
                batch,
                cluster_id=self.cluster_id,
                weight_q=self.weight_q,
                q_value=self.q_value,
                nn_state=self.nn_state,
                target_tau_q=tau_q,
            )
        return CompressedBatch(
            tau_q=tau_q,
            weight_q=self.weight_q.copy(),
            cluster_id=self.cluster_id.copy(),
            rayleigh_tau_q=rayleigh_tau_q,
            incoming_flux_q=incoming_flux_q,
        )

    def save(self, path: str | Path) -> None:
        if self.cluster_id is None or self.weight_q is None:
            raise RuntimeError("cannot save an unfitted model")
        payload = {
            "domain": self.domain,
            "band": self.band,
            "method": self.method,
            "q_value": self.q_value,
            "seed": self.seed,
            "frozen": self.frozen,
            **self.metadata,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            cluster_id=self.cluster_id,
            weight_q=self.weight_q,
            metadata_json=json.dumps(payload, sort_keys=True),
            **_encode_nn_state(self.nn_state),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NLPQModel":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            model = cls(
                domain=str(metadata["domain"]),
                band=int(metadata["band"]),
                method=str(metadata["method"]),
                q_value=int(metadata["q_value"]),
                seed=int(metadata.get("seed", 0)),
                metadata=metadata,
            )
            model.cluster_id = np.asarray(data["cluster_id"], dtype=np.int64)
            model.weight_q = np.asarray(data["weight_q"], dtype=np.float64)
            model.nn_state = _decode_nn_state(data)
            model.frozen = bool(metadata.get("frozen", True))
            return model

    def assert_compatible(self, *, domain: str, band: int, method: str, q_value: int) -> None:
        errors = []
        if self.domain != domain:
            errors.append(f"domain {self.domain!r} != {domain!r}")
        if self.band != int(band):
            errors.append(f"band {self.band} != {int(band)}")
        if self.method != method:
            errors.append(f"method {self.method!r} != {method!r}")
        if self.q_value != int(q_value):
            errors.append(f"Q {self.q_value} != {int(q_value)}")
        if errors:
            raise ValueError("model metadata mismatch: " + "; ".join(errors))


def compress_tau(
    tau_native: np.ndarray,
    spectral_weight: np.ndarray,
    cluster_id: np.ndarray,
    q_value: int,
) -> np.ndarray:
    tau = np.asarray(tau_native, dtype=np.float64)
    weight = np.asarray(spectral_weight, dtype=np.float64)
    cluster = np.asarray(cluster_id, dtype=np.int64)
    if tau.shape[-1] != cluster.shape[0] or weight.shape != cluster.shape:
        raise ValueError("native spectral dimensions do not match assignment")
    if q_value == tau.shape[-1] and np.array_equal(cluster, np.arange(tau.shape[-1])):
        return np.maximum(tau, EPS)
    out = np.empty(tau.shape[:-1] + (q_value,), dtype=np.float64)
    trans = np.exp(-np.maximum(tau, 0.0))
    for q in range(q_value):
        mask = cluster == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        w = weight[mask]
        w = w / np.sum(w)
        avg_trans = np.sum(trans[..., mask] * w, axis=-1)
        out[..., q] = -np.log(np.clip(avg_trans, EPS, 1.0))
    return np.maximum(out, EPS)


def compress_additive_spectral(values: np.ndarray, cluster_id: np.ndarray, q_value: int) -> np.ndarray:
    """Sum already-integrated spectral quantities into pseudo-lines."""

    array = np.asarray(values, dtype=np.float64)
    cluster = np.asarray(cluster_id, dtype=np.int64)
    if array.shape[-1] != cluster.shape[0]:
        raise ValueError("native spectral dimensions do not match assignment")
    out = np.zeros(array.shape[:-1] + (q_value,), dtype=np.float64)
    for q in range(q_value):
        mask = cluster == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        out[..., q] = np.sum(array[..., mask], axis=-1)
    return out


def _cluster_weights(spectral_weight: np.ndarray, cluster_id: np.ndarray, q_value: int) -> np.ndarray:
    out = np.zeros(q_value, dtype=np.float64)
    for q in range(q_value):
        mask = cluster_id == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        out[q] = float(np.sum(spectral_weight[mask]))
    total = float(np.sum(out))
    if total <= 0.0:
        raise ValueError("cluster weights sum to zero")
    return out / total


def _spectral_score(batch: NativeBatch, method: str) -> np.ndarray:
    tau = np.asarray(batch.tau_native, dtype=np.float64)
    if method == "det":
        return np.arange(tau.shape[-1], dtype=np.float64)
    raise ValueError(f"unsupported deterministic assignment method: {method}")


def _assignment_training_name(method: str) -> str:
    if method == "det":
        return "regular_native_index_binning"
    return method


def compression_settings(domain: str, training_config: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(training_config or {})
    if domain != "sw":
        return {}
    tau_mode = str(options.get("sw_tau_mode", "direct_beam"))
    rayleigh_mode = str(options.get("sw_rayleigh_mode", "arithmetic"))
    mu_ref = float(options.get("sw_tau_mu_ref", options.get("sw_mu_ref", 0.5)))
    if tau_mode not in {"transmittance", "direct_beam"}:
        raise ValueError("sw_tau_mode must be transmittance or direct_beam")
    if rayleigh_mode not in {"transmittance", "arithmetic", "direct_beam"}:
        raise ValueError("sw_rayleigh_mode must be transmittance, arithmetic, or direct_beam")
    if mu_ref <= 0.0 or mu_ref > 1.0:
        raise ValueError("sw_tau_mu_ref must be in (0, 1]")
    return {
        "sw_tau_mode": tau_mode,
        "sw_rayleigh_mode": rayleigh_mode,
        "sw_tau_mu_ref": mu_ref,
    }


def compress_sw_optics(
    batch: NativeBatch,
    cluster_id: np.ndarray,
    q_value: int,
    *,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    tau_mode = str(settings.get("sw_tau_mode", "direct_beam"))
    rayleigh_mode = str(settings.get("sw_rayleigh_mode", "arithmetic"))
    mu_ref = float(settings.get("sw_tau_mu_ref", 0.5))
    if tau_mode == "direct_beam":
        tau_q = compress_direct_beam_tau(
            batch.tau_native,
            batch.incoming_flux_native,
            batch.spectral_weight,
            cluster_id,
            q_value,
            mu_ref=mu_ref,
        )
    else:
        tau_q = compress_tau(batch.tau_native, batch.spectral_weight, cluster_id, q_value)

    rayleigh_tau_q = None
    if batch.rayleigh_tau_native is not None:
        if rayleigh_mode == "arithmetic":
            rayleigh_tau_q = compress_solar_weighted_mean_tau(
                batch.rayleigh_tau_native,
                batch.incoming_flux_native,
                batch.spectral_weight,
                cluster_id,
                q_value,
            )
        elif rayleigh_mode == "direct_beam":
            rayleigh_tau_q = compress_direct_beam_tau(
                batch.rayleigh_tau_native,
                batch.incoming_flux_native,
                batch.spectral_weight,
                cluster_id,
                q_value,
                mu_ref=mu_ref,
            )
        else:
            rayleigh_tau_q = compress_tau(batch.rayleigh_tau_native, batch.spectral_weight, cluster_id, q_value)

    incoming_flux_q = None
    if batch.incoming_flux_native is not None:
        incoming_flux_q = compress_additive_spectral(batch.incoming_flux_native, cluster_id, q_value)
    return tau_q, rayleigh_tau_q, incoming_flux_q


def compress_direct_beam_tau(
    tau_native: np.ndarray,
    incoming_flux_native: np.ndarray | None,
    spectral_weight: np.ndarray,
    cluster_id: np.ndarray,
    q_value: int,
    *,
    mu_ref: float,
) -> np.ndarray:
    tau = np.asarray(tau_native, dtype=np.float64)
    cluster = np.asarray(cluster_id, dtype=np.int64)
    if tau.shape[-1] != cluster.shape[0]:
        raise ValueError("native spectral dimensions do not match assignment")
    if q_value == tau.shape[-1] and np.array_equal(cluster, np.arange(tau.shape[-1])):
        return np.maximum(tau, EPS)
    source_weight = _source_weights(incoming_flux_native, spectral_weight, tau.shape[0], tau.shape[-1])
    out = np.empty(tau.shape[:-1] + (q_value,), dtype=np.float64)
    mu = max(float(mu_ref), EPS)
    trans = np.exp(-np.minimum(np.maximum(tau, 0.0) / mu, 700.0))
    for q in range(q_value):
        mask = cluster == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        w = source_weight[:, mask]
        denom = np.sum(w, axis=1)
        if np.any(denom <= 0.0):
            fallback = _fallback_weights(spectral_weight, mask)
            w = np.where(denom[:, None] > 0.0, w, fallback[None, :])
            denom = np.sum(w, axis=1)
        avg_trans = np.sum(trans[..., mask] * w[:, None, :], axis=-1) / denom[:, None]
        out[..., q] = -mu * np.log(np.clip(avg_trans, EPS, 1.0))
    return np.maximum(out, EPS)


def compress_solar_weighted_mean_tau(
    tau_native: np.ndarray,
    incoming_flux_native: np.ndarray | None,
    spectral_weight: np.ndarray,
    cluster_id: np.ndarray,
    q_value: int,
) -> np.ndarray:
    tau = np.asarray(tau_native, dtype=np.float64)
    cluster = np.asarray(cluster_id, dtype=np.int64)
    if tau.shape[-1] != cluster.shape[0]:
        raise ValueError("native spectral dimensions do not match assignment")
    if q_value == tau.shape[-1] and np.array_equal(cluster, np.arange(tau.shape[-1])):
        return np.maximum(tau, EPS)
    source_weight = _source_weights(incoming_flux_native, spectral_weight, tau.shape[0], tau.shape[-1])
    out = np.empty(tau.shape[:-1] + (q_value,), dtype=np.float64)
    for q in range(q_value):
        mask = cluster == q
        if not np.any(mask):
            raise ValueError("every pseudo-line cluster must be nonempty")
        w = source_weight[:, mask]
        denom = np.sum(w, axis=1)
        if np.any(denom <= 0.0):
            fallback = _fallback_weights(spectral_weight, mask)
            w = np.where(denom[:, None] > 0.0, w, fallback[None, :])
            denom = np.sum(w, axis=1)
        out[..., q] = np.sum(np.maximum(tau[..., mask], 0.0) * w[:, None, :], axis=-1) / denom[:, None]
    return np.maximum(out, EPS)


def _source_weights(
    incoming_flux_native: np.ndarray | None,
    spectral_weight: np.ndarray,
    profile_count: int,
    spectral_count: int,
) -> np.ndarray:
    if incoming_flux_native is None:
        weight = np.asarray(spectral_weight, dtype=np.float64)
        if weight.shape != (spectral_count,):
            raise ValueError("spectral_weight shape does not match native spectral dimension")
        return np.broadcast_to(np.maximum(weight, 0.0), (profile_count, spectral_count)).copy()
    incoming = np.asarray(incoming_flux_native, dtype=np.float64)
    if incoming.ndim == 1:
        if incoming.shape != (spectral_count,):
            raise ValueError("incoming_flux_native spectral dimension does not match assignment")
        return np.broadcast_to(np.maximum(incoming, 0.0), (profile_count, spectral_count)).copy()
    if incoming.ndim == 2:
        if incoming.shape != (profile_count, spectral_count):
            raise ValueError("incoming_flux_native must have shape [M] or [B,M]")
        return np.maximum(incoming, 0.0)
    raise ValueError("incoming_flux_native must have shape [M] or [B,M]")


def _fallback_weights(spectral_weight: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = np.asarray(spectral_weight, dtype=np.float64)[mask]
    weight = np.maximum(weight, 0.0)
    total = float(np.sum(weight))
    if total <= 0.0:
        return np.full(np.count_nonzero(mask), 1.0 / float(np.count_nonzero(mask)), dtype=np.float64)
    return weight / total


def _encode_nn_state(nn_state: dict[str, Any] | None) -> dict[str, np.ndarray]:
    if nn_state is None:
        return {}
    arrays = {key: value for key, value in nn_state.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in nn_state.items() if key not in arrays}
    payload: dict[str, np.ndarray] = {
        "nn_state_keys": np.asarray(list(arrays), dtype="U"),
        "nn_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for key, value in arrays.items():
        payload[f"nn_array_{key}"] = np.asarray(value)
    return payload


def _decode_nn_state(data: Any) -> dict[str, Any] | None:
    if "nn_state_keys" not in data.files:
        return None
    metadata = json.loads(str(data["nn_metadata_json"])) if "nn_metadata_json" in data.files else {}
    state: dict[str, Any] = dict(metadata)
    for key in np.asarray(data["nn_state_keys"]).tolist():
        state[str(key)] = np.asarray(data[f"nn_array_{key}"])
    return state


def _contiguous_quantile_clusters(spectral_weight: np.ndarray, q_value: int, *, score: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    sorted_weight = np.asarray(spectral_weight, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weight) / np.sum(sorted_weight)
    sorted_cluster = np.minimum((cumulative * q_value).astype(np.int64), q_value - 1)
    sorted_cluster[0] = 0
    sorted_cluster[-1] = q_value - 1
    cluster = np.empty_like(sorted_cluster)
    cluster[order] = sorted_cluster
    for q in range(q_value):
        if not np.any(cluster == q):
            native = int(order[min(q, order.size - 1)])
            cluster[native] = q
    return cluster.astype(np.int64)
