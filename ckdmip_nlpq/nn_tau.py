"""Frozen neural optical-depth residual for CKDMIP NLPQ exports."""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import EPS, NativeBatch, compress_tau


def train_neural_overlap_residual(
    batch: NativeBatch,
    *,
    cluster_id: np.ndarray,
    weight_q: np.ndarray,
    q_value: int,
    training_config: dict[str, Any] | None = None,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a small Q-space residual from species tau to mixture tau."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when torch is absent
        raise ImportError("torch is required for rt-aware-nn training") from exc

    options = dict(training_config or {})
    species_tau = _required_species_tau(batch)
    species_tau_q, base_tau_q, target_tau_q, singleton_mask = _compressed_overlap_targets(
        batch,
        species_tau=species_tau,
        cluster_id=cluster_id,
        q_value=q_value,
    )
    features = _feature_matrix(batch, species_tau_q, base_tau_q, weight_q)
    feature_mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
    feature_std = features.reshape(-1, features.shape[-1]).std(axis=0)
    feature_std = np.maximum(feature_std, 1.0e-6)

    dtype = _torch_dtype(str(options.get("dtype", "float32")), torch)
    device = torch.device(str(options.get("device", "cpu")))
    hidden_dim = int(options.get("nn_hidden_dim", options.get("neural_overlap_hidden_dim", 64)))
    steps = int(options.get("nn_steps", options.get("neural_overlap_steps", 200)))
    lr = float(options.get("nn_lr", options.get("neural_overlap_lr", 1.0e-3)))
    weight_decay = float(options.get("nn_weight_decay", options.get("neural_overlap_weight_decay", 1.0e-5)))
    rmax = float(options.get("nn_rmax", options.get("neural_overlap_rmax", 2.0)))
    log_every = max(1, int(options.get("nn_log_every_steps", max(1, steps // 5))))
    if steps < 1:
        raise ValueError("rt-aware-nn nn_steps must be positive")
    if lr <= 0.0:
        raise ValueError("rt-aware-nn nn_lr must be positive")
    if hidden_dim < 1:
        raise ValueError("rt-aware-nn nn_hidden_dim must be positive")
    if rmax <= 0.0:
        raise ValueError("rt-aware-nn nn_rmax must be positive")

    torch.manual_seed(int(seed))
    x = torch.tensor((features - feature_mean) / feature_std, dtype=dtype, device=device)
    base = torch.tensor(base_tau_q, dtype=dtype, device=device)
    target = torch.tensor(target_tau_q, dtype=dtype, device=device)
    mutable = torch.tensor((~singleton_mask).astype(np.float64), dtype=dtype, device=device).view(1, 1, q_value)

    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[-1], hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, 1),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.zero_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    target_scale = torch.sqrt(torch.mean(torch.square(target))).clamp_min(EPS)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        raw = model(x).squeeze(-1)
        delta = rmax * torch.tanh(raw) * mutable
        corrected = torch.clamp(base + delta, min=EPS)
        loss = torch.mean(torch.square((corrected - target) / target_scale))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(options.get("grad_clip", 5.0)))
        optimizer.step()
        if step == 0 or step + 1 == steps or (step + 1) % log_every == 0:
            with torch.no_grad():
                rmse = torch.sqrt(torch.mean(torch.square(corrected - target)))
            history.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "tau_rmse": float(rmse.detach().cpu()),
                }
            )

    state_dict = model.state_dict()
    nn_state: dict[str, Any] = {
        "architecture": "q_space_species_overlap_residual_v1",
        "species_count": int(species_tau.shape[2]),
        "species_names": list(batch.species_names),
        "q_value": int(q_value),
        "hidden_dim": int(hidden_dim),
        "rmax": float(rmax),
        "feature_mean": feature_mean.astype(np.float64),
        "feature_std": feature_std.astype(np.float64),
        "singleton_mask": singleton_mask.astype(np.int8),
    }
    for key, value in state_dict.items():
        nn_state[_state_key(key)] = value.detach().cpu().numpy()
    log = {
        "kind": "q_space_species_overlap_tau_residual",
        "steps": int(steps),
        "lr": float(lr),
        "hidden_dim": int(hidden_dim),
        "rmax": float(rmax),
        "species_count": int(species_tau.shape[2]),
        "species_names": list(batch.species_names),
        "history": history,
    }
    if history:
        log["loss_final"] = history[-1]["loss"]
        log["tau_rmse_final"] = history[-1]["tau_rmse"]
    return nn_state, log


def apply_neural_overlap_residual(
    batch: NativeBatch,
    *,
    cluster_id: np.ndarray,
    weight_q: np.ndarray,
    q_value: int,
    nn_state: dict[str, Any],
    target_tau_q: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a frozen Q-space species-overlap tau residual."""

    species_tau = _required_species_tau(batch)
    expected_species = int(nn_state["species_count"])
    if species_tau.shape[2] != expected_species:
        raise ValueError(f"rt-aware-nn species count mismatch: {species_tau.shape[2]} != {expected_species}")
    species_tau_q = _compress_species_tau(species_tau, batch.spectral_weight, cluster_id, q_value)
    base_tau_q = np.sum(species_tau_q, axis=2)
    features = _feature_matrix(batch, species_tau_q, base_tau_q, weight_q)
    feature_mean = np.asarray(nn_state["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(nn_state["feature_std"], dtype=np.float64)
    x = (features - feature_mean) / np.maximum(feature_std, 1.0e-6)

    z1 = _linear(x, np.asarray(nn_state["net_0_weight"]), np.asarray(nn_state["net_0_bias"]))
    h1 = _silu(z1)
    z2 = _linear(h1, np.asarray(nn_state["net_2_weight"]), np.asarray(nn_state["net_2_bias"]))
    h2 = _silu(z2)
    raw = _linear(h2, np.asarray(nn_state["net_4_weight"]), np.asarray(nn_state["net_4_bias"]))[..., 0]
    singleton_mask = np.asarray(nn_state["singleton_mask"], dtype=bool)
    delta = float(nn_state["rmax"]) * np.tanh(raw)
    delta[..., singleton_mask] = 0.0
    corrected = np.maximum(base_tau_q + delta, EPS)
    if target_tau_q is not None:
        # Keep the Q=M identity endpoint exact even if a stale residual is present.
        identity = q_value == batch.tau_native.shape[-1] and np.array_equal(cluster_id, np.arange(q_value))
        if identity:
            corrected = np.maximum(target_tau_q, EPS)
    return corrected


def _required_species_tau(batch: NativeBatch) -> np.ndarray:
    if batch.species_tau_native is None:
        raise ValueError("rt-aware-nn requires species_tau_native from official CKDMIP spectra")
    species_tau = np.asarray(batch.species_tau_native, dtype=np.float64)
    if species_tau.ndim != 4:
        raise ValueError("species_tau_native must have shape [B,L,S,M]")
    if species_tau.shape[:2] != batch.tau_native.shape[:2] or species_tau.shape[-1] != batch.tau_native.shape[-1]:
        raise ValueError("species_tau_native dimensions must match tau_native")
    return np.maximum(species_tau, 0.0)


def _compressed_overlap_targets(
    batch: NativeBatch,
    *,
    species_tau: np.ndarray,
    cluster_id: np.ndarray,
    q_value: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    species_tau_q = _compress_species_tau(species_tau, batch.spectral_weight, cluster_id, q_value)
    base_tau_q = np.sum(species_tau_q, axis=2)
    target_tau_q = compress_tau(batch.tau_native, batch.spectral_weight, cluster_id, q_value)
    counts = np.bincount(np.asarray(cluster_id, dtype=np.int64), minlength=q_value)
    singleton_mask = counts == 1
    return species_tau_q, base_tau_q, target_tau_q, singleton_mask


def _compress_species_tau(
    species_tau: np.ndarray,
    spectral_weight: np.ndarray,
    cluster_id: np.ndarray,
    q_value: int,
) -> np.ndarray:
    parts = [
        compress_tau(species_tau[:, :, species_idx, :], spectral_weight, cluster_id, q_value)
        for species_idx in range(species_tau.shape[2])
    ]
    return np.stack(parts, axis=2)


def _feature_matrix(
    batch: NativeBatch,
    species_tau_q: np.ndarray,
    base_tau_q: np.ndarray,
    weight_q: np.ndarray,
) -> np.ndarray:
    layer_pressure = 0.5 * (batch.pressure_hl[:, :-1] + batch.pressure_hl[:, 1:])
    layer_temperature = 0.5 * (batch.temperature_hl[:, :-1] + batch.temperature_hl[:, 1:])
    batch_count, layer_count, species_count, q_value = species_tau_q.shape
    q_coord = np.linspace(0.0, 1.0, q_value, dtype=np.float64)
    log_weight = np.log(np.maximum(np.asarray(weight_q, dtype=np.float64), EPS))
    features = [
        np.broadcast_to(np.log(np.maximum(layer_pressure, EPS))[:, :, None], (batch_count, layer_count, q_value)),
        np.broadcast_to((layer_temperature / 300.0)[:, :, None], (batch_count, layer_count, q_value)),
        np.broadcast_to(q_coord[None, None, :], (batch_count, layer_count, q_value)),
        np.broadcast_to(log_weight[None, None, :], (batch_count, layer_count, q_value)),
        np.log1p(base_tau_q),
    ]
    for species_idx in range(species_count):
        features.append(np.log1p(species_tau_q[:, :, species_idx, :]))
    return np.stack(features, axis=-1).astype(np.float64)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.einsum("...i,oi->...o", x, weight) + bias


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _state_key(key: str) -> str:
    return "net_" + key.replace(".", "_")


def _torch_dtype(name: str, torch: Any) -> Any:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be float32 or float64")
