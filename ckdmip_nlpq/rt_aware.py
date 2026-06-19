"""RT-aware assignment training for longwave NLPQ models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .export import PLANCK_C1, PLANCK_C2
from .model import NativeBatch
from .rt import check_py2sess_forward_flux_available


EPS = 1.0e-12
GRAVITY_M_S2 = 9.80665
SECONDS_PER_DAY = 86400.0


def train_rt_aware_assignment(
    batch: NativeBatch,
    *,
    domain: str,
    q_value: int,
    seed: int,
    training_config: dict[str, Any] | None = None,
    py2sess_repo: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a hard LW assignment using differentiable flux/heating losses."""

    if domain != "lw":
        raise NotImplementedError("rt-aware training is currently implemented for longwave only")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when torch is absent
        raise ImportError("torch is required for rt-aware training") from exc

    options = dict(training_config or {})
    teacher = str(options.get("rt_train_teacher", options.get("train_teacher", "py2sess")))
    if teacher != "py2sess":
        raise ValueError("rt_train_teacher must be py2sess")
    py2sess_version = check_py2sess_forward_flux_available(py2sess_repo)

    dtype = _torch_dtype(str(options.get("dtype", "float32")), torch)
    device = torch.device(str(options.get("device", "cpu")))
    steps = int(options.get("steps", 220))
    lr = float(options.get("lr", 0.05))
    if steps < 1:
        raise ValueError("rt-aware training steps must be positive")
    if lr <= 0.0:
        raise ValueError("rt-aware training lr must be positive")

    tau = torch.tensor(np.maximum(batch.tau_native, 0.0), dtype=dtype, device=device)
    alpha = torch.tensor(_positive_normalized(batch.spectral_weight), dtype=dtype, device=device)
    pressure_hl = torch.tensor(batch.pressure_hl, dtype=dtype, device=device)
    source_level = _planck_level_source(batch, dtype=dtype, device=device, torch=torch)
    source_layer = 0.5 * (source_level[:, :-1, :] + source_level[:, 1:, :])
    surface_source = source_level[:, -1, :]

    m_count = int(tau.shape[-1])
    if q_value > m_count:
        raise ValueError(f"Q={q_value} exceeds native spectral count M={m_count}")
    if q_value == m_count:
        cluster_id = np.arange(m_count, dtype=np.int64)
        return cluster_id, _cluster_weights(batch.spectral_weight, cluster_id, q_value), {
            "rt_aware_training": "identity",
            "teacher_requested": teacher,
            "teacher_kernel": "py2sess_forward_flux",
            "py2sess_version": py2sess_version,
            "steps": 0,
        }

    streams = int(options.get("streams", options.get("lw_streams", 4)))
    cp_air = float(options.get("cp_air_j_kg_k", 1004.0))
    flux_weight = float(options.get("flux_loss_weight", 1.0))
    heating_weight = float(options.get("heating_loss_weight", 1.0))
    feature_weight = float(options.get("feature_loss_weight", 0.05))
    usage_weight = float(options.get("usage_loss_weight", 0.001))
    entropy_weight = float(options.get("entropy_loss_weight", 0.0005))
    temperature = float(options.get("assignment_temperature", 1.0))
    log_every = max(1, int(options.get("log_every_steps", max(1, steps // 5))))

    layer_mask = _layer_pressure_mask(
        batch.pressure_hl,
        min_hpa=float(options.get("train_pressure_min_hpa", 0.001)),
        max_hpa=float(options.get("train_pressure_max_hpa", 1100.0)),
    )
    level_mask = _level_pressure_mask(
        batch.pressure_hl,
        min_hpa=float(options.get("train_pressure_min_hpa", 0.001)),
        max_hpa=float(options.get("train_pressure_max_hpa", 1100.0)),
    )
    layer_mask_t = torch.tensor(layer_mask, dtype=torch.bool, device=device)
    level_mask_t = torch.tensor(level_mask, dtype=torch.bool, device=device)

    with torch.no_grad():
        ref_up, ref_down, ref_heat = py2sess_forward_flux_rt(
            tau,
            alpha,
            source_level,
            surface_source,
            pressure_hl,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
        )
        flux_scale = torch.sqrt(
            torch.mean(torch.square(torch.cat([ref_up[level_mask_t], ref_down[level_mask_t]])))
        ).clamp_min(EPS)
        heat_scale = torch.sqrt(torch.mean(torch.square(ref_heat[layer_mask_t]))).clamp_min(EPS)

    initial_cluster = _weighted_contiguous_clusters(batch.spectral_weight, q_value)
    logits = _initial_logits(
        initial_cluster,
        q_value=q_value,
        seed=seed,
        strength=float(options.get("init_strength", 4.0)),
        dtype=dtype,
        device=device,
        torch=torch,
    )
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    history: list[dict[str, float | int]] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.softmax(logits / max(temperature, EPS), dim=1)
        weights_q, tau_q, surface_q = compress_soft(
            probabilities,
            alpha,
            tau,
            surface_source,
            torch=torch,
        )
        source_level_q = compress_soft_level_source(
            probabilities,
            alpha,
            source_level,
            torch=torch,
        )
        up, down, heat = py2sess_forward_flux_rt(
            tau_q,
            weights_q,
            source_level_q,
            surface_q,
            pressure_hl,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
        )
        flux_err = torch.cat([(up - ref_up)[level_mask_t], (down - ref_down)[level_mask_t]])
        heat_err = (heat - ref_heat)[layer_mask_t]
        flux_loss = torch.mean(torch.square(flux_err / flux_scale))
        heat_loss = torch.mean(torch.square(heat_err / heat_scale))
        feature_loss = feature_reconstruction_loss(probabilities, alpha, tau, source_layer, torch=torch)
        target_weight = torch.full_like(weights_q, 1.0 / float(q_value))
        usage_loss = torch.mean(torch.square((weights_q - target_weight) / target_weight.clamp_min(EPS)))
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(EPS)), dim=1).mean()
        loss = (
            flux_weight * flux_loss
            + heating_weight * heat_loss
            + feature_weight * feature_loss
            + usage_weight * usage_loss
            + entropy_weight * entropy
        )
        loss.backward()
        optimizer.step()

        if step == 0 or step + 1 == steps or (step + 1) % log_every == 0:
            history.append(
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "flux_loss": float(flux_loss.detach().cpu()),
                    "heating_loss": float(heat_loss.detach().cpu()),
                    "feature_loss": float(feature_loss.detach().cpu()),
                    "usage_loss": float(usage_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                }
            )

    with torch.no_grad():
        probabilities = torch.softmax(logits / max(temperature, EPS), dim=1)
        prob_np = probabilities.detach().cpu().numpy()
    cluster_id = _repair_empty_clusters(np.argmax(prob_np, axis=1).astype(np.int64), prob_np, q_value)
    weight_q = _cluster_weights(batch.spectral_weight, cluster_id, q_value)
    log = {
        "rt_aware_training": "optimized",
        "teacher_requested": teacher,
        "teacher_kernel": "py2sess_forward_flux",
        "py2sess_version": py2sess_version,
        "steps": steps,
        "lr": lr,
        "streams": streams,
        "dtype": str(options.get("dtype", "float32")),
        "device": str(device),
        "train_pressure_min_hpa": float(options.get("train_pressure_min_hpa", 0.001)),
        "train_pressure_max_hpa": float(options.get("train_pressure_max_hpa", 1100.0)),
        "history": history,
    }
    if history:
        log["teacher_loss_final"] = history[-1]["loss"]
        log["teacher_flux_loss_final"] = history[-1]["flux_loss"]
        log["teacher_heating_loss_final"] = history[-1]["heating_loss"]
    return cluster_id, weight_q, log


def compress_soft(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    surface_source_bm: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, Any]:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    trans = torch.exp(-torch.clamp(tau_blm, min=0.0, max=700.0))
    avg_trans = torch.einsum("mq,blm->blq", weighted_mq, trans) / weights_q[None, None, :]
    tau_blq = -torch.log(avg_trans.clamp_min(EPS))
    surface_source_bq = torch.einsum("mq,bm->bq", weighted_mq, surface_source_bm) / weights_q[None, :]
    return weights_q, tau_blq.clamp_min(EPS), surface_source_bq


def compress_soft_level_source(
    probabilities_mq: Any,
    alpha_m: Any,
    source_level_bkm: Any,
    *,
    torch: Any,
) -> Any:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    return torch.einsum("mq,bkm->bkq", weighted_mq, source_level_bkm) / weights_q[None, None, :]


def py2sess_forward_flux_rt(
    tau_blq: Any,
    weights_q: Any,
    source_level_bkq: Any,
    surface_source_bq: Any,
    pressure_hl_pa: Any,
    *,
    streams: int,
    cp_air_j_kg_k: float,
    py2sess_repo: Path | None,
    dtype_name: str,
    torch: Any,
) -> tuple[Any, Any, Any]:
    check_py2sess_forward_flux_available(py2sess_repo)
    from py2sess import TwoStreamEss, TwoStreamEssOptions

    batch, layers, q_count = tau_blq.shape
    if source_level_bkq.shape != (batch, layers + 1, q_count):
        raise ValueError("source_level_bkq must have shape [B,L+1,Q]")
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=layers,
            mode="thermal",
            backend="torch",
            upwelling=True,
            downwelling=True,
            delta_scaling=False,
            plane_parallel=True,
            fo_flux_n_mu=int(streams),
            torch_dtype=dtype_name,
            torch_enable_grad=True,
        )
    )
    tau_rows = tau_blq.permute(0, 2, 1).reshape(batch * q_count, layers).contiguous().clamp_min(EPS)
    source_rows = source_level_bkq.permute(0, 2, 1).reshape(batch * q_count, layers + 1).contiguous()
    surface_rows = surface_source_bq.reshape(batch * q_count).contiguous()
    zeros = torch.zeros_like(tau_rows)
    result = solver.forward_flux(
        tau=tau_rows,
        ssa=zeros,
        g=zeros,
        z=_height_grid_from_pressure(pressure_hl_pa),
        angles=[0.0],
        stream=1.0 / math.sqrt(3.0),
        planck=source_rows,
        surface_planck=surface_rows,
        emissivity=torch.ones(batch * q_count, dtype=tau_blq.dtype, device=tau_blq.device),
        albedo=torch.zeros(batch * q_count, dtype=tau_blq.dtype, device=tau_blq.device),
        include_fo=True,
        return_net=True,
    )
    up_rows = result.flux_up
    down_rows = result.flux_down
    if up_rows.ndim == 3:
        up_rows = up_rows[..., 0, :]
        down_rows = down_rows[..., 0, :]
    up_bqk = up_rows.reshape(batch, q_count, layers + 1)
    down_bqk = down_rows.reshape(batch, q_count, layers + 1)
    up_flux = torch.einsum("q,bqk->bk", weights_q, up_bqk)
    down_flux = torch.einsum("q,bqk->bk", weights_q, down_bqk)
    net_flux = up_flux - down_flux
    dp = torch.diff(pressure_hl_pa, dim=1).clamp_min(EPS)
    heating = (net_flux[:, 1:] - net_flux[:, :-1]) * GRAVITY_M_S2 * SECONDS_PER_DAY / float(cp_air_j_kg_k) / dp
    return up_flux, down_flux, heating


def feature_reconstruction_loss(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    source_blm: Any,
    *,
    torch: Any,
) -> Any:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    log_tau = torch.log1p(tau_blm)
    log_tau_q = torch.einsum("mq,blm->blq", weighted_mq, log_tau) / weights_q[None, None, :]
    source_q = torch.einsum("mq,blm->blq", weighted_mq, source_blm) / weights_q[None, None, :]
    log_tau_hat = torch.einsum("mq,blq->blm", probabilities_mq, log_tau_q)
    source_hat = torch.einsum("mq,blq->blm", probabilities_mq, source_q)
    alpha_norm = alpha_m / alpha_m.sum().clamp_min(EPS)
    tau_loss = torch.einsum("m,blm->", alpha_norm, torch.square(log_tau_hat - log_tau)) / float(
        tau_blm.shape[0] * tau_blm.shape[1]
    )
    source_scale = torch.sqrt(torch.mean(torch.square(source_blm))).clamp_min(EPS)
    source_loss = torch.einsum(
        "m,blm->",
        alpha_norm,
        torch.square((source_hat - source_blm) / source_scale),
    ) / float(source_blm.shape[0] * source_blm.shape[1])
    return tau_loss + 0.05 * source_loss


def _planck_level_source(batch: NativeBatch, *, dtype: Any, device: Any, torch: Any) -> Any:
    wn = torch.tensor(batch.wavenumber, dtype=dtype, device=device)
    temperature = torch.tensor(batch.temperature_hl, dtype=dtype, device=device)
    exponent = PLANCK_C2 * wn[None, None, :] / temperature[:, :, None].clamp_min(1.0)
    return PLANCK_C1 * torch.pow(wn[None, None, :], 3) / torch.expm1(torch.clamp(exponent, min=1.0e-12, max=700.0))


def _height_grid_from_pressure(pressure_hl_pa: Any) -> np.ndarray:
    pressure = np.asarray(pressure_hl_pa.detach().cpu().numpy(), dtype=np.float64)
    mean_pressure = np.mean(pressure, axis=0) if pressure.ndim == 2 else pressure
    surface_pressure = float(mean_pressure[-1])
    height = 7.0 * np.log(surface_pressure / np.clip(mean_pressure, EPS, None))
    height[-1] = 0.0
    return height


def _torch_dtype(name: str, torch: Any) -> Any:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("training.dtype must be float32 or float64")


def _positive_normalized(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("spectral_weight must be a non-empty vector")
    if np.any(out <= 0.0):
        raise ValueError("spectral_weight must be strictly positive for rt-aware training")
    return out / float(np.sum(out))


def _weighted_contiguous_clusters(spectral_weight: np.ndarray, q_value: int) -> np.ndarray:
    weight = _positive_normalized(spectral_weight)
    m_count = weight.size
    if q_value > m_count:
        raise ValueError(f"Q={q_value} exceeds native spectral count M={m_count}")
    centers = np.cumsum(weight) - 0.5 * weight
    edges = np.linspace(0.0, 1.0, q_value + 1)
    cluster = np.searchsorted(edges[1:-1], centers, side="right").astype(np.int64)
    confidence = np.linspace(0.0, 1.0, m_count)
    return _repair_empty_clusters(cluster, np.repeat(confidence[:, None], q_value, axis=1), q_value)


def _initial_logits(
    cluster_id: np.ndarray,
    *,
    q_value: int,
    seed: int,
    strength: float,
    dtype: Any,
    device: Any,
    torch: Any,
) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    logits_np = np.full((cluster_id.size, q_value), -float(strength), dtype=np.float32)
    logits_np[np.arange(cluster_id.size), cluster_id] = float(strength)
    logits = torch.tensor(logits_np, dtype=dtype, device=device)
    noise = 0.01 * torch.randn(logits_np.shape, generator=generator, dtype=torch.float32)
    return logits + noise.to(dtype=dtype, device=device)


def _repair_empty_clusters(cluster_id: np.ndarray, probabilities_mq: np.ndarray, q_value: int) -> np.ndarray:
    cluster = np.asarray(cluster_id, dtype=np.int64).copy()
    if cluster.size < q_value:
        raise ValueError(f"Q={q_value} exceeds native spectral count M={cluster.size}")
    counts = np.bincount(cluster, minlength=q_value)
    missing = [idx for idx, count in enumerate(counts) if count == 0]
    if not missing:
        return cluster
    confidence = np.max(np.asarray(probabilities_mq, dtype=np.float64), axis=1)
    for missing_q in missing:
        counts = np.bincount(cluster, minlength=q_value)
        moved = False
        for native_idx in np.argsort(confidence):
            old_q = int(cluster[native_idx])
            if counts[old_q] > 1:
                cluster[native_idx] = missing_q
                moved = True
                break
        if not moved:
            raise ValueError("could not repair empty pseudo-line cluster")
    if np.any(np.bincount(cluster, minlength=q_value) == 0):
        raise ValueError("every pseudo-line cluster must be nonempty")
    return cluster


def _cluster_weights(spectral_weight: np.ndarray, cluster_id: np.ndarray, q_value: int) -> np.ndarray:
    weight = _positive_normalized(spectral_weight)
    out = np.bincount(np.asarray(cluster_id, dtype=np.int64), weights=weight, minlength=q_value).astype(np.float64)
    if np.any(out <= 0.0):
        raise ValueError("every pseudo-line cluster must be nonempty")
    return out / float(np.sum(out))


def _layer_pressure_mask(pressure_hl_pa: np.ndarray, *, min_hpa: float, max_hpa: float) -> np.ndarray:
    pressure = np.asarray(pressure_hl_pa, dtype=np.float64)
    mid_hpa = 0.5 * (pressure[:, :-1] + pressure[:, 1:]) * 0.01
    mask = (mid_hpa >= min_hpa) & (mid_hpa <= max_hpa)
    return mask if np.any(mask) else np.ones_like(mid_hpa, dtype=bool)


def _level_pressure_mask(pressure_hl_pa: np.ndarray, *, min_hpa: float, max_hpa: float) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hl_pa, dtype=np.float64) * 0.01
    mask = (pressure_hpa >= min_hpa) & (pressure_hpa <= max_hpa)
    return mask if np.any(mask) else np.ones_like(pressure_hpa, dtype=bool)
