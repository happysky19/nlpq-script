"""RT-aware assignment training for NLPQ models."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .export import PLANCK_C1, PLANCK_C2
from .model import NativeBatch, compression_settings
from .rt import check_py2sess_forward_flux_available


EPS = 1.0e-12
GRAVITY_M_S2 = 9.80665
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
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
    """Train a hard assignment using differentiable flux/heating losses."""

    if domain == "sw":
        return train_sw_rt_aware_assignment(
            batch,
            q_value=q_value,
            seed=seed,
            training_config=training_config,
            py2sess_repo=py2sess_repo,
        )

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
    temperature_hl = torch.tensor(batch.temperature_hl, dtype=dtype, device=device)
    source_level = _planck_level_source(batch, dtype=dtype, device=device, torch=torch)
    source_layer = 0.5 * (source_level[:, :-1, :] + source_level[:, 1:, :])
    surface_source = source_level[:, -1, :]
    lw_source_mode = str(options.get("lw_source_mode", options.get("source_mode", "ckdmip_integrated")))
    if lw_source_mode != "ckdmip_integrated":
        raise ValueError("lw_source_mode must be ckdmip_integrated for CKDMIP export")
    if lw_source_mode == "ckdmip_integrated":
        reference_source_level = source_level * alpha[None, None, :]
        reference_surface_source = surface_source * alpha[None, :]
        reference_weights = torch.ones_like(alpha)
    else:
        reference_source_level = source_level
        reference_surface_source = surface_source
        reference_weights = alpha

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
    flux_weight = _training_float(options, "rt_aware_flux_weight", "flux_loss_weight", default=1.0)
    heating_weight = _training_float(options, "rt_aware_heating_weight", "heating_loss_weight", default=10.0)
    path_weight = _training_float(options, "rt_aware_path_weight", "path_loss_weight", default=0.05)
    feature_weight = _training_float(options, "rt_aware_feature_weight", "feature_loss_weight", default=0.05)
    usage_weight = float(options.get("usage_loss_weight", 0.001))
    entropy_weight = float(options.get("entropy_loss_weight", 0.0005))
    path_spectral_chunk = int(options.get("path_loss_spectral_chunk", 8192))
    temperature = float(options.get("assignment_temperature", 1.0))
    log_every = max(1, int(options.get("log_every_steps", max(1, steps // 5))))
    max_rt_rows = _max_py2sess_rows(options)
    use_checkpoint = _use_gradient_checkpointing(options)

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
            reference_weights,
            reference_source_level,
            reference_surface_source,
            pressure_hl,
            temperature_hl,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
            max_rows=max_rt_rows,
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
        if lw_source_mode == "ckdmip_integrated":
            weights_q, tau_q, source_level_q, surface_q = _checkpoint_if_enabled(
                use_checkpoint,
                lambda p: compress_soft_integrated_source(p, alpha, tau, source_level, surface_source, torch=torch),
                probabilities,
            )
            flux_weights_q = torch.ones_like(weights_q)
        else:
            weights_q, tau_q, surface_q = _checkpoint_if_enabled(
                use_checkpoint,
                lambda p: compress_soft(p, alpha, tau, surface_source, torch=torch),
                probabilities,
            )
            source_level_q = _checkpoint_if_enabled(
                use_checkpoint,
                lambda p: compress_soft_level_source(p, alpha, source_level, torch=torch),
                probabilities,
            )
            flux_weights_q = weights_q
        up, down, heat = py2sess_forward_flux_rt(
            tau_q,
            flux_weights_q,
            source_level_q,
            surface_q,
            pressure_hl,
            temperature_hl,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
            max_rows=max_rt_rows,
        )
        flux_err = torch.cat([(up - ref_up)[level_mask_t], (down - ref_down)[level_mask_t]])
        heat_err = (heat - ref_heat)[layer_mask_t]
        flux_loss = torch.mean(torch.square(flux_err / flux_scale))
        heat_loss = torch.mean(torch.square(heat_err / heat_scale))
        feature_loss = _checkpoint_if_enabled(
            use_checkpoint,
            lambda p: feature_reconstruction_loss(p, alpha, tau, source_layer, torch=torch),
            probabilities,
        )
        path_loss = tau.new_tensor(0.0)
        if path_weight != 0.0:
            path_loss = lw_source_path_loss(
                probabilities,
                alpha,
                tau,
                source_level,
                tau_q,
                source_level_q,
                weights_q,
                source_mode=lw_source_mode,
                spectral_chunk=path_spectral_chunk,
                torch=torch,
            )
        target_weight = torch.full_like(weights_q, 1.0 / float(q_value))
        usage_loss = torch.mean(torch.square((weights_q - target_weight) / target_weight.clamp_min(EPS)))
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(EPS)), dim=1).mean()
        loss = (
            flux_weight * flux_loss
            + heating_weight * heat_loss
            + path_weight * path_loss
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
                    "path_loss": float(path_loss.detach().cpu()),
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
        "lw_source_mode": lw_source_mode,
        "rt_aware_method_variant": str(options.get("rt_aware_method_variant", "rt-aware")),
        "flux_loss_weight": flux_weight,
        "heating_loss_weight": heating_weight,
        "path_loss_weight": path_weight,
        "feature_loss_weight": feature_weight,
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
        log["teacher_path_loss_final"] = history[-1]["path_loss"]
    return cluster_id, weight_q, log


def train_sw_rt_aware_assignment(
    batch: NativeBatch,
    *,
    q_value: int,
    seed: int,
    training_config: dict[str, Any] | None = None,
    py2sess_repo: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train a hard SW assignment using py2sess solar level-flux losses."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when torch is absent
        raise ImportError("torch is required for rt-aware training") from exc

    options = dict(training_config or {})
    teacher = str(options.get("rt_train_teacher", options.get("train_teacher", "py2sess")))
    if teacher != "py2sess":
        raise ValueError("rt_train_teacher must be py2sess")
    py2sess_version = check_py2sess_forward_flux_available(py2sess_repo)
    if batch.incoming_flux_native is None:
        raise ValueError("SW rt-aware training requires incoming_flux_native from CKDMIP SSI")
    if batch.rayleigh_tau_native is None and not bool(options.get("allow_zero_rayleigh", False)):
        raise ValueError("SW rt-aware training requires rayleigh_tau_native")
    settings = compression_settings("sw", options)
    sw_tau_mode = str(settings["sw_tau_mode"])
    sw_rayleigh_mode = str(settings["sw_rayleigh_mode"])
    sw_tau_mu_ref = float(settings["sw_tau_mu_ref"])

    dtype = _torch_dtype(str(options.get("dtype", "float32")), torch)
    device = torch.device(str(options.get("device", "cpu")))
    steps = int(options.get("steps", 220))
    lr = float(options.get("lr", 0.05))
    if steps < 1:
        raise ValueError("rt-aware training steps must be positive")
    if lr <= 0.0:
        raise ValueError("rt-aware training lr must be positive")

    tau_abs = torch.tensor(np.maximum(batch.tau_native, 0.0), dtype=dtype, device=device)
    if batch.rayleigh_tau_native is None:
        rayleigh = torch.zeros_like(tau_abs)
    else:
        rayleigh = torch.tensor(np.maximum(batch.rayleigh_tau_native, 0.0), dtype=dtype, device=device)
    incoming = _incoming_flux_tensor(batch, dtype=dtype, device=device, torch=torch)
    alpha = torch.tensor(_positive_normalized(batch.spectral_weight), dtype=dtype, device=device)
    pressure_hl = torch.tensor(batch.pressure_hl, dtype=dtype, device=device)
    temperature_hl = torch.tensor(batch.temperature_hl, dtype=dtype, device=device)

    m_count = int(tau_abs.shape[-1])
    if q_value > m_count:
        raise ValueError(f"Q={q_value} exceeds native spectral count M={m_count}")
    if q_value == m_count:
        cluster_id = np.arange(m_count, dtype=np.int64)
        return cluster_id, _cluster_weights(batch.spectral_weight, cluster_id, q_value), {
            "rt_aware_training": "identity",
            "teacher_requested": teacher,
            "teacher_kernel": "py2sess_forward_flux_sw",
            "py2sess_version": py2sess_version,
            "steps": 0,
        }

    mu_values = [float(v) for v in options.get("mu_values", [0.5])]
    streams = int(options.get("streams", options.get("sw_streams", 4)))
    cp_air = float(options.get("cp_air_j_kg_k", 1004.0))
    surf_albedo = float(options.get("surf_albedo", 0.15))
    include_fo = bool(options.get("include_fo", options.get("sw_include_fo", True)))
    plane_parallel = bool(options.get("sw_plane_parallel", options.get("plane_parallel", True)))
    if include_fo and not plane_parallel:
        raise ValueError("SW forward_flux include_fo=True requires sw_plane_parallel=true")
    stream_value = float(options.get("stream", 1.0 / math.sqrt(3.0)))
    flux_weight = _training_float(options, "rt_aware_flux_weight", "flux_loss_weight", default=1.0)
    heating_weight = _training_float(options, "rt_aware_heating_weight", "heating_loss_weight", default=4.0)
    path_weight = _training_float(options, "rt_aware_path_weight", "path_loss_weight", default=0.05)
    feature_weight = _training_float(options, "rt_aware_feature_weight", "feature_loss_weight", default=0.05)
    usage_weight = float(options.get("usage_loss_weight", 0.001))
    entropy_weight = float(options.get("entropy_loss_weight", 0.0005))
    path_spectral_chunk = int(options.get("path_loss_spectral_chunk", 8192))
    temperature = float(options.get("assignment_temperature", 1.0))
    log_every = max(1, int(options.get("log_every_steps", max(1, steps // 5))))
    max_rt_rows = _max_py2sess_rows(options)
    use_checkpoint = _use_gradient_checkpointing(options)

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
    layer_mask_t = torch.tensor(layer_mask, dtype=torch.bool, device=device).unsqueeze(0)
    level_mask_t = torch.tensor(level_mask, dtype=torch.bool, device=device).unsqueeze(0)

    with torch.no_grad():
        ref_up, ref_down, ref_heat = py2sess_forward_flux_sw(
            tau_abs,
            rayleigh,
            incoming,
            pressure_hl,
            temperature_hl,
            mu_values=mu_values,
            surf_albedo=surf_albedo,
            include_fo=include_fo,
            plane_parallel=plane_parallel,
            stream=stream_value,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
            max_rows=max_rt_rows,
        )
        level_mask_g = level_mask_t.expand(ref_up.shape[0], -1, -1)
        layer_mask_g = layer_mask_t.expand(ref_heat.shape[0], -1, -1)
        flux_scale = torch.sqrt(
            torch.mean(torch.square(torch.cat([ref_up[level_mask_g], ref_down[level_mask_g]])))
        ).clamp_min(EPS)
        heat_scale = torch.sqrt(torch.mean(torch.square(ref_heat[layer_mask_g]))).clamp_min(EPS)

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
        weights_q, tau_abs_q, rayleigh_q, incoming_q = _checkpoint_if_enabled(
            use_checkpoint,
            lambda p: compress_soft_sw(
                p,
                alpha,
                tau_abs,
                rayleigh,
                incoming,
                tau_mode=sw_tau_mode,
                rayleigh_mode=sw_rayleigh_mode,
                mu_ref=sw_tau_mu_ref,
                torch=torch,
            ),
            probabilities,
        )
        up, down, heat = py2sess_forward_flux_sw(
            tau_abs_q,
            rayleigh_q,
            incoming_q,
            pressure_hl,
            temperature_hl,
            mu_values=mu_values,
            surf_albedo=surf_albedo,
            include_fo=include_fo,
            plane_parallel=plane_parallel,
            stream=stream_value,
            streams=streams,
            cp_air_j_kg_k=cp_air,
            py2sess_repo=py2sess_repo,
            dtype_name=str(options.get("dtype", "float32")),
            torch=torch,
            max_rows=max_rt_rows,
        )
        level_mask_g = level_mask_t.expand(up.shape[0], -1, -1)
        layer_mask_g = layer_mask_t.expand(heat.shape[0], -1, -1)
        flux_err = torch.cat([(up - ref_up)[level_mask_g], (down - ref_down)[level_mask_g]])
        heat_err = (heat - ref_heat)[layer_mask_g]
        flux_loss = torch.mean(torch.square(flux_err / flux_scale))
        heat_loss = torch.mean(torch.square(heat_err / heat_scale))
        feature_loss = _checkpoint_if_enabled(
            use_checkpoint,
            lambda p: sw_feature_reconstruction_loss(p, alpha, tau_abs, rayleigh, incoming, torch=torch),
            probabilities,
        )
        path_loss = tau_abs.new_tensor(0.0)
        if path_weight != 0.0:
            path_loss = sw_direct_path_loss(
                probabilities,
                tau_abs,
                rayleigh,
                incoming,
                tau_abs_q,
                rayleigh_q,
                incoming_q,
                mu_values=mu_values,
                spectral_chunk=path_spectral_chunk,
                torch=torch,
            )
        target_weight = torch.full_like(weights_q, 1.0 / float(q_value))
        usage_loss = torch.mean(torch.square((weights_q - target_weight) / target_weight.clamp_min(EPS)))
        entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(EPS)), dim=1).mean()
        loss = (
            flux_weight * flux_loss
            + heating_weight * heat_loss
            + path_weight * path_loss
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
                    "path_loss": float(path_loss.detach().cpu()),
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
        "teacher_kernel": "py2sess_forward_flux_sw",
        "py2sess_version": py2sess_version,
        "steps": steps,
        "lr": lr,
        "streams": streams,
        "mu_values": mu_values,
        "surf_albedo": surf_albedo,
        "include_fo": include_fo,
        "sw_plane_parallel": plane_parallel,
        "sw_tau_mode": sw_tau_mode,
        "sw_rayleigh_mode": sw_rayleigh_mode,
        "sw_tau_mu_ref": sw_tau_mu_ref,
        "rt_aware_method_variant": str(options.get("rt_aware_method_variant", "rt-aware")),
        "flux_loss_weight": flux_weight,
        "heating_loss_weight": heating_weight,
        "path_loss_weight": path_weight,
        "feature_loss_weight": feature_weight,
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
        log["teacher_path_loss_final"] = history[-1]["path_loss"]
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


def compress_soft_integrated_source(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    source_level_bkm: Any,
    surface_source_bm: Any,
    *,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    """Compress LW tau with transmittance closure and CKDMIP-style integrated source."""

    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    trans = torch.exp(-torch.clamp(tau_blm, min=0.0, max=700.0))
    avg_trans = torch.einsum("mq,blm->blq", weighted_mq, trans) / weights_q[None, None, :]
    tau_blq = (-torch.log(avg_trans.clamp_min(EPS))).clamp_min(EPS)
    source_level_bkq = torch.einsum("mq,bkm->bkq", weighted_mq, source_level_bkm)
    surface_source_bq = torch.einsum("mq,bm->bq", weighted_mq, surface_source_bm)
    return weights_q, tau_blq, source_level_bkq, surface_source_bq


def compress_soft_sw(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_abs_blm: Any,
    rayleigh_blm: Any,
    incoming_bm: Any,
    *,
    tau_mode: str = "direct_beam",
    rayleigh_mode: str = "arithmetic",
    mu_ref: float = 0.5,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    if tau_mode == "direct_beam":
        tau_abs_blq = _compress_tau_soft_direct_beam(
            probabilities_mq,
            alpha_m,
            tau_abs_blm,
            incoming_bm,
            mu_ref=mu_ref,
            torch=torch,
        )
    elif tau_mode == "transmittance":
        tau_abs_blq = _compress_tau_soft(probabilities_mq, alpha_m, tau_abs_blm, torch=torch)
    else:
        raise ValueError("sw_tau_mode must be transmittance or direct_beam")

    if rayleigh_mode == "arithmetic":
        rayleigh_blq = _compress_tau_soft_solar_mean(probabilities_mq, alpha_m, rayleigh_blm, incoming_bm, torch=torch)
    elif rayleigh_mode == "direct_beam":
        rayleigh_blq = _compress_tau_soft_direct_beam(
            probabilities_mq,
            alpha_m,
            rayleigh_blm,
            incoming_bm,
            mu_ref=mu_ref,
            torch=torch,
        )
    elif rayleigh_mode == "transmittance":
        rayleigh_blq = _compress_tau_soft(probabilities_mq, alpha_m, rayleigh_blm, torch=torch)
    else:
        raise ValueError("sw_rayleigh_mode must be transmittance, arithmetic, or direct_beam")
    incoming_bq = torch.einsum("mq,bm->bq", probabilities_mq, incoming_bm).clamp_min(0.0)
    return weights_q, tau_abs_blq, rayleigh_blq, incoming_bq


def _compress_tau_soft(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    *,
    torch: Any,
) -> Any:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    trans = torch.exp(-torch.clamp(tau_blm, min=0.0, max=700.0))
    avg_trans = torch.einsum("mq,blm->blq", weighted_mq, trans) / weights_q[None, None, :]
    return (-torch.log(avg_trans.clamp_min(EPS))).clamp_min(EPS)


def _compress_tau_soft_direct_beam(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    incoming_bm: Any,
    *,
    mu_ref: float,
    torch: Any,
) -> Any:
    source_bmq, denom_bq = _soft_source_weights(probabilities_mq, alpha_m, incoming_bm, torch=torch)
    mu = max(float(mu_ref), EPS)
    trans = torch.exp(-torch.clamp(torch.clamp(tau_blm, min=0.0) / mu, max=700.0))
    avg_trans = torch.einsum("bmq,blm->blq", source_bmq, trans) / denom_bq[:, None, :]
    return (-mu * torch.log(avg_trans.clamp_min(EPS))).clamp_min(EPS)


def _compress_tau_soft_solar_mean(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    incoming_bm: Any,
    *,
    torch: Any,
) -> Any:
    source_bmq, denom_bq = _soft_source_weights(probabilities_mq, alpha_m, incoming_bm, torch=torch)
    return (
        torch.einsum("bmq,blm->blq", source_bmq, torch.clamp(tau_blm, min=0.0))
        / denom_bq[:, None, :]
    ).clamp_min(EPS)


def _soft_source_weights(probabilities_mq: Any, alpha_m: Any, incoming_bm: Any, *, torch: Any) -> tuple[Any, Any]:
    source_bmq = torch.clamp(incoming_bm, min=0.0)[:, :, None] * probabilities_mq[None, :, :]
    source_denom_bq = source_bmq.sum(dim=1)
    fallback_mq = torch.clamp(alpha_m, min=0.0)[:, None] * probabilities_mq
    fallback_denom_q = fallback_mq.sum(dim=0).clamp_min(EPS)
    use_source_bq = source_denom_bq > EPS
    weights_bmq = torch.where(use_source_bq[:, None, :], source_bmq, fallback_mq[None, :, :])
    denom_bq = torch.where(use_source_bq, source_denom_bq, fallback_denom_q[None, :]).clamp_min(EPS)
    return weights_bmq, denom_bq


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
    temperature_hl_k: Any,
    *,
    streams: int,
    cp_air_j_kg_k: float,
    py2sess_repo: Path | None,
    dtype_name: str,
    torch: Any,
    max_rows: int | None = None,
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
    height_grid = _hydrostatic_height_grid(pressure_hl_pa, temperature_hl_k)
    up_flux = tau_blq.new_zeros((batch, layers + 1))
    down_flux = tau_blq.new_zeros((batch, layers + 1))
    for q_slice in _q_slices(batch, q_count, _resolve_max_rows(max_rows)):
        q_len = int(q_slice.stop - q_slice.start)
        tau_rows = tau_blq[:, :, q_slice].permute(0, 2, 1).reshape(batch * q_len, layers).contiguous().clamp_min(EPS)
        source_rows = source_level_bkq[:, :, q_slice].permute(0, 2, 1).reshape(batch * q_len, layers + 1).contiguous()
        surface_rows = surface_source_bq[:, q_slice].reshape(batch * q_len).contiguous()
        zeros = torch.zeros_like(tau_rows)
        result = solver.forward_flux(
            tau=tau_rows,
            ssa=zeros,
            g=zeros,
            z=height_grid,
            angles=[0.0],
            stream=1.0 / math.sqrt(3.0),
            planck=source_rows,
            surface_planck=surface_rows,
            emissivity=torch.ones(batch * q_len, dtype=tau_blq.dtype, device=tau_blq.device),
            albedo=torch.zeros(batch * q_len, dtype=tau_blq.dtype, device=tau_blq.device),
            include_fo=True,
            return_net=True,
        )
        up_rows = result.flux_up
        down_rows = result.flux_down
        if up_rows.ndim == 3:
            up_rows = up_rows[..., 0, :]
            down_rows = down_rows[..., 0, :]
        up_bqk = up_rows.reshape(batch, q_len, layers + 1)
        down_bqk = down_rows.reshape(batch, q_len, layers + 1)
        up_flux = up_flux + torch.einsum("q,bqk->bk", weights_q[q_slice], up_bqk)
        down_flux = down_flux + torch.einsum("q,bqk->bk", weights_q[q_slice], down_bqk)
    net_flux = up_flux - down_flux
    dp = torch.diff(pressure_hl_pa, dim=1).clamp_min(EPS)
    heating = (net_flux[:, 1:] - net_flux[:, :-1]) * GRAVITY_M_S2 * SECONDS_PER_DAY / float(cp_air_j_kg_k) / dp
    return up_flux, down_flux, heating


def py2sess_forward_flux_sw(
    tau_abs_blq: Any,
    rayleigh_blq: Any,
    incoming_bq: Any,
    pressure_hl_pa: Any,
    temperature_hl_k: Any,
    *,
    mu_values: list[float],
    surf_albedo: float,
    include_fo: bool,
    plane_parallel: bool,
    stream: float,
    streams: int,
    cp_air_j_kg_k: float,
    py2sess_repo: Path | None,
    dtype_name: str,
    torch: Any,
    max_rows: int | None = None,
) -> tuple[Any, Any, Any]:
    check_py2sess_forward_flux_available(py2sess_repo)
    from py2sess import TwoStreamEss, TwoStreamEssOptions

    batch, layers, q_count = tau_abs_blq.shape
    if rayleigh_blq.shape != (batch, layers, q_count):
        raise ValueError("rayleigh_blq must have shape [B,L,Q]")
    if incoming_bq.ndim == 1:
        incoming_bq = incoming_bq[None, :].expand(batch, -1)
    if incoming_bq.shape != (batch, q_count):
        raise ValueError("incoming_bq must have shape [B,Q]")
    if not mu_values:
        raise ValueError("mu_values must be non-empty for SW rt-aware training")

    tau_total = (tau_abs_blq + rayleigh_blq).clamp_min(EPS)
    ssa_blq = torch.clamp(rayleigh_blq / tau_total, min=0.0, max=0.999999)
    g_blq = torch.zeros_like(tau_total)
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=layers,
            mode="solar",
            backend="torch",
            upwelling=True,
            downwelling=True,
            delta_scaling=False,
            plane_parallel=plane_parallel,
            fo_flux_n_mu=int(streams),
            torch_dtype=dtype_name,
            torch_enable_grad=True,
        )
    )
    height_grid = _hydrostatic_height_grid(pressure_hl_pa, temperature_hl_k)
    up_by_mu = []
    down_by_mu = []
    heat_by_mu = []
    for mu0 in mu_values:
        mu0_clamped = max(min(float(mu0), 1.0), 1.0e-6)
        sza = math.degrees(math.acos(mu0_clamped))
        up_flux = tau_abs_blq.new_zeros((batch, layers + 1))
        down_flux = tau_abs_blq.new_zeros((batch, layers + 1))
        net_flux = tau_abs_blq.new_zeros((batch, layers + 1))
        for q_slice in _q_slices(batch, q_count, _resolve_max_rows(max_rows)):
            q_len = int(q_slice.stop - q_slice.start)
            row_count = batch * q_len
            tau_rows = tau_total[:, :, q_slice].permute(0, 2, 1).reshape(row_count, layers).contiguous()
            ssa_rows = ssa_blq[:, :, q_slice].permute(0, 2, 1).reshape(row_count, layers).contiguous()
            g_rows = g_blq[:, :, q_slice].permute(0, 2, 1).reshape(row_count, layers).contiguous()
            incoming_rows = incoming_bq[:, q_slice].reshape(row_count).contiguous()
            albedo_rows = torch.full((row_count,), float(surf_albedo), dtype=tau_abs_blq.dtype, device=tau_abs_blq.device)
            result = solver.forward_flux(
                tau=tau_rows,
                ssa=ssa_rows,
                g=g_rows,
                z=height_grid,
                angles=[sza, 0.0, 0.0],
                stream=stream,
                fbeam=incoming_rows,
                albedo=albedo_rows,
                include_fo=include_fo,
                return_net=True,
            )
            up_rows = _flux_rows(result.flux_up, row_count)
            down_rows = _flux_rows(result.flux_down, row_count)
            net_rows = _net_flux_rows(result, up_rows, down_rows, row_count)
            up_flux = up_flux + up_rows.reshape(batch, q_len, layers + 1).sum(dim=1)
            down_flux = down_flux + down_rows.reshape(batch, q_len, layers + 1).sum(dim=1)
            net_flux = net_flux + net_rows.reshape(batch, q_len, layers + 1).sum(dim=1)
        dp = torch.diff(pressure_hl_pa, dim=1).clamp_min(EPS)
        heating = (net_flux[:, 1:] - net_flux[:, :-1]) * GRAVITY_M_S2 * SECONDS_PER_DAY / float(cp_air_j_kg_k) / dp
        up_by_mu.append(up_flux)
        down_by_mu.append(down_flux)
        heat_by_mu.append(heating)
    return torch.stack(up_by_mu, dim=0), torch.stack(down_by_mu, dim=0), torch.stack(heat_by_mu, dim=0)


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


def sw_feature_reconstruction_loss(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_abs_blm: Any,
    rayleigh_blm: Any,
    incoming_bm: Any,
    *,
    torch: Any,
) -> Any:
    weighted_mq = alpha_m[:, None] * probabilities_mq
    weights_q = weighted_mq.sum(dim=0).clamp_min(EPS)
    total_tau = torch.log1p(tau_abs_blm + rayleigh_blm)
    total_tau_q = torch.einsum("mq,blm->blq", weighted_mq, total_tau) / weights_q[None, None, :]
    total_tau_hat = torch.einsum("mq,blq->blm", probabilities_mq, total_tau_q)
    alpha_norm = alpha_m / alpha_m.sum().clamp_min(EPS)
    tau_loss = torch.einsum("m,blm->", alpha_norm, torch.square(total_tau_hat - total_tau)) / float(
        tau_abs_blm.shape[0] * tau_abs_blm.shape[1]
    )
    incoming_m = torch.mean(incoming_bm, dim=0)
    incoming_density = incoming_m / alpha_m.clamp_min(EPS)
    incoming_density_q = torch.einsum("mq,m->q", probabilities_mq, incoming_m) / weights_q
    incoming_density_hat = torch.einsum("mq,q->m", probabilities_mq, incoming_density_q)
    incoming_scale = torch.sqrt(torch.einsum("m,m->", alpha_norm, torch.square(incoming_density))).clamp_min(EPS)
    incoming_loss = torch.einsum(
        "m,m->",
        alpha_norm,
        torch.square((incoming_density_hat - incoming_density) / incoming_scale),
    )
    return tau_loss + 0.05 * incoming_loss


def lw_source_path_loss(
    probabilities_mq: Any,
    alpha_m: Any,
    tau_blm: Any,
    source_level_bkm: Any,
    tau_blq: Any,
    source_level_bkq: Any,
    weights_q: Any,
    *,
    source_mode: str,
    spectral_chunk: int,
    torch: Any,
) -> Any:
    """Match source-weighted cumulative LW escape proxies, not only layer tau."""

    if source_mode == "ckdmip_integrated":
        source_weight_bkq = source_level_bkq
    elif source_mode == "weighted_mean":
        source_weight_bkq = source_level_bkq * weights_q[None, None, :]
    else:
        raise ValueError("unknown LW source mode")
    mu = 1.0 / math.sqrt(3.0)
    q_top, q_surface = _level_cumulative_tau(tau_blq, torch=torch)
    pred_top = torch.sum(source_weight_bkq * torch.exp(-torch.clamp(q_top / mu, max=700.0)), dim=-1)
    pred_surface = torch.sum(source_weight_bkq * torch.exp(-torch.clamp(q_surface / mu, max=700.0)), dim=-1)

    ref_top = tau_blm.new_zeros(pred_top.shape)
    ref_surface = tau_blm.new_zeros(pred_surface.shape)
    chunk = max(1, int(spectral_chunk))
    for start in range(0, int(tau_blm.shape[-1]), chunk):
        stop = min(int(tau_blm.shape[-1]), start + chunk)
        tau_chunk = torch.clamp(tau_blm[:, :, start:stop], min=0.0)
        source_chunk = source_level_bkm[:, :, start:stop] * alpha_m[start:stop][None, None, :]
        native_top, native_surface = _level_cumulative_tau(tau_chunk, torch=torch)
        ref_top = ref_top + torch.sum(
            source_chunk * torch.exp(-torch.clamp(native_top / mu, max=700.0)),
            dim=-1,
        )
        ref_surface = ref_surface + torch.sum(
            source_chunk * torch.exp(-torch.clamp(native_surface / mu, max=700.0)),
            dim=-1,
        )

    reference = torch.cat([ref_top.reshape(-1), ref_surface.reshape(-1)])
    error = torch.cat([(pred_top - ref_top).reshape(-1), (pred_surface - ref_surface).reshape(-1)])
    scale = torch.sqrt(torch.mean(torch.square(reference))).clamp_min(EPS)
    return torch.mean(torch.square(error / scale))


def sw_direct_path_loss(
    probabilities_mq: Any,
    tau_abs_blm: Any,
    rayleigh_blm: Any,
    incoming_bm: Any,
    tau_abs_blq: Any,
    rayleigh_blq: Any,
    incoming_bq: Any,
    *,
    mu_values: list[float],
    spectral_chunk: int,
    torch: Any,
) -> Any:
    """Match cumulative direct-beam SW path transmittance at model levels."""

    del probabilities_mq  # The compressed tensors already carry the differentiable dependence.
    if incoming_bq.ndim == 1:
        incoming_bq = incoming_bq[None, :].expand(tau_abs_blq.shape[0], -1)
    total_q = torch.clamp(tau_abs_blq + rayleigh_blq, min=0.0)
    total_native = torch.clamp(tau_abs_blm + rayleigh_blm, min=0.0)
    chunk = max(1, int(spectral_chunk))
    errors = []
    references = []
    for raw_mu in mu_values:
        mu = max(min(float(raw_mu), 1.0), 1.0e-6)
        q_path, _ = _level_cumulative_tau(total_q, torch=torch)
        pred = torch.sum(incoming_bq[:, None, :] * torch.exp(-torch.clamp(q_path / mu, max=700.0)), dim=-1)
        ref = tau_abs_blm.new_zeros(pred.shape)
        for start in range(0, int(total_native.shape[-1]), chunk):
            stop = min(int(total_native.shape[-1]), start + chunk)
            native_path, _ = _level_cumulative_tau(total_native[:, :, start:stop], torch=torch)
            ref = ref + torch.sum(
                incoming_bm[:, None, start:stop] * torch.exp(-torch.clamp(native_path / mu, max=700.0)),
                dim=-1,
            )
        references.append(ref.reshape(-1))
        errors.append((pred - ref).reshape(-1))
    reference = torch.cat(references)
    error = torch.cat(errors)
    scale = torch.sqrt(torch.mean(torch.square(reference))).clamp_min(EPS)
    return torch.mean(torch.square(error / scale))


def _level_cumulative_tau(tau_blx: Any, *, torch: Any) -> tuple[Any, Any]:
    tau = torch.clamp(tau_blx, min=0.0)
    batch, _, count = tau.shape
    zero = tau.new_zeros((batch, 1, count))
    top_path = torch.cat([zero, torch.cumsum(tau, dim=1)], dim=1)
    suffix = torch.flip(torch.cumsum(torch.flip(tau, dims=[1]), dim=1), dims=[1])
    surface_path = torch.cat([suffix, zero], dim=1)
    return top_path, surface_path


def _training_float(options: dict[str, Any], canonical: str, legacy: str, *, default: float) -> float:
    if canonical in options:
        return float(options[canonical])
    if legacy in options:
        return float(options[legacy])
    return float(default)


def _planck_level_source(batch: NativeBatch, *, dtype: Any, device: Any, torch: Any) -> Any:
    wn = torch.tensor(batch.wavenumber, dtype=dtype, device=device)
    temperature = torch.tensor(batch.temperature_hl, dtype=dtype, device=device)
    exponent = PLANCK_C2 * wn[None, None, :] / temperature[:, :, None].clamp_min(1.0)
    return PLANCK_C1 * torch.pow(wn[None, None, :], 3) / torch.expm1(torch.clamp(exponent, min=1.0e-12, max=700.0))


def _incoming_flux_tensor(batch: NativeBatch, *, dtype: Any, device: Any, torch: Any) -> Any:
    if batch.incoming_flux_native is None:
        raise ValueError("incoming_flux_native is required")
    incoming = np.asarray(batch.incoming_flux_native, dtype=np.float64)
    batch_count = int(batch.tau_native.shape[0])
    spectral_count = int(batch.tau_native.shape[-1])
    if incoming.ndim == 1:
        if incoming.shape[0] != spectral_count:
            raise ValueError("incoming_flux_native spectral dimension does not match tau_native")
        incoming = np.broadcast_to(incoming[None, :], (batch_count, spectral_count))
    elif incoming.ndim == 2:
        if incoming.shape != (batch_count, spectral_count):
            raise ValueError("incoming_flux_native must have shape [M] or [B,M]")
    else:
        raise ValueError("incoming_flux_native must have shape [M] or [B,M]")
    return torch.tensor(np.maximum(incoming, 0.0), dtype=dtype, device=device)


def _hydrostatic_height_grid(pressure_hl_pa: Any, temperature_hl_k: Any) -> np.ndarray:
    pressure = _to_numpy(pressure_hl_pa)
    temperature = _to_numpy(temperature_hl_k)
    mean_pressure = np.mean(pressure, axis=0) if pressure.ndim == 2 else pressure
    mean_temperature = np.mean(temperature, axis=0) if temperature.ndim == 2 else temperature
    if mean_pressure.shape != mean_temperature.shape:
        raise ValueError("pressure_hl_pa and temperature_hl_k must have matching level dimensions")
    if mean_pressure.ndim != 1 or mean_pressure.size < 2:
        raise ValueError("height grid requires at least two half levels")

    pressure_clipped = np.clip(mean_pressure.astype(np.float64), EPS, None)
    temperature_clipped = np.clip(mean_temperature.astype(np.float64), 1.0, None)
    layer_temperature = 0.5 * (temperature_clipped[:-1] + temperature_clipped[1:])
    height = np.zeros_like(pressure_clipped)
    if pressure_clipped[-1] >= pressure_clipped[0]:
        height[-1] = 0.0
        layer_dz_km = (
            DRY_AIR_GAS_CONSTANT_J_KG_K
            * layer_temperature
            / GRAVITY_M_S2
            * np.log(pressure_clipped[1:] / pressure_clipped[:-1])
            / 1000.0
        )
        for idx in range(pressure_clipped.size - 2, -1, -1):
            height[idx] = height[idx + 1] + max(float(layer_dz_km[idx]), 0.0)
    else:
        height[0] = 0.0
        layer_dz_km = (
            DRY_AIR_GAS_CONSTANT_J_KG_K
            * layer_temperature
            / GRAVITY_M_S2
            * np.log(pressure_clipped[:-1] / pressure_clipped[1:])
            / 1000.0
        )
        for idx in range(1, pressure_clipped.size):
            height[idx] = height[idx - 1] + max(float(layer_dz_km[idx - 1]), 0.0)
    return height


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _max_py2sess_rows(options: dict[str, Any]) -> int:
    value = options.get("py2sess_max_rows")
    if value is None:
        value = os.environ.get("_PY2SESS_MAX_ROWS", os.environ.get("PY2SESS_MAX_ROWS", "131072"))
    return int(value)


def _resolve_max_rows(max_rows: int | None) -> int:
    if max_rows is None:
        return _max_py2sess_rows({})
    if int(max_rows) <= 0:
        return 2**62
    return int(max_rows)


def _q_slices(batch_count: int, q_count: int, max_rows: int) -> list[slice]:
    q_per_chunk = max(1, int(max_rows) // max(1, int(batch_count)))
    return [slice(start, min(q_count, start + q_per_chunk)) for start in range(0, q_count, q_per_chunk)]


def _use_gradient_checkpointing(options: dict[str, Any]) -> bool:
    return bool(options.get("gradient_checkpointing", options.get("use_gradient_checkpointing", True)))


def _checkpoint_if_enabled(enabled: bool, function: Any, *args: Any) -> Any:
    if not enabled:
        return function(*args)
    from torch.utils.checkpoint import checkpoint

    return checkpoint(function, *args, use_reentrant=False)


def _flux_rows(value: Any, row_count: int) -> Any:
    rows = value
    if rows.ndim == 2:
        if rows.shape[0] != row_count:
            rows = rows.reshape(row_count, -1)
        return rows
    if rows.ndim == 3:
        if rows.shape[0] == row_count:
            return rows[:, 0, :]
        if rows.shape[1] == row_count:
            return rows[0, :, :]
        return rows.reshape(row_count, -1, rows.shape[-1])[:, 0, :]
    raise ValueError("py2sess forward_flux returned unsupported flux shape")


def _net_flux_rows(result: Any, up_rows: Any, down_rows: Any, row_count: int) -> Any:
    net = getattr(result, "flux_net", None)
    if net is None:
        return up_rows - down_rows
    return _flux_rows(net, row_count)


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
