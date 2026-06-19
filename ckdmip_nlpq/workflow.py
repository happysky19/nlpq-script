"""Single-band workflow stages for the minimal CKDMIP NLPQ runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig, build_output_paths, parse_profile_spec
from .data import (
    CKDMIPNativeBatch,
    build_download_plan,
    download_items,
    flux_path,
    load_native_batch_from_ckdmip,
    load_native_batch_from_npz,
    missing_required_files,
    write_download_plan,
)
from .metrics import build_flux_metrics
from .export import write_ckdmip_netcdf, write_membership_csv
from .model import NLPQModel, NativeBatch
from .plotting import (
    align_flux_profiles,
    heating_rate,
    plot_band_outputs,
    read_model_flux,
    read_truth_flux,
    with_profile_ids,
)
from .rt import (
    CKDMIPRunner,
    check_ckdmip_executable,
    check_py2sess_forward_flux_available,
    ckdmip_executable,
    write_ckdmip_namelist,
    write_command_manifest,
)
from .tuning import expand_candidates, rank_candidates, write_csv, write_selected_settings


def run_stage(config: RunConfig, *, stage: str, dry_run: bool = False) -> None:
    for band in config.bands:
        paths = build_output_paths(config, band)
        if stage in {"all", "preflight"}:
            preflight(config, band=band, dry_run=dry_run)
        if stage in {"all", "download"}:
            download(config, band=band, dry_run=dry_run)
        if stage in {"all", "dev_tune"}:
            dev_tune(config, band=band, dry_run=dry_run)
        if stage in {"all", "final_train"}:
            final_train(config, band=band, dry_run=dry_run)
        if stage in {"all", "final_test"}:
            final_test(config, band=band, dry_run=dry_run)
        if stage in {"all", "plot"}:
            if dry_run:
                write_plot_dry_run(paths.plot_path())
            else:
                plot_band_outputs(config, band=band)
        if stage in {"all", "report"}:
            write_report(config, band=band)


def preflight(config: RunConfig, *, band: int, dry_run: bool = False) -> None:
    paths = build_output_paths(config, band)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        check_ckdmip_executable(config, config.domain)
    except Exception as exc:
        errors.append(str(exc))
    if "rt-aware" in config.methods:
        try:
            check_py2sess_forward_flux_available(config.py2sess_repo)
        except Exception as exc:
            errors.append(str(exc))
    try:
        primary_scenario(config)
    except Exception as exc:
        errors.append(str(exc))
    if not dry_run:
        missing = missing_required_files(config, band, stage="preflight")
        if missing:
            errors.append("missing CKDMIP files: " + ", ".join(str(v) for v in missing[:8]))
    manifest = {
        "stage": "preflight",
        "domain": config.domain,
        "band": int(band),
        "run_id": config.run_id,
        "dry_run": dry_run,
        "errors": errors,
        "required_identity_check": "Q=M identity export and CKDMIP RT smoke must pass before formal evaluation",
    }
    paths.manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if errors and not dry_run:
        raise RuntimeError("; ".join(errors))


def download(config: RunConfig, *, band: int, dry_run: bool = False) -> None:
    paths = build_output_paths(config, band)
    items = build_download_plan(config, band)
    plan_path = paths.run_dir / "download_plan.csv"
    write_download_plan(plan_path, items)
    if not dry_run:
        download_items(items)
    print(f"wrote {plan_path}")


def dev_tune(config: RunConfig, *, band: int, dry_run: bool = False) -> None:
    paths = build_output_paths(config, band)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        rows = [candidate.as_dict() | {"status": "DRY_RUN"} for candidate in expand_candidates(config.raw)]
        write_csv(paths.metric_path("dev_tuning_candidates"), rows)
        ranked = rank_candidates([row | {"tau_rmse": 0.0, "runtime_ms_per_profile": 0.0} for row in rows])
        write_csv(paths.metric_path("dev_tuning_ranked"), ranked)
        write_selected_settings(paths.run_dir / "selected_settings.yaml", paths.run_dir / "selected_settings.json", ranked)
        return
    ensure_training_supported(config, method=None)

    train = load_batch_for_split(config, band=band, split="dev_train")
    val = load_batch_for_split(config, band=band, split="dev_val")
    rows: list[dict[str, Any]] = []
    for candidate in expand_candidates(config.raw):
        started = time.perf_counter()
        train_options = training_options_for_candidate(config, candidate.as_dict())
        print(
            f"training {config.domain} band {band:02d} candidate {candidate.candidate_id}: "
            f"{candidate.method} Q={candidate.q_value} steps={train_options['steps']}"
        )
        model = NLPQModel(
            domain=config.domain,
            band=band,
            method=candidate.method,
            q_value=candidate.q_value,
            seed=int(config.raw["nlpq"].get("seed", 0)) + candidate.candidate_id,
            metadata={"candidate_id": candidate.candidate_id, "phase": "dev"},
        )
        model.fit(to_model_batch(train), training_config=train_options, py2sess_repo=config.py2sess_repo).freeze()
        compressed_val = model.apply(to_model_batch(val))
        tau_rmse = float(np.sqrt(np.mean(np.square(reexpand_tau(compressed_val.tau_q, compressed_val.cluster_id) - val.tau_native))))
        model_path = paths.model_path(candidate.method, candidate.q_value, "dev")
        model.save(model_path)
        candidate_input = paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{candidate.method}_q{candidate.q_value}_dev.nc"
        candidate_config = paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{candidate.method}_q{candidate.q_value}_dev.nam"
        candidate_flux = paths.run_dir / "ckdmip_fluxes" / f"{paths.prefix}_{candidate.method}_q{candidate.q_value}_dev_fluxes.nc"
        write_ckdmip_netcdf(
            candidate_input,
            to_model_batch(val),
            compressed_val,
            domain=config.domain,
            band=band,
            rt_options=config.raw.get("rt", {}),
        )
        write_ckdmip_namelist(candidate_config, domain=config.domain, raw_config=config.raw)
        rt_result = CKDMIPRunner(ckdmip_executable(config, config.domain), domain=config.domain).run(
            input_file=candidate_input,
            output_file=candidate_flux,
            config_file=candidate_config,
            scenario=primary_scenario(config),
            dry_run=False,
        )
        truth_path = flux_path(config, config.domain, "evaluation1", primary_scenario(config))
        metric_fields = (
            build_flux_metrics(
                domain=config.domain,
                band=band,
                model_flux_path=candidate_flux,
                truth_flux_path=truth_path,
                profile_ids=val.profile_ids,
            )
            if rt_result.status == "OK"
            else {"metric_status": "ckdmip_rt_failed", "truth_flux": str(truth_path)}
        )
        if bool(config.raw.get("tuning", {}).get("require_ckdmip_rt_for_selection", True)):
            if rt_result.status != "OK" or metric_fields.get("metric_status") != "compared":
                raise RuntimeError(
                    "dev tuning requires successful CKDMIP RT comparison; "
                    f"candidate {candidate.candidate_id} status={rt_result.status} metrics={metric_fields}"
                )
        rows.append(
            candidate.as_dict()
            | {
                "domain": config.domain,
                "band": int(band),
                "model_path": str(model_path),
                "tau_rmse": tau_rmse,
                "runtime_ms_per_profile": 1000.0 * (time.perf_counter() - started) / max(1, val.profile_ids.size),
                "ckdmip_input": str(candidate_input),
                "ckdmip_config": str(candidate_config),
                "ckdmip_flux": str(candidate_flux),
                "ckdmip_rt_status": rt_result.status,
                "ckdmip_returncode": "" if rt_result.returncode is None else int(rt_result.returncode),
                **training_summary_fields(model.metadata),
                **metric_fields,
            }
        )
    write_csv(paths.metric_path("dev_tuning_candidates"), rows)
    ranked = rank_candidates(rows, config.raw.get("tuning", {}).get("score_weights"))
    write_csv(paths.metric_path("dev_tuning_ranked"), ranked)
    write_selected_settings(paths.run_dir / "selected_settings.yaml", paths.run_dir / "selected_settings.json", ranked)
    selected = ranked[0]
    selected_flux = Path(str(selected.get("ckdmip_flux", "")))
    if selected_flux.exists():
        write_vertical_outputs(
            paths.vertical_path("val"),
            val,
            domain=config.domain,
            band=band,
            model_flux_path=selected_flux,
            truth_flux_path=truth_path,
        )


def final_train(config: RunConfig, *, band: int, dry_run: bool = False) -> None:
    paths = build_output_paths(config, band)
    selected = load_selected(paths.run_dir / "selected_settings.json")
    ensure_training_supported(config, method=str(selected["method"]))
    if dry_run:
        write_csv(
            paths.metric_path("final_train_log"),
            [
                {
                    "domain": config.domain,
                    "band": int(band),
                    "status": "DRY_RUN",
                    "method": selected["method"],
                    "q_value": selected["q_value"],
                }
            ],
        )
        return
    train = load_batch_for_split(config, band=band, split="final_train")
    model = NLPQModel(
        domain=config.domain,
        band=band,
        method=str(selected["method"]),
        q_value=int(selected["q_value"]),
        seed=int(config.raw["nlpq"].get("seed", 0)),
        metadata={"phase": "final", "selected_candidate": selected},
    )
    train_options = training_options_for_candidate(config, selected)
    print(
        f"training final {config.domain} band {band:02d}: "
        f"{selected['method']} Q={selected['q_value']} steps={train_options['steps']}"
    )
    model.fit(to_model_batch(train), training_config=train_options, py2sess_repo=config.py2sess_repo).freeze()
    path = paths.model_path(str(selected["method"]), int(selected["q_value"]), "final")
    model.save(path)
    compressed = model.apply(to_model_batch(train))
    train_input = paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{selected['method']}_q{selected['q_value']}_train.nc"
    train_config = paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{selected['method']}_q{selected['q_value']}_train.nam"
    train_flux = paths.run_dir / "ckdmip_fluxes" / f"{paths.prefix}_{selected['method']}_q{selected['q_value']}_train_fluxes.nc"
    write_ckdmip_netcdf(
        train_input,
        to_model_batch(train),
        compressed,
        domain=config.domain,
        band=band,
        rt_options=config.raw.get("rt", {}),
    )
    write_ckdmip_namelist(train_config, domain=config.domain, raw_config=config.raw)
    train_result = CKDMIPRunner(ckdmip_executable(config, config.domain), domain=config.domain).run(
        input_file=train_input,
        output_file=train_flux,
        config_file=train_config,
        scenario=primary_scenario(config),
        dry_run=False,
    )
    write_command_manifest(paths.run_dir / "ckdmip_train_commands.json", [train_result])
    train_truth = flux_path(config, config.domain, "evaluation1", primary_scenario(config))
    train_metric_fields = (
        build_flux_metrics(
            domain=config.domain,
            band=band,
            model_flux_path=train_flux,
            truth_flux_path=train_truth,
            profile_ids=train.profile_ids,
        )
        if train_result.status == "OK"
        else {"metric_status": "ckdmip_rt_failed", "truth_flux": str(train_truth)}
    )
    if train_result.status == "OK":
        write_vertical_outputs(
            paths.vertical_path("train"),
            train,
            domain=config.domain,
            band=band,
            model_flux_path=train_flux,
            truth_flux_path=train_truth,
        )
    write_csv(
        paths.metric_path("final_train_log"),
        [
            {
                "domain": config.domain,
                "band": int(band),
                "status": train_result.status,
                "method": selected["method"],
                "q_value": selected["q_value"],
                "model_path": str(path),
                "ckdmip_input": str(train_input),
                "ckdmip_config": str(train_config),
                "ckdmip_flux": str(train_flux),
                "train_profile_count": int(train.profile_ids.size),
                **training_summary_fields(model.metadata),
                **train_metric_fields,
            }
        ],
    )
    if train_result.status != "OK":
        raise RuntimeError(f"final-train CKDMIP RT failed for {config.domain} band {band:02d}")


def final_test(config: RunConfig, *, band: int, dry_run: bool = False) -> None:
    paths = build_output_paths(config, band)
    selected = load_selected(paths.run_dir / "selected_settings.json")
    model_path = paths.model_path(str(selected["method"]), int(selected["q_value"]), "final")
    input_path = paths.ckdmip_input_path(str(selected["method"]), int(selected["q_value"]))
    flux_path = paths.ckdmip_flux_path(str(selected["method"]), int(selected["q_value"]))
    config_path = paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{selected['method']}_q{selected['q_value']}.nam"
    scenario = primary_scenario(config)
    if not model_path.exists() and not dry_run:
        raise FileNotFoundError(model_path)
    if dry_run:
        result = CKDMIPRunner(ckdmip_executable(config, config.domain), domain=config.domain).run(
            input_file=input_path,
            output_file=flux_path,
            config_file=config_path,
            scenario=scenario,
            dry_run=True,
        )
        write_command_manifest(paths.run_dir / "ckdmip_commands.json", [result])
        write_csv(
            paths.metric_path("final_test_metrics"),
            [
                {
                    "domain": config.domain,
                    "band": int(band),
                    "status": "DRY_RUN",
                    "method": selected["method"],
                    "q_value": selected["q_value"],
                    "model_path": str(model_path),
                    "ckdmip_input": str(input_path),
                    "ckdmip_config": str(config_path),
                    "ckdmip_flux": str(flux_path),
                    "scenario": scenario,
                }
            ],
        )
        return
    test = load_batch_for_split(config, band=band, split="final_test")
    model = NLPQModel.load(model_path)
    model.assert_compatible(domain=config.domain, band=band, method=str(selected["method"]), q_value=int(selected["q_value"]))
    compressed = model.apply(to_model_batch(test))
    write_ckdmip_netcdf(
        input_path,
        to_model_batch(test),
        compressed,
        domain=config.domain,
        band=band,
        rt_options=config.raw.get("rt", {}),
    )
    write_ckdmip_namelist(config_path, domain=config.domain, raw_config=config.raw)
    write_membership_csv(paths.run_dir / "ckdmip_inputs" / f"{paths.prefix}_{selected['method']}_q{selected['q_value']}_membership.csv", compressed.cluster_id)
    runner = CKDMIPRunner(ckdmip_executable(config, config.domain), domain=config.domain)
    result = runner.run(
        input_file=input_path,
        output_file=flux_path,
        config_file=config_path,
        scenario=scenario,
        dry_run=False,
    )
    write_command_manifest(paths.run_dir / "ckdmip_commands.json", [result])
    truth_path = flux_path_for_final(config, scenario=scenario)
    metric_fields = build_flux_metrics(
        domain=config.domain,
        band=band,
        model_flux_path=flux_path,
        truth_flux_path=truth_path,
        profile_ids=test.profile_ids,
    ) if result.status == "OK" else {"metric_status": "ckdmip_rt_failed", "truth_flux": str(truth_path)}
    if result.status == "OK":
        write_vertical_outputs(
            paths.vertical_path("test"),
            test,
            domain=config.domain,
            band=band,
            model_flux_path=flux_path,
            truth_flux_path=truth_path,
        )
    write_csv(
        paths.metric_path("final_test_metrics"),
        [
            {
                "domain": config.domain,
                "band": int(band),
                "status": result.status,
                "method": selected["method"],
                "q_value": selected["q_value"],
                "model_path": str(model_path),
                "ckdmip_input": str(input_path),
                "ckdmip_config": str(config_path),
                "ckdmip_flux": str(flux_path),
                "scenario": scenario,
                "returncode": "" if result.returncode is None else int(result.returncode),
                **metric_fields,
            }
        ],
    )


def load_batch_for_split(config: RunConfig, *, band: int, split: str) -> CKDMIPNativeBatch:
    batch_npz = config.raw.get("training", {}).get("batch_npz")
    if batch_npz:
        full = load_native_batch_from_npz(batch_npz)
        if split == "dev_train":
            profile_spec = str(config.raw["split"]["dev"]["train_profiles"])
        elif split == "dev_val":
            profile_spec = str(config.raw["split"]["dev"]["val_profiles"])
        elif split == "final_train":
            profile_spec = str(config.raw["split"]["final"]["train_profiles"])
        elif split == "final_test":
            profile_spec = str(config.raw["split"]["final"].get("test_profiles", "0-49"))
        else:
            raise ValueError(f"unknown split: {split}")
        return subset_batch(full, parse_profile_spec(profile_spec))
    if split == "dev_train":
        profile_spec = str(config.raw["split"]["dev"]["train_profiles"])
        dataset = "evaluation1"
    elif split == "dev_val":
        profile_spec = str(config.raw["split"]["dev"]["val_profiles"])
        dataset = "evaluation1"
    elif split == "final_train":
        profile_spec = str(config.raw["split"]["final"]["train_profiles"])
        dataset = "evaluation1"
    elif split == "final_test":
        profile_spec = str(config.raw["split"]["final"].get("test_profiles", "0-49"))
        dataset = str(config.raw["split"]["final"]["test_dataset"])
    else:
        raise ValueError(f"unknown split: {split}")
    return load_native_batch_from_ckdmip(
        config,
        band=band,
        dataset=dataset,
        profile_spec=profile_spec,
        scenario=primary_scenario(config),
    )


def subset_batch(batch: CKDMIPNativeBatch, profiles: list[int]) -> CKDMIPNativeBatch:
    available = set(int(v) for v in batch.profile_ids.tolist())
    missing = [profile for profile in profiles if profile not in available]
    if missing:
        raise ValueError(f"selected profiles are not present in batch: {missing}")
    positions = [int(np.where(batch.profile_ids == profile)[0][0]) for profile in profiles]
    return CKDMIPNativeBatch(
        profile_ids=batch.profile_ids[positions],
        pressure_hl=batch.pressure_hl[positions],
        temperature_hl=batch.temperature_hl[positions],
        wavenumber=batch.wavenumber,
        spectral_weight=batch.spectral_weight,
        tau_native=batch.tau_native[positions],
        rayleigh_tau_native=None if batch.rayleigh_tau_native is None else batch.rayleigh_tau_native[positions],
        incoming_flux_native=batch.incoming_flux_native,
    )


def to_model_batch(batch: CKDMIPNativeBatch) -> NativeBatch:
    return NativeBatch(
        profile_ids=batch.profile_ids,
        pressure_hl=batch.pressure_hl,
        temperature_hl=batch.temperature_hl,
        wavenumber=batch.wavenumber,
        spectral_weight=batch.spectral_weight,
        tau_native=batch.tau_native,
        rayleigh_tau_native=batch.rayleigh_tau_native,
        incoming_flux_native=batch.incoming_flux_native,
    )


def reexpand_tau(tau_q: np.ndarray, cluster_id: np.ndarray) -> np.ndarray:
    return tau_q[..., cluster_id]


def load_selected(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"selected settings not found: {path}")
    payload = json.loads(path.read_text())
    return dict(payload["selected"])


def ensure_training_supported(config: RunConfig, *, method: str | None) -> None:
    methods = [method] if method is not None else config.methods
    if config.domain == "sw" and "rt-aware" in methods:
        raise NotImplementedError("rt-aware training is currently implemented for longwave only")


def training_options_for_candidate(config: RunConfig, selected: dict[str, Any]) -> dict[str, Any]:
    options = dict(config.raw.get("training", {}))
    rt_options = config.raw.get("rt", {})
    nlpq_options = config.raw.get("nlpq", {})
    options["rt_train_teacher"] = str(rt_options.get("train_teacher", options.get("train_teacher", "py2sess")))
    options["lr"] = float(selected.get("lr", options.get("lr", 0.05)))
    options["steps"] = int(selected.get("steps", options.get("steps", 220)))
    options["train_pressure_min_hpa"] = float(
        selected.get("train_pressure_min_hpa", nlpq_options.get("train_pressure_min_hpa", 0.001))
    )
    options["train_pressure_max_hpa"] = float(nlpq_options.get("train_pressure_max_hpa", 1100.0))
    options["streams"] = int(rt_options.get("lw_streams", options.get("streams", 4)))
    return options


def training_summary_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    log = metadata.get("training_log", {})
    if not isinstance(log, dict):
        return {}
    fields: dict[str, Any] = {}
    for key in (
        "rt_aware_training",
        "assignment_training",
        "teacher_requested",
        "teacher_kernel",
        "py2sess_version",
        "steps",
        "lr",
        "streams",
        "teacher_loss_final",
        "teacher_flux_loss_final",
        "teacher_heating_loss_final",
        "train_pressure_min_hpa",
        "train_pressure_max_hpa",
    ):
        if key in log:
            fields[key] = log[key]
    return fields


def primary_scenario(config: RunConfig) -> str:
    scenarios = config.raw.get("run", {}).get("scenarios", ["present"])
    if not scenarios:
        return "present"
    if len(scenarios) != 1:
        raise NotImplementedError("run one scenario per YAML config; per-scenario looping is not implemented")
    return str(scenarios[0])


def flux_path_for_final(config: RunConfig, *, scenario: str) -> Path:
    return flux_path(config, config.domain, str(config.raw["split"]["final"]["test_dataset"]), scenario)


def write_vertical_outputs(
    path: Path,
    batch: CKDMIPNativeBatch,
    *,
    domain: str,
    band: int,
    model_flux_path: Path | None = None,
    truth_flux_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "profile_ids": batch.profile_ids,
        "pressure_hl": batch.pressure_hl,
        "temperature_hl": batch.temperature_hl,
    }
    model_flux: dict[str, np.ndarray] | None = None
    truth_flux: dict[str, np.ndarray] | None = None
    if model_flux_path is not None and model_flux_path.exists():
        model_flux = with_profile_ids(read_model_flux(model_flux_path, domain), batch.profile_ids)
        payload.update(
            {
                "model_flux_up": model_flux["up"],
                "model_flux_down": model_flux["down"],
                "model_heating_rate": heating_rate(model_flux["up"], model_flux["down"], model_flux["pressure_hl"]),
                "model_flux_path": np.asarray([str(model_flux_path)]),
            }
        )
    if truth_flux_path is not None and truth_flux_path.exists():
        truth_flux = read_truth_flux(truth_flux_path, domain, band)
        if model_flux is None:
            reference_flux = {
                "profile_ids": batch.profile_ids,
                "pressure_hl": batch.pressure_hl,
                "up": np.zeros_like(batch.pressure_hl, dtype=np.float64),
                "down": np.zeros_like(batch.pressure_hl, dtype=np.float64),
            }
            truth_flux = align_flux_profiles(
                reference_flux,
                truth_flux,
            )[1]
        else:
            model_flux, truth_flux = align_flux_profiles(model_flux, truth_flux)
        payload.update(
            {
                "truth_flux_up": truth_flux["up"],
                "truth_flux_down": truth_flux["down"],
                "truth_heating_rate": heating_rate(truth_flux["up"], truth_flux["down"], truth_flux["pressure_hl"]),
                "truth_flux_path": np.asarray([str(truth_flux_path)]),
            }
        )
    if model_flux is not None and truth_flux is not None:
        model_heat = heating_rate(model_flux["up"], model_flux["down"], model_flux["pressure_hl"])
        truth_heat = heating_rate(truth_flux["up"], truth_flux["down"], truth_flux["pressure_hl"])
        payload.update(
            {
                "flux_up_error": model_flux["up"] - truth_flux["up"],
                "flux_down_error": model_flux["down"] - truth_flux["down"],
                "heating_rate_error": model_heat - truth_heat,
            }
        )
    np.savez_compressed(path, **payload)


def write_plot_dry_run(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".txt").write_text("DRY_RUN: plot stage requires CKDMIP flux outputs\n")


def write_report(config: RunConfig, *, band: int) -> None:
    paths = build_output_paths(config, band)
    selected_path = paths.run_dir / "selected_settings.json"
    selected_text = selected_path.read_text() if selected_path.exists() else "{}"
    report = (
        f"# CKDMIP NLPQ Report: {config.domain} band {band:02d}\n\n"
        f"- Run id: `{config.run_id}`\n"
        f"- Domain: `{config.domain}`\n"
        f"- Band: `{band}`\n"
        f"- Selected settings: `{selected_path}`\n"
        f"- Final metrics: `{paths.metric_path('final_test_metrics')}`\n\n"
        "## Selected Settings\n\n"
        f"```json\n{selected_text}\n```\n\n"
        "## Notes\n\n"
        "Evaluation-2 is reserved for frozen final testing only. Dev-val metrics are used for selection.\n"
    )
    path = paths.report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
