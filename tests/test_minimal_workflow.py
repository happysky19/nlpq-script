from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml
from netCDF4 import Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ckdmip_nlpq.config import build_output_paths, load_run_config, parse_profile_spec  # noqa: E402
from ckdmip_nlpq.data import build_download_plan, species_scale_for_scenario, write_download_plan  # noqa: E402
from ckdmip_nlpq.metrics import build_flux_metrics  # noqa: E402
from ckdmip_nlpq.model import NLPQModel, NativeBatch  # noqa: E402
from ckdmip_nlpq.tuning import rank_candidates, write_selected_settings  # noqa: E402
from ckdmip_nlpq.workflow import run_stage, write_vertical_outputs  # noqa: E402


class MinimalWorkflowTest(unittest.TestCase):
    def test_profile_parser_and_leakage_validation(self) -> None:
        self.assertEqual(parse_profile_spec("0-2,4"), [0, 1, 2, 4])
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = write_config(Path(tmpdir), train="0-2", val="2-3")
            with self.assertRaisesRegex(ValueError, "profile leakage"):
                load_run_config(cfg_path)

    def test_path_builder_includes_domain_and_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_run_config(write_config(Path(tmpdir)))
            paths = build_output_paths(cfg, 2)
            self.assertIn("sw/band02/pilot", str(paths.run_dir))
            self.assertEqual(paths.model_path("det", 6, "final").name, "sw_band02_det_q6_final.npz")
            self.assertEqual(paths.metric_path("dev_tuning_ranked").name, "sw_band02_dev_tuning_ranked.csv")

    def test_download_dry_run_plan_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = load_run_config(write_config(root))
            items = build_download_plan(cfg, 2)
            self.assertTrue(items)
            self.assertTrue(all("band02" not in item.destination.name for item in items))
            plan = root / "plan.csv"
            write_download_plan(plan, items)
            text = plan.read_text()
            self.assertIn("ckdmip_evaluation1_sw_fluxes_present.h5", text)
            self.assertIn("ckdmip_evaluation1_sw_spectra_h2o_present_1-10.h5", text)
            self.assertIn("ckdmip_ssi.h5", text)

    def test_standard_scenario_scaling(self) -> None:
        self.assertAlmostEqual(species_scale_for_scenario("co2", "future"), 1120.0 / 415.0)
        self.assertAlmostEqual(species_scale_for_scenario("ch4", "preindustrial"), 700.0 / 1921.0)
        self.assertEqual(species_scale_for_scenario("n2", "n2-0"), 0.0)
        self.assertEqual(species_scale_for_scenario("h2o", "future"), 1.0)

    def test_model_identity_and_freeze(self) -> None:
        batch = native_batch(profile_count=2, spectral_count=5)
        model = NLPQModel(domain="sw", band=2, method="det", q_value=5)
        model.fit(batch).freeze()
        compressed = model.apply(batch)
        np.testing.assert_allclose(compressed.tau_q, batch.tau_native)
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            model.fit(batch)

    def test_lw_rt_aware_model_trains_and_freezes(self) -> None:
        batch = native_batch(profile_count=3, spectral_count=6)
        model = NLPQModel(domain="lw", band=4, method="rt-aware", q_value=3, seed=2)
        with tempfile.TemporaryDirectory() as repo_dir:
            repo = write_fake_py2sess_repo(Path(repo_dir))
            model.fit(
                batch,
                training_config={
                    "rt_train_teacher": "py2sess",
                    "steps": 2,
                    "lr": 0.02,
                    "dtype": "float32",
                    "device": "cpu",
                    "streams": 2,
                    "train_pressure_min_hpa": 0.001,
                    "train_pressure_max_hpa": 1100.0,
                },
                py2sess_repo=repo,
            ).freeze()
        compressed = model.apply(batch)
        self.assertEqual(compressed.tau_q.shape, (3, 3, 3))
        self.assertTrue(np.all(compressed.tau_q >= 0.0))
        self.assertEqual(set(compressed.cluster_id.tolist()), {0, 1, 2})
        training_log = model.metadata["training_log"]
        self.assertEqual(training_log["rt_aware_training"], "optimized")
        self.assertEqual(training_log["teacher_kernel"], "py2sess_forward_flux")
        self.assertIn("teacher_loss_final", training_log)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lw_rt_aware.npz"
            model.save(path)
            loaded = NLPQModel.load(path)
            self.assertEqual(loaded.metadata["training_log"]["rt_aware_training"], "optimized")

    def test_lw_rt_aware_requires_py2sess_forward_flux(self) -> None:
        batch = native_batch(profile_count=2, spectral_count=4)
        model = NLPQModel(domain="lw", band=4, method="rt-aware", q_value=2, seed=4)
        with tempfile.TemporaryDirectory() as repo_dir:
            repo = write_fake_py2sess_repo(Path(repo_dir), include_forward_flux=False)
            with self.assertRaisesRegex(ImportError, "forward_flux"):
                model.fit(
                    batch,
                    training_config={
                        "rt_train_teacher": "py2sess",
                        "steps": 1,
                        "lr": 0.02,
                        "dtype": "float32",
                        "device": "cpu",
                    },
                    py2sess_repo=repo,
                )

    def test_sw_rt_aware_model_trains_and_exports_source_terms(self) -> None:
        batch = sw_native_batch(profile_count=3, spectral_count=6)
        model = NLPQModel(domain="sw", band=2, method="rt-aware", q_value=3, seed=5)
        with tempfile.TemporaryDirectory() as repo_dir:
            repo = write_fake_py2sess_repo(Path(repo_dir))
            model.fit(
                batch,
                training_config={
                    "rt_train_teacher": "py2sess",
                    "steps": 2,
                    "lr": 0.02,
                    "dtype": "float32",
                    "device": "cpu",
                    "streams": 2,
                    "mu_values": [0.5],
                    "surf_albedo": 0.1,
                    "include_fo": True,
                    "train_pressure_min_hpa": 0.001,
                    "train_pressure_max_hpa": 1100.0,
                },
                py2sess_repo=repo,
            ).freeze()
        compressed = model.apply(batch)
        self.assertEqual(compressed.tau_q.shape, (3, 3, 3))
        self.assertEqual(compressed.rayleigh_tau_q.shape, (3, 3, 3))
        self.assertEqual(compressed.incoming_flux_q.shape, (3,))
        self.assertTrue(np.all(compressed.tau_q >= 0.0))
        self.assertTrue(np.all(compressed.rayleigh_tau_q >= 0.0))
        self.assertTrue(np.all(compressed.incoming_flux_q >= 0.0))
        training_log = model.metadata["training_log"]
        self.assertEqual(training_log["rt_aware_training"], "optimized")
        self.assertEqual(training_log["teacher_kernel"], "py2sess_forward_flux_sw")
        self.assertEqual(training_log["mu_values"], [0.5])

    def test_sw_rt_aware_requires_source_terms(self) -> None:
        batch = native_batch(profile_count=2, spectral_count=4)
        model = NLPQModel(domain="sw", band=2, method="rt-aware", q_value=2, seed=4)
        with tempfile.TemporaryDirectory() as repo_dir:
            repo = write_fake_py2sess_repo(Path(repo_dir))
            with self.assertRaisesRegex(ValueError, "incoming_flux_native"):
                model.fit(
                    batch,
                    training_config={
                        "rt_train_teacher": "py2sess",
                        "steps": 1,
                        "lr": 0.02,
                        "dtype": "float32",
                        "device": "cpu",
                    },
                    py2sess_repo=repo,
                )

    def test_tuner_ranking_and_selected_settings(self) -> None:
        rows = [
            {"candidate_id": 0, "method": "det", "q_value": 9, "tau_rmse": 0.2, "runtime_ms_per_profile": 1.0},
            {"candidate_id": 1, "method": "det", "q_value": 6, "tau_rmse": 0.1, "runtime_ms_per_profile": 2.0},
            {"candidate_id": 2, "method": "det", "q_value": 3, "tau_rmse": 0.1, "runtime_ms_per_profile": 1.0},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["candidate_id"], 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = write_selected_settings(Path(tmpdir) / "selected.yaml", Path(tmpdir) / "selected.json", ranked)
            self.assertEqual(payload["selected"]["candidate_id"], 2)

    def test_dry_run_workflow_writes_ranked_settings_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cfg = load_run_config(write_config(root, methods=["det"]))
            run_stage(cfg, stage="dev_tune", dry_run=True)
            run_stage(cfg, stage="final_train", dry_run=True)
            run_stage(cfg, stage="final_test", dry_run=True)
            run_stage(cfg, stage="report", dry_run=True)
            paths = build_output_paths(cfg, 2)
            self.assertTrue((paths.run_dir / "selected_settings.json").exists())
            self.assertTrue(paths.report_path().exists())
            commands = json.loads((paths.run_dir / "ckdmip_commands.json").read_text())
            self.assertIn("--ckd", commands[0]["command"])
            self.assertIn("--scenario", commands[0]["command"])
            report = paths.report_path().read_text()
            self.assertIn("Evaluation-2", report)

    def test_metrics_and_vertical_outputs_align_profile_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_flux = root / "model_flux.nc"
            truth_flux = root / "truth_flux.h5"
            profile_ids = np.asarray([40, 41], dtype=np.int64)
            pressure = np.asarray([[1.0, 500.0, 1000.0], [1.0, 500.0, 1000.0]], dtype=np.float64)
            up = np.asarray([[1.0, 1.2, 1.4], [2.0, 2.2, 2.4]], dtype=np.float64)
            down = np.asarray([[6.0, 5.4, 4.8], [7.0, 6.4, 5.8]], dtype=np.float64)
            write_model_sw_flux(model_flux, profile_ids, pressure, up, down)
            write_truth_sw_flux(truth_flux, profile_ids, pressure, up, down, band=2)

            metrics = build_flux_metrics(
                domain="sw",
                band=2,
                model_flux_path=model_flux,
                truth_flux_path=truth_flux,
                profile_ids=profile_ids,
            )
            self.assertEqual(metrics["metric_status"], "compared")
            self.assertAlmostEqual(metrics["toa_up_rmse"], 0.0)

            batch = NativeBatch(
                profile_ids=profile_ids,
                pressure_hl=pressure,
                temperature_hl=np.full_like(pressure, 280.0),
                wavenumber=np.linspace(2600.0, 3249.0, 4),
                spectral_weight=np.full(4, 0.25),
                tau_native=np.ones((2, 2, 4)),
            )
            vertical_path = root / "vertical.npz"
            write_vertical_outputs(
                vertical_path,
                batch,
                domain="sw",
                band=2,
                model_flux_path=model_flux,
                truth_flux_path=truth_flux,
            )
            with np.load(vertical_path, allow_pickle=False) as data:
                self.assertIn("model_heating_rate", data.files)
                self.assertIn("truth_heating_rate", data.files)
                self.assertEqual(data["model_heating_rate"].shape, (2, 2))
                np.testing.assert_allclose(data["flux_up_error"], 0.0)


def write_config(root: Path, *, train: str = "0-39", val: str = "40-49", methods: list[str] | None = None) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for exe in ("ckdmip_sw", "ckdmip_lw"):
        path = bin_dir / exe
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    payload = {
        "paths": {
            "data_root": str(root / "data"),
            "run_root": str(root / "runs"),
            "ckdmip_bin": str(bin_dir),
            "py2sess_repo": "",
        },
        "run": {
            "domain": "sw",
            "bands": [2],
            "run_id": "pilot",
            "datasets": ["evaluation1"],
            "scenarios": ["present"],
            "profile_blocks": ["1-10"],
            "species": [{"name": "h2o", "tag": "present"}],
        },
        "split": {
            "dev": {"train_profiles": train, "val_profiles": val},
            "final": {"train_profiles": "0-49", "test_dataset": "evaluation2", "test_profiles": "0-49"},
        },
        "nlpq": {
            "methods": methods or ["det"],
            "q_values": [3, 6],
            "train_pressure_min_hpa": 0.001,
            "train_pressure_max_hpa": 1100.0,
            "seed": 1,
        },
        "training": {"steps": 2, "lr": 0.05, "dtype": "float32", "device": "cpu"},
        "tuning": {"datasets": ["evaluation1"], "grid": {"lr": [0.05], "steps": [2]}},
        "rt": {"train_teacher": "py2sess", "final_solver": "ckdmip", "mu_values": [0.5]},
    }
    batch = native_batch(profile_count=50, spectral_count=8)
    batch_path = root / "batch.npz"
    np.savez_compressed(
        batch_path,
        profile_ids=batch.profile_ids,
        pressure_hl=batch.pressure_hl,
        temperature_hl=batch.temperature_hl,
        wavenumber=batch.wavenumber,
        spectral_weight=batch.spectral_weight,
        tau_native=batch.tau_native,
    )
    payload["training"]["batch_npz"] = str(batch_path)
    cfg = root / "config.yaml"
    cfg.write_text(yaml.safe_dump(payload, sort_keys=False))
    return cfg


def native_batch(*, profile_count: int, spectral_count: int) -> NativeBatch:
    profile_ids = np.arange(profile_count, dtype=np.int64)
    pressure_hl = np.tile(np.linspace(1.0, 1000.0, 4), (profile_count, 1))
    temperature_hl = np.tile(np.linspace(220.0, 290.0, 4), (profile_count, 1))
    wavenumber = np.linspace(2600.0, 3249.0, spectral_count)
    spectral_weight = np.full(spectral_count, 1.0 / spectral_count)
    profile = profile_ids[:, None, None]
    layer = np.arange(3)[None, :, None]
    spectral = np.arange(spectral_count)[None, None, :]
    tau_native = 0.01 + 0.001 * profile + 0.002 * layer + 0.003 * spectral
    return NativeBatch(
        profile_ids=profile_ids,
        pressure_hl=pressure_hl,
        temperature_hl=temperature_hl,
        wavenumber=wavenumber,
        spectral_weight=spectral_weight,
        tau_native=tau_native,
    )


def sw_native_batch(*, profile_count: int, spectral_count: int) -> NativeBatch:
    batch = native_batch(profile_count=profile_count, spectral_count=spectral_count)
    rayleigh = 0.002 + 0.0002 * np.arange(spectral_count)[None, None, :]
    rayleigh = np.broadcast_to(rayleigh, batch.tau_native.shape)
    incoming = 0.5 + 0.1 * np.arange(spectral_count, dtype=np.float64)
    return NativeBatch(
        profile_ids=batch.profile_ids,
        pressure_hl=batch.pressure_hl,
        temperature_hl=batch.temperature_hl,
        wavenumber=batch.wavenumber,
        spectral_weight=batch.spectral_weight,
        tau_native=batch.tau_native,
        rayleigh_tau_native=rayleigh,
        incoming_flux_native=incoming,
    )


def write_fake_py2sess_repo(root: Path, *, include_forward_flux: bool = True) -> Path:
    package = root / "src" / "py2sess"
    package.mkdir(parents=True)
    forward_flux = ""
    if include_forward_flux:
        forward_flux = '''
    def forward_flux(self, *, tau, ssa, g, z=None, angles=None, stream=None, fbeam=1.0,
                     planck=None, surface_planck=0.0, emissivity=1.0, albedo=0.0,
                     include_fo=False, return_net=False, **_kwargs):
        import torch
        class Result:
            pass
        result = Result()
        if planck is None:
            trans = torch.exp(-torch.clamp(tau * (1.0 - 0.5 * torch.clamp(ssa, 0.0, 1.0)), min=0.0, max=80.0))
            down = torch.as_tensor(fbeam, dtype=tau.dtype, device=tau.device)
            if down.ndim == 0:
                down = down.expand(tau.shape[0])
            down_levels = [down]
            for layer in range(tau.shape[1]):
                down = down * trans[:, layer]
                down_levels.append(down)
            surf_albedo = torch.as_tensor(albedo, dtype=tau.dtype, device=tau.device)
            if surf_albedo.ndim == 0:
                surf_albedo = surf_albedo.expand(tau.shape[0])
            up = surf_albedo * down
            up_levels = [up]
            for layer in range(tau.shape[1] - 1, -1, -1):
                up = up * trans[:, layer]
                up_levels.append(up)
            result.flux_up = torch.stack(list(reversed(up_levels)), dim=1)
            result.flux_down = torch.stack(down_levels, dim=1)
            result.flux_net = result.flux_up - result.flux_down if return_net else None
            return result
        trans = torch.exp(-torch.clamp(tau, min=0.0, max=80.0))
        down = torch.zeros(tau.shape[0], dtype=tau.dtype, device=tau.device)
        down_levels = [down]
        for layer in range(tau.shape[1]):
            down = down * trans[:, layer] + planck[:, layer] * (1.0 - trans[:, layer])
            down_levels.append(down)
        up = surface_planck
        up_levels = [up]
        for layer in range(tau.shape[1] - 1, -1, -1):
            up = up * trans[:, layer] + planck[:, layer + 1] * (1.0 - trans[:, layer])
            up_levels.append(up)
        result.flux_up = torch.stack(list(reversed(up_levels)), dim=1)
        result.flux_down = torch.stack(down_levels, dim=1)
        result.flux_net = result.flux_up - result.flux_down if return_net else None
        return result
'''
    (package / "__init__.py").write_text(
        f'''
class TwoStreamEssOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TwoStreamEss:
    def __init__(self, options):
        self.options = options
{forward_flux}
'''
    )
    return root


def write_model_sw_flux(
    path: Path,
    profile_ids: np.ndarray,
    pressure: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("column", profile_ids.size)
        ds.createDimension("half_level", pressure.shape[1])
        pid = ds.createVariable("profile_id", "i4", ("column",))
        p = ds.createVariable("pressure_hl", "f8", ("column", "half_level"))
        u = ds.createVariable("flux_up_sw", "f8", ("column", "half_level"))
        d = ds.createVariable("flux_dn_sw", "f8", ("column", "half_level"))
        pid[:] = profile_ids
        p[:, :] = pressure
        u[:, :] = up
        d[:, :] = down


def write_truth_sw_flux(
    path: Path,
    profile_ids: np.ndarray,
    pressure: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    *,
    band: int,
) -> None:
    band_index = band - 1
    pressure_all = np.zeros((50, pressure.shape[1]), dtype=np.float64)
    up_all = np.zeros((50, 5, pressure.shape[1], 13), dtype=np.float64)
    down_all = np.zeros((50, 5, pressure.shape[1], 13), dtype=np.float64)
    for row, profile_id in enumerate(profile_ids.tolist()):
        pressure_all[profile_id] = pressure[row]
        up_all[profile_id, 2, :, band_index] = up[row]
        down_all[profile_id, 2, :, band_index] = down[row]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pressure_hl", data=pressure_all)
        handle.create_dataset("band_flux_up_sw", data=up_all)
        handle.create_dataset("band_flux_dn_sw", data=down_all)


if __name__ == "__main__":
    unittest.main()
