# CKDMIP NLPQ Workflow

This directory contains the internal CKDMIP workflow for NLPQ experiments. The
workflow is controlled by one YAML file and runs one domain/band job at a time.
Generated data, CKDMIP files, models, logs, and plots are not part of the source
tree.

## Setup

Create a Python environment and install the local requirements:

```bash
python -m pip install -r subprojects/ckdmip_nlpq_suite/requirements.txt
```

Install the py2sess version that includes `TwoStreamEss.forward_flux` when
running `rt-aware` training:

```bash
git clone <py2sess-repo-url> /path/to/py2sess
cd /path/to/py2sess
python -m pip install -e ".[torch]"
```

Build or install the CKDMIP tools and set the YAML `paths.ckdmip_bin` value to
the directory containing `ckdmip_sw` and `ckdmip_lw`.

The `det` and `rt-aware` paths support longwave and shortwave. `rt-aware`
trains a frozen assignment with py2sess `forward_flux` as the differentiable
flux/heating teacher when `rt.train_teacher: py2sess`. Longwave uses the
thermal source terms. Shortwave uses plane-parallel flux-only solar training
with absorption optical depth, Rayleigh optical depth, CKDMIP solar irradiance,
configured `rt.mu_values`, and surface albedo. SW heating loss uses net-flux
divergence; SW flux loss still compares upwelling and downwelling fluxes to
avoid cancellation. Formal validation and final scoring still use CKDMIP
`ckdmip_lw` or `ckdmip_sw`, not py2sess.

## Configure

Copy and edit the longwave `det + rt-aware` example:

```bash
cp subprojects/ckdmip_nlpq_suite/configs/example.yaml run_lw_band04.yaml
```

For a shortwave `det + rt-aware` run, start from:

```bash
cp subprojects/ckdmip_nlpq_suite/configs/example_sw.yaml run_sw_band02.yaml
```

For a shortwave deterministic-only run, start from:

```bash
cp subprojects/ckdmip_nlpq_suite/configs/example_sw_det.yaml run_sw_band02.yaml
```

Set these paths in the YAML:

- `paths.data_root`: local or shared CKDMIP data directory.
- `paths.run_root`: output directory for run products.
- `paths.ckdmip_bin`: directory with CKDMIP executables.
- `paths.py2sess_repo`: py2sess checkout, required for `rt-aware`.

The example configs use repo-local ignored directories so dry-run commands are
safe to run immediately. For real HPC runs, change these paths to shared
scratch/project storage before downloading data or launching training.

Use one CKDMIP scenario per YAML file. The loader applies the standard CKDMIP
trace-gas scaling for scenarios such as `present`, `preindustrial`, `future`,
`glacialmax`, and single-gas perturbations such as `co2-560`. Per-scenario
looping with combined ranking is intentionally not hidden behind one config yet.

The split is fixed by default:

```yaml
split:
  dev:
    train_profiles: "0-39"
    val_profiles: "40-49"
  final:
    train_profiles: "0-49"
    test_dataset: evaluation2
```

Development tuning uses only Evaluation-1 profiles 0-39 for fitting and 40-49
for frozen validation. The final model is retrained on all 50 Evaluation-1
profiles and Evaluation-2 is used only once for final testing.

## Run

Preflight:

```bash
python subprojects/ckdmip_nlpq_suite/scripts/run_ckdmip_nlpq.py \
  --config run_lw_band04.yaml \
  --stage preflight
```

Plan downloads without writing raw data:

```bash
python subprojects/ckdmip_nlpq_suite/scripts/download_ckdmip_data.py \
  --config run_lw_band04.yaml \
  --dry-run
```

Add `--estimate-size` to query remote file sizes while writing the same plan.

Run the full workflow:

```bash
python subprojects/ckdmip_nlpq_suite/scripts/run_ckdmip_nlpq.py \
  --config run_lw_band04.yaml \
  --stage all
```

On Slurm:

```bash
sbatch subprojects/ckdmip_nlpq_suite/slurm/download_band.sbatch /absolute/path/to/run_lw_band04.yaml
sbatch subprojects/ckdmip_nlpq_suite/slurm/run_band_all.sbatch /absolute/path/to/run_lw_band04.yaml
```

## Outputs

Run products are written under:

```text
runs/{domain}/bandXX/{run_id}/
```

Important files include:

- `download_plan.csv`
- `metrics/{domain}_bandXX_dev_tuning_candidates.csv`
- `metrics/{domain}_bandXX_dev_tuning_ranked.csv`
- `selected_settings.yaml`
- `selected_settings.json`
- `models/{domain}_bandXX_{method}_q{Q}_dev.npz`
- `models/{domain}_bandXX_{method}_q{Q}_final.npz`
- `ckdmip_inputs/{domain}_bandXX_{method}_q{Q}.nc`
- `ckdmip_fluxes/{domain}_bandXX_{method}_q{Q}_fluxes.nc`
- `ckdmip_inputs/{domain}_bandXX_{method}_q{Q}.nam`
- `vertical/{domain}_bandXX_train_vertical_outputs.npz`
- `vertical/{domain}_bandXX_val_vertical_outputs.npz`
- `vertical/{domain}_bandXX_test_vertical_outputs.npz`
- `reports/{domain}_bandXX_final_report.md`
- `manifest_{domain}_bandXX.json`

All output names include the domain and band id.

`dev_tune` writes validation vertical outputs for the selected candidate.
`final_train` retrains the selected model on all Evaluation-1 profiles, then
runs CKDMIP RT on that train set to write train vertical outputs. `final_test`
loads the frozen final model and writes test vertical outputs from Evaluation-2.

## Required Gates

- Missing CKDMIP executables fail preflight.
- Missing py2sess or missing `TwoStreamEss.forward_flux` fails preflight when
  `rt-aware` is requested.
- Missing official CKDMIP spectra or flux truth fails outside download/dry-run
  stages.
- SW `rt-aware` requires official CKDMIP Rayleigh optical depth and solar
  irradiance unless `rt.allow_zero_rayleigh: true` is explicitly set for a
  diagnostic run.
- SW `rt-aware` uses py2sess plane-parallel flux-only training; keep
  `rt.sw_include_fo: false`.
- Evaluation-2 must not appear in tuning.
- Train/validation profile leakage is rejected.
- Dev tuning exports frozen candidates, runs CKDMIP RT, and ranks on flux and
  heating metrics when `tuning.require_ckdmip_rt_for_selection` is true.
- Formal compressed evaluation requires official CKDMIP spectra, flux truth,
  CKDMIP executables, and the domain-specific source terms required by CKDMIP
  CKD mode.
