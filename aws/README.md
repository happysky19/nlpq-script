# AWS Runbook

This directory contains AWS-facing wrappers for the CKDMIP NLPQ workflow.  The
physics workflow remains the YAML-controlled runner in `scripts/run_ckdmip_nlpq.py`.

## Recommended instance layout

- CPU/table/data jobs: `c7i.8xlarge` or `r7i.8xlarge` if memory becomes limiting.
- RT-aware training: `g5.xlarge` or `g5.2xlarge`.
- Storage: start with a 500 GB `gp3` EBS volume mounted at `/mnt/nlpq`.

The AWS pilot YAMLs assume:

```text
/mnt/nlpq/data
/mnt/nlpq/runs
/mnt/nlpq/external/ckdmip/bin
/mnt/nlpq/external/py2sess
```

Adjust `paths.*` in the copied YAMLs if your mount points differ.

## Bootstrap

```bash
cd /mnt/nlpq/ckdmip_nlpq_suite
bash aws/bootstrap_ubuntu.sh /mnt/nlpq/ckdmip_nlpq_suite
source .venv/bin/activate
```

The bootstrap script also installs the two external runtime dependencies:

1. py2sess from `https://github.com/happysky19/py2sess.git`.
2. CKDMIP executables `ckdmip_lw` and `ckdmip_sw`.

By default it installs them under:

```text
/mnt/nlpq/external/py2sess
/mnt/nlpq/external/ckdmip/bin
```

If the default CKDMIP source URL changes, pass the official tarball explicitly:

```bash
CKDMIP_SOURCE_URL=https://.../ckdmip-1.0.tar.gz \
  bash aws/bootstrap_ubuntu.sh /mnt/nlpq/ckdmip_nlpq_suite
```

To skip external installation because the paths are already mounted:

```bash
SKIP_EXTERNAL_DEPS=1 bash aws/bootstrap_ubuntu.sh /mnt/nlpq/ckdmip_nlpq_suite
```

## Preflight only

```bash
python scripts/aws_preflight.py \
  --config configs/aws_lw_band04_pilot.yaml \
  --config configs/aws_sw_band02_pilot.yaml \
  --json-out runs/_aws_logs/preflight.json
```

Use `--require-data` after downloading CKDMIP files.

## Pilot run

```bash
bash aws/run_pilot.sh
```

This runs LW band04 and SW band02 through:

```text
preflight -> download -> dev_tune -> final_train -> final_test -> plot -> report
```

Logs and the batch manifest are written under:

```text
runs/_aws_logs/<UTC timestamp>/
```

## Full run pattern

Create one YAML per domain/band group.  Then run:

```bash
python scripts/aws_run_batch.py \
  --config configs/my_lw.yaml \
  --config configs/my_sw.yaml \
  --stages preflight download dev_tune final_train final_test plot report
```

For Spot instances, keep `run_root` and `data_root` on persistent EBS.  The
batch runner writes a manifest after each stage so interrupted runs can be
restarted from the failed stage.

## Hard rules

- Do not tune on Evaluation-2.
- Do not set `rt.allow_zero_rayleigh: true` for formal SW runs.
- Keep `rt.sw_include_fo: true` and `rt.sw_plane_parallel: true` unless running
  a diagnostic ablation.
- Final scoring is CKDMIP `ckdmip_lw`/`ckdmip_sw`, not py2sess.
- `rt-aware-path` is frozen assignment training; `rt-aware-nn` is a separate
  manual ablation and should not be hidden in production runs.
