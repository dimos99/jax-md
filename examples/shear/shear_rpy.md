# `shear_rpy.py` User Guide

This guide explains how to run:

`examples/shear/shear_rpy.py`

## 1. What this script does

It runs Brownian colloid simulations with:

- RPY hydrodynamic interactions
- optional shear (`--peclet`)
- user-provided interparticle potential (`--potential`)

Console output uses structured tags like `[INFO]`, `[WARN]`, `[RUN]`, `[DONE]`.
If `colorama` is installed and your terminal supports color, tags are colorized.

## 2. Required arguments

You must always provide:

- `--potential`
- `--dt`
- `--out_dir`

Initialization is also required (choose one mode in section 4).

## 3. Fixed physics units and constants

These are hardcoded:

- `a = 1.0`
- `kT = 1.0`
- `viscosity = 1 / (6*pi)`

So `D0 = 1`, and shear mapping is:

- `gammadot = 2 * Pe * D0 / a^2 = 2 * Pe`
- `Pe = 0.5 * gammadot`

If `--peclet` is omitted, default is `0.0` (equilibrium).

## 4. Initialization modes

Choose exactly one:

1. Random + relax mode:
- provide both `--n_particles` and `--phi`
2. Dump mode:
- provide `--init-traj path/to/traj.dump`
- do not pass `--n_particles` or `--phi`
- `n_particles`, `phi`, and box are derived from dump
- run starts at fresh `t=0`, `step=0`
3. LAMMPS data mode:
- provide `--init-data path/to/confin.data`
- do not pass `--n_particles` or `--phi`
- `n_particles`, `phi`, and box are derived from the data file
- run starts at fresh `t=0`, `step=0`

## 5. Potential module contract

Your `--potential` module must define:

1. `pair_potential(dr, **params)`
2. `POTENTIAL_PARAMS` dict with finite `r_cut > 0`
3. `POTENTIAL_NEIGHBOR_PARAMS` dict with:
- `format` in `dense|sparse|ordered`
- `dr_threshold >= 0`
- `capacity_multiplier > 0`

Optional:

- `POTENTIAL_NAME`

## 6. Built-in example potentials

Available in:

- `examples/shear/potentials/ao_wca.py`
- `examples/shear/potentials/auer_derjaguin_brush.py`
- `examples/shear/potentials/high_exp_lj.py`
- `examples/shear/potentials/varga_ao_rpy_overlap.py`

The Auer brush defaults are normalized by hydrodynamic radius `245.17 nm`,
corresponding to physical hydrodynamic diameter `490.34 nm`.

You can copy one and edit it, or pass any of them directly via `--potential`.

## 7. Minimal commands

Random initialization:

```bash
python examples/shear/shear_rpy.py \
  --potential examples/shear/potentials/ao_wca.py \
  --dt 2e-5 \
  --out_dir examples/out/shear_random \
  --n_particles 128 \
  --phi 0.45
```

Dump initialization:

```bash
python examples/shear/shear_rpy.py \
  --potential examples/shear/potentials/ao_wca.py \
  --dt 2e-5 \
  --out_dir examples/out/shear_from_dump \
  --init-traj examples/out/shear_random/traj.dump
```

LAMMPS data initialization:

```bash
python examples/shear/shear_rpy.py \
  --potential examples/shear/potentials/ao_wca.py \
  --dt 2e-5 \
  --out_dir examples/out/shear_from_data \
  --init-data examples/out/shear_from_data/confin.data
```

## 8. Important optional controls (defaults)

- `--peclet 0.0`
- `--progress_every 1000`
- `--mr-skin 0.5`
- `--seed 42`
- `--xi 0.5`
- `--traj_every 100` (`0` disables trajectory output)
- `--stress_every 0` (`0` disables stress calculation/output)
- `--batch-outdir-naming auto`
- `--batch-outdir-run-start 1`

For integer-valued count/step flags, scientific notation is accepted as long as
the value is still an integer (example: `--n_steps 6e6`).

## 8a. In-process batch mode

`shear_rpy.py` can now run multiple same-shape replicas in one JAX process.

- repeat `--seed` to batch multiple random-initialized runs
- repeat `--init-data` to batch multiple LAMMPS data inputs
- repeat `--init-traj` to batch multiple dump inputs
- choose one repeated input mode: repeated `--init-data` or repeated `--init-traj`
- all batched runs must have the same particle count and the same box matrix
- `--batch-outdir-naming seed` uses `seed_<seed>`
- `--batch-outdir-naming input` uses the input filename stem
- `--batch-outdir-naming run` uses `run_0001`, `run_0002`, ...
- `--batch-outdir-run-start N` offsets the first run number for `run` naming

## 9. Outputs

Production files in `--out_dir` (depending on settings):

- `params.json`
- `stress.dat` (only if `--stress_every > 0`)
- `traj.dump` (only if `--traj_every > 0`)
- `confin.data` (always; start-of-run configuration)
- for `--init-data`, `confin.data` is an exact copy of input
- `confout.data` (always; final frame)
- `confout*` step metadata is written from exact integer step counters

In batch mode, `--out_dir` becomes a root directory and each run writes its own
subdirectory containing the same per-run files listed above.

## 10. Varga-style gel postprocessing

To reproduce the single-condition versions of Varga et al. (2015)
Figs. 3, 5, 8, and 10 from a `shear_rpy.py` run, use:

`examples/shear/shear_varga_postprocess.py`

Single-run example:

```bash
python examples/shear/shear_varga_postprocess.py \
  --run-dir output/swan_gelation \
  --out-dir output/swan_gelation/varga_analysis \
  --start-time 100.0 \
  --end-time 300.0 \
  --stride 10
```

Replica-averaged example:

```bash
python examples/shear/shear_varga_postprocess.py \
  --run-dir run_01 \
  --run-dir run_02 \
  --run-dir run_03 \
  --out-dir ensemble_varga \
  --start-time 100.0 \
  --end-time 300.0
```

Important notes:

- Multiple `--run-dir` inputs are treated as replicas of the same condition.
- `--start-time` and `--end-time` are mapped to the nearest stored dump frames.
- `--stride N` analyzes every `N`th stored frame inside the resolved window.
- `--bond-cutoff` defaults to `2a(1 + delta)` when the AO range is present in
  `params.json`.
- `--q-qa` defaults to the paper values `0.52 1.05 2.50 3.96 5.93`.
- The tool writes four PNGs plus `varga_postprocess_results.npz` and
  `varga_postprocess_meta.json`.
- If the automatic fractal-dimension fit is not well constrained, pass
  `--theory-df` explicitly.

## 11. Common errors

- `--potential is required`
- `--dt is required`
- `--out_dir is required`
- `Random initialization mode requires both --n_particles and --phi`
- `--init-traj and --init-data cannot be used together`
- `When --init-traj or --init-data is provided, do not pass --n_particles or --phi`
- `potential module must define POTENTIAL_PARAMS`
- `potential module must define POTENTIAL_NEIGHBOR_PARAMS`
