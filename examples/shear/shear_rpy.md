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
- `examples/shear/potentials/high_exp_lj.py`
- `examples/shear/potentials/varga_ao_rpy_overlap.py`

You can copy one and edit it, or pass either directly via `--potential`.

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

For integer-valued count/step flags, scientific notation is accepted as long as
the value is still an integer (example: `--n_steps 6e6`).

## 9. Outputs

Production files in `--out_dir` (depending on settings):

- `params.json`
- `stress.dat` (only if `--stress_every > 0`)
- `traj.dump` (only if `--traj_every > 0`)
- `confin.data` (always; start-of-run configuration)
- for `--init-data`, `confin.data` is an exact copy of input
- `confout.data` (always; final frame)
- `confout*` step metadata is written from exact integer step counters

## 10. Common errors

- `--potential is required`
- `--dt is required`
- `--out_dir is required`
- `Random initialization mode requires both --n_particles and --phi`
- `--init-traj and --init-data cannot be used together`
- `When --init-traj or --init-data is provided, do not pass --n_particles or --phi`
- `potential module must define POTENTIAL_PARAMS`
- `potential module must define POTENTIAL_NEIGHBOR_PARAMS`
