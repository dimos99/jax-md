# `shear_slab_surface.py` User Guide

This script runs a **single** RPY shear trajectory in a 3D periodic slab while
constraining all particle motion to a fixed z plane.

## What It Simulates

- 3D RPY hydrodynamics (`simulate.rpy_with_shear`)
- xy shear with remapping enabled
- Varga-style overlap repulsion (repulsive-only)
- Surface constraint: no z integration (`dR_z = 0` each step)

This is a quasi-2D setup (3D hydrodynamics, 2D particle motion).

## Required Arguments

- `--n_particles`
- `--phi` (in-plane area fraction)
- `--dt`
- `--out_dir`

## Optional Arguments

- `--n_steps` (default `30000`)
- `--peclet` (default `0.0`)
- `--seed` (default `42`)
- `--traj_every` (default `100`, must be `> 0`)
- `--progress_every` (default `1000`, set `0` to disable progress logs)
- `--xi` (default `0.5`)
- `--tol` (default `1e-4`)
- `--slab_height_factor` (default `4.0`, sets `Lz = slab_height_factor * Lxy`)
- `--z0_frac` (default `0.5`)
- `--relax_steps` (default `250`)

## Example

```bash
python examples/shear/shear_slab_surface.py \
  --n_particles 128 \
  --phi 0.35 \
  --dt 2e-5 \
  --n_steps 20000 \
  --peclet 0.5 \
  --out_dir examples/out/shear_slab_surface
```

## Outputs

Written to `--out_dir`:

- `params.json`
- `confin.data`
- `traj.dump`
- `confout.data`

`params.json` includes:

- `surface_constraint = true`
- `phi_area`
- `phi_volume_for_estimator`
- `slab_height_factor`
- `z0_frac`
