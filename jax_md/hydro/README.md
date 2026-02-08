# JAX-MD Hydrodynamics Module

This module provides hydrodynamic mobility operators for Stokes flow simulations of colloidal suspensions in periodic domains.

## Overview

The hydrodynamic interactions between particles suspended in a viscous fluid are computed using a split-Ewald RPY method, which decomposes the mobility operator into:

```
M = M^(r) + M^(w)
```

where:
- **M^(r)**: Real-space contribution (short-range, computed with closed-form kernels)
- **M^(w)**: Wave-space contribution (long-range, computed using Spectral Ewald with FFTs)

## Modules

### `rpy.py` - Main Interface
High-level functions for building complete RPY mobility operators:
- `build_rpy_mobility()` / `build_rpy_matvec()`: Accepts JAX-MD space functions (works with static and shearing boxes)

### `rpy_real.py` - Real-Space Mobility
Implements M^(r) using Fiore's closed-form F1,F2 coefficients:
- `build_Mr_apply()`: Returns `(init_fn, apply_fn)` that manage neighbor lists automatically
- `F1F2_closed_form()`: Viscosity-independent geometric coefficients F1(r; a, ξ) and F2(r; a, ξ) from Fiore Appendix A
- `Mr_self()`: Eta-independent self-mobility factor; multiply by 1/(6πηa) for Cartesian self-mobility

### `rpy_wave.py` - Wave-Space Mobility  
Implements M^(w) using Spectral Ewald:
- `build_wave_modes()`: Precomputes shape (P), fluid kernels (B), and metadata, returning a `WaveSpaceState` for M^(w)
- `build_Mw_apply()`: Constructs the wave-space mobility operator from a `WaveSpaceState`
- `build_B_modes()`: Fluid kernel only (for advanced/custom pipelines)
- NUFFT operations: `spread()`, `gather()` for particle-grid transfers
- Deterministic implementation: `rpy_wave_det.py` (Fiore Ch. 3, Eq. 3.16–3.19)
- Stochastic utilities: `rpy_wave_stoch.py` (square-root sampling / fused apply+sample)

## Quick Start

```python
import jax
from jax_md import space
from jax_md.hydro import rpy
import jax.numpy as jnp

# Define periodic box
box = jnp.eye(3) * 10.0  # 10x10x10 cubic box
space_fns = space.periodic_general(box, fractional_coordinates=True)

# Build mobility operator (init/apply pair)
init_fn, apply_fn = rpy.build_rpy_mobility(
    space_fns,
    a=0.03,      # particle radius
    xi=0.7,      # Ewald splitting parameter (xi * a ≈ 0.5 is a good starting point)
    eta=1.0,     # fluid viscosity
    P=16,        # quadrature points
    Mgrid=64     # FFT grid size
)

# Apply to compute velocities (positions in fractional coords)
positions = jnp.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])  # fractional coords
forces = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
state = init_fn(positions)
velocities, state = apply_fn(state, positions, forces)

# With Brownian noise
key = jax.random.PRNGKey(0)
velocities, noise, state, info = apply_fn(
    state,
    positions,
    forces,
    brownian_key=key,
    kT=1.0,
    dt=1e-3,
)
```

`xi` must be positive; values with `xi * a ≈ 0.5` are a good starting point.

## Fiore 2017 Figure Reproduction

Reproduce the analytically/algorithmically reproducible Fiore 2017 figures
(condition number, error vs tolerance, RPY vs FCM kernel speedup):

```bash
python jax_md/hydro/rpy_2017_figures.py --outdir output/rpy_2017
```

Performance/timing figures (FIG. 4–8) require implementation-specific benchmarks.

## Fiore Ch. 3 Performance Benchmarks

Reproduce Fiore Ch. 3 performance figures (3.6–3.9) with the JAX RPY
implementation. Outputs PNGs plus CSV/JSON metadata under
`output/rpy_ch3_benchmarks/`:

```bash
python jax_md/hydro/rpy_ch3_benchmarks.py --figs 6,7,8,9
```

These timings are hardware-dependent; defaults assume a strong GPU. For a
quick smoke run:

```bash
python jax_md/hydro/rpy_ch3_benchmarks.py --figs 6 --N_list 1024 --xi_a_vals 0.3,0.5,0.7 --steps 2 --warmup 1
```

## Tutorial Script

Run a compact end-to-end tutorial with equilibrium + shear simulations and an xi-scan
diagnostic (plots saved under `output/rpy_tutorial/`):

```bash
python examples/rpy_tutorial.py
```

## Shearing Flows

For non-equilibrium molecular dynamics (NEMD) with simple shear:

```python
from jax_md import space
from jax_md.hydro import rpy

# Define shearing box
gamma_dot = 0.1  # shear rate
shear_fn = lambda t: gamma_dot * t
space_fns = space.shearing(box, shear_schedule=shear_fn, fractional_coordinates=True)

# Build mobility operator (same pattern)
init_fn, apply_fn = rpy.build_rpy_mobility(space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64)

# Use with time-dependent box (pass kwargs through)
state = init_fn(positions, t=0.0)
velocities, state = apply_fn(state, positions, forces, t=current_time)
```

## Implementation Details

### Coordinate Systems
- **Fractional coordinates** (default): Positions in [0,1)³, compatible with arbitrary triclinic boxes
- **Real coordinates**: Physical positions in box units (used internally in real-space)

### Deterministic wave-space
The deterministic wave-space operator follows Fiore Ch. 3 (Eq. 3.16–3.19):
`M^(w) = D† · P† · B · P · D`, accelerated via NUFFT quadrature and Spectral Ewald
deconvolution. See `rpy_wave_det.py` for the canonical implementation.

### Error Control
The method has three independent error sources:
1. **Real-space truncation** (ε_R): exponentially decaying with cutoff radius
2. **Wave-space truncation** (ε_W): exponentially decaying with FFT grid size
3. **Quadrature error** (ε_q): exponentially decaying with stencil support P

### Performance Notes
- Wave-space: O(N P³ + M³ log M) with FFTs
- Real-space: O(N) with automated neighbor lists (rebuilds triggered by dr_threshold)
- For small systems: real-space dominates; for large systems: FFT dominates

## References

1. Fiore, A. M., et al. "Fast Stokesian dynamics." *J. Fluid Mech.* 878 (2019): 544-597.
2. Wang, M., & Brady, J. F. "Spectral Ewald acceleration of Stokesian dynamics." *J. Comput. Phys.* 306 (2016): 443-477.
3. Lindbo, D., & Tornberg, A. K. "Spectral accuracy in fast Ewald-based methods." *J. Comput. Phys.* 230 (2011): 8744-8761.

## Testing

See `tests/rpy_test_concise.py` for fast coverage including:
- Deterministic wave-space mobility vs `Mw_bruteforce` reference (xi sweep subset)
- Shear invariance via the `current_box` wave-space mapping
- Parameter validation for invalid grid/support combinations

See `tests/rpy_physical_test.py` for Fiore-inspired physical validation:
- Analytic RPY limits (self + pair mobility) with overlap / non-overlap
- Symmetry / PSD / translational invariance for the full RPY operator
- Brownian covariance checks
- Xi sweep (0.1 → 1.0) using `estimate_rpy_params`

Slow tests are marked with `@pytest.mark.slow` and can be run with:
`pytest tests/rpy_physical_test.py -m "slow" -v`
