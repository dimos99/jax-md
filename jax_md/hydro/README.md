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
High-level function for building complete RPY mobility operators:
- `build_rpy_mobility()`: Accepts JAX-MD space functions (works with static and shearing boxes)

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

## Stresslet-Constrained Mobility (Fiore & Swan 2018)

Rigid spheres cannot deform with the local fluid strain. Enforcing this
*stresslet constraint* extends the RPY mobility to the **grand mobility**, which
couples particle forces **F** and couplets **C** to velocities **U** and
velocity gradients **D** (Fiore & Swan 2018, Eqs. 30–31):

```
[U; D] = M · [F; C],   M = M^(w) + M^(r)
```

Decomposing the couplet into its symmetric stresslet **S** and antisymmetric
torque **L**, and the gradient into rate of strain **E** and angular velocity
**Ω**, the rigid constraint **E = 0** is solved for the stresslet (Eq. 36),
giving the stresslet-constrained mobility (Eqs. 37–38):

```
R_FU^{-1} = M_UF − M_US · M_ES^{-1} · M_EF
```

`M_ES` is symmetric positive-definite and well conditioned, so the stresslet is
recovered matrix-free with GMRES (~8–10 iterations, independent of `N`).

```python
from jax_md import space
from jax_md.hydro import rpy

space_fns = space.periodic_general(box, fractional_coordinates=True)

# Grand mobility: forces + couplets -> velocities + velocity gradients.
init_fn, apply_fn = rpy.build_rpy_mobility(
    space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64, use_stresslet=True)
state = init_fn(positions)
(velocities, gradients), state = apply_fn(state, positions, forces, couplets=C)

# Stresslet-constrained mobility (the stresslet is solved for, not supplied).
init_fn, apply_fn = rpy.build_rpy_mobility(
    space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64,
    use_stresslet=True, constrained=True)
state = init_fn(positions)
(velocities, stresslets), state = apply_fn(state, positions, forces)
```

### Constrained Brownian dynamics

Adding the stresslet constraint turns the equations of motion into an index-1
stochastic differential-algebraic equation (SDAE). It is integrated with the
Fiore & Swan midpoint scheme, which produces Brownian displacements consistent
with the fluctuation-dissipation theorem for the constrained system *without*
forming the square root of the constrained operator: the unconstrained slip is
drawn by positive split sampling (real-space Lanczos square root + wave-space
analytic square root), and the midpoint correlation reproduces the thermal
drift `kT ∇·R_FU^{-1}` to O(dt).

```python
brownian_init, step = apply_fn.make_brownian_step(kT=1.0, dt=1e-3)
state = brownian_init(positions, jax.random.PRNGKey(0))
state, info = step(state)   # advance one constrained Brownian step
```

### Full simulations (recommended entry point)

For writing simulation *scripts* — a chosen pair potential, time stepping, and
shear — use the JAX-MD-conventional integrators in `simulate.py`, which wrap the
stepper above and add the Lees-Edwards shear schedule + fractional remap so the
strain (and the neighbor-list cell size) stay bounded over long runs:

```python
from jax_md import energy, simulate, space

displacement, shift, box_of = space.shearing(
    box, shear_schedule=lambda t: gamma_dot * t,
    fractional_coordinates=True, remap=True)
energy_fn = energy.soft_sphere_pair(displacement, sigma=0.6, epsilon=2.0)

init_fn, apply_fn = simulate.constrained_rpy_with_shear(
    (displacement, shift, box_of), energy_fn, dt=1e-3, kT=1.0,
    a=0.15, eta=1.0,
    shear_vector_schedule=lambda t: (gamma_dot * t, 0.0 * t, 0.0 * t),
    tol=1e-3, n_particles=positions.shape[0])

state = init_fn(jax.random.PRNGKey(0), positions)
apply_fn = jax.jit(apply_fn)
for _ in range(n_steps):
    state = apply_fn(state)
# Per-particle hydrodynamic stresslets: state.brownian_state.stresslet  (N, 5)
```

`xi` and the grid (`rcut`/`P`/`Mgrid`/`theta`/`lattice_extent`) are left unset
here, so the integrator sizes them from `tol` via
[`estimate_rpy_params`](#stresslet-constrained-mobility-fiore--swan-2018) (pass
`n_particles`/`phi` for the cost-optimal `xi`). Supply any of them explicitly to
pin it; pass `xi=...` to fix the splitting and have only the unset grid
parameters estimated.

Use `simulate.constrained_rpy` (with `space.periodic_general`) for the free,
unsheared case. Worked end-to-end examples:
[`examples/shear/shear_constrained_rpy.py`](../../examples/shear/shear_constrained_rpy.py)
and the notebook
[`notebooks/constrained_rpy_shear_tutorial.ipynb`](../../notebooks/constrained_rpy_shear_tutorial.ipynb).

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

1. Fiore, A. M., et al. "Rapid sampling of stochastic displacements in Brownian dynamics simulations." *J. Chem. Phys.* 146, 124116 (2017). — PSE method.
2. Fiore, A. M., & Swan, J. W. "Rapid sampling of stochastic displacements in Brownian dynamics simulations with stresslet constraints." *J. Chem. Phys.* 148, 044114 (2018). — Grand mobility, stresslet constraint, constrained Brownian dynamics.
3. Fiore, A. M., & Swan, J. W. "Fast Stokesian dynamics." *J. Fluid Mech.* 878 (2019): 544-597.
4. Wang, M., & Brady, J. F. "Spectral Ewald acceleration of Stokesian dynamics." *J. Comput. Phys.* 306 (2016): 443-477.
5. Lindbo, D., & Tornberg, A. K. "Spectral accuracy in fast Ewald-based methods." *J. Comput. Phys.* 230 (2011): 8744-8761.

## Testing

The test suite lives under `tests/`. Fast (`not slow`) coverage:
- `tests/rpy_test.py` — core mobility: analytic RPY limits (self + pair),
  two-particle tensor symmetry, split consistency (`M = M^(r) + M^(w)`) and
  xi-invariance, symmetry / PSD / translational / Galilean invariance,
  lightweight fluctuation-dissipation covariance, and parameter validation.
- `tests/rpy_stresslet_test.py` — grand mobility / couplet conventions,
  rotlet-pinned moment signs, post-NUFFT traceless checks.
- `tests/rpy_constrained_test.py` — stresslet-constrained mobility:
  zero-strain residual, `M_ES` symmetry/PSD, dense Schur-complement agreement,
  short-time self-diffusivity pinned to Fiore & Swan (2018) Fig. 7.
- `tests/rpy_brownian_constrained_test.py` — positive-split covariance,
  midpoint vs Euler-Maruyama equilibrium (Boltzmann / structure-factor) checks.
- `tests/rpy_constrained_integrator_test.py` — `simulate.py` integrator wiring.

Shared diagnostics live in `tests/rpy_test_utils.py`. Slow physical-validation
tests are marked `@pytest.mark.slow`; run them with, e.g.:

```bash
pytest tests/rpy_test.py -m "slow" -v
```
