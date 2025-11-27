# JAX-MD Hydrodynamics Module

This module provides hydrodynamic mobility operators for Stokes flow simulations of colloidal suspensions in periodic domains.

## Overview

The hydrodynamic interactions between particles suspended in a viscous fluid are computed using the Pse-split method, which decomposes the mobility operator into:

```
M = M^(r) + M^(w)
```

where:
- **M^(r)**: Real-space contribution (short-range, computed with closed-form kernels)
- **M^(w)**: Wave-space contribution (long-range, computed using Spectral Pse with FFTs)

## Modules

### `pse.py` - Main Interface
High-level functions for building complete Pse mobility operators:
- `build_pse_mobility()`: Accepts JAX-MD space functions (works with static and shearing boxes)
- `build_pse_mobility_direct()`: Direct interface with box matrix
- `suggest_pse_params()`: Parameter selection based on error tolerances

### `pse_real.py` - Real-Space Mobility
Implements M^(r) using Fiore's closed-form F1,F2 coefficients:
- `build_Mr_apply()`: Returns `(init_fn, apply_fn)` that manage neighbor lists automatically
- `F1F2_closed_form()`: Viscosity-independent geometric coefficients F1(r; a, ξ) and F2(r; a, ξ) from Fiore Appendix A
- `Mr_self()`: Eta-independent self-mobility factor; multiply by 1/(6πηa) for Cartesian self-mobility

### `pse_wave.py` - Wave-Space Mobility  
Implements M^(w) using Spectral Pse method:
- `build_Mw_apply()`: Constructs wave-space mobility operator
- `build_B_modes()`: Builds fluid kernel and shape operator separately
- NUFFT operations: `spread()`, `gather()` for particle-grid transfers

## Quick Start

```python
from jax_md import space
from jax_md.hydro import pse
import jax.numpy as jnp

# Define periodic box
box = jnp.eye(3) * 10.0  # 10x10x10 cubic box
space_fns = space.periodic_general(box, fractional_coordinates=True)

# Build mobility operator (init/apply pair)
init_fn, apply_fn = pse.build_pse_mobility(
    space_fns,
    a=0.03,      # particle radius
    xi=10.0,     # Pse splitting parameter
    eta=1.0,     # fluid viscosity
    P=16,        # quadrature points
    Mgrid=64     # FFT grid size
)

# Apply to compute velocities (positions in fractional coords)
positions = jnp.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]])  # fractional coords
forces = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
state = init_fn(positions)
velocities, state = apply_fn(state, positions, forces)
```

## Shearing Flows

For non-equilibrium molecular dynamics (NEMD) with simple shear:

```python
from jax_md import space
from jax_md.hydro import pse

# Define shearing box
gamma_dot = 0.1  # shear rate
shear_fn = lambda t: gamma_dot * t
space_fns = space.shearing(box, shear_schedule=shear_fn, fractional_coordinates=True)

# Build mobility operator (same pattern)
init_fn, apply_fn = pse.build_pse_mobility(space_fns, a=0.03, xi=10.0, eta=1.0, P=16, Mgrid=64)

# Use with time-dependent box (pass kwargs through)
state = init_fn(positions, t=0.0)
velocities, state = apply_fn(state, positions, forces, t=current_time)
```

## Parameter Selection

Use `suggest_pse_params()` to automatically choose parameters for a target accuracy:

```python
from jax_md.hydro.pse import suggest_pse_params

params = suggest_pse_params(
    tol=1e-6,        # target tolerance
    a=0.03,          # particle radius  
    A=box,           # box matrix
    N_particles=1000
)

print(f"Suggested: ξ={params['xi']}, M={params['M']}, P={params['P']}")
```

## Implementation Details

### Coordinate Systems
- **Fractional coordinates** (default): Positions in [0,1)³, compatible with arbitrary triclinic boxes
- **Real coordinates**: Physical positions in box units (used internally in real-space)

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
2. Wang, M., & Brady, J. F. "Spectral Pse acceleration of Stokesian dynamics." *J. Comput. Phys.* 306 (2016): 443-477.
3. Lindbo, D., & Tornberg, A. K. "Spectral accuracy in fast Pse-based methods." *J. Comput. Phys.* 230 (2011): 8744-8761.

## Testing

See `tests/pse_test.py` for comprehensive tests including:
- Adjointness of spread/gather operators
- Wave-space vs brute-force k-sum comparison
- Symmetry and positive-definiteness checks
- Convergence studies (P-sweep, M-sweep)
