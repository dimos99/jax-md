"""
Hydrodynamic mobility operators for Stokes flow.

This package provides positively-split Ewald (PSE) Rotne-Prager-Yamakawa (RPY)
mobility operators for suspensions of spherical particles in periodic domains,
the stresslet extension of Fiore & Swan, *J. Chem. Phys.* **148**, 044114
(2018), and full Fast Stokesian Dynamics (Fiore & Swan, *J. Fluid Mech.* **878**,
544-597, 2019), which adds near-field lubrication.

Three levels of accuracy, cheapest first:

  1. RPY mobility            -- far-field coupling only.
  2. Stresslet-constrained   -- adds rigidity (E = 0).
  3. Stokesian Dynamics      -- adds near-field lubrication.

See ``README.md`` for a guided tour and ``STOKESIAN_DYNAMICS.md`` for a
self-contained treatment of level 3 (method, code map, sign conventions,
tuning). Read the latter before modifying any ``sd_*`` module.

Modules
-------
Public API (re-exported below):

  rpy                : Combined mobility M = M^r + M^w; ``build_rpy_mobility``,
                       ``estimate_rpy_params``, and the grand /
                       stresslet-constrained builders
  rpy_moments        : Couplet / stresslet / torque conventions -- the single
                       source of truth for moment packing and index order
  rpy_constrained    : Stresslet-constrained mobility solver (E = 0)
  rpy_brownian_constrained : Constrained Brownian midpoint SDAE integrator
  sd_nearfield       : Near-field lubrication resistance R^nf (matrix-free)
  sd_saddle          : SD saddle-point solve R_FU = B^T M^-1 B + R^nf_FU
  sd_brownian        : Brownian SD step (split noise + implicit tangent drift,
                       with an RFD fallback)

Implementation modules (import directly when modifying):

  rpy_real           : Re-export shim for the real-space M^r modules
  rpy_real_det       : Deterministic M^r, force-only
  rpy_real_det_dipole: Deterministic M^r, grand (force + couplet)
  rpy_real_stoch     : Lanczos sampler for (M^r)^{1/2}
  rpy_wave           : Re-export shim for the wave-space M^w modules
  rpy_wave_det       : Deterministic M^w, force-only (Spectral Ewald)
  rpy_wave_det_dipole: Deterministic M^w, grand (force + couplet)
  rpy_wave_stoch     : Fourier-space sampler for (M^w)^{1/2}
  sd_nearfield_table : Loads / interpolates the 22 lubrication scalars
  *_helpers          : Scalar kernels, NUFFT spread/gather, lattice bookkeeping

Only the stable, user-facing names are re-exported here. Lower-level building
blocks (scalar kernels, FFT primitives, mode builders, square-root samplers)
remain available from their submodules, e.g. ``jax_md.hydro.rpy_wave``.

For writing simulation *scripts*, prefer the integrators in ``jax_md.simulate``
(``rpy``, ``constrained_rpy``, ``sd``, and their ``_with_shear`` variants) over
driving these builders directly.

Example
-------
>>> from jax_md import space
>>> from jax_md.hydro import rpy
>>>
>>> # Periodic box and space functions.
>>> box = jnp.eye(3) * 10.0  # 10x10x10 cubic box
>>> space_fns = space.periodic_general(box, fractional_coordinates=True)
>>>
>>> # Deterministic mobility (velocities from forces).
>>> init_fn, apply_fn = rpy.build_rpy_mobility(
>>>     space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64)
>>> state = init_fn(positions_fractional)
>>> velocities, state = apply_fn(state, positions_fractional, forces)
>>>
>>> # Stresslet-constrained mobility (rigid particles, E = 0):
>>> init_fn, apply_fn = rpy.build_rpy_mobility(
>>>     space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64,
>>>     use_stresslet=True, constrained=True)
>>> # Constrained Brownian dynamics:
>>> brownian_init, step = apply_fn.make_brownian_step(kT=1.0, dt=1e-3)
>>>
>>> # Full Stokesian Dynamics (near-field lubrication + saddle-point solve).
>>> from jax_md.hydro import build_saddle_solve
>>> sd_init, solve_fn = build_saddle_solve(
>>>     space_fns, a=1.0, eta=1.0, P=16, Mgrid=64)
>>> sd_state = sd_init(positions_fractional)
>>> U, Omega, S5, F_moments, info = solve_fn(
>>>     sd_state, positions_fractional, force, torque, E_inf)
>>>
>>> # ...or, for a whole simulation, use the integrators in jax_md.simulate:
>>> from jax_md import simulate
>>> init_fn, apply_fn = simulate.sd(
>>>     space_fns, energy_fn, dt=1e-4, kT=1.0, a=1.0, eta=1.0)
"""

from jax_md.hydro.rpy_real import RealSpaceState

from jax_md.hydro.rpy_wave import WaveSpaceParams, WaveSpaceState

from jax_md.hydro.rpy import (
    build_rpy_mobility,
    estimate_rpy_params,
    RpyParameterEstimate,
    RpyParameterDiagnostics,
    RpyState,
    brownian_increment,
)

from jax_md.hydro.rpy_constrained import make_constrained_solver

from jax_md.hydro.sd_nearfield import (
    build_nearfield_resistance,
    NearFieldState,
)

from jax_md.hydro.sd_saddle import (
    build_saddle_solve,
    SaddleState,
    Ic0Preconditioner,
)

from jax_md.hydro.sd_brownian import (
    build_sd_brownian_step,
    nearfield_brownian_force,
    make_nearfield_brownian_sampler,
    make_far_field_slip_sampler,
)

from jax_md.hydro.rpy_moments import (
    couplet_to_stresslet_torque,
    couplet_to_orthonormal,
    decompose_gradient,
    flat_to_grand,
    grand_to_flat,
    orthonormal_to_couplet,
    stresslet_basis,
    stresslet_to_couplet,
    torque_to_couplet,
)

from jax_md.hydro.rpy_brownian_constrained import (
    ConstrainedBrownianState,
    make_constrained_brownian_step,
    run_brownian_chunked,
)

__all__ = [
    # Mobility builders and parameter estimation.
    'build_rpy_mobility',
    'estimate_rpy_params',
    'brownian_increment',
    # State and diagnostic containers.
    'RpyState',
    'RpyParameterEstimate',
    'RpyParameterDiagnostics',
    'RealSpaceState',
    'WaveSpaceState',
    'WaveSpaceParams',
    'ConstrainedBrownianState',
    # Stresslet-constrained mobility.
    'make_constrained_solver',
    # Full Stokesian Dynamics: near-field + saddle-point solve.
    'build_nearfield_resistance',
    'NearFieldState',
    'build_saddle_solve',
    'SaddleState',
    'Ic0Preconditioner',
    # Full Stokesian Dynamics: Brownian step and its samplers.
    'build_sd_brownian_step',
    'nearfield_brownian_force',
    'make_nearfield_brownian_sampler',
    'make_far_field_slip_sampler',
    # Constrained Brownian dynamics.
    'make_constrained_brownian_step',
    'run_brownian_chunked',
    # Moment (couplet / stresslet / torque) decomposition.
    'couplet_to_stresslet_torque',
    'decompose_gradient',
    'stresslet_basis',
    'stresslet_to_couplet',
    'torque_to_couplet',
    'couplet_to_orthonormal',
    'orthonormal_to_couplet',
    'grand_to_flat',
    'flat_to_grand',
]
