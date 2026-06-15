"""
Hydrodynamic mobility operators for Stokes flow.

This package provides positively-split Ewald (PSE) Rotne-Prager-Yamakawa (RPY)
mobility operators for suspensions of spherical particles in periodic domains,
including the stresslet extension of Fiore & Swan, *J. Chem. Phys.* **148**,
044114 (2018).

Modules
-------
rpy_real             : Real-space RPY mobility (M^r)
rpy_wave             : Wave-space Spectral Ewald RPY mobility (M^w)
rpy                  : Combined mobility operator (M = M^r + M^w) and the
                       grand / stresslet-constrained builders
rpy_moments          : Couplet / stresslet / torque moment conventions
rpy_constrained      : Stresslet-constrained mobility solver
rpy_brownian_constrained : Constrained Brownian midpoint SDAE integrator

Only the stable, user-facing names are re-exported here. Lower-level building
blocks (scalar kernels, FFT primitives, mode builders, square-root samplers)
remain available from their submodules, e.g. ``jax_md.hydro.rpy_wave``.

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
