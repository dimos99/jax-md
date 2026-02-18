"""
Hydrodynamic mobility operators for Stokes flow.

This module provides split-Ewald RPY mobility operators for
suspensions of spherical particles in periodic domains.

Modules:
--------
rpy_real : Real-space RPY mobility (M^r)
rpy_wave : Wave-space Spectral Ewald RPY mobility (M^w)
rpy : Combined mobility operator (M = M^r + M^w)

Example:
--------
>>> from jax_md import space
>>> from jax_md.hydro import rpy
>>> 
>>> # Setup periodic box and space functions
>>> box = jnp.eye(3) * 10.0  # 10x10x10 cubic box
>>> space_fns = space.periodic_general(box, fractional_coordinates=True)
>>> 
>>> # Build mobility operator (space must be provided explicitly)
>>> init_fn, apply_fn = rpy.build_rpy_mobility(
>>>     space_fns, a=0.03, xi=0.7, eta=1.0, P=16, Mgrid=64
>>> )
>>> 
>>> # Apply to positions and forces (fractional coordinates)
>>> state = init_fn(positions_fractional)
>>> velocities, state = apply_fn(state, positions_fractional, forces)
"""

from jax_md.hydro.rpy_real import (
    F1F2_closed_form,
    Mr_pair_block,
    Mr_self,
    RealSpaceState,
    build_Mr_apply,
    mr_matvec,
    sample_mr_sqrt_precond,
    lanczos_sqrt_mv,
    sample_mr_sqrt
)

from jax_md.hydro.rpy_wave import (
    WaveSpaceParams,
    WaveSpaceState,
    build_Mw,
    build_Mw_state,
    build_B_modes,
    build_wave_modes,
    build_Mw_apply,
    mw_matvec,
    build_Mw_sqrt_sampler,
    build_Mw_apply_and_sample,
    make_reciprocal,
    q_grid,
    k_from_q,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
)

from jax_md.hydro.rpy import (
    build_rpy_mobility,
    estimate_rpy_params,
    RpyParameterEstimate,
    RpyParameterDiagnostics,
    RpyState,
    brownian_increment,
)

__all__ = [
    # Real-space
    'F1F2_closed_form',
    'Mr_pair_block',
    'Mr_self',
    'RealSpaceState',
    'build_Mr_apply',
    'mr_matvec',
    'sample_mr_sqrt_precond',
    'lanczos_sqrt_mv',
    'sample_mr_sqrt',
    # Wave-space
    'WaveSpaceParams',
    'WaveSpaceState',
    'build_Mw',
    'build_Mw_state',
    'build_B_modes',
    'build_wave_modes',
    'build_Mw_apply',
    'mw_matvec',
    'build_Mw_sqrt_sampler',
    'build_Mw_apply_and_sample',
    'make_reciprocal',
    'q_grid',
    'k_from_q',
    'build_stencils_frac',
    'spread',
    'gather',
    'fft_vec',
    'ifft_vec',
    # Combined
    'build_rpy_mobility',
    'estimate_rpy_params',
    'RpyParameterEstimate',
    'RpyParameterDiagnostics',
    'RpyState',
    'brownian_increment',
]
