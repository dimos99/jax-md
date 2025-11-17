"""
Hydrodynamic mobility operators for Stokes flow.

This module provides Pse-split hydrodynamic mobility operators for
suspensions of spherical particles in periodic domains.

Modules:
--------
pse_real : Real-space Pse mobility (M^r)
pse_wave : Wave-space Spectral Pse mobility (M^w)
pse : Combined mobility operator (M = M^r + M^w)

Example:
--------
>>> from jax_md import space
>>> from jax_md.hydro import pse
>>> 
>>> # Setup periodic box and space functions
>>> box = jnp.eye(3) * 10.0  # 10x10x10 cubic box
>>> space_fns = space.periodic_general(box, fractional_coordinates=True)
>>> 
>>> # Build mobility operator
>>> init_fn, apply_fn = pse.build_pse_mobility(
>>>     space_fns, a=0.03, xi=10.0, eta=1.0, P=16, Mgrid=64
>>> )
>>> 
>>> # Apply to positions and forces (fractional coordinates)
>>> state = init_fn(positions_fractional)
>>> velocities, state = apply_fn(state, positions_fractional, forces)
"""

from jax_md.hydro.pse_real import (
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

from jax_md.hydro.pse_wave import (
    build_B_modes,
    build_Mw_apply,
    build_Mw_sqrt_sampler,
    make_reciprocal,
    q_grid,
    k_from_q,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
    Mw_bruteforce
)

from jax_md.hydro.pse import (
    build_pse_mobility,
    build_pse_mobility_direct,
    PseState,
    suggest_pse_params,
    brownian_increment,
    build_euler_maruyama_step
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
    'build_B_modes',
    'build_Mw_apply',
    'build_Mw_sqrt_sampler',
    'make_reciprocal',
    'q_grid',
    'k_from_q',
    'build_stencils_frac',
    'spread',
    'gather',
    'fft_vec',
    'ifft_vec',
    'Mw_bruteforce',
    # Combined
    'build_pse_mobility',
    'build_pse_mobility_direct',
    'PseState',
    'suggest_pse_params',
    'brownian_increment',
    'build_euler_maruyama_step'
]
