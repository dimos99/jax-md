"""Varga et al. (2015) repulsive-only overlap potential for RPY shear.

This module is for the paper's equilibration stage where attractions are off.
Only the overlap-repulsion term is retained:

  U_R(r) = (16*pi*eta*a**2 / delta_t) * (2a*log((2a)/r) + r - 2a),  0 < r < 2a
         = 0,                                                         otherwise

Reference: Varga et al., Soft Matter 2015, Eq. (5).
"""

import jax.numpy as jnp

POTENTIAL_NAME = "varga_rpy_repulsive_only"

# Required by shear_rpy runner.
POTENTIAL_PARAMS = {
  "particle_radius": 1.0,              # a (length)
  "viscosity": 1.0 / (6.0 * jnp.pi),   # eta
  "repulsion_dt": 1e-4,                # overridden to --dt in shear_rpy.py
  "r_cut": 2.0,                        # 2a
  "r_min": 1e-6,                       # avoids r=0 in log term
}

POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",
  "dr_threshold": 0.5,
  "capacity_multiplier": 2.5,
}


def pair_potential(
  dr,
  particle_radius,
  viscosity,
  repulsion_dt,
  r_cut,
  r_min=1e-6,
  **unused_kwargs,
):
  del r_cut

  a = particle_radius
  diameter = 2.0 * a
  positive = dr > 0.0
  safe_r = jnp.maximum(dr, r_min)
  prefactor = 16.0 * jnp.pi * viscosity * a**2 / repulsion_dt

  return jnp.where(
    (dr < diameter) & positive,
    prefactor * (diameter * jnp.log(diameter / safe_r) + safe_r - diameter),
    0.0,
  )
