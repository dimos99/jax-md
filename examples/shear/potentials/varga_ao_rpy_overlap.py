"""Varga et al. (2015) AO + overlap-repulsion potential for RPY shear.

  Functional form:
  U(r) = U_A(r) + U_R(r)

  Attractive AO contribution (Varga et al., Soft Matter 2015, Eq. 3):
    alpha = 2a * (1 + delta)

    U_A(r) = U_A(2a) * (2*alpha**3 - 3*r*alpha**2 + r**3)
                        / (2*alpha**3 - 3*(2a)*alpha**2 + (2a)**3)
             for 2a < r < alpha
           = 0
             for r >= alpha

  To match Varga et al. strictly, U_A is applied only on 2a < r < alpha and
  is zero otherwise (including r <= 2a).

  Repulsive overlap term for hydrodynamically interacting particles
  (Varga et al., Soft Matter 2015, Eq. 5):
    U_R(r) = (16*pi*eta*a**2 / delta_t) * (2a*log((2a)/r) + r - 2a)
             for 0 < r < 2a
           = 0
             otherwise
"""

import jax.numpy as jnp

POTENTIAL_NAME = "varga_ao_rpy_overlap"

# Required: provide finite r_cut > 0 so the runner can build a neighbor list.
# Keep:
#   r_cut >= 2*particle_radius*(1 + ao_attr_range).
POTENTIAL_PARAMS = {
  "ua_contact": -5.0,                  # kT, U_A(2a)
  "ao_attr_range": 0.1,                # delta = R_g / a
  "particle_radius": 1.0,              # a (length)
  "viscosity": 1.0 / (6.0 * jnp.pi),   # eta
  "repulsion_dt": 1e-4,                # delta_t used in overlap repulsion
  "r_cut": 2.2,                        # 2a(1+delta) for defaults above
  "r_min": 1e-6,                       # length; avoids r=0 in log term
}

# Required: define interaction-neighbor defaults for this potential.
POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",  # one of: dense, sparse, ordered
  "dr_threshold": 0.5,
  "capacity_multiplier": 2.5,
}


def pair_potential(
  dr,
  ua_contact,
  ao_attr_range,
  particle_radius,
  viscosity,
  repulsion_dt,
  r_cut,
  r_min=1e-6,
  **unused_kwargs,
):
  del r_cut

  # Geometric scales.
  a = particle_radius
  diameter = 2.0 * a
  alpha = diameter * (1.0 + ao_attr_range)
  positive = dr > 0.0

  # AO attraction (Eq. 3), applied only on 2a < r < alpha.
  denom = 2.0 * alpha**3 - 3.0 * diameter * alpha**2 + diameter**3
  ao_shape = (2.0 * alpha**3 - 3.0 * dr * alpha**2 + dr**3) / denom
  u_attr = jnp.where((dr > diameter) & (dr < alpha), ua_contact * ao_shape, 0.0)
  # Optional continuity extension at contact (disabled for strict paper form):
  # u_attr = jnp.where(
  #   (dr <= diameter) & positive,
  #   ua_contact,
  #   jnp.where((dr < alpha) & positive, ua_contact * ao_shape, 0.0),
  # )

  # Overlap repulsion for RPY-interacting particles (Eq. 5).
  safe_r = jnp.maximum(dr, r_min)
  prefactor = 16.0 * jnp.pi * viscosity * a**2 / repulsion_dt
  u_rep = jnp.where(
    (dr < diameter) & positive,
    prefactor * (diameter * jnp.log(diameter / safe_r) + safe_r - diameter),
    0.0,
  )

  return u_attr + u_rep
