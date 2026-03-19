"""Hard-core inverse-power tail potential.

  Functional form:
  U(r) = inf                                      for r <= sigma_h
       = epsilon / ((r / sigma_h)**n - 1.0)       for sigma_h < r < r_cut
       = 0                                        for r >= r_cut

"""

import jax.numpy as jnp

POTENTIAL_NAME = "hard_core_inverse_power"

POTENTIAL_PARAMS = {
  "epsilon": 72.0,
  "sigma_h": 2.124,
  "n": 87.0,
  "r_cut": 6.372,   # 3*sigma_h for the defaults above
}

POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",
  "dr_threshold": 0.5,
  "capacity_multiplier": 2.5,
}


def pair_potential(dr, epsilon, sigma_h, n, r_cut, **unused_kwargs):
  # Treat contact as part of the hard core because the analytic tail diverges
  # as r -> sigma_h+.
  safe_r = jnp.maximum(dr, sigma_h + 1e-6 * sigma_h)
  denom = (safe_r / sigma_h) ** n - 1.0
  u = epsilon / denom

  return jnp.where(
    dr <= sigma_h,
    jnp.inf,
    jnp.where(dr < r_cut, u, 0.0),
  )
