"""Example AO + WCA pair potential for the colloid-gel setup.
  Functional form:
  V(r) = V_WCA(r) + V_AO(r)

  WCA (Weeks-Chandler-Andersen; LJ truncated at the minimum and shifted to 0 at cutoff):
    r_wca_cut = 2**(1/6) * sigma
    V_WCA(r) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6) - V_LJ(r_wca_cut)   for r < r_wca_cut
               = 0                                                          otherwise

  AO attraction (cubic depletion-like shape, cutoff at alpha):
  alpha = ao_diameter * (1 + ao_attr_range)

  shape(r) = -((r**3)/3 - alpha**2*r + 2*alpha**3/3) / denom
  denom    = 2*alpha**3 - 3*alpha**2*ao_diameter + ao_diameter**3

  shape(alpha) = 0 and shape(ao_diameter) = -1/3.
  To avoid a potential jump at r = ao_diameter, V_AO is extended as a constant for r <= ao_diameter:
    V_AO(r) = ao_depth * (-1/3)              for r <= ao_diameter
            = ao_depth * shape(r)            for ao_diameter < r < alpha
            = 0                              for r >= alpha
"""

import jax.numpy as jnp

POTENTIAL_NAME = "ao_wca"

# Required: provide finite r_cut > 0 so the runner can build a neighbor list.
# Keep:
#   r_cut >= max(2**(1/6)*wca_sigma_contact, ao_diameter*(1+ao_attr_range)).
POTENTIAL_PARAMS = {
  "wca_epsilon": 8000.0,        # kT
  "wca_sigma_contact": 1.8,     # length
  "ao_depth": 20.0,             # kT (multiplies shape)
  "ao_attr_range": 0.1,         # dimensionless, relative to ao_diameter
  "ao_diameter": 2.0,           # length
  "r_cut": 2.25,                # length; overwritten by max(WCA cutoff, AO cutoff) if too small
  "r_min": 1e-6,                # length; avoids r=0 blowups
}

# Required: define interaction-neighbor defaults for this potential.
POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",  # one of: dense, sparse, ordered
  "dr_threshold": 0.5,
  "capacity_multiplier": 2.5,
}


def pair_potential(
  dr,
  wca_epsilon,
  wca_sigma_contact,
  ao_depth,
  ao_attr_range,
  ao_diameter,
  r_cut,
  r_min=1e-6,
  **unused_kwargs,
):
  # WCA repulsion (LJ shifted so V(r_wca_cut)=0).
  sigma_eff = wca_sigma_contact
  wca_cut = sigma_eff * (2.0 ** (1.0 / 6.0))
  safe_r = jnp.maximum(dr, r_min)

  r_over_sigma = safe_r / sigma_eff
  r_cut_over_sigma = wca_cut / sigma_eff

  lj_energy = 4.0 * wca_epsilon * ((1.0 / r_over_sigma) ** 12 - (1.0 / r_over_sigma) ** 6)
  lj_cut = 4.0 * wca_epsilon * ((1.0 / r_cut_over_sigma) ** 12 - (1.0 / r_cut_over_sigma) ** 6)
  wca = jnp.where((dr < wca_cut) & (dr > 0.0), lj_energy - lj_cut, 0.0)

  # AO attraction (continuous at r=ao_diameter by extending contact value).
  alpha = ao_diameter * (1.0 + ao_attr_range)
  denom = 2.0 * alpha**3 - 3.0 * alpha**2 * ao_diameter + ao_diameter**3
  ao_shape = -((dr**3) / 3.0 - alpha**2 * dr + 2.0 * alpha**3 / 3.0) / denom

  ao_shape_contact = -1.0 / 3.0  # ao_shape(dr=ao_diameter) for this normalization
  ao = jnp.where(
    dr <= ao_diameter,
    ao_depth * ao_shape_contact,
    jnp.where(dr < alpha, ao_depth * ao_shape, 0.0),
  )

  return wca + ao
