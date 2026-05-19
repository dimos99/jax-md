"""Auer brush repulsion mapped to spheres via the Derjaguin approximation.

For identical particles with core radius ``a_c``, Auer's flat-plate brush
pressure gives the sphere-sphere pair potential

  U(r) = inf,  r <= 2 a_c

       = (pi a_c alpha / s**3) * [
           (64 * 2**(1/4) / 5) * L**(9/4) * (r - 2 a_c)**(-1/4)
           - (192 / 11) * L**2
           + (96 / 35) * L * (r - 2 a_c)
           - (8 * 2**(1/4) / 77) * L**(-3/4) * (r - 2 a_c)**(11/4)
         ],  2 a_c < r < 2 a_c + 2 L

       = 0,  r >= 2 a_c + 2 L

Energies are expressed in kT units, so the explicit k_B T factor is absorbed
into the shear runner's default convention.

Physical baseline used for the defaults:
  core radius a_c = 237.5 nm
  brush length L = 13.5 nm
  graft spacing s = 2.0 nm
  alpha = 0.025

Normalization:
  the shear examples use hydrodynamic radius a = 1. For a physical
  hydrodynamic diameter of 490.34 nm, the length unit is the hydrodynamic
  radius 245.17 nm.
"""

import jax.numpy as jnp

POTENTIAL_NAME = "auer_derjaguin_brush"

_HYDRO_DIAMETER_NM = 490.34
_HYDRO_RADIUS_NM = 0.5 * _HYDRO_DIAMETER_NM
_CORE_RADIUS_NM = 237.5
_BRUSH_LENGTH_NM = 13.5
_GRAFT_SPACING_NM = 2.0
_ALPHA = 0.025

_DEFAULT_A_C = _CORE_RADIUS_NM / _HYDRO_RADIUS_NM
_DEFAULT_L = _BRUSH_LENGTH_NM / _HYDRO_RADIUS_NM
_DEFAULT_S = _GRAFT_SPACING_NM / _HYDRO_RADIUS_NM

POTENTIAL_PARAMS = {
  "a_c": _DEFAULT_A_C,
  "L": _DEFAULT_L,
  "s": _DEFAULT_S,
  "alpha": _ALPHA,
  "r_cut": 2.0 * (_DEFAULT_A_C + _DEFAULT_L),
}

POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",
  "dr_threshold": 0.5,
  "capacity_multiplier": 4.5,
}


def pair_potential(dr, a_c, L, s, alpha, r_cut, **unused_kwargs):
  contact = 2.0 * a_c
  gap = dr - contact
  safe_gap = jnp.maximum(gap, 1e-6 * a_c)
  two_to_quarter = 2.0 ** 0.25

  prefactor = jnp.pi * a_c * alpha / (s ** 3)
  bracket = (
    (64.0 * two_to_quarter / 5.0) * (L ** (9.0 / 4.0)) * (safe_gap ** (-1.0 / 4.0))
    - (192.0 / 11.0) * (L ** 2)
    + (96.0 / 35.0) * L * safe_gap
    - (8.0 * two_to_quarter / 77.0) * (L ** (-3.0 / 4.0)) * (safe_gap ** (11.0 / 4.0))
  )
  u = prefactor * bracket

  return jnp.where(
    dr <= contact,
    jnp.inf,
    jnp.where(dr < r_cut, u, 0.0),
  )
