"""High-exponent Lennard-Jones pair potential for stiff gel simulations.

  Functional form:
  V(r) = epsilon * ((sigma/r)**96 - 2*(sigma/r)**48 + c)   for r < r_cut
       = 0                                                  otherwise

  This is a generalized LJ(96-48) potential with steeper repulsive and
  attractive walls than the standard LJ(12-6), producing a narrower and
  deeper well.  The parameter c shifts the baseline:
    c = 0  → gel mode: well minimum is at V = -epsilon (pure attraction)
    c = 1  → equilibration mode: adds +epsilon offset, raising well floor

  The minimum of the bare (c=0, no cutoff) potential occurs at r = sigma,
  where V(sigma) = -epsilon (i.e. the well depth equals epsilon in kT units).

  Cutoff:
    r_cut = 1.5*sigma by default; V(r_cut) ≈ -7e-8*epsilon, negligible.
"""

import jax.numpy as jnp

POTENTIAL_NAME = "high_exp_lj"

POTENTIAL_PARAMS = {
  "epsilon": 10.0,   # ΔU = 10 kT — strong gel (use 5.0 for weak gel)
  "sigma": 2.0,
  "c": 0.0,          # 0 = gel; switch to 1.0 for equilibration
  "r_cut": 3.0,      # 1.5σ — U(r_cut) ≈ -7e-8 kT, negligible
  "r_min": 1e-6,
}

POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",
  "dr_threshold": 0.5,
  "capacity_multiplier": 2.5,
}


def pair_potential(dr, epsilon, sigma, c, r_cut, r_min=1e-6, **unused_kwargs):
  # Clamp to r_min to avoid r=0 singularity under jit/vmap.
  safe_r = jnp.maximum(dr, r_min)
  sr = sigma / safe_r

  # LJ(96-48): repulsive sr^96 and attractive -2*sr^48 terms, shifted by c.
  # Well minimum sits at r = sigma with depth V = -epsilon (when c = 0).
  u = epsilon * (sr**96 - 2.0 * sr**48 + c)

  # Hard cutoff: zero for r >= r_cut.
  return jnp.where(dr < r_cut, u, 0.0)
