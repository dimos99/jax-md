"""Scalar helpers for the real-space force-dipole (couplet) RPY kernels.

Implements the UC scalars (G1, G2) and DC scalars (K1, K2, K3) of the
stresslet-extended PSE method (Fiore & Swan 2018), plus the DC self-mobility
scalar ``Mr_self_dipole``.  Each scalar follows the same skeleton as
``F1F2_closed_form`` in ``rpy_real_det_helpers.py``:

  f0 + f1 e^{-(r xi)^2} + f2 e^{-((r-2a) xi)^2} + f3 e^{-((r+2a) xi)^2}
     + f4 erfc(r xi) + f5 erfc((r-2a) xi) + f6 erfc((r+2a) xi)

PROVENANCE / VALIDATION
  The coefficients are NOT transcribed from the densely typeset paper
  appendix (several of the printed coefficients are corrupt).  They were
  extracted mechanically from the reference HOOMD implementation by the
  method's authors -- the FSD plugin table builder (``Stokes.cc`` in
  github.com/GeZhouyang/FSD, scalars ``g1, g2, h1, h2, h3``) -- by parsing
  the quadruple-precision C expressions with sympy in exact rational
  arithmetic and collecting them onto the skeleton above.  Two facts fall
  out of that decomposition and are relied upon here:

  * f1..f6 are IDENTICAL for the overlapping (r <= 2a) and non-overlapping
    (r > 2a) branches; only the constant f0 differs (0 for r > 2a, a
    xi-independent polynomial that vanishes at r = 2a for r <= 2a).  Hence
    no ``lax.cond`` is needed -- a single ``jnp.where`` on f0 suffices and
    continuity at r = 2a is automatic.
  * The h2/h3 coefficients genuinely satisfy f5 == f6 (the paper prints the
    same; it is correct, not a typo).

  The scalars were validated against an independent quadrature ground truth
  (``tests/rpy_quadrature_reference.py``): G1 = -2*dI_j1 + 2*dI_j2x,
  G2 = -2*dI_j2x, K1 = -dW3, K2 = dW4 - dW2, K3 = dW4, where d denotes
  (free-space minus Hasimoto-screened) radial integrals.

NORMALIZATION
  All five scalars are the dimensioned radial functions multiplying a
  single ``1/(6 pi eta)`` prefactor (the scalars carry the 1/a^2 (UC/DF)
  and 1/a^3 (DC) length dimensions internally).  This was pinned externally
  at radius a != 1 against both the quadrature reference and FSD, and by
  the free-space limits: Mr_self_dipole(a, xi->0) = -3/(10 a^3), which
  reproduces the single-sphere strain mobility 3/(20 pi eta a^3) and
  rotation mobility 1/(8 pi eta a^3).

  Sign/half-factor bookkeeping lives in the tensor contraction, not here
  (FSD likewise tabulates UC1 = g1/2, UC2 = -g2/2 from the raw g1, g2):

    6 pi eta M^r_UC,imn = -(G1/2) (d_im rh_n - rh_i rh_m rh_n)
                        + (G2/2) (d_in rh_m + d_mn rh_i - 4 rh_i rh_m rh_n)

NUMERICAL RANGE
  Like FSD (which evaluates this table in __float128), the skeleton incurs
  catastrophic cancellation as xi*a -> 0: terms grow like 1/(xi*a)^8 while
  the result stays O(1).  In float64 the relative error is ~1e-8 at
  xi*a = 0.2 and ~1e-5 at xi*a = 0.1 (small r).  Keep xi*a >= 0.2 (the
  parameter estimator does) or accept the loss.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

from jax_md.hydro.rpy_real_det_helpers import (
    PAIR_EPS_FRACTION_OF_DIAMETER,
    SQRT_PI,
)


def _skeleton(r, a, xi, f0_lt, f1, f2, f3, f4, f5, f6):
  """Assemble f0 + sum(f_i * gaussian/erfc factors); f0 = 0 for r > 2a."""
  f0 = jnp.where(r > 2.0 * a, 0.0, f0_lt)
  return (f0
          + f1 * jnp.exp(-(r * xi) ** 2)
          + f2 * jnp.exp(-((r - 2.0 * a) * xi) ** 2)
          + f3 * jnp.exp(-((r + 2.0 * a) * xi) ** 2)
          + f4 * erfc(r * xi)
          + f5 * erfc((r - 2.0 * a) * xi)
          + f6 * erfc((r + 2.0 * a) * xi))


# ----------------------------------------------------------------------------
# Coefficients, auto-generated from FSD Stokes.cc -- DO NOT EDIT BY HAND.
# Regenerate with the sympy decomposition described in the module docstring.
# ----------------------------------------------------------------------------


def _G1_coeffs(r, a, xi):
  f1 = -3/64*(10*r**4*xi**4 - 5*r**2*xi**2 - 3)/(SQRT_PI*a**4*r**3*xi**5)
  f2 = (3/640)*(-128*a**5*xi**4 - 64*a**4*r*xi**4 - 112*a**3*r**2*xi**4
                + 16*a**3*xi**2 - 56*a**2*r**3*xi**4 + 24*a**2*r*xi**2
                - 28*a*r**4*xi**4 + 34*a*r**2*xi**2 - 6*a + 50*r**5*xi**4
                - 25*r**3*xi**2 - 15*r)/(SQRT_PI*a**4*r**4*xi**5)
  f3 = (3/640)*(128*a**5*xi**4 - 64*a**4*r*xi**4 + 112*a**3*r**2*xi**4
                - 16*a**3*xi**2 - 56*a**2*r**3*xi**4 + 24*a**2*r*xi**2
                + 28*a*r**4*xi**4 - 34*a*r**2*xi**2 + 6*a + 50*r**5*xi**4
                - 25*r**3*xi**2 - 15*r)/(SQRT_PI*a**4*r**4*xi**5)
  f4 = (3/128)*(20*r**6*xi**6 + 3*r**2*xi**2 + 3)/(a**4*r**4*xi**6)
  f5 = -3/1280*(512*a**6*xi**6 + 320*a**4*r**2*xi**6 - 256*a*r**5*xi**6
                + 100*r**6*xi**6 + 15*r**2*xi**2 + 15)/(a**4*r**4*xi**6)
  f6 = -3/1280*(512*a**6*xi**6 + 320*a**4*r**2*xi**6 + 256*a*r**5*xi**6
                + 100*r**6*xi**6 + 15*r**2*xi**2 + 15)/(a**4*r**4*xi**6)
  f0_lt = (3/160)*(-2*a + r)**2*(32*a**4 + 32*a**3*r + 44*a**2*r**2
                                 + 36*a*r**3 + 25*r**4)/(a**4*r**4)
  return f0_lt, f1, f2, f3, f4, f5, f6


def _G2_coeffs(r, a, xi):
  f1 = -3/64*(2*r**4*xi**4 - r**2*xi**2 + 3)/(SQRT_PI*a**4*r**3*xi**5)
  f2 = (3/640)*(128*a**5*xi**4 + 64*a**4*r*xi**4 - 48*a**3*r**2*xi**4
                - 16*a**3*xi**2 - 24*a**2*r**3*xi**4 - 24*a**2*r*xi**2
                - 12*a*r**4*xi**4 - 14*a*r**2*xi**2 + 6*a + 10*r**5*xi**4
                - 5*r**3*xi**2 + 15*r)/(SQRT_PI*a**4*r**4*xi**5)
  f3 = (3/640)*(-128*a**5*xi**4 + 64*a**4*r*xi**4 + 48*a**3*r**2*xi**4
                + 16*a**3*xi**2 - 24*a**2*r**3*xi**4 - 24*a**2*r*xi**2
                + 12*a*r**4*xi**4 + 14*a*r**2*xi**2 - 6*a + 10*r**5*xi**4
                - 5*r**3*xi**2 + 15*r)/(SQRT_PI*a**4*r**4*xi**5)
  f4 = (3/128)*(4*r**6*xi**6 + 3*r**2*xi**2 - 3)/(a**4*r**4*xi**6)
  f5 = -3/1280*(-512*a**6*xi**6 + 320*a**4*r**2*xi**6 - 64*a*r**5*xi**6
                + 20*r**6*xi**6 + 15*r**2*xi**2 - 15)/(a**4*r**4*xi**6)
  f6 = -3/1280*(-512*a**6*xi**6 + 320*a**4*r**2*xi**6 + 64*a*r**5*xi**6
                + 20*r**6*xi**6 + 15*r**2*xi**2 - 15)/(a**4*r**4*xi**6)
  f0_lt = (3/160)*(-2*a + r)**3*(16*a**3 + 24*a**2*r + 14*a*r**2
                                 + 5*r**3)/(a**4*r**4)
  return f0_lt, f1, f2, f3, f4, f5, f6


def _K1_coeffs(r, a, xi):
  f1 = (3/4096)*(-192*a**2*r**4*xi**6 + 96*a**2*r**2*xi**4 - 288*a**2*xi**2
                 + 8*r**6*xi**6 - 4*r**4*xi**4 - 30*r**2*xi**2
                 + 27)/(SQRT_PI*a**6*r**4*xi**7)
  f2 = -3/40960*(-3072*a**7*xi**6 - 1536*a**6*r*xi**6 + 1792*a**5*r**2*xi**6
                 + 384*a**5*xi**4 + 896*a**4*r**3*xi**6 + 576*a**4*r*xi**4
                 + 448*a**3*r**4*xi**6 + 256*a**3*r**2*xi**4 - 144*a**3*xi**2
                 - 800*a**2*r**5*xi**6 - 360*a**2*r*xi**2 + 80*a*r**6*xi**6
                 - 120*a*r**4*xi**4 - 60*a*r**2*xi**2 - 270*a + 40*r**7*xi**6
                 - 20*r**5*xi**4 - 150*r**3*xi**2
                 + 135*r)/(SQRT_PI*a**6*r**5*xi**7)
  f3 = -3/40960*(3072*a**7*xi**6 - 1536*a**6*r*xi**6 - 1792*a**5*r**2*xi**6
                 - 384*a**5*xi**4 + 896*a**4*r**3*xi**6 + 576*a**4*r*xi**4
                 - 448*a**3*r**4*xi**6 - 256*a**3*r**2*xi**4 + 144*a**3*xi**2
                 - 800*a**2*r**5*xi**6 - 360*a**2*r*xi**2 - 80*a*r**6*xi**6
                 + 120*a*r**4*xi**4 + 60*a*r**2*xi**2 + 270*a + 40*r**7*xi**6
                 - 20*r**5*xi**4 - 150*r**3*xi**2
                 + 135*r)/(SQRT_PI*a**6*r**5*xi**7)
  f4 = -3/8192*(-384*a**2*r**6*xi**8 - 288*a**2*r**2*xi**4 + 288*a**2*xi**2
                + 16*r**8*xi**8 - 72*r**4*xi**4 + 48*r**2*xi**2
                - 27)/(a**6*r**5*xi**8)
  f5 = (3/81920)*(12288*a**8*xi**8 - 10240*a**6*r**2*xi**8
                  + 4096*a**3*r**5*xi**8 - 1920*a**2*r**6*xi**8
                  - 1440*a**2*r**2*xi**4 + 1440*a**2*xi**2 + 80*r**8*xi**8
                  - 360*r**4*xi**4 + 240*r**2*xi**2 - 135)/(a**6*r**5*xi**8)
  f6 = (3/81920)*(12288*a**8*xi**8 - 10240*a**6*r**2*xi**8
                  - 4096*a**3*r**5*xi**8 - 1920*a**2*r**6*xi**8
                  - 1440*a**2*r**2*xi**4 + 1440*a**2*xi**2 + 80*r**8*xi**8
                  - 360*r**4*xi**4 + 240*r**2*xi**2 - 135)/(a**6*r**5*xi**8)
  f0_lt = -3/2560*(-2*a + r)**4*(48*a**4 + 96*a**3*r + 80*a**2*r**2
                                 + 40*a*r**3 + 5*r**4)/(a**6*r**5)
  return f0_lt, f1, f2, f3, f4, f5, f6


def _K2_coeffs(r, a, xi):
  f1 = -9/4096*(-320*a**2*r**4*xi**6 - 608*a**2*r**2*xi**4 - 480*a**2*xi**2
                + 56*r**6*xi**6 - 28*r**4*xi**4 + 78*r**2*xi**2
                + 45)/(SQRT_PI*a**6*r**4*xi**7)
  f2 = (9/8192)*(-1024*a**7*xi**6 - 512*a**6*r*xi**6 - 768*a**5*r**2*xi**6
                 + 128*a**5*xi**4 - 384*a**4*r**3*xi**6 + 192*a**4*r*xi**4
                 - 192*a**3*r**4*xi**6 + 256*a**3*r**2*xi**4 - 48*a**3*xi**2
                 - 96*a**2*r**5*xi**6 + 256*a**2*r**3*xi**4 - 120*a**2*r*xi**2
                 + 112*a*r**6*xi**6 - 168*a*r**4*xi**4 - 276*a*r**2*xi**2
                 - 90*a + 56*r**7*xi**6 - 28*r**5*xi**4 + 78*r**3*xi**2
                 + 45*r)/(SQRT_PI*a**6*r**5*xi**7)
  f3 = (9/8192)*(1024*a**7*xi**6 - 512*a**6*r*xi**6 + 768*a**5*r**2*xi**6
                 - 128*a**5*xi**4 - 384*a**4*r**3*xi**6 + 192*a**4*r*xi**4
                 + 192*a**3*r**4*xi**6 - 256*a**3*r**2*xi**4 + 48*a**3*xi**2
                 - 96*a**2*r**5*xi**6 + 256*a**2*r**3*xi**4 - 120*a**2*r*xi**2
                 - 112*a*r**6*xi**6 + 168*a*r**4*xi**4 + 276*a*r**2*xi**2
                 + 90*a + 56*r**7*xi**6 - 28*r**5*xi**4 + 78*r**3*xi**2
                 + 45*r)/(SQRT_PI*a**6*r**5*xi**7)
  f4 = (9/8192)*(-640*a**2*r**6*xi**8 + 288*a**2*r**2*xi**4 + 480*a**2*xi**2
                 + 112*r**8*xi**8 + 72*r**4*xi**4 - 48*r**2*xi**2
                 - 45)/(a**6*r**5*xi**8)
  f5 = -9/16384*(4096*a**8*xi**8 + 2048*a**6*r**2*xi**8 - 640*a**2*r**6*xi**8
                 + 288*a**2*r**2*xi**4 + 480*a**2*xi**2 + 112*r**8*xi**8
                 + 72*r**4*xi**4 - 48*r**2*xi**2 - 45)/(a**6*r**5*xi**8)
  # f6 == f5 exactly (verified; not a transcription artifact).
  f6 = f5
  f0_lt = (9/512)*(-2*a + r)**2*(2*a + r)**2*(16*a**4 + 16*a**2*r**2
                                              + 7*r**4)/(a**6*r**5)
  return f0_lt, f1, f2, f3, f4, f5, f6


def _K3_coeffs(r, a, xi):
  f1 = (9/4096)*(-64*a**2*r**4*xi**6 + 32*a**2*r**2*xi**4 + 480*a**2*xi**2
                 + 8*r**6*xi**6 - 4*r**4*xi**4 + 18*r**2*xi**2
                 - 45)/(SQRT_PI*a**6*r**4*xi**7)
  f2 = -9/8192*(1024*a**7*xi**6 + 512*a**6*r*xi**6 - 256*a**5*r**2*xi**6
                - 128*a**5*xi**4 - 128*a**4*r**3*xi**6 - 192*a**4*r*xi**4
                - 64*a**3*r**4*xi**6 - 128*a**3*r**2*xi**4 + 48*a**3*xi**2
                - 32*a**2*r**5*xi**6 - 64*a**2*r**3*xi**4 + 120*a**2*r*xi**2
                + 16*a*r**6*xi**6 - 24*a*r**4*xi**4 + 84*a*r**2*xi**2 + 90*a
                + 8*r**7*xi**6 - 4*r**5*xi**4 + 18*r**3*xi**2
                - 45*r)/(SQRT_PI*a**6*r**5*xi**7)
  f3 = -9/8192*(-1024*a**7*xi**6 + 512*a**6*r*xi**6 + 256*a**5*r**2*xi**6
                + 128*a**5*xi**4 - 128*a**4*r**3*xi**6 - 192*a**4*r*xi**4
                + 64*a**3*r**4*xi**6 + 128*a**3*r**2*xi**4 - 48*a**3*xi**2
                - 32*a**2*r**5*xi**6 - 64*a**2*r**3*xi**4 + 120*a**2*r*xi**2
                - 16*a*r**6*xi**6 + 24*a*r**4*xi**4 - 84*a*r**2*xi**2 - 90*a
                + 8*r**7*xi**6 - 4*r**5*xi**4 + 18*r**3*xi**2
                - 45*r)/(SQRT_PI*a**6*r**5*xi**7)
  f4 = -9/8192*(-128*a**2*r**6*xi**8 + 288*a**2*r**2*xi**4 - 480*a**2*xi**2
                + 16*r**8*xi**8 + 24*r**4*xi**4 - 48*r**2*xi**2
                + 45)/(a**6*r**5*xi**8)
  f5 = (9/16384)*(-4096*a**8*xi**8 + 2048*a**6*r**2*xi**8 - 128*a**2*r**6*xi**8
                  + 288*a**2*r**2*xi**4 - 480*a**2*xi**2 + 16*r**8*xi**8
                  + 24*r**4*xi**4 - 48*r**2*xi**2 + 45)/(a**6*r**5*xi**8)
  # f6 == f5 exactly (verified; not a transcription artifact).
  f6 = f5
  f0_lt = -9/512*(-2*a + r)**3*(2*a + r)**3*(4*a**2 + r**2)/(a**6*r**5)
  return f0_lt, f1, f2, f3, f4, f5, f6


# ----------------------------------------------------------------------------
# Public closed forms (mirror F1F2_closed_form)
# ----------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(1, 2, 3))
def G1G2_closed_form(
    r,
    a,
    xi,
    pair_eps_fraction_of_diameter: float = PAIR_EPS_FRACTION_OF_DIAMETER):
  """Return (G1, G2) for all r >= 0 (UC coupling scalars, monodisperse)."""
  if pair_eps_fraction_of_diameter <= 0.0:
    raise ValueError(
        "pair_eps_fraction_of_diameter must be positive for near-zero stability.")
  r = jnp.asarray(r)
  eps = jnp.asarray((2.0 * a) * pair_eps_fraction_of_diameter, dtype=r.dtype)
  r = jnp.maximum(r, eps)
  G1 = _skeleton(r, a, xi, *_G1_coeffs(r, a, xi))
  G2 = _skeleton(r, a, xi, *_G2_coeffs(r, a, xi))
  return G1, G2


@partial(jax.jit, static_argnums=(1, 2, 3))
def K1K2K3_closed_form(
    r,
    a,
    xi,
    pair_eps_fraction_of_diameter: float = PAIR_EPS_FRACTION_OF_DIAMETER):
  """Return (K1, K2, K3) for all r >= 0 (DC coupling scalars, monodisperse)."""
  if pair_eps_fraction_of_diameter <= 0.0:
    raise ValueError(
        "pair_eps_fraction_of_diameter must be positive for near-zero stability.")
  r = jnp.asarray(r)
  eps = jnp.asarray((2.0 * a) * pair_eps_fraction_of_diameter, dtype=r.dtype)
  r = jnp.maximum(r, eps)
  K1 = _skeleton(r, a, xi, *_K1_coeffs(r, a, xi))
  K2 = _skeleton(r, a, xi, *_K2_coeffs(r, a, xi))
  K3 = _skeleton(r, a, xi, *_K3_coeffs(r, a, xi))
  return K1, K2, K3


@partial(jax.jit, static_argnums=(1,))
def Mr_self_dipole(a, xi):
  """Eta-independent DC self-mobility scalar K1(r -> 0, xi).

  Identical to the FSD reference ``m_self.y``.  Multiply by
  ``1 / (6 pi eta)`` and contract with the K1 tensor structure to obtain
  the Cartesian DC self-mobility.  In the xi -> 0 (free-space) limit this
  evaluates to ``-3/(10 a^3)``, which reproduces the single-sphere strain
  mobility ``3/(20 pi eta a^3)`` and rotation mobility ``1/(8 pi eta a^3)``.
  """
  a2 = a * a
  a3 = a2 * a
  a6 = a3 * a3
  xi2 = xi * xi
  xi3 = xi2 * xi
  val = (-3.0 * (6.0 * a2 * xi2 + 1.0) / (80.0 * SQRT_PI * a6 * xi3)
         + 3.0 * (10.0 * a2 * xi2 + 1.0) / (80.0 * SQRT_PI * a6 * xi3)
         * jnp.exp(-4.0 * a2 * xi2)
         - 3.0 / (10.0 * a3) * erfc(2.0 * a * xi))
  return val
