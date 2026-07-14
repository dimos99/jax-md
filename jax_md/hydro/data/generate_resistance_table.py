"""Generate the production monodisperse resistance table from equations.

This generator is deliberately independent of Fiore's committed C++ table and
of Townsend's precomputed ``.npy`` files.  It evaluates the corrected
Jeffrey--Onishi near-contact expressions published by Townsend (2023), uses a
JAX port of Wilson's Lamb/reflection method in the midfield, subtracts the
two-body FTS far-field resistance, and shifts every scalar so the lubrication
correction vanishes at ``r/a = 4``.

The default output grid contains 2000 points, logarithmically spaced in the
surface gap ``xi = r/a - 2`` from exactly ``1e-8`` to ``2``.  All numerical
kernels use JAX in float64 and are jitted.  NumPy is used only for CLI I/O,
metadata, and comparison reporting.

The Wilson implementation is specialized to equal spheres whose line of
centers is the x axis.  Translation preserves azimuthal order in that geometry,
so only the physically populated modes ``m=-2,...,2`` are stored.  Apart from
that memory reduction, the initialization, translation, inversion, convergence
test, and readout are direct ports of Townsend's ``helen_fortran`` two-sphere
code (the implementation of Wilson 2013 used to build Townsend's midfield
table).

References
----------
* Townsend, Phys. Fluids 35, 127126 (2023), corrected near-field scalars.
* Wilson, J. Comput. Phys. 245, 302--316 (2013), Lamb/reflection method.
* Fiore and Swan, J. Fluid Mech. 878, 544--597 (2019), FSD table convention.

Run::

  python jax_md/hydro/data/generate_resistance_table.py --compare
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from typing import Dict, Tuple

import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


F64 = jnp.float64
PI = math.pi
N_WILSON = 160
WILSON_TOL = 1e-8
WILSON_MAX_ITERS = 10_000
XI_NEAR_END = 1e-2
XI_MID_START = 2e-2
N_ROWS_DEFAULT = 2000
XI_MIN_DEFAULT = 1e-8
XI_MAX_DEFAULT = 2.0

COLUMN_NAMES = (
    'XA11', 'XA12', 'YA11', 'YA12', 'YB11', 'YB12', 'XC11', 'XC12',
    'YC11', 'YC12', 'XG11', 'XG12', 'YG11', 'YG12', 'YH11', 'YH12',
    'XM11', 'XM12', 'YM11', 'YM12', 'ZM11', 'ZM12')

# Townsend/Kim--Karrila family units -> the common FSD 6*pi*eta*a^k units.
_FAMILY_CONV = {'A': 1.0, 'B': 2.0 / 3.0, 'C': 4.0 / 3.0,
                'G': 2.0 / 3.0, 'H': 4.0 / 3.0, 'M': 10.0 / 9.0}
CONV = jnp.asarray([_FAMILY_CONV[name[1]] for name in COLUMN_NAMES],
                   dtype=F64)


# ---------------------------------------------------------------------------
# Townsend (2023), lambda=1.
#
# Every corrected equal-sphere near-field scalar has the form
#   c_-1 / xi + c_log log(1/xi) + c_0 + c_1 xi log(1/xi).
# The constants below are the converged recurrence constants in Townsend's
# equations (series limits 100/130/200 as appropriate).  Keeping the constants
# here makes the runtime expression compact and JAX-vectorizable; the formulas
# themselves, not a table of values, are evaluated for every requested gap.
# ---------------------------------------------------------------------------
_TOWNSEND_COEFF = jnp.asarray([
    [ .25,  .225,  .99541916595484681,  .026785714285714284],
    [-.25, -.225, -.35015349115857464, -.026785714285714284],
    [0.,  1/6,  .99831711300020709, 0.],
    [0., -1/6, -.27365201265678574, 0.],
    [0., -.25,  .23891957435376154, -.125],
    [0.,  .25, -.0016226816604061195, .125],
    [0., 0., 1.0517997902646448, -.125],
    [0., 0., -.15025711289494925, .125],
    [0., .2,  .70283421759506626, .188],
    [0., .05, -.027463966276946954, .062],
    [.375,  .3375, -.46921067271694444,  .20892857142857144],
    [-.375, -.3375, .19550806425956466, -.20892857142857144],
    [0., .125, -.14115738329992006, .0625],
    [0., -.125, .10250059688271042, -.0625],
    [0., .025, -.074011572159512221, .0685],
    [0., .1, -.029290289606687538, .0565],
    [.15, .135, .71680011134251265, .12607142857142858],
    [.15, .135, -.14542415601786476, .17607142857142857],
    [0., .12, .88467122496831818, .0228],
    [0., .03, -.20758133009780641, .1332],
    [0., 0., 1.0250970014597653, -.075],
    [0., 0., -.07231065739084265, .075],
], dtype=F64)


@jax.jit
def townsend_exact_scalars(xi: jnp.ndarray) -> jnp.ndarray:
  """Corrected equal-sphere near-field resistance in K&K family units."""
  xi = jnp.asarray(xi, dtype=F64)
  log_inv = jnp.log(1.0 / xi)
  c = _TOWNSEND_COEFF
  return (c[:, 0] / xi[..., None]
          + c[:, 1] * log_inv[..., None]
          + c[:, 2]
          + c[:, 3] * (xi * log_inv)[..., None])


# ---------------------------------------------------------------------------
# Wilson reflection constants and kernels.
# ---------------------------------------------------------------------------
_M_VALUES = np.arange(-2, 3, dtype=np.int32)


def _wilson_constants(nmax: int):
  """Build Wilson's g, FracH, and H constants without tabulated physics."""
  g = np.zeros((5, nmax + 1, nmax + 1), dtype=np.float64)  # m, nu, n
  frac_h = np.zeros((5, nmax + 1, nmax + 1), dtype=np.float64)  # m,n,nu

  for nu in range(nmax + 1):
    for n in range(nmax + 1):
      vals = np.zeros(2 * n + 1, dtype=np.float64)
      vals[2 * n] = 1.0  # m=n
      for m in range(n - 1, 0, -1):
        vals[m + n] = (nu + m + 1.0) * vals[m + 1 + n] / (n - m)
      vals[0] = 1.0  # m=-n
      if n != 0:
        vals[1] = nu + n
        for m in range(-n + 1, -1):
          vals[m + 1 + n] = (nu - m) * vals[m + n] / (n + m + 1.0)
        # This is the numerically stable recurrence used by Wilson's Fill_G.
        if n == 1:
          vals[n] = nu + 1.0
        else:
          vals[n] = g[2, nu, n - 1] * (nu + n) / n
      else:
        vals[0] = 1.0
      for mi, m in enumerate(_M_VALUES):
        if abs(m) <= n:
          g[mi, nu, n] = vals[m + n]

  # Wilson declares root through 2*Nmax, but the final FracH recurrence reads
  # root[2*Nmax+1] when constants are prepared at the static safety ceiling.
  # Production NN is smaller than that ceiling; allocating the extra element
  # removes the latent Fortran bounds overrun without changing used values.
  root = np.sqrt(np.arange(2 * nmax + 2, dtype=np.float64))
  for mi, m in enumerate(_M_VALUES):
    im = abs(int(m))
    for n in range(im, nmax + 1):
      frac_h[mi, n, n] = 1.0
      for nu in range(n - 1, im - 1, -1):
        frac_h[mi, n, nu] = (
            root[2 * nu + 3] * root[nu - im + 1]
            * frac_h[mi, n, nu + 1]
            / (root[2 * nu + 1] * root[nu + im + 1]))
      for nu in range(n + 1, nmax + 1):
        frac_h[mi, n, nu] = (
            root[2 * nu - 1] * root[nu + im]
            * frac_h[mi, n, nu - 1]
            / (root[2 * nu + 1] * root[nu - im]))

  h = np.zeros((3, 5), dtype=np.float64)
  for n in (1, 2):
    for mi, m in enumerate(_M_VALUES):
      im = abs(int(m))
      if im <= n:
        # Wilson's ``fact`` array stores sqrt(n!), not n!.
        ratio = math.exp(.5 * (math.lgamma(n - im + 1)
                               - math.lgamma(n + im + 1)))
        h[n, mi] = math.sqrt(2 * n + 1) * ratio / (2 * math.sqrt(PI))
  return g, frac_h, h


_G_NP, _FRAC_H_NP, _H_NP = _wilson_constants(N_WILSON)
_G = jnp.asarray(_G_NP, dtype=F64)
_FRAC_H = jnp.asarray(_FRAC_H_NP, dtype=F64)
_H = jnp.asarray(_H_NP, dtype=F64)


def _case_loads():
  """The 14 canonical loads used by Townsend's midfield extraction."""
  f = np.zeros((14, 2, 3), dtype=np.float64)
  t = np.zeros_like(f)
  e = np.zeros((14, 2, 3, 3), dtype=np.float64)
  # F11,F12,F21,F22,T11,T12,T21,T22
  f[0, 0, 0] = 1.; f[1, 0, 1] = 1.
  f[2, 1, 0] = 1.; f[3, 1, 1] = 1.
  t[4, 0, 0] = 1.; t[5, 0, 1] = 1.
  t[6, 1, 0] = 1.; t[7, 1, 1] = 1.
  # E11,E13,E14,E21,E23,E24.  Wilson uses symmetric, traceless E.
  for case, sphere in ((8, 0), (11, 1)):
    e[case, sphere] = np.diag([1., -.5, -.5])
  for case, sphere in ((9, 0), (12, 1)):
    e[case, sphere, 1, 2] = e[case, sphere, 2, 1] = 1.
  for case, sphere in ((10, 0), (13, 1)):
    e[case, sphere, 0, 1] = e[case, sphere, 1, 0] = 1.
  return (jnp.asarray(f, dtype=F64), jnp.asarray(t, dtype=F64),
          jnp.asarray(e, dtype=F64))


_CASE_F, _CASE_T, _CASE_E = _case_loads()


def _initial_wilson_fields():
  """Return initial outgoing ABC and accumulated background DEF fields."""
  # (case, family, real/imag, sphere, n, m-index)
  shape = (14, 3, 2, 2, N_WILSON + 1, 5)
  abc = jnp.zeros(shape, dtype=F64)
  d0 = jnp.zeros(shape, dtype=F64)
  hm0, hm1, hm2 = _H[1, 2], _H[1, 3], _H[2, 4]

  # Force initial conditions: C, followed by A=C/6.
  c = jnp.zeros((14, 2, 2, N_WILSON + 1, 5), dtype=F64)
  c = c.at[:, 0, :, 1, 2].set(_CASE_F[:, :, 0] / (4 * PI * hm0))
  c = c.at[:, 0, :, 1, 3].set(-_CASE_F[:, :, 1] / (8 * PI * hm1))
  c = c.at[:, 0, :, 1, 1].set(_CASE_F[:, :, 1] / (8 * PI * hm1))
  c = c.at[:, 1, :, 1, 3].set(-_CASE_F[:, :, 2] / (8 * PI * hm1))
  c = c.at[:, 1, :, 1, 1].set(-_CASE_F[:, :, 2] / (8 * PI * hm1))
  abc = abc.at[:, 2].set(c)
  abc = abc.at[:, 0].set(c / 6.0)

  # Torque initial conditions: B.
  b = jnp.zeros_like(c)
  b = b.at[:, 0, :, 1, 2].set(_CASE_T[:, :, 0] / (8 * PI * hm0))
  b = b.at[:, 0, :, 1, 3].set(-_CASE_T[:, :, 1] / (16 * PI * hm1))
  b = b.at[:, 0, :, 1, 1].set(_CASE_T[:, :, 1] / (16 * PI * hm1))
  b = b.at[:, 1, :, 1, 3].set(-_CASE_T[:, :, 2] / (16 * PI * hm1))
  b = b.at[:, 1, :, 1, 1].set(-_CASE_T[:, :, 2] / (16 * PI * hm1))
  abc = abc.at[:, 1].set(b)

  # Background strain is an incoming D field.  xg_sign=-1 matches the
  # corrected Townsend Fortran source.  The final factor 1/2 is its G=2D fix.
  xg_sign = -1.0
  d = jnp.zeros_like(c)
  d = d.at[:, 0, :, 2, 2].set(
      .5 * xg_sign * _CASE_E[:, :, 0, 0] / _H[2, 2])
  d = d.at[:, 0, :, 2, 3].set(
      .5 * _CASE_E[:, :, 0, 1] / (3 * _H[2, 3]))
  d = d.at[:, 0, :, 2, 1].set(
      -.5 * _CASE_E[:, :, 0, 1] / (3 * _H[2, 1]))
  d = d.at[:, 1, :, 2, 3].set(
      .5 * _CASE_E[:, :, 0, 2] / (3 * _H[2, 3]))
  d = d.at[:, 1, :, 2, 1].set(
      .5 * _CASE_E[:, :, 0, 2] / (3 * _H[2, 1]))
  diag2 = 2 * _CASE_E[:, :, 1, 1] + _CASE_E[:, :, 0, 0]
  d = d.at[:, 0, :, 2, 4].set(
      .5 * xg_sign * diag2 / (12 * hm2))
  d = d.at[:, 0, :, 2, 0].set(
      .5 * xg_sign * diag2 / (12 * _H[2, 0]))
  d = d.at[:, 1, :, 2, 4].set(
      -.5 * _CASE_E[:, :, 1, 2] / (6 * hm2))
  d = d.at[:, 1, :, 2, 0].set(
      .5 * _CASE_E[:, :, 1, 2] / (6 * _H[2, 0]))
  d0 = d0.at[:, 0].set(d)
  return abc, d0


_INITIAL_ABC, _INITIAL_DEF = _initial_wilson_fields()


@jax.jit
def _invert_field(incoming: jnp.ndarray, n_active: jnp.ndarray) -> jnp.ndarray:
  """Wilson boundary inversion: incoming DEF -> outgoing ABC."""
  d, e, f = incoming[:, 0], incoming[:, 1], incoming[:, 2]
  n = jnp.arange(N_WILSON + 1, dtype=F64)[None, None, None, :, None]
  m = jnp.asarray(np.abs(_M_VALUES))[None, None, None, None, :]
  valid = (n >= jnp.maximum(2, m)) & (n <= n_active)
  safe_n = jnp.maximum(n, 1.0)
  a = -safe_n / (4 * (safe_n + 1) * (2 * safe_n + 3)) * (
      2 * (2 * safe_n - 1) * (2 * safe_n + 3) * d
      + (2 * safe_n + 1) * f)
  b = -e
  c = -safe_n * (2 * safe_n - 1) / (2 * (safe_n + 1)) * (
      2 * (2 * safe_n + 1) * d + f)
  a = jnp.where(valid, a, 0.0)
  b = jnp.where(valid, b, 0.0)
  c = jnp.where(valid, c, 0.0)
  # Wilson's special n=1 inversion.
  valid1 = (n == 1) & (m <= 1) & (n <= n_active)
  a = jnp.where(valid1, -f / 30.0, a)
  return jnp.stack([a, b, c], axis=1)


def _translation_matrices(s: jnp.ndarray, direction: float,
                          n_active: jnp.ndarray):
  """Matrices in Wilson's collinear translation theorem."""
  n = jnp.arange(N_WILSON + 1, dtype=F64)[None, None, :]
  nu = jnp.arange(N_WILSON + 1, dtype=F64)[None, :, None]
  m = jnp.asarray(_M_VALUES, dtype=F64)[:, None, None]
  im = jnp.abs(m)
  valid = ((n >= jnp.maximum(1, im)) & (nu >= jnp.maximum(1, im))
           & (n <= n_active) & (nu <= n_active))
  ns = jnp.maximum(n, 1.0)
  nus = jnp.maximum(nu, 1.0)
  sign = -1.0 if direction > 0 else 1.0
  signed_r = direction * s
  parity = jnp.where((jnp.asarray(_M_VALUES)[:, None, None]
                      + jnp.arange(N_WILSON + 1)[None, None, :]) % 2 == 0,
                     1.0, -1.0)
  dir_sign = jnp.where((jnp.arange(N_WILSON + 1)[None, :, None]
                        + jnp.arange(N_WILSON + 1)[None, None, :]) % 2 == 0,
                       1.0, sign)
  base = (parity * dir_sign * jnp.swapaxes(_FRAC_H, 1, 2)
          / s ** (n + nu + 1))
  g = _G
  gprev = jnp.pad(g[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
  cterm = (gprev * (nu - im) * ((nu - 1) * (n - 2) - (n + 1))
           / (nus * (2 * nus - 1)) - g * (n - 2) / 2.0)
  mats = {
      'da': base * g,
      'db': base * (m * g * signed_r / nus),
      'dc': base * cterm * s ** 2 / (ns * (2 * ns - 1)),
      'eb': base * g * (-ns / (1 + nus)),
      'ec': base * g * (m * signed_r / ((1 + nus) * ns * nus)),
      'fc': base * g,
  }
  return {k: jnp.where(valid, v, 0.0) for k, v in mats.items()}


def _translate_one(source: jnp.ndarray, mats: Dict[str, jnp.ndarray]):
  """Translate one sphere's outgoing ABC field to incoming DEF."""
  a, b, c = source[:, 0], source[:, 1], source[:, 2]  # case,p,n,m
  # Matrices are (m,nu,n); fields are (case,p,n,m).
  def mv(mat, x):
    return jnp.einsum('mun,bpnm->bpum', mat, x, precision='highest')
  rot_b = jnp.stack([b[:, 1], -b[:, 0]], axis=1)
  rot_c = jnp.stack([c[:, 1], -c[:, 0]], axis=1)
  d = mv(mats['da'], a) + mv(mats['db'], rot_b) + mv(mats['dc'], c)
  e = mv(mats['eb'], b) + mv(mats['ec'], rot_c)
  f = mv(mats['fc'], c)
  return jnp.stack([d, e, f], axis=1)


def _translate_both(abc: jnp.ndarray, s: jnp.ndarray,
                    n_active: jnp.ndarray) -> jnp.ndarray:
  incoming = jnp.zeros_like(abc)
  # r(1,2)=+s: source sphere 0 -> target sphere 1.
  d1 = _translate_one(abc[:, :, :, 0],
                      _translation_matrices(s, 1.0, n_active))
  # r(2,1)=-s: source sphere 1 -> target sphere 0.
  d0 = _translate_one(abc[:, :, :, 1],
                      _translation_matrices(s, -1.0, n_active))
  incoming = incoming.at[:, :, :, 1].set(d1)
  incoming = incoming.at[:, :, :, 0].set(d0)
  return incoming


def _find_force(abc: jnp.ndarray, incoming: jnp.ndarray) -> jnp.ndarray:
  """Wilson's reflection convergence norm, one value per canonical case."""
  a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
  d, e = incoming[:, 0], incoming[:, 1]
  # n=1, m=-1,0,1 (stored at indices 1,2,3).
  v = _H[1][None, None, None, :] * (
      -5 * a[:, :, :, 1, :] + 1.5 * c[:, :, :, 1, :]
      + d[:, :, :, 1, :])
  om = _H[1][None, None, None, :] * (
      b[:, :, :, 1, :] + e[:, :, :, 1, :])
  # Only m=-1,0,1 contribute to rigid velocity/rotation.
  vr, vi = v[:, 0], v[:, 1]
  or_, oi = om[:, 0], om[:, 1]
  vel = jnp.stack([
      vr[..., 2], vr[..., 1] - vr[..., 3], -vi[..., 3] - vi[..., 1],
      vi[..., 2], vi[..., 1] - vi[..., 3], vr[..., 3] + vr[..., 1]], -1)
  omg = jnp.stack([
      or_[..., 2], or_[..., 1] - or_[..., 3], -oi[..., 3] - oi[..., 1],
      oi[..., 2], oi[..., 1] - oi[..., 3], or_[..., 3] + or_[..., 1]], -1)
  per_sphere = 6 * PI * (jnp.linalg.norm(vel, axis=-1)
                         + jnp.linalg.norm(omg, axis=-1))
  return jnp.sum(per_sphere, axis=-1)


def _readout_wilson(abc: jnp.ndarray, deff: jnp.ndarray) -> jnp.ndarray:
  """Read Wilson coefficient sums as the 22-value Fortran short output."""
  a, b, c = abc[:, 0], abc[:, 1], abc[:, 2]
  d, e, f = deff[:, 0], deff[:, 1], deff[:, 2]
  v = _H[1][None, None, None, :] * (
      -5 * a[:, :, :, 1, :] + 1.5 * c[:, :, :, 1, :]
      + d[:, :, :, 1, :])
  om = _H[1][None, None, None, :] * (
      b[:, :, :, 1, :] + e[:, :, :, 1, :])
  sst = (_H[2][None, None, None, :] * (4 * PI / 15) * (
      20 * d[:, :, :, 2, :] - 3 * c[:, :, :, 2, :]
      + 2 * f[:, :, :, 2, :]))
  vr, vi = v[:, 0], v[:, 1]
  or_, oi = om[:, 0], om[:, 1]
  sr, si = sst[:, 0], sst[:, 1]
  ux = vr[..., 2]; uy = vr[..., 1] - vr[..., 3]
  uz = -vi[..., 3] - vi[..., 1]
  ox = or_[..., 2]; oy = or_[..., 1] - or_[..., 3]
  oz = -oi[..., 3] - oi[..., 1]
  xg_sign = -1.0
  sxx = sr[..., 2] * xg_sign
  syy = (-.5 * sr[..., 2] + 3 * (sr[..., 0] + sr[..., 4])) * xg_sign
  sxy = 1.5 * (sr[..., 3] - sr[..., 1])
  sxz = 1.5 * (si[..., 3] + si[..., 1])
  syz = 3 * (si[..., 0] - si[..., 4])
  particle = jnp.stack([ux, uy, uz, ox, oy, oz,
                        sxx, syy, sxy, sxz, syz], axis=-1)
  return particle.reshape((14, 22))


def _reorder_matrix():
  rp = .5 * (1 + math.sqrt(3)); rm = .5 * (-1 + math.sqrt(3))
  r2 = math.sqrt(2)
  q = np.zeros((22, 22), dtype=np.float64)
  for out, inp in enumerate((0, 1, 2, 11, 12, 13, 3, 4, 5,
                             14, 15, 16)):
    q[out, inp] = 1.
  q[12, 6] = rp; q[12, 7] = rm; q[13, 8] = r2
  q[14, 6] = rm; q[14, 7] = rp; q[15, 9] = r2; q[16, 10] = r2
  q[17, 17] = rp; q[17, 18] = rm; q[18, 19] = r2
  q[19, 17] = rm; q[19, 18] = rp; q[20, 20] = r2; q[21, 21] = r2
  return jnp.asarray(q, dtype=F64)


_REORDER = _reorder_matrix()


def _mobility_scalars(short_output: jnp.ndarray) -> jnp.ndarray:
  """Townsend's extraction of 44 K&K mobility scalars."""
  u = short_output @ _REORDER.T
  rfac = (3 + jnp.sqrt(3.0)) / 6; sfac = jnp.sqrt(2.0)
  mfac = (3 + jnp.sqrt(3.0)) / 4
  vals = (
      u[0, 0], u[2, 0], u[0, 3], u[2, 3],
      u[1, 1], u[3, 1], u[1, 4], u[3, 4],
      -u[1, 8], -u[3, 8], -u[1, 11], -u[3, 11],
      u[4, 6], u[6, 6], u[4, 9], u[6, 9],
      u[5, 7], u[7, 7], u[5, 10], u[7, 10],
      u[0, 12] / rfac, u[2, 12] / rfac,
      u[0, 17] / rfac, u[2, 17] / rfac,
      u[1, 13] / sfac, u[3, 13] / sfac,
      u[1, 18] / sfac, u[3, 18] / sfac,
      u[5, 15] / (-sfac), u[7, 15] / (-sfac),
      u[5, 20] / (-sfac), u[7, 20] / (-sfac),
      u[8, 12] / mfac, u[11, 12] / mfac,
      u[8, 17] / mfac, u[11, 17] / mfac,
      u[10, 13] / sfac, u[13, 13] / sfac,
      u[10, 18] / sfac, u[13, 18] / sfac,
      u[9, 16] / sfac, u[12, 16] / sfac,
      u[9, 21] / sfac, u[12, 21] / sfac)
  return jnp.stack(vals)


def _mobility_to_resistance(x: jnp.ndarray) -> jnp.ndarray:
  """Kim mobility scalars -> Kim resistance scalars (Townsend A.1.2/3)."""
  (x11a,x12a,x21a,x22a, y11a,y12a,y21a,y22a,
   y11b,y12b,y21b,y22b, x11c,x12c,x21c,x22c,
   y11c,y12c,y21c,y22c, x11g,x12g,x21g,x22g,
   y11g,y12g,y21g,y22g, y11h,y12h,y21h,y22h,
   x11m,x12m,x21m,x22m, y11m,y12m,y21m,y22m,
   z11m,z12m,z21m,z22m) = tuple(x)
  xa = jnp.linalg.inv(jnp.array([[x11a,x12a],[x21a,x22a]]))
  xc = jnp.linalg.inv(jnp.array([[x11c,x12c],[x21c,x22c]]))
  xg0 = jnp.array([[x11g,x12g],[x21g,x22g]])
  xg = xg0 @ xa
  yabc = jnp.linalg.inv(jnp.array([
      [y11a,y12a,y11b,y21b], [y12a,y22a,y12b,y22b],
      [y11b,y12b,y11c,y12c], [y21b,y22b,y12c,y22c]]))
  ygh = yabc @ jnp.array([[y11g,y21g],[y12g,y22g],
                           [-y11h,-y21h],[-y12h,-y22h]])
  # Townsend names the second matrix index first for the intermediate G/H
  # solve; transpose into the conventional [[11,12],[21,22]] layout.
  yg = ygh[:2].T; yh = (-ygh[2:]).T
  xm = jnp.array([[x11m,x12m],[x21m,x22m]]) + 2/3 * xg @ xg0.T
  ym = (jnp.array([[y11m,y12m],[y21m,y22m]])
        + 2 * (yg @ jnp.array([[y11g,y21g],[y12g,y22g]])
               + yh @ jnp.array([[y11h,y21h],[y12h,y22h]])))
  return jnp.stack([
      xa[0,0],xa[0,1],xa[1,0],xa[1,1],
      yabc[0,0],yabc[0,1],yabc[1,0],yabc[1,1],
      yabc[0,2],yabc[1,2],yabc[0,3],yabc[1,3],
      xc[0,0],xc[0,1],xc[1,0],xc[1,1],
      yabc[2,2],yabc[2,3],yabc[3,2],yabc[3,3],
      xg[0,0],xg[0,1],xg[1,0],xg[1,1],
      yg[0,0],yg[0,1],yg[1,0],yg[1,1],
      yh[0,0],yh[0,1],yh[1,0],yh[1,1],
      xm[0,0],xm[0,1],xm[1,0],xm[1,1],
      ym[0,0],ym[0,1],ym[1,0],ym[1,1],
      z11m,z12m,z21m,z22m])


_RESISTANCE_SCALES = jnp.asarray(
    [6*PI]*8 + [4*PI]*4 + [8*PI]*8 + [4*PI]*8 + [8*PI]*4
    + [20/3*PI]*12, dtype=F64)
_GENERAL_22 = jnp.asarray([0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,
                           32,33,36,37,40,41])


@jax.jit
def wilson_short_output(
    s_dash: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Raw Wilson outputs for the 14 canonical cases and reflection counts."""
  s_dash = jnp.asarray(s_dash, dtype=F64)
  q = .5 * (s_dash - jnp.sqrt(s_dash * s_dash - 4))
  n_active = jnp.minimum(
      N_WILSON, jnp.floor(jnp.log(WILSON_TOL) / jnp.log(q)).astype(jnp.int32) + 1)
  initial_from_e = _invert_field(_INITIAL_DEF, n_active)
  current0 = _INITIAL_ABC + initial_from_e
  total0 = jnp.zeros_like(current0)
  def0 = _INITIAL_DEF
  active0 = jnp.ones((14,), dtype=bool)
  counts0 = jnp.zeros((14,), dtype=jnp.int32)

  def cond(carry):
    it, _cur, _tot, _defs, active, _counts = carry
    return (it < WILSON_MAX_ITERS) & jnp.any(active)

  def body(carry):
    it, cur, tot, defs, active, counts = carry
    incoming = _translate_both(cur, s_dash, n_active)
    nxt = _invert_field(incoming, n_active)
    eps = _find_force(nxt, incoming)
    mask = active.reshape((14, 1, 1, 1, 1, 1))
    tot = jnp.where(mask, tot + cur, tot)
    defs = jnp.where(mask, defs + incoming, defs)
    cur = jnp.where(mask, nxt, cur)
    counts = counts + active.astype(jnp.int32)
    still = ((it + 1) < 6) | (eps > WILSON_TOL)
    active = active & still
    return it + 1, cur, tot, defs, active, counts

  it, cur, total, defs, active, counts = jax.lax.while_loop(
      cond, body, (jnp.int32(0), current0, total0, def0, active0, counts0))
  del it
  # NaNs make a nonconverged case fail validation instead of silently writing.
  short = _readout_wilson(total + cur, defs)
  short = jnp.where(jnp.any(active), jnp.nan, short)
  return short, counts


@jax.jit
def wilson_exact_scalars(s_dash: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Wilson exact resistance scalars and per-case reflection counts."""
  short, counts = wilson_short_output(s_dash)
  mobility = _mobility_scalars(short)
  dimensional_r = _mobility_to_resistance(mobility)
  nondim_r = dimensional_r / _RESISTANCE_SCALES
  return nondim_r[_GENERAL_22], counts


# ---------------------------------------------------------------------------
# Two-sphere far-field FTS grand mobility and resistance subtraction.
# These are the tensor formulae used by Townsend's generate_Minfinity, reduced
# to two equal spheres.  The 5-vector condensation convention is kept exactly
# so the extraction stencil below is directly comparable with that reference.
# ---------------------------------------------------------------------------
_DELTA = jnp.eye(3, dtype=F64)
_LEVI = jnp.asarray([[[0, 0, 0], [0, 0, 1], [0, -1, 0]],
                     [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
                     [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]], dtype=F64)
_COND_IDX = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2))
_S3 = math.sqrt(3); _S2 = math.sqrt(2)
_COND_E = jnp.asarray([
    [(_S3 + 1) / 2, 0, (_S3 - 1) / 2, 0, 0],
    [0, _S2, 0, 0, 0],
    [(_S3 - 1) / 2, 0, (_S3 + 1) / 2, 0, 0],
    [0, 0, 0, _S2, 0],
    [0, 0, 0, 0, _S2]], dtype=F64)


def _j_tensor(r, s, i, j):
  return _DELTA[i, j] / s + r[i] * r[j] / s ** 3


def _dj(r, s, l, i, j):
  return ((-_DELTA[i,j]*r[l] + _DELTA[i,l]*r[j] + _DELTA[j,l]*r[i]) / s**3
          - 3*r[i]*r[j]*r[l]/s**5)


def _ddj(r, s, m, l, i, j):
  return ((-_DELTA[i,j]*_DELTA[l,m] + _DELTA[i,l]*_DELTA[j,m]
           + _DELTA[j,l]*_DELTA[i,m]) / s**3
          - 3*(-_DELTA[i,j]*r[l]*r[m] + _DELTA[i,l]*r[j]*r[m]
               + _DELTA[j,l]*r[i]*r[m] + _DELTA[i,m]*r[j]*r[l]
               + r[i]*_DELTA[j,m]*r[l] + r[i]*r[j]*_DELTA[l,m]) / s**5
          + 15*r[i]*r[j]*r[l]*r[m]/s**7)


def _rotlet(r, s, i, j):
  return -.5 * sum(_LEVI[j,k,l] * _dj(r,s,k,i,l)
                   for k in range(3) for l in range(3))


def _k_tensor(r, s, i, j, k):
  return .5 * (_dj(r,s,k,i,j) + _dj(r,s,j,i,k))


def _dr(r, s, l, i, j):
  return -.5 * sum(_LEVI[j,m,n] * _ddj(r,s,l,m,i,n)
                   for m in range(3) for n in range(3))


def _dk(r, s, l, i, j, k):
  return .5 * (_ddj(r,s,l,k,i,j) + _ddj(r,s,l,j,i,k))


def _lap_j(r, s, i, j):
  return 2*_DELTA[i,j]/s**3 - 6*r[i]*r[j]/s**5


def _dlap_j(r, s, k, i, j):
  return (-6/s**5)*(_DELTA[i,j]*r[k] + _DELTA[i,k]*r[j]
                    + _DELTA[j,k]*r[i]) + 30*r[i]*r[j]*r[k]/s**7


def _lap_r(r, s, i, j):
  return -.5 * sum(_LEVI[j,k,l] * _dlap_j(r,s,k,i,l)
                   for k in range(3) for l in range(3))


def _lap_k(r, s, i, j, k):
  return _dlap_j(r, s, i, j, k)


def _dlap_k(r, s, l, i, j, k):
  return ((-6/s**5)*(_DELTA[i,j]*_DELTA[k,l]
                     + _DELTA[i,k]*_DELTA[j,l]
                     + _DELTA[j,k]*_DELTA[i,l])
          - 210*r[i]*r[j]*r[k]*r[l]/s**9
          + (30/s**7)*(_DELTA[i,j]*r[k]*r[l]
                       + _DELTA[i,k]*r[j]*r[l]
                       + _DELTA[j,k]*r[i]*r[l]
                       + _DELTA[i,l]*r[j]*r[k]
                       + _DELTA[j,l]*r[i]*r[k]
                       + _DELTA[k,l]*r[i]*r[j]))


def _m11(r, s, i, j, c):
  return c * (_j_tensor(r,s,i,j) + (1/3)*_lap_j(r,s,i,j))


def _m12(r, s, i, j, c):
  return c * (_rotlet(r,s,i,j) + (1/6)*_lap_r(r,s,i,j))


def _m13(r, s, i, j, k, c):
  return -c * (_k_tensor(r,s,i,j,k) + (4/15)*_lap_k(r,s,i,j,k))


def _m22(r, s, i, j, c):
  return .5*c*sum(_LEVI[i,k,l] * _dr(r,s,k,l,j)
                  for k in range(3) for l in range(3))


def _m23(r, s, i, j, k, c):
  return -.5*c*sum(_LEVI[i,l,m] * (
      _dk(r,s,l,m,j,k) + (1/6)*_dlap_k(r,s,l,m,j,k))
                     for l in range(3) for m in range(3))


def _m33(r, s, i, j, k, l, c):
  return -.5*c*((_dk(r,s,j,i,k,l) + _dk(r,s,i,j,k,l))
                 + .2*(_dlap_k(r,s,j,i,k,l)
                       + _dlap_k(r,s,i,j,k,l)))


def _con_m13_row(r, s, i, c):
  a = _m13(r,s,i,0,0,c); b = _m13(r,s,i,1,1,c)
  hp=(_S3+1)/2; hm=(_S3-1)/2
  return jnp.stack([hp*a+hm*b, _S2*_m13(r,s,i,0,1,c),
                    hm*a+hp*b, _S2*_m13(r,s,i,0,2,c),
                    _S2*_m13(r,s,i,1,2,c)])


def _con_m23_row(r, s, i, c):
  a = _m23(r,s,i,0,0,c); b = _m23(r,s,i,1,1,c)
  hp=(_S3+1)/2; hm=(_S3-1)/2
  return jnp.stack([hp*a+hm*b, _S2*_m23(r,s,i,0,1,c),
                    hm*a+hp*b, _S2*_m23(r,s,i,0,2,c),
                    _S2*_m23(r,s,i,1,2,c)])


@jax.jit
def farfield_grand_mobility(s_dash: jnp.ndarray) -> jnp.ndarray:
  """Unbounded two-sphere FTS grand mobility in Townsend's 11N basis."""
  # Townsend deliberately clamps the far-field pair operator below 2.001.
  s = jnp.maximum(jnp.asarray(s_dash, dtype=F64), 2.001)
  r = jnp.array([-s, 0., 0.], dtype=F64)
  c0 = 1/(8*PI)
  mat = jnp.zeros((22, 22), dtype=F64)

  # Single-sphere blocks.
  self_a = jnp.eye(3, dtype=F64)/(6*PI)
  self_c = jnp.eye(3, dtype=F64)/(8*PI)
  self_m = jnp.eye(5, dtype=F64)/(20/3*PI)
  for particle in range(2):
    u = 3*particle; o = 6+3*particle; e = 12+5*particle
    mat = mat.at[u:u+3,u:u+3].set(self_a)
    mat = mat.at[o:o+3,o:o+3].set(self_c)
    mat = mat.at[e:e+5,e:e+5].set(self_m)

  a = jnp.stack([jnp.stack([_m11(r,s,i,j,c0) for j in range(3)])
                 for i in range(3)])
  bt = jnp.stack([jnp.stack([_m12(r,s,i,j,c0) for j in range(3)])
                  for i in range(3)])
  cc = jnp.stack([jnp.stack([_m22(r,s,i,j,c0) for j in range(3)])
                  for i in range(3)])
  gt = jnp.stack([_con_m13_row(r,s,i,c0) for i in range(3)])
  ht = jnp.stack([_con_m23_row(r,s,i,c0) for i in range(3)])
  mraw = jnp.stack([jnp.stack([
      _m33(r,s,*_COND_IDX[i],*_COND_IDX[j],c0) for j in range(5)])
                     for i in range(5)])
  mm = _COND_E @ mraw @ _COND_E

  # Cross blocks are placed in the upper triangle exactly as in
  # generate_Minfinity_loop; symmetry fills the reciprocal blocks.
  mat = mat.at[0:3,3:6].set(a)
  mat = mat.at[0:3,9:12].set(bt)
  mat = mat.at[3:6,6:9].set(-bt)
  mat = mat.at[6:9,9:12].set(cc)
  mat = mat.at[0:3,17:22].set(gt)
  mat = mat.at[3:6,12:17].set(-gt)
  mat = mat.at[6:9,17:22].set(ht)
  mat = mat.at[9:12,12:17].set(ht)
  mat = mat.at[12:17,17:22].set(mm)
  upper = jnp.triu(mat)
  return upper + jnp.triu(mat, 1).T


@jax.jit
def farfield_resistance_scalars(s_dash: jnp.ndarray) -> jnp.ndarray:
  """22 two-body FTS resistance scalars in K&K family units."""
  r = jnp.linalg.inv(farfield_grand_mobility(s_dash))
  sa=6*PI; sb=4*PI; sc=8*PI; sg=4*PI; sh=8*PI; sm=20/3*PI
  rf=(math.sqrt(3)+3)/6; sf=math.sqrt(2)
  zm11=r[16,16]/sm; zm12=r[16,21]/sm
  return jnp.stack([
      r[0,0]/sa, r[0,3]/sa, r[1,1]/sa, r[1,4]/sa,
      r[2,7]/sb, r[5,7]/sb, r[6,6]/sc, r[6,9]/sc,
      r[7,7]/sc, r[7,10]/sc,
      r[0,12]/rf/sg, r[3,12]/rf/sg,
      r[1,13]/sf/sg, r[4,13]/sf/sg,
      r[7,15]/(-sf)/sh, r[10,15]/(-sf)/sh,
      zm11-4*r[14,12]/sm, zm12-4*r[14,17]/sm,
      r[13,13]/sm, r[13,18]/sm, zm11, zm12])


# ---------------------------------------------------------------------------
# End-to-end residual, handoff, generation, and validation.
# ---------------------------------------------------------------------------
@jax.jit
def nearfield_residual(xi: jnp.ndarray) -> jnp.ndarray:
  """Townsend exact minus two-body FTS resistance, in FSD units."""
  return ((townsend_exact_scalars(xi)
           - farfield_resistance_scalars(2.0 + xi)) * CONV)


@jax.jit
def midfield_residual(
    xi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Wilson exact minus two-body FTS resistance, in FSD units."""
  exact, counts = wilson_exact_scalars(2.0 + xi)
  return ((exact - farfield_resistance_scalars(2.0 + xi)) * CONV, counts)


@jax.jit
def _handoff_data():
  """Endpoint values and limited log-slopes for the 0.01--0.02 bridge."""
  h = jnp.asarray(1e-3, dtype=F64)
  x0 = jnp.asarray(XI_NEAR_END, dtype=F64)
  x1 = jnp.asarray(XI_MID_START, dtype=F64)
  y0 = nearfield_residual(x0)
  ym = nearfield_residual(x0 * jnp.exp(-h))
  y1, _ = midfield_residual(x1)
  yp, _ = midfield_residual(x1 * jnp.exp(h))
  m0 = (y0 - ym) / h
  m1 = (yp - y1) / h
  length = jnp.log(x1 / x0)
  delta = (y1 - y0) / length

  # Fritsch--Carlson limiting prevents a cubic bridge from overshooting either
  # endpoint, including for signed cross scalars.
  safe = jnp.abs(delta) > 1e-14
  alpha = jnp.where(safe, m0 / delta, 0.0)
  beta = jnp.where(safe, m1 / delta, 0.0)
  alpha = jnp.maximum(alpha, 0.0)
  beta = jnp.maximum(beta, 0.0)
  radius = jnp.sqrt(alpha * alpha + beta * beta)
  tau = jnp.where(radius > 3.0, 3.0 / radius, 1.0)
  m0 = jnp.where(safe, tau * alpha * delta, 0.0)
  m1 = jnp.where(safe, tau * beta * delta, 0.0)
  return y0, y1, m0, m1


@jax.jit
def handoff_residual(xi: jnp.ndarray) -> jnp.ndarray:
  """C1 monotone Hermite bridge between Townsend and Wilson residuals."""
  y0, y1, m0, m1 = _handoff_data()
  length = jnp.log(XI_MID_START / XI_NEAR_END)
  t = jnp.log(xi / XI_NEAR_END) / length
  t2 = t*t; t3 = t2*t
  return ((2*t3 - 3*t2 + 1)*y0 + (t3 - 2*t2 + t)*length*m0
          + (-2*t3 + 3*t2)*y1 + (t3 - t2)*length*m1)


def _sequential_jit_map(fn, values: np.ndarray, batch_size: int):
  """Evaluate an already-jitted scalar kernel in bounded host-side chunks.

  Wrapping Wilson's large dynamic reflection loop in an additional XLA
  ``lax.map`` makes compilation much more expensive than the calculation.
  Scalar dispatch reuses one compiled executable and keeps memory constant.
  """
  outputs = []
  n = len(values)
  for start in range(0, n, batch_size):
    chunk = values[start:start + batch_size]
    for value in chunk:
      result = fn(jnp.asarray(value, dtype=F64))
      if isinstance(result, tuple):
        outputs.append(tuple(np.asarray(x) for x in result))
      else:
        outputs.append(np.asarray(result))
  if not outputs:
    return np.empty((0, 22), dtype=np.float64)
  if isinstance(outputs[0], tuple):
    return tuple(np.stack([x[i] for x in outputs], axis=0)
                 for i in range(len(outputs[0])))
  return np.stack(outputs, axis=0)


def generate_table(rows: int = N_ROWS_DEFAULT,
                   xi_min: float = XI_MIN_DEFAULT,
                   xi_max: float = XI_MAX_DEFAULT,
                   batch_size: int = 8):
  """Generate unshifted/shifted residuals and Wilson convergence diagnostics."""
  if rows < 3:
    raise ValueError('rows must be at least 3')
  if not (0 < xi_min < XI_NEAR_END < XI_MID_START < xi_max):
    raise ValueError(
        'require 0 < xi_min < 0.01 < 0.02 < xi_max; got '
        f'{xi_min:g}, {xi_max:g}')
  if batch_size < 1:
    raise ValueError('batch_size must be positive')
  xi = np.geomspace(xi_min, xi_max, rows, dtype=np.float64)
  # Pin endpoints exactly instead of accepting exp/log construction roundoff.
  xi[0] = xi_min; xi[-1] = xi_max
  raw = np.empty((rows, 22), dtype=np.float64)
  counts = np.zeros((rows, 14), dtype=np.int32)
  near = xi <= XI_NEAR_END
  bridge = (xi > XI_NEAR_END) & (xi < XI_MID_START)
  mid = xi >= XI_MID_START

  # Townsend evaluation is natively vectorized.  The FTS operator is clamped
  # below xi=0.001, so reuse that matrix rather than inverting it ~1000 times.
  xi_near = xi[near]
  exact_near = np.asarray(townsend_exact_scalars(
      jnp.asarray(xi_near, dtype=F64)))
  ff_near = np.empty_like(exact_near)
  clamped = xi_near <= 1e-3
  ff_near[clamped] = np.asarray(farfield_resistance_scalars(2.001))
  ff_near[~clamped] = _sequential_jit_map(
      lambda x: farfield_resistance_scalars(2.0+x),
      xi_near[~clamped], batch_size)
  raw[near] = (exact_near-ff_near)*np.asarray(CONV)

  # Vectorized Hermite evaluation uses the already-validated JAX endpoint
  # values, avoiding repeated Wilson solves inside the bridge.
  if np.any(bridge):
    y0,y1,m0,m1 = (np.asarray(x) for x in _handoff_data())
    length = math.log(XI_MID_START/XI_NEAR_END)
    t = np.log(xi[bridge]/XI_NEAR_END)/length; t2=t*t; t3=t2*t
    raw[bridge] = ((2*t3-3*t2+1)[:,None]*y0
                   + (t3-2*t2+t)[:,None]*length*m0
                   + (-2*t3+3*t2)[:,None]*y1
                   + (t3-t2)[:,None]*length*m1)
  mid_vals, mid_counts = _sequential_jit_map(
      midfield_residual, xi[mid], batch_size)
  raw[mid] = mid_vals; counts[mid] = mid_counts
  shift, cutoff_counts = midfield_residual(jnp.asarray(xi_max, dtype=F64))
  shift = np.asarray(shift)
  vals = raw - shift[None, :]
  vals[-1] = 0.0
  counts[-1] = np.asarray(cutoff_counts)
  return xi, vals, shift, counts


def validate_table(xi: np.ndarray, vals: np.ndarray,
                   counts: np.ndarray) -> Dict[str, float]:
  """Fail-fast invariants for a generated archive."""
  if xi.shape != (vals.shape[0],) or vals.shape[1] != 22:
    raise AssertionError(f'bad generated shapes: xi={xi.shape}, vals={vals.shape}')
  if not np.all(np.diff(xi) > 0) or not np.all(np.isfinite(vals)):
    raise AssertionError('generated table is not finite and strictly ordered')
  if xi[0] != XI_MIN_DEFAULT or xi[-1] != XI_MAX_DEFAULT:
    # Non-default CLI grids are valid too; this assertion is handled by main.
    pass
  if np.max(np.abs(vals[-1])) != 0.0:
    raise AssertionError('cutoff row is not exactly zero')
  if counts.max(initial=0) >= WILSON_MAX_ITERS:
    raise AssertionError('at least one Wilson solve hit the iteration cap')
  c = {name: i for i, name in enumerate(COLUMN_NAMES)}
  deep = xi <= 1e-6
  squeeze = vals[deep, c['XA11']] * xi[deep]
  if not np.all((squeeze > .245) & (squeeze < .255)):
    raise AssertionError('XA11 does not follow the 1/(4 xi) asymptote')
  slope = np.polyfit(np.log(1/xi[deep]), vals[deep, c['YA11']], 1)[0]
  if abs(slope - 1/6) > 5e-3:
    raise AssertionError(f'YA11 log slope is {slope:g}, expected 1/6')
  # Direct endpoint evaluations pin value continuity independently of grid.
  y0 = np.asarray(nearfield_residual(XI_NEAR_END))
  b0 = np.asarray(handoff_residual(XI_NEAR_END))
  y1 = np.asarray(midfield_residual(XI_MID_START)[0])
  b1 = np.asarray(handoff_residual(XI_MID_START))
  seam_error = max(float(np.max(np.abs(y0-b0))),
                   float(np.max(np.abs(y1-b1))))
  if seam_error > 1e-10:
    raise AssertionError(f'handoff value discontinuity: {seam_error:g}')
  return {'xa11_xi_min': float(squeeze.min()),
          'xa11_xi_max': float(squeeze.max()),
          'ya11_log_slope': float(slope),
          'handoff_max_abs_jump': seam_error,
          'wilson_max_reflections': int(counts.max(initial=0))}


def _interpolate_generated(xi: np.ndarray, vals: np.ndarray,
                           xi_query: np.ndarray):
  # Runtime uses the log grid only to locate the bracketing rows; its final
  # interpolation weight is linear in raw center distance, equivalently xi.
  q = np.clip(xi_query, xi[0], xi[-1])
  return np.stack([np.interp(q, xi, vals[:, j]) for j in range(22)], axis=1)


def compare_archive(xi: np.ndarray, vals: np.ndarray, path: str):
  """Comparison metrics using the same log-gap interpolation as runtime."""
  with np.load(path, allow_pickle=True) as ref:
    rxi = np.asarray(ref['dist'], dtype=np.float64) - 2.0
    rvals = np.asarray(ref['vals'], dtype=np.float64)
  got = _interpolate_generated(xi, vals, rxi)
  abs_err = np.abs(got-rvals)
  mixed = abs_err/(1+np.abs(rvals))
  regions = {'near': rxi <= XI_NEAR_END,
             'handoff': (rxi > XI_NEAR_END) & (rxi < XI_MID_START),
             'midfield': rxi >= XI_MID_START,
             'all': np.ones_like(rxi, dtype=bool)}
  result = {'path': path, 'rows': int(rxi.size), 'regions': {}}
  for name, mask in regions.items():
    if not np.any(mask):
      continue
    ae = abs_err[mask]; me = mixed[mask]
    flat = int(np.argmax(me)); row, col = np.unravel_index(flat, me.shape)
    source_row = np.flatnonzero(mask)[row]
    result['regions'][name] = {
        'max_abs': float(ae.max()),
        'rms_abs': float(np.sqrt(np.mean(ae*ae))),
        'p99_mixed': float(np.percentile(me, 99)),
        'max_mixed': float(me.max()),
        'worst_gap': float(rxi[source_row]),
        'worst_scalar': COLUMN_NAMES[col]}
  return result


def _atomic_savez(path: str, **arrays):
  folder = os.path.dirname(os.path.abspath(path))
  os.makedirs(folder, exist_ok=True)
  mode = (os.stat(path).st_mode & 0o777) if os.path.exists(path) else 0o644
  fd, tmp = tempfile.mkstemp(prefix='.resistance_table_', suffix='.npz',
                             dir=folder)
  os.close(fd)
  try:
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    os.chmod(path, mode)
  finally:
    if os.path.exists(tmp):
      os.unlink(tmp)


def main():
  here = os.path.dirname(__file__)
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--out', default=os.path.join(here, 'resistance_table.npz'))
  ap.add_argument('--rows', type=int, default=N_ROWS_DEFAULT)
  ap.add_argument('--xi-min', type=float, default=XI_MIN_DEFAULT)
  ap.add_argument('--xi-max', type=float, default=XI_MAX_DEFAULT)
  ap.add_argument('--batch-size', type=int, default=8)
  ap.add_argument('--compare', action='store_true')
  ap.add_argument('--legacy-reference',
                  default=os.path.join(here, 'resistance_table_legacy.npz'))
  ap.add_argument('--current-reference',
                  default=os.path.join(here, 'resistance_table.npz'))
  ap.add_argument('--report',
                  default=os.path.join(here, 'resistance_table_comparison.json'))
  args = ap.parse_args()

  # Read references before atomically replacing the production archive.
  reference_paths = []
  if args.compare:
    for path in (args.legacy_reference, args.current_reference):
      if os.path.exists(path) and os.path.abspath(path) not in {
          os.path.abspath(p) for p in reference_paths}:
        reference_paths.append(path)

  print(f'Generating {args.rows} rows over xi=[{args.xi_min:g}, '
        f'{args.xi_max:g}] in float64...')
  xi, vals, shift, counts = generate_table(
      args.rows, args.xi_min, args.xi_max, args.batch_size)
  checks = validate_table(xi, vals, counts)
  dist = 2.0 + xi
  dr = math.log10(args.xi_max/args.xi_min)/(args.rows-1)
  comparisons = [compare_archive(xi, vals, p) for p in reference_paths]
  report = {'validation': checks, 'comparisons': comparisons,
            'grid': {'rows': args.rows, 'xi_min': args.xi_min,
                     'xi_max': args.xi_max, 'dr': dr}}

  # Independent methods need not reproduce Fiore bit-for-bit, but large
  # departures indicate a convention or extraction error.
  for comp in comparisons:
    near = comp['regions'].get('near', {})
    mid = comp['regions'].get('midfield', {})
    if near.get('max_mixed', 0.0) > 1e-2:
      raise AssertionError(
          f"near-field comparison failed for {comp['path']}: {near}")
    if mid.get('max_mixed', 0.0) > .5:
      raise AssertionError(
          f"midfield comparison failed for {comp['path']}: {mid}")

  provenance = np.asarray([
      'all rows generated from equations; no committed resistance values copied',
      'xi <= 0.01: Townsend (2023) corrected Jeffrey-Onishi expressions',
      '0.01 < xi < 0.02: slope-limited cubic Hermite bridge in log(xi)',
      'xi >= 0.02: x64 JAX port of Wilson (2013) Lamb/reflection method',
      'two-body FTS Minfinity inverse subtracted; residual at xi=2 removed'])
  _atomic_savez(
      args.out, dist=dist, vals=vals, xi_min=np.float64(args.xi_min),
      dr=np.float64(dr), column_names=np.asarray(COLUMN_NAMES),
      shift_constants=shift, method_bounds=np.asarray(
          [XI_NEAR_END, XI_MID_START]),
      wilson_tolerance=np.float64(WILSON_TOL), provenance=provenance)
  if args.compare:
    with open(args.report, 'w') as fh:
      json.dump(report, fh, indent=2, sort_keys=True)
      fh.write('\n')
  print(json.dumps(report, indent=2, sort_keys=True))
  print(f'Wrote {args.out}')


if __name__ == '__main__':
  main()
