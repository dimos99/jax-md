"""Exact free-space / Hasimoto-screened RPY-dipole pair tensors by quadrature.

External ground truth for the grand-mobility (stresslet) extension.  Pure
numpy/scipy, no JAX.  Everything is derived from the single defining Fourier
representation of the RPY multipole mobilities (monodisperse spheres,
unbounded fluid; conventions matching the wave-space operator):

  force density  f_hat_m(k) = j0(ka) F_m - i Pdip(ka) k_n C_mn
  fluid solve    u_hat = (1/(eta k^2)) (I - khat khat) f_hat
  outputs        U = j0(ka) u_hat,   D_ij = i Pdip(ka) k_j u_hat_i

with j0(x) = sin x / x and Pdip(x) = 3 (sin x - x cos x)/x^3 = 3 j1(x)/x.

Pair tensors are M(r) = (2 pi)^{-3} Int d^3k e^{i k.r} (per-mode kernel),
with r = x_receiver - x_source.  The angular integrals are done analytically
(spherical Bessel identities); the remaining 1D radial integrals are either

  * free space: decomposed into Sum c * trig(omega k)/k^n terms and integrated
    with QUADPACK's oscillatory QAWF rule on [delta, inf) plus an adaptive
    panel on [0, delta], or
  * Hasimoto-screened (factor H(k, xi) = (1 + k^2/4xi^2) e^{-k^2/4xi^2}):
    plain adaptive quadrature (Gaussian-damped integrand).

For any xi, (real-space closed forms) + (screened tensors here) must equal
the free-space tensors here.  That identity validates every transcribed
real-space scalar coefficient, every tensor index convention, and the sign
of the DF block, component by component.

All tensors are reported as 6*pi*eta*M (eta-independent).
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn

# ----------------------------------------------------------------------------
# Term-list algebra: a "term" is (coef, n, kind, omega) == coef * trig(omega k)/k^n
# ----------------------------------------------------------------------------


def _norm_term(coef, n, kind, omega):
  if omega < 0.0:
    if kind == 'sin':
      coef, omega = -coef, -omega
    else:
      omega = -omega
  return (coef, n, kind, omega)


def tmul(t1, t2):
  """Product of two term lists via product-to-sum identities."""
  out = []
  for (c1, n1, k1, w1) in t1:
    for (c2, n2, k2, w2) in t2:
      c = 0.5 * c1 * c2
      n = n1 + n2
      if k1 == 'sin' and k2 == 'sin':
        out.append(_norm_term(c, n, 'cos', w1 - w2))
        out.append(_norm_term(-c, n, 'cos', w1 + w2))
      elif k1 == 'cos' and k2 == 'cos':
        out.append(_norm_term(c, n, 'cos', w1 - w2))
        out.append(_norm_term(c, n, 'cos', w1 + w2))
      elif k1 == 'sin' and k2 == 'cos':
        out.append(_norm_term(c, n, 'sin', w1 + w2))
        out.append(_norm_term(c, n, 'sin', w1 - w2))
      else:  # cos * sin: cos A sin B = (sin(A+B) + sin(B-A)) / 2
        out.append(_norm_term(c, n, 'sin', w1 + w2))
        out.append(_norm_term(c, n, 'sin', w2 - w1))
  return collect(out)


def collect(terms):
  """Merge terms with identical (n, kind, omega); drop zero/sin(0) terms."""
  acc = {}
  for (c, n, kind, w) in terms:
    if kind == 'sin' and w == 0.0:
      continue
    key = (n, kind, w)
    acc[key] = acc.get(key, 0.0) + c
  return [(c, n, kind, w) for (n, kind, w), c in sorted(acc.items())
          if c != 0.0]


def integrate_terms_tail(terms, delta):
  """Integrate a term list on [delta, inf) (QAWF for omega>0, exact else)."""
  total = 0.0
  for (c, n, kind, w) in terms:
    if w == 0.0:
      # cos(0) == 1 constant: exact power-law tail (requires n > 1).
      assert kind == 'cos' and n > 1
      total += c * delta ** (1 - n) / (n - 1)
    else:
      val, _ = quad(lambda k, n=n: k ** (-n), delta, np.inf,
                    weight=kind, wvar=w, limlst=200, limit=400)
      total += c * val
  return total


def integrate_panel(f, delta):
  """Adaptive quadrature of the full (analytic-at-0) integrand on [0, delta]."""
  val, _ = quad(f, 0.0, delta, limit=400, epsabs=1e-13, epsrel=1e-12)
  return val


# ----------------------------------------------------------------------------
# Elementary term lists
# ----------------------------------------------------------------------------


def terms_jl_over_xp(l, p, r):
  """Term list (in k) for j_l(k r) / (k r)^p, l <= 4."""
  # j_l(x) = S_l(1/x) sin x + C_l(1/x) cos x with polynomial coefficients
  # listed by power of 1/x.
  S = {0: {1: 1.0},
       1: {2: 1.0},
       2: {3: 3.0, 1: -1.0},
       3: {4: 15.0, 2: -6.0},
       4: {5: 105.0, 3: -45.0, 1: 1.0}}
  C = {0: {},
       1: {1: -1.0},
       2: {2: -3.0},
       3: {3: -15.0, 1: 1.0},
       4: {4: -105.0, 2: 10.0}}
  out = []
  for pw, c in S[l].items():
    out.append((c / r ** (pw + p), pw + p, 'sin', r))
  for pw, c in C[l].items():
    out.append((c / r ** (pw + p), pw + p, 'cos', r))
  return out


def terms_k_pow(c, n):
  """c / k^n as a term list."""
  return [(c, n, 'cos', 0.0)]


def terms_shape_uf(a):
  """k^2 * j0(ka)^2 = (1 - cos 2ak) / (2 a^2)."""
  return [(0.5 / a ** 2, 0, 'cos', 0.0), (-0.5 / a ** 2, 0, 'cos', 2.0 * a)]


def terms_shape_uc(a):
  """k * j0(ka) * Pdip(ka) = (3/(2 a^4)) (1 - cos 2ak - ak sin 2ak)/k^3."""
  c = 3.0 / (2.0 * a ** 4)
  return [(c, 3, 'cos', 0.0), (-c, 3, 'cos', 2.0 * a),
          (-c * a, 2, 'sin', 2.0 * a)]


def terms_shape_dc(a):
  """k^2 * Pdip(ka)^2 (from 9[(1-cos)/2 - ak sin + a^2k^2(1+cos)/2]/(k^4 a^6))."""
  c6 = 9.0 / (2.0 * a ** 6)
  return [(c6, 4, 'cos', 0.0), (-c6, 4, 'cos', 2.0 * a),
          (-9.0 / a ** 5, 3, 'sin', 2.0 * a),
          (9.0 / (2.0 * a ** 4), 2, 'cos', 0.0),
          (9.0 / (2.0 * a ** 4), 2, 'cos', 2.0 * a)]


# Stable numeric shapes for the [0, delta] panel and screened integrals.
def _j0(x):
  return spherical_jn(0, x)


def _pdip(x):
  x = np.asarray(x, dtype=float)
  small = np.abs(x) < 1e-6
  xs = np.where(small, 1.0, x)
  out = 3.0 * spherical_jn(1, xs) / xs
  return np.where(small, 1.0 - x * x / 10.0, out)


def _hasimoto(k, xi):
  t = (k / (2.0 * xi)) ** 2
  return (1.0 + t) * np.exp(-t)


# ----------------------------------------------------------------------------
# Radial integrals per block.  All are (3/pi) * Int_0^inf dk shape * bessel.
# ----------------------------------------------------------------------------

_DELTA = 2.0


def _radial(shape_terms, shape_fn, bessel_terms, bessel_fn, xi=None):
  """(3/pi) Int_0^inf shape(k) * bessel(k) dk, free (xi=None) or screened."""
  if xi is None:
    terms = tmul(shape_terms, bessel_terms)
    tail = integrate_terms_tail(terms, _DELTA)
    panel = integrate_panel(lambda k: shape_fn(k) * bessel_fn(k), _DELTA)
    return (3.0 / np.pi) * (panel + tail)
  f = lambda k: shape_fn(k) * bessel_fn(k) * _hasimoto(k, xi)
  kmax = _DELTA + 12.0 * xi
  val, _ = quad(f, 0.0, kmax, limit=800, epsabs=1e-13, epsrel=1e-12)
  return (3.0 / np.pi) * val


def uf_radials(r, a, xi=None):
  """(F1-like, F2-like) radial integrals for the UF block (x 6 pi eta a)."""
  shape_t = [(c * a, n, k, w) for (c, n, k, w) in terms_shape_uf(a)]
  shape_f = lambda k: a * k ** 2 * _j0(k * a) ** 2

  def make(bt, bf):
    return _radial(shape_t, shape_f, bt, bf, xi)

  # transverse: (j0 - j1/x)(kr); longitudinal: (j0 - j1/x + j2)(kr)
  bt_trans = collect(terms_jl_over_xp(0, 0, r) +
                     [(-c, n, k, w) for (c, n, k, w) in terms_jl_over_xp(1, 1, r)])
  bf_trans = lambda k: (spherical_jn(0, k * r)
                        - spherical_jn(1, k * r) / np.maximum(k * r, 1e-300))
  bt_long = collect(bt_trans + terms_jl_over_xp(2, 0, r))
  bf_long = lambda k: bf_trans(k) + spherical_jn(2, k * r)
  # note: shape includes 1/k^2 from the Stokeslet; fold it in.
  shape_t = [(c, n + 2, k, w) for (c, n, k, w) in shape_t]
  shape_f2 = lambda k: shape_f(k) / np.maximum(k * k, 1e-300)
  F1 = _radial(shape_t, shape_f2, bt_trans, bf_trans, xi)
  F2 = _radial(shape_t, shape_f2, bt_long, bf_long, xi)
  return F1, F2


def uc_radials(r, a, xi=None):
  """(I_j1, I_j2x, I_j3) for the UC block (x 6 pi eta).

  6 pi eta M_UC,imn = I_j1 d_im rhat_n
                    - I_j2x (d_im rhat_n + d_in rhat_m + d_mn rhat_i)
                    + I_j3 rhat_i rhat_m rhat_n
  """
  st = terms_shape_uc(a)
  sf = lambda k: k * _j0(k * a) * _pdip(k * a)
  I_j1 = _radial(st, sf, terms_jl_over_xp(1, 0, r),
                 lambda k: spherical_jn(1, k * r), xi)
  I_j2x = _radial(st, sf, terms_jl_over_xp(2, 1, r),
                  lambda k: spherical_jn(2, k * r) / np.maximum(k * r, 1e-300),
                  xi)
  I_j3 = _radial(st, sf, terms_jl_over_xp(3, 0, r),
                 lambda k: spherical_jn(3, k * r), xi)
  return I_j1, I_j2x, I_j3


def dc_radials(r, a, xi=None):
  """(W1..W5) for the DC block (x 6 pi eta).

  6 pi eta M_DC,ijmn = W1 d_im d_jn - W2 d_im rhat_j rhat_n
                     - W3 (d_ij d_mn + d_im d_jn + d_in d_jm)
                     + W4 (six d-rhat-rhat symmetrized terms)
                     - W5 rhat_i rhat_j rhat_m rhat_n
  """
  st = terms_shape_dc(a)
  sf = lambda k: k ** 2 * _pdip(k * a) ** 2
  x = lambda k: np.maximum(k * r, 1e-300)
  W1 = _radial(st, sf, terms_jl_over_xp(1, 1, r),
               lambda k: spherical_jn(1, k * r) / x(k), xi)
  W2 = _radial(st, sf, terms_jl_over_xp(2, 0, r),
               lambda k: spherical_jn(2, k * r), xi)
  W3 = _radial(st, sf, terms_jl_over_xp(2, 2, r),
               lambda k: spherical_jn(2, k * r) / x(k) ** 2, xi)
  W4 = _radial(st, sf, terms_jl_over_xp(3, 1, r),
               lambda k: spherical_jn(3, k * r) / x(k), xi)
  W5 = _radial(st, sf, terms_jl_over_xp(4, 0, r),
               lambda k: spherical_jn(4, k * r), xi)
  return W1, W2, W3, W4, W5


def dc_self_scalar(a, xi=None):
  """(3/pi) Int k^2 Pdip(ka)^2 [H] dk; free value is 9/(2 a^3 ... ) check.

  6 pi eta M_DC,self = -(scalar/5) * (d_ij d_mn + d_in d_jm - 4 d_im d_jn)/3
  -- see tensor assembly below; returned raw scalar = (3/pi) * integral.
  """
  st = terms_shape_dc(a)
  sf = lambda k: k ** 2 * _pdip(k * a) ** 2
  one = [(1.0, 0, 'cos', 0.0)]
  return _radial(st, sf, one, lambda k: np.ones_like(np.asarray(k, float)), xi)


# ----------------------------------------------------------------------------
# Tensor assembly (r = x_receiver - x_source; outputs are 6*pi*eta*M)
# ----------------------------------------------------------------------------


def muf_tensor(r_vec, a, xi=None):
  """6 pi eta a M_UF (3,3): F1 (I - rr) + F2 rr."""
  r_vec = np.asarray(r_vec, float)
  r = np.linalg.norm(r_vec)
  rh = r_vec / r
  F1, F2 = uf_radials(r, a, xi)
  P = np.eye(3) - np.outer(rh, rh)
  return F1 * P + F2 * np.outer(rh, rh)


def muc_tensor(r_vec, a, xi=None):
  """6 pi eta M_UC (3,3,3): U_i response to couplet C_mn at -r_vec away."""
  r_vec = np.asarray(r_vec, float)
  r = np.linalg.norm(r_vec)
  rh = r_vec / r
  I_j1, I_j2x, I_j3 = uc_radials(r, a, xi)
  d = np.eye(3)
  M = np.zeros((3, 3, 3))
  for i in range(3):
    for m in range(3):
      for n in range(3):
        M[i, m, n] = (I_j1 * d[i, m] * rh[n]
                      - I_j2x * (d[i, m] * rh[n] + d[i, n] * rh[m]
                                 + d[m, n] * rh[i])
                      + I_j3 * rh[i] * rh[m] * rh[n])
  return M


def mdf_tensor(r_vec, a, xi=None):
  """6 pi eta M_DF (3,3,3): D_ij response to force F_m.

  Analytically M_DF,ijm(r) = -M_UC,mij(r) (same r vector).
  """
  M_uc = muc_tensor(r_vec, a, xi)
  return -np.transpose(M_uc, (1, 2, 0))


def mdc_tensor(r_vec, a, xi=None):
  """6 pi eta M_DC (3,3,3,3): D_ij response to couplet C_mn."""
  r_vec = np.asarray(r_vec, float)
  r = np.linalg.norm(r_vec)
  rh = r_vec / r
  W1, W2, W3, W4, W5 = dc_radials(r, a, xi)
  d = np.eye(3)
  M = np.zeros((3, 3, 3, 3))
  for i in range(3):
    for j in range(3):
      for m in range(3):
        for n in range(3):
          six = (d[i, j] * rh[m] * rh[n] + d[i, m] * rh[j] * rh[n]
                 + d[i, n] * rh[j] * rh[m] + d[j, m] * rh[i] * rh[n]
                 + d[j, n] * rh[i] * rh[m] + d[m, n] * rh[i] * rh[j])
          M[i, j, m, n] = (W1 * d[i, m] * d[j, n]
                           - W2 * d[i, m] * rh[j] * rh[n]
                           - W3 * (d[i, j] * d[m, n] + d[i, m] * d[j, n]
                                   + d[i, n] * d[j, m])
                           + W4 * six
                           - W5 * rh[i] * rh[j] * rh[m] * rh[n])
  return M


def mdc_self_tensor(a, xi=None):
  """6 pi eta M_DC self (3,3,3,3): D_ij response to own couplet."""
  s = dc_self_scalar(a, xi)
  d = np.eye(3)
  M = np.zeros((3, 3, 3, 3))
  for i in range(3):
    for j in range(3):
      for m in range(3):
        for n in range(3):
          M[i, j, m, n] = s * (d[i, m] * d[j, n] / 3.0
                               - (d[i, j] * d[m, n] + d[i, m] * d[j, n]
                                  + d[i, n] * d[j, m]) / 15.0)
  return M


def muf_self_tensor(a, xi=None):
  """6 pi eta a M_UF self (3,3)."""
  st = [(c * a, n + 2, k, w) for (c, n, k, w) in terms_shape_uf(a)]
  sf = lambda k: a * _j0(k * a) ** 2
  one = [(1.0, 0, 'cos', 0.0)]
  s = _radial(st, sf, one, lambda k: np.ones_like(np.asarray(k, float)), xi)
  return s * (2.0 / 3.0) * np.eye(3)
