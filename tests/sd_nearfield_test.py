"""Deterministic validation of the near-field lubrication resistance R^nf.

Phase 1 (monodisperse, no Brownian, no saddle solve).  See
``/Users/Aslan016/.claude/plans/phase-1-near-field-async-cascade.md``.

Validation layers, weakest-pin caveats noted inline:
  #1  structural: basis orthonormality, Caveat A (scalars -> 0 at r=4a),
      per-pair symmetry + PSD, sub-block parity, SU == FE^T (self).
  #1b multi-particle apply-path vs an independent explicit-index dense assembly
      (periodic + triclinic/sheared), covering the linked-cell machinery.
  #2  transcription regression vs a literal numpy replica of the FSD kernel
      forms (implementation-equivalence, NOT an external pin).
  #3  units/prefactor: FU prefactor pinned externally by the analytic squeeze
      asymptotic; far-field reduction trend near r=4a (internal).
  #4  EXTERNAL GATE (deferred): analytic Jeffrey-Onishi two-body resistance.
      xfail(strict) -- a Phase-2 entry blocker, never silently "done".
"""

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_md import space  # noqa: E402
from jax_md.hydro import rpy_moments  # noqa: E402
from jax_md.hydro import sd_nearfield as nf  # noqa: E402
from jax_md.hydro import sd_nearfield_table as nf_table  # noqa: E402


_BASIS = np.array(rpy_moments.stresslet_basis())  # (5,3,3)
_EPS = nf._EPS_NP


def _cross(v, r):
  return np.cross(v, r)


def _sym3(t):
  return 0.5 * (t + t.T)


def _s5(t3):
  """Project a symmetric (3,3) onto the orthonormal 5-basis."""
  return np.einsum('aij,ij->a', _BASIS, _sym3(t3))


def _e3(e5):
  """Expand 5-basis coords into a symmetric traceless (3,3) tensor."""
  return np.einsum('a,aij->ij', e5, _BASIS)


# ---------------------------------------------------------------------------
# Literal numpy replica of the FSD Lubrication.cu kernel forms (dimensionalized
# with the same Kim-Karrila prefactors).  Independent of the jax einsum path:
# this catches transcription bugs, not physics-understanding errors.
# ---------------------------------------------------------------------------
def _fsd_literal_pair_force(rhat, scal, a, eta, gi, gj):
  """Return (F_i, L_i, S_i) for one pair from gi=(U,W,E5)_i, gj=(...)_j."""
  pref = nf._kim_karrila_prefactors(a, eta)
  c = nf_table.COLUMN_INDEX
  g = lambda k, p: pref[p] * scal[c[k]]
  XA11, XA12 = g('XA11', 'A'), g('XA12', 'A')
  YA11, YA12 = g('YA11', 'A'), g('YA12', 'A')
  YB11, YB12 = g('YB11', 'B'), g('YB12', 'B')
  XC11, XC12 = g('XC11', 'C'), g('XC12', 'C')
  YC11, YC12 = g('YC11', 'C'), g('YC12', 'C')
  XG11, XG12 = g('XG11', 'G'), g('XG12', 'G')
  YG11, YG12 = g('YG11', 'G'), g('YG12', 'G')
  YH11, YH12 = g('YH11', 'H'), g('YH12', 'H')
  XM11, XM12 = g('XM11', 'M'), g('XM12', 'M')
  YM11, YM12 = g('YM11', 'M'), g('YM12', 'M')
  ZM11, ZM12 = g('ZM11', 'M'), g('ZM12', 'M')

  Ui, Wi, Ei = gi[0:3], gi[3:6], _e3(gi[6:11])
  Uj, Wj, Ej = gj[0:3], gj[3:6], _e3(gj[6:11])
  r = rhat

  # --- FU force/torque (Lubrication.cu RFU) ---
  rdui, rduj = r @ Ui, r @ Uj
  rdwi, rdwj = r @ Wi, r @ Wj
  fi = ((XA11 - YA11) * rdui * r + YA11 * Ui
        + (XA12 - YA12) * rduj * r + YA12 * Uj
        + YB11 * (-_cross(Wi, r)) + YB12 * (_cross(Wj, r)))  # YB21=-YB12
  li = (YB11 * _cross(Ui, r) + YB12 * _cross(Uj, r)
        + (XC11 - YC11) * rdwi * r + YC11 * Wi
        + (XC12 - YC12) * rdwj * r + YC12 * Wj)

  # Force/torque from strain.  We assemble FSD's RFE forms (XG21=-XG12,
  # YG21=-YG12, YH21=+YH12) and then NEGATE: jax-md's grand resistance is
  # symmetric in (U,Omega,E) so force-from-strain = +(stresslet-from-velocity)^T,
  # whereas FSD's RFE is symmetric in (U,Omega,-E) and equals the negative
  # transpose.  This documents the convention; all tensor *structure* is FSD's.
  XG21, YG21, YH21 = -XG12, -YG12, YH12
  Edri, Edrj = Ei @ r, Ej @ r
  rdEdri, rdEdrj = r @ Edri, r @ Edrj
  fsd_fi_E = ((XG11 - 2.0 * YG11) * (-rdEdri) * r + 2.0 * YG11 * (-Edri)
              + (XG21 - 2.0 * YG21) * (-rdEdrj) * r + 2.0 * YG21 * (-Edrj))
  fsd_li_E = (YH11 * (2.0 * _cross(Edri, r)) + YH21 * (2.0 * _cross(Edrj, r)))
  fi = fi - fsd_fi_E
  li = li - fsd_li_E

  # --- stresslet from U (RSU, G) + from Omega (RSU, H) ---
  def G_S(XG, YG, U):
    rdu = r @ U
    return (XG * np.outer(r, r) * rdu - XG / 3.0 * np.eye(3) * rdu
            + YG * (np.outer(U, r) + np.outer(r, U) - 2.0 * np.outer(r, r) * rdu))

  def H_S(YH, W):
    ew = _cross(W, r)  # (W x r)
    return YH * (np.outer(r, ew) + np.outer(ew, r))

  S = (G_S(XG11, YG11, Ui) + G_S(XG12, YG12, Uj)
       + H_S(YH11, Wi) + H_S(YH12, Wj))

  # --- stresslet from E (RSE, M) ---
  def M_S(XM, YM, ZM, Eb):
    Edr = Eb @ r
    rdEdr = r @ Edr
    out = np.zeros((3, 3))
    for i in range(3):
      for j in range(3):
        dij = 1.0 if i == j else 0.0
        out[i, j] = (
            1.5 * XM * (r[i] * r[j] - dij / 3.0) * rdEdr
            + 0.5 * YM * (2.0 * r[i] * Edr[j] + 2.0 * r[j] * Edr[i]
                          - 4.0 * rdEdr * r[i] * r[j])
            + 0.5 * ZM * (2.0 * Eb[i, j] + (dij + r[i] * r[j]) * rdEdr
                          - 2.0 * r[i] * Edr[j] - 2.0 * r[j] * Edr[i]))
    return out

  S = S + M_S(XM11, YM11, ZM11, Ei) + M_S(XM12, YM12, ZM12, Ej)
  return fi, li, _s5(S)


def _interp(r, a):
  return np.asarray(nf_table.interpolate_scalars(np.asarray([r]), a))[0]


_SEPS = [2.05, 2.3, 2.8, 3.2, 3.6, 3.9]
_RHAT = np.array([0.3, -0.4, np.sqrt(1 - 0.25)])
_RHAT = _RHAT / np.linalg.norm(_RHAT)


# ===========================================================================
# #1 structural
# ===========================================================================
def test_basis_orthonormality():
  """SU == FE^T relies on the stresslet basis being orthonormal (not just orth)."""
  gram = np.einsum('aij,bij->ab', _BASIS, _BASIS)
  np.testing.assert_allclose(gram, np.eye(5), atol=1e-12)


def test_nearfield_scalars_vanish_at_cutoff():
  """Caveat A: every subtracted scalar decays to ~0 as r -> 4a."""
  s = _interp(3.999, 1.0)
  assert np.max(np.abs(s)) < 1e-4, np.max(np.abs(s))


def test_resistance_table_metadata_loaded_from_archive():
  """Table spacing metadata lives in the artifact; runtime clamp policy does not."""
  table = nf_table.load_resistance_table()
  with np.load(nf_table._DATA_PATH, allow_pickle=True) as npz:
    assert 'regularization_index' not in npz.files
    np.testing.assert_allclose(np.asarray(table.xi_min), npz['xi_min'])
    np.testing.assert_allclose(np.asarray(table.dr), npz['dr'])


def test_nearfield_interpolation_uses_full_table():
  """Gaps below FSD's roughness row should use the committed table."""
  table = nf_table.load_resistance_table()
  r = jnp.asarray([table.dist[0]])
  scal = np.asarray(nf_table.interpolate_scalars(r, 1.0, table))[0]
  np.testing.assert_allclose(scal, np.asarray(table.vals[0]), rtol=0.0,
                             atol=1e-12)
  xa11 = nf_table.COLUMN_INDEX['XA11']
  old_cap = np.asarray(table.vals[232])
  assert scal[xa11] > 5.0 * old_cap[xa11]


@pytest.mark.parametrize('sep', _SEPS)
def test_pair_grand_symmetric_psd(sep):
  R = np.array(nf.pair_grand_resistance(_RHAT, sep, 1.0, 1.0))
  rel = np.linalg.norm(R - R.T) / max(np.linalg.norm(R), 1e-30)
  assert rel < 1e-10, rel  # symmetry is exact (congruence of the bare table)
  # PSD caveat: R^nf = R^2B - Rbar^2B is a DIFFERENCE of PSD operators, so it is
  # strictly PSD only where lubrication dominates (near contact).  At larger gaps
  # the strain-coupled blocks slightly over-subtract and lambda_min dips a little
  # negative; only the assembled R = M^-1 + R^nf is PSD everywhere.  (The uniform
  # 6*pi*eta*a^k prefactors are a true congruence of the bare table -- p_G =
  # sqrt(p_A p_M) etc. -- which both restores the near-contact 1/xi cancellation
  # that makes eta'_inf finite AND exposes this physical tail.  The old
  # Kim-Karrila pi(2a)^k prefactors were NOT a congruence: they distorted the
  # spectrum, masking the tail while breaking the cancellation.)
  Rsym = 0.5 * (R + R.T)
  lam = np.linalg.eigvalsh(Rsym).min()
  scale = max(np.linalg.norm(Rsym), 1.0)
  if sep <= 2.5:                          # lubrication-dominated: strictly PSD
    assert lam >= -1e-9 * scale, (sep, lam)
  else:                                   # subtracted tail: small, bounded
    assert lam >= -1e-2 * scale, (sep, lam, scale)


@pytest.mark.parametrize('sep', _SEPS)
def test_fu_block_symmetric_psd(sep):
  """The 12x12 (U,Omega)->(F,L) block alone is symmetric and PSD."""
  R = np.array(nf.pair_grand_resistance(_RHAT, sep, 1.0, 1.0))
  idx = np.r_[0:6, 11:17]  # U,W of particle i then particle j
  fu = R[np.ix_(idx, idx)]
  np.testing.assert_allclose(fu, fu.T, atol=1e-10)
  assert np.linalg.eigvalsh(0.5 * (fu + fu.T)).min() >= -1e-9


def test_subblock_parity():
  """A,C,M even; B,G odd; H even (FSD YH21=+YH12) under r_hat -> -r_hat."""
  scal = np.asarray(nf_table.interpolate_scalars(np.array([2.6]), 1.0))
  Rp, _ = nf._build_pair_operators(jnp.asarray(_RHAT)[None, :], scal, 1.0, 1.0)
  Rm, _ = nf._build_pair_operators(jnp.asarray(-_RHAT)[None, :], scal, 1.0, 1.0)
  Rp, Rm = np.array(Rp[0]), np.array(Rm[0])
  sl = {'U': slice(0, 3), 'W': slice(3, 6), 'E': slice(6, 11)}
  cases = [('U', 'U', 'even'), ('W', 'W', 'even'), ('E', 'E', 'even'),
           ('U', 'W', 'odd'), ('U', 'E', 'odd'), ('W', 'E', 'even')]
  for ro, co, par in cases:
    bp, bm = Rp[sl[ro], sl[co]], Rm[sl[ro], sl[co]]
    if par == 'even':
      np.testing.assert_allclose(bp, bm, atol=1e-12, err_msg=f'{ro}{co}')
    else:
      np.testing.assert_allclose(bp, -bm, atol=1e-12, err_msg=f'{ro}{co}')


def test_su_equals_fe_transpose_self():
  """Self-block: stresslet-from-velocity == (force-from-strain)^T."""
  scal = np.asarray(nf_table.interpolate_scalars(np.array([2.4]), 1.0))
  Rs, _ = nf._build_pair_operators(jnp.asarray(_RHAT)[None, :], scal, 1.0, 1.0)
  Rs = np.array(Rs[0])
  np.testing.assert_allclose(Rs[6:11, 0:3], Rs[0:3, 6:11].T, atol=1e-12)
  np.testing.assert_allclose(Rs[6:11, 3:6], Rs[3:6, 6:11].T, atol=1e-12)


# ===========================================================================
# #1b multi-particle apply-path vs explicit dense
# ===========================================================================
def _explicit_dense(positions_frac, box, a, eta):
  R = np.asarray(positions_frac)
  N = R.shape[0]
  dense = np.zeros((N * 11, N * 11))
  for i in range(N):
    for j in range(N):
      if i == j:
        continue
      df = (R[j] - R[i] + 0.5) % 1.0 - 0.5
      rij = box @ df
      r = np.linalg.norm(rij)
      if r >= 4.0 * a or r == 0.0:
        continue
      rhat = rij / r
      Rs, Rc = nf._build_pair_operators(
          jnp.asarray(rhat)[None, :],
          np.asarray(nf_table.interpolate_scalars(np.array([r]), a)), a, eta)
      dense[i * 11:(i + 1) * 11, i * 11:(i + 1) * 11] += np.array(Rs[0])
      dense[i * 11:(i + 1) * 11, j * 11:(j + 1) * 11] += np.array(Rc[0])
  return dense


def _apply_path_check(box_matrix):
  a, eta = 1.0, 1.0
  R = jnp.array([[0.10, 0.10, 0.10],
                 [0.18, 0.13, 0.11],   # close to particle 0
                 [0.55, 0.52, 0.50],
                 [0.60, 0.55, 0.52]])  # close to particle 2
  space_fns = space.periodic_general(box_matrix, fractional_coordinates=True)
  init_fn, apply_fn = nf.build_nearfield_resistance(space_fns, a, eta)
  state = init_fn(R)
  key = jax.random.PRNGKey(2)
  gv = jax.random.normal(key, (4, 11), dtype=jnp.float64)
  gf, _ = apply_fn(state, R, gv)
  gf = np.array(gf)

  box = np.array(box_matrix)
  dense = _explicit_dense(R, box, a, eta)
  gf_ref = (dense @ np.array(gv).reshape(-1)).reshape(4, 11)
  np.testing.assert_allclose(gf, gf_ref, atol=1e-9, rtol=0.0)
  # global operator symmetric + PSD
  assert np.linalg.norm(dense - dense.T) / np.linalg.norm(dense) < 1e-10
  assert np.linalg.eigvalsh(0.5 * (dense + dense.T)).min() >= -1e-9


def test_apply_matches_explicit_dense_periodic():
  _apply_path_check(jnp.eye(3) * 20.0)


def test_apply_matches_explicit_dense_triclinic():
  """Sheared (triclinic) box exercises the min-image transform under strain."""
  box = jnp.array([[20.0, 6.0, 2.0],
                   [0.0, 20.0, 3.0],
                   [0.0, 0.0, 20.0]])
  _apply_path_check(box)


@pytest.mark.parametrize('box_matrix', [
    jnp.eye(3) * 20.0,
    jnp.array([[20.0, 6.0, 2.0],
               [0.0, 20.0, 3.0],
               [0.0, 0.0, 20.0]]),
], ids=['periodic', 'triclinic'])
def test_prepared_blocks_match_core(box_matrix):
  """prepare + apply_blocks == the matrix-free reference apply (fixed config).

  The prepared-blocks path amortizes geometry/table/block assembly once per
  solve; ``_core`` stays the reference implementation, so equality here (f64,
  near-contact + regularized-overlap pairs, periodic and triclinic) is the
  regression gate for the fast path.
  """
  a, eta = 1.0, 1.0
  # Mix of regularized-overlap, lubrication-gap, and isolated particles; the
  # last pair sits at a proper near-contact separation (2.05a).
  R = jnp.array([[0.10, 0.10, 0.10],
                 [0.18, 0.13, 0.11],
                 [0.55, 0.52, 0.50],
                 [0.60, 0.55, 0.52],
                 [0.30, 0.80, 0.30],
                 [0.30, 0.80, 0.30 + 2.05 / 20.0],
                 [0.85, 0.20, 0.85]])
  space_fns = space.periodic_general(box_matrix, fractional_coordinates=True)
  init_fn, apply_fn = nf.build_nearfield_resistance(space_fns, a, eta)
  state = init_fn(R)
  n = R.shape[0]
  key = jax.random.PRNGKey(7)
  gv = jax.random.normal(key, (n, 11), dtype=jnp.float64)

  prepared = apply_fn.prepare(state, R)

  # Full 11-dof apply equals the matrix-free reference.
  ref = np.array(apply_fn.apply_prepared(state, R, gv))
  fast = np.array(apply_fn.apply_blocks(prepared, gv))
  np.testing.assert_allclose(fast, ref, atol=1e-12, rtol=0.0)

  # FU sub-apply equals the full apply on a zero-strain embedding.
  u6 = gv[:, :6]
  gv_embed = jnp.concatenate([u6, jnp.zeros((n, 5), dtype=gv.dtype)], axis=-1)
  fu = np.array(apply_fn.apply_blocks_FU(prepared, u6))
  np.testing.assert_allclose(
      fu, np.array(apply_fn.apply_blocks(prepared, gv_embed))[:, :6],
      atol=1e-12, rtol=0.0)

  # Diagonal and neighbored-mask agree with the dedicated extraction.
  diag_ref = np.array(apply_fn.diagonal_FU_prepared(state, R))
  np.testing.assert_allclose(
      np.array(nf.prepared_diag_FU(prepared)), diag_ref, atol=1e-12, rtol=0.0)
  _, has_neighbor_ref = apply_fn.diagonal_FU_mask_prepared(state, R)
  np.testing.assert_array_equal(np.array(prepared.has_neighbor),
                                np.array(has_neighbor_ref))


def test_apply_live_shear_matches_static_deformed_box():
  """Raw ``apply_fn``/``diagonal_FU`` under a live ``space.shearing`` box.

  Regression: the neighbor update used to drop the ``box`` kwarg on the live
  shear path, falling back to the builder's scalar default -- a ``lax.cond``
  pytree mismatch against the (3,3) allocation box.  The pin: the live-box
  apply at strain ``gamma`` must equal the static ``periodic_general`` build
  at the literal deformed box.
  """
  a, eta = 1.0, 1.0
  L, gamma = 20.0, 0.25
  R = jnp.array([[0.10, 0.10, 0.10],
                 [0.18, 0.13, 0.11],
                 [0.55, 0.52, 0.50],
                 [0.60, 0.55, 0.52]])
  gv = jax.random.normal(jax.random.PRNGKey(2), (4, 11), dtype=jnp.float64)
  shear = dict(gamma_xy=gamma, gamma_xz=0.0, gamma_yz=0.0)

  disp, shift, box_of = space.shearing(L * jnp.eye(3))
  init_s, apply_s = nf.build_nearfield_resistance((disp, shift, box_of), a, eta)
  st = init_s(R, **shear)
  gf_live, _ = apply_s(st, R, gv, **shear)
  diag_live = apply_s.diagonal_FU(st, R, **shear)

  Hdef = (L * jnp.eye(3)).at[0, 1].set(gamma * L)  # space.shearing convention
  init_d, apply_d = nf.build_nearfield_resistance(
      space.periodic_general(Hdef, fractional_coordinates=True), a, eta)
  st_d = init_d(R)
  gf_static, _ = apply_d(st_d, R, gv)
  diag_static = apply_d.diagonal_FU(st_d, R)

  assert float(jnp.linalg.norm(gf_static)) > 0.0  # pairs inside r_lub
  np.testing.assert_allclose(np.asarray(gf_live), np.asarray(gf_static),
                             rtol=0.0, atol=1e-11)
  np.testing.assert_allclose(np.asarray(diag_live), np.asarray(diag_static),
                             rtol=0.0, atol=1e-11)


# ===========================================================================
# #2 transcription regression vs literal FSD replica
# ===========================================================================
@pytest.mark.parametrize('sep', _SEPS)
def test_transcription_vs_fsd_literal(sep):
  a, eta = 1.0, 1.0
  rng = np.random.default_rng(0)
  gi = rng.standard_normal(11)
  gj = rng.standard_normal(11)
  scal = _interp(sep, a)

  Rs, Rc = nf._build_pair_operators(jnp.asarray(_RHAT)[None, :],
                                    np.asarray(scal)[None, :], a, eta)
  out = np.array(Rs[0]) @ gi + np.array(Rc[0]) @ gj  # (11,) F_i,L_i,S_i

  fi, li, s5 = _fsd_literal_pair_force(_RHAT, scal, a, eta, gi, gj)
  ref = np.concatenate([fi, li, s5])
  np.testing.assert_allclose(out, ref, atol=1e-10, rtol=0.0)


# ===========================================================================
# #3 units / prefactor
# ===========================================================================
def test_fu_prefactor_squeeze_asymptotic():
  """EXTERNAL pin for the A prefactor: near contact the FU squeeze-mode
  resistance R^nf_FU . r_hat . r_hat ~ 6 pi eta a * (1/4) / xi (Kim-Karrila /
  lubrication theory), since the far-field part is negligible at small gap."""
  a, eta = 1.0, 1.0
  for xi in (3e-3, 5e-3, 1e-2):
    s = 2.0 + xi
    R = np.array(nf.pair_grand_resistance(np.array([1.0, 0.0, 0.0]), s, a, eta))
    # squeeze mode: equal-and-opposite velocities along the line of centers.
    u = np.zeros(22)
    u[0] = 1.0      # U_i = +x
    u[11] = -1.0    # U_j = -x
    f = R @ u
    squeeze_resist = f[0]  # force on particle i along x
    # Leading lubrication: F_i = 6 pi eta a (X^A_11 - X^A_12) U_i, with
    # X^A_11 - X^A_12 -> (1/2)/xi as xi -> 0 (monodisperse g1 = 1/4 each).
    analytic = 6.0 * np.pi * eta * a * 0.5 / xi
    assert abs(squeeze_resist - analytic) / analytic < 0.05, (
        xi, squeeze_resist, analytic)


def test_farfield_reduction_trend_near_cutoff():
  """INTERNAL pin (do not promote to external): the near-field correction is a
  small, monotonically growing add-on to the far field as we move inward from
  r=4a, and vanishes at the cutoff.  A gross prefactor error (esp. SE/M) would
  make R^nf comparable to / larger than the far field where it should be tiny.
  """
  a, eta = 1.0, 1.0
  seps = np.array([3.9, 3.7, 3.5, 3.3])
  fu_norm = []
  se_norm = []
  for s in seps:
    R = np.array(nf.pair_grand_resistance(np.array([1.0, 0.0, 0.0]), s, a, eta))
    idx_fu = np.r_[0:6, 11:17]
    idx_se = np.r_[6:11, 17:22]
    fu_norm.append(np.linalg.norm(R[np.ix_(idx_fu, idx_fu)]))
    se_norm.append(np.linalg.norm(R[np.ix_(idx_se, idx_se)]))
  # Monotonic growth inward, and vanishing at the cutoff.
  assert np.all(np.diff(fu_norm) > 0), fu_norm
  assert np.all(np.diff(se_norm) > 0), se_norm
  s_cut = np.array(nf.pair_grand_resistance(
      np.array([1.0, 0.0, 0.0]), 3.999, a, eta))
  assert np.linalg.norm(s_cut) < 1e-2, np.linalg.norm(s_cut)


# ===========================================================================
# #4 EXTERNAL GATE -- deferred Phase-2 entry blocker
# ===========================================================================
@pytest.mark.xfail(strict=True, reason='Phase-2 entry gate: analytic '
                   'Jeffrey-Onishi (1984)/Jeffrey (1992) two-body resistance '
                   'not yet implemented. Phase 2 must not begin until this '
                   'passes. See plan Decision 2.')
def test_isolated_pair_vs_analytic_jeffrey_onishi():
  raise NotImplementedError(
      'TODO: implement analytic JO two-body resistance and assert '
      'R_FU = B^T (M^2B)^-1 B + R^nf reproduces it across separations.')
