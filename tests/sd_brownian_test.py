"""Fluctuation--dissipation validation for the FSD Brownian step (Phase 3).

Gate ordering (each isolates one new piece; see the Phase-3 plan):

  * **Check 1 -- near-field force covariance.**  The new sampler
    ``nearfield_brownian_force`` in isolation: empirical
    ``Cov(F^B_nf) = (2kT/dt) R^nf_FU`` (densely probed).  The *bare* Lanczos
    square root is the first-light gate; the *preconditioned* one must give the
    same covariance (only faster).  A neighborless particle's output force AND
    torque must be exactly zero -- the gate on the ``Shift_nn`` / ``Proj``
    separation (a ``4/3``-in-the-projector bug would leak into the rotational
    rows of neighbored particles).

  * **Check 2 -- full displacement covariance.**  Stochastic-only solve
    (no drift): empirical ``Cov([U,Omega]) = (2kT/dt) R_FU^{-1}`` against the
    *validated* Phase-2 deterministic ``solve_fn`` (internal localization pin).
    The Ladd short-time self-diffusion sweep (``@slow``) is the external
    correctness pin.

  * **Check 4 -- RFD drift eps-independence.**  The drift estimate is stable
    across a decade of ``eps``.  A negative control (loosen the second displaced
    solve's absolute tolerance) confirms the test actually detects a
    warm-start-induced residual asymmetry.

  * **Key-stream independence.**  The three random inputs {far-field slip,
    near-field force, RFD displacement} are pairwise uncorrelated.
"""

import math

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_md import space  # noqa: E402
from jax_md.hydro import sd_brownian as sdb  # noqa: E402
from jax_md.hydro.rpy_saddle import (  # noqa: E402
    build_saddle_solve,
    _gv_from_u6,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build(cart, L, *, a=1.0, eta=1.0, xi=0.5, P=16, Mgrid=48):
  box = jnp.eye(3) * L
  disp, shift_fn = space.periodic_general(box, fractional_coordinates=True)
  pos = jnp.asarray(np.asarray(cart, dtype=np.float64) / L)
  init_fn, solve_fn = build_saddle_solve(
      (disp, shift_fn), a, eta, xi=xi, P=P, Mgrid=Mgrid,
      gmres_tol=1e-10, gmres_restart=40, gmres_maxiter=6)
  return init_fn, solve_fn, pos, shift_fn


def _pair_and_isolated(a=1.0, gap=0.2):
  """A near-contact pair (gap 0.2a) plus one neighborless particle."""
  return np.array([
      [10.0, 10.0, 10.0],
      [10.0 + 2.0 * a + gap, 10.0, 10.0],
      [22.0, 22.0, 22.0],
  ])


def _dense_rnf_FU(solve_fn, state, pos):
  N = pos.shape[0]

  def rnf(u6):
    gf, _ = solve_fn.nf_apply(state.nf, pos, _gv_from_u6(u6))
    return gf[..., :6]

  R = np.zeros((6 * N, 6 * N))
  for c in range(6 * N):
    e = np.zeros((N, 6))
    e.reshape(-1)[c] = 1.0
    R[:, c] = np.asarray(rnf(jnp.asarray(e))).reshape(-1)
  return R


def _dense_RFU_inv(solve_fn, state, pos):
  """Dense ``R_FU^{-1}`` from the validated deterministic solve (force probes)."""
  N = pos.shape[0]
  Minv = np.zeros((6 * N, 6 * N))
  for c in range(6 * N):
    e = np.zeros((N, 6))
    e.reshape(-1)[c] = 1.0
    e2 = jnp.asarray(e)
    U, Om, _s5, _q, _info = solve_fn(
        state, pos, force=e2[:, :3], torque=e2[:, 3:])
    Minv[:, c] = np.asarray(jnp.concatenate([U, Om], axis=-1)).reshape(-1)
  return 0.5 * (Minv + Minv.T)


def _rel(a, b):
  return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# ---------------------------------------------------------------------------
# Assembled step (Sec. 5) -- smoke test of the full overdamped timestep
# ---------------------------------------------------------------------------
def test_sd_brownian_step_runs_end_to_end():
  a, eta, L = 1.0, 1.0, 30.0
  box = jnp.eye(3) * L
  disp, shift_fn = space.periodic_general(box, fractional_coordinates=True)
  cart = _pair_and_isolated()
  pos = jnp.asarray(cart / L)
  init_fn, step_fn = sdb.build_sd_brownian_step(
      (disp, shift_fn), a, eta, dt=1e-3, kT=1.0, xi=0.5, P=16, Mgrid=48,
      gmres_tol=1e-8, nf_iters=30, mr_iters=40, lanczos_tol=1e-6)
  state = init_fn(pos)
  force = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
  q_new, S5, info = step_fn(state, pos, jax.random.PRNGKey(1), force=force)
  assert q_new.shape == pos.shape and bool(jnp.all(jnp.isfinite(q_new)))
  assert S5.shape == (3, 5) and bool(jnp.all(jnp.isfinite(S5)))
  assert bool(jnp.all(jnp.isfinite(info['U_drift'])))
  # A finite step actually moved the particles.
  assert float(jnp.linalg.norm(q_new - pos)) > 0.0


# ---------------------------------------------------------------------------
# Check 1 -- near-field force covariance
# ---------------------------------------------------------------------------
def test_nearfield_force_covariance_bare():
  init_fn, solve_fn, pos, _ = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  R = _dense_rnf_FU(solve_fn, state, pos)
  assert _rel(R, R.T) < 1e-6
  target = 2.0 * R  # (2 kT / dt) R with kT=dt=1

  # Bare sampler is pure JAX -> vmap + jit over the noise keys.
  sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, 1.0, 1.0, preconditioned=False, iters=40, tol=1e-8)
  keys = jax.random.split(jax.random.PRNGKey(1), 4000)
  F = np.asarray(jax.vmap(sampler)(keys)).reshape(4000, -1)
  C = np.cov(F.T, bias=True)
  assert _rel(C, target) < 0.1


def test_nearfield_force_covariance_preconditioned_matches_bare():
  init_fn, solve_fn, pos, _ = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  R = _dense_rnf_FU(solve_fn, state, pos)
  target = 2.0 * R

  # Preconditioned sampler is the on-device Jacobi-split square root (no host
  # IC(0), no pure_callback) and must give the same covariance as the bare one.
  sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, 1.0, 1.0, preconditioned=True, iters=40, tol=1e-8)
  keys = jax.random.split(jax.random.PRNGKey(2), 800)
  Fp = np.array([np.asarray(sampler(k)).reshape(-1) for k in keys])
  Cp = np.cov(Fp.T, bias=True)
  # Same covariance as the bare sampler / the dense operator (sampling noise).
  assert _rel(Cp, target) < 0.13
  # The neighbored pair's rotational-row diagonal must NOT be reduced by 1/3
  # (the Shift_nn-in-the-projector leak).  Rows 3:6 (particle 0) and 9:12.
  for rows in ((3, 6), (9, 12)):
    s = slice(*rows)
    assert _rel(np.diag(Cp)[s], np.diag(target)[s]) < 0.2


def test_neighborless_force_exactly_zero_preconditioned():
  # The Proj projector zeros all six rows of the neighborless particle exactly,
  # independent of the input -- the gate on the Shift_nn / Proj separation
  # (force rows 0:3 AND torque rows 3:6, where a 4/3-in-the-projector bug leaks).
  init_fn, solve_fn, pos, _ = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, 1.0, 1.0, preconditioned=True, iters=40, tol=1e-8)
  for seed in range(8):
    F = sampler(jax.random.PRNGKey(seed))
    assert float(jnp.max(jnp.abs(F[2]))) == 0.0


def test_neighborless_force_near_zero_bare():
  # The bare sampler has no explicit projector: R^{1/2} has zero rows only
  # analytically, so the Lanczos approximation leaves a machine-epsilon residue.
  init_fn, solve_fn, pos, _ = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, 1.0, 1.0, preconditioned=False, iters=40, tol=1e-8)
  for seed in range(8):
    F = sampler(jax.random.PRNGKey(seed))
    # Relative to the (large, lubrication-dominated) force scale on the pair:
    # the bare Lanczos residue on the zero-eigenvalue subspace is ~sqrt(lam_max)
    # * eps_machine, ~1e-8 of the force scale here (vs exactly 0 preconditioned).
    assert float(jnp.max(jnp.abs(F[2]))) < 1e-6 * float(jnp.linalg.norm(F))


# ---------------------------------------------------------------------------
# Check 2 -- full displacement covariance (internal pin)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_displacement_covariance_internal():
  # Close pair only (N=2): R_FU is 12x12, cheaper to sample densely.
  cart = np.array([[14.0, 15.0, 15.0], [16.2, 15.0, 15.0]])
  init_fn, solve_fn, pos, _ = _build(cart, 30.0)
  state = init_fn(pos)
  kT, dt = 1.0, 1.0
  RFU_inv = _dense_RFU_inv(solve_fn, state, pos)

  slip_sampler = sdb.make_far_field_slip_sampler(solve_fn, state, pos, kT, dt)
  nf_sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, kT, dt, preconditioned=True, iters=40, tol=1e-8)

  def stochastic_velocity(key):
    k_slip, k_nf = jax.random.split(key)
    U_B_flat = slip_sampler(k_slip)
    F_B_nf = nf_sampler(k_nf)
    U, Om, _s5, _q, _info = solve_fn(
        state, pos, slip_top=U_B_flat, extra_force=F_B_nf)
    return np.asarray(jnp.concatenate([U, Om], axis=-1)).reshape(-1)

  keys = jax.random.split(jax.random.PRNGKey(7), 800)
  V = np.array([stochastic_velocity(k) for k in keys])
  C = np.cov(V.T, bias=True) * (dt / (2.0 * kT))
  assert _rel(C, RFU_inv) < 0.15


@pytest.mark.slow
def test_dilute_stokes_einstein_external():
  # External correctness pin (breaks the internal-consistency symmetry of the
  # check above): an isolated sphere has R^nf = 0, so the full stochastic step
  # reduces to the far-field self-mobility.  Its short-time self-diffusion must
  # match the analytic Hasimoto periodic self-mobility mu0 (1 - 2.8373 a/L) --
  # a value independent of solve_fn, pinning the absolute covariance scale
  # end-to-end (slip sampling -> saddle injection -> solve -> displacement).
  a, eta, L = 1.0, 1.0, 40.0
  kT, dt = 1.0, 1.0
  init_fn, solve_fn, pos, _ = _build(
      np.array([[L / 2, L / 2, L / 2]]), L, a=a, eta=eta, xi=0.4, P=20, Mgrid=64)
  state = init_fn(pos)
  mu0 = 1.0 / (6.0 * math.pi * eta * a)
  analytic = mu0 * (1.0 - 2.8373 * a / L)   # leading periodic image correction

  # R^nf = 0 (no neighbors) -> use the pure-JAX block-Jacobi preconditioner.
  # (solve_fn is eager: keep the sample count modest so the per-call gmres
  # compilation cache does not grow without bound.)
  slip_sampler = sdb.make_far_field_slip_sampler(solve_fn, state, pos, kT, dt)
  vels = []
  for seed in range(800):
    U_B_flat = slip_sampler(jax.random.PRNGKey(seed))
    U, _Om, _s5, _q, _info = solve_fn(
        state, pos, slip_top=U_B_flat, preconditioner='jacobi')
    vels.append(np.asarray(U).reshape(-1))
  vels = np.array(vels)
  D_s = float(np.mean(np.sum(vels * vels, axis=1))) * (dt / (2.0 * kT)) / 3.0
  # ~1/sqrt(3*800) ~ 2% statistical; tolerance covers it plus the O((a/L)^3)
  # RPY finite-size term dropped from the analytic value.
  assert abs(D_s - analytic) / analytic < 0.05


# ---------------------------------------------------------------------------
# Check 4 -- RFD drift eps-independence (+ negative control)
# ---------------------------------------------------------------------------
def test_rfd_drift_eps_independence():
  init_fn, solve_fn, pos, shift_fn = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  key = jax.random.PRNGKey(11)
  drifts = {}
  for eps in (1e-3, 1e-4, 1e-5):
    drifts[eps] = np.asarray(sdb.rfd_drift(
        solve_fn, state, pos, key, eps=eps, kT=1.0,
        shift_fn=shift_fn, atol=1e-11))
  ref = drifts[1e-4]
  scale = np.linalg.norm(ref)
  # The drift must be genuinely nonzero (else eps-independence is vacuous).
  assert scale > 1e-6
  for eps in (1e-3, 1e-5):
    assert np.linalg.norm(drifts[eps] - ref) / scale < 5e-2


def test_rfd_warmstart_tolerance_matters():
  # Negative control: the fixed-absolute-tolerance matched solves give the true,
  # nonzero drift.  Loosening only the second (warm-started) solve makes it
  # return the warm-start x0 essentially unconverged -- a corrupted drift.  This
  # proves the matched-absolute-tolerance discipline (not just the warm-start
  # speedup) is what makes the RFD estimate correct.
  init_fn, solve_fn, pos, shift_fn = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  key = jax.random.PRNGKey(11)
  good = np.asarray(sdb.rfd_drift(
      solve_fn, state, pos, key, eps=1e-4, kT=1.0,
      shift_fn=shift_fn, atol=1e-11))
  bad = np.asarray(sdb.rfd_drift(
      solve_fn, state, pos, key, eps=1e-4, kT=1.0,
      shift_fn=shift_fn, atol=1e-11, atol2=1e-2))   # under-converged 2nd solve
  assert np.linalg.norm(good) > 1e-6                       # true drift nonzero
  assert np.linalg.norm(good - bad) / np.linalg.norm(good) > 0.1  # bad differs


# ---------------------------------------------------------------------------
# Key-stream independence
# ---------------------------------------------------------------------------
def test_key_streams_independent():
  init_fn, solve_fn, pos, _ = _build(_pair_and_isolated(), 30.0)
  state = init_fn(pos)
  N = pos.shape[0]
  kT, dt = 1.0, 1.0

  slip_sampler = sdb.make_far_field_slip_sampler(solve_fn, state, pos, kT, dt)
  nf_sampler = sdb.make_nearfield_brownian_sampler(
      solve_fn, state, pos, kT, dt, preconditioned=True, iters=30, tol=1e-8)
  slips, nfs, dqs = [], [], []
  for seed in range(256):
    k_slip, k_nf, k_rfd = jax.random.split(jax.random.PRNGKey(1000 + seed), 3)
    U_B_flat = slip_sampler(k_slip)
    F_B_nf = nf_sampler(k_nf)
    dq6 = jax.random.normal(k_rfd, (N, 6), dtype=jnp.float64)
    slips.append(np.asarray(U_B_flat).reshape(-1))
    nfs.append(np.asarray(F_B_nf).reshape(-1))
    dqs.append(np.asarray(dq6).reshape(-1))
  slips = np.array(slips)
  nfs = np.array(nfs)
  dqs = np.array(dqs)

  def max_cross_corr(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-15)
    B = (B - B.mean(0)) / (B.std(0) + 1e-15)
    return float(np.max(np.abs(A.T @ B) / A.shape[0]))

  # ~1/sqrt(256) = 0.06 sampling floor; independent streams stay well below 0.25.
  assert max_cross_corr(slips, nfs) < 0.25
  assert max_cross_corr(slips, dqs) < 0.25
  assert max_cross_corr(nfs, dqs) < 0.25
