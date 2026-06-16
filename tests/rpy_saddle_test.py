"""Validation for the deterministic FSD saddle-point solve (Phase 2).

In increasing stringency:

  * **Check 0 (Gate A).**  B/Bᵀ are an exact Euclidean adjoint pair
    ``<B u, f> = <u, Bᵀ f>`` and ``rot_embed`` round-trips through
    ``decompose_gradient``.  Independent of the solve; catches the
    factor-2/normalization landmine at its source.
  * **Check 1a (degenerate, force+torque).**  With ``R^nf = 0`` the saddle
    Schur complement is ``Bᵀ M⁻¹ B`` -- the existing stresslet-constrained
    resistance -- so the saddle velocities/angular velocities must reproduce
    ``build_rpy_mobility(..., constrained=True, with_torque=True)`` for the same
    applied force and torque.
  * **Check 1b (degenerate, strain path).**  A single isolated sphere in an
    imposed rate-of-strain develops the Einstein stresslet
    ``S = (20/3) pi eta a^3 E`` with zero translation/rotation.  Exercises the
    couplet embed/extract that the force path never touches.
  * **Gate B sign pin (near-field ON).**  A near-contact pair under *compressive*
    strain must physically approach (and *separate* under extensional strain);
    the imposed-strain stresslet is positive (``S:E > 0``).  Pins the near-field
    block signs end-to-end through ``b2`` -> GMRES -> U.

Slow physical pins (isolated-pair vs analytic Jeffrey-Onishi; simple-cubic
lattice transport coefficients) are added with the IC(0) preconditioner.
"""

import math

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_md import space  # noqa: E402
from jax_md.hydro import rpy_saddle as sad  # noqa: E402
from jax_md.hydro.rpy import build_rpy_mobility  # noqa: E402
from jax_md.hydro.rpy_moments import decompose_gradient  # noqa: E402


def _rel_err(actual, expected):
  actual = np.asarray(actual, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15)


# ---------------------------------------------------------------------------
# Check 0 -- Gate A: B/Bᵀ adjoint pair
# ---------------------------------------------------------------------------
def test_b_bt_adjoint_pair():
  key = jax.random.PRNGKey(0)
  for n in (1, 4, 13):
    ku, kf = jax.random.split(jax.random.fold_in(key, n))
    u6 = jax.random.normal(ku, (n, 6))
    f11 = jax.random.normal(kf, (n, 11))
    lhs = float(jnp.vdot(sad.b_apply(u6), f11))
    rhs = float(jnp.vdot(u6, sad.bt_apply(f11)))
    assert abs(lhs - rhs) <= 1e-11 * (abs(lhs) + 1.0)


def test_rot_embed_round_trip():
  omega = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
  e5, omega_back = decompose_gradient(sad.rot_embed(omega))
  assert float(jnp.linalg.norm(e5)) <= 1e-12
  assert _rel_err(omega_back, omega) <= 1e-12


# ---------------------------------------------------------------------------
# Check 1a -- degenerate reduction to stresslet-constrained mobility
# ---------------------------------------------------------------------------
def test_degenerate_reduces_to_constrained_force_torque():
  a, eta, xi = 1.0, 1.0, 1.0  # xi=1 keeps rcut < L/2 (fast, no image blow-up)
  box = jnp.eye(3) * 12.0
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  n = 5
  pos = jax.random.uniform(jax.random.PRNGKey(1), (n, 3))
  forces = jax.random.normal(jax.random.PRNGKey(2), (n, 3))
  torques = jax.random.normal(jax.random.PRNGKey(3), (n, 3))

  init_c, apply_c = build_rpy_mobility(
      space_fns, a, xi, eta, P=16, Mgrid=32,
      use_stresslet=True, constrained=True, with_torque=True,
      solve_tol=1e-10, solve_maxiter=60)
  sc = init_c(pos)
  (u_ref, _s_ref, om_ref), _ = apply_c(sc, pos, forces, torques=torques)

  sinit, ssolve = sad.build_saddle_solve(
      space_fns, a, eta, xi=xi, P=16, Mgrid=32,
      gmres_tol=1e-10, gmres_restart=60, gmres_maxiter=4)
  st = sinit(pos)
  u_rel, om_rel, _s5, _q, info = ssolve(
      st, pos, force=forces, torque=torques, zero_nearfield=True)

  assert info['rel_residual'] <= 1e-8
  assert _rel_err(u_rel, u_ref) <= 1e-7
  assert _rel_err(om_rel, om_ref) <= 1e-7


# ---------------------------------------------------------------------------
# Check 1b -- strain path: single-sphere Einstein stresslet
# ---------------------------------------------------------------------------
def test_single_sphere_einstein_stresslet():
  a, eta, xi = 1.0, 1.0, 0.5
  box = jnp.eye(3) * 60.0  # large box -> free-space single sphere
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  pos = jnp.array([[0.5, 0.5, 0.5]])
  E = jnp.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])

  sinit, ssolve = sad.build_saddle_solve(
      space_fns, a, eta, xi=xi, P=20, Mgrid=64,
      gmres_tol=1e-11, gmres_restart=20, gmres_maxiter=4)
  st = sinit(pos)
  u_rel, om_rel, s5, _q, info = ssolve(
      st, pos, E_inf=E, zero_nearfield=True)

  assert info['rel_residual'] <= 1e-8
  assert float(jnp.linalg.norm(u_rel)) <= 1e-9
  assert float(jnp.linalg.norm(om_rel)) <= 1e-9
  s5_expected = decompose_gradient((20.0 / 3.0) * math.pi * eta * a**3 * E)[0]
  # Finite-box periodic image -> few x 1e-5; would vanish as L -> infinity.
  assert _rel_err(s5[0], s5_expected) <= 5e-4


# ---------------------------------------------------------------------------
# Gate B -- near-field block signs, end-to-end through b2
# ---------------------------------------------------------------------------
def _pair_state(a=1.0, eta=1.0, xi=0.5, L=30.0, gap=0.2):
  box = jnp.eye(3) * L
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  d = a + 0.5 * gap  # half centre-to-centre distance (r = 2a + gap)
  cart = jnp.array([[-d, 0.0, 0.0], [d, 0.0, 0.0]])
  pos = cart / L + 0.5
  sinit, ssolve = sad.build_saddle_solve(
      space_fns, a, eta, xi=xi, P=16, Mgrid=48,
      gmres_tol=1e-10, gmres_restart=40, gmres_maxiter=6)
  return sinit(pos), ssolve, pos, jnp.array([1.0, 0.0, 0.0])


def test_compressive_strain_pair_approaches():
  st, ssolve, pos, rhat01 = _pair_state()
  # Compressive along x (E_xx < 0), traceless.
  E = jnp.array([[-1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])

  u_rel, _om, _s5, _q, info = ssolve(st, pos, E_inf=E, zero_nearfield=False)
  u_abs = u_rel + info['U_inf']
  approach = float((u_abs[1] - u_abs[0]) @ rhat01)
  assert approach < 0.0  # particles close the gap

  u_rel2, _om2, _s52, _q2, info2 = ssolve(st, pos, E_inf=-E, zero_nearfield=False)
  u_abs2 = u_rel2 + info2['U_inf']
  separate = float((u_abs2[1] - u_abs2[0]) @ rhat01)
  assert separate > 0.0  # extensional pulls them apart
  assert abs(approach + separate) <= 1e-9  # linear in E


def test_imposed_strain_stresslet_positive():
  st, ssolve, pos, _rhat = _pair_state()
  E = jnp.array([[1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]])
  e5 = decompose_gradient(E)[0]
  _u, _om, s5, _q, _info = ssolve(st, pos, E_inf=E, zero_nearfield=False)
  # S:E = sum over particles of S5 . E5 (orthonormal) must be positive (R_SE PSD).
  s_dot_e = float(jnp.sum(s5 * e5[None, :]))
  assert s_dot_e > 0.0


# ---------------------------------------------------------------------------
# IC(0) preconditioner -- factorization unit tests (fast)
# ---------------------------------------------------------------------------
def test_ic0_dense_equals_cholesky():
  import numpy as _np
  import scipy.sparse as _sp
  rng = _np.random.default_rng(0)
  M = rng.standard_normal((9, 9))
  A = M @ M.T + 9.0 * _np.eye(9)             # SPD, full (dense) pattern
  L = sad._ic0(_sp.csc_matrix(A)).toarray()  # IC(0) == exact Cholesky here
  assert _np.allclose(L @ L.T, A, atol=1e-9)
  assert _np.allclose(L, _np.tril(L))


def test_ic0_preconditioner_inverts_stilde():
  import numpy as _np
  box = _np.eye(3) * 8.0
  # two near-contact pairs within r_p so S~ has off-diagonal coupling
  cart = _np.array([[-1.02, 0, 0], [1.02, 0, 0], [0, 3.0, 0], [0, 3.0, 2.03]])
  stilde = sad.assemble_stilde(cart, box, 1.0, 1.0, r_p=2.1, zeta=6 * math.pi)
  assert stilde.nnz > 6 * cart.shape[0]      # genuine off-diagonal blocks present
  pre = sad.Ic0Preconditioner(stilde, zeta=6 * math.pi)
  rng = _np.random.default_rng(1)
  b = rng.standard_normal(stilde.shape[0])
  x = pre.solve(b)
  # within r_p the pattern is (near) complete per cluster -> near-exact solve.
  assert _np.linalg.norm(stilde @ x - b) / _np.linalg.norm(b) < 1e-8


# ---------------------------------------------------------------------------
# Default preconditioner is IC(0): it must agree with block-Jacobi on the
# solution but reach a tighter residual at an equal (small) iteration budget.
# ---------------------------------------------------------------------------
def test_default_preconditioner_is_ic0():
  a, eta, xi = 1.0, 1.0, 1.0
  N = 8
  L = (N * (4.0 / 3.0 * math.pi * a**3) / 0.3) ** (1.0 / 3.0)
  rng = np.random.default_rng(3)
  xs = (np.arange(2) + 0.5) / 2
  g = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), -1).reshape(-1, 3)
  pos = jnp.asarray((g + 0.03 * rng.standard_normal(g.shape)) % 1.0)
  space_fns = space.periodic_general(jnp.eye(3) * L, fractional_coordinates=True)
  _init, solve = sad.build_saddle_solve(
      space_fns, a, eta, xi=xi, P=12, Mgrid=24,
      gmres_restart=40, gmres_maxiter=2)
  st = _init(pos)
  f = jax.random.normal(jax.random.PRNGKey(7), (N, 3))
  U_d, _, _, _, info_d = solve(st, pos, force=f)                      # default
  U_j, _, _, _, info_j = solve(st, pos, force=f, preconditioner='jacobi')
  # Same physical solution, IC(0) residual strictly tighter at equal budget.
  assert _rel_err(U_d, U_j) < 1e-4
  assert info_d['rel_residual'] < info_j['rel_residual']


# ---------------------------------------------------------------------------
# Check 3 -- preconditioner convergence (Fiore & Swan Fig. 1), eager harness
# ---------------------------------------------------------------------------
def _perturbed_lattice(n, phi, a=1.0, seed=0):
  import numpy as _np
  N = n**3
  L = (N * (4.0 / 3.0 * math.pi * a**3) / phi) ** (1.0 / 3.0)
  xs = (_np.arange(n) + 0.5) / n
  g = _np.stack(_np.meshgrid(xs, xs, xs, indexing='ij'), -1).reshape(-1, 3)
  rng = _np.random.default_rng(seed)
  g = (g + 0.02 * rng.standard_normal(g.shape)) % 1.0
  return jnp.asarray(g), L


@pytest.mark.slow
def test_preconditioner_iteration_count_ordering_and_scaling():
  a, eta, xi, phi = 1.0, 1.0, 1.0, 0.3
  counts = {}
  for n in (3, 4):
    pos, L = _perturbed_lattice(n, phi, a)
    space_fns = space.periodic_general(jnp.eye(3) * L, fractional_coordinates=True)
    _init, solve = sad.build_saddle_solve(
        space_fns, a, eta, xi=xi, P=12, Mgrid=24, gmres_restart=80)
    st = _init(pos)
    f = jax.random.normal(jax.random.PRNGKey(n), (n**3, 3))
    res = {}
    for pc in ('none', 'jacobi', 'ic0'):
      it, rel, diag = solve.count_iterations(
          st, pos, force=f, preconditioner=pc, rtol=1e-6)
      assert rel <= 1e-5
      res[pc] = it
      if pc == 'ic0':
        assert diag['relaxed'] == 0.0          # healthy IC(0), no relaxation
    # Ordering: IC(0) best, then Jacobi, then unpreconditioned.
    assert res['ic0'] <= res['jacobi'] <= res['none']
    # Absolute caps guard the block-LDL sign/scaling: dropping the zeta factors
    # or flipping the Schur sign makes sigma(P A) straddle zero and roughly
    # triples these counts (jacobi ~142/294, ic0 ~108/155 in that regime).
    assert res['jacobi'] < 90
    assert res['ic0'] < 70
    counts[n] = res
  # IC(0) grows sub-linearly in N (here N x2.37).  With the correct block-LDL
  # preconditioner the growth is ~x1.1; the broken-sign version grew ~x1.44.
  growth = counts[4]['ic0'] / counts[3]['ic0']
  assert growth < 1.5
