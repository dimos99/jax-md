"""Validation gates for the Phase-2 stresslet-constrained mobility.

Gates (in increasing stringency; spec Part E):

  A. Decomposition round-trips ``C <-> (S, L)`` / ``D <-> (E, Omega)`` to
     machine precision, and the torque embed matches the rotlet-pinned
     convention ``C = -(1/2) eps . L`` from ``rpy_stresslet_test.py``.
  1. Constraint residual: the solved stresslet, fed back through the
     *unconstrained* Phase-1 grand operator together with the forces, must
     produce vanishing rate of strain E = 0 (this catches embed/extract
     basis mismatches end to end).
  2. M_ES probed densely in orthonormal coordinates is symmetric PSD,
     including a configuration with an overlapping pair.
  3. Dense cross-check: matrix-free constrained velocities match the dense
     Schur complement R_FU^{-1} = M_UF - M_US M_ES^{-1} M_EF formed from
     the probed 11N x 11N grand matrix (with and without applied torques).
  5. Regression: the unconstrained paths are bit-identical with the new
     builder kwargs at their defaults.
  6. xi-invariance of the constrained velocities (composes Phase-1
     invariance with a correct solve).

Gate 4 (slow): short-time translational self-diffusivity D^s(phi) must
decrease with volume fraction -- bare RPY is phi-independent, so the decrease
is the signature of a working constraint -- and match digitized values from
Fiore & Swan 2018 Fig. 7 to +-0.05.
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import partition, space
from jax_md.hydro import rpy
from jax_md.hydro.rpy_constrained import m_es_operator, make_constrained_solver
from jax_md.hydro.rpy_moments import (
    couplet_to_stresslet_torque,
    decompose_gradient,
    stresslet_basis,
    stresslet_to_couplet,
    torque_to_couplet,
    traceless,
    traceless_orthonormal_basis,
)

sys.path.insert(0, 'tests')
import rpy_test_utils as rtu  # pylint: disable=wrong-import-position


def _dtype():
  return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _box(length):
  return jnp.eye(3, dtype=_dtype()) * _dtype()(length)


def _machine_tol():
  return 1e-13 if jax.config.jax_enable_x64 else 1e-5


def _relative_error(actual, expected):
  actual = np.asarray(actual, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15)


def _build_operator(positions, box, *, xi=0.75, a=0.4, eta=1.0, rcut=3.2,
                    Mgrid=10, P=5, **builder_kwargs):
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns,
      a=a,
      xi=xi,
      eta=eta,
      rcut=rcut,
      Mgrid=Mgrid,
      P=P,
      include_brownian=False,
      lattice_extent=1,
      real_space_mode='lattice',
      **builder_kwargs,
  )
  return init_fn(positions), apply_fn


def _grand_matvec_for(positions, box, **op_kwargs):
  """Fixed-state unconstrained grand matvec (F, C) -> (U, D)."""
  state, apply_fn = _build_operator(positions, box, use_stresslet=True,
                                    **op_kwargs)
  def grand_mv(F, C):
    (U, D), _ = apply_fn(state, positions, F, couplets=C)
    return U, D
  return grand_mv


# Dense-block index helpers for the 11N x 11N orthonormal grand matrix:
# per particle, slots 0-2 force/velocity, 3-5 antisymmetric couplet
# (torque channels), 6-10 symmetric couplet (stresslet channels).
def _slot_indices(n_particles, slots):
  return np.concatenate(
      [11 * p + np.asarray(slots, dtype=np.int64) for p in range(n_particles)])


def _dense_blocks(M, n_particles):
  force = _slot_indices(n_particles, range(0, 3))
  antisym = _slot_indices(n_particles, range(3, 6))
  sym = _slot_indices(n_particles, range(6, 11))
  return {
      'UF': M[np.ix_(force, force)],
      'UA': M[np.ix_(force, antisym)],
      'US': M[np.ix_(force, sym)],
      'AF': M[np.ix_(antisym, force)],
      'AA': M[np.ix_(antisym, antisym)],
      'AS': M[np.ix_(antisym, sym)],
      'EF': M[np.ix_(sym, force)],
      'EA': M[np.ix_(sym, antisym)],
      'ES': M[np.ix_(sym, sym)],
  }


# -----------------------------------------------------------------------------
# Gate A: decomposition maps
# -----------------------------------------------------------------------------
def test_decomposition_round_trip():
  rng = np.random.default_rng(0)
  C = traceless(jnp.asarray(rng.normal(size=(9, 3, 3)), dtype=_dtype()))
  S5, L3 = couplet_to_stresslet_torque(C)
  C_back = stresslet_to_couplet(S5) + torque_to_couplet(L3)
  assert _relative_error(C_back, C) < _machine_tol()

  # Other direction: (S5, L3) -> C -> (S5, L3).
  S5_rand = jnp.asarray(rng.normal(size=(9, 5)), dtype=_dtype())
  L3_rand = jnp.asarray(rng.normal(size=(9, 3)), dtype=_dtype())
  S5_back, L3_back = couplet_to_stresslet_torque(
      stresslet_to_couplet(S5_rand) + torque_to_couplet(L3_rand))
  assert _relative_error(S5_back, S5_rand) < _machine_tol()
  assert _relative_error(L3_back, L3_rand) < _machine_tol()

  # decompose_gradient shares the strain projection but its antisymmetric
  # channel is the Frobenius *adjoint* of the embed (half the inverse map):
  # Omega = L/2 when applied to the same tensor.
  E5, Om = decompose_gradient(C)
  np.testing.assert_array_equal(np.asarray(E5), np.asarray(S5))
  assert _relative_error(Om, 0.5 * np.asarray(L3)) < _machine_tol()

  # Basis orthonormality (what makes M_ES symmetric in these coordinates).
  B = np.asarray(stresslet_basis(), dtype=np.float64)
  gram = np.einsum('aij,bij->ab', B, B)
  assert _relative_error(gram, np.eye(5)) < _machine_tol()


def test_torque_channel_matches_rotlet_embed():
  """The torque embed must equal the rotlet-pinned convention C = -eps.L/2.

  ``rpy_stresslet_test.test_antisymmetric_couplet_reproduces_rotlet`` pins
  this sign against external rotlet physics; here we only assert our embed
  reproduces the identical tensor, so the external pin transfers.
  """
  eps = np.zeros((3, 3, 3))
  for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    eps[i, j, k] = 1.0
    eps[j, i, k] = -1.0
  rng = np.random.default_rng(1)
  for _ in range(4):
    torque = rng.normal(size=(3,))
    expected = -0.5 * np.einsum('mnp,p->mn', eps, torque)
    actual = np.asarray(torque_to_couplet(jnp.asarray(torque, dtype=_dtype())))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    # Round trip through the extraction recovers the torque.
    _, L3 = couplet_to_stresslet_torque(jnp.asarray(expected, dtype=_dtype()))
    np.testing.assert_allclose(np.asarray(L3), torque, atol=1e-6)


def test_embed_extract_match_fsd_kernels():
  """Pin the Phase-2 embed/extract maps against the FSD reference source.

  Mechanical transcription of Fiore's FSD ``Helper_Mobility.cu``:
  ``Mobility_TS2C_kernel`` (torque/stresslet -> couplet, including the
  kernel's internal sign flip on the torque) and ``Mobility_D2WE_kernel``
  (velocity gradient -> angular velocity / rate of strain).  FSD stores the
  gradient as ``D_ij = d_i u_j`` -- the transpose of our ``du_i/dx_j`` --
  so the gradient is transposed before feeding the transcribed kernel.
  Constrained-mode equivalence note: with the near-field lubrication
  resistance set to zero, FSD's saddle-point system (``Saddle.cu``,
  ``Saddle_Multiply``) eliminates to exactly the far-field Schur complement
  R_FU^{-1} = M_UF - M_US M_ES^{-1} M_EF solved here, and the underlying
  M^ff kernels were matched to ~2e-15 by the Phase-1 gates.
  """
  rng = np.random.default_rng(20)
  for _ in range(4):
    T = rng.normal(size=3)
    Sxx, Sxy, Sxz, Syz, Syy = rng.normal(size=5)

    # --- Mobility_TS2C_kernel (with its internal L' = -T flip) ---
    Lx, Ly, Lz = -T
    C_fsd = np.array([
        [Sxx, Sxy + 0.5 * Lz, Sxz - 0.5 * Ly],
        [Sxy - 0.5 * Lz, Syy, Syz + 0.5 * Lx],
        [Sxz + 0.5 * Ly, Syz - 0.5 * Lx, -(Sxx + Syy)],
    ])
    S_tensor = np.array([
        [Sxx, Sxy, Sxz],
        [Sxy, Syy, Syz],
        [Sxz, Syz, -(Sxx + Syy)],
    ])
    S5, _ = couplet_to_stresslet_torque(jnp.asarray(S_tensor, dtype=_dtype()))
    C_ours = np.asarray(
        stresslet_to_couplet(S5) +
        torque_to_couplet(jnp.asarray(T, dtype=_dtype())))
    np.testing.assert_allclose(C_ours, C_fsd, atol=1e-6)

    # --- Mobility_D2WE_kernel on a random traceless gradient ---
    D_ours = np.asarray(
        traceless(jnp.asarray(rng.normal(size=(3, 3)), dtype=_dtype())))
    D_fsd = D_ours.T  # FSD stores d_i u_j
    W_fsd = 0.5 * np.array([D_fsd[1, 2] - D_fsd[2, 1],
                            D_fsd[2, 0] - D_fsd[0, 2],
                            D_fsd[0, 1] - D_fsd[1, 0]])
    E_fsd = 0.5 * (D_fsd + D_fsd.T)
    E5, Omega = decompose_gradient(jnp.asarray(D_ours, dtype=_dtype()))
    np.testing.assert_allclose(np.asarray(Omega), W_fsd, atol=1e-6)
    E_ours = np.asarray(stresslet_to_couplet(E5))
    np.testing.assert_allclose(E_ours, E_fsd, atol=1e-6)


# -----------------------------------------------------------------------------
# Gate 1: constraint residual / E = 0 feedback through the grand operator
# -----------------------------------------------------------------------------
def test_constraint_residual_and_zero_strain():
  box = _box(10.0)
  positions = rtu._nonoverlap_positions(5, np.asarray(box), a=0.4, seed=2)
  rng = np.random.default_rng(3)
  forces = jnp.asarray(rng.normal(size=(5, 3)), dtype=_dtype())

  solve_tol = 1e-8 if jax.config.jax_enable_x64 else 1e-5
  state, apply_fn = _build_operator(
      positions, box, use_stresslet=True, constrained=True,
      solve_tol=solve_tol, solve_maxiter=50)
  (U, S), _ = apply_fn(state, positions, forces)

  # Feed (F, S) back through the *unconstrained* grand operator: the rate of
  # strain must vanish and the velocities must coincide.
  grand_mv = _grand_matvec_for(positions, box)
  U_grand, D_grand = grand_mv(forces, S)
  E = 0.5 * (D_grand + jnp.swapaxes(D_grand, -1, -2))
  rel_strain = float(jnp.linalg.norm(E) / jnp.linalg.norm(D_grand))
  assert rel_strain < 5 * solve_tol
  assert _relative_error(U_grand, U) < 5 * solve_tol

  # The returned stresslet is symmetric traceless.
  assert _relative_error(S, 0.5 * (S + jnp.swapaxes(S, -1, -2))) < _machine_tol()
  assert float(jnp.max(jnp.abs(jnp.trace(S, axis1=-2, axis2=-1)))) < 1e-6


# -----------------------------------------------------------------------------
# Gate 2: M_ES symmetry / PSD in orthonormal coordinates
# -----------------------------------------------------------------------------
@pytest.mark.parametrize('config', ['nonoverlap', 'overlap'])
def test_m_es_symmetry_psd_dense(config):
  box = _box(8.0)
  if config == 'nonoverlap':
    positions = rtu._nonoverlap_positions(4, np.asarray(box), a=0.4, seed=4)
  else:
    positions = rtu._random_positions_with_overlaps(4, np.asarray(box), a=0.4,
                                                    seed=5)
  n = int(positions.shape[0])
  m_es = m_es_operator(_grand_matvec_for(positions, box))

  dense = np.zeros((5 * n, 5 * n), dtype=np.float64)
  for col in range(5 * n):
    S5 = np.zeros((n, 5), dtype=np.float64)
    S5[col // 5, col % 5] = 1.0
    dense[:, col] = np.asarray(
        m_es(jnp.asarray(S5, dtype=_dtype())), dtype=np.float64).reshape(-1)

  sym_tol = 2e-7 if jax.config.jax_enable_x64 else 2e-3
  psd_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-3
  scale = max(np.linalg.norm(dense), 1e-15)
  assert np.linalg.norm(dense - dense.T) / scale < sym_tol
  eigvals = np.linalg.eigvalsh(0.5 * (dense + dense.T))
  assert eigvals.min() >= -psd_tol * max(np.max(np.abs(eigvals)), 1.0)


# -----------------------------------------------------------------------------
# Gate 3: dense cross-check of R_FU^{-1}
# -----------------------------------------------------------------------------
def test_constrained_vs_dense_resistance_inverse():
  box = _box(10.0)
  n = 5
  positions = rtu._nonoverlap_positions(n, np.asarray(box), a=0.4, seed=6)
  rng = np.random.default_rng(7)
  forces = jnp.asarray(rng.normal(size=(n, 3)), dtype=_dtype())

  grand_mv = _grand_matvec_for(positions, box)
  dense = rtu._grand_dense_from_matvec(grand_mv, n)
  blocks = _dense_blocks(dense, n)
  R_inv = blocks['UF'] - blocks['US'] @ np.linalg.solve(blocks['ES'],
                                                        blocks['EF'])
  U_ref = (R_inv @ np.asarray(forces, dtype=np.float64).reshape(-1)).reshape(n, 3)

  solve_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-5
  state, apply_fn = _build_operator(
      positions, box, use_stresslet=True, constrained=True,
      solve_tol=solve_tol, solve_maxiter=60)
  (U, S), _ = apply_fn(state, positions, forces)
  tol = 1e-8 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(U, U_ref) < tol

  # The solved stresslet matches the dense constraint solve.
  S5_ref = np.linalg.solve(
      blocks['ES'],
      -blocks['EF'] @ np.asarray(forces, dtype=np.float64).reshape(-1))
  S5, _ = couplet_to_stresslet_torque(S)
  assert _relative_error(np.asarray(S5).reshape(-1), S5_ref) < tol


def test_constrained_with_torque_vs_dense():
  box = _box(10.0)
  n = 4
  positions = rtu._nonoverlap_positions(n, np.asarray(box), a=0.4, seed=8)
  rng = np.random.default_rng(9)
  forces = jnp.asarray(rng.normal(size=(n, 3)), dtype=_dtype())
  torques = jnp.asarray(rng.normal(size=(n, 3)), dtype=_dtype())

  grand_mv = _grand_matvec_for(positions, box)
  dense = rtu._grand_dense_from_matvec(grand_mv, n)

  # Dense reference, run through the *same* embed/extract maps (which are
  # independently pinned by the round-trip and rotlet gates): build the 11N
  # input vector from (F, L), solve the sym-block constraint, assemble.
  basis = np.asarray(traceless_orthonormal_basis(), dtype=np.float64)
  C_applied = np.asarray(torque_to_couplet(torques), dtype=np.float64)
  coords_in = np.zeros((n, 11))
  coords_in[:, :3] = np.asarray(forces, dtype=np.float64)
  coords_in[:, 3:] = np.einsum('nij,aij->na', C_applied, basis)
  out_applied = dense @ coords_in.reshape(-1)

  blocks = _dense_blocks(dense, n)
  sym = _slot_indices(n, range(6, 11))
  force_idx = _slot_indices(n, range(0, 3))
  antisym = _slot_indices(n, range(3, 6))
  S5_ref = np.linalg.solve(blocks['ES'], -out_applied[sym])
  U_ref = (out_applied[force_idx] + blocks['US'] @ S5_ref).reshape(n, 3)
  # Angular velocity: antisym output coords -> tensor -> Omega extraction.
  d_anti = out_applied[antisym] + blocks['AS'] @ S5_ref
  D_anti = np.einsum('na,aij->nij', d_anti.reshape(n, 3), basis[:3])
  Omega_ref = np.asarray(
      decompose_gradient(jnp.asarray(D_anti, dtype=_dtype()))[1])

  solve_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-5
  state, apply_fn = _build_operator(
      positions, box, use_stresslet=True, constrained=True, with_torque=True,
      solve_tol=solve_tol, solve_maxiter=60)
  (U, S, Omega), _ = apply_fn(state, positions, forces, torques=torques)
  tol = 1e-8 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(U, U_ref) < tol
  assert _relative_error(Omega, Omega_ref) < tol
  S5, _ = couplet_to_stresslet_torque(S)
  assert _relative_error(np.asarray(S5).reshape(-1), S5_ref) < tol


# -----------------------------------------------------------------------------
# Gate 5: regression -- unconstrained paths unchanged
# -----------------------------------------------------------------------------
def test_unconstrained_regression_bit_identical():
  box = _box(10.0)
  positions = rtu._nonoverlap_positions(4, np.asarray(box), a=0.4, seed=10)
  rng = np.random.default_rng(11)
  forces = jnp.asarray(rng.normal(size=(4, 3)), dtype=_dtype())
  couplets = traceless(jnp.asarray(rng.normal(size=(4, 3, 3)), dtype=_dtype()))

  # Legacy RPY path with explicit constrained=False vs default kwargs.
  state_a, apply_a = _build_operator(positions, box)
  state_b, apply_b = _build_operator(positions, box, constrained=False)
  U_a, _ = apply_a(state_a, positions, forces)
  U_b, _ = apply_b(state_b, positions, forces)
  np.testing.assert_array_equal(np.asarray(U_a), np.asarray(U_b))

  # Unconstrained grand path likewise.
  state_c, apply_c = _build_operator(positions, box, use_stresslet=True)
  state_d, apply_d = _build_operator(positions, box, use_stresslet=True,
                                     constrained=False)
  (U_c, D_c), _ = apply_c(state_c, positions, forces, couplets=couplets)
  (U_d, D_d), _ = apply_d(state_d, positions, forces, couplets=couplets)
  np.testing.assert_array_equal(np.asarray(U_c), np.asarray(U_d))
  np.testing.assert_array_equal(np.asarray(D_c), np.asarray(D_d))


def test_constrained_mode_input_validation():
  box = _box(10.0)
  positions = rtu._nonoverlap_positions(3, np.asarray(box), a=0.4, seed=12)
  forces = jnp.zeros((3, 3), dtype=_dtype())
  couplets = jnp.zeros((3, 3, 3), dtype=_dtype())
  torques = jnp.zeros((3, 3), dtype=_dtype())

  with pytest.raises(ValueError):
    _build_operator(positions, box, constrained=True)  # needs use_stresslet
  with pytest.raises(ValueError):
    _build_operator(positions, box, use_stresslet=True, with_torque=True)

  state, apply_fn = _build_operator(positions, box, use_stresslet=True,
                                    constrained=True)
  with pytest.raises(ValueError):
    apply_fn(state, positions, forces, couplets=couplets)
  with pytest.raises(ValueError):
    apply_fn(state, positions, forces, torques=torques)  # needs with_torque

  state_g, apply_g = _build_operator(positions, box, use_stresslet=True)
  with pytest.raises(ValueError):
    apply_g(state_g, positions, forces, torques=torques)
  with pytest.raises(ValueError):
    apply_g(state_g, positions, forces, stresslet_guess=couplets)


def test_warm_start_and_jit():
  box = _box(10.0)
  n = 4
  positions = rtu._nonoverlap_positions(n, np.asarray(box), a=0.4, seed=13)
  rng = np.random.default_rng(14)
  forces = jnp.asarray(rng.normal(size=(n, 3)), dtype=_dtype())

  solve_tol = 1e-9 if jax.config.jax_enable_x64 else 1e-5
  state, apply_fn = _build_operator(
      positions, box, use_stresslet=True, constrained=True,
      solve_tol=solve_tol, solve_maxiter=50)
  (U_cold, S_cold), _ = apply_fn(state, positions, forces)

  # Warm start from the converged stresslet reproduces the same solution.
  (U_warm, S_warm), _ = apply_fn(state, positions, forces,
                                 stresslet_guess=S_cold)
  tol = 1e-7 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(U_warm, U_cold) < tol
  assert _relative_error(S_warm, S_cold) < tol

  # The constrained apply composes with jit.
  jitted = jax.jit(lambda s, x, f, g: apply_fn(s, x, f, stresslet_guess=g))
  (U_jit, S_jit), _ = jitted(state, positions, forces,
                             jnp.zeros((n, 5), dtype=_dtype()))
  assert _relative_error(U_jit, U_cold) < tol
  assert _relative_error(S_jit, S_cold) < tol


# -----------------------------------------------------------------------------
# Gate 6: xi-invariance of the constrained mobility
# -----------------------------------------------------------------------------
@pytest.mark.slow
def test_constrained_xi_invariance():
  positions = jnp.asarray([[0.1, 0.1, 0.1],
                           [0.52, 0.18, 0.12],
                           [0.15, 0.55, 0.6],
                           [0.6, 0.62, 0.55]], dtype=_dtype())
  box = _box(6.0)
  rng = np.random.default_rng(15)
  forces = jnp.asarray(rng.normal(size=(4, 3)), dtype=_dtype())
  torques = jnp.asarray(rng.normal(size=(4, 3)), dtype=_dtype())

  outputs = []
  for xi in (0.5, 0.75, 1.0):
    state, apply_fn = _build_operator(
        positions, box, a=1.0, xi=xi, rcut=min(5.5 / xi, 11.0),
        Mgrid=28, P=11, use_stresslet=True, constrained=True,
        with_torque=True, solve_tol=1e-10, solve_maxiter=60)
    (U, S, Omega), _ = apply_fn(state, positions, forces, torques=torques)
    outputs.append((np.asarray(U), np.asarray(S), np.asarray(Omega)))

  tol = 1e-5 if jax.config.jax_enable_x64 else 5e-3
  for U, S, Omega in outputs[1:]:
    assert _relative_error(U, outputs[0][0]) < tol
    assert _relative_error(S, outputs[0][1]) < tol
    assert _relative_error(Omega, outputs[0][2]) < tol


@pytest.mark.slow
def test_shear_constrained_gamma_zero_matches_static():
  """Live-box constrained path at gamma = 0 equals the static-box path."""
  box = _box(6.0)
  positions = rtu._nonoverlap_positions(4, np.asarray(box), a=0.4, seed=16)
  rng = np.random.default_rng(17)
  forces = jnp.asarray(rng.normal(size=(4, 3)), dtype=_dtype())

  def _shear_box_fn(**kwargs):
    gamma = kwargs.get('gamma_xy', 0.0)
    deformed = np.eye(3)
    deformed[0, 1] = gamma
    return jnp.asarray(deformed, dtype=_dtype()) @ box

  static_space = space.periodic_general(box, fractional_coordinates=True)
  shear_space = static_space + (_shear_box_fn,)

  common = dict(a=0.4, xi=0.75, eta=1.0, rcut=3.2, Mgrid=12, P=5,
                include_brownian=False, lattice_extent=1,
                real_space_mode='lattice', use_stresslet=True,
                constrained=True, solve_tol=1e-9, solve_maxiter=50)
  init_s, apply_s = rpy.build_rpy_mobility(static_space, **common)
  init_l, apply_l = rpy.build_rpy_mobility(shear_space, **common)

  (U_s, S_s), _ = apply_s(init_s(positions), positions, forces)
  (U_l, S_l), _ = apply_l(init_l(positions, gamma_xy=0.0), positions, forces,
                          gamma_xy=0.0)
  tol = 1e-7 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(U_l, U_s) < tol
  assert _relative_error(S_l, S_s) < tol


# -----------------------------------------------------------------------------
# Gate 4: short-time self-diffusivity vs Fiore & Swan 2018 Fig. 7
# -----------------------------------------------------------------------------
# Digitized from Fiore & Swan, J. Chem. Phys. 148, 044114 (2018), Fig. 7
# (translational short-time self-diffusivity of the stresslet-constrained
# RPY model; digitization supplied 2026-06-11, stated accuracy +-0.05).
_FIORE_SWAN_FIG7 = (
    (0.05048479131550872, 0.9299455572675912),
    (0.1, 0.8560472521828454),
    (0.1500518038566812, 0.7976271186440678),
    (0.19977732073546015, 0.73628145865434),
    (0.2498662378028979, 0.6853600410888546),
    (0.29979742372462, 0.6357452491011812),
)


def _hs_relax(positions, box_matrix, a, *, sweeps, seed):
  """Decorrelate an RSA configuration with hard-sphere displacement MC."""
  rng = np.random.default_rng(seed)
  pos = np.asarray(positions, dtype=np.float64).copy()
  n = pos.shape[0]
  box = np.asarray(box_matrix, dtype=np.float64)
  L = float(box[0, 0])
  step = 0.4 * a / L
  min_dist = 2.0 * a
  for _ in range(sweeps):
    for i in range(n):
      trial = np.mod(pos[i] + rng.uniform(-step, step, size=3), 1.0)
      delta = trial - np.delete(pos, i, axis=0)
      delta -= np.round(delta)
      r = np.linalg.norm(delta @ box.T, axis=1)
      if np.all(r >= min_dist):
        pos[i] = trial
  return jnp.asarray(pos, dtype=_dtype())


def _self_diffusivity_at(phi, n, *, a, eta, n_configs, n_probes):
  """Finite-N short-time self-diffusivity D^s_N / D_0 (variance-reduced).

  The bare RPY diagonal of M_UF in a periodic box is configuration
  independent (self term + periodic self-images only), so it is computed
  exactly with one unconstrained matvec; only the constraint correction
  ``z . (R_FU^{-1} - M_UF) z`` (which decays fast and has small variance)
  is estimated by Hutchinson probing with Rademacher forces over
  equilibrated hard-sphere configurations (RSA + displacement MC).
  """
  mu0 = 1.0 / (6.0 * math.pi * eta * a)
  L = (4.0 * math.pi * n * a ** 3 / (3.0 * phi)) ** (1.0 / 3.0)
  box = _box(L)
  # Keep the split error ~1e-7 at every phi: rcut pinned to the box,
  # xi from rcut, grid from kcut ~ 9.6 xi (Fiore 2017 error bounds).
  rcut = 0.42 * L
  xi = 5.5 / rcut
  Mgrid = 2 * int(math.ceil(9.6 * xi * L / (2.0 * math.pi))) + 4
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns, a=a, xi=xi, eta=eta, rcut=rcut, Mgrid=Mgrid, P=11,
      include_brownian=False, use_stresslet=True, constrained=True,
      solve_tol=1e-6, solve_maxiter=60)
  init_rpy, apply_rpy = rpy.build_rpy_mobility(
      space_fns, a=a, xi=xi, eta=eta, rcut=rcut, Mgrid=Mgrid, P=11,
      include_brownian=False)

  mu_self = None
  corr_samples = []
  for cfg in range(n_configs):
    seed = 1000 * cfg + int(1e4 * phi)
    positions = rtu._nonoverlap_positions(
        n, np.asarray(box), a=a, seed=seed, clearance=0.005)
    positions = _hs_relax(positions, np.asarray(box), a,
                          sweeps=40, seed=seed + 1)
    state = init_fn(positions)
    state_rpy = init_rpy(positions)
    if mu_self is None:
      probe = jnp.zeros((n, 3), dtype=_dtype()).at[0, 0].set(1.0)
      U_self, state_rpy = apply_rpy(state_rpy, positions, probe)
      mu_self = float(U_self[0, 0])
    rng = np.random.default_rng(seed + 2)
    for _ in range(n_probes):
      z = jnp.asarray(rng.choice((-1.0, 1.0), size=(n, 3)), dtype=_dtype())
      (U_con, _), state = apply_fn(state, positions, z)
      U_unc, state_rpy = apply_rpy(state_rpy, positions, z)
      corr_samples.append(float(jnp.vdot(z, U_con - U_unc)))
  return mu_self / mu0 + np.mean(corr_samples) / (3.0 * n * mu0)


@pytest.mark.slow
def test_self_diffusivity_fiore_swan():
  """Constrained D^s(phi) decreases and matches Fiore-Swan Fig. 7 to +-0.05.

  Bare RPY has a phi-independent self-mobility diagonal, so the decrease
  with phi is the definitive signature that the stresslet constraint is
  being enforced.  The periodic-box D^s_N carries the O((phi/N)^{1/3})
  finite-size deficit from image hydrodynamics; following the established
  procedure (Ladd, J. Chem. Phys. 93, 3484 (1990); Banchio et al.,
  J. Chem. Phys. 148, 134902 (2018), Eq. 46), D^s_N is computed at two
  system sizes and extrapolated linearly in N^{-1/3} to the infinite
  system, which requires no high-frequency-viscosity input.
  """
  if not jax.config.jax_enable_x64:
    pytest.skip('needs float64 for a converged Ewald split')

  a = 1.0
  eta = 1.0
  sizes = (32, 108)
  x = np.asarray([n ** (-1.0 / 3.0) for n in sizes])

  results = []
  for phi, _ in _FIORE_SWAN_FIG7:
    d_n = np.asarray([
        _self_diffusivity_at(phi, n, a=a, eta=eta, n_configs=2, n_probes=6)
        for n in sizes
    ])
    # Linear extrapolation in N^{-1/3} to x = 0.
    slope = (d_n[1] - d_n[0]) / (x[1] - x[0])
    results.append(d_n[0] - slope * x[0])

  # Monotonic decrease with phi -- the constraint signature.
  for lo, hi in zip(results[1:], results[:-1]):
    assert lo < hi, results
  # Quantitative pin against the digitized Fiore-Swan curve.
  for (phi, ref), value in zip(_FIORE_SWAN_FIG7, results):
    assert abs(value - ref) <= 0.05, (phi, value, ref)
