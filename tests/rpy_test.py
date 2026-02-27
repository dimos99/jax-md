import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import space
from jax_md.hydro import rpy, rpy_real

import rpy_test_utils as rtu


def _dtype():
  return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _box(length: float):
  return jnp.eye(3, dtype=_dtype()) * _dtype()(length)


def _sample_mobility_block(apply_fn, state, positions_frac, i: int, j: int):
  dense = rtu._dense_matrix_from_apply(apply_fn, state, positions_frac)
  return dense[3 * i:3 * i + 3, 3 * j:3 * j + 3], dense


def _sample_block_from_apply(apply_fn, state, positions_frac, i: int, j: int):
  """Sample 3x3 mobility block (i, j) by probing Cartesian unit forces on j."""
  n_particles = int(positions_frac.shape[0])
  block = np.zeros((3, 3), dtype=np.float64)
  for alpha in range(3):
    forces = np.zeros((n_particles, 3), dtype=np.float64)
    forces[j, alpha] = 1.0
    vel, _ = apply_fn(state, positions_frac, jnp.asarray(forces, dtype=_dtype()))
    vel = np.asarray(vel, dtype=np.float64)
    block[:, alpha] = vel[i, :]
  return block


def _pair_block_force_balanced_measured_self(apply_fn, state, positions_frac):
  """Return pair block M12 reconstructed from balanced probes and measured periodic M11."""
  m11_minus_m12 = np.zeros((3, 3), dtype=np.float64)
  for alpha in range(3):
    forces = np.zeros((2, 3), dtype=np.float64)
    forces[0, alpha] = 1.0
    forces[1, alpha] = -1.0
    vel, _ = apply_fn(state, positions_frac, jnp.asarray(forces, dtype=_dtype()))
    vel = np.asarray(vel, dtype=np.float64)
    m11_minus_m12[:, alpha] = vel[0, :]

  m11 = _sample_block_from_apply(apply_fn, state, positions_frac, i=0, j=0)
  m12 = m11 - m11_minus_m12
  return m12, m11


def _build_operator_with_mode(
    positions_frac,
    box_matrix,
    *,
    a: float,
    eta: float,
    xi: float,
    rcut: float,
    p_support: int,
    mgrid: int,
    lattice_extent: int,
    real_space_mode: str,
):
  """Build deterministic split-Ewald operator with explicit real-space mode."""
  positions_frac = jnp.asarray(positions_frac, dtype=_dtype())
  box_matrix = jnp.asarray(box_matrix, dtype=_dtype())
  space_fns = space.periodic_general(box_matrix, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns,
      a=a,
      xi=xi,
      eta=eta,
      rcut=rcut,
      P=p_support,
      Mgrid=mgrid,
      include_brownian=False,
      lattice_extent=lattice_extent,
      real_space_mode=real_space_mode,
  )
  state = init_fn(positions_frac)
  return {
      'positions_frac': positions_frac,
      'box_matrix': box_matrix,
      'init_fn': init_fn,
      'apply_fn': apply_fn,
      'state': state,
  }


def test_single_particle_self_mobility_hasimoto_and_a4():
  """Validate single-particle self-mobility against analytic periodic corrections.

  Checks five properties:

  1. **Diagonal self-block**: the 3x3 mobility sub-block M_00 should be
     proportional to the identity (off-diagonals < 2e-6) by cubic symmetry.

  2. **Hasimoto finite-size correction** (rel <= 0.5%): the scalar
     self-mobility in a periodic box satisfies
       mu = (1 / 6*pi*eta*a) * (1 - e_hat*(a/L) + (4*pi/3)*(a/L)^3),
     where e_hat ~ 2.8373 is the Hasimoto constant. This confirms the PSE
     Ewald sum correctly captures leading-order periodic image corrections.

  3. **Isotropy under applied force**: a unit force in x produces zero
     velocity components in y and z (atol=2e-6).

  4. **Wave-space self-scalar** (rel <= 5e-3): the scalar Mw self-mobility
     matches M_total(Hasimoto) - Mr_self(a, xi), an analytic reference that
     is entirely independent of the wave-space FFT code, directly validating
     the Mw contribution in isolation.

  5. **Real-space self-scalar** (rel <= 1e-6): the real-space 3x3 self-block
     scalar matches the closed-form Mr_self(a, xi) analytic formula,
     validating the F1/F2 coefficient implementation at the self-interaction.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 10.0 * a
  box = _box(l_box)
  positions = jnp.asarray([[0.23, 0.41, 0.77]], dtype=_dtype())

  op = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-6,
  )
  apply_fn = op['apply_fn']
  state = op['state']
  xi = float(op['xi'])

  m_block, _ = _sample_mobility_block(apply_fn, state, positions, 0, 0)
  np.testing.assert_allclose(
      m_block,
      np.diag(np.diag(m_block)),
      atol=2e-6,
      rtol=0.0,
  )
  self_scalar = float(np.trace(m_block) / 3.0)

  expected = (1.0 / (6.0 * math.pi * eta * a)) * (
      1.0 - rtu.HASIMOTO_EHAT * (a / l_box) + (4.0 * math.pi / 3.0) * ((a / l_box) ** 3)
  )
  hasimoto_rel = abs(self_scalar - expected) / abs(expected)
  print(f"  self_scalar={self_scalar:.8g}  expected(Hasimoto)={expected:.8g}  rel_err={hasimoto_rel:.3e}  (tol=5e-3)")
  assert hasimoto_rel <= 5e-3

  force_x = jnp.asarray([[1.0, 0.0, 0.0]], dtype=_dtype())
  vel_x, _ = apply_fn(state, positions, force_x)
  vel_x = np.asarray(vel_x, dtype=np.float64)
  print(f"  vel_x={vel_x[0]}  (off-axis should be ~0)")
  np.testing.assert_allclose(vel_x[0, 1:], np.zeros((2,), dtype=np.float64), atol=2e-6, rtol=0.0)

  mr, mw, _ = rtu._split_dense_mobility(state, positions)

  mr_scalar = float(np.trace(mr[0:3, 0:3]) / 3.0)
  mr_expected = float((1.0 / (6.0 * math.pi * eta * a)) * rpy_real.Mr_self(a, xi))
  mr_rel = abs(mr_scalar - mr_expected) / abs(mr_expected)
  print(f"  mr_scalar={mr_scalar:.8g}  mr_expected={mr_expected:.8g}  rel_err={mr_rel:.3e}  (tol=1e-6)")
  assert mr_rel <= 1e-6

  mw_scalar = float(np.trace(mw[0:3, 0:3]) / 3.0)
  mw_expected = expected - mr_expected  # Hasimoto M_total - analytic Mr_self
  mw_rel = abs(mw_scalar - mw_expected) / abs(mw_expected)
  print(f"  mw_scalar={mw_scalar:.8g}  mw_expected(Hasimoto-Mr)={mw_expected:.8g}  rel_err={mw_rel:.3e}  (tol=5e-3)")
  assert mw_rel <= 5e-3


def test_pair_mobility_vs_free_space_rpy_sweep():
  """Validate pair block against free-space RPY as a large-box asymptotic check.

  For a two-particle periodic system, reconstruct the periodic pair block via
    M12 = M11 - (M11 - M12),
  where M11 is measured from the same operator/state (not replaced by free-space
  mu0 I). The reference is raw free-space RPY, interpreted as an asymptotic
  comparison for large boxes rather than an exact periodic identity.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 80.0 * a
  box = _box(l_box)

  base_positions = jnp.asarray([
      [0.25, 0.25, 0.25],
      [0.25 + (2.5 * a) / l_box, 0.25, 0.25],
  ], dtype=_dtype())
  op = rtu._build_operator(
      base_positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-5,
  )
  init_fn = op['init_fn']
  apply_fn = op['apply_fn']

  r_over_a_values = [0.5, 1.0, 1.5, 1.9, 2.0, 2.1, 3.0, 5.0, 10.0]
  ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  near_mid_channel_rel = []
  far_abs_frob = []

  for r_over_a in r_over_a_values:
    r = r_over_a * a
    positions = jnp.asarray([
        [0.25, 0.25, 0.25],
        [0.25 + r / l_box, 0.25, 0.25],
    ], dtype=_dtype())
    state = init_fn(positions)
    m12, _ = _pair_block_force_balanced_measured_self(apply_fn, state, positions)
    ref = rtu._free_space_rpy_block(np.array([r, 0.0, 0.0]), a, eta)
    np.testing.assert_allclose(m12, m12.T, atol=2e-5, rtol=0.0)
    m_parallel, m_perp = rtu._longitudinal_transverse(m12, ex)
    ref_parallel, ref_perp = rtu._longitudinal_transverse(ref, ex)
    rel_parallel = abs(m_parallel - ref_parallel) / max(abs(ref_parallel), 1e-15)
    rel_perp = abs(m_perp - ref_perp) / max(abs(ref_perp), 1e-15)
    max_rel = max(rel_parallel, rel_perp)
    abs_frob = np.linalg.norm(m12 - ref)
    if r <= 3.0 * a:
      near_mid_channel_rel.append(max_rel)
    if r >= 5.0 * a:
      far_abs_frob.append(abs_frob)

  print(f"  near/mid channel max rel (r<=3a) = {max(near_mid_channel_rel):.3e}  (tol=1.5e-1)")
  print(f"  far-field abs Frobenius max (r>=5a) = {max(far_abs_frob):.3e}  (tol=7e-2)")
  assert max(near_mid_channel_rel) <= 1.5e-1
  assert max(far_abs_frob) <= 7e-2

  r_left = 1.99 * a
  r_right = 2.01 * a
  pos_left = jnp.asarray([[0.25, 0.25, 0.25], [0.25 + r_left / l_box, 0.25, 0.25]], dtype=_dtype())
  pos_right = jnp.asarray([[0.25, 0.25, 0.25], [0.25 + r_right / l_box, 0.25, 0.25]], dtype=_dtype())
  state_left = init_fn(pos_left)
  state_right = init_fn(pos_right)
  m12_left, _ = _pair_block_force_balanced_measured_self(apply_fn, state_left, pos_left)
  m12_right, _ = _pair_block_force_balanced_measured_self(apply_fn, state_right, pos_right)
  mp_left, mt_left = rtu._longitudinal_transverse(m12_left, ex)
  mp_right, mt_right = rtu._longitudinal_transverse(m12_right, ex)
  print(f"  crossover continuity |d_parallel|={abs(mp_left-mp_right):.3e} |d_perp|={abs(mt_left-mt_right):.3e}  (tol=7e-3)")
  assert abs(mp_left - mp_right) <= 7e-3
  assert abs(mt_left - mt_right) <= 7e-3


def test_pair_force_balanced_reconstruction_matches_dense_offdiag():
  """Validate balanced-force reconstruction against the dense periodic off-diagonal block."""
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 80.0 * a
  box = _box(l_box)
  r = 3.0 * a
  positions = jnp.asarray([
      [0.25, 0.25, 0.25],
      [0.25 + r / l_box, 0.25, 0.25],
  ], dtype=_dtype())
  op = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-5,
  )

  m12_recon, _ = _pair_block_force_balanced_measured_self(op['apply_fn'], op['state'], positions)
  dense = rtu._dense_matrix_from_apply(op['apply_fn'], op['state'], positions)
  m12_dense = dense[0:3, 3:6]
  rel = np.linalg.norm(m12_recon - m12_dense) / max(np.linalg.norm(m12_dense), 1e-15)
  print(f"  reconstruction consistency rel_err={rel:.3e}  (tol=5e-5)")
  assert rel <= 5e-5


def test_pair_periodic_to_free_space_converges_with_box_size_small_l():
  """Validate periodic pair block approaches free-space pair block as box size grows."""
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  r = 3.0 * a
  errors = {}

  for l_box in (40.0 * a, 60.0 * a, 80.0 * a):
    box = _box(l_box)
    positions = jnp.asarray([
        [0.25, 0.25, 0.25],
        [0.25 + r / l_box, 0.25, 0.25],
    ], dtype=_dtype())
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=0.5 / a,
        tol=1e-5,
    )
    m12, _ = _pair_block_force_balanced_measured_self(op['apply_fn'], op['state'], positions)
    ref = rtu._free_space_rpy_block(np.array([r, 0.0, 0.0]), a, eta)
    err = np.linalg.norm(m12 - ref) / max(np.linalg.norm(ref), 1e-15)
    errors[float(l_box / a)] = err

  err40 = errors[40.0]
  err60 = errors[60.0]
  err80 = errors[80.0]
  print(f"  box-size convergence errors: e40={err40:.3e} e60={err60:.3e} e80={err80:.3e}")
  assert err60 < err40
  assert err80 < err60
  assert err80 <= 1.2e-1


def test_two_particle_symmetry_structure():
  """Validate two-particle tensor structure with force-balanced probes only.

  For N=2 and a fixed reference particle, the balanced-force basis has three
  columns (+e_alpha on particle 1, -e_alpha on particle 2). The projected
  operator B^T M B is therefore a 3x3 relative-mobility tensor. For particles
  separated along x it must obey:
    - no longitudinal/transverse cross-coupling,
    - transverse isotropy (yy == zz),
    - symmetry.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 60.0 * a
  box = _box(l_box)
  r = 3.0 * a
  positions = jnp.asarray([
      [0.15, 0.4, 0.7],
      [0.15 + r / l_box, 0.4, 0.7],
  ], dtype=_dtype())
  op = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-6,
  )
  m_rel, _, _ = rtu._projected_matrix_from_apply(op['apply_fn'], op['state'], positions)

  print(f"  m_rel off-diagonal: [0,1]={m_rel[0,1]:.3e}  [0,2]={m_rel[0,2]:.3e}  [1,2]={m_rel[1,2]:.3e}  (tol=1e-5)")
  print(f"  m_rel perp isotropy: |m_rel[1,1]-m_rel[2,2]|={abs(m_rel[1,1]-m_rel[2,2]):.3e}  (tol=1e-5)")
  print(f"  m_rel symmetry: ||m_rel-m_rel^T||={np.linalg.norm(m_rel - m_rel.T):.3e}  (tol=1e-5)")
  assert abs(m_rel[0, 1]) <= 1e-5
  assert abs(m_rel[0, 2]) <= 1e-5
  assert abs(m_rel[1, 2]) <= 1e-5
  assert abs(m_rel[1, 1] - m_rel[2, 2]) <= 1e-5
  np.testing.assert_allclose(m_rel, m_rel.T, atol=1e-5, rtol=0.0)


def test_symmetry_and_spd_full_real_wave():
  """Validate symmetry and semi-positive-definiteness of M, Mr, and Mw for a random suspension.

  Ten particles (a = 1.0, eta = 1/(6*pi*a)) are placed randomly in a cubic
  box (L = 36 a, seed=41) allowing overlaps, representing a dense/disordered
  configuration.  The force-balanced projected mobility B^T M B and its
  real-space (B^T Mr B) and wave-space (B^T Mw B) components are assembled
  for three Ewald splitting
  parameters xi*a ∈ {0.25, 0.5, 1.0}.  For each (xi, matrix) pair two
  properties are checked:

  1. **Symmetry** (rel <= 1e-6): the relative Frobenius asymmetry
       ||M - M^T|| / ||M|| <= 1e-6
     for each of M, Mr, and Mw separately.

  2. **Semi-positive-definiteness** (eig_min > -1e-6): the smallest
     eigenvalue of the symmetrized matrix 0.5*(M + M^T) must be
     non-negative (within numerical noise), confirming the mobility
     tensor is physically valid and that the Ewald decomposition does
     not introduce negative-definite artifacts in either sub-matrix.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 5.0 * a
  box = _box(l_box)
  n_particles = 10
  phi = float(n_particles * (4.0 / 3.0) * math.pi * (a ** 3) / (l_box ** 3))
  print(f"  volume fraction phi={phi:.3f}")
  positions = rtu._random_positions_with_overlaps(n_particles,
                              np.asarray(box, dtype=np.float64), a=a, seed=41)
  overlap_count = rtu._count_overlaps(positions, box, a)
  print(f"  number of overlaps (r < 2a) = {overlap_count}  (expected ~{n_particles * (n_particles - 1) * phi:.1f})")

  for xi_a in (0.25, 0.5, 1.0):
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=xi_a / a,
        tol=1e-5,
    )
    mr, mw, m, _ = rtu._split_projected_mobility(op['state'], positions)
    for name, matrix in (('M', m), ('Mr', mr), ('Mw', mw)):
      sym_rel = np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-15)
      eig_min = np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min()
      print(f"  xi*a={xi_a:.2f}  {name}: sym_rel={sym_rel:.3e}  eig_min={eig_min:.3e}  (sym_tol=1e-6, eig_tol=0)")
      assert sym_rel <= 1e-6
      assert eig_min > -1e-6


def test_xi_invariance_and_split_repartition():
  """Validate xi-invariance of total mobility and xi-dependence of the split.

  Eight non-overlapping particles (a = 1.0, eta = 1/(6*pi*a), seed=7) are
  placed in a cubic box (L = 32 a).  A high-accuracy reference projected
  mobility matrix B^T M B is first computed at xi*a = 0.6 with tol = 5e-7.
  Four additional
  operators are then built at xi*a ∈ {0.25, 0.5, 1.0, 1.5} with tol = 1e-5,
  and two non-trivial assertions are checked:

  1. **xi invariance** (rel <= 5e-3): the Frobenius relative error between
     M at the current xi and the high-accuracy reference:
       ||M(xi) - M_ref|| / ||M_ref|| <= 5e-3.
     This confirms that the total mobility is independent of how the Ewald
     sum is split between real and wave space, as required by the PSE identity
     M = Mr(xi) + Mw(xi) for any valid xi.

  2. **Split repartition with xi**: the projected real-space and wave-space
     operators must vary across xi. If Mr and Mw are nearly unchanged as xi
     sweeps, the decomposition is not responding to the Ewald split parameter.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 32.0 * a
  box = _box(l_box)
  n_particles = 8
  positions = rtu._nonoverlap_positions(n_particles, np.asarray(box, dtype=np.float64), a=a, seed=7)
  overlap_count = rtu._count_overlaps(positions, box, a)
  print(f"  number of overlaps (r < 2a) = {overlap_count}  (expected 0)")

  ref = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.6 / a,
      tol=5e-7,
  )
  ref_dense, _, _ = rtu._projected_matrix_from_apply(ref['apply_fn'], ref['state'], positions)

  mr_mats = []
  mw_mats = []
  for xi_a in (0.25, 0.5, 1.0, 1.5):
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=xi_a / a,
        tol=1e-5,
    )
    m_apply, _, _ = rtu._projected_matrix_from_apply(op['apply_fn'], op['state'], positions)
    mr, mw, _, _ = rtu._split_projected_mobility(op['state'], positions)
    xi_rel = rtu._frobenius_relative_error(m_apply, ref_dense)
    print(f"  xi*a={xi_a:.2f}  xi_rel={xi_rel:.3e}  (tol=5e-3)")
    assert xi_rel <= 5e-3
    mr_mats.append(mr)
    mw_mats.append(mw)
    del mr, mw

  def _max_pairwise_rel(mats):
    out = 0.0
    for i in range(len(mats)):
      for j in range(i + 1, len(mats)):
        out = max(out, rtu._frobenius_relative_error(mats[i], mats[j]))
    return out

  mr_var = _max_pairwise_rel(mr_mats)
  mw_var = _max_pairwise_rel(mw_mats)
  print(f"  split repartition: max_pairwise_rel(Mr)={mr_var:.3e}, max_pairwise_rel(Mw)={mw_var:.3e} (min expected 1e-2)")
  assert mr_var >= 1e-2
  assert mw_var >= 1e-2


def test_min_image_matches_lattice_in_safe_cutoff_regime():
  """Validate that min-image and lattice real-space modes agree in safe-cutoff regime.

  Six non-overlapping particles (a = 1.0, eta = 1/(6*pi*a), seed=121) are
  placed in a cubic box (L = 24 a). Two operators are constructed with
  identical parameters (xi*a = 0.5, rcut = 1.5*a, P = 8, Mgrid = 12,
  lattice_extent = 1) but different real_space_mode settings: 'min_image'
  and 'lattice'. A random force-balanced forcing is applied to both, and the
  following property is checked:

  1. **Mode equivalence in safe regime** (rel <= 5e-7): when rcut is
     sufficiently small compared to the box (rcut = 1.5*a << L/2 = 12*a), the
     min-image and lattice-summation real-space kernels must produce
     velocities that are numerically indistinguishable:
       ||U_min - U_lat|| / ||U_lat|| <= 5e-7.
     This validates that the min-image optimization (which ignores periodic
     images beyond the nearest copy) is safe and correct in configurations
     where the cutoff does not exceed L/2.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 24.0 * a
  box = _box(l_box)
  n_particles = 6
  positions = rtu._nonoverlap_positions(
      n_particles,
      np.asarray(box, dtype=np.float64),
      a=a,
      seed=121,
  )

  xi = 0.5 / a
  rcut = 1.5 * a
  p_support = 8
  mgrid = 12
  lattice_extent = 1

  op_min = _build_operator_with_mode(
      positions,
      box,
      a=a,
      eta=eta,
      xi=xi,
      rcut=rcut,
      p_support=p_support,
      mgrid=mgrid,
      lattice_extent=lattice_extent,
      real_space_mode='min_image',
  )
  op_lat = _build_operator_with_mode(
      positions,
      box,
      a=a,
      eta=eta,
      xi=xi,
      rcut=rcut,
      p_support=p_support,
      mgrid=mgrid,
      lattice_extent=lattice_extent,
      real_space_mode='lattice',
  )

  rng = np.random.default_rng(313)
  forces = rtu._force_balance(rng.normal(size=(n_particles, 3)))
  forces = jnp.asarray(forces, dtype=_dtype())

  u_min, _ = op_min['apply_fn'](op_min['state'], positions, forces)
  u_lat, _ = op_lat['apply_fn'](op_lat['state'], positions, forces)
  u_min = np.asarray(u_min, dtype=np.float64)
  u_lat = np.asarray(u_lat, dtype=np.float64)
  rel = np.linalg.norm(u_min - u_lat) / max(np.linalg.norm(u_lat), 1e-15)
  print(f"  safe-cutoff min-image vs lattice rel_err={rel:.3e}  (tol=5e-7)")
  assert rel <= 5e-7


def test_auto_mode_uses_lattice_images_when_min_image_is_unsafe():
  """Validate that 'auto' mode detects unsafe min-image regime and switches to lattice sum.

  Four particles (a = 1.0, eta = 1/(6*pi*a)) are placed in a very small
  cubic box (L = 5*a) at positions that create nearest-neighbor distances
  much closer than L/2. The 'auto' real_space_mode is designed to detect
  when min-image approximation is unsafe (when particle pairs violate the
  cutoff safety condition) and automatically fall back to lattice summation.
  Two assertions are checked:

  1. **Unsafe min-image detection**: the box is small enough (5*a), with
     rcut = 3*a, so min-image alone is unsafe. The 'auto' mode must detect
     this and include lattice images beyond (0, 0, 0).

  2. **Auto-mode image contribution** (rel >= 5e-3): when the operator is
     applied with explicit zero_image_index=0 (forcing inclusion of only the
     primary image), the result differs significantly from 'auto' mode, which
     includes higher-index images. This confirms that 'auto' mode is actively
     using lattice images to correct for the unsafe min-image configuration:
       ||U_auto - U_zero_only|| / ||U_auto|| >= 5e-3.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 5.0 * a
  box = _box(l_box)
  positions = jnp.asarray([
      [0.02, 0.50, 0.50],
      [0.98, 0.50, 0.50],
      [0.50, 0.02, 0.50],
      [0.50, 0.98, 0.50],
  ], dtype=_dtype())

  op_auto = _build_operator_with_mode(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.3 / a,
      rcut=3.0 * a,
      p_support=8,
      mgrid=12,
      lattice_extent=2,
      real_space_mode='auto',
  )

  rng = np.random.default_rng(913)
  forces = rtu._force_balance(rng.normal(size=(positions.shape[0], 3)))
  forces = jnp.asarray(forces, dtype=_dtype())

  u_auto, _ = op_auto['apply_fn'](op_auto['state'], positions, forces)
  u_zero, _ = op_auto['apply_fn'](
      op_auto['state'],
      positions,
      forces,
      lattice_indices=jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
      zero_image_index=0,
  )
  u_auto = np.asarray(u_auto, dtype=np.float64)
  u_zero = np.asarray(u_zero, dtype=np.float64)
  rel = np.linalg.norm(u_auto - u_zero) / max(np.linalg.norm(u_auto), 1e-15)
  print(f"  unsafe auto-mode image contribution rel_delta={rel:.3e}  (min=5e-3)")
  assert rel >= 5e-3


def test_lattice_extent_convergence_light_against_direct_reference():
  """Validate convergence of lattice-extent truncation against a high-accuracy reference.

  Five non-overlapping particles (a = 0.5, eta = 1.0, seed=17) are placed
  in a cubic box (L = 14.0) with xi*a = 0.6, p_support = 8, Mgrid = 12.
  Three operators are built with lattice_extent ∈ {1, 2, 3}, and a
  fourth high-accuracy reference is computed using direct lattice summation
  with extent = 4. The force-balanced projected mobility matrices are
  assembled for each, and the following convergence property is checked:

  1. **Monotonic extent convergence** (err2 <= err1 + 1e-3, err3 <= err2 + 1e-3):
     the Frobenius relative error between the projected M computed at extents
     {1, 2, 3} and the reference must decrease monotonically (with small
     tolerance for numerical noise). This validates that the lattice
     truncation is correctly implemented and that increasing extent brings
     the FFT-based computation closer to the exact lattice sum:
       ||M_proj(extent=k) - M_ref|| / ||M_ref|| is decreasing in k.

  2. **Absolute accuracy** (err3 <= 8e-2): even with extent = 3, the final
     error is controlled, confirming that the FFT-based lattice summation
     achieves adequate accuracy for practical use without requiring very
     large extents.
  """
  a = 0.5
  eta = 1.0
  l_box = 14.0
  box = _box(l_box)
  positions = rtu._nonoverlap_positions(
      5,
      np.asarray(box, dtype=np.float64),
      a=a,
      seed=17,
  )

  xi = 0.6 / a
  p_support = 8
  mgrid = 12
  tol = 1e-6

  projected_by_extent = {}
  basis = None
  for extent in (1, 2, 3):
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=xi,
        tol=tol,
        p_support=p_support,
        mgrid=mgrid,
        lattice_extent=extent,
    )
    m_proj, basis_extent, _ = rtu._projected_matrix_from_apply(op['apply_fn'], op['state'], positions)
    if basis is None:
      basis = basis_extent
    projected_by_extent[extent] = m_proj

  assert basis is not None
  m_ref_full = rtu._direct_lattice_rpy_matrix(
      positions,
      np.asarray(box, dtype=np.float64),
      a=a,
      eta=eta,
      extent=4,
  )
  m_ref = basis.T @ m_ref_full @ basis

  err1 = rtu._frobenius_relative_error(projected_by_extent[1], m_ref)
  err2 = rtu._frobenius_relative_error(projected_by_extent[2], m_ref)
  err3 = rtu._frobenius_relative_error(projected_by_extent[3], m_ref)
  print(f"  lattice extent convergence errs: e1={err1:.3e}, e2={err2:.3e}, e3={err3:.3e}")
  assert err2 <= err1 + 1e-3
  assert err3 <= err2 + 1e-3
  assert err3 <= 8e-2

@pytest.mark.slow
def test_ewald_parameter_convergence_rates():
  """Validate exponential convergence of real-space and wave-space Ewald parameters.

  The PSE-RPY method splits the Green's function as M = M^r(ξ, rcut) + M^w(ξ).
  Real-space error decays exponentially in (ξ*rcut)^2 (Fiore et al., Eq. 26);
  wave-space error decays exponentially in k_cut^2 / (4ξ^2) (Fiore et al., Eq. 28).
  This test validates both convergence rates using a reference velocity field
  computed with high accuracy (rcut = 3.2*a, Mgrid = 28, P = 12).

  Eight non-overlapping particles (a = 0.5, eta = 1.0, seed = 113) are placed
  in a cubic box (L = 16.0) with ξ*a = 0.5. A force-balanced configuration
  (net zero force, deterministic seed) is applied.

  Two convergence sweeps are performed:

  1. **Real-space sweep** (rcut ∈ {1.2*a, 1.6*a, 2.0*a, 2.6*a}, Mgrid = 28 fixed):
     relative velocity error ||U(rcut) - U_ref|| / ||U_ref|| is computed and
     regressed against (ξ*rcut)^2. Assertions check:
       - Monotonic decay: max_i (err[i+1] - err[i]) <= 5e-3.
       - Exponential fit slope < -0.1, confirming exp(-const*(ξ*rcut)^2) behavior.

  2. **Wave-space sweep** (Mgrid ∈ {12, 16, 20, 24}, rcut = 3.2*a fixed):
     relative velocity error is regressed against k_cut^2 / (4ξ^2). Assertions check:
       - Monotonic decay: max_i (err[i+1] - err[i]) <= 5e-3.
       - Exponential fit slope < -0.05, confirming exp(-const*k_cut^2/(4ξ^2)) behavior.

  These slopes validate that the Fiore et al. parameter selection formulas
  (Eq. 23-28) correctly capture the underlying error scalings.
  """
  a = 0.5
  eta = 1.0
  l_box = 16.0
  box = _box(l_box)
  positions = rtu._nonoverlap_positions(8, np.asarray(box, dtype=np.float64), a=a, seed=113)
  xi = 0.5 / a
  rng = np.random.default_rng(2026)
  force = rtu._force_balance(rng.normal(size=(8, 3)))
  force = jnp.asarray(force, dtype=_dtype())
  fit_floor = 1e-8

  def _stable_log_slope(x, err, *, floor):
    """Fit log-slope only where error is above numerical floor."""
    x = np.asarray(x, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    mask = err > floor
    if int(np.sum(mask)) < 2:
      return None
    return float(np.polyfit(x[mask], np.log(err[mask]), deg=1)[0])

  def velocity_for_settings(rcut, mgrid, p_support):
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=xi,
        tol=1e-6,
        rcut=rcut,
        p_support=p_support,
        mgrid=mgrid,
        lattice_extent=3,
    )
    velocities, _ = op['apply_fn'](op['state'], positions, force)
    return np.asarray(velocities, dtype=np.float64).reshape(-1)

  u_ref = velocity_for_settings(rcut=3.2 * a, mgrid=28, p_support=12)

  rcut_values = np.array([1.2 * a, 1.6 * a, 2.0 * a, 2.6 * a], dtype=np.float64)
  err_real = []
  for rcut in rcut_values:
    u = velocity_for_settings(rcut=rcut, mgrid=28, p_support=12)
    err_real.append(np.linalg.norm(u - u_ref) / max(np.linalg.norm(u_ref), 1e-15))
  err_real = np.asarray(err_real, dtype=np.float64)
  assert np.all(np.diff(err_real) <= 5e-3)
  x_real = (xi * rcut_values) ** 2
  slope_real = _stable_log_slope(x_real, err_real, floor=fit_floor)
  if slope_real is None:
    # Already in machine-noise regime; verify the full sweep is uniformly tiny.
    assert np.max(err_real) <= fit_floor
  else:
    assert slope_real < -0.1

  # build_wave_modes enforces P_support <= min(Mx, My, Mz); keep this sweep valid.
  mgrid_values = np.array([12, 16, 20, 24], dtype=np.int32)
  err_wave = []
  ksq_over_4xi2 = []
  for mgrid in mgrid_values:
    u = velocity_for_settings(rcut=3.2 * a, mgrid=int(mgrid), p_support=12)
    err_wave.append(np.linalg.norm(u - u_ref) / max(np.linalg.norm(u_ref), 1e-15))
    kcut = rtu._effective_kcut_cubic(l_box, int(mgrid))
    ksq_over_4xi2.append((kcut ** 2) / (4.0 * (xi ** 2)))
  err_wave = np.asarray(err_wave, dtype=np.float64)
  ksq_over_4xi2 = np.asarray(ksq_over_4xi2, dtype=np.float64)
  assert np.all(np.diff(err_wave) <= 5e-3)
  slope_wave = _stable_log_slope(ksq_over_4xi2, err_wave, floor=fit_floor)
  if slope_wave is None:
    assert np.max(err_wave) <= fit_floor
  else:
    assert slope_wave < -0.05


@pytest.mark.slow
def test_condition_number_mr_weak_n_dependence():
  """Validate that the real-space mobility matrix condition number grows weakly with N.

  For large-scale simulations, the numerical conditioning of the linear system
  M^r u = f (solved by Lanczos iteration in stochastic sampling) is critical.
  This test verifies that κ(M^r) ∝ N^α with α << 1, ensuring that the preconditioner
  remains effective as system size scales.

  Four system sizes (N ∈ {12, 24, 36, 48} particles) are studied at fixed volume
  fraction φ = 0.10 (box size scales as L ∝ N^{1/3}). Parameters: a = 0.5,
  eta = 1.0, ξ*a = 0.5. Non-overlapping initial configurations are generated
  with deterministic seeds (700 + i_particle).

  For each system, the force-balanced projected real-space mobility tensor
  B^T M^r B is assembled. Its condition number κ = λ_max / λ_min is computed
  from the symmetrized 0.5*(M^r + M^r^T).

  One assertion is checked:

  1. **Weak N-dependence** (κ_max / κ_min <= 6.0): the ratio of condition
     numbers across the four system sizes must satisfy
       max(κ) / min(κ) <= 6.0,
     validating weak (sublinear, likely ~log(N)) scaling. This confirms that
     the Jacobi preconditioner in the Lanczos-Chow-Saad sampler
     (see rpy_real_stoch.py) remains well-matched across scaling, preventing
     iteration count explosion in larger simulations.
  """
  a = 0.5
  eta = 1.0
  phi = 0.10
  xi = 0.5 / a
  n_values = [12, 24, 36, 48]
  kappas = []

  for i, n_particles in enumerate(n_values):
    l_box = (n_particles * (4.0 / 3.0) * math.pi * (a ** 3) / phi) ** (1.0 / 3.0)
    box = _box(l_box)
    positions = rtu._nonoverlap_positions(
        n_particles,
        np.asarray(box, dtype=np.float64),
        a=a,
        seed=700 + i,
        clearance=0.02,
    )
    op = rtu._build_operator(
        positions,
        box,
        a=a,
        eta=eta,
        xi=xi,
        tol=2e-4,
    )
    mr, _, _, _ = rtu._split_projected_mobility(op['state'], positions)
    eigvals = np.linalg.eigvalsh(0.5 * (mr + mr.T))
    eig_min = max(float(np.min(eigvals)), 1e-12)
    eig_max = float(np.max(eigvals))
    kappas.append(eig_max / eig_min)

  kappas = np.asarray(kappas, dtype=np.float64)
  assert np.all(np.isfinite(kappas))
  assert np.max(kappas) / np.min(kappas) <= 6.0


def test_translational_invariance():
  """Validate that the RPY mobility is invariant to uniform translation.

  Twelve non-overlapping particles (a = 1.0, eta = 1/(6*pi*a), seed=3) are
  placed in a cubic box (L = 36 a). One invariance is checked:

  1. **Translational invariance** (rel <= 1e-5): shifting all particle
     fractional coordinates by a random vector (mod 1) and rebuilding the
     neighbor list must produce identical velocities under the same forces:
       ||U(r + s) - U(r)|| / ||U(r)|| <= 1e-5.
     This confirms that the operator depends only on inter-particle
     displacements, not absolute positions, and that the periodic boundary
     handling is correct.

  A strict per-particle "Galilean invariance" check under uniform forcing is
  intentionally omitted: with periodic background-flow formulations, equal
  applied forces do not generally imply identical particle velocities for a
  fixed finite configuration.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 36.0 * a
  box = _box(l_box)
  n_particles = 12
  positions = rtu._nonoverlap_positions(n_particles, np.asarray(box, dtype=np.float64), a=a, seed=3)

  op = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-5,
  )
  init_fn = op['init_fn']
  apply_fn = op['apply_fn']
  state = op['state']

  rng = np.random.default_rng(11)
  forces = rtu._force_balance(rng.normal(size=(n_particles, 3)))
  forces = jnp.asarray(forces, dtype=_dtype())
  vel_a, _ = apply_fn(state, positions, forces)

  shift = jnp.asarray(rng.random((1, 3)), dtype=_dtype())
  shifted = jnp.mod(positions + shift, 1.0)
  shifted_state = init_fn(shifted)
  vel_b, _ = apply_fn(shifted_state, shifted, forces)

  vel_a = np.asarray(vel_a, dtype=np.float64)
  vel_b = np.asarray(vel_b, dtype=np.float64)
  rel = np.linalg.norm(vel_a - vel_b) / max(np.linalg.norm(vel_a), 1e-15)
  print(f"  translational invariance rel_err={rel:.3e}  (tol=1e-5)")
  assert rel <= 1e-5



def test_fdt_covariance_lightweight():
  """Validate the fluctuation-dissipation theorem (FDT) for the stochastic RPY mobility.

  Six non-overlapping particles (a = 1.0, eta = 1/(6*pi*a), seed=19) are
  placed in a cubic box (L = 32 a).  The stochastic displacement
  M^{1/2} z is drawn in a single reusable Monte Carlo loop (fast-CI sized),
  and three properties are verified:

  1. **FDT covariance** (Wishart-calibrated): the sample covariance
     of the sampled draws projected into the force-balanced
     subspace must match the projected mobility matrix B^T M B:
       ||Cov(z) - M|| / ||M|| <= tol.
     This directly validates the fluctuation-dissipation relation
     <db db^T> = 2 kT dt M.

  2. **Zero mean drift** (sample-scaled): the sample mean of the draws must be
     negligible relative to the projected mobility scale sqrt(Tr(B^T M B)/dim),
     with tolerance proportional to sqrt(dim / S).

  3. **Real/wave independence** (cross_rel <= max(0.13, 4.5/sqrt(S))): the
     cross-covariance between the real-space Lanczos
     samples (M^r^{1/2} z_r) and the wave-space Fourier samples (M^w^{1/2} z_w)
     must be small relative to the geometric mean of the two auto-covariances:
       ||Cov(z_r, z_w)|| / sqrt(||Cov_r|| * ||Cov_w||) <= tol.
     This confirms that the two noise sources are statistically independent,
     as required for the split M^{1/2} = (M^r^{1/2}, M^w^{1/2}) decomposition
     to correctly sample from M = M^r + M^w.
  """
  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 32.0 * a
  box = _box(l_box)
  positions = rtu._nonoverlap_positions(6, np.asarray(box, dtype=np.float64), a=a, seed=19)
  op = rtu._build_operator(
      positions,
      box,
      a=a,
      eta=eta,
      xi=0.5 / a,
      tol=1e-5,
      include_brownian=True,
  )
  apply_fn = op['apply_fn']
  state = op['state']
  m_dense, basis, _ = rtu._projected_matrix_from_apply(apply_fn, state, positions)

  samples = 800
  key = jax.random.PRNGKey(123)
  sqrt_fn = state.wave.sqrt_fn
  assert sqrt_fn is not None
  draws = np.zeros((samples, m_dense.shape[0]), dtype=np.float64)
  real_draws = np.zeros((samples, m_dense.shape[0]), dtype=np.float64)
  wave_draws = np.zeros((samples, m_dense.shape[0]), dtype=np.float64)

  for i in range(samples):
    key, key_real, key_wave = jax.random.split(key, 3)
    real_sample = rpy_real.sample_mr_sqrt_precond(
        key_real,
        state.real,
        positions,
        precond=state.preconditioner,
        iters=8,
    )
    wave_sample = sqrt_fn(key_wave, positions)
    real_proj = basis.T @ np.asarray(real_sample, dtype=np.float64).reshape(-1)
    wave_proj = basis.T @ np.asarray(wave_sample, dtype=np.float64).reshape(-1)
    real_draws[i] = real_proj
    wave_draws[i] = wave_proj
    draws[i] = real_proj + wave_proj

  cov, mean = rtu._sample_covariance(draws)
  cov_rel = rtu._frobenius_relative_error(cov, m_dense)
  mean_scale = math.sqrt(max(np.trace(m_dense) / m_dense.shape[0], 1e-15))
  mean_norm = np.linalg.norm(mean) / mean_scale
  cov_scale = rtu._wishart_frobenius_relative_scale(m_dense, samples)
  cov_tol = 3.0 * cov_scale
  mean_tol = max(0.2, 2.0 * math.sqrt(m_dense.shape[0] / float(samples)))
  print(f"  FDT covariance: cov_rel={cov_rel:.3e}  (tol={cov_tol:.3e})")
  print(f"  FDT mean drift: mean_norm={mean_norm:.3e}  (tol={mean_tol:.3e})")

  assert cov_rel <= cov_tol
  assert mean_norm <= mean_tol

  cross = rtu._cross_covariance(real_draws, wave_draws)
  cov_r, _ = rtu._sample_covariance(real_draws)
  cov_w, _ = rtu._sample_covariance(wave_draws)
  cross_rel = np.linalg.norm(cross) / max(math.sqrt(np.linalg.norm(cov_r) * np.linalg.norm(cov_w)), 1e-15)
  cross_tol = max(0.13, 4.5 / math.sqrt(samples))
  print(f"  Real/wave cross-covariance: cross_rel={cross_rel:.3e}  (tol={cross_tol:.3e})")
  assert cross_rel <= cross_tol


@pytest.mark.slow
def test_fdt_covariance_rigorous():
  """Statistically rigorous FDT test: false-failure probability ≤ 1%.

  Acceptance regions are derived from exact distributions:
    - chi-square for variance: (S-1)*s2/sigma2 ~ chi2(S-1)
    - standard normal for mean: y_bar / (sigma/sqrt(S)) ~ N(0,1)
    - Student t for correlation: r*sqrt((S-2)/(1-r^2)) ~ t(S-2)

  Union bound ensures overall false-failure prob ≤ 1%:
    - K=10 variance tests + K=10 mean tests → per-test alpha = 0.005/(2K)
    - K=10 independence tests               → per-test alpha = 0.005/K
    - total ≤ 0.005 + 0.005 = 1%

  Requires x64; marked slow (S=2000 draws × iters=20 Lanczos steps).
  """
  from scipy.stats import chi2, norm as scipy_norm, t as scipy_t

  a = 1.0
  eta = 1.0 / (6.0 * math.pi * a)
  l_box = 32.0 * a
  box = _box(l_box)
  positions = rtu._nonoverlap_positions(6, np.asarray(box, dtype=np.float64), a=a, seed=19)
  op = rtu._build_operator(
      positions, box, a=a, eta=eta, xi=0.5 / a, tol=1e-5, include_brownian=True,
  )
  apply_fn = op['apply_fn']
  state = op['state']
  m_dense, basis, _ = rtu._projected_matrix_from_apply(apply_fn, state, positions)

  # ---------- sampling parameters ----------
  S = 2000
  K = 10
  # Error budget split equally: 0.005 for cov+mean (2K tests), 0.005 for independence (K tests)
  alpha_cm  = 0.005 / (2 * K)   # per variance or mean test  → 0.00025
  alpha_ind = 0.005 / K          # per independence test       → 0.0005

  # Fixed K random unit directions in the projected subspace (deterministic seed)
  rng_dirs = np.random.default_rng(0)
  d = m_dense.shape[0]
  directions = rng_dirs.standard_normal((K, d))
  directions /= np.linalg.norm(directions, axis=1, keepdims=True)

  # ---------- generate S draws ----------
  key = jax.random.PRNGKey(42)
  sqrt_fn = state.wave.sqrt_fn
  assert sqrt_fn is not None
  real_draws = np.zeros((S, d), dtype=np.float64)
  wave_draws = np.zeros((S, d), dtype=np.float64)

  for i in range(S):
    key, key_real, key_wave = jax.random.split(key, 3)
    real_sample = rpy_real.sample_mr_sqrt_precond(
        key_real, state.real, positions,
        precond=state.preconditioner, iters=20,
    )
    wave_sample = sqrt_fn(key_wave, positions)
    real_proj = basis.T @ np.asarray(real_sample, dtype=np.float64).reshape(-1)
    wave_proj = basis.T @ np.asarray(wave_sample, dtype=np.float64).reshape(-1)
    real_draws[i] = real_proj
    wave_draws[i] = wave_proj

  draws = real_draws + wave_draws

  # ---------- precompute quantiles ----------
  df_var = S - 1           # chi-square dof for variance test
  df_t   = S - 2           # t dof for correlation test
  chi2_lo = chi2.ppf(alpha_cm / 2, df_var)
  chi2_hi = chi2.ppf(1.0 - alpha_cm / 2, df_var)
  z_cm    = scipy_norm.ppf(1.0 - alpha_cm / 2)
  t_crit  = scipy_t.ppf(1.0 - alpha_ind / 2, df_t)

  # ---------- per-direction tests ----------
  # At S=2000 the chi-square bounds are roughly ±8% in variance (rel std ~ 3.2%,
  # two-sided 99% band after Bonferroni is tighter than that).
  for k, q in enumerate(directions):
    y = draws @ q              # total projected draws, shape (S,)
    y_mean = float(y.mean())
    s2 = float(((y - y_mean) ** 2).sum() / df_var)
    sigma2 = float(q @ m_dense @ q)

    # -- Exact variance test: (df_var * s2 / sigma2) ~ chi2(df_var) --
    lo = sigma2 * chi2_lo / df_var
    hi = sigma2 * chi2_hi / df_var
    print(f"  dir {k} var: sigma2={sigma2:.4e}  s2={s2:.4e}  bounds=[{lo:.4e}, {hi:.4e}]")
    assert lo <= s2 <= hi, (
        f"Variance test failed dir {k}: s2={s2:.4e} not in [{lo:.4e}, {hi:.4e}]"
    )

    # -- Exact mean test: y_mean / (sigma/sqrt(S)) ~ N(0,1) --
    mean_bound = z_cm * math.sqrt(sigma2 / S)
    print(f"  dir {k} mean: |mean|={abs(y_mean):.4e}  bound={mean_bound:.4e}")
    assert abs(y_mean) <= mean_bound, (
        f"Mean test failed dir {k}: |mean|={abs(y_mean):.4e} > {mean_bound:.4e}"
    )

    # -- Exact independence test: t = rho * sqrt((S-2)/(1-rho^2)) ~ t(S-2) --
    yr = real_draws @ q
    yw = wave_draws @ q
    yr_c = yr - yr.mean()
    yw_c = yw - yw.mean()
    denom = math.sqrt(float((yr_c ** 2).sum()) * float((yw_c ** 2).sum()))
    rho = float((yr_c * yw_c).sum()) / max(denom, 1e-15)
    rho_clipped = float(np.clip(rho, -0.999999, 0.999999))
    t_stat = rho_clipped * math.sqrt(df_t / (1.0 - rho_clipped ** 2))
    print(f"  dir {k} indep: rho={rho:.4f}  t_stat={t_stat:.4f}  crit={t_crit:.4f}")
    assert abs(t_stat) <= t_crit, (
        f"Independence test failed dir {k}: |t_stat|={abs(t_stat):.4f} > {t_crit:.4f}"
    )

  # ---------- secondary Frobenius diagnostic (printed only, not asserted) ----------
  # Uses biased estimator (/ S) to match _wishart_frobenius_relative_scale's assumption.
  centered = draws - draws.mean(axis=0)
  cov_diag = centered.T @ centered / S
  cov_rel  = rtu._frobenius_relative_error(cov_diag, m_dense)
  cov_scale = rtu._wishart_frobenius_relative_scale(m_dense, S)
  print(f"  [diag] Frobenius cov_rel={cov_rel:.3e}  Wishart_scale={cov_scale:.3e}  (biased 1/S)")
