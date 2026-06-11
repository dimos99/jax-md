"""Validation gates for the Phase-1 grand mobility (force + couplet).

The stresslet extension is gated by external pins, not internal
self-consistency alone:

  * Scalar radial functions: golden values from the FSD reference code
    (quad-precision ``Stokes.cc``) and continuity across the r = 2a branch
    switch.
  * Real-space tensors: pair blocks against the independent quadrature ground
    truth in ``rpy_quadrature_reference.py`` (UF/UC/DF/DC, both branches),
    plus the antisymmetric-couplet translation sign, which pins the couplet
    index convention against rotlet physics.
  * Wave space: cached NUFFT matvec against a direct k-space sum
    (``_mw_bruteforce_grand``), drop-zz packing against a full 9-component
    pipeline including the post-NUFFT grid-traceless property, and
    spread/gather adjointness at 11 moment channels.
  * Ewald split: per-block xi-invariance of the total grand mobility (the
    decisive transcription-error detector), symmetry + positive
    semi-definiteness in Frobenius-orthonormal moment coordinates, Dense vs
    OrderedSparse parity (including self-image terms when rcut exceeds the
    box), the DC r -> 0 fallback limit, and the live-box shear path.
  * Regression: ``use_stresslet=False`` must reproduce the legacy operator
    bit-for-bit, and the grand operator at C = 0 must reproduce legacy
    velocities.

Long-running gates are marked ``slow`` (deselected by default; run with
``pytest -m slow``).  Tolerances adapt to ``jax_enable_x64``; the scalar
cancellation tests are skipped in float32 where the skeleton loses too many
digits by construction.
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import partition, space
from jax_md.hydro import rpy, rpy_real, rpy_wave
from jax_md.hydro.rpy_moments import (
    N_MOMENTS,
    couplet_to_components,
    components_to_couplet,
    traceless,
    traceless_orthonormal_basis,
)
from jax_md.hydro.rpy_real_det_dipole_helpers import (
    G1G2_closed_form,
    K1K2K3_closed_form,
    Mr_self_dipole,
)
from jax_md.hydro.rpy_real_det_dipole import _build_pair_contrib_grand_fn
from jax_md.hydro.rpy_wave_det import _mw_bruteforce
from jax_md.hydro.rpy_wave_det_dipole import _mw_bruteforce_grand

sys.path.insert(0, 'tests')
import rpy_test_utils as rtu  # pylint: disable=wrong-import-position


def _dtype():
  return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _box(length):
  return jnp.eye(3, dtype=_dtype()) * _dtype()(length)


def _traceless_np(C):
  C = np.asarray(C, dtype=np.float64)
  return C - np.trace(C, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0


def _relative_error(actual, expected):
  actual = np.asarray(actual, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15)


def _random_couplets(n_particles, seed=0):
  rng = np.random.default_rng(seed)
  return _traceless_np(rng.normal(size=(n_particles, 3, 3)))


def _split_grand_blocks(M, n_particles):
  force = []
  couplet = []
  for p in range(n_particles):
    force.extend(range(11 * p, 11 * p + 3))
    couplet.extend(range(11 * p + 3, 11 * p + 11))
  force = np.asarray(force, dtype=np.int64)
  couplet = np.asarray(couplet, dtype=np.int64)
  return {
      'UF': M[np.ix_(force, force)],
      'UC': M[np.ix_(force, couplet)],
      'DF': M[np.ix_(couplet, force)],
      'DC': M[np.ix_(couplet, couplet)],
  }


def _assert_symmetric_psd(M, *, sym_tol=1e-8, psd_tol=1e-10):
  M = np.asarray(M, dtype=np.float64)
  scale = max(np.linalg.norm(M), 1e-15)
  assert np.linalg.norm(M - M.T) / scale < sym_tol
  eigvals = np.linalg.eigvalsh(0.5 * (M + M.T))
  assert eigvals.min() >= -psd_tol * max(np.max(np.abs(eigvals)), 1.0)


def _build_grand_operator(positions, box, *, xi, neighbor_format=partition.NeighborListFormat.Dense):
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns,
      a=0.4,
      xi=xi,
      eta=1.0,
      rcut=3.2,
      Mgrid=10,
      P=5,
      include_brownian=False,
      lattice_extent=1,
      real_space_mode='lattice',
      neighbor_format=neighbor_format,
      use_stresslet=True,
  )
  state = init_fn(positions)
  return state, apply_fn


def _real_pair_outputs(r_vec, force, couplet, *, a, xi, eta):
  r_vec = jnp.asarray(r_vec, dtype=_dtype())[None, :]
  r2 = jnp.sum(r_vec * r_vec, axis=-1)
  mask = jnp.ones_like(r2, dtype=bool)
  force = jnp.asarray(force, dtype=_dtype())[None, :]
  couplet = jnp.asarray(couplet, dtype=_dtype())[None, :, :]
  (_, _, pair_eps2, prefactors, pair_fn) = _build_pair_contrib_grand_fn(a, xi, eta)
  u, d = pair_fn(
      r_vec,
      r2,
      mask,
      force,
      couplet,
      prefactor_uf=jnp.asarray(prefactors[0], dtype=_dtype()),
      prefactor_uc=jnp.asarray(prefactors[1], dtype=_dtype()),
      prefactor_dc=jnp.asarray(prefactors[2], dtype=_dtype()),
      pair_eps2=jnp.asarray(pair_eps2, dtype=_dtype()),
      self_dipole=jnp.asarray(Mr_self_dipole(a, xi), dtype=_dtype()),
  )
  return np.asarray(u[0], dtype=np.float64), np.asarray(d[0], dtype=np.float64)


def _full9_grand_wave_matvec(state, positions, forces, couplets):
  params = state.params
  modes = state.modes
  Mx, My, Mz, P = params.Mx, params.My, params.Mz, params.P
  V_box = jnp.asarray(params.volume, dtype=_dtype())
  sigma_inv = jnp.asarray(Mx * My * Mz, dtype=_dtype()) / V_box
  st = rpy_wave.build_stencils_frac(positions, Mx, My, Mz, P, params.alpha)
  C = traceless(jnp.asarray(couplets, dtype=_dtype()))
  moments9 = jnp.concatenate([forces, C.reshape(C.shape[:-2] + (9,))], axis=-1)
  moment_grid = sigma_inv * rpy_wave.spread(moments9, st, Mx, My, Mz)
  moment_q = rpy_wave.fft_vec(moment_grid)
  Fq = moment_q[..., :3]
  Cq = moment_q[..., 3:].reshape(moment_q.shape[:-1] + (3, 3))
  fq = (modes['Pshape'][..., None] * Fq -
        1j * modes['Pdip'][..., None] * jnp.einsum('...mn,...n->...m', Cq, modes['k']))
  uq = jnp.einsum('...ij,...j->...i', modes['Bfluid'], fq)
  Uq = modes['Pshape'][..., None] * uq
  Dq = 1j * modes['Pdip'][..., None, None] * jnp.einsum('...i,...j->...ij', uq, modes['k'])
  out_q = jnp.concatenate([Uq, Dq.reshape(Dq.shape[:-2] + (9,))], axis=-1)
  out = V_box * rpy_wave.gather(rpy_wave.ifft_vec(out_q), st, Mx, My, Mz)
  return out[..., :3], traceless(out[..., 3:].reshape(out.shape[:-1] + (3, 3))), Dq


def test_moment_pack_round_trip_and_basis():
  C = jnp.asarray([
      [[1.0, 2.0, -1.0],
       [0.5, -3.0, 4.0],
       [2.0, -0.25, 5.0]],
      [[-2.0, 0.0, 1.5],
       [3.0, 4.0, -1.0],
       [0.25, 2.0, -6.0]],
  ], dtype=_dtype())
  C0 = traceless(C)
  packed = couplet_to_components(C0)
  unpacked = components_to_couplet(packed)
  np.testing.assert_allclose(np.asarray(unpacked), np.asarray(C0), atol=1e-6)

  basis = np.asarray(traceless_orthonormal_basis(), dtype=np.float64)
  gram = np.einsum('aij,bij->ab', basis, basis)
  np.testing.assert_allclose(gram, np.eye(8), atol=1e-12)
  np.testing.assert_allclose(np.trace(basis, axis1=-2, axis2=-1), np.zeros(8), atol=1e-12)


def test_dipole_scalar_continuity_at_contact():
  a = 1.0
  for xi in (0.1, 0.5, 1.0):
    left = 2.0 * a - 1.0e-8 * a
    right = 2.0 * a + 1.0e-8 * a
    vals_left = G1G2_closed_form(jnp.asarray(left), a, xi) + K1K2K3_closed_form(jnp.asarray(left), a, xi)
    vals_right = G1G2_closed_form(jnp.asarray(right), a, xi) + K1K2K3_closed_form(jnp.asarray(right), a, xi)
    for v_left, v_right in zip(vals_left, vals_right):
      np.testing.assert_allclose(
          np.asarray(v_left),
          np.asarray(v_right),
          rtol=2e-5,
          atol=2e-5,
      )


def test_dipole_scalars_match_fsd_reference_values():
  """Golden-value pin against the FSD HOOMD plugin (Fiore & Swan).

  Reference values were evaluated from the quadruple-precision analytic
  expressions in FSD ``source/Stokes.cc`` (table scalars g1, g2 -> G1, G2 and
  h1, h2, h3 -> K1, K2, K3) via mpmath at dps=40; both non-overlapping and
  overlapping branches are covered.  The same scalars were independently
  validated against ``tests/rpy_quadrature_reference.py`` (free-minus-screened
  radial integrals) to ~1e-9.
  """
  golden = {
      (2.6, 1.0, 0.5): dict(G1=2.161852916613451e-02, G2=-6.077084745460597e-02,
          K1=-1.255155876569465e-02, K2=-3.602254823048653e-02, K3=5.950821017364771e-02),
      (1.4, 1.0, 0.5): dict(G1=-4.221843366683528e-01, G2=-1.416527928973453e-01,
          K1=-8.140319337852986e-02, K2=-5.540423229569001e-01, K3=1.316338628124974e-01),
      (3.1, 0.7, 0.9): dict(G1=1.101112580275946e-02, G2=-1.157814856171519e-03,
          K1=-1.402246329487043e-04, K2=1.665563409178265e-02, K3=1.987250479159282e-03),
      (1.1, 0.7, 0.9): dict(G1=-5.325100446310969e-01, G2=-2.025755284720192e-01,
          K1=-1.453788218718847e-01, K2=-1.244959969416009e+00, K3=3.296755295976904e-01),
  }
  rtol = 1e-7 if jax.config.jax_enable_x64 else 2e-4
  for (r, a, xi), ref in golden.items():
    G1, G2 = G1G2_closed_form(jnp.asarray(r, dtype=_dtype()), a, xi)
    K1, K2, K3 = K1K2K3_closed_form(jnp.asarray(r, dtype=_dtype()), a, xi)
    got = dict(G1=float(G1), G2=float(G2), K1=float(K1), K2=float(K2), K3=float(K3))
    for name, expected in ref.items():
      np.testing.assert_allclose(got[name], expected, rtol=rtol, err_msg=f'{name} at {(r, a, xi)}')


def test_mr_self_dipole_matches_corrected_thesis_expression():
  a = 0.8
  xi = 0.6
  expected = (
      -3.0 * (6.0 * a * a * xi * xi + 1.0) /
      (80.0 * math.sqrt(math.pi) * a ** 6 * xi ** 3)
      + 3.0 * (10.0 * a * a * xi * xi + 1.0) /
      (80.0 * math.sqrt(math.pi) * a ** 6 * xi ** 3) *
      math.exp(-4.0 * a * a * xi * xi)
      - 3.0 / (10.0 * a ** 3) * math.erfc(2.0 * a * xi)
  )
  np.testing.assert_allclose(
      np.asarray(Mr_self_dipole(a, xi)), expected, rtol=1e-7, atol=1e-7)


def test_pdip_series_and_spread_gather_adjointness():
  K = jnp.asarray([0.0, 1e-4, 1e-2, 0.2, 1.0], dtype=_dtype())
  Pdip = rpy_wave.build_Pdip_modes(K, a=0.7)
  x = np.asarray(K * 0.7, dtype=np.float64)
  safe_x = np.where(np.abs(x) < 1e-12, 1.0, x)
  closed = 3.0 * (np.sin(safe_x) - safe_x * np.cos(safe_x)) / (safe_x ** 3)
  closed = np.where(np.abs(x) < 1e-12, 1.0, closed)
  np.testing.assert_allclose(np.asarray(Pdip), closed, rtol=5e-6, atol=5e-6)

  positions = jnp.asarray([[0.12, 0.27, 0.43], [0.55, 0.31, 0.86]], dtype=_dtype())
  Mx = My = Mz = 6
  P = 4
  alpha = 8.0
  st = rpy_wave.build_stencils_frac(positions, Mx, My, Mz, P, alpha)
  moments = jnp.arange(positions.shape[0] * N_MOMENTS, dtype=_dtype()).reshape((positions.shape[0], N_MOMENTS)) / 17.0
  grid = jnp.arange(Mx * My * Mz * N_MOMENTS, dtype=_dtype()).reshape((Mx, My, Mz, N_MOMENTS)) / 23.0
  lhs = jnp.sum(rpy_wave.gather(grid, st, Mx, My, Mz) * moments)
  rhs = jnp.sum(grid * rpy_wave.spread(moments, st, Mx, My, Mz))
  np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-6, atol=1e-6)


def test_wave_bruteforce_grand_c_zero_matches_legacy_reference():
  box = _box(9.0)
  positions = jnp.asarray([[0.13, 0.21, 0.34], [0.61, 0.44, 0.72]], dtype=_dtype())
  forces = jnp.asarray([[1.0, -0.5, 0.25], [0.2, 0.7, -0.3]], dtype=_dtype())
  couplets = jnp.zeros((2, 3, 3), dtype=_dtype())
  a = 0.4
  xi = 1.1
  eta = 0.9
  M = 5
  P = 3
  U_legacy = _mw_bruteforce(positions, forces, box, a, xi, eta, M, M, M, P)
  U_grand, D_grand = _mw_bruteforce_grand(
      positions, forces, couplets, box, a, xi, eta, M, M, M, P)
  np.testing.assert_allclose(np.asarray(U_grand), np.asarray(U_legacy), atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(
      np.asarray(jnp.trace(D_grand, axis1=-2, axis2=-1)),
      np.zeros(positions.shape[0]),
      atol=1e-6,
  )


def test_wave_traceless_dropzz_pipeline_matches_full9_pipeline():
  box = _box(7.0)
  positions = jnp.asarray([[0.13, 0.21, 0.34], [0.61, 0.44, 0.72]], dtype=_dtype())
  forces = jnp.asarray([[1.0, -0.5, 0.25], [0.2, 0.7, -0.3]], dtype=_dtype())
  couplets = jnp.asarray(_random_couplets(2, seed=13), dtype=_dtype())
  state = rpy_wave.build_grand_wave_modes(box, 0.4, 1.0, 0.9, 8, 8, 8, 4)
  U8, D8 = state.apply_fn(positions, forces, couplets)
  U9, D9, Dq9 = _full9_grand_wave_matvec(state, positions, forces, couplets)
  np.testing.assert_allclose(np.asarray(U8), np.asarray(U9), rtol=2e-5, atol=2e-5)
  np.testing.assert_allclose(np.asarray(D8), np.asarray(D9), rtol=2e-5, atol=2e-5)
  np.testing.assert_allclose(
      np.asarray(jnp.trace(Dq9, axis1=-2, axis2=-1)),
      np.zeros(Dq9.shape[:-2]),
      atol=2e-12 if jax.config.jax_enable_x64 else 2e-5,
  )


def test_stresslet_api_and_c_zero_velocity_regression():
  box = _box(8.0)
  positions = jnp.asarray([
      [0.10, 0.20, 0.30],
      [0.45, 0.50, 0.55],
      [0.20, 0.70, 0.40],
  ], dtype=_dtype())
  forces = jnp.asarray([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
  ], dtype=_dtype())
  kwargs = dict(
      a=0.5,
      xi=1.0,
      eta=1.0,
      rcut=2.0,
      Mgrid=8,
      P=4,
      include_brownian=False,
      lattice_extent=1,
      real_space_mode='lattice',
  )
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_legacy, apply_legacy = rpy.build_rpy_mobility(space_fns, **kwargs)
  legacy_state = init_legacy(positions)
  U_legacy, _ = apply_legacy(legacy_state, positions, forces)

  init_grand, apply_grand = rpy.build_rpy_mobility(
      space_fns, **kwargs, use_stresslet=True)
  grand_state = init_grand(positions)
  (U_grand, D_grand), _ = apply_grand(grand_state, positions, forces)
  np.testing.assert_allclose(np.asarray(U_grand), np.asarray(U_legacy), atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(
      np.asarray(jnp.trace(D_grand, axis1=-2, axis2=-1)),
      np.zeros(positions.shape[0]),
      atol=1e-6,
  )

  with pytest.raises(ValueError):
    apply_legacy(legacy_state, positions, forces, couplets=jnp.zeros((3, 3, 3), dtype=_dtype()))

  with pytest.raises(NotImplementedError):
    apply_grand(
        grand_state,
        positions,
        forces,
        brownian_key=jax.random.PRNGKey(0),
        kT=1.0,
        dt=0.1,
    )


@pytest.mark.slow
def test_shear_grand_path_matches_static_at_zero_gamma():
  """Live-box grand path: gamma=0 equals the static-box operator, and the
  total grand operator stays symmetric under finite shear."""
  L = 6.0
  positions = jnp.asarray([[0.1, 0.1, 0.1],
                           [0.52, 0.18, 0.12],
                           [0.15, 0.55, 0.6],
                           [0.6, 0.62, 0.55]], dtype=_dtype())
  forces = jnp.asarray(np.random.default_rng(3).normal(size=(4, 3)), dtype=_dtype())
  couplets = jnp.asarray(_random_couplets(4, seed=4), dtype=_dtype())
  box = _box(L)
  space_fns = space.periodic_general(box, fractional_coordinates=True)

  def box_fn(gamma_xy=0.0, gamma_xz=0.0, gamma_yz=0.0, **kw):
    B = jnp.eye(3, dtype=_dtype()) * _dtype()(L)
    return B.at[0, 1].set(_dtype()(gamma_xy * L))

  common = dict(a=1.0, xi=0.55, eta=1.0, rcut=2.9, Mgrid=20, P=9,
                include_brownian=False)
  init_s, apply_s = rpy.build_rpy_mobility(
      (space_fns[0], space_fns[1], box_fn), use_stresslet=True, **common)
  init_g, apply_g = rpy.build_rpy_mobility(space_fns, use_stresslet=True, **common)

  state_s = init_s(positions, gamma_xy=0.0)
  state_g = init_g(positions)
  (Us, Ds), _ = apply_s(state_s, positions, forces, couplets=couplets, gamma_xy=0.0)
  (Ug, Dg), _ = apply_g(state_g, positions, forces, couplets=couplets)
  # The live-box and cached wave paths are algebraically identical but run
  # through different jit programs; they agree only to the operator's own
  # NUFFT accuracy at this resolution (~1e-5 at Mgrid=20, P=9).  Convention
  # bugs would show up as O(1) differences.
  tol = 5e-5 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(Us, Ug) < tol
  assert _relative_error(Ds, Dg) < tol

  # Symmetry probes at finite shear: <y, M x> == <x, M y>.
  gamma = 0.3
  state_sh = init_s(positions, gamma_xy=gamma)
  rng = np.random.default_rng(11)

  def _mv(F, C):
    (U, D), _ = apply_s(state_sh, positions, jnp.asarray(F, dtype=_dtype()),
                        couplets=jnp.asarray(C, dtype=_dtype()), gamma_xy=gamma)
    return np.asarray(U, dtype=np.float64), np.asarray(D, dtype=np.float64)

  sym_tol = 1e-9 if jax.config.jax_enable_x64 else 1e-3
  for _ in range(4):
    Fx = rng.normal(size=(4, 3)); Cx = _traceless_np(rng.normal(size=(4, 3, 3)))
    Fy = rng.normal(size=(4, 3)); Cy = _traceless_np(rng.normal(size=(4, 3, 3)))
    Ux, Dx = _mv(Fx, Cx)
    Uy, Dy = _mv(Fy, Cy)
    lhs = np.sum(Uy * Fx) + np.sum(Dy * Cx)
    rhs = np.sum(Ux * Fy) + np.sum(Dx * Cy)
    scale = max(abs(lhs), abs(rhs), 1e-15)
    assert abs(lhs - rhs) / scale < sym_tol


@pytest.mark.slow
def test_real_space_pair_tensors_match_quadrature_reference():
  pytest.importorskip('scipy')
  import rpy_quadrature_reference as qref

  a = 0.7
  xi = 0.8
  eta = 1.0
  force = np.asarray([0.3, -0.4, 0.5])
  couplet = _traceless_np(np.asarray([[0.2, 0.3, -0.4],
                                      [0.1, -0.5, 0.2],
                                      [-0.2, 0.7, 0.3]]))
  # The quadrature tensors take r = x_receiver - x_sender; the pair kernel
  # takes rij = x_sender - x_receiver, hence the sign flip below.
  for r_vec in (np.asarray([2.1, 0.4, -0.3]), np.asarray([0.7, 0.2, 0.1])):
    u_force, d_force = _real_pair_outputs(-r_vec, force, np.zeros((3, 3)), a=a, xi=xi, eta=eta)
    u_couplet, d_couplet = _real_pair_outputs(-r_vec, np.zeros(3), couplet, a=a, xi=xi, eta=eta)
    ref_uf = np.einsum(
        'im,m->i',
        np.asarray(qref.muf_tensor(r_vec, a, None)) - np.asarray(qref.muf_tensor(r_vec, a, xi)),
        force,
    ) / (6.0 * math.pi * eta * a)
    ref_uc = np.einsum(
        'imn,mn->i',
        np.asarray(qref.muc_tensor(r_vec, a, None)) - np.asarray(qref.muc_tensor(r_vec, a, xi)),
        couplet,
    ) / (6.0 * math.pi * eta)
    ref_df = np.einsum(
        'ijm,m->ij',
        np.asarray(qref.mdf_tensor(r_vec, a, None)) - np.asarray(qref.mdf_tensor(r_vec, a, xi)),
        force,
    ) / (6.0 * math.pi * eta)
    ref_dc = np.einsum(
        'ijmn,mn->ij',
        np.asarray(qref.mdc_tensor(r_vec, a, None)) - np.asarray(qref.mdc_tensor(r_vec, a, xi)),
        couplet,
    ) / (6.0 * math.pi * eta)
    np.testing.assert_allclose(u_force, ref_uf, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(u_couplet, ref_uc, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(d_force, ref_df, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(d_couplet, ref_dc, rtol=3e-5, atol=3e-6)


def _grand_total_dense(positions, box_length, *, a, xi, eta, rcut, Mgrid, P):
  """Dense total grand mobility (real + wave) in orthonormal coordinates."""
  box = _box(box_length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_r, _ = rpy_real.build_Mr_grand_apply(space_fns, a, xi, eta, rcut)
  state_r = init_r(positions)
  state_w = rpy_wave.build_grand_wave_modes(box, a, xi, eta, Mgrid, Mgrid, Mgrid, P)

  def _mv(F, C):
    Ur, Dr = rpy_real.mr_grand_matvec(state_r, positions, F, C)
    Uw, Dw = rpy_wave.mw_grand_matvec(state_w, positions, F, C)
    return Ur + Uw, Dr + Dw

  return rtu._grand_dense_from_matvec(_mv, int(positions.shape[0]))


@pytest.mark.slow
def test_grand_mobility_xi_invariance_by_block():
  # The real-space cutoff must track xi (rcut ~ 5.5/xi, spilling into periodic
  # images) and the wave grid must resolve kcut ~ 9.6*xi for the split to be
  # accurate enough to observe invariance.
  positions = jnp.asarray([[0.1, 0.1, 0.1],
                           [0.52, 0.18, 0.12],
                           [0.15, 0.55, 0.6],
                           [0.6, 0.62, 0.55]], dtype=_dtype())
  matrices = []
  for xi in (0.5, 0.75, 1.0):
    matrices.append(_grand_total_dense(
        positions, 6.0, a=1.0, xi=xi, eta=1.0,
        rcut=min(5.5 / xi, 11.0), Mgrid=28, P=11))
  tol = 1e-5 if jax.config.jax_enable_x64 else 5e-3
  base = _split_grand_blocks(matrices[0], int(positions.shape[0]))
  for matrix in matrices[1:]:
    blocks = _split_grand_blocks(matrix, int(positions.shape[0]))
    for name in ('UF', 'UC', 'DF', 'DC'):
      assert _relative_error(blocks[name], base[name]) < tol, name


@pytest.mark.slow
def test_grand_mobility_symmetry_and_psd():
  # Includes one overlapping pair (particles 0 and 1).
  positions = jnp.asarray([[0.1, 0.1, 0.1],
                           [0.3, 0.15, 0.12],
                           [0.15, 0.55, 0.6],
                           [0.6, 0.62, 0.55]], dtype=_dtype())
  a, xi, eta = 1.0, 0.75, 1.0
  box = _box(6.0)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_r, _ = rpy_real.build_Mr_grand_apply(space_fns, a, xi, eta, 5.5 / xi)
  state_r = init_r(positions)
  state_w = rpy_wave.build_grand_wave_modes(box, a, xi, eta, 28, 28, 28, 11)

  real = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_real.mr_grand_matvec(state_r, positions, F, C),
      int(positions.shape[0]))
  wave = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_wave.mw_grand_matvec(state_w, positions, F, C),
      int(positions.shape[0]))
  sym_tol = 2e-7 if jax.config.jax_enable_x64 else 2e-3
  psd_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-3
  _assert_symmetric_psd(real, sym_tol=sym_tol, psd_tol=psd_tol)
  _assert_symmetric_psd(wave, sym_tol=sym_tol, psd_tol=psd_tol)
  _assert_symmetric_psd(real + wave, sym_tol=sym_tol, psd_tol=psd_tol)


@pytest.mark.slow
def test_dense_and_ordered_sparse_real_space_agree():
  positions = jnp.asarray([[0.10, 0.20, 0.30],
                           [0.36, 0.22, 0.33],
                           [0.71, 0.64, 0.52]], dtype=_dtype())
  box = _box(9.0)
  dense_state, dense_apply = _build_grand_operator(
      positions, box, xi=0.9, neighbor_format=partition.NeighborListFormat.Dense)
  sparse_state, sparse_apply = _build_grand_operator(
      positions, box, xi=0.9, neighbor_format=partition.NeighborListFormat.OrderedSparse)
  Md = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_real.mr_grand_matvec(dense_state.real, positions, F, C), int(positions.shape[0]))
  Ms = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_real.mr_grand_matvec(sparse_state.real, positions, F, C), int(positions.shape[0]))
  np.testing.assert_allclose(Ms, Md, rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_ordered_sparse_self_image_contributions():
  """rcut > box: OrderedSparse must accumulate own-periodic-image terms.

  OrderedSparse drops (i, i) edges, which Dense uses to pick up self-image
  contributions whenever rcut exceeds the box; the grand lattice core adds
  them explicitly (the legacy UF-only core does not, so this guards the new
  path only).  Opposite images cancel for the odd UC/DF blocks, so a missing
  self-image pass shows up only in UF/DC.
  """
  positions = jnp.asarray([[0.1, 0.1, 0.1],
                           [0.52, 0.18, 0.12],
                           [0.15, 0.55, 0.6]], dtype=_dtype())
  box = _box(6.0)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  states = {}
  for fmt in (partition.NeighborListFormat.Dense,
              partition.NeighborListFormat.OrderedSparse):
    init_fn, _ = rpy_real.build_Mr_grand_apply(
        space_fns, 1.0, 0.55, 1.0, 10.0, neighbor_format=fmt)
    states[fmt] = init_fn(positions)
  Md = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_real.mr_grand_matvec(
          states[partition.NeighborListFormat.Dense], positions, F, C),
      int(positions.shape[0]))
  Ms = rtu._grand_dense_from_matvec(
      lambda F, C: rpy_real.mr_grand_matvec(
          states[partition.NeighborListFormat.OrderedSparse], positions, F, C),
      int(positions.shape[0]))
  if jax.config.jax_enable_x64:
    np.testing.assert_allclose(Ms, Md, rtol=1e-10, atol=1e-12)
  else:
    scale = np.linalg.norm(Md)
    assert np.linalg.norm(Ms - Md) / scale < 1e-4


@pytest.mark.slow
def test_dc_fallback_limit_matches_pair_expression():
  """The DC pair tensor approaches the exact r->0 limit K1(0)(C^T - 4C).

  The closed-form scalars converge linearly toward the limit down to
  r ~ 1e-2 a (below which float64 cancellation grows and the eps-clamped
  fallback path returns exactly the limit structure).  The convergence part
  requires float64: the skeleton coefficients cancel ~1/(r xi)^8 digits.
  """
  a = 0.6
  xi = 0.7
  eta = 1.0
  couplet = _traceless_np(np.asarray([[0.2, -0.4, 0.1],
                                      [0.5, -0.1, 0.3],
                                      [-0.2, 0.7, -0.1]]))
  limit = (float(Mr_self_dipole(a, xi)) / (6.0 * math.pi * eta)) * (
      couplet.T - 4.0 * couplet)
  if jax.config.jax_enable_x64:
    errors = []
    for scale in (1e-1, 3e-2, 1e-2):
      _, d_couplet = _real_pair_outputs(
          np.asarray([scale * a, 0.0, 0.0]), np.zeros(3), couplet, a=a, xi=xi, eta=eta)
      errors.append(_relative_error(d_couplet, limit))
    assert errors[-1] < 3e-2
    assert errors[0] > errors[-1]
  # Below the eps clamp the fallback returns the limit structure exactly
  # (up to the working precision of Mr_self_dipole itself).
  _, d_fallback = _real_pair_outputs(
      np.asarray([1e-6 * a, 0.0, 0.0]), np.zeros(3), couplet, a=a, xi=xi, eta=eta)
  tol = 1e-10 if jax.config.jax_enable_x64 else 1e-4
  assert _relative_error(d_fallback, limit) < tol


@pytest.mark.slow
def test_antisymmetric_couplet_reproduces_rotlet():
  """Pin the couplet index convention against external rotlet physics.

  A particle exerting torque T sources the force density
  ``f = (1/2) curl(T delta)``, i.e. the antisymmetric couplet
  ``C = -(1/2) eps . T`` in our convention (``f_m = -C_mn d_n delta``), and
  drives the far-field rotlet flow ``U = (T x r) / (8 pi eta r^3)``.

  This is the only gate with *external* ground truth for the couplet index
  convention: symmetry and xi-invariance hold under either index choice, and
  a wrong choice would silently break the Phase-2 constraint solve.  Together
  with ``test_real_space_pair_tensors_match_quadrature_reference`` (operator
  == quadrature tensors at machine precision) this pins the operator itself.
  """
  pytest.importorskip('scipy')
  import rpy_quadrature_reference as qref

  a = 0.4
  eta = 1.0
  r_vec = np.asarray([20.0 * a, 0.0, 0.0])
  torque = np.asarray([0.0, 0.0, 1.0])
  eps = np.zeros((3, 3, 3))
  for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    eps[i, j, k] = 1.0
    eps[j, i, k] = -1.0
  couplet = -0.5 * np.einsum('mnp,p->mn', eps, torque)
  U = np.einsum('imn,mn->i',
                np.asarray(qref.muc_tensor(r_vec, a, None)),
                couplet) / (6.0 * math.pi * eta)
  r = np.linalg.norm(r_vec)
  U_rotlet = np.cross(torque, r_vec) / (8.0 * math.pi * eta * r ** 3)
  # Finite-size corrections to the point rotlet are O((a/r)^2) = 2.5e-3.
  np.testing.assert_allclose(U, U_rotlet, rtol=1e-2, atol=1e-12)
