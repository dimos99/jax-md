"""Validation gates for Phase-3 constrained Brownian dynamics (midpoint SDAE).

Gates (spec Part E):

  0. Flat 11-component orthonormal layout: round trip and -- decisively --
     the dense flat grand operator is symmetric.  Lanczos assumes the
     Euclidean inner product, which equals the Frobenius pairing only in
     orthonormal coordinates; in drop-zz the flat operator is asymmetric
     and the square root would be silently wrong.
  1. Real-space slip covariance: samples of M^(r)^{1/2} dW converge to the
     dense M^(r)_grand probed from the matrix-free operator.
  2. Wave-space noise: (a) the gathered field is real (Hermitian conjugacy
     of the random Fourier modes -- the single most common split-sampler
     bug); (b) covariance converges to the dense M^(w)_grand.
  3. Combined slip covariance converges to (2 kT / dt) * M_grand.
  4. (slow) Two-particle Boltzmann distribution under a harmonic spring with
     the midpoint integrator (noise/drift consistency at ~1% mean / ~4% var
     precision; see the in-test power notes -- the isolated pair has no
     power against the drift omission specifically).
  5a. (slow) Non-interacting suspension: midpoint S(q) = 1 at low q while
     Euler-Maruyama shows the documented low-q suppression (3.5 sigma
     separation) -- the decisive falsifiable test of the many-body thermal
     drift, including its wave-space part.
  7. (slow) Large-dt weak consistency: no resolvable stationary bias at
     dt = 0.1 (k mu dt ~ 0.2/step); the O(dt) coefficient is below
     resolution, so a scaling exponent is not extractable (power note in
     the test).
  8. Regression: all earlier modes unchanged; API gating.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_md import space
from jax_md.hydro import rpy
from jax_md.hydro.rpy_brownian_constrained import (
    grand_jacobi_preconditioner,
    make_grand_slip_sampler,
    make_real_grand_slip_sampler,
)
from jax_md.hydro.rpy_real_det_dipole import build_Mr_grand_apply, mr_grand_matvec
from jax_md.hydro.rpy_wave_det_dipole import build_grand_wave_modes, mw_grand_matvec
from jax_md.hydro.rpy_wave_stoch import (
    _hermitian_gaussian_modes,
    build_Mw_grand_sqrt_sampler,
    grand_readout_noise_modes,
)
from jax_md.hydro.rpy_moments import (
    couplet_to_orthonormal,
    flat_to_grand,
    grand_to_flat,
    orthonormal_to_couplet,
    traceless,
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
      lattice_extent=1,
      real_space_mode='lattice',
      **builder_kwargs,
  )
  return init_fn(positions), apply_fn


def _grand_matvec_for(positions, box, **op_kwargs):
  """Fixed-state unconstrained grand matvec (F, C) -> (U, D)."""
  state, apply_fn = _build_operator(positions, box, use_stresslet=True,
                                    include_brownian=False, **op_kwargs)
  def grand_mv(F, C):
    (U, D), _ = apply_fn(state, positions, F, couplets=C)
    return U, D
  return grand_mv


def _test_positions(n=4, length=8.0, a=0.4, seed=7):
  pos = rtu._nonoverlap_positions(
      n, np.eye(3) * length, a=a, seed=seed)
  return jnp.asarray(pos, dtype=_dtype())


# -----------------------------------------------------------------------------
# Gate 0: flat orthonormal layout
# -----------------------------------------------------------------------------
def test_flat_layout_round_trip():
  rng = np.random.default_rng(3)
  C = traceless(jnp.asarray(rng.normal(size=(6, 3, 3)), dtype=_dtype()))
  U = jnp.asarray(rng.normal(size=(6, 3)), dtype=_dtype())

  c8 = couplet_to_orthonormal(C)
  assert c8.shape == (6, 8)
  assert _relative_error(orthonormal_to_couplet(c8), C) < _machine_tol()

  x = grand_to_flat(U, C)
  assert x.shape == (6, 11)
  U_back, C_back = flat_to_grand(x)
  assert _relative_error(U_back, U) < _machine_tol()
  assert _relative_error(C_back, C) < _machine_tol()

  # Euclidean product on flat coordinates equals the Frobenius pairing --
  # the property the Lanczos square root relies on.
  C2 = traceless(jnp.asarray(rng.normal(size=(6, 3, 3)), dtype=_dtype()))
  U2 = jnp.asarray(rng.normal(size=(6, 3)), dtype=_dtype())
  euclid = jnp.vdot(grand_to_flat(U, C), grand_to_flat(U2, C2))
  frob = jnp.vdot(U, U2) + jnp.vdot(C, C2)
  assert abs(float(euclid - frob)) < 1e-5 * max(abs(float(frob)), 1.0)


def test_flat_grand_operator_is_symmetric():
  """The dense flat grand operator must be symmetric (Lanczos prerequisite)."""
  box_length = 8.0
  positions = _test_positions(n=4, length=box_length)
  grand_mv = _grand_matvec_for(positions, _box(box_length))

  dense = rtu._grand_dense_from_matvec(grand_mv, 4)
  sym_tol = 1e-10 if jax.config.jax_enable_x64 else 2e-3
  assert _relative_error(dense, dense.T) < sym_tol

  # Cross-check: probing through the flat helpers reproduces the same matrix
  # (guards grand_to_flat/flat_to_grand against the test utility convention).
  ndof = 11 * 4
  dense_flat = np.zeros((ndof, ndof))
  eye = np.eye(ndof, dtype=np.float64)
  for col in range(ndof):
    F, C = flat_to_grand(jnp.asarray(eye[:, col].reshape(4, 11), dtype=_dtype()))
    U, D = grand_mv(F, C)
    dense_flat[:, col] = np.asarray(grand_to_flat(U, D)).reshape(-1)
  assert _relative_error(dense_flat, dense) < (
      1e-12 if jax.config.jax_enable_x64 else 1e-4)


# -----------------------------------------------------------------------------
# Gate 1: real-space slip sampler
# -----------------------------------------------------------------------------
_A = 0.4
_XI = 0.75
_ETA = 1.0
_RCUT = 3.2


def _real_grand_state(positions, length, **builder_kwargs):
  """Refreshed real-space-only grand state bound at `positions`."""
  box = _box(length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_fn, mr_apply = build_Mr_grand_apply(
      space_fns, _A, _XI, _ETA, _RCUT,
      lattice_extent=1, real_space_mode='lattice', **builder_kwargs)
  state0 = init_fn(positions)
  n = positions.shape[0]
  zeros_f = jnp.zeros((n, 3), dtype=_dtype())
  zeros_c = jnp.zeros((n, 3, 3), dtype=_dtype())
  _, real_state = mr_apply(state0, positions, zeros_f, zeros_c)
  return real_state


def test_grand_jacobi_matches_single_particle_self_block():
  """The preconditioner diagonal is exactly the 1-particle M^(r) self block."""
  positions = jnp.asarray([[0.5, 0.5, 0.5]], dtype=_dtype())
  length = 16.0  # rcut << L/2: no images, the dense probe is the self block
  real_state = _real_grand_state(positions, length)
  dense = rtu._grand_dense_from_matvec(
      lambda F, C: mr_grand_matvec(real_state, positions, F, C), 1)

  precond = grand_jacobi_preconditioner(_A, _XI, _ETA)
  g = np.asarray(precond.apply(jnp.ones((11,), dtype=_dtype())))
  d11 = 1.0 / g**2

  tol = 1e-10 if jax.config.jax_enable_x64 else 1e-4
  assert _relative_error(np.diag(dense), d11) < tol
  # Off-diagonal of the self block is zero in orthonormal coordinates.
  off = dense - np.diag(np.diag(dense))
  assert np.abs(off).max() < tol * np.abs(d11).max()


def _covariance_gate(samples_flat, reference, n_samples, k_sigma=3.0):
  cov, mean = rtu._sample_covariance(np.asarray(samples_flat))
  scale = rtu._wishart_frobenius_relative_scale(reference, n_samples)
  cov_err = rtu._frobenius_relative_error(cov, reference)
  assert cov_err <= k_sigma * scale, (
      f'covariance error {cov_err:.3e} > {k_sigma} x Wishart scale {scale:.3e}')
  # Mean must vanish like 1/sqrt(S) relative to the per-mode std.
  per_mode_std = np.sqrt(np.maximum(np.diag(reference), 1e-30))
  mean_z = np.abs(mean) / (per_mode_std / np.sqrt(n_samples))
  assert mean_z.max() < 5.0, f'max standardized mean {mean_z.max():.2f}'


def _run_real_slip_covariance(n_samples, iters, tol, k_sigma=3.0):
  length = 8.0
  positions = _test_positions(n=4, length=length)
  real_state = _real_grand_state(positions, length)

  reference = rtu._grand_dense_from_matvec(
      lambda F, C: mr_grand_matvec(real_state, positions, F, C), 4)
  # PSD of the real-space split (Lanczos clips negative Ritz values, which
  # would silently bias the covariance if the split were indefinite).
  eigs = np.linalg.eigvalsh(0.5 * (reference + reference.T))
  psd_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-3
  assert eigs.min() > -psd_tol * max(eigs.max(), 1e-15)

  sampler = make_real_grand_slip_sampler(
      real_state=real_state,
      positions=positions,
      preconditioner=grand_jacobi_preconditioner(_A, _XI, _ETA),
      iters=iters,
      tol=tol,
  )

  @jax.jit
  def draw(key):
    U, D, info = sampler(key)
    return grand_to_flat(U, D).reshape(-1), info

  keys = jax.random.split(jax.random.PRNGKey(11), n_samples)
  samples, infos = jax.lax.map(draw, keys)
  assert bool(np.all(np.asarray(infos['real_sqrt_converged']))), (
      f"Lanczos did not converge: max rel_change "
      f"{np.max(np.asarray(infos['real_sqrt_rel_change'])):.3e}")
  _covariance_gate(np.asarray(samples), reference, n_samples, k_sigma=k_sigma)


def test_real_slip_covariance_lightweight():
  _run_real_slip_covariance(n_samples=800, iters=30, tol=1e-3)


@pytest.mark.slow
def test_real_slip_covariance_rigorous():
  _run_real_slip_covariance(n_samples=2000, iters=40, tol=1e-6)


# -----------------------------------------------------------------------------
# Gate 2: wave-space noise (conjugacy + covariance)
# -----------------------------------------------------------------------------
_MGRID = 10
_PSUP = 5


def _grand_wave_state(length):
  return build_grand_wave_modes(
      np.eye(3) * length, _A, _XI, _ETA, _MGRID, _MGRID, _MGRID, _PSUP)


def test_wave_noise_hermitian_conjugacy():
  """The random Fourier field must be Hermitian so its IFFT is real.

  ``ifft_vec`` silently discards the imaginary part, so a broken mirroring
  (wrong half-space, mistreated self-conjugate/Nyquist modes) corrupts the
  covariance without any visible failure -- this test probes the complex
  IFFT *before* the .real truncation.
  """
  wave_state = _grand_wave_state(8.0)
  modes = wave_state.modes
  Pshape, Pdip, k = modes['Pshape'], modes['Pdip'], modes['k']
  Bhalf = jnp.asarray(modes['Bhalf'],
                      dtype=jnp.complex128 if jax.config.jax_enable_x64
                      else jnp.complex64)

  for seed in range(3):
    draw = _hermitian_gaussian_modes(
        jax.random.PRNGKey(seed), (_MGRID, _MGRID, _MGRID, 3))
    # Raw noise satisfies G(-k) = conj(G(k)) and zero k=0 mode.
    G = np.asarray(draw)
    neg = np.conj(G[(-np.arange(_MGRID)) % _MGRID][:, (-np.arange(_MGRID)) % _MGRID]
                  [:, :, (-np.arange(_MGRID)) % _MGRID])
    assert np.abs(G - neg).max() < 1e-12 * max(np.abs(G).max(), 1.0)
    assert np.abs(G[0, 0, 0]).max() == 0.0

    # The full 11-channel noise grid must be Hermitian.  The naive readout
    # ``i Pdip uq (x) k`` is NOT at self-conjugate (Nyquist) modes, where uq
    # is real and the gradient channel comes out purely imaginary;
    # grand_readout_noise_modes substitutes an independent real draw there.
    out_q = grand_readout_noise_modes(
        jax.random.PRNGKey(100 + seed), Pshape=Pshape, Pdip=Pdip, k=k,
        Bhalf_complex=Bhalf, Mx=_MGRID, My=_MGRID, Mz=_MGRID)
    out_grid = np.asarray(jnp.fft.ifftn(out_q, axes=(0, 1, 2)))
    scale = max(np.abs(out_grid.real).max(), 1e-30)
    rel_imag = np.abs(out_grid.imag).max() / scale
    assert rel_imag < (1e-12 if jax.config.jax_enable_x64 else 1e-4), (
        f'IFFT of the noise grid is not real: rel imag {rel_imag:.3e}')


def _run_wave_slip_covariance(n_samples, k_sigma=3.0):
  length = 8.0
  positions = _test_positions(n=4, length=length)
  wave_state = _grand_wave_state(length)

  reference = rtu._grand_dense_from_matvec(
      lambda F, C: mw_grand_matvec(wave_state, positions, F, C), 4)
  eigs = np.linalg.eigvalsh(0.5 * (reference + reference.T))
  psd_tol = 1e-10 if jax.config.jax_enable_x64 else 1e-3
  assert eigs.min() > -psd_tol * max(eigs.max(), 1e-15)

  sampler = build_Mw_grand_sqrt_sampler(wave_state)

  @jax.jit
  def draw(key):
    U, D = sampler(key, positions)
    return grand_to_flat(U, D).reshape(-1)

  keys = jax.random.split(jax.random.PRNGKey(23), n_samples)
  samples = jax.lax.map(draw, keys)
  _covariance_gate(np.asarray(samples), reference, n_samples, k_sigma=k_sigma)


def test_wave_slip_covariance_lightweight():
  _run_wave_slip_covariance(n_samples=800)


@pytest.mark.slow
def test_wave_slip_covariance_rigorous():
  _run_wave_slip_covariance(n_samples=2000)


# -----------------------------------------------------------------------------
# Gate 3: combined unconstrained slip covariance
# -----------------------------------------------------------------------------
def _run_combined_slip_covariance(n_samples, kT, dt, k_sigma=3.0):
  length = 8.0
  positions = _test_positions(n=4, length=length)
  real_state = _real_grand_state(positions, length)
  wave_state = _grand_wave_state(length)

  def grand_mv(F, C):
    Ur, Dr = mr_grand_matvec(real_state, positions, F, C)
    Uw, Dw = mw_grand_matvec(wave_state, positions, F, C)
    return Ur + Uw, traceless(Dr + Dw)

  reference = (2.0 * kT / dt) * rtu._grand_dense_from_matvec(grand_mv, 4)

  sampler = make_grand_slip_sampler(
      real_sampler=make_real_grand_slip_sampler(
          real_state=real_state,
          positions=positions,
          preconditioner=grand_jacobi_preconditioner(_A, _XI, _ETA),
          iters=30,
          tol=1e-3,
      ),
      wave_sampler=lambda key: build_Mw_grand_sqrt_sampler(wave_state)(
          key, positions),
      kT=kT,
      dt=dt,
  )

  @jax.jit
  def draw(key):
    U, D, _ = sampler(key)
    return grand_to_flat(U, D).reshape(-1)

  keys = jax.random.split(jax.random.PRNGKey(37), n_samples)
  samples = jax.lax.map(draw, keys)
  _covariance_gate(np.asarray(samples), reference, n_samples, k_sigma=k_sigma)


def test_combined_slip_covariance_lightweight():
  # kT = dt = 0.5 exercises the 2 kT / dt prefactor (scale = sqrt(2) != 1).
  _run_combined_slip_covariance(n_samples=800, kT=0.5, dt=0.5)


@pytest.mark.slow
def test_combined_slip_covariance_rigorous():
  _run_combined_slip_covariance(n_samples=2000, kT=1.3, dt=0.05)


# -----------------------------------------------------------------------------
# Gate 8: stepper wiring, API gating, regression
# -----------------------------------------------------------------------------
def _constrained_builder(positions, length, **extra):
  box = _box(length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  return rpy.build_rpy_mobility(
      space_fns, a=_A, xi=_XI, eta=_ETA, rcut=_RCUT, Mgrid=_MGRID, P=_PSUP,
      lattice_extent=1, real_space_mode='lattice',
      use_stresslet=True, constrained=True, **extra)


def test_step_kT_zero_matches_deterministic_constrained():
  """At kT = 0 the midpoint step degenerates to the deterministic
  constrained mobility: zero slip, x_mid = x_k, one solve with the applied
  forces only.  Catches mis-wired applied_output / scale bugs at O(1)."""
  length = 8.0
  dt = 0.1
  positions = _test_positions(n=4, length=length)
  rng = np.random.default_rng(5)
  forces = jnp.asarray(rng.normal(size=(4, 3)), dtype=_dtype())
  init_fn, apply_fn = _constrained_builder(positions, length, solve_tol=1e-8)

  (U_det, _), _ = apply_fn(init_fn(positions), positions, forces)

  binit, step = apply_fn.make_brownian_step(
      lambda x, **kw: forces, kT=0.0, dt=dt)
  st = binit(positions, jax.random.PRNGKey(0))
  st2, info = jax.jit(step)(st)
  U_step = (np.asarray(st2.positions) - np.asarray(st.positions)) * length / dt

  tol = 1e-7 if jax.config.jax_enable_x64 else 5e-4
  assert _relative_error(U_step, U_det) < tol
  assert not bool(info['nbr_did_overflow'])


def test_step_determinism_replay():
  """Same initial state -> bit-identical trajectory (overflow-replay
  contract relies on the PRNG key living in the state)."""
  length = 8.0
  positions = _test_positions(n=4, length=length)
  _, apply_fn = _constrained_builder(positions, length)
  binit, step = apply_fn.make_brownian_step(None, kT=1.0, dt=1e-3, mr_iters=30)

  step_j = jax.jit(step)
  def run(n):
    st = binit(positions, jax.random.PRNGKey(11))
    for _ in range(n):
      st, _ = step_j(st)
    return np.asarray(st.positions)

  np.testing.assert_array_equal(run(3), run(3))


def test_em_and_midpoint_differ_with_noise():
  """EM and midpoint must produce different trajectories at kT > 0 with the
  same key (collapsing the midpoint to EM loses the thermal drift; gate 4
  is the physical check, this is the cheap structural one)."""
  length = 8.0
  positions = _test_positions(n=4, length=length)
  _, apply_fn = _constrained_builder(positions, length)

  results = {}
  for kind in ('midpoint', 'euler_maruyama'):
    binit, step = apply_fn.make_brownian_step(
        None, kT=1.0, dt=1e-2, integrator=kind, mr_iters=30)
    st, _ = jax.jit(step)(binit(positions, jax.random.PRNGKey(4)))
    results[kind] = np.asarray(st.positions)
  assert np.abs(results['midpoint'] - results['euler_maruyama']).max() > 0.0


def test_api_gating():
  length = 8.0
  positions = _test_positions(n=4, length=length)
  box = _box(length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  common = dict(a=_A, xi=_XI, eta=_ETA, rcut=_RCUT, Mgrid=_MGRID, P=_PSUP,
                lattice_extent=1, real_space_mode='lattice')

  # Attribute present only for constrained + stresslet + brownian.
  _, apply_c = rpy.build_rpy_mobility(space_fns, use_stresslet=True,
                                      constrained=True, **common)
  assert hasattr(apply_c, 'make_brownian_step')
  _, apply_nb = rpy.build_rpy_mobility(space_fns, use_stresslet=True,
                                       constrained=True,
                                       include_brownian=False, **common)
  assert not hasattr(apply_nb, 'make_brownian_step')
  _, apply_g = rpy.build_rpy_mobility(space_fns, use_stresslet=True, **common)
  assert not hasattr(apply_g, 'make_brownian_step')
  _, apply_p = rpy.build_rpy_mobility(space_fns, **common)
  assert not hasattr(apply_p, 'make_brownian_step')

  # Constrained apply with brownian_key points at the stepper.
  init_c, _ = rpy.build_rpy_mobility(space_fns, use_stresslet=True,
                                     constrained=True, **common)
  state = init_c(positions)
  forces = jnp.zeros_like(positions)
  with pytest.raises(ValueError, match='make_brownian_step'):
    apply_c(state, positions, forces, brownian_key=jax.random.PRNGKey(0),
            kT=1.0, dt=1e-3)

  # Unconstrained grand Brownian stays unimplemented.
  init_g, apply_g2 = rpy.build_rpy_mobility(space_fns, use_stresslet=True,
                                            **common)
  with pytest.raises(NotImplementedError):
    apply_g2(init_g(positions), positions, forces,
             brownian_key=jax.random.PRNGKey(0), kT=1.0, dt=1e-3)

  # torque_fn requires with_torque; bad integrator rejected.
  with pytest.raises(ValueError, match='with_torque'):
    apply_c.make_brownian_step(None, kT=1.0, dt=1e-3,
                               torque_fn=lambda x, **kw: jnp.zeros_like(x))
  with pytest.raises(ValueError, match='integrator'):
    apply_c.make_brownian_step(None, kT=1.0, dt=1e-3, integrator='heun')


# -----------------------------------------------------------------------------
# Gate 4: two-particle Boltzmann distribution (slow)
# -----------------------------------------------------------------------------
def _spring_system(*, kT, dt, integrator, k_spring=10.0, r0=1.2, length=6.0,
                   a=0.5):
  box = _box(length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  disp = space_fns[0]

  def force_fn(x, **kw):
    dr = jax.vmap(disp)(x, x[jnp.array([1, 0])])  # r_i - r_other, real units
    r = jnp.linalg.norm(dr, axis=-1, keepdims=True)
    return -k_spring * (r - r0) * dr / r

  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns, a=a, xi=0.75, eta=1.0, rcut=2.8, Mgrid=10, P=_PSUP,
      lattice_extent=1, real_space_mode='lattice',
      use_stresslet=True, constrained=True)
  binit, step = apply_fn.make_brownian_step(
      force_fn, kT=kT, dt=dt, integrator=integrator, mr_iters=20)

  def separation(state):
    dr = disp(state.positions[0], state.positions[1])
    return jnp.linalg.norm(dr)

  pos0 = jnp.asarray([[0.4, 0.5, 0.5], [0.4 + r0 / length, 0.5, 0.5]],
                     dtype=_dtype())
  return binit, step, separation, pos0


def _boltzmann_moments(k_spring, r0, kT, r_max):
  """Moments of p(r) ~ r^2 exp(-k (r - r0)^2 / (2 kT)) on (0, r_max)."""
  r = np.linspace(1e-6, r_max, 20001)
  w = r**2 * np.exp(-k_spring * (r - r0)**2 / (2.0 * kT))
  w /= np.trapezoid(w, r)
  mean = np.trapezoid(r * w, r)
  var = np.trapezoid((r - mean)**2 * w, r)
  return mean, var


def _run_spring_trajectory(integrator, *, kT, dt, n_steps, burn_in, thin,
                           seed, k_spring=10.0, r0=1.2):
  from jax_md.hydro.rpy_brownian_constrained import run_brownian_chunked
  binit, step, separation, pos0 = _spring_system(
      kT=kT, dt=dt, integrator=integrator, k_spring=k_spring, r0=r0)
  st = binit(pos0, jax.random.PRNGKey(seed))
  st, _ = run_brownian_chunked(step, binit, st, burn_in, chunk_size=burn_in)
  st, obs = run_brownian_chunked(
      step, binit, st, n_steps, chunk_size=min(n_steps, 25000),
      observe_fn=separation)
  r = np.concatenate([np.asarray(o) for o in obs])[::thin]
  return r


def _blocked_se(samples, n_blocks=30):
  """Autocorrelation-robust standard error of the mean via block averaging."""
  m = len(samples) // n_blocks
  blocks = np.asarray(samples[:m * n_blocks]).reshape(n_blocks, m).mean(axis=1)
  return blocks.mean(), blocks.std(ddof=1) / np.sqrt(n_blocks)


@pytest.mark.slow
def test_two_particle_boltzmann_midpoint():
  """Boltzmann gate (paper Fig. 3 analogue): the midpoint integrator must
  sample the analytic equilibrium distribution of a harmonic pair.

  Power note (calibrated 2026-06-12): at these parameters the gate resolves
  the mean to ~1% and the variance to ~4% (5 sigma); a mis-scaled slip
  (e.g. a sqrt(2) covariance error) fails at >50 sigma.  It has *no* power
  against the drift omission specifically: an Euler-Maruyama control run at
  the same budget was statistically indistinguishable from Boltzmann (the
  isolated-pair constrained-mobility divergence is tiny at r ~ 2.5a, err
  0.003 +- 0.0034 in the variance), and so was EM at dt up to 0.1.  The
  falsifiable drift detection lives in the many-body structure-factor gate
  (test_ideal_suspension_structure_factor_midpoint_vs_em, EM separated at
  3.5 sigma) and in the structural test_em_and_midpoint_differ_with_noise.
  """
  kT, dt, k_spring, r0 = 0.5, 2e-3, 10.0, 1.2
  mean_ref, var_ref = _boltzmann_moments(k_spring, r0, kT, r_max=3.0)

  r_mid = _run_spring_trajectory('midpoint', kT=kT, dt=dt, n_steps=150000,
                                 burn_in=20000, thin=10, seed=42)
  mean_mid, se_mid = _blocked_se(r_mid)
  err_mid = abs(mean_mid - mean_ref)
  assert err_mid < 5.0 * se_mid, (
      f'midpoint mean separation {mean_mid:.4f} vs Boltzmann {mean_ref:.4f} '
      f'({err_mid / se_mid:.1f} sigma, se={se_mid:.4f})')
  # Variance with a blocked (autocorrelation-robust) error bar.
  _, se_var_mid = _blocked_se((r_mid - r_mid.mean())**2)
  var_mid = np.var(r_mid)
  err_var_mid = abs(var_mid - var_ref)
  assert err_var_mid < 5.0 * se_var_mid, (
      f'midpoint var {var_mid:.4f} vs Boltzmann {var_ref:.4f} '
      f'({err_var_mid / se_var_mid:.1f} sigma)')


# -----------------------------------------------------------------------------
# Gate 7: large-dt weak consistency (slow)
# -----------------------------------------------------------------------------
@pytest.mark.slow
def test_large_dt_midpoint_consistency():
  """Weak consistency at an aggressive time step: at dt = 0.1 the spring
  relaxes by k mu dt ~ 0.2 per step, yet the midpoint stationary state must
  still match Boltzmann within the run's ~0.5% (mean) / ~2% (var) precision.

  Power note (calibrated 2026-06-12): the midpoint O(dt) bias coefficient
  for this observable is below resolution even at dt = 0.1 (mean err
  0.0036 +- 0.0049, var err 0.0017 +- 0.0010), so a first-order *scaling
  exponent* cannot be honestly extracted at any feasible budget -- this
  gate instead pins the absence of resolvable bias at 50x the production
  step.  A drift or noise error of relative size O(k mu dt) = 0.2 would
  fail at >40 sigma.
  """
  kT, k_spring, r0 = 0.5, 10.0, 1.2
  mean_ref, var_ref = _boltzmann_moments(k_spring, r0, kT, r_max=3.0)

  r_large = _run_spring_trajectory('midpoint', kT=kT, dt=0.1,
                                   n_steps=60000, burn_in=5000, thin=5,
                                   seed=101)
  mean_l, se_l = _blocked_se(r_large)
  _, se_var_l = _blocked_se((r_large - r_large.mean())**2)
  var_l = np.var(r_large)
  err_l = abs(mean_l - mean_ref)
  err_var_l = abs(var_l - var_ref)
  assert err_l < 5.0 * se_l, (
      f'large-dt mean {mean_l:.4f} vs Boltzmann {mean_ref:.4f} '
      f'({err_l / se_l:.1f} sigma)')
  assert err_var_l < 5.0 * se_var_l, (
      f'large-dt var {var_l:.4f} vs Boltzmann {var_ref:.4f} '
      f'({err_var_l / se_var_l:.1f} sigma)')


# -----------------------------------------------------------------------------
# Gate 5a: non-interacting suspension S(q) = 1, midpoint vs EM (slow)
# -----------------------------------------------------------------------------
def _structure_factor_low_q(positions_frac_batch):
  """S(q) averaged over the lowest three q shells (|n|^2 = 1, 2, 3).

  positions_frac_batch: (T, N, 3) fractional snapshots.  Because S(q) is
  computed from fractional coordinates, the box length drops out.
  """
  nvec = []
  for nx in range(-1, 2):
    for ny in range(-1, 2):
      for nz in range(-1, 2):
        if 0 < nx * nx + ny * ny + nz * nz <= 3:
          # Half-space only: S(-q) = S(q) carries no new information.
          if (nx, ny, nz) > (-nx, -ny, -nz):
            nvec.append((nx, ny, nz))
  nvec = np.asarray(sorted(set(nvec)), dtype=np.float64)  # (Q, 3)
  x = np.asarray(positions_frac_batch)  # (T, N, 3)
  phase = np.exp(2j * np.pi * np.einsum('tnd,qd->tnq', x, nvec))
  rho = phase.sum(axis=1)  # (T, Q)
  n_particles = x.shape[1]
  return (np.abs(rho)**2 / n_particles).mean(axis=0)  # (Q,)


def _run_ideal_suspension(integrator, *, n, phi, kT, dt, n_steps, seed,
                          replicas):
  from jax_md.hydro.rpy_brownian_constrained import run_brownian_chunked
  # Accuracy note: this gate tests FDT *consistency* (noise/drift matched to
  # whatever splitting M is implemented), which holds at any Ewald
  # parameters -- so the split is deliberately cheap/loose to buy physical
  # time per wall-clock second.  The small box (n = 16, L ~ 6.9a) makes the
  # lowest q modes decorrelate in ~150 steps at dt = 0.15, which is what
  # gives the estimator its statistical power.
  a = 1.0
  length = (4.0 * np.pi * n * a**3 / (3.0 * phi))**(1.0 / 3.0)
  box = _box(length)
  space_fns = space.periodic_general(box, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns, a=a, xi=0.7, eta=1.0, rcut=3.0, Mgrid=10, P=5,
      lattice_extent=1, real_space_mode='lattice',
      use_stresslet=True, constrained=True)
  binit, step = apply_fn.make_brownian_step(
      None, kT=kT, dt=dt, integrator=integrator, mr_iters=30)

  sq_acc = []
  for rep in range(replicas):
    rng = np.random.default_rng(seed + rep)
    pos0 = jnp.asarray(rng.random((n, 3)), dtype=_dtype())  # exact uniform law
    st = binit(pos0, jax.random.PRNGKey(seed + rep))
    st, obs = run_brownian_chunked(
        step, binit, st, n_steps, chunk_size=min(n_steps, 1000),
        observe_fn=lambda s: s.positions)
    snaps = np.concatenate([np.asarray(o) for o in obs])[::20]
    sq_acc.append(_structure_factor_low_q(snaps))
  sq = np.asarray(sq_acc)  # (replicas, Q)
  per_replica = sq.mean(axis=1)
  return sq.mean(), per_replica


@pytest.mark.slow
def test_ideal_suspension_structure_factor_midpoint_vs_em():
  """Non-interacting (ideal) particles at phi = 0.2 with full hydrodynamic
  coupling: the uniform distribution is stationary, so the midpoint scheme
  must keep S(q) = 1 at low q.  Euler-Maruyama omits the many-body thermal
  drift k_B T div R_FU^{-1} -- an O(1) error in the Fokker-Planck equation
  -- and drives the low-q structure away from 1 (paper Figs. 4-5).  This is
  the only in-budget gate sensitive to the *wave-space* (collective) part
  of the drift.
  """
  common = dict(n=16, phi=0.2, kT=1.0, dt=0.15, n_steps=4000, seed=70,
                replicas=4)
  sq_mid, reps_mid = _run_ideal_suspension('midpoint', **common)
  sq_em, reps_em = _run_ideal_suspension('euler_maruyama', **common)

  # Honest error bars from the independent-replica scatter.
  se_mid = reps_mid.std(ddof=1) / np.sqrt(len(reps_mid))
  se_em = reps_em.std(ddof=1) / np.sqrt(len(reps_em))
  se_diff = np.hypot(se_mid, se_em)

  # (i) Midpoint preserves the uniform stationary law: S consistent with 1.
  assert abs(sq_mid - 1.0) < 3.5 * se_mid, (
      f'midpoint S(q->0) = {sq_mid:.3f} +- {se_mid:.3f} deviates from 1 '
      f'(replicas: {np.round(reps_mid, 3)})')
  # (ii) EM shows the documented low-q suppression (one-sided)...
  assert sq_em < 1.0 - 3.0 * se_em, (
      f'EM S(q->0) = {sq_em:.3f} +- {se_em:.3f} is not suppressed below 1 '
      f'-- the gate has no power to detect the omitted drift.')
  # (iii) ...and is cleanly separated from the midpoint result.
  assert sq_mid - sq_em > 2.5 * se_diff, (
      f'midpoint {sq_mid:.3f} vs EM {sq_em:.3f} not separated '
      f'(se_diff {se_diff:.3f}).')


def test_shear_step_gamma_zero_matches_static():
  """Live-box stepper at gamma = 0 equals the static-box stepper (same key)."""
  length = 6.0
  box = _box(length)
  positions = rtu._nonoverlap_positions(4, np.asarray(box), a=_A, seed=16)

  def _shear_box_fn(**kwargs):
    gamma = kwargs.get('gamma_xy', 0.0)
    deformed = np.eye(3)
    deformed[0, 1] = gamma
    return jnp.asarray(deformed, dtype=_dtype()) @ box

  static_space = space.periodic_general(box, fractional_coordinates=True)
  shear_space = static_space + (_shear_box_fn,)
  common = dict(a=_A, xi=_XI, eta=_ETA, rcut=2.8, Mgrid=12, P=_PSUP,
                lattice_extent=1, real_space_mode='lattice',
                use_stresslet=True, constrained=True, solve_tol=1e-8)
  _, apply_s = rpy.build_rpy_mobility(static_space, **common)
  _, apply_l = rpy.build_rpy_mobility(shear_space, **common)

  kT, dt = 0.7, 1e-3
  binit_s, step_s = apply_s.make_brownian_step(None, kT=kT, dt=dt, mr_iters=30)
  binit_l, step_l = apply_l.make_brownian_step(None, kT=kT, dt=dt, mr_iters=30)

  st_s, _ = step_s(binit_s(positions, jax.random.PRNGKey(8)))
  st_l, _ = step_l(binit_l(positions, jax.random.PRNGKey(8), gamma_xy=0.0),
                   gamma_xy=0.0)
  tol = 1e-7 if jax.config.jax_enable_x64 else 1e-3
  assert _relative_error(st_l.positions, st_s.positions) < tol
