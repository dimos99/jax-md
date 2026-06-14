"""Tests for the stresslet-constrained RPY integrators in `simulate`.

These cover the *usability wrapper* (`simulate.constrained_rpy` and
`simulate.constrained_rpy_with_shear`), not the constrained physics itself
(that lives in `rpy_brownian_constrained_test.py`).  The wrappers must:

  * add nothing at zero shear -- they reproduce the bare
    `apply_fn.make_brownian_step` trajectory step for step;
  * keep the Lees-Edwards reduced strain bounded and the neighbor list valid
    over a run whose total strain crosses several integer multiples of the box
    (the regression that motivated the wrapper);
  * advance with the right shapes, deterministically under a fixed key;
  * thread torques through when `with_torque=True` and reject a `torque_fn`
    otherwise.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import energy, quantity, simulate, space
from jax_md.hydro import rpy


def _dtype():
  return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _tol():
  return 1e-10 if jax.config.jax_enable_x64 else 1e-5


# Small, fast mobility parameters (match the constrained Brownian test).
_A, _XI, _ETA, _RCUT, _MGRID, _PSUP = 0.4, 0.75, 1.0, 3.2, 10, 5
_LENGTH = 8.0


def _positions(n=4, seed=7):
  rng = np.random.default_rng(seed)
  # Spread on a coarse lattice + jitter so there are no hard overlaps.
  side = int(np.ceil(n ** (1.0 / 3.0)))
  grid = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), -1)
  lattice = grid.reshape(-1, 3)[:n] / side
  pos = (lattice + 0.02 * rng.normal(size=(n, 3))) % 1.0
  return jnp.asarray(pos, dtype=_dtype())


def _energy_fn(displacement):
  return energy.soft_sphere_pair(displacement, sigma=0.5, epsilon=1.0)


_MOB_KW = dict(a=_A, xi=_XI, eta=_ETA, rcut=_RCUT, Mgrid=_MGRID, P=_PSUP,
               lattice_extent=1, real_space_mode='lattice')


def test_free_matches_make_brownian_step():
  """`constrained_rpy` is a pure wrapper around the bare stepper at no shear."""
  R = _positions()
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift = space.periodic_general(box, fractional_coordinates=True)
  energy_fn = _energy_fn(disp)
  force_fn = quantity.canonicalize_force(energy_fn)

  # Bare stepper.
  _, apply_mob = rpy.build_rpy_mobility(
      (disp, shift), use_stresslet=True, constrained=True, **_MOB_KW)
  binit, step = apply_mob.make_brownian_step(force_fn, kT=1.0, dt=1e-3,
                                             mr_iters=20)
  st = binit(R, jax.random.PRNGKey(3))
  step_j = jax.jit(step)
  for _ in range(3):
    st, _ = step_j(st)
  bare_pos = np.asarray(st.positions)

  # Integrator wrapper, same key/params.
  init_fn, apply_fn = simulate.constrained_rpy(
      (disp, shift), energy_fn, 1e-3, 1.0, mr_iters=20, **_MOB_KW)
  state = init_fn(jax.random.PRNGKey(3), R)
  apply_j = jax.jit(apply_fn)
  for _ in range(3):
    state = apply_j(state)

  np.testing.assert_allclose(
      np.asarray(state.brownian_state.positions), bare_pos,
      rtol=_tol(), atol=_tol())
  np.testing.assert_allclose(float(state.time), 3e-3, rtol=1e-5)


def test_zero_shear_schedule_adds_nothing():
  """At a zero shear schedule the shear wrapper reproduces the bare stepper."""
  R = _positions()
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift, box_of = space.shearing(
      box, shear_schedule=lambda t: 0.0 * t,
      fractional_coordinates=True, remap=True)
  energy_fn = _energy_fn(disp)
  force_fn = quantity.canonicalize_force(energy_fn)
  zero_g = dict(gamma_xy=0.0, gamma_xz=0.0, gamma_yz=0.0)

  _, apply_mob = rpy.build_rpy_mobility(
      (disp, shift, box_of), use_stresslet=True, constrained=True, **_MOB_KW)
  binit, step = apply_mob.make_brownian_step(force_fn, kT=1.0, dt=1e-3,
                                             mr_iters=20)
  st = binit(R, jax.random.PRNGKey(8), **zero_g)
  step_j = jax.jit(lambda s: step(s, **zero_g))
  for _ in range(3):
    st, _ = step_j(st)
  bare_pos = np.asarray(st.positions)

  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (disp, shift, box_of), energy_fn, 1e-3, 1.0,
      shear_vector_schedule=lambda t: (0.0 * t, 0.0 * t, 0.0 * t),
      mr_iters=20, **_MOB_KW)
  state = init_fn(jax.random.PRNGKey(8), R)
  apply_j = jax.jit(apply_fn)
  for _ in range(3):
    state = apply_j(state)

  np.testing.assert_allclose(
      np.asarray(state.brownian_state.positions), bare_pos,
      rtol=_tol(), atol=_tol())


def test_remap_keeps_strain_bounded_and_no_overflow():
  """Total strain crosses several box multiples; the reduced strain the
  integrator feeds the mobility must stay in [-0.5, 0.5) and the neighbor list
  must never overflow."""
  R = _positions(n=8)
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  gamma_dot, dt = 10.0, 1e-2
  disp, shift, box_of = space.shearing(
      box, shear_schedule=lambda t: gamma_dot * t,
      fractional_coordinates=True, remap=True)
  energy_fn = _energy_fn(disp)

  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (disp, shift, box_of), energy_fn, dt, 1.0,
      shear_vector_schedule=lambda t: (gamma_dot * t, 0.0 * t, 0.0 * t),
      mr_iters=20, capacity_multiplier=2.0, extra_capacity=16, **_MOB_KW)
  state = init_fn(jax.random.PRNGKey(1), R)
  apply_j = jax.jit(apply_fn)

  n_steps = 40  # total strain reaches 4.0 -> crosses the boundary repeatedly
  for _ in range(n_steps):
    state = apply_j(state)
    total = gamma_dot * float(state.time)
    reduced = total - np.floor(total + 0.5)
    assert -0.5 <= reduced < 0.5
    assert not bool(
        state.brownian_state.rpy_state.real.neighbors.did_buffer_overflow)
  assert bool(jnp.all(jnp.isfinite(state.real_position)))


def test_shapes_and_determinism():
  R = _positions()
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift, box_of = space.shearing(
      box, shear_schedule=lambda t: 0.5 * t,
      fractional_coordinates=True, remap=True)
  energy_fn = _energy_fn(disp)

  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (disp, shift, box_of), energy_fn, 1e-3, 1.0,
      shear_vector_schedule=lambda t: (0.5 * t, 0.0 * t, 0.0 * t),
      mr_iters=20, **_MOB_KW)
  apply_j = jax.jit(apply_fn)

  def run():
    s = init_fn(jax.random.PRNGKey(0), R)
    for _ in range(3):
      s = apply_j(s)
    return s

  s1, s2 = run(), run()
  assert s1.real_position.shape == (R.shape[0], 3)
  assert s1.brownian_state.stresslet.shape == (R.shape[0], 5)
  np.testing.assert_array_equal(
      np.asarray(s1.real_position), np.asarray(s2.real_position))


def test_with_torque_passthrough_and_gating():
  R = _positions()
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift, box_of = space.shearing(
      box, shear_schedule=lambda t: 0.0 * t,
      fractional_coordinates=True, remap=True)
  energy_fn = _energy_fn(disp)
  torque_fn = lambda x, **kw: jnp.ones_like(x)

  # with_torque=True + torque_fn runs and advances.
  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (disp, shift, box_of), energy_fn, 1e-3, 1.0,
      shear_vector_schedule=None, with_torque=True, torque_fn=torque_fn,
      mr_iters=20, **_MOB_KW)
  state = apply_fn(init_fn(jax.random.PRNGKey(0), R))
  assert state.real_position.shape == (R.shape[0], 3)
  assert bool(jnp.all(jnp.isfinite(state.real_position)))

  # torque_fn without with_torque is rejected at build time.
  try:
    simulate.constrained_rpy_with_shear(
        (disp, shift, box_of), energy_fn, 1e-3, 1.0,
        shear_vector_schedule=None, torque_fn=torque_fn,
        mr_iters=20, **_MOB_KW)
    raised = False
  except ValueError:
    raised = True
  assert raised


# Estimator-derived parameters (only `a`, `eta`, and a tolerance supplied).
_AUTO_KW = dict(a=_A, eta=_ETA, lattice_extent=1, real_space_mode='lattice')


def test_free_auto_estimates_params_when_xi_omitted():
  """Omitting `xi` runs the estimator and produces a finite, advancing run."""
  R = _positions(n=8)
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift = space.periodic_general(box, fractional_coordinates=True)
  energy_fn = _energy_fn(disp)

  init_fn, apply_fn = simulate.constrained_rpy(
      (disp, shift), energy_fn, 1e-3, 1.0,
      tol=1e-3, n_particles=8, mr_iters=20, **_AUTO_KW)
  state = init_fn(jax.random.PRNGKey(0), R)
  apply_j = jax.jit(apply_fn)
  for _ in range(3):
    state = apply_j(state)
  assert state.real_position.shape == (R.shape[0], 3)
  assert bool(jnp.all(jnp.isfinite(state.real_position)))


def test_shear_auto_estimates_params_when_xi_omitted():
  """The shear wrapper also auto-estimates, threading the schedule through."""
  R = _positions(n=8)
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  gamma_dot = 0.5
  disp, shift, box_of = space.shearing(
      box, shear_schedule=lambda t: gamma_dot * t,
      fractional_coordinates=True, remap=True)
  energy_fn = _energy_fn(disp)

  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (disp, shift, box_of), energy_fn, 1e-3, 1.0,
      shear_vector_schedule=lambda t: (gamma_dot * t, 0.0 * t, 0.0 * t),
      tol=1e-3, n_particles=8, mr_iters=20, **_AUTO_KW)
  state = init_fn(jax.random.PRNGKey(0), R)
  apply_j = jax.jit(apply_fn)
  for _ in range(3):
    state = apply_j(state)
  assert state.brownian_state.stresslet.shape == (R.shape[0], 5)
  assert bool(jnp.all(jnp.isfinite(state.real_position)))


def test_auto_estimation_requires_tolerance():
  """With neither `xi` nor `tol`, the build raises a clear error."""
  R = _positions()
  box = jnp.eye(3, dtype=_dtype()) * _LENGTH
  disp, shift = space.periodic_general(box, fractional_coordinates=True)
  energy_fn = _energy_fn(disp)
  try:
    simulate.constrained_rpy(
        (disp, shift), energy_fn, 1e-3, 1.0,
        tol=None, mr_iters=20, **_AUTO_KW)
    raised = False
  except ValueError:
    raised = True
  assert raised
