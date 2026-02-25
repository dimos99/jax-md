"""Timebase and shear-remap helper utilities for RPY shear runs."""

import jax.numpy as jnp
from jax import lax


def _time_from_step(step, *, dt: float, t0: float, dtype):
  """Returns simulation time `t0 + step * dt` in the requested dtype."""
  step_arr = jnp.asarray(step, dtype=jnp.int32)
  dt_arr = jnp.asarray(dt, dtype=dtype)
  t0_arr = jnp.asarray(t0, dtype=dtype)
  return t0_arr + step_arr.astype(dtype) * dt_arr


def _state_step(state, *, dt: float, t0: float):
  """Returns an integer step, deriving it from time when needed."""
  state_step = getattr(state, 'step', None)
  if state_step is not None:
    return jnp.asarray(state_step, dtype=jnp.int32)
  time_arr = jnp.asarray(state.time)
  dt_arr = jnp.asarray(dt, dtype=time_arr.dtype)
  t0_arr = jnp.asarray(t0, dtype=time_arr.dtype)
  step_float = (time_arr - t0_arr) / dt_arr
  return jnp.asarray(jnp.floor(step_float + 0.5), dtype=jnp.int32)


def _state_time_from_step(state, *, dt: float, t0: float):
  """Returns current simulation time, preferring integer step when present."""
  state_step = getattr(state, 'step', None)
  state_time = getattr(state, 'time', None)
  if state_step is not None:
    if state_time is not None:
      time_dtype = jnp.asarray(state_time).dtype
    else:
      time_dtype = state.mobility_position.dtype
    return _time_from_step(state_step, dt=dt, t0=t0, dtype=time_dtype)
  if state_time is None:
    raise AttributeError('State must provide either step or time.')
  return jnp.asarray(state_time)


def _state_next_time_from_step(state, *, dt: float, t0: float):
  """Returns the next-step simulation time from step or time state fields."""
  state_step = getattr(state, 'step', None)
  state_time = getattr(state, 'time', None)
  if state_step is not None:
    next_step = jnp.asarray(state_step, dtype=jnp.int32) + jnp.int32(1)
    if state_time is not None:
      time_dtype = jnp.asarray(state_time).dtype
    else:
      time_dtype = state.mobility_position.dtype
    return _time_from_step(next_step, dt=dt, t0=t0, dtype=time_dtype)
  if state_time is None:
    raise AttributeError('State must provide either step or time.')
  time_arr = jnp.asarray(state_time)
  dt_arr = jnp.asarray(dt, dtype=time_arr.dtype)
  return time_arr + dt_arr


def _predict_xy_remapped_positions_for_next_force(
  state,
  dt: float,
  shear_rate: float,
  t0: float,
):
  """Predicts positions after any discrete xy shear-remap wrap at next force eval."""
  gamma_prev = shear_rate * _state_time_from_step(state, dt=dt, t0=t0)
  gamma_next = shear_rate * _state_next_time_from_step(state, dt=dt, t0=t0)
  m_prev = jnp.floor(gamma_prev + 0.5)
  m_next = jnp.floor(gamma_next + 0.5)
  dm = (m_next - m_prev).astype(jnp.int32)

  def _apply(R):
    dm_cast = jnp.asarray(dm, dtype=R.dtype)
    x_new = jnp.mod(R[:, 0] + dm_cast * R[:, 1], 1.0)
    return R.at[:, 0].set(x_new)

  return lax.cond(jnp.not_equal(dm, 0), _apply, lambda R: R, state.mobility_position)
