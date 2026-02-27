"""Shared utility helpers for shear example scripts."""

import argparse
from decimal import Decimal
from decimal import InvalidOperation

import jax.numpy as jnp
from jax import lax


def parse_int_like(value: str) -> int:
  """Parses integer-like CLI values, including scientific notation."""
  s = str(value).strip()
  try:
    return int(s, 10)
  except ValueError:
    pass
  try:
    d = Decimal(s)
  except InvalidOperation as err:
    raise argparse.ArgumentTypeError(
      f'Expected integer value, got {value!r}.'
    ) from err
  if not d.is_finite():
    raise argparse.ArgumentTypeError(
      f'Expected finite integer value, got {value!r}.'
    )
  integral = d.to_integral_value()
  if d != integral:
    raise argparse.ArgumentTypeError(
      f'Expected integer value, got {value!r}.'
    )
  return int(integral)


def wrap_neighbor_energy(
  energy_neighbor_fn,
  *,
  energy_all_pairs_fn=None,
  missing_neighbor_error='Missing required neighbor list for pair-interaction force evaluation.',
):
  """Wraps neighbor-list energy with optional all-pairs fallback."""
  def _wrapped(R, interaction_neighbor=None, **kwargs):
    if interaction_neighbor is None:
      if energy_all_pairs_fn is not None:
        return energy_all_pairs_fn(R, **kwargs)
      raise ValueError(missing_neighbor_error)
    return energy_neighbor_fn(R, neighbor=interaction_neighbor, **kwargs)
  return _wrapped


def predict_xy_remapped_positions_for_next_force(
  positions_fractional,
  *,
  gamma_prev,
  gamma_next,
):
  """Predicts positions after any discrete xy shear-remap wrap."""
  m_prev = jnp.floor(jnp.asarray(gamma_prev) + 0.5)
  m_next = jnp.floor(jnp.asarray(gamma_next) + 0.5)
  dm = (m_next - m_prev).astype(jnp.int32)

  def _apply(R):
    dm_cast = jnp.asarray(dm, dtype=R.dtype)
    x_new = jnp.mod(R[:, 0] + dm_cast * R[:, 1], 1.0)
    return R.at[:, 0].set(x_new)

  return lax.cond(jnp.not_equal(dm, 0), _apply, lambda R: R, positions_fractional)
