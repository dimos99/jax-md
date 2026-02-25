"""Runtime wrappers and health checks for RPY shear simulations."""

import jax.numpy as jnp
import numpy as np

from rpy_console import get_console

_CONSOLE = get_console()


def _wrap_neighbor_energy(energy_neighbor_fn, energy_all_pairs_fn=None):
  """Wraps neighbor-list energy with optional all-pairs fallback."""
  def _wrapped(R, interaction_neighbor=None, **kwargs):
    if interaction_neighbor is None:
      if energy_all_pairs_fn is not None:
        return energy_all_pairs_fn(R, **kwargs)
      raise ValueError('Missing required interaction_neighbor kwarg for pair-interaction force evaluation.')
    return energy_neighbor_fn(R, neighbor=interaction_neighbor, **kwargs)
  return _wrapped


def _neighbor_list_health(neighbors, stage: str, label: str, console=None) -> bool:
  """Returns True when a neighbor list is present and free of health flags."""
  log = _CONSOLE if console is None else console
  if neighbors is None:
    log.error(f'Missing {label} in stage={stage}.')
    return False
  overflow = np.asarray(neighbors.did_buffer_overflow)
  cell_small = np.asarray(neighbors.cell_size_too_small)
  malformed = np.asarray(neighbors.malformed_box)
  if np.any(overflow):
    log.error(
      f'{label} overflow in stage={stage}. '
      'Try increasing capacity_multiplier or dr_threshold.'
    )
    return False
  if np.any(cell_small):
    log.error(
      f'{label} cell size too small in stage={stage}.'
    )
    return False
  if np.any(malformed):
    log.error(
      f'{label} malformed box in stage={stage}.'
    )
    return False
  return True


def _check_neighbor_status(state, stage: str, console=None) -> bool:
  """Checks the real-space mobility neighbor list on the simulation state."""
  neighbors = state.rpy_state.real.neighbors
  return _neighbor_list_health(
    neighbors,
    stage=stage,
    label='RPY real-space neighbor list',
    console=console,
  )


def _check_interaction_neighbor_status(
  interaction_neighbor,
  stage: str,
  console=None,
) -> bool:
  """Checks the pair-interaction neighbor list used for pair forces/stress."""
  return _neighbor_list_health(
    interaction_neighbor,
    stage=stage,
    label='pair-interaction neighbor list',
    console=console,
  )


def _check_nan_positions(state, stage: str, console=None) -> bool:
  """Returns True if mobility positions are finite; logs and returns False otherwise."""
  log = _CONSOLE if console is None else console
  has_nan = bool(np.asarray(jnp.any(jnp.isnan(state.mobility_position))))
  if has_nan:
    log.error(f'NaN detected in mobility positions during {stage}.')
    return False
  return True
