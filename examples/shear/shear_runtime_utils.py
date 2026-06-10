"""Runtime wrappers and health checks for shear simulations."""

import jax.numpy as jnp
import numpy as np

from jax_md import dataclasses
from jax_md import space

from shear_console import get_console

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


def _allocate_probe_sized_neighbor(
  neighbor_fn,
  position,
  *,
  box,
  build_box,
  extra_capacity: int = 0,
):
  """Allocates actual neighbors while sizing cell capacity from a build-box probe.

  The returned neighbor list keeps the real `position` / `box` reference state
  and actual neighbor pairs at `t0`. When `build_box` is present, we also
  estimate the worst-tilt cell-list capacity from positions remapped into that
  box and transplant only the larger `cell_list_capacity`.
  """
  neighbor = neighbor_fn.allocate(
    position,
    extra_capacity=extra_capacity,
    box=box,
    build_box=build_box,
  )
  if build_box is None or neighbor.cell_list_capacity is None:
    return neighbor

  probe_position = space.remap_fractional_positions(position, box, build_box)
  probe_neighbor = neighbor_fn.allocate(
    probe_position,
    extra_capacity=extra_capacity,
    box=box,
    build_box=build_box,
  )
  probe_capacity = probe_neighbor.cell_list_capacity
  if probe_capacity is None:
    return neighbor

  base_capacity = int(neighbor.cell_list_capacity)
  probe_capacity = int(probe_capacity)
  if probe_capacity <= base_capacity:
    return neighbor
  return dataclasses.replace(neighbor, cell_list_capacity=probe_capacity)


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
  """Returns True if integrator positions are finite; logs and returns False otherwise."""
  log = _CONSOLE if console is None else console
  has_nan = bool(np.asarray(jnp.any(jnp.isnan(state.integrator_position))))
  if has_nan:
    log.error(f'NaN detected in integrator positions during {stage}.')
    return False
  return True
