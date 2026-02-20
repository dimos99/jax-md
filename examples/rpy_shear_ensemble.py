"""
RPY shear ensemble runner with pluggable pair-interaction potentials.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from typing import Any
from typing import Callable
from typing import Dict

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import numpy as np

from jax_md import energy
from jax_md import minimize
from jax_md import partition
from jax_md import rheo
from jax_md import simulate
from jax_md import smap
from jax_md import space
from jax_md.hydro import rpy


def _box_size_from_phi(n_particles: int, radius: float, phi: float, dim: int = 3) -> float:
  """Computes cubic/square box size from target packing fraction."""
  if dim == 2:
    area = n_particles * np.pi * radius ** 2 / phi
    return float(np.sqrt(area))
  if dim == 3:
    volume = n_particles * (4.0 / 3.0) * np.pi * radius ** 3 / phi
    return float(volume ** (1.0 / 3.0))
  raise ValueError(f'Unsupported dim={dim}; RPY currently expects 3D.')


def _build_unwrapped_xy_box_fn(base_box: np.ndarray, shear_rate: float):
  """Returns H(t) with unwrapped gamma(t)=shear_rate*t in xy for dump output."""
  base_arr = np.asarray(base_box, dtype=float)
  if base_arr.ndim == 1:
    base_arr = np.diag(base_arr)

  def _box_fn(t: float = 0.0):
    gamma_xy = float(shear_rate) * float(t)
    h = np.array(base_arr, copy=True)
    h[0, 1] = base_arr[0, 1] + gamma_xy * base_arr[1, 1]
    return h

  return _box_fn


def _build_reduced_xy_box_fn(base_box: np.ndarray, shear_rate: float):
  """Returns H(t) with remapped gamma in [-0.5, 0.5) for dump box output."""
  base_arr = np.asarray(base_box, dtype=float)
  if base_arr.ndim == 1:
    base_arr = np.diag(base_arr)

  def _box_fn(t: float = 0.0):
    gamma_xy = float(shear_rate) * float(t)
    gamma_xy = gamma_xy - math.floor(gamma_xy + 0.5)
    h = np.array(base_arr, copy=True)
    h[0, 1] = base_arr[0, 1] + gamma_xy * base_arr[1, 1]
    return h

  return _box_fn


def inverse_power(
  dr: jnp.ndarray,
  epsilon: float,
  sigma: float,
  exponent: float,
  r_cut: float = None,
  r_min: float = 1e-3,
  **unused_kwargs,
) -> jnp.ndarray:
  """Steep inverse-power repulsion with optional cutoff."""
  r = jnp.maximum(dr, r_min)
  log_term = exponent * jnp.log(sigma / r)
  max_log = 80.0 if dr.dtype == jnp.float32 else 700.0
  val = epsilon * jnp.exp(jnp.minimum(log_term, max_log))
  if r_cut is not None:
    return jnp.where(r < r_cut, val, 0.0)
  return val


def _validate_potential_cutoff(r_cut: Any, context: str) -> float:
  try:
    r_cut_val = float(r_cut)
  except (TypeError, ValueError) as err:
    raise ValueError(f'{context} requires numeric r_cut; got {r_cut!r}.') from err
  if (not math.isfinite(r_cut_val)) or r_cut_val <= 0.0:
    raise ValueError(f'{context} requires finite r_cut > 0, got {r_cut_val}.')
  return r_cut_val


def _normalize_interaction_neighbor_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
  required_keys = {'format', 'dr_threshold', 'capacity_multiplier'}
  if not isinstance(raw, dict):
    raise ValueError('POTENTIAL_NEIGHBOR_PARAMS must be a dict.')
  unknown = sorted(set(raw.keys()) - required_keys)
  missing = sorted(required_keys - set(raw.keys()))
  if unknown:
    raise ValueError(
      f'POTENTIAL_NEIGHBOR_PARAMS contains unsupported key(s): {unknown}. '
      f'Supported keys: {sorted(required_keys)}.'
    )
  if missing:
    raise ValueError(
      f'POTENTIAL_NEIGHBOR_PARAMS is missing required key(s): {missing}.'
    )
  out = {
    'format': str(raw['format']).lower(),
    'dr_threshold': float(raw['dr_threshold']),
    'capacity_multiplier': float(raw['capacity_multiplier']),
  }
  if out['dr_threshold'] < 0.0:
    raise ValueError(
      'Potential POTENTIAL_NEIGHBOR_PARAMS.dr_threshold must be >= 0.'
    )
  if out['capacity_multiplier'] <= 0.0:
    raise ValueError(
      'Potential POTENTIAL_NEIGHBOR_PARAMS.capacity_multiplier must be > 0.'
    )
  return out


def _load_module_from_spec(path_or_module: str):
  if os.path.isfile(path_or_module):
    module_path = os.path.abspath(path_or_module)
    digest = hashlib.sha1(module_path.encode('utf-8')).hexdigest()[:12]
    module_name = f'rpy_shear_potential_{digest}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
      raise ValueError(f'Failed to load custom potential module from path: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path_or_module
  module = importlib.import_module(path_or_module)
  return module, path_or_module


def _validate_pair_fn(fn: Callable[..., Any], context: str):
  if not callable(fn):
    raise ValueError(f'{context} pair function is not callable.')
  sig = inspect.signature(fn)
  if len(sig.parameters) < 1:
    raise ValueError(f'{context} pair function must accept at least one positional argument (dr).')


def _parse_last_complete_dump_frame(path: str, dim: int):
  """Returns the last complete frame from a LAMMPS text dump."""
  if dim not in (2, 3):
    raise ValueError(f'Unsupported dim={dim} for dump parsing.')
  if not os.path.isfile(path):
    raise ValueError(f'--init-traj file does not exist: {path}')

  last = None
  n_complete = 0
  truncated_tail = False

  with open(path, 'r') as handle:
    while True:
      line = handle.readline()
      if not line:
        break
      if not line.startswith('ITEM: TIMESTEP'):
        continue

      timestep_line = handle.readline()
      if not timestep_line:
        truncated_tail = True
        break
      timestep = int(float(timestep_line.strip()))

      n_header = handle.readline()
      if not n_header.startswith('ITEM: NUMBER OF ATOMS'):
        raise ValueError(f'{path}: missing "ITEM: NUMBER OF ATOMS".')
      n_line = handle.readline()
      if not n_line:
        truncated_tail = True
        break
      n_particles = int(float(n_line.strip()))

      bounds_header = handle.readline()
      if not bounds_header.startswith('ITEM: BOX BOUNDS'):
        raise ValueError(f'{path}: missing "ITEM: BOX BOUNDS".')
      b1 = handle.readline()
      b2 = handle.readline()
      b3 = handle.readline()
      if not (b1 and b2 and b3):
        truncated_tail = True
        break
      box_matrix = _parse_box_matrix_from_bounds(bounds_header, b1, b2, b3)

      atoms_header = handle.readline()
      if not atoms_header.startswith('ITEM: ATOMS'):
        raise ValueError(f'{path}: missing "ITEM: ATOMS".')
      columns = atoms_header.strip().split()[2:]
      required = ['id', 'x', 'y'] if dim == 2 else ['id', 'x', 'y', 'z']
      if not all(col in columns for col in required):
        raise ValueError(
          f'{path}: ATOMS header must include {required}, got {columns}.'
        )

      i_id = columns.index('id')
      i_x = columns.index('x')
      i_y = columns.index('y')
      i_z = columns.index('z') if dim == 3 else None
      needed = max([i_id, i_x, i_y] + ([i_z] if i_z is not None else []))

      ids = np.empty((n_particles,), dtype=np.int64)
      pos = np.empty((n_particles, dim), dtype=float)
      malformed = False
      for k in range(n_particles):
        row = handle.readline()
        if not row:
          malformed = True
          break
        parts = row.split()
        if len(parts) <= needed:
          malformed = True
          break
        ids[k] = int(parts[i_id])
        pos[k, 0] = float(parts[i_x])
        pos[k, 1] = float(parts[i_y])
        if dim == 3:
          pos[k, 2] = float(parts[i_z])

      if malformed:
        truncated_tail = True
        break

      order = np.argsort(ids)
      ids = ids[order]
      pos = pos[order]
      last = {
        'timestep': timestep,
        'n_particles': n_particles,
        'ids': ids,
        'positions_real': pos,
        'box_matrix': box_matrix,
      }
      n_complete += 1

  if last is None:
    raise ValueError(f'No complete frames were found in dump: {path}')
  return last, n_complete, truncated_tail


def _parse_box_matrix_from_bounds(bounds_header: str, b1: str, b2: str, b3: str) -> np.ndarray:
  """Parses LAMMPS BOX BOUNDS (+tilts) into an upper-triangular box matrix."""
  tokens = bounds_header.strip().split()
  triclinic = ('xy' in tokens) or ('xz' in tokens) or ('yz' in tokens)
  p1 = [float(x) for x in b1.split()]
  p2 = [float(x) for x in b2.split()]
  p3 = [float(x) for x in b3.split()]
  if not triclinic:
    xlo, xhi = p1[:2]
    ylo, yhi = p2[:2]
    zlo, zhi = p3[:2]
    return np.array(
      [[xhi - xlo, 0.0, 0.0], [0.0, yhi - ylo, 0.0], [0.0, 0.0, zhi - zlo]],
      dtype=float,
    )

  if len(p1) < 3 or len(p2) < 3 or len(p3) < 3:
    raise ValueError('Triclinic BOX BOUNDS requires x/y/z tilt factors.')
  xlo_b, xhi_b, xy = p1[:3]
  ylo_b, yhi_b, xz = p2[:3]
  zlo_b, zhi_b, yz = p3[:3]
  x_correction = max(0.0, xy, xz, xy + xz) - min(0.0, xy, xz, xy + xz)
  y_correction = max(0.0, yz) - min(0.0, yz)
  lx = (xhi_b - xlo_b) - x_correction
  ly = (yhi_b - ylo_b) - y_correction
  lz = zhi_b - zlo_b
  return np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]], dtype=float)


def _load_initial_state_from_dump(path: str, *, dim: int, radius: float):
  """Loads initial positions/box from dump and derives N and phi."""
  frame, n_complete, truncated_tail = _parse_last_complete_dump_frame(path, dim)
  n_particles = int(frame['n_particles'])
  box_matrix = np.asarray(frame['box_matrix'], dtype=float)
  pos_real = np.asarray(frame['positions_real'], dtype=float)
  if box_matrix.shape != (dim, dim):
    raise ValueError(
      f'Unexpected dump box shape {box_matrix.shape}; expected {(dim, dim)}.'
    )
  volume = float(abs(np.linalg.det(box_matrix)))
  if volume <= 0.0 or (not math.isfinite(volume)):
    raise ValueError(f'Invalid dump box determinant: {volume}.')
  phi = n_particles * (4.0 / 3.0) * np.pi * (float(radius) ** 3) / volume
  frac = np.mod(pos_real @ np.linalg.inv(box_matrix).T, 1.0)
  return {
    'n_particles': n_particles,
    'phi': float(phi),
    'box_matrix': box_matrix,
    'positions_fractional': frac,
    'source_timestep': int(frame['timestep']),
    'n_complete_frames': int(n_complete),
    'truncated_tail': bool(truncated_tail),
  }


def _resolve_potential(potential_path: str) -> Dict[str, Any]:
  module, source = _load_module_from_spec(potential_path)
  pair_fn = getattr(module, 'pair_potential', None)
  _validate_pair_fn(pair_fn, 'potential module')

  default_params = getattr(module, 'POTENTIAL_PARAMS', None)
  if not isinstance(default_params, dict):
    raise ValueError('potential module must define POTENTIAL_PARAMS as a dict.')
  if 'r_cut' not in default_params:
    raise ValueError('potential module POTENTIAL_PARAMS must include finite r_cut > 0.')
  r_cut = _validate_potential_cutoff(default_params['r_cut'], 'potential POTENTIAL_PARAMS')

  neighbor_defaults_raw = getattr(module, 'POTENTIAL_NEIGHBOR_PARAMS', None)
  if neighbor_defaults_raw is None:
    raise ValueError(
      'potential module must define POTENTIAL_NEIGHBOR_PARAMS.'
    )
  neighbor_defaults = _normalize_interaction_neighbor_settings(neighbor_defaults_raw)

  potential_name = getattr(module, 'POTENTIAL_NAME', 'custom')
  return {
    'name': str(potential_name),
    'source': source,
    'pair_fn': pair_fn,
    'params': dict(default_params),
    'r_cut': r_cut,
    'neighbor_defaults': neighbor_defaults,
  }


def _write_potential_template(path: str):
  template = '''"""Custom pair potential template for examples/rpy_shear_ensemble.py."""

import jax.numpy as jnp

POTENTIAL_NAME = "ao_wca_example"
"""Functional form:
  V(r) = V_WCA(r) + V_AO(r)

WCA (Weeks-Chandler-Andersen; LJ truncated at the minimum and shifted to 0 at cutoff):
  r_wca_cut = 2**(1/6) * sigma
  V_WCA(r) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6) - V_LJ(r_wca_cut)   for r < r_wca_cut
           = 0                                                          otherwise

AO attraction (cubic depletion-like shape, cutoff at alpha):
  alpha = ao_diameter * (1 + ao_attr_range)

  shape(r) = -((r**3)/3 - alpha**2*r + 2*alpha**3/3) / denom
  denom    = 2*alpha**3 - 3*alpha**2*ao_diameter + ao_diameter**3

  shape(alpha) = 0 and shape(ao_diameter) = -1/3.
  To avoid a potential jump at r = ao_diameter, V_AO is extended as a constant for r <= ao_diameter:
    V_AO(r) = ao_depth * (-1/3)              for r <= ao_diameter
            = ao_depth * shape(r)            for ao_diameter < r < alpha
            = 0                              for r >= alpha

Args:
  path: Absolute path where the template module will be written.
"""

# Required: provide finite r_cut > 0 so the runner can build a neighbor list.
# Keep:
#   r_cut >= max(2**(1/6)*wca_sigma_contact, ao_diameter*(1+ao_attr_range)).
POTENTIAL_PARAMS = {
  "wca_epsilon": 8000.0,        # kT
  "wca_sigma_contact": 1.8,     # length
  "ao_depth": 20.0,           # kT (multiplies shape)
  "ao_attr_range": 0.1,         # dimensionless, relative to ao_diameter
  "ao_diameter": 2.0,           # length
  "r_cut": 2.25,                # length; overwritten by max(WCA cutoff, AO cutoff) if too small
  "r_min": 1e-6,                # length; avoids r=0 blowups
}

# Required: define interaction-neighbor defaults for this potential.
POTENTIAL_NEIGHBOR_PARAMS = {
  "format": "sparse",  # one of: dense, sparse, ordered
  "dr_threshold": 1.0,
  "capacity_multiplier": 2.5,
}


def pair_potential(
  dr,
  wca_epsilon,
  wca_sigma_contact,
  ao_depth,
  ao_attr_range,
  ao_diameter,
  r_cut,
  r_min=1e-6,
  **unused_kwargs,
):
  """AO + WCA example from the colloid-gel setup."""
  # WCA repulsion (LJ shifted so V(r_wca_cut)=0).
  sigma_eff = wca_sigma_contact
  wca_cut = sigma_eff * (2.0 ** (1.0 / 6.0))
  safe_r = jnp.maximum(dr, r_min)

  r_over_sigma = safe_r / sigma_eff
  r_cut_over_sigma = wca_cut / sigma_eff

  lj_energy = 4.0 * wca_epsilon * ((1.0 / r_over_sigma) ** 12 - (1.0 / r_over_sigma) ** 6)
  lj_cut = 4.0 * wca_epsilon * ((1.0 / r_cut_over_sigma) ** 12 - (1.0 / r_cut_over_sigma) ** 6)
  wca = jnp.where((dr < wca_cut) & (dr > 0.0), lj_energy - lj_cut, 0.0)

  # AO attraction (Option A: continuous V at r=ao_diameter by extending contact value).
  alpha = ao_diameter * (1.0 + ao_attr_range)
  denom = 2.0 * alpha**3 - 3.0 * alpha**2 * ao_diameter + ao_diameter**3
  ao_shape = -((dr**3) / 3.0 - alpha**2 * dr + 2.0 * alpha**3 / 3.0) / denom

  ao_shape_contact = -1.0 / 3.0  # ao_shape(dr=ao_diameter) for this normalization
  ao = jnp.where(
      dr <= ao_diameter,
      ao_depth * ao_shape_contact,
      jnp.where(dr < alpha, ao_depth * ao_shape, 0.0),
  )

  return wca + ao
'''
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(path, 'w') as handle:
    handle.write(template)


def _wrap_neighbor_energy(energy_neighbor_fn, energy_all_pairs_fn=None):
  def _wrapped(R, interaction_neighbor=None, **kwargs):
    if interaction_neighbor is None:
      if energy_all_pairs_fn is not None:
        return energy_all_pairs_fn(R, **kwargs)
      raise ValueError('Missing required interaction_neighbor kwarg for pair-interaction force evaluation.')
    return energy_neighbor_fn(R, neighbor=interaction_neighbor, **kwargs)
  return _wrapped


def _to_jsonable(x):
  if isinstance(x, (str, int, float, bool)) or x is None:
    return x
  if isinstance(x, (np.floating, np.integer)):
    return x.item()
  if isinstance(x, (np.ndarray, jnp.ndarray)):
    return np.asarray(x).tolist()
  if isinstance(x, dict):
    return {str(k): _to_jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_to_jsonable(v) for v in x]
  return str(x)


def _neighbor_list_health(neighbors, stage: str, run_offset: int, label: str) -> bool:
  if neighbors is None:
    print(f'Error: missing {label} in stage={stage}.')
    return False
  overflow = np.asarray(neighbors.did_buffer_overflow)
  cell_small = np.asarray(neighbors.cell_size_too_small)
  malformed = np.asarray(neighbors.malformed_box)
  if np.any(overflow):
    idx = _first_true_index(overflow)
    run_id = (run_offset + idx) if idx is not None else 'unknown'
    print(
      f'Error: {label} overflow in stage={stage}, run={run_id}. '
      'Try increasing capacity_multiplier or dr_threshold.'
    )
    return False
  if np.any(cell_small):
    idx = _first_true_index(cell_small)
    run_id = (run_offset + idx) if idx is not None else 'unknown'
    print(
      f'Error: {label} cell size too small in stage={stage}, run={run_id}.'
    )
    return False
  if np.any(malformed):
    idx = _first_true_index(malformed)
    run_id = (run_offset + idx) if idx is not None else 'unknown'
    print(
      f'Error: {label} malformed box in stage={stage}, run={run_id}.'
    )
    return False
  return True


def _pad_neighbor_list_capacity(
  nbr: partition.NeighborList,
  target_capacity: int,
) -> partition.NeighborList:
  if nbr is None:
    return nbr
  current_capacity = int(nbr.max_occupancy)
  if current_capacity >= target_capacity:
    return nbr
  n_particles = int(nbr.reference_position.shape[0])
  pad_width = target_capacity - current_capacity
  fill_value = n_particles
  pad_cfg = [(0, 0)] * nbr.idx.ndim
  pad_cfg[-1] = (0, pad_width)
  new_idx = jnp.pad(
    nbr.idx,
    tuple(pad_cfg),
    mode='constant',
    constant_values=fill_value,
  )
  return nbr.set(idx=new_idx, max_occupancy=target_capacity)


def _align_neighbor_static_fields(
  nbr: partition.NeighborList,
  canonical: partition.NeighborList,
) -> partition.NeighborList:
  if nbr is None or canonical is None:
    return nbr
  return nbr.set(
    cell_list_capacity=canonical.cell_list_capacity,
    max_occupancy=canonical.max_occupancy,
    format=canonical.format,
    cell_size=canonical.cell_size,
    cell_list_fn=canonical.cell_list_fn,
    update_fn=canonical.update_fn,
  )


def _stack_neighbor_lists(neighbors):
  if not neighbors:
    raise ValueError('Expected at least one neighbor list to stack.')
  occupancies = [int(n.max_occupancy) for n in neighbors]
  target_capacity = max(occupancies)
  template = _pad_neighbor_list_capacity(neighbors[0], target_capacity)
  aligned = []
  for nbr in neighbors:
    padded = _pad_neighbor_list_capacity(nbr, target_capacity)
    aligned.append(_align_neighbor_static_fields(padded, template))
  return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *aligned)


def _predict_xy_remapped_positions_for_next_force(state, dt: float, shear_rate: float):
  gamma_prev = shear_rate * state.time
  gamma_next = shear_rate * (state.time + dt)
  m_prev = jnp.floor(gamma_prev + 0.5)
  m_next = jnp.floor(gamma_next + 0.5)
  dm = (m_next - m_prev).astype(jnp.int32)

  def _apply(R):
    dm_cast = jnp.asarray(dm, dtype=R.dtype)
    x_new = jnp.mod(R[:, 0] + dm_cast * R[:, 1], 1.0)
    return R.at[:, 0].set(x_new)

  return lax.cond(jnp.not_equal(dm, 0), _apply, lambda R: R, state.mobility_position)


def _relax_positions(
  R_init: jnp.ndarray,
  displacement_0: space.DisplacementFn,
  shift_0: space.ShiftFn,
  diameter: float,
  box: jnp.ndarray,
  steps: int,
  neighbor_format: partition.NeighborListFormat,
  neighbor_capacity_multiplier: float,
  dr_threshold: float,
) -> jnp.ndarray:
  """Relaxes initial positions to remove large overlaps before RPY dynamics."""
  if steps <= 0:
    return R_init

  neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
    displacement_0,
    box,
    sigma=diameter,
    epsilon=1.0,
    alpha=2.0,
    dr_threshold=dr_threshold,
    fractional_coordinates=True,
    format=neighbor_format,
    capacity_multiplier=neighbor_capacity_multiplier,
  )
  neighbor = neighbor_fn.allocate(R_init, box=box)

  init_min, apply_min = minimize.fire_descent(energy_fn, shift_0)
  state = init_min(R_init, neighbor=neighbor)

  @jax.jit
  def _run(state_in, neighbor_in):
    def _step(_, carry):
      s, n, overflow = carry
      n = n.update(s.position, box=box)
      overflow = jnp.logical_or(overflow, n.did_buffer_overflow)
      s = apply_min(s, neighbor=n)
      return s, n, overflow
    return lax.fori_loop(0, steps, _step, (state_in, neighbor_in, False))

  state, _, overflow = _run(state, neighbor)
  if bool(np.asarray(overflow)):
    print('Warning: overlap-relaxation neighbor list overflow detected.')
  return state.position


def _min_pair_distance(
  R: jnp.ndarray,
  displacement_fn: space.DisplacementFn,
) -> float:
  """Computes the minimum pair distance (O(N^2), used once for diagnostics)."""
  n = int(R.shape[0])
  if n < 2:
    return 0.0
  i_idx, j_idx = jnp.triu_indices(n, 1)
  dR = jax.vmap(displacement_fn)(R[i_idx], R[j_idx])
  dist = jnp.sqrt(jnp.sum(dR * dR, axis=-1))
  return float(np.asarray(jnp.min(dist)))


def _serialize_rpy_parameter_estimate(estimate) -> dict:
  """Converts an RpyParameterEstimate object into a JSON-safe dict."""
  diagnostics = getattr(estimate, 'diagnostics', None)
  diagnostics_dict = None
  if diagnostics is not None:
    diagnostics_dict = {
      'sigma_min': float(diagnostics.sigma_min),
      'lattice_extent': int(diagnostics.lattice_extent),
      'L_mean': float(diagnostics.L_mean),
      'L_max': float(diagnostics.L_max),
      'eps_r_target': float(diagnostics.eps_r_target),
      'eps_w_target': float(diagnostics.eps_w_target),
      'eps_q_target': float(diagnostics.eps_q_target),
      'eps_r_est': float(diagnostics.eps_r_est),
      'eps_w_est': float(diagnostics.eps_w_est),
      'eps_q_est': float(diagnostics.eps_q_est),
      'quadrature_lambda_max': float(diagnostics.quadrature_lambda_max),
      'quadrature_safety_nodes': int(diagnostics.quadrature_safety_nodes),
      'shear_remap': bool(diagnostics.shear_remap),
      'shear_schedule_provided': bool(diagnostics.shear_schedule_provided),
      'N': int(diagnostics.N),
      'phi': float(diagnostics.phi),
      'd_f': float(diagnostics.d_f),
      'n_iter': int(diagnostics.n_iter),
    }

  return {
    'xi': float(estimate.xi),
    'P': int(estimate.P),
    'M': int(estimate.M),
    'grid_shape': [int(x) for x in estimate.grid_shape],
    'rcut': float(estimate.rcut),
    'kcut': float(estimate.kcut),
    'theta': float(estimate.theta),
    'm': float(estimate.m),
    'lattice_extent': int(estimate.lattice_extent),
    'diagnostics': diagnostics_dict,
  }


def _run_label(run_id: int) -> str:
  return f'{int(run_id):03d}'


def _first_true_index(x: np.ndarray):
  hits = np.where(np.asarray(x).reshape(-1))[0]
  return int(hits[0]) if hits.size else None


def _check_neighbor_status(state, stage: str, run_offset: int = 0) -> bool:
  """Returns True if the neighbor state is healthy; otherwise prints and fails."""
  neighbors = state.rpy_state.real.neighbors
  return _neighbor_list_health(
    neighbors,
    stage=stage,
    run_offset=run_offset,
    label='RPY real-space neighbor list',
  )


def _check_interaction_neighbor_status(
  interaction_neighbor,
  stage: str,
  run_offset: int = 0,
) -> bool:
  return _neighbor_list_health(
    interaction_neighbor,
    stage=stage,
    run_offset=run_offset,
    label='pair-interaction neighbor list',
  )


def _check_nan_positions(state, stage: str) -> bool:
  """Returns True if positions are finite; otherwise prints and fails."""
  has_nan = bool(np.asarray(jnp.any(jnp.isnan(state.mobility_position))))
  if has_nan:
    print(f'Error: NaN detected in mobility positions during {stage}.')
    return False
  return True


def _stack_states(states):
  """Stacks a list of per-run states into one batched pytree state."""
  if not states:
    raise ValueError('Expected at least one state to stack.')
  template = states[0]
  target_neighbor_occupancy = None
  template_neighbor = None

  def _pad_neighbor_list_capacity(
    nbr: partition.NeighborList,
    target_capacity: int,
  ) -> partition.NeighborList:
    if nbr is None:
      return nbr
    current_capacity = int(nbr.max_occupancy)
    if current_capacity == target_capacity:
      return nbr
    if current_capacity > target_capacity:
      # Keep existing allocation if already larger.
      return nbr

    n_particles = int(nbr.reference_position.shape[0])
    pad_width = target_capacity - current_capacity
    fill_value = n_particles

    if partition.is_sparse(nbr.format):
      # Sparse/ordered sparse: idx has shape [2, max_occupancy].
      pad_cfg = ((0, 0), (0, pad_width))
    else:
      # Dense: idx has shape [N, max_occupancy].
      pad_cfg = ((0, 0), (0, pad_width))
    new_idx = jnp.pad(nbr.idx, pad_cfg, mode='constant', constant_values=fill_value)
    return nbr.set(idx=new_idx, max_occupancy=target_capacity)

  if (hasattr(template, 'rpy_state')
      and hasattr(template.rpy_state, 'real')
      and getattr(template.rpy_state.real, 'neighbors', None) is not None):
    occupancies = [int(s.rpy_state.real.neighbors.max_occupancy) for s in states]
    target_neighbor_occupancy = max(occupancies)
    template_neighbor = _pad_neighbor_list_capacity(
      template.rpy_state.real.neighbors, target_neighbor_occupancy)

  def _align_neighbor_static_fields(
    nbr: partition.NeighborList,
    canonical: partition.NeighborList,
  ) -> partition.NeighborList:
    if nbr is None or canonical is None:
      return nbr
    return nbr.set(
      cell_list_capacity=canonical.cell_list_capacity,
      max_occupancy=canonical.max_occupancy,
      format=canonical.format,
      cell_size=canonical.cell_size,
      cell_list_fn=canonical.cell_list_fn,
      update_fn=canonical.update_fn,
    )

  def _align_static_fields(state):
    # Some nested dataclass nodes include static callable fields (e.g., wave
    # apply/sqrt functions). Align them to the template so pytree metadata
    # matches across runs before stacking.
    if not hasattr(state, 'rpy_state'):
      return state

    template_rpy = template.rpy_state
    state_rpy = state.rpy_state

    if hasattr(state_rpy, 'real') and hasattr(state_rpy.real, 'set'):
      state_real = state_rpy.real.set(core_fn=template_rpy.real.core_fn)
      if (target_neighbor_occupancy is not None
          and getattr(state_real, 'neighbors', None) is not None):
        padded = _pad_neighbor_list_capacity(
          state_real.neighbors, target_neighbor_occupancy)
        state_real = state_real.set(
          neighbors=_align_neighbor_static_fields(padded, template_neighbor))
    else:
      state_real = state_rpy.real

    if hasattr(state_rpy, 'wave') and hasattr(state_rpy.wave, 'set'):
      state_wave = state_rpy.wave.set(
        apply_fn=template_rpy.wave.apply_fn,
        sqrt_fn=template_rpy.wave.sqrt_fn,
        fused_fn=template_rpy.wave.fused_fn,
      )
    else:
      state_wave = state_rpy.wave

    state_rpy = state_rpy.set(
      real=state_real,
      wave=state_wave,
      preconditioner=template_rpy.preconditioner,
    )
    return state.set(rpy_state=state_rpy)

  aligned = [_align_static_fields(s) for s in states]
  return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *aligned)


class RunDumper:
  """Streams per-run stress and trajectory data to disk."""

  def __init__(
    self,
    out_dir: str,
    run_id: int,
    box_size: float,
    dim: int,
    dt: float,
    traj_every: int,
    box_fn=None,
    base_box=None,
    shear_rate: float = 0.0,
    shear_remap: bool = True,
    unwrap_trajectory: bool = True,
  ):
    label = _run_label(run_id)
    self.box_size = float(box_size)
    self.dim = int(dim)
    self.dt = float(dt)
    self.traj_every = int(traj_every)
    self.box_fn = box_fn
    self.shear_rate = float(shear_rate)
    self.shear_remap = bool(shear_remap)
    self.unwrap_trajectory = bool(unwrap_trajectory)
    self.prev_frac_unwrapped = None
    self.base_box = None
    if base_box is not None:
      self.base_box = np.asarray(base_box, dtype=float)
      if self.base_box.ndim == 1:
        self.base_box = np.diag(self.base_box)
    else:
      # Default to t=0 box if not explicitly provided.
      self.base_box = self._box_matrix_at_time(0.0)
    self.stress_filename = os.path.join(out_dir, f'stress_{label}.dat')
    self.traj_filename = os.path.join(out_dir, f'traj_{label}.dump')

    self.stress_file = open(self.stress_filename, 'w')
    if self.dim == 3:
      stress_labels = ['sigma_xx', 'sigma_yy', 'sigma_zz', 'sigma_xy', 'sigma_xz', 'sigma_yz']
    elif self.dim == 2:
      stress_labels = ['sigma_xx', 'sigma_yy', 'sigma_xy']
    else:
      axes = ('x', 'y', 'z')[:self.dim]
      stress_labels = [f'sigma_{a}{b}' for a in axes for b in axes]
    self.stress_file.write('# time strain ' + ' '.join(stress_labels) + '\n')

    if self.traj_every > 0:
      self.traj_file = open(self.traj_filename, 'w')
    else:
      self.traj_file = None

  def _box_matrix_at_time(self, t: float) -> np.ndarray:
    if self.box_fn is None:
      return np.eye(self.dim, dtype=float) * self.box_size
    box = np.asarray(self.box_fn(t=t), dtype=float)
    if box.ndim == 0:
      return np.eye(self.dim, dtype=float) * float(box)
    if box.ndim == 1:
      return np.diag(box)
    return box

  def _estimate_shear_wrap_counts(self, t: float):
    """Infer cumulative remap wrap counts from unwrapped strain gamma(t)."""
    if not self.shear_remap:
      return 0, 0, 0
    gamma_xy = self.shear_rate * float(t)
    n_xy = int(np.floor(gamma_xy + 0.5))
    # This simulator only applies xy shear.
    return n_xy, 0, 0

  def _deremap_fractional(self, frac: np.ndarray, t: float) -> np.ndarray:
    """Map remapped fractional coords back to unwrapped-lab fractional coords."""
    out = np.asarray(frac, dtype=float).copy()
    if not self.shear_remap or out.size == 0:
      return out

    n_xy, n_xz, n_yz = self._estimate_shear_wrap_counts(t)
    if self.dim >= 3:
      # Inverse of simulate._apply_fractional_shear_remap (3D):
      #   y' = y + myz*z
      #   x' = x + mxy*y + (mxz + mxy*myz)*z
      # Solve for (x, y) using z unchanged. Note x depends on the *unremapped* y.
      z_r = out[:, 2].copy()
      y_r = out[:, 1].copy()
      y_un = y_r - n_yz * z_r
      x_un = out[:, 0] - n_xy * y_un - (n_xz + n_xy * n_yz) * z_r
      out[:, 1] = y_un
      out[:, 0] = x_un
    elif self.dim == 2:
      y_r = out[:, 1]
      out[:, 0] = out[:, 0] - n_xy * y_r
    return out

  def _unwrapped_box_matrix_at_time(self, t: float) -> np.ndarray:
    """Returns the continuously deformed triclinic box with gamma(t)=shear_rate*t."""
    base = np.asarray(self.base_box, dtype=float)
    box_t = np.array(base, copy=True)
    if self.dim >= 2:
      gamma_xy = self.shear_rate * float(t)
      box_t[0, 1] = float(base[0, 1]) + float(gamma_xy) * float(base[1, 1])
    return box_t

  def _unwrap_fractional_continuously(self, frac: np.ndarray) -> np.ndarray:
    """Unwrap fractional trajectory over time by nearest-image continuity."""
    if not self.unwrap_trajectory:
      return frac
    if self.prev_frac_unwrapped is None:
      self.prev_frac_unwrapped = np.asarray(frac, dtype=float).copy()
      return self.prev_frac_unwrapped

    diff = np.asarray(frac, dtype=float) - self.prev_frac_unwrapped
    diff -= np.round(diff)
    self.prev_frac_unwrapped = self.prev_frac_unwrapped + diff
    return self.prev_frac_unwrapped

  def dump(
    self,
    stress_times,
    stress_strains,
    stress,
    traj_times=None,
    traj_positions=None,
  ):
    def _ordered_stress_components(s):
      s = np.asarray(s)
      if self.dim == 3:
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[0, 2], s[1, 2]])
      if self.dim == 2:
        return np.array([s[0, 0], s[1, 1], s[0, 1]])
      return s.reshape(-1)

    lines = []
    for t, g, s in zip(stress_times, stress_strains, stress):
      comps = _ordered_stress_components(s)
      values = ' '.join(f'{val:.6e}' for val in comps)
      lines.append(f'{t:.6e} {g:.6e} {values}\n')
    if lines:
      self.stress_file.writelines(lines)

    if self.traj_file is None or traj_times is None or traj_positions is None:
      return

    traj_lines = []
    for t, pos_frac in zip(traj_times, traj_positions):
      timestep = int(round(float(t) / self.dt))
      box_t = self._box_matrix_at_time(float(t))
      pos_frac = np.asarray(pos_frac, dtype=float)
      pos_frac = self._deremap_fractional(pos_frac, float(t))
      pos_frac = self._unwrap_fractional_continuously(pos_frac)
      traj_lines.append('ITEM: TIMESTEP\n')
      traj_lines.append(f'{timestep}\n')
      traj_lines.append('ITEM: NUMBER OF ATOMS\n')
      traj_lines.append(f'{len(pos_frac)}\n')
      if self.dim == 3:
        lx = float(box_t[0, 0])
        ly = float(box_t[1, 1])
        lz = float(box_t[2, 2])
        xy = float(box_t[0, 1])
        xz = float(box_t[0, 2])
        yz = float(box_t[1, 2])

        # LAMMPS triclinic bounds format.
        xlo = 0.0
        xhi = lx
        ylo = 0.0
        yhi = ly
        zlo = 0.0
        zhi = lz
        xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
        xhi_bound = xhi + max(0.0, xy, xz, xy + xz)
        ylo_bound = ylo + min(0.0, yz)
        yhi_bound = yhi + max(0.0, yz)
        zlo_bound = zlo
        zhi_bound = zhi

        traj_lines.append('ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
        traj_lines.append(f'{xlo_bound:.6e} {xhi_bound:.6e} {xy:.6e}\n')
        traj_lines.append(f'{ylo_bound:.6e} {yhi_bound:.6e} {xz:.6e}\n')
        traj_lines.append(f'{zlo_bound:.6e} {zhi_bound:.6e} {yz:.6e}\n')
      else:
        traj_lines.append('ITEM: BOX BOUNDS pp pp pp\n')
        traj_lines.append(f'0 {self.box_size:.6e}\n')
        traj_lines.append(f'0 {self.box_size:.6e}\n')
        traj_lines.append('0 1.000000e+00\n')
      traj_lines.append('ITEM: ATOMS id x y z\n')
      unwrapped_box_t = self._unwrapped_box_matrix_at_time(float(t))
      for i, p in enumerate(pos_frac):
        p_real = np.asarray(unwrapped_box_t @ np.asarray(p), dtype=float)
        if self.dim == 2:
          traj_lines.append(f'{i} {p_real[0]:.6e} {p_real[1]:.6e} 0.0\n')
        else:
          traj_lines.append(f'{i} {p_real[0]:.6e} {p_real[1]:.6e} {p_real[2]:.6e}\n')
    if traj_lines:
      self.traj_file.writelines(traj_lines)

  def close(self):
    self.stress_file.close()
    if self.traj_file is not None:
      self.traj_file.close()


def _build_thermalize_runner(apply_fn_eq, steps: int, base_box: jnp.ndarray):
  """Returns a vmapped thermalization runner for exactly `steps` steps."""
  if steps <= 0:
    return None

  def _single(state, interaction_neighbor):
    def _step(_, carry):
      s, pn = carry
      pn = pn.update(s.mobility_position, box=base_box)
      s = apply_fn_eq(s, interaction_neighbor=pn)
      return s, pn
    return lax.fori_loop(0, steps, _step, (state, interaction_neighbor))

  return jax.jit(jax.vmap(_single))


def parse_args():
  parser = argparse.ArgumentParser(
    description='RPY shear ensemble runner with chunked stress/trajectory dumping.')

  # Experiment-facing controls.
  parser.add_argument(
    '--n_particles',
    type=int,
    default=None,
    help='Particle count for random initialization mode (required without --init-traj).',
  )
  parser.add_argument(
    '--phi',
    type=float,
    default=None,
    help='Packing fraction for random initialization mode (required without --init-traj).',
  )
  parser.add_argument('--peclet', type=float, default=0.0)
  parser.add_argument(
    '--dt',
    type=float,
    default=None,
    help='Integration timestep in the current simulation units (required).',
  )
  parser.add_argument('--xi', type=float, default=0.7,
                      help='RPY splitting parameter xi passed as xi_override.')
  parser.add_argument('--n_steps', type=int, default=30000)
  parser.add_argument('--thermalize_steps', type=int, default=0)
  parser.add_argument(
    '--buffer-steps',
    type=int,
    default=1000,
    help='Simulation chunk size in steps before returning to Python/output.',
  )
  parser.add_argument('--n_runs', type=int, default=8)
  parser.add_argument(
    '--runs_per_batch',
    type=int,
    default=None,
    help='How many runs to execute simultaneously; remaining runs execute sequentially.')
  parser.add_argument('--stress_every', type=int, default=10)
  parser.add_argument('--traj_every', type=int, default=100,
                      help='Set to 0 to disable trajectory output.')
  parser.add_argument('--progress_every', type=int, default=1000)
  parser.add_argument(
    '--mr-skin',
    '--mr-dr-threshold',
    dest='mr_skin',
    type=float,
    default=0.5,
    help='Real-space mobility neighbor-list skin (dr_threshold).',
  )
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument(
    '--out_dir',
    '--out',
    dest='out_dir',
    type=str,
    default=None,
    help='Output directory for params/stress/trajectory files (required).',
  )
  parser.add_argument(
    '--init-traj',
    type=str,
    default=None,
    help='Optional LAMMPS dump file; initialize from its last complete frame.',
  )
  parser.add_argument(
    '--potential',
    type=str,
    default=None,
    help='Python module path/name providing pair_potential(dr, **params) and defaults.',
  )
  parser.add_argument(
    '--write-potential-template',
    type=str,
    default=None,
    help='Write a custom-potential module template to the provided path and exit.',
  )

  args = parser.parse_args()
  if args.write_potential_template is not None:
    return args
  if args.potential is None:
    raise ValueError('--potential is required (module path or importable module name).')
  if args.dt is None:
    raise ValueError('--dt is required.')
  if args.out_dir is None:
    raise ValueError('--out_dir is required.')
  if float(args.dt) <= 0.0:
    raise ValueError('dt must be > 0.')
  if float(args.xi) <= 0.0:
    raise ValueError('xi must be > 0.')
  if args.n_runs <= 0:
    raise ValueError('n_runs must be > 0.')
  if args.runs_per_batch is not None and args.runs_per_batch <= 0:
    raise ValueError('runs_per_batch must be > 0 when provided.')
  if args.n_steps <= 0:
    raise ValueError('n_steps must be > 0.')
  if args.peclet < 0.0:
    raise ValueError('peclet must be >= 0.')
  if args.stress_every <= 0:
    raise ValueError('stress_every must be > 0.')
  if args.traj_every < 0:
    raise ValueError('traj_every must be >= 0.')
  if args.thermalize_steps < 0:
    raise ValueError('thermalize_steps must be >= 0.')
  if args.buffer_steps <= 0:
    raise ValueError('buffer_steps must be > 0.')
  if args.progress_every < 0:
    raise ValueError('progress_every must be >= 0.')
  if args.mr_skin < 0.0:
    raise ValueError('mr_skin must be >= 0.')
  if args.init_traj is not None:
    if args.n_particles is not None or args.phi is not None:
      raise ValueError(
        'When --init-traj is provided, do not pass --n_particles or --phi. '
        'These are derived from the dump file.'
      )
    return args
  if args.n_particles is None or args.phi is None:
    raise ValueError(
      'Random initialization mode requires both --n_particles and --phi.'
    )
  if args.n_particles <= 1:
    raise ValueError('n_particles must be > 1.')
  if not (0.0 < float(args.phi) <= 1.0):
    raise ValueError(
      f'--phi must be in (0, 1], got {args.phi}. '
      'Did you mean e.g. "--phi 0.45" (not "--phi 045")?'
    )
  if float(args.phi) > 0.74:
    print(
      f'Warning: --phi={args.phi} exceeds hard-sphere close packing (~0.74). '
      'This can lead to severe overlaps and NaNs in the RPY mobility.'
    )
  return args


def build_internal_config():
  """Algorithmic defaults intentionally kept out of the user-facing CLI."""
  return {
    # Physics + integrator
    'a': 1.0,
    'kT': 1.0,
    'viscosity': 1.0 / (6.0 * np.pi),
    'mr_iters': 10,
    # RPY estimator
    'tol': 1e-4,
    # RPY real-space neighbor list
    'mr_neighbor_format': 'sparse',
    'mr_capacity_multiplier': 2.5,
    'real_space_mode': 'auto',
    # Initial overlap relaxation
    'relax_steps': 250,
    'relax_neighbor_format': 'sparse',
    'relax_neighbor_dr_threshold': 0.2,
    'relax_neighbor_capacity_multiplier': 2.0,
  }


def main():
  args = parse_args()
  if args.write_potential_template is not None:
    _write_potential_template(args.write_potential_template)
    print(f'Wrote potential template to {args.write_potential_template}')
    return

  internal = build_internal_config()
  a = float(internal['a'])
  kT = float(internal['kT'])
  viscosity = float(internal['viscosity'])
  dt = float(args.dt)
  args.dt = dt
  mr_iters = int(internal['mr_iters'])
  tol = float(internal['tol'])
  xi_override = float(args.xi)

  mr_neighbor_format = str(internal['mr_neighbor_format'])
  mr_dr_threshold = float(args.mr_skin)
  mr_capacity_multiplier = float(internal['mr_capacity_multiplier'])
  real_space_mode = str(internal['real_space_mode'])

  relax_steps = int(internal['relax_steps'])
  relax_neighbor_format = str(internal['relax_neighbor_format'])
  relax_neighbor_dr_threshold = float(internal['relax_neighbor_dr_threshold'])
  relax_neighbor_capacity_multiplier = float(internal['relax_neighbor_capacity_multiplier'])

  buffer_steps_default = int(args.buffer_steps)

  if a <= 0.0:
    raise ValueError('internal default a must be > 0.')
  if kT <= 0.0:
    raise ValueError('internal default kT must be > 0.')
  if viscosity <= 0.0:
    raise ValueError('internal default viscosity must be > 0.')
  if mr_iters <= 0:
    raise ValueError('internal default mr_iters must be > 0.')
  if tol <= 0.0:
    raise ValueError('internal default tol must be > 0.')
  if mr_dr_threshold < 0.0:
    raise ValueError('mr_skin must be >= 0.')
  if mr_capacity_multiplier <= 0.0:
    raise ValueError('internal default mr_capacity_multiplier must be > 0.')
  if real_space_mode not in ('auto', 'min_image', 'lattice'):
    raise ValueError("internal default real_space_mode must be one of 'auto', 'min_image', or 'lattice'.")
  if relax_steps < 0:
    raise ValueError('internal default relax_steps must be >= 0.')
  if relax_neighbor_dr_threshold < 0.0:
    raise ValueError('internal default relax_neighbor_dr_threshold must be >= 0.')
  if relax_neighbor_capacity_multiplier <= 0.0:
    raise ValueError('internal default relax_neighbor_capacity_multiplier must be > 0.')
  if buffer_steps_default <= 0:
    raise ValueError('buffer_steps must be > 0.')

  format_map = {
    'dense': partition.NeighborListFormat.Dense,
    'sparse': partition.NeighborListFormat.Sparse,
    'ordered': partition.NeighborListFormat.OrderedSparse,
  }

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  print(f'JAX backend: {jax.default_backend()}')
  print(f'JAX devices: {device_labels}')

  dim = 3
  diameter = 2.0 * a
  init_mode = 'random_relax'
  dump_info = None
  if args.init_traj is not None:
    init_mode = 'dump'
    dump_info = _load_initial_state_from_dump(
      args.init_traj,
      dim=dim,
      radius=a,
    )
    n_particles = int(dump_info['n_particles'])
    phi = float(dump_info['phi'])
    if n_particles <= 1:
      raise ValueError(f'Dump-derived n_particles must be > 1, got {n_particles}.')
    if not math.isfinite(phi) or phi <= 0.0:
      raise ValueError(f'Dump-derived phi must be finite and > 0, got {phi}.')
    if phi > 0.74:
      print(
        f'Warning: dump-derived phi={phi:.6g} exceeds hard-sphere close packing '
        '(~0.74). This can lead to severe overlaps and NaNs in the RPY mobility.'
      )
    base_box = jnp.asarray(dump_info['box_matrix'])
    print(
      f'Loaded initialization from dump {args.init_traj}: '
      f'source_step={dump_info["source_timestep"]}, '
      f'frames={dump_info["n_complete_frames"]}, '
      f'truncated_tail={dump_info["truncated_tail"]}'
    )
  else:
    n_particles = int(args.n_particles)
    phi = float(args.phi)
    box_size = _box_size_from_phi(n_particles, a, phi, dim=dim)
    base_box = jnp.eye(dim) * box_size

  base_box_np = np.asarray(base_box, dtype=float)
  box_volume = float(abs(np.linalg.det(base_box_np)))
  box_size = float(box_volume ** (1.0 / 3.0))

  D0 = kT / (6.0 * math.pi * viscosity * a)
  shear_rate = 2.0 * args.peclet * D0 / (a ** 2)
  shear_schedule = {'xy': lambda t: shear_rate * t}
  shear_vector_schedule = lambda t: (shear_rate * t, 0.0, 0.0)

  print(f'Equivalent box size L = {box_size:.6f}')
  print(f'D0 = {D0:.6e}')
  print(f'Shear rate = {shear_rate:.6e}')
  print(f'Strain per step = {shear_rate * dt:.6e}')

  potential_cfg = _resolve_potential(args.potential)
  interaction_neighbor_defaults = potential_cfg['neighbor_defaults']
  interaction_neighbor_format_name = str(interaction_neighbor_defaults['format']).lower()
  if interaction_neighbor_format_name not in format_map:
    raise ValueError(
      f'Unsupported interaction neighbor format {interaction_neighbor_format_name!r}. '
      f'Expected one of {sorted(format_map.keys())}.'
    )
  interaction_neighbor_format = format_map[interaction_neighbor_format_name]
  interaction_neighbor_dr_threshold = float(interaction_neighbor_defaults['dr_threshold'])
  interaction_neighbor_capacity_multiplier = float(
    interaction_neighbor_defaults['capacity_multiplier'])
  pair_potential_fn = potential_cfg['pair_fn']
  potential_params = dict(potential_cfg['params'])
  potential_r_cut = float(potential_cfg['r_cut'])
  potential_name = str(potential_cfg['name'])
  potential_source = str(potential_cfg['source'])

  print(
    f'Potential module: {args.potential} -> {potential_name} '
    f'(source={potential_source}, r_cut={potential_r_cut:.6f})'
  )
  print(
    'Interaction neighbor defaults: '
    f'format={interaction_neighbor_format_name}, '
    f'dr_threshold={interaction_neighbor_dr_threshold:.3g}, '
    f'capacity_multiplier={interaction_neighbor_capacity_multiplier:.3g}'
  )

  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  displacement_0, shift_0 = space.periodic_general(base_box, fractional_coordinates=True)

  key = random.PRNGKey(args.seed)
  key, init_key, thermalize_key, run_key = random.split(key, 4)
  if init_mode == 'dump':
    R0 = jnp.asarray(
      dump_info['positions_fractional'],
      dtype=base_box.dtype,
    )
  else:
    R0 = random.uniform(init_key, (n_particles, dim), minval=0.0, maxval=1.0)
    R0 = _relax_positions(
      R0,
      displacement_0,
      shift_0,
      diameter,
      base_box,
      relax_steps,
      format_map[relax_neighbor_format],
      relax_neighbor_capacity_multiplier,
      relax_neighbor_dr_threshold,
    )
  min_dist = _min_pair_distance(R0, displacement_0)
  print(f'Post-relax minimum pair distance: {min_dist:.6f}')

  metric_0 = space.canonicalize_displacement_or_metric(displacement_0)
  metric_shear = space.canonicalize_displacement_or_metric(displacement)
  energy_fn_0_all_pairs = smap.pair(
    pair_potential_fn,
    metric_0,
    ignore_unused_parameters=True,
    **potential_params,
  )
  energy_fn_all_pairs = smap.pair(
    pair_potential_fn,
    metric_shear,
    ignore_unused_parameters=True,
    **potential_params,
  )
  energy_fn_0_neighbor = smap.pair_neighbor_list(
    pair_potential_fn,
    metric_0,
    ignore_unused_parameters=True,
    **potential_params,
  )
  energy_fn_neighbor = smap.pair_neighbor_list(
    pair_potential_fn,
    metric_shear,
    ignore_unused_parameters=True,
    **potential_params,
  )
  energy_fn_0 = _wrap_neighbor_energy(
    energy_fn_0_neighbor,
    energy_all_pairs_fn=energy_fn_0_all_pairs,
  )
  energy_fn = _wrap_neighbor_energy(
    energy_fn_neighbor,
    energy_all_pairs_fn=energy_fn_all_pairs,
  )
  interaction_neighbor_fn_0 = partition.neighbor_list(
    metric_0,
    base_box,
    r_cutoff=potential_r_cut,
    dr_threshold=interaction_neighbor_dr_threshold,
    capacity_multiplier=interaction_neighbor_capacity_multiplier,
    fractional_coordinates=True,
    format=interaction_neighbor_format,
  )
  interaction_neighbor_fn = partition.neighbor_list(
    metric_shear,
    base_box,
    r_cutoff=potential_r_cut,
    dr_threshold=interaction_neighbor_dr_threshold,
    capacity_multiplier=interaction_neighbor_capacity_multiplier,
    fractional_coordinates=True,
    format=interaction_neighbor_format,
  )

  shear_t_bounds = (
    0.0,
    float(dt * float(args.thermalize_steps + args.n_steps)),
  )
  rpy_params = rpy.estimate_rpy_params(
    tol=tol,
    A=base_box,
    a=a,
    N=n_particles,
    phi=phi,
    xi_override=xi_override,
    shear_vector_schedule=shear_vector_schedule,
    shear_t_bounds=shear_t_bounds,
    shear_remap=True,
    notes=True,
  )
  xi = float(rpy_params.xi)
  rpy_rcut = float(rpy_params.rcut)
  rpy_P = int(rpy_params.P)
  rpy_M = int(rpy_params.M)
  rpy_theta = float(rpy_params.theta)
  rpy_lattice_extent = int(rpy_params.lattice_extent)
  print(
    'RPY parameters: '
    f'xi={xi:.6f}, rcut={rpy_rcut:.6f}, P={rpy_P}, M={rpy_M}, theta={rpy_theta:.6f}'
  )
  diagnostics = rpy_params.diagnostics
  if diagnostics is not None:
    print(
      'Quadrature deformation bound: '
      f'lambda_max={diagnostics.quadrature_lambda_max:.6f} '
      f'(remap={diagnostics.shear_remap})'
    )

  init_fn, apply_fn = simulate.rpy_with_shear(
    (displacement, shift, box_of),
    energy_fn,
    dt=dt,
    kT=kT,
    a=a,
    xi=xi,
    eta=viscosity,
    shear_vector_schedule=shear_vector_schedule,
    neighbor_format=format_map[mr_neighbor_format],
    dr_threshold=mr_dr_threshold,
    capacity_multiplier=mr_capacity_multiplier,
    real_space_mode=real_space_mode,
    rcut=rpy_rcut,
    P=rpy_P,
    Mgrid=rpy_M,
    theta=rpy_theta,
    lattice_extent=rpy_lattice_extent,
    mr_iters=mr_iters,
  )

  init_fn_eq, apply_fn_eq = simulate.rpy(
    (displacement_0, shift_0),
    energy_fn_0,
    dt=dt,
    kT=kT,
    a=a,
    xi=xi,
    eta=viscosity,
    neighbor_format=format_map[mr_neighbor_format],
    dr_threshold=mr_dr_threshold,
    capacity_multiplier=mr_capacity_multiplier,
    real_space_mode=real_space_mode,
    rcut=rpy_rcut,
    P=rpy_P,
    Mgrid=rpy_M,
    theta=rpy_theta,
    lattice_extent=rpy_lattice_extent,
    mr_iters=mr_iters,
  )

  stress_fn = rheo.make_pairwise_stress_fn(
    pair_potential_fn,
    **potential_params,
  )

  thermalize_keys = random.split(thermalize_key, args.n_runs)
  run_keys = random.split(run_key, args.n_runs)

  runs_per_batch = (
    args.n_runs if args.runs_per_batch is None else min(args.runs_per_batch, args.n_runs)
  )
  n_batches = (args.n_runs + runs_per_batch - 1) // runs_per_batch

  # Chunk schedule: scan at gcd(stress_every, traj_every) for reusable samples.
  scan_interval = (
    math.gcd(args.stress_every, args.traj_every) if args.traj_every > 0 else args.stress_every
  )
  sample_period = (
    math.lcm(args.stress_every, args.traj_every) if args.traj_every > 0 else args.stress_every
  )
  buffer_steps = buffer_steps_default
  if buffer_steps % sample_period != 0:
    buffer_steps = ((buffer_steps // sample_period) + 1) * sample_period
    print(
      f'Adjusted buffer_steps to {buffer_steps} so it is divisible by sample period '
      f'{sample_period}.'
    )
  if args.n_steps % buffer_steps != 0:
    planned_steps = ((args.n_steps // buffer_steps) + 1) * buffer_steps
    print(
      f'Warning: n_steps={args.n_steps} is not a multiple of buffer_steps={buffer_steps}. '
      f'Running {planned_steps} steps.'
    )
  else:
    planned_steps = args.n_steps

  thermalize_chunk_steps = buffer_steps
  if args.thermalize_steps > 0:
    thermalize_chunk_steps = max(1, min(buffer_steps, args.thermalize_steps))

  steps_per_scan = scan_interval
  scans_per_buffer = buffer_steps // steps_per_scan
  stress_stride = args.stress_every // scan_interval
  traj_stride = args.traj_every // scan_interval if args.traj_every > 0 else None

  def _run_chunk_single(carry_in):
    state_in, interaction_neighbor_in = carry_in

    def _scan_body(carry, _):
      state, interaction_neighbor = carry

      def _inner(_, inner_carry):
        s, pn = inner_carry
        next_box = box_of(t=s.time + dt)
        pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
          s, dt=dt, shear_rate=shear_rate)
        pn = pn.update(pos_for_neighbor, box=next_box)
        s = apply_fn(s, interaction_neighbor=pn)
        return s, pn

      state, interaction_neighbor = lax.fori_loop(
        0, steps_per_scan, _inner, (state, interaction_neighbor))
      curr_box = box_of(t=state.time)
      interaction_neighbor = interaction_neighbor.update(state.mobility_position, box=curr_box)
      stress = stress_fn(
        state.mobility_position,
        box=curr_box,
        neighbor=interaction_neighbor,
        fractional_coordinates=True,
      )
      strain = shear_rate * state.time
      return (state, interaction_neighbor), (state.time, strain, stress, state.mobility_position)

    (state_out, interaction_neighbor_out), (times, strains, stresses, positions) = lax.scan(
      _scan_body,
      (state_in, interaction_neighbor_in),
      None,
      length=scans_per_buffer,
    )

    stress_times = times[::stress_stride]
    stress_strains = strains[::stress_stride]
    stress_out = stresses[::stress_stride]

    if traj_stride is None:
      return (state_out, interaction_neighbor_out), (stress_times, stress_strains, stress_out)

    traj_times = times[::traj_stride]
    traj_positions = positions[::traj_stride]
    return (
      (state_out, interaction_neighbor_out),
      (stress_times, stress_strains, stress_out, traj_times, traj_positions),
    )

  run_chunk = jax.jit(jax.vmap(_run_chunk_single))
  thermalize_runner_cache = {}

  def _get_thermalize_runner(step_count: int):
    runner = thermalize_runner_cache.get(step_count, None)
    if runner is None:
      runner = _build_thermalize_runner(apply_fn_eq, step_count, base_box)
      thermalize_runner_cache[step_count] = runner
    return runner

  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)

  params = {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi,
      'n_runs': args.n_runs,
      'runs_per_batch': args.runs_per_batch,
      'dt': dt,
      'n_steps': args.n_steps,
      'thermalize_steps': args.thermalize_steps,
      'buffer_steps': args.buffer_steps,
      'peclet': args.peclet,
      'xi': args.xi,
      'stress_every': args.stress_every,
      'traj_every': args.traj_every,
      'progress_every': args.progress_every,
      'mr_skin': args.mr_skin,
      'seed': args.seed,
      'out_dir': args.out_dir,
      'init_traj': args.init_traj,
      'potential': args.potential,
    },
    'internal': {
      'a': a,
      'kT': kT,
      'viscosity': viscosity,
      'dt': dt,
      'mr_iters': mr_iters,
      'tol': tol,
      'xi_override': xi_override,
      'mr_neighbor_format': mr_neighbor_format,
      'mr_dr_threshold': mr_dr_threshold,
      'mr_capacity_multiplier': mr_capacity_multiplier,
      'relax_steps': relax_steps,
      'relax_neighbor_format': relax_neighbor_format,
      'relax_neighbor_dr_threshold': relax_neighbor_dr_threshold,
      'relax_neighbor_capacity_multiplier': relax_neighbor_capacity_multiplier,
    },
    'derived': {
      'initialization_mode': init_mode,
      'n_runs': args.n_runs,
      'runs_per_batch': runs_per_batch,
      'n_batches': n_batches,
      'dim': dim,
      'box_size': box_size,
      'box_matrix': _to_jsonable(base_box_np),
      'box_volume': box_volume,
      'diameter': diameter,
      'potential_r_cut': potential_r_cut,
      'D0': D0,
      'shear_rate': shear_rate,
      'traj_box_frame': 'reduced_lab_xy',
      'traj_coords_frame': 'unwrapped_lab_continuous',
      'traj_remap_aware': True,
      'dump_source_step': (
        int(dump_info['source_timestep']) if dump_info is not None else None
      ),
      'dump_frames_in_source': (
        int(dump_info['n_complete_frames']) if dump_info is not None else None
      ),
      'dump_truncated_tail': (
        bool(dump_info['truncated_tail']) if dump_info is not None else None
      ),
      'planned_steps': planned_steps,
      'buffer_steps': buffer_steps,
      'thermalize_chunk_steps': thermalize_chunk_steps,
      'rpy_xi': xi,
      'rpy_rcut': rpy_rcut,
      'rpy_P': int(rpy_P),
      'rpy_M': int(rpy_M),
      'rpy_theta': rpy_theta,
      'rpy_lattice_extent': int(rpy_lattice_extent),
      'rpy_estimator': _serialize_rpy_parameter_estimate(rpy_params),
      'Mr_params': {
        'neighbor_format': mr_neighbor_format,
        'dr_threshold': mr_dr_threshold,
        'capacity_multiplier': mr_capacity_multiplier,
        'real_space_mode': real_space_mode,
      },
    },
    'potential': {
      'selected': args.potential,
      'resolved_name': potential_name,
      'source': potential_source,
      'r_cut': potential_r_cut,
      'params': _to_jsonable(potential_params),
      'neighbor_defaults': _to_jsonable(interaction_neighbor_defaults),
    },
  }
  params_path = os.path.join(out_dir, 'params.json')
  with open(params_path, 'w') as handle:
    json.dump(params, handle, indent=2, sort_keys=True)
  print(f'Wrote parameters to {params_path}')

  dump_box_fn = _build_reduced_xy_box_fn(np.asarray(base_box, dtype=float), shear_rate)
  base_box_np = np.asarray(base_box, dtype=float)
  dumpers = [
    RunDumper(
      out_dir,
      i,
      box_size,
      dim,
      dt,
      args.traj_every,
      box_fn=dump_box_fn,
      base_box=base_box_np,
      shear_rate=shear_rate,
      shear_remap=True,
      unwrap_trajectory=True,
    )
    for i in range(args.n_runs)
  ]

  print(
    f'Running {args.n_runs} sheared trajectories for {planned_steps} steps '
    f'(requested {args.n_steps}) in {n_batches} batch(es) of up to {runs_per_batch}.'
  )
  if args.thermalize_steps > 0:
    print(
      'Thermalization execution chunk: '
      f'{thermalize_chunk_steps} step(s) per JAX call.'
    )
  if args.traj_every > 0:
    print(f'Outputs: stress_XXX.dat + traj_XXX.dump in {out_dir}')
  else:
    print(f'Outputs: stress_XXX.dat in {out_dir}')
  if init_mode == 'dump':
    print(
      f'Dump initialization enabled: starting a fresh run from step 0 '
      f'with positions loaded from {args.init_traj}.'
    )

  target_step = int(args.n_steps)
  try:
    for batch_idx, batch_start in enumerate(range(0, args.n_runs, runs_per_batch), start=1):
      batch_end = min(batch_start + runs_per_batch, args.n_runs)
      batch_ids = list(range(batch_start, batch_end))
      batch_size = len(batch_ids)

      print(
        f'Batch {batch_idx}/{n_batches}: runs '
        f'{batch_start:03d}-{batch_end - 1:03d} (size={batch_size}).'
      )

      state_eq_list = [init_fn_eq(thermalize_keys[i], R0) for i in batch_ids]
      state_eq = _stack_states(state_eq_list)
      interaction_neighbor_eq = _stack_neighbor_lists([
        interaction_neighbor_fn_0.allocate(s.mobility_position, box=base_box)
        for s in state_eq_list
      ])
      if not _check_neighbor_status(state_eq, 'equilibrium_init', run_offset=batch_start):
        return
      if not _check_interaction_neighbor_status(
        interaction_neighbor_eq,
        'equilibrium_init',
        run_offset=batch_start,
      ):
        return

      if args.thermalize_steps > 0:
        therm_done = 0
        next_progress_mark = args.progress_every if args.progress_every > 0 else None
        while therm_done < args.thermalize_steps:
          step_count = min(thermalize_chunk_steps, args.thermalize_steps - therm_done)
          runner = _get_thermalize_runner(step_count)
          state_eq, interaction_neighbor_eq = runner(state_eq, interaction_neighbor_eq)
          therm_done += step_count

          if not _check_nan_positions(state_eq, f'thermalization step {therm_done}'):
            return
          if not _check_neighbor_status(
            state_eq, f'thermalization step {therm_done}', run_offset=batch_start):
            return
          if not _check_interaction_neighbor_status(
            interaction_neighbor_eq,
            f'thermalization step {therm_done}',
            run_offset=batch_start,
          ):
            return
          while (
            next_progress_mark is not None
            and therm_done >= next_progress_mark
            and next_progress_mark <= args.thermalize_steps
          ):
            print(
              f'Batch {batch_idx}/{n_batches} thermalize '
              f'{next_progress_mark}/{args.thermalize_steps}'
            )
            next_progress_mark += args.progress_every
      else:
        print(f'Batch {batch_idx}/{n_batches}: skipping thermalization (thermalize_steps=0).')

      positions_init = state_eq.mobility_position
      state_list = [
        init_fn(run_keys[run_id], positions_init[local_i])
        for local_i, run_id in enumerate(batch_ids)
      ]
      state = _stack_states(state_list)
      box_t0 = box_of(t=0.0)
      interaction_neighbor = _stack_neighbor_lists([
        interaction_neighbor_fn.allocate(s.mobility_position, box=box_t0)
        for s in state_list
      ])
      if not _check_neighbor_status(state, 'shear_init', run_offset=batch_start):
        return
      if not _check_interaction_neighbor_status(
        interaction_neighbor,
        'shear_init',
        run_offset=batch_start,
      ):
        return

      # Write the initial configuration (t=0) before the loop
      # so the trajectory file always starts from the very first frame.
      t0 = float(state_list[0].time)
      for local_i, run_id in enumerate(batch_ids):
        dumper = dumpers[run_id]
        pos0 = np.asarray(positions_init[local_i], dtype=float)
        if traj_stride is not None:
          dumper.dump(
            np.array([], dtype=float),        # no stress at t=0
            np.array([], dtype=float),
            np.zeros((0, dim, dim), dtype=float),
            np.array([t0], dtype=float),
            pos0[np.newaxis],                 # shape (1, N, dim)
          )

      steps_done = 0
      while steps_done < planned_steps:
        if traj_stride is None:
          (state, interaction_neighbor), (stress_times, stress_strains, stresses) = run_chunk(
            (state, interaction_neighbor))
          traj_times = None
          traj_positions = None
        else:
          (state, interaction_neighbor), (
            stress_times,
            stress_strains,
            stresses,
            traj_times,
            traj_positions,
          ) = run_chunk((state, interaction_neighbor))

        if not _check_nan_positions(state, f'shear step {steps_done + buffer_steps}'):
          return
        if not _check_neighbor_status(
          state, f'shear step {steps_done + buffer_steps}', run_offset=batch_start):
          return
        if not _check_interaction_neighbor_status(
          interaction_neighbor,
          f'shear step {steps_done + buffer_steps}',
          run_offset=batch_start,
        ):
          return

        stress_times_np = np.asarray(stress_times)
        stress_strains_np = np.asarray(stress_strains)
        stresses_np = np.asarray(stresses)
        traj_times_np = np.asarray(traj_times) if traj_times is not None else None
        traj_positions_np = np.asarray(traj_positions) if traj_positions is not None else None

        for local_i, run_id in enumerate(batch_ids):
          dumper = dumpers[run_id]
          stress_steps = np.rint(stress_times_np[local_i] / dt).astype(np.int64)
          stress_mask = stress_steps <= target_step
          if traj_times_np is None:
            dumper.dump(
              stress_times_np[local_i][stress_mask],
              stress_strains_np[local_i][stress_mask],
              stresses_np[local_i][stress_mask],
            )
            continue

          traj_steps = np.rint(traj_times_np[local_i] / dt).astype(np.int64)
          traj_mask = traj_steps <= target_step
          dumper.dump(
            stress_times_np[local_i][stress_mask],
            stress_strains_np[local_i][stress_mask],
            stresses_np[local_i][stress_mask],
            traj_times_np[local_i][traj_mask],
            traj_positions_np[local_i][traj_mask],
          )

        steps_done += buffer_steps
        if args.progress_every > 0 and (steps_done % args.progress_every == 0):
          print(
            f'Batch {batch_idx}/{n_batches} step '
            f'{min(steps_done, planned_steps)}/{planned_steps}'
          )
  finally:
    for dumper in dumpers:
      dumper.close()

  print('Done.')


if __name__ == '__main__':
  main()
