"""Preparation helpers for configuring and initializing shear runs."""

import json
import math
import os

import jax.numpy as jnp
from jax import random
import numpy as np

from jax_md import partition

from shear_init import _box_size_from_phi
from shear_init import _load_initial_state_from_data
from shear_init import _load_initial_state_from_dump
from shear_init import _min_pair_distance
from shear_init import _relax_positions
from shear_output import _serialize_rpy_parameter_estimate
from shear_output import _to_jsonable
from shear_potential import _resolve_potential


def _build_format_map() -> dict:
  """Returns the configured neighbor-list format mapping."""
  return {
    'dense': partition.NeighborListFormat.Dense,
    'sparse': partition.NeighborListFormat.Sparse,
    'ordered': partition.NeighborListFormat.OrderedSparse,
  }


def _resolve_initial_system(args, *, a: float, dim: int, console) -> dict:
  """Resolves initialization mode, particle count, packing fraction, and box."""
  init_mode = 'random_relax'
  data_info = None
  dump_info = None
  if args.init_data is not None:
    init_mode = 'data'
    data_info = _load_initial_state_from_data(
      args.init_data,
      dim=dim,
      radius=a,
    )
    n_particles = int(data_info['n_particles'])
    phi = float(data_info['phi'])
    if n_particles <= 1:
      raise ValueError(f'Data-derived n_particles must be > 1, got {n_particles}.')
    if not math.isfinite(phi) or phi <= 0.0:
      raise ValueError(f'Data-derived phi must be finite and > 0, got {phi}.')
    if phi > 0.74:
      console.warn(
        f'data-derived phi={phi:.6g} exceeds hard-sphere close packing '
        '(~0.74). This can lead to severe overlaps and NaNs in the RPY mobility.'
      )
    base_box = jnp.asarray(data_info['box_matrix'])
    console.info(
      f'Loaded initialization from data {args.init_data}: '
      f'n_particles={n_particles}, atom_style={data_info.get("atom_style", "") or "unspecified"}'
    )
  elif args.init_traj is not None:
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
      console.warn(
        f'dump-derived phi={phi:.6g} exceeds hard-sphere close packing '
        '(~0.74). This can lead to severe overlaps and NaNs in the RPY mobility.'
      )
    base_box = jnp.asarray(dump_info['box_matrix'])
    console.info(
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

  return {
    'init_mode': init_mode,
    'data_info': data_info,
    'dump_info': dump_info,
    'n_particles': n_particles,
    'phi': phi,
    'base_box': base_box,
  }


def _derive_system_dynamics(*, base_box, a, kT, viscosity, peclet) -> dict:
  """Computes box and shear-related derived scalar quantities."""
  base_box_np = np.asarray(base_box, dtype=float)
  box_volume = float(abs(np.linalg.det(base_box_np)))
  box_size = float(box_volume ** (1.0 / 3.0))

  D0 = kT / (6.0 * math.pi * viscosity * a)
  shear_rate = 2.0 * peclet * D0 / (a ** 2)
  shear_t0 = 0.0
  shear_schedule = {'xy': lambda t: shear_rate * t}
  shear_vector_schedule = lambda t: (shear_rate * t, 0.0, 0.0)

  return {
    'base_box_np': base_box_np,
    'box_volume': box_volume,
    'box_size': box_size,
    'D0': D0,
    'shear_rate': shear_rate,
    'shear_t0': shear_t0,
    'shear_schedule': shear_schedule,
    'shear_vector_schedule': shear_vector_schedule,
  }


def _resolve_potential_setup(*, potential_arg: str | None, dt: float, format_map: dict) -> dict:
  """Loads and validates potential configuration and interaction neighbor settings."""
  if potential_arg is None:
    return {
      'use_pair_potential': False,
      'pair_potential_fn': None,
      'potential_params': {},
      'potential_r_cut': 0.0,
      'potential_name': 'none',
      'potential_source': None,
      'interaction_neighbor_defaults': {},
      'interaction_neighbor_format_name': None,
      'interaction_neighbor_format': None,
      'interaction_neighbor_dr_threshold': 0.0,
      'interaction_neighbor_capacity_multiplier': 0.0,
    }

  potential_cfg = _resolve_potential(potential_arg)
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
  if 'repulsion_dt' in potential_params:
    potential_params['repulsion_dt'] = float(dt)
  potential_r_cut = float(potential_cfg['r_cut'])
  potential_name = str(potential_cfg['name'])
  potential_source = str(potential_cfg['source'])

  return {
    'use_pair_potential': True,
    'pair_potential_fn': pair_potential_fn,
    'potential_params': potential_params,
    'potential_r_cut': potential_r_cut,
    'potential_name': potential_name,
    'potential_source': potential_source,
    'interaction_neighbor_defaults': interaction_neighbor_defaults,
    'interaction_neighbor_format_name': interaction_neighbor_format_name,
    'interaction_neighbor_format': interaction_neighbor_format,
    'interaction_neighbor_dr_threshold': interaction_neighbor_dr_threshold,
    'interaction_neighbor_capacity_multiplier': interaction_neighbor_capacity_multiplier,
  }


def _build_initial_positions(
  *,
  init_mode,
  dump_info,
  data_info,
  base_box,
  n_particles,
  dim,
  seed,
  diameter,
  displacement_0,
  shift_0,
  relax_steps,
  relax_neighbor_format,
  relax_neighbor_capacity_multiplier,
  relax_neighbor_dr_threshold,
  format_map,
) -> dict:
  """Creates initial fractional positions and PRNG key for dynamics state init."""
  key = random.PRNGKey(seed)
  _, init_key, run_key = random.split(key, 3)
  if init_mode in ('dump', 'data'):
    init_info = dump_info if init_mode == 'dump' else data_info
    R0 = jnp.asarray(
      init_info['positions_fractional'],
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
  return {
    'R0': R0,
    'min_dist': min_dist,
    'run_key': run_key,
  }


def _build_params_payload(
  *,
  args,
  n_particles: int,
  phi: float,
  dt: float,
  a: float,
  kT: float,
  viscosity: float,
  mr_iters: int,
  tol: float,
  xi_override: float,
  mr_neighbor_format: str,
  mr_dr_threshold: float,
  mr_capacity_multiplier: float,
  relax_steps: int,
  relax_neighbor_format: str,
  relax_neighbor_dr_threshold: float,
  relax_neighbor_capacity_multiplier: float,
  init_mode: str,
  dim: int,
  box_size: float,
  base_box_np: np.ndarray,
  box_volume: float,
  diameter: float,
  potential_r_cut: float,
  D0: float,
  shear_rate: float,
  dump_info,
  data_info,
  confin_path: str,
  planned_steps: int,
  xi: float,
  rpy_rcut: float,
  rpy_P: int,
  rpy_M: int,
  rpy_theta: float,
  rpy_lattice_extent: int,
  rpy_params,
  real_space_mode: str,
  potential_name: str,
  potential_source: str | None,
  potential_params,
  interaction_neighbor_defaults,
) -> dict:
  """Builds the full `params.json` payload for a run."""
  return {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi,
      'dt': dt,
      'n_steps': args.n_steps,
      'peclet': args.peclet,
      'xi': args.xi,
      'stress_every': args.stress_every,
      'traj_every': args.traj_every,
      'progress_every': args.progress_every,
      'mr_skin': args.mr_skin,
      'seed': args.seed,
      'out_dir': args.out_dir,
      'init_traj': args.init_traj,
      'init_data': args.init_data,
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
      'data_source_path': args.init_data,
      'data_atom_style': (
        str(data_info.get('atom_style', '')) if data_info is not None else None
      ),
      'confin_path': confin_path,
      'planned_steps': planned_steps,
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


def _write_params_json(out_dir: str, params: dict) -> str:
  """Writes params payload to `<out_dir>/params.json` and returns the path."""
  params_path = os.path.join(out_dir, 'params.json')
  with open(params_path, 'w') as handle:
    json.dump(params, handle, indent=2, sort_keys=True)
  return params_path
