"""Hard-sphere shear runner with `simulate.brownian_hard_sphere` integrator."""

import os
import shutil
import time

from jax import jit
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import partition
from jax_md import rheo
from jax_md import simulate
from jax_md import smap
from jax_md import space

from shear_hard_sphere_cli import parse_args
from shear_hard_sphere_batch_utils import _check_batch_collision_loop_status
from shear_hard_sphere_batch_utils import _check_batch_nan_positions
from shear_hard_sphere_batch_utils import _check_batch_neighbor_status
from shear_hard_sphere_batch_utils import _prepare_batch_runs
from shear_hard_sphere_batch_utils import _resolve_batch_initial_systems
from shear_hard_sphere_batch_utils import _resolve_hs_common_config
from shear_hard_sphere_batch_utils import _stack_hs_states
from shear_hard_sphere_batch_utils import _stack_neighbor_lists
from shear_hard_sphere_batch_utils import resolve_batch_run_specs
from shear_console import get_console
from shear_init import _build_reduced_xy_box_fn
from shear_output import RunDumper
from shear_output import _to_jsonable
from shear_output import write_lammps_data
from shear_prepare_utils import _build_format_map
from shear_prepare_utils import _build_initial_positions
from shear_prepare_utils import _derive_system_dynamics
from shear_prepare_utils import _resolve_initial_system
from shear_prepare_utils import _resolve_potential_setup
from shear_prepare_utils import _write_params_json
from shear_runtime_utils import _allocate_probe_sized_neighbor
from shear_runtime_utils import _check_interaction_neighbor_status
from shear_time_utils import _predict_xy_remapped_positions_for_next_force
from shear_time_utils import _state_next_time_from_step
from shear_time_utils import _state_step
from shear_time_utils import _state_time_from_step

_CONSOLE = get_console()
_HS_STRESS_FILENAMES = {
  'stress': 'stress.dat',
  'stress_col': 'stress_col.dat',
  'stress_virial': 'stress_virial.dat',
}


def _as_box_matrix(box, *, dim: int) -> np.ndarray:
  """Returns a square box matrix for scalar/vector/matrix box encodings."""
  arr = np.asarray(box, dtype=float)
  if arr.ndim == 0:
    return np.eye(dim, dtype=float) * float(arr)
  if arr.ndim == 1:
    return np.diag(arr)
  return arr


def _packing_fraction(*, n_particles: int, radius: float, box_measure: float, dim: int) -> float:
  """Computes packing fraction from particle count, radius, and box measure."""
  if dim == 2:
    return float(n_particles * np.pi * radius ** 2 / box_measure)
  if dim == 3:
    return float(n_particles * (4.0 / 3.0) * np.pi * radius ** 3 / box_measure)
  raise ValueError(f'Unsupported dim={dim} for packing-fraction calculation.')


def _neighbor_api_extra_capacity(
  *,
  absolute_extra_capacity: int,
  n_particles: int,
  neighbor_format,
) -> int:
  """Converts an absolute pair-slot reserve to neighbor-list API units.

  Sparse neighbor lists interpret `extra_capacity` per particle, so round up
  the absolute reserve to the smallest API value that preserves the request.
  """
  absolute_extra_capacity = int(absolute_extra_capacity)
  if absolute_extra_capacity <= 0:
    return 0
  n_particles = int(n_particles)
  if n_particles <= 0:
    raise ValueError('n_particles must be > 0.')
  if partition.is_sparse(neighbor_format):
    return (absolute_extra_capacity + n_particles - 1) // n_particles
  return absolute_extra_capacity


def _fractional_cell_size_for_cutoff(box, cutoff: float) -> float:
  """Mirrors `partition._fractional_cell_size` for build-box selection."""
  cutoff = float(cutoff)
  if cutoff <= 0.0:
    raise ValueError('cutoff must be > 0.')

  box_arr = np.asarray(box, dtype=float)
  if box_arr.ndim == 0:
    return float(cutoff / float(box_arr))
  if box_arr.ndim == 1:
    return float(cutoff / float(np.min(box_arr)))
  if box_arr.ndim != 2:
    raise ValueError(
      'Expected box to be either a scalar, a vector, or a matrix.'
    )

  dim = int(box_arr.shape[0])
  if dim == 1:
    nmin = int(np.floor(float(box_arr[0, 0]) / cutoff))
    nmin = max(nmin, 1)
    return float(1.0 / nmin)
  if dim == 2:
    xx = float(box_arr[0, 0])
    yy = float(box_arr[1, 1])
    xy = float(box_arr[0, 1] / yy)
    nx = xx / float(np.sqrt(1.0 + xy ** 2))
    ny = yy
    nmin = int(np.floor(min(nx, ny) / cutoff))
    nmin = max(nmin, 1)
    return float(1.0 / nmin)
  if dim == 3:
    xx = float(box_arr[0, 0])
    yy = float(box_arr[1, 1])
    zz = float(box_arr[2, 2])
    xy = float(box_arr[0, 1] / yy)
    xz = float(box_arr[0, 2] / zz)
    yz = float(box_arr[1, 2] / zz)
    nx = xx / float(np.sqrt(1.0 + xy ** 2 + (xy * yz - xz) ** 2))
    ny = yy / float(np.sqrt(1.0 + yz ** 2))
    nz = zz
    nmin = int(np.floor(min(nx, ny, nz) / cutoff))
    nmin = max(nmin, 1)
    return float(1.0 / nmin)
  raise ValueError(
    'Expected box to be either 1-, 2-, or 3-dimensional '
    f'found {dim}.'
  )


def _resolve_worst_xy_remap_build_box(*, box_of, base_box, cutoff: float) -> dict:
  """Selects the build box from gamma_xy in {-0.5, 0.0, +0.5}."""
  dim = int(np.asarray(base_box).shape[0])
  candidates = []
  for gamma_xy in (-0.5, 0.0, 0.5):
    candidate_box = _as_box_matrix(
      box_of(gamma={'xy': float(gamma_xy), 'xz': 0.0, 'yz': 0.0}),
      dim=dim,
    )
    candidate_fractional_cell_size = _fractional_cell_size_for_cutoff(
      candidate_box, cutoff)
    candidates.append({
      'gamma_xy': float(gamma_xy),
      'box': candidate_box,
      'fractional_cell_size': float(candidate_fractional_cell_size),
    })
  return max(candidates, key=lambda c: c['fractional_cell_size'])


def _check_nan_positions(state, stage: str, console=None) -> bool:
  """Returns True if state positions are finite; logs and returns False otherwise."""
  log = _CONSOLE if console is None else console
  has_nan = bool(np.asarray(jnp.any(jnp.isnan(state.position))))
  if has_nan:
    log.error(f'NaN detected in positions during {stage}.')
    return False
  return True


def _check_collision_loop_status(state, stage: str, console=None) -> bool:
  """Checks whether the hard-sphere collision loop hit its per-step cap."""
  log = _CONSOLE if console is None else console
  reached_cap = bool(np.asarray(state.reached_max_collision_loops))
  if reached_cap:
    log.error(
      'Hard-sphere collision loop reached max_collision_loops '
      f'during {stage}. Reduce dt, lower phi, or increase max_collision_loops.'
    )
    return False
  return True


def _check_collision_neighbor_status(collision_neighbor, stage: str, console=None) -> bool:
  """Checks the hard-sphere collision neighbor list health flags."""
  log = _CONSOLE if console is None else console
  if collision_neighbor is None:
    log.error(f'Missing collision neighbor list in stage={stage}.')
    return False
  overflow = np.asarray(collision_neighbor.did_buffer_overflow)
  cell_small = np.asarray(collision_neighbor.cell_size_too_small)
  malformed = np.asarray(collision_neighbor.malformed_box)
  if np.any(overflow):
    log.error(
      f'Collision neighbor list overflow in stage={stage}. '
      'Try increasing --collision-neighbor-capacity-multiplier or '
      '--collision-neighbor-extra-capacity or --collision-neighbor-skin.'
    )
    return False
  if np.any(cell_small):
    log.error(f'Collision neighbor list cell size too small in stage={stage}.')
    return False
  if np.any(malformed):
    log.error(f'Collision neighbor list malformed box in stage={stage}.')
    return False
  return True


def _resolve_collision_neighbor_settings(
  *,
  args,
  diameter: float,
  D0: float,
  dt: float,
  format_map: dict,
) -> dict:
  """Resolves the hard-sphere collision neighbor-list settings."""
  collision_neighbor_format_name = 'sparse'
  collision_neighbor_format = format_map[collision_neighbor_format_name]
  collision_neighbor_zsigma = float(args.collision_neighbor_zsigma)
  collision_neighbor_capacity_multiplier = float(
    args.collision_neighbor_capacity_multiplier
  )
  collision_neighbor_extra_capacity = int(args.collision_neighbor_extra_capacity)
  sigma_rel = float(np.sqrt(4.0 * D0 * dt))
  skin_from_cli = args.collision_neighbor_skin is not None
  if skin_from_cli:
    collision_neighbor_skin = float(args.collision_neighbor_skin)
  else:
    collision_neighbor_skin = float(collision_neighbor_zsigma * sigma_rel)
  collision_neighbor_r_cutoff = float(diameter + collision_neighbor_zsigma * sigma_rel)
  collision_neighbor_threshold = float(
    collision_neighbor_r_cutoff + collision_neighbor_skin
  )
  return {
    'format_name': collision_neighbor_format_name,
    'format': collision_neighbor_format,
    'zsigma': collision_neighbor_zsigma,
    'skin': collision_neighbor_skin,
    'sigma_rel': sigma_rel,
    'r_cutoff': collision_neighbor_r_cutoff,
    'threshold': collision_neighbor_threshold,
    'capacity_multiplier': collision_neighbor_capacity_multiplier,
    'extra_capacity': collision_neighbor_extra_capacity,
    'skin_from_cli': skin_from_cli,
  }


def _build_hs_params_payload(
  *,
  args,
  n_particles: int,
  hydrodynamic_phi: float,
  hs_core_phi: float,
  dt: float,
  hydro_radius: float,
  hs_core_radius: float,
  kT: float,
  viscosity: float,
  mobility: float,
  diameter: float,
  max_collision_loops: int,
  event_time_tol,
  init_mode: str,
  dim: int,
  box_size: float,
  base_box_np: np.ndarray,
  box_volume: float,
  potential_r_cut: float,
  D0: float,
  shear_rate: float,
  dump_info,
  data_info,
  confin_path: str,
  planned_steps: int,
  potential_name: str,
  potential_source: str | None,
  potential_params,
  interaction_neighbor_defaults,
  collision_neighbor_settings,
):
  """Builds a `params.json` payload for hard-sphere shear runs."""
  return {
    'user_args': {
      'n_particles': n_particles,
      'phi': hydrodynamic_phi,
      'dt': dt,
      'n_steps': args.n_steps,
      'peclet': args.peclet,
      'stress_every': args.stress_every,
      'traj_every': args.traj_every,
      'progress_every': args.progress_every,
      'seed': args.seed,
      'out_dir': args.out_dir,
      'init_traj': args.init_traj,
      'init_data': args.init_data,
      'potential': args.potential,
      'hydro_radius': args.hydro_radius,
      'hs_core_radius': args.hs_core_radius,
      'max_collision_loops': args.max_collision_loops,
      'collision_neighbor_zsigma': args.collision_neighbor_zsigma,
      'collision_neighbor_skin': args.collision_neighbor_skin,
      'collision_neighbor_capacity_multiplier': (
        args.collision_neighbor_capacity_multiplier
      ),
      'collision_neighbor_extra_capacity': args.collision_neighbor_extra_capacity,
    },
    'internal': {
      'hydrodynamic_radius': hydro_radius,
      'hs_core_radius': hs_core_radius,
      'kT': kT,
      'viscosity': viscosity,
      'mobility': mobility,
      'diameter': diameter,
      'max_collision_loops': int(max_collision_loops),
      'event_time_tol': event_time_tol,
    },
    'derived': {
      'integrator': 'simulate.brownian_hard_sphere',
      'initialization_mode': init_mode,
      'dim': dim,
      'hydrodynamic_phi': hydrodynamic_phi,
      'hs_core_phi': hs_core_phi,
      'box_size': box_size,
      'box_matrix': _to_jsonable(base_box_np),
      'box_volume': box_volume,
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
    },
    'potential': {
      'selected': args.potential,
      'resolved_name': potential_name,
      'source': potential_source,
      'r_cut': potential_r_cut,
      'params': _to_jsonable(potential_params),
      'neighbor_defaults': _to_jsonable(interaction_neighbor_defaults),
    },
    'collision_neighbor': {
      'format': str(collision_neighbor_settings['format_name']),
      'zsigma': float(collision_neighbor_settings['zsigma']),
      'sigma_rel': float(collision_neighbor_settings['sigma_rel']),
      'r_cutoff': float(collision_neighbor_settings['r_cutoff']),
      'skin': float(collision_neighbor_settings['skin']),
      'threshold': float(collision_neighbor_settings['threshold']),
      'skin_from_cli': bool(collision_neighbor_settings['skin_from_cli']),
      'capacity_multiplier': float(
        collision_neighbor_settings['capacity_multiplier']
      ),
      'extra_capacity': int(collision_neighbor_settings['extra_capacity']),
      'build_policy': str(collision_neighbor_settings['build_policy']),
      'build_gamma_xy': float(collision_neighbor_settings['build_gamma_xy']),
      'build_box': _to_jsonable(collision_neighbor_settings['build_box']),
      'build_fractional_cell_size': float(
        collision_neighbor_settings['build_fractional_cell_size']
      ),
    },
  }


def _build_run_kernels(
  *,
  use_pair_potential: bool,
  apply_fn,
  stress_fn,
  box_of,
  dt: float,
  shear_rate: float,
  shear_t0: float,
):
  """Builds step and stress kernels shared by single-run and batch modes."""
  if use_pair_potential:
    @jit
    def run_one_step(
      state_in,
      collision_neighbor_in,
      interaction_neighbor_in,
    ):
      next_time = _state_next_time_from_step(state_in, dt=dt, t0=shear_t0)
      next_box = box_of(t=next_time)
      pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
        state_in, dt=dt, shear_rate=shear_rate, t0=shear_t0)
      collision_neighbor_out = collision_neighbor_in.update(
        pos_for_neighbor, box=next_box)
      interaction_neighbor_out = interaction_neighbor_in.update(
        pos_for_neighbor, box=next_box)
      state_out = apply_fn(
        state_in,
        neighbor=collision_neighbor_out,
        interaction_neighbor=interaction_neighbor_out,
      )
      curr_time = _state_time_from_step(state_out, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      collision_neighbor_out = collision_neighbor_out.update(
        state_out.position, box=curr_box)
      interaction_neighbor_out = interaction_neighbor_out.update(
        state_out.position, box=curr_box)
      return state_out, collision_neighbor_out, interaction_neighbor_out
  else:
    @jit
    def _run_one_step_without_pair_potential(state_in, collision_neighbor_in):
      next_time = _state_next_time_from_step(state_in, dt=dt, t0=shear_t0)
      next_box = box_of(t=next_time)
      pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
        state_in, dt=dt, shear_rate=shear_rate, t0=shear_t0)
      collision_neighbor_out = collision_neighbor_in.update(
        pos_for_neighbor, box=next_box)
      state_out = apply_fn(state_in, neighbor=collision_neighbor_out)
      curr_time = _state_time_from_step(state_out, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      collision_neighbor_out = collision_neighbor_out.update(
        state_out.position, box=curr_box)
      return state_out, collision_neighbor_out

    def run_one_step(state_in, collision_neighbor_in, interaction_neighbor_in):
      del interaction_neighbor_in
      state_out, collision_neighbor_out = _run_one_step_without_pair_potential(
        state_in, collision_neighbor_in)
      return state_out, collision_neighbor_out, None

  if use_pair_potential:
    @jit
    def evaluate_stress(state_in, interaction_neighbor_in):
      step = _state_step(state_in, dt=dt, t0=shear_t0)
      curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      strain = jnp.asarray(shear_rate, dtype=curr_time.dtype) * curr_time
      stress_col = state_in.stress
      stress_virial = stress_fn(
        state_in.position,
        box=curr_box,
        neighbor=interaction_neighbor_in,
        fractional_coordinates=True,
      )
      return (
        step,
        curr_time,
        strain,
        stress_col,
        stress_virial,
        stress_col + stress_virial,
      )
  else:
    @jit
    def _evaluate_stress_without_pair_potential(state_in):
      step = _state_step(state_in, dt=dt, t0=shear_t0)
      curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
      strain = jnp.asarray(shear_rate, dtype=curr_time.dtype) * curr_time
      stress_col = state_in.stress
      stress_virial = jnp.zeros_like(stress_col)
      return step, curr_time, strain, stress_col, stress_virial, stress_col

    def evaluate_stress(state_in, interaction_neighbor_in):
      del interaction_neighbor_in
      return _evaluate_stress_without_pair_potential(state_in)

  return run_one_step, evaluate_stress


def _run_single(args, wall_start: float):
  """Runs one sheared hard-sphere trajectory and writes run artifacts."""
  # ============================================================================
  # PREPARATION
  # ============================================================================
  common_cfg = _resolve_hs_common_config(args)
  hydro_radius = common_cfg['hydro_radius']
  kT = common_cfg['kT']
  viscosity = common_cfg['viscosity']
  dt = common_cfg['dt']
  relax_steps = common_cfg['relax_steps']
  relax_neighbor_format = common_cfg['relax_neighbor_format']
  relax_neighbor_dr_threshold = common_cfg['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = common_cfg['relax_neighbor_capacity_multiplier']

  # Hard-sphere integrator controls kept explicit in the script.
  hs_core_radius = common_cfg['hs_core_radius']
  diameter = 2.0 * hs_core_radius
  relax_diameter = 2.0 * max(hs_core_radius, hydro_radius)
  max_collision_loops = common_cfg['max_collision_loops']
  event_time_tol = common_cfg['event_time_tol']
  format_map = common_cfg['format_map']

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  dim = 3
  initial_system = _resolve_initial_system(
    args, a=hydro_radius, dim=dim, console=_CONSOLE)
  init_mode = initial_system['init_mode']
  data_info = initial_system['data_info']
  dump_info = initial_system['dump_info']
  n_particles = initial_system['n_particles']
  hydrodynamic_phi = float(initial_system['phi'])
  base_box = initial_system['base_box']

  dynamics = _derive_system_dynamics(
    base_box=base_box,
    a=hydro_radius,
    kT=kT,
    viscosity=viscosity,
    peclet=args.peclet,
  )
  base_box_np = dynamics['base_box_np']
  box_volume = dynamics['box_volume']
  box_size = dynamics['box_size']
  D0 = dynamics['D0']
  shear_rate = dynamics['shear_rate']
  shear_t0 = dynamics['shear_t0']
  shear_schedule = dynamics['shear_schedule']
  hs_core_phi = _packing_fraction(
    n_particles=n_particles,
    radius=hs_core_radius,
    box_measure=box_volume,
    dim=dim,
  )

  # kT * mobility = D0 for isolated spheres.
  mobility = D0 / kT
  collision_neighbor_settings = _resolve_collision_neighbor_settings(
    args=args,
    diameter=diameter,
    D0=D0,
    dt=dt,
    format_map=format_map,
  )
  collision_neighbor_format_name = collision_neighbor_settings['format_name']
  collision_neighbor_format = collision_neighbor_settings['format']
  collision_neighbor_zsigma = collision_neighbor_settings['zsigma']
  collision_neighbor_skin = collision_neighbor_settings['skin']
  collision_neighbor_sigma_rel = collision_neighbor_settings['sigma_rel']
  collision_neighbor_r_cutoff = collision_neighbor_settings['r_cutoff']
  collision_neighbor_threshold = collision_neighbor_settings['threshold']
  collision_neighbor_skin_from_cli = collision_neighbor_settings['skin_from_cli']
  collision_neighbor_capacity_multiplier = collision_neighbor_settings[
    'capacity_multiplier'
  ]
  collision_neighbor_extra_capacity = collision_neighbor_settings['extra_capacity']

  _CONSOLE.section('System')
  _CONSOLE.info(f'Equivalent box size L = {box_size:.6f}')
  _CONSOLE.info(f'D0 = {D0:.6e}')
  _CONSOLE.info(f'Mobility = {mobility:.6e}')
  _CONSOLE.info(f'Hydrodynamic radius a = {hydro_radius:.6f}')
  _CONSOLE.info(f'Hard-sphere core radius = {hs_core_radius:.6f}')
  _CONSOLE.info(f'Hydrodynamic packing fraction phi = {hydrodynamic_phi:.6f}')
  if not np.isclose(hydrodynamic_phi, hs_core_phi, rtol=1e-12, atol=1e-12):
    _CONSOLE.info(f'Hard-sphere core packing fraction phi = {hs_core_phi:.6f}')
  _CONSOLE.info(f'Hard-sphere diameter = {diameter:.6f}')
  _CONSOLE.info(f'Relaxation diameter = {relax_diameter:.6f}')
  _CONSOLE.info(f'Max collision loops/step = {max_collision_loops}')
  _CONSOLE.info(f'Shear rate = {shear_rate:.6e}')
  _CONSOLE.info(f'Strain per step = {shear_rate * dt:.6e}')
  _CONSOLE.info(
    'Collision neighbors: '
    f'format={collision_neighbor_format_name}, '
    f'zsigma={collision_neighbor_zsigma:.3g}, '
    f'sigma_rel={collision_neighbor_sigma_rel:.6e}, '
    f'r_cutoff={collision_neighbor_r_cutoff:.6f}, '
    f'skin={collision_neighbor_skin:.6f}'
    f'({"cli" if collision_neighbor_skin_from_cli else "auto"}), '
    f'threshold={collision_neighbor_threshold:.6f}, '
    f'capacity_multiplier={collision_neighbor_capacity_multiplier:.3g}, '
    f'extra_capacity={collision_neighbor_extra_capacity}'
  )

  potential_setup = _resolve_potential_setup(
    potential_arg=args.potential,
    dt=dt,
    format_map=format_map,
  )
  use_pair_potential = bool(potential_setup['use_pair_potential'])
  interaction_neighbor_defaults = potential_setup['interaction_neighbor_defaults']
  interaction_neighbor_format_name = potential_setup['interaction_neighbor_format_name']
  interaction_neighbor_format = potential_setup['interaction_neighbor_format']
  interaction_neighbor_dr_threshold = potential_setup['interaction_neighbor_dr_threshold']
  interaction_neighbor_capacity_multiplier = potential_setup['interaction_neighbor_capacity_multiplier']
  pair_potential_fn = potential_setup['pair_potential_fn']
  potential_params = potential_setup['potential_params']
  potential_r_cut = potential_setup['potential_r_cut']
  potential_name = potential_setup['potential_name']
  potential_source = potential_setup['potential_source']

  _CONSOLE.section('Potential')
  if use_pair_potential:
    _CONSOLE.info(
      f'Potential module: {args.potential} -> {potential_name} '
      f'(source={potential_source}, r_cut={potential_r_cut:.6f})'
    )
    _CONSOLE.info(
      'Interaction neighbor defaults: '
      f'format={interaction_neighbor_format_name}, '
      f'dr_threshold={interaction_neighbor_dr_threshold:.3g}, '
      f'capacity_multiplier={interaction_neighbor_capacity_multiplier:.3g}'
    )
  else:
    _CONSOLE.info(
      'No potential selected; integrating with zero potential energy.'
    )
    _CONSOLE.info('Pair-interaction neighbor list disabled.')

  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  displacement_0, shift_0 = space.periodic_general(base_box, fractional_coordinates=True)

  initial_positions = _build_initial_positions(
    init_mode=init_mode,
    dump_info=dump_info,
    data_info=data_info,
    base_box=base_box,
    n_particles=n_particles,
    dim=dim,
    seed=args.seed,
    diameter=relax_diameter,
    displacement_0=displacement_0,
    shift_0=shift_0,
    relax_steps=relax_steps,
    relax_neighbor_format=relax_neighbor_format,
    relax_neighbor_capacity_multiplier=relax_neighbor_capacity_multiplier,
    relax_neighbor_dr_threshold=relax_neighbor_dr_threshold,
    format_map=format_map,
  )
  R0 = initial_positions['R0']
  run_key = initial_positions['run_key']
  min_dist = initial_positions['min_dist']
  if init_mode == 'random_relax':
    _CONSOLE.info(f'Post-relax minimum pair distance: {min_dist:.6f}')
  else:
    _CONSOLE.info(f'Loaded minimum pair distance: {min_dist:.6f}')

  metric_shear = space.canonicalize_displacement_or_metric(displacement)
  collision_build_box_info = _resolve_worst_xy_remap_build_box(
    box_of=box_of,
    base_box=base_box,
    cutoff=collision_neighbor_threshold,
  )
  collision_neighbor_settings['build_policy'] = 'worst_xy_remap_corner'
  collision_neighbor_settings['build_gamma_xy'] = collision_build_box_info[
    'gamma_xy'
  ]
  collision_neighbor_settings['build_box'] = np.asarray(
    collision_build_box_info['box'], dtype=float)
  collision_neighbor_settings['build_fractional_cell_size'] = float(
    collision_build_box_info['fractional_cell_size']
  )
  _CONSOLE.info(
    'Collision build box: '
    'policy=worst_xy_remap_corner, '
    f'gamma_xy={collision_build_box_info["gamma_xy"]:+.3f}, '
    f'r_cutoff={collision_neighbor_r_cutoff:.6f}, '
    f'threshold={collision_neighbor_threshold:.6f}, '
    f'fractional_cell_size={collision_build_box_info["fractional_cell_size"]:.6f}'
  )

  collision_neighbor_fn = partition.neighbor_list(
    metric_shear,
    base_box,
    r_cutoff=collision_neighbor_r_cutoff,
    dr_threshold=collision_neighbor_skin,
    capacity_multiplier=collision_neighbor_capacity_multiplier,
    fractional_coordinates=True,
    format=collision_neighbor_format,
  )
  if use_pair_potential:
    energy_fn_all_pairs = smap.pair(
      pair_potential_fn,
      metric_shear,
      ignore_unused_parameters=True,
      **potential_params,
    )
    energy_fn_neighbor = smap.pair_neighbor_list(
      pair_potential_fn,
      metric_shear,
      ignore_unused_parameters=True,
      **potential_params,
    )

    def energy_fn(R, interaction_neighbor=None, **kwargs):
      kwargs = dict(kwargs)
      kwargs.pop('interaction_neighbor', None)
      kwargs.pop('neighbor', None)
      if interaction_neighbor is None:
        return energy_fn_all_pairs(R, **kwargs)
      return energy_fn_neighbor(R, neighbor=interaction_neighbor, **kwargs)

    interaction_neighbor_fn = partition.neighbor_list(
      metric_shear,
      base_box,
      r_cutoff=potential_r_cut,
      dr_threshold=interaction_neighbor_dr_threshold,
      capacity_multiplier=interaction_neighbor_capacity_multiplier,
      fractional_coordinates=True,
      format=interaction_neighbor_format,
    )
    interaction_neighbor_threshold = float(
      potential_r_cut + interaction_neighbor_dr_threshold
    )
    interaction_build_box_info = _resolve_worst_xy_remap_build_box(
      box_of=box_of,
      base_box=base_box,
      cutoff=interaction_neighbor_threshold,
    )
    _CONSOLE.info(
      'Interaction build box: '
      'policy=worst_xy_remap_corner, '
      f'gamma_xy={interaction_build_box_info["gamma_xy"]:+.3f}, '
      f'r_cutoff={potential_r_cut:.6f}, '
      f'threshold={interaction_neighbor_threshold:.6f}, '
      f'fractional_cell_size={interaction_build_box_info["fractional_cell_size"]:.6f}'
    )
  else:
    def energy_fn(R, **unused_kwargs):
      return jnp.zeros((), dtype=R.dtype)

    interaction_neighbor_fn = None
    interaction_build_box_info = None

  do_stress = args.stress_every > 0
  do_traj = args.traj_every > 0
  stress_fn = None
  if do_stress and use_pair_potential:
    stress_fn = rheo.make_pairwise_stress_fn(
      pair_potential_fn,
      **potential_params,
    )
  elif do_stress:
    _CONSOLE.warn(
      '--stress_every > 0 with no potential selected; '
      'writing zero virial stress.'
    )

  init_fn, apply_fn = simulate.brownian_hard_sphere(
    energy_fn,
    displacement,
    shift,
    dt=dt,
    kT=kT,
    diameter=diameter,
    mobility=mobility,
    max_collision_loops=max_collision_loops,
    event_time_tol=event_time_tol,
    shear_schedule=shear_schedule,
    t0=shear_t0,
    fractional_coordinates=True,
    remap=True,
    box_fn=box_of,
  )

  planned_steps = int(args.n_steps)
  run_one_step, evaluate_stress = _build_run_kernels(
    use_pair_potential=use_pair_potential,
    apply_fn=apply_fn,
    stress_fn=stress_fn,
    box_of=box_of,
    dt=dt,
    shear_rate=shear_rate,
    shear_t0=shear_t0,
  )

  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  confin_path = os.path.join(out_dir, 'confin.data')
  if init_mode == 'data':
    shutil.copyfile(args.init_data, confin_path)
    _CONSOLE.info(f'Copied init data file to {confin_path}')
  else:
    init_frac = np.mod(np.asarray(R0, dtype=float), 1.0)
    init_pos_real = np.asarray(init_frac @ base_box_np.T, dtype=float)
    write_lammps_data(
      confin_path,
      base_box_np,
      init_pos_real,
      comment=(
        'Generated by examples/shear/shear_hard_sphere.py '
        f'(init_mode={init_mode})'
      ),
    )
    _CONSOLE.info(f'Wrote initial configuration to {confin_path}')

  params = _build_hs_params_payload(
    args=args,
    n_particles=n_particles,
    hydrodynamic_phi=hydrodynamic_phi,
    hs_core_phi=hs_core_phi,
    dt=dt,
    hydro_radius=hydro_radius,
    hs_core_radius=hs_core_radius,
    kT=kT,
    viscosity=viscosity,
    mobility=mobility,
    diameter=diameter,
    max_collision_loops=max_collision_loops,
    event_time_tol=event_time_tol,
    init_mode=init_mode,
    dim=dim,
    box_size=box_size,
    base_box_np=base_box_np,
    box_volume=box_volume,
    potential_r_cut=potential_r_cut,
    D0=D0,
    shear_rate=shear_rate,
    dump_info=dump_info,
    data_info=data_info,
    confin_path=confin_path,
    planned_steps=planned_steps,
    potential_name=potential_name,
    potential_source=potential_source,
    potential_params=potential_params,
    interaction_neighbor_defaults=interaction_neighbor_defaults,
    collision_neighbor_settings=collision_neighbor_settings,
  )
  params_path = _write_params_json(out_dir, params)
  _CONSOLE.info(f'Wrote parameters to {params_path}')

  dump_box_fn = _build_reduced_xy_box_fn(np.asarray(base_box, dtype=float), shear_rate)
  base_box_np = np.asarray(base_box, dtype=float)
  dumper = RunDumper(
    out_dir,
    box_size,
    dim,
    dt,
    args.traj_every,
    args.stress_every,
    box_fn=dump_box_fn,
    base_box=base_box_np,
    shear_rate=shear_rate,
    time_offset=shear_t0,
    shear_remap=True,
    unwrap_trajectory=True,
    stress_filenames=_HS_STRESS_FILENAMES,
  )

  _CONSOLE.section('Run Plan')
  _CONSOLE.info(
    f'Running one sheared hard-sphere trajectory for {planned_steps} steps '
    f'(requested {args.n_steps}).'
  )
  if do_stress and do_traj:
    _CONSOLE.info(
      'Outputs: stress.dat + stress_col.dat + stress_virial.dat '
      f'+ traj.dump in {out_dir}'
    )
  elif do_stress:
    _CONSOLE.info(
      f'Outputs: stress.dat + stress_col.dat + stress_virial.dat in {out_dir}'
    )
  elif do_traj:
    _CONSOLE.info(f'Outputs: traj.dump in {out_dir}')
  else:
    _CONSOLE.info('Outputs: none (both --stress_every and --traj_every are 0).')

  # ============================================================================
  # SIMULATION
  # ============================================================================
  try:
    state = init_fn(run_key, R0)
    collision_neighbor_api_extra_capacity = _neighbor_api_extra_capacity(
      absolute_extra_capacity=collision_neighbor_extra_capacity,
      n_particles=state.position.shape[0],
      neighbor_format=collision_neighbor_format,
    )
    # `box` is the physical/reference box at t0; `build_box` is the
    # worst-remap envelope used only for conservative cell-list sizing.
    box_t0 = box_of(t=shear_t0)
    collision_build_box = jnp.asarray(
      collision_build_box_info['box'], dtype=base_box.dtype)
    collision_neighbor = _allocate_probe_sized_neighbor(
      collision_neighbor_fn,
      state.position,
      box=box_t0,
      build_box=collision_build_box,
      extra_capacity=collision_neighbor_api_extra_capacity,
    )
    if not _check_collision_neighbor_status(collision_neighbor, 'shear_init'):
      return
    interaction_neighbor = None
    if use_pair_potential:
      interaction_build_box = jnp.asarray(
        interaction_build_box_info['box'], dtype=base_box.dtype)
      interaction_neighbor = _allocate_probe_sized_neighbor(
        interaction_neighbor_fn,
        state.position,
        box=box_t0,
        build_box=interaction_build_box,
      )
      if not _check_interaction_neighbor_status(interaction_neighbor, 'shear_init'):
        return

    if do_traj:
      if do_stress:
        (
          stress_step,
          _,
          stress_strain,
          stress_col,
          stress_virial,
          stress_total,
        ) = evaluate_stress(state, interaction_neighbor)
        out_stress_times = np.array(
          [int(np.asarray(stress_step)) * dt + float(shear_t0)], dtype=float)
        out_stress_strains = np.array([float(np.asarray(stress_strain))], dtype=float)
        out_stresses_col = np.asarray(stress_col, dtype=float)[np.newaxis]
        out_stresses_virial = np.asarray(stress_virial, dtype=float)[np.newaxis]
        out_stresses = np.asarray(stress_total, dtype=float)[np.newaxis]
      else:
        out_stress_times = np.array([], dtype=float)
        out_stress_strains = np.array([], dtype=float)
        out_stresses_col = np.zeros((0, dim, dim), dtype=float)
        out_stresses_virial = np.zeros((0, dim, dim), dtype=float)
        out_stresses = np.zeros((0, dim, dim), dtype=float)
      dumper.dump(
        out_stress_times,
        out_stress_strains,
        out_stresses,
        None,
        np.asarray(state.position, dtype=float)[np.newaxis],
        traj_steps=np.array([0], dtype=np.int64),
        stress_components={
          'stress_col': out_stresses_col,
          'stress_virial': out_stresses_virial,
        },
      )

    steps_done = 0
    while steps_done < planned_steps:
      state, collision_neighbor, interaction_neighbor = run_one_step(
        state,
        collision_neighbor,
        interaction_neighbor,
      )
      steps_done += 1

      if not _check_nan_positions(state, f'shear step {steps_done}'):
        return
      if not _check_collision_loop_status(state, f'shear step {steps_done}'):
        return
      if not _check_collision_neighbor_status(
        collision_neighbor, f'shear step {steps_done}'):
        return
      if use_pair_potential and not _check_interaction_neighbor_status(
        interaction_neighbor, f'shear step {steps_done}'):
        return

      emit_stress = do_stress and (steps_done % args.stress_every == 0)
      emit_traj = do_traj and (steps_done % args.traj_every == 0)
      if emit_stress:
        (
          stress_step,
          _,
          stress_strain,
          stress_col,
          stress_virial,
          stress_total,
        ) = evaluate_stress(state, interaction_neighbor)
        out_stress_times = np.array(
          [int(np.asarray(stress_step)) * dt + float(shear_t0)], dtype=float)
        out_stress_strains = np.array([float(np.asarray(stress_strain))], dtype=float)
        out_stresses_col = np.asarray(stress_col, dtype=float)[np.newaxis]
        out_stresses_virial = np.asarray(stress_virial, dtype=float)[np.newaxis]
        out_stresses = np.asarray(stress_total, dtype=float)[np.newaxis]
      else:
        out_stress_times = None
        out_stress_strains = None
        out_stresses_col = None
        out_stresses_virial = None
        out_stresses = None

      if emit_traj:
        out_traj_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
        out_traj_steps = np.array([out_traj_step], dtype=np.int64)
        out_traj_positions = np.asarray(state.position, dtype=float)[np.newaxis]
      else:
        out_traj_steps = None
        out_traj_positions = None

      if emit_stress or emit_traj:
        dumper.dump(
          out_stress_times,
          out_stress_strains,
          out_stresses,
          None,
          out_traj_positions,
          traj_steps=out_traj_steps,
          stress_components={
            'stress_col': out_stresses_col,
            'stress_virial': out_stresses_virial,
          },
        )

      if args.progress_every > 0 and (steps_done % args.progress_every == 0):
        _CONSOLE.progress(f'Step {min(steps_done, planned_steps)}/{planned_steps}')

    final_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
    final_time = float(final_step * dt + shear_t0)
    final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
    pos_frac = np.mod(np.asarray(state.position, dtype=float), 1.0)
    pos_real = np.asarray(pos_frac @ final_box.T, dtype=float)

    confout_path = os.path.join(out_dir, 'confout.data')
    write_lammps_data(
      confout_path,
      final_box,
      pos_real,
      comment=(
        'Generated by examples/shear/shear_hard_sphere.py '
        f'(step={final_step})'
      ),
    )
    _CONSOLE.info(f'Wrote final data snapshot {confout_path}')
  finally:
    dumper.close()

  # ============================================================================
  # OUTPUT
  # ============================================================================
  elapsed_s = time.perf_counter() - wall_start
  total_steps = int(planned_steps)
  _CONSOLE.section('Timing')
  _CONSOLE.info(f'Total wall time: {elapsed_s:.3f} s')
  if total_steps > 0 and elapsed_s > 0.0:
    seconds_per_step = elapsed_s / float(total_steps)
    ptps = (float(n_particles) * float(total_steps)) / elapsed_s
    _CONSOLE.info(
      'Time per step: '
      f'{seconds_per_step:.6e} s/step ({seconds_per_step * 1e3:.6f} ms/step)'
    )
    _CONSOLE.info(f'PTPS: {ptps:.6e} particle-timesteps/s')
  else:
    _CONSOLE.warn('Skipping per-step timing/PTPS (no executed steps).')

  _CONSOLE.success('Done.')


def _run_batch(args, wall_start: float):
  """Runs multiple same-shape hard-sphere trajectories in one JAX process."""
  common_cfg = _resolve_hs_common_config(args)
  hydro_radius = common_cfg['hydro_radius']
  kT = common_cfg['kT']
  viscosity = common_cfg['viscosity']
  dt = common_cfg['dt']
  relax_steps = common_cfg['relax_steps']
  relax_neighbor_format = common_cfg['relax_neighbor_format']
  relax_neighbor_dr_threshold = common_cfg['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = common_cfg[
    'relax_neighbor_capacity_multiplier'
  ]
  hs_core_radius = common_cfg['hs_core_radius']
  diameter = 2.0 * hs_core_radius
  relax_diameter = 2.0 * max(hs_core_radius, hydro_radius)
  max_collision_loops = common_cfg['max_collision_loops']
  event_time_tol = common_cfg['event_time_tol']
  format_map = common_cfg['format_map']

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  dim = 3
  run_specs, resolved_runs = _resolve_batch_initial_systems(
    args,
    packing_radius=hydro_radius,
    dim=dim,
    console=_CONSOLE,
  )
  n_runs = len(run_specs)
  _CONSOLE.section('Batch')
  _CONSOLE.info(f'Running {n_runs} same-shape simulations in one process.')
  _CONSOLE.info('Run labels: ' + ', '.join(run_spec.label for run_spec in run_specs))

  base_box = resolved_runs[0][2]['base_box']
  n_particles = int(resolved_runs[0][2]['n_particles'])
  hydrodynamic_phi = float(resolved_runs[0][2]['phi'])
  dynamics = _derive_system_dynamics(
    base_box=base_box,
    a=hydro_radius,
    kT=kT,
    viscosity=viscosity,
    peclet=args.peclet,
  )
  base_box_np = dynamics['base_box_np']
  box_volume = dynamics['box_volume']
  box_size = dynamics['box_size']
  D0 = dynamics['D0']
  shear_rate = dynamics['shear_rate']
  shear_t0 = dynamics['shear_t0']
  shear_schedule = dynamics['shear_schedule']
  hs_core_phi = _packing_fraction(
    n_particles=n_particles,
    radius=hs_core_radius,
    box_measure=box_volume,
    dim=dim,
  )

  mobility = D0 / kT
  collision_neighbor_settings = _resolve_collision_neighbor_settings(
    args=args,
    diameter=diameter,
    D0=D0,
    dt=dt,
    format_map=format_map,
  )
  collision_neighbor_format_name = collision_neighbor_settings['format_name']
  collision_neighbor_format = collision_neighbor_settings['format']
  collision_neighbor_zsigma = collision_neighbor_settings['zsigma']
  collision_neighbor_skin = collision_neighbor_settings['skin']
  collision_neighbor_sigma_rel = collision_neighbor_settings['sigma_rel']
  collision_neighbor_r_cutoff = collision_neighbor_settings['r_cutoff']
  collision_neighbor_threshold = collision_neighbor_settings['threshold']
  collision_neighbor_skin_from_cli = collision_neighbor_settings['skin_from_cli']
  collision_neighbor_capacity_multiplier = collision_neighbor_settings[
    'capacity_multiplier'
  ]
  collision_neighbor_extra_capacity = collision_neighbor_settings['extra_capacity']

  _CONSOLE.section('System')
  _CONSOLE.info(f'Equivalent box size L = {box_size:.6f}')
  _CONSOLE.info(f'D0 = {D0:.6e}')
  _CONSOLE.info(f'Mobility = {mobility:.6e}')
  _CONSOLE.info(f'Hydrodynamic radius a = {hydro_radius:.6f}')
  _CONSOLE.info(f'Hard-sphere core radius = {hs_core_radius:.6f}')
  _CONSOLE.info(f'Hydrodynamic packing fraction phi = {hydrodynamic_phi:.6f}')
  if not np.isclose(hydrodynamic_phi, hs_core_phi, rtol=1e-12, atol=1e-12):
    _CONSOLE.info(f'Hard-sphere core packing fraction phi = {hs_core_phi:.6f}')
  _CONSOLE.info(f'Hard-sphere diameter = {diameter:.6f}')
  _CONSOLE.info(f'Relaxation diameter = {relax_diameter:.6f}')
  _CONSOLE.info(f'Max collision loops/step = {max_collision_loops}')
  _CONSOLE.info(f'Shear rate = {shear_rate:.6e}')
  _CONSOLE.info(f'Strain per step = {shear_rate * dt:.6e}')
  _CONSOLE.info(
    'Collision neighbors: '
    f'format={collision_neighbor_format_name}, '
    f'zsigma={collision_neighbor_zsigma:.3g}, '
    f'sigma_rel={collision_neighbor_sigma_rel:.6e}, '
    f'r_cutoff={collision_neighbor_r_cutoff:.6f}, '
    f'skin={collision_neighbor_skin:.6f}'
    f'({"cli" if collision_neighbor_skin_from_cli else "auto"}), '
    f'threshold={collision_neighbor_threshold:.6f}, '
    f'capacity_multiplier={collision_neighbor_capacity_multiplier:.3g}, '
    f'extra_capacity={collision_neighbor_extra_capacity}'
  )

  potential_setup = _resolve_potential_setup(
    potential_arg=args.potential,
    dt=dt,
    format_map=format_map,
  )
  use_pair_potential = bool(potential_setup['use_pair_potential'])
  interaction_neighbor_defaults = potential_setup['interaction_neighbor_defaults']
  interaction_neighbor_format_name = potential_setup['interaction_neighbor_format_name']
  interaction_neighbor_format = potential_setup['interaction_neighbor_format']
  interaction_neighbor_dr_threshold = potential_setup['interaction_neighbor_dr_threshold']
  interaction_neighbor_capacity_multiplier = potential_setup[
    'interaction_neighbor_capacity_multiplier'
  ]
  pair_potential_fn = potential_setup['pair_potential_fn']
  potential_params = potential_setup['potential_params']
  potential_r_cut = potential_setup['potential_r_cut']
  potential_name = potential_setup['potential_name']
  potential_source = potential_setup['potential_source']

  _CONSOLE.section('Potential')
  if use_pair_potential:
    _CONSOLE.info(
      f'Potential module: {args.potential} -> {potential_name} '
      f'(source={potential_source}, r_cut={potential_r_cut:.6f})'
    )
    _CONSOLE.info(
      'Interaction neighbor defaults: '
      f'format={interaction_neighbor_format_name}, '
      f'dr_threshold={interaction_neighbor_dr_threshold:.3g}, '
      f'capacity_multiplier={interaction_neighbor_capacity_multiplier:.3g}'
    )
  else:
    _CONSOLE.info(
      'No potential selected; integrating with zero potential energy.'
    )
    _CONSOLE.info('Pair-interaction neighbor list disabled.')

  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  displacement_0, shift_0 = space.periodic_general(
    base_box, fractional_coordinates=True)

  prepared_runs = _prepare_batch_runs(
    resolved_runs,
    base_box=base_box,
    n_particles=n_particles,
    dim=dim,
    diameter=relax_diameter,
    displacement_0=displacement_0,
    shift_0=shift_0,
    relax_steps=relax_steps,
    relax_neighbor_format=relax_neighbor_format,
    relax_neighbor_capacity_multiplier=relax_neighbor_capacity_multiplier,
    relax_neighbor_dr_threshold=relax_neighbor_dr_threshold,
    format_map=format_map,
  )
  for prepared_run in prepared_runs:
    if prepared_run.init_mode == 'random_relax':
      _CONSOLE.info(
        f'[{prepared_run.spec.label}] Post-relax minimum pair distance: '
        f'{prepared_run.min_dist:.6f}'
      )
    else:
      _CONSOLE.info(
        f'[{prepared_run.spec.label}] Loaded minimum pair distance: '
        f'{prepared_run.min_dist:.6f}'
      )

  metric_shear = space.canonicalize_displacement_or_metric(displacement)
  collision_build_box_info = _resolve_worst_xy_remap_build_box(
    box_of=box_of,
    base_box=base_box,
    cutoff=collision_neighbor_threshold,
  )
  collision_neighbor_settings['build_policy'] = 'worst_xy_remap_corner'
  collision_neighbor_settings['build_gamma_xy'] = collision_build_box_info[
    'gamma_xy'
  ]
  collision_neighbor_settings['build_box'] = np.asarray(
    collision_build_box_info['box'], dtype=float)
  collision_neighbor_settings['build_fractional_cell_size'] = float(
    collision_build_box_info['fractional_cell_size']
  )
  _CONSOLE.info(
    'Collision build box: '
    'policy=worst_xy_remap_corner, '
    f'gamma_xy={collision_build_box_info["gamma_xy"]:+.3f}, '
    f'r_cutoff={collision_neighbor_r_cutoff:.6f}, '
    f'threshold={collision_neighbor_threshold:.6f}, '
    f'fractional_cell_size={collision_build_box_info["fractional_cell_size"]:.6f}'
  )

  collision_neighbor_fn = partition.neighbor_list(
    metric_shear,
    base_box,
    r_cutoff=collision_neighbor_r_cutoff,
    dr_threshold=collision_neighbor_skin,
    capacity_multiplier=collision_neighbor_capacity_multiplier,
    fractional_coordinates=True,
    format=collision_neighbor_format,
  )
  if use_pair_potential:
    energy_fn_all_pairs = smap.pair(
      pair_potential_fn,
      metric_shear,
      ignore_unused_parameters=True,
      **potential_params,
    )
    energy_fn_neighbor = smap.pair_neighbor_list(
      pair_potential_fn,
      metric_shear,
      ignore_unused_parameters=True,
      **potential_params,
    )

    def energy_fn(R, interaction_neighbor=None, **kwargs):
      kwargs = dict(kwargs)
      kwargs.pop('interaction_neighbor', None)
      kwargs.pop('neighbor', None)
      if interaction_neighbor is None:
        return energy_fn_all_pairs(R, **kwargs)
      return energy_fn_neighbor(R, neighbor=interaction_neighbor, **kwargs)

    interaction_neighbor_fn = partition.neighbor_list(
      metric_shear,
      base_box,
      r_cutoff=potential_r_cut,
      dr_threshold=interaction_neighbor_dr_threshold,
      capacity_multiplier=interaction_neighbor_capacity_multiplier,
      fractional_coordinates=True,
      format=interaction_neighbor_format,
    )
    interaction_neighbor_threshold = float(
      potential_r_cut + interaction_neighbor_dr_threshold
    )
    interaction_build_box_info = _resolve_worst_xy_remap_build_box(
      box_of=box_of,
      base_box=base_box,
      cutoff=interaction_neighbor_threshold,
    )
    _CONSOLE.info(
      'Interaction build box: '
      'policy=worst_xy_remap_corner, '
      f'gamma_xy={interaction_build_box_info["gamma_xy"]:+.3f}, '
      f'r_cutoff={potential_r_cut:.6f}, '
      f'threshold={interaction_neighbor_threshold:.6f}, '
      f'fractional_cell_size={interaction_build_box_info["fractional_cell_size"]:.6f}'
    )
  else:
    def energy_fn(R, **unused_kwargs):
      return jnp.zeros((), dtype=R.dtype)

    interaction_neighbor_fn = None
    interaction_build_box_info = None

  do_stress = args.stress_every > 0
  do_traj = args.traj_every > 0
  stress_fn = None
  if do_stress and use_pair_potential:
    stress_fn = rheo.make_pairwise_stress_fn(
      pair_potential_fn,
      **potential_params,
    )
  elif do_stress:
    _CONSOLE.warn(
      '--stress_every > 0 with no potential selected; '
      'writing zero virial stress.'
    )

  init_fn, apply_fn = simulate.brownian_hard_sphere(
    energy_fn,
    displacement,
    shift,
    dt=dt,
    kT=kT,
    diameter=diameter,
    mobility=mobility,
    max_collision_loops=max_collision_loops,
    event_time_tol=event_time_tol,
    shear_schedule=shear_schedule,
    t0=shear_t0,
    fractional_coordinates=True,
    remap=True,
    box_fn=box_of,
  )

  planned_steps = int(args.n_steps)
  run_one_step, evaluate_stress = _build_run_kernels(
    use_pair_potential=use_pair_potential,
    apply_fn=apply_fn,
    stress_fn=stress_fn,
    box_of=box_of,
    dt=dt,
    shear_rate=shear_rate,
    shear_t0=shear_t0,
  )
  if use_pair_potential:
    batched_run_one_step = jit(jax.vmap(run_one_step))
  else:
    batched_run_one_step = jit(jax.vmap(run_one_step, in_axes=(0, 0, None)))
  if use_pair_potential:
    batched_evaluate_stress = jit(jax.vmap(evaluate_stress))
  else:
    batched_evaluate_stress = jit(jax.vmap(evaluate_stress, in_axes=(0, None)))

  dump_box_fn = _build_reduced_xy_box_fn(np.asarray(base_box, dtype=float), shear_rate)
  box_t0 = box_of(t=shear_t0)
  collision_build_box = jnp.asarray(
    collision_build_box_info['box'], dtype=base_box.dtype)
  interaction_build_box = None
  if use_pair_potential:
    interaction_build_box = jnp.asarray(
      interaction_build_box_info['box'], dtype=base_box.dtype)

  _CONSOLE.section('Run Plan')
  _CONSOLE.info(
    f'Running {n_runs} batched sheared hard-sphere trajectories for '
    f'{planned_steps} steps (requested {args.n_steps}).'
  )
  if do_stress and do_traj:
    _CONSOLE.info(
      'Outputs: per-run stress.dat + stress_col.dat + stress_virial.dat '
      f'+ traj.dump under {args.out_dir}'
    )
  elif do_stress:
    _CONSOLE.info(
      'Outputs: per-run stress.dat + stress_col.dat + stress_virial.dat '
      f'under {args.out_dir}'
    )
  elif do_traj:
    _CONSOLE.info(f'Outputs: per-run traj.dump under {args.out_dir}')
  else:
    _CONSOLE.info('Outputs: none (both --stress_every and --traj_every are 0).')

  dumpers = []
  try:
    os.makedirs(args.out_dir, exist_ok=True)
    states = []
    collision_neighbors = []
    interaction_neighbors = []
    for prepared_run in prepared_runs:
      out_dir = prepared_run.spec.out_dir
      os.makedirs(out_dir, exist_ok=True)
      confin_path = os.path.join(out_dir, 'confin.data')
      if prepared_run.init_mode == 'data':
        shutil.copyfile(prepared_run.args.init_data, confin_path)
        _CONSOLE.info(
          f'[{prepared_run.spec.label}] Copied init data file to {confin_path}'
        )
      else:
        init_frac = np.mod(np.asarray(prepared_run.R0, dtype=float), 1.0)
        init_pos_real = np.asarray(init_frac @ base_box_np.T, dtype=float)
        write_lammps_data(
          confin_path,
          base_box_np,
          init_pos_real,
          comment=(
            'Generated by examples/shear/shear_hard_sphere.py '
            f'(init_mode={prepared_run.init_mode})'
          ),
        )
        _CONSOLE.info(
          f'[{prepared_run.spec.label}] Wrote initial configuration to {confin_path}'
        )

      params = _build_hs_params_payload(
        args=prepared_run.args,
        n_particles=prepared_run.n_particles,
        hydrodynamic_phi=prepared_run.phi,
        hs_core_phi=hs_core_phi,
        dt=dt,
        hydro_radius=hydro_radius,
        hs_core_radius=hs_core_radius,
        kT=kT,
        viscosity=viscosity,
        mobility=mobility,
        diameter=diameter,
        max_collision_loops=max_collision_loops,
        event_time_tol=event_time_tol,
        init_mode=prepared_run.init_mode,
        dim=dim,
        box_size=box_size,
        base_box_np=base_box_np,
        box_volume=box_volume,
        potential_r_cut=potential_r_cut,
        D0=D0,
        shear_rate=shear_rate,
        dump_info=prepared_run.dump_info,
        data_info=prepared_run.data_info,
        confin_path=confin_path,
        planned_steps=planned_steps,
        potential_name=potential_name,
        potential_source=potential_source,
        potential_params=potential_params,
        interaction_neighbor_defaults=interaction_neighbor_defaults,
        collision_neighbor_settings=collision_neighbor_settings,
      )
      params_path = _write_params_json(out_dir, params)
      _CONSOLE.info(
        f'[{prepared_run.spec.label}] Wrote parameters to {params_path}'
      )

      dumpers.append(
        RunDumper(
          out_dir,
          box_size,
          dim,
          dt,
          args.traj_every,
          args.stress_every,
          box_fn=dump_box_fn,
          base_box=base_box_np,
          shear_rate=shear_rate,
          time_offset=shear_t0,
          shear_remap=True,
          unwrap_trajectory=True,
          stress_filenames=_HS_STRESS_FILENAMES,
        )
      )

      state = init_fn(prepared_run.run_key, prepared_run.R0)
      collision_neighbor_api_extra_capacity = _neighbor_api_extra_capacity(
        absolute_extra_capacity=collision_neighbor_extra_capacity,
        n_particles=state.position.shape[0],
        neighbor_format=collision_neighbor_format,
      )
      collision_neighbor = _allocate_probe_sized_neighbor(
        collision_neighbor_fn,
        state.position,
        box=box_t0,
        build_box=collision_build_box,
        extra_capacity=collision_neighbor_api_extra_capacity,
      )
      if not _check_collision_neighbor_status(
        collision_neighbor, f'shear_init ({prepared_run.spec.label})'):
        return
      interaction_neighbor = None
      if use_pair_potential:
        interaction_neighbor = _allocate_probe_sized_neighbor(
          interaction_neighbor_fn,
          state.position,
          box=box_t0,
          build_box=interaction_build_box,
        )
        if not _check_interaction_neighbor_status(
          interaction_neighbor, f'shear_init ({prepared_run.spec.label})'):
          return
      states.append(state)
      collision_neighbors.append(collision_neighbor)
      if use_pair_potential:
        interaction_neighbors.append(interaction_neighbor)

    state = _stack_hs_states(states)
    collision_neighbor = _stack_neighbor_lists(collision_neighbors)
    interaction_neighbor = (
      _stack_neighbor_lists(interaction_neighbors)
      if use_pair_potential
      else None
    )

    if do_traj:
      if do_stress:
        (
          stress_steps,
          _,
          stress_strains,
          stresses_col,
          stresses_virial,
          stresses,
        ) = batched_evaluate_stress(state, interaction_neighbor)
      for run_index, dumper in enumerate(dumpers):
        if do_stress:
          out_stress_times = np.array(
            [int(np.asarray(stress_steps[run_index])) * dt + float(shear_t0)],
            dtype=float,
          )
          out_stress_strains = np.array(
            [float(np.asarray(stress_strains[run_index]))], dtype=float)
          out_stresses_col = np.asarray(
            stresses_col[run_index], dtype=float)[np.newaxis]
          out_stresses_virial = np.asarray(
            stresses_virial[run_index], dtype=float)[np.newaxis]
          out_stresses = np.asarray(stresses[run_index], dtype=float)[np.newaxis]
        else:
          out_stress_times = np.array([], dtype=float)
          out_stress_strains = np.array([], dtype=float)
          out_stresses_col = np.zeros((0, dim, dim), dtype=float)
          out_stresses_virial = np.zeros((0, dim, dim), dtype=float)
          out_stresses = np.zeros((0, dim, dim), dtype=float)
        dumper.dump(
          out_stress_times,
          out_stress_strains,
          out_stresses,
          None,
          np.asarray(state.position[run_index], dtype=float)[np.newaxis],
          traj_steps=np.array([0], dtype=np.int64),
          stress_components={
            'stress_col': out_stresses_col,
            'stress_virial': out_stresses_virial,
          },
        )

    steps_done = 0
    while steps_done < planned_steps:
      state, collision_neighbor, interaction_neighbor = batched_run_one_step(
        state,
        collision_neighbor,
        interaction_neighbor,
      )
      steps_done += 1

      if not _check_batch_nan_positions(state, run_specs, f'shear step {steps_done}'):
        return
      if not _check_batch_collision_loop_status(
        state, run_specs, f'shear step {steps_done}'):
        return
      if not _check_batch_neighbor_status(
        collision_neighbor,
        run_specs,
        f'shear step {steps_done}',
        'Collision',
      ):
        return
      if use_pair_potential and not _check_batch_neighbor_status(
        interaction_neighbor,
        run_specs,
        f'shear step {steps_done}',
        'Interaction',
      ):
        return

      emit_stress = do_stress and (steps_done % args.stress_every == 0)
      emit_traj = do_traj and (steps_done % args.traj_every == 0)
      if emit_stress:
        (
          stress_steps,
          _,
          stress_strains,
          stresses_col,
          stresses_virial,
          stresses,
        ) = batched_evaluate_stress(state, interaction_neighbor)
      else:
        stress_steps = None
        stress_strains = None
        stresses_col = None
        stresses_virial = None
        stresses = None

      if emit_stress or emit_traj:
        step_values = np.asarray(_state_step(state, dt=dt, t0=shear_t0), dtype=np.int64)
        positions_np = np.asarray(state.position, dtype=float)
        for run_index, dumper in enumerate(dumpers):
          if emit_stress:
            out_stress_times = np.array(
              [int(np.asarray(stress_steps[run_index])) * dt + float(shear_t0)],
              dtype=float,
            )
            out_stress_strains = np.array(
              [float(np.asarray(stress_strains[run_index]))], dtype=float)
            out_stresses_col = np.asarray(
              stresses_col[run_index], dtype=float)[np.newaxis]
            out_stresses_virial = np.asarray(
              stresses_virial[run_index], dtype=float)[np.newaxis]
            out_stresses = np.asarray(
              stresses[run_index], dtype=float)[np.newaxis]
          else:
            out_stress_times = None
            out_stress_strains = None
            out_stresses_col = None
            out_stresses_virial = None
            out_stresses = None

          if emit_traj:
            out_traj_steps = np.array([int(step_values[run_index])], dtype=np.int64)
            out_traj_positions = positions_np[run_index][np.newaxis]
          else:
            out_traj_steps = None
            out_traj_positions = None

          dumper.dump(
            out_stress_times,
            out_stress_strains,
            out_stresses,
            None,
            out_traj_positions,
            traj_steps=out_traj_steps,
            stress_components={
              'stress_col': out_stresses_col,
              'stress_virial': out_stresses_virial,
            },
          )

      if args.progress_every > 0 and (steps_done % args.progress_every == 0):
        _CONSOLE.progress(f'Step {min(steps_done, planned_steps)}/{planned_steps}')

    final_steps = np.asarray(_state_step(state, dt=dt, t0=shear_t0), dtype=np.int64)
    for run_index, prepared_run in enumerate(prepared_runs):
      final_step = int(final_steps[run_index])
      final_time = float(final_step * dt + shear_t0)
      final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
      pos_frac = np.mod(np.asarray(state.position[run_index], dtype=float), 1.0)
      pos_real = np.asarray(pos_frac @ final_box.T, dtype=float)

      confout_path = os.path.join(prepared_run.spec.out_dir, 'confout.data')
      write_lammps_data(
        confout_path,
        final_box,
        pos_real,
        comment=(
          'Generated by examples/shear/shear_hard_sphere.py '
          f'(step={final_step})'
        ),
      )
      _CONSOLE.info(
        f'[{prepared_run.spec.label}] Wrote final data snapshot {confout_path}'
      )
  finally:
    for dumper in dumpers:
      dumper.close()

  elapsed_s = time.perf_counter() - wall_start
  total_steps = int(planned_steps)
  _CONSOLE.section('Timing')
  _CONSOLE.info(f'Total wall time: {elapsed_s:.3f} s')
  if total_steps > 0 and elapsed_s > 0.0:
    seconds_per_step = elapsed_s / float(total_steps)
    ptps = (float(n_particles) * float(total_steps) * float(n_runs)) / elapsed_s
    _CONSOLE.info(
      'Time per batch step: '
      f'{seconds_per_step:.6e} s/step ({seconds_per_step * 1e3:.6f} ms/step)'
    )
    _CONSOLE.info(f'Aggregate PTPS: {ptps:.6e} particle-timesteps/s')
  else:
    _CONSOLE.warn('Skipping per-step timing/PTPS (no executed steps).')

  _CONSOLE.success('Done.')


def main():
  """Dispatches to single-run or in-process batch hard-sphere execution."""
  args = parse_args()
  wall_start = time.perf_counter()
  if args.batch_mode:
    _run_batch(args, wall_start)
  else:
    _run_single(args, wall_start)


if __name__ == '__main__':
  main()
