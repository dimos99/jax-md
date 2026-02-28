"""Hard-sphere shear runner with `simulate.brownian_hard_sphere` integrator."""

import os
import shutil
import time

from jax import jit
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import partition
from jax_md import simulate
from jax_md import smap
from jax_md import space

from shear_hard_sphere_cli import build_internal_config
from shear_hard_sphere_cli import parse_args
from shear_hard_sphere_cli import resolve_runtime_settings
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
from shear_runtime_utils import _check_interaction_neighbor_status

_CONSOLE = get_console()


def _state_step_from_time(state, *, dt: float, t0: float):
  """Returns nearest integer step from state time."""
  time_arr = jnp.asarray(state.time)
  dt_arr = jnp.asarray(dt, dtype=time_arr.dtype)
  t0_arr = jnp.asarray(t0, dtype=time_arr.dtype)
  step_float = (time_arr - t0_arr) / dt_arr
  return jnp.asarray(jnp.floor(step_float + 0.5), dtype=jnp.int32)


def _predict_xy_remapped_positions_for_next_force(state, *, dt: float, shear_rate: float):
  """Predicts fractional positions after any pending xy shear remap."""
  gamma_prev = jnp.asarray(shear_rate) * jnp.asarray(state.time)
  gamma_next = jnp.asarray(shear_rate) * (jnp.asarray(state.time) + jnp.asarray(dt))
  m_prev = jnp.floor(gamma_prev + 0.5)
  m_next = jnp.floor(gamma_next + 0.5)
  dm = (m_next - m_prev).astype(jnp.int32)

  def _apply(R):
    dm_cast = jnp.asarray(dm, dtype=R.dtype)
    x_new = jnp.mod(R[:, 0] + dm_cast * R[:, 1], 1.0)
    return R.at[:, 0].set(x_new)

  return lax.cond(jnp.not_equal(dm, 0), _apply, lambda R: R, state.position)


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


def _build_hs_params_payload(
  *,
  args,
  n_particles: int,
  phi: float,
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
):
  """Builds a `params.json` payload for hard-sphere shear runs."""
  return {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi,
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
  }


def main():
  """Runs one sheared hard-sphere trajectory and writes run artifacts."""
  # ============================================================================
  # PREPARATION
  # ============================================================================
  args = parse_args()
  wall_start = time.perf_counter()

  internal = build_internal_config()
  runtime = resolve_runtime_settings(args, internal)
  hydro_radius = (
    float(args.hydro_radius)
    if args.hydro_radius is not None
    else float(runtime['a'])
  )
  kT = runtime['kT']
  viscosity = runtime['viscosity']
  dt = runtime['dt']
  relax_steps = runtime['relax_steps']
  relax_neighbor_format = runtime['relax_neighbor_format']
  relax_neighbor_dr_threshold = runtime['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = runtime['relax_neighbor_capacity_multiplier']

  # Hard-sphere integrator controls kept explicit in the script.
  hs_core_radius = (
    float(args.hs_core_radius)
    if args.hs_core_radius is not None
    else float(hydro_radius)
  )
  diameter = 2.0 * hs_core_radius
  max_collision_loops = (
    int(args.max_collision_loops)
    if args.max_collision_loops is not None
    else int(1e7)
  )
  event_time_tol = None

  format_map = _build_format_map()

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  dim = 3
  initial_system = _resolve_initial_system(
    args, a=hs_core_radius, dim=dim, console=_CONSOLE)
  init_mode = initial_system['init_mode']
  data_info = initial_system['data_info']
  dump_info = initial_system['dump_info']
  n_particles = initial_system['n_particles']
  phi = initial_system['phi']
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

  # kT * mobility = D0 for isolated spheres.
  mobility = D0 / kT

  _CONSOLE.section('System')
  _CONSOLE.info(f'Equivalent box size L = {box_size:.6f}')
  _CONSOLE.info(f'D0 = {D0:.6e}')
  _CONSOLE.info(f'Mobility = {mobility:.6e}')
  _CONSOLE.info(f'Hydrodynamic radius a = {hydro_radius:.6f}')
  _CONSOLE.info(f'Hard-sphere core radius = {hs_core_radius:.6f}')
  _CONSOLE.info(f'Hard-sphere diameter = {diameter:.6f}')
  _CONSOLE.info(f'Max collision loops/step = {max_collision_loops}')
  _CONSOLE.info(f'Shear rate = {shear_rate:.6e}')
  _CONSOLE.info(f'Strain per step = {shear_rate * dt:.6e}')

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
    diameter=diameter,
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
  _CONSOLE.info(f'Post-relax minimum pair distance: {min_dist:.6f}')

  metric_shear = space.canonicalize_displacement_or_metric(displacement)
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

    def energy_fn(R, neighbor=None, **kwargs):
      kwargs = dict(kwargs)
      kwargs.pop('neighbor', None)
      if neighbor is None:
        return energy_fn_all_pairs(R, **kwargs)
      return energy_fn_neighbor(R, neighbor=neighbor, **kwargs)

    interaction_neighbor_fn = partition.neighbor_list(
      metric_shear,
      base_box,
      r_cutoff=potential_r_cut,
      dr_threshold=interaction_neighbor_dr_threshold,
      capacity_multiplier=interaction_neighbor_capacity_multiplier,
      fractional_coordinates=True,
      format=interaction_neighbor_format,
    )
  else:
    def energy_fn(R, **unused_kwargs):
      return jnp.zeros((), dtype=R.dtype)

    interaction_neighbor_fn = None

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

  do_stress = args.stress_every > 0
  do_traj = args.traj_every > 0
  planned_steps = int(args.n_steps)

  if use_pair_potential:
    @jit
    def run_one_step(state_in, interaction_neighbor_in):
      next_time = jnp.asarray(state_in.time) + jnp.asarray(dt, dtype=state_in.time.dtype)
      next_box = box_of(t=next_time)
      pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
        state_in, dt=dt, shear_rate=shear_rate)
      interaction_neighbor_out = interaction_neighbor_in.update(
        pos_for_neighbor, box=next_box)
      state_out = apply_fn(state_in, neighbor=interaction_neighbor_out)
      curr_box = box_of(t=state_out.time)
      interaction_neighbor_out = interaction_neighbor_out.update(
        state_out.position, box=curr_box)
      return state_out, interaction_neighbor_out
  else:
    @jit
    def _run_one_step_without_pair_potential(state_in):
      return apply_fn(state_in)

    def run_one_step(state_in, interaction_neighbor_in):
      del interaction_neighbor_in
      state_out = _run_one_step_without_pair_potential(state_in)
      return state_out, None

  @jit
  def evaluate_stress(state_in):
    step = _state_step_from_time(state_in, dt=dt, t0=shear_t0)
    curr_time = jnp.asarray(state_in.time)
    strain = jnp.asarray(shear_rate, dtype=curr_time.dtype) * curr_time
    return step, curr_time, strain, state_in.stress

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
    phi=phi,
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
  )

  _CONSOLE.section('Run Plan')
  _CONSOLE.info(
    f'Running one sheared hard-sphere trajectory for {planned_steps} steps '
    f'(requested {args.n_steps}).'
  )
  if do_stress and do_traj:
    _CONSOLE.info(f'Outputs: stress.dat + traj.dump in {out_dir}')
  elif do_stress:
    _CONSOLE.info(f'Outputs: stress.dat in {out_dir}')
  elif do_traj:
    _CONSOLE.info(f'Outputs: traj.dump in {out_dir}')
  else:
    _CONSOLE.info('Outputs: none (both --stress_every and --traj_every are 0).')

  # ============================================================================
  # SIMULATION
  # ============================================================================
  try:
    state = init_fn(run_key, R0)
    interaction_neighbor = None
    if use_pair_potential:
      box_t0 = box_of(t=shear_t0)
      interaction_neighbor = interaction_neighbor_fn.allocate(state.position, box=box_t0)
      if not _check_interaction_neighbor_status(interaction_neighbor, 'shear_init'):
        return

    if do_traj:
      if do_stress:
        stress_step, _, stress_strain, stress = evaluate_stress(state)
        out_stress_times = np.array(
          [int(np.asarray(stress_step)) * dt + float(shear_t0)], dtype=float)
        out_stress_strains = np.array([float(np.asarray(stress_strain))], dtype=float)
        out_stresses = np.asarray(stress, dtype=float)[np.newaxis]
      else:
        out_stress_times = np.array([], dtype=float)
        out_stress_strains = np.array([], dtype=float)
        out_stresses = np.zeros((0, dim, dim), dtype=float)
      dumper.dump(
        out_stress_times,
        out_stress_strains,
        out_stresses,
        None,
        np.asarray(state.position, dtype=float)[np.newaxis],
        traj_steps=np.array([0], dtype=np.int64),
      )

    steps_done = 0
    while steps_done < planned_steps:
      state, interaction_neighbor = run_one_step(state, interaction_neighbor)
      steps_done += 1

      if not _check_nan_positions(state, f'shear step {steps_done}'):
        return
      if not _check_collision_loop_status(state, f'shear step {steps_done}'):
        return
      if use_pair_potential and not _check_interaction_neighbor_status(
        interaction_neighbor, f'shear step {steps_done}'):
        return

      emit_stress = do_stress and (steps_done % args.stress_every == 0)
      emit_traj = do_traj and (steps_done % args.traj_every == 0)
      if emit_stress:
        stress_step, _, stress_strain, stress = evaluate_stress(state)
        out_stress_times = np.array(
          [int(np.asarray(stress_step)) * dt + float(shear_t0)], dtype=float)
        out_stress_strains = np.array([float(np.asarray(stress_strain))], dtype=float)
        out_stresses = np.asarray(stress, dtype=float)[np.newaxis]
      else:
        out_stress_times = None
        out_stress_strains = None
        out_stresses = None

      if emit_traj:
        out_traj_step = int(np.asarray(_state_step_from_time(state, dt=dt, t0=shear_t0)))
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
        )

      if args.progress_every > 0 and (steps_done % args.progress_every == 0):
        _CONSOLE.progress(f'Step {min(steps_done, planned_steps)}/{planned_steps}')

    final_step = int(np.asarray(_state_step_from_time(state, dt=dt, t0=shear_t0)))
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


if __name__ == '__main__':
  main()
