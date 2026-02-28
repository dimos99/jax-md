"""
RPY shear runner with pluggable pair-interaction potentials.
"""

import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import partition
from jax_md import rheo
from jax_md import simulate
from jax_md import smap
from jax_md import space
from jax_md.hydro import rpy

from shear_rpy_cli import build_internal_config
from shear_rpy_cli import parse_args
from shear_rpy_cli import resolve_runtime_settings
from shear_console import get_console
from shear_init import _build_reduced_xy_box_fn
from shear_output import RunDumper
from shear_output import write_lammps_data
from shear_prepare_utils import _build_format_map
from shear_prepare_utils import _build_initial_positions
from shear_prepare_utils import _build_params_payload
from shear_prepare_utils import _derive_system_dynamics
from shear_prepare_utils import _resolve_initial_system
from shear_prepare_utils import _resolve_potential_setup
from shear_prepare_utils import _write_params_json
from shear_runtime_utils import _check_interaction_neighbor_status
from shear_runtime_utils import _check_nan_positions
from shear_runtime_utils import _check_neighbor_status
from shear_runtime_utils import _wrap_neighbor_energy
from shear_time_utils import _predict_xy_remapped_positions_for_next_force
from shear_time_utils import _state_next_time_from_step
from shear_time_utils import _state_step
from shear_time_utils import _state_time_from_step

_CONSOLE = get_console()


def main():
  """Runs one configurable RPY shear trajectory and writes run artifacts."""
  # ============================================================================
  # PREPARATION
  # ============================================================================
  # Parse CLI arguments and capture wall-clock start for end-of-run timing.
  args = parse_args()
  wall_start = time.perf_counter()

  # Resolve runtime settings.
  # Resolve typed runtime settings from internal defaults and CLI values.
  internal = build_internal_config()
  runtime = resolve_runtime_settings(args, internal)
  a = runtime['a']
  kT = runtime['kT']
  viscosity = runtime['viscosity']
  dt = runtime['dt']
  # Keep downstream serialization/logging on a single normalized timestep value.
  args.dt = dt
  mr_iters = runtime['mr_iters']
  tol = runtime['tol']
  xi_override = runtime['xi_override']
  mr_neighbor_format = runtime['mr_neighbor_format']
  mr_dr_threshold = runtime['mr_dr_threshold']
  mr_capacity_multiplier = runtime['mr_capacity_multiplier']
  real_space_mode = runtime['real_space_mode']
  relax_steps = runtime['relax_steps']
  relax_neighbor_format = runtime['relax_neighbor_format']
  relax_neighbor_dr_threshold = runtime['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = runtime['relax_neighbor_capacity_multiplier']

  # Resolve enum-like settings once before building runtime operators.
  format_map = _build_format_map()

  # Log backend/device context once for reproducibility diagnostics.
  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  # Resolve initial conditions.
  # Resolve initialization mode and initial box from CLI inputs.
  dim = 3
  diameter = 2.0 * a
  initial_system = _resolve_initial_system(args, a=a, dim=dim, console=_CONSOLE)
  init_mode = initial_system['init_mode']
  data_info = initial_system['data_info']
  dump_info = initial_system['dump_info']
  n_particles = initial_system['n_particles']
  phi = initial_system['phi']
  base_box = initial_system['base_box']

  # Derive box/shear dynamics.
  # Derive scalar transport and shear quantities from the resolved system.
  dynamics = _derive_system_dynamics(
    base_box=base_box,
    a=a,
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
  shear_vector_schedule = dynamics['shear_vector_schedule']

  _CONSOLE.section('System')
  _CONSOLE.info(f'Equivalent box size L = {box_size:.6f}')
  _CONSOLE.info(f'D0 = {D0:.6e}')
  _CONSOLE.info(f'Shear rate = {shear_rate:.6e}')
  _CONSOLE.info(f'Strain per step = {shear_rate * dt:.6e}')

  # Resolve pair potential and interaction-neighbor defaults.
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

  # Build sheared displacement/shift operators used throughout the run.
  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  displacement_0, shift_0 = space.periodic_general(base_box, fractional_coordinates=True)

  # Build initial particle coordinates.
  # Build the initial particle configuration in fractional coordinates.
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

  # Build force and neighbor operators.
  # Construct force/energy and interaction neighbor-list operators.
  metric_shear = space.canonicalize_displacement_or_metric(displacement)
  if use_pair_potential:
    # Use canonicalized metric with remapped shearing so pair distances match runtime geometry.
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
    energy_fn = _wrap_neighbor_energy(
      energy_fn_neighbor,
      energy_all_pairs_fn=energy_fn_all_pairs,
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
  else:
    def energy_fn(R, **unused_kwargs):
      return jnp.zeros((), dtype=R.dtype)

    interaction_neighbor_fn = None

  # Bound the estimator over the full planned simulation time interval.
  shear_t_bounds = (
    0.0,
    float(dt * float(args.n_steps)),
  )
  # Estimate Ewald/real-space split parameters before integrator construction.
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
  _CONSOLE.section('RPY Parameters')
  _CONSOLE.info(
    'RPY parameters: '
    f'xi={xi:.6f}, rcut={rpy_rcut:.6f}, P={rpy_P}, M={rpy_M}, theta={rpy_theta:.6f}'
  )
  diagnostics = rpy_params.diagnostics
  if diagnostics is not None:
    _CONSOLE.info(
      'Quadrature deformation bound: '
      f'lambda_max={diagnostics.quadrature_lambda_max:.6f} '
      f'(remap={diagnostics.shear_remap})'
    )

  # Configure integrator and optional stress path.
  # Build sheared RPY integrator from the resolved parameters.
  init_fn, apply_fn = simulate.rpy_with_shear(
    (displacement, shift, box_of),
    energy_fn,
    dt=dt,
    kT=kT,
    a=a,
    xi=xi,
    eta=viscosity,
    shear_vector_schedule=shear_vector_schedule,
    t0=shear_t0,
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
      'writing zero pair-interaction stress.'
    )
  planned_steps = int(args.n_steps)

  # This is the main JIT-compiled step function used in the inner loop of the shearing run.
  if use_pair_potential:
    @jax.jit
    def run_one_step(state_in, interaction_neighbor_in):
      # Update interaction neighbors at the next box/time, then integrate one step.
      next_time = _state_next_time_from_step(state_in, dt=dt, t0=shear_t0)
      next_box = box_of(t=next_time)
      pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
        state_in, dt=dt, shear_rate=shear_rate, t0=shear_t0)
      # Predictive update keeps pair-force neighbors aligned with the next force evaluation geometry.
      interaction_neighbor_out = interaction_neighbor_in.update(
        pos_for_neighbor, box=next_box)
      state_out = apply_fn(state_in, interaction_neighbor=interaction_neighbor_out)
      curr_time = _state_time_from_step(state_out, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      # Refresh neighbors at the accepted state so stress/output use current-step neighborhoods.
      interaction_neighbor_out = interaction_neighbor_out.update(
        state_out.integrator_position, box=curr_box)
      return state_out, interaction_neighbor_out
  else:
    @jax.jit
    def _run_one_step_without_pair_potential(state_in):
      return apply_fn(state_in)

    def run_one_step(state_in, interaction_neighbor_in):
      del interaction_neighbor_in
      state_out = _run_one_step_without_pair_potential(state_in)
      return state_out, None

  evaluate_stress = None
  if do_stress:
    if use_pair_potential:
      @jax.jit
      def evaluate_stress(state_in, interaction_neighbor_in):
        # Stress uses the current sheared box to stay consistent with periodic triclinic geometry.
        curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
        curr_box = box_of(t=curr_time)
        stress = stress_fn(
          state_in.integrator_position,
          box=curr_box,
          neighbor=interaction_neighbor_in,
          fractional_coordinates=True,
        )
        strain = shear_rate * curr_time
        return _state_step(state_in, dt=dt, t0=shear_t0), curr_time, strain, stress
    else:
      @jax.jit
      def _evaluate_stress_without_pair_potential(state_in):
        curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
        strain = shear_rate * curr_time
        stress = jnp.zeros((dim, dim), dtype=state_in.integrator_position.dtype)
        return _state_step(state_in, dt=dt, t0=shear_t0), curr_time, strain, stress

      def evaluate_stress(state_in, interaction_neighbor_in):
        del interaction_neighbor_in
        return _evaluate_stress_without_pair_potential(state_in)

  # Prepare output artifacts.
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
        'Generated by examples/shear/shear_rpy.py '
        f'(init_mode={init_mode})'
      ),
    )
    _CONSOLE.info(f'Wrote initial configuration to {confin_path}')

  # Persist full runtime configuration once before launching trajectories.
  # Persist full run metadata once so downstream analysis has complete provenance.
  params = _build_params_payload(
    args=args,
    n_particles=n_particles,
    phi=phi,
    dt=dt,
    a=a,
    kT=kT,
    viscosity=viscosity,
    mr_iters=mr_iters,
    tol=tol,
    xi_override=xi_override,
    mr_neighbor_format=mr_neighbor_format,
    mr_dr_threshold=mr_dr_threshold,
    mr_capacity_multiplier=mr_capacity_multiplier,
    relax_steps=relax_steps,
    relax_neighbor_format=relax_neighbor_format,
    relax_neighbor_dr_threshold=relax_neighbor_dr_threshold,
    relax_neighbor_capacity_multiplier=relax_neighbor_capacity_multiplier,
    init_mode=init_mode,
    dim=dim,
    box_size=box_size,
    base_box_np=base_box_np,
    box_volume=box_volume,
    diameter=diameter,
    potential_r_cut=potential_r_cut,
    D0=D0,
    shear_rate=shear_rate,
    dump_info=dump_info,
    data_info=data_info,
    confin_path=confin_path,
    planned_steps=planned_steps,
    xi=xi,
    rpy_rcut=rpy_rcut,
    rpy_P=rpy_P,
    rpy_M=rpy_M,
    rpy_theta=rpy_theta,
    rpy_lattice_extent=rpy_lattice_extent,
    rpy_params=rpy_params,
    real_space_mode=real_space_mode,
    potential_name=potential_name,
    potential_source=potential_source,
    potential_params=potential_params,
    interaction_neighbor_defaults=interaction_neighbor_defaults,
  )
  params_path = _write_params_json(out_dir, params)
  _CONSOLE.info(f'Wrote parameters to {params_path}')

  # Dumper writes reduced-box snapshots while preserving continuous unwrapped trajectories.
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

  # Run plan summary.
  _CONSOLE.section('Run Plan')
  _CONSOLE.info(
    f'Running one sheared trajectory for {planned_steps} steps '
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
  if init_mode == 'dump':
    _CONSOLE.info(
      f'Dump initialization enabled: starting a fresh run from step 0 '
      f'with positions loaded from {args.init_traj}.'
    )
  elif init_mode == 'data':
    _CONSOLE.info(
      f'Data initialization enabled: starting a fresh run from step 0 '
      f'with positions loaded from {args.init_data}.'
    )



  # ============================================================================
  # SIMULATION
  # ============================================================================
  # Execute timestepping and periodic emissions.
  # Execute one shearing run.
  try: # Ensure dumper is closed to flush buffers even if the run fails.
    state = init_fn(run_key, R0) # Initializes integrator state (positions, forces, and RPY quadrature cache).
    positions_init = state.integrator_position # shape (N, dim) in fractional coordinates
    interaction_neighbor = None
    if use_pair_potential:
      box_t0 = box_of(t=shear_t0)
      # Allocate the initial neighbor list at the first box
      interaction_neighbor = interaction_neighbor_fn.allocate(
        state.integrator_position, box=box_t0)
    if not _check_neighbor_status(state, 'shear_init'):
      return
    if use_pair_potential and not _check_interaction_neighbor_status(interaction_neighbor, 'shear_init'):
      return

    # Write the initial configuration (t=0) before the loop
    # so the trajectory file always starts from the very first frame.
    if do_traj:
      if do_stress:
        stress_step, _, stress_strain, stress = evaluate_stress(state, interaction_neighbor)
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
        np.asarray(positions_init, dtype=float)[np.newaxis],  # shape (1, N, dim)
        traj_steps=np.array([0], dtype=np.int64),
      )

    # Loop over the planned number of steps, emitting stress and trajectory on independent cadences.
    steps_done = 0
    while steps_done < planned_steps:
      state, interaction_neighbor = run_one_step(state, interaction_neighbor)
      steps_done += 1

      if not _check_nan_positions(state, f'shear step {steps_done}'):
        return
      if not _check_neighbor_status(state, f'shear step {steps_done}'):
        return
      if use_pair_potential and not _check_interaction_neighbor_status(
        interaction_neighbor, f'shear step {steps_done}'):
        return

      # Stress and trajectory are emitted on independent cadences.
      emit_stress = do_stress and (steps_done % args.stress_every == 0)
      emit_traj = do_traj and (steps_done % args.traj_every == 0)
      if emit_stress:
        stress_step, _, stress_strain, stress = evaluate_stress(state, interaction_neighbor)
        out_stress_times = np.array(
          [int(np.asarray(stress_step)) * dt + float(shear_t0)], dtype=float)
        out_stress_strains = np.array([float(np.asarray(stress_strain))], dtype=float)
        out_stresses = np.asarray(stress, dtype=float)[np.newaxis]
      else:
        out_stress_times = None
        out_stress_strains = None
        out_stresses = None

      if emit_traj:
        # Use integer-step metadata to avoid float-time drift in long trajectories.
        out_traj_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
        out_traj_steps = np.array([out_traj_step], dtype=np.int64)
        out_traj_positions = np.asarray(state.integrator_position, dtype=float)[np.newaxis]
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

    # Final snapshot uses exact integer step metadata and the corresponding reduced sheared box.
    final_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
    final_time = float(final_step * dt + shear_t0)
    final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
    pos_frac = np.mod(np.asarray(state.integrator_position, dtype=float), 1.0)
    pos_real = np.asarray(pos_frac @ final_box.T, dtype=float)

    confout_path = os.path.join(out_dir, 'confout.data')
    write_lammps_data(
      confout_path,
      final_box,
      pos_real,
      comment=(
        'Generated by examples/shear/shear_rpy.py '
        f'(step={final_step})'
      ),
    )
    _CONSOLE.info(f'Wrote final data snapshot {confout_path}')
  finally:
    dumper.close()

  # ============================================================================
  # OUTPUT
  # ============================================================================
  # Emit wall-clock timing diagnostics.
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
