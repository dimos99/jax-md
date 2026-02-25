"""
RPY shear runner with pluggable pair-interaction potentials.
"""

import json
import math
import os
import shutil
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import numpy as np

from jax_md import partition
from jax_md import rheo
from jax_md import simulate
from jax_md import smap
from jax_md import space
from jax_md.hydro import rpy

from rpy_cli import build_internal_config
from rpy_cli import parse_args
from rpy_console import get_console
from rpy_init import _box_size_from_phi
from rpy_init import _build_reduced_xy_box_fn
from rpy_init import _load_initial_state_from_data
from rpy_init import _load_initial_state_from_dump
from rpy_init import _min_pair_distance
from rpy_init import _relax_positions
from rpy_output import RunDumper
from rpy_output import _serialize_rpy_parameter_estimate
from rpy_output import _to_jsonable
from rpy_output import write_lammps_data
from rpy_potential import _resolve_potential

_CONSOLE = get_console()

def _wrap_neighbor_energy(energy_neighbor_fn, energy_all_pairs_fn=None):
  def _wrapped(R, interaction_neighbor=None, **kwargs):
    if interaction_neighbor is None:
      if energy_all_pairs_fn is not None:
        return energy_all_pairs_fn(R, **kwargs)
      raise ValueError('Missing required interaction_neighbor kwarg for pair-interaction force evaluation.')
    return energy_neighbor_fn(R, neighbor=interaction_neighbor, **kwargs)
  return _wrapped


def _neighbor_list_health(neighbors, stage: str, label: str) -> bool:
  if neighbors is None:
    _CONSOLE.error(f'Missing {label} in stage={stage}.')
    return False
  overflow = np.asarray(neighbors.did_buffer_overflow)
  cell_small = np.asarray(neighbors.cell_size_too_small)
  malformed = np.asarray(neighbors.malformed_box)
  if np.any(overflow):
    _CONSOLE.error(
      f'{label} overflow in stage={stage}. '
      'Try increasing capacity_multiplier or dr_threshold.'
    )
    return False
  if np.any(cell_small):
    _CONSOLE.error(
      f'{label} cell size too small in stage={stage}.'
    )
    return False
  if np.any(malformed):
    _CONSOLE.error(
      f'{label} malformed box in stage={stage}.'
    )
    return False
  return True

def _time_from_step(step, *, dt: float, t0: float, dtype):
  step_arr = jnp.asarray(step, dtype=jnp.int32)
  dt_arr = jnp.asarray(dt, dtype=dtype)
  t0_arr = jnp.asarray(t0, dtype=dtype)
  return t0_arr + step_arr.astype(dtype) * dt_arr


def _state_step(state, *, dt: float, t0: float):
  """Returns an integer step, deriving it from time for legacy states."""
  state_step = getattr(state, 'step', None)
  if state_step is not None:
    return jnp.asarray(state_step, dtype=jnp.int32)
  time_arr = jnp.asarray(state.time)
  dt_arr = jnp.asarray(dt, dtype=time_arr.dtype)
  t0_arr = jnp.asarray(t0, dtype=time_arr.dtype)
  step_float = (time_arr - t0_arr) / dt_arr
  return jnp.asarray(jnp.floor(step_float + 0.5), dtype=jnp.int32)


def _state_time_from_step(state, *, dt: float, t0: float):
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


def _check_neighbor_status(state, stage: str) -> bool:
  """Returns True if the neighbor state is healthy; otherwise prints and fails."""
  neighbors = state.rpy_state.real.neighbors
  return _neighbor_list_health(
    neighbors,
    stage=stage,
    label='RPY real-space neighbor list',
  )


def _check_interaction_neighbor_status(
  interaction_neighbor,
  stage: str,
) -> bool:
  return _neighbor_list_health(
    interaction_neighbor,
    stage=stage,
    label='pair-interaction neighbor list',
  )


def _check_nan_positions(state, stage: str) -> bool:
  """Returns True if positions are finite; otherwise prints and fails."""
  has_nan = bool(np.asarray(jnp.any(jnp.isnan(state.mobility_position))))
  if has_nan:
    _CONSOLE.error(f'NaN detected in mobility positions during {stage}.')
    return False
  return True


def _build_thermalize_runner(apply_fn_eq, steps: int, base_box: jnp.ndarray):
  """Returns a thermalization runner for exactly `steps` steps."""
  if steps <= 0:
    return None

  def _single(state, interaction_neighbor):
    def _step(_, carry):
      s, pn = carry
      pn = pn.update(s.mobility_position, box=base_box)
      s = apply_fn_eq(s, interaction_neighbor=pn)
      return s, pn
    return lax.fori_loop(0, steps, _step, (state, interaction_neighbor))

  return jax.jit(_single)


def main():
  args = parse_args()
  wall_start = time.perf_counter()

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

  # Resolve enum-like CLI/internal settings once, before runtime setup.
  format_map = {
    'dense': partition.NeighborListFormat.Dense,
    'sparse': partition.NeighborListFormat.Sparse,
    'ordered': partition.NeighborListFormat.OrderedSparse,
  }

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  # Build the initial state either from dump metadata or random+relax.
  dim = 3
  diameter = 2.0 * a
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
      _CONSOLE.warn(
        f'data-derived phi={phi:.6g} exceeds hard-sphere close packing '
        '(~0.74). This can lead to severe overlaps and NaNs in the RPY mobility.'
      )
    base_box = jnp.asarray(data_info['box_matrix'])
    _CONSOLE.info(
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
      _CONSOLE.warn(
        f'dump-derived phi={phi:.6g} exceeds hard-sphere close packing '
        '(~0.74). This can lead to severe overlaps and NaNs in the RPY mobility.'
      )
    base_box = jnp.asarray(dump_info['box_matrix'])
    _CONSOLE.info(
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
  shear_t0 = 0.0
  shear_schedule = {'xy': lambda t: shear_rate * t}
  shear_vector_schedule = lambda t: (shear_rate * t, 0.0, 0.0)

  _CONSOLE.section('System')
  _CONSOLE.info(f'Equivalent box size L = {box_size:.6f}')
  _CONSOLE.info(f'D0 = {D0:.6e}')
  _CONSOLE.info(f'Shear rate = {shear_rate:.6e}')
  _CONSOLE.info(f'Strain per step = {shear_rate * dt:.6e}')

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
  # Keep overlap-repulsion timestep consistent with the integrator timestep.
  # If the selected potential exposes `repulsion_dt`, bind it to --dt.
  if 'repulsion_dt' in potential_params:
    potential_params['repulsion_dt'] = float(dt)
  potential_r_cut = float(potential_cfg['r_cut'])
  potential_name = str(potential_cfg['name'])
  potential_source = str(potential_cfg['source'])

  _CONSOLE.section('Potential')
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

  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  displacement_0, shift_0 = space.periodic_general(base_box, fractional_coordinates=True)

  # Build the initial particle configuration in fractional coordinates.
  key = random.PRNGKey(args.seed)
  key, init_key, thermalize_key, run_key = random.split(key, 4)
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
  _CONSOLE.info(f'Post-relax minimum pair distance: {min_dist:.6f}')

  # Construct force/energy and interaction neighbor-list operators.
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

  # Build equilibrium/sheared RPY integrators from the resolved parameters.
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

  do_stress = args.stress_every > 0
  do_traj = args.traj_every > 0
  stress_fn = None
  if do_stress:
    stress_fn = rheo.make_pairwise_stress_fn(
      pair_potential_fn,
      **potential_params,
    )

  # Choose chunk/sampling cadence based on enabled outputs.
  if do_stress and do_traj:
    scan_interval = math.gcd(args.stress_every, args.traj_every)
    sample_period = math.lcm(args.stress_every, args.traj_every)
  elif do_stress:
    scan_interval = args.stress_every
    sample_period = args.stress_every
  elif do_traj:
    scan_interval = args.traj_every
    sample_period = args.traj_every
  else:
    # No sampled outputs; advance one full buffer per scan call.
    scan_interval = buffer_steps_default
    sample_period = 1

  buffer_steps = buffer_steps_default
  if sample_period > 1 and (buffer_steps % sample_period != 0):
    buffer_steps = ((buffer_steps // sample_period) + 1) * sample_period
    _CONSOLE.warn(
      f'Adjusted buffer_steps to {buffer_steps} so it is divisible by sample period '
      f'{sample_period}.'
    )
  if args.n_steps % buffer_steps != 0:
    _CONSOLE.warn(
      f'n_steps={args.n_steps} is not a multiple of buffer_steps={buffer_steps}. '
      f'Running a final tail chunk to end exactly at n_steps.'
    )
  planned_steps = args.n_steps

  thermalize_chunk_steps = buffer_steps
  if args.thermalize_steps > 0:
    thermalize_chunk_steps = max(1, min(buffer_steps, args.thermalize_steps))

  steps_per_scan = scan_interval
  scans_per_buffer = buffer_steps // steps_per_scan
  stress_stride = (args.stress_every // scan_interval) if do_stress else None
  traj_stride = (args.traj_every // scan_interval) if do_traj else None
  stress_offset = (stress_stride - 1) if do_stress else None
  traj_offset = (traj_stride - 1) if do_traj else None

  def _run_chunk_single(carry_in):
    state_in, interaction_neighbor_in = carry_in

    def _inner(_, inner_carry):
      s, pn = inner_carry
      next_time = _state_next_time_from_step(s, dt=dt, t0=shear_t0)
      next_box = box_of(t=next_time)
      pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
        s, dt=dt, shear_rate=shear_rate, t0=shear_t0)
      pn = pn.update(pos_for_neighbor, box=next_box)
      s = apply_fn(s, interaction_neighbor=pn)
      return s, pn

    if not do_stress and not do_traj:
      state_out, interaction_neighbor_out = lax.fori_loop(
        0, buffer_steps, _inner, (state_in, interaction_neighbor_in))
      curr_time = _state_time_from_step(state_out, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      interaction_neighbor_out = interaction_neighbor_out.update(
        state_out.mobility_position, box=curr_box)
      return (state_out, interaction_neighbor_out), ()

    def _scan_body(carry, _):
      state, interaction_neighbor = carry

      state, interaction_neighbor = lax.fori_loop(
        0, steps_per_scan, _inner, (state, interaction_neighbor))
      curr_time = _state_time_from_step(state, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      interaction_neighbor = interaction_neighbor.update(state.mobility_position, box=curr_box)
      if do_stress:
        stress = stress_fn(
          state.mobility_position,
          box=curr_box,
          neighbor=interaction_neighbor,
          fractional_coordinates=True,
        )
        strain = shear_rate * curr_time
        step = _state_step(state, dt=dt, t0=shear_t0)
        if do_traj:
          out = (step, strain, stress, state.mobility_position)
        else:
          out = (step, strain, stress)
      else:
        out = (_state_step(state, dt=dt, t0=shear_t0), state.mobility_position)
      return (state, interaction_neighbor), out

    (state_out, interaction_neighbor_out), scan_out = lax.scan(
      _scan_body,
      (state_in, interaction_neighbor_in),
      None,
      length=scans_per_buffer,
    )

    if do_stress and do_traj:
      steps, strains, stresses, positions = scan_out
      stress_steps = steps[stress_offset::stress_stride]
      stress_strains = strains[stress_offset::stress_stride]
      stress_out = stresses[stress_offset::stress_stride]
      traj_steps = steps[traj_offset::traj_stride]
      traj_positions = positions[traj_offset::traj_stride]
      return (
        (state_out, interaction_neighbor_out),
        (stress_steps, stress_strains, stress_out, traj_steps, traj_positions),
      )
    if do_stress:
      steps, strains, stresses = scan_out
      stress_steps = steps[stress_offset::stress_stride]
      stress_strains = strains[stress_offset::stress_stride]
      stress_out = stresses[stress_offset::stress_stride]
      return (state_out, interaction_neighbor_out), (stress_steps, stress_strains, stress_out)
    steps, positions = scan_out
    traj_steps = steps[traj_offset::traj_stride]
    traj_positions = positions[traj_offset::traj_stride]
    return (state_out, interaction_neighbor_out), (traj_steps, traj_positions)

  run_chunk = jax.jit(_run_chunk_single)

  @jax.jit
  def run_one_step(state_in, interaction_neighbor_in):
    next_time = _state_next_time_from_step(state_in, dt=dt, t0=shear_t0)
    next_box = box_of(t=next_time)
    pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
      state_in, dt=dt, shear_rate=shear_rate, t0=shear_t0)
    interaction_neighbor_out = interaction_neighbor_in.update(
      pos_for_neighbor, box=next_box)
    state_out = apply_fn(state_in, interaction_neighbor=interaction_neighbor_out)
    curr_time = _state_time_from_step(state_out, dt=dt, t0=shear_t0)
    curr_box = box_of(t=curr_time)
    interaction_neighbor_out = interaction_neighbor_out.update(
      state_out.mobility_position, box=curr_box)
    return state_out, interaction_neighbor_out

  evaluate_stress = None
  if do_stress:
    @jax.jit
    def evaluate_stress(state_in, interaction_neighbor_in):
      curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
      curr_box = box_of(t=curr_time)
      stress = stress_fn(
        state_in.mobility_position,
        box=curr_box,
        neighbor=interaction_neighbor_in,
        fractional_coordinates=True,
      )
      strain = shear_rate * curr_time
      return _state_step(state_in, dt=dt, t0=shear_t0), curr_time, strain, stress

  thermalize_runner_cache = {}

  def _get_thermalize_runner(step_count: int):
    runner = thermalize_runner_cache.get(step_count, None)
    if runner is None:
      runner = _build_thermalize_runner(apply_fn_eq, step_count, base_box)
      thermalize_runner_cache[step_count] = runner
    return runner

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
        'Generated by examples/rpy_shear/rpy_shear.py '
        f'(init_mode={init_mode})'
      ),
    )
    _CONSOLE.info(f'Wrote initial configuration to {confin_path}')

  # Persist full runtime configuration once before launching trajectories.
  params = {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi,
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
    f'Running one sheared trajectory for {planned_steps} steps '
    f'(requested {args.n_steps}).'
  )
  if args.thermalize_steps > 0:
    _CONSOLE.info(
      'Thermalization execution chunk: '
      f'{thermalize_chunk_steps} step(s) per JAX call.'
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

  # Execute one run: optional thermalization followed by production.
  target_step = int(args.n_steps)
  try:
    state_eq = init_fn_eq(thermalize_key, R0)
    interaction_neighbor_eq = interaction_neighbor_fn_0.allocate(
      state_eq.mobility_position, box=base_box)
    if not _check_neighbor_status(state_eq, 'equilibrium_init'):
      return
    if not _check_interaction_neighbor_status(interaction_neighbor_eq, 'equilibrium_init'):
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
        if not _check_neighbor_status(state_eq, f'thermalization step {therm_done}'):
          return
        if not _check_interaction_neighbor_status(
          interaction_neighbor_eq, f'thermalization step {therm_done}'):
          return
        while (
          next_progress_mark is not None
          and therm_done >= next_progress_mark
          and next_progress_mark <= args.thermalize_steps
        ):
          _CONSOLE.progress(f'Thermalize {next_progress_mark}/{args.thermalize_steps}')
          next_progress_mark += args.progress_every
    else:
      _CONSOLE.info('Skipping thermalization (thermalize_steps=0).')

    positions_init = state_eq.mobility_position
    state = init_fn(run_key, positions_init)
    box_t0 = box_of(t=shear_t0)
    interaction_neighbor = interaction_neighbor_fn.allocate(state.mobility_position, box=box_t0)
    if not _check_neighbor_status(state, 'shear_init'):
      return
    if not _check_interaction_neighbor_status(interaction_neighbor, 'shear_init'):
      return

    # Write the initial configuration (t=0) before the loop
    # so the trajectory file always starts from the very first frame.
    if traj_stride is not None:
      dumper.dump(
        np.array([], dtype=float),        # no stress at t=0
        np.array([], dtype=float),
        np.zeros((0, dim, dim), dtype=float),
        None,
        np.asarray(positions_init, dtype=float)[np.newaxis],  # shape (1, N, dim)
        traj_steps=np.array([0], dtype=np.int64),
      )

    steps_done = 0
    while steps_done + buffer_steps <= planned_steps:
      if do_stress and do_traj:
        (state, interaction_neighbor), (
          stress_steps,
          stress_strains,
          stresses,
          traj_steps,
          traj_positions,
        ) = run_chunk((state, interaction_neighbor))
      elif do_stress:
        (state, interaction_neighbor), (stress_steps, stress_strains, stresses) = run_chunk(
          (state, interaction_neighbor))
        traj_steps = None
        traj_positions = None
      elif do_traj:
        (state, interaction_neighbor), (traj_steps, traj_positions) = run_chunk(
          (state, interaction_neighbor))
        stress_steps = None
        stress_strains = None
        stresses = None
      else:
        (state, interaction_neighbor), _ = run_chunk((state, interaction_neighbor))
        stress_steps = None
        stress_strains = None
        stresses = None
        traj_steps = None
        traj_positions = None

      if not _check_nan_positions(state, f'shear step {steps_done + buffer_steps}'):
        return
      if not _check_neighbor_status(state, f'shear step {steps_done + buffer_steps}'):
        return
      if not _check_interaction_neighbor_status(
        interaction_neighbor, f'shear step {steps_done + buffer_steps}'):
        return

      stress_steps_np = np.asarray(stress_steps, dtype=np.int64) if do_stress else None
      stress_strains_np = np.asarray(stress_strains) if do_stress else None
      stresses_np = np.asarray(stresses) if do_stress else None
      traj_steps_np = np.asarray(traj_steps, dtype=np.int64) if do_traj else None
      traj_positions_np = np.asarray(traj_positions) if do_traj else None

      if do_stress or do_traj:
        if do_stress:
          stress_mask = stress_steps_np <= target_step
          out_stress_steps = stress_steps_np[stress_mask]
          out_stress_times = out_stress_steps.astype(float) * dt + float(shear_t0)
          out_stress_strains = stress_strains_np[stress_mask]
          out_stresses = stresses_np[stress_mask]
        else:
          out_stress_times = None
          out_stress_strains = None
          out_stresses = None

        if do_traj:
          traj_mask = traj_steps_np <= target_step
          out_traj_steps = traj_steps_np[traj_mask]
          out_traj_positions = traj_positions_np[traj_mask]
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
        )

      steps_done += buffer_steps
      if args.progress_every > 0 and (steps_done % args.progress_every == 0):
        _CONSOLE.progress(f'Step {min(steps_done, planned_steps)}/{planned_steps}')

    tail_steps = planned_steps - steps_done
    while tail_steps > 0:
      state, interaction_neighbor = run_one_step(state, interaction_neighbor)
      steps_done += 1
      tail_steps -= 1

      if not _check_nan_positions(state, f'shear step {steps_done}'):
        return
      if not _check_neighbor_status(state, f'shear step {steps_done}'):
        return
      if not _check_interaction_neighbor_status(interaction_neighbor, f'shear step {steps_done}'):
        return

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
        out_traj_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
        out_traj_steps = np.array([out_traj_step], dtype=np.int64)
        out_traj_positions = np.asarray(state.mobility_position, dtype=float)[np.newaxis]
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

    final_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
    final_time = float(final_step * dt + shear_t0)
    final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
    pos_frac = np.mod(np.asarray(state.mobility_position, dtype=float), 1.0)
    pos_real = np.asarray(pos_frac @ final_box.T, dtype=float)

    confout_path = os.path.join(out_dir, 'confout.data')
    write_lammps_data(
      confout_path,
      final_box,
      pos_real,
      comment=(
        'Generated by examples/rpy_shear/rpy_shear.py '
        f'(step={final_step})'
      ),
    )
    _CONSOLE.info(f'Wrote final data snapshot {confout_path}')
  finally:
    dumper.close()

  elapsed_s = time.perf_counter() - wall_start
  total_steps = int(args.thermalize_steps + planned_steps)
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
