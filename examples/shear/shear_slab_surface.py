"""Minimal slab-constrained RPY shear example.

This script keeps standard 3D RPY hydrodynamics but constrains particle motion
to a single z plane, producing an effective 2D surface dynamics in a 3D slab.
"""

import argparse
import json
import math
import os

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from jax_md import partition
from jax_md import simulate
from jax_md import smap
from jax_md import space
from jax_md.hydro import rpy

from shear_console import get_console
from shear_init import _build_reduced_xy_box_fn
from shear_init import _min_pair_distance
from shear_init import _relax_positions
from shear_output import RunDumper
from shear_output import _to_jsonable
from shear_output import write_lammps_data
from shear_shared import parse_int_like
from shear_shared import predict_xy_remapped_positions_for_next_force
from shear_shared import wrap_neighbor_energy

_CONSOLE = get_console()


def _parse_int_like(value: str) -> int:
  """Parses integer-like CLI values, including scientific notation."""
  return parse_int_like(value)


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description='RPY shear on a slab-constrained 2D surface (single run).'
  )
  parser.add_argument('--n_particles', type=_parse_int_like, required=True)
  parser.add_argument('--phi', type=float, required=True,
                      help='In-plane area fraction used to size Lx=Ly.')
  parser.add_argument('--dt', type=float, required=True)
  parser.add_argument('--out_dir', type=str, required=True)
  parser.add_argument('--n_steps', type=_parse_int_like, default=30000)
  parser.add_argument('--peclet', type=float, default=0.0)
  parser.add_argument('--seed', type=_parse_int_like, default=42)
  parser.add_argument('--traj_every', type=_parse_int_like, default=100)
  parser.add_argument('--progress_every', type=_parse_int_like, default=1000)
  parser.add_argument('--xi', type=float, default=0.5,
                      help='RPY xi override passed to parameter estimator.')
  parser.add_argument('--tol', type=float, default=1e-4,
                      help='Target tolerance used by the RPY parameter estimator.')
  parser.add_argument('--slab_height_factor', type=float, default=4.0,
                      help='Lz = slab_height_factor * Lxy.')
  parser.add_argument('--z0_frac', type=float, default=0.5,
                      help='Fixed fractional z coordinate for all particles.')
  parser.add_argument('--relax_steps', type=_parse_int_like, default=250)

  args = parser.parse_args(argv)
  if args.n_particles <= 1:
    raise ValueError('n_particles must be > 1.')
  if not (0.0 < float(args.phi) <= 1.0):
    raise ValueError('phi must be in (0, 1].')
  if float(args.dt) <= 0.0:
    raise ValueError('dt must be > 0.')
  if args.n_steps <= 0:
    raise ValueError('n_steps must be > 0.')
  if args.peclet < 0.0:
    raise ValueError('peclet must be >= 0.')
  if args.traj_every <= 0:
    raise ValueError('traj_every must be > 0.')
  if args.progress_every < 0:
    raise ValueError('progress_every must be >= 0.')
  if float(args.xi) <= 0.0:
    raise ValueError('xi must be > 0.')
  if float(args.tol) <= 0.0:
    raise ValueError('tol must be > 0.')
  if float(args.slab_height_factor) <= 0.0:
    raise ValueError('slab_height_factor must be > 0.')
  if not (0.0 <= float(args.z0_frac) < 1.0):
    raise ValueError('z0_frac must be in [0, 1).')
  if args.relax_steps < 0:
    raise ValueError('relax_steps must be >= 0.')
  return args


def _surface_shift(base_shift_fn):
  """Wraps a shift function so the displacement has no z component."""
  def _shift(R, dR, **kwargs):
    dR = dR.at[:, 2].set(0.0)
    return base_shift_fn(R, dR, **kwargs)
  return _shift


def _varga_repulsive_pair(
  dr,
  particle_radius,
  viscosity,
  repulsion_dt,
  r_min=1e-6,
  **unused_kwargs,
):
  """Varga et al. overlap repulsion (repulsive-only form)."""
  a = particle_radius
  diameter = 2.0 * a
  positive = dr > 0.0
  safe_r = jnp.maximum(dr, r_min)
  prefactor = 16.0 * jnp.pi * viscosity * a ** 2 / repulsion_dt
  return jnp.where(
    (dr < diameter) & positive,
    prefactor * (diameter * jnp.log(diameter / safe_r) + safe_r - diameter),
    0.0,
  )


def _wrap_neighbor_energy(energy_neighbor_fn, energy_all_pairs_fn=None):
  return wrap_neighbor_energy(
    energy_neighbor_fn,
    energy_all_pairs_fn=energy_all_pairs_fn,
    missing_neighbor_error='Missing interaction_neighbor for pair-interaction force evaluation.',
  )


def _check_neighbor_health(neighbors, stage: str) -> None:
  if neighbors is None:
    raise RuntimeError(f'Missing interaction neighbor list at stage={stage}.')
  overflow = np.asarray(neighbors.did_buffer_overflow)
  cell_small = np.asarray(neighbors.cell_size_too_small)
  malformed = np.asarray(neighbors.malformed_box)
  if np.any(overflow):
    raise RuntimeError(
      f'Interaction neighbor overflow at stage={stage}; '
      'increase capacity_multiplier or dr_threshold.'
    )
  if np.any(cell_small):
    raise RuntimeError(
      f'Interaction neighbor cell size too small at stage={stage}.'
    )
  if np.any(malformed):
    raise RuntimeError(
      f'Interaction neighbor malformed box at stage={stage}.'
    )


def _predict_xy_remapped_positions_for_next_force(state, dt: float, shear_rate: float):
  gamma_prev = shear_rate * jnp.asarray(state.time)
  gamma_next = shear_rate * (jnp.asarray(state.time) + jnp.asarray(dt))
  return predict_xy_remapped_positions_for_next_force(
    state.integrator_position,
    gamma_prev=gamma_prev,
    gamma_next=gamma_next,
  )


def _max_abs_z_drift(positions_frac: np.ndarray, z0_frac: float) -> float:
  return float(np.max(np.abs(np.asarray(positions_frac)[:, 2] - float(z0_frac))))


def _internal_defaults():
  return {
    'a': 1.0,
    'kT': 1.0,
    'viscosity': 1.0 / (6.0 * math.pi),
    'mr_iters': 10,
    'mr_neighbor_format': partition.NeighborListFormat.Sparse,
    'mr_dr_threshold': 0.5,
    'mr_capacity_multiplier': 2.5,
    'relax_neighbor_format': partition.NeighborListFormat.Sparse,
    'relax_neighbor_dr_threshold': 0.2,
    'relax_neighbor_capacity_multiplier': 2.0,
    'interaction_neighbor_dr_threshold': 0.5,
    'interaction_neighbor_capacity_multiplier': 2.5,
    'z_drift_tol': 1e-10,
  }


def _build_geometry(args, *, a: float, kT: float, viscosity: float):
  n_particles = int(args.n_particles)
  phi_area = float(args.phi)
  dt = float(args.dt)
  n_steps = int(args.n_steps)
  z0_frac = float(args.z0_frac)

  lxy = math.sqrt(n_particles * math.pi * (a ** 2) / phi_area)
  lz = float(args.slab_height_factor) * lxy
  base_box = jnp.diag(jnp.asarray([lxy, lxy, lz], dtype=jnp.float32))
  base_box_np = np.asarray(base_box, dtype=float)
  phi_volume = (
    n_particles * (4.0 / 3.0) * math.pi * (a ** 3) /
    float(np.linalg.det(base_box_np))
  )
  D0 = kT / (6.0 * math.pi * viscosity * a)
  shear_rate = 2.0 * float(args.peclet) * D0 / (a ** 2)
  shear_schedule = {'xy': lambda t: shear_rate * t}
  shear_vector_schedule = lambda t: (shear_rate * t, 0.0, 0.0)
  return {
    'n_particles': n_particles,
    'phi_area': phi_area,
    'dt': dt,
    'n_steps': n_steps,
    'z0_frac': z0_frac,
    'lxy': lxy,
    'lz': lz,
    'base_box': base_box,
    'base_box_np': base_box_np,
    'phi_volume': phi_volume,
    'D0': D0,
    'shear_rate': shear_rate,
    'shear_schedule': shear_schedule,
    'shear_vector_schedule': shear_vector_schedule,
  }


def _write_params_json(out_dir: str, params: dict) -> str:
  params_path = os.path.join(out_dir, 'params.json')
  with open(params_path, 'w') as handle:
    json.dump(params, handle, indent=2, sort_keys=True)
  return params_path


def _write_initial_configuration(
  *,
  out_dir: str,
  base_box_np: np.ndarray,
  positions_fractional: np.ndarray,
) -> str:
  init_frac = np.mod(np.asarray(positions_fractional, dtype=float), 1.0)
  init_pos_real = np.asarray(init_frac @ base_box_np.T, dtype=float)
  confin_path = os.path.join(out_dir, 'confin.data')
  write_lammps_data(
    confin_path,
    base_box_np,
    init_pos_real,
    comment='Generated by examples/shear/shear_slab_surface.py (initial)',
  )
  return confin_path


def main(argv=None):
  args = parse_args(argv)

  internal = _internal_defaults()
  a = internal['a']
  kT = internal['kT']
  viscosity = internal['viscosity']
  diameter = 2.0 * a
  mr_iters = internal['mr_iters']
  mr_neighbor_format = internal['mr_neighbor_format']
  mr_dr_threshold = internal['mr_dr_threshold']
  mr_capacity_multiplier = internal['mr_capacity_multiplier']
  relax_neighbor_format = internal['relax_neighbor_format']
  relax_neighbor_dr_threshold = internal['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = internal['relax_neighbor_capacity_multiplier']
  interaction_neighbor_dr_threshold = internal['interaction_neighbor_dr_threshold']
  interaction_neighbor_capacity_multiplier = internal['interaction_neighbor_capacity_multiplier']
  z_drift_tol = internal['z_drift_tol']

  geometry = _build_geometry(args, a=a, kT=kT, viscosity=viscosity)
  n_particles = geometry['n_particles']
  phi_area = geometry['phi_area']
  dt = geometry['dt']
  n_steps = geometry['n_steps']
  z0_frac = geometry['z0_frac']
  lxy = geometry['lxy']
  lz = geometry['lz']
  base_box = geometry['base_box']
  base_box_np = geometry['base_box_np']
  phi_volume = geometry['phi_volume']
  D0 = geometry['D0']
  shear_rate = geometry['shear_rate']
  shear_schedule = geometry['shear_schedule']
  shear_vector_schedule = geometry['shear_vector_schedule']

  _CONSOLE.section('System')
  _CONSOLE.info(f'N = {n_particles}')
  _CONSOLE.info(f'phi_area = {phi_area:.6g}')
  _CONSOLE.info(f'Lxy = {lxy:.6g}, Lz = {lz:.6g}')
  _CONSOLE.info(f'phi_volume_for_estimator = {phi_volume:.6g}')
  _CONSOLE.info(f'shear_rate = {shear_rate:.6e}, strain/step = {shear_rate * dt:.6e}')

  displacement, shift_base, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  shift = _surface_shift(shift_base)
  displacement_0, shift_base_0 = space.periodic_general(
    base_box,
    fractional_coordinates=True,
  )
  shift_0 = _surface_shift(shift_base_0)

  key = random.PRNGKey(args.seed)
  key, init_key, run_key = random.split(key, 3)
  Rxy = random.uniform(init_key, (n_particles, 2), minval=0.0, maxval=1.0)
  z_col = jnp.full((n_particles, 1), jnp.asarray(z0_frac, dtype=Rxy.dtype))
  R0 = jnp.concatenate([Rxy, z_col], axis=1)
  R0 = _relax_positions(
    R0,
    displacement_0,
    shift_0,
    diameter,
    base_box,
    int(args.relax_steps),
    relax_neighbor_format,
    relax_neighbor_capacity_multiplier,
    relax_neighbor_dr_threshold,
  )
  # Ensure the relaxed state remains exactly on the target surface plane.
  R0 = R0.at[:, 2].set(jnp.asarray(z0_frac, dtype=R0.dtype))
  min_dist = _min_pair_distance(R0, displacement_0)
  _CONSOLE.info(f'Post-relax minimum pair distance: {min_dist:.6f}')

  potential_params = {
    'particle_radius': a,
    'viscosity': viscosity,
    'repulsion_dt': dt,
    'r_min': 1e-6,
  }
  potential_r_cut = diameter

  metric_shear = space.canonicalize_displacement_or_metric(displacement)
  energy_fn_all_pairs = smap.pair(
    _varga_repulsive_pair,
    metric_shear,
    ignore_unused_parameters=True,
    **potential_params,
  )
  energy_fn_neighbor = smap.pair_neighbor_list(
    _varga_repulsive_pair,
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
    format=partition.NeighborListFormat.Sparse,
  )

  shear_t_bounds = (0.0, float(dt * n_steps))
  rpy_params = rpy.estimate_rpy_params(
    tol=float(args.tol),
    A=base_box,
    a=a,
    N=n_particles,
    phi=phi_volume,
    xi_override=float(args.xi),
    shear_vector_schedule=shear_vector_schedule,
    shear_t_bounds=shear_t_bounds,
    shear_remap=True,
    notes=True,
  )
  _CONSOLE.section('RPY Parameters')
  _CONSOLE.info(
    f'xi={float(rpy_params.xi):.6f}, '
    f'rcut={float(rpy_params.rcut):.6f}, '
    f'P={int(rpy_params.P)}, '
    f'M={int(rpy_params.M)}, '
    f'theta={float(rpy_params.theta):.6f}'
  )

  init_fn, apply_fn = simulate.rpy_with_shear(
    (displacement, shift, box_of),
    energy_fn,
    dt=dt,
    kT=kT,
    a=a,
    xi=float(rpy_params.xi),
    eta=viscosity,
    shear_vector_schedule=shear_vector_schedule,
    t0=0.0,
    neighbor_format=mr_neighbor_format,
    dr_threshold=mr_dr_threshold,
    capacity_multiplier=mr_capacity_multiplier,
    real_space_mode='auto',
    rcut=float(rpy_params.rcut),
    P=int(rpy_params.P),
    Mgrid=int(rpy_params.M),
    theta=float(rpy_params.theta),
    lattice_extent=int(rpy_params.lattice_extent),
    mr_iters=mr_iters,
  )

  state = init_fn(run_key, R0)
  interaction_neighbor = interaction_neighbor_fn.allocate(
    state.integrator_position,
    box=box_of(t=0.0),
  )
  _check_neighbor_health(interaction_neighbor, stage='init')

  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)

  params = {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi_area,
      'dt': dt,
      'n_steps': n_steps,
      'peclet': float(args.peclet),
      'seed': int(args.seed),
      'traj_every': int(args.traj_every),
      'progress_every': int(args.progress_every),
      'xi': float(args.xi),
      'tol': float(args.tol),
      'slab_height_factor': float(args.slab_height_factor),
      'z0_frac': z0_frac,
      'relax_steps': int(args.relax_steps),
      'out_dir': out_dir,
    },
    'internal': {
      'a': a,
      'kT': kT,
      'viscosity': viscosity,
      'mr_iters': mr_iters,
      'mr_neighbor_format': 'sparse',
      'mr_dr_threshold': mr_dr_threshold,
      'mr_capacity_multiplier': mr_capacity_multiplier,
      'interaction_neighbor_dr_threshold': interaction_neighbor_dr_threshold,
      'interaction_neighbor_capacity_multiplier': interaction_neighbor_capacity_multiplier,
      'potential_name': 'varga_rpy_repulsive_only_in_script',
      'potential_r_cut': potential_r_cut,
      'surface_constraint': True,
      'z_drift_tolerance': z_drift_tol,
    },
    'derived': {
      'dim': 3,
      'surface_constraint': True,
      'phi_area': phi_area,
      'phi_volume_for_estimator': phi_volume,
      'slab_height_factor': float(args.slab_height_factor),
      'z0_frac': z0_frac,
      'box_matrix': _to_jsonable(base_box_np),
      'Lxy': lxy,
      'Lz': lz,
      'D0': D0,
      'shear_rate': shear_rate,
      'rpy_xi': float(rpy_params.xi),
      'rpy_rcut': float(rpy_params.rcut),
      'rpy_P': int(rpy_params.P),
      'rpy_M': int(rpy_params.M),
      'rpy_theta': float(rpy_params.theta),
      'rpy_lattice_extent': int(rpy_params.lattice_extent),
    },
  }
  params_path = _write_params_json(out_dir, params)
  _CONSOLE.info(f'Wrote parameters to {params_path}')

  confin_path = _write_initial_configuration(
    out_dir=out_dir,
    base_box_np=base_box_np,
    positions_fractional=np.asarray(state.integrator_position, dtype=float),
  )
  _CONSOLE.info(f'Wrote initial configuration to {confin_path}')

  dump_box_fn = _build_reduced_xy_box_fn(base_box_np, shear_rate)
  dumper = RunDumper(
    out_dir=out_dir,
    box_size=lxy,
    dim=3,
    dt=dt,
    traj_every=int(args.traj_every),
    stress_every=0,
    box_fn=dump_box_fn,
    base_box=base_box_np,
    shear_rate=shear_rate,
    time_offset=0.0,
    shear_remap=True,
    unwrap_trajectory=True,
  )

  @jax.jit
  def _step(state_in, interaction_neighbor_in):
    next_time = jnp.asarray(state_in.time) + jnp.asarray(dt, dtype=state_in.time.dtype)
    next_box = box_of(t=next_time)
    pos_for_neighbor = _predict_xy_remapped_positions_for_next_force(
      state_in,
      dt=dt,
      shear_rate=shear_rate,
    )
    interaction_neighbor_out = interaction_neighbor_in.update(
      pos_for_neighbor,
      box=next_box,
    )
    state_out = apply_fn(state_in, interaction_neighbor=interaction_neighbor_out)
    return state_out, interaction_neighbor_out

  z_drift_events = 0
  try:
    # Emit the t=0 frame so trajectories always include the initial state.
    dumper.dump(
      np.array([], dtype=float),
      np.array([], dtype=float),
      np.zeros((0, 3, 3), dtype=float),
      None,
      np.asarray(state.integrator_position, dtype=float)[np.newaxis],
      traj_steps=np.array([0], dtype=np.int64),
    )

    for _ in range(n_steps):
      state, interaction_neighbor = _step(state, interaction_neighbor)
      _check_neighbor_health(
        interaction_neighbor,
        stage=f'step_{int(np.asarray(state.step))}',
      )

      step_i = int(np.asarray(state.step))
      time_i = float(step_i * dt)

      if step_i % int(args.traj_every) == 0:
        dumper.dump(
          np.array([], dtype=float),
          np.array([], dtype=float),
          np.zeros((0, 3, 3), dtype=float),
          None,
          np.asarray(state.integrator_position, dtype=float)[np.newaxis],
          traj_steps=np.array([step_i], dtype=np.int64),
        )

      if int(args.progress_every) > 0 and (step_i % int(args.progress_every) == 0):
        z_drift = _max_abs_z_drift(np.asarray(state.integrator_position), z0_frac)
        if z_drift > z_drift_tol:
          z_drift_events += 1
          _CONSOLE.warn(
            f'Z drift detected at step={step_i}: max_abs_drift={z_drift:.3e}'
          )
        _CONSOLE.progress(
          f'step {step_i}/{n_steps}, time={time_i:.6e}, max_abs_z_drift={z_drift:.3e}'
        )
  finally:
    dumper.close()

  final_step = int(np.asarray(state.step))
  final_time = float(final_step * dt)
  final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
  final_pos_frac = np.mod(np.asarray(state.integrator_position, dtype=float), 1.0)
  final_pos_real = np.asarray(final_pos_frac @ final_box.T, dtype=float)
  confout_path = os.path.join(out_dir, 'confout.data')
  write_lammps_data(
    confout_path,
    final_box,
    final_pos_real,
    comment=(
      'Generated by examples/shear/shear_slab_surface.py '
      f'(step={final_step})'
    ),
  )
  _CONSOLE.info(f'Wrote final configuration to {confout_path}')

  final_z_drift = _max_abs_z_drift(np.asarray(state.integrator_position), z0_frac)
  if final_z_drift > z_drift_tol:
    z_drift_events += 1
    _CONSOLE.warn(
      f'Final z drift detected: max_abs_drift={final_z_drift:.3e}'
    )
  if z_drift_events == 0:
    _CONSOLE.success('Done. No z drift detected.')
  else:
    _CONSOLE.warn(
      f'Done with {z_drift_events} z-drift warning event(s).'
    )


if __name__ == '__main__':
  main()
