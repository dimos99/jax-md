"""2D bidisperse hard-sphere runner with `simulate.brownian_hard_sphere`."""

import math
import os
import time

from jax import jit
from jax import random
import jax
import jax.numpy as jnp
import numpy as np

from jax_md import energy
from jax_md import minimize
from jax_md import partition
from jax_md import simulate
from jax_md import space

from hard_sphere_2d_polydisperse_cli import build_internal_config
from hard_sphere_2d_polydisperse_cli import parse_args
from hard_sphere_2d_polydisperse_cli import resolve_runtime_settings
from shear_console import get_console
from shear_init import _build_reduced_xy_box_fn
from shear_init import _min_pair_distance
from shear_output import RunDumper
from shear_output import _to_jsonable
from shear_output import write_lammps_data
from shear_prepare_utils import _build_format_map
from shear_prepare_utils import _write_params_json
from shear_runtime_utils import _allocate_probe_sized_neighbor
from shear_time_utils import _predict_xy_remapped_positions_for_next_force
from shear_time_utils import _state_next_time_from_step
from shear_time_utils import _state_step
from shear_time_utils import _state_time_from_step

_CONSOLE = get_console()


def _bidisperse_species_counts(
  *,
  n_particles: int,
  species_b_fraction: float,
) -> dict:
  """Returns exact bidisperse species counts using rounded fraction targets."""
  n_b = int(round(float(species_b_fraction) * int(n_particles)))
  n_b = min(max(n_b, 0), int(n_particles))
  n_a = int(n_particles) - n_b
  return {
    'n_a': n_a,
    'n_b': n_b,
  }


def _build_species_labels(
  *,
  n_particles: int,
  species_b_fraction: float,
  key,
) -> dict:
  """Builds an exact-count bidisperse species vector and random permutation."""
  counts = _bidisperse_species_counts(
    n_particles=n_particles,
    species_b_fraction=species_b_fraction,
  )
  n_a = counts['n_a']
  n_b = counts['n_b']
  species = jnp.concatenate([
    jnp.zeros((n_a,), dtype=jnp.int32),
    jnp.ones((n_b,), dtype=jnp.int32),
  ])
  species = random.permutation(key, species)
  return {
    'species': species,
    'n_a': n_a,
    'n_b': n_b,
  }


def _bidisperse_box_size_from_phi(
  *,
  n_a: int,
  n_b: int,
  hs_core_radius_a: float,
  hs_core_radius_b: float,
  phi: float,
) -> float:
  """Computes the exact square-box length for a 2D bidisperse hard-disk system."""
  occupied_area = (
    float(n_a) * math.pi * float(hs_core_radius_a) ** 2
    + float(n_b) * math.pi * float(hs_core_radius_b) ** 2
  )
  return float(math.sqrt(occupied_area / float(phi)))


def _build_particle_properties(
  *,
  species,
  hydro_radius_a: float,
  hydro_radius_b: float,
  hs_core_radius_a: float,
  hs_core_radius_b: float,
  kT: float,
  viscosity: float,
  dtype,
) -> dict:
  """Builds per-species and per-particle transport/core properties."""
  hydro_radii = jnp.asarray(
    [hydro_radius_a, hydro_radius_b], dtype=dtype)
  core_radii = jnp.asarray(
    [hs_core_radius_a, hs_core_radius_b], dtype=dtype)
  diameter_species = 2.0 * core_radii
  mobility_species = 1.0 / (6.0 * math.pi * float(viscosity) * hydro_radii)
  diffusivity_species = float(kT) * mobility_species
  diameter = diameter_species[species]
  mobility = mobility_species[species]
  return {
    'hydro_radii': hydro_radii,
    'core_radii': core_radii,
    'diameter_species': diameter_species,
    'mobility_species': mobility_species,
    'diffusivity_species': diffusivity_species,
    'diameter': diameter,
    'mobility': mobility,
  }


def _as_box_matrix(box, *, dim: int) -> np.ndarray:
  """Returns a square box matrix for scalar/vector/matrix box encodings."""
  arr = np.asarray(box, dtype=float)
  if arr.ndim == 0:
    return np.eye(dim, dtype=float) * float(arr)
  if arr.ndim == 1:
    return np.diag(arr)
  return arr


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
      '--collision-neighbor-skin.'
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
  diameter_max: float,
  diffusivity_max: float,
  dt: float,
  shear_rate: float,
  box_size: float,
  format_map: dict,
) -> dict:
  """Resolves conservative collision-neighbor settings from the largest species."""
  collision_neighbor_format_name = 'sparse'
  collision_neighbor_format = format_map[collision_neighbor_format_name]
  collision_neighbor_zsigma = float(args.collision_neighbor_zsigma)
  collision_neighbor_capacity_multiplier = float(
    args.collision_neighbor_capacity_multiplier
  )
  sigma_rel = float(math.sqrt(4.0 * diffusivity_max * dt))
  sigma_shear = float(abs(float(shear_rate)) * float(dt) * float(box_size))
  skin_from_cli = args.collision_neighbor_skin is not None
  if skin_from_cli:
    collision_neighbor_skin = float(args.collision_neighbor_skin)
  else:
    collision_neighbor_skin = float(collision_neighbor_zsigma * sigma_rel)
  collision_neighbor_r_cutoff = float(
    diameter_max + collision_neighbor_zsigma * sigma_rel + sigma_shear
  )
  collision_neighbor_threshold = float(
    collision_neighbor_r_cutoff + collision_neighbor_skin
  )
  return {
    'format_name': collision_neighbor_format_name,
    'format': collision_neighbor_format,
    'zsigma': collision_neighbor_zsigma,
    'skin': collision_neighbor_skin,
    'sigma_rel': sigma_rel,
    'sigma_shear': sigma_shear,
    'r_cutoff': collision_neighbor_r_cutoff,
    'threshold': collision_neighbor_threshold,
    'capacity_multiplier': collision_neighbor_capacity_multiplier,
    'skin_from_cli': skin_from_cli,
  }


def _relax_positions_bidisperse(
  R_init,
  displacement_0,
  shift_0,
  species,
  sigma_species,
  box,
  steps: int,
  neighbor_format,
  neighbor_capacity_multiplier: float,
  dr_threshold: float,
):
  """Relaxes initial positions using a species-aware soft-sphere surrogate."""
  if steps <= 0:
    return R_init

  neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
    displacement_0,
    box,
    species=species,
    sigma=sigma_species,
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

  @jit
  def _run(state_in, neighbor_in):
    def _step(_, carry):
      state_curr, neighbor_curr, overflow = carry
      neighbor_curr = neighbor_curr.update(state_curr.position, box=box)
      overflow = jnp.logical_or(overflow, neighbor_curr.did_buffer_overflow)
      state_curr = apply_min(state_curr, neighbor=neighbor_curr)
      return state_curr, neighbor_curr, overflow
    return jax.lax.fori_loop(0, steps, _step, (state_in, neighbor_in, False))

  state, _, overflow = _run(state, neighbor)
  if bool(np.asarray(overflow)):
    _CONSOLE.warn('overlap-relaxation neighbor list overflow detected.')
  return state.position


def _build_params_payload(
  *,
  args,
  n_particles: int,
  phi: float,
  dt: float,
  kT: float,
  viscosity: float,
  max_collision_loops: int,
  event_time_tol,
  dim: int,
  box_size: float,
  base_box_np: np.ndarray,
  box_volume: float,
  shear_rate: float,
  shear_reference_hydro_radius: float,
  shear_reference_diffusivity: float,
  planned_steps: int,
  confin_path: str,
  relax_steps: int,
  relax_neighbor_format: str,
  relax_neighbor_dr_threshold: float,
  relax_neighbor_capacity_multiplier: float,
  species,
  n_a: int,
  n_b: int,
  hydro_radii,
  core_radii,
  diameter_species,
  mobility_species,
  diffusivity_species,
  diameter,
  mobility,
  collision_neighbor_settings,
) -> dict:
  """Builds the `params.json` payload for 2D bidisperse hard-sphere runs."""
  species_np = np.asarray(species, dtype=int)
  atom_types = species_np + 1
  return {
    'user_args': {
      'n_particles': n_particles,
      'phi': phi,
      'dt': dt,
      'peclet': args.peclet,
      'n_steps': args.n_steps,
      'stress_every': args.stress_every,
      'traj_every': args.traj_every,
      'progress_every': args.progress_every,
      'seed': args.seed,
      'out_dir': args.out_dir,
      'species_b_fraction': args.species_b_fraction,
      'hydro_radius_a': args.hydro_radius_a,
      'hydro_radius_b': args.hydro_radius_b,
      'hs_core_radius_a': args.hs_core_radius_a,
      'hs_core_radius_b': args.hs_core_radius_b,
      'max_collision_loops': args.max_collision_loops,
      'collision_neighbor_zsigma': args.collision_neighbor_zsigma,
      'collision_neighbor_skin': args.collision_neighbor_skin,
      'collision_neighbor_capacity_multiplier': (
        args.collision_neighbor_capacity_multiplier
      ),
    },
    'internal': {
      'kT': kT,
      'viscosity': viscosity,
      'max_collision_loops': int(max_collision_loops),
      'event_time_tol': event_time_tol,
      'relax_steps': int(relax_steps),
      'relax_neighbor_format': relax_neighbor_format,
      'relax_neighbor_dr_threshold': float(relax_neighbor_dr_threshold),
      'relax_neighbor_capacity_multiplier': float(
        relax_neighbor_capacity_multiplier
      ),
    },
    'derived': {
      'integrator': 'simulate.brownian_hard_sphere',
      'initialization_mode': 'random_relax',
      'dim': dim,
      'box_size': box_size,
      'box_matrix': _to_jsonable(base_box_np),
      'box_volume': box_volume,
      'shear_rate': shear_rate,
      'shear_reference_species': 'A',
      'shear_reference_hydro_radius': shear_reference_hydro_radius,
      'shear_reference_diffusivity': shear_reference_diffusivity,
      'traj_box_frame': 'reduced_lab_xy',
      'traj_coords_frame': 'unwrapped_lab_continuous',
      'traj_remap_aware': True,
      'confin_path': confin_path,
      'planned_steps': planned_steps,
    },
    'polydispersity': {
      'counts': {'a': int(n_a), 'b': int(n_b)},
      'fraction_b_target': float(args.species_b_fraction),
      'fraction_b_actual': float(n_b / n_particles),
      'species_labels': _to_jsonable(species_np),
      'atom_types': _to_jsonable(atom_types),
      'hydro_radii': _to_jsonable(np.asarray(hydro_radii, dtype=float)),
      'hs_core_radii': _to_jsonable(np.asarray(core_radii, dtype=float)),
      'diameter_species': _to_jsonable(np.asarray(diameter_species, dtype=float)),
      'mobility_species': _to_jsonable(np.asarray(mobility_species, dtype=float)),
      'diffusivity_species': _to_jsonable(np.asarray(diffusivity_species, dtype=float)),
      'diameter': _to_jsonable(np.asarray(diameter, dtype=float)),
      'mobility': _to_jsonable(np.asarray(mobility, dtype=float)),
    },
    'collision_neighbor': {
      'format': str(collision_neighbor_settings['format_name']),
      'zsigma': float(collision_neighbor_settings['zsigma']),
      'sigma_rel': float(collision_neighbor_settings['sigma_rel']),
      'sigma_shear': float(collision_neighbor_settings['sigma_shear']),
      'r_cutoff': float(collision_neighbor_settings['r_cutoff']),
      'skin': float(collision_neighbor_settings['skin']),
      'threshold': float(collision_neighbor_settings['threshold']),
      'skin_from_cli': bool(collision_neighbor_settings['skin_from_cli']),
      'capacity_multiplier': float(
        collision_neighbor_settings['capacity_multiplier']
      ),
      'build_policy': str(
        collision_neighbor_settings.get('build_policy', 'static_box')
      ),
      'build_gamma_xy': (
        float(collision_neighbor_settings['build_gamma_xy'])
        if 'build_gamma_xy' in collision_neighbor_settings
        else None
      ),
      'build_box': _to_jsonable(
        collision_neighbor_settings.get('build_box', base_box_np)
      ),
      'build_fractional_cell_size': float(
        collision_neighbor_settings.get(
          'build_fractional_cell_size',
          collision_neighbor_settings['threshold'] / box_size,
        )
      ),
    },
  }


def _zero_energy(R, **unused_kwargs):
  """Returns zero potential energy for pure hard-sphere runs."""
  return jnp.zeros((), dtype=R.dtype)


def main():
  """Runs one 2D bidisperse hard-sphere trajectory and writes run artifacts."""
  args = parse_args()
  wall_start = time.perf_counter()

  internal = build_internal_config()
  runtime = resolve_runtime_settings(args, internal)
  kT = runtime['kT']
  viscosity = runtime['viscosity']
  dt = float(args.dt)
  relax_steps = runtime['relax_steps']
  relax_neighbor_format = runtime['relax_neighbor_format']
  relax_neighbor_dr_threshold = runtime['relax_neighbor_dr_threshold']
  relax_neighbor_capacity_multiplier = runtime['relax_neighbor_capacity_multiplier']
  max_collision_loops = (
    int(args.max_collision_loops)
    if args.max_collision_loops is not None
    else int(1e7)
  )
  event_time_tol = None
  format_map = _build_format_map()
  do_stress = args.stress_every > 0
  do_traj = args.traj_every > 0
  planned_steps = int(args.n_steps)
  if do_stress:
    raise ValueError(
      'Collisional stress output is unavailable for '
      'hard_sphere_2d_polydisperse because this runner uses per-particle '
      'mobility. Set --stress_every 0 to disable stress output.'
    )

  devices = jax.devices()
  device_labels = ', '.join(
    f'{d.platform}:{getattr(d, "device_kind", "device")}' for d in devices)
  _CONSOLE.section('Environment')
  _CONSOLE.info(f'JAX backend: {jax.default_backend()}')
  _CONSOLE.info(f'JAX devices: {device_labels}')

  dim = 2
  n_particles = int(args.n_particles)
  phi = float(args.phi)
  seed = int(args.seed)
  key = random.PRNGKey(seed)
  species_key, init_key, run_key = random.split(key, 3)

  species_info = _build_species_labels(
    n_particles=n_particles,
    species_b_fraction=float(args.species_b_fraction),
    key=species_key,
  )
  species = species_info['species']
  n_a = species_info['n_a']
  n_b = species_info['n_b']

  box_size = _bidisperse_box_size_from_phi(
    n_a=n_a,
    n_b=n_b,
    hs_core_radius_a=float(args.hs_core_radius_a),
    hs_core_radius_b=float(args.hs_core_radius_b),
    phi=phi,
  )
  base_box = jnp.eye(dim) * box_size
  base_box_np = np.asarray(base_box, dtype=float)
  box_volume = float(abs(np.linalg.det(base_box_np)))

  properties = _build_particle_properties(
    species=species,
    hydro_radius_a=float(args.hydro_radius_a),
    hydro_radius_b=float(args.hydro_radius_b),
    hs_core_radius_a=float(args.hs_core_radius_a),
    hs_core_radius_b=float(args.hs_core_radius_b),
    kT=kT,
    viscosity=viscosity,
    dtype=base_box.dtype,
  )
  hydro_radii = properties['hydro_radii']
  core_radii = properties['core_radii']
  diameter_species = properties['diameter_species']
  mobility_species = properties['mobility_species']
  diffusivity_species = properties['diffusivity_species']
  diameter = properties['diameter']
  mobility = properties['mobility']
  diameter_max = float(np.max(np.asarray(diameter_species, dtype=float)))
  diffusivity_max = float(np.max(np.asarray(diffusivity_species, dtype=float)))
  shear_reference_hydro_radius = float(np.asarray(hydro_radii, dtype=float)[0])
  shear_reference_diffusivity = float(
    np.asarray(diffusivity_species, dtype=float)[0])
  shear_rate = float(
    2.0 * float(args.peclet) * shear_reference_diffusivity /
    (shear_reference_hydro_radius ** 2)
  )
  shear_t0 = 0.0
  shear_schedule = {'xy': lambda t: shear_rate * t}

  collision_neighbor_settings = _resolve_collision_neighbor_settings(
    args=args,
    diameter_max=diameter_max,
    diffusivity_max=diffusivity_max,
    dt=dt,
    shear_rate=shear_rate,
    box_size=box_size,
    format_map=format_map,
  )

  _CONSOLE.section('System')
  _CONSOLE.info(f'Box size L = {box_size:.6f}')
  _CONSOLE.info(f'Counts: n_A={n_a}, n_B={n_b}')
  _CONSOLE.info(
    'Hydrodynamic radii: '
    f'a_A={float(hydro_radii[0]):.6f}, a_B={float(hydro_radii[1]):.6f}'
  )
  _CONSOLE.info(
    'Hard-sphere core radii: '
    f'r_A={float(core_radii[0]):.6f}, r_B={float(core_radii[1]):.6f}'
  )
  _CONSOLE.info(
    'Hard-sphere diameters: '
    f'd_A={float(diameter_species[0]):.6f}, d_B={float(diameter_species[1]):.6f}'
  )
  _CONSOLE.info(
    'Mobilities: '
    f'mu_A={float(mobility_species[0]):.6e}, '
    f'mu_B={float(mobility_species[1]):.6e}'
  )
  _CONSOLE.info(
    'Diffusivities: '
    f'D_A={float(diffusivity_species[0]):.6e}, '
    f'D_B={float(diffusivity_species[1]):.6e}'
  )
  _CONSOLE.info(
    'Shear: '
    f'Pe={float(args.peclet):.6g}, '
    f'reference=A, shear_rate={shear_rate:.6e}, '
    f'strain/step={shear_rate * dt:.6e}'
  )
  _CONSOLE.info(f'Max collision loops/step = {max_collision_loops}')
  _CONSOLE.info(
    'Collision neighbors: '
    f'format={collision_neighbor_settings["format_name"]}, '
    f'zsigma={collision_neighbor_settings["zsigma"]:.3g}, '
    f'sigma_rel={collision_neighbor_settings["sigma_rel"]:.6e}, '
    f'sigma_shear={collision_neighbor_settings["sigma_shear"]:.6e}, '
    f'r_cutoff={collision_neighbor_settings["r_cutoff"]:.6f}, '
    f'skin={collision_neighbor_settings["skin"]:.6f}'
    f'({"cli" if collision_neighbor_settings["skin_from_cli"] else "auto"}), '
    f'threshold={collision_neighbor_settings["threshold"]:.6f}, '
    f'capacity_multiplier={collision_neighbor_settings["capacity_multiplier"]:.3g}'
  )

  displacement_0, shift_0 = space.periodic_general(
    base_box, fractional_coordinates=True)
  displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=shear_schedule,
    fractional_coordinates=True,
    remap=True,
  )
  sigma_soft = jnp.asarray([
    [float(diameter_species[0]), 0.5 * float(diameter_species[0] + diameter_species[1])],
    [0.5 * float(diameter_species[0] + diameter_species[1]), float(diameter_species[1])],
  ], dtype=base_box.dtype)

  R0 = random.uniform(init_key, (n_particles, dim), minval=0.0, maxval=1.0)
  R0 = _relax_positions_bidisperse(
    R0,
    displacement_0,
    shift_0,
    species,
    sigma_soft,
    base_box,
    relax_steps,
    format_map[relax_neighbor_format],
    relax_neighbor_capacity_multiplier,
    relax_neighbor_dr_threshold,
  )
  min_dist = _min_pair_distance(R0, displacement_0)
  _CONSOLE.info(f'Post-relax minimum pair distance: {min_dist:.6f}')

  metric = space.canonicalize_displacement_or_metric(displacement)
  collision_build_box_info = _resolve_worst_xy_remap_build_box(
    box_of=box_of,
    base_box=base_box,
    cutoff=collision_neighbor_settings['threshold'],
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
    f'r_cutoff={collision_neighbor_settings["r_cutoff"]:.6f}, '
    f'threshold={collision_neighbor_settings["threshold"]:.6f}, '
    f'fractional_cell_size={collision_build_box_info["fractional_cell_size"]:.6f}'
  )
  collision_neighbor_fn = partition.neighbor_list(
    metric,
    base_box,
    r_cutoff=collision_neighbor_settings['r_cutoff'],
    dr_threshold=collision_neighbor_settings['skin'],
    capacity_multiplier=collision_neighbor_settings['capacity_multiplier'],
    fractional_coordinates=True,
    format=collision_neighbor_settings['format'],
  )

  init_fn, apply_fn = simulate.brownian_hard_sphere(
    _zero_energy,
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
  )

  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)

  atom_types = np.asarray(species, dtype=int) + 1
  confin_path = os.path.join(out_dir, 'confin.data')
  init_frac = np.mod(np.asarray(R0, dtype=float), 1.0)
  init_pos_real = np.asarray(init_frac @ base_box_np.T, dtype=float)
  write_lammps_data(
    confin_path,
    base_box_np,
    init_pos_real,
    atom_types=atom_types,
    comment=(
      'Generated by examples/shear/hard_sphere_2d_polydisperse.py '
      '(init_mode=random_relax)'
    ),
  )
  _CONSOLE.info(f'Wrote initial configuration to {confin_path}')

  params = _build_params_payload(
    args=args,
    n_particles=n_particles,
    phi=phi,
    dt=dt,
    kT=kT,
    viscosity=viscosity,
    max_collision_loops=max_collision_loops,
    event_time_tol=event_time_tol,
    dim=dim,
    box_size=box_size,
    base_box_np=base_box_np,
    box_volume=box_volume,
    shear_rate=shear_rate,
    shear_reference_hydro_radius=shear_reference_hydro_radius,
    shear_reference_diffusivity=shear_reference_diffusivity,
    planned_steps=planned_steps,
    confin_path=confin_path,
    relax_steps=relax_steps,
    relax_neighbor_format=relax_neighbor_format,
    relax_neighbor_dr_threshold=relax_neighbor_dr_threshold,
    relax_neighbor_capacity_multiplier=relax_neighbor_capacity_multiplier,
    species=species,
    n_a=n_a,
    n_b=n_b,
    hydro_radii=hydro_radii,
    core_radii=core_radii,
    diameter_species=diameter_species,
    mobility_species=mobility_species,
    diffusivity_species=diffusivity_species,
    diameter=diameter,
    mobility=mobility,
    collision_neighbor_settings=collision_neighbor_settings,
  )
  params_path = _write_params_json(out_dir, params)
  _CONSOLE.info(f'Wrote parameters to {params_path}')

  dump_box_fn = _build_reduced_xy_box_fn(base_box_np, shear_rate)
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
    atom_types=atom_types,
  )

  _CONSOLE.section('Run Plan')
  _CONSOLE.info(
    f'Running one 2D bidisperse hard-sphere trajectory for {planned_steps} steps.'
  )
  if do_stress and do_traj:
    _CONSOLE.info(f'Outputs: stress.dat + traj.dump in {out_dir}')
  elif do_stress:
    _CONSOLE.info(f'Outputs: stress.dat in {out_dir}')
  elif do_traj:
    _CONSOLE.info(f'Outputs: traj.dump in {out_dir}')
  else:
    _CONSOLE.info('Outputs: none (both --stress_every and --traj_every are 0).')

  @jit
  def run_one_step(state_in, collision_neighbor_in):
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

  @jit
  def evaluate_stress(state_in):
    step = _state_step(state_in, dt=dt, t0=shear_t0)
    curr_time = _state_time_from_step(state_in, dt=dt, t0=shear_t0)
    strain = jnp.asarray(shear_rate, dtype=curr_time.dtype) * curr_time
    return step, curr_time, strain, state_in.stress

  try:
    state = init_fn(run_key, R0)
    box_t0 = box_of(t=shear_t0)
    collision_build_box = jnp.asarray(
      collision_build_box_info['box'], dtype=base_box.dtype)
    collision_neighbor = _allocate_probe_sized_neighbor(
      collision_neighbor_fn,
      state.position,
      box=box_t0,
      build_box=collision_build_box,
    )
    if not _check_collision_neighbor_status(collision_neighbor, 'init'):
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
      state, collision_neighbor = run_one_step(
        state, collision_neighbor)
      steps_done += 1

      if not _check_nan_positions(state, f'step {steps_done}'):
        return
      if not _check_collision_loop_status(state, f'step {steps_done}'):
        return
      if not _check_collision_neighbor_status(collision_neighbor, f'step {steps_done}'):
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
        )

      if args.progress_every > 0 and (steps_done % args.progress_every == 0):
        _CONSOLE.progress(f'Step {min(steps_done, planned_steps)}/{planned_steps}')

    final_step = int(np.asarray(_state_step(state, dt=dt, t0=shear_t0)))
    final_time = final_step * dt + float(shear_t0)
    final_box = np.asarray(dump_box_fn(t=final_time), dtype=float)
    pos_frac = np.mod(np.asarray(state.position, dtype=float), 1.0)
    pos_real = np.asarray(pos_frac @ final_box.T, dtype=float)
    confout_path = os.path.join(out_dir, 'confout.data')
    write_lammps_data(
      confout_path,
      final_box,
      pos_real,
      atom_types=atom_types,
      comment=(
        'Generated by examples/shear/hard_sphere_2d_polydisperse.py '
        f'(step={final_step})'
      ),
    )
    _CONSOLE.info(f'Wrote final data snapshot {confout_path}')
  finally:
    dumper.close()

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
