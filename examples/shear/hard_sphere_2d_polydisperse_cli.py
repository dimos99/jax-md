"""CLI parsing and static config for the 2D bidisperse hard-sphere runner."""

import argparse
from decimal import Decimal
from decimal import InvalidOperation
import math

from shear_console import get_console

_CONSOLE = get_console()


def _parse_int_like(value: str) -> int:
  """Parses integer-like CLI values, including scientific notation."""
  s = str(value).strip()
  try:
    return int(s, 10)
  except ValueError:
    pass
  try:
    d = Decimal(s)
  except InvalidOperation as err:
    raise argparse.ArgumentTypeError(
      f'Expected integer value, got {value!r}.'
    ) from err
  if not d.is_finite():
    raise argparse.ArgumentTypeError(
      f'Expected finite integer value, got {value!r}.'
    )
  integral = d.to_integral_value()
  if d != integral:
    raise argparse.ArgumentTypeError(
      f'Expected integer value, got {value!r}.'
    )
  return int(integral)


def parse_args():
  parser = argparse.ArgumentParser(
    description='2D bidisperse hard-sphere Brownian runner without shear.'
  )
  parser.add_argument('--n_particles', type=_parse_int_like, default=None)
  parser.add_argument('--phi', type=float, default=None)
  parser.add_argument('--dt', type=float, default=None)
  parser.add_argument('--n_steps', type=_parse_int_like, default=30000)
  parser.add_argument('--stress_every', type=_parse_int_like, default=0,
                      help='Set to 0 to disable stress calculation/output.')
  parser.add_argument('--traj_every', type=_parse_int_like, default=100,
                      help='Set to 0 to disable trajectory output.')
  parser.add_argument('--progress_every', type=_parse_int_like, default=1000)
  parser.add_argument('--seed', type=_parse_int_like, default=42)
  parser.add_argument(
    '--out_dir',
    '--out',
    dest='out_dir',
    type=str,
    default=None,
    help='Output directory for params/stress/trajectory files (required).',
  )
  parser.add_argument(
    '--species-b-fraction',
    type=float,
    default=None,
    help='Target fraction of species-B particles in [0, 1].',
  )
  parser.add_argument(
    '--hydro-radius-a',
    type=float,
    default=None,
    help='Hydrodynamic radius of species A.',
  )
  parser.add_argument(
    '--hydro-radius-b',
    type=float,
    default=None,
    help='Hydrodynamic radius of species B.',
  )
  parser.add_argument(
    '--hs-core-radius-a',
    type=float,
    default=None,
    help='Hard-sphere core radius of species A.',
  )
  parser.add_argument(
    '--hs-core-radius-b',
    type=float,
    default=None,
    help='Hard-sphere core radius of species B.',
  )
  parser.add_argument(
    '--max-collision-loops',
    type=_parse_int_like,
    default=None,
    help='Maximum hard-sphere collision-resolution loops per timestep.',
  )
  parser.add_argument(
    '--collision-neighbor-zsigma',
    type=float,
    default=6.0,
    help=(
      'Collision-neighbor cutoff multiplier z where '
      'r_cutoff = diameter_max + z*sqrt(4*D_max*dt). Must be > 0.'
    ),
  )
  parser.add_argument(
    '--collision-neighbor-skin',
    type=float,
    default=None,
    help=(
      'Optional collision-neighbor skin added to r_cutoff via dr_threshold; '
      'search threshold is r_cutoff + skin. If omitted, defaults to '
      'zsigma*sqrt(4*D_max*dt).'
    ),
  )
  parser.add_argument(
    '--collision-neighbor-capacity-multiplier',
    type=float,
    default=2.5,
    help='Capacity multiplier for hard-sphere collision neighbor list (must be > 0).',
  )

  args = parser.parse_args()
  if args.n_particles is None:
    raise ValueError('--n_particles is required.')
  if args.phi is None:
    raise ValueError('--phi is required.')
  if args.dt is None:
    raise ValueError('--dt is required.')
  if args.species_b_fraction is None:
    raise ValueError('--species-b-fraction is required.')
  if args.hydro_radius_a is None:
    raise ValueError('--hydro-radius-a is required.')
  if args.hydro_radius_b is None:
    raise ValueError('--hydro-radius-b is required.')
  if args.hs_core_radius_a is None:
    raise ValueError('--hs-core-radius-a is required.')
  if args.hs_core_radius_b is None:
    raise ValueError('--hs-core-radius-b is required.')
  if args.out_dir is None:
    raise ValueError('--out_dir is required.')

  if args.n_particles <= 1:
    raise ValueError('n_particles must be > 1.')
  if not (0.0 < float(args.phi) <= 1.0):
    raise ValueError(
      f'--phi must be in (0, 1], got {args.phi}. '
      'Did you mean e.g. "--phi 0.45" (not "--phi 045")?'
    )
  phi_close_packing_2d = math.pi / (2.0 * math.sqrt(3.0))
  if float(args.phi) > phi_close_packing_2d:
    _CONSOLE.warn(
      f'--phi={args.phi} exceeds 2D hard-disk close packing '
      f'(~{phi_close_packing_2d:.6f}). This can lead to severe overlaps and NaNs.'
    )
  if float(args.dt) <= 0.0:
    raise ValueError('dt must be > 0.')
  if args.n_steps <= 0:
    raise ValueError('n_steps must be > 0.')
  if args.stress_every < 0:
    raise ValueError('stress_every must be >= 0.')
  if args.traj_every < 0:
    raise ValueError('traj_every must be >= 0.')
  if args.progress_every < 0:
    raise ValueError('progress_every must be >= 0.')
  if not (0.0 <= float(args.species_b_fraction) <= 1.0):
    raise ValueError('species_b_fraction must be in [0, 1].')
  if float(args.hydro_radius_a) <= 0.0:
    raise ValueError('hydro_radius_a must be > 0.')
  if float(args.hydro_radius_b) <= 0.0:
    raise ValueError('hydro_radius_b must be > 0.')
  if float(args.hs_core_radius_a) <= 0.0:
    raise ValueError('hs_core_radius_a must be > 0.')
  if float(args.hs_core_radius_b) <= 0.0:
    raise ValueError('hs_core_radius_b must be > 0.')
  if args.max_collision_loops is not None and int(args.max_collision_loops) <= 0:
    raise ValueError('max_collision_loops must be > 0 when provided.')
  if float(args.collision_neighbor_zsigma) <= 0.0:
    raise ValueError('collision_neighbor_zsigma must be > 0.')
  if (
    args.collision_neighbor_skin is not None
    and float(args.collision_neighbor_skin) < 0.0
  ):
    raise ValueError('collision_neighbor_skin must be >= 0 when provided.')
  if float(args.collision_neighbor_capacity_multiplier) <= 0.0:
    raise ValueError('collision_neighbor_capacity_multiplier must be > 0.')
  return args


def build_internal_config():
  """Algorithmic defaults intentionally kept out of the user-facing CLI."""
  return {
    'kT': 1.0,
    'viscosity': 1.0 / (6.0 * math.pi),
    'relax_steps': 250,
    'relax_neighbor_format': 'sparse',
    'relax_neighbor_dr_threshold': 0.2,
    'relax_neighbor_capacity_multiplier': 2.0,
  }


def resolve_runtime_settings(args, internal_cfg) -> dict:
  """Builds validated runtime settings for 2D hard-sphere runs."""
  del args
  kT = float(internal_cfg['kT'])
  viscosity = float(internal_cfg['viscosity'])
  relax_steps = int(internal_cfg['relax_steps'])
  relax_neighbor_format = str(internal_cfg['relax_neighbor_format'])
  relax_neighbor_dr_threshold = float(internal_cfg['relax_neighbor_dr_threshold'])
  relax_neighbor_capacity_multiplier = float(
    internal_cfg['relax_neighbor_capacity_multiplier']
  )

  if kT <= 0.0:
    raise ValueError('internal default kT must be > 0.')
  if viscosity <= 0.0:
    raise ValueError('internal default viscosity must be > 0.')
  if relax_steps < 0:
    raise ValueError('internal default relax_steps must be >= 0.')
  if relax_neighbor_dr_threshold < 0.0:
    raise ValueError('internal default relax_neighbor_dr_threshold must be >= 0.')
  if relax_neighbor_capacity_multiplier <= 0.0:
    raise ValueError('internal default relax_neighbor_capacity_multiplier must be > 0.')

  return {
    'kT': kT,
    'viscosity': viscosity,
    'relax_steps': relax_steps,
    'relax_neighbor_format': relax_neighbor_format,
    'relax_neighbor_dr_threshold': relax_neighbor_dr_threshold,
    'relax_neighbor_capacity_multiplier': relax_neighbor_capacity_multiplier,
  }
