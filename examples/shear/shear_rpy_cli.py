"""CLI parsing and static config for the RPY shear runner."""

import argparse
from decimal import Decimal
from decimal import InvalidOperation
import math

from shear_console import get_console

_CONSOLE = get_console()


def _parse_int_like(value: str) -> int:
  """Parses integer-like CLI values, including scientific notation (e.g. 6e6)."""
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
    description='Shear runner with configurable stress/trajectory output.')

  # Experiment-facing controls.
  parser.add_argument(
    '--n_particles',
    type=_parse_int_like,
    default=None,
    help='Particle count for random initialization mode (required without --init-traj/--init-data).',
  )
  parser.add_argument(
    '--phi',
    type=float,
    default=None,
    help='Packing fraction for random initialization mode (required without --init-traj/--init-data).',
  )
  parser.add_argument('--peclet', type=float, default=0.0)
  parser.add_argument(
    '--dt',
    type=float,
    default=None,
    help='Integration timestep in the current simulation units (required).',
  )
  parser.add_argument('--xi', type=float, default=0.5,
                      help='RPY splitting parameter xi (used by shear_rpy.py).')
  parser.add_argument('--n_steps', type=_parse_int_like, default=30000)
  parser.add_argument('--stress_every', type=_parse_int_like, default=0,
                      help='Set to 0 to disable stress calculation/output.')
  parser.add_argument('--traj_every', type=_parse_int_like, default=100,
                      help='Set to 0 to disable trajectory output.')
  parser.add_argument('--progress_every', type=_parse_int_like, default=1000)
  parser.add_argument(
    '--mr-skin',
    '--mr-dr-threshold',
    dest='mr_skin',
    type=float,
    default=0.5,
    help='Real-space mobility neighbor-list skin (dr_threshold).',
  )
  parser.add_argument(
    '--mr-capacity-multiplier',
    type=float,
    default=None,
    help=(
      'Override real-space mobility neighbor-list capacity multiplier. '
      'If omitted, the internal default is used.'
    ),
  )
  parser.add_argument(
    '--seed',
    dest='seed_values',
    action='append',
    type=_parse_int_like,
    default=None,
  )
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
    dest='init_traj_values',
    action='append',
    type=str,
    default=None,
    help='Optional LAMMPS dump file; initialize from its last complete frame.',
  )
  parser.add_argument(
    '--init-data',
    dest='init_data_values',
    action='append',
    type=str,
    default=None,
    help='Optional LAMMPS data file; initialize from its Atoms section.',
  )
  parser.add_argument(
    '--potential',
    type=str,
    default=None,
    help=(
      'Optional Python module path/name providing pair_potential(dr, **params) '
      'and defaults. If omitted, the run uses zero potential energy.'
    ),
  )
  parser.add_argument(
    '--batch-outdir-naming',
    choices=('auto', 'seed', 'input', 'run'),
    default='auto',
    help=(
      'Batch-only output subdirectory naming. "auto" keeps the current '
      'seed/input-based labels, "seed" uses seed_<seed>, "input" uses the '
      'input filename, and "run" uses run_####.'
    ),
  )
  parser.add_argument(
    '--batch-outdir-run-start',
    type=_parse_int_like,
    default=1,
    help=(
      'Starting run number for --batch-outdir-naming=run. For example, 5 '
      'produces run_0005, run_0006, ...'
    ),
  )

  args = parser.parse_args()
  seed_values = tuple([42] if args.seed_values is None else args.seed_values)
  init_traj_values = tuple(
    [] if args.init_traj_values is None else args.init_traj_values
  )
  init_data_values = tuple(
    [] if args.init_data_values is None else args.init_data_values
  )
  args.seed_values = seed_values
  args.init_traj_values = init_traj_values
  args.init_data_values = init_data_values
  args.seed = int(seed_values[0])
  args.init_traj = init_traj_values[0] if init_traj_values else None
  args.init_data = init_data_values[0] if init_data_values else None
  args.batch_mode = bool(
    len(seed_values) > 1
    or len(init_traj_values) > 1
    or len(init_data_values) > 1
  )
  if args.dt is None:
    raise ValueError('--dt is required.')
  if args.out_dir is None:
    raise ValueError('--out_dir is required.')
  if float(args.dt) <= 0.0:
    raise ValueError('dt must be > 0.')
  if float(args.xi) <= 0.0:
    raise ValueError('xi must be > 0.')
  if args.n_steps <= 0:
    raise ValueError('n_steps must be > 0.')
  if args.peclet < 0.0:
    raise ValueError('peclet must be >= 0.')
  if args.stress_every < 0:
    raise ValueError('stress_every must be >= 0.')
  if args.traj_every < 0:
    raise ValueError('traj_every must be >= 0.')
  if args.progress_every < 0:
    raise ValueError('progress_every must be >= 0.')
  if args.mr_skin < 0.0:
    raise ValueError('mr_skin must be >= 0.')
  if args.mr_capacity_multiplier is not None and args.mr_capacity_multiplier <= 0.0:
    raise ValueError('mr_capacity_multiplier must be > 0 when provided.')
  if args.batch_outdir_run_start <= 0:
    raise ValueError('batch_outdir_run_start must be > 0.')
  if args.init_traj_values and args.init_data_values:
    raise ValueError('--init-traj and --init-data cannot be used together.')
  if args.init_traj is not None or args.init_data is not None:
    if args.n_particles is not None or args.phi is not None:
      raise ValueError(
        'When --init-traj or --init-data is provided, do not pass '
        '--n_particles or --phi. These are derived from the input file.'
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
    _CONSOLE.warn(
      f'--phi={args.phi} exceeds hard-sphere close packing (~0.74). '
      'This can lead to severe overlaps and NaNs in the RPY mobility.'
    )
  return args


def build_internal_config():
  """Algorithmic defaults intentionally kept out of the user-facing CLI."""
  return {
    # Physics + integrator
    'a': 1.0,
    'kT': 1.0,
    'viscosity': 1.0 / (6.0 * math.pi),
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


def resolve_runtime_settings(args, internal_cfg) -> dict:
  """Builds validated runtime settings for RPY runs."""
  a = float(internal_cfg['a'])
  kT = float(internal_cfg['kT'])
  viscosity = float(internal_cfg['viscosity'])
  dt = float(args.dt)
  mr_iters = int(internal_cfg['mr_iters'])
  tol = float(internal_cfg['tol'])
  xi_override = float(args.xi)

  mr_neighbor_format = str(internal_cfg['mr_neighbor_format'])
  mr_dr_threshold = float(args.mr_skin)
  mr_capacity_multiplier = (
    float(args.mr_capacity_multiplier)
    if args.mr_capacity_multiplier is not None
    else float(internal_cfg['mr_capacity_multiplier'])
  )
  real_space_mode = str(internal_cfg['real_space_mode'])

  relax_steps = int(internal_cfg['relax_steps'])
  relax_neighbor_format = str(internal_cfg['relax_neighbor_format'])
  relax_neighbor_dr_threshold = float(internal_cfg['relax_neighbor_dr_threshold'])
  relax_neighbor_capacity_multiplier = float(
    internal_cfg['relax_neighbor_capacity_multiplier']
  )

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
    raise ValueError('mr_capacity_multiplier must be > 0.')
  if real_space_mode not in ('auto', 'min_image', 'lattice'):
    raise ValueError(
      "internal default real_space_mode must be one of 'auto', 'min_image', or 'lattice'."
    )
  if relax_steps < 0:
    raise ValueError('internal default relax_steps must be >= 0.')
  if relax_neighbor_dr_threshold < 0.0:
    raise ValueError('internal default relax_neighbor_dr_threshold must be >= 0.')
  if relax_neighbor_capacity_multiplier <= 0.0:
    raise ValueError('internal default relax_neighbor_capacity_multiplier must be > 0.')

  return {
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
    'real_space_mode': real_space_mode,
    'relax_steps': relax_steps,
    'relax_neighbor_format': relax_neighbor_format,
    'relax_neighbor_dr_threshold': relax_neighbor_dr_threshold,
    'relax_neighbor_capacity_multiplier': relax_neighbor_capacity_multiplier,
  }
