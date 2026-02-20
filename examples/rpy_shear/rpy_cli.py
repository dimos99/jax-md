"""CLI parsing and static config for RPY shear."""

import argparse
import math

from rpy_console import get_console

_CONSOLE = get_console()


def parse_args():
  parser = argparse.ArgumentParser(
    description='RPY shear runner with chunked stress/trajectory dumping.')

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
  parser.add_argument('--xi', type=float, default=0.5,
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
  parser.add_argument('--stress_every', type=int, default=0,
                      help='Set to 0 to disable stress calculation/output.')
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

  args = parser.parse_args()
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
  if args.stress_every < 0:
    raise ValueError('stress_every must be >= 0.')
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
