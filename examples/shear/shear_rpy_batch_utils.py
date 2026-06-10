"""Internal helpers for batched RPY shear runs."""

import os
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from shear_console import get_console
from shear_prepare_utils import _build_initial_positions
from shear_prepare_utils import _resolve_initial_system

_CONSOLE = get_console()


@dataclass(frozen=True)
class _RunSpec:
  label: str
  out_dir: str
  seed: int
  init_traj: str | None
  init_data: str | None


@dataclass(frozen=True)
class _PreparedBatchRun:
  spec: _RunSpec
  args: object
  init_mode: str
  dump_info: dict | None
  data_info: dict | None
  n_particles: int
  phi: float
  base_box: jnp.ndarray
  R0: jnp.ndarray
  run_key: jnp.ndarray
  min_dist: float


def _slugify_label_component(value: str) -> str:
  """Returns a filesystem-safe label fragment."""
  label = re.sub(r'[^A-Za-z0-9_]+', '-', value).strip('-')
  return label or 'run'


def _build_input_label(
  *,
  input_kind: str,
  input_value: str,
  run_index: int,
  n_runs: int,
) -> str:
  """Builds an input-based batch label."""
  stem = os.path.splitext(os.path.basename(input_value))[0]
  stem = _slugify_label_component(stem)
  prefix = f'{input_kind}_{run_index + 1:03d}' if n_runs > 1 else input_kind
  return '__'.join((prefix, stem))


def _format_run_label(run_number: int) -> str:
  """Formats a sequential run label with stable zero padding."""
  return f'run_{run_number:04d}'


def resolve_batch_run_specs(args) -> tuple[_RunSpec, ...]:
  """Builds per-run specs from repeated seed/init CLI arguments."""
  seed_values = tuple(int(seed) for seed in args.seed_values)
  init_data_values = tuple(str(path) for path in args.init_data_values)
  init_traj_values = tuple(str(path) for path in args.init_traj_values)
  naming = str(getattr(args, 'batch_outdir_naming', 'auto'))
  run_start = int(getattr(args, 'batch_outdir_run_start', 1))
  if init_data_values and init_traj_values:
    raise ValueError(
      'Batch mode requires choosing either repeated --init-data or repeated '
      '--init-traj, not both.'
    )
  if run_start <= 0:
    raise ValueError('batch_outdir_run_start must be > 0.')

  input_kind = None
  input_values: tuple[str, ...] = ()
  if init_data_values:
    input_kind = 'data'
    input_values = init_data_values
  elif init_traj_values:
    input_kind = 'traj'
    input_values = init_traj_values

  if naming == 'input' and input_kind is None:
    raise ValueError(
      '--batch-outdir-naming=input requires --init-data or --init-traj.'
    )

  if len(seed_values) > 1 and len(input_values) > 1:
    if len(seed_values) != len(input_values):
      raise ValueError(
        'Repeated --seed and repeated init files must have equal lengths.'
      )

  n_runs = max(len(seed_values), len(input_values), 1)
  specs = []
  label_counts = {}
  for run_index in range(n_runs):
    seed = seed_values[run_index] if len(seed_values) > 1 else seed_values[0]
    input_value = None
    if input_values:
      input_value = (
        input_values[run_index]
        if len(input_values) > 1
        else input_values[0]
      )

    if naming == 'seed':
      base_label = f'seed_{seed}'
    elif naming == 'input':
      base_label = _build_input_label(
        input_kind=input_kind,
        input_value=input_value,
        run_index=run_index,
        n_runs=n_runs,
      )
    elif naming == 'run':
      base_label = _format_run_label(run_start + run_index)
    else:
      label_parts = []
      if input_kind is not None and input_value is not None:
        label_parts.append(_build_input_label(
          input_kind=input_kind,
          input_value=input_value,
          run_index=run_index,
          n_runs=len(input_values),
        ))
      if len(seed_values) > 1 or input_kind is None:
        label_parts.append(f'seed_{seed}')
      base_label = '__'.join(label_parts)

    count = label_counts.get(base_label, 0) + 1
    label_counts[base_label] = count
    label = (
      base_label if count == 1 else f'{base_label}__run_{count:03d}'
    )
    specs.append(
      _RunSpec(
        label=label,
        out_dir=os.path.join(args.out_dir, label),
        seed=int(seed),
        init_traj=input_value if input_kind == 'traj' else None,
        init_data=input_value if input_kind == 'data' else None,
      )
    )
  return tuple(specs)


def _clone_args_for_run(args, run_spec: _RunSpec):
  """Copies CLI args and applies one run's seed/input/output overrides."""
  run_args = type(args)(**vars(args))
  run_args.seed = int(run_spec.seed)
  run_args.init_traj = run_spec.init_traj
  run_args.init_data = run_spec.init_data
  run_args.out_dir = run_spec.out_dir
  run_args.seed_values = (int(run_spec.seed),)
  run_args.init_traj_values = (
    () if run_spec.init_traj is None else (str(run_spec.init_traj),)
  )
  run_args.init_data_values = (
    () if run_spec.init_data is None else (str(run_spec.init_data),)
  )
  run_args.batch_mode = False
  return run_args


def _resolve_batch_initial_systems(
  args,
  *,
  packing_radius: float,
  dim: int,
  console=None,
):
  """Resolves one initial system per batch member and validates same-shape input."""
  log = _CONSOLE if console is None else console
  run_specs = resolve_batch_run_specs(args)
  resolved_runs = []
  common_box = None
  common_n_particles = None
  for run_spec in run_specs:
    run_args = _clone_args_for_run(args, run_spec)
    initial_system = _resolve_initial_system(
      run_args, a=packing_radius, dim=dim, console=log)
    candidate_box = np.asarray(initial_system['base_box'], dtype=float)
    candidate_n_particles = int(initial_system['n_particles'])
    if common_box is None:
      common_box = candidate_box
      common_n_particles = candidate_n_particles
    else:
      if candidate_n_particles != common_n_particles:
        raise ValueError(
          'Batch mode requires the same particle count across all runs, '
          f'but {run_spec.label} has n_particles={candidate_n_particles} '
          f'instead of {common_n_particles}.'
        )
      if (
        candidate_box.shape != common_box.shape
        or not np.allclose(candidate_box, common_box, rtol=1e-12, atol=1e-12)
      ):
        raise ValueError(
          'Batch mode requires the same box matrix across all runs, '
          f'but {run_spec.label} has a different box.'
        )
    resolved_runs.append((run_spec, run_args, initial_system))
  return run_specs, tuple(resolved_runs)


def _prepare_batch_runs(
  resolved_runs,
  *,
  base_box,
  n_particles: int,
  dim: int,
  diameter: float,
  displacement_0,
  shift_0,
  relax_steps: int,
  relax_neighbor_format: str,
  relax_neighbor_capacity_multiplier: float,
  relax_neighbor_dr_threshold: float,
  format_map: dict,
):
  """Builds per-run initial positions and batch run descriptors."""
  prepared_runs = []
  for run_spec, run_args, initial_system in resolved_runs:
    initial_positions = _build_initial_positions(
      init_mode=initial_system['init_mode'],
      dump_info=initial_system['dump_info'],
      data_info=initial_system['data_info'],
      base_box=base_box,
      n_particles=n_particles,
      dim=dim,
      seed=run_args.seed,
      diameter=diameter,
      displacement_0=displacement_0,
      shift_0=shift_0,
      relax_steps=relax_steps,
      relax_neighbor_format=relax_neighbor_format,
      relax_neighbor_capacity_multiplier=relax_neighbor_capacity_multiplier,
      relax_neighbor_dr_threshold=relax_neighbor_dr_threshold,
      format_map=format_map,
    )
    prepared_runs.append(
      _PreparedBatchRun(
        spec=run_spec,
        args=run_args,
        init_mode=initial_system['init_mode'],
        dump_info=initial_system['dump_info'],
        data_info=initial_system['data_info'],
        n_particles=n_particles,
        phi=float(initial_system['phi']),
        base_box=base_box,
        R0=initial_positions['R0'],
        run_key=initial_positions['run_key'],
        min_dist=float(np.asarray(initial_positions['min_dist'])),
      )
    )
  return tuple(prepared_runs)


def _stack_optional_arrays(values):
  """Stacks arrays or returns None when every value is None."""
  if all(value is None for value in values):
    return None
  if any(value is None for value in values):
    raise ValueError('Expected optional batch values to be all None or all arrays.')
  return jnp.stack(values, axis=0)


def _pad_neighbor_idx(neighbor, *, target_width: int):
  """Pads neighbor-list indices to a shared width using the invalid-id sentinel."""
  current_width = int(neighbor.idx.shape[-1])
  if current_width == target_width:
    return neighbor.idx
  if current_width > target_width:
    raise ValueError(
      f'Expected target_width >= {current_width}, got {target_width}.'
    )
  invalid_id = int(np.asarray(neighbor.reference_position).shape[0])
  pad_shape = neighbor.idx.shape[:-1] + (target_width - current_width,)
  padding = jnp.full(pad_shape, invalid_id, dtype=neighbor.idx.dtype)
  return jnp.concatenate([neighbor.idx, padding], axis=-1)


def _stack_neighbor_lists(neighbors):
  """Stacks neighbor-list dynamic fields with shared static capacities."""
  if all(neighbor is None for neighbor in neighbors):
    return None
  if any(neighbor is None for neighbor in neighbors):
    raise ValueError(
      'Expected neighbor lists to either all be present or all be absent.'
    )

  template = neighbors[0]
  max_occupancy = max(int(neighbor.max_occupancy) for neighbor in neighbors)
  cell_list_capacity = [neighbor.cell_list_capacity for neighbor in neighbors]
  if all(capacity is None for capacity in cell_list_capacity):
    shared_cell_capacity = None
  elif any(capacity is None for capacity in cell_list_capacity):
    raise ValueError(
      'Expected neighbor lists to either all use cell lists or all disable them.'
    )
  else:
    shared_cell_capacity = max(int(capacity) for capacity in cell_list_capacity)
  return template.set(
    idx=jnp.stack(
      [
        _pad_neighbor_idx(neighbor, target_width=max_occupancy)
        for neighbor in neighbors
      ],
      axis=0,
    ),
    reference_position=jnp.stack(
      [neighbor.reference_position for neighbor in neighbors], axis=0
    ),
    reference_box=jnp.stack(
      [neighbor.reference_box for neighbor in neighbors], axis=0
    ),
    box_at_build=_stack_optional_arrays(
      [neighbor.box_at_build for neighbor in neighbors]
    ),
    error=template.error.set(
      code=jnp.stack([neighbor.error.code for neighbor in neighbors], axis=0)
    ),
    last_box=jnp.stack([neighbor.last_box for neighbor in neighbors], axis=0),
    cell_list_capacity=shared_cell_capacity,
    max_occupancy=max_occupancy,
  )


def _stack_wave_modes(wave_states):
  """Stacks wave-space mode arrays while preserving a shared key set."""
  template_modes = wave_states[0].modes
  mode_keys = tuple(template_modes.keys())
  for wave_state in wave_states[1:]:
    if tuple(wave_state.modes.keys()) != mode_keys:
      raise ValueError(
        'Expected identical wave-space mode keys across all batched runs.'
      )
  return {
    key: jnp.stack([wave_state.modes[key] for wave_state in wave_states], axis=0)
    for key in mode_keys
  }


def _stack_wave_state(wave_states):
  """Stacks dynamic wave-space fields while reusing template callables."""
  template = wave_states[0]
  try:
    stacked_params = jax.tree_util.tree_map(
      lambda *xs: jnp.stack(xs, axis=0),
      *[wave_state.params for wave_state in wave_states],
    )
  except ValueError as err:
    raise ValueError(
      'Expected identical static wave-space parameters across all batched runs.'
    ) from err
  return template.set(
    params=stacked_params,
    modes=_stack_wave_modes(wave_states),
  )


def _stack_rpy_states(states):
  """Stacks sheared RPY states along a leading batch axis."""
  template = states[0]
  real_template = template.rpy_state.real
  rpy_template = template.rpy_state

  real_neighbors = [state.rpy_state.real.neighbors for state in states]
  stacked_real = real_template.set(
    neighbors=_stack_neighbor_lists(real_neighbors),
    lattice_indices=jnp.stack(
      [state.rpy_state.real.lattice_indices for state in states], axis=0
    ),
    zero_image_index=jnp.stack(
      [state.rpy_state.real.zero_image_index for state in states], axis=0
    ),
    box_matrix=jnp.stack(
      [state.rpy_state.real.box_matrix for state in states], axis=0
    ),
  )
  stacked_wave = _stack_wave_state([state.rpy_state.wave for state in states])
  stacked_rpy_state = rpy_template.set(
    real=stacked_real,
    wave=stacked_wave,
    preconditioner=rpy_template.preconditioner,
  )
  return template.set(
    real_position=jnp.stack([state.real_position for state in states], axis=0),
    integrator_position=jnp.stack(
      [state.integrator_position for state in states], axis=0),
    rpy_state=stacked_rpy_state,
    rng=jnp.stack([state.rng for state in states], axis=0),
    step=jnp.stack([state.step for state in states], axis=0),
    time=jnp.stack([state.time for state in states], axis=0),
    force=jnp.stack([state.force for state in states], axis=0),
  )


def _failed_run_labels(flags, run_specs: tuple[_RunSpec, ...]) -> list[str]:
  """Returns run labels for truthy entries in a boolean array."""
  return [
    run_spec.label
    for run_spec, flag in zip(run_specs, np.asarray(flags, dtype=bool))
    if flag
  ]


def _check_batch_nan_positions(
  state,
  run_specs: tuple[_RunSpec, ...],
  stage: str,
  console=None,
) -> bool:
  """Returns False when any batched run contains NaN integrator positions."""
  log = _CONSOLE if console is None else console
  axes = tuple(range(1, state.integrator_position.ndim))
  failed = _failed_run_labels(
    jnp.any(jnp.isnan(state.integrator_position), axis=axes),
    run_specs,
  )
  if failed:
    log.error(
      f'NaN detected in integrator positions during {stage} for runs: '
      f'{", ".join(failed)}.'
    )
    return False
  return True


def _check_batch_neighbor_status(
  neighbor,
  run_specs: tuple[_RunSpec, ...],
  stage: str,
  label: str,
  console=None,
) -> bool:
  """Returns False when any batched neighbor list reports an error flag."""
  log = _CONSOLE if console is None else console
  if neighbor is None:
    log.error(f'Missing {label} in stage={stage}.')
    return False
  overflow = _failed_run_labels(neighbor.did_buffer_overflow, run_specs)
  if overflow:
    log.error(
      f'{label} overflow in stage={stage} for runs: '
      f'{", ".join(overflow)}.'
    )
    return False
  cell_small = _failed_run_labels(neighbor.cell_size_too_small, run_specs)
  if cell_small:
    log.error(
      f'{label} cell size too small in stage={stage} for runs: '
      f'{", ".join(cell_small)}.'
    )
    return False
  malformed = _failed_run_labels(neighbor.malformed_box, run_specs)
  if malformed:
    log.error(
      f'{label} malformed box in stage={stage} for runs: '
      f'{", ".join(malformed)}.'
    )
    return False
  return True
