"""Initialization, dump parsing, and overlap relaxation utilities."""

import math
import os
import re

import numpy as np

from rpy_console import get_console

_CONSOLE = get_console()


def _box_size_from_phi(n_particles: int, radius: float, phi: float, dim: int = 3) -> float:
  """Computes cubic/square box size from target packing fraction."""
  if dim == 2:
    area = n_particles * np.pi * radius ** 2 / phi
    return float(np.sqrt(area))
  if dim == 3:
    volume = n_particles * (4.0 / 3.0) * np.pi * radius ** 3 / phi
    return float(volume ** (1.0 / 3.0))
  raise ValueError(f'Unsupported dim={dim}; RPY currently expects 3D.')

def _build_unwrapped_xy_box_fn(base_box: np.ndarray, shear_rate: float):
  """Returns H(t) with unwrapped gamma(t)=shear_rate*t in xy for dump output."""
  base_arr = np.asarray(base_box, dtype=float)
  if base_arr.ndim == 1:
    base_arr = np.diag(base_arr)

  def _box_fn(t: float = 0.0):
    gamma_xy = float(shear_rate) * float(t)
    h = np.array(base_arr, copy=True)
    h[0, 1] = base_arr[0, 1] + gamma_xy * base_arr[1, 1]
    return h

  return _box_fn

def _build_reduced_xy_box_fn(base_box: np.ndarray, shear_rate: float):
  """Returns H(t) with remapped gamma in [-0.5, 0.5) for dump box output."""
  base_arr = np.asarray(base_box, dtype=float)
  if base_arr.ndim == 1:
    base_arr = np.diag(base_arr)

  def _box_fn(t: float = 0.0):
    gamma_xy = float(shear_rate) * float(t)
    gamma_xy = gamma_xy - math.floor(gamma_xy + 0.5)
    h = np.array(base_arr, copy=True)
    h[0, 1] = base_arr[0, 1] + gamma_xy * base_arr[1, 1]
    return h

  return _box_fn

_LAMMPS_FLOAT_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$')


def _strip_inline_comment(line: str) -> str:
  if '#' not in line:
    return line
  return line.split('#', 1)[0]


def _looks_like_int(token: str) -> bool:
  try:
    int(token)
    return True
  except ValueError:
    return False


def _parse_int(token: str, *, field: str) -> int:
  try:
    return int(token)
  except ValueError:
    try:
      val = float(token)
      if not math.isfinite(val) or (not float(val).is_integer()):
        raise ValueError
      return int(val)
    except ValueError as err:
      raise ValueError(f'Invalid integer token for {field}: {token!r}.') from err


def _parse_float(token: str, *, field: str) -> float:
  tok = token.replace('D', 'e').replace('d', 'e')
  try:
    val = float(tok)
  except ValueError as err:
    raise ValueError(f'Invalid float token for {field}: {token!r}.') from err
  if not math.isfinite(val):
    raise ValueError(f'Expected finite float for {field}, got {token!r}.')
  return val


def _is_lammps_data_section_header(line: str) -> bool:
  stripped = _strip_inline_comment(line).strip()
  if not stripped:
    return False
  first = stripped.split()[0]
  if _LAMMPS_FLOAT_RE.match(first):
    return False
  # Data-section headers are title-like identifiers ("Atoms", "Bonds", ...).
  return first[0].isalpha()


def _extract_atom_xyz_tokens(tokens, atom_style: str, path: str):
  style_to_start = {
    'atomic': 2,
    'charge': 3,
    'molecular': 3,
    'bond': 3,
    'angle': 3,
    'full': 4,
    'sphere': 4,
    'ellipsoid': 4,
    'line': 4,
    'tri': 4,
  }

  if atom_style in style_to_start:
    start = style_to_start[atom_style]
    if len(tokens) >= start + 3:
      return (
        _parse_float(tokens[start], field=f'{path}: atom x'),
        _parse_float(tokens[start + 1], field=f'{path}: atom y'),
        _parse_float(tokens[start + 2], field=f'{path}: atom z'),
      )

  end = len(tokens)
  if len(tokens) >= 8 and all(_looks_like_int(tok) for tok in tokens[-3:]):
    end = len(tokens) - 3

  for i in range(end - 3, 0, -1):
    triple = tokens[i:i + 3]
    if len(triple) < 3:
      continue
    try:
      return (
        _parse_float(triple[0], field=f'{path}: atom x'),
        _parse_float(triple[1], field=f'{path}: atom y'),
        _parse_float(triple[2], field=f'{path}: atom z'),
      )
    except ValueError:
      continue

  raise ValueError(
    f'{path}: could not infer x/y/z coordinates from Atoms row: {" ".join(tokens)}'
  )


def _parse_lammps_data(path: str, *, dim: int):
  if dim not in (2, 3):
    raise ValueError(f'Unsupported dim={dim} for LAMMPS data parsing.')
  if not os.path.isfile(path):
    raise ValueError(f'--init-data file does not exist: {path}')

  with open(path, 'r') as handle:
    raw_lines = handle.readlines()

  n_particles = None
  xlo = xhi = ylo = yhi = zlo = zhi = None
  xy = xz = yz = 0.0
  atoms_header_idx = None
  atom_style = ''

  for idx, raw in enumerate(raw_lines):
    stripped = _strip_inline_comment(raw).strip()
    if not stripped:
      continue
    tokens = stripped.split()

    if len(tokens) >= 2 and tokens[1].lower() == 'atoms':
      n_particles = _parse_int(tokens[0], field=f'{path}: atom count')
      continue
    if len(tokens) >= 4 and tokens[2] == 'xlo' and tokens[3] == 'xhi':
      xlo = _parse_float(tokens[0], field=f'{path}: xlo')
      xhi = _parse_float(tokens[1], field=f'{path}: xhi')
      continue
    if len(tokens) >= 4 and tokens[2] == 'ylo' and tokens[3] == 'yhi':
      ylo = _parse_float(tokens[0], field=f'{path}: ylo')
      yhi = _parse_float(tokens[1], field=f'{path}: yhi')
      continue
    if len(tokens) >= 4 and tokens[2] == 'zlo' and tokens[3] == 'zhi':
      zlo = _parse_float(tokens[0], field=f'{path}: zlo')
      zhi = _parse_float(tokens[1], field=f'{path}: zhi')
      continue
    if len(tokens) >= 6 and tokens[3] == 'xy' and tokens[4] == 'xz' and tokens[5] == 'yz':
      xy = _parse_float(tokens[0], field=f'{path}: xy')
      xz = _parse_float(tokens[1], field=f'{path}: xz')
      yz = _parse_float(tokens[2], field=f'{path}: yz')
      continue

    first = tokens[0]
    if first == 'Atoms':
      atoms_header_idx = idx
      if '#' in raw:
        atom_style = raw.split('#', 1)[1].strip().split()[0].lower()
      break

  if n_particles is None:
    raise ValueError(f'{path}: missing "<N> atoms" header.')
  if n_particles <= 0:
    raise ValueError(f'{path}: atom count must be > 0, got {n_particles}.')
  if atoms_header_idx is None:
    raise ValueError(f'{path}: missing "Atoms" section.')
  if None in (xlo, xhi, ylo, yhi, zlo, zhi):
    raise ValueError(f'{path}: missing one or more box-bound headers.')

  lx = float(xhi - xlo)
  ly = float(yhi - ylo)
  lz = float(zhi - zlo)
  if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
    raise ValueError(
      f'{path}: invalid box lengths (lx={lx}, ly={ly}, lz={lz}); expected all > 0.'
    )

  box_matrix = np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]], dtype=float)
  origin = np.array([xlo, ylo, zlo], dtype=float)

  ids = np.empty((n_particles,), dtype=np.int64)
  pos = np.empty((n_particles, 3), dtype=float)
  n_seen = 0
  for raw in raw_lines[atoms_header_idx + 1:]:
    stripped = _strip_inline_comment(raw).strip()
    if not stripped:
      continue
    tokens = stripped.split()
    if not _looks_like_int(tokens[0]):
      if _is_lammps_data_section_header(raw):
        break
      raise ValueError(f'{path}: malformed Atoms row: {raw.rstrip()}')
    if n_seen >= n_particles:
      break
    ids[n_seen] = _parse_int(tokens[0], field=f'{path}: atom id')
    px, py, pz = _extract_atom_xyz_tokens(tokens, atom_style, path)
    pos[n_seen, 0] = px
    pos[n_seen, 1] = py
    pos[n_seen, 2] = pz
    n_seen += 1

  if n_seen != n_particles:
    raise ValueError(
      f'{path}: Atoms section has {n_seen} entries, expected {n_particles}.'
    )

  order = np.argsort(ids)
  ids = ids[order]
  pos = pos[order]

  return {
    'n_particles': int(n_particles),
    'ids': ids,
    'positions_real': pos[:, :dim].copy(),
    'box_matrix': box_matrix[:dim, :dim].copy(),
    'origin': origin[:dim].copy(),
    'atom_style': atom_style,
  }


def _load_initial_state_from_data(path: str, *, dim: int, radius: float):
  """Loads initial positions/box from a LAMMPS data file and derives N and phi."""
  parsed = _parse_lammps_data(path, dim=dim)
  n_particles = int(parsed['n_particles'])
  box_matrix = np.asarray(parsed['box_matrix'], dtype=float)
  origin = np.asarray(parsed['origin'], dtype=float)
  pos_real = np.asarray(parsed['positions_real'], dtype=float)
  if box_matrix.shape != (dim, dim):
    raise ValueError(
      f'Unexpected data-file box shape {box_matrix.shape}; expected {(dim, dim)}.'
    )
  volume = float(abs(np.linalg.det(box_matrix)))
  if volume <= 0.0 or (not math.isfinite(volume)):
    raise ValueError(f'Invalid data-file box determinant: {volume}.')
  phi = n_particles * (4.0 / 3.0) * np.pi * (float(radius) ** 3) / volume
  shifted = pos_real - origin[None, :]
  frac = np.mod(shifted @ np.linalg.inv(box_matrix).T, 1.0)
  return {
    'n_particles': n_particles,
    'phi': float(phi),
    'box_matrix': box_matrix,
    'positions_fractional': frac,
    'source_timestep': None,
    'n_complete_frames': 1,
    'truncated_tail': False,
    'atom_style': parsed.get('atom_style', ''),
  }


def _parse_last_complete_dump_frame(path: str, dim: int):
  """Returns the last complete frame from a LAMMPS text dump."""
  if dim not in (2, 3):
    raise ValueError(f'Unsupported dim={dim} for dump parsing.')
  if not os.path.isfile(path):
    raise ValueError(f'--init-traj file does not exist: {path}')

  last = None
  n_complete = 0
  truncated_tail = False

  with open(path, 'r') as handle:
    while True:
      line = handle.readline()
      if not line:
        break
      if not line.startswith('ITEM: TIMESTEP'):
        continue

      timestep_line = handle.readline()
      if not timestep_line:
        truncated_tail = True
        break
      timestep = int(float(timestep_line.strip()))

      n_header = handle.readline()
      if not n_header.startswith('ITEM: NUMBER OF ATOMS'):
        raise ValueError(f'{path}: missing "ITEM: NUMBER OF ATOMS".')
      n_line = handle.readline()
      if not n_line:
        truncated_tail = True
        break
      n_particles = int(float(n_line.strip()))

      bounds_header = handle.readline()
      if not bounds_header.startswith('ITEM: BOX BOUNDS'):
        raise ValueError(f'{path}: missing "ITEM: BOX BOUNDS".')
      b1 = handle.readline()
      b2 = handle.readline()
      b3 = handle.readline()
      if not (b1 and b2 and b3):
        truncated_tail = True
        break
      box_matrix = _parse_box_matrix_from_bounds(bounds_header, b1, b2, b3)

      atoms_header = handle.readline()
      if not atoms_header.startswith('ITEM: ATOMS'):
        raise ValueError(f'{path}: missing "ITEM: ATOMS".')
      columns = atoms_header.strip().split()[2:]
      required = ['id', 'x', 'y'] if dim == 2 else ['id', 'x', 'y', 'z']
      if not all(col in columns for col in required):
        raise ValueError(
          f'{path}: ATOMS header must include {required}, got {columns}.'
        )

      i_id = columns.index('id')
      i_x = columns.index('x')
      i_y = columns.index('y')
      i_z = columns.index('z') if dim == 3 else None
      needed = max([i_id, i_x, i_y] + ([i_z] if i_z is not None else []))

      ids = np.empty((n_particles,), dtype=np.int64)
      pos = np.empty((n_particles, dim), dtype=float)
      malformed = False
      for k in range(n_particles):
        row = handle.readline()
        if not row:
          malformed = True
          break
        parts = row.split()
        if len(parts) <= needed:
          malformed = True
          break
        ids[k] = int(parts[i_id])
        pos[k, 0] = float(parts[i_x])
        pos[k, 1] = float(parts[i_y])
        if dim == 3:
          pos[k, 2] = float(parts[i_z])

      if malformed:
        truncated_tail = True
        break

      order = np.argsort(ids)
      ids = ids[order]
      pos = pos[order]
      last = {
        'timestep': timestep,
        'n_particles': n_particles,
        'ids': ids,
        'positions_real': pos,
        'box_matrix': box_matrix,
      }
      n_complete += 1

  if last is None:
    raise ValueError(f'No complete frames were found in dump: {path}')
  return last, n_complete, truncated_tail

def _parse_box_matrix_from_bounds(bounds_header: str, b1: str, b2: str, b3: str) -> np.ndarray:
  """Parses LAMMPS BOX BOUNDS (+tilts) into an upper-triangular box matrix."""
  tokens = bounds_header.strip().split()
  triclinic = ('xy' in tokens) or ('xz' in tokens) or ('yz' in tokens)
  p1 = [float(x) for x in b1.split()]
  p2 = [float(x) for x in b2.split()]
  p3 = [float(x) for x in b3.split()]
  if not triclinic:
    xlo, xhi = p1[:2]
    ylo, yhi = p2[:2]
    zlo, zhi = p3[:2]
    return np.array(
      [[xhi - xlo, 0.0, 0.0], [0.0, yhi - ylo, 0.0], [0.0, 0.0, zhi - zlo]],
      dtype=float,
    )

  if len(p1) < 3 or len(p2) < 3 or len(p3) < 3:
    raise ValueError('Triclinic BOX BOUNDS requires x/y/z tilt factors.')
  xlo_b, xhi_b, xy = p1[:3]
  ylo_b, yhi_b, xz = p2[:3]
  zlo_b, zhi_b, yz = p3[:3]
  x_correction = max(0.0, xy, xz, xy + xz) - min(0.0, xy, xz, xy + xz)
  y_correction = max(0.0, yz) - min(0.0, yz)
  lx = (xhi_b - xlo_b) - x_correction
  ly = (yhi_b - ylo_b) - y_correction
  lz = zhi_b - zlo_b
  return np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]], dtype=float)

def _load_initial_state_from_dump(path: str, *, dim: int, radius: float):
  """Loads initial positions/box from dump and derives N and phi."""
  frame, n_complete, truncated_tail = _parse_last_complete_dump_frame(path, dim)
  n_particles = int(frame['n_particles'])
  box_matrix = np.asarray(frame['box_matrix'], dtype=float)
  pos_real = np.asarray(frame['positions_real'], dtype=float)
  if box_matrix.shape != (dim, dim):
    raise ValueError(
      f'Unexpected dump box shape {box_matrix.shape}; expected {(dim, dim)}.'
    )
  volume = float(abs(np.linalg.det(box_matrix)))
  if volume <= 0.0 or (not math.isfinite(volume)):
    raise ValueError(f'Invalid dump box determinant: {volume}.')
  phi = n_particles * (4.0 / 3.0) * np.pi * (float(radius) ** 3) / volume
  frac = np.mod(pos_real @ np.linalg.inv(box_matrix).T, 1.0)
  return {
    'n_particles': n_particles,
    'phi': float(phi),
    'box_matrix': box_matrix,
    'positions_fractional': frac,
    'source_timestep': int(frame['timestep']),
    'n_complete_frames': int(n_complete),
    'truncated_tail': bool(truncated_tail),
  }

def _relax_positions(
  R_init,
  displacement_0,
  shift_0,
  diameter: float,
  box,
  steps: int,
  neighbor_format,
  neighbor_capacity_multiplier: float,
  dr_threshold: float,
):
  """Relaxes initial positions to remove large overlaps before RPY dynamics."""
  if steps <= 0:
    return R_init
  import jax
  import jax.numpy as jnp
  from jax import lax
  from jax_md import energy
  from jax_md import minimize

  neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(
    displacement_0,
    box,
    sigma=diameter,
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

  @jax.jit
  def _run(state_in, neighbor_in):
    def _step(_, carry):
      s, n, overflow = carry
      n = n.update(s.position, box=box)
      overflow = jnp.logical_or(overflow, n.did_buffer_overflow)
      s = apply_min(s, neighbor=n)
      return s, n, overflow
    return lax.fori_loop(0, steps, _step, (state_in, neighbor_in, False))

  state, _, overflow = _run(state, neighbor)
  if bool(np.asarray(overflow)):
    _CONSOLE.warn('overlap-relaxation neighbor list overflow detected.')
  return state.position

def _min_pair_distance(
  R,
  displacement_fn,
) -> float:
  """Computes the minimum pair distance (O(N^2), used once for diagnostics)."""
  import jax
  import jax.numpy as jnp
  n = int(R.shape[0])
  if n < 2:
    return 0.0
  i_idx, j_idx = jnp.triu_indices(n, 1)
  dR = jax.vmap(displacement_fn)(R[i_idx], R[j_idx])
  dist = jnp.sqrt(jnp.sum(dR * dR, axis=-1))
  return float(np.asarray(jnp.min(dist)))
