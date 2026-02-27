"""Lattice and runtime-selection helpers for deterministic real-space RPY."""

import inspect
import itertools
import warnings

from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_md.hydro.rpy_real_det_helpers import generate_lattice_hypercube


RealSpaceMode = Literal['auto', 'min_image', 'lattice']
_REAL_SPACE_MODES = frozenset({'auto', 'min_image', 'lattice'})


def _validate_real_space_mode(mode: str) -> RealSpaceMode:
  mode_str = str(mode)
  if mode_str not in _REAL_SPACE_MODES:
    raise ValueError(
        f"real_space_mode must be one of 'auto', 'min_image', or 'lattice'; got {mode_str!r}."
    )
  return mode_str  # type: ignore[return-value]


def _neighbor_box_from_matrix(box_matrix: jnp.ndarray,
                              fractional_coordinates: bool) -> jnp.ndarray:
  """Box argument passed to neighbor_list depending on coordinate convention."""
  if fractional_coordinates:
    return box_matrix
  # Neighbor lists in real coordinates disallow `box` kwargs; rely on the space.
  return None


def _is_traced_value(value) -> bool:
  return isinstance(value, jax.core.Tracer)


def _box_fn_supports_shear_kwargs(box_fn: Callable, dim: int) -> bool:
  """Whether ``box_fn`` supports per-plane shear kwargs."""
  try:
    signature = inspect.signature(box_fn)
  except (TypeError, ValueError):
    return False

  parameters = signature.parameters.values()
  if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
    return True

  required = ("gamma_xy",) if dim == 2 else ("gamma_xy", "gamma_xz", "gamma_yz")
  keywordable = {
      name for name, param in signature.parameters.items()
      if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
  }
  return all(name in keywordable for name in required)


def _iter_shear_corner_probes(dim: int):
  """Yield dim-aware per-plane corner probes in bounded shear space."""
  if dim == 2:
    planes = ("gamma_xy",)
  elif dim == 3:
    planes = ("gamma_xy", "gamma_xz", "gamma_yz")
  else:
    return

  # Shearing canonicalizes to f32 and remaps with floor(g + 0.5). Use 0.5-eps so
  # the remap path keeps a strict open-edge value rather than rounding to -0.5.
  hi_open = float(np.float32(0.5) - np.finfo(np.float32).eps)
  probe_values = (-0.5, hi_open, 0.5)
  for values in itertools.product(probe_values, repeat=len(planes)):
    yield dict(zip(planes, values))


def _box_like_to_matrix_np(box_like, dim: int) -> np.ndarray:
  probe_jnp = jnp.asarray(box_like)
  if probe_jnp.ndim == 0:
    return np.diag(np.full(dim, float(probe_jnp), dtype=np.float64))
  if probe_jnp.ndim == 1:
    return np.diag(np.asarray(probe_jnp, dtype=np.float64))
  return np.asarray(probe_jnp, dtype=np.float64)


def _box_min_effective_dim_np(box_np: np.ndarray, dim: int) -> float:
  """Minimum effective box dimension for fractional cell sizing (mirrors partition._fractional_cell_size).

  Returns min(nx, ny, nz) where nx, ny, nz are the effective side-lengths
  that determine how many cell-list cells fit along each direction.
  Matches the formula in partition._fractional_cell_size for 2D and 3D boxes.
  """
  box_np = np.asarray(box_np, dtype=np.float64)
  if box_np.ndim != 2 or box_np.shape[0] != dim:
    return float(np.min(np.abs(np.diag(box_np)))) if box_np.ndim == 2 else float(np.min(np.abs(box_np)))
  if dim == 2:
    xx, yy = float(box_np[0, 0]), float(box_np[1, 1])
    xy = float(box_np[0, 1]) / yy if yy != 0.0 else 0.0
    nx = xx / np.sqrt(1.0 + xy ** 2)
    return min(nx, yy)
  # dim == 3
  xx = float(box_np[0, 0])
  yy = float(box_np[1, 1])
  zz = float(box_np[2, 2])
  xy = float(box_np[0, 1]) / yy if yy != 0.0 else 0.0
  xz = float(box_np[0, 2]) / zz if zz != 0.0 else 0.0
  yz = float(box_np[1, 2]) / zz if zz != 0.0 else 0.0
  nx = xx / np.sqrt(1.0 + xy ** 2 + (xy * yz - xz) ** 2)
  ny = yy / np.sqrt(1.0 + yz ** 2)
  return min(nx, ny, zz)


def _worst_case_shear_neighbor_box(
    box_fn: Callable,
    dim: int,
    current_box_np: np.ndarray,
) -> np.ndarray:
  """Return the box geometry that requires the largest fractional cell size.

  For shearing simulations with ``remap=True`` the reduced shear strain is
  bounded to ``[-0.5, 0.5)``.  Pre-allocating the neighbor list at the most
  deformed box (smallest effective dimension) prevents ``cell_size_too_small``
  errors that would otherwise be raised at runtime once the box becomes skewed
  enough to require fewer cell-list cells per side than were allocated.

  Probes all dim-aware corner combinations over per-plane values:
  ``{-0.5, nextafter(0.5, 0.0), +0.5}`` and returns the box with the smallest
  ``min(nx, ny, nz)``. Including the open-edge ``0.5-`` value keeps probing
  effective for remapped shear definitions that wrap exact ``+0.5`` to ``-0.5``.
  """
  worst_box = np.array(current_box_np, dtype=np.float64)
  worst_min = _box_min_effective_dim_np(worst_box, dim)

  for probe_kw in _iter_shear_corner_probes(dim):
    probe_raw = box_fn(**probe_kw)
    probe_np = _box_like_to_matrix_np(probe_raw, dim)
    if probe_np.shape != worst_box.shape:
      continue
    probe_min = _box_min_effective_dim_np(probe_np, dim)
    if probe_min < worst_min:
      worst_min = probe_min
      worst_box = probe_np

  return worst_box


def _compute_lattice_indices(
    box_matrix: jnp.ndarray,
    *,
    rcut: float,
    lattice_extent: Optional[int],
    lattice_extra: float,
    warn_stacklevel: int = 4,
) -> Tuple[jnp.ndarray, int]:
  """
  Compute lattice vectors for periodic image summation in real-space mobility.

  This function generates the set of integer lattice shifts L = (n_x, n_y, n_z)
  needed to evaluate hydrodynamic interactions across periodic boundaries:

      M^r_ij = ∑_L M^r(r_ij + L·Box)

  The lattice extent is chosen to ensure all images needed for strict
  real-space coverage within r_cut are present.

  Algorithm
  ---------
  1. **Extent Estimation**: If not user-specified, compute the lattice extent N
     from the ratio 2·r_cut / σ_min(Box), where σ_min is the smallest singular
     value of the box matrix. This ensures coverage in the 'thinnest' box direction,
     accounting for neighbor-list minimum-image shifts L_min ∈ {-1,0,1}³.

  2. **Hypercube Generation**: Create a symmetric grid of integer shifts in
     {-N, ..., N}^d, yielding (2N+1)^d candidate lattice vectors.

  3. **Zero Image Preservation**: Ensure the primary cell (L=0) is included and
     track its index for self-interaction masking in the mobility kernel.

  Parameters
  ----------
  box_matrix : (d, d) array
      Box transformation matrix mapping fractional to real coordinates.

  Returns
  -------
  lattice : (n_images, d) array of int32
      Integer lattice vectors to sum over.
  zero_idx : int
      Index of the zero lattice vector (L=0, primary cell) in the returned array.

  Notes
  -----
  - For non-cubic or shearing boxes, the SVD-based extent estimate adapts to the
    box geometry, ensuring sufficient coverage without excessive oversampling.
  - Degenerate boxes (σ_min → 0) are protected by the safe_sigma clamp to 1e-12.
  """
  box_np = np.asarray(box_matrix, dtype=np.float64)
  dim = box_np.shape[0]
  if lattice_extent is None:
    # Compute smallest singular value to estimate 'thinnest' box dimension.
    svals = np.linalg.svd(box_np, compute_uv=False)
    sigma_min = float(np.min(svals))
    safe_sigma = max(sigma_min, 1e-12)
    base_extent = int(np.ceil(2.0 * float(rcut) / safe_sigma))
    extent_padding = int(np.ceil(float(lattice_extra)))
    extent_val = max(base_extent + extent_padding, 0)
    if extent_val > 1:
      warnings.warn(
          f"Real-space lattice extent is {extent_val} (box thinnest direction "
          f"σ_min={safe_sigma:.4g}, rcut={float(rcut):.4g}). "
          f"Each pair will be evaluated over {(2*extent_val+1)**dim} periodic images, "
          f"which may be slow. Consider increasing ξ (xi) to reduce rcut, or "
          f"ensure the box is not too elongated.",
          stacklevel=warn_stacklevel,
      )
  else:
    extent_val = int(lattice_extent)

  # Generate symmetric hypercube: L ∈ {-N, ..., N}^d.
  lattice_np, zero_idx = generate_lattice_hypercube(dim, extent_val)

  lattice = jnp.asarray(lattice_np, dtype=jnp.int32)
  return lattice, zero_idx


def _sigma_min(box_matrix: jnp.ndarray) -> float:
  box_np = np.asarray(box_matrix, dtype=np.float64)
  svals = np.linalg.svd(box_np, compute_uv=False)
  return max(float(np.min(svals)), 1e-12)


def _is_min_image_safe(box_matrix: jnp.ndarray, *, rcut: float) -> bool:
  return float(rcut) <= 0.5 * _sigma_min(box_matrix)


def _select_real_space_core(
    box_matrix: jnp.ndarray,
    *,
    mode: RealSpaceMode,
    core_lattice: Callable,
    core_min_image: Callable,
    rcut: float,
) -> Tuple[Callable, str]:
  is_tracer = isinstance(box_matrix, jax.core.Tracer)

  if mode == 'lattice':
    return core_lattice, 'lattice'

  if mode == 'min_image':
    if is_tracer:
      raise ValueError(
          "real_space_mode='min_image' with traced/dynamic box is unsupported "
          "because safety cannot be verified at trace time."
      )
    if not _is_min_image_safe(box_matrix, rcut=rcut):
      sigma_min = _sigma_min(box_matrix)
      raise ValueError(
          "real_space_mode='min_image' requires rcut <= 0.5 * sigma_min(box). "
          f"Got rcut={float(rcut):.6g}, 0.5*sigma_min={0.5 * sigma_min:.6g}."
      )
    return core_min_image, 'min_image'

  # mode == 'auto'
  if is_tracer:
    return core_lattice, 'lattice'
  if _is_min_image_safe(box_matrix, rcut=rcut):
    return core_min_image, 'min_image'
  return core_lattice, 'lattice'
