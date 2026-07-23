"""Near-field lubrication resistance table: loading and interpolation.

Loads the committed ``data/resistance_table.npz``, regenerated from the
Townsend and Wilson expressions by ``data/generate_resistance_table.py``,
and provides a pure-JAX, vmappable scalar lookup using FSD's interpolation
scheme (``Lubrication.cu:140-238``).

The 22 monodisperse scalar functions are *already* the near-field part (exact
Jeffrey-Onishi minus the two-body far-field multipole), tabulated against the
center-to-center distance ``s = r/a`` and indexed in log-space of the surface
gap ``xi = s - 2``.

Column order (see :data:`COLUMN_NAMES`)::

    [ XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12,
      XG11, XG12, YG11, YG12, YH11, YH12,
      XM11, XM12, YM11, YM12, ZM11, ZM12 ]

Interpolation contract (must match FSD bit-for-bit; the transcription test
cannot catch an error here, so it is pinned to the source):

  * ``s = r / a`` and ``xi = s - 2``.
  * Lower row index in log-space, floored using the table's ``xi_min`` and
    ``dr`` metadata:
    ``ind = floor(log10(xi / xi_min) / dr)``  (C++ truncates the float cast;
    ``floor`` matches for the positive argument), then clipped to
    ``[REGULARIZATION_INDEX, N_DIST - 2]`` so ``ind`` and ``ind + 1`` are
    valid.
  * Lerp weight in *raw distance* using the stored tabulated distances:
    ``fac = (s - dist[ind]) / (dist[ind + 1] - dist[ind])``, clipped to [0, 1].
  * Result: ``vals[ind] + fac * (vals[ind + 1] - vals[ind])``.

Unlike FSD's roughness-regularized production path, float64 uses the full
committed table (``REGULARIZATION_INDEX = 0``). Float32 clamps at the first row
whose gap is at least ``1e-5`` because smaller center-to-center increments are
poorly resolved at ``s ~= 2``. Approaching ``s = 4``, the weight ``fac -> 1``
lerps toward the all-zero final row.

The environment variable ``JAX_MD_SD_MIN_GAP`` (a surface gap in units of
``a``, e.g. ``1e-4``) raises the clamp row above the precision default: pairs
closer than that gap take the resistance at the clamp row, exactly like FSD's
roughness regularization. It caps the near-contact stiffness (``XA11 ~
1/(4 xi)``), which bounds the saddle conditioning -- and therefore GMRES
iteration counts -- on contact-rich configurations. It never lowers the clamp
below the float32 floor and must select a row that has a following row for
interpolation; non-finite values and gaps beyond that range are rejected. Like
the dtype, it is read once at import: set it before ``jax_md.hydro`` is first
imported.

The lubrication cutoff itself (``r < r_lub = 4a``) is enforced by the caller's
neighbor mask, never by this table.

The 2000 rows span gaps ``1e-8..2``. Townsend's (2023) corrected
Jeffrey-Onishi expressions cover gaps through ``0.01``, a slope-limited bridge
joins Wilson's (2013) Lamb/reflection solution at ``0.02``, and the two-body FTS
far-field resistance is subtracted throughout.

The float64 clamp makes near-contact resistance up to ``1e4``x stiffer than the
legacy table for overlapping / deeply contacting pairs (XA11 ~ 1/(4 xi)),
which increases saddle GMRES iteration counts on contact-rich configurations.
The float32 clamp avoids relying on table spacing below its useful precision.
"""

import os
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE


COLUMN_NAMES = (
    'XA11', 'XA12', 'YA11', 'YA12', 'YB11', 'YB12', 'XC11', 'XC12', 'YC11',
    'YC12', 'XG11', 'XG12', 'YG11', 'YG12', 'YH11', 'YH12', 'XM11', 'XM12',
    'YM11', 'YM12', 'ZM11', 'ZM12',
)
COLUMN_INDEX = {name: i for i, name in enumerate(COLUMN_NAMES)}

R_LUB_OVER_A = 4.0        # lubrication cutoff in units of the radius a

_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'resistance_table.npz')
_FLOAT32_REGULARIZATION_GAP = 1e-5
_MIN_GAP_ENV = 'JAX_MD_SD_MIN_GAP'


def _regularization_gap_for_dtype(dtype) -> float:
  """Smallest surface gap the table resolves at the requested precision."""
  if np.dtype(dtype) == np.dtype(np.float64):
    return 0.0
  return _FLOAT32_REGULARIZATION_GAP


def _resolve_regularization_index(dtype) -> int:
  """Clamp row from the precision floor and the optional env-var cap."""
  gap = _regularization_gap_for_dtype(dtype)
  env = os.environ.get(_MIN_GAP_ENV)
  if env is not None:
    try:
      requested = float(env)
    except ValueError as exc:
      raise ValueError(
          '%s=%r is not a valid surface gap.' % (_MIN_GAP_ENV, env)) from exc
    if not np.isfinite(requested):
      raise ValueError(
          '%s=%r must be a finite surface gap.' % (_MIN_GAP_ENV, env))
    if requested < 0.0:
      raise ValueError(
          '%s=%r must be a non-negative surface gap.' % (_MIN_GAP_ENV, env))
    gap = max(gap, requested)
  if gap == 0.0:
    return 0
  with np.load(_TABLE_PATH, allow_pickle=False) as npz:
    gaps = np.asarray(npz['dist'], dtype=np.float64) - 2.0
  index = int(np.searchsorted(gaps, gap, side='left'))
  if index > len(gaps) - 2:
    raise ValueError(
        '%s=%r exceeds the largest supported regularization gap %r.'
        % (_MIN_GAP_ENV, env, gaps[-2]))
  return index


REGULARIZATION_INDEX = _resolve_regularization_index(REAL_DTYPE)


class ResistanceTable(NamedTuple):
  """Immutable container for the loaded table arrays."""
  dist: jnp.ndarray   # (N_DIST,)  tabulated center-to-center distances s = r/a
  vals: jnp.ndarray   # (N_DIST, 22) near-field scalar functions
  xi_min: jnp.ndarray  # scalar, smallest tabulated surface gap (s - 2)
  dr: jnp.ndarray     # scalar, log-space discretization step


_CACHE = None


def load_resistance_table() -> ResistanceTable:
  """Load and cache the regenerated resistance table as ``REAL_DTYPE`` arrays."""
  global _CACHE
  if _CACHE is None:
    if not os.path.exists(_TABLE_PATH):
      raise FileNotFoundError(
          '%s not found; run generate_resistance_table.py in '
          'jax_md/hydro/data/.' % _TABLE_PATH)
    with np.load(_TABLE_PATH, allow_pickle=True) as npz:
      dist = jnp.asarray(npz['dist'], dtype=REAL_DTYPE)
      vals = jnp.asarray(npz['vals'], dtype=REAL_DTYPE)
      xi_min = jnp.asarray(npz['xi_min'], dtype=REAL_DTYPE)
      dr = jnp.asarray(npz['dr'], dtype=REAL_DTYPE)
      # Sanity: column ordering in the file matches our expectation.
      file_cols = tuple(str(c) for c in npz['column_names'])
    if file_cols != COLUMN_NAMES:
      raise ValueError('Table column order mismatch: %s' % (file_cols,))
    _CACHE = ResistanceTable(
        dist=dist, vals=vals, xi_min=xi_min, dr=dr)
  return _CACHE


def interpolate_scalars(r: jnp.ndarray, a, table: ResistanceTable = None):
  """Interpolate the 22 near-field scalar functions at separation(s) ``r``.

  Args:
    r: center-to-center distance(s), arbitrary leading shape ``(...)``.
    a: sphere radius (scalar).
    table: optional preloaded :class:`ResistanceTable`; loaded if omitted.

  Returns:
    Array of shape ``(..., 22)`` with the interpolated scalars (in FSD's
    dimensionless ``a = eta = 1`` convention; dimensional prefactors are applied
    later at tensor assembly).
  """
  if table is None:
    table = load_resistance_table()
  dist = table.dist
  vals = table.vals
  xi_min = table.xi_min
  dr = table.dr
  n_dist = dist.shape[0]

  r = jnp.asarray(r, dtype=REAL_DTYPE)
  a = jnp.asarray(a, dtype=REAL_DTYPE)
  s = r / a
  xi = s - jnp.asarray(2.0, dtype=REAL_DTYPE)

  # Lower row index in log-space, floored (matches the C++ int cast for xi > 0).
  # Guard the log against non-positive gaps (overlaps): they clip to the
  # regularization row anyway.
  xi_safe = jnp.maximum(xi, xi_min)
  ind_f = jnp.floor(jnp.log10(xi_safe / xi_min) / dr)
  ind = jnp.clip(
      ind_f.astype(jnp.int32), REGULARIZATION_INDEX, n_dist - 2)

  d_lo = dist[ind]
  d_hi = dist[ind + 1]
  delta = d_hi - d_lo
  # Adjacent production-grid distances can coincide after float32 rounding.
  safe_delta = jnp.where(delta > 0, delta, jnp.ones_like(delta))
  fac = (s - d_lo) / safe_delta
  fac = jnp.where(delta > 0, fac, jnp.zeros_like(fac))
  fac = jnp.clip(fac, 0.0, 1.0)

  v_lo = vals[ind]                       # (..., 22)
  v_hi = vals[ind + 1]                   # (..., 22)
  return v_lo + fac[..., None] * (v_hi - v_lo)
