"""Near-field lubrication resistance table: loading and interpolation.

Loads the committed ``data/resistance_table.npz`` (extracted from Fiore's FSD
``Stokes_ResistanceTable.cc`` by ``data/extract_resistance_table.py``) and
provides a pure-JAX, vmappable scalar lookup that reproduces the FSD
interpolation (``Lubrication.cu:140-238``) exactly.

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
  * Lower row index in log-space, floored:
    ``ind = floor(log10(xi / XI_MIN) / DR)``  (C++ truncates the float cast;
    ``floor`` matches for the positive argument), then clipped to
    ``[REGULARIZATION_INDEX, N_DIST - 2]`` so ``ind`` and ``ind + 1`` are valid.
  * Lerp weight in *raw distance* using the stored tabulated distances:
    ``fac = (s - dist[ind]) / (dist[ind + 1] - dist[ind])``, clipped to [0, 1].
  * Result: ``vals[ind] + fac * (vals[ind + 1] - vals[ind])``.

This reproduces the two FSD regimes without host branching: for ``xi <= ~1e-3``
the clip pins ``ind = REGULARIZATION_INDEX`` and ``s < dist[ind]`` drives
``fac -> 0``, returning the regularization row verbatim; approaching ``s = 4``
the weight ``fac -> 1`` lerps toward the all-zero final row.

The lubrication cutoff itself (``r < r_lub = 4a``) is enforced by the caller's
neighbor mask, never by this table.
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

N_FUNC = 22

# FSD tabulation metadata (named constants; see module docstring).
XI_MIN = 1.0e-4           # smallest tabulated surface gap (s - 2)
DR = 0.004305             # log-space discretization step
# Near-contact "roughness" regularization clamp.  This is a *choice* inherited
# from FSD, not a converged value: it caps the maximum lubrication resistance
# and therefore sets near-contact stiffness / timestep stability.  Tunable knob.
REGULARIZATION_INDEX = 232
R_LUB_OVER_A = 4.0        # lubrication cutoff in units of the radius a

_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'resistance_table.npz')


class ResistanceTable(NamedTuple):
  """Immutable container for the loaded table arrays."""
  dist: jnp.ndarray   # (N_DIST,)  tabulated center-to-center distances s = r/a
  vals: jnp.ndarray   # (N_DIST, 22) near-field scalar functions


_CACHE = {}


def load_resistance_table() -> ResistanceTable:
  """Load (and cache) the resistance table as JAX arrays in ``REAL_DTYPE``."""
  if 'table' not in _CACHE:
    if not os.path.exists(_DATA_PATH):
      raise FileNotFoundError(
          'resistance_table.npz not found at %s; run '
          'jax_md/hydro/data/extract_resistance_table.py first.' % _DATA_PATH)
    npz = np.load(_DATA_PATH, allow_pickle=True)
    dist = jnp.asarray(npz['dist'], dtype=REAL_DTYPE)
    vals = jnp.asarray(npz['vals'], dtype=REAL_DTYPE)
    # Sanity: column ordering in the file matches our expectation.
    file_cols = tuple(str(c) for c in npz['column_names'])
    if file_cols != COLUMN_NAMES:
      raise ValueError('Table column order mismatch: %s' % (file_cols,))
    _CACHE['table'] = ResistanceTable(dist=dist, vals=vals)
  return _CACHE['table']


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
  n_dist = dist.shape[0]

  r = jnp.asarray(r, dtype=REAL_DTYPE)
  a = jnp.asarray(a, dtype=REAL_DTYPE)
  s = r / a
  xi = s - jnp.asarray(2.0, dtype=REAL_DTYPE)

  # Lower row index in log-space, floored (matches the C++ int cast for xi > 0).
  # Guard the log against non-positive gaps (overlaps): they clip to the
  # regularization row anyway.
  xi_safe = jnp.maximum(xi, jnp.asarray(XI_MIN, dtype=REAL_DTYPE))
  ind_f = jnp.floor(jnp.log10(xi_safe / XI_MIN) / DR)
  ind = jnp.clip(ind_f.astype(jnp.int32), REGULARIZATION_INDEX, n_dist - 2)

  d_lo = dist[ind]
  d_hi = dist[ind + 1]
  fac = (s - d_lo) / (d_hi - d_lo)
  fac = jnp.clip(fac, 0.0, 1.0)

  v_lo = vals[ind]                       # (..., 22)
  v_hi = vals[ind + 1]                   # (..., 22)
  return v_lo + fac[..., None] * (v_hi - v_lo)
