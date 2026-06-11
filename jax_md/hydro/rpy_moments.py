"""Couplet (force-dipole) moment conventions shared by all grand-mobility code.

This module is the single source of truth for how the traceless couplet tensor
is represented throughout the stresslet extension (Fiore & Swan 2018).  Both
the real-space and wave-space operators import these helpers; do not redefine
packing or index conventions elsewhere.

Index convention
----------------
The couplet ``C_mn`` couples to flow through the *second* index contracting the
gradient direction, following Fiore thesis Eqs. 4.15 / 4.25:

  * wave space:  force density  f_m(k) ~ -i * Pdip(ka) * C_mn k_n
  * real space:  M_UC,imn ~ G1 (d_im rhat_n - ...) C_mn  with rhat_n on the
    second couplet index.

(The thesis Eq. 4.2 contracts the first index instead; the two are
inconsistent in the source notes.  We standardize on the 4.15/4.25 convention
on both sides of the Ewald split.  The absolute convention is pinned against
external ground truth by the antisymmetric-couplet rotation-translation test
in ``tests/rpy_stresslet_test.py``.)

The velocity-gradient output ``D_ij`` uses the matching convention
``D_ij = (Faxen-filtered) du_i/dx_j``; it is traceless by incompressibility.

8-component packing (drop-zz)
-----------------------------
A traceless 3x3 tensor has 8 independent components.  On FFT grids we store
the fixed row-major order

    COUPLET_COMPONENTS = (xx, xy, xz, yx, yy, yz, zx, zy)

and reconstruct ``zz = -(xx + yy)``.  This is lossless for traceless input
because (a) packing/unpacking are linear, so they commute with spreading and
FFTs, and (b) the gridded velocity-gradient modes ``D_hat_ij = i Pdip k_j
u_hat_i`` are traceless mode-by-mode since ``B`` projects ``u_hat``
perpendicular to ``k`` (``tr D_hat = i Pdip (k . u_hat) = 0``), exactly, at
every grid mode, regardless of spreading/aliasing of the *input* grids.  This
claim is asserted analytically here and gated empirically by the post-NUFFT
trace test in ``tests/rpy_stresslet_test.py``; if that gate ever fails, the
drop-zz storage is lossy and must be revisited.

Note the 8 raw components are *not* Frobenius-orthonormal coordinates of the
traceless subspace (the diagonal entries are linearly constrained).  Any test
probing the grand mobility as a dense matrix must use
``traceless_orthonormal_basis()`` instead, or a symmetric operator will look
asymmetric.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE


# Fixed component order for grid storage: row-major with zz dropped.
COUPLET_COMPONENTS: Tuple[str, ...] = (
    'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy')

# (row, col) index pairs matching COUPLET_COMPONENTS.
_COMPONENT_INDICES: Tuple[Tuple[int, int], ...] = (
    (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1))

# Number of moment channels per particle: 3 force + 8 couplet.
N_MOMENTS = 11


def traceless(C: jnp.ndarray) -> jnp.ndarray:
  """Project a (..., 3, 3) tensor onto its traceless part."""
  C = jnp.asarray(C)
  tr = jnp.trace(C, axis1=-2, axis2=-1)
  eye = jnp.eye(3, dtype=C.dtype)
  return C - (tr / 3.0)[..., None, None] * eye


def couplet_to_components(C: jnp.ndarray) -> jnp.ndarray:
  """Pack a traceless (..., 3, 3) tensor into (..., 8) drop-zz components.

  Works for any leading batch shape (particles or grid axes).  The input is
  assumed traceless; the zz entry is discarded without projection.
  """
  C = jnp.asarray(C)
  flat = C.reshape(C.shape[:-2] + (9,))
  # Row-major flat indices 0..8 with index 8 (zz) dropped.
  return flat[..., :8]


def components_to_couplet(c8: jnp.ndarray) -> jnp.ndarray:
  """Unpack (..., 8) drop-zz components into a traceless (..., 3, 3) tensor.

  ``zz`` is reconstructed as ``-(xx + yy)``.
  """
  c8 = jnp.asarray(c8)
  zz = -(c8[..., 0] + c8[..., 4])
  flat = jnp.concatenate([c8, zz[..., None]], axis=-1)
  return flat.reshape(c8.shape[:-1] + (3, 3))


# Grid-axis aliases (same functions; trailing axes are generic).
pack8 = couplet_to_components
unpack8 = components_to_couplet


def traceless_orthonormal_basis() -> jnp.ndarray:
  """Frobenius-orthonormal basis of the traceless 3x3 subspace, shape (8, 3, 3).

  Ordering: 3 normalized antisymmetric elements (xy, xz, yz planes), then
  3 normalized symmetric off-diagonal elements, then 2 diagonal traceless
  elements.  Orthonormal under ``<A, B> = A_ij B_ij``.  Use these (not the raw
  drop-zz components) as moment coordinates when probing dense matrices.
  """
  basis = np.zeros((8, 3, 3), dtype=np.float64)
  s2 = 1.0 / np.sqrt(2.0)
  # Antisymmetric.
  for n, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
    basis[n, i, j] = s2
    basis[n, j, i] = -s2
  # Symmetric off-diagonal.
  for n, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
    basis[3 + n, i, j] = s2
    basis[3 + n, j, i] = s2
  # Diagonal traceless.
  basis[6] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
  basis[7] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
  return jnp.asarray(basis, dtype=REAL_DTYPE)
