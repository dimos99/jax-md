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
claim is asserted analytically here and verified empirically by the post-NUFFT
trace test in ``tests/rpy_stresslet_test.py``; if that test ever fails, the
drop-zz storage is lossy and must be revisited.

Note the 8 raw components are *not* Frobenius-orthonormal coordinates of the
traceless subspace (the diagonal entries are linearly constrained).  Any test
probing the grand mobility as a dense matrix must use
``traceless_orthonormal_basis()`` instead, or a symmetric operator will look
asymmetric.

Symmetric/antisymmetric (rigid) decomposition
---------------------------------------------
The stresslet constraint works in the rigid representation
``(F, L, S) <-> (U, Omega, E)``.  The couplet splits as

    C = S - (1/2) eps . L        i.e.  C_mn = S_mn - (1/2) eps_mnk L_k,

where ``S`` is the (symmetric traceless) stresslet and ``L`` the torque.
This embed sign is pinned externally by the rotlet test in
``tests/rpy_stresslet_test.py`` (and matches Fiore's FSD ``Stokes.cc``).
The extraction signs are then *forced* by requiring the round-trip
``C -> (S, L) -> C`` to be the identity on traceless tensors:

    L_k = -(1/2) eps_kmn (C_mn - C_nm) = -eps_kmn C_mn.

The velocity-gradient output decomposes the same way,

    E   = sym(D)  (rate of strain, 5 dof),
    Omega_k = -(1/2) eps_kij D_ij,

which, with our convention ``D_ij = du_i/dx_j``, is exactly the physical
angular velocity ``omega = (1/2) curl(u)`` and is the Frobenius adjoint of
the torque embed -- so the grand mobility stays symmetric in
``(F, L, S)`` coordinates.

The 5-dof stresslet/strain coordinates use the *orthonormal* symmetric
members of ``traceless_orthonormal_basis()`` (indices 3..7), never the
drop-zz packing: orthonormality is what makes ``M_ES`` genuinely symmetric
in the working coordinates.
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


# Levi-Civita symbol eps_ijk.
_LEVI_CIVITA_NP = np.zeros((3, 3, 3), dtype=np.float64)
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
  _LEVI_CIVITA_NP[_i, _j, _k] = 1.0
  _LEVI_CIVITA_NP[_i, _k, _j] = -1.0


def levi_civita() -> jnp.ndarray:
  """Levi-Civita symbol as a (3, 3, 3) array in the working real dtype."""
  return jnp.asarray(_LEVI_CIVITA_NP, dtype=REAL_DTYPE)


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


def stresslet_basis() -> jnp.ndarray:
  """Orthonormal basis of the symmetric traceless subspace, shape (5, 3, 3).

  These are members 3..7 of ``traceless_orthonormal_basis()`` -- the same
  coordinates the dense-probing test helpers use, so a symmetric operator
  stays symmetric in these coordinates.
  """
  return traceless_orthonormal_basis()[3:8]


def stresslet_to_couplet(S5: jnp.ndarray) -> jnp.ndarray:
  """Embed 5-dof orthonormal stresslet coordinates as a (..., 3, 3) couplet.

  The result is symmetric and traceless (zero torque channel).
  """
  S5 = jnp.asarray(S5)
  basis = jnp.asarray(stresslet_basis(), dtype=S5.dtype)
  return jnp.einsum('...a,aij->...ij', S5, basis)


def torque_to_couplet(L3: jnp.ndarray) -> jnp.ndarray:
  """Embed a torque (axial) vector as an antisymmetric couplet.

  ``C_mn = -(1/2) eps_mnk L_k`` -- the rotlet-pinned convention (module
  docstring); do not change the sign or factor independently of
  ``couplet_to_stresslet_torque``.
  """
  L3 = jnp.asarray(L3)
  eps = jnp.asarray(_LEVI_CIVITA_NP, dtype=L3.dtype)
  return jnp.einsum('mnk,...k->...mn', -0.5 * eps, L3)


def couplet_to_stresslet_torque(
    C: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Split a traceless couplet into (stresslet S5, torque L3).

  Exact inverse of ``stresslet_to_couplet(S5) + torque_to_couplet(L3)``:
  ``S5`` are orthonormal coordinates of ``sym(C)`` and
  ``L_k = -(1/2) eps_kmn (C_mn - C_nm)``.
  """
  C = jnp.asarray(C)
  basis = jnp.asarray(stresslet_basis(), dtype=C.dtype)
  eps = jnp.asarray(_LEVI_CIVITA_NP, dtype=C.dtype)
  sym = 0.5 * (C + jnp.swapaxes(C, -1, -2))
  S5 = jnp.einsum('...ij,aij->...a', sym, basis)
  L3 = -jnp.einsum('kmn,...mn->...k', eps, C)
  return S5, L3


def decompose_gradient(D: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Split a traceless velocity gradient into (strain E5, angular velocity).

  ``E5`` are orthonormal coordinates of the rate of strain ``sym(D)`` and
  ``Omega_k = -(1/2) eps_kij D_ij``, the physical angular velocity
  ``omega = (1/2) curl(u)`` for ``D_ij = du_i/dx_j``.

  Note the antisymmetric factor differs from ``couplet_to_stresslet_torque``
  by 2: the gradient extraction is the Frobenius *adjoint* of the torque
  embed (which keeps the grand mobility symmetric in (F, L, S)
  coordinates and matches FSD's ``Mobility_D2WE_kernel``), whereas the
  couplet extraction is the *inverse* of the embed (round-trip identity).
  The two differ because the embed ``-(1/2) eps`` is not orthonormal.
  """
  D = jnp.asarray(D)
  basis = jnp.asarray(stresslet_basis(), dtype=D.dtype)
  eps = jnp.asarray(_LEVI_CIVITA_NP, dtype=D.dtype)
  sym = 0.5 * (D + jnp.swapaxes(D, -1, -2))
  E5 = jnp.einsum('...ij,aij->...a', sym, basis)
  Omega = -0.5 * jnp.einsum('kij,...ij->...k', eps, D)
  return E5, Omega


def couplet_to_orthonormal(C: jnp.ndarray) -> jnp.ndarray:
  """Coordinates of a traceless (..., 3, 3) tensor in the orthonormal basis.

  Returns (..., 8) coordinates in ``traceless_orthonormal_basis()`` order.
  Unlike the drop-zz packing these are Frobenius-orthonormal, so the
  Euclidean inner product on the coordinates equals the Frobenius pairing
  on the tensors -- the property the Lanczos square root relies on.
  """
  C = jnp.asarray(C)
  basis = jnp.asarray(traceless_orthonormal_basis(), dtype=C.dtype)
  return jnp.einsum('...ij,aij->...a', C, basis)


def orthonormal_to_couplet(c8: jnp.ndarray) -> jnp.ndarray:
  """Inverse of ``couplet_to_orthonormal`` (exact on traceless tensors)."""
  c8 = jnp.asarray(c8)
  basis = jnp.asarray(traceless_orthonormal_basis(), dtype=c8.dtype)
  return jnp.einsum('...a,aij->...ij', c8, basis)


def grand_to_flat(U: jnp.ndarray, D: jnp.ndarray) -> jnp.ndarray:
  """Flatten a grand pair (U (..., 3), D (..., 3, 3)) to (..., 11).

  Channels 0:3 are Cartesian; channels 3:11 are orthonormal-basis
  coordinates of the traceless tensor channel.  The grand mobility is a
  symmetric matrix in these flat coordinates (it is not in drop-zz).
  """
  U = jnp.asarray(U)
  return jnp.concatenate([U, couplet_to_orthonormal(D)], axis=-1)


def flat_to_grand(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Inverse of ``grand_to_flat``: (..., 11) -> (F (..., 3), C (..., 3, 3))."""
  x = jnp.asarray(x)
  return x[..., :3], orthonormal_to_couplet(x[..., 3:])
