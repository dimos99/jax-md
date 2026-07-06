"""Near-field lubrication resistance operator ``R^nf`` (monodisperse spheres).

Phase 1 of Fast Stokesian Dynamics: the pairwise-additive, short-ranged
(``r_lub = 4a``), matrix-free linked-cell lubrication resistance.  The 22
tabulated scalar functions already have the two-body far-field multipole
subtracted (``R^nf = R^2B - R-bar^2B``), so this operator simply adds the
missing near-field moments that the PSE far-field mobility ``M`` cannot resolve.
It is fully additive: RPY / stresslet-RPY code paths never construct it.

Generalized coordinates.  Per particle the operator maps a generalized velocity
``(U, Omega, E)`` to a generalized force ``(F, L, S)`` with the packed layout

    g[..., 0:3]  = U / F       (translation / force, Cartesian)
    g[..., 3:6]  = Omega / L   (rotation / torque, Cartesian)
    g[..., 6:11] = E5 / S5     (rate of strain / stresslet, orthonormal 5-basis)

The rate-of-strain and stresslet use the *existing* orthonormal symmetric
traceless basis (:func:`jax_md.hydro.rpy_moments.stresslet_basis`) so that the
``R^nf_SU``/``R^nf_SE`` blocks add into the same stresslet vectors the
saddle-point RHS and stresslet output use in Phase 2 (Caveat B).

Tensor forms are taken verbatim from Fiore's FSD ``Lubrication.cu`` (the matched
pair with the table), with the unit vector ``r_hat`` pointing from the receiver
(center) particle ``i`` to the neighbor (sender) particle ``j`` -- identical to
the ``R = pos_j - pos_i`` convention there.  FSD assembles bare scalars in
``a = eta = 1`` units; here we restore physical dimensions with the Kim-Karrila
prefactors (:data:`_PREFACTOR_POWERS`).  The ``A`` (FU) prefactor ``6 pi eta a``
is verified analytically against the squeeze-flow lubrication limit; the
``G/H/M`` prefactors are confirmed near ``r = 4a`` by the far-field reduction
test (the analytic Jeffrey-Onishi gate is the deferred external pin).
"""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import ops

from jax_md import dataclasses
from jax_md import partition
from jax_md import space

from jax_md.hydro import rpy_moments
from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE, current_box_matrix
from jax_md.hydro.rpy_real_lattice_helpers import (
    _box_fn_supports_shear_kwargs,
    _is_traced_value,
    _neighbor_box_from_matrix,
    _worst_case_shear_neighbor_box,
)
from jax_md.hydro import sd_nearfield_table as nf_table


# Packed generalized-coordinate layout (per particle).
N_GEN = 11
SL_U = slice(0, 3)     # translation / force
SL_W = slice(3, 6)     # rotation / torque
SL_E = slice(6, 11)    # strain / stresslet (orthonormal 5-basis)


# Levi-Civita symbol.
_EPS_NP = np.zeros((3, 3, 3), dtype=np.float64)
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
  _EPS_NP[_i, _j, _k] = 1.0
  _EPS_NP[_i, _k, _j] = -1.0


def _kim_karrila_prefactors(a, eta):
  """Dimensional prefactors re-dimensionalizing the bare FSD resistance table.

  The resistance table (``resistance_table.npz``, ported from FSD) is in
  **Stokesian-Dynamics normalization** (``a = 1``, drag-scaled) and FSD applies
  it *bare* in every kernel -- no ``pi``, no ``(2a)^k``, no per-block numeric
  factor (Lubrication.cu RFU:280, RSU/G:583, RSE/M:1160).  The bare functions
  therefore already satisfy the near-contact rank-1 squeeze relation
  ``R_SE . R_FU = R_SU . R_FE``, which is what makes the force-free squeeze
  stresslet (hence eta'_inf) FINITE at contact.

  Re-dimensionalizing to physical ``(a, eta)`` must preserve that relation, so it
  uses the **uniform Brady-Bossis scaling** ``6 pi eta a^k`` on every block
  (``k`` = 1 + number of length factors), NOT the textbook Kim-Karrila per-block
  ``pi (a_i + a_j)^k`` numerics:

    A (FU)        : 6 pi eta a     (verified analytically; self-mobility)
    B (F-Omega)   : 6 pi eta a^2
    C (L-Omega)   : 6 pi eta a^3
    G (S-U / F-E) : 6 pi eta a^2
    H (S-Omega)   : 6 pi eta a^3
    M (S-E)       : 6 pi eta a^3

  Rank-1 check: ``p_A p_M / p_G^2 = (6)(6)/6^2 = 1`` -> the leading ``1/xi``
  cancels exactly (two-sphere squeeze probe: ``a*m/(g_su*g_fe) -> 1.0000``).

  PREVIOUS (BUGGY) prefactors -- textbook Kim-Karrila ``pi eta (a_i + a_j)^k``,
  kept here for retrieval.  These gave ``p_A p_M / p_G^2 = 6*8/4^2 = 3``, which
  broke the cancellation and left a spurious ``1/xi`` in eta'_inf (inflated and
  falsely cutoff-sensitive; only ``A`` happened to be correct):
      s = 2.0 * a  # a_i + a_j for equal spheres
      {'A': 3*pi*eta*s,   'B': pi*eta*s**2, 'C': pi*eta*s**3,
       'G': pi*eta*s**2,  'H': pi*eta*s**3, 'M': pi*eta*s**3}

  Returns a dict keyed by tensor family.
  """
  pi = np.pi
  return {
      'A': 6.0 * pi * eta * a,
      'B': 6.0 * pi * eta * a ** 2,
      'C': 6.0 * pi * eta * a ** 3,
      'G': 6.0 * pi * eta * a ** 2,
      'H': 6.0 * pi * eta * a ** 3,
      'M': 6.0 * pi * eta * a ** 3,
  }


def _outer(u, v):
  """Batched outer product ``u_i v_j`` -> (..., 3, 3)."""
  return u[..., :, None] * v[..., None, :]


def _build_pair_operators(rhat, scalars, a, eta):
  """Build per-edge generalized self/cross resistance blocks.

  Args:
    rhat: (E, 3) unit vectors from receiver i to sender j.
    scalars: (E, 22) interpolated near-field scalar functions (FSD bare units),
      column order :data:`sd_nearfield_table.COLUMN_NAMES`.
    a, eta: sphere radius and viscosity.

  Returns:
    (R_self, R_cross), each (E, 11, 11).  The edge contributes
    ``F_i += R_self @ g_i + R_cross @ g_j`` where ``g`` is the packed
    generalized velocity.
  """
  dtype = rhat.dtype
  E = rhat.shape[0]
  eye = jnp.eye(3, dtype=dtype)
  eps = jnp.asarray(_EPS_NP, dtype=dtype)
  basis = jnp.asarray(rpy_moments.stresslet_basis(), dtype=dtype)  # (5,3,3)

  pref = _kim_karrila_prefactors(a, eta)
  c = nf_table.COLUMN_INDEX
  # Dimensionalize each scalar family.  Self = "11", cross = "12".
  XA11 = pref['A'] * scalars[:, c['XA11']]
  XA12 = pref['A'] * scalars[:, c['XA12']]
  YA11 = pref['A'] * scalars[:, c['YA11']]
  YA12 = pref['A'] * scalars[:, c['YA12']]
  YB11 = pref['B'] * scalars[:, c['YB11']]
  YB12 = pref['B'] * scalars[:, c['YB12']]
  XC11 = pref['C'] * scalars[:, c['XC11']]
  XC12 = pref['C'] * scalars[:, c['XC12']]
  YC11 = pref['C'] * scalars[:, c['YC11']]
  YC12 = pref['C'] * scalars[:, c['YC12']]
  XG11 = pref['G'] * scalars[:, c['XG11']]
  XG12 = pref['G'] * scalars[:, c['XG12']]
  YG11 = pref['G'] * scalars[:, c['YG11']]
  YG12 = pref['G'] * scalars[:, c['YG12']]
  YH11 = pref['H'] * scalars[:, c['YH11']]
  YH12 = pref['H'] * scalars[:, c['YH12']]
  XM11 = pref['M'] * scalars[:, c['XM11']]
  XM12 = pref['M'] * scalars[:, c['XM12']]
  YM11 = pref['M'] * scalars[:, c['YM11']]
  YM12 = pref['M'] * scalars[:, c['YM12']]
  ZM11 = pref['M'] * scalars[:, c['ZM11']]
  ZM12 = pref['M'] * scalars[:, c['ZM12']]

  # Geometric primitives (E, ...).
  P = _outer(rhat, rhat)                              # (E,3,3) r_i r_j
  Iperp = eye[None] - P                               # (E,3,3) delta - r r
  # eps_klm r_m  -> (E,3,3); acts as (cross with r) on the right index.
  epsr = jnp.einsum('klm,em->ekl', eps, rhat)         # (E,3,3)

  def sym3(t):
    return 0.5 * (t + jnp.swapaxes(t, -1, -2))

  def s5_from_3(t):
    """Project symmetric traceless (E,3,3) onto the 5 orthonormal coords."""
    return jnp.einsum('aij,eij->ea', basis, sym3(t))

  # ---- 3x3 sub-blocks --------------------------------------------------
  def A_block(XA, YA):
    return XA[:, None, None] * P + YA[:, None, None] * Iperp

  # FU force from U: A.  (E,3,3)
  A11 = A_block(XA11, YA11)
  A12 = A_block(XA12, YA12)
  # LOmega torque from Omega: C.
  C11 = A_block(XC11, YC11)
  C12 = A_block(XC12, YC12)
  # F-Omega and L-U couplings (B family); see module docstring for signs.
  # FSD: fi += YB11*(-eps r . wi) + (-YB12)*(-eps r . wj); li += YB11*(eps r . ui)+YB12*(eps r . uj)
  # (eps r . w)_k = eps_klm r_m w_l = epsr_kl w_l.
  R_FW11 = -YB11[:, None, None] * epsr        # F_i from Omega_i
  R_FW12 = YB12[:, None, None] * epsr         # F_i from Omega_j (YB21 = -YB12)
  R_LU11 = YB11[:, None, None] * epsr         # L_i from U_i
  R_LU12 = YB12[:, None, None] * epsr         # L_i from U_j

  # ---- rank-3 G (S from U) and H (S from Omega) ------------------------
  # G_ijk = XG (r_i r_j - d_ij/3) r_k + YG (d_ik r_j + r_i d_jk - 2 r_i r_j r_k)
  Pt = P - eye[None] / 3.0                                   # (E,3,3)
  def G_tensor(XG, YG):
    t1 = XG[:, None, None, None] * Pt[..., None] * rhat[:, None, None, :]
    dik_rj = eye[None, :, None, :] * rhat[:, None, :, None]  # d_ik r_j
    ri_djk = rhat[:, :, None, None] * eye[None, None, :, :]  # r_i d_jk
    rrr = P[..., None] * rhat[:, None, None, :]              # r_i r_j r_k
    t2 = YG[:, None, None, None] * (dik_rj + ri_djk - 2.0 * rrr)
    return t1 + t2                                           # (E,3,3,3) S_ij from U_k
  G11 = G_tensor(XG11, YG11)
  G12 = G_tensor(XG12, YG12)

  # H_ijp = YH (r_i eps_jpq r_q + r_j eps_ipq r_q);  (eps . r) on last index.
  epsr_jp = epsr  # eps_jpq r_q = epsr_jp
  def H_tensor(YH):
    ri_epsr = rhat[:, :, None, None] * epsr_jp[:, None, :, :]   # r_i eps_jpq r_q
    rj_epsr = rhat[:, None, :, None] * epsr_jp[:, :, None, :]   # r_j eps_ipq r_q
    return YH[:, None, None, None] * (ri_epsr + rj_epsr)        # (E,3,3,3) S_ij from Omega_p
  H11 = H_tensor(YH11)
  H12 = H_tensor(YH12)

  # ---- rank-4 M (S from E) --------------------------------------------
  # See Lubrication.cu:1151-1198 (and the comment block reproduced there).
  def M_tensor(XM, YM, ZM):
    d = eye
    # XM term: 3/2 (r_i r_j - d_ij/3)(r_k r_l - d_kl/3)
    tX = 1.5 * XM[:, None, None, None, None] * (Pt[:, :, :, None, None]
                                                * Pt[:, None, None, :, :])
    # YM term: 1/2 (r_i d_jl r_k + r_j d_il r_k + r_i d_jk r_l + r_j d_ik r_l
    #               - 4 r_i r_j r_k r_l)
    ri = rhat[:, :, None, None, None]
    rj = rhat[:, None, :, None, None]
    rk = rhat[:, None, None, :, None]
    rl = rhat[:, None, None, None, :]
    d_jl = d[None, None, :, None, :]
    d_il = d[None, :, None, None, :]
    d_jk = d[None, None, :, :, None]
    d_ik = d[None, :, None, :, None]
    d_ij = d[None, :, :, None, None]
    d_kl = d[None, None, None, :, :]
    tY = 0.5 * YM[:, None, None, None, None] * (
        ri * d_jl * rk + rj * d_il * rk + ri * d_jk * rl + rj * d_ik * rl
        - 4.0 * ri * rj * rk * rl)
    # ZM term: 1/2 ( d_ik d_jl + d_jk d_il - d_ij d_kl + r_i r_j d_kl
    #               + d_ij r_k r_l + r_i r_j r_k r_l
    #               - r_i d_jl r_k - r_j d_il r_k - r_i d_jk r_l - r_j d_ik r_l )
    tZ = 0.5 * ZM[:, None, None, None, None] * (
        d_ik * d_jl + d_jk * d_il - d_ij * d_kl
        + ri * rj * d_kl + d_ij * rk * rl + ri * rj * rk * rl
        - ri * d_jl * rk - rj * d_il * rk - ri * d_jk * rl - rj * d_ik * rl)
    return tX + tY + tZ          # (E,3,3,3,3) S_ij from E_kl
  M11 = M_tensor(XM11, YM11, ZM11)
  M12 = M_tensor(XM12, YM12, ZM12)

  # ---- map rank-3/4 blocks into the orthonormal 5-coordinate basis -----
  # S5 from U/Omega: (5,3) = basis_a:ij  T_ijk
  def su_block(T):           # T: (E,3,3,3) -> (E,5,3)
    return jnp.einsum('aij,eijk->eak', basis, T)
  # S5 from E5: (5,5) = basis_a:ij M_ijkl basis_b:kl
  def se_block(M):           # M: (E,3,3,3,3) -> (E,5,5)
    return jnp.einsum('aij,eijkl,bkl->eab', basis, M, basis)

  R_SU11 = su_block(G11)     # (E,5,3)
  R_SU12 = su_block(G12)
  R_SW11 = su_block(H11)     # S from Omega
  R_SW12 = su_block(H12)
  R_SE11 = se_block(M11)     # (E,5,5)
  R_SE12 = se_block(M12)

  # Symmetry: force-from-strain / torque-from-strain are transposes of
  # stresslet-from-velocity / stresslet-from-rotation.  The *self* blocks use
  # XG11/YH11 directly (positive transpose).  The *cross* blocks invoke the
  # Jeffrey-Onishi 21<-12 relations: XG21=-XG12, YG21=-YG12 (G is odd) so the
  # cross FE block is the NEGATIVE transpose; YH21=+YH12 (H is even) so the
  # cross LE block stays a positive transpose.
  R_FE11 = jnp.swapaxes(R_SU11, -1, -2)    # (E,3,5)
  R_FE12 = -jnp.swapaxes(R_SU12, -1, -2)
  R_LE11 = jnp.swapaxes(R_SW11, -1, -2)
  R_LE12 = jnp.swapaxes(R_SW12, -1, -2)

  def assemble(A, R_FW, R_LU, Cc, R_SU, R_SW, R_FE, R_LE, R_SE):
    R = jnp.zeros((E, N_GEN, N_GEN), dtype=dtype)
    R = R.at[:, SL_U, SL_U].set(A)
    R = R.at[:, SL_U, SL_W].set(R_FW)
    R = R.at[:, SL_W, SL_U].set(R_LU)
    R = R.at[:, SL_W, SL_W].set(Cc)
    R = R.at[:, SL_E, SL_U].set(R_SU)
    R = R.at[:, SL_E, SL_W].set(R_SW)
    R = R.at[:, SL_U, SL_E].set(R_FE)
    R = R.at[:, SL_W, SL_E].set(R_LE)
    R = R.at[:, SL_E, SL_E].set(R_SE)
    return R

  R_self = assemble(A11, R_FW11, R_LU11, C11, R_SU11, R_SW11,
                    R_FE11, R_LE11, R_SE11)
  R_cross = assemble(A12, R_FW12, R_LU12, C12, R_SU12, R_SW12,
                     R_FE12, R_LE12, R_SE12)
  return R_self, R_cross


@dataclasses.dataclass
class NearFieldState:
  """State for the near-field resistance operator."""
  neighbors: partition.NeighborList
  box_matrix: jnp.ndarray
  fractional_coordinates: bool = dataclasses.static_field()


class PreparedNearField(NamedTuple):
  """Per-pair ``R^nf`` blocks precomputed at a FIXED configuration.

  During one saddle solve / Lanczos run the positions (and hence all pair
  geometry, table lookups, and 11x11 blocks) are constant, yet the matrix-free
  ``_core`` re-derives them on every matvec -- and the Chebyshev-Schur
  preconditioner alone does ``cheb_degree`` matvecs per GMRES iteration.  The
  reference CUDA FSD amortizes this setup once per step.  ``prepare`` runs the
  geometry + table + block assembly once; :func:`apply_prepared_blocks` then
  reduces each matvec to gather -> batched block multiply -> ``segment_sum``.

  Fields (a plain ``NamedTuple`` pytree -- deliberately NOT stored on
  ``NearFieldState``, so state treedefs and jit caches are unaffected):
    self_blocks: ``(N, 11, 11)`` -- the masked ``R_self`` edge blocks already
      segment-summed per receiver (the self half of the matvec collapses to a
      single dense per-particle block multiply).
    cross_blocks: ``(E, 11, 11)`` -- masked per-directed-edge ``R_cross``
      blocks (edge mask pre-applied; masked edges are exactly zero).  Memory:
      ``E*121`` floats, ~150 MB at N=4000 / max_k~80 in float32.
    receivers, senders: ``(E,)`` int32 directed-edge endpoints.
    has_neighbor: ``(N,)`` bool -- >=1 live lubrication edge (same mask as
      ``diagonal_FU_mask_prepared``).
  """
  self_blocks: jnp.ndarray
  cross_blocks: jnp.ndarray
  receivers: jnp.ndarray
  senders: jnp.ndarray
  has_neighbor: jnp.ndarray


def apply_prepared_blocks(prepared: PreparedNearField, gen_velocity):
  """``R^nf @ gen_velocity`` from precomputed blocks; ``(N, 11) -> (N, 11)``.

  Numerically equivalent to ``apply_prepared`` at the configuration
  ``prepared`` was built for (pinned by ``test_prepared_blocks_match_core``).
  """
  gv = jnp.asarray(gen_velocity, dtype=REAL_DTYPE)
  n = prepared.self_blocks.shape[0]
  out = jnp.einsum('nab,nb->na', prepared.self_blocks, gv)
  cross = jnp.einsum('eab,eb->ea', prepared.cross_blocks,
                     gv[prepared.senders])
  return out + ops.segment_sum(cross, prepared.receivers, n)


def apply_prepared_blocks_FU(prepared: PreparedNearField, u6):
  """``R^nf_FU @ u6`` from precomputed blocks; ``(N, 6) -> (N, 6)``.

  Exact FU sub-block of :func:`apply_prepared_blocks` (the discarded strain
  columns would multiply zeros), at ~(6/11)^2 of the flops -- this is the
  Chebyshev-Schur ``S~`` inner matvec.
  """
  u6 = jnp.asarray(u6, dtype=REAL_DTYPE)
  n = prepared.self_blocks.shape[0]
  out = jnp.einsum('nab,nb->na', prepared.self_blocks[:, :6, :6], u6)
  cross = jnp.einsum('eab,eb->ea', prepared.cross_blocks[:, :6, :6],
                     u6[prepared.senders])
  return out + ops.segment_sum(cross, prepared.receivers, n)


def prepared_diag_FU(prepared: PreparedNearField):
  """``(N, 6)`` diagonal of ``R^nf_FU`` read off the summed self blocks."""
  return jnp.diagonal(prepared.self_blocks[:, :6, :6], axis1=-2, axis2=-1)


def build_nearfield_resistance(
    space_fns,
    a,
    eta,
    *,
    r_lub: Optional[float] = None,
    fractional_coordinates: bool = True,
    dr_threshold: Optional[float] = None,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
):
  """Construct the matrix-free near-field lubrication resistance operator.

  Args:
    space_fns: ``(displacement_fn, shift_fn)`` or ``+(box_fn)`` for shearing.
    a, eta: sphere radius and viscosity.
    r_lub: lubrication cutoff (default ``4a``).
    fractional_coordinates: positions in ``[0,1)^d`` (default) or real.
    dr_threshold, capacity_multiplier, disable_cell_list: neighbor-list knobs.

  Returns:
    ``(init_fn, apply_fn)``.

    ``init_fn(positions, **shear_kwargs) -> NearFieldState`` allocates the
    ``r_lub`` neighbor list (at the worst-case shear box when ``box_fn`` is
    given, to avoid ``cell_size_too_small``).

    ``apply_fn(state, positions, gen_velocity, **shear_kwargs) ->
    (gen_force, next_state)`` where both generalized vectors are ``(N, 11)``
    with the packed ``[U, Omega, E5]`` / ``[F, L, S5]`` layout.  Uses a
    minimum-image kernel; valid when ``r_lub <= 0.5 * min box dimension``.
  """
  if r_lub is None:
    r_lub = nf_table.R_LUB_OVER_A * float(a)
  if r_lub <= 0.0:
    raise ValueError('r_lub must be positive.')
  if dr_threshold is None:
    dr_threshold = 0.1 * r_lub

  if len(space_fns) < 2:
    raise ValueError('space_fns must contain displacement and shift functions.')
  displacement_fn, _ = space_fns[:2]
  box_fn = space_fns[2] if len(space_fns) > 2 else None

  r_lub2 = float(r_lub * r_lub)
  table = nf_table.load_resistance_table()

  neighbor_fn = partition.neighbor_list(
      displacement_fn,
      box=1.0,
      r_cutoff=r_lub,
      dr_threshold=dr_threshold,
      capacity_multiplier=capacity_multiplier,
      disable_cell_list=disable_cell_list,
      mask_self=False,
      fractional_coordinates=fractional_coordinates,
      format=partition.NeighborListFormat.Dense,
  )

  def _allocate(positions, box_matrix, **kwargs):
    neighbor_kwargs = dict(kwargs)
    neighbor_box = _neighbor_box_from_matrix(box_matrix, fractional_coordinates)
    dim = int(positions.shape[1])
    if box_fn is not None and fractional_coordinates and neighbor_box is not None:
      if (not _is_traced_value(neighbor_box)
          and _box_fn_supports_shear_kwargs(box_fn, dim)):
        worst_np = _worst_case_shear_neighbor_box(
            box_fn, dim, np.asarray(neighbor_box, dtype=np.float64))
        neighbor_box = jnp.asarray(worst_np, dtype=neighbor_box.dtype)
      neighbor_kwargs['box'] = neighbor_box
    elif neighbor_box is not None:
      neighbor_kwargs.setdefault('box', neighbor_box)
    else:
      neighbor_kwargs.pop('box', None)
    return neighbor_fn.allocate(positions, **neighbor_kwargs)

  def init_fn(positions, **kwargs):
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    dim = int(positions.shape[1])
    box_matrix = current_box_matrix(
        displacement_fn, box_fn, dim,
        fractional_coordinates=fractional_coordinates, **kwargs)
    neighbors = _allocate(positions, box_matrix, **kwargs)
    return NearFieldState(
        neighbors=neighbors,
        box_matrix=box_matrix,
        fractional_coordinates=fractional_coordinates,
    )

  def _edge_geometry(positions, neighbor_idx, neighbor_mask, box_matrix):
    """Shared edge list + geometry + interpolated scalars (receiver i, sender j).

    Returns ``(receivers, senders, rhat, scalars, edge_mask, N)`` with all
    masked-off edges zeroed.  Used by both the full resistance apply and the
    on-device FU-diagonal extraction so they stay in lock-step.
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    box_matrix = jnp.asarray(box_matrix, dtype=REAL_DTYPE)
    N = positions.shape[0]

    if fractional_coordinates:
      positions_frac = positions
    else:
      positions_frac = space.transform(jnp.linalg.inv(box_matrix), positions)

    # Dense -> flat directed edge list (receiver i, sender j).
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    if neighbor_idx.ndim == 1:
      neighbor_idx = neighbor_idx[:, None]
      neighbor_mask = neighbor_mask[:, None]
    max_k = neighbor_idx.shape[1]
    receivers = jnp.repeat(jnp.arange(N, dtype=jnp.int32), max_k)
    senders = jnp.where(neighbor_mask, neighbor_idx, 0).ravel()
    flat_mask = neighbor_mask.ravel()

    # Minimum-image displacement (handles shear via the live deformed box).
    delta_frac = positions_frac[senders] - positions_frac[receivers]
    delta_frac = jnp.mod(delta_frac + 0.5, 1.0) - 0.5
    rij = space.transform(box_matrix, delta_frac)
    r2 = jnp.sum(rij * rij, axis=-1)

    is_self = receivers == senders
    edge_mask = flat_mask & (~is_self) & (r2 < r_lub2) & (r2 > 0.0)

    safe_r = jnp.where(edge_mask, jnp.sqrt(jnp.maximum(r2, 1e-30)), 1.0)
    rhat = jnp.where(edge_mask[:, None], rij / safe_r[:, None], 0.0)

    scalars = nf_table.interpolate_scalars(safe_r, a, table)
    scalars = jnp.where(edge_mask[:, None], scalars, 0.0)
    return receivers, senders, rhat, scalars, edge_mask, N

  @jax.jit
  def _core(positions, gen_velocity, neighbor_idx, neighbor_mask, box_matrix):
    gen_velocity = jnp.asarray(gen_velocity, dtype=REAL_DTYPE)
    receivers, senders, rhat, scalars, edge_mask, N = _edge_geometry(
        positions, neighbor_idx, neighbor_mask, box_matrix)

    R_self, R_cross = _build_pair_operators(rhat, scalars, a, eta)
    g_recv = gen_velocity[receivers]
    g_send = gen_velocity[senders]
    contrib = (jnp.einsum('eab,eb->ea', R_self, g_recv)
               + jnp.einsum('eab,eb->ea', R_cross, g_send))
    contrib = jnp.where(edge_mask[:, None], contrib, 0.0)
    return ops.segment_sum(contrib, receivers, N)

  pref = _kim_karrila_prefactors(a, eta)
  _c = nf_table.COLUMN_INDEX

  @jax.jit
  def _core_diag_FU(positions, neighbor_idx, neighbor_mask, box_matrix):
    """On-device diagonal of the ``R^nf_FU`` (6N) block, packed ``(N, 6)``.

    ``diag(sum_e R_self_e) = sum_e diag(R_self_e)``, so this is the segment-sum
    of the per-edge FU self-block diagonal -- the on-device equivalent of the
    host :func:`sd_saddle.nearfield_FU_diagonal`.  Feeds the diagonal-Schur
    preconditioner ``S~ = zeta I + diag(R^nf_FU)``.

    Only the translational ``A`` and rotational ``C`` self-blocks have nonzero
    FU diagonal (``A11 = XA*P + YA*Iperp`` etc.), so we evaluate just those four
    scalar families rather than the full 11x11 pair operators -- the diagonal
    entry along axis ``k`` is ``X*rhat_k^2 + Y*(1 - rhat_k^2)``.
    """
    receivers, _senders, rhat, scalars, edge_mask, N = _edge_geometry(
        positions, neighbor_idx, neighbor_mask, box_matrix)
    rh2 = rhat * rhat                                          # (E,3)
    XA = pref['A'] * scalars[:, _c['XA11']]
    YA = pref['A'] * scalars[:, _c['YA11']]
    XC = pref['C'] * scalars[:, _c['XC11']]
    YC = pref['C'] * scalars[:, _c['YC11']]
    diag_A = XA[:, None] * rh2 + YA[:, None] * (1.0 - rh2)     # (E,3) translation
    diag_C = XC[:, None] * rh2 + YC[:, None] * (1.0 - rh2)     # (E,3) rotation
    diag_edge = jnp.concatenate([diag_A, diag_C], axis=-1)     # (E,6)
    diag_edge = jnp.where(edge_mask[:, None], diag_edge, 0.0)
    diag6 = ops.segment_sum(diag_edge, receivers, N)          # (N,6)
    # A particle is "neighbored" iff it has >=1 live lubrication edge -- the
    # exact mask the near-field Brownian Shift/Proj separation needs (matches
    # the host ``nearfield_FU_diagonal`` ``has_neighbor``).
    has_neighbor = ops.segment_sum(
        edge_mask.astype(jnp.int32), receivers, N) > 0       # (N,)
    return diag6, has_neighbor

  @jax.jit
  def _prepare_core(positions, neighbor_idx, neighbor_mask, box_matrix):
    """One-time geometry + table + block assembly for a fixed configuration."""
    receivers, senders, rhat, scalars, edge_mask, N = _edge_geometry(
        positions, neighbor_idx, neighbor_mask, box_matrix)
    R_self, R_cross = _build_pair_operators(rhat, scalars, a, eta)
    m = edge_mask[:, None, None]
    R_self = jnp.where(m, R_self, 0.0)
    R_cross = jnp.where(m, R_cross, 0.0)
    self_blocks = ops.segment_sum(R_self, receivers, N)
    has_neighbor = ops.segment_sum(
        edge_mask.astype(jnp.int32), receivers, N) > 0
    return PreparedNearField(
        self_blocks=self_blocks,
        cross_blocks=R_cross,
        receivers=receivers,
        senders=senders,
        has_neighbor=has_neighbor,
    )

  def prepare_fn(state, positions):
    """Precompute :class:`PreparedNearField` reusing ``state.neighbors`` as-is.

    Same fixed-configuration contract as ``apply_prepared``: the caller
    guarantees ``state.neighbors`` / ``state.box_matrix`` were built for
    ``positions``.  Amortize over repeated matvecs via
    :func:`apply_prepared_blocks` / :func:`apply_prepared_blocks_FU`.
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    return _prepare_core(positions, state.neighbors.idx,
                         partition.neighbor_list_mask(state.neighbors),
                         state.box_matrix)

  def _update_neighbors(state, positions, box_matrix, **kwargs):
    # Always pass the resolved box explicitly in fractional coordinates: the
    # stored neighbor list was allocated with a matrix ``box`` (worst-case
    # shear under a live ``box_fn``), so an update without one would fall back
    # to the builder's scalar default and the two ``lax.cond`` branches inside
    # ``partition.neighbor_list`` would carry mismatched box pytree leaves
    # (scalar vs (3,3)).  Mirrors the real-space update in ``rpy_real_det``.
    neighbor_kwargs = dict(kwargs)
    neighbor_box = _neighbor_box_from_matrix(box_matrix, fractional_coordinates)
    if neighbor_box is not None:
      neighbor_kwargs.setdefault('box', neighbor_box)
    else:
      neighbor_kwargs.pop('box', None)
    return state.neighbors.update(positions, **neighbor_kwargs)

  def apply_fn(state, positions, gen_velocity, **kwargs):
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    dim = int(positions.shape[1])
    box_matrix = current_box_matrix(
        displacement_fn, box_fn, dim,
        fractional_coordinates=fractional_coordinates, **kwargs)
    neighbors = _update_neighbors(state, positions, box_matrix, **kwargs)
    gen_force = _core(positions, gen_velocity, neighbors.idx,
                      partition.neighbor_list_mask(neighbors), box_matrix)
    next_state = NearFieldState(
        neighbors=neighbors,
        box_matrix=box_matrix,
        fractional_coordinates=fractional_coordinates,
    )
    return gen_force, next_state

  def diagonal_FU_fn(state, positions, **kwargs):
    """``(N, 6)`` diagonal of the near-field ``R^nf_FU`` block (on-device)."""
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    dim = int(positions.shape[1])
    box_matrix = current_box_matrix(
        displacement_fn, box_fn, dim,
        fractional_coordinates=fractional_coordinates, **kwargs)
    neighbors = _update_neighbors(state, positions, box_matrix, **kwargs)
    diag6, _ = _core_diag_FU(positions, neighbors.idx,
                             partition.neighbor_list_mask(neighbors), box_matrix)
    return diag6

  def apply_prepared_fn(state, positions, gen_velocity):
    """Apply ``R^nf`` reusing ``state.neighbors`` as-is (no ``.update()``).

    For repeated matvecs at a FIXED configuration (e.g. one GMRES solve): the
    caller guarantees ``state.neighbors`` and ``state.box_matrix`` were built
    for ``positions``, so the per-matvec neighbor rebuild -- otherwise paid on
    every iteration -- is skipped.  Returns just the ``(N, 11)`` gen-force.
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    return _core(positions, gen_velocity, state.neighbors.idx,
                 partition.neighbor_list_mask(state.neighbors),
                 state.box_matrix)

  def diagonal_FU_prepared_fn(state, positions):
    """``diagonal_FU`` reusing ``state.neighbors`` as-is (no ``.update()``)."""
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    diag6, _ = _core_diag_FU(positions, state.neighbors.idx,
                             partition.neighbor_list_mask(state.neighbors),
                             state.box_matrix)
    return diag6

  def diagonal_FU_mask_prepared_fn(state, positions):
    """``(diag6 (N,6), has_neighbor (N,))`` reusing ``state.neighbors`` as-is.

    The near-field Brownian square root needs both the diagonal (for the Jacobi
    split ``D``) and the exact neighbored mask (for the Shift/Proj separation).
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    return _core_diag_FU(positions, state.neighbors.idx,
                         partition.neighbor_list_mask(state.neighbors),
                         state.box_matrix)

  apply_fn.diagonal_FU = diagonal_FU_fn
  apply_fn.apply_prepared = apply_prepared_fn
  apply_fn.diagonal_FU_prepared = diagonal_FU_prepared_fn
  apply_fn.diagonal_FU_mask_prepared = diagonal_FU_mask_prepared_fn
  apply_fn.prepare = prepare_fn
  apply_fn.apply_blocks = apply_prepared_blocks
  apply_fn.apply_blocks_FU = apply_prepared_blocks_FU
  apply_fn.prepared_diag_FU = prepared_diag_FU

  return init_fn, apply_fn


# ---------------------------------------------------------------------------
# Test / introspection helpers
# ---------------------------------------------------------------------------
def pair_grand_resistance(rhat, r, a, eta):
  """Dense per-pair 22x22 grand near-field resistance for a single pair.

  Returns the symmetric ``(22, 22)`` block
  ``[[R_self, R_cross], [R_cross^T-style, R_self]]`` acting on
  ``[g_i (11), g_j (11)]``.  Built by reusing :func:`_build_pair_operators` for
  both edge orientations (i<-j and j<-i).  Intended for unit tests.
  """
  rhat = jnp.asarray(rhat, dtype=REAL_DTYPE)[None, :]
  r = jnp.asarray(r, dtype=REAL_DTYPE).reshape((1,))
  table = nf_table.load_resistance_table()
  scal = nf_table.interpolate_scalars(r, a, table)
  # Edge i<-j uses rhat (i->j); edge j<-i uses -rhat.
  R_self_ij, R_cross_ij = _build_pair_operators(rhat, scal, a, eta)
  R_self_ji, R_cross_ji = _build_pair_operators(-rhat, scal, a, eta)
  R = jnp.zeros((2 * N_GEN, 2 * N_GEN), dtype=REAL_DTYPE)
  R = R.at[:N_GEN, :N_GEN].set(R_self_ij[0])
  R = R.at[:N_GEN, N_GEN:].set(R_cross_ij[0])
  R = R.at[N_GEN:, N_GEN:].set(R_self_ji[0])
  R = R.at[N_GEN:, :N_GEN].set(R_cross_ji[0])
  return R
