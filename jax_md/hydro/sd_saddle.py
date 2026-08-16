"""Deterministic saddle-point solve for Fast Stokesian Dynamics (Phase 2).

Couples the far-field grand mobility ``M`` (PSE), the stresslet-constraint
projectors ``B``/``Bᵀ``, and the Phase-1 near-field lubrication resistance
``R^nf`` into the full Stokesian Dynamics resistance

    R_FU = Bᵀ M⁻¹ B + R^nf_FU

*without* inverting ``M``.  ``R_FU`` is the Schur complement of the symmetric
indefinite saddle-point matrix (Fiore & Swan 2019, "Fast Stokesian Dynamics",
Eq. 2.8)

    A = [[ M    B        ],
         [ Bᵀ  -R^nf_FU   ]]

which is solved matrix-free with GMRES and a block preconditioner.  Everything
is additive: this module is the only caller of the SD path; the RPY and
stresslet-RPY mobility paths are untouched.

Conventions (both gates resolved analytically -- see the project plan):

* **Moment space** is the grand-mobility flat-11 layout
  ``[F(3), couplet_orthonormal(8)]`` (``rpy_moments.grand_to_flat``); the 8
  couplet coordinates split into antisymmetric(3)=torque and symmetric(5)=
  stresslet.  Euclidean inner product on flat-11 equals the Frobenius pairing,
  so ``M`` is symmetric in these coordinates.
* **FU space** is the physical ``[U(3) | Omega(3)]`` / ``[F(3) | L(3)]`` stack
  ``(N, 6)``, matching the near-field operator.
* **B / Bᵀ (Gate A).**  ``B(U,Omega) = grand_to_flat(U, rot_embed(Omega))`` with
  ``rot_embed(Omega) = -eps . Omega = 2 * torque_to_couplet(Omega)`` the
  *physical* rigid-rotation velocity gradient (no 1/2).  Its exact Euclidean
  adjoint is ``Bᵀ q = (q[:3], couplet_to_stresslet_torque(C)[1])`` -- the inverse
  torque-extraction, which here *coincides* with the physical torque, so
  adjointness and physical correctness do not conflict.  (Distinct from the
  mobility's own D2WE path, which uses ``decompose_gradient``/
  ``torque_to_couplet``; do not conflate the two.)
* **Near-field signs (Gate B).**  The task's ``b2``/``S`` equations are in the
  symmetric ``(U,Omega,E)`` convention, which is exactly jax-md's near-field
  convention (``-R_SU``, ``+R_SE``, ``+R_FE``).  Use the near-field outputs
  directly; do NOT re-transcribe FSD ``Lubrication.cu`` signs (jax-md already
  corrected the FE flip in Phase 1).

Box deformation (Lees-Edwards shear): when the space carries a ``box_fn``, the
solve accepts runtime shear kwargs (``gamma_xy``/``gamma_xz``/``gamma_yz`` or
``shear=``).  The deformed box is threaded into the real-space kernel and the
*exact* deformed-box grand wave operator (``_apply_wave_exact_grand`` rebuilds
the screened k-modes for the deformed reciprocal lattice); the near-field
already minimum-images via its stored ``box_matrix``.  The imposed rate-of-strain
``E^inf`` enters the RHS (resistance problem); the full ambient velocity gradient
``L_inf`` drives the background-flow add-back ``info['U_inf']``/``Omega_inf']``.
With no ``box_fn`` (static box) the path is bit-for-bit the original behavior.

Module layout (top to bottom):

* **Projectors** -- ``rot_embed`` / ``b_apply`` / ``bt_apply`` /
  ``stresslet_from_moment`` (Gate A) and the ``_gv_from_*`` near-field packers.
* **Input normalization** -- ``_resolve_e_inf`` (strain conventions) and
  ``_normalize_solve_inputs`` (all optional ``solve_fn`` inputs -> concrete
  arrays; zeros mean "absent").
* **Fixed-configuration operator factories** -- ``_make_real_grand_mv`` and
  ``_make_wave_grand_mv`` (separately differentiable far-field pieces),
  ``_make_grand_mv`` (their sum), and ``_make_rnf`` (near-field block applies
  from one ``prepare`` pass).
* **Saddle system** -- ``_saddle_operator`` (the matrix ``A``),
  ``_saddle_rhs`` (``b1``/``b2``), ``_ambient_addback`` (``U_inf``/
  ``Omega_inf`` convenience outputs).
* **Preconditioners** -- one block-LDL apply ``_block_ldl_pinv`` parameterized
  by a signed Schur solve: jacobi (``-t2/zeta``), diagonal, Chebyshev
  (``_cheb_bounds`` + ``_make_cheb_schur_solve``) or host IC(0)
  (``_make_ic0_schur_solve``).
* **``build_saddle_solve``** -- binds the above to a concrete space/parameter
  set and returns ``(init_fn, solve_fn)``; nests only what must close over the
  builder state: ``init_fn``/``refresh_state``, the jitted ``_body_impl`` /
  ``_device_body`` (jit cache keyed on the static GMRES configuration),
  ``solve_fn``, its attached ``mobility_tangent`` implicit derivative, and the
  eager ``count_iterations`` validation harness (scipy GMRES over the SAME
  operator/RHS helpers).
* **Host IC(0) machinery** -- ``assemble_stilde`` / ``_ic0`` /
  ``Ic0Preconditioner`` / ``build_ic0_from_state`` (validation-oriented;
  everything else runs on device).
"""

import functools
import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import spsolve_triangular
from jax.scipy.sparse import linalg as sparse_linalg

from jax_md import dataclasses
from jax_md import partition
from jax_md import space

from jax_md.hydro.rpy import (
    build_rpy_mobility,
    estimate_rpy_params,
    RpyState,
    _apply_wave_exact_grand,
)
from jax_md.hydro.rpy_real_det_dipole import (
    mr_grand_apply_blocks,
    mr_grand_prepare,
)
from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE, current_box_matrix
from jax_md.hydro.rpy_real_lattice_helpers import _neighbor_box_from_matrix
from jax_md.hydro.rpy_moments import (
    couplet_to_stresslet_torque,
    decompose_gradient,
    flat_to_grand,
    grand_to_flat,
    orthonormal_to_couplet,
    stresslet_to_couplet,
    torque_to_couplet,
    traceless,
)
from jax_md.hydro import sd_nearfield_table as nf_table
from jax_md.hydro.sd_nearfield import (
    build_nearfield_resistance,
    NearFieldState,
    PreparedNearField,
    _build_pair_operators,
)


# ---------------------------------------------------------------------------
# B / Bᵀ projectors (Gate A -- exact adjoint pair)
# ---------------------------------------------------------------------------
def rot_embed(omega: jnp.ndarray) -> jnp.ndarray:
  """Rigid-rotation velocity gradient ``D_ij = -eps_ijk Omega_k``.

  Equals ``2 * torque_to_couplet(Omega)`` and satisfies
  ``decompose_gradient(rot_embed(Omega)) == (0, Omega)``.  NOTE the factor of 2
  vs ``torque_to_couplet`` (the -1/2 eps embed); do not substitute one for the
  other.
  """
  return 2.0 * torque_to_couplet(jnp.asarray(omega))


def b_apply(u6: jnp.ndarray) -> jnp.ndarray:
  """``B``: FU velocity ``(N,6)=[U|Omega]`` -> moment flat-11 ``(N,11)``."""
  u6 = jnp.asarray(u6)
  U, Omega = u6[..., :3], u6[..., 3:]
  return grand_to_flat(U, rot_embed(Omega))


def bt_apply(q11: jnp.ndarray) -> jnp.ndarray:
  """``Bᵀ``: moment flat-11 ``(N,11)`` -> FU force ``(N,6)=[F|L]``.

  Exact Euclidean adjoint of :func:`b_apply`: ``L_k = -eps_kmn C_mn`` is the
  inverse torque-extraction (``couplet_to_stresslet_torque``), which is also the
  physical torque.
  """
  q11 = jnp.asarray(q11)
  F = q11[..., :3]
  C = orthonormal_to_couplet(q11[..., 3:])
  L = couplet_to_stresslet_torque(C)[1]
  return jnp.concatenate([F, L], axis=-1)


def stresslet_from_moment(q11: jnp.ndarray) -> jnp.ndarray:
  """Far-field stresslet ``S^ff`` (orthonormal ``(N,5)``) from moment flat-11."""
  q11 = jnp.asarray(q11)
  C = orthonormal_to_couplet(q11[..., 3:])
  return couplet_to_stresslet_torque(C)[0]


# ---------------------------------------------------------------------------
# Near-field block extractions (symmetric (U,Omega,E) convention, Gate B)
# ---------------------------------------------------------------------------
def _gv_from_u6(u6: jnp.ndarray) -> jnp.ndarray:
  """Pack ``[U|Omega] (N,6)`` into the near-field gen-velocity ``[U,Omega,0] (N,11)``."""
  u6 = jnp.asarray(u6)
  zeros5 = jnp.zeros(u6.shape[:-1] + (5,), dtype=u6.dtype)
  return jnp.concatenate([u6, zeros5], axis=-1)


def _gv_from_e5(e5: jnp.ndarray) -> jnp.ndarray:
  """Pack strain ``E5 (N,5)`` into the near-field gen-velocity ``[0,0,E5] (N,11)``."""
  e5 = jnp.asarray(e5)
  zeros6 = jnp.zeros(e5.shape[:-1] + (6,), dtype=e5.dtype)
  return jnp.concatenate([zeros6, e5], axis=-1)


def _resolve_e_inf(E_inf, N, dtype):
  """Normalize an imposed rate-of-strain input to ``(e5 (N,5), E_inf_mat (3,3))``.

  ``E_inf`` may be ``None`` (no imposed strain), a single symmetric-traceless
  ``(3,3)`` gradient (symmetrized + broadcast), or orthonormal ``(N,5)``
  coefficients.  Resolved eagerly (Python branch on shape) so the jitted body
  receives plain arrays; ``E_inf_mat`` feeds only the ``U_inf`` add-back.
  """
  if E_inf is None:
    return (jnp.zeros((N, 5), dtype=dtype), jnp.zeros((3, 3), dtype=dtype))
  E_inf = jnp.asarray(E_inf, dtype=dtype)
  if E_inf.shape[-2:] == (3, 3):
    E_inf_mat = traceless(0.5 * (E_inf + jnp.swapaxes(E_inf, -1, -2)))
    e5_single = decompose_gradient(E_inf_mat)[0]
    e5 = jnp.broadcast_to(e5_single, (N, 5))
  else:
    e5 = jnp.broadcast_to(E_inf, (N, 5))
    E_inf_mat = stresslet_to_couplet(e5[0])  # for U_inf add-back only
  return e5, E_inf_mat


def _normalize_solve_inputs(positions_frac, force, torque, E_inf, L_inf,
                            slip_top, extra_force, x0):
  """Resolve ``solve_fn``'s optional inputs to concrete ``REAL_DTYPE`` arrays.

  Runs eagerly (Python ``None`` branches) so the jitted body never sees a
  ``None`` where an array is expected.  A zeroed array is the "absent" value
  for every optional input: zero applied force/torque, zero imposed strain,
  zero Brownian slip / extra force, and a zero (cold) GMRES warm start.

  Returns ``(positions_frac, fp6, e5, E_inf_mat, L_inf_mat, slip_arr, extra,
  x0)`` where ``fp6`` is the stacked ``[F|L] (N,6)`` applied generalized
  force; see :func:`_resolve_e_inf` for the strain conventions.
  """
  positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
  N = positions_frac.shape[0]
  dtype = REAL_DTYPE

  force = (jnp.zeros((N, 3), dtype=dtype) if force is None
           else jnp.asarray(force, dtype=dtype))
  torque = (jnp.zeros((N, 3), dtype=dtype) if torque is None
            else jnp.asarray(torque, dtype=dtype))
  fp6 = jnp.concatenate([force, torque], axis=-1)

  # Imposed strain in orthonormal (N,5) + (3,3) add-back matrix.
  e5, E_inf_mat = _resolve_e_inf(E_inf, N, dtype)
  # Full ambient velocity gradient for the add-back: default to the symmetric
  # rate-of-strain (no ambient vorticity -> Omega_inf = 0, Phase-2 behavior).
  L_inf_mat = (E_inf_mat if L_inf is None
               else jnp.asarray(L_inf, dtype=dtype))

  slip_arr = (jnp.zeros((N, 11), dtype=dtype) if slip_top is None
              else jnp.asarray(slip_top, dtype=dtype))
  extra = (jnp.zeros((N, 6), dtype=dtype) if extra_force is None
           else jnp.asarray(extra_force, dtype=dtype))
  if x0 is None:
    x0 = (jnp.zeros((N, 11), dtype=dtype), jnp.zeros((N, 6), dtype=dtype))
  return positions_frac, fp6, e5, E_inf_mat, L_inf_mat, slip_arr, extra, x0


# ---------------------------------------------------------------------------
# Fixed-configuration operator factories
# ---------------------------------------------------------------------------
def _make_real_grand_mv(rpy_state: RpyState, positions: jnp.ndarray,
                        current_box=None):
  """Prepared real-space grand matvec on flat-11 at fixed positions."""
  prepared_real = mr_grand_prepare(
      rpy_state.real, positions, box_matrix=current_box)

  def real_mv_flat(q11: jnp.ndarray) -> jnp.ndarray:
    force, couplet = flat_to_grand(q11)
    velocity, gradient = mr_grand_apply_blocks(
        prepared_real, force, couplet)
    return grand_to_flat(velocity, traceless(gradient))
  return real_mv_flat


def _make_wave_grand_mv(rpy_state: RpyState, positions: jnp.ndarray,
                        current_box=None, *, wave_static, a, xi, eta):
  """Wave-space grand matvec on flat-11 at fixed positions and box."""
  def wave_mv_flat(q11: jnp.ndarray) -> jnp.ndarray:
    force, couplet = flat_to_grand(q11)
    if current_box is None:
      velocity, gradient = rpy_state.wave.apply_fn(
          positions, force, couplet)
    else:
      velocity, gradient = _apply_wave_exact_grand(
          static=wave_static, current_box=current_box,
          positions_frac=positions, forces=force, couplets=couplet,
          a=a, xi=xi, eta=eta)
    return grand_to_flat(velocity, traceless(gradient))
  return wave_mv_flat


def _make_grand_mv(rpy_state: RpyState, positions: jnp.ndarray,
                   current_box=None, *, wave_static, a, xi, eta):
  """Far-field grand matvec on flat-11 (real + wave, fixed state).

  With ``current_box`` (live shear) the real-space kernel runs under the
  deformed box and the wave-space operator is re-evaluated *exactly*
  (``_apply_wave_exact_grand`` rebuilds the screened k-modes for the deformed
  reciprocal lattice -- the position-remap-only path in ``Mw_core`` keeps the
  base-box modes and is wrong under shear).  ``current_box=None`` is the
  static-box path (bit-for-bit the Phase-2 behavior).
  """
  real_mv_flat = _make_real_grand_mv(rpy_state, positions, current_box)
  wave_mv_flat = _make_wave_grand_mv(
      rpy_state, positions, current_box,
      wave_static=wave_static, a=a, xi=xi, eta=eta)

  def grand_mv_flat(q11: jnp.ndarray) -> jnp.ndarray:
    return real_mv_flat(q11) + wave_mv_flat(q11)
  return grand_mv_flat


def _make_rnf(nf_apply, nf_state: NearFieldState, positions: jnp.ndarray,
              zero_nf: bool, prepared: Optional[PreparedNearField] = None):
  """Fixed-config near-field block applies; ``zero_nf`` forces R^nf=0.

  Precomputes the per-pair resistance blocks ONCE per solve
  (``nf_apply.prepare``: geometry + table interpolation + 11x11 assembly at
  the fixed ``positions``), so every subsequent matvec -- the GMRES operator,
  the Chebyshev-Schur inner loop (``cheb_degree`` applies per iteration), the
  power-iteration bounds, the RHS and the output stresslet -- reduces to
  gather -> batched block multiply -> segment_sum.  This is the FSD
  amortize-per-step pattern; equivalence to the matrix-free reference apply
  is pinned by ``test_prepared_blocks_match_core``.

  ``prepared`` optionally supplies blocks already built by
  ``nf_apply.prepare(nf_state, positions)`` so a caller holding several
  consumers at one fixed configuration (the Brownian step) prepares once.

  Returns ``(rnf_FU, rnf_FE, rnf_SU, rnf_SE, diag_FU)`` where ``diag_FU()``
  reads the FU diagonal off the same prepared blocks (no extra geometry
  pass).
  """
  if zero_nf:
    n = positions.shape[0]

    def rnf_FU(u6):
      return jnp.zeros_like(u6)

    def rnf_FE(e5):
      return jnp.zeros(e5.shape[:-1] + (6,), dtype=e5.dtype)

    def rnf_SU(u6):
      return jnp.zeros(u6.shape[:-1] + (5,), dtype=u6.dtype)

    def rnf_SE(e5):
      return jnp.zeros_like(e5)

    def diag_FU():
      return jnp.zeros((n, 6), dtype=REAL_DTYPE)

    return rnf_FU, rnf_FE, rnf_SU, rnf_SE, diag_FU

  if prepared is None:
    prepared = nf_apply.prepare(nf_state, positions)

  def nf_full(gv11: jnp.ndarray) -> jnp.ndarray:
    return nf_apply.apply_blocks(prepared, gv11)

  def rnf_FU(u6):
    # FU sub-blocks directly: the cheb/GMRES hot path at ~(6/11)^2 the flops.
    return nf_apply.apply_blocks_FU(prepared, u6)

  def rnf_FE(e5):
    return nf_full(_gv_from_e5(e5))[..., :6]

  def rnf_SU(u6):
    return nf_full(_gv_from_u6(u6))[..., 6:11]

  def rnf_SE(e5):
    return nf_full(_gv_from_e5(e5))[..., 6:11]

  def diag_FU():
    return nf_apply.prepared_diag_FU(prepared)

  return rnf_FU, rnf_FE, rnf_SU, rnf_SE, diag_FU


# ---------------------------------------------------------------------------
# Saddle system: matrix apply, right-hand side, ambient add-back
# ---------------------------------------------------------------------------
def _saddle_operator(grand_mv_flat, rnf_FU):
  """Matvec of the saddle matrix ``A = [[M, B], [Bᵀ, -R^nf_FU]]`` (Eq. 2.8).

  Acts on the solution pytree ``x = (q11 (N,11) far-field moments,
  u6 (N,6) [U|Omega])`` and returns the matching ``(flat-11, FU-6)`` pair.
  Built from a fixed-configuration far-field matvec (:func:`_make_grand_mv`)
  and near-field FU apply (:func:`_make_rnf`).

  Shared by the jitted GMRES body and the eager ``count_iterations``
  validation harness, so both solve the same system by construction.
  """
  def apply_A(x):
    q11, u6 = x
    top = grand_mv_flat(q11) + b_apply(u6)
    bot = bt_apply(q11) - rnf_FU(u6)
    return (top, bot)
  return apply_A


def _saddle_rhs(fp6, e5, rnf_FE, slip_top=None, extra=None):
  """Right-hand side ``(b1, b2)`` of the saddle system.

  b1 = (0_rigid, E^inf) in the velocity-output flat-11 (strain slots), plus
  the optional far-field Brownian slip ``U^B`` (Phase 3).
  b2 = -(F^P + extra_force + R^nf_FE : E^inf) in FU force space, where
  ``extra`` is the optional Phase-3 bottom-block force (near-field Brownian
  force for the main solve, RFD displacement for the drift solves).

  ``slip_top``/``extra`` default to ``None`` (omitted) for the deterministic
  validation paths that never carry Brownian terms.
  """
  N = fp6.shape[0]
  b1 = grand_to_flat(
      jnp.zeros((N, 3), dtype=fp6.dtype), stresslet_to_couplet(e5))
  if slip_top is not None:
    b1 = b1 + slip_top
  if extra is not None:
    b2 = -(fp6 + extra + rnf_FE(e5))
  else:
    b2 = -(fp6 + rnf_FE(e5))
  return b1, b2


def _warn_nonconvergence(converged, rel_residual, tol, atol):
  """Print a GMRES non-convergence warning, as the original FSD code does.

  Emitted from inside the jitted solve via ``jax.debug.print`` under a
  ``lax.cond``, so the host callback runs *only* on the solves that actually
  fail; a converged solve pays one scalar predicate.  ``tol``/``atol`` are
  static Python floats (the requested target); ``converged``/``rel_residual``
  are traced scalars measured on the true residual.
  """
  def _emit(r):
    jax.debug.print(
        '  SD saddle GMRES failed to converge: true relative residual = {r} '
        '(requested tol={t}, atol={a}).  jax GMRES stops on the '
        'PRECONDITIONED residual, so this can trip well inside the iteration '
        'budget; on contact-rich configurations the usual cause is saddle '
        'conditioning rather than the budget (see JAX_MD_SD_MIN_GAP).',
        r=r, t=tol, a=atol)
    return None
  jax.lax.cond(converged, lambda _r: None, _emit, rel_residual)


def _ambient_addback(L_inf_mat, positions_frac, box):
  """Background-flow add-back ``(U_inf (N,3), Omega_inf (N,3))``.

  Convenience outputs only -- the solve itself works in the relative frame
  (the pinned convention); callers add these back if they want lab-frame
  velocities.  Evaluated at the box-centred Cartesian positions
  ``r = box . (q - 1/2)``.

  The translational add-back uses the FULL velocity gradient ``L_inf_mat``
  (``u^inf = L . r``), not just the symmetric rate-of-strain, so the ambient
  vorticity is included; ``Omega_inf = 1/2 curl u^inf`` is the angular
  add-back (a torque-free sphere co-rotates with the ambient spin).
  With ``L_inf_mat == E_inf_mat`` (symmetric, the default when no spin is
  supplied) this reduces bit-for-bit to the Phase-2 behavior (Omega_inf = 0).

  Omega_inf_k = 1/2 (curl u^inf)_k = 1/2 eps_kij d_i u^inf_j with
  d_i u^inf_j = L_ji.  Simple-shear check: L[0,1]=gamma_dot
  (u_x = gamma_dot * y) => Omega_z = 1/2 (L[1,0]-L[0,1]) = -gamma_dot/2
  (fluid above moves +x, below -x: the sphere rolls clockwise in the xy
  plane).  Same convention as ``rpy_moments.decompose_gradient``.
  """
  N = positions_frac.shape[0]
  dtype = positions_frac.dtype
  cart = space.transform(box, positions_frac - jnp.asarray(0.5, dtype=dtype))
  U_inf = jnp.einsum('ij,nj->ni', L_inf_mat, cart)
  Omega_inf_vec = 0.5 * jnp.stack([
      L_inf_mat[2, 1] - L_inf_mat[1, 2],
      L_inf_mat[0, 2] - L_inf_mat[2, 0],
      L_inf_mat[1, 0] - L_inf_mat[0, 1],
  ])
  Omega_inf = jnp.broadcast_to(Omega_inf_vec, (N, 3))
  return U_inf, Omega_inf


# ---------------------------------------------------------------------------
# Saddle preconditioners (block-LDL applies differing only in the Schur solve)
# ---------------------------------------------------------------------------
def _block_ldl_pinv(zeta, schur_solve):
  """Block-LDL preconditioner apply for the saddle matrix, given a Schur solve.

  Exact block-LDL inverse of A = [[M, B], [Bᵀ, -R^nf_FU]] with M^-1 ~ zeta I
  and (negative) Schur S = -(R^nf_FU + Bᵀ M^-1 B) ~ -S~:
    t2 = y2 - Bᵀ M^-1 y1         (note the zeta on Bᵀ y1)
    z2 = S^-1 t2 = schur_solve(t2)   (the SIGNED approximate Schur inverse)
    z1 = M^-1 y1 - M^-1 B z2 = zeta y1 - zeta B z2
  Dropping the zeta factors / the Schur sign (as a previous version did)
  leaves sigma(P A) straddling zero and ~doubles the GMRES iteration count.

  ``schur_solve``: ``t2 (N,6) -> z2 (N,6)`` applying ``-S~^-1`` for the
  chosen Schur approximation ``S~`` (jacobi ``zeta I``, diagonal, Chebyshev,
  IC(0)); it must be a fixed LINEAR map (required for GMRES).
  """
  def apply_pinv(x):
    y1, y2 = x                       # y1 (N,11) moment, y2 (N,6) FU
    t2 = y2 - zeta * bt_apply(y1)    # y2 - Bᵀ M^-1 y1
    z2 = schur_solve(t2)             # S^-1 t2
    z1 = zeta * y1                   # M^-1 y1
    return (z1 - zeta * b_apply(z2), z2)
  return apply_pinv


def _cheb_bounds(stil, diag_S, *, zeta, power_iters, safety):
  """(lo, hi) eigenvalue bounds of D^{-1/2} S~ D^{-1/2}, D = diag_S.

  lo: rigorous lower bound zeta/max(diag_S) (R^nf_FU >= 0).
  hi: safety * Rayleigh-quotient power-iteration estimate of lambda_max,
      seeded with a fixed RANDOM vector (NOT the RHS, and NOT a structured
      vector -- on a symmetric lattice a structured seed lies in an invariant
      subspace that misses the dominant eigenvector, so ``hi`` underestimates
      lambda_max and the Chebyshev polynomial diverges with degree).
  """
  lo = zeta / jnp.max(diag_S)
  dinv_sqrt = jax.lax.rsqrt(diag_S)               # D^{-1/2}, (N,6)
  # Fixed random seed: deterministic (keeps the preconditioner linear) but
  # breaks lattice symmetry so power iteration finds the true lambda_max.
  v = jax.random.normal(jax.random.PRNGKey(0), diag_S.shape, dtype=diag_S.dtype)
  v = v / jnp.sqrt(jnp.vdot(v, v).real)
  hi = lo
  for _ in range(power_iters):
    w = dinv_sqrt * stil(dinv_sqrt * v)           # D^{-1/2} S~ D^{-1/2} v
    hi = jnp.vdot(v, w).real                       # Rayleigh quotient
    nrm = jnp.sqrt(jnp.vdot(w, w).real)
    v = w / jnp.maximum(nrm, 1e-300)
  hi = jnp.maximum(hi * safety, lo * (1.0 + 1e-6))
  return lo, hi


def _make_cheb_schur_solve(diag_S, stil, lo, hi, degree):
  """Jacobi-preconditioned degree-``degree`` Chebyshev approximation of S~^-1.

  ``S~ = zeta I + R^nf_FU`` (SPD), applied matrix-free as ``stil(v)``.  All
  applies are near-field (no FFTs) so the per-GMRES-iteration cost is small;
  the win is fewer (expensive) GMRES iterations.  The spectral bounds
  ``(lo, hi)`` of the Jacobi-scaled operator are computed ONCE per solve
  (independent of the RHS), so the returned map is a fixed LINEAR operator --
  required for GMRES.  Returns the POSITIVE solve ``t2 -> ~S~^-1 t2`` (the
  caller wires the Schur sign).
  """
  theta = 0.5 * (hi + lo)
  delta = 0.5 * (hi - lo)

  def schur_solve(t2):
    # Preconditioned Chebyshev iteration (Saad, Alg. 12.1) for S~ y = t2,
    # D = diag_S as the inner Jacobi preconditioner.  Returns y ~ S~^-1 t2.
    y = jnp.zeros_like(t2)
    r = t2
    p = jnp.zeros_like(t2)
    alpha = 1.0 / theta
    for i in range(degree):
      z = r / diag_S
      if i == 0:
        p = z
        alpha = 1.0 / theta
      else:
        beta = (delta * alpha * 0.5) ** 2
        alpha = 1.0 / (theta - beta / alpha)
        p = z + beta * p
      y = y + alpha * p
      r = r - alpha * stil(p)
    return y

  return schur_solve


def _make_ic0_schur_solve(ic0):
  """Signed Schur solve from the host RCM + incomplete-Cholesky factor.

  ``ic0.solve`` applies ``S~^-1`` with ``S~ = zeta I + R~^nf_FU`` (positive
  definite); the true Schur is ``S = -S~``, so ``z2 = S^-1 t2 =
  -ic0.solve(t2)``.  The host solve is bridged into the traced GMRES via
  ``jax.pure_callback``.
  """
  def _host_solve(v):
    return np.asarray(ic0.solve(np.asarray(v, dtype=np.float64)),
                      dtype=np.float64)

  def schur_solve(t2):
    flat = t2.reshape(-1)
    z2flat = jax.pure_callback(
        _host_solve, jax.ShapeDtypeStruct(flat.shape, REAL_DTYPE), flat)
    return -z2flat.reshape(t2.shape)

  return schur_solve


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SaddleState:
  """Per-configuration state for the saddle solve.

  ``positions`` are the fractional coordinates the neighbor lists (and any
  host-side preconditioner factor) were built for; ``solve_fn`` must be called
  with the same configuration (Phase-2 static probes always are).
  """
  rpy: RpyState
  nf: NearFieldState
  positions: jnp.ndarray


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_saddle_solve(
    space_fns,
    a: float,
    eta: float,
    *,
    xi: Optional[float] = None,
    n_particles: Optional[int] = None,
    phi: Optional[float] = None,
    tol: float = 1e-3,
    gmres_tol: float = 1e-3,
    gmres_restart: int = 50,
    gmres_maxiter: int = 20,
    gmres_solve_method: str = 'batched',
    r_lub: Optional[float] = None,
    r_p: Optional[float] = None,
    preconditioner: str = 'cheb',
    cheb_degree: int = 24,
    cheb_power_iters: int = 12,
    cheb_safety: float = 1.2,
    fractional_coordinates: bool = True,
    nf_capacity_multiplier: Optional[float] = None,
    nf_extra_capacity: int = 0,
    nf_neighbor_format: Optional[partition.NeighborListFormat] = None,
    **rpy_kwargs,
):
  """Build the deterministic FSD saddle-point solve.

  Args:
    space_fns: ``(displacement_fn, shift_fn)`` (static box; Phase 2).
    a, eta: sphere radius and solvent viscosity.
    xi: Ewald split.  If None, estimated from ``tol`` via
      ``estimate_rpy_params`` (requires ``n_particles`` and ``phi``).
    n_particles, phi: needed only for ``xi`` estimation / cost-optimal split.
    tol: target accuracy used by the parameter estimator when ``xi`` is None.
    gmres_tol: relative GMRES tolerance on the saddle residual.  NOTE that
      ``jax``'s GMRES tests the *preconditioned* residual ``||M(b - Ax)||``
      against ``max(tol ||b||, atol)`` with an unpreconditioned ``||b||``, so
      the tolerance actually delivered on the true residual is looser than
      ``tol`` by roughly the scale of ``M``.  The original FSD code monitors
      the true residual instead; ``info['rel_residual']`` and
      ``info['converged']`` report that (FSD-comparable) criterion, and a
      warning is printed whenever it is not met.
    gmres_restart, gmres_maxiter: Krylov basis size and number of restarts.
      The default ``50 x 20 = 1000`` matches the original FSD code's restart
      length and iteration limit.  Both of jax's GMRES loops are
      ``lax.while_loop``s with early exit and the Krylov memory depends only
      on ``restart``, so a larger ``maxiter`` costs nothing on solves that
      converge -- it only buys headroom on the ones that would otherwise be
      truncated silently.
    gmres_solve_method: which of jax's two GMRES implementations to use,
      ``'batched'`` (default) or ``'incremental'``.  They differ in how the
      Krylov least-squares problem is solved *within* one restart:

      * ``'batched'`` builds the full ``restart``-dimensional basis and then
        solves the least-squares problem from scratch.  It ignores the
        within-restart tolerance entirely (jax's ``_gmres_batched`` does
        ``del ptol``), so it always runs the full ``restart`` matvecs unless
        it hits an exact Arnoldi breakdown -- and therefore routinely
        *overshoots* the requested tolerance.
      * ``'incremental'`` builds the QR incrementally with Givens rotations
        and stops inside a restart as soon as its residual *estimate* meets
        ``ptol``.

      **Keep the default.**  ``'incremental'``'s early exit looks like free
      savings and is not: measured on a disordered N=108 phi=0.50 config
      (min gap 0.073a, ``'cheb'`` preconditioner), matching the two on
      *delivered* true residual rather than on the nominal tolerance gives

        resid 3.5e-3: batched 13 matvecs | incremental 14
        resid 4.7e-6: batched 33 matvecs | incremental 34
        resid 1.8e-9: batched 53 matvecs | incremental 53   (2.09 s vs 2.62 s)

      i.e. the same Krylov space and the same iteration count, plus a ~25%
      wall-clock penalty from the per-iteration Givens work.  The apparent
      speedup of ``'incremental'`` at a fixed nominal ``tol`` is entirely the
      accuracy it declines to deliver.

      The knob is exposed because the per-iteration overhead is
      architecture-dependent (jax documents ``'batched'`` as the GPU-friendly
      path, which is where this was measured to matter least), NOT because
      warm-started solves want ``'incremental'`` -- they specifically do not;
      see :func:`~jax_md.hydro.sd_brownian.rfd_drift`.
    r_lub: near-field cutoff (default ``4a``).
    r_p: near-field truncation for the IC(0) Schur factor (default ``2.1a``).
    preconditioner: default GMRES preconditioner.  ``'cheb'`` (the default) is
      the fully on-device Jacobi-preconditioned **Chebyshev** semi-iteration on
      the Schur block ``S~ = zeta I + R^nf_FU`` (matrix-free, no host work, no
      ``jax.pure_callback``, jittable).  On disordered (physical) suspensions it
      converges in fewer iterations than the host ``'ic0'`` factor; the
      near-field couplings it captures matter exactly when neighbour gaps vary,
      which is where ``'jacobi'`` collapses.  ``'jacobi'`` is the cruder pure
      block-Jacobi ``S~ = zeta I`` (cheapest, but degrades badly on disordered
      dense configs).  ``'diag'`` is the diagonal-Schur
      ``S~ = zeta I + diag(R^nf_FU)``.  ``'ic0'`` (RCM + zero-fill incomplete
      Cholesky of the truncated Schur ``S~ = zeta I + R~^nf_FU``) is built
      host-side per configuration and applied via ``jax.pure_callback`` --
      accurate but it forces a device->host round trip per GMRES iteration and
      cannot run under ``jit``/``vmap``; keep it for validation/comparison only.
      Overridable per call on ``solve_fn``.
    cheb_degree: number of Chebyshev iterations (matrix-free ``S~`` applies) per
      Schur solve for ``'cheb'`` (default 24).  Higher degree trades cheap
      near-field matvecs for fewer (expensive FFT) GMRES iterations.
    cheb_power_iters: power iterations used once per solve to bound the largest
      eigenvalue of the Jacobi-scaled Schur for ``'cheb'`` (default 12).
    cheb_safety: multiplicative safety factor (>=1) on the estimated largest
      eigenvalue so Chebyshev stays stable (default 1.2).
    nf_capacity_multiplier: capacity headroom for the near-field ``r_lub``
      neighbor list.  ``None`` (default) inherits ``capacity_multiplier`` from
      ``rpy_kwargs`` (else 1.25).  This scales the near-field edge count ``E``,
      which multiplies the cost of every near-field matvec -- and the
      Chebyshev-Schur preconditioner does ``cheb_degree`` of them per GMRES
      iteration, ~1e4 per SD step.  It is the single most expensive capacity
      knob in the solver: keep it near the expected max coordination growth
      (~3-4x for the **Dense** default, whose capacity is a per-particle buffer
      width) rather than inheriting a large far-field multiplier.  Under
      ``nf_neighbor_format=Sparse`` it instead means headroom on the *total*
      pair count -- real headroom, where Dense's ``N * max_k`` buffer happens to
      carry several times the live edge count -- so on systems that densify
      (aggregation, quenches) size it from the expected FINAL pair count and
      watch ``did_buffer_overflow``.
    nf_extra_capacity: additional per-particle slots for the near-field list
      (default 0).  NOT inherited from the ``extra_capacity`` in
      ``rpy_kwargs``.  Specified per particle for both formats, and multiplied
      by ``N`` internally to obtain total pair capacity for Sparse lists
      (near-field and far-field alike), so passing a total-pair count would
      cause an N-fold over-allocation.
    nf_neighbor_format: near-field neighbor-list format, or ``None`` (default)
      to take ``build_nearfield_resistance``'s own default (``Dense``).
      ``Dense`` sizes ``E = N * max_k`` from the single most-crowded particle,
      so heterogeneous configs (gels) pay the worst-case occupancy on every
      particle -- 2-3x more edges streamed than are live.  ``Sparse`` sizes from
      the true pair count instead and is the faster choice there, at the cost of
      Dense's accidental overflow slack.  Both enumerate the same directed edges
      and agree bit-for-bit.
    **rpy_kwargs: forwarded to ``build_rpy_mobility`` (e.g. ``P``, ``Mgrid``,
      ``rcut``, ``capacity_multiplier``, ``extra_capacity``).

  Returns:
    ``(init_fn, solve_fn)``.  The solve function exposes
    ``solve_fn.mobility_tangent`` for a random-response directional derivative
    of the configuration-dependent mobility and stresslet.
  """
  if preconditioner not in ('cheb', 'diag', 'ic0', 'jacobi'):
    raise ValueError(
        "preconditioner must be 'cheb', 'diag', 'ic0', or 'jacobi', got %r"
        % (preconditioner,))
  cheb_degree = int(cheb_degree)
  cheb_power_iters = int(cheb_power_iters)
  cheb_safety = float(cheb_safety)
  default_preconditioner = preconditioner
  if gmres_solve_method not in ('batched', 'incremental'):
    raise ValueError(
        "gmres_solve_method must be 'batched' or 'incremental', got %r"
        % (gmres_solve_method,))
  default_gmres_solve_method = gmres_solve_method
  displacement_fn = space_fns[0]
  box_fn = space_fns[2] if len(space_fns) > 2 else None

  zeta = 6.0 * math.pi * float(eta) * float(a)
  # Builder-level GMRES budget defaults (solve_fn shadows the names with its own
  # per-call override params, so capture them here for resolution).
  _default_gmres_restart = int(gmres_restart)
  _default_gmres_maxiter = int(gmres_maxiter)
  r_p = 2.1 * float(a) if r_p is None else float(r_p)
  r_lub_resolved = (nf_table.R_LUB_OVER_A * float(a) if r_lub is None
                    else float(r_lub))

  # -- Resolve xi (and any grid params the estimator supplies) --------------
  grid_kwargs = dict(rpy_kwargs)
  if xi is None:
    if n_particles is None or phi is None:
      raise ValueError(
          'xi is None: provide xi explicitly, or pass n_particles and phi so '
          'it can be estimated via estimate_rpy_params.')
    box = current_box_matrix(
        displacement_fn, box_fn, 3,
        fractional_coordinates=fractional_coordinates)
    est = estimate_rpy_params(tol, box, a, int(n_particles), float(phi))
    xi = float(est.xi)
    grid_kwargs.setdefault('P', int(est.P))
    grid_kwargs.setdefault('Mgrid', int(est.M))
    grid_kwargs.setdefault('rcut', float(est.rcut))

  # -- Far-field grand mobility (deterministic; no Brownian sampler) --------
  rpy_init, rpy_apply = build_rpy_mobility(
      space_fns, a, xi, eta,
      use_stresslet=True,
      constrained=False,
      include_brownian=False,
      fractional_coordinates=fractional_coordinates,
      **grid_kwargs,
  )
  # Live-box (shear) support: the resolved wave-space static factors let the
  # grand wave operator be re-evaluated *exactly* under a deformed box each
  # step (vs the static cached modes).  ``box_fn is None`` => static box, the
  # Phase-2 path, and ``_resolve_current_box`` returns None throughout.
  wave_static = rpy_apply.wave_static
  has_box_fn = box_fn is not None

  def _resolve_current_box(positions, **shear_kwargs):
    """Deformed box matrix from runtime shear kwargs, or None (static box)."""
    if not has_box_fn:
      return None
    dim = int(jnp.asarray(positions).shape[-1])
    return current_box_matrix(
        displacement_fn, box_fn, dim,
        fractional_coordinates=fractional_coordinates, **shear_kwargs)

  # -- Near-field resistance (Phase 1) -------------------------------------
  if nf_capacity_multiplier is None:
    nf_capacity_multiplier = rpy_kwargs.get('capacity_multiplier', 1.25)
  nf_kwargs = {}
  if nf_neighbor_format is not None:
    nf_kwargs['neighbor_format'] = nf_neighbor_format
  nf_init, nf_apply = build_nearfield_resistance(
      space_fns, a, eta,
      r_lub=r_lub,
      fractional_coordinates=fractional_coordinates,
      capacity_multiplier=float(nf_capacity_multiplier),
      extra_capacity=int(nf_extra_capacity),
      **nf_kwargs,
  )

  # -- init_fn -------------------------------------------------------------
  def init_fn(positions_frac, **shear_kwargs) -> SaddleState:
    """Allocate the saddle state.

    ``shear_kwargs`` (``gamma_xy``/``gamma_xz``/``gamma_yz`` or ``shear=``) set
    the initial deformed box; the underlying real/near-field builders allocate
    their neighbor lists at the *worst-case* shear box, so the fixed capacity
    stays valid for every strain in ``[-0.5, 0.5)`` over the run.
    """
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    rpy_state = rpy_init(positions_frac, **shear_kwargs)
    nf_state = nf_init(positions_frac, **shear_kwargs)
    return SaddleState(rpy=rpy_state, nf=nf_state, positions=positions_frac)

  def refresh_state(state: SaddleState, positions_frac,
                    **shear_kwargs) -> SaddleState:
    """Advance the state to ``positions_frac`` for stepping a simulation.

    Uses the **shape-preserving** neighbor-list ``.update()`` (NOT ``init`` /
    ``allocate``): the capacity stays fixed, so the jitted ``solve_fn`` /
    Brownian step compiles only **once**.  Re-allocating instead re-derives the
    capacity from the current positions, so as particles move the array shapes
    change and the whole (large) graph recompiles every step.  The wave state,
    lattice indices and ``core_fn`` are reused as-is, keeping the pytree
    treedef identical.

    Under live shear (``shear_kwargs`` supplied with a ``box_fn``) the deformed
    ``box_matrix`` is recomputed and stored on both the real and near-field
    states so the neighbor-list rebuild *and* the matrix-free minimum-image
    matvecs (the near-field ``apply_prepared`` reads ``nf.box_matrix``) use the
    live box.  With no ``box_fn`` this reduces to the static-box behavior.

    The fixed capacity (allocated by ``init_fn`` with ``capacity_multiplier``
    headroom) can in principle overflow under large displacements; the overflow
    flag rides along in ``neighbors.did_buffer_overflow`` -- reallocate with
    ``init_fn`` if it trips (accepting the one-off recompile)."""
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    current_box = _resolve_current_box(positions_frac, **shear_kwargs)

    real = state.rpy.real
    real_box = real.box_matrix if current_box is None else current_box
    rbox = _neighbor_box_from_matrix(real_box, real.fractional_coordinates)
    real_nbrs = (real.neighbors.update(positions_frac, box=rbox)
                 if rbox is not None else real.neighbors.update(positions_frac))
    real2 = dataclasses.replace(real, neighbors=real_nbrs, box_matrix=real_box)
    rpy_state = dataclasses.replace(state.rpy, real=real2)  # reuse wave/precond
    nf = state.nf
    nf_box = nf.box_matrix if current_box is None else current_box
    nbox = _neighbor_box_from_matrix(nf_box, nf.fractional_coordinates)
    nf_nbrs = (nf.neighbors.update(positions_frac, box=nbox)
               if nbox is not None else nf.neighbors.update(positions_frac))
    nf2 = dataclasses.replace(nf, neighbors=nf_nbrs, box_matrix=nf_box)
    return SaddleState(rpy=rpy_state, nf=nf2, positions=positions_frac)

  # -- Operator factories (fixed state), builder constants bound once -------
  make_grand_mv = functools.partial(
      _make_grand_mv, wave_static=wave_static, a=a, xi=xi, eta=eta)
  make_rnf = functools.partial(_make_rnf, nf_apply)

  # Block-Jacobi preconditioner (Schur approx S~ = zeta I): see
  # :func:`_block_ldl_pinv` for the LDL derivation and the zeta/sign pitfalls.
  jacobi_pinv = _block_ldl_pinv(zeta, lambda t2: -t2 / zeta)

  def _build_fixed_system(state, positions, current_box, pc, zero_nf,
                          ic0_obj=None, prepared_nf=None):
    """Build one fixed-configuration saddle operator and preconditioner.

    The returned objects may be reused for several right-hand sides.  This is
    essential for the implicit Brownian drift: its response and tangent solves
    must see exactly the same operator and solver preconditioner.
    """
    grand_mv_flat = make_grand_mv(state.rpy, positions, current_box)
    nf_state = (state.nf if current_box is None else
                dataclasses.replace(state.nf, box_matrix=current_box))
    rnf_FU, rnf_FE, rnf_SU, rnf_SE, diag_FU = make_rnf(
        nf_state, positions, zero_nf, prepared=prepared_nf)
    apply_A = _saddle_operator(grand_mv_flat, rnf_FU)

    if pc == 'ic0' and not zero_nf:
      preconditioner_fn = _block_ldl_pinv(
          zeta, _make_ic0_schur_solve(ic0_obj))
    elif pc in ('cheb', 'diag') and not zero_nf:
      diag_S = zeta + diag_FU()
      if pc == 'cheb':
        def stil(v):
          return zeta * v + rnf_FU(v)
        lo, hi = _cheb_bounds(stil, diag_S, zeta=zeta,
                              power_iters=cheb_power_iters,
                              safety=cheb_safety)
        cheb_solve = _make_cheb_schur_solve(
            diag_S, stil, lo, hi, cheb_degree)
        preconditioner_fn = _block_ldl_pinv(
            zeta, lambda t2: -cheb_solve(t2))
      else:
        preconditioner_fn = _block_ldl_pinv(
            zeta, lambda t2: -t2 / diag_S)
    else:
      preconditioner_fn = jacobi_pinv

    return (apply_A, preconditioner_fn, nf_state,
            rnf_FU, rnf_FE, rnf_SU, rnf_SE)

  # -- Numeric body (shared by the eager IC(0) path and the jitted device path).
  # ``pc``/``zero_nf``/``ret_s``/``ret_r``/``tol_``/``atol_`` are static; for the
  # on-device preconditioners (``'diag'``/``'jacobi'``) ``ic0_obj`` is None and
  # the whole body is jittable, so a single XLA program covers the grand matvec
  # (with wave FFTs), the near-field neighbor applies, and GMRES.
  def _body_impl(state, positions_frac, fp6, e5, E_inf_mat, L_inf_mat,
                 current_box, slip_top, extra, x0,
                 pc, zero_nf, ret_s, ret_r, tol_, atol_, ic0_obj,
                 restart_, maxiter_, solve_method_, prepared_nf=None):
    # ``prepared_nf`` is a TRACED trailing arg (a PreparedNearField pytree or
    # None), deliberately after the static block so _STATIC_ARGNUMS is stable.
    N = positions_frac.shape[0]
    dtype = REAL_DTYPE
    (apply_A, M_op, _nf_state, _rnf_FU, rnf_FE, rnf_SU,
     rnf_SE) = _build_fixed_system(
         state, positions_frac, current_box, pc, zero_nf,
         ic0_obj=ic0_obj, prepared_nf=prepared_nf)
    b1, b2 = _saddle_rhs(fp6, e5, rnf_FE, slip_top=slip_top, extra=extra)

    x, conv_info = sparse_linalg.gmres(
        apply_A, (b1, b2), x0=x0, tol=tol_, atol=atol_,
        restart=int(restart_), maxiter=int(maxiter_), M=M_op,
        solve_method=solve_method_)

    q11, u6 = x
    U_rel, Omega_rel = u6[..., :3], u6[..., 3:]

    # Total stresslet (Eq. 2.9): S = S^ff - R^nf_SU (U-U^inf) + R^nf_SE : E^inf.
    # Skipped when ``ret_s`` is False (the SU/SE near-field applies are not
    # free) -- the original FSD code likewise computes it only on output.
    if ret_s:
      sff5 = stresslet_from_moment(q11)
      S5 = sff5 - rnf_SU(u6) + rnf_SE(e5)
    else:
      S5 = jnp.zeros((N, 5), dtype=dtype)

    # Background-flow add-back (convenience; relative frame is the pinned one);
    # see :func:`_ambient_addback` for the vorticity/curl conventions.
    box = state.rpy.real.box_matrix if current_box is None else current_box
    U_inf, Omega_inf = _ambient_addback(L_inf_mat, positions_frac, box)

    info = {'gmres_info': conv_info, 'U_inf': U_inf, 'Omega_inf': Omega_inf}
    # Residual diagnostic, on-device (jnp, no host float) so the body stays
    # jittable; costs one extra grand+near-field matvec -> gated by ``ret_r``.
    if ret_r:
      Ax = apply_A(x)
      resid = (Ax[0] - b1, Ax[1] - b2)
      res = jnp.sqrt(_pytree_dot(resid, resid))
      bnorm = jnp.sqrt(_pytree_dot((b1, b2), (b1, b2)))
      rel = res / jnp.maximum(bnorm, 1e-300)
      info['rel_residual'] = rel
      # Convergence verdict on the TRUE residual -- the criterion the original
      # FSD code monitors, NOT jax's internal preconditioned-residual test,
      # which stops on ||M(b - Ax)|| and so can declare success while the true
      # residual is O(1) on ill-conditioned (contact-degenerate) systems.  This
      # is the only convergence signal the caller gets: jax's ``gmres_info`` is
      # 0 unless the solution went NaN.
      converged = res <= jnp.maximum(tol_ * bnorm, atol_)
      info['converged'] = converged
      _warn_nonconvergence(converged, rel, tol_, atol_)
    return U_rel, Omega_rel, S5, q11, info

  # Cache jitted device-path bodies keyed by their static configuration so the
  # XLA program is compiled once per (pc, flags, tol, gmres budget) combination,
  # not per call.
  _STATIC_ARGNUMS = tuple(range(10, 20))  # pc..maxiter_, solve_method_
  _body_jit_cache = {}

  def _device_body(state, positions_frac, fp6, e5, E_inf_mat, L_inf_mat,
                   current_box, slip_top, extra,
                   x0, pc, zero_nf, ret_s, ret_r, tol_, atol_,
                   restart_, maxiter_, solve_method_, prepared_nf=None):
    key = (pc, zero_nf, ret_s, ret_r, tol_, atol_, restart_, maxiter_,
           solve_method_)
    fn = _body_jit_cache.get(key)
    if fn is None:
      fn = jax.jit(_body_impl, static_argnums=_STATIC_ARGNUMS)
      _body_jit_cache[key] = fn
    return fn(state, positions_frac, fp6, e5, E_inf_mat, L_inf_mat, current_box,
              slip_top, extra, x0,
              pc, zero_nf, ret_s, ret_r, tol_, atol_, None, restart_, maxiter_,
              solve_method_, prepared_nf)

  # -- solve_fn ------------------------------------------------------------
  def solve_fn(
      state: SaddleState,
      positions_frac: jnp.ndarray,
      force: Optional[jnp.ndarray] = None,
      torque: Optional[jnp.ndarray] = None,
      E_inf: Optional[jnp.ndarray] = None,
      *,
      L_inf: Optional[jnp.ndarray] = None,
      x0=None,
      zero_nearfield: bool = False,
      preconditioner: Optional[str] = None,
      slip_top: Optional[jnp.ndarray] = None,
      extra_force: Optional[jnp.ndarray] = None,
      prepared_nf: Optional[PreparedNearField] = None,
      ic0=None,
      tol: Optional[float] = None,
      atol: Optional[float] = None,
      gmres_restart: Optional[int] = None,
      gmres_maxiter: Optional[int] = None,
      gmres_solve_method: Optional[str] = None,
      return_stresslet: bool = True,
      return_residual: bool = True,
      **shear_kwargs,
  ):
    """Solve the saddle system at a fixed configuration.

    Args:
      state: from ``init_fn`` (same configuration).
      positions_frac: fractional coordinates (must match ``state.positions``).
      force: applied force ``F^P`` ``(N,3)`` (default 0).
      torque: applied torque ``(N,3)`` (default 0).
      E_inf: imposed rate-of-strain, either a single symmetric traceless
        ``(3,3)`` (broadcast to all particles) or orthonormal ``(N,5)``
        (default 0).  Enters the RHS (resistance problem).
      L_inf: optional full ambient velocity gradient ``(3,3)`` (``u^inf=L.r``)
        used only for the background-flow add-back ``info['U_inf']`` /
        ``info['Omega_inf']``.  Defaults to the symmetric part of ``E_inf`` (no
        ambient vorticity), recovering the Phase-2 behavior bit-for-bit.  Pass
        the full gradient (e.g. simple shear ``L[0,1]=gamma_dot``) so the
        add-back carries the ambient spin.
      shear_kwargs: ``gamma_xy``/``gamma_xz``/``gamma_yz`` (or ``shear=``) for a
        live (Lees-Edwards) deformed box; ignored when the space has no
        ``box_fn`` (static box).
      x0: optional warm-start ``(moment_flat11, u_rel6)`` pytree.
      zero_nearfield: if True, force ``R^nf = 0`` (degenerate check 1).
      preconditioner: override the builder default (``'cheb'``, ``'diag'``,
        ``'ic0'`` or ``'jacobi'``).  ``'ic0'`` is built host-side from ``state``
        and needs concrete positions; the on-device ones run under
        ``jit``/``vmap``.
      slip_top: optional far-field Brownian slip ``U^B`` ``(N,11)`` added to the
        RHS top block (Phase 3); default 0.  Covariance ``(2kT/dt) M_grand``.
      extra_force: optional generalized force ``(N,6)`` added inside the bottom
        block (Phase 3): the near-field Brownian force ``F^B_nf`` for the main
        solve, or the RFD displacement ``Delta q`` for the drift solves.
      prepared_nf: optional near-field blocks already built by
        ``solve_fn.nf_apply.prepare(state.nf, positions_frac)`` -- MUST be at
        the same positions and box this solve uses (under live shear, refresh
        the state first; ``build_sd_brownian_step`` does).  Lets the Brownian
        step share one prepare across its sampler / solve / drift-stresslet
        consumers.  Ignored when ``zero_nearfield=True``; default ``None``
        prepares internally (the previous behavior).
      ic0: optional prebuilt :class:`Ic0Preconditioner` used in place of a
        host rebuild from ``state`` -- lets the RFD displaced solves reuse the
        factor built at ``q`` (only affects convergence, never the solution).
      tol: GMRES relative tolerance override (default builder ``gmres_tol``).
      atol: GMRES absolute tolerance override (default 0.0).  The RFD displaced
        solves set a fixed ``atol`` (with ``tol=0``) so both converge to the
        same *absolute* residual regardless of warm-start -- the drift divides
        ``U_+ - U_-`` by a tiny ``eps`` and amplifies any residual asymmetry.
      gmres_solve_method: override the builder's ``'batched'`` /
        ``'incremental'`` choice for this call.  Pass ``'incremental'`` on
        warm-started solves -- see ``build_saddle_solve``.
      return_residual: compute the true relative residual (one extra saddle
        matvec).  Also gates ``info['converged']`` and the non-convergence
        warning; the RFD displaced solves run to a fixed truncated budget by
        design and pass ``False``.

    Returns:
      ``(U_rel, Omega_rel, S5, F_moments, info)`` -- relative velocities
      ``(N,3)``, angular velocities ``(N,3)``, total stresslet ``(N,5)``
      orthonormal, far-field moments ``(N,11)``, and an info dict.  Work in the
      relative frame; ``info['U_inf']`` carries the background-flow add-back.
      With ``return_residual=True`` the info dict also carries
      ``rel_residual`` (true ``||b - Ax|| / ||b||``) and ``converged`` (that
      residual against ``max(tol ||b||, atol)``, the criterion the original
      FSD code monitors).  Do NOT read ``info['gmres_info']`` as a convergence
      flag -- jax sets it to 0 unless the solution is NaN.
    """
    # All optional array inputs -> concrete REAL_DTYPE arrays (zeros = absent).
    (positions_frac, fp6, e5, E_inf_mat, L_inf_mat,
     slip_arr, extra, x0) = _normalize_solve_inputs(
        positions_frac, force, torque, E_inf, L_inf, slip_top, extra_force, x0)

    # Live deformed box from the shear kwargs (None for a static box).
    current_box = _resolve_current_box(positions_frac, **shear_kwargs)

    pc = preconditioner if preconditioner is not None else default_preconditioner
    if pc not in ('cheb', 'diag', 'ic0', 'jacobi'):
      raise ValueError(
          "preconditioner must be 'cheb', 'diag', 'ic0', or 'jacobi', got %r"
          % (pc,))
    _tol = gmres_tol if tol is None else float(tol)
    _atol = 0.0 if atol is None else float(atol)
    _restart = (_default_gmres_restart if gmres_restart is None
                else int(gmres_restart))
    _maxiter = (_default_gmres_maxiter if gmres_maxiter is None
                else int(gmres_maxiter))
    _solve_method = (default_gmres_solve_method if gmres_solve_method is None
                     else gmres_solve_method)
    if _solve_method not in ('batched', 'incremental'):
      raise ValueError(
          "gmres_solve_method must be 'batched' or 'incremental', got %r"
          % (_solve_method,))

    # IC(0) stays eager (host RCM + scipy factor + pure_callback triangular
    # solves); the on-device preconditioners run the cached jitted body.
    if pc == 'ic0' and not zero_nearfield:
      ic0_obj = (ic0 if ic0 is not None
                 else build_ic0_from_state(state, a, eta, r_p=r_p, zeta=zeta))
      return _body_impl(
          state, positions_frac, fp6, e5, E_inf_mat, L_inf_mat, current_box,
          slip_arr, extra, x0,
          pc, zero_nearfield, return_stresslet, return_residual, _tol, _atol,
          ic0_obj, _restart, _maxiter, _solve_method, prepared_nf)
    return _device_body(
        state, positions_frac, fp6, e5, E_inf_mat, L_inf_mat, current_box,
        slip_arr, extra, x0,
        pc, zero_nearfield, return_stresslet, return_residual, _tol, _atol,
        _restart, _maxiter, _solve_method, prepared_nf)

  def mobility_tangent(
      state: SaddleState,
      positions: jnp.ndarray,
      direction6: jnp.ndarray,
      *,
      preconditioner: Optional[str] = None,
      prepared_nf: Optional[PreparedNearField] = None,
      tol: Optional[float] = None,
      atol: Optional[float] = None,
      gmres_restart: Optional[int] = None,
      gmres_maxiter: Optional[int] = None,
      gmres_solve_method: Optional[str] = None,
      return_stresslet: bool = True,
      return_residual: bool = False,
      **shear_kwargs,
  ):
    """Directional derivative of ``R_FU(q)^-1 direction6``.

    This is the implicit-function counterpart of a centred mobility RFD.  For
    ``A(q) x(q) = (0, -direction6)``, it solves the response problem once,
    forms ``dA = D_q[A(q) x][direction6[..., :3]]`` with ``x`` held fixed,
    and solves ``A x_dot = -dA`` with the same operator and preconditioner.

    Position tangents in JAX-MD's real/near-field displacement functions are
    physical vectors even when stored positions are fractional.  The spectral
    Ewald stencil consumes fractional coordinates directly, so only its JVP
    receives ``box^-1 direction``.  The box and neighbor-list topology remain
    fixed during the derivative.

    Returns ``(u6_dot, s5_dot, info)``.  ``s5_dot`` is the complete stresslet
    derivative, including the explicit position derivative of ``R^nf_SU``.
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    direction6 = jnp.asarray(direction6, dtype=REAL_DTYPE)
    N = positions.shape[0]
    if direction6.shape != (N, 6):
      raise ValueError(
          'direction6 must have shape (N, 6), got %r' %
          (direction6.shape,))

    pc = preconditioner if preconditioner is not None else default_preconditioner
    if pc not in ('cheb', 'diag', 'jacobi'):
      raise ValueError(
          "implicit mobility tangent requires an on-device preconditioner "
          "('cheb', 'diag', or 'jacobi'), got %r" % (pc,))
    _tol = gmres_tol if tol is None else float(tol)
    _atol = 0.0 if atol is None else float(atol)
    _restart = (_default_gmres_restart if gmres_restart is None
                else int(gmres_restart))
    _maxiter = (_default_gmres_maxiter if gmres_maxiter is None
                else int(gmres_maxiter))
    _solve_method = (default_gmres_solve_method if gmres_solve_method is None
                     else gmres_solve_method)
    if _solve_method not in ('batched', 'incremental'):
      raise ValueError(
          "gmres_solve_method must be 'batched' or 'incremental', got %r"
          % (_solve_method,))

    current_box = _resolve_current_box(positions, **shear_kwargs)
    (apply_A, M_op, nf_state, _rnf_FU, _rnf_FE, rnf_SU,
     _rnf_SE) = _build_fixed_system(
         state, positions, current_box, pc, False,
         prepared_nf=prepared_nf)

    zeros11 = jnp.zeros((N, 11), dtype=REAL_DTYPE)
    zeros6 = jnp.zeros((N, 6), dtype=REAL_DTYPE)
    zero_x = (zeros11, zeros6)
    rhs = (zeros11, -direction6)

    def solve_rhs(linear_rhs):
      return sparse_linalg.gmres(
          apply_A, linear_rhs, x0=zero_x, tol=_tol, atol=_atol,
          restart=_restart, maxiter=_maxiter, M=M_op,
          solve_method=_solve_method)

    response, response_info = solve_rhs(rhs)
    q11, u6 = response
    direction_physical = direction6[..., :3]
    box = state.rpy.real.box_matrix if current_box is None else current_box
    if fractional_coordinates:
      direction_wave = jnp.einsum(
          'ij,...j->...i', jnp.linalg.inv(box), direction_physical)
    else:
      direction_wave = direction_physical

    def real_part(pos):
      top = _make_real_grand_mv(state.rpy, pos, current_box)(q11)
      return top, zeros6

    def wave_part(pos):
      top = _make_wave_grand_mv(
          state.rpy, pos, current_box,
          wave_static=wave_static, a=a, xi=xi, eta=eta)(q11)
      return top, zeros6

    def near_part(pos):
      rnf_FU, _fe, _su, _se, _diag = make_rnf(
          nf_state, pos, False)
      return zeros11, -rnf_FU(u6)

    _, real_dot = jax.jvp(
        real_part, (positions,), (direction_physical,))
    _, wave_dot = jax.jvp(
        wave_part, (positions,), (direction_wave,))
    _, near_dot = jax.jvp(
        near_part, (positions,), (direction_physical,))
    operator_dot = jax.tree.map(
        lambda r, w, n: r + w + n, real_dot, wave_dot, near_dot)
    tangent_rhs = jax.tree.map(lambda value: -value, operator_dot)
    tangent, tangent_info = solve_rhs(tangent_rhs)
    q11_dot, u6_dot = tangent

    if return_stresslet:
      stresslet_dot = stresslet_from_moment(q11_dot) - rnf_SU(u6_dot)

      def explicit_stresslet(pos):
        _fu, _fe, su, _se, _diag = make_rnf(nf_state, pos, False)
        return -su(u6)

      _, explicit_dot = jax.jvp(
          explicit_stresslet, (positions,), (direction_physical,))
      stresslet_dot = stresslet_dot + explicit_dot
    else:
      stresslet_dot = jnp.zeros((N, 5), dtype=REAL_DTYPE)

    info = {
        'response_gmres_info': response_info,
        'tangent_gmres_info': tangent_info,
    }
    if return_residual:
      def residual_verdict(solution, linear_rhs):
        """``(rel_residual, converged)`` on the TRUE residual.

        Same criterion as the deterministic body: jax's ``gmres_info`` is 0
        unless the solution went NaN, and its internal test is on the
        *preconditioned* residual, so neither is a convergence signal.
        """
        residual = jax.tree.map(
            lambda actual, target: actual - target,
            apply_A(solution), linear_rhs)
        norm = jnp.sqrt(_pytree_dot(residual, residual))
        rhs_norm = jnp.sqrt(_pytree_dot(linear_rhs, linear_rhs))
        rel = norm / jnp.maximum(rhs_norm, 1e-300)
        return rel, norm <= jnp.maximum(_tol * rhs_norm, _atol)

      response_rel, response_ok = residual_verdict(response, rhs)
      tangent_rel, tangent_ok = residual_verdict(tangent, tangent_rhs)
      info['response_rel_residual'] = response_rel
      info['tangent_rel_residual'] = tangent_rel
      info['converged'] = jnp.logical_and(response_ok, tangent_ok)
      # Unlike the RFD displaced solves (truncated by design), these two run at
      # the full budget, so a failure here is a real one -- warn as the
      # deterministic solve does.
      _warn_nonconvergence(response_ok, response_rel, _tol, _atol)
      _warn_nonconvergence(tangent_ok, tangent_rel, _tol, _atol)
    return u6_dot, stresslet_dot, info

  # -- Eager iteration-count harness (check 3; runs outside JIT) -----------
  def count_iterations(
      state: SaddleState,
      positions_frac: jnp.ndarray,
      *,
      force=None,
      torque=None,
      E_inf=None,
      preconditioner: str = 'ic0',
      rtol: float = 1e-6,
      restart: Optional[int] = None,
  ):
    """Count GMRES iterations for the saddle solve under a given preconditioner.

    Runs an eager (non-JIT) ``scipy.sparse.linalg.gmres`` so the IC(0) factor and
    its triangular solves are validated as a *property of the preconditioned
    operator* (Fiore & Swan Fig. 1) without forcing a host callback inside JIT.

    ``preconditioner`` in ``{'none', 'jacobi', 'cheb', 'ic0'}``.  Returns
    ``(n_iter, rel_residual, diag)``.
    """
    import scipy.sparse.linalg as spla

    # Same input normalization as solve_fn (slip/extra/x0 unused here).
    (positions_frac, fp6, e5, _E_inf_mat, _L_inf_mat,
     _slip, _extra, _x0) = _normalize_solve_inputs(
        positions_frac, force, torque, E_inf, None, None, None, None)
    N = positions_frac.shape[0]

    grand_mv_flat = make_grand_mv(state.rpy, positions_frac)
    rnf_FU, rnf_FE, _su, _se, diag_FU = make_rnf(
        state.nf, positions_frac, False)

    nm, nf6 = 11 * N, 6 * N

    def split(v):
      return (jnp.asarray(v[:nm].reshape(N, 11), REAL_DTYPE),
              jnp.asarray(v[nm:].reshape(N, 6), REAL_DTYPE))

    def join(q11, u6):
      return np.concatenate([np.asarray(q11).ravel(), np.asarray(u6).ravel()])

    # The SAME saddle matvec/RHS the jitted body uses (shared helpers),
    # wrapped flat for scipy's LinearOperator.
    apply_A = _saddle_operator(grand_mv_flat, rnf_FU)

    def matvec(v):
      top, bot = apply_A(split(v))
      return join(top, bot)

    A_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=matvec)

    b1, b2 = _saddle_rhs(fp6, e5, rnf_FE)
    b = join(b1, b2)

    M_op = None
    diag = {'relaxed': 0.0, 'preconditioner': preconditioner}
    if preconditioner == 'jacobi':
      def mjac(v):
        z1, z2 = jacobi_pinv(split(v))
        return join(z1, z2)
      M_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=mjac)
    elif preconditioner == 'cheb':
      # Same on-device Chebyshev Schur operator the jitted solve uses, applied
      # eagerly to a flat scipy vector.
      diag_S = zeta + diag_FU()
      def stil(vv):
        return zeta * vv + rnf_FU(vv)
      lo, hi = _cheb_bounds(stil, diag_S, zeta=zeta,
                            power_iters=cheb_power_iters, safety=cheb_safety)
      cheb_solve = _make_cheb_schur_solve(diag_S, stil, lo, hi, cheb_degree)
      cheb_pinv = _block_ldl_pinv(zeta, lambda t2: -cheb_solve(t2))
      def mcheb(v):
        z1, z2 = cheb_pinv(split(v))
        return join(z1, z2)
      M_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=mcheb)
    elif preconditioner == 'ic0':
      ic0 = build_ic0_from_state(state, a, eta, r_p=r_p, zeta=zeta)
      diag['relaxed'] = ic0.relaxed
      def mic0(v):
        q11, u6 = split(v)
        # ic0.solve applies S~^-1 with S~ = zeta I + R~^nf_FU (positive);
        # the true Schur is S = -S~, so z2 = S^-1 t2 = -ic0.solve(t2).
        t2 = np.asarray(u6 - zeta * bt_apply(q11)).ravel()   # y2 - Bᵀ M^-1 y1
        z2 = jnp.asarray(-ic0.solve(t2).reshape(N, 6), REAL_DTYPE)
        z1 = zeta * q11                                       # M^-1 y1
        return join(z1 - zeta * b_apply(z2), z2)
      M_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=mic0)

    count = [0]
    def cb(_pr):
      count[0] += 1
    x, _flag = spla.gmres(
        A_op, b, M=M_op, rtol=rtol, atol=0.0,
        restart=int(restart or gmres_restart), maxiter=200,
        callback=cb, callback_type='pr_norm')
    rel = float(np.linalg.norm(A_op.matvec(x) - b) / max(np.linalg.norm(b), 1e-300))
    return count[0], rel, diag

  solve_fn.count_iterations = count_iterations
  solve_fn.mobility_tangent = mobility_tangent
  solve_fn.refresh_state = refresh_state
  solve_fn.zeta = zeta
  solve_fn.r_p = r_p
  # Exposed for the Phase-3 Brownian builder (sd_brownian.py): the resolved
  # Ewald split, the near-field apply, and the physical scales.
  solve_fn.xi = xi
  solve_fn.nf_apply = nf_apply
  solve_fn.a = float(a)
  solve_fn.eta = float(eta)
  solve_fn.r_lub = r_lub_resolved
  # Live-box (shear) plumbing for the Phase-3 Brownian builder: the resolved
  # wave static factors feed the exact deformed-box wave noise
  # (``_sample_wave_grand_noise``) and ``resolve_current_box`` maps runtime
  # shear kwargs -> deformed box (None for a static box).
  solve_fn.wave_static = wave_static
  solve_fn.resolve_current_box = _resolve_current_box
  solve_fn.has_box_fn = has_box_fn
  solve_fn.fractional_coordinates = fractional_coordinates

  return init_fn, solve_fn


def _pytree_dot(a, b):
  return sum(jnp.vdot(ai, bi) for ai, bi in zip(a, b)).real


# ===========================================================================
# Stage-2 preconditioner: host-side RCM + IC(0) on the truncated Schur S~.
#
# These run on host (numpy/scipy) once per configuration -- the sparsity
# pattern, RCM ordering and incomplete Cholesky have no native JAX form.  The
# resulting factor is applied via two triangular solves.  Built for *correctness
# and the N-independent iteration-count claim* (Phase 2); not production-fast.
# ===========================================================================
class _ICholFailure(Exception):
  """Raised when IC(0) hits a non-positive pivot (triggers relaxation)."""


def _min_image_cart(cart, box):
  """All pairwise minimum-image displacements; ``out[i,j] = r_j - r_i`` (i->j)."""
  inv = np.linalg.inv(box)
  frac = cart @ inv.T
  d = frac[None, :, :] - frac[:, None, :]      # (N,N,3) fractional j - i
  d -= np.round(d)
  return d @ box.T                              # cartesian (N,N,3)


def assemble_stilde(cart, box, a, eta, r_p, zeta):
  """Truncated approximate Schur ``S~ = zeta I + R~^nf_FU`` as a scipy CSR.

  ``R~^nf_FU`` is the 6N x 6N near-field FU resistance restricted to pairs within
  ``r_p`` (default 2.1a, <=12 neighbours).  Built on host from the same per-pair
  operators the matrix-free near-field uses, so the truncation is consistent.
  """
  cart = np.asarray(cart, dtype=np.float64)
  box = np.asarray(box, dtype=np.float64)
  N = cart.shape[0]
  disp = _min_image_cart(cart, box)            # (N,N,3)
  r2 = np.sum(disp * disp, axis=-1)
  blocks = {}                                   # (i,j) -> 6x6
  diag = np.zeros((N, 6, 6))
  r_p2 = float(r_p) ** 2
  for i in range(N):
    js = np.where((r2[i] < r_p2) & (r2[i] > 1e-12))[0]
    if js.size == 0:
      continue
    rij = disp[i, js]                           # (k,3) i->j
    r = np.sqrt(r2[i, js])
    rhat = rij / r[:, None]
    scal = np.asarray(nf_table.interpolate_scalars(jnp.asarray(r), a))
    R_self, R_cross = _build_pair_operators(
        jnp.asarray(rhat), jnp.asarray(scal), a, eta)
    R_self = np.asarray(R_self)[:, :6, :6]
    R_cross = np.asarray(R_cross)[:, :6, :6]
    diag[i] += R_self.sum(axis=0)
    for k, j in enumerate(js):
      blocks[(i, int(j))] = blocks.get((i, int(j)), np.zeros((6, 6))) + R_cross[k]
  for i in range(N):
    blocks[(i, i)] = diag[i] + zeta * np.eye(6)

  rows, cols, vals = [], [], []
  for (i, j), blk in blocks.items():
    base_i, base_j = 6 * i, 6 * j
    for r_ in range(6):
      for c_ in range(6):
        v = blk[r_, c_]
        if v != 0.0:
          rows.append(base_i + r_)
          cols.append(base_j + c_)
          vals.append(v)
  S = sp.csr_matrix((vals, (rows, cols)), shape=(6 * N, 6 * N))
  return 0.5 * (S + S.T)                        # symmetrize tiny asymmetries


def nearfield_FU_diagonal(cart, box, a, eta, r_lub):
  """Per-DOF diagonal of the *full* ``R^nf_FU`` and the neighborless mask.

  Returns ``(diag6 (N,6), has_neighbor (N,) bool)``: the six diagonal entries
  of each particle's ``6x6`` FU self block, and whether the particle has any
  neighbor within ``r_lub``.  Used by the Phase-3 near-field Brownian sampler to
  build the diagonal scaling ``D`` (small/large/zero cases) and the neighborless
  projector ``Proj`` / conditioning shift ``Shift_nn``.

  Shares the cutoff membership (strict ``r2 < r_lub^2``) and the
  ``interpolate_scalars`` + ``_build_pair_operators`` path with the matrix-free
  matvec (``sd_nearfield._core``), so "my diagonal is exactly zero" agrees with
  "I do not contribute" bit-for-bit at the cutoff.  This is why ``Proj`` zeros
  exactly the rows that vanish in the operator -- no covariance error at ``r_lub``.
  """
  cart = np.asarray(cart, dtype=np.float64)
  box = np.asarray(box, dtype=np.float64)
  N = cart.shape[0]
  disp = _min_image_cart(cart, box)             # (N,N,3), out[i,j] = r_j - r_i
  r2 = np.sum(disp * disp, axis=-1)
  r_lub2 = float(r_lub) ** 2
  diag6 = np.zeros((N, 6))
  has_neighbor = np.zeros((N,), dtype=bool)
  for i in range(N):
    js = np.where((r2[i] < r_lub2) & (r2[i] > 1e-12))[0]
    if js.size == 0:
      continue
    has_neighbor[i] = True
    rij = disp[i, js]                            # (k,3) i->j
    r = np.sqrt(r2[i, js])
    rhat = rij / r[:, None]
    scal = np.asarray(nf_table.interpolate_scalars(jnp.asarray(r), a))
    R_self, _ = _build_pair_operators(
        jnp.asarray(rhat), jnp.asarray(scal), a, eta)
    self_block = np.asarray(R_self)[:, :6, :6].sum(axis=0)   # (6,6)
    diag6[i] = np.diag(self_block)
  return diag6, has_neighbor


def _ic0(A_csc, relax=0.0):
  """Incomplete Cholesky, zero fill-in, of SPD ``A`` (lower factor ``L``).

  ``L`` has the lower-triangular sparsity pattern of ``A``; ``relax`` is added
  to the diagonal first (adaptive relaxation guarantees existence).  Raises
  :class:`_ICholFailure` on a non-positive pivot.
  """
  A = sp.csc_matrix(A_csc)
  n = A.shape[0]
  L = sp.tril(A, format='csc').copy().astype(np.float64)
  if relax:
    L = sp.csc_matrix(L + relax * sp.eye(n, format='csc'))
  indptr, indices, data = L.indptr, L.indices, L.data
  pos_of = [dict() for _ in range(n)]
  for j in range(n):
    for p in range(indptr[j], indptr[j + 1]):
      pos_of[j][indices[p]] = p
  for k in range(n):
    dkk = data[pos_of[k][k]]
    if dkk <= 0.0:
      raise _ICholFailure()
    dkk = math.sqrt(dkk)
    data[pos_of[k][k]] = dkk
    for p in range(indptr[k], indptr[k + 1]):
      if indices[p] > k:
        data[p] /= dkk
    for p in range(indptr[k], indptr[k + 1]):
      i = indices[p]
      if i <= k:
        continue
      lik = data[p]
      crp = pos_of[i]                            # column j == i
      for p2 in range(indptr[k], indptr[k + 1]):
        m = indices[p2]
        if m < i:
          continue
        q = crp.get(m)
        if q is not None:
          data[q] -= lik * data[p2]
  return sp.csc_matrix((data, indices, indptr), shape=(n, n))


class Ic0Preconditioner:
  """RCM-reordered IC(0) factor of ``S~`` with two-triangular-solve apply.

  ``relaxed`` reports the diagonal relaxation that was needed (0.0 ideally);
  a nonzero value degrades the factor toward block-Jacobi (safe direction) and
  is a prime suspect if the iteration count grows with N.
  """

  def __init__(self, stilde_csr, zeta, *, max_relax_doublings=20):
    self.zeta = float(zeta)
    n = stilde_csr.shape[0]
    self.perm = reverse_cuthill_mckee(sp.csc_matrix(stilde_csr), symmetric_mode=True)
    P = stilde_csr.tocsr()[self.perm][:, self.perm]
    relax, L = 0.0, None
    trial = 0.0
    for _ in range(max_relax_doublings + 1):
      try:
        L = _ic0(P, relax=trial)
        relax = trial
        break
      except _ICholFailure:
        trial = self.zeta if trial == 0.0 else trial * 2.0
    if L is None:
      raise RuntimeError('IC(0) failed to factor even after relaxation.')
    self.relaxed = relax
    self.L = sp.csr_matrix(L)
    self.LT = sp.csr_matrix(L.T)
    self.inv_perm = np.argsort(self.perm)

  def solve(self, b):
    """Apply ``S~^{-1} ~ P^T L^{-T} L^{-1} P`` to a ``(6N,)`` vector."""
    b = np.asarray(b, dtype=np.float64)
    bp = b[self.perm]
    w = spsolve_triangular(self.L, bp, lower=True)
    u = spsolve_triangular(self.LT, w, lower=False)
    out = np.empty_like(b)
    out[self.perm] = u
    return out

  # -- Forward / inverse triangular applies in the RCM-permuted ordering -----
  # These expose the factor pieces the Phase-3 near-field Brownian square root
  # needs (sd_brownian.py): the Lanczos operator uses L^{-1}, L^{-T}; the
  # unwind uses the forward L.  All act on vectors already in *permuted*
  # ordering (caller permutes at the boundary via ``perm`` / ``inv_perm``).
  def apply_L(self, x):
    """``L @ x`` (forward, lower-triangular) in permuted ordering."""
    return self.L @ np.asarray(x, dtype=np.float64)

  def apply_L_inv(self, x):
    """``L^{-1} x`` (lower-triangular solve) in permuted ordering."""
    return spsolve_triangular(self.L, np.asarray(x, dtype=np.float64),
                              lower=True)

  def apply_LT_inv(self, x):
    """``L^{-T} x`` (upper-triangular solve) in permuted ordering."""
    return spsolve_triangular(self.LT, np.asarray(x, dtype=np.float64),
                              lower=False)


def build_ic0_from_state(state, a, eta, *, r_p, zeta):
  """Convenience: build :class:`Ic0Preconditioner` for a ``SaddleState``."""
  box = np.asarray(state.rpy.real.box_matrix, dtype=np.float64)
  cart = np.asarray(state.positions, dtype=np.float64) @ box.T
  stilde = assemble_stilde(cart, box, a, eta, r_p, zeta)
  return Ic0Preconditioner(stilde, zeta)
