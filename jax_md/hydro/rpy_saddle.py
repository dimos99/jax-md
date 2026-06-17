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

Phase-2 scope: static box only.  The imposed rate-of-strain ``E^inf`` enters via
the RHS, so no Lees-Edwards box deformation is needed; the live-box/shear path
belongs to the (deferred) Phase-3 integrator.
"""

import math
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import spsolve_triangular
from jax.scipy.sparse import linalg as sparse_linalg

from jax_md import dataclasses
from jax_md import space

from jax_md.hydro.rpy import build_rpy_mobility, estimate_rpy_params, RpyState
from jax_md.hydro.rpy_real_det_dipole import mr_grand_matvec
from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE, current_box_matrix
from jax_md.hydro.rpy_moments import (
    couplet_to_orthonormal,
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
    gmres_maxiter: int = 4,
    r_lub: Optional[float] = None,
    r_p: Optional[float] = None,
    preconditioner: str = 'ic0',
    fractional_coordinates: bool = True,
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
    gmres_tol: relative GMRES tolerance on the saddle residual.
    gmres_restart, gmres_maxiter: Krylov basis size and number of restarts.
    r_lub: near-field cutoff (default ``4a``).
    r_p: near-field truncation for the IC(0) Schur factor (default ``2.1a``).
    preconditioner: default GMRES preconditioner, ``'ic0'`` (RCM + zero-fill
      incomplete Cholesky of the truncated Schur ``S~ = zeta I + R~^nf_FU``;
      ~3-5x fewer iterations and near-N-independent) or ``'jacobi'`` (the
      pure-JAX block-Jacobi ``S~ = zeta I``).  IC(0) is built host-side per
      configuration and applied via ``jax.pure_callback``; it requires concrete
      positions, so a ``jit``/``vmap``-wrapped ``solve_fn`` must pass
      ``preconditioner='jacobi'``.  Overridable per call on ``solve_fn``.
    **rpy_kwargs: forwarded to ``build_rpy_mobility`` (e.g. ``P``, ``Mgrid``,
      ``rcut``).

  Returns:
    ``(init_fn, solve_fn)``.
  """
  if preconditioner not in ('ic0', 'jacobi'):
    raise ValueError("preconditioner must be 'ic0' or 'jacobi', got %r"
                     % (preconditioner,))
  default_preconditioner = preconditioner
  if len(space_fns) > 2 and space_fns[2] is not None:
    # A box_fn is allowed in the tuple but live shear is not supported here.
    pass
  displacement_fn = space_fns[0]
  box_fn = space_fns[2] if len(space_fns) > 2 else None

  zeta = 6.0 * math.pi * float(eta) * float(a)
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
  rpy_init, _ = build_rpy_mobility(
      space_fns, a, xi, eta,
      use_stresslet=True,
      constrained=False,
      include_brownian=False,
      fractional_coordinates=fractional_coordinates,
      **grid_kwargs,
  )

  # -- Near-field resistance (Phase 1) -------------------------------------
  nf_init, nf_apply = build_nearfield_resistance(
      space_fns, a, eta,
      r_lub=r_lub,
      fractional_coordinates=fractional_coordinates,
  )

  # -- init_fn -------------------------------------------------------------
  def init_fn(positions_frac) -> SaddleState:
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    rpy_state = rpy_init(positions_frac)
    nf_state = nf_init(positions_frac)
    return SaddleState(rpy=rpy_state, nf=nf_state, positions=positions_frac)

  # -- Operator factories (fixed state) -----------------------------------
  def _make_grand_mv(rpy_state: RpyState, positions: jnp.ndarray):
    """Static-box far-field grand matvec on flat-11 (real + wave, fixed state)."""
    def grand_mv_flat(q11: jnp.ndarray) -> jnp.ndarray:
      F, C = flat_to_grand(q11)
      Ur, Dr = mr_grand_matvec(rpy_state.real, positions, F, C)
      Uw, Dw = rpy_state.wave.apply_fn(positions, F, C)
      return grand_to_flat(Ur + Uw, traceless(Dr + Dw))
    return grand_mv_flat

  def _make_rnf(nf_state: NearFieldState, positions: jnp.ndarray, zero_nf: bool):
    """Fixed-config near-field block applies; ``zero_nf`` forces R^nf=0."""
    def nf_full(gv11: jnp.ndarray) -> jnp.ndarray:
      if zero_nf:
        return jnp.zeros_like(gv11)
      gf, _ = nf_apply(nf_state, positions, gv11)
      return gf

    def rnf_FU(u6):
      return nf_full(_gv_from_u6(u6))[..., :6]

    def rnf_FE(e5):
      return nf_full(_gv_from_e5(e5))[..., :6]

    def rnf_SU(u6):
      return nf_full(_gv_from_u6(u6))[..., 6:11]

    def rnf_SE(e5):
      return nf_full(_gv_from_e5(e5))[..., 6:11]

    return rnf_FU, rnf_FE, rnf_SU, rnf_SE

  # -- Block-Jacobi preconditioner (Stage 1; Schur approx S~ = zeta I) ------
  # Exact block-LDL inverse of A = [[M, B], [Bᵀ, -R^nf_FU]] with M^-1 ~ zeta I
  # and (negative) Schur S = -(R^nf_FU + Bᵀ M^-1 B) ~ -zeta I:
  #   t2 = y2 - Bᵀ M^-1 y1         (note the zeta on Bᵀ y1)
  #   z2 = S^-1 t2 ~ -t2 / zeta
  #   z1 = M^-1 y1 - M^-1 B z2 = zeta y1 - zeta B z2
  # Dropping the zeta factors / the Schur sign (as a previous version did)
  # leaves sigma(P A) straddling zero and ~doubles the GMRES iteration count.
  def _apply_pinv(x):
    y1, y2 = x                       # y1 (N,11) moment, y2 (N,6) FU
    t2 = y2 - zeta * bt_apply(y1)    # y2 - Bᵀ M^-1 y1
    z2 = -t2 / zeta                  # S^-1 ~ -(1/zeta) I
    z1 = zeta * y1                   # M^-1 y1
    return (z1 - zeta * b_apply(z2), z2)

  # -- IC(0) preconditioner: identical block-LDL apply, but the Schur solve
  # uses the host-side RCM + incomplete-Cholesky factor of S~ = zeta I + R~^nf_FU
  # (positive definite), so z2 = S^-1 t2 = -S~^-1 t2 = -ic0.solve(t2).  The host
  # solve is bridged into the traced GMRES via jax.pure_callback.
  def _make_ic0_pinv(ic0):
    def _host_solve(v):
      return np.asarray(ic0.solve(np.asarray(v, dtype=np.float64)),
                        dtype=np.float64)

    def _apply_pinv_ic0(x):
      y1, y2 = x
      t2 = y2 - zeta * bt_apply(y1)
      flat = t2.reshape(-1)
      z2flat = jax.pure_callback(
          _host_solve, jax.ShapeDtypeStruct(flat.shape, REAL_DTYPE), flat)
      z2 = -z2flat.reshape(y2.shape)
      z1 = zeta * y1
      return (z1 - zeta * b_apply(z2), z2)

    return _apply_pinv_ic0

  # -- solve_fn ------------------------------------------------------------
  def solve_fn(
      state: SaddleState,
      positions_frac: jnp.ndarray,
      force: Optional[jnp.ndarray] = None,
      torque: Optional[jnp.ndarray] = None,
      E_inf: Optional[jnp.ndarray] = None,
      *,
      x0=None,
      zero_nearfield: bool = False,
      preconditioner: Optional[str] = None,
      slip_top: Optional[jnp.ndarray] = None,
      extra_force: Optional[jnp.ndarray] = None,
      ic0=None,
      tol: Optional[float] = None,
      atol: Optional[float] = None,
  ):
    """Solve the saddle system at a fixed configuration.

    Args:
      state: from ``init_fn`` (same configuration).
      positions_frac: fractional coordinates (must match ``state.positions``).
      force: applied force ``F^P`` ``(N,3)`` (default 0).
      torque: applied torque ``(N,3)`` (default 0).
      E_inf: imposed rate-of-strain, either a single symmetric traceless
        ``(3,3)`` (broadcast to all particles) or orthonormal ``(N,5)``
        (default 0).
      x0: optional warm-start ``(moment_flat11, u_rel6)`` pytree.
      zero_nearfield: if True, force ``R^nf = 0`` (degenerate check 1).
      preconditioner: override the builder default (``'ic0'`` or ``'jacobi'``).
        ``'ic0'`` is built host-side from ``state`` and needs concrete positions;
        use ``'jacobi'`` under ``jit``/``vmap``.
      slip_top: optional far-field Brownian slip ``U^B`` ``(N,11)`` added to the
        RHS top block (Phase 3); default 0.  Covariance ``(2kT/dt) M_grand``.
      extra_force: optional generalized force ``(N,6)`` added inside the bottom
        block (Phase 3): the near-field Brownian force ``F^B_nf`` for the main
        solve, or the RFD displacement ``Delta q`` for the drift solves.
      ic0: optional prebuilt :class:`Ic0Preconditioner` used in place of a
        host rebuild from ``state`` -- lets the RFD displaced solves reuse the
        factor built at ``q`` (only affects convergence, never the solution).
      tol: GMRES relative tolerance override (default builder ``gmres_tol``).
      atol: GMRES absolute tolerance override (default 0.0).  The RFD displaced
        solves set a fixed ``atol`` (with ``tol=0``) so both converge to the
        same *absolute* residual regardless of warm-start -- the drift divides
        ``U_+ - U_-`` by a tiny ``eps`` and amplifies any residual asymmetry.

    Returns:
      ``(U_rel, Omega_rel, S5, F_moments, info)`` -- relative velocities
      ``(N,3)``, angular velocities ``(N,3)``, total stresslet ``(N,5)``
      orthonormal, far-field moments ``(N,11)``, and an info dict.  Work in the
      relative frame; ``info['U_inf']`` carries the background-flow add-back.
    """
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    N = positions_frac.shape[0]
    dtype = REAL_DTYPE

    if force is None:
      force = jnp.zeros((N, 3), dtype=dtype)
    else:
      force = jnp.asarray(force, dtype=dtype)
    if torque is None:
      torque = jnp.zeros((N, 3), dtype=dtype)
    else:
      torque = jnp.asarray(torque, dtype=dtype)
    fp6 = jnp.concatenate([force, torque], axis=-1)

    # Imposed strain in orthonormal (N,5).
    if E_inf is None:
      e5 = jnp.zeros((N, 5), dtype=dtype)
      E_inf_mat = jnp.zeros((3, 3), dtype=dtype)
    else:
      E_inf = jnp.asarray(E_inf, dtype=dtype)
      if E_inf.shape[-2:] == (3, 3):
        E_inf_mat = traceless(0.5 * (E_inf + jnp.swapaxes(E_inf, -1, -2)))
        e5_single = decompose_gradient(E_inf_mat)[0]
        e5 = jnp.broadcast_to(e5_single, (N, 5))
      else:
        e5 = jnp.broadcast_to(E_inf, (N, 5))
        E_inf_mat = stresslet_to_couplet(e5[0])  # for U_inf add-back only

    grand_mv_flat = _make_grand_mv(state.rpy, positions_frac)
    rnf_FU, rnf_FE, rnf_SU, rnf_SE = _make_rnf(
        state.nf, positions_frac, zero_nearfield)

    # -- A x ---------------------------------------------------------------
    def apply_A(x):
      q11, u6 = x
      top = grand_mv_flat(q11) + b_apply(u6)
      bot = bt_apply(q11) - rnf_FU(u6)
      return (top, bot)

    # -- RHS ---------------------------------------------------------------
    # b1 = (0_rigid, E^inf) in the velocity-output flat-11 (strain slots),
    # plus the optional far-field Brownian slip U^B (Phase 3).
    b1 = grand_to_flat(
        jnp.zeros((N, 3), dtype=dtype), stresslet_to_couplet(e5))
    if slip_top is not None:
      b1 = b1 + jnp.asarray(slip_top, dtype=dtype)
    # b2 = -(F^P + extra_force + R^nf_FE : E^inf) in FU force space.  The
    # optional extra_force is the near-field Brownian force F^B_nf (main solve)
    # or the RFD displacement Delta q (drift solves).
    extra = (jnp.zeros((N, 6), dtype=dtype) if extra_force is None
             else jnp.asarray(extra_force, dtype=dtype))
    b2 = -(fp6 + extra + rnf_FE(e5))

    if x0 is None:
      x0 = (jnp.zeros((N, 11), dtype=dtype), jnp.zeros((N, 6), dtype=dtype))

    # -- Preconditioner selection -----------------------------------------
    pc = preconditioner if preconditioner is not None else default_preconditioner
    if pc not in ('ic0', 'jacobi'):
      raise ValueError("preconditioner must be 'ic0' or 'jacobi', got %r" % (pc,))
    # With R^nf = 0 the Schur is exactly zeta I, so IC(0) degenerates to
    # block-Jacobi -- skip the host factor in that case.  A prebuilt ic0
    # (e.g. reused across RFD displaced solves) takes precedence over a rebuild.
    if pc == 'ic0' and not zero_nearfield:
      ic0_obj = (ic0 if ic0 is not None
                 else build_ic0_from_state(state, a, eta, r_p=r_p, zeta=zeta))
      M_op = _make_ic0_pinv(ic0_obj)
    else:
      M_op = _apply_pinv

    _tol = gmres_tol if tol is None else float(tol)
    _atol = 0.0 if atol is None else float(atol)
    x, conv_info = sparse_linalg.gmres(
        apply_A, (b1, b2), x0=x0,
        tol=_tol, atol=_atol,
        restart=int(gmres_restart), maxiter=int(gmres_maxiter),
        M=M_op,
    )

    q11, u6 = x
    U_rel, Omega_rel = u6[..., :3], u6[..., 3:]

    # Total stresslet (Eq. 2.9): S = S^ff - R^nf_SU (U-U^inf) + R^nf_SE : E^inf.
    sff5 = stresslet_from_moment(q11)
    S5 = sff5 - rnf_SU(u6) + rnf_SE(e5)

    # Background-flow add-back (convenience; relative frame is the pinned one).
    # Origin at the box centre: U^inf = E . (x - x_centre).  The choice of
    # origin is the Lees-Edwards frame ambiguity (deferred to Phase 3); it
    # cancels in the origin-independent relative velocity U_i - U_j.
    box = state.rpy.real.box_matrix
    cart = space.transform(
        box, positions_frac - jnp.asarray(0.5, dtype=dtype))
    U_inf = jnp.einsum('ij,nj->ni', E_inf_mat, cart)

    # Residual diagnostic.
    Ax = apply_A(x)
    res = math.sqrt(
        float(_pytree_dot((Ax[0] - b1, Ax[1] - b2), (Ax[0] - b1, Ax[1] - b2))))
    bnorm = math.sqrt(float(_pytree_dot((b1, b2), (b1, b2))))
    info = {
        'rel_residual': res / max(bnorm, 1e-300),
        'gmres_info': conv_info,
        'U_inf': U_inf,
    }
    return U_rel, Omega_rel, S5, q11, info

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

    ``preconditioner`` in ``{'none', 'jacobi', 'ic0'}``.  Returns
    ``(n_iter, rel_residual, diag)``.
    """
    import scipy.sparse.linalg as spla

    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    N = positions_frac.shape[0]
    if force is None:
      force = jnp.zeros((N, 3), dtype=REAL_DTYPE)
    if torque is None:
      torque = jnp.zeros((N, 3), dtype=REAL_DTYPE)
    fp6 = jnp.concatenate([jnp.asarray(force, REAL_DTYPE),
                           jnp.asarray(torque, REAL_DTYPE)], axis=-1)
    if E_inf is None:
      e5 = jnp.zeros((N, 5), dtype=REAL_DTYPE)
    else:
      E_inf = jnp.asarray(E_inf, dtype=REAL_DTYPE)
      if E_inf.shape[-2:] == (3, 3):
        e5 = jnp.broadcast_to(
            decompose_gradient(traceless(0.5 * (E_inf + E_inf.T)))[0], (N, 5))
      else:
        e5 = jnp.broadcast_to(E_inf, (N, 5))

    grand_mv_flat = _make_grand_mv(state.rpy, positions_frac)
    rnf_FU, rnf_FE, _su, _se = _make_rnf(state.nf, positions_frac, False)

    nm, nf6 = 11 * N, 6 * N

    def split(v):
      return (jnp.asarray(v[:nm].reshape(N, 11), REAL_DTYPE),
              jnp.asarray(v[nm:].reshape(N, 6), REAL_DTYPE))

    def join(q11, u6):
      return np.concatenate([np.asarray(q11).ravel(), np.asarray(u6).ravel()])

    def matvec(v):
      q11, u6 = split(v)
      top = grand_mv_flat(q11) + b_apply(u6)
      bot = bt_apply(q11) - rnf_FU(u6)
      return join(top, bot)

    A_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=matvec)

    b1 = grand_to_flat(jnp.zeros((N, 3), REAL_DTYPE), stresslet_to_couplet(e5))
    b2 = -(fp6 + rnf_FE(e5))
    b = join(b1, b2)

    M_op = None
    diag = {'relaxed': 0.0, 'preconditioner': preconditioner}
    if preconditioner == 'jacobi':
      def mjac(v):
        q11, u6 = split(v)
        t2 = u6 - zeta * bt_apply(q11)   # y2 - Bᵀ M^-1 y1
        z2 = -t2 / zeta                  # S^-1 ~ -(1/zeta) I
        z1 = zeta * q11                  # M^-1 y1
        return join(z1 - zeta * b_apply(z2), z2)
      M_op = spla.LinearOperator((nm + nf6, nm + nf6), matvec=mjac)
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
  solve_fn.zeta = zeta
  solve_fn.r_p = r_p
  # Exposed for the Phase-3 Brownian builder (sd_brownian.py): the resolved
  # Ewald split, the near-field apply, and the physical scales.
  solve_fn.xi = xi
  solve_fn.nf_apply = nf_apply
  solve_fn.a = float(a)
  solve_fn.eta = float(eta)
  solve_fn.r_lub = r_lub_resolved

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
