"""Stresslet-constrained mobility: U = (M_UF - M_US M_ES^{-1} M_EF) F.

A rigid sphere cannot strain: its local rate of strain must vanish, E = 0.
The stresslet S is the force dipole the particle develops to enforce that,
and is *solved for* from the constraint equation (Fiore & Swan 2018, Eq. 36)

    M_ES S = -(M_EF F + M_EL L),

after which the constrained velocity is assembled as

    U = M_UF F + M_UL L + M_US S  =  R_FU^{-1} [F, L].

Everything here is matrix-free: each sub-block application is one call of
the grand matvec ``grand_mv(F, C) -> (U, D)`` with the irrelevant
input zeroed and the relevant output extracted via the orthonormal
symmetric/antisymmetric decomposition in ``rpy_moments``.  M_ES is
symmetric PSD in those coordinates and well conditioned (the strain-
stresslet coupling decays as r^-3), so GMRES converges in ~8-10 iterations
independent of N and volume fraction.  Do not reach for the Lanczos
square-root machinery here -- that is for the ill-conditioned M_UF noise
sampling only.

Cost per constrained application: ``n_iter + 2`` grand matvecs (one RHS
build, the solve, one final assembly).

Autodiff note: do not differentiate through the GMRES iterations;
``jax.scipy.sparse.linalg.gmres`` carries a custom VJP that solves the
(symmetric) adjoint system via the implicit-function theorem, which
triggers as long as the linear operator is passed as a pure callable.
"""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax.scipy.sparse import linalg as sparse_linalg

from jax_md.hydro.rpy_moments import (
    decompose_gradient,
    stresslet_to_couplet,
    torque_to_couplet,
)

# grand_mv(F (N,3), C (N,3,3) traceless) -> (U (N,3), D (N,3,3) traceless)
GrandMatvec = Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]


def m_es_operator(grand_mv: GrandMatvec) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """The strain <- stresslet sub-block as a matrix-free operator on (N, 5).

  Symmetric PSD in the orthonormal stresslet/strain coordinates (inherited
  from the PSD of the grand operator); this is the GMRES system matrix.
  """

  def m_es(S5: jnp.ndarray) -> jnp.ndarray:
    zero_forces = jnp.zeros(S5.shape[:-1] + (3,), dtype=S5.dtype)
    _, D = grand_mv(zero_forces, stresslet_to_couplet(S5))
    return decompose_gradient(D)[0]

  return m_es


def solve_stresslet(
    grand_mv: GrandMatvec,
    rhs: jnp.ndarray,
    *,
    x0: jnp.ndarray,
    solve_tol: float,
    solve_maxiter: int,
) -> jnp.ndarray:
  """Solve M_ES S5 = rhs matrix-free with (non-restarted) GMRES.

  ``solve_maxiter`` must be a static Python int; it sets the Krylov basis
  size (single GMRES cycle, no restart -- M_ES needs ~8-10 iterations).
  """
  S5, _ = sparse_linalg.gmres(
      m_es_operator(grand_mv),
      rhs,
      x0=x0,
      tol=solve_tol,
      atol=0.0,
      restart=int(solve_maxiter),
      maxiter=1,
      solve_method='batched',
  )
  return S5


def make_constrained_solver(
    grand_mv: GrandMatvec,
    *,
    with_torque: bool = False,
    solve_tol: float = 1e-3,
    solve_maxiter: int = 50,
    return_residual: bool = False,
):
  """Build the matrix-free stresslet-constrained mobility apply.

  Args:
    grand_mv: Fixed-state grand matvec ``(F, C) -> (U, D)``.
    with_torque: If True, accept applied torques and also return the
      angular velocity Omega.
    solve_tol: Relative GMRES tolerance on the constraint residual.
    solve_maxiter: Krylov basis size (static int).  If this limit is ever
      approached, check for a sign/indefiniteness error, not a tight tolerance.
    return_residual: If True, spend one extra grand matvec per call to
      report the post-solve relative residual in ``info`` (a supported
      diagnostic for monitoring solver convergence).

  Returns:
    ``solve_fn(forces, torques=None, stresslet_guess=None,
    applied_output=None) -> (U, S5, Omega, info)`` where ``S5`` is the
    converged stresslet in orthonormal (N, 5) coordinates (thread it back
    as ``stresslet_guess`` to warm-start the next step), ``Omega`` is the
    angular velocity (None unless ``with_torque``), and ``info`` is a dict
    (contains ``'solve_rel_residual'`` when ``return_residual``).
    ``applied_output`` optionally supplies a precomputed
    ``grand_mv(F, torque_to_couplet(L))`` result so the RHS grand call can
    be fused with a state refresh by the caller.
  """

  def solve_fn(
      forces: jnp.ndarray,
      torques: Optional[jnp.ndarray] = None,
      stresslet_guess: Optional[jnp.ndarray] = None,
      applied_output: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
  ):
    forces = jnp.asarray(forces)
    if torques is not None and not with_torque:
      raise ValueError('torques were supplied but the solver was built with '
                       'with_torque=False.')
    if with_torque and torques is not None:
      C_applied = torque_to_couplet(jnp.asarray(torques, dtype=forces.dtype))
    else:
      C_applied = jnp.zeros(forces.shape[:-1] + (3, 3), dtype=forces.dtype)

    # RHS build: one grand call yields -(M_EF F + M_EL L) after extracting
    # the symmetric-traceless strain channel.
    if applied_output is None:
      applied_output = grand_mv(forces, C_applied)
    U_applied, D_applied = applied_output
    rhs = -decompose_gradient(D_applied)[0]

    if stresslet_guess is None:
      stresslet_guess = jnp.zeros(forces.shape[:-1] + (5,), dtype=forces.dtype)
    S5 = solve_stresslet(
        grand_mv,
        rhs,
        x0=stresslet_guess,
        solve_tol=solve_tol,
        solve_maxiter=solve_maxiter,
    )

    # Final assembly: one grand call with the solved stresslet only; add the
    # already-computed applied-moment response (exact by linearity).
    zero_forces = jnp.zeros_like(forces)
    U_stress, D_stress = grand_mv(zero_forces, stresslet_to_couplet(S5))
    U = U_applied + U_stress

    Omega = None
    if with_torque:
      Omega = (decompose_gradient(D_applied)[1] +
               decompose_gradient(D_stress)[1])

    info = {}
    if return_residual:
      residual = m_es_operator(grand_mv)(S5) - rhs
      info['solve_rel_residual'] = (
          jnp.linalg.norm(residual) /
          jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny))
    return U, S5, Omega, info

  return solve_fn
