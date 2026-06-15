"""Stochastic sampling utilities for the real-space RPY mobility.

The current implementation is force-only: ``mr_matvec`` applies a 3Nx3N mobility
and the sampler draws/returns arrays of shape ``(N, 3)`` in real coordinates.
The Lanczos machinery itself is agnostic to the underlying physics, so stresslet
or higher-order modes would require an expanded ``mr_matvec`` plus matching
noise/return shapes, which are not wired up here.
"""

import jax
import jax.numpy as jnp
from jax import lax

from typing import Callable, Optional
from functools import partial

from jax_md import dataclasses
from jax_md.hydro.rpy_real_det import REAL_DTYPE, RealSpaceState, mr_matvec

# --- Preconditioner for Chow–Saad-style sampling ---

@dataclasses.dataclass
class Preconditioner:
    """Linear-operator preconditioner G for Chow-Saad-style sampling.

    We assume GT G \approx M^{-1} so that \bar{M} = G M G^T is better conditioned.
    Only operator applications are required; G and G^T need not be formed explicitly.
    """
    apply: Callable[[jnp.ndarray], jnp.ndarray] = dataclasses.field(metadata={'static': True})   # y = G x
    apply_T: Callable[[jnp.ndarray], jnp.ndarray] = dataclasses.field(metadata={'static': True}) # y = G^T x
    solve: Callable[[jnp.ndarray], jnp.ndarray] = dataclasses.field(metadata={'static': True})   # y solves G y = x
    solve_T: Callable[[jnp.ndarray], jnp.ndarray] = dataclasses.field(metadata={'static': True}) # y solves G^T y = x


def identity_preconditioner() -> Preconditioner:
    """G = I. Useful as a no-op default."""
    @jax.jit
    def _id(x):
        return x
    return Preconditioner(apply=_id, apply_T=_id, solve=_id, solve_T=_id)


def scalar_preconditioner(scale: float) -> Preconditioner:
    """G = s I with scalar s (broadcasted). Choose s ≈ 1/sqrt(diag(M))."""
    s = jnp.asarray(scale, dtype=REAL_DTYPE)

    @jax.jit
    def _apply(x):
        return s * x

    @jax.jit
    def _solve(x):
        return x / s

    return Preconditioner(apply=_apply, apply_T=_apply, solve=_solve, solve_T=_solve)


def diagonal_preconditioner(diag: jnp.ndarray) -> Preconditioner:
    """G = diag(d) with elementwise d (broadcasts to x's shape)."""
    d = jnp.asarray(diag, dtype=REAL_DTYPE)

    @jax.jit
    def _apply(x):
        return d * x

    @jax.jit
    def _solve(x):
        return x / d

    return Preconditioner(apply=_apply, apply_T=_apply, solve=_solve, solve_T=_solve)


def jacobi_from_self(self_coeff: float) -> Preconditioner:
    """Simple Jacobi G using only the self term: M ≈ self_coeff * I ⇒ G = 1/sqrt(self_coeff) * I."""
    s = 1.0 / jnp.sqrt(jnp.asarray(self_coeff, dtype=REAL_DTYPE))
    return scalar_preconditioner(float(s))


_IDENTITY_PRECONDITIONER = identity_preconditioner()

def lanczos_sqrt_mv(matvec: Callable[[object, jnp.ndarray], jnp.ndarray],
                    mv_params,
                    v: jnp.ndarray,
                    *,
                    iters: int = 10,
                    tol: float = 1e-3,
                    return_info: bool = False):
    """
    Approximate ``(M)^{1/2} v`` with a Lanczos projection (Chow & Saad, 2014, Eq. (2.6)).

    Parameters
    ----------
    matvec : callable
        Function with signature ``matvec(params, x)`` that returns ``M @ x``.
    mv_params : pytree or tuple
        Auxiliary data passed to ``matvec`` (e.g. operator state). ``None`` is allowed.
    v : array
        Right-hand side to which the square root is applied.
    iters : int, optional
        Maximum Lanczos iterations (dimension of Krylov space). Must be positive.
    tol : float, optional
        Relative tolerance used for convergence monitoring. Outside of JIT, a
        ``RuntimeError`` is raised if the final iterate fails to satisfy the
        tolerance (mirrors the stopping criterion suggested by Chow & Saad,
        *SIAM J. Sci. Comput.* 36(2), 2014). Inside JIT, the function returns
        the best-effort approximation; callers can request diagnostics via
        ``return_info`` to monitor convergence.
    return_info : bool, optional
        If ``True``, also return ``(rel_change, iters_used, converged)`` for
        convergence monitoring (useful inside JIT where exceptions are disallowed).

    Returns
    -------
    array or tuple
        If ``return_info`` is ``False`` (default), the approximation of
        ``sqrt(M) @ v`` with the same shape/dtype as ``v``. Otherwise returns
        ``(approx, rel_change, iters_used, converged)`` where ``converged`` is
        ``rel_change <= tol`` evaluated in device precision.
    """
    if iters <= 0:
        raise ValueError("iters must be positive.")

    vec = jnp.asarray(v, dtype=REAL_DTYPE)
    params = mv_params if mv_params is not None else ()

    def _vec_norm(x):
        return jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(x, x)), jnp.array(0.0, dtype=REAL_DTYPE)))

    breakdown_tol = jnp.asarray(1e-12, dtype=REAL_DTYPE)
    zero = jnp.zeros_like(vec)
    vec_norm = _vec_norm(vec)
    tol_arr = jnp.asarray(tol, dtype=REAL_DTYPE)
    e1 = jnp.zeros((iters,), dtype=REAL_DTYPE).at[0].set(1.0)
    diag_idx = jnp.arange(iters, dtype=jnp.int32)
    off_idx = jnp.arange(max(iters - 1, 0), dtype=jnp.int32)

    def _sqrt_coeff(alpha_arr, beta_arr, active_dim):
        """Compute c = sqrt(T_m) e1 for the active Krylov subspace."""
        diag_mask = (diag_idx < active_dim).astype(REAL_DTYPE)
        diag = alpha_arr * diag_mask

        off_arr = beta_arr[:-1]
        off_mask = (off_idx < jnp.maximum(active_dim - 1, 0)).astype(REAL_DTYPE)
        off = off_arr * off_mask

        T = jnp.diag(diag)
        if off.shape[0]:
            T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
        eigvals, eigvecs = jnp.linalg.eigh(T)
        eigvals = jnp.clip(eigvals, min=0.0)
        sqrt_eigs = jnp.sqrt(eigvals)
        return eigvecs @ (sqrt_eigs * (eigvecs.T @ e1))

    def _run_lanczos(beta0):
        q_curr = vec / beta0
        q_prev = jnp.zeros_like(q_curr)
        beta_prev = jnp.array(0.0, dtype=REAL_DTYPE)
        alphas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        betas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        basis = jnp.zeros((iters,) + vec.shape, dtype=REAL_DTYPE)
        rel_change0 = jnp.asarray(jnp.inf, dtype=REAL_DTYPE)
        it_count0 = jnp.array(0, dtype=jnp.int32)
        done0 = jnp.array(False)

        def body(k, state):
            q_prev_i, q_curr_i, beta_prev_i, alpha_arr_i, beta_arr_i, basis_i, rel_change_i, it_count_i, done_i = state

            def _skip(vals):
                return vals

            def _step(vals):
                q_prev_s, q_curr_s, beta_prev_s, alpha_arr_s, beta_arr_s, basis_s, _, it_count_s, _ = vals
                basis_s = basis_s.at[k].set(q_curr_s)

                w = matvec(params, q_curr_s)
                w = jnp.asarray(w, dtype=REAL_DTYPE)
                w = w - beta_prev_s * q_prev_s
                alpha = jnp.real(jnp.vdot(q_curr_s, w))
                w = w - alpha * q_curr_s
                beta = _vec_norm(w)

                nonzero = beta > breakdown_tol
                beta_safe = jnp.where(nonzero, beta, jnp.array(1.0, dtype=REAL_DTYPE))
                q_next = jnp.where(nonzero, w / beta_safe, jnp.zeros_like(q_curr_s))

                alpha_arr_s = alpha_arr_s.at[k].set(alpha)
                beta_arr_s = beta_arr_s.at[k].set(beta)
                it_count_new = it_count_s + 1

                def _compute_rel(_):
                    coeff_curr = _sqrt_coeff(alpha_arr_s, beta_arr_s, it_count_new)
                    prev_dim = jnp.maximum(jnp.array(1, dtype=jnp.int32), it_count_new - 1)
                    coeff_prev = _sqrt_coeff(alpha_arr_s, beta_arr_s, prev_dim)
                    diff = coeff_curr - coeff_prev
                    numer = jnp.sqrt(jnp.maximum(
                        jnp.real(jnp.vdot(diff, diff)),
                        jnp.array(0.0, dtype=REAL_DTYPE)))
                    denom = jnp.sqrt(jnp.maximum(
                        jnp.real(jnp.vdot(coeff_curr, coeff_curr)),
                        jnp.array(0.0, dtype=REAL_DTYPE)))
                    denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=REAL_DTYPE))
                    return numer / denom

                rel_change_new = lax.cond(
                    it_count_new > 1,
                    _compute_rel,
                    lambda _: jnp.asarray(jnp.inf, dtype=REAL_DTYPE),
                    operand=None,
                )
                converged = jnp.logical_and(it_count_new > 1, rel_change_new <= tol_arr)
                done_new = jnp.logical_or(jnp.logical_not(nonzero), converged)
                return (
                    q_curr_s,
                    q_next,
                    beta,
                    alpha_arr_s,
                    beta_arr_s,
                    basis_s,
                    rel_change_new,
                    it_count_new,
                    done_new,
                )

            return lax.cond(
                done_i,
                _skip,
                _step,
                (q_prev_i, q_curr_i, beta_prev_i, alpha_arr_i, beta_arr_i, basis_i, rel_change_i, it_count_i, done_i),
            )

        init_state = (q_prev, q_curr, beta_prev, alphas, betas, basis, rel_change0, it_count0, done0)
        q_prev_f, q_curr_f, beta_prev_f, alpha_f, beta_f, basis_f, rel_change_f, iters_used, done_f = lax.fori_loop(
            0, iters, body, init_state
        )
        del q_prev_f, q_curr_f, beta_prev_f, done_f

        active_dim = jnp.maximum(jnp.array(1, dtype=jnp.int32), iters_used)
        coeff = _sqrt_coeff(alpha_f, beta_f, active_dim)
        coeff_mask = coeff * (diag_idx < active_dim).astype(REAL_DTYPE)
        approx = beta0 * jnp.einsum('i,i...->...', coeff_mask, basis_f)
        return approx, rel_change_f, iters_used

    approx, rel_change, actual_iters = lax.cond(
        vec_norm <= 0.0,
        lambda _: (zero, jnp.array(0.0, dtype=REAL_DTYPE), jnp.array(0, dtype=jnp.int32)),
        lambda beta0: _run_lanczos(beta0),
        vec_norm,
    )

    converged = rel_change <= tol_arr

    def _pack(result):
        if return_info:
            return result, rel_change, actual_iters, converged
        return result

    # Outside of JIT, enforce tolerance with a hard error.
    try:
        rel_host = float(rel_change)
        tol_host = float(tol_arr)
    except TypeError:
        # Inside JIT / tracing context: return best-effort approximation along
        # with diagnostics if requested.
        return _pack(approx)

    if rel_host > tol_host:
        raise RuntimeError(
            f"lanczos_sqrt_mv failed to reach tol={tol_host:.3e} "
            f"(rel_change={rel_host:.3e}, iters={int(actual_iters)}, requested={iters})."
        )
    return _pack(approx)


@partial(jax.jit, static_argnames=('iters', 'tol', 'return_info'))
def sample_mr_sqrt_precond(key: jax.Array,
                           state: RealSpaceState,
                           positions: jnp.ndarray,
                           *,
                           precond: Optional[Preconditioner] = None,
                           iters: int = 10,
                           tol: float = 1e-3,
                           return_info: bool = False):
    """
    Sample ``M^(r)^{1/2} W`` using the preconditioned Lanczos scheme of
    Chow & Saad (2014, §2.2).

    The method forms the preconditioned operator ``\\bar{M} = G M^(r) G^T`` where
    ``G`` is the linear operator supplied by ``precond`` (``G^T G ≈ (M^(r))^{-1}``).
    After approximating ``\\sqrt{\\bar{M}}`` via Lanczos we map the sample back by
    applying ``G^{-1}``, which recovers a sample with covariance ``M^(r)``.

    Parameters
    ----------
    key : PRNGKey
        Random key for the Gaussian draw.
    state : RealSpaceState
        Current real-space state (after updating with ``Mr_apply``).
    positions : (N,3) array
        Positions corresponding to ``state`` (fractional if
        ``state.fractional_coordinates`` is True, otherwise real).
    precond : Preconditioner, optional
        Linear operator ``G`` used for preconditioning. Defaults to the identity.
    iters : int, optional
        Maximum number of Lanczos iterations (default 10).
    tol : float, optional
        Convergence tolerance (default 1e-3). Monitors the change in approximation
        between consecutive iterations and enforces it as described in
        ``lanczos_sqrt_mv``.
    return_info : bool, optional
        If True, also return convergence diagnostics from ``lanczos_sqrt_mv``.

    Returns
    -------
    (N,3) array
        Sample from the real-space stochastic increment (in real coordinates). If
        ``return_info`` is True, returns ``(sample, rel_change, iters_used, converged)``.
    """
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    precond = _IDENTITY_PRECONDITIONER if precond is None else precond

    noise = jax.random.normal(key, shape=positions.shape, dtype=REAL_DTYPE)

    def mv(mv_params, x):
        st, pos, pc = mv_params
        x_phys = pc.apply_T(x)
        m_x = mr_matvec(st, pos, x_phys)
        return pc.apply(m_x)

    mv_params = (state, positions, precond)
    lanczos_out = lanczos_sqrt_mv(
        mv, mv_params, noise, iters=iters, tol=tol, return_info=return_info)

    if return_info:
        precond_sample, rel_change, iters_used, converged = lanczos_out
        sample = precond.solve(precond_sample)
        return sample, rel_change, iters_used, converged

    precond_sample = lanczos_out
    return precond.solve(precond_sample)


@partial(jax.jit, static_argnames=('iters', 'tol', 'return_info'))
def sample_mr_sqrt(key: jax.Array,
                   state: RealSpaceState,
                   positions: jnp.ndarray,
                   *,
                   iters: int = 10,
                   tol: float = 1e-3,
                   return_info: bool = False):
    """
    Convenience wrapper for ``sample_mr_sqrt_precond`` using the identity preconditioner.
    """
    return sample_mr_sqrt_precond(
        key,
        state,
        positions,
        precond=_IDENTITY_PRECONDITIONER,
        iters=iters,
        tol=tol,
        return_info=return_info,
    )
