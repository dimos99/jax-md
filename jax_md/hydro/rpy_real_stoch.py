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

    def _run_lanczos(beta0):
        q_curr = vec / beta0
        q_prev = jnp.zeros_like(q_curr)
        alphas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        betas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        beta_prev = jnp.array(0.0, dtype=REAL_DTYPE)
        it_count_init = jnp.array(0, dtype=jnp.int32)
        finished = jnp.array(False)

        def body(k, state):
            q_prev, q_curr, alpha_arr, beta_arr, beta_prev, it_count, done_flag = state

            def iterate(vals):
                q_prev_i, q_curr_i, alpha_arr_i, beta_arr_i, beta_prev_i, it_count_i, _ = vals
                w = matvec(params, q_curr_i)
                w = jnp.asarray(w, dtype=REAL_DTYPE)
                w = w - beta_prev_i * q_prev_i
                alpha = jnp.real(jnp.vdot(q_curr_i, w))
                w = w - alpha * q_curr_i
                beta = _vec_norm(w)
                nonzero = beta > breakdown_tol
                beta_safe = jnp.where(nonzero, beta, jnp.array(1.0, dtype=REAL_DTYPE))
                q_next = jnp.where(nonzero, w / beta_safe, jnp.zeros_like(q_curr_i))
                alpha_arr_i = alpha_arr_i.at[k].set(alpha)
                beta_arr_i = beta_arr_i.at[k].set(beta)
                new_done = jnp.logical_not(nonzero)
                return (
                    q_curr_i,
                    q_next,
                    alpha_arr_i,
                    beta_arr_i,
                    beta,
                    it_count_i + 1,
                    new_done,
                )

            updated = lax.cond(
                done_flag,
                lambda s: s,
                iterate,
                (q_prev, q_curr, alpha_arr, beta_arr, beta_prev, it_count, done_flag),
            )
            q_prev_new, q_curr_new, alpha_new, beta_new, beta_prev_new, it_count_new, just_finished = updated
            total_done = jnp.logical_or(done_flag, just_finished)
            return (
                q_prev_new,
                q_curr_new,
                alpha_new,
                beta_new,
                beta_prev_new,
                it_count_new,
                total_done,
            )

        init_state = (
            q_prev,
            q_curr,
            alphas,
            betas,
            beta_prev,
            it_count_init,
            finished,
        )
        _, _, alpha_f, beta_f, _, iters_used, _ = lax.fori_loop(0, iters, body, init_state)
        active_dim = jnp.maximum(jnp.array(1, dtype=jnp.int32), iters_used)

        diag_idx = jnp.arange(iters, dtype=jnp.int32)
        diag_mask_curr = (diag_idx < active_dim).astype(REAL_DTYPE)
        diag_curr = alpha_f * diag_mask_curr

        off_arr = beta_f[:-1]
        off_idx = jnp.arange(max(iters - 1, 0), dtype=jnp.int32)
        off_mask_curr = (off_idx < jnp.maximum(active_dim - 1, 0)).astype(REAL_DTYPE)
        off_curr = off_arr * off_mask_curr

        prev_dim = jnp.maximum(jnp.array(1, dtype=jnp.int32), active_dim - 1)
        diag_mask_prev = (diag_idx < prev_dim).astype(REAL_DTYPE)
        diag_prev = alpha_f * diag_mask_prev
        off_mask_prev = (off_idx < jnp.maximum(prev_dim - 1, 0)).astype(REAL_DTYPE)
        off_prev = off_arr * off_mask_prev

        e1 = jnp.zeros((iters,), dtype=REAL_DTYPE).at[0].set(1.0)

        def _sqrt_T_e1(diag, off):
            T = jnp.diag(diag)
            if off.shape[0]:
                T = T + jnp.diag(off, 1) + jnp.diag(off, -1)
            eigvals, eigvecs = jnp.linalg.eigh(T)
            eigvals = jnp.clip(eigvals, a_min=0.0)
            sqrt_eigs = jnp.sqrt(eigvals)
            return eigvecs @ (sqrt_eigs * (eigvecs.T @ e1))

        coeff_curr = _sqrt_T_e1(diag_curr, off_curr)

        coeff_prev = lax.cond(
            active_dim > 1,
            lambda _: _sqrt_T_e1(diag_prev, off_prev),
            lambda _: coeff_curr,
            operand=None,
        )

        # Estimate relative change purely in coefficient space using orthonormality of V_m:
        #   ||y_m - y_{m-1}|| / ||y_m|| = ||c_m - c_{m-1}|| / ||c_m||,
        # where y_m = beta0 * V_m c_m.
        diff = coeff_curr - coeff_prev
        numer = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(diff, diff)), jnp.array(0.0, dtype=REAL_DTYPE)))
        denom = jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(coeff_curr, coeff_curr)), jnp.array(0.0, dtype=REAL_DTYPE)))
        denom = jnp.maximum(denom, jnp.asarray(1e-12, dtype=REAL_DTYPE))
        rel_change = numer / denom

        def second_pass(coeffs, m_dim):
            """Reconstruct y = beta0 * V_m coeffs via a second Lanczos pass."""

            def body2(j, state2):
                q_prev2, q_curr2, acc = state2
                j_idx = j + 1  # 1-based index for current step

                # β_j (with β_1 = 0, β_j = beta_f[j-2] for j >= 2)
                beta_j = jnp.where(
                    j_idx == 1,
                    jnp.array(0.0, dtype=REAL_DTYPE),
                    beta_f[j_idx - 2],
                )
                w2 = matvec(params, q_curr2)
                w2 = jnp.asarray(w2, dtype=REAL_DTYPE)
                w2 = w2 - beta_j * q_prev2
                alpha_j = alpha_f[j_idx - 1]
                w2 = w2 - alpha_j * q_curr2
                beta_j1 = beta_f[j_idx - 1]

                nonzero2 = beta_j1 > breakdown_tol
                beta_safe2 = jnp.where(nonzero2, beta_j1, jnp.array(1.0, dtype=REAL_DTYPE))
                q_next2 = jnp.where(nonzero2, w2 / beta_safe2, jnp.zeros_like(q_curr2))

                acc = acc + coeffs[j_idx] * q_next2
                return q_curr2, q_next2, acc

            # j = 0 corresponds to v_1 = vec / beta0
            q1 = vec / beta0
            acc0 = coeffs[0] * q1
            init_state2 = (jnp.zeros_like(q1), q1, acc0)

            # Run for j = 1 .. m_dim-1
            final_state2 = lax.fori_loop(0, jnp.maximum(m_dim - 1, 0), body2, init_state2)
            _, _, acc_final = final_state2
            return beta0 * acc_final

        approx = second_pass(coeff_curr, active_dim)
        return approx, rel_change, active_dim

    approx, rel_change, actual_iters = lax.cond(
        vec_norm <= 0.0,
        lambda _: (zero, jnp.array(0.0, dtype=REAL_DTYPE), jnp.array(0, dtype=jnp.int32)),
        lambda beta0: _run_lanczos(beta0),
        vec_norm,
    )

    tol_arr = jnp.asarray(tol, dtype=REAL_DTYPE)

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


def lanczos_sqrt_mv_test(key: jax.Array,
                         *,
                         n: int = 64,
                         iters: int = 32,
                         tol: float = 1e-3) -> float:
    """
    Convenience test helper for ``lanczos_sqrt_mv``.

    Generates a random SPD matrix ``M = A^T A + ε I`` and compares the Lanczos
    approximation against the dense eigen-decomposition reference. The return
    value is the relative 2-norm error.
    """
    key_mat, key_vec = jax.random.split(key)
    A = jax.random.normal(key_mat, (n, n), dtype=REAL_DTYPE)
    M = A.T @ A + 1e-3 * jnp.eye(n, dtype=REAL_DTYPE)

    def mv(params, x):
        mat, = params
        return mat @ x

    noise = jax.random.normal(key_vec, (n,), dtype=REAL_DTYPE)
    approx = lanczos_sqrt_mv(mv, (M,), noise, iters=iters, tol=tol)
    evals, evecs = jnp.linalg.eigh(M)
    evals = jnp.clip(evals, a_min=0.0)
    exact = evecs @ (jnp.sqrt(evals) * (evecs.T @ noise))
    rel_err = jnp.linalg.norm(approx - exact) / jnp.linalg.norm(exact)
    return float(rel_err)


@partial(jax.jit, static_argnames=('iters', 'tol', 'return_info'))
def sample_mr_sqrt_precond(key: jax.Array,
                           state: RealSpaceState,
                           positions: jnp.ndarray,
                           *,
                           precond: Optional[Preconditioner] = None,
                           iters: int = 3,
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
        Maximum number of Lanczos iterations (default 3).
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
                   iters: int = 20,
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
