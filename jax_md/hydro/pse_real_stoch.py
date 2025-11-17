"""Stochastic sampling utilities for the real-space PSE mobility."""

import jax
import jax.numpy as jnp
from jax import debug as jax_debug
from jax import lax
import warnings

from typing import Callable, Optional
from functools import partial

from jax_md import dataclasses
from jax_md.hydro.pse_real_det import REAL_DTYPE, RealSpaceState, mr_matvec

# --- Preconditioner for Chow–Saad-style sampling ---

@dataclasses.dataclass
class Preconditioner:
    """Linear-operator preconditioner G for Chow–Saad-style sampling.

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
                    iters: int = 20,
                    tol: float = 1e-6) -> jnp.ndarray:
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
        Relative tolerance used for convergence monitoring. The routine emits a
        ``RuntimeWarning`` via ``jax.debug.callback`` if the final iterate fails to
        satisfy the tolerance (mirrors the stopping criterion suggested by
        Chow & Saad, *SIAM J. Sci. Comput.* 36(2), 2014).

    Returns
    -------
    array
        Approximation of ``sqrt(M) @ v`` with the same shape/dtype as ``v``.
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
        q_store = jnp.zeros((iters,) + vec.shape, dtype=REAL_DTYPE)
        alphas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        betas = jnp.zeros((iters,), dtype=REAL_DTYPE)
        beta_prev = jnp.array(0.0, dtype=REAL_DTYPE)
        iters_done = jnp.array(0, dtype=jnp.int32)
        finished = jnp.array(False)

        def body(k, state):
            q_prev, q_curr, q_store, alpha_arr, beta_arr, beta_prev, it_count, done_flag = state

            def iterate(vals):
                q_prev_i, q_curr_i, q_store_i, alpha_arr_i, beta_arr_i, beta_prev_i, it_count_i, _ = vals
                w = matvec(params, q_curr_i)
                w = jnp.asarray(w, dtype=REAL_DTYPE)
                w = w - beta_prev_i * q_prev_i
                alpha = jnp.real(jnp.vdot(q_curr_i, w))
                w = w - alpha * q_curr_i
                beta = _vec_norm(w)
                nonzero = beta > breakdown_tol
                beta_safe = jnp.where(nonzero, beta, jnp.array(1.0, dtype=REAL_DTYPE))
                q_next = jnp.where(nonzero, w / beta_safe, jnp.zeros_like(q_curr_i))
                q_store_i = q_store_i.at[k].set(q_curr_i)
                alpha_arr_i = alpha_arr_i.at[k].set(alpha)
                beta_arr_i = beta_arr_i.at[k].set(beta)
                new_done = jnp.logical_not(nonzero)
                return (
                    q_curr_i,
                    q_next,
                    q_store_i,
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
                (q_prev, q_curr, q_store, alpha_arr, beta_arr, beta_prev, it_count, done_flag),
            )
            q_prev_new, q_curr_new, q_store_new, alpha_new, beta_new, beta_prev_new, it_count_new, just_finished = updated
            total_done = jnp.logical_or(done_flag, just_finished)
            return (
                q_prev_new,
                q_curr_new,
                q_store_new,
                alpha_new,
                beta_new,
                beta_prev_new,
                it_count_new,
                total_done,
            )

        init_state = (
            q_prev,
            q_curr,
            q_store,
            alphas,
            betas,
            beta_prev,
            iters_done,
            finished,
        )
        _, _, q_store_f, alpha_f, beta_f, _, iters_used, _ = lax.fori_loop(0, iters, body, init_state)
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

        def _combine(coeffs):
            return jnp.tensordot(coeffs, q_store_f, axes=((0,), (0,)))

        approx = beta0 * _combine(coeff_curr)
        approx_prev = beta0 * _combine(coeff_prev)
        denom = jnp.maximum(_vec_norm(approx), jnp.asarray(1e-12, dtype=REAL_DTYPE))
        rel_change = _vec_norm(approx - approx_prev) / denom
        return approx, rel_change, active_dim

    approx, rel_change, actual_iters = lax.cond(
        vec_norm <= 0.0,
        lambda _: (zero, jnp.array(0.0, dtype=REAL_DTYPE), jnp.array(0, dtype=jnp.int32)),
        lambda beta0: _run_lanczos(beta0),
        vec_norm,
    )

    tol_arr = jnp.asarray(tol, dtype=REAL_DTYPE)
    warn_payload = {
        'rel_change': rel_change,
        'tol': tol_arr,
        'iters': actual_iters,
        'requested': jnp.asarray(iters, dtype=jnp.int32),
    }

    def _warn_callback(data):
        rel = float(data['rel_change'])
        tol_val = float(data['tol'])
        used = int(data['iters'])
        requested = int(data['requested'])
        warnings.warn(
            f"lanczos_sqrt_mv stopped with rel_change={rel:.3e} > tol={tol_val:.3e} "
            f"after {used}/{requested} iterations.",
            RuntimeWarning,
        )

    def _warn_branch(payload):
        jax_debug.callback(_warn_callback, payload)
        return jnp.array(0, dtype=jnp.int32)

    _ = lax.cond(rel_change > tol_arr, _warn_branch, lambda _: jnp.array(0, dtype=jnp.int32), warn_payload)
    return approx


def lanczos_sqrt_mv_test(key: jax.Array,
                         *,
                         n: int = 64,
                         iters: int = 32,
                         tol: float = 1e-6) -> float:
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


@partial(jax.jit, static_argnames=('iters', 'tol'))
def sample_mr_sqrt_precond(key: jax.Array,
                           state: RealSpaceState,
                           positions_frac: jnp.ndarray,
                           *,
                           precond: Optional[Preconditioner] = None,
                           iters: int = 20,
                           tol: float = 1e-5) -> jnp.ndarray:
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
    positions_frac : (N,3) array
        Fractional positions corresponding to ``state``.
    precond : Preconditioner, optional
        Linear operator ``G`` used for preconditioning. Defaults to the identity.
    iters : int, optional
        Maximum number of Lanczos iterations (default 20).
    tol : float, optional
        Convergence tolerance (default 1e-6). Monitors the change in approximation
        between consecutive iterations.

    Returns
    -------
    (N,3) array
        Sample from the real-space stochastic increment (in real coordinates).
    """
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    precond = _IDENTITY_PRECONDITIONER if precond is None else precond

    noise = jax.random.normal(key, shape=positions_frac.shape, dtype=REAL_DTYPE)

    def mv(mv_params, x):
        st, pos, pc = mv_params
        x_phys = pc.apply_T(x)
        m_x = mr_matvec(st, pos, x_phys)
        return pc.apply(m_x)

    mv_params = (state, positions_frac, precond)
    precond_sample = lanczos_sqrt_mv(mv, mv_params, noise, iters=iters, tol=tol)
    return precond.solve(precond_sample)


@partial(jax.jit, static_argnames=('iters', 'tol'))
def sample_mr_sqrt(key: jax.Array,
                   state: RealSpaceState,
                   positions_frac: jnp.ndarray,
                   *,
                   iters: int = 20,
                   tol: float = 1e-5) -> jnp.ndarray:
    """
    Convenience wrapper for ``sample_mr_sqrt_precond`` using the identity preconditioner.
    """
    return sample_mr_sqrt_precond(
        key,
        state,
        positions_frac,
        precond=_IDENTITY_PRECONDITIONER,
        iters=iters,
        tol=tol,
    )
