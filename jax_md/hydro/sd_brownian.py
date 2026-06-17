"""Brownian motion for Fast Stokesian Dynamics (Phase 3).

Adds thermal fluctuations to the deterministic Phase-2 saddle solve
(:mod:`jax_md.hydro.rpy_saddle`).  One overdamped Euler--Maruyama timestep
samples the stochastic displacement from the fluctuation--dissipation
distribution ``N(0, 2kT dt R_FU^{-1})`` *without* forming ``R_FU^{-1/2}``,
plus the thermal drift ``kT div R_FU^{-1}`` by random finite differencing.

The Brownian force splits into a far-field slip velocity and a near-field
force, both injected on the RHS of the *same* indefinite saddle matrix the
deterministic solve already inverts (Fiore & Swan 2019):

    [[ M    B      ]] (F_hat)   (U^B + E^inf)
    [[ Bᵀ  -R^nf_FU ]] ( U  ) = (-(F^P + R^nf_FE:E^inf + F^B_nf))

* **Far-field slip** ``U^B ~ N(0, (2kT/dt) M)`` -- reuses the existing PSE grand
  sampler (real-space Lanczos square root + analytic wave-space square root) in
  the flat-11 moment layout the saddle top block expects.  No new math (Sec. 1).
* **Near-field force** ``F^B_nf ~ N(0, (2kT/dt) R^nf_FU)`` -- new: a
  preconditioned Krylov square root of the matrix-free ``R^nf_FU`` (Sec. 2).
* **RFD drift** -- two extra displaced saddle solves with a fixed RHS (Sec. 4).

Everything is additive and gated behind the SD-Brownian path; the RPY and
stresslet-RPY Brownian paths are untouched.  The step runs *eagerly* (host
RCM/IC(0) factor + ``jax.pure_callback`` triangular solves), like the Phase-2
``solve_fn``; production-fast jitting is deferred.

Near-field preconditioner -- the covariance-correctness landmine
----------------------------------------------------------------
The preconditioned near-field square root uses two *distinct* objects on the
neighborless rows (a particle with no neighbor within ``r_lub`` has an exactly
zero ``R^nf_FU`` row):

* ``Shift_nn`` -- an additive conditioning shift ``zeta * (1,1,1,4/3,4/3,4/3)``
  that lives **only** inside the sampled (Lanczos) operator.  Its value is
  irrelevant to the output covariance; it only makes the IC(0) factor ``L`` a
  good preconditioner on those rows.
* ``Proj`` -- a strict 0/1 idempotent projector (zero on all six rows of
  neighborless particles) applied **only** in the unwind.  It carries no ``4/3``
  and is *not* ``1 - Shift_nn``.

With ``c = 2kT/dt`` the unwind ``F = sqrt(c) * Proj * D * Pᵀ * L * z`` yields
``Cov(F) = c * Proj (R^nf_FU + Shift_nn) Proj = c * R^nf_FU`` exactly, because
``R^nf_FU`` already vanishes on neighborless rows/cols (so ``Proj R Proj = R``)
and ``Proj`` annihilates the ``Shift_nn`` rows (so ``Proj Shift Proj = 0``).
Collapsing the two into one ``4/3``-scaled ``I_nn`` and using ``1 - I_nn`` would
subtract ``1/3`` from the *neighbored* particles' rotational rows -- a silent
covariance corruption.  The separation prevents it by construction; check 1's
torque-row assertion is the empirical gate.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_md.hydro.rpy_saddle import (
    build_saddle_solve,
    build_ic0_from_state,
    nearfield_FU_diagonal,
    _gv_from_u6,
)
from jax_md.hydro.rpy_real_det_helpers import REAL_DTYPE
from jax_md.hydro.rpy_real_stoch import lanczos_sqrt_mv
from jax_md.hydro.rpy_brownian_constrained import (
    make_real_grand_slip_sampler,
    make_grand_slip_sampler,
    grand_jacobi_preconditioner,
)
from jax_md.hydro.rpy_wave_stoch import build_Mw_grand_sqrt_sampler
from jax_md.hydro.rpy_moments import grand_to_flat


# Isolated-sphere rotational self-resistance scale (in a-units) for the
# neighborless conditioning shift; matches FSD Helper_Precondition.cu.
_SHIFT_PATTERN = np.array([1.0, 1.0, 1.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0])


# ---------------------------------------------------------------------------
# Sec. 1 -- far-field Brownian slip (reuses the existing PSE grand sampler)
# ---------------------------------------------------------------------------
def make_far_field_slip_sampler(solve_fn, state, positions, kT, dt, *,
                                mr_iters=50, lanczos_tol=1e-3):
  """Build a jitted far-field slip sampler ``sampler(key) -> U_B_flat (N,11)``.

  ``Cov(U_B_flat) = (2kT/dt) M_grand`` -- the slip *velocity* in the flat-11
  grand layout the saddle top block expects (translation + couplet).  Runs the
  real-space grand Lanczos under ``jit`` (suppresses the eager non-convergence
  raise) and reuses the analytic wave-space square root.  ``state.rpy.wave``
  must carry the grand ``Pdip`` modes (the saddle's stresslet mobility does).
  """
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  precond = grand_jacobi_preconditioner(solve_fn.a, solve_fn.xi, solve_fn.eta)
  real_sampler = make_real_grand_slip_sampler(
      real_state=state.rpy.real, positions=positions,
      preconditioner=precond, iters=mr_iters, tol=lanczos_tol)
  wave_sqrt = build_Mw_grand_sqrt_sampler(state.rpy.wave)
  slip_sampler = make_grand_slip_sampler(
      real_sampler=real_sampler,
      wave_sampler=lambda k: wave_sqrt(k, positions, None),
      kT=kT, dt=dt)

  @jax.jit
  def sampler(key):
    U_B, D_B, _info = slip_sampler(key)
    return grand_to_flat(U_B, D_B)

  return sampler


def far_field_slip(solve_fn, state, positions, kT, dt, key, *,
                   mr_iters=50, lanczos_tol=1e-3):
  """Convenience single draw of the far-field slip ``U_B_flat`` ``(N,11)``."""
  sampler = make_far_field_slip_sampler(
      solve_fn, state, positions, kT, dt,
      mr_iters=mr_iters, lanczos_tol=lanczos_tol)
  return sampler(key), {}


# ---------------------------------------------------------------------------
# Sec. 2 -- near-field Brownian force (the new preconditioned square root)
# ---------------------------------------------------------------------------
def make_nearfield_brownian_sampler(solve_fn, state, positions, kT, dt, *,
                                    preconditioned=True, iters=20, tol=1e-3,
                                    ic0=None):
  """Build a jitted sampler ``F^B_nf ~ N(0, (2kT/dt) R^nf_FU)`` ``(N,6)``.

  Returns ``sampler(key) -> F (N,6)``.  The square root runs under ``jit`` so a
  non-converged / early-breakdown Lanczos returns the best-effort iterate rather
  than raising (the eager ``lanczos_sqrt_mv`` raises on non-convergence even with
  ``return_info``); jit also amortizes compilation across a sample loop.

  ``preconditioned=False`` is the bare square root of the matrix-free
  ``R^nf_FU`` (first-light covariance gate).  ``preconditioned=True`` adds the
  ``D / L / Shift_nn / Proj`` conditioning and must give the *same* covariance.
  """
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  N = positions.shape[0]
  a, eta = solve_fn.a, solve_fn.eta
  zeta, r_p, r_lub = solve_fn.zeta, solve_fn.r_p, solve_fn.r_lub
  nf_apply = solve_fn.nf_apply
  scale = jnp.sqrt(jnp.asarray(2.0 * kT / dt, dtype=REAL_DTYPE))

  def rnf_FU(u6):
    """Matrix-free ``R^nf_FU``: ``(N,6) [U|Omega] -> (N,6) [F|L]``."""
    gf, _ = nf_apply(state.nf, positions, _gv_from_u6(u6))
    return gf[..., :6]

  if not preconditioned:
    @jax.jit
    def sampler(key):
      noise = jax.random.normal(key, (N, 6), dtype=REAL_DTYPE)
      half = lanczos_sqrt_mv(lambda _p, x: rnf_FU(x), None, noise,
                             iters=iters, tol=tol, return_info=True)[0]
      return scale * half
    return sampler

  # -- Host-side preconditioner pieces (built once at this configuration) ----
  box = np.asarray(state.rpy.real.box_matrix, dtype=np.float64)
  cart = np.asarray(positions, dtype=np.float64) @ box.T
  if ic0 is None:
    ic0 = build_ic0_from_state(state, a, eta, r_p=r_p, zeta=zeta)
  diag6, has_neighbor = nearfield_FU_diagonal(cart, box, a, eta, r_lub)

  d = diag6.reshape(-1)                                    # (6N,) per-DOF diag
  D = np.where((d > 0.0) & (d < zeta), np.sqrt(d), 1.0)    # diagonal scaling
  # Additive conditioning shift (operator only); zeta*(1,1,1,4/3,4/3,4/3) on
  # neighborless rows -- value irrelevant to the covariance, projected out below.
  shift = np.zeros((N, 6), dtype=np.float64)
  shift[~has_neighbor] = zeta * _SHIFT_PATTERN
  shift = shift.reshape(-1)
  # Strict 0/1 output projector (unwind only); zeros the (exactly zero) rows of
  # neighborless particles.  NOT 1 - shift: carries no 4/3.
  proj = np.ones((N, 6), dtype=np.float64)
  proj[~has_neighbor] = 0.0
  proj = proj.reshape(-1)

  perm = np.asarray(ic0.perm)
  D_j = jnp.asarray(D, dtype=REAL_DTYPE)
  Dp_j = jnp.asarray(D[perm], dtype=REAL_DTYPE)            # permuted diagonal
  shift_j = jnp.asarray(shift, dtype=REAL_DTYPE)
  proj_j = jnp.asarray(proj, dtype=REAL_DTYPE)
  perm_j = jnp.asarray(perm)
  inv_perm_j = jnp.asarray(ic0.inv_perm)

  # Host triangular solves / forward apply bridged into JAX.
  def _Linv(v):
    return jax.pure_callback(
        lambda x: ic0.apply_L_inv(x).astype(np.float64),
        jax.ShapeDtypeStruct(v.shape, REAL_DTYPE), v)

  def _LTinv(v):
    return jax.pure_callback(
        lambda x: ic0.apply_LT_inv(x).astype(np.float64),
        jax.ShapeDtypeStruct(v.shape, REAL_DTYPE), v)

  def _L(v):
    return jax.pure_callback(
        lambda x: ic0.apply_L(x).astype(np.float64),
        jax.ShapeDtypeStruct(v.shape, REAL_DTYPE), v)

  def rshift_perm(wp):
    """``P (R^nf_FU + Shift_nn) Pᵀ`` on a permuted vector ``wp`` (6N,)."""
    w_orig = wp[inv_perm_j]                                # Pᵀ wp -> original
    Rw = rnf_FU(w_orig.reshape(N, 6)).reshape(-1) + shift_j * w_orig
    return Rw[perm_j]                                      # P (...) -> permuted

  def apc(_params, vp):
    """Symmetric PSD preconditioned shifted operator (permuted ordering)."""
    a1 = _LTinv(vp)
    a2 = a1 / Dp_j
    a3 = rshift_perm(a2)
    a4 = a3 / Dp_j
    return _Linv(a4)

  @jax.jit
  def sampler(key):
    noise = jax.random.normal(key, (6 * N,), dtype=REAL_DTYPE)
    z = lanczos_sqrt_mv(apc, None, noise, iters=iters, tol=tol,
                        return_info=True)[0]
    # Unwind: F = sqrt(c) * Proj * D * Pᵀ * L * z  (original ordering).
    Lz_orig = _L(z)[inv_perm_j]                            # L z, then Pᵀ
    return scale * (proj_j * D_j * Lz_orig).reshape(N, 6)

  return sampler


def nearfield_brownian_force(solve_fn, state, positions, kT, dt, key, *,
                             preconditioned=True, iters=20, tol=1e-3, ic0=None):
  """Convenience single draw of ``F^B_nf`` (builds and calls a jitted sampler)."""
  sampler = make_nearfield_brownian_sampler(
      solve_fn, state, positions, kT, dt,
      preconditioned=preconditioned, iters=iters, tol=tol, ic0=ic0)
  return sampler(key)


# ---------------------------------------------------------------------------
# Sec. 4 -- thermal drift via random finite differencing (Delong et al. 2014)
# ---------------------------------------------------------------------------
def rfd_drift(init_fn, solve_fn, positions, key, *, ic0, eps, kT, shift_fn,
              atol=1e-8, atol2=None, step_kwargs=None):
  """Random-finite-difference estimate of ``kT div R_FU^{-1}`` as a velocity.

  Two displaced saddle solves at ``q +/- (eps/2) dq`` with the fixed RHS
  ``(0; -dq)`` (over generalized force ``(N,6)``); drift ``= (kT/eps)(U_+ - U_-)``.
  The IC(0) factor ``ic0`` (built at ``q``) is **reused** for both displaced
  solves (it only affects convergence, never the solution).  Both solves use a
  fixed *absolute* GMRES tolerance ``atol`` (with ``tol=0``) so the difference
  is not corrupted by a warm-start residual asymmetry (amplified by ``1/eps``).

  ``atol2`` overrides the absolute tolerance of the second (warm-started) solve;
  default ``atol``.  Used only by the validation negative control -- loosening it
  deliberately reintroduces the warm-start asymmetry that check 4 must catch.
  """
  step_kwargs = step_kwargs or {}
  atol2 = atol if atol2 is None else atol2
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  N = positions.shape[0]
  dq6 = jax.random.normal(key, (N, 6), dtype=REAL_DTYPE)
  dq_pos = dq6[..., :3]

  q_plus = shift_fn(positions, 0.5 * eps * dq_pos, **step_kwargs)
  q_minus = shift_fn(positions, -0.5 * eps * dq_pos, **step_kwargs)
  state_p = init_fn(q_plus)
  state_m = init_fn(q_minus)

  Up, Op, _s5p, q11p, _ip = solve_fn(
      state_p, q_plus, extra_force=dq6, ic0=ic0, tol=0.0, atol=atol)
  x0 = (q11p, jnp.concatenate([Up, Op], axis=-1))          # warm start
  Um, Om, _s5m, _q11m, _im = solve_fn(
      state_m, q_minus, extra_force=dq6, ic0=ic0, x0=x0, tol=0.0, atol=atol2)

  U_plus = jnp.concatenate([Up, Op], axis=-1)
  U_minus = jnp.concatenate([Um, Om], axis=-1)
  return (kT / eps) * (U_plus - U_minus)


# ---------------------------------------------------------------------------
# Sec. 5 -- one Brownian timestep
# ---------------------------------------------------------------------------
def build_sd_brownian_step(
    space_fns,
    a: float,
    eta: float,
    dt: float,
    kT: float,
    *,
    xi: Optional[float] = None,
    n_particles: Optional[int] = None,
    phi: Optional[float] = None,
    rfd_epsilon: float = 1e-4,
    rfd_atol: float = 1e-8,
    r_lub: Optional[float] = None,
    r_p: Optional[float] = None,
    mr_iters: int = 50,
    nf_iters: int = 20,
    lanczos_tol: float = 1e-3,
    gmres_tol: float = 1e-3,
    **rpy_kwargs,
) -> Tuple[Callable, Callable]:
  """Build the overdamped FSD Brownian timestep (Euler--Maruyama + RFD drift).

  Args:
    space_fns: ``(displacement_fn, shift_fn)`` (static box; Phase 3 scope).
    a, eta: sphere radius and solvent viscosity.
    dt, kT: timestep and thermal energy.
    xi, n_particles, phi: Ewald split (estimated from ``tol`` if ``xi`` is None).
    rfd_epsilon: RFD finite-difference step (confirm drift is ``eps``-independent).
    rfd_atol: absolute GMRES tolerance for the two RFD displaced solves.
    r_lub, r_p: near-field and IC(0)-truncation cutoffs (defaults ``4a`` / ``2.1a``).
    mr_iters, nf_iters: Lanczos iteration caps for the far/near-field samplers.
    lanczos_tol, gmres_tol: square-root and saddle-solve tolerances.
    **rpy_kwargs: forwarded to ``build_saddle_solve`` / ``build_rpy_mobility``.

  Returns:
    ``(init_fn, step_fn)``.

    ``init_fn(positions_frac) -> SaddleState`` (the Phase-2 state).

    ``step_fn(state, positions_frac, key, *, force=None, torque=None,
    E_inf=None) -> (positions_new, S5, info)``.  Euler--Maruyama:
    ``q' = q + dt (U_total + U^inf)`` with ``U_total = U_main + U_drift``;
    ``U_main`` already superposes the deterministic and Brownian contributions
    from the single combined solve.  ``S5`` is the total stresslet ``(N,5)``.
  """
  init_fn, solve_fn = build_saddle_solve(
      space_fns, a, eta,
      xi=xi, n_particles=n_particles, phi=phi,
      r_lub=r_lub, r_p=r_p, gmres_tol=gmres_tol,
      **rpy_kwargs)
  shift_fn = space_fns[1]
  a_f, eta_f = float(a), float(eta)
  zeta = solve_fn.zeta

  def step_fn(state, positions_frac, key, *,
              force=None, torque=None, E_inf=None):
    q = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    # Clean key tree: the three random inputs must be independent (FD theorem,
    # positive split).  far_field_slip splits k_slip -> (real, wave) internally.
    k_slip, k_nf, k_rfd = jax.random.split(key, 3)

    # IC(0) factor built once at q, reused for the main and both RFD solves.
    ic0 = build_ic0_from_state(state, a_f, eta_f, r_p=solve_fn.r_p, zeta=zeta)

    # (1) far-field slip; (2) near-field Brownian force.
    slip_sampler = make_far_field_slip_sampler(
        solve_fn, state, q, kT, dt, mr_iters=mr_iters, lanczos_tol=lanczos_tol)
    nf_sampler = make_nearfield_brownian_sampler(
        solve_fn, state, q, kT, dt, preconditioned=True,
        iters=nf_iters, tol=lanczos_tol, ic0=ic0)
    U_B_flat = slip_sampler(k_slip)
    F_B_nf = nf_sampler(k_nf)

    # (3) one combined deterministic + Brownian saddle solve.
    U_main, Om_main, S5, _q11, info = solve_fn(
        state, q, force=force, torque=torque, E_inf=E_inf,
        slip_top=U_B_flat, extra_force=F_B_nf, ic0=ic0)

    # (4) RFD thermal drift (reuses ic0; absolute-tol displaced solves).
    U_drift6 = rfd_drift(
        init_fn, solve_fn, q, k_rfd,
        ic0=ic0, eps=rfd_epsilon, kT=kT, shift_fn=shift_fn, atol=rfd_atol)

    U_total6 = jnp.concatenate([U_main, Om_main], axis=-1) + U_drift6
    q_new = shift_fn(q, dt * (U_total6[..., :3] + info['U_inf']))

    out_info = dict(info)
    out_info['U_drift'] = U_drift6
    return q_new, S5, out_info

  return init_fn, step_fn
