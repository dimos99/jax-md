"""Brownian motion for Fast Stokesian Dynamics (Phase 3).

Adds thermal fluctuations to the deterministic Phase-2 saddle solve
(:mod:`jax_md.hydro.sd_saddle`).  One overdamped Euler--Maruyama timestep
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
stresslet-RPY Brownian paths are untouched.  The whole step is **device
resident and jitted**: the deterministic/RFD saddle solves use the on-device
Chebyshev-Schur preconditioner (the ``build_saddle_solve`` default) and the
Brownian square roots use on-device
samplers (real-space Lanczos + analytic wave-space + Jacobi-preconditioned
near-field Lanczos), so there is no host IC(0) factor and no ``jax.pure_callback``
anywhere in the timestep.

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

from functools import partial
from typing import Callable, Optional, Tuple
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from jax_md.hydro import sd_nearfield_table as nf_table
from jax_md.hydro.sd_saddle import (
    build_saddle_solve,
)
from jax_md.hydro.rpy import _sample_wave_grand_noise
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
# neighborless conditioning shift; matches the original FSD code.
_SHIFT_PATTERN = np.array([1.0, 1.0, 1.0, 4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0])


# --------------------------------------------------------------------------- #
# RFD step-size selection
# --------------------------------------------------------------------------- #
# The drift is a centered difference of the saddle solve, so ``eps`` is bounded
# from both sides.
#
# Ceiling: the lubrication clamp flattens ``R^nf`` below a surface gap, leaving
# a C0 corner.  A difference straddling it returns the mean of the two one-sided
# slopes rather than either, and shrinking ``eps`` does not repair that -- the
# drift comes out too large inside the plateau (true slope ~0 there) and too
# small just outside it.  Pairs away from contact are unaffected.
#
# Floor: the drift divides by ``eps``, amplifying the displaced solves' residual
# by ``1/eps``.  Set by the residual ACHIEVED, not by ``rfd_atol`` -- float64
# converges below any requested tolerance, so sweeping ``rfd_atol`` there says
# nothing about this bound.
#
# The clamp cancels out of the floor: the condition number grows as the clamp
# gap shrinks while the response's variation length shrinks with it.  So the
# clamp sets the ceiling directly and the floor only through stiffness -- when
# the window closes, widen the clamp rather than shrink ``eps``.
#
# The floor constants are calibrations, not derivations; the achievable residual
# is configuration dependent.  To check, sweep ``eps`` and look for a plateau in
# the drift magnitude, holding the RFD direction fixed (single-sample estimator,
# so re-drawing compares samples rather than step sizes).
_RFD_CEILING_FRAC = 0.3
_RFD_FLOOR_F32 = 1e-4
_RFD_FLOOR_F64 = 1e-5
# Below this the clamp bounds nothing (float64 defaults to the table's lowest
# tabulated gap): no corner, but also no bound on pair separation, so no ``eps``
# is valid for the tightest pairs.  Treat the ceiling as absent rather than warn
# on a configuration the user never chose.
_RFD_CLAMP_ACTIVE_GAP = 1e-6


def lubrication_clamp_gap() -> float:
  """Surface gap (units of ``a``) at the near-field resistance clamp row.

  Resolved once at import from the dtype and the ``JAX_MD_SD_MIN_GAP``
  environment variable; see :mod:`jax_md.hydro.sd_nearfield_table`.
  """
  table = nf_table.load_resistance_table()
  return float(table.dist[nf_table.REGULARIZATION_INDEX]) - 2.0


def rfd_epsilon_bounds(a) -> Tuple[float, Optional[float], float]:
  """``(floor, ceiling, clamp_gap)`` for the RFD step at sphere radius ``a``.

  ``ceiling`` is ``None`` when no lubrication clamp is active.  ``eps`` is a
  displacement, so both bounds are lengths and scale with ``a``; ``clamp_gap``
  is dimensionless (units of ``a``).
  """
  clamp_gap = lubrication_clamp_gap()
  floor = float(a) * (_RFD_FLOOR_F32 if REAL_DTYPE == jnp.float32
                      else _RFD_FLOOR_F64)
  if clamp_gap <= _RFD_CLAMP_ACTIVE_GAP:
    return floor, None, clamp_gap
  return floor, _RFD_CEILING_FRAC * float(a) * clamp_gap, clamp_gap


def _resolve_rfd_epsilon(rfd_epsilon, gmres_tol, a):
  """Pick (or validate) the RFD step against the ceiling and floor above.

  Returns ``(eps, floor, ceiling, clamp_gap)``.  A user-supplied value is never
  overridden -- only warned about -- so an explicit choice always wins.
  """
  floor, ceiling, clamp_gap = rfd_epsilon_bounds(a)
  # Historical default: FSD ties eps to the solver tolerance.  Kept as the
  # starting point so runs already inside the window are unchanged.
  legacy = max(float(gmres_tol), 1e-4)
  supplied = rfd_epsilon is not None
  eps = float(rfd_epsilon) if supplied else legacy

  if ceiling is not None and ceiling < floor:
    # No valid eps exists -- a clamp problem, not an eps problem.  Name the
    # clamp's origin: in float32 the default comes from table resolution, so the
    # user may have chosen nothing.
    origin = ('the JAX_MD_SD_MIN_GAP setting'
              if os.environ.get(nf_table._MIN_GAP_ENV)
              else f'the default {REAL_DTYPE.__name__} table resolution')
    warnings.warn(
        f"The lubrication clamp gap is {clamp_gap:.2e} (units of a), from "
        f"{origin}, which is too tight for {REAL_DTYPE.__name__}: the RFD step "
        f"must be below {ceiling:.2e} to resolve the clamp corner but above "
        f"~{floor:.2e} to survive 1/eps amplification of the solve residual. "
        f"No rfd_epsilon satisfies both, so the Brownian drift is unconverged "
        f"for pairs within ~{eps / (_RFD_CEILING_FRAC * float(a)):.1e} of "
        f"contact. Set JAX_MD_SD_MIN_GAP to at least "
        f"{floor / (_RFD_CEILING_FRAC * float(a)):.1e} (before the first "
        f"jax_md.hydro import) to open the window, or proceed knowing the "
        f"near-contact drift is unresolved.",
        UserWarning, stacklevel=3)
  elif supplied:
    if ceiling is not None and eps > ceiling:
      warnings.warn(
          f"rfd_epsilon={eps:.2e} exceeds the lubrication-clamp ceiling "
          f"{ceiling:.2e} (= {_RFD_CEILING_FRAC} * a * clamp gap "
          f"{clamp_gap:.2e}). The RFD probes will straddle the clamp corner, "
          f"which biases the Brownian drift for near-contact pairs by factors "
          f"of 3-40x (over-estimating inside the clamp, under-estimating just "
          f"outside). Use rfd_epsilon <= {ceiling:.2e}, or raise "
          f"JAX_MD_SD_MIN_GAP.",
          UserWarning, stacklevel=3)
    elif eps < floor:
      warnings.warn(
          f"rfd_epsilon={eps:.2e} is below the ~{floor:.2e} floor for "
          f"{REAL_DTYPE.__name__}: the drift divides by eps, so the solve "
          f"residual is amplified by 1/eps and can dominate the result "
          f"(observed in f32 near contact as a drift velocity of the wrong "
          f"sign). Confirm the drift is eps-independent before relying on it.",
          UserWarning, stacklevel=3)
  elif eps > (ceiling if ceiling is not None else eps):
    # Default above the ceiling: move it in, landing on the geometric centre for
    # maximum log-margin from both bounds (neither is a sharp edge).
    eps = float(np.sqrt(floor * ceiling))
  elif eps < floor:
    eps = floor

  return eps, floor, ceiling, clamp_gap


# ---------------------------------------------------------------------------
# Sec. 1 -- far-field Brownian slip (reuses the existing PSE grand sampler)
# ---------------------------------------------------------------------------
def make_far_field_slip_sampler(solve_fn, state, positions, kT, dt, *,
                                mr_iters=50, lanczos_tol=1e-3, precond=None,
                                wave_sqrt=None, current_box=None):
  """Build a jitted far-field slip sampler ``sampler(key) -> U_B_flat (N,11)``.

  ``Cov(U_B_flat) = (2kT/dt) M_grand`` -- the slip *velocity* in the flat-11
  grand layout the saddle top block expects (translation + couplet).  Runs the
  real-space grand Lanczos under ``jit`` (suppresses the eager non-convergence
  raise) and reuses the analytic wave-space square root.  ``state.rpy.wave``
  must carry the grand ``Pdip`` modes (the saddle's stresslet mobility does).

  ``precond`` is the config-independent grand Jacobi preconditioner; pass a
  prebuilt one (it calls the jitted ``Mr_self``, so building it *inside* an
  outer ``jit`` would raise a ConcretizationTypeError on its ``float(...)``).

  ``current_box`` (live shear): the real-space Lanczos already follows the
  deformed box via ``state.rpy.real.box_matrix`` (refreshed each step), so only
  the wave-space sqrt needs the live box -- and it needs the *exact*
  deformed-box noise (``_sample_wave_grand_noise`` rebuilds the screened k-modes)
  rather than the cached static sampler's position-remap-only path, which keeps
  the base-box modes and is wrong under shear.  ``None`` -> static box.
  """
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  if precond is None:
    precond = grand_jacobi_preconditioner(
        solve_fn.a, solve_fn.xi, solve_fn.eta)
  real_sampler = make_real_grand_slip_sampler(
      real_state=state.rpy.real, positions=positions,
      preconditioner=precond, iters=mr_iters, tol=lanczos_tol)
  if current_box is None:
    # The wave sqrt sampler bakes in the (static-box) grid stencil support P /
    # mode arrays, which must be concrete -- pass a prebuilt one when constructing
    # this inside an outer jit (see ``build_sd_brownian_step``).
    if wave_sqrt is None:
      wave_sqrt = build_Mw_grand_sqrt_sampler(state.rpy.wave)
    wave_sampler = lambda k: wave_sqrt(k, positions, None)
  else:
    # Exact deformed-box wave noise (covariance M^(w)_grand at the live box).
    wave_sampler = lambda k: _sample_wave_grand_noise(
        static=solve_fn.wave_static, current_box=current_box,
        positions_frac=positions, key_wave=k,
        a=solve_fn.a, xi=solve_fn.xi, eta=solve_fn.eta)
  slip_sampler = make_grand_slip_sampler(
      real_sampler=real_sampler,
      wave_sampler=wave_sampler,
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
                                    prepared=None):
  """Build a jitted sampler ``F^B_nf ~ N(0, (2kT/dt) R^nf_FU)`` ``(N,6)``.

  Returns ``sampler(key) -> F (N,6)``.  The square root runs under ``jit`` so a
  non-converged / early-breakdown Lanczos returns the best-effort iterate rather
  than raising (the eager ``lanczos_sqrt_mv`` raises on non-convergence even with
  ``return_info``); jit also amortizes compilation across a sample loop.

  ``preconditioned=False`` is the bare square root of the matrix-free
  ``R^nf_FU`` (first-light covariance gate).  ``preconditioned=True`` adds the
  on-device ``D / Shift_nn / Proj`` Jacobi conditioning and must give the *same*
  covariance.

  ``prepared`` optionally supplies blocks already built by
  ``solve_fn.nf_apply.prepare(state.nf, positions)`` (the Brownian step shares
  one prepare across its consumers); default ``None`` prepares here.
  """
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  N = positions.shape[0]
  zeta = solve_fn.zeta
  nf_apply = solve_fn.nf_apply
  scale = jnp.sqrt(jnp.asarray(2.0 * kT / dt, dtype=REAL_DTYPE))

  # Per-pair blocks precomputed ONCE for the fixed configuration ``positions``
  # (for which ``state.nf.neighbors`` was built): every one of the ``iters``
  # Lanczos matvecs is then gather -> block multiply -> segment_sum instead of
  # re-deriving geometry + table + block assembly (the FSD amortization).
  if prepared is None:
    prepared = nf_apply.prepare(state.nf, positions)

  def rnf_FU(u6):
    """``R^nf_FU`` from prepared blocks: ``(N,6) [U|Omega] -> (N,6) [F|L]``."""
    return nf_apply.apply_blocks_FU(prepared, u6)

  if not preconditioned:
    @jax.jit
    def sampler(key):
      noise = jax.random.normal(key, (N, 6), dtype=REAL_DTYPE)
      half = lanczos_sqrt_mv(lambda _p, x: rnf_FU(x), None, noise,
                             iters=iters, tol=tol, return_info=True)[0]
      return scale * half
    return sampler

  # -- On-device Jacobi (diagonal) preconditioner pieces (built once) --------
  # The preconditioner only conditions the Lanczos square root; it does not
  # change the sampled covariance (the Shift_nn / Proj separation below does
  # that exactly).  The host RCM + IC(0) factor is replaced by the diagonal
  # split ``D = sqrt(diag(R^nf_FU))`` -- fully jittable, no ``pure_callback``.
  # Diagonal and neighbored mask read off the same prepared blocks (equality
  # with ``diagonal_FU_mask_prepared`` pinned by test_prepared_blocks_match_core).
  diag6 = nf_apply.prepared_diag_FU(prepared)
  has_neighbor = prepared.has_neighbor
  d = diag6.reshape(-1)                                    # (6N,) per-DOF diag
  D = jnp.where((d > 0.0) & (d < zeta), jnp.sqrt(d), 1.0)  # diagonal scaling
  no_nbr = (~has_neighbor)[:, None]                        # (N,1)
  # Additive conditioning shift (operator only); zeta*(1,1,1,4/3,4/3,4/3) on
  # neighborless rows -- value irrelevant to the covariance, projected out below.
  shift = (no_nbr * (zeta * jnp.asarray(_SHIFT_PATTERN, dtype=REAL_DTYPE))
           ).reshape(-1)                                   # (6N,)
  # Strict 0/1 output projector (unwind only); zeros the (exactly zero) rows of
  # neighborless particles.  NOT 1 - shift: carries no 4/3.
  proj = jnp.where(no_nbr, jnp.asarray(0.0, REAL_DTYPE),
                   jnp.asarray(1.0, REAL_DTYPE))
  proj = jnp.broadcast_to(proj, (N, 6)).reshape(-1)        # (6N,)

  def apc(_params, v):
    """Symmetric PSD Jacobi-preconditioned shifted operator ``D^-1(R+Shift)D^-1``."""
    a2 = v / D
    Ra = rnf_FU(a2.reshape(N, 6)).reshape(-1) + shift * a2
    return Ra / D

  @jax.jit
  def sampler(key):
    noise = jax.random.normal(key, (6 * N,), dtype=REAL_DTYPE)
    z = lanczos_sqrt_mv(apc, None, noise, iters=iters, tol=tol,
                        return_info=True)[0]
    # Unwind: F = sqrt(c) * Proj * D * z.  Cov(F) = c * Proj (R+Shift) Proj
    # = c * R^nf_FU exactly (Proj annihilates the Shift rows; R vanishes there).
    return scale * (proj * D * z).reshape(N, 6)

  return sampler


def nearfield_brownian_force(solve_fn, state, positions, kT, dt, key, *,
                             preconditioned=True, iters=20, tol=1e-3):
  """Convenience single draw of ``F^B_nf`` (builds and calls a jitted sampler)."""
  sampler = make_nearfield_brownian_sampler(
      solve_fn, state, positions, kT, dt,
      preconditioned=preconditioned, iters=iters, tol=tol)
  return sampler(key)


# ---------------------------------------------------------------------------
# Sec. 4 -- thermal drift via random finite differencing (Delong et al. 2014)
# ---------------------------------------------------------------------------
def rfd_drift(solve_fn, state, positions, key, *, eps, kT, shift_fn,
              atol=1e-8, atol2=None, step_kwargs=None,
              gmres_restart=None, gmres_maxiter=None,
              gmres_solve_method='batched',
              return_stresslet=False):
  """Random-finite-difference estimate of ``kT div R_FU^{-1}`` as a velocity.

  Two displaced saddle solves at ``q +/- (eps/2) dq`` with the fixed RHS
  ``(0; -dq)`` (over generalized force ``(N,6)``); drift ``= (kT/eps)(U_+ - U_-)``.
  The displaced positions are within ``eps/2`` (``~1e-4``) of ``q`` -- far inside
  the neighbor-list skin ``dr_threshold`` -- so the ``state`` (neighbor lists +
  wave precompute) built at ``q`` is **reused** for both displaced solves: no
  host ``init_fn`` rebuild, and the whole drift is jittable.  Both solves use a
  fixed *absolute* GMRES tolerance ``atol`` (with ``tol=0``) so the difference is
  not corrupted by a warm-start residual asymmetry (amplified by ``1/eps``).

  With ``return_stresslet=True`` the same two solves also return the drift
  stresslet ``S5_drift = (kT/eps)(S5_+ - S5_-)``: a single-sample estimator of
  the mean Brownian stresslet ``<S^B> = -kT div(R_SU . R_FU^{-1})`` (Foss &
  Brady 2000, Eq. 10c).  Each displaced solve assembles
  ``S = S^ff - R^nf_SU u`` from near-field blocks prepared at the **displaced**
  positions, so the estimator carries all three product-rule terms of Eq. 10c
  -- the far-field response divergence, ``R^nf_SU . div(R_FU^{-1})``, and
  ``(grad R^nf_SU) : R_FU^{-1}`` -- with the near-contact ``1/gap``
  cancellation of the last two happening inside the ``+/-`` subtraction.  The
  stresslet difference is amplified by the same ``1/eps`` as the velocity, so
  the matched absolute-tolerance discipline applies to it unchanged.

  ``atol2`` overrides the absolute tolerance of the second (warm-started) solve;
  default ``atol``.  Used only by the validation negative control -- loosening it
  deliberately reintroduces the warm-start asymmetry that check 4 must catch.

  ``gmres_solve_method`` **must stay ``'batched'`` here**, and the reason is the
  same matched-tolerance discipline.  jax's ``'incremental'`` GMRES exits a
  restart as soon as its *preconditioned residual estimate* meets ``ptol``,
  which the cold and the warm solve reach at different true residuals; that
  asymmetry is then amplified by ``1/eps``.  ``'batched'`` cannot exit early, so
  both displaced solves run the identical iteration count and land symmetrically
  far below ``atol``.  Measured on a disordered N=108 phi=0.50 config (min gap
  0.073a), sweeping ``eps`` at the production budget (restart=50, maxiter=2):

    f64, atol=1e-8   -- ``'batched'`` returns 0.24768 flat for eps 1e-5..1e-8;
      ``'incremental'`` drifts 0.24768 -> 0.24774 -> 0.24827 -> 0.25350, i.e.
      the two disagree by 3.9e-5 at eps=1e-5 growing to 3.6e-2 at eps=1e-8.
      At the default eps it is already 180x less accurate against a tight
      reference (3.50e-6 vs 1.93e-8 relative) for 35% fewer matvecs.
    f32, atol=1e-4   -- at eps<=1e-7 the displacement is below f32 resolution
      and ``'batched'`` correctly returns exactly 0; ``'incremental'`` returns
      8.4 and 74 against a true scale of 0.19.

  Matching on delivered accuracy instead (``'incremental'`` at atol=1e-10 vs
  ``'batched'`` at atol=1e-8, both ~2e-8 relative error) leaves only a 17%
  matvec saving in f64 -- and none of it survives f32 or small eps.  The knob
  exists so this can be re-measured on other hardware, not because
  ``'incremental'`` is the safer choice for a warm-started solve.

  Returns ``U_drift (N, 6)``, or ``(U_drift, S5_drift (N, 5))`` when
  ``return_stresslet=True``.
  """
  step_kwargs = step_kwargs or {}
  atol2 = atol if atol2 is None else atol2
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  N = positions.shape[0]
  dq6 = jax.random.normal(key, (N, 6), dtype=REAL_DTYPE)
  dq_pos = dq6[..., :3]

  q_plus = shift_fn(positions, 0.5 * eps * dq_pos, **step_kwargs)
  q_minus = shift_fn(positions, -0.5 * eps * dq_pos, **step_kwargs)

  # ``step_kwargs`` carries the live shear gammas; forward them to the displaced
  # solves so the wave/real/near-field matvecs use the deformed box (the
  # neighbor lists built at ``q`` are reused -- the eps-displacement is far
  # inside the skin, and the box is essentially unchanged over eps).
  Up, Op, s5p, q11p, _ip = solve_fn(
      state, q_plus, extra_force=dq6, tol=0.0, atol=atol,
      gmres_restart=gmres_restart, gmres_maxiter=gmres_maxiter,
      gmres_solve_method=gmres_solve_method,
      return_stresslet=return_stresslet, return_residual=False, **step_kwargs)
  x0 = (q11p, jnp.concatenate([Up, Op], axis=-1))          # warm start
  Um, Om, s5m, _q11m, _im = solve_fn(
      state, q_minus, extra_force=dq6, x0=x0, tol=0.0, atol=atol2,
      gmres_restart=gmres_restart, gmres_maxiter=gmres_maxiter,
      gmres_solve_method=gmres_solve_method,
      return_stresslet=return_stresslet, return_residual=False, **step_kwargs)

  U_plus = jnp.concatenate([Up, Op], axis=-1)
  U_minus = jnp.concatenate([Um, Om], axis=-1)
  U_drift = (kT / eps) * (U_plus - U_minus)
  if return_stresslet:
    return U_drift, (kT / eps) * (s5p - s5m)
  return U_drift


def _sd_coordinate_velocity(U_main, U_drift6, U_inf, advect_ambient: bool):
  """Translational velocity used to shift SD coordinates."""
  U_advect = U_main + U_drift6[..., :3]
  if advect_ambient:
    U_advect = U_advect + U_inf
  return U_advect


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
    rfd_epsilon: Optional[float] = None,
    rfd_atol: float = 1e-8,
    r_lub: Optional[float] = None,
    r_p: Optional[float] = None,
    mr_iters: int = 50,
    nf_iters: int = 40,
    lanczos_tol: float = 1e-3,
    gmres_tol: float = 1e-3,
    rfd_gmres_restart: Optional[int] = None,
    rfd_gmres_maxiter: Optional[int] = None,
    rfd_gmres_solve_method: str = 'batched',
    return_stresslet: bool = True,
    return_residual: bool = True,
    **rpy_kwargs,
) -> Tuple[Callable, Callable]:
  """Build the overdamped FSD Brownian timestep (Euler--Maruyama + RFD drift).

  Args:
    space_fns: ``(displacement_fn, shift_fn)`` or ``+(box_fn)`` for live
      Lees-Edwards shear.  Under shear, ``step_fn`` accepts the runtime gammas
      (``gamma_xy``/``gamma_xz``/``gamma_yz`` or ``shear=``) and an ambient
      velocity gradient ``L_inf`` for the affine-flow add-back; the deformed box
      threads through the deterministic solve, the exact wave-space slip noise,
      the near-field, and the RFD drift.
    a, eta: sphere radius and solvent viscosity.
    dt, kT: timestep and thermal energy.
    xi, n_particles, phi: Ewald split (estimated from ``tol`` if ``xi`` is None).
    rfd_epsilon: RFD finite-difference step.  ``None`` (default) takes the FSD
      convention ``max(gmres_tol, 1e-4)`` clipped into the window from
      :func:`rfd_epsilon_bounds`, landing on its geometric centre when that
      value is above the ceiling.  The ceiling is a fixed fraction of
      ``a * clamp_gap`` (a straddling difference cannot resolve the clamp
      corner); the floor comes from ``1/eps`` amplification of the achievable
      solve residual.  An explicit value is warned about, never overridden.  The
      ``O(eps^2)`` centered-difference bias assumes a smooth resistance, which
      is what the corner breaks; confirm the drift is ``eps``-independent when
      overriding.
    rfd_atol: absolute GMRES tolerance for the two RFD displaced solves.  In
      float32 this is clamped up to a reachable floor (~1e-5): with the f32
      residual floor at ~1e-6, an unreachable ``atol`` (e.g. 1e-8) would make
      every RFD GMRES burn its full
      ``rfd_gmres_restart * rfd_gmres_maxiter`` budget.
    r_lub, r_p: near-field and IC(0)-truncation cutoffs (defaults ``4a`` / ``2.1a``).
    mr_iters, nf_iters: Lanczos iteration caps for the far/near-field samplers.
      ``nf_iters`` defaults to 40: the near-field square root is stiff at
      dense packings, where a 20-iteration cap is unconverged at phi=0.45 near
      contact in both precisions; the far-field Lanczos converges in ~5
      iterations there, so ``mr_iters`` is pure headroom.
    lanczos_tol, gmres_tol: square-root and saddle-solve tolerances.
    rfd_gmres_restart, rfd_gmres_maxiter: GMRES budget for the two RFD displaced
      solves (default ``restart=50, maxiter=2`` in both precisions -- the
      original FSD restart length, two cycles).  On near-contact phi=0.45
      configurations this gives ~0.5% drift error; one cycle gives 10-40%,
      while the main solve's budget reaches 1e-6 at twice the cost -- far
      tighter than this single-sample estimator can use.  Deliberately
      truncated, so these solves report no residual and never warn.
    rfd_gmres_solve_method: jax GMRES implementation for the two displaced
      solves.  Leave at ``'batched'``: ``'incremental'`` exits a restart on a
      residual *estimate*, which the cold and warm solve hit at different true
      residuals, and ``1/eps`` amplifies the asymmetry -- it is measurably less
      accurate in f64 and returns garbage in f32 at small ``eps``.  See
      :func:`rfd_drift` for the numbers.
    **rpy_kwargs: forwarded to ``build_saddle_solve`` / ``build_rpy_mobility``.

  Returns:
    ``(init_fn, step_fn)``.

    ``init_fn(positions_frac) -> SaddleState`` (the Phase-2 state).

    ``step_fn(state, positions_frac, key, *, force=None, torque=None,
    E_inf=None, x0=None) -> (positions_new, S5, info)``.  ``x0`` warm-starts
    the main saddle solve with the previous step's solution (thread
    ``info['x0']`` between steps, as the FSD reference does; ``None`` = cold
    start -- convergence-only, never changes the converged solution).
    Euler--Maruyama advances by
    ``U_total = U_main + U_drift``; ``U_main`` already superposes the
    deterministic and Brownian contributions from the single combined solve.
    For static boxes with a manually supplied ``L_inf``, the ambient
    translational add-back ``U^inf`` is also applied.  For live sheared boxes in
    fractional coordinates, the changing box basis already carries the affine
    motion, so applying ``U^inf`` to the coordinates would double-count the
    relative affine shear.  ``S5`` is the total stresslet ``(N,5)``, including
    the drift stresslet ``(kT/eps)(S5_+ - S5_-)`` from the RFD displaced
    solves -- a single-sample estimator of the mean Brownian stresslet
    ``<S^B> = -kT div(R_SU . R_FU^{-1})`` (Foss & Brady 2000, Eq. 10c; also
    exposed as ``info['S5_drift']``).  Per-step values are noisy; only time
    averages are physically meaningful (as for the fluctuating stresslet).
  """
  # The whole step is jitted end-to-end, so the saddle solve must use an
  # on-device preconditioner.  ``'ic0'`` builds a host RCM + incomplete-Cholesky
  # factor from the (traced) state arrays inside the jitted solve and would raise
  # a TracerArrayConversionError on the first step -- reject it here rather than
  # let it leak through ``**rpy_kwargs`` into ``build_saddle_solve``.
  if rpy_kwargs.get('preconditioner') == 'ic0':
    raise ValueError(
        "build_sd_brownian_step is jitted end-to-end; the host 'ic0' "
        "preconditioner needs a pure_callback and cannot run under jit. Use "
        "'cheb' (default), 'diag', or 'jacobi'.")

  # Precision-aware tolerance floors: float32 GMRES/Lanczos residuals bottom out
  # around ~1e-6, so a tighter target is unreachable and only burns iterations.
  _f32 = REAL_DTYPE == jnp.float32
  _tol_floor = 1e-5 if _f32 else 0.0
  rfd_atol = max(float(rfd_atol), _tol_floor)
  lanczos_tol = max(float(lanczos_tol), _tol_floor)
  gmres_tol = max(float(gmres_tol), _tol_floor)
  # Resolve eps inside [floor, ceiling]; see the block comment above
  # ``_RFD_CEILING_FRAC``.
  rfd_epsilon, _rfd_floor, _rfd_ceiling, _rfd_clamp_gap = _resolve_rfd_epsilon(
      rfd_epsilon, gmres_tol, a)
  # Bound the RFD GMRES budget in BOTH precisions, independently of the main
  # solve's (larger) budget.  restart=50 is the original FSD restart length;
  # two cycles bring a cheb-preconditioned solve to true rel residual ~1e-4,
  # i.e. drift error ~0.5% via err ~ residual / (|U_+ - U_-|/|U_+| ~ 0.02) at
  # N <= 4000, phi = 0.45 near contact, in both precisions.  One cycle is NOT
  # enough (drift error 0.1-0.4); the main solve's budget reaches ~1e-6, three
  # decades tighter than this single-sample estimator can use, at twice the
  # iterations.  These solves are truncated by design and therefore pass
  # ``return_residual=False`` (no convergence warning).
  if rfd_gmres_restart is None:
    rfd_gmres_restart = 50
  if rfd_gmres_maxiter is None:
    rfd_gmres_maxiter = 2
  if rfd_gmres_solve_method not in ('batched', 'incremental'):
    raise ValueError(
        "rfd_gmres_solve_method must be 'batched' or 'incremental', got %r"
        % (rfd_gmres_solve_method,))

  init_fn, solve_fn = build_saddle_solve(
      space_fns, a, eta,
      xi=xi, n_particles=n_particles, phi=phi,
      r_lub=r_lub, r_p=r_p, gmres_tol=gmres_tol,
      **rpy_kwargs)
  shift_fn = space_fns[1]
  # The RFD displaced solves reuse the neighbor lists built at ``q`` (no rebuild
  # at q +/- (eps/2) dq) -- valid only while the displacement stays well inside
  # the neighbor-list skin, otherwise a pair entering the cutoff at the displaced
  # config would be silently omitted from the drift.  The realized displacement
  # is ``~rfd_epsilon`` (dq is unit-Gaussian), and the default skin is a fixed
  # fraction (~0.1) of the near-field cutoff ``r_lub``; guard against a gross
  # misconfiguration where ``rfd_epsilon`` is not negligible against that scale.
  if rfd_epsilon >= 0.01 * solve_fn.r_lub:
    raise ValueError(
        "rfd_epsilon=%g is too large relative to the near-field cutoff "
        "r_lub=%g: the RFD displaced solves reuse the neighbor list built at q, "
        "which is only valid for displacements far inside the skin. Use "
        "rfd_epsilon << 0.01*r_lub (default max(gmres_tol, 1e-4))."
        % (rfd_epsilon, solve_fn.r_lub))
  # Config-independent grand Jacobi preconditioner -- built once here (eagerly):
  # it calls the jitted ``Mr_self``, so constructing it inside ``_step_core``
  # would raise a ConcretizationTypeError on its ``float(...)``.
  slip_precond = grand_jacobi_preconditioner(
      solve_fn.a, solve_fn.xi, solve_fn.eta)
  # The wave-space sqrt sampler bakes in the (static-box) grid stencil support /
  # mode arrays, which must be concrete inside the jitted step.  It depends only
  # on the box + grid (not positions or N), so it is built once from the first
  # state and reused for every step.
  _wave_cache = {}

  has_box_fn = bool(getattr(solve_fn, 'has_box_fn', False))
  fractional_coordinates = bool(
      getattr(solve_fn, 'fractional_coordinates', True))
  advect_ambient = (not has_box_fn) or (not fractional_coordinates)

  @partial(jax.jit, static_argnums=(9,))
  def _step_core(state, q, key, force, torque, E_inf, L_inf, x0, shear_kwargs,
                 wave_sqrt):
    # Live deformed box from the shear gammas (None for a static box).  Under
    # shear, re-bind the incoming state's neighbor lists + box_matrix to THIS
    # step's box so the real-space Lanczos (reads ``rpy.real.box_matrix``) and
    # near-field samplers (read ``nf.box_matrix``) are box-consistent with the
    # wave-space exact path and the solve; cheap shape-preserving ``.update()``.
    current_box = solve_fn.resolve_current_box(q, **shear_kwargs)
    if current_box is not None:
      state = solve_fn.refresh_state(state, q, **shear_kwargs)

    # One near-field prepare per step, shared by the two fixed-configuration
    # consumers below (the near-field sampler and the main solve) -- the
    # state is already refreshed to this step's box, so the blocks match what
    # each consumer would have built itself.  The two RFD solves displace the
    # positions and correctly re-prepare internally (which is what lets the
    # drift stresslet estimator see the near-field resistance gradients).
    prepared_nf = solve_fn.nf_apply.prepare(state.nf, q)

    # Clean key tree: the three random inputs must be independent (FD theorem,
    # positive split).  far_field_slip splits k_slip -> (real, wave) internally.
    k_slip, k_nf, k_rfd = jax.random.split(key, 3)

    # (1) far-field slip; (2) near-field Brownian force.  Both samplers are
    # fully on-device (real-space Lanczos + analytic wave-space square root;
    # on-device Jacobi-preconditioned near-field Lanczos) -- no host IC(0), no
    # pure_callback -- so the whole step compiles to one XLA program.
    slip_sampler = make_far_field_slip_sampler(
        solve_fn, state, q, kT, dt, mr_iters=mr_iters,
        lanczos_tol=lanczos_tol, precond=slip_precond, wave_sqrt=wave_sqrt,
        current_box=current_box)
    nf_sampler = make_nearfield_brownian_sampler(
        solve_fn, state, q, kT, dt, preconditioned=True,
        iters=nf_iters, tol=lanczos_tol, prepared=prepared_nf)
    U_B_flat = slip_sampler(k_slip)
    F_B_nf = nf_sampler(k_nf)

    # (3) one combined deterministic + Brownian saddle solve.  ``x0`` is the
    # previous step's solution (the warm start the original FSD code uses):
    # positions move O(U dt) per step, so the smooth (deterministic) part of
    # the solution is an excellent initial guess; the fresh Brownian part of
    # the RHS is independent each step, so at worst the initial residual is
    # ~sqrt(2) of the cold start's -- a fraction of one GMRES iteration.
    U_main, Om_main, S5, q11_sol, info = solve_fn(
        state, q, force=force, torque=torque, E_inf=E_inf, L_inf=L_inf,
        slip_top=U_B_flat, extra_force=F_B_nf, x0=x0,
        prepared_nf=prepared_nf,
        return_stresslet=return_stresslet, return_residual=return_residual,
        **shear_kwargs)
    S5_main = S5

    # (4) RFD thermal drift (reuses state at q; absolute-tol displaced solves).
    # With return_stresslet, the SAME two displaced solves also return the
    # drift stresslet S5_drift = (kT/eps)(S5+ - S5-): a single-sample estimator
    # of the full mean Brownian stresslet <S^B> = -kT div(R_SU . R_FU^{-1})
    # (Foss & Brady 2000, Eq. 10c).  The product rule splits Eq. 10c into the
    # far-field response divergence, R^nf_SU . div(R_FU^{-1}), and
    # (grad R^nf_SU) : R_FU^{-1}; the last two diverge like 1/gap near contact
    # and cancel, and the +/- subtraction performs that cancellation exactly.
    # This deliberately deviates from the original FSD code, which keeps only
    # the coupling term -R^nf_SU U_drift and so leaves the un-cancelled 1/gap
    # piece in near-contact pair stress.  S5_drift already CONTAINS that
    # coupling term: no -R^nf_SU U_drift add-back may ever be applied on top
    # of it (double counting).
    if return_stresslet:
      U_drift6, S5_drift = rfd_drift(
          solve_fn, state, q, k_rfd,
          eps=rfd_epsilon, kT=kT, shift_fn=shift_fn, atol=rfd_atol,
          step_kwargs=shear_kwargs,
          gmres_restart=rfd_gmres_restart, gmres_maxiter=rfd_gmres_maxiter,
          gmres_solve_method=rfd_gmres_solve_method,
          return_stresslet=True)
      S5 = S5 + S5_drift
    else:
      U_drift6 = rfd_drift(
          solve_fn, state, q, k_rfd,
          eps=rfd_epsilon, kT=kT, shift_fn=shift_fn, atol=rfd_atol,
          step_kwargs=shear_kwargs,
          gmres_restart=rfd_gmres_restart, gmres_maxiter=rfd_gmres_maxiter,
          gmres_solve_method=rfd_gmres_solve_method)

    # Advance with the deterministic+Brownian relative velocity.  In a live
    # fractional sheared box, H(t) already carries the affine translational
    # motion; adding ``U_inf = L^inf . r`` here would double the relative shear.
    # Static-box/manual-flow callers still need the ambient add-back.
    U_total6 = jnp.concatenate([U_main, Om_main], axis=-1) + U_drift6
    U_advect = _sd_coordinate_velocity(
        U_main, U_drift6, info['U_inf'], advect_ambient)
    q_new = shift_fn(q, dt * U_advect, **shear_kwargs)

    # Refresh the neighbor lists to q_new *inside* the jitted step (fused,
    # on-device, shape-preserving) and hand the next state back in ``info`` --
    # threading this avoids an eager host ``refresh_state`` between steps, whose
    # per-step neighbor-list rebuilds are catastrophically slow on GPU.
    next_state = solve_fn.refresh_state(state, q_new, **shear_kwargs)

    out_info = dict(info)
    out_info['U_drift'] = U_drift6
    if return_stresslet:
      out_info['S5_main'] = S5_main
      out_info['S5_drift'] = S5_drift
    out_info['next_state'] = next_state
    # Warm start for the NEXT step's main solve (moments + relative velocity;
    # the drift is excluded -- it is not part of the saddle solution).
    out_info['x0'] = (q11_sol, jnp.concatenate([U_main, Om_main], axis=-1))
    return q_new, S5, out_info

  def step_fn(state, positions_frac, key, *,
              force=None, torque=None, E_inf=None, L_inf=None, x0=None,
              **shear_kwargs):
    q = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    N = q.shape[0]
    # Resolve optional inputs to concrete arrays eagerly (None is untraceable);
    # a zero (3,3) E_inf is the no-imposed-strain case solve_fn expects.
    force = (jnp.zeros((N, 3), dtype=REAL_DTYPE) if force is None
             else jnp.asarray(force, dtype=REAL_DTYPE))
    torque = (jnp.zeros((N, 3), dtype=REAL_DTYPE) if torque is None
              else jnp.asarray(torque, dtype=REAL_DTYPE))
    E_inf = (jnp.zeros((3, 3), dtype=REAL_DTYPE) if E_inf is None
             else jnp.asarray(E_inf, dtype=REAL_DTYPE))
    # L_inf (full ambient gradient for the add-back) stays None unless supplied
    # -> solve_fn defaults it to the symmetric part of E_inf (no ambient spin).
    L_inf = None if L_inf is None else jnp.asarray(L_inf, dtype=REAL_DTYPE)
    # Warm start (previous step's ``info['x0']``); None -> cold (zero) start.
    if x0 is None:
      x0 = (jnp.zeros((N, 11), dtype=REAL_DTYPE),
            jnp.zeros((N, 6), dtype=REAL_DTYPE))
    else:
      x0 = (jnp.asarray(x0[0], dtype=REAL_DTYPE),
            jnp.asarray(x0[1], dtype=REAL_DTYPE))
    # Build the wave sqrt sampler once from the concrete wave state (static box)
    # and reuse it as a static arg so the jitted step compiles a single program.
    # Unused on the live-shear path (the exact deformed-box noise is rebuilt per
    # step) but always passed so the jitted signature is stable.
    wave_sqrt = _wave_cache.get('wave_sqrt')
    if wave_sqrt is None:
      wave_sqrt = build_Mw_grand_sqrt_sampler(state.rpy.wave)
      _wave_cache['wave_sqrt'] = wave_sqrt
    return _step_core(state, q, key, force, torque, E_inf, L_inf, x0,
                      shear_kwargs, wave_sqrt)

  # Cheap host neighbor-list rebuild that reuses the wave state -- thread this
  # between steps instead of calling ``init_fn`` again (which rebuilds the wave
  # ``PjitFunction``s and forces the jitted step to recompile every step).
  step_fn.refresh_state = solve_fn.refresh_state
  # RFD step provenance for run metadata: resolved from a mix of arguments,
  # dtype and an import-time environment variable.
  step_fn.rfd_epsilon = rfd_epsilon
  step_fn.rfd_epsilon_floor = _rfd_floor
  step_fn.rfd_epsilon_ceiling = _rfd_ceiling
  step_fn.lubrication_clamp_gap = _rfd_clamp_gap
  # Surface gap (units of ``a``) above which the drift is resolved; reported
  # rather than warned about, since with no clamp it is unavoidable.
  step_fn.rfd_trust_gap = rfd_epsilon / (_RFD_CEILING_FRAC * float(a))

  return init_fn, step_fn
