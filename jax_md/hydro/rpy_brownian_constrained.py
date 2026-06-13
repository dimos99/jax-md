"""Constrained Brownian dynamics via the midpoint SDAE integrator.

Implements the Fiore & Swan (2018) midpoint scheme for the stresslet-
constrained mobility: sample the *unconstrained* grand Brownian slip with the
positively split sampler (real-space Lanczos square root + wave-space
analytic square root), advance to a midpoint with the slip velocity, and
impose the rigidity constraint with a single stresslet-constraint solve at the
midpoint.
The Taylor expansion of the step reproduces, to O(dt), both the constrained
covariance ``2 kT dt R_FU^{-1}`` and the thermal drift
``kT div R_FU^{-1}`` -- no square root of the constrained operator and no
random finite differences are ever formed.

Flat moment coordinates
-----------------------
All Lanczos vectors are (N, 11) arrays in the *orthonormal* flat layout of
``rpy_moments.grand_to_flat``: channels 0:3 Cartesian force/velocity,
channels 3:11 coordinates in ``traceless_orthonormal_basis()`` order
(3 antisymmetric, 3 symmetric off-diagonal, 2 diagonal traceless).  The
grand mobility is a symmetric matrix in these coordinates -- a Lanczos
prerequisite -- and is *not* symmetric in the drop-zz grid packing.

Unit conventions
----------------
The slip produced by ``make_grand_slip_sampler`` is a *velocity*: its
covariance is ``(2 kT / dt) M_grand`` so that ``x_{k+1} = x_k + dt * U``
carries the displacement covariance ``2 kT dt M``.  This differs from the
unconstrained RPY Brownian path, whose ``noise`` output is a *displacement*
with covariance ``2 kT dt M`` (scale ``sqrt(2 kT dt)``).
"""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_md import dataclasses
from jax_md.hydro.rpy_constrained import make_constrained_solver
from jax_md.hydro.rpy_real_det import REAL_DTYPE, RealSpaceState
from jax_md.hydro.rpy_real_det_helpers import Mr_self
from jax_md.hydro.rpy_real_det_dipole import mr_grand_matvec
from jax_md.hydro.rpy_real_det_dipole_helpers import Mr_self_dipole
from jax_md.hydro.rpy_real_stoch import (
    Preconditioner,
    diagonal_preconditioner,
    lanczos_sqrt_mv,
)
from jax_md.hydro.rpy_moments import (
    flat_to_grand,
    grand_to_flat,
    torque_to_couplet,
    traceless,
)


def grand_jacobi_preconditioner(a: float, xi: float, eta: float
                                ) -> Preconditioner:
  """Per-channel Jacobi preconditioner for the flat (N, 11) grand operator.

  ``G = diag(1/sqrt(d))`` built from the exact self block of M^(r)_grand in
  orthonormal flat coordinates.  The UF self block is
  ``Mr_self(a, xi) / (6 pi eta a) * I3``.  The DC self block is
  ``Mr_self_dipole(a, xi) / (6 pi eta) * (C^T - 4 C)``, which is diagonal in
  the orthonormal basis: antisymmetric members (B^T = -B) pick up the
  eigenvalue ``-5 * m_d`` and symmetric members (B^T = B) the eigenvalue
  ``-3 * m_d`` (``m_d = Mr_self_dipole / (6 pi eta)`` is negative, so both
  diagonals are positive).  This fixes the ~a^{-2} scale mismatch between
  the force and dipole channels that would otherwise inflate the Lanczos
  iteration count.
  """
  m_uf = float(Mr_self(a, xi)) / (6.0 * np.pi * eta * a)
  m_d = float(Mr_self_dipole(a, xi)) / (6.0 * np.pi * eta)
  d11 = np.empty((11,), dtype=np.float64)
  d11[0:3] = m_uf
  d11[3:6] = -5.0 * m_d
  d11[6:11] = -3.0 * m_d
  if np.any(d11 <= 0.0):
    raise ValueError(
        f'grand self mobilities must be positive, got {d11}; '
        'check a/xi/eta.')
  return diagonal_preconditioner(
      jnp.asarray(1.0 / np.sqrt(d11), dtype=REAL_DTYPE))


def make_real_grand_slip_sampler(
    *,
    real_state: RealSpaceState,
    positions: jnp.ndarray,
    preconditioner: Optional[Preconditioner] = None,
    iters: int = 50,
    tol: float = 1e-3,
) -> Callable[[jax.Array], Tuple[jnp.ndarray, jnp.ndarray, dict]]:
  """Sampler for ``M^(r)_grand^{1/2} dW`` at a fixed configuration.

  Wraps the preconditioned Lanczos square root (Chow & Saad) around the
  real-space grand matvec in flat orthonormal coordinates.  ``real_state``
  must already be refreshed at ``positions``.

  Returns ``sampler(key) -> (U_half (N, 3), D_half (N, 3, 3), info)`` where
  the pair has covariance ``M^(r)_grand`` and ``info`` carries the Lanczos
  diagnostics ``rel_change`` / ``iters`` / ``converged``.
  """
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  if preconditioner is None:
    raise ValueError('make_real_grand_slip_sampler requires a '
                     'preconditioner; use grand_jacobi_preconditioner.')
  pc = preconditioner

  def mv(mv_params, x):
    st, pos = mv_params
    F, C = flat_to_grand(pc.apply_T(x))
    U, D = mr_grand_matvec(st, pos, F, C)
    return pc.apply(grand_to_flat(U, D))

  def sampler(key: jax.Array):
    noise = jax.random.normal(
        key, shape=positions.shape[:-1] + (11,), dtype=REAL_DTYPE)
    approx, rel_change, iters_used, converged = lanczos_sqrt_mv(
        mv, (real_state, positions), noise,
        iters=iters, tol=tol, return_info=True)
    U_half, D_half = flat_to_grand(pc.solve(approx))
    info = {
        'real_sqrt_rel_change': rel_change,
        'real_sqrt_iters': iters_used,
        'real_sqrt_converged': converged,
    }
    return U_half, traceless(D_half), info

  return sampler


def make_grand_slip_sampler(
    *,
    real_sampler: Callable,
    wave_sampler: Callable,
    kT: float,
    dt: float,
) -> Callable[[jax.Array], Tuple[jnp.ndarray, jnp.ndarray, dict]]:
  """Assemble the positively-split unconstrained Brownian slip.

  ``real_sampler(key) -> (U, D, info)`` and ``wave_sampler(key) -> (U, D)``
  must be bound at the same configuration; their outputs are statistically
  independent samples with covariances M^(r) and M^(w), so the sum has the
  unconstrained grand covariance M = M^(r) + M^(w) (positive split).

  Returns ``sampler(key) -> (U_B (N, 3), D_B (N, 3, 3), info)`` with joint
  covariance ``(2 kT / dt) * M_grand`` -- slip *velocity* units, so that the
  corrector ``x + dt * U`` carries displacement covariance ``2 kT dt M``
  (unlike the unconstrained noise output, which is a displacement scaled by
  ``sqrt(2 kT dt)``).
  """

  def sampler(key: jax.Array):
    key_real, key_wave = jax.random.split(key)
    U_r, D_r, info = real_sampler(key_real)
    U_w, D_w = wave_sampler(key_wave)
    scale = jnp.sqrt(jnp.asarray(2.0 * kT / dt, dtype=REAL_DTYPE))
    U_B = scale * (U_r + U_w)
    D_B = scale * traceless(D_r + D_w)
    return U_B, D_B, info

  return sampler


# -----------------------------------------------------------------------------
# Midpoint SDAE integrator
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class ConstrainedBrownianState:
  """Stepper state for the constrained Brownian midpoint integrator.

  ``rpy_state`` is the mobility's own state pytree (``rpy.RpyState``:
  neighbor list, wave arrays, static preconditioner).  ``stresslet`` is the
  previous midpoint solve's (N, 5) orthonormal stresslet, used to warm-start
  the next GMRES solve.  All array leaves make the state ``lax.scan``
  compatible.
  """
  positions: jnp.ndarray   # fractional (N, 3)
  rpy_state: Any           # RpyState pytree (real + wave + preconditioner)
  stresslet: jnp.ndarray   # (N, 5) orthonormal warm start
  rng: jax.Array           # PRNG key, split once per step
  step: jnp.ndarray        # int32 step counter


def make_constrained_brownian_step(
    *,
    # Configuration-binding callables (supplied by build_rpy_mobility's shim):
    mobility_init_fn: Callable,     # (positions, *, extra_capacity_override=None, **kw) -> RpyState
    mr_apply: Callable,             # (real_state, pos, F, C, **kw) -> ((U, D), real_state); has .refresh
    wave_apply_grand: Callable,     # (wave_state, pos, F, C, current_box) -> (U, D, wave_state)
    wave_matvec: Callable,          # (wave_state, pos, F, C, current_box) -> (U, D), fixed state
    wave_noise: Callable,           # (wave_state, pos, key, current_box) -> (U_half, D_half)
    normalize_kwargs: Callable,     # (kwargs, dim) -> combined mobility kwargs
    resolve_current_box: Callable,  # (dim, combined_kwargs) -> Optional box
    grand_precond: Preconditioner,
    real_space_first: bool,
    shift_fn: Callable,
    # Physics / integrator parameters:
    force_fn: Optional[Callable],   # (positions_frac, **step_kwargs) -> (N, 3); None -> no forces
    kT: float,
    dt: float,
    integrator: str = 'midpoint',
    torque_fn: Optional[Callable] = None,
    with_torque: bool = False,
    mr_iters: int = 50,
    lanczos_tol: float = 1e-3,
    solve_tol: float = 1e-3,
    solve_maxiter: int = 50,
) -> Tuple[Callable, Callable]:
  """Build the constrained Brownian stepper (Fiore & Swan midpoint SDAE).

  Per step: (1) sample the unconstrained grand slip ``[U_B, D_B]`` at the
  current configuration ``x_k`` (covariance ``2 kT/dt M_grand``); (2) move
  to the midpoint ``x_mid = x_k + (dt/2) U_B``; (3) perform the *single*
  stresslet-constraint solve at ``x_mid`` with the slip folded into the
  applied output (RHS ``-(E_B + M_EF F)``, assembly
  ``U_B + M_UF F + M_US S``); (4) advance ``x_{k+1} = x_k + dt U_mid``.
  The midpoint correlation between slip and solve reproduces the thermal
  drift ``kT div R_FU^{-1}`` to O(dt) -- no divergence is ever computed.

  ``integrator='euler_maruyama'`` collapses the solve to ``x_k`` (no
  midpoint, no drift) -- deliberately wrong equilibrium, kept only to
  reproduce the paper's drift-omission demonstrations in validation.

  Forces ``F^P = force_fn(x_k)`` are evaluated at the start-of-step
  configuration (Fiore & Swan ordering) and applied through the midpoint
  mobility.

  Returns ``(brownian_init_fn, step_fn)``:
    brownian_init_fn(positions_frac, key, *, stresslet_guess=None,
                     extra_capacity_override=None, wave_state=None, **kwargs)
        -> ConstrainedBrownianState
    step_fn(state, **step_kwargs) -> (next_state, info)
  ``info`` carries the Lanczos diagnostics and ``nbr_did_overflow``; on
  overflow the step's results are invalid and the chunk must be replayed
  after re-allocation (see ``run_brownian_chunked``).
  """
  if integrator not in ('midpoint', 'euler_maruyama'):
    raise ValueError(f"integrator must be 'midpoint' or 'euler_maruyama', "
                     f"got {integrator!r}.")
  if torque_fn is not None and not with_torque:
    raise ValueError('torque_fn requires with_torque=True.')
  use_midpoint = integrator == 'midpoint'

  def brownian_init_fn(positions_frac, key, *, stresslet_guess=None,
                       extra_capacity_override=None, wave_state=None,
                       **kwargs):
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    n = positions_frac.shape[0]
    init_kwargs = {}
    if wave_state is not None:
      init_kwargs['wave_state'] = wave_state
    rpy_state = mobility_init_fn(
        positions_frac, extra_capacity_override=extra_capacity_override,
        **init_kwargs, **kwargs)
    if stresslet_guess is None:
      stresslet_guess = jnp.zeros((n, 5), dtype=REAL_DTYPE)
    return ConstrainedBrownianState(
        positions=positions_frac,
        rpy_state=rpy_state,
        stresslet=jnp.asarray(stresslet_guess, dtype=REAL_DTYPE),
        rng=key,
        step=jnp.zeros((), dtype=jnp.int32),
    )

  def step_fn(state: ConstrainedBrownianState, **step_kwargs):
    x_k = state.positions
    dim = int(x_k.shape[1])
    n = x_k.shape[0]
    combined_kwargs = normalize_kwargs(step_kwargs, dim)
    current_box = resolve_current_box(dim, combined_kwargs)

    key_next, key_slip = jax.random.split(state.rng)

    # Forces and applied couplets at x_k (Fiore & Swan ordering).
    if force_fn is not None:
      F_P = jnp.asarray(force_fn(x_k, **step_kwargs), dtype=REAL_DTYPE)
    else:
      F_P = jnp.zeros((n, 3), dtype=REAL_DTYPE)
    if with_torque:
      if torque_fn is not None:
        torques = jnp.asarray(torque_fn(x_k, **step_kwargs), dtype=REAL_DTYPE)
      else:
        torques = jnp.zeros((n, 3), dtype=REAL_DTYPE)
      C_applied = torque_to_couplet(torques)
    else:
      torques = None
      C_applied = jnp.zeros((n, 3, 3), dtype=REAL_DTYPE)

    # Step 1: bind at x_k (bookkeeping only) and sample the slip.
    real_k = mr_apply.refresh(state.rpy_state.real, x_k, **combined_kwargs)
    slip_sampler = make_grand_slip_sampler(
        real_sampler=make_real_grand_slip_sampler(
            real_state=real_k,
            positions=x_k,
            preconditioner=grand_precond,
            iters=mr_iters,
            tol=lanczos_tol,
        ),
        wave_sampler=lambda key: wave_noise(
            state.rpy_state.wave, x_k, key, current_box),
        kT=kT,
        dt=dt,
    )
    U_B, D_B, slip_info = slip_sampler(key_slip)

    # Step 2: predictor to the midpoint (EM solves at x_k instead).
    if use_midpoint:
      x_solve = shift_fn(x_k, (0.5 * dt) * U_B, **step_kwargs)
    else:
      x_solve = x_k

    # Step 3: single constraint solve at x_solve with the slip folded in.
    (Ur, Dr), real_s = mr_apply(real_k, x_solve, F_P, C_applied,
                                **combined_kwargs)
    Uw, Dw, wave_s = wave_apply_grand(
        state.rpy_state.wave, x_solve, F_P, C_applied, current_box)
    if real_space_first:
      U_app, D_app = Ur + Uw, traceless(Dr + Dw)
    else:
      U_app, D_app = Uw + Ur, traceless(Dw + Dr)

    def grand_mv(F, C):
      Ur_, Dr_ = mr_grand_matvec(real_s, x_solve, F, C)
      Uw_, Dw_ = wave_matvec(wave_s, x_solve, F, C, current_box)
      if real_space_first:
        return Ur_ + Uw_, traceless(Dr_ + Dw_)
      return Uw_ + Ur_, traceless(Dw_ + Dr_)

    solve_fn = make_constrained_solver(
        grand_mv,
        with_torque=with_torque,
        solve_tol=solve_tol,
        solve_maxiter=solve_maxiter,
    )
    U_mid, S5, Omega, _ = solve_fn(
        F_P,
        torques=torques,
        stresslet_guess=state.stresslet,
        applied_output=(U_app + U_B, traceless(D_app + D_B)),
    )

    # Step 4: corrector from the original configuration.
    x_new = shift_fn(x_k, dt * U_mid, **step_kwargs)

    next_rpy = dataclasses.replace(state.rpy_state, real=real_s, wave=wave_s)
    overflow = (real_s.neighbors.did_buffer_overflow |
                real_k.neighbors.did_buffer_overflow)
    info = dict(slip_info)
    info['nbr_did_overflow'] = overflow
    if with_torque:
      info['angular_velocities'] = Omega

    next_state = ConstrainedBrownianState(
        positions=x_new,
        rpy_state=next_rpy,
        stresslet=S5,
        rng=key_next,
        step=state.step + 1,
    )
    return next_state, info

  return brownian_init_fn, step_fn


def run_brownian_chunked(
    step_fn: Callable,
    brownian_init_fn: Callable,
    state: ConstrainedBrownianState,
    n_steps: int,
    *,
    chunk_size: int = 100,
    max_retries: int = 3,
    observe_fn: Optional[Callable] = None,
    **step_kwargs,
) -> Tuple[ConstrainedBrownianState, list]:
  """Drive the stepper in jitted chunks with neighbor-overflow replay.

  Neighbor reallocation cannot happen inside ``lax.scan``, so each chunk of
  ``chunk_size`` steps is scanned under jit and the overflow flags reduced.
  On overflow the whole chunk is discarded (post-overflow steps are
  invalid), the mobility is re-allocated at the chunk-start snapshot with
  escalating ``extra_capacity``, and the chunk is replayed -- replay is
  bit-deterministic because the PRNG key lives in the state.  Raises
  ``RuntimeError`` after ``max_retries`` consecutive failures of one chunk.

  ``observe_fn(state) -> pytree``, if given, is applied after every step;
  the per-chunk stacked observations are collected in the returned list
  (one entry per chunk, each with leading axis ``chunk_size``).
  """
  @partial(jax.jit, static_argnames='length')
  def run_chunk(state0, length):
    def body(s, _):
      s, info = step_fn(s, **step_kwargs)
      obs = observe_fn(s) if observe_fn is not None else 0
      return s, (info['nbr_did_overflow'], obs)
    s_final, (overflow, obs) = lax.scan(body, state0, xs=None, length=length)
    return s_final, jnp.any(overflow), obs

  observations = []
  chunk_plan = [chunk_size] * (n_steps // chunk_size)
  if n_steps % chunk_size:
    chunk_plan.append(n_steps % chunk_size)
  base_extra = 0
  for length in chunk_plan:
    snapshot = state
    retries = 0
    while True:
      state, overflowed, obs = run_chunk(snapshot, length)
      if not bool(overflowed):
        break
      retries += 1
      if retries > max_retries:
        raise RuntimeError(
            f'neighbor list overflowed {max_retries} consecutive times at '
            f'step {int(snapshot.step)}; increase capacity_multiplier.')
      base_extra = max(2 * base_extra, 64)
      fresh = brownian_init_fn(
          snapshot.positions,
          snapshot.rng,
          stresslet_guess=snapshot.stresslet,
          extra_capacity_override=base_extra,
          **step_kwargs,
      )
      snapshot = dataclasses.replace(fresh, step=snapshot.step)
    if observe_fn is not None:
      observations.append(jax.device_get(obs))
  return state, observations
