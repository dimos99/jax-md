"""
Spectral-Ewald RPY mobility implementation.

This module provides functions to build and apply the Rotne–Prager–Yamakawa
(RPY) mobility operator using a split-Ewald formulation. It combines real-space
and wave-space contributions for efficient hydrodynamic interactions in
periodic systems. It supports deterministic mobility application and
stochastic Brownian velocity sampling.
"""
from typing import Callable, Optional, Tuple
import math
import warnings

import jax
import jax.numpy as jnp
from jax import config as jax_config

from jax_md import dataclasses, space
from jax_md.hydro.rpy_wave import (
    WaveSpaceState,
    build_Mw_state,
    choose_theta,
)
from jax_md.hydro.rpy_real import (
    RealSpaceState,
    build_Mr_apply,
    sample_mr_sqrt_precond,
    current_box_matrix,
    jacobi_from_self,
    identity_preconditioner,
    Preconditioner,
)

XI_OPT_A = 0.5  # default target for xi * a in Fiore (2017)
REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32


def _quadrature_error_bound(P: int) -> Tuple[float, float]:
  """Quadrature error bound ε_q and Gaussian width m (Fiore 2017, Sec. II.C).

  Uses the spectral Ewald choice m = sqrt(pi P), consistent with the
  theta-selection in `choose_theta`.
  """
  P = int(P)
  m = math.sqrt(math.pi * float(P))
  term1 = math.exp(-0.5 * math.pi * float(P))
  term2 = math.erfc(m / math.sqrt(2.0))
  return term1 + term2, m


def _epsilon_k_bound(kcut: float, xi: float, a: float) -> float:
  """Wave-space truncation bound ε_k from Fiore 2017, Eq. (28)."""
  kcut = float(kcut)
  xi = float(xi)
  a = float(a)
  if kcut <= 0.0 or xi <= 0.0 or a <= 0.0:
    return float('inf')
  exp_term = math.exp(-(kcut * kcut) / (4.0 * xi * xi))
  erfc_term = math.erfc(kcut / (2.0 * xi))
  num = 4.0 * exp_term / kcut - math.sqrt(math.pi) * erfc_term / xi
  eps = num / (2.0 * math.pi * a)
  return max(0.0, eps)


def _select_kcut(xi: float, a: float, eps_w: float) -> float:
  """Choose k_cut so that ε_k <= eps_w using Eq. (28) bound."""
  eps_w = float(max(eps_w, 1e-16))
  # Start with the Hasimoto truncation bound: eps ~ exp(-k^2/4xi^2)
  k_guess = 2.0 * float(xi) * math.sqrt(max(1e-16, math.log(1.0 / eps_w)))
  k_high = max(k_guess, 1e-6)
  if _epsilon_k_bound(k_high, xi, a) <= eps_w:
    return k_high
  # Increase until bound satisfied.
  for _ in range(40):
    k_high *= 1.5
    if _epsilon_k_bound(k_high, xi, a) <= eps_w:
      break
  # Binary search for a tight bound.
  k_low = 0.0
  for _ in range(60):
    k_mid = 0.5 * (k_low + k_high)
    if _epsilon_k_bound(k_mid, xi, a) > eps_w:
      k_low = k_mid
    else:
      k_high = k_mid
  return k_high


def _select_P_and_m(eps_q: float) -> Tuple[int, float, float]:
  """Choose the smallest integer P with ε_q <= eps_q."""
  eps_q = float(max(eps_q, 1e-16))
  P = 4
  err, m = _quadrature_error_bound(P)
  while err > eps_q and P < 64:
    P += 1
    err, m = _quadrature_error_bound(P)
  return P, m, err


def estimate_rpy_params(tol: float,
                        A: jnp.ndarray,
                        a: float,
                        N: int,
                        phi: float,
                        *,
                        xi_override: Optional[float] = None,
                        d_f: float = 3.0,
                        n_iter: int = 10,
                        C_R: float = 1.0,
                        C_W: float = 1.0,
                        error_split: Tuple[float, float, float] = (1.0 / 3.0,
                                                                  1.0 / 3.0,
                                                                  1.0 / 3.0),
                        notes: bool = False) -> dict:
  """Fiore (2017) parameter estimator for split-Ewald RPY.

  Uses the paper's decoupled error bounds to choose the real-space cutoff
  (rcut), wave-space cutoff (kcut), and quadrature parameters (P, m), with
  the default spectral Ewald choice m = sqrt(pi P). When xi is not provided,
  the asymptotic optimal xi* from Eq. (23) is used with implementation
  constants set to unity; for dilute/underspecified cases, xi is set by
  xi * a ≈ 0.5 (Fiore 2017, Fig. 4/Table I).

  Parameters
  ----------
  tol : float
      Target relative tolerance for mobility accuracy.
  A : array_like (3,3)
      Periodic cell matrix (real units).
  a : float
      Sphere radius.
  N : int
      Number of particles (used for xi* estimate).
  phi : float
      Volume fraction (used for xi* estimate).
  xi_override : float, optional
      If provided, force xi to this value.
  d_f : float, optional
      Fractal dimension (defaults to 3 for random suspensions).
  n_iter : int, optional
      Lanczos iteration count (used in xi* estimate).
  C_R, C_W : float, optional
      Implementation constants in Eq. (22/23); default to 1.
  error_split : tuple of 3 floats
      Fractions of tol assigned to (eps_R, eps_W, eps_q).
  notes : bool, optional
      If True, include diagnostic fields in the returned dict.

  Returns
  -------
  dict with keys:
      xi, P, M, grid_shape, rcut, kcut, theta, m
      (plus optional 'notes' diagnostic map when requested)
  """
  tol = float(tol)
  A = jnp.asarray(A, dtype=REAL_DTYPE)
  if tol <= 0.0:
    raise ValueError("tol must be positive.")
  if a <= 0.0:
    raise ValueError("a must be positive.")

  L_cols = jnp.linalg.norm(A, axis=0)
  L_min = float(jnp.min(L_cols))
  L_max = float(jnp.max(L_cols))
  L_mean = float(jnp.mean(L_cols))

  split_r, split_w, split_q = error_split
  if split_r <= 0 or split_w <= 0 or split_q <= 0:
    raise ValueError("error_split entries must be positive.")
  split_sum = split_r + split_w + split_q
  if split_sum > 1.0 + 1e-12:
    raise ValueError("error_split entries must sum to <= 1.")

  if xi_override is not None:
    xi = float(xi_override)
  else:
    if phi > 0.0 and N > 1 and d_f > 0.0:
      xi = (3.0 * C_W * math.log(float(N)) /
            (C_R * float(d_f) * float(n_iter) * float(phi)**2)) ** (-1.0 / (float(d_f) + 3.0))
    else:
      xi = float(XI_OPT_A / a)

  if xi <= 0.0 or not math.isfinite(xi):
    raise ValueError(f"xi must be positive; got xi={xi}.")

  eps_r = max(tol * split_r, 1e-16)
  eps_w = max(tol * split_w, 1e-16)
  eps_q = max(tol * split_q, 1e-16)

  rcut_candidate = math.sqrt(math.log(1.0 / eps_r)) / xi
  rcut_guard = 0.49 * L_min
  rcut = min(rcut_candidate, rcut_guard)
  rcut_capped = rcut < rcut_candidate

  P, m, eps_q_est = _select_P_and_m(eps_q)

  kcut = _select_kcut(xi, a, eps_w)

  # FFT grid size from kcut: k_max ≈ π M / L.
  M_est = int(math.ceil(kcut * L_max / math.pi))
  M = max(M_est, P, 8)
  if M % 2 != 0:
    M += 1

  theta = float(choose_theta(P, xi, M, m=m))

  result = {
      'xi': float(xi),
      'P': int(P),
      'M': int(M),
      'grid_shape': (int(M), int(M), int(M)),
      'rcut': float(rcut),
      'kcut': float(kcut),
      'theta': float(theta),
      'm': float(m),
  }

  if notes:
    eps_w_est = _epsilon_k_bound(kcut, xi, a)
    eps_r_est = math.exp(-(xi * rcut) ** 2)
    result['notes'] = {
        'rcut_candidate': float(rcut_candidate),
        'rcut_guard': float(rcut_guard),
        'rcut_capped': bool(rcut_capped),
        'L_min': float(L_min),
        'L_mean': float(L_mean),
        'L_max': float(L_max),
        'eps_r_target': float(eps_r),
        'eps_w_target': float(eps_w),
        'eps_q_target': float(eps_q),
        'eps_r_est': float(eps_r_est),
        'eps_w_est': float(eps_w_est),
        'eps_q_est': float(eps_q_est),
        'N': int(N),
        'phi': float(phi),
        'd_f': float(d_f),
        'n_iter': int(n_iter),
    }

  return result


def _check_xi(xi: float) -> float:
  """Validate xi is positive."""
  xi_float = float(xi)
  if xi_float <= 0.0:
    raise ValueError(
        f"xi must be positive; got xi={xi_float}."
    )
  return xi_float


@dataclasses.dataclass
class RpyState:
  real: RealSpaceState
  wave: WaveSpaceState
  preconditioner: Optional[Preconditioner] = dataclasses.field(
      default=None, metadata={'static': True}
  )


def _base_box_kwargs(dim: int, kwargs: dict) -> dict:
  """Zero out shear entries to recover the base box used for wave modes."""
  base_kwargs = dict(kwargs)
  base_kwargs.pop('gamma', None)
  base_kwargs.pop('gamma_xy', None)
  base_kwargs.pop('gamma_xz', None)
  base_kwargs.pop('gamma_yz', None)
  if dim >= 3:
    base_kwargs['gamma'] = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
  elif dim >= 2:
    base_kwargs['gamma'] = 0.0
  return base_kwargs


def _map_positions_to_base(base_inv: Optional[jnp.ndarray],
                           current_box: Optional[jnp.ndarray],
                           positions_frac: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Map fractional coordinates into the base box used by the wave operator."""
  dim = positions_frac.shape[-1]
  if base_inv is None or current_box is None:
    transform = jnp.eye(dim, dtype=positions_frac.dtype)
    return positions_frac, transform
  transform = base_inv @ current_box
  mapped = jnp.mod(positions_frac @ transform.T, 1.0)
  return mapped, transform


def build_rpy_matvec(space_fns,
                     a: float,
                     xi: float,
                     eta: float,
                     *,
                     Mr_params=None,
                     Mw_params=None,
                     rcut: Optional[float] = None,
                     P: Optional[int] = None,
                     Mgrid: Optional[int] = None,
                     theta: Optional[float] = None,
                     real_space_first: bool = True,
                     include_brownian: bool = True,
                     preconditioner: Optional[Preconditioner] = None):
  """
  Construct matvec apply functions for the split-Ewald RPY mobility.

  - `space_fns` must include displacement/shift and may include `box_fn`; when
    `box_fn` is present, the wave operator is rebuilt on the *current* box each
    call (matches Wang–Brady Eq. 39 / Fiore–Swan deformed-grid treatment).
  - Positions are always used directly in that live frame (no base-frame mapping).
  - Brownian sampling (if enabled) uses the same live-box wave modes; provide
    `brownian_key`, `kT`, and `dt` via `apply_fn` or call `apply_fn.with_brownian`.
  - xi must be positive; values with xi * a ≈ 0.5 are a good starting point.
  """

  xi = _check_xi(xi)

  if len(space_fns) < 2:
    raise ValueError('space_fns must contain displacement and shift functions.')

  displacement_fn, _ = space_fns[:2]
  box_fn = space_fns[2] if len(space_fns) > 2 else None
  has_box_fn = box_fn is not None

  mr_kwargs = dict(Mr_params or {})
  box_kwargs_default = dict(mr_kwargs.pop('box_kwargs', {}))
  rcut_override = mr_kwargs.pop('rcut', None)

  if rcut is not None:
    rcut_value = rcut
  elif rcut_override is not None:
    rcut_value = rcut_override
  else:
    target_epsR = 1e-6
    rcut_value = (1.0 / xi) * jnp.sqrt(jnp.log(1.0 / target_epsR))

  Mr_init, Mr_apply = build_Mr_apply(
      space_fns,
      a,
      xi,
      eta,
      rcut_value,
      **mr_kwargs,
  )

  mw_kwargs = dict(Mw_params or {})
  grid_default = mw_kwargs.pop('M', Mgrid)
  Mx = mw_kwargs.pop('Mx', grid_default)
  My = mw_kwargs.pop('My', grid_default)
  Mz = mw_kwargs.pop('Mz', grid_default)
  P_ = mw_kwargs.pop('P', P)
  theta_ = mw_kwargs.pop('theta', theta)

  if Mx is None or My is None or Mz is None:
    raise ValueError('Specify Mgrid or Mx/My/Mz to size the wave-space grid.')
  if P_ is None:
    P_ = 16

  if preconditioner is None:
    self_coeff = 1.0 / (6.0 * jnp.pi * eta * a)
    precond = jacobi_from_self(self_coeff)
  else:
    precond = preconditioner

  def init_fn(positions_frac, **kwargs):
    positions_frac = jnp.asarray(positions_frac)
    combined_kwargs = dict(box_kwargs_default)
    combined_kwargs.update(kwargs)

    dim = int(positions_frac.shape[1])
    if has_box_fn:
      current_box = current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
      base_kwargs = _base_box_kwargs(dim, combined_kwargs)
      base_box = current_box_matrix(displacement_fn, box_fn, dim, **base_kwargs)
    else:
      base_box = current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
      current_box = base_box

    L_min = float(jnp.min(jnp.linalg.norm(base_box, axis=0)))
    if rcut_value > 0.5 * L_min:
      warnings.warn(
          (
              f"Real-space cutoff rcut={rcut_value:.2f} exceeds half the minimum box "
              f"dimension ({0.5 * L_min:.2f}). Consider increasing xi={xi:.2f} "
              "and compensating with a finer wave-space grid."
          ),
          UserWarning,
          stacklevel=2,
      )

    real_state = Mr_init(positions_frac, **combined_kwargs)

    wave_state = build_Mw_state(
        base_box,
        a,
        xi,
        eta,
        Mx,
        My,
        Mz,
        P_,
        theta=theta_,
        fractional_coordinates=True,
        attach_sqrt=include_brownian,
        attach_fused=False,
    )

    return RpyState(
        real=real_state,
        wave=wave_state,
        preconditioner=precond,
    )

  def apply_fn(state: RpyState,
               positions_frac,
               forces,
               *,
               brownian_key: Optional[jax.Array] = None,
               kT: Optional[float] = None,
               dt: Optional[float] = None,
               mr_iters: int = 10,
               **kwargs):
    positions_frac = jnp.asarray(positions_frac)
    forces = jnp.asarray(forces)

    combined_kwargs = dict(box_kwargs_default)
    combined_kwargs.update(kwargs)

    if has_box_fn:
      dim = int(positions_frac.shape[1])
      current_box = current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
    else:
      current_box = None

    Ur, real_state = Mr_apply(state.real, positions_frac, forces, **combined_kwargs)

    # Reuse the base wave operator; remapping is handled by passing the current box.
    wave_state = state.wave

    if wave_state.apply_fn is None:
      raise ValueError('wave-space operator missing from state; run init_fn first.')

    Uw = wave_state.apply_fn(positions_frac, forces, current_box=current_box)
    velocities = Ur + Uw if real_space_first else Uw + Ur

    next_state = RpyState(
        real=real_state,
        wave=wave_state,
        preconditioner=state.preconditioner,
    )

    if brownian_key is None:
      return velocities, next_state

    if not include_brownian:
      raise ValueError('Brownian sampling was disabled; rebuild with include_brownian=True.')
    if wave_state.sqrt_fn is None:
      raise ValueError('wave-space sampler missing from state; run init_fn first.')
    if kT is None or dt is None:
      raise ValueError('kT and dt are required when requesting Brownian noise.')

    key_real, key_wave = jax.random.split(brownian_key)
    precond_local = state.preconditioner or identity_preconditioner()
    real_noise, rel_change, iters_used, converged = sample_mr_sqrt_precond(
        key_real,
        real_state,
        positions_frac,
        precond=precond_local,
        iters=mr_iters,
        return_info=True,
    )
    wave_noise = wave_state.sqrt_fn(key_wave, positions_frac, current_box=current_box)
    noise = jnp.sqrt(2.0 * kT * dt) * (real_noise + wave_noise)

    info = {
        'real_solver_rel_change': rel_change,
        'real_solver_iters': iters_used,
        'real_solver_converged': converged,
    }
    return velocities, noise, next_state, info

  def apply_with_brownian(state: RpyState,
                          positions_frac,
                          forces,
                          key: jax.Array,
                          *,
                          kT: float,
                          dt: float,
                          mr_iters: int = 10,
                          **kwargs):
    velocities, noise, next_state, info = apply_fn(
        state,
        positions_frac,
        forces,
        brownian_key=key,
        kT=kT,
        dt=dt,
        mr_iters=mr_iters,
        **kwargs,
    )
    return (
        velocities,
        noise,
        next_state,
        info['real_solver_rel_change'],
        info['real_solver_iters'],
        info['real_solver_converged'],
    )

  apply_fn.with_brownian = apply_with_brownian

  return init_fn, apply_fn


def build_rpy_mobility(space_fns,
                       a: float,
                       xi: float,
                       eta: float,
                       **kwargs):
  return build_rpy_matvec(space_fns, a, xi, eta, **kwargs)


def brownian_increment(key: jax.Array,
                       state: RpyState,
                       positions_frac: jnp.ndarray,
                       *,
                       kT: float,
                       dt: float,
                       mr_iters: int = 10) -> jnp.ndarray:
  """Draw the total Brownian increment dB for the provided state and box.

  The wave-space sampler is tied to the base box stored in `state.wave`, so we
  forward `state.real.box_matrix` as the current box when it is available.
  """
  positions_frac = jnp.asarray(positions_frac)
  key_real, key_wave = jax.random.split(key)
  precond = state.preconditioner if state.preconditioner is not None else identity_preconditioner()

  real_sample_real, _, _, _ = sample_mr_sqrt_precond(
      key_real,
      state.real,
      positions_frac,
      precond=precond,
      iters=mr_iters,
      return_info=True,
  )

  current_box = getattr(state.real, 'box_matrix', None)

  if state.wave.sqrt_fn is None:
    raise ValueError('wave-space sampler missing; rebuild with include_brownian=True.')
  wave_sample_real = state.wave.sqrt_fn(
      key_wave,
      positions_frac,
      current_box=current_box,
  )

  scale = jnp.sqrt(2.0 * kT * dt)
  return scale * (real_sample_real + wave_sample_real)
