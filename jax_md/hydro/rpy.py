"""
Spectral-Ewald RPY mobility implementation.

This module provides functions to build and apply the Rotne–Prager–Yamakawa
(RPY) mobility operator using a split-Ewald formulation. It combines real-space
and wave-space contributions for efficient hydrodynamic interactions in
periodic systems. It supports deterministic mobility application and
stochastic Brownian velocity sampling.
"""

from typing import Callable, Dict, Mapping, Optional, Tuple, Union
import itertools
import math
import warnings

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax import config as jax_config

from jax_md import dataclasses, space
from jax_md.hydro.rpy_wave import (
  WaveSpaceState,
  build_Mw_state,
  choose_theta,
  se_alpha,
  make_reciprocal,
  q_grid,
  k_from_q,
  build_P_modes,
  build_B_modes,
  build_stencils_frac,
  spread,
  gather,
  fft_vec,
  ifft_vec,
)
from jax_md.hydro.rpy_real import (
  RealSpaceState,
  build_Mr_apply,
  sample_mr_sqrt_precond,
  Mr_self,
  current_box_matrix,
  jacobi_from_self,
  identity_preconditioner,
  Preconditioner,
)
from jax_md.hydro.rpy_wave_stoch import _hermitian_gaussian_modes

XI_OPT_A = 0.5  # default target for xi * a in Fiore (2017)
REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32
COMPLEX_DTYPE = jnp.complex128 if jax_config.jax_enable_x64 else jnp.complex64


def _shear_planes_for_dim(dim: int) -> Tuple[str, ...]:
  if dim == 2:
    return ('xy',)
  if dim == 3:
    return ('xy', 'xz', 'yz')
  raise ValueError(f"Unsupported dimensionality for shear estimator: dim={dim}.")


def _normalize_shear_schedule(
    shear_schedule: Union[Callable[[float], float], Mapping[str, Callable[[float], float]]],
    dim: int,
) -> Dict[str, Callable[[float], float]]:
  """Normalize estimator shear schedule into a per-plane callable map."""
  planes = set(_shear_planes_for_dim(dim))
  if callable(shear_schedule):
    if 'xy' not in planes:
      raise ValueError("Callable shear schedule is only supported for dimensions with an 'xy' plane.")
    return {'xy': shear_schedule}
  if isinstance(shear_schedule, Mapping):
    fn_map = {}
    for key, fn in shear_schedule.items():
      if key not in planes or fn is None:
        continue
      if not callable(fn):
        raise ValueError(f"shear_schedule['{key}'] must be callable or None.")
      fn_map[str(key)] = fn
    return fn_map
  raise ValueError("shear_schedule must be a callable or a dict mapping shear planes to callables.")


def _deformation_matrix_from_gammas(dim: int, gammas: Mapping[str, float]) -> jnp.ndarray:
  """Build dimensionless deformation tensor F from plane shear strains."""
  F = jnp.eye(dim, dtype=REAL_DTYPE)
  for plane, gamma in gammas.items():
    if plane == 'xy' and dim >= 2:
      F = F.at[0, 1].set(jnp.asarray(gamma, dtype=REAL_DTYPE))
    elif plane == 'xz' and dim >= 3:
      F = F.at[0, 2].set(jnp.asarray(gamma, dtype=REAL_DTYPE))
    elif plane == 'yz' and dim >= 3:
      F = F.at[1, 2].set(jnp.asarray(gamma, dtype=REAL_DTYPE))
  return F


def quadrature_lambda_from_deformation(F: jnp.ndarray) -> float:
  """Return max eigenvalue of F^T F used in deformed-grid quadrature bounds."""
  F = jnp.asarray(F, dtype=REAL_DTYPE)
  if F.ndim != 2 or F.shape[0] != F.shape[1]:
    raise ValueError("Deformation tensor F must be square.")
  lam = float(jnp.max(jnp.linalg.eigvalsh(F.T @ F)))
  if lam <= 0.0 or not math.isfinite(lam):
    raise ValueError(f"Invalid deformation eigenvalue lambda_max={lam}.")
  return lam


def _max_quadrature_lambda_from_shear(
    shear_schedule: Union[Callable[[float], float], Mapping[str, Callable[[float], float]]],
    dim: int,
    *,
    shear_t_bounds: Optional[Tuple[float, float]],
    shear_remap: bool,
) -> float:
  """Compute exact quadrature deformation penalty from shear schedule assumptions."""
  fn_map = _normalize_shear_schedule(shear_schedule, dim)
  if not fn_map:
    return 1.0

  if shear_remap:
    planes = tuple(sorted(fn_map.keys()))
    lambda_max = 1.0
    for signs in itertools.product((-0.5, 0.5), repeat=len(planes)):
      gammas = {plane: sign for plane, sign in zip(planes, signs)}
      F = _deformation_matrix_from_gammas(dim, gammas)
      lambda_max = max(lambda_max, quadrature_lambda_from_deformation(F))
    return lambda_max

  if shear_t_bounds is None:
    raise ValueError("shear_t_bounds=(t0, t1) is required when shear_schedule is provided with shear_remap=False.")
  t0, t1 = float(shear_t_bounds[0]), float(shear_t_bounds[1])
  if not math.isfinite(t0) or not math.isfinite(t1):
    raise ValueError("shear_t_bounds entries must be finite.")
  if t1 < t0:
    raise ValueError("shear_t_bounds must satisfy t1 >= t0.")

  tm = 0.5 * (t0 + t1)
  gammas_t0 = {}
  gammas_t1 = {}
  for plane, fn in fn_map.items():
    g0 = float(fn(t0))
    g1 = float(fn(t1))
    gm = float(fn(tm))
    if not (math.isfinite(g0) and math.isfinite(g1) and math.isfinite(gm)):
      raise ValueError(f"Non-finite shear value encountered on plane '{plane}'.")
    g_affine = 0.5 * (g0 + g1)
    tol = max(1e-10, 1e-8 * max(abs(g0), abs(g1), abs(gm), 1.0))
    if abs(gm - g_affine) > tol:
      raise ValueError(
          f"shear_schedule for plane '{plane}' is not affine on [{t0}, {t1}] "
          f"(midpoint check failed: gm={gm}, expected={g_affine})."
      )
    gammas_t0[plane] = g0
    gammas_t1[plane] = g1

  F0 = _deformation_matrix_from_gammas(dim, gammas_t0)
  F1 = _deformation_matrix_from_gammas(dim, gammas_t1)
  return max(
      quadrature_lambda_from_deformation(F0),
      quadrature_lambda_from_deformation(F1),
  )


def _quadrature_error_bound(P: int, quadrature_lambda_max: float = 1.0) -> Tuple[float, float]:
  """Quadrature error bound ε_q and Gaussian width m (Fiore 2017, Sec. II.C).

  Uses the spectral Ewald choice m = sqrt(pi P), consistent with the
  theta-selection in `choose_theta`, with deformation penalty lambda_max from
  Fiore & Swan (2018) Eq. (55).
  """
  quadrature_lambda_max = float(quadrature_lambda_max)
  if quadrature_lambda_max <= 0.0 or not math.isfinite(quadrature_lambda_max):
    raise ValueError(f"quadrature_lambda_max must be positive and finite; got {quadrature_lambda_max}.")
  P = int(P)
  m = math.sqrt(math.pi * float(P))
  term1 = math.exp(-0.5 * math.pi * float(P) / quadrature_lambda_max)
  term2 = math.erfc(m / math.sqrt(2.0 * quadrature_lambda_max))
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


def _select_P_and_m(
    eps_q: float,
    quadrature_lambda_max: float = 1.0,
    quadrature_safety_nodes: int = 0,
) -> Tuple[int, float, float]:
  """Choose the smallest integer P with ε_q <= eps_q."""
  eps_q = float(max(eps_q, 1e-16))
  quadrature_safety_nodes = int(quadrature_safety_nodes)
  if quadrature_safety_nodes < 0:
    raise ValueError("quadrature_safety_nodes must be >= 0.")
  P = 4
  err, m = _quadrature_error_bound(P, quadrature_lambda_max=quadrature_lambda_max)
  while err > eps_q and P < 64:
    P += 1
    err, m = _quadrature_error_bound(P, quadrature_lambda_max=quadrature_lambda_max)
  if quadrature_safety_nodes:
    P += quadrature_safety_nodes
    err, m = _quadrature_error_bound(P, quadrature_lambda_max=quadrature_lambda_max)
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
                        shear_schedule: Optional[
                            Union[Callable[[float], float], Mapping[str, Callable[[float], float]]]
                        ] = None,
                        shear_t_bounds: Optional[Tuple[float, float]] = None,
                        shear_remap: bool = False,
                        quadrature_safety_nodes: int = 1,
                        notes: bool = False) -> dict:
  """Fiore (2017) parameter estimator for split-Ewald RPY.

  Uses the paper's decoupled error bounds to choose the real-space cutoff
  (rcut), wave-space cutoff (kcut), and quadrature parameters (P, m), with
  the default spectral Ewald choice m = sqrt(pi P). When xi is not provided,
  the asymptotic optimal xi* from Eq. (23) is used with implementation
  constants set to unity; for dilute/underspecified cases, xi is set by
  xi * a ≈ 0.5 (Fiore 2017, Fig. 4/Table I).

  Returns dict with keys:
    xi, P, M, grid_shape, rcut, kcut, theta, m, lattice_extent
  """
  tol = float(tol)
  A = jnp.asarray(A, dtype=REAL_DTYPE)
  if tol <= 0.0:
    raise ValueError("tol must be positive.")
  if a <= 0.0:
    raise ValueError("a must be positive.")
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError("A must be a square 2D box matrix.")
  dim = int(A.shape[0])

  L_cols = jnp.linalg.norm(A, axis=0)
  L_max = float(jnp.max(L_cols))
  L_mean = float(jnp.mean(L_cols))
  sigma_vals = jnp.linalg.svd(A, compute_uv=False)
  sigma_min = float(jnp.min(sigma_vals))
  safe_sigma = max(sigma_min, 1e-12)

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

  rcut = math.sqrt(math.log(1.0 / eps_r)) / xi
  lattice_extent = int(math.ceil(rcut / safe_sigma))

  if shear_schedule is None:
    quadrature_lambda_max = 1.0
  else:
    quadrature_lambda_max = _max_quadrature_lambda_from_shear(
        shear_schedule,
        dim,
        shear_t_bounds=shear_t_bounds,
        shear_remap=shear_remap,
    )

  P, m, eps_q_est = _select_P_and_m(
      eps_q,
      quadrature_lambda_max=quadrature_lambda_max,
      quadrature_safety_nodes=quadrature_safety_nodes,
  )

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
      'lattice_extent': int(lattice_extent),
  }

  if notes:
    eps_w_est = _epsilon_k_bound(kcut, xi, a)
    eps_r_est = math.exp(-(xi * rcut) ** 2)
    result['notes'] = {
        'sigma_min': float(sigma_min),
        'lattice_extent': int(lattice_extent),
        'L_mean': float(L_mean),
        'L_max': float(L_max),
        'eps_r_target': float(eps_r),
        'eps_w_target': float(eps_w),
        'eps_q_target': float(eps_q),
        'eps_r_est': float(eps_r_est),
        'eps_w_est': float(eps_w_est),
        'eps_q_est': float(eps_q_est),
        'quadrature_lambda_max': float(quadrature_lambda_max),
        'quadrature_safety_nodes': int(quadrature_safety_nodes),
        'shear_remap': bool(shear_remap),
        'shear_schedule_provided': bool(shear_schedule is not None),
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

  - `space_fns` must include displacement/shift and may include `box_fn`.
  - Wave modes are rebuilt on the *current* box each call for exact
    instantaneous wave-space mobility under deformation.
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
    self_coeff = float((1.0 / (6.0 * jnp.pi * eta * a)) * Mr_self(a, xi))
    precond = jacobi_from_self(self_coeff)
  else:
    precond = preconditioner

  # Static wave quadrature/grid factors shared across exact live-box evaluation.
  if theta_ is None:
    M_eff = (Mx + My + Mz) / 3.0
    theta_eff = float(choose_theta(P_, xi, M_eff))
  else:
    theta_eff = float(theta_)
  alpha_eff = float(se_alpha(xi, theta_eff))
  alpha_arr = jnp.asarray(alpha_eff, dtype=REAL_DTYPE)
  QX, QY, QZ = q_grid(Mx, My, Mz)
  Q2 = QX * QX + QY * QY + QZ * QZ
  Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
  deconv_pref = (alpha_arr / jnp.pi) ** 3 / (Ngrid ** 2)
  deconv = deconv_pref * jnp.exp(2.0 * (jnp.pi ** 2) * Q2 / alpha_arr)

  def _wave_operators_exact(current_box: jnp.ndarray,
                            positions_frac: jnp.ndarray,
                            forces: jnp.ndarray,
                            *,
                            key_wave: Optional[jax.Array] = None):
    """Exact wave-space apply under the live box without host-side rebuilds."""
    box = jnp.asarray(current_box, dtype=REAL_DTYPE)
    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    forces = jnp.asarray(forces, dtype=REAL_DTYPE)

    Brecip = make_reciprocal(box)
    k, K, K2 = k_from_q(QX, QY, QZ, Brecip)
    V_box = jnp.linalg.det(box)
    sigma_inv = Ngrid / V_box
    Pshape = build_P_modes(K, a)
    Bfluid, Bhalf = build_B_modes(k, K, K2, xi, eta, V_box, deconv)

    st = build_stencils_frac(positions_frac, Mx, My, Mz, P_, alpha_eff)
    force_grid = spread(forces, st, Mx, My, Mz)
    force_q = fft_vec(sigma_inv * force_grid)
    P_force_q = Pshape[..., None] * force_q
    BP_force_q = jnp.einsum('...ij,...j->...i', Bfluid, P_force_q)
    Uq = Pshape[..., None] * BP_force_q
    u_grid = ifft_vec(Uq)
    velocities = V_box * gather(u_grid, st, Mx, My, Mz)

    if key_wave is None:
      return velocities, None

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    draw = _hermitian_gaussian_modes(key_wave, (Mx, My, Mz, 3))
    modes_q = jnp.einsum('...ij,...j->...i', Bhalf_complex, draw)
    modes_q = Pshape[..., None] * modes_q
    u_grid_noise = ifft_vec(modes_q)
    vel_noise = gather(u_grid_noise, st, Mx, My, Mz)
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    wave_noise = noise_scale * (jnp.sqrt(V_box) * vel_noise)
    return velocities, wave_noise

  def _build_wave_state_for_box(box_matrix: jnp.ndarray) -> WaveSpaceState:
    return build_Mw_state(
        box_matrix,
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

  def init_fn(positions_frac, **kwargs):
    positions_frac = jnp.asarray(positions_frac)
    combined_kwargs = dict(box_kwargs_default)
    combined_kwargs.update(kwargs)

    dim = int(positions_frac.shape[1])
    active_box = current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)

    L_min = float(jnp.min(jnp.linalg.norm(active_box, axis=0)))
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

    wave_state = _build_wave_state_for_box(active_box)

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

    wave_state = state.wave
    if current_box is not None:
      # In traced loops (lax.scan/fori_loop), avoid rebuilding WaveSpaceState objects.
      # We evaluate the exact live-box wave operator with pure JAX arrays instead.
      Uw, _ = _wave_operators_exact(current_box, positions_frac, forces)
      if not isinstance(current_box, jax_core.Tracer):
        wave_state = _build_wave_state_for_box(current_box)
    else:
      if wave_state.apply_fn is None:
        raise ValueError('wave-space operator missing from state; run init_fn first.')
      Uw = wave_state.apply_fn(positions_frac, forces)
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
    if current_box is not None:
      _, wave_noise = _wave_operators_exact(
          current_box,
          positions_frac,
          forces,
          key_wave=key_wave,
      )
      if wave_noise is None:
        raise ValueError('Exact wave-space sampler unexpectedly returned None.')
    else:
      if wave_state.sqrt_fn is None:
        raise ValueError('wave-space sampler missing from state; run init_fn first.')
      wave_noise = wave_state.sqrt_fn(
          key_wave,
          positions_frac,
      )
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

  The wave-space sampler is attached to `state.wave`, which must correspond to
  the box encoded in `state.real`.
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

  if state.wave.sqrt_fn is None:
    raise ValueError('wave-space sampler missing; rebuild with include_brownian=True.')
  wave_sample_real = state.wave.sqrt_fn(
      key_wave,
      positions_frac,
  )

  scale = jnp.sqrt(2.0 * kT * dt)
  return scale * (real_sample_real + wave_sample_real)

