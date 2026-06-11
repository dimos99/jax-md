"""Spectral-Ewald RPY mobility implementation.

This module provides functions to build and apply the periodic
Rotne-Prager-Yamakawa (RPY) mobility operator using a split-Ewald
formulation. It combines real-space and wave-space contributions for
efficient hydrodynamic interactions in periodic systems. It supports
deterministic mobility application and stochastic Brownian velocity
sampling.

Real-space contributions are computed using a neighbor list with a cutoff
radius determined. Wave-space contributions are computed on a uniform grid
using FFTs, with parameters chosen to meet user-specified error tolerances.

Reference: Fiore et al., J. Chem. Phys. 146, 124116 (2017).
"""

from typing import Callable, Optional, Sequence, Tuple
import itertools
import math
import warnings

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax import config as jax_config

from jax_md import dataclasses, partition
from jax_md.hydro.rpy_wave import (
  WaveSpaceState,
  build_Mw_state,
  choose_theta,
  se_alpha,
  make_reciprocal,
  q_grid,
  k_from_q,
  build_P_modes,
  build_Pdip_modes,
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
from jax_md.hydro.rpy_moments import (
  N_MOMENTS,
  couplet_to_components,
  components_to_couplet,
  traceless,
)
from jax_md.hydro.rpy_real_det_dipole import build_Mr_grand_apply
from jax_md.hydro.rpy_wave_det_dipole import build_Mw_grand_state
from jax_md.hydro.rpy_wave_stoch import _hermitian_gaussian_modes

XI_OPT_A = 0.5  # default target for xi * a
REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32 #type: ignore[assignment]
COMPLEX_DTYPE = jnp.complex128 if jax_config.jax_enable_x64 else jnp.complex64 #type: ignore[assignment]


ShearVector = Tuple[float, float, float] # (gamma_xy, gamma_xz, gamma_yz)
ShearVectorSchedule = Callable[[float], Sequence[float]] # function of time


# -----------------------------------------------------------------------------
# Shear normalization and deformation helpers
# -----------------------------------------------------------------------------
def _shear_plane_count(dim: int) -> int:
  """Return the number of independent shear planes for a given dimension."""
  if dim == 2:
    return 1
  if dim == 3:
    return 3
  raise ValueError(f"Unsupported dimensionality for shear estimator: dim={dim}.")


def _as_shear_vector(value: Sequence[float], dim: int, *, label: str) -> ShearVector:
  """Convert a length-3 sequence to a ShearVector, validating zero components for 2D."""
  try:
    gamma = tuple(float(v) for v in value)
  except TypeError as err:
    raise ValueError(f"{label} must be a length-3 sequence (gamma_xy, gamma_xz, gamma_yz).") from err
  if len(gamma) != 3:
    raise ValueError(f"{label} must be length 3; got length {len(gamma)}.")
  if dim == 2:
    tol = 1e-12
    if abs(gamma[1]) > tol or abs(gamma[2]) > tol:
      raise ValueError(
          f"{label} has non-zero xz/yz components for dim=2: "
          f"gamma_xz={gamma[1]}, gamma_yz={gamma[2]}."
      )
  return (gamma[0], gamma[1], gamma[2])


def _deformation_matrix_from_shear(dim: int, gamma: ShearVector) -> jnp.ndarray:
  """Build dimensionless deformation tensor F from shear strains."""
  gamma_xy, gamma_xz, gamma_yz = gamma
  F = jnp.eye(dim, dtype=REAL_DTYPE)
  if dim >= 2:
    F = F.at[0, 1].set(jnp.asarray(gamma_xy, dtype=REAL_DTYPE))
  if dim >= 3:
    F = F.at[0, 2].set(jnp.asarray(gamma_xz, dtype=REAL_DTYPE))
    F = F.at[1, 2].set(jnp.asarray(gamma_yz, dtype=REAL_DTYPE))
  return F


def _normalize_runtime_shear_kwargs(kwargs, dim: int):
  """Reject legacy dict shear inputs and expand typed shear vectors."""
  result = dict(kwargs)
  gamma = result.get('gamma', None)
  if isinstance(gamma, dict):
    raise ValueError(
        "gamma dict inputs are not supported. Use shear=(gamma_xy, gamma_xz, gamma_yz) "
        "or explicit gamma_xy/gamma_xz/gamma_yz keyword arguments."
    )
  shear = result.pop('shear', None)
  if shear is not None:
    shear_arr = jnp.asarray(shear)
    if shear_arr.shape != (3,):
      raise ValueError("shear must be a length-3 vector (gamma_xy, gamma_xz, gamma_yz).")
    result['gamma_xy'] = shear_arr[0]
    if dim >= 3:
      result['gamma_xz'] = shear_arr[1]
      result['gamma_yz'] = shear_arr[2]
  return result


# -----------------------------------------------------------------------------
# Fiore-style estimator bounds and parameter-selection helpers
# -----------------------------------------------------------------------------
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
    shear_schedule: ShearVectorSchedule,
    dim: int,
    *,
    shear_t_bounds: Optional[Tuple[float, float]],
    shear_remap: bool,
) -> float:
  """Compute exact quadrature deformation penalty from shear schedule assumptions."""
  n_planes = _shear_plane_count(dim)

  def gamma_at(t: float, label: str) -> ShearVector:
    return _as_shear_vector(shear_schedule(float(t)), dim, label=label)

  if shear_remap:
    if shear_t_bounds is not None:
      t0, t1 = float(shear_t_bounds[0]), float(shear_t_bounds[1])
      sample_times = (t0, 0.5 * (t0 + t1), t1)
    else:
      sample_times = (0.0, 0.5, 1.0)
    component_max = [0.0] * n_planes
    for t in sample_times:
      gamma = gamma_at(t, f"shear_schedule({t})")
      for idx in range(n_planes):
        component_max[idx] = max(component_max[idx], abs(float(gamma[idx])))
    active_components = tuple(i for i, vmax in enumerate(component_max) if vmax > 1e-12)
    if not active_components:
      return 1.0

    lambda_max = 1.0
    for signs in itertools.product((-0.5, 0.5), repeat=len(active_components)):
      gamma = [0.0, 0.0, 0.0]
      for idx, sign in zip(active_components, signs):
        gamma[idx] = sign
      F = _deformation_matrix_from_shear(dim, (gamma[0], gamma[1], gamma[2]))
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
  gamma0 = gamma_at(t0, "shear_schedule(t0)")
  gamma1 = gamma_at(t1, "shear_schedule(t1)")
  gamma_mid = gamma_at(tm, "shear_schedule(midpoint)")

  plane_names = ('xy',) if dim == 2 else ('xy', 'xz', 'yz')
  for idx, plane in enumerate(plane_names):
    g0 = float(gamma0[idx])
    g1 = float(gamma1[idx])
    gm = float(gamma_mid[idx])
    if not (math.isfinite(g0) and math.isfinite(g1) and math.isfinite(gm)):
      raise ValueError(f"Non-finite shear value encountered on plane '{plane}'.")
    g_affine = 0.5 * (g0 + g1)
    tol = max(1e-10, 1e-8 * max(abs(g0), abs(g1), abs(gm), 1.0))
    if abs(gm - g_affine) > tol:
      raise ValueError(
          f"shear_schedule for plane '{plane}' is not affine on [{t0}, {t1}] "
          f"(midpoint check failed: gm={gm}, expected={g_affine})."
      )
  F0 = _deformation_matrix_from_shear(dim, gamma0)
  F1 = _deformation_matrix_from_shear(dim, gamma1)
  return max(
      quadrature_lambda_from_deformation(F0),
      quadrature_lambda_from_deformation(F1),
  )


@dataclasses.dataclass
class RpyParameterDiagnostics:
  sigma_min: float
  lattice_extent: int
  L_mean: float
  L_max: float
  eps_r_target: float
  eps_w_target: float
  eps_q_target: float
  eps_r_est: float
  eps_w_est: float
  eps_q_est: float
  quadrature_lambda_max: float
  quadrature_safety_nodes: int
  shear_remap: bool
  shear_schedule_provided: bool
  N: int
  phi: float
  d_f: float
  n_iter: int


@dataclasses.dataclass
class RpyParameterEstimate:
  xi: float
  P: int
  M: int
  grid_shape: Tuple[int, int, int]
  rcut: float
  kcut: float
  theta: float
  m: float
  lattice_extent: int
  diagnostics: Optional[RpyParameterDiagnostics] = None


# -----------------------------------------------------------------------------
# Estimator internals and public estimator
# -----------------------------------------------------------------------------
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


def _validate_estimator_inputs(
    tol: float,
    A: jnp.ndarray,
    a: float,
    error_split: Tuple[float, float, float],
) -> Tuple[float, jnp.ndarray, int, Tuple[float, float, float]]:
  tol = float(tol)
  A = jnp.asarray(A, dtype=REAL_DTYPE)
  if tol <= 0.0:
    raise ValueError("tol must be positive.")
  if a <= 0.0:
    raise ValueError("a must be positive.")
  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError("A must be a square 2D box matrix.")

  split_r, split_w, split_q = error_split
  if split_r <= 0 or split_w <= 0 or split_q <= 0:
    raise ValueError("error_split entries must be positive.")
  split_sum = split_r + split_w + split_q
  if split_sum > 1.0 + 1e-12:
    raise ValueError("error_split entries must sum to <= 1.")

  return tol, A, int(A.shape[0]), (float(split_r), float(split_w), float(split_q))


def _compute_box_scales(A: jnp.ndarray) -> Tuple[float, float, float, float]:
  L_cols = jnp.linalg.norm(A, axis=0)
  L_max = float(jnp.max(L_cols))
  L_mean = float(jnp.mean(L_cols))
  sigma_vals = jnp.linalg.svd(A, compute_uv=False)
  sigma_min = float(jnp.min(sigma_vals))
  safe_sigma = max(sigma_min, 1e-12)
  return L_max, L_mean, sigma_min, safe_sigma


def _compute_xi_candidate(
    *,
    xi_override: Optional[float],
    a: float,
    N: int,
    phi: float,
    d_f: float,
    n_iter: int,
    C_R: float,
    C_W: float,
) -> float:
  if xi_override is not None:
    xi = float(xi_override)
  else:
    if phi > 0.0 and N > 1 and d_f > 0.0:
      xi = (3.0 * C_W * math.log(float(N)) /
            (C_R * float(d_f) * float(n_iter) * float(phi) ** 2)) ** (
                -1.0 / (float(d_f) + 3.0))
    else:
      xi = float(XI_OPT_A / a)
  if xi <= 0.0 or not math.isfinite(xi):
    raise ValueError(f"xi must be positive; got xi={xi}.")
  return xi


def _compute_error_targets(
    tol: float,
    error_split: Tuple[float, float, float],
) -> Tuple[float, float, float]:
  split_r, split_w, split_q = error_split
  eps_r = max(tol * split_r, 1e-16)
  eps_w = max(tol * split_w, 1e-16)
  eps_q = max(tol * split_q, 1e-16)
  return eps_r, eps_w, eps_q


def _compute_wave_grid_selection(
    *,
    xi: float,
    a: float,
    eps_w: float,
    P: int,
    L_max: float,
    m: float,
) -> Tuple[float, int, float]:
  kcut = _select_kcut(xi, a, eps_w)
  # FFT grid size from kcut: k_max ≈ π M / L.
  M_est = int(math.ceil(kcut * L_max / math.pi))
  M = max(M_est, P, 8)
  if M % 2 != 0:
    M += 1
  theta = float(choose_theta(P, xi, M, m=m))
  return kcut, M, theta


def _build_parameter_diagnostics(
    *,
    sigma_min: float,
    lattice_extent: int,
    L_mean: float,
    L_max: float,
    eps_r: float,
    eps_w: float,
    eps_q: float,
    rcut: float,
    kcut: float,
    xi: float,
    a: float,
    eps_q_est: float,
    quadrature_lambda_max: float,
    quadrature_safety_nodes: int,
    shear_remap: bool,
    shear_schedule_provided: bool,
    N: int,
    phi: float,
    d_f: float,
    n_iter: int,
) -> RpyParameterDiagnostics:
  eps_w_est = _epsilon_k_bound(kcut, xi, a)
  eps_r_est = math.exp(-(xi * rcut) ** 2)
  return RpyParameterDiagnostics(
      sigma_min=float(sigma_min),
      lattice_extent=int(lattice_extent),
      L_mean=float(L_mean),
      L_max=float(L_max),
      eps_r_target=float(eps_r),
      eps_w_target=float(eps_w),
      eps_q_target=float(eps_q),
      eps_r_est=float(eps_r_est),
      eps_w_est=float(eps_w_est),
      eps_q_est=float(eps_q_est),
      quadrature_lambda_max=float(quadrature_lambda_max),
      quadrature_safety_nodes=int(quadrature_safety_nodes),
      shear_remap=bool(shear_remap),
      shear_schedule_provided=bool(shear_schedule_provided),
      N=int(N),
      phi=float(phi),
      d_f=float(d_f),
      n_iter=int(n_iter),
  )


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
                        shear_vector_schedule: Optional[ShearVectorSchedule] = None,
                        shear_t_bounds: Optional[Tuple[float, float]] = None,
                        shear_remap: bool = False,
                        quadrature_safety_nodes: int = 1,
                        notes: bool = False) -> RpyParameterEstimate:
  """Fiore (2017) parameter estimator for split-Ewald RPY.

  Uses the paper's decoupled error bounds to choose the real-space cutoff
  (rcut), wave-space cutoff (kcut), and quadrature parameters (P, m), with
  the default spectral Ewald choice m = sqrt(pi P). When xi is not provided,
  the asymptotic optimal xi* from Eq. (23) is used with implementation
  constants set to unity; for dilute/underspecified cases, xi is set by
  xi * a ≈ 0.5 (Fiore 2017, Fig. 4/Table I).

  Returns:
    RpyParameterEstimate with xi/P/M/rcut/kcut/theta/m and lattice extent.
  """
  tol, A, dim, split = _validate_estimator_inputs(
      tol,
      A,
      a,
      error_split,
  )
  L_max, L_mean, sigma_min, safe_sigma = _compute_box_scales(A)
  xi = _compute_xi_candidate(
      xi_override=xi_override,
      a=a,
      N=N,
      phi=phi,
      d_f=d_f,
      n_iter=n_iter,
      C_R=C_R,
      C_W=C_W,
  )
  eps_r, eps_w, eps_q = _compute_error_targets(tol, split)

  rcut = math.sqrt(math.log(1.0 / eps_r)) / xi
  lattice_extent = int(math.ceil(2.0 * rcut / safe_sigma))

  if shear_vector_schedule is None:
    quadrature_lambda_max = 1.0
  else:
    quadrature_lambda_max = _max_quadrature_lambda_from_shear(
        shear_vector_schedule,
        dim,
        shear_t_bounds=shear_t_bounds,
        shear_remap=shear_remap,
    )

  P, m, eps_q_est = _select_P_and_m(
      eps_q,
      quadrature_lambda_max=quadrature_lambda_max,
      quadrature_safety_nodes=quadrature_safety_nodes,
  )

  kcut, M, theta = _compute_wave_grid_selection(
      xi=xi,
      a=a,
      eps_w=eps_w,
      P=P,
      L_max=L_max,
      m=m,
  )

  diagnostics = None

  if notes:
    diagnostics = _build_parameter_diagnostics(
        sigma_min=sigma_min,
        lattice_extent=lattice_extent,
        L_mean=L_mean,
        L_max=L_max,
        eps_r=eps_r,
        eps_w=eps_w,
        eps_q=eps_q,
        rcut=rcut,
        kcut=kcut,
        xi=xi,
        a=a,
        eps_q_est=eps_q_est,
        quadrature_lambda_max=quadrature_lambda_max,
        quadrature_safety_nodes=quadrature_safety_nodes,
        shear_remap=shear_remap,
        shear_schedule_provided=shear_vector_schedule is not None,
        N=N,
        phi=phi,
        d_f=d_f,
        n_iter=n_iter,
    )

  return RpyParameterEstimate(
      xi=float(xi),
      P=int(P),
      M=int(M),
      grid_shape=(int(M), int(M), int(M)),
      rcut=float(rcut),
      kcut=float(kcut),
      theta=float(theta),
      m=float(m),
      lattice_extent=int(lattice_extent),
      diagnostics=diagnostics,
  )


# -----------------------------------------------------------------------------
# RPY operator state and mobility-construction helpers
# -----------------------------------------------------------------------------
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


@dataclasses.dataclass
class _WaveStaticFactors:
  Mx: int
  My: int
  Mz: int
  P_support: int
  alpha_eff: float
  Ngrid: jnp.ndarray
  QX: jnp.ndarray
  QY: jnp.ndarray
  QZ: jnp.ndarray
  deconv: jnp.ndarray


def _resolve_rcut(xi: float, rcut: Optional[float]) -> float:
  if rcut is not None:
    return float(rcut)
  target_epsR = 1e-6
  return float((1.0 / xi) * jnp.sqrt(jnp.log(1.0 / target_epsR)))


def _resolve_wave_grid(
    Mgrid: Optional[int],
    Mx: Optional[int],
    My: Optional[int],
    Mz: Optional[int],
    P: Optional[int],
) -> Tuple[int, int, int, int]:
  grid_default = Mgrid
  Mx_ = Mx if Mx is not None else grid_default
  My_ = My if My is not None else grid_default
  Mz_ = Mz if Mz is not None else grid_default
  if Mx_ is None or My_ is None or Mz_ is None:
    raise ValueError('Specify Mgrid or Mx/My/Mz to size the wave-space grid.')
  P_ = 16 if P is None else int(P)
  return int(Mx_), int(My_), int(Mz_), P_


def _resolve_preconditioner(
    preconditioner: Optional[Preconditioner],
    *,
    a: float,
    xi: float,
    eta: float,
) -> Preconditioner:
  if preconditioner is not None:
    return preconditioner
  self_coeff = float((1.0 / (6.0 * jnp.pi * eta * a)) * Mr_self(a, xi))
  return jacobi_from_self(self_coeff)


def _prepare_wave_static_factors(
    *,
    xi: float,
    theta: Optional[float],
    Mx: int,
    My: int,
    Mz: int,
    P_support: int,
) -> _WaveStaticFactors:
  if theta is None:
    M_eff = (Mx + My + Mz) / 3.0
    theta_eff = float(choose_theta(P_support, xi, M_eff))
  else:
    theta_eff = float(theta)
  alpha_eff = float(se_alpha(xi, theta_eff))
  alpha_arr = jnp.asarray(alpha_eff, dtype=REAL_DTYPE)
  QX, QY, QZ = q_grid(Mx, My, Mz)
  Q2 = QX * QX + QY * QY + QZ * QZ
  Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
  deconv_pref = (alpha_arr / jnp.pi) ** 3 / (Ngrid ** 2)
  deconv = deconv_pref * jnp.exp(2.0 * (jnp.pi ** 2) * Q2 / alpha_arr)
  return _WaveStaticFactors(
      Mx=Mx,
      My=My,
      Mz=Mz,
      P_support=P_support,
      alpha_eff=alpha_eff,
      Ngrid=Ngrid,
      QX=QX,
      QY=QY,
      QZ=QZ,
      deconv=deconv,
  )


def _apply_wave_exact(
    *,
    static: _WaveStaticFactors,
    current_box: jnp.ndarray,
    positions_frac: jnp.ndarray,
    forces: jnp.ndarray,
    a: float,
    xi: float,
    eta: float,
    key_wave: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
  """Exact wave-space operator and optional stochastic sample under the live box."""
  box = jnp.asarray(current_box, dtype=REAL_DTYPE)
  positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
  forces = jnp.asarray(forces, dtype=REAL_DTYPE)

  Brecip = make_reciprocal(box)
  k, K, K2 = k_from_q(static.QX, static.QY, static.QZ, Brecip)
  V_box = jnp.linalg.det(box)
  sigma_inv = static.Ngrid / V_box
  Pshape = build_P_modes(K, a)
  Bfluid, Bhalf = build_B_modes(k, K, K2, xi, eta, V_box, static.deconv)

  st = build_stencils_frac(
      positions_frac,
      static.Mx,
      static.My,
      static.Mz,
      static.P_support,
      static.alpha_eff,
  )
  force_grid = spread(forces, st, static.Mx, static.My, static.Mz)
  force_q = fft_vec(sigma_inv * force_grid)
  P_force_q = Pshape[..., None] * force_q
  BP_force_q = jnp.einsum('...ij,...j->...i', Bfluid, P_force_q)
  Uq = Pshape[..., None] * BP_force_q
  u_grid = ifft_vec(Uq)
  velocities = V_box * gather(u_grid, st, static.Mx, static.My, static.Mz)

  if key_wave is None:
    return velocities, None

  Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
  draw = _hermitian_gaussian_modes(
      key_wave,
      (static.Mx, static.My, static.Mz, 3),
  )
  modes_q = jnp.einsum('...ij,...j->...i', Bhalf_complex, draw)
  modes_q = Pshape[..., None] * modes_q
  u_grid_noise = ifft_vec(modes_q)
  vel_noise = gather(u_grid_noise, st, static.Mx, static.My, static.Mz)
  noise_scale = jnp.sqrt(sigma_inv * static.Ngrid)
  wave_noise = noise_scale * (jnp.sqrt(V_box) * vel_noise)
  return velocities, wave_noise


def _apply_wave_exact_grand(
    *,
    static: _WaveStaticFactors,
    current_box: jnp.ndarray,
    positions_frac: jnp.ndarray,
    forces: jnp.ndarray,
    couplets: jnp.ndarray,
    a: float,
    xi: float,
    eta: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Exact grand wave-space operator under the live box."""
  box = jnp.asarray(current_box, dtype=REAL_DTYPE)
  positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
  forces = jnp.asarray(forces, dtype=REAL_DTYPE)
  couplets = traceless(jnp.asarray(couplets, dtype=REAL_DTYPE))

  Brecip = make_reciprocal(box)
  k, K, K2 = k_from_q(static.QX, static.QY, static.QZ, Brecip)
  V_box = jnp.linalg.det(box)
  sigma_inv = static.Ngrid / V_box
  Pshape = build_P_modes(K, a)
  Pdip = build_Pdip_modes(K, a)
  Bfluid, _ = build_B_modes(k, K, K2, xi, eta, V_box, static.deconv)

  st = build_stencils_frac(
      positions_frac,
      static.Mx,
      static.My,
      static.Mz,
      static.P_support,
      static.alpha_eff,
  )
  moments = jnp.concatenate([forces, couplet_to_components(couplets)], axis=-1)
  moment_grid = sigma_inv * spread(moments, st, static.Mx, static.My, static.Mz)
  moment_q = fft_vec(moment_grid)
  Fq = moment_q[..., :3]
  Cq = components_to_couplet(moment_q[..., 3:N_MOMENTS])

  fq = (Pshape[..., None] * Fq -
        1j * Pdip[..., None] * jnp.einsum('...mn,...n->...m', Cq, k))
  uq = jnp.einsum('...ij,...j->...i', Bfluid, fq)
  Uq = Pshape[..., None] * uq
  Dq = 1j * Pdip[..., None, None] * jnp.einsum('...i,...j->...ij', uq, k)

  out_q = jnp.concatenate([Uq, couplet_to_components(Dq)], axis=-1)
  out_grid = ifft_vec(out_q)
  out = V_box * gather(out_grid, st, static.Mx, static.My, static.Mz)
  velocities = out[..., :3]
  gradients = components_to_couplet(out[..., 3:N_MOMENTS])
  return velocities, traceless(gradients)


def _compose_next_state(
    *,
    real_state: RealSpaceState,
    wave_state: WaveSpaceState,
    preconditioner: Optional[Preconditioner],
) -> RpyState:
  return RpyState(
      real=real_state,
      wave=wave_state,
      preconditioner=preconditioner,
  )


# -----------------------------------------------------------------------------
# Brownian sampling helpers (shared by apply_fn and brownian_increment)
# -----------------------------------------------------------------------------
def _sample_real_space_noise(
    *,
    key_real: jax.Array,
    real_state: RealSpaceState,
    positions_frac: jnp.ndarray,
    state_preconditioner: Optional[Preconditioner],
    mr_iters: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  precond_local = state_preconditioner or identity_preconditioner()
  return sample_mr_sqrt_precond(
      key_real,
      real_state,
      positions_frac,
      precond=precond_local,
      iters=mr_iters,
      return_info=True,
  )


def _combine_brownian_noise(
    *,
    real_noise: jnp.ndarray,
    wave_noise: jnp.ndarray,
    kT: float,
    dt: float,
) -> jnp.ndarray:
  return jnp.sqrt(2.0 * kT * dt) * (real_noise + wave_noise)


# -----------------------------------------------------------------------------
# Public mobility builder
# -----------------------------------------------------------------------------
def build_rpy_mobility(space_fns,
                       a: float,
                       xi: float,
                       eta: float,
                       *,
                       rcut: Optional[float] = None,
                       P: Optional[int] = None,
                       Mgrid: Optional[int] = None,
                       Mx: Optional[int] = None,
                       My: Optional[int] = None,
                       Mz: Optional[int] = None,
                       theta: Optional[float] = None,
                       real_space_first: bool = True,
                       include_brownian: bool = True,
                       preconditioner: Optional[Preconditioner] = None,
                       fractional_coordinates: bool = True,
                       dr_threshold: Optional[float] = None,
                       capacity_multiplier: float = 1.25,
                       disable_cell_list: bool = False,
                       neighbor_format=partition.NeighborListFormat.OrderedSparse,
                       extra_capacity: int = 0,
                       lattice_extent: Optional[int] = None,
                       lattice_extra: float = 0.0,
                       box_jump_threshold: Optional[float] = None,
                       real_space_mode: str = 'auto',
                       use_stresslet: bool = False):
  """Construct init/apply functions for the split-Ewald RPY mobility.

  Args:
    space_fns: Tuple of (displacement_fn, shift_fn) or
      (displacement_fn, shift_fn, box_fn). The box function is required for
      sheared or deformable boxes.
    a: Particle hydrodynamic radius.
    xi: Ewald splitting parameter. Must be positive; xi * a ≈ 0.5 is a good
      starting point (Fiore 2017, Table I).
    eta: Solvent dynamic viscosity.
    rcut: Real-space cutoff distance. If None, chosen so that
      ε_r ≈ 1e-6.
    P: Quadrature support (number of grid points per particle in each
      dimension for spreading/interpolation). Defaults to 16.
    Mgrid: Uniform FFT grid size in all dimensions. Overridden by per-axis
      Mx/My/Mz if provided.
    Mx: FFT grid size along x. Overrides Mgrid for this axis.
    My: FFT grid size along y. Overrides Mgrid for this axis.
    Mz: FFT grid size along z. Overrides Mgrid for this axis.
    theta: Spectral Ewald Gaussian width parameter. If None, computed from
      P, xi, and M via ``choose_theta``.
    real_space_first: If True, evaluate real-space before wave-space and sum
      as U_r + U_w; otherwise U_w + U_r. Affects floating-point ordering
      only.
    include_brownian: If True (default), attach the wave-space square-root
      sampler so that Brownian noise can be drawn via ``apply_fn``.
    preconditioner: Preconditioner for the real-space Lanczos M_r^{1/2}
      solve. Defaults to a Jacobi (diagonal) preconditioner built from the
      RPY self-mobility.
    fractional_coordinates: If True (default), positions are interpreted as
      fractional (scaled) coordinates in [0, 1)^d.
    dr_threshold: Neighbor-list rebuild threshold. Passed to
      ``build_Mr_apply``.
    capacity_multiplier: Neighbor-list capacity safety factor. Default is 1.25,
      passed to ``build_Mr_apply``.
    disable_cell_list: If True, use brute-force neighbor search instead of
      cell lists.
    neighbor_format: Neighbor list storage format (Dense, Sparse, or
      OrderedSparse). Default is OrderedSparse.
    extra_capacity: Additional neighbor-list slots beyond the estimate. Default
      is 0, passed to ``build_Mr_apply``.
    lattice_extent: Number of periodic lattice images in each direction for
      the real-space sum. If None, determined from rcut and the box.
    lattice_extra: Extra fractional padding added to the lattice extent. 
      Default is 0.0, passed to ``build_Mr_apply``.
    box_jump_threshold: Maximum allowed fractional box change before forcing
      a neighbor-list rebuild. Default is None, passed to ``build_Mr_apply``.
    real_space_mode: Real-space kernel policy. ``'lattice'`` always uses
      lattice-image accumulation, ``'min_image'`` enforces minimum-image-only
      evaluation (requires ``rcut <= 0.5 * sigma_min(box)``), and ``'auto'``
      picks the safe option per box (default).
    use_stresslet: If True, build the Phase-1 grand mobility from force and
      traceless couplet inputs and return velocity plus traceless velocity
      gradient. Brownian sampling is not implemented for this mode.

  Returns:
    A tuple ``(init_fn, apply_fn)`` where:

    **init_fn(positions_frac, **kwargs) -> RpyState**
      Build the initial mobility state (neighbor list + wave-space arrays)
      for a given configuration. Keyword arguments such as ``gamma_xy`` or
      ``shear=(γ_xy, γ_xz, γ_yz)`` are forwarded to set the current box
      deformation.

    **apply_fn(state, positions_frac, forces, *, couplets=None,
    brownian_key=None,
    kT=None, dt=None, mr_iters=10, **kwargs)**
      Apply the mobility operator and optionally sample Brownian noise.

      *Deterministic mode* (``brownian_key=None``):
        Returns ``(velocities, next_state)`` where
        ``velocities = M · forces`` (shape ``(N, dim)``).
        With ``use_stresslet=True``, returns
        ``((velocities, gradients), next_state)`` where ``gradients`` has
        shape ``(N, 3, 3)`` and ``couplets`` defaults to zeros.

      *Stochastic mode* (``brownian_key`` provided, requires ``kT`` and
      ``dt``):
        Returns ``(velocities, noise, next_state, info)`` where ``noise``
        is the Brownian displacement √(2 kT dt) M^{1/2} z (shape
        ``(N, dim)``) and ``info`` is a dict with keys
        ``'real_solver_rel_change'``, ``'real_solver_iters'``, and
        ``'real_solver_converged'``.

    **apply_fn.with_brownian(state, positions_frac, forces, key, *,
    kT, dt, mr_iters=10, **kwargs)**
      Convenience wrapper that always includes Brownian noise. Returns
      ``(velocities, noise, next_state, rel_change, iters, converged)``.
      
  Usage:
    ```python
    # Build the mobility operator.
    init_fn, apply_fn = build_rpy_mobility(
        space_fns=(displacement_fn, shift_fn, box_fn),
        a=0.5,
        xi=3.0,
        eta=1.0,
        rcut=1.0,
        Mgrid=32,
        theta=None,
        real_space_first=True,
        include_brownian=True,
    )

    # Initialize the state for a given configuration.
    state = init_fn(positions_frac, gamma_xy=0.1)

    # Apply the operator to get velocities and Brownian noise.
    velocities, noise, next_state, info = apply_fn.with_brownian(
        state,
        positions_frac,
        forces,
        key=jax.random.PRNGKey(0),
        kT=1.0,
        dt=0.01,
        mr_iters=10,
    )
  """

  # -- Validate inputs and unpack space functions ---------------------------
  xi = _check_xi(xi)

  if len(space_fns) < 2:
    raise ValueError('space_fns must contain displacement and shift functions.')

  displacement_fn, _ = space_fns[:2]
  box_fn = space_fns[2] if len(space_fns) > 2 else None
  has_box_fn = box_fn is not None

  # -- Build real-space operator (neighbor list + pairwise RPY) -------------
  rcut_value = _resolve_rcut(xi, rcut)

  real_builder = build_Mr_grand_apply if use_stresslet else build_Mr_apply
  Mr_init, Mr_apply = real_builder(
      space_fns,
      a,
      xi,
      eta,
      rcut_value,
      fractional_coordinates=fractional_coordinates,
      dr_threshold=dr_threshold,
      capacity_multiplier=capacity_multiplier,
      disable_cell_list=disable_cell_list,
      neighbor_format=neighbor_format,
      extra_capacity=extra_capacity,
      lattice_extent=lattice_extent,
      lattice_extra=lattice_extra,
      box_jump_threshold=box_jump_threshold,
      real_space_mode=real_space_mode,
  )

  # -- Resolve wave-space grid, preconditioner, and static factors ---------
  Mx_, My_, Mz_, P_ = _resolve_wave_grid(Mgrid, Mx, My, Mz, P)
  theta_ = theta
  precond = _resolve_preconditioner(
      preconditioner,
      a=a,
      xi=xi,
      eta=eta,
  )
  # Pre-compute static quadrature weights and deconvolution filter
  # (Eq. 16/20); these are reused for every live-box wave evaluation.
  wave_static = _prepare_wave_static_factors(
      xi=xi,
      theta=theta_,
      Mx=Mx_,
      My=My_,
      Mz=Mz_,
      P_support=P_,
  )

  # -- Wave-state factory (rebuilt per box for exact deformed mobility) -----
  def _build_wave_state_for_box(box_matrix: jnp.ndarray) -> WaveSpaceState:
    if use_stresslet:
      return build_Mw_grand_state(
          box_matrix,
          a,
          xi,
          eta,
          Mx_,
          My_,
          Mz_,
          P_,
          theta=theta_,
          fractional_coordinates=True,
      )
    return build_Mw_state(
        box_matrix,
        a,
        xi,
        eta,
        Mx_,
        My_,
        Mz_,
        P_,
        theta=theta_,
        fractional_coordinates=True,
        attach_sqrt=include_brownian,
        attach_fused=include_brownian,
    )

  # -- init_fn: allocate neighbor list and wave-space arrays ---------------
  def init_fn(positions_frac, **kwargs):
    positions_frac = jnp.asarray(positions_frac)
    dim = int(positions_frac.shape[1])
    combined_kwargs = _normalize_runtime_shear_kwargs(kwargs, dim)
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

  def _resolve_current_box(dim: int, combined_kwargs) -> Optional[jnp.ndarray]:
    if has_box_fn:
      return current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
    return None

  def _finalize_step(
      *,
      Ur: jnp.ndarray,
      Uw: jnp.ndarray,
      real_state: RealSpaceState,
      wave_state: WaveSpaceState,
      state_preconditioner: Optional[Preconditioner],
  ) -> Tuple[jnp.ndarray, RpyState]:
    velocities = Ur + Uw if real_space_first else Uw + Ur
    next_state = _compose_next_state(
        real_state=real_state,
        wave_state=wave_state,
        preconditioner=state_preconditioner,
    )
    return velocities, next_state

  def _finalize_grand_step(
      *,
      Ur: jnp.ndarray,
      Dr: jnp.ndarray,
      Uw: jnp.ndarray,
      Dw: jnp.ndarray,
      real_state: RealSpaceState,
      wave_state: WaveSpaceState,
      state_preconditioner: Optional[Preconditioner],
  ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], RpyState]:
    if real_space_first:
      velocities = Ur + Uw
      gradients = Dr + Dw
    else:
      velocities = Uw + Ur
      gradients = Dw + Dr
    next_state = _compose_next_state(
        real_state=real_state,
        wave_state=wave_state,
        preconditioner=state_preconditioner,
    )
    return (velocities, traceless(gradients)), next_state

  def _apply_wave_components(
      *,
      wave_state: WaveSpaceState,
      positions_frac: jnp.ndarray,
      forces: jnp.ndarray,
      current_box: Optional[jnp.ndarray],
      key_wave: Optional[jax.Array] = None,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], WaveSpaceState]:
    # Exact live-box path: one call yields deterministic wave velocity and
    # optional stochastic wave sample.
    if current_box is not None:
      Uw, wave_noise = _apply_wave_exact(
          static=wave_static,
          current_box=current_box,
          positions_frac=positions_frac,
          forces=forces,
          a=a,
          xi=xi,
          eta=eta,
          key_wave=key_wave,
      )
      if not isinstance(current_box, jax_core.Tracer):
        wave_state = _build_wave_state_for_box(current_box)
      return Uw, wave_noise, wave_state

    # Static-box deterministic-only path.
    if key_wave is None:
      if wave_state.apply_fn is None:
        raise ValueError('wave-space operator missing from state; run init_fn first.')
      return wave_state.apply_fn(positions_frac, forces), None, wave_state

    # Static-box stochastic path: prefer fused deterministic+stochastic wave call.
    if wave_state.fused_fn is not None:
      Uw, wave_noise = wave_state.fused_fn(
          key_wave,
          positions_frac,
          forces,
      )
      return Uw, wave_noise, wave_state

    # Defensive fallback when fused_fn is unavailable.
    if wave_state.apply_fn is None:
      raise ValueError('wave-space operator missing from state; run init_fn first.')
    if wave_state.sqrt_fn is None:
      raise ValueError('wave-space sampler missing from state; run init_fn first.')
    Uw = wave_state.apply_fn(positions_frac, forces)
    wave_noise = wave_state.sqrt_fn(key_wave, positions_frac)
    return Uw, wave_noise, wave_state

  def _apply_wave_components_grand(
      *,
      wave_state: WaveSpaceState,
      positions_frac: jnp.ndarray,
      forces: jnp.ndarray,
      couplets: jnp.ndarray,
      current_box: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, jnp.ndarray, WaveSpaceState]:
    if current_box is not None:
      Uw, Dw = _apply_wave_exact_grand(
          static=wave_static,
          current_box=current_box,
          positions_frac=positions_frac,
          forces=forces,
          couplets=couplets,
          a=a,
          xi=xi,
          eta=eta,
      )
      if not isinstance(current_box, jax_core.Tracer):
        wave_state = _build_wave_state_for_box(current_box)
      return Uw, Dw, wave_state

    if wave_state.apply_fn is None:
      raise ValueError('grand wave-space operator missing from state; run init_fn first.')
    Uw, Dw = wave_state.apply_fn(positions_frac, forces, couplets)
    return Uw, Dw, wave_state

  # -- apply_fn: deterministic + optional Brownian in one call -------------
  def apply_fn(state: RpyState,
               positions_frac,
               forces,
               *,
               couplets: Optional[jnp.ndarray] = None,
               brownian_key: Optional[jax.Array] = None,
               kT: Optional[float] = None,
               dt: Optional[float] = None,
               mr_iters: int = 10,
               **kwargs):
    positions_frac = jnp.asarray(positions_frac)
    forces = jnp.asarray(forces)

    dim = int(positions_frac.shape[1])
    combined_kwargs = _normalize_runtime_shear_kwargs(kwargs, dim)
    current_box = _resolve_current_box(dim, combined_kwargs)

    if use_stresslet:
      if brownian_key is not None:
        raise NotImplementedError(
            'Brownian sampling for stresslet/couplet moments is not implemented.')
      if couplets is None:
        couplets_local = jnp.zeros(positions_frac.shape[:-1] + (3, 3), dtype=REAL_DTYPE)
      else:
        couplets_local = traceless(jnp.asarray(couplets, dtype=REAL_DTYPE))
      (Ur, Dr), real_state = Mr_apply(
          state.real, positions_frac, forces, couplets_local, **combined_kwargs)
      Uw, Dw, wave_state = _apply_wave_components_grand(
          wave_state=state.wave,
          positions_frac=positions_frac,
          forces=forces,
          couplets=couplets_local,
          current_box=current_box,
      )
      return _finalize_grand_step(
          Ur=Ur,
          Dr=Dr,
          Uw=Uw,
          Dw=Dw,
          real_state=real_state,
          wave_state=wave_state,
          state_preconditioner=state.preconditioner,
      )

    if couplets is not None:
      raise ValueError("couplets may only be passed when build_rpy_mobility(..., use_stresslet=True).")

    Ur, real_state = Mr_apply(state.real, positions_frac, forces, **combined_kwargs)

    if brownian_key is None:
      Uw, _, wave_state = _apply_wave_components(
          wave_state=state.wave,
          positions_frac=positions_frac,
          forces=forces,
          current_box=current_box,
      )
      velocities, next_state = _finalize_step(
          Ur=Ur,
          Uw=Uw,
          real_state=real_state,
          wave_state=wave_state,
          state_preconditioner=state.preconditioner,
      )
      return velocities, next_state

    if not include_brownian:
      raise ValueError('Brownian sampling was disabled; rebuild with include_brownian=True.')
    if kT is None or dt is None:
      raise ValueError('kT and dt are required when requesting Brownian noise.')

    key_real, key_wave = jax.random.split(brownian_key)
    real_noise, rel_change, iters_used, converged = _sample_real_space_noise(
        key_real=key_real,
        real_state=real_state,
        positions_frac=positions_frac,
        state_preconditioner=state.preconditioner,
        mr_iters=mr_iters,
    )
    Uw, wave_noise, wave_state = _apply_wave_components(
        wave_state=state.wave,
        positions_frac=positions_frac,
        forces=forces,
        current_box=current_box,
        key_wave=key_wave,
    )
    if wave_noise is None:
      raise ValueError('Wave-space stochastic sampler returned None in Brownian mode.')

    velocities, next_state = _finalize_step(
        Ur=Ur,
        Uw=Uw,
        real_state=real_state,
        wave_state=wave_state,
        state_preconditioner=state.preconditioner,
    )
    noise = _combine_brownian_noise(
        real_noise=real_noise,
        wave_noise=wave_noise,
        kT=kT,
        dt=dt,
    )
    info = {
        'real_solver_rel_change': rel_change,
        'real_solver_iters': iters_used,
        'real_solver_converged': converged,
    }
    return velocities, noise, next_state, info

  # -- Convenience wrapper: always returns Brownian components --------------
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


# -----------------------------------------------------------------------------
# Public Brownian increment utility
# -----------------------------------------------------------------------------
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
  real_sample_real, _, _, _ = _sample_real_space_noise(
      key_real=key_real,
      real_state=state.real,
      positions_frac=positions_frac,
      state_preconditioner=state.preconditioner,
      mr_iters=mr_iters,
  )
  if state.wave.sqrt_fn is None:
    raise ValueError('wave-space sampler missing; rebuild with include_brownian=True.')
  wave_sample_real = state.wave.sqrt_fn(key_wave, positions_frac)
  return _combine_brownian_noise(
      real_noise=real_sample_real,
      wave_noise=wave_sample_real,
      kT=kT,
      dt=dt,
  )
