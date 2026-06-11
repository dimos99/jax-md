"""Wave-space Spectral Ewald RPY mobility (deterministic part).

Reference: `jax_md/hydro/literature/fiore_chapt3.pdf` (Fiore, Ch. 3).

In Fiore's notation, the positively split wave-space mobility is factorized as
(Eq. 3.16)

  M^(w) = D† · P† · B · P · D

with:
  - D / D† : NUDFT / adjoint NUDFT (Eq. 3.16), accelerated via NUFFT
    quadrature (Eq. 3.19). In code, this corresponds to
      `build_stencils_frac` + `spread` / `gather` + `fft_vec` / `ifft_vec`.
  - P      : particle shape factor (Eq. 3.18), `P(k) = sinc(ka)`, implemented by
      `build_P_modes`.
  - B      : Hasimoto-screened Stokeslet with transverse projection (Eq. 3.17),
      implemented by `build_B_modes`.
  - deconv : Spectral Ewald deconvolution |ĥ(q)|^{-2} used to correct NUFFT
      quadrature error, built analytically by `_build_se_deconv_window`
      (Lindbo–Tornberg 2010).

This module provides helpers to precompute grid modes and apply the resulting
deterministic matvec:
  - `build_wave_modes(...)`  -> precomputed P/B tensors on the FFT grid
  - `build_Mw_apply(state)`  -> Mw(positions, forces)
Stochastic utilities live in `rpy_wave_stoch.py`.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional

from jax_md import dataclasses

from jax_md.hydro.rpy_wave_det_helpers import (
    REAL_DTYPE,
    make_reciprocal,
    positions_to_fractional,
    q_grid,
    k_from_q,  # noqa: F401 -- re-exported via rpy_wave
    se_alpha,
    choose_theta,
    sinc,
    build_P_modes,
    build_Pdip_modes,  # noqa: F401 -- re-exported via rpy_wave
    _build_mode_grid,
    _build_se_deconv_window,
    build_B_modes,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
)


@dataclasses.dataclass
class WaveSpaceParams:
  """Input parameters used to build the wave-space operator.

  This dataclass collects the parameters required to precompute the wave-space
  Spectral Ewald (SE) operator modes on an FFT grid. Instances of
  ``WaveSpaceParams`` are stored in a :class:`WaveSpaceState` and used by the
  builder functions ``build_wave_modes`` / ``build_Mw_apply`` / ``build_Mw_state``.

  Fields
  ------
  A : jnp.ndarray
      3x3 matrix describing the periodic unit cell in real-space coordinates.
      Units: length. This matrix is used to convert between fractional and
      real coordinates and to construct reciprocal-space grid vectors.
  a : float
      Sphere radius (real-space length units). Used for the particle shape
      factor in the Fourier-space shape function (sinc(ka)).
  xi : float
      Ewald splitting parameter (inverse length) controlling the real/vs
      Fourier-space partitioning of the hydrodynamic kernel.
  eta : float
      Fluid viscosity (in consistent units with positions and forces).
  Mx, My, Mz : int
      FFT grid dimensions along each axis. These specify the number of
      discretization points used to represent modes in reciprocal space.
  P : int
      Spectral Ewald Gaussian stencil support size (number of support points)
      used by the spatial spread/gather operations.
  theta : float
      Spectral Ewald SE window parameter. When ``None`` at build time, the
      appropriate theta is chosen automatically using ``choose_theta``.
  fractional_coordinates : bool
      If True, the ``positions`` passed to the wave-space operators are
      interpreted as fractional (relative to the unit cell defined by ``A``).
      When False, positions are provided in the real-space units of ``A`` and
      are converted internally.
  alpha : float
      Gaussian exponent for the SE spreading window used when building
      stencils; stored so downstream operators can avoid recomputing it.
  volume : float
      Volume of the periodic cell (determinant of ``A``), used when scaling
      FFT-grid fields back to particle velocities.
  """

  A: jnp.ndarray
  a: float
  xi: float
  eta: float
  Mx: int
  My: int
  Mz: int
  P: int
  theta: float
  alpha: float
  volume: float
  fractional_coordinates: bool = dataclasses.field(default=True, metadata={'static': True})


@dataclasses.dataclass
class WaveSpaceState:
  """Container for wave-space parameters, precomputed modes, and cached callables.

  Instances of :class:`WaveSpaceState` are produced by the ``build_*`` helpers
  (e.g., :func:`build_Mw_state` / :func:`build_Mw`) and are used to store the
  precomputed FFT-grid tensors (``modes``) and convenient, cached callables for
  applying the wave-space mobility or drawing Brownian samples.

  Fields
  ------
  params : WaveSpaceParams
      Input parameters used to build the wave-space modes (cell matrix, particle
      radius, SE parameters, grid size, etc.). This records how the ``modes``
      dictionary was constructed and whether positions are treated as
      fractional coordinates.
  modes : dict
      Modes mapping produced by :func:`build_wave_modes`. This dictionary
      contains grid metadata (``QX``, ``QY``, ``QZ``, ``k``, ``K``, ``K2``),
      the deconvolution window, and the precomputed tensors used by the operator
      (``Bfluid``, ``Bhalf``, and ``Pshape``). See :func:`build_wave_modes` for
      exact entries.
  apply_fn : Optional[Callable]
      Cached deterministic matvec ``Mw(positions, forces)`` suitable for JIT
      compilation. Stored as a static field to avoid JAX tracing over it.
  sqrt_fn : Optional[Callable]
      Cached function that returns a stochastic square root sampler for the
      wave-space Brownian contribution (if built). Static by design.
  fused_fn : Optional[Callable]
      Cached function that simultaneously applies the deterministic matvec and
      draws Brownian samples (if built). Also static.

  Notes
  -----
  - The ``modes`` dictionary is the canonical representation of all precomputed
    wave-space tensors; building new modes with different input parameters must
    be reflected by a new ``WaveSpaceState``.
  - The callable fields are optional and can be attached when instantiating a
    state with the relevant build helpers (e.g., ``attach_sqrt=True``).
  """

  params: WaveSpaceParams
  modes: dict
  apply_fn: Optional[Callable] = dataclasses.field(default=None, metadata={'static': True})
  sqrt_fn: Optional[Callable] = dataclasses.field(default=None, metadata={'static': True})
  fused_fn: Optional[Callable] = dataclasses.field(default=None, metadata={'static': True})

  @property
  def fractional_coordinates(self) -> bool:
    """Whether positions are interpreted as fractional coordinates."""
    return self.params.fractional_coordinates


def _build_wave_state(
  template: WaveSpaceState,
  *,
  attach_sqrt: bool = False,
  attach_fused: bool = False,
) -> WaveSpaceState:
  """Package precomputed modes with the relevant apply/sampling callables."""
  apply_fn = template.apply_fn or make_wave_matvec(template)

  sqrt_fn = template.sqrt_fn
  fused_fn = template.fused_fn
  if attach_sqrt or attach_fused:
    from jax_md.hydro import rpy_wave_stoch
    if attach_sqrt and sqrt_fn is None:
      sqrt_fn = rpy_wave_stoch.build_Mw_sqrt_sampler(template)
    if attach_fused and fused_fn is None:
      fused_fn = rpy_wave_stoch.build_Mw_apply_and_sample(template)

  return WaveSpaceState(
      params=template.params,
      modes=template.modes,
      apply_fn=apply_fn,
      sqrt_fn=sqrt_fn,
      fused_fn=fused_fn,
  )


def _resolve_apply_fn(state: WaveSpaceState) -> Callable:
  """Return an apply_fn from state or build one if missing."""
  if state.apply_fn is not None:
    return state.apply_fn
  return make_wave_matvec(state)


def build_wave_modes(A,
                     a,
                     xi,
                     eta,
                     Mx,
                     My,
                     Mz,
                     P_support,
                     theta=None,
                     *,
                     fractional_coordinates: bool = True):
    """Precompute P†BP tensors and metadata on the FFT grid.

    Parameters
    ----------
    A : array_like (3,3)
        Periodic cell matrix in real units.
    a : float
        Sphere radius.
    xi : float
        Ewald splitting parameter.
    eta : float
        Fluid viscosity.
    Mx, My, Mz : int
        Grid dimensions along each axis.
    P_support : int
        Gaussian stencil support size.
    theta : float, optional
        SE window parameter; auto-chosen if None.
    fractional_coordinates : bool, optional
        Whether input positions for the resulting operators are fractional (default True).

    Returns
    -------
    WaveSpaceState
        WaveSpaceState containing grid metadata and the precomputed Pshape, Bfluid, and Bhalf tensors
        (accessible under ``state.modes``).
    """

    # Prepare parameters
    A = jnp.asarray(A, dtype=REAL_DTYPE)
    if A.shape != (3, 3):
        raise ValueError(f"A must have shape (3, 3); got {A.shape}.")
    a = float(a)
    xi = float(xi)
    eta = float(eta)
    Mx, My, Mz, P_support = int(Mx), int(My), int(Mz), int(P_support)
    if Mx <= 0 or My <= 0 or Mz <= 0:
        raise ValueError(f"Grid sizes must be positive; got Mx={Mx}, My={My}, Mz={Mz}.")
    if P_support <= 0:
        raise ValueError(f"P_support must be positive; got P_support={P_support}.")
    min_M = min(Mx, My, Mz)
    if P_support > min_M:
        raise ValueError(
            f"P_support={P_support} exceeds the smallest grid dimension min(Mx,My,Mz)={min_M}. "
            "Choose P_support <= min(Mx, My, Mz) (typical values are 6–16)."
        )
    if theta is None:
        M_eff = (Mx + My + Mz) / 3.0
        theta = float(choose_theta(P_support, xi, M_eff))
    theta = float(theta)
    alpha = float(se_alpha(xi, theta))

    # Build grid and modes
    grid = _build_mode_grid(A, Mx, My, Mz)
    deconv = _build_se_deconv_window(Mx, My, Mz, P_support, alpha)
    Bfluid, Bhalf = build_B_modes(grid["k"], grid["K"], grid["K2"], xi, eta, grid["V"], deconv)
    Pshape = build_P_modes(grid["K"], a)

    modes = dict(
        QX=grid["QX"], QY=grid["QY"], QZ=grid["QZ"],
        k=grid["k"], K=grid["K"], K2=grid["K2"],
        deconv=deconv,
        Pshape=Pshape,
        Bfluid=Bfluid,
        Bhalf=Bhalf,
    )
    params = WaveSpaceParams(
        A=A,
        a=a,
        xi=xi,
        eta=eta,
        Mx=Mx,
        My=My,
        Mz=Mz,
        P=P_support,
        theta=theta,
        alpha=alpha,
        volume=float(grid["V"]),
        fractional_coordinates=fractional_coordinates,
    )
    return WaveSpaceState(params=params, modes=modes)


# ----------------------------
# Core deterministic operator
# ----------------------------
def make_wave_matvec(state: WaveSpaceState) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return Mw(positions, forces) using precomputed modes.

    Parameters
    ----------
    state : WaveSpaceState
        WaveSpaceState carrying precomputed modes and coordinate convention.

    Returns
    -------
    Callable
        JIT-compiled Mw(positions, forces) -> velocities.
    """
    modes = state.modes
    params = state.params
    fractional_coordinates = params.fractional_coordinates
    Mx, My, Mz, P = params.Mx, params.My, params.Mz, params.P
    alpha = params.alpha
    Bfluid = modes["Bfluid"]
    Pshape = modes["Pshape"]
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(params.volume, dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box

    base_A = jnp.asarray(params.A, dtype=REAL_DTYPE)
    base_inv = jnp.linalg.inv(base_A)

    @jax.jit
    def Mw_core(positions, forces, current_box=None, transform=None):
        """Apply Mw using the precomputed wave-space modes.

        Parameters
        ----------
        positions : array
            Particle coordinates (fractional if ``fractional_coordinates=True``).
        forces : array
            Particle forces.
        current_box : array, optional
            Real-space box at this step; positions are mapped into the base grid frame.
        transform : array, optional
            Precomputed transform ``T = A_base^{-1} @ A_current``; overrides ``current_box`` if set.

        Returns
        -------
        jnp.ndarray
            Particle velocities scaled to real units.
        """
        A_curr = base_A if current_box is None else current_box
        if transform is None:
            transform = jnp.eye(3, dtype=REAL_DTYPE) if current_box is None else base_inv @ A_curr

        positions_frac_curr = positions_to_fractional(positions, A_curr, fractional_coordinates)
        positions_frac = jnp.mod(positions_frac_curr @ transform.T, 1.0)
        forces = jnp.asarray(forces, dtype=REAL_DTYPE)

        st = build_stencils_frac(positions_frac, Mx, My, Mz, P, alpha)
        force_grid = spread(forces, st, Mx, My, Mz)
        force_grid = sigma_inv * force_grid
        force_q = fft_vec(force_grid)

        P_force_q = Pshape[..., None] * force_q
        BP_force_q = jnp.einsum('...ij,...j->...i', Bfluid, P_force_q)
        Uq = Pshape[..., None] * BP_force_q

        u_grid = ifft_vec(Uq)
        velocities = gather(u_grid, st, Mx, My, Mz)
        return V_box * velocities

    return Mw_core


def build_Mw(A,
             a,
             xi,
             eta,
             Mx,
             My,
             Mz,
             P_support,
             *,
             theta=None,
             fractional_coordinates: bool = True,
             attach_sqrt: bool = False,
             attach_fused: bool = False):
  """Construct init/apply pair mirroring the real-space builder style.

  Returns
  -------
  init_fn, apply_fn : Callable
      ``init_fn() -> WaveSpaceState`` builds and stores wave-space parameters,
      modes, and cached callables. ``apply_fn(state, positions, forces, ...)``
      applies M^(w) using the provided state and returns (velocities, state).
  """
  base_state = build_wave_modes(
      A,
      a,
      xi,
      eta,
      Mx,
      My,
      Mz,
      P_support,
      theta,
      fractional_coordinates=fractional_coordinates,
  )

  def init_fn():
    return _build_wave_state(
        base_state,
        attach_sqrt=attach_sqrt,
        attach_fused=attach_fused,
    )

  def apply_fn(state: WaveSpaceState, positions, forces, *, current_box=None, transform=None):
    if not isinstance(state, WaveSpaceState):
      raise ValueError("Wave-space state must be a WaveSpaceState instance.")
    apply_core = _resolve_apply_fn(state)
    velocities = apply_core(
        positions, forces, current_box=current_box, transform=transform
    )
    return velocities, state

  return init_fn, apply_fn


def build_Mw_apply(state: WaveSpaceState):
  """Return Mw(positions, forces) from a WaveSpaceState."""
  if not isinstance(state, WaveSpaceState):
    raise ValueError("build_Mw_apply requires a WaveSpaceState.")
  return _resolve_apply_fn(state)


def build_Mw_state(A, a, xi, eta, Mx, My, Mz, P_support, *,
                   theta=None, fractional_coordinates=True,
                   attach_sqrt=False, attach_fused=False):
  """Construct a WaveSpaceState with explicit parameters and cached operators."""
  base_state = build_wave_modes(
      A,
      a,
      xi,
      eta,
      Mx,
      My,
      Mz,
      P_support,
      theta,
      fractional_coordinates=fractional_coordinates,
  )
  return _build_wave_state(
      base_state,
      attach_sqrt=attach_sqrt,
      attach_fused=attach_fused,
  )

def mw_matvec(state: WaveSpaceState,
              positions: jnp.ndarray,
              vec: jnp.ndarray,
              *,
              current_box=None,
              transform=None) -> jnp.ndarray:
  """Apply the wave-space mobility using an existing state.

  Note: the underlying `state.apply_fn` / `make_wave_matvec(state)` is already
  JIT-compiled. This wrapper is intentionally *not* decorated with `jax.jit`
  because `WaveSpaceState` contains non-array metadata (grid sizes, etc.) that
  must remain static; users who want a fully-jitted matvec should capture
  `state` in a closure and JIT that closure.
  """
  if not isinstance(state, WaveSpaceState):
    raise ValueError("mw_matvec expects a WaveSpaceState.")
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  vec = jnp.asarray(vec, dtype=REAL_DTYPE)
  apply_fn = _resolve_apply_fn(state)
  return apply_fn(positions, vec, current_box=current_box, transform=transform)


# ----------------------------
# Brute-force reference (for testing)
# ----------------------------
def _mw_bruteforce(t, F, A, a, xi, eta, Mx, My, Mz, P, theta=None):
    """Direct k-space sum using the FFT mode set (slow; testing only).

    Parameters
    ----------
    t : array
        Fractional particle coordinates with shape (N, 3).
    F : array
        Particle forces with shape (N, 3).
    A : array_like
        Periodic cell matrix in real units.
    a : float
        Sphere radius.
    xi : float
        Ewald splitting parameter.
    eta : float
        Fluid viscosity.
    Mx, My, Mz : int
        Grid dimensions along each axis.
    P : int
        Stencil support size (kept for interface consistency).
    theta : float, optional
        SE window parameter; auto-chosen if None.

    Returns
    -------
    jnp.ndarray
        Velocities from the explicit k-space evaluation.
    """
    Brecip = make_reciprocal(A)
    V = jnp.linalg.det(A)
    QX, QY, QZ = q_grid(Mx, My, Mz)
    q = jnp.stack([QX, QY, QZ], axis=-1).reshape(-1, 3)
    k = jnp.einsum('ab,qb->qa', Brecip, q)
    K2 = jnp.sum(k * k, axis=1)
    K = jnp.sqrt(jnp.maximum(K2, 0.0))

    tH = (K / (2.0 * xi)) ** 2
    H = (1.0 + tH) * jnp.exp(-tH)
    ka = K * a

    rpy = jnp.where(K2 > 0.0, sinc(ka) ** 2 / K2, 0.0)
    scal = jnp.where(K2 > 0.0, (H * rpy) / (eta * V), 0.0)

    I = jnp.eye(3)
    denom = jnp.where(K2 > 0.0, jnp.sqrt(K2), 1.0)
    kh = k / denom[:, None]
    Pkk = I - kh[..., None] * kh[:, None, :]
    Gk = scal[:, None, None] * Pkk

    phase = jnp.exp(2j * jnp.pi * (t @ q.T))
    phi_ij = phase[:, None, :] * jnp.conj(phase)[None, :, :]
    W_qj = jnp.einsum('qij,nj->qni', Gk, F)
    U = jnp.einsum('ijq,qjv->iv', phi_ij, W_qj).real
    return U
