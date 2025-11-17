# Combined Pse mobility operator: M = M^(r) + M^(w)
# Provides high-level interface for building complete hydrodynamic mobility
#
# USAGE WITH JAX-MD SPACES:
# ==========================
# The Pse mobility operator works with JAX-MD's space functions:
#
# 1. Static periodic box (periodic_general):
#    >>> space_fns = space.periodic_general(A, fractional_coordinates=True)
#    >>> init_fn, apply_fn = pse.build_pse_mobility(space_fns, a=0.03, xi=10.0, eta=1.0, P=16, Mgrid=64)
#    >>> state = init_fn(positions_fractional)
#    >>> velocities, state = apply_fn(state, positions_fractional, forces)
#
# 2. Shearing box for NEMD/rheology (shearing):
#    >>> space_fns = space.shearing(A, shear_fn=lambda t: gamma_dot * t, fractional_coordinates=True)
#    >>> init_fn, apply_fn = pse.build_pse_mobility(space_fns, a=0.03, xi=10.0, eta=1.0, P=16, Mgrid=64)
#    >>> state = init_fn(positions_fractional, t=0.0)
#    >>> velocities, state = apply_fn(state, positions_fractional, forces, t=current_time)
#
# 3. Direct box specification:
#    >>> init_fn, apply_fn = pse.build_pse_mobility_direct(A, a=0.03, xi=10.0, eta=1.0, P=16, Mgrid=64)
#    >>> state = init_fn(positions_fractional)
#    >>> velocities, state = apply_fn(state, positions_fractional, forces)

import jax
import jax.numpy as jnp
from math import ceil, sqrt, pi, log, erfc
import numpy as _py_np
from typing import Callable, Dict, Optional, Tuple
import warnings

from jax import config as jax_config
from jax_md import dataclasses, space
from jax_md.hydro.pse_wave import (
    build_B_modes, build_Mw_apply, build_Mw_apply_and_sample, build_Mw_sqrt_sampler
)
from jax_md.hydro.pse_real import (
    RealSpaceState, build_Mr_apply, sample_mr_sqrt_precond, _current_box_matrix,
    jacobi_from_self, identity_preconditioner, Preconditioner
)

REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32


@dataclasses.dataclass
class PseState:
    real: RealSpaceState
    mw_apply: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = dataclasses.field(
        default=None, metadata={'static': True}
    )
    mw_sqrt_sampler: Optional[Callable[[jax.Array, jnp.ndarray], jnp.ndarray]] = dataclasses.field(
        default=None, metadata={'static': True}
    )
    mw_apply_and_sample: Optional[
        Callable[[jax.Array, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
    ] = dataclasses.field(default=None, metadata={'static': True})
    wave_cfg: Optional[Dict[str, jnp.ndarray]] = None
    preconditioner: Optional[Preconditioner] = dataclasses.field(
        default=None, metadata={'static': True}
    )


# ================================================================
# Combined mobility operator: M = M^(r) + M^(w)
# ================================================================

def build_pse_mobility_direct(A, a, xi, eta,
                                 Mr_params=None, Mw_params=None,
                                 rcut=None,
                                 P=None, Mgrid=None,
                                 theta=None,
                                 real_space_first=True,
                                 preconditioner=None):
    """Convenience wrapper that constructs a periodic space from a box matrix."""

    space_fns = space.periodic_general(A, fractional_coordinates=True)
    return build_pse_mobility(
        space_fns,
        a,
        xi,
        eta,
        Mr_params=Mr_params,
        Mw_params=Mw_params,
        rcut=rcut,
        P=P,
        Mgrid=Mgrid,
        theta=theta,
        real_space_first=real_space_first,
        preconditioner=preconditioner,
    )


def build_pse_mobility(space_fns, a, xi, eta,
                          Mr_params=None, Mw_params=None,
                          rcut=None,
                          P=None, Mgrid=None,
                          theta=None,
                          real_space_first=True,
                          preconditioner=None):
    """Construct total Pse mobility using fractional-coordinate space functions."""

    if len(space_fns) < 2:
        raise ValueError("space_fns must contain at least displacement and shift functions.")

    displacement_fn, _ = space_fns[:2]
    box_fn = space_fns[2] if len(space_fns) > 2 else None
    has_box_fn = box_fn is not None

    mr_kwargs = dict(Mr_params or {})
    box_kwargs_default = dict(mr_kwargs.pop("box_kwargs", {}))
    rcut_override = mr_kwargs.pop("rcut", None)

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
    use_fused_wave = bool(mw_kwargs.pop("fused_wave", mw_kwargs.pop("fused", False)))
    Mx = mw_kwargs.get("M", Mgrid)
    My = mw_kwargs.get("M", Mgrid)
    Mz = mw_kwargs.get("M", Mgrid)
    P_ = mw_kwargs.get("P", P)
    theta_ = mw_kwargs.get("theta", theta)

    # Set up preconditioner (default to Jacobi based on self-mobility)
    if preconditioner is None:
        # Compute self-mobility coefficient from Fiore formula
        # M_self ≈ 1/(6πηa) for Stokes drag
        self_coeff = 1.0 / (6.0 * jnp.pi * eta * a)
        precond = jacobi_from_self(self_coeff)
    else:
        precond = preconditioner

    def _base_box_kwargs(dim, kwargs):
        """Replace shear arguments with zeros to recover the base box."""
        base_kwargs = dict(kwargs)
        base_kwargs.pop("gamma", None)
        base_kwargs.pop("gamma_xy", None)
        base_kwargs.pop("gamma_xz", None)
        base_kwargs.pop("gamma_yz", None)
        if dim >= 3:
            base_kwargs["gamma"] = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
        elif dim >= 2:
            base_kwargs["gamma"] = 0.0
        return base_kwargs

    def _map_positions_to_base(cfg, current_box, positions_frac):
        if not has_box_fn or current_box is None:
            dim = positions_frac.shape[-1]
            transform = jnp.eye(dim, dtype=positions_frac.dtype)
            return positions_frac, transform
        base_inv = cfg.get("A_inv")
        if base_inv is None:
            base_inv = jnp.linalg.inv(cfg["A"])
        transform = base_inv @ current_box
        mapped = positions_frac @ transform.T
        mapped = jnp.mod(mapped, 1.0)
        return mapped, transform

    def init_fn(positions_frac, **kwargs):
        positions_frac = jnp.asarray(positions_frac)
        combined_kwargs = dict(box_kwargs_default)
        combined_kwargs.update(kwargs)

        dim = int(positions_frac.shape[1])
        if has_box_fn:
            current_box = _current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
            base_kwargs = _base_box_kwargs(dim, combined_kwargs)
            base_box = _current_box_matrix(displacement_fn, box_fn, dim, **base_kwargs)
        else:
            base_box = _current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
            current_box = base_box

        L_min = float(jnp.min(jnp.linalg.norm(base_box, axis=0)))
        if rcut_value > 0.5 * L_min:
            warnings.warn(
                (
                    f"Real-space cutoff rcut={rcut_value:.2f} exceeds half the minimum box "
                    f"dimension ({0.5 * L_min:.2f}). Consider increasing xi={xi:.2f} and compensating "
                    "with a finer wave-space grid."
                ),
                UserWarning,
                stacklevel=2,
            )

        real_state = Mr_init(positions_frac, **combined_kwargs)

        cfg = build_B_modes(
            base_box,
            a,
            xi,
            eta,
            Mx,
            My,
            Mz,
            P_,
            theta=theta_,
        )

        cfg = dict(cfg)
        cfg["A_inv"] = jnp.linalg.inv(base_box)
        if has_box_fn:
            cfg["last_transform"] = cfg["A_inv"] @ current_box
        else:
            cfg["last_transform"] = jnp.eye(dim, dtype=positions_frac.dtype)

        mw_apply = build_Mw_apply(cfg)
        mw_sqrt = build_Mw_sqrt_sampler(cfg)
        mw_fused = build_Mw_apply_and_sample(cfg) if use_fused_wave else None

        return PseState(
            real=real_state,
            mw_apply=mw_apply,
            mw_sqrt_sampler=mw_sqrt,
            mw_apply_and_sample=mw_fused,
            wave_cfg=cfg,
            preconditioner=precond,
        )

    def apply_fn(state: PseState, positions_frac, forces, **kwargs):
        positions_frac = jnp.asarray(positions_frac)
        forces = jnp.asarray(forces)

        combined_kwargs = dict(box_kwargs_default)
        combined_kwargs.update(kwargs)

        if has_box_fn:
            dim = int(positions_frac.shape[1])
            current_box = _current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
        else:
            current_box = None

        Ur, real_state = Mr_apply(state.real, positions_frac, forces, **combined_kwargs)

        cfg = state.wave_cfg
        if cfg is None or state.mw_apply is None:
            raise ValueError("wave-space operator missing from state; run init_fn first.")

        if has_box_fn:
            positions_wave, transform = _map_positions_to_base(cfg, current_box, positions_frac)
            wave_cfg = dict(cfg)
            wave_cfg["last_transform"] = transform
        else:
            positions_wave = positions_frac
            wave_cfg = cfg

        Uw = state.mw_apply(positions_wave, forces)
        mw_apply = state.mw_apply
        mw_sqrt = state.mw_sqrt_sampler
        mw_fused = state.mw_apply_and_sample if use_fused_wave else None

        total = Ur + Uw if real_space_first else Uw + Ur
        return total, PseState(
            real=real_state,
            mw_apply=mw_apply,
            mw_sqrt_sampler=mw_sqrt,
            mw_apply_and_sample=mw_fused,
            wave_cfg=wave_cfg,
            preconditioner=state.preconditioner,
        )

    def apply_with_brownian(state: PseState,
                             positions_frac,
                             forces,
                             key,
                             *,
                             kT: float,
                             dt: float,
                             mr_iters: int = 10,
                             **kwargs):
        positions_frac = jnp.asarray(positions_frac)
        forces = jnp.asarray(forces)

        combined_kwargs = dict(box_kwargs_default)
        combined_kwargs.update(kwargs)

        key_real, key_wave = jax.random.split(key)

        if has_box_fn:
            dim = int(positions_frac.shape[1])
            current_box = _current_box_matrix(displacement_fn, box_fn, dim, **combined_kwargs)
        else:
            current_box = None

        Ur, real_state = Mr_apply(state.real, positions_frac, forces, **combined_kwargs)

        cfg = state.wave_cfg
        if cfg is None or state.mw_apply is None or state.mw_sqrt_sampler is None:
            raise ValueError("wave-space operator missing from state; run init_fn first.")

        if has_box_fn:
            positions_wave, transform = _map_positions_to_base(cfg, current_box, positions_frac)
            wave_cfg_next = dict(cfg)
            wave_cfg_next["last_transform"] = transform
        else:
            positions_wave = positions_frac
            wave_cfg_next = cfg

        if use_fused_wave and state.mw_apply_and_sample is not None:
            Uw, wave_noise = state.mw_apply_and_sample(key_wave, positions_wave, forces)
            mw_fused_next = state.mw_apply_and_sample
        else:
            Uw = state.mw_apply(positions_wave, forces)
            wave_noise = state.mw_sqrt_sampler(key_wave, positions_wave)
            mw_fused_next = state.mw_apply_and_sample if use_fused_wave else None
        mw_apply_next = state.mw_apply
        mw_sqrt_next = state.mw_sqrt_sampler

        # Draw stochastic increments from each split (both in real coordinates).
        real_noise_real = sample_mr_sqrt_precond(
            key_real, real_state, positions_frac, 
            precond=precond, iters=mr_iters
        )
        total_noise = jnp.sqrt(2.0 * kT * dt) * (real_noise_real + wave_noise)

        total_velocity = Ur + Uw if real_space_first else Uw + Ur

        next_state = PseState(
            real=real_state,
            mw_apply=mw_apply_next,
            mw_sqrt_sampler=mw_sqrt_next,
            mw_apply_and_sample=mw_fused_next if use_fused_wave else None,
            wave_cfg=wave_cfg_next,
            preconditioner=state.preconditioner,
        )

        return total_velocity, total_noise, next_state

    apply_fn.with_brownian = apply_with_brownian

    return init_fn, apply_fn


def _wave_sqrt_sample(state: PseState,
                      positions_frac: jnp.ndarray,
                      key: jax.Array) -> jnp.ndarray:
    """Helper to draw wave-space stochastic increments."""
    cfg = state.wave_cfg
    positions_base = positions_frac
    if cfg is not None:
        transform = None
        if isinstance(cfg, dict):
            transform = cfg.get("last_transform", None)
        if transform is not None:
            positions_base = jnp.mod(positions_frac @ transform.T, 1.0)
    if state.mw_sqrt_sampler is not None:
        return state.mw_sqrt_sampler(key, positions_base)
    if cfg is None:
        raise ValueError("wave_cfg missing; initialize with build_pse_mobility first.")
    sampler = build_Mw_sqrt_sampler(cfg)
    return sampler(key, positions_base)


def brownian_increment(key: jax.Array,
                       state: PseState,
                       positions_frac: jnp.ndarray,
                       *,
                       kT: float,
                       dt: float,
                       mr_iters: int = 10) -> jnp.ndarray:
    """
    Generate the total Brownian increment for the Pse-split mobility.

    Parameters
    ----------
    key : PRNGKey
        Random key for generating Gaussian noise.
    state : PseState
        Current Pse state (after evaluating deterministic velocities).
    positions_frac : (N,3) array
        Fractional coordinates aligned with ``state``.
    kT : float
        Thermal energy ``k_B T``.
    dt : float
        Time step.
    mr_iters : int, optional
        Lanczos iterations for the real-space square root (default 10).

    Returns
    -------
    (N,3) array
        Brownian displacement increment.
    """
    key_r, key_w = jax.random.split(key)
    # Both samplers emit increments in real coordinates compatible with the shift function.
    precond = state.preconditioner if state.preconditioner is not None else identity_preconditioner()
    real_sample_real = sample_mr_sqrt_precond(
        key_r, state.real, positions_frac, 
        precond=precond, iters=mr_iters
    )
    wave_sample_real = _wave_sqrt_sample(state, positions_frac, key_w)

    scale = jnp.sqrt(2.0 * kT * dt)
    return scale * (real_sample_real + wave_sample_real)


def build_euler_maruyama_step(apply_fn: Callable,
                              shift_fn: Callable,
                              *,
                              mr_iters: int = 10):
    """
    Construct an Euler-Maruyama integrator using the Pse mobility.

    Parameters
    ----------
    apply_fn : callable
        Deterministic mobility apply returned by ``build_pse_mobility``.
    shift_fn : callable
        Shift function from the space definition (e.g. ``space_fns[1]``).
    mr_iters : int, optional
        Lanczos iterations for the real-space square root.

    Returns
    -------
    function
        Stepper ``step(key, positions_frac, forces, state, *, dt, kT, **kwargs)``.
    """

    def step(key: jax.Array,
             positions_frac: jnp.ndarray,
             forces: jnp.ndarray,
             state: PseState,
             *,
             dt: float,
             kT: float,
             **kwargs):
        # Deterministic velocity in REAL units
        velocities, next_state = apply_fn(state, positions_frac, forces, **kwargs)
        # Stochastic increment in REAL units
        dX_B = brownian_increment(key, next_state, positions_frac, kT=kT, dt=dt, mr_iters=mr_iters)

        # Total REAL displacement
        disp_real = dt * velocities + dX_B

        # Convert REAL → FRACTIONAL using the current box matrix
        A = next_state.real.box_matrix        # shape (3,3)
        # fractional displacement = A^{-1} * disp_real
        disp_frac = jnp.linalg.solve(A, disp_real.T).T

        # Now update fractional positions
        next_positions = shift_fn(positions_frac, disp_frac, **kwargs)
        return next_positions, next_state

    return step


# ================================================================
# Parameter suggestion helpers (Fiore-style error budgets)
# ================================================================

def _quad_error_bound(P, m=None, A_max=1.0):
    """
    Spectral-Pse quadrature bound (standard SE literature form):
      ε_q  ≲  exp(-π² P² / (2 m²) · A_max) + erfc(m · √(A_max/2))
    
    where:
      - P: stencil support (number of grid points per dimension)
      - m: Gaussian width parameter (controls window shape)
      - A_max: cell deformation factor (max eigenvalue of A^T A)
    
    Standard recipe: choose m ≈ √(π P) for near-optimal balance, which gives:
      ε_q  ≲  exp(-π P / 2 · A_max) + erfc(√(π P A_max / 2))
    
    This form makes the m parameter explicit, enabling:
      1. Exploration of alternative m(P) recipes for different geometries
      2. Proper accounting of the P² dependence in the Gaussian term
      3. Independent tuning of the two error contributions
    
    The explicit m parameter is essential for understanding convergence:
    the Gaussian term exp(-π² P² / (2 m²)) shows quadratic decay in P
    (for fixed m), not linear as the simplified form might suggest.
    
    References
    ----------
    Lindbo & Tornberg (2011), "Spectral accuracy in fast Pse methods"
    Fiore et al. (2017), "Fast Stokesian Dynamics"
    
    Parameters
    ----------
    P : float
        Stencil support size
    m : float, optional
        Gaussian width. If None, uses m = √(π P) (standard choice).
    A_max : float, optional
        Cell deformation factor (default 1.0 for cubic cells)
        
    Returns
    -------
    float
        Upper bound on quadrature error
    """
    from math import erfc
    if m is None:
        m = sqrt(pi * P)
    
    # Standard SE bound with both terms (explicit P² dependence)
    gaussian_term = _py_np.exp(-((pi * pi) * (P * P) / (2.0 * (m * m))) * A_max)
    tail_term = erfc(m * sqrt(A_max / 2.0))
    return gaussian_term + tail_term


def _min_P_for_tolerance(eps_q_target, P_min=6, P_max=64, A_max=1.0, m_recipe=None):
    """
    Find smallest even P satisfying the SE quadrature bound ε_q ≤ target.
    
    Uses the standard spectral-Pse bound:
      ε_q ≲ exp(-π² P² / (2 m²) · A_max) + erfc(m · √(A_max/2))
    
    with Gaussian width m chosen by m_recipe (default: m = √(π P)).
    
    Also enforces stability guard on SE deconvolution: with our choose_theta,
    the exponent at Nyquist is ≈ (3/4) P; we cap it at ~22 to avoid blow-up.
    
    Parameters
    ----------
    eps_q_target : float
        Target quadrature error tolerance
    P_min : int, optional
        Minimum stencil size to consider (default 6)
    P_max : int, optional
        Maximum stencil size to consider (default 64)
    A_max : float, optional
        Cell deformation factor (max eigenvalue of A^T A, default 1.0)
    m_recipe : callable, optional
        Function m = m_recipe(P) to determine Gaussian width.
        If None, uses m = √(π P).
        
    Returns
    -------
    int
        Smallest even P satisfying the bound, or P_max if none found
    """
    if m_recipe is None:
        m_recipe = lambda P: sqrt(pi * P)
    
    EXP_CAP = 22.0   # conservative cap; exp(22) ≈ 3.6e9 (avoid bigger)
    P_cap   = int(_py_np.floor((4.0/3.0) * EXP_CAP))
    hard_max = min(P_max, P_cap)
    best = None
    for P in range(P_min, hard_max+1, 2):  # even steps keep stencils symmetric
        m = m_recipe(P)
        if _quad_error_bound(P, m=m, A_max=A_max) <= eps_q_target:
            best = P
            break
    if best is None:
        best = hard_max  # last resort
    return best


def _min_M_for_wave_trunc(xi, eps_w_target, A=None):
    """
    Choose M (cubic grid) so that ε_W ~ exp(-k_cut^2 / (4 ξ^2)) ≤ target.
    In fractional coords with A=I, a simple k_cut ≈ π M works well for
    your test harness (you compare on the same FFT mode box).
    For general triclinic A, you can replace π M with the smallest singular
    value of (B = 2π A^{-T}) times (M/2)√3 to be more precise.
    """
    if eps_w_target <= 0:
        return 16  # default minimum
    # crude but effective: k_cut ≈ π M  →  exp( - (π M)^2 / (4 ξ^2) ) ≤ eps_w
    kfac = pi
    M_min = ceil((2.0 * xi / kfac) * sqrt(max(0.0, _py_np.log(1.0/eps_w_target))))
    return int(max(8, M_min))


def estimate_spectral_pse_params_fiore(
    tol,
    A,
    a,
    N,
    phi,
    *,
    df=3.0,
    n_iter=4.0,
    C_R=1.0,
    C_W=1.0,
    C_Q=1.0,
    eps_split=None,
    xi_override=None,
    xi_growth=1.25,
    max_growth_steps=5,
    P_bounds=(6, 64),
    M_min=16,
    notes=True,
):
    """Implement Fiore's Spectral-Pse parameter workflow.

    The routine mirrors the straight "Fiore-style" recipe summarised in the
    thesis / JCP paper:

      1. Split the total tolerance ``tol`` into independent budgets
         (ε_R, ε_W, ε_q).
      2. Choose ξ from Fiore's cost balance (or via ``xi_override``).
      3. Solve ε_R ≲ exp(-ξ² r_cut²) for ``r_cut`` and enforce the minimum-image
         guard ``r_cut ≤ L_min/2`` by increasing ξ as needed.
      4. Solve ε_W ≲ exp(-k_cut²/(4 ξ²)) for ``k_cut`` and map it to an FFT grid
         size using the smallest Nyquist frequency of ``A``.
      5. Solve the SE quadrature bound with ``m = √(π P)`` for the smallest
         even stencil size ``P`` (adjusted by the deformation factor A_max).
      6. Map the quadrature choice to ``θ``/``α`` using the existing helpers.
      7. Return the full parameter set together with the cost-model components
         ``C_R``, ``C_W``, ``C_Q`` evaluated at the chosen parameters.

    Parameters
    ----------
    tol : float
        Target total tolerance ``ε``.
    A : (3, 3) array_like
        Periodic cell matrix (columns are lattice vectors).
    a : float
        Particle radius.
    N : int
        Number of particles.
    phi : float
        Volume fraction (0 < φ ≤ 1 typically).
    df : float, optional
        Fractal dimension of the configuration (Fiore recommends 3 for dense 3D).
    n_iter : float, optional
        Iteration count for the real-space solver inside the cost model.
    C_R, C_W, C_Q : float, optional
        Implementation-specific cost constants for the real-, wave-, and
        quadrature-space work, respectively.
    eps_split : tuple of 3 floats, optional
        Explicit budgets for (ε_R, ε_W, ε_q). Defaults to equal shares.
    xi_override : float, optional
        Use this ξ instead of the cost-model estimate (still subject to the
        real-space guard.
    xi_growth : float, optional
        Multiplicative factor applied when enforcing ``r_cut ≤ L_min/2``.
    max_growth_steps : int, optional
        Maximum number of ξ adjustments when the guard is violated.
    P_bounds : (int, int), optional
        Minimum/maximum even stencil size to consider.
    M_min : int, optional
        Lower bound on FFT grid size per axis.
    notes : bool, optional
        Include a human-readable notes string in the result.

    Returns
    -------
    dict
        Contains ξ, r_cut, k_cut, grid size M, stencil size P, window m, θ, α,
        the individual tolerance budgets, and a decomposition of Fiore's cost
        model evaluated at the chosen parameters.
    """

    if tol <= 0:
        raise ValueError("tol must be positive.")
    if N <= 0:
        raise ValueError("N must be positive.")
    if phi <= 0:
        raise ValueError("phi must be positive.")

    A = _py_np.asarray(A, dtype=_py_np.float64)
    if A.shape != (3, 3):
        raise ValueError("A must be a 3x3 cell matrix.")

    if eps_split is None:
        eps_R = eps_W = eps_q = float(tol) / 3.0
    else:
        if len(eps_split) != 3:
            raise ValueError("eps_split must contain three entries (ε_R, ε_W, ε_q).")
        eps_R, eps_W, eps_q = map(float, eps_split)
    if eps_R <= 0 or eps_W <= 0 or eps_q <= 0:
        raise ValueError("Tolerance budgets must be positive.")
    for name, budget in (("eps_R", eps_R), ("eps_W", eps_W), ("eps_q", eps_q)):
        if budget >= 1.0:
            raise ValueError(f"{name} must be < 1.0 to keep the exponential bounds meaningful.")

    # Singular values capture the cell geometry: min ↔ shortest Nyquist, max ↔ deformation.
    try:
        # B = 2π A^{-T}
        B = 2.0 * pi * _py_np.linalg.inv(A).T
    except _py_np.linalg.LinAlgError as exc:
        raise ValueError("Cell matrix A must be invertible.") from exc

    sigma_B = _py_np.linalg.svd(B, compute_uv=False)
    if sigma_B.min() <= 0:
        raise ValueError("Reciprocal lattice singular values must be positive.")
    sigma_min_B = float(sigma_B.min())

    sigma_A = _py_np.linalg.svd(A, compute_uv=False)
    L_min = float(sigma_A.min())
    if L_min <= 0:
        raise ValueError("Cell matrix must have positive singular values.")

    eigvals_A = _py_np.linalg.eigvalsh(A.T @ A)
    A_max = float(_py_np.max(eigvals_A))
    if not _py_np.isfinite(A_max) or A_max <= 0:
        raise ValueError("Invalid deformation factor extracted from A.")

    # Determine ξ via Fiore's asymptotic optimum unless overridden.
    if xi_override is not None:
        xi = float(xi_override)
        if xi <= 0:
            raise ValueError("xi_override must be positive.")
        xi_source = "override"
    else:
        logN = log(max(float(N), 2.0))
        denom = max(C_R * df * n_iter * (phi ** 2), 1e-30)
        ratio = (3.0 * C_W * logN) / denom
        ratio = max(ratio, 1e-30)
        xi = ratio ** (-1.0 / (df + 3.0))
        xi_source = "cost-model"

    # Minimum-image guard: ensure r_cut ≤ L_min/2 by inflating ξ if needed.
    # L_min is the shortest repeat length (σ_min of A), covering triclinic boxes.
    adjustments = []
    for _ in range(max_growth_steps + 1):
        r_cut = (1.0 / xi) * sqrt(log(1.0 / eps_R))
        if L_min > 0 and r_cut > 0.5 * L_min:
            xi *= xi_growth
            adjustments.append({"reason": "rcut_guard", "new_xi": xi})
            continue
        break
    else:
        import warnings
        warnings.warn(
            "Failed to satisfy r_cut ≤ L_min/2 after max_growth_steps; proceeding with last ξ.",
            UserWarning,
        )

    r_cut = (1.0 / xi) * sqrt(log(1.0 / eps_R))

    # Wave-space truncation and FFT grid sizing (per-axis Nyquist via σ(B)).
    k_cut = 2.0 * xi * sqrt(log(1.0 / eps_W))
    M_vec = _py_np.ceil((2.0 * k_cut) / sigma_B)
    M_vec = _py_np.maximum(M_min, M_vec)
    M_vec = _py_np.ceil(M_vec / 8.0) * 8.0
    grid_shape = tuple(int(m_val) for m_val in M_vec)

    # Quadrature bound with deformation factor A_max (use centralized helper).
    # Standard SE form: ε_q ≲ exp(-π² P² / (2 m²) · A_max) + erfc(m √(A_max/2))
    # with m = √(π P) for near-optimal balance.
    def quad_bound(P):
        return _quad_error_bound(P, m=None, A_max=A_max)

    P_min, P_max = P_bounds
    if P_min % 2 != 0:
        P_min += 1
    P_choice = None
    for P in range(int(max(P_min, 2)), int(P_max) + 1, 2):
        if quad_bound(P) <= eps_q:
            P_choice = P
            break
    if P_choice is None:
        P_choice = int(P_max)
        import warnings
        warnings.warn(
            "Quadrature tolerance not met within P_bounds; using maximum P.",
            UserWarning,
        )

    m_window = sqrt(pi * P_choice)

    # Bridge to existing SE helpers.
    from jax_md.hydro.pse_wave import choose_theta, se_alpha

    M_eff = float(sum(grid_shape)) / len(grid_shape)
    theta = float(choose_theta(P_choice, xi, M_eff))
    alpha = float(se_alpha(xi, theta))

    # Fiore cost model components for bookkeeping.
    xi_pow = xi ** (-df)
    phi_inv = 1.0 / phi
    wave_work = C_W * (xi ** 3 * phi_inv * N) * log(max(xi ** 3 * phi_inv * N, 1.0))
    real_work = C_R * n_iter * N * phi * xi_pow
    quad_work = C_Q * N * (P_choice ** 3)

    result = {
        "xi": xi,
        "xi_source": xi_source,
        "adjustments": adjustments,
        "budgets": {
            "eps_R": eps_R,
            "eps_W": eps_W,
            "eps_q": eps_q,
        },
        "rcut": r_cut,
        "kcut": k_cut,
        "grid_shape": grid_shape,
        "M": grid_shape,
        "P": P_choice,
        "m": m_window,
        "theta": theta,
        "alpha": alpha,
        "A_max": A_max,
        "sigma_min_B": sigma_min_B,
        "cost_model": {
            "real_space": real_work,
            "wave_space": wave_work,
            "quadrature": quad_work,
            "total": real_work + wave_work + quad_work,
        },
    }

    if notes:
        result["notes"] = (
            "Fiore parameter recipe: split independent exponential bounds, use ξ*"
            " from the cost balance, enforce r_cut guard (increase ξ if needed),"
            " map k_cut to the smallest Nyquist of the reciprocal lattice, and"
            " choose P,m via the SE quadrature bound with m=√(πP)."
        )

    return result


def suggest_pse_params(
    tol=1e-6,
    a=0.03,
    A=None,                 # periodic cell; if None, assume identity
    xi_candidates=None,     # list of ξ to try; if None, build from a
    target_split=(1/3, 1/3, 1/3),  # (ε_R, ε_W, ε_q) shares
    M_bounds=(16, 256),     # practical bounds
    P_bounds=(6, 64),       # practical bounds
    N_particles=1_000,      # used only for a rough cost score
):
    """
    Propose (ξ, M, P, θ, α) for target 'tol' by meeting Fiore's three
    independent error budgets and picking the cheapest candidate via a
    simple cost proxy (FFT + spread/gather).

    Returns:
      dict with xi, M, P, theta, alpha, eps_est (components), and notes.
    """
    # Split tolerance (independent budgets)
    eps_R = tol * target_split[0]
    eps_W = tol * target_split[1]
    eps_q = tol * target_split[2]

    # ξ candidates: Fiore's stochastic optimum is often around ξ a ~ 0.5
    # (paper, Fig. 4/5 discussions). Build a small scan around that.
    if xi_candidates is None:
        # Assume box units for a; keep ξ a in [0.3, 0.8]
        xis = [max(1e-6, s/(a+1e-12)) for s in (0.3, 0.4, 0.5, 0.6, 0.8)]
    else:
        xis = list(xi_candidates)

    best = None
    for xi in xis:
        # Wave truncation → M
        M = _min_M_for_wave_trunc(xi, eps_W, A=A)
        M = int(min(max(M, M_bounds[0]), M_bounds[1]))

        # Quadrature → P
        P = _min_P_for_tolerance(eps_q, P_min=P_bounds[0], P_max=P_bounds[1])

        # Real-space error target ε_R translates to r_cut ~ (1/ξ)*sqrt(log(1/ε_R)).
        # We do not build real-space here, but include its implied neighbor size
        # into a crude cost proxy via r_cut(ξ).
        rcut = (1.0/xi) * sqrt(max(0.0, _py_np.log(1.0/max(eps_R, 1e-300)) ))

        # Build θ, α like your code
        from jax_md.hydro.pse_wave import choose_theta, se_alpha
        M_eff = M  # (cube assumption)
        theta = choose_theta(P, xi, M_eff)
        alpha = se_alpha(xi, theta)

        # Crude cost proxy: wave cost ~ M^3 log(M^3) + N P^3 (FFT + spread)
        # plus real-space cost ~ N^2 (rcut/L)^3 or with a neighbor list ~ N neighbors
        # For simplicity we skip the real-space piece (it's very ξ-dependent).
        cost = (M**3) * _py_np.log(max(1.0, M**3)) + N_particles * (P**3)

        if (best is None) or (cost < best["cost"]):
            best = {
                "xi": xi,
                "M": M,
                "P": P,
                "theta": theta,
                "alpha": alpha,
                "rcut": rcut,
                "cost": cost,
                "eps_est": {"eps_R": eps_R, "eps_W": eps_W, "eps_q": eps_q}
            }

    # Finalize and add notes
    if best is not None:
        best["notes"] = (
            "Targets are independent exponential bounds: "
            "ε_R ≲ exp(-ξ² r_cut²), ε_W ≲ exp(-k_cut²/(4 ξ²)), "
            "ε_q ≲ exp(-π² P² / (2 m²)) + erfc(m/√2) with m ≈ √(π P). "
            "θ, α follow SE choices (m = √(πP) for near-optimal balance). "
            "Guardrail: keep P modest on tiny grids to avoid SE deconvolution blow-up."
        )
    return best
