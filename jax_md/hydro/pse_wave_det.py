# JAX implementation of the wave-space Spectral Pse mobility
# Wang Eq. (39) -> Fiore operator chain: M^w = S^† F^{-1} B F S
# Fractional coords throughout. Monodisperse spheres (radius a).
#
# THESIS NOTATION (Ch. 3):
# ========================
# The mobility operator is factorized as:
#   M^(w) = D† P† B P D
#
# where (with NUFFT D = FQ, where F=FFT, Q=stencils):
#   - P: shape operator (sinc factor for monodisperse spheres)
#   - B: fluid kernel (Hasimoto, projector, 1/k², viscosity, SE deconv)
#   - D: NUFFT operator (spreading S and gathering S†)
#
# This refactored implementation provides:
#   1. build_P_modes(K, a, include_faxen): builds shape operator P = sinc(ka)
#   2. build_B_modes(...): builds Bfluid (fluid kernel) and Pshape separately
#   3. build_Mw_apply(cfg): applies sandwich P† B P in k-space
#
# NAMING CONVENTIONS:
#   - Pshape: thesis P (shape operator, scalar per mode)
#   - Bfluid: thesis B (fluid kernel, 3×3 tensor per mode)
#   - projector_from_k: transverse projector (I - k̂k̂) inside B, NOT thesis P
#   - Brecip: reciprocal lattice matrix (2π A^{-T}), NOT related to B kernel
#
# SE deconvolution lives with B (grid implementation detail), keeping P physical-only.
# This separation enables future polydispersity (P becomes per-particle, not scalar).

import jax
import jax.numpy as jnp
from jax import config as jax_config

REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32
COMPLEX_DTYPE = jnp.complex128 if jax_config.jax_enable_x64 else jnp.complex64


# ----------------------------
# Lattice helpers
# ----------------------------
def make_reciprocal(A):
    """B = 2π A^{-T} so that A^T B = 2π I."""
    return 2.0 * jnp.pi * jnp.linalg.inv(A).T


def to_fractional(x, A):
    """Real coords x -> fractional coords t in [0,1)^3."""
    Ainv = jnp.linalg.inv(A)
    return (x @ Ainv.T) % 1.0


# ----------------------------
# FFT mode grid and k-vectors
# ----------------------------
def q_grid(Mx, My, Mz):
    """Generate FFT mode grid in fractional coordinates."""
    qx = jnp.fft.fftfreq(Mx, d=1.0/Mx)
    qy = jnp.fft.fftfreq(My, d=1.0/My)
    qz = jnp.fft.fftfreq(Mz, d=1.0/Mz)
    QX, QY, QZ = jnp.meshgrid(qx, qy, qz, indexing='ij')
    return QX, QY, QZ


def k_from_q(QX, QY, QZ, B):
    """Physical k = B q; return k(vec), |k|, and |k|^2."""
    q = jnp.stack([QX, QY, QZ], axis=-1)            # (...,3)
    k = jnp.einsum('ab,...b->...a', B, q)           # (...,3)
    K2 = jnp.sum(k*k, axis=-1)                      # (...)
    K  = jnp.sqrt(jnp.maximum(K2, 0.0))
    return k, K, K2


# ----------------------------
# SE window (Gaussian) parameters
# ----------------------------
# RELATIONSHIP BETWEEN SE PARAMETERS:
# ==================================
# The Spectral Pse method uses Gaussian spreading/gathering with window h(t).
# In fractional coordinates, h(t) = exp(-α ||t||²) where:
#   α = 8π² ξ² / θ
#
# The Fourier transform satisfies |ĥ(q)|² = exp(-θ |q|²/(4ξ²)).
# The quadrature error bound involves the Gaussian width m:
#   ε_q ≲ exp(-π² P² / (2 m²)) + erfc(m/√2)
#
# Standard choice: m = √(π P), which relates to θ via choose_theta below.
# This gives θ = (2π P ξ / (M_eff · m))² ≈ (2π P ξ / (M_eff √(πP)))².

def se_alpha(xi, theta):
    """
    Real-space Gaussian exponent α for SE spreading window h(t) = exp(-α||t||²).
    
    Parameters
    ----------
    xi : float
        Pse splitting parameter
    theta : float
        SE window parameter (from choose_theta)
        
    Returns
    -------
    float
        Gaussian exponent α = 8π² ξ² / θ
    """
    return 8.0 * (jnp.pi**2) * (xi**2) / theta


def hhat_sq_inv(QX, QY, QZ, xi, theta):
    """
    1/|ĥ(q)|^2 for SE deconvolution; |ĥ|^2 = exp(-(θ/4)|q|^2/ξ^2).
    NOTE: q = integer mode indices (from fftfreq), not physical k-vectors.
    This matches SE literature conventions in integer mode space.
    """
    Q2 = QX*QX + QY*QY + QZ*QZ
    return jnp.exp(0.25 * theta * Q2 / (xi*xi))


def choose_theta(P, xi, M_eff):
    """
    Compute SE window parameter θ for near-optimal quadrature accuracy.
    
    Uses the standard recipe m = √(π P) where m is the Gaussian width
    and P is the stencil support. This choice balances the two terms in
    the SE quadrature bound:
      ε_q ≲ exp(-π² P² / (2 m²)) + erfc(m/√2)
    
    Parameters
    ----------
    P : int
        Stencil support (number of grid points per dimension)
    xi : float
        Pse splitting parameter
    M_eff : float
        Effective grid size (typically average of Mx, My, Mz)
        
    Returns
    -------
    float
        Window parameter θ for SE Gaussian spreading
    """
    m = jnp.sqrt(jnp.pi * P)
    return (2.0 * jnp.pi * P * xi / (M_eff * m))**2


# ----------------------------
# Physics: Hasimoto, projector, sinc
# ----------------------------
def Hasimoto(K, xi):
    """Hasimoto screening function."""
    t = K / (2.0*xi)
    return (1.0 + t*t) * jnp.exp(-t*t)


def sinc(z):
    """Sinc function: sin(z)/z, safely returning 1 when z == 0."""
    safe_z = jnp.where(z == 0.0, 1.0, z)
    return jnp.where(z != 0.0, jnp.sin(z) / safe_z, 1.0)


def projector_from_k(k, K2):
    """
    Transverse projector P = I - k̂⊗k̂; returns zeros for k=0.
    
    Note: We build k̂ only where K2 > 0 to avoid unnecessary division.
    This is already efficient; if profiling shows hotspots, one could skip
    forming kh entirely on zero modes (micro-optimization).
    """
    I = jnp.eye(3)
    # Build k̂ only where K2 > 0 to avoid unnecessary division
    K = jnp.sqrt(jnp.maximum(K2, 0.0))
    safe_K = jnp.where(K == 0.0, 1.0, K)
    kh = jnp.where((K2 > 0)[..., None], k / safe_K[..., None], 0.0)  # (...,3)
    P = I - kh[..., None] * kh[..., None, :]                    # (...,3,3)
    return jnp.where((K2 > 0)[..., None, None], P, jnp.zeros_like(P))


# ----------------------------
# NEW: Shape operator P (thesis notation)
# ----------------------------
def build_P_modes(K, a):
    """
    Thesis P: wave-space shape operator for monodisperse spheres.
    For spheres, P(k) is a scalar (applied componentwise to vectors).
    
    P = sinc(ka)  (so P† B P gives sinc²(ka) inside the sandwich)
    
    Rationale: In the thesis sandwich P† B P, using P=sinc(ka) produces sinc²(ka)
    overall, matching the paper's Eq. (9). Keeping P as the square root preserves
    the clean split and helps later if you add polydispersity (then P is no longer
    a scalar per mode).
    """
    ka = K * a
    Pshape = sinc(ka)  # scalar per-mode (...,)
    return Pshape  # (...,)


# ----------------------------
# Build B(q): per-mode 3x3 block (Wang Eq. 39 + Fiore form)
# ----------------------------
# ----------------------------
# Thesis-style factorization: M^(w) = S† F^{-1} P† B P F S
# ----------------------------
def build_B_modes(A, a, xi, eta, Mx, My, Mz, P_support, theta=None):
    """
    Thesis-style factorization:
      - Pshape: sinc(ka)
      - Bfluid: (1/ηV) * H(k,ξ) * (I - k̂k̂) / k² * (SE deconvolution)
      - M^w = S† F^{-1} [ P† ⋅ Bfluid ⋅ P ] F S
    
    This separates the shape operator P from the fluid kernel B, matching the
    thesis notation in Ch. 3. The sinc² factor is now split: P=sinc(ka) appears
    on both sides of the sandwich P† B P.
    
    SE deconvolution lives with B (grid implementation detail), keeping P purely physical.
    
    Returns dict with:
      - Bfluid: per-mode fluid kernel (Mx,My,Mz,3,3)
      - Pshape: per-mode shape factor (...,) scalar
      - other cached arrays
    """
    Brecip = make_reciprocal(A)
    V = jnp.linalg.det(A)

    if theta is None:
        M_eff = (Mx + My + Mz) / 3.0
        theta = choose_theta(P_support, xi, M_eff)

    QX, QY, QZ = q_grid(Mx, My, Mz)
    k, K, K2 = k_from_q(QX, QY, QZ, Brecip)

    # Thesis P (scalar per mode)
    Pshape = build_P_modes(K, a)  # (...,)

    # Fluid kernel B (matrix per mode)
    Pkk = projector_from_k(k, K2)                   # (..,3,3), I - k̂k̂
    H    = Hasimoto(K, xi)                          # Hasimoto split
    # Empirical NUFFT-window deconvolution: build the actual grid window from the same stencils
    alpha = se_alpha(xi, theta)
    t0 = jnp.zeros((1,3), dtype=REAL_DTYPE)
    st0 = build_stencils_frac(t0, Mx, My, Mz, P_support, alpha)
    F0 = jnp.array([[1.0, 0.0, 0.0]], dtype=REAL_DTYPE)
    wgrid = spread(F0, st0, Mx, My, Mz)  # (Mx,My,Mz,3)
    Hq = jnp.fft.fftn(wgrid[...,0], axes=(0,1,2))
    deconv = 1.0 / jnp.maximum(jnp.abs(Hq)**2, jnp.finfo(REAL_DTYPE).tiny)

    # 1/(ηV) * H * (1/k²) * deconv, zero k=0
    scal = jnp.where(K2 > 0.0, (H * deconv.real) / (eta * V * K2), 0.0)   # (...,)
    sqrt_scal = jnp.sqrt(jnp.clip(scal, a_min=0.0))

    Bfluid = scal[..., None, None] * Pkk                              # (...,3,3)
    Bfluid = Bfluid.at[0, 0, 0].set(jnp.zeros((3, 3)))                # explicit zero mode

    Bhalf = sqrt_scal[..., None, None] * Pkk
    Bhalf = Bhalf.at[0, 0, 0].set(jnp.zeros((3, 3)))

    return dict(Brecip=Brecip, V=V, Bfluid=Bfluid, Bhalf=Bhalf, Pshape=Pshape,
                QX=QX, QY=QY, QZ=QZ, k=k, K=K, K2=K2,
                xi=xi, eta=eta, a=a, theta=theta, alpha=alpha,
                A=A, Mx=Mx, My=My, Mz=Mz, P=P_support)


# ----------------------------
# Stencils (indices + separable Gaussian weights)
# ----------------------------
def _stencil_1d(t_scalar, M, P):
    """Build 1D stencil indices and weights for NUFFT."""
    dt = 1.0 / M
    u  = (t_scalar / dt) % M
    i0 = jnp.floor(u).astype(int)
    offs = jnp.arange(-(P//2), -(P//2) + P)
    idx  = (i0 + offs) % M
    du   = (u - (i0 + offs)) * dt
    du   = (du + 0.5) % 1.0 - 0.5
    return idx, du


def build_stencils_frac(t, Mx, My, Mz, P, alpha):
    """Build 3D stencils for all particles using separable Gaussian weights."""
    v = jax.vmap(_stencil_1d, in_axes=(0, None, None))
    ix, dx = v(t[:,0], Mx, P)
    iy, dy = v(t[:,1], My, P)
    iz, dz = v(t[:,2], Mz, P)
    # Gaussian 1D weights per axis (no normalization; empirical deconvolution will remove their spectral footprint)
    wx = jnp.exp(-alpha * dx*dx)
    wy = jnp.exp(-alpha * dy*dy)
    wz = jnp.exp(-alpha * dz*dz)
    return (ix, iy, iz, wx, wy, wz)


# ----------------------------
# Spread (S) and Gather (S^†)
# ----------------------------
def spread(F, st, Mx, My, Mz):
    """
    Spread particle forces F to grid using stencils st (fully vectorized).
    
    Parameters
    ----------
    F : (N,3) array
        Force vectors at particle positions
    st : tuple
        Stencils from build_stencils_frac
    Mx, My, Mz : int
        Grid dimensions
        
    Returns
    -------
    grid : (Mx,My,Mz,3) array
        Force grid
    """
    ix, iy, iz, wx, wy, wz = st
    N, P = ix.shape
    
    # Compute all 3D weights: (N, P, P, P)
    w3 = (wx[:, :, None, None] * 
          wy[:, None, :, None] * 
          wz[:, None, None, :])
    
    # Create stencil coordinate grids (P, P, P)
    px, py, pz = jnp.meshgrid(jnp.arange(P), jnp.arange(P), jnp.arange(P), indexing='ij')
    px_flat, py_flat, pz_flat = px.ravel(), py.ravel(), pz.ravel()
    
    # Get all grid indices for all particles: (N, P^3)
    XI = ix[:, px_flat]  # (N, P^3)
    YI = iy[:, py_flat]
    ZI = iz[:, pz_flat]
    w_flat = w3.reshape(N, P**3)  # (N, P^3)
    
    # Compute weighted contributions: (N, P^3, 3)
    contrib = w_flat[:, :, None] * F[:, None, :]
    
    # Flatten everything for scatter
    XI_all = XI.ravel()  # (N*P^3,)
    YI_all = YI.ravel()
    ZI_all = ZI.ravel()
    contrib_all = contrib.reshape(-1, 3)  # (N*P^3, 3)
    
    # Convert 3D indices to flat indices
    flat_indices = XI_all * My * Mz + YI_all * Mz + ZI_all
    
    # Use segment_sum for efficient scatter-add (handles each component)
    grid = jnp.zeros((Mx * My * Mz, 3), dtype=F.dtype)
    
    # Process all 3 components at once using segment_sum
    grid = jax.ops.segment_sum(contrib_all, flat_indices, Mx * My * Mz)
    
    return grid.reshape(Mx, My, Mz, 3)


def gather(u_grid, st, Mx, My, Mz):
    """
    Gather (S^†): interpolate grid values to particle positions (vectorized).
    
    Uses vmap instead of fori_loop for performance.
    
    Normalization: this routine returns the unscaled interpolation result.
    Callers apply the grid-cell volume σ = V / (Mx*My*Mz) as needed.
    The inverse FFT (`ifftn`) already includes the 1/N_grid factor.
    """
    ix, iy, iz, wx, wy, wz = st
    # Use unit scaling; ifftn provides the only 1/N normalization
    sigma = 1.0

    def gather_one(indices, weights):
        """Gather velocity for a single particle (no stencil normalization; ifftn provides 1/N)."""
        iix, iiy, iiz = indices
        wwx, wwy, wwz = weights

        # Compute 3D weight tensor
        w3 = (wwx[:, None, None] *
              wwy[None, :, None] *
              wwz[None, None, :])

        # Create index grids and gather values
        XI, YI, ZI = jnp.meshgrid(iix, iiy, iiz, indexing='ij')
        vals = u_grid[XI, YI, ZI, :]  # (P, P, P, 3)

        # Contract: sum over P^3 stencil points
        ui = jnp.sum(w3[..., None] * vals, axis=(0, 1, 2))
        return sigma * ui

    # Vectorize over all particles
    U = jax.vmap(gather_one, in_axes=(
        (0, 0, 0),  # indices
        (0, 0, 0)   # weights
    ))((ix, iy, iz), (wx, wy, wz))

    return U


# ----------------------------
# FFT wrappers
# ----------------------------
def fft_vec(grid):
    """Forward FFT of vector grid."""
    return jnp.fft.fftn(grid, axes=(0,1,2))


def ifft_vec(Gq):
    """Inverse FFT with standard JAX normalization (1/N_grid factor)."""
    return jnp.fft.ifftn(Gq, axes=(0,1,2)).real


# ----------------------------
# Wave operator apply:  u = S^† F^{-1} B F S F
# ----------------------------
def build_Mw_apply(cfg):
    """
    Return a function Mw_apply(t, F) for the wave-space mobility.
    

    Supports thesis mode only:
    u = S† F^{-1} [ P† ⋅ B ⋅ P ] F S

    The formulation matches Ch. 3 notation: M^(w) = D† P† B P D where
    P is the shape operator (sinc factor) and B is the fluid kernel.
    
    Parameters
    ----------
    cfg : dict
        Configuration from build_B_modes
        
    Returns
    -------
    Mw_apply : function
        Callable Mw_apply(t_frac, F) -> U, where:
        - t_frac: (N,3) fractional positions in [0,1)
        - F: (N,3) forces
        - U: (N,3) velocities
    """
    Mx, My, Mz = cfg["Mx"], cfg["My"], cfg["Mz"]
    P, alpha = cfg["P"], cfg["alpha"]

    Bfluid = cfg["Bfluid"]
    Pshape = cfg["Pshape"]
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(cfg["V"], dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box

    def Mw_apply(t_frac, F_part):
        st = build_stencils_frac(t_frac, Mx, My, Mz, P, alpha)
        g_grid = spread(F_part, st, Mx, My, Mz)        # S
        g_grid = sigma_inv * g_grid                   # convert to physical density
        Gq     = fft_vec(g_grid)                       # F
        # Thesis sandwich: Uq = P† ⋅ (B ⋅ (P ⋅ Gq))
        PGq    = Pshape[..., None] * Gq                # P (right)
        BPGq   = jnp.einsum('...ij,...j->...i', Bfluid, PGq)
        Uq     = Pshape[..., None] * BPGq              # P† (left); same scalar here
        u_grid = ifft_vec(Uq)                          # F^{-1} (includes 1/N_grid)
        U_part = gather(u_grid, st, Mx, My, Mz)        # S^† (already in physical units)
        return V_box * U_part
    return jax.jit(Mw_apply)

# ----------------------------
# Brute-force reference implementation for testing
# ----------------------------
def Mw_bruteforce(t, F, A, a, xi, eta, Mx, My, Mz, P, theta=None):
    """
    Direct k-space sum using the same mode set as the FFT (minus q=0).
    This is slow: use tiny N and M.
    NO SE deconvolution here - this is the true k-space kernel.
    
    Use this for testing and validation only, not for production.
    
    Parameters
    ----------
    t : (N,3) array
        Fractional positions in [0,1)
    F : (N,3) array
        Forces
    A : (3,3) array
        Periodic cell matrix
    a : float
        Sphere radius
    xi : float
        Pse splitting parameter
    eta : float
        Fluid viscosity
    Mx, My, Mz : int
        Grid dimensions (defines mode set)
    P : int
        Stencil support (unused here, for signature compatibility)
    theta : float, optional
        SE parameter (unused here, for signature compatibility)
        
    Returns
    -------
    U : (N,3) array
        Velocities from wave-space operator
    """
    Brecip = make_reciprocal(A)
    V = jnp.linalg.det(A)
    QX, QY, QZ = q_grid(Mx, My, Mz)
    q = jnp.stack([QX, QY, QZ], axis=-1).reshape(-1, 3)  # (Q,3)
    k = jnp.einsum('ab,qb->qa', Brecip, q)               # (Q,3)
    K2 = jnp.sum(k*k, axis=1)
    K = jnp.sqrt(jnp.maximum(K2, 0.0))

    # Hasimoto + finite-size (NO deconvolution here)
    tH = (K/(2.0*xi))**2
    H = (1.0 + tH) * jnp.exp(-tH)
    ka = K * a
    
    # sinc squared for finite-size sphere
    rpy = jnp.where(K2 > 0.0, sinc(ka)**2 / K2, 0.0)
    scal = jnp.where(K2>0.0, (H * rpy) / (eta * V), 0.0)                 # (Q,)

    # projector
    I = jnp.eye(3)
    denom = jnp.where(K2>0.0, jnp.sqrt(K2), 1.0)
    kh = k / denom[:,None]
    Pkk = I - kh[...,None] * kh[:,None,:]                                # (Q,3,3)
    Gk = scal[:,None,None] * Pkk                                          # (Q,3,3)

    # phases and contraction
    phase  = jnp.exp(2j * jnp.pi * (t @ q.T))                             # (N,Q)
    phi_ij = phase[:,None,:] * jnp.conj(phase)[None,:,:]                  # (N,N,Q)
    W_qj   = jnp.einsum('qij,nj->qni', Gk, F)                             # (Q,N,3)
    U      = jnp.einsum('ijq,qjv->iv', phi_ij, W_qj).real                 # (N,3)
    return U
