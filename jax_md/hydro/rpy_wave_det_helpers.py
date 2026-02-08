"""Helper utilities for wave-space Spectral Ewald mobility (deterministic)."""

import jax
import jax.numpy as jnp
from jax import config as jax_config
from functools import partial

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


def positions_to_fractional(positions, A, fractional_coordinates: bool):
    """Convert positions to fractional coordinates when needed."""
    pos = jnp.asarray(positions, dtype=REAL_DTYPE)
    return pos if fractional_coordinates else to_fractional(pos, A)


# ----------------------------
# FFT mode grid and k-vectors
# ----------------------------
def q_grid(Mx: int, My: int, Mz: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate 3D FFT mode indices grid in fractional coordinates."""
    qx = jnp.fft.fftfreq(Mx, d=1.0/Mx)
    qy = jnp.fft.fftfreq(My, d=1.0/My)
    qz = jnp.fft.fftfreq(Mz, d=1.0/Mz)
    QX, QY, QZ = jnp.meshgrid(qx, qy, qz, indexing='ij')
    return QX, QY, QZ


def k_from_q(QX: jnp.ndarray, QY: jnp.ndarray, QZ: jnp.ndarray, B: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Map integer FFT mode indices q -> physical wavevectors k and magnitudes."""
    q = jnp.stack([QX, QY, QZ], axis=-1)            # (...,3)
    k = jnp.einsum('ab,...b->...a', B, q)           # (...,3)
    K2 = jnp.sum(k*k, axis=-1)                      # (...)
    K  = jnp.sqrt(jnp.maximum(K2, 0.0))
    return k, K, K2


# ----------------------------
# SE window (Gaussian) parameters
# ----------------------------
def se_alpha(xi, theta):
    """Gaussian exponent α for SE spreading window h(t) = exp(-α||t||²)."""
    return 8.0 * (jnp.pi**2) * (xi**2) / theta


def choose_theta(P, xi, M_eff, *, m=None):
    """Compute SE window parameter θ for near-optimal quadrature accuracy.

    Parameters
    ----------
    P : int
        Gaussian stencil support size.
    xi : float
        Ewald splitting parameter.
    M_eff : float
        Effective grid size (e.g., mean of Mx, My, Mz).
    m : float, optional
        Gaussian truncation width in standard deviations. If not provided,
        use the spectral Ewald choice m = sqrt(pi * P).
    """
    if m is None:
        m = jnp.sqrt(jnp.pi * P)
    return (2.0 * jnp.pi * P * xi / (M_eff * m))**2


# ----------------------------
# Physics: Hasimoto, projector, sinc
# ----------------------------
@jax.jit
def Hasimoto(K: jnp.ndarray, xi: float) -> jnp.ndarray:
    """Hasimoto screening function H(k, ξ) used for the screened Stokeslet."""
    temp = K / (2.0*xi)
    temp = jnp.pow(temp, 2)
    return (1.0 + temp) * jnp.exp(-temp)

@jax.jit
def sinc(z: jnp.ndarray) -> jnp.ndarray:
    """Sinc function: sin(z) / z with removable singularity handled."""
    safe_z = jnp.where(z == 0.0, 1.0, z)
    return jnp.where(z != 0.0, jnp.sin(z) / safe_z, 1.0)


@jax.jit
def projector_from_k(k: jnp.ndarray, K2: jnp.ndarray) -> jnp.ndarray:
    """Transverse projection tensor P(k) = I - k̂ ⊗ k̂ for each mode."""
    I = jnp.eye(3)
    K = jnp.sqrt(jnp.maximum(K2, 0.0))
    safe_K = jnp.where(K == 0.0, 1.0, K)
    kh = jnp.where((K2 > 0)[..., None], k / safe_K[..., None], 0.0)
    P = I - kh[..., None] * kh[..., None, :]
    return jnp.where((K2 > 0)[..., None, None], P, jnp.zeros_like(P))


# ----------------------------
# Shape operator P
# ----------------------------
@partial(jax.jit, static_argnames=["a"])
def build_P_modes(K: jnp.ndarray, a: float) -> jnp.ndarray:
    """Build the shape operator P(k) = sinc(ka) for monodisperse spheres."""
    ka = K * a
    Pshape = sinc(ka)
    return Pshape


# ----------------------------
# Stencils (indices + separable Gaussian weights)
# ----------------------------
def _stencil_1d(t_scalar, M, P):
    """Build 1D NUFFT stencil indices and centered offsets.

    Parameters
    ----------
    t_scalar : float
        Fractional coordinate along one axis.
    M : int
        Grid size for the axis.
    P : int
        Stencil support size per dimension.

    Returns
    -------
    tuple of ndarray
        Grid indices and centered offsets in fractional units.
    """
    dt = 1.0 / M
    u = (t_scalar / dt) % M
    i0 = jnp.floor(u).astype(int)
    offs = jnp.arange(-(P // 2), -(P // 2) + P)
    idx = (i0 + offs) % M
    du = (u - (i0 + offs)) * dt
    du = (du + 0.5) % 1.0 - 0.5
    return idx, du


def build_stencils_frac(t, Mx, My, Mz, P, alpha):
    """Construct 3D NUFFT stencils from fractional coordinates.

    Parameters
    ----------
    t : array
        Fractional coordinates in [0, 1) with shape (N, 3).
    Mx, My, Mz : int
        Grid sizes along each axis.
    P : int
        Stencil support per dimension.
    alpha : float
        Gaussian window parameter for the SE kernel.

    Returns
    -------
    tuple of ndarray
        (ix, iy, iz, wx, wy, wz) indices and Gaussian weights per axis.
    """
    v = jax.vmap(_stencil_1d, in_axes=(0, None, None))
    ix, dx = v(t[:, 0], Mx, P)
    iy, dy = v(t[:, 1], My, P)
    iz, dz = v(t[:, 2], Mz, P)
    wx = jnp.exp(-alpha * dx * dx)
    wy = jnp.exp(-alpha * dy * dy)
    wz = jnp.exp(-alpha * dz * dz)
    return (ix, iy, iz, wx, wy, wz)


# ----------------------------
# Spread (S) and Gather (S†)
# ----------------------------
def spread(F, st, Mx, My, Mz):
    """Spread particle forces to the FFT grid using SE stencils.

    Parameters
    ----------
    F : array
        Particle forces with shape (N, 3).
    st : tuple
        Stencil tuple returned by ``build_stencils_frac``.
    Mx, My, Mz : int
        Grid dimensions along each axis.

    Returns
    -------
    jnp.ndarray
        Force grid with shape (Mx, My, Mz, 3).
    """
    ix, iy, iz, wx, wy, wz = st
    N, P = ix.shape

    w3 = (wx[:, :, None, None] *
          wy[:, None, :, None] *
          wz[:, None, None, :])

    px, py, pz = jnp.meshgrid(jnp.arange(P), jnp.arange(P), jnp.arange(P), indexing='ij')
    px_flat, py_flat, pz_flat = px.ravel(), py.ravel(), pz.ravel()

    XI = ix[:, px_flat]
    YI = iy[:, py_flat]
    ZI = iz[:, pz_flat]
    w_flat = w3.reshape(N, P**3)

    contrib = w_flat[:, :, None] * F[:, None, :]

    XI_all = XI.ravel()
    YI_all = YI.ravel()
    ZI_all = ZI.ravel()
    contrib_all = contrib.reshape(-1, 3)

    flat_indices = XI_all * My * Mz + YI_all * Mz + ZI_all
    grid = jnp.zeros((Mx * My * Mz, 3), dtype=F.dtype)
    grid = jax.ops.segment_sum(contrib_all, flat_indices, Mx * My * Mz)
    return grid.reshape(Mx, My, Mz, 3)


def gather(u_grid, st, Mx, My, Mz):
    """Interpolate grid values back to particle positions (S†).

    Parameters
    ----------
    u_grid : array
        Grid values with shape (Mx, My, Mz, 3).
    st : tuple
        Stencil tuple returned by ``build_stencils_frac``.
    Mx, My, Mz : int
        Grid dimensions along each axis.

    Returns
    -------
    jnp.ndarray
        Interpolated values at particle locations with shape (N, 3).
    """
    ix, iy, iz, wx, wy, wz = st
    sigma = 1.0  # ifftn already provides the 1/N normalization

    def gather_one(indices, weights):
        """Interpolate grid values for a single particle.

        Parameters
        ----------
        indices : tuple
            One-dimensional stencil indices for x, y, and z.
        weights : tuple
            Corresponding stencil weights along each axis.

        Returns
        -------
        jnp.ndarray
            Interpolated value for the particle (shape (3,)).
        """
        iix, iiy, iiz = indices
        wwx, wwy, wwz = weights

        w3 = (wwx[:, None, None] *
              wwy[None, :, None] *
              wwz[None, None, :])

        XI, YI, ZI = jnp.meshgrid(iix, iiy, iiz, indexing='ij')
        vals = u_grid[XI, YI, ZI, :]
        ui = jnp.sum(w3[..., None] * vals, axis=(0, 1, 2))
        return sigma * ui

    return jax.vmap(gather_one, in_axes=((0, 0, 0), (0, 0, 0)))((ix, iy, iz), (wx, wy, wz))


# ----------------------------
# FFT wrappers
# ----------------------------
def fft_vec(grid):
    """Compute forward FFT of a vector grid.

    Parameters
    ----------
    grid : array
        Real-space grid of shape (Mx, My, Mz, 3).

    Returns
    -------
    jnp.ndarray
        Complex grid in frequency space with matching shape.
    """
    return jnp.fft.fftn(grid, axes=(0, 1, 2))


def ifft_vec(Gq):
    """Inverse FFT with standard JAX normalization (1/N_grid factor).

    Parameters
    ----------
    Gq : array
        Complex grid in frequency space of shape (Mx, My, Mz, 3).

    Returns
    -------
    jnp.ndarray
        Real-valued grid in real space with shape (Mx, My, Mz, 3).
    """
    return jnp.fft.ifftn(Gq, axes=(0, 1, 2)).real


# ----------------------------
# Mode construction (P†BP only)
# ----------------------------
def _build_mode_grid(A, Mx, My, Mz):
    """Precompute reciprocal grid, k-vectors, and magnitudes.

    Parameters
    ----------
    A : array_like
        Periodic cell matrix in real units.
    Mx, My, Mz : int
        Grid dimensions along each axis.

    Returns
    -------
    dict
        Grid metadata including reciprocal lattice vectors and k magnitudes.
    """
    Brecip = make_reciprocal(A)
    V = jnp.linalg.det(A)
    QX, QY, QZ = q_grid(Mx, My, Mz)
    k, K, K2 = k_from_q(QX, QY, QZ, Brecip)
    return dict(A=jnp.asarray(A, dtype=REAL_DTYPE),
                V=V,
                QX=QX, QY=QY, QZ=QZ,
                k=k, K=K, K2=K2,
                Brecip=Brecip)


def _build_se_deconv_window(Mx, My, Mz, P, alpha):
    """Analytic |h_hat(q)|^{-2} for the SE Gaussian window.

    Following Lindbo & Tornberg (2010), the Gaussian window used for the
    NUFFT gridding has Fourier transform

      ĥ(q) = (π/α)^{3/2} exp(-π^2 |q|^2 / α)

    in fractional coordinates. The deconvolution factor is therefore

      |ĥ(q)|^{-2} = (α/π)^3 exp(2 π^2 |q|^2 / α),

    which corrects the double application of the Gaussian (spread + gather).

    Parameters
    ----------
    Mx, My, Mz : int
        Grid dimensions along each axis.
    P : int
        Stencil support size per dimension (included for signature parity).
    alpha : float
        Gaussian window parameter for the SE kernel.

    Returns
    -------
    jnp.ndarray
        Deconvolution factors on the FFT grid.
    """
    del P  # unused (alpha already encodes the SE window width)
    QX, QY, QZ = q_grid(Mx, My, Mz)
    Q2 = QX * QX + QY * QY + QZ * QZ
    alpha = jnp.asarray(alpha, dtype=REAL_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    # Discrete FFT uses unnormalized sums, so include the grid-spacing factor
    # (1/Ngrid)^2 in addition to the continuous Fourier prefactor.
    prefactor = (alpha / jnp.pi) ** 3 / (Ngrid ** 2)
    return prefactor * jnp.exp(2.0 * (jnp.pi ** 2) * Q2 / alpha)


@jax.jit
def build_B_modes(k, K, K2, xi, eta, V, deconv):
    """Construct the fluid kernel B(k) (per-mode 3x3 tensor, no shape factors).

    Parameters
    ----------
    k : array
        Wavevectors in reciprocal space.
    K : array
        Magnitudes of the wavevectors.
    K2 : array
        Squared magnitudes of the wavevectors.
    xi : float
        Ewald splitting parameter.
    eta : float
        Fluid viscosity.
    V : float
        Box volume.
    deconv : array
        Deconvolution factors for the SE window.

    Returns
    -------
    tuple of jnp.ndarray
        Bfluid and Bhalf tensors defined on the grid.
    """
    Pkk = projector_from_k(k, K2)
    H = Hasimoto(K, xi)
    scal = jnp.where(K2 > 0.0, (H * deconv.real) / (eta * V * K2), 0.0)
    sqrt_scal = jnp.sqrt(jnp.clip(scal, a_min=0.0))

    Bfluid = scal[..., None, None] * Pkk
    Bfluid = Bfluid.at[0, 0, 0].set(jnp.zeros((3, 3)))

    Bhalf = sqrt_scal[..., None, None] * Pkk
    Bhalf = Bhalf.at[0, 0, 0].set(jnp.zeros((3, 3)))
    return Bfluid, Bhalf
