"""Stochastic sampling utilities for the wave-space Spectral Pse mobility."""

import jax
import jax.numpy as jnp

from jax_md.hydro.pse_wave_det import (
    REAL_DTYPE,
    COMPLEX_DTYPE,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
)

def _hermitian_gaussian_modes(key, shape):
    """Draw complex Gaussian FFT modes with Hermitian symmetry.

    The construction follows the standard half-spectrum recipe: independent
    complex standard normals are drawn on a canonical half of reciprocal space,
    self-conjugate modes are purely real, and the remaining half is populated by
    complex conjugation. This yields unit variance per independent mode and
    ensures the inverse FFT produces a real-valued field.
    """

    Mx, My, Mz, vec_dim = shape
    key_r, key_i, key_self = jax.random.split(key, 3)

    ix = jnp.arange(Mx, dtype=jnp.int32)
    iy = jnp.arange(My, dtype=jnp.int32)
    iz = jnp.arange(Mz, dtype=jnp.int32)
    IX, IY, IZ = jnp.meshgrid(ix, iy, iz, indexing='ij')

    neg_ix = (-IX) % Mx
    neg_iy = (-IY) % My
    neg_iz = (-IZ) % Mz

    self_mask = (IX == neg_ix) & (IY == neg_iy) & (IZ == neg_iz)

    # Canonical half-spectrum: lexicographic order (k >= -k).
    canonical = (
        (IX > neg_ix)
        | ((IX == neg_ix) & (IY > neg_iy))
        | ((IX == neg_ix) & (IY == neg_iy) & (IZ >= neg_iz))
    )

    real = jax.random.normal(key_r, shape, dtype=REAL_DTYPE)
    imag = jax.random.normal(key_i, shape, dtype=REAL_DTYPE)
    canonical_modes = (real + 1j * imag) / jnp.sqrt(2.0)

    # Self-conjugate entries must be purely real.
    real_self = jax.random.normal(key_self, (Mx, My, Mz, vec_dim), dtype=REAL_DTYPE)
    canonical_modes = jnp.where(
        self_mask[..., None],
        jnp.asarray(real_self, dtype=COMPLEX_DTYPE),
        canonical_modes,
    )

    # Build full spectrum by mirroring the canonical half.
    mirrored = canonical_modes[neg_ix, neg_iy, neg_iz].conj()
    G = jnp.where(canonical[..., None], canonical_modes, mirrored)

    # Remove the zero mode.
    zero_vec = jnp.zeros((vec_dim,), dtype=COMPLEX_DTYPE)
    G = G.at[0, 0, 0].set(zero_vec)
    return G


def build_Mw_sqrt_sampler(cfg):
    """
    Build the stochastic square-root sampler for the wave-space mobility.

    The sampler is calibrated to satisfy fluctuation-dissipation by matching
    the covariance of stochastic increments to the deterministic mobility.
    """

    Mx, My, Mz = cfg["Mx"], cfg["My"], cfg["Mz"]
    P, alpha = cfg["P"], cfg["alpha"]
    Pshape = cfg["Pshape"]
    Bhalf = cfg["Bhalf"]

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(cfg["V"], dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    V_sqrt = jnp.sqrt(V_box)

    def Mw_sqrt(key, t_frac):
        st = build_stencils_frac(t_frac, Mx, My, Mz, P, alpha)
        modes = _hermitian_gaussian_modes(key, (Mx, My, Mz, 3))
        modes = jnp.einsum('...ij,...j->...i', Bhalf_complex, modes)
        modes = Pshape[..., None] * modes
        u_grid = ifft_vec(modes)
        Ub = gather(u_grid, st, Mx, My, Mz)  # S^T (unscaled)
        return noise_scale * (V_sqrt * Ub)

    return jax.jit(Mw_sqrt)


def build_Mw_apply_and_sample(cfg):
    """
    Build a fused wave-space operator that computes both deterministic and
    stochastic contributions with consistent FFT/S† normalization.

    Deterministic:  U_det = S^T F^{-1} [ P ⋅ B ⋅ P ] F (σ^{-1} S) ⋅ F_part
    Stochastic:     U_sto = √(σ^{-1} Ngrid) · S^T F^{-1} [ P ⋅ B^{1/2} ] · W
    where σ^{-1} = Ngrid / V converts particle forces to grid densities.
    """
    import jax
    import jax.numpy as jnp

    Mx, My, Mz = cfg["Mx"], cfg["My"], cfg["Mz"]
    P, alpha = cfg["P"], cfg["alpha"]

    Pshape = cfg["Pshape"]                 # (...,)
    Bfluid = cfg["Bfluid"]                 # (...,3,3)
    Bhalf  = cfg["Bhalf"]                  # (...,3,3)

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(cfg["V"], dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    V_sqrt = jnp.sqrt(V_box)

    @jax.jit
    def fused(key, t_frac, F_part):
        # Build stencils at current fractional positions
        st = build_stencils_frac(t_frac, Mx, My, Mz, P, alpha)

        # ---------------- Deterministic branch ----------------
        g_grid = spread(F_part, st, Mx, My, Mz)                  # S
        g_grid = sigma_inv * g_grid                              # convert to physical density
        Gq = fft_vec(g_grid)                                     # F (unnormalized)
        PGq = Pshape[..., None] * Gq                             # P (right)
        BPGq = jnp.einsum('...ij,...j->...i', Bfluid, PGq)       # B
        Uq_det = Pshape[..., None] * BPGq                        # P (left)
        u_grid_det = ifft_vec(Uq_det)                            # F^{-1} (1/Ngrid)
        U_det = gather(u_grid_det, st, Mx, My, Mz)               # S^T
        U_det = V_box * U_det

        # ---------------- Stochastic branch ----------------
        modes = _hermitian_gaussian_modes(key, (Mx, My, Mz, 3))  # unit i.i.d. modes
        modes = jnp.einsum('...ij,...j->...i', Bhalf_complex, modes)  # B^{1/2}
        modes = Pshape[..., None] * modes                        # P (right)
        u_grid_sto = ifft_vec(modes)                             # F^{-1}
        U_sto = gather(u_grid_sto, st, Mx, My, Mz)               # S^T
        U_sto = noise_scale * (V_sqrt * U_sto)

        return U_det, U_sto

    return fused
