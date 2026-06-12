"""Stochastic sampling utilities for the wave-space Spectral Ewald mobility."""

import jax
import jax.numpy as jnp

from jax_md.hydro.rpy_wave_det_helpers import (
    REAL_DTYPE,
    COMPLEX_DTYPE,
    positions_to_fractional,
)
from jax_md.hydro.rpy_wave_det import (
    WaveSpaceState,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
)


def _hermitian_gaussian_modes(key, shape):
    """Draw complex Gaussian FFT modes with Hermitian symmetry.
    
    Generates random complex-valued modes on a 3D FFT grid that satisfy
    the Hermitian symmetry property: G(-k) = conj(G(k)). This ensures
    that the inverse FFT yields real-valued fields. Modes with k = -k
    (self-conjugate) are drawn as purely real.
    
    Parameters
    ----------
    key : PRNGKey
        JAX random key for sampling.
    shape : tuple of int
        (Mx, My, Mz, vec_dim) where Mx, My, Mz are grid dimensions
        and vec_dim is the vector dimension (typically 3).
    
    Returns
    -------
    array
        Complex array of shape (Mx, My, Mz, vec_dim) with Hermitian symmetry.
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
    canonical = (
        (IX > neg_ix)
        | ((IX == neg_ix) & (IY > neg_iy))
        | ((IX == neg_ix) & (IY == neg_iy) & (IZ >= neg_iz))
    )

    real = jax.random.normal(key_r, shape, dtype=REAL_DTYPE)
    imag = jax.random.normal(key_i, shape, dtype=REAL_DTYPE)
    canonical_modes = (real + 1j * imag) / jnp.sqrt(2.0)

    real_self = jax.random.normal(key_self, (Mx, My, Mz, vec_dim), dtype=REAL_DTYPE)
    canonical_modes = jnp.where(
        self_mask[..., None],
        jnp.asarray(real_self, dtype=COMPLEX_DTYPE),
        canonical_modes,
    )

    mirrored = canonical_modes[neg_ix, neg_iy, neg_iz].conj()
    G = jnp.where(canonical[..., None], canonical_modes, mirrored)
    G = G.at[0, 0, 0].set(jnp.zeros((vec_dim,), dtype=COMPLEX_DTYPE))
    return G


def grand_readout_noise_modes(key, *, Pshape, Pdip, k, Bhalf_complex,
                              Mx, My, Mz):
    """Hermitian 11-channel noise grid with covariance matching M^(w)_grand.

    Let ``W(q) = [Pshape B^{1/2} ; +i Pdip (B^{1/2} .) (x) k]`` be the grand
    readout applied to the per-mode square root, so the deterministic
    operator at mode q is ``A(q) = W(q) W(q)^H``.  Because ``ifft_vec``
    keeps only the real part, the *effective* deterministic kernel on a
    conjugate pair ``(q, q' = -q mod M)`` is the Hermitian PSD matrix

        T(q) = (A(q) + conj(A(q'))) / 2 = X X^H,
        X = [W(q), conj(W(q'))] / sqrt(2).

    For ordinary pairs ``k(q') = -k(q)`` gives ``W(q') = conj(W(q))`` and
    T = A, recovering the textbook sampler.  On Nyquist planes (even grid
    dims) fftfreq assigns the *same* sign to the Nyquist component of k at
    q and q', so ``W(q') != conj(W(q))`` and a single-draw sampler is
    non-Hermitian there (its imaginary part is silently dropped by the
    .real, under-counting the covariance).  The factorization above fixes
    this exactly and cheaply: draw two independent complex Gaussian fields
    z1, z3 with E[z z^H] = I, push both through W, and set

        n(q) = (W(q) z1(q) + conj(W(q') z3(q'))) / sqrt(2)   (canonical q)
        n(q') = conj(n(q));   n(q) = Re(W z1 + W z3)(q)      (q = q')

    which has per-mode covariance exactly T(q), zero pseudo-covariance, and
    exact Hermitian symmetry (real IFFT to machine precision).  The k = 0
    mode is zeroed (no net drift).

    Returns the complex (Mx, My, Mz, 11) grid (channels: 3 velocity + 8
    drop-zz gradient components); covariance normalization (sigma, V_box)
    is applied by the caller.
    """
    from jax_md.hydro.rpy_moments import couplet_to_components

    def apply_W(z):
        uq = jnp.einsum('...ij,...j->...i', Bhalf_complex, z)
        Uq = Pshape[..., None] * uq
        Dq = 1j * Pdip[..., None, None] * jnp.einsum(
            '...i,...j->...ij', uq, k)
        return jnp.concatenate([Uq, couplet_to_components(Dq)], axis=-1)

    def complex_normal(subkey, shape):
        kr, ki = jax.random.split(subkey)
        re = jax.random.normal(kr, shape, dtype=REAL_DTYPE)
        im = jax.random.normal(ki, shape, dtype=REAL_DTYPE)
        return ((re + 1j * im) / jnp.sqrt(2.0)).astype(COMPLEX_DTYPE)

    key1, key3 = jax.random.split(key)
    z1 = complex_normal(key1, (Mx, My, Mz, 3))
    z3 = complex_normal(key3, (Mx, My, Mz, 3))

    n1 = apply_W(z1)
    v = apply_W(z3)

    ix = jnp.arange(Mx, dtype=jnp.int32)
    iy = jnp.arange(My, dtype=jnp.int32)
    iz = jnp.arange(Mz, dtype=jnp.int32)
    IX, IY, IZ = jnp.meshgrid(ix, iy, iz, indexing='ij')
    neg_ix, neg_iy, neg_iz = (-IX) % Mx, (-IY) % My, (-IZ) % Mz
    self_mask = (IX == neg_ix) & (IY == neg_iy) & (IZ == neg_iz)
    canonical = (
        (IX > neg_ix)
        | ((IX == neg_ix) & (IY > neg_iy))
        | ((IX == neg_ix) & (IY == neg_iy) & (IZ >= neg_iz))
    )

    # conj(W(q') z3(q')) evaluated at q (q' = -q mod M).
    v_mirror = v[neg_ix, neg_iy, neg_iz].conj()
    n = (n1 + v_mirror) / jnp.sqrt(2.0)
    # Self-conjugate modes must be real: sqrt(2) Re(W zeta),
    # zeta = (z1 + z3)/sqrt(2), i.e. Re(n1 + v).
    n = jnp.where(self_mask[..., None],
                  jnp.real(n1 + v).astype(COMPLEX_DTYPE), n)
    # Enforce Hermitian symmetry from the canonical half.
    n = jnp.where(canonical[..., None], n, n[neg_ix, neg_iy, neg_iz].conj())
    n = n.at[0, 0, 0].set(jnp.zeros((n.shape[-1],), dtype=COMPLEX_DTYPE))
    return n


def _modes_from_state(state: WaveSpaceState):
    """Extract modes dict and params from a WaveSpaceState."""
    if not isinstance(state, WaveSpaceState):
        raise ValueError("Expected a WaveSpaceState instance.")
    return state.modes, state.params


def build_Mw_sqrt_sampler(state: WaveSpaceState):
    """Build stochastic sampler matching the wave-space covariance.
    
    Constructs a sampler that generates Brownian velocities with covariance
    M^(w) by applying the operator sqrt(M^(w)) to Gaussian white noise.
    The sampler uses the precomputed Bhalf tensor (square root of the fluid
    kernel) and respects the fluctuation-dissipation theorem.
    
    The stochastic contribution is computed as:
      U_stochastic = sqrt(2 kB T) * D† · P† · sqrt(B) · ξ
    where ξ are complex Hermitian Gaussian modes on the FFT grid.
    
    Parameters
    ----------
    state : WaveSpaceState
        WaveSpaceState returned by build_wave_modes containing Pshape, Bhalf,
        grid metadata, and box properties.
    
    Returns
    -------
    Callable
        JIT-compiled function Mw_sqrt(key, positions, current_box=None, transform=None)
        that returns stochastic velocities with shape (N, 3).
        
        - key : PRNGKey for random sampling
        - positions : array of particle positions (N, 3)
        - current_box : optional box matrix for deformed configurations
        - transform : optional precomputed transformation matrix T = A_base^{-1} @ A_current
    """
    modes, params = _modes_from_state(state)
    Mx, My, Mz = params.Mx, params.My, params.Mz
    P = params.P
    alpha = params.alpha
    fractional_coordinates = params.fractional_coordinates
    Pshape = modes["Pshape"]
    Bhalf = modes["Bhalf"]

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(params.volume, dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    V_sqrt = jnp.sqrt(V_box)

    base_A = jnp.asarray(params.A, dtype=REAL_DTYPE)
    base_inv = jnp.linalg.inv(base_A)

    @jax.jit
    def Mw_sqrt(key, positions, current_box=None, transform=None):
        A_curr = base_A if current_box is None else current_box
        if transform is None:
            transform = jnp.eye(3, dtype=REAL_DTYPE) if current_box is None else base_inv @ A_curr
        positions_frac_curr = positions_to_fractional(positions, A_curr, fractional_coordinates)
        positions_frac = jnp.mod(positions_frac_curr @ transform.T, 1.0)
        st = build_stencils_frac(positions_frac, Mx, My, Mz, P, alpha)
        draw = _hermitian_gaussian_modes(key, (Mx, My, Mz, 3))
        modes_q = jnp.einsum('...ij,...j->...i', Bhalf_complex, draw)
        modes_q = Pshape[..., None] * modes_q
        u_grid = ifft_vec(modes_q)
        velocities = gather(u_grid, st, Mx, My, Mz)
        return noise_scale * (V_sqrt * velocities)

    return Mw_sqrt


def build_Mw_apply_and_sample(state: WaveSpaceState):
    """Fused deterministic + stochastic operator.
    
    Constructs a combined operator that computes both the deterministic mobility
    response (M^(w) · F) and the stochastic Brownian velocities (sqrt(M^(w)) · ξ)
    in a single pass. This is more efficient than calling the deterministic and
    stochastic operators separately because it shares the stencil computation
    and position processing.
    
    The operator computes:
      U_det = M^(w) · F
      U_sto = sqrt(2 kB T) * sqrt(M^(w)) · ξ
    
    where M^(w) = D† · P† · B · P · D is the wave-space mobility operator.
    
    Parameters
    ----------
    state : WaveSpaceState
        WaveSpaceState returned by build_wave_modes containing Pshape, Bfluid,
        Bhalf, grid metadata, and box properties.
    
    Returns
    -------
    Callable
        JIT-compiled function fused(key, positions, forces, current_box=None, transform=None)
        that returns (U_det, U_sto), both arrays of shape (N, 3).
        
        - key : PRNGKey for stochastic sampling
        - positions : array of particle positions (N, 3)
        - forces : array of particle forces (N, 3)
        - current_box : optional box matrix for deformed configurations
        - transform : optional precomputed transformation matrix T = A_base^{-1} @ A_current
    """
    modes, params = _modes_from_state(state)
    Mx, My, Mz = params.Mx, params.My, params.Mz
    P = params.P
    alpha = params.alpha
    fractional_coordinates = params.fractional_coordinates
    Pshape = modes["Pshape"]
    Bfluid = modes["Bfluid"]
    Bhalf = modes["Bhalf"]

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(params.volume, dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    V_sqrt = jnp.sqrt(V_box)

    base_A = jnp.asarray(params.A, dtype=REAL_DTYPE)
    base_inv = jnp.linalg.inv(base_A)

    @jax.jit
    def fused(key, positions, forces, current_box=None, transform=None):
        A_curr = base_A if current_box is None else current_box
        if transform is None:
            transform = jnp.eye(3, dtype=REAL_DTYPE) if current_box is None else base_inv @ A_curr
        positions_frac_curr = positions_to_fractional(positions, A_curr, fractional_coordinates)
        positions_frac = jnp.mod(positions_frac_curr @ transform.T, 1.0)
        st = build_stencils_frac(positions_frac, Mx, My, Mz, P, alpha)

        force_grid = spread(forces, st, Mx, My, Mz)
        force_grid = sigma_inv * force_grid
        force_q = fft_vec(force_grid)
        P_force_q = Pshape[..., None] * force_q
        BP_force_q = jnp.einsum('...ij,...j->...i', Bfluid, P_force_q)
        Uq_det = Pshape[..., None] * BP_force_q
        u_grid_det = ifft_vec(Uq_det)
        velocities_det = gather(u_grid_det, st, Mx, My, Mz)
        U_det = V_box * velocities_det

        modes_q = _hermitian_gaussian_modes(key, (Mx, My, Mz, 3))
        modes_q = jnp.einsum('...ij,...j->...i', Bhalf_complex, modes_q)
        modes_q = Pshape[..., None] * modes_q
        u_grid_sto = ifft_vec(modes_q)
        velocities_sto = gather(u_grid_sto, st, Mx, My, Mz)
        U_sto = noise_scale * (V_sqrt * velocities_sto)

        return U_det, U_sto

    return fused


def build_Mw_grand_sqrt_sampler(state: WaveSpaceState):
    """Build the stochastic sampler for the wave-space *grand* mobility.

    Returns velocities and velocity gradients with joint covariance
    M^(w)_grand by applying the analytic per-mode square root: with the
    readout map R = [Pshape I ; +i Pdip u (x) k] (the adjoint of the grand
    moment-to-source map, see ``rpy_wave_det_dipole``), the deterministic
    operator factors as M^(w) = R B R†, so ``R B^{1/2} xi`` with Hermitian
    Gaussian grid noise ``xi`` has covariance M^(w)_grand exactly -- no
    iteration.  Hermitian conjugacy of the noise (and evenness of
    Pshape/Pdip/B in k together with the odd i*k factor) makes both
    gathered channels real-valued.

    Parameters
    ----------
    state : WaveSpaceState
        Grand wave state from ``build_grand_wave_modes`` (must carry the
        ``Pdip`` mode array).

    Returns
    -------
    Callable
        JIT-compiled ``Mw_grand_sqrt(key, positions, current_box=None,
        transform=None) -> (U_half (N, 3), D_half (N, 3, 3))`` whose outputs
        have joint covariance M^(w)_grand (no kT/dt scaling).
    """
    from jax_md.hydro.rpy_moments import (
        N_MOMENTS,
        components_to_couplet,
        traceless,
    )

    modes, params = _modes_from_state(state)
    if "Pdip" not in modes:
        raise ValueError(
            "build_Mw_grand_sqrt_sampler needs a grand WaveSpaceState "
            "(missing 'Pdip'); build it with build_grand_wave_modes.")
    Mx, My, Mz = params.Mx, params.My, params.Mz
    P = params.P
    alpha = params.alpha
    fractional_coordinates = params.fractional_coordinates
    Pshape = modes["Pshape"]
    Pdip = modes["Pdip"]
    k = modes["k"]
    Bhalf = modes["Bhalf"]

    Bhalf_complex = jnp.asarray(Bhalf, dtype=COMPLEX_DTYPE)
    Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
    V_box = jnp.asarray(params.volume, dtype=REAL_DTYPE)
    sigma_inv = Ngrid / V_box
    noise_scale = jnp.sqrt(sigma_inv * Ngrid)
    V_sqrt = jnp.sqrt(V_box)

    base_A = jnp.asarray(params.A, dtype=REAL_DTYPE)
    base_inv = jnp.linalg.inv(base_A)

    @jax.jit
    def Mw_grand_sqrt(key, positions, current_box=None, transform=None):
        A_curr = base_A if current_box is None else current_box
        if transform is None:
            transform = jnp.eye(3, dtype=REAL_DTYPE) if current_box is None else base_inv @ A_curr
        positions_frac_curr = positions_to_fractional(positions, A_curr, fractional_coordinates)
        positions_frac = jnp.mod(positions_frac_curr @ transform.T, 1.0)
        st = build_stencils_frac(positions_frac, Mx, My, Mz, P, alpha)

        out_q = grand_readout_noise_modes(
            key, Pshape=Pshape, Pdip=Pdip, k=k, Bhalf_complex=Bhalf_complex,
            Mx=Mx, My=My, Mz=Mz)
        out_grid = ifft_vec(out_q)
        out = noise_scale * (V_sqrt * gather(out_grid, st, Mx, My, Mz))
        U_half = out[..., :3]
        D_half = traceless(components_to_couplet(out[..., 3:N_MOMENTS]))
        return U_half, D_half

    return Mw_grand_sqrt


__all__ = [
    "build_Mw_sqrt_sampler",
    "build_Mw_apply_and_sample",
    "build_Mw_grand_sqrt_sampler",
]
