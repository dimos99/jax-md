"""Wave-space grand RPY mobility with force and traceless couplet moments.

Extends the deterministic wave-space operator of ``rpy_wave_det`` to the grand
mobility ``[U, D] = M^w [F, C]`` (Fiore & Swan 2018, Eqs. 4.20-4.23).  The
moment-to-force-density map and its exact adjoint are

  f_hat_m  = Pshape F_m - i Pdip k_n C_mn      (force + couplet sources)
  U_hat    = Pshape u_hat
  D_hat_ij = +i Pdip k_j u_hat_i               (velocity-gradient readout)

with ``Pshape = sinc(ka)``, ``Pdip = 3 j1(ka)/(ka)``, and the
Hasimoto-screened Stokeslet ``B`` unchanged and moment-agnostic.  Mutual
adjointness of the C -> f map (-i) and the u -> D map (+i) makes the
wave-space grand operator symmetric positive semi-definite by construction;
the jnp.fft sign convention ``u(x) = sum_k u_hat e^{+i k.x}`` fixes
``d/dx_n <-> +i k_n``.

Couplet index and packing conventions live in ``rpy_moments``; all 11 moment
channels (3 force + 8 drop-zz couplet components) share a single
spread/FFT/gather pass.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jax_md.hydro.rpy_moments import (
    N_MOMENTS,
    couplet_to_components,
    components_to_couplet,
    traceless,
)
from jax_md.hydro.rpy_wave_det import (
    WaveSpaceState,
    build_wave_modes,
)
from jax_md.hydro.rpy_wave_det_helpers import (
    REAL_DTYPE,
    make_reciprocal,
    positions_to_fractional,
    q_grid,
    sinc,
    build_Pdip_modes,
    build_stencils_frac,
    spread,
    gather,
    fft_vec,
    ifft_vec,
)


def build_grand_wave_modes(A,
                           a,
                           xi,
                           eta,
                           Mx,
                           My,
                           Mz,
                           P_support,
                           theta=None,
                           *,
                           fractional_coordinates: bool = True,
                           attach_sqrt: bool = False) -> WaveSpaceState:
  """Precompute grand wave-space modes and attach the cached grand matvec.

  Identical to ``build_wave_modes`` plus the dipole shape factor ``Pdip``;
  the returned state's ``apply_fn`` is ``make_grand_wave_matvec`` over the
  precomputed modes.  With ``attach_sqrt=True`` (constrained Brownian
  dynamics) the grand stochastic sampler from
  ``rpy_wave_stoch.build_Mw_grand_sqrt_sampler`` is attached as ``sqrt_fn``;
  the default leaves it off (the deterministic grand mobility has no Brownian
  dipole moments).
  """
  state = build_wave_modes(
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
  modes = dict(state.modes)
  modes["Pdip"] = build_Pdip_modes(modes["K"], a)
  template = WaveSpaceState(params=state.params, modes=modes)
  sqrt_fn = None
  if attach_sqrt:
    from jax_md.hydro.rpy_wave_stoch import build_Mw_grand_sqrt_sampler
    sqrt_fn = build_Mw_grand_sqrt_sampler(template)
  return WaveSpaceState(
      params=template.params,
      modes=template.modes,
      apply_fn=make_grand_wave_matvec(template),
      sqrt_fn=sqrt_fn,
  )


# Alias kept for naming parity with the force-only ``build_Mw_state``; used by
# ``build_rpy_mobility`` when ``use_stresslet=True``.
build_Mw_grand_state = build_grand_wave_modes


def make_grand_wave_matvec(
    state: WaveSpaceState,
) -> Callable[[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], Tuple[jnp.ndarray, jnp.ndarray]]:
  """Return Mw_grand(positions, forces, couplets) using precomputed modes."""
  modes = state.modes
  params = state.params
  fractional_coordinates = params.fractional_coordinates
  Mx, My, Mz, P = params.Mx, params.My, params.Mz, params.P
  alpha = params.alpha
  Bfluid = modes["Bfluid"]
  Pshape = modes["Pshape"]
  Pdip = modes["Pdip"]
  k = modes["k"]
  Ngrid = jnp.asarray(Mx * My * Mz, dtype=REAL_DTYPE)
  V_box = jnp.asarray(params.volume, dtype=REAL_DTYPE)
  sigma_inv = Ngrid / V_box

  base_A = jnp.asarray(params.A, dtype=REAL_DTYPE)
  base_inv = jnp.linalg.inv(base_A)

  @jax.jit
  def Mw_core(positions, forces, couplets=None, current_box=None, transform=None):
    """Apply grand wave mobility under the precomputed or supplied box."""
    A_curr = base_A if current_box is None else current_box
    if transform is None:
      transform = jnp.eye(3, dtype=REAL_DTYPE) if current_box is None else base_inv @ A_curr

    positions_frac_curr = positions_to_fractional(positions, A_curr, fractional_coordinates)
    positions_frac = jnp.mod(positions_frac_curr @ transform.T, 1.0)
    forces_local = jnp.asarray(forces, dtype=REAL_DTYPE)
    if couplets is None:
      couplets_local = jnp.zeros(forces_local.shape[:-1] + (3, 3), dtype=REAL_DTYPE)
    else:
      couplets_local = traceless(jnp.asarray(couplets, dtype=REAL_DTYPE))

    # Factored grand operator D^dagger P^dagger B P D over all 11 channels in a
    # single spread/FFT/gather pass (Fiore thesis Eq. 3.16-3.19 + 4.20-4.23).
    # D: NUFFT spread the (force, couplet) moments to the grid, then FFT.
    moments = jnp.concatenate(
        [forces_local, couplet_to_components(couplets_local)], axis=-1)
    st = build_stencils_frac(positions_frac, Mx, My, Mz, P, alpha)
    moment_grid = sigma_inv * spread(moments, st, Mx, My, Mz)
    moment_q = fft_vec(moment_grid)
    Fq = moment_q[..., :3]
    Cq = components_to_couplet(moment_q[..., 3:N_MOMENTS])

    # P: particle shape factors map moments to a force density. The couplet
    # source -i Pdip C.k and the velocity-gradient readout +i Pdip k u are
    # mutual adjoints, which keeps the grand operator symmetric PSD.
    fq = (Pshape[..., None] * Fq -
          1j * Pdip[..., None] * jnp.einsum("...mn,...n->...m", Cq, k))
    # B: Hasimoto-screened Stokeslet maps force density to flow.
    uq = jnp.einsum("...ij,...j->...i", Bfluid, fq)
    # P^dagger: read velocity (sinc) and velocity gradient (i Pdip k) back out.
    Uq = Pshape[..., None] * uq
    Dq = 1j * Pdip[..., None, None] * jnp.einsum("...i,...j->...ij", uq, k)

    # D^dagger: inverse FFT and NUFFT gather back to particles.
    out_q = jnp.concatenate([Uq, couplet_to_components(Dq)], axis=-1)
    out_grid = ifft_vec(out_q)
    out = V_box * gather(out_grid, st, Mx, My, Mz)
    velocities = out[..., :3]
    gradients = components_to_couplet(out[..., 3:N_MOMENTS])
    return velocities, traceless(gradients)

  return Mw_core


def mw_grand_matvec(state: WaveSpaceState,
                    positions: jnp.ndarray,
                    forces: jnp.ndarray,
                    couplets: Optional[jnp.ndarray] = None,
                    *,
                    current_box=None,
                    transform=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Apply the wave-space grand mobility using an existing state."""
  if not isinstance(state, WaveSpaceState):
    raise ValueError("mw_grand_matvec expects a WaveSpaceState.")
  apply_fn = state.apply_fn or make_grand_wave_matvec(state)
  return apply_fn(positions, forces, couplets, current_box=current_box, transform=transform)


def _mw_bruteforce_grand(t, F, C, A, a, xi, eta, Mx, My, Mz, P, theta=None):
  """Direct O(N^2 Nk) k-space grand mobility over the FFT mode set.

  Independent of the NUFFT machinery (no spreading, deconvolution, or FFTs);
  pins the phases, signs, and shape factors of ``make_grand_wave_matvec`` in
  tests.  Testing only.
  """
  del P, theta
  Brecip = make_reciprocal(A)
  V = jnp.linalg.det(A)
  QX, QY, QZ = q_grid(Mx, My, Mz)
  q = jnp.stack([QX, QY, QZ], axis=-1).reshape(-1, 3)
  k = jnp.einsum("ab,qb->qa", Brecip, q)
  K2 = jnp.sum(k * k, axis=1)
  K = jnp.sqrt(jnp.maximum(K2, 0.0))

  tH = (K / (2.0 * xi)) ** 2
  H = (1.0 + tH) * jnp.exp(-tH)
  Pshape = sinc(K * a)
  Pdip = build_Pdip_modes(K, a)

  scal = jnp.where(K2 > 0.0, H / (eta * V * K2), 0.0)
  eye = jnp.eye(3, dtype=REAL_DTYPE)
  denom = jnp.where(K2 > 0.0, jnp.sqrt(K2), 1.0)
  kh = k / denom[:, None]
  Pkk = eye - kh[..., None] * kh[:, None, :]
  Bfluid = scal[:, None, None] * Pkk
  Bfluid = jnp.where(K2[:, None, None] > 0.0, Bfluid, jnp.zeros_like(Bfluid))

  C = traceless(jnp.asarray(C, dtype=REAL_DTYPE))
  F = jnp.asarray(F, dtype=REAL_DTYPE)
  phase = jnp.exp(2j * jnp.pi * (t @ q.T))
  phi_ij = phase[:, None, :] * jnp.conj(phase)[None, :, :]

  fq = (Pshape[:, None, None] * F[None, :, :] -
        1j * Pdip[:, None, None] * jnp.einsum("pmr,qr->qpm", C, k))
  uq = jnp.einsum("qab,qnb->qna", Bfluid, fq)
  Uq = Pshape[:, None, None] * uq
  Dq = 1j * Pdip[:, None, None, None] * jnp.einsum("qni,qj->qnij", uq, k)

  U = jnp.einsum("ijq,qjv->iv", phi_ij, Uq).real
  D = jnp.einsum("ijq,qjmn->imn", phi_ij, Dq).real
  return U, traceless(D)
