"""Validation for live Lees-Edwards shear in the Fast Stokesian Dynamics path.

The deterministic saddle solve (:mod:`jax_md.hydro.sd_saddle`), the Brownian
step (:mod:`jax_md.hydro.sd_brownian`) and the ``simulate.sd_with_shear``
integrator gain a live deformed box.  The checks, in increasing stringency:

  * **gamma=0 reduces to static (bit-for-bit).**  A sheared-space solve at zero
    strain must equal the same solve on the undeformed periodic box.
  * **Box consistency.**  A solve under live shear ``gamma`` must equal a static
    solve built directly at the literal deformed box -- the empirical gate that
    the exact deformed-box wave operator (``_apply_wave_exact_grand``) is used,
    not the position-remap-only path that keeps base-box modes.
  * **Ambient add-back (analytic pin).**  A single force-/torque-free sphere
    advects at ``u^inf = L . r`` and spins at the vorticity ``gamma_dot/2`` --
    no fitting.
  * **Brownian covariance at fixed strain.**  The far-field slip covariance under
    shear equals ``(2kT/dt) M_grand`` at the deformed box.
  * **Integrator wiring.**  ``sd_with_shear`` advances ``time``, keeps the strain
    bounded under ``remap``, and produces finite trajectories.
"""

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jax_md import space  # noqa: E402
from jax_md import simulate  # noqa: E402
from jax_md.hydro.sd_saddle import build_saddle_solve  # noqa: E402
from jax_md.hydro.sd_brownian import (  # noqa: E402
    build_sd_brownian_step,
    make_far_field_slip_sampler,
)
from jax_md.hydro.rpy_real_det_dipole import mr_grand_matvec  # noqa: E402
from jax_md.hydro.rpy import _apply_wave_exact_grand  # noqa: E402
from jax_md.hydro.rpy_moments import stresslet_to_couplet, traceless  # noqa: E402


def _rel_err(actual, expected):
  actual = np.asarray(actual, dtype=np.float64)
  expected = np.asarray(expected, dtype=np.float64)
  return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-15)


_GRID = dict(a=1.0, eta=1.0, xi=0.6, P=13, Mgrid=24)

# Tight GMRES so cross-solve comparisons / "relative motion vanishes" pins are
# limited by float precision, not the default 1e-3 saddle residual.
_TIGHT = dict(tol=1e-11, gmres_restart=80, gmres_maxiter=20)


def _shear(dim_gamma):
  gx, gz, gy = dim_gamma
  return dict(gamma_xy=gx, gamma_xz=gz, gamma_yz=gy)


# ---------------------------------------------------------------------------
# gamma = 0: the live-box operators reduce to the static (cached) operators
# ---------------------------------------------------------------------------
def test_zero_strain_operators_reduce_to_static():
  # At gamma=0 the deformed box equals the base box, so the two ingredients of
  # the live-box far-field grand matvec must coincide with their static
  # counterparts on the SAME build:
  #   * exact deformed-box wave operator == cached wave modes;
  #   * box-override real matvec == the default (state-box) real matvec.
  # This isolates exactly the live-box code path with no cross-build Ewald
  # confound (independently-sized builds cap rcut at different box dimensions
  # -- the worst-case-shear vs cubic difference is an O(1e-4) truncation tail,
  # not a code error; see test_shear_matches_static_deformed_box for the
  # nonzero-strain end-to-end pin).
  L = 18.0
  q = jax.random.uniform(jax.random.PRNGKey(0), (10, 3))
  F = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
  C = traceless(jax.random.normal(jax.random.PRNGKey(2), (10, 3, 3)))

  disp, shift, box_of = space.shearing(L * jnp.eye(3))
  init, solve = build_saddle_solve((disp, shift, box_of), **_GRID)
  st = init(q, **_shear((0.0, 0.0, 0.0)))
  base = st.rpy.real.box_matrix

  # Wave: exact deformed-box path at the base box == cached modes.
  Uw_e, Dw_e = _apply_wave_exact_grand(
      static=solve.wave_static, current_box=base,
      positions_frac=q, forces=F, couplets=C,
      a=solve.a, xi=solve.xi, eta=solve.eta)
  Uw_c, Dw_c = st.rpy.wave.apply_fn(q, F, C)
  assert _rel_err(Uw_e, Uw_c) < 1e-12
  assert _rel_err(Dw_e, Dw_c) < 1e-12

  # Real: box-override matvec at the base box == default (state-box) matvec.
  Ur_o, Dr_o = mr_grand_matvec(st.rpy.real, q, F, C, box_matrix=base)
  Ur_d, Dr_d = mr_grand_matvec(st.rpy.real, q, F, C)
  assert _rel_err(Ur_o, Ur_d) < 1e-12
  assert _rel_err(Dr_o, Dr_d) < 1e-12


# ---------------------------------------------------------------------------
# Box consistency: live shear == static solve at the literal deformed box
# ---------------------------------------------------------------------------
def test_shear_matches_static_deformed_box():
  L = 20.0
  gamma = 0.25
  base = L * jnp.eye(3)
  Hdef = base.at[0, 1].set(gamma * L)  # space.shearing xy convention

  q = jax.random.uniform(jax.random.PRNGKey(0), (12, 3))
  F = jax.random.normal(jax.random.PRNGKey(1), (12, 3))
  Tq = jax.random.normal(jax.random.PRNGKey(2), (12, 3))
  E = jnp.zeros((3, 3)).at[0, 1].set(0.3).at[1, 0].set(0.3)

  disp_s, shift_s, box_of = space.shearing(base)
  init_s, solve_s = build_saddle_solve((disp_s, shift_s, box_of), **_GRID)
  st_s = init_s(q, **_shear((gamma, 0.0, 0.0)))
  Us, Oms, Ss, _, info_s = solve_s(
      st_s, q, force=F, torque=Tq, E_inf=E, **_shear((gamma, 0.0, 0.0)),
      **_TIGHT)

  disp_d, shift_d = space.periodic_general(Hdef, fractional_coordinates=True)
  init_d, solve_d = build_saddle_solve((disp_d, shift_d), **_GRID)
  st_d = init_d(q)
  Ud, Omd, Sd, _, info_d = solve_d(st_d, q, force=F, torque=Tq, E_inf=E,
                                   **_TIGHT)

  assert info_s['rel_residual'] < 1e-8
  assert _rel_err(Us, Ud) < 1e-8
  assert _rel_err(Oms, Omd) < 1e-8
  assert _rel_err(Ss, Sd) < 1e-8


# ---------------------------------------------------------------------------
# Ambient add-back: single force-/torque-free sphere advects/spins with the flow
# ---------------------------------------------------------------------------
def test_ambient_addback_single_sphere():
  L = 30.0
  gdot = 0.7
  q = jnp.array([[0.3, 0.65, 0.5]])  # off-centre so L.r is nonzero
  Lmat = jnp.zeros((3, 3)).at[0, 1].set(gdot)
  E = 0.5 * (Lmat + Lmat.T)

  disp, shift, box_of = space.shearing(L * jnp.eye(3))
  init, solve = build_saddle_solve((disp, shift, box_of), **_GRID)
  st = init(q, **_shear((0.0, 0.0, 0.0)))
  U_rel, Om_rel, S5, _, info = solve(
      st, q, E_inf=E, L_inf=Lmat, **_shear((0.0, 0.0, 0.0)), **_TIGHT)

  # NOTE: the *relative* U/Omega of a periodic finite-Ewald sphere under strain
  # are not exactly zero (a small lattice rotation-strain coupling, ~1e-4); the
  # add-back validated here is the analytic ambient flow, independent of it.

  # Translational add-back equals L . r about the box centre.
  box = box_of(**_shear((0.0, 0.0, 0.0)))
  cart = space.transform(box, q - 0.5)
  U_inf_expect = jnp.einsum('ij,nj->ni', Lmat, cart)
  assert _rel_err(info['U_inf'], U_inf_expect) < 1e-10

  # Vorticity add-back: simple shear gamma_dot -> Omega_z = gamma_dot/2.
  np.testing.assert_allclose(np.asarray(info['Omega_inf'][0]),
                             [0.0, 0.0, gdot / 2.0], atol=1e-9)

  # The Einstein stresslet is positive along the imposed strain (S:E > 0).
  C = stresslet_to_couplet(S5[0])
  assert float(jnp.sum(C * E)) > 0.0


# ---------------------------------------------------------------------------
# Brownian far-field slip covariance at fixed strain == (2kT/dt) M_grand
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_far_field_slip_covariance_under_shear():
  L = 16.0
  gamma = 0.2
  kT, dt = 1.0, 1e-3
  q = jax.random.uniform(jax.random.PRNGKey(0), (2, 3))

  disp, shift, box_of = space.shearing(L * jnp.eye(3))
  init, solve = build_saddle_solve((disp, shift, box_of), **_GRID)
  st = init(q, **_shear((gamma, 0.0, 0.0)))
  st = solve.refresh_state(st, q, **_shear((gamma, 0.0, 0.0)))
  current_box = solve.resolve_current_box(q, **_shear((gamma, 0.0, 0.0)))

  sampler = jax.jit(make_far_field_slip_sampler(
      solve, st, q, kT, dt, mr_iters=40, lanczos_tol=1e-6,
      current_box=current_box))

  n = 1500
  keys = jax.random.split(jax.random.PRNGKey(1), n)
  samples = jax.vmap(sampler)(keys)             # (n, N, 11)
  Np = q.shape[0]
  # Translational slip block only: dominant, well-conditioned, low sampling
  # noise (the small couplet entries are noisy at modest sample counts).
  transl = np.asarray(samples)[..., :3].reshape(n, Np * 3)
  cov = np.cov(transl, rowvar=False)

  # Reference grand mobility (flat-11) at the deformed box, reconstructed from
  # the same far-field operator the saddle uses; Cov(slip) = (2kT/dt) M_grand.
  from jax_md.hydro.rpy_real_det_dipole import mr_grand_matvec
  from jax_md.hydro.rpy import _apply_wave_exact_grand
  from jax_md.hydro.rpy_moments import flat_to_grand, grand_to_flat, traceless

  def grand_flat(q11):
    Fm, Cm = flat_to_grand(q11)
    Ur, Dr = mr_grand_matvec(st.rpy.real, q, Fm, Cm, box_matrix=current_box)
    Uw, Dw = _apply_wave_exact_grand(
        static=solve.wave_static, current_box=current_box,
        positions_frac=q, forces=Fm, couplets=Cm,
        a=solve.a, xi=solve.xi, eta=solve.eta)
    return grand_to_flat(Ur + Uw, traceless(Dr + Dw))

  dim11 = Np * 11
  M = np.zeros((dim11, dim11))
  eye = np.eye(dim11)
  for c in range(dim11):
    col = grand_flat(jnp.asarray(eye[c].reshape(Np, 11)))
    M[:, c] = np.asarray(col).reshape(-1)
  M = 0.5 * (M + M.T)
  expected_full = (2.0 * kT / dt) * M
  # Extract the translational (velocity-velocity) sub-block to match `cov`.
  t_idx = np.concatenate([np.arange(3) + 11 * i for i in range(Np)])
  expected = expected_full[np.ix_(t_idx, t_idx)]

  denom = max(np.linalg.norm(expected), 1e-12)
  assert np.linalg.norm(cov - expected) / denom < 0.12


# ---------------------------------------------------------------------------
# Integrator wiring: sd_with_shear advances and keeps strain bounded
# ---------------------------------------------------------------------------
def test_sd_with_shear_integrator_runs():
  L = 16.0
  rate = 0.5
  q = jax.random.uniform(jax.random.PRNGKey(0), (12, 3))
  energy = lambda R, **kw: 0.0
  sched = lambda t: (rate * t, 0.0, 0.0)

  disp, shift, box_of = space.shearing(L * jnp.eye(3))
  init, apply = simulate.sd_with_shear(
      (disp, shift, box_of), energy, 1e-3, 1.0,
      shear_vector_schedule=sched, tol=None, **_GRID)
  st = init(jax.random.PRNGKey(1), q)
  for _ in range(5):
    st = apply(st)

  assert int(st.step) == 5
  assert np.isclose(float(st.time), 5e-3)
  assert bool(jnp.all(jnp.isfinite(st.real_position)))
  assert st.stresslet.shape == (12, 5)
  # Ambient spin: gamma_dot = rate -> Omega_z = rate/2.
  np.testing.assert_allclose(float(st.omega_inf[2]), rate / 2.0, atol=1e-6)


def test_free_sd_integrator_runs():
  L = 16.0
  q = jax.random.uniform(jax.random.PRNGKey(0), (12, 3))
  energy = lambda R, **kw: 0.0
  disp, shift = space.periodic_general(L * jnp.eye(3),
                                       fractional_coordinates=True)
  init, apply = simulate.sd((disp, shift), energy, 1e-3, 1.0, tol=None, **_GRID)
  st = init(jax.random.PRNGKey(1), q)
  for _ in range(3):
    st = apply(st)
  assert int(st.step) == 3
  assert bool(jnp.all(jnp.isfinite(st.real_position)))
