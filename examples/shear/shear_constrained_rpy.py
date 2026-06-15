"""Stresslet-constrained RPY Brownian dynamics under shear.

A minimal, self-contained example of simulating a suspension of rigid spheres
with hydrodynamic interactions under simple shear, using the stresslet-
constrained mobility of Fiore & Swan (2018).

The script shows the full JAX-MD pattern end to end:

  1. Build a sheared, periodic box with ``space.shearing``.
  2. Pick any pair potential from ``jax_md.energy`` (soft spheres here).
  3. Build the integrator with ``simulate.constrained_rpy_with_shear`` -- this
     couples the constrained midpoint SDAE stepper to the Lees-Edwards shear
     bookkeeping (strain reduction + fractional remap), so the run stays stable
     for arbitrary total strain.
  4. ``jit`` the per-step ``apply_fn`` and advance in a Python loop.
  5. Read off the trajectory and the per-particle hydrodynamic stresslets that
     the constraint solve produces, and form a bulk hydrodynamic stress.

Run:
    python examples/shear/shear_constrained_rpy.py
    python examples/shear/shear_constrained_rpy.py --n 128 --steps 500 --gamma-dot 1.0
"""

import argparse

import jax

# 64-bit precision must be enabled before any jax_md import that fixes dtypes.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax_md import energy, simulate, space
from jax_md.hydro import stresslet_to_couplet


def build_simulation(args):
  """Return (init_fn, apply_fn, box, R0, key) for the configured run."""
  key = jax.random.PRNGKey(args.seed)
  key, pos_key = jax.random.split(key)

  # Cubic periodic box; positions live in fractional coordinates [0, 1)^3.
  base_box = jnp.eye(3) * args.box
  # Seed on a near-cubic lattice with a small jitter so the initial config has
  # no hard overlaps (a random fill would blow up the soft-sphere forces).
  side = int(jnp.ceil(args.n ** (1.0 / 3.0)))
  grid = jnp.stack(jnp.meshgrid(*([jnp.arange(side)] * 3), indexing="ij"), -1)
  lattice = grid.reshape(-1, 3)[:args.n] / side
  R0 = (lattice + 0.02 * jax.random.normal(pos_key, (args.n, 3))) % 1.0

  # Simple-shear schedule gamma_xy(t) = gamma_dot * t in the xy plane.
  # `remap=True` keeps the deformed box well conditioned; the integrator
  # applies the matching fractional-coordinate remap each step.
  scalar_schedule = lambda t: args.gamma_dot * t
  displacement, shift, box_of = space.shearing(
      base_box, shear_schedule=scalar_schedule,
      fractional_coordinates=True, remap=True)

  # Any jax_md pair potential works; soft spheres give a repulsive suspension.
  energy_fn = energy.soft_sphere_pair(
      displacement, sigma=args.sigma, epsilon=args.epsilon)

  # The integrator drives strain reduction from the *vector* schedule
  # (gamma_xy, gamma_xz, gamma_yz).
  shear_vector_schedule = lambda t: (args.gamma_dot * t, 0.0 * t, 0.0 * t)

  # `xi=None` (the default) lets the integrator size the Ewald split and grid
  # from `tol` via hydro.estimate_rpy_params; pass --xi to set it by hand.
  init_fn, apply_fn = simulate.constrained_rpy_with_shear(
      (displacement, shift, box_of),
      energy_fn,
      dt=args.dt,
      kT=args.kT,
      a=args.a,
      eta=args.eta,
      xi=args.xi,
      tol=args.tol,
      n_particles=args.n,
      shear_vector_schedule=shear_vector_schedule,
      P=args.P,
      Mgrid=args.Mgrid,
      mr_iters=args.mr_iters,
      capacity_multiplier=2.0,
      extra_capacity=32,
  )
  return init_fn, apply_fn, base_box, R0, key


def hydrodynamic_stress(state, volume):
  """Bulk hydrodynamic stress from the per-particle stresslets.

  The constraint solve returns each particle's stresslet in orthonormal
  ``(N, 5)`` coordinates; expand to the symmetric-traceless ``(N, 3, 3)``
  couplet and sum: ``sigma_H = -(1/V) sum_i S_i``.
  """
  stresslets = stresslet_to_couplet(state.brownian_state.stresslet)  # (N, 3, 3)
  return -jnp.sum(stresslets, axis=0) / volume


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--n", type=int, default=64, help="number of particles")
  parser.add_argument("--box", type=float, default=8.0, help="cubic box edge")
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--chunk", type=int, default=50,
                      help="steps between overflow checks")
  parser.add_argument("--dt", type=float, default=1e-3)
  parser.add_argument("--kT", type=float, default=1.0)
  parser.add_argument("--gamma-dot", type=float, default=0.5,
                      dest="gamma_dot", help="shear rate")
  # Potential.
  parser.add_argument("--sigma", type=float, default=0.6)
  parser.add_argument("--epsilon", type=float, default=2.0)
  # Mobility / Ewald parameters.
  parser.add_argument("--a", type=float, default=0.15,
                      help="hydrodynamic radius")
  parser.add_argument("--xi", type=float, default=None,
                      help="Ewald splitting parameter (default: auto from --tol)")
  parser.add_argument("--tol", type=float, default=1e-3,
                      help="target error for auto Ewald-parameter estimation")
  parser.add_argument("--eta", type=float, default=1.0, help="viscosity")
  parser.add_argument("--P", type=int, default=None, help="spreading support")
  parser.add_argument("--Mgrid", type=int, default=None, help="FFT grid size")
  parser.add_argument("--mr-iters", type=int, default=30, dest="mr_iters")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  init_fn, apply_fn, base_box, R0, key = build_simulation(args)
  volume = float(jnp.linalg.det(base_box))

  state = init_fn(key, R0)
  apply_jit = jax.jit(apply_fn)

  print(f"# constrained-RPY shear: N={args.n} box={args.box} "
        f"gamma_dot={args.gamma_dot} dt={args.dt} steps={args.steps}")
  print(f"# {'step':>6} {'time':>9} {'gamma_xy':>10} {'gamma_red':>10} "
        f"{'sigma_xy_H':>12}")

  done = 0
  while done < args.steps:
    chunk = min(args.chunk, args.steps - done)
    for _ in range(chunk):
      state = apply_jit(state)
    done += chunk

    overflow = bool(
        state.brownian_state.rpy_state.real.neighbors.did_buffer_overflow)
    if overflow:
      raise RuntimeError(
          "neighbor list overflowed; increase capacity_multiplier/"
          "extra_capacity, or drive the lower-level stepper with "
          "hydro.run_brownian_chunked for automatic replay.")

    t = float(state.time)
    gamma_xy = args.gamma_dot * t                 # total (unreduced) strain
    reduced = gamma_xy - jnp.floor(gamma_xy + 0.5)  # remapped into [-0.5, 0.5)
    # The reduced strain must stay bounded -- this is what the Lees-Edwards
    # remap inside the integrator guarantees (and what keeps the neighbor
    # list cell size valid over long runs).
    assert -0.5 <= float(reduced) < 0.5
    sigma = hydrodynamic_stress(state, volume)
    print(f"  {done:6d} {t:9.4f} {gamma_xy:10.4f} {float(reduced):10.4f} "
          f"{float(sigma[0, 1]):12.4e}")

  assert jnp.all(jnp.isfinite(state.real_position)), "non-finite positions"
  print("# done; trajectory and stresslets are on `state`.")


if __name__ == "__main__":
  main()
