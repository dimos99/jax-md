# Hard-sphere Brownian dynamics (Strating) with shear and periodic boundaries

This note describes the method implemented by `simulate.brownian_hard_sphere` for overdamped Brownian dynamics of **hard spheres** under **periodic boundary conditions** and (optional) **simple shear** using a shearing cell (`space.shearing`, i.e. Lees–Edwards-type BCs).

The goal is to advance an overdamped system with thermal noise **without ever producing overlaps** (interpenetration), by interpreting each Brownian displacement over a step as a constant velocity and resolving overlaps as **elastic binary collisions** processed in correct time order.

---

## What is being simulated

We evolve particle positions `R = {r_i}` using an overdamped Langevin / Brownian update over a timestep `dt`:

$$
\Delta r_i = \mu_i F_i(R,t)\,dt + \sqrt{2 \mu_i kT\,dt}\,\xi_i,
\quad \xi_i \sim \mathcal{N}(0, I),
$$

optionally superposed with an imposed affine shear flow with instantaneous shear rates $\dot\gamma$.

Hard-sphere constraints enforce

$$
|r_i - r_j| \ge \sigma \quad \text{for all } i \ne j,
$$

where $\sigma$ is the particle diameter (`diameter`).

Hydrodynamic interactions are **not** included in this integrator (no long-range mobility coupling); `mobility` can be scalar or per-particle but acts locally.

---

## Coordinate conventions (JAX MD)

JAX MD spaces can represent positions in either **real coordinates** or **fractional coordinates**.

`simulate.brownian_hard_sphere` supports both via `fractional_coordinates`:

- If `fractional_coordinates=True`:
  - `R` is stored in fractional coordinates in $[0,1)^d$.
  - **Displacements** returned by `displacement_fn` and passed to `shift_fn` are in **real space**.
  - With a time-dependent shearing box (`space.shearing(..., fractional_coordinates=True)`), the evolving box already accounts for the affine advection of fractional coordinates; the integrator should shift by the **peculiar** displacement only.

- If `fractional_coordinates=False`:
  - `R` and `dR` are both in real space.
  - The integrator must explicitly add the affine shear drift $u(r)\,dt$ to the peculiar displacement.

---

## The core idea: treat each Brownian displacement as a constant velocity

Following Strating’s “event-driven Brownian dynamics” viewpoint, interpret the per-particle displacement over `dt` as a constant **peculiar** velocity:

$$
v_i^{\mathrm{pec}} = \frac{\Delta r_i}{dt}.
$$

This converts each step into a short event-driven dynamics problem with constant velocities (piecewise constant if collisions occur).

Under shear, each particle also experiences an affine velocity $u(r)$. For example, in 2D xy shear:

$$
u(r) = (\dot\gamma_{xy} y, 0).
$$

The **total** velocity used for collision timing is:

$$
v_i^{\mathrm{tot}} = v_i^{\mathrm{pec}} + u(r_i).
$$

In practice, collision prediction only needs the **relative** affine velocity, which can be expressed using the separation $dr = r_i - r_j$:

$$
u(r_i) - u(r_j) = u(dr).
$$

This is what the implementation computes (see `_affine_relative_velocity` in `simulate.brownian_hard_sphere`).

---

## Event-driven time stepping within one `dt`

Within one outer step, the algorithm repeatedly advances to the next “event” until the full `dt` is consumed:

**Events**

1. **Hard-sphere collision** between a pair $(i,j)$, occurring when $|r_{ij}(t)| = \sigma$.
2. **Shear remap event** (only for `space.shearing(..., remap=True)` with fractional coordinates), occurring at half-integer strain crossings where the reduced shear branch changes.
3. **End of the timestep**.

**Loop**

At the current substep time `t_curr` within `[0, dt]`:

1. Compute candidate collision times for all candidate pairs (all pairs, or those given by a neighbor list).
2. Compute the next shear-remap crossing time (if any) that lies ahead in the remainder of the step.
3. Choose `step = min(next_collision_time, next_remap_time, dt - t_curr)`.
4. Advance positions by `step` using `shift_fn` at the appropriate time.
5. If the event is a collision, apply an elastic collision update to the **peculiar** velocities.
6. If the event is a remap boundary, apply the unimodular remap to fractional coordinates and continue.

The loop is capped by `max_collision_loops` as a safety guard.

---

## Collision prediction for a pair

Let `dr` be the minimum-image displacement between particles $i$ and $j$ at the current time (in real space), and let `dv` be their relative total velocity:

$$
dr = r_i - r_j,
\qquad
dv = (v_i^{\mathrm{pec}} - v_j^{\mathrm{pec}}) + (u(r_i) - u(r_j)).
$$

A collision occurs when:

$$
\|dr + dv\,\tau\| = \sigma.
$$

Squaring gives a quadratic in $\tau$:

$$
a\tau^2 + b\tau + c = 0,
$$
with

$$
a = \|dv\|^2,\quad b = 2\,dr\cdot dv,\quad c = \|dr\|^2 - \sigma^2.
$$

The algorithm:

- Requires the pair to be **approaching** along the separation (`b < 0`).
- Takes the smallest nonnegative root:
  $$
  \tau = \frac{-b - \sqrt{b^2 - 4ac}}{2a}.
  $$
- Treats **existing overlaps** (`c < 0`) as an “immediate event” to be resolved at a small positive time `time_tol` to avoid stalling on $\tau \approx 0$.

---

## Elastic collision update (equal “masses”)

At the collision time, compute the unit normal $n = dr / \|dr\|$ (pointing from $j$ to $i$). Let $dv_{\mathrm{tot}}$ be the total relative velocity at contact.
For an elastic collision between equal masses, the normal component of the relative velocity is reversed. In the implementation this is written as an impulse along $n$ applied only if the pair is approaching (`dv_tot·n < 0`).

`brownian_hard_sphere` stores and updates **peculiar** velocities; affine contributions are re-added when needed for collision detection.

Small numerical overlaps after advancing are corrected by shifting the pair back to contact along $n$.

---

## Shear boxes, remap, and why remap is an event

`space.shearing(..., remap=True)` keeps the instantaneous shear strain in a reduced range (typically $[-0.5, 0.5)$) by subtracting a nearest integer:

$$
\gamma_{\mathrm{red}} = \gamma - \lfloor \gamma + 1/2 \rfloor.
$$

This improves conditioning of the box, but $\gamma_{\mathrm{red}}$ is **discontinuous** when $\gamma$ crosses a half-integer.

If positions are stored in fractional coordinates, the corresponding change-of-basis must be applied to the coordinates at the crossing time so that **real-space trajectories remain continuous**.

The remap is a unimodular transformation:

- **2D (xy):**
  $$
  x' = x + m_{xy}\,y \pmod 1,\quad y' = y
  $$
- **3D (xy, xz, yz):**
  $$
  x' = x + m_{xy}\,y + (m_{xz} + m_{xy}m_{yz})\,z \pmod 1,
  $$
  $$
  y' = y + m_{yz}\,z \pmod 1,\quad z' = z
  $$

Because the collision timing depends on a consistent displacement function across the step, the integrator treats remap crossings as events inside the same event loop as collisions.

Crossing times are computed assuming constant shear rates over the step (consistent with the constant-velocity assumption within `dt`).

---

## Collisional stress (hard-sphere)

`brownian_hard_sphere` accumulates a **collisional** stress tensor over each
step and stores it in `state.stress` using the manuscript event-weighted
estimator:

$$
\sigma = -\frac{1}{V\,dt}\sum_{\text{collisions}} r_c \otimes \Delta J_{\text{event}},
\quad r_c = \sigma\,\hat n,\quad
\Delta J_{\text{event}} = \frac{\Delta X_{\text{event}}}{\mu},
\quad \Delta X_{\text{event}} = \Delta v_i (dt - t_c).
$$

Here $\Delta v_i$ is the collision-induced jump in the **peculiar**
velocity of one particle at contact and $t_c$ is the event time within the
current step. The stress uses the remaining-time displacement correction
generated by that kick, not the numerical overlap-fix shift. This is the
hard-sphere constraint contribution only (no ideal term).

To normalize by volume, pass either a constant `box` or a `box_fn` (e.g. the
`box_of` returned by `space.shearing`). If neither is provided, `state.stress`
is returned as zeros. The manuscript collisional stress estimator is currently
implemented only for scalar positive mobility `μ`; array-valued or non-positive
mobility inputs are unsupported when collisional stress is enabled.

---

## Neighbor lists: supported formats and correctness contract

`apply_fn(state, neighbor=...)` accepts neighbor lists from `partition.neighbor_list` in any supported format:

- `Dense` (`neighbor.idx` shape `[N, max_occupancy]`)
- `Sparse` (`neighbor.idx` shape `[2, max_neighbors]`)
- `OrderedSparse` (`neighbor.idx` shape `[2, max_neighbors]` with `i < j`)

**Important correctness contract**

When a neighbor list is supplied, collision detection is performed **only for pairs present in the neighbor list**. To guarantee “no overlaps”, your neighbor list cutoff/skin must be conservative enough that any pair that could reach contact within one `dt` is included as a candidate.

Because Brownian increments are unbounded, a strict guarantee with a finite cutoff is not possible in a mathematical sense; in practice you choose `dt` small and a sufficiently large skin (or omit `neighbor` to check all pairs).

**Time-dependent boxes (shear) and `cell_size_too_small`**

When using a time-dependent triclinic box (e.g. `space.shearing(..., remap=True)`), the cell-list geometry that underpins `partition.neighbor_list` can become invalid as the box skews. In this case `neighbor.update(..., box=...)` may set `neighbor.cell_size_too_small=True`. This is a signal that you must **reallocate** the neighbor list using `neighbor_fn.allocate(...)` (typically with the current box, or a worst-case box for your shear range) before continuing; otherwise the neighbor list may miss candidate pairs.

---

## Implementation reference

- Integrator: `jax_md/simulate.py` → `brownian_hard_sphere`
- Shearing space: `jax_md/space.py` → `shearing(...)`
- Fractional remap utility: `jax_md/simulate.py` → `_apply_fractional_shear_remap`

## Visualization (presentation)

To generate a simple 2D “disks” movie that illustrates overlap handling and
secondary collisions in the event loop:

```bash
python docs/hard_sphere_bd_2d_algorithm_viz.py \
  --out-prefix docs/_static/hard_sphere_bd_2d_algorithm \
  --movie
```

## References (method)

- Strating, R. P. A. “Brownian dynamics simulation of a hard-sphere suspension.”
  *Phys. Rev. E* **59**, 2175 (1999). DOI: 10.1103/PhysRevE.59.2175
- Scala, A., Voigtmann, T., & De Michele, C. “Event-driven Brownian dynamics for hard spheres.”
  *J. Chem. Phys.* **126**, 134109 (2007). DOI: 10.1063/1.2719190

---

## Minimal usage example (shear, fractional coordinates)

```python
from jax import random
import jax.numpy as jnp
from jax_md import space, simulate

box = jnp.eye(2) * 10.0
shear_rate = 1.0
shear_schedule = lambda t: shear_rate * t

displacement, shift, _ = space.shearing(
  box,
  shear_schedule=shear_schedule,
  fractional_coordinates=True,
  remap=True,
)

init_fn, apply_fn = simulate.brownian_hard_sphere(
  energy_or_force_fn=lambda R, **kwargs: 0.0,
  displacement_fn=displacement,
  shift_fn=shift,
  dt=1e-5,
  kT=1.0,
  diameter=1.0,
  mobility=1.0,
  shear_schedule=shear_schedule,
  fractional_coordinates=True,
  remap=True,
)

key = random.PRNGKey(0)
R0 = random.uniform(key, (128, 2))  # fractional coords in [0,1)
state = init_fn(key, R0)
state = apply_fn(state)  # advances time by dt, no overlaps
```
