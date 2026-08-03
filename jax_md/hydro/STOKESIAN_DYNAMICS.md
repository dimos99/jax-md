# Fast Stokesian Dynamics in `jax_md.hydro`

This note explains the Fast Stokesian Dynamics (FSD) implementation in this
directory. It is written for a scientific programmer who is comfortable with
linear operators, iterative solvers, and Brownian dynamics, but who may not know
the Stokesian Dynamics method in detail.

The implementation follows the Fiore and Swan FSD structure:

1. Use a positively split Ewald Rotne-Prager-Yamakawa (RPY) grand mobility for
   the far field.
2. Add the short-ranged lubrication resistance missing from that far-field
   approximation.
3. Solve the resulting resistance problem as a matrix-free saddle-point system.
4. For Brownian dynamics, inject far-field slip noise, near-field Brownian force,
   and random finite difference (RFD) drift into the same resistance solve.

The most relevant files are:

| File | Role |
| --- | --- |
| `rpy.py` | Public split-Ewald RPY mobility builder and grand-mobility plumbing. |
| `rpy_real_det.py` | Force-only real-space RPY mobility. |
| `rpy_real_det_dipole.py` | Real-space grand mobility for force plus couplet moments. |
| `rpy_wave_det.py` | Force-only Spectral Ewald wave-space mobility. |
| `rpy_wave_det_dipole.py` | Wave-space grand mobility for force plus couplet moments. |
| `rpy_moments.py` | Couplet, torque, stresslet, strain, and flat-coordinate conventions. |
| `sd_nearfield_table.py` | Loads and interpolates the FSD lubrication table. |
| `sd_nearfield.py` | Matrix-free near-field lubrication resistance `R^nf`. |
| `sd_saddle.py` | Deterministic FSD saddle-point solve. |
| `sd_brownian.py` | Brownian FSD step: split noise plus RFD drift. |
| `simulate.py` | User-facing `simulate.sd` and `simulate.sd_with_shear` wrappers. |

The original FSD code (Fiore & Swan, C++/CUDA) is the reference
implementation used for comparison. The
closest source-file correspondences are:

| JAX file | FSD file(s) |
| --- | --- |
| `sd_nearfield_table.py`, `data/extract_legacy_resistance_table.py` | `Stokes_ResistanceTable.cc` |
| `sd_nearfield.py` | `Lubrication.cu` |
| `sd_saddle.py` | `Saddle.cu`, `Precondition.cu`, `Solvers.cu` |
| `sd_brownian.py` | `Brownian_FarField.cu`, `Brownian_NearField.cu`, `Integrator.cu` |
| `rpy_*` | `Mobility.cu` and helpers |

## Internal Code Map

Where to find each piece inside the `sd_*` files. All names are module-level
functions unless noted; each carries a docstring with its conventions.

`sd_nearfield.py`:

| Piece | Where |
| --- | --- |
| Dimensional prefactors | `_kim_karrila_prefactors` (docstring explains the uniform Brady-Bossis scaling) |
| Scalar dimensionalization | `_dim_scalars` (all 22 table columns; family = second letter of the column name) |
| Pair tensor families | `_a_block` (A and C), `_g_tensor`, `_h_tensor`, `_m_tensor` |
| 5-basis projections | `_project_su`, `_project_se` |
| 11x11 block packing | `_assemble_gen_block` |
| Per-edge block orchestrator | `_build_pair_operators` (calls all of the above) |
| Fixed-configuration precompute | `PreparedNearField`, `apply_prepared_blocks`, `apply_prepared_blocks_FU`, `prepared_diag_FU` |
| Builder | `build_nearfield_resistance` (neighbor list, edge geometry, `apply_fn` + its attached methods) |

`sd_saddle.py` (the module docstring carries the same map with more detail):

| Piece | Where |
| --- | --- |
| `B` / `B^T` projectors | `rot_embed`, `b_apply`, `bt_apply`, `stresslet_from_moment` |
| Input normalization | `_resolve_e_inf`, `_normalize_solve_inputs` |
| Operator factories | `_make_grand_mv` (far field), `_make_rnf` (near field, one `prepare` pass) |
| Saddle matvec / RHS / add-back | `_saddle_operator`, `_saddle_rhs`, `_ambient_addback` |
| Preconditioners | `_block_ldl_pinv` (one shared block-LDL apply) + the Schur solves: jacobi inline, diagonal inline, `_make_cheb_schur_solve` (+ `_cheb_bounds`), `_make_ic0_schur_solve` |
| Builder | `build_saddle_solve` (jitted `_body_impl`/`_device_body`, `solve_fn`, eager `count_iterations`) |
| Host IC(0) machinery | `assemble_stilde`, `_ic0`, `Ic0Preconditioner`, `build_ic0_from_state` |

`sd_brownian.py`:

| Piece | Where |
| --- | --- |
| Far-field slip sampler | `make_far_field_slip_sampler` (+ `far_field_slip` single draw) |
| Near-field force sampler | `make_nearfield_brownian_sampler` (+ `nearfield_brownian_force`) |
| RFD drift | `rfd_drift` |
| Advection frame helper | `_sd_coordinate_velocity` |
| Timestep builder | `build_sd_brownian_step` (jitted `_step_core` + eager `step_fn` wrapper) |

The eager `count_iterations` harness and the jitted solve share
`_saddle_operator` / `_saddle_rhs` / the preconditioner factories, so the
validated operator is the solved operator by construction.

## Physical Problem

The particles are monodisperse rigid spheres of radius `a` in a Newtonian fluid
of viscosity `eta`. In the overdamped limit there is no inertia. Velocities are
determined instantaneously by a hydrodynamic resistance problem.

The full Stokesian Dynamics resistance couples particle translational and
angular velocities to applied forces and torques:

```text
[F, L] = R_FU [U, Omega] + imposed-flow terms.
```

The FSD approximation used here writes the resistance as

```text
R_FU = B^T M^{-1} B + R^nf_FU.
```

Here:

- `M` is the far-field grand mobility. It maps force and couplet moments to
  translational velocities and velocity gradients.
- `B` embeds rigid particle velocity degrees of freedom `[U, Omega]` into the
  grand velocity space used by `M`.
- `R^nf_FU` is the short-range near-field lubrication correction. It is
  pairwise additive and nonzero only for separations below `r_lub = 4a`.

This formula should be read as an operator identity. The implementation never
forms `M^{-1}` or `R_FU` as dense matrices in production paths.

## Coordinate Spaces and Conventions

There are three closely related vector spaces. Keeping them distinct prevents
most sign and factor-of-two mistakes.

### Moment space: flat 11 coordinates

The far-field grand mobility works on an 11-component vector per particle:

```text
q11 = [F(3), C(8)]
```

`F` is a Cartesian force. `C` is a traceless 3x3 couplet tensor represented in an
8-dimensional Frobenius-orthonormal basis. The helpers are in `rpy_moments.py`:

- `grand_to_flat(U, D)` packs `[U, D]` into flat-11 coordinates.
- `flat_to_grand(q11)` unpacks `[F, C]`.
- `couplet_to_orthonormal` and `orthonormal_to_couplet` convert the 8
  orthonormal tensor coordinates.

The orthonormal basis matters because GMRES and Lanczos use Euclidean dot
products. In these coordinates the grand mobility is symmetric. The FFT grid
internally uses a different "drop-zz" 8-component storage for efficiency, but
that representation is not used as the solver coordinate system.

### Rigid FU space: flat 6 coordinates

The saddle solve's physical rigid variables are

```text
u6 = [U(3), Omega(3)]
f6 = [F(3), L(3)]
```

`U` is translational velocity, `Omega` is angular velocity, `F` is force, and
`L` is torque.

### Near-field generalized space: flat 11 coordinates

The near-field resistance maps generalized velocities to generalized forces:

```text
g11 = [U(3), Omega(3), E5(5)]
h11 = [F(3), L(3), S5(5)]
```

`E5` and `S5` are the rate-of-strain and stresslet in the 5-dimensional
orthonormal symmetric traceless basis from `stresslet_basis()`.

### Couplet, torque, stresslet, and gradient definitions

The couplet split is

```text
C = S - (1/2) eps . L
```

where `S` is the symmetric traceless stresslet tensor and `L` is torque. The
inverse torque extraction is

```text
L_k = - eps_kmn C_mn.
```

The velocity gradient convention is

```text
D_ij = du_i / dx_j
E = sym(D)
Omega_k = -(1/2) eps_kij D_ij.
```

These choices match the FSD physical convention and keep the grand mobility
symmetric in `(F, L, S) -> (U, Omega, E)` coordinates.

## The `B` and `B^T` Operators

`sd_saddle.py` defines the exact adjoint pair:

```text
B [U, Omega] = grand_to_flat(U, rot_embed(Omega))
rot_embed(Omega)_ij = - eps_ijk Omega_k

B^T [F, C] = [F, L],
L_k = - eps_kmn C_mn
```

In code:

- `rot_embed(omega)` is the physical rigid-rotation velocity gradient. It is
  `2 * torque_to_couplet(omega)` because `torque_to_couplet` embeds a torque
  couplet with the `-1/2 eps` factor.
- `b_apply(u6)` implements `B`.
- `bt_apply(q11)` implements `B^T`.

The tests in `tests/sd_saddle_test.py` explicitly check

```text
<B u, q> = <u, B^T q>
```

and that `rot_embed` decomposes back to zero strain plus the original angular
velocity.

## Far-Field Grand Mobility

The far-field operator is the split-Ewald RPY grand mobility

```text
M = M^r + M^w.
```

It maps force and traceless couplet moments to velocities and velocity
gradients:

```text
[U, D] = M [F, C].
```

### Real-space part

The real-space part is in `rpy_real_det.py` and
`rpy_real_det_dipole.py`.

For force-only RPY, the pair block has the form

```text
M^r_ij F_j = (1 / (6 pi eta a))
             [F1 F_j + (F2 - F1) (F_j . rhat) rhat]
```

where `F1` and `F2` are Fiore closed-form Ewald coefficients. The grand version
adds:

- `UC`: velocity from a couplet, using `G1`, `G2`.
- `DF`: velocity gradient from a force, the adjoint of `UC` with the code's
  fixed sign convention.
- `DC`: velocity gradient from a couplet, using `K1`, `K2`, `K3`.

The real-space matvec uses neighbor lists and either lattice-image accumulation
or a minimum-image kernel, depending on the cutoff and box. State is stored in
`RealSpaceState`, including the neighbor list, lattice indices, current box, and
the compiled core function.

### Wave-space part

The wave-space part is in `rpy_wave_det.py` and
`rpy_wave_det_dipole.py`. It uses Spectral Ewald quadrature and has the factored
form

```text
M^w = D^dagger P^dagger B P D.
```

The pieces are:

- `D`: spread particle data to an FFT grid using Gaussian Spectral Ewald
  stencils.
- `P`: apply particle shape factors.
- `B`: apply the Hasimoto-screened Stokeslet with transverse projection.
- `D^dagger`: inverse FFT and gather grid data back to particles.

For force-only RPY:

```text
Pshape(k) = sinc(|k| a)
H(k, xi) = (1 + (|k| / (2 xi))^2) exp(-( |k| / (2 xi) )^2)
B(k) = H(k, xi) (I - khat khat) / (eta V |k|^2)
```

The zero mode is set to zero.

For the grand mobility, the force density and readout are

```text
f_hat_m = Pshape F_m - i Pdip C_mn k_n
D_hat_ij = + i Pdip k_j u_hat_i
Pdip(ka) = 3 (sin(ka) - ka cos(ka)) / (ka)^3.
```

The `-i` source map and `+i` gradient readout are adjoints, so the grand
wave-space operator is symmetric positive semidefinite.

### Live shear and deformed boxes

For static boxes, wave modes are cached in `WaveSpaceState`. For live
Lees-Edwards shear, the code must not reuse reciprocal vectors from the base
box. The SD path uses `_apply_wave_exact_grand` and
`_sample_wave_grand_noise` to rebuild the wave-space mode factors for the
current deformed reciprocal lattice while reusing static grid metadata such as
FFT sizes and stencil support.

The real-space and near-field states also carry the live box matrix so minimum
images and neighbor-list updates are consistent with the deformed geometry.

## Near-Field Lubrication Resistance

The near-field resistance is implemented in `sd_nearfield.py` and tabulated by
`sd_nearfield_table.py`.

### What the table contains

The committed file `jax_md/hydro/data/resistance_table_legacy.npz` is the legacy
regression reference extracted from FSD's `Stokes_ResistanceTable.cc` by
`data/extract_legacy_resistance_table.py`. It is retained for comparisons, but is not
used by default.

It contains 22 scalar functions at 1000 center-to-center distances

```text
s = r / a
gap = s - 2
gap in [1e-4, 2.0]
s in [2.0001, 4.0]
```

The columns are:

```text
XA11 XA12 YA11 YA12 YB11 YB12 XC11 XC12 YC11 YC12
XG11 XG12 YG11 YG12 YH11 YH12 XM11 XM12 YM11 YM12 ZM11 ZM12
```

These scalars are already the near-field correction

```text
R^nf = R^{2B, exact} - Rbar^{2B, far-field}.
```

That is, they are not the full two-body resistance. They are the missing
short-range piece that must be added to the far-field approximation.

The production `resistance_table.npz` is independently regenerated by
`data/generate_resistance_table.py`. It contains 2000 log-gap points over
`gap in [1e-8, 2]`: corrected Townsend near-field expressions through `0.01`,
a slope-limited bridge over `0.01..0.02`, and a JAX port of Wilson's
Lamb/reflection method thereafter. The generator also constructs and subtracts
the two-body FTS far-field resistance and removes its regenerated value at the
`s=4` cutoff. No rows are copied from the FSD archive.

Normal SD operator construction loads only this regenerated table. The legacy
archive is never read at runtime; the generator accepts it as an optional
offline comparison input through `--legacy-reference`.

### Difference from FSD's production clamp

FSD's CUDA kernels clamp very small gaps to row 232 of the legacy table,
corresponding to a roughness-regularized distance near `s = 2.000997`. This JAX
implementation uses the full production table from row 0 in float64.

This is an intentional difference. In float64 the runtime applies no additional
roughness clamp beyond the production table's minimum gap of `1e-8`. In
float32, `REGULARIZATION_INDEX` selects the first row at or above a gap of
`1e-5`, avoiding unresolved center-to-center spacing near `r/a = 2`.

### Interpolation

Given a separation `r`, `interpolate_scalars` computes:

```text
s = r / a
gap = s - 2
ind = floor(log10(gap / xi_min) / dr)
```

The index is clipped so that `ind` and `ind + 1` are valid. The value is then
linearly interpolated in the raw distance `s`, matching the FSD interpolation
contract.

The caller, not the table, enforces the lubrication cutoff:

```text
r < r_lub,   r_lub = 4a by default.
```

### Pair tensor construction

`_build_pair_operators(rhat, scalars, a, eta)` builds two `(11, 11)` blocks per
directed edge:

```text
h_i += R_self(i <- j)  g_i
     + R_cross(i <- j) g_j.
```

The unit vector `rhat` points from receiver particle `i` to sender particle
`j`, the same convention used in FSD's `Lubrication.cu`.

The tensor families are:

| Family | Block meaning |
| --- | --- |
| `A` | force from translation (`F-U`) |
| `B` | force/torque cross coupling (`F-Omega`, `L-U`) |
| `C` | torque from rotation (`L-Omega`) |
| `G` | stresslet from translation and force from strain (`S-U`, `F-E`) |
| `H` | stresslet from rotation and torque from strain (`S-Omega`, `L-E`) |
| `M` | stresslet from strain (`S-E`) |

The implementation writes the FSD tensor forms in vectorized `jax.numpy`
einsums and projects stresslet/strain tensors onto the orthonormal 5-basis.

The dimensional prefactors are a uniform Brady-Bossis scaling:

```text
A: 6 pi eta a
B: 6 pi eta a^2
C: 6 pi eta a^3
G: 6 pi eta a^2
H: 6 pi eta a^3
M: 6 pi eta a^3
```

This uniform scaling preserves the near-contact rank-one squeeze cancellation.
The comments in `sd_nearfield.py` explain why the textbook-looking
`pi eta (2a)^k` prefactors are not used here.

### Matrix-free apply

`build_nearfield_resistance` returns `(init_fn, apply_fn)`.

`init_fn` allocates a **Dense** neighbor list for the cutoff `r_lub` by default.
`neighbor_format=Sparse` is available and is the faster option on heterogeneous
configurations; the two agree **bit-for-bit**, since they enumerate the same
directed edges and segment-sum over the same receiver partition in the same
order, so the choice is purely a cost/allocation trade-off.

The edge count `E` sets the cost of every near-field matvec, and the
Chebyshev-Schur preconditioner does `cheb_degree` of them per GMRES iteration
(~1e4 per SD step). `Dense` sizes `E = N * max_k` from the single most-crowded
particle, so clustered configurations (gels) pay worst-case occupancy on *every*
particle — measured 2-3x more edges streamed than live, a penalty no capacity
multiplier can remove. `Sparse` sizes from the true pair count instead.

The flip side is that the two formats scale `capacity_multiplier` differently:
Dense's `N * max_k` buffer carries several times the live edge count as
*accidental* overflow slack, whereas Sparse headroom is on the *total* pair count
(1.25 really is 25%). Under Sparse, on runs whose coordination grows
(aggregation, quenches), size from the expected *final* pair count and check
`did_buffer_overflow`.

`OrderedSparse` is rejected: it keeps only `i < j`, and the matvec segment-sums
over receivers, so it needs both `i->j` and `j->i`.

With a shearing `box_fn`, allocation uses a worst-case shear neighbor box to
reduce capacity changes during a run.

`apply_fn(state, positions, gen_velocity, **kwargs)`:

1. Updates the neighbor list.
2. Flattens the neighbor list to directed receiver/sender edges. For Sparse,
   `jax_md` packs `idx = stack((receiver_idx, sender_idx))` with `sender_idx`
   the broadcast `arange(N)` *centre* — the opposite of the naming used here —
   so `idx[1]` is the receiver `i` and `idx[0]` the sender `j`, which makes the
   edge list agree with the Dense one (whose receiver is the row index).
   Overflow slots set *both* entries to `N`, so `partition.neighbor_list_mask`'s
   `idx[0] < N` test covers both gathers.
3. Computes minimum-image displacements under the current box.
4. Masks self edges and edges outside `r_lub`.
5. Interpolates the 22 scalars.
6. Builds per-edge pair blocks.
7. Segment-sums edge contributions into an `(N, 11)` generalized force.

For repeated matvecs at a fixed configuration there are two amortization
levels:

- `apply_prepared` reuses the already-built neighbor list (skips the update)
  but still re-derives geometry, table lookups, and blocks per matvec.
- `prepare` runs geometry + table + block assembly once and returns a
  `PreparedNearField` (pre-summed self blocks + masked per-edge cross blocks);
  `apply_blocks` / `apply_blocks_FU` then reduce each matvec to
  gather -> batched block multiply -> segment-sum. This is what the saddle
  solve and the Brownian step use on their hot paths, and one prepared object
  can be shared across consumers at the same configuration (see
  `solve_fn(..., prepared_nf=)` below).

The near-field builder also exposes `prepared_diag_FU`, `diagonal_FU_prepared`
and `diagonal_FU_mask_prepared`, which are used by the Schur preconditioner and
the near-field Brownian square root.

## Deterministic Saddle-Point Solve

`sd_saddle.build_saddle_solve` constructs the deterministic FSD resistance
solver.

The unknowns are:

```text
q11: far-field generalized moments, shape (N, 11)
u6 : relative rigid velocities [U, Omega], shape (N, 6)
```

The matrix is

```text
A = [ M       B        ]
    [ B^T    -R^nf_FU ]
```

and the matvec is implemented as

```text
top = M q + B u
bot = B^T q - R^nf_FU u.
```

For a deterministic solve with applied force/torque `fp6 = [F^P, L^P]` and
imposed strain `E_inf`, the right-hand side is

```text
b1 = [0, E_inf]                  in flat-11 grand velocity space
b2 = -(fp6 + R^nf_FE E_inf)      in FU force/torque space
```

The solve returns `u = [U_rel, Omega_rel]`. It is relative to the imposed
background flow. The background add-back is reported in `info`:

```text
U_inf = L_inf r
Omega_inf = (1/2) curl u_inf.
```

`E_inf` is the symmetric rate-of-strain used in the resistance RHS. `L_inf` is
the full ambient velocity gradient used for `U_inf` and `Omega_inf`. If `L_inf`
is not supplied, the code defaults it to the symmetric `E_inf`, which has no
ambient vorticity.

### Stresslet reconstruction

After GMRES, the total hydrodynamic stresslet is

```text
S = S^ff - R^nf_SU u + R^nf_SE E_inf.
```

Here `S^ff` is extracted from the far-field moment `q11`. The near-field
`SU` and `SE` applies are skipped when `return_stresslet=False`, because they
cost extra near-field matvecs.

### Brownian mean stresslet

At `kT > 0` the reported stresslet additionally carries the mean Brownian
stresslet (Foss & Brady 2000, Eq. 10c):

```text
<S^B> = -kT div(R_SU R_FU^{-1}).
```

Expanding by the product rule (with the FSD far/near splitting) gives three
terms: the far-field response divergence, `R^nf_SU div(R_FU^{-1})`, and
`(grad R^nf_SU) : R_FU^{-1}`. The last two diverge like `1/gap` near contact
with opposite signs and cancel at leading order (in a scalar caricature
`f = A/xi`, `g = B xi`: `f g' = +AB/xi`, `f' g = -AB/xi`, while `(fg)' = 0`).
What survives is much weaker sub-leading growth — measured on a two-sphere
pair (f64): the uncancelled shortcut term overstates the true drift stresslet
by ~15x at gap `1e-2 a` and ~35x at `1e-3 a`, growing ~11x per gap decade
(the 1/gap singularity) versus ~5x for the true term (comparable to the drift
velocity's own near-contact growth).

The Brownian step samples the full divergence with the drift stresslet

```text
S5_drift = (kT / eps) (S5_plus - S5_minus)
```

read from the *same* two displaced solves that produce the RFD drift velocity
(see below). Because each displaced solve assembles its stresslet from
near-field blocks prepared at the displaced positions, the estimator carries
all three product-rule terms, and the near-contact `1/gap` cancellation
happens inside the `+/-` subtraction where it is exact. Like the fluctuating
stresslet, `S5_drift` is a noisy single-sample estimate per step: only time
averages are physically meaningful.

### Public API

Typical deterministic use:

```python
from jax_md import space
from jax_md.hydro import build_saddle_solve
import jax.numpy as jnp

box = jnp.eye(3) * 20.0
space_fns = space.periodic_general(box, fractional_coordinates=True)

init_fn, solve_fn = build_saddle_solve(
    space_fns, a=1.0, eta=1.0, xi=0.6, P=16, Mgrid=48)

state = init_fn(positions_frac)
U_rel, Omega_rel, S5, q11, info = solve_fn(
    state, positions_frac, force=forces, torque=torques)

U_total = U_rel + info["U_inf"]
Omega_total = Omega_rel + info["Omega_inf"]
```

The solve function also accepts:

- `E_inf`: imposed rate-of-strain, as `(3, 3)` or `(N, 5)`.
- `L_inf`: full ambient velocity gradient for add-back.
- `x0`: warm start `(q11, u6)`.
- `zero_nearfield=True`: diagnostic reduction to far-field stresslet-constrained
  mobility.
- `preconditioner`: one of `'cheb'`, `'diag'`, `'ic0'`, or `'jacobi'`.
- `slip_top` and `extra_force`: Brownian/RFD injections used by
  `sd_brownian.py`.
- `prepared_nf`: near-field blocks already built by
  `solve_fn.nf_apply.prepare(state.nf, positions)`, letting several consumers
  at one fixed configuration share a single prepare pass (the Brownian step
  does this). Must match the positions and box the solve uses; default `None`
  prepares internally.

### Preconditioners

The exact saddle Schur complement is expensive because it contains `M^{-1}`.
The preconditioners approximate

```text
M^{-1} ~= zeta I,   zeta = 6 pi eta a,
S_tilde = zeta I + R^nf_FU.
```

The code uses a block-LDL inverse of the approximate saddle matrix. The
available Schur approximations are:

| Name | Description |
| --- | --- |
| `jacobi` | Uses only `zeta I`. Cheapest and weakest. |
| `diag` | Uses `zeta I + diag(R^nf_FU)`. Fully on-device. |
| `cheb` | Default. Matrix-free Chebyshev semi-iteration on `zeta I + R^nf_FU`, Jacobi-scaled. Fully on-device and jittable. |
| `ic0` | Host-side RCM plus zero-fill incomplete Cholesky of a truncated `zeta I + R^nf_FU` with cutoff `r_p = 2.1a` by default. Good for validation, not the default production path. |

The `ic0` path mirrors FSD's RCM/incomplete-Cholesky preconditioner most
closely, but it uses SciPy on the host and is applied through `jax.pure_callback`.
The Brownian timestep rejects `ic0` because the timestep is jitted end-to-end.

### State refresh

`solve_fn.refresh_state(state, positions, **shear_kwargs)` updates real-space
and near-field neighbor lists with shape-preserving `.update()` calls. This is
important for JIT reuse: reallocating neighbor lists changes array shapes and
can trigger recompilation.

If a neighbor-list buffer overflow flag is raised, the user must reallocate by
calling the original `init_fn` again with more capacity.

## Brownian Fast Stokesian Dynamics

`sd_brownian.py` implements the overdamped Brownian step.

The target stochastic velocity covariance is

```text
Cov([U, Omega]) = (2 kT / dt) R_FU^{-1}.
```

The method samples this without forming `R_FU^{-1/2}`. It uses the Fiore-Swan
split:

1. Draw a far-field slip in the saddle top block:

   ```text
   U_B^ff ~ N(0, (2 kT / dt) M).
   ```

2. Draw a near-field Brownian force in the saddle bottom block:

   ```text
   F_B^nf ~ N(0, (2 kT / dt) R^nf_FU).
   ```

3. Solve the same saddle system with `slip_top=U_B^ff` and
   `extra_force=F_B^nf`, plus deterministic applied forces.

4. Add the thermal drift `kT div R_FU^{-1}` using random finite differencing.

### Far-field slip sampler

`make_far_field_slip_sampler` returns a sampler for flat-11 grand slip with
covariance `(2 kT / dt) M`.

It reuses the existing positively split grand mobility sampler:

- real-space part: Lanczos square root of `M^r`;
- wave-space part: analytic Fourier square root of `M^w`;
- live shear: exact deformed-box wave-space noise via `_sample_wave_grand_noise`.

The far-field sample is injected into the saddle top block, whose units are
grand velocities `[U, D]`.

### Near-field Brownian force sampler

`make_nearfield_brownian_sampler` samples

```text
F_B^nf ~ N(0, (2 kT / dt) R^nf_FU)
```

in `(N, 6)` force/torque space.

The unpreconditioned version applies a Lanczos square root directly to the
matrix-free `R^nf_FU` operator.

The default preconditioned version uses an on-device diagonal conditioning
split. The subtle part is the treatment of particles with no lubrication
neighbors. For such particles, the corresponding rows of `R^nf_FU` are exactly
zero. The code uses two separate objects:

- `Shift_nn`: an additive conditioning shift used only inside the Lanczos
  operator for neighborless rows.
- `Proj`: a strict 0/1 projector applied only when unwinding the preconditioned
  sample.

This gives

```text
Cov(F) = c Proj (R^nf_FU + Shift_nn) Proj
       = c R^nf_FU,
c = 2 kT / dt.
```

The projector annihilates the artificial shift exactly. This separation is
checked in `tests/sd_brownian_test.py`, including torque rows.

### RFD thermal drift

`rfd_drift` estimates

```text
kT div R_FU^{-1}
```

with a centered random finite difference.

It draws a Gaussian `dq6 = [dq_pos, dq_rot]`, displaces positions by

```text
q_plus  = q + (eps / 2) dq_pos
q_minus = q - (eps / 2) dq_pos
```

and solves two saddle problems with the same fixed generalized force RHS. The
drift is

```text
U_drift = (kT / eps) (U_plus - U_minus).
```

With `return_stresslet=True`, the same two solves also return the drift
stresslet

```text
S5_drift = (kT / eps) (S5_plus - S5_minus),
```

the single-sample estimator of `<S^B> = -kT div(R_SU R_FU^{-1})` described
under "Brownian mean stresslet". The estimator is unbiased for the full
divergence even though only positions are displaced: sphere hydrodynamics is
orientation-independent, so the rotational-coordinate terms of the divergence
vanish and the translational divergence (which `E[dq_pos dq6^T]` picks out) is
the whole answer.

The two displaced solves use fixed absolute GMRES tolerances. This is important:
the difference is divided by a small `eps`, so asymmetric residual errors from
warm-starting can dominate the drift if the solves are not controlled in an
absolute sense. The `1/eps` amplification acts on the stresslet difference
equally, so `S5_drift` is covered by the same discipline (pinned by the
eps-independence and dense-divergence tests).

The displaced solves reuse the neighbor lists built at `q`; therefore
`rfd_epsilon` must be much smaller than the neighbor-list skin. The builder
rejects grossly large `rfd_epsilon` values.

### One timestep

`build_sd_brownian_step` returns `(init_fn, step_fn)`.

Each `step_fn` call:

1. Refreshes live shear state if needed.
2. Precomputes the near-field pair blocks once (`nf_apply.prepare`) and shares
   them across the near-field sampler and the main solve. The two RFD solves
   displace the positions and correctly re-prepare internally — which is what
   lets the drift stresslet estimator see the near-field resistance gradients.
3. Splits the PRNG key into independent far-field, near-field, and RFD keys.
4. Draws far-field slip and near-field Brownian force.
5. Runs one combined deterministic plus Brownian saddle solve, warm-started
   from the previous step's solution when the caller threads `info["x0"]`
   back in (the `simulate.py` wrappers do; matches the persistent solution
   buffer of the original FSD code). Convergence-only: the converged solution
   is unchanged.
6. Computes the RFD drift velocity *and* drift stresslet from two additional
   saddle solves, and adds `S5_drift = (kT/eps)(S5_plus - S5_minus)` to the
   reported stresslet (also exposed as `info["S5_drift"]`).

   **Deviation from the original FSD code.** That implementation keeps only
   the coupling term `-R^nf_SU U_drift` (it feeds the drift-inclusive velocity
   into its near-field `RSU` kernel) and does not sample the stresslet tail of
   the RFD divergence. The shortcut retains one half of the canceling `1/gap`
   pair, leaving a spurious `1/gap` contribution in near-contact pair stress,
   and drops the far-field response divergence entirely — at `phi = 0.45`,
   `Pe = 1` the Brownian stress it corrupts is roughly half of `sigma_xy`
   (Foss & Brady 2000, Fig. 2). This implementation deviates deliberately: the
   full estimator costs only the stresslet assembly of two solves that already
   run. The coupling term must never be re-added on top of `S5_drift` (double
   counting); `tests/sd_brownian_test.py` pins this.
7. Advances positions by an Euler-Maruyama update.
8. Refreshes neighbor lists for the new positions and returns the next state in
   `info["next_state"]`.

For static boxes, the coordinate update includes the ambient translational
add-back if `L_inf` was supplied. For live fractional Lees-Edwards shear, the
changing box basis already carries affine motion, so adding `U_inf` to
fractional coordinates would double-count the affine shear. The helper
`_sd_coordinate_velocity` implements this distinction.

## User-Facing Simulation Wrappers

Most users should call the wrappers in `simulate.py`.

### Free Brownian FSD

```python
from jax_md import energy, simulate, space
import jax.numpy as jnp

box = jnp.eye(3) * 20.0
displacement, shift = space.periodic_general(
    box, fractional_coordinates=True)

energy_fn = energy.soft_sphere_pair(
    displacement, sigma=2.0, epsilon=1.0)

init_fn, apply_fn = simulate.sd(
    (displacement, shift), energy_fn, dt=1e-3, kT=1.0,
    a=1.0, eta=1.0, xi=0.6, P=16, Mgrid=48)

state = init_fn(key, positions_frac)
state = apply_fn(state)
S5 = state.stresslet
```

### Lees-Edwards shear

```python
gamma_dot = 0.1
base_box = jnp.eye(3) * 20.0

displacement, shift, box_of = space.shearing(
    base_box,
    shear_schedule=lambda t: gamma_dot * t,
    fractional_coordinates=True,
    remap=True)

shear_vector_schedule = lambda t: (gamma_dot * t, 0.0 * t, 0.0 * t)

init_fn, apply_fn = simulate.sd_with_shear(
    (displacement, shift, box_of), energy_fn, dt=1e-3, kT=1.0,
    a=1.0, eta=1.0,
    shear_vector_schedule=shear_vector_schedule,
    xi=0.6, P=16, Mgrid=48)

state = init_fn(key, positions_frac)
state = apply_fn(state)
```

`simulate.sd_with_shear` computes the full ambient gradient

```text
L = dH/dt H^{-1}
E_inf = sym(L)
```

and forwards both to the Brownian step. The symmetric part drives the resistance
RHS; the full gradient drives translational and angular add-back.

## Comparison with the Original FSD Code

The JAX implementation is intentionally close to the FSD algorithm, but not a
line-for-line port.

### Same algorithmic structure

Both implementations perform:

1. Build/update lubrication neighbor lists and a saddle preconditioner.
2. Compute far-field grand mobility with PSE.
3. Apply near-field lubrication blocks `RFU`, `RFE`, `RSU`, `RSE`.
4. Assemble a saddle RHS containing imposed strain, applied force, and Brownian
   terms.
5. Solve the saddle system with GMRES.
6. Reconstruct stresslets from far-field and near-field contributions.
7. Use RFD for Brownian drift.

In FSD this sequence is spread across `Stokes.cu`, `Integrator.cu`,
`Saddle.cu`, `Lubrication.cu`, `Precondition.cu`, and the Brownian source files.
In this repo it is concentrated in `sd_nearfield.py`, `sd_saddle.py`, and
`sd_brownian.py`, with the far-field mobility delegated to the `rpy_*` modules.

### Important implementation differences

| Topic | FSD | This repo |
| --- | --- | --- |
| Runtime | C++/CUDA/HOOMD, cuFFT/cuBLAS/cuSPARSE/CUSP. | JAX arrays and transformations, with SciPy only for optional host `ic0`. |
| Near-field table | Hard-coded arrays in `Stokes_ResistanceTable.cc`. | Regenerated from Townsend/Wilson expressions in `resistance_table.npz`; extracted `resistance_table_legacy.npz` is a regression reference. |
| Small-gap regularization | Production kernels clamp near contact to row 232. | Uses full committed table from row 0; clipping policy is explicit. |
| Near-field kernels | Separate `RFU`, `RFE`, `RSU`, `RSE` CUDA kernels. | One unified pair-operator builder over `[U, Omega, E] -> [F, L, S]`. |
| Moment basis | FSD arrays are organized as force/torque/stresslet blocks. | Solver algebra uses orthonormal tensor coordinates for symmetry. |
| Saddle preconditioner | RCM plus incomplete Cholesky is central. | Default is on-device Chebyshev Schur; optional `ic0` mirrors FSD for validation. |
| Brownian near-field preconditioning | Preconditioner code exists, with parts commented/debugged in the checked source. | Uses an on-device diagonal split with explicit shift/projector covariance proof. |
| Live shear wave operator | FSD updates sheared grid vectors. | JAX exact live-box path rebuilds reciprocal mode factors for the current box. |
| State updates | CUDA work arrays and HOOMD neighbor lists. | Shape-preserving JAX neighbor-list updates to avoid recompilation. |

### Sign convention notes

The FSD source contains comments about corrected signs in lubrication and strain
handling. The JAX implementation expresses the near-field operator in the
symmetric `(U, Omega, E) -> (F, L, S)` convention. In this convention:

- self `SU` and `FE` blocks are transposes;
- cross `G` blocks use `XG21 = -XG12`, `YG21 = -YG12`;
- `H` is even, so `YH21 = YH12`;
- the saddle bottom RHS uses `-(F^P + R^nf_FE E_inf)`;
- total stresslet uses `S^ff - R^nf_SU u + R^nf_SE E_inf`.

The tests compare these tensor forms against literal NumPy replicas of the FSD
kernel forms and then pin the end-to-end physical sign with a compressive-strain
pair test.

## Validation Tests

The SD-specific tests are:

```bash
pytest tests/sd_nearfield_test.py
pytest tests/sd_saddle_test.py
pytest tests/sd_brownian_test.py
pytest tests/sd_shear_test.py
```

Some checks are marked slow because they sample Brownian covariance or compare
larger preconditioned systems.

The validation layers include:

- Orthonormal stresslet basis checks.
- Near-field table interpolation and cutoff behavior.
- Per-pair near-field symmetry, PSD behavior near contact, and parity under
  `rhat -> -rhat`.
- Matrix-free near-field apply compared to explicit dense assembly.
- Literal FSD-kernel transcription checks for the lubrication tensors.
- `B`/`B^T` adjointness.
- Degenerate reduction to stresslet-constrained RPY when `R^nf = 0`.
- Single-sphere Einstein stresslet under imposed strain.
- End-to-end near-field sign pins under compressive/extensional strain.
- Preconditioner convergence ordering and scaling.
- Near-field Brownian force covariance.
- Full stochastic velocity covariance against dense deterministic probes.
- RFD drift epsilon-independence and a negative control for mismatched GMRES
  tolerances.
- Live-shear consistency against a static solve at the literal deformed box.
- Ambient add-back and affine-motion double-counting checks.

## Parameters and Tuning

### Ewald parameters

You may provide `xi`, `rcut`, `P`, `Mgrid`, `theta`, and `lattice_extent`
explicitly. If `xi` is omitted in the lower-level builder, provide
`n_particles` and `phi` so `estimate_rpy_params` can choose the split.

The estimator chooses:

- `rcut` from a real-space error target;
- FFT grid size from a wave-space cutoff;
- Spectral Ewald support `P` and Gaussian width from quadrature error bounds;
- larger support when a shear schedule can deform the box.

For production scripts, estimate once and pass the same parameters to
equilibration and shear phases. `examples/hydro/sd_shear_equilibrate.py` follows
that pattern.

### Near-field cutoffs

Defaults:

```text
r_lub = 4a
r_p   = 2.1a
```

`r_lub` is the full lubrication cutoff. `r_p` is only the truncated cutoff for
the optional `ic0` preconditioner.

### Solver tolerances

The deterministic saddle solve uses GMRES. Important knobs:

- `gmres_tol`: relative tolerance for the main solve.
- `gmres_restart`, `gmres_maxiter`: Krylov budget, default `50 x 20 = 1000`
  iterations (the same budget as the original FSD code). Both of jax's GMRES
  loops exit early and the Krylov memory depends only on `restart`, so the
  cap costs nothing on solves that converge.
- `preconditioner`: default `'cheb'`.
- `cheb_degree`, `cheb_power_iters`, `cheb_safety`: default Chebyshev Schur
  controls.

Note that `gmres_tol` does not mean quite what it says. jax's GMRES stops on
the *preconditioned* residual `||M(b - Ax)||` compared against
`max(tol ||b||, atol)` with an unpreconditioned `||b||`, so the accuracy
delivered on the true residual is looser than `tol` by roughly the scale of
`M`; the original FSD code monitors the true residual instead. Every solve
with `return_residual=True` (the default) therefore reports

- `info['rel_residual']`: the true `||b - Ax|| / ||b||`, and
- `info['converged']`: that residual against `max(tol ||b||, atol)`,

and prints a warning when the solve returns unconverged. `info['gmres_info']`
is jax's own flag and is 0 unless the solution is NaN — it is not a
convergence signal. The RFD displaced solves run to a fixed truncated budget
by design and pass `return_residual=False`, so they stay silent.

A non-convergence warning on a contact-rich configuration usually indicates
saddle conditioning rather than an exhausted budget: with no small-gap clamp
the lubrication resistance diverges as `1/gap`, and raising the iteration cap
does not help. `JAX_MD_SD_MIN_GAP` is the lever there.

For Brownian RFD solves, `rfd_atol` is more important than relative tolerance
because the drift is a finite difference of two velocities. In float32 the code
raises very tight tolerances to a reachable floor.

### Neighbor-list capacity

The near-field and real-space operators rely on fixed-shape neighbor-list
buffers in jitted paths. Increase `capacity_multiplier` or `extra_capacity` if
you see buffer overflow.

## Limitations and Assumptions

- The near-field table and tensor code are monodisperse equal-sphere formulas.
- The implementation assumes 3D hydrodynamics.
- Periodic boxes are the intended geometry; shear is Lees-Edwards style through
  `space.shearing`.
- `r_lub` should fit within the minimum-image assumptions of the box.
- Brownian timesteps are jitted and therefore do not support the host `ic0`
  preconditioner.
- The RFD displaced solves reuse neighbor lists built at the undisplaced
  configuration, so `rfd_epsilon` must be small compared with the neighbor-list
  skin.
- In fractional live-shear coordinates, affine motion is carried by the changing
  box basis; adding `U_inf` again would double-count it.
- For detailed covariance or stresslet validation, enable 64-bit JAX before
  importing `jax_md`.

## Mental Model for Extending the Code

If you need to modify or extend this implementation, keep these invariants in
mind:

1. The far-field grand mobility must remain symmetric in flat-11 orthonormal
   coordinates.
2. `B` and `B^T` must remain an exact Euclidean adjoint pair.
3. The near-field table is already a subtracted correction. Do not add another
   far-field subtraction.
4. Near-field pair signs should be reasoned in the unified
   `[U, Omega, E] -> [F, L, S]` convention, not by copying isolated FSD lines
   without translating conventions.
5. The saddle bottom block is `B^T q - R^nf_FU u`.
6. Brownian far-field noise belongs in the top block; near-field Brownian force
   belongs in the bottom block.
7. Any preconditioned Brownian square root must preserve covariance after
   unwinding. Conditioning shifts must not leak into sampled physical rows.
8. Shape-preserving neighbor-list updates are part of the performance contract
   for jitted simulation loops.
