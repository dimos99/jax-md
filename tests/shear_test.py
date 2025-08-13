import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space, partition
import os
import matplotlib.pyplot as plt

# --- Test helpers -------------------------------------------------------------

def make_box(Lx, Ly, Lz=None):
    """Upper-triangular base box from lengths (orthorhombic)."""
    if Lz is None:  # 2D
        return jnp.diag(jnp.array([Lx, Ly], dtype=jnp.float32))
    return jnp.diag(jnp.array([Lx, Ly, Lz], dtype=jnp.float32))

def sheared_xy(H):
    """Return upper-triangular (0,1) entry."""
    return H[0, 1]


ART_DIR = os.path.join(os.path.dirname(__file__), "_artifacts")
PLOT = os.environ.get("SHEAR_TEST_PLOTS", "0") == "1"

def _ensure_artdir():
    if PLOT and not os.path.exists(ART_DIR):
        os.makedirs(ART_DIR, exist_ok=True)


def _plot_unit_cell(H, fname, points=None, title=None):
    if not PLOT: return
    _ensure_artdir()
    # Map unit square corners through H (2D visualization; ignore z).
    a1 = jnp.array([H[0,0], 0.0])
    a2 = jnp.array([H[0,1], H[1,1]])
    O  = jnp.array([0.0, 0.0])
    P  = jnp.stack([O, a1, a1+a2, a2, O])
    fig, ax = plt.subplots()
    ax.plot(P[:,0], P[:,1])
    if points is not None:
        ax.scatter(points[:,0], points[:,1], s=8)
    if title:
        ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(ART_DIR, fname), dpi=160)
    plt.close(fig)
    
# --- Energy / force helpers (for validation) ---------------------------------

def _make_pair_energy_and_forces(disp_fn, rc: float, k: float = 1.0, *, gamma=None, t=None):
    """Return (E_fn, F_fn) that compute energy and forces for a *fixed* pair list.

    Each returned function expects (R, pairs) where pairs is int array (M,2) with i<j.
    Potential:  U(d) = 0.5*k*max(rc - d, 0)^2  (C^1 at r = rc)
    """
    import jax
    import jax.numpy as jnp

    def _disp(a, b):
        if gamma is not None:
            return disp_fn(a, b, gamma=gamma)
        if t is not None:
            return disp_fn(a, b, t=t)
        return disp_fn(a, b)

    def _energy(R, pairs):
        if pairs.size == 0:
            return jnp.array(0.0, dtype=R.dtype)
        Ra = R[pairs[:, 0]]
        Rb = R[pairs[:, 1]]
        dR = jax.vmap(_disp)(Ra, Rb)
        d = jnp.linalg.norm(dR, axis=1)
        x = jnp.clip(rc - d, a_min=0.0)
        return 0.5 * k * jnp.sum(x * x)

    def _forces(R, pairs):
        return -jax.grad(_energy)(R, pairs)

    return _energy, _forces

# --- Neighbor-list helpers ---------------------------------------------------

def _pairs_from_nl(nl):
    """Extract undirected (i,j) neighbor pairs from a JAX MD NeighborList.
    Returns a NumPy array of shape (M, 2) with i<j.
    Handles Dense (idx shape [N, max_k], sentinel = N) and
    Sparse/OrderedSparse (idx shape [2, M]).
    """
    import numpy as _np

    # Prefer format if present; otherwise infer from idx shape.
    fmt = getattr(nl, 'format', None)

    if hasattr(nl, 'idx'):
        idx = _np.asarray(nl.idx)
        # Sparse/OrderedSparse: idx has shape (2, M) -> [receivers, senders]
        if (idx.ndim == 2 and idx.shape[0] == 2) or (
            fmt is getattr(partition, 'Sparse', None) or fmt is getattr(partition, 'OrderedSparse', None)
        ):
            receivers = idx[0]
            senders = idx[1]
            i = _np.asarray(senders)
            j = _np.asarray(receivers)
        else:
            # Dense: idx has shape (N, max_k) with sentinel value N for empty
            N = idx.shape[0]
            i_list = []
            j_list = []
            for a in range(N):
                for b in idx[a]:
                    if b >= N:   # skip sentinel entries
                        continue
                    i_list.append(a)
                    j_list.append(int(b))
            i = _np.asarray(i_list)
            j = _np.asarray(j_list)
    elif hasattr(nl, 'senders') and hasattr(nl, 'receivers'):
        i = _np.asarray(nl.senders)
        j = _np.asarray(nl.receivers)
    elif hasattr(nl, 'pairs') and hasattr(nl.pairs, 'senders') and hasattr(nl.pairs, 'receivers'):
        i = _np.asarray(nl.pairs.senders)
        j = _np.asarray(nl.pairs.receivers)
    else:
        raise AttributeError("Unrecognized neighbor list structure; can't extract pairs")

    # Make undirected unique pairs with i<j
    mask = i != j
    i = i[mask]; j = j[mask]
    i2 = _np.minimum(i, j)
    j2 = _np.maximum(i, j)
    pairs = _np.unique(_np.stack([i2, j2], axis=1), axis=0)
    return pairs



def _bruteforce_pairs(disp_fn, R, *, gamma=None, t=None, r_cutoff=1.0):
    """Return undirected (i,j) with i<j whose distance < r_cutoff, using disp_fn.
    R is an array-like of shape (N, dim); disp_fn has signature disp(Ra,Rb, **kw).
    Pass either gamma or t (or neither if disp_fn ignores both).
    """
    import numpy as _np
    R = _np.asarray(R)
    N = R.shape[0]

    def _d(a, b):
        if gamma is not None:
            return _np.linalg.norm(_np.asarray(disp_fn(a, b, gamma=gamma)))
        if t is not None:
            return _np.linalg.norm(_np.asarray(disp_fn(a, b, t=t)))
        return _np.linalg.norm(_np.asarray(disp_fn(a, b)))

    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if _d(R[i], R[j]) < r_cutoff:
                pairs.append((i, j))
    return _np.asarray(pairs, dtype=int)

# --- Core correctness tests ---------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("fractional", [True, False])
def test_zero_shear_matches_periodic_general(dim, fractional):
    # If gamma = 0, this should equal periodic_general with the same base box.
    L = [3.0, 2.0] if dim == 2 else [3.0, 2.0, 4.0]
    B = make_box(*L)

    disp_pg, shift_pg = space.periodic_general(B, fractional_coordinates=fractional)
    disp_s, shift_s, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=fractional)

    key = jax.random.PRNGKey(0)
    Ra = jax.random.uniform(key, (5, dim))
    Rb = jax.random.uniform(key, (5, dim))
    # Vectorize displacement comparisons.
    dR1 = jax.vmap(disp_pg)(Ra, Rb)
    dR2 = jax.vmap(lambda a, b: disp_s(a, b, t=1.23))(Ra, Rb)   # any t; gamma = 0 anyway
    np.testing.assert_allclose(dR1, dR2, atol=1e-6)

    dR = jax.random.normal(key, (5, dim))
    # Vectorize shift comparisons.
    R1 = jax.vmap(shift_pg)(Ra, dR)
    R2 = jax.vmap(lambda r, dr: shift_s(r, dr, t=0.5))(Ra, dR)
    np.testing.assert_allclose(R1, R2, atol=1e-6)

@pytest.mark.parametrize("dim", [2, 3])
def test_box_of_time_dependence_and_remap(dim):
    L = [3.0, 2.0] if dim == 2 else [3.0, 2.0, 4.0]
    B = make_box(*L)
    gamma_dot = 0.1
    disp, shift, box_of = space.shearing(B, shear_rate=gamma_dot, fractional_coordinates=True, remap=False)

    t1, t2 = 1.0, 3.5
    H1 = box_of(t=t1)
    H2 = box_of(t=t2)
    Ly = L[1]
    # xy entry should differ by (t2 - t1) * gamma_dot * Ly
    np.testing.assert_allclose(sheared_xy(H2) - sheared_xy(H1), (t2 - t1) * gamma_dot * Ly, atol=1e-7)

    # With remap=True, gamma is wrapped into [-0.5, 0.5)
    dispR, shiftR, box_ofR = space.shearing(B, shear_rate=1.0, fractional_coordinates=True, remap=True)
    for t in [0.0, 0.49, 0.51, 1.49, -0.51]:
        H = box_ofR(t=t)
        # gamma_wrapped ∈ [-0.5, 0.5) ⇒ xy - base_xy ∈ [-0.5*Ly, 0.5*Ly)
        delta_xy = sheared_xy(H) - sheared_xy(B)
        assert -0.5 * Ly - 1e-7 <= float(delta_xy) < 0.5 * Ly + 1e-7

@pytest.mark.parametrize("dim", [2, 3])
def test_gamma_override_wins_over_rate(dim):
    L = [3.0, 2.0] if dim == 2 else [3.0, 2.0, 4.0]
    B = make_box(*L)
    disp, shift, box_of = space.shearing(B, shear_rate=123.0, fractional_coordinates=True, remap=False)

    H1 = box_of(t=10.0, gamma=0.2)     # should ignore shear_rate * t
    Ly = L[1]
    np.testing.assert_allclose(sheared_xy(H1), sheared_xy(B) + 0.2 * Ly, atol=1e-7)

def test_keep_base_xy_behavior():
    # Start from a base box with existing tilt xy0.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    base = make_box(Lx, Ly, Lz).at[0,1].set(0.25)  # base_xy = 0.25
    gamma = 0.1
    # keep_base_xy=True keeps base tilt and adds gamma*Ly
    _, _, box_keep = space.shearing(base, shear_rate=0.0, keep_base_xy=True)
    Hk = box_keep(gamma=gamma)
    np.testing.assert_allclose(sheared_xy(Hk), 0.25 + gamma * Ly, atol=1e-7)
    # keep_base_xy=False overwrites tilt with gamma*Ly
    _, _, box_ow = space.shearing(base, shear_rate=0.0, keep_base_xy=False)
    Ho = box_ow(gamma=gamma)
    np.testing.assert_allclose(sheared_xy(Ho), gamma * Ly, atol=1e-7)

@pytest.mark.parametrize("fractional", [True, False])
def test_fractional_vs_real_paths_agree(fractional):
    # Compare displacement from fractional path vs real path for same physical config.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.3
    dispF, shiftF, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    dispR, shiftR, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=False)

    key = jax.random.PRNGKey(1)
    Ra_frac = jax.random.uniform(key, (10, 3))
    Rb_frac = jax.random.uniform(key, (10, 3))
    H = box_of(gamma=gamma)

    # Real-space positions corresponding to those fractional ones
    Ra_real = space.transform(H, Ra_frac)
    Rb_real = space.transform(H, Rb_frac)

    dRF = dispF(Ra_frac, Rb_frac, gamma=gamma)
    dRR = dispR(Ra_real, Rb_real, gamma=gamma)
    np.testing.assert_allclose(dRF, dRR, atol=1e-6)

def test_minimum_image_and_antisymmetry_fractional():
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.4
    disp, shift, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    key = jax.random.PRNGKey(2)
    Ra = jax.random.uniform(key, (20, 3))
    # Put Rb as Ra + random vector in [-0.6,0.6) (fractional)
    d = jax.random.uniform(key, (20, 3), minval=-0.6, maxval=0.6)
    Rb = Ra + d

    # Displacement should be H * (wrap(Ra-Rb))
    H = box_of(gamma=gamma)
    wrap = lambda x: x - jnp.round(x)
    dRf = wrap(Ra - Rb)
    dR_expected = space.transform(H, dRf)
    dR = disp(Ra, Rb, gamma=gamma)
    np.testing.assert_allclose(dR, dR_expected, atol=1e-6)
    # Antisymmetry
    np.testing.assert_allclose(dR, -disp(Rb, Ra, gamma=gamma), atol=1e-6)

def test_periodicity_across_y_face_includes_shear_offset():
    # Crossing y by +1 in fractional should shift x by gamma*Ly (in the metric),
    # so displacement should still be minimum-image equivalent.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.25
    disp, shift, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    H = box_of(gamma=gamma)

    Ra = jnp.array([[0.1, 0.99, 0.2]], dtype=jnp.float32)  # close to +y face
    Rb = jnp.array([[0.15, 0.01, 0.2]], dtype=jnp.float32) # just across the boundary
    # Direct formula: wrap in fractional, then map by H
    dRf = (Ra - Rb) - jnp.round(Ra - Rb)
    dR_expected = space.transform(H, dRf)
    dR = disp(Ra, Rb, gamma=gamma)
    np.testing.assert_allclose(dR, dR_expected, atol=1e-6)

@pytest.mark.parametrize("fractional", [True, False])
def test_shift_roundtrip_small_steps(fractional):
    # For small |dR|, shift then measure displacement back gives (approximately) dR.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.3
    disp, shift, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=fractional)
    key = jax.random.PRNGKey(3)
    R = jax.random.uniform(key, (30, 3))
    dR = 1e-3 * jax.random.normal(key, (30, 3))  # tiny real step

    if fractional:
        R2 = shift(R, dR, gamma=gamma)
        # displacement from R2 back to R ≈ dR
        dR_meas = disp(R2, R, gamma=gamma)
    else:
        H = box_of(gamma=gamma)
        R_real = space.transform(H, R)
        R2 = shift(R_real, dR, gamma=gamma)
        dR_meas = disp(R2, R_real, gamma=gamma)

    np.testing.assert_allclose(dR_meas, dR, atol=1e-6, rtol=1e-4)

def test_jit_and_vmap_ok():
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    disp, shift, box_of = space.shearing(B, shear_rate=0.2, fractional_coordinates=True)

    Ra = jnp.array([[0.1, 0.2, 0.3],
                    [0.3, 0.0, 0.9]], dtype=jnp.float32)
    Rb = jnp.array([[0.4, 0.8, 0.7],
                    [0.31,0.99,0.1]], dtype=jnp.float32)

    f = jax.jit(lambda Ra, Rb, t: disp(Ra, Rb, t=t))
    g = jax.jit(lambda R, dR, t: shift(R, dR, t=t))

    dR = f(Ra, Rb, 1.5)
    R2 = g(Ra, jnp.ones_like(Ra) * 1e-3, 1.5)
    assert dR.shape == Ra.shape
    assert R2.shape == Ra.shape

def test_invalid_box_errors():
    # Scalar box should raise (ambiguous dimension).
    with pytest.raises(Exception):
        space.shearing(jnp.array(3.0), shear_rate=0.1)
    # 1D box should raise for space.shearing (needs at least 2D).
    with pytest.raises(Exception):
        space.shearing(jnp.array([3.0]), shear_rate=0.1)
        
        
def test_lattice_invariance_fractional():
    # disp should be invariant under integer fractional shifts.
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.33
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    key = jax.random.PRNGKey(5)
    Ra = jax.random.uniform(key, (20, 3))
    Rb = jax.random.uniform(key, (20, 3))
    H = box_of(gamma=gamma)

    d0 = disp(Ra, Rb, gamma=gamma)
    # random integer lattice shifts in {-1,0,1}
    S = jax.random.randint(key, (20, 3), -1, 2).astype(Ra.dtype)
    d1 = disp(Ra + S, Rb + S, gamma=gamma)
    np.testing.assert_allclose(d0, d1, atol=1e-6)

    # Visual: sheared cell at this gamma with a few points
    pts = space.transform(H, (Ra % 1.0))
    _plot_unit_cell(H[:2,:2], "shear_cell_gamma033.png", points=pts[:, :2], title="Sheared cell (gamma=0.33)")

def test_shift_displacement_consistency_finite_steps():
    # For finite random dR, shifting and then measuring the displacement back should equal dR
    # up to periodic wrap (i.e., the minimum image).
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.41
    disp, shift, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    key = jax.random.PRNGKey(6)
    R = jax.random.uniform(key, (100, 3))
    dR = jax.random.normal(key, (100, 3)) * 0.2  # not tiny

    R2 = shift(R, dR, gamma=gamma)
    dR_meas = disp(R2, R, gamma=gamma)

    # Fold dR into the same minimum-image convention as disp uses.
    # In fractional mode, disp = H * wrap(Δf); but here dR is real, so only compare norms.
    np.testing.assert_allclose(jnp.linalg.norm(dR_meas, axis=1),
                               jnp.linalg.norm(dR, axis=1),
                               rtol=1e-3, atol=1e-3)

def test_remap_boundary_continuity():
    # When remap wraps gamma at ±0.5, results should be continuous.
    B = make_box(3.0, 2.0, 4.0)
    dispR, _, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=True, remap=True)
    key = jax.random.PRNGKey(7)
    Ra = jax.random.uniform(key, (50, 3))
    Rb = jax.random.uniform(key, (50, 3))
    eps = 1e-6
    d1 = dispR(Ra, Rb, gamma=0.5 - eps)
    d2 = dispR(Ra, Rb, gamma=-0.5 + eps)
    np.testing.assert_allclose(d1, d2, atol=1e-5, rtol=1e-5)

def test_autodiff_gradients_match_geometry():
    # ∂/∂Ra ||d||^2 = 2 d, ∂/∂Rb ||d||^2 = -2 d
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.2
    disp, _, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)

    def f(Ra, Rb):
        d = disp(Ra, Rb, gamma=gamma)
        return jnp.sum(d * d)

    Ra = jnp.array([0.2, 0.7, 0.1])
    Rb = jnp.array([0.8, 0.05, 0.4])

    gRa = jax.grad(lambda a: f(a, Rb))(Ra)
    gRb = jax.grad(lambda b: f(Ra, b))(Rb)
    d = disp(Ra, Rb, gamma=gamma)

    np.testing.assert_allclose(gRa,  2.0 * d, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gRb, -2.0 * d, atol=1e-6, rtol=1e-6)

def test_malformed_boxes_rejected():
    # Ly must be positive; non-UT shapes should be rejected (if your code enforces it).
    # Adjust these asserts depending on how strict your constructor is.
    with pytest.raises(Exception):
        space.shearing(jnp.diag(jnp.array([3.0, 0.0, 4.0], dtype=jnp.float32)), shear_rate=0.1)
        
def test_matches_periodic_general_at_fixed_gamma():
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.37
    disp_s, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    H = box_of(gamma=gamma)

    disp_pg, _ = space.periodic_general(H, fractional_coordinates=True)

    key = jax.random.PRNGKey(123)
    Ra = jax.random.uniform(key, (32, 3))
    Rb = jax.random.uniform(key, (32, 3))

    d1 = disp_s(Ra, Rb, gamma=gamma)
    d2 = jax.vmap(disp_pg)(Ra, Rb)
    np.testing.assert_allclose(d1, d2, atol=1e-6)

from jax_md import partition

def test_zero_shear_integration_consistency():
    # Under gamma=0 the shearing space reduces to periodic_general; a few NVE-like
    # shifts should agree between the two spaces.
    B = make_box(3.0, 2.0, 4.0)
    disp_s, shift_s, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    disp_pg, shift_pg = space.periodic_general(B, fractional_coordinates=True)

    key = jax.random.PRNGKey(9)
    R = jax.random.uniform(key, (64, 3))
    dR = 1e-2 * jax.random.normal(key, (64, 3))

    # 10 steps
    for _ in range(10):
        R = shift_s(R, dR)  # shearing space (gamma=0)
    R_s = R

    key2 = jax.random.PRNGKey(9)
    R = jax.random.uniform(key2, (64, 3))
    dR = 1e-2 * jax.random.normal(key2, (64, 3))
    for _ in range(10):
        R = shift_pg(R, dR) # periodic_general

    np.testing.assert_allclose(R_s, R, atol=1e-6)

def test_visual_face_crossing_displacements_png():
    if not PLOT: 
        return
    # 2D example with clearly visible shear-tilted displacements
    Lx, Ly = 3.0, 2.0
    B = make_box(Lx, Ly)
    gamma = 0.35  # => xy = gamma * Ly = 0.7
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    H = box_of(gamma=gamma)
    xy = float(H[0,1])

    # Choose pairs that cross the y-boundary with the SAME fractional x.
    # This makes the minimum-image displacement align with ±a2.
    # After wrapping, Δfy = -0.20 and -0.30 respectively (clearly visible).
    Ra = jnp.stack([
        jnp.array([0.25, 0.98]),
        jnp.array([0.75, 0.97])
    ])
    Rb = jnp.stack([
        jnp.array([0.25, 0.18]),  # 0.98 - 0.18 = 0.80 -> wrap -> -0.20
        jnp.array([0.75, 0.27])   # 0.97 - 0.27 = 0.70 -> wrap -> -0.30
    ])

    dR = disp(Ra, Rb, gamma=gamma)

    # Real-space anchors
    Ra_real = space.transform(H, Ra)

    # Draw the sheared unit cell, the lattice vector a2, and the displacement arrows
    _ensure_artdir()
    fig, ax = plt.subplots()

    # Parallelogram of the cell in real space (2D)
    a1 = jnp.array([H[0,0], 0.0])
    a2 = jnp.array([H[0,1], H[1,1]])
    O  = jnp.array([0.0, 0.0])
    P  = jnp.stack([O, a1, a1+a2, a2, O])
    ax.plot(P[:,0], P[:,1], linewidth=1.5)

    # Lattice vector a2 from the origin for reference
    ax.arrow(0.0, 0.0, float(a2[0]), float(a2[1]),
             head_width=0.06, length_includes_head=True, linewidth=1.2)

    # Displacement arrows
    ax.quiver(Ra_real[:,0], Ra_real[:,1], dR[:,0], dR[:,1],
              angles='xy', scale_units='xy', scale=1, width=0.004)

    # Start points
    ax.scatter(Ra_real[:,0], Ra_real[:,1], s=12)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Face-crossing displacements (gamma={gamma}, xy={xy:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Pad the view slightly around the cell
    xmin, xmax = -0.2, float(max(a1[0], a1[0]+a2[0])) + 0.2
    ymin, ymax = -0.2, float(max(a2[1], 0.0)) + 0.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(os.path.join(ART_DIR, "face_crossing_displacements.png"), dpi=160)
    plt.close(fig)
    
def test_plot_tilt_over_time_png():
    if not PLOT: return
    B = make_box(3.0, 2.0)
    _, _, box_of = space.shearing(B, shear_rate=0.4, fractional_coordinates=True, remap=True)
    ts = jnp.linspace(0, 5, 200)
    xs = jnp.array([float(sheared_xy(box_of(t=float(t)))) for t in ts])
    _ensure_artdir()
    plt.figure(); plt.plot(ts, xs); plt.xlabel('t'); plt.ylabel('xy'); plt.tight_layout()
    plt.savefig(os.path.join(ART_DIR, "tilt_over_time.png"), dpi=160); plt.close()


# --- Neighbor-list box-change trigger tests (require box_at_build logic) ---

@pytest.mark.skipif(
    not hasattr(partition, "NeighborList") or
    not hasattr(getattr(partition, "NeighborList"), "__annotations__") or
    "box_at_build" not in partition.NeighborList.__annotations__,
    reason="requires neighbor-list box-change trigger patch"
)
def test_neighbor_list_rebuild_on_box_change():
    # Fractional coordinates + shearing; ensure rebuild when box tilt changes enough.
    B = make_box(3.0, 2.0, 4.0)
    Ly = 2.0
    rc, skin = 1.0, 0.4
    dr_threshold = skin / 2  # common choice in JAX MD
    gamma_dot = 0.02
    dt = 1.0

    disp, _, box_of = space.shearing(B, shear_rate=gamma_dot, fractional_coordinates=True)
    neighbor_fn = partition.neighbor_list(
        disp,
        box=box_of(t=0.0),
        r_cutoff=rc,
        dr_threshold=dr_threshold,
        fractional_coordinates=True,
        format=partition.OrderedSparse,
    )

    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (128, 3))

    nbrs = neighbor_fn.allocate(R, box=box_of(t=0.0))
    ref_pos0 = nbrs.reference_position
    ref_box0 = nbrs.box_at_build

    # Each step changes xy by Δxy = Ly * gamma_dot * dt.
    dxy = Ly * gamma_dot * dt
    # Rebuild triggers when 0.5 * |ΔH|_F > dr_threshold. Here |ΔH|_F = |Δxy|.
    # So need |Δxy_total| > 2 * dr_threshold.
    steps_until_trigger = int(np.floor((2 * dr_threshold) / dxy))

    # Advance up to (but not exceeding) the threshold: no rebuild expected.
    for k in range(steps_until_trigger):
        t = (k + 1) * dt
        nbrs2 = neighbor_fn.update(R, nbrs, box=box_of(t=t))
        # Should not rebuild yet
        np.testing.assert_allclose(nbrs2.reference_position, nbrs.reference_position)
        np.testing.assert_allclose(nbrs2.box_at_build, nbrs.box_at_build)
        nbrs = nbrs2

    # Next step should exceed the bound and rebuild.
    t = (steps_until_trigger + 1) * dt
    nbrs2 = neighbor_fn.update(R, nbrs, box=box_of(t=t))
    # Reference positions can remain identical if R is unchanged; the box must update.
    assert not np.allclose(nbrs2.box_at_build, ref_box0)
    np.testing.assert_allclose(nbrs2.box_at_build, box_of(t=t))


@pytest.mark.skipif(
    not hasattr(partition, "NeighborList") or
    not hasattr(getattr(partition, "NeighborList"), "__annotations__") or
    "box_at_build" not in partition.NeighborList.__annotations__,
    reason="requires neighbor-list box-change trigger patch"
)
def test_neighbor_list_no_rebuild_for_small_box_change():
    # Small box change below threshold should not rebuild.
    B = make_box(3.0, 2.0, 4.0)
    Ly = 2.0
    rc, skin = 1.0, 0.4
    dr_threshold = skin / 2
    gamma_dot = 0.02

    disp, _, box_of = space.shearing(B, shear_rate=gamma_dot, fractional_coordinates=True)
    neighbor_fn = partition.neighbor_list(
        disp,
        box=box_of(t=0.0),
        r_cutoff=rc,
        dr_threshold=dr_threshold,
        fractional_coordinates=True,
        format=partition.OrderedSparse,
    )

    key = jax.random.PRNGKey(1)
    R = jax.random.uniform(key, (64, 3))
    nbrs = neighbor_fn.allocate(R, box=box_of(t=0.0))

    # Pick a tiny Δt so that |Δxy| = Ly * gamma_dot * Δt ≤ dr_threshold.
    dt_small = (dr_threshold * 0.5) / (Ly * gamma_dot)
    nbrs2 = neighbor_fn.update(R, nbrs, box=box_of(t=float(dt_small)))

    # No rebuild expected: references remain identical.
    np.testing.assert_allclose(nbrs2.reference_position, nbrs.reference_position)
    np.testing.assert_allclose(nbrs2.box_at_build, nbrs.box_at_build)


@pytest.mark.skipif(
    not hasattr(partition, "NeighborList") or
    not hasattr(getattr(partition, "NeighborList"), "__annotations__") or
    "box_at_build" not in partition.NeighborList.__annotations__,
    reason="requires neighbor-list box-change trigger patch"
)
def test_neighbor_list_motion_trigger_still_works():
    # Rebuild on particle motion beyond dr_threshold/2 in the metric.
    B = make_box(3.0, 2.0, 4.0)
    rc, skin = 1.0, 0.4
    dr_threshold = skin / 2

    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    H = box_of(t=0.0)

    neighbor_fn = partition.neighbor_list(
        disp,
        box=H,
        r_cutoff=rc,
        dr_threshold=dr_threshold,
        fractional_coordinates=True,
        format=partition.OrderedSparse,
    )

    key = jax.random.PRNGKey(2)
    R = jax.random.uniform(key, (32, 3))
    nbrs = neighbor_fn.allocate(R, box=H)

    # Move a single particle just beyond the threshold in real space along x.
    Lx = float(H[0,0])
    delta_real = (dr_threshold / 2) * 1.05  # slightly over the (dr_threshold/2)
    delta_frac_x = delta_real / Lx
    R2 = R.at[0, 0].add(delta_frac_x)

    nbrs2 = neighbor_fn.update(R2, nbrs, box=H)
    assert not np.allclose(nbrs2.reference_position, nbrs.reference_position)
    
def test_remap_exact_half_integers():
    B = make_box(3.0, 2.0, 4.0)
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True, remap=True)
    Ra = jnp.array([0.1,0.2,0.3]); Rb = jnp.array([0.9,0.8,0.7])
    d1 = disp(Ra, Rb, gamma=0.5)   # tie case
    d2 = disp(Ra, Rb, gamma=-0.5)  # expected equivalent under wrapping
    np.testing.assert_allclose(d1, d2, atol=1e-6)

    # Box consistency at ties
    H1 = box_of(gamma=0.5); H2 = box_of(gamma=-0.5)
    np.testing.assert_allclose(H1, H2, atol=1e-7)
    
def test_neighbor_list_no_rebuild_on_integer_wrap():
    B = make_box(3.0, 2.0, 4.0)
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    rc, skin = 1.0, 0.4
    neighbor_fn = partition.neighbor_list(
        disp, box=box_of(t=0.0), r_cutoff=rc, dr_threshold=skin/2,
        fractional_coordinates=True, format=partition.OrderedSparse)

    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (64, 3))
    nbrs = neighbor_fn.allocate(R, box=box_of(t=0.0))

    # Add an integer lattice shift in fractional coords (wrap)
    S = jnp.array([1.0, -1.0, 0.0])[None, :]
    R2 = (R + S) % 1.0
    nbrs2 = neighbor_fn.update(R2, nbrs, box=box_of(t=0.0))

    # No rebuild: references unchanged (min-image drift is zero)
    np.testing.assert_allclose(nbrs2.reference_position, nbrs.reference_position)
    np.testing.assert_allclose(nbrs2.box_at_build, nbrs.box_at_build)
    
    
def test_disp_grad_wrt_gamma_matches_analytic():
    B = make_box(3.0, 2.0, 4.0)
    Ly = float(B[1,1])
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)

    Ra = jnp.array([0.2, 0.9, 0.1])
    Rb = jnp.array([0.8, 0.1, 0.4])
    gamma = 0.3

    def g(gam):
        return disp(Ra, Rb, gamma=gam)

    J = jax.jacrev(g)(gamma)  # shape (3,)
    # analytic: dH/dγ has only (0,1)=Ly; d(Δf)=wrap(Ra-Rb) independent of γ
    dft = (Ra - Rb) - jnp.round(Ra - Rb)
    analytic = jnp.array([Ly * dft[1], 0.0, 0.0])
    np.testing.assert_allclose(J, analytic, atol=1e-6)
    
def test_neighbor_list_rejects_box_without_fractional():
    from jax_md import partition
    B = make_box(3.0, 2.0, 4.0)
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=False)
    with pytest.raises(Exception):
        partition.neighbor_list(disp, box=B, r_cutoff=1.0,
                                fractional_coordinates=False)
        
def test_cell_size_too_small_flag_on_more_skewed_box():
    from jax_md import partition
    B = make_box(6.0, 2.0, 4.0)   # ample sizes
    rc, skin = 1.0, 0.4
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    neighbor_fn = partition.neighbor_list(disp, box=box_of(gamma=0.0),
                                          r_cutoff=rc, dr_threshold=skin/2,
                                          fractional_coordinates=True)
    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (128, 3))
    nbrs = neighbor_fn.allocate(R, box=box_of(gamma=0.0))
    # Jump to larger |gamma| to reduce nx
    nbrs2 = neighbor_fn.update(R, nbrs, box=box_of(gamma=0.49))
    flag = bool(np.asarray(nbrs2.cell_size_too_small != 0).item())
    assert flag or np.allclose(nbrs2.box_at_build, box_of(gamma=0.49))


# --- Metric-specific tests for shearing --------------------------------------

@pytest.mark.parametrize("fractional", [True, False])
def test_metric_matches_displacement_norm_shearing(fractional):
    """Metric constructed from shearing displacement matches displacement norm."""
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.27
    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=fractional)
    # Partially apply gamma so the metric has signature (Ra, Rb)
    disp_g = lambda a, b: disp(a, b, gamma=gamma)
    metric_fn = space.metric(disp_g)

    key = jax.random.PRNGKey(1234)
    if fractional:
        Ra = jax.random.uniform(key, (32, 3))
        Rb = jax.random.uniform(key, (32, 3))
        disp_vals = jax.vmap(disp_g)(Ra, Rb)
        metric_vals = jax.vmap(metric_fn)(Ra, Rb)
    else:
        # Generate fractional, map to real for use with real-coordinate displacement.
        Ra_f = jax.random.uniform(key, (32, 3))
        Rb_f = jax.random.uniform(key, (32, 3))
        H = box_of(gamma=gamma)
        Ra = space.transform(H, Ra_f)
        Rb = space.transform(H, Rb_f)
        disp_vals = jax.vmap(disp_g)(Ra, Rb)
        metric_vals = jax.vmap(metric_fn)(Ra, Rb)

    np.testing.assert_allclose(metric_vals, jnp.linalg.norm(disp_vals, axis=-1), atol=1e-6)


@pytest.mark.parametrize("fractional", [True, False])
def test_zero_shear_metric_matches_periodic_general_metric(fractional):
    """At zero shear, shearing metric equals periodic_general metric."""
    B = make_box(3.0, 2.0, 4.0)
    disp_s, _, _ = space.shearing(B, shear_rate=0.0, fractional_coordinates=fractional)
    disp_pg, _ = space.periodic_general(B, fractional_coordinates=fractional)

    t_test = 1.234  # arbitrary; shear_rate=0 so gamma=0 regardless of t
    disp_s_t = lambda a, b: disp_s(a, b, t=t_test)
    metric_s = space.metric(disp_s_t)
    metric_pg = space.metric(disp_pg)

    key = jax.random.PRNGKey(5678)
    if fractional:
        Ra = jax.random.uniform(key, (16, 3))
        Rb = jax.random.uniform(key, (16, 3))
    else:
        Ra_f = jax.random.uniform(key, (16, 3))
        Rb_f = jax.random.uniform(key, (16, 3))
        Ra = space.transform(B, Ra_f)
        Rb = space.transform(B, Rb_f)

    dr_s = jax.vmap(metric_s)(Ra, Rb)
    dr_pg = jax.vmap(metric_pg)(Ra, Rb)
    np.testing.assert_allclose(dr_s, dr_pg, atol=1e-6)


# Additional neighbor-list tests

@pytest.mark.parametrize("fractional", [True, False])
def test_neighbor_list_matches_bruteforce_static(fractional):
    """Neighbor-list pairs match brute force at fixed shear (static)."""
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.33
    rc = 0.9

    disp, _, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=fractional)
    H = box_of(gamma=gamma)

    key = jax.random.PRNGKey(42)
    if fractional:
        R = jax.random.uniform(key, (96, 3))
        neighbor_fn = partition.neighbor_list(
            disp,
            box=H,
            r_cutoff=rc,
            fractional_coordinates=True,
            format=getattr(partition, "OrderedSparse", None) or partition.Dense,
        )
        nl = neighbor_fn.allocate(R, box=H)
        # Force a build in case allocate doesn't populate pairs immediately
        nl = neighbor_fn.update(R, nl, box=H)
        nl_pairs = _pairs_from_nl(nl)
        bf_pairs = _bruteforce_pairs(lambda a,b,**kw: disp(a,b,**kw), np.asarray(R), gamma=gamma, r_cutoff=rc)
    else:
        # Real coordinates: sample in fractional then map to real to get a uniform distribution in the cell
        Rf = jax.random.uniform(key, (96, 3))
        R = space.transform(H, Rf)
        neighbor_fn = partition.neighbor_list(
            disp,
            box=jnp.array([H[0,0], H[1,1], H[2,2]]),  # side lengths only for real-coords cell list
            r_cutoff=rc,
            fractional_coordinates=False,
            format=getattr(partition, "OrderedSparse", None) or partition.Dense,
        )
        # Thread the shear via gamma to the displacement (avoid passing a matrix box through neighbor_list in real coords).
        nl = neighbor_fn.allocate(R, gamma=gamma)
        nl = neighbor_fn.update(R, nl, gamma=gamma)
        nl_pairs = _pairs_from_nl(nl)
        bf_pairs = _bruteforce_pairs(lambda a,b,**kw: disp(a,b,**kw), np.asarray(R), gamma=gamma, r_cutoff=rc)

    # Compare as sets of tuples
    nl_set = set(map(tuple, np.asarray(nl_pairs)))
    bf_set = set(map(tuple, np.asarray(bf_pairs)))
    assert nl_set == bf_set


def test_neighbor_list_matches_bruteforce_after_shear():
    """After changing shear (box tilt), neighbor-list pairs match brute force."""
    B = make_box(3.0, 2.0, 4.0)
    rc = 0.9
    gamma0 = 0.0
    gamma1 = 0.41

    # Fractional coordinates here so positions stay in [0,1) and only the metric/box changes
    disp, shift, box_of = space.shearing(B, shear_rate=0.0, fractional_coordinates=True)
    H0 = box_of(gamma=gamma0)
    H1 = box_of(gamma=gamma1)

    key = jax.random.PRNGKey(99)
    R = jax.random.uniform(key, (128, 3))

    neighbor_fn = partition.neighbor_list(
        disp,
        box=H0,
        r_cutoff=rc,
        fractional_coordinates=True,
        format=getattr(partition, "OrderedSparse", None) or partition.Dense,
    )

    nl = neighbor_fn.allocate(R, box=H0)
    # Update only the box (shear) to H1, mimicking dynamics under changing tilt
    nl1 = neighbor_fn.update(R, nl, box=H1)

    # Extract pairs and compare to brute force at the new shear
    nl_pairs = _pairs_from_nl(nl1)
    bf_pairs = _bruteforce_pairs(lambda a,b,**kw: disp(a,b,**kw), np.asarray(R), gamma=gamma1, r_cutoff=rc)

    nl_set = set(map(tuple, np.asarray(nl_pairs)))
    bf_set = set(map(tuple, np.asarray(bf_pairs)))
    assert nl_set == bf_set
    


# --- Remapping invariance: energy and forces ---

@pytest.mark.parametrize("fractional", [True, False])
@pytest.mark.parametrize("use_remap", [True, False])
def test_remap_invariance_energy_and_forces(fractional, use_remap):
    """Remapping gamma (γ → γ + n with n ∈ ℤ) must not change physics.

    We check that pairwise energies and forces are identical for γ, γ+1, γ-2.
    This holds both when the shearing constructor uses remap=True (internally
    wraps γ into [-0.5, 0.5)) and when remap=False (no wrapping). In either
    case, adding an integer to γ corresponds to adding a lattice vector to the
    box, which leaves minimum-image displacements invariant.
    """
    # Base box and displacement
    B = make_box(3.0, 2.0, 4.0)
    disp, _, box_of = space.shearing(B, shear_rate=0.0,
                                     fractional_coordinates=fractional,
                                     remap=use_remap)

    # Make a small random configuration
    key = jax.random.PRNGKey(42)
    N = 16
    dim = 3

    if fractional:
        R = jax.random.uniform(key, (N, dim))  # live in unit cube
    else:
        # Create fractional first, then map to real using some reference γ0
        Rf = jax.random.uniform(key, (N, dim))
        gamma0 = 0.37
        H0 = box_of(gamma=gamma0)
        R = space.transform(H0, Rf)

    # Use *all* pairs so results are independent of any cutoff-induced changes
    pairs = np.array([(i, j) for i in range(N) for j in range(i+1, N)], dtype=int)

    # Big cutoff ensures every pair contributes (energy form is 0.5*k*(rc-d)^2)
    rc = 100.0

    def energy_and_forces_at_gamma(gam):
        E_fn, F_fn = _make_pair_energy_and_forces(disp, rc=rc, k=1.0, gamma=gam)
        E = float(np.asarray(E_fn(R, pairs)))
        F = np.asarray(F_fn(R, pairs))
        return E, F

    gamma = 0.37
    gammas = [gamma, gamma + 1.0, gamma - 2.0]

    E0, F0 = energy_and_forces_at_gamma(gammas[0])
    for g in gammas[1:]:
        E, F = energy_and_forces_at_gamma(g)
        np.testing.assert_allclose(E, E0, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(F, F0, rtol=1e-6, atol=1e-6)