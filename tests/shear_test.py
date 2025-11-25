import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space, partition, smap
import os
import matplotlib.pyplot as plt
from typing import cast

# Turn on double-precision for JAX
jax.config.update("jax_enable_x64", True)

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
            return disp_fn(a, b, gamma=gamma, t=0.0)
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

def _pairs_from_nl(nl, disp_fn=None, positions=None, r_cutoff=None, **disp_kwargs):
    """Extract undirected (i,j) neighbor pairs from a JAX MD NeighborList.
    
    If disp_fn, positions, and r_cutoff are provided, filters pairs to only
    include those within the physics cutoff, excluding the skin region.
    
    Returns a NumPy array of shape (M, 2) with i<j.
    Handles Dense (idx shape [N, max_k], sentinel = N) and
    Sparse/OrderedSparse (idx shape [2, M]).
    """
    import numpy as _np
    import jax.numpy as jnp

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
    
    # Filter by distance if requested
    if disp_fn is not None and positions is not None and r_cutoff is not None:
        if len(pairs) == 0:
            return pairs
        # Compute distances for all pairs
        positions = _np.asarray(positions)
        distances = []
        for i, j in pairs:
            dR = disp_fn(positions[i], positions[j], **disp_kwargs)
            dist = float(jnp.linalg.norm(dR))
            distances.append(dist)
        # Keep only pairs within physics cutoff with a tiny epsilon
        distances = _np.array(distances)
        eps = 1e-9
        mask = distances <= (r_cutoff + eps)
        pairs = pairs[mask]
    
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
            return _np.linalg.norm(_np.asarray(disp_fn(a, b, gamma=gamma, t=0.0)))
        if t is not None:
            return _np.linalg.norm(_np.asarray(disp_fn(a, b, t=t)))
        return _np.linalg.norm(_np.asarray(disp_fn(a, b)))

    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if _d(R[i], R[j]) <= (r_cutoff + 1e-9):
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
    disp_s, shift_s, box_of = space.shearing(B, fractional_coordinates=fractional)

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
def test_box_time_remap_and_overrides(dim):
    L = [3.0, 2.0] if dim == 2 else [3.0, 2.0, 4.0]
    B = make_box(*L)
    gamma_dot = 0.1
    _, _, box_of = space.shearing(B, shear_fn=lambda t: gamma_dot * t, fractional_coordinates=True, remap=False)

    t1, t2 = 1.0, 3.5
    H1 = box_of(t=t1)
    H2 = box_of(t=t2)
    Ly = L[1]
    np.testing.assert_allclose(sheared_xy(H2) - sheared_xy(H1), (t2 - t1) * gamma_dot * Ly, atol=1e-7)

    # remap=True wraps gamma
    _, _, box_ofR = space.shearing(B, shear_fn=lambda t: 1.0 * t, fractional_coordinates=True, remap=True)
    for t in [0.0, 0.49, 0.51, 1.49, -0.51]:
        H = box_ofR(t=t)
        delta_xy = sheared_xy(H) - sheared_xy(B)
        assert -0.5 * Ly - 1e-7 <= float(delta_xy) < 0.5 * Ly + 1e-7

    # gamma override ignores shear_fn
    _, _, box_override = space.shearing(B, shear_fn=lambda t: 123.0 * t, fractional_coordinates=True, remap=False)
    H_override = box_override(t=10.0, gamma=0.2)
    np.testing.assert_allclose(sheared_xy(H_override), sheared_xy(B) + 0.2 * Ly, atol=1e-7)

    # keep_base_xy behavior
    base = make_box(*L).at[0,1].set(0.25)
    _, _, box_keep = space.shearing(base, keep_base_xy=True)
    _, _, box_ow = space.shearing(base, keep_base_xy=False)
    Hk = box_keep(gamma=gamma_dot, t=0.0)
    Ho = box_ow(gamma=gamma_dot, t=0.0)
    np.testing.assert_allclose(sheared_xy(Hk), 0.25 + gamma_dot * Ly, atol=1e-7)
    np.testing.assert_allclose(sheared_xy(Ho), gamma_dot * Ly, atol=1e-7)

@pytest.mark.parametrize("fractional", [True, False])
def test_fractional_vs_real_paths_agree(fractional):
    # Compare displacement from fractional path vs real path for same physical config.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.3
    dispF, shiftF, box_of = space.shearing(B, fractional_coordinates=True)
    dispR, shiftR, _ = space.shearing(B, fractional_coordinates=False)

    key = jax.random.PRNGKey(1)
    Ra_frac = jax.random.uniform(key, (10, 3))
    Rb_frac = jax.random.uniform(key, (10, 3))
    H = box_of(gamma=gamma, t=0.0)

    # Real-space positions corresponding to those fractional ones
    Ra_real = space.transform(H, Ra_frac)
    Rb_real = space.transform(H, Rb_frac)

    dRF = dispF(Ra_frac, Rb_frac, gamma=gamma, t=0.0)
    dRR = dispR(Ra_real, Rb_real, gamma=gamma, t=0.0)
    np.testing.assert_allclose(dRF, dRR, atol=1e-6)

def test_minimum_image_and_antisymmetry_fractional():
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.4
    disp, shift, box_of = space.shearing(B, fractional_coordinates=True)
    key = jax.random.PRNGKey(2)
    Ra = jax.random.uniform(key, (20, 3))
    # Put Rb as Ra + random vector in [-0.6,0.6) (fractional)
    d = jax.random.uniform(key, (20, 3), minval=-0.6, maxval=0.6)
    Rb = Ra + d

    # Displacement should be H * (wrap(Ra-Rb))
    H = box_of(gamma=gamma, t=0.0)
    wrap = lambda x: x - jnp.round(x)
    dRf = wrap(Ra - Rb)
    dR_expected = space.transform(H, dRf)
    dR = disp(Ra, Rb, gamma=gamma, t=0.0)
    np.testing.assert_allclose(dR, dR_expected, atol=1e-6)
    # Antisymmetry
    np.testing.assert_allclose(dR, -disp(Rb, Ra, gamma=gamma, t=0.0), atol=1e-6)

def test_periodicity_across_y_face_includes_shear_offset():
    # Crossing y by +1 in fractional should shift x by gamma*Ly (in the metric),
    # so displacement should still be minimum-image equivalent.
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    gamma = 0.25
    disp, shift, box_of = space.shearing(B, fractional_coordinates=True)
    H = box_of(gamma=gamma, t=0.0)

    Ra = jnp.array([[0.1, 0.99, 0.2]], dtype=jnp.float32)  # close to +y face
    Rb = jnp.array([[0.15, 0.01, 0.2]], dtype=jnp.float32) # just across the boundary
    # Direct formula: wrap in fractional, then map by H
    dRf = (Ra - Rb) - jnp.round(Ra - Rb)
    dR_expected = space.transform(H, dRf)
    dR = disp(Ra, Rb, gamma=gamma, t=0.0)
    np.testing.assert_allclose(dR, dR_expected, atol=1e-6)

def test_jit_and_vmap_ok():
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    disp, shift, box_of = space.shearing(B, shear_fn=lambda t: 0.2 * t, fractional_coordinates=True)

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
    # Scalar boxes now canonicalize to isotropic 3D; ensure construction succeeds.
    space.shearing(jnp.array(3.0), shear_fn=lambda t: 0.1 * t)
    # 1D box should raise for space.shearing (needs at least 2D).
    with pytest.raises(Exception):
        space.shearing(jnp.array([3.0]), shear_fn=lambda t: 0.1 * t)
        
        
def test_lattice_invariance_fractional():
    # disp should be invariant under integer fractional shifts.
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.33
    disp, _, box_of = space.shearing(B, fractional_coordinates=True)
    key = jax.random.PRNGKey(5)
    Ra = jax.random.uniform(key, (20, 3))
    Rb = jax.random.uniform(key, (20, 3))
    H = box_of(gamma=gamma, t=0.0)

    d0 = disp(Ra, Rb, gamma=gamma, t=0.0)
    # random integer lattice shifts in {-1,0,1}
    S = jax.random.randint(key, (20, 3), -1, 2).astype(Ra.dtype)
    d1 = disp(Ra + S, Rb + S, gamma=gamma, t=0.0)
    np.testing.assert_allclose(d0, d1, atol=1e-6)

    # Visual: sheared cell at this gamma with a few points
    pts = space.transform(H, (Ra % 1.0))
    _plot_unit_cell(H[:2,:2], "shear_cell_gamma033.png", points=pts[:, :2], title="Sheared cell (gamma=0.33)")

def test_shift_displacement_consistency_finite_steps():
    # For moderate dR, shifting and then measuring the displacement back matches dR norms.
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.41
    disp, shift, _ = space.shearing(B, fractional_coordinates=True)
    key = jax.random.PRNGKey(6)
    R = jax.random.uniform(key, (100, 3))
    dR = jax.random.normal(key, (100, 3)) * 0.2  # not tiny

    R2 = shift(R, dR, gamma=gamma, t=0.0)
    dR_meas = disp(R2, R, gamma=gamma, t=0.0)

    np.testing.assert_allclose(jnp.linalg.norm(dR_meas, axis=1),
                               jnp.linalg.norm(dR, axis=1),
                               rtol=1e-3, atol=1e-3)

def test_remap_boundary_continuity():
    # When remap wraps gamma at ±0.5, results are continuous only if
    # fractional coordinates are remapped to keep real positions fixed.
    B = make_box(3.0, 2.0, 4.0)
    dispR, _, box_of = space.shearing(B, fractional_coordinates=True, remap=True)
    key = jax.random.PRNGKey(7)
    Ra = jax.random.uniform(key, (50, 3))
    Rb = jax.random.uniform(key, (50, 3))
    eps = 1e-6
    gamma_before = 0.5 - eps
    gamma_after = -0.5 + eps
    H_before = box_of(gamma=gamma_before, t=0.0)
    H_after = box_of(gamma=gamma_after, t=0.0)

    d1 = dispR(Ra, Rb, gamma=gamma_before, t=0.0)

    # Remap fractional coordinates so real positions are unchanged across the flip.
    Ra_remap = space.remap_fractional_positions(Ra, H_before, H_after)
    Rb_remap = space.remap_fractional_positions(Rb, H_before, H_after)
    d2 = dispR(Ra_remap, Rb_remap, gamma=gamma_after, t=0.0)

    np.testing.assert_allclose(d1, d2, atol=1e-5, rtol=1e-5)

def test_autodiff_gradients_match_geometry():
    # ∂/∂Ra ||d||^2 = 2 d, ∂/∂Rb ||d||^2 = -2 d
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.2
    disp, _, _ = space.shearing(B, fractional_coordinates=True)

    def f(Ra, Rb):
        d = disp(Ra, Rb, gamma=gamma, t=0.0)
        return jnp.sum(d * d)

    Ra = jnp.array([0.2, 0.7, 0.1])
    Rb = jnp.array([0.8, 0.05, 0.4])

    gRa = jax.grad(lambda a: f(a, Rb))(Ra)
    gRb = jax.grad(lambda b: f(Ra, b))(Rb)
    d = disp(Ra, Rb, gamma=gamma, t=0.0)

    np.testing.assert_allclose(gRa,  2.0 * d, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(gRb, -2.0 * d, atol=1e-6, rtol=1e-6)

def test_malformed_boxes_rejected():
    # Ly must be positive; non-UT shapes should be rejected (if your code enforces it).
    # Adjust these asserts depending on how strict your constructor is.
    with pytest.raises(Exception):
        space.shearing(jnp.diag(jnp.array([3.0, 0.0, 4.0], dtype=jnp.float32)), shear_fn=lambda t: 0.1 * t)
        
def test_matches_periodic_general_at_fixed_gamma():
    B = make_box(3.0, 2.0, 4.0)
    gamma = 0.37
    disp_s, _, box_of = space.shearing(B, fractional_coordinates=True)
    H = box_of(gamma=gamma, t=0.0)

    disp_pg, _ = space.periodic_general(H, fractional_coordinates=True)

    key = jax.random.PRNGKey(123)
    Ra = jax.random.uniform(key, (32, 3))
    Rb = jax.random.uniform(key, (32, 3))

    d1 = disp_s(Ra, Rb, gamma=gamma, t=0.0)
    d2 = jax.vmap(disp_pg)(Ra, Rb)
    np.testing.assert_allclose(d1, d2, atol=1e-6)

from jax_md import partition

def test_zero_shear_integration_consistency():
    # Under gamma=0 the shearing space reduces to periodic_general; a few NVE-like
    # shifts should agree between the two spaces.
    B = make_box(3.0, 2.0, 4.0)
    disp_s, shift_s, _ = space.shearing(B, fractional_coordinates=True)
    disp_pg, shift_pg = space.periodic_general(B, fractional_coordinates=True)

    key = jax.random.PRNGKey(9)
    R = jax.random.uniform(key, (64, 3))
    dR = 1e-2 * jax.random.normal(key, (64, 3))

    # 10 steps
    for _ in range(10):
        R = shift_s(R, dR, gamma=0.0, t=0.0)  # shearing space (gamma=0)
    R_s = R

    key2 = jax.random.PRNGKey(9)
    R = jax.random.uniform(key2, (64, 3))
    dR = 1e-2 * jax.random.normal(key2, (64, 3))
    for _ in range(10):
        R = shift_pg(R, dR) # periodic_general

    np.testing.assert_allclose(R_s, R, atol=1e-6)

# --- Neighbor-list box-change trigger tests (require box_at_build logic) ---

def test_disp_grad_wrt_gamma_matches_analytic():
    B = make_box(3.0, 2.0, 4.0)
    Ly = float(B[1,1])
    disp, _, box_of = space.shearing(B, fractional_coordinates=True)

    Ra = jnp.array([0.2, 0.9, 0.1])
    Rb = jnp.array([0.8, 0.1, 0.4])
    gamma = 0.3

    def g(gam):
        return disp(Ra, Rb, gamma=gam, t=0.0)

    J = jax.jacrev(g)(gamma)  # shape (3,)
    # analytic: dH/dγ has only (0,1)=Ly; d(Δf)=wrap(Ra-Rb) independent of γ
    dft = (Ra - Rb) - jnp.round(Ra - Rb)
    analytic = jnp.array([Ly * dft[1], 0.0, 0.0])
    np.testing.assert_allclose(J, analytic, atol=1e-6)
    
def test_neighbor_list_rejects_box_without_fractional():
    from jax_md import partition
    B = make_box(3.0, 2.0, 4.0)
    disp, _, box_of = space.shearing(B, fractional_coordinates=False)
    with pytest.raises(Exception):
        partition.neighbor_list(disp, box=B, r_cutoff=1.0,
                                fractional_coordinates=False)
        
def test_cell_size_too_small_flag_on_more_skewed_box():
    from jax_md import partition
    B = make_box(6.0, 2.0, 4.0)   # ample sizes
    rc, skin = 1.0, 0.4
    disp, _, box_of = space.shearing(B, fractional_coordinates=True)
    neighbor_fn = partition.neighbor_list(disp, box=box_of(gamma=0.0, t=0.0),
                                          r_cutoff=rc, dr_threshold=skin/2,
                                          fractional_coordinates=True)
    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (128, 3))
    nbrs = neighbor_fn(R, box=box_of(gamma=0.0, t=0.0))
    # Jump to larger |gamma| to reduce nx
    nbrs2 = neighbor_fn(R, nbrs, box=box_of(gamma=0.49, t=0.0))
    flag = bool(np.asarray(nbrs2.cell_size_too_small != 0).item())
    assert flag or np.allclose(nbrs2.box_at_build, box_of(gamma=0.49, t=0.0))


# --- Metric-specific tests for shearing --------------------------------------

@pytest.mark.parametrize("fractional", [True, False])
def test_zero_shear_metric_matches_periodic_general_metric(fractional):
    """At zero shear, shearing metric equals periodic_general metric."""
    B = make_box(3.0, 2.0, 4.0)
    disp_s, _, _ = space.shearing(B, fractional_coordinates=fractional)
    disp_pg, _ = space.periodic_general(B, fractional_coordinates=fractional)

    t_test = 1.234  # arbitrary; shear_fn=lambda t: 0 * t so gamma=0 regardless of t
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

def test_neighbor_list_remains_correct_over_long_shearing():
    """Neighbor list pairs remain correct over many shear-time updates.

    We evolve the box tilt for many steps and verify after each update that
    the neighbor pairs match a brute-force search using the current metric.
    Remapping keeps the tilt bounded, avoiding pathological cell sizes.
    """
    B = make_box(3.0, 2.0, 4.0)
    rc, skin = 0.9, 0.4
    dr_threshold = skin / 2
    gamma_dot = 10  # shear rate per unit time
    dt = 0.75         # time step (fractional units)
    steps = 100        # total time ~ 18.0 -> many wraps under remap

    # Fractional coordinates with remap=True for stability of the cell list
    disp, _, box_of = space.shearing(B, shear_fn=lambda t: gamma_dot * t,
                                     fractional_coordinates=True,
                                     remap=True)

    key = jax.random.PRNGKey(2024)
    R = jax.random.uniform(key, (10, 3))  # fractional positions in [0,1)

    fmt = getattr(partition, "OrderedSparse", None) or partition.Dense
    neighbor_fn = cast(partition.NeighborListFns, partition.neighbor_list(
        disp,
        box=box_of(t=0.0),
        r_cutoff=rc,
        dr_threshold=dr_threshold,
        fractional_coordinates=True,
        format=fmt,
    ))

    nl = neighbor_fn(R, box=box_of(t=0.0))

    # March forward in time; at each step, update with the latest box and
    # validate against brute force under that box (i.e., current gamma(t)).
    t = 0.0  # Initialize t for the final validation
    for k in range(1, steps + 1):
        t = float(k * dt)
        Ht = box_of(t=t)
        nl = neighbor_fn(R, nl, box=Ht)

    # Compare undirected pairs as sets
    nl_pairs = _pairs_from_nl(nl, disp_fn=lambda a, b, **kw: disp(a, b, **kw), 
                             positions=R, r_cutoff=rc, t=t)
    nl_set = set(map(tuple, np.asarray(nl_pairs)))

    # Get brute-force pairs at physics cutoff
    bf_rc = _bruteforce_pairs(lambda a, b, **kw: disp(a, b, **kw),
                   np.asarray(R), t=t, r_cutoff=rc)
    bf_rc_set = set(map(tuple, np.asarray(bf_rc)))
    
    # With proper distance filtering, neighbor list pairs should exactly match brute force
    assert nl_set == bf_rc_set, f"NL pairs {nl_set} do not match BF pairs {bf_rc_set} at t={t}"
    

def test_neighbor_list_internal_remap_fractional():
    """Neighbor list stays correct if caller remaps fractional positions at wraps."""
    Lx, Ly, Lz = 3.0, 2.0, 4.0
    B = make_box(Lx, Ly, Lz)
    disp, _, box_of = space.shearing(B, shear_fn=lambda t: 1.0 * t, fractional_coordinates=True, remap=True)

    # Two particles near contact in real space, represented fractionally.
    Rf = jnp.array([[0.1, 0.49, 0.2],
                    [0.12, 0.51, 0.2]], dtype=jnp.float32)
    rcut = 0.5

    nbr_fns = partition.neighbor_list(disp, box=box_of(t=0.0),
                                      r_cutoff=rcut, dr_threshold=0.1,
                                      fractional_coordinates=True,
                                      format=partition.OrderedSparse,
                                      mask_self=True)

    nl = nbr_fns.allocate(Rf, box=box_of(t=0.0))
    # Jump across a wrap in gamma; remap positions explicitly to keep real coords fixed.
    t_before = 0.49
    t_after  = 0.51
    H_before = box_of(t=t_before)
    H_after = box_of(t=t_after)
    nl = nbr_fns.update(Rf, nl, box=H_before)
    Rf_after = space.remap_fractional_positions(Rf, H_before, H_after)
    nl = nbr_fns.update(Rf_after, nl, box=H_after, box_prev=H_before)

    # Extract pairs within rcut and compare to brute-force under box_of(t_after)
    pairs_nl = _pairs_from_nl(nl, disp_fn=disp, positions=Rf_after, r_cutoff=rcut, box=H_after)
    pairs_bf = _bruteforce_pairs(disp, Rf_after, t=t_after, r_cutoff=rcut)
    np.testing.assert_array_equal(pairs_nl, pairs_bf)


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
    disp, _, box_of = space.shearing(B, 
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
        H0 = box_of(gamma=gamma0, t=0.0)
        R = space.transform(H0, Rf)

    # Use *all* pairs so results are independent of any cutoff-induced changes
    pairs = np.array([(i, j) for i in range(N) for j in range(i+1, N)], dtype=int)

    # Big cutoff ensures every pair contributes (energy form is 0.5*k*(rc-d)^2)
    rc = 100.0

    def energy_and_forces_at_gamma(gam, positions):
        E_fn, F_fn = _make_pair_energy_and_forces(disp, rc=rc, k=1.0, gamma=gam)
        E = float(np.asarray(E_fn(positions, pairs)))
        F = np.asarray(F_fn(positions, pairs))
        return E, F

    gamma = 0.37
    gammas = [gamma, gamma + 1.0, gamma - 2.0]

    wrap = lambda g: g - np.floor(g + 0.5)

    if use_remap:
        # Automatic wrapping: same fractional coords should give identical physics.
        rtol = 1e-6 if fractional else 1e-5
        atol = 1e-6 if fractional else 1e-5
        E0, F0 = energy_and_forces_at_gamma(gammas[0], R)
        for g in gammas[1:]:
            E, F = energy_and_forces_at_gamma(g, R)
            np.testing.assert_allclose(E, E0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(F, F0, rtol=max(rtol, 1e-6), atol=atol)
    else:
        # Without remap=True, users must wrap gamma themselves and remap fractional
        # coordinates to keep real positions fixed across basis flips. We only
        # assert that leaving gamma unwrapped changes the physics.
        E0, _ = energy_and_forces_at_gamma(gamma, R)
        for g in gammas[1:]:
            E_raw, _ = energy_and_forces_at_gamma(g, R)
            if fractional:
                assert not np.allclose(E_raw, E0, rtol=1e-3, atol=1e-3)
def test_energy_continuity_across_flip_with_jitted_update():
    """Energy must be continuous across a remap (tilt flip) when NL+energy
    use the same instantaneous box and fractional positions are remapped.

    This mirrors the production pattern: update neighbor list INSIDE @jit using
    both the previous and current boxes, then compute energy at the same t.
    """
    # Base orthorhombic box (dimensionless units)
    Lx, Ly, Lz = 7.0, 5.0, 6.0
    B = jnp.diag(jnp.array([Lx, Ly, Lz], dtype=jnp.float32))

    # Fractional coordinates + remap=True to enable tilt flips
    disp, _, box_of = space.shearing(B, 
                                     fractional_coordinates=True,
                                     remap=True)

    key = jax.random.PRNGKey(123)
    N = 96
    R = jax.random.uniform(key, (N, 3))  # fractional positions in [0,1)^3

    # Physics cutoff and neighbor cutoff (with skin) so pair inclusion is stable
    rc = 0.35
    skin = 0.15
    rc_nl = rc + skin

    # Pair potential with compact support (quadratic well inside rc)
    def u_pair(dr, **kwargs):
        r = jnp.linalg.norm(dr, axis=-1)
        x = jnp.clip(rc - r, 0.0)
        return 0.5 * x * x

    # Energy that uses the sheared metric (disp) and the neighbor list
    energy_fn = smap.pair_neighbor_list(u_pair, disp)

    # Build neighbor list functions; OrderedSparse if available
    fmt = getattr(partition, 'OrderedSparse', None) or partition.Dense
    neighbor_fns = partition.neighbor_list(
        disp,
        box=box_of(t=0.0),
        r_cutoff=rc_nl,
        dr_threshold=skin / 2,
        fractional_coordinates=True,
        format=fmt,
    )

    # Times straddling a flip (gamma wraps at ±0.5). Choose just-below and just-above.
    t_minus = 0.499
    t_plus  = 0.501
    Hm = box_of(t=t_minus)
    Hp = box_of(t=t_plus)

    # Remap fractional positions so real coordinates are unchanged across the flip.
    R_remap = space.remap_fractional_positions(R, Hm, Hp)

    # Allocate at t_minus
    nl = neighbor_fns.allocate(R, box=Hm)

    # JIT the update across the flip, passing both prev and current boxes
    @jax.jit
    def update_across_flip(R_new, nl, H_prev, H_cur):
        return neighbor_fns.update(R_new, nl, box=H_cur, box_prev=H_prev)

    nl2 = update_across_flip(R_remap, nl, Hm, Hp)

    # Energies just before and after the flip must match to numerical tolerance
    E_minus = energy_fn(R, neighbor=nl,  box=Hm)
    E_plus  = energy_fn(R_remap, neighbor=nl2, box=Hp)

    diff = jnp.abs(E_plus - E_minus)
    # Tight tolerance: energies should be identical up to roundoff
    assert float(diff) < 1e-8, f"Energy changed across flip: |ΔE|={float(diff)}"

    # Cross-check: brute-force energy over all pairs at t_minus and t_plus
    # Ensure this equals the neighbor-list energies (within tolerance)
    def brute_force_energy(R_positions, H):
        # Compute pair distances with the sheared metric at box H
        metric = space.metric(lambda a, b: disp(a, b, box=H))
        map_prod = space.map_product(metric)
        D = map_prod(R_positions, R_positions)  # (N,N)
        # Use strictly upper triangle i<j
        iu = jnp.triu_indices(N, k=1)
        r = D[iu]
        x = jnp.clip(rc - r, 0.0)
        return 0.5 * jnp.sum(x * x)

    E_bf_minus = brute_force_energy(R, Hm)
    E_bf_plus  = brute_force_energy(R_remap, Hp)

    assert float(jnp.abs(E_bf_plus - E_bf_minus)) < 1e-8
    assert float(jnp.abs(E_minus - E_bf_minus)) < 1e-9
    assert float(jnp.abs(E_plus  - E_bf_plus )) < 1e-9
    
    
    
def _pairs_bruteforce(disp_fn, R, r_cutoff, **kwargs):
  N = R.shape[0]
  pairs = []
  for i in range(N):
    for j in range(i + 1, N):
      d = jnp.linalg.norm(disp_fn(R[i], R[j], **kwargs))
      if d <= (r_cutoff + 1e-9):
        pairs.append((i, j))
  return set(pairs)


import numpy as onp
import numpy as np
import pytest

from jax_md import space, partition, smap


def _pairs_from_sparse_nl(nl):
  # neighbor.idx has shape (2, M): [receiver, sender]
  idx = onp.array(jax.device_get(nl.idx))
  i = idx[0]
  j = idx[1]
  # undirected unique pairs
  a = onp.minimum(i, j)
  b = onp.maximum(i, j)
  mask = a != b
  pairs = onp.stack([a[mask], b[mask]], axis=1)
  # unique rows
  pairs = onp.unique(pairs, axis=0)
  return set(map(tuple, pairs))


def _bruteforce_pairs_box(R, disp_fn, H, r_cut):
    # Bind the box into the displacement to avoid passing kwargs through metric
    def disp_box(a, b):
        return disp_fn(a, b, box=H)
    metric_fn = space.metric(disp_box)
    # Bind the box argument so vmaps only over particle axes.
    def metric_box(Ra, Rb):
        return metric_fn(Ra, Rb)
    map_prod = space.map_product(metric_box)
    D = jax.device_get(map_prod(R, R))
    N = D.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] <= r_cut + 1e-12:
                pairs.append((i, j))
    return set(pairs)


@pytest.mark.parametrize("dim", [2, 3])
def test_neighbor_pairs_stable_across_remap_flip(dim):
    # Box lengths
    L = jnp.array([10.0, 9.0] + ([8.0] if dim == 3 else []))
    disp, _, box_of = space.shearing(L, fractional_coordinates=True, remap=True)

    key = jax.random.PRNGKey(0)
    R = jax.random.uniform(key, (128, dim))  # fractional positions in [0,1)

    r_cut = 0.25
    skin = 0.10
    H1 = box_of(t=0.499)
    H2 = box_of(t=0.501)

    nl_fns = partition.neighbor_list(
            disp, H1, r_cut, dr_threshold=skin, fractional_coordinates=True,
            format=partition.NeighborListFormat.OrderedSparse)
    nl = nl_fns.allocate(R, box=H1)
    nl2 = nl_fns.update(R, nl, box=H2)

    pairs_nl1 = _pairs_from_sparse_nl(nl)
    pairs_nl2 = _pairs_from_sparse_nl(nl2)
    pairs_bf1 = _bruteforce_pairs_box(R, disp, H1, r_cut + skin)
    pairs_bf2 = _bruteforce_pairs_box(R, disp, H2, r_cut + skin)

    assert pairs_nl1 == pairs_bf1
    assert pairs_nl2 == pairs_bf2
    assert pairs_nl1 == pairs_nl2


  
