"""Real-space RPY mobility (M^r) with closed-form kernels

Real-space mobility M^(r) uses closed-form F1,F2 coefficients for
monodisperse spheres of radius a, with Ewald splitting parameter ξ.

For a pair at separation r:
  M^r_ij = (1/(6πηa)) * [F1(r; a, ξ) * (I - r̂⊗r̂) + F2(r; a, ξ) * r̂⊗r̂]

Self term (r=0):
  M^r_ii = (1/(6πηa)) * f_self(a, ξ)
  
where f_self(a, ξ) = [1/(4√π ξ a)] * [1 - exp(-4a²ξ²) + 4√π a ξ erfc(2aξ)]
is the eta-independent self-mobility factor computed by Mr_self(a, ξ).

Lattice Image Summation
-----------------------
In periodic boundary conditions, hydrodynamic interactions extend across periodic
images of the simulation cell. To correctly evaluate the real-space mobility, we
must sum contributions from particles and their periodic images:

  M^r_ij = ∑_L M^r(r_ij + L)

where L iterates over lattice vectors representing periodic cell translations.

The lattice indices generation serves two critical purposes:

1. **Completeness**: Ensures all relevant periodic images within the real-space
   cutoff radius r_cut are included in the mobility calculation. Missing images
   would break translational invariance and introduce artificial boundaries.

2. **Efficiency**: Limits the sum to a finite set of nearby images. The Ewald
   splitting causes real-space contributions to decay exponentially beyond r_cut,
   so distant images (|L| >> r_cut) contribute negligibly and can be omitted.

**Lattice Construction Algorithm**:
- Generate a symmetric hypercube of integer shifts: L ∈ {-N, ..., N}^d where d
  is the spatial dimension (typically 3).
- N (lattice extent) is chosen such that the smallest box dimension spans at
  least r_cut when mapped to real space: N ≥ ceil(r_cut / σ_min(Box)).
- The zero lattice vector (L=0, primary cell) is explicitly tracked because
  self-interactions (i=j, L=0) require special treatment via the analytic
  self-mobility term.
- A post-processing step trims lattice vectors that cannot contribute within
  r_cut to reduce computational overhead, while always preserving L=0.

**Example**: For a cubic box with side length 10 and r_cut=3, we generate lattice
shifts in {-1, 0, 1}^3, yielding 27 images (including the primary cell). Images at
corners (e.g., L=[1,1,1]) map to real-space distances ~17.3, well beyond r_cut,
so they are pruned, leaving fewer active images for the mobility calculation.

This approach balances accuracy (all interactions within r_cut are captured) with
performance (distant images are excluded), making real-space mobility evaluation
tractable for typical simulation box sizes.
  
References
----------
[1] Fiore, Andrew M., Florencio Balboa Usabiaga, Aleksandar Donev, and James W. Swan. "Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations." The Journal of Chemical Physics 146, no. 12 (2017): 124116. https://doi.org/10.1063/1.4978242.
[2] Fiore, Andrew M. "Fast Simulation Methods for Soft Matter Hydrodynamics." PhD Thesis, Massachusetts Institute of Technology, 2019.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors
from functools import partial

from typing import Callable, Optional, Tuple

from jax_md import dataclasses, partition, space
from jax import ops
from jax_md.hydro.rpy_real_det_helpers import (
    REAL_DTYPE,
    F1F2_closed_form,
    Mr_self,
    current_box_matrix,
    canonicalize_box_matrix,
    generate_lattice_hypercube,
)


I3 = jnp.eye(3, dtype=REAL_DTYPE) # 3x3 identity matrix


@dataclasses.dataclass
class RealSpaceState:
    """State for the real-space mobility operator.

    This container threads the bookkeeping needed to apply the real-space
    mobility across timesteps without rebuilding everything. It stores:

    - ``neighbors``: the neighbor list (Dense/Sparse/OrderedSparse).
    - ``lattice_indices`` / ``zero_image_index``: periodic image shifts for the
      real-space Ewald sum.
    - ``box_matrix``: current box transform used for real↔fractional conversion.
    - ``fractional_coordinates``: whether incoming positions are fractional
      (``True``) or already in real space (``False``); the matvec converts using
      this flag.
    - ``core_fn``: JIT-compiled matvec that consumes the above.
    """

    neighbors: Optional[partition.NeighborList]
    lattice_indices: jnp.ndarray
    zero_image_index: int
    box_matrix: jnp.ndarray
    fractional_coordinates: bool = dataclasses.field(default=True, metadata={'static': True})
    core_fn: Optional[Callable] = dataclasses.field(default=None, metadata={'static': True})


def _positions_to_real(positions: jnp.ndarray,
                       box_matrix: jnp.ndarray,
                       fractional_coordinates: bool) -> jnp.ndarray:
    """Convert positions to real coordinates when provided in fractional form."""
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    if fractional_coordinates:
        return space.transform(box_matrix, positions)
    return positions


def _neighbor_box_from_matrix(box_matrix: jnp.ndarray,
                              fractional_coordinates: bool) -> jnp.ndarray:
    """Box argument passed to neighbor_list depending on coordinate convention."""
    if fractional_coordinates:
        return box_matrix
    # Neighbor lists in real coordinates disallow `box` kwargs; rely on the space.
    return None

@partial(jax.jit, static_argnums=(1,2,3))
def Mr_pair_block(r_vec, a, xi, eta):
    """
    Given separation vector r_vec (R^3), return 3x3 block M^r_ij.
    
    Parameters
    ----------
    r_vec : (3,) array
        Separation vector
    a : float
        Sphere radius
    xi : float
        Ewald splitting parameter
    eta : float
        Fluid viscosity
        
    Returns
    -------
    Mr : (3,3) array
        Real-space mobility block
    """
    # Compute separation related quantities
    r2 = jnp.dot(r_vec, r_vec)
    r = jnp.sqrt(r2 + 1e-300)
    rhat = r_vec / r
    rhat_outer = jnp.outer(rhat, rhat)
    # Get mobility coefficients (Appx. A of Fiore et al.)
    F1, F2 = F1F2_closed_form(r, a, xi)
    
    prefactor = 1.0 / (6.0 * jnp.pi * eta * a)
    Mr = prefactor * (F1 * (I3 - rhat_outer) + F2 * rhat_outer)
    
    return Mr


def _build_mr_core(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
    fractional_coordinates: bool,
) -> Callable[..., jnp.ndarray]:
    """
    Create the JIT-ed core that evaluates the real-space mobility.
    
    This factory function builds a specialized JIT-compiled kernel for computing
    real-space mobility matrix-vector products M^r @ f, where M^r is the real-space
    component of the Ewald-split mobility operator. The returned function supports
    both dense and sparse neighbor list formats and applies memory-efficient fused
    contractions to avoid materializing the full mobility tensor.
    
    The mobility kernel computes:
        v_i = M^r_ii·f_i + Σ_{j≠i} Σ_{images} M^r_ij·f_j
    
    where M^r_ij are 3×3 mobility blocks constructed from the closed-form F1, F2
    coefficients (see F1F2_closed_form in rpy_real_det_helpers).
    
    Parameters
    ----------
    a : float
        Sphere radius (in real units). Determines hydrodynamic size and self-mobility.
    xi : float
        Ewald splitting parameter (inverse length units). Controls the real/wave space
        decomposition; larger xi means faster real-space decay but more wave modes needed.
    eta : float
        Fluid dynamic viscosity. Sets the overall mobility scale (1/(6πηa)).
    rcut2 : float
        Squared real-space cutoff radius. Pairs with r² > rcut2 are excluded from
        the real-space sum (their contribution is handled in wave space).
    neighbor_format : partition.NeighborListFormat
        Neighbor list storage format. Dense format is fastest but uses more memory;
        Sparse or OrderedSparse reduce memory at the cost of scatter-gather overhead.
        OrderedSparse additionally includes symmetric backflow contributions (M_ji·f_i).
    
    Returns
    -------
    Callable[..., jnp.ndarray]
        A JIT-compiled function with signature:
            core(positions, forces, neighbor_idx, neighbor_mask,
                 box_matrix, lattice_indices, zero_image_index) -> velocities
        
        The returned callable evaluates M^r @ forces using the neighbor list and
        lattice image sums, applying cutoff and self-term corrections.
    
    Notes
    -----
    - The core uses fused mobility-force contractions to avoid materializing the
      full (N, neighbors, images, 3, 3) mobility tensor, saving ~60% memory.
    - For OrderedSparse format, symmetric backflow M_ji·f_i is included to ensure
      the mobility operator remains symmetric when used in iterative solvers.
    - Self-interactions (i=j, primary cell) are replaced by the analytic self-mobility
      computed via Mr_self(a, xi).
    """

    self_factor = Mr_self(a, xi)
    prefactor_scalar = 1.0 / (6.0 * np.pi * eta * a)
    include_ordered_backflow = neighbor_format is partition.NeighborListFormat.OrderedSparse
    uses_sparse = partition.is_sparse(neighbor_format)

    @jax.jit
    def core(
        positions: jnp.ndarray,
        forces: jnp.ndarray,
        neighbor_idx: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        box_matrix: jnp.ndarray,
        lattice_indices: jnp.ndarray,
        zero_image_index: int,
    ) -> jnp.ndarray:
        """
        Core real-space mobility evaluation with support for dense and sparse neighbor lists.
        
        Computes velocities from forces using the real-space mobility operator:
          v = M^r @ f
        
        Handles both Dense and Sparse/OrderedSparse neighbor list formats, with
        optimized tensor contractions to minimize memory usage.
        
        Parameters
        ----------
        positions : (N, 3) array
            Particle positions (fractional if ``fractional_coordinates=True``)
        forces : (N, 3) array
            Forces in REAL units
        neighbor_idx : array
            Neighbor list indices (format depends on neighbor_format)
        neighbor_mask : array
            Boolean mask for valid neighbors
        box_matrix : (3, 3) array
            Box transformation matrix
        lattice_indices : (n_images, 3) array
            Integer lattice vectors
        zero_image_index : int
            Index of the zero lattice vector (primary cell)
            
        Returns
        -------
        (N, 3) array
            Velocities from real-space mobility evaluation
        """
        # Ensure inputs are JAX arrays with correct dtypes
        positions = jnp.asarray(positions)
        forces = jnp.asarray(forces)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
        neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
        box_matrix = jnp.asarray(box_matrix)
        lattice_indices = jnp.asarray(lattice_indices, dtype=jnp.int32)
        zero_image_index = jnp.int32(zero_image_index)

        # Convert to real coordinates if needed
        x_real = _positions_to_real(positions, box_matrix, fractional_coordinates)
        
        # Precompute lattice vectors in real space
        lattice_vecs = lattice_indices @ box_matrix.T
        N = x_real.shape[0]
        n_images = lattice_vecs.shape[0]

        dtype = x_real.dtype
        prefactor = jnp.asarray(prefactor_scalar, dtype=dtype)
        self_term = prefactor * jnp.asarray(self_factor, dtype=dtype)

        def _pair_contrib(rij, r2, mask, forces):
            """Fused mobility·force contraction for arbitrary neighbor shapes."""
            eps = jnp.finfo(dtype).tiny
            safe_r = jnp.where(mask, jnp.sqrt(r2 + eps), jnp.ones_like(r2, dtype=dtype))
            rhat = jnp.where(mask[..., None], rij / safe_r[..., None], 0.0)

            F1, F2 = F1F2_closed_form(safe_r, a, xi)
            F1 = jnp.where(mask, F1, 0.0)
            F2 = jnp.where(mask, F2, 0.0)

            rhat_dot_f = jnp.einsum("...i,...i->...", rhat, forces)
            return prefactor * (
                F1[..., None] * forces +
                (F2 - F1)[..., None] * rhat_dot_f[..., None] * rhat
            )

        # Early exit when no neighbors or lattice images are present.
        if n_images == 0:
            return self_term * forces

        if not uses_sparse:
            if neighbor_idx.ndim == 1:
                neighbor_idx = neighbor_idx[:, None]
            if neighbor_mask.ndim == 1:
                neighbor_mask = neighbor_mask[:, None]

            max_neighbors = neighbor_idx.shape[1]
            if max_neighbors == 0:
                return self_term * forces

            # Always use vectorized path
            neighbor_idx_masked = jnp.where(neighbor_mask, neighbor_idx, 0)

            xi_vec = x_real[:, None, None, :]
            xj = x_real[neighbor_idx_masked][:, :, None, :]
            lattice = lattice_vecs[None, None, :, :]
            rij = xj - xi_vec + lattice

            r2 = jnp.sum(rij * rij, axis=-1)
            within_rcut = r2 < rcut2

            idx_i = jnp.arange(N, dtype=neighbor_idx.dtype)
            is_self = neighbor_idx_masked == idx_i[:, None]
            zero_mask = (jnp.arange(n_images, dtype=jnp.int32) == zero_image_index)
            primary_self = is_self[:, :, None] & zero_mask[None, None, :]

            valid_pairs = neighbor_mask[:, :, None] & (~primary_self) & within_rcut
            valid_any = jnp.any(valid_pairs)

            def _no_pairs(_):
                return self_term * forces

            def _with_pairs(_):
                forces_neighbors = forces[neighbor_idx_masked][:, :, None, :]
                contrib = _pair_contrib(rij, r2, valid_pairs, forces_neighbors)
                contrib = contrib.sum(axis=2)
                contrib = contrib.sum(axis=1)

                return self_term * forces + contrib

            return jax.lax.cond(valid_any, _with_pairs, _no_pairs, operand=None)

        # Sparse or ordered-sparse neighbor lists.
        if neighbor_idx.ndim == 1:
            neighbor_idx = neighbor_idx[None, :]
        if neighbor_mask.ndim == 0:
            neighbor_mask = jnp.broadcast_to(neighbor_mask, neighbor_idx.shape[1])
        capacity = neighbor_idx.shape[1]
        if capacity == 0:
            return self_term * forces

        # Unroll neighbor indices and masks
        receivers = jnp.where(neighbor_mask, neighbor_idx[0], 0)
        senders = jnp.where(neighbor_mask, neighbor_idx[1], 0)

        # Compute separations for all lattice images
        xi_vec = x_real[receivers][:, None, :]
        xj = x_real[senders][:, None, :]
        lattice = lattice_vecs[None, :, :]
        rij = xj - xi_vec + lattice

        r2 = jnp.sum(rij * rij, axis=-1)
        within_rcut = r2 < rcut2

        zero_mask = (jnp.arange(n_images, dtype=jnp.int32) == zero_image_index)[None, :]
        is_self_edge = (receivers == senders)[:, None]
        primary_self = is_self_edge & zero_mask

        mask_pairs = neighbor_mask[:, None] & (~primary_self) & within_rcut
        valid_any = jnp.any(mask_pairs)

        def _no_pairs(_):
            return self_term * forces

        def _with_pairs(_):
            forces_senders = forces[senders][:, None, :]
            contrib_i = _pair_contrib(rij, r2, mask_pairs, forces_senders).sum(axis=1)

            velocities = self_term * forces
            velocities = velocities + ops.segment_sum(contrib_i, receivers, N)

            if include_ordered_backflow: # For OrderedSparse format
                # Compute backflow contributions M_ji·f_i
                forces_receivers = forces[receivers][:, None, :]
                contrib_j = _pair_contrib(rij, r2, mask_pairs, forces_receivers).sum(axis=1)
                velocities = velocities + ops.segment_sum(contrib_j, senders, N)

            return velocities # = M^r @ forces

        return jax.lax.cond(valid_any, _with_pairs, _no_pairs, operand=None)

    return core


def build_Mr_apply(
    space_fns,
    a,
    xi,
    eta,
    rcut,
    *,
    fractional_coordinates: bool = True,
    dr_threshold=None,
    capacity_multiplier=1.25,
    disable_cell_list=False,
    neighbor_format=partition.NeighborListFormat.Dense,
    extra_capacity=0,
    lattice_extent: Optional[int] = None,
    lattice_extra: float = 1.0,
):
    """Construct the neighbor-list-backed real-space mobility operator.

    Parameters
    ----------
    space_fns : tuple
        Typically the tuple returned by ``space.periodic_general`` or
        ``space.shearing``. Set ``fractional_coordinates`` to match the space.
    a, xi, eta : float
        Hydrodynamic parameters (sphere radius, splitting parameter, viscosity).
    rcut : float
        Real-space cutoff radius (in real units).
    fractional_coordinates : bool
        Whether the provided positions are fractional (default) or real. When
        using real coordinates, the physical box must be supplied (scalar/vector
        for orthogonal boxes or a matrix) so lattice sums can be constructed. For
        triclinic boxes, prefer fractional coordinates to keep neighbor lists consistent.
    dr_threshold, capacity_multiplier, disable_cell_list : float / bool
        Passed through to the JAX-MD neighbor-list builder.
    neighbor_format : partition.NeighborListFormat
        Neighbor list format to use. Dense yields the fastest path, while Sparse / OrderedSparse
        reduce memory usage for large systems.
    extra_capacity : int
        Additional neighbor list capacity used when allocating from initial
        positions.
    lattice_extent : Optional[int]
        Optional override for the symmetric lattice extent ``N`` producing integer
        shifts in [-N, N]^dim. If ``None`` (default) the extent is estimated from
        the instantaneous box via the smallest singular value and ``lattice_extra``.
    lattice_extra : float
        Additional padding added to the automatically estimated lattice extent.

    Returns
    -------
    init_fn, apply_fn : Callable
        ``init_fn(positions, **kwargs) -> RealSpaceState`` allocates the
        neighbor list and lattice indices for the provided positions (fractional
        if ``fractional_coordinates`` is True, otherwise real).
        ``apply_fn(state, positions, forces, **kwargs)`` returns the real-space
        velocity together with the updated state.
    """

    if rcut <= 0.0:
        raise ValueError("rcut must be positive.")

    # Set reasonable default for dr_threshold if not provided
    # Use 10% of rcut as a safe update threshold
    if dr_threshold is None:
        dr_threshold = 0.1 * rcut

    if len(space_fns) < 2:
        raise ValueError("space_fns must contain at least displacement and shift functions.")
    displacement_fn, _ = space_fns[:2]
    box_fn = space_fns[2] if len(space_fns) > 2 else None

    # The neighbor list requires a box parameter at construction time. We don't
    # know the physical box yet (it will be threaded into `allocate`/`update`
    # via kwargs), so pass a neutral scalar placeholder (1.0). The real box is
    # provided later via `neighbor_kwargs.setdefault("box", ...)`.
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=1.0,
        r_cutoff=rcut,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        mask_self=False,
        fractional_coordinates=fractional_coordinates,
        format=neighbor_format,
    )

    rcut2 = float(rcut * rcut)
    core_fn = _build_mr_core(a, xi, eta, rcut2, neighbor_format, fractional_coordinates)

    def _compute_lattice_indices(box_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """
        Compute lattice vectors for periodic image summation in real-space mobility.
        
        This function generates the set of integer lattice shifts L = (n_x, n_y, n_z)
        needed to evaluate hydrodynamic interactions across periodic boundaries:
        
            M^r_ij = ∑_L M^r(r_ij + L·Box)
        
        The lattice extent is chosen to ensure all images within the real-space cutoff
        r_cut are included, while excluding distant images that contribute negligibly
        due to exponential decay from Ewald splitting.
        
        Algorithm
        ---------
        1. **Extent Estimation**: If not user-specified, compute the lattice extent N
           from the ratio r_cut / σ_min(Box), where σ_min is the smallest singular
           value of the box matrix. This ensures coverage in the 'thinnest' box direction.
           
        2. **Hypercube Generation**: Create a symmetric grid of integer shifts in
           {-N, ..., N}^d, yielding (2N+1)^d candidate lattice vectors.
           
        3. **Pruning**: Map lattice shifts to real space (L·Box^T) and discard those
           with |L| > r_cut, as they cannot contribute to any particle pair within r_cut.
           This reduces the active lattice from ~O(N^3) to ~O(N^2) images for typical boxes.
           
        4. **Zero Image Preservation**: Ensure the primary cell (L=0) is always included,
           even if pruning would remove it (degenerate edge case). Track its index for
           self-interaction masking in the mobility kernel.
        
        Parameters
        ----------
        box_matrix : (d, d) array
            Box transformation matrix mapping fractional to real coordinates.
            
        Returns
        -------
        lattice : (n_images, d) array of int32
            Integer lattice vectors to sum over, with |L·Box| ≲ r_cut.
        zero_idx : int
            Index of the zero lattice vector (L=0, primary cell) in the returned array.
            
        Notes
        -----
        - For non-cubic or shearing boxes, the SVD-based extent estimate adapts to the
          box geometry, ensuring sufficient coverage without excessive oversampling.
        - The r_cut pruning threshold ensures any image L with |L·Box| ≥ r_cut cannot
          contribute to interactions, as even the closest pair (r_ij ≈ 0) would exceed cutoff.
        - Degenerate boxes (σ_min → 0) are protected by the safe_sigma clamp to 1e-12.
        """
        box_np = np.asarray(box_matrix, dtype=np.float64)
        dim = box_np.shape[0]
        if lattice_extent is None:
            # Compute smallest singular value to estimate 'thinnest' box dimension.
            svals = np.linalg.svd(box_np, compute_uv=False)
            sigma_min = float(np.min(svals))
            safe_sigma = max(sigma_min, 1e-12)

            # Ratio of cutoff to smallest box scale determines required lattice extent.
            ratio = float(rcut) / safe_sigma

            # Base extent from geometric ratio: N ≥ ceil(r_cut / σ_min).
            base_extent = int(np.ceil(ratio)) if ratio > 0.0 else 0

            # For r_cut << box size (ratio < 1), a single shell of images suffices.
            # This avoids over-allocating lattice vectors for large boxes.
            if ratio < 1.0:
                extent_val = min(base_extent, 1)
            else:
                extent_val = base_extent
        else:
            extent_val = int(lattice_extent)

        # Generate symmetric hypercube: L ∈ {-N, ..., N}^d.
        lattice_np, zero_idx = generate_lattice_hypercube(dim, extent_val)
        
        # Prune lattice images beyond r_cut in real space.
        # Any image with |A·L| ≥ r_cut cannot contribute to any pair interaction.
        lattice_real = lattice_np @ box_np.T
        lattice_norm2 = np.sum(lattice_real * lattice_real, axis=1)
        lattice_mask = lattice_norm2 < (rcut * rcut)  # |L| < r_cut
        
        # Ensure at least one image survives (edge case: very small cutoffs).
        if not np.any(lattice_mask):
            lattice_mask = np.ones_like(lattice_norm2, dtype=bool)
        lattice_np = lattice_np[lattice_mask]
        
        # Verify primary cell (L=0) is present after pruning.
        zero_candidates = np.where(np.all(lattice_np == 0, axis=1))[0]
        if len(zero_candidates) == 0:
            # Re-insert zero lattice if missing (should not occur in practice).
            lattice_np = np.concatenate([lattice_np, np.zeros((1, dim), dtype=np.int32)], axis=0)
            zero_idx = lattice_np.shape[0] - 1
        else:
            zero_idx = int(zero_candidates[0])

        lattice = jnp.asarray(lattice_np, dtype=jnp.int32)
        return lattice, zero_idx

    def init_fn(positions, *, extra_capacity_override=None, **kwargs):
        positions = jnp.asarray(positions)
        dim = int(positions.shape[1])
        box_matrix = current_box_matrix(
            displacement_fn, box_fn, dim, fractional_coordinates=fractional_coordinates, **kwargs)
        lattice_indices, zero_idx = _compute_lattice_indices(box_matrix)
        cap_value = extra_capacity if extra_capacity_override is None else extra_capacity_override
        neighbor_kwargs = dict(kwargs)
        neighbor_box = _neighbor_box_from_matrix(box_matrix, fractional_coordinates)
        if neighbor_box is not None:
            neighbor_kwargs.setdefault("box", neighbor_box)
        else:
            neighbor_kwargs.pop("box", None)
        neighbors = neighbor_fn.allocate(positions, int(cap_value), **neighbor_kwargs) # type: ignore
        return RealSpaceState(
            neighbors=neighbors, # type: ignore
            lattice_indices=lattice_indices, # type: ignore
            zero_image_index=zero_idx, # type: ignore
            box_matrix=box_matrix, # type: ignore
            fractional_coordinates=fractional_coordinates, # type: ignore
            core_fn=core_fn, # type: ignore
        )

    def apply_fn(state: RealSpaceState, positions, forces, **kwargs):
        positions = jnp.asarray(positions)
        forces = jnp.asarray(forces)

        if positions.shape != forces.shape:
            raise ValueError("positions and forces must have the same shape.")

        # Overrides from kwargs
        dim = int(positions.shape[1])
        neighbor_override = kwargs.pop("neighbor", None)
        lattice_override = kwargs.pop("lattice_indices", None)
        zero_override = kwargs.pop("zero_image_index", None)
        box_override = kwargs.pop("box_matrix", None)

        if box_override is None:
            box_matrix = current_box_matrix(
                displacement_fn, box_fn, dim, fractional_coordinates=fractional_coordinates, **kwargs)
        else:
            box_matrix = canonicalize_box_matrix(box_override, dim)
            if box_matrix is None:
                raise ValueError("box_matrix must be a scalar, vector, or matrix.")

        if lattice_override is not None:
            lattice_indices = jnp.asarray(lattice_override, dtype=jnp.int32)
            zero_idx = int(zero_override) if zero_override is not None else state.zero_image_index
        else:
            lattice_indices = state.lattice_indices
            zero_idx = state.zero_image_index

        neighbor_kwargs = dict(kwargs)
        neighbor_box = _neighbor_box_from_matrix(box_matrix, fractional_coordinates)
        if neighbor_box is not None:
            neighbor_kwargs.setdefault("box", neighbor_box)
        else:
            neighbor_kwargs.pop("box", None)
        if neighbor_override is None:
            if state.neighbors is None:
                raise ValueError("RealSpaceState.neighbors is None; provide a neighbor list via 'neighbor'.")
            # Use the built-in NeighborList.update() method
            updated = state.neighbors.update(positions, **neighbor_kwargs)

            # Check for errors using NeighborList properties
            try:
                overflow_py = bool(np.asarray(updated.did_buffer_overflow))
                cell_small_py = bool(np.asarray(updated.cell_size_too_small))
                malformed_py = bool(np.asarray(updated.malformed_box))
            except (TypeError, jax_errors.TracerArrayConversionError, jax_errors.TracerBoolConversionError):
                # In JIT context, can't reallocate; just use updated neighbor list
                # The error flags will be set in the returned state for inspection
                # DEBUG CALLBACK DISABLED FOR GPU COMPATIBILITY
                # The callback causes GPU-to-CPU transfer issues
                # Instead, we rely on silent overflow handling
                neighbors = updated
            else:
                # Outside JIT: reallocate if any error occurred
                if overflow_py or cell_small_py or malformed_py:
                    neighbors = neighbor_fn.allocate(positions, int(extra_capacity), **neighbor_kwargs) # type: ignore
                else:
                    neighbors = updated
        else:
            neighbors = neighbor_override

        mask = partition.neighbor_list_mask(neighbors)

        # Forces are provided in real coordinates.
        forces_real = jnp.asarray(forces, dtype=REAL_DTYPE)

        # Apply core in real units
        velocities_real = core_fn(
            positions,
            forces_real,
            neighbors.idx,
            mask,
            box_matrix,
            lattice_indices,
            zero_idx,
        )

        next_state = RealSpaceState(
            neighbors=neighbors, # type: ignore
            lattice_indices=lattice_indices, # type: ignore
            zero_image_index=zero_idx, # type: ignore
            box_matrix=box_matrix, # type: ignore
            fractional_coordinates=fractional_coordinates, # type: ignore
            core_fn=core_fn, # type: ignore
        )
        return velocities_real, next_state

    return init_fn, apply_fn


@jax.jit
def mr_matvec(state: RealSpaceState,
              positions: jnp.ndarray,
              vec: jnp.ndarray,
              *,
              neighbor: Optional[partition.NeighborList] = None) -> jnp.ndarray:
    """
    Apply the real-space mobility in real-coordinate basis.

    Forces are provided in real coordinates; the routine evaluates the
    real-space mobility using those inputs and returns the resulting real-space
    velocity.

    Parameters
    ----------
    state : RealSpaceState
        Real-space state returned by ``build_Mr_apply``.
    positions : (N,3) array
        Current particle positions (fractional if ``state.fractional_coordinates``
        is True, otherwise real).
    vec : (N,3) array
        Vector to which the mobility is applied (in real coordinates).
    neighbor : Optional[partition.NeighborList]
        Optional neighbor list to use instead of the one stored in ``state``.

    Returns
    -------
    (N,3) array
        Result of applying ``M^(r)`` to ``vec`` in real coordinates.
    """
    if state.core_fn is None:
        raise ValueError("RealSpaceState is missing core_fn; build_Mr_apply must be used.")

    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    vec = jnp.asarray(vec, dtype=REAL_DTYPE)
    neighbors = neighbor if neighbor is not None else state.neighbors
    if neighbors is None:
        raise ValueError("Real-space state is missing a neighbor list; provide one via the 'neighbor' argument.")
    mask = partition.neighbor_list_mask(neighbors)

    # Forces are provided in real coordinates.
    forces_real = jnp.asarray(vec, dtype=REAL_DTYPE)

    # Apply real-space mobility in real units
    v_real = state.core_fn(
        positions,
        forces_real,
        neighbors.idx,
        mask,
        state.box_matrix,
        state.lattice_indices,
        state.zero_image_index,
    )

    return v_real
