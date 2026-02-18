"""Real-space RPY mobility (M^r) with closed-form kernels.

Computes pair mobility blocks using F1, F2 coefficients for monodisperse
spheres of radius a with Ewald splitting parameter ξ:

    M^r_ij = (1/(6πηa)) [F1(I - r̂⊗r̂) + F2 r̂⊗r̂]
    M^r_ii = (1/(6πηa)) f_self(a, ξ)

Periodic images are summed over a lattice hypercube L ∈ {-N,…,N}^d with
N ≥ ⌈2·r_cut / σ_min(Box)⌉, and pair-level masking enforces |r_ij + L| < r_cut.

References
----------
[1] Fiore et al., J. Chem. Phys. 146, 124116 (2017).
[2] Fiore, PhD Thesis, MIT (2019).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors
from functools import partial

from typing import Callable, Literal, Optional, Tuple

from jax_md import dataclasses, partition, space
from jax import ops
from jax_md.hydro.rpy_real_det_helpers import (
    REAL_DTYPE,
    PAIR_EPS_FRACTION_OF_DIAMETER,
    F1F2_closed_form,
    Mr_self,
    current_box_matrix,
    canonicalize_box_matrix,
    generate_lattice_hypercube,
)


I3 = jnp.eye(3, dtype=REAL_DTYPE) # 3x3 identity matrix


RealSpaceMode = Literal['auto', 'min_image', 'lattice']
_REAL_SPACE_MODES = frozenset({'auto', 'min_image', 'lattice'})


def _validate_real_space_mode(mode: str) -> RealSpaceMode:
    mode_str = str(mode)
    if mode_str not in _REAL_SPACE_MODES:
        raise ValueError(
            f"real_space_mode must be one of {_REAL_SPACE_MODES}; got {mode_str!r}."
        )
    return mode_str  # type: ignore[return-value]


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
    # Compute separation related quantities.
    r2 = jnp.dot(r_vec, r_vec)
    pair_eps = jnp.asarray((2.0 * a) * PAIR_EPS_FRACTION_OF_DIAMETER, dtype=r_vec.dtype)
    pair_eps2 = pair_eps * pair_eps
    tiny_pair = r2 <= pair_eps2
    safe_r = jnp.sqrt(jnp.maximum(r2, pair_eps2))
    rhat = jnp.where(tiny_pair, jnp.zeros_like(r_vec), r_vec / safe_r)
    rhat_outer = jnp.outer(rhat, rhat)
    # Get mobility coefficients (Appx. A of Fiore et al.)
    F1, F2 = F1F2_closed_form(safe_r, a, xi)
    
    prefactor = 1.0 / (6.0 * jnp.pi * eta * a)
    anisotropic = prefactor * (F1 * (I3 - rhat_outer) + F2 * rhat_outer)
    # Near exact overlap, use an isotropic contraction evaluated at r=0+eps.
    F_iso = (2.0 * F1 + F2) / 3.0
    isotropic = prefactor * F_iso * I3
    return jnp.where(tiny_pair, isotropic, anisotropic)


def _build_pair_contrib_fn(a: float, xi: float, eta: float):
    """Build shared pairwise mobility-force contraction helpers."""
    self_factor = Mr_self(a, xi)
    pair_eps2_scalar = float(((2.0 * a) * PAIR_EPS_FRACTION_OF_DIAMETER) ** 2)
    prefactor_scalar = 1.0 / (6.0 * np.pi * eta * a)

    def pair_contrib(rij, r2, mask, forces, *, prefactor, pair_eps2):
        """Fused mobility·force contraction for arbitrary neighbor shapes."""
        safe_r = jnp.where(
            mask,
            jnp.sqrt(jnp.maximum(r2, pair_eps2)),
            jnp.ones_like(r2, dtype=rij.dtype))
        rhat = jnp.where(mask[..., None], rij / safe_r[..., None], 0.0)

        F1, F2 = F1F2_closed_form(safe_r, a, xi)
        F1 = jnp.where(mask, F1, 0.0)
        F2 = jnp.where(mask, F2, 0.0)

        rhat_dot_f = jnp.einsum("...i,...i->...", rhat, forces)
        anisotropic = prefactor * (
            F1[..., None] * forces +
            (F2 - F1)[..., None] * rhat_dot_f[..., None] * rhat
        )
        F_iso = (2.0 * F1 + F2) / 3.0
        isotropic = prefactor * F_iso[..., None] * forces
        tiny_pairs = mask & (r2 <= pair_eps2)
        return jnp.where(tiny_pairs[..., None], isotropic, anisotropic)

    return self_factor, pair_eps2_scalar, prefactor_scalar, pair_contrib


def _build_mr_core_lattice(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
    fractional_coordinates: bool,
) -> Callable[..., jnp.ndarray]:
    """Build the original lattice-image real-space kernel."""
    self_factor, pair_eps2_scalar, prefactor_scalar, pair_contrib = _build_pair_contrib_fn(a, xi, eta)
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
        positions = jnp.asarray(positions)
        forces = jnp.asarray(forces)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
        neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
        box_matrix = jnp.asarray(box_matrix)
        lattice_indices = jnp.asarray(lattice_indices, dtype=jnp.int32)
        zero_image_index = jnp.int32(zero_image_index)

        x_real = _positions_to_real(positions, box_matrix, fractional_coordinates)
        lattice_vecs = lattice_indices @ box_matrix.T
        N = x_real.shape[0]
        n_images = lattice_vecs.shape[0]

        dtype = x_real.dtype
        prefactor = jnp.asarray(prefactor_scalar, dtype=dtype)
        self_term = prefactor * jnp.asarray(self_factor, dtype=dtype)
        pair_eps2 = jnp.asarray(pair_eps2_scalar, dtype=dtype)

        if n_images == 0:
            return self_term * forces

        # --- Normalise all formats to a flat (capacity,) edge list ---
        if not uses_sparse:
            # Dense: neighbor_idx shape (N, max_K), neighbor_mask shape (N, max_K)
            if neighbor_idx.ndim == 1:
                neighbor_idx = neighbor_idx[:, None]
            if neighbor_mask.ndim == 1:
                neighbor_mask = neighbor_mask[:, None]
            max_neighbors = neighbor_idx.shape[1]
            if max_neighbors == 0:
                return self_term * forces
            neighbor_idx_masked = jnp.where(neighbor_mask, neighbor_idx, 0)
            idx_i = jnp.arange(N, dtype=jnp.int32)
            receivers = jnp.repeat(idx_i, max_neighbors)
            senders = neighbor_idx_masked.ravel()
            flat_mask = neighbor_mask.ravel()
        else:
            # Sparse / OrderedSparse: neighbor_idx shape (2, capacity)
            if neighbor_idx.ndim == 1:
                neighbor_idx = neighbor_idx[None, :]
            if neighbor_mask.ndim == 0:
                neighbor_mask = jnp.broadcast_to(neighbor_mask, neighbor_idx.shape[1:])
            capacity = neighbor_idx.shape[1]
            if capacity == 0:
                return self_term * forces
            receivers = jnp.where(neighbor_mask, neighbor_idx[0], 0)
            senders = jnp.where(neighbor_mask, neighbor_idx[1], 0)
            flat_mask = neighbor_mask

        # --- Shared accumulation (lattice kernel) ---
        zero_mask = (jnp.arange(n_images, dtype=jnp.int32) == zero_image_index)[None, :]
        lattice = lattice_vecs[None, :, :]
        velocities_init = self_term * forces

        def _accumulate_sparse_batch(
            velocities: jnp.ndarray,
            receivers_batch: jnp.ndarray,
            senders_batch: jnp.ndarray,
            edge_mask_batch: jnp.ndarray,
        ) -> jnp.ndarray:
            xi_vec = x_real[receivers_batch][:, None, :]
            xj = x_real[senders_batch][:, None, :]
            rij = xj - xi_vec + lattice

            r2 = jnp.sum(rij * rij, axis=-1)
            within_rcut = r2 < rcut2
            is_self_edge = (receivers_batch == senders_batch)[:, None]
            primary_self = is_self_edge & zero_mask
            mask_pairs = edge_mask_batch[:, None] & (~primary_self) & within_rcut

            forces_senders = forces[senders_batch][:, None, :]
            contrib_i = pair_contrib(
                rij, r2, mask_pairs, forces_senders,
                prefactor=prefactor, pair_eps2=pair_eps2).sum(axis=1)
            velocities = velocities + ops.segment_sum(contrib_i, receivers_batch, N)

            if include_ordered_backflow:
                forces_receivers = forces[receivers_batch][:, None, :]
                contrib_j = pair_contrib(
                    rij, r2, mask_pairs, forces_receivers,
                    prefactor=prefactor, pair_eps2=pair_eps2).sum(axis=1)
                velocities = velocities + ops.segment_sum(contrib_j, senders_batch, N)
            return velocities

        capacity = flat_mask.shape[0]
        pair_image_work = capacity * n_images
        pair_image_limit = 32_000_000

        if pair_image_work <= pair_image_limit:
            return _accumulate_sparse_batch(velocities_init, receivers, senders, flat_mask)

        chunk_size = max(1, pair_image_limit // max(n_images, 1))
        n_chunks = (capacity + chunk_size - 1) // chunk_size
        padded_capacity = n_chunks * chunk_size
        pad = padded_capacity - capacity

        receivers_padded = jnp.pad(receivers, (0, pad))
        senders_padded = jnp.pad(senders, (0, pad))
        mask_padded = jnp.pad(flat_mask, (0, pad), constant_values=False)

        receivers_chunks = receivers_padded.reshape((n_chunks, chunk_size))
        senders_chunks = senders_padded.reshape((n_chunks, chunk_size))
        mask_chunks = mask_padded.reshape((n_chunks, chunk_size))

        def _scan_body(velocities, chunk):
            receivers_chunk, senders_chunk, mask_chunk = chunk
            velocities = _accumulate_sparse_batch(
                velocities, receivers_chunk, senders_chunk, mask_chunk
            )
            return velocities, None

        velocities, _ = jax.lax.scan(
            _scan_body,
            velocities_init,
            (receivers_chunks, senders_chunks, mask_chunks),
        )
        return velocities

    return core


def _build_mr_core_min_image(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
    fractional_coordinates: bool,
) -> Callable[..., jnp.ndarray]:
    """Build a minimum-image real-space kernel (single image per pair)."""
    self_factor, pair_eps2_scalar, prefactor_scalar, pair_contrib = _build_pair_contrib_fn(a, xi, eta)
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
        del lattice_indices, zero_image_index
        positions = jnp.asarray(positions, dtype=REAL_DTYPE)
        forces = jnp.asarray(forces, dtype=REAL_DTYPE)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
        neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
        box_matrix = jnp.asarray(box_matrix, dtype=REAL_DTYPE)
        dtype = forces.dtype

        prefactor = jnp.asarray(prefactor_scalar, dtype=dtype)
        self_term = prefactor * jnp.asarray(self_factor, dtype=dtype)
        pair_eps2 = jnp.asarray(pair_eps2_scalar, dtype=dtype)

        if fractional_coordinates:
            positions_frac = positions
        else:
            inv_box = jnp.linalg.inv(box_matrix)
            positions_frac = space.transform(inv_box, positions)

        def _wrapped_frac_delta(delta_frac):
            return jnp.mod(delta_frac + 0.5, 1.0) - 0.5

        def _to_real(delta_frac):
            return space.transform(box_matrix, delta_frac)

        N = positions_frac.shape[0]

        # --- Normalise all formats to a flat (capacity,) edge list ---
        if not uses_sparse:
            # Dense: neighbor_idx shape (N, max_K), neighbor_mask shape (N, max_K)
            if neighbor_idx.ndim == 1:
                neighbor_idx = neighbor_idx[:, None]
            if neighbor_mask.ndim == 1:
                neighbor_mask = neighbor_mask[:, None]
            max_neighbors = neighbor_idx.shape[1]
            if max_neighbors == 0:
                return self_term * forces
            neighbor_idx_masked = jnp.where(neighbor_mask, neighbor_idx, 0)
            idx_i = jnp.arange(N, dtype=jnp.int32)
            receivers = jnp.repeat(idx_i, max_neighbors)
            senders = neighbor_idx_masked.ravel()
            flat_mask = neighbor_mask.ravel()
        else:
            # Sparse / OrderedSparse: neighbor_idx shape (2, capacity)
            if neighbor_idx.ndim == 1:
                neighbor_idx = neighbor_idx[None, :]
            if neighbor_mask.ndim == 0:
                neighbor_mask = jnp.broadcast_to(neighbor_mask, neighbor_idx.shape[1:])
            capacity = neighbor_idx.shape[1]
            if capacity == 0:
                return self_term * forces
            receivers = jnp.where(neighbor_mask, neighbor_idx[0], 0)
            senders = jnp.where(neighbor_mask, neighbor_idx[1], 0)
            flat_mask = neighbor_mask

        # --- Shared accumulation (min-image kernel) ---
        delta_frac = _wrapped_frac_delta(positions_frac[senders] - positions_frac[receivers])
        rij = _to_real(delta_frac)
        r2 = jnp.sum(rij * rij, axis=-1)
        within_rcut = r2 < rcut2
        is_self_edge = receivers == senders
        mask_pairs = flat_mask & (~is_self_edge) & within_rcut

        velocities = self_term * forces
        forces_senders = forces[senders]
        contrib_i = pair_contrib(
            rij, r2, mask_pairs, forces_senders,
            prefactor=prefactor, pair_eps2=pair_eps2)
        velocities = velocities + ops.segment_sum(contrib_i, receivers, N)

        if include_ordered_backflow:
            forces_receivers = forces[receivers]
            contrib_j = pair_contrib(
                rij, r2, mask_pairs, forces_receivers,
                prefactor=prefactor, pair_eps2=pair_eps2)
            velocities = velocities + ops.segment_sum(contrib_j, senders, N)

        return velocities

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
    lattice_extra: float = 0.0,
    box_jump_threshold: Optional[float] = None,
    real_space_mode: RealSpaceMode = 'auto',
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
        Non-negative padding added to the automatically estimated lattice extent.
        Ignored when ``lattice_extent`` is explicitly provided.
    box_jump_threshold : Optional[float]
        Outside JIT, force neighbor-list reallocation when the Frobenius norm
        of the box-matrix change exceeds this threshold. If None, defaults to
        ``dr_threshold``.
        In traced/JIT dynamic-box contexts with implicit ``lattice_extent``, strict
        Fiore mode raises an error instead of silently under-covering image sums.
    real_space_mode : {'auto', 'min_image', 'lattice'}
        Real-space evaluation kernel. ``lattice`` always uses the periodic-image
        lattice sum, ``min_image`` always uses a minimum-image-only kernel (valid
        only when ``rcut <= 0.5 * sigma_min(box)``), and ``auto`` selects the
        minimum-image kernel when this safety condition holds.

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
    if lattice_extra < 0.0:
        raise ValueError("lattice_extra must be non-negative.")
    if lattice_extent is not None and int(lattice_extent) < 0:
        raise ValueError("lattice_extent must be non-negative when provided.")

    # Set reasonable default for dr_threshold if not provided
    # Use 10% of rcut as a safe update threshold
    if dr_threshold is None:
        dr_threshold = 0.1 * rcut
    if box_jump_threshold is None:
        box_jump_threshold = float(dr_threshold)
    if box_jump_threshold < 0.0:
        raise ValueError("box_jump_threshold must be non-negative.")
    mode = _validate_real_space_mode(real_space_mode)

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
    core_lattice = _build_mr_core_lattice(
        a, xi, eta, rcut2, neighbor_format, fractional_coordinates)
    core_min_image = _build_mr_core_min_image(
        a, xi, eta, rcut2, neighbor_format, fractional_coordinates)

    def _compute_lattice_indices(box_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """
        Compute lattice vectors for periodic image summation in real-space mobility.
        
        This function generates the set of integer lattice shifts L = (n_x, n_y, n_z)
        needed to evaluate hydrodynamic interactions across periodic boundaries:
        
            M^r_ij = ∑_L M^r(r_ij + L·Box)
        
        The lattice extent is chosen to ensure all images needed for strict
        real-space coverage within r_cut are present.
        
        Algorithm
        ---------
        1. **Extent Estimation**: If not user-specified, compute the lattice extent N
           from the ratio 2·r_cut / σ_min(Box), where σ_min is the smallest singular
           value of the box matrix. This ensures coverage in the 'thinnest' box direction,
           accounting for neighbor-list minimum-image shifts L_min ∈ {-1,0,1}³.
           
        2. **Hypercube Generation**: Create a symmetric grid of integer shifts in
           {-N, ..., N}^d, yielding (2N+1)^d candidate lattice vectors.
           
        3. **Zero Image Preservation**: Ensure the primary cell (L=0) is included and
           track its index for self-interaction masking in the mobility kernel.
        
        Parameters
        ----------
        box_matrix : (d, d) array
            Box transformation matrix mapping fractional to real coordinates.
            
        Returns
        -------
        lattice : (n_images, d) array of int32
            Integer lattice vectors to sum over.
        zero_idx : int
            Index of the zero lattice vector (L=0, primary cell) in the returned array.
            
        Notes
        -----
        - For non-cubic or shearing boxes, the SVD-based extent estimate adapts to the
          box geometry, ensuring sufficient coverage without excessive oversampling.
        - Degenerate boxes (σ_min → 0) are protected by the safe_sigma clamp to 1e-12.
        """
        box_np = np.asarray(box_matrix, dtype=np.float64)
        dim = box_np.shape[0]
        if lattice_extent is None:
            # Compute smallest singular value to estimate 'thinnest' box dimension.
            svals = np.linalg.svd(box_np, compute_uv=False)
            sigma_min = float(np.min(svals))
            safe_sigma = max(sigma_min, 1e-12)
            base_extent = int(np.ceil(2.0 * float(rcut) / safe_sigma))
            extent_padding = int(np.ceil(float(lattice_extra)))
            extent_val = max(base_extent + extent_padding, 0)
            if extent_val > 1:
                warnings.warn(
                    f"Real-space lattice extent is {extent_val} (box thinnest direction "
                    f"σ_min={safe_sigma:.4g}, rcut={float(rcut):.4g}). "
                    f"Each pair will be evaluated over {(2*extent_val+1)**dim} periodic images, "
                    f"which may be slow. Consider increasing ξ (xi) to reduce rcut, or "
                    f"ensure the box is not too elongated.",
                    stacklevel=4,
                )
        else:
            extent_val = int(lattice_extent)

        # Generate symmetric hypercube: L ∈ {-N, ..., N}^d.
        lattice_np, zero_idx = generate_lattice_hypercube(dim, extent_val)

        lattice = jnp.asarray(lattice_np, dtype=jnp.int32)
        return lattice, zero_idx

    def _sigma_min(box_matrix: jnp.ndarray) -> float:
        box_np = np.asarray(box_matrix, dtype=np.float64)
        svals = np.linalg.svd(box_np, compute_uv=False)
        return max(float(np.min(svals)), 1e-12)

    def _is_min_image_safe(box_matrix: jnp.ndarray) -> bool:
        return float(rcut) <= 0.5 * _sigma_min(box_matrix)

    def _select_core(box_matrix: jnp.ndarray) -> Tuple[Callable, str]:
        is_tracer = isinstance(box_matrix, jax.core.Tracer)

        if mode == 'lattice':
            return core_lattice, 'lattice'

        if mode == 'min_image':
            if is_tracer:
                raise ValueError(
                    "real_space_mode='min_image' with traced/dynamic box is unsupported "
                    "because safety cannot be verified at trace time."
                )
            if not _is_min_image_safe(box_matrix):
                sigma_min = _sigma_min(box_matrix)
                raise ValueError(
                    "real_space_mode='min_image' requires rcut <= 0.5 * sigma_min(box). "
                    f"Got rcut={float(rcut):.6g}, 0.5*sigma_min={0.5 * sigma_min:.6g}."
                )
            return core_min_image, 'min_image'

        # mode == 'auto'
        if is_tracer:
            return core_lattice, 'lattice'
        if _is_min_image_safe(box_matrix):
            return core_min_image, 'min_image'
        return core_lattice, 'lattice'

    def init_fn(positions, *, extra_capacity_override=None, **kwargs):
        positions = jnp.asarray(positions)
        dim = int(positions.shape[1])
        box_matrix = current_box_matrix(
            displacement_fn, box_fn, dim, fractional_coordinates=fractional_coordinates, **kwargs)
        core_fn_init, _ = _select_core(box_matrix)
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
            core_fn=core_fn_init, # type: ignore
        )

    def apply_fn(state: RealSpaceState,
                 positions,
                 forces,
                 *,
                 neighbor: Optional[partition.NeighborList] = None,
                 lattice_indices: Optional[jnp.ndarray] = None,
                 zero_image_index: Optional[int] = None,
                 box_matrix: Optional[jnp.ndarray] = None,
                 **kwargs):
        positions = jnp.asarray(positions)
        forces = jnp.asarray(forces)

        if positions.shape != forces.shape:
            raise ValueError("positions and forces must have the same shape.")

        dim = int(positions.shape[1])
        if box_matrix is None:
            box_matrix_local = current_box_matrix(
                displacement_fn, box_fn, dim, fractional_coordinates=fractional_coordinates, **kwargs)
        else:
            box_matrix_local = canonicalize_box_matrix(box_matrix, dim)
            if box_matrix_local is None:
                raise ValueError("box_matrix must be a scalar, vector, or matrix.")

        core_fn_selected = state.core_fn
        if core_fn_selected is None:
            raise ValueError("RealSpaceState is missing core_fn; reinitialize with build_Mr_apply().")
        using_lattice_kernel = core_fn_selected is core_lattice
        using_min_image_kernel = core_fn_selected is core_min_image

        if using_min_image_kernel and not isinstance(box_matrix_local, jax.core.Tracer):
            if not _is_min_image_safe(box_matrix_local):
                raise ValueError(
                    "Current box violates minimum-image safety (rcut > 0.5 * sigma_min(box)) "
                    "for the selected real-space kernel. Rebuild in lattice mode."
                )

        # Outside `jit`, we optionally rebuild neighbor/lattice bookkeeping when the
        # box changes abruptly (e.g. due to a host-side resize).
        #
        # Inside `jit`, we must *not* attempt any NumPy conversion or data-dependent
        # Python branching. When the box is *dynamic* (i.e. `box_matrix` itself is a
        # Tracer), we cannot safely recompute lattice indices with a data-dependent
        # extent because that would change array shapes; require an explicit
        # `lattice_extent` (or an explicit `lattice_indices` override) in that case.
        if (using_lattice_kernel and lattice_extent is None and
                lattice_indices is None and isinstance(box_matrix_local, jax.core.Tracer)):
            raise ValueError(
                "Dynamic/traced box requires explicit lattice_extent (or lattice_indices) "
                "so the lattice-image set has static shape under jit."
            )
        force_rebuild_py = False
        if not (isinstance(box_matrix_local, jax.core.Tracer) or isinstance(state.box_matrix, jax.core.Tracer)):
            delta_box = np.asarray(box_matrix_local - state.box_matrix, dtype=np.float64)
            box_jump = float(np.linalg.norm(delta_box))
            force_rebuild_py = box_jump > float(box_jump_threshold)

        if lattice_indices is not None:
            lattice_indices_local = jnp.asarray(lattice_indices, dtype=jnp.int32)
            zero_idx = int(zero_image_index) if zero_image_index is not None else state.zero_image_index
        else:
            if force_rebuild_py and lattice_extent is None:
                lattice_indices_local, zero_idx = _compute_lattice_indices(box_matrix_local)
            else:
                lattice_indices_local = state.lattice_indices
                zero_idx = state.zero_image_index

        neighbor_kwargs = dict(kwargs)
        neighbor_box = _neighbor_box_from_matrix(box_matrix_local, fractional_coordinates)
        if neighbor_box is not None:
            neighbor_kwargs.setdefault("box", neighbor_box)
        else:
            neighbor_kwargs.pop("box", None)
        if neighbor is None:
            if state.neighbors is None:
                raise ValueError("RealSpaceState.neighbors is None; provide a neighbor list via 'neighbor'.")
            if force_rebuild_py:
                neighbors = neighbor_fn.allocate(positions, int(extra_capacity), **neighbor_kwargs) # type: ignore
            else:
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
            neighbors = neighbor

        mask = partition.neighbor_list_mask(neighbors)

        # Forces are provided in real coordinates.
        forces_real = jnp.asarray(forces, dtype=REAL_DTYPE)

        # Apply core in real units
        velocities_real = core_fn_selected(
            positions,
            forces_real,
            neighbors.idx,
            mask,
            box_matrix_local,
            lattice_indices_local,
            zero_idx,
        )

        next_state = RealSpaceState(
            neighbors=neighbors, # type: ignore
            lattice_indices=lattice_indices_local, # type: ignore
            zero_image_index=zero_idx, # type: ignore
            box_matrix=box_matrix_local, # type: ignore
            fractional_coordinates=fractional_coordinates, # type: ignore
            core_fn=core_fn_selected, # type: ignore
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
