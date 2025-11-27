"""
Real-space PSE mobility (M^r) with closed-form kernels

Real-space mobility M^(r) uses closed-form F1,F2 coefficients for
monodisperse spheres of radius a, with Pse splitting parameter ξ.

For a pair at separation r:
  M^r_ij = (1/(6πηa)) * [F1(r) * (I - r̂⊗r̂) + F2(r) * r̂⊗r̂]

Self term (r=0):
  M^r_ii = (1/(6πηa)) * [1/(4√π ξ a)] * [1 - exp(-4a²ξ²) + 4√π a ξ erfc(2aξ)]
  
References
----------
[1] Fiore, Andrew M., Florencio Balboa Usabiaga, Aleksandar Donev, and James W. Swan. “Rapid Sampling of Stochastic Displacements in Brownian Dynamics Simulations.” The Journal of Chemical Physics 146, no. 12 (2017): 124116. https://doi.org/10.1063/1.4978242.
[2] Fiore, Andrew M. “Fast Simulation Methods for Soft Matter Hydrodynamics.” PhD Thesis, Massachusetts Institute of Technology, 2019.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors

from typing import Callable, Optional, Tuple

from jax_md import dataclasses, partition, space
from jax import ops
from jax_md.hydro.pse_real_det_helpers import (
    REAL_DTYPE,
    F1F2_closed_form,
    Mr_self,
)


I3 = jnp.eye(3, dtype=REAL_DTYPE)


@dataclasses.dataclass
class RealSpaceState:
    """State for the real-space mobility operator."""

    neighbors: Optional[partition.NeighborList]
    lattice_indices: jnp.ndarray
    zero_image_index: int
    box_matrix: jnp.ndarray
    core_fn: Callable = dataclasses.field(metadata={'static': True})

@jax.jit
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
        Pse splitting parameter
    eta : float
        Fluid viscosity
        
    Returns
    -------
    Mr : (3,3) array
        Real-space mobility block
    """
    r2 = jnp.dot(r_vec, r_vec)
    r = jnp.sqrt(r2 + 1e-300)
    rhat = r_vec / r
    F1, F2 = F1F2_closed_form(r, a, xi)
    
    prefactor = 1.0 / (6.0 * jnp.pi * eta * a)
    rhat_outer = jnp.outer(rhat, rhat)
    Mr = prefactor * (F1 * (I3 - rhat_outer) + F2 * rhat_outer)
    return Mr


def _current_box_matrix(
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    box_fn: Optional[Callable[..., jnp.ndarray]],
    dim: int,
    **kwargs,
) -> jnp.ndarray:
    """Infer the physical box matrix for a fractional-coordinate space."""

    if box_fn is not None:
        return jnp.asarray(box_fn(**kwargs))

    closure = getattr(displacement_fn, "__closure__", None)
    if closure is not None:
        for cell in closure:
            val = cell.cell_contents
            if hasattr(val, "shape") and val.shape == (dim, dim):
                return jnp.asarray(val)

    origin = jnp.zeros((dim,), dtype=REAL_DTYPE)
    basis = jnp.eye(dim, dtype=REAL_DTYPE)
    cols = [displacement_fn(origin, basis[i], **kwargs) for i in range(dim)]
    return jnp.stack(cols, axis=1)


def _generate_lattice_hypercube(dim: int, extent: int) -> Tuple[np.ndarray, int]:
    """Generate integer lattice indices on the symmetric hypercube [-extent, extent]^dim."""
    extent = max(int(extent), 0)
    ranges = [np.arange(-extent, extent + 1, dtype=np.int32) for _ in range(dim)]
    mesh = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, dim)
    zero_mask = np.all(mesh == 0, axis=1)
    if not zero_mask.any():
        raise RuntimeError("Lattice hypercube generation failed to include the zero vector.")
    zero_idx = int(np.argmax(zero_mask))
    return mesh.astype(np.int32, copy=False), zero_idx


def _build_mr_core(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
) -> Callable[..., jnp.ndarray]:
    """Create the JIT-ed core that evaluates the real-space mobility."""

    self_factor = Mr_self(a, xi)
    prefactor_scalar = 1.0 / (6.0 * np.pi * eta * a)
    include_ordered_backflow = neighbor_format is partition.NeighborListFormat.OrderedSparse
    uses_sparse = partition.is_sparse(neighbor_format)

    @jax.jit
    def core(
        positions_frac: jnp.ndarray,
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
        positions_frac : (N, 3) array
            Fractional particle positions
        forces : (N, 3) array
            Forces in real coordinates
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
        positions_frac = jnp.asarray(positions_frac)
        forces = jnp.asarray(forces)
        neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
        neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
        box_matrix = jnp.asarray(box_matrix)
        lattice_indices = jnp.asarray(lattice_indices, dtype=jnp.int32)
        zero_image_index = jnp.int32(zero_image_index)

        x_real = space.transform(box_matrix, positions_frac)
        lattice_vecs = lattice_indices @ box_matrix.T
        N = x_real.shape[0]
        n_images = lattice_vecs.shape[0]

        dtype = x_real.dtype
        prefactor = jnp.asarray(prefactor_scalar, dtype=dtype)
        self_term = prefactor * jnp.asarray(self_factor, dtype=dtype)

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

            eps = jnp.finfo(dtype).tiny
            safe_r = jnp.sqrt(r2 + eps)
            safe_r = jnp.where(valid_pairs, safe_r, jnp.ones_like(safe_r, dtype=dtype))

            # Normalize separation vectors; drop invalid entries to avoid NaNs.
            rhat = jnp.where(valid_pairs[..., None], rij / safe_r[..., None], 0.0)

            F1, F2 = F1F2_closed_form(safe_r, a, xi)
            F1 = jnp.where(valid_pairs, F1, 0.0)
            F2 = jnp.where(valid_pairs, F2, 0.0)

            # Fused mobility-force contraction to avoid materializing Mr_blocks tensor.
            # M·f = prefactor * [(F1 (I - rr^T) + F2 rr^T)] · f
            #     = prefactor * [F1·f + (F2-F1)·(r^T·f)·r]
            # This eliminates the (edges, images, 3, 3) tensor, saving ~60% memory.
            forces_neighbors = forces[neighbor_idx_masked][:, :, None, :]
            rhat_dot_f = jnp.einsum("...i,...i->...", rhat, forces_neighbors)
            contrib = prefactor * (
                F1[..., None] * forces_neighbors +
                (F2 - F1)[..., None] * rhat_dot_f[..., None] * rhat
            )
            contrib = contrib.sum(axis=2)
            contrib = contrib.sum(axis=1)

            return self_term * forces + contrib

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

        eps = jnp.finfo(dtype).tiny
        safe_r = jnp.sqrt(r2 + eps)
        safe_r = jnp.where(mask_pairs, safe_r, jnp.ones_like(safe_r, dtype=dtype))

        rhat = jnp.where(mask_pairs[..., None], rij / safe_r[..., None], 0.0)

        F1, F2 = F1F2_closed_form(safe_r, a, xi)
        F1 = jnp.where(mask_pairs, F1, 0.0)
        F2 = jnp.where(mask_pairs, F2, 0.0)

        # Fused mobility-force contraction.
        # M·f = prefactor * [(F1 (I - rr^T) + F2 rr^T)] · f
        #     = prefactor * [F1·f + (F2-F1)·(r^T·f)·r]
        forces_senders = forces[senders][:, None, :]
        rhat_dot_f = jnp.einsum("...i,...i->...", rhat, forces_senders)
        contrib_i = prefactor * (
            F1[..., None] * forces_senders +
            (F2 - F1)[..., None] * rhat_dot_f[..., None] * rhat
        )
        contrib_i = contrib_i.sum(axis=1)


        velocities = self_term * forces
        velocities = velocities + ops.segment_sum(contrib_i, receivers, N)

        if include_ordered_backflow:
            forces_receivers = forces[receivers][:, None, :]
            rhat_dot_f_back = jnp.einsum("...i,...i->...", rhat, forces_receivers)
            contrib_j = prefactor * (
                F1[..., None] * forces_receivers +
                (F2 - F1)[..., None] * rhat_dot_f_back[..., None] * rhat
            )
            contrib_j = contrib_j.sum(axis=1)
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
    dr_threshold=0.0,
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
        ``space.shearing`` with ``fractional_coordinates=True``.
    a, xi, eta : float
        Hydrodynamic parameters (sphere radius, splitting parameter, viscosity).
    rcut : float
        Real-space cutoff radius (in real units).
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
        ``init_fn(positions_frac, **kwargs) -> RealSpaceState`` allocates the
        neighbor list and lattice indices for the provided fractional positions.
        ``apply_fn(state, positions_frac, forces, **kwargs)`` returns the real-space
        velocity together with the updated state.
    """

    if rcut <= 0.0:
        raise ValueError("rcut must be positive.")

    if len(space_fns) < 2:
        raise ValueError("space_fns must contain at least displacement and shift functions.")
    displacement_fn, _ = space_fns[:2]
    box_fn = space_fns[2] if len(space_fns) > 2 else None

    # When using fractional coordinates the neighbor list requires a box
    # parameter at construction time. We don't know the physical box yet
    # here (it will be threaded in to `allocate`/`update` as kwargs), so
    # pass a neutral scalar placeholder (1.0). The real box_matrix is
    # provided later via `neighbor_kwargs.setdefault("box", box_matrix)`.
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box=1.0,
        r_cutoff=rcut,
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        mask_self=False,
        fractional_coordinates=True,
        format=neighbor_format,
    )

    rcut2 = float(rcut * rcut)
    core_fn = _build_mr_core(a, xi, eta, rcut2, neighbor_format)

    def _compute_lattice_indices(box_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        box_np = np.asarray(box_matrix, dtype=np.float64)
        dim = box_np.shape[0]
        if lattice_extent is None:
            svals = np.linalg.svd(box_np, compute_uv=False)
            sigma_min = float(np.min(svals))
            safe_sigma = max(sigma_min, 1e-12)

            # Ratio of cutoff to smallest box scale.
            ratio = float(rcut) / safe_sigma

            # Base extent from geometric ratio.
            base_extent = int(np.ceil(ratio)) if ratio > 0.0 else 0

            # For rcut much smaller than the box, a single shell of images is
            # sufficient in typical periodic setups. Cap to at most one shell
            # unless rcut is comparable to the box size.
            if ratio < 1.0:
                extent_val = min(base_extent, 1)
            else:
                extent_val = base_extent
        else:
            extent_val = int(lattice_extent)

        lattice_np, zero_idx = _generate_lattice_hypercube(dim, extent_val)
        lattice = jnp.asarray(lattice_np, dtype=jnp.int32)
        return lattice, zero_idx

    def init_fn(positions_frac, *, extra_capacity_override=None, **kwargs):
        positions_frac = jnp.asarray(positions_frac)
        dim = int(positions_frac.shape[1])
        box_matrix = _current_box_matrix(displacement_fn, box_fn, dim, **kwargs)
        lattice_indices, zero_idx = _compute_lattice_indices(box_matrix)
        cap_value = extra_capacity if extra_capacity_override is None else extra_capacity_override
        neighbor_kwargs = dict(kwargs)
        neighbor_kwargs.setdefault("box", box_matrix)
        neighbors = neighbor_fn.allocate(positions_frac, int(cap_value), **neighbor_kwargs)
        return RealSpaceState(
            neighbors=neighbors,
            lattice_indices=lattice_indices,
            zero_image_index=zero_idx,
            box_matrix=box_matrix,
            core_fn=core_fn,
        )

    def apply_fn(state: RealSpaceState, positions_frac, forces, **kwargs):
        positions_frac = jnp.asarray(positions_frac)
        forces = jnp.asarray(forces)

        if positions_frac.shape != forces.shape:
            raise ValueError("positions and forces must have the same shape.")

        dim = int(positions_frac.shape[1])
        neighbor_override = kwargs.pop("neighbor", None)
        lattice_override = kwargs.pop("lattice_indices", None)
        zero_override = kwargs.pop("zero_image_index", None)
        box_override = kwargs.pop("box_matrix", None)

        if box_override is None:
            box_matrix = _current_box_matrix(displacement_fn, box_fn, dim, **kwargs)
        else:
            box_matrix = jnp.asarray(box_override, dtype=REAL_DTYPE)

        if lattice_override is not None:
            lattice_indices = jnp.asarray(lattice_override, dtype=jnp.int32)
            zero_idx = int(zero_override) if zero_override is not None else state.zero_image_index
        else:
            lattice_indices = state.lattice_indices
            zero_idx = state.zero_image_index

        neighbor_kwargs = dict(kwargs)
        neighbor_kwargs.setdefault("box", box_matrix)
        if neighbor_override is None:
            if state.neighbors is None:
                raise ValueError("RealSpaceState.neighbors is None; provide a neighbor list via 'neighbor'.")
            # Use the built-in NeighborList.update() method
            updated = state.neighbors.update(positions_frac, **neighbor_kwargs)

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
                    neighbors = neighbor_fn.allocate(positions_frac, int(extra_capacity), **neighbor_kwargs)
                else:
                    neighbors = updated
        else:
            neighbors = neighbor_override

        mask = partition.neighbor_list_mask(neighbors)

        # Forces are provided in real coordinates.
        forces_real = jnp.asarray(forces, dtype=REAL_DTYPE)

        # Apply core in real units
        velocities_real = core_fn(
            positions_frac,
            forces_real,
            neighbors.idx,
            mask,
            box_matrix,
            lattice_indices,
            zero_idx,
        )

        next_state = RealSpaceState(
            neighbors=neighbors,
            lattice_indices=lattice_indices,
            zero_image_index=zero_idx,
            box_matrix=box_matrix,
            core_fn=core_fn,
        )
        return velocities_real, next_state

    return init_fn, apply_fn


@jax.jit
def mr_matvec(state: RealSpaceState,
              positions_frac: jnp.ndarray,
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
    positions_frac : (N,3) array
        Current fractional particle positions.
    vec : (N,3) array
        Vector to which the mobility is applied (force-like, in fractional basis).

    Returns
    -------
    (N,3) array
        Result of applying ``M^(r)`` to ``vec`` in real coordinates.
    """
    if state.core_fn is None:
        raise ValueError("RealSpaceState is missing core_fn; build_Mr_apply must be used.")

    positions_frac = jnp.asarray(positions_frac, dtype=REAL_DTYPE)
    vec = jnp.asarray(vec, dtype=REAL_DTYPE)
    neighbors = neighbor if neighbor is not None else state.neighbors
    if neighbors is None:
        raise ValueError("Real-space state is missing a neighbor list; provide one via the 'neighbor' argument.")
    mask = partition.neighbor_list_mask(neighbors)

    # Forces are provided in real coordinates.
    forces_real = jnp.asarray(vec, dtype=REAL_DTYPE)

    # Apply real-space mobility in real units
    v_real = state.core_fn(
        positions_frac,
        forces_real,
        neighbors.idx,
        mask,
        state.box_matrix,
        state.lattice_indices,
        state.zero_image_index,
    )

    return v_real
