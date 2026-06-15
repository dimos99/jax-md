"""Real-space grand RPY mobility with force and traceless couplet inputs.

Extends the deterministic real-space operator of ``rpy_real_det`` to the grand
mobility ``[U, D] = M^r [F, C]``, adding the UC (couplet -> velocity), DF
(force -> velocity gradient), and DC (couplet -> velocity gradient) blocks on
top of the existing UF pair kernel.  The radial scalars (G1, G2, K1, K2, K3)
come from ``rpy_real_det_dipole_helpers``; the tensor contraction here was
validated against the quadrature ground truth in
``tests/rpy_quadrature_reference.py`` and, in convention-free physical
variables (F, L, S) -> (U, W, E), against the FSD reference implementation
(machine precision; see ``tests/rpy_stresslet_test.py``).

Conventions (shared with ``rpy_moments`` / the wave-space side):

  * ``rij = x_sender - x_receiver`` (+ lattice image), so ``rhat`` points
    receiver -> sender.  The literature tensors use the opposite direction;
    odd-in-rhat blocks (UC/DF) absorb the sign, even blocks (UF/DC) do not.
  * ``D_ij = du_i/dx_j`` (Faxen-filtered), traceless; couplets are projected
    traceless on entry.
  * Neighbor-list bookkeeping (worst-case shear allocation, live-box lattice
    refresh) is shared with the force-only operator via
    ``rpy_real_det._resolve_apply_bookkeeping``.
"""

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax import ops
from jax_md import partition, space
from jax_md.hydro.rpy_moments import traceless
from jax_md.hydro.rpy_real_det import (
    RealSpaceState,
    _positions_to_real,
    _resolve_apply_bookkeeping,
)
from jax_md.hydro.rpy_real_det_helpers import (
    REAL_DTYPE,
    PAIR_EPS_FRACTION_OF_DIAMETER,
    F1F2_closed_form,
    Mr_self,
    current_box_matrix,
)
from jax_md.hydro.rpy_real_det_dipole_helpers import (
    G1G2_closed_form,
    K1K2K3_closed_form,
    Mr_self_dipole,
)
from jax_md.hydro.rpy_real_lattice_helpers import (
    RealSpaceMode,
    _validate_real_space_mode,
    _neighbor_box_from_matrix,
    _is_traced_value,
    _box_fn_supports_shear_kwargs,
    _worst_case_shear_neighbor_box,
    _compute_lattice_indices,
    _select_real_space_core,
)


I3 = jnp.eye(3, dtype=REAL_DTYPE)

# M_DF,ijm(r) = -M_UC,mij(r): the DF block is the adjoint of UC with a sign
# flip inherited from the -i/+i pair in the wave-space moment maps (Fiore
# Eqs. 4.21/4.22).  Not a free choice -- pinned externally by the quadrature
# pair-tensor test and the FSD physical-map comparison, and confirmed
# internally by grand-mobility symmetry.
DF_ADJOINT_SIGN = -1.0


def _build_pair_contrib_grand_fn(a: float, xi: float, eta: float):
  """Build pairwise grand-mobility contractions for arbitrary edge shapes."""
  self_factor = Mr_self(a, xi)
  self_dipole_factor = Mr_self_dipole(a, xi)
  pair_eps2_scalar = float(((2.0 * a) * PAIR_EPS_FRACTION_OF_DIAMETER) ** 2)
  prefactor_uf_scalar = 1.0 / (6.0 * np.pi * eta * a)
  # The G/K scalars (FSD-pinned) carry their a-dimensions internally, so all
  # dipole blocks share the bare 1/(6 pi eta) prefactor (see
  # rpy_real_det_dipole_helpers docstring).
  prefactor_uc_scalar = 1.0 / (6.0 * np.pi * eta)
  prefactor_dc_scalar = 1.0 / (6.0 * np.pi * eta)

  def pair_contrib(rij, r2, mask, forces, couplets, *, prefactor_uf,
                   prefactor_uc, prefactor_dc, pair_eps2, self_dipole):
    """Return pair contributions ``(dU, dD)`` from sender moments."""
    safe_r = jnp.where(
        mask,
        jnp.sqrt(jnp.maximum(r2, pair_eps2)),
        jnp.ones_like(r2, dtype=rij.dtype))
    rhat = jnp.where(mask[..., None], rij / safe_r[..., None], 0.0)
    eye = I3.astype(rij.dtype)

    F1, F2 = F1F2_closed_form(safe_r, a, xi)
    G1, G2 = G1G2_closed_form(safe_r, a, xi)
    K1, K2, K3 = K1K2K3_closed_form(safe_r, a, xi)
    F1 = jnp.where(mask, F1, 0.0)
    F2 = jnp.where(mask, F2, 0.0)
    G1 = jnp.where(mask, G1, 0.0)
    G2 = jnp.where(mask, G2, 0.0)
    K1 = jnp.where(mask, K1, 0.0)
    K2 = jnp.where(mask, K2, 0.0)
    K3 = jnp.where(mask, K3, 0.0)

    Fr = jnp.einsum("...i,...i->...", forces, rhat)
    rr = rhat[..., :, None] * rhat[..., None, :]

    # UF block: M_UF F = F1 F + (F2 - F1)(F.rhat) rhat (Fiore & Swan Eq. 25).
    # F_iso is the trace-isotropic limit used as the r -> 0 fallback below.
    u_force = prefactor_uf * (
        F1[..., None] * forces +
        (F2 - F1)[..., None] * Fr[..., None] * rhat)
    F_iso = (2.0 * F1 + F2) / 3.0
    u_force_iso = prefactor_uf * F_iso[..., None] * forces

    Cr = jnp.einsum("...mn,...n->...m", couplets, rhat)
    rC = jnp.einsum("...m,...mn->...n", rhat, couplets)
    rCr = jnp.einsum("...m,...m->...", rhat, Cr)

    # rhat here points receiver -> sender (rij = xj - xi); the quadrature
    # ground-truth tensors use r = x_receiver - x_sender = -rhat, which flips
    # the sign of every odd-in-rhat (UC/DF) term and leaves DC unchanged.
    # UC block: M_UC C = velocity from a neighbour's couplet (G1, G2 scalars).
    u_couplet = prefactor_uc * (
        0.5 * G1[..., None] * (Cr - rCr[..., None] * rhat) -
        0.5 * G2[..., None] * (rC - 4.0 * rCr[..., None] * rhat))

    # DF block: M_DF F = velocity-gradient sourced by a neighbour's force.
    # Adjoint of UC (same G1, G2) up to DF_ADJOINT_SIGN, so it reuses
    # prefactor_uc.
    F_i_r_j = forces[..., :, None] * rhat[..., None, :]
    r_i_F_j = rhat[..., :, None] * forces[..., None, :]
    d_force = DF_ADJOINT_SIGN * prefactor_uc * (
        0.5 * G1[..., None, None] * (F_i_r_j - Fr[..., None, None] * rr) -
        0.5 * G2[..., None, None] * (
            r_i_F_j + Fr[..., None, None] * eye -
            4.0 * Fr[..., None, None] * rr))

    # DC block: M_DC C = velocity-gradient sourced by a neighbour's couplet
    # (K1, K2, K3 scalars). Even in rhat, so no sign flip relative to the
    # quadrature reference.
    C_T = jnp.swapaxes(couplets, -1, -2)
    r_i_Cr_j = rhat[..., :, None] * Cr[..., None, :]
    Cr_i_r_j = Cr[..., :, None] * rhat[..., None, :]
    r_i_rC_j = rhat[..., :, None] * rC[..., None, :]
    rC_i_r_j = rC[..., :, None] * rhat[..., None, :]
    d_couplet = prefactor_dc * (
        K1[..., None, None] * (C_T - 4.0 * couplets) +
        K2[..., None, None] * (Cr_i_r_j - rCr[..., None, None] * rr) +
        K3[..., None, None] * (
            rCr[..., None, None] * eye + rC_i_r_j + r_i_Cr_j + r_i_rC_j -
            6.0 * rCr[..., None, None] * rr - couplets))

    # Near-coincident pairs (r below the regularization floor) would divide by
    # ~0 in the G/K scalars; replace those entries with the r -> 0 limits: the
    # isotropic UF mobility for velocity, zero UC, zero DF, and the self-dipole
    # DC structure (matching the self term accumulated in the cores).
    tiny_pairs = mask & (r2 <= pair_eps2)
    zeros_u = jnp.zeros_like(u_couplet)
    d_fallback = prefactor_dc * self_dipole * (C_T - 4.0 * couplets)

    u_force = jnp.where(tiny_pairs[..., None], u_force_iso, u_force)
    u_couplet = jnp.where(tiny_pairs[..., None], zeros_u, u_couplet)
    d_force = jnp.where(tiny_pairs[..., None, None], jnp.zeros_like(d_force), d_force)
    d_couplet = jnp.where(tiny_pairs[..., None, None], d_fallback, d_couplet)
    return u_force + u_couplet, d_force + d_couplet

  prefactors = (prefactor_uf_scalar, prefactor_uc_scalar, prefactor_dc_scalar)
  return self_factor, self_dipole_factor, pair_eps2_scalar, prefactors, pair_contrib


def _normalize_edges(neighbor_idx, neighbor_mask, neighbor_format, n_particles):
  """Normalize Dense/Sparse neighbor-list formats to flat receiver/sender arrays."""
  uses_sparse = partition.is_sparse(neighbor_format)
  if not uses_sparse:
    if neighbor_idx.ndim == 1:
      neighbor_idx = neighbor_idx[:, None]
    if neighbor_mask.ndim == 1:
      neighbor_mask = neighbor_mask[:, None]
    max_neighbors = neighbor_idx.shape[1]
    if max_neighbors == 0:
      receivers = jnp.zeros((0,), dtype=jnp.int32)
      senders = jnp.zeros((0,), dtype=jnp.int32)
      flat_mask = jnp.zeros((0,), dtype=bool)
    else:
      neighbor_idx_masked = jnp.where(neighbor_mask, neighbor_idx, 0)
      idx_i = jnp.arange(n_particles, dtype=jnp.int32)
      receivers = jnp.repeat(idx_i, max_neighbors)
      senders = neighbor_idx_masked.ravel()
      flat_mask = neighbor_mask.ravel()
  else:
    if neighbor_idx.ndim == 1:
      neighbor_idx = neighbor_idx[None, :]
    if neighbor_mask.ndim == 0:
      neighbor_mask = jnp.broadcast_to(neighbor_mask, neighbor_idx.shape[1:])
    capacity = neighbor_idx.shape[1]
    if capacity == 0:
      receivers = jnp.zeros((0,), dtype=jnp.int32)
      senders = jnp.zeros((0,), dtype=jnp.int32)
      flat_mask = jnp.zeros((0,), dtype=bool)
    else:
      receivers = jnp.where(neighbor_mask, neighbor_idx[0], 0)
      senders = jnp.where(neighbor_mask, neighbor_idx[1], 0)
      flat_mask = neighbor_mask
  return receivers, senders, flat_mask


def _build_mr_core_lattice_grand(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
    fractional_coordinates: bool,
) -> Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]:
  """Build the lattice-image grand real-space kernel."""
  (self_factor, self_dipole_factor, pair_eps2_scalar, prefactor_scalars,
   pair_contrib) = _build_pair_contrib_grand_fn(a, xi, eta)
  include_ordered_backflow = neighbor_format is partition.NeighborListFormat.OrderedSparse

  @jax.jit
  def core(positions, forces, couplets, neighbor_idx, neighbor_mask, box_matrix,
           lattice_indices, zero_image_index):
    positions = jnp.asarray(positions)
    forces = jnp.asarray(forces)
    couplets = jnp.asarray(couplets)
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
    box_matrix = jnp.asarray(box_matrix)
    lattice_indices = jnp.asarray(lattice_indices, dtype=jnp.int32)
    zero_image_index = jnp.int32(zero_image_index)

    x_real = _positions_to_real(positions, box_matrix, fractional_coordinates)
    lattice_vecs = lattice_indices @ box_matrix.T
    n_particles = x_real.shape[0]
    n_images = lattice_vecs.shape[0]
    dtype = x_real.dtype
    prefactor_uf = jnp.asarray(prefactor_scalars[0], dtype=dtype)
    prefactor_uc = jnp.asarray(prefactor_scalars[1], dtype=dtype)
    prefactor_dc = jnp.asarray(prefactor_scalars[2], dtype=dtype)
    self_term = prefactor_uf * jnp.asarray(self_factor, dtype=dtype)
    self_dipole = jnp.asarray(self_dipole_factor, dtype=dtype)
    pair_eps2 = jnp.asarray(pair_eps2_scalar, dtype=dtype)

    velocities_init = self_term * forces
    gradients_init = prefactor_dc * self_dipole * (
        jnp.swapaxes(couplets, -1, -2) - 4.0 * couplets)

    if n_images == 0:
      return velocities_init, gradients_init

    receivers, senders, flat_mask = _normalize_edges(
        neighbor_idx, neighbor_mask, neighbor_format, n_particles)
    capacity = flat_mask.shape[0]
    if capacity == 0:
      return velocities_init, gradients_init

    zero_mask = (jnp.arange(n_images, dtype=jnp.int32) == zero_image_index)[None, :]
    lattice = lattice_vecs[None, :, :]

    if include_ordered_backflow:
      # OrderedSparse keeps only i < j edges, so the (i, i) self edges that
      # Dense uses to accumulate own-periodic-image contributions (relevant
      # when rcut exceeds the box) are absent.  Add them explicitly: every
      # particle receives from its own images at rij = lattice (primary image
      # excluded).  This matters only for the even-in-rhat blocks (UF/DC);
      # opposite images cancel for UC/DF.
      rij_self = jnp.broadcast_to(lattice, (n_particles, n_images, 3))
      r2_self = jnp.sum(rij_self * rij_self, axis=-1)
      mask_self_img = (~zero_mask) & (r2_self < rcut2)
      contrib_u_self, contrib_d_self = pair_contrib(
          rij_self, r2_self, mask_self_img, forces[:, None, :],
          couplets[:, None, :, :],
          prefactor_uf=prefactor_uf, prefactor_uc=prefactor_uc,
          prefactor_dc=prefactor_dc, pair_eps2=pair_eps2,
          self_dipole=self_dipole)
      velocities_init = velocities_init + contrib_u_self.sum(axis=1)
      gradients_init = gradients_init + contrib_d_self.sum(axis=1)

    def _accumulate_batch(carry, receivers_batch, senders_batch, edge_mask_batch):
      velocities, gradients = carry
      xi_vec = x_real[receivers_batch][:, None, :]
      xj = x_real[senders_batch][:, None, :]
      rij = xj - xi_vec + lattice
      r2 = jnp.sum(rij * rij, axis=-1)
      within_rcut = r2 < rcut2
      is_self_edge = (receivers_batch == senders_batch)[:, None]
      primary_self = is_self_edge & zero_mask
      mask_pairs = edge_mask_batch[:, None] & (~primary_self) & within_rcut

      forces_senders = forces[senders_batch][:, None, :]
      couplets_senders = couplets[senders_batch][:, None, :, :]
      contrib_u, contrib_d = pair_contrib(
          rij, r2, mask_pairs, forces_senders, couplets_senders,
          prefactor_uf=prefactor_uf, prefactor_uc=prefactor_uc,
          prefactor_dc=prefactor_dc, pair_eps2=pair_eps2,
          self_dipole=self_dipole)
      velocities = velocities + ops.segment_sum(contrib_u.sum(axis=1), receivers_batch, n_particles)
      gradients = gradients + ops.segment_sum(contrib_d.sum(axis=1), receivers_batch, n_particles)

      if include_ordered_backflow:
        forces_receivers = forces[receivers_batch][:, None, :]
        couplets_receivers = couplets[receivers_batch][:, None, :, :]
        contrib_u_rev, contrib_d_rev = pair_contrib(
            -rij, r2, mask_pairs, forces_receivers, couplets_receivers,
            prefactor_uf=prefactor_uf, prefactor_uc=prefactor_uc,
            prefactor_dc=prefactor_dc, pair_eps2=pair_eps2,
            self_dipole=self_dipole)
        velocities = velocities + ops.segment_sum(contrib_u_rev.sum(axis=1), senders_batch, n_particles)
        gradients = gradients + ops.segment_sum(contrib_d_rev.sum(axis=1), senders_batch, n_particles)
      return velocities, gradients

    # Each edge is evaluated against every lattice image at once, so the peak
    # intermediate is (capacity x n_images x 3x3). Cap that product to bound
    # memory: small problems run in a single batch, larger ones are scanned in
    # edge-chunks of `chunk_size` (chosen so chunk_size * n_images stays under
    # the limit). The limit is a memory knob only -- results are identical.
    pair_image_limit = 8_000_000
    pair_image_work = capacity * n_images
    if pair_image_work <= pair_image_limit:
      return _accumulate_batch(
          (velocities_init, gradients_init), receivers, senders, flat_mask)

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

    def _scan_body(carry, chunk):
      receivers_chunk, senders_chunk, mask_chunk = chunk
      return _accumulate_batch(carry, receivers_chunk, senders_chunk, mask_chunk), None

    (velocities, gradients), _ = jax.lax.scan(
        _scan_body,
        (velocities_init, gradients_init),
        (receivers_chunks, senders_chunks, mask_chunks),
    )
    return velocities, gradients

  return core


def _build_mr_core_min_image_grand(
    a: float,
    xi: float,
    eta: float,
    rcut2: float,
    neighbor_format: partition.NeighborListFormat,
    fractional_coordinates: bool,
) -> Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]:
  """Build the minimum-image grand real-space kernel."""
  (self_factor, self_dipole_factor, pair_eps2_scalar, prefactor_scalars,
   pair_contrib) = _build_pair_contrib_grand_fn(a, xi, eta)
  include_ordered_backflow = neighbor_format is partition.NeighborListFormat.OrderedSparse

  @jax.jit
  def core(positions, forces, couplets, neighbor_idx, neighbor_mask, box_matrix,
           lattice_indices, zero_image_index):
    del lattice_indices, zero_image_index
    positions = jnp.asarray(positions, dtype=REAL_DTYPE)
    forces = jnp.asarray(forces, dtype=REAL_DTYPE)
    couplets = jnp.asarray(couplets, dtype=REAL_DTYPE)
    neighbor_idx = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    neighbor_mask = jnp.asarray(neighbor_mask, dtype=bool)
    box_matrix = jnp.asarray(box_matrix, dtype=REAL_DTYPE)
    dtype = forces.dtype

    prefactor_uf = jnp.asarray(prefactor_scalars[0], dtype=dtype)
    prefactor_uc = jnp.asarray(prefactor_scalars[1], dtype=dtype)
    prefactor_dc = jnp.asarray(prefactor_scalars[2], dtype=dtype)
    self_term = prefactor_uf * jnp.asarray(self_factor, dtype=dtype)
    self_dipole = jnp.asarray(self_dipole_factor, dtype=dtype)
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

    n_particles = positions_frac.shape[0]
    velocities = self_term * forces
    gradients = prefactor_dc * self_dipole * (
        jnp.swapaxes(couplets, -1, -2) - 4.0 * couplets)

    receivers, senders, flat_mask = _normalize_edges(
        neighbor_idx, neighbor_mask, neighbor_format, n_particles)
    capacity = flat_mask.shape[0]
    if capacity == 0:
      return velocities, gradients

    delta_frac = _wrapped_frac_delta(positions_frac[senders] - positions_frac[receivers])
    rij = _to_real(delta_frac)
    r2 = jnp.sum(rij * rij, axis=-1)
    within_rcut = r2 < rcut2
    is_self_edge = receivers == senders
    mask_pairs = flat_mask & (~is_self_edge) & within_rcut

    contrib_u, contrib_d = pair_contrib(
        rij, r2, mask_pairs, forces[senders], couplets[senders],
        prefactor_uf=prefactor_uf, prefactor_uc=prefactor_uc,
        prefactor_dc=prefactor_dc, pair_eps2=pair_eps2,
        self_dipole=self_dipole)
    velocities = velocities + ops.segment_sum(contrib_u, receivers, n_particles)
    gradients = gradients + ops.segment_sum(contrib_d, receivers, n_particles)

    if include_ordered_backflow:
      contrib_u_rev, contrib_d_rev = pair_contrib(
          -rij, r2, mask_pairs, forces[receivers], couplets[receivers],
          prefactor_uf=prefactor_uf, prefactor_uc=prefactor_uc,
          prefactor_dc=prefactor_dc, pair_eps2=pair_eps2,
          self_dipole=self_dipole)
      velocities = velocities + ops.segment_sum(contrib_u_rev, senders, n_particles)
      gradients = gradients + ops.segment_sum(contrib_d_rev, senders, n_particles)

    return velocities, gradients

  return core


def build_Mr_grand_apply(
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
  """Construct the neighbor-list-backed real-space grand mobility operator.

  The grand mobility couples particle forces and traceless couplets to
  velocities and traceless velocity gradients via the real-space coupling
  tensors M_UF, M_UC, M_DF, M_DC (Fiore & Swan 2018, Eqs. 25-27; scalar
  radial functions F1/F2, G1/G2, K1/K2/K3 from Appendix A).

  Returns ``(init_fn, apply_fn)`` mirroring ``build_Mr_apply``: ``init_fn``
  allocates the neighbor-list-backed ``RealSpaceState`` and ``apply_fn``
  evaluates ``(F, C) -> (U, D)`` at a configuration, refreshing the state for
  the live (possibly sheared) box.
  """
  if rcut <= 0.0:
    raise ValueError("rcut must be positive.")
  if lattice_extra < 0.0:
    raise ValueError("lattice_extra must be non-negative.")
  if lattice_extent is not None and int(lattice_extent) < 0:
    raise ValueError("lattice_extent must be non-negative when provided.")
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
  core_lattice = _build_mr_core_lattice_grand(
      a, xi, eta, rcut2, neighbor_format, fractional_coordinates)
  core_min_image = _build_mr_core_min_image_grand(
      a, xi, eta, rcut2, neighbor_format, fractional_coordinates)

  def init_fn(positions, *, extra_capacity_override=None, **kwargs):
    positions = jnp.asarray(positions)
    dim = int(positions.shape[1])
    box_matrix = current_box_matrix(
        displacement_fn, box_fn, dim, fractional_coordinates=fractional_coordinates, **kwargs)
    core_fn_init, _ = _select_real_space_core(
        box_matrix,
        mode=mode,
        core_lattice=core_lattice,
        core_min_image=core_min_image,
        rcut=float(rcut),
    )
    lattice_indices, zero_idx = _compute_lattice_indices(
        box_matrix,
        rcut=float(rcut),
        lattice_extent=lattice_extent,
        lattice_extra=lattice_extra,
        warn_stacklevel=5,
    )
    cap_value = extra_capacity if extra_capacity_override is None else extra_capacity_override
    neighbor_kwargs = dict(kwargs)
    neighbor_box = _neighbor_box_from_matrix(box_matrix, fractional_coordinates)
    if box_fn is not None and fractional_coordinates and neighbor_box is not None:
      if (not _is_traced_value(neighbor_box) and
          _box_fn_supports_shear_kwargs(box_fn, dim)):
        worst_np = _worst_case_shear_neighbor_box(
            box_fn, dim, np.asarray(neighbor_box, dtype=np.float64))
        neighbor_box = jnp.asarray(worst_np, dtype=neighbor_box.dtype)
      neighbor_kwargs["box"] = neighbor_box
    elif neighbor_box is not None:
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
               couplets,
               *,
               neighbor: Optional[partition.NeighborList] = None,
               lattice_indices: Optional[jnp.ndarray] = None,
               zero_image_index: Optional[int] = None,
               box_matrix: Optional[jnp.ndarray] = None,
               **kwargs):
    positions = jnp.asarray(positions)
    forces = jnp.asarray(forces, dtype=REAL_DTYPE)
    couplets = traceless(jnp.asarray(couplets, dtype=REAL_DTYPE))
    if positions.shape != forces.shape:
      raise ValueError("positions and forces must have the same shape.")
    if couplets.shape != positions.shape[:-1] + (3, 3):
      raise ValueError("couplets must have shape (N, 3, 3).")

    (box_matrix_local, core_fn_selected, lattice_indices_local, zero_idx,
     neighbors, mask) = _resolve_apply_bookkeeping(
        state=state,
        positions=positions,
        displacement_fn=displacement_fn,
        box_fn=box_fn,
        fractional_coordinates=fractional_coordinates,
        rcut=float(rcut),
        lattice_extent=lattice_extent,
        lattice_extra=lattice_extra,
        box_jump_threshold=float(box_jump_threshold),
        extra_capacity=extra_capacity,
        neighbor_fn=neighbor_fn,
        core_lattice=core_lattice,
        core_min_image=core_min_image,
        neighbor=neighbor,
        lattice_indices=lattice_indices,
        zero_image_index=zero_image_index,
        box_matrix=box_matrix,
        **kwargs,
    )

    velocities_real, gradients_real = core_fn_selected(
        positions,
        forces,
        couplets,
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
    return (velocities_real, traceless(gradients_real)), next_state

  def refresh_fn(state: RealSpaceState, positions, **kwargs) -> RealSpaceState:
    """Rebind the state at new positions without evaluating the grand kernel.

    Same neighbor-list / lattice / box bookkeeping as ``apply_fn``, minus the
    core evaluation -- used by the constrained Brownian stepper to bind the
    slip sampler at the start-of-step configuration at zero kernel cost.
    """
    positions = jnp.asarray(positions)
    (box_matrix_local, core_fn_selected, lattice_indices_local, zero_idx,
     neighbors, _) = _resolve_apply_bookkeeping(
        state=state,
        positions=positions,
        displacement_fn=displacement_fn,
        box_fn=box_fn,
        fractional_coordinates=fractional_coordinates,
        rcut=float(rcut),
        lattice_extent=lattice_extent,
        lattice_extra=lattice_extra,
        box_jump_threshold=float(box_jump_threshold),
        extra_capacity=extra_capacity,
        neighbor_fn=neighbor_fn,
        core_lattice=core_lattice,
        core_min_image=core_min_image,
        **kwargs,
    )
    return RealSpaceState(
        neighbors=neighbors, # type: ignore
        lattice_indices=lattice_indices_local, # type: ignore
        zero_image_index=zero_idx, # type: ignore
        box_matrix=box_matrix_local, # type: ignore
        fractional_coordinates=fractional_coordinates, # type: ignore
        core_fn=core_fn_selected, # type: ignore
    )

  apply_fn.refresh = refresh_fn
  return init_fn, apply_fn


def mr_grand_matvec(state: RealSpaceState,
                    positions: jnp.ndarray,
                    forces: jnp.ndarray,
                    couplets: Optional[jnp.ndarray] = None,
                    *,
                    neighbor: Optional[partition.NeighborList] = None):
  """Apply the real-space grand mobility using an existing state."""
  if state.core_fn is None:
    raise ValueError("RealSpaceState is missing core_fn; build_Mr_grand_apply must be used.")
  positions = jnp.asarray(positions, dtype=REAL_DTYPE)
  forces = jnp.asarray(forces, dtype=REAL_DTYPE)
  if couplets is None:
    couplets = jnp.zeros(positions.shape[:-1] + (3, 3), dtype=REAL_DTYPE)
  couplets = traceless(jnp.asarray(couplets, dtype=REAL_DTYPE))
  neighbors = neighbor if neighbor is not None else state.neighbors
  if neighbors is None:
    raise ValueError("Real-space state is missing a neighbor list; provide one via the 'neighbor' argument.")
  mask = partition.neighbor_list_mask(neighbors)
  velocities, gradients = state.core_fn(
      positions,
      forces,
      couplets,
      neighbors.idx,
      mask,
      state.box_matrix,
      state.lattice_indices,
      state.zero_image_index,
  )
  return velocities, traceless(gradients)
