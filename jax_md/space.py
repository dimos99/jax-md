# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spaces in which particles are simulated.

Spaces are pairs of functions containing:
  `displacement_fn(Ra, Rb, **kwargs)`:
    Computes displacements between pairs of particles. `Ra` and `Rb` should
    be ndarrays of shape `[spatial_dim]`. Returns an ndarray of shape `[spatial_dim]`.
    To compute the displacement over more than one particle at a time see the
    :meth:`map_product`, :meth:`map_bond`, and :meth:`map_neighbor` functions.
  `shift_fn(R, dR, **kwargs)`:
    Moves points at position `R` by an amount `dR`.

Spaces can accept keyword arguments allowing the space to be changed over the
course of a simulation. For an example of this use see :meth:`periodic_general`.

Although displacement functions are compute the displacement between two
points, it is often useful to compute displacements between multiple particles
in a vectorized fashion. To do this we provide three functions: `map_product`,
`map_bond`, and `map_neighbor`:
  map_product:
    Computes displacements between all pairs of points such that if
    `Ra` has shape `[n, spatial_dim]` and `Rb` has shape `[m, spatial_dim]` then the
    output has shape `[n, m, spatial_dim]`.
  map_bond:
    Computes displacements between all points in a list such that if
    `Ra` has shape `[n, spatial_dim]` and `Rb` has shape `[m, spatial_dim]` then the
    output has shape `[n, spatial_dim]`.
  map_neighbor:
    Computes displacements between points and all of their
    neighbors such that if `Ra` has shape `[n, spatial_dim]` and `Rb` has shape
    `[n, neighbors, spatial_dim]` then the output has shape
    `[n, neighbors, spatial_dim]`.
"""

from typing import Callable, Union, Tuple, Any, Optional

from jax.core import ShapedArray

from jax import eval_shape
from jax import vmap
from jax import custom_jvp

import jax

import jax.numpy as jnp

from jax_md.util import Array
from jax_md.util import f32
from jax_md.util import f64
from jax_md.util import safe_mask


# Types


DisplacementFn = Callable[[Array, Array], Array]
MetricFn = Callable[[Array, Array], float]
DisplacementOrMetricFn = Union[DisplacementFn, MetricFn]

ShiftFn = Callable[[Array, Array], Array]

Space = Tuple[DisplacementFn, ShiftFn]
Box = Array


# Exceptions


class UnexpectedBoxException(Exception):
  pass


# Primitive Spatial Transforms


def inverse(box: Box) -> Box:
  """Compute the inverse of an affine transformation."""
  if jnp.isscalar(box) or box.size == 1:
    return 1 / box
  elif box.ndim == 1:
    return 1 / box
  elif box.ndim == 2:
    return jnp.linalg.inv(box)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


def _get_free_indices(n: int) -> str:
  return ''.join([chr(ord('a') + i) for i in range(n)])


def raw_transform(box: Box, R: Array) -> Array:
  """Apply an affine transformation to positions.

  See `periodic_general` for a description of the semantics of `box`.

  Args:
    box: An affine transformation described in `periodic_general`.
    R: Array of positions. Should have  shape `(..., spatial_dimension)`.

  Returns:
    A transformed array positions of shape `(..., spatial_dimension)`.
  """
  if jnp.isscalar(box) or box.size == 1:
    return R * box
  elif box.ndim == 1:
    indices = _get_free_indices(R.ndim - 1) + 'i'
    return jnp.einsum(f'i,{indices}->{indices}', box, R)
  elif box.ndim == 2:
    free_indices = _get_free_indices(R.ndim - 1)
    left_indices = free_indices + 'j'
    right_indices = free_indices + 'i'
    return jnp.einsum(f'ij,{left_indices}->{right_indices}', box, R)
  raise ValueError(('Box must be either: a scalar, a vector, or a matrix. '
                    f'Found {box}.'))


@custom_jvp
def transform(box: Box, R: Array) -> Array:
  """Apply an affine transformation to positions.

  See `periodic_general` for a description of the semantics of `box`.

  Args:
    box: An affine transformation described in `periodic_general`.
    R: Array of positions. Should have  shape `(..., spatial_dimension)`.

  Returns:
    A transformed array positions of shape `(..., spatial_dimension)`.
  """
  return raw_transform(box, R)


@transform.defjvp
def transform_jvp(primals, tangents):
  box, R = primals
  dbox, dR = tangents
  return (transform(box, R), dR + transform(dbox, R))


def pairwise_displacement(Ra: Array, Rb: Array) -> Array:
  """Compute a matrix of pairwise displacements given two sets of positions.

  Args:
    Ra: Vector of positions; `ndarray(shape=[spatial_dim])`.
    Rb: Vector of positions; `ndarray(shape=[spatial_dim])`.

  Returns:
    Matrix of displacements; `ndarray(shape=[spatial_dim])`.
  """
  if len(Ra.shape) != 1:
    msg = (
      'Can only compute displacements between vectors. To compute '
      'displacements between sets of vectors use vmap or TODO.'
    )
    raise ValueError(msg)

  if Ra.shape != Rb.shape:
    msg = 'Can only compute displacement between vectors of equal dimension.'
    raise ValueError(msg)

  return Ra - Rb


def periodic_displacement(side: Box, dR: Array) -> Array:
  """Wraps displacement vectors into a hypercube.

  Args:
    side: Specification of hypercube size. Either,
      (a) float if all sides have equal length.
      (b) ndarray(spatial_dim) if sides have different lengths.
    dR: Matrix of displacements; `ndarray(shape=[..., spatial_dim])`.
  Returns:
    Matrix of wrapped displacements; `ndarray(shape=[..., spatial_dim])`.
  """
  return jnp.mod(dR + side * f32(0.5), side) - f32(0.5) * side


def square_distance(dR: Array) -> Array:
  """Computes square distances.

  Args:
    dR: Matrix of displacements; `ndarray(shape=[..., spatial_dim])`.
  Returns:
    Matrix of squared distances; `ndarray(shape=[...])`.
  """
  return jnp.sum(dR ** 2, axis=-1)


def distance(dR: Array) -> Array:
  """Computes distances.

  Args:
    dR: Matrix of displacements; `ndarray(shape=[..., spatial_dim])`.
  Returns:
    Matrix of distances; `ndarray(shape=[...])`.
  """
  dr = square_distance(dR)
  return safe_mask(dr > 0, jnp.sqrt, dr)


def periodic_shift(side: Box, R: Array, dR: Array) -> Array:
  """Shifts positions, wrapping them back within a periodic hypercube."""
  return jnp.mod(R + dR, side)


""" Spaces """


def free() -> Space:
  """Free boundary conditions."""
  def displacement_fn(Ra: Array, Rb: Array, perturbation: Optional[Array]=None,
                      **unused_kwargs) -> Array:
    dR = pairwise_displacement(Ra, Rb)
    if perturbation is not None:
      dR = raw_transform(perturbation, dR)
    return dR
  def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
    return R + dR
  return displacement_fn, shift_fn


def periodic(side: Box, wrapped: bool=True) -> Space:
  """Periodic boundary conditions on a hypercube of sidelength side.

  Args:
    side: Either a float or an ndarray of shape [spatial_dimension] specifying
      the size of each side of the periodic box.
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    `(displacement_fn, shift_fn)` tuple.
  """
  def displacement_fn(Ra: Array, Rb: Array,
                      perturbation: Optional[Array] = None,
                      **unused_kwargs) -> Array:
    if 'box' in unused_kwargs:
      raise UnexpectedBoxException(('`space.periodic` does not accept a box '
                                    'argument. Perhaps you meant to use '
                                    '`space.periodic_general`?'))
    dR = periodic_displacement(side, pairwise_displacement(Ra, Rb))
    if perturbation is not None:
      dR = raw_transform(perturbation, dR)
    return dR
  if wrapped:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      if 'box' in unused_kwargs:
        raise UnexpectedBoxException(('`space.periodic` does not accept a box '
                                      'argument. Perhaps you meant to use '
                                      '`space.periodic_general`?'))

      return periodic_shift(side, R, dR)
  else:
    def shift_fn(R: Array, dR: Array, **unused_kwargs) -> Array:
      if 'box' in unused_kwargs:
        raise UnexpectedBoxException(('`space.periodic` does not accept a box '
                                      'argument. Perhaps you meant to use '
                                      '`space.periodic_general`?'))
      return R + dR
  return displacement_fn, shift_fn


def periodic_general(box: Box,
                     fractional_coordinates: bool=True,
                     wrapped: bool=True) -> Space:
  """Periodic boundary conditions on a parallelepiped.

  This function defines a simulation on a parallelepiped, :math:`X`, formed by
  applying an affine transformation, :math:`T`, to the unit hypercube
  :math:`U = [0, 1]^d` along with periodic boundary conditions across all
  of the faces.

  Formally, the space is defined such that :math:`X = {Tu : u \in [0, 1]^d}`.

  The affine transformation, :math:`T`, can be specified in a number of different
  ways. For a parallelepiped that is: 1) a cube of side length :math:`L`, the affine
  transformation can simply be a scalar; 2) an orthorhombic unit cell can be
  specified by a vector `[Lx, Ly, Lz]` of lengths for each axis; 3) a general
  triclinic cell can be specified by an upper triangular matrix.

  There are a number of ways to parameterize a simulation on :math:`X`.
  `periodic_general` supports two parametrizations of :math:`X` that can be selected
  using the `fractional_coordinates` keyword argument.

    1) When `fractional_coordinates=True`, particle positions are stored in the
       unit cube, :math:`u\in U`. Here, the displacement function computes the
       displacement between :math:`x, y \in X` as :math:`d_X(x, y) = Td_U(u, v)` where
       :math:`d_U` is the displacement function on the unit cube, :math:`U`, :math:`x = Tu`, and
       :math:`v = Tv` with :math:`u, v \in U`. The derivative of the displacement function
       is defined so that derivatives live in :math:`X` (as opposed to being
       backpropagated to :math:`U`). The shift function, `shift_fn(R, dR)` is defined
       so that :math:`R` is expected to lie in :math:`U` while :math:`dR` should lie in :math:`X`. This
       combination enables code such as `shift_fn(R, force_fn(R))` to work as
       intended.

    2) When `fractional_coordinates=False`, particle positions are stored in
       the parallelepiped :math:`X`. Here, for :math:`x, y \in X`, the displacement function
       is defined as :math:`d_X(x, y) = Td_U(T^{-1}x, T^{-1}y)`. Since there is an
       extra multiplication by :math:`T^{-1}`, this parameterization is typically
       slower than `fractional_coordinates=False`. As in 1), the displacement
       function is defined to compute derivatives in :math:`X`. The shift function
       is defined so that :math:`R` and :math:`dR` should both lie in :math:`X`.

  Example:
  
  .. code-block:: python

     from jax import random
     side_length = 10.0
     disp_frac, shift_frac = periodic_general(side_length,
                                               fractional_coordinates=True)
     disp_real, shift_real = periodic_general(side_length,
                                               fractional_coordinates=False) 

     # Instantiate random positions in both parameterizations.
     R_frac = random.uniform(random.PRNGKey(0), (4, 3))
     R_real = side_length * R_frac

     # Make some shift vectors.
     dR = random.normal(random.PRNGKey(0), (4, 3))

     disp_real(R_real[0], R_real[1]) == disp_frac(R_frac[0], R_frac[1])
     transform(side_length, shift_frac(R_frac, 1.0)) == shift_real(R_real, 1.0)

  It is often desirable to deform a simulation cell either: using a finite
  deformation during a simulation, or using an infinitesimal deformation while
  computing elastic constants. To do this using fractional coordinates, we can
  supply a new affine transformation as `displacement_fn(Ra, Rb, box=new_box)`.
  When using real coordinates, we can specify positions in a space :math:`X` defined
  by an affine transformation :math:`T` and compute displacements in a deformed space
  :math:`X'` defined by an affine transformation :math:`T'`. This is done by writing
  `displacement_fn(Ra, Rb, new_box=new_box)`.

  There are a few caveats when using `periodic_general`. `periodic_general`
  uses the minimum image convention, and so it will fail for potentials whose
  cutoff is longer than the half of the side-length of the box. It will also
  fail to find the correct image when the box is too deformed. We hope to add a
  more robust box for small simulations soon (TODO) along with better error
  checking. In the meantime caution is recommended.

  Args:
    box: A `(spatial_dim, spatial_dim)` affine transformation.
    fractional_coordinates: A boolean specifying whether positions are stored
      in the parallelepiped or the unit cube.
    wrapped: A boolean specifying whether or not particle positions are
      remapped back into the box after each step
  Returns:
    `(displacement_fn, shift_fn)` tuple.
  """
  inv_box = inverse(box)

  def displacement_fn(Ra, Rb, perturbation=None, **kwargs):
    _box, _inv_box = box, inv_box

    if 'box' in kwargs:
      _box = kwargs['box']

      if not fractional_coordinates:
        _inv_box = inverse(_box)

    if 'new_box' in kwargs:
      _box = kwargs['new_box']

    if not fractional_coordinates:
      Ra = transform(_inv_box, Ra)
      Rb = transform(_inv_box, Rb)

    dR = periodic_displacement(f32(1.0), pairwise_displacement(Ra, Rb))
    dR = transform(_box, dR)

    if perturbation is not None:
      dR = raw_transform(perturbation, dR)

    return dR

  def u(R, dR):
    if wrapped:
      return periodic_shift(f32(1.0), R, dR)
    return R + dR

  def shift_fn(R, dR, **kwargs):
    if not fractional_coordinates and not wrapped:
      return R + dR

    _box, _inv_box = box, inv_box
    if 'box' in kwargs:
      _box = kwargs['box']
      _inv_box = inverse(_box)

    if 'new_box' in kwargs:
      _box = kwargs['new_box']

    dR = transform(_inv_box, dR)
    if not fractional_coordinates:
      R = transform(_inv_box, R)

    R = u(R, dR)

    if not fractional_coordinates:
      R = transform(_box, R)
    return R

  return displacement_fn, shift_fn

def shearing(box: Box,
             shear_fn: Optional[Callable[[Array], Array]] = None,
             fractional_coordinates: bool = True,
             remap: bool = False,
             keep_base_xy: bool = True,
             shear_fns: Optional[dict] = None):
  """
  Simple shear in one or more planes, each driven by its own schedule.

  Provide either a single function `shear_fn(t)` (applied to 'xy') or a dict
  `shear_fns` mapping plane names ('xy','xz','yz') to functions of time. For
  each plane `(i,j)` present (i<j), we set `H[i,j] = (keep? H[i,j] : 0) +
  gamma_ij(t) * H[j,j]`.

  Args:
    box: Base box (scalar, vector, or upper-triangular matrix).
    shear_fn: Optional function of time returning shear for 'xy' (convenience).
    fractional_coordinates: If True, store positions in fractional coords and
      return real displacements. If False, positions and displacements are real.
    remap: If True, wrap gamma into [-0.5, 0.5) to keep the box well-conditioned.
           Be careful when using this option: it changes the topology of the
           simulation, and requires remapping fractional coordinates when the
           box basis flips (see `remap_fractional_positions`).
    keep_base_xy: If True, preserve the base box's existing off-diagonals for all planes.
    shear_fns: Optional dict mapping 'xy','xz','yz' to functions of time.

  kwargs for displacement/shift/box_fn:
    - t: time (float). Used as input to `shear_fn` when `gamma` is not provided.
    - gamma: instantaneous shear (float). If provided, overrides `t`/`shear_fn`.
    - box: optional override for the current physical box (scalar/vector/matrix).

  Coordinate conventions:
    - If `fractional_coordinates=True`:
        positions live in [0,1)^d; displacement returns REAL dR; shift expects
        FRACTIONAL R and REAL dR and returns FRACTIONAL R (compatible with
        e.g. `shift(R, force(R))`).
    - If `fractional_coordinates=False`:
        positions and dR are REAL; internally we map to fractional for
        minimum-image, then map back.

  Note: See Tuckerman, ch. 13 for background on shear-periodic BCs.
  """

  # Helper Functions
  def _canonical_box(b: Box) -> Box:
    """
    Convert a box representation into a matrix.

    Supported inputs:
      scalar -> [[L, 0], [0, L]] or [[L, 0, 0], [0, L, 0], [0, 0, L]]
      vector -> [[Lx, 0], [0, Ly]] or [[Lx, 0, 0], [0, Ly, 0], [0, 0, Lz]]
      matrix -> as-is
    """
    b = jnp.asarray(b)
    if jnp.ndim(b) == 0:              # scalar -> isotropic matrix
      # Infer spatial dimension using the outer `box` argument where possible.
      # Use jnp.shape / jnp.ndim helpers which accept Python scalars safely.
      print("Warning: treating scalar 'box' as 3D by default; provide an explicit box to select 2D.")
      dim = 3
      return jnp.diag(jnp.ones((dim,), dtype=b.dtype) * b).astype(b.dtype)
    if jnp.ndim(b) == 1:              # orthorhombic -> diagonal matrix
      return jnp.diag(b).astype(b.dtype)
    if jnp.ndim(b) == 2:
      return b
    raise ValueError("Box must be a scalar, vector, or matrix.")
  
  def _gammas(**kwargs):
    """Return dict of instantaneous shear values per plane.

    Accepts any of:
      - gamma: scalar -> {'xy': gamma}; or dict {'xy': ..., 'xz': ..., 'yz': ...}
      - gamma_xy, gamma_xz, gamma_yz as separate scalars
      - t (time) together with provided shear_fn / shear_fns to compute values
    Applies wrapping to [-0.5, 0.5) per plane if `remap=True`.
    """
    planes = []
    dim = base.shape[0]
    if dim >= 2:
      planes.append('xy')
    if dim >= 3:
      planes.extend(['xz', 'yz'])

    g = {}
    if 'gamma' in kwargs:
      val = kwargs['gamma']
      if isinstance(val, dict):
        for k, v in val.items():
          if k in planes:
            g[k] = f32(v)
      else:
        # scalar provided, apply to xy only for compatibility
        if 'xy' in planes:
          g['xy'] = f32(val)
    # Individual overrides
    for k in ['xy', 'xz', 'yz']:
      key = f'gamma_{k}'
      if key in kwargs and k in planes:
        g[k] = f32(kwargs[key])

    # If still missing components and time provided, use functions
    missing = [k for k in planes if k not in g]
    if missing:
      if 't' not in kwargs:
        raise ValueError("Either gamma/gamma_* or t must be provided.")
      t = f32(kwargs['t'])
      fn_map = {}
      if shear_fns is not None:
        fn_map.update(shear_fns)
      if shear_fn is not None:
        # convenience: default xy if not present in map
        fn_map.setdefault('xy', shear_fn)
      for k in missing:
        if k in fn_map and callable(fn_map[k]):
          g[k] = f32(fn_map[k](t))
        else:
          g[k] = f32(0.0)

    if remap:
      g = {k: (v - jnp.floor(v + f32(0.5))) for k, v in g.items()}
    return g

  # Convert and validate base box.
  base = _canonical_box(box)
  # Early validation: must be at least 2D box (2x2 or 3x3).
  if base.ndim != 2 or base.shape[0] < 2:
    raise ValueError("shearing requires a box of dimension >= 2 (2x2 or 3x3).")
  # Validate diagonal lengths for planes in use (if specified via functions).
  used_planes = set()
  if shear_fns is not None:
    used_planes.update([p for p in shear_fns.keys() if p in ('xy','xz','yz')])
  if shear_fn is not None:
    used_planes.add('xy')
  if 'xy' in used_planes and base[1, 1] <= 0.0:
    raise ValueError("shearing requires positive length along y (base[1,1] > 0).")
  if base.shape[0] >= 3 and any(p in used_planes for p in ('xz','yz')) and base[2, 2] <= 0.0:
    raise ValueError("shearing requires positive length along z (base[2,2] > 0) for xz/yz planes.")
  # The box should be square or cubic. Variations are not supported yet (TODO).
  # if base.shape[0] >= 3 and not jnp.isclose(base[2, 2], base[1, 1]):
    # raise ValueError("shearing currently requires a cubic box.")
  # if not jnp.isclose(base[0, 0], base[1, 1]):
    # raise ValueError("shearing currently requires a square box.")
  # TODO: properly test fractional_coordinates=False case.
  if not fractional_coordinates:
    print("Warning: shearing with fractional_coordinates=False is not tested much.")

  # Delegate minimum-image logic to periodic_general; always pass the current box.
  disp_pg, shift_pg = periodic_general(
      base,
      fractional_coordinates=fractional_coordinates,
      wrapped=True)

  def _box_of(**kwargs) -> Box:
    """
    Compute a sheared box configuration from base box parameters and shear.
    This function creates a box matrix by applying shear deformation to a base box
    configuration. The shear is applied in the xy direction, modifying the off-diagonal
    element b[0,1] of the box matrix.
    Args:
      **kwargs: Keyword arguments that may include:
        - box: Base box matrix to apply shear to. If not provided, uses the 
             default 'base' box.
        - gamma: Shear strain parameter, or
        - t: Time parameter (used with shear_rate to compute gamma)
        - shear_rate: Rate of shear (used with t to compute gamma)
    Returns:
      Box: A box matrix with applied shear deformation. The xy component b[0,1]
         is set to (base_xy + gamma * Ly), where Ly is the y-dimension length
         and base_xy is either the original xy component (if keep_base_xy is True)
         or 0.0.
    Note:
      The function relies on module-level variables 'base' and 'keep_base_xy',
      and helper functions '_canonical_box' and '_gamma'.
    """
    # Compute the current sheared box from base and gamma.
    b = _canonical_box(kwargs.get('box', base))
    b = b.astype(jnp.result_type(b, f32))
    gammas = _gammas(**kwargs)  # dict per plane
    # Apply in upper-triangular convention: (i<j)
    if b.shape[0] >= 2 and 'xy' in gammas:
      base_off = b[0, 1] if keep_base_xy else f32(0.0)
      b = b.at[0, 1].set(base_off + gammas['xy'] * b[1, 1])
    if b.shape[0] >= 3:
      if 'xz' in gammas:
        base_off = b[0, 2] if keep_base_xy else f32(0.0)
        b = b.at[0, 2].set(base_off + gammas['xz'] * b[2, 2])
      if 'yz' in gammas:
        base_off = b[1, 2] if keep_base_xy else f32(0.0)
        b = b.at[1, 2].set(base_off + gammas['yz'] * b[2, 2])
    return b

  def displacement_fn(Ra, Rb, **kwargs):

    kwargs_no_box = dict(kwargs)

    if 'box' in kwargs_no_box:
      # Use the provided physical box as-is (except for vector->diag conversion).
      H = _canonical_box(kwargs_no_box.pop('box'))
    else:
      # Compute the current sheared box from base and gamma.
      H = _box_of(**kwargs_no_box)

    def _single(a, b):
      return disp_pg(a, b, box=H, **kwargs_no_box)

    if Ra.ndim == 1:
      return _single(Ra, Rb)

    # Allow batched inputs (e.g., shape [N, dim]) by vmapping over leading axes.
    lead_shape = Ra.shape[:-1]
    a_flat = jnp.reshape(Ra, (-1, Ra.shape[-1]))
    b_flat = jnp.reshape(Rb, (-1, Rb.shape[-1]))
    d_flat = jax.vmap(_single)(a_flat, b_flat)
    return jnp.reshape(d_flat, lead_shape + (Ra.shape[-1],))

  def shift_fn(R, dR, **kwargs):

    kwargs_no_box = dict(kwargs)

    if 'box' in kwargs_no_box:
      # Use the provided physical box as-is (except for vector->diag conversion).
      H = _canonical_box(kwargs_no_box.pop('box'))
    else:
      # Compute the current sheared box from base and gamma.
      H = _box_of(**kwargs_no_box)

    return shift_pg(R, dR, box=H, **kwargs_no_box)

  # Return the usual (displacement, shift), plus a helper to get the box at time t.
  return displacement_fn, shift_fn, _box_of


def metric(displacement: DisplacementFn) -> MetricFn:
  """Takes a displacement function and creates a metric."""
  return lambda Ra, Rb, **kwargs: distance(displacement(Ra, Rb, **kwargs))


def map_product(metric_or_displacement: DisplacementOrMetricFn
                ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over all pairs."""
  return vmap(vmap(metric_or_displacement, (0, None), 0), (None, 0), 0)


def map_bond(metric_or_displacement: DisplacementOrMetricFn
             ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over bonds."""
  return vmap(metric_or_displacement, (0, 0), 0)


def map_neighbor(metric_or_displacement: DisplacementOrMetricFn
                 ) -> DisplacementOrMetricFn:
  """Vectorizes a metric or displacement function over neighborhoods."""
  def wrapped_fn(Ra, Rb, **kwargs):
    return vmap(vmap(metric_or_displacement, (0, None)))(Rb, Ra, **kwargs)
  return wrapped_fn


def canonicalize_displacement_or_metric(displacement_or_metric):
  """Checks whether or not a displacement or metric was provided."""
  for dim in range(1, 4):
    try:
      R = ShapedArray((dim,), f32)
      dR_or_dr = eval_shape(displacement_or_metric, R, R, t=0)
      if len(dR_or_dr.shape) == 0:
        return displacement_or_metric
      else:
        return metric(displacement_or_metric)
    except TypeError:
      continue
    except ValueError:
      continue
  raise ValueError(
    'Canonicalize displacement not implemented for spatial dimension larger'
    'than 4.')


def remap_fractional_positions(Rf: Array, old_box: Box, new_box: Box) -> Array:
  """Remap fractional coordinates from old_box to new_box preserving real pos.

  Given fractional positions Rf defined with respect to old_box (upper-triangular
  or diagonal), return fractional positions Rf' with respect to new_box such that
  old_box @ Rf ≡ new_box @ Rf' (mod lattice vectors), and wrap Rf' into [0,1)^d.

  This is useful when using shearing with remap=True: when the box basis flips
  at gamma crossing a half-integer, map the fractional coordinates accordingly
  to avoid discontinuities in real space.
  """
  def _as_matrix(b, dim):
    arr = jnp.asarray(b)
    if arr.ndim == 0:
      return jnp.diag(jnp.ones((dim,), dtype=arr.dtype)) * arr
    if arr.ndim == 1:
      return jnp.diag(arr)
    return arr

  dim = Rf.shape[-1]
  H_old = _as_matrix(old_box, dim)
  H_new = _as_matrix(new_box, dim)
  R_real = transform(H_old, Rf)
  Rf_new = transform(jnp.linalg.inv(H_new), R_real)
  # Wrap into [0,1)
  return Rf_new - jnp.floor(Rf_new)
