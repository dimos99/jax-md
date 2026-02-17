"""Scalar helpers for deterministic real-space RPY kernels."""

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import config as jax_config
from jax.scipy.special import erfc
import numpy as np


# Reused constants
REAL_DTYPE = jnp.float64 if jax_config.jax_enable_x64 else jnp.float32  # type: ignore
SQRT_PI = jnp.sqrt(jnp.array(jnp.pi, dtype=REAL_DTYPE))
PAIR_EPS_FRACTION_OF_DIAMETER = 1e-3


@partial(jax.jit, static_argnums=(1, 2))
def _F1F2_case_r_gt_2a(r, a, xi) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Fiore Appx. A, Case 1: r > 2a (monodisperse)."""
  xi2 = xi * xi
  xi3 = xi2 * xi
  xi4 = xi2 * xi2
  r2 = r * r
  r3 = r2 * r
  twoa = 2.0 * a
  ap = twoa + r
  am = twoa - r
  ap2 = ap * ap
  am2 = am * am
  ap3 = twoa + 3.0 * r
  am3 = twoa - 3.0 * r
  p_quad = 4.0 * a * a + 4.0 * a * r + 9.0 * r2
  m_quad = 4.0 * a * a - 4.0 * a * r + 9.0 * r2

  r_xi = r * xi
  rm2_xi = (r - twoa) * xi
  rp2_xi = (r + twoa) * xi
  ex_r = jnp.exp(-(r_xi) * (r_xi))
  ex_rm2 = jnp.exp(-(rm2_xi) * (rm2_xi))
  ex_rp2 = jnp.exp(-(rp2_xi) * (rp2_xi))
  erfc_r = erfc(r_xi)
  erfc_rm2 = erfc(rm2_xi)
  erfc_rp2 = erfc(rp2_xi)

  f10 = 0.0
  f11 = (18.0 * r2 * xi2 + 3.0) / (64.0 * SQRT_PI * a * r2 * xi3)
  f12 = (2.0 * xi2 * am * p_quad - ap3) / (128.0 * SQRT_PI * a * r3 * xi3)
  f13 = (-2.0 * xi2 * ap * m_quad + am3) / (128.0 * SQRT_PI * a * r3 * xi3)
  f14 = (3.0 - 36.0 * r2 * r2 * xi4) / (128.0 * a * r3 * xi4)
  f15 = (4.0 * xi4 * (r - twoa) * (r - twoa) * p_quad - 3.0) / (
      256.0 * a * r3 * xi4
  )
  f16 = (4.0 * xi4 * ap2 * m_quad - 3.0) / (256.0 * a * r3 * xi4)

  f20 = 0.0
  f21 = (6.0 * r2 * xi2 - 3.0) / (32.0 * SQRT_PI * a * r2 * xi3)
  f22 = (-2.0 * xi2 * am2 * ap3 + ap3) / (64.0 * SQRT_PI * a * r3 * xi3)
  f23 = (2.0 * xi2 * ap2 * am3 - am3) / (64.0 * SQRT_PI * a * r3 * xi3)
  f24 = -3.0 * (4.0 * r2 * r2 * xi4 + 1.0) / (64.0 * a * r3 * xi4)
  f25 = (3.0 - 4.0 * xi4 * am * am * am * ap3) / (128.0 * a * r3 * xi4)
  f26 = (3.0 - 4.0 * xi4 * am3 * ap * ap * ap) / (128.0 * a * r3 * xi4)

  F1 = (
      f10 + f11 * ex_r + f12 * ex_rm2 + f13 * ex_rp2 + f14 * erfc_r +
      f15 * erfc_rm2 + f16 * erfc_rp2
  )

  F2 = (
      f20 + f21 * ex_r + f22 * ex_rm2 + f23 * ex_rp2 + f24 * erfc_r +
      f25 * erfc_rm2 + f26 * erfc_rp2
  )

  return F1, F2


@partial(jax.jit, static_argnums=(1, 2))
def _F1F2_case_r_le_2a(r, a, xi) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Fiore Appx. A, Case 2: r <= 2a (monodisperse)."""
  xi2 = xi * xi
  xi3 = xi2 * xi
  xi4 = xi2 * xi2
  r2 = r * r
  r3 = r2 * r
  twoa = 2.0 * a
  ap = twoa + r
  am = twoa - r
  ap2 = ap * ap
  am2 = am * am
  ap3 = twoa + 3.0 * r
  am3 = twoa - 3.0 * r
  p_quad = 4.0 * a * a + 4.0 * a * r + 9.0 * r2
  m_quad = 4.0 * a * a - 4.0 * a * r + 9.0 * r2

  r_xi = r * xi
  rm2_xi = (r - twoa) * xi
  rp2_xi = (r + twoa) * xi
  ex_r = jnp.exp(-(r_xi) * (r_xi))
  ex_rm2 = jnp.exp(-(rm2_xi) * (rm2_xi))
  ex_rp2 = jnp.exp(-(rp2_xi) * (rp2_xi))
  erfc_r = erfc(r_xi)
  erfc_rm2 = erfc(rm2_xi)
  erfc_rp2 = erfc(rp2_xi)

  f10 = -((r - twoa) * (r - twoa) * p_quad) / (32.0 * a * r3)
  f11 = (18.0 * r2 * xi2 + 3.0) / (64.0 * SQRT_PI * a * r2 * xi3)
  f12 = (2.0 * xi2 * am * p_quad - ap3) / (128.0 * SQRT_PI * a * r3 * xi3)
  f13 = (-2.0 * xi2 * ap * m_quad + am3) / (128.0 * SQRT_PI * a * r3 * xi3)
  f14 = (3.0 - 36.0 * r2 * r2 * xi4) / (128.0 * a * r3 * xi4)
  f15 = (4.0 * xi4 * (r - twoa) * (r - twoa) * p_quad - 3.0) / (
      256.0 * a * r3 * xi4
  )
  f16 = (4.0 * xi4 * ap2 * m_quad - 3.0) / (256.0 * a * r3 * xi4)

  f20 = (am * am * am * ap3) / (16.0 * a * r3)
  f21 = (6.0 * r2 * xi2 - 3.0) / (32.0 * SQRT_PI * a * r2 * xi3)
  f22 = (-2.0 * xi2 * am2 * ap3 + ap3) / (64.0 * SQRT_PI * a * r3 * xi3)
  f23 = (2.0 * xi2 * ap2 * am3 - am3) / (64.0 * SQRT_PI * a * r3 * xi3)
  f24 = -3.0 * (4.0 * r2 * r2 * xi4 + 1.0) / (64.0 * a * r3 * xi4)
  f25 = (3.0 - 4.0 * xi4 * am * am * am * ap3) / (128.0 * a * r3 * xi4)
  f26 = (3.0 - 4.0 * xi4 * am3 * ap * ap * ap) / (128.0 * a * r3 * xi4)

  F1 = (
      f10 + f11 * ex_r + f12 * ex_rm2 + f13 * ex_rp2 + f14 * erfc_r +
      f15 * erfc_rm2 + f16 * erfc_rp2
  )

  F2 = (
      f20 + f21 * ex_r + f22 * ex_rm2 + f23 * ex_rp2 + f24 * erfc_r +
      f25 * erfc_rm2 + f26 * erfc_rp2
  )

  return F1, F2


@partial(jax.jit, static_argnums=(1, 2, 3))
def F1F2_closed_form(
    r,
    a,
    xi,
    pair_eps_fraction_of_diameter: float = PAIR_EPS_FRACTION_OF_DIAMETER):
  """
  Return (F1, F2) for all r >= 0 using Fiore's closed forms (monodisperse).

  Parameters
  ----------
  r : array_like
    Separation distance(s)
  a : float
    Sphere radius
  xi : float
    Ewald splitting parameter

  Returns
  -------
  F1, F2 : arrays
    Mobility coefficients (same shape as r)
  """
  if pair_eps_fraction_of_diameter <= 0.0:
    raise ValueError(
        "pair_eps_fraction_of_diameter must be positive for near-zero stability.")
  r = jnp.asarray(r)
  eps = jnp.asarray((2.0 * a) * pair_eps_fraction_of_diameter, dtype=r.dtype)
  r_eval = jnp.maximum(r, eps)
  twoa = 2.0 * a

  def f1f2_one(ri):
    return jax.lax.cond(
        ri > twoa,
        lambda rv: _F1F2_case_r_gt_2a(rv, a, xi),
        lambda rv: _F1F2_case_r_le_2a(rv, a, xi),
        ri,
    )

  if r_eval.ndim == 0:
    return f1f2_one(r_eval)
  r_flat = r_eval.reshape(-1)
  F1_flat, F2_flat = jax.vmap(f1f2_one)(r_flat)
  return F1_flat.reshape(r_eval.shape), F2_flat.reshape(r_eval.shape)


@partial(jax.jit, static_argnums=(1,))
def Mr_self(a, xi):
  """
  Eta-independent self-mobility factor (Fiore Appx. A, Eq. A4).

  Parameters
  ----------
  a : float
    Sphere radius
  xi : float
    Ewald splitting parameter

  Returns
  -------
  float
    Eta-independent factor; multiply by ``1 / (6π η a)`` to obtain the
    Cartesian self-mobility coefficient.
  """
  val = (1.0 / (4.0 * jnp.sqrt(jnp.pi) * xi * a)) * (
      1.0 - jnp.exp(-4.0 * a * a * xi * xi) +
      4.0 * jnp.sqrt(jnp.pi) * a * xi * erfc(2.0 * a * xi)
  )
  return val


def canonicalize_box_matrix(box_like, dim: int):
  """Normalize a box specification (scalar / vector / matrix) to a matrix."""
  if box_like is None:
    return None
  arr = jnp.asarray(box_like, dtype=REAL_DTYPE)
  if arr.ndim == 0:
    return jnp.eye(dim, dtype=REAL_DTYPE) * arr
  if arr.ndim == 1:
    if arr.shape[0] != dim:
      raise ValueError(f"Box vector has dim {arr.shape[0]}, expected {dim}.")
    return jnp.diag(arr).astype(REAL_DTYPE)
  if arr.ndim == 2:
    if arr.shape[0] != dim or arr.shape[1] != dim:
      raise ValueError(f"Box matrix must be ({dim},{dim}); got {arr.shape}.")
    return arr.astype(REAL_DTYPE)
  return None


def current_box_matrix(
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    box_fn: Optional[Callable[..., jnp.ndarray]],
    dim: int,
    *,
    fractional_coordinates: bool = True,
    **kwargs,
) -> jnp.ndarray:
  """Infer the physical box matrix for the current space."""

  if "box" in kwargs:
    candidate = canonicalize_box_matrix(kwargs["box"], dim)
    if candidate is not None:
      return candidate
  if box_fn is not None:
    candidate = canonicalize_box_matrix(box_fn(**kwargs), dim)
    if candidate is not None:
      return candidate

  closure = getattr(displacement_fn, "__closure__", None)
  freevars = getattr(getattr(displacement_fn, "__code__", None), "co_freevars", None)
  if closure is not None:
    if freevars is not None:
      for name, cell in zip(freevars, closure):
        if name in ("box", "_box"):
          candidate = canonicalize_box_matrix(cell.cell_contents, dim)
          if candidate is not None:
            return candidate
    for cell in closure:
      candidate = canonicalize_box_matrix(cell.cell_contents, dim)
      if candidate is not None:
        return candidate

  if not fractional_coordinates:
    raise ValueError(
        "Real-coordinate spaces require a physical `box` (scalar, vector, or matrix); "
        "provide one via the space or `box` kwargs.")

  # Probe fractional displacement with a small, non-wrapping step to recover box cols.
  origin = jnp.zeros((dim,), dtype=REAL_DTYPE)
  basis = jnp.eye(dim, dtype=REAL_DTYPE)
  eps = jnp.asarray(1e-3, dtype=REAL_DTYPE)
  cols = [displacement_fn(origin, eps * basis[i], **kwargs) / eps for i in range(dim)]
  return jnp.stack(cols, axis=1)


def generate_lattice_hypercube(dim: int, extent: int) -> Tuple[np.ndarray, int]:
  """Generate integer lattice indices on the symmetric hypercube [-extent, extent]^dim."""
  extent = max(int(extent), 0)
  ranges = [np.arange(-extent, extent + 1, dtype=np.int32) for _ in range(dim)]
  mesh = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, dim)
  zero_mask = np.all(mesh == 0, axis=1)
  if not zero_mask.any():
    raise RuntimeError("Lattice hypercube generation failed to include the zero vector.")
  zero_idx = int(np.argmax(zero_mask))
  return mesh.astype(np.int32, copy=False), zero_idx
