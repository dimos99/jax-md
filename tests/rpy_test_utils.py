import itertools
import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_md import space
from jax_md.hydro import rpy, rpy_real, rpy_wave


HASIMOTO_EHAT = 2.837297


def _dtype_for_tests():
  return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _stokes_self_mobility(a: float, eta: float) -> float:
  return 1.0 / (6.0 * math.pi * eta * a)


def _free_space_rpy_block(r_vec: np.ndarray, a: float, eta: float) -> np.ndarray:
  """Free-space 3x3 RPY block (Fiore 2017 Eq. (4), overlap + non-overlap)."""
  r_vec = np.asarray(r_vec, dtype=np.float64)
  r = float(np.linalg.norm(r_vec))
  pref = _stokes_self_mobility(a, eta)
  eye = np.eye(3, dtype=np.float64)
  if r <= 1e-14:
    return pref * eye

  r_hat = r_vec / r
  rr = np.outer(r_hat, r_hat)

  if r > 2.0 * a:
    c_i = (3.0 * a) / (4.0 * r) + (a ** 3) / (2.0 * (r ** 3))
    c_rr = (3.0 * a) / (4.0 * r) - (3.0 * (a ** 3)) / (2.0 * (r ** 3))
    return pref * (c_i * eye + c_rr * rr)

  c_i = 1.0 - (9.0 * r) / (32.0 * a)
  c_rr = (3.0 * r) / (32.0 * a)
  return pref * (c_i * eye + c_rr * rr)


def _dense_matrix_from_matvec(matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
                              n_particles: int,
                              *,
                              dtype=np.float64) -> np.ndarray:
  """Build dense 3N x 3N matrix from a matvec over (N, 3)-shaped vectors."""
  ndof = 3 * n_particles
  basis = np.eye(ndof, dtype=dtype)
  dense = np.zeros((ndof, ndof), dtype=np.float64)

  for k in range(ndof):
    vec = jnp.asarray(basis[:, k].reshape((n_particles, 3)))
    out = matvec_fn(vec)
    if hasattr(out, 'block_until_ready'):
      out = out.block_until_ready()
    dense[:, k] = np.asarray(out, dtype=np.float64).reshape(-1)
  return dense


def _force_balance(forces: np.ndarray) -> np.ndarray:
  """Return a force-balanced array with zero net force per Cartesian axis."""
  forces = np.asarray(forces, dtype=np.float64)
  if forces.ndim != 2 or forces.shape[1] != 3:
    raise ValueError('forces must have shape (N, 3).')
  if forces.shape[0] <= 1:
    return np.array(forces, copy=True)
  return forces - forces.mean(axis=0, keepdims=True)


def _force_balanced_basis(n_particles: int,
                          *,
                          ref_particle: Optional[int] = None,
                          dtype=np.float64) -> np.ndarray:
  """Build a basis of balanced force vectors (+/- unit pairs) on 3N dofs.

  Returns a matrix B with shape (3N, 3(N-1)) whose columns each satisfy
  zero net force. Column vectors are formed by placing +1 on particle i and
  -1 on a fixed reference particle for one Cartesian axis.
  """
  if n_particles <= 1:
    raise ValueError('Force-balanced basis requires at least 2 particles.')
  ref = n_particles - 1 if ref_particle is None else int(ref_particle) % n_particles
  ncols = 3 * (n_particles - 1)
  basis = np.zeros((3 * n_particles, ncols), dtype=dtype)
  col = 0
  for i in range(n_particles):
    if i == ref:
      continue
    for alpha in range(3):
      basis[3 * i + alpha, col] = 1.0
      basis[3 * ref + alpha, col] = -1.0
      col += 1
  return basis


def _projected_matrix_from_matvec(
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_particles: int,
    *,
    ref_particle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Build projected operator B^T M B using only force-balanced probes."""
  basis = _force_balanced_basis(n_particles, ref_particle=ref_particle, dtype=np.float64)
  ncols = basis.shape[1]
  responses = np.zeros((3 * n_particles, ncols), dtype=np.float64)
  for k in range(ncols):
    force = jnp.asarray(basis[:, k].reshape((n_particles, 3)))
    out = matvec_fn(force)
    if hasattr(out, 'block_until_ready'):
      out = out.block_until_ready()
    responses[:, k] = np.asarray(out, dtype=np.float64).reshape(-1)
  projected = basis.T @ responses
  return projected, basis, responses


def _dense_matrix_from_apply(apply_fn: Callable,
                             state,
                             positions_frac: jnp.ndarray) -> np.ndarray:
  n_particles = int(positions_frac.shape[0])

  def _mv(vec):
    vel, _ = apply_fn(state, positions_frac, vec)
    return vel

  return _dense_matrix_from_matvec(_mv, n_particles)


def _projected_matrix_from_apply(
    apply_fn: Callable,
    state,
    positions_frac: jnp.ndarray,
    *,
    ref_particle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  n_particles = int(positions_frac.shape[0])

  def _mv(vec):
    vel, _ = apply_fn(state, positions_frac, vec)
    return vel

  return _projected_matrix_from_matvec(_mv, n_particles, ref_particle=ref_particle)


def _split_dense_mobility(state,
                          positions_frac: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  n_particles = int(positions_frac.shape[0])

  mr = _dense_matrix_from_matvec(
      lambda v: rpy_real.mr_matvec(state.real, positions_frac, v),
      n_particles)
  mw = _dense_matrix_from_matvec(
      lambda v: rpy_wave.mw_matvec(state.wave, positions_frac, v),
      n_particles)
  mtot = mr + mw
  return mr, mw, mtot


def _split_projected_mobility(
    state,
    positions_frac: jnp.ndarray,
    *,
    ref_particle: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  n_particles = int(positions_frac.shape[0])
  mr, basis, _ = _projected_matrix_from_matvec(
      lambda v: rpy_real.mr_matvec(state.real, positions_frac, v),
      n_particles,
      ref_particle=ref_particle,
  )
  mw, _, _ = _projected_matrix_from_matvec(
      lambda v: rpy_wave.mw_matvec(state.wave, positions_frac, v),
      n_particles,
      ref_particle=ref_particle,
  )
  mtot = mr + mw
  return mr, mw, mtot, basis


def _effective_kcut_cubic(box_length: float, mgrid: int) -> float:
  return math.pi * float(mgrid) / float(box_length)


def _build_operator(
    positions_frac: jnp.ndarray,
    box_matrix: jnp.ndarray,
    *,
    a: float,
    eta: float,
    xi: Optional[float] = None,
    tol: float = 1e-4,
    rcut: Optional[float] = None,
    p_support: Optional[int] = None,
    mgrid: Optional[int] = None,
    lattice_extent: Optional[int] = None,
    include_brownian: bool = True,
) -> Dict[str, object]:
  """Build deterministic/stochastic split-Ewald operator with deterministic defaults."""
  positions_frac = jnp.asarray(positions_frac, dtype=_dtype_for_tests())
  box_matrix = jnp.asarray(box_matrix, dtype=positions_frac.dtype)
  n_particles = int(positions_frac.shape[0])
  volume = float(np.linalg.det(np.asarray(box_matrix, dtype=np.float64)))
  phi = float(n_particles * (4.0 / 3.0) * math.pi * (a ** 3) / volume)
  estimate = rpy.estimate_rpy_params(
      tol=tol,
      A=box_matrix,
      a=a,
      N=n_particles,
      phi=phi,
      xi_override=xi,
  )

  xi_use = float(estimate.xi)
  rcut_use = float(estimate.rcut) if rcut is None else float(rcut)
  p_use = int(estimate.P) if p_support is None else int(p_support)
  m_use = int(estimate.M) if mgrid is None else int(mgrid)
  lattice_use = int(estimate.lattice_extent) if lattice_extent is None else int(lattice_extent)

  space_fns = space.periodic_general(box_matrix, fractional_coordinates=True)
  init_fn, apply_fn = rpy.build_rpy_mobility(
      space_fns,
      a=a,
      xi=xi_use,
      eta=eta,
      rcut=rcut_use,
      P=p_use,
      Mgrid=m_use,
      include_brownian=include_brownian,
      lattice_extent=lattice_use,
      real_space_mode='lattice',
  )
  state = init_fn(positions_frac)
  return {
      'positions_frac': positions_frac,
      'box_matrix': box_matrix,
      'init_fn': init_fn,
      'apply_fn': apply_fn,
      'state': state,
      'xi': xi_use,
      'rcut': rcut_use,
      'P': p_use,
      'Mgrid': m_use,
      'lattice_extent': lattice_use,
      'estimate': estimate,
  }


def _direct_lattice_rpy_matrix(positions_frac: jnp.ndarray,
                               box_matrix: jnp.ndarray,
                               *,
                               a: float,
                               eta: float,
                               extent: int) -> np.ndarray:
  """Brute-force periodic-image RPY sum on a symmetric image cube."""
  positions_frac = np.asarray(positions_frac, dtype=np.float64)
  box_matrix = np.asarray(box_matrix, dtype=np.float64)
  positions_real = np.asarray(space.transform(jnp.asarray(box_matrix), jnp.asarray(positions_frac)))
  n_particles = int(positions_real.shape[0])
  ndof = 3 * n_particles
  dense = np.zeros((ndof, ndof), dtype=np.float64)

  shifts = list(itertools.product(
      range(-extent, extent + 1),
      range(-extent, extent + 1),
      range(-extent, extent + 1),
  ))

  for i in range(n_particles):
    for j in range(n_particles):
      block = np.zeros((3, 3), dtype=np.float64)
      for shift in shifts:
        shift_vec = np.asarray(shift, dtype=np.float64)
        if i == j and np.all(shift_vec == 0.0):
          block += _stokes_self_mobility(a, eta) * np.eye(3, dtype=np.float64)
          continue
        lattice_vec = shift_vec @ box_matrix.T
        r_vec = (positions_real[j] + lattice_vec) - positions_real[i]
        block += _free_space_rpy_block(r_vec, a, eta)
      dense[3 * i:3 * i + 3, 3 * j:3 * j + 3] = block
  return dense


def _minimum_image_displacement(frac_delta: np.ndarray, box_matrix: np.ndarray) -> np.ndarray:
  wrapped = frac_delta - np.round(frac_delta)
  return wrapped @ box_matrix.T


def _nonoverlap_positions(n_particles: int,
                          box_matrix: np.ndarray,
                          *,
                          a: float,
                          seed: int = 0,
                          clearance: float = 0.03,
                          max_trials: int = 200000) -> jnp.ndarray:
  """Generate random fractional positions with hard-sphere minimum distance."""
  rng = np.random.default_rng(seed)
  box_matrix = np.asarray(box_matrix, dtype=np.float64)
  positions = np.zeros((n_particles, 3), dtype=np.float64)
  min_dist = 2.0 * a * (1.0 + clearance)

  placed = 0
  trials = 0
  while placed < n_particles:
    if trials > max_trials:
      raise RuntimeError(
          f'Could not place {n_particles} non-overlapping particles after {max_trials} trials.')
    trials += 1
    candidate = rng.random((3,), dtype=np.float64)

    ok = True
    for j in range(placed):
      dr = _minimum_image_displacement(candidate - positions[j], box_matrix)
      if np.linalg.norm(dr) < min_dist:
        ok = False
        break

    if ok:
      positions[placed] = candidate
      placed += 1

  return jnp.asarray(positions, dtype=_dtype_for_tests())


def _random_positions_with_overlaps(n_particles: int,
                                    box_matrix: np.ndarray,
                                    *,
                                    a: float,
                                    seed: int = 0) -> jnp.ndarray:
  """Generate random fractional positions and enforce deliberate overlap(s)."""
  rng = np.random.default_rng(seed)
  positions = rng.random((n_particles, 3), dtype=np.float64)
  if n_particles >= 2:
    lx = float(np.linalg.norm(np.asarray(box_matrix, dtype=np.float64)[:, 0]))
    delta = min(0.4 * a / max(lx, 1e-12), 0.2)
    positions[1] = np.mod(positions[0] + np.array([delta, 0.0, 0.0]), 1.0)
  if n_particles >= 3:
    positions[2] = np.mod(positions[0] + np.array([0.0, 0.35 * delta if n_particles >= 2 else 0.0, 0.0]), 1.0)
  return jnp.asarray(positions, dtype=_dtype_for_tests())


def _longitudinal_transverse(mij: np.ndarray, r_hat: Sequence[float]) -> Tuple[float, float]:
  mij = np.asarray(mij, dtype=np.float64)
  r_hat = np.asarray(r_hat, dtype=np.float64)
  r_hat = r_hat / max(np.linalg.norm(r_hat), 1e-15)
  m_parallel = float(r_hat @ mij @ r_hat)
  m_perp = float(0.5 * (np.trace(mij) - m_parallel))
  return m_parallel, m_perp


def _sample_covariance(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  samples = np.asarray(samples, dtype=np.float64)
  mean = samples.mean(axis=0)
  centered = samples - mean
  cov = centered.T @ centered / float(samples.shape[0])
  return cov, mean


def _cross_covariance(samples_a: np.ndarray, samples_b: np.ndarray) -> np.ndarray:
  a = np.asarray(samples_a, dtype=np.float64)
  b = np.asarray(samples_b, dtype=np.float64)
  a_centered = a - a.mean(axis=0)
  b_centered = b - b.mean(axis=0)
  return a_centered.T @ b_centered / float(a.shape[0])


def _count_overlaps(positions_frac: jnp.ndarray,
                    box_matrix: np.ndarray,
                    a: float) -> int:
  """Count overlapping pairs using a naive O(N^2) minimum-image search.

  Two particles i < j overlap when their minimum-image distance is
  strictly less than 2*a.  Returns the number of such pairs.
  """
  positions_frac = np.asarray(positions_frac, dtype=np.float64)
  box_matrix = np.asarray(box_matrix, dtype=np.float64)
  n_particles = int(positions_frac.shape[0])
  count = 0
  for i in range(n_particles):
    for j in range(i + 1, n_particles):
      dr = _minimum_image_displacement(positions_frac[i] - positions_frac[j], box_matrix)
      if np.linalg.norm(dr) < 2.0 * a:
        count += 1
  return count


def _frobenius_relative_error(observed: np.ndarray, reference: np.ndarray) -> float:
  observed = np.asarray(observed, dtype=np.float64)
  reference = np.asarray(reference, dtype=np.float64)
  denom = max(np.linalg.norm(reference), 1e-15)
  return float(np.linalg.norm(observed - reference) / denom)


def _wishart_frobenius_relative_scale(reference: np.ndarray, samples: int) -> float:
  """RMS scale of ||C_hat - C||_F / ||C||_F for Gaussian sample covariance.

  For z ~ N(0, C), C_hat = (1/S) sum z z^T, this returns
  sqrt((||C||_F^2 + tr(C)^2) / (S * ||C||_F^2)).
  """
  if samples <= 0:
    raise ValueError('samples must be positive.')
  reference = np.asarray(reference, dtype=np.float64)
  fro2 = float(np.sum(reference * reference))
  tr = float(np.trace(reference))
  denom = float(samples) * max(fro2, 1e-30)
  return math.sqrt((fro2 + tr * tr) / denom)
