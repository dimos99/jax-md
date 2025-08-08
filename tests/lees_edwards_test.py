"""Tests for Lees-Edwards boundary conditions."""

from absl.testing import absltest

import jax.numpy as jnp

import os, sys, types, importlib, numpy as onp

jax_md_path = os.path.join(os.path.dirname(__file__), '..', 'jax_md')
jax_md_pkg = types.ModuleType('jax_md')
jax_md_pkg.__path__ = [jax_md_path]
sys.modules['jax_md'] = jax_md_pkg

space = importlib.import_module('jax_md.space')
partition = importlib.import_module('jax_md.partition')
util = importlib.import_module('jax_md.util')
f32 = util.f32


class LeesEdwardsTest(absltest.TestCase):

  def test_zero_shear_matches_periodic(self):
    side = jnp.array([10.0, 10.0, 10.0])
    disp_p, _ = space.periodic(side)
    disp_le, _ = space.lees_edwards(side, shear_rate=0.0)
    Ra = jnp.array([1.0, 2.0, 3.0])
    Rb = jnp.array([4.0, 5.0, 6.0])
    t = 1.23
    onp.testing.assert_allclose(disp_p(Ra, Rb), disp_le(Ra, Rb, t=t))

  def test_cross_boundary_displacement(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.1
    disp, _ = space.lees_edwards(side, gamma_dot)
    Ra = jnp.array([1.0, 0.0, 1.0])
    Rb = jnp.array([2.0, 0.0, 9.0])
    dR = disp(Ra, Rb, t=1.0)
    expected = jnp.array([0.0, 0.0, 2.0])
    onp.testing.assert_allclose(dR, expected)

  def test_large_shear_wraps(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.1
    disp, _ = space.lees_edwards(side, gamma_dot)
    Ra = jnp.array([1.0, 0.0, 1.0])
    Rb = jnp.array([2.0, 0.0, 9.0])
    dR0 = disp(Ra, Rb, t=0.0)
    dR = disp(Ra, Rb, t=20.0)
    onp.testing.assert_allclose(dR, dR0)

  def test_neighbor_list(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.2
    disp, _ = space.lees_edwards(side, gamma_dot)
    nl_fn = partition.lees_edwards_neighbor_list(
        disp, side, gamma_dot, 3.0,
        format=partition.NeighborListFormat.Dense)
    R = jnp.array([[1.0, 0.0, 1.0],
                   [2.0, 0.0, 9.0]], dtype=f32)
    neighbors = nl_fn.allocate(R, t=1.0)
    self.assertEqual(neighbors.idx[0, 0], 1)
    self.assertEqual(neighbors.idx[1, 0], 0)

  def test_neighbor_list_zero_shear_matches_periodic(self):
    side = jnp.array([10.0, 10.0, 10.0])
    disp_p, _ = space.periodic(side)
    disp_le, _ = space.lees_edwards(side, shear_rate=0.0)
    nl_p_fn = partition.neighbor_list(disp_p, side, 3.0,
                                      format=partition.NeighborListFormat.Dense)
    nl_le_fn = partition.lees_edwards_neighbor_list(
        disp_le, side, 0.0, 3.0,
        format=partition.NeighborListFormat.Dense)
    R = jnp.array([[1.0, 0.0, 1.0],
                   [2.0, 0.0, 9.0]], dtype=f32)
    nbrs_p = nl_p_fn.allocate(R)
    nbrs_le = nl_le_fn.allocate(R, t=1.0)
    onp.testing.assert_array_equal(nbrs_p.idx, nbrs_le.idx)

  def test_neighbor_list_changes_with_time(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.5
    disp, _ = space.lees_edwards(side, gamma_dot)
    nl_fn = partition.lees_edwards_neighbor_list(
        disp, side, gamma_dot, 3.0,
        format=partition.NeighborListFormat.Dense)
    R = jnp.array([[1.0, 0.0, 1.0],
                   [1.5, 0.0, 9.0]], dtype=f32)
    nbrs0 = nl_fn.allocate(R, t=0.0)
    nbrs = nl_fn.allocate(R, t=5.0)
    self.assertEqual(nbrs0.idx.shape[1], 1)
    self.assertEqual(nbrs.idx.shape[1], 0)

  def test_neighbor_list_rebuilds_only_when_needed(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.3
    disp, shift = space.lees_edwards(side, gamma_dot)
    nl_fn = partition.lees_edwards_neighbor_list(
        disp, side, gamma_dot, 3.0, dr_threshold=0.4,
        format=partition.NeighborListFormat.Dense)

    R = jnp.array([[1.0, 0.0, 1.0],
                   [1.5, 0.0, 9.0]], dtype=f32)
    t = 0.0
    nbrs = nl_fn.allocate(R, t=t)

    # Displace slightly below the rebuild threshold; neighbor list should not rebuild.
    dR_small = jnp.array([[0.1, 0.0, 0.0],
                          [0.0, 0.0, 0.0]], dtype=f32)
    R_small = shift(R, dR_small, t=t)
    nbrs_small = nl_fn.update(R_small, nbrs, t=t)
    # Neighbor list shouldn't rebuild; reference positions remain unchanged.
    onp.testing.assert_array_equal(nbrs_small.reference_position, nbrs.reference_position)
    expected_small = nl_fn.allocate(R_small, t=t)
    onp.testing.assert_array_equal(nbrs_small.idx, expected_small.idx)

    # Move beyond the threshold; neighbor list should rebuild and remain correct.
    dR_large = jnp.array([[3.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]], dtype=f32)
    R_large = shift(R_small, dR_large, t=t)
    nbrs_large = nl_fn.update(R_large, nbrs_small, t=t)
    # Rebuild updates the reference positions to the new configuration.
    onp.testing.assert_array_equal(nbrs_large.reference_position, R_large)
    expected_large = nl_fn.allocate(R_large, t=t)
    onp.testing.assert_array_equal(nbrs_large.idx, expected_large.idx)

  def test_neighbor_list_sequence_correctness(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.4
    disp, shift = space.lees_edwards(side, gamma_dot)
    nl_fn = partition.lees_edwards_neighbor_list(
        disp, side, gamma_dot, 3.0, dr_threshold=0.4,
        format=partition.NeighborListFormat.Dense)

    R = jnp.array([[1.0, 0.0, 1.0],
                   [2.0, 0.0, 1.2],
                   [8.0, 0.0, 1.0]], dtype=f32)

    t = 0.0
    nbrs = nl_fn.allocate(R, t=t)

    # Small displacement should not trigger a rebuild.
    t = 0.5
    dR_small = jnp.array([[0.1, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]], dtype=f32)
    R = shift(R, dR_small, t=t)
    nbrs_small = nl_fn.update(R, nbrs, t=t)
    onp.testing.assert_array_equal(nbrs_small.reference_position,
                                   nbrs.reference_position)
    expected_small = nl_fn.allocate(R, t=t)
    onp.testing.assert_array_equal(nbrs_small.idx, expected_small.idx)

    # A larger displacement should rebuild the neighbor list.
    t = 1.5
    dR_large = jnp.array([[3.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]], dtype=f32)
    R = shift(R, dR_large, t=t)
    nbrs_large = nl_fn.update(R, nbrs_small, t=t)
    onp.testing.assert_array_equal(nbrs_large.reference_position, R)
    expected_large = nl_fn.allocate(R, t=t)
    pad = jnp.full(nbrs_large.idx.shape, R.shape[0], dtype=nbrs_large.idx.dtype)
    pad = pad.at[:, :expected_large.idx.shape[1]].set(expected_large.idx)
    onp.testing.assert_array_equal(nbrs_large.idx, pad)

    # Advancing time without motion should keep the neighbor list intact.
    t = 2.5
    nbrs_final = nl_fn.update(R, nbrs_large, t=t)
    onp.testing.assert_array_equal(nbrs_final.reference_position, R)
    expected_final = nl_fn.allocate(R, t=t)
    onp.testing.assert_array_equal(nbrs_final.idx, expected_final.idx)


  def test_neighbor_list_dynamic_shear_matches_bruteforce(self):
    side = jnp.array([10.0, 10.0, 10.0])
    gamma_dot = 0.4
    disp, shift = space.lees_edwards(side, gamma_dot)
    nl_fn = partition.lees_edwards_neighbor_list(
        disp, side, gamma_dot, 3.0, dr_threshold=1.0,
        format=partition.NeighborListFormat.Dense)

    R = jnp.array([[1.0, 0.0, 1.0],
                   [1.5, 0.0, 9.0],
                   [8.0, 8.0, 1.0]], dtype=f32)

    dt = 0.1
    steps = 50
    t = 0.0
    nbrs = nl_fn.allocate(R, t=t)
    cell_size = float(nbrs.cell_size)
    last_shift = nbrs.shear_shift

    rebuilds = 0
    for step in range(1, steps + 1):
      t = step * dt
      dR = jnp.stack((gamma_dot * R[:, 2] * dt,
                      jnp.zeros(R.shape[0]),
                      jnp.zeros(R.shape[0])), axis=1)
      R = shift(R, dR, t=t)
      nbrs_new = nl_fn.update(R, nbrs, t=t)

      shear_shift = 0.5 * gamma_dot * float(side[2]) * t
      expected_rebuild = abs(shear_shift - last_shift) >= cell_size - 1e-6
      actual_rebuild = not onp.isclose(nbrs_new.shear_shift, nbrs.shear_shift)
      self.assertEqual(actual_rebuild, expected_rebuild)
      if actual_rebuild:
        last_shift = shear_shift
        rebuilds += 1

      dists = onp.zeros((R.shape[0], R.shape[0]))
      for i in range(R.shape[0]):
        for j in range(R.shape[0]):
          dists[i, j] = onp.linalg.norm(disp(R[i], R[j], t=t))
      for i in range(R.shape[0]):
        expected = set(onp.where((dists[i] < 3.0) &
                                 (onp.arange(R.shape[0]) != i))[0])
        actual = set(onp.array(nbrs_new.idx[i])[nbrs_new.idx[i] < R.shape[0]])
        self.assertTrue(expected.issubset(actual))

      nbrs = nbrs_new

    self.assertEqual(rebuilds, 2)


if __name__ == '__main__':
  absltest.main()
