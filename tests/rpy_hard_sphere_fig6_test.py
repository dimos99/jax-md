import math
import os

import numpy as np

from examples.rpy_hard_sphere_fig6 import common
from examples.rpy_hard_sphere_fig6 import export_modes_text
from examples.rpy_hard_sphere_fig6 import postprocess_hq


def _write_q_file(out_dir, mode_kind, phi, q_vectors, qa, bin_indices, bin_centers):
  q_data = {
      'n_vectors': np.rint(q_vectors / (2.0 * math.pi)).astype(np.int32),
      'q_vectors': np.asarray(q_vectors, dtype=np.float64),
      'qa': np.asarray(qa, dtype=np.float64),
      'bin_indices': np.asarray(bin_indices, dtype=np.int64),
      'bin_centers': np.asarray(bin_centers, dtype=np.float64),
      'bin_counts_all': np.bincount(
          np.asarray(bin_indices, dtype=np.int64),
          minlength=len(bin_centers),
      ).astype(np.int64),
  }
  common.save_q_metadata(
      common.q_metadata_path(str(out_dir), mode_kind, phi), q_data)


def test_box_size_matches_protocol_values():
  assert np.isclose(common.box_size_from_phi(8000, 0.1), 69.4586, atol=5e-5)
  assert np.isclose(common.box_size_from_phi(8000, 0.2), 55.1293, atol=5e-5)
  assert np.isclose(common.box_size_from_phi(8000, 0.3), 48.1599, atol=5e-5)


def test_overlap_repulsion_values():
  prefactor = 16.0 * math.pi * common.ETA / common.DT
  expected = prefactor * (2.0 * math.log(2.0) + 1.0 - 2.0)
  assert np.isclose(common.overlap_repulsion_energy(1.0), expected)
  assert common.overlap_repulsion_energy(2.0) == 0.0
  assert common.overlap_repulsion_energy(2.1) == 0.0


def test_reciprocal_vector_sampling_is_deterministic_and_capped():
  q1 = common.generate_reciprocal_vectors(
      12.0,
      q_min=1.0,
      q_max=4.0,
      bin_width=0.5,
      q_per_bin=3,
      seed=7,
  )
  q2 = common.generate_reciprocal_vectors(
      12.0,
      q_min=1.0,
      q_max=4.0,
      bin_width=0.5,
      q_per_bin=3,
      seed=7,
  )
  assert np.array_equal(q1['n_vectors'], q2['n_vectors'])
  selected_counts = np.bincount(
      q1['bin_indices'], minlength=q1['bin_centers'].shape[0])
  assert np.all(selected_counts[selected_counts > 0] <= 3)
  assert np.all(q1['qa'] >= 1.0)
  assert np.all(q1['qa'] <= 4.0)


def test_percus_yevick_low_density_limit():
  q = np.linspace(0.5, 10.0, 8)
  sq = common.percus_yevick_sq(q, 1e-8)
  assert np.allclose(sq, np.ones_like(q), atol=1e-6)


def test_export_modes_text_format(tmp_path):
  phi = 0.1
  common.ensure_output_dirs(str(tmp_path))
  q_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
  _write_q_file(
      tmp_path,
      'static',
      phi,
      q_vectors,
      qa=np.array([1.0, 2.0]),
      bin_indices=np.array([0, 1]),
      bin_centers=np.array([1.0, 2.0]),
  )
  np.savez(
      common.mode_chunk_path(str(tmp_path), 'static', phi, 0),
      time=np.array([0.1]),
      rho=np.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=np.complex128),
  )
  output = export_modes_text.export_modes_text(
      out_dir=str(tmp_path), mode_kind='static', phi=phi)
  rows = [
      line for line in open(output, encoding='utf-8').read().splitlines()
      if not line.startswith('#')
  ]
  assert len(rows) == 2
  first = rows[0].split()
  assert first[:4] == ['1.0000000000e-01', '1.0000000000e+00', '0.0000000000e+00', '0.0000000000e+00']
  assert first[4:] == ['1.0000000000e+00', '2.0000000000e+00']


def test_synthetic_dynamic_modes_recover_diffusivity(tmp_path):
  phi = 0.1
  n_particles = 128
  dt = 0.01
  q_value = 1.0
  diffusivity = 0.5
  n_time = 4096
  n_q = 128
  rng = np.random.default_rng(0)
  alpha = math.exp(-q_value * q_value * diffusivity * dt)
  noise_scale = math.sqrt(1.0 - alpha * alpha)
  x = np.empty((n_time, n_q), dtype=np.complex128)
  x[0] = (
      rng.normal(size=n_q) + 1j * rng.normal(size=n_q)
  ) / math.sqrt(2.0)
  for i in range(1, n_time):
    noise = (
        rng.normal(size=n_q) + 1j * rng.normal(size=n_q)
    ) / math.sqrt(2.0)
    x[i] = alpha * x[i - 1] + noise_scale * noise

  common.ensure_output_dirs(str(tmp_path))
  q_vectors = np.zeros((n_q, 3), dtype=np.float64)
  q_vectors[:, 0] = q_value
  _write_q_file(
      tmp_path,
      'dynamic',
      phi,
      q_vectors,
      qa=np.full((n_q,), q_value),
      bin_indices=np.zeros((n_q,), dtype=np.int64),
      bin_centers=np.array([1.1]),
  )
  common.write_json(
      common.mode_metadata_json_path(str(tmp_path), 'dynamic', phi),
      {
          'n_particles': n_particles,
          'dt': dt,
          'particle_radius': 1.0,
      },
  )
  np.savez(
      common.mode_chunk_path(str(tmp_path), 'dynamic', phi, 0),
      time=dt * np.arange(1, n_time + 1, dtype=np.float64),
      rho=math.sqrt(n_particles) * x,
  )
  postprocess_hq.compute_hq(
      out_dir=str(tmp_path), phi=phi, q_column_batch=17)
  data = np.loadtxt(common.dat_path(str(tmp_path), 'Hq', phi), comments='#')
  assert np.isclose(float(data[2]), diffusivity, rtol=0.08, atol=0.03)

