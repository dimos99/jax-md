import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_script_module():
  root = Path(__file__).resolve().parents[1]
  path = root / 'examples' / 'hydro' / 'fiore_swan_self_diffusivity.py'
  spec = importlib.util.spec_from_file_location(
      'fiore_swan_self_diffusivity', path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_self_diffusivity_script_smoke(tmp_path):
  script = _load_script_module()
  rc = script.main([
      '--profile', 'fast',
      '--out-dir', str(tmp_path),
      '--phis', '0.1',
      '--sizes', '2',
      '--n-configs', '1',
      '--n-probes', '1',
      '--mc-sweeps', '1',
      '--traj-bursts', '1',
      '--traj-steps', '2',
      '--fit-lags', '1',
      '--P', '5',
      '--mgrid', '8',
      '--solve-maxiter', '30',
      '--mr-iters', '4',
      '--chunk-size', '2',
      '--no-plot',
      '--allow-f32',
  ])
  assert rc == 0

  summary_path = tmp_path / 'summary.csv'
  finite_path = tmp_path / 'finite_size.csv'
  raw_path = tmp_path / 'raw_samples.npz'
  metadata_path = tmp_path / 'metadata.json'
  assert summary_path.exists()
  assert finite_path.exists()
  assert raw_path.exists()
  assert metadata_path.exists()

  with open(summary_path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
  assert len(rows) == 1
  resistance = float(rows[0]['resistance'])
  trajectory = float(rows[0]['trajectory'])
  assert np.isfinite(resistance)
  assert np.isfinite(trajectory)
  assert resistance > 0.0
  assert trajectory > 0.0

  with open(finite_path, newline='', encoding='utf-8') as f:
    finite_rows = list(csv.DictReader(f))
  assert {row['method'] for row in finite_rows} == {'resistance', 'trajectory'}

  raw = np.load(raw_path)
  assert raw['resistance_correction'].shape == (1,)
  assert raw['trajectory_msd'].shape == (1,)
