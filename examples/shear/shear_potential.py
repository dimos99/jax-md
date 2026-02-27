"""Potential-module loading and validation for shear runners."""

import hashlib
import importlib
import importlib.util
import inspect
import math
import os
from typing import Any
from typing import Callable
from typing import Dict


def _validate_potential_cutoff(r_cut: Any, context: str) -> float:
  try:
    r_cut_val = float(r_cut)
  except (TypeError, ValueError) as err:
    raise ValueError(f'{context} requires numeric r_cut; got {r_cut!r}.') from err
  if (not math.isfinite(r_cut_val)) or r_cut_val <= 0.0:
    raise ValueError(f'{context} requires finite r_cut > 0, got {r_cut_val}.')
  return r_cut_val

def _normalize_interaction_neighbor_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
  required_keys = {'format', 'dr_threshold', 'capacity_multiplier'}
  if not isinstance(raw, dict):
    raise ValueError('POTENTIAL_NEIGHBOR_PARAMS must be a dict.')
  unknown = sorted(set(raw.keys()) - required_keys)
  missing = sorted(required_keys - set(raw.keys()))
  if unknown:
    raise ValueError(
      f'POTENTIAL_NEIGHBOR_PARAMS contains unsupported key(s): {unknown}. '
      f'Supported keys: {sorted(required_keys)}.'
    )
  if missing:
    raise ValueError(
      f'POTENTIAL_NEIGHBOR_PARAMS is missing required key(s): {missing}.'
    )
  out = {
    'format': str(raw['format']).lower(),
    'dr_threshold': float(raw['dr_threshold']),
    'capacity_multiplier': float(raw['capacity_multiplier']),
  }
  if out['dr_threshold'] < 0.0:
    raise ValueError(
      'Potential POTENTIAL_NEIGHBOR_PARAMS.dr_threshold must be >= 0.'
    )
  if out['capacity_multiplier'] <= 0.0:
    raise ValueError(
      'Potential POTENTIAL_NEIGHBOR_PARAMS.capacity_multiplier must be > 0.'
    )
  return out

def _load_module_from_spec(path_or_module: str):
  if os.path.isfile(path_or_module):
    module_path = os.path.abspath(path_or_module)
    digest = hashlib.sha1(module_path.encode('utf-8')).hexdigest()[:12]
    module_name = f'shear_potential_{digest}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
      raise ValueError(f'Failed to load custom potential module from path: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path_or_module
  module = importlib.import_module(path_or_module)
  return module, path_or_module

def _validate_pair_fn(fn: Callable[..., Any], context: str):
  if not callable(fn):
    raise ValueError(f'{context} pair function is not callable.')
  sig = inspect.signature(fn)
  if len(sig.parameters) < 1:
    raise ValueError(f'{context} pair function must accept at least one positional argument (dr).')

def _resolve_potential(potential_path: str) -> Dict[str, Any]:
  module, source = _load_module_from_spec(potential_path)
  pair_fn = getattr(module, 'pair_potential', None)
  _validate_pair_fn(pair_fn, 'potential module')

  default_params = getattr(module, 'POTENTIAL_PARAMS', None)
  if not isinstance(default_params, dict):
    raise ValueError('potential module must define POTENTIAL_PARAMS as a dict.')
  if 'r_cut' not in default_params:
    raise ValueError('potential module POTENTIAL_PARAMS must include finite r_cut > 0.')
  r_cut = _validate_potential_cutoff(default_params['r_cut'], 'potential POTENTIAL_PARAMS')

  neighbor_defaults_raw = getattr(module, 'POTENTIAL_NEIGHBOR_PARAMS', None)
  if neighbor_defaults_raw is None:
    raise ValueError(
      'potential module must define POTENTIAL_NEIGHBOR_PARAMS.'
    )
  neighbor_defaults = _normalize_interaction_neighbor_settings(neighbor_defaults_raw)

  potential_name = getattr(module, 'POTENTIAL_NAME', 'custom')
  return {
    'name': str(potential_name),
    'source': source,
    'pair_fn': pair_fn,
    'params': dict(default_params),
    'r_cut': r_cut,
    'neighbor_defaults': neighbor_defaults,
  }
