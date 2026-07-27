"""One-time extractor for the FSD near-field resistance table.

Parses the hard-coded arrays in Fiore's Fast Stokesian Dynamics source
(``Stokes_ResistanceTable.cc``) into ``resistance_table_legacy.npz`` so that
jax-md does not depend on the FSD C++/CUDA tree at runtime.

The 22 monodisperse scalar functions are the *difference* between the exact
Jeffrey-Onishi (1984) / Jeffrey (1992) two-body resistance functions and the
two-body far-field multipole contribution (the subtraction is baked in).  They
are tabulated at 1000 center-to-center distances ``s = r/a`` spaced
logarithmically in the surface gap ``xi = s - 2`` over ``[1e-4, 2.0]``.

Column order (per row, matching FSD ``Stokes_ResistanceTable.cc``):

    [ XA11, XA12, YA11, YA12, YB11, YB12, XC11, XC12, YC11, YC12,
      XG11, XG12, YG11, YG12, YH11, YH12,
      XM11, XM12, YM11, YM12, ZM11, ZM12 ]

Usage::

    python jax_md/hydro/data/extract_legacy_resistance_table.py \
        [--source /path/to/FSD/source/Stokes_ResistanceTable.cc] \
        [--out jax_md/hydro/data/resistance_table_legacy.npz]
"""

import argparse
import os
import re

import numpy as np

N_DIST = 1000
N_FUNC = 22

COLUMN_NAMES = (
    'XA11', 'XA12', 'YA11', 'YA12', 'YB11', 'YB12', 'XC11', 'XC12', 'YC11',
    'YC12', 'XG11', 'XG12', 'YG11', 'YG12', 'YH11', 'YH12', 'XM11', 'XM12',
    'YM11', 'YM12', 'ZM11', 'ZM12',
)

# Tabulation metadata of the original FSD resistance table.
XI_MIN = 1.0e-4          # smallest tabulated surface gap (s - 2)
DR = 0.004305            # log-space discretization: ind*dr = log10(xi/xi_min)

_DEFAULT_SOURCE = os.path.normpath(
    os.path.join(os.path.dirname(__file__),
                 '..', '..', '..', '..', 'FSD', 'source',
                 'Stokes_ResistanceTable.cc'))
_DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), 'resistance_table_legacy.npz')

# Matches e.g. ``h_ResTable_dist.data[12] = 2.000113;`` (any whitespace).
_DIST_RE = re.compile(
    r'h_ResTable_dist\.data\[\s*(\d+)\s*\]\s*=\s*([-+0-9.eE]+)\s*;')
_VALS_RE = re.compile(
    r'h_ResTable_vals\.data\[\s*(\d+)\s*\]\s*=\s*([-+0-9.eE]+)\s*;')


def parse_source(text):
  """Parse distance and value arrays from the FSD source text."""
  dist = np.full(N_DIST, np.nan, dtype=np.float64)
  vals_flat = np.full(N_DIST * N_FUNC, np.nan, dtype=np.float64)

  for m in _DIST_RE.finditer(text):
    dist[int(m.group(1))] = float(m.group(2))
  for m in _VALS_RE.finditer(text):
    vals_flat[int(m.group(1))] = float(m.group(2))

  if np.isnan(dist).any():
    raise ValueError(
        'Missing %d distance entries' % int(np.isnan(dist).sum()))
  if np.isnan(vals_flat).any():
    raise ValueError(
        'Missing %d value entries' % int(np.isnan(vals_flat).sum()))

  # Stored contiguously per distance: [row0(22), row1(22), ...].
  vals = vals_flat.reshape(N_DIST, N_FUNC)
  return dist, vals


def _sanity_check(dist, vals):
  """Cheap invariants that catch a botched parse."""
  # Distances span the documented range, monotone increasing.
  assert abs(dist[0] - 2.0001) < 1e-6, dist[0]
  assert abs(dist[-1] - 4.0) < 1e-6, dist[-1]
  assert np.all(np.diff(dist) > 0), 'distances not strictly increasing'
  # Caveat A: every function decays to ~0 at the r=4a cutoff.
  assert np.all(np.abs(vals[-1]) < 1e-5), vals[-1]
  # Near contact XA11 ~ (1/4)/xi (squeeze-flow leading lubrication term).
  xi0 = dist[0] - 2.0
  assert abs(vals[0, 0] - 0.25 / xi0) / (0.25 / xi0) < 0.05, vals[0, 0]


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--source', default=_DEFAULT_SOURCE)
  ap.add_argument('--out', default=_DEFAULT_OUT)
  args = ap.parse_args()

  with open(args.source, 'r') as fh:
    text = fh.read()

  dist, vals = parse_source(text)
  _sanity_check(dist, vals)

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  np.savez(
      args.out,
      dist=dist,
      vals=vals,
      column_names=np.array(COLUMN_NAMES),
      xi_min=np.float64(XI_MIN),
      dr=np.float64(DR),
  )
  print('Wrote %s  (dist %s, vals %s)' % (args.out, dist.shape, vals.shape))


if __name__ == '__main__':
  main()
