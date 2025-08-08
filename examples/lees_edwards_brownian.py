# Copyright 2024 Google LLC
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

"""Brownian dynamics example with Lees-Edwards boundaries."""

from absl import app

from jax import random
import jax.numpy as jnp

from jax_md import space, energy, simulate, partition, util


def main(_):
  key = random.PRNGKey(0)

  N = 32
  side = jnp.array([10.0, 10.0, 10.0])
  shear_rate = 0.1
  displacement, shift = space.lees_edwards(side, shear_rate)

  neighbor_fn = partition.lees_edwards_neighbor_list(
      displacement, side, shear_rate, 2.5, dr_threshold=0.3)

  key, split = random.split(key)
  R = random.uniform(split, (N, 3), minval=0.0, maxval=side, dtype=util.f32)

  energy_fn = energy.soft_sphere_pair(displacement)
  dt = 1e-3
  init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, kT=1.0)
  state = init_fn(key, R)
  nbrs = neighbor_fn.allocate(state.position, t=0.0)

  for step in range(10):
    t = step * dt
    nbrs = neighbor_fn.update(state.position, nbrs, t=t)
    state = apply_fn(state, neighbor_idx=nbrs.idx, t=t)

  print(state.position)


if __name__ == '__main__':
  app.run(main)
