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

"""Code to simulate systems in various statistical ensembles.

  This file contains a number of different methods that can be used to
  simulate systems in a variety of ensembles.

  In general, simulation code follows the same overall structure as optimizers
  in JAX. Simulations are tuples of two functions:

    init_fn:
      Function that initializes the  state of a system. Should take
      positions as an ndarray of shape `[n, output_dimension]`. Returns a state
      which will be a namedtuple.
    apply_fn:
      Function that takes a state and produces a new state after one
      step of optimization.

  One question that we need to think about is whether the simulations should
  also return a function that computes the invariant for that ensemble. This
  can be used for testing purposes, but is not often used otherwise.
"""

from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

import jax

from jax import grad
from jax import jit
from jax import random
from jax import debug
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap
from jax_md.hydro import pse as hydro_pse

static_cast = util.static_cast


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


"""Dispatch By State Code.

JAX MD allows for simulations to be extensible using a dispatch strategy where
functions are dispatched to specific cases based on the type of state provided.
In particular, we make decisions about which function to call based on the type
of the position argument. For those familiar with C / C++, our dispatch code is
essentially function overloading based on the type of the positions.

If you are interested in setting up a simulation using a different type of
system you can do so in a relatively light weight manner by introducing a new
type for storing the state that is compatible with the JAX PyTree system
(we usually choose a dataclass) and then overriding the functions below.

These extensions allow a range of simulations to be run by just changing the
type of the position argument. There are essentially two types of functions to
be overloaded. Functions that compute physical quantities, such as the kinetic
energy, and functions that evolve a state according to the Suzuki-Trotter
decomposition. Specifically, one might want to override the position step,
momentum step for deterministic and stochastic simulations or the
`stochastic_step` for stochastic simulations (e.g Langevin).
"""


class dispatch_by_state:
  """Wrap a function and dispatch based on the type of positions."""
  def __init__(self, fn):
    self._fn = fn
    self._registry = {}

  def __call__(self, state, *args, **kwargs):
    if type(state.position) in self._registry:
      return self._registry[type(state.position)](state, *args, **kwargs)
    return self._fn(state, *args, **kwargs)

  def register(self, oftype):
    def register_fn(fn):
      self._registry[oftype] = fn
    return register_fn


@dispatch_by_state
def canonicalize_mass(state: T) -> T:
  """Reshape mass vector for broadcasting with positions."""
  def canonicalize_fn(mass):
    if isinstance(mass, float):
      return mass
    if mass.ndim == 2 and mass.shape[1] == 1:
      return mass
    elif mass.ndim == 1:
      return jnp.reshape(mass, (mass.shape[0], 1))
    elif mass.ndim == 0:
      return mass
    msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
    )
    raise ValueError(msg)
  return state.set(mass=tree_map(canonicalize_fn, state.mass))

@dispatch_by_state
def canonicalize_mobility(state: T) -> T:
  """Reshape mobility vector for broadcasting with positions."""
  def canonicalize_fn(mobility):
    if isinstance(mobility, float):
      return mobility
    if mobility.ndim == 2 and mobility.shape[1] == 1:
      return mobility
    elif mobility.ndim == 1:
      return jnp.reshape(mobility, (mobility.shape[0], 1))
    elif mobility.ndim == 0:
      return mobility
    msg = (
      'Expected mobility to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mobility)
    )
    raise ValueError(msg)
  return state.set(mobility=tree_map(canonicalize_fn, state.mobility))


@dispatch_by_state
def initialize_momenta(state: T, key: Array, kT: float) -> T:
  """Initialize momenta with the Maxwell-Boltzmann distribution."""
  R, mass = state.position, state.mass

  R, treedef = tree_flatten(R)
  mass, _ = tree_flatten(mass)
  keys = random.split(key, len(R))

  def initialize_fn(k, r, m):
    p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
    # If simulating more than one particle, center the momentum.
    if r.shape[0] > 1:
      p = p - jnp.mean(p, axis=0, keepdims=True)
    return p

  P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

  return state.set(momentum=tree_unflatten(treedef, P))


@dispatch_by_state
def momentum_step(state: T, dt: float) -> T:
  """Apply a single step of the time evolution operator for momenta."""
  assert hasattr(state, 'momentum')
  new_momentum = tree_map(lambda p, f: p + dt * f,
                          state.momentum,
                          state.force)
  return state.set(momentum=new_momentum)


@dispatch_by_state
def position_step(state: T, shift_fn: Callable, dt: float, **kwargs) -> T:
  """Apply a single step of the time evolution operator for positions."""
  if isinstance(shift_fn, Callable):
    shift_fn = tree_map(lambda r: shift_fn, state.position)
  new_position = tree_map(lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs),
                          shift_fn,
                          state.position,
                          state.momentum,
                          state.mass)
  return state.set(position=new_position)


@dispatch_by_state
def kinetic_energy(state: T) -> Array:
  """Compute the kinetic energy of a state."""
  return quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)


@dispatch_by_state
def temperature(state: T) -> Array:
  """Compute the temperature of a state."""
  return quantity.temperature(momentum=state.momentum, mass=state.mass)


"""Deterministic Simulations

JAX MD includes integrators for deterministic simulations of the NVE, NVT, and
NPT ensembles. For a qualitative description of statistical physics ensembles
see the wikipedia article here:
en.wikipedia.org/wiki/Statistical_ensemble_(mathematical_physics)

Integrators are based direct translation method outlined in the paper,

"A Liouville-operator derived measure-preserving integrator for molecular
dynamics simulations in the isothermal–isobaric ensemble"

M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L Jochim, and G. J. Martyna
J. Phys. A: Math. Gen. 39 5629 (2006)

As such, we define several primitives that are generically useful in describing
simulations of this type. Namely, the velocity-Verlet integration step that is
used in the NVE and NVT simulations. We also define a general Nose-Hoover chain
primitive that is used to couple components of the system to a chain that
regulates the temperature. These primitives can be combined to construct more
interesting simulations that involve e.g. temperature gradients.
"""


def velocity_verlet(force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    state: T,
                    **kwargs) -> T:
  """Apply a single step of velocity Verlet integration to a state."""
  dt = f32(dt)
  dt_2 = f32(dt / 2)

  state = momentum_step(state, dt_2)
  state = position_step(state, shift_fn, dt, **kwargs)
  state = state.set(force=force_fn(state.position, **kwargs))
  state = momentum_step(state, dt_2)

  return state


# Constant Energy Simulations


@dataclasses.dataclass
class NVEState:
  """A struct containing the state of an NVE simulation.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    position: An ndarray of shape `[n, spatial_dimension]` storing the position
      of particles.
    momentum: An ndarray of shape `[n, spatial_dimension]` storing the momentum
      of particles.
    force: An ndarray of shape `[n, spatial_dimension]` storing the force
      acting on particles from the previous step.
    mass: A float or an ndarray of shape `[n]` containing the masses of the
      particles.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


# pylint: disable=invalid-name
def nve(energy_or_force_fn, shift_fn, dt=1e-3, **sim_kwargs):
  """Simulates a system in the NVE ensemble.

  Samples from the microcanonical ensemble in which the number of particles
  (N), the system volume (V), and the energy (E) are held constant. We use a
  standard velocity Verlet integration scheme.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
  Returns:
    See above.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
    force = force_fn(R, **kwargs)
    state = NVEState(R, None, force, mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    return velocity_verlet(force_fn, shift_fn, _dt, state, **kwargs)

  return init_fn, step_fn


# Constant Temperature Simulations


# Suzuki-Yoshida weights for integrators of different order.
# These are copied from OpenMM at
# https://github.com/openmm/openmm/blob/master/openmmapi/src/NoseHooverChain.cpp


SUZUKI_YOSHIDA_WEIGHTS = {
    1: [1],
    3: [0.828981543588751, -0.657963087177502, 0.828981543588751],
    5: [0.2967324292201065, 0.2967324292201065, -0.186929716880426,
        0.2967324292201065, 0.2967324292201065],
    7: [0.784513610477560, 0.235573213359357, -1.17767998417887,
        1.31518632068391, -1.17767998417887, 0.235573213359357,
        0.784513610477560]
}


@dataclasses.dataclass
class NoseHooverChain:
  """State information for a Nose-Hoover chain.

  Attributes:
    position: An ndarray of shape `[chain_length]` that stores the position of
      the chain.
    momentum: An ndarray of shape `[chain_length]` that stores the momentum of
      the chain.
    mass: An ndarray of shape `[chain_length]` that stores the mass of the
      chain.
    tau: The desired period of oscillation for the chain. Longer periods result
      is better stability but worse temperature control.
    kinetic_energy: A float that stores the current kinetic energy of the
      system that the chain is coupled to.
    degrees_of_freedom: An integer specifying the number of degrees of freedom
      that the chain is coupled to.
  """
  position: Array
  momentum: Array
  mass: Array
  tau: Array
  kinetic_energy: Array
  degrees_of_freedom: int=dataclasses.static_field()


@dataclasses.dataclass
class NoseHooverChainFns:
  initialize: Callable
  half_step: Callable
  update_mass: Callable


def nose_hoover_chain(dt: float,
                      chain_length: int,
                      chain_steps: int,
                      sy_steps: int,
                      tau: float
                      ) -> NoseHooverChainFns:
  """Helper function to simulate a Nose-Hoover Chain coupled to a system.

  This function is used in simulations that sample from thermal ensembles by
  coupling the system to one, or more, Nose-Hoover chains. We use the direct
  translation method outlined in Martyna et al. [#martyna92]_ and the
  Nose-Hoover chains are updated using two half steps: one at the beginning of
  a simulation step and one at the end. The masses of the Nose-Hoover chains
  are updated automatically to enforce a specific period of oscillation, `tau`.
  Larger values of `tau` will yield systems that reach the target temperature
  more slowly but are also more stable.

  As described in Martyna et al. [#martyna92]_, the Nose-Hoover chain often
  evolves on a faster timescale than the rest of the simulation. Therefore, it
  sometimes necessary
  to integrate the chain over several substeps for each step of MD. To do this
  we follow the Suzuki-Yoshida scheme. Specifically, we subdivide our chain
  simulation into :math:`n_c` substeps. These substeps are further subdivided
  into :math:`n_sy` steps. Each :math:`n_sy` step has length
  :math:`\\delta_i = \\Delta t w_i / n_c` where :math:`w_i` are constants such
  that :math:`\\sum_i w_i = 1`. See the table of Suzuki-Yoshida weights above
  for specific values. The number of substeps and the number of Suzuki-Yoshida
  steps are set using the `chain_steps` and `sy_steps` arguments.

  Consequently, the Nose-Hoover chains are described by three functions: an
  `init_fn` that initializes the state of the chain, a `half_step_fn` that
  updates the chain for one half-step, and an `update_chain_mass_fn` that
  updates the masses of the chain to enforce the correct period of oscillation.

  Note that a system can have many Nose-Hoover chains coupled to it to produce,
  for example, a temperature gradient. We also note that the NPT ensemble
  naturally features two chains: one that couples to the thermal degrees of
  freedom and one that couples to the barostat.

  Attributes:
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number :math:`n_c` of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of `dt`. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    A triple of functions that initialize the chain, do a half step of
    simulation, and update the chain masses respectively.
  """

  def init_fn(degrees_of_freedom, KE, kT):
    xi = jnp.zeros(chain_length, KE.dtype)
    p_xi = jnp.zeros(chain_length, KE.dtype)

    Q = kT * tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(degrees_of_freedom)
    return NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

  def substep_fn(delta, P, state, kT):
    """Apply a single update to the chain parameters and rescales velocity."""
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    delta_2 = delta   / f32(2.0)
    delta_4 = delta_2 / f32(2.0)
    delta_8 = delta_4 / f32(2.0)

    M = chain_length - 1

    G = (p_xi[M - 1] ** f32(2) / Q[M - 1] - kT)
    p_xi = p_xi.at[M].add(delta_4 * G)

    def backward_loop_fn(p_xi_new, m):
      G = p_xi[m - 1] ** 2 / Q[m - 1] - kT
      scale = jnp.exp(-delta_8 * p_xi_new / Q[m + 1])
      p_xi_new = scale * (scale * p_xi[m] + delta_4 * G)
      return p_xi_new, p_xi_new
    idx = jnp.arange(M - 1, 0, -1)
    _, p_xi_update = lax.scan(backward_loop_fn, p_xi[M], idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)

    G = f32(2.0) * KE - DOF * kT
    scale = jnp.exp(-delta_8 * p_xi[1] / Q[1])
    p_xi = p_xi.at[0].set(scale * (scale * p_xi[0] + delta_4 * G))

    scale = jnp.exp(-delta_2 * p_xi[0] / Q[0])
    KE = KE * scale ** f32(2)
    P = tree_map(lambda p: p * scale, P)

    xi = xi + delta_2 * p_xi / Q

    G = f32(2) * KE - DOF * kT
    def forward_loop_fn(G, m):
      scale = jnp.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
      p_xi_update = scale * (scale * p_xi[m] + delta_4 * G)
      G = p_xi_update ** 2 / Q[m] - kT
      return G, p_xi_update
    idx = jnp.arange(M)
    G, p_xi_update = lax.scan(forward_loop_fn, G, idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)
    p_xi = p_xi.at[M].add(delta_4 * G)

    return P, NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF), kT

  def half_step_chain_fn(P, state, kT):
    if chain_steps == 1 and sy_steps == 1:
      P, state, _ = substep_fn(dt, P, state, kT)
      return P, state

    delta = dt / chain_steps
    ws = jnp.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps])
    def body_fn(cs, i):
      d = f32(delta * ws[i % sy_steps])
      return substep_fn(d, *cs), 0
    P, state, _ = lax.scan(body_fn,
                           (P, state, kT),
                           jnp.arange(chain_steps * sy_steps))[0]
    return P, state

  def update_chain_mass_fn(state, kT):
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    Q = kT * _tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(DOF)

    return NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF)

  return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)


def default_nhc_kwargs(tau: float, overrides: Dict) -> Dict:
  default_kwargs = {
      'chain_length': 3,
      'chain_steps': 2,
      'sy_steps': 3,
      'tau': tau
  }

  if overrides is None:
    return default_kwargs

  return {
      key: overrides.get(key, default_kwargs[key])
      for key in default_kwargs
  }


@dataclasses.dataclass
class NVTNoseHooverState:
  """State information for an NVT system with a Nose-Hoover chain thermostat.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    chain: The variables describing the Nose-Hoover chain.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  chain: NoseHooverChain

  @property
  def velocity(self):
    return self.momentum / self.mass


def nvt_nose_hoover(energy_or_force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    kT: float,
                    chain_length: int=5,
                    chain_steps: int=2,
                    sy_steps: int=3,
                    tau: Optional[float]=None,
                    **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [#martyna92]_ [#martyna98]_
  [#tuckerman]_. We follow the direct translation method outlined in
  Tuckerman et al. [#tuckerman]_ and the interested reader might want to look
  at that paper as a reference.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, :math:`n_c`, of outer
      substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration
      occurs. Measured in units of `dt`. The performance of the Nose-Hoover
      chain thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  .. rubric:: References
  .. [#martyna92] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
    "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
    The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  .. [#martyna98] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
    "Explicit reversible integrators for extended systems dynamics."
    Molecular Physics 87. (1998) 1117-1157.
  .. [#tuckerman] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
    Andrea L. Jochim, and Glenn J. Martyna.
    "A Liouville-operator derived measure-preserving integrator for molecular
    dynamics simulations in the isothermal-isobaric ensemble."
    Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)

  thermostat = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    dof = quantity.count_dof(R)

    state = NVTNoseHooverState(R, None, force_fn(R, **kwargs), mass, None)
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(chain=thermostat.initialize(dof, KE, _kT))

  @jit
  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = thermostat.update_mass(chain, _kT)

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p)

    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

    chain = chain.set(kinetic_energy=kinetic_energy(state))

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p, chain=chain)

    return state
  return init_fn, apply_fn


def nvt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NVTNoseHooverState,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NVT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the Nose-Hoover thermostat.

  Arguments:
    energy_fn: The energy function of the Nose-Hoover system.
    state: The current state of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NVT dynamics.
  """
  PE = energy_fn(state.position, **kwargs)
  KE = kinetic_energy(state)

  DOF = quantity.count_dof(state.position)
  E = PE + KE

  c = state.chain

  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r
  return E


@dataclasses.dataclass
class NPTNoseHooverState:
  """State information for an NPT system with Nose-Hoover chain thermostats.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The velocity of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    reference_box: A box used to measure relative changes to the simulation
      environment.
    box_position: A positional degree of freedom used to describe the current
      box. box_position is parameterized as `box_position = (1/d)log(V/V_0)`
      where `V` is the current volume, `V_0` is the reference volume, and `d`
      is the spatial dimension.
    box_velocity: A velocity degree of freedom for the box.
    box_mass: The mass assigned to the box.
    barostat: The variables describing the Nose-Hoover chain coupled to the
      barostat.
    thermostsat: The variables describing the Nose-Hoover chain coupled to the
      thermostat.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  reference_box: Box

  box_position: Array
  box_momentum: Array
  box_mass: Array

  barostat: NoseHooverChain
  thermostat: NoseHooverChain

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass

  @property
  def box(self) -> Array:
    """Get the current box from an NPT simulation."""
    dim = self.position.shape[1]
    ref = self.reference_box
    V_0 = quantity.volume(dim, ref)
    V = V_0 * jnp.exp(dim * self.box_position)
    return (V / V_0) ** (1 / dim) * ref


def _npt_box_info(state: NPTNoseHooverState
                  ) -> Tuple[float, Callable[[float], float]]:
  """Gets the current volume and a function to compute the box from volume."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position)
  return V, lambda V: (V / V_0) ** (1 / dim) * ref


def npt_box(state: NPTNoseHooverState) -> Box:
  """Get the current box from an NPT simulation."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position)
  return (V / V_0) ** (1 / dim) * ref


def npt_nose_hoover(energy_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    pressure: float,
                    kT: float,
                    barostat_kwargs: Optional[Dict]=None,
                    thermostat_kwargs: Optional[Dict]=None) -> Simulator:
  """Simulation in the NPT ensemble using a pair of Nose Hoover Chains.

  Samples from the canonical ensemble in which the number of particles (N),
  the system pressure (P), and the temperature (T) are held constant.
  We use a pair of Nose Hoover Chains (NHC) described in
  [#martyna92]_ [#martyna98]_ [#tuckerman]_ coupled to the
  barostat and the thermostat respectively. We follow the direct translation
  method outlined in Tuckerman et al. [#tuckerman]_ and the interested reader
  might want to look at that paper as a reference.

  Args:
    energy_fn: A function that produces either an energy from a set of particle
      positions specified as an ndarray of shape `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    pressure: Floating point number specifying the target pressure. To update
      the pressure dynamically during a simulation one should pass `pressure`
      as a keyword argument to the step function.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    barostat_kwargs: A dictionary of keyword arguments passed to the barostat
      NHC. Any parameters not set are drawn from a relatively robust default
      set.
    thermostat_kwargs: A dictionary of keyword arguments passed to the
      thermostat NHC. Any parameters not set are drawn from a relatively robust
      default set.

  Returns:
    See above.

  """

  t = f32(dt)
  dt_2 = f32(dt / 2)

  force_fn = quantity.force(energy_fn)

  barostat_kwargs = default_nhc_kwargs(1000 * dt, barostat_kwargs)
  barostat = nose_hoover_chain(dt, **barostat_kwargs)

  thermostat_kwargs = default_nhc_kwargs(100 * dt, thermostat_kwargs)
  thermostat = nose_hoover_chain(dt, **thermostat_kwargs)

  def init_fn(key, R, box, mass=f32(1.0), **kwargs):
    N, dim = R.shape

    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # The box position is defined via pos = (1 / d) log V / V_0.
    zero = jnp.zeros((), dtype=R.dtype)
    one = jnp.ones((), dtype=R.dtype)
    box_position = zero
    box_momentum = zero
    box_mass = dim * (N + 1) * kT * barostat_kwargs['tau'] ** 2 * one
    KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

    if jnp.isscalar(box) or box.ndim == 0:
      # TODO(schsam): This is necessary because of JAX issue #5849.
      box = jnp.eye(R.shape[-1]) * box

    state = NPTNoseHooverState(
      R, None, force_fn(R, box=box, **kwargs),
      mass, box, box_position, box_momentum, box_mass,
      barostat.initialize(1, KE_box, _kT),
      None)  # pytype: disable=wrong-arg-count
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(
      thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT))

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
    return state.set(box_mass=box_mass)

  def box_force(alpha, vol, box_fn, position, momentum, mass, force, pressure,
                **kwargs):
    N, dim = position.shape

    def U(eps):
      return energy_fn(position, box=box_fn(vol), perturbation=(1 + eps),
                       **kwargs)

    dUdV = grad(U)
    KE2 = util.high_precision_sum(momentum ** 2 / mass)

    return alpha * KE2 - dUdV(0.0) - pressure * vol * dim

  def sinhx_x(x):
    """Taylor series for sinh(x) / x as x -> 0."""
    return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040 +
            x ** 8 / 362_880 + x ** 10 / 39_916_800)

  def exp_iL1(box, R, V, V_b, **kwargs):
    x = V_b * dt
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return shift_fn(R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV,
                    box=box, **kwargs)  # pytype: disable=wrong-keyword-args

  def exp_iL2(alpha, P, F, V_b):
    x = alpha * V_b * dt_2
    x_2 = x / 2
    sinhP = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

  def inner_step(state, **kwargs):
    _pressure = kwargs.pop('pressure', pressure)

    R, P, M, F = state.position, state.momentum, state.mass, state.force
    R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass

    N, dim = R.shape

    vol, box_fn = _npt_box_info(state)

    alpha = 1 + 1 / N
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e
    P = exp_iL2(alpha, P, F, P_b / M_b)

    R_b = R_b + P_b / M_b * dt
    state = state.set( box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, P / M, P_b / M_b)
    F = force_fn(R, box=box, **kwargs)

    P = exp_iL2(alpha, P, F, P_b / M_b)
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e

    return state.set(position=R, momentum=P, mass=M, force=F,
                     box_position=R_b, box_momentum=P_b, box_mass=M_b)

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
    P, tc = thermostat.half_step(S.momentum, tc, _kT)

    S = S.set(momentum=P, box_momentum=P_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
    tc = tc.set(kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
    bc = bc.set(kinetic_energy=KE_box)

    P, tc = thermostat.half_step(S.momentum, tc, _kT)
    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

    S = S.set(thermostat=tc, barostat=bc, momentum=P, box_momentum=P_b)

    return S
  return init_fn, apply_fn


def npt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NPTNoseHooverState,
                              pressure: float,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NPT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the NPT simulation.

  Arguments:
    energy_fn: The energy function of the system.
    state: The current state of the system.
    pressure: The current goal pressure of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NPT dynamics.
  """
  volume, box_fn = _npt_box_info(state)
  PE = energy_fn(state.position, box=box_fn(volume), **kwargs)
  KE = kinetic_energy(state)

  DOF = state.position.size
  E = PE + KE

  c = state.thermostat
  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r

  c = state.barostat
  for r, p, m in zip(c.position, c.momentum, c.mass):
    E += p ** 2 / (2 * m) + kT * r

  E += pressure * volume
  E += state.box_momentum ** 2 / (2 * state.box_mass)

  return E


"""Stochastic Simulations

JAX MD includes integrators for stochastic simulations of Langevin dynamics and
Brownian motion for systems in the NVT ensemble with a solvent.
"""


@dataclasses.dataclass
class Normal:
  """A simple normal distribution."""
  mean: jnp.ndarray
  var: jnp.ndarray

  def sample(self, key):
    mu, sigma = self.mean, jnp.sqrt(self.var)
    return mu + sigma * random.normal(key, mu.shape ,dtype=mu.dtype)

  def log_prob(self, x):
    return (-0.5 * jnp.log(2 * jnp.pi * self.var) -
            1 / (2 * self.var) * (x - self.mean)**2)


@dataclasses.dataclass
class NVTLangevinState:
  """A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    force: The (non-stochastic) force on particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


@dispatch_by_state
def stochastic_step(state: NVTLangevinState, dt:float, kT: float, gamma: float):
  """A single stochastic step (the `O` step)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(kT * (1 - c1**2))
  momentum_dist = Normal(c1 * state.momentum, c2**2 * state.mass)
  key, split = random.split(state.rng)
  return state.set(momentum=momentum_dist.sample(split), rng=key)


def nvt_langevin(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True,
                 **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using the BAOAB Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting
  with fictitious microscopic degrees of freedom. An example of this would be
  large particles in a solvent such as water. Thus, Langevin dynamics are a
  stochastic ODE described by a friction coefficient and noise of a given
  covariance.

  Our implementation follows the paper [#davidcheck] by Davidchack, Ouldridge,
  and Tretyakov.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    center_velocity: A boolean specifying whether or not the center of mass
      position should be subtracted.
  Returns:
    See above.

  .. rubric:: References
  .. [#carlon] R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov.
    "New Langevin and gradient thermostats for rigid body dynamics."
    The Journal of Chemical Physics 142, 144114 (2015)
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)
    state = NVTLangevinState(R, None, force, mass, key)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    dt_2 = _dt / 2

    state = momentum_step(state, dt_2)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = stochastic_step(state, _dt, _kT, gamma)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_step(state, dt_2)

    return state

  return init_fn, step_fn


def _normalize_shear_schedule(
    shear_schedule: Union[Callable[[Array], Array], Dict[str, Callable[[Array], Array]], None]
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array], Callable[[Array], Array]]:
  """Return per-plane shear callables for xy/xz/yz with zero fallbacks."""
  def _wrap(fn):
    if fn is None:
      return lambda t: f32(0.0)
    return lambda t: f32(fn(t))

  if isinstance(shear_schedule, dict):
    return (
        _wrap(shear_schedule.get('xy')),
        _wrap(shear_schedule.get('xz')),
        _wrap(shear_schedule.get('yz')),
    )
  if callable(shear_schedule):
    return _wrap(shear_schedule), _wrap(None), _wrap(None)
  return _wrap(None), _wrap(None), _wrap(None)


def _make_shear_at(sf_xy, sf_xz, sf_yz):
  """Return a `(time, dim) -> (gamma_xy, gamma_xz, gamma_yz)` shear helper."""
  def _shear_at(time, dim):
    gamma_xy = f32(sf_xy(time))
    if dim >= 3:
      gamma_xz = f32(sf_xz(time))
      gamma_yz = f32(sf_yz(time))
    else:
      gamma_xz = gamma_yz = f32(0.0)
    return gamma_xy, gamma_xz, gamma_yz
  return _shear_at


def _reduce_shear_strain(gamma_xy, gamma_xz, gamma_yz, dim):
  """Reduce shear strain into [-0.5, 0.5) and return integer remap counters."""
  m_xy = jnp.floor(gamma_xy + f32(0.5))
  if dim >= 3:
    m_xz = jnp.floor(gamma_xz + f32(0.5))
    m_yz = jnp.floor(gamma_yz + f32(0.5))
  else:
    m_xz = m_yz = f32(0.0)
  return (
      gamma_xy - m_xy,
      gamma_xz - m_xz,
      gamma_yz - m_yz,
      m_xy.astype(jnp.int32),
      m_xz.astype(jnp.int32),
      m_yz.astype(jnp.int32),
  )


def _apply_fractional_shear_remap(R, m_xy, m_xz, m_yz, dim):
  """Apply the unimodular fractional-coordinate remap for shearing(..., remap=True)."""
  if dim >= 3:
    mxy = jnp.asarray(m_xy, R.dtype)
    mxz = jnp.asarray(m_xz, R.dtype)
    myz = jnp.asarray(m_yz, R.dtype)

    def _apply_3d_remap(Rin):
      add_x = mxy * Rin[:, 1] + (mxz + mxy * myz) * Rin[:, 2]
      add_y = myz * Rin[:, 2]
      Rout = Rin.at[:, 0].add(add_x).at[:, 1].add(add_y)
      return jnp.mod(Rout, 1.0)

    any_flip = (jnp.not_equal(m_xy, 0) |
                jnp.not_equal(m_xz, 0) |
                jnp.not_equal(m_yz, 0))
    return lax.cond(any_flip, _apply_3d_remap, lambda Rin: Rin, R)

  mxy = jnp.asarray(m_xy, R.dtype)

  def _apply_2d_remap(Rin):
    Rout = Rin.at[:, 0].add(mxy * Rin[:, 1])
    return jnp.mod(Rout, 1.0)

  any_flip = jnp.not_equal(m_xy, 0)
  return lax.cond(any_flip, _apply_2d_remap, lambda Rin: Rin, R)


@dataclasses.dataclass
class BrownianState:
  """A tuple containing state information for Brownian dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mobility: The mobility of particles. Will either be a float or an ndarray
      of floats with shape `[n]`. It is not really time-dependent but we include it
      in the state so that it can be overridden at each step if desired.
      (for example, to implement spatially varying mobility, which is NOT
      supported at this time).
    rng: The current state of the random number generator.
  """
  position: Array
  mobility: Array
  rng: Array


def brownian(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             mobility) -> Simulator:
  """Simulation of Brownian dynamics with explicit (possibly per-particle) mobility.

  Implements the overdamped Langevin (Brownian) update using Euler–Maruyama:

    R_{t+dt} = R_t + μ ∘ F(R_t) dt + sqrt(2 μ kT dt) ∘ ξ,

  where μ ("mobility") can be a scalar or per-particle array; broadcasting over
  coordinates is supported when μ has shape [n, 1].

  Args:
    energy_or_force: Callable returning either an energy or a force given positions
      of shape `[n, spatial_dimension]`. Energies are converted to forces via
      `quantity.canonicalize_force`.
    shift: Position update function handling boundary conditions. Called as
      `shift(R, dR, **kwargs)` and should return the shifted positions.
    dt: Integrator time step (float).
    kT: Thermal energy (k_B T) in the same units as the potential energy.
    mobility: Mobility μ. Can be a scalar or ndarray with shape `[n]` or `[n,1]`.

  Returns:
    A pair `(init_fn, apply_fn)` consistent with other JAX MD simulators.
  """
  force_fn = quantity.canonicalize_force(energy_or_force)

  # Ensure `dt` is a static JAX scalar with the right dtype.
  (_dt,) = static_cast(dt)

  @jit
  def init_fn(key, R, **kwargs):
    # Stash μ in the state; allow overriding at init via kwargs if desired.
    mu = kwargs.get('mobility', mobility)
    state = BrownianState(R, mu, key)
    # Reshape μ for broadcasting with positions (e.g., [n] -> [n,1]).
    state = canonicalize_mobility(state)
    return state

  @jit
  def apply_fn(state, **kwargs):
    # Allow temperature to be overridden at step time.
    _kT = kwargs.get('kT', kT)

    R, mu, key = dataclasses.astuple(state)

    # Compute deterministic force.
    F = force_fn(R, **kwargs)

    # Draw i.i.d. standard normal noise per coordinate.
    key, split = random.split(key)
    xi = random.normal(split, R.shape, R.dtype)

    # Broadcast μ to coordinates: expected shapes are scalar or [n,1].
    mu = jnp.asarray(mu, dtype=R.dtype)

    # Brownian increment: μ F dt + sqrt(2 μ kT dt) ξ
    dR = mu * F * _dt + jnp.sqrt(f32(2) * mu * _kT * _dt) * xi
    R  = shift(R, dR, **kwargs)

    return BrownianState(R, mu, key)

  return init_fn, apply_fn


@dataclasses.dataclass
class ShearedBrownianState:
  """A tuple containing state information for Brownian dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mobility: The mobility of particles. Will either be a float or an ndarray
    time: The current simulation time (float).
    rng: The current state of the random number generator.
    force: Deterministic force on each particle; shape `[n, spatial_dimension]`.
  """
  position: Array
  mobility: Array
  rng: Array
  time: float
  force: Array


def brownian_with_shear(energy_or_force: Callable[..., Array],
                        shift: ShiftFn,
                        dt: float,
                        kT: float,
                        mobility,
                        shear_schedule: Union[Callable[[Array], Array], Dict[str, Callable[[Array], Array]], None],
                        t0: float = 0.0,
                        fractional_coordinates: bool = True,
                        remap = True) -> Simulator:
  """Overdamped Langevin (Brownian) dynamics under simple shear with PBCs.

  This integrator advances positions using Euler-Maruyama in the presence of a
  time-dependent simple shear defined by a user-specified schedule.
  The update is:

    R_{t+dt} = shift(R_t, μ ∘ F(R_t) dt + sqrt(2 μ kT dt) ∘ ξ; gamma)

  where `mu` (mobility) can be a scalar or per-particle array, `F` is the force
  derived from `energy_or_force`, `ξ ~ N(0, I)`, and `gamma` is the reduced
  shear strain computed from `shear_rate * t` and (optionally) wrapped into
  `[-0.5, 0.5)` when `remap=True`.

  This routine also maintains useful diagnostics in the returned state:
  - `time`: the simulation clock.
  - `force`: the deterministic force used at the step.

  Expected geometry setup: construct `displacement`, `shift`, and `box_of`
  using `jax_md.space.shearing(...)` so that all components share a consistent
  definition of the sheared periodic boundary conditions. The `apply_fn`
  forwards `gamma` (the reduced shear) via `**kwargs` to both `shift` and the
  force function, enabling correct use with neighbor lists and sheared metrics.

  Args:
    energy_or_force: Callable mapping positions to an energy or to a force.
      Energies are converted to forces via `quantity.canonicalize_force`. The
      callable should accept `**kwargs` such as `gamma` and `neighbor` if
      required by your setup (e.g., neighbor-list aware pair functions).
    shift: Position update function handling boundary conditions, typically the
      `shift` returned by `space.shearing(...)`. It will be called as
      `shift(R, dR, gamma=..., ...)` each step.
    dt: Time step.
    kT: Thermal energy (k_B T) in the same units as the potential energy.
    mobility: Particle mobility μ. Can be a scalar or an array with shape
      `[n]` or `[n, 1]` (the latter broadcasts cleanly over coordinates). The
      value is stored in the state and can be overridden at initialization via
      `init_fn(..., mobility=...)`.
    shear_schedule: Either a single function returning the reduced shear `gamma(t)`
      (applied to the 'xy' tilt), or a dict mapping plane names ('xy','xz','yz')
      to functions of time. The geometry is determined by the provided
      `space.shearing` configuration.
    t0: Initial time.
    fractional_coordinates: If True, `R` are interpreted as fractional coordinates
      inside the unit cell; real positions are obtained via
      `space.transform(box_of(...), R)`. For stress, real positions are used.
    remap: If True, apply a nearest-integer remap so that `gamma` is kept in
      `[-0.5, 0.5)` and, when using fractional positions, apply the exact
      unimodular change-of-basis to keep coordinates inside the base cell.

  Returns:
    A pair `(init_fn, apply_fn)`:
      - `init_fn(key, R, **kwargs)` -> ShearedBrownianState. Recognizes an
        optional `mobility` override.
      - `apply_fn(state, **kwargs)` -> ShearedBrownianState. Recognizes an
        optional `kT` override and forwards all other `**kwargs` (including
        `neighbor`, etc.) to the force and shift functions. The field
        `state.time` is advanced by `dt` each call.

  Example:
    disp, shift, box_of = space.shearing(box=H0, shear_schedule=sr_fn,
                                         remap=True, fractional_coordinates=True)
    pair_fn = smap.pair_neighbor_list(..., displacement_or_metric=disp, ...)
    init_fn, apply_fn = simulate.brownian_with_shear(pair_fn, shift, dt, kT,
                                                     mobility=1.0,
                                                     shear_schedule=sr_fn,
                                                     fractional_coordinates=True,
                                                     remap=True)
    state = init_fn(key, R0)
    state = apply_fn(state, neighbor=nbrs)  # kwargs forwarded as needed
    stress_fn = rheo.make_pairwise_stress_fn(pair_energy_for_stress)
    stress = stress_fn(state.position, box_of(t=state.time),
                       neighbor=nbrs, fractional_coordinates=True)

  Notes:
    - The mobility stored in the state is canonicalized for broadcasting.
    - Stress is no longer accumulated inside the integrator; derive a stress
      function with `rheo.make_pairwise_stress_fn(pair_energy_for_stress)` and
      evaluate it outside the integration loop using the current positions and
      box (e.g., `box_of(t=state.time)` when using sheared boxes).
  """
  
  force_fn = quantity.canonicalize_force(energy_or_force)

  # Ensure `dt` is a static JAX scalar with the right dtype.
  _dt = f32(dt)
  t0 = f32(t0)

  sf_xy, sf_xz, sf_yz = _normalize_shear_schedule(shear_schedule)
  shear_at = _make_shear_at(sf_xy, sf_xz, sf_yz)

  @jit
  def init_fn(key, R, **kwargs):
    # Stash μ in the state; allow overriding at init via kwargs if desired.
    mu = kwargs.get('mobility', mobility)
    # Initial gamma and current box for stress/real positions.
    # Use reduced gamma to match neighbor-list/space kernels.
    dim = R.shape[1]
    time0 = jnp.array(t0, dtype=R.dtype)
    gamma_xy, gamma_xz, gamma_yz = shear_at(time0, dim)
      
    if remap:
      curr_xy, curr_xz, curr_yz, _, _, _ = _reduce_shear_strain(
          gamma_xy, gamma_xz, gamma_yz, dim)
    else:
      curr_xy, curr_xz, curr_yz = gamma_xy, gamma_xz, gamma_yz

    # Compute deterministic force at t0 (passes gamma for sheared BCs if needed).
    init_kwargs = dict(kwargs)
    if dim >= 3:
      init_kwargs.update({'gamma_xy': curr_xy, 'gamma_xz': curr_xz, 'gamma_yz': curr_yz})
    else:
      init_kwargs['gamma'] = curr_xy
    F0 = force_fn(R, **init_kwargs)

    state = ShearedBrownianState(position=R, mobility=mu, rng=key,  #type: ignore
                                 time=t0, force=F0) #type: ignore
    
    # Reshape μ for broadcasting with positions (e.g., [n] -> [n,1]).
    state = canonicalize_mobility(state)
    return state
  
  @jit
  def apply_fn(state, **kwargs):
    # Allow temperature to be overridden at step time.
    _kT = kwargs.get('kT', kT)

    R, mu, key, time, _ = dataclasses.astuple(state)
    dim = R.shape[1]

    if remap:
      # --- Shear handling: compute reduced gamma and optional integer remap ---
      # Unwrapped shears at start/end of the step
      prev_xy, prev_xz, prev_yz = shear_at(time, dim)
      curr_xy, curr_xz, curr_yz = shear_at(time + _dt, dim)

      # Nearest-integer counters (how many unit tilts elapsed)
      _, _, _, m_prev_xy, m_prev_xz, m_prev_yz = _reduce_shear_strain(
          prev_xy, prev_xz, prev_yz, dim)
      curr_xy, curr_xz, curr_yz, m_curr_xy, m_curr_xz, m_curr_yz = _reduce_shear_strain(
          curr_xy, curr_xz, curr_yz, dim)
      m_xy = (m_curr_xy - m_prev_xy).astype(jnp.int32)
      m_xz = (m_curr_xz - m_prev_xz).astype(jnp.int32)
      m_yz = (m_curr_yz - m_prev_yz).astype(jnp.int32)

      # If we store fractional coordinates and a tilt index changed, apply the exact
      # unimodular change-of-basis map to the coordinates. Handle 2D and 3D.
      if fractional_coordinates:
        R = _apply_fractional_shear_remap(R, m_xy, m_xz, m_yz, dim)
    else:
      curr_xy, curr_xz, curr_yz = shear_at(time + _dt, dim)

    # Expose the reduced shear to downstream kernels (displacement/shift)
    if dim >= 3:
      kwargs.update({'gamma_xy': curr_xy, 'gamma_xz': curr_xz, 'gamma_yz': curr_yz})
    else:
      kwargs['gamma'] = curr_xy
    
    # Compute deterministic force.
    F = force_fn(R, **kwargs)

    # Now, advance time.
    # Draw i.i.d. standard normal noise per coordinate.
    key, split = random.split(key)
    xi = random.normal(split, R.shape, R.dtype)

    # Broadcast μ to coordinates: expected shapes are scalar or [n,1].
    mu = jnp.asarray(mu, dtype=R.dtype)

    # Brownian increment: μ F dt + sqrt(2 μ kT dt) ξ
    dR = mu * F * _dt + jnp.sqrt(f32(2) * mu * _kT * _dt) * xi
    R  = shift(R, dR, **kwargs)
    # Package updated state, canonicalize mobility broadcasting, and return.
    new_state = ShearedBrownianState(position=R, mobility=mu, rng=key, #type: ignore
                                     time=time + _dt, force=F) #type: ignore
    new_state = canonicalize_mobility(new_state)
    return new_state
  
  return init_fn, apply_fn


@dataclasses.dataclass
class RPYState:
  """State for pse/Rotne-Prager–Yamakawa Brownian dynamics."""
  position: Array
  mobility_position: Array
  pse_state: hydro_pse.PseState
  rng: Array


@dataclasses.dataclass
class ShearedRPYState:
  """State for PSE RPY dynamics under shear."""
  position: Array
  mobility_position: Array
  pse_state: hydro_pse.PseState
  rng: Array
  time: float
  force: Array


def rpy(space_fns: Tuple[Callable, ...],
        energy_or_force: Callable[..., Array],
        dt: float,
        kT: float,
        *,
        a: float,
        xi: float,
        eta: float,
        Mr_params: Optional[Dict[str, Any]] = None,
        Mw_params: Optional[Dict[str, Any]] = None,
        rcut: Optional[float] = None,
        P: Optional[int] = None,
        Mgrid: Optional[int] = None,
        theta: Optional[float] = None,
        mr_iters: int = 10,
        real_space_first: bool = True) -> Simulator:
  """Brownian dynamics with hydrodynamic interactions via Spectral Ewald."""
  if len(space_fns) < 2:
    raise ValueError("space_fns must contain displacement and shift functions.")

  # Unpack space functions and optional box function for real<->fractional transforms.
  displacement_fn, shift_fn = space_fns[:2]
  box_fn = space_fns[2] if len(space_fns) > 2 else None
  force_fn = quantity.canonicalize_force(energy_or_force)
  (_dt,) = static_cast(dt)

  Mw_params_local = dict(Mw_params) if Mw_params is not None else {}
  if 'fused_wave' not in Mw_params_local and 'fused' not in Mw_params_local:
    Mw_params_local['fused_wave'] = True

  pse_init, pse_apply = hydro_pse.build_pse_mobility(
      space_fns,
      a,
      xi,
      eta,
      Mr_params=Mr_params,
      Mw_params=Mw_params_local,
      rcut=rcut,
      P=P,
      Mgrid=Mgrid,
      theta=theta,
      real_space_first=real_space_first,
  )

  def init_fn(key, R, **kwargs):
    mobility_position = jnp.asarray(R)
    pse_state = pse_init(mobility_position, **kwargs)
    box_matrix = pse_state.real.box_matrix
    position_real = space.transform(box_matrix, mobility_position)
    return RPYState(position=position_real,
                    mobility_position=mobility_position,
                    pse_state=pse_state,
                    rng=key)

  def apply_fn(state, **kwargs):
    step_kwargs = dict(kwargs)
    _kT = step_kwargs.pop('kT', kT)
    _mr_iters = step_kwargs.pop('mr_iters', mr_iters)

    R_mobility = state.mobility_position
    R_real = state.position
    pse_state = state.pse_state
    key = state.rng

    force = force_fn(R_mobility, **step_kwargs)
    apply_with_brownian = getattr(pse_apply, 'with_brownian', None)
    if apply_with_brownian is not None:
      key, brownian_key = random.split(key)
      velocities, dB, pse_state, _, _, _ = apply_with_brownian(
          pse_state,
          R_mobility,
          force,
          brownian_key,
          kT=_kT,
          dt=_dt,
          mr_iters=_mr_iters,
          **step_kwargs,
      )
    else:
      velocities, pse_state = pse_apply(pse_state, R_mobility, force, **step_kwargs)
      key, noise_key = random.split(key)
      dB = hydro_pse.brownian_increment(
          noise_key, pse_state, R_mobility, kT=_kT, dt=_dt, mr_iters=_mr_iters
      )
    # Combine deterministic and stochastic increments consistently in REAL coordinates.
    # periodic_general with fractional_coordinates=True expects REAL displacement.
    # NOTE: PSE mobility already returns velocities and dB in real coordinates,
    # so no transformation is needed.
    displacement_real = _dt * velocities + dB
    R_mobility = shift_fn(R_mobility, displacement_real, **step_kwargs)
    R_real = R_real + displacement_real

    return RPYState(position=R_real,
                    mobility_position=R_mobility,
                    pse_state=pse_state,
                    rng=key)

  return init_fn, apply_fn


def rpy_with_shear(space_fns: Tuple[Callable, ...],
                   energy_or_force: Callable[..., Array],
                   dt: float,
                   kT: float,
                   *,
                   a: float,
                   xi: float,
                   eta: float,
                   shear_schedule: Union[Callable[[Array], Array], Dict[str, Callable[[Array], Array]], None],
                   t0: float = 0.0,
                   fractional_coordinates: bool = True,
                   remap: bool = True,
                   pair_energy_for_stress: Optional[Callable[..., Array]] = None,
                   Mr_params: Optional[Dict[str, Any]] = None,
                   Mw_params: Optional[Dict[str, Any]] = None,
                   rcut: Optional[float] = None,
                   P: Optional[int] = None,
                   Mgrid: Optional[int] = None,
                   theta: Optional[float] = None,
                   mr_iters: int = 10,
                   real_space_first: bool = True) -> Simulator:
  """Spectral-Ewald RPY dynamics with the shearing box utilities.

  Stress diagnostics are not accumulated inside the integrator; derive a stress
  callable with `rheo.make_pairwise_stress_fn` and evaluate it externally if
  required.
  """
  if len(space_fns) < 3:
    raise ValueError(
        "rpy_with_shear expects (displacement, shift, box_of) from space.shearing."
    )

  _, shift_fn, box_of = space_fns[:3]
  force_fn = quantity.canonicalize_force(energy_or_force)

  _dt = f32(dt)
  t0 = f32(t0)

  box = box_of(t=t0)
  dim = box.shape[0]

  sf_xy, sf_xz, sf_yz = _normalize_shear_schedule(shear_schedule)

  Mw_params_local = dict(Mw_params) if Mw_params is not None else {}
  if 'fused_wave' not in Mw_params_local and 'fused' not in Mw_params_local:
    Mw_params_local['fused_wave'] = True

  pse_init, pse_apply = hydro_pse.build_pse_mobility(
      space_fns,
      a,
      xi,
      eta,
      Mr_params=Mr_params,
      Mw_params=Mw_params_local,
      rcut=rcut,
      P=P,
      Mgrid=Mgrid,
      theta=theta,
      real_space_first=real_space_first,
  )

  def _reduced_shear(time):
    gamma_xy = f32(sf_xy(time))
    if dim >= 3:
      gamma_xz = f32(sf_xz(time))
      gamma_yz = f32(sf_yz(time))
    else:
      gamma_xz = gamma_yz = f32(0.0)
    if remap:
      m_xy = jnp.floor(gamma_xy + 0.5)
      if dim >= 3:
        m_xz = jnp.floor(gamma_xz + 0.5)
        m_yz = jnp.floor(gamma_yz + 0.5)
      else:
        m_xz = m_yz = f32(0.0)
      return (
          gamma_xy - m_xy,
          gamma_xz - m_xz,
          gamma_yz - m_yz,
          m_xy.astype(jnp.int32),
          m_xz.astype(jnp.int32),
          m_yz.astype(jnp.int32),
      )
    return (
        gamma_xy,
        gamma_xz,
        gamma_yz,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

  def init_fn(key, R, **kwargs):
    mobility_position = jnp.asarray(R)
    shear_kwargs = dict(kwargs)

    curr_xy, curr_xz, curr_yz, _, _, _ = _reduced_shear(t0)
    if dim >= 3:
      shear_kwargs.update({'gamma_xy': curr_xy, 'gamma_xz': curr_xz, 'gamma_yz': curr_yz})
    else:
      shear_kwargs['gamma'] = curr_xy

    pse_state = pse_init(mobility_position, **shear_kwargs)
    F0 = force_fn(mobility_position, **shear_kwargs)

    if fractional_coordinates:
      box_matrix0 = pse_state.real.box_matrix
      position_real0 = space.transform(box_matrix0, mobility_position)
    else:
      position_real0 = mobility_position

    return ShearedRPYState(position=position_real0,
                           mobility_position=mobility_position,
                           pse_state=pse_state,
                           rng=key,
                           time=t0,
                           force=F0)

  def apply_fn(state, **kwargs):
    step_kwargs = dict(kwargs)
    _kT = step_kwargs.pop('kT', kT)
    _mr_iters = step_kwargs.pop('mr_iters', mr_iters)

    mobility_position = state.mobility_position
    position_real = state.position
    pse_state = state.pse_state
    key = state.rng
    time = state.time

    if remap:
      prev_xy = f32(sf_xy(time))
      curr_xy = f32(sf_xy(time + _dt))
      if mobility_position.shape[1] >= 3:
        prev_xz = f32(sf_xz(time)); curr_xz = f32(sf_xz(time + _dt))
        prev_yz = f32(sf_yz(time)); curr_yz = f32(sf_yz(time + _dt))
      else:
        prev_xz = curr_xz = prev_yz = curr_yz = f32(0.0)

      m_prev_xy = jnp.floor(prev_xy + 0.5); m_curr_xy = jnp.floor(curr_xy + 0.5)
      m_xy = (m_curr_xy - m_prev_xy).astype(jnp.int32)
      if mobility_position.shape[1] >= 3:
        m_prev_xz = jnp.floor(prev_xz + 0.5); m_curr_xz = jnp.floor(curr_xz + 0.5)
        m_prev_yz = jnp.floor(prev_yz + 0.5); m_curr_yz = jnp.floor(curr_yz + 0.5)
        m_xz = (m_curr_xz - m_prev_xz).astype(jnp.int32)
        m_yz = (m_curr_yz - m_prev_yz).astype(jnp.int32)
      else:
        m_xz = m_yz = jnp.array(0, dtype=jnp.int32)

      curr_xy = curr_xy - m_curr_xy
      if mobility_position.shape[1] >= 3:
        curr_xz = curr_xz - m_curr_xz
        curr_yz = curr_yz - m_curr_yz

      if fractional_coordinates:
        if mobility_position.shape[1] >= 3:
          mxy = jnp.asarray(m_xy, mobility_position.dtype)
          mxz = jnp.asarray(m_xz, mobility_position.dtype)
          myz = jnp.asarray(m_yz, mobility_position.dtype)
          def _apply_3d_remap(Rin):
            add_x = mxy * Rin[:, 1] + (mxz + mxy * myz) * Rin[:, 2]
            add_y = myz * Rin[:, 2]
            Rout = Rin.at[:, 0].add(add_x).at[:, 1].add(add_y)
            return jnp.mod(Rout, 1.0)
          any_flip = jnp.not_equal(m_xy, 0) | jnp.not_equal(m_xz, 0) | jnp.not_equal(m_yz, 0)
          mobility_position = lax.cond(any_flip, _apply_3d_remap, lambda Rin: Rin, mobility_position)
        else:
          mxy = jnp.asarray(m_xy, mobility_position.dtype)
          def _apply_2d_remap(Rin):
            Rout = Rin.at[:, 0].add(mxy * Rin[:, 1])
            return jnp.mod(Rout, 1.0)
          any_flip = jnp.not_equal(m_xy, 0)
          mobility_position = lax.cond(any_flip, _apply_2d_remap, lambda Rin: Rin, mobility_position)
    else:
      curr_xy = f32(sf_xy(time + _dt))
      if mobility_position.shape[1] >= 3:
        curr_xz = f32(sf_xz(time + _dt))
        curr_yz = f32(sf_yz(time + _dt))
      else:
        curr_xz = curr_yz = f32(0.0)

    if mobility_position.shape[1] >= 3:
      step_kwargs.update({'gamma_xy': curr_xy, 'gamma_xz': curr_xz, 'gamma_yz': curr_yz})
    else:
      step_kwargs['gamma'] = curr_xy

    force = force_fn(mobility_position, **step_kwargs)

    if mobility_position.shape[1] >= 3:
      current_box = box_of(gamma={'xy': curr_xy, 'xz': curr_xz, 'yz': curr_yz})
    else:
      current_box = box_of(gamma=curr_xy)

    apply_with_brownian = getattr(pse_apply, 'with_brownian', None)
    if apply_with_brownian is not None:
      key, brownian_key = random.split(key)
      velocities, dB, pse_state = apply_with_brownian(
          pse_state,
          mobility_position,
          force,
          brownian_key,
          kT=_kT,
          dt=_dt,
          mr_iters=_mr_iters,
          **step_kwargs,
      )
    else:
      velocities, pse_state = pse_apply(pse_state, mobility_position, force, **step_kwargs)
      key, noise_key = random.split(key)
      dB = hydro_pse.brownian_increment(
          noise_key, pse_state, mobility_position, kT=_kT, dt=_dt, mr_iters=_mr_iters
      )
    displacement = _dt * velocities + dB
    mobility_position = shift_fn(mobility_position, displacement, **step_kwargs)
    if fractional_coordinates:
      position_real = space.transform(current_box, mobility_position)
    else:
      position_real = mobility_position

    return ShearedRPYState(position=position_real,
                           mobility_position=mobility_position,
                           pse_state=pse_state,
                           rng=key,
                           time=time + _dt,
                           force=force)

  return init_fn, apply_fn


"""Experimental Simulations.


Below are simulation environments whose implementation is somewhat
experimental / preliminary. These environments might not be as ergonomic
as the more polished environments above.
"""


@dataclasses.dataclass
class SwapMCState:
  """A struct containing state information about a Hybrid Swap MC simulation.

  Attributes:
    md: A NVTNoseHooverState containing continuous molecular dynamics data.
    sigma: An `[n,]` array of particle radii.
    key: A JAX PRGNKey used for random number generation.
    neighbor: A NeighborList for the system.
  """
  md: NVTNoseHooverState
  sigma: Array
  key: Array
  neighbor: partition.NeighborList


# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
def hybrid_swap_mc(space_fns: space.Space,
                   energy_fn: Callable[[Array, Array], Array],
                   neighbor_fn: partition.NeighborFn,
                   dt: float,
                   kT: float,
                   t_md: float,
                   N_swap: int,
                   sigma_fn: Optional[Callable[[Array], Array]]=None
                   ) -> Simulator:
  """Simulation of Hybrid Swap Monte-Carlo.

  This code simulates the hybrid Swap Monte Carlo algorithm introduced in
  Berthier et al. [#berthier]_
  Here an NVT simulation is performed for `t_md` time and then `N_swap` MC
  moves are performed that swap the radii of randomly chosen particles. The
  random swaps are accepted with Metropolis-Hastings step. Each call to the
  step function runs molecular dynamics for `t_md` and then performs the swaps.

  Note that this code doesn't feature some of the convenience functions in the
  other simulations. In particular, there is no support for dynamics keyword
  arguments and the energy function must be a simple callable of two variables:
  the distance between adjacent particles and the diameter of the particles.
  If you want support for a better notion of potential or dynamic keyword
  arguments, please file an issue!

  Args:
    space_fns: A tuple of a displacement function and a shift function defined
      in `space.py`.
    energy_fn: A function that computes the energy between one pair of
      particles as a function of the distance between the particles and the
      diameter. This function should not have been passed to `smap.xxx`.
    neighbor_fn: A function to construct neighbor lists outlined in
      `partition.py`.
    dt: The timestep used for the continuous time MD portion of the simulation.
    kT: The temperature of heat bath that the system is coupled to during MD.
    t_md: The time of each MD block.
    N_swap: The number of swapping moves between MD blocks.
    sigma_fn: An optional function for combining radii if they are to be
      non-additive.

  Returns:
    See above.

  .. rubric:: References
  .. [#berthier] L. Berthier, E. Flenner, C. J. Fullerton, C. Scalliet, and M. Singh.
    "Efficient swap algorithms for molecular dynamics simulations of
    equilibrium supercooled liquids", J. Stat. Mech. (2019) 064004
  """
  displacement_fn, shift_fn = space_fns
  metric_fn = space.metric(displacement_fn)
  nbr_metric_fn = space.map_neighbor(metric_fn)

  md_steps = int(t_md // dt)

  # Canonicalize the argument names to be dr and sigma.
  wrapped_energy_fn = lambda dr, sigma: energy_fn(dr, sigma)
  if sigma_fn is None:
    sigma_fn = lambda si, sj: 0.5 * (si + sj)
  nbr_energy_fn = smap.pair_neighbor_list(wrapped_energy_fn,
                                          metric_fn,
                                          sigma=sigma_fn)

  nvt_init_fn, nvt_step_fn = nvt_nose_hoover(nbr_energy_fn,
                                             shift_fn,
                                             dt,
                                             kT=kT,
                                             chain_length=3)
  def init_fn(key, position, sigma, nbrs=None):
    key, sim_key = random.split(key)
    nbrs = neighbor_fn(position, nbrs)  # pytype: disable=wrong-arg-count
    md_state = nvt_init_fn(sim_key, position, neighbor=nbrs, sigma=sigma)
    return SwapMCState(md_state, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def md_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)
    md = nvt_step_fn(md, neighbor=nbrs, sigma=sigma)  # pytype: disable=wrong-keyword-args
    nbrs = neighbor_fn(md.position, nbrs)
    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def swap_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)

    N = md.position.shape[0]

    # Swap a random pair of particle radii.
    key, particle_key, accept_key = random.split(key, 3)
    ij = random.randint(particle_key, (2,), jnp.array(0), jnp.array(N))
    new_sigma = sigma.at[ij].set([sigma[ij[1]], sigma[ij[0]]])

    # Collect neighborhoods around the two swapped particles.
    nbrs_ij = nbrs.idx[ij]
    R_ij = md.position[ij]
    R_neigh = md.position[nbrs_ij]

    sigma_ij = sigma[ij][:, None]
    sigma_neigh = sigma[nbrs_ij]

    new_sigma_ij = new_sigma[ij][:, None]
    new_sigma_neigh = new_sigma[nbrs_ij]

    dR = nbr_metric_fn(R_ij, R_neigh)

    # Compute the energy before the swap.
    energy = energy_fn(dR, sigma_fn(sigma_ij, sigma_neigh))
    energy = jnp.sum(energy * (nbrs_ij < N))

    # Compute the energy after the swap.
    new_energy = energy_fn(dR, sigma_fn(new_sigma_ij, new_sigma_neigh))
    new_energy = jnp.sum(new_energy * (nbrs_ij < N))

    # Accept or reject with a metropolis probability.
    p = random.uniform(accept_key, ())
    accept_prob = jnp.minimum(1, jnp.exp(-(new_energy - energy) / kT))
    sigma = jnp.where(p < accept_prob, new_sigma, sigma)

    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def block_fn(state):
    state = lax.fori_loop(0, md_steps, md_step_fn, state)
    state = lax.fori_loop(0, N_swap, swap_step_fn, state)
    return state

  return init_fn, block_fn
# pytype: enable=wrong-arg-count
# pytype: enable=wrong-keyword-args


def temp_rescale(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 window: float,
                 fraction: float,
                 **sim_kwargs) -> Simulator:
  """Simulation using explicit velocity rescaling.

  Rescale the velocities of atoms explicitly so that the desired temperature is
  reached.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    window: Floating point number specifying the temperature window outside which 
      rescaling is performed. Measured in units of `kT`.
    fraction: Floating point number which determines the amount of rescaling 
      applied to the velocities. Takes values from 0.0-1.0.
  Returns:
    See above.

  .. rubric:: References
  .. [#berendsen84] Woodcock, L. V.
    "ISOTHERMAL MOLECULAR DYNAMICS CALCULATIONS FOR LIQUID SALTS."
    Chem. Phys. Lett. 1971, 10, 257–261.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def velocity_rescale(state, window, fraction, kT):
    """Rescale the momentum if the the difference between current and target
    temperature is more than the window"""
    kT_current = temperature(state)
    cond = jnp.abs(kT_current - kT) > window
    kT_target = kT_current - fraction*(kT_current - kT)
    lam = jnp.where(cond, jnp.sqrt(kT_target / kT_current), 1)
    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum = new_momentum)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  def apply_fn(state, **kwargs):
    state = velocity_rescale(state, window, fraction, kT)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn


def temp_berendsen(energy_or_force_fn: Callable[..., Array],
                   shift_fn: ShiftFn,
                   dt: float,
                   kT: float,
                   tau: float,
                   **sim_kwargs) -> Simulator:
  """Simulation using the Berendsen thermostat.

  Berendsen (weak coupling) thermostat rescales the velocities of atoms such
  that the desired temperature is reached. This rescaling is performed at each
  timestep (dt) and the rescaling factor is calculated using
  Eq.10 Berendsen et al. [#berendsen84]_.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    tau: A floating point number determining how fast the temperature
      is relaxed during the simulation. Measured in units of `dt`.
  Returns:
    See above.

  .. rubric:: References
  .. [#berendsen84] H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola, J. R. Haak.
    "Molecular dynamics with coupling to an external bath."
    J. Chem. Phys. 15 October 1984; 81 (8): 3684-3690.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def berendsen_update(state, tau, kT, dt):
    """Rescaling the momentum of the particle by the factor lam."""
    _kT = temperature(state)
    lam = jnp.sqrt(1 + ((dt/tau) * ((kT/_kT) - 1)))
    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum=new_momentum)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  def apply_fn(state, **kwargs):
    state = berendsen_update(state, tau, kT, dt)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn


def nvk(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float,
        kT: float,
        **sim_kwargs) -> Simulator:
  """Simulation in the NVK (isokinetic) ensemble using the Gaussian thermostat.

  Samples from the isokinetic ensemble in which the number of particles (N),
  the system volume (V), and the kinetic energy (K) are held constant. A 
  Gaussian thermostat is used for the integration and the kinetic energy is 
  held constant during the simulation. The implementation follows the steps 
  described in [#minary2003]_ and [#zhang97]_. See section 4(B) equation 
  4.12-4.17 in [#minary2003]_ for detailed description.     

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
  Returns:
    See above.

  .. rubric:: References
  .. [#minary2003] Minary, Peter and Martyna, Glenn J. and Tuckerman, Mark E.
    "Algorithms and novel applications based on the isokinetic ensemble. I. 
    Biophysical and path integral molecular dynamics"
    J. Chem. Phys., Vol. 118, No. 6, 8 February 2003.
  .. [#zhang97] Zhang, Fei.
    "Operator-splitting integrators for constant-temperature molecular dynamics"
    J. Chem. Phys. 106, 6102–6106 (1997).
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)
  dt_2 = f32(dt / 2)

  def momentum_update(state, KE):
    # eps to avoid edge cases when forces are zero
    eps = 1e-16

    # Equation 4.13 to compute a and b
    update_fn = (lambda f, p, m: f * p / m)
    a = util.high_precision_sum(update_fn(state.force, state.momentum, state.mass)) + eps
    b = util.high_precision_sum(update_fn(state.force, state.force, state.mass)) + eps
    a /= (2.0 * KE)
    b /= (2.0 * KE)

    # Equation 4.12 to compute s(t) and s_dot(t)
    b_sqrt = jnp.sqrt(b)
    s_t = ((a / b) * (jnp.cosh(dt_2 * b_sqrt) - 1.0)) + jnp.sinh(dt_2 * b_sqrt) / b_sqrt
    s_dot_t = (b_sqrt * (a / b) * jnp.sinh(dt_2 * b_sqrt)) + jnp.cosh(dt_2 * b_sqrt)

    # Get the new momentum using Equation 4.15  
    new_momentum = tree_map(lambda p, f, s, sdot: (p + f * s) / sdot,
                            state.momentum,
                            state.force,
                            s_t,
                            s_dot_t)
    return state.set(momentum=new_momentum)

  def position_update(state, shift_fn, **kwargs):
    if isinstance(shift_fn, Callable):
      shift_fn = tree_map(lambda r: shift_fn, state.position)
    # Get the new positions using Equation 4.16 (Should read r = r + dt * p / m)
    new_position = tree_map(lambda s_fn, r, v: s_fn(r, dt * v, **kwargs),
                            shift_fn,
                            state.position,
                            state.velocity)
    return state.set(position=new_position)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    # Reuse the NVEState dataclass
    state = NVEState(R, None, force_fn(R, **kwargs), mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  def apply_fn(state, **kwargs):
    _KE = kinetic_energy(state)
    state = momentum_update(state, _KE)
    state = position_update(state, shift_fn)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_update(state, _KE)
    return state
  return init_fn, apply_fn


def temp_csvr(energy_or_force_fn: Callable[..., Array],
              shift_fn: ShiftFn,
              dt: float,
              kT: float,
              tau: float,
              **sim_kwargs) -> Simulator:
  """Simulation using the canonical sampling through velocity rescaling (CSVR) thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. CSVR
  algorithmn samples the canonical distribution by rescaling the velocities
  by a appropritely chosen random factor. At each timestep (dt) the rescaling
  takes place and the rescaling factor is calculated using
  A7 Bussi et al. [#bussi2007]_. CSVR updates to the velocity are stochastic in
  nature and unlike the Berendsen thermostat it samples the true canonical
  distribution [#Braun2018]_.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    tau: A floating point number determining how fast the temperature
      is relaxed during the simulation. Measured in units of `dt`.
  Returns:
    See above.

  .. rubric:: References
  .. [#bussi2007] Bussi G, Donadio D, Parrinello M.
    "Canonical sampling through velocity rescaling."
    The Journal of chemical physics, 126(1), 014101.
  .. [#Braun2018] Efrem Braun, Seyed Mohamad Moosavi, and Berend Smit.
    "Anomalous Effects of Velocity Rescaling Algorithms: The Flying Ice Cube Effect Revisited."
    Journal of Chemical Theory and Computation 2018 14 (10), 5262-5272.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)

  def sum_noises(state, key):
    """Sum of N independent gaussian noises squared.
    Adapted from https://github.com/GiovanniBussi/StochasticVelocityRescaling
    For more details see Eq.A7 Bussi et al. [#bussi2007]_"""
    dof = quantity.count_dof(state.position) - 1
    _dtype = state.position.dtype

    if dof == 0:
      """If there are no terms return zero."""
      return 0

    elif dof == 1:
      """For a single noise term, directly calculate the square of the Gaussian
      noise value."""
      rr = random.normal(key, dtype=_dtype)
      return rr * rr

    elif dof % 2 == 0:
      """For an even number of noise terms, use the gamma-distributed random
      number generator"""
      return 2.0 * random.gamma(key, dof // 2, dtype=_dtype)

    else:
      """For an odd number of noise terms, sum two terms: one from the
      gamma-distributed generator and another from the square of a
      Gaussian-distributed random number."""
      rr = random.normal(key, dtype=_dtype)
      return 2.0 * random.gamma(key, (dof - 1) // 2, dtype=_dtype) + (rr * rr)

  def csvr_update(state, tau, kT, dt):
    """Update the momentum by an scaling factor as described by
    Eq.A7 Bussi et al. [#bussi2007]_"""
    key, split = random.split(state.rng)
    dof = quantity.count_dof(state.position)

    _kT = temperature(state)

    KE_old = dof * _kT / 2
    KE_new = dof * kT / 2

    r1 = random.normal(key, dtype=state.position.dtype)
    r2 = sum_noises(state, key)

    c1 = jnp.exp(-dt / tau)
    c2 = (1 - c1) * KE_new / KE_old / dof

    scale = c1 + (c2*((r1 * r1) + r2)) + (2 * r1 * jnp.sqrt(c1 * c2))
    lam = jnp.sqrt(scale)

    new_momentum = tree_map(lambda p: p * lam, state.momentum)
    return state.set(momentum=new_momentum, rng=key)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    # Reuse the NVTLangevinState dataclass
    state = NVTLangevinState(R, None, force_fn(R, **kwargs), mass, key)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  def apply_fn(state, **kwargs):
    state = csvr_update(state, tau, kT, dt)
    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    return state
  return init_fn, apply_fn


@dataclasses.dataclass
class HardSphereBrownianState:
  """State for hard-sphere Brownian dynamics.

  Attributes:
    position: The current position of the particles.
    mobility: The mobility of particles.
    rng: Key for random number generation.
    time: Current simulation time.
    stress: Collisional stress tensor from hard-sphere constraints at the
      latest step (shape [dim, dim], excludes ideal term).
    collided: Boolean mask of particles that collided during the latest step.
  """
  position: Array
  mobility: Array
  rng: Array
  time: float
  stress: Array
  collided: Array
  reached_max_collision_loops: Array


def brownian_hard_sphere(energy_or_force_fn: Callable[..., Array],
                         displacement_fn: space.DisplacementFn,
                         shift_fn: ShiftFn,
                         dt: float,
                         kT: float,
                         diameter: float,
                         mobility: Union[float, Array] = 1.0,
                         max_collision_loops: int = 100,
                         event_time_tol: Optional[float] = None,
                         dense_neighbor_update: bool = False,
                         shear_schedule: Union[Callable[[Array], Array], Dict[str, Callable[[Array], Array]], None] = None,
                         t0: float = 0.0,
                         fractional_coordinates: bool = True,
                         remap: bool = True,
                         box: Optional[Box] = None,
                         box_fn: Optional[Callable[..., Box]] = None) -> Simulator:
  """Hard-sphere Brownian dynamics via Strating's event-driven method under shear.

  Args:
    energy_or_force_fn: Function to calculate energy or force.
    displacement_fn: Function to compute displacements.
    shift_fn: Function to shift positions.
    dt: Time step.
    kT: Thermal energy.
    diameter: Hard sphere diameter (sigma).
    mobility: Mobility coefficient.
    max_collision_loops: Safety cap on collision events per step.
    event_time_tol: Absolute tolerance for treating near-zero collision times
      as immediate events (to avoid zero-time stalls). If None, defaults to
      1e-9 and is capped by `dt`.
    shear_schedule: Shear schedule function or dict.
    t0: Initial time.
    fractional_coordinates: If True, positions are fractional coordinates in [0, 1).
    remap: If True, use reduced shear (gamma in [-0.5, 0.5)) and remap coords at
      half-integer crossings when using fractional positions.
    dense_neighbor_update: If True and a dense neighbor list is provided,
      use local event-table updates. This assumes the displacement function is
      time-independent (e.g., no shear or other explicit time dependence).
    box: Optional constant box used to compute the collisional stress tensor.
    box_fn: Optional callable returning the box (e.g. `box_of` from
      `space.shearing`). If provided, stress is normalized by the box volume.
      Note: if you pass a `neighbor` kwarg to `apply_fn`, it may be `Dense`,
      `Sparse`, or `OrderedSparse` (from `partition.neighbor_list`). For
      time-dependent displacement (e.g., shear), keep
      `dense_neighbor_update=False` to use the global pair-time recomputation
      path.
      Important: when `neighbor` is provided, collisions are checked only for
      the listed pairs; to guarantee no overlaps, ensure the neighbor list's
      cutoff/skin is large enough to include any pair that could reach contact
      within one `dt` (or omit `neighbor` to check all pairs).

  Returns:
    A pair `(init_fn, apply_fn)`.

  Notes:
    - If `fractional_coordinates=True`, positions are stored in fractional
      coordinates in `[0, 1)^d`, but displacements / forces / velocities are
      in real space.
    - Within each `dt`, the Brownian increment is interpreted as a constant
      peculiar velocity (Strating), and overlaps are removed by processing
      elastic binary collisions in temporal order using the total relative
      velocity (peculiar + affine-from-shear).
    - With `space.shearing(..., remap=True)`, reduced shear is discontinuous at
      half-integer gamma. When using fractional coordinates, the corresponding
      unimodular coordinate remap must be applied at the crossing time to keep
      real-space trajectories continuous and avoid missed collisions.
    - The returned state includes `stress`, the collisional hard-sphere stress
      over the last step computed from collision-induced velocity kicks as
        σ = -(1 / (V dt)) Σ ((Δv_i dt) ⊗ r_c),
      where r_c = diameter * n is the separation vector at contact and
      Δv_i = v_i^* - v_i is the collision-induced jump in the peculiar velocity
      of one particle. For equal masses, Δv_rel = (v_i^* - v_j^*) - (v_i - v_j)
      satisfies Δv_rel = 2 Δv_i, so if you only track the relative change you
      should use 0.5 * Δv_rel. If no `box`/`box_fn` is provided, `stress` is
      returned as zeros.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = jnp.asarray(dt)
  diameter = jnp.asarray(diameter)
  diameter_sq = diameter ** 2
  t0 = jnp.asarray(t0, dtype=dt.dtype)

  if box is not None and box_fn is not None:
    raise ValueError(
        "brownian_hard_sphere: specify at most one of box and box_fn.")

  if box is not None:
    box_const = jnp.asarray(box)

    def _box_at_time(**kwargs):
      return box_const
  else:
    _box_at_time = box_fn

  sf_xy, sf_xz, sf_yz = _normalize_shear_schedule(shear_schedule)
  shear_at = _make_shear_at(sf_xy, sf_xz, sf_yz)

  def _affine_velocity(R, g_dot_xy, g_dot_xz, g_dot_yz):
    """Affine velocity u(R) for simple shear in 2D/3D."""
    dim = R.shape[-1]
    vx = g_dot_xy * R[..., 1]
    if dim == 2:
      zeros = jnp.zeros_like(vx)
      return jnp.stack([vx, zeros], axis=-1)
    vz = R[..., 2]
    vx = vx + g_dot_xz * vz
    vy = g_dot_yz * vz
    zeros = jnp.zeros_like(vx)
    return jnp.stack([vx, vy, zeros], axis=-1)

  def _affine_relative_velocity(dr, g_dot_xy, g_dot_xz, g_dot_yz):
    """Relative affine velocity u_i - u_j expressed via separation dr = r_i - r_j."""
    return _affine_velocity(dr, g_dot_xy, g_dot_xz, g_dot_yz)

  @jit
  def init_fn(key, R, **kwargs):
    mu = kwargs.get('mobility', mobility)
    time0 = jnp.array(t0, dtype=R.dtype)
    stress0 = jnp.zeros((R.shape[1], R.shape[1]), dtype=R.dtype)
    collided0 = jnp.zeros((R.shape[0],), dtype=bool)
    reached_max0 = jnp.array(False)
    state = HardSphereBrownianState(R, mu, key, time0, stress0, collided0, reached_max0)
    state = canonicalize_mobility(state)
    return state

  def _pair_indices_and_mask(neighbor, n_particles):
    """Return (i_idx, j_idx, mask) for all candidate collision pairs."""
    def _sort_pairs(i_idx, j_idx, mask):
      sent = jnp.asarray(n_particles, dtype=i_idx.dtype)
      i_safe = jnp.where(mask, i_idx, sent)
      j_safe = jnp.where(mask, j_idx, sent)
      key = i_safe * (n_particles + 1) + j_safe
      order = jnp.argsort(key)
      return i_idx[order], j_idx[order], mask[order]

    if neighbor is None:
      i_idx, j_idx = jnp.triu_indices(n_particles, 1)
      mask = jnp.ones_like(i_idx, dtype=bool)
      return i_idx, j_idx, mask

    idx = neighbor.idx
    if partition.is_sparse(neighbor.format):
      if idx.ndim != 2 or idx.shape[0] != 2:
        raise ValueError(
            "brownian_hard_sphere: sparse neighbor.idx must have shape "
            "[2, max_neighbors].")
      i_idx = idx[0]
      j_idx = idx[1]
      mask = i_idx < n_particles
      i_idx = jnp.where(mask, i_idx, 0)
      j_idx = jnp.where(mask, j_idx, 0)
      return _sort_pairs(i_idx, j_idx, mask)

    if idx.ndim != 2 or idx.shape[0] != n_particles:
      raise ValueError(
          "brownian_hard_sphere: dense neighbor.idx must have shape "
          "[N, max_occupancy].")
    i_broad = jnp.broadcast_to(jnp.arange(n_particles)[:, None], idx.shape)
    i_idx = i_broad.reshape(-1)
    j_idx = idx.reshape(-1)
    mask = (j_idx < n_particles) & (j_idx > i_idx)
    j_idx = jnp.where(mask, j_idx, 0)
    return _sort_pairs(i_idx, j_idx, mask)

  if event_time_tol is None:
    time_tol = jnp.asarray(1e-9, dtype=dt.dtype)
  else:
    time_tol = jnp.asarray(event_time_tol, dtype=dt.dtype)
  time_tol = jnp.minimum(time_tol, dt)
  # Used to prevent pathological near-zero event times from stalling the loop.
  NO_EVENT_TIME = jnp.asarray(1e8, dtype=dt.dtype)

  if dense_neighbor_update and shear_schedule is not None:
    raise ValueError(
        "brownian_hard_sphere: dense_neighbor_update assumes time-independent "
        "displacement; disable or remove shear_schedule.")

  def _supports_batched_displacement(dim):
    # Many displacement fns (e.g. space.periodic) accept only vector inputs,
    # while `space.shearing` explicitly supports batched inputs. We detect this
    # once at construction time so collision prediction can use the fast path.
    try:
      dummy = jax.ShapeDtypeStruct((2, dim), jnp.float32)
      out = jax.eval_shape(lambda a, b: displacement_fn(a, b, t=f32(0.0)), dummy, dummy)
      return out.shape == (2, dim)
    except Exception:
      return False

  _batched_disp_2d = _supports_batched_displacement(2)
  _batched_disp_3d = _supports_batched_displacement(3)

  @jit
  def apply_fn(state, **kwargs):
    step_kwargs = dict(kwargs)
    _kT = step_kwargs.pop('kT', kT)

    R_start, mu, key, time, _, _, _ = dataclasses.astuple(state)
    dim = R_start.shape[1]
    N = R_start.shape[0]
    t_zero = jnp.asarray(0.0, dtype=dt.dtype)

    neighbor = step_kwargs.get('neighbor', None)

    # --- Shear rate (unwrapped, for affine velocity) ---
    gamma_xy_0, gamma_xz_0, gamma_yz_0 = shear_at(time, dim)
    gamma_xy_1, gamma_xz_1, gamma_yz_1 = shear_at(time + dt, dim)
    g_dot_xy = (gamma_xy_1 - gamma_xy_0) / dt
    g_dot_xz = (gamma_xz_1 - gamma_xz_0) / dt
    g_dot_yz = (gamma_yz_1 - gamma_yz_0) / dt

    # --- Reduced shear (for force kernels / consistency with shearing remap) ---
    if remap:
      curr_xy, curr_xz, curr_yz, m_prev_xy, m_prev_xz, m_prev_yz = _reduce_shear_strain(
          gamma_xy_0, gamma_xz_0, gamma_yz_0, dim)
      _, _, _, m_curr_xy, m_curr_xz, m_curr_yz = _reduce_shear_strain(
          gamma_xy_1, gamma_xz_1, gamma_yz_1, dim)
      m_xy = (m_curr_xy - m_prev_xy).astype(jnp.int32)
      m_xz = (m_curr_xz - m_prev_xz).astype(jnp.int32)
      m_yz = (m_curr_yz - m_prev_yz).astype(jnp.int32)
    else:
      curr_xy, curr_xz, curr_yz = gamma_xy_0, gamma_xz_0, gamma_yz_0
      m_xy = m_xz = m_yz = jnp.array(0, dtype=jnp.int32)

    # Guard against multiple remap crossings in a single step. This is rare and
    # can be avoided by reducing dt (or disabling remap).
    if remap and fractional_coordinates:
      too_many = (jnp.abs(m_xy) > 1) | (jnp.abs(m_xz) > 1) | (jnp.abs(m_yz) > 1)

      def _remap_error(flag):
        if flag:
          raise RuntimeError(
              "brownian_hard_sphere: dt crosses multiple shear remap boundaries; "
              "reduce dt or set remap=False.")

      jax.debug.callback(_remap_error, too_many)

    # Setup kwargs for force/energy at the start of the step.
    force_kwargs = dict(step_kwargs)
    if dim >= 3:
      force_kwargs.update({'gamma_xy': curr_xy, 'gamma_xz': curr_xz, 'gamma_yz': curr_yz})
    else:
      force_kwargs['gamma'] = curr_xy
    force_kwargs['t'] = time
    
    # --- Construct Strating "velocity" from force + noise over dt ---
    F = force_fn(R_start, **force_kwargs)
    mobility_arr = jnp.asarray(mu, dtype=R_start.dtype)

    key, split = random.split(key)
    xi = random.normal(split, R_start.shape, R_start.dtype)
    dR_noise = jnp.sqrt(f32(2) * mobility_arr * _kT * dt) * xi
    v_peculiar = (mobility_arr * F * dt + dR_noise) / dt

    # --- Collisional stress bookkeeping ---
    box_for_volume = step_kwargs.get('box', None)
    if box_for_volume is None and _box_at_time is not None:
      box_for_volume = _box_at_time(t=time)

    compute_stress = box_for_volume is not None
    if compute_stress:
      box_for_volume = jnp.asarray(box_for_volume, dtype=R_start.dtype)
      volume = quantity.volume(dim, box_for_volume)
    else:
      volume = f32(1.0)

    stress_zero = jnp.zeros((dim, dim), dtype=R_start.dtype)
    collided_zero = jnp.zeros((N,), dtype=bool)

    # --- Shear-remap discontinuities (fractional positions only) ---
    if remap and fractional_coordinates:
      remap_eps = f32(1e-12)

      def _remap_cross_time(g0, gdot, m_prev, m_step):
        # Crossing occurs when floor(g + 0.5) changes, i.e. at half-integers.
        m_step_f = jnp.asarray(m_step, dtype=g0.dtype)
        sign = jnp.sign(m_step_f)
        g_cross = jnp.asarray(m_prev, dtype=g0.dtype) + f32(0.5) * sign
        safe_gdot = jnp.where(jnp.abs(gdot) > f32(0.0), gdot, f32(1.0))
        t_cross = (g_cross - g0) / safe_gdot
        t_cross = jnp.clip(t_cross, f32(0.0), dt)
        return jnp.where(m_step != 0, t_cross, NO_EVENT_TIME)

      t_cross_xy = _remap_cross_time(gamma_xy_0, g_dot_xy, m_prev_xy, m_xy)
      if dim >= 3:
        t_cross_xz = _remap_cross_time(gamma_xz_0, g_dot_xz, m_prev_xz, m_xz)
        t_cross_yz = _remap_cross_time(gamma_yz_0, g_dot_yz, m_prev_yz, m_yz)
      else:
        t_cross_xz = t_cross_yz = NO_EVENT_TIME

      pending_xy = m_xy != 0
      pending_xz = (dim >= 3) & (m_xz != 0)
      pending_yz = (dim >= 3) & (m_yz != 0)
    else:
      remap_eps = f32(0.0)
      t_cross_xy = t_cross_xz = t_cross_yz = NO_EVENT_TIME
      pending_xy = jnp.array(False)
      pending_xz = jnp.array(False)
      pending_yz = jnp.array(False)

    # --- Event-driven collision loop ---
    # Dense neighbor lists enable local time-table updates; sparse/none fall
    # back to the global pair-time recomputation path.
    use_dense_neighbor = (
        dense_neighbor_update and
        (neighbor is not None) and
        (not partition.is_sparse(neighbor.format))
    )
    use_batched_disp = _batched_disp_3d if dim >= 3 else _batched_disp_2d

    def _predict_times(Ra, Rb, Va, Vb, abs_t):
      shape = Ra.shape[:-1]
      ra_flat = Ra.reshape(-1, dim)
      rb_flat = Rb.reshape(-1, dim)
      va_flat = Va.reshape(-1, dim)
      vb_flat = Vb.reshape(-1, dim)

      if use_batched_disp:
        dr_flat = displacement_fn(ra_flat, rb_flat, t=abs_t)
      else:
        dr_flat = jax.vmap(lambda a, b: displacement_fn(a, b, t=abs_t))(ra_flat, rb_flat)

      dv_flat = (va_flat - vb_flat) + _affine_relative_velocity(
          dr_flat, g_dot_xy, g_dot_xz, g_dot_yz)

      a = jnp.sum(dv_flat * dv_flat, axis=-1)
      b = 2 * jnp.sum(dr_flat * dv_flat, axis=-1)
      c = jnp.sum(dr_flat * dr_flat, axis=-1) - diameter_sq
      delta = b**2 - 4 * a * c

      is_overlapping = c < 0
      is_approaching = b < 0
      valid_quad = (delta >= 0) & (a > f32(1e-12))

      safe_delta = jnp.where(valid_quad, delta, f32(1.0))
      safe_a = jnp.where(valid_quad, a, f32(1.0))
      t_hit = (-b - jnp.sqrt(safe_delta)) / (2 * safe_a)
      t_hit = jnp.where(valid_quad, t_hit, NO_EVENT_TIME)
      t_hit = jnp.where(is_approaching & (t_hit >= 0), t_hit, NO_EVENT_TIME)

      resolve_now = is_overlapping | (is_approaching & (t_hit < time_tol))
      t_event = jnp.where(resolve_now, time_tol, t_hit)
      t_event = jnp.where(t_event >= time_tol, t_event, NO_EVENT_TIME)
      return t_event.reshape(shape)

    def _advance_no_collision(loop_init):
      def loop_cond(loop_state):
        t_curr, _, _, _, _, count, *_ = loop_state
        return (t_curr < dt - time_tol) & (count < max_collision_loops)

      def loop_body(loop_state):
        t_curr, R_curr, V_curr, stress_accum, collided_accum, count, pend_xy, pend_xz, pend_yz = loop_state

        pend_any = pend_xy | pend_xz | pend_yz
        t_next_remap = jnp.minimum(
            jnp.minimum(
                jnp.where(pend_xy, t_cross_xy, NO_EVENT_TIME),
                jnp.where(pend_xz, t_cross_xz, NO_EVENT_TIME),
            ),
            jnp.where(pend_yz, t_cross_yz, NO_EVENT_TIME),
        )

        do_remap_now = pend_any & (t_curr >= t_next_remap - remap_eps)

        def _apply_remap_event(_):
          apply_xy = pend_xy & (jnp.abs(t_cross_xy - t_next_remap) <= remap_eps)
          apply_xz = pend_xz & (jnp.abs(t_cross_xz - t_next_remap) <= remap_eps)
          apply_yz = pend_yz & (jnp.abs(t_cross_yz - t_next_remap) <= remap_eps)

          m_apply_xy = jnp.where(apply_xy, m_xy, jnp.array(0, dtype=jnp.int32))
          m_apply_xz = jnp.where(apply_xz, m_xz, jnp.array(0, dtype=jnp.int32))
          m_apply_yz = jnp.where(apply_yz, m_yz, jnp.array(0, dtype=jnp.int32))

          R_remap = _apply_fractional_shear_remap(R_curr, m_apply_xy, m_apply_xz, m_apply_yz, dim)
          t_new = jnp.minimum(t_next_remap + remap_eps, dt)

          return (
              t_new,
              R_remap,
              V_curr,
              stress_accum,
              collided_accum,
              count + 1,
              pend_xy & (~apply_xy),
              pend_xz & (~apply_xz),
              pend_yz & (~apply_yz),
          )

        def _advance_linear(_):
          step_limit_remap = jnp.where(pend_any, t_next_remap - remap_eps, dt)
          step = jnp.minimum(dt - t_curr, step_limit_remap - t_curr)
          step = jnp.maximum(step, f32(0.0))

          dR_step = V_curr * step
          if not fractional_coordinates:
            dR_step = dR_step + _affine_velocity(R_curr, g_dot_xy, g_dot_xz, g_dot_yz) * step

          kwargs_step = dict(step_kwargs)
          kwargs_step.pop('gamma', None)
          kwargs_step.pop('gamma_xy', None)
          kwargs_step.pop('gamma_xz', None)
          kwargs_step.pop('gamma_yz', None)
          kwargs_step['t'] = time + t_curr + step

          R_next = shift_fn(R_curr, dR_step, **kwargs_step)

          return (
              t_curr + step,
              R_next,
              V_curr,
              stress_accum,
              collided_accum,
              count + 1,
              pend_xy,
              pend_xz,
              pend_yz,
          )

        return lax.cond(do_remap_now, _apply_remap_event, _advance_linear, operand=None)

      return lax.while_loop(loop_cond, loop_body, loop_init)

    if use_dense_neighbor and neighbor.idx.shape[1] == 0:
      use_dense_neighbor = False

    if use_dense_neighbor:
      neighbor_idx = neighbor.idx
      if neighbor_idx.ndim != 2 or neighbor_idx.shape[0] != N:
        raise ValueError(
            "brownian_hard_sphere: dense neighbor.idx must have shape "
            "[N, max_occupancy].")
      max_occupancy = neighbor_idx.shape[1]
      neighbor_mask = neighbor_idx < N
      self_idx = jnp.arange(N)[:, None]
      neighbor_mask = neighbor_mask & (neighbor_idx != self_idx)
      neighbor_idx = jnp.where(neighbor_mask, neighbor_idx, 0)

      def _build_event_tables(R_curr, V_curr, abs_t, t_curr, stamps):
        Rj = R_curr[neighbor_idx]
        Vj = V_curr[neighbor_idx]
        Ri = jnp.broadcast_to(R_curr[:, None, :], Rj.shape)
        Vi = jnp.broadcast_to(V_curr[:, None, :], Vj.shape)
        times_rel = _predict_times(Ri, Rj, Vi, Vj, abs_t)
        times_abs = jnp.where(neighbor_mask, t_curr + times_rel, NO_EVENT_TIME)
        invalid_stamp = jnp.array(-1, dtype=stamps.dtype)
        neigh_stamp = jnp.where(neighbor_mask, stamps[neighbor_idx], invalid_stamp)
        return times_abs, neigh_stamp

      def _update_rows(rows, R_curr, V_curr, abs_t, t_curr, stamps, times, neigh_stamp):
        nbrs = neighbor_idx[rows]
        mask = neighbor_mask[rows]
        nbrs_safe = jnp.where(mask, nbrs, 0)
        Rb = R_curr[nbrs_safe]
        Vb = V_curr[nbrs_safe]
        Ra = jnp.broadcast_to(R_curr[rows][:, None, :], Rb.shape)
        Va = jnp.broadcast_to(V_curr[rows][:, None, :], Vb.shape)
        times_rel = _predict_times(Ra, Rb, Va, Vb, abs_t)
        times_abs = jnp.where(mask, t_curr + times_rel, NO_EVENT_TIME)
        invalid_stamp = jnp.array(-1, dtype=stamps.dtype)
        neigh_rows = jnp.where(mask, stamps[nbrs_safe], invalid_stamp)
        times = times.at[rows].set(times_abs)
        neigh_stamp = neigh_stamp.at[rows].set(neigh_rows)
        return times, neigh_stamp

      stamps0 = jnp.zeros((N,), dtype=jnp.int32)
      times0, neigh_stamp0 = _build_event_tables(R_start, v_peculiar, time, t_zero, stamps0)

      loop_init = (
          t_zero,
          R_start,
          v_peculiar,
          stress_zero,
          collided_zero,
          0,
          pending_xy,
          pending_xz,
          pending_yz,
          stamps0,
          times0,
          neigh_stamp0,
      )

      def loop_cond(loop_state):
        t_curr, _, _, _, _, count, *_ = loop_state
        return (t_curr < dt - time_tol) & (count < max_collision_loops)

      def loop_body(loop_state):
        (t_curr, R_curr, V_curr, stress_accum, collided_accum, count,
         pend_xy, pend_xz, pend_yz, stamps, times, neigh_stamp) = loop_state

        pend_any = pend_xy | pend_xz | pend_yz
        t_next_remap = jnp.minimum(
            jnp.minimum(
                jnp.where(pend_xy, t_cross_xy, NO_EVENT_TIME),
                jnp.where(pend_xz, t_cross_xz, NO_EVENT_TIME),
            ),
            jnp.where(pend_yz, t_cross_yz, NO_EVENT_TIME),
        )

        do_remap_now = pend_any & (t_curr >= t_next_remap - remap_eps)

        def _apply_remap_event(_):
          apply_xy = pend_xy & (jnp.abs(t_cross_xy - t_next_remap) <= remap_eps)
          apply_xz = pend_xz & (jnp.abs(t_cross_xz - t_next_remap) <= remap_eps)
          apply_yz = pend_yz & (jnp.abs(t_cross_yz - t_next_remap) <= remap_eps)

          m_apply_xy = jnp.where(apply_xy, m_xy, jnp.array(0, dtype=jnp.int32))
          m_apply_xz = jnp.where(apply_xz, m_xz, jnp.array(0, dtype=jnp.int32))
          m_apply_yz = jnp.where(apply_yz, m_yz, jnp.array(0, dtype=jnp.int32))

          R_remap = _apply_fractional_shear_remap(R_curr, m_apply_xy, m_apply_xz, m_apply_yz, dim)

          t_new = jnp.minimum(t_next_remap + remap_eps, dt)
          times_new, neigh_stamp_new = _build_event_tables(
              R_remap, V_curr, time + t_new, t_new, stamps)

          return (
              t_new,
              R_remap,
              V_curr,
              stress_accum,
              collided_accum,
              count,
              pend_xy & (~apply_xy),
              pend_xz & (~apply_xz),
              pend_yz & (~apply_yz),
              stamps,
              times_new,
              neigh_stamp_new,
          )

        def _advance_to_next_event(_):
          abs_t = time + t_curr

          neighbor_stamp = stamps[neighbor_idx]
          valid = neighbor_mask & (neigh_stamp == neighbor_stamp)
          time_ok = times >= (t_curr + time_tol)
          times_valid = jnp.where(valid & time_ok, times, NO_EVENT_TIME)

          min_t_abs = jnp.min(times_valid)
          flat_idx = jnp.argmin(times_valid)
          coll_i = flat_idx // max_occupancy
          coll_k = flat_idx - coll_i * max_occupancy
          coll_j = neighbor_idx[coll_i, coll_k]

          step_limit_remap = jnp.where(pend_any, t_next_remap - remap_eps, dt)
          t_target = jnp.minimum(min_t_abs, jnp.minimum(step_limit_remap, dt))
          step = jnp.maximum(t_target - t_curr, f32(0.0))

          dR_step = V_curr * step
          if not fractional_coordinates:
            dR_step = dR_step + _affine_velocity(R_curr, g_dot_xy, g_dot_xz, g_dot_yz) * step

          kwargs_step = dict(step_kwargs)
          kwargs_step.pop('gamma', None)
          kwargs_step.pop('gamma_xy', None)
          kwargs_step.pop('gamma_xz', None)
          kwargs_step.pop('gamma_yz', None)
          kwargs_step['t'] = abs_t + step

          R_next = shift_fn(R_curr, dR_step, **kwargs_step)
          t_next = t_curr + step

          is_coll = ((min_t_abs <= dt + time_tol) &
                     (min_t_abs <= step_limit_remap + time_tol) &
                     (min_t_abs <= t_next + time_tol))

          def _apply_elastic_collision(v_arr):
            ii = coll_i
            jj = coll_j
            p_i = R_next[ii]
            p_j = R_next[jj]
            v_i = v_arr[ii]
            v_j = v_arr[jj]

            t_coll = abs_t + step
            dr = displacement_fn(p_i, p_j, t=t_coll)
            dist = space.distance(dr)
            n = dr / (dist + f32(1e-7))

            dv_aff = _affine_relative_velocity(dr, g_dot_xy, g_dot_xz, g_dot_yz)
            dv_dot_n = jnp.dot((v_i - v_j) + dv_aff, n)
            impulse = jnp.where(dv_dot_n < 0, dv_dot_n, f32(0.0))

            v_i_new = v_i - impulse * n
            v_j_new = v_j + impulse * n
            v_next = v_arr.at[ii].set(v_i_new).at[jj].set(v_j_new)

            overlap = diameter - dist

            def _correct_positions(Rin):
              corr = f32(0.5) * overlap * n
              Ri_new = shift_fn(Rin[ii], corr, **kwargs_step)
              Rj_new = shift_fn(Rin[jj], -corr, **kwargs_step)
              return Rin.at[ii].set(Ri_new).at[jj].set(Rj_new)

            if compute_stress:
              # Use the single-particle collision kick (not the relative change)
              # and accumulate as (Δv ⊗ r) to match σ_xy ∝ Δv_x Δr_y.
              dv_ij = v_i_new - v_i
              r_contact = diameter * n
              impulse = dv_ij * dt
              stress_inc = jnp.einsum('i,j->ij', impulse, r_contact)
            else:
              stress_inc = stress_zero

            R_corr = lax.cond(overlap > 0, _correct_positions, lambda Rin: Rin, R_next)
            return v_next, R_corr, stress_inc, ii, jj

          zero_idx = jnp.array(0, dtype=neighbor_idx.dtype)
          V_next, R_step_next, stress_inc, ii, jj = lax.cond(
              is_coll,
              _apply_elastic_collision,
              lambda v_arr: (v_arr, R_next, stress_zero, zero_idx, zero_idx),
              V_curr,
          )

          collided_next = lax.cond(
              is_coll,
              lambda c: c.at[ii].set(True).at[jj].set(True),
              lambda c: c,
              collided_accum,
          )

          def _update_after_collision(args):
            v_arr, R_arr, stamps_in, times_in, neigh_in = args
            stamps_next = stamps_in.at[ii].add(1).at[jj].add(1)
            # Update rows for the collided particles and their neighbors.
            # This is required because pairs (k, ii)/(k, jj) live in row k and
            # are invalidated when ii/jj collide.
            rows_i = jnp.where(neighbor_mask[ii], neighbor_idx[ii], ii)
            rows_j = jnp.where(neighbor_mask[jj], neighbor_idx[jj], jj)
            rows = jnp.concatenate(
                [jnp.array([ii, jj], dtype=neighbor_idx.dtype), rows_i, rows_j],
                axis=0,
            )
            times_next, neigh_next = _update_rows(
                rows, R_arr, v_arr, time + t_next, t_next, stamps_next, times_in, neigh_in)
            return stamps_next, times_next, neigh_next

          stamps_next, times_next, neigh_next = lax.cond(
              is_coll,
              _update_after_collision,
              lambda args: (args[2], args[3], args[4]),
              operand=(V_next, R_step_next, stamps, times, neigh_stamp),
          )

          return (
              t_next,
              R_step_next,
              V_next,
              stress_accum + stress_inc,
              collided_next,
              count + 1,
              pend_xy,
              pend_xz,
              pend_yz,
              stamps_next,
              times_next,
              neigh_next,
          )

        return lax.cond(do_remap_now, _apply_remap_event, _advance_to_next_event, operand=None)

      (
          t_final,
          R_final,
          _,
          stress_accum,
          collided_accum,
          loop_count,
          pend_xy_f,
          pend_xz_f,
          pend_yz_f,
          _,
          _,
          _,
      ) = lax.while_loop(loop_cond, loop_body, loop_init)
    else:
      # Fallback: global pair-time recomputation.
      i_idx, j_idx, pair_mask = _pair_indices_and_mask(neighbor, N)
      if i_idx.size == 0:
        loop_init = (
            t_zero,
            R_start,
            v_peculiar,
            stress_zero,
            collided_zero,
            0,
            pending_xy,
            pending_xz,
            pending_yz,
        )
        (
            t_final,
            R_final,
            _,
            stress_accum,
            collided_accum,
            loop_count,
            pend_xy_f,
            pend_xz_f,
            pend_yz_f,
        ) = _advance_no_collision(loop_init)
      else:

        loop_init = (
            t_zero,
            R_start,
            v_peculiar,
            stress_zero,
            collided_zero,
            0,
            pending_xy,
            pending_xz,
            pending_yz,
        )

        def loop_cond(loop_state):
          t_curr, _, _, _, _, count, *_ = loop_state
          return (t_curr < dt - time_tol) & (count < max_collision_loops)

        def loop_body(loop_state):
          t_curr, R_curr, V_curr, stress_accum, collided_accum, count, pend_xy, pend_xz, pend_yz = loop_state

          pend_any = pend_xy | pend_xz | pend_yz
          t_next_remap = jnp.minimum(
              jnp.minimum(
                  jnp.where(pend_xy, t_cross_xy, NO_EVENT_TIME),
                  jnp.where(pend_xz, t_cross_xz, NO_EVENT_TIME),
              ),
              jnp.where(pend_yz, t_cross_yz, NO_EVENT_TIME),
          )

          do_remap_now = pend_any & (t_curr >= t_next_remap - remap_eps)

          def _apply_remap_event(_):
            apply_xy = pend_xy & (jnp.abs(t_cross_xy - t_next_remap) <= remap_eps)
            apply_xz = pend_xz & (jnp.abs(t_cross_xz - t_next_remap) <= remap_eps)
            apply_yz = pend_yz & (jnp.abs(t_cross_yz - t_next_remap) <= remap_eps)

            m_apply_xy = jnp.where(apply_xy, m_xy, jnp.array(0, dtype=jnp.int32))
            m_apply_xz = jnp.where(apply_xz, m_xz, jnp.array(0, dtype=jnp.int32))
            m_apply_yz = jnp.where(apply_yz, m_yz, jnp.array(0, dtype=jnp.int32))

            R_remap = _apply_fractional_shear_remap(R_curr, m_apply_xy, m_apply_xz, m_apply_yz, dim)

            t_new = jnp.minimum(t_next_remap + remap_eps, dt)

            return (
                t_new,
                R_remap,
                V_curr,
                stress_accum,
                collided_accum,
                count,
                pend_xy & (~apply_xy),
                pend_xz & (~apply_xz),
                pend_yz & (~apply_yz),
            )

          def _advance_to_next_event(_):
            dt_rem = dt - t_curr
            abs_t = time + t_curr

            Ri = R_curr[i_idx]
            Rj = R_curr[j_idx]
            Vi = V_curr[i_idx]
            Vj = V_curr[j_idx]

            if use_batched_disp:
              dr = displacement_fn(Ri, Rj, t=abs_t)
            else:
              dr = jax.vmap(lambda a, b: displacement_fn(a, b, t=abs_t))(Ri, Rj)

            dv = (Vi - Vj) + _affine_relative_velocity(dr, g_dot_xy, g_dot_xz, g_dot_yz)

            a = jnp.sum(dv * dv, axis=-1)
            b = 2 * jnp.sum(dr * dv, axis=-1)
            c = jnp.sum(dr * dr, axis=-1) - diameter_sq
            delta = b**2 - 4 * a * c

            is_overlapping = c < 0
            is_approaching = b < 0
            valid_quad = (delta >= 0) & (a > f32(1e-12))

            safe_delta = jnp.where(valid_quad, delta, f32(1.0))
            safe_a = jnp.where(valid_quad, a, f32(1.0))
            t_hit = (-b - jnp.sqrt(safe_delta)) / (2 * safe_a)
            t_hit = jnp.where(valid_quad, t_hit, NO_EVENT_TIME)
            t_hit = jnp.where(is_approaching & (t_hit >= 0), t_hit, NO_EVENT_TIME)

            resolve_now = is_overlapping | (is_approaching & (t_hit < time_tol))
            t_event = jnp.where(resolve_now, time_tol, t_hit)
            times = jnp.where(t_event >= time_tol, t_event, NO_EVENT_TIME)
            times = jnp.where(pair_mask, times, NO_EVENT_TIME)

            min_t_rel = jnp.min(times)
            coll_idx = jnp.argmin(times)

            step_limit_remap = jnp.where(pend_any, (t_next_remap - remap_eps) - t_curr, dt_rem)
            step = jnp.minimum(min_t_rel, jnp.minimum(dt_rem, step_limit_remap))

            dR_step = V_curr * step
            if not fractional_coordinates:
              dR_step = dR_step + _affine_velocity(R_curr, g_dot_xy, g_dot_xz, g_dot_yz) * step

            kwargs_step = dict(step_kwargs)
            kwargs_step.pop('gamma', None)
            kwargs_step.pop('gamma_xy', None)
            kwargs_step.pop('gamma_xz', None)
            kwargs_step.pop('gamma_yz', None)
            kwargs_step['t'] = abs_t + step

            R_next = shift_fn(R_curr, dR_step, **kwargs_step)
            t_next = t_curr + step

            is_coll = (min_t_rel <= dt_rem + time_tol) & (min_t_rel <= step + time_tol)

            def _apply_elastic_collision(v_arr):
              idx_dtype = i_idx.dtype
              ii = i_idx[coll_idx].astype(idx_dtype)
              jj = j_idx[coll_idx].astype(idx_dtype)
              p_i = R_next[ii]
              p_j = R_next[jj]
              v_i = v_arr[ii]
              v_j = v_arr[jj]

              t_coll = abs_t + step
              dr = displacement_fn(p_i, p_j, t=t_coll)
              dist = space.distance(dr)
              n = dr / (dist + f32(1e-7))

              dv_aff = _affine_relative_velocity(dr, g_dot_xy, g_dot_xz, g_dot_yz)
              dv_dot_n = jnp.dot((v_i - v_j) + dv_aff, n)
              impulse = jnp.where(dv_dot_n < 0, dv_dot_n, f32(0.0))

              v_i_new = v_i - impulse * n
              v_j_new = v_j + impulse * n
              v_next = v_arr.at[ii].set(v_i_new).at[jj].set(v_j_new)

              overlap = diameter - dist

              def _correct_positions(Rin):
                corr = f32(0.5) * overlap * n
                Ri_new = shift_fn(Rin[ii], corr, **kwargs_step)
                Rj_new = shift_fn(Rin[jj], -corr, **kwargs_step)
                return Rin.at[ii].set(Ri_new).at[jj].set(Rj_new)

              if compute_stress:
                # Use the single-particle collision kick (not the relative change)
                # and accumulate as (Δv ⊗ r) to match σ_xy ∝ Δv_x Δr_y.
                dv_ij = v_i_new - v_i
                r_contact = diameter * n
                impulse = dv_ij * dt
                stress_inc = jnp.einsum('i,j->ij', impulse, r_contact)
              else:
                stress_inc = stress_zero

              R_corr = lax.cond(overlap > 0, _correct_positions, lambda Rin: Rin, R_next)
              return v_next, R_corr, stress_inc, ii, jj

            zero_idx = jnp.array(0, dtype=i_idx.dtype)
            V_next, R_step_next, stress_inc, ii, jj = lax.cond(
                is_coll,
                _apply_elastic_collision,
                lambda v_arr: (v_arr, R_next, stress_zero, zero_idx, zero_idx),
                V_curr,
            )

            collided_next = lax.cond(
                is_coll,
                lambda c: c.at[ii].set(True).at[jj].set(True),
                lambda c: c,
                collided_accum,
            )

            return (
                t_next,
                R_step_next,
                V_next,
                stress_accum + stress_inc,
                collided_next,
                count + 1,
                pend_xy,
                pend_xz,
                pend_yz,
            )

          return lax.cond(do_remap_now, _apply_remap_event, _advance_to_next_event, operand=None)

        (
            t_final,
            R_final,
            _,
            stress_accum,
            collided_accum,
            loop_count,
            pend_xy_f,
            pend_xz_f,
            pend_yz_f,
        ) = lax.while_loop(loop_cond, loop_body, loop_init)

    def _collision_loop_error(reached_limit):
      if reached_limit:
        raise RuntimeError(
            "Hard sphere BD: exceeded max_collision_loops="
            f"{max_collision_loops} within one dt.")

    is_unfinished = (t_final < dt - time_tol) & (loop_count >= max_collision_loops)
    # Safety check: raise a Python error if the collision loop exceeds the
    # max per-step limit. This uses a host callback (requires a CPU backend)
    # and can slow GPU runs. Uncomment to enable strict checking.
    # jax.debug.callback(_collision_loop_error, is_unfinished)

    # Safety: apply any pending remaps if we exited early.
    if remap and fractional_coordinates:
      pend_any_f = pend_xy_f | pend_xz_f | pend_yz_f

      def _apply_remaining(Rin):
        m_xy_rem = jnp.where(pend_xy_f, m_xy, jnp.array(0, dtype=jnp.int32))
        m_xz_rem = jnp.where(pend_xz_f, m_xz, jnp.array(0, dtype=jnp.int32))
        m_yz_rem = jnp.where(pend_yz_f, m_yz, jnp.array(0, dtype=jnp.int32))
        return _apply_fractional_shear_remap(Rin, m_xy_rem, m_xz_rem, m_yz_rem, dim)

      R_final = lax.cond(pend_any_f, _apply_remaining, lambda Rin: Rin, R_final)

    # Convention: return the Cauchy stress (negative of the momentum flux /
    # pressure tensor), matching common rheology/Irving-Kirkwood conventions.
    stress = -stress_accum / (volume * dt) if compute_stress else stress_zero
    
    return HardSphereBrownianState(R_final, mu, key, time + dt, stress, collided_accum, is_unfinished)

  return init_fn, apply_fn
