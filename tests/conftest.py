import jax
import pytest


@pytest.fixture(scope='session', autouse=True)
def print_jax_device():
  devices = jax.devices()
  print(f"\n  JAX devices: {devices}")
  print(f"  Default backend: {jax.default_backend()}")
  yield
