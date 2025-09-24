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

"""
Rheological analysis tools for JAX-MD.

This module provides functions for computing rheological properties from
molecular dynamics simulations, including:
- Autocorrelation function calculations
- Maxwell model fitting (Prony series)
- Viscosity calculations using Green-Kubo formalism
- Frequency-dependent moduli calculations

All functions are JAX-compatible for efficient computation and automatic
differentiation.
"""

from typing import Tuple, Optional, Union, Dict, Any
import functools

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy import optimize
from jax.scipy.integrate import trapezoid
from jax.scipy.signal import correlate
import numpy as np
from scipy.optimize import minimize

from jax_md import util


# Type aliases
f32 = jnp.float32
f64 = jnp.float64


@functools.partial(jax.jit, static_argnames=('normalize',))
def autocorrelation_fft(data: Array, normalize: bool = False) -> Array:
    """
    Compute autocorrelation function using FFT.
    
    This is the most efficient method for computing autocorrelation functions
    and is suitable for large datasets.
    
    Args:
        data: Input time series data of shape (N,)
        normalize: If True, normalize by variance and subtract mean
        
    Returns:
        Autocorrelation function of shape (N,)
    """
    data = jnp.asarray(data, dtype=f32)
    n = data.shape[0]
    
    # Normalize data if requested
    if normalize:
        data_centered = data - jnp.mean(data)
        data_var = jnp.var(data)
    else:
        data_centered = data
        data_var = 1.0
    
    # Zero-pad to avoid circular correlation
    size = 2 * n
    data_padded = jnp.pad(data_centered, (0, n), mode='constant')
    
    # Compute FFT
    fft_data = jnp.fft.fft(data_padded)
    
    # Get power spectrum
    power_spectrum = jnp.abs(fft_data) ** 2
    
    # Compute autocorrelation via inverse FFT
    acf = jnp.fft.ifft(power_spectrum).real[:n]
    
    # Normalize by number of pairs at each lag
    normalization = jnp.arange(n, 0, -1, dtype=f32)
    acf = acf / normalization
    
    if normalize:
        acf = acf / data_var
        
    return acf


@functools.partial(jax.jit, static_argnames=('normalize',))
def autocorrelation_direct(data: Array, normalize: bool = False) -> Array:
    """
    Compute autocorrelation function using direct correlation.
    
    This method uses JAX's correlate function and is suitable for
    medium-sized datasets.
    
    Args:
        data: Input time series data of shape (N,)
        normalize: If True, normalize by variance and subtract mean
        
    Returns:
        Autocorrelation function of shape (N,)
    """
    data = jnp.asarray(data, dtype=f32)
    
    if normalize:
        data_centered = data - jnp.mean(data)
        data_var = jnp.var(data)
    else:
        data_centered = data
        data_var = 1.0
    
    # Use JAX correlate function
    acf = correlate(data_centered, data_centered, mode='full')
    acf = acf[acf.shape[0] // 2:]
    
    # Normalize by number of pairs at each lag
    normalization = jnp.arange(data.shape[0], 0, -1, dtype=f32)
    acf = acf / normalization
    
    if normalize:
        acf = acf / data_var
        
    return acf


def stress_autocorrelation(stress_tensor: Array, 
                          volume: float,
                          temperature: float,
                          components: Optional[Tuple[int, ...]] = None) -> Array:
    """
    Compute stress autocorrelation function for Green-Kubo viscosity.
    
    Args:
        stress_tensor: Stress tensor time series of shape (N, 3, 3) or (N, 6)
                      where the 6 components are [xx, yy, zz, xy, xz, yz]
        volume: System volume in consistent units
        temperature: Temperature in consistent units
        components: Which stress components to use for shear viscosity.
                   If None, uses off-diagonal components (xy, xz, yz)
                   
    Returns:
        Stress autocorrelation function
    """
    stress_tensor = jnp.asarray(stress_tensor)
    
    if stress_tensor.ndim == 3:
        # Extract off-diagonal components for shear viscosity
        if components is None:
            # For 3x3 tensor, extract xy, xz, yz components
            stress_components = jnp.stack([
                stress_tensor[:, 0, 1],  # xy
                stress_tensor[:, 0, 2],  # xz 
                stress_tensor[:, 1, 2]   # yz
            ], axis=1)
        else:
            # Use specified component indices as (i,j) pairs
            stress_components = jnp.stack([
                stress_tensor[:, components[k*2], components[k*2+1]] 
                for k in range(len(components)//2)
            ], axis=1)
    else:
        # Assume flattened format [xx, yy, zz, xy, xz, yz]
        if components is None:
            components = (3, 4, 5)  # xy, xz, yz indices
        stress_components = stress_tensor[:, jnp.array(components)]
    
    # Compute autocorrelation for each component
    acfs = jax.vmap(autocorrelation_fft, in_axes=1, out_axes=1)(stress_components)
    
    # Average over components
    acf_avg = jnp.mean(acfs, axis=1)
    
    # Apply Green-Kubo prefactor with numerical stability
    # Factor for converting to appropriate units
    if temperature <= 0 or volume <= 0:
        raise ValueError("Temperature and volume must be positive")
    
    prefactor = volume / temperature
    result = prefactor * acf_avg
    
    # Ensure finite values
    result = jnp.where(jnp.isfinite(result), result, 0.0)
    
    return result


def viscosity_integral_direct(autocorr_func: Array, time: Array) -> Array:
    """
    Compute cumulative viscosity integral directly from autocorrelation function.
    
    This function integrates the stress autocorrelation function cumulatively
    to show how the viscosity converges as a function of integration time.
    The final value in the returned array is the total viscosity.
    
    Args:
        autocorr_func: Stress autocorrelation function values
        time: Corresponding time array
        
    Returns:
        Array of cumulative viscosity integrals, where element i contains
        the integral from time[0] to time[i]
    """
    autocorr_func = jnp.asarray(autocorr_func)
    time = jnp.asarray(time)
    
    # Compute cumulative integral using trapezoidal rule
    def scan_fn(carry, i):
        # Trapezoidal rule for interval [time[i-1], time[i]]
        dt = jnp.where(i == 0, 0.0, time[i] - time[i-1])
        acf_avg = jnp.where(i == 0, 0.0, 0.5 * (autocorr_func[i-1] + autocorr_func[i]))
        integral_increment = acf_avg * dt
        new_cumulative = carry + integral_increment
        return new_cumulative, new_cumulative
    
    _, cumulative_viscosity = jax.lax.scan(scan_fn, 0.0, jnp.arange(len(time)))
    
    return cumulative_viscosity


class MaxwellModel:
    """
    Maxwell model (Prony series) for fitting stress autocorrelation functions.
    
    This class provides methods for fitting a sum of exponentials to
    autocorrelation functions and computing rheological properties.
    """
    
    def __init__(self, n_modes: int):
        """
        Initialize Maxwell model.
        
        Args:
            n_modes: Number of Maxwell modes (exponential terms)
        """
        self.n_modes = n_modes
        # For backward compatibility
        self.amplitudes = None
        self.decay_rates = None
        # New format following gk.py
        self.moduli = None
        self.tau_values = None
        
    @staticmethod
    @jax.jit
    def evaluate(params: Array, t: Array) -> Array:
        """
        Evaluate Maxwell model at given times using Prony series format.
        
        Args:
            params: Parameters array of shape (2 * n_modes,) containing
                   [G1, tau1, G2, tau2, ...] where G are moduli and tau are time constants
            t: Time array
            
        Returns:
            Model prediction at times t
        """
        n_modes = params.shape[0] // 2
        moduli = params[::2]      # G values
        tau_values = params[1::2]  # tau values (relaxation times)
        
        # Compute sum of exponentials: G(t) = Σ Gi * exp(-t/τi)
        exp_terms = moduli[:, None] * jnp.exp(-t[None, :] / tau_values[:, None])
        return jnp.sum(exp_terms, axis=0)
    
    @staticmethod
    def _objective_function(params: Array, t: Array, data: Array, 
                           use_log_space: bool = True) -> Array:
        """Objective function for least squares fitting with numerical stability."""
        # Ensure parameters are positive to avoid numerical issues
        n_modes = params.shape[0] // 2
        moduli = jnp.abs(params[::2])        # G values (must be positive)
        tau_values = jnp.abs(params[1::2])   # tau values (must be positive)
        
        # Clamp tau values to reasonable range to prevent overflow
        tau_values = jnp.clip(tau_values, 1e-6, 1e6)
        
        # Repack parameters
        stable_params = jnp.zeros_like(params)
        stable_params = stable_params.at[::2].set(moduli)
        stable_params = stable_params.at[1::2].set(tau_values)
        
        prediction = MaxwellModel.evaluate(stable_params, t)
        
        # Check for non-finite values
        prediction = jnp.where(jnp.isfinite(prediction), prediction, 0.0)
        
        # Convert to arrays to ensure proper type
        data = jnp.asarray(data)
        prediction = jnp.asarray(prediction)
        
        if use_log_space and jnp.all(data > 0) and jnp.all(prediction > 0):
            # Fit in log space for exponential data - this often works better
            log_data = jnp.log(jnp.maximum(data, 1e-12))
            log_pred = jnp.log(jnp.maximum(prediction, 1e-12))
            residuals = log_pred - log_data
        else:
            # Linear space fitting
            residuals = prediction - data
        
        objective = jnp.sum(residuals ** 2)
        
        # Return a large value if objective is not finite
        return jnp.where(jnp.isfinite(objective), objective, 1e12)
    
    def _estimate_initial_parameters(self, t: Array, data: Array) -> Array:
        """
        Estimate initial parameters using NNLS approach from gk.py.
        
        Args:
            t: Time array
            data: Data to fit
            
        Returns:
            Initial parameter guess in [G1, tau1, G2, tau2, ...] format
        """
        # Check for valid inputs
        if len(t) < 2 or len(data) < 2:
            raise ValueError("Need at least 2 data points for fitting")
        
        # Remove any non-positive time values
        valid_mask = (t > 0) & jnp.isfinite(t) & jnp.isfinite(data) & (data > 0)
        if not jnp.any(valid_mask):
            raise ValueError("No valid positive data points found")
        
        t_valid = t[valid_mask]
        data_valid = data[valid_mask]
        
        if len(t_valid) < 2:
            raise ValueError("Insufficient valid data points after filtering")
        
        # Use log-spaced tau values spanning the time range (like gk.py)
        min_tau = max(float(jnp.min(t_valid)) * 0.1, 1e-10)
        max_tau = float(jnp.max(t_valid)) * 10
        tau_est = jnp.logspace(jnp.log10(min_tau), jnp.log10(max_tau), self.n_modes)
        
        # Build design matrix A where A[i,j] = exp(-t[i]/tau[j])
        A = jnp.exp(-t_valid[:, None] / tau_est[None, :])
        
        # Verify A matrix has no NaN or inf values
        if not jnp.all(jnp.isfinite(A)):
            print("Warning: Design matrix contains non-finite values, using fallback")
            # Fallback to simple estimation
            G_est = jnp.full(self.n_modes, float(jnp.max(data_valid)) / self.n_modes)
        else:
            # Use non-negative least squares to estimate G values (like gk.py)
            from scipy.optimize import nnls
            G_est, _ = nnls(np.array(A), np.array(data_valid))
            G_est = jnp.array(G_est)
        
        # Interleave G and tau values: [G1, tau1, G2, tau2, ...]
        params = jnp.zeros(2 * self.n_modes)
        params = params.at[::2].set(G_est)
        params = params.at[1::2].set(tau_est)
        
        return params
    
    def fit(self, t: Array, data: Array, 
            initial_params: Optional[Array] = None) -> Dict[str, Any]:
        """
        Fit Maxwell model to data.
        
        Args:
            t: Time array
            data: Autocorrelation function data
            initial_params: Initial parameter guess. If None, estimated automatically.
            
        Returns:
            Dictionary containing fitted parameters and fit statistics
        """
        t = jnp.asarray(t)
        data = jnp.asarray(data)
        
        if initial_params is None:
            initial_params = self._estimate_initial_parameters(t, data)
        
        # Get data characteristics for better bounds
        max_data = float(jnp.max(data))
        t_min, t_max = float(jnp.min(t)), float(jnp.max(t))
        
        # Perform optimization using scipy since JAX optimize doesn't have bounds
        # Convert to numpy for scipy optimization
        def objective_np(params):
            try:
                params_jax = jnp.array(params)
                result = self._objective_function(params_jax, t, data, use_log_space=True)
                return float(result)
            except:
                return 1e12  # Return large value on any error
        
        # Set up better bounds based on data characteristics
        bounds_list = []
        for i in range(self.n_modes):
            # G (modulus) bounds: reasonable fraction of max data
            G_min = max_data * 1e-6
            G_max = max_data * 10  # Allow some overshoot for noisy data
            bounds_list.append((G_min, G_max))
            
            # tau (relaxation time) bounds: based on time range
            tau_min = t_min * 0.01   # Fast relaxation
            tau_max = t_max * 100    # Slow relaxation
            bounds_list.append((tau_min, tau_max))
        
        # Perform optimization using scipy with curve_fit (like gk.py)
        from scipy.optimize import curve_fit
        
        def maxwell_curve(t, *params):
            """Maxwell model function for curve_fit (like gk.py)"""
            n_modes = len(params) // 2
            Gs = params[0::2]  # G values
            taus = params[1::2]  # tau values
            result = np.zeros_like(t)
            for G, tau in zip(Gs, taus):
                result += G * np.exp(-t / tau)
            return result
        
        # Convert to numpy for scipy optimization
        t_np = np.array(t)
        data_np = np.array(data)
        initial_params_np = np.array(initial_params)
        
        # Try multiple optimization strategies with different starting points
        best_result = None
        best_objective = float('inf')
        
        methods = ['trf', 'lm', 'dogbox']  # Trust region methods work well for curve_fit
        n_tries = 5  # More random starts
        
        for method in methods:
            for trial in range(n_tries):
                try:
                    # Add some randomness to initial guess for multiple tries
                    if trial == 0:
                        p0 = initial_params_np
                    else:
                        # Perturb initial parameters more systematically
                        perturbation = np.exp(0.3 * np.random.randn(len(initial_params_np)))
                        p0 = initial_params_np * perturbation
                        # Ensure bounds are respected
                        for i, (low, high) in enumerate(bounds_list):
                            p0[i] = np.clip(p0[i], low * 1.1, high * 0.9)
                    
                    # Set up bounds for curve_fit
                    bounds_lower = [bound[0] for bound in bounds_list]
                    bounds_upper = [bound[1] for bound in bounds_list]
                    
                    fitted_params, _ = curve_fit(
                        maxwell_curve, t_np, data_np,
                        p0=p0,
                        bounds=(bounds_lower, bounds_upper),
                        method=method,
                        maxfev=5000
                    )
                    
                    # Calculate objective value to compare fits
                    pred = maxwell_curve(t_np, *fitted_params)
                    objective = np.sum((pred - data_np) ** 2)
                    
                    if objective < best_objective:
                        # Create a result-like object
                        class CurveFitResult:
                            def __init__(self, x, fun):
                                self.x = x
                                self.fun = fun
                                self.success = True
                        
                        best_result = CurveFitResult(fitted_params, objective)
                        best_objective = objective
                        
                except Exception as e:
                    continue  # Try next method/trial
                        
            # If we found a good solution, don't try other methods
            if best_result is not None and best_objective < 1e6:
                break
        
        if best_result is None:
            # If all methods fail, try a simple approach
            try:
                result = minimize(
                    fun=objective_np,
                    x0=np.array(initial_params),
                    method='Nelder-Mead',
                    options={'maxfev': 3000}
                )
                best_result = result
            except:
                # Create a dummy failed result
                class DummyResult:
                    success = False
                    x = initial_params
                    fun = 1e12
                best_result = DummyResult()
        
        # Convert back to JAX arrays
        fitted_params = jnp.array(best_result.x)
        self.amplitudes = fitted_params[::2]    # G values (for backward compatibility)
        self.decay_rates = 1.0 / fitted_params[1::2]  # Convert tau to decay rates for compatibility
        
        # Store the actual moduli and relaxation times
        self.moduli = fitted_params[::2]
        self.tau_values = fitted_params[1::2]
        
        # Compute fit statistics
        prediction = self.evaluate(fitted_params, t)
        residuals = prediction - data
        rmse = jnp.sqrt(jnp.mean(residuals ** 2))
        r_squared = 1 - jnp.var(residuals) / jnp.var(data)
        
        return {
            'amplitudes': self.amplitudes,      # G values (for compatibility)
            'decay_rates': self.decay_rates,    # 1/tau values (for compatibility) 
            'moduli': self.moduli,              # G values (actual)
            'tau_values': self.tau_values,      # tau values (actual)
            'fitted_params': fitted_params,
            'prediction': prediction,
            'residuals': residuals,
            'rmse': rmse,
            'r_squared': r_squared,
            'success': best_result.success
        }
    
    def viscosity_integral(self, t_max: Optional[float] = None) -> float:
        """
        Compute zero-shear viscosity by integrating the fitted model.
        
        Args:
            t_max: Maximum integration time. If None, integrates to infinity.
            
        Returns:
            Zero-shear viscosity
        """
        if self.moduli is None or self.tau_values is None:
            raise ValueError("Model must be fitted before computing viscosity")
        
        if t_max is None:
            # Analytical integration to infinity: ∫ G_i * exp(-t/τ_i) dt = G_i * τ_i
            viscosity = jnp.sum(self.moduli * self.tau_values)
        else:
            # Numerical integration to t_max
            t = jnp.linspace(0, t_max, 1000)
            # Use current fitted parameters in correct format [G1, tau1, G2, tau2, ...]
            params = jnp.zeros(2 * len(self.moduli))
            params = params.at[::2].set(self.moduli)
            params = params.at[1::2].set(self.tau_values)
            g_t = self.evaluate(params, t)
            viscosity = trapezoid(g_t, t)
        
        return float(viscosity)
    
    def viscosity_integral_cumulative(self, time: Array) -> Array:
        """
        Compute cumulative viscosity integral over given time array.
        
        This method returns an array where each element i contains the integral
        of the stress autocorrelation function from time 0 to time[i], allowing
        you to see how the viscosity converges as a function of integration time.
        
        Args:
            time: Time array for integration
            
        Returns:
            Array of cumulative viscosity integrals, where the final value
            is the converged viscosity
        """
        if self.moduli is None or self.tau_values is None:
            raise ValueError("Model must be fitted before computing viscosity")
        
        time = jnp.asarray(time)
        
        # Use current fitted parameters in correct format [G1, tau1, G2, tau2, ...]
        params = jnp.zeros(2 * len(self.moduli))
        params = params.at[::2].set(self.moduli)
        params = params.at[1::2].set(self.tau_values)
        
        # Evaluate the model at all time points
        g_t = self.evaluate(params, time)
        
        # Compute cumulative integral using trapezoidal rule
        # For cumulative integration, we need to compute the integral from 0 to each time point
        
        # Use JAX's scan to compute cumulative integral efficiently
        def scan_fn(carry, i):
            # Trapezoidal rule for interval [time[i-1], time[i]]
            dt = jnp.where(i == 0, 0.0, time[i] - time[i-1])
            g_avg = jnp.where(i == 0, 0.0, 0.5 * (g_t[i-1] + g_t[i]))
            integral_increment = g_avg * dt
            new_cumulative = carry + integral_increment
            return new_cumulative, new_cumulative
        
        _, cumulative_viscosity = jax.lax.scan(scan_fn, 0.0, jnp.arange(len(time)))
        
        return cumulative_viscosity
    
    def frequency_response(self, frequencies: Array) -> Tuple[Array, Array]:
        """
        Compute frequency-dependent storage and loss moduli.
        
        Args:
            frequencies: Angular frequency array
            
        Returns:
            Tuple of (storage_modulus, loss_modulus)
        """
        if self.moduli is None or self.tau_values is None:
            raise ValueError("Model must be fitted before computing frequency response")
        
        omega = jnp.asarray(frequencies)
        
        # Storage modulus G'(ω) = Σ G_i * (ω*τ_i)² / (1 + (ω*τ_i)²)
        # Loss modulus G''(ω) = Σ G_i * (ω*τ_i) / (1 + (ω*τ_i)²)
        
        omega_tau = omega[:, None] * self.tau_values[None, :]  # Shape: (n_freq, n_modes)
        omega_tau_sq = omega_tau ** 2
        
        denominator = 1 + omega_tau_sq
        
        storage_terms = self.moduli[None, :] * omega_tau_sq / denominator
        loss_terms = self.moduli[None, :] * omega_tau / denominator
        
        storage_modulus = jnp.sum(storage_terms, axis=1)
        loss_modulus = jnp.sum(loss_terms, axis=1)
        
        return storage_modulus, loss_modulus


def select_best_model(t: Array, data: Array, 
                     max_modes: int = 10,
                     criterion: str = 'bic') -> Tuple[MaxwellModel, Dict[str, Any]]:
    """
    Select the best Maxwell model using information criteria.
    
    Args:
        t: Time array
        data: Autocorrelation function data
        max_modes: Maximum number of modes to try
        criterion: Information criterion to use ('bic', 'aic')
        
    Returns:
        Tuple of (best_model, fit_results)
    """
    results = {}
    criteria_values = {}
    
    n = len(data)
    
    for n_modes in range(1, max_modes + 1):
        model = MaxwellModel(n_modes)
        
        try:
            fit_result = model.fit(t, data)
            
            if fit_result['success']:
                residual_sum_squares = jnp.sum(fit_result['residuals'] ** 2)
                n_params = 2 * n_modes
                
                if criterion.lower() == 'bic':
                    ic_value = n * jnp.log(residual_sum_squares / n) + n_params * jnp.log(n)
                elif criterion.lower() == 'aic':
                    ic_value = n * jnp.log(residual_sum_squares / n) + 2 * n_params
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                results[n_modes] = (model, fit_result)
                criteria_values[n_modes] = ic_value
                
        except Exception as e:
            print(f"Failed to fit model with {n_modes} modes: {e}")
            continue
    
    if not criteria_values:
        raise RuntimeError("No successful fits found")
    
    # Select model with minimum criterion value
    best_n_modes = min(criteria_values.keys(), key=lambda k: criteria_values[k])
    best_model, best_fit = results[best_n_modes]
    
    return best_model, {
        'best_n_modes': best_n_modes,
        'fit_result': best_fit,
        'all_criteria': criteria_values
    }


def green_kubo_viscosity(stress_tensor: Array,
                        time: Array,
                        volume: float,
                        temperature: float,
                        max_modes: int = 10,
                        components: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
    """
    Compute viscosity using Green-Kubo formalism.
    
    This is the main function that combines autocorrelation calculation,
    Maxwell model fitting, and viscosity computation.
    
    Args:
        stress_tensor: Stress tensor time series
        time: Time array
        volume: System volume
        temperature: Temperature
        max_modes: Maximum number of Maxwell modes to try
        components: Stress tensor components to use
        
    Returns:
        Dictionary containing viscosity and fitting results
    """
    # Compute stress autocorrelation function
    acf = stress_autocorrelation(stress_tensor, volume, temperature, components)
    
    # Only use positive part of ACF
    positive_mask = acf > 0
    if not jnp.any(positive_mask):
        raise ValueError("No positive autocorrelation values found")
    
    # Truncate to positive values
    first_negative = jnp.argmax(~positive_mask) if jnp.any(~positive_mask) else len(acf)
    if first_negative == 0:
        first_negative = len(acf)
    
    t_fit = time[:first_negative]
    acf_fit = acf[:first_negative]
    
    # Select best Maxwell model
    best_model, model_results = select_best_model(t_fit, acf_fit, max_modes)
    
    # Produce fitted G(t) for times
    acf_fitted = best_model.evaluate(model_results['fit_result']['fitted_params'], time)
    
    # Compute viscosity (final value)
    viscosity = best_model.viscosity_integral()
    
    # Compute cumulative viscosity integral from fitted model
    cumulative_viscosity_fitted = best_model.viscosity_integral_cumulative(time)
    
    # Compute cumulative viscosity integral directly from raw ACF
    cumulative_viscosity_raw = viscosity_integral_direct(acf, time)
    
    # Compute frequency response for a standard frequency range
    log_freq_range = jnp.logspace(-3, 3, 100)  # 10^-3 to 10^3 rad/s
    storage_modulus, loss_modulus = best_model.frequency_response(log_freq_range)
    
    return {
        'viscosity': viscosity,
        'cumulative_viscosity_fitted': cumulative_viscosity_fitted,
        'cumulative_viscosity_raw': cumulative_viscosity_raw,
        'autocorrelation_function': acf,
        'time': time,
        # 'fitted_time': t_fit,
        'fitted_acf': acf_fitted,
        'model': best_model,
        'model_results': model_results,
        'frequencies': log_freq_range,
        'storage_modulus': storage_modulus,
        'loss_modulus': loss_modulus
    }


def complex_viscosity(frequencies: Array, 
                     storage_modulus: Array, 
                     loss_modulus: Array) -> Array:
    """
    Compute complex viscosity from frequency-dependent moduli.
    
    Args:
        frequencies: Angular frequency array
        storage_modulus: Storage modulus G'(ω)
        loss_modulus: Loss modulus G''(ω)
        
    Returns:
        Complex viscosity magnitude |η*(ω)|
    """
    omega = jnp.asarray(frequencies)
    g_prime = jnp.asarray(storage_modulus)
    g_double_prime = jnp.asarray(loss_modulus)
    
    # |η*| = √(G'² + G''²) / ω
    complex_modulus_magnitude = jnp.sqrt(g_prime**2 + g_double_prime**2)
    return complex_modulus_magnitude / omega


# Utility functions for common rheological calculations

def relaxation_spectrum(model: MaxwellModel) -> Tuple[Array, Array]:
    """
    Get the discrete relaxation spectrum from a fitted Maxwell model.
    
    Args:
        model: Fitted Maxwell model
        
    Returns:
        Tuple of (relaxation_times, moduli)
    """
    if model.moduli is None or model.tau_values is None:
        raise ValueError("Model must be fitted first")
    
    return model.tau_values, model.moduli


def characteristic_times(model: MaxwellModel) -> Dict[str, float]:
    """
    Compute characteristic rheological times from a fitted model.
    
    Args:
        model: Fitted Maxwell model
        
    Returns:
        Dictionary with characteristic times
    """
    if model.moduli is None or model.tau_values is None:
        raise ValueError("Model must be fitted first")
    
    # Weight-averaged relaxation time
    total_modulus = jnp.sum(model.moduli)
    avg_relaxation_time = jnp.sum(model.moduli * model.tau_values) / total_modulus
    
    # Longest relaxation time
    longest_relaxation_time = jnp.max(model.tau_values)
    
    # Shortest relaxation time  
    shortest_relaxation_time = jnp.min(model.tau_values)
    
    return {
        'average_relaxation_time': float(avg_relaxation_time),
        'longest_relaxation_time': float(longest_relaxation_time),
        'shortest_relaxation_time': float(shortest_relaxation_time)
    }