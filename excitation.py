# excitation.py
# =============================================================
# Impact force pulse definition and global force vector builder
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from config import (N_NODES, N_ELEMENTS, N_STEPS, DT, PULSE_DURATION,
                     BOUNDARY_CONDITION, RANDOM_BANDWIDTH)

# -------------------------------------------------------------
# UNIT NOTE
# F0      : lbf
# tau     : seconds
# F_vector: lbf  (force at each free DOF at each timestep)
# -------------------------------------------------------------

def half_sine_pulse(F0, tau, dt, n_steps):
    """
    Generate half-sine impact pulse time history.

    F(t) = F0 * sin(pi * t / tau)   for 0 <= t <= tau
    F(t) = 0                         for t > tau

    Inputs:
        F0      : peak impact force (lbf)
        tau     : pulse duration (s)
        dt      : timestep size (s)
        n_steps : total number of timesteps

    Output:
        force_time : 1D array of force values (lbf), length = n_steps
        time_vector: 1D array of time values (s),  length = n_steps
    """
    time_vector = np.linspace(0, dt * (n_steps - 1), n_steps)
    force_time  = np.zeros(n_steps)

    # Apply half sine pulse where t <= tau
    pulse_mask             = time_vector <= tau
    force_time[pulse_mask] = F0 * np.sin(np.pi * time_vector[pulse_mask] / tau)

    return force_time, time_vector


def harmonic_load(F0, f_exc, dt, n_steps):
    """
    Generate sinusoidal force time history.

    F(t) = F0 * sin(2 * pi * f_exc * t)

    Applied for the entire simulation duration.

    Inputs:
        F0      : force amplitude (lbf)
        f_exc   : excitation frequency (Hz)
        dt      : timestep size (s)
        n_steps : total number of timesteps

    Output:
        force_time  : 1D array of force values (lbf), length = n_steps
        time_vector : 1D array of time values (s), length = n_steps
    """
    time_vector = np.linspace(0, dt * (n_steps - 1), n_steps)
    force_time  = F0 * np.sin(2.0 * np.pi * f_exc * time_vector)
    return force_time, time_vector


def random_load(F0, dt, n_steps, f_band=None, seed=None):
    """
    Generate band-limited random force time history.

    Creates Gaussian white noise, applies FFT bandpass filter to
    [f_low, f_high], then scales so that RMS amplitude equals F0.

    Inputs:
        F0      : target RMS force amplitude (lbf)
        dt      : timestep size (s)
        n_steps : total number of timesteps
        f_band  : tuple (f_low, f_high) in Hz for bandpass filter
                  None uses RANDOM_BANDWIDTH from config
        seed    : random seed for reproducibility (int or None)

    Output:
        force_time  : 1D array of force values (lbf), length = n_steps
        time_vector : 1D array of time values (s), length = n_steps
    """
    if f_band is None:
        f_band = RANDOM_BANDWIDTH

    rng = np.random.default_rng(seed)

    time_vector = np.linspace(0, dt * (n_steps - 1), n_steps)

    # Generate white noise
    noise = rng.standard_normal(n_steps)

    # Bandpass filter via FFT
    freqs    = np.fft.rfftfreq(n_steps, d=dt)
    fft_vals = np.fft.rfft(noise)

    f_low, f_high = f_band
    mask = (freqs >= f_low) & (freqs <= f_high)
    fft_vals[~mask] = 0.0

    # Inverse FFT
    force_time = np.fft.irfft(fft_vals, n=n_steps)

    # Scale to target RMS
    current_rms = np.sqrt(np.mean(force_time**2))
    if current_rms > 1e-15:
        force_time = force_time * (F0 / current_rms)

    return force_time, time_vector


def hermite_shape_functions(xi, Le):
    """
    Evaluate Hermite cubic shape functions at local coordinate xi.

    For an Euler-Bernoulli beam element of length Le, the shape functions
    interpolate transverse displacement w(xi) from nodal DOFs
    [w_i, theta_i, w_j, theta_j].

    Inputs:
        xi : local coordinate in [0, 1], where xi = (x - x_i) / Le
        Le : element length (in)

    Output:
        N  : array of 4 shape function values [N1, N2, N3, N4]
             N1 -> transverse displacement at node i
             N2 -> rotation at node i
             N3 -> transverse displacement at node j
             N4 -> rotation at node j
    """
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = Le * (xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = Le * (-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4])


def get_impact_dofs_and_weights(position_frac, L, free_dofs):
    """
    Compute force distribution DOFs and Hermite weights for an
    arbitrary impact position along the beam.

    Inputs:
        position_frac : fractional position along beam [0.0, 1.0]
                        0.0 = left end (node 0), 1.0 = right end (node N)
        L             : beam length (in)
        free_dofs     : array of free DOF indices from BC application

    Output:
        dof_weight_pairs : list of (free_dof_index, weight) tuples
    """
    Le    = L / N_ELEMENTS
    x_abs = position_frac * L

    # Determine which element the load falls in
    elem_idx = int(x_abs / Le)

    # Clamp to valid element range [0, N_ELEMENTS-1]
    if elem_idx >= N_ELEMENTS:
        elem_idx = N_ELEMENTS - 1

    # Local coordinate within element
    x_left = elem_idx * Le
    xi     = (x_abs - x_left) / Le

    # Nodes bounding this element
    node_i = elem_idx
    node_j = elem_idx + 1

    # Global DOFs for these two nodes
    global_dofs = [
        2 * node_i,       # transverse at node i
        2 * node_i + 1,   # rotation at node i
        2 * node_j,       # transverse at node j
        2 * node_j + 1,   # rotation at node j
    ]

    # Hermite weights
    N = hermite_shape_functions(xi, Le)

    # Map to free DOF indices, skipping fixed DOFs
    dof_weight_pairs = []
    for gdof, weight in zip(global_dofs, N):
        if abs(weight) < 1e-15:
            continue
        free_idx = np.where(free_dofs == gdof)[0]
        if len(free_idx) > 0:
            dof_weight_pairs.append((free_idx[0], weight))

    if len(dof_weight_pairs) == 0:
        raise ValueError(
            f"All DOFs at position {position_frac:.4f} (x={x_abs:.4f} in) "
            f"are fixed. Cannot apply load at a fully constrained location.")

    return dof_weight_pairs


def build_force_vector(F0, free_dofs, L=None,
                       tau=PULSE_DURATION, dt=DT,
                       n_steps=N_STEPS, bc_type=BOUNDARY_CONDITION,
                       loading_type='impact',
                       position_frac=None,
                       excitation_freq=None,
                       random_seed=None):
    """
    Build global force vector for all timesteps.

    Supports three loading types:
        'impact'   : half-sine pulse (original behavior)
        'harmonic' : continuous sinusoidal excitation
        'random'   : band-limited Gaussian random

    Force is applied at an arbitrary beam position using Hermite
    shape function interpolation to distribute to element DOFs.

    Inputs:
        F0             : peak/amplitude/RMS force (lbf)
        free_dofs      : array of free DOF indices (from assembly)
        L              : beam length (in), needed for Hermite interpolation
        tau            : pulse duration for impact (s)
        dt             : timestep size (s)
        n_steps        : number of timesteps
        bc_type        : boundary condition type string
        loading_type   : 'impact', 'harmonic', or 'random'
        position_frac  : fractional position along beam [0.0, 1.0]
                         None = use legacy get_impact_dof() behavior
        excitation_freq: excitation frequency for harmonic (Hz)
        random_seed    : seed for random load generation

    Output:
        F_global    : 2D array shape (n_free_dofs, n_steps)
        force_time  : 1D array of force time history at load point (lbf)
        time_vector : 1D array of time values (s)
    """
    n_free = len(free_dofs)
    F_global = np.zeros((n_free, n_steps))

    # --- Step 1: Generate time history based on loading type ---
    if loading_type == 'impact':
        force_time, time_vector = half_sine_pulse(F0, tau, dt, n_steps)
    elif loading_type == 'harmonic':
        if excitation_freq is None:
            raise ValueError("excitation_freq is required for harmonic loading")
        force_time, time_vector = harmonic_load(F0, excitation_freq, dt, n_steps)
    elif loading_type == 'random':
        force_time, time_vector = random_load(F0, dt, n_steps, seed=random_seed)
    else:
        raise ValueError(f"Unknown loading type: '{loading_type}'. "
                         f"Use 'impact', 'harmonic', or 'random'.")

    # --- Step 2: Distribute force across DOFs ---
    if position_frac is not None and L is not None:
        # Hermite interpolation at arbitrary position
        dof_weight_pairs = get_impact_dofs_and_weights(
            position_frac, L, free_dofs)

        for free_idx, weight in dof_weight_pairs:
            F_global[free_idx, :] += weight * force_time
    else:
        # Legacy path: single DOF from get_impact_dof
        impact_dof_global = get_impact_dof(bc_type)
        impact_dof_free   = np.where(free_dofs == impact_dof_global)[0]

        if len(impact_dof_free) == 0:
            raise ValueError(
                f"Impact DOF {impact_dof_global} not found in free DOFs. "
                f"Check boundary condition.")

        F_global[impact_dof_free[0], :] = force_time

    return F_global, force_time, time_vector


def get_impact_dof(bc_type):
    """
    Return the global DOF index where impact force is applied.

    For cantilever: free end is node 100, transverse DOF = 2*100 = 200
    This can be extended for other BC types easily.

    Inputs:
        bc_type : boundary condition string

    Output:
        impact_dof : global DOF index (int)
    """
    if bc_type == 'cantilever':
        # Free end = last node = node 100
        # Transverse DOF of node 100 = 2 * 100 = 200
        impact_dof = 2 * (N_NODES - 1)

    elif bc_type == 'simply_supported':
        # Apply at ~22% span (node 22) to excite both odd and even modes.
        # Midspan (node 50) is a nodal point for all even modes,
        # so hitting there would miss modes 2, 4, 6, ...
        # 0.22L avoids nodal points of the first ~8 modes.
        impact_node = int(round(0.22 * (N_NODES - 1)))   # node 22
        impact_dof  = 2 * impact_node

    elif bc_type == 'fixed_fixed':
        # Same reasoning as simply_supported — avoid midspan.
        impact_node = int(round(0.22 * (N_NODES - 1)))   # node 22
        impact_dof  = 2 * impact_node

    else:
        raise ValueError(f"Unknown BC type: '{bc_type}'")

    return impact_dof