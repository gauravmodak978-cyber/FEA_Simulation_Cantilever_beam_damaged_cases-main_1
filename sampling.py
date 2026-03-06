# sampling.py
# =============================================================
# Parameter set generator for 500 beam simulations
# Produces randomized combinations of beam parameters
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
import random
from itertools import product
from materials import get_material
from config import (
    LENGTH_VALUES, WIDTH_VALUES, THICKNESS_VALUES, FORCE_VALUES,
    MATERIAL_NAMES, BOUNDARY_CONDITIONS, PULSE_DURATION,
    RAYLEIGH_ALPHA, RAYLEIGH_BETA, RAYLEIGH_DAMPING_PAIRS,
    T_TOTAL, N_STEPS, DT,
    N_ELEMENTS, N_NODES,
    RANDOM_SEED, N_SIMULATIONS,
    LOADING_TYPES,
    IMPACT_POSITION_VALUES, IMPACT_POSITION_DEFAULT,
    EXCITATION_FREQ_VALUES,
    SNR_VALUES,
    DAMAGE_ENABLED, INCLUDE_UNDAMAGED, UNDAMAGED_FRACTION,
    MIN_DAMAGES_PER_BEAM, MAX_DAMAGES_PER_BEAM,
    DAMAGE_SEVERITY_RANGE, DAMAGE_LOCATION_RANGE, DAMAGE_EXTENT_RANGE,
)
from damage import generate_random_damage_zones, combine_damage_zones

# -------------------------------------------------------------
# TWO SAMPLING MODES
# Mode 1 (discrete) : build all combinations from candidate
#                     lists, shuffle, take first N
# Mode 2 (continuous): uniform random draws from ranges
# Default            : Mode 1 if candidate lists provided
# -------------------------------------------------------------

def generate_parameter_sets(
        length_values       = LENGTH_VALUES,
        width_values        = WIDTH_VALUES,
        thickness_values    = THICKNESS_VALUES,
        force_values        = FORCE_VALUES,
        material_names      = MATERIAL_NAMES,
        boundary_conditions = BOUNDARY_CONDITIONS,
        loading_types       = LOADING_TYPES,
        impact_positions    = IMPACT_POSITION_VALUES,
        excitation_freqs    = EXCITATION_FREQ_VALUES,
        damping_pairs       = RAYLEIGH_DAMPING_PAIRS,
        snr_values          = SNR_VALUES,
        n_simulations       = N_SIMULATIONS,
        seed                = RANDOM_SEED,
        mode                = 1):
    """
    Generate n_simulations parameter sets.

    Mode 1 — Discrete permutation (default):
        Builds all combinations of L x b x t x F0 x material x bc_type
        x position x damping x loading_type (x freq for harmonic) x SNR,
        shuffles using seed, takes first n_simulations.

    Mode 2 — Continuous random:
        Draws uniformly from ranges for L, b, t, F0, position,
        randomly picks material, BC type, loading type, damping, and SNR.

    Inputs:
        length_values       : list of candidate lengths (in)
        width_values        : list of candidate widths (in)
        thickness_values    : list of candidate thicknesses (in)
        force_values        : list of candidate forces (lbf)
        material_names      : list of material name strings
        boundary_conditions : list of BC type strings
        loading_types       : list of loading type strings
        impact_positions    : list of fractional positions [0.0, 1.0]
        excitation_freqs    : list of excitation frequencies (Hz)
        damping_pairs       : list of (alpha, beta) tuples
        snr_values          : list of SNR values in dB for sensor noise
        n_simulations       : number of parameter sets to generate
        seed                : random seed for reproducibility
        mode                : 1 = discrete, 2 = continuous

    Output:
        param_sets : list of dicts, each containing
                     all parameters for one simulation
    """
    random.seed(seed)
    np.random.seed(seed)

    if mode == 1:
        param_sets = _discrete_mode(
            length_values, width_values, thickness_values,
            force_values, material_names, boundary_conditions,
            loading_types, impact_positions, excitation_freqs,
            damping_pairs, snr_values, n_simulations, seed)
    else:
        param_sets = _continuous_mode(
            length_values, width_values, thickness_values,
            force_values, material_names, boundary_conditions,
            loading_types, impact_positions, excitation_freqs,
            damping_pairs, snr_values, n_simulations, seed)

    return param_sets


def _validate_position_for_bc(pos, bc_type):
    """
    Ensure load position is not at a fully constrained node.
    Shifts position to the nearest interior point if needed.
    """
    min_frac = 1.0 / N_ELEMENTS  # first interior element midpoint
    if bc_type == 'cantilever' and pos < min_frac:
        return min_frac
    if bc_type in ('simply_supported', 'fixed_fixed'):
        if pos < min_frac:
            return min_frac
        if pos > 1.0 - min_frac:
            return 1.0 - min_frac
    return pos


def _discrete_mode(length_values, width_values, thickness_values,
                   force_values, material_names, boundary_conditions,
                   loading_types, impact_positions, excitation_freqs,
                   damping_pairs, snr_values, n_simulations, seed):
    """
    Build all combinations, shuffle, take first n_simulations.

    For 'harmonic' loading, each excitation frequency creates a separate combo.
    For 'impact' and 'random', excitation_freq is set to None.
    Each damping pair (alpha, beta) creates a separate combo.
    Each SNR value creates a separate combo.
    """
    # Build base combinations (geometry + material + BC + position + damping + SNR)
    base_combos = list(product(
        length_values,
        width_values,
        thickness_values,
        force_values,
        material_names,
        boundary_conditions,
        impact_positions,
        damping_pairs,
        snr_values,
    ))

    # Expand with loading types (and frequencies for harmonic)
    all_combos = []
    for combo in base_combos:
        bc_type = combo[5]
        pos     = combo[6]
        damp    = combo[7]        # (alpha, beta) tuple
        snr     = combo[8]        # SNR in dB
        pos     = _validate_position_for_bc(pos, bc_type)
        combo_with_pos = combo[:6] + (pos, damp, snr)

        for lt in loading_types:
            if lt == 'harmonic':
                for freq in excitation_freqs:
                    all_combos.append(combo_with_pos + (lt, freq))
            else:
                all_combos.append(combo_with_pos + (lt, None))

    total_combos = len(all_combos)
    print(f"Total possible combinations : {total_combos}")

    # None means use all combinations
    if n_simulations is None:
        n_simulations = total_combos

    print(f"Simulations requested       : {n_simulations}")

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(all_combos)

    # If not enough unique combos, cap at available count
    if total_combos < n_simulations:
        print(f"\u26a0  Only {total_combos} unique combos available.")
        print(f"   Capping simulations to {total_combos} (no repeats).")
        n_simulations = total_combos

    # Take first n_simulations
    selected = all_combos[:n_simulations]

    # Build parameter dicts with damage
    param_sets = []
    rng = np.random.default_rng(seed)

    # Determine how many undamaged baselines to include
    if DAMAGE_ENABLED and INCLUDE_UNDAMAGED:
        n_undamaged = max(1, int(round(len(selected) * UNDAMAGED_FRACTION)))
    elif not DAMAGE_ENABLED:
        n_undamaged = len(selected)  # all undamaged
    else:
        n_undamaged = 0

    for sim_id, (L, b, t, F0, mat_name, bc_type,
                 pos, damp, snr, lt, freq) in enumerate(selected):
        mat   = get_material(mat_name)
        alpha, beta_r = damp

        # Assign damage: first n_undamaged sims are healthy baselines
        if sim_id < n_undamaged or not DAMAGE_ENABLED:
            damage_zones = []
        else:
            damage_zones = generate_random_damage_zones(
                rng,
                min_damages=MIN_DAMAGES_PER_BEAM,
                max_damages=MAX_DAMAGES_PER_BEAM,
                severity_range=DAMAGE_SEVERITY_RANGE,
                location_range=DAMAGE_LOCATION_RANGE,
                extent_range=DAMAGE_EXTENT_RANGE,
            )

        entry = _build_param_dict(sim_id, L, b, t, F0, mat_name, mat, bc_type,
                                  loading_type=lt, impact_position=pos,
                                  excitation_freq=freq,
                                  rayleigh_alpha=alpha, rayleigh_beta=beta_r,
                                  snr_db=snr,
                                  damage_zones=damage_zones)
        param_sets.append(entry)

    return param_sets


def _continuous_mode(length_values, width_values, thickness_values,
                     force_values, material_names, boundary_conditions,
                     loading_types, impact_positions, excitation_freqs,
                     damping_pairs, snr_values, n_simulations, seed):
    """
    Draw uniformly from ranges for L, b, t, F0, position, random material,
    BC, loading type, damping pair, and SNR.
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    L_min,  L_max  = min(length_values),    max(length_values)
    b_min,  b_max  = min(width_values),     max(width_values)
    t_min,  t_max  = min(thickness_values), max(thickness_values)
    F0_min, F0_max = min(force_values),     max(force_values)
    pos_min, pos_max = min(impact_positions), max(impact_positions)
    freq_min, freq_max = min(excitation_freqs), max(excitation_freqs)

    # Determine undamaged count
    if DAMAGE_ENABLED and INCLUDE_UNDAMAGED:
        n_undamaged = max(1, int(round(n_simulations * UNDAMAGED_FRACTION)))
    elif not DAMAGE_ENABLED:
        n_undamaged = n_simulations
    else:
        n_undamaged = 0

    param_sets = []
    for sim_id in range(n_simulations):
        L        = np.random.uniform(L_min, L_max)
        b        = np.random.uniform(b_min, b_max)
        t        = np.random.uniform(t_min, t_max)
        F0       = np.random.uniform(F0_min, F0_max)
        mat_name = np.random.choice(material_names)
        bc_type  = np.random.choice(boundary_conditions)
        lt       = np.random.choice(loading_types)
        pos      = np.random.uniform(pos_min, pos_max)
        pos      = _validate_position_for_bc(pos, bc_type)
        freq     = (np.random.uniform(freq_min, freq_max)
                    if lt == 'harmonic' else None)
        damp_idx = np.random.randint(len(damping_pairs))
        alpha, beta_r = damping_pairs[damp_idx]
        snr      = np.random.choice(snr_values)
        mat      = get_material(mat_name)

        # Assign damage
        if sim_id < n_undamaged or not DAMAGE_ENABLED:
            damage_zones = []
        else:
            damage_zones = generate_random_damage_zones(
                rng,
                min_damages=MIN_DAMAGES_PER_BEAM,
                max_damages=MAX_DAMAGES_PER_BEAM,
                severity_range=DAMAGE_SEVERITY_RANGE,
                location_range=DAMAGE_LOCATION_RANGE,
                extent_range=DAMAGE_EXTENT_RANGE,
            )

        entry    = _build_param_dict(sim_id, L, b, t, F0, mat_name, mat, bc_type,
                                     loading_type=lt, impact_position=pos,
                                     excitation_freq=freq,
                                     rayleigh_alpha=alpha, rayleigh_beta=beta_r,
                                     snr_db=snr,
                                     damage_zones=damage_zones)
        param_sets.append(entry)

    return param_sets


def _build_param_dict(sim_id, L, b, t, F0, mat_name, mat, bc_type,
                      loading_type='impact',
                      impact_position=IMPACT_POSITION_DEFAULT,
                      excitation_freq=None,
                      rayleigh_alpha=RAYLEIGH_ALPHA,
                      rayleigh_beta=RAYLEIGH_BETA,
                      snr_db=40,
                      damage_zones=None):
    """
    Build a single simulation parameter dictionary.
    Contains all parameters needed for one simulation run.

    damage_zones : list of dicts with 'location', 'severity', 'extent'
                   or None / empty list for undamaged beam.
    """
    if damage_zones is None:
        damage_zones = []

    condition = 'undamaged' if len(damage_zones) == 0 else 'damaged'

    entry = {
        # Simulation ID
        'sim_id'        : sim_id,

        # Boundary condition
        'bc_type'       : bc_type,

        # Material
        'material'      : mat_name,
        'E_psi'         : mat['E'],
        'rho_lbm_in3'   : mat['rho_lbm'],
        'rho_consistent': mat['rho'],

        # Geometry
        'length_in'     : L,
        'width_in'      : b,
        'thickness_in'  : t,

        # Impact / Loading
        'impact_F0_lbf'  : F0,
        'impact_tau_s'   : PULSE_DURATION,
        'loading_type'   : loading_type,
        'impact_position': impact_position,
        'excitation_freq': excitation_freq,

        # Damping
        'rayleigh_alpha': rayleigh_alpha,
        'rayleigh_beta' : rayleigh_beta,

        # Noise
        'snr_db'        : snr_db,

        # Time
        'dt_s'          : DT,
        'T_s'           : T_TOTAL,
        'n_steps'       : N_STEPS,

        # Mesh
        'n_elements'    : N_ELEMENTS,
        'n_nodes'       : N_NODES,

        # Damage
        'condition'     : condition,
        'n_damages'     : len(damage_zones),
        'damage_zones'  : damage_zones,  # list of dicts (used by batch_runner)
    }

    # Flatten damage zone info for CSV export (up to 4 zones)
    for i in range(4):
        if i < len(damage_zones):
            z = damage_zones[i]
            entry[f'damage_{i+1}_location'] = round(z['location'], 4)
            entry[f'damage_{i+1}_severity'] = round(z['severity'], 4)
            entry[f'damage_{i+1}_extent']   = round(z['extent'], 4)
        else:
            entry[f'damage_{i+1}_location'] = 0.0
            entry[f'damage_{i+1}_severity'] = 0.0
            entry[f'damage_{i+1}_extent']   = 0.0

    return entry