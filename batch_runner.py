# batch_runner.py
# =============================================================
# Parallel simulation runner for 500 beam FEA simulations
# Uses joblib for parallel execution across CPU cores
# =============================================================

import numpy as np
import traceback
from joblib import Parallel, delayed
from tqdm import tqdm

from materials import get_material
from assembly import assemble_global_matrices, apply_boundary_conditions
from damping import build_rayleigh_damping
from excitation import build_force_vector
from time_integrator import newmark_beta_solver
from sensors import extract_node_accelerations
from noise import add_sensor_noise
from damage import combine_damage_zones
from config import N_ELEMENTS

# -------------------------------------------------------------
# SINGLE SIMULATION FUNCTION
# This function runs one complete simulation for one param set
# It is called in parallel for all 500 simulations
# -------------------------------------------------------------

def run_single_simulation(params):
    """
    Run one complete FEA simulation for a given parameter set.

    Inputs:
        params : dict containing all simulation parameters

    Output:
        result : dict containing params + node_accels (101 x n_steps)
                 OR error info if simulation failed
    """
    try:
        # --- Unpack parameters ---
        L        = params['length_in']
        b        = params['width_in']
        t        = params['thickness_in']
        F0       = params['impact_F0_lbf']
        mat_name = params['material']
        bc_type  = params.get('bc_type', 'cantilever')
        mat      = get_material(mat_name)
        E        = mat['E']
        rho      = mat['rho']
        alpha    = params['rayleigh_alpha']
        beta_r   = params['rayleigh_beta']

        # --- Unpack loading parameters ---
        loading_type    = params.get('loading_type', 'impact')
        impact_pos      = params.get('impact_position', None)
        excitation_freq = params.get('excitation_freq', None)

        # --- Compute damage stiffness multipliers ---
        damage_zones = params.get('damage_zones', [])
        if damage_zones:
            stiffness_multipliers = combine_damage_zones(N_ELEMENTS, damage_zones)
        else:
            stiffness_multipliers = None

        # --- Assemble global matrices ---
        K, M = assemble_global_matrices(E, rho, b, t, L,
                                        stiffness_multipliers=stiffness_multipliers)

        # --- Apply boundary conditions ---
        K_free, M_free, free_dofs = apply_boundary_conditions(K, M, bc_type=bc_type)

        # --- Build damping matrix ---
        C_free = build_rayleigh_damping(M_free, K_free, alpha, beta_r)

        # --- Build force vector ---
        F_global, _, _ = build_force_vector(
            F0, free_dofs, L=L, bc_type=bc_type,
            loading_type=loading_type,
            position_frac=impact_pos,
            excitation_freq=excitation_freq,
            random_seed=params.get('sim_id', None))

        # --- Run Newmark-Beta solver ---
        accel_history = newmark_beta_solver(M_free, C_free, K_free, F_global)

        # --- Extract node accelerations ---
        node_accels = extract_node_accelerations(accel_history, free_dofs)

        # --- Sanity check on clean data ---
        if np.any(np.isnan(node_accels)) or np.any(np.isinf(node_accels)):
            raise ValueError("NaN or Inf detected in acceleration output.")

        # --- Add sensor noise ---
        snr_db = params.get('snr_db', 40)
        noise_seed = params.get('sim_id', None)
        noisy_accels, _ = add_sensor_noise(node_accels, snr_db, seed=noise_seed)

        # --- Return result ---
        return {
            'status'       : 'success',
            'params'       : params,
            'node_accels'  : node_accels,    # clean (101, n_steps)
            'noisy_accels' : noisy_accels,   # noisy (101, n_steps)
        }

    except Exception as e:
        # Return error info without crashing the batch
        return {
            'status' : 'failed',
            'params' : params,
            'error'  : str(e),
            'trace'  : traceback.format_exc()
        }


def run_batch(param_sets, n_jobs=-1):
    """
    Run all simulations in parallel using joblib.

    Inputs:
        param_sets : list of parameter dicts (500 dicts)
        n_jobs     : number of CPU cores to use
                     -1 = use all available cores

    Output:
        results    : list of result dicts (one per simulation)
        n_success  : number of successful simulations
        n_failed   : number of failed simulations
    """
    print(f"Starting batch of {len(param_sets)} simulations...")
    print(f"Using n_jobs = {n_jobs} (−1 means all CPU cores)")
    print()

    # Run in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_simulation)(params)
        for params in tqdm(param_sets, desc="Simulations", unit="sim")
    )

    # Count success and failures
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_failed  = sum(1 for r in results if r['status'] == 'failed')

    print()
    print(f"Batch complete.")
    print(f"  Succeeded : {n_success}")
    print(f"  Failed    : {n_failed}")

    # Print any failures
    if n_failed > 0:
        print()
        print("Failed simulations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  sim_id {r['params']['sim_id']} "
                      f"→ {r['error']}")

    return results, n_success, n_failed