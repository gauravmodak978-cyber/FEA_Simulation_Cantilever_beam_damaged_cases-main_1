# exporter.py
# =============================================================
# CSV export for beam FEA simulation results
# One CSV file per simulation
# Each CSV: 1 row of metadata + 101 node columns
# =============================================================

import numpy as np
import pandas as pd
import os
from sensors import get_node_labels, serialize_node_accel
from config import N_NODES, N_STEPS, DT, T_TOTAL, EXPORT_MODE

def _build_metadata_row(params):
    """
    Build the metadata dict shared by both clean and noisy exports.
    Includes damage metadata columns.
    """
    row = {
        'sim_id'        : params['sim_id'],
        'bc_type'       : params.get('bc_type', 'cantilever'),
        'material'      : params['material'],
        'E_psi'         : params['E_psi'],
        'rho_lbm_in3'   : params['rho_lbm_in3'],
        'rho_consistent': params['rho_consistent'],
        'length_in'     : params['length_in'],
        'width_in'      : params['width_in'],
        'thickness_in'  : params['thickness_in'],
        'impact_F0_lbf'  : params['impact_F0_lbf'],
        'impact_tau_s'   : params['impact_tau_s'],
        'loading_type'   : params.get('loading_type', 'impact'),
        'impact_position': params.get('impact_position', None),
        'excitation_freq': params.get('excitation_freq', None),
        'rayleigh_alpha': params['rayleigh_alpha'],
        'rayleigh_beta' : params['rayleigh_beta'],
        'snr_db'        : params.get('snr_db', None),
        'dt_s'          : params['dt_s'],
        'T_s'           : params['T_s'],
        'n_steps'       : params['n_steps'],
        'n_elements'    : params['n_elements'],
        'n_nodes'       : params['n_nodes'],
        # Damage metadata
        'condition'     : params.get('condition', 'undamaged'),
        'n_damages'     : params.get('n_damages', 0),
    }

    # Flatten per-zone damage info (up to 4 zones)
    for i in range(4):
        row[f'damage_{i+1}_location'] = params.get(f'damage_{i+1}_location', 0.0)
        row[f'damage_{i+1}_severity'] = params.get(f'damage_{i+1}_severity', 0.0)
        row[f'damage_{i+1}_extent']   = params.get(f'damage_{i+1}_extent', 0.0)

    return row


def _write_csv(row, node_accels, sim_id, output_dir, file_suffix,
               encoding, delimiter):
    """
    Write one CSV file for a given node_accels array.

    Inputs:
        row          : metadata dict
        node_accels  : (n_nodes, n_steps) acceleration array
        sim_id       : simulation ID (int)
        output_dir   : folder path
        file_suffix  : '' for clean, '_noisy' for noisy
        encoding     : 'A', 'B', or 'C'
        delimiter    : delimiter for Encoding A

    Output:
        file_path : path of saved CSV file
    """
    row = dict(row)  # copy so we don't mutate the original

    if encoding == 'A':
        labels = get_node_labels(N_NODES)
        for i, label in enumerate(labels):
            row[label] = serialize_node_accel(
                node_accels[i, :], delimiter)

    elif encoding == 'B':
        for i in range(N_NODES):
            for t in range(N_STEPS):
                col_name = f"node_{i+1:03d}_t{t:04d}"
                row[col_name] = node_accels[i, t]

    elif encoding == 'C':
        labels      = get_node_labels(N_NODES)
        time_vector = np.linspace(0, DT * (N_STEPS - 1), N_STEPS)

        ts_data = {'time_s': time_vector}
        for i, label in enumerate(labels):
            ts_data[label] = node_accels[i, :]
        df = pd.DataFrame(ts_data)

        for k, v in row.items():
            df.insert(df.columns.get_loc('time_s'), k, v)

        os.makedirs(output_dir, exist_ok=True)
        file_name = f"sim_{sim_id:04d}{file_suffix}.csv"
        file_path = os.path.join(output_dir, file_name)
        df.to_csv(file_path, index=False)
        return file_path

    else:
        raise ValueError(f"Unknown encoding '{encoding}'. Use 'A', 'B', or 'C'.")

    df = pd.DataFrame([row])
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"sim_{sim_id:04d}{file_suffix}.csv"
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)
    return file_path


def export_single_simulation(result, output_dir='simulation_results',
                              encoding='A', delimiter=';',
                              export_mode=EXPORT_MODE):
    """
    Export one simulation result to CSV file(s).

    File naming:
        Clean : sim_0001.csv
        Noisy : sim_0001_noisy.csv
        Both  : both files are created

    Inputs:
        result      : single result dict from batch_runner
        output_dir  : folder to save CSV files
        encoding    : 'A', 'B', or 'C'
        delimiter   : delimiter for Encoding A (default ';')
        export_mode : 'clean', 'noisy', or 'both'

    Output:
        file_paths : list of saved file paths
    """
    if result['status'] != 'success':
        return []

    params      = result['params']
    sim_id      = params['sim_id']
    row         = _build_metadata_row(params)

    file_paths = []

    if export_mode in ('clean', 'both'):
        path = _write_csv(row, result['node_accels'], sim_id,
                          output_dir, '', encoding, delimiter)
        if path:
            file_paths.append(path)

    if export_mode in ('noisy', 'both'):
        noisy_data = result.get('noisy_accels', result['node_accels'])
        path = _write_csv(row, noisy_data, sim_id,
                          output_dir, '_noisy', encoding, delimiter)
        if path:
            file_paths.append(path)

    return file_paths


def export_all_simulations(results, output_dir='simulation_results',
                           encoding='A', delimiter=';',
                           export_mode=EXPORT_MODE):
    """
    Export all simulation results to individual CSV files.

    Inputs:
        results     : list of result dicts from batch_runner
        output_dir  : folder to save all CSV files
        encoding    : 'A', 'B', or 'C' (see export_single_simulation)
        delimiter   : delimiter for Encoding A
        export_mode : 'clean', 'noisy', or 'both'

    Output:
        file_paths  : list of saved file paths
        n_exported  : number of simulations exported
        n_skipped   : number of failed sims skipped
    """
    print(f"Exporting simulations to folder: '{output_dir}'")
    print(f"Encoding    : {encoding}")
    print(f"Export mode : {export_mode}")
    print()

    os.makedirs(output_dir, exist_ok=True)

    file_paths = []
    n_exported = 0
    n_skipped  = 0

    for r in results:
        if r['status'] != 'success':
            n_skipped += 1
            continue

        paths = export_single_simulation(
            r, output_dir, encoding, delimiter, export_mode)

        if paths:
            file_paths.extend(paths)
            n_exported += 1

    print(f"Simulations exported : {n_exported}")
    print(f"Total files written  : {len(file_paths)}")
    print(f"Simulations skipped  : {n_skipped} (failed)")
    print(f"Output folder        : {os.path.abspath(output_dir)}")

    if file_paths:
        size_mb = os.path.getsize(file_paths[0]) / 1e6
        print(f"Size per file        : ~{round(size_mb, 2)} MB")
        print(f"Total est size       : ~{round(size_mb * len(file_paths), 1)} MB")

    return file_paths, n_exported, n_skipped


def export_time_vector(output_dir='simulation_results',
                       output_path='time_vector.csv'):
    """
    Export shared time vector to a separate CSV file.
    Saved in the same output folder as simulation CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    time_vector = np.linspace(0, DT * (N_STEPS - 1), N_STEPS)
    df_time     = pd.DataFrame({'time_s': time_vector})

    full_path   = os.path.join(output_dir, output_path)
    df_time.to_csv(full_path, index=False)

    print(f"Time vector saved : {full_path}")
    print(f"  Steps   : {len(time_vector)}")
    print(f"  t_start : {time_vector[0]} s")
    print(f"  t_end   : {round(time_vector[-1], 4)} s")
    print(f"  dt      : {round(time_vector[1]-time_vector[0], 6)} s")

    return df_time