# sensors.py - Extract accelerations
# sensors.py
# =============================================================
# Extract transverse acceleration at each of 101 nodes
# from the full acceleration history (free DOFs only)
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from config import N_NODES, N_STEPS

# -------------------------------------------------------------
# DOF MAPPING REMINDER
# Full system DOFs (before BC):
#   Node i → DOF 2i   (transverse w)  ← we want this
#   Node i → DOF 2i+1 (rotation θ)    ← ignore this
#
# After cantilever BC (remove DOFs 0,1):
#   Free DOF index = original DOF index - 2
#   Node 0  → fixed, not in free system
#   Node 1  → free DOF 0 (w), free DOF 1 (θ)
#   Node 2  → free DOF 2 (w), free DOF 3 (θ)
#   Node i  → free DOF 2*(i-1) (w)
#   Node 100→ free DOF 198 (w)
# -------------------------------------------------------------

def extract_node_accelerations(accel_history, free_dofs, n_nodes=N_NODES):
    """
    Extract transverse acceleration at each node from
    full free-DOF acceleration history.

    Inputs:
        accel_history : (n_free, n_steps) acceleration array (in/s²)
        free_dofs     : array of free DOF global indices
        n_nodes       : total number of nodes (101)

    Output:
        node_accels : (n_nodes, n_steps) array
                      Row i = transverse acceleration at node i
                      shape: (101, 2000)
    """
    n_steps     = accel_history.shape[1]
    node_accels = np.zeros((n_nodes, n_steps))

    for i in range(n_nodes):
        # Global DOF index for transverse displacement of node i
        global_dof = 2 * i

        # Check if this DOF is in the free system
        # (node 0 is fixed in cantilever — stays zero)
        if global_dof in free_dofs:
            # Map global DOF to free DOF index
            free_idx = np.where(free_dofs == global_dof)[0][0]
            node_accels[i, :] = accel_history[free_idx, :]
        else:
            # Fixed DOF — acceleration is zero
            node_accels[i, :] = 0.0

    return node_accels


def get_node_labels(n_nodes=N_NODES):
    """
    Generate node label strings for CSV column headers.

    Output:
        labels : list of strings
                 ['node_001_accel', 'node_002_accel', ..., 'node_101_accel']
    """
    return [f"node_{i+1:03d}_accel" for i in range(n_nodes)]


def serialize_node_accel(accel_array, delimiter=';'):
    """
    Serialize a 1D acceleration array into a single delimited string.
    Used for Encoding A CSV output.

    Inputs:
        accel_array : 1D numpy array of acceleration values (n_steps,)
        delimiter   : separator character (default ';')

    Output:
        string of acceleration values joined by delimiter
        e.g. "0.0;0.12;0.45;...;0.03"
    """
    return delimiter.join(f"{v:.6f}" for v in accel_array)