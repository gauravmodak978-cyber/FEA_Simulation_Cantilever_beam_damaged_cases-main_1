# assembly.py
# =============================================================
# Mesh generation, global K and M assembly, boundary conditions
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from beam_element import element_stiffness, element_mass, compute_section_properties
from config import N_ELEMENTS, N_NODES, BOUNDARY_CONDITION

# -------------------------------------------------------------
# UNIT NOTE
# node_coords : inches
# K_global    : lbf/in
# M_global    : lbf·s²/in
# -------------------------------------------------------------

def generate_mesh(L):
    """
    Generate equally spaced node coordinates along beam.

    Inputs:
        L : beam length (in)

    Output:
        node_coords : array of 101 node positions (in)
        Le          : element length (in)
    """
    node_coords = np.linspace(0, L, N_NODES)
    Le = L / N_ELEMENTS
    return node_coords, Le


def assemble_global_matrices(E, rho, b, t, L, stiffness_multipliers=None):
    """
    Assemble global stiffness K and mass M matrices.

    Inputs:
        E                     : Young's modulus (psi)
        rho                   : consistent mass density (lbf·s²/in⁴)
        b                     : width (in)
        t                     : thickness (in)
        L                     : beam length (in)
        stiffness_multipliers : optional 1D array (N_ELEMENTS,)
                                per-element stiffness scale factors.
                                1.0 = healthy, <1.0 = damaged.
                                Mass matrix is NOT affected.
                                If None, all elements are healthy.

    Output:
        K_global : sparse (202 x 202) global stiffness matrix
        M_global : sparse (202 x 202) global mass matrix
    """
    total_dofs = N_NODES * 2   # 101 nodes x 2 DOFs = 202

    # Use lil_matrix for efficient assembly
    K_global = lil_matrix((total_dofs, total_dofs))
    M_global = lil_matrix((total_dofs, total_dofs))

    # Cross section properties
    A, I = compute_section_properties(b, t)

    # Element length
    Le = L / N_ELEMENTS

    # Healthy mass matrix (same for all elements — mass not affected by damage)
    Me = element_mass(rho, A, Le)

    # Assemble element by element
    for e in range(N_ELEMENTS):
        # Per-element stiffness scaling (damage)
        if stiffness_multipliers is not None:
            E_eff = E * stiffness_multipliers[e]
        else:
            E_eff = E

        Ke = element_stiffness(E_eff, I, Le)

        # DOF indices for element e
        # Node e  -> DOFs [2e,   2e+1  ]
        # Node e+1-> DOFs [2e+2, 2e+3  ]
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]

        # Scatter element matrices into global matrices
        for i in range(4):
            for j in range(4):
                K_global[dofs[i], dofs[j]] += Ke[i, j]
                M_global[dofs[i], dofs[j]] += Me[i, j]

    # Convert to csr for efficient math operations
    K_global = csr_matrix(K_global)
    M_global = csr_matrix(M_global)

    return K_global, M_global


def apply_boundary_conditions(K_global, M_global, bc_type=BOUNDARY_CONDITION):
    """
    Apply boundary conditions by removing fixed DOFs.

    Supported bc_type:
        'cantilever'       : fixed at node 0 (DOFs 0,1 removed)
        'simply_supported' : fixed w at node 0 and node N (DOFs 0, 2*N removed)
        'fixed_fixed'      : fixed at both ends (DOFs 0,1 and 2N, 2N+1 removed)

    Inputs:
        K_global : sparse (202 x 202)
        M_global : sparse (202 x 202)
        bc_type  : string boundary condition type

    Output:
        K_free   : reduced stiffness matrix (free DOFs only)
        M_free   : reduced mass matrix (free DOFs only)
        free_dofs: list of free DOF indices
    """
    total_dofs = N_NODES * 2
    all_dofs   = np.arange(total_dofs)

    # Define fixed DOFs based on BC type
    if bc_type == 'cantilever':
        # Fix both DOFs at node 0 (left end)
        fixed_dofs = [0, 1]

    elif bc_type == 'simply_supported':
        # Fix transverse DOF at node 0 and node 100
        fixed_dofs = [0, 2 * (N_NODES - 1)]

    elif bc_type == 'fixed_fixed':
        # Fix all DOFs at both ends
        fixed_dofs = [0, 1, 2*(N_NODES-1), 2*(N_NODES-1)+1]

    else:
        raise ValueError(f"Unknown BC type: '{bc_type}'. "
                         f"Choose from: cantilever, simply_supported, fixed_fixed")

    # Free DOFs = all DOFs minus fixed DOFs
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Extract submatrices for free DOFs only
    K_free = csr_matrix(K_global[np.ix_(free_dofs, free_dofs)])
    M_free = csr_matrix(M_global[np.ix_(free_dofs, free_dofs)])

    return K_free, M_free, free_dofs