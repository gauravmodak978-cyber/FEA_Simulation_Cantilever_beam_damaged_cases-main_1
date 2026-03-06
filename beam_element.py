# beam_element.py
# =============================================================
# Euler-Bernoulli 2-node beam element stiffness and mass matrices
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np

# -------------------------------------------------------------
# UNIT NOTE
# Ke : lbf/in   (stiffness)
# Me : lbf·s²/in (mass)
# Le : inches
# E  : psi (lbf/in²)
# I  : in⁴
# rho: lbf·s²/in⁴
# A  : in²
# -------------------------------------------------------------

# -------------------------------------------------------------
# DOF ORDER PER ELEMENT
# Local DOF index:
#   0 → w_i   (transverse displacement at node i)
#   1 → θ_i   (rotation at node i)
#   2 → w_j   (transverse displacement at node j)
#   3 → θ_j   (rotation at node j)
# -------------------------------------------------------------

def element_stiffness(E, I, Le):
    """
    Compute 4x4 Euler-Bernoulli element stiffness matrix.

    Inputs:
        E  : Young's modulus (psi)
        I  : second moment of area (in⁴)
        Le : element length (in)

    Output:
        Ke : 4x4 numpy array (lbf/in)
    """
    c = E * I / Le**3

    Ke = c * np.array([
        [ 12,      6*Le,   -12,      6*Le  ],
        [  6*Le,   4*Le**2, -6*Le,   2*Le**2],
        [-12,     -6*Le,    12,     -6*Le  ],
        [  6*Le,   2*Le**2, -6*Le,   4*Le**2]
    ])

    return Ke


def element_mass(rho, A, Le):
    """
    Compute 4x4 Euler-Bernoulli consistent mass matrix.

    Inputs:
        rho : mass density in consistent units (lbf·s²/in⁴)
        A   : cross-sectional area (in²)
        Le  : element length (in)

    Output:
        Me : 4x4 numpy array (lbf·s²/in)
    """
    c = rho * A * Le / 420

    Me = c * np.array([
        [ 156,      22*Le,    54,     -13*Le   ],
        [  22*Le,   4*Le**2,  13*Le,   -3*Le**2],
        [  54,      13*Le,   156,     -22*Le   ],
        [ -13*Le,   -3*Le**2, -22*Le,   4*Le**2]
    ])

    return Me


def compute_section_properties(b, t):
    """
    Compute cross-section area and second moment of area.

    Inputs:
        b : width (in)
        t : thickness (in)

    Output:
        A : cross-sectional area (in²)
        I : second moment of area (in⁴)
    """
    A = b * t
    I = b * t**3 / 12

    return A, I