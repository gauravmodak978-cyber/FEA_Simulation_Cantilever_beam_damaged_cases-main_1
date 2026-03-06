# damping.py
# =============================================================
# Rayleigh damping matrix builder
# C = alpha * M + beta * K
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from scipy.sparse import csr_matrix
from config import RAYLEIGH_ALPHA, RAYLEIGH_BETA

# -------------------------------------------------------------
# UNIT NOTE
# C : lbf·s/in  (damping)
# M : lbf·s²/in (mass)
# K : lbf/in    (stiffness)
# alpha : 1/s
# beta  : s
# -------------------------------------------------------------

def build_rayleigh_damping(M_free, K_free, alpha=RAYLEIGH_ALPHA, beta=RAYLEIGH_BETA):
    """
    Build Rayleigh damping matrix from mass and stiffness matrices.

    C = alpha * M + beta * K

    Inputs:
        M_free : reduced mass matrix (free DOFs only)
        K_free : reduced stiffness matrix (free DOFs only)
        alpha  : mass proportional damping coefficient (1/s)
        beta   : stiffness proportional damping coefficient (s)

    Output:
        C_free : Rayleigh damping matrix (same shape as M and K)
    """
    C_free = alpha * M_free + beta * K_free
    return csr_matrix(C_free)


def rayleigh_from_damping_ratios(zeta1, zeta2, omega1, omega2):
    """
    Compute alpha and beta from target damping ratios at two frequencies.

    Useful when you know the physical damping of your beam
    (e.g. 2% damping at first two natural frequencies).

    Inputs:
        zeta1  : damping ratio at frequency omega1 (dimensionless, e.g. 0.02)
        zeta2  : damping ratio at frequency omega2 (dimensionless, e.g. 0.02)
        omega1 : first angular frequency (rad/s)
        omega2 : second angular frequency (rad/s)

    Output:
        alpha  : mass proportional coefficient (1/s)
        beta   : stiffness proportional coefficient (s)

    Solving:
        [zeta1]   1   [1/omega1  omega1] [alpha]
        [zeta2] = - * [1/omega2  omega2] [beta ]
                  2
    """
    # Build 2x2 system
    A = 0.5 * np.array([
        [1/omega1,  omega1],
        [1/omega2,  omega2]
    ])

    zeta = np.array([zeta1, zeta2])

    # Solve for alpha and beta
    coeffs = np.linalg.solve(A, zeta)
    alpha  = coeffs[0]
    beta   = coeffs[1]

    return alpha, beta