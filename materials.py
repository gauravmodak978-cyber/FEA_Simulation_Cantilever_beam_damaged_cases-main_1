# materials.py
# =============================================================
# Material properties registry for Euler-Bernoulli Beam FEA
# All units in IPS system: inches, lbf, seconds
# =============================================================

from config import G_C

# -------------------------------------------------------------
# UNIT NOTE
# E   : psi (lbf/in²)
# rho : lbf·s²/in⁴  (consistent mass density)
#       converted from lbm/in³ by dividing by G_C = 386.088
# -------------------------------------------------------------

# -------------------------------------------------------------
# MATERIAL REGISTRY
# Each material is a dictionary with:
#   E       : Young's modulus (psi)
#   rho     : mass density in consistent units (lbf·s²/in⁴)
#   rho_lbm : original density in lbm/in³ (for reference/export)
#   nu      : Poisson's ratio 
# -------------------------------------------------------------

MATERIALS = {
    'steel': {
        'E'       : 29.0e6,               # psi
        'rho'     : 0.2830 / G_C,         # lbf·s²/in⁴
        'rho_lbm' : 0.2830,               # lbm/in³ (reference)
        'nu'      : 0.30,                 # Poisson's ratio
    },
    'aluminum': {
        'E'       : 10.0e6,               # psi
        'rho'     : 0.0998 / G_C,         # lbf·s²/in⁴
        'rho_lbm' : 0.0998,               # lbm/in³ (reference)
        'nu'      : 0.33,                 # Poisson's ratio
    },
}

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------

def get_material(name):
    """
    Retrieve material properties by name.
    Returns a dict with E, rho, rho_lbm, nu.
    """
    name = name.lower()
    if name not in MATERIALS:
        available = list(MATERIALS.keys())
        raise ValueError(f"Material '{name}' not found. Available: {available}")
    return MATERIALS[name]


def add_material(name, E, rho_lbm, nu=0.0):
    """
    Add a new material to the registry.
    Inputs:
        name    : string identifier
        E       : Young's modulus (psi)
        rho_lbm : density in lbm/in³  (auto converted to consistent units)
        nu      : Poisson's ratio (optional, for future use)
    """
    name = name.lower()
    MATERIALS[name] = {
        'E'       : E,
        'rho'     : rho_lbm / G_C,
        'rho_lbm' : rho_lbm,
        'nu'      : nu,
    }
    print(f"Material '{name}' added successfully.")


def list_materials():
    """
    Print all available materials and their properties.
    """
    print(f"{'Material':<12} {'E (psi)':<14} {'rho (lbm/in³)':<16} {'rho (lbf·s²/in⁴)':<20} {'nu'}")
    print("-" * 70)
    for name, props in MATERIALS.items():
        print(f"{name:<12} {props['E']:<14.3e} {props['rho_lbm']:<16.4f} {props['rho']:<20.6e} {props['nu']}")