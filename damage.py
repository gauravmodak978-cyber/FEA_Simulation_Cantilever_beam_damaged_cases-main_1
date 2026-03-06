# damage.py
# =============================================================
# Damage modeling for Euler-Bernoulli beam FEA
# Implements local stiffness reduction with Gaussian taper
# Mass matrix is NOT affected (crack-type damage model)
# =============================================================

import numpy as np
from config import N_ELEMENTS


def gaussian_damage_profile(n_elements, center_element, severity, extent):
    """
    Compute per-element stiffness reduction using a Gaussian taper.

    The reduction is maximum at the center element and tapers off
    smoothly following a Gaussian bell curve.

    Inputs:
        n_elements     : total number of beam elements (int)
        center_element : element index at damage center (int, 0-based)
        severity       : peak stiffness reduction fraction (0.0 to 1.0)
                         e.g. 0.3 means 30% stiffness loss at center
        extent         : Gaussian spread in number of elements (float)
                         sigma = extent / 3 so ~99.7% of effect is
                         within 'extent' elements on each side

    Output:
        reductions : 1D array (n_elements,) of stiffness reduction
                     fractions per element (0.0 = no reduction)
    """
    if severity <= 0.0:
        return np.zeros(n_elements)

    sigma = max(extent / 3.0, 0.5)  # minimum sigma to avoid division issues
    elements = np.arange(n_elements)
    distances = np.abs(elements - center_element)
    reductions = severity * np.exp(-distances**2 / (2 * sigma**2))

    return reductions


def combine_damage_zones(n_elements, damage_zones):
    """
    Combine multiple damage zones into a single per-element
    stiffness reduction array using multiplicative stacking.

    Multiplicative combination ensures total stiffness never
    goes negative, even with overlapping damages:
        E_eff(e) = E * product(1 - reduction_i(e)) for all zones i

    Inputs:
        n_elements  : total number of beam elements (int)
        damage_zones: list of dicts, each with keys:
                        'location'  : fractional position (0.0 to 1.0)
                        'severity'  : peak reduction (0.0 to 1.0)
                        'extent'    : Gaussian spread (elements)

    Output:
        stiffness_multipliers : 1D array (n_elements,)
                                multiply element E by this value
                                1.0 = healthy, <1.0 = damaged
    """
    if not damage_zones:
        return np.ones(n_elements)

    # Start with no reduction (multiplier = 1.0)
    stiffness_multipliers = np.ones(n_elements)

    for zone in damage_zones:
        location = zone['location']   # fractional (0.0 to 1.0)
        severity = zone['severity']   # peak reduction (0.0 to 1.0)
        extent   = zone['extent']     # spread in elements

        # Convert fractional position to element index
        center_element = int(round(location * (n_elements - 1)))
        center_element = np.clip(center_element, 0, n_elements - 1)

        # Get Gaussian reduction profile for this zone
        reductions = gaussian_damage_profile(
            n_elements, center_element, severity, extent)

        # Multiplicative stacking: multiply by (1 - reduction)
        stiffness_multipliers *= (1.0 - reductions)

    # Clamp to [0.01, 1.0] — never allow zero stiffness (singular matrix)
    stiffness_multipliers = np.clip(stiffness_multipliers, 0.01, 1.0)

    return stiffness_multipliers


def generate_random_damage_zones(rng,
                                  min_damages=1,
                                  max_damages=4,
                                  severity_range=(0.05, 0.60),
                                  location_range=(0.05, 0.95),
                                  extent_range=(1, 10)):
    """
    Generate a random set of damage zones for one beam.

    Inputs:
        rng             : numpy RandomState or Generator for reproducibility
        min_damages     : minimum number of damage zones (int)
        max_damages     : maximum number of damage zones (int)
        severity_range  : (min, max) peak stiffness reduction
        location_range  : (min, max) fractional position along beam
        extent_range    : (min, max) Gaussian spread in elements

    Output:
        damage_zones : list of dicts with 'location', 'severity', 'extent'
    """
    n_damages = rng.integers(min_damages, max_damages + 1)

    damage_zones = []
    for _ in range(n_damages):
        zone = {
            'location': float(rng.uniform(location_range[0], location_range[1])),
            'severity': float(rng.uniform(severity_range[0], severity_range[1])),
            'extent':   float(rng.uniform(extent_range[0], extent_range[1])),
        }
        damage_zones.append(zone)

    return damage_zones
