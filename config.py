# config.py
# =============================================================
# Central configuration file for Euler-Bernoulli Beam FEA
# All units in IPS system: inches, lbf, seconds
# =============================================================

# -------------------------------------------------------------
# UNIT SYSTEM NOTE
# Length   : inches (in)
# Force    : pound-force (lbf)
# Time     : seconds (s)
# Pressure : psi (lbf/in²)
# Mass     : lbf·s²/in  (consistent mass unit)
# -------------------------------------------------------------

# Gravitational constant (for density unit conversion)
G_C = 386.088  # in/s²

# -------------------------------------------------------------
# GEOMETRY
# -------------------------------------------------------------
THICKNESS_VALUES = [0.4
                   # , 0.1, 0.3
                   # , 0.5
                    ]  # t (in)

LENGTH_VALUES = [50
                 # , 60, 80, 100, 120, 140, 160, 180
                 ]  # L (in)

WIDTH_VALUES  = [2 
                # ,0.5, 1 ,1.5 
                # ,2.5,  3 , 4 , 5
                 ]             # b (in)

# -------------------------------------------------------------
# IMPACT FORCE
# -------------------------------------------------------------
FORCE_VALUES   = [5
                # , 20, 45, 60
                 #, 75, 90, 150, 300, 500
                  ]  # F0 (lbf)
PULSE_DURATION = 0.0021  # tau, impact duration (s) — 25ms pulse

# -------------------------------------------------------------
# LOADING TYPE
# -------------------------------------------------------------
# Supported types: 'impact', 'harmonic', 'random'
LOADING_TYPES = ['impact', 'harmonic', 'random']
LOADING_TYPE  = 'impact'   # default for single runs (backward compat)

# -------------------------------------------------------------
# IMPACT / LOAD POSITION
# -------------------------------------------------------------
# Fractional position along beam length [0.0, 1.0]
# 0.0 = left end (node 0), 1.0 = right end (node N)
IMPACT_POSITION_VALUES  = [0.25, 0.5, 0.75, 1.0]
IMPACT_POSITION_DEFAULT = 1.0  # backward compat: tip for cantilever

# -------------------------------------------------------------
# HARMONIC LOADING
# -------------------------------------------------------------
EXCITATION_FREQ_VALUES  = [10.0, 25.0, 50.0, 100.0, 200.0]  # Hz
EXCITATION_FREQ_DEFAULT = 50.0                                 # Hz

# -------------------------------------------------------------
# RANDOM LOADING
# -------------------------------------------------------------
RANDOM_BANDWIDTH = [0.0, 500.0]  # [f_low, f_high] Hz bandpass limits

# -------------------------------------------------------------
# TIME SETTINGS
# Matched to Photron FASTCAM at 2000 fps
# dt = 0.0005 s, f_max captured = 1000 Hz
# Resolution at 2000 fps: 1024 x 500 pixels
# -------------------------------------------------------------
T_TOTAL = 1.0   # total simulation time (s)
N_STEPS = 2000  # number of timesteps (matches SA5 at 2000 fps)
DT      = T_TOTAL / (N_STEPS - 1)  # 0.0005005 s per step

# -------------------------------------------------------------
# MESH
# -------------------------------------------------------------
N_ELEMENTS = 100  # number of beam elements
N_NODES    = 101  # number of nodes (N_ELEMENTS + 1)

# -------------------------------------------------------------
# RAYLEIGH DAMPING
# -------------------------------------------------------------
RAYLEIGH_ALPHA = 0.01     # mass proportional damping coefficient (1/s)
RAYLEIGH_BETA  = 0.00001  # stiffness proportional damping coefficient (s)

# Damping variation for batch simulations
# Typical steel/aluminum beams: zeta ~ 0.1% to 2% of critical
# Each tuple is (alpha, beta) representing a damping level
# Paired to keep physically meaningful combinations
RAYLEIGH_DAMPING_PAIRS = [
    (0.005,  0.000005)
    #,   # low damping   (~0.1-0.3% zeta)
    #(0.01,   0.00001),    # medium damping (~0.5-1% zeta) — current default
    #(0.05,   0.00005),    # high damping   (~1-2% zeta)
]

# -------------------------------------------------------------
# NEWMARK-BETA PARAMETERS
# -------------------------------------------------------------
NEWMARK_BETA  = 0.25  # average acceleration (unconditionally stable)
NEWMARK_GAMMA = 0.50

# -------------------------------------------------------------
# BOUNDARY CONDITION
# -------------------------------------------------------------
# Supported types: 'cantilever', 'simply_supported', 'fixed_fixed'
BOUNDARY_CONDITIONS = ['cantilever', 'simply_supported', 'fixed_fixed']
BOUNDARY_CONDITION  = 'cantilever'   # default for single-BC runs (backward compat)

# -------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------
N_SIMULATIONS = None  # None = all unique combinations
RANDOM_SEED   = 42

# -------------------------------------------------------------
# SENSOR NOISE
# -------------------------------------------------------------
# Gaussian white noise added to acceleration output (post-simulation)
# SNR = signal-to-noise ratio in dB
#   SNR = 20 * log10(RMS_signal / RMS_noise)
#   60 dB → noise is 1/1000 of signal (very clean)
#   40 dB → noise is 1/100  of signal (clean)
#   30 dB → noise is 1/32   of signal (moderate)
#   20 dB → noise is 1/10   of signal (noisy)
SNR_VALUES = [40, 60]  # dB, mild noise (realistic lab-grade sensors)

# Export mode: 'clean', 'noisy', or 'both'
#   'clean' : export only clean (noise-free) acceleration data
#   'noisy' : export only noise-corrupted acceleration data
#   'both'  : export both clean and noisy in separate CSV files
EXPORT_MODE = 'noisy'

# -------------------------------------------------------------
# MATERIALS AVAILABLE
# -------------------------------------------------------------
MATERIAL_NAMES = ['steel', 'aluminum']

# -------------------------------------------------------------
# DAMAGE MODELING
# -------------------------------------------------------------
# Damage is modeled as local stiffness reduction (E_eff = E * (1 - reduction))
# using a Gaussian taper centered at the damage location.
# Mass matrix is NOT affected (crack-type damage model).

DAMAGE_ENABLED = True  # set False to run only undamaged simulations

# Include undamaged (healthy) simulations in the batch?
INCLUDE_UNDAMAGED    = False
UNDAMAGED_FRACTION   = 0.15  # ~15% of total sims will be undamaged baselines

# Number of independent damage zones per beam
MIN_DAMAGES_PER_BEAM = 1
MAX_DAMAGES_PER_BEAM = 4

# Per-zone severity: peak stiffness reduction fraction
#   0.05 = 5% loss (barely detectable hairline crack)
#   0.60 = 60% loss (severe open crack / deep corrosion)
DAMAGE_SEVERITY_RANGE = (0.05, 0.60)

# Per-zone location: fractional position along beam [0, 1]
#   Avoids very edges to stay away from boundary condition nodes
DAMAGE_LOCATION_RANGE = (0.05, 0.95)

# Per-zone extent: Gaussian spread in number of elements
#   1  = very localized (single-element crack)
#   10 = broad damage zone (corrosion patch)
DAMAGE_EXTENT_RANGE = (1, 10)