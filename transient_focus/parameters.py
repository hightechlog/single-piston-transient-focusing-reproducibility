"""Numerical parameters used for the published single-piston study.

All dimensional quantities use SI units.
"""

PISTON_RADIUS_M = 50e-3
SOUND_SPEED_M_S = 1500.0
CENTER_FREQUENCY_HZ = 100e3
BANDWIDTH_HZ = 100e3
FREQUENCY_MIN_HZ = CENTER_FREQUENCY_HZ - BANDWIDTH_HZ / 2.0
FREQUENCY_MAX_HZ = CENTER_FREQUENCY_HZ + BANDWIDTH_HZ / 2.0

TARGET_DEPTHS_M = (20e-3, 50e-3, 100e-3, 150e-3)
NOMINAL_TARGET_M = 50e-3

# Frequency grids. The scalar-delay baseline is identified on the high-resolution
# axial grid and then frozen on the lower-resolution grid used for 2-D propagation.
DELAY_SEARCH_N_FREQUENCY = 4096
SYNTHESIS_N_FREQUENCY = 256

# Scalar-delay search used in the published figures.
DELAY_GLOBAL_SAMPLES = 4001
DELAY_LOCAL_SAMPLES = 801
TARGET_EQUIVALENT_REL_TOL = 1e-4
DELAY_BATCH_SIZE = 256

# Rayleigh integral quadrature used for the full axisymmetric field.
N_RHO = 32
N_PHI = 64
PATH_SAMPLES_PER_MIN_WAVELENGTH = 80.0
