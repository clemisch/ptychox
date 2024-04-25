import numpy as np

def energy_to_wavelen(energy):
    """ [keV] to [m] """
    h = 4.1357e-15   # [eVs]
    c = 299792458    # [m/s]

    return h * c / (energy * 1e3)
