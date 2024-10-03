from pylab import *
import jax
import jax.numpy as jnp
import scipy.ndimage as nd
import scipy.optimize as opt
from itertools import product
from functools import partial

import ptychox as px

rng = default_rng(1337)

###############################################################################
# ENV
###############################################################################

exec(open("/home/clem/Documents/code/fig_helpers.py").read())

import IPython
ipython = IPython.get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

###############################################################################
# SIMULATION
###############################################################################

N = 1001  # Number of detector pixels
w_FZP = 50e-6  # 50 μm FZP diameter
L0 = 12.1e-3  # 12.1 mm FZP-focus
L1 = 4.1e-3  # 5 mm focus-sample
L2 = 8.  # 8 meters sample-detector
Δd = 55e-6  # 55 μm detector pixels
E = 6  # 6 keV beam energy

λ = px.physics.energy_to_wavelen(E)  # wavelength in [m]
Δs = λ * L2 / (N * Δd)  # pixelsize in sample plane

w_s = Δs * N
w_p_max = w_s / 2
print(f"       FOV : {w_s*1e6:4.1f} μm")
print(f"Max. probe : {w_p_max*1e6:4.1f} μm")

M_FZP = L0 / w_FZP
print(f"   Max. L1 : {M_FZP*w_p_max*1e3:4.1f} mm")







yy, xx = meshgrid(
    linspace(-1., 1., N, True), 
    linspace(-1., 1., N, True), 
    indexing="ij"
)
R = sqrt(yy**2 + xx**2)

annulus = where((R > 0.15) & (R < 0.4), 1., 0)
annulus = nd.gaussian_filter(annulus, 1.)
focus = px.prop.from_farfield(annulus)

dim = (N, N)
kernel_fresnel = px.prop.get_kernel_fresnel(N, Δs, λ, L1)
kernel_as = px.prop.get_kernel_angular_spectrum(N, Δs, λ, L1)

probe_fresnel = px.prop.to_nearfield(focus, kernel_fresnel)
probe_as = px.prop.to_nearfield(focus, kernel_as)



M_geo = (L1 + L2) / L1

field_ff = px.prop.to_farfield(probe_as)
field_nf = px.prop.from_nearfield(
    field_ff,
    px.prop.get_kernel_fresnel(dim, Δd, λ, L2, M_geo)
)

test = px.prop.to_nearfield(
    field_nf,
    px.prop.get_kernel_fresnel(dim, Δs, λ, L2, M_geo)
)




annulus_ff = px.prop.to_nearfield(
    annulus + 0j,
    px.prop.get_kernel_fresnel(dim, Δs, λ, L2, M*10)
)
