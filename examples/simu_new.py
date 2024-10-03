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


from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    kwargs["bar_format"] = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"
    return _tqdm(*args, **kwargs)


def refresh(fig):
    fig.canvas.draw_idle()
    fig.canvas.draw()
    fig.canvas.flush_events()


###############################################################################
# SIMULATION
###############################################################################

N = 1001  # Number of detector pixels
L1 = 5e-3  # 5 mm focus-sample
L2 = 8.  # 8 meters sample-detector
Δd = 55e-6  # 55 μm detector pixels
E = 6  # 6 keV beam energy

λ = px.physics.energy_to_wavelen(E)  # wavelength in [m]
Δs = λ * L2 / (N * Δd)  # pixelsize in sample plane




yy, xx = meshgrid(
    linspace(-1., 1., N, True), 
    linspace(-1., 1., N, True), 
    indexing="ij"
)
R = sqrt(yy**2 + xx**2)

annulus = where((R > 0.15) & (R < 0.4), 1., 0)
annulus = nd.gaussian_filter(annulus, 1.)
focus = px.prop.from_farfield(annulus)

kernel_fresnel = px.prop.get_kernel_fresnel(focus, Δs, λ, 5e-3)
kernel_as = px.prop.get_kernel_angular_spectrum(focus, Δs, λ, 5e-3)

probe_fresnel = px.prop.to_nearfield(focus, kernel_fresnel)
probe_as = px.prop.to_nearfield(focus, kernel_as)




# n_photons = 1e6
# probe *= sqrt(n_photons / sum(abs(probe)**2))



# probes = array([px.prop.to_nearfield(focus, 1e-7, wlen, d) for d in logspace(-3, -1, 20, True)])

# manual probe
# probe = where((r > 0.2) & (r < 0.6), 1., 0)
# probe = nd.gaussian_filter(probe, 1)
# # probe = probe * jnp.exp(1j * 2 * pi * (xx + yy))
# probe = probe * exp(1j)
# probe *= sqrt(n_photons / sum(abs(probe)**2))






# OBJECT
lenna = imread("/home/clem/Documents/data/lenna.png")
# lenna = imread("/das/home/schmid_c2/mypgroup/data/lenna.png")
lenna = lenna[..., :-1].mean(-1)
lenna = (lenna - lenna.min()) / ptp(lenna)

earth = imread("/home/clem/Documents/data/earth.png")
# earth = imread("/das/home/schmid_c2/mypgroup/data/earth.png")
earth = earth[..., :-1].mean(-1)
earth = (earth - earth.min()) / ptp(earth)

scaling = 2
lenna = nd.zoom(lenna, scaling * N / array(lenna.shape), order=1)
earth = nd.zoom(earth, scaling * N / array(earth.shape), order=1)

obj_magn = 1 - earth
obj_phase = pi * (lenna - 0.5)
obj = obj_magn * exp(obj_phase * 1j)


M = 11
shifts0 = linspace(-0.3, 0.3, M, True) * N * scaling
shifts0 = array(list(product(shifts0, shifts0)))
shifts = shifts0 + rng.uniform(-2, 2, shifts0.shape)


@jax.jit
def get_forward(obj, probe, shift):
    exit = px.utils.get_exit_wave(obj, probe, shift)
    fwd = px.prop.to_farfield(exit)
    return fwd

get_forwards = jax.vmap(get_forward, (None, None, 0))

fwds = get_forwards(obj, probe, shifts)

I_meas = abs(fwds)**2
I_meas_noise = poisson(I_meas).astype("float32")

# measured amplitudes
A_meas = sqrt(I_meas)
A_meas_noise = sqrt(I_meas_noise)

###############################################################################
# VISUALIZE STEPS
###############################################################################

colors = cm.cool(linspace(0, 1, M**2, True))

extent = [-N*scaling/2, N*scaling/2-1, N*scaling/2-1, -N*scaling/2]
    
fig, ax = subplots(1, 2)
ax[0].imshow(abs(obj), cmap="gray", extent=extent)
ax[0].scatter(*-shifts.T, c=colors, marker="x")

ax[1].imshow(angle(obj), cmap="gray", extent=extent)
ax[1].scatter(*-shifts.T, c=colors, marker="x")

tight_layout()
refresh(fig)
