from pylab import *
import jax
import jax.numpy as jnp
import scipy.ndimage as nd
import scipy.optimize as opt
from itertools import product
from tqdm import tqdm

import ptychox as px

rng = default_rng(1337)


import IPython
ipython = IPython.get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

###############################################################################
# SIMULATION
###############################################################################

N = 256

yy, xx = meshgrid(
    linspace(-1, 1, N, True), 
    linspace(-1, 1, N, True), 
    indexing="ij"
)
r = sqrt(yy**2 + xx**2)

# PROBE
pinhole = where(r < 0.2, 1., 0.)
pinhole = nd.gaussian_filter(pinhole, 1)
pinhole = pinhole + 0j

wlen = px.physics.energy_to_wavelen(6.)
probe = px.prop.to_nearfield(pinhole, 1e-6, wlen, 1.)

n_photons = 1e6
probe *= sqrt(n_photons / sum(abs(probe)**2))


# OBJECT
lenna = imread("/home/clem/Documents/data/lenna.png")
lenna = lenna[..., :-1].mean(-1)
lenna = (lenna - lenna.min()) / lenna.ptp()

earth = imread("/home/clem/Documents/data/earth.png")
earth = earth[..., :-1].mean(-1)
earth = (earth - earth.min()) / earth.ptp()

lenna = nd.zoom(lenna, N / array(lenna.shape), order=1)
earth = nd.zoom(earth, N / array(earth.shape), order=1)

obj_magn = earth
obj_phase = pi * (lenna - 0.5)
obj = obj_magn * exp(obj_phase * 1j)


M = 11
shifts = linspace(-0.35, 0.35, M, True) * N
shifts = array(list(product(shifts, shifts)), dtype="int32")

@jax.jit
def get_rolls(x, shifts):
    x_rolled = jax.vmap(jnp.roll, (None, 0, None), 0)(x, shifts, (0, 1))
    return x_rolled

probes = get_rolls(probe, shifts)


@jax.jit
def get_forward(obj, probe, shift):
    probe = jnp.roll(probe, shift, (0, 1))
    fwd = px.prop.to_farfield(obj * probe)
    return fwd

get_forwards = jax.vmap(get_forward, (None, None, 0))

fwds = get_forwards(obj, probe, shifts)

I_meas = abs(fwds)**2
I_meas_noise = poisson(I_meas).astype("float32")

# measured amplitudes
A_meas = sqrt(I_meas)
A_meas_noise = sqrt(I_meas_noise)


###############################################################################
# VISUALIZATION
###############################################################################

def pclip(x):
    lo, hi = [nanpercentile(x, p) for p in [1, 99]]
    return clip(x, lo, hi)


fig, ax = subplots(M, M, figsize=(10, 10), dpi=100)
for i, a in enumerate(ax.ravel()):
    eff = roll(probe, shifts[i], (0, 1)) * obj
    a.imshow(pclip(abs(eff)), cmap="viridis")
    a.set_xticks([]); a.set_yticks([])
    a.set_xticklabels([]); a.set_xticklabels([])
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

fig, ax = subplots(M, M, figsize=(10, 10), dpi=100)
for i, a in enumerate(ax.ravel()):
    a.imshow(pclip(I_meas_noise[i]), cmap="inferno")
    a.set_xticks([]); a.set_yticks([])
    a.set_xticklabels([]); a.set_xticklabels([])
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


###############################################################################
# DIFFERENCE MAP
###############################################################################

obj_0 = jnp.ones_like(obj)
probe_0 = nd.gaussian_filter(abs(probe), 1.) + 0j

psi = px.dm.set_amplitudes(obj_0, probe_0, A_meas_noise, shifts)
O = px.dm.update_object(psi, probe_0, shifts)
P = px.dm.update_probe(psi, O, shifts)


def wrap(x):
    return arctan2(sin(x), cos(x))

fig, ax = subplots(2, 2, figsize=(8, 8))
ax00 = ax[0, 0].imshow(abs(O), cmap="gray", vmin=-0.1, vmax=1.1)
ax01 = ax[0, 1].imshow(wrap(angle(O)), cmap="twilight", vmin=-pi, vmax=pi)
ax10 = ax[1, 0].imshow(abs(P), cmap="gray", vmin=-5, vmax=50)
ax11 = ax[1, 1].imshow(wrap(angle(P)), cmap="twilight", vmin=-pi, vmax=pi)
for a in ax.ravel(): 
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()


for i in tqdm(range(30)):
    psi = px.dm.set_amplitudes(O, P, A_meas_noise, shifts)
    O = px.dm.update_object(psi, P, shifts)
    P = px.dm.update_probe(psi, O, shifts)

    ax00.set_data(abs(O))
    ax01.set_data(angle(O))
    ax10.set_data(abs(P))
    ax11.set_data(angle(P))
    pause(1e-3)
