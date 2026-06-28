from pylab import *
import jax
import jax.numpy as jnp
import scipy.ndimage as nd
from itertools import product
from tqdm import tqdm

import ptychox as px

rng = default_rng(1337)


try:
    import IPython
    ipython = IPython.get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
except ImportError:
    pass

###############################################################################
# VISUALIZATION
###############################################################################

def wrap(x):
    return arctan2(sin(x), cos(x))


def percentile_limits(x):
    lo, hi = nanpercentile(asarray(x), [1, 99])
    if not isfinite(lo) or not isfinite(hi):
        return 0., 1.
    if hi <= lo:
        delta = max(abs(float(lo)) * 1e-6, 1e-6)
        return lo - delta, hi + delta
    return lo, hi


def imshow_percentile(axis, data, **kwargs):
    lo, hi = percentile_limits(data)
    return axis.imshow(data, vmin=lo, vmax=hi, **kwargs)


def update_image(image, data):
    image.set_data(data)
    image.set_clim(*percentile_limits(data))


###############################################################################
# SIMULATION
###############################################################################

Np = 201
scaling = 2
N = scaling * Np
psize = 5e-5 / Np

yy, xx = meshgrid(
    linspace(-1, 1, Np, True),
    linspace(-1, 1, Np, True),
    indexing="ij"
)
r = sqrt(yy**2 + xx**2)

# PROBE
annulus = where((r > 0.15) & (r < 0.4), 1., 0.)
annulus = nd.gaussian_filter(annulus, 5)
focus = px.prop.from_farfield(annulus)

wlen = px.physics.energy_to_wavelen(6.)
kernel = px.prop.get_kernel_angular_spectrum(
    focus.shape, psize, wlen, 50e-3
)
probe = px.prop.to_nearfield(focus, kernel)

n_photons = 1e6
probe *= sqrt(n_photons / sum(abs(probe)**2))


# OBJECT
lenna = imread("/home/clem/Documents/data/lenna.png")
lenna = lenna[..., :-1].mean(-1)
lenna = (lenna - lenna.min()) / ptp(lenna)

earth = imread("/home/clem/Documents/data/earth.png")
earth = earth[..., :-1].mean(-1)
earth = (earth - earth.min()) / ptp(earth)

lenna = nd.zoom(lenna, N / array(lenna.shape), order=1)
earth = nd.zoom(earth, N / array(earth.shape), order=1)

obj_magn = 1 - earth
obj_phase = pi * (lenna - 0.5)
obj = obj_magn * exp(obj_phase * 1j)


M = 11
scan_extent = (N - Np) / 2 - 3
shifts = linspace(-scan_extent, scan_extent, M, True)
shifts = array(list(product(shifts, shifts)))
shifts += rng.uniform(-2, 2, shifts.shape)


@jax.jit
def get_forward(obj, probe, shift):
    return px.prop.to_farfield(px.utils.get_exit_wave(obj, probe, shift))

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


fig, ax = subplots(M, M, figsize=(8, 8), dpi=100)
for i, a in enumerate(ax.ravel()):
    eff = px.utils.get_exit_wave(obj, probe, shifts[i], reshift=True)
    a.imshow(pclip(abs(eff)), cmap="viridis")
    a.set_xticks([]); a.set_yticks([])
    a.set_xticklabels([]); a.set_xticklabels([])
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)

fig, ax = subplots(M, M, figsize=(8, 8), dpi=100)
for i, a in enumerate(ax.ravel()):
    a.imshow(pclip(I_meas_noise[i]), cmap="inferno")
    a.set_xticks([]); a.set_yticks([])
    a.set_xticklabels([]); a.set_xticklabels([])
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)


###############################################################################
# DIFFERENCE MAP
###############################################################################

# O, P = obj_0, probe_0
# psi = px.dm.get_exit_waves(O, P, shifts)
# psi, O, P = px.dm.step(psi, O, P, A_meas_noise, shifts)

P_ff = jnp.sqrt(jnp.mean(A_meas**2, axis=0))
P_focus = px.prop.from_farfield(P_ff)
P = px.prop.to_nearfield(P_focus, kernel)

O = ones((N, N)) + 0.j

fig, ax = subplots(2, 2, figsize=(8, 8))
ax00 = imshow_percentile(ax[0, 0], abs(O), cmap="gray")
ax01 = imshow_percentile(ax[0, 1], wrap(angle(O)), cmap="twilight")
ax10 = imshow_percentile(ax[1, 0], abs(P), cmap="gray")
ax11 = imshow_percentile(ax[1, 1], wrap(angle(P)), cmap="twilight")
for a in ax.ravel(): 
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()



psi = px.dm.get_exit_waves(O, P, shifts)
for _ in tqdm(range(20)):
    psi, O, P = px.dm.step(psi, O, P, A_meas, shifts, update_probe=False)

    update_image(ax00, abs(O))
    update_image(ax01, wrap(angle(O)))
    update_image(ax10, abs(P))
    update_image(ax11, wrap(angle(P)))
    pause(1e-3)


psi = px.dm.get_exit_waves(O, P, shifts)
for _ in tqdm(range(50)):
    psi, O, P = px.dm.step(psi, O, P, A_meas, shifts, update_probe=True)

    update_image(ax00, abs(O))
    update_image(ax01, wrap(angle(O)))
    update_image(ax10, abs(P))
    update_image(ax11, wrap(angle(P)))
    pause(1e-3)


###############################################################################
# INIT WITH SMOOTHED RESULT
###############################################################################

def gaussian_filter_circular(x, sig):
    xsin = nd.gaussian_filter(sin(x), sig)
    xcos = nd.gaussian_filter(cos(x), sig)
    out = arctan2(xsin, xcos)
    return out


O_smooth = nd.gaussian_filter(abs(O), 2) * exp(1j * gaussian_filter_circular(angle(O), 2))

O = O_smooth
psi = px.dm.get_exit_waves(O, P, shifts)
psi, O, P = px.dm.step(psi, O, P, A_meas_noise, shifts)


fig, ax = subplots(2, 2, figsize=(8, 8))
ax00 = ax[0, 0].imshow(abs(O), cmap="gray", vmin=-0.1, vmax=1.1)
ax01 = ax[0, 1].imshow(wrap(angle(O)), cmap="twilight", vmin=-pi, vmax=pi)
ax10 = ax[1, 0].imshow(abs(P), cmap="gray", vmin=-5, vmax=50)
ax11 = ax[1, 1].imshow(wrap(angle(P)), cmap="twilight", vmin=-pi, vmax=pi)
for a in ax.ravel(): 
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()

λ = 1.0

for i in tqdm(range(50)):
    psi, O_new, P_new = px.dm.step(psi, O, P, A_meas_noise, shifts)
    O = (1 - λ) * O + λ * O_new
    P = (1 - λ) * P + λ * P_new

    ax00.set_data(abs(O))
    ax01.set_data(angle(O))
    ax10.set_data(abs(P))
    ax11.set_data(angle(P))
    pause(1e-3)
