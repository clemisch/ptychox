from pylab import *
import jax
import jax.numpy as jnp
import scipy.ndimage as nd
import scipy.optimize as opt
from itertools import product

import ptychox as px

rng = default_rng(1337)


import IPython
ipython = IPython.get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")



from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    kwargs["bar_format"] = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]"
    return _tqdm(*args, **kwargs)

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
# pinhole = where(r < 0.2, 1., 0.)
# pinhole = nd.gaussian_filter(pinhole, 1)
# pinhole = pinhole + 0j

annulus = where((r > 0.2) & (r < 0.5), 1., 0)
annulus = nd.gaussian_filter(annulus, 1)

focus = fftshift(ifft2(fftshift(annulus)))

wlen = px.physics.energy_to_wavelen(6.)  # [keV] -> [m]
probe = px.prop.to_nearfield(focus, 1e-7, wlen, 15e-3)

n_photons = 1e6
probe *= sqrt(n_photons / sum(abs(probe)**2))



probes = array([px.prop.to_nearfield(focus, 1e-7, wlen, d) for d in logspace(-3, -1, 20, True)])



# OBJECT
lenna = imread("/home/clem/Documents/data/lenna.png")
lenna = lenna[..., :-1].mean(-1)
lenna = (lenna - lenna.min()) / ptp(lenna)

earth = imread("/home/clem/Documents/data/earth.png")
earth = earth[..., :-1].mean(-1)
earth = (earth - earth.min()) / ptp(earth)

scaling = 3
lenna = nd.zoom(lenna, scaling * N / array(lenna.shape), order=1)
earth = nd.zoom(earth, scaling * N / array(earth.shape), order=1)

obj_magn = 1 - earth
obj_phase = pi * (lenna - 0.5)
obj = obj_magn * exp(obj_phase * 1j)


M = 11
shifts0 = linspace(-0.20, 0.20, M, True) * N * scaling
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
    
fig, ax = subplots(1, 2)
ax[0].imshow(abs(obj), cmap="gray", extent=[-256, 255, -256, 255])
ax[0].scatter(*shifts.T, fc="none", ec="red")

ax[1].imshow(angle(obj), cmap="gray", extent=[-256, 255, -256, 255])
ax[1].scatter(*shifts.T, fc="none", ec="red")

pause(1e-2)

###############################################################################
# ML
###############################################################################

def pclip(x):
    lo, hi = [nanpercentile(x, p) for p in [1, 99]]
    return clip(x, lo, hi)

def wrap(x):
    return arctan2(sin(x), cos(x))

@jax.jit
def get_tv(x):
    cost = 0.0
    for i in range(x.ndim):
        xd = jnp.diff(x, axis=i)
        cost += jnp.sum(jnp.abs(xd))
    return cost


@jax.jit
def get_cost(O_real, O_imag, P_real, P_imag, ampls, shifts):
    O = O_real + 1j * O_imag
    P = P_real + 1j * P_imag

    def get_cost_shot(ampl, shift):
        exit = px.utils.get_exit_wave(O, P, shift)
        psi = px.prop.to_farfield(exit)
        fwd = jnp.abs(psi)
        cost = jnp.sum(jnp.square(fwd - ampl))
        return cost

    costs = jax.vmap(get_cost_shot)(ampls, shifts)
    cost = jnp.sum(costs)

    cost_reg = get_tv(jnp.abs(O))
    cost += cost_reg

    return cost

get_grad = jax.jit(jax.grad(get_cost, argnums=(0, 1, 2, 3)))





O = jnp.ones_like(obj)
P = probe
# P = nd.gaussian_filter(abs(probe), 1) * exp(1j * nd.gaussian_filter(angle(probe), 1))

xopt, hist, res = px.lbfgs.lbfgs_aux(
    get_cost,
    get_grad,
    (O.real, O.imag, P.real, P.imag),
    aux=(A_meas_noise, shifts),
    tol=1e-4,
    maxiter=10,
    move_gpu=True,
    is_silent=False,
    history=True,
)

O_ = xopt[0] + 1j * xopt[1]
P_ = xopt[2] + 1j * xopt[3]

O_hist = array([item[0] + 1j * item[1] for item in hist])
P_hist = array([item[2] + 1j * item[3] for item in hist])


###############################################################################
# DM
###############################################################################


O = jnp.ones_like(obj)
P = probe

O, P = px.dm.get_update(O, P, A_meas, shifts)

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
    O_, P_ = px.dm.get_update(O, P, A_meas, shifts)
    O = (1 - λ) * O + λ * O_
    P = (1 - λ) * P + λ * P_

    ax00.set_data(abs(O))
    ax01.set_data(angle(O))
    ax10.set_data(abs(P))
    ax11.set_data(angle(P))
    pause(1e-3)
