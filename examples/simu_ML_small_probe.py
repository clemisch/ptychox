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
pinhole = where(r < 0.2, 1., 0.)
pinhole = nd.gaussian_filter(pinhole, 1)
pinhole = pinhole + 0j

wlen = px.physics.energy_to_wavelen(6.)  # [keV] -> [m]
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

lenna = nd.zoom(lenna, 2 * N / array(lenna.shape), order=1)
earth = nd.zoom(earth, 2 * N / array(earth.shape), order=1)

obj_magn = 1 - earth
obj_phase = pi * (lenna - 0.5)
obj = obj_magn * exp(obj_phase * 1j)


M = 11
shifts0 = linspace(-0.25, 0.25, M, True) * N
shifts0 = array(list(product(shifts0, shifts0)))
shifts = shifts0 + rng.uniform(-1, 1, shifts0.shape)


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
# EXTENDED PIE
###############################################################################

def pclip(x):
    lo, hi = [nanpercentile(x, p) for p in [1, 99]]
    return clip(x, lo, hi)

def wrap(x):
    return arctan2(sin(x), cos(x))


O = jnp.ones_like(obj)
P = probe


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

    return cost


get_grad = jax.jit(jax.grad(get_cost, argnums=(0, 1, 2, 3)))



xopt, hist, res = px.lbfgs.lbfgs_aux(
    get_cost,
    get_grad,
    (O.real, O.imag, P.real, P.imag),
    aux=(A_meas_noise, shifts),
    tol=1e-4,
    maxiter=100,
    move_gpu=True,
    is_silent=False,
    history=True,
)

O_ = xopt[0] + 1j * xopt[1]
P_ = xopt[2] + 1j * xopt[3]
