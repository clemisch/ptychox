from pylab import *
import jax
import jax.numpy as jnp
import scipy.ndimage as nd
import scipy.optimize as opt
from itertools import product
from functools import partial

import ptychox as px

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
# PROBE
###############################################################################

rng = default_rng(1337)

N = 501
psize = 5e-5 / N

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

annulus = where((r > 0.15) & (r < 0.4), 1., 0)
annulus = nd.gaussian_filter(annulus, 5)
focus = px.prop.from_farfield(annulus)

wlen = px.physics.energy_to_wavelen(6.)  # [keV] -> [m]
probe = px.prop.to_nearfield(
    focus, 
    px.prop.get_kernel_angular_spectrum(focus.shape, psize, wlen, 20e-3)
)

n_photons = 1e6
probe *= sqrt(n_photons / sum(abs(probe)**2))


w_d = 0.5 * N * psize * 1e6
extent = [-w_d, w_d, -w_d, w_d]

fig, ax = subplots(1, 2, figsize=(8, 4), dpi=150, sharex=True, sharey=True)
ax[0].imshow(abs(probe)**.5, extent=extent, cmap="viridis")
ax[1].imshow(angle(probe), extent=extent, cmap="twilight")
ax[0].set_title("abs(P)")
ax[1].set_title("angle(P)")

fig.tight_layout()


###############################################################################
# OBJECT
###############################################################################

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

###############################################################################
# ML
###############################################################################

def pclip(x):
    lo, hi = nanpercentile(x, (1, 99))
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

    cost_reg = get_tv(jnp.abs(O)) + get_tv(jnp.angle(O))
    cost += 10 * cost_reg

    return cost

get_grad = jax.jit(jax.grad(get_cost, argnums=(0, 1, 2, 3)))





O = jnp.ones_like(obj)
P = probe







def get_callback():
    i = [0]
    fig, ax = subplots(1, 2)
    ax0 = ax[0].imshow(abs(O), cmap="gray", vmin=-0.1, vmax=1.1)
    ax1 = ax[1].imshow(wrap(angle(O)), cmap="twilight", vmin=-pi, vmax=pi)
    for a in ax.ravel(): 
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    refresh(fig)

    def callback(x):
        if i[0] % 2 == 0:
            O = x[0] + 1j * x[1]
            P = x[2] + 1j * x[3]
            ax0.set_data(abs(O))
            ax1.set_data(angle(O))
            refresh(fig)
        i[0] += i[0] + 1

    return fig, ax, callback

fig, ax, callback = get_callback()


# P = nd.gaussian_filter(abs(probe), 1) * exp(1j * nd.gaussian_filter(angle(probe), 1))


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
    # callback=callback,
    # history=False,
    callback=None,
)

O_ = xopt[0] + 1j * xopt[1]
P_ = xopt[2] + 1j * xopt[3]

O_hist = array([item[0] + 1j * item[1] for item in hist])
P_hist = array([item[2] + 1j * item[3] for item in hist])



lo, hi = percentile(A_meas_noise[7], (1, 99))
fig, ax = subplots(1, 2, figsize=(8, 4), dpi=150, sharex=True)
ax[0].imshow(A_meas_noise[7])
ax[1].imshow(A_meas_noise[7], vmin=lo, vmax=hi)
fig.tight_layout()


################################################################################
# APPROXIMATE FISHER INF
###############################################################################

# from jax.flatten_util import ravel_pytree
# ravel_pytree_jit = jax.jit(lambda tree: ravel_pytree(tree)[0])

# @jax.jit
# def get_cost_complex(O_abs, O_angle, P, meas, shifts):
#     O_abs = O_abs.reshape(obj.shape)
#     O_angle = O_angle.reshape(obj.shape)
#     O = O_abs * jnp.exp(1j * O_angle)
#     cost = get_cost(O.real, O.imag, P.real, P.imag, meas, shifts)
#     return cost

# get_cost_complex_grad = jax.jit(jax.grad(get_cost_complex, argnums=0))
# get_cost_complex_hess = jax.jit(jax.hessian(get_cost_complex, argnums=0))


# @partial(jax.jit, static_argnums=0)
# def hvp(f, x, v):
#     """ Matrix-free hessian-vector product """
#     return jax.jvp(jax.grad(f), (x,), (v,))[1]


# @partial(jax.jit, static_argnums=0)
# def hess_diag(f, x):
#     hd = hvp(f, x, jnp.ones(x.size))
#     return hd

# dd = hess_diag(
#     lambda x: get_cost_complex(x, angle(O_), P_, A_meas, shifts),
#     abs(O_).ravel(),
# ).reshape(O.shape)


###############################################################################
# HUTCHINSON
###############################################################################

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
ravel_pytree_jit = jax.jit(lambda tree: ravel_pytree(tree)[0])

@jax.jit
def get_cost_complex(O_abs, O_angle, P, meas, shifts):
    O_abs = O_abs.reshape(obj.shape)
    O_angle = O_angle.reshape(obj.shape)
    O = O_abs * jnp.exp(1j * O_angle)
    cost = get_cost(O.real, O.imag, P.real, P.imag, meas, shifts)
    return cost

get_cost_complex_grad = jax.jit(jax.grad(get_cost_complex, argnums=1))
get_cost_complex_hess = jax.jit(jax.hessian(get_cost_complex, argnums=1))



@partial(jax.jit, static_argnums=0)
def hvp(f, x, v):
    """ Matrix-free hessian-vector product """
    return jax.jvp(jax.grad(f), (x,), (v,))[1]


@partial(jax.jit, static_argnums=1)
def rademacher(key, N):
    """ Sample `N` points from Rademacher distribution """
    a = jnp.array([-1.0, 1.0], dtype="float32")
    return jax.random.choice(key, a, (N,))


@partial(jax.jit, static_argnames=("f", "niters"))
def hutchinson(key, f, x, niters):
    """ Approximate Hessian diagonal of f(x) """
    assert x.ndim == 1
    ndim = len(x)
    
    def body_fun(carry, _):
        key, diag = carry
        z = rademacher(key, ndim)
        diag += z * hvp(f, x, z)
        key, _ = jax.random.split(key)
        return (key, diag), None

    (_, diag), _ = jax.lax.scan(
        body_fun, 
        (key, jnp.zeros(ndim)), 
        None, length=niters
    )
    diag /= niters

    return diag



key = jax.random.PRNGKey(42)
O_, P_, A_meas, shifts = jax.device_put((O_, P_, A_meas, shifts))

dd2 = hutchinson(
    key,
    lambda x: get_cost_complex(jnp.abs(O_).ravel(), x, P_, A_meas, shifts),
    jnp.angle(O_).ravel(),
    500
).reshape(O.shape).block_until_ready()


fisher = where(dd2 > 1e-3, 1 / dd2, 1e-3)


###############################################################################
# GRADIENT COST WRT DATA
###############################################################################





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

refresh(fig)


λ = 1.0

for i in tqdm(range(50)):
    O_, P_ = px.dm.get_update(O, P, A_meas, shifts)
    O = (1 - λ) * O + λ * O_
    P = (1 - λ) * P + λ * P_

    if i % 2 == 0:
        ax00.set_data(abs(O))
        ax01.set_data(angle(O))
        ax10.set_data(abs(P))
        ax11.set_data(angle(P))
        refresh(fig)


###############################################################################
# MPL SLICER
###############################################################################

class IndexTracker(object):
    def __init__(self, fig, ax, X, **kwargs):
        self.fig = fig
        self.ax = ax
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0  

        lo, hi = [nanpercentile(self.X, p) for p in [1, 99]]
        self.im = ax.imshow(self.X[self.ind], vmin=lo, vmax=hi, **kwargs)

        # colorbar(self.im)
        cbar(self.ax, ticks=(lo, hi), size="5%")

        self.update()
        magic_layout(self.fig)



    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = min(self.ind + 1, self.slices - 1)
        else:
            self.ind = max(self.ind - 1, 0)
        self.update()


    def on_press(self, event):
        if event.key == 'right':
            self.ind = min(self.ind + 1, self.slices - 1)
        elif event.key == 'left':
            self.ind = max(self.ind - 1, 0)
        if event.key == 'down':
            self.ind = min(self.ind + 10, self.slices - 1)
        if event.key == 'up':
            self.ind = max(self.ind - 10, 0)

        if event.key == 'r':
            self.update_limits()

        self.update()


    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_title('%s / %s' % (self.ind, self.slices))
        self.fig.canvas.draw()



def ss(x, **kwargs):
    fig, ax = subplots(1, 1)
    tracker = IndexTracker(fig, ax, x, **kwargs)
    # fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    fig.canvas.mpl_connect('key_press_event', tracker.on_press)
    return fig, ax, tracker



ss(abs(O_hist), cmap="viridis")
ss(A_meas_noise, cmap="inferno")
