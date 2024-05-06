import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
from functools import partial

EPS = 1e-6


@partial(jax.jit, inline=True)
def set_magn(x, m):
    out = m * jnp.exp(1j * jnp.angle(x))
    # out = m * c / (jnp.abs(c) + EPS)
    return out


@jax.jit
def get_shifted_roll(img, shifts):
    shifts = jnp.asarray(shifts)
    assert shifts.size == img.ndim

    axes = range(img.ndim)

    # integer shifts
    if shifts.dtype == "int32":
        out = jnp.roll(img, shifts, axes)

    # subpixel shifts
    else:
        shape = jnp.array(img.shape)
        shifts = (shifts + shape) % shape
        shifts, rem = jnp.divmod(shifts, 1.)

        # shift each axis
        out = img
        for i, shift in enumerate(shifts):
            out = (
                jnp.roll(out, shift, i) * (1 - rem[i]) + 
                jnp.roll(out, shift + 1, i) * rem[i]
            )

    return out


@jax.jit
def get_shifted_bilinear(img, shift):
    V, U = img.shape
    dv, du = shift
    vv = jnp.arange(V, dtype="float32") - dv
    uu = jnp.arange(U, dtype="float32") - du
    coords = jnp.meshgrid(vv, uu, indexing="ij", sparse=True)

    out = jnd.map_coordinates(
        img, 
        coords,
        order=1,
        mode="constant", 
        cval=0.0
    )

    return out
