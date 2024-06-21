import numpy as np
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

    # integer shifts
    if shifts.dtype == "int32":
        axes = range(img.ndim)
        out = jnp.roll(img, shifts, axes)

    # subpixel shifts
    else:
        # enforce positive shifts
        shape = jnp.array(img.shape)
        shifts = (shifts + shape) % shape
        shifts, rem = jnp.divmod(shifts, 1.)

        # shift each axis
        out = img
        for i, shift in enumerate(shifts):
            axis = i
            λ = rem[i]
            out = (
                (1 - λ) * jnp.roll(out, shift, axis) + 
                λ * jnp.roll(out, shift + 1, axis)
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



@partial(jax.jit, static_argnames="p_size")
def get_obj_crop_(obj, shifts, p_size):
    """ Slow (roll) but `shifts` is no static_argnum """
    V, U = obj.shape
    Y, X = p_size

    # roll object by integer shift
    obj = jnp.roll(obj, shifts, (0, 1))

    # cut object center with probe size
    # TODO: more efficient to slice in corner and adapt rolls?
    sl_center = np.s_[
        V//2 - Y//2 : V//2 - Y//2 + Y, 
        U//2 - X//2 : U//2 - X//2 + X
    ]
    obj = obj[sl_center]

    return obj


@partial(jax.jit, static_argnames="p_size")
def get_obj_crop(obj, shifts_int, p_size):
    """
    * Assumes that shifted probe doesn't clip out of object
    * positive shift means negative slice offset (we shift obj, not viewport)
    """
    assert shifts_int.dtype == "int32", "Only supports integer shifts"

    V, U = obj.shape
    Y, X = p_size
    dy, dx = shifts_int 

    y_lo = V // 2 - Y // 2 - dy
    x_lo = U // 2 - X // 2 - dx

    obj_cropped = jax.lax.dynamic_slice(obj, (y_lo, x_lo), (Y, X))

    return obj_cropped




@partial(jax.jit, static_argnames="o_size")
def get_obj_uncrop(obj_cropped, shifts_int, o_size):
    """
    `shifts_int` are values with which obj -> obj_cropped were computed, 
    *not* the pixel shift were to put `obj_cropped`
    """
    assert shifts_int.dtype == "int32", "Only supports integer shifts_int"

    Y, X = obj_cropped.shape
    V, U = o_size
    dy, dx = shifts_int

    y_lo = V//2 - Y//2 - dy
    x_lo = U//2 - X//2 - dx

    obj = jnp.zeros_like(obj_cropped, shape=o_size)
    obj = jax.lax.dynamic_update_slice(obj, obj_cropped, (y_lo, x_lo))

    return obj





@jax.jit
def get_probe_subpixel_shift(probe, shifts):
    s0, s1 = shifts
    probe = (1 - s0) * probe + s0 * jnp.roll(probe, 1, 0)
    probe = (1 - s1) * probe + s1 * jnp.roll(probe, 1, 1)

    return probe



@jax.jit
def get_exit_wave(obj, probe, shifts):
    """\
    Get exit wave with shifted object
    Object is rolled to integer shift
    Remaining subpixel shift is applied inversely to probe (linear interp)
    """

    # enforce positive shifts
    shape = jnp.array(obj.shape)
    shifts = (shifts + shape) % shape
    shifts_int, shifts_rem = jnp.divmod(shifts, 1.)

    # shift object by integer amount and crop
    obj = get_obj_crop(obj, shifts_int.astype("int32"), probe.shape)

    # roll probe by remaining subpixel shift
    probe = get_probe_subpixel_shift(probe, shifts_rem)

    # exit wave
    exit = obj * probe

    return exit
