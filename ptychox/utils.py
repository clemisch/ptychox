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


@jax.jit
def get_shifted_sinc(img, shift):
    """\
    Subpixel shift with sinc interpolation via FFT
    """
    Y, X = img.shape
    dy, dx = shift

    ky = jnp.fft.fftfreq(Y)[:, None]
    kx = jnp.fft.fftfreq(X)[None]

    img_ft = jnp.fft.fft2(img)
    phase_shift = jnp.exp(-2j * jnp.pi * (dy * ky + dx * kx))
    img_shift_ft = img_ft * phase_shift
    img_shift = jnp.fft.ifft2(img_shift_ft)

    return img_shift





@partial(jax.jit, static_argnames="p_shape")
def get_obj_crop(obj, shifts_int, p_shape):
    """
    * Assumes that shifted probe doesn't clip out of object
    * positive shift means negative slice offset (we shift obj, not viewport)
    """
    assert shifts_int.dtype == "int32", "Only supports integer shifts"

    V, U = obj.shape
    Y, X = p_shape
    dy, dx = shifts_int 

    y_lo = V // 2 - Y // 2 - dy
    x_lo = U // 2 - X // 2 - dx

    obj_cropped = jax.lax.dynamic_slice(obj, (y_lo, x_lo), (Y, X))

    return obj_cropped




@partial(jax.jit, static_argnames="o_shape")
def get_obj_uncrop(obj_cropped, shifts_int, o_shape):
    """
    `shifts_int` are values with which obj -> obj_cropped were computed, 
    *not* the pixel shift where to put `obj_cropped`
    """
    assert shifts_int.dtype == "int32", "Only supports integer shifts"

    Y, X = obj_cropped.shape
    V, U = o_shape
    dy, dx = shifts_int

    y_lo = V//2 - Y//2 - dy
    x_lo = U//2 - X//2 - dx

    obj = jnp.zeros_like(obj_cropped, shape=o_shape)
    obj = jax.lax.dynamic_update_slice(obj, obj_cropped, (y_lo, x_lo))

    return obj





@jax.jit
def get_probe_subpixel_shift(probe, shifts_rem):
    s0, s1 = shifts_rem
    
    probe = (1 - s0) * probe + s0 * jnp.roll(probe, jnp.sign(s0), 0)
    probe = (1 - s1) * probe + s1 * jnp.roll(probe, jnp.sign(s1), 1)

    return probe



@partial(jax.jit, static_argnames="reshift")
def get_exit_wave(obj, probe, shifts, reshift=False):
    """\
    Get exit wave with shifted object
    Object is shifted by integer shift and cropped to probe
    Remaining subpixel shift is applied (inversely) to probe via linear interp
    """

    # integer and subpixel shift
    shifts_int, shifts_rem = jnp.divmod(shifts, 1.)
    shifts_int = shifts_int.astype("int32")

    # shift object by integer amount and crop
    obj = get_obj_crop(obj, shifts_int, probe.shape)

    # shift probe by inverse subpixel shift
    probe = get_probe_subpixel_shift(probe, shifts_rem - 1)

    # exit wave
    exit = obj * probe

    if reshift:
        # shift exit wave to physically correct position
        # disabled by default because unnecessary for far-field ptycho
        # (translation in real space => phase ramp in Fourier space => no change in intensity)
        exit = get_probe_subpixel_shift(exit, shifts_rem)

    return exit
