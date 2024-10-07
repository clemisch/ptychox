import jax
import jax.numpy as jnp
from functools import partial

from . import prop, utils


EPS = 1e-6

@jax.jit
def set_amplitudes(obj, probe, ampls, shifts):
    psis = jax.vmap(utils.get_exit_wave, (None, None, 0))(obj, probe, shifts)
    fwds = jax.vmap(prop.to_farfield)(psis)
    fwds = utils.set_magn(fwds, ampls)
    psis_new = jax.vmap(prop.from_farfield)(fwds)

    return psis_new


@jax.jit
def update_probe(psi, obj, shifts):
    p_shape = psi.shape[1:]

    # integer and subpixel shifts
    shifts_int, shifts_rem = jnp.divmod(shifts, 1.)
    shifts_int = shifts_int.astype("int32")

    obj = jax.vmap(utils.get_obj_crop, (None, 0, None))(obj, shifts_int, p_shape)

    subpx = lambda x: jax.vmap(utils.get_probe_subpixel_shift, 0)(x, shifts_rem)

    nom = jnp.sum(subpx(jnp.conj(obj) * psi), axis=0)
    denom = jnp.sum(jnp.square(subpx(jnp.abs(obj))), axis=0)

    probe = nom / (denom + EPS)

    return probe


@partial(jax.jit, static_argnames="o_shape")
def update_object(psi, probe, shifts, o_shape):
    # integer and subpixel shifts
    shifts_int, _ = jnp.divmod(shifts, 1.)
    shifts_int = shifts_int.astype("int32")

    nom = jax.vmap(utils.get_obj_uncrop, (0, 0, None))(
        jnp.conj(probe)[None] * psi,
        shifts_int,
        o_shape
    )
    nom = jnp.sum(nom, axis=0)

    denom = jnp.sum(jnp.square(jnp.abs(
        jax.vmap(utils.get_obj_uncrop, (None, 0, None))(probe, shifts_int, o_shape)
    )), axis=0)

    obj = nom / (denom + EPS)

    return obj


@jax.jit
def get_update(O, P, meas, shifts):
    psis = set_amplitudes(O, P, meas, shifts)
    O_ = update_object(psis, P, shifts, O.shape)
    P_ = update_probe(psis, O, shifts)

    return O_, P_
