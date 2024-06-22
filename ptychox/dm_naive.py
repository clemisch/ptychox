import jax
import jax.numpy as jnp

from . import prop, utils

EPS = 1e-6


@jax.jit
def update_probe(psi, obj, shifts):
    obj = jax.vmap(utils.get_shifted_roll, (None, 0))(obj, -shifts)
    psi = jax.vmap(utils.get_shifted_roll, (0, 0))(psi, -shifts)

    nom = jnp.sum(jnp.conj(obj) * psi, axis=0)
    denom = jnp.sum(jnp.square(jnp.abs(obj)), axis=0)
    probe = nom / (denom + EPS)

    return probe


@jax.jit
def update_object(psi, probe, shifts):
    probes = jax.vmap(utils.get_shifted_roll, (None, 0))(probe, shifts)

    nom = jnp.sum(jnp.conj(probes) * psi, axis=0)
    denom = jnp.sum(jnp.square(jnp.abs(probes)), axis=0)
    obj = nom / (denom + EPS)

    return obj


@jax.jit
def set_amplitudes(obj, probe, ampls, shifts):
    probes = jax.vmap(utils.get_shifted_roll, (None, 0))(probe, shifts)
    fwds = jax.vmap(prop.to_farfield)(obj[None] * probes)
    fwds = utils.set_magn(fwds, ampls)
    bcks = jax.vmap(prop.from_farfield)(fwds)

    return bcks
