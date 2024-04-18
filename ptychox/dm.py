import jax
import jax.numpy as jnp

from . import prop


@jax.jit
def update_probe(psi, obj, shifts):
    obj = jax.vmap(jnp.roll, (None, 0, None), 0)(obj, -shifts, (0, 1))
    psi = jax.vmap(jnp.roll, (0, 0, None), 0)(psi, -shifts, (0, 1))

    nom = jnp.sum(jnp.conj(obj) * psi, axis=0)
    denom = jnp.sum(jnp.square(jnp.abs(obj)), axis=0)
    probe = nom / (denom + 1e-6)

    return probe


@jax.jit
def update_object(psi, probe, shifts):
    probes = jax.vmap(jnp.roll, (None, 0, None), 0)(probe, shifts, (0, 1))

    nom = jnp.sum(jnp.conj(probes) * psi, axis=0)
    denom = jnp.sum(jnp.square(jnp.abs(probes)), axis=0)
    obj = nom / (denom + 1e-6)

    return obj


@jax.jit
def set_amplitudes(obj, probe, ampl, shifts):
    probes = jax.vmap(jnp.roll, (None, 0, None))(probe, shifts, (0, 1))
    fwds = jax.vmap(prop.to_farfield)(obj[None] * probes)
    fwds = ampl * jnp.exp(1j * jnp.angle(fwds))
    bcks = jax.vmap(prop.from_farfield)(fwds)
    return bcks
