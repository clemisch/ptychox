import jax
import jax.numpy as jnp

from . import prop, utils

EPS = 1e-6


@jax.jit
def update_object(psi_diff, obj, probe, shift, alpha=1.0):
    probe = utils.get_shifted_roll(probe, shift)
    nom = jnp.conj(probe) 
    denom = jnp.max(jnp.square(jnp.abs(probe)))
    step = nom / (denom + EPS) * psi_diff
    obj = obj + alpha * step

    return obj


@jax.jit
def update_probe(psi_diff, obj, probe, shift, beta=1.0):
    obj = utils.get_shifted_roll(obj, -shift)
    nom = jnp.conj(obj) 
    denom = jnp.max(jnp.square(jnp.abs(obj)))
    step = nom / (denom + EPS) * psi_diff
    probe = probe + beta * utils.get_shifted_roll(step, -shift)

    return probe


@jax.jit
def update_shot(obj, probe, shift, ampl, alpha=1.0, beta=1.0):
    psi = obj * utils.get_shifted_roll(probe, shift)
    fwd = prop.to_farfield(psi)
    fwd = utils.set_magn(fwd, ampl)
    psi_new = prop.from_farfield(fwd)
    psi_diff = psi_new - psi

    obj_new = update_object(psi_diff, obj, probe, shift, alpha)
    probe_new = update_probe(psi_diff, obj, probe, shift, beta)

    return obj_new, probe_new


@jax.jit
def update_shots(obj, probe, shifts, ampls, alpha=1.0, beta=1.0):
    def worker(carry, x):
        obj, probe = carry
        shift, ampl = x
        obj, probe = update_shot(obj, probe, shift, ampl)
        return (obj, probe), None

    (obj, probe), _ = jax.lax.scan(
        worker,
        (obj, probe),
        (shifts, ampls),
    )

    return obj, probe
