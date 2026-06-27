import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from functools import partial

from . import prop, utils


EPS = 1e-6


@jax.jit
def get_exit_waves(obj, probe, shifts):
    return jax.vmap(utils.get_exit_wave, (None, None, 0))(obj, probe, shifts)


@jax.jit
def project_amplitudes(psi, ampls):
    fwds = jax.vmap(prop.to_farfield)(psi)
    fwds = utils.set_magn(fwds, ampls)
    return jax.vmap(prop.from_farfield)(fwds)


@jax.jit
def set_amplitudes(obj, probe, ampls, shifts):
    return project_amplitudes(get_exit_waves(obj, probe, shifts), ampls)


@jax.jit
def get_probe(psi, obj, shifts):
    shifts_int, shifts_rem = jnp.divmod(shifts, 1.)
    obj = jax.vmap(utils.get_obj_crop, (None, 0, None))(
        obj, shifts_int.astype(jnp.int32), psi.shape[1:]
    )

    shift_probe = jax.vmap(utils.get_shifted_sinc, (None, 0))
    unshift = jax.vmap(utils.get_shifted_sinc, (0, 0))

    rhs = jnp.sum(
        unshift(jnp.conj(obj) * psi, shifts_rem),
        axis=0,
    )

    def normal(probe):
        probes = shift_probe(probe, -shifts_rem)
        weighted = jnp.abs(obj) ** 2 * probes
        return jnp.sum(unshift(weighted, shifts_rem), axis=0) + EPS * probe

    probe, _ = cg(normal, rhs, tol=1e-5, maxiter=5)

    return probe


@partial(jax.jit, static_argnames="o_shape")
def get_object(psi, probe, shifts, o_shape):
    shifts_int, shifts_rem = jnp.divmod(shifts, 1.)
    shifts_int = shifts_int.astype(jnp.int32)
    probes = jax.vmap(utils.get_shifted_sinc, (None, 0))(
        probe, -shifts_rem
    )

    nom = jnp.sum(
        jax.vmap(utils.get_obj_uncrop, (0, 0, None))(
            jnp.conj(probes) * psi, shifts_int, o_shape
        ),
        axis=0,
    )
    denom = jnp.sum(
        jax.vmap(utils.get_obj_uncrop, (0, 0, None))(
            jnp.abs(probes) ** 2, shifts_int, o_shape
        ),
        axis=0,
    )

    obj = nom / (denom + EPS)

    return obj


@partial(jax.jit, static_argnames="update_probe")
def project_overlap(psi, obj, probe, shifts, update_probe=True):
    """Project exit waves onto the shared object/probe model."""
    obj = get_object(psi, probe, shifts, obj.shape)
    if update_probe:
        probe = get_probe(psi, obj, shifts)
        # Refit the object after changing the probe so the returned factors are
        # mutually consistent rather than simultaneous estimates.
        obj = get_object(psi, probe, shifts, obj.shape)
    psi_overlap = get_exit_waves(obj, probe, shifts)
    return psi_overlap, obj, probe


@partial(jax.jit, static_argnames="update_probe")
def step(psi, obj, probe, ampls, shifts, beta=1.0, update_probe=True):
    """Perform one Difference Map iteration.

    ``psi`` is the persistent DM state.  The object and probe are the current
    factorization used to evaluate the non-convex overlap projection.
    """
    psi_overlap, obj, probe = project_overlap(
        psi, obj, probe, shifts, update_probe
    )
    psi_modulus = project_amplitudes(2 * psi_overlap - psi, ampls)
    psi = psi + beta * (psi_modulus - psi_overlap)
    return psi, obj, probe
