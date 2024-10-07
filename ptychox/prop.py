import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

τ = 2 * np.pi


@jax.jit
def to_farfield(field):
    field = jnp.fft.ifftshift(field)
    fwd = jnp.fft.fft2(field, norm="ortho") 
    fwd = jnp.fft.fftshift(fwd)

    return fwd



@jax.jit
def from_farfield(field):
    field = jnp.fft.ifftshift(field)
    bck = jnp.fft.ifft2(field, norm="ortho")
    bck = jnp.fft.fftshift(bck)

    return bck



@jax.jit
def to_nearfield(field, kernel):
    out = jnp.fft.ifftn(kernel * jnp.fft.fftn(field))

    return out



@jax.jit
def from_nearfield(field, kernel):
    out = jnp.fft.ifftn(jnp.conj(kernel) * jnp.fft.fftn(field))

    return out



@partial(jax.jit, static_argnames=("dim", "is_physical"))
def get_kernel_fresnel(dim, psize, wlen, dist, M=1., is_physical=False):
    """\
    Construct kernel for Fresnel propagator
    All distances in meters 
    `M` : geometric magnification and used for Fresnel scaling theorem
    """
    freqs = [τ * np.fft.fftfreq(n, psize) for n in dim]
    coords = jnp.meshgrid(*freqs, indexing="ij", sparse=True)

    R2 = jnp.zeros(dim, "float32")
    for c in coords:
        R2 += jnp.square(c)

    dist = dist / M
    k = τ / wlen
    kernel = jnp.exp(-1j * dist / (2. * k) * R2)

    if is_physical:
        # phase shift from distance only, doesn't matter for contrast
        kernel = kernel * jnp.exp(1j * k * dist)

    return kernel



@partial(jax.jit, static_argnames="dim")
def get_kernel_angular_spectrum(dim, psize, wlen, dist):
    """\
    Construct kernel for angular spectrum propagator
    All distances in meters

    Implies `is_physical=False`, such that we can remove term for pure propagation
    from sqrt in exponent and make it numerically stable.
    """
    psize = psize / wlen
    dist = dist / wlen

    freqs = [jnp.fft.fftfreq(n, psize) for n in dim]
    coords = jnp.meshgrid(*freqs, indexing="ij", sparse=True)

    R2 = jnp.zeros(dim, "float32")
    for c in coords:
        R2 += jnp.square(c)

    kernel = jnp.sqrt(1 - R2 + 0j) + 1
    kernel = -τ * dist * R2 / kernel
    kernel = jnp.exp(1j * kernel)

    return kernel
