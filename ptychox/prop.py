import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

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



def get_kernel_fresnel(dim, psize, wlen, dist, M=1.):
    """\
    Construct kernel for Fresnel propagator
    All distances in meters 
    `M` : geometric magnification and used for Fresnel scaling theorem
    """
    freqs = [τ * np.fft.fftfreq(n, psize) for n in dim]
    coords = np.meshgrid(*freqs, indexing="ij", sparse=True)

    R2 = np.zeros(dim, "float64")
    for c in coords:
        R2 += np.square(c)

    dist = dist / M
    k = τ / wlen
    kernel = np.exp(1j * k * dist) * np.exp(-1j * dist / (2. * k) * R2)

    return kernel



def get_kernel_angular_spectrum(dim, psize, wlen, dist):
    """\
    Construct kernel for angular spectrum propagator
    All distances in meters 
    """
    freqs = [np.fft.fftfreq(n, psize) for n in dim]
    coords = np.meshgrid(*freqs, indexing="ij", sparse=True)

    R2 = np.zeros(dim, "float64")
    for c in coords:
        R2 += np.square(c)

    kernel = np.square(1. / wlen) - R2
    kernel = 1j * τ * dist * np.sqrt(kernel)
    kernel = np.exp(kernel)

    return kernel
