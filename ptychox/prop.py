import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd


@jax.jit
def to_farfield(field):
    fwd = jnp.fft.fft2(field, norm="ortho") 
    fwd = jnp.fft.fftshift(fwd)

    return fwd


@jax.jit
def from_farfield(field):
    field = jnp.fft.ifftshift(field)
    bck = jnp.fft.ifft2(field, norm="ortho") 

    return bck


@jax.jit
def to_nearfield(field, psize, wlen, dist, M=1.):
    """\
    Fresnel propagator
    All distances in meters 
    """
    dim = field.shape
    freqs = [2. * pi * jnp.fft.fftfreq(n, psize) for n in dim]
    coords = jnp.meshgrid(*freqs, indexing="ij", sparse=True)

    R2 = jnp.zeros_like(field)
    for i in range(field.ndim):
        R2 += jnp.square(coords[i])

    dist = dist / M
    k = 2 * np.pi / wlen
    kernel = jnp.exp(1j * k * dist) * jnp.exp(-1j * dist / (2. * k) * R2)
    out = jnp.fft.ifftn(kernel * jnp.fft.fftn(field))

    return out
