import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd


@jax.jit
def to_farfield(field):
    # field = jnp.fft.ifftshift(field)
    fwd = jnp.fft.fft2(field, norm="ortho") 
    fwd = jnp.fft.fftshift(fwd)

    return fwd


@jax.jit
def from_farfield(field):
    field = jnp.fft.ifftshift(field)
    bck = jnp.fft.ifft2(field, norm="ortho") 
    # bck = jnp.fft.fftshift(bck)

    return bck


@jax.jit
def to_nearfield(field, psize, wlen, dist, M=1.):
    """\
    Fresnel propagator
    All distances in meters 
    """
    V, U = field.shape
    uu = 2. * jnp.pi * jnp.fft.fftfreq(U, psize)
    vv = 2. * jnp.pi * jnp.fft.fftfreq(V, psize)

    # magification == fan to parallel beam
    # (Fresnel scaling theorem)
    dist = dist / M

    k = 2 * jnp.pi / wlen
    r2 = jnp.square(vv)[:, None] + jnp.square(uu)[None]
    kernel = jnp.exp(1j * k * dist) * jnp.exp(-1j * dist / (2. * k) * r2)

    out = jnp.fft.ifft2(kernel * jnp.fft.fft2(field))

    return out
