import jax
import jax.numpy as jnp


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
def get_nearfield(field):
    return field
