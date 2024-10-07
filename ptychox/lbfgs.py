import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize as opt
import warnings

from functools import wraps
from time import time

from jax.flatten_util import ravel_pytree
ravel_pytree_jit = jax.jit(lambda tree: ravel_pytree(tree)[0])

NOOP = lambda *args, **kwargs: None


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(func.__name__.ljust(20, '_') + "%.1e" % (end - start))
        return result

    return wrapper





def lbfgs(cost_function, grad_function, x0, tol=1e-6, maxiter=1000):
    """\
    Small wrapper around scipy LBFGS.
    """
    if not isinstance(x0, tuple):
        x0 = (x0,)

    # get flat coefficients and function to unravel coeffs from flat to tuple
    x0, unravel_coeffs = ravel_pytree(x0)
    unravel_coeffs = jax.jit(unravel_coeffs)

    def cost_wrap(x_flat):
        x = unravel_coeffs(x_flat)
        cost = cost_function(*x)
        cost = float(cost)
        if not np.isfinite(cost):
            warnings.warn("cost_function returned NaN")
        return cost


    def grad_wrap(x_flat):
        x = unravel_coeffs(x_flat)
        grad = grad_function(*x)
        grad = ravel_pytree_jit(grad)
        grad = np.array(grad, dtype="float64")
        if not np.all(np.isfinite(grad)):
            warnings.warn("grad_function returned NaN")
        return grad

    if not is_silent: 
        cost_wrap = timed(cost_wrap)        
        grad_wrap = timed(grad_wrap)        

    res = opt.minimize(
        cost_wrap,
        np.asarray(x0, dtype="float64"),
        method="L-BFGS-B",
        jac=grad_wrap,
        options={"maxiter": maxiter, "disp": None},
        tol=tol
    )
    if not res.success:
        print(res.message)

    x_opt = unravel_coeffs(res.x)

    return x_opt




def lbfgs_aux(
        cost_function,
        grad_function,
        x0,
        aux=None,
        tol=1e-6,
        maxiter=1000,
        move_gpu=False,
        is_silent=False,
        history=False,
        callback=None,
    ):
    """\
    Wrapper around scipy LBFGS with support for auxiliary arguments.

    Arguments:
    ----------
    cost_function : function
        cost_function(arg1, arg2, arg3, ..., aux1, aux2, aux3) -> float
    grad_function : function
        cost_function(arg1, arg2, arg3, ..., aux1, aux2, aux3) -> (grad1, grad2, ...)
        Can be jax.jit(jax.grad(grad_function, argnums=...))
        Arguments for which gradient is computed must stand at the front, before 
        auxiliary arguments.
    x0 : tuple (arg1_init, arg2_init, ...)
        Initial value(s) for optimization
    aux : tuple (aux1, aux2, ...)
        Auxiliary arguments. Are passed to `cost_function` and `grad_function`
        after arguments being optimized. Can be measured data or regularizer 
        parameters for example.
    tol : float
        Tolerance level. Passed directly to scipy LBFGS.
    maxiter : int
        Maximum number of iterations of scipy LBFGS
    move_gpu : bool
        Flag to move arrays in `aux` to GPU before optimization.
    is_silent : bool
        Flag to print timings of cost and grad functions
    history : bool
        Flag to create and return list of intermediate guesses `x`
    callback : function 
        Function f(x) receiving current guess `x` at every LBFGS iteration 

    Returns:
    --------
    x_opt : tuple
        Tuple containing optimized arguments as arrays.
    hist : None or list
        List if intermediate guesses `x`. Only if ``history == True``
    res : dict-like
        Return dict of scipy LBFGS

    """
    if not isinstance(x0, tuple):
        x0 = (x0,)

    if aux is None:
        aux = ()

    if move_gpu:
        aux = [jnp.asarray(a) if isinstance(a, np.ndarray) else a for a in aux]
        aux = tuple(aux)


    # get flat coefficients and function to unravel coeffs from flat to tuple
    x0, unravel_coeffs = ravel_pytree(x0)
    unravel_coeffs = jax.jit(unravel_coeffs)

    def cost_wrap(x_flat):
        x = unravel_coeffs(x_flat)
        full_args = x + aux
        cost = cost_function(*full_args)
        cost = float(cost)
        if not np.isfinite(cost):
            warnings.warn("cost_function returned NaN")

        return cost

    def grad_wrap(x_flat):
        x = unravel_coeffs(x_flat)
        full_args = x + aux
        grad = grad_function(*full_args)
        grad = ravel_pytree_jit(grad)
        grad = np.array(grad, dtype="float64")
        if not np.all(np.isfinite(grad)):
            warnings.warn("grad_function returned NaN")
        return grad

    # handle history and external callback
    f_history = NOOP
    hist = []
    if history:
        f_history = lambda x: hist.append(x)

    if callback is None:
        callback = NOOP

    internal_callback = NOOP
    if history or callback:
        def internal_callback(x_flat):        
            x = unravel_coeffs(x_flat)
            f_history(x)
            callback(x)

    if not is_silent: 
        cost_wrap = timed(cost_wrap)        
        grad_wrap = timed(grad_wrap)        

    res = opt.minimize(
        cost_wrap,
        np.array(x0, dtype="float64"),
        method="L-BFGS-B",
        jac=grad_wrap,
        options={"maxiter": maxiter, "disp": None},
        tol=tol,
        callback=internal_callback
    )
    if not res.success:
        print(res.message)

    x_opt = unravel_coeffs(res.x)

    return x_opt, hist, res
