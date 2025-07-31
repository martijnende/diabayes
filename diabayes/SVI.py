from typing import Callable, Tuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax_tqdm import scan_tqdm  # type:ignore
from jaxtyping import Array, Float

from diabayes.typedefs import Variables, _Params

ParticleArray = Float[Array, "N M"]
GradientArray: TypeAlias = ParticleArray


@eqx.filter_jit
def _distance(x1: ParticleArray, x2: ParticleArray) -> Float[Array, "N N"]:
    return jnp.square(x1 - x2).sum(axis=-1)


@eqx.filter_jit
def _exponential_kernel(
    x1: ParticleArray, x2: ParticleArray, h: float
) -> Float[Array, "N N"]:
    """Radial basis distance between two particles `x1` and `x2`"""
    dist_sq = _distance(x1, x2)
    return jnp.exp(-dist_sq / h)


@eqx.filter_jit
def _median_trick_h(x: ParticleArray) -> Float:
    """
    Compute the scaling factor `h` proportional to the
    squared median of the RMS distance between all pairs
    of particles.
    """
    dist_sq = jnp.square(x[:, None] - x[None]).sum(axis=-1)
    # Replace this one with jnp.nanmedian to avoid spreading NaNs?
    med_sq = jnp.median(jnp.sqrt(dist_sq))
    h = med_sq**2 / jnp.log(len(x) + 1)
    return h


@eqx.filter_jit
def compute_phi(
    x: ParticleArray, gradp: GradientArray, gradq: GradientArray
) -> GradientArray:
    """
    Compute the Stein variational gradients for a set of
    particles (`x`) and the gradients of the log-likelihood
    function (`gradp`) and the log-prior (`gradq`).

    Parameters
    ----------
    x : ParticleArray
        The set of invertible parameters ("particles")
        of shape (Nparticles, Ndimensions)
    gradp, gradq : GradientArray
        The gradients of the log-likelihood (`gradp`) and the
        log-prior (`gradq`) with respect to the invertible parameters.
        Has a shape (Nparticles, Ndimensions)

    Returns
    -------
    grad_x : Gradients
        The directional gradients for the particle updates, e.g.
        `new_x = x + step_size * grad_x`. Has the same
        shape as `x` and `gradp, gradq`.
    """
    h = _median_trick_h(x)
    map_over_a = (0, None, None)
    map_over_b = (None, 0, None)

    map1 = jax.vmap(_exponential_kernel, map_over_a, 0)
    kernel = jax.vmap(map1, map_over_b, 1)
    K = kernel(x, x, h)

    map1 = jax.vmap(jax.grad(_exponential_kernel), map_over_a, 0)
    grad_kernel = jax.vmap(map1, map_over_b, 1)
    grad_K = grad_kernel(x, x, h).sum(axis=0)

    grad_x = (jnp.matmul(K, gradp) + jnp.matmul(K, gradq) + grad_K) / len(gradp)
    return -grad_x


@eqx.filter_value_and_grad
def _log_likelihood(
    log_params: Float[Array, "..."],
    mu_obs: Float[Array, "Nt"],
    noise_std: Float,
    v0: Float,
    forward_fn: Callable[[Float[Array, "2"], Float[Array, "..."]], Variables],
) -> Float:
    # Transform the log_params by taking exponential
    params = jnp.exp(log_params)
    # Get initial values
    # TODO: replace with `get_steady_state` or something
    y0 = jnp.array([mu_obs[0], params[2] / v0])
    # Forward pass to get friction curve
    mu_hat = forward_fn(y0, params).mu
    # Log-likelihood of residuals
    p = -(jnp.square(mu_obs - mu_hat).mean() / (2 * noise_std**2))
    return p


"""
vmap _log_likelihood along particle axis. Since _log_likelihood is 
decorated with eqx.filter_value_and_grad, it returns the log-
likelihood and its gradient (hence it needs 2 values for the out_axes).
"""
mapped_log_likelihood = jax.vmap(
    _log_likelihood, in_axes=(0, None, None, None, None), out_axes=(0, 0)
)
