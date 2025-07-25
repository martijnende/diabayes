from typing import Callable, TypeAlias

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
def _distance(x: ParticleArray) -> Float[Array, "N N"]:
    return jnp.square(x[:, None] - x[None]).sum(axis=-1)


@eqx.filter_value_and_grad
def _exponential_kernel(theta: ParticleArray, h: float) -> Float[Array, "N N"]:
    """Radial basis distance between the particles `theta`"""
    pairwise_dists_sq = _distance(theta)
    return jnp.exp(-pairwise_dists_sq / h)


@eqx.filter_jit
def _median_trick_h(theta: ParticleArray) -> Float:
    """
    Compute the scaling factor `h` proportional to the
    squared median of the RMS distance between all pairs
    of particles.
    """
    pairwise_dists_sq = _distance(theta)
    # Replace this one with jnp.nanmedian to avoid spreading NaNs?
    med_sq = jnp.median(jnp.sqrt(pairwise_dists_sq))
    h = med_sq**2 / jnp.log(len(theta) + 1)
    return h


@eqx.filter_jit
def compute_phi(
    theta: ParticleArray, gradp: GradientArray, gradq: GradientArray
) -> GradientArray:
    """
    Compute the Stein variational gradients for a set of
    particles (`theta`) and the gradients of the log-likelihood
    function (`gradp`) and the log-prior (`gradq`).

    Parameters
    ----------
    theta : ParticleArray
        The set of invertible parameters ("particles")
        of shape (Nparticles, Ndimensions)
    gradp, gradq : GradientArray
        The gradients of the log-likelihood (`gradp`) and the
        log-prior (`gradq`) with respect to the invertible parameters.
        Has a shape (Nparticles, Ndimensions)

    Returns
    -------
    grad_theta : Gradients
        The directional gradients for the particle updates, e.g.
        `new_theta = theta + step_size * grad_theta`. Has the same
        shape as `theta` and `gradp, gradq`.
    """
    h = _median_trick_h(theta)
    K, grad_K = _exponential_kernel(theta, h)
    grad_theta = (jnp.matmul(K, gradp) + jnp.matmul(K, gradq) + grad_K) / len(gradp)
    return -grad_theta


@eqx.filter_value_and_grad
def _log_likelihood(
    log_params: _Params,
    mu_obs: Float[Array, "Nt"],
    sigma: Float,
    forward_fn: Callable[[_Params], Variables],
) -> Float:
    # Transform the log_params by taking exponential
    params = type(log_params).from_array(jnp.exp(log_params.to_array()))
    # Forward pass to get friction curve
    mu_hat = forward_fn(params).mu
    # Log-likelihood of residuals
    p = -(jnp.square(mu_obs - mu_hat).mean() / (2 * sigma**2))
    return p


"""
vmap _log_likelihood along particle axis. Since _log_likelihood is 
decorated with eqx.filter_value_and_grad, it returns the log-
likelihood and its gradient (hence it needs 2 values for the out_axes).
"""
mapped_log_likelihood = jax.vmap(
    _log_likelihood, in_axes=(0, None, None, None), out_axes=(0, 0)
)
