from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax_tqdm import scan_tqdm  # type:ignore
from jaxtyping import Array, Float

from diabayes.typedefs import Variables, _Params

Particles = Float[Array, "N M"]
Gradients = Float[Array, "N M"]


@eqx.filter_jit
def _distance(x: Particles) -> Float[Array, "N N"]:
    return jnp.square(x[:, None] - x[None]).sum(axis=-1)


@eqx.filter_value_and_grad
def _exponential_kernel(theta: Particles, h: float) -> Float[Array, "N N"]:
    """Radial basis distance between the particles `theta`"""
    pairwise_dists_sq = _distance(theta)
    return jnp.exp(-pairwise_dists_sq / h)


@eqx.filter_jit
def _median_trick_h(theta: Particles) -> Float:
    """
    Compute the scaling factor `h` proportional to the
    squared median of the RMS distance between all pairs
    of particles.
    """
    pairwise_dists_sq = _distance(theta)
    # Replace this one with jnp.nanmedian to avoid spreading NaNs?
    med_sq = jnp.median(jnp.sqrt(pairwise_dists_sq))
    h = med_sq**2 / jnp.log(theta.shape[0] + 1)
    return h


@eqx.filter_jit
def compute_phi(theta: Particles, gradp: Gradients, gradq: Gradients) -> Gradients:
    """
    Compute the Stein variational gradients for a set of
    particles (`theta`) and the gradients of the log-likelihood
    function (`gradp`) and the log-prior (`gradq`).
    """
    h = _median_trick_h(theta)
    K, grad_K = _exponential_kernel(theta, h)
    grad_theta = (jnp.matmul(K, gradp) + jnp.matmul(K, gradq) + grad_K) / gradp.shape[0]
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
