import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from diabayes.typedefs import (
    RSFConstants,
    RSFParams,
    Variables,
)


@eqx.filter_jit
def rsf(variables: Variables, params: RSFParams, constants: RSFConstants) -> Float:
    r"""
    The classical rate-and-state friction law

    .. math::

        v(\mu, \theta) = v_0 \exp \left( \frac{1}{a} \left[ \mu - \mu_0 - b \log \left( \frac{v_0 \theta}{D_c} \right) \right] \right)

    Parameters
    ----------
    variables : Variables
        The friction coefficient ``mu`` and state parameter ``theta``
    params : RSFParams
        The rate-and-state parameters ``a``, ``b``, and ``D_c``
    constants : RSFConstants
        The constant parameters ``mu0`` and ``v0``

    Returns
    -------
    v : Float
        The instantaneous slip rate in the same units as ``v0``
    """
    Omega = params.b * jnp.log(variables.theta * constants.v0 / params.Dc)
    v = constants.v0 * jnp.exp((variables.mu - constants.mu0 - Omega) / params.a)
    return jnp.squeeze(v)


@eqx.filter_jit
def ageing_law(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    r"""
    The conventional ageing law state evolution formulation

    .. math::

        \frac{\mathrm{d}\theta}{\mathrm{d}t} = 1 - \frac{v \theta}{D_c}

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient ``mu`` and state parameter ``theta``
    params : RSFParams
        The rate-and-state parameters, including ``D_c``
    constants : RSFConstants
        The constant parameters ``mu0`` and ``v0`` (not used)

    Returns
    -------
    dtheta : Float
        The rate of change of the state variable [s/s]
    """
    return 1 - v * variables.theta / params.Dc


@eqx.filter_jit
def aging_law(*args, **kwargs):
    """
    Alias for the ``ageing_law`` function
    """
    return ageing_law(*args, **kwargs)


@eqx.filter_jit
def slip_law(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    r"""
    The conventional slip law state evolution formulation

    .. math::

        \frac{\mathrm{d}\theta}{\mathrm{d}t} = - \frac{v \theta}{D_c} \ln \left( \frac{v \theta}{D_c} \right)

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient ``mu`` and state parameter ``theta``
    params : RSFParams
        The rate-and-state parameters, including ``D_c``
    constants : RSFConstants
        The constant parameters ``mu0`` and ``v0`` (not used)

    Returns
    -------
    dtheta : Float
        The rate of change of the state variable [s/s]
    """
    O = v * variables.theta / params.Dc
    return -O * jnp.log(O)
