import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from diabayes.typedefs import (
    CNSConstants,
    CNSParams,
    Variables,
)


def steady_state_porosity(
    v: Float, mu: Float, params: CNSParams, constants: CNSConstants
) -> Float:
    Z = params.xi * constants.v0 / v
    A = 2 * params.beta * (params.phi_c - constants.phi0)
    B = (A * mu - 1) * Z / (1 + mu * Z)
    tan_psi_ss = 0.5 * B * (1 - jnp.sqrt(1 + 4 * A / (B * (A * mu - 1))))
    phi_ss = params.phi_c - tan_psi_ss / (2 * params.beta)
    return phi_ss


@eqx.filter_jit
def sundman_cns(
    v: Float, variables: Variables, params: CNSParams, constants: CNSConstants
) -> Float:
    return jnp.squeeze(params.phi_c - variables.phi)


@eqx.filter_jit
def _porosity_func(
    variables: Variables, params: CNSParams, constants: CNSConstants
) -> Float:
    r"""
    The porosity modulation function proposed by [1]_

    .. math::

        f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}

    Parameters
    ----------
    variables : Variables
        The friction coefficient ``mu`` and gouge porosity ``phi``
    params : CNSParams
        The CNS parameters ``alpha``, ``beta``, ``phi_c``, ``xi``, and ``mu0``
    constants : CNSConstants
        The constant parameters ``v0``, ``h``, and ``phi0``

    Returns
    -------
    f_phi : Float
        The porosity modulation function

    References
    ----------
    .. [1] van den Ende, Chen, Ampuero, Niemeijer (2018), A comparison between rate-and-state friction and microphysical models, based on numerical simulations of fault slip, doi:10.1016/j.tecto.2017.11.040
    """
    phi = variables.phi
    phi0 = constants.phi0
    phi_c = params.phi_c
    f_phi = (phi - phi0) / (phi_c - phi)
    return f_phi


@eqx.filter_jit
def cns(variables: Variables, params: CNSParams, constants: CNSConstants) -> Float:
    r"""
    The Chen-Niemeijer-Spiers (CNS) friction law

    .. math::

        v(\mu, \phi) = v_0 \left( v_{\text{gr}}(\mu, \phi) + v_{\text{creep}}(\mu, \phi) \right)

        v_{\text{gr}} = \exp \left( \frac{\mu \left[1 - \mu_0 \tan \psi \right] - \mu_0 - \tan \psi}{\alpha \left[ 1 + \mu \tan \psi \right]} \right)

        v_{\text{creep}} = \xi \mu f(\phi)

        \tan \psi = 2 \beta \left( \phi_c - \phi \right)

        f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}


    Parameters
    ----------
    variables : Variables
        The friction coefficient ``mu`` and gouge porosity ``phi``
    params : CNSParams
        The CNS parameters ``alpha``, ``beta``, ``phi_c``, ``xi``, and ``mu0``
    constants : CNSConstants
        The constant parameters ``v0``, ``h``, and ``phi0``

    Returns
    -------
    v : Float
        The instantaneous slip rate in the same units as ``v0``
    """

    # Granular flow components
    tan_psi = 2 * params.beta * (params.phi_c - variables.phi)
    A = variables.mu * (1 - params.mu0 * tan_psi) - params.mu0 - tan_psi
    B = params.alpha * (1 + variables.mu * tan_psi)
    v_gr = jnp.exp(A / B)

    # Creep components
    f_phi = _porosity_func(variables, params, constants)
    v_creep = params.xi * variables.mu * f_phi

    # Assembly
    v = constants.v0 * (v_gr + v_creep)

    return jnp.squeeze(v)


@eqx.filter_jit
def cns_porosity(
    v: Float, variables: Variables, params: CNSParams, constants: CNSConstants
) -> Float:
    r"""
    The porosity (state) evolution for the Chen-Niemeijer-Spiers model

    .. math::

        \frac{\mathrm{d}\phi}{\mathrm{d}t} = - \left(1 - \phi \right) \left(\dot{\varepsilon}_{\text{gr}} + \dot{\varepsilon}_{\text{creep}} \right)

        \dot{\varepsilon}_{\text{gr}} = - \frac{\tan \psi}{h} \left( v - v_{\text{creep}} \right)

        \dot{\varepsilon}_{\text{creep}} = \frac{v_0}{h} \xi f(\phi)

        v_{\text{creep}} = v_0 \xi f(\phi) \mu

        \tan \psi = 2 \alpha \left( \phi_c - \phi \right)

        f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}


    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient ``mu`` and gouge porosity ``phi``
    params : CNSParams
        The CNS parameters ``alpha``, ``beta``, ``phi_c``, ``xi``, and ``mu0``
    constants : CNSConstants
        The constant parameters ``v0``, ``h``, and ``phi0``

    Returns
    -------
    dphi : Float
        The rate of change of the porosity [1/s]
    """

    # Creep components
    f_phi = _porosity_func(variables, params, constants)
    e_creep = (constants.v0 / constants.h) * params.xi * f_phi
    v_creep = constants.h * e_creep * variables.mu

    # Granular flow components
    tan_psi = 2 * params.beta * (params.phi_c - variables.phi)
    e_gr = -tan_psi * (v - v_creep) / constants.h

    # Assembly
    dphi = -(1 - variables.phi) * (e_gr + e_creep)

    return jnp.squeeze(dphi)
