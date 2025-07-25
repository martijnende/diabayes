from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from diabayes.typedefs import (
    FrictionModel,
    RSFConstants,
    RSFParams,
    SpringBlockConstants,
    StateEvolution,
    Variables,
    _BlockConstants,
    _Constants,
    _Params,
)


@eqx.filter_jit
def rsf(variables: Variables, params: RSFParams, constants: RSFConstants) -> Float:
    r"""
    The classical rate-and-state friction law, given by:

    .. math::

        v(\mu, \theta) = v_0 \exp \left( \frac{1}{a} \left[ \mu - \mu_0 - b \log \left( \frac{v_0 \theta}{D_c} \right) \right] \right)

    Parameters
    ----------
    variables : Variables
        The friction coefficient `mu` and state parameter `theta`
    params : RSFParams
        The rate-and-state parameters `a`, `b`, and `D_c`
    constants : RSFConstants
        The constant parameters `mu0` and `v0`

    Returns
    -------
    v : Float
        The instantaneous slip rate in the same units as `v0`
    """
    Omega = params.b * jnp.log(variables.state * constants.v0 / params.Dc)
    return constants.v0 * jnp.exp((variables.mu - constants.mu0 - Omega) / params.a)


@eqx.filter_jit
def ageing_law(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    r"""
    The conventional ageing law state evolution formulation:

    .. math::

        \frac{\mathrm{d}\theta}{\mathrm{d}t} = 1 - \frac{v \theta}{D_c}

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient `mu` and state parameter `theta`
    params : RSFParams
        The rate-and-state parameters `a`, `b`, and `D_c`
    constants : RSFConstants
        The constant parameters `mu0` and `v0`

    Returns
    -------
    dtheta : Float
        The rate of change of the state variable `theta` [s/s]
    """
    return 1 - v * variables.state / params.Dc


@eqx.filter_jit
def springblock(
    v: Float, variables: Variables, constants: SpringBlockConstants
) -> Float:
    r"""
    A conventional (non-inertial) spring-block loading formulation:

    .. math::

        \frac{\mathrm{d} \mu}{\mathrm{d} t} = k \left ( v_{lp} - v(t) \right)

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient `mu` and state parameter `theta`. This argument
        is not used, but included for call signature consistency.
    constants : SpringBlockConstants
        The constant parameters stiffness `k` (units of "friction per metre")
        and load-point velocity `v_lp` (same units as `v`).

    Returns
    -------
    dmu : Float
        The rate of change of the friction coefficient `mu` [1/s]
    """
    return constants.k * (constants.v_lp - v)


class Forward:
    r"""
    The `Forward` class assembles the various components that comprise
    a forward model, such that `Forward.__call__` takes some variables
    and returns the rate of change of these variables, i.e.:

    .. math::

        \frac{\mathrm{d} \vec{X}}{\mathrm{d}t} = f \left( \vec{X} \right)

    This forward ODE can then be solved by any ODE solver.

    The `Forward` class is instantiated by providing a friction model
    (of type `FrictionModel`), a "state" evolution law (of type `StateEvolution`),
    and a stress transfer model.

    Examples
    --------
    >>> from diabayes.forward_models import ageing_law, rsf, springblock, Forward
    >>> foward_model = Forward(rsf, ageing_law, springblock)
    >>> X_dot = forward_model(variables=..., params=..., friction_constants=..., block_constants=...)
    """

    def __init__(
        self,
        friction_model: FrictionModel,
        state_evolution: StateEvolution,
        block_type: Callable[[Float, Variables, _BlockConstants], Float],
    ) -> None:
        self.friction_model = friction_model
        self.state_evolution = state_evolution
        self.stress_transfer = block_type
        pass

    @eqx.filter_jit
    def __call__(
        self,
        variables: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Variables:
        v = self.friction_model(variables, params, friction_constants)
        dstate = self.state_evolution(v, variables, params, friction_constants)
        dmu = self.stress_transfer(v, variables, block_constants)
        return Variables(mu=dmu, state=dstate)
