import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from diabayes.typedefs import (
    InertialSpringBlockConstants,
    SpringBlockConstants,
    Variables,
)


@eqx.filter_jit
def springblock(
    t: Float,
    v: Float,
    v_partials: Variables,
    variables: Variables,
    dstate: Float[Array, "..."],
    constants: SpringBlockConstants,
) -> Float:
    r"""
    A conventional (non-inertial) spring-block loading formulation

    .. math::

        \frac{\mathrm{d} \mu}{\mathrm{d} t} = k \left ( v_{lp} - v(t) \right)

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The instantaneous variables. This argument is not used,
        but included for call signature consistency.
    constants : SpringBlockConstants
        The constant parameters stiffness ``k`` (units of "friction per metre")
        and load-point velocity ``v_lp`` (same units as ``v``).

    Returns
    -------
    dmu : Float
        The rate of change of the friction coefficient [1/s]
    """
    return constants.k * (constants.v_lp - v)


@eqx.filter_jit
def inertial_springblock(
    t: Float,
    v: Float,
    v_partials: Variables,
    variables: Variables,
    dstate: Float[Array, "..."],
    constants: InertialSpringBlockConstants,
) -> Float:
    r"""
    An inertial spring-block loading formulation

    .. math::
        \frac{\mathrm{d} \mu}{\mathrm{d} t} = \left[ \frac{\partial v}{\partial \mu} \right]^{-1} \left( \frac{1}{M} \left[ k \left( v_{lp} t - x \right) - \mu \right] - \frac{\partial v}{\partial \theta} \frac{\mathrm{d} \theta}{\mathrm{d} t} - \dots \right)

    The acceleration term in the classical inertial spring-block formulation
    is decomposed into its partial derivatives, avoiding the need for solving
    a second-order ODE. These partial derivatives (``v_partials``) are computed
    using the JAX autodiff framework.

    Notes
    -----
    This formulation is rather stiff, and for certain parameter values could
    lead to extremely small time steps necessary to maintain numerical
    accuracy. It is recommended to use a conventional (non-inenrtial)
    ``springblock`` formulation for basic velocity-steps and slide-hold-slide
    simuilations. Inertia is only really needed for stick-slip simulations.

    Parameters
    ----------
    t : Float
        Current value of time [s]
    v : Float
        Instantaneous fault slip rate [m/s]
    v_partials : Variables
        The partial derivatives of slip rate to the relevant variables
        (friction and state variables)
    variables: Variables
        The instantaneous values of the variables: friction (``mu``),
        slip (``slip``), and other state variables (not used)
    dstate : Array
        The time derivatives of the state variables. The radiation term
        is ``v_partials @ dstate`` (excluding ``mu``)
    constants : InertialSpringBlockConstants
        The spring-block constants containing the mass term ``M``, the
        stiffness ``k``, and the load-point velocity ``v_lp``

    Returns
    -------
    dmu : Float
        The rate of change of the friction coefficient [1/s]

    """
    mass_term = (
        constants.k * (constants.v_lp * t - variables.slip) - variables.mu
    ) / constants.M
    # The partials_term contains the summation of the partial derivatives of
    # v with respect to some variable y, times the time-derivative of y
    # The first partial derivative is v with respect to mu, and is excluded.
    partials_term = jnp.dot(v_partials.state.vals, dstate)
    return (mass_term - partials_term) / v_partials.mu
