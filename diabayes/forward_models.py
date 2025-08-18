from dataclasses import make_dataclass
from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from diabayes.typedefs import (
    FrictionModel,
    InertialSpringBlockConstants,
    RSFConstants,
    RSFParams,
    SpringBlockConstants,
    StateDict,
    StateEvolution,
    StressTransfer,
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
    Omega = params.b * jnp.log(variables.theta * constants.v0 / params.Dc)
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
    return 1 - v * variables.theta / params.Dc


@eqx.filter_jit
def slip_rate(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    return v


@eqx.filter_jit
def springblock(
    t: Float,
    v: Float,
    v_partials: Float[Array, "..."],
    variables: Variables,
    dstate: Variables,
    constants: SpringBlockConstants,
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
    An inertial spring-block loading formulation:
    """
    mass_term = (
        constants.k * (constants.v_lp * t - variables.slip) - variables.mu
    ) / constants.M
    # The partials_term contains the summation of the partial derivatives of
    # v with respect to some variable y, times the time-derivative of y
    # The first partial derivative is v with respect to mu, and is excluded.
    partials_term = jnp.dot(v_partials.state.vals, dstate)
    return (mass_term - partials_term) / v_partials.mu


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
        state_evolution: Union[StateEvolution, Tuple[StateEvolution]],
        stress_transfer: StressTransfer,
    ) -> None:
        # Set the friction model and stress transfer model (easy...)
        self.friction_model = friction_model
        self.stress_transfer = stress_transfer
        # If a StatEvolution function is passed directly, make it a tuple
        if not isinstance(state_evolution, tuple):
            state_evolution = (state_evolution,)

        self.state_evolution_fns = state_evolution

        # Compile a function that calls each provided StateEvolution and
        # stacks the results in an array. This way, the "state" can host
        # an arbitrary number of variables (porosity, temperature, slip, ...)
        self.state_evolution = eqx.filter_jit(
            lambda v, variables, params, constants: jnp.stack(
                [fi(v, variables, params, constants) for fi in state_evolution]
            )
        )
        pass

    def create_variables(self) -> Variables:

        import ast
        import inspect

        accessed = set()

        def get_accessed_attrs(func, arg_name):
            """Helper routine to automatically extract the variable names"""
            tree = ast.parse(inspect.getsource(func))
            accessed = set()

            class Visitor(ast.NodeVisitor):
                def visit_Attribute(self, node: ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == arg_name:
                        accessed.add(node.attr)
                    self.generic_visit(node)

            Visitor().visit(tree)
            return accessed

        # Loop over the forward model components and evaluate
        # which variable names are accessed
        for func in (
            self.friction_model,
            self.stress_transfer,
            *self.state_evolution_fns,
        ):
            accessed.update(get_accessed_attrs(func, "variables"))

        # Ensure that we have at least one function that touches mu
        assert "mu" in accessed
        accessed.remove("mu")

        # Ensure that the number of state variables equals the number
        # of state evolution functions
        assert len(accessed) == len(self.state_evolution_fns)

        # Initialise values to -1
        state_obj = StateDict(keys=tuple(accessed), vals=-1.0 * jnp.ones(len(accessed)))
        variables = Variables(
            mu=jnp.asarray([-1.0], dtype=jnp.float64), state=state_obj
        )
        return variables

    @eqx.filter_jit
    def __call__(
        self,
        t: Float,
        variables: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Variables:
        # Calculate v and its partial derivatives with respect to
        # the variables (mu, state1, state2, ...)
        v, v_derivs = eqx.filter_value_and_grad(self.friction_model)(
            variables, params, friction_constants
        )
        # Rate of change of state variables
        dstate = self.state_evolution(v, variables, params, friction_constants)
        # Rate of change of mu (stress transfer)
        dmu = self.stress_transfer(t, v, v_derivs, variables, dstate, block_constants)
        # Create a new state container for dstate
        state_obj = StateDict(variables.state.keys, dstate)
        # Create a new variables container
        return Variables(mu=dmu, state=state_obj)
