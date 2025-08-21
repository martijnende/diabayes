from typing import Callable, Dict, Generic, Iterable, Tuple, TypeVar, Union

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
    StressTransfer,
    Variables,
    _BlockConstants,
    _Constants,
    _Params,
)

BC = TypeVar("BC", bound=_BlockConstants)


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
    v = constants.v0 * jnp.exp((variables.mu - constants.mu0 - Omega) / params.a)
    return jnp.squeeze(v)


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
    v_partials: Variables,
    variables: Variables,
    dstate: Float[Array, "..."],
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


class Forward(Generic[BC]):
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
        state_evolution: Dict[str, Callable],
        stress_transfer: StressTransfer[BC],
    ) -> None:
        # Set the friction model and stress transfer model (easy...)
        self.friction_model = friction_model
        self.stress_transfer = stress_transfer

        state_obj = StateDict(
            keys=tuple(state_evolution.keys()),
            vals=-1.0 * jnp.ones(len(state_evolution)),
        )
        variables = Variables(
            mu=jnp.asarray([-1.0], dtype=jnp.float64), state=state_obj
        )
        self.variables = variables

        # Compile a function that calls each provided state_evolution item and
        # stacks the results in an array. This way, the "state" can host
        # an arbitrary number of variables (porosity, temperature, slip, ...)
        self.state_evolution = eqx.filter_jit(
            lambda v, variables, params, constants: jnp.stack(
                [
                    fi(v, variables, params, constants)
                    for _, fi in state_evolution.items()
                ]
            )
        )
        pass

    @staticmethod
    def inspect(fn: Union[Callable, Tuple[Callable, ...]]) -> None:

        import ast
        import inspect

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

        if not isinstance(fn, Iterable):
            fn = (fn,)

        accessed = set()

        for fn_i in fn:
            accessed.update(get_accessed_attrs(fn_i, "variables"))

        if "mu" in accessed:
            accessed.remove("mu")

        print("Auto-detected the following state variable names:")
        print(accessed)

    def set_initial_values(self, **kwargs) -> None:
        self.variables = self.variables.set_values(**kwargs)

    @eqx.filter_jit
    def __call__(
        self,
        t: Float,
        variables: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: BC,
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
