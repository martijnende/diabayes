from typing import Callable, Dict, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from diabayes.typedefs import (
    FrictionModel,
    StateDict,
    StressTransfer,
    Variables,
    _BlockConstants,
    _Constants,
    _Params,
)

from diabayes.physics import (
    rsf,
    ageing_law,
    aging_law,
    slip_law,
    cns,
    cns_porosity,
    sundman_cns,
    steady_state_porosity,
    slip_rate,
    springblock,
    inertial_springblock,
)

BC = TypeVar("BC", bound=_BlockConstants)


@eqx.filter_jit
def _identity(*args, **kwargs) -> Float:
    return 1


class Forward(Generic[BC]):
    r"""
    The ``Forward`` class assembles the various components that comprise
    a forward model, such that ``Forward.__call__`` takes some variables
    and returns the rate of change of these variables

    .. math::

        \frac{\mathrm{d} \vec{X}}{\mathrm{d}t} = f \left( \vec{X} \right)

    This forward ODE can then be solved by any ODE solver.

    The ``Forward`` class is instantiated by providing a friction model,
    a "state" evolution law, and a stress transfer model.

    Examples
    --------
    >>> from diabayes.forward_models import ageing_law, rsf, springblock, Forward
    >>> foward_model = Forward(rsf, {"theta": ageing_law}, springblock)
    >>> X_dot = forward_model(variables=..., params=..., friction_constants=..., block_constants=...)
    """

    def __init__(
        self,
        friction_model: FrictionModel,
        state_evolution: Dict[str, Callable],
        stress_transfer: StressTransfer[BC],
        sundman: Callable | None = None,
    ) -> None:
        # Set the friction model and stress transfer model (easy...)
        self.friction_model = friction_model
        self.stress_transfer = stress_transfer

        state_evolution["t"] = _identity
        state_obj = StateDict(
            keys=tuple(state_evolution.keys()),
            vals=-1.0 * jnp.ones(len(state_evolution)),
        )
        variables = Variables(
            mu=jnp.asarray([-1.0], dtype=jnp.float64), state=state_obj
        )
        self.variables = variables
        self.sundman = sundman

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

    def set_initial_values(self, **kwargs) -> None:
        """
        Set the initial values of the ``variables``. This sets the
        values as ``self.variables``, which is a ``Variables`` object.

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs of variable names and corresponding values

        """
        scalars = {k: float(jnp.atleast_1d(v).item()) for k, v in kwargs.items()}
        if scalars.get("t") is None:
            scalars["t"] = 0.0
        self.variables = self.variables.set_values(**scalars)

    @eqx.filter_jit
    def __call__(
        self,
        t: Float,
        variables: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: BC,
    ) -> Variables:
        """
        Calculate the rate of change of the variables, defining the ODE
        to be solved.

        Parameters
        ----------
        t : Float
            Current value of time [s]
        variables : diabayes.Variables
            The instantaneous values of the variables for which the time
            derivative will be computed
        params : _Params
            The forward model (invertible) parameters
        friction_constants : _Constants
            The forward model constants
        block_constants : _BlockConstants
            The constants associated with the stress transfer

        Returns
        -------
        dvars : Variables
            The instantaneous rate of change of the ``variables``
        """
        # Calculate v and its partial derivatives with respect to
        # the variables (mu, state1, state2, ...)
        v, v_derivs = eqx.filter_value_and_grad(self.friction_model)(
            variables, params, friction_constants
        )
        # Rate of change of state variables
        dstate = self.state_evolution(v, variables, params, friction_constants)

        # When using a Sundman transformation, the variable `t` is actually
        # the transformed variable, and so we need to transform it back to
        # physical time.
        t_phys = variables.t if self.sundman is not None else t
        # Rate of change of mu (stress transfer)
        dmu = self.stress_transfer(
            t_phys, v, v_derivs, variables, dstate, block_constants
        )

        # Sundman transformation
        if self.sundman is not None:
            sund = self.sundman(v, variables, params, friction_constants)
            dstate = dstate * sund
            dmu = dmu * sund

        # Create a new state container for dstate
        state_obj = StateDict(variables.state.keys, dstate)
        # Create a new variables container
        return Variables(mu=dmu, state=state_obj)

    # Alias for documentation
    call = __call__
