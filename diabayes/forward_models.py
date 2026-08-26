from typing import Callable, Dict, Generic, Iterable, Tuple, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from diabayes.typedefs import (
    FrictionModel,
    InertialSpringBlockConstants,
    CNSConstants,
    CNSParams,
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
        The CNS parameters ``alpha``, ``phi_c``, ``z``, ``mu0``, ``v0`` and ``a``
    constants : CNSConstants
        The constant parameters ``h`` and ``phi0``

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

        v(\mu, \phi) = v_{\text{gr}}(\mu, \phi) + v_{\text{creep}}(\mu, \phi)

        v_{\text{gr}} = v_0 \exp \left( \frac{\mu \left[1 - \mu_0 \tan \psi \right] - \mu_0 - \tan \psi}{a \left[ 1 + \mu \tan \psi \right]} \right)

        v_{\text{creep}} = h z \mu f(\phi)

        \tan \psi = 2 \alpha \left( \phi_c - \phi \right)

        f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}


    Parameters
    ----------
    variables : Variables
        The friction coefficient ``mu`` and gouge porosity ``phi``
    params : CNSParams
        The CNS parameters ``alpha``, ``phi_c``, ``z``, ``mu0``, ``v0`` and ``a``
    constants : CNSConstants
        The constant parameters ``h`` and ``phi0``

    Returns
    -------
    v : Float
        The instantaneous slip rate in the same units as ``v0``
    """

    # Granular flow components
    tan_psi = 2 * params.alpha * (params.phi_c - variables.phi)
    A = variables.mu * (1 - params.mu0 * tan_psi) - params.mu0 - tan_psi
    B = params.a * (1 + variables.mu * tan_psi)
    v_gr = params.v0 * jnp.exp(A / B)

    # Creep components
    f_phi = _porosity_func(variables, params, constants)
    v_creep = constants.h * params.z * variables.mu * f_phi

    # Assembly
    v = v_gr + v_creep

    return jnp.squeeze(v)


@eqx.filter_jit
def cns_porosity(
    v: Float, variables: Variables, params: CNSParams, constants: CNSConstants
) -> Float:
    r"""
    The porosity (state) evolution for the Chen-Niemeijer-Spiers model

    .. math::

        \frac{\mathrm{d}\phi}{\mathrm{d}t} = - \left(1 - \phi \right) \left(\dot{\varepsilon}_{\text{gr}} + \dot{\varepsilon}_{\text{creep}} \right)

        \dot{\varepsilon}_{\text{gr}} = - \frac{\tan \psi}{h} \left(v - v_{\text{creep}} \right)

        \dot{\varepsilon}_{\text{creep}} = z f(\phi)

        v_{\text{creep}} = h z f(\phi) \mu

        \tan \psi = 2 \alpha \left( \phi_c - \phi \right)

        f(\phi) = \frac{\phi - \phi_0}{\phi_c - \phi}


    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s].
    variables : Variables
        The friction coefficient ``mu`` and gouge porosity ``phi``
    params : CNSParams
        The CNS parameters ``alpha``, ``phi_c``, ``z``, ``mu0``, ``v0`` and ``a``
    constants : CNSConstants
        The constant parameters ``h`` and ``phi0``

    Returns
    -------
    dphi : Float
        The rate of change of the porosity [1/s]
    """

    # Creep components
    f_phi = _porosity_func(variables, params, constants)
    e_creep = params.z * f_phi
    v_creep = constants.h * e_creep * variables.mu

    # Granular flow components
    tan_psi = 2 * params.alpha * (params.phi_c - variables.phi)
    e_gr = -tan_psi * (v - v_creep) / constants.h

    # Assembly
    dphi = -(1 - variables.phi) * (e_gr + e_creep)

    return jnp.squeeze(dphi)


@eqx.filter_jit
def slip_rate(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    r"""
    Evolve slip from slip rate

    .. math::

        \frac{\mathrm{d} x}{\mathrm{d} t} = v

    Parameters
    ----------
    v : Float
        Instantaneous fault slip rate [m/s]
    variables : Variables
        The friction coefficient ``mu`` and state parameter ``theta`` (not used)
    params : RSFParams
        The rate-and-state parameters (not used)
    constants : RSFConstants
        The constant parameters ``mu0`` and ``v0`` (not used)

    Returns
    -------
    v : Float
        The rate of change of slip [m/s]
    """
    return v


@eqx.filter_jit
def _identity(*args, **kwargs) -> Float:
    return 1


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
