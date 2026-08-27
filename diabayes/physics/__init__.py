import equinox as eqx
from jaxtyping import Float

from diabayes.typedefs import (
    _Constants,
    _Params,
    Variables,
)

from .rate_and_state import rsf, ageing_law, aging_law, slip_law
from .chen_niemeijer_spiers import cns, cns_porosity, sundman_cns, steady_state_porosity
from .stress_transfer import springblock, inertial_springblock


@eqx.filter_jit
def slip_rate(
    v: Float, variables: Variables, params: _Params, constants: _Constants
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
