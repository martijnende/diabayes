from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from diabayes.typedefs import (
    BlockConstants,
    Constants,
    Params,
    RSFConstants,
    RSFParams,
    SpringBlockConstants,
    Variables,
)


@eqx.filter_jit
def rsf(variables: Variables, params: RSFParams, constants: RSFConstants) -> Float:
    Omega = params.b * jnp.log(variables.state * constants.v0 / params.Dc)
    return constants.v0 * jnp.exp((variables.mu - constants.mu0 - Omega) / params.a)


@eqx.filter_jit
def ageing_law(
    variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    return 1 - variables.v * variables.state / params.Dc


@eqx.filter_jit
def springblock(variables: Variables, constants: SpringBlockConstants) -> Float:
    return constants.k * (constants.v_lp - variables.v)


class Forward:

    def __init__(
        self,
        friction_model: Callable[[Variables, Params, Constants], Float],
        state_evolution: Callable[[Variables, Params, Constants], Float],
        block_type: Callable[[Variables, BlockConstants], Float],
    ) -> None:
        self.friction_model = friction_model
        self.state_evolution = state_evolution
        self.stress_transfer = block_type
        pass

    @eqx.filter_jit
    def __call__(
        self,
        variables: Variables,
        params: Params,
        friction_constants: Constants,
        block_constants: BlockConstants,
    ) -> Variables:
        variables.v = self.friction_model(variables, params, friction_constants)
        dstate = self.state_evolution(variables, params, friction_constants)
        dmu = self.stress_transfer(variables, block_constants)
        return Variables(mu=dmu, state=dstate)
