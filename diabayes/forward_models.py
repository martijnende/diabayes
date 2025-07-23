from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

from diabayes.typedefs import (
    RSFConstants,
    RSFParams,
    SpringBlockConstants,
    Variables,
    _BlockConstants,
    _Constants,
    _FrictionModel,
    _Params,
    _StateEvolution,
)


@eqx.filter_jit
def rsf(variables: Variables, params: RSFParams, constants: RSFConstants) -> Float:
    Omega = params.b * jnp.log(variables.state * constants.v0 / params.Dc)
    return constants.v0 * jnp.exp((variables.mu - constants.mu0 - Omega) / params.a)


@eqx.filter_jit
def ageing_law(
    v: Float, variables: Variables, params: RSFParams, constants: RSFConstants
) -> Float:
    return 1 - v * variables.state / params.Dc


@eqx.filter_jit
def springblock(
    v: Float, variables: Variables, constants: SpringBlockConstants
) -> Float:
    return constants.k * (constants.v_lp - v)


class Forward:

    def __init__(
        self,
        friction_model: _FrictionModel,
        state_evolution: _StateEvolution,
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
