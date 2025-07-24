from dataclasses import dataclass
from typing import Any, Protocol, Union

import jax.numpy as jnp
from jax import tree_util
from jaxtyping import Float


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Variables:
    mu: Float
    state: Float

    def tree_flatten(self):
        return (self.mu, self.state), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_array(self):
        return jnp.array([self.mu, self.state])


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RSFParams:
    a: Float
    b: Float
    Dc: Float

    def tree_flatten(self):
        return (self.a, self.b, self.Dc), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_array(self):
        return jnp.array([self.a, self.b, self.Dc])


@dataclass(frozen=True)
class RSFConstants:
    v0: Float
    mu0: Float


@dataclass(frozen=True)
class CNSParams:
    phi_c: Float
    Z: Float


@dataclass(frozen=True)
class CNSConstants:
    phi_c: Float
    Z: Float


@dataclass(frozen=True)
class SpringBlockConstants:
    k: Float
    v_lp: Float


class FrictionModel(Protocol):
    def __call__(self, variables: Variables, params: Any, constants: Any) -> Float: ...


class StateEvolution(Protocol):
    def __call__(
        self, v: Float, variables: Variables, params: Any, constants: Any
    ) -> Float: ...


# These typedefs should only be used for type checking,
# and should not be instantiated.
_Params = Union[RSFParams, CNSParams]
_Constants = Union[RSFConstants]
_BlockConstants = Union[SpringBlockConstants]
