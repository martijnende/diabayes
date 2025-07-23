from dataclasses import dataclass
from typing import Callable, Union

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


@dataclass(frozen=True)
class RSFParams:
    a: Float
    b: Float
    Dc: Float


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


# These typedefs should only be used for type checking,
# and should not be instantiated.
_Params = Union[RSFParams, CNSParams]
_Constants = Union[RSFConstants]
_BlockConstants = Union[SpringBlockConstants]

_FrictionModel = Union[
    Callable[[Variables, RSFParams, RSFConstants], Float],
    Callable[[Variables, CNSParams, CNSConstants], Float],
]
_StateEvolution = Union[
    Callable[[Float, Variables, RSFParams, RSFConstants], Float],
    Callable[[Float, Variables, CNSParams, CNSConstants], Float],
]
