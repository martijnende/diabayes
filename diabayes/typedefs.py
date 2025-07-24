from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Union

import jax.numpy as jnp
from jax import tree_util
from jaxtyping import Float


class Container:

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @classmethod
    def from_array(cls, x: Iterable):
        return cls.tree_unflatten(None, x)

    def tree_flatten(self):
        raise NotImplementedError

    def to_array(self):
        return jnp.array(self.tree_flatten()[0])


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Variables(Container):
    mu: Float
    state: Float

    def tree_flatten(self):
        return (self.mu, self.state), None


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RSFParams(Container):
    a: Float
    b: Float
    Dc: Float

    def tree_flatten(self):
        return (self.a, self.b, self.Dc), None


@dataclass(frozen=True)
class RSFConstants:
    v0: Float
    mu0: Float


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CNSParams(Container):
    phi_c: Float
    Z: Float

    def tree_flatten(self):
        return (self.phi_c, self.Z), None


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
