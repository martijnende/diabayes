from dataclasses import dataclass
from typing import Union

from jax import tree_util
from jaxtyping import Float


@tree_util.register_pytree_node_class
@dataclass(frozen=False)
class Variables:
    mu: Float
    state: Float
    v: Float = 0.0

    def tree_flatten(self):
        return (self.mu, self.state), self.v

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mu=children[0], state=children[1], v=aux_data)


# @tree_util.register_pytree_node_class
# @dataclass(frozen=False)
# class RSFParams:
#     a: Float
#     b: Float
#     Dc: Float

#     def tree_flatten(self):
#         return (self.a, self.b, self.Dc), None

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)


@dataclass(frozen=True)
class RSFParams:
    a: Float
    b: Float
    Dc: Float


@dataclass(frozen=True)
class CNSParams:
    phi_c: Float
    Z: Float


@dataclass(frozen=True)
class RSFConstants:
    v0: Float
    mu0: Float


@dataclass(frozen=True)
class SpringBlockConstants:
    k: Float
    v_lp: Float


Params = Union[RSFParams, CNSParams]
Constants = Union[RSFConstants]
BlockConstants = Union[SpringBlockConstants]
