from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    NamedTuple,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
from jaxtyping import Array, Float

"""
--------------------
Parameter containers
--------------------
"""


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


# These typedefs should only be used for type checking,
# and should not be instantiated.
_Params = Union[RSFParams, CNSParams]
_Constants = Union[RSFConstants]
_BlockConstants = Union[SpringBlockConstants]


"""
-------------------
Function signatures
-------------------
"""


class FrictionModel(Protocol):
    def __call__(self, variables: Variables, params: Any, constants: Any) -> Float: ...


class StateEvolution(Protocol):
    def __call__(
        self, v: Float, variables: Variables, params: Any, constants: Any
    ) -> Float: ...


"""
--------------
SVI containers
--------------
"""


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Particles:

    _particles: Union[Tuple[RSFParams, ...], Tuple[CNSParams, ...]]

    def __getitem__(self, idx):
        # Check if the requested selection is of the kind `x[1, 3]`
        if isinstance(idx, tuple):
            assert (
                len(idx) == 2
            ), "Particles do not support item selection with more than 2 indices"
            i, j = idx
            return self._particles[i].to_array()[j]
        # Else the selection is of the kind `x[4]`
        return self._particles[idx]

    def __iter__(self):
        return iter(self._particles)

    def __len__(self):
        return len(self._particles)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        raise NotImplementedError

    @classmethod
    def from_array(cls, x: Iterable):
        return cls.tree_unflatten(None, x)

    def tree_flatten(self):
        return self._particles, None

    def to_array(self):
        return jnp.array([p.tree_flatten()[0] for p in self._particles])

    @classmethod
    def generate(
        cls, N: int, loc: Float[Array, "M"], scale: Float[Array, "M"], key: jax.Array
    ):
        assert len(loc) == len(scale)
        x = jr.normal(key, shape=(N, len(loc))) * scale[None] + loc[None]
        return cls.from_array(x)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RSFParticles(Particles):

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x = tuple(RSFParams(*child) for child in children)
        return cls(x)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Chains(Container):

    _chains: Sequence[Particles]

    def __getitem__(self, idx):

        # Check if the requested selection is of the kind `x[1, 3]` or `x[1, 3, 2]`
        if isinstance(idx, tuple):
            assert (
                1 < len(idx) <= 3
            ), "Chains do not support item selection with more than 3 indices"

            if len(idx) == 2:
                i, j = idx
                return self._chains[i][j]

            i, j, k = idx
            return self._chains[i][j].to_array(k)

        # Else the selection is of the kind `x[4]`
        return self._chains[idx]

    def __iter__(self):
        return iter(self._chains)

    def __len__(self):
        return len(self._chains)

    def tree_flatten(self):
        return self._chains, None

    def to_array(self):
        return jnp.array([c.to_array() for c in self._chains])


# Alias gradients for semantic clarity
Gradients: TypeAlias = Particles


class Statistics(NamedTuple):
    mean: Float
    median: Float
    std: Float
    q5: Float
    q95: Float


class RSFStatistics(NamedTuple):
    a: Statistics
    b: Statistics
    Dc: Statistics


class CNSStatistics(NamedTuple):
    a: Statistics
    b: Statistics
    Dc: Statistics


@dataclass
class BayesianSolution:

    chains: Chains
    statistics: Union[RSFStatistics, CNSStatistics]

    def __init__(self, chains: Chains):
        self.chains = chains
        final_state = chains[-1]
        # TODO: final_state contains N particles
        #
