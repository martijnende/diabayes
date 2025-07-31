from dataclasses import dataclass
from time import time_ns
from typing import Any, Iterable, NamedTuple, Protocol, Tuple, TypeAlias, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import tree_util
from jaxtyping import Array, Float

"""
--------------------
Parameter containers
--------------------
"""


class Container:

    def __len__(self):
        return len(self.tree_flatten()[0])

    def __iter__(self):
        return iter(self.tree_flatten()[0])

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


# Alias gradients for semantic clarity
Gradients: TypeAlias = Particles

"""
-----------------
Inversion results
-----------------
"""

Chains = Float[Array, "Nsteps Nparticles Nparams"]


class Statistics(NamedTuple):
    mean: Float
    median: Float
    std: Float
    q5: Float
    q95: Float

    @classmethod
    def from_array(cls, x: Float[Array, "..."]):
        mean = jnp.mean(x)
        std = jnp.std(x)
        q5, median, q95 = jnp.quantile(x, jnp.array([0.05, 0.5, 0.95]))
        return cls(mean=mean, median=median, std=std, q5=q5, q95=q95)

    def get(self, x: str) -> Float:
        return getattr(self, x)


class ParamStatistics(eqx.Module):

    @classmethod
    def from_state(cls, state):
        cov = jnp.cov(state.T)
        stats = tuple(Statistics.from_array(param) for param in state.T)
        return cls(*(stats + (cov,)))  # type:ignore

    def get(self, x: str) -> Statistics:
        return getattr(self, x)

    def get_param_names(self) -> Tuple:
        return tuple(self.__dataclass_fields__.keys())[:-1]


class RSFStatistics(ParamStatistics):

    a: Statistics
    b: Statistics
    Dc: Statistics
    cov: Float[Array, "N N"]


class CNSStatistics(ParamStatistics):
    a: Statistics
    b: Statistics
    Dc: Statistics


@dataclass
class BayesianSolution:

    chains: Chains
    log_likelihood: Float[Array, "Nsteps"]
    nan_count: Float[Array, "Nsteps"]
    statistics: Union[RSFStatistics, CNSStatistics, None] = None

    def __init__(
        self,
        chains: Chains,
        log_likelihood: Float[Array, "Nsteps"],
        nan_count: Float[Array, "Nsteps"],
    ):
        self.chains = jnp.exp(chains)
        self.final_state = jnp.exp(chains[-1])
        self.log_likelihood = log_likelihood
        self.nan_count = nan_count
        self.statistics = RSFStatistics.from_state(self.final_state)

    def plot_convergence(self):

        assert self.statistics is not None

        params = self.statistics.get_param_names()
        nparams = len(params)

        plt.close("all")
        fig, axes = plt.subplots(
            nrows=nparams + 1,
            figsize=(6, (nparams + 1) * 3),
            constrained_layout=True,
            sharex="all",
        )

        ax = axes[0]
        ax.plot(self.log_likelihood)
        ax.set_ylabel("log-likelihood")

        for i, (ax, chains, param) in enumerate(
            zip(axes[1:], self.chains.transpose((2, 0, 1)), params)
        ):
            ax.plot(chains, c="k", alpha=0.1, lw=1)
            ax.set_ylabel(f"Parameter {param}")

        ax.set_xlabel("step")

        return fig

    def cornerplot(self, nbins=20):

        assert self.statistics is not None
        params = self.statistics.get_param_names()
        nparams = len(params)
        state = self.final_state.T

        plt.close("all")
        fig, axes = plt.subplots(
            nrows=nparams,
            ncols=nparams,
            figsize=(8, 5),
            constrained_layout=True,
            sharex="col",
        )

        for i, (axrow, param1) in enumerate(zip(axes, params)):
            axrow[0].set_ylabel(param1)
            for j, (ax, param2) in enumerate(zip(axrow, params)):
                if j > i:
                    fig.delaxes(ax)
                    continue

                stat1 = self.statistics.get(param1)
                stat2 = self.statistics.get(param2)
                ax.axvline(stat2.median, ls=":", c="C1")

                if i == j:
                    ax.hist(state[i], bins=nbins, alpha=0.7)
                else:
                    ax.scatter(state[j], state[i], c="k", alpha=0.1, s=10)

                    xerr = [[stat2.median - stat2.q5], [stat2.q95 - stat2.median]]
                    yerr = [[stat1.median - stat1.q5], [stat1.q95 - stat1.median]]
                    ax.errorbar(
                        stat2.median,
                        stat1.median,
                        xerr=xerr,
                        yerr=yerr,
                        fmt="o",
                        c="C1",
                        capsize=5,
                    )

                if i == nparams - 1:
                    ax.set_xlabel(param2)

        return fig

    def __repr__(self) -> str:

        assert self.statistics is not None

        params = self.statistics.get_param_names()
        stats = self.statistics.get(params[0])._fields
        nl = "\n"
        tab = "\t"
        gutter = 2 * tab
        header = gutter.join(params)
        s = "Bayesian inversion results" + nl
        s += "-" * 60 + nl * 2
        s += f"Number of particles:\t{self.chains.shape[1]}" + nl
        s += f"Number of iterations:\t{self.chains.shape[0]}" + nl
        s += f"Number of parameters:\t{self.chains.shape[2]} {params}" + nl * 2
        s += gutter + header + nl
        for stat in stats:
            param_stats = tab.join(
                f"{self.statistics.get(param).get(stat):.2e}" for param in params
            )
            s += stat + 2 * tab + param_stats + nl
        s += nl + "-" * 60
        return s

    def sample(
        self,
        solver,
        y0: Variables,
        t: Float[Array, "Nt"],
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        nsamples: int = 10,
        rng: Union[None, int, jax.Array] = None,
    ):

        if rng is None:
            key = jr.PRNGKey(time_ns())
        elif isinstance(rng, int):
            key = jr.PRNGKey(rng)
        elif isinstance(rng, jax.Array):
            key = rng

        key, split_key = jr.split(key)
        samples = jr.choice(split_key, self.final_state, shape=(nsamples,), axis=0)

        sample_results = jax.vmap(
            solver._forward_wrapper_SVI, in_axes=(None, 0, None, None, None), out_axes=0
        )(y0.to_array(), samples, t, friction_constants, block_constants)
        return samples, sample_results
