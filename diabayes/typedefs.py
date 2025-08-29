import dataclasses as dcs
from time import time_ns
from typing import Any, NamedTuple, Protocol, Tuple, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jaxtyping import Array, Float

"""
--------------------
Parameter containers
--------------------
"""


class StateDict(eqx.Module):
    """
    A container class for state variables. A user would typically not
    interact with this class directly (only through ``Variables`` and
    ``Forward.set_initial_values``).

    Examples
    --------
    >>> from diabayes.typedefs import StateDict
    >>> import jax.numpy as jnp
    >>> keys = ("x", "y")
    >>> vals = jnp.array([1.0, -5.1])
    >>> state_dict = StateDict(keys=keys, vals=vals)
    """

    keys: tuple[str, ...] = eqx.field(static=True)
    """A tuple of key names (strings) corresponding with the state variable names"""
    vals: jax.Array
    """An array of values corresponding (in order) with the variables defined by ``keys``"""

    def __getitem__(self, k: str) -> Array:
        """Get the value of the variable ``k``"""
        i = self.keys.index(k)
        return self.vals[i]

    def replace_values(self, **kwargs) -> "StateDict":
        """
        Update the values of the state variables. Since immutability is
        required for JIT caching, this method will return a copy of the
        class with the updated values.

        Examples
        --------
        >>> from diabayes.typedefs import StateDict
        >>> import jax.numpy as jnp
        >>> state_dict = StateDict(keys=("x",), vals=jnp.asarray(1.0))
        >>> state_dict = state_dict.replace_values({"x": jnp.asarray(2.0)})

        Parameters
        ----------
        **kwargs : dict
            Key-value pairs to update. These can overwrite existing values
            or create entirely new ones.

        Returns
        -------
        state_dict : StateDict
            A copy of the current StateDict with updated values
        """
        mapping = dict(zip(self.keys, self.vals))
        mapping.update(kwargs)
        return StateDict(self.keys, jnp.array([mapping[k] for k in self.keys]))


class Variables(eqx.Module):
    """
    The main container class for variables (friction and state). It
    contains additional convenience routines for mapping between
    this class object and JAX arrays.

    Users would typically not instantiate a ``Variables`` object
    directly; instead, it is created through the ``Forward.set_initial_values``
    method and retreived as ``Forward.variables``.
    """

    mu: jax.Array
    """The friction coefficient"""
    state: StateDict
    """A container with the various state variables"""

    def __getattr__(self, name: str):
        # NOTE: __getattr__ is only called when __getattribute__,
        # lookup fails, i.e, it looks for self.name before
        # checking self.state.name
        if name in self.state.keys:
            return self.state[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def set_values(self, **kwargs) -> "Variables":
        """
        Set the values of friction and the state variables through key-value
        pair arguments. The state parameters are provided individually as
        key-value pairs as well. Because this class is immutable, ``set_values``
        returns a copy of the class with updated values.

        Examples
        --------
        >>> state_dict = StateDict(keys=("x", "y"), vals=jnp.array([0.1, 0.2]))
        >>> variables = Variables(mu=jnp.asarray(0.6), state=state_dict)
        >>> variables = variables.set_values(mu=..., x=..., y=...)
        """
        mu = jnp.atleast_1d(kwargs.pop("mu"))
        return dcs.replace(self, mu=mu, state=self.state.replace_values(**kwargs))

    def __repr__(self) -> str:
        state_str = ", ".join(
            f"{k}={v}" for k, v in zip(self.state.keys, self.state.vals)
        )
        return f"Variables(mu={self.mu}, {state_str})"

    def to_array(self) -> Float[Array, "..."]:
        """
        Convert the container values to a JAX array. The order of the output
        follows the order of `StateDict.keys`, with the first item being
        the friction coefficient. For ``n`` state variables, the output is
        an array of shape ``(1+n,)`` for scalars, and ``(t, 1+n)`` for
        time series.

        Examples
        --------
        >>> scalars = Variables(...)
        >>> mu = scalars.to_array()[0]  # friction is first element
        >>> timeseries = Variables(...)
        >>> mu = timeseries.to_array()[:, 0]
        >>> all_final_values = timeseries.to_array()[-1, :]

        Of course, in this example one could simply do ``scalars.mu``
        and ``timeseries.mu`` to extract ``mu`` directly.
        """
        mu = jnp.asarray(jnp.squeeze(self.mu))
        state = jnp.asarray(jnp.squeeze(self.state.vals))

        # First case: mu and state are scalars
        # Result shape (2,)
        if mu.ndim == state.ndim == 0:
            return jnp.hstack([mu, state])
        # Second case: mu is scalar, state is vector
        # (i.e., multiple state variables)
        # Result shape (1+n,)
        elif mu.ndim == 0 and state.ndim == 1:
            return jnp.hstack([mu, *state])
        # Third case: mu and state are time series of scalars
        # Result shape: (2, t)
        elif mu.ndim == state.ndim == 1:
            return jnp.vstack([mu[None, :], state[None, :]])
        # Fourth case: mu is time series of scalars,
        # state is time series of vectors (t, n)
        # Result shape: (1+n, t)
        elif mu.ndim == 1 and state.ndim == 2:
            return jnp.vstack([mu[None, :], state])
        # Other combinations of shapes should not exist
        else:
            raise ValueError(f"Unsupported shapes: mu={mu.shape}, state={state.shape}")

    @classmethod
    def from_array(cls, x: Float[Array, "..."], keys: tuple[str, ...]) -> "Variables":
        """
        Import a JAX array to instantiate the class. The array `x` can either be an
        array of scalars (size n for n variables), or an array of time series of
        shape `(n, t)`. The first element in the array (i.e., `x[0]`) is assumed to
        be the friction coefficient `mu`. The remaining elements are the state
        variables, matching the number and order of the `keys` tuple.
        """
        # The first element is assumed to be mu
        mu = jnp.atleast_1d(x[0])
        # If x is 2D: time series
        if x.ndim == 2:
            state = jnp.atleast_2d(x[1:])
        # Else: scalars
        else:
            state = jnp.atleast_1d(x[1:])
        # Map `state` to `keys`
        state_obj = StateDict(keys=keys, vals=state)
        return cls(mu=mu, state=state_obj)


class Params(eqx.Module):

    def __init__(self, *args, **kwargs):
        field_names = [f.name for f in dcs.fields(self)]
        for key, val in zip(field_names, args):
            kwargs[key] = val

        for key, val in kwargs.items():
            object.__setattr__(self, key, jnp.asarray(val))

        for name in field_names:
            if not hasattr(self, name):
                object.__setattr__(self, name, jnp.asarray(-1.0))

    def __getitem__(self, idx):
        x = dict((key.name, getattr(self, key.name)[idx]) for key in dcs.fields(self))
        return type(self)(**x)

    def __len__(self):
        key = dcs.fields(self)[0].name
        return len(getattr(self, key))

    def to_array(self):
        return jnp.squeeze(jnp.array(jax.tree_util.tree_flatten(self)[0]))

    @classmethod
    def from_array(cls, x):
        return cls(*x)

    @classmethod
    def generate(
        cls, N: int, loc: Float[Array, "M"], scale: Float[Array, "M"], key: jax.Array
    ):
        assert len(loc) == len(scale)
        x = jr.normal(key, shape=(N, len(loc))) * scale + loc
        return cls.from_array(x.T)


class RSFParams(Params):
    a: Union[Float, Float[Array, "..."]]
    b: Union[Float, Float[Array, "..."]]
    Dc: Union[Float, Float[Array, "..."]]


@dcs.dataclass(frozen=True)
class RSFConstants:
    v0: Float
    mu0: Float


class CNSParams(Params):
    phi_c: Float
    Z: Float


@dcs.dataclass(frozen=True)
class CNSConstants:
    phi_c: Float
    Z: Float


@dcs.dataclass(frozen=True)
class SpringBlockConstants:
    k: Float
    v_lp: Float


@dcs.dataclass(frozen=True)
class InertialSpringBlockConstants:
    k: Float
    v_lp: Float
    M: Float


# These typedefs should only be used for type checking,
# and should not be instantiated.
_Params = Union[RSFParams, CNSParams]
_Constants = Union[RSFConstants]
_BlockConstants = Union[SpringBlockConstants, InertialSpringBlockConstants]
BC = TypeVar("BC", bound=_BlockConstants, contravariant=True)

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


class StressTransfer(Protocol[BC]):
    def __call__(
        self,
        t: Float,
        v: Float,
        v_partials: Variables,
        variables: Variables,
        dstate: Float[Array, "..."],
        constants: BC,
    ) -> Float: ...


"""
-----------------
Inversion results
-----------------
"""


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


@dcs.dataclass
class BayesianSolution:
    """
    The result of a Bayesian inversion. This data class stores the
    particle trajectories (equivalent of MCMC chains), the final
    equilibrum state, and various statistcs, as well as helper
    routines to diagnose or visualise the results.

    Examples
    --------
    >>> result = solver.bayesian_inversion(...)
    >>> print(result.nan_count)  # Check for NaN values
    >>> print(result.statistics.Dc.median)  # Get the median of Dc
    >>> result.plot_convergence()  # Visually inspect convergence
    >>> result.cornerplot(nbins=15)  # Create a cornerplot
    """

    log_params: _Params
    """
    The particles representing the (log-valued) parameters,
    including all the iteration steps
    """
    chains: Float[Array, "Nsteps Nparticles Nparams"]
    """The particle trajectories (to inspect convergence)"""
    log_likelihood: Float[Array, "Nsteps"]
    """The log-likelihood evolution"""
    nan_count: Float[Array, "Nsteps"]
    """The number of NaNs encountered during each iteration step"""
    final_state: Float[Array, "Nparticles Nparams"]
    """The final state of the particle swarm"""
    statistics: Union[RSFStatistics, CNSStatistics, None] = None
    """Pre-computed statistics of the parameters"""

    def __init__(
        self,
        log_params: _Params,
        log_likelihood: Float[Array, "Nsteps"],
        nan_count: Float[Array, "Nsteps"],
    ):
        self.log_params = log_params
        # Convert the inversion result to an array and transpose
        chains = log_params.to_array().transpose(1, 2, 0)
        # Undo the log-transform and store chains
        self.chains = jnp.exp(chains)
        # Undo the log-transform and store final state
        self.final_state = jnp.exp(chains[-1])
        # Store the log-likelihood
        self.log_likelihood = log_likelihood
        # Store the number of NaNs encountered during each step
        self.nan_count = nan_count
        # TODO: need to generalise this...
        self.statistics = RSFStatistics.from_state(self.final_state)

    def plot_convergence(self):
        """
        Helper routine to visualise the convergence of the inversion.
        Convergence is achieved when the particles settle in an
        equilibrium position (i.e., they stop moving).

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The figure object that can be manipulated after creation
            (e.g. to save to disk)
        """

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
        """
        Create a corner plot from the particle positions. For `N`
        parameters, a corner plot is the lower half of an `N x N`
        grid of subplots. On the diagonal, the histogram of the
        marginal posterior over the `i-th` parameter is plotted.
        Below the diagonal, the `(i,j)`-th panel is a scatter
        plot of parameter `i` against parameter `j`, which shows
        the co-variance between the two quantities.

        Parameters
        ----------
        nbins : int
            The number of bins to use for the histograms on the
            diagonal of the corner plot

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The figure object that can be manipulated after creation
            (e.g. to save to disk)
        """

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
        """Print out nicely-formatted statistics"""

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
        """
        Draw random samples from the posterior distribution. Each
        particle should represent one realisation of a plausible
        friction curve. For ``nsamples`` realisations requested by a
        user, it is more beneficial to `vmap` the forward simulations
        rather than compute them one by one.

        The resulting friction curves can be used for validation
        purposes (does the posterior match the data?).

        Examples
        --------
        >>> result = solver.bayesian_inversion(...)
        >>> samples, sample_results = result.sample(
        ...    solver=solver, y0=y0, t=t,
        ...    friction_constants=constants,
        ...    block_constants=block_constants,
        ...    nsamples=20, rng=42)
        ... )

        Parameters
        ----------
        solver : ODESolver
            The ``ODESolver`` class instantiated with the ``Forward``
            model (typically the same one as used for the inversion)
        y0 : Variables
            The initial values of friction and state
        t : Array
            The time samples at which a solution is requested. This
            does not need to be the same as the time samples recorded
            in the experiment (from which the observed friction curve
            is obtained)
        friction_constants : _Constants
            The friction and state model constants
        block_constants : _BlockConstants
            The stress transfer constants
        nsamples : int
            The number of random samples for which a solution should
            be computed
        rng: None, int, jax.random.PRNGKey
            The random seed used to sample from the particle distribution.
            If ``None``, the current time will be used as a seed, which leads
            to different result for each realisation. When an integer is
            provided, a new ``jax.random.PRNGKey`` is generated.

        Returns
        -------
        samples : _Params
            The samples drawn from the posterior distribution
        sample_results : Variables
            The forward simulation results corresponding with ``samples``
        """

        if rng is None:
            key = jr.PRNGKey(time_ns())
        elif isinstance(rng, int):
            key = jr.PRNGKey(rng)
        elif isinstance(rng, jax.Array):
            key = rng

        key, split_key = jr.split(key)
        inds = jr.choice(split_key, jnp.arange(self.chains.shape[1]), shape=(nsamples,))
        log_samples = self.log_params[-1][inds]
        samples = type(log_samples)(*jnp.exp(log_samples.to_array()))

        sample_results = jax.vmap(
            solver._forward_wrapper_SVI, in_axes=(0, None, None, None, None), out_axes=0
        )(samples, y0, t, friction_constants, block_constants)
        return samples, sample_results
