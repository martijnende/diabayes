from functools import partial
from time import time_ns
from typing import Any, Tuple, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import optimistix as optx
from jax import lax
from jax_tqdm import scan_tqdm  # type:ignore
from jaxtyping import Array, Float
from scipy.integrate import solve_ivp

from diabayes.forward_models import Forward
from diabayes.SVI import compute_phi, mapped_log_likelihood
from diabayes.typedefs import (
    BayesianSolution,
    RSFParams,
    RSFParticles,
    Variables,
    _BlockConstants,
    _Constants,
    _Params,
)

jax.config.update("jax_enable_x64", True)


class ODESolver:

    forward_model: Forward
    rtol: float
    atol: float
    checkpoints: int
    learning_rate: float = 1e-2

    def __init__(
        self,
        forward_model: Forward,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        checkpoints: int = 100,
    ) -> None:
        self.forward_model = forward_model
        self.rtol = rtol
        self.atol = atol
        self.checkpoints = checkpoints
        pass

    def solve_forward(
        self,
        t: Float[Array, "Nt"],
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Any:
        """
        Solve a forward problem using SciPy's `solve_ivp` routine.
        While this routine doesn't propagate any gradients, it is
        much faster to initialise and to perform a single forward
        run. Hence for playing around with different parameters,
        it is preferred over a JITed JAX implementation.

        Parameters
        ----------
        t : Float[Array, "Nt"]
            A vector of time samples where a solution is requested
        y0 : Variables
            The initial values (fricton and state) wrapped in a
            `Variables` container.
        params : _Params
            The (invertible) parameters that govern the dynamics,
            wrapped in a `Params` container.
        friction_constants : _Constants
            A container object containing the friction constants
        block_constants : _BlockConstants
            A container object containing the block constants

        Returns
        -------
        result : Variables
            Solution time series of friction and state
        """

        _forward = lambda t, y, *args: self.forward_model(
            Variables.from_array(y), *args
        ).to_array()

        result = solve_ivp(
            fun=_forward,
            t_span=(t.min(), t.max()),
            y0=y0.to_array(),
            t_eval=t,
            args=(params, friction_constants, block_constants),
            rtol=self.rtol,
            atol=self.atol,
        )

        assert result.y is not None

        return Variables.from_array(result.y)

    @eqx.filter_jit
    def _forward_wrapper(
        self,
        t: Float,
        variables: Variables,
        args: Tuple[_Params, _Constants, _BlockConstants],
    ) -> Variables:
        params, friction_constants, block_constants = args
        return self.forward_model(
            variables=variables,
            params=params,
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

    @eqx.filter_jit
    def _forward_wrapper_SVI(
        self,
        y0: Float[Array, "2"],
        params: Float[Array, "..."],
        t: Float[Array, "Nt"],
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Variables:
        result = self._solve_forward(
            t=t,
            y0=Variables.from_array(y0),
            params=RSFParams.from_array(params),
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

        assert result is not None
        assert result.ys is not None

        return result.ys

    @eqx.filter_jit
    def _solve_forward(
        self,
        t: Float[Array, "Nt"],
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        adjoint: Union[None, dfx.AbstractAdjoint] = None,
    ) -> dfx.Solution:

        term = dfx.ODETerm(self._forward_wrapper)
        t0 = t.min()
        t1 = t.max()
        dt0 = t[1] - t[0]
        saveat = dfx.SaveAt(ts=t)
        args = (params, friction_constants, block_constants)

        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)
        if adjoint is None:
            adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=self.checkpoints)
        assert isinstance(adjoint, dfx.AbstractAdjoint)

        sol = dfx.diffeqsolve(
            terms=term,
            solver=dfx.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            saveat=saveat,
            y0=y0,
            args=args,
            stepsize_controller=controller,
            adjoint=adjoint,
        )

        assert sol is not None

        return sol

    @eqx.filter_jit
    def _residuals(
        self,
        params: _Params,
        t: Float[Array, "N"],
        mu: Float[Array, "N"],
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Float[Array, "N"]:
        adjoint = dfx.ForwardMode()
        # TODO: replace this with something like `get_steadystate()`
        # because this is different for e.g. CNS
        theta0 = params.Dc / friction_constants.v0  # type:ignore
        y0 = Variables(mu=mu[0], state=theta0)
        result = self._solve_forward(
            t, y0, params, friction_constants, block_constants, adjoint
        )
        mu_hat = result.ys.mu  # type:ignore
        return mu - mu_hat

    def max_likelihood_inversion(
        self,
        t: Float[Array, "Nt"],
        mu: Float[Array, "Nt"],
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        verbose: bool = False,
    ) -> optx.Solution:

        if verbose:
            verbose_opts = frozenset(["step", "loss"])
        else:
            verbose_opts = frozenset([])

        options = {"autodiff_mode": "fwd"}

        _residuals = lambda params, mu: self._residuals(
            params, t, mu, friction_constants, block_constants
        )

        lm_solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5, verbose=verbose_opts)
        sol = optx.least_squares(
            _residuals, lm_solver, params, args=mu, options=options
        )

        return sol

    def bayesian_inversion(
        self,
        t: Float[Array, "Nt"],
        mu: Float[Array, "Nt"],
        noise_std: Float,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        Nparticles: int = 1000,
        Nsteps: int = 150,
        rng: Union[None, int, jax.Array] = None,
    ):

        assert isinstance(
            params, RSFParams
        ), "Bayesian inversion is only implemented for RSF"

        if rng is None:
            key = jr.PRNGKey(time_ns())
        elif isinstance(rng, int):
            key = jr.PRNGKey(rng)
        else:
            key = rng

        key, split_key = jr.split(key)

        scale = jnp.ones(len(params)) * 0.1
        inv_scale = 1 / (jnp.sqrt(2) * scale)
        log_params = jnp.log(params.to_array())

        # Sample particles from a log-normal distribution
        log_particles = RSFParticles.generate(
            N=Nparticles, loc=log_params, scale=scale, key=split_key
        ).to_array()

        # Instantiate optimiser
        opt = optax.adam(learning_rate=self.learning_rate)
        opt_state = opt.init(log_particles)

        forward_fn = partial(
            self._forward_wrapper_SVI,
            t=t,
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

        @scan_tqdm(Nsteps)
        def body_fun(carry, i):
            params, state = carry
            loss, gradp = mapped_log_likelihood(
                params, mu, noise_std, friction_constants.v0, forward_fn
            )
            """
            Sometimes the adjoint back-propagation becomes unstable, 
            producing NaNs in the gradients. By setting the NaN-gradients
            to zero, the particle will be attracted towards the prior.
            This is fine, because in the next step the gradients will
            likely be stable again and the particle will continue
            to be attracted by the maximum likelihood
            """
            nan_count = jnp.isnan(gradp).sum() / gradp.shape[1]
            gradp = jnp.asarray(jnp.where(jnp.isnan(gradp), 0.0, gradp))
            gradq = -2 * (params - log_params) * inv_scale
            phi = compute_phi(params, gradp, gradq)
            updates, state = opt.update(phi, state, params)
            params = optax.apply_updates(params, updates)
            return (params, state), (loss.mean(), nan_count, params)

        carry = (log_particles, opt_state)
        _, (loss, nan_count, states) = lax.scan(
            body_fun, carry, jnp.arange(Nsteps)  # type:ignore
        )

        return BayesianSolution(states, loss, nan_count)
