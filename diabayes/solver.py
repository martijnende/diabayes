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
    Variables,
    _BlockConstants,
    _Constants,
    _Params,
)

jax.config.update("jax_enable_x64", True)


class ODESolver:
    """
    The main solver class that contains forward and inverse modelling
    methods.

    Attributes
    ----------
    forward_model : Forward
        An instantiated ``diabayes.Forward`` class, including
        friction law, state evolution equations, and stress
        transfer equation
    rtol, atol : float
        The relative and absolute tolerances used by the ODE solver
    checkpoints : int
        The number of checkpoints to use to compute the (adjoint)
        gradients through the ODE routine. A higher number increases
        stability and speed, at the expense of more GPU memory
    learning_rate : float
        The initial learning rate provided to the Adam algorithm
        for the Stein Variational Inference. The default value of
        ``1e-2`` seems like a sensible choice for most models.
    """

    forward_model: Forward
    rtol: float
    atol: float
    checkpoints: int
    learning_rate: float = 1e-2

    def __init__(
        self,
        forward_model: Forward,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        checkpoints: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        forward_model : Forward
            An instantiated ``diabayes.Forward`` class, including
            friction law, state evolution equations, and stress
            transfer equation
        rtol, atol : float
            The relative and absolute tolerances used by the ODE solver
        checkpoints : int
            The number of checkpoints to use to compute the (adjoint)
            gradients through the ODE routine. A higher number increases
            stability and speed, at the expense of more GPU memory
        """
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
        method: str = "RK45",
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

        keys = y0.state.keys

        _forward = lambda t, y, *args: self.forward_model(
            t, Variables.from_array(y, keys), *args
        ).to_array()

        result = solve_ivp(
            fun=_forward,
            t_span=(t.min(), t.max()),
            y0=y0.to_array(),
            t_eval=t,
            args=(params, friction_constants, block_constants),
            rtol=self.rtol,
            atol=self.atol,
            method=method,
        )

        assert result.y is not None

        return Variables.from_array(result.y, keys)

    @eqx.filter_jit
    def _forward_wrapper(
        self,
        t: Float,
        variables: Variables,
        args: Tuple[_Params, _Constants, _BlockConstants],
    ) -> Variables:
        params, friction_constants, block_constants = args
        return self.forward_model(
            t=t,
            variables=variables,
            params=params,
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

    @eqx.filter_jit
    def _forward_wrapper_SVI(
        self,
        params: _Params,
        y0: Variables,
        t: Float[Array, "Nt"],
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Variables:
        result = self._solve_forward(
            t=t,
            y0=y0,
            params=params,
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
        y0: Variables,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
    ) -> Float[Array, "N"]:
        adjoint = dfx.ForwardMode()
        result = self._solve_forward(
            t, y0, params, friction_constants, block_constants, adjoint
        )
        mu_hat = jnp.squeeze(result.ys.mu)  # type:ignore
        return mu - mu_hat

    def max_likelihood_inversion(
        self,
        t: Float[Array, "Nt"],
        mu: Float[Array, "Nt"],
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        verbose: bool = False,
    ) -> optx.Solution:
        r"""
        Minimises the least-squares residuals between the observed friction
        curve and the parameterised one, using the Levenberg-Marquardt
        algorithm.

        Parameters
        ----------
        t : Array
            A vector or time values (in units of seconds). The time steps
            do not need to be uniform
        mu : Array
            The observed friction curve sampled at ``t``
        y0 : Variables
            The initial values for the modelled friction and any state
            variables
        params : _Params
            The initial guess for the invertible parameters that characterise
            the forward problem. These need to be sufficiently close to the
            "true" values for the algorithm to converge
        friction_constants : _Constants
            The non-invertible constants that characterise the forward
            problem
        block_constants : _BlockConstants
            The stress transfer constants (e.g. stiffness and loading rate)
        verbose : bool
            Whether or not to output detailed progress of the inversion.
            Defaults to ``False``

        Returns
        -------
        sol : optimistix.Solution
            The inversion result, including various diagnostics. The
            inverted parameter values can be accessed as ``sol.values``

        Notes
        -----
        If an error is produced in the first iteration step, it is quite
        possible that the initial guess parameters were too far off from
        the "true" values (i.e., the mismatch between the observed and
        modelled friction curves is too large), breaking the Gauss-Newton
        step of the Levenberg-Marquardt algorithm. Initial manual tuning
        is recommended.
        """

        if verbose:
            verbose_opts = frozenset(["step", "loss"])
        else:
            verbose_opts = frozenset([])

        options = {"autodiff_mode": "fwd"}

        _residuals = lambda params, mu: self._residuals(
            params, t, mu, y0, friction_constants, block_constants
        )

        lm_solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5, verbose=verbose_opts)
        sol = optx.least_squares(
            _residuals,
            lm_solver,
            params,
            args=mu,
            options=options,
        )

        return sol

    def bayesian_inversion(
        self,
        t: Float[Array, "Nt"],
        mu: Float[Array, "Nt"],
        noise_std: Float,
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        Nparticles: int = 1000,
        Nsteps: int = 150,
        rng: Union[None, int, jax.Array] = None,
    ) -> BayesianSolution:
        """
        A Bayesian inversion routine using the Stein Variational Inference
        method.

        Parameters
        ----------
        t : Array
            A vector or time values (in units of seconds). The time steps
            do not need to be uniform
        mu : Array
            The observed friction curve sampled at ``t``
        noise_std : Float
            An estimate of the standard deviation of the noise in the
            measured friction curve. A conservative value is recommended,
            i.e. if the noise has a standard deviation of $10^{-3}$, a good
            starting point would be to set ``noise_std = 0.5e-3``
        y0 : Variables
            The initial values for the modelled friction and any state
            variables
        params : _Params
            The initial guess for the invertible parameters that characterise
            the forward problem. It is recommended to use the result from
            ``ODESolver.max_likelihood_inversion``. The prior distribution
            will be centered around this initial guess
        friction_constants : _Constants
            The non-invertible constants that characterise the forward
            problem
        block_constants : _BlockConstants
            The stress transfer constants (e.g. stiffness and loading rate)
        Nparticles : int
            The number of particles (= posterior samples) to include. A higher
            value gives a more accurate estimation of the posterior
            distribution, at a higher computational cost.
        Nsteps : int
            The number of iterations before convergence is expected to be
            achieved. It is recommended to start with a value of 100 and then
            see if an equilibrium was actually achieved. Increasing this value
            beyond the point of equilibrium does not do anything.
        rng: None, int, jax.random.PRNGKey
            The random seed used to initialise the particle swarm distribution.
            If ``None``, the current time will be used as a seed, which leads
            to different result for each realisation. When an integer is
            provided, a new ``jax.random.PRNGKey`` is generated.

        Returns
        -------
        result : diabayes.BayesianSolution
            The result of the inversion incapsulated in a ``BayesianSolution``
            container, which provides access to the full convergence chains,
            as well as diagnostic and visualisation routines.

        Notes
        -----
        This method is currently only compatible with standard rate-and-state
        friction models.
        """

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

        Nparams = len(params.__dict__.keys())

        scale = jnp.ones(Nparams) * 0.1
        inv_scale = 1 / (jnp.sqrt(2) * scale)
        log_params = jnp.log(params.to_array())

        # Sample particles from a log-normal distribution
        log_particles = RSFParams.generate(
            N=Nparticles, loc=log_params, scale=scale, key=split_key
        )

        # Instantiate optimiser
        opt = optax.adam(learning_rate=self.learning_rate)
        opt_state = opt.init(log_particles)  # type:ignore

        forward_fn = partial(
            self._forward_wrapper_SVI,
            y0=y0,
            t=t,
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

        @scan_tqdm(Nsteps)
        def body_fun(carry, i):
            params, state = carry
            loss, gradp = mapped_log_likelihood(params, mu, noise_std, forward_fn)
            """
            Sometimes the adjoint back-propagation becomes unstable, 
            producing NaNs in the gradients. By setting the NaN-gradients
            to zero, the particle will be attracted towards the prior.
            This is fine, because in the next step the gradients will
            likely be stable again and the particle will continue
            to be attracted by the maximum likelihood
            """
            params_array = params.to_array().T
            gradp_array = gradp.to_array().T
            nans = jnp.isnan(gradp_array)
            nan_count = nans.sum() / nans.shape[1]
            gradp_array = jnp.asarray(jnp.where(nans, 0.0, gradp_array))
            gradq_array = -2 * (params_array - log_params) * inv_scale
            phi_array = compute_phi(params_array, gradp_array, gradq_array)
            phi = type(params).from_array(phi_array.T)
            updates, state = opt.update(phi, state, params)
            params = optax.apply_updates(params, updates)
            return (params, state), (loss.mean(), nan_count, params)

        carry = (log_particles, opt_state)
        _, (loss, nan_count, states) = lax.scan(
            body_fun, carry, jnp.arange(Nsteps)  # type:ignore
        )

        return BayesianSolution(states, loss, nan_count)
