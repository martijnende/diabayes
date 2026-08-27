from functools import partial
from time import time_ns
from typing import Tuple, Union

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import optimistix as optx
from jax import lax
from jax_tqdm import scan_tqdm  # type: ignore
from jaxtyping import Array, Float
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from diabayes.forward_models import Forward
from diabayes.SVI import compute_phi, mapped_log_likelihood
from diabayes.typedefs import (
    BayesianSolution,
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
    """

    forward_model: Forward
    """
    An instantiated ``diabayes.Forward`` class, including
    friction law, state evolution equations, and stress
    transfer equation
    """
    rtol: float
    """The absolute tolerance used by the ODE solver"""
    atol: float
    """The absolute tolerance used by the ODE solver"""
    checkpoints: int
    """
    The number of checkpoints to use to compute the (adjoint)
    gradients through the ODE routine. A higher number increases
    stability and speed, at the expense of more GPU memory
    """
    learning_rate: float = 1e-2
    """
    The initial learning rate provided to the Adam algorithm
    for the Stein Variational Inference. The default value of
    ``1e-2`` seems like a sensible choice for most models.
    """

    def __init__(
        self,
        forward_model: Forward,
        rtol: float = 1e-7,
        atol: float = 1e-10,
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
        method: str = "RK45",
        interpolate_time: bool = True,
    ) -> Variables:
        """
        Solve a forward problem using SciPy's ``solve_ivp`` routine.
        While this routine doesn't propagate any gradients, it is
        much faster to initialise and to perform a single forward
        run. Hence for playing around with different parameters,
        it is preferred over a JITed JAX implementation.

        Parameters
        ----------
        t : Float[Array, "Nt"]
            A vector of time samples where a solution is requested.
        y0 : Variables
            The initial values (fricton and state) wrapped in a
            `Variables` container.
        params : _Params
            The (invertible) parameters that govern the dynamics,
            wrapped in a ``Params`` container.
        friction_constants : _Constants
            A container object containing the friction constants
        block_constants : _BlockConstants
            A container object containing the block constants
        method : str
            The solver used by SciPy's ``solve_ivp``.
            Default: ``RK45``
        interpolate_time : bool
            Whether to interpolate the result to the user-provided
            time samples (``True``), or to use the samples from the
            adaptive ODE solver (``False``).
            Default: ``True``

        Returns
        -------
        result : Variables
            Solution time series of friction and state
        """

        keys = y0.state.keys

        _forward = lambda t, y, *args: self.forward_model(
            t, Variables.from_array(y, keys), *args
        ).to_array()

        # If a Sundman transformation is requested, add
        # termination events based on time
        if self.forward_model.sundman is not None:
            ind_t = keys.index("t") + 1
            t_stop = t.max()
            events = lambda t, y, *args: y[ind_t] - t_stop
            events.terminal = True
            t_eval = None
            # TODO: set initial condition based on Sundman transformation!
            t_span = (t.min(), jnp.inf)
        else:
            events = None
            t_eval = t
            t_span = (t.min(), t.max())

        if not interpolate_time:
            t_eval = None

        result = solve_ivp(
            fun=_forward,
            t_span=t_span,
            y0=y0.to_array(),
            t_eval=t_eval,
            events=events,
            args=(params, friction_constants, block_constants),
            rtol=self.rtol,
            atol=self.atol,
            method=method,
        )

        assert result.y is not None

        if (self.forward_model.sundman is not None) and (interpolate_time is True):
            # Find the array index containing the integrated time
            ind_t = keys.index("t") + 1
            result_t = result.y[ind_t]
            # Instantiate interpolator
            intp = PchipInterpolator(x=result_t, y=result.y.T, axis=0)
            # Interpolate to requested time base
            result_int = intp(t)
            return Variables.from_array(result_int, keys)

        return Variables.from_array(result.y.T, keys)

    def generate_sequence(
        self,
        t_steps: Float[Array, "Nsteps"],
        v_steps: Float[Array, "Nsteps"],
        dt: Float,
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        method: str = "RK45",
    ) -> Tuple[Variables, Float[Array, "Nt"]]:
        """
        Solve the forward problem for a sequence of velocity steps.
        For each load-point velocity in `v_steps`, a forward simulation
        is run using the previous step's final state as the initial state.

        This routine can be used to generate a sequence of (up/down)
        velocity steps, or a slide-hold-slide sequence (by setting a given
        v_step to zero).

        Parameters
        ----------
        t_steps : Float[Array, "Nsteps"]
            A vector of change point times of each step in the sequence
        v_steps : Float[Array, "Nsteps"]
            A vector of the load-point velocity values for each step
            in the sequence
        dt : Float
            The desired time sample spacing of the solution
        y0 : Variables
            The initial values (fricton and state) wrapped in a
            `Variables` container.
        params : _Params
            The (invertible) parameters that govern the dynamics,
            wrapped in a `Params` container.
        friction_constants : _Constants
            A container object containing the friction constants
        block_constants : _BlockConstants
            A container object containing the block constants. Note
            that the load-point velocity will be updated for each step

        Returns
        -------
        result : Variables
            Solution time series of friction and state
        t : Float[Array, "Nt"]
            Solution time samples
        """

        t_start = 0.0

        # Loop over velocity-steps
        for i, (t_stop, v) in enumerate(zip(t_steps, v_steps)):
            # Update load-point velocity
            block_dict = block_constants.__dict__
            block_dict["v_lp"] = float(v)
            block_constants = type(block_constants)(**block_dict)
            # Define time vector
            t_i = jnp.arange(t_start, t_stop, dt)
            # Solve forward problem
            result_i = self.solve_forward(
                t_i, y0, params, friction_constants, block_constants, method
            )
            # Append results
            if i == 0:
                result = result_i
                t = t_i
            else:
                result = result.append(result_i)
                t = jnp.concatenate([t, t_i])

            # Increment start time
            t_start = t_stop

            # Update initial state
            y0 = result_i[-1]

        return result, t

    def generate_sequence(
        self,
        t_steps: Float[Array, "Nsteps"],
        v_steps: Float[Array, "Nsteps"],
        dt: Float,
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        method: str = "RK45",
    ) -> Tuple[Variables, Float[Array, "Nt"]]:
        """
        Solve the forward problem for a sequence of velocity steps.
        For each load-point velocity in `v_steps`, a forward simulation
        is run using the previous step's final state as the initial state.

        This routine can be used to generate a sequence of (up/down)
        velocity steps, or a slide-hold-slide sequence (by setting a given
        v_step to zero).

        Parameters
        ----------
        t_steps : Float[Array, "Nsteps"]
            A vector of change point times of each step in the sequence
        v_steps : Float[Array, "Nsteps"]
            A vector of the load-point velocity values for each step
            in the sequence
        dt : Float
            The desired time sample spacing of the solution
        y0 : Variables
            The initial values (fricton and state) wrapped in a
            `Variables` container.
        params : _Params
            The (invertible) parameters that govern the dynamics,
            wrapped in a `Params` container.
        friction_constants : _Constants
            A container object containing the friction constants
        block_constants : _BlockConstants
            A container object containing the block constants. Note
            that the load-point velocity will be updated for each step

        Returns
        -------
        result : Variables
            Solution time series of friction and state
        t : Float[Array, "Nt"]
            Solution time samples
        """

        t_start = 0.0

        # Loop over velocity-steps
        for i, (t_stop, v) in enumerate(zip(t_steps, v_steps)):
            # Update load-point velocity
            block_dict = block_constants.__dict__
            block_dict["v_lp"] = float(v)
            block_constants = type(block_constants)(**block_dict)
            # Define time vector
            t_i = jnp.arange(t_start, t_stop, dt)
            # Solve forward problem
            result_i = self.solve_forward(
                t_i, y0, params, friction_constants, block_constants, method
            )
            # Append results
            if i == 0:
                result = result_i
                t = t_i
            else:
                result = result.append(result_i)
                t = jnp.concatenate([t, t_i])

            # Increment start time
            t_start = t_stop

            # Update initial state
            y0 = result_i[-1]

        return result, t

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
    def evaluate_at_t(self, sol: dfx.Solution, t_eval: Float[Array, "Nt"]) -> Variables:

        ts = sol.interpolation.ts  # type: ignore
        tmax = jnp.nanmax(jnp.where(jnp.isfinite(ts), ts, jnp.nan))

        def root_fn(r, target_t):
            state = sol.evaluate(r)
            return state.t - target_t

        @jax.vmap
        def get_state(target_t):
            solver = optx.Bisection(rtol=1e-5, atol=1e-5)
            options = {"lower": sol.t0, "upper": tmax}
            r_guess = jnp.array(0.5 * (sol.t0 + tmax))
            root = optx.root_find(
                fn=root_fn, solver=solver, y0=r_guess, args=target_t, options=options
            )
            return sol.evaluate(root.value)

        return get_state(t_eval + 1e-12)

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
        args = (params, friction_constants, block_constants)
        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)
        if adjoint is None:
            adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=self.checkpoints)
        assert isinstance(adjoint, dfx.AbstractAdjoint)

        if self.forward_model.sundman is not None:
            saveat = dfx.SaveAt(dense=True)
            t0 = 0
            t1 = jnp.inf
            tmax = t.max()
            root_finder = optx.Bisection(rtol=self.rtol, atol=self.atol)
            cond_fn = lambda t, y, *args, **kwargs: y.t - tmax
            event = dfx.Event(cond_fn, root_finder)
            kwargs = {
                "saveat": saveat,
                "t0": t0,
                "t1": t1,
                "dt0": 1e-6,
                "event": event,
                "max_steps": int(1e5),
            }
        else:
            t0 = t.min()
            t1 = t.max()
            dt0 = t[1] - t[0]
            saveat = dfx.SaveAt(ts=t)
            kwargs = {"t0": t0, "t1": t1, "dt0": dt0, "saveat": saveat}

        sol = dfx.diffeqsolve(
            terms=term,
            solver=dfx.Tsit5(),
            y0=y0,
            args=args,
            stepsize_controller=controller,
            adjoint=adjoint,
            throw=False,  # Essential for Bayesian (batch) optimisation
            **kwargs,
        )

        assert sol is not None

        if self.forward_model.sundman is not None:
            ys_at_t = self.evaluate_at_t(sol, t)
            is_leaf = lambda x: x is None
            sol = eqx.tree_at(lambda s: s.ys, sol, ys_at_t, is_leaf=is_leaf)
            sol = eqx.tree_at(lambda s: s.ts, sol, ys_at_t.t, is_leaf=is_leaf)

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
        mu_hat = jnp.squeeze(result.ys.mu)  # type: ignore
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
        retries: int = 3,
        seed: int = 42,
    ) -> optx.Solution | None:
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
        retries : int
            The maximum number of inversion attempts. When the inversion fails
            to converge, it will retry up to ``retries`` times with randomly
            perturbed initial parameters.
        seed : int
            Seed for the random number generator. This is only used when the
            initial inversion attempt fails, and the initial parameters are
            randomly perturbed before the next attempt.

        Returns
        -------
        sol : optimistix.Solution
            The inversion result, including various diagnostics. The
            inverted parameter values can be accessed as ``sol.values``
        """

        options = {"autodiff_mode": "fwd"}

        _residuals = lambda params, mu: self._residuals(
            params, t, mu, y0, friction_constants, block_constants
        )

        lm_solver = optx.LevenbergMarquardt(rtol=1e-5, atol=1e-5, verbose=verbose)

        key = jr.PRNGKey(seed)

        sol = None

        for i in range(retries):

            try:
                sol = optx.least_squares(
                    _residuals,
                    lm_solver,
                    params,
                    args=mu,
                    options=options,
                )
            except eqx.EquinoxRuntimeError:
                print(
                    f"[Attempt {(i+1)}/{retries}] Inversion failed. Retrying with randomised initial parameters..."
                )
                key, split_key = jr.split(key)
                param_vals = params.to_array()
                jitter = jr.normal(split_key, shape=param_vals.shape)
                scale = (
                    0.1 * param_vals
                )  # Standard deviation equal to 10% of current value
                param_vals = param_vals + scale * jitter
                # Recreate params and retry
                params = type(params).from_array(param_vals)

        if sol is None:
            print(
                f"Inversion failed {retries} attempts. Increase the value of `retries` or check the stability of the forward model"
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

        # assert isinstance(
        #     params, RSFParams
        # ), "Bayesian inversion is only implemented for RSF"

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
        log_particles = type(params).generate(
            N=Nparticles, loc=log_params, scale=scale, key=split_key
        )

        # Instantiate optimiser
        opt = optax.adam(learning_rate=self.learning_rate)
        opt_state = opt.init(log_particles)  # type: ignore

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
            body_fun, carry, jnp.arange(Nsteps)  # type: ignore
        )

        return BayesianSolution(states, loss, nan_count)
