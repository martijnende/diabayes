import jax
import jax.numpy as jnp

import diabayes as db
from diabayes.forward_models import Forward, ageing_law, rsf, springblock
from diabayes.solver import ODESolver

from .aux import init_params

jax.config.update("jax_enable_x64", True)


class TestSolvers:

    def test_forward_solver(self):
        """
        Test to see whether the ODE solver runs correctly,
        and whether the steady-state result is as expected.
        """

        variables, params, constants, block_constants = init_params()
        forward = Forward(rsf, {"theta": ageing_law}, springblock)
        solver = ODESolver(forward)

        t = jnp.linspace(0.0, 100.0, 1000)
        result = solver.solve_forward(t, variables, params, constants, block_constants)
        result_jax = solver._solve_forward(
            t, variables, params, constants, block_constants
        )

        assert result is not None
        assert result_jax is not None
        assert result_jax.ys is not None

        assert jnp.allclose(result.to_array(), result_jax.ys.to_array())

        # Steady-state tests

        mu_final = result.mu[-1]
        mu_pred = constants.mu0 + (params.a - params.b) * jnp.log(
            block_constants.v_lp / constants.v0
        )
        assert jnp.allclose(mu_final, mu_pred)

        state_final = result.theta[-1]
        state_pred = params.Dc / block_constants.v_lp
        assert jnp.allclose(state_final, state_pred)

    def test_levenberg_marquardt(self):
        """
        Test for the Levenberg-Marquardt maximum-likelihood
        inversion solver. A forward simulation generates the
        friction "observation" for a given set of parameters.
        Then the true parameters are pertubed and an inversion
        is performed. The test checks whether the inverted
        parameters are identical to the initial ones, and
        whether the friction curves are identical.
        """

        variables, params, constants, block_constants = init_params()
        forward = Forward(rsf, {"theta": ageing_law}, springblock)
        forward.set_initial_values(mu=variables.mu, theta=variables.theta)

        solver = ODESolver(forward)

        y0 = forward.variables

        # Forward simulation to generate "observations"
        t = jnp.linspace(0.0, 10.0, 1000)
        result = solver.solve_forward(t, y0, params, constants, block_constants)
        mu = result.mu

        # Perturb parameters (initial guess parameters)
        params2 = db.RSFParams(a=params.a * 1.1, b=params.b * 1.2, Dc=params.Dc * 0.9)

        # Invert parameters
        result_inv = solver.max_likelihood_inversion(
            t, mu, y0, params2, constants, block_constants, verbose=True
        )
        params_inv = result_inv.value

        # Reproduce friction curve
        result2 = solver.solve_forward(t, y0, params_inv, constants, block_constants)

        assert result2 is not None

        params_array = jax.flatten_util.ravel_pytree(params)[0]  # type:ignore
        params_inv_array = jax.flatten_util.ravel_pytree(params_inv)[0]  # type:ignore

        # Check that inverted parameters and resulting friction
        # curves are identical to the original ones
        assert jnp.allclose(params_array, params_inv_array)
        assert jnp.allclose(result.mu, result2.mu)
